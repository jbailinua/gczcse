# GCZCSE:
# Clumpy globular cluster self-enrichment.
# Jeremy Bailin (jbailin@ua.edu)

# Note unit system:
#   Masses always in solar masses
#   Radii always in pc
#   Energy always in erg
#   Time always in yr
# Don't use astropy.Quantities for them because it slows things down,
# but use them for conversion when necessary (e.g. when using G).

import numpy as np
from astropy import units, constants
import imf
from scipy.interpolate import interp1d

Zsun = 0.016

class GCException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class cloud(object):
    """A protocluster cloud, which will contain one or more clumps within it.

    Initialize the class with the global parameter dict and a mass. Then
    fragment() it into clumps, starformation() it, and plot the results."""

    def __init__(self, params, cloudmass, cloudradius, preZ=0.):
        """Initializes the cloud using params (a dict containing the global
        parameters), the mass of the cloud in solar masses, the radius of the
        cloud in parsecs, and the pre-enrichment metal fraction (0.016 is
        solar)."""
        self.params = params
        self.initialcloudmass = cloudmass
        self.currentcloudmass = cloudmass
        self.radius = cloudradius
        self.initial_Zfrac = preZ
        self.initial_metalmass = self.initial_Zfrac * self.initialcloudmass
        self.nclump = False
        self.IMF = imf.IMF(self.params['imf_type'], self.params)

    def stellarmass(self):
        """Returns the total stellar mass of the cloud, i.e. the sum of the
        stellar masses of its clumps."""
        return np.sum([x.stellarmass for x in self.clumps])
        
    def stellarmass_longlived(self):
        """Returns the total stellar mass of the GC after 10 Gyr have passed."""
        return np.sum([x.stellarmass_after_time(10e9)[1] for x in self.clumps])

    def fragment(self):
        """Takes an initial cloud and breaks it up into subclumps with specific
        masses and formation times."""
        if self.nclump:
            raise GCException("cloud.fragment() called with pre-fragmented cloud.")

        # calculate the expected number of clumps for this cloud mass
        # Nclump = Nclump,ref (M_GMC / M_GMC,ref)^s
        Nclump_expected = self.params['Nclumpref'] * (self.initialcloudmass / self.params['MGMCref'])**self.params['s']
        if self.params['Nclump_poisson']:
            # Actual number of clumps is a Poisson random variable with this
            # expectation
            self.nclump = np.random.poisson(Nclump_expected)
        else:
            # round to nearest integer
            self.nclump = np.rint(Nclump_expected)
        # minimum of one clump
        if self.nclump==0:
            self.nclump=1

        # draw the clump masses from a log-normal distribution with natural log
        # dispersion given by Mclump_dispersion. For small values of
        # Mclump_dispersion, this is equal to the fractional dispersion.
        Mclump_mean = self.currentcloudmass / self.nclump * self.params['clumpy_massfrac']
        Mclumps = np.random.lognormal(mean=np.log(Mclump_mean), sigma=self.params['Mclump_dispersion'],
                size=self.nclump)

        # note that, because of this random sampling, the sum of the clump
        # masses is unlikely to actually match the expected amount of mass
	# in clumps. Therefore, renormalize clump masses to fix that.
        mass_in_clumps = np.sum(Mclumps)
        Mclumps *= self.initialcloudmass * self.params['clumpy_massfrac'] / mass_in_clumps

        # formation time comes from a normal distribution of width
        # tform_dispersion in years.
        # Sort them so that when we run through the list of clumps, they are in
        # order of formation.
        tforms = sorted( np.random.normal(scale=float(self.params['tform_dispersion']), size=self.nclump) )

        # make the clumps
        self.clumps = [clump(self, x_mass, self.initial_Zfrac, x_tform)
                for (x_mass, x_tform) in zip(Mclumps, tforms)]


    def starformation(self):
        """Workhorse routine that forms stars in each clump and calculates all
        of the self- and cross-enrichment."""
        # loop through clumps, so they can each form stars and enrich
        for clumpi in xrange(len(self.clumps)):
            clump = self.clumps[clumpi]

            # first determine cross-enrichment from all previous clumps
            if self.params['cross_enrichment']:
                cross_metalmass = 0.
                cross_masstot = 0.
                cross_ESN = 0.
                for preclumpi in xrange(clumpi):
                    preclump = self.clumps[preclumpi]
                    # how old is the preclump?
                    delta_t = clump.tform - preclump.tform
                    # calculate the metals and energy produced after that length
                    # of time
                    metals_ejected, mass_ejected, energy_ejected, metals_retained = preclump.metals_and_energy_ejected(delta_t)
                    # add to the running total of cross-enrichment metal mass
                    # and total energy
                    cross_metalmass += metals_ejected
                    cross_masstot += mass_ejected
                    cross_ESN += energy_ejected


                if cross_masstot > 0:
                	cross_Zfrac = cross_metalmass / cross_masstot
                else:
                	cross_Zfrac = 0.
                # how much of it is retained in the parent cloud? See metals_and_energy_ejected() for description
                retained_fraction = np.exp(-float(cross_ESN*units.erg * self.radius*units.pc /
                        (constants.G * (self.currentcloudmass*constants.M_sun)**2)))
                ejected_fraction = 1. - retained_fraction
                ejected_metalmass_maximalmixing = cross_metalmass * ejected_fraction
                ejected_metalmass_minimalmixing = np.min([cross_metalmass,
                    self.currentcloudmass * ejected_fraction * cross_Zfrac])
                ejected_metalmass = np.interp(self.params['mixing_efficiency'], [0.,1.],
                    [ejected_metalmass_minimalmixing, ejected_metalmass_maximalmixing])
                retained_metalmass = cross_metalmass - ejected_metalmass
                # the retained cross-enrichment adds to what's already there to give current metalicity
                current_metallicity = (self.initial_metalmass + retained_metalmass) / self.currentcloudmass
                # store the amount of metals the clump received from
                # cross-enrichment, as a fraction of its total mass
                clump.CEmetalfrac = retained_metalmass / clump.totalmass

                # update the current clump to the correct metallicity
                clump.update_metallicity(current_metallicity)

            # form stellar pop
            clump.starform()
            # self-enrich
            clump.selfenrich()

    def clump_Z_mass(self):
        """Returns a tuple containing the metallicity and stellar mass of each clump."""
        metallicities = np.array([x.ssp.metallicity for x in self.clumps])
        clumpmasses = np.array([x.stellarmass for x in self.clumps])
        return (metallicities, clumpmasses)

    def clump_Z_mass_N_longlived(self):
        """Returns a tuple containing the metallicity and stellar mass extant after 10 Gyr of each clump."""
        metallicities = np.array([x.ssp.metallicity for x in self.clumps])
        clumpmasses = np.array([x.stellarmass_after_time(10e9)[1] for x in self.clumps])
        clumpNstars = np.array([x.numstars_after_time(10e9) for x in self.clumps])
        return (metallicities, clumpmasses, clumpNstars)


    def metallicities(self):
        """Returns a tuple containing the mean metallicity of the stars and their dispersion.
        Clumps are weighted by their stellar mass."""
        metallicities, clumpmasses = self.clump_Z_mass()
        metallicity_mean = np.average(metallicities, weights=clumpmasses)
        metallicity_var = np.average((metallicities-metallicity_mean)**2,
                weights=clumpmasses)
        metallicity_std = np.sqrt(metallicity_var)
        return (metallicity_mean, metallicity_std)

    def NSN(self):
        """If enrichment has occurred, returns the total number of supernovae
        in all clumps, otherwise returns 0."""
        if self.clumps[0].SEmetalfrac:
        	# star formation has happened
        	return np.sum([x.ssp.NSN for x in self.clumps], dtype=int)
        else:
        	return 0



class clump(object):
    """A clump within a protocluster cloud, which will turn into an SSP.
    Standard usage is to initialize it, then starform() it to create the star
    properties and selfenrich() it to update those stars' metallicities. The
    amount of metals and energy available outside the clump for cross-enrichment
    is found using metals_and_energy_ejected()."""

    def __init__(self, cloud, clumpmass, preZ=0., tform=0.):
        """Initializes a clump that belongs to the parent cloud (a cloud()
        object) with mass clumpmass in solar masses, pre-enrichment metal
        fraction preZ (0.016 is solar), and a formation time tform in years."""
        self.parentcloud = cloud
        self.params = cloud.params
        self.totalmass = clumpmass
        self.gasmass = clumpmass
        self.stellarmass = 0.
        self.metalmass = preZ * clumpmass
        self.SEmetalfrac = False
        self.CEmetalfrac = False
        self.tform = tform
        self.assign_radius()

    def assign_radius(self):
        if self.params['clumprad_type']=='constant_density':
            # Bertoldi & McKee: mean density within clumps are pretty consistent from
            # clump to clump, implying that r is proportional to M^1/3
            # From their Figure 4, the mean n_H of the self-gravitating clumps is 10^3.5
            # So M = (4/3) pi r^3 n_H mu m_H
            # But if I put these numbers in, I get radii that are much higher than
            # what I see in Bonnell or Bate.
            # Those appear to have radii ~0.1pc for masses of 200 Msun, or a mean mass
            # density of 5e4 Msun/pc^3, or n_H=1.4e6.
            self.clumprad = (self.totalmass / self.params['clumprad_density']) ** (-1./3)
        elif self.params['clumprad_type']=='cloud_fraction':
            # But that can't make sense either, because then you reach a radius of 1pc
            # for a clump mass of 5e4 (perfectly reasonable for a 3e5 solar mass cloud)
            # which is as large as the whole cloud!
            # Perhaps a better option is to use a constant fraction of the total cloud
            # radius.
            self.clumprad = self.params['clumprad_fraction'] * self.parentcloud.radius
        elif self.params['clumprad_type']=='constant_surface_density':
            # Larson's relations imply mass essentially proportional to radius squared, i.e.
            # constant surface density. See Ballesteros-Paredes et al. (2012) and refs therein.
            # From Ellsworth-Bowers et al. (2015), the typical surface density is about
            # 100 Msun/pc^2
            self.clumprad = np.sqrt(self.totalmass / (np.pi * self.params['clumprad_surface_density']))
        else:
           raise GCException("Unknown clumprad_type.")

    def metalfrac(self):
        """Returns the current metal fraction of the clump."""
        return self.metalmass / self.totalmass

    def update_metallicity(self, Zfrac):
        """Update the clump to have a total metal fraction of Zfrac."""
        self.metalmass = Zfrac * self.gasmass

    def starform(self):
        """Turn gas mass in the clump into stellar mass, according to the global
        parameters of the parent cloud. Star properties are stored in clump.ssp."""
        # precompute the metallicity of this event
        metallicity = self.metalfrac()
        # determine the number of stars that will give the appropriate amount of
        # total stellar mass (on average) and sample them from the IMF.
        # Note that the total star formation efficiency across the entire cloud
		# is fstar, but that only clumpy_massfrac of the cloud mass is available,
		# so the efficiency within a clump is the higher value clump_fstar
        clump_fstar = self.params['fstar'] / self.params['clumpy_massfrac']
        Ntosample = int(round(clump_fstar * self.gasmass * self.parentcloud.IMF.N_per_solar_mass))
        starmasses = self.parentcloud.IMF.sample(Ntosample)
        # create the star objects for each
        self.ssp = SSP(self, starmasses, self.tform, metallicity)
        # calculate the actual stellar mass of that population, and remove that
        # mass from the gas
        self.stellarmass = np.sum(starmasses)
        self.gasmass -= self.stellarmass

    def selfenrich(self):
        """Update the metallicities of the stars in the clump according to the
        self-enrichment scheme from high mass stars within this clump."""

        # Use metals_and_energy_ejected at a time after all stars that can die
        # are gone to determine what's available for self-enrichment
        ejected_metalmass, ejected_masstot, total_ESN, retained_metalmass = self.metals_and_energy_ejected( np.max(self.ssp.stars['lifetime']) )
        
        # increase the cloud's metal mass by the retained metal mass and change
        # the metallicities of the stars to the new value
        self.metalmass += retained_metalmass
        self.ssp.metallicity = self.metalfrac()
        # store the amount of metals received via self-enrichment as a function
        # of the total clump mass
        self.SEmetalfrac = retained_metalmass / self.totalmass

    def metals_and_energy_ejected(self, time):
        """Returns a tuple cotaining the total mass of metals that are ejected
        from this clump (in solar masses) and the total energy of the supernovae
        that formed those metals (in ergs), and are therefore available to the
        other clumps within the cloud."""
        # the metals lost by the SSP formed in this clump to the extra-clump
        # cloud after a given point in time and the energy of the supernovae
        # that ejected them there.
        #
        # Follows the same calculation as for self-enrichment, but only using
        # the supernova energy ejected to that point in time and the metals
        # produced to that point in time.
        #
        # Find the star whose lifetime is just shorter than the current time, so
        # we can use the cumulative properties of the population to that point.
        star_that_just_went_boom = np.searchsorted(self.ssp.stars['lifetime'], time)
        # Get the total SN energy and metals produced to that point
        if star_that_just_went_boom == 0:
        	# No supernovae, so all quantities are 0
        	ejected_metalmass = 0.
        	ejected_masstot = 0.
        	total_ESN = 0.
        	retained_metalmass = 0.
        else:
        	total_ESN = self.ssp.stars['cumulative_ESN'][star_that_just_went_boom]
        	total_Zproduced = self.ssp.stars['cumulative_Zproduced'][star_that_just_went_boom]
        	total_massproduced = self.ssp.stars['cumulative_massproduced'][star_that_just_went_boom]
        	if total_massproduced > 0:
        		total_Zfrac = total_Zproduced / total_massproduced
        	else:
        		total_Zfrac = 0.
        	# Calculate the retained fraction in a logarithmic potential
        	# use float(...) to get rid of slow Quantity
        	retained_fraction = np.exp(-float(total_ESN*units.erg * self.clumprad*units.pc /
        	        (constants.G * (self.totalmass*constants.M_sun)**2)))
        	ejected_fraction = 1. - retained_fraction
        	# mix the ejecta.
        	ejected_metalmass_maximalmixing = total_Zproduced * ejected_fraction
        	ejected_metalmass_minimalmixing = np.min([total_Zproduced,
        	    self.totalmass * ejected_fraction * total_Zfrac])
        	ejected_metalmass = np.interp(self.params['mixing_efficiency'], [0., 1.],
        	    [ejected_metalmass_minimalmixing, ejected_metalmass_maximalmixing])
        	retained_metalmass = total_Zproduced - ejected_metalmass
        	ejected_masstot = ejected_fraction * self.totalmass
                
        return (ejected_metalmass, ejected_masstot, total_ESN, retained_metalmass,)
        
    def stellarmass_after_time(self, age):
        """Returns the total mass in stars that live for at least age years,
        i.e. the stellar mass you would see that long after formation.
        Returns the mass in MS stars and the total mass including remnants.
        Remnants are assumed to be 2.0 Msun for everything that goes SN, and
        0.56 Msun for anything smaller."""
        most_massive_star_remaining = np.searchsorted(self.ssp.stars['lifetime'], age)
        # all stars from here to the end still exist
        mainsequence_mstar = np.sum(self.ssp.stars['mass'][most_massive_star_remaining:])
        supernova_remnant_mass = 2.0
        whitedwarf_mass = 0.56
        remnant_masses = self.ssp.stars['SNp']*supernova_remnant_mass + (~self.ssp.stars['SNp'])*whitedwarf_mass
        remnant_mstar = np.sum(remnant_masses[:most_massive_star_remaining])
        return (mainsequence_mstar, mainsequence_mstar+remnant_mstar)
        
    def numstars_after_time(self, age):
    	"""Returns the total number of stars that live for at least age years,
    	i.e. the number of stars you would see after that long after formation."""
    	most_massive_star_remaining = np.searchsorted(self.ssp.stars['lifetime'], age)
    	# all stars from here to the end still exist
    	return len(self.ssp.stars['mass'][most_massive_star_remaining:])
        


class SSP(object):
    """A simple stellar population, consisting of a list of stars that have the
    same formation time and metallicity. Usually just initialized and then its
    properties (especially SSP.stars) are examined."""
    
    def __init__(self, clump, masses, tform, metallicity):
        """Create the SSP from the parent clump. masses contains a list of the
        masses of the stars in the SSP (in solar masses), tform is their
        formation time (a scalar), and metallicity is their metal fraction
        (0.016 is solar).

        After initialization, SSP.stars is a structured numpy array of all stars
        in the SSP, with the following useful columns:
            'mass': mass of the star in solar masses
            'lifetime': lifetime of the star in years
            'SNp': does this star explode?
            'Zproduced': what mass of metals (in solar masses) are produced from
                         any supernova of this star
            'cumulative_Zproduced': cumulative sum of Zproduced
            'massproduced':  what is the total ejected mass (in solar masses) produced
                            from any supernova of this star
            'cumulative_massproduced':  cumulative sum of massproduced
            'cumulative_ESN': cumulative amount of supernova energy produced
        Note that SSP.stars is sorted in order of decreasing stellar mass,
        regardless of the ordering of the "masses" input parameter."""

        self.parentclump = clump
        self.params = clump.params
        self.tform = tform
        self.metallicity = metallicity
        self.nstars = len(masses)

        self.stars = np.zeros(self.nstars, dtype=[('mass','f4'), ('SNp','b1'),
            ('Zproduced','f4'), ('massproduced','f4'), ('lifetime','f4'), ('cumulative_ESN','f8'),
            ('cumulative_Zproduced','f4'), ('cumulative_massproduced','f4')])
        # convenient to sort stars inversely by mass, i.e. increasing in
        # lifetime
        self.stars['mass'] = sorted(masses, reverse=True)

        # fill remaining columns with derived quantities
        # does this star go supernova, and what is the total energy ejected by
        # supernovae up to this point
        self.stars['SNp'] = (self.stars['mass'] > 8.)
        self.stars['cumulative_ESN'] = self.params['ESN'] * np.cumsum(self.stars['SNp'])
        # metals produced by each star, mass produced by each star,
        # and totals produced by the population to that point.
        self.stars['Zproduced'] = mass_to.Z(self.stars['mass'], self.params['ccsn_yield']) * self.stars['SNp']
        self.stars['massproduced'] = mass_to.masstot(self.stars['mass'], self.params['ccsn_yield']) * self.stars['SNp']
        self.stars['cumulative_Zproduced'] = np.cumsum(self.stars['Zproduced'])
        self.stars['cumulative_massproduced'] = np.cumsum(self.stars['massproduced'])
        # lifetime of star
        self.stars['lifetime'] = mass_to.lifetime(self.stars['mass'], self.params['stellar_lifetimes'])

        # how many total supernovae are there?
        self.NSN = np.sum(self.stars['SNp'])


class mass_to(object):
	"""Convert stellar mass in solar masses to properties of stars (e.g. lifetimes,
	metal production)."""
	# Preload MIST data for lifetimes and create interpolation function. This only runs
	# when the module is first imported, and then we can use the class methods to do the
	# calculation without instantiating the class, as in mass_to.lifetime(mass, 'mist').
	_lifetime_fname = 'stellar-lifetimes/MIST-lifetimes.txt'
	_lifedat = np.loadtxt(_lifetime_fname)
	_lifefunc = interp1d(_lifedat[:,0], _lifedat[:,1])
	# Preload yields from nuGrid and West & Heger simulations and create
	# interpolation function
	_yield_nugrid = np.loadtxt('yields/yielddata-nuGrid.dat')
	# Add an extra 100 Msun point that has the same fractional yield as the highest-mass model
	topmass = 100.0
	nugrid_topmass_frac = _yield_nugrid[-1,1] / _yield_nugrid[-1,0]
	nugrid_topmass_yield = topmass * nugrid_topmass_frac
	new_nugrid_line = np.array([topmass, nugrid_topmass_yield, topmass-nugrid_topmass_yield])[np.newaxis,:]
	_yield_nugrid = np.concatenate( (_yield_nugrid, new_nugrid_line), axis=0)
	_yield_nugrid_interpfunc = interp1d(_yield_nugrid[:,0], _yield_nugrid[:,1], \
	    bounds_error=False, fill_value=nugrid_topmass_yield)
	_yield_nugrid_massinterpfunc = interp1d(_yield_nugrid[:,0], _yield_nugrid[:,2], \
	    bounds_error=False, fill_value=topmass-nugrid_topmass_yield)
	_nugrid_minmass = np.min(_yield_nugrid[:,0])
	_yield_westheger = np.loadtxt('yields/yielddata-westheger.dat')
	_yield_westheger_interpfunc = interp1d(_yield_westheger[:,0], _yield_westheger[:,1], \
	    bounds_error=False, fill_value=0.0)
	_yield_westheger_massinterpfunc = interp1d(_yield_westheger[:,0], _yield_westheger[:,2], \
	    bounds_error=False, fill_value=0.0)
	_westheger_minmass = np.min(_yield_westheger[:,0])
	_yield_nomoto = np.loadtxt('yields/yielddata-nomoto13.dat')
	# Add an extra 100 Msun point that has the same fractional yield as the highest-mass model
	topmass = 100.0
	nomoto_topmass_frac = _yield_nomoto[-1,1] / _yield_nomoto[-1,0]
	nomoto_topmass_yield = topmass * nomoto_topmass_frac
	new_nomoto_line = np.array([topmass, nomoto_topmass_yield, topmass-nomoto_topmass_yield])[np.newaxis,:]
	_yield_nomoto = np.concatenate( (_yield_nomoto, new_nomoto_line), axis=0)
	_yield_nomoto_interpfunc = interp1d(_yield_nomoto[:,0], _yield_nomoto[:,1], \
	    bounds_error=False, fill_value=nomoto_topmass_yield)
	_yield_nomoto_massinterpfunc = interp1d(_yield_nomoto[:,0], _yield_nomoto[:,2], \
	    bounds_error=False, fill_value=topmass-nomoto_topmass_yield)
	_nomoto_minmass = np.min(_yield_nomoto[:,0])

	@classmethod
	def lifetime(cls, mass, lifetimeparam):
		"""Convert stellar mass in solar masses to lifetime in years."""
		if lifetimeparam=='analytic':
			return 1e10 * mass**(-2.5)   # in years, with mass in solar masses
		elif lifetimeparam=='mist':   # Choi et al. (2016) [Fe/H]=-1.75
			return 10.0**cls._lifefunc(mass)
		elif lifetimeparam=='polyfit':    # polynomial fit to Choi et al.
			logmass = np.log10(mass)
			return 10.0**(-0.086 * logmass**3 + 0.95 * logmass**2 - 3.17 * logmass + 9.77)
		else:
			raise GCException("Unknown type of stellar lifetime.")

	@classmethod
	def bh09_adopted(cls, mass):
		"""Convert stellar mass in solar masses to metal production in solar masses for BH09 parametrization."""
		snp = (mass > 8.)
		Bconst = 0.0118
		Cconst = 0.00548
		return snp * (Bconst + Cconst*mass)*mass
        
	@classmethod
	def bh09_adopted_masstot(cls, mass):
		"""Convert stellar mass in solar masses to total ejected mass in solar masses for BH09 parametrization."""
		# Assumes 2.0 Msun remnant
		snp = (mass > 8.)
		return snp * (mass - 2.0)
        
	@classmethod
	def nugrid_adopted(cls, mass):
		"""Convert stellar mass in solar masses to metal production in solar masses for nuGrid simulations."""
		return (mass < cls._nugrid_minmass)*cls.bh09_adopted(mass) + \
			(mass >= cls._nugrid_minmass)*cls._yield_nugrid_interpfunc(mass)
            
	@classmethod
	def nugrid_adopted_masstot(cls, mass):
		"""Convert stellar mass in solar masses to total ejected mass in solar masses for nuGrid simulations."""
		return (mass < cls._nugrid_minmass)*cls.bh09_adopted_masstot(mass) + \
			(mass >= cls._nugrid_minmass)*cls._yield_nugrid_massinterpfunc(mass)
            
	@classmethod
	def westheger_adopted(cls, mass):
		"""Convert stellar mass in solar masses to metal production in solar masses for West & Heger simulations."""
		return (mass < cls._westheger_minmass)*cls.bh09_adopted(mass) + \
			(mass >= cls._westheger_minmass)*cls._yield_westheger_interpfunc(mass)

	@classmethod
	def westheger_adopted_masstot(cls, mass):
		"""Convert stellar mass in solar masses to total ejected mass in solar masses for westheger simulations."""
		return (mass < cls._westheger_minmass)*cls.bh09_adopted_masstot(mass) + \
			(mass >= cls._westheger_minmass)*cls._yield_westheger_massinterpfunc(mass)
            
	@classmethod
	def nomoto_adopted(cls, mass):
		"""Convert stellar mass in solar masses to metal production in solar masses for Nomoto13 table."""
		return (mass < cls._nomoto_minmass)*cls.bh09_adopted(mass) + \
			(mass >= cls._nomoto_minmass)*cls._yield_nomoto_interpfunc(mass)
    
	@classmethod
	def nomoto_adopted_masstot(cls, mass):
		"""Convert stellar mass in solar masses to total ejected mass in solar masses for nomoto simulations."""
		return (mass < cls._nomoto_minmass)*cls.bh09_adopted_masstot(mass) + \
			(mass >= cls._nomoto_minmass)*cls._yield_nomoto_massinterpfunc(mass)
            
	@classmethod
	def Z(cls, mass, yieldparam):
	    """Convert stellar mass in solar masses to metal production mass in solar masses."""
	    if yieldparam=='bh09':
	        return cls.bh09_adopted(mass)
	    elif yieldparam=='nugrid':
	        return cls.nugrid_adopted(mass)
	    elif yieldparam=='westheger':
	        return cls.westheger_adopted(mass)
	    elif yieldparam=='nomoto13':
	    	return cls.nomoto_adopted(mass)
	    else:
	    	raise GCException("Unknown yield type.")

	@classmethod
	def masstot(cls, mass, yieldparam):
	    """Convert stellar mass in solar masses to total ejected mass mass in solar masses."""
	    if yieldparam=='bh09':
	        return cls.bh09_adopted_masstot(mass)
	    elif yieldparam=='nugrid':
	        return cls.nugrid_adopted_masstot(mass)
	    elif yieldparam=='westheger':
	        return cls.westheger_adopted_masstot(mass)
	    elif yieldparam=='nomoto13':
	    	return cls.nomoto_adopted_masstot(mass)
	    else:
	    	raise GCException("Unknown yield type.")


def ZZsun(Z):
    """log Z / Zsun"""
    return np.log10(Z/Zsun)

def mean_std_sun(metal, mass):
    """Given an array of metal fractions and masses, computes the weighted
    mean and dispersion of log Z/Zsun."""
    metalsun = ZZsun(metal)
    metalmean = np.average(metalsun, weights=mass)
    metalstd = np.sqrt( np.average((metalsun-metalmean)**2, weights=mass) )
    return (metalmean, metalstd)
    
