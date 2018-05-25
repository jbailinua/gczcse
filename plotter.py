import yaml
import gczcse as gc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.ticker import FuncFormatter
import astropy.units as units
import astropy.constants as const
import cPickle as pickle
import multiprocessing
import sys
from scipy.interpolate import interp1d


# Create plots?
plot_flag = True
# obj_flag specifies whether the metallicity of every single star is stored, or
# the full object containing the whole cluster.
# NOTE!!! multiprocessing.Pool.map has problems with functions that return
# very large objects. So if obj_flag is set to True, parallel will automatically
# be switched to False regardless of what it says here.
obj_flag = False
parallel = True
# Note that the histogram plots need the objects, so if obj_flag=False then
# there will be no histograms, regardless of the value of histplot_flag.
histplot_flag = False
histplot_highlight_flag = False  # highlight the indices on the other plots
histplot_upsidedown = True
object_indices = [26, 39, 46, 52, 69]
histplot_ind = np.array(object_indices)
pkl_dir = False    # Set to False for no output pickle, or directory for output pickle file.

# Range of log(M_cloud) to model. They get very slow
# above 10^7 Msun, so it is recommended to stop there when testing, then use the
# full range for publication plots.
logmass = np.arange(4.5, 8, 0.05)


# Plot defaults
plt.rc('legend', numpoints=1)
plt.rc('font', family='sans-serif')
plt.rc('font', size=12.0)
plt.rc('axes', labelsize=24.0)
plt.rc('axes', linewidth=1.5)
plt.rc('xtick.major', width=1.5)
plt.rc('xtick.minor', width=1.5)
plt.rc('ytick.major', width=1.5)
plt.rc('ytick.minor', width=1.5)
plt.rc('lines', linewidth=2.0)


# Radius of cloud, in pc, for BH09 comparison
rt = 1.0

np.random.seed(43)   # for reproducibility

# Create LaTeX string for scientific notation.
# http://stackoverflow.com/questions/31453422/displaying-numbers-with-x-instead-of-e-scientific-notation-in-matplotlib
def sci_not(x, ndecimal):
	if x==0:
		return '0'
	else:
		s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndecimal)
		m, e = s.split('e')
		return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))

# Express as 10^x. Uses sci_not as a guideline.
def logsci_not(x, ndecimal):
	logval = np.log10(x)
	s = '{x:0.{ndp:d}e}'.format(x=logval, ndp=ndecimal)
	m, e = s.split('e')
	return r'10^{{{m:s}}}'.format(m=m)

# Calculate the low and high error in log10 of a value.
def logerr(value, err):
	logval = np.log10(value)
	return [logval-np.log10(value-err), np.log10(value+err)-logval]

# Convert Z to [Fe/H] for typical GC alpha enhancement.
def FeH(logZZs):
    return logZZs - 0.25

# Workhorse routine that creates a cloud, fragments it, forms stars, and
# gathers up the metallicity information of the stars.
def evaluate_cloud(cloudmass, return_object=False):
    print 'Mc=%e' % cloudmass
    # Create cloud object
    gmc = gc.cloud(cfg, cloudmass, rt, preZ=gc.Zsun*(10.0**(cfg['pre_enrich_level'])))
    # Fragment into clumps
    gmc.fragment()
    # Form stars and do the self- and cross-enrichment
    gmc.starformation()
    # Calculate mean metallicity of the resulting population and its standard deviation
    metal_meanstd = gc.mean_std_sun(*gmc.clump_Z_mass())
    if return_object:
        return (gmc, metal_meanstd[0], metal_meanstd[1], gmc.stellarmass_longlived(), gmc.stellarmass())
    else:
        return (metal_meanstd[0], metal_meanstd[1], gmc.stellarmass_longlived(), gmc.stellarmass())
    

# load parameters
# default parameters
cfg = yaml.load(file('defaultconfig.cfg'))
# update with config file, if specified
if len(sys.argv) > 1:
	override_cfg = yaml.load(file(sys.argv[1]))
	for option in override_cfg:
		cfg[option] = override_cfg[option]
cloudmasses = 10.0**logmass
ncloud = len(cloudmasses)
nobjects = len(object_indices)


if parallel and not obj_flag:
	# Create parallel multiprocessing pool
	pool_size = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(processes=pool_size)
	pool_outputs = pool.map(evaluate_cloud, cloudmasses)
	pool.close()
	pool.join()

	# Create arrays to store cluster statistics
	metalmean, metalstd, gcmass, gcmassinit = zip(*pool_outputs)
	metalmean = np.array(metalmean)
	metalstd = np.array(metalstd)
	gcmass = np.array(gcmass)
	gcmassinit = np.array(gcmassinit)
else:
	# Create arrays to store cluster statistics and cluster objects
    metalmean = np.zeros(ncloud)
    metalstd = np.zeros(ncloud)
    gcmass = np.zeros(ncloud)
    gcmassinit = np.zeros(ncloud)
    gmcs = [False] * nobjects

	# Go through each cloud mass that's specified and do its full evaluation. Store
	# statistics and, if specified and in object_indices, the objects themselves.
    for gmci in xrange(ncloud):
        if obj_flag and (gmci in object_indices):
            objnum = object_indices.index(gmci)
            gmcs[objnum], metalmean[gmci], metalstd[gmci], gcmass[gmci], gcmassinit[gmci] = evaluate_cloud(cloudmasses[gmci], \
            	return_object=True)
        else:
            metalmean[gmci], metalstd[gmci], gcmass[gmci], gcmassinit[gmci] = evaluate_cloud(cloudmasses[gmci])


log_gcmass = np.log10(gcmass)
log_gcmassinit = np.log10(gcmassinit)

if plot_flag:
	# Read in Mieske10 blue tilt points
	mieske_obs = np.loadtxt('obsdata/mieske-f5a-bluetilt.txt', delimiter=',')
	# Read in Willman & Strader 2012 plus supplement points
	ws_obs = np.genfromtxt('obsdata/WillmanStrader.dat', dtype={'names': ('Name', 'FeH',
		'FeHerr', 'sigma', 'sigma_uperr', 'sigma_loerr', 'MV', 'Mi', 'Mierr'), 
		'formats': ('S20', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4')},
		missing_values='...')
	# Barbinot's masses are in units of 1e5 Msun
	ws_obs['Mi'] *= 1e5
	ws_obs['Mierr'] *= 1e5
	ws_metal_poor = (ws_obs['FeH'] < -1.4)
	ws_metal_rich = (ws_obs['FeH'] >= -1.4)
	# Assume M/L=2
	MVsun = 4.83
	MLratio = 2.0
	ws_mass = MLratio * 10**(-0.4 * (ws_obs['MV']-MVsun))
	ws_logmass = np.log10(ws_mass)

	# Figure 1: [Fe/H] vs GC mass
	
	fig = plt.figure()
	
	plt.plot(log_gcmass, FeH(metalmean), 'k.', label='GCZCSE')
	plt.ylabel('[Fe/H]')
	plt.xlabel('log $M_{\mathrm{GC}}$')
	plt.ylim(-2.0, -0.5)


	# Highlight histogram points?
	if histplot_highlight_flag:
	    plt.plot(log_gcmass[histplot_ind], FeH(metalmean[histplot_ind]), 'kD')

	# Add observational points
	plt.plot(mieske_obs[:,1], FeH(mieske_obs[:,0]), 'o', color='orange', label='Mieske10')
	
	plt.legend(loc='upper left')

	plt.tight_layout()
	plt.text(0.01, 0.95, 'a)', size=20, transform=fig.transFigure)
	
	plt.savefig('%s_MZ.pdf' % cfg['output'])
	
	
	# Figure 1b: [Fe/H] vs GC initial mass (obs: shifted by Goudfrooij mass loss)
	
	fig = plt.figure()

	# Read in Goudfrooij14 mass loss points (their figure 5b)
	massloss = np.loadtxt('goudfrooij14/Goudfrooij_2014_ApJ_780_43_f5.txt', delimiter=',')
	massloss_mgc = massloss[:,0]
	massloss_logmgcmgci = massloss[:,1]
	massloss_mgci = massloss_mgc - massloss_logmgcmgci  # log initial mass
	# create interpolation function. Range is almost identical to Mieske data points, so
	# it is safe to extrapolate
	mgci_interp = interp1d(massloss_mgc, massloss_mgci, fill_value='extrapolate')
	
	# plot model
	plt.plot(log_gcmassinit, FeH(metalmean), 'k.', label='GCZCSE')
	plt.ylabel('[Fe/H]')
	plt.xlabel('log $M_{\mathrm{init}}$')
	plt.ylim(-2.0, -0.5)
	
	# Highlight histogram points?
	if histplot_highlight_flag:
	    plt.plot(log_gcmassinit[histplot_ind], FeH(metalmean[histplot_ind]), 'kD')

	# Add observational points
	plt.plot(mgci_interp(mieske_obs[:,1]), FeH(mieske_obs[:,0]), 'o', color='orange', label='Mieske10')
	
	plt.legend(loc='upper left')
	
	plt.tight_layout()
	plt.text(0.01, 0.95, 'b)', size=20, transform=fig.transFigure)
	
	plt.savefig('%s_MiZ.pdf' % cfg['output'])
	


	# Figure 2: sigma_[Fe/H] vs GC mass

	fig = plt.figure()
	plt.plot(log_gcmass, metalstd, 'k.', label='GCZCSE')
	plt.ylabel('$\sigma_{\mathrm{[Fe/H]}}$')
	plt.xlabel('log $M_{\mathrm{GC}}$')

	# Highlight histogram points?
	if histplot_highlight_flag:
	    plt.plot(log_gcmass[histplot_ind], metalstd[histplot_ind], 'kD')

	# Add observational points
	plt.errorbar(ws_logmass[ws_metal_poor], ws_obs['sigma'][ws_metal_poor], yerr=[ws_obs['sigma_loerr'][ws_metal_poor], ws_obs['sigma_uperr'][ws_metal_poor]],
		fmt='o', color='blue', label='MW GCs [Fe/H]<-1.4')
	plt.errorbar(ws_logmass[ws_metal_rich], ws_obs['sigma'][ws_metal_rich], yerr=[ws_obs['sigma_loerr'][ws_metal_rich], ws_obs['sigma_uperr'][ws_metal_rich]],
		fmt='o', color='red', label='MW GCs [Fe/H]>-1.4')

	plt.ylim(ymin=0)
	plt.legend(loc='upper left')

	plt.tight_layout()
	plt.text(0.01, 0.95, 'a)', size=20, transform=fig.transFigure)

	plt.savefig('%s_MsigmaZ.pdf' % cfg['output'])



	# Figure 2a: sigma_[Fe/H] vs initial GC mass (obs: use Barbinot)
	
	fig = plt.figure()
	plt.plot(log_gcmassinit, metalstd, 'k.', label='GCZCSE')
	
	plt.errorbar(np.log10(ws_obs['Mi'][ws_metal_poor]), ws_obs['sigma'][ws_metal_poor], \
		yerr=[ws_obs['sigma_loerr'][ws_metal_poor], ws_obs['sigma_uperr'][ws_metal_poor]], \
		xerr=logerr(ws_obs['Mi'][ws_metal_poor], ws_obs['Mierr'][ws_metal_poor]), \
		fmt='o', color='blue', label='MW GCs [Fe/H]<-1.4')
	plt.errorbar(np.log10(ws_obs['Mi'][ws_metal_rich]), ws_obs['sigma'][ws_metal_rich], \
		yerr=[ws_obs['sigma_loerr'][ws_metal_rich], ws_obs['sigma_uperr'][ws_metal_rich]], \
		xerr=logerr(ws_obs['Mi'][ws_metal_rich], ws_obs['Mierr'][ws_metal_rich]), \
		fmt='o', color='red', label='MW GCs [Fe/H]>-1.4')
	plt.ylabel('$\sigma_{\mathrm{[Fe/H]}}$')
	plt.xlabel('log $M_{\mathrm{init}}$')
	
	# Highlight histogram points?
	if histplot_highlight_flag:
	    plt.plot(log_gcmassinit[histplot_ind], metalstd[histplot_ind], 'kD')

	plt.ylim(ymin=0)
	plt.legend(loc='upper left')
	
	plt.tight_layout()
	plt.text(0.01, 0.95, 'b)', size=20, transform=fig.transFigure)

	plt.savefig('%s_Mi_sigmaZ.pdf' % cfg['output'])
	
	

	# Figure 3: sigma_[Fe/H]  vs  [Fe/H]
	
	plt.figure()
	plt.plot(FeH(metalmean), metalstd, 'k.', label='GCZCSE')
	plt.xlabel('[Fe/H]')
	plt.ylabel('$\sigma_{\mathrm{[Fe/H]}}$')
	
	# Highlight histogram points?
	if histplot_highlight_flag:
	    plt.plot(FeH(metalmean[histplot_ind]), metalstd[histplot_ind], 'kD')

	# Add observational points
	plt.errorbar(ws_obs['FeH'][ws_metal_poor], ws_obs['sigma'][ws_metal_poor], xerr=ws_obs['FeHerr'][ws_metal_poor],
		yerr=[ws_obs['sigma_loerr'][ws_metal_poor], ws_obs['sigma_uperr'][ws_metal_poor]], fmt='o', color='blue', label='MW GCs [Fe/H]<-1.4')
	plt.errorbar(ws_obs['FeH'][ws_metal_rich], ws_obs['sigma'][ws_metal_rich], xerr=ws_obs['FeHerr'][ws_metal_rich],
		yerr=[ws_obs['sigma_loerr'][ws_metal_rich], ws_obs['sigma_uperr'][ws_metal_rich]], fmt='o', color='red', label='MW GCs [Fe/H]>-1.4')

	# Slope of the sigma_Z-Z plot. Force it to go through initial metallicity as intercept.
	FeH_init = cfg['pre_enrich_level'] - 0.25
	FeH_minus_initial = FeH(metalmean) - FeH_init
	ZsigmaZ_slope = FeH_minus_initial.dot(metalstd) / FeH_minus_initial.dot(FeH_minus_initial)
	Zax = np.arange(FeH_init, FeH(metalmean.max()), 0.01)
	#plt.plot(Zax, (Zax-FeH_init)*ZsigmaZ_slope, 'g--')
	
	outf = open('%s_slope.txt' % cfg['output'], 'w')
	outf.write('Slope: sigma_[Fe/H] = %0.3f ([Fe/H] - [Fe/H]_init)\n' % (ZsigmaZ_slope))
	outf.write('Slope: [Fe/H] = [Fe/H]_init + %0.3f sigma_[Fe/H]\n' % (1./ZsigmaZ_slope))
	outf.close()

	plt.xlim(-2.5,0.1)
	plt.ylim(ymin=0.0)
	plt.legend(loc='best')

	plt.tight_layout()
	plt.savefig('%s_Z_sigmaZ.pdf' % cfg['output'])



if histplot_flag and obj_flag:
    Zrange = np.array([cfg['pre_enrich_level']-0.1, np.max(gmcs[-1].clump_Z_mass()[0])+0.1])
    FeHrange = FeH(Zrange)
    fig, axarr = plt.subplots(len(histplot_ind), sharex=True, figsize=(8,14), gridspec_kw={'left':0.2, 'right':0.95})
    for ploti, gci in enumerate(histplot_ind):
    	# plt.subplots gives an unindexable Axes object if given an argument of 1, so need to make
    	# a special case to pick out the right Axes in both cases
    	if len(histplot_ind)==1:
    		thisax = axarr
    	else:
    		if histplot_upsidedown:
    			thisax = axarr[nobjects-ploti-1]
    		else:
	    		thisax = axarr[ploti]
    
    	print ploti, gci
    	
        clump_metal, clump_mass, clump_numstars = gmcs[ploti].clump_Z_mass_N_longlived()
        meanpt = FeH(metalmean[gci])
        stdpt = metalstd[gci]
        
        histheights, histbins, histpatches = thisax.hist(FeH(gc.ZZsun(clump_metal)), bins=50, weights=clump_mass, \
            range=FeHrange, color='red', zorder=10)
        thisax.text(0.9, 0.8, '$%s \> M_{\odot}$' % logsci_not(gcmass[gci], 1), horizontalalignment='center', \
        	transform=thisax.transAxes, zorder=15)
        thisax.locator_params(axis='y', nbins=3)
            
        hist_top = thisax.get_ylim()[1] * 1.05
            
        thisax.add_patch(patches.Rectangle( (meanpt-stdpt, 0), 2*stdpt, hist_top, 
        	linestyle=None, fill=True, color='black', alpha=0.3, zorder=1))
        thisax.plot( [meanpt,meanpt], [0,hist_top], 'k-', zorder=5)
        thisax.set_ylim(0, hist_top)
        thisax.yaxis.set_major_formatter(FuncFormatter(lambda x,p: '$%s$' % sci_not(x,1)))
        
	if len(histplot_ind)==1:
		lastax = axarr
		middleax = axarr
	else:
		lastax = axarr[-1]		    
		middleax = axarr[len(axarr)//2]
    lastax.set_xlabel('[Fe/H]', size=18.0)

    
    middleax.set_ylabel('Mass [$M_{\odot}$]', size=18.0)
    
    plt.savefig('%s_histZ.pdf' % cfg['output'])


# Write out clusters to file
np.savetxt('%s_clusters.dat' % cfg['output'], zip(log_gcmass, log_gcmassinit, FeH(metalmean), metalstd), header='Log(M_GC) Log(M_init) [Fe/H] sigma([Fe/H])', fmt='%.4f')

# Write out objects to a pickle
if pkl_dir:
	gmcdata = {"cfg":cfg, "cloudmasses":cloudmasses, "ncloud":ncloud, "gmcs":gmcs, \
		"metalmean":metalmean, "metalstd":metalstd, "logmass":logmass, "gcmass":gcmass, \
		"gcmassinit":gcmassinit, "log_gcmass":log_gcmass, "log_gcmassinit":log_gcmassinit, \
		"object_indices":object_indices}
	pickle.dump(gmcdata, open("%s/%s.pkl" % (pkl_dir, cfg['output']), "wb"))
