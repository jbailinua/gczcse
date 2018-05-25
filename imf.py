# Clumpy globular cluster self-enrichment.
# Jeremy Bailin (jbailin@ua.edu)

# module for creating and sampling IMFs

import numpy as np

class Salpeter(object):
    """An IMF object that can be sampled from that produces dN/dM proportional
    to M^alpha between a minimum and maximum mass. A truncated Salpeter IMF is
    an example of this, with alpha=-2.35. The object is normally created using
    the global parameters, and then sampled from using Salpeter.sample(N).
    Useful properties are Salpeter.averagemass (average mass per star) and
    Salpeter.N_per_solar_mass (the inverse -- average number of stars per solar
    mass)."""
    def __init__(self, params):
        """Create the Salpeter IMF object with the global parameters. The
        parameters used in the IMF are:
            'imf_salpeter_slope': the exponent alpha in dN/dM = M^alpha
            'imf_salpeter_mmin':  minimum mass of the IMF
            'imf_salpeter_mmax':  maximum mass of the IMF"""

        self.params = params
        # useful numbers to precompute
        self.aplus1 = self.params['imf_salpeter_slope']+1.
        self.inv_aplus1 = 1. / self.aplus1
        self.aplus2 = self.params['imf_salpeter_slope']+2.
        self.factor_min = self.params['imf_salpeter_mmin']**self.aplus1
        self.factor_maxmin = self.params['imf_salpeter_mmax']**self.aplus1 - self.factor_min
        # normalization constant can be expressed as average mass per star
        # or number of stars per solar mass. Both are useful, so store them
        # both.
        self.averagemass = (self.aplus1 / self.aplus2) * \
            (self.params['imf_salpeter_mmax']**self.aplus2 - self.params['imf_salpeter_mmin']**self.aplus2) / \
            (self.params['imf_salpeter_mmax']**self.aplus1 - self.factor_min)
        self.N_per_solar_mass = 1. / self.averagemass

    def sample(self, N):
        """Sample N stars from the IMF."""
        xi = np.random.rand(N)
        return (self.factor_maxmin*xi + self.factor_min) ** self.inv_aplus1


imftypedict = {'salpeter':Salpeter}

class IMFException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)



def IMF(typename, *arg):
    """Returns the appropriate IMF class to be sampled from. Current
    possibilities are:
        'salpeter': Power law truncated at bottom and top ends (Salpeter is an
                    example of this).

    For example, to create an IMF object with a Salpeter IMF with global
    parameters given in the cfg dict:
        imfobj = IMF('salpeter', cfg)"""

    if typename in imftypedict:
        return imftypedict[typename](*arg)
    else:
        raise IMFException("Unknown IMF type %s." % (typename))



