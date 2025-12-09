import numpy as np
#import healpy as hp
import pandas as pd
import h5py

from utilities.standard_cosmology import *
#from ..utilities.schechter_function import *

class galaxyCatalog(object):
    """
    Galaxy catalog class stores a dictionary of galaxy objects.

    Parameters
    ----------
    catalog_file : Path to catalog.hdf5 file
    skymap_filename : Path to skymap.fits(.gz) file
    thresh: Probablity contained within sky map region
    band: key of band
    """
    def __init__(self, catalog_file=None):
        self.catalog_file = catalog_file
        self.dictionary = self.__load_catalog()
        self.extract_galaxies()

        if catalog_file is None:
            self.catalog_name = ""
            self.dictionary = {'ra': [], 'dec': [], 'z': [], 'radec_lim': [], 'm': []}

    def __load_catalog(self):
        return h5py.File(self.catalog_file,'r')

    def nGal(self):
        return len(self.dictionary['z'])

    def get_galaxy(self, index):
        return galaxy(index, self.ra[index], self.dec[index],
                      self.z[index], self.m[index])

    def mth(self, type="median"):
        m = self.m
        if sum(m) == 0:
            mth = 25.0
        elif type=="median":
            mth = np.median(m)
        elif type=="max":
            mth=max(m)
        return mth

    def extract_galaxies(self):
        ra = self.dictionary['ra'][:]
        dec = self.dictionary['dec'][:]
        z = self.dictionary['z'][:]
        m = self.dictionary['m'][:]
        radec_lim = self.dictionary['radec_lim'][:]
        print("radec_lims (if all zeros, assumed no lims (but check) )", radec_lim)
        self.radec_lim = radec_lim
        mask = ~np.isnan(m)
        ra, dec, z, m = ra[mask], dec[mask], z[mask], m[mask]
        self.ra, self.dec, self.z, self.m = ra, dec, z, m
        return ra, dec, z, m

class galaxyCatalogweighted(object):
    """
    Galaxy catalog class stores a dictionary of galaxy objects.

    Parameters
    ----------
    catalog_file : Path to catalog.hdf5 file
    skymap_filename : Path to skymap.fits(.gz) file
    thresh: Probablity contained within sky map region
    band: key of band
    """
    def __init__(self, catalog_file=None):
        self.catalog_file = catalog_file
        self.dictionary = self.__load_catalog()
        self.extract_galaxies()

        if catalog_file is None:
            self.catalog_name = ""
            self.dictionary = {'ra': [], 'dec': [], 'z': [], 'radec_lim': [], 'm': [], 'weights':[]}

    def __load_catalog(self):
        return h5py.File(self.catalog_file,'r')

    def nGal(self):
        return len(self.dictionary['z'])

    def get_galaxy(self, index):
        return galaxyweighted(index, self.ra[index], self.dec[index], self.z[index], self.m[index], self.weight[index])

    def mth(self, type="median"):
        m = self.m
        if sum(m) == 0:
            mth = 25.0
        elif type=="median":
            mth = np.median(m)
        elif type=="max":
            mth=min(m)
        return mth

    def extract_galaxies(self):
        ra = self.dictionary['ra'][:]
        dec = self.dictionary['dec'][:]
        z = self.dictionary['z'][:]
        m = self.dictionary['m'][:]
        weights = self.dictionary['weights'][:]
        radec_lim = self.dictionary['radec_lim'][:]
        print("radec_lims (if all zeros, assumed no lims (but check) )", radec_lim)
        self.radec_lim = radec_lim
        mask = ~np.isnan(m)
        ra, dec, z, m, weights = ra[mask], dec[mask], z[mask], m[mask], weights[mask]
        self.ra, self.dec, self.z, self.m, self.weights = ra, dec, z, m, weights
        return ra, dec, z, m

class galaxy(object):
    """
    Class to store galaxy objects.

    Parameters
    ----------
    index : galaxy index
    ra : Right ascension in radians
    dec : Declination in radians
    z : Galaxy redshift
    m : Apparent magnitude 
    """
    def __init__(self, index=0, ra=0, dec=0, z=1, m=20):
        self.index = index
        self.ra = ra
        self.dec = dec
        self.z = z
        self.m = m
        self.dl = self.luminosity_distance()
        self.M = self.absolute_magnitude()
        self.L = self.luminosity()

    def luminosity_distance(self, H0=70.):
        return dl_zH0(self.z, H0)
    
    def absolute_magnitude(self, H0=70.):
        return M_mdl(self.m, self.dl)
    
    def luminosity(self, band=None):
        return L_M(self.M)

class galaxyweighted(object):
    """
    Class to store galaxy objects.

    Parameters
    ----------
    index : galaxy index
    ra : Right ascension in radians
    dec : Declination in radians
    z : Galaxy redshift
    m : Apparent magnitude 
    """
    def __init__(self, index=0, ra=0, dec=0, z=1, m=20, weight=1):
        self.index = index
        self.ra = ra
        self.dec = dec
        self.z = z
        self.m = m
        self.weight=weight
        self.dl = self.luminosity_distance()
        self.M = self.absolute_magnitude()
        self.L = self.luminosity()

    def luminosity_distance(self, H0=70.):
        return dl_zH0(self.z, H0)
    
    def absolute_magnitude(self, H0=70.):
        return M_mdl(self.m, self.dl)
    
    def luminosity(self, band=None):
        return L_M(self.M)

 