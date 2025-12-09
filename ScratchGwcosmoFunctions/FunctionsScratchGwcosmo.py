import numpy as np
import scipy 
import sys
import h5py
import progressbar
from scipy import special
from scipy.stats import gaussian_kde
from scipy.interpolate import splev, splrep, interp1d
from utilities.standard_cosmology import *
from utilities.schechter_function import *
from utilities.schechter_params import *
from scipy.integrate import quad, dblquad
from scipy.special import logsumexp
import matplotlib.pyplot as plt

epsrelforFunctionsScratch=1.49e-4
epsabsforFunctionsScratch=0

def GetNormalizationAndNormalized(x, y):
    norm=np.trapz(y,x)
    return norm, y/norm

def Cubeprior(z):
    return z**3

def Miceprior(z):
    splrepvalues=[[0.0005, 0.0005, 0.0005, 0.0005, 0.0025, 0.0035, 0.0045, 0.0055,
       0.0065, 0.0075, 0.0085, 0.0095, 0.0105, 0.0115, 0.0125, 0.0135,
       0.0145, 0.0155, 0.0165, 0.0175, 0.0185, 0.0195, 0.0205, 0.0215,
       0.0225, 0.0235, 0.0245, 0.0255, 0.0265, 0.0275, 0.0285, 0.0295,
       0.0305, 0.0315, 0.0325, 0.0335, 0.0345, 0.0355, 0.0365, 0.0375,
       0.0385, 0.0395, 0.0405, 0.0415, 0.0425, 0.0435, 0.0445, 0.0455,
       0.0465, 0.0475, 0.0485, 0.0495, 0.0505, 0.0515, 0.0525, 0.0535,
       0.0545, 0.0555, 0.0565, 0.0575, 0.0585, 0.0595, 0.0605, 0.0615,
       0.0625, 0.0635, 0.0645, 0.0655, 0.0665, 0.0675, 0.0685, 0.0695,
       0.0705, 0.0715, 0.0725, 0.0735, 0.0745, 0.0755, 0.0765, 0.0775,
       0.0785, 0.0795, 0.0805, 0.0815, 0.0825, 0.0835, 0.0845, 0.0855,
       0.0865, 0.0875, 0.0885, 0.0895, 0.0905, 0.0915, 0.0925, 0.0935,
       0.0945, 0.0955, 0.0965, 0.0975, 0.0995, 0.0995, 0.0995, 0.0995], [-4.47545209e-17,  2.87807613e+00,  5.24384774e+00, -3.69910495e+00,
        1.17883102e+01,  1.05458641e+01,  3.00282334e+01,  4.93412022e+01,
        4.86069576e+01,  3.82309673e+01,  9.24691731e+01,  5.38923402e+01,
        1.59961466e+02,  1.34261796e+02,  8.29913514e+01,  2.23772799e+02,
        1.31917453e+02,  2.08557390e+02,  2.09852989e+02,  7.40306548e+01,
        4.12024392e+02,  2.39871778e+02,  3.02488496e+02,  4.28174238e+02,
        3.90814554e+02,  4.32567547e+02,  2.54915257e+02,  2.81771424e+02,
        4.65999047e+02,  5.60232388e+02,  3.59071400e+02,  5.77482012e+02,
        4.57000553e+02,  8.04515774e+02,  8.78936349e+02,  1.16373883e+03,
        1.28210833e+03,  9.85827853e+02,  1.33258026e+03,  1.68785111e+03,
        1.76801529e+03,  2.78408771e+03,  2.11963386e+03,  2.35737685e+03,
        2.15485875e+03,  2.01318814e+03,  1.79838870e+03,  2.39125705e+03,
        2.31058310e+03,  1.96241053e+03,  1.94177476e+03,  2.43249042e+03,
        2.09826357e+03,  1.96045531e+03,  2.14391518e+03,  2.65188395e+03,
        2.22454902e+03,  2.24991999e+03,  2.35377103e+03,  2.48899590e+03,
        2.58824538e+03,  2.48202258e+03,  3.46766430e+03,  2.60732024e+03,
        3.26905476e+03,  2.98246073e+03,  2.77710232e+03,  3.16513001e+03,
        3.11437765e+03,  3.31335937e+03,  3.02418485e+03,  4.32390123e+03,
        2.99021021e+03,  3.85125791e+03,  4.06275816e+03,  3.57370947e+03,
        4.56840396e+03,  3.90467470e+03,  4.80889723e+03,  4.13573639e+03,
        5.25815722e+03,  4.55563474e+03,  4.72530382e+03,  5.38514998e+03,
        4.52609627e+03,  5.25646494e+03,  4.65204399e+03,  5.35535912e+03,
        5.56451953e+03,  5.30256276e+03,  6.09322943e+03,  5.98251953e+03,
        5.81469247e+03,  6.88471061e+03,  6.70846510e+03,  6.53542899e+03,
        7.39181894e+03,  6.46312071e+03,  6.70893965e+03,  7.09400000e+03,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00], 3]
    return splev(z, splrepvalues, ext=0)

def GaussianSigmad(x,mu, sigmaprop):
    return 1/(sigmaprop*x*np.sqrt(2*np.pi))*np.exp(-0.5*(x-mu)**2/(sigmaprop*x)**2)#/norm
    
def UniformSigmad(dl, dl_gw, sigma):
    if sigma/(sigma+1) * dl_gw <= dl and dl<=sigma/(sigma-1)*dl_gw:
        return sigma/(dl*np.log((sigma+1)/(sigma-1)))
    else:
        return 0

def ztoZshell(z, zhist_edges):
    return np.where(z>=zhist_edges)[0][-1]

def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

class PrecomputingClass(object):
    def __init__ (self, galaxy_catalog, CatalogFull, dl_det, Omega_m, linear, assumed_band, zmax, sigmaprop, weightedcat, zpriortouse, dl_distr, pdettype, weighted):
        
        self.dl_det=dl_det
        self.sigmaprop=sigmaprop
        self.Omega_m=Omega_m
        self.linear=linear
        self.assumed_band=assumed_band
        self.zmax=zmax
        self.cosmo = fast_cosmology(Omega_m=Omega_m, linear=linear)
        
        self.zpriortouse = zpriortouse
        print(self.zpriortouse)
        
        if self.zpriortouse == "uniform":
            self.zprior = redshift_prior(Omega_m=Omega_m, linear=linear)
        elif self.zpriortouse == "Mice":
            fullzs=CatalogFull.z
            zbins=np.linspace(0,zmax, 100)
            zcenters=[(zbins[1]+zbins[0])/2+(zbins[1]-zbins[0])*i for i in range(len(zbins)-1)]
            hist=np.histogram(fullzs, zbins)
            #self.zprior = interp1d(zcenters, hist[0])
            
            self.zprior = np.polynomial.polynomial.Polynomial.fit(zcenters, hist[0], deg=20)
            #sys.exit()
        
        elif zpriortouse=="Micesplev":
            self.zprior = Miceprior

        elif self.zpriortouse == "cube":
            self.zprior = Cubeprior
        
        sp = SchechterParams(assumed_band)

        self.alpha = sp.alpha
        self.Mstar_obs = sp.Mstar
        self.Mobs_min = sp.Mmin
        self.Mobs_max = sp.Mmax

        if galaxy_catalog is not None:
            self.galaxy_catalog = galaxy_catalog #add hdf5 extraction and mth calculation
            self.mth = galaxy_catalog.mth(type="max")
            
            self.allz = galaxy_catalog.z
            self.allra = galaxy_catalog.ra
            self.alldec = galaxy_catalog.dec
            self.allm = galaxy_catalog.m
            self.nGal = len(self.allz)


        #other less important options
        self.weightedcat=weightedcat
        self.dl_distr=dl_distr
        self.pdettype = pdettype
        self.weighted=weighted

    def ps_z_precompute(self, z):
        rate="constant"
        if rate == 'constant':
            return 1.0
        if rate == 'evolving':
            return (1.0+z)**Lambda
        
    def px_dl_direct_gaussiansigmad(self, dl):
        return GaussianSigmad(x=dl, mu=self.dl_gw, sigmaprop=self.sigmaprop)
    
    def pD_Heaviside_theoretic_precompute(self, z, H0):
        dl=self.cosmo.dl_zH0(z, H0)
        erfvariable=(dl-self.dl_det)/(np.sqrt(2)*self.sigmaprop*dl) #should be this one instead of next one. Jon says no sqrt(2), shouldn't change regardless cause it's just overall factor
        #erfvariable=(dl-dl_det)/(sigmaprop*dl) 
        erfvariable*=-1 #There might be an inconsistency between hitchhiker and scipy in definition of error function, the minus sign produces something much closer to H0**3/gwcosmo
        return 0.5 * (1 + special.erf(erfvariable))

    def fofz(self, z):
        # fraction of observed galaxies at redshift z

        H0ass=70. #It actually does not depend on H0, but easy to just put as a placeholder and not modify the basic functions
        Mmin = M_Mobs(H0ass,self.Mobs_min)
        Mmax = M_Mobs(H0ass,self.Mobs_max)

        def I(M):
            return SchechterMagFunction(H0=H0ass,Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)

        num = quad(I,Mmin, min(max(M_mdl(self.mth,self.cosmo.dl_zH0(z,H0ass)),Mmin),Mmax),epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0]
        den = quad(I, Mmin, Mmax, epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0]
        
        return num/den

    def Comov_volume(self, z):
        #volume element within z. Note that eventual normalization factor don't matter, as this only appear in likelihood through dVc/dz / Vc(z)
        if self.linear==True:
            return z**3
    
    def Derivative_comov_volume(self, z):
        #derivative of volume element within z. Note that eventual normalization factor don't matter, as this only appear in likelihood through dVc/dz / Vc(z)
        if self.linear==True:
            return 3*z**2

    def pzout(self, z):
        #this computes the out of catalog part of equation  9
        return (1-self.fofz(z))*self.Derivative_comov_volume(z)/self.Comov_volume(self.zmax)
    
    def pD_inside_theoretic_precompute(self, H0):
        "This computes the pdet theoretically from eq 22 of hitchikers"
        den = np.zeros(len(H0))
        zs = self.allz
        if self.weightedcat==True:
            weights=self.galaxy_catalog.weights

            
        bar = progressbar.ProgressBar()
        print("Calculating p(D|G, H0)")
        for i in bar(range(len(zs))):
            # loop over random draws from galaxies
            if self.weightedcat==True:
                weight = weights[i]
            else:
                weight = 1.0

            if self.dl_distr=="gaussian-sigmad":
                prob = self.pD_Heaviside_theoretic_precompute(zs[i],H0).flatten()

            deninner = prob*weight*self.ps_z_precompute(zs[i])
            den += deninner

        pDG = den/len(zs)
        #print(pDG)
        return pDG
    
    def pG_theoretic_precompute(self, H0):
        # Warning - this integral misbehaves for small values of H0 (<25 kms-1Mpc-1).  TODO: fix this.
        num = np.zeros(len(H0)) 
        den = np.zeros(len(H0))
        bar = progressbar.ProgressBar()
        print("Calculating p(G|H0,D)")
        for i in bar(range(len(H0))):
            def I(z,M):
                
                if self.pdettype==True:
                    temp = SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)*self.pdet.pD_zH0_eval(z,H0[i])*self.zprior(z)*self.ps_z(z)
                elif self.pdettype=='cube':
                    temp = SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)*H0[i]**3*self.zprior(z)*self.ps_z(z)
                elif self.pdettype=="theoretic":
                    temp = SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)*self.pD_Heaviside_theoretic_precompute(z,H0[i])*self.zprior(z)*self.ps_z_precompute(z)   
                elif self.pdettype=="ones":
                    temp = SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)*self.zprior(z)*self.ps_z(z)
                #if self.weighted:
                #    return temp*L_M(M)
                #else:
                return temp
            
            # Mmin and Mmax currently corresponding to 10L* and 0.001L* respectively, to correspond with MDC
            # Will want to change in future.
            # TODO: test how sensitive this result is to changing Mmin and Mmax.
            Mmin = M_Mobs(H0[i],self.Mobs_min)
            Mmax = M_Mobs(H0[i],self.Mobs_max)
            #num[i] = dblquad(I,0,zmax,lambda x: Mmin,lambda x: min(max(M_mdl(mth,cosmo.dl_zH0(x,H0[i])),Mmin),Mmax),epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0]
            #num[i] = dblquad(I,0,self.zcut,lambda x: Mmin,lambda x: M_mdl(self.mth,self.cosmo.dl_zH0(x,H0[i])),epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0]
            #den[i] = dblquad(I,0,zmax,lambda x: Mmin,lambda x: Mmax,epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0]
            num[i] = dblquad(I,Mmin, Mmax, lambda x: 0, lambda x: min(z_dlH0(dl_mM(self.mth,x),H0[i],linear=self.linear), self.zmax),epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0]
            den[i] = dblquad(I, Mmin, Mmax, lambda x: 0, lambda x: self.zmax, epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0]

        pGD = num/den
        #print("PG NUM : ", num)
        #print("PG DEN : ", den)
        return pGD

    def pD_outside_theoretic_precompute(self, H0):
        den = np.zeros(len(H0))
        
        #All legacy stuff probably not needed anymore
        #def skynorm(dec,ra):
        #    return np.cos(dec)
        #redshiftnorm=zmax**3/3
        #norm = dblquad(skynorm,ra_min,ra_max,lambda x: dec_min,lambda x: dec_max,epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0]/(4.*np.pi)
        #print("Sky stuff: ramin, ramax, decmin, decmax, norm : ", ra_min, ra_max, dec_min, dec_max, norm)

        bar = progressbar.ProgressBar()
        print("Calculating p(D|H0,bar{G})")
        for i in bar(range(len(H0))):
            Mmin = M_Mobs(H0[i], self.Mobs_min)
            Mmax = M_Mobs(H0[i], self.Mobs_max)
            
            #This seems to renormalize the pdet fairly well. Remember must be present both here and in px
            def Itonorm(z,M):
                temp=SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)*self.zprior(z)*self.ps_z_precompute(z)
                return temp
            totnorm=dblquad(Itonorm,Mmin,Mmax,lambda x: z_dlH0(dl_mM(self.mth,x),H0[i],linear=self.linear),lambda x: self.zmax,epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0] #check whether to use this style of integral or the one in modificationskypatch=False/pG
        
            def Iden(z,M):
                temp = SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)*self.pD_Heaviside_theoretic_precompute(z,H0[i])*self.zprior(z)*self.ps_z_precompute(z)
                
                if self.weighted:
                    return temp*L_M(M)
                else:
                    return temp
                
            #SchechterNorm=quad(SchechterMagFunction(H0=H0[i],Mstar_obs=Mstar_obs,alpha=alpha), Mmin, Mmax)[0]
            #den[i] = dblquad(Iden,Mmin,Mmax,lambda x: min(z_dlH0(dl_mM(mth,x),H0[i],linear=linear), zmax),lambda x: zmax,epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0]#/(SchechterNorm*redshiftnorm) #check whether to use this style of integral or the one in modificationskypatch=False/pG
            den[i] = dblquad(Iden,Mmin,Mmax,lambda x: min(z_dlH0(dl_mM(self.mth,x),H0[i],linear=self.linear), self.zmax),lambda x: self.zmax,epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0]/totnorm#/(SchechterNorm*redshiftnorm) #check whether to use this style of integral or the one in modificationskypatch=False/pG
        
        #if whole_sky_cat == True: #TODO check what's up with this allsky thing, should be whole_cat but not sure
        #    pDnG = den*norm
        #else:
        #    pDnG = den*(1.-norm)
        #print(den)
        pDnG=den
        return pDnG
    
    def pD_outside_theoretic_precompute_Jon(self, H0):
        den = np.zeros(len(H0))
        bar = progressbar.ProgressBar()
        print("Calculating p(D|H0,bar{G})")
        for i in bar(range(len(H0))):
            def Integrand(z):
                return self.pzout(z)*self.pD_Heaviside_theoretic_precompute(z,H0[i])
            den[i] = quad(Integrand, 0, self.zmax, epsabs=epsabsforFunctionsScratch,epsrel=1.49e-9)[0]
        return den


class likelihood(object):
    """
    A class to hold all the individual components of the posterior for H0 in 1D,
    and methods to stitch them together in the right way.

    Parameters
    ----------
    GW_data : hdf5 samples. Groups are the str of event numbers, datasets are ra_samps, dec and dl, and ra_inj, dec and dl
    galaxy_catalog : FunctionsCatalog.galaxyCatalog object
        The relevant galaxy catalog
    Omega_m : float, optional
        The matter fraction of the universe (default=0.25)
    linear : bool, optional
        Use linear cosmology (default=False)
    weighted : bool, optional
        Use luminosity weighting (default=False)
    band : str, optional
        specify B or K band catalog (and hence Schechter function parameters) (default='B')
    modificationskypatch: if true, use the function to compute px_nG and pD_nG from the skypatch version, since the original ones cause trouble with FitActual and FitActualLinear schechter params
    """

    def __init__(self, EventNumber, GW_data, galaxy_catalog, insidepdet, outsidepdet, pGpdet, precomputedinsidepdet=False, precomputedpGpdet=False, precomputedoutsidepdet=False, Omega_m=0.25, linear=False, weighted=False, weights=None, assumed_band='B', samplestype='gaussian', rate='constant', pdettype='theoretic', sigmaprop=0.2, sigmaradec=0.2, dl_det=200, whole_sky_cat=True, weightedcatalog=False, directpx=False, directradec=False, dl_distr="gaussian-sigmad", summationtype="None", smoothedcatalog=False, normalizedgaussiansigmad=False, zpriortouse="uniform", zmax=0.1, saveprecomputes=True):
        self.precomputedinsidepdet=precomputedinsidepdet
        self.precomputedoutsidepdet=precomputedoutsidepdet
        self.precomputedpGpdet=precomputedpGpdet
        self.saveprecomputes=saveprecomputes
        self.insidepdet = insidepdet
        self.outsidepdet = outsidepdet
        self.pGpdet = pGpdet

        self.Omega_m = Omega_m
        self.linear = linear
        self.cosmo = fast_cosmology(Omega_m=self.Omega_m, linear=self.linear)
        self.H0_ass = 70. #This is used only in the normalization of gaussiansigmad, and should have close to no impact.

        self.weighted = weighted
        self.weights = weights #should be a list of 2 arrays, one with the z_hist_edges and one with the weights if custom. Otherwise None
        self.assumed_band = assumed_band
        sp = SchechterParams(self.assumed_band)
        self.alpha = sp.alpha
        self.Mstar_obs = sp.Mstar
        self.Mobs_min = sp.Mmin
        self.Mobs_max = sp.Mmax
        print("assumed band: ", assumed_band, " params ", self.alpha, self.Mstar_obs, self.Mobs_min, self.Mobs_max)
        self.rate=rate
        self.pdettype = pdettype
        self.dl_det=dl_det
        self.sigmaprop=sigmaprop
        self.sigmaradec=sigmaradec
        self.weightedcatalog=weightedcatalog
        self.directpx=directpx
        self.directradec=directradec
        self.dl_distr=dl_distr
        self.summationtype=summationtype
        self.smoothedcatalog=smoothedcatalog

        if self.directpx==True:
            self.normalizedgaussiansigmad=normalizedgaussiansigmad
        else: 
            self.normalizedgaussiansigmad=False

        if self.smoothedcatalog==True:
            z_bins=np.linspace(min(self.allz), max(self.allz), 1001)
            dz_bins=(max(self.allz)-min(self.allz))/1000
            self.z_array=np.linspace(min(self.allz)+dz_bins/2, max(self.allz)-dz_bins/2, 1000)
            self.z_vals=np.histogram(self.allz, bins=z_bins)[0]
            #self.tempz = splrep(self.z_array,self.z_vals)
        
        if galaxy_catalog is not None:
            self.galaxy_catalog = galaxy_catalog #add hdf5 extraction and mth calculation
            self.mth = galaxy_catalog.mth(type="max")
            
            self.allz = galaxy_catalog.z
            self.allra = galaxy_catalog.ra
            self.alldec = galaxy_catalog.dec
            self.allm = galaxy_catalog.m
            self.nGal = len(self.allz)


            
            if self.weightedcatalog==True:
                self.allcatalogweights = galaxy_catalog.weights
            
        

            self.whole_sky_cat=whole_sky_cat #assume that the catalog covers the whole sky? Shouldn't make much of a difference
            if self.whole_sky_cat == False:
            
                self.ra_min = min(self.allra)
                self.ra_max = max(self.allra)
                self.dec_min = min(self.alldec)
                self.dec_max = max(self.alldec)
                #self.zcut = max(self.allz) # TODO: replace with something read in from the catalogue itself (radec_lim), also check its usage. In any case only used for out of catalog
                self.zcut = max(self.allz) #TODO: this impacts significantly the estimate of pdetout and pG, see prepareSchechtercompletion script. Unclear distinction between zcut and zmax. Setting them equal and give an estimate which is safely above the max (2*max) z but also not too far seems to be working ok for high redshift catalog. Simply max is better for low redshift if completeness not too low
                def skynorm(dec,ra):
                    return np.cos(dec)
                self.catalog_fraction = dblquad(skynorm,self.ra_min,self.ra_max,lambda x: self.dec_min,lambda x: self.dec_max,epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0]/(4.*np.pi)
                self.rest_fraction = 1-self.catalog_fraction
                #print('This catalog covers {}% of the full sky'.format(self.catalog_fraction*100))

            elif self.whole_sky_cat==True:
                self.ra_min = 0.0
                self.ra_max = np.pi*2.0
                self.dec_min = -np.pi/2.0
                self.dec_max = np.pi/2.0
                self.zcut = max(self.allz) #see above zcut for comments
            else:
                print("WRONG whole_sky_cat. Please insert a valid option!")
                sys.exit()

        
        self.zmax=zmax
        if zpriortouse=="uniform":
            self.zprior = redshift_prior(Omega_m=self.Omega_m, linear=self.linear)
        elif zpriortouse=="Micepoly":
            zbins=np.linspace(0,self.zmax, 100)
            zcenters=[(zbins[1]+zbins[0])/2+(zbins[1]-zbins[0])*i for i in range(len(zbins)-1)]
            hist=np.histogram(self.allz, zbins)[0]
            #self.zprior = interp1d(zcenters, hist[0])
            self.zprior = np.polynomial.polynomial.Polynomial.fit(zcenters, hist, deg=20)
            #sys.exit()
        elif zpriortouse=="Micesplev":
            self.zprior = Miceprior
        elif zpriortouse == "cube":
            self.zprior = Cubeprior
        elif zpriortouse == "Micepolyextended":
            zbins=np.linspace(0,self.zmax, 100)
            zcenters=[(zbins[1]+zbins[0])/2+(zbins[1]-zbins[0])*i for i in range(len(zbins)-1)]
            hist=np.histogram(self.allz, zbins)[0]
            deltaz=zcenters[1]-zcenters[0]
            hist=list(hist)

            norm=len(self.allz)*deltaz*3/self.zmax**3

            zcentersextended=zcenters
            histextended=hist

            lastz=zcenters[-1]

            for i in range(len(zcenters)):
                newz=lastz+deltaz*(i+1)
                zcentersextended.append(newz)
                histextended.append(norm*newz**2)
            self.zprior = np.polynomial.polynomial.Polynomial.fit(zcenters, hist, deg=20)

        if GW_data is not None:
            #dl
            samps=h5py.File(GW_data, 'r')
            ##this next bit is used only for direct px_inside calculation, "skipping" the samples
            if self.directpx!=False:
                self.dl_gw=samps[EventNumber]["dl_gw"][()]
                if self.normalizedgaussiansigmad==True:
                    if self.normalizedgaussiansigmad==True:
                        def Integrand(x):
                            return GaussianSigmad(x, mu=self.dl_gw, sigmaprop=self.sigmaprop)
                        self.gaussiansigmadnorm = quad(Integrand, 0, self.cosmo.dl_zH0(self.zmax, self.H0_ass))[0]
            if self.directradec!=False:
                self.ra_gw=samps[EventNumber]["ra_gw"][()]
                self.dec_gw=samps[EventNumber]["dec_gw"][()]
            if self.directpx==False:
                dlssamps=samps[EventNumber]["dl_samps"][:]

                distkernel=gaussian_kde(dlssamps)
                distmin = 0.5*np.amin(dlssamps)
                distmax = 2.0*np.amax(dlssamps)
                dl_array = np.linspace(distmin, distmax, 500)
                vals = distkernel(dl_array)
                self.tempdl = splrep(dl_array,vals)
                #ra
                rasamps=samps[EventNumber]["ra_samps"][:]
                decsamps=samps[EventNumber]["dec_samps"][:]
                #radecsamps=np.array(list(zip(rasamps, decsamps)))
                #print(radecsamps)
                #radeckernel=gaussian_kde(radecsamps)
                
                ra_array = np.linspace(self.ra_min, self.ra_max, 500)
                if any(rasamps)>0:
                    rakernel=gaussian_kde(rasamps)
                    ravals = rakernel(ra_array)
                else:
                    ravals=np.zeros(len(ra_array))
                self.tempra = splrep(ra_array,ravals)

                

                dec_array = np.linspace(self.dec_min, self.dec_max, 500)
                if any(decsamps)>0:

                    deckernel=gaussian_kde(decsamps)
                    decvals = deckernel(dec_array)
                else:
                    decvals=np.zeros(len(dec_array))
                self.tempdec = splrep(dec_array,decvals)
                
            self.tempsky = self.px_radec_independent(self.allra, self.alldec)
            self.avaragetempsky = sum(self.tempsky)/self.nGal

            if self.whole_sky_cat=="Truefast":
                print("True fast option outdated, to be revised!")
                sys.exit()
                """rafaststd=np.std(rasamps)
                decfaststd=np.std(decsamps)
                print(rafaststd, decfaststd)
                print("Minimum ra would be ", min(rasamps)-rafaststd)
                print("Maximum ra would be ", max(rasamps)+rafaststd)
                print("Minimum dec would be ", min(decsamps)-decfaststd)
                print("Maximum dec would be ", max(decsamps)+decfaststd)
                self.rafastmin=max(0, min(rasamps)-rafaststd) #define a min and max of ras so, when looping through galaxies in the inside catalog component, I can quickly exclude galaxies that are way out of bounds.  
                self.rafastmax=min(2*np.pi, max(rasamps)+rafaststd) #This probably means that when I have an event across the boundary (ra ~0 = 6.28) this limits don't do anything, but I'll still be faster in most of the other ones
                self.decfastmin=max(-np.pi/2, min(decsamps)-decfaststd) #same thing for dec, here there shouldn't be issues with boundary
                self.decfastmax=min(np.pi/2, max(decsamps)+decfaststd) 
                print("Event center ra ", np.mean(rasamps), " limits ", self.rafastmin, self.rafastmax)
                print("Event center dec ", np.mean(decsamps), " limits ", self.decfastmin, self.decfastmax)"""
            else:
                self.rafastmin=self.ra_min
                self.rafastmax=self.ra_max
                self.decfastmin=self.dec_min
                self.decfastmax=self.dec_max 
                
        self.pDG = None
        self.pGD = None
        self.pnGD = None
        self.pDnG = None

        # Note that zmax is an artificial limit that
        # should be well above any redshift value that could
        # impact the results for the considered H0 values. 

    def px_dl(self, dl):
        """
        Returns a probability for a given distance dl
        from the interpolated function.
        """
        return splev(dl, self.tempdl, ext=3)
    
    def px_dl_direct_gaussiansigmad(self, dl):
        #print("using direct gaussian!")
        "Computing px_dl directly, assuming sigmaprop is a known parameter and dl_gw is the measurement from the event. This function assumes a gaussian sigma d distribution, the one below a uniform distribution"
        if self.normalizedgaussiansigmad==False:
            return GaussianSigmad(x=dl, mu=self.dl_gw, sigmaprop=self.sigmaprop)
        else:
            return GaussianSigmad(x=dl, mu=self.dl_gw, sigmaprop=self.sigmaprop)/self.gaussiansigmadnorm
    
    def px_dl_direct_uniform(self,dl):
        #print("Using px uniform!")
        return UniformSigmad(dl=dl, dl_gw=self.dl_gw, sigma=(1./(2.5*self.sigmaprop))) #BEWARE HARDCODING 

    def px_ra(self, ra):
        """
        Returns a probability for a given ra
        from the interpolated function.
        """
        return splev(ra, self.tempra, ext=3)
    
    def px_dec(self, dec):
        """
        Returns a probability for a given dec
        from the interpolated function.
        """
        return splev(dec, self.tempdec, ext=3)
    
    def px_radec_independent(self, ra, dec):
        """
        Returns a probability for a given ra dec
        from the interpolated functions, assuming the correlation is zero
        """
        if self.directradec!=False:
            #print("Using direct radec!")
            return self.px_ra_direct(ra)*self.px_dec_direct(dec)
        else:
            return self.px_ra(ra)*self.px_dec(dec)
    
    def px_ra_direct(self, ra):
        return gaussian(ra, mu=self.ra_gw, sig=self.sigmaradec)

    def px_dec_direct(self, dec):
        return gaussian(dec, mu=self.dec_gw, sig=self.sigmaradec)
    
    def ps_z(self, z):
        if self.rate == 'constant':
            return 1.0
        if self.rate == 'evolving':
            return (1.0+z)**self.Lambda
        
    def likelihood(self,H0,complete=False,population=False, dimensions=3, renorminglike=False):
        """
        The likelihood for a single event
        This corresponds to Eq 3 (statistical) or Eq 6 (counterpart) in the doc, depending on parameter choices.
        
        Parameters
        ----------
        H0 : float or array_like
            Hubble constant value(s) in kms-1Mpc-1
        complete : bool, optional
            Is the galaxy catalog complete to all relevant distances/redshifts? (default=False)
           
        Returns
        -------
        float or array_like
            p(x|H0,D)
        """        
        if population==True:
            pxG = self.px_H0_empty(H0)
            self.pDG = self.pD_H0_empty(H0)
            likelihood = pxG/self.pDG

        else:

            if dimensions==1:
                if self.smoothedcatalog==True:
                    pxG = self.px_inside_1D_smoothed(H0, direct=self.directpx)
                else:    
                    pxG = self.px_inside_1D(H0, direct=self.directpx)
            elif dimensions==3:
                pxG = self.px_inside_3D(H0, direct=self.directpx)

            if self.pDG==None:
                if self.precomputedinsidepdet==True:
                    if self.saveprecomputes==True:
                        self.pDG = self.Read_precomputed_pdet_or_pG(self.insidepdet)
                    else:
                        self.pDG=self.insidepdet
                elif self.pdettype==True:
                    self.pDG = self.pD_inside_gwcosmo(H0)
                elif self.pdettype=='cube':
                    self.pDG = self.pD_H0cube(H0)
                elif self.pdettype=="theoretic":
                    self.pDG = self.pD_inside_theoretic(H0)
                elif self.pdettype=="ones":
                    self.pDG=np.ones(len(pxG))
                else:
                    print("Pdet chosen not supported")
            
            if complete==True:
                likelihood = pxG/self.pDG # Eq 3 with p(G|H0,D)=1 and p(bar{G}|H0,D)=0
            else:
                if self.pGD==None:
                    if self.precomputedpGpdet==True:
                        if self.saveprecomputes==True:
                            self.pGD = self.Read_precomputed_pdet_or_pG(self.pGpdet)
                        else:
                            self.pGD = self.pGpdet
                    else:
                        self.pGD = self.pG(H0)
                    

                if self.pnGD==None:
                    self.pnGD = self.pnG(H0)

                #if dimensions==3:    
                #    pxnG = self.px_outside(H0)
                #elif dimensions==1:
                pxnG = self.px_outside(H0)

                if self.pDnG==None:
                    if self.precomputedoutsidepdet==True:
                        if self.saveprecomputes==True:
                            self.pDnG=self.Read_precomputed_pdet_or_pG(self.outsidepdet)
                        else:
                            self.pDnG=self.outsidepdet
                    else:
                        self.pDnG = self.pD_outside(H0)
                

                likeinside=pxG/self.pDG
                likeoutside=pxnG/self.pDnG

                if renorminglike==False:
                    likelihood = self.pGD*likeinside + self.pnGD*likeoutside # Eq 3
                elif renorminglike==True:
                    #likelihood = self.pGD*(likeinside)/np.trapz(likeinside, H0) + self.pnGD*(likeoutside)/np.trapz(likeoutside, H0)
                    pxGnorm=pxG/np.trapz(pxG, H0)
                    pxnGnorm=pxnG/np.trapz(pxnG, H0)
                    likelihood = self.pGD*pxGnorm/self.pDG + self.pnGD*pxnGnorm/self.pDnG
                
        if (complete==True) or (population==True):
            self.pGD = np.ones(len(H0))
            self.pnGD = np.zeros(len(H0))
            pxnG = np.zeros(len(H0))
            self.pDnG = np.ones(len(H0))

        """pxGnorm, pxGnormed = GetNormalizationAndNormalized(H0, pxG)
        pGnorm, pGnormed = GetNormalizationAndNormalized(H0, self.pGD)
        pDGnorm, pDGnormed = GetNormalizationAndNormalized(H0, self.pDG)
        pxnGnorm, pxnGnormed = GetNormalizationAndNormalized(H0, pxnG)
        pDnGnorm, pDnGnormed = GetNormalizationAndNormalized(H0, self.pDnG)
        likelihoodnorm, likelihoodnormed = GetNormalizationAndNormalized(H0, likelihood)
        print("pxGnorm, pxGnormed", pxGnorm)#, pxGnormed)
        print("pDGnorm, pDGnormed", pDGnorm)#, pDGnormed)
        print("pGnorm, pGnormed", pGnorm)#, pGnormed)
        print("pxnGnorm, pxnGnormed", pxnGnorm)#, pxnGnormed)
        print("pDnGnorm, pDnGnormed", pDnGnorm)#, pDnGnormed)
        print("likenorm, likenormed", likelihoodnorm)#, likelihoodnormed)"""
        #print(min(likelihood),min(pxG),min(self.pDG),min(self.pGD), min(pxnG),min(self.pDnG),min(self.pnGD))
        return likelihood,pxG,self.pDG,self.pGD, pxnG,self.pDnG,self.pnGD
        
    def px_inside_3D(self, H0, direct=False):
        num = np.zeros(len(H0))
        
        zs = self.allz
        ms = self.allm
        ras = self.allra
        decs = self.alldec
        #tempsky=np.ones(len(zs))

        if self.weighted:
            mlim = np.percentile(np.sort(ms),0.01) # more draws for galaxies in brightest 0.01 percent
        else:
            mlim = 1.0
        
        bar = progressbar.ProgressBar()
        print("Calculating p(x|H0,G) in 3D")
        # loop over galaxies
        
        initiated=0 #silly variable to check if I have initiated the 
        #numinners=[]
        for i in bar(range(len(zs))):

            if ras[i]>self.rafastmin and ras[i]<self.rafastmax and decs[i]>self.decfastmin and decs[i]<self.decfastmax:
                #print("approved", ras[i], decs[i])
                numinner=np.zeros(len(H0))
                if self.weightedcatalog==True:
                    weight = self.allcatalogweights[i]
                elif self.weighted=="luminosity":
                    weight = L_mdl(ms[i], self.cosmo.dl_zH0(zs[i], H0))
                elif self.weighted=="custom":
                    zindex=ztoZshell(zs[i], self.weights[0])
                    weight = self.weights[1][zindex]
                elif self.weighted=="custom_interp":
                    weight = splev(zs[i], self.weights, ext=3)
                else:
                    weight = 1.0
                
                if direct==False:
                    tempdist = self.px_dl(self.cosmo.dl_zH0(zs[i], H0))#/self.cosmo.dl_zH0(zs[i], H0)**2 # remove dl^2 prior from samples but I'm creating the likelihood myself so probably not needed
                else:
                    if self.dl_distr=="gaussian-sigmad":
                        tempdist = self.px_dl_direct_gaussiansigmad(self.cosmo.dl_zH0(zs[i], H0))#/self.cosmo.dl_zH0(zs[i], H0)**2 # remove dl^2 prior from samples but I'm creating the likelihood myself so probably not needed
                    if self.dl_distr=="uniform":
                        tempdist=[]
                        for H0ele in H0:
                            tempdistele = self.px_dl_direct_uniform(self.cosmo.dl_zH0(zs[i], H0ele))
                            tempdist.append(tempdistele)
                        tempdist=np.array(tempdist)
                
                numinner = tempdist*self.tempsky[i]*weight*self.ps_z(zs[i])#tempsky contains the rad and dec infos

                if self.summationtype=="None":
                    num += numinner#sum contributions #this line for normal sum
                elif self.summationtype=="logsumexp":
                    if all(numinner>0.): #TODO TOCHECK I'm imposing that every single numinner (so every one for each possible value of H0) is above zero. Maybe too much
                        if initiated==0:
                            max=numinner
                            logmax=np.log(max)
                            logsumarg=np.ones(len(logmax))

                            initiated=1#change so that it's initiated only once ofc
                        else:
                            logsumarg+=np.exp(np.log(numinner)-logmax)
                            for j in np.arange(len(H0)):
                                if numinner[j]>max[j]:
                                    logsumarg[j]=logsumarg[j]*max[j]/numinner[j]
                                    max[j]=numinner[j]
                                    logmax[j]=np.log(max[j])
            
                    #if numinner!=0.: 
                    #numinners.append(numinner)
        

        if self.summationtype=="logsumexp":
            num=np.exp(logmax+np.log(logsumarg))
        #print(num)
        numnorm = num/(self.nGal * self.avaragetempsky)   #normalize by number of gal and avarage tempsky angle weight

        #print(numnorm)
        return numnorm

    def px_inside_1D(self, H0, direct=False):
        """
        Returns p(x|H0,G) for given values of H0.
        The likelihood of the GW data given H0 and conditioned on
        the source being inside the galaxy catalog. 1D version

        Parameters
        ----------
        H0 : float or array_like
            Hubble constant value(s) in kms-1Mpc-1

        Returns
        -------
        float or array_like
            p(x|H0,G)
        """
        num = np.zeros(len(H0))
        #print("Using 1D!")

        """
        ind = np.argwhere(tempsky >= 0.)
        tempsky = tempsky[ind].flatten()
        zs = self.allz[ind].flatten()
        ras = self.allra[ind].flatten()
        decs = self.alldec[ind].flatten()
        ms = self.allm[ind].flatten()
        """
        zs = self.allz
        ras = self.allra
        decs = self.alldec
        ms = self.allm
        
        bar = progressbar.ProgressBar()
        print("Calculating p(x|H0,G)")
        # loop over galaxies
        for i in bar(range(len(zs))):
            
            if self.weightedcatalog==True:
                weight = self.allcatalogweights[i]
            elif self.weighted=="luminosity":
                weight = L_mdl(ms[i], self.cosmo.dl_zH0(zs[i], H0))
            elif self.weighted=="custom":
                zindex=ztoZshell(zs[i], self.weights[0])
                weight = self.weights[1][zindex]
            elif self.weighted=="custom_interp":
                weight = splev(zs[i], self.weights, ext=3)
            
            else:
                weight = 1.0

            if direct==False:
                tempdist = self.px_dl(self.cosmo.dl_zH0(zs[i], H0))#/self.cosmo.dl_zH0(zs[i], H0)**2 # remove dl^2 prior from samples but I'm creating the likelihood myself so probably not needed
            else:
                if self.dl_distr=="gaussian-sigmad":
                    tempdist = self.px_dl_direct_gaussiansigmad(self.cosmo.dl_zH0(zs[i], H0))#/self.cosmo.dl_zH0(zs[i], H0)**2 # remove dl^2 prior from samples but I'm creating the likelihood myself so probably not needed
                if self.dl_distr=="uniform":
                    tempdist=[]
                    for H0ele in H0:
                        tempdistele = self.px_dl_direct_uniform(self.cosmo.dl_zH0(zs[i], H0ele))
                        tempdist.append(tempdistele)
                    tempdist=np.array(tempdist)
                
            numinner = tempdist*weight*self.ps_z(zs[i])
            num += numinner
        

        numnorm = num/self.nGal
            
        return numnorm

    def px_inside_1D_smoothed(self, H0, direct=False):
        """
        Returns p(x|H0,G) for given values of H0.
        The likelihood of the GW data given H0 and conditioned on
        the source being inside the galaxy catalog. 1D version

        Parameters
        ----------
        H0 : float or array_like
            Hubble constant value(s) in kms-1Mpc-1

        Returns
        -------
        float or array_like
            p(x|H0,G)
        """
        num = np.zeros(len(H0))
        #print("Using 1D!")

        zs = self.allz
        ras = self.allra
        decs = self.alldec
        ms = self.allm
        
        bar = progressbar.ProgressBar()

        print("Calculating p(x|H0,G)")
        # loop over galaxies
        numH0s=[]

        for H0_iter in H0:
            if self.dl_distr=="gaussian-sigmad":
                tempdist = self.px_dl_direct_gaussiansigmad(self.cosmo.dl_zH0(self.z_array, H0_iter)) * self.z_vals#/self.cosmo.dl_zH0(zs[i], H0)**2 # remove dl^2 prior from samples but I'm creating the likelihood myself so probably not needed
            
                
                numH0 = np.trapz(tempdist, self.z_array)/np.trapz(self.z_vals, self.z_array)
                numH0s.append(numH0)
        
        

        numnorm = np.array(numH0s)
            
        return numnorm

    def Read_precomputed_pdet_or_pG(self, pdetfiletoload):
        #print("Reading pdet or PG from ", pdetfiletoload)
        pDG=np.loadtxt(pdetfiletoload)
        return pDG
    
    def pD_inside_theoretic(self,H0):
        "This computes the pdet theoretically from eq 22 of hitchikers"
        den = np.zeros(len(H0))
        zs = self.allz
        ras = self.allra
        decs = self.alldec
        ms = self.allm
        if self.weighted:
            mlim = np.percentile(np.sort(self.allm),0.01) # more draws for galaxies in brightest 0.01 percent
        else:
            mlim = 1.0
            
        bar = progressbar.ProgressBar()
        print("Calculating p(D|H0,G) using theoretic Heaviside step function (Eq 22)")
        # loop over galaxies
        for i in bar(range(len(zs))):
            # loop over random draws from galaxies
            if self.weightedcatalog==True:
                weight = self.allcatalogweights[i]
            elif self.weighted=="luminosity":
                weight = L_mdl(ms[i], self.cosmo.dl_zH0(zs[i], H0))
            elif self.weighted=="custom":
                zindex=ztoZshell(zs[i], self.weights[0])
                weight = self.weights[1][zindex]
            elif self.weighted=="custom_interp":
                weight = splev(zs[i], self.weights, ext=3)
            else:
                weight = 1.0
            if self.dl_distr=="gaussian-sigmad":
                prob = self.pD_Heaviside_theoretic(zs[i],H0).flatten()
            else:
                print("PD NOT PRECOMPUTED NOT IMPLEMENTED FOR ANY DL_DISTR APART FROM GAUSSIAN-SIGMAD")
            deninner = prob*weight*self.ps_z(zs[i])
            den += deninner

        self.pDG = den/self.nGal

        return self.pDG
    
    def pD_H0cube(self, H0):
        # super simple pdet that return H0**3. This should be the case for an empty catalog
        print("Calculating p(D|H0,G) with a simple H0**3")
        return H0**3

    def pD_Heaviside_theoretic(self, z, H0):
        dl=self.cosmo.dl_zH0(z, H0)
        erfvariable=(dl-self.dl_det)/(np.sqrt(2)*self.sigmaprop*dl) #should be this one instead of next one. Jon says no sqrt(2), shouldn't change regardless cause it's just overall factor
        #erfvariable=(dl-dl_det)/(sigmaprop*dl) 
        erfvariable*=-1 #There might be an inconsistency between hitchhiker and scipy in definition of error function, the minus sign produces something much closer to H0**3/gwcosmo
        return 0.5 * (1 + special.erf(erfvariable))
    
    def pG(self,H0):
        
        # Warning - this integral misbehaves for small values of H0 (<25 kms-1Mpc-1).  TODO: fix this.
        num = np.zeros(len(H0)) 
        den = np.zeros(len(H0))
        bar = progressbar.ProgressBar()
        print("Calculating p(G|H0,D)")
        for i in bar(range(len(H0))):
            def I(M,z):
                if self.pdettype==True:
                    temp = SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)*self.pdet.pD_zH0_eval(z,H0[i])*self.zprior(z)*self.ps_z(z)
                elif self.pdettype=='cube':
                    temp = SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)*H0[i]**3*self.zprior(z)*self.ps_z(z)
                elif self.pdettype=="theoretic":
                    temp = SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)*self.pD_Heaviside_theoretic(z,H0[i])*self.zprior(z)*self.ps_z(z)
                elif self.pdettype=="ones":
                    temp = SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)*self.zprior(z)*self.ps_z(z)
                if self.weighted:
                    return temp*L_M(M)
                else:
                    return temp
            # Mmin and Mmax currently corresponding to 10L* and 0.001L* respectively, to correspond with MDC
            # Will want to change in future.
            # TODO: test how sensitive this result is to changing Mmin and Mmax.
            Mmin = M_Mobs(H0[i],self.Mobs_min)
            Mmax = M_Mobs(H0[i],self.Mobs_max)

            num[i] = dblquad(I,0,self.zmax,lambda x: Mmin,lambda x: min(max(M_mdl(self.mth,self.cosmo.dl_zH0(x,H0[i])),Mmin),Mmax),epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0]
            #num[i] = dblquad(I,0,self.zcut,lambda x: Mmin,lambda x: M_mdl(self.mth,self.cosmo.dl_zH0(x,H0[i])),epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0]
            den[i] = dblquad(I,0,self.zmax,lambda x: Mmin,lambda x: Mmax,epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0]

        self.pGD = num/den
        #print("PG NUM : ", num)
        #print("PG DEN : ", den)
        return self.pGD    

    def pnG(self,H0):
        
        if all(self.pGD)==None:
            self.pGD = self.pG(H0)
        self.pnGD = 1.0 - self.pGD
        return self.pnGD
          
    def px_outside(self,H0):
        
        num = np.zeros(len(H0))

        bar = progressbar.ProgressBar()
        print("Calculating p(x|H0,bar{G})")
        redshiftnorm=self.zmax**3/3

        for i in bar(range(len(H0))):
            Mmin = M_Mobs(H0[i],self.Mobs_min)
            Mmax = M_Mobs(H0[i],self.Mobs_max)
            """if self.modificationskypatch==False: #there are two versions of this, I think one before skypatch modification in gwcosmo (not mine) and one after. Probably the one after is the correct one
                def Inum(M,z):
                    temp = self.px_dl(self.cosmo.dl_zH0(z,H0[i]))*self.zprior(z)*SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)*self.ps_z(z)/self.cosmo.dl_zH0(z,H0[i])**2 # remove dl^2 prior from samples
                    if self.weighted:
                        return temp*L_M(M)
                    else:
                        return temp

                
                if i==0 or i==(len(H0)-1):
                    print("H0, Mmdlz0, Mmdlzcut, Mmax ", H0[i], M_mdl(self.mth,self.cosmo.dl_zH0(0,H0[i])), M_mdl(self.mth,self.cosmo.dl_zH0(self.zcut,H0[i])), Mmax)
            
                if allsky == True:
                    distnum[i] = dblquad(Inum,0.0,self.zcut, lambda x: M_mdl(self.mth,self.cosmo.dl_zH0(x,H0[i])), lambda x: Mmax,epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0] \
                            + dblquad(Inum,self.zcut,self.zmax, lambda x: Mmin, lambda x: Mmax,epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0]
                else:
                    distnum[i] = dblquad(Inum,0.0,self.zmax,lambda x: Mmin,lambda x: Mmax,epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0]
            if self.modificationskypatch==True:"""

            def Itonorm(z,M):
                temp=SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)*self.zprior(z)*self.ps_z(z)
                return temp
            totnorm=dblquad(Itonorm,Mmin,Mmax,lambda x: z_dlH0(dl_mM(self.mth,x),H0[i],linear=self.linear),lambda x: self.zmax,epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0] #check whether to use this style of integral or the one in modificationskypatch=False/pG
    
                
            def Inum(z,M):
                if self.directpx==False:
                    temp = self.px_dl(self.cosmo.dl_zH0(z,H0[i]))*self.zprior(z)*SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)*self.ps_z(z)#/self.cosmo.dl_zH0(z,H0[i])**2 # remove dl^2 prior from samples
                elif self.directpx==True:
                    temp = self.px_dl_direct_gaussiansigmad(self.cosmo.dl_zH0(z, H0[i]))*self.zprior(z)*SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)*self.ps_z(z)#/self.cosmo.dl_zH0(z,H0[i])**2 # remove dl^2 prior from samples
                

                if self.weighted:
                    #if self.zweight_outcat:
                    #    return temp*L_M(M)*self.Zweightfunc(z)
                    #else:
                    return temp*L_M(M)
                else:
                    #if self.zweight_outcat:
                    #    return temp*self.Zweightfunc(z)
                    #else:
                    return temp
            #if i==0 or i==(len(H0)-1):
            #    print("H0, Mmdlz0, Mmdlzcut, Mmax ", H0[i], M_mdl(self.mth,self.cosmo.dl_zH0(0,H0[i])), M_mdl(self.mth,self.cosmo.dl_zH0(self.zcut,H0[i])), Mmax)
            #    print("H0, zdlMmin, zdlMmax, zmax ", H0[i], z_dlH0(dl_mM(self.mth,Mmin),H0[i],linear=self.linear), z_dlH0(dl_mM(self.mth,Mmax),H0[i],linear=self.linear), self.zmax)
            SchechterNorm=quad(SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha), Mmin, Mmax)[0]
            #num[i] = dblquad(Inum,Mmin,Mmax,lambda x: min(z_dlH0(dl_mM(self.mth,x),H0[i],linear=self.linear), self.zmax),lambda x: self.zmax,epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0]#/(SchechterNorm*redshiftnorm)
            num[i] = dblquad(Inum,Mmin,Mmax,lambda x: min(z_dlH0(dl_mM(self.mth,x),H0[i],linear=self.linear), self.zmax),lambda x: self.zmax,epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0]/totnorm#/(SchechterNorm*redshiftnorm)
    
        return num

    def pD_outside(self,H0):
        
        # TODO: same fixes as for pG_H0D 
        den = np.zeros(len(H0))
        
        def skynorm(dec,ra):
            return np.cos(dec)
                
        norm = dblquad(skynorm,self.ra_min,self.ra_max,lambda x: self.dec_min,lambda x: self.dec_max,epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0]/(4.*np.pi)
        #print("Sky stuff: ramin, ramax, decmin, decmax, norm : ", self.ra_min, self.ra_max, self.dec_min, self.dec_max, norm)
        bar = progressbar.ProgressBar()
        print("Calculating p(D|H0,bar{G})")
        for i in bar(range(len(H0))):
            Mmin = M_Mobs(H0[i],self.Mobs_min)
            Mmax = M_Mobs(H0[i],self.Mobs_max)
            
            """if self.modificationskypatch==False:#same as
                def I(M,z):
                    if self.basic:
                        temp = SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)*self.pdet.pD_dl_eval_basic(self.cosmo.dl_zH0(z,H0[i]))*self.zprior(z)*self.ps_z(z)
                    else:
                        temp = SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)*self.pdet.pD_zH0_eval(z,H0[i])*self.zprior(z)*self.ps_z(z)
                    if self.weighted:
                        return temp*L_M(M)
                    else:
                        return temp

                if allsky == True:
                    den[i] = dblquad(I,0.0,self.zcut, lambda x: M_mdl(self.mth,self.cosmo.dl_zH0(x,H0[i])), lambda x: Mmax,epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0] \
                        + dblquad(I,self.zcut,self.zmax, lambda x: Mmin, lambda x: Mmax,epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0]
                else:
                    den[i] = dblquad(I,0.0,self.zmax,lambda x: Mmin,lambda x: Mmax,epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0]
            if self.modificationskypatch==True:"""
            def Iden(z,M):
                if self.pdettype==True:
                    temp = SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)*self.pdet.pD_zH0_eval(z,H0[i])*self.zprior(z)*self.ps_z(z)
                elif self.pdettype=='cube':
                    temp = SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)*H0[i]**3*self.zprior(z)*self.ps_z(z)
                elif self.pdettype=="theoretic":
                    temp = SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)*self.pD_Heaviside_theoretic(z,H0[i])*self.zprior(z)*self.ps_z(z)
                elif self.pdettype=="ones":
                    temp = SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)*self.zprior(z)*self.ps_z(z)
            
                if self.weighted:
                    return temp*L_M(M)
                else:
                    return temp
            den[i] = dblquad(Iden,Mmin,Mmax,lambda x: z_dlH0(dl_mM(self.mth,x),H0[i],linear=self.linear),lambda x: self.zmax,epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0] #check whether to use this style of integral or the one in modificationskypatch=False/pG
        return pDnG

class likelihoodJon(object):
    #This is a function that computes the likelihood "Jon's way". Instead of splitting in in catalog and out of catalog components, they are combined in simply numerator vs denominator. These two methods should be equivalent, but maybe in this one there isn't a normalization problem
    def px_dl_direct_gaussiansigmad(self, dl):
        #print("using direct gaussian!")
        "Computing px_dl directly, assuming sigmaprop is a known parameter and dl_gw is the measurement from the event. This function assumes a gaussian sigma d distribution, the one below a uniform distribution"
        return GaussianSigmad(x=dl, mu=self.dl_gw, sigmaprop=self.sigmaprop)
    
    def Read_precomputed_pdet_or_pG(self, pdetfiletoload):
        print("Reading pdet from ", pdetfiletoload)
        pDG=np.loadtxt(pdetfiletoload)
        return pDG
    
    def px_radec_independent(self, ra, dec):
        """
        Returns a probability for a given ra dec, given an observed ra and dec
        """
        return self.px_ra_direct(ra)*self.px_dec_direct(dec)
    
    def px_ra_direct(self, ra):
        return gaussian(ra, mu=self.ra_gw, sig=self.sigmaradec)

    def px_dec_direct(self, dec):
        return gaussian(dec, mu=self.dec_gw, sig=self.sigmaradec)
    
    def __init__(self, EventNumber, galaxy_catalog, GW_data, dl_det, sigmaprop, sigmaradec, assumed_band='r', linear=True, Omega_m=0.25, zmax=0.1, precomputedpdets=False, insidepdetpath=None, outsidepdetpath=None, completeness=None, emptycatalogrun=False):
        #self.precomputedinsidepdet=precomputedinsidepdet
        #self.precomputedoutsidepdet=precomputedoutsidepdet
        #self.precomputedpGpdet=precomputedpGpdet
        #self.insidepdet = insidepdet
        #self.outsidepdet = outsidepdet
        #self.pGpdet = pGpdet
        self.Omega_m = Omega_m
        self.linear = linear
        self.assumed_band = assumed_band
        sp = SchechterParams(self.assumed_band)
        self.alpha = sp.alpha
        self.Mstar_obs = sp.Mstar
        self.Mobs_min = sp.Mmin
        self.Mobs_max = sp.Mmax
        #print("assumed band: ", assumed_band, " params ", self.alpha, self.Mstar_obs, self.Mobs_min, self.Mobs_max)
        self.dl_det=dl_det
        self.sigmaprop=sigmaprop
        self.sigmaradec=sigmaradec
        self.emptycatalogrun=emptycatalogrun
        
        if self.emptycatalogrun==False:
            self.allz = galaxy_catalog.z
            self.allra = galaxy_catalog.ra
            self.alldec = galaxy_catalog.dec
            self.allm = galaxy_catalog.m
            self.mth = galaxy_catalog.mth(type="max")
            self.nGal = len(self.allz)
        
        if GW_data is not None:
            samps=h5py.File(GW_data, 'r')
            self.dl_gw=samps[EventNumber]["dl_gw"][()]
            self.ra_gw=samps[EventNumber]["ra_gw"][()]
            self.dec_gw=samps[EventNumber]["dec_gw"][()]
            if self.emptycatalogrun==False:
                self.tempsky = self.px_radec_independent(self.allra, self.alldec)

        self.precomputedpdets=precomputedpdets
        self.insidepdetpath=insidepdetpath
        self.outsidepdetpath=outsidepdetpath
        self.completeness=completeness
                
        self.pDG = None
        self.pGD = None
        self.pnGD = None
        self.pDnG = None

        # Note that zmax is an artificial limit that
        # should be well above any redshift value that could
        # impact the results for the considered H0 values.

        

        self.zmax = zmax #see zcut for comments on this parameter. TODO: probably take it out
        
        self.zprior = redshift_prior(Omega_m=self.Omega_m, linear=self.linear)
        self.cosmo = fast_cosmology(Omega_m=self.Omega_m, linear=self.linear)

    def likelihoodJon(self, H0, dimensions=3):
        #likelihood Jon's way
        if self.emptycatalogrun==False:
        
            if self.completeness is not None:
                self.pG=np.ones(len(H0))*self.completeness
            else:
                self.pG=self.pGJon(H0)
                self.completeness=self.pG[0]
            self.pnG=np.ones(len(H0))-self.pG
            
            if self.precomputedpdets==True:
                self.pDG=self.Read_precomputed_pdet_or_pG(self.insidepdetpath)
                self.pDnG=self.Read_precomputed_pdet_or_pG(self.outsidepdetpath)
                #self.pDnG=self.pdetout(H0)

            self.den=self.pDG*self.pG + self.pDnG#Careful about this term, not sure
        
            if dimensions==1:
                self.pxG=self.pxG_1D(H0)
            else:
                self.pxG=self.pxG_3D(H0)
    
            self.pxnonG=self.pxnG(H0)
            self.num=self.pxG+self.pxnonG
        
        else:
            self.pG=np.zeros(len(H0))
            self.pnG=np.ones(len(H0))
            self.pDG=np.ones(len(H0))
            self.pxG=np.zeros(len(H0))

            if dimensions==1:
                self.pxnonG=self.pxnG(H0)
            self.pDnG=self.pdetout(H0)

            self.num=self.pxnonG
            self.den=self.pDnG
            print("Num ", self.num)
            print("Den ", self.den)
        

        likelihood=self.num/self.den
        print("likelihood ", likelihood)
        

        return likelihood, self.pxG, self.pDG, self.pG, self.pxnonG, self.pDnG, self.pnG

    def pGJon(self,H0):
        
        # Warning - this integral misbehaves for small values of H0 (<25 kms-1Mpc-1).  TODO: fix this.
        num = np.zeros(len(H0)) 
        den = np.zeros(len(H0))
        bar = progressbar.ProgressBar()
        print("Calculating p(G|H0,D)")
        for i in bar(range(len(H0))):
            def I(M,z):
                return SchechterMagFunction(H0=H0[i],Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)*self.zprior(z)

            Mmin = M_Mobs(H0[i],self.Mobs_min)
            Mmax = M_Mobs(H0[i],self.Mobs_max)
            
            num[i] = dblquad(I, 0, self.zmax, lambda x : Mmin, lambda x: min(max(M_mdl(self.mth,self.cosmo.dl_zH0(x,H0[i])),Mmin),Mmax),epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0]
            den[i] = dblquad(I, 0, self.zmax, lambda x: Mmin, lambda x: Mmax, epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0]
            
        self.pGD = num/den
        print(self.pGD)
        return self.pGD    
    
    def fofz(self, z):
        # fraction of observed galaxies at redshift z

        H0ass=70. #It actually does not depend on H0, but easy to just put as a placeholder and not modify the basic functions
        Mmin = M_Mobs(H0ass,self.Mobs_min)
        Mmax = M_Mobs(H0ass,self.Mobs_max)

        def I(M):
            return SchechterMagFunction(H0=H0ass,Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)

        num = quad(I,Mmin, min(max(M_mdl(self.mth,self.cosmo.dl_zH0(z,H0ass)),Mmin),Mmax),epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0]
        den = quad(I, Mmin, Mmax, epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0]
        
        return num/den

    def pxG_1D(self, H0):
        #this computes the first half of equation 5
        num = np.zeros(len(H0))
        bar = progressbar.ProgressBar()
        print("Calculating p(x|H0,G)")
        # loop over galaxies
        for i in bar(range(len(self.allz))):

            tempdist = self.px_dl_direct_gaussiansigmad(self.cosmo.dl_zH0(self.allz[i], H0))#/self.cosmo.dl_zH0(zs[i], H0)**2 # remove dl^2 prior from samples but I'm creating the likelihood myself so probably not needed
                
            numinner = tempdist
            num += numinner
        
        numnorm=num*self.completeness/self.nGal
        
        return numnorm
    
    def pxG_3D(self, H0):
        #this computes the first half of equation 5
        num = np.zeros(len(H0))
        bar = progressbar.ProgressBar()
        print("Calculating p(x|H0,G)")
        # loop over galaxies
        for i in bar(range(len(self.allz))):

            tempdist = self.px_dl_direct_gaussiansigmad(self.cosmo.dl_zH0(self.allz[i], H0))#/self.cosmo.dl_zH0(zs[i], H0)**2 # remove dl^2 prior from samples but I'm creating the likelihood myself so probably not needed
                
            numinner = tempdist*self.tempsky[i]
            num += numinner
        
        numnorm=num*self.completeness/self.nGal
        
        return numnorm
    
    def Comov_volume(self, z):
        #volume element within z. Note that eventual normalization factor don't matter, as this only appear in likelihood through dVc/dz / Vc(z)
        if self.linear==True:
            return z**3
        else:
            return volume_z(z, Omega_m=Omega_m)
    
    def Derivative_comov_volume(self, z):
        #derivative of volume element within z. Note that eventual normalization factor don't matter, as this only appear in likelihood through dVc/dz / Vc(z)
        if self.linear==True:
            return 3*z**2
        else:
            return self.zprior(z)

    def pzout(self, z):
        #this computes the out of catalog part of equation  9
        if self.emptycatalogrun==False:
            return (1-self.fofz(z))*self.Derivative_comov_volume(z)/self.Comov_volume(self.zmax)
        else:
            return self.Derivative_comov_volume(z)/self.Comov_volume(self.zmax)
        
    def pxnG(self, H0):
        num = np.zeros(len(H0))
        #this computes second half of equation 5
        print("Calculating px(H0|nG)")
        bar = progressbar.ProgressBar()
        for i in bar(range(len(H0))):
            def Integrand(z):
                return self.px_dl_direct_gaussiansigmad(self.cosmo.dl_zH0(z, H0[i]))*self.pzout(z)
            num[i] = quad(Integrand, 0, self.zmax, epsabs=epsabsforFunctionsScratch,epsrel=1.49e-9)[0]
        return num
    
    def pD_Heaviside_theoretic(self, z, H0):
        dl=self.cosmo.dl_zH0(z, H0)
        erfvariable=(dl-self.dl_det)/(np.sqrt(2)*self.sigmaprop*dl) #should be this one instead of next one. Jon says no sqrt(2), shouldn't change regardless cause it's just overall factor
        #erfvariable=(dl-dl_det)/(sigmaprop*dl) 
        erfvariable*=-1 #There might be an inconsistency between hitchhiker and scipy in definition of error function, the minus sign produces something much closer to H0**3/gwcosmo
        return 0.5 * (1 + special.erf(erfvariable))
    
    def pdetout(self, H0):
        den = np.zeros(len(H0))
        bar = progressbar.ProgressBar()
        print("Calculating p(D|H0,bar{G})")
        for i in bar(range(len(H0))):
            def Integrand(z):
                return self.pzout(z)*self.pD_Heaviside_theoretic(z,H0[i])
            den[i] = quad(Integrand, 0, self.zmax, epsabs=epsabsforFunctionsScratch,epsrel=1.49e-9)[0]
        return den


    
    