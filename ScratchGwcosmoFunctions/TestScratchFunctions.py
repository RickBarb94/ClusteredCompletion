import numpy as np
import matplotlib.pyplot as plt
#from FunctionsSimulateData import ChooseHost, ExtractObservedGWEvent, RejectSampleLikelihood, GaussianSigmad
from utilities.standard_cosmology import dl_zH0, z_dlH0, fast_cosmology, redshift_prior
from FunctionsCatalog import galaxy, galaxyCatalog, galaxyweighted, galaxyCatalogweighted
import scipy.constants
import h5py
import sys
import scipy.optimize
from scipy import special

def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def pD_Heaviside_theoretic(z, H0, dl_det, sigmaprop):
        dl=cosmo.dl_zH0(z, H0)
        erfvariable=(dl-dl_det)/(np.sqrt(2)*sigmaprop*dl) #should be this one instead of next one. Jon says no sqrt(2), shouldn't change regardless cause it's just overall factor
        #erfvariable=(dl-dl_det)/(sigmaprop*dl) 
        erfvariable*=-1 #There might be an inconsistency between hitchhiker and scipy in definition of error function, the minus sign produces something much closer to H0**3/gwcosmo
        return 0.5 * (1 + special.erf(erfvariable))

def ps_z(z):
        if rate == 'constant':
            return 1.0
        if rate == 'evolving':
            return (1.0+z)**Lambda
        
def pD_inside_theoretic(H0, Catalog, dl_det, sigmaprop):
        "This computes the pdet theoretically from eq 22 of hitchikers"

        den = np.zeros(len(H0))
        zs = Catalog.z
        ras = Catalog.ra
        decs = Catalog.dec
        ms = Catalog.m
            
        
        for i in range(len(zs)):
            # loop over random draws from galaxies
            weight = 1.0
            prob = pD_Heaviside_theoretic(zs[i],H0, dl_det, sigmaprop).flatten()
            deninner = prob*weight*ps_z(zs[i])
            den += deninner

        pDG = den/len(zs)

        return pDG

def GetDetectedEventFraction(H0, Catalog, dl_det, sigmaprop, angle_cut_fraction, z_cut_fraction):
    ramin=0
    ramax=np.pi/2
    decmin=0.
    decmax=np.pi/2
    #edge cuts
    racut=np.pi/2*angle_cut_fraction#don't get this close to the ra edges
    ra_cut_min=racut
    ra_cut_max=np.pi/2-racut
    deccut=np.pi/2*angle_cut_fraction #don't get this close to the dec edges
    dec_cut_min=racut
    dec_cut_max=np.pi/2-racut

    zs = Catalog.z
    ras = Catalog.ra
    decs = Catalog.dec
    ms = Catalog.m
    
    maxzcat=0.1                                                             ###############THIS if using scaled, otherwise 0.1#########################
    zcut=maxzcat*z_cut_fraction#don't get this close to the z edges
    z_cut_min=0
    z_cut_max=maxzcat-zcut


    
    N_events=250
    EventFractions=[]
    for H0_true in H0:
        print(H0_true)
        i=0
        totcount=0
        observed=[]
        while i<N_events:
            #print(i)
            totcount+=1
            hostind=ChooseHost(ras, decs, zs, ms, ra_cut_min, ra_cut_max, dec_cut_min, dec_cut_max, z_cut_min, z_cut_max)
            dl_host=dl_zH0(zs[hostind], H0=H0_true, Omega_m=Omega_m, linear=linear_data_gen)#Introduced after linear Implementation
                
            ra_gw, dec_gw, dl_gw = ExtractObservedGWEvent(ras[hostind], decs[hostind], dl_host, sigmaprop*dl_host, cov_radec_ass)#this or the next one?#BeforeLinearImplementationVersion
            if dl_gw<dl_det: 
                i+=1

        EventFractions.append(N_events/totcount)
    return EventFractions

def GaussianSigmad(x, mu, sigmaprop): #x here would be dltrue, mu is dlobs. It's a distribution to find dltrue
    return 1/(sigmaprop*x*np.sqrt(2*np.pi))*np.exp(-0.5*(x-mu)**2/(sigmaprop*x)**2)#/norm

def GaussianSigmadtofit(x,mu):
    sigmaprop=0.2
    return 1/(sigmaprop*x*np.sqrt(2*np.pi))*np.exp(-0.5*(x-mu)**2/(sigmaprop*x)**2)#/norm


def RejectSampleLikelihood(likelihoodtype, dl_gw, Nsamps, x_min, x_max, sigmaprop, plotcheck=False):
    dltrapz=np.linspace(x_min, x_max, 1000)
    if likelihoodtype=="gaussian-sigmad": #note that in this case, the dl_sigma is the proportionality constant between sigma_dl and dl
        #norm=np.trapz(GaussianSigmad(dltrapz, mu=dl_gw, sigmaprop=sigma_dl_prop_ass), dltrapz)
        #def FunToSample(y):
        #    return GaussianSigmad(y, mu=dl_gw, sigmaprop=sigma_dl_prop_ass)/norm
        #print(GaussianSigmad(dltrapz, mu=dl_gw, sigmaprop=sigma_dl_prop_ass))
        y_max=max(GaussianSigmad(dltrapz, mu=dl_gw, sigmaprop=sigmaprop))
    
        batch=int(Nsamps/5)
        samples=[]
        
        while len(samples)<Nsamps:
            x=np.random.uniform(low=x_min, high=x_max, size=batch)
            x=x[x>0.]
            y=np.random.uniform(low=0, high=y_max, size=len(x))
            samples+=list(x[y<GaussianSigmad(x, dl_gw, sigmaprop)])
            #print("Samples number: ", len(samples))
        samps=samples[:Nsamps]
    if plotcheck==True:
        plt.hist(samps, color='green', alpha=0.5, density=True, bins=100)
        dltrapz=np.linspace(x_min, x_max, 1000)
        norm=np.trapz(GaussianSigmad(dltrapz, mu=dl_gw, sigmaprop=sigmaprop), dltrapz)
        
        plt.plot(dltrapz, GaussianSigmad(dltrapz, mu=dl_gw, sigmaprop=sigmaprop)/norm, color='black')
        plt.axvline(dl_gw, color='red', label='mu')
        plt.axvline(np.median(samps), color='green', label='median')
        plt.axvline(np.mean(samps), color='blue', label='mean')
        plt.show()
        #sys.exit()
    return np.array(samps)

def samples(dl_gw, sigmaprop, Nsamps):
    dlsamples=[]  

    dlsamps=RejectSampleLikelihood(likelihoodtype="gaussian-sigmad", dl_gw=dl_gw, Nsamps=Nsamps, x_min=dl_gw/100, x_max=3*dl_gw, sigmaprop=sigmaprop, plotcheck=False)#TODO check impact of x_max here
    print("LEN new dlsamps", len(dlsamps))
    dlsamps_mask=np.where(dlsamps>0)[0]#keep only positive samples
    print("Surviving new samps after mask cut ", len(list(dlsamps[dlsamps_mask])))
    dlsamples+=list(dlsamps[dlsamps_mask])
    return dlsamples

"""
mu=100
sigmaprop=0.2

a=samples(mu, sigmaprop, 10000)
histy, binedges=np.histogram(a, density=True, bins=100)
histx=np.array([(binedges[i]+binedges[i+1])/2 for i in range(len(histy))])
plt.plot(histx, histy, label='histplot')
plt.hist(a, bins=100, density=True)


fit=scipy.optimize.curve_fit(GaussianSigmadtofit, histx, histy, p0=100)
xs=np.linspace(20,200,1000)
ys=GaussianSigmad(xs, mu, sigmaprop)
plt.plot(xs, ys, label="GaussianSigmad")
plt.legend()
plt.show()

print(mu)
print(np.median(a))
print(np.mean(a))
print(fit[0])
sys.exit()"""

        

#Constants
#H0_true=70.
linear_data_gen=True
Omega_m=0.25
c=scipy.constants.c/1000
rate='constant'
cosmo = fast_cosmology(Omega_m=Omega_m, linear=linear_data_gen)

sigma_ra_ass=0.1
sigma_dec_ass=sigma_ra_ass
cov_radec_ass=np.zeros((2,2)) #set radec covariance matrix
cov_radec_ass[0][0]=sigma_ra_ass**2
cov_radec_ass[1][1]=sigma_dec_ass**2




H0arr=np.linspace(40,100,100)
CatalogFullName='Catalogs/Micev1WithMTrueScratch.hdf5'
CatalogName="Mice"
Catalog=galaxyCatalog(CatalogFullName)



dldet=100
sigmaprop=0.2
angle_cut_fraction=0.0
z_cut_fraction=0.1

ComputedPdet=pD_inside_theoretic(H0arr, Catalog, dldet, sigmaprop)
ObservedPdet=GetDetectedEventFraction(H0arr, Catalog, dldet, sigmaprop, angle_cut_fraction, z_cut_fraction)

plt.plot(H0arr, ComputedPdet, label="Computed Pdet")
plt.plot(H0arr, ObservedPdet, label="Observed Pdet as fractions of detected")
plt.legend()
plt.savefig("CheckForStatistics-of-ScratchFunctions/PdetComputedvsFractionEvents/ComputedvsEventFractions_Catalog"+CatalogName+"_nominaldldet"+str(dldet)+"_sigmaprop"+str(sigmaprop)+"_zcut"+str(z_cut_fraction)+"_anglecut"+str(angle_cut_fraction)+".pdf")
sys.exit()
















##################### CHECK EXTRACTION PROCESS #########################################################################
angle_cut_fraction=0.0#cut this amount from extremes of ra, dec and z
z_cut_fraction=0.0

racut=np.pi/2*angle_cut_fraction#don't get this close to the ra edges
ra_cut_min=racut
ra_cut_max=np.pi/2-racut
deccut=np.pi/2*angle_cut_fraction #don't get this close to the dec edges
dec_cut_min=racut
dec_cut_max=np.pi/2-racut

maxzcat=0.1   
zcut=maxzcat*z_cut_fraction#don't get this close to the z edges
z_cut_min=0
z_cut_max=maxzcat-zcut


dl_det_threshold=400
dl_cut_max=dl_zH0(z_cut_max, H0=H0_true, Omega_m=Omega_m, linear=linear_data_gen)#z_cut_max*c/H0_true#BeforeLinearImplementationVersion DON'T DELETE THIS IS USED IN SCRATCH
#dl_cut_max is the maximum possible distance at which I can have gravitational waves. Actual detection threshold then will be the minimum between the two
dl_det_threshold_used=min(dl_cut_max, dl_det_threshold) #Not sure about this though, if I use it in the analysis is it giving me info about H0?

#Distribution parameters
sigma_dl_prop_ass=0.2

sigma_ra_ass=0.1
sigma_dec_ass=sigma_ra_ass
cov_radec_ass=np.zeros((2,2)) #set radec covariance matrix
cov_radec_ass[0][0]=sigma_ra_ass**2
cov_radec_ass[1][1]=sigma_dec_ass**2

CatalogFullName='Catalogs/Micev1WithMTrueScratch.hdf5'
with h5py.File(CatalogFullName, 'r') as f:
    ras=f['ra'][()]
    decs=f['dec'][()]
    zs=f['z'][()]
    ms=f['m'][()]


studydir="CheckForStatistics-of-ScratchFunctions/"
Nevents=10
gaussianextractions=10000
for i in range(0,Nevents):
    raextractions=[]
    decextractions=[]
    dlextractions=[]
    hostind=ChooseHost(ras, decs, zs, ms, ra_cut_min, ra_cut_max, dec_cut_min, dec_cut_max, z_cut_min, z_cut_max)
    dl_host=dl_zH0(zs[hostind], H0=H0_true, Omega_m=Omega_m, linear=linear_data_gen)#Introduced after linear Implementation
    for j in range(0,gaussianextractions):
        ra_gw, dec_gw, dl_gw = ExtractObservedGWEvent(ras[hostind], decs[hostind], dl_host, sigma_dl_prop_ass*dl_host, cov_radec_ass)#this or the next one?#BeforeLinearImplementationVersion
        raextractions.append(ra_gw)
        decextractions.append(dec_gw)
        dlextractions.append(dl_gw)

    plt.hist(raextractions, label='extracted gw', bins=100, density=True)
    plt.axvline(ras[hostind], label='host')
    raarray=np.linspace(min(raextractions), max(raextractions), 100)
    plt.plot(raarray, gaussian(raarray, ras[hostind], sigma_ra_ass), label='gaussian')
    plt.xlabel('ra')
    plt.legend()
    plt.savefig(studydir+str(i)+"_"+CatalogFullName[9:13]+"ras_sigmara-"+str(sigma_ra_ass)+".png")
    plt.close()

    plt.hist(decextractions, label='extracted gw', bins=100, density=True)
    plt.axvline(decs[hostind], label='host')
    decarray=np.linspace(min(decextractions), max(decextractions), 100)
    plt.plot(decarray, gaussian(decarray, decs[hostind], sigma_dec_ass), label='gaussian')
    plt.xlabel('dec')
    plt.legend()
    plt.savefig(studydir+str(i)+"_"+CatalogFullName[9:13]+"decs_sigmadec-"+str(sigma_dec_ass)+".png")
    plt.close()

    plt.hist(dlextractions, label='extracted gw', bins=100, density=True)
    plt.axvline(dl_host, label='host')
    dlarray=np.linspace(min(dlextractions), max(dlextractions), 100)
    plt.plot(dlarray, gaussian(dlarray, dl_host, sigma_dl_prop_ass*dl_host), label='gaussian')
    plt.xlabel('dl')
    plt.legend()
    plt.savefig(studydir+str(i)+"_"+CatalogFullName[9:13]+"dls_sigmadlprop-"+str(sigma_dl_prop_ass)+".png")
    plt.close()
    #if dl_gw<dl_det_threshold_used: 
            
                    