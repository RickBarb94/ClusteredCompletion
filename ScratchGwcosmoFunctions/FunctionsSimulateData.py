import numpy as np
import scipy
import scipy.constants
import h5py
import matplotlib.pyplot as plt
import sys
import os
import matplotlib
from utilities.standard_cosmology import fast_cosmology
from scipy.interpolate import interp1d

#matplotlib.use("TkAgg")
#matplotlib.rcParams['agg.path.chunksize'] = 10000

c=scipy.constants.c/1000



class DataGeneration(object):

    def __init__(self, CatalogFullName, dl_det_threshold, H0_true=70, linear_data_gen=True, Omega_m=0.25, sigma_dl_prop_ass=0.2, sigma_ra_ass=0.1, sigma_dec_ass=0.1, Nsamps=10, angle_cut_fraction=0.0, fullanglesky=False, radec_distr='gaussian', dl_distr='gaussian-sigmad', useglobalpdl=False, globalpdlfile=None, weightedcatevents=False):
        if useglobalpdl==True:
            self.globalpdlfile=globalpdlfile
            self.fitted_inverse_cdf=self.Getglobalcdf_fromfile(globalpdlfile)

        self.weightedcatevents=weightedcatevents
        self.CatalogFullName=CatalogFullName

        with h5py.File(self.CatalogFullName, 'r') as f:
            self.ras=f['ra'][()]
            self.decs=f['dec'][()]
            self.zs=f['z'][()]
            self.ms=f['m'][()]
            if self.weightedcatevents==True:
                self.weights=f['weights'][()]
                self.pweights=self.weights/sum(self.weights) #This done here just so I do it once and not renormalize every time I choose a host
                
        ###Catalog host choice parameterslikelihood
        self.angle_cut_fraction=angle_cut_fraction #cut this amount from extremes of ra, dec and z. Ra and dec only if not fullsky   ###############THIS RACUT#########################
        self.fullanglesky=fullanglesky                                                                                    ###############THIS WHETHER USING OCTAVE OR FULLSKY COPIES ###########
        self.dl_det_threshold=dl_det_threshold                                                                                 ###############THIS always, in the thousands for scaled#########################

        if self.fullanglesky==True:
            self.ramin=0
            self.ramax=2*np.pi
            self.decmin=-np.pi/2
            self.decmax=np.pi/2
            self.ra_cut_min=self.ramin
            self.ra_cut_max=self.ramax
            self.dec_cut_min=self.decmin
            self.dec_cut_max=self.decmax
            #no angle edge cuts if fullsky
        else:
            self.ramin=0.
            self.ramax=np.pi/2
            self.decmin=0.
            self.decmax=np.pi/2
            #edge cuts
            self.racut=np.pi/2*angle_cut_fraction#don't get this close to the ra edges
            self.ra_cut_min=self.racut
            self.ra_cut_max=np.pi/2-self.racut
            self.deccut=np.pi/2*self.angle_cut_fraction #don't get this close to the dec edges
            self.dec_cut_min=self.racut
            self.dec_cut_max=np.pi/2-self.racut

        self.mask=np.where((self.ras>=self.ra_cut_min) & (self.ras<=self.ra_cut_max) & (self.decs>=self.dec_cut_min) & (self.decs<=self.dec_cut_max))[0] #check the mask works properly
        #mask is basically a list of index of galaxies inside the cuts. If no cuts, the mask is just a list of indexes of all galaxies.

        #Constants
        self.H0_true=H0_true                                                              
        self.linear_data_gen=linear_data_gen
        self.Omega_m=Omega_m
        self.cosmo=fast_cosmology(Omega_m=self.Omega_m, linear=self.linear_data_gen)

        #Distribution parameters
        self.sigma_dl_prop_ass=sigma_dl_prop_ass                   
        self.sigma_ra_ass=sigma_ra_ass
        self.sigma_dec_ass=sigma_dec_ass
        cov_radec_ass=np.zeros((2,2)) #set radec covariance matrix
        cov_radec_ass[0][0]=sigma_ra_ass**2
        cov_radec_ass[1][1]=sigma_dec_ass**2
        self.radec_cov=cov_radec_ass
        self.radec_distr=radec_distr
        self.dl_distr=dl_distr 
        self.Nsamps=Nsamps

    def ProduceAndSaveSamples_fromglobalpDl(self, targetfile="Samples.hdf5", N_events=250, plotcheck=False): 
        
        file=h5py.File(targetfile, 'w')                      
        
        if plotcheck==True:
            if os.path.exists(targetfile[:-5])==False:
                os.mkdir(targetfile[:-5])
                os.mkdir(targetfile[:-5]+"/dlsamps")
                os.mkdir(targetfile[:-5]+"/decsamps")
                os.mkdir(targetfile[:-5]+"/rasamps")
        
        hostsras=[]
        hostsdecs=[]
        hostsdls=[]
        gwsras=[]
        gwsdecs=[]
        gwsdls=[]
        i=0
        while i<N_events:
            dl_gw = self.ExtractObservedGWEvent_fromglobalpdl()#Extract from global, prebuilt, pdl. it will automatically be below threshold
            ra_gw=np.pi/4
            dec_gw=np.pi/4 #method probably makes sense only in 1d, so putting all ras and decs the same. I can modify this later if I want to extend to 3d

            print("Event Number ", str(i), " ra, dec, dist ",ra_gw, dec_gw, dl_gw)

            hostsras.append(np.pi/4)
            hostsdecs.append(np.pi/4)
            hostsdls.append(200) #hosts are useless in this case, keeping this just to keep formatting
            gwsras.append(ra_gw)
            gwsdecs.append(dec_gw)
            gwsdls.append(dl_gw)

            #rasamps, decsamps, dlsamps=self.GenerateSamples(ra_gw, dec_gw, dl_gw) Not using samples generation anymore!

            rasamps=np.zeros(2)
            decsamps=np.zeros(2)
            dlsamps=np.zeros(2) #leftover from samps generation
            
            grp=file.create_group(str(i))
            grp.create_dataset("ra_samps", data=rasamps)
            grp.create_dataset("dec_samps", data=decsamps)
            grp.create_dataset("dl_samps", data=dlsamps)
            grp.create_dataset("ra_inj", data=np.pi/4)
            grp.create_dataset("dec_inj", data=np.pi/4)
            grp.create_dataset("dl_inj", data=200)# putting all host at 200, only 1d for now (see above)
            grp.create_dataset("ra_gw", data=ra_gw)
            grp.create_dataset("dec_gw", data=dec_gw)
            grp.create_dataset("dl_gw", data=dl_gw)


            if plotcheck==True:
                if self.dl_distr=="gaussian":
                    dlpl=np.linspace(0,400,1000)
                    plt.plot(dlpl, gaussian(dlpl, mu=dl_gw, sig=self.sigma_dl_prop_ass*dl_gw))
                    plt.axvline(dl_host, color='red', ls='dashed', label='injected dist')
                    plt.axvline(dl_gw, color='blue', ls='dashed', label='measured gw dist')
                    plt.hist(dlsamps, density=True, label='dl samples', color='green', alpha=0.5, bins=50)
                    plt.xlabel('dl')
                    plt.legend()
                    plt.savefig(targetfile[:-5]+"/dlsamps/dlsamps_check"+str(i)+".png")
                    plt.close()
                elif self.dl_distr=='gaussian-sigmad':
                    dltrapz=np.linspace(dl_gw/100, 3*dl_gw, 1000)
                    norm=np.trapz(GaussianSigmad(dltrapz, mu=dl_gw, sigmaprop=self.sigma_dl_prop_ass), dltrapz)
                    plt.plot(dltrapz, GaussianSigmad(dltrapz, mu=dl_gw, sigmaprop=self.sigma_dl_prop_ass)/norm, color='black')
                    plt.axvline(dl_host, color='red', ls='dashed', label='injected dist')
                    plt.axvline(dl_gw, color='blue', ls='dashed', label='measured gw dist')
                    plt.hist(dlsamps, color='green', alpha=0.5, density=True, bins=50, label='samples')
                    plt.xlabel('dl')
                    plt.legend()
                    plt.savefig(targetfile[:-5]+"/dlsamps/dlsamps_check"+str(i)+".png")
                    plt.close()
                rapl=np.linspace(self.ramin, self.ramax, 1000)
                plt.plot(rapl, gaussian(rapl, mu=ra_gw, sig=self.sigma_ra_ass), color='black', label='distribution')
                plt.hist(rasamps, density=True, bins=50, color='green', label='samples')
                plt.axvline(self.ras[hostind], color='red', ls='dashed', label='injected ra')
                plt.axvline(ra_gw, color='blue', ls='dashed', label='measured gw ra')
                plt.xlabel('ra')
                plt.legend()
                plt.savefig(targetfile[:-5]+"/rasamps/rasamps_check"+str(i)+".png")
                plt.close()
                decpl=np.linspace(self.decmin, self.decmax, 1000)
                plt.plot(decpl, gaussian(decpl, mu=dec_gw, sig=self.sigma_dec_ass), color='black', label='distribution')
                plt.hist(decsamps, density=True, bins=50, color='green', label='samples')
                plt.axvline(self.decs[hostind], color='red', ls='dashed', label='injected dec')
                plt.axvline(dec_gw, color='blue', ls='dashed', label='measured gw dec')
                plt.xlabel('dec')
                plt.legend()
                plt.savefig(targetfile[:-5]+"/decsamps/decsamps_check"+str(i)+".png")
                plt.close()
            i+=1
        if plotcheck==True:
            plt.hist(gwsras, alpha=0.5, color='red', density=True, label='Observed events')
            plt.hist(hostsras, alpha=0.5, color='green', density=True, label='Actual hosts')
            plt.hist(self.ras, alpha=0.5, color='blue', density=True, label='Catalog galaxies')
            plt.legend()
            plt.xlabel('ra')
            plt.savefig(targetfile[:-5]+"/Ra_Gw-Hosts_hist.pdf")
            plt.close()
            plt.hist(gwsdecs, alpha=0.5, color='red', density=True, label='Observed events')
            plt.hist(hostsdecs, alpha=0.5, color='green', density=True, label='Actual hosts')
            plt.hist(self.decs, alpha=0.5, color='blue', density=True, label='Catalog galaxies')
            plt.legend()
            plt.xlabel('dec')
            plt.savefig(targetfile[:-5]+"/Dec_Gw-Hosts_hist.pdf")
            plt.close()
            plt.hist(gwsdls, alpha=0.5, color='red', density=True, label='Observed events')
            plt.hist(hostsdls, alpha=0.5, color='green', density=True, label='Actual hosts')
            dls=[self.cosmo.dl_zH0(self.zs[j], H0=self.H0_true) for j in np.arange(len(self.zs))]
            plt.hist(dls, alpha=0.5, color='blue', density=True, label='Catalog galaxies')
            plt.xlabel('dl')
            plt.legend()
            plt.savefig(targetfile[:-5]+"/Dl_Gw-Hosts_hist.pdf")
            plt.close()
            
            
                
        file.close()
        return 0

    def ProduceAndSaveSamples(self, targetfile="Samples.hdf5", N_events=250, plotcheck=False, centered=False, withinbounds=False):
       
        file=h5py.File(targetfile, 'w')
        
        if plotcheck==True:
            if os.path.exists(targetfile[:-5])==False:
                os.mkdir(targetfile[:-5])
                os.mkdir(targetfile[:-5]+"/dlsamps")
                os.mkdir(targetfile[:-5]+"/decsamps")
                os.mkdir(targetfile[:-5]+"/rasamps")

        hostsras=[]
        hostsdecs=[]
        hostsdls=[]
        gwsras=[]
        gwsdecs=[]
        gwsdls=[]
        i=0
        while i<N_events:
            #print(i)
            hostind=self.ChooseHost()
            dl_host=self.cosmo.dl_zH0(self.zs[hostind], H0=self.H0_true)#Introduced after linear Implementation
            #print(self.ras[hostind])
            if centered==True:
                ra_gw=self.ras[hostind]
                dec_gw=self.decs[hostind]
                dl_gw=dl_host#zs[hostind]*c/H0_true#BeforeLinearImplementationVersion
            else:
                ra_gw, dec_gw, dl_gw = self.ExtractObservedGWEvent(self.ras[hostind], self.decs[hostind], dl_host, self.sigma_dl_prop_ass*dl_host, self.radec_cov, withinbounds=withinbounds)#this or the next one?#BeforeLinearImplementationVersion
                

            #With this, I might get hosts that are above dl_det_threshold. These two lines are probably the right ones (confirmed by Jon)
            if dl_gw<self.dl_det_threshold: 
                
            #if zs[hostind]*c/H0_true<dl_det_threshold: #With this, I might get dl_gws that are above dl_det_threshold# Not modified with linear implementation

                print("selected host for Event " +str(i)+" is ", hostind)
                print("Host ra, dec, z, dl ", self.ras[hostind], self.decs[hostind], self.zs[hostind], dl_host)#assuming linear cosmo
                print("Event Number ", str(i), " ra, dec, dist ",ra_gw, dec_gw, dl_gw)

                hostsras.append(self.ras[hostind])
                hostsdecs.append(self.decs[hostind])
                hostsdls.append(dl_host)
                gwsras.append(ra_gw)
                gwsdecs.append(dec_gw)
                gwsdls.append(dl_gw)

                
                #rasamps, decsamps, dlsamps=self.GenerateSamples(ra_gw, dec_gw, dl_gw) Not using samples generation anymore!

                rasamps=np.zeros(2)
                decsamps=np.zeros(2)
                dlsamps=np.zeros(2) #leftover from samps generation
                
                grp=file.create_group(str(i))
                grp.create_dataset("ra_samps", data=rasamps)
                grp.create_dataset("dec_samps", data=decsamps)
                grp.create_dataset("dl_samps", data=dlsamps)
                grp.create_dataset("ra_inj", data=self.ras[hostind])
                grp.create_dataset("dec_inj", data=self.decs[hostind])
                grp.create_dataset("dl_inj", data=dl_host)
                grp.create_dataset("ra_gw", data=ra_gw)
                grp.create_dataset("dec_gw", data=dec_gw)
                grp.create_dataset("dl_gw", data=dl_gw)


                if plotcheck==True:
                    if self.dl_distr=="gaussian":
                        dlpl=np.linspace(0,400,1000)
                        plt.plot(dlpl, gaussian(dlpl, mu=dl_gw, sig=self.sigma_dl_prop_ass*dl_gw))
                        plt.axvline(dl_host, color='red', ls='dashed', label='injected dist')
                        plt.axvline(dl_gw, color='blue', ls='dashed', label='measured gw dist')
                        plt.hist(dlsamps, density=True, label='dl samples', color='green', alpha=0.5, bins=50)
                        plt.xlabel('dl')
                        plt.legend()
                        plt.savefig(targetfile[:-5]+"/dlsamps/dlsamps_check"+str(i)+".png")
                        plt.close()
                    elif self.dl_distr=='gaussian-sigmad':
                        dltrapz=np.linspace(dl_gw/100, 3*dl_gw, 1000)
                        norm=np.trapz(GaussianSigmad(dltrapz, mu=dl_gw, sigmaprop=self.sigma_dl_prop_ass), dltrapz)
                        plt.plot(dltrapz, GaussianSigmad(dltrapz, mu=dl_gw, sigmaprop=self.sigma_dl_prop_ass)/norm, color='black')
                        plt.axvline(dl_host, color='red', ls='dashed', label='injected dist')
                        plt.axvline(dl_gw, color='blue', ls='dashed', label='measured gw dist')
                        plt.hist(dlsamps, color='green', alpha=0.5, density=True, bins=50, label='samples')
                        plt.xlabel('dl')
                        plt.legend()
                        plt.savefig(targetfile[:-5]+"/dlsamps/dlsamps_check"+str(i)+".png")
                        plt.close()
                    rapl=np.linspace(self.ramin, self.ramax, 1000)
                    plt.plot(rapl, gaussian(rapl, mu=ra_gw, sig=self.sigma_ra_ass), color='black', label='distribution')
                    plt.hist(rasamps, density=True, bins=50, color='green', label='samples')
                    plt.axvline(self.ras[hostind], color='red', ls='dashed', label='injected ra')
                    plt.axvline(ra_gw, color='blue', ls='dashed', label='measured gw ra')
                    plt.xlabel('ra')
                    plt.legend()
                    plt.savefig(targetfile[:-5]+"/rasamps/rasamps_check"+str(i)+".png")
                    plt.close()
                    decpl=np.linspace(self.decmin, self.decmax, 1000)
                    plt.plot(decpl, gaussian(decpl, mu=dec_gw, sig=self.sigma_dec_ass), color='black', label='distribution')
                    plt.hist(decsamps, density=True, bins=50, color='green', label='samples')
                    plt.axvline(self.decs[hostind], color='red', ls='dashed', label='injected dec')
                    plt.axvline(dec_gw, color='blue', ls='dashed', label='measured gw dec')
                    plt.xlabel('dec')
                    plt.legend()
                    plt.savefig(targetfile[:-5]+"/decsamps/decsamps_check"+str(i)+".png")
                    plt.close()
                i+=1
        if plotcheck==True:
            plt.hist(gwsras, alpha=0.5, color='red', density=True, label='Observed events')
            plt.hist(hostsras, alpha=0.5, color='green', density=True, label='Actual hosts')
            plt.hist(self.ras, alpha=0.5, color='blue', density=True, label='Catalog galaxies')
            plt.legend()
            plt.xlabel('ra')
            plt.savefig(targetfile[:-5]+"/Ra_Gw-Hosts_hist.pdf")
            plt.close()
            plt.hist(gwsdecs, alpha=0.5, color='red', density=True, label='Observed events')
            plt.hist(hostsdecs, alpha=0.5, color='green', density=True, label='Actual hosts')
            plt.hist(self.decs, alpha=0.5, color='blue', density=True, label='Catalog galaxies')
            plt.legend()
            plt.xlabel('dec')
            plt.savefig(targetfile[:-5]+"/Dec_Gw-Hosts_hist.pdf")
            plt.close()
            plt.hist(gwsdls, alpha=0.5, color='red', density=True, label='Observed events')
            plt.hist(hostsdls, alpha=0.5, color='green', density=True, label='Actual hosts')
            dls=[self.cosmo.dl_zH0(self.zs[j], H0=self.H0_true) for j in np.arange(len(self.zs))]
            plt.hist(dls, alpha=0.5, color='blue', density=True, label='Catalog galaxies')
            plt.xlabel('dl')
            plt.legend()
            plt.savefig(targetfile[:-5]+"/Dl_Gw-Hosts_hist.pdf")
            plt.close()
            
            
                
        file.close()
        return 0

    def ChooseHost(self):
        if self.weightedcatevents==False:
            hostind=np.random.choice(self.mask)
        else:
            hostind=np.random.choice(self.mask, p=self.pweights)
            
            
        return hostind


    def ExtractObservedGWEvent(self, ra_host, dec_host, dl_host, sigma_dl, sigma_radec, withinbounds=False): #TODO: BETTER LOCATION OF EVENT, WEIRD IN DEC ESPECIALLY WITH FULL SKY!
        if self.dl_distr=="gaussian-sigmad":
            dl_obs=np.random.normal(dl_host, sigma_dl)
        if self.dl_distr=="uniform":
            dl_obs=np.random.uniform(dl_host-dl_host*self.sigma_dl_prop_ass*2.5, dl_host+dl_host*self.sigma_dl_prop_ass*2.5)#BEWARE HARDCODING
        if withinbounds==True:
            while True:
                ra_obs, dec_obs = np.random.multivariate_normal([ra_host, dec_host], sigma_radec, 1)[0] #this guarantees that event is within catalog limits
                if ra_obs>self.ramin and ra_obs<self.ramax and dec_obs>self.decmin and dec_obs<self.decmax:
                    break
        else:
            ra_obs, dec_obs = np.random.multivariate_normal([ra_host, dec_host], sigma_radec, 1)[0] #this guarantees that event is within catalog limits
                
        return ra_obs, dec_obs, dl_obs 

    def ExtractObservedGWEvent_fromglobalpdl(self):
        return self.fitted_inverse_cdf(np.random.random())
    
    def Getglobalcdf_fromfile(self, globalpdlfile):
        with h5py.File(globalpdlfile, "r") as f:
            cdL_values=f["cdL_values"][:]
            dl_array=f["dl_array"][:]
        

        fitted_inverse_cdf = interp1d(cdL_values, dl_array)
        
        return fitted_inverse_cdf

        
    def GenerateSamples(self, ra_gw, dec_gw, dl_gw):
        #generate samples given GW event sky location, luminosity distance, uncertainty in radec and dl.
        #last two arguments set the distribution: draw from a gaussian or draw from a gaussian with sigma/propto d ('gaussian-sigmad')
        dl_sigma=dl_gw*self.sigma_dl_prop_ass

        batch=3000
        rasamples=[]
        decsamples=[]
        safetycount=0

        while len(rasamples)<self.Nsamps:
            print("radec samps ", len(rasamples))
            if self.radec_distr=='gaussian':
                radecsamps = np.random.multivariate_normal([ra_gw, dec_gw], self.radec_cov, size=batch)
                if self.fullanglesky==False:
                    radecsamps_mask = np.where((radecsamps[:,0]>self.ramin) & (radecsamps[:,0]<self.ramax) & (radecsamps[:,1]>self.decmin) & (radecsamps[:,1]<self.decmax))[0] #keep only samples within catalog bounds 
                else:
                    radecsamps_mask = np.where((radecsamps[:,1]>self.decmin) & (radecsamps[:,1]<self.decmax))[0] #keep only samples within catalog bounds, here only dec checked and ra converted #TODO: maybe add boundary condition to dec? Use different distribution for dec?

                radecsamps=radecsamps[radecsamps_mask]
                rasamples+=list(radecsamps[:,0])
                decsamples+=list(radecsamps[:,1])
                if len(radecsamps[:,0])<3:
                    safetycount+=1
                    print("safetycount ", safetycount, " newsamps ", radecsamps[:,0])
                    
                    if safetycount>=10:
                        print("Something wrong in ra and dec samps generation!")
                        print("ra gw, dec gw ", ra_gw, dec_gw )
                        print("Produced up to now ", len(rasamples), " samples")
                        sys.exit()
        dlsamples=[]  

        x_min_samps=dl_gw/100
        x_max_samps=10*dl_gw
        while len(dlsamples)<self.Nsamps:
            print("dlsamples len ", len(dlsamples))
            if self.dl_distr=='gaussian':
                dlsamps=np.random.normal(dl_gw, dl_sigma, self.Nsamps)
                print("LEN new dlsamps ", len(dlsamps))
                dlsamps_mask=np.where(dlsamps>0)[0]#keep only positive samples
                print("Surviving new samps after mask cut ", len(list(dlsamps[dlsamps_mask])))
                dlsamples+=list(dlsamps[dlsamps_mask]) 

            elif self.dl_distr=="gaussian-sigmad":
                dlsamps=self.RejectSampleLikelihood(likelihoodtype="gaussian-sigmad", dl_gw=dl_gw, x_min=x_min_samps, x_max=x_max_samps, plotcheck=False)#TODO check impact of x_max here
                print("LEN new dlsamps", len(dlsamps))
                dlsamps_mask=np.where(dlsamps>0)[0]#keep only positive samples
                print("Surviving new samps after mask cut ", len(list(dlsamps[dlsamps_mask])))
                dlsamples+=list(dlsamps[dlsamps_mask])

            elif self.dl_distr=="uniform":
                dlsamps=self.UniformSampsWRONGTOCHANGE(dl_gw=dl_gw, plotcheck=False)#TODO check impact of x_max here #TODO CHANGE THIS SAMPLES, FOR NOW OK SINCE I SKIP SAMPLE THINGY
                print("LEN new dlsamps", len(dlsamps))
                dlsamps_mask=np.where(dlsamps>0)[0]#keep only positive samples
                print("Surviving new samps after mask cut ", len(list(dlsamps[dlsamps_mask])))
                dlsamples+=list(dlsamps[dlsamps_mask])

            elif self.dl_distr=="gaussian-sigmad_truncatedatdldetused":
                dlsamps=self.RejectSampleLikelihood(likelihoodtype="gaussian-sigmad", dl_gw=dl_gw, x_min=x_min_samps, x_max=x_max_samps, plotcheck=False)
                dlsamps_mask=np.where((dlsamps>0) & (dlsamps<self.dl_det_threshold))[0]#keep only positive samples
                dlsamples+=list(dlsamps[dlsamps_mask]) 

            elif self.dl_distr=="gaussian-sigmad_truncatedatdldetnominal":
                dlsamps=self.RejectSampleLikelihood(likelihoodtype="gaussian-sigmad", dl_gw=dl_gw, x_min=x_min_samps, x_max=x_max_samps, plotcheck=False)
                dlsamps_mask=np.where((dlsamps>0) & (dlsamps<self.dl_det_threshold))[0]#keep only positive samples
                dlsamples+=list(dlsamps[dlsamps_mask]) 

            elif self.dl_distr=="gaussian-sigmad_truncatedatcatalogedge":
                dlsamps=self.RejectSampleLikelihood(likelihoodtype="gaussian-sigmad", dl_gw=dl_gw, x_min=x_min_samps, x_max=x_max_samps, plotcheck=False)
                print("LEN new dlsamps", len(dlsamps))
                dlsamps_mask=np.where((dlsamps>0) & (dlsamps<self.dlmaxcat))[0]#keep only positive samples
                print("Surviving new samps after mask cut ", len(list(dlsamps[dlsamps_mask])))
                dlsamples+=list(dlsamps[dlsamps_mask]) 
        
        rasamples=rasamples[:self.Nsamps]
        decsamples=decsamples[:self.Nsamps]
        dlsamples=dlsamples[:self.Nsamps]

        if self.fullanglesky==True:
            for i in range(len(rasamples)):
                rasamples[i]=ConvertOutofBondsra(rasamples[i]) #check boundarys and convert to inboundaries
            
        
        return rasamples, decsamples, dlsamples

    def RejectSampleLikelihood(self, likelihoodtype, dl_gw, x_min, x_max, plotcheck=False):
        dltrapz=np.linspace(x_min, x_max, 1000)
        if likelihoodtype=="gaussian-sigmad": #note that in this case, the dl_sigma is the proportionality constant between sigma_dl and dl
            #norm=np.trapz(GaussianSigmad(dltrapz, mu=dl_gw, sigmaprop=sigma_dl_prop_ass), dltrapz)
            #def FunToSample(y):
            #    return GaussianSigmad(y, mu=dl_gw, sigmaprop=sigma_dl_prop_ass)/norm
            #print(GaussianSigmad(dltrapz, mu=dl_gw, sigmaprop=sigma_dl_prop_ass))
            y_max=max(GaussianSigmad(dltrapz, mu=dl_gw, sigmaprop=self.sigma_dl_prop_ass))
        
            batch=int(self.Nsamps/5)
            samples=[]
            
            while len(samples)<self.Nsamps:
                x=np.random.uniform(low=x_min, high=x_max, size=batch)
                x=x[x>0.]
                y=np.random.uniform(low=0, high=y_max, size=len(x))
                samples+=list(x[y<GaussianSigmad(x, dl_gw, self.sigma_dl_prop_ass)])
                #print("dl samples number: ", len(samples))
            samps=samples[:self.Nsamps]
        if plotcheck==True:
            plt.hist(samps, color='green', alpha=0.5, density=True)
            norm=np.trapz(GaussianSigmad(dltrapz, mu=dl_gw, sigmaprop=self.sigma_dl_prop_ass), dltrapz)
            
            plt.plot(dltrapz, GaussianSigmad(dltrapz, mu=dl_gw, sigmaprop=self.sigma_dl_prop_ass)/norm, color='black')
            plt.axvline(dl_gw, color='red')
            plt.show()
            sys.exit()
        print("dl samples number: ", len(samps))
        return np.array(samps)

    
    def UniformSampsWRONGTOCHANGE(self, dl_gw, plotcheck=False):
        sigma_dl_gw=self.sigma_dl_prop_ass * 2.5 * dl_gw #BEWARE HARDCODING #This means that the default is a spread of the uniform distribution of half the injected distance for the usual value of 0.2
        xmin=dl_gw-sigma_dl_gw
        xmax=dl_gw+sigma_dl_gw
        samples=np.random.uniform(xmin, xmax, size=self.Nsamps)
        return samples

    def TOFIXORCHECK_ProduceAndSaveSamplesfromFixedLocations(CatalogFullName, radec_distr, dl_distr, previous_events_file_name, targetfile="newsamples.hdf5", N_events=250, plotcheck=False):
        #same as above, but using some fixed event hosts or location (for now only location)
        with h5py.File(CatalogFullName, 'r') as f:
            ras=f['ra'][()]
            decs=f['dec'][()]
            zs=f['z'][()]
            ms=f['m'][()]
        
        file=h5py.File(targetfile, 'w')
        i=0
        if plotcheck==True:
            if os.path.exists(targetfile[:-5])==False:
                os.mkdir(targetfile[:-5])
                os.mkdir(targetfile[:-5]+"/dlsamps")
                os.mkdir(targetfile[:-5]+"/decsamps")
                os.mkdir(targetfile[:-5]+"/rasamps")
        hostsras=[]
        hostsdecs=[]
        hostsdls=[]
        gwsras=[]
        gwsdecs=[]
        gwsdls=[]
        while i<N_events:
            print(i)
            with h5py.File(previous_events_file_name, 'r') as previous_events_file:
        
                rasampsprev=previous_events_file[str(i)]["ra_samps"][()]
                decsampsprev=previous_events_file[str(i)]["dec_samps"][()]
                dlsampsprev=previous_events_file[str(i)]["dl_samps"][()]
                rainjprev=previous_events_file[str(i)]["ra_inj"][()]
                decinjprev=previous_events_file[str(i)]["dec_inj"][()]
                dlinjprev=previous_events_file[str(i)]["dl_inj"][()]

            dl_host=dlinjprev

            dl_gw=np.median(dlsampsprev)
            ra_gw=np.median(rasampsprev)
            dec_gw=np.median(decsampsprev)

        
            print("Host ra, dec, dl ", rainjprev, decinjprev, dl_host)#assuming linear cosmo
            print("Event Number ", str(i), " ra, dec, dist ",ra_gw, dec_gw, dl_gw)
            hostsras.append(rainjprev)
            hostsdecs.append(decinjprev)
            hostsdls.append(dl_host)
            gwsras.append(ra_gw)
            gwsdecs.append(dec_gw)
            gwsdls.append(dl_gw)
            grp=file.create_group(str(i))
            rasamps, decsamps, dlsamps=GenerateSamples(ra_gw, dec_gw, self.radec_cov, dl_gw, sigma_dl_prop_ass*dl_gw, NSamps_ass, radec_distr=radec_distr, dl_distr=dl_distr)
            grp.create_dataset("ra_samps", data=rasamps)
            grp.create_dataset("dec_samps", data=decsamps)
            grp.create_dataset("dl_samps", data=dlsamps)
            grp.create_dataset("ra_inj", data=rainjprev)
            grp.create_dataset("dec_inj", data=decinjprev)
            grp.create_dataset("dl_inj", data=dl_host)
            if plotcheck==True:
                if dl_distr=="gaussian":
                    dlpl=np.linspace(0,400,1000)
                    plt.plot(dlpl, gaussian(dlpl, mu=dl_gw, sig=sigma_dl_prop_ass*dl_gw))
                    plt.axvline(dl_host, color='red', ls='dashed', label='injected dist')
                    plt.axvline(dl_gw, color='blue', ls='dashed', label='measured gw dist')
                    plt.hist(dlsamps, density=True, label='dl samples', color='green', alpha=0.5, bins=50)
                    plt.xlabel('dl')
                    plt.legend()
                    plt.savefig(targetfile[:-5]+"/dlsamps/dlsamps_check"+str(i)+".png")
                    plt.close()
                elif dl_distr=='gaussian-sigmad':
                    dltrapz=np.linspace(dl_gw/100, 3*dl_gw, 1000)
                    norm=np.trapz(GaussianSigmad(dltrapz, mu=dl_gw, sigmaprop=sigma_dl_prop_ass), dltrapz)
                    plt.plot(dltrapz, GaussianSigmad(dltrapz, mu=dl_gw, sigmaprop=sigma_dl_prop_ass)/norm, color='black')
                    plt.axvline(dl_host, color='red', ls='dashed', label='injected dist')
                    plt.axvline(dl_gw, color='blue', ls='dashed', label='measured gw dist')
                    plt.hist(dlsamps, color='green', alpha=0.5, density=True, bins=50, label='samples')
                    plt.xlabel('dl')
                    plt.legend()
                    plt.savefig(targetfile[:-5]+"/dlsamps/dlsamps_check"+str(i)+".png")
                    plt.close()
                rapl=np.linspace(ramin, ramax, 1000)
                plt.plot(rapl, gaussian(rapl, mu=ra_gw, sig=sigma_ra_ass), color='black', label='distribution')
                plt.hist(rasamps, density=True, bins=50, color='green', label='samples')
                plt.axvline(rainjprev, color='red', ls='dashed', label='injected ra')
                plt.axvline(ra_gw, color='blue', ls='dashed', label='measured gw ra')
                plt.xlabel('ra')
                plt.legend()
                plt.savefig(targetfile[:-5]+"/rasamps/rasamps_check"+str(i)+".png")
                plt.close()
                decpl=np.linspace(decmin, decmax, 1000)
                plt.plot(decpl, gaussian(decpl, mu=dec_gw, sig=sigma_dec_ass), color='black', label='distribution')
                plt.hist(decsamps, density=True, bins=50, color='green', label='samples')
                plt.axvline(decinjprev, color='red', ls='dashed', label='injected dec')
                plt.axvline(dec_gw, color='blue', ls='dashed', label='measured gw dec')
                plt.xlabel('dec')
                plt.legend()
                plt.savefig(targetfile[:-5]+"/decsamps/decsamps_check"+str(i)+".png")
                plt.close()
            i+=1
        if plotcheck==True:
            plt.hist(gwsras, alpha=0.5, color='red', density=True, label='Observed events')
            plt.hist(hostsras, alpha=0.5, color='green', density=True, label='Actual hosts')
            plt.hist(ras, alpha=0.5, color='blue', density=True, label='Catalog galaxies')
            plt.legend()
            plt.xlabel('ra')
            plt.savefig(targetfile[:-5]+"/Ra_Gw-Hosts_hist.pdf")
            plt.close()
            plt.hist(gwsdecs, alpha=0.5, color='red', density=True, label='Observed events')
            plt.hist(hostsdecs, alpha=0.5, color='green', density=True, label='Actual hosts')
            plt.hist(decs, alpha=0.5, color='blue', density=True, label='Catalog galaxies')
            plt.legend()
            plt.xlabel('dec')
            plt.savefig(targetfile[:-5]+"/Dec_Gw-Hosts_hist.pdf")
            plt.close()
            plt.hist(gwsdls, alpha=0.5, color='red', density=True, label='Observed events')
            plt.hist(hostsdls, alpha=0.5, color='green', density=True, label='Actual hosts')
            dls=[dl_zH0(zs[j], H0=H0_true, Omega_m=Omega_m, linear=linear_data_gen) for j in np.arange(len(zs))]
            plt.hist(dls, alpha=0.5, color='blue', density=True, label='Catalog galaxies')
            plt.xlabel('dl')
            plt.legend()
            plt.savefig(targetfile[:-5]+"/Dl_Gw-Hosts_hist.pdf")
            plt.close()
                
        file.close()
        return 0






def ConvertOutofBondsra(ra):
    newra=ra
    while newra<0:
        newra+=np.pi*2
    while newra>np.pi*2:
        newra-=np.pi*2
    #if newra!=ra:
    #    print("Corrected ra! original ", ra, " corrected ", newra)
    return newra

def DumpEventsLikelihoods(savefile, likelihoods):
    f=h5py.File(savefile, 'w')
    for i in np.arange(len(likelihoods)):
        grp=f.create_dataset(str(i), data=likelihoods[i])
    f.close()
    return 0

def DumpEventsBreakdowns(savefile, datatodump):
    f=h5py.File(savefile, 'w')
    for i in np.arange(len(datatodump)):
        grp=f.create_group(str(i))
        grp.create_dataset('likelihood', data=datatodump[i][0])
        grp.create_dataset('pxG', data=datatodump[i][1])
        grp.create_dataset('pDG', data=datatodump[i][2])
        grp.create_dataset('pG', data=datatodump[i][3])
        grp.create_dataset('pxnG', data=datatodump[i][4])
        grp.create_dataset('pDnG', data=datatodump[i][5])
    f.close()
    return 0

def GaussianSigmad(x, mu, sigmaprop): #x here would be dltrue, mu is dlobs. It's a distribution to find dltrue
    return 1/(sigmaprop*x*np.sqrt(2*np.pi))*np.exp(-0.5*(x-mu)**2/(sigmaprop*x)**2)#/norm

def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def SubSampleFromExistingSamples(EventsFile, subsampling_method="median_check", originalsampledldet=400, N_events=250, plotcheck=True, chunks=250):
    subsamplingstring='_subsampled-'+subsampling_method+'-from-'+str(originalsampledldet)
    print(EventsFile)
    prelimOriginalEventsFile=EventsFile.replace(subsamplingstring, '')
    OriginalEventsFile=prelimOriginalEventsFile.replace("dldet-"+str(dl_det_threshold), "dldet-"+str(originalsampledldet))
    counterorigrealizations=1
    OriginalEventsFile=OriginalEventsFile.replace("_R"+str(counterorigrealizations), "_R"+str(counterorigrealizations))

    if os.path.exists(OriginalEventsFile)==False:
        print("Original samples files to subsample from do not exist!")
        print(OriginalEventsFile)
        sys.exit()
    
    h5newfile=h5py.File(EventsFile, "w")
    counterevents=0
    rainjs=[]
    decinjs=[]
    dlinjs=[]
    if plotcheck==True:
        if os.path.exists(EventsFile[:-5])==False:
            os.mkdir(EventsFile[:-5])
            os.mkdir(EventsFile[:-5]+"/dlsamps")
            os.mkdir(EventsFile[:-5]+"/decsamps")
            os.mkdir(EventsFile[:-5]+"/rasamps")
    N_events_target=N_events
    while counterevents<N_events_target:
        print("Going through original realization ", counterorigrealizations)
            
        with h5py.File(OriginalEventsFile, "r") as h5originalfile:
            for i in np.arange(chunks):
                if counterevents==N_events_target:
                    break
                rasamps=h5originalfile[str(i)]["ra_samps"]
                decsamps=h5originalfile[str(i)]["dec_samps"]
                dlsamps=h5originalfile[str(i)]["dl_samps"]
                rainj=h5originalfile[str(i)]["ra_inj"][()]
                decinj=h5originalfile[str(i)]["dec_inj"][()]
                dlinj=h5originalfile[str(i)]["dl_inj"][()]
                
                if subsampling_method=="host_threshold":
                    if dlinj<dl_det_threshold:
                        check=True
                    else:
                        check=False
                elif subsampling_method=="median_check":#this method is basically the same as the original check on dl_gw, as the extracted gw distance is extremely close to the median of the posterior
                    median=np.median(dlsamps)
                    if median<dl_det_threshold:
                        check=True
                    else:
                        check=False
                elif subsampling_method=="2sigma_host_threshold":
                    if dlinj<dl_det_threshold+2*sigma_dl_prop_ass*dl_det_threshold:
                        check=True
                    else:
                        check=False
                else:
                    print("Unknown subsampling method, please check!")
                if check==True:
                    print("Got event ", counterevents)
                    rainjs.append(rainj)
                    decinjs.append(decinj)
                    dlinjs.append(dlinj)
                    grp=h5newfile.create_group(str(counterevents))
                    grp.create_dataset("ra_samps", data=rasamps)
                    grp.create_dataset("dec_samps", data=decsamps)
                    grp.create_dataset("dl_samps", data=dlsamps)
                    grp.create_dataset("ra_inj", data=rainj)
                    grp.create_dataset("dec_inj", data=decinj)
                    grp.create_dataset("dl_inj", data=dlinj)
                    if plotcheck==True:
                        
                        dltrapz=np.linspace(dlinj/100, 3*dlinj, 1000)
                        norm=np.trapz(GaussianSigmad(dltrapz, mu=dlinj, sigmaprop=sigma_dl_prop_ass), dltrapz)
                        #plt.plot(dltrapz, GaussianSigmad(dltrapz, mu=dl_gw, sigmaprop=sigma_dl_prop_ass)/norm, color='black')
                        plt.axvline(dlinj, color='red', ls='dashed', label='injected dist')
                        #plt.axvline(dl_gw, color='blue', ls='dashed', label='measured gw dist')
                        plt.hist(dlsamps, color='green', alpha=0.5, density=True, bins=50, label='samples')
                        plt.xlabel('dl')
                        plt.legend()
                        plt.savefig(EventsFile[:-5]+"/dlsamps/dlsamps_check"+str(counterevents)+".png")
                        plt.close()
                        rapl=np.linspace(ramin, ramax, 1000)
                        #plt.plot(rapl, gaussian(rapl, mu=ra_gw, sig=sigma_ra_ass), color='black', label='distribution')
                        plt.hist(rasamps, density=True, bins=50, color='green', label='samples')
                        plt.axvline(rainj, color='red', ls='dashed', label='injected ra')
                        #plt.axvline(ra_gw, color='blue', ls='dashed', label='measured gw ra')
                        plt.xlabel('ra')
                        plt.legend()
                        plt.savefig(EventsFile[:-5]+"/rasamps/rasamps_check"+str(counterevents)+".png")
                        plt.close()
                        decpl=np.linspace(decmin, decmax, 1000)
                        #plt.plot(decpl, gaussian(decpl, mu=dec_gw, sig=sigma_dec_ass), color='black', label='distribution')
                        plt.hist(decsamps, density=True, bins=50, color='green', label='samples')
                        plt.axvline(decinj, color='red', ls='dashed', label='injected dec')
                        #plt.axvline(dec_gw, color='blue', ls='dashed', label='measured gw dec')
                        plt.xlabel('dec')
                        plt.legend()
                        plt.savefig(EventsFile[:-5]+"/decsamps/decsamps_check"+str(counterevents)+".png")
                        plt.close()
                    counterevents+=1
                    
        counterorigrealizations+=1 #update for next original realization
        OriginalEventsFile=OriginalEventsFile.replace("_R"+str(counterorigrealizations-1), "_R"+str(counterorigrealizations))
        
        if os.path.exists(OriginalEventsFile)==False:#if I finished pool of original samples, update the target N events so that while condition is met
            N_events_target=counterevents
        
        
    if plotcheck==True:
        #plt.hist(gwsras, alpha=0.5, color='red', density=True, label='Observed events')
        plt.hist(rainjs, alpha=0.5, color='green', density=True, label='Actual hosts')
        #plt.hist(ras, alpha=0.5, color='blue', density=True, label='Catalog galaxies')
        plt.legend()
        plt.xlabel('ra')
        plt.savefig(EventsFile[:-5]+"/Ra_Gw-Hosts_hist.pdf")
        plt.close()
        #plt.hist(gwsdecs, alpha=0.5, color='red', density=True, label='Observed events')
        plt.hist(decinjs, alpha=0.5, color='green', density=True, label='Actual hosts')
        #plt.hist(decs, alpha=0.5, color='blue', density=True, label='Catalog galaxies')
        plt.legend()
        plt.xlabel('dec')
        plt.savefig(EventsFile[:-5]+"/Dec_Gw-Hosts_hist.pdf")
        plt.close()
        #plt.hist(gwsdls, alpha=0.5, color='red', density=True, label='Observed events')
        plt.hist(dlinjs, alpha=0.5, color='green', density=True, label='Actual hosts')
        #plt.hist(zs*c/H0_true, alpha=0.5, color='blue', density=True, label='Catalog galaxies')
        plt.xlabel('dl')
        plt.legend()
        plt.savefig(EventsFile[:-5]+"/Dl_Gw-Hosts_hist.pdf")
        plt.close()
    if N_events_target<N_events:
        print("Only managed to subsample a total of ", N_events_target)
        print("Consider renaming accordingly the Eventsfile ", EventsFile)
    
    return EventsFile


def GenerateLowerDldetfromHigher(dldet, EventsFiletosubsample, eventsfiletowrite, StartingNevents=250):
    rainjs=[]
    decinjs=[]
    dlinjs=[]

    ragws=[]
    decgws=[]
    dlgws=[]

    rasamps=[]
    decsamps=[]
    dlsamps=[]

    with h5py.File(EventsFiletosubsample, "r") as f:
        for i in range(StartingNevents):
            if f[str(i)]["dl_gw"][()]<dldet:

                rainjs.append(f[str(i)]["ra_inj"][()])
                decinjs.append(f[str(i)]["dec_inj"][()])
                dlinjs.append(f[str(i)]["dl_inj"][()])

                ragws.append(f[str(i)]["ra_gw"][()])
                decgws.append(f[str(i)]["dec_gw"][()])
                dlgws.append(f[str(i)]["dl_gw"][()])

                rasamps.append(f[str(i)]["ra_samps"][()])
                decsamps.append(f[str(i)]["dec_samps"][()])
                dlsamps.append(f[str(i)]["dl_samps"][()])

    NewNevents=len(rainjs)

    with h5py.File(eventsfiletowrite, "w") as file:
        for i in range(NewNevents):
            grp=file.create_group(str(i))
            grp.create_dataset("ra_samps", data=rasamps[i])
            grp.create_dataset("dec_samps", data=decsamps[i])
            grp.create_dataset("dl_samps", data=dlsamps[i])
            grp.create_dataset("ra_inj", data=rainjs[i])
            grp.create_dataset("dec_inj", data=decinjs[i])
            grp.create_dataset("dl_inj", data=dlinjs[i])
            grp.create_dataset("ra_gw", data=ragws[i])
            grp.create_dataset("dec_gw", data=decgws[i])
            grp.create_dataset("dl_gw", data=dlgws[i])
    
    return NewNevents

"""def CreatePdet(CatalogFullName):
    with h5py.File(CatalogFullName, 'r') as f:
        ras=f['ra'][()]
        decs=f['dec'][()]
        zs=f['z'][()]
        ms=f['m'][()]"""
    
