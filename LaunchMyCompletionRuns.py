#!/usr/bin/env python

import numpy as np
import scipy
import scipy.constants
import h5py
import matplotlib.pyplot as plt
import matplotlib
import sys
import os
import pickle
import time
import shutil

from scipy.interpolate import splev, splrep
from scipy.stats import gaussian_kde

from ScratchGwcosmoFunctions.FunctionsSimulateData import DataGeneration, DumpEventsLikelihoods, DumpEventsBreakdowns, GenerateLowerDldetfromHigher
from ScratchGwcosmoFunctions.FunctionsCatalog import galaxy, galaxyCatalog, galaxyweighted, galaxyCatalogweighted
from ScratchGwcosmoFunctions.FunctionsScratchGwcosmo import likelihood, likelihoodJon, ztoZshell, PrecomputingClass

from ScratchGwcosmoFunctions.ClusteringCompletionFunctions import CatalogCompletion

                                                                                                                                                    
path_to_run_direc="./"

input = int(sys.argv[1])

SetupAndEvents=False #if true, the code generates only the events and prepares the directories. Useful for running multiple realizations in parallel on a cluster using the same set of events.
humantitle="ClusteredCompletion_"
####################Cosmo params####################################
linear=True
Omega_m=0.25
H0_true=70.
####################schechter params##################################
assumed_band="r"

##################Catalog Params#####################################
completenessfraction=10 #global/total completeness fraction to be used

#load complete catalog
CatalogFullName="Catalogs/Micev1WithMTrueScratch_newms-r.hdf5"
CatalogFull=galaxyCatalog(CatalogFullName)
fullzs=CatalogFull.z
fullras=CatalogFull.ra
fulldecs=CatalogFull.dec

##load incomplete catalog
CatalogCutName="Catalogs/Micev1WithMTrueScratch_newms-r"+str(completenessfraction)+".hdf5"
CatalogCut=galaxyCatalog(CatalogCutName)
cutzs=CatalogCut.z
cutras=CatalogCut.ra
cutdecs=CatalogCut.dec

zmax=0.1 #redshif limit of the catalog
zpriortouse="uniform" #assumed prior, used if using LIGO style incompleteness or when filling the catalog in very empty redshift shells
###################################################################

###################Events params###############################
dldets=[100,150,200,300,400] #loop over these detection thresholds
dl_distr="gaussian-sigmad"  #type of distance posterior for GW events
radec_distr='gaussian'        #type of angle posterior
#dl_det_threshold=200
N_events=1000  #number of gw events
linear_data_gen=True    #assume linear cosmology?
sigma_dl_prop_ass=0.1    #distance error uncertainty
sigma_ra_ass=0.1        #angle error uncertainty
sigma_dec_ass=0.1


##all these variable are mostly just test for weird variants, safe to be ignored
useglobalpdl=False 
globalpdlfile=None
fullanglesky=False
withinbounds=False #Enforce that ra and dec are within catalog bounds?
weightedcatevents=False
centered=False
#####################################################################

##################clustering params################################
H0_assumed=70. #this is an assumed value of H0 that makes calculation easier code-wise, but has no impact on final result.
r0=5.4 #Mpc #NORMAL VALUE IS 5.4 #scale correlation function parameter
gamma=1.8 #NORMAL VALUE IS 1.8 #power correlation function parameter
rmin=0.01 #min and max limit of correlation function
rmax=10

resol=100  #how many redshift shell to split the catalog into?
scale=5. # relev pixels/gals for correlation function calculation are ones that are scale*R0 away from point considered
CorrelationFunction="fit"


recomputeclusteringprobs=False #after putting back a galaxies, do I recompute the clustering probabilities?
#mostly variables for weird variants
rateasnumberofgalsperpix=True
renormrateasgals=True
Npointavaragelowcount=None
startatzero=False
puttinggalaxiesbyshell=True
if recomputeclusteringprobs==True:
    puttinggalaxiesbyshell=False #incompatible with recomputeclusteringprobs=True
###################################################################

#####################Cosmo anal params##############################
completecatalog=True #is the catalog used in the cosmo analysis complete?
completeness=True #should the code assume the catalog is ocmplete?
weightedcatalog=False
zmaxtouse=zmax

H0min=65 #H0 array
H0max=75
H0arr=np.linspace(H0min,H0max,101)

linear=linear_data_gen#Assuming to analyze datas with same assumption as data generation
weighted=False #either luminosity, custom (in case provide weights), or False
weights=None
assumed_band='r'
populationmethod=False
dimensions=3 #used to do 1d vs 3d analysis
pxdirect=True #skip samples generation
radecdirect=pxdirect
pdetstyle="theoretic"#True#"cube" #choose between cube, True (gwcosmopdet) or theoretic (Heaviside step function pdet, equation 22 from hitchhikers) or "ones"
summationtype="None"#"logsumexp" #choose between logsumexp and  None
smoothedcatalog=useglobalpdl
JonStyleLikelihood=False
normalizedgaussiansigmad=False
pdetrenormtosimul=False
emptycatalog=False

if smoothedcatalog==True:
    z_bins=np.linspace(min(fullzs), max(fullzs), 1001)
    dz_bins=(max(fullzs)-min(fullzs))/1000
    z_array=np.linspace(min(fullzs)+dz_bins/2, max(fullzs)-dz_bins/2, 1000)
    z_vals=np.histogram(fullzs, bins=z_bins)[0]

whole_sky_cat=fullanglesky
precomputedinsidepdet=True
precomputedoutsidepdet=False
precomputedpGpdet=False
saveprecomputes=True
############################################################################

################################################################################START OF ACTUAL CODE########################################################################################
eventsrealization = 0 #change number to get a new gw event realization

#SETUP directories
# 
direcname=path_to_run_direc+humantitle+CatalogFullName[9:-5]+"_R"+str(eventsrealization)+"/"
if os.path.exists(direcname)==False:
    os.mkdir(direcname)

CorrectedCatalogsdirec=direcname+"CorrectedCatalogs/"
if os.path.exists(CorrectedCatalogsdirec)==False:
    os.mkdir(CorrectedCatalogsdirec)

CosmoAnalParentDirec=direcname+"AnalsCosmo/"
if os.path.exists(CosmoAnalParentDirec)==False:
    os.mkdir(CosmoAnalParentDirec)

CorrectedPdetsDirec=direcname+"PdetsCorrectedCatalogs/"
if os.path.exists(CorrectedPdetsDirec)==False:
    os.mkdir(CorrectedPdetsDirec)

EventDirec=direcname+"Events/"
if os.path.exists(EventDirec)==False:
    os.mkdir(EventDirec)
##############################################################################################################################################################################################

###############################################################################EVENT GENERATION######################################################################################
for dl_det_threshold in dldets:
    EventsFile=EventDirec+dl_distr+"_dldet-"+str(int(dl_det_threshold))+"_N"+str(N_events)+"_events.hdf5"
    if linear_data_gen==False:
        EventsFile=EventsFile.replace("_dldet-", "_LinearFalse_dldet-")
    print(EventsFile)
    if os.path.exists(EventsFile)==False:
        DataGenerationClass=DataGeneration(CatalogFullName=CatalogFullName, dl_det_threshold=dl_det_threshold, H0_true=H0_true, linear_data_gen=linear_data_gen, Omega_m=Omega_m, sigma_dl_prop_ass=sigma_dl_prop_ass, sigma_ra_ass=sigma_ra_ass, sigma_dec_ass=sigma_dec_ass, fullanglesky=fullanglesky, radec_distr=radec_distr, dl_distr=dl_distr, useglobalpdl=useglobalpdl, globalpdlfile=globalpdlfile, weightedcatevents=weightedcatevents)
        if useglobalpdl==True:
            DataGenerationClass.ProduceAndSaveSamples_fromglobalpDl(targetfile=EventsFile, N_events=N_events, plotcheck=False)
        else:
            DataGenerationClass.ProduceAndSaveSamples(targetfile=EventsFile, N_events=N_events, plotcheck=False, centered=centered, withinbounds=withinbounds) 
        print("Done creating the samples! Now time to analyze them :)")
    else:
        print("EventsAlreadyExists, not generating events!") 
if SetupAndEvents==True:
    sys.exit()
###############################################################################################################################################################################################

#######################################################################CLUSTERING COMPLETION###################################################################################################
realizationnumber=input

print("Recompute ", recomputeclusteringprobs)
print("By shell ", puttinggalaxiesbyshell)

CatalogCorrectedName=CorrectedCatalogsdirec+"Completed_csi"+CorrelationFunction+"_"+str(CatalogCutName[9:-5])+"_Recomp"+str(recomputeclusteringprobs)+"_resol"+str(resol)+"_rand"+str(realizationnumber)+".hdf5"
print(CatalogCorrectedName)

if os.path.exists(CatalogCorrectedName)==False:
    CompletionClass=CatalogCompletion(galaxy_catalog=CatalogCut, Omega_m=Omega_m, linear=linear, assumed_band=assumed_band, zmax=zmax, zpriortouse=zpriortouse, H0_assumed=H0_assumed, r0=r0, gamma=gamma, rmin=rmin, rmax=rmax, scale=scale, CorrelationFunction=CorrelationFunction, resol=resol)
    rates=CompletionClass.CorrectCatalogHealpyFixedNgal013(Npointavaragelowcount=Npointavaragelowcount, newname=CatalogCorrectedName, rateasnumberofgalsperpix=rateasnumberofgalsperpix, renormrateasgals=renormrateasgals, CatalogFullzs=fullzs, recomputeProbsclust=recomputeclusteringprobs, puttinggalaxiesbyshell=puttinggalaxiesbyshell, startatzero=startatzero)
else:
    print("Already exists! ", CatalogCorrectedName)


#######################################################################COSMO ANALYSIS WITH CORRECTED CAT#####################################################################################
#Now do cosmological analysis using the newly corrected catalog, assuming that to be a complete catalog
CatalogCorr=galaxyCatalog(CatalogCorrectedName)
################Precompute Pdets##########################################
#pdets are the same for every event, so precompute them once at the start
def PrecomputePdetOnLaunch(precomputeclass, pdettype, pathtoprecomputedpdet, saveprecomputes, smoothedcatalog, emptycatalog, JonLikelihood):
    if smoothedcatalog==True:
        return("Smoothed catalog not implemented in precompute launch function!")
        sys.exit()
    if emptycatalog==True:
        return("Empty catalog not implemented in precompute launch function!")
        sys.exit()
    if JonLikelihood==True:
        return("Jon style likelihood not implemented in precompute launch function!")
        sys.exit()

    if saveprecomputes==True:      
        if os.path.exists(pathtoprecomputedpdet)==False:
            print("Precomputing the ", pdettype, " pdet for this run!")
            if smoothedcatalog==True:
                sys.exit()
                #pdettosave=pD_inside_theoretic_precompute_smoothed(H0arr, z_array, z_vals, dl_det_threshold, sigma_dl_prop_ass, linear_data_gen, Omega_m, pdettype=pdettype, dl_distr=dl_distr)
            elif emptycatalog==True:
                return("Empty catalog not implemented in precompute launch function!")
                sys.exit()
                #pdettosave=np.ones(len(H0arr))
            else:
                #pdettosave=pD_inside_theoretic_precompute(H0arr, CatalogCut, dl_det_threshold, sigma_dl_prop_ass, linear_data_gen, Omega_m, pdettype=pdettype, dl_distr=dl_distr, weightedcat=weightedcatalog)
                if pdettype=="inside":
                    pdettosave=precomputeclass.pD_inside_theoretic_precompute(H0=H0arr)
                elif pdettype=="outside":
                    pdettosave=precomputeclass.pD_outside_theoretic_precompute(H0=H0arr)
                elif pdettype=="pG":
                    pdettosave=precomputeclass.pG_theoretic_precompute(H0=H0arr)
            if saveprecomputes==True:
                np.savetxt(fname=pathtoprecomputedpdet, X=pdettosave)
        else:
            print("Using precomputed insidepdet, already calculated!")
        pdet=pathtoprecomputedpdet
    else:
        if smoothedcatalog==True:
            sys.exit()
            #pdettosave=pD_inside_theoretic_precompute_smoothed(H0arr, z_array, z_vals, dl_det_threshold, sigma_dl_prop_ass, linear_data_gen, Omega_m, pdettype=pdettype, dl_distr=dl_distr)
        elif emptycatalog==True:
            sys.exit()
            #pdettosave=np.ones(len(H0arr))
        else:
            #pdettosave=pD_inside_theoretic_precompute(H0arr, CatalogCut, dl_det_threshold, sigma_dl_prop_ass, linear_data_gen, Omega_m, pdettype=pdettype, dl_distr=dl_distr, weightedcat=weightedcatalog)
            if pdettype=="inside":
                pdettosave=precomputeclass.pD_inside_theoretic_precompute(H0=H0arr)
            elif pdettype=="outside":
                pdettosave=precomputeclass.pD_outside_theoretic_precompute(H0=H0arr)
            elif pdettype=="pG":
                pdettosave=precomputeclass.pG_theoretic_precompute(H0=H0arr)
        pdet=pdettosave
    return pdet

print("Precomputing Pdets")
for dl_det_threshold in dldets:
    print("Dldet ", dl_det_threshold)
    precomputeclass=PrecomputingClass(galaxy_catalog=CatalogCorr, CatalogFull=CatalogFull, dl_det=dl_det_threshold, Omega_m=Omega_m, linear=linear, assumed_band=assumed_band, zmax=zmaxtouse, sigmaprop=sigma_dl_prop_ass, weightedcat=weightedcatalog, zpriortouse=zpriortouse, dl_distr=dl_distr, pdettype=pdetstyle, weighted=weighted)
    pathtoprecomputedinsidepdet=CorrectedPdetsDirec+"insidepdet_zmaxass"+str(zmaxtouse)+"_"+CatalogCorrectedName[len(CorrectedCatalogsdirec):-5]+"_lin-"+str(linear_data_gen)+"_dl-distr-"+dl_distr+"_dldetused"+str(dl_det_threshold)+"_sigmaprop"+str(sigma_dl_prop_ass)+"_"+str(H0min)+"-"+str(H0max)+"-len"+str(len(H0arr))+".txt"
    insidepdet=PrecomputePdetOnLaunch(precomputeclass=precomputeclass, pdettype="inside", pathtoprecomputedpdet=pathtoprecomputedinsidepdet, saveprecomputes=saveprecomputes, smoothedcatalog=smoothedcatalog, emptycatalog=emptycatalog, JonLikelihood=JonStyleLikelihood)
    print("inside pdet corr ", np.loadtxt(insidepdet))
    #these next 3 lines is just a test they are useless
    precomputeclassfull=PrecomputingClass(galaxy_catalog=CatalogFull, CatalogFull=CatalogFull, dl_det=dl_det_threshold, Omega_m=Omega_m, linear=linear, assumed_band=assumed_band, zmax=zmaxtouse, sigmaprop=sigma_dl_prop_ass, weightedcat=weightedcatalog, zpriortouse=zpriortouse, dl_distr=dl_distr, pdettype=pdetstyle, weighted=weighted)
    pathtoprecomputedinsidepdetfull=CorrectedPdetsDirec+"FULLTESTinsidepdet_zmaxass"+str(zmaxtouse)+"_"+CatalogCorrectedName[len(CorrectedCatalogsdirec):-5]+"_lin-"+str(linear_data_gen)+"_dl-distr-"+dl_distr+"_dldetused"+str(dl_det_threshold)+"_sigmaprop"+str(sigma_dl_prop_ass)+"_"+str(H0min)+"-"+str(H0max)+"-len"+str(len(H0arr))+".txt"
    insidepdetfull=PrecomputePdetOnLaunch(precomputeclass=precomputeclass, pdettype="inside", pathtoprecomputedpdet=pathtoprecomputedinsidepdetfull, saveprecomputes=saveprecomputes, smoothedcatalog=smoothedcatalog, emptycatalog=emptycatalog, JonLikelihood=JonStyleLikelihood)
    print("inside pdet full ", np.loadtxt(insidepdetfull))

outsidepdet=None
pGpdet=None
###########################################################################

###########################################################################
EventNumbers=[i for i in range(N_events)]


t0=time.time()

print("Running Cosmo Analysis")

#loop through detection thresholds
for dl_det_threshold in dldets:
    comblikelihood=np.ones(len(H0arr))
    comblikelihoodlist=[]
    likelihoodnormlist=[]
    datastodumplist=[]
    EventsFiletouse=EventsFile.replace("_dldet-"+str(dldets[-1]), "_dldet-"+str(dl_det_threshold)) #the EventsFile variable from the event generating bit will be set to last det threshold (usually 400), here I change it so everyone has different det threshold
    
    CosmoAnalDirec=CosmoAnalParentDirec+CatalogCorrectedName[len(CorrectedCatalogsdirec):-5]+"_dldet"+str(dl_det_threshold)+"/"
    if os.path.exists(CosmoAnalDirec)==False:
        os.mkdir(CosmoAnalDirec)
    print("dldet ", dl_det_threshold)
    #analyze all events
    for EventNumber in EventNumbers:
        print("EVENT "+str(EventNumber))
        EventNumber=str(EventNumber)
        if JonStyleLikelihood==False:
            likelihoodclass=likelihood(EventNumber=EventNumber, GW_data=EventsFiletouse, galaxy_catalog=CatalogCorr, precomputedinsidepdet=precomputedinsidepdet, precomputedoutsidepdet=precomputedoutsidepdet, precomputedpGpdet=precomputedpGpdet, insidepdet=insidepdet, outsidepdet=outsidepdet, pGpdet=pGpdet, Omega_m=Omega_m, linear=linear, weighted=weighted, weights=weights, assumed_band=assumed_band, pdettype=pdetstyle, sigmaprop=sigma_dl_prop_ass, sigmaradec=sigma_ra_ass, dl_det=dl_det_threshold, whole_sky_cat=whole_sky_cat, weightedcatalog=weightedcatalog, directpx=pxdirect, directradec=radecdirect, dl_distr=dl_distr, summationtype=summationtype, smoothedcatalog=smoothedcatalog, normalizedgaussiansigmad=normalizedgaussiansigmad, zpriortouse=zpriortouse, zmax=zmaxtouse, saveprecomputes=saveprecomputes)
            likelihoodevent,pxG,pDG,pGD, pxnG,pDnG,pnGD=likelihoodclass.likelihood(H0arr, complete=completeness, population=populationmethod, dimensions=dimensions)
            #print(likelihoodevent,pxG,pDG,pGD, pxnG,pDnG,pnGD)
            #print("pG ", pGD)
            #print(likelihoodevent)
            likelihoodnorm = likelihoodevent/np.trapz(likelihoodevent, H0arr)
            likelihoodnormlist.append(likelihoodnorm)
            comblikelihood *= likelihoodnorm#changed here, used to be likelihoodevent (not normalized)
            comblikelihood=comblikelihood/np.trapz(comblikelihood, H0arr)
            comblikelihoodlist.append(comblikelihood)
            datastodump=[likelihoodevent, pxG, pDG, pGD, pxnG, pDnG]
            datastodumplist.append(datastodump)
        elif JonStyleLikelihood==True:
            likelihoodclass=likelihoodJon(EventNumber=EventNumber, galaxy_catalog=CatalogCorr, GW_data=EventsFiletouse, dl_det=dl_det_threshold, sigmaprop=sigma_dl_prop_ass, sigmaradec=sigma_ra_ass, assumed_band=assumed_band, linear=linear, Omega_m=Omega_m, zmax=zmaxtouse, precomputedpdets=precomputedinsidepdet, insidepdetpath=insidepdet, outsidepdetpath=outsidepdet, completeness=completenessfraction/100, emptycatalogrun=emptycatalog, saveprecomputes=saveprecomputes)
            likelihoodevent,pxG, pDG, pGD, pxnG, pDnG, pnGD =likelihoodclass.likelihoodJon(H0arr, dimensions=dimensions)
            likelihoodnorm = likelihoodevent/np.trapz(likelihoodevent, H0arr)
            likelihoodnormlist.append(likelihoodnorm)
            comblikelihood *= likelihoodnorm#changed here, used to be likelihoodevent (not normalized)
            comblikelihood=comblikelihood/np.trapz(comblikelihood, H0arr)
            comblikelihoodlist.append(comblikelihood)
            datastodump=[likelihoodevent, pxG, pDG, pGD, pxnG, pDnG]
            datastodumplist.append(datastodump)
        plt.plot(H0arr, likelihoodnorm, ls='dashed', alpha=0.6)
        print(" ")

    #plot and save stuff
    t1=time.time()
    averagetime=(t1-t0)/len(EventNumbers)
    print("Total time for "+str(len(EventNumbers))+" events was "+str(t1-t0)+" s for an average of "+str(averagetime)+" s per event")
    plt.plot(H0arr, comblikelihood, color='black')
    plt.axvline(H0_true, color='red', ls='dashdot')
    plt.savefig(CosmoAnalDirec+"final-likes.pdf")
    plt.close()

    c_norm = matplotlib.colors.Normalize(vmin=0, vmax=len(EventNumbers))
    c_map  = matplotlib.cm.viridis

    # Scalar mappable of normalized array to colormap
    s_map  = matplotlib.cm.ScalarMappable(cmap=c_map, norm=c_norm)
    s_map.set_array([])

    DumpEventsBreakdowns(CosmoAnalDirec+"breakdowns.hdf5", datastodumplist)
    DumpEventsLikelihoods(CosmoAnalDirec+"combined-normlikes.hdf5", comblikelihoodlist)
    print(" ")
    print(" ")
    print(" ")
    print(" ")

#MakeGif(direc, EventNumbers=N_events)
                        