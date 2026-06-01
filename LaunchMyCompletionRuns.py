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
                                                                                                                                                  
path_to_run_direc="/home/rbarbieri/ClusteringProject/"

input = int(sys.argv[1])
NrunsCompletion=1000 #this is Nrunpercomp when doing completion times the length of different completenesses used

SetupAndEvents=True
humantitle="sigmadl0.2_sigmaradec0.4_" #this is added after an automatic specified depending on runmode and correlation function



RunMode="CompleteCatalog" #Choose between "CompleteCatalog" (Use the complete catalog, don't even consider incompleteness), "LIGO" (account for incompleteness using usual LIGO analysis), "ClusteredCompletion" (completing the catalog)
if RunMode!= "CompleteCatalog" and RunMode!="LIGO" and RunMode!="ClusteredCompletion":
    print("RunMode not known or misspelled")
    sys.exit()

everythingrunmode=True
if everythingrunmode==True:
    if input<NrunsCompletion:
        RunMode="ClusteredCompletion"
        CorrelationFunction="fit"
        humantitle="FullCosmoRuns_corrfit_"+humantitle
        
    if input>=NrunsCompletion and input<2*NrunsCompletion:
        input=input-NrunsCompletion
        RunMode="ClusteredCompletion"
        CorrelationFunction="zero"
        humantitle="FullCosmoRuns_corrzero_"+humantitle
    
    if input==2*NrunsCompletion:
        input=input-2*NrunsCompletion
        RunMode="CompleteCatalog"
        CorrelationFunction="zero"
        humantitle="CompleteCatalogReference_"+humantitle

    if input>2*NrunsCompletion:
        input=input-2*NrunsCompletion-1
        RunMode="LIGO"
        CorrelationFunction="zero"
        humantitle="LIGOIncompletenessReference_"+humantitle

print("Input", input)
print("RunMode", RunMode)
print("CorrFunc", CorrelationFunction)
print("humantitle", humantitle)
####################Cosmo params####################################
linear=True
Omega_m=0.25
H0_true=70.
####################schechter params##################################
assumed_band="r"
##################Catalog Params#####################################
differentcompletenesses=True
allcomps=[0.01,0.1,0.2,0.5,1,2,5,10,20,50]

if RunMode=="ClusteredCompletion":
    Nrunpercomp=100
else:
    Nrunpercomp=1

morerunsoffset=0 #if you want to do more runs with exact same parameters/titles of run already done, use this to offset the realization number by the amount of old realizations

if differentcompletenesses==True:
    comp_index=int(input/Nrunpercomp)
    completenessfraction=allcomps[comp_index]
    print("Comp ", completenessfraction)
    realizationnumber=input-comp_index*Nrunpercomp
else:
    completenessfraction=90
    realizationnumber=input

realizationnumber=realizationnumber+morerunsoffset

CatalogFullName="Catalogs/Micev1WithMTrueScratch_newms-r.hdf5"
CatalogCutName="Catalogs/Micev1WithMTrueScratch_newms-r"+str(completenessfraction)+".hdf5"
CatalogFull=galaxyCatalog(CatalogFullName)
fullzs=CatalogFull.z
fullras=CatalogFull.ra
fulldecs=CatalogFull.dec

CatalogCut=galaxyCatalog(CatalogCutName)
cutzs=CatalogCut.z
cutras=CatalogCut.ra
cutdecs=CatalogCut.dec

zmax=0.1
zpriortouse="uniform"
###################################################################

###################Events params###############################
dldets=[100,150,200,300,400]
dl_distr="gaussian-sigmad"
#dl_det_threshold=200
N_events=1000
linear_data_gen=True
sigma_dl_prop_ass=0.2
sigma_ra_ass=0.4
sigma_dec_ass=sigma_ra_ass

useglobalpdl=False
globalpdlfile=None
fullanglesky=False
radec_distr='gaussian'
withinbounds=False #Enforce that ra and dec are within catalog bounds?
weightedcatevents=False
centered=False

#SpecifiedEventDirec=path_to_run_direc+"FullCosmoRuns_corrfit_recompFalse_resol50_Micev1WithMTrueScratch_newms-r_R0/Events/"
#SpecifiedEventDirec=path_to_run_direc+"FullCosmoRuns-NewEvents_Poisson_100reals_corrfit_recompFalse_resol100_Micev1WithMTrueScratch_newms-r_R0/Events/"
#SpecifiedEventDirec=path_to_run_direc+"FullCosmoRuns_sigmadl0.2_Poisson_100reals_corrzero_recompFalse_resol100_Micev1WithMTrueScratch_newms-r_R0/Events/"
#SpecifiedEventDirec=path_to_run_direc+"FullCosmoRuns_sigmadl0.2-rerunwnewevents_Poisson_100reals_corrzero_recompFalse_resol100_Micev1WithMTrueScratch_newms-r_R0/Events/"
#SpecifiedEventDirec=path_to_run_direc+"FullCosmoRuns_sigmaradec0.2_corrfit_Micev1WithMTrueScratch_newms-r_R0/Events/"
#SpecifiedEventDirec=path_to_run_direc+"FullCosmoRuns_corrfit_sigmadl0.2_Micev1WithMTrueScratch_newms-r_R0/Events/"
#SpecifiedEventDirec=path_to_run_direc+"FullCosmoRuns_corrfit_sigmadl0.05_sigmaradec0.05_Micev1WithMTrueScratch_newms-r_R0/Events/"
SpecifiedEventDirec=None
#####################################################################

##################clustering params################################
H0_assumed=70. #WEIRD VALUE CHANGE BACK
r0=5.4 #Mpc #NORMAL VALUE IS 5.4
gamma=1.8 #NORMAL VALUE IS 1.8, MODIFY AS SOON AS YOU START RUN
rmin=0.01
rmax=10
if everythingrunmode==False:
    CorrelationFunction="zero"

resol=100
recomputeclusteringprobs=False

numberofgalsperpix="poisson" #choose between "normal" (estimated missing gals is exactly what I add back), "poisson" (estimated is rates for a poisson distirbution) or "weight" (estimated used as probability weight)
renormrateasgals=True
Npointavaragelowcount=None
startatzero=False
puttinggalaxiesbyshell=True
if recomputeclusteringprobs==True:
    puttinggalaxiesbyshell=False #incompatible with recomputeclusteringprobs=True
if CorrelationFunction=="zero":
    skippixelizationforpopulation=True #this is mostly a test, populating a whole redshift shell instead of a pixel when correlation function is zero
else:
    skippixelizationforpopulation=False
###################################################################

#####################Cosmo anal params##############################
if RunMode=="CompleteCatalog" or RunMode=="ClusteredCompletion":
    completeness=True    #Assume that the catalog is complete? If using LIGO incompleteness, this should be false. True in all other cases
else:
    completeness=False

weightedcatalog=False
zmaxtouse=zmax

H0min=65
H0max=75
H0arr=np.linspace(H0min,H0max,101) #prior

linear=linear_data_gen#Assuming to analyze datas with same assumption as data generation
weighted=False #either luminosity, custom (in case provide weights), or False
weights=None
assumed_band='r'
populationmethod=False
clustercompletion=False
dimensions=3
pxdirect=True
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
precomputedoutsidepdet=True
precomputedpGpdet=True
saveprecomputes=True
############################################################################

################################################################################START OF CODE########################################################################################
eventsrealization = 0
direcname=path_to_run_direc+humantitle+CatalogFullName[9:-5]+"_R"+str(eventsrealization)+"/"
if os.path.exists(direcname)==False:
    os.mkdir(direcname)

#####################################################################SETUP DIRECS##############################################################################################################
CorrectedCatalogsdirec=direcname+"CorrectedCatalogs/"
if os.path.exists(CorrectedCatalogsdirec)==False:
    os.mkdir(CorrectedCatalogsdirec)

CosmoAnalParentDirec=direcname+"AnalsCosmo/"
if os.path.exists(CosmoAnalParentDirec)==False:
    os.mkdir(CosmoAnalParentDirec)

CorrectedPdetsDirec=direcname+"PdetsCorrectedCatalogs/"
if os.path.exists(CorrectedPdetsDirec)==False:
    os.mkdir(CorrectedPdetsDirec)

if SpecifiedEventDirec==None:
    EventDirec=direcname+"Events/"
    if os.path.exists(EventDirec)==False:
        os.mkdir(EventDirec)
else:
    EventDirec=SpecifiedEventDirec
##############################################################################################################################################################################################

###############################################################################EVENT GENERATION######################################################################################
for dl_det_threshold in dldets:
    #print(dl_det_threshold)
    EventsFile=EventDirec+dl_distr+"_dldet-"+str(int(dl_det_threshold))+"_N"+str(N_events)+"_events.hdf5"
    """f=h5py.File(EventsFile, "r")
    for i in range(N_events):
        dlinj=f[str(i)]["dl_inj"][()]
        print(dlinj)
        if dlinj==0:
            print("DLINJ 0!!")"""
    #sys.exit()
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
if RunMode=="ClusteredCompletion":
    CatalogCorrectedName=CorrectedCatalogsdirec+"Completed_csi"+CorrelationFunction+"_"+str(CatalogCutName[9:-5])+"_Recomp"+str(recomputeclusteringprobs)+"_resol"+str(resol)+"_rand"+str(realizationnumber)+".hdf5"
    print(CatalogCorrectedName)
    CompletionClass=CatalogCompletion(galaxy_catalog=CatalogCut, Omega_m=Omega_m, linear=linear, assumed_band=assumed_band, zmax=zmax, zpriortouse=zpriortouse, H0_assumed=H0_assumed, r0=r0, gamma=gamma, rmin_corr=rmin, rmax_corr=rmax, CorrelationFunction=CorrelationFunction, resol=resol)
    rates=CompletionClass.CorrectCatalogHealpyFixedNgal013(Npointavaragelowcount=Npointavaragelowcount, newname=CatalogCorrectedName, numberofgalsperpix=numberofgalsperpix, renormrateasgals=renormrateasgals, CatalogFullzs=fullzs, recomputeProbsclust=recomputeclusteringprobs, puttinggalaxiesbyshell=puttinggalaxiesbyshell, startatzero=startatzero, skipshellpixelizationforzerocorr=skippixelizationforpopulation)
if RunMode=="CompleteCatalog":
    CatalogCorrectedName=CatalogFullName
if RunMode=="LIGO":
    CatalogCorrectedName=CatalogCutName

#######################################################################COSMO ANALYSIS WITH CORRECTED CAT#####################################################################################
CatalogCorr=galaxyCatalog(CatalogCorrectedName)
print(CatalogCorrectedName)
################Precompute Pdets##########################################

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
    if completeness==True:
        pathtoprecomputedinsidepdet=CorrectedPdetsDirec+"insidepdet_zmaxass"+str(zmaxtouse)+"_"+CatalogCorrectedName[len(CorrectedCatalogsdirec):-5]+"_lin-"+str(linear_data_gen)+"_dl-distr-"+dl_distr+"_dldetused"+str(dl_det_threshold)+"_sigmaprop"+str(sigma_dl_prop_ass)+"_"+str(H0min)+"-"+str(H0max)+"-len"+str(len(H0arr))+".txt"
        insidepdet=PrecomputePdetOnLaunch(precomputeclass=precomputeclass, pdettype="inside", pathtoprecomputedpdet=pathtoprecomputedinsidepdet, saveprecomputes=saveprecomputes, smoothedcatalog=smoothedcatalog, emptycatalog=emptycatalog, JonLikelihood=JonStyleLikelihood)
        print("inside pdet corr ", np.loadtxt(insidepdet))

    if completeness==False:
        print("Preparing inside pdet...")
        pathtoprecomputedinsidepdet=CorrectedPdetsDirec+"insidepdet_zmaxass"+str(zmaxtouse)+"_"+CatalogCutName[9:-5]+"_lin-"+str(linear_data_gen)+"_dl-distr-"+dl_distr+"_dldetused"+str(dl_det_threshold)+"_sigmaprop"+str(sigma_dl_prop_ass)+"_"+str(H0min)+"-"+str(H0max)+"-len"+str(len(H0arr))+".txt"
        insidepdet=PrecomputePdetOnLaunch(precomputeclass=precomputeclass, pdettype="inside", pathtoprecomputedpdet=pathtoprecomputedinsidepdet, saveprecomputes=saveprecomputes, smoothedcatalog=smoothedcatalog, emptycatalog=emptycatalog, JonLikelihood=JonStyleLikelihood)
        print("inside pdet LIGO ", np.loadtxt(insidepdet))

        print("Preparing outside pdet...")
        pathtoprecomputedoutsidepdet=CorrectedPdetsDirec+"outsidepdet_zmaxass"+str(zmaxtouse)+"_"+CatalogCutName[9:-5]+"_lin-"+str(linear_data_gen)+"dl-distr-"+dl_distr+"_zprior-"+str(zpriortouse)+"_dldetused"+str(dl_det_threshold)+"_sigmaprop"+str(sigma_dl_prop_ass)+"_"+str(H0min)+"-"+str(H0max)+"-len"+str(len(H0arr))+".txt"
        outsidepdet=PrecomputePdetOnLaunch(precomputeclass=precomputeclass, pdettype="outside", pathtoprecomputedpdet=pathtoprecomputedoutsidepdet, saveprecomputes=saveprecomputes, smoothedcatalog=smoothedcatalog, emptycatalog=emptycatalog, JonLikelihood=JonStyleLikelihood)
        print("outside pdet LIGO ", np.loadtxt(outsidepdet))

        print("Preparing pGpdet...")
        pathtoprecomputedpGpdet=CorrectedPdetsDirec+"pGpdet_zmaxass"+str(zmaxtouse)+"_"+CatalogCutName[9:-5]+"_lin-"+str(linear_data_gen)+"_assband-"+assumed_band+"_lin-"+str(linear)+"_dl-distr-"+dl_distr+"_zprior-"+str(zpriortouse)+"_dldetused"+str(dl_det_threshold)+"_sigmaprop"+str(sigma_dl_prop_ass)+"_"+str(H0min)+"-"+str(H0max)+"-len"+str(len(H0arr))+".txt"
        pGpdet=PrecomputePdetOnLaunch(precomputeclass=precomputeclass, pdettype="pG", pathtoprecomputedpdet=pathtoprecomputedpGpdet, saveprecomputes=saveprecomputes, smoothedcatalog=smoothedcatalog, emptycatalog=emptycatalog, JonLikelihood=JonStyleLikelihood)
        print("pG pdet LIGO ", np.loadtxt(pGpdet))

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
    EventsFiletouse=EventsFile.replace("_dldet-"+str(dldets[-1]), "_dldet-"+str(dl_det_threshold)) 
    insidepdettouse=insidepdet.replace("_dldetused"+str(dldets[-1]), "_dldetused"+str(dl_det_threshold))
    if completeness==False:
        outsidepdettouse=outsidepdet.replace("_dldetused"+str(dldets[-1]), "_dldetused"+str(dl_det_threshold))
        pGpdettouse=pGpdet.replace("_dldetused"+str(dldets[-1]), "_dldetused"+str(dl_det_threshold))
    else:
        outsidepdettouse=None
        pGpdettouse=None
        
    if RunMode=="ClusteredCompletion":
        CosmoAnalDirec=CosmoAnalParentDirec+CatalogCorrectedName[len(CorrectedCatalogsdirec):-5]+"_dldet"+str(dl_det_threshold)+"/"
    else:
        CosmoAnalDirec=CosmoAnalParentDirec+CatalogCorrectedName[9:-5]+"_dldet"+str(dl_det_threshold)+"/"
    
    if os.path.exists(CosmoAnalDirec)==False:
        os.mkdir(CosmoAnalDirec)
    print("dldet ", dl_det_threshold)

    for EventNumber in EventNumbers:
        print("EVENT "+str(EventNumber))
        EventNumber=str(EventNumber)
        if JonStyleLikelihood==False:
            likelihoodclass=likelihood(EventNumber=EventNumber, GW_data=EventsFiletouse, galaxy_catalog=CatalogCorr, precomputedinsidepdet=precomputedinsidepdet, precomputedoutsidepdet=precomputedoutsidepdet, precomputedpGpdet=precomputedpGpdet, insidepdet=insidepdettouse, outsidepdet=outsidepdettouse, pGpdet=pGpdettouse, Omega_m=Omega_m, linear=linear, weighted=weighted, weights=weights, assumed_band=assumed_band, pdettype=pdetstyle, sigmaprop=sigma_dl_prop_ass, sigmaradec=sigma_ra_ass, dl_det=dl_det_threshold, whole_sky_cat=whole_sky_cat, weightedcatalog=weightedcatalog, directpx=pxdirect, directradec=radecdirect, dl_distr=dl_distr, summationtype=summationtype, smoothedcatalog=smoothedcatalog, normalizedgaussiansigmad=normalizedgaussiansigmad, zpriortouse=zpriortouse, zmax=zmaxtouse, saveprecomputes=saveprecomputes)
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
                        