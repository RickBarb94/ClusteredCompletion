import matplotlib.pyplot as plt
import imageio
import numpy as np
#import seaborn as sns
import os
import sys
import h5py

planck_h = 0.6774*100
sigma_planck_h = 0.0062*100
riess_h = 0.7324*100
sigma_riess_h = 0.0174*100
#c=sns.color_palette('colorblind')




#bands=["r", "FitActualLinear", "FitActual", "Actualr", "ActualrLinear", "FitFromOgCatActual"]
#linears=["False", "True"]
#pdets=["None"]#, "old-clustO2PSD_BNS_Nsamps20000_linearTrue_basicTrue_Omega_m0.25.p"]
#comps=["1", "5", "20"]
#zweights=["FalseFalse", "TrueFalse", "TrueTrue"]
#version="013"

#halfredshift=False
"""
dl_det_threshold=400
dimensions=3
dl_distr="gaussian-sigmad" #choose between 'gaussian' and gaussian-sigmad. Likelihood to generate the data (in distance)
completeness=True
gwcosmopdet='theoretic'
centered=False #Default is False. If true, the samples are always centered around the host. If false, center is extracted from a gaussian
#Eventhumantitle="SamplesCenteredonHost"
#Eventhumantitle="UniformCatalog"
Eventhumantitle='zeroedgecuts'

if dl_distr=="gaussian-sigmad":
        EventsFile="EventSamples/gaussin-sigmad_samples_"+Eventhumantitle+"_dldet-"+str(dl_det_threshold)+".hdf5"
elif dl_distr=="gaussian":
        EventsFile="EventSamples/gaussian-gaussian_samples_"+Eventhumantitle+"_dldet-"+str(dl_det_threshold)+".hdf5"
elif dl_distr=="F2Y-samples":
        EventsFile="EventSamples/F2Y_samples_linearTrueclosestrescaled.hdf5"


eventstitle=EventsFile[13:-5]
humantitle="Nosquareroot_MinusSign_Nodl2fromsamples_"
dirpath=humantitle+eventstitle+str(dimensions)+"D"+"_Complete"+str(completeness)+"_pdetgwcosmo"+str(gwcosmopdet)+"/"
"""
def MakeGif(dirpath, EventNumbers):
    H0arr=np.linspace(40,100,100)

    if os.path.isdir(dirpath)==False:
        print("Dir doesn't exist")
        sys.exit()
    else:
        
        gifdir=dirpath+"/GifDir"
        if os.path.isdir(gifdir)==False:
            os.mkdir(gifdir)
        


        ymax=0.
        ymin=0.

        combinedlikesfile=dirpath+"combined-normlikes.hdf5"
        print(combinedlikesfile)
        likelihood_comb_list = []
        with h5py.File(combinedlikesfile, 'r') as postf:
            
            for i in np.arange(EventNumbers):

                likelihood_comb=postf[str(i)][:]
                likelihood_comb_list.append(likelihood_comb)
            

        for k in range(len(likelihood_comb_list)):
            like=likelihood_comb_list[k]
            ymax=max(like)
            ymin=min(ymin, min(like))
            ymax=1.2*ymax   
            plt.plot(H0arr, like)
            plt.title(str(k)+ "Events")
            print(k)
            plt.axvline(planck_h,color='purple')
            plt.fill_betweenx([ymin,ymax],planck_h-2*sigma_planck_h,planck_h+2*sigma_planck_h,color='purple',alpha=0.2)
            plt.axvline(riess_h,color='green')
            plt.fill_betweenx([ymin,ymax],riess_h-2*sigma_riess_h,riess_h+2*sigma_riess_h,color='green',alpha=0.2)
            plt.axvline(70,ls='--', color='k',alpha=0.8)#, label = r'$H_0 = 70$ (km s$^{-1}$ Mpc$^{-1}$)')
            plt.axhline(0.)
            plt.xlim(H0arr[0],H0arr[-1])
            plt.ylim(ymin, ymax)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)',fontsize=16)
            plt.ylabel(r'$p(H_0)$ (km$^{-1}$ s Mpc)', fontsize=16)
            plt.savefig(gifdir+'/gifyplot'+str(k)+'.png')
            plt.close()
            #sys.exit()

        frames = []
        #for k in range(len(likelihood_comb_list)):
        #    image = imageio.v2.imread(gifdir+'/gifyplot'+str(k)+'.png')
        #    frames.append(image)
        #imageio.mimsave(dircontainer+dirpath+'/Gif.gif', # output gif
        #                frames,          # array of input frames
        #                duration = 50)         # optional: frames per second
        

#dirpath="BugRaDecKernelSolved_MinusSign_Nodl2fromsamples-gaussin-sigmad_samples_zeroedgecuts_sigmaprop-insteadof-fixed_dldet-100_N250_R5_3D_CompleteTrue_pdetgwcosmotheoretic/"
#EventNumbers=250

#MakeGif(dirpath, EventNumbers)