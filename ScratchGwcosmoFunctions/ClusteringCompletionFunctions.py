import numpy as np 
import scipy.constants
import time
import healpy as hp
import sys
from bisect import bisect_left
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import progressbar
from scipy.integrate import quad, dblquad
import h5py

from utilities.standard_cosmology import *
from utilities.schechter_function import *
from utilities.schechter_params import *

c=scipy.constants.c/1000
epsrelforFunctionsScratch=1.49e-4
epsabsforFunctionsScratch=0
r0fit=7.30627815
gammafit=1.53939311


def M_L(L):
    return -2.5*np.log10(L/3.0128e28)

def ThetatoDec(Theta):
	dec=np.pi/2-Theta
	return dec

def DectoTheta(dec):
	Theta=np.pi/2-dec
	return Theta

def ExtractSkyPos(ramin, ramax, decmin, decmax):
    ra = np.random.uniform(ramin,ramax)

    thetamin=DectoTheta(decmax)
    thetamax=DectoTheta(decmin)
    u=np.random.uniform()

    N=np.cos(thetamin)-np.cos(thetamax)
    theta=np.arccos(np.cos(thetamin)-N*u)
    dec=ThetatoDec(theta)
    return ra,dec

def Extractz(zmin, zmax):
	u=np.random.uniform(0,1)
	N=zmax**3/3-zmin**3/3
	z=(3*N*u+zmin**3)**(1./3.)
	return z

def SphereSectionVolume(ramin, ramax, decmin, decmax, zmin, zmax):
    thetamin=np.pi/2-decmax
    thetamax=np.pi/2-decmin      
    return 1./3.*(zmax**3-zmin**3)*(np.cos(thetamin)-np.cos(thetamax))*(ramax-ramin)

def GenerateSchechterSamples(assumed_band, Nsamples, H0_true):
    SchechterParamsClass = SchechterParams(assumed_band)
    alpha, Mstar_obs, Mmin_obs, Mmax_obs = SchechterParamsClass.values(assumed_band)
    Mstar=Mstar_obs + 5.*np.log10(H0_true/100.)
    Mmax=Mmax_obs + 5.*np.log10(H0_true/100.)
    L_star=L_M(Mstar)
    L_min=L_M(Mmax)
    Ms=M_L(simulate_schechter_distribution(alpha, L_star, L_min, Nsamples))

    return Ms

def simulate_schechter_distribution(alpha, L_star, L_min, N):
    """ 
    Generate N samples from a Schechter distribution, which is like a gamma distribution 
    but with a negative alpha parameter and cut off on the left somewhere above zero so that
    it converges.
    
    If you pass in stupid enough parameters then it will get stuck in a loop forever, and it
    will be all your own fault.
    
    Based on algorithm in http://www.math.leidenuniv.nl/~gill/teaching/astro/stanSchechter.pdf
    """
    output = []
    n=0
    while n<N:
        L = np.random.gamma(scale=L_star, shape=alpha+2, size=N)
        L = L[L>L_min]
        u = np.random.uniform(size=L.size)
        L = L[u<L_min/L]
        output.append(L)
        n+=L.size
    return np.concatenate(output)[:N]

def SkyCootoCart(ra, dec, dist):
	#I use this both knowing dist and not knowing dist but using z as a notion of distance in
	x=np.cos(dec)*np.cos(ra)
	y=np.cos(dec)*np.sin(ra)
	z=np.sin(dec)
	galvec=dist*np.array([x,y,z])
	return galvec

def CarttoSkyCoo(galvec):
	dist=np.linalg.norm(galvec)
	galvec=galvec/dist
	dec=np.arcsin(galvec[2])
	ra=np.arctan2(galvec[1], galvec[0])
	if ra<0:
		ra+=2*np.pi
	return ra, dec, dist

class CatalogCompletion(object):
    def __init__ (self, galaxy_catalog, Omega_m, linear, assumed_band, zmax, zpriortouse, H0_assumed, r0, gamma, rmin_corr, rmax_corr, resol=100, CorrelationFunction=None, fullskyorfraction="octave"):
        #cosmology params
        self.Omega_m=Omega_m
        self.linear=linear
        
        #catalog params
        if galaxy_catalog is not None:
            self.galaxy_catalog = galaxy_catalog #add hdf5 extraction and mth calculation
            self.mth = galaxy_catalog.mth(type="max")
            self.allz = galaxy_catalog.z
            self.allra = galaxy_catalog.ra
            self.alldec = galaxy_catalog.dec
            self.allm = galaxy_catalog.m
            self.radec_lims = galaxy_catalog.radec_lim
            if self.radec_lims[0]==0 and self.radec_lims[1]==0:
                if fullskyorfraction=="octave":
                    self.minra=0.
                    self.maxra=np.pi/2
                    self.mindec=0
                    self.maxdec=np.pi/2            
            else:
                self.minra=self.radec_lims[0]
                self.maxra=self.radec_lims[1]
                self.mindec=self.radec_lims[2]
                self.maxdec=self.radec_lims[3]

            self.nGal = len(self.allz)
             
        self.zmax=zmax
        self.zpriortouse=zpriortouse

        #Schechter Params
        self.assumed_band=assumed_band
        sp = SchechterParams(assumed_band)
        self.alpha = sp.alpha
        self.Mstar_obs = sp.Mstar
        self.Mobs_min = sp.Mmin
        self.Mobs_max = sp.Mmax

        self.cosmo = fast_cosmology(Omega_m=Omega_m, linear=linear)

        #Clustering params
        self.H0_assumed=H0_assumed
        self.r0=r0 #in MPC

        self.z0=self.r0*self.H0_assumed/c
        self.gamma=gamma #NORMAL VALUE IS 1.8, MODIFY AS SOON AS YOU START RUN
        self.rmin_corr=rmin_corr
        self.rmax_corr=rmax_corr
        self.zmax_corr=self.cosmo.z_dlH0(self.rmax_corr, H0_assumed)
        
        if CorrelationFunction=='csi':
            self.CorrFunction=self.csi
        elif CorrelationFunction=='plateud':
            self.CorrFunction=self.csi_plateud
        elif CorrelationFunction=='zero':
            self.CorrFunction=self.csi_zero
        elif CorrelationFunction=="corrfromcat":
            self.CorrFunction=CorrFromCat(corrfromcat)
        elif CorrelationFunction=="fit":
            self.CorrFunction=self.csi_fit
            self.r0=r0fit #in MPc
            self.z0=self.r0*self.H0_assumed/c
            self.gamma=gammafit #NORMAL VALUE IS 1.8, MODIFY AS SOON AS YOU START RUN
            
            
        else:
            print("Please choose a correlation function")
            sys.exit()

        #Completion params
        self.resol=resol


        z_hist=np.histogram(self.allz, bins=np.linspace(0,self.zmax, self.resol+1))
        self.z_hist_count=z_hist[0]
        self.z_hist_edges=z_hist[1]
        self.z_hist_means=np.array([(self.z_hist_edges[i]+self.z_hist_edges[i+1])/2 for i in np.arange(self.resol)])
        self.deltaz=self.z_hist_means[1]-self.z_hist_means[0]

        self.ResolPixNumberReferenceCum=np.array([sum(12*(i)**2 for i in np.arange(j+1)) for j in np.arange(resol+5)]) #this is used as reference later and computed here. resol+5 just so that I am sure I don't go out of bounds
        self.maxpixindex=self.ResolPixNumberReferenceCum[resol]
        self.totalPixelNumber=sum(12*i**2 for i in np.arange(self.resol+1)) #NSIDE pixel cannot be 0, so must start at 1 (for which I have 12 pixels), so loop is until arange +1

    def fz(self, z):
        # fraction of observed galaxies at redshift z
        H0ass=70. #It actually does not depend on H0, but easy to just put as a placeholder and not modify the basic functions
        Mmin = M_Mobs(H0ass,self.Mobs_min)
        Mmax = M_Mobs(H0ass,self.Mobs_max)
        
        def I(M):
            return SchechterMagFunction(H0=H0ass,Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)

        num = quad(I,Mmin, min(max(M_mdl(self.mth,self.cosmo.dl_zH0(z,H0ass)),Mmin),Mmax),epsabs=0,epsrel=1.49e-4)[0]
        den = quad(I, Mmin, Mmax, epsabs=0,epsrel=1.49e-4)[0]
        
        return num/den
    
    def RedshiftShell(self, z):
        """
        Assumes z_hist_means is sorted. Returns closest value to z.

        If two numbers are equally close, return the smallest number.
        """

        pos = bisect_left(self.z_hist_means, z)
        if pos == 0:
            return self.z_hist_means[0]
        if pos == len(self.z_hist_means):
            return self.z_hist_means[-1]
        before = self.z_hist_means[pos - 1]
        after = self.z_hist_means[pos]
        if after - z < z - before:
            return after
        else:
            return before
        
    def HealpyAngtoradec(self, theta, phi):
        #return ra and dec
        return phi, np.pi/2-theta

    def radectoHealpyCoo(self, ra, dec):
        #return phi and theta
        return ra, np.pi/2-dec
    
    def ztoNSIDE(self, z):
        #resolution defined as the NSIDE for the maximum redshift of 0.1
        return int(self.resol/self.zmax*z)+1 #See below for the +1, NSIDE cannot be 0

    def ztoBasePix(self, z):
        #given a resolution and a redshift, return the sum of pixels of previous redshift shells
        NSIDE=self.ztoNSIDE(z)
        return sum(12*i**2 for i in np.arange(NSIDE)) 

    def zindextoBasePix(self, zindex):
        #given a resolution and a redshift, return the sum of pixels of previous redshift shells
        return sum(12*i**2 for i in np.arange(zindex+1)) 

    def zindextoPixNumber(self, index):
        #given a redshift shell index, return the number of pixels in that redshift shell. Is index+1 because first index is 0 (duh) but NSIDE must be at least 1
        return 12*(index+1)**2
    
    def HealpytoSkyCoo(self, healpyindex, zindex):
        #print(zindex, healpyindex)
        z=self.z_hist_means[zindex]
        theta, phi = hp.pix2ang(zindex+1, healpyindex)
        ra, dec = self.HealpyAngtoradec(theta, phi)
        return [ra, dec, z]

    def HealpytoRaDec(self, healpyindex, zindex):
        #print(zindex, healpyindex)
        theta, phi = hp.pix2ang(zindex+1, healpyindex)
        ra, dec = self.HealpyAngtoradec(theta, phi)
        return [ra, dec]

    def PixIndexToCartCoo(self, index):
        SkyCoo=self.PixIndextoSkyCoo(index)
        dist=SkyCoo[2]*c/self.H0_assumed
        cartcoo=SkyCootoCart(SkyCoo[0], SkyCoo[1], dist)
        return cartcoo

    def PixIndextoSkyCoo(self, pixindex):
        zindex=self.PixIndextoZindex(pixindex)
        healpy_index=int(pixindex-self.zindextoBasePix(zindex))
        return self.HealpytoSkyCoo(healpyindex=healpy_index, zindex=zindex)

    def PixIndextoZindex(self, pixindex):
        #print(np.where(pixindex>ResolPixNumberReferenceCum))
        return np.where(pixindex>=self.ResolPixNumberReferenceCum)[0][-1]
    
    def AnalyticalSpherePointsDistance(self, pix1, pix2, type="distance"):
        phi1=pix1[0]
        phi2=pix2[0]
        theta1=np.pi/2-pix1[1]
        theta2=np.pi/2-pix2[1]
                
        if type=="distance":
            r1=self.cosmo.dl_zH0(pix1[2], H0=self.H0_assumed)
            r2=self.cosmo.dl_zH0(pix2[2], H0=self.H0_assumed)
        elif type=="redshift":
            r1=pix1[2]
            r2=pix2[2]

        return np.sqrt(r1**2+r2**2-2*r1*r2*( np.sin(theta1)*np.sin(theta2)*np.cos(phi1-phi2)+np.cos(theta1)*np.cos(theta2)))
    
    def BatchAnalyticalSpherePointsDistance(self, centerpixcoo, batchpixels, type='distance'):
        phicenter=centerpixcoo[0]
        thetacenter=np.pi/2-centerpixcoo[1]
        if type=="distance":
            rcenter=self.cosmo.dl_zH0(centerpixcoo[2], H0=self.H0_assumed)
        elif type=="redshift":
            rcenter=centerpixcoo[2]
        

        batchpixelscoo=np.array([self.PixIndextoSkyCoo(pixindex=pix) for pix in batchpixels])

        phisbatch=batchpixelscoo[:,0]
        thetasbatch=np.pi/2-batchpixelscoo[:,1]
        if type=="distance":
            rsbatch=self.cosmo.dl_zH0(batchpixelscoo[:,2], H0=self.H0_assumed)
        elif type=="redshift":
            rsbatch=batchpixelscoo[:,2]

        distances=np.sqrt(rcenter**2+rsbatch**2-2*rcenter*rsbatch*( np.sin(thetacenter)*np.sin(thetasbatch)*np.cos(phicenter-phisbatch)+np.cos(thetacenter)*np.cos(thetasbatch)))
        return distances

                    
    def csi(self, r):
        csi=(self.r0/r)**self.gamma 
        return csi 
    
    def csi_fit(self, r):
        csi=(r0fit/r)**gammafit
        return csi
    
    def csi_plateud(self, r, plateau=5):
        csi=(self.r0/r)**self.gamma
        if csi>plateau:
            return plateau
        else:
            return csi

    def csi_zero(self, r):
        #this is a function I use to turn off clustering.
        return 0.

    def CorrFromCat(self, txtFile):
        #function to get the correlation function csi from the catalog, via neighbours counting
        #instead of rom theory. txtFile is a file containing the value of the correlation function
        #at different radii. THen we do a fit and return it.
        print("Correlation function from catalog!")
        f=self.FitCorrfromtxt(txtFile)
        return f

    def FitCorrfromtxt(self, txtFile):
        rsold=[]
        epsiold=[]
        with open(txtFile) as f:
            for line in f:
                currentline=line.split(",")
                rsold.append(float(currentline[0]))
                epsiold.append(float(currentline[1]))
        rsold=np.array(rsold)
        if rsold[-1]<1.: #if correlation function given in redshifts, convert to distances
            if self.linear==True:
                rsold=rsold*self.H0_true/c #TODO check this expression
            
            elif self.linear==False:
                rsold=rsold*self.H0_true/c #TODO check this expression

        epsiold=np.array(epsiold) 
        f=interp1d(rsold, epsiold)
        return f

    def BinningGalaxiesHealpy(self):
        t0=time.perf_counter()
        pixs=np.zeros(self.totalPixelNumber)
        
        for i in np.arange(len(self.allz)):
            gal=self.galaxy_catalog.get_galaxy(i)
            z_shell=self.RedshiftShell(gal.z)
            phi, theta = self.radectoHealpyCoo(gal.ra, gal.dec)
            
            #print(ztoNSIDE(resol, z_shell))
            pix_shell_number=hp.ang2pix(self.ztoNSIDE(z_shell), theta, phi)
            
            pix_index=self.ztoBasePix(z_shell)+pix_shell_number
            if pix_index>len(pixs):
                print("PROBLEM")
                print("gal z = ", gal.z)
                print("Redshift shell = ", z_shell)
                print("Base pixels (sum of pixels of shells before) =", pix_index)
            pixs[pix_index]+=1
        print("Time for binning ", time.perf_counter()-t0)
        return pixs
    
    def ProbAtPixH0indv2Healpy(self, index, startatzero=False):
	
        closepixels=self.GetrelevPixelsHealpy(index)
        #print("This pixel has ", len(closepixels), " neighbours")
        #print("Evaluating clust prob at pix ", index)
        pix_coo= self.PixIndextoSkyCoo(index)#get coordinate of pixels in ra, dec, z
        if startatzero==True:
            prob=0.
        else:
            prob=1.

        
        distances=self.BatchAnalyticalSpherePointsDistance(centerpixcoo=pix_coo, batchpixels=closepixels, type='distance')#Get distances of closeby pixels
        
        for i, closepix_index in enumerate(closepixels):
            if self.pix_occup[closepix_index]>0:
                r=distances[i]
                if r<self.rmax_corr:#this if might be necessary for speed but I already choose only closeby pixels so maybe not
                    cor=self.CorrFunction(r)*self.pix_occup[closepix_index]
                    prob+=cor
                   
        raedgemin, raedgemax, decedgemin, decedgemax = self.NewGetPixelEdges(index) #compute pixel volume 
        chopix_ra, chopix_dec, chopix_z = self.PixIndextoSkyCoo(index)
        zedgemin=max(chopix_z-self.deltaz/2, 0.)
        zedgemax=min(chopix_z+self.deltaz/2, self.zmax)  
        
        raactualmin=max(self.minra, raedgemin)
        raactualmax=min(self.maxra, raedgemax)
        decactualmin=max(self.mindec, decedgemin)
        decactualmax=min(self.maxdec, decedgemax)

        PixelVolumeWithinCatalog=SphereSectionVolume(ramin=raactualmin, ramax=raactualmax, decmin=decactualmin, decmax=decactualmax, zmin=zedgemin, zmax=zedgemax)
        
        """if PixelVolumeWithinCatalog<0.:
            input("Volume less than zero!")
            print("ras ", raactualmin, raactualmax)
            print("decs ", decactualmin, decactualmax)
            print("zs ", zedgemin, zedgemax)
            input("Continue?")"""
            
            #print("Volume ", PixelVolumeWithinCatalog)
        #print("Pix lies at the edge!")
        #print("Coordinates ", chopix_ra, chopix_dec, chopix_z)
        #print("Actual volume is this fraction: ", f*100)
        #print("")
        return prob*PixelVolumeWithinCatalog #return clustering prob * volume, so that smaller pixels (usually at the edges) don't form overdensity of galaxies (same prob, smaller volume means higher density)
       
    
    def AssignPixelProbs(self, startatzero=False, readclustpfromfile=False, rateasnumberofgalsperpix=False, savefile=False):
        probs=np.zeros(self.totalPixelNumber)
        actualpixcount=0
        nonzeropixels=0
        if readclustpfromfile is not False:
            clustpfile=open(readclustpfromfile)
            #rint(readclustpfromfile)
            lines=clustpfile.readlines()

        for m in np.arange(len(self.z_hist_means)): 
            actualpixperredshiftcount=0 #counter to see how many pixels per redshift are inside radec limits, useful for the histograms as mean per redshift
            print("Shell number ", m)
            #print("######################################### NEW REDSHIFT SHELL ####################################")
            t0=time.perf_counter()
            zindexbase=self.zindextoBasePix(m)
            for i in np.arange(self.zindextoPixNumber(m)):
                pixindex=i+zindexbase
                pixraedgemin, pixraedgemax, pixdecedgemin, pixdecedgemax = self.NewGetPixelEdges(pixindex)#check if the pixel falls at least partly within the bounds of the catalog
                if pixraedgemax>self.minra and pixraedgemin<self.maxra and pixdecedgemax>self.mindec and pixdecedgemin<self.maxdec:
                    #print("Pix within limits!")
                    actualpixcount+=1
                    actualpixperredshiftcount+=1
                    if readclustpfromfile is not False:
                        clustp=float(lines[pixindex])
                    else:
                        clustp=self.ProbAtPixH0indv2Healpy(pixindex, startatzero=startatzero)
                        if clustp!=0:
                            #print(pixindex, " has non zero prob ", clustp)
                            nonzeropixels+=1

                else:
                    clustp=0
            
                if rateasnumberofgalsperpix==False: #this should be the main one
                    prob=clustp*self.redshiftrates[m] ####TODO: area not needed anymore _kinda_, pixels on the border will have some different area.
                else: 
                    prob=clustp

                probs[pixindex]=prob 
                #clustpfile.write(str(clustp)+"\n")

        print("Number of non zero pixels: ", nonzeropixels)	
        if readclustpfromfile is not False:
            clustpfile.close()

        probs[np.isfinite(probs)==False]=0 #set every not finite number to 0 in probs
        if rateasnumberofgalsperpix==True: #put to zero the probs of pixel in shells already complete
            for zindex in np.arange(len(self.NgalperRedshift)):
                if self.NgalperRedshift[zindex]<=0:
                    pixtozerobase=self.zindextoBasePix(zindex)
                    probs[pixtozerobase:pixtozerobase+self.zindextoPixNumber(zindex)]=0.
                    print("Redshift shell ", zindex, " starting complete! ")
                    
        summ=sum(probs)
        probsnorm=probs/summ
        print("Length of probsnorm", len(probsnorm))
        np.savetxt(self.newname[:-5]+"_ProbsNorm.txt", probsnorm)
        return probs, probsnorm
    
    def GetRates(self, zs):
        rates=np.ones(len(zs))
        for i in range(len(zs)):
            rates[i]=1-self.fz(zs[i])
        return rates

    def ComputeCompleteness(self,H0):
        
        # Warning - this integral misbehaves for small values of H0 (<25 kms-1Mpc-1).  TODO: fix this.
        print("Calculating p(G|H0,D)")
        def I(M,z):
            return SchechterMagFunction(H0=H0,Mstar_obs=self.Mstar_obs,alpha=self.alpha)(M)*self.zprior(z)

        Mmin = M_Mobs(H0,self.Mobs_min)
        Mmax = M_Mobs(H0,self.Mobs_max)
        
        num = dblquad(I, 0, self.zmax, lambda x : Mmin, lambda x: min(max(M_mdl(self.mth,self.cosmo.dl_zH0(x,H0)),Mmin),Mmax),epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0]
        den = dblquad(I, 0, self.zmax, lambda x: Mmin, lambda x: Mmax, epsabs=epsabsforFunctionsScratch,epsrel=epsrelforFunctionsScratch)[0]
        
        self.pGD = num/den
        return self.pGD  

    def LowCountAvarage(self, Npointavaragelowcount, completeness_lowcount_tolerance):
        shift=Npointavaragelowcount//2 #This is how much I move to the left or to the right when computing the Npoint avarage
        z_hist_prov=np.zeros(len(self.z_hist_count))
        for i in np.arange(len(self.z_hist_count)):
            if self.redshiftrates[i]>completeness_lowcount_tolerance:
                for j in list(range(-shift, shift+1)):
                    index=max(min(len(self.z_hist_count)-1, i+j), 0)
                    z_hist_prov[i]+=self.z_hist_count[index]
                z_hist_prov[i]=z_hist_prov[i]/Npointavaragelowcount#int(round(z_hist_prov[i]/Npointavaragelowcount))
            else:
                z_hist_prov[i]=self.z_hist_count[i]
        self.z_hist_count=z_hist_prov

    def AssignGalperRed(self, NGalAdd, renorm=True, incompleteness_threshold=0.99, combine_zsquared=False):
        #schechter weights don't give right normalization, TODO?
        #rates=rates/sum(rates) #normalize rates so it's a pdf
        #However, I have regions where the completeness is extremely low, aka rates extremely high (close to one). Remember rates is basically 1-completeness.
        #For this regions, the number of observed galaxies, z_hist_count, is almost useless and provokes artificial variability (as the galaxy number is very low, one or two randomly more galaxies is a huge difference)
        # For this areas, use instead a simple zsquared
        #print(z_hist_count)
        ratestouse=self.redshiftrates

        first_untrust_zshell=None
        combine_zsquared_zerogal=False

        for i in np.arange(len(self.z_hist_count)):
            if self.z_hist_count[i]<=2 and i>=5: #cannot "trust" redshift shell with so few galaxies. use here a zsquared
                ratestouse[i]=0.
                if first_untrust_zshell==None:
                    combine_zsquared_zerogal=True
                    first_untrust_zshell=i
                
        NGalperRedshift=np.array([round(ratestouse[i]*self.z_hist_count[i]/(1-ratestouse[i])) for i in np.arange(len(ratestouse))]) #this would be the answer (eventualmente renormalized)
        #print(NGalperRedshift)
        zsquared_count=self.z_hist_means**2
        if combine_zsquared==True: # this is a combination with a zsquared distribution for low completenesses, competitor with the Npoint avarage. Usually not used
            if renorm == True:
                
            
                
                NGalzsquared=0
                first_untrust_zshell=None
                for i in np.arange(len(NGalperRedshift)):
                    if rates[i]>incompleteness_threshold: #if completeness is less than 1%
                        NGalzsquared+=NGalperRedshift[i] #Compute number of galaxies that "cannot be trusted"
                        if first_untrust_zshell is None:
                            first_untrust_zshell=i
                
                

                fact=NGalzsquared/sum(zsquared_count[first_untrust_zshell:]) #Renormalization factor so that the sum of the untrusty redshift shell, with squared weights, is the galaxies that I have to add based on zsquared.
                zsquared_count=zsquared_count*fact
                for i in np.arange(len(NGalperRedshift)): #now change the untrusty shells
                    if rates[i]>incompleteness_threshold:
                        NGalperRedshift[i]=zsquared_count[i]

            elif renorm==False:
                fact=sum(NGalperRedshift)
                zsquared_count=zsquared_count*fact
                for i in np.arange(len(NGalperRedshift)): #now change the untrusty shells
                    if rates[i]>incompleteness_threshold:
                        NGalperRedshift[i]=zsquared_count[i]
                NGalperRedshift=NGalperRedshift/sum(NGalperRedshift)*NGalAdd



        if combine_zsquared_zerogal==True: # this is a necessary use of the zsquared function: when the completeness is so low I have no galaxies in certain bins, have to use zsquared for those bins.
            print("correcting with a z**2 distribution when galaxies in shell <2")
            factor=sum(NGalperRedshift[:first_untrust_zshell])/sum(zsquared_count[:first_untrust_zshell]) #this factor calibrates the distributions so that they have the same number until the first untrusty zshell, aka where I have 0 gals in the cut catalog
            zsquared_count_new=zsquared_count*factor
            NGalperRedshift[first_untrust_zshell:]=zsquared_count_new[first_untrust_zshell:]
            #print(factor, zsquared_count_new, NGalperRedshift)

        NGalperRedshiftNotnorm=NGalperRedshift
        
        print("If not renormalizing, I would have to add ", sum(NGalperRedshift), " instead of ", NGalAdd, " corresponding to a ", round(abs(sum(NGalperRedshift)-NGalAdd)/NGalAdd*100), "% difference")
        NGalperRedshift=NGalperRedshift/sum(NGalperRedshift)*NGalAdd #finally, renormalize
        #print("Ngaladd is ", NGalAdd, " and I will add ", sum(NGalperRedshift))

        return NGalperRedshift, NGalperRedshiftNotnorm 
    
    def GetrelevPixelsHealpy(self, index):
        z_index=self.PixIndextoZindex(index)
        deltaz=self.z_hist_means[1]-self.z_hist_means[0]
        En=int(self.zmax_corr//deltaz+1)#determine how many redshift pixels do I move to consider a pixel "close" purely in redshift
        
        cartcoo=self.PixIndexToCartCoo(index=index)
        univec=cartcoo/np.linalg.norm(cartcoo)

        sel=[]
        for i in list(range(-En, En+1)):
            z_shell_index=z_index+i
            if z_shell_index>=0 and z_shell_index<self.resol:
                zpix=self.z_hist_means[z_shell_index]
                r=self.zmax_corr/zpix
                shellbaseindex=self.zindextoBasePix(z_shell_index)
                shell_healpy_indexes=hp.query_disc(nside=z_shell_index+1, vec=univec, radius=r)
                for hpind in shell_healpy_indexes:
                    pixindex=shellbaseindex+hpind
                    if pixindex!=index:#make sure I don't add to the relevant pixels around that pixel the pixel itself
                        sel.append(pixindex)
        return sel 
    
    def NewGetPixelEdges(self, pixindex):
        zindex=self.PixIndextoZindex(pixindex)
        healpyindex=pixindex-self.zindextoBasePix(zindex)
        pix_ra, pix_dec = self.HealpytoRaDec(healpyindex=healpyindex, zindex=zindex)
        
        #borders=["SW", "W", "NW", "N", "NE", "E", "SE", "S"] #maybe directly loop over this?
        sel=hp.get_all_neighbours(zindex+1, healpyindex)
        neighralow=None
        neighrahigh=None
        neighdeclow=None
        neighdechigh=None

        for neighindex in sel:
            if neighindex!=-1:
                neigh_ra, neigh_dec = self.HealpytoRaDec(healpyindex=neighindex, zindex=zindex)
                #print("Neigh coo ", neigh_ra, neigh_dec)
                
                if neigh_dec==pix_dec:
                    if neighralow is None:
                        if neigh_ra<pix_ra:
                            neighralow=neigh_ra
                    else: 
                        if neigh_ra<pix_ra and neigh_ra>neighralow:
                            neighralow=neigh_ra

                    if neighrahigh is None:
                        if neigh_ra>pix_ra:
                            neighrahigh=neigh_ra
                    else: 
                        if neigh_ra>pix_ra and neigh_ra<neighrahigh:
                            neighrahigh=neigh_ra
                
                else:
                    if neighdeclow is None:
                        if neigh_dec<pix_dec:
                            neighdeclow=neigh_dec
                    else:
                        if neigh_dec<pix_dec and neigh_dec>neighdeclow:
                            neighdeclow=neigh_dec

                    if neighdechigh is None:
                        if neigh_dec>pix_dec:
                            neighdechigh=neigh_dec
                    else:
                        if neigh_dec>pix_dec and neigh_dec<neighdechigh:
                            neighdechigh=neigh_dec

        if neighralow is None:
            raedgemin=0.
        else:
            raedgemin=(neighralow+pix_ra)/2

        if neighrahigh is None:
            raedgemax=2*np.pi
        else:
            raedgemax=(neighrahigh+pix_ra)/2

        if neighdeclow is None:
            decedgemin=-np.pi/2
        else:
            decedgemin=(neighdeclow+pix_dec)/2

        if neighdechigh is None:
            decedgemax=np.pi/2
        else:
            decedgemax=(neighdechigh+pix_dec)/2
        return raedgemin, raedgemax, decedgemin, decedgemax

    def PixtoSkyCoo(self, pixindex):
        zindex=self.PixIndextoZindex(pixindex)
        healpyindex=pixindex-self.zindextoBasePix(zindex)
        z=self.z_hist_means[zindex]
        theta, phi = hp.pix2ang(zindex+1, healpyindex)
        ra, dec = self.HealpyAngtoradec(theta, phi)
        return ra, dec, z

    def PopulPixelHealpyH0ind(self, pixindex, generatealsomagnitude=False):

        #print("index choice ", index, "prob and normprob ", probs[index], probsnorm[index])
        zindex=self.PixIndextoZindex(pixindex)
        healpyindex=pixindex-self.zindextoBasePix(zindex)
        chopix_ra, chopix_dec, chopix_z = self.HealpytoSkyCoo(healpyindex, zindex)
        
        # create 1 galaxy randomly distributed in the pixel pix
        # TODO: CHECK APPARENT LUMINOSITY, NOW BIG NUMBERS COME UP (86 OR SIMILAR)
        zmin=max(chopix_z-self.deltaz/2, 0.)
        zmax=min(chopix_z+self.deltaz/2, self.zmax)       #don't want to go over the limits of the catalog. The redshift limit is hardcoded as a global variable

        raedgemin, raedgemax, decedgemin, decedgemax = self.NewGetPixelEdges(pixindex)

        ramin=max(self.minra, raedgemin)
        ramax=min(self.maxra, raedgemax)
        decmin=max(self.mindec, decedgemin)
        decmax=min(self.maxdec, decedgemax)

        if ramax>ramin and decmax>decmin and ramin>=self.minra and ramax<=self.maxra and decmin>=self.mindec and decmax<=self.maxdec: #silly check given the previous 4 lines, just to check everything is working properly
            a=1
        else:
            print("ERROR")
            print(ramin, ramax, decmin, decmax)
            sys.exit(0)

        ra, dec=ExtractSkyPos(ramin, ramax, decmin, decmax)    
        
        z=Extractz(zmin, zmax) #extract z
        
        #original extracting from distribution found onine
        #lum=simulate_schechter_distribution(alpha_ass, L_s, L_min, 1) #extract luminosity and convert it to magnitude #TODO introducing dependance on H0
        #M=-5./2.*np.log10(lum)#/L0) #should be without division, but with division makes it right
        #Extracting magnitudes from discretized pdf of the assumed schechter
        if generatealsomagnitude==True:
            M=GenerateSchechterSamples(assumed_band=self.assumed_band, Nsamples=1, H0_true=self.H0_assumed)[0]
            print(" Extracted M ", M)
            if self.linear==True:
                m=M+DistanceModulus(z*c/self.H0_assumed) 

            elif self.linear==False:
                m=M+DistanceModulus(ztoDFull(z, self.H0_assumed))
        else:
            m=1

        return ra, dec, z, m

    def ComputeClusteringProbabilityDifferencesforRecomputing(self, chopix_index):
        #finally, I have a new galaxy at the pix pix_chosen, so I increse the clustering probability of nearby pixels, which translates in an addition of
        #the correlation function between chosen pixel and nearby pixels
        chopix_zindex=self.PixIndextoZindex(chopix_index)
        chopix_healpyindex=chopix_index-self.zindextoBasePix(chopix_zindex)
        chopix_coo = self.HealpytoSkyCoo(chopix_healpyindex, chopix_zindex)
        
        if self.dist_eval=="binbin":
            t0=time.perf_counter()
            sel=self.GetrelevPixelsHealpy(index=chopix_index)
            #print("Time to get relev pixels ("+str(len(sel))+") ", time.perf_counter()-t0)

        distances=self.BatchAnalyticalSpherePointsDistance(centerpixcoo=chopix_coo, batchpixels=sel, type='distance')
        
        IndsToBeUpdated=[]
        DiffProbstoupdate=[]
        for pix_i in range(len(sel)): #loop over closeby pixels
            #print("##########")
            
            ind=sel[pix_i]
            #print(ind)
            pix_recomp_zindex=self.PixIndextoZindex(ind)
            if self.NgalperRedshift[pix_recomp_zindex]>0.:#check if that pixel belongs to a shell that is already filled
                #print("Is in a incomplete redshift shell")
                sel_raedgemin, sel_raedgemax, sel_decedgemin, sel_decedgemax = self.NewGetPixelEdges(ind)#check if the pixel falls at least partly within the bounds of the catalog
                if sel_raedgemax>self.minra and sel_raedgemin<self.maxra and sel_decedgemax>self.mindec and sel_decedgemin<self.maxdec:
                    #print("Lies at least partially within bounds")
                    r=distances[pix_i]
                    if r<self.rmax_corr and r>0.:
                        #tcorr=time.perf_counter()
                        DiffProbPix=self.CorrFunction(r)#*rates[pix_recomp_zindex]#Only difference is the corr function wrt the newly added galaxy
                        #print("Time for correlation function ", time.time()-tcorr)
                        IndsToBeUpdated.append(ind)
                        DiffProbstoupdate.append(DiffProbPix)
                        #print("DiffProb ", DiffProbPix)
            #        else:
                        #print("Distance ", r, " too big")
            #    else:
                    #print("Completely utside catalog bounds: ", sel_raedgemin, sel_raedgemax, sel_raedgemin, sel_decedgemax)
            #else:
                #print("Belongs to redshift shell ", pix_recomp_zindex, " already full")
            #print("##########")
            
        return IndsToBeUpdated, DiffProbstoupdate


    def CorrectCatalogHealpyFixedNgal013(self, newname=None, givenactualcompleteness=None, recomputeProbsclust=False, dist_eval="binbin", readclustpfromfile=False, rateasnumberofgalsperpix=False, renormrateasgals=False, Npointavaragelowcount=None, completeness_lowcount_tolerance=0.99, CatalogFullzs=None, puttinggalaxiesbyshell=True, startatzero=False):
        #Numbering Conventions of Pixels: each redshift shell has a number of pixels = 12*(shell_index+1)**2 (first has 12, bla bla), and inside the shell, the numbering convention is the one of healpy
        #TODO Add check on inputs and force some parameters inputs (nothing assumed) like chice of correlation function
        # 
        # rai, deci, zi, mi, Kcorri, sigmazi= Catalog.extract_galaxies()
        to=time.perf_counter()
        self.newname=newname

        self.dist_eval=dist_eval
        self.redshiftrates=self.GetRates(self.z_hist_means)

        rai_orig=self.allra
        deci_orig=self.alldec

        numgal_start=self.nGal
        print("Galaxies before treatment = ", numgal_start)
        
        
        if Npointavaragelowcount is not None: #TODO MAKEFUNC this could be a function
            self.LowCountAvarage(Npointavaragelowcount=Npointavaragelowcount, completeness_lowcount_tolerance=completeness_lowcount_tolerance)
        
        if givenactualcompleteness is not None:
            if givenactualcompleteness>1:
                givenactualcompleteness=givenactualcompleteness/100
            completeness=givenactualcompleteness
        else:
            self.zprior = redshift_prior(Omega_m=self.Omega_m, linear=self.linear)#TODO THIS ZPRIOR LIKE IN THE LIGO CASE IS TRICKY, MAYBE USE SPLINE INSTEAD FOR MICE? BUT IF USED ONLY FOR COMPLETENESS MAYBE IT"S OK
            completeness=self.ComputeCompleteness(H0=self.H0_assumed)
            print("Estimated completeness ", completeness*100, "%")

        targetnGal=round(self.nGal/completeness)
        print(targetnGal)
        NGalAdd=targetnGal-numgal_start
        

        #These two ifs should be option that are not the main ones, skipped for now
        
        if rateasnumberofgalsperpix==True:
            self.NgalperRedshift, self.NgalperRedshiftnotnorm = self.AssignGalperRed(NGalAdd=NGalAdd, renorm=renormrateasgals, combine_zsquared=False)
            #plt.plot(self.z_hist_means, NgalperRedshift+self.z_hist_count, label="Corrected")
            #plt.hist(self.z_hist_count, bins=self.z_hist_edges, label="Cut", alpha=0.4)
            #plt.hist(CatalogFullzs, bins=self.z_hist_edges, label="Full", alpha=0.4)
            #plt.legend()
            #plt.savefig("CorrectedCatalogs/ComparisonZsBeforePopulatingTESTLOWRESOLCORR.png")
            #plt.show()
            #plt.close()
            #sys.exit()
        if renormrateasgals==False:
            self.NgalperRedshift=self.NgalperRedshiftnotnorm
            NGalAdd=sum(self.NgalperRedshift)
        
        #tbinpro=time.perf_counter()
        #self.pix_occup, probsnew = self.BinningGalaxiesandAssignProbs(startatzero=False)
        #print("Time for binning and probs new ", time.perf_counter()-tbinpro)

        tbinpro=time.perf_counter()
        self.pix_occup=self.BinningGalaxiesHealpy()
        probs, probsnorm=self.AssignPixelProbs(startatzero=startatzero, rateasnumberofgalsperpix=rateasnumberofgalsperpix, readclustpfromfile=readclustpfromfile, savefile=True)
        print("Time for binning and probs old ", time.perf_counter()-tbinpro)
        
        print("Populating")
        print("Sum of non zero pixels before and after renorm (should be equal to actual pix count) ", sum(probs>0.), sum(probsnorm>0.))
        count=0
        tpop=time.perf_counter()        

        #POPULATE WITH NEW GALS
        print("Ngaladd = ", NGalAdd)
        newras=list(self.allra)
        newdecs=list(self.alldec)
        newzs=list(self.allz)
        newms=list(self.allm)

        pixelchoices=[]
        if puttinggalaxiesbyshell==True:
            if rateasnumberofgalsperpix==True:
                for m in range(len(self.z_hist_means)):
                    NGalAddShell=int(self.NgalperRedshift[m])
                    print("Shell ", m, ", NGalAddShell ", NGalAddShell)
                    if NGalAddShell>=1:
                        startindex=self.zindextoBasePix(m)
                        nextindex=self.zindextoBasePix(m+1)
                        if m == len(self.z_hist_means)-1:
                            shellprobs=probs[startindex:]
                        else:
                            shellprobs=probs[startindex:nextindex]

                        shellprobsnorm=shellprobs/sum(shellprobs)
                        for l in range(NGalAddShell):
                            shellindex=np.random.choice(a=len(shellprobsnorm), p=shellprobsnorm)
                            index=shellindex+startindex #Reconvert to global index for populpixel func
                            pixelchoices.append(index)
                            newra, newdec, newz, newm = self.PopulPixelHealpyH0ind(index, generatealsomagnitude=False) #place a galaxy in that pixel #TODO NEED TO MODIFY THIS with new pixels, also think about how to handle boundaries
                            newras.append(newra)
                            newdecs.append(newdec)
                            newzs.append(newz)
                            newms.append(newm)
            else:
                print("ERROR")
                sys.exit()
        

        else:
            tshell=time.perf_counter()
            for l in np.arange(NGalAdd):
                #print(l)
                if l/NGalAdd*100>count:
                    print("Added "+str(count)+"% of the galaxies! Time = ", time.perf_counter()-tshell)
                    tshell=time.perf_counter()
                    count+=1
                index=np.random.choice(a=len(probsnorm), p=probsnorm)
                pixelchoices.append(index)

                newra, newdec, newz, newm = self.PopulPixelHealpyH0ind(index, generatealsomagnitude=False) #place a galaxy in that pixel #TODO NEED TO MODIFY THIS with new pixels, also think about how to handle boundaries
                
                newras.append(newra)
                newdecs.append(newdec)
                newzs.append(newz)
                newms.append(newm)

                zindex=self.PixIndextoZindex(index)
                    
                if rateasnumberofgalsperpix==True:
                    self.NgalperRedshift[zindex]=self.NgalperRedshift[zindex]-1
                    if self.NgalperRedshift[zindex]<=0:
                        pixtozerobase=self.zindextoBasePix(zindex)
                        probs[pixtozerobase:pixtozerobase+self.zindextoPixNumber(zindex)]=0.
                        probsnorm=probs/sum(probs)
                        print("Redshift shell ", zindex, " finished! Ngaladd = ", self.NgalperRedshift)
                        ProbsPerRedshift=[probsnorm[self.zindextoBasePix(i):(self.zindextoBasePix(i)+self.zindextoPixNumber(i))] for i in np.arange(len(self.NgalperRedshift))]
                        sumProbsPerRedshift=[sum(ProbsPerRedshift[i]) for i in np.arange(len(ProbsPerRedshift))]
                        print("Sum of probsnorm per redshift = ", sumProbsPerRedshift)
                
                
                #This is if I want to recompute the clustering probability. Should take longer, and make clustering stronger. Probably not necessary     
                if recomputeProbsclust==True: #TODO FINISH AND CHECK THIS
                    #t_recomputing=time.perf_counter()
                    #tpixelrecompstart=time.perf_counter()                    
                    recomp_pix_indexes, DiffProbsPixels = self.ComputeClusteringProbabilityDifferencesforRecomputing(chopix_index=index)
                    #print(recomp_pix_indexes, DiffProbsPixels)
                    for ind, DiffProbPix in zip(recomp_pix_indexes, DiffProbsPixels):
                        #print(ind, DiffProbPix)
                        if np.isnan(DiffProbPix) or np.isinf(DiffProbPix) or DiffProbPix<0.:
                            print("DIFFPROB PROBLEM!", DiffProbPix)
                            sys.exit()
                        else:
                            oldprob=probs[ind]*1
                            probs[ind]=probs[ind]+DiffProbPix
                            #print("Old vs new Probs for pixel:  ", oldprob, probs[ind], " difference: ", probs[ind]-oldprob, " percentagewise ", (probs[ind]-oldprob)/oldprob*100)
                            
                   # tpixelrecomend=time.perf_counter()
                    #print("Time for recomputing nearby pixels one by one ", tpixelrecomend-tpixelrecompstart)

                    #tpixelrecompstart=time.perf_counter()                    
                    
                    #t0=time.perf_counter()
                    probsnorm=probs/sum(probs)
                    #print("Time for renorming probs ", time.perf_counter()-t0)
                    #print("Total time for recomputing after adding a gal ", time.perf_counter()-t_recomputing)
                    #sys.exit()
                    if np.isnan(probsnorm).any() or any(probsnorm<0.):
                        print("nan or negative presents!")
                        #print(probsnorm)
                        #print("Total: index of pixel where new gal was added ", index, "pix ", pixs[index], "pix that should still be the same", pix_chosen, " sum ", sum(probs), "total number of modified pixels ", len(sel))
                        #print("Modified pixels: ", ind)
                        for ir in np.arange(len(probs)):
                            if probsnorm[ir]<0. or np.isnan(probsnorm[ir]) or probs[ir]<0. or np.isnan(probs[ir]):
                                print("PROB Index ", ind, "prob notnorm ", probs[ir], "probnorm ", probsnorm[ir])
                        #print("Mean time per gal ", sum(tlist)/len(tlist))
                        sys.exit()

        np.savetxt(self.newname[:-5]+"_PixelChoices.txt", pixelchoices)

        print("Total time for population ", time.perf_counter()-tpop)
        print("Galaxy after treatment = ", len(newras))
        
        hf=h5py.File(self.newname, "w")
                        
        hf.create_dataset('ra', data=newras)
        hf.create_dataset('dec', data=newdecs)
        hf.create_dataset('z', data=newzs)
        hf.create_dataset('m', data=newms)
        hf.create_dataset('radec_lim', data=self.radec_lims)
        hf.close()

        print("Time for complete correction = ", time.perf_counter()-to)
        return self.newname
            
    