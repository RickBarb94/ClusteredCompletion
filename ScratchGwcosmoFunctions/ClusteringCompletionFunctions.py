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

    thetamin=DectoTheta(decmin)
    thetamax=DectoTheta(decmax)
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
    def __init__ (self, galaxy_catalog, Omega_m, linear, assumed_band, zmax, zpriortouse, H0_assumed, r0, gamma, rmin, rmax, scale, resol=100, CorrelationFunction=None, fullskyorfraction="octave"):
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

        self.z0=r0*H0_assumed/c
        self.gamma=gamma #NORMAL VALUE IS 1.8, MODIFY AS SOON AS YOU START RUN
        self.rmin=rmin
        self.rmax=rmax

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
            
        else:
            print("Please choose a correlation function")
            sys.exit()

        #Completion params
        self.scale=scale # relev pixels/gals are ones that are scale*R0 away from point considered
        self.resol=resol

        self.ResolPixNumberReferenceCum=np.array([sum(12*(i)**2 for i in np.arange(j+1)) for j in np.arange(resol+5)]) #this is used as reference later and computed here. resol+5 just so that I am sure I don't go out of bounds



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

    def PixIndextoZindex(self, pixindex):
        #print(np.where(pixindex>ResolPixNumberReferenceCum))
        return np.where(pixindex>=self.ResolPixNumberReferenceCum)[0][-1]

    def SpherePointsDistance(self, pix1, pix2, type):
        #naturally return either redshift or distance depending on if pix[2] defined as redshift or distance
        if type=='distance':
            pix1vec=SkyCootoCart(pix1[0], pix1[1], dl_zH0(pix1[2], H0=self.H0_assumed, Omega_m=self.Omega_m))
            pix2vec=SkyCootoCart(pix2[0], pix2[1], dl_zH0(pix2[2], H0=self.H0_assumed, Omega_m=self.Omega_m)) #TODO make it so it's independent of H0
        elif type=='redshift':
            pix1vec=SkyCootoCart(pix1[0], pix1[1], pix1[2])
            pix2vec=SkyCootoCart(pix2[0], pix2[1], pix2[2])
        #print(pix1vec, pix2vec)
        r=np.linalg.norm(pix1vec-pix2vec)
        return r
    
    def csi(self, r):
        csi=(self.r0/r)**self.gamma 
        return csi 
    
    def csi_fit(self, r):
        r0fit=7.30627815
        gammafit=1.53939311
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
        t0=time.time()
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
        print("Time for binning ", time.time()-t0)
        return pixs
    
    def GetRates(self, zs):
        rates=np.ones(len(zs))
        for i in range(len(zs)):
            rates[i]=1-self.fz(zs[i])
        return rates

    def ComputeCompleteness(self,H0):
        
        # Warning - this integral misbehaves for small values of H0 (<25 kms-1Mpc-1).  TODO: fix this.
        bar = progressbar.ProgressBar()
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

    def ProbAtPixH0indv2Healpy(self, z_index, index, startatzero=False):
	
        closepix_numbers=self.GetrelevPixelsHealpy(z_index, index)
        #print("Evaluating clust prob at pix ", index)
        healpy_index=index-self.zindextoBasePix(z_index)
        
        pix_coo= self.HealpytoSkyCoo(healpy_index, z_index)#get coordinate of pixels in ra, dec, z

        if startatzero==True:
            prob=0.
        else:
            prob=1.
        #print(closepix_numbers)
        for i in closepix_numbers:
            if self.pix_occup[i]>0:
                #print("Contributing pixel!", i)
                closepix_z_index=self.PixIndextoZindex(i)
                closepix_healpy_index= i-self.zindextoBasePix(closepix_z_index)  #TODO:Maybe have two arrays, one with occupation numbers and one with the healpy pixel numbers? Maybe it's faster so I don't have to recompute but just read?
                closepix_coo=self.HealpytoSkyCoo(closepix_healpy_index, closepix_z_index)
                
                if self.dist_eval=="binbin":
                    r=self.SpherePointsDistance(pix_coo, closepix_coo, type='distance')
                #if r>self.scale*self.r0:#this if might be necessary for speed but I already choose only closeby pixels so maybe not
                #    cor=0.
                #print("Dist = ", r)
                #else:
                cor=self.CorrFunction(r)*self.pix_occup[i]
                prob+=cor
        
        return prob
    
    def GetrelevPixelsHealpy(self, z_index, index):
        sel=[]
        zpix=self.z_hist_means[z_index]
        deltaz=self.z_hist_means[1]-self.z_hist_means[0]
        #given a pixel index and the spacing in z of the pixels, return sel of pixels close to the pixel within scale*r0
        angle0=np.arcsin(self.scale*self.z0/2./zpix) #I have to consider pixels within [z-scale*z0, z+scale*z0] and with ra and dec within + and - angle0
        #print(z_index, index)
        thetapix, phipix = hp.pix2ang(z_index+1, index-self.zindextoBasePix(z_index))
        
        
        
        En=int(self.scale*self.z0//deltaz+1)#determine how many redshift pixels do I move to consider a pixel "close" purely in redshift
        if zpix<self.scale*self.z0: #just add all pixels in those shells
            for i in list(range(-En,En+1)):
                z_shell_index=z_index+i
                basepix_shell=self.zindextoBasePix(z_shell_index)
                for k in np.arange(self.zindextoPixNumber(z_shell_index)):
                    if i!=0 and k!=0:
                        sel.append(basepix_shell+k)
        else:	
            for i in list(range(-En, En+1)):
                z_shell_index=z_index+i
                if z_shell_index<len(self.z_hist_means):
                    basepix_shell=self.zindextoBasePix(z_shell_index)
                    deg_res=hp.nside2resol(z_shell_index+1) #NSIDE is zindex+1
                    Em=int(angle0//deg_res+1)
                    El=Em
                    for k in list(range(-El, El+1)): #move up with theta angle, get the pixel number corresponding to that "vector", and move horizontally of the Em amount
                        thetainq=thetapix+k*deg_res
                        decinq=np.pi/2-thetainq
                        if self.mindec<decinq and decinq<self.maxdec: #hardcoded the conversion between phi and dec. Don't go over bound for a not full sky cat
                            pix=hp.ang2pix(z_shell_index+1, thetainq, phipix)
                            
                            for j in list(range(-Em, Em+1)):#loop on dec
                                if basepix_shell+pix+j<self.totalPixelNumber:
                                    thetainq, phiinq = hp.pix2ang(z_shell_index+1,pix+j)
                                    rainq, decinq = self.HealpyAngtoradec(thetainq, phiinq)
                                    if self.minra<rainq and rainq<self.maxra and self.mindec<decinq and decinq<self.maxdec:
                                        if i!=0 or k!=0 or j!=0:
                                            sel.append(basepix_shell+pix+j) #pix+j is the relevant pixels left and right of pixel at that shell
        #print(list(set(sel)))
        return list(set(sel))#clean out duplicates

    def PopulPixelHealpyH0ind(self, pixindex, generatealsomagnitude=False):

        #print("index choice ", index, "prob and normprob ", probs[index], probsnorm[index])
        zindex=self.PixIndextoZindex(pixindex)
        healpyindex=pixindex-self.zindextoBasePix(zindex)
        chopix_ra, chopix_dec, chopix_z = self.HealpytoSkyCoo(healpyindex, zindex)
        
        # create 1 galaxy randomly distributed in the pixel pix
        # TODO: CHECK APPARENT LUMINOSITY, NOW BIG NUMBERS COME UP (86 OR SIMILAR)
        deg_res=hp.nside2resol(zindex+1)#TODO is this right? 
        
        #This kinda creates a square around the pixel center. I put a galaxy in this square and then check if it falls within the pixel. Otherwise don't keep it
        #print("Pixel ra dec ", chopix_ra, chopix_dec)
        #print("deg res ", deg_res)
        #print("minra, maxra, mindec, maxdec ",  minra, maxra, mindec, maxdec)


        ramin=max(chopix_ra-deg_res/2, self.minra)
        ramax=min(chopix_ra+deg_res/2, self.maxra)
        decmin=max(chopix_dec-deg_res/2, self.mindec)
        decmax=min(chopix_dec+deg_res/2, self.maxdec)
        zmin=max(chopix_z-self.deltaz/2, 0.)
        zmax=min(chopix_z+self.deltaz/2, self.zmax)       #don't want to go over the limits of the catalog. The redshift limit is hardcoded as a global variable
        
        while True: #extract a sky position, check whether it actually falls inside that pixel, if not continue on extracting until it does.
            ra, dec=ExtractSkyPos(ramin, ramax, decmin, decmax)
            #print("Healpy pixel theta phi ", hp.pix2ang(zindex+1, healpyindex))
            phi, theta = self.radectoHealpyCoo(ra, dec)
            #print("Extracted theta phi ", theta, phi)
            pixcheck = hp.ang2pix(nside=zindex+1, theta=theta, phi=phi)
            if pixcheck==healpyindex:
                break
            
        
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

    def CorrectCatalogHealpyFixedNgal013(self, newname=None, givenactualcompleteness=None, recomputeProbsclust=False, dist_eval="binbin", readclustpfromfile=False, usejustclustpminfromfile=False, usejustclustpminfromfilemeanfrac=1., rateasnumberofgalsperpix=False, renormrateasgals=False, Npointavaragelowcount=None, completeness_lowcount_tolerance=0.99, CatalogFullzs=None, puttinggalaxiesbyshell=True, startatzero=False):
        #Numbering Conventions of Pixels: each redshift shell has a number of pixels = 12*(shell_index+1)**2 (first has 12, bla bla), and inside the shell, the numbering convention is the one of healpy
        #TODO Add check on inputs and force some parameters inputs (nothing assumed) like chice of correlation function
        # 
        # rai, deci, zi, mi, Kcorri, sigmazi= Catalog.extract_galaxies()
        to=time.time()

        self.totalPixelNumber=sum(12*i**2 for i in np.arange(self.resol+1)) #NSIDE pixel cannot be 0, so must start at 1 (for which I have 12 pixels), so loop is until arange +1

        z_hist=np.histogram(self.allz, bins=np.linspace(0,self.zmax, self.resol+1))
        self.z_hist_count=z_hist[0]
        self.z_hist_edges=z_hist[1]
        self.z_hist_means=np.array([(self.z_hist_edges[i]+self.z_hist_edges[i+1])/2 for i in np.arange(self.resol)])
        self.deltaz=self.z_hist_means[1]-self.z_hist_means[0]

        self.pix_occup=self.BinningGalaxiesHealpy()
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
            print(self.ComputeCompleteness(H0=self.H0_assumed))
            completeness=round(100*self.ComputeCompleteness(H0=self.H0_assumed))/100 #TODO this is so that I'm closer because I use integers as completeness
        
        targetnGal=round(self.nGal/completeness)
        print(targetnGal)
        NGalAdd=targetnGal-numgal_start
        

        #These two ifs should be option that are not the main ones, skipped for now
        
        if rateasnumberofgalsperpix==True:
            NgalperRedshift, NgalperRedshiftnotnorm = self.AssignGalperRed(NGalAdd=NGalAdd, renorm=renormrateasgals, combine_zsquared=False)
            #plt.plot(self.z_hist_means, NgalperRedshift+self.z_hist_count, label="Corrected")
            #plt.hist(self.z_hist_count, bins=self.z_hist_edges, label="Cut", alpha=0.4)
            #plt.hist(CatalogFullzs, bins=self.z_hist_edges, label="Full", alpha=0.4)
            #plt.legend()
            #plt.savefig("CorrectedCatalogs/ComparisonZsBeforePopulatingTESTLOWRESOLCORR.png")
            #plt.show()
            #plt.close()
            #sys.exit()
        if renormrateasgals==False:
            NgalperRedshift=NgalperRedshiftnotnorm
            NGalAdd=sum(NgalperRedshift)
        
        

        
        #other couple of ifs for particular setups I don't remember
        if readclustpfromfile is not False:
            clustpfile=open(readclustpfromfile)
            #rint(readclustpfromfile)
            lines=clustpfile.readlines()
        if usejustclustpminfromfile is not False:
            linefloat=np.array([float(lines[b]) for b in np.arange(len(lines))])
            meanclustp=np.mean(linefloat[np.where(np.array(linefloat)>0.)[0]])
            print("mean clust p ", meanclustp)
            meanclustp=meanclustp*usejustclustpminfromfilemeanfrac
            print("Using a fraction ", usejustclustpminfromfilemeanfrac, " of the mean, effective mean clust p ", meanclustp)
            
            readclustpfromfile=False
        
        #compute probs
        problist=np.zeros(self.totalPixelNumber)
        actualpixcount=0
        actualpixperredshiftcounts=[]
        t1=time.time()

        for m in np.arange(len(self.z_hist_means)): 

            actualpixperredshiftcount=0 #counter to see how many pixels per redshift are inside radec limits, useful for the histograms as mean per redshift
            print("Shell number ", m)
            #print("######################################### NEW REDSHIFT SHELL ####################################")
            t0=time.time()
            zindexbase=self.zindextoBasePix(m)
            for i in np.arange(self.zindextoPixNumber(m)):
                pixindex=i+zindexbase
                thetapix, phipix = hp.pix2ang(m+1, i) #NSIDE is always z_index+1
                rapix, decpix = self.HealpyAngtoradec(theta=thetapix, phi=phipix)
                #print("Pix number ",pixindex, "   ", rapix, decpix)
                if rapix>self.minra and rapix<self.maxra and decpix>self.mindec and decpix<self.maxdec: #TODO IMPORTANT check in this loop and in probatpix what to do with pixels outside limits but with support inside limits
                    #print("Pix within limits!")
                    actualpixcount+=1
                    actualpixperredshiftcount+=1
                    if readclustpfromfile is not False:
                        clustp=float(lines[pixindex])
                    else:
                        clustp=self.ProbAtPixH0indv2Healpy(m, pixindex, startatzero=startatzero)

                        #some ifs for special circumstances
                        if usejustclustpminfromfile=='SumMeanToAll':
                            clustp+=meanclustp
                        if usejustclustpminfromfile=='ThresholdMean':
                            clustp=max(clustp, meanclustp)
                else:
                    clustp=0
            
                if rateasnumberofgalsperpix==False: #this should be the main one
                    prob=clustp*self.redshiftrates[m] ####TODO: area not needed anymore _kinda_, pixels on the border will have some different area.
                else: 
                    prob=clustp

                problist[pixindex]=prob 
                #clustpfile.write(str(clustp)+"\n")	

            
            actualpixperredshiftcounts.append(actualpixperredshiftcount)
            print("Time for redshift shell ", time.time()-t0)
            print(" ")
            
        if readclustpfromfile is not False:
            clustpfile.close()
        
        print("Total number of pixels included in octave " , actualpixcount)
        
        
        t7=time.time()
        
        
        
        print("Time for computing all probs ", time.time()-t1)
        
        #########renorming probs############################
        probs=np.array(problist)
        probs[np.isfinite(probs)==False]=0 #set every not finite number to 0 in probs
        if rateasnumberofgalsperpix==True: #put to zero the probs of pixel in shells already complete
            for zindex in np.arange(len(NgalperRedshift)):
                if NgalperRedshift[zindex]<=0:
                    pixtozerobase=self.zindextoBasePix(zindex)
                    probs[pixtozerobase:pixtozerobase+self.zindextoPixNumber(zindex)]=0.
                    print("Redshift shell ", zindex, " starting complete! ")
                    
        summ=sum(probs)
        probsnorm=probs/summ
        print("Length of probsnorm", len(probsnorm))
        print("Time for normalizing probs ", time.time()-t7)
        
        #bit for random choice
        
        print("starting sum ", summ)
        
        
        print("Populating")
        print("Sum of non zero pixels before and after renorm (should be equal to actual pix count) ", sum(probs>0.), sum(probsnorm>0.))
        count=0
        tpop=time.time()

        #This shouldn't be the thing to do I think
        

        #POPULATE WITH NEW GALS
        print("Ngaladd = ", NGalAdd)
        newras=list(self.allra)
        newdecs=list(self.alldec)
        newzs=list(self.allz)
        newms=list(self.allm)

        if puttinggalaxiesbyshell==True:
            if rateasnumberofgalsperpix==True:
                for m in range(len(self.z_hist_means)):
                    NGalAddShell=int(NgalperRedshift[m])
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
                        
                            newra, newdec, newz, newm = self.PopulPixelHealpyH0ind(index, generatealsomagnitude=False) #place a galaxy in that pixel #TODO NEED TO MODIFY THIS with new pixels, also think about how to handle boundaries
                            
                            newras.append(newra)
                            newdecs.append(newdec)
                            newzs.append(newz)
                            newms.append(newm)
            else:
                print("ERROR")
                sys.exit()
        

        else:
            for l in np.arange(NGalAdd):
                if l/NGalAdd*100>count:
                    print("Added "+str(count)+"% of the galaxies! Time = ", time.time()-tpop)
                    count+=1
                index=np.random.choice(a=len(probsnorm), p=probsnorm)
                newra, newdec, newz, newm = self.PopulPixelHealpyH0ind(index, generatealsomagnitude=False) #place a galaxy in that pixel #TODO NEED TO MODIFY THIS with new pixels, also think about how to handle boundaries
                
                newras.append(newra)
                newdecs.append(newdec)
                newzs.append(newz)
                newms.append(newm)

                zindex=self.PixIndextoZindex(index)
                    
                if rateasnumberofgalsperpix==True:
                    NgalperRedshift[zindex]=NgalperRedshift[zindex]-1
                    if NgalperRedshift[zindex]<=0:
                        pixtozerobase=self.zindextoBasePix(zindex)
                        probs[pixtozerobase:pixtozerobase+self.zindextoPixNumber(zindex)]=0.
                        probsnorm=probs/sum(probs)
                        print("Redshift shell ", zindex, " finished! Ngaladd = ", NgalperRedshift)
                        ProbsPerRedshift=[probsnorm[self.zindextoBasePix(i):(self.zindextoBasePix(i)+self.zindextoPixNumber(i))] for i in np.arange(len(NgalperRedshift))]
                        sumProbsPerRedshift=[sum(ProbsPerRedshift[i]) for i in np.arange(len(ProbsPerRedshift))]
                        print("Sum of probsnorm per redshift = ", sumProbsPerRedshift)
                        
                #This is if I want to recompute the clustering probability. Should take longer, and make clustering stronger. Probably not necessary     
                if recomputeProbsclust==True: #TODO FINISH AND CHECK THIS
                    #finally, I have a new galaxy at the pix pix_chosen, so I increse the clustering probability of nearby pixels, which translates in an addition of
                    #the correlation function between chosen pixel and nearby pixels, times the two weights (pix volume and rates)
                    chopix_healpyindex=index-self.zindextoBasePix(zindex)
                    chopix_coo = self.HealpytoSkyCoo(chopix_healpyindex, zindex)

                    if dist_eval=="binbin":
                        sel=self.GetrelevPixelsHealpy(z_index=zindex, index=index)
                    
                    for ind in sel:
                        pix_recomp_zindex=self.PixIndextoZindex(ind)
                        if NgalperRedshift[pix_recomp_zindex]>0.:
                            pix_recomp_healpyindex=ind-self.zindextoBasePix(pix_recomp_zindex)
                            pix_recomp_coo=self.HealpytoSkyCoo(pix_recomp_healpyindex, pix_recomp_zindex)
                            if dist_eval=="binbin":
                                r=self.SpherePointsDistance(chopix_coo, pix_recomp_coo, type='distance')
                            if r>self.scale*self.r0:
                                DiffProbPix=0.
                            elif r>0.:
                                DiffProbPix=self.CorrFunction(r)#*rates[pix_recomp_zindex]#Only difference is the corr function wrt the newly added galaxy
                            if np.isnan(DiffProbPix) or np.isinf(DiffProbPix) or DiffProbPix<0.:
                                print("DIFFPROB PROBLEM! distance ", r, " corr fun value ", DiffProbPix, " coo ", pix_recomp_coo)
                            else:
                                if NgalperRedshift[pix_recomp_zindex]>=0:
                                    probs[ind]=probs[ind]+DiffProbPix
                        DiffProbPix=0.
                    probsnorm=probs/sum(probs)
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

        print("Total time for population ", time.time()-tpop)
        print("Galaxy after treatment = ", len(newras))
        
        hf=h5py.File(newname, "w")
                        
        hf.create_dataset('ra', data=newras)
        hf.create_dataset('dec', data=newdecs)
        hf.create_dataset('z', data=newzs)
        hf.create_dataset('m', data=newms)
        hf.create_dataset('radec_lim', data=self.radec_lims)
        hf.close()

        print("Time for complete correction = ", time.time()-to)
        return newname
            
    