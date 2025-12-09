def pD_inside_theoretic_precompute(H0, Catalog, dl_det, sigmaprop, linear_data_gen, Omega_m, pdettype, dl_distr, weightedcat=False):
    "This computes the pdet theoretically from eq 22 of hitchikers"
    cosmo = fast_cosmology(Omega_m=Omega_m, linear=linear_data_gen)

    den = np.zeros(len(H0))
    zs = Catalog.z
    if weightedcat==True:
        weights=Catalog.weights

        
    bar = progressbar.ProgressBar()
    print("Calculating p(D|G, H0)")
    for i in bar(range(len(zs))):
        # loop over random draws from galaxies
        if weightedcat==True:
            weight = weights[i]
        else:
            weight = 1.0
        if pdettype=="theoretic":
            if dl_distr=="gaussian-sigmad":
                prob = pD_Heaviside_theoretic_precompute(zs[i],H0, dl_det, sigmaprop,cosmo).flatten()
            if dl_distr=="uniform":
                prob=[]
                for H0ele in H0:
                    probele = pD_Heaviside_theoretic_uniform_precompute(zs[i],H0ele, dl_det, sigmaprop,cosmo)
                    prob.append(probele)
                prob=np.array(prob)
        else:
            print("Unknown pdet ", pdettype)
            sys.exit()
        deninner = prob*weight*ps_z_precompute(zs[i])
        den += deninner

    pDG = den/len(zs)
    print(pDG)

    return pDG

def pD_inside_theoretic_precompute_smoothed(H0, z_array, z_vals, dl_det, sigmaprop, linear_data_gen, Omega_m, pdettype, dl_distr):
    "This computes the pdet theoretically from eq 22 of hitchikers"
    cosmo = fast_cosmology(Omega_m=Omega_m, linear=linear_data_gen)

    den = np.zeros(len(H0))
    
    denH0s=[]
    
    for H0_iter in H0:
       
        if pdettype=="theoretic":
            if dl_distr=="gaussian-sigmad":
                prob = pD_Heaviside_theoretic_precompute(z_array, H0_iter, dl_det, sigmaprop, cosmo).flatten() * z_vals
                denH0=np.trapz(prob, z_array)/np.trapz(z_vals, z_array)

            else:
                print("Unknown dl_distr")
                sys.exit()
        else:
            print("Unknown pdet ", pdettype)
            sys.exit()
        denH0s.append(denH0)

    pDG = np.array(denH0s)

    return pDG

def pG_theoretic_precompute(H0, mth, assumed_band, dl_det, sigmaprop, Omega_m=0.25, linear=True, zmax=10., pdettype="theoretic", zpriortouse="uniform"):
    
    cosmo = fast_cosmology(Omega_m=Omega_m, linear=linear)
    if zpriortouse=="uniform":
        zprior=redshift_prior(Omega_m=Omega_m, linear=linear)
    elif zpriortouse=="Mice":
        zprior=Miceprior

    sp = SchechterParams(assumed_band)

    alpha = sp.alpha
    Mstar_obs = sp.Mstar
    Mobs_min = sp.Mmin
    Mobs_max = sp.Mmax

    # Warning - this integral misbehaves for small values of H0 (<25 kms-1Mpc-1).  TODO: fix this.
    num = np.zeros(len(H0)) 
    den = np.zeros(len(H0))
    bar = progressbar.ProgressBar()
    print("Calculating p(G|H0,D)")
    for i in bar(range(len(H0))):
        def I(z,M):
            
            if pdettype==True:
                temp = SchechterMagFunction(H0=H0[i],Mstar_obs=Mstar_obs,alpha=alpha)(M)*self.pdet.pD_zH0_eval(z,H0[i])*self.zprior(z)*self.ps_z(z)
            elif pdettype=='cube':
                temp = SchechterMagFunction(H0=H0[i],Mstar_obs=Mstar_obs,alpha=alpha)(M)*H0[i]**3*self.zprior(z)*self.ps_z(z)
            elif pdettype=="theoretic":
                temp = SchechterMagFunction(H0=H0[i],Mstar_obs=Mstar_obs,alpha=alpha)(M)*pD_Heaviside_theoretic_precompute(z,H0[i], dl_det, sigmaprop, cosmo)*zprior(z)*ps_z_precompute(z)
                
            elif pdettype=="ones":
                temp = SchechterMagFunction(H0=H0[i],Mstar_obs=Mstar_obs,alpha=alpha)(M)*self.zprior(z)*self.ps_z(z)
            #if self.weighted:
            #    return temp*L_M(M)
            #else:
            return temp
        # Mmin and Mmax currently corresponding to 10L* and 0.001L* respectively, to correspond with MDC
        # Will want to change in future.
        # TODO: test how sensitive this result is to changing Mmin and Mmax.
        Mmin = M_Mobs(H0[i],Mobs_min)
        Mmax = M_Mobs(H0[i],Mobs_max)
        #num[i] = dblquad(I,0,zmax,lambda x: Mmin,lambda x: min(max(M_mdl(mth,cosmo.dl_zH0(x,H0[i])),Mmin),Mmax),epsabs=0,epsrel=1.49e-4)[0]
        #num[i] = dblquad(I,0,self.zcut,lambda x: Mmin,lambda x: M_mdl(self.mth,self.cosmo.dl_zH0(x,H0[i])),epsabs=0,epsrel=1.49e-4)[0]
        #den[i] = dblquad(I,0,zmax,lambda x: Mmin,lambda x: Mmax,epsabs=0,epsrel=1.49e-4)[0]
        num[i] = dblquad(I,Mmin, Mmax, lambda x: 0, lambda x: min(z_dlH0(dl_mM(mth,x),H0[i],linear=linear), zmax),epsabs=0,epsrel=1.49e-4)[0]
        den[i] = dblquad(I, Mmin, Mmax, lambda x: 0, lambda x: zmax, epsabs=0,epsrel=1.49e-4)[0]

    pGD = num/den
    #print("PG NUM : ", num)
    #print("PG DEN : ", den)
    return pGD    

def pD_outside_theoretic_precompute(H0, mth, assumed_band, dldet, sigmaprop, Omega_m=0.25, linear=True, whole_sky_cat=True, ra_min=0., ra_max=np.pi/2, dec_min=0., dec_max=np.pi/2, weighted=False, zmax=10., renormtosimul=False, CatalogFullName=None, CatalogCutName=None, zpriortouse="uniform"):
    
    cosmo = fast_cosmology(Omega_m=Omega_m, linear=linear)
    if zpriortouse=="uniform":
        zprior = redshift_prior(Omega_m=Omega_m, linear=linear)
    elif zpriortouse=="Mice":
        zprior = Miceprior
    sp = SchechterParams(assumed_band)

    alpha = sp.alpha
    Mstar_obs = sp.Mstar
    Mobs_min = sp.Mmin
    Mobs_max = sp.Mmax

    den = np.zeros(len(H0))
    
    #def skynorm(dec,ra):
    #    return np.cos(dec)
    #redshiftnorm=zmax**3/3
    #norm = dblquad(skynorm,ra_min,ra_max,lambda x: dec_min,lambda x: dec_max,epsabs=0,epsrel=1.49e-4)[0]/(4.*np.pi)
    #print("Sky stuff: ramin, ramax, decmin, decmax, norm : ", ra_min, ra_max, dec_min, dec_max, norm)
    bar = progressbar.ProgressBar()
    print("Calculating p(D|H0,bar{G})")
    for i in bar(range(len(H0))):
        Mmin = M_Mobs(H0[i],Mobs_min)
        Mmax = M_Mobs(H0[i],Mobs_max)
        
        #This seems to renormalize the pdet fairly well. Remember must be present both here and in px
        def Itonorm(z,M):
            temp=SchechterMagFunction(H0=H0[i],Mstar_obs=Mstar_obs,alpha=alpha)(M)*zprior(z)*ps_z_precompute(z)
            return temp
        totnorm=dblquad(Itonorm,Mmin,Mmax,lambda x: z_dlH0(dl_mM(mth,x),H0[i],linear=linear),lambda x: zmax,epsabs=0,epsrel=1.49e-4)[0] #check whether to use this style of integral or the one in modificationskypatch=False/pG
    
        def Iden(z,M):
            temp = SchechterMagFunction(H0=H0[i],Mstar_obs=Mstar_obs,alpha=alpha)(M)*pD_Heaviside_theoretic_precompute(z,H0[i], dl_det=dldet, sigmaprop=sigmaprop, cosmo=cosmo)*zprior(z)*ps_z_precompute(z)
            
            if weighted:
                return temp*L_M(M)
            else:
                return temp
            
        #SchechterNorm=quad(SchechterMagFunction(H0=H0[i],Mstar_obs=Mstar_obs,alpha=alpha), Mmin, Mmax)[0]
        #den[i] = dblquad(Iden,Mmin,Mmax,lambda x: min(z_dlH0(dl_mM(mth,x),H0[i],linear=linear), zmax),lambda x: zmax,epsabs=0,epsrel=1.49e-4)[0]#/(SchechterNorm*redshiftnorm) #check whether to use this style of integral or the one in modificationskypatch=False/pG
        den[i] = dblquad(Iden,Mmin,Mmax,lambda x: min(z_dlH0(dl_mM(mth,x),H0[i],linear=linear), zmax),lambda x: zmax,epsabs=0,epsrel=1.49e-4)[0]/totnorm#/(SchechterNorm*redshiftnorm) #check whether to use this style of integral or the one in modificationskypatch=False/pG
    #if whole_sky_cat == True: #TODO check what's up with this allsky thing, should be whole_cat but not sure
    #    pDnG = den*norm
    #else:
    #    pDnG = den*(1.-norm)
    print(den)
    pDnG=den
    if renormtosimul==True:
        simdictionary = SimulatePGandPdets(CatalogFullName=CatalogFullName, CatalogCutName=CatalogCutName, dldets=[dldet], cosmo=cosmo, H0arr=H0, sigma_dl_prop_ass=sigmaprop, Nevents=1000000)
        pGsim=simdictionary[str(dldet)+"pg"]
        pdetinsim=simdictionary[str(dldet)+"pdin"]
        pdetoutsim=simdictionary[str(dldet)+"pdout"]

        pDnG=pDnG/np.trapz(pDnG, H0)*np.trapz(pdetoutsim, H0)
    return pDnG

def pD_outside_theoretic_precompute_Jon(H0, zmax, mth, Omega_m, linear, assumed_band, dldet, sigmaprop):
    cosmo = fast_cosmology(Omega_m=Omega_m, linear=linear)
    zprior = redshift_prior(Omega_m=Omega_m, linear=linear)
    den = np.zeros(len(H0))
    bar = progressbar.ProgressBar()

    print("Calculating p(D|H0,bar{G})")
    
    for i in bar(range(len(H0))):
            
        def Integrand(z):
            return pzout(z, H0[i], mth, Omega_m, linear, assumed_band, zmax, zprior)*pD_Heaviside_theoretic_precompute(z,H0[i], dldet, sigmaprop, cosmo)
        
        #zarray=np.linspace(0,zmax, 100)
        #yarray=np.array([Integrand(z) for z in zarray])
        #den[i]=np.trapz(yarray, zarray)
        den[i] = quad(Integrand, 0, zmax, epsabs=0,epsrel=1.49e-4)[0]
    return den


        
def pzout(z, H0, mth, Omega_m, linear, assumed_band, zmax, zprior):
    #this computes the out of catalog part of equation  9
    return (1-fz(z, H0, mth, Omega_m, linear, assumed_band))*Derivative_comov_volume(z, linear, zprior)/Comov_volume(z=zmax, linear=linear, Omega_m=Omega_m)

def Derivative_comov_volume(z, linear, zprior):
    #derivative of volume element within z. Note that eventual normalization factor don't matter, as this only appear in likelihood through dVc/dz / Vc(z)
    if linear==True:
        return 3*z**2
    else:
        return zprior(z)
    

def Comov_volume(z, linear, Omega_m):
    #volume element within z. Note that eventual normalization factor don't matter, as this only appear in likelihood through dVc/dz / Vc(z)
    if linear==True:
        return z**3 
    else:
        return volume_z(z, Omega_m=Omega_m)
    
def fz(z, H0, mth, Omega_m, linear, assumed_band):
    cosmo = fast_cosmology(Omega_m=Omega_m, linear=linear)
    sp = SchechterParams(assumed_band)
    
    alpha = sp.alpha
    Mstar_obs = sp.Mstar
    Mobs_min = sp.Mmin
    Mobs_max = sp.Mmax

    Mmin = M_Mobs(H0,Mobs_min)
    Mmax = M_Mobs(H0,Mobs_max)

    def I(M):
        return SchechterMagFunction(H0=H0,Mstar_obs=Mstar_obs,alpha=alpha)(M)

    num = quad(I,Mmin, min(max(M_mdl(mth,cosmo.dl_zH0(z,H0)),Mmin),Mmax),epsabs=0,epsrel=1.49e-4)[0]
    den = quad(I, Mmin, Mmax, epsabs=0,epsrel=1.49e-4)[0]

    return num/den


def logsumexpsum(thingtosum):
    
    logargs=np.log(thingtosum)
    logmax=max(logargs)
    
    logargsminuslogmax=logargs-logmax
    LSElogsum = logmax + logsumexp(logargsminuslogmax)
    return np.exp(LSElogsum)

def SimulatePGandPdets(CatalogFullName, CatalogCutName, dldets, cosmo, sigma_dl_prop_ass, H0arr, Nevents=10000):

    with h5py.File(CatalogCutName, "r") as f:
        #cut_ms=f["m"][:]
        cut_zs=f["z"][:]
        cut_ras=f["ra"][:]
        cut_decs=f["dec"][:]
    with h5py.File(CatalogFullName, "r") as f:
        #full_ms=f["m"][:]
        full_zs=f["z"][:]
        full_ras=f["ra"][:]
        full_decs=f["dec"][:]

    mask=np.where(full_zs>-1)[0]

    print("Simulating pdet for renormalization!")

    pG=[]
    pdetin=[]
    pdetout=[]
    bar=progressbar.ProgressBar()
    returndictionary={}
    for dldet in dldets:
        returndictionary.update({str(dldet)+"pg":[]})
        returndictionary.update({str(dldet)+"pdin":[]})
        returndictionary.update({str(dldet)+"pdout":[]})
        
    
    for j in bar(range(len(H0arr))):
        dictionary={}
        for dldet in dldets:
            dictionary.update({str(dldet)+"countdetin":0})
            dictionary.update({str(dldet)+"countdetout":0})
            dictionary.update({str(dldet)+"countnotdetin":0})
            dictionary.update({str(dldet)+"countnotdetout":0})
            
        H0=H0arr[j]
        #print(H0)
        
        for i in range(Nevents):
            hostind=ChooseHost(mask)
            rahost=full_ras[hostind]
            dl_host=cosmo.dl_zH0(full_zs[hostind], H0=H0)
            dl_gw = ExtractObservedGWEvent(full_ras[hostind], full_decs[hostind], dl_host, sigma_dl_prop_ass*dl_host)#this or the next one?#BeforeLinearImplementationVersion
            
            for dldet in dldets:
                if rahost in cut_ras:#careful this doesn't work for MICE as mice has low resolution and many duplicates
                    if dl_gw<dldet:
                        dictionary[str(dldet)+"countdetin"]+=1
                    else:
                        dictionary[str(dldet)+"countnotdetin"]+=1
                else:
                    if dl_gw<dldet:
                        dictionary[str(dldet)+"countdetout"]+=1
                    else:
                        dictionary[str(dldet)+"countnotdetout"]+=1
        
        for dldet in dldets:
            countdetin=dictionary[str(dldet)+"countdetin"]
            countdetout=dictionary[str(dldet)+"countdetout"]
            countnotdetin=dictionary[str(dldet)+"countnotdetin"]
            countnotdetout=dictionary[str(dldet)+"countnotdetout"]

            pg=countdetin/(countdetin+countdetout)
            pdin=countdetin/(countdetin+countnotdetin)
            pdout=countdetout/(countdetout+countnotdetout)

            returndictionary[str(dldet)+"pg"].append(pg)
            returndictionary[str(dldet)+"pdin"].append(pdin)
            returndictionary[str(dldet)+"pdout"].append(pdout)
       
    
    return returndictionary

def ExtractObservedGWEvent(ra_host, dec_host, dl_host, sigma_dl, dl_distr="gaussian-sigmad"): #TODO: BETTER LOCATION OF EVENT, WEIRD IN DEC ESPECIALLY WITH FULL SKY!
    if dl_distr=="gaussian-sigmad":
        dl_obs=np.random.normal(dl_host, sigma_dl)
    #ra_obs, dec_obs = np.random.multivariate_normal([ra_host, dec_host], sigma_radec, 1)[0] 
    return dl_obs

def ChooseHost(mask, weightedcatevents=False, pweights=None):
    if weightedcatevents==False:
        hostind=np.random.choice(mask)
    else:
        hostind=np.random.choice(mask, p=pweights)        
    return hostind


def ps_z_precompute(z):
    rate="constant"
    if rate == 'constant':
        return 1.0
    if rate == 'evolving':
        return (1.0+z)**Lambda
    
def pD_Heaviside_theoretic_precompute(z, H0, dl_det, sigmaprop, cosmo):
    dl=cosmo.dl_zH0(z, H0)
    erfvariable=(dl-dl_det)/(np.sqrt(2)*sigmaprop*dl) #should be this one instead of next one. Jon says no sqrt(2), shouldn't change regardless cause it's just overall factor
    #erfvariable=(dl-dl_det)/(sigmaprop*dl) 
    erfvariable*=-1 #There might be an inconsistency between hitchhiker and scipy in definition of error function, the minus sign produces something much closer to H0**3/gwcosmo
    return 0.5 * (1 + special.erf(erfvariable))

def pD_Heaviside_theoretic_uniform_precompute(z, H0, dl_det, sigmaprop, cosmo): #TODO DOUBLE CHECK
    #print("Using uniform pdet!")
    dl=cosmo.dl_zH0(z, H0)
    sigma=(1./(2.5*sigmaprop))

    if dl_det>dl+dl/sigma:
        return 1
    elif dl_det<dl-dl/sigma:
        return 0
    else:
        return (dl_det-(dl-dl/sigma))/dl