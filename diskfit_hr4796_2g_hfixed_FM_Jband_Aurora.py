####### This is the MCMC fitting code for fitting a disk to HR 4796 data #######



# mpi_or_notmpi = 'mpi'
# mpi_or_notmpi = 'notmpi'

# if new_backend = 0, reset the backend, if not restart the chains.
# Be careful if you change the parameters or walkers #, you have to put new_backend = 1
new_backend = 0

# if first_time = 1 old the mask, reduced data, noise map, and KL vectors are recalculated.
# be careful, for some reason the KL vectors are slightly different on different machines. 
# if you see weird stuff in the FM models (for example in plotting the results), just remake them
first_time = 0

import sys, os, glob
import socket 

if socket.gethostname()== 'MT-101942':
    datadir='/Users/jmazoyer/Dropbox/ExchangeFolder/data_python/Aurora/160323_J_Spec/'
else:
    datadir='/home/jmazoyer/data_python/Aurora/160323_J_Spec/'


os.environ["OMP_NUM_THREADS"] = "1"
import distutils.dir_util
import warnings

import numpy as np
import math as mt
import astropy.io.fits as fits

import time
from datetime import datetime
from astropy.convolution import convolve
import scipy.ndimage.interpolation as interpol
from scipy.integrate import quad

import pyklip.instruments.GPI as GPI
import pyklip.parallelized as parallelized
from pyklip.fmlib.diskfm import DiskFM
import pyklip.fm as fm

from check_gpi_satspots import check_gpi_satspots

# import pickle as pickle
from emcee import EnsembleSampler
from emcee import backends
import contextlib

from multiprocessing import cpu_count

from multiprocessing import Pool
from schwimmbad import MPIPool as MPIPool


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

########################################################
########################################################
#### Fonctions extraite de anadisk
########################################################
########################################################
def integrand_dxdy_2g(xp,yp_dy2,yp2,zp,zp2,zpsi_dx,zpci,R1,R2,beta,a_r,g1,g1_2,g2,g2_2, alpha, ci, si,maxe,dx, dy,k):

    # compute the scattering integrand
    # see analytic-disk.nb

    xx=(xp * ci + zpsi_dx)

    d1 = mt.sqrt((yp_dy2 + xx*xx))

    if (d1  < R1 or d1 > R2):
        return 0.0

    d2 = xp*xp + yp2 + zp2

    #The line of sight scattering angle
    cos_phi=xp/mt.sqrt(d2)
    # phi=np.arccos(cos_phi)

    #Henyey Greenstein function
    hg1=k*alpha*(1. - g1_2)/(1. + g1_2 - (2*g1*cos_phi))**1.5
    hg2=k*(1-alpha)*(1. - g2_2)/(1. + g2_2 - (2*g2*cos_phi))**1.5

    hg = hg1+hg2

    #Radial power low r propto -beta
    int1 = hg*(R1/d1)**beta

    #The scale height function
    zz = (zpci - xp*si)
    hh = (a_r*d1)
    expo = zz*zz/(hh*hh)

    # if expo > 2*maxe:   # cut off exponential after 28 e-foldings (~ 1E-06)
    #     return 0.0 

    int2 = np.exp(0.5*expo) 
    int3 = int2 * d2

    return int1/int3

def gen_disk_dxdy_2g(R1=74.42, R2=82.45, beta=1.0, aspect_ratio=0.1, g1=0.6, g2=-0.6, alpha=0.7, inc=76.49, pa=30, distance=72.8, psfcenx=140,psfceny=140, sampling=1, mask=None, dx=0, dy=0.):

    # starttime=datetime.now()

    #GPI FOV stuff
    # 
    # nspaxels = 200 #The number of lenslets along one side 
    # fov_square = pixscale*nspaxels #The square FOV of GPI 
    # max_fov = mt.sqrt(2)*fov_square/2 #maximum FOV in arcseconds from the center to the edge
    # xsize = max_fov*distance #maximum radial distance in AU from the center to the edge

    #Get The number of resolution elements in the radial direction from the center
    # sampling = 1
    # npts = mt.ceil(nspaxels*mt.sqrt(2)/2/sampling)#The number of pixels to use from the center to the edge
    pixscale = 0.01414 #GPI's pixel scale
    dim=281. #for now the GPI dimensions are hardcoded
    
    max_fov = dim/2.*pixscale #maximum radial distance in AU from the center to the edge
    npts=int(np.floor(dim/sampling))
    xsize = max_fov*distance #maximum radial distance in AU from the center to the edge

    #The coordinate system here [x,y,z] is defined :
    # +ve x is the line of sight 
    # +ve y is going right from the center
    # +ve z is going up from the center

    # y = np.linspace(0,xsize,num=npts/2)
    y = np.linspace(-xsize,xsize,num=npts)
    z = np.linspace(-xsize,xsize,num=npts)

    #Only need to compute half the image
    # image =np.zeros((npts,npts/2+1))
    image =np.zeros((npts,npts))

    #Some things we can precompute ahead of time
    maxe = mt.log(np.finfo('f').max) #The log of the machine precision
    
    #Inclination Calculations
    incl = np.radians(90-inc)
    ci = mt.cos(incl) #Cosine of inclination
    si = mt.sin(incl) #Sine of inclination
    
    #Position angle calculations
    pa_rad=np.radians(90-pa) #The position angle in radians
    cos_pa=mt.cos(pa_rad) #Calculate these ahead of time
    sin_pa=mt.sin(pa_rad)
    
    #HG g value squared
    g1_2 = g1*g1 #First HG g squared
    g2_2 = g2*g2 #Second HG g squared
    #Constant for HG function
    k=1./(4*np.pi)

    #The aspect ratio
    a_r=aspect_ratio

    #Henyey Greenstein function at 90
    hg1_90=k*alpha*(1. - g1_2)/(1. + g1_2)**1.5
    hg2_90=k*(1-alpha)*(1. - g2_2)/(1. + g2_2)**1.5

    hg_90 = hg1_90+hg2_90

    #If there's no mask then calculate for the full image
    if len(np.shape(mask)) < 2:

        for i,yp in enumerate(y):
            for j,zp in enumerate(z):

                #This rotates the coordinates in the image frame
                yy=yp*cos_pa - zp * sin_pa  #Rotate the y coordinate by the PA
                zz=yp*sin_pa + zp * cos_pa  #Rotate the z coordinate by the PA

                #The distance from the center (in each coordinate) squared
                y2 = yy*yy
                z2 = zz*zz

                #This rotates the coordinates in and out of the sky
                zpci=zz*ci #Rotate the z coordinate by the inclination. 
                zpsi=zz*si
                #Subtract the offset
                zpsi_dx = zpsi - dx

                #The distance from the offset squared
                yy_dy=yy-dy
                yy_dy2=yy_dy*yy_dy

                image[j,i]=  quad(integrand_dxdy_2g, -R2, R2, epsrel=0.5e-3,limit=75,args=(yy_dy2,y2,zp,z2,zpsi_dx,zpci,R1,R2,beta,a_r,g1,g1_2,g2,g2_2, alpha,ci,si,maxe,dx,dy,k))[0]
            

    #If there is a mask then don't calculate disk there
    else: 
        hmask = mask
        # hmask = mask[:,140:] #Use only half the mask

        for i,yp in enumerate(y):
            for j,zp in enumerate(z):
                
                # if hmask[j,npts/2+i]: #This assumes that the input mask has is the same size as the desired image (i.e. ~ size / sampling)
                if hmask[j,i]:

                   image[j,i] = 0.#np.nan
                
                else:

                    #This rotates the coordinates in the image frame
                    yy=yp*cos_pa - zp * sin_pa  #Rotate the y coordinate by the PA
                    zz=yp*sin_pa + zp * cos_pa  #Rotate the z coordinate by the PA

                    #The distance from the center (in each coordinate) squared
                    y2 = yy*yy
                    z2 = zz*zz

                    #This rotates the coordinates in and out of the sky
                    zpci=zz*ci #Rotate the z coordinate by the inclination. 
                    zpsi=zz*si
                    #Subtract the offset
                    zpsi_dx = zpsi - dx

                    #The distance from the offset squared
                    yy_dy=yy-dy
                    yy_dy2=yy_dy*yy_dy

                    image[j,i]=  quad(integrand_dxdy_2g, -R2, R2, epsrel=0.5e-3,limit=75,args=(yy_dy2,y2,zp,z2,zpsi_dx,zpci,R1,R2,beta,a_r,g1,g1_2,g2,g2_2, alpha,ci,si,maxe,dx,dy,k))[0]

    # print("Running time: ", datetime.now()-starttime)

    # # normalize the HG function by the width
    image = image/a_r

    # normalize the HG function at the PA
    image = image/hg_90

    return image

########################################################
########################################################

def lnpb(theta):
    lp = logp(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + logl(theta)

########################################################

# create model 2g h fixed
def call_gen_disk_2g(theta, wheremask2generatedisk):
    r1 = mt.exp(theta[0])
    r2 = mt.exp(theta[1])
    beta = theta[2]
    g1 = theta[3]
    g2 = theta[4]
    alpha = theta[5]
    inc = np.degrees(np.arccos(theta[6]))
    pa = theta[7]
    dx = theta[8]
    dy = theta[9]
    norm = theta[10]
    # offset = theta[11]

    distance = 72 

    #generate the model
    model = mt.exp(norm)*gen_disk_dxdy_2g(R1=r1,R2=r2, beta=beta, aspect_ratio=0.01, g1=g1, g2=g2,alpha=alpha, inc=inc, pa=pa, distance=distance, dx=dx, dy=dy, mask = wheremask2generatedisk) #+ offset

    return model


########################################################
# Log likelihood 2g h fixed
def logl(theta):
    
    model=call_gen_disk_2g(theta, wheremask2generatedisk)
    
    modelconvolved = convolve(model,psf, boundary = 'wrap') 
    diskobj.update_disk(modelconvolved)
    model_fm = diskobj.fm_parallelized()[0]

    # fits.writeto('/Users/jmazoyer/Dropbox/ExchangeFolder/data_python/Aurora/160318_H_Spec/klip_fm_files/model_insidelogl.fits', model, clobber='True')
    # fits.writeto('/Users/jmazoyer/Dropbox/ExchangeFolder/data_python/Aurora/160318_H_Spec/klip_fm_files/modelnonfm_insidelogl.fits', modelconvolved, clobber='True')
    # fits.writeto('/Users/jmazoyer/Dropbox/ExchangeFolder/data_python/Aurora/160318_H_Spec/klip_fm_files/modelfm_insidelogl.fits', model_fm, clobber='True')
    # fits.writeto('/Users/jmazoyer/Dropbox/ExchangeFolder/data_python/Aurora/160318_H_Spec/klip_fm_files/residuals_insidelogl.fits', reduced_data-model_fm, clobber='True')

    model_fm[wheremask2minimize]=float('nan')

    res=(reduced_data-model_fm)/noise
    
    lp = np.nansum (-0.5*(res*res))

    return (lp)
########################################################
# Priors 
def logp(theta):

    r1 = mt.exp(theta[0])
    r2 = mt.exp(theta[1])
    beta = theta[2]
    g1 = theta[3]
    g2 = theta[4]
    alpha = theta[5]
    cinc = theta[6]
    pa = theta[7]
    dx = theta[8]
    dy = theta[9]
    lognorm = theta[10]
    # offset = theta[10]

    if ( r1 < 60 or r1 > 79 ): #Can't be bigger than 200 AU
        return -np.inf
 
    # - rout = Logistic We arbitralily cut the prior at r2 = 100 (~25 AU large) because this parameter is very limited by the ADI
    if ((r2 > 82) and (r2 < 102)):
        prior_rout = 1./(1.+np.exp(40.*(r2-100)))
    else: return -np.inf

    # if ( r2 < 82  or r2 > 110 ): #I've tested some things empirically and I don't see a big difference between 500 and 1000 AU. 
    #     return -np.inf

    if (beta < 6 or beta > 25):
        return -np.inf

    # if (a_r < 0.0001 or a_r > 0.5 ): #The aspect ratio  
    #     return -np.inf

    if (g1 < 0.05 or g1 > 0.9999):
        return -np.inf

    if (g2 < -0.9999 or g2 > -0.05):
        return -np.inf

    if (alpha < 0.1 or alpha > 0.9999):
        return -np.inf    

    if (np.arccos(cinc) < np.radians(70)  or np.arccos(cinc) > np.radians(80)): 
        return -np.inf

    if (pa < 20 or pa > 30):
        return -np.inf
   
    if (dx > 0) or (dx < -10): #The x offset
        return -np.inf

    if (dy > 10) or (dy < 0): #The y offset
        return -np.inf

    if (lognorm < np.log(0.0001) or lognorm > np.log(100)):
        return -np.inf    
    # otherwise ... 

    return np.log(prior_rout)
    # return 0.0

######################################################## 
def make_disk_mask(dim, estimPA, estiminclin, estimminr,estimmaxr, xcen=140., ycen=140.):
    PA_rad = (90 + estimPA)*np.pi/180.
    x = np.arange(dim, dtype=np.float)[None,:] - xcen
    y = np.arange(dim, dtype=np.float)[:,None] - ycen
    rho2d = np.sqrt(x**2 + y**2)

    x1 = x * np.cos(PA_rad) + y * np.sin(PA_rad)
    y1 = -x * np.sin(PA_rad) + y * np.cos(PA_rad)
    x = x1
    y = y1 / np.cos(estiminclin*np.pi/180.)
    rho2dellip = np.sqrt(x**2 + y**2)

    mask_object_astro_zeros = np.ones((dim,dim))
    mask_object_astro_zeros[np.where((rho2dellip > estimminr) & (rho2dellip < estimmaxr))] = 0.

    return mask_object_astro_zeros

########################################################
def make_noise_map(reduced_data, PAs, mask_object_astro_zeros, xcen=140., ycen=140., delta_raddii=3):

    dim = mask_object_astro_zeros.shape[0]
    # create rho2D for the rings
    x = np.arange(dim, dtype=np.float)[None,:] - xcen
    y = np.arange(dim, dtype=np.float)[:,None] - ycen
    rho2d = np.sqrt(x**2 + y**2)

    total_paralax = np.max(PAs) - np.min(PAs)

    # create nan and zeros masks for the disk    
    mask_disk_all_angles_nans = np.ones((dim, dim))
    
    anglesrange = PAs - np.median(PAs)
    anglesrange = np.append(anglesrange,[0])

    for index_angle, PAshere in enumerate(anglesrange):
        rot_disk = np.round(interpol.rotate(mask_object_astro_zeros, PAshere, reshape=False, mode='wrap'))
        mask_disk_all_angles_nans[np.where(rot_disk == 0)] = np.nan
    
    
    noise_map = np.zeros((dim, dim))
    for i_ring in range(0, int(np.floor(xcen / delta_raddii)) - 2):
        image_masked = reduced_data * mask_disk_all_angles_nans
        wh_rings = np.where((rho2d >= i_ring * delta_raddii) &
                        (rho2d < (i_ring + 1) * delta_raddii))
        noise_map[wh_rings] = np.nanstd(image_masked[wh_rings])
    return noise_map


def make_noise_map_no_mask(reduced_data_wahaj_trick, xcen=140., ycen=140., delta_raddii=3):

    dim = reduced_data_wahaj_trick.shape[1]
    # create rho2D for the rings
    x = np.arange(dim, dtype=np.float)[None,:] - xcen
    y = np.arange(dim, dtype=np.float)[:,None] - ycen
    rho2d = np.sqrt(x**2 + y**2)

    noise_map = np.zeros((dim, dim))
    for i_ring in range(0, int(np.floor(xcen / delta_raddii)) - 2):
        wh_rings = np.where((rho2d >= i_ring * delta_raddii) &
                        (rho2d < (i_ring + 1) * delta_raddii))
        noise_map[wh_rings] = np.nanstd(reduced_data_wahaj_trick[wh_rings])
    return noise_map

#####################################################
############### Start of the fitting ################
#####################################################

#MCMC Parameters
nwalkers = 192
ndim     = 11 #Number of dimensions

niter = 4000


#The PSF centers
xcen=140. #np.float(rstokes.extheader['PSFCENTX'])
ycen=140. #np.float(rstokes.extheader['PSFCENTY'])


file_prefix = 'Jband_hr4796_klipFM_MaxiNoise'

klipdir = datadir + 'klip_fm_files/'
distutils.dir_util.mkpath(klipdir)
mcmcresultdir = datadir + 'results_MCMC/'
distutils.dir_util.mkpath(mcmcresultdir)

# removed_slices = np.concatenate(([0,1,2],[36]))
removed_slices = None

# measure the PSF from the satspots and identify angles where the disk intersect the satspots
excluded_files = check_gpi_satspots(datadir, removed_slices= removed_slices, SavePSF=True, name_psf= file_prefix+'_SatSpotPSF', SaveAll=False)

psf = fits.getdata(datadir + file_prefix+'_SatSpotPSF.fits')

# load the raw data
filelist = glob.glob(datadir + "*distorcorr.fits")
# for excluded_filesi in excluded_files: 
#     if excluded_filesi in filelist: filelist.remove(excluded_filesi)

dataset = GPI.GPIData(filelist, quiet=True, skipslices=removed_slices)


dataset.OWA = 98
mov_here = 8
KLMODE = [3] 

#collapse the data spectrally
dataset.spectral_collapse(align_frames=True, numthreads =1)

#create the masks
if first_time == 1:
    #create the mask where the non convoluted disk is going to be generated. To gain time, it is tightely adjusted to the expected models BEFORE convolution
    mask_disk_zeros =  make_disk_mask(281, 27., 76.,40.,100., xcen=xcen, ycen=ycen)
    mask2generatedisk = 1 - mask_disk_zeros
    fits.writeto( klipdir + file_prefix+ '_mask2generatedisk.fits', mask2generatedisk, clobber='True')

    ### we create a second mask for the minimization a little bit larger (because model expect to grow with the PSF convolution and the FM) 
    ### and we can also exclude the center region where there are too much speckles 
    mask_disk_zeros =  make_disk_mask(281, 27., 76.,40.,120., xcen=xcen, ycen=ycen)

    mask_speckle_region = np.ones((281,281))
    # x = np.arange(281, dtype=np.float)[None,:] - xcen
    # y = np.arange(281, dtype=np.float)[:,None] - ycen
    # rho2d = np.sqrt(x**2 + y**2)
    # mask_speckle_region[np.where(rho2d < 21)] = 0.
    mask2minimize = mask_speckle_region*(1-mask_disk_zeros)

    fits.writeto( klipdir + file_prefix+ '_mask2minimize.fits', mask2minimize, clobber='True')


mask2generatedisk = fits.getdata(klipdir + file_prefix + '_mask2generatedisk.fits')
mask2minimize = fits.getdata(klipdir + file_prefix + '_mask2minimize.fits')

mask2generatedisk[np.where(mask2generatedisk == 0.)] = np.nan
wheremask2generatedisk = (mask2generatedisk != mask2generatedisk)

mask2minimize[np.where(mask2minimize == 0.)] = np.nan
wheremask2minimize = (mask2minimize != mask2minimize)

if first_time == 1:
    # measure the data, you can do it only once to save time
    parallelized.klip_dataset(dataset, numbasis=KLMODE, maxnumbasis=len(filelist), annuli=1, subsections=1, mode='ADI',
                              outputdir=klipdir, fileprefix=file_prefix + '_Measurewithparallelized',
                              aligned_center=[140, 140], highpass=False, minrot=mov_here, calibrate_flux=False)

# load the data
reduced_data = fits.getdata(klipdir + file_prefix + '_Measurewithparallelized-KLmodes-all.fits')[0] ### we take only the first KL mode

if first_time == 1:
    #measure the noise Wahhaj trick
    dataset.PAs = -dataset.PAs
    parallelized.klip_dataset(dataset, numbasis=KLMODE, maxnumbasis=len(filelist), annuli=1, subsections=1, mode='ADI',
                              outputdir=klipdir, fileprefix=file_prefix + '_WahhajTrick',
                              aligned_center=[xcen, ycen], highpass=False, minrot=mov_here, calibrate_flux=False)

    reduced_data_wahhajtrick = fits.getdata(klipdir + file_prefix + '_WahhajTrick-KLmodes-all.fits')[0]
    noise = make_noise_map_no_mask(reduced_data_wahhajtrick, xcen=xcen, ycen=ycen, delta_raddii=3)
    noise[np.where(noise == 0)] = np.nan
    
    #### We know our noise is too small
    noise = 3*noise
    fits.writeto( klipdir + file_prefix+ '_noisemap.fits', noise, clobber='True')
    
    dataset.PAs = -dataset.PAs
    os.remove(klipdir + file_prefix + '_WahhajTrick-KLmodes-all.fits')
    del reduced_data_wahhajtrick

# load the noise
noise = fits.getdata(klipdir + file_prefix + '_noisemap.fits')



if first_time == 1:
    ### create a first model to check the begining parameter and initialize the FM. We will clear all useless variables befire starting the MCMC
    ### Be careful that this model is the one that you think is the minimum because the FM is not completely linear so you have to measure the FM on something 
    ### already close to the disk
    theta_here = (np.log(74.4),np.log(99.9),10.4, 0.82, -0.16, 0.33, np.cos(np.radians(76.79)), 27, -0.2,1.4, np.log(80))

    #generate the model
    model_here =call_gen_disk_2g(theta_here, wheremask2generatedisk)

    model_here_convolved = convolve(model_here,psf, boundary = 'wrap')
    fits.writeto( klipdir + file_prefix+ '_model_convolved_first.fits', model_here_convolved, clobber='True')

model_here_convolved = fits.getdata(klipdir + file_prefix + '_model_convolved_first.fits')

if first_time == 1:
    # measure the KL vector
    diskobj = DiskFM([len(filelist), 281, 281], KLMODE,dataset, model_here_convolved, annuli=1, subsections=1, 
                            basis_filename = klipdir + file_prefix+ '_KLbasis.pkl', save_basis = True, load_from_basis = False, numthreads =1)

    fm.klip_dataset(dataset, diskobj, numbasis=KLMODE,maxnumbasis = len(filelist), annuli=1, subsections=1, mode='ADI',
                            outputdir=klipdir, fileprefix= file_prefix,
                            aligned_center=[xcen, ycen], mute_progression=True, highpass=False, minrot=mov_here, calibrate_flux=False,numthreads =1)


diskobj = DiskFM([len(filelist), 281, 281], KLMODE,dataset, model_here_convolved, annuli=1, subsections=1,
                        basis_filename = klipdir + file_prefix+ '_KLbasis.pkl', save_basis = False, load_from_basis = True, numthreads =1)


diskobj.update_disk(model_here_convolved)
modelfm_here = diskobj.fm_parallelized()[0] ### we take only the first KL modemode
fits.writeto( klipdir + file_prefix+ '_modelfm_first.fits', modelfm_here, clobber='True')

## We have initialized the variables we need and we now cleaned the ones that do not 
## need to be passed to the pus during the MCMC

del mask2minimize
del mask2generatedisk

del dataset
del KLMODE
del filelist
del mov_here

del model_here_convolved
del modelfm_here

## Let's start the MCMC

sys.stdout.flush() 
f1=open(mcmcresultdir + 'output_job_aurora2.txt', 'a+')
f1.write("\n Begin the MCMC")
f1.close()
sys.stdout.flush() 

# Set up the backend
# Don't forget to clear it in case the file already exists
filename_backend = mcmcresultdir + file_prefix+"backend_file_mcmc.h5"
backend = backends.HDFBackend(filename_backend)
if new_backend ==1:
    backend.reset(nwalkers, ndim)

f1=open(mcmcresultdir + 'output_job_aurora2.txt', 'a+')
f1.write("\n Size of the backend initially: {0}".format(backend.iteration))
f1.close()
sys.stdout.flush() 


#Let's start the clock
startTime = datetime.now()

if __name__ == '__main__':
    with contextlib.closing(Pool()) as pool:
    # with contextlib.closing(MPIPool()) as pool:
        # if not pool.is_master():
        #     pool.wait()
        #     sys.exit(0)
        
        # Set up the Sampler. I purposefully passed the variables (KL modes, reduced data, masks) in global variables to save time
        # as advised in https://emcee.readthedocs.io/en/latest/tutorials/parallel/
        sampler = EnsembleSampler(nwalkers, ndim, lnpb, pool = pool, backend=backend)
        
        if new_backend ==1:
            #############################################################
            # Initialize the walkers. The best technique seems to be 
            # to start in a small ball around the a priori preferred position. 
            # Dont worry, the walkers quickly branch out and explore the 
            # rest of the space.
            w0 = np.random.uniform( theta_here[0]*0.999 ,   theta_here[0]*1.001 ,   size=(nwalkers))    # r1 log[AU]
            w1 = np.random.uniform( theta_here[1]*0.999 ,   theta_here[1]*1.001 ,   size=(nwalkers))    # r2 log[AU]
            w2 = np.random.uniform( theta_here[2]*0.99  ,   theta_here[2]*1.01  ,   size=(nwalkers))    #beta 
            w3 = np.random.uniform( theta_here[3]*0.99  ,   theta_here[3]*1.01  ,   size=(nwalkers))    #g1
            w4 = np.random.uniform( theta_here[4]*0.99  ,   theta_here[4]*1.01  ,   size=(nwalkers))    #g2
            w5 = np.random.uniform( theta_here[5]*0.99  ,   theta_here[5]*1.01  ,   size=(nwalkers))    #alpha
            w6 = np.random.uniform( theta_here[6]*0.99  ,   theta_here[6]*1.01  ,   size=(nwalkers))    #cinc 
            w7 = np.random.uniform( theta_here[7]*0.99  ,   theta_here[7]*1.01  ,   size=(nwalkers))    #pa [degrees]
            w8 = np.random.uniform( theta_here[8]*0.99  ,   theta_here[8]*1.01  ,   size=(nwalkers))    # Xoffset is in AU in minor axis direction + => towards NORTH WEST
            w9 = np.random.uniform( theta_here[9]*0.99  ,   theta_here[9]*1.01  ,   size=(nwalkers))    # Yoffset is in AU in major axis direction + => towards SOUTH WEST
            w10= np.random.uniform( theta_here[10]*0.99 ,   theta_here[10]*1.01  ,   size=(nwalkers))    #log normalizing factor

            p0 = np.dstack((w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10))
            #############################################################

            sampler.run_mcmc(p0[0], niter, progress=True)
        else:
            sampler.run_mcmc(None, niter, progress=True)


sys.stdout.flush() 
ncpu = cpu_count()
f1=open(mcmcresultdir + 'output_job_aurora2.txt', 'a+')
f1.write("\n time for {0} iterations with {1} walkers and {2} cpus: {3}".format(niter, nwalkers,ncpu, datetime.now()-startTime))
f1.close()
sys.stdout.flush() 
