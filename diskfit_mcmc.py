####### This is the MCMC fitting code for fitting a disk to HR 4796 data #######



# mpi_or_notmpi = 'mpi'
# mpi_or_notmpi = 'notmpi'

# if new_backend = 0, reset the backend, if not restart the chains.
# Be careful if you change the parameters or walkers #, you have to put new_backend = 1
new_backend = 0

# if first_time = 1 old the mask, reduced data, noise map, and KL vectors are recalculated.
# be careful, for some reason the KL vectors are slightly different on different machines.
# if you see weird stuff in the FM models (for example in plotting the results), just remake them
first_time = 1


import sys, os, glob
import socket


if socket.gethostname()== 'MT-101942':
    datadir='/Users/jmazoyer/Dropbox/ExchangeFolder/data_python/Aurora/20180128_H_Spec_injection/'
else:
    datadir='/home/jmazoyer/data_python/Aurora/20180128_H_Spec_injection/'


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

from anadisk_johan import gen_disk_dxdy_2g, gen_disk_dxdy_3g


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)



distance = 72
dimension = 281 # pixel
distance = 72 #pc

pixscale = 0.01414 #GPI's pixel scale



def lnpb(theta):

    lp = logp(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + logl(theta)

########################################################

#######################################################
def call_gen_disk_2g(theta):
    """ call the disk model from a set of parameters. 2g SPF
        use DIMENSION, PIXSCALE_INS and DISTANCE_STAR and
        wheremask2generatedisk as global variables

    Args:
        theta: list of parameters of the MCMC

    Returns:
        a 2d model
    """

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

    #generate the model
    model = mt.exp(norm) * gen_disk_dxdy_2g(DIMENSION,
                                            R1=r1,
                                            R2=r2,
                                            beta=beta,
                                            aspect_ratio=0.01,
                                            g1=g1,
                                            g2=g2,
                                            alpha=alpha,
                                            inc=inc,
                                            pa=pa,
                                            dx=dx,
                                            dy=dy,
                                            mask=wheremask2generatedisk,
                                            pixscale=PIXSCALE_INS,
                                            distance=DISTANCE_STAR)  #+ offset

    return model


########################################################
def call_gen_disk_3g(theta):
    """ call the disk model from a set of parameters. 3g SPF
        use DIMENSION, PIXSCALE_INS and DISTANCE_STAR  and
        wheremask2generatedisk as global variables

    Args:
        theta: list of parameters of the MCMC

    Returns:
        a 2d model
    """

    r1 = mt.exp(theta[0])
    r2 = mt.exp(theta[1])
    beta = theta[2]
    g1 = theta[3]
    g2 = theta[4]
    g3 = theta[5]
    alpha1 = theta[6]
    alpha2 = theta[7]
    inc = np.degrees(np.arccos(theta[8]))
    pa = theta[9]
    dx = theta[10]
    dy = theta[11]
    norm = mt.exp(theta[12])

    #generate the model
    model = mt.exp(norm) * gen_disk_dxdy_3g(DIMENSION,
                                            R1=r1,
                                            R2=r2,
                                            beta=beta,
                                            aspect_ratio=0.01,
                                            g1=g1,
                                            g2=g2,
                                            g3=g3,
                                            alpha1=alpha1,
                                            alpha2=alpha2,
                                            inc=inc,
                                            pa=pa,
                                            dx=dx,
                                            dy=dy,
                                            mask=wheremask2generatedisk,
                                            pixscale=PIXSCALE_INS,
                                            distance=DISTANCE_STAR)  #+ offset
    return model


########################################################
# Log likelihood
def logl(theta):

    """ measure the Chisquare (log of the likelyhood) of the parameter set.
        create disk
        convolve by the PSF (psf is global)
        do the forward modeling (diskFM obj is global)
        nan out when it is out of the zone (zone mask is global)
        subctract from data and divide by noise (data and noise are global)

    Args:
        theta: list of parameters of the MCMC

    Returns:
        Chisquare
    """

    if len(theta) == 11:
        model=call_gen_disk_2g(theta)

    if len(theta) == 13:
        model=call_gen_disk_3g(theta)


    modelconvolved = convolve(model,psf, boundary = 'wrap')
    diskobj.update_disk(modelconvolved)
    model_fm = diskobj.fm_parallelized()[0]

    # reduced data have already been naned outside of the minimization
    # zone, so we don't need to do it also for model_fm
    res=(reduced_data-model_fm)/noise

    Chisquare= np.nansum (-0.5*(res*res))

    return Chisquare

########################################################
# Log Priors
def logp(theta):
    """ measure the log of the priors of the parameter set.

    Args:
        theta: list of parameters of the MCMC

    Returns:
        log of priors
    """

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

    if ( r1 < 60 or r1 > 80 ): #Can't be bigger than 200 AU
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
    if len(theta_here) == 11:
        if (g1 < 0.05 or g1 > 0.9999):
            return -np.inf

        if (g2 < -0.9999 or g2 > -0.05):
            return -np.inf

        if (alpha < 0.1 or alpha > 0.9999):
            return -np.inf

    if len(theta_here) == 13:
        if (g1 < -0.9999999 or g1 > 0.9999999):
            return -np.inf

        if (g2 < -0.999999 or g2 >  0.999999):
            return -np.inf

        if (g3 < -0.999999 or g3 > 0.999999):
            return -np.inf

        if (alpha1 < -0.9999999 or alpha1 > 0.9999999):
            return -np.inf

        if (alpha2 < -0.999999 or alpha2 > 0.999999):
            return -np.inf


    if (np.arccos(cinc) < np.radians(70)  or np.arccos(cinc) > np.radians(80)):
        return -np.inf

    if (pa < 20 or pa > 30):
        return -np.inf

    if (dx > 0) or (dx < -10): #The x offset
        return -np.inf

    if (dy > 10) or (dy < 0): #The y offset
        return -np.inf

    if (lognorm < np.log(0.5) or lognorm > np.log(50000)):
        return -np.inf
    # otherwise ...

    return np.log(prior_rout)
    # return 0.0

########################################################
def make_disk_mask(dim, estimPA, estiminclin, estimminr,estimmaxr, xcen=140., ycen=140.):
    PA_rad = (90 + estimPA)*np.pi/180.
    x = np.arange(dim, dtype=np.float)[None,:] - xcen
    y = np.arange(dim, dtype=np.float)[:,None] - ycen

    x1 = x * np.cos(PA_rad) + y * np.sin(PA_rad)
    y1 = -x * np.sin(PA_rad) + y * np.cos(PA_rad)
    x = x1
    y = y1 / np.cos(estiminclin*np.pi/180.)
    rho2dellip = np.sqrt(x**2 + y**2)

    mask_object_astro_zeros = np.ones((dim,dim))
    mask_object_astro_zeros[np.where((rho2dellip > estimminr) & (rho2dellip < estimmaxr))] = 0.

    return mask_object_astro_zeros

########################################################

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

global psf, wheremask2generatedisk, reduced_data
global diskobj

#MCMC Parameters
nwalkers = 192
ndim     = 11 #Number of dimensions

niter = 3000


#The PSF centers
xcen=140. #np.float(rstokes.extheader['PSFCENTX'])
ycen=140. #np.float(rstokes.extheader['PSFCENTY'])

OWA = 98
mov_here = 6
KLMODE_NUMBER = 3

noise_multiplication_factor = 5


file_prefix = 'Hband_hd48524_klipFM_injecteddiskMaxiNoise'


def initialize_the_disk(first_time = 0):



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
    filelist = glob.glob(datadir + "*_distorcorr.fits")

    # We do not remove files for H
    # for excluded_filesi in excluded_files:
    #     if excluded_filesi in filelist: filelist.remove(excluded_filesi)

    dataset = GPI.GPIData(filelist, quiet=True, skipslices=removed_slices)

    dataset.OWA = OWA
    KLMODE = [KLMODE_NUMBER]

    #collapse the data spectrally
    dataset.spectral_collapse(align_frames=True, numthreads =1)

    #create the masks
    if first_time == 1:
        #create the mask where the non convoluted disk is going to be generated. To gain time, it is tightely adjusted to the expected models BEFORE convolution
        mask_disk_zeros =  make_disk_mask(281, 27., 76.,40.,100., xcen=xcen, ycen=ycen)
        mask2generatedisk = 1 - mask_disk_zeros
        fits.writeto( klipdir + file_prefix+ '_mask2generatedisk.fits', mask2generatedisk, overwrite='True')

        ### we create a second mask for the minimization a little bit larger (because model expect to grow with the PSF convolution and the FM)
        ### and we can also exclude the center region where there are too much speckles
        mask_disk_zeros =  make_disk_mask(281, 27., 76.,40.,120., xcen=xcen, ycen=ycen)

        mask_speckle_region = np.ones((281,281))
        # x = np.arange(281, dtype=np.float)[None,:] - xcen
        # y = np.arange(281, dtype=np.float)[:,None] - ycen
        # rho2d = np.sqrt(x**2 + y**2)
        # mask_speckle_region[np.where(rho2d < 21)] = 0.
        mask2minimize = mask_speckle_region*(1-mask_disk_zeros)

        fits.writeto( klipdir + file_prefix+ '_mask2minimize.fits', mask2minimize, overwrite='True')


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

    # we multiply the data by the mask2minimize to avoid having to pass it as a global
    # variable
    reduced_data[wheremask2minimize] = np.nan

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
        noise = noise_multiplication_factor*noise

        fits.writeto( klipdir + file_prefix+ '_noisemap.fits', noise, overwrite='True')

        dataset.PAs = -dataset.PAs
        os.remove(klipdir + file_prefix + '_WahhajTrick-KLmodes-all.fits')
        del reduced_data_wahhajtrick

    # load the noise
    noise = fits.getdata(klipdir + file_prefix + '_noisemap.fits')



    if first_time == 1:
        ### create a first model to check the begining parameter and initialize the FM. We will clear all useless variables befire starting the MCMC
        ### Be careful that this model is the one that you think is the minimum because the FM is not completely linear so you have to measure the FM on something
        ### already close to the disk
        theta_here = (np.log(74.5),np.log(95.8),12.4, 0.825, -0.201, 0.298, np.cos(np.radians(76.8)), 26.64, -2.,0.94, np.log(80))

        #generate the model
        model_here =call_gen_disk_2g(theta_here)

        model_here_convolved = convolve(model_here,psf, boundary = 'wrap')
        fits.writeto( klipdir + file_prefix+ '_model_convolved_first.fits', model_here_convolved, overwrite='True')

    model_here_convolved = fits.getdata(klipdir + file_prefix + '_model_convolved_first.fits')

    if first_time == 1:
        # initialize the DiskFM object
        diskobj = DiskFM(dataset.input.shape, KLMODE,dataset, model_here_convolved,
                                basis_filename = klipdir + file_prefix+ '_basis.h5', save_basis = True,
                                aligned_center=[xcen, ycen]
                                )
        # measure the KL basis and save it
        fm.klip_dataset(dataset, diskobj, numbasis=KLMODE,maxnumbasis = len(filelist), annuli=1, subsections=1, mode='ADI',
                                outputdir=klipdir, fileprefix= file_prefix,
                                aligned_center=[xcen, ycen], mute_progression=True, highpass=False, minrot=mov_here, calibrate_flux=False,numthreads =1)

    # load the the KL basis and define the diskFM object
    diskobj = DiskFM(dataset.input.shape, KLMODE,dataset, model_here_convolved,
                            basis_filename = klipdir + file_prefix+ '_basis.h5', load_from_basis = True)


    # test the diskFM object
    diskobj.update_disk(model_here_convolved)
    modelfm_here = diskobj.fm_parallelized()[0] ### we take only the first KL modemode
    fits.writeto( klipdir + file_prefix+ '_modelfm_first.fits', modelfm_here, overwrite='True')

    ## We have initialized the variables we need and we now cleaned the ones that do not
    ## need to be passed to the cores during the MCMC

    del dataset
    del KLMODE
    del filelist

    del wheremask2minimize

    del mov_here
    del model_here_convolved
    del modelfm_here

## Let's start the MCMC

# sys.stdout.flush()
# f1=open(mcmcresultdir + 'output_job_aurora2.txt', 'a+')
# f1.write("\n Begin the MCMC")
# f1.close()
# sys.stdout.flush()

# # Set up the backend
# # Don't forget to clear it in case the file already exists
# filename_backend = mcmcresultdir + file_prefix+"backend_file_mcmc.h5"
# backend = backends.HDFBackend(filename_backend)
# if new_backend ==1:
#     backend.reset(nwalkers, ndim)

# #Let's start the clock
# startTime = datetime.now()

# if __name__ == '__main__':
#     with contextlib.closing(Pool()) as pool:
#     # with contextlib.closing(MPIPool()) as pool:
#         # if not pool.is_master():
#         #     pool.wait()
#         #     sys.exit(0)

#         # Set up the Sampler. I purposefully passed the variables (KL modes, reduced data, masks) in global variables to save time
#         # as advised in https://emcee.readthedocs.io/en/latest/tutorials/parallel/
#         sampler = EnsembleSampler(nwalkers, ndim, lnpb, pool = pool, backend=backend)

#         if new_backend ==1:
#             #############################################################
#             # Initialize the walkers. The best technique seems to be
#             # to start in a small ball around the a priori preferred position.
#             # Dont worry, the walkers quickly branch out and explore the
#             # rest of the space.
#             w0 = np.random.uniform( theta_here[0]*0.999 ,   theta_here[0]*1.001 ,   size=(nwalkers))    # r1 log[AU]
#             w1 = np.random.uniform( theta_here[1]*0.999 ,   theta_here[1]*1.001 ,   size=(nwalkers))    # r2 log[AU]
#             w2 = np.random.uniform( theta_here[2]*0.99  ,   theta_here[2]*1.01  ,   size=(nwalkers))    #beta
#             w3 = np.random.uniform( theta_here[3]*0.99  ,   theta_here[3]*1.01  ,   size=(nwalkers))    #g1
#             w4 = np.random.uniform( theta_here[4]*0.99  ,   theta_here[4]*1.01  ,   size=(nwalkers))    #g2
#             w5 = np.random.uniform( theta_here[5]*0.99  ,   theta_here[5]*1.01  ,   size=(nwalkers))    #alpha
#             w6 = np.random.uniform( theta_here[6]*0.99  ,   theta_here[6]*1.01  ,   size=(nwalkers))    #cinc
#             w7 = np.random.uniform( theta_here[7]*0.99  ,   theta_here[7]*1.01  ,   size=(nwalkers))    #pa [degrees]
#             w8 = np.random.uniform( theta_here[8]*0.99  ,   theta_here[8]*1.01  ,   size=(nwalkers))    # Xoffset is in AU in minor axis direction + => towards NORTH WEST
#             w9 = np.random.uniform( theta_here[9]*0.99  ,   theta_here[9]*1.01  ,   size=(nwalkers))    # Yoffset is in AU in major axis direction + => towards SOUTH WEST
#             w10= np.random.uniform( theta_here[10]*0.99 ,   theta_here[10]*1.01  ,   size=(nwalkers))    #log normalizing factor

#             p0 = np.dstack((w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10))
#             #############################################################

#             sampler.run_mcmc(p0[0], niter, progress=True)
#         else:
#             sampler.run_mcmc(None, niter, progress=True)


# sys.stdout.flush()
# ncpu = cpu_count()
# f1=open(mcmcresultdir + 'output_job_aurora2.txt', 'a+')
# f1.write("\n time for {0} iterations with {1} walkers and {2} cpus: {3}".format(niter, nwalkers,ncpu, datetime.now()-startTime))
# f1.close()
# sys.stdout.flush()
