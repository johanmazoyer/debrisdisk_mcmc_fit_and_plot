####### This is the MCMC fitting code for fitting a disk to HR 4796 data #######

import os
import sys
import glob
import socket

import distutils.dir_util
import warnings

from multiprocessing import cpu_count
from multiprocessing import Pool

import contextlib

from datetime import datetime

import math as mt
import numpy as np

import astropy.io.fits as fits
from astropy.utils.exceptions import AstropyWarning
from astropy.convolution import convolve

import yaml

import pyklip.instruments.GPI as GPI
import pyklip.parallelized as parallelized
from pyklip.fmlib.diskfm import DiskFM
import pyklip.fm as fm

from emcee import EnsembleSampler
from emcee import backends

from check_gpi_satspots import check_gpi_satspots
from anadisk_johan import gen_disk_dxdy_2g, gen_disk_dxdy_3g
import astro_unit_conversion as convert


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
        model = call_gen_disk_2g(theta)

    if len(theta) == 13:
        model = call_gen_disk_3g(theta)

    modelconvolved = convolve(model, psf, boundary='wrap')
    diskobj.update_disk(modelconvolved)
    model_fm = diskobj.fm_parallelized()[0]

    # reduced data have already been naned outside of the minimization
    # zone, so we don't need to do it also for model_fm
    res = (reduced_data - model_fm) / noise

    Chisquare = np.nansum(-0.5 * (res * res))

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

    if len(theta) == 11:
        alpha1 = theta[5]
        cinc = theta[6]
        pa = theta[7]
        dx = theta[8]
        dy = theta[9]
        lognorm = theta[10]
    if len(theta) == 13:
        g3 = theta[5]
        alpha1 = theta[6]
        alpha2 = theta[7]
        cinc = theta[8]
        pa = theta[9]
        dx = theta[10]
        dy = theta[11]
        lognorm = theta[12]

    # offset = theta[10]

    if (r1 < 60 or r1 > 80):  #Can't be bigger than 200 AU
        return -np.inf

    # - rout = Logistic We arbitralily cut the prior at r2 = 100
    # (~25 AU large) because this parameter is very limited by the ADI
    if ((r2 > 82) and (r2 < 102)):
        prior_rout = 1. / (1. + np.exp(40. * (r2 - 100)))
    else:
        return -np.inf

    # or we can just cut it normally
    # if ( r2 < 82  or r2 > 110 ):
    #     return -np.inf

    if (beta < 6 or beta > 25):
        return -np.inf

    # if (a_r < 0.0001 or a_r > 0.5 ): #The aspect ratio
    #     return -np.inf
    if len(theta) == 11:
        if (g1 < 0.05 or g1 > 0.9999):
            return -np.inf

        if (g2 < -0.9999 or g2 > -0.05):
            return -np.inf

        if (alpha1 < 0.1 or alpha1 > 0.9999):
            return -np.inf

    if len(theta) == 13:
        if (g1 < -0.9999999 or g1 > 0.9999999):
            return -np.inf

        if (g2 < -0.999999 or g2 > 0.999999):
            return -np.inf

        if (g3 < -0.999999 or g3 > 0.999999):
            return -np.inf

        if (alpha1 < -0.9999999 or alpha1 > 0.9999999):
            return -np.inf

        if (alpha2 < -0.999999 or alpha2 > 0.999999):
            return -np.inf

    if (np.arccos(cinc) < np.radians(70) or np.arccos(cinc) > np.radians(80)):
        return -np.inf

    if (pa < 20 or pa > 30):
        return -np.inf

    if (dx > 0) or (dx < -10):  #The x offset
        return -np.inf

    if (dy > 10) or (dy < 0):  #The y offset
        return -np.inf

    if (lognorm < np.log(0.5) or lognorm > np.log(50000)):
        return -np.inf
    # otherwise ...

    return np.log(prior_rout)
    # return 0.0


########################################################
# Log Priors + Log likelihood
def lnpb(theta):
    """ sum the logs of the priors (return of the logp funciton)
        and of the likelyhood (return of the logl function)


    Args:
        theta: list of parameters of the MCMC

    Returns:
        log of priors + log of likelyhood
    """
    lp = logp(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + logl(theta)


########################################################
def make_disk_mask(dim,
                   estimPA,
                   estiminclin,
                   estimminr,
                   estimmaxr,
                   xcen=140.,
                   ycen=140.):
    """ make a zeros mask for a disk


    Args:
        dim: pixel, dimension of the square mask
        estimPA: degree, estimation of the PA
        estiminclin: degree, estimation of the inclination
        estimminr: pixel, inner radius of the mask
        estimmaxr: pixel, outer radius of the mask
        xcen: pixel, center of the mask
        ycen: pixel, center of the mask

    Returns:
        a [dim,dim] array where the mask is at 0 and the rest at 1
    """

    PA_rad = (90 + estimPA) * np.pi / 180.
    x = np.arange(dim, dtype=np.float)[None, :] - xcen
    y = np.arange(dim, dtype=np.float)[:, None] - ycen

    x1 = x * np.cos(PA_rad) + y * np.sin(PA_rad)
    y1 = -x * np.sin(PA_rad) + y * np.cos(PA_rad)
    x = x1
    y = y1 / np.cos(estiminclin * np.pi / 180.)
    rho2dellip = np.sqrt(x**2 + y**2)

    mask_object_astro_zeros = np.ones((dim, dim))
    mask_object_astro_zeros[np.where((rho2dellip > estimminr)
                                     & (rho2dellip < estimmaxr))] = 0.

    return mask_object_astro_zeros


########################################################
def make_noise_map_no_mask(reduced_data, xcen=140., ycen=140., delta_raddii=3):
    """ create a noise map from a image using concentring rings
        and measuring the standard deviation on them

    Args:
        reduced_data: [dim dim] array containing the reduced data
        xcen: pixel, center of the mask
        ycen: pixel, center of the mask
        delta_raddii: pixel, widht of the small concentric rings

    Returns:
        a [dim,dim] array where each concentric rings is at a constant value
            of the standard deviation of the reduced_data
    """

    dim = reduced_data.shape[1]
    # create rho2D for the rings
    x = np.arange(dim, dtype=np.float)[None, :] - xcen
    y = np.arange(dim, dtype=np.float)[:, None] - ycen
    rho2d = np.sqrt(x**2 + y**2)

    noise_map = np.zeros((dim, dim))
    for i_ring in range(0, int(np.floor(xcen / delta_raddii)) - 2):
        wh_rings = np.where((rho2d >= i_ring * delta_raddii)
                            & (rho2d < (i_ring + 1) * delta_raddii))
        noise_map[wh_rings] = np.nanstd(reduced_data[wh_rings])
    return noise_map


def initialize_the_disk(params_mcmc_yaml):
    """ initialize the MCMC by preparing the disk (measure the data, the psf
        the diskFM object, the noise, the masks).
        all the things that will be used by the MCMC are beeing passed as
        global parameters to speed up the parallel MCMC

    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file

    Returns:
        None
    """

    global psf, wheremask2generatedisk, reduced_data, diskobj, DIMENSION, noise
    global DISTANCE_STAR, PIXSCALE_INS
    DISTANCE_STAR = params_mcmc_yaml['DISTANCE_STAR']
    PIXSCALE_INS = params_mcmc_yaml['PIXSCALE_INS']

    # if FIRST_TIME = 1 old the mask, reduced data, noise map, and KL vectors
    # are recalculated. be careful, for some reason the KL vectors are slightly
    # different on different machines. if you see weird stuff in the FM models
    # (for example in plotting the results), just remake them
    FIRST_TIME = params_mcmc_yaml['FIRST_TIME']

    OWA = params_mcmc_yaml['OWA']
    MOVE_HERE = params_mcmc_yaml['MOVE_HERE']
    KLMODE_NUMBER = params_mcmc_yaml['KLMODE_NUMBER']
    REMOVED_SLICES = params_mcmc_yaml['REMOVED_SLICES']
    NOISE_MULTIPLICATION_FACTOR = params_mcmc_yaml[
        'NOISE_MULTIPLICATION_FACTOR']

    theta_init = from_param_to_theta_init(params_mcmc_yaml)

    DATADIR = basedir + params_mcmc_yaml['BAND_DIR']
    FILE_PREFIX = params_mcmc_yaml['FILE_PREFIX']
    klipdir = DATADIR + 'klip_fm_files/'
    distutils.dir_util.mkpath(klipdir)

    #The PSF centers
    xcen = 140.
    ycen = 140.

    # list of the raw data file
    filelist = glob.glob(DATADIR + "*_distorcorr.fits")

    # measure the PSF from the satspots and identify angles where the
    # disk intersect the satspots
    if FIRST_TIME == 1:
        excluded_files = check_gpi_satspots(
            DATADIR,
            removed_slices=params_mcmc_yaml['REMOVED_SLICES'],
            SavePSF=True,
            name_psf=FILE_PREFIX + '_SatSpotPSF',
            SaveAll=False)

    # We do not remove files for now
    # for excluded_filesi in excluded_files:
    #     if excluded_filesi in filelist: filelist.remove(excluded_filesi)

    psf = fits.getdata(DATADIR + FILE_PREFIX + '_SatSpotPSF.fits')

    # load the rww data
    dataset = GPI.GPIData(filelist, quiet=True, skipslices=REMOVED_SLICES)

    #collapse the data spectrally
    dataset.spectral_collapse(align_frames=True, numthreads=1)

    dataset.OWA = OWA
    KLMODE = [KLMODE_NUMBER]
    #assuming square data
    DIMENSION = dataset.input.shape[2]

    #create the masks
    if FIRST_TIME == 1:
        #create the mask where the non convoluted disk is going to be generated.
        # To gain time, it is tightely adjusted to the expected models BEFORE convolution
        mask_disk_zeros = make_disk_mask(
            DIMENSION,
            params_mcmc_yaml['pa_init'],
            params_mcmc_yaml['inc_init'],
            convert.au_to_pix(40, PIXSCALE_INS, DISTANCE_STAR),
            convert.au_to_pix(105, PIXSCALE_INS, DISTANCE_STAR),
            xcen=xcen,
            ycen=ycen)
        mask2generatedisk = 1 - mask_disk_zeros
        fits.writeto(klipdir + FILE_PREFIX + '_mask2generatedisk.fits',
                     mask2generatedisk,
                     overwrite='True')

        # we create a second mask for the minimization a little bit larger
        # (because model expect to grow with the PSF convolution and the FM)
        # and we can also exclude the center region where there are too much speckles
        mask_disk_zeros = make_disk_mask(
            DIMENSION,
            params_mcmc_yaml['pa_init'],
            params_mcmc_yaml['inc_init'],
            convert.au_to_pix(40, PIXSCALE_INS, DISTANCE_STAR),
            convert.au_to_pix(125, PIXSCALE_INS, DISTANCE_STAR),
            xcen=xcen,
            ycen=ycen)

        mask_speckle_region = np.ones((DIMENSION, DIMENSION))
        # x = np.arange(DIMENSION, dtype=np.float)[None,:] - xcen
        # y = np.arange(DIMENSION, dtype=np.float)[:,None] - ycen
        # rho2d = np.sqrt(x**2 + y**2)
        # mask_speckle_region[np.where(rho2d < 21)] = 0.
        mask2minimize = mask_speckle_region * (1 - mask_disk_zeros)

        fits.writeto(klipdir + FILE_PREFIX + '_mask2minimize.fits',
                     mask2minimize,
                     overwrite='True')

    mask2generatedisk = fits.getdata(klipdir + FILE_PREFIX +
                                     '_mask2generatedisk.fits')
    mask2minimize = fits.getdata(klipdir + FILE_PREFIX + '_mask2minimize.fits')

    mask2generatedisk[np.where(mask2generatedisk == 0.)] = np.nan
    wheremask2generatedisk = (mask2generatedisk != mask2generatedisk)

    mask2minimize[np.where(mask2minimize == 0.)] = np.nan
    wheremask2minimize = (mask2minimize != mask2minimize)

    if FIRST_TIME == 1:
        #measure the noise Wahhaj trick
        dataset.PAs = -dataset.PAs
        parallelized.klip_dataset(dataset,
                                  numbasis=KLMODE,
                                  maxnumbasis=len(filelist),
                                  annuli=1,
                                  subsections=1,
                                  mode='ADI',
                                  outputdir=klipdir,
                                  fileprefix=FILE_PREFIX + '_WahhajTrick',
                                  aligned_center=[xcen, ycen],
                                  highpass=False,
                                  minrot=MOVE_HERE,
                                  calibrate_flux=False)

        reduced_data_wahhajtrick = fits.getdata(
            klipdir + FILE_PREFIX + '_WahhajTrick-KLmodes-all.fits')[0]
        noise = make_noise_map_no_mask(reduced_data_wahhajtrick,
                                       xcen=xcen,
                                       ycen=ycen,
                                       delta_raddii=3)
        noise[np.where(noise == 0)] = np.nan

        #### We know our noise is too small
        noise = NOISE_MULTIPLICATION_FACTOR * noise

        fits.writeto(klipdir + FILE_PREFIX + '_noisemap.fits',
                     noise,
                     overwrite='True')

        dataset.PAs = -dataset.PAs
        os.remove(klipdir + FILE_PREFIX + '_WahhajTrick-KLmodes-all.fits')
        del reduced_data_wahhajtrick

    # load the noise
    noise = fits.getdata(klipdir + FILE_PREFIX + '_noisemap.fits')

    if FIRST_TIME == 1:
        # create a first model to check the begining parameter and initialize the FM.
        # We will clear all useless variables befire starting the MCMC
        # Be careful that this model is close to what you think is the minimum
        # because the FM is not completely linear so you have to measure the FM on
        # something already close to the best one

        #generate the model
        model_here = call_gen_disk_2g(theta_init)

        model_here_convolved = convolve(model_here, psf, boundary='wrap')
        fits.writeto(klipdir + FILE_PREFIX + '_model_convolved_first.fits',
                     model_here_convolved,
                     overwrite='True')

    model_here_convolved = fits.getdata(klipdir + FILE_PREFIX +
                                        '_model_convolved_first.fits')

    if FIRST_TIME == 1:
        # initialize the DiskFM object
        diskobj = DiskFM(dataset.input.shape,
                         KLMODE,
                         dataset,
                         model_here_convolved,
                         basis_filename=klipdir + FILE_PREFIX + '_klbasis.h5',
                         save_basis=True,
                         aligned_center=[xcen, ycen])
        # measure the KL basis and save it
        fm.klip_dataset(dataset,
                        diskobj,
                        numbasis=KLMODE,
                        maxnumbasis=len(filelist),
                        annuli=1,
                        subsections=1,
                        mode='ADI',
                        outputdir=klipdir,
                        fileprefix=FILE_PREFIX,
                        aligned_center=[xcen, ycen],
                        mute_progression=True,
                        highpass=False,
                        minrot=MOVE_HERE,
                        calibrate_flux=False,
                        numthreads=1)

    # load the data
    reduced_data = fits.getdata(klipdir + FILE_PREFIX +
                                '-klipped-KLmodes-all.fits')[
                                    0]  ### we take only the first KL mode

    # we multiply the data by the mask2minimize to avoid having to pass it as a global
    # variable
    reduced_data[wheremask2minimize] = np.nan

    # load the the KL basis and define the diskFM object
    diskobj = DiskFM(dataset.input.shape,
                     KLMODE,
                     dataset,
                     model_here_convolved,
                     basis_filename=klipdir + FILE_PREFIX + '_klbasis.h5',
                     load_from_basis=True)

    # test the diskFM object
    diskobj.update_disk(model_here_convolved)
    modelfm_here = diskobj.fm_parallelized()[
        0]  ### we take only the first KL modemode
    fits.writeto(klipdir + FILE_PREFIX + '_modelfm_first.fits',
                 modelfm_here,
                 overwrite='True')

    ## We have initialized the variables we need and we now cleaned the ones that do not
    ## need to be passed to the cores during the MCMC


def initialize_walkers_backend(params_mcmc_yaml):
    """ initialize the MCMC by preparing the initial position of the
        walkers and the backend file

    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file

    Returns:
        if NEW_BACKEND ==1 then [intial position of the walkers, a clean BACKEND]
        if NEW_BACKEND ==0 then [None, the loaded BACKEND]
    """

    # if NEW_BACKEND = 0, reset the backend, if not restart the chains.
    # Be careful if you change the parameters or walkers #, you have to put NEW_BACKEND = 1
    NEW_BACKEND = params_mcmc_yaml['NEW_BACKEND']

    DATADIR = basedir + params_mcmc_yaml['BAND_DIR']
    FILE_PREFIX = params_mcmc_yaml['FILE_PREFIX']

    mcmcresultdir = DATADIR + 'results_MCMC/'
    distutils.dir_util.mkpath(mcmcresultdir)

    NWALKERS = params_mcmc_yaml['NWALKERS']
    N_DIM_MCMC = params_mcmc_yaml['N_DIM_MCMC']

    theta_init = from_param_to_theta_init(params_mcmc_yaml)

    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename_backend = mcmcresultdir + FILE_PREFIX + "_backend_file_mcmc.h5"
    backend_ini = backends.HDFBackend(filename_backend)

    #############################################################
    # Initialize the walkers. The best technique seems to be
    # to start in a small ball around the a priori preferred position.
    # Dont worry, the walkers quickly branch out and explore the
    # rest of the space.
    if NEW_BACKEND == 1:
        init_ball0 = np.random.uniform(theta_init[0] * 0.999,
                                       theta_init[0] * 1.001,
                                       size=(NWALKERS))  # r1 log[AU]
        init_ball1 = np.random.uniform(theta_init[1] * 0.999,
                                       theta_init[1] * 1.001,
                                       size=(NWALKERS))  # r2 log[AU]
        init_ball2 = np.random.uniform(theta_init[2] * 0.99,
                                       theta_init[2] * 1.01,
                                       size=(NWALKERS))  #beta
        init_ball3 = np.random.uniform(theta_init[3] * 0.99,
                                       theta_init[3] * 1.01,
                                       size=(NWALKERS))  #g1
        init_ball4 = np.random.uniform(theta_init[4] * 0.99,
                                       theta_init[4] * 1.01,
                                       size=(NWALKERS))  #g2

        if N_DIM_MCMC == 11:
            init_ball5 = np.random.uniform(theta_init[5] * 0.99,
                                           theta_init[5] * 1.01,
                                           size=(NWALKERS))  #alpha1
            init_ball6 = np.random.uniform(theta_init[6] * 0.99,
                                           theta_init[6] * 1.01,
                                           size=(NWALKERS))  #cinc
            init_ball7 = np.random.uniform(theta_init[7] * 0.99,
                                           theta_init[7] * 1.01,
                                           size=(NWALKERS))  #pa [degrees]
            init_ball8 = np.random.uniform(
                theta_init[8] * 0.99, theta_init[8] * 1.01,
                size=(NWALKERS))  # offset in minor axis
            init_ball9 = np.random.uniform(
                theta_init[9] * 0.99, theta_init[9] * 1.01,
                size=(NWALKERS))  # offset in major axis
            init_ball10 = np.random.uniform(
                theta_init[10] * 0.99, theta_init[10] * 1.01,
                size=(NWALKERS))  #log normalizing factor
            p0 = np.dstack((init_ball0, init_ball1, init_ball2, init_ball3,
                            init_ball4, init_ball5, init_ball6, init_ball7,
                            init_ball8, init_ball9, init_ball10))

        if N_DIM_MCMC == 13:
            init_ball5 = np.random.uniform(theta_init[5] * 0.99,
                                           theta_init[5] * 1.01,
                                           size=(NWALKERS))  #g3
            init_ball6 = np.random.uniform(theta_init[6] * 0.99,
                                           theta_init[6] * 1.01,
                                           size=(NWALKERS))  #alpha1
            init_ball7 = np.random.uniform(theta_init[7] * 0.99,
                                           theta_init[7] * 1.01,
                                           size=(NWALKERS))  #alpha2

            init_ball8 = np.random.uniform(theta_init[8] * 0.99,
                                           theta_init[8] * 1.01,
                                           size=(NWALKERS))  #cinc
            init_ball9 = np.random.uniform(theta_init[9] * 0.99,
                                           theta_init[9] * 1.01,
                                           size=(NWALKERS))  #pa [degrees]
            init_ball10 = np.random.uniform(
                theta_init[10] * 0.99, theta_init[10] * 1.01,
                size=(NWALKERS))  # offset in minor axis
            init_ball11 = np.random.uniform(
                theta_init[11] * 0.99, theta_init[11] * 1.01,
                size=(NWALKERS))  # offset in major axis
            init_ball12 = np.random.uniform(
                theta_init[12] * 0.99, theta_init[12] * 1.01,
                size=(NWALKERS))  #log normalizing factor
            p0 = np.dstack(
                (init_ball0, init_ball1, init_ball2, init_ball3, init_ball4,
                 init_ball5, init_ball6, init_ball7, init_ball8, init_ball9,
                 init_ball10, init_ball11, init_ball12))

        backend_ini.reset(NWALKERS, N_DIM_MCMC)
        return p0[0], backend_ini
    else:
        return None, backend_ini


def from_param_to_theta_init(params_mcmc_yaml):
    """ create a initial set of MCMCparameter from the initial parmeters
        store in the init yaml file
    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file

    Returns:
        initial set of MCMC parameter
    """

    N_DIM_MCMC = params_mcmc_yaml['N_DIM_MCMC']  #Number of interation

    r1_init = params_mcmc_yaml['r1_init']
    r2_init = params_mcmc_yaml['r2_init']
    beta_init = params_mcmc_yaml['beta_init']
    g1_init = params_mcmc_yaml['g1_init']
    g2_init = params_mcmc_yaml['g2_init']
    alpha1_init = params_mcmc_yaml['alpha1_init']
    inc_init = params_mcmc_yaml['inc_init']
    pa_init = params_mcmc_yaml['pa_init']
    dx_init = params_mcmc_yaml['dx_init']
    dy_init = params_mcmc_yaml['dy_init']
    N_init = params_mcmc_yaml['N_init']

    if N_DIM_MCMC == 11:
        theta_init = (np.log(r1_init), np.log(r2_init), beta_init, g1_init,
                      g2_init, alpha1_init, np.cos(np.radians(inc_init)),
                      pa_init, dx_init, dy_init, np.log(N_init))
    if N_DIM_MCMC == 13:
        g3_init = params_mcmc_yaml['g3_init']
        alpha2_init = params_mcmc_yaml['alpha2_init']
        theta_init = (np.log(r1_init), np.log(r2_init), beta_init, g1_init,
                      g2_init, g3_init, alpha1_init, alpha2_init,
                      np.cos(np.radians(inc_init)), pa_init, dx_init, dy_init,
                      np.log(N_init))

    return theta_init


if __name__ == '__main__':

    # warnings.filterwarnings("ignore", category=RuntimeWarning)
    # warnings.filterwarnings("ignore", category=UserWarning)
    # warnings.simplefilter('ignore', category=AstropyWarning)
    if len(sys.argv) == 1:
        str_yalm = 'GPI_Hband_MCMC.yaml'
    else:
        str_yalm = sys.argv[1]

    # test on which machine I am
    if socket.gethostname() == 'MT-101942':
        basedir = '/Users/jmazoyer/Dropbox/ExchangeFolder/data_python/Aurora/'
        progress = True  # if on my local machine, showing the MCMC progress bar
    else:
        basedir = '/home/jmazoyer/data_python/Aurora/'
        progress = False

    # open the parameter file
    with open('initialization_files/' + str_yalm, 'r') as yaml_file:
        params_mcmc_yaml = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # initialize the disk. create a bunch of global variables
    # that will be used in the MCMC to avoid passing them at each core
    # at each iteration
    initialize_the_disk(params_mcmc_yaml)

    # initialize the walkers if necessary. initialize/load the backend
    init_walkers, backend = initialize_walkers_backend(params_mcmc_yaml)

    # load the Parameters necessary to launch the MCMC
    NWALKERS = params_mcmc_yaml['NWALKERS']  #Number of walkers
    N_ITER_MCMC = params_mcmc_yaml['N_ITER_MCMC']  #Number of interation
    N_DIM_MCMC = params_mcmc_yaml['N_DIM_MCMC']  #Number of dimension of the MCMC

    # last chance to remove some global variable to be as light as possible
    # in the MCMC
    del params_mcmc_yaml

    #Let's start the MCMC
    startTime = datetime.now()
    with contextlib.closing(Pool()) as pool:

        # Set up the Sampler. I purposefully passed the variables (KL modes,
        # reduced data, masks) in global variables to save time as advised in
        # https://emcee.readthedocs.io/en/latest/tutorials/parallel/
        sampler = EnsembleSampler(NWALKERS,
                                  N_DIM_MCMC,
                                  lnpb,
                                  pool=pool,
                                  backend=backend)

        sampler.run_mcmc(init_walkers, N_ITER_MCMC, progress=progress)

    print(
        "\n time for {0} iterations with {1} walkers and {2} cpus: {3}".format(
            N_ITER_MCMC, NWALKERS, cpu_count(),
            datetime.now() - startTime))
