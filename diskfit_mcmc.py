# pylint: disable=C0103
"""
MCMC code for fitting a disk 
author: Johan Mazoyer
"""

import os

mpi = True  # mpi or not for parallelization.
basedir = os.environ["EXCHANGE_PATH"]  # the base directory where is 
# your data (using OS environnement variable allow to use same code on 
# different computer without changing this).

progress = False  # if on my local machine and print on console, showing the
# MCMC progress bar. Avoid if print resutls of the code in a file, it will 
# not look pretty

import sys
import glob

import distutils.dir_util
import warnings

from multiprocessing import cpu_count

if mpi:
    from schwimmbad import MPIPool as MultiPool
else:
    # from schwimmbad import MultiPool as MultiPool
    from multiprocessing import Pool as MultiPool

from datetime import datetime

import math as mt
import numpy as np

import astropy.io.fits as fits
from astropy.convolution import convolve
from astropy.wcs import FITSFixedWarning

import yaml

from emcee import EnsembleSampler
from emcee import backends

import pyklip.instruments.GPI as GPI
import pyklip.instruments.Instrument as Instrument

import pyklip.parallelized as parallelized
from pyklip.fmlib.diskfm import DiskFM
import pyklip.fm as fm
import pyklip.rdi as rdi

import make_gpi_psf_for_disks as gpidiskpsf

from anadisk_johan import gen_disk_dxdy_2g, gen_disk_dxdy_3g
import astro_unit_conversion as convert

# recommended by emcee https://emcee.readthedocs.io/en/stable/tutorials/parallel/ 
# and by PyKLIPto avoid that NumPy automatically parallelizes some operations, 
# which kill the speed
os.environ["OMP_NUM_THREADS"] = "1"


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
    # TODO check where is the position of the star in the model
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
    norm = mt.exp(theta[10])
    # offset = theta[11]

    #generate the model
    model = norm * gen_disk_dxdy_2g(DIMENSION,
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
                                    mask=WHEREMASK2GENERATEDISK,
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
    model = norm * gen_disk_dxdy_3g(DIMENSION,
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
                                    mask=WHEREMASK2GENERATEDISK,
                                    pixscale=PIXSCALE_INS,
                                    distance=DISTANCE_STAR)  #+ offset
    return model


########################################################
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

    modelconvolved = convolve(model, PSF, boundary='wrap')
    DISKOBJ.update_disk(modelconvolved)
    model_fm = DISKOBJ.fm_parallelized()[0]

    # reduced data have already been naned outside of the minimization
    # zone, so we don't need to do it also for model_fm
    res = (REDUCED_DATA - model_fm) / NOISE

    Chisquare = np.nansum(-0.5 * (res * res))

    return Chisquare


########################################################
def logp(theta):
    """ measure the log of the priors of the parameter set.
     This function still have a lot of parameters hard coded here
     Also you can change the prior shape directly here.

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
    # else:
    #     prior_rout = 0.

    if (beta < 1 or beta > 30):
        return -np.inf

    # if (a_r < 0.0001 or a_r > 0.5 ): #The aspect ratio
    #     return -np.inf
    if len(theta) == 11:
        if (g1 < 0.05 or g1 > 0.9999):
            return -np.inf

        if (g2 < -0.9999 or g2 > -0.05):
            return -np.inf

        if (alpha1 < 0.01 or alpha1 > 0.9999):
            return -np.inf

    if len(theta) == 13:
        if (g1 < -1 or g1 > 1):
            return -np.inf

        if (g2 < -1 or g2 > 1):
            return -np.inf

        if (g3 < -1 or g3 > 1):
            return -np.inf

        if (alpha1 < -1 or alpha1 > 1):
            return -np.inf

        if (alpha2 < -1 or alpha2 > 1):
            return -np.inf

    if (np.arccos(cinc) < np.radians(70) or np.arccos(cinc) > np.radians(80)):
        return -np.inf

    if (pa < 20 or pa > 30):
        return -np.inf

    if (dx > 10) or (dx < -10):  #The x offset
        return -np.inf

    if (dy > 10) or (dy < -10):  #The y offset
        return -np.inf

    if (lognorm < np.log(0.5) or lognorm > np.log(50000)):
        return -np.inf
    # otherwise ...

    return np.log(prior_rout)
    # return 0.0


########################################################
def lnpb(theta):
    """ sum the logs of the priors (return of the logp funciton)
        and of the likelyhood (return of the logl function)


    Args:
        theta: list of parameters of the MCMC

    Returns:
        log of priors + log of likelyhood
    """
    # from datetime import datetime
    # starttime=datetime.now()
    lp = logp(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = logl(theta)
    # print("Running time model + FM: ", datetime.now()-starttime)

    return lp + ll


########################################################
def make_noise_map_no_mask(reduced_data,
                           aligned_center=[140., 140.],
                           delta_raddii=3):
    """ create a noise map from a image using concentring rings
        and measuring the standard deviation on them

    Args:
        reduced_data: [dim dim] array containing the reduced data
        aligned_center: [pixel,pixel], position of the star in the mask
        delta_raddii: pixel, widht of the small concentric rings

    Returns:
        a [dim,dim] array where each concentric rings is at a constant value
            of the standard deviation of the reduced_data
    """

    dim = reduced_data.shape[1]
    # create rho2D for the rings
    x = np.arange(dim, dtype=np.float)[None, :] - aligned_center[0]
    y = np.arange(dim, dtype=np.float)[:, None] - aligned_center[1]
    rho2d = np.sqrt(x**2 + y**2)

    noise_map = np.zeros((dim, dim))
    for i_ring in range(0,
                        int(np.floor(aligned_center[0] / delta_raddii)) - 2):
        wh_rings = np.where((rho2d >= i_ring * delta_raddii)
                            & (rho2d < (i_ring + 1) * delta_raddii))
        noise_map[wh_rings] = np.nanstd(reduced_data[wh_rings])
    return noise_map


########################################################
def initialize_mask_psf_noise(params_mcmc_yaml, quietklip=True):
    """ initialize the MCMC by preparing the useful things to measure the
    likelyhood (measure the data, the psf, the uncertainty map, the masks).

    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file

    Returns:
        a dataset a pyklip instance of Instrument.Data
    """

    # if first_time=True, all the masks, reduced data, noise map, and KL vectors
    # are recalculated. be careful, for some reason the KL vectors are slightly
    # different on different machines. if you see weird stuff in the FM models
    # (for example in plotting the results), just remake them
    first_time = params_mcmc_yaml['FIRST_TIME']

    file_prefix = params_mcmc_yaml['FILE_PREFIX']

    distance_star = params_mcmc_yaml['DISTANCE_STAR']
    pixscale_ins = params_mcmc_yaml['PIXSCALE_INS']

    owa = params_mcmc_yaml['OWA']
    move_here = params_mcmc_yaml['MOVE_HERE']
    numbasis = [params_mcmc_yaml['KLMODE_NUMBER']]
    mode = params_mcmc_yaml['MODE']

    noise_multiplication_factor = params_mcmc_yaml[
        'NOISE_MULTIPLICATION_FACTOR']

    klipdir = os.path.join(DATADIR, 'klip_fm_files') + os.path.sep

    #The PSF centers
    aligned_center = params_mcmc_yaml['ALIGNED_CENTER']

    ### This is the only part of the code different for GPI IFS anf SPHERE
    # For SPHERE We load the PSF and the parangs, crop the data

    # For GPI, we load the raw data, emasure hte PSF from sat spots and
    # collaspe the data

    if params_mcmc_yaml['BAND_DIR'] == 'SPHERE_Hdata':
        # only for SPHERE. This part is very dependent on the format of the data since there are
        # several pipelines to reduce the data, which have there own way of preparing the
        # frames from the raw data. A. Vigan has made a pyklip mode to treat the data
        # created from his pipeline, but I've never used it.

        if first_time:
            psf_init = fits.getdata(os.path.join(DATADIR,
                                                 "psf_sphere_h2.fits"))
            size_init = psf_init.shape[1]
            size_small = 31
            small_psf = psf_init[size_init // 2 -
                                 size_small // 2:size_init // 2 +
                                 size_small // 2 + 1, size_init // 2 -
                                 size_small // 2:size_init // 2 +
                                 size_small // 2 + 1]

            small_psf = small_psf / np.max(small_psf)
            small_psf[np.where(small_psf < 0.005)] = 0.
            # crop, normalize and clean the SPHERE PSF. We call is SatSpotPSF
            # so that it's transparent in the code, but it's obviously not a sat spot PSF,
            # because there are no sat spots in GPI data.

            fits.writeto(os.path.join(klipdir,
                                      file_prefix + '_SatSpotPSF.fits'),
                         small_psf,
                         overwrite='True')

            # load the raw data
            datacube_sphere_init = fits.getdata(
                os.path.join(DATADIR, "cube_H2.fits"))
            parangs = fits.getdata(os.path.join(DATADIR, "parang.fits"))
            parangs = parangs - 135.99 + 90  ## true north

            datacube_sphere_init = np.delete(datacube_sphere_init, (72, 81),
                                             0)  ## 2 slices are bad
            parangs = np.delete(parangs, (72, 81), 0)  ## 2 slices are bad

            olddim = datacube_sphere_init.shape[1]

            # we resize the SPHERE data to the same size as GPI (281)
            # to avoid a problem of centering.
            # Main thing to be carefull: the data and the PSFs for all instrument used
            # must be centered on the same way (on a pixel or between pixel) and must have the
            # same parity. We chose here, as usual in pyklip. odd number of pixel dim = 281 and
            # centering of the PSF and data on the pixel dim//2 = 140
            # Also be careful in the cropping, part of the code assumes square data (dimx = dimy)
            newdim = 281
            datacube_sphere_newdim = np.zeros(
                (datacube_sphere_init.shape[0], newdim, newdim))

            for i in range(datacube_sphere_init.shape[0]):
                datacube_sphere_newdim[i, :, :] = datacube_sphere_init[
                    i, olddim // 2 - newdim // 2:olddim // 2 + newdim // 2 + 1,
                    olddim // 2 - newdim // 2:olddim // 2 + newdim // 2 + 1]

            # we flip the dataset (and inverse the parangs) to obtain
            # the good PA after pyklip reduction. Once again, dependent on how raw data were produced
            parangs = -parangs
            nb_images = datacube_sphere_newdim.shape[0]  # pylint: disable=E1136
            for i in range(nb_images):
                datacube_sphere_newdim[i] = np.flip(datacube_sphere_newdim[i],
                                                    axis=0)

            datacube_sphere = datacube_sphere_newdim

            fits.writeto(os.path.join(DATADIR,
                                      file_prefix + '_true_parangs.fits'),
                         parangs,
                         overwrite='True')

            fits.writeto(os.path.join(DATADIR,
                                      file_prefix + '_true_dataset.fits'),
                         datacube_sphere,
                         overwrite='True')

        datacube_sphere = fits.getdata(
            os.path.join(DATADIR, file_prefix + '_true_dataset.fits'))
        parangs_sphere = fits.getdata(
            os.path.join(DATADIR, file_prefix + '_true_parangs.fits'))

        size_datacube = datacube_sphere.shape
        centers_sphere = np.zeros((size_datacube[0], 2)) + aligned_center
        dataset = Instrument.GenericData(datacube_sphere,
                                         centers_sphere,
                                         parangs=parangs_sphere,
                                         wvs=None)

    else:
        #only for GPI. Data reduction is simpler for GPI but PSF 
        # measurement form satspot is more complicated.

        if first_time:
            print("\n Create a PSF from the sat spots")
            filelist4psf = sorted(glob.glob(os.path.join(DATADIR, "*.fits")))

            dataset4psf = GPI.GPIData(filelist4psf, quiet=True)

            # identify angles where the
            # disk intersect the satspots
            excluded_files = gpidiskpsf.check_satspots_disk_intersection(
                dataset4psf, params_mcmc_yaml, quiet=True)

            # exclude those angles for the PSF measurement
            for excluded_filesi in excluded_files:
                if excluded_filesi in filelist4psf:
                    filelist4psf.remove(excluded_filesi)

            # create the data this time wihtout the bad files
            dataset4psf = GPI.GPIData(filelist4psf, quiet=True)

            # Find the IFS slices for which the satspots are too faint
            # if SNR time_mean(sat spot) <3 they are removed
            # Mostly for K2 and sometime K1
            excluded_slices = gpidiskpsf.check_satspots_snr(dataset4psf,
                                                            params_mcmc_yaml,
                                                            quiet=True)

            params_mcmc_yaml['EXCLUDED_SLICES'] = excluded_slices

            # extract the data this time wihtout the bad files nor slices
            dataset4psf = GPI.GPIData(filelist4psf,
                                      quiet=True,
                                      skipslices=excluded_slices)

            # finally measure the good psf
            instrument_psf = gpidiskpsf.make_collapsed_psf(dataset4psf,
                                                           params_mcmc_yaml,
                                                           boxrad=15)

            # save the excluded_slices in the psf header
            hdr_psf = fits.Header()
            hdr_psf['N_BADSLI'] = len(excluded_slices)
            for badslice_i, excluded_slices_num in enumerate(excluded_slices):
                hdr_psf['BADSLI' +
                        str(badslice_i).zfill(2)] = excluded_slices_num

            #save the psf
            fits.writeto(os.path.join(klipdir,
                                      file_prefix + '_SatSpotPSF.fits'),
                         instrument_psf,
                         header=hdr_psf,
                         overwrite=True)

        filelist = sorted(glob.glob(os.path.join(DATADIR, "*.fits")))

        # in the general case we can choose to
        # keep the files where the disk intersect the disk.
        # We can removed those if rm_file_disk_cross_satspots=True
        rm_file_disk_cross_satspots = params_mcmc_yaml[
            'RM_FILE_DISK_CROSS_SATSPOTS']
        if rm_file_disk_cross_satspots:
            dataset_for_exclusion = GPI.GPIData(filelist, quiet=True)
            excluded_files = gpidiskpsf.check_satspots_disk_intersection(
                dataset_for_exclusion, params_mcmc_yaml, quiet=True)
            for excluded_filesi in excluded_files:
                if excluded_filesi in filelist:
                    filelist.remove(excluded_filesi)

        # load the bad slices in the psf header
        hdr_psf = fits.getheader(
            os.path.join(klipdir, file_prefix + '_SatSpotPSF.fits'))

        # in IFS mode, we always exclude the slices with too much noise. We
        # chose the criteria as "SNR(mean of sat spot)< 3""
        excluded_slices = []
        if hdr_psf['N_BADSLI'] > 0:
            for badslice_i in range(hdr_psf['N_BADSLI']):
                excluded_slices.append(hdr_psf['BADSLI' +
                                               str(badslice_i).zfill(2)])

        # load the raw data without the bad slices
        dataset = GPI.GPIData(filelist, quiet=True, skipslices=excluded_slices)

        #collapse the data spectrally
        dataset.spectral_collapse(align_frames=True,
                                  aligned_center=aligned_center)

    #define the outer working angle
    dataset.OWA = owa

    #assuming square data
    dimension = dataset.input.shape[2]

    #create the masks
    print("\n Create the binary masks to define model zone and chisquare zone")
    if first_time:
        #create the mask where the non convoluted disk is going to be generated.
        # To gain time, it is tightely adjusted to the expected models BEFORE 
        # convolution. Inded, the models are generated pixel by pixels. 0.1 s 
        # gained on every model is a day of calculation gain on one million model, 
        # so adjust your mask tightly to your model. Carefull mask paramters are 
        # hardcoded here

        mask_disk_zeros = gpidiskpsf.make_disk_mask(
            dimension,
            params_mcmc_yaml['pa_init'],
            params_mcmc_yaml['inc_init'],
            convert.au_to_pix(55, pixscale_ins, distance_star),
            convert.au_to_pix(105, pixscale_ins, distance_star),
            aligned_center=aligned_center)
        mask2generatedisk = 1 - mask_disk_zeros
        fits.writeto(os.path.join(klipdir,
                                  file_prefix + '_mask2generatedisk.fits'),
                     mask2generatedisk,
                     overwrite='True')

        # we create a second mask for the minimization a little bit larger
        # (because model expect to grow with the PSF convolution and the FM)
        # and we can also exclude the center region where there are too much speckles
        mask_disk_zeros = gpidiskpsf.make_disk_mask(
            dimension,
            params_mcmc_yaml['pa_init'],
            params_mcmc_yaml['inc_init'],
            convert.au_to_pix(40, pixscale_ins, distance_star),
            convert.au_to_pix(130, pixscale_ins, distance_star),
            aligned_center=aligned_center)

        mask_speckle_region = np.ones((dimension, dimension))
        # a few lines to create a circular central mask to hide regions with a lot of speckles
        # Currently not using it but it's there
        # x = np.arange(dimension, dtype=np.float)[None,:] - aligned_center[0]
        # y = np.arange(dimension, dtype=np.float)[:,None] - aligned_center[1]
        # rho2d = np.sqrt(x**2 + y**2)
        # mask_speckle_region[np.where(rho2d < 21)] = 0.
        mask2minimize = mask_speckle_region * (1 - mask_disk_zeros)

        fits.writeto(os.path.join(klipdir,
                                  file_prefix + '_mask2minimize.fits'),
                     mask2minimize,
                     overwrite='True')

    mask2generatedisk = fits.getdata(
        os.path.join(klipdir, file_prefix + '_mask2generatedisk.fits'))
    mask2minimize = fits.getdata(
        os.path.join(klipdir, file_prefix + '_mask2minimize.fits'))

    if params_mcmc_yaml['MODE'] == 'RDI':
        psflib = initialize_rdi(dataset, params_mcmc_yaml)
    else:
        psflib = None

    if first_time:
        #measure the uncertainty map using the counter rotation trick
        # described in Sec4 of Gerard&Marois SPIE 2016 and probabaly elsewhere
        print("\n Create the uncertainty map")
        dataset.PAs = -dataset.PAs

        # Disable print for pyklip
        if quietklip:
            sys.stdout = open(os.devnull, 'w')

        parallelized.klip_dataset(dataset,
                                  numbasis=numbasis,
                                  maxnumbasis=120,
                                  annuli=1,
                                  subsections=1,
                                  mode=mode,
                                  outputdir=klipdir,
                                  fileprefix=file_prefix +
                                  '_couter_rotate_trick',
                                  aligned_center=aligned_center,
                                  highpass=False,
                                  minrot=move_here,
                                  calibrate_flux=False,
                                  psf_library=psflib)
        sys.stdout = sys.__stdout__

        reduced_data_nodisk = fits.getdata(
            os.path.join(klipdir, file_prefix +
                         '_couter_rotate_trick-KLmodes-all.fits'))[0]
        noise = make_noise_map_no_mask(reduced_data_nodisk,
                                       aligned_center=aligned_center,
                                       delta_raddii=3)
        noise[np.where(
            noise == 0)] = np.nan  #we are going to divide by this noise

        #### We know our noise is too small so we multiply by a given factor
        noise = noise_multiplication_factor * noise

        fits.writeto(os.path.join(klipdir, file_prefix + '_noisemap.fits'),
                     noise,
                     overwrite='True')

        dataset.PAs = -dataset.PAs
        os.remove(
            os.path.join(klipdir, file_prefix +
                         '_couter_rotate_trick-KLmodes-all.fits'))

    return dataset, psflib


########################################################
def initialize_rdi(dataset, params_mcmc_yaml):
    """ initialize the rdi librairy. This should maybe not be done in this 
        code since it can be extremely time consuming. Feel free to run this 
        routine elsewhere. Currently very GPI oriented.

    Args:
        dataset: a pyklip instance of Instrument.Data containing the data
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file

    Returns:
        a  psflib object
    """

    print("\n Initialize RDI")
    first_time = params_mcmc_yaml['FIRST_TIME']
    aligned_center = params_mcmc_yaml['ALIGNED_CENTER']
    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    rdidir = os.path.join(DATADIR, params_mcmc_yaml['RDI_DIR'])
    rdi_matrix_dir = os.path.join(rdidir, 'rdi_matrix')
    distutils.dir_util.mkpath(rdi_matrix_dir)

    if first_time:

        # load the bad slices in the psf header (IFS slices where satspots SNR < 3).
        # This is only for GPI. Normally this should not happen because RDI is in 
        # H band and H band data have very little thermal noise.

        hdr_psf = fits.getheader(
            os.path.join(klipdir, file_prefix + '_SatSpotPSF.fits'))

        # in IFS mode, we always exclude the slices with too much noise. We
        # chose the criteria as "SNR(mean of sat spot)< 3""
        excluded_slices = []
        if hdr_psf['N_BADSLI'] > 0:
            for badslice_i in range(hdr_psf['N_BADSLI']):
                excluded_slices.append(hdr_psf['BADSLI' +
                                               str(badslice_i).zfill(2)])

        # be carefull the librairy files must includes the data files !
        lib_files = sorted(glob.glob(os.path.join(rdidir, "*.fits")))

        datasetlib = GPI.GPIData(lib_files,
                                 quiet=True,
                                 skipslices=excluded_slices)

        #collapse the data spectrally
        datasetlib.spectral_collapse(align_frames=True,
                                     aligned_center=aligned_center)

        #we save the psf librairy aligned and collapsed, this is long to do

        # save the filenames in the header
        hdr_psf_lib = fits.Header()

        hdr_psf_lib['N_PSFLIB'] = len(datasetlib.filenames)
        for i, filename in enumerate(datasetlib.filenames):
            hdr_psf_lib['PSF' + str(i).zfill(4)] = filename

        #save the psf librairy aligned and collapsed
        fits.writeto(os.path.join(rdi_matrix_dir,
                                  'PSFlib_aligned_collasped.fits'),
                     datasetlib.input,
                     header=hdr_psf_lib,
                     overwrite=True)

        # make the PSF library
        # we need to compute the correlation matrix of all images vs each 
        # other since we haven't computed it before
        psflib = rdi.PSFLibrary(datasetlib.input,
                                aligned_center,
                                datasetlib.filenames,
                                compute_correlation=True)

        # save the correlation matrix to disk so that we also don't need to 
        # recomptue this ever again. In the future we can just pass in the 
        # correlation matrix into the PSFLibrary object rather than having it 
        # compute it
        psflib.save_correlation(os.path.join(rdi_matrix_dir,
                                             "corr_matrix.fits"),
                                overwrite=True)

    # load the PSF librairy aligned and collapse
    PSFlib_input = fits.getdata(
        os.path.join(rdi_matrix_dir, 'PSFlib_aligned_collasped.fits'))

    # Load filenames from the header
    hdr_psf_lib = fits.getheader(
        os.path.join(rdi_matrix_dir, 'PSFlib_aligned_collasped.fits'))

    PSFlib_filenames = []
    if hdr_psf_lib['N_PSFLIB'] > 0:
        for i in range(hdr_psf_lib['N_PSFLIB']):
            PSFlib_filenames.append(hdr_psf_lib['PSF' + str(i).zfill(4)])

    # load the correlation matrix
    corr_matrix = fits.getdata(os.path.join(rdi_matrix_dir,
                                            "corr_matrix.fits"))

    # make the PSF library again, this time we have the correlation matrix
    psflib = rdi.PSFLibrary(PSFlib_input,
                            aligned_center,
                            PSFlib_filenames,
                            correlation_matrix=corr_matrix)

    psflib.prepare_library(dataset)

    return psflib


########################################################
def initialize_diskfm(dataset, params_mcmc_yaml, psflib=None, quietklip=True):
    """ initialize the MCMC by preparing the diskFM object

    Args:
        dataset: a pyklip instance of Instrument.Data
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file

    Returns:
        a  diskFM object
    """
    print("\n Initialize diskFM")
    first_time = params_mcmc_yaml['FIRST_TIME']
    aligned_center = params_mcmc_yaml['ALIGNED_CENTER']
    numbasis = [params_mcmc_yaml['KLMODE_NUMBER']]
    move_here = params_mcmc_yaml['MOVE_HERE']
    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    n_dim_mcmc = params_mcmc_yaml['N_DIM_MCMC']
    mode = params_mcmc_yaml['MODE']
    klipdir = os.path.join(DATADIR, 'klip_fm_files')

    if first_time:
        # create a first model to check the begining parameter and initialize the FM.
        # We will clear all useless variables befire starting the MCMC
        # Be careful that this model is close to what you think is the minimum
        # because the FM is not completely linear so you have to measure the FM on
        # something already close to the best one

        theta_init = from_param_to_theta_init(params_mcmc_yaml)

        #generate the model
        if n_dim_mcmc == 11:
            model_here = call_gen_disk_2g(theta_init)
        if n_dim_mcmc == 13:
            model_here = call_gen_disk_3g(theta_init)

        fits.writeto(os.path.join(klipdir, file_prefix + '_FirstModel.fits'),
                     model_here,
                     overwrite='True')

        model_here_convolved = convolve(model_here, PSF, boundary='wrap')
        fits.writeto(os.path.join(klipdir,
                                  file_prefix + '_FirstModel_Conv.fits'),
                     model_here_convolved,
                     overwrite='True')

    model_here_convolved = fits.getdata(
        os.path.join(klipdir, file_prefix + '_FirstModel_Conv.fits'))

    if first_time:
        # Disable print for pyklip
        if quietklip:
            sys.stdout = open(os.devnull, 'w')

        # initialize the DiskFM object
        diskobj = DiskFM(dataset.input.shape,
                         numbasis,
                         dataset,
                         model_here_convolved,
                         basis_filename=os.path.join(
                             klipdir, file_prefix + '_klbasis.h5'),
                         save_basis=True,
                         aligned_center=aligned_center,
                         numthreads=1)
        # measure the KL basis and save it
        fm.klip_dataset(dataset,
                        diskobj,
                        numbasis=numbasis,
                        maxnumbasis=120,
                        annuli=1,
                        subsections=1,
                        mode=mode,
                        outputdir=klipdir,
                        fileprefix=file_prefix,
                        aligned_center=aligned_center,
                        mute_progression=True,
                        highpass=False,
                        minrot=move_here,
                        calibrate_flux=False,
                        numthreads=1,
                        psf_library=psflib)
        sys.stdout = sys.__stdout__

    # load the the KL basis and define the diskFM object
    diskobj = DiskFM(dataset.input.shape,
                     numbasis,
                     dataset,
                     model_here_convolved,
                     basis_filename=os.path.join(klipdir,
                                                 file_prefix + '_klbasis.h5'),
                     load_from_basis=True,
                     numthreads=1)

    # test the diskFM object
    diskobj.update_disk(model_here_convolved)
    ### we take only the first KL modemode

    if first_time:
        modelfm_here = diskobj.fm_parallelized()[0]
        fits.writeto(os.path.join(klipdir,
                                  file_prefix + '_FirstModel_FM.fits'),
                     modelfm_here,
                     overwrite='True')

    return diskobj


########################################################
def initialize_walkers_backend(params_mcmc_yaml):
    """ initialize the MCMC by preparing the initial position of the
        walkers and the backend file

    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file

    Returns:
        if new_backend=True then [intial position of the walkers, a clean BACKEND]
        if new_backend=False then [None, the loaded BACKEND]
    """

    # if new_backend=False, reset the backend, if not restart the chains.
    # Be careful if you change the parameters or walkers #, you have to put 
    # new_backend=True
    new_backend = params_mcmc_yaml['NEW_BACKEND']

    nwalkers = params_mcmc_yaml['NWALKERS']
    n_dim_mcmc = params_mcmc_yaml['N_DIM_MCMC']

    file_prefix = params_mcmc_yaml['FILE_PREFIX']

    mcmcresultdir = os.path.join(DATADIR, 'results_MCMC')

    theta_init = from_param_to_theta_init(params_mcmc_yaml)

    # Set up the backend h5
    # Don't forget to clear it in case the file already exists
    filename_backend = os.path.join(mcmcresultdir,
                                    file_prefix + "_backend_file_mcmc.h5")
    backend_ini = backends.HDFBackend(filename_backend)

    #############################################################
    # Initialize the walkers. The best technique seems to be
    # to start in a small ball around the a priori preferred position.
    # Dont worry, the walkers quickly branch out and explore the
    # rest of the space.
    if new_backend:
        init_ball0 = np.random.uniform(theta_init[0] * 0.999,
                                       theta_init[0] * 1.001,
                                       size=(nwalkers))  # r1 log[AU]
        init_ball1 = np.random.uniform(theta_init[1] * 0.999,
                                       theta_init[1] * 1.001,
                                       size=(nwalkers))  # r2 log[AU]
        init_ball2 = np.random.uniform(theta_init[2] * 0.99,
                                       theta_init[2] * 1.01,
                                       size=(nwalkers))  #beta
        init_ball3 = np.random.uniform(theta_init[3] * 0.99,
                                       theta_init[3] * 1.01,
                                       size=(nwalkers))  #g1
        init_ball4 = np.random.uniform(theta_init[4] * 0.99,
                                       theta_init[4] * 1.01,
                                       size=(nwalkers))  #g2

        if n_dim_mcmc == 11:
            init_ball5 = np.random.uniform(theta_init[5] * 0.99,
                                           theta_init[5] * 1.01,
                                           size=(nwalkers))  #alpha1
            init_ball6 = np.random.uniform(theta_init[6] * 0.99,
                                           theta_init[6] * 1.01,
                                           size=(nwalkers))  #cinc
            init_ball7 = np.random.uniform(theta_init[7] * 0.99,
                                           theta_init[7] * 1.01,
                                           size=(nwalkers))  #pa [degrees]
            init_ball8 = np.random.uniform(
                theta_init[8] * 0.99, theta_init[8] * 1.01,
                size=(nwalkers))  # offset in minor axis
            init_ball9 = np.random.uniform(
                theta_init[9] * 0.99, theta_init[9] * 1.01,
                size=(nwalkers))  # offset in major axis
            init_ball10 = np.random.uniform(
                theta_init[10] * 0.99, theta_init[10] * 1.01,
                size=(nwalkers))  #log normalizing factor
            p0 = np.dstack((init_ball0, init_ball1, init_ball2, init_ball3,
                            init_ball4, init_ball5, init_ball6, init_ball7,
                            init_ball8, init_ball9, init_ball10))

        if n_dim_mcmc == 13:
            init_ball5 = np.random.uniform(theta_init[5] * 0.99,
                                           theta_init[5] * 1.01,
                                           size=(nwalkers))  #g3
            init_ball6 = np.random.uniform(theta_init[6] * 0.99,
                                           theta_init[6] * 1.01,
                                           size=(nwalkers))  #alpha1
            init_ball7 = np.random.uniform(theta_init[7] * 0.99,
                                           theta_init[7] * 1.01,
                                           size=(nwalkers))  #alpha2

            init_ball8 = np.random.uniform(theta_init[8] * 0.99,
                                           theta_init[8] * 1.01,
                                           size=(nwalkers))  #cinc
            init_ball9 = np.random.uniform(theta_init[9] * 0.99,
                                           theta_init[9] * 1.01,
                                           size=(nwalkers))  #pa [degrees]
            init_ball10 = np.random.uniform(
                theta_init[10] * 0.99, theta_init[10] * 1.01,
                size=(nwalkers))  # offset in minor axis
            init_ball11 = np.random.uniform(
                theta_init[11] * 0.99, theta_init[11] * 1.01,
                size=(nwalkers))  # offset in major axis
            init_ball12 = np.random.uniform(
                theta_init[12] * 0.99, theta_init[12] * 1.01,
                size=(nwalkers))  #log normalizing factor
            p0 = np.dstack(
                (init_ball0, init_ball1, init_ball2, init_ball3, init_ball4,
                 init_ball5, init_ball6, init_ball7, init_ball8, init_ball9,
                 init_ball10, init_ball11, init_ball12))

        backend_ini.reset(nwalkers, n_dim_mcmc)
        return p0[0], backend_ini

    return None, backend_ini


########################################################
def from_param_to_theta_init(params_mcmc_yaml):
    """ create a initial set of MCMCparameter from the initial parmeters
        store in the init yaml file
    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file

    Returns:
        initial set of MCMC parameter
    """

    n_dim_mcmc = params_mcmc_yaml['N_DIM_MCMC']  #Number of interation

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

    if n_dim_mcmc == 11:
        theta_init = (np.log(r1_init), np.log(r2_init), beta_init, g1_init,
                      g2_init, alpha1_init, np.cos(np.radians(inc_init)),
                      pa_init, dx_init, dy_init, np.log(N_init))
    if n_dim_mcmc == 13:
        g3_init = params_mcmc_yaml['g3_init']
        alpha2_init = params_mcmc_yaml['alpha2_init']
        theta_init = (np.log(r1_init), np.log(r2_init), beta_init, g1_init,
                      g2_init, g3_init, alpha1_init, alpha2_init,
                      np.cos(np.radians(inc_init)), pa_init, dx_init, dy_init,
                      np.log(N_init))

    return theta_init


if __name__ == '__main__':

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.simplefilter('ignore', FITSFixedWarning)
    # warnings.filterwarnings("ignore", category=UserWarning)
    # warnings.simplefilter('ignore', category=AstropyWarning)

    if len(sys.argv) == 1:
        str_yalm = 'GPI_Hband_MCMC_ADI.yaml'
    else:
        str_yalm = sys.argv[1]

    # open the parameter file
    yaml_path_file = os.path.join(os.getcwd(), 'initialization_files',
                                  str_yalm)
    with open(yaml_path_file, 'r') as yaml_file:
        params_mcmc_yaml = yaml.load(yaml_file)

    DATADIR = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'])
    FILE_PREFIX = params_mcmc_yaml['FILE_PREFIX']

    klipdir = os.path.join(DATADIR, 'klip_fm_files')
    distutils.dir_util.mkpath(klipdir)

    mcmcresultdir = os.path.join(DATADIR, 'results_MCMC')
    distutils.dir_util.mkpath(mcmcresultdir)

    if params_mcmc_yaml['FIRST_TIME'] and mpi:
        raise ValueError("""Because the way the code is set up right now, 
                saving .fits seems complicated to do in MPI mode so we cannot 
                initialiaze in mpi mode, please initialiaze using 'FIRST_TIME=True' 
                only in sequential (to measure and save all the necessary .fits files 
                (PSF, masks, etc) and then run MPI with 'FIRST_TIME=False' """)

    # load the Parameters necessary to launch the MCMC
    NWALKERS = params_mcmc_yaml['NWALKERS']  #Number of walkers
    N_ITER_MCMC = params_mcmc_yaml['N_ITER_MCMC']  #Number of interation
    N_DIM_MCMC = params_mcmc_yaml['N_DIM_MCMC']  #Number of MCMC dimension

    # load DISTANCE_STAR & PIXSCALE_INS & DIMENSION and make them global
    DISTANCE_STAR = params_mcmc_yaml['DISTANCE_STAR']
    PIXSCALE_INS = params_mcmc_yaml['PIXSCALE_INS']

    # initialize the things necessary to measure the model (PSF, masks, etc)
    dataset, psflib = initialize_mask_psf_noise(params_mcmc_yaml,
                                                quietklip=True)

    # measure the size of images DIMENSION and make it global
    DIMENSION = dataset.input.shape[1]

    # load PSF and make it global
    PSF = fits.getdata(os.path.join(klipdir, FILE_PREFIX + '_SatSpotPSF.fits'))

    # load wheremask2generatedisk and make it global
    WHEREMASK2GENERATEDISK = (fits.getdata(
        os.path.join(klipdir, FILE_PREFIX + '_mask2generatedisk.fits')) == 0)

    # load noise and make it global
    NOISE = fits.getdata(os.path.join(klipdir, FILE_PREFIX + '_noisemap.fits'))

    # initialize_diskfm and make diskobj global
    DISKOBJ = initialize_diskfm(dataset,
                                params_mcmc_yaml,
                                psflib=psflib,
                                quietklip=True)

    # load reduced_dataand make it a global variable
    REDUCED_DATA = fits.getdata(
        os.path.join(klipdir, FILE_PREFIX + '-klipped-KLmodes-all.fits'))[
            0]  ### we take only the first KL mode

    # we multiply the reduced_data by the mask2minimize to avoid having
    # to pass it as a global variable
    mask2minimize = fits.getdata(
        os.path.join(klipdir, FILE_PREFIX + '_mask2minimize.fits'))
    mask2minimize[np.where(mask2minimize == 0.)] = np.nan
    REDUCED_DATA *= mask2minimize
    del mask2minimize, dataset

    # Make a final test by printing the likelyhood of the iniatial model
    lnpb_model = lnpb(from_param_to_theta_init(params_mcmc_yaml))
    if not np.isfinite(lnpb_model):
        raise ValueError(""" Test Likelyhood on initial parameter set is {0}. 
            Do not launch MCMC, your initial guess is probably out of the prior 
            range for one of the parameter""".format(lnpb_model))

    startTime = datetime.now()
    if mpi:
        mpistr = "\n In MPI mode"
    else:
        mpistr = "\n In sequential mode"

    print(mpistr + ", initialize walkers and start the MCMC...")
    
    with MultiPool() as pool:

        if mpi:
            if not pool.is_master():
                pool.wait()
                sys.exit(0)

        # initialize the walkers if necessary. initialize/load the backend
        # make them global
        init_walkers, BACKEND = initialize_walkers_backend(params_mcmc_yaml)

        #Let's start the MCMC
        # Set up the Sampler. I purposefully passed the variables (KL modes,
        # reduced data, masks) in global variables to save time as advised in
        # https://emcee.readthedocs.io/en/latest/tutorials/parallel/
        # mode mpi or not

        sampler = EnsembleSampler(NWALKERS,
                                  N_DIM_MCMC,
                                  lnpb,
                                  pool=pool,
                                  backend=BACKEND)

        sampler.run_mcmc(init_walkers, N_ITER_MCMC, progress=progress)
    print(mpistr +
          ", time {0} iterations with {1} walkers and {2} cpus: {3}".format(
              N_ITER_MCMC, NWALKERS, cpu_count(),
              datetime.now() - startTime))

