# pylint: disable=C0103
"""
MCMC code for fitting a disk 
author: Johan Mazoyer
"""

import os
import copy
import argparse
# careful on Python 3.8 mac multiprocessing switched to spawn so the global varialbe do not work

basedir = os.environ["EXCHANGE_PATH"]  # the base directory where is
# your data (using OS environnement variable allow to use same code on
# different computer without changing this).

# default_parameter_file = 'FakeHr4796bright_MCMC_ADI.yaml'  # name of the parameter file
# default_parameter_file = 'GPI_Hband_MCMC_ADI.yaml'  # name of the parameter file
default_parameter_file = 'SPHERE_Hband_MCMC_ADI.yaml'  # name of the parameter file
# you can also call it with the python function argument -p

MPI = False  ## by default the MCMC is not mpi. you can change it
## in the the python function argument --mpi

parser = argparse.ArgumentParser(description='run diskFM MCMC')
parser.add_argument('-p',
                    '--param_file',
                    required=False,
                    help='parameter file name')
parser.add_argument("--mpi", help="run in mpi mode", action="store_true")
args = parser.parse_args()

import sys
import glob

import distutils.dir_util
import warnings

if args.mpi:  # MPI or not for parallelization.
    from schwimmbad import MPIPool as MultiPool
else:
    import multiprocessing as mp
    MultiPool = mp.get_context('fork').Pool
    # from multiprocessing import Pool as MultiPool

from multiprocessing import cpu_count

from datetime import datetime

import math as mt
import numpy as np
import scipy.ndimage as snd

import astropy.io.fits as fits
from astropy.convolution import convolve
from astropy.wcs import FITSFixedWarning

import yaml

from emcee import EnsembleSampler
from emcee import backends

from numba.core.errors import NumbaWarning

import pyklip.instruments.GPI as GPI
import pyklip.instruments.SPHERE as SPHERE
import pyklip.instruments.Instrument as Instrument

import pyklip.parallelized as parallelized
from pyklip.fmlib.diskfm import DiskFM, _load_dict_from_hdf5
import pyklip.fm as fm
import pyklip.rdi as rdi

from anadisk_model.anadisk_sum_mask import phase_function_spline, generate_disk

from disk_models import hg_1g, hg_2g, hg_3g

import make_gpi_psf_for_disks as gpidiskpsf
import astro_unit_conversion as convert

# recommended by emcee https://emcee.readthedocs.io/en/stable/tutorials/parallel/
# and by PyKLIPto avoid that NumPy automatically parallelizes some operations,
# which kill the speed
os.environ["OMP_NUM_THREADS"] = "1"



def sigma_filter(image, box_width, n_sigma=3, ignore_edges=False, monitor=False):

    # NAME:
    #	SIGMA_FILTER
    # PURPOSE:
    #	Replace pixels more than a specified pixels deviant from its neighbors
    # EXPLANATION:
    #	Computes the mean and standard deviation of pixels in a box centered at
    #	each pixel of the image, but excluding the center pixel. If the center
    #	pixel value exceeds some # of standard deviations from the mean, it is
    #	replaced by the mean in box. Note option to process pixels on the edges.
    # CALLING SEQUENCE:
    #	Result = sigma_filter( image, box_width, n_sigma=(#), /ALL,/MON )
    # INPUTS:
    #	image = 2-D image (matrix)
    #	box_width = width of square filter box, in # pixels (default = 3)
    #	n_sigma = # standard deviations to define outliers, floating point,
    #			recommend > 2, default = 3. For gaussian statistics:
    #			n_sigma = 1 smooths 35% of pixels, 2 = 5%, 3 = 1%.
    #   ignore_edges: if False, we also apply the sigma filter to the edges. 
    #               If true, they're left untouched.
    #   monitor: prints information about % pixels replaced.
    #   
    # CALLS:
    #	function filter_image( )
    # PROCEDURE:
    #	Compute mean over moving box-cars using smooth, subtract center values,
    #	compute variance using smooth on deviations from mean,
    #	check where pixel deviation from mean is within variance of box,
    #	replace those pixels in smoothed image (mean) with orignal values,
    #	return the resulting partial mean image.
    # MODIFICATION HISTORY:
    #	Written, 1991, Frank Varosi and Dan Gezari NASA/GSFC
    #	F.V.1992, added optional keywords /ITER,/MON,VAR=,DEV=,N_CHANGE=.
    #	Converted to IDL V5.0   W. Landsman   September 1997
    #   Translated to python with chat GPT by Johan
    #-

    if box_width <3:
        raise ValueError("box_width must be an odd integer > 2")
    
    if box_width % 2 == 0:
         raise ValueError("box_width must be an odd integer > 2")
    
    bw2 = box_width**2

    smooth = np.ones((box_width, box_width))
    smooth[1:-1, 1:-1] = 0

    if ignore_edges:
        mean = (snd.generic_filter(image, np.mean, footprint=smooth, mode='constant', cval=np.nan) * bw2 - image) / (bw2 - 1)
        wh_nan = np.isnan(mean)
        mean[wh_nan] = 0
    else:
        mean = (snd.generic_filter(image, np.mean, footprint=smooth, mode='mirror') * bw2 - image) / (bw2 -1)
        # mean = (generic_filter(image, np.nanmean, footprint=smooth, mode='constant', cval=np.nan) * bw2 - image) / (bw2 -1)

    imdev = (image - mean)**2
    fact = float(n_sigma**2) / (bw2 - 2)

    if ignore_edges:
        imvar = fact * (snd.generic_filter(imdev, np.mean, footprint=smooth, mode='constant', cval=np.nan) * bw2 - imdev)
        imdev[np.isnan(imvar)] = 0
        imvar[np.isnan(imvar)] = 0
    else:
        imvar = fact * (snd.generic_filter(imdev, np.nanmean, footprint=smooth, mode='mirror') * bw2 - imdev)
        # imvar = fact * (generic_filter(imdev, np.nanmean, footprint=smooth, mode='constant', cval=np.nan) * bw2 - imdev)

    # chek which pixels are ok
    wok = np.where(imdev <= imvar)
    nok = wok[0].size

    npix = image.size
    nchange = npix - nok

    if monitor:
        if ignore_edges:
            print(f"{(nchange)*100./npix:.2f}% of pixels replaced (edges ignored), n_sigma={n_sigma:.1f}")
        else:
            print(f"{nchange*100./npix:.2f}% of pixels replaced, n_sigma={n_sigma:.1f}")

    if nok == npix:
        return image
    if nok > 0:
        mean[wok] = image[wok]

    if ignore_edges:
        mean[wh_nan] = image[wh_nan]

    return mean

def from_theta_to_params(theta):
    param_disk = {}

    if (SPF_MODEL == 'spf_fix'):
        # param_disk['a_r'] = 0.01  # we can fix the aspect ratio
        param_disk['offset'] = 0.  # no vertical offset in KLIP

        param_disk['r1'] = mt.exp(theta[0])
        param_disk['r2'] = mt.exp(theta[1])
        param_disk['beta_in'] = theta[2]
        param_disk['beta_out'] = theta[3]
        param_disk['a_r'] = theta[4]
        param_disk['inc'] = np.degrees(np.arccos(theta[5]))
        param_disk['PA'] = theta[6]
        param_disk['dx'] = theta[7]
        param_disk['dy'] = theta[8]
        param_disk['Norm'] = mt.exp(theta[9])

        vector_param = [
            param_disk['r1'], param_disk['r2'], param_disk['beta_in'],
            param_disk['beta_out'], param_disk['a_r'], param_disk['inc'],
            param_disk['PA'], param_disk['dx'], param_disk['dy'],
            param_disk['Norm']
        ]

    elif (SPF_MODEL == 'hg_1g') or (SPF_MODEL == 'hg_2g') or (SPF_MODEL
                                                              == 'hg_3g'):

        param_disk['a_r'] = 0.01  # we fix the aspect ratio
        param_disk['offset'] = 0.  # no vertical offset in KLIP

        param_disk['beta_in'] = -100  # we fix the inner power law

        param_disk['r1'] = mt.exp(theta[0])
        param_disk['r2'] = mt.exp(theta[1])
        param_disk['beta_out'] = theta[2]
        param_disk['inc'] = np.degrees(np.arccos(theta[3]))
        param_disk['PA'] = theta[4]
        param_disk['dx'] = theta[5]
        param_disk['dy'] = theta[6]
        param_disk['Norm'] = mt.exp(theta[7])

        param_disk['g1'] = theta[8]

        vector_param = [
            param_disk['r1'], param_disk['r2'], param_disk['beta_out'],
            param_disk['inc'], param_disk['PA'], param_disk['dx'],
            param_disk['dy'], param_disk['Norm'], param_disk['g1']
        ]

        if (SPF_MODEL == 'hg_2g') or (SPF_MODEL == 'hg_3g'):
            param_disk['g2'] = theta[9]
            param_disk['alpha1'] = theta[10]
            vector_param = np.concatenate(
                (vector_param, (param_disk['g2'], param_disk['alpha1'])))

            if SPF_MODEL == 'hg_3g':
                param_disk['g3'] = theta[11]
                param_disk['alpha2'] = theta[12]
                vector_param = np.concatenate(
                    (vector_param, (param_disk['g3'], param_disk['alpha2'])))

    return param_disk, vector_param


#######################################################
def call_gen_disk(theta):
    """ call the disk model from a set of parameters.
        
        use SPF_MODEL, DIMENSION, PIXSCALE_INS, DISTANCE_STAR
        ALIGNED_CENTER and WHEREMASK2GENERATEDISK 
        as global variables

    Args:
        theta: list of parameters of the MCMC

    Returns:
        a 2d model
    """
    param_disk, _ = from_theta_to_params(theta)

    if (SPF_MODEL == 'spf_fix'):
        spf = F_SPF

    elif (SPF_MODEL == 'hg_1g'):

        # we fix the SPF using a HG parametrization with parameters in the init file
        n_points = 21  # odd number to ensure that scattangl=pi/2 is in the list for normalization
        scatt_angles = np.linspace(0, np.pi, n_points)

        # 1g henyey greenstein, normalized at 1 at 90 degrees
        spf_norm90 = hg_1g(np.degrees(scatt_angles), param_disk['g1'], 1)
        #measure fo the spline and param_disk
        spf = phase_function_spline(scatt_angles, spf_norm90)

    elif SPF_MODEL == 'hg_2g':
        # we fix the SPF using a HG parametrization with parameters in the init file

        # starttime = datetime.now()
        n_points = 21  # odd number to ensure that scattangl=pi/2 is in the list for normalization
        scatt_angles = np.linspace(0, np.pi, n_points)

        # 2g henyey greenstein, normalized at 1 at 90 degrees
        spf_norm90 = hg_2g(np.degrees(scatt_angles), param_disk['g1'],
                           param_disk['g2'], param_disk['alpha1'], 1)
        #measure fo the spline and param_disk
        spf = phase_function_spline(scatt_angles, spf_norm90)
        # print("spf_time: ", datetime.now() - starttime)

    elif SPF_MODEL == 'hg_3g':

        # we fix the SPF using a HG parametrization with parameters in the init file
        n_points = 21  # odd number to ensure that scattangl=pi/2 is in the list for normalization
        scatt_angles = np.linspace(0, np.pi, n_points)

        # 3g henyey greenstein, normalized at 1 at 90 degrees
        spf_norm90 = hg_3g(np.degrees(scatt_angles), param_disk['g1'],
                           param_disk['g2'], param_disk['g3'],
                           param_disk['alpha1'], param_disk['alpha2'], 1)
        #measure fo the spline and param_disk
        spf = phase_function_spline(scatt_angles, spf_norm90)

    #generate the model
    model = generate_disk(scattering_function_list=[spf],
                          R1=param_disk['r1'],
                          R2=param_disk['r2'],
                          beta_in=param_disk['beta_in'],
                          beta_out=param_disk['beta_out'],
                          aspect_ratio=param_disk['a_r'],
                          inc=param_disk['inc'],
                          pa=param_disk['PA'],
                          dx=param_disk['dx'],
                          dy=param_disk['dy'],
                          psfcenx=ALIGNED_CENTER[0],
                          psfceny=ALIGNED_CENTER[1],
                          sampling=1,
                          los_factor=2,
                          dim=DIMENSION,
                          mask=WHEREMASK2GENERATEDISK,
                          pixscale=PIXSCALE_INS,
                          distance=DISTANCE_STAR)[:, :, 0]

    # remove the nans to avoid problems when convolving
    model[model != model] = 0
    # I normalize by value of a_r to avoid degenerascies between a_r and Normalization
    model = param_disk['Norm'] * model / param_disk['a_r']

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
    model = call_gen_disk(theta)

    modelconvolved = convolve(model, PSF, boundary='wrap')

    # DISKOBJ = DiskFM(None,
    #                  None,
    #                  None,
    #                  modelconvolved,
    #                  basis_filename=os.path.join(KLIPDIR,
    #                                              FILE_PREFIX + '_klbasis.h5'),
    #                  load_from_basis=True)

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
    param_disk, _ = from_theta_to_params(theta)

    prior_rout = 1.
    # define the prior values
    if (param_disk['r1'] < 60 or param_disk['r1'] > 80):
        print('r1 out of prior')
        return -np.inf
    else:
        prior_rout = prior_rout * 1.

    # - rout = Logistic We  cut the prior at r2 = xx
    # because this parameter is very limited by the ADI
    if (param_disk['r2'] < 82 or param_disk['r2'] > 102):
        print('r2 out of prior')
        return -np.inf
    else:
        prior_rout = prior_rout / (1. + np.exp(40. * (param_disk['r2'] - 100)))
        # prior_rout = prior_rout * 1.  # or we can just use a flat prior

    if (param_disk['inc'] < 10 or param_disk['inc'] > 90):
        print('inc out of prior')
        return -np.inf
    else:
        prior_rout = prior_rout * 1.

    if (param_disk['PA'] < 10 or param_disk['PA'] > 90):
        print('PA out of prior')
        return -np.inf
    else:
        prior_rout = prior_rout * 1.

    if (param_disk['dx'] < -90) or (param_disk['dx'] > 90):  #The x offset
        print('dx out of prior')
        return -np.inf
    else:
        prior_rout = prior_rout * 1.

    if (param_disk['dy'] < -90) or (param_disk['dy'] > 90):  #The y offset
        print('dy out of prior')
        return -np.inf
    else:
        prior_rout = prior_rout * 1.

    if (param_disk['Norm'] < 0.5 or param_disk['Norm'] > 50000):
        print('Norm out of prior')
        return -np.inf
    else:
        prior_rout = prior_rout * 1.

    if (param_disk['beta_out'] < 1 or param_disk['beta_out'] > 30):
        print('beta_out out of prior')
        return -np.inf
    else:
        prior_rout = prior_rout * 1.

    if (SPF_MODEL == 'hg_1g') or (SPF_MODEL == 'hg_2g') or (SPF_MODEL
                                                            == 'hg_3g'):

        if (param_disk['g1'] < 0.001 or param_disk['g1'] > 0.9999):
            print('g1 out of prior')
            return -np.inf
        else:
            prior_rout = prior_rout * 1.

        if (SPF_MODEL == 'hg_2g') or (SPF_MODEL == 'hg_3g'):
            if (param_disk['g2'] < -0.9999 or param_disk['g2'] > -0.0001):
                print('g2 out of prior')
                return -np.inf
            else:
                prior_rout = prior_rout * 1.

            if (param_disk['alpha1'] < 0.0001
                    or param_disk['alpha1'] > 0.9999):
                print('alpha1 out of prior')
                return -np.inf
            else:
                prior_rout = prior_rout * 1.

            if SPF_MODEL == 'hg_3g':

                if (param_disk['g3'] < -1 or param_disk['g3'] > 1):
                    print('g3 out of prior')
                    return -np.inf
                else:
                    prior_rout = prior_rout * 1.

                if (param_disk['alpha2'] < -1 or param_disk['alpha2'] > 1):
                    print('alpha2 out of prior')
                    return -np.inf
                else:
                    prior_rout = prior_rout * 1.

    elif (SPF_MODEL == 'spf_fix'):
        if (param_disk['beta_in'] < -30 or param_disk['beta_in'] > -1):
            print('beta_in out of prior')
            return -np.inf
        else:
            prior_rout = prior_rout * 1.

        if (param_disk['a_r'] < 0.0001
                or param_disk['a_r'] > 0.5):  #The aspect ratio
            print('a_r out of prior')
            return -np.inf
        else:
            prior_rout = prior_rout * 1.

    # otherwise ...
    return np.log(prior_rout)


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
    starttime = datetime.now()
    lp = logp(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = logl(theta)
    # print("Running time model + FM: ", datetime.now() - starttime)

    return lp + ll


########################################################
def make_noise_map_rings(nodisk_data,
                         aligned_center=[140., 140.],
                         delta_raddii=3):
    """ create a noise map from a image using concentring rings
        and measuring the standard deviation on them

    Args:
        nodisk_data: [dim dim] data array containing speckle without disk
        aligned_center: [pixel,pixel], position of the star in the mask
        delta_raddii: pixel, widht of the small concentric rings

    Returns:
        a [dim,dim] array where each concentric rings is at a constant value
            of the standard deviation of the reduced_data
    """

    dim = nodisk_data.shape[1]
    # create rho2D for the rings
    x = np.arange(dim, dtype=np.float64)[None, :] - aligned_center[0]
    y = np.arange(dim, dtype=np.float64)[:, None] - aligned_center[1]
    rho2d = np.sqrt(x**2 + y**2)

    noise_map = np.zeros((dim, dim))
    for i_ring in range(0,
                        int(np.floor(aligned_center[0] / delta_raddii)) - 2):
        wh_rings = np.where((rho2d >= i_ring * delta_raddii)
                            & (rho2d < (i_ring + 1) * delta_raddii))
        noise_map[wh_rings] = np.nanstd(nodisk_data[wh_rings])
    return noise_map


def create_uncertainty_map(dataset, params_mcmc_yaml, psflib=None):
    """ measure the uncertainty map using the counter rotation trick
    described in Sec4 of Gerard&Marois SPIE 2016 and probabaly elsewhere

    Args:
        dataset: a pyklip instance of Instrument.Data containing the data
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file
        delta_raddii: pixel, widht of the small concentric rings
        psflib: a PSF librairy if RDI

    Returns:
        a [dim,dim] array containing only speckles and the disk has been removed

    """
    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    move_here = params_mcmc_yaml['MOVE_HERE']
    numbasis = [params_mcmc_yaml['KLMODE_NUMBER']]
    mode = params_mcmc_yaml['MODE']
    annuli = params_mcmc_yaml['ANNULI']
    aligned_center = params_mcmc_yaml['ALIGNED_CENTER']
    noise_multiplication_factor = params_mcmc_yaml[
        'NOISE_MULTIPLICATION_FACTOR']
    
    datadir = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'])
    klipdir = os.path.join(datadir, 'klip_fm_files')

    dataset.PAs = -dataset.PAs
    maxnumbasis = dataset.input.shape[0]

    parallelized.klip_dataset(dataset,
                              numbasis=numbasis,
                              maxnumbasis=maxnumbasis,
                              annuli=annuli,
                              subsections=1,
                              mode=mode,
                              outputdir=klipdir,
                              fileprefix=file_prefix + '_couter_rotate_trick',
                              aligned_center=aligned_center,
                              highpass=False,
                              minrot=move_here,
                              calibrate_flux=False,
                              psf_library=psflib)

    reduced_data_nodisk = fits.getdata(
        os.path.join(klipdir,
                     file_prefix + '_couter_rotate_trick-KLmodes-all.fits'))[0]
    noise = make_noise_map_rings(reduced_data_nodisk,
                                 aligned_center=aligned_center,
                                 delta_raddii=3)
    noise[np.where(noise == 0)] = np.nan  #we are going to divide by this noise

    #### We know our noise is too small so we multiply by a given factor
    noise = noise_multiplication_factor * noise

    dataset.PAs = -dataset.PAs
    os.remove(
        os.path.join(klipdir,
                     file_prefix + '_couter_rotate_trick-KLmodes-all.fits'))

    return noise


########################################################
def initialize_mask_psf_noise(params_mcmc_yaml, quietklip=True):
    """ initialize the MCMC by preparing the useful things to measure the
    likelyhood (measure the data, the psf, the uncertainty map, the masks).

    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file
        quietklip : if True, pyklip and DiskFM are quiet


    Returns:
        a dataset a pyklip instance of Instrument.Data
    """
    instrument = params_mcmc_yaml['INSTRUMENT']

    # if first_time=True, all the masks, reduced data, noise map, and KL vectors
    # are recalculated. be careful, for some reason the KL vectors are slightly
    # different on different machines. if you see weird stuff in the FM models
    # (for example in plotting the results), just remake them
    first_time = params_mcmc_yaml['FIRST_TIME']

    datadir = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'])
    klipdir = os.path.join(datadir, 'klip_fm_files')

    distutils.dir_util.mkpath(klipdir)

    file_prefix = params_mcmc_yaml['FILE_PREFIX']

    #The PSF centers
    aligned_center = params_mcmc_yaml['ALIGNED_CENTER']

    ### This is the only part of the code different for GPI IFS anf SPHERE
    # For SPHERE We load and crop the PSF and the parangs
    # For GPI, we load the raw data, emasure hte PSF from sat spots and
    # collaspe the data
    if first_time:
        if instrument == 'SPHERE-IRDIS':
            # only for SPHERE.

            # using the SPHERE data frame developed by Vigan in pyklip
            # data_files_str = params_mcmc_yaml['DATA_FILES_STR']
            # psf_files_str = params_mcmc_yaml['PSF_FILES_STR']
            # angles_str = params_mcmc_yaml['ANGLES_STR']
            # band_name = params_mcmc_yaml['BAND_NAME']

            # dataset = SPHERE.Irdis(data_files_str,
            #                        psf_files_str,
            #                        angles_str,
            #                        band_name,
            #                        psf_cube_size=31)
            # #collapse the data spectrally
            # dataset.spectral_collapse(align_frames=True,
            #                           aligned_center=aligned_center)

            # fits.writeto(os.path.join(klipdir, file_prefix + '_SmallPSF.fits'),
            #              dataset.psfs,
            #              overwrite='True')

            psf_init = fits.getdata(os.path.join(datadir, "psf_sphere_h2.fits"))
            size_init = psf_init.shape[1]
            size_small = 31
            small_psf = psf_init[size_init // 2 - size_small // 2:size_init // 2 +
                                size_small // 2 + 1,
                                size_init // 2 - size_small // 2:size_init // 2 +
                                size_small // 2 + 1]

            small_psf = small_psf / np.max(small_psf)
            small_psf[np.where(small_psf < 0.005)] = 0.

            fits.writeto(os.path.join(klipdir,
                                    file_prefix + '_SmallPSF.fits'),
                        small_psf,
                        overwrite='True')

            # load the raw data
            datacube_sphere_init = fits.getdata(
                os.path.join(datadir, "cube_H2.fits")
            )  
            parangs = fits.getdata(os.path.join(datadir, "parang.fits"))
            parangs = parangs - 135.99 + 90  ## true north Maire et al. 2016

            datacube_sphere_init = np.delete(datacube_sphere_init, (72, 81),
                                            0)  ## 2 slices are bad
            parangs = np.delete(parangs, (72, 81), 0)  ## 2 slices are bad

            # we keep only 1 slide out of N to run test on my laptop
            datacube_sphere_init = datacube_sphere_init[0::2]
             # we keep only 1 slide out of N to run test on my laptop
            parangs = parangs[0::2] 

            olddim = datacube_sphere_init.shape[1]

            # we resize the SPHERE data to the same size as GPI (281)
            # to avoid a problem of centering
            newdim = 241
            datacube_sphere_newdim = np.zeros(
                (datacube_sphere_init.shape[0], newdim, newdim))

            for i in range(datacube_sphere_init.shape[0]):
                # datacube_sphere_newdim[i, :, :] = sigma_filter(datacube_sphere_init[
                #     i, olddim // 2 - newdim // 2:olddim // 2 + newdim // 2 + 1,
                #     olddim // 2 - newdim // 2:olddim // 2 + newdim // 2 + 1],3,n_sigma=5, monitor=True)

                datacube_sphere_newdim[i, :, :] = datacube_sphere_init[
                    i, olddim // 2 - newdim // 2:olddim // 2 + newdim // 2 + 1,
                    olddim // 2 - newdim // 2:olddim // 2 + newdim // 2 + 1]


            # we flip the dataset (and therefore inverse the parangs) to obtain
            # the good PA after pyklip reduction
            parangs = -parangs

            for i in range(datacube_sphere_newdim.shape[0]):
                datacube_sphere_newdim[i] = np.flip(datacube_sphere_newdim[i],
                                                    axis=0)

            datacube_sphere = datacube_sphere_newdim

            fits.writeto(os.path.join(datadir, file_prefix + '_true_parangs.fits'),
                        parangs,
                        overwrite='True')

            fits.writeto(os.path.join(datadir, file_prefix + '_true_dataset.fits'),
                        datacube_sphere,
                        overwrite='True')

            datacube_sphere = fits.getdata(
                os.path.join(datadir, file_prefix + '_true_dataset.fits'))
            parangs_sphere = fits.getdata(
                os.path.join(datadir, file_prefix + '_true_parangs.fits'))

            size_datacube = datacube_sphere.shape
            centers_sphere = np.zeros((size_datacube[0], 2)) + [aligned_center[0], aligned_center[1]]
            dataset = Instrument.GenericData(datacube_sphere,
                                            centers_sphere,
                                            parangs=parangs_sphere,
                                            flipx=True,
                                            wvs=None)


        elif instrument == 'GPI':
            # only for GPI

            filelist = sorted(glob.glob(os.path.join(datadir, "*.fits")))

            #check that these are GPI files
            non_GPI_files = []
            filetype_here = None
            for filename in filelist:
                headerfile = fits.getheader(filename)
                if ('FILETYPE' in headerfile.keys()) and (
                        headerfile['FILETYPE'] == 'Stokes Cube'
                        or headerfile['FILETYPE']
                        == 'Spectral Cube'):  # This is a GPI file
                    if filetype_here is not None:
                        if filetype_here != headerfile[
                                'FILETYPE']:  # check if Spec of Pol
                            raise ValueError(
                                """ There are simultaneously Pol and Spec 
                                                type files in this folder. Code only works 
                                                with a single kind of obs""")
                    filetype_here = headerfile['FILETYPE']

                else:  # This is not a GPI file (maybe PSF file). Ignore when loading.
                    non_GPI_files.append(filename)

            for nonGPIfilename in non_GPI_files:
                filelist.remove(nonGPIfilename)

            pol_or_spec = filetype_here
            print(pol_or_spec)

            if len(filelist) == 0:
                raise ValueError("Could not find files in the dir")

            if pol_or_spec == 'Spectral Cube':  # GPI spec mode
                filelist4psf = copy.copy(filelist)
                dataset4psf = GPI.GPIData(filelist4psf, quiet=True)

                print("\n Create a PSF from the sat spots")
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
                excluded_slices = gpidiskpsf.check_satspots_snr(
                    dataset4psf, params_mcmc_yaml, quiet=True)

                # extract the data this time wihtout the bad files nor slices
                dataset4psf = GPI.GPIData(filelist4psf,
                                          quiet=True,
                                          skipslices=excluded_slices)

                # finally measure the good psf
                instrument_psf = gpidiskpsf.make_collapsed_psf(
                    dataset4psf, params_mcmc_yaml, boxrad=11)[0]
                # monocrhomatic here

                # save the excluded_slices in the psf header (SNR too low)
                hdr_psf = fits.Header()
                hdr_psf['N_BADSLI'] = len(excluded_slices)
                for badslice_i, excluded_slices_num in enumerate(
                        excluded_slices):
                    hdr_psf['BADSLI' +
                            str(badslice_i).zfill(2)] = excluded_slices_num

                # save the excluded_files in the psf header (disk on satspots)
                hdr_psf['N_BADFIL'] = len(excluded_files)
                for badfile_i, badfilestr in enumerate(excluded_files):
                    hdr_psf['BADFIL' + str(badfile_i).zfill(2)] = badfilestr

            else:  # GPI Pol mode
                instrument_psf = fits.getdata(
                    os.path.join(datadir, params_mcmc_yaml['PSF_FILES_STR']))
                hdr_psf = fits.Header()
                hdr_psf['N_BADSLI'] = 0

            #save the psf
            fits.writeto(os.path.join(klipdir, file_prefix + '_SmallPSF.fits'),
                         instrument_psf,
                         header=hdr_psf,
                         overwrite=True)

            if pol_or_spec == 'Spectral Cube':  # GPI spec mode
                # load the bad slices and bad files in the psf header
                hdr_psf = fits.getheader(
                    os.path.join(klipdir, file_prefix + '_SmallPSF.fits'))

                # We can choose to remove completely from the correction
                # the angles where the disk intersect the disk (they are exlcuded
                # from the PSF measurement by defaut).
                # We can removed those if rm_file_disk_cross_satspots=True
                if params_mcmc_yaml['RM_FILE_DISK_CROSS_SATSPOTS']:

                    excluded_files = []
                    if hdr_psf['N_BADFIL'] > 0:
                        for badfile_i in range(hdr_psf['N_BADFIL']):
                            excluded_files.append(
                                hdr_psf['BADFIL' + str(badfile_i).zfill(2)])

                    for excluded_filesi in excluded_files:
                        if excluded_filesi in filelist:
                            filelist.remove(excluded_filesi)

                # in IFS mode, we always exclude the IFS slices with too much noise. We
                # chose the criteria as "SNR(mean of sat spot)< 3""
                excluded_slices = []
                if hdr_psf['N_BADSLI'] > 0:
                    for badslice_i in range(hdr_psf['N_BADSLI']):
                        excluded_slices.append(
                            hdr_psf['BADSLI' + str(badslice_i).zfill(2)])

                # load the raw data without the bad slices
                dataset = GPI.GPIData(filelist,
                                      quiet=True,
                                      skipslices=excluded_slices)

            else:  # pol mode
                dataset = GPI.GPIData(filelist, quiet=True)

            #collapse the data spectrally and center
            dataset.spectral_collapse(align_frames=True,
                                      aligned_center=aligned_center)

        #After this, this is for both GPI and SPHERE
        #define the outer working angle
        dataset.OWA = params_mcmc_yaml['OWA']

        if dataset.input.shape[1] != dataset.input.shape[2]:
            raise ValueError(""" Data slices are not square (dimx!=dimy), 
                            please make them square""")

        #create the masks
        #create the mask where the non convoluted disk is going to be generated.
        # To gain time, it is ~tightely adjusted to the expected models BEFORE
        # convolution. Inded, the models are generated pixel by pixels. 0.1 s
        # gained on every model is a day of calculation gain on one million model,
        # so adjust your mask tightly to your model. You can change the harcoded parameter
        # here if you neet to go faster (reduced it) or it the slope beta is very slow (increase it)
        print(
            "\n Create the binary masks to define model zone and chisquare zone"
        )

        mask_disk_zeros = gpidiskpsf.make_disk_mask(
            dataset.input.shape[1],
            params_mcmc_yaml['pa_init'],
            params_mcmc_yaml['inc_init'],
            convert.au_to_pix(params_mcmc_yaml['r1_init'],
                              params_mcmc_yaml['PIXSCALE_INS'],
                              params_mcmc_yaml['DISTANCE_STAR']) -
            18 / np.cos(np.radians(params_mcmc_yaml['inc_init'])),
            convert.au_to_pix(params_mcmc_yaml['r2_init'],
                              params_mcmc_yaml['PIXSCALE_INS'],
                              params_mcmc_yaml['DISTANCE_STAR']) +
            18 / np.cos(np.radians(params_mcmc_yaml['inc_init'])),
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
            dataset.input.shape[1],
            params_mcmc_yaml['pa_init'],
            params_mcmc_yaml['inc_init'],
            convert.au_to_pix(params_mcmc_yaml['r1_init'],
                              params_mcmc_yaml['PIXSCALE_INS'],
                              params_mcmc_yaml['DISTANCE_STAR']) -
            20 / np.cos(np.radians(params_mcmc_yaml['inc_init'])),
            convert.au_to_pix(params_mcmc_yaml['r2_init'],
                              params_mcmc_yaml['PIXSCALE_INS'],
                              params_mcmc_yaml['DISTANCE_STAR']) +
            20 / np.cos(np.radians(params_mcmc_yaml['inc_init'])),
            aligned_center=aligned_center)

        mask2minimize = (1 - mask_disk_zeros)

        ### a few lines to create a circular central mask to hide center regions with a lot
        ### of speckles. Currently not using it but it's there
        # mask_speckle_region = np.ones((dataset.input.shape[1], dataset.input.shape[2]))
        # x = np.arange(dataset.input.shape[1], dtype=np.float)[None,:] - aligned_center[0]
        # y = np.arange(dataset.input.shape[2], dtype=np.float)[:,None] - aligned_center[1]
        # rho2d = np.sqrt(x**2 + y**2)
        # mask_speckle_region[np.where(rho2d < 21)] = 0.
        # mask2minimize = mask2minimize*mask_speckle_region

        fits.writeto(os.path.join(klipdir,
                                  file_prefix + '_mask2minimize.fits'),
                     mask2minimize,
                     overwrite='True')

        # RDI case, if it's not the first time, we do not even need to load the correlation
        # psflib, all needed information is already loaded in the KL modes
        if params_mcmc_yaml['MODE'] == 'RDI':
            psflib = initialize_rdi(dataset, params_mcmc_yaml)
        else:
            psflib = None

        print("\n Create the uncertainty map")

        # Disable print for pyklip
        if quietklip:
            sys.stdout = open(os.devnull, 'w')

        noise = create_uncertainty_map(dataset,
                                       params_mcmc_yaml,
                                       psflib=psflib)
        fits.writeto(os.path.join(klipdir, file_prefix + '_noisemap.fits'),
                     noise,
                     overwrite='True')

        sys.stdout = sys.__stdout__

    else:
        dataset = None
        psflib = None

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
    do_rdi_correlation = params_mcmc_yaml['DO_RDI_CORRELATION']
    aligned_center = params_mcmc_yaml['ALIGNED_CENTER']
    file_prefix = params_mcmc_yaml['FILE_PREFIX']

    datadir = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'])

    klipdir = os.path.join(datadir, 'klip_fm_files')
    rdidir = os.path.join(datadir, params_mcmc_yaml['RDI_DIR'])
    rdi_matrix_dir = os.path.join(rdidir, 'rdi_matrix')
    distutils.dir_util.mkpath(rdi_matrix_dir)

    if do_rdi_correlation:

        # load the bad slices in the psf header (IFS slices where satspots SNR < 3).
        # This is only for GPI spec mode. Normally this should not happen because RDI is in
        # H band and H band data have very little thermal noise.

        hdr_psf = fits.getheader(
            os.path.join(KLIPDIR, file_prefix + '_SmallPSF.fits'))

        # in IFS mode, we always exclude the slices with too much noise. We
        # chose the criteria as "SNR(mean of sat spot)< 3""
        excluded_slices = []
        if hdr_psf['N_BADSLI'] > 0:
            for badslice_i in range(hdr_psf['N_BADSLI']):
                excluded_slices.append(hdr_psf['BADSLI' +
                                               str(badslice_i).zfill(2)])

        # be carefull the librairy files must includes the data files !
        lib_files = sorted(glob.glob(os.path.join(rdidir, "*.fits")))

        non_GPI_files = []
        filetype_here = None
        for filename in lib_files:
            headerfile = fits.getheader(filename)
            if 'FILETYPE' in headerfile.keys():  # This is a GPI file
                if filetype_here is not None:
                    if headerfile[
                            'FILETYPE'] != 'Spectral Cube':  # check if Spec
                        raise ValueError(
                            """ There are non Spec type files in this 
                                            RDI lib folder. Code only works for spec"""
                        )
            else:  # This is not a GPI file (maybe PSF file). Ignore whem loading.
                non_GPI_files.append(filename)
        for nonGPIfilename in non_GPI_files:
            lib_files.remove(nonGPIfilename)

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
        psflib : a librairy of PSF if RDI
        quietklip : if True, pyklip and DiskFM are quiet

    Returns:
        a  diskFM object
    """
    print("\n Initialize diskFM")
    first_time = params_mcmc_yaml['FIRST_TIME']
    aligned_center = params_mcmc_yaml['ALIGNED_CENTER']
    numbasis = [params_mcmc_yaml['KLMODE_NUMBER']]
    move_here = params_mcmc_yaml['MOVE_HERE']
    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    mode = params_mcmc_yaml['MODE']
    annuli = params_mcmc_yaml['ANNULI']
    datadir = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'])
    klipdir = os.path.join(datadir, 'klip_fm_files')

    if first_time:
        # create a first model to check the begining parameter and initialize the FM.
        # We will clear all useless variables befire starting the MCMC
        # Be careful that this model is close to what you think is the minimum
        # because the FM is not completely linear so you have to measure the FM on
        # something already close to the best one

        theta_init = from_param_to_theta_init(params_mcmc_yaml)

        #generate the model
        model_here = call_gen_disk(theta_init)

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
                         aligned_center=aligned_center)
        # measure the KL basis and save it

        maxnumbasis = dataset.input.shape[0]
        fm.klip_dataset(dataset,
                        diskobj,
                        numbasis=numbasis,
                        maxnumbasis=maxnumbasis,
                        annuli=annuli,
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
    diskobj = DiskFM(None,
                     None,
                     None,
                     model_here_convolved,
                     basis_filename=os.path.join(klipdir,
                                                 file_prefix + '_klbasis.h5'),
                     load_from_basis=True)

    # manager = mp.Manager()
    # kl_basis_file = manager.dict(_load_dict_from_hdf5(os.path.join(klipdir,
    #                                               file_prefix + '_klbasis.h5')))
    # diskobj = DiskFM(None,
    #                  None,
    #                  None,
    #                  model_here_convolved,
    #                  kl_basis_file = kl_basis_file,
    #                  load_from_basis=True)
    

    # test the diskFM object
    diskobj.update_disk(model_here_convolved)

    if first_time:
        ### we take only the first KL modemode
        modelfm_here = diskobj.fm_parallelized()[0]
        fits.writeto(os.path.join(klipdir,
                                  file_prefix + '_FirstModel_FM.fits'),
                     modelfm_here,
                     overwrite='True')

    return diskobj


########################################################
def initialize_walkers_backend(nwalkers,
                               n_dim_mcmc,
                               theta_init,
                               file_prefix='prefix',
                               mcmcresultdir='.',
                               new_backend=False):
    """ initialize the MCMC by preparing the initial position of the
        walkers and the backend file

    Args:
        n_dim_mcmc: int, number of parameter in the MCMC
        nwalkers: int, number of walkers (at least 2 times n_dim_mcmc)
        theta_init: numpy array of dim n_dim_mcmc, set of initial parameters
        file_prefix: prefix name to save the backend
        mcmcresultdir='.': folder where to save the backend
        new_backend: bool, if new_backend=False, reset the backend, 
                           if new_backend=Falserestart the chains.
                           If you change the parameters or walkers numbers,
                            you have to restart with new_backend=True

    Returns:
        if new_backend=True then [intial position of the walkers, a clean BACKEND]
        if new_backend=False then [None, the loaded BACKEND]
    """

    distutils.dir_util.mkpath(mcmcresultdir)

    # Set up the backend h5
    # Don't forget to clear it in case the file already exists
    filename_backend = os.path.join(mcmcresultdir,
                                    file_prefix + "_backend_file_mcmc.h5")
    backend_ini = backends.HDFBackend(filename_backend)

    #############################################################
    # Initialize the walkers. The best technique seems to be
    # to start in a small ball around the a priori preferred position.
    # I start with a +/-0.1% ball for parameters defined in log and
    # +/-1% ball for the others

    if new_backend:
        p0 = np.zeros((1, nwalkers, n_dim_mcmc))
        for i in range(n_dim_mcmc):
            p0[:, :, i] = np.random.uniform(theta_init[i] * 0.999,
                                            theta_init[i] * 1.001,
                                            size=(nwalkers))

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

    logr1_init = np.log(params_mcmc_yaml['r1_init'])
    logr2_init = np.log(params_mcmc_yaml['r2_init'])
    cosinc_init = np.cos(np.radians(params_mcmc_yaml['inc_init']))
    pa_init = params_mcmc_yaml['pa_init']
    dx_init = params_mcmc_yaml['dx_init']
    dy_init = params_mcmc_yaml['dy_init']
    logN_init = np.log(params_mcmc_yaml['N_init'])

    if (SPF_MODEL == 'spf_fix'):
        beta_in_init = params_mcmc_yaml['beta_in_init']
        beta_out_init = params_mcmc_yaml['beta_out_init']
        a_r_init = params_mcmc_yaml['a_r_init']

        theta_init = (logr1_init, logr2_init, beta_in_init, beta_out_init,
                      a_r_init, cosinc_init, pa_init, dx_init, dy_init,
                      logN_init)

    elif (SPF_MODEL == 'hg_1g'):
        beta_init = params_mcmc_yaml['beta_init']
        g1_init = params_mcmc_yaml['g1_init']

        theta_init = (logr1_init, logr2_init, beta_init, cosinc_init, pa_init,
                      dx_init, dy_init, logN_init, g1_init)

    elif (SPF_MODEL == 'hg_2g'):
        beta_init = params_mcmc_yaml['beta_init']
        g1_init = params_mcmc_yaml['g1_init']
        g2_init = params_mcmc_yaml['g2_init']
        alpha1_init = params_mcmc_yaml['alpha1_init']

        theta_init = (logr1_init, logr2_init, beta_init, cosinc_init, pa_init,
                      dx_init, dy_init, logN_init, g1_init, g2_init,
                      alpha1_init)

    elif (SPF_MODEL == 'hg_3g'):
        beta_init = params_mcmc_yaml['beta_init']
        g1_init = params_mcmc_yaml['g1_init']
        g2_init = params_mcmc_yaml['g2_init']
        alpha1_init = params_mcmc_yaml['alpha1_init']
        g3_init = params_mcmc_yaml['g3_init']
        alpha2_init = params_mcmc_yaml['alpha2_init']

        theta_init = (logr1_init, logr2_init, beta_init, cosinc_init, pa_init,
                      dx_init, dy_init, logN_init, g1_init, g2_init,
                      alpha1_init, g3_init, alpha2_init)

    return np.asarray(theta_init)


if __name__ == '__main__':

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.simplefilter('ignore', FITSFixedWarning)
    warnings.simplefilter('ignore', NumbaWarning)
    # warnings.filterwarnings("ignore", category=UserWarning)
    # warnings.simplefilter('ignore', category=AstropyWarning)

    if args.mpi:  # MPI or not for parallelization.
        MPI = True
        progress = False
        mpistr = "\n In MPI mode"
    else:
        MPI = False
        progress = True
        mpistr = "\n In non MPI mode"

    if args.param_file is None:
        str_yalm = default_parameter_file
    else:
        str_yalm = args.param_file

    print("Read " + str_yalm + " parameter file")
    # open the parameter file
    yaml_path_file = os.path.join(os.getcwd(), 'initialization_files',
                                  str_yalm)
    with open(yaml_path_file, 'r') as yaml_file:
        params_mcmc_yaml = yaml.safe_load(yaml_file)

    FILE_PREFIX = params_mcmc_yaml['FILE_PREFIX']
    NEW_BACKEND = params_mcmc_yaml['NEW_BACKEND']

    if not os.path.isdir(os.path.join(basedir, params_mcmc_yaml['BAND_DIR'])):
        raise ValueError(
            "Could not find the data directory (BAND_DIR parameter)")

    KLIPDIR = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'],
                           'klip_fm_files')
    MCMCRESULTDIR = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'],
                                 'results_MCMC')

    # load in global the Parameters necessary to launch the MCMC
    NWALKERS = params_mcmc_yaml['NWALKERS']  #Number of walkers
    N_ITER_MCMC = params_mcmc_yaml['N_ITER_MCMC']  #Number of interation
    SPF_MODEL = params_mcmc_yaml['SPF_MODEL']  #Type of description for the SPF

    if SPF_MODEL == "spf_fix":  #1g henyey greenstein, SPF described with 1 parameter
        N_DIM_MCMC = 10  #Number of dimension of the parameter space

        # we fix the SPF using a HG parametrization with parameters in the init file
        n_points = 21  # odd number to ensure that scattangl=pi/2 is in the list for normalization
        scatt_angles = np.linspace(0, np.pi, n_points)

        # 2g henyey greenstein, normalized at 1 at 90 degrees
        spf_norm90 = hg_2g(np.degrees(scatt_angles),
                           params_mcmc_yaml['g1_init'],
                           params_mcmc_yaml['g2_init'],
                           params_mcmc_yaml['alpha1_init'], 1)
        #measure fo the spline and save as global value
        global F_SPF
        F_SPF = phase_function_spline(scatt_angles, spf_norm90)

        print(F_SPF)

    elif SPF_MODEL == "hg_1g":  #1g henyey greenstein, SPF described with 1 parameter
        N_DIM_MCMC = 9  #Number of dimension of the parameter space
    elif SPF_MODEL == "hg_2g":  #2g henyey greenstein, SPF described with 3 parameter
        N_DIM_MCMC = 11  #Number of dimension of the parameter space
    elif SPF_MODEL == "hg_3g":  #1g henyey greenstein, SPF described with 5 parameter
        N_DIM_MCMC = 13  #Number of dimension of the parameter space
    else:
        raise ValueError(SPF_MODEL + " not a valid SPF model")

    # load DISTANCE_STAR & PIXSCALE_INS and make them global
    DISTANCE_STAR = params_mcmc_yaml['DISTANCE_STAR']
    PIXSCALE_INS = params_mcmc_yaml['PIXSCALE_INS']
    ALIGNED_CENTER = params_mcmc_yaml['ALIGNED_CENTER']

    if params_mcmc_yaml['FIRST_TIME'] and MPI:
        raise ValueError("""Because the way the code is set up right now, 
                saving .fits seems complicated to do in MPI mode so we cannot 
                initialiaze in mpi mode, please initialiaze using 'FIRST_TIME=True' 
                only in sequential (to measure and save all the necessary .fits files 
                (PSF, masks, etc) and then run MPI with 'FIRST_TIME=False' """)

    # initialize the things necessary to measure the model (PSF, masks,
    # uncertainities). In RDI mode, psflib is also initiliazed here
    dataset, psflib = initialize_mask_psf_noise(params_mcmc_yaml,
                                                quietklip=True)

    ## Load all variables necessary for the MCMC and make them global
    ## to avoid very long transfert time at each iteration

    # load wheremask2generatedisk and make it global
    WHEREMASK2GENERATEDISK = (fits.getdata(
        os.path.join(KLIPDIR, FILE_PREFIX + '_mask2generatedisk.fits')) == 0)

    # load noise and make it global
    NOISE = fits.getdata(os.path.join(KLIPDIR, FILE_PREFIX + '_noisemap.fits'))

    # load PSF and make it global
    PSF = fits.getdata(os.path.join(KLIPDIR, FILE_PREFIX + '_SmallPSF.fits'))

    # load initial parameter value and make them global
    THETA_INIT = from_param_to_theta_init(params_mcmc_yaml)

    # measure the size of images DIMENSION and make it global
    DIMENSION = NOISE.shape[0]
    
    # initialize_diskfm and make diskobj global
    DISKOBJ = initialize_diskfm(dataset,
                                params_mcmc_yaml,
                                psflib=psflib,
                                quietklip=True)
    
    # Modification for Justin to save memory, slightly slower
    # del DISKOBJ

    # manager = mp.Manager()
    # KL_BASIS_FILE = manager.dict(_load_dict_from_hdf5(os.path.join(KLIPDIR,
    #                                               FILE_PREFIX + '_klbasis.h5')))

    
    # load reduced_data and make it a global variable
    REDUCED_DATA = fits.getdata(
        os.path.join(KLIPDIR, FILE_PREFIX + '-klipped-KLmodes-all.fits'))[
            0]  ### we take only the first KL mode

    # we multiply the reduced_data by the nan mask2minimize to avoid having
    # to pass mask2minimize as a global variable
    mask2minimize = fits.getdata(
        os.path.join(KLIPDIR, FILE_PREFIX + '_mask2minimize.fits'))
    mask2minimize[np.where(mask2minimize == 0.)] = np.nan
    REDUCED_DATA *= mask2minimize

    #last chance to delete useless big variables to avoid sending them
    # to every CPUs when paralelizing
    del mask2minimize, dataset, psflib, params_mcmc_yaml
    # print(globals())

    # Before launching th parallel MCMC
    # Make a final test "in c" by printing the likelyhood of the iniatial
    # set of parameter
    startTime = datetime.now()
    lnpb_model = lnpb(THETA_INIT)
    print("""Test: Likelyhood on initial parameter set is {0}. Time 
            from parameter values to Likelyhood (create model+FM+Likelyhood): 
            {1}""".format(lnpb_model,
                          datetime.now() - startTime))

    if not np.isfinite(lnpb_model):
        raise ValueError(
            """Do not launch MCMC, Likelyhood=-inf:your initial guess 
                            is probably out of the prior range for one of the parameter"""
        )

    print(mpistr + ", initialize walkers and start the MCMC...")
    startTime = datetime.now()

    with MultiPool() as pool:

        if MPI:
            if not pool.is_master():
                pool.wait()
                sys.exit(0)

        # initialize the walkers if necessary. initialize/load the backend
        # make them global
        init_walkers, BACKEND = initialize_walkers_backend(
            NWALKERS,
            N_DIM_MCMC,
            THETA_INIT,
            file_prefix=FILE_PREFIX,
            mcmcresultdir=MCMCRESULTDIR,
            new_backend=NEW_BACKEND)

        #Let's start the MCMC
        # Set up the Sampler. I purposefully passed the variables (KL modes,
        # reduced data, masks) in global variables to save time as advised in
        # https://emcee.readthedocs.io/en/latest/tutorials/parallel/
        # mode MPI or not

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
