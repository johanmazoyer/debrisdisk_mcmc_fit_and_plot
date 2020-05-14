# pylint: disable=C0103
"""
MCMC code for fitting a disk 
author: Johan Mazoyer
"""

import os
import argparse

basedir = os.environ["EXCHANGE_PATH"]  # the base directory where is
# your data (using OS environnement variable allow to use same code on
# different computer without changing this).
# default_parameter_file = 'FakeHr4796bright_MCMC_ADI.yaml'  # name of the parameter file

default_parameter_file = 'FakeHd181327bright_MCMC_ADI.yaml'  # name of the parameter file
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
    from multiprocessing import Pool as MultiPool

from multiprocessing import cpu_count

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
import pyklip.instruments.SPHERE as SPHERE

import pyklip.parallelized as parallelized
from pyklip.fmlib.diskfm import DiskFM
import pyklip.fm as fm
import pyklip.rdi as rdi

from disk_models import gen_disk_dxdy_1g, gen_disk_dxdy_2g, gen_disk_dxdy_3g
import make_gpi_psf_for_disks as gpidiskpsf
import astro_unit_conversion as convert

# recommended by emcee https://emcee.readthedocs.io/en/stable/tutorials/parallel/
# and by PyKLIPto avoid that NumPy automatically parallelizes some operations,
# which kill the speed
os.environ["OMP_NUM_THREADS"] = "1"


#######################################################
def call_gen_disk(theta):
    """ call the disk model from a set of parameters.
        
        use DIMENSION, PIXSCALE_INS and DISTANCE_STAR and
        wheremask2generatedisk as global variables

    Args:
        theta: list of parameters of the MCMC

    Returns:
        a 2d model
    """

    param_disk = {}

    param_disk['a_r'] = 0.01  # we fix the aspect ratio
    param_disk['offset'] = 0.  # no vertical offset in KLIP

    param_disk['r1'] = mt.exp(theta[0])
    param_disk['r2'] = mt.exp(theta[1])
    param_disk['beta'] = theta[2]
    param_disk['inc'] = np.degrees(np.arccos(theta[3]))
    param_disk['PA'] = theta[4]
    param_disk['dx'] = theta[5]
    param_disk['dy'] = theta[6]
    param_disk['Norm'] = mt.exp(theta[7])

    param_disk['SPF_MODEL'] = SPF_MODEL

    if (SPF_MODEL == 'hg_1g'):
        param_disk['g1'] = theta[8]

        #generate the model
        model = gen_disk_dxdy_1g(DIMENSION,
                                 param_disk,
                                 mask=WHEREMASK2GENERATEDISK,
                                 pixscale=PIXSCALE_INS,
                                 distance=DISTANCE_STAR)

    elif SPF_MODEL == 'hg_2g':
        param_disk['g1'] = theta[8]
        param_disk['g2'] = theta[9]
        param_disk['alpha1'] = theta[10]

        #generate the model
        model = gen_disk_dxdy_2g(DIMENSION,
                                 param_disk,
                                 mask=WHEREMASK2GENERATEDISK,
                                 pixscale=PIXSCALE_INS,
                                 distance=DISTANCE_STAR)

    elif SPF_MODEL == 'hg_3g':
        param_disk['g1'] = theta[8]
        param_disk['g2'] = theta[9]
        param_disk['alpha1'] = theta[10]
        param_disk['g3'] = theta[11]
        param_disk['alpha2'] = theta[12]

        #generate the model
        model = gen_disk_dxdy_3g(DIMENSION,
                                 param_disk,
                                 mask=WHEREMASK2GENERATEDISK,
                                 pixscale=PIXSCALE_INS,
                                 distance=DISTANCE_STAR)

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
    inc = np.degrees(np.arccos(theta[3]))
    pa = theta[4]
    dx = theta[5]
    dy = theta[6]
    Norm = mt.exp(theta[7])

    if (SPF_MODEL == 'hg_1g') or (SPF_MODEL == 'hg_2g') or (
            SPF_MODEL == 'hg_3g'):
        g1 = theta[8]

        if (SPF_MODEL == 'hg_2g') or (SPF_MODEL == 'hg_3g'):
            g2 = theta[9]
            alpha1 = theta[10]

            if SPF_MODEL == 'hg_3g':
                g3 = theta[11]
                alpha2 = theta[12]

    prior_rout = 1.
    # define the prior values
    if (r1 < 60 or r1 > 80):
        return -np.inf
    else:
        prior_rout = prior_rout * 1.

    # - rout = Logistic We arbitralily cut the prior at r2 = 100
    # (~25 AU large) because this parameter is very limited by the ADI
    if ((r2 < 82) and (r2 > 102)):
        return -np.inf
    else:
        prior_rout = prior_rout / (1. + np.exp(40. * (r2 - 100)))
        # prior_rout = prior_rout  *1. # or we can just use a flat prior

    if (beta < 1 or beta > 30):
        return -np.inf
    else:
        prior_rout = prior_rout * 1.

    # if (a_r < 0.0001 or a_r > 0.5 ): #The aspect ratio
    #     return -np.inf
    # else:
    #    prior_rout = prior_rout  *1.

    if (inc < 78 or inc > 98):
        return -np.inf
    else:
        prior_rout = prior_rout * 1.

    if (pa < 40 or pa > 50):
        return -np.inf
    else:
        prior_rout = prior_rout * 1.

    if (dx < -10) or (dx > 10):  #The x offset
        return -np.inf
    else:
        prior_rout = prior_rout * 1.

    if (dy < -10) or (dy > 10):  #The y offset
        return -np.inf
    else:
        prior_rout = prior_rout * 1.

    if (Norm < 0.5 or Norm > 50000):
        return -np.inf
    else:
        prior_rout = prior_rout * 1.

    if (SPF_MODEL == 'hg_1g') or (SPF_MODEL == 'hg_2g') or (
            SPF_MODEL == 'hg_3g'):
        if (g1 < 0.05 or g1 > 0.9999):
            return -np.inf
        else:
            prior_rout = prior_rout * 1.

        if (SPF_MODEL == 'hg_2g') or (SPF_MODEL == 'hg_3g'):
            if (g2 < -0.9999 or g2 > -0.05):
                return -np.inf
            else:
                prior_rout = prior_rout * 1.

            if (alpha1 < 0.01 or alpha1 > 0.9999):
                return -np.inf
            else:
                prior_rout = prior_rout * 1.

            if SPF_MODEL == 'hg_3g':
                if (g3 < -1 or g3 > 1):
                    return -np.inf
                else:
                    prior_rout = prior_rout * 1.

                if (alpha2 < -1 or alpha2 > 1):
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
    # starttime=datetime.now()
    lp = logp(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = logl(theta)
    # print("Running time model + FM: ", datetime.now()-starttime)

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
    x = np.arange(dim, dtype=np.float)[None, :] - aligned_center[0]
    y = np.arange(dim, dtype=np.float)[:, None] - aligned_center[1]
    rho2d = np.sqrt(x**2 + y**2)

    noise_map = np.zeros((dim, dim))
    for i_ring in range(0,
                        int(np.floor(aligned_center[0] / delta_raddii)) - 2):
        wh_rings = np.where((rho2d >= i_ring * delta_raddii)
                            & (rho2d < (i_ring + 1) * delta_raddii))
        noise_map[wh_rings] = np.nanstd(nodisk_data[wh_rings])
    return noise_map


def create_uncertainty_map(dataset, params_mcmc_yaml, psflib=None):

    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    move_here = params_mcmc_yaml['MOVE_HERE']
    numbasis = [params_mcmc_yaml['KLMODE_NUMBER']]
    mode = params_mcmc_yaml['MODE']
    annuli = params_mcmc_yaml['ANNULI']
    aligned_center = params_mcmc_yaml['ALIGNED_CENTER']
    noise_multiplication_factor = params_mcmc_yaml[
        'NOISE_MULTIPLICATION_FACTOR']

    #measure the uncertainty map using the counter rotation trick
    # described in Sec4 of Gerard&Marois SPIE 2016 and probabaly elsewhere

    dataset.PAs = -dataset.PAs

    parallelized.klip_dataset(dataset,
                              numbasis=numbasis,
                              maxnumbasis=120,
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

    if instrument == 'SPHERE':
        # only for SPHERE. 
        data_files_str = params_mcmc_yaml['DATA_FILES_STR']
        psf_files_str = params_mcmc_yaml['PSF_FILES_STR']
        angles_str = params_mcmc_yaml['ANGLES_STR']
        band_name = params_mcmc_yaml['BAND_NAME']

        dataset = SPHERE.Irdis(data_files_str, psf_files_str, angles_str, band_name, psf_cube_size=31)
        #collapse the data spectrally
        dataset.spectral_collapse(align_frames=True,
                                  aligned_center=aligned_center)
        
        if first_time:     
            fits.writeto(os.path.join(klipdir,
                                      file_prefix + '_SmallPSF.fits'),
                         dataset.psfs,
                         overwrite='True')

            

    elif instrument == 'GPI':
        #only for GPI. Data reduction is simpler for GPI but PSF
        # measurement form satspot is more complicated.

        if first_time:
            print("\n Create a PSF from the sat spots")
            filelist4psf = sorted(glob.glob(os.path.join(datadir, "*.fits")))

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

            # extract the data this time wihtout the bad files nor slices
            dataset4psf = GPI.GPIData(filelist4psf,
                                      quiet=True,
                                      skipslices=excluded_slices)

            # finally measure the good psf
            instrument_psf = gpidiskpsf.make_collapsed_psf(dataset4psf,
                                                           params_mcmc_yaml,
                                                           boxrad=15)

            # save the excluded_slices in the psf header (SNR too low)
            hdr_psf = fits.Header()
            hdr_psf['N_BADSLI'] = len(excluded_slices)
            for badslice_i, excluded_slices_num in enumerate(excluded_slices):
                hdr_psf['BADSLI' +
                        str(badslice_i).zfill(2)] = excluded_slices_num

            # save the excluded_files in the psf header (disk on satspots)
            hdr_psf['N_BADFIL'] = len(excluded_files)
            for badfile_i, badfilestr in enumerate(excluded_files):
                hdr_psf['BADFIL' + str(badfile_i).zfill(2)] = badfilestr

            #save the psf
            fits.writeto(os.path.join(klipdir,
                                      file_prefix + '_SmallPSF.fits'),
                         instrument_psf,
                         header=hdr_psf,
                         overwrite=True)

        filelist = sorted(glob.glob(os.path.join(datadir, "*.fits")))

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
                    excluded_files.append(hdr_psf['BADFIL' +
                                                  str(badfile_i).zfill(2)])

            for excluded_filesi in excluded_files:
                if excluded_filesi in filelist:
                    filelist.remove(excluded_filesi)

        # in IFS mode, we always exclude the IFS slices with too much noise. We
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

    #After this, this is for both GPI and SPHERE
    #define the outer working angle
    dataset.OWA = params_mcmc_yaml['OWA']

    if dataset.input.shape[1] != dataset.input.shape[2]:
        raise ValueError(""" Data slices are not square (dimx!=dimy), 
                        please make them square""")

    #create the masks
    print("\n Create the binary masks to define model zone and chisquare zone")
    if first_time:
        #create the mask where the non convoluted disk is going to be generated.
        # To gain time, it is tightely adjusted to the expected models BEFORE
        # convolution. Inded, the models are generated pixel by pixels. 0.1 s
        # gained on every model is a day of calculation gain on one million model,
        # so adjust your mask tightly to your model. Carefull somes mask paramters are
        # hardcoded here

        mask_disk_zeros = gpidiskpsf.make_disk_mask(
            dataset.input.shape[1],
            params_mcmc_yaml['pa_init'],
            params_mcmc_yaml['inc_init'],
            convert.au_to_pix(params_mcmc_yaml['r1_init'] - 69,
                              params_mcmc_yaml['PIXSCALE_INS'],
                              params_mcmc_yaml['DISTANCE_STAR']),
            convert.au_to_pix(params_mcmc_yaml['r2_init'] + 100,
                              params_mcmc_yaml['PIXSCALE_INS'],
                              params_mcmc_yaml['DISTANCE_STAR']),
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
            convert.au_to_pix(params_mcmc_yaml['r1_init'] - 69,
                              params_mcmc_yaml['PIXSCALE_INS'],
                              params_mcmc_yaml['DISTANCE_STAR']),
            convert.au_to_pix(params_mcmc_yaml['r2_init'] + 100,
                              params_mcmc_yaml['PIXSCALE_INS'],
                              params_mcmc_yaml['DISTANCE_STAR']),
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

    mask2generatedisk = fits.getdata(
        os.path.join(klipdir, file_prefix + '_mask2generatedisk.fits'))
    mask2minimize = fits.getdata(
        os.path.join(klipdir, file_prefix + '_mask2minimize.fits'))

    # RDI case, if it's not the first time, we do not even need to load the correlation
    # psflib, all needed information is already loaded in the KL modes
    if params_mcmc_yaml['MODE'] == 'RDI' and first_time:
        psflib = initialize_rdi(dataset, params_mcmc_yaml)
    else:
        psflib = None

    if first_time:
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

    rdidir = os.path.join(datadir, params_mcmc_yaml['RDI_DIR'])
    rdi_matrix_dir = os.path.join(rdidir, 'rdi_matrix')
    distutils.dir_util.mkpath(rdi_matrix_dir)

    if do_rdi_correlation:

        # load the bad slices in the psf header (IFS slices where satspots SNR < 3).
        # This is only for GPI. Normally this should not happen because RDI is in
        # H band and H band data have very little thermal noise.

        hdr_psf = fits.getheader(
            os.path.join(klipdir, file_prefix + '_SmallPSF.fits'))

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
                         aligned_center=aligned_center,
                         numthreads=1)
        # measure the KL basis and save it
        fm.klip_dataset(dataset,
                        diskobj,
                        numbasis=numbasis,
                        maxnumbasis=120,
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
        init_ball0 = np.random.uniform(theta_init[0] * 0.999,
                                       theta_init[0] * 1.001,
                                       size=(nwalkers))  #logr1
        init_ball1 = np.random.uniform(theta_init[1] * 0.999,
                                       theta_init[1] * 1.001,
                                       size=(nwalkers))  #logr2
        init_ball2 = np.random.uniform(theta_init[2] * 0.99,
                                       theta_init[2] * 1.01,
                                       size=(nwalkers))  #beta
        init_ball3 = np.random.uniform(theta_init[3] * 0.99,
                                       theta_init[3] * 1.01,
                                       size=(nwalkers))  #cosinc
        init_ball4 = np.random.uniform(theta_init[4] * 0.99,
                                       theta_init[4] * 1.01,
                                       size=(nwalkers))  #pa
        init_ball5 = np.random.uniform(theta_init[5] * 0.99,
                                       theta_init[5] * 1.01,
                                       size=(nwalkers))  #dx
        init_ball6 = np.random.uniform(theta_init[6] * 0.99,
                                       theta_init[6] * 1.01,
                                       size=(nwalkers))  #dy
        init_ball7 = np.random.uniform(theta_init[7] * 0.999,
                                       theta_init[7] * 1.001,
                                       size=(nwalkers))  #logNorm

        if (SPF_MODEL == 'hg_1g'):
            init_ball8 = np.random.uniform(theta_init[8] * 0.99,
                                           theta_init[8] * 1.01,
                                           size=(nwalkers))  #g1
            p0 = np.dstack(
                (init_ball0, init_ball1, init_ball2, init_ball3, init_ball4,
                 init_ball5, init_ball6, init_ball7, init_ball8))

        elif (SPF_MODEL == 'hg_2g'):
            init_ball8 = np.random.uniform(theta_init[8] * 0.99,
                                           theta_init[8] * 1.01,
                                           size=(nwalkers))  #g1
            init_ball9 = np.random.uniform(theta_init[9] * 0.99,
                                           theta_init[9] * 1.01,
                                           size=(nwalkers))  #g2
            init_ball10 = np.random.uniform(theta_init[10] * 0.99,
                                            theta_init[10] * 1.01,
                                            size=(nwalkers))  #alpha1
            p0 = np.dstack((init_ball0, init_ball1, init_ball2, init_ball3,
                            init_ball4, init_ball5, init_ball6, init_ball7,
                            init_ball8, init_ball9, init_ball10))

        elif (SPF_MODEL == 'hg_3g'):
            init_ball8 = np.random.uniform(theta_init[8] * 0.99,
                                           theta_init[8] * 1.01,
                                           size=(nwalkers))  #g1
            init_ball9 = np.random.uniform(theta_init[9] * 0.99,
                                           theta_init[9] * 1.01,
                                           size=(nwalkers))  #g2
            init_ball10 = np.random.uniform(theta_init[10] * 0.99,
                                            theta_init[10] * 1.01,
                                            size=(nwalkers))  #alpha1
            init_ball11 = np.random.uniform(theta_init[11] * 0.99,
                                            theta_init[11] * 1.01,
                                            size=(nwalkers))  #g3
            init_ball12 = np.random.uniform(theta_init[12] * 0.99,
                                            theta_init[12] * 1.01,
                                            size=(nwalkers))  #alpha2

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

    logr1_init = np.log(params_mcmc_yaml['r1_init'])
    logr2_init = np.log(params_mcmc_yaml['r2_init'])
    beta_init = params_mcmc_yaml['beta_init']
    cosinc_init = np.cos(np.radians(params_mcmc_yaml['inc_init']))
    pa_init = params_mcmc_yaml['pa_init']
    dx_init = params_mcmc_yaml['dx_init']
    dy_init = params_mcmc_yaml['dy_init']
    logN_init = np.log(params_mcmc_yaml['N_init'])

    if (SPF_MODEL == 'hg_1g'):
        g1_init = params_mcmc_yaml['g1_init']

        theta_init = (logr1_init, logr2_init, beta_init, cosinc_init, pa_init,
                      dx_init, dy_init, logN_init, g1_init)

    elif (SPF_MODEL == 'hg_2g'):
        g1_init = params_mcmc_yaml['g1_init']
        g2_init = params_mcmc_yaml['g2_init']
        alpha1_init = params_mcmc_yaml['alpha1_init']

        theta_init = (logr1_init, logr2_init, beta_init, cosinc_init, pa_init,
                      dx_init, dy_init, logN_init, g1_init, g2_init,
                      alpha1_init)

    elif (SPF_MODEL == 'hg_3g'):
        g1_init = params_mcmc_yaml['g1_init']
        g2_init = params_mcmc_yaml['g2_init']
        alpha1_init = params_mcmc_yaml['alpha1_init']
        g3_init = params_mcmc_yaml['g3_init']
        alpha2_init = params_mcmc_yaml['alpha2_init']

        theta_init = (logr1_init, logr2_init, beta_init, cosinc_init, pa_init,
                      dx_init, dy_init, logN_init, g1_init, g2_init,
                      alpha1_init, g3_init, alpha2_init)

    return theta_init


if __name__ == '__main__':

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.simplefilter('ignore', FITSFixedWarning)
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

    if args.param_file == None:
        str_yalm = default_parameter_file
    else:
        str_yalm = args.param_file

    print("Read " + str_yalm + " parameter file")
    # open the parameter file
    yaml_path_file = os.path.join(os.getcwd(), 'initialization_files',
                                  str_yalm)
    with open(yaml_path_file, 'r') as yaml_file:
        params_mcmc_yaml = yaml.load(yaml_file)

    FILE_PREFIX = params_mcmc_yaml['FILE_PREFIX']
    NEW_BACKEND = params_mcmc_yaml['NEW_BACKEND']

    klipdir = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'],
                           'klip_fm_files')
    MCMCRESULTDIR = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'],
                                 'results_MCMC')

    # load in global the Parameters necessary to launch the MCMC
    NWALKERS = params_mcmc_yaml['NWALKERS']  #Number of walkers
    N_ITER_MCMC = params_mcmc_yaml['N_ITER_MCMC']  #Number of interation
    SPF_MODEL = params_mcmc_yaml['SPF_MODEL']  #Type of description for the SPF

    if SPF_MODEL == "hg_1g":  #1g henyey greenstein, SPF described with 1 parameter
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

    # measure the size of images DIMENSION and make it global
    DIMENSION = dataset.input.shape[1]
    # load wheremask2generatedisk and make it global
    WHEREMASK2GENERATEDISK = (fits.getdata(
        os.path.join(klipdir, FILE_PREFIX + '_mask2generatedisk.fits')) == 0)

    # load noise and make it global
    NOISE = fits.getdata(os.path.join(klipdir, FILE_PREFIX + '_noisemap.fits'))

    # load PSF and make it global
    PSF = fits.getdata(os.path.join(klipdir, FILE_PREFIX + '_SmallPSF.fits'))

    # load initial parameter value and make them global
    THETA_INIT = from_param_to_theta_init(params_mcmc_yaml)

    # initialize_diskfm and make diskobj global
    DISKOBJ = initialize_diskfm(dataset,
                                params_mcmc_yaml,
                                psflib=psflib,
                                quietklip=True)

    # load reduced_data and make it a global variable
    REDUCED_DATA = fits.getdata(
        os.path.join(klipdir, FILE_PREFIX + '-klipped-KLmodes-all.fits'))[
            0]  ### we take only the first KL mode
    # we multiply the reduced_data by the nan mask2minimize to avoid having
    # to pass mask2minimize as a global variable
    mask2minimize = fits.getdata(
        os.path.join(klipdir, FILE_PREFIX + '_mask2minimize.fits'))
    mask2minimize[np.where(mask2minimize == 0.)] = np.nan
    REDUCED_DATA *= mask2minimize

    #last chance to delete useless big variables to avoid sending them
    # to every CPUs when paralelizing
    del mask2minimize, dataset, psflib, params_mcmc_yaml, klipdir
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
