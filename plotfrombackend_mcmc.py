# pylint: disable=C0103

####### This is the MCMC plotting code for HR 4796 data #######
import sys
import glob
import socket
import warnings

from datetime import datetime

import math as mt
import numpy as np

import astropy.io.fits as fits
from astropy.convolution import convolve

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.switch_backend('agg')
# There is a conflict when I import
# matplotlib with pyklip if I don't use this line

import corner
from emcee import backends

import yaml

import pyklip.instruments.GPI as GPI
import pyklip.instruments.Instrument as Instrument
from pyklip.fmlib.diskfm import DiskFM

from anadisk_johan import gen_disk_dxdy_2g, gen_disk_dxdy_3g
import astro_unit_conversion as convert
from kowalsky import kowalsky

# define global variables in the global scope
DISTANCE_STAR = PIXSCALE_INS = DIMENSION = None
wheremask2generatedisk = 12310120398


########################################################
def call_gen_disk_2g(theta):
    """ call the disk model from a set of parameters. 2g SPF
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
def crop_center(img, crop):
    y, x = img.shape
    startx = (x - 1) // 2 - crop // 2
    starty = (y - 1) // 2 - crop // 2
    return img[starty:starty + crop, startx:startx + crop]


########################################################
def offset_2_RA_dec(dx, dy, inclination, principal_angle, distance_star):
    """ right ascension and declination of the ellipse centre with respect to the star
        location from the offset in AU in the disk plane define by the max disk code

    Args:
        dx: offsetx of the star in AU in the disk plane define by the max disk code
            au, + -> NW offset disk plane Minor Axis
        dy: offsety of the star in AU in the disk plane define by the max disk code
            au, + -> SW offset disk plane Major Axis
        inclination: inclination in degrees
        principal_angle: prinipal angle in degrees

    Returns:
        [right ascension, declination]
    """

    dx_disk_mas = convert.au_to_mas(dx * np.cos(np.radians(inclination)),
                                    distance_star)
    dy_disk_mas = convert.au_to_mas(-dy, distance_star)

    dx_sky = np.cos(np.radians(principal_angle)) * dx_disk_mas - np.sin(
        np.radians(principal_angle)) * dy_disk_mas
    dy_sky = np.sin(np.radians(principal_angle)) * dx_disk_mas + np.cos(
        np.radians(principal_angle)) * dy_disk_mas

    dAlpha = -dx_sky
    dDelta = dy_sky

    return dAlpha, dDelta


########################################################
def make_chain_plot(params_mcmc_yaml):
    """ make_chain_plot reading the .h5 file from emcee

    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file

    Returns:
        None
    """

    THIN = params_mcmc_yaml['THIN']
    BURNIN = params_mcmc_yaml['BURNIN']
    QUALITY_PLOT = params_mcmc_yaml['QUALITY_PLOT']
    LABELS = params_mcmc_yaml['LABELS']
    NAMES = params_mcmc_yaml['NAMES']

    FILE_PREFIX = params_mcmc_yaml['FILE_PREFIX']

    name_h5 = FILE_PREFIX + '_backend_file_mcmc'

    reader = backends.HDFBackend(mcmcresultdir + name_h5 + '.h5')
    chain = reader.get_chain(discard=0, thin=THIN)
    log_prob_samples_flat = reader.get_log_prob(discard=BURNIN,
                                                flat=True,
                                                thin=THIN)
    # print(log_prob_samples_flat)
    tau = reader.get_autocorr_time(tol=0)

    print("")
    print("")
    print(name_h5)
    print("# of iteration in the backend chain initially: {0}".format(
        reader.iteration))
    print("Max Tau times 50: {0}".format(50 * np.max(tau)))
    print("")

    print("Maximum Likelyhood: {0}".format(np.max(log_prob_samples_flat)))

    print("burn-in: {0}".format(BURNIN))
    print("thin: {0}".format(THIN))
    print("chain shape: {0}".format(chain.shape))

    N_DIM_MCMC = chain.shape[2]
    NWALKERS = chain.shape[1]

    if N_DIM_MCMC == 11:
        ## change log and arccos values to physical
        chain[:, :, 0] = np.exp(chain[:, :, 0])
        chain[:, :, 1] = np.exp(chain[:, :, 1])
        chain[:, :, 6] = np.degrees(np.arccos(chain[:, :, 6]))
        chain[:, :, 10] = np.exp(chain[:, :, 10])

        ## change g1, g2 and alpha to percentage
        chain[:, :, 3] = 100 * chain[:, :, 3]
        chain[:, :, 4] = 100 * chain[:, :, 4]
        chain[:, :, 5] = 100 * chain[:, :, 5]

    if N_DIM_MCMC == 13:
        ## change log values to physical
        chain[:,:,0]=np.exp(chain[:,:,0])
        chain[:,:,1]=np.exp(chain[:,:,1])
        chain[:,:,8]=np.degrees(np.arccos(chain[:,:,8]))
        chain[:,:,12]=np.exp(chain[:,:,12])

        ## change g1, g2, g3, alpha1 and alpha2 to percentage
        chain[:,:,3]=100*chain[:,:,3]
        chain[:,:,4]=100*chain[:,:,4]
        chain[:,:,5]=100*chain[:,:,5]
        chain[:,:,6]=100*chain[:,:,6]
        chain[:,:,7]=100*chain[:,:,7]


    _, axarr = plt.subplots(N_DIM_MCMC,
                            sharex=True,
                            figsize=(6.4 * QUALITY_PLOT, 4.8 * QUALITY_PLOT))

    for i in range(N_DIM_MCMC):
        axarr[i].set_ylabel(LABELS[NAMES[i]], fontsize=5 * QUALITY_PLOT)
        axarr[i].tick_params(axis='y', labelsize=4 * QUALITY_PLOT)

        for j in range(NWALKERS):
            axarr[i].plot(chain[:, j, i], linewidth=QUALITY_PLOT)

        axarr[i].axvline(x=BURNIN, color='black', linewidth=1.5 * QUALITY_PLOT)

    axarr[N_DIM_MCMC - 1].tick_params(axis='x', labelsize=6 * QUALITY_PLOT)
    axarr[N_DIM_MCMC - 1].set_xlabel('Iterations', fontsize=10 * QUALITY_PLOT)

    plt.savefig(mcmcresultdir + name_h5 + '_chains.jpg')


########################################################
def make_corner_plot(params_mcmc_yaml):
    """ make corner plot reading the .h5 file from emcee

    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file


    Returns:
        None
    """

    THIN = params_mcmc_yaml['THIN']
    BURNIN = params_mcmc_yaml['BURNIN']
    LABELS = params_mcmc_yaml['LABELS']
    NAMES = params_mcmc_yaml['NAMES']
    sigma = params_mcmc_yaml['sigma']

    FILE_PREFIX = params_mcmc_yaml['FILE_PREFIX']

    name_h5 = FILE_PREFIX + '_backend_file_mcmc'

    BAND_NAME = params_mcmc_yaml['BAND_NAME']

    reader = backends.HDFBackend(mcmcresultdir + name_h5 + '.h5')
    chain = reader.get_chain(discard=BURNIN, thin=THIN)

    N_DIM_MCMC = chain.shape[2]

    if N_DIM_MCMC == 11:
        ## change log and arccos values to physical
        chain[:, :, 0] = np.exp(chain[:, :, 0])
        chain[:, :, 1] = np.exp(chain[:, :, 1])
        chain[:, :, 6] = np.degrees(np.arccos(chain[:, :, 6]))
        chain[:, :, 10] = np.exp(chain[:, :, 10])

        ## change g1, g2 and alpha to percentage
        chain[:, :, 3] = 100 * chain[:, :, 3]
        chain[:, :, 4] = 100 * chain[:, :, 4]
        chain[:, :, 5] = 100 * chain[:, :, 5]

    if N_DIM_MCMC == 13:
        ## change log values to physical
        chain[:,:,0]=np.exp(chain[:,:,0])
        chain[:,:,1]=np.exp(chain[:,:,1])
        chain[:,:,8]=np.degrees(np.arccos(chain[:,:,8]))
        chain[:,:,12]=np.exp(chain[:,:,12])

        ## change g1, g2, g3, alpha1 and alpha2 to percentage
        chain[:,:,3]=100*chain[:,:,3]
        chain[:,:,4]=100*chain[:,:,4]
        chain[:,:,5]=100*chain[:,:,5]
        chain[:,:,6]=100*chain[:,:,6]
        chain[:,:,7]=100*chain[:,:,7]


    samples = chain[:, :].reshape(-1, N_DIM_MCMC)

    rcParams['axes.labelsize'] = 19
    rcParams['axes.titlesize'] = 14

    rcParams['xtick.labelsize'] = 13
    rcParams['ytick.labelsize'] = 13

    ### cumulative percentiles
    ### value at 50% is the center of the Normal law
    ### value at 50% - value at 15.9% is -1 sigma
    ### value at 84.1%% - value at 50% is 1 sigma
    if sigma == 1:
        quants = (0.159, 0.5, 0.841)
    if sigma == 2:
        quants = (0.023, 0.5, 0.977)
    if sigma == 3:
        quants = (0.001, 0.5, 0.999)

    #### Check truths = bests parameters

    LABELS_hash = [LABELS[NAMES[i]] for i in range(N_DIM_MCMC)]
    fig = corner.corner(samples,
                        labels=LABELS_hash,
                        quantiles=quants,
                        show_titles=True,
                        plot_datapoints=False,
                        verbose=False)  # levels=(1-np.exp(-0.5),) , ))

    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0)

    fig.gca().annotate(BAND_NAME +
                       ": {0:,} iterations (192 walkers): {1:,} models".format(
                           reader.iteration, reader.iteration * 192),
                       xy=(0.55, 0.99),
                       xycoords="figure fraction",
                       xytext=(-20, -10),
                       textcoords="offset points",
                       ha="center",
                       va="top",
                       fontsize=44)

    plt.savefig(mcmcresultdir + name_h5 + '_pdfs.pdf')


########################################################
def create_header(params_mcmc_yaml):
    """ measure all the important parameters and exctract their error bars
        and print them and save them in a hdr file

    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file


    Returns:
        header for all the fits
    """

    THIN = params_mcmc_yaml['THIN']
    BURNIN = params_mcmc_yaml['BURNIN']

    COMMENTS = params_mcmc_yaml['COMMENTS']
    NAMES = params_mcmc_yaml['NAMES']

    DISTANCE_STAR = params_mcmc_yaml['DISTANCE_STAR']
    PIXSCALE_INS = params_mcmc_yaml['PIXSCALE_INS']



    sigma = params_mcmc_yaml['sigma']

    FILE_PREFIX = params_mcmc_yaml['FILE_PREFIX']
    name_h5 = FILE_PREFIX + '_backend_file_mcmc'

    reader = backends.HDFBackend(mcmcresultdir + name_h5 + '.h5')
    chain = reader.get_chain(discard=BURNIN, thin=THIN)
    log_prob_samples_flat = reader.get_log_prob(discard=BURNIN,
                                                flat=True,
                                                thin=THIN)

    N_DIM_MCMC = chain.shape[2]
    if N_DIM_MCMC == 11:
        ## change log and arccos values to physical
        chain[:, :, 0] = np.exp(chain[:, :, 0])
        chain[:, :, 1] = np.exp(chain[:, :, 1])
        chain[:, :, 6] = np.degrees(np.arccos(chain[:, :, 6]))
        chain[:, :, 10] = np.exp(chain[:, :, 10])

        ## change g1, g2 and alpha to percentage
        chain[:, :, 3] = 100 * chain[:, :, 3]
        chain[:, :, 4] = 100 * chain[:, :, 4]
        chain[:, :, 5] = 100 * chain[:, :, 5]

    if N_DIM_MCMC == 13:
        ## change log values to physical
        chain[:,:,0]=np.exp(chain[:,:,0])
        chain[:,:,1]=np.exp(chain[:,:,1])
        chain[:,:,8]=np.degrees(np.arccos(chain[:,:,8]))
        chain[:,:,12]=np.exp(chain[:,:,12])

        ## change g1, g2, g3, alpha1 and alpha2 to percentage
        chain[:,:,3]=100*chain[:,:,3]
        chain[:,:,4]=100*chain[:,:,4]
        chain[:,:,5]=100*chain[:,:,5]
        chain[:,:,6]=100*chain[:,:,6]
        chain[:,:,7]=100*chain[:,:,7]



    samples = chain[:, :].reshape(-1, chain.shape[2])

    N_DIM_MCMC = chain.shape[2]
    NWALKERS = chain.shape[1]

    samples_dict = dict()
    comments_dict = COMMENTS
    MLval_mcmc_val_mcmc_err_dict = dict()

    for i, key in enumerate(NAMES[:N_DIM_MCMC]):
        samples_dict[key] = samples[:, i]

    for i, key in enumerate(NAMES[N_DIM_MCMC:]):
        samples_dict[key] = samples[:, i] * 0.

    # measure of 6 other parameters:  right ascension, declination, and Kowalsky
    # (true_a, true_ecc, longnode, argperi)
    for j in range(samples.shape[0]):
        r1_here = samples_dict['R1'][j]
        inc_here = samples_dict['inc'][j]
        pa_here = samples_dict['PA'][j]
        dx_here = samples_dict['dx'][j]
        dy_here = samples_dict['dy'][j]
        dAlpha, dDelta = offset_2_RA_dec(dx_here, dy_here, inc_here, pa_here,
                                         DISTANCE_STAR)

        samples_dict['RA'][j] = dAlpha
        samples_dict['Decl'][j] = dDelta

        semimajoraxis = convert.au_to_mas(r1_here, DISTANCE_STAR)
        ecc = np.sin(np.radians(inc_here))

        true_a, true_ecc, argperi, inc, longnode = kowalsky(
            semimajoraxis, ecc, pa_here, dAlpha, dDelta)

        samples_dict['Rkowa'][j] = true_a
        samples_dict['ekowa'][j] = true_ecc
        samples_dict['ikowa'][j] = inc
        samples_dict['Omega'][j] = longnode
        samples_dict['Argpe'][j] = argperi

    wheremin = np.where(log_prob_samples_flat == np.max(log_prob_samples_flat))
    wheremin0 = np.array(wheremin).flatten()[0]

    if sigma == 1:
        quants = [0.159, 0.5, 0.841]
    if sigma == 2:
        quants = [0.023, 0.5, 0.977]
    if sigma == 3:
        quants = [0.001, 0.5, 0.999]

    for key in samples_dict.keys():
        MLval_mcmc_val_mcmc_err_dict[key] = np.zeros(4)

        percent = np.percentile(samples_dict[key], quants)

        MLval_mcmc_val_mcmc_err_dict[key][0] = samples_dict[key][wheremin0]
        MLval_mcmc_val_mcmc_err_dict[key][1] = percent[1]
        MLval_mcmc_val_mcmc_err_dict[key][2] = percent[0] - percent[1]
        MLval_mcmc_val_mcmc_err_dict[key][3] = percent[2] - percent[1]

    MLval_mcmc_val_mcmc_err_dict['RAp'] = convert.mas_to_pix(
        MLval_mcmc_val_mcmc_err_dict['RA'], PIXSCALE_INS)
    MLval_mcmc_val_mcmc_err_dict['Declp'] = convert.mas_to_pix(
        MLval_mcmc_val_mcmc_err_dict['Decl'], PIXSCALE_INS)

    print(" ")

    for key in samples_dict.keys():
        print(key +
              '_ML: {0:.3f}, MCMC {1:.3f}, -/+1sig: {2:.3f}/+{3:.3f}'.format(
                  MLval_mcmc_val_mcmc_err_dict[key][0],
                  MLval_mcmc_val_mcmc_err_dict[key][1],
                  MLval_mcmc_val_mcmc_err_dict[key][2],
                  MLval_mcmc_val_mcmc_err_dict[key][3]) + comments_dict[key])

    hdr = fits.Header()
    hdr['COMMENT'] = 'Best model of the MCMC reduction'
    hdr['COMMENT'] = 'PARAM_ML are the parameters producing the best LH'
    hdr['COMMENT'] = 'PARAM_MM are the parameters at the 50% percentile in the MCMC'
    hdr['COMMENT'] = 'PARAM_M and PARAM_P are the -/+ sigma error bars (16%, 84%)'
    hdr['KL_FILE'] = name_h5
    hdr['FITSDATE'] = str(datetime.now())
    hdr['BURNIN'] = BURNIN
    hdr['THIN'] = THIN

    hdr['TOT_ITER'] = reader.iteration

    hdr['n_walker'] = NWALKERS
    hdr['n_param'] = N_DIM_MCMC

    hdr['MAX_LH'] = (np.max(log_prob_samples_flat),
                     'Max likelyhood, obtained for the ML parameters')

    for key in samples_dict.keys():
        hdr[key + '_ML'] = (MLval_mcmc_val_mcmc_err_dict[key][0],
                            comments_dict[key])
        hdr[key + '_MC'] = MLval_mcmc_val_mcmc_err_dict[key][1]
        hdr[key + '_M'] = MLval_mcmc_val_mcmc_err_dict[key][2]
        hdr[key + '_P'] = MLval_mcmc_val_mcmc_err_dict[key][3]

    return hdr


########################################################
def best_model_plot(params_mcmc_yaml, hdr):
    """ Make the best models plot and save fits of
        BestModelBeforeConv
        BestModelAfterConv
        BestModelAfterFM
        BestModelResiduals

    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file
        hdr: the header obtained from create_header

    Returns:
        None
    """

    # I am going to plot the model, I need to define some of the
    # global variables to do so

    global PIXSCALE_INS, DISTANCE_STAR, wheremask2generatedisk, DIMENSION

    DISTANCE_STAR = params_mcmc_yaml['DISTANCE_STAR']
    PIXSCALE_INS = params_mcmc_yaml['PIXSCALE_INS']

    QUALITY_PLOT = params_mcmc_yaml['QUALITY_PLOT']
    FILE_PREFIX = params_mcmc_yaml['FILE_PREFIX']
    BAND_NAME = params_mcmc_yaml['BAND_NAME']
    name_h5 = FILE_PREFIX + '_backend_file_mcmc'

    KLMODE = [params_mcmc_yaml['KLMODE_NUMBER']]
    N_DIM_MCMC = hdr['n_param']

    #Format the most likely values
    #generate the best model
    if N_DIM_MCMC == 11:
        theta_ml = [
            np.log(hdr['R1_ML']),
            np.log(hdr['R2_ML']), hdr['Beta_ML'], 1 / 100. * hdr['g1_ML'],
            1 / 100. * hdr['g2_ML'], 1 / 100. * hdr['Alph1_ML'],
            np.cos(np.radians(hdr['inc_ML'])), hdr['PA_ML'], hdr['dx_ML'],
            hdr['dy_ML'],
            np.log(hdr['Norm_ML'])
        ]
    if N_DIM_MCMC == 13:
        theta_ml = [
            np.log(hdr['R1_ML']),
            np.log(hdr['R2_ML']), hdr['Beta_ML'], 1 / 100. * hdr['g1_ML'],
            1 / 100. * hdr['g2_ML'], 1 / 100. * hdr['g3_ML'],
            1 / 100. * hdr['Alph1_ML'], 1 / 100. * hdr['Alph2_ML'],
            np.cos(np.radians(hdr['inc_ML'])), hdr['PA_ML'], hdr['dx_ML'],
            hdr['dy_ML'],
            np.log(hdr['Norm_ML'])
        ]

    psf = fits.getdata(DATADIR + FILE_PREFIX + '_SatSpotPSF.fits')

    mask2generatedisk = fits.getdata(klipdir + FILE_PREFIX +
                                     '_mask2generatedisk.fits')

    mask2generatedisk[np.where(mask2generatedisk == 0.)] = np.nan
    wheremask2generatedisk = (mask2generatedisk != mask2generatedisk)

    # load the raw data (necessary to create the DiskFM obj)
    # this is the only part different for SPHERE and GPI

    if params_mcmc_yaml['BAND_DIR'] == 'SPHERE_Hdata/':
        #only for SPHERE
        xcen = params_mcmc_yaml['xcen']
        ycen = params_mcmc_yaml['ycen']
        datacube_sphere = fits.getdata(DATADIR + FILE_PREFIX + '_true_dataset.fits')
        parangs_sphere = fits.getdata(DATADIR + FILE_PREFIX + '_true_parangs.fits')

        size_datacube = datacube_sphere.shape
        centers_sphere = np.zeros((size_datacube[0], 2)) + [xcen,ycen]
        dataset = Instrument.GenericData(datacube_sphere,
                                        centers_sphere,
                                        parangs=parangs_sphere,
                                        wvs=None)
    else:
        #only for GPI
        filelist = glob.glob(DATADIR + "*distorcorr.fits")
        dataset = GPI.GPIData(filelist, quiet=True)

        #collapse the data spectrally
        dataset.spectral_collapse(align_frames=True)

    DIMENSION = dataset.input.shape[1]

    # load the data
    reduced_data = fits.getdata(klipdir + FILE_PREFIX +
                                '-klipped-KLmodes-all.fits')[
                                    0]  ### we take only the first KL mode

    # load the noise
    noise = fits.getdata(klipdir + FILE_PREFIX + '_noisemap.fits')

    #generate the best model
    if N_DIM_MCMC == 11:
        disk_ml = call_gen_disk_2g(theta_ml)
    if N_DIM_MCMC == 13:
        disk_ml = call_gen_disk_3g(theta_ml)

    new_fits = fits.HDUList()
    new_fits.append(fits.ImageHDU(data=disk_ml, header=hdr))
    new_fits.writeto(mcmcresultdir + name_h5 + '_BestModelBeforeConv.fits',
                     overwrite=True)

    #convolve by the PSF
    disk_ml_convolved = convolve(disk_ml, psf, boundary='wrap')

    new_fits = fits.HDUList()
    new_fits.append(fits.ImageHDU(data=disk_ml_convolved, header=hdr))
    new_fits.writeto(mcmcresultdir + name_h5 + '_BestModelAfterConv.fits',
                     overwrite=True)

    # load the KL numbers
    diskobj = DiskFM(dataset.input.shape,
                     KLMODE,
                     dataset,
                     disk_ml_convolved,
                     basis_filename=klipdir + FILE_PREFIX + '_klbasis.h5',
                     load_from_basis=True)

    #do the FM
    diskobj.update_disk(disk_ml_convolved)
    disk_ml_FM = diskobj.fm_parallelized()[
        0]  ### we take only the first KL modemode

    new_fits = fits.HDUList()
    new_fits.append(fits.ImageHDU(data=disk_ml_FM, header=hdr))
    new_fits.writeto(mcmcresultdir + name_h5 + '_BestModelAfterFM.fits',
                     overwrite=True)

    new_fits = fits.HDUList()
    new_fits.append(
        fits.ImageHDU(data=np.abs(reduced_data - disk_ml_FM), header=hdr))
    new_fits.writeto(mcmcresultdir + name_h5 + '_BestModelResiduals.fits',
                     overwrite=True)

    disk_ml_FM = fits.getdata(mcmcresultdir + name_h5 +
                              '_BestModelAfterFM.fits')

    #Set the colormap
    vmin = -2
    vmax = 16

    if params_mcmc_yaml['BAND_DIR'] == 'SPHERE_Hdata/':
        dim_crop_image = 232
    else:
        dim_crop_image = 196

    reduced_data_crop = crop_center(reduced_data, dim_crop_image)
    disk_ml_FM_crop = crop_center(disk_ml_FM, dim_crop_image)
    disk_ml_convolved_crop = crop_center(disk_ml_convolved, dim_crop_image)
    noise_crop = crop_center(noise, dim_crop_image)
    disk_ml_crop = crop_center(disk_ml, dim_crop_image)

    caracsize = 40 * QUALITY_PLOT / 2

    fig = plt.figure(figsize=(6.4 * 2 * QUALITY_PLOT, 4.8 * 2 * QUALITY_PLOT))
    #The data
    ax1 = fig.add_subplot(235)
    cax = plt.imshow(reduced_data_crop + 0.1,
                     origin='lower',
                     vmin=vmin,
                     vmax=vmax,
                     cmap=plt.cm.get_cmap('hot'))
    ax1.set_title("Original Data", fontsize=caracsize, pad=caracsize / 3.)
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=caracsize * 3 / 4)
    plt.axis('off')

    #The residuals
    ax1 = fig.add_subplot(233)
    cax = plt.imshow(np.abs(reduced_data_crop - disk_ml_FM_crop),
                     origin='lower',
                     vmin=0,
                     vmax=vmax / 3,
                     cmap=plt.cm.get_cmap('hot'))
    ax1.set_title("Residuals", fontsize=caracsize, pad=caracsize / 3.)
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=caracsize * 3 / 4)
    plt.axis('off')

    #The SNR of the residuals
    ax1 = fig.add_subplot(236)
    cax = plt.imshow(np.abs(reduced_data_crop - disk_ml_FM_crop) / noise_crop,
                     origin='lower',
                     vmin=0,
                     vmax=2,
                     cmap=plt.cm.get_cmap('hot'))
    ax1.set_title("SNR Residuals", fontsize=caracsize, pad=caracsize / 3.)
    cbar = fig.colorbar(cax, ticks=[0, 1, 2], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=caracsize * 3 / 4)
    cbar.ax.set_yticklabels(['0', '1', '2'])  # vertically oriented colorbar
    plt.axis('off')

    # The model
    ax1 = fig.add_subplot(231)
    cax = plt.imshow(disk_ml_crop,
                     origin='lower',
                     vmin=-2,
                     vmax=np.max(disk_ml_crop) / 1.5,
                     cmap=plt.cm.get_cmap('hot'))
    ax1.set_title("Best Model", fontsize=caracsize, pad=caracsize / 3.)
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=caracsize * 3 / 4)
    plt.axis('off')

    #The convolved model
    ax1 = fig.add_subplot(234)
    cax = plt.imshow(disk_ml_convolved_crop,
                     origin='lower',
                     vmin=vmin,
                     vmax=vmax,
                     cmap=plt.cm.get_cmap('hot'))

    ax1.set_title("Model Convolved", fontsize=caracsize, pad=caracsize / 3.)
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=caracsize * 3 / 4)
    plt.axis('off')

    #The FM convolved model
    ax1 = fig.add_subplot(232)
    cax = plt.imshow(disk_ml_FM_crop,
                     origin='lower',
                     vmin=vmin,
                     vmax=vmax,
                     cmap=plt.cm.get_cmap('hot'))
    ax1.set_title("Model Convolved + FM",
                  fontsize=caracsize,
                  pad=caracsize / 3.)
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=caracsize * 3 / 4)
    plt.axis('off')

    fig.subplots_adjust(hspace=-0.4, wspace=0.2)

    fig.suptitle(BAND_NAME + ': Best Model and Residuals',
                 fontsize=5 / 4. * caracsize,
                 y=0.985)

    fig.tight_layout()

    plt.savefig(mcmcresultdir + name_h5 + '_PlotBestModel.jpg')


########################################################
def print_geometry_parameter(params_mcmc_yaml, hdr):
    """ Print some of the important values from the header to put in
        excel

    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file
        hdr: the header obtained from create_header

    Returns:
        None
    """

    FILE_PREFIX = params_mcmc_yaml['FILE_PREFIX']
    DISTANCE_STAR = params_mcmc_yaml['DISTANCE_STAR']

    name_h5 = FILE_PREFIX + '_backend_file_mcmc'

    reader = backends.HDFBackend(mcmcresultdir + name_h5 + '.h5')

    f1 = open(mcmcresultdir + name_h5 + 'mcmcfit_geometrical_params.txt', 'w+')
    f1.write("\n'{0} / {1}".format(reader.iteration, reader.iteration * 192))
    f1.write("\n")

    to_print_str = 'R1'
    to_print = [
        hdr[to_print_str + '_MC'], hdr[to_print_str + '_M'],
        hdr[to_print_str + '_P']
    ]
    to_print = convert.au_to_mas(to_print, DISTANCE_STAR)
    f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(to_print[0], to_print[1],
                                                  to_print[2]))

    to_print_str = 'R2'
    to_print = [
        hdr[to_print_str + '_MC'], hdr[to_print_str + '_M'],
        hdr[to_print_str + '_P']
    ]
    to_print = convert.au_to_mas(to_print, DISTANCE_STAR)
    f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(to_print[0], to_print[1],
                                                  to_print[2]))

    to_print_str = 'PA'
    to_print = [
        hdr[to_print_str + '_MC'], hdr[to_print_str + '_M'],
        hdr[to_print_str + '_P']
    ]
    f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(to_print[0], to_print[1],
                                                  to_print[2]))

    to_print_str = 'RA'
    to_print = [
        hdr[to_print_str + '_MC'], hdr[to_print_str + '_M'],
        hdr[to_print_str + '_P']
    ]
    f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(to_print[0], to_print[1],
                                                  to_print[2]))

    to_print_str = 'Decl'
    to_print = [
        hdr[to_print_str + '_MC'], hdr[to_print_str + '_M'],
        hdr[to_print_str + '_P']
    ]
    f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(to_print[0], to_print[1],
                                                  to_print[2]))

    to_print_str = 'dx'
    to_print = [
        hdr[to_print_str + '_MC'], hdr[to_print_str + '_M'],
        hdr[to_print_str + '_P']
    ]
    to_print = convert.au_to_mas(to_print, DISTANCE_STAR)
    f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(to_print[0], to_print[1],
                                                  to_print[2]))

    to_print_str = 'dy'
    to_print = [
        hdr[to_print_str + '_MC'], hdr[to_print_str + '_M'],
        hdr[to_print_str + '_P']
    ]
    to_print = convert.au_to_mas(to_print, DISTANCE_STAR)
    f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(to_print[0], to_print[1],
                                                  to_print[2]))

    f1.write("\n")
    f1.write("\n")

    to_print_str = 'Rkowa'
    to_print = [
        hdr[to_print_str + '_MC'], hdr[to_print_str + '_M'],
        hdr[to_print_str + '_P']
    ]
    f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(to_print[0], to_print[1],
                                                  to_print[2]))

    to_print_str = 'eKOWA'
    to_print = [
        hdr[to_print_str + '_MC'], hdr[to_print_str + '_M'],
        hdr[to_print_str + '_P']
    ]
    f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(to_print[0], to_print[1],
                                                  to_print[2]))

    to_print_str = 'ikowa'
    to_print = [
        hdr[to_print_str + '_MC'], hdr[to_print_str + '_M'],
        hdr[to_print_str + '_P']
    ]
    f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(to_print[0], to_print[1],
                                                  to_print[2]))

    to_print_str = 'Omega'
    to_print = [
        hdr[to_print_str + '_MC'], hdr[to_print_str + '_M'],
        hdr[to_print_str + '_P']
    ]
    f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(to_print[0], to_print[1],
                                                  to_print[2]))

    to_print_str = 'Argpe'
    to_print = [
        hdr[to_print_str + '_MC'], hdr[to_print_str + '_M'],
        hdr[to_print_str + '_P']
    ]
    f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(to_print[0], to_print[1],
                                                  to_print[2]))

    f1.close()


if __name__ == '__main__':

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if len(sys.argv) == 1:
        str_yalm = 'SPHERE_Hband_MCMC.yaml'
    else:
        str_yalm = sys.argv[1]

    with open('initialization_files/' + str_yalm, 'r') as yaml_file:
        params_mcmc_yaml = yaml.load(yaml_file)

    # test on which machine I am
    if socket.gethostname() == 'MT-101942':
        basedir = '/Users/jmazoyer/Dropbox/ExchangeFolder/data_python/Aurora/'
    else:
        basedir = '/home/jmazoyer/data_python/Aurora/'

    BAND_DIR = params_mcmc_yaml['BAND_DIR']

    DATADIR = basedir + BAND_DIR
    klipdir = DATADIR + 'klip_fm_files/'
    mcmcresultdir = DATADIR + 'results_MCMC/'

    # Plot the chain values
    make_chain_plot(params_mcmc_yaml)

    # Plot the PDFs
    make_corner_plot(params_mcmc_yaml)

    # measure the best likelyhood model and excract MCMC errors
    hdr = create_header(params_mcmc_yaml)

    # save the fits, plot the model and residuals
    best_model_plot(params_mcmc_yaml, hdr)

    # print the values to put in excel sheet easily
    print_geometry_parameter(params_mcmc_yaml, hdr)
