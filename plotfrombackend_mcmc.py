# pylint: disable=C0103

####### This is the MCMC plotting code for HR 4796 data #######
import sys
import os
import glob
import socket
import warnings

from datetime import datetime

import math as mt
import numpy as np

import astropy.io.fits as fits
from astropy.convolution import convolve

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines

import yaml

import corner
from emcee import backends

import pyklip.instruments.GPI as GPI
import pyklip.instruments.Instrument as Instrument
from pyklip.fmlib.diskfm import DiskFM

from anadisk_johan import gen_disk_dxdy_2g, gen_disk_dxdy_3g
import astro_unit_conversion as convert
from kowalsky import kowalsky
import make_gpi_psf_for_disks as gpidiskpsf

plt.switch_backend('agg')
# There is a conflict when I import
# matplotlib with pyklip if I don't use this line

# define global variables in the global scope
DISTANCE_STAR = PIXSCALE_INS = DIMENSION = None
wheremask2generatedisk = 12310120398


########################################################
def call_gen_disk_2g(theta):
    """ call the disk model from a set of parameters. 2g SPF
        use DIMENSION, PIXSCALE_INS and distance_star  and
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

    thin = params_mcmc_yaml['THIN']
    burnin = params_mcmc_yaml['BURNIN']
    quality_plot = params_mcmc_yaml['QUALITY_PLOT']
    labels = params_mcmc_yaml['LABELS']
    names = params_mcmc_yaml['NAMES']

    file_prefix = params_mcmc_yaml['FILE_PREFIX']

    name_h5 = file_prefix + '_backend_file_mcmc'

    reader = backends.HDFBackend(os.path.join(mcmcresultdir, name_h5 + '.h5'))
    chain = reader.get_chain(discard=0, thin=thin)
    log_prob_samples_flat = reader.get_log_prob(discard=burnin,
                                                flat=True,
                                                thin=thin)
    # print(log_prob_samples_flat)
    tau = reader.get_autocorr_time(tol=0)
    if burnin > reader.iteration - 1:
        raise ValueError(
            "the burnin cannot be larger than the # of iterations")
    print("")
    print("")
    print(name_h5)
    print("# of iteration in the backend chain initially: {0}".format(
        reader.iteration))
    print("Max Tau times 50: {0}".format(50 * np.max(tau)))
    print("")

    print("Maximum Likelyhood: {0}".format(np.nanmax(log_prob_samples_flat)))

    print("burn-in: {0}".format(burnin))
    print("thin: {0}".format(thin))
    print("chain shape: {0}".format(chain.shape))

    n_dim_mcmc = chain.shape[2]
    nwalkers = chain.shape[1]

    if n_dim_mcmc == 11:
        ## change log and arccos values to physical
        chain[:, :, 0] = np.exp(chain[:, :, 0])
        chain[:, :, 1] = np.exp(chain[:, :, 1])
        chain[:, :, 6] = np.degrees(np.arccos(chain[:, :, 6]))
        chain[:, :, 10] = np.exp(chain[:, :, 10])

        ## change g1, g2 and alpha to percentage
        chain[:, :, 3] = 100 * chain[:, :, 3]
        chain[:, :, 4] = 100 * chain[:, :, 4]
        chain[:, :, 5] = 100 * chain[:, :, 5]

    if n_dim_mcmc == 13:
        ## change log values to physical
        chain[:, :, 0] = np.exp(chain[:, :, 0])
        chain[:, :, 1] = np.exp(chain[:, :, 1])
        chain[:, :, 8] = np.degrees(np.arccos(chain[:, :, 8]))
        chain[:, :, 12] = np.exp(chain[:, :, 12])

        ## change g1, g2, g3, alpha1 and alpha2 to percentage
        chain[:, :, 3] = 100 * chain[:, :, 3]
        chain[:, :, 4] = 100 * chain[:, :, 4]
        chain[:, :, 5] = 100 * chain[:, :, 5]
        chain[:, :, 6] = 100 * chain[:, :, 6]
        chain[:, :, 7] = 100 * chain[:, :, 7]

    _, axarr = plt.subplots(n_dim_mcmc,
                            sharex=True,
                            figsize=(6.4 * quality_plot, 4.8 * quality_plot))

    for i in range(n_dim_mcmc):
        axarr[i].set_ylabel(labels[names[i]], fontsize=5 * quality_plot)
        axarr[i].tick_params(axis='y', labelsize=4 * quality_plot)

        for j in range(nwalkers):
            axarr[i].plot(chain[:, j, i], linewidth=quality_plot)

        axarr[i].axvline(x=burnin, color='black', linewidth=1.5 * quality_plot)

    axarr[n_dim_mcmc - 1].tick_params(axis='x', labelsize=6 * quality_plot)
    axarr[n_dim_mcmc - 1].set_xlabel('Iterations', fontsize=10 * quality_plot)

    plt.savefig(os.path.join(mcmcresultdir, name_h5 + '_chains.jpg'))


########################################################
def make_corner_plot(params_mcmc_yaml):
    """ make corner plot reading the .h5 file from emcee

    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file


    Returns:
        None
    """

    thin = params_mcmc_yaml['THIN']
    burnin = params_mcmc_yaml['BURNIN']
    labels = params_mcmc_yaml['LABELS']
    names = params_mcmc_yaml['NAMES']
    sigma = params_mcmc_yaml['sigma']
    nwalkers = params_mcmc_yaml['NWALKERS']

    n_dim_mcmc = params_mcmc_yaml['N_DIM_MCMC']

    file_prefix = params_mcmc_yaml['FILE_PREFIX']

    name_h5 = file_prefix + '_backend_file_mcmc'

    band_name = params_mcmc_yaml['BAND_NAME']

    reader = backends.HDFBackend(os.path.join(mcmcresultdir, name_h5 + '.h5'))
    chain_flat = reader.get_chain(discard=burnin, thin=thin, flat=True)

    if n_dim_mcmc == 11:
        ## change log and arccos values to physical
        chain_flat[:, 0] = np.exp(chain_flat[:, 0])
        chain_flat[:, 1] = np.exp(chain_flat[:, 1])
        chain_flat[:, 6] = np.degrees(np.arccos(chain_flat[:, 6]))
        chain_flat[:, 10] = np.exp(chain_flat[:, 10])

        ## change g1, g2 and alpha to percentage
        chain_flat[:, 3] = 100 * chain_flat[:, 3]
        chain_flat[:, 4] = 100 * chain_flat[:, 4]
        chain_flat[:, 5] = 100 * chain_flat[:, 5]

    if n_dim_mcmc == 13:
        ## change log values to physical
        chain_flat[:, 0] = np.exp(chain_flat[:, 0])
        chain_flat[:, 1] = np.exp(chain_flat[:, 1])
        chain_flat[:, 8] = np.degrees(np.arccos(chain_flat[:, 8]))
        chain_flat[:, 12] = np.exp(chain_flat[:, 12])

        ## change g1, g2, g3, alpha1 and alpha2 to percentage
        chain_flat[:, 3] = 100 * chain_flat[:, 3]
        chain_flat[:, 4] = 100 * chain_flat[:, 4]
        chain_flat[:, 5] = 100 * chain_flat[:, 5]
        chain_flat[:, 6] = 100 * chain_flat[:, 6]
        chain_flat[:, 7] = 100 * chain_flat[:, 7]

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
    if file_prefix == 'Hband_hd48524_fake':
        shouldweplotalldatapoints = True
    else:
        shouldweplotalldatapoints = False
    labels_hash = [labels[names[i]] for i in range(n_dim_mcmc)]
    fig = corner.corner(chain_flat,
                        labels=labels_hash,
                        quantiles=quants,
                        show_titles=True,
                        plot_datapoints=shouldweplotalldatapoints,
                        verbose=False)

    if file_prefix == 'Hband_hd48524_fake':
        initial_values = [
            74.5, 100, 12.4, 82.5, -20.1, 29.8, 76.8, 26.64, -2., 0.94, 80
        ]

        green_line = mlines.Line2D([], [],
                                   color='green',
                                   label='True injected values')
        plt.legend(handles=[green_line],
                   loc='upper right',
                   bbox_to_anchor=(-1, 10),
                   fontsize=30)

        # log_prob_samples_flat = reader.get_log_prob(discard=burnin,
        #                                             flat=True,
        #                                             thin=thin)
        # wheremin = np.where(
        #     log_prob_samples_flat == np.max(log_prob_samples_flat))
        # wheremin0 = np.array(wheremin).flatten()[0]

        # red_line = mlines.Line2D([], [],
        #                         color='red',
        #                         label='Maximum likelyhood values')
        # plt.legend(handles=[green_line, red_line],
        #         loc='upper right',
        #         bbox_to_anchor=(-1, 10),
        #         fontsize=30)

        # Extract the axes
        axes = np.array(fig.axes).reshape((n_dim_mcmc, n_dim_mcmc))

        # Loop over the diagonal
        for i in range(n_dim_mcmc):
            ax = axes[i, i]
            ax.axvline(initial_values[i], color="g")
            # ax.axvline(samples[wheremin0, i], color="r")

        # Loop over the histograms
        for yi in range(n_dim_mcmc):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(initial_values[xi], color="g")
                ax.axhline(initial_values[yi], color="g")

                # ax.axvline(samples[wheremin0, xi], color="r")
                # ax.axhline(samples[wheremin0, yi], color="r")

    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0)

    fig.gca().annotate(band_name +
                       ": {0:,} iterations (with {1:,} burn-in)".format(
                           reader.iteration, burnin),
                       xy=(0.55, 0.99),
                       xycoords="figure fraction",
                       xytext=(-20, -10),
                       textcoords="offset points",
                       ha="center",
                       va="top",
                       fontsize=44)

    fig.gca().annotate("{0:,} walkers: {1:,} models".format(
        nwalkers, reader.iteration * nwalkers),
                       xy=(0.55, 0.95),
                       xycoords="figure fraction",
                       xytext=(-20, -10),
                       textcoords="offset points",
                       ha="center",
                       va="top",
                       fontsize=44)

    plt.savefig(os.path.join(mcmcresultdir, name_h5 + '_pdfs.pdf'))


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

    thin = params_mcmc_yaml['THIN']
    burnin = params_mcmc_yaml['BURNIN']

    comments = params_mcmc_yaml['COMMENTS']
    names = params_mcmc_yaml['NAMES']

    distance_star = params_mcmc_yaml['DISTANCE_STAR']
    PIXSCALE_INS = params_mcmc_yaml['PIXSCALE_INS']

    sigma = params_mcmc_yaml['sigma']
    n_dim_mcmc = params_mcmc_yaml['N_DIM_MCMC']
    nwalkers = params_mcmc_yaml['NWALKERS']

    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    name_h5 = file_prefix + '_backend_file_mcmc'

    reader = backends.HDFBackend(os.path.join(mcmcresultdir, name_h5 + '.h5'))
    chain_flat = reader.get_chain(discard=burnin, thin=thin, flat=True)
    log_prob_samples_flat = reader.get_log_prob(discard=burnin,
                                                flat=True,
                                                thin=thin)

    if n_dim_mcmc == 11:
        ## change log and arccos values to physical
        chain_flat[:, 0] = np.exp(chain_flat[:, 0])
        chain_flat[:, 1] = np.exp(chain_flat[:, 1])
        chain_flat[:, 6] = np.degrees(np.arccos(chain_flat[:, 6]))
        chain_flat[:, 10] = np.exp(chain_flat[:, 10])

        ## change g1, g2 and alpha to percentage
        chain_flat[:, 3] = 100 * chain_flat[:, 3]
        chain_flat[:, 4] = 100 * chain_flat[:, 4]
        chain_flat[:, 5] = 100 * chain_flat[:, 5]

    if n_dim_mcmc == 13:
        ## change log values to physical
        chain_flat[:, 0] = np.exp(chain_flat[:, 0])
        chain_flat[:, 1] = np.exp(chain_flat[:, 1])
        chain_flat[:, 8] = np.degrees(np.arccos(chain_flat[:, 8]))
        chain_flat[:, 12] = np.exp(chain_flat[:, 12])

        ## change g1, g2, g3, alpha1 and alpha2 to percentage
        chain_flat[:, 3] = 100 * chain_flat[:, 3]
        chain_flat[:, 4] = 100 * chain_flat[:, 4]
        chain_flat[:, 5] = 100 * chain_flat[:, 5]
        chain_flat[:, 6] = 100 * chain_flat[:, 6]
        chain_flat[:, 7] = 100 * chain_flat[:, 7]

    samples_dict = dict()
    comments_dict = comments
    MLval_mcmc_val_mcmc_err_dict = dict()

    for i, key in enumerate(names[:n_dim_mcmc]):
        samples_dict[key] = chain_flat[:, i]

    for i, key in enumerate(names[n_dim_mcmc:]):
        samples_dict[key] = chain_flat[:, i] * 0.

    # measure of 6 other parameters:  right ascension, declination, and Kowalsky
    # (true_a, true_ecc, longnode, argperi)
    for j in range(chain_flat.shape[0]):
        r1_here = samples_dict['R1'][j]
        inc_here = samples_dict['inc'][j]
        pa_here = samples_dict['PA'][j]
        dx_here = samples_dict['dx'][j]
        dy_here = samples_dict['dy'][j]
        dAlpha, dDelta = offset_2_RA_dec(dx_here, dy_here, inc_here, pa_here,
                                         distance_star)

        samples_dict['RA'][j] = dAlpha
        samples_dict['Decl'][j] = dDelta

        semimajoraxis = convert.au_to_mas(r1_here, distance_star)
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
        quants = [15.9, 50., 84.1]
    if sigma == 2:
        quants = [2.3, 50., 97.77]
    if sigma == 3:
        quants = [0.1, 50., 99.9]

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
    hdr['BURNIN'] = burnin
    hdr['THIN'] = thin

    hdr['TOT_ITER'] = reader.iteration

    hdr['n_walker'] = nwalkers
    hdr['n_param'] = n_dim_mcmc

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
        BestModel
        BestModel_Conv
        BestModel_FM
        BestModel_Res

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

    quality_plot = params_mcmc_yaml['QUALITY_PLOT']
    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    band_name = params_mcmc_yaml['BAND_NAME']
    name_h5 = file_prefix + '_backend_file_mcmc'

    numbasis = [params_mcmc_yaml['KLMODE_NUMBER']]
    n_dim_mcmc = hdr['n_param']

    xcen = params_mcmc_yaml['xcen']
    ycen = params_mcmc_yaml['ycen']

    #Format the most likely values
    #generate the best model
    if n_dim_mcmc == 11:
        theta_ml = [
            np.log(hdr['R1_ML']),
            np.log(hdr['R2_ML']), hdr['Beta_ML'], 1 / 100. * hdr['g1_ML'],
            1 / 100. * hdr['g2_ML'], 1 / 100. * hdr['Alph1_ML'],
            np.cos(np.radians(hdr['inc_ML'])), hdr['PA_ML'], hdr['dx_ML'],
            hdr['dy_ML'],
            np.log(hdr['Norm_ML'])
        ]
    if n_dim_mcmc == 13:
        theta_ml = [
            np.log(hdr['R1_ML']),
            np.log(hdr['R2_ML']), hdr['Beta_ML'], 1 / 100. * hdr['g1_ML'],
            1 / 100. * hdr['g2_ML'], 1 / 100. * hdr['g3_ML'],
            1 / 100. * hdr['Alph1_ML'], 1 / 100. * hdr['Alph2_ML'],
            np.cos(np.radians(hdr['inc_ML'])), hdr['PA_ML'], hdr['dx_ML'],
            hdr['dy_ML'],
            np.log(1391)
            # np.log(hdr['Norm_ML'])
        ]

    psf = fits.getdata(os.path.join(DATADIR, file_prefix + '_SatSpotPSF.fits'))

    mask2generatedisk = fits.getdata(
        os.path.join(klipdir, file_prefix + '_mask2generatedisk.fits'))

    mask2generatedisk[np.where(mask2generatedisk == 0.)] = np.nan
    wheremask2generatedisk = (mask2generatedisk != mask2generatedisk)

    # load the raw data (necessary to create the DiskFM obj)
    # this is the only part different for SPHERE and GPI

    if params_mcmc_yaml['BAND_DIR'] == 'SPHERE_Hdata':
        #only for SPHERE
        datacube_sphere = fits.getdata(
            os.path.join(DATADIR, file_prefix + '_true_dataset.fits'))
        parangs_sphere = fits.getdata(
            os.path.join(DATADIR, file_prefix + '_true_parangs.fits'))

        size_datacube = datacube_sphere.shape
        centers_sphere = np.zeros((size_datacube[0], 2)) + [xcen, ycen]
        dataset = Instrument.GenericData(datacube_sphere,
                                         centers_sphere,
                                         parangs=parangs_sphere,
                                         wvs=None)
    else:
        #only for GPI
        filelist = sorted(glob.glob(os.path.join(DATADIR,
                                                 "*_distorcorr.fits")))

        # in the general case we can choose to
        # keep the files where the disk intersect the disk.
        # We can removed those if rm_file_disk_cross_satspots == 1
        rm_file_disk_spots = params_mcmc_yaml['RM_FILE_DISK_CROSS_SATSPOTS']
        if rm_file_disk_spots == 1:
            dataset_for_exclusion = GPI.GPIData(filelist, quiet=True)
            excluded_files = gpidiskpsf.check_satspots_disk_intersection(
                dataset_for_exclusion, params_mcmc_yaml, quiet=True)
            for excluded_filesi in excluded_files:
                if excluded_filesi in filelist:
                    filelist.remove(excluded_filesi)

        # load the bad slices in the psf header
        hdr_psf = fits.getheader(
            os.path.join(DATADIR, file_prefix + '_SatSpotPSF.fits'))

        excluded_slices = []
        if hdr_psf['N_BADSLI'] > 0:
            for badslice_i in range(hdr_psf['N_BADSLI']):
                excluded_slices.append(hdr_psf['BADSLI' +
                                               str(badslice_i).zfill(2)])

        # load the raw data without the bad slices
        dataset = GPI.GPIData(filelist, quiet=True, skipslices=excluded_slices)

        #collapse the data spectrally
        dataset.spectral_collapse(align_frames=True, numthreads=1)

    DIMENSION = dataset.input.shape[1]

    # load the data
    reduced_data = fits.getdata(
        os.path.join(klipdir, file_prefix + '-klipped-KLmodes-all.fits'))[
            0]  ### we take only the first KL mode

    # load the noise
    noise = fits.getdata(os.path.join(klipdir, file_prefix + '_noisemap.fits'))

    #generate the best model
    if n_dim_mcmc == 11:
        disk_ml = call_gen_disk_2g(theta_ml)
    if n_dim_mcmc == 13:
        disk_ml = call_gen_disk_3g(theta_ml)

    fits.writeto(os.path.join(mcmcresultdir, name_h5 + '_BestModel.fits'),
                 disk_ml,
                 header=hdr,
                 overwrite=True)

    #convolve by the PSF
    disk_ml_convolved = convolve(disk_ml, psf, boundary='wrap')

    fits.writeto(os.path.join(mcmcresultdir, name_h5 + '_BestModel_Conv.fits'),
                 disk_ml_convolved,
                 header=hdr,
                 overwrite=True)

    # load the KL numbers
    diskobj = DiskFM(dataset.input.shape,
                     numbasis,
                     dataset,
                     disk_ml_convolved,
                     basis_filename=os.path.join(klipdir,
                                                 file_prefix + '_klbasis.h5'),
                     load_from_basis=True)

    #do the FM
    diskobj.update_disk(disk_ml_convolved)
    disk_ml_FM = diskobj.fm_parallelized()[0]
    ### we take only the first KL modemode

    fits.writeto(os.path.join(mcmcresultdir, name_h5 + '_BestModel_FM.fits'),
                 disk_ml_FM,
                 header=hdr,
                 overwrite=True)

    fits.writeto(os.path.join(mcmcresultdir, name_h5 + '_BestModel_Res.fits'),
                 np.abs(reduced_data - disk_ml_FM),
                 header=hdr,
                 overwrite=True)

    #Mesaure the residuals
    residuals = np.abs(reduced_data - disk_ml_FM)
    snr_residuals = np.abs(reduced_data - disk_ml_FM) / noise

    #Set the colormap
    vmin = 0.3 * np.min(disk_ml_FM)
    vmax = 0.9 * np.max(disk_ml_FM)

    #The convolved model
    if file_prefix == 'Hband_hd48524_fake':
        # We are showing only here a white line the extension of the minimization line

        mask_disk_int = make_disk_mask(
            disk_ml_FM.shape[0],
            params_mcmc_yaml['pa_init'],
            params_mcmc_yaml['inc_init'],
            convert.au_to_pix(40, PIXSCALE_INS, DISTANCE_STAR),
            convert.au_to_pix(41, PIXSCALE_INS, DISTANCE_STAR),
            xcen=xcen,
            ycen=ycen)

        mask_disk_ext = make_disk_mask(
            disk_ml_FM.shape[0],
            params_mcmc_yaml['pa_init'],
            params_mcmc_yaml['inc_init'],
            convert.au_to_pix(129, PIXSCALE_INS, DISTANCE_STAR),
            convert.au_to_pix(130, PIXSCALE_INS, DISTANCE_STAR),
            xcen=xcen,
            ycen=ycen)

        mask_disk_int[np.where(mask_disk_int == 0.)] = np.nan
        mask_disk_ext[np.where(mask_disk_ext == 0.)] = np.nan

        disk_ml_FM = disk_ml_FM * mask_disk_int * mask_disk_ext

    dim_crop_image = int(
        4 * convert.au_to_pix(102, PIXSCALE_INS, DISTANCE_STAR) // 2)

    disk_ml_crop = crop_center(disk_ml, dim_crop_image)
    disk_ml_convolved_crop = crop_center(disk_ml_convolved, dim_crop_image)
    disk_ml_FM_crop = crop_center(disk_ml_FM, dim_crop_image)

    reduced_data_crop = crop_center(reduced_data, dim_crop_image)
    residuals_crop = crop_center(residuals, dim_crop_image)
    snr_residuals_crop = crop_center(snr_residuals, dim_crop_image)

    caracsize = 40 * quality_plot / 2.

    fig = plt.figure(figsize=(6.4 * 2 * quality_plot, 4.8 * 2 * quality_plot))
    #The data
    ax1 = fig.add_subplot(235)
    cax = plt.imshow(reduced_data_crop + 0.1,
                     origin='lower',
                     vmin=int(np.round(vmin)),
                     vmax=int(np.round(vmax)),
                     cmap=plt.cm.get_cmap('viridis'))
    ax1.set_title("Original Data", fontsize=caracsize, pad=caracsize / 3.)
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=caracsize * 3 / 4.)
    plt.axis('off')

    #The residuals
    ax1 = fig.add_subplot(233)
    cax = plt.imshow(residuals_crop,
                     origin='lower',
                     vmin=0,
                     vmax=int(np.round(vmax) // 2),
                     cmap=plt.cm.get_cmap('viridis'))
    ax1.set_title("Residuals", fontsize=caracsize, pad=caracsize / 3.)

    # make the colobar ticks integer only for gpi
    if params_mcmc_yaml['BAND_DIR'] != 'SPHERE_Hdata':
        tick_int = list(np.arange(int(np.round(vmax) // 2) + 1))
        tick_int_st = [str(i) for i in tick_int]
        cbar = fig.colorbar(cax, ticks=tick_int, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=caracsize * 3 / 4.)
        cbar.ax.set_yticklabels(tick_int_st)
    else:
        cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=caracsize * 3 / 4.)
    plt.axis('off')

    #The SNR of the residuals
    ax1 = fig.add_subplot(236)
    cax = plt.imshow(snr_residuals_crop,
                     origin='lower',
                     vmin=0,
                     vmax=2,
                     cmap=plt.cm.get_cmap('viridis'))
    ax1.set_title("SNR Residuals", fontsize=caracsize, pad=caracsize / 3.)
    cbar = fig.colorbar(cax, ticks=[0, 1, 2], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=caracsize * 3 / 4.)
    cbar.ax.set_yticklabels(['0', '1', '2'])
    plt.axis('off')

    # The model
    ax1 = fig.add_subplot(231)
    cax = plt.imshow(disk_ml_crop,
                     origin='lower',
                     vmin=-2,
                     vmax=int(np.round(np.max(disk_ml_crop) / 1.5)),
                     cmap=plt.cm.get_cmap('plasma'))
    ax1.set_title("Best Model", fontsize=caracsize, pad=caracsize / 3.)
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=caracsize * 3 / 4.)
    plt.axis('off')

    rect = Rectangle((9.5, 9.5),
                     psf.shape[0],
                     psf.shape[1],
                     edgecolor='white',
                     facecolor='none',
                     linewidth=2)

    disk_ml_convolved_crop[10:10 + psf.shape[0], 10:10 +
                           psf.shape[1]] = 2 * vmax * psf

    ax1 = fig.add_subplot(234)
    cax = plt.imshow(disk_ml_convolved_crop,
                     origin='lower',
                     vmin=int(np.round(vmin)),
                     vmax=int(np.round(vmax * 2)),
                     cmap=plt.cm.get_cmap('viridis'))
    ax1.add_patch(rect)

    ax1.set_title("Model Convolved", fontsize=caracsize, pad=caracsize / 3.)
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=caracsize * 3 / 4.)
    plt.axis('off')

    #The FM convolved model
    ax1 = fig.add_subplot(232)
    cax = plt.imshow(disk_ml_FM_crop,
                     origin='lower',
                     vmin=int(np.round(vmin)),
                     vmax=int(np.round(vmax)),
                     cmap=plt.cm.get_cmap('viridis'))
    ax1.set_title("Model Convolved + FM",
                  fontsize=caracsize,
                  pad=caracsize / 3.)
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=caracsize * 3 / 4.)
    plt.axis('off')

    fig.subplots_adjust(hspace=-0.4, wspace=0.2)

    fig.suptitle(band_name + ': Best Model and Residuals',
                 fontsize=5 / 4. * caracsize,
                 y=0.985)

    fig.tight_layout()

    plt.savefig(os.path.join(mcmcresultdir, name_h5 + '_BestModel_Plot.jpg'))


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

    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    distance_star = params_mcmc_yaml['DISTANCE_STAR']

    name_h5 = file_prefix + '_backend_file_mcmc'

    reader = backends.HDFBackend(os.path.join(mcmcresultdir, name_h5 + '.h5'))

    f1 = open(
        os.path.join(mcmcresultdir, name_h5 + '_fit_geometrical_params.txt'),
        'w+')
    f1.write("\n'{0} / {1}".format(reader.iteration, reader.iteration * 192))
    f1.write("\n")

    to_print_str = 'R1'
    to_print = [
        hdr[to_print_str + '_MC'], hdr[to_print_str + '_M'],
        hdr[to_print_str + '_P']
    ]
    to_print = convert.au_to_mas(to_print, distance_star)
    f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(to_print[0], to_print[1],
                                                  to_print[2]))

    to_print_str = 'R2'
    to_print = [
        hdr[to_print_str + '_MC'], hdr[to_print_str + '_M'],
        hdr[to_print_str + '_P']
    ]
    to_print = convert.au_to_mas(to_print, distance_star)
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
    to_print = convert.au_to_mas(to_print, distance_star)
    f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(to_print[0], to_print[1],
                                                  to_print[2]))

    to_print_str = 'dy'
    to_print = [
        hdr[to_print_str + '_MC'], hdr[to_print_str + '_M'],
        hdr[to_print_str + '_P']
    ]
    to_print = convert.au_to_mas(to_print, distance_star)
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
        # str_yalm = 'SPHERE_Hband_3g_MCMC.yaml'
        str_yalm = 'GPI_K2band_MCMC.yaml'
    else:
        str_yalm = sys.argv[1]

    with open(os.path.join('initialization_files', str_yalm),
              'r') as yaml_file:
        params_mcmc_yaml = yaml.load(yaml_file)

    # test on which machine I am
    if socket.gethostname() == 'MT-101942':
        basedir = '/Users/jmazoyer/Dropbox/ExchangeFolder/data_python/Aurora/'
    else:
        basedir = '/home/jmazoyer/data_python/Aurora/'

    DATADIR = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'])
    klipdir = os.path.join(DATADIR, 'klip_fm_files')
    mcmcresultdir = os.path.join(DATADIR, 'results_MCMC')

    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    name_h5 = file_prefix + '_backend_file_mcmc'

    if not os.path.isfile(os.path.join(mcmcresultdir, name_h5 + '.h5')):
        raise ValueError("the mcmc h5 file does not exist")

    # Plot the chain values
    make_chain_plot(params_mcmc_yaml)

    # # Plot the PDFs
    make_corner_plot(params_mcmc_yaml)

    # measure the best likelyhood model and excract MCMC errors
    hdr = create_header(params_mcmc_yaml)

    # save the fits, plot the model and residuals
    best_model_plot(params_mcmc_yaml, hdr)

    # print the values to put in excel sheet easily
    print_geometry_parameter(params_mcmc_yaml, hdr)
