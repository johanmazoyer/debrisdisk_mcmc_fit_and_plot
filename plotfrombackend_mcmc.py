####### This is the MCMC plotting code for HR 4796 data #######

import sys, os, glob
import socket
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
from kowalsky import kowalsky

import matplotlib
matplotlib.use(
    'Agg'
)  ### I dont use matplotlib but I notice there is a conflict when I import matplotlib with pyklip if I don't use this line

import corner
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.font_manager import FontProperties

import astro_unit_conversion as convert

# from emcee import EnsembleSampler
from emcee import backends

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

quality_plot = 4

DIMENSION = 281  # pixel
distance = 72  #pc

pixscale = 0.01414  #GPI's pixel scale

########################################################
########################################################
#### Fonctions extraite de anadisk
########################################################
########################################################


def integrand_dxdy_2g(xp, yp_dy2, yp2, zp, zp2, zpsi_dx, zpci, R1, R2, beta,
                      a_r, g1, g1_2, g2, g2_2, alpha, ci, si, maxe, dx, dy, k):

    # compute the scattering integrand
    # see analytic-disk.nb

    xx = (xp * ci + zpsi_dx)

    d1 = mt.sqrt((yp_dy2 + xx * xx))

    if (d1 < R1 or d1 > R2):
        return 0.0

    d2 = xp * xp + yp2 + zp2

    #The line of sight scattering angle
    cos_phi = xp / mt.sqrt(d2)
    # phi=np.arccos(cos_phi)

    #Henyey Greenstein function
    hg1 = k * alpha * (1. - g1_2) / (1. + g1_2 - (2 * g1 * cos_phi))**1.5
    hg2 = k * (1 - alpha) * (1. - g2_2) / (1. + g2_2 - (2 * g2 * cos_phi))**1.5

    hg = hg1 + hg2

    #Radial power low r propto -beta
    int1 = hg * (R1 / d1)**beta

    #The scale height function
    zz = (zpci - xp * si)
    hh = (a_r * d1)
    expo = zz * zz / (hh * hh)

    # if expo > 2*maxe:   # cut off exponential after 28 e-foldings (~ 1E-06)
    #     return 0.0

    int2 = np.exp(0.5 * expo)
    int3 = int2 * d2

    return int1 / int3


def gen_disk_dxdy_2g(dim,
                     R1=74.42,
                     R2=82.45,
                     beta=1.0,
                     aspect_ratio=0.1,
                     g1=0.6,
                     g2=-0.6,
                     alpha=0.7,
                     inc=76.49,
                     pa=30,
                     distance=72.8,
                     sampling=1,
                     mask=None,
                     dx=0,
                     dy=0.,
                     pixscale=pixscale):

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

    max_fov = dim / 2. * pixscale  #maximum radial distance in AU from the center to the edge
    npts = int(np.floor(dim / sampling))
    xsize = max_fov * distance  #maximum radial distance in AU from the center to the edge

    #The coordinate system here [x,y,z] is defined :
    # +ve x is the line of sight
    # +ve y is going right from the center
    # +ve z is going up from the center

    # y = np.linspace(0,xsize,num=npts/2)
    y = np.linspace(-xsize, xsize, num=npts)
    z = np.linspace(-xsize, xsize, num=npts)

    #Only need to compute half the image
    # image =np.zeros((npts,npts/2+1))
    image = np.zeros((npts, npts))

    #Some things we can precompute ahead of time
    maxe = mt.log(np.finfo('f').max)  #The log of the machine precision

    #Inclination Calculations
    incl = np.radians(90 - inc)
    ci = mt.cos(incl)  #Cosine of inclination
    si = mt.sin(incl)  #Sine of inclination

    #Position angle calculations
    pa_rad = np.radians(90 - pa)  #The position angle in radians
    cos_pa = mt.cos(pa_rad)  #Calculate these ahead of time
    sin_pa = mt.sin(pa_rad)

    #HG g value squared
    g1_2 = g1 * g1  #First HG g squared
    g2_2 = g2 * g2  #Second HG g squared
    #Constant for HG function
    k = 1. / (4 * np.pi)

    #The aspect ratio
    a_r = aspect_ratio

    #Henyey Greenstein function at 90
    hg1_90 = k * alpha * (1. - g1_2) / (1. + g1_2)**1.5
    hg2_90 = k * (1 - alpha) * (1. - g2_2) / (1. + g2_2)**1.5

    hg_90 = hg1_90 + hg2_90

    #If there's no mask then calculate for the full image
    if len(np.shape(mask)) < 2:

        for i, yp in enumerate(y):
            for j, zp in enumerate(z):

                #This rotates the coordinates in the image frame
                yy = yp * cos_pa - zp * sin_pa  #Rotate the y coordinate by the PA
                zz = yp * sin_pa + zp * cos_pa  #Rotate the z coordinate by the PA

                #The distance from the center (in each coordinate) squared
                y2 = yy * yy
                z2 = zz * zz

                #This rotates the coordinates in and out of the sky
                zpci = zz * ci  #Rotate the z coordinate by the inclination.
                zpsi = zz * si
                #Subtract the offset
                zpsi_dx = zpsi - dx

                #The distance from the offset squared
                yy_dy = yy - dy
                yy_dy2 = yy_dy * yy_dy

                image[j, i] = quad(integrand_dxdy_2g,
                                   -R2,
                                   R2,
                                   epsrel=0.5e-3,
                                   limit=75,
                                   args=(yy_dy2, y2, zp, z2, zpsi_dx, zpci, R1,
                                         R2, beta, a_r, g1, g1_2, g2, g2_2,
                                         alpha, ci, si, maxe, dx, dy, k))[0]

    #If there is a mask then don't calculate disk there
    else:
        hmask = mask
        # hmask = mask[:,140:] #Use only half the mask

        for i, yp in enumerate(y):
            for j, zp in enumerate(z):

                # if hmask[j,npts/2+i]: #This assumes that the input mask has is the same size as the desired image (i.e. ~ size / sampling)
                if hmask[j, i]:

                    image[j, i] = 0.  #np.nan

                else:

                    #This rotates the coordinates in the image frame
                    yy = yp * cos_pa - zp * sin_pa  #Rotate the y coordinate by the PA
                    zz = yp * sin_pa + zp * cos_pa  #Rotate the z coordinate by the PA

                    #The distance from the center (in each coordinate) squared
                    y2 = yy * yy
                    z2 = zz * zz

                    #This rotates the coordinates in and out of the sky
                    zpci = zz * ci  #Rotate the z coordinate by the inclination.
                    zpsi = zz * si
                    #Subtract the offset
                    zpsi_dx = zpsi - dx

                    #The distance from the offset squared
                    yy_dy = yy - dy
                    yy_dy2 = yy_dy * yy_dy

                    image[j, i] = quad(integrand_dxdy_2g,
                                       -R2,
                                       R2,
                                       epsrel=0.5e-3,
                                       limit=75,
                                       args=(yy_dy2, y2, zp, z2, zpsi_dx, zpci,
                                             R1, R2, beta, a_r, g1, g1_2, g2,
                                             g2_2, alpha, ci, si, maxe, dx, dy,
                                             k))[0]

    # print("Running time: ", datetime.now()-starttime)

    # # normalize the HG function by the width
    image = image / a_r

    # normalize the HG function at the PA
    image = image / hg_90

    return image


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
                                            distance=distance,
                                            dx=dx,
                                            dy=dy,
                                            mask=wheremask2generatedisk,
                                            pixscale=pixscale)  #+ offset

    return model


########################################################
def crop_center(img, crop):
    y, x = img.shape
    startx = (x - 1) // 2 - crop // 2
    starty = (y - 1) // 2 - crop // 2
    return img[starty:starty + crop, startx:startx + crop]


########################################################
def offset_2_RA_dec(dx, dy, inclination, principal_angle, dist_star):
    """ right ascension and declination of the ellipse centre with respect to the star
        location from the offset in AU in the disk plane define by the max disk code

    Args:
        dx: offsetx of the star in AU in the disk plane define by the max disk code
            au, + -> NW offset disk plane Minor Axis
        dy: offsety of the star in AU in the disk plane define by the max disk code
            au, + -> SW offset disk plane Major Axis
        inclination: inclination in degrees
        principal_angle: prinipal angle in degrees
        dist_star: distance to the star

    Returns:
        [right ascension, declination]
    """

    dx_disk_mas = convert.au_to_mas(dx * np.cos(np.radians(inclination)),
                                    dist_star)
    dy_disk_mas = convert.au_to_mas(-dy, dist_star)

    dx_sky = np.cos(np.radians(principal_angle)) * dx_disk_mas - np.sin(
        np.radians(principal_angle)) * dy_disk_mas
    dy_sky = np.sin(np.radians(principal_angle)) * dx_disk_mas + np.cos(
        np.radians(principal_angle)) * dy_disk_mas

    dAlpha = -dx_sky
    dDelta = dy_sky

    return dAlpha, dDelta


def make_chain_plot(reader, burnin=0, thin=1, labels=[''], quality_plot=4):

    """ make_chain_plot reading the .h5 file from emcee

    Args:
        reader: a reader obtained from backends.HDFBackend
        burnin: the number of iteration you want to remove from the chains
        thin: bin the iterations
        labels: name of the parameters
        quality_plot: quality of the plot, integers
        figpath: where to save the figure
        file_prefix = file prefix to save the figure

    Returns:
        None
    """

    chain = reader.get_chain(discard=0, thin=1)
    ndim = chain.shape[2]
    nwalkers = chain.shape[1]

    ## change log and arccos values to physical
    chain[:, :, 0] = np.exp(chain[:, :, 0])
    chain[:, :, 1] = np.exp(chain[:, :, 1])
    chain[:, :, 6] = np.degrees(np.arccos(chain[:, :, 6]))
    chain[:, :, 10] = np.exp(chain[:, :, 10])

    ## change g1, g2 and alpha to percentage
    chain[:, :, 3] = 100 * chain[:, :, 3]
    chain[:, :, 4] = 100 * chain[:, :, 4]
    chain[:, :, 5] = 100 * chain[:, :, 5]

    _, axarr = plt.subplots(ndim,
                            sharex=True,
                            figsize=(6.4 * quality_plot, 4.8 * quality_plot))

    for i in range(ndim):
        axarr[i].set_ylabel(labels[i], fontsize=5 * quality_plot)
        axarr[i].tick_params(axis='y', labelsize=4 * quality_plot)

        for j in range(nwalkers):
            axarr[i].plot(chain[:, j, i], linewidth=quality_plot)

        axarr[i].axvline(x=burnin, color='black', linewidth=1.5 * quality_plot)

    axarr[ndim - 1].tick_params(axis='x', labelsize=6 * quality_plot)
    axarr[ndim - 1].set_xlabel('Iterations', fontsize=10 * quality_plot)

    plt.savefig( + name_h5 + '_chains.jpg')


def make_corner_plot(reader,
                     burnin=0,
                     thin=1,
                     labels=[''],
                     quality_plot=4,
                     sigma=1,
                     figpath='',
                     file_prefix=''):

    """ make corner plot reading the .h5 file from emcee

    Args:
        reader: a reader obtained from backends.HDFBackend
        burnin: the number of iteration you want to remove from the chains
        thin: bin the iterations
        labels: name of the parameters
        quality_plot: quality of the plot, integers
        sigma: the number of sigma you want plotted as vertical bars on the corner
               plot (1 2, or 3)
        figpath: where to save the figure
        file_prefix = file prefix to save the figure


    Returns:
        None
    """


    chain = reader.get_chain(discard=0, thin=1)

    ndim = chain.shape[2]

    chain[:, :, 0] = np.exp(chain[:, :, 0])
    chain[:, :, 1] = np.exp(chain[:, :, 1])
    chain[:, :, 6] = np.degrees(np.arccos(chain[:, :, 6]))
    chain[:, :, 10] = np.exp(chain[:, :, 10])

    ## change g1, g2 and alpha to percentage
    chain[:, :, 3] = 100 * chain[:, :, 3]
    chain[:, :, 4] = 100 * chain[:, :, 4]
    chain[:, :, 5] = 100 * chain[:, :, 5]

    samples = chain[:, :].reshape(-1, ndim)

    matplotlib.rcParams['axes.labelsize'] = 19
    matplotlib.rcParams['axes.titlesize'] = 14

    matplotlib.rcParams['xtick.labelsize'] = 13
    matplotlib.rcParams['ytick.labelsize'] = 13

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
    fig = corner.corner(samples,
                        labels=labels,
                        quantiles=quants,
                        show_titles=True,
                        plot_datapoints=False,
                        verbose=False)  # levels=(1-np.exp(-0.5),) , ))

    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0)

    fig.gca().annotate(
        "GPI H band, {0:,} iterations (192 walkers): {1:,} models".format(
            reader.iteration, reader.iteration * 192),
        xy=(0.55, 0.99),
        xycoords="figure fraction",
        xytext=(-20, -10),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=44)

    plt.savefig(figpath + file_prefix + '_pdfs.pdf')


def create_header(reader,
                  burnin=0,
                  thin=1,
                  names=[''],
                  dist_star=100,
                  pixscale=0.01414):

    chain = reader.get_chain(discard=burnin, thin=thin)
    log_prob_samples_flat = reader.get_log_prob(discard=burnin,
                                                flat=True,
                                                thin=thin)

    samples = chain[:, :].reshape(-1, chain.shape[2])

    ndim = chain.shape[2]
    nwalkers = chain.shape[1]

    samples_dict = dict()
    comments_dict = dict()
    MLval_mcmc_val_mcmc_err_dict = dict()

    for i, key in enumerate(names[:ndim]):
        comments_dict[key] = comments[i]
        samples_dict[key] = samples[:, i]

    for i, key in enumerate(names[ndim:]):
        comments_dict[key] = comments[i]
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
                                         dist_star)

        samples_dict['RA'][j] = dAlpha
        samples_dict['Decl'][j] = dDelta

        semimajoraxis = convert.au_to_mas(r1_here, dist_star)
        ecc = np.sin(np.radians(inc_here))

        true_a, true_ecc, argperi, inc, longnode = kowalsky(
            semimajoraxis, ecc, pa_here, dAlpha, dDelta)

        samples_dict['a'][j] = true_a
        samples_dict['ecc'][j] = true_ecc
        samples_dict['Omega'][j] = longnode
        samples_dict['ARGPE'][j] = argperi

    wheremin = np.where(log_prob_samples_flat == np.max(log_prob_samples_flat))
    wheremin0 = np.array(wheremin).flatten()[0]

    for key in samples_dict.keys():
        MLval_mcmc_val_mcmc_err_dict[key] = np.zeros(4)

        percent = np.percentile(samples[:, i], [15.9, 50, 84.1])
        # print(key,samples_dict[key].shape)
        MLval_mcmc_val_mcmc_err_dict[key][0] = samples_dict[key][wheremin0]
        MLval_mcmc_val_mcmc_err_dict[key][1] = percent[1]
        MLval_mcmc_val_mcmc_err_dict[key][2] = percent[0] - percent[1]
        MLval_mcmc_val_mcmc_err_dict[key][3] = percent[2] - percent[1]

    MLval_mcmc_val_mcmc_err_dict['RApix'] = convert.mas_to_pix(
        MLval_mcmc_val_mcmc_err_dict['RA'], pixscale)
    MLval_mcmc_val_mcmc_err_dict['Depix'] = convert.mas_to_pix(
        MLval_mcmc_val_mcmc_err_dict['Decl'], pixscale)

    print(" ")

    for key in samples_dict.keys():
        print(key+' ML: {0:.3f}, MCMC {1:.3f}, -/+1sig: {2:.3f}/+{3:.3f} '.format(MLval_mcmc_val_mcmc_err_dict[key][0], MLval_mcmc_val_mcmc_err_dict[key][1], \
            MLval_mcmc_val_mcmc_err_dict[key][2],MLval_mcmc_val_mcmc_err_dict[key][3])+ comments_dict[key])

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
    hdr['n_param'] = ndim

    hdr['MAX_LH'] = (np.max(log_prob_samples_flat),
                     'Max likelyhood, obtained for the ML parameters')

    for key in samples_dict.keys():
        hdr[key + '_ML'] = (MLval_mcmc_val_mcmc_err_dict[key][0],
                            comments_dict[key])
        hdr[key + '_MC'] = MLval_mcmc_val_mcmc_err_dict[key][1]
        hdr[key + '_M'] = MLval_mcmc_val_mcmc_err_dict[key][2]
        hdr[key + '_P'] = MLval_mcmc_val_mcmc_err_dict[key][3]


    return hdr

def best_model_plot(hdr,
                    OWA=3,
                    mov_here=8,
                    KLMODE=[3],
                    datadir='.',
                    file_prefix='',
                    removed_slices=None,
                    quality_plot=4,
                    band_name=''):
    #Format the most likely values
    theta_ml = [
        np.log(hdr['R1_ML']),
        np.log(hdr['R2_ML']), hdr['R1_ML'], 1 / 100. * hdr['g1_ML'],
        1 / 100. * hdr['g2_ML'], 1 / 100. * hdr['Alpha_ML'],
        np.cos(np.radians(hdr['inc_ML'])), hdr['PA_ML'], hdr['dx_ML'],
        hdr['dy_ML'],
        np.log(hdr['N_ML'])
    ]

    klipdir = datadir + 'klip_fm_files/'
    mcmcresultdir = datadir + 'results_MCMC/'

    psf = fits.getdata(datadir + file_prefix + '_SatSpotPSF.fits')

    mask2generatedisk = fits.getdata(klipdir + file_prefix +
                                     '_mask2generatedisk.fits')
    mask2minimize = fits.getdata(klipdir + file_prefix + '_mask2minimize.fits')

    mask2generatedisk[np.where(mask2generatedisk == 0.)] = np.nan
    wheremask2generatedisk = (mask2generatedisk != mask2generatedisk)

    mask2minimize[np.where(mask2minimize == 0.)] = np.nan
    wheremask2minimize = (mask2minimize != mask2minimize)

    # load the raw data
    filelist = glob.glob(datadir + "*distorcorr.fits")

    dataset = GPI.GPIData(filelist, quiet=True, skipslices=removed_slices)

    dataset.OWA = OWA
    #collapse the data spectrally
    dataset.spectral_collapse(align_frames=True, numthreads=1)

    # load the data
    reduced_data = fits.getdata(klipdir + file_prefix +
                                '_Measurewithparallelized-KLmodes-all.fits')[
                                    0]  ### we take only the first KL mode

    # load the noise
    noise = fits.getdata(klipdir + file_prefix + '_noisemap.fits')

    #generate the best model
    disk_ml = call_gen_disk_2g(theta_ml, wheremask2generatedisk)

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

    if socket.gethostname() != 'MT-101942':
        # load the KL numbers
        diskobj = DiskFM(dataset.input.shape,
                         KLMODE,
                         dataset,
                         disk_ml_convolved,
                         basis_filename=klipdir + file_prefix + '_basis.pkl',
                         save_basis=False,
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

    reduced_data_crop = crop_center(reduced_data, 196)
    disk_ml_FM_crop = crop_center(disk_ml_FM, 196)
    disk_ml_convolved_crop = crop_center(disk_ml_convolved, 196)
    noise_crop = crop_center(noise, 196)
    disk_ml_crop = crop_center(disk_ml, 196)

    caracsize = 40 * quality_plot / 2

    fig = plt.figure(figsize=(6.4 * 2 * quality_plot, 4.8 * 2 * quality_plot))
    #The data
    ax1 = fig.add_subplot(235)
    # cax = plt.imshow(reduced_data_crop+ vmin +0.1, origin='lower', norm=LogNorm(),vmin=vmin, vmax=vmax, cmap=plt.cm.get_cmap('summer'))
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
    # cax = plt.imshow(np.abs(reduced_data_crop-disk_ml_FM_crop) + vmin +0.1, origin='lower', norm=LogNorm(),vmin=vmin, vmax=vmax, cmap=plt.cm.get_cmap('summer'))
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
    # cax = plt.imshow(disk_ml_convolved_crop + vmin +0.1, origin='lower', norm=LogNorm(),vmin=vmin, vmax=vmax, cmap=plt.cm.get_cmap('summer'))
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
    # cax = plt.imshow(disk_ml_FM_crop + vmin +0.1, origin='lower', norm=LogNorm(),vmin=vmin, vmax=vmax, cmap=plt.cm.get_cmap('summer'))
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

    fig.suptitle(band_name+': Best Model and Residuals',
                 fontsize=5 / 4. * caracsize,
                 y=0.985)

    fig.tight_layout()

    plt.savefig(mcmcresultdir + name_h5 + '_PlotBestModel.jpg')


dist_star = 72.
pixscale = 0.01414  #GPI's pixel scale

burnin = 9500
thin = 1

OWA = 98
mov_here = 8
KLMODE = [3]
removed_slices = None

labels = [
    "R1[AU]", "R2[AU]", r"$\beta$", "g1[%]", "g2[%]", r"$\alpha$[%]",
    r"$i[^{\circ}]$", r"$pa[^{\circ}]$", 'dx[AU]', 'dy[AU]', "N[ADU]"
]

names = [
    "R1", "R2", "Beta", "g1", "g2", "Alpha", "inc", "PA", 'dx', 'dy', "N",
    "RA", "Decl", "RApix", "Depix", "a", "ecc", "Omega", "ARGPE"
]

comments = [
    'AU, inner radius', 'AU, outer radius', 'radial power law',
    '%, 1st HG param', '%, 2nd HG param', '%, relative HG weight',
    'degree, inclination', 'degree, principal angle',
    'au, + -> NW offset disk plane Minor Axis',
    'au, + -> SW offset disk plane Major Axis', 'ADU, normalisation',
    'mas, ->E right ascension (dalpha in Milli+2017)',
    'mas, ->N declination (ddelta in Milli+2017)', 'pix, ->E right ascension',
    'pix, ->N declination', 'mas, deproj. (true) semi major axis (Kowalsky)',
    'deproj. (true) eccentricity (Kowalsky)',
    'deg, longitude of the ascending node (Kowalsky)',
    'deg, argument of pericentre (Kowalsky)'
]

if socket.gethostname() == 'MT-101942':
    datadir = '/Users/jmazoyer/Dropbox/ExchangeFolder/data_python/Aurora/160318_H_Spec/'
else:
    datadir = '/home/jmazoyer/data_python/Aurora/160318_H_Spec/'

file_prefix = 'Hband_hr4796_klipFM_MaxiNoise'

band_name = 'GPI H band'

klipdir = datadir + 'klip_fm_files/'
distutils.dir_util.mkpath(klipdir)
mcmcresultdir = datadir + 'results_MCMC/'
distutils.dir_util.mkpath(mcmcresultdir)

# read the backend in h5

name_h5 = file_prefix + 'backend_file_mcmc'
filename = mcmcresultdir + name_h5 + '.h5'

reader = backends.HDFBackend(filename)
tau = reader.get_autocorr_time(tol=0)

print("")
print("")
print(name_h5)
print("# of iteration in the backend chain initially: {0}".format(
    reader.iteration))
print("Max Tau times 50: {0}".format(50 * np.max(tau)))
print("")

chain = reader.get_chain(discard=burnin, thin=thin)
log_prob_samples_flat = reader.get_log_prob(discard=burnin,
                                            flat=True,
                                            thin=thin)

print(np.max(log_prob_samples_flat))

print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("chain shape: {0}".format(chain.shape))

#######################################################
#################### Make plot 1  #####################
#######################################################
#######################################################
# Plot the chain values

# make_chain_plot(reader,
#                 burnin=burnin,
#                 thin=thin,
#                 labels=labels,
#                 quality_plot=quality_plot,
#                 figpath = mcmcresultdir,
#                 file_prefix=name_h5)

# ####################################################################
# ### #Plot the PDFs
# ####################################################################

# make_corner_plot(reader,
                #  burnin=burnin,
                #  thin=thin,
                #  labels=labels,
                #  quality_plot=quality_plot,
                #  sigma=1,
                #  figpath=mcmcresultdir,
                #  file_prefix=name_h5)

####################################################################
### Best likelyhood model and MCMC errors
####################################################################

hdr = create_header(reader,
                    burnin=burnin,
                    thin=thin,
                    names=names,
                    dist_star=dist_star,
                    pixscale=pixscale)

#######################################################
#### save the fits, plot the model and residuals
#######################################################

best_model_plot(hdr,
                OWA=OWA,
                mov_here=mov_here,
                KLMODE=KLMODE,
                datadir=datadir,
                file_prefix=file_prefix,
                removed_slices=removed_slices,
                quality_plot=quality_plot,
                band_name=band_name)



# f1=open(mcmcresultdir+name_h5+'mcmcfit_geometrical_params.txt', 'w+')
# f1.write("\n'{0} / {1}".format(reader.iteration, reader.iteration*192))
# f1.write("\n")

# r1_mas =  convert.au_to_mas(MLval_mcmc_val_mcmc_err_dict['R1'], dist_star)

# names=["R1","R2","Beta", "g1","g2","Alpha", "inc", "PA",'dx', 'dy',
#         "N", "RA","Decl","RApix","Depix","a","ecc","Omega", "ARGPE" ]

# f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(r1_mas[1],r1_mas[2],r1_mas[3]))
# f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(MLval_mcmc_val_mcmc_err[1,1]/dist_star*1000, MLval_mcmc_val_mcmc_err[1,2]/dist_star*1000,MLval_mcmc_val_mcmc_err[1,3]/dist_star*1000))
# f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(MLval_mcmc_val_mcmc_err[7,1], MLval_mcmc_val_mcmc_err[7,2],MLval_mcmc_val_mcmc_err[7,3]))

# f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(MLval_mcmc_val_mcmc_err[11,1], MLval_mcmc_val_mcmc_err[11,2],MLval_mcmc_val_mcmc_err[11,3]))
# f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(MLval_mcmc_val_mcmc_err[12,1], MLval_mcmc_val_mcmc_err[12,2],MLval_mcmc_val_mcmc_err[12,3]))

# f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(MLval_mcmc_val_mcmc_err[8,1]/dist_star*1000, MLval_mcmc_val_mcmc_err[8,2]/dist_star*1000,MLval_mcmc_val_mcmc_err[8,3]/dist_star*1000))
# f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(MLval_mcmc_val_mcmc_err[9,1]/dist_star*1000, MLval_mcmc_val_mcmc_err[9,2]/dist_star*1000,MLval_mcmc_val_mcmc_err[9,3]/dist_star*1000))

# f1.write("\n")
# f1.write("\n")

# f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(MLval_mcmc_val_mcmc_err[13,1], MLval_mcmc_val_mcmc_err[13,2],MLval_mcmc_val_mcmc_err[13,3]))
# f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(MLval_mcmc_val_mcmc_err[14,1], MLval_mcmc_val_mcmc_err[14,2],MLval_mcmc_val_mcmc_err[14,3]))
# f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(MLval_mcmc_val_mcmc_err[6,1], MLval_mcmc_val_mcmc_err[6,2],MLval_mcmc_val_mcmc_err[6,3]))
# f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(MLval_mcmc_val_mcmc_err[15,1], MLval_mcmc_val_mcmc_err[15,2],MLval_mcmc_val_mcmc_err[15,3]))
# f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(MLval_mcmc_val_mcmc_err[16,1], MLval_mcmc_val_mcmc_err[16,2],MLval_mcmc_val_mcmc_err[16,3]))

# f1.close()
