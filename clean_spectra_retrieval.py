"""Measure the SNR of a GPI image
author: johan mazoyer
"""

import sys, os
import glob
import distutils.dir_util
import warnings
import numpy as np
import astropy.io.fits as fits
import scipy.ndimage.interpolation as interpol
import scipy.ndimage.filters as scipy_filters

import pyklip.instruments.GPI as GPI
import pyklip.parallelized as parallelized
from pyklip.fmlib.diskfm import DiskFM
import pyklip.fm as fm
import time
import pyklip.parallelized as parallelized
from astropy.convolution import convolve

import matplotlib
matplotlib.use(
    'Agg'
)  ### I dont use matplotlib but I notice there is a conflict when I import matplotlib with pyklip if I don't use this line
warnings.filterwarnings("ignore", category=RuntimeWarning)


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


def make_collapsed_psf(dataset,
                       boxrad=10,
                       collapse_channels=1,
                       smoothed=False):
    """ create a PSF from the satspots, with a smoothed box
    Args:
        dataset: a pyklip instance of Instrument.Data
        boxrad: size of the PSF. Must be larger than 12 to have the box
        collapse_channels (int): number of output channels to evenly-ish collapse the 
                        dataset into. Default is 1 (broadband). Collapsed is done the same
                        way as in pyklip spectral_collapse function


    Returns:
        the PSF
    """

    # file_prefix = params_mcmc_yaml['FILE_PREFIX']
    # aligned_center = params_mcmc_yaml['ALIGNED_CENTER']
    # dimx = dataset.input.shape[1]
    # dimy = dataset.input.shape[2]

    # # create a traingle nan mask for the bright regions in 2015 probably due
    # # to the malfunctionning diode
    # if (file_prefix == 'K2band_hr4796') or (file_prefix == 'K1band_hr4796'):
    #     x_image = np.arange(dimx, dtype=np.float)[None, :] - aligned_center[0]
    #     y_image = np.arange(dimy, dtype=np.float)[:, None] - aligned_center[1]
    #     triangle1 = 0.67 * x_image + y_image - 114.5
    #     triangle2 = -3.2 * x_image + y_image - 330

    #     mask_triangle1 = np.ones((dimx, dimy))
    #     mask_triangle2 = np.ones((dimx, dimy))

    #     mask_triangle1[np.where((triangle1 > 0))] = np.nan
    #     mask_triangle2[np.where((triangle2 > 0))] = np.nan
    #     mask_triangle = mask_triangle1 * mask_triangle2
    # else:
    #     mask_triangle = np.ones((dimx, dimy))

    # dataset.input = dataset.input * mask_triangle

    dataset.generate_psfs(boxrad=boxrad)
    psfs_here = dataset.psfs

    numwvs = dataset.numwvs
    slices_per_group = numwvs // collapse_channels  # how many wavelengths per each output channel
    leftover_slices = numwvs % collapse_channels

    return_psf = np.zeros(
        [collapse_channels, psfs_here.shape[1], psfs_here.shape[2]])

    # populate the output image
    next_start_channel = 0  # initialize which channel to start with for the input images
    for i in range(collapse_channels):
        # figure out which slices to pick
        slices_this_group = slices_per_group
        if leftover_slices > 0:
            # take one extra slice, yummy
            slices_this_group += 1
            leftover_slices -= 1

        i_start = next_start_channel
        i_end = next_start_channel + slices_this_group  # this is the index after the last one in this group

        psfs_this_group = psfs_here[i_start:i_end, :, :]

        # Remove annoying RuntimeWarnings when input_4d is all nans
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return_psf[i, :, :] = np.nanmean(psfs_this_group, axis=0)
        next_start_channel = i_end

    if smoothed:
        r_smooth = boxrad - 1
        # # create rho2D for the psf square
        x_square = np.arange(2 * boxrad + 1, dtype=np.float)[None, :] - boxrad
        y_square = np.arange(2 * boxrad + 1, dtype=np.float)[:, None] - boxrad
        rho2d_square = np.sqrt(x_square**2 + y_square**2)

        smooth_mask = np.ones((2 * boxrad + 1, 2 * boxrad + 1))
        smooth_mask[np.where(rho2d_square > r_smooth - 1)] = 0.
        smooth_mask = scipy_filters.gaussian_filter(smooth_mask, 2.)
        smooth_mask[np.where(rho2d_square < r_smooth)] = 1.
        smooth_mask[np.where(smooth_mask < 0.01)] = 0.
        for i in range(collapse_channels):
            return_psf[i, :, :] *= smooth_mask

    # return_psf = return_psf / np.nanmax(return_psf)
    return_psf[np.where(return_psf < 0.)] = 0.
    return return_psf


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


def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore print
def enablePrint():
    sys.stdout = sys.__stdout__


## load the raw data
# basedir = '/Users/jmazoyer/Dropbox/STSCI/python/python_data/Empty_dataset_for_test_spectra/20180128_H_Spec_subset4injection/'
# basedir = '/Users/jmazoyer/Dropbox/ExchangeFolder/data_python/tycho/'

basedir = os.environ["EXCHANGE_PATH"]  # the base directory where is


sequence = '20161221_H_Spec'
# sequence = '160318_H_Spec_smalltest'
# sequence = '150403_K1_Spec'
# sequence = '150403_K2_Spec'
# sequence = '160318_H_Spec'
# sequence = '160323_J_Spec'

basedir = basedir + sequence + '/'

filelist = glob.glob(basedir + "*_spdc_distorcorr.fits")

dataset = GPI.GPIData(filelist, quiet=True)

initial_wl_number = np.round(dataset.input.shape[0] / len(filelist))

## load the model. We don't care about the normalization of the model
# model_initial_non_convolved = fits.getdata("/Users/jmazoyer/Dropbox/STSCI/python/python_data/KlipFM_for_SPF/hr_4796/model_mcmc/Hband_modelbeforeConv.fits")
model_initial_non_convolved = fits.getdata(
    basedir + "results_MCMC/Hd32297_Hband_adikl15_backend_file_mcmc_BestModel.fits"
)

## either use the PSF made from the sat spots
# psf_by_wavelength = fits.getdata(basedir + sequence + '_sat_spots_averaged.fits')
# psf_all = np.nanmean(psf_by_wavelength, axis = 0)

## or the PSF made from the unobstructed image
# if sequence == '140325_K1_Spec' |sequence == '150402_K1_Spec' | sequence == '150403_K1_Spec':
#     psf_all = fits.getdata('/Users/jmazoyer/Dropbox/STSCI/python/python_data/KlipFM_for_SPF/hr_4796/nice_psfs/psf_gpi_k1.fits')
# if sequence == '150403_K2_Spec':
#     psf_all = fits.getdata('/Users/jmazoyer/Dropbox/STSCI/python/python_data/KlipFM_for_SPF/hr_4796/nice_psfs/psf_gpi_k2.fits')
# if sequence == 'psf_gpi_h':
#     psf_all = fits.getdata('/Users/jmazoyer/Dropbox/STSCI/python/python_data/KlipFM_for_SPF/hr_4796/nice_psfs/psf_gpi_h.fits')
# if sequence == 'psf_gpi_j':
#     psf_all = fits.getdata('/Users/jmazoyer/Dropbox/STSCI/python/python_data/KlipFM_for_SPF/hr_4796/nice_psfs/psf_gpi_j.fits')

# psf_by_wavelength = np.zeros((initial_wl_number,dim,dim))

# for i in range(0, initial_wl_number):
#     psf_by_wavelength[i,:,:] = psf_all

dataset.OWA = 102
mov_here = 6
KLMODE = [
    3
]  # You should use only one KL mode for a start, because you might run into memroy problems.

nb_wl = 2  ### number of bin for the spectra (nb_wl=1 corresponds to all the WL collapsed, nb_wl=initial_wl_number uses all the WL)

print("Out of the ", len(np.unique(dataset.wvs)),
      "initial WL slices, we bin in", nb_wl, "WL slices")

psf_by_wavelength = make_collapsed_psf(dataset,
                                       boxrad=10,
                                       collapse_channels=nb_wl,
                                       smoothed=False)

Measure_KLbasis = True  # if Measure_KLbasis = False, jump the first steps and only and read the pickle file with the KL basis and do the

reduc = 'ADI_only/'
resultdir = basedir + reduc
distutils.dir_util.mkpath(resultdir)
starname = dataset.prihdrs[0]['OBJECT'].replace(" ", "_")
file_prefix_all = starname + '_' + sequence + '_' + str(nb_wl) + 'WL'

result_spectra_dir = resultdir + 'spectra/'
distutils.dir_util.mkpath(result_spectra_dir)

if nb_wl < initial_wl_number:
    dataset.spectral_collapse(collapse_channels=nb_wl,
                              align_frames=True,
                              numthreads=3)

the_wavelengths = np.unique(dataset.wvs)
dim = dataset.input.shape[1]

#### define the zone for the spectra. this zone has to be small (one PSF) and in a zone of your disk where your model well agree

x = np.arange(dim, dtype=np.float)[None, :] - int(np.round((dim - 1) / 2))
y = np.arange(dim, dtype=np.float)[:, None] - int(np.round((dim - 1) / 2))
rho2d = np.sqrt(x**2 + y**2)

radius_aperture_at_1point6 = 2.4
center_ap1 = [91, 187]
# center_ap2 = [174, 71]

mask_zone_spectra = np.zeros((nb_wl, dim, dim))
# toto = np.zeros((nb_wl,dim,dim))
for i in range(0, nb_wl):
    radius_aperture = radius_aperture_at_1point6 / (1.6) * the_wavelengths[i]

    aperture = np.zeros((dim, dim))
    x_ap1 = np.arange(dim, dtype=np.float)[None, :] - center_ap1[0]
    y_ap1 = np.arange(dim, dtype=np.float)[:, None] - center_ap1[1]
    rho2d_ap1 = np.sqrt(x_ap1**2 + y_ap1**2)
    aperture[np.where(rho2d_ap1 < radius_aperture)] = 1

    # x_ap2 = np.arange(dim, dtype=np.float)[None, :] - center_ap2[0]
    # y_ap2 = np.arange(dim, dtype=np.float)[:, None] - center_ap2[1]
    # rho2d_ap2 = np.sqrt(x_ap2**2 + y_ap2**2)

    # aperture[np.where(rho2d_ap2 < radius_aperture)] = 1

    mask_zone_spectra[i, :, :] = aperture

    # fits.writeto("/Users/jmazoyer/Desktop/aperture.fits", aperture , overwrite=True)
    # mask_zone_spectrai = mask_zone_spectra[i,:,:]
    # toto[i,:,:] = (1 + 2*mask_zone_spectra[i,:,:])*convolve(model_initial_non_convolved,psf_by_wavelength[i,:,:], boundary = 'wrap')
    # print(convolve(model_initial_non_convolved,psf_by_wavelength[i,:,:], boundary = 'wrap')[np.where(mask_zone_spectrai)])

# fits.writeto("/Users/jmazoyer/Desktop/show_zone_on_model.fits", toto , overwrite=True)

#### First reduction of the data using parallelized.klip_dataset (no FM)
blockPrint()
parallelized.klip_dataset(dataset,
                          numbasis=KLMODE,
                          maxnumbasis=100,
                          annuli=1,
                          subsections=1,
                          mode='ADI',
                          outputdir=resultdir,
                          fileprefix=file_prefix_all +
                          '_Measurewithparallelized',
                          aligned_center=[140, 140],
                          highpass=False,
                          minrot=mov_here,
                          calibrate_flux=True)
enablePrint()

if nb_wl == 1:
    data_reduction = fits.getdata(resultdir + file_prefix_all +
                                  "_Measurewithparallelized-KLmodes-all.fits")
    # fits.writeto("/Users/jmazoyer/Desktop/show_zone_on_data.fits", (1 + 2*mask_zone_spectra)*data_reduction, overwrite=True)

else:
    data_reduction = fits.getdata(resultdir + file_prefix_all +
                                  "_Measurewithparallelized-KL3-speccube.fits")
    # fits.writeto("/Users/jmazoyer/Desktop/show_zone_on_data.fits", np.mean((1 + 2*mask_zone_spectra)*data_reduction, axis=0), overwrite=True)

## Create noise a noise map for the Chisquare
delta_raddii = 3.
noise_map_multi = np.zeros((nb_wl, dim, dim))

blockPrint()
dataset.PAs = -dataset.PAs
parallelized.klip_dataset(dataset,
                          numbasis=KLMODE,
                          maxnumbasis=100,
                          annuli=1,
                          subsections=1,
                          mode='ADI',
                          outputdir=resultdir,
                          fileprefix=file_prefix_all +
                          '_couter_rotate_trick',
                          aligned_center=[140, 140],
                          highpass=False,
                          minrot=mov_here,
                          calibrate_flux=True)
dataset.PAs = -dataset.PAs                         
enablePrint()




# PAs = dataset.PAs
# anglesrange = PAs - np.median(PAs)
# anglesrange = np.append(anglesrange, [0])

# # create nan and zeros masks for the disk for the noise map
# mask_object_astro_zeros = np.ones((dim, dim))
# mask_disk_all_angles_nans = np.ones((dim, dim))

# # for hd32297 disk
# estimPA = 47.3
# estiminclin = 76.
# estimminr = 1.
# estimmaxr = 120.


# # for hr4796 disk
# # estimPA = 27.
# # estiminclin = 88.3
# # estimminr = 65.
# # estimmaxr = 85.

# PA_rad = (90 + estimPA) * np.pi / 180.

# x1 = x * np.cos(PA_rad) + y * np.sin(PA_rad)
# y1 = -x * np.sin(PA_rad) + y * np.cos(PA_rad)
# x = x1
# y = y1 / np.cos(estiminclin * np.pi / 180.)
# rho2dellip = np.sqrt(x**2 + y**2)
# mask_object_astro_zeros[np.where((rho2dellip > estimminr)
#                                  & (rho2dellip < estimmaxr))] = 0.

# for index_angle, PAshere in enumerate(anglesrange):
#     rot_disk = np.round(
#         interpol.rotate(mask_object_astro_zeros,
#                         PAshere,
#                         reshape=False,
#                         mode='wrap'))
#     mask_disk_all_angles_nans[np.where(rot_disk == 0)] = np.nan


#### set the first multiwavelength model by normalizing it at the value of the reduced data (initial spectra = spectra_without_FM)
initial_spectra = np.zeros(nb_wl)
model_multi_spectra_initial = np.zeros((nb_wl, dim, dim))

error_on_zone_spectra = np.zeros(nb_wl)
SNR_on_zone_spectr = np.zeros(nb_wl)

for i in range(0, nb_wl):
    data_reduction_wli = data_reduction[i, :, :]
    psfi = psf_by_wavelength[i, :, :]
    model_convolvei = convolve(model_initial_non_convolved,
                               psfi,
                               boundary='wrap')
    mask_zone_spectrai = mask_zone_spectra[i, :, :]

    spec_value = np.sum(
        data_reduction_wli[np.where(mask_zone_spectrai)]) / np.sum(
            model_convolvei[np.where(mask_zone_spectrai)])
    # modeli = model_convolvei * spec_value
    model_multi_spectra_initial[i, :, :] = model_convolvei
    initial_spectra[i] = np.mean(model_convolvei[np.where(mask_zone_spectrai)])
    str_counter_rot = file_prefix_all + '_couter_rotate_trick-KL{0}-speccube.fits'.format(KLMODE[0])
    
    reduced_data_nodisk = fits.getdata(
        os.path.join(resultdir,
                     str_counter_rot))[i]
    
    noise = make_noise_map_rings(reduced_data_nodisk,
                                 aligned_center=[140,140],
                                 delta_raddii=3)
    noise[np.where(noise == 0)] = np.nan  #we are going to divide by this noise

    # noise = np.zeros((dim, dim))
    # for i_ring in range(0, int(np.floor(140 / delta_raddii)) - 2):
    #     image_masked = data_reduction_wli * mask_disk_all_angles_nans
    #     wh_rings = np.where((rho2d >= i_ring * delta_raddii)
    #                         & (rho2d < (i_ring + 1) * delta_raddii))
    #     noise[wh_rings] = np.nanstd(image_masked[wh_rings])

    noise_map_multi[i] = noise
    error_on_zone_spectra[i] = np.mean(noise[np.where(mask_zone_spectrai)])
    SNR_on_zone_spectr[i] = np.mean(data_reduction_wli[np.where(
        mask_zone_spectrai)]) / error_on_zone_spectra[i]

# fits.writeto(result_spectra_dir + file_prefix_all + '_model_multi_spectra_initial.fits', model_multi_spectra_initial, overwrite=True)
# asd
## We also check that the SNR is good enough in the zone we use to minimize.
print("SNR on zone spectra:", SNR_on_zone_spectr)
if len(SNR_on_zone_spectr[np.where(SNR_on_zone_spectr < 2)]) > 0:
    print(
        "the SNR in the spectra zone is <2 in at least one of the binned slice. You should decrease the # of bin of the spectra"
    )

## we define the zone for the Chisquare. It might be the same as the spectra zone or not. It is better to have something where you know that the
## model and the data agree very well in the collapsed data. First try with nb_wl = 1 and find a place where you have minimal in chisquare zone
mask_for_chisquare = fits.getdata(basedir + "klip_fm_files/Hd32297_Hband_adikl15_mask2minimize.fits")

# mask_for_chisquare = np.zeros((dim, dim))
# toto = np.zeros((nb_wl,dim,dim))

# radius_aperture = 2
# x_ap1 = np.arange(dim, dtype=np.float)[None, :] - center_ap1[0]
# y_ap1 = np.arange(dim, dtype=np.float)[:, None] - center_ap1[1]
# rho2d_ap1 = np.sqrt(x_ap1**2 + y_ap1**2)
# mask_for_chisquare[np.where(rho2d_ap1 < radius_aperture)] = 1

# x_ap2 = np.arange(dim, dtype=np.float)[None, :] - center_ap2[0]
# y_ap2 = np.arange(dim, dtype=np.float)[:, None] - center_ap2[1]
# rho2d_ap2 = np.sqrt(x_ap2**2 + y_ap2**2)

# mask_for_chisquare[np.where(rho2d_ap2 < radius_aperture)] = 1

# fits.writeto(result_spectra_dir +file_prefix_all + 'model_multi_spectra_initial.fits', model_multi_spectra_initial, overwrite=True)

#### Using the spectra_without_FM as initial point, we do a first forward modelling
if Measure_KLbasis == True:
    blockPrint()
    # initialize the DiskFM object
    diskobj = DiskFM(dataset.input.shape,
                     KLMODE,
                     dataset,
                     model_multi_spectra_initial,
                     basis_filename=os.path.join(
                         resultdir, file_prefix_all + '_klip-basis.h5'),
                     save_basis=True,
                     aligned_center=[140, 140])
    # measure the KL basis and save it

    maxnumbasis = dataset.input.shape[0]
    fm.klip_dataset(dataset,
                    diskobj,
                    numbasis=KLMODE,
                    maxnumbasis=maxnumbasis,
                    annuli=1,
                    subsections=1,
                    mode='ADI',
                    outputdir=resultdir,
                    fileprefix=file_prefix_all,
                    aligned_center=[140, 140],
                    mute_progression=True,
                    highpass=False,
                    minrot=3,
                    calibrate_flux=True,
                    numthreads=1)

    enablePrint()

# We load the KL basis. This step is fairly long. Once loaded the variable diskobj can be passed without having to reload the KL basis.
blockPrint()
diskobj = DiskFM(dataset.input.shape,
                 KLMODE,
                 dataset,
                 model_multi_spectra_initial,
                 basis_filename=os.path.join(
                     resultdir, file_prefix_all + '_klip-basis.h5'),
                 save_basis=False,
                 load_from_basis=True,
                 aligned_center=[140, 140])
# diskobj = DiskFM([len(filelist)*nb_wl, model_multi_spectra_initial.shape[1], model_multi_spectra_initial.shape[2]], KLMODE,dataset, model_multi_spectra_initial, annuli=1, subsections=1,
#                         basis_filename = resultdir+ file_prefix_all+'_klip-basis.h5', save_basis = False, load_from_basis = True)
enablePrint()

n_iter_fm = 5  # number of iteration for the spectra (it should converge really fast)
nb_points_Chisquare = 100

spectra_zone = np.zeros((n_iter_fm + 3, nb_wl))
print("the wavelengths: ", the_wavelengths)

spectra_zone[0, :] = the_wavelengths
spectra_zone[1, :] = initial_spectra

model_multi_spectra = model_multi_spectra_initial
print("non FM spectra: ", initial_spectra)

for iter_fm in range(0, n_iter_fm):
    # We update the model and measure the new KL number.
    diskobj.update_disk(model_multi_spectra)
    FMmodels = diskobj.fm_parallelized()[0]  # There is only one KL mode

    toto = np.arange(1, nb_points_Chisquare / 2 +
                     1) / float(nb_points_Chisquare) * 4. / float(
                         (iter_fm + 2)**3) + 1.
    spec_values = np.concatenate([1 / toto, toto])

    for Spec_index in range(0, nb_wl):

        FMmodels_wli = FMmodels[Spec_index, :, :]
        data_reduction_wli = data_reduction[Spec_index, :, :]
        noise_map_wli = noise_map_multi[Spec_index, :, :]
        mask_zone_spectrai = mask_zone_spectra[i, :, :]

        Chisquare_best = 10000000000.
        Best_spec_value = 0.

        for i_chisquare in range(0, len(spec_values)):

            #### measurement of the chisquare
            Chisquare_zone = np.sum(
                np.abs((data_reduction_wli[np.where(mask_for_chisquare)] -
                        spec_values[i_chisquare] *
                        FMmodels_wli[np.where(mask_for_chisquare)]) /
                       noise_map_wli[np.where(mask_for_chisquare)]))
            if Chisquare_zone < Chisquare_best:
                Chisquare_best = Chisquare_zone
                Best_spec_value = spec_values[i_chisquare]

        best_model_wli = model_multi_spectra[
            Spec_index, :, :] * Best_spec_value
        FMmodels[
            Spec_index, :, :] = FMmodels[Spec_index, :, :] * Best_spec_value
        model_multi_spectra[Spec_index, :, :] = best_model_wli
        spectra_zone[iter_fm + 2, Spec_index] = np.mean(
            best_model_wli[np.where(mask_zone_spectrai)])

    print("FM spectra iter", iter_fm, ":", spectra_zone[iter_fm + 2, :])

spectra_zone[-1, :] = error_on_zone_spectra
print("STD on zone spectra:", error_on_zone_spectra)

fits.writeto(result_spectra_dir + file_prefix_all + '_all_spectra.fits',
             spectra_zone,
             overwrite=True)

if nb_wl == 1:
    residuals = np.nanmean(FMmodels - data_reduction, axis=0)
    noise_map_1d = np.nanmean(np.abs(noise_map_multi), axis=0)
    SNR_residuals = np.mean(residuals[np.where(mask_for_chisquare)] /
                            noise_map_1d[np.where(mask_for_chisquare)])
    print("SNR of the residuals in Chisquare zone :", SNR_residuals)

fits.writeto(result_spectra_dir + file_prefix_all +
             '_model_multi_spectra_final_iter.fits',
             model_multi_spectra,
             overwrite=True)
fits.writeto(result_spectra_dir + file_prefix_all +
             '_modelFM_multi_spectra_final_iter.fits',
             FMmodels,
             overwrite=True)
fits.writeto(result_spectra_dir + file_prefix_all +
             '_residuals_multi_spectra_final_iter.fits',
             np.abs(FMmodels - data_reduction),
             overwrite=True)

### cheat code here measurement of the real spectra if we are on a test case with injection
# real_spectra_models = fits.getdata("/Users/jmazoyer/Dropbox/STSCI/python/python_data/Empty_dataset_for_test_spectra/models_initial_spectra.fits")

# real_spectra = np.zeros(nb_wl)

# if nb_wl == 1:
#     real_spectra_models_concat = np.nanmean(real_spectra_models, axis = 0) ### Collapse is doing a mean of the wavelength
#     print(real_spectra_models_concat.shape)
#     real_spectra[0] = np.mean(real_spectra_models_concat[np.where(mask_zone_spectra[0,:,:])])
# else:
#     slices_per_group = initial_wl_number // nb_wl # how many wavelengths per each output channel
#     leftover_slices = initial_wl_number % nb_wl
#     next_start_channel = 0 # initialize which channel to start with for the input images
#     for i in range(nb_wl):
#             # figure out which slices to pick
#             slices_this_group = slices_per_group
#             if leftover_slices > 0:
#                 # take one extra slice, yummy THIS IS THE WAY THIS IS CODED IN JASON's spectral_collapse
#                 slices_this_group += 1
#                 leftover_slices -= 1

#             i_start = next_start_channel
#             i_end = next_start_channel + slices_this_group # this is the index after the last one in this group

#             realmodeli = np.nanmean(real_spectra_models[i_start:i_end,:,:], axis=0)
#             mask_zone_spectrai = mask_zone_spectra[i,:,:]
#             real_spectra[i] = np.mean(realmodeli[np.where(mask_zone_spectrai)])
#             next_start_channel = i_end

# print("real spectra: ", real_spectra)

# print("real spectra - recovered FM spectra : ", real_spectra - spectra_zone[-2,:] )

# diskobj.update_disk(real_spectra_models)
# FMmodels = diskobj.fm_parallelized()[0] # There is only one KL mode

# fits.writeto(result_spectra_dir +file_prefix_all + 'residuals_multi_spectra_with_actual_spectra.fits', np.abs(FMmodels-data_reduction), overwrite=True)
