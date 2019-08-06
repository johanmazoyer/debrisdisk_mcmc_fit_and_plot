"""check the sat spots. 
Return the filename where the sat spots intersect the disk
save in fits all the sat spots if they do not intersect with disk
author: Johan Mazoyer
"""

import os
import glob
import numpy as np
import astropy.io.fits as fits

import scipy.ndimage.interpolation as interpol
import scipy.ndimage.filters as scipy_filters

import pyklip.instruments.GPI as GPI
import pyklip.klip as klip

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def check_gpi_satspots(basedir,SavePSF = True,  name_psf = 'psf_satspot', removed_slices = None, radius_aperture_at_1point6 = 2.4, r_in_ring_noise_at_1point6 = 9, r_out_ring_noise_at_1point6 = 12,SaveAll = False, quiet = True):

    excluded_files = list()
    sequence = basedir.split('/')[-2]
    filelist = glob.glob(basedir + "*_distorcorr.fits")
    nb_init = len(filelist)

    dataset = GPI.GPIData(filelist, quiet=True)

    PAs = dataset.PAs
    Wavelengths = dataset.wvs
    Starpos = dataset.centers
    filenames = dataset.filenames
    header0 = dataset.prihdrs
    header1 = dataset.exthdrs

    dim = dataset.input.shape[1]
    initial_wl_number = int(np.round(dataset.input.shape[0]/len(filelist)))

    ### Where is the disk
    # create nan and zeros masks for the disk
    mask_object_astro_ones = np.zeros((dim, dim))

    # for hr4796 disk
    estimPA = 27.
    estiminclin = 76.
    estimminr = 65.
    estimmaxr = 83.
    PA_rad = (90 + estimPA)*np.pi/180.

    x = np.arange(dim, dtype=np.float)[None,:] - (dim-1)/2
    y = np.arange(dim, dtype=np.float)[:,None] - (dim-1)/2
    rho2d = np.sqrt(x**2 + y**2)

    x1 = x * np.cos(PA_rad) + y * np.sin(PA_rad)
    y1 = -x * np.sin(PA_rad) + y * np.cos(PA_rad)
    x = x1
    y = y1 / np.cos(estiminclin*np.pi/180.)
    rho2dellip = np.sqrt(x**2 + y**2)
    mask_object_astro_ones[np.where(
        (rho2dellip > estimminr) & (rho2dellip < estimmaxr))] = 1.

    # create a nan mask for the bright regions in 2015 probably due to the malfunctionning diode
    if sequence == '150403_K1_Spec' or sequence == '150403_K2_Spec' :
        x_image = np.arange(dim, dtype=np.float)[None,:] - 140
        y_image = np.arange(dim, dtype=np.float)[:,None] - 140
        triangle1 = 0.67*x_image + y_image - 114.5
        triangle2 = -3.2*x_image + y_image - 330

        mask_triangle1 = np.ones((dim, dim))
        mask_triangle2 = np.ones((dim, dim))

        mask_triangle1[np.where( (triangle1>0) )] = np.nan
        mask_triangle2[np.where( (triangle2>0) )] = np.nan

    #define the array for saving the satspot aperture flux
    value_sat_spot = np.zeros((initial_wl_number,len(filelist)))

    ## for the square to save the sat spots, we make it slighlty larger than the minimum size to include the noise of the rings
    half_size_square = int(np.round((r_out_ring_noise_at_1point6/(1.6)*np.max(Wavelengths)+4)/2.)*2)
    save_sat_spots = np.zeros((4,len(filelist),initial_wl_number,half_size_square*2 +1,half_size_square*2 +1))

    save_SNR_sat_spots = np.zeros((4,len(filelist),initial_wl_number))

    # create rho2D for the psf square
    x_square = np.arange(2*half_size_square +1, dtype=np.float)[None,:] - half_size_square
    y_square = np.arange(2*half_size_square +1, dtype=np.float)[:,None] - half_size_square
    rho2d_square = np.sqrt(x_square**2 + y_square**2)

    for index_angle in range(0, len(filelist)):
        
        disk_intercept_sat_spot_bool = False

        header_anglei = header1[index_angle]
        data_in_the_fits = dataset.input[initial_wl_number * index_angle + np.arange(initial_wl_number)]

        for index_wl in range(0, initial_wl_number):
            
            image_here = data_in_the_fits[index_wl]
            
            value_sat_spot_image_here = np.zeros(4)

            # Just a test
            test_satspot = np.zeros((dim,dim))

            radius_aperture = radius_aperture_at_1point6/(1.6)*Wavelengths[initial_wl_number * index_angle + index_wl]
            r_in_ring_noise = r_in_ring_noise_at_1point6/(1.6)*Wavelengths[initial_wl_number * index_angle + index_wl]
            r_out_ring_noise = r_out_ring_noise_at_1point6/(1.6)*Wavelengths[initial_wl_number * index_angle + index_wl]

            str_head_satspot = 'SATS'+str(index_wl)

            papath, filename_here =  os.path.split(filenames[initial_wl_number * index_angle + index_wl])

            model_mask_rot = np.round(np.abs(klip.rotate(mask_object_astro_ones, PAs[initial_wl_number * index_angle + index_wl], [140, 140], \
                                [Starpos[initial_wl_number * index_angle + index_wl, 0], Starpos[initial_wl_number * index_angle + index_wl, 1]])))
            # fits.writeto("/Users/jmazoyer/Desktop/model_mask_rot.fits",model_mask_rot, overwrite=True)



            model_mask_rot[np.where(model_mask_rot != model_mask_rot )] = 0
            # fits.writeto("/Users/jmazoyer/Desktop/model_mask_rot.fits", model_mask_rot, overwrite=True)

            for sat_spot_number in range(0, 4):

                ## find the center of this sat spot        
                satspotcenter = list(filter(None,header_anglei[str_head_satspot + '_'+str(sat_spot_number)].split(' ')))
                center_sat_x = float(satspotcenter[0])
                center_sat_y = float(satspotcenter[1])

                # create rho2D for this sat spot
                x_sat = np.arange(dim, dtype=np.float)[None,:] - center_sat_x
                y_sat = np.arange(dim, dtype=np.float)[:,None] - center_sat_y
                rho2d_sat = np.sqrt(x_sat**2 + y_sat**2)

                # create sat spot zone
                wh_aperture_sat = np.where( (rho2d_sat < radius_aperture))
                wh_noise_sat = np.where((rho2d_sat < r_out_ring_noise) & (rho2d_sat > r_in_ring_noise) & (image_here == image_here) )

                # Just a test
                test_satspot[wh_aperture_sat] = 1
                test_satspot[wh_noise_sat] = 1

                # exclude images where the disk is too close to the sat spot
                if np.sum(model_mask_rot[wh_aperture_sat]) >0:
                    disk_intercept_sat_spot_bool = True
                    if not quiet: 
                        print(filename_here, 'removed because of the sat spot #'+str(sat_spot_number))
                    save_sat_spots[sat_spot_number,index_angle,index_wl,:,:] *= np.nan
                else:
                    # save the sat spot centered on a square 

                    # Crop around the sat and center the sat spot on the square
                    spot_square = image_here[int(round(center_sat_y)) - half_size_square :  int(round(center_sat_y)) + half_size_square +1,int(round(center_sat_x)) - half_size_square :  int(round(center_sat_x)) + half_size_square+1]
                    
                    # sub-pixel shift to center the sat spot on [half_size_square, half_size_square] pixel
                    
                    # interpol.shift does not work well with nans, so if my square reach the edge I replace them by zeros, then shift, then put the nans back.
                    # This is ~ok because this is a sub pixel shift and because the nans are usually the edge of the IFS slice.
                    # this is not perfect, so for that reasons, avoid taking half_size_square too large
                    wh_spot_square_nan = np.where(spot_square != spot_square)
                    spot_square[wh_spot_square_nan] = 0.
                    spot_square = interpol.shift(spot_square, ( int(round(center_sat_y)) - center_sat_y,int(round(center_sat_x))- center_sat_x))
                    spot_square[wh_spot_square_nan] = np.nan
                    
                    # now that we know that the disk is not on this sat spot, we remove the disk
                    crop_model_mask_rot = model_mask_rot[int(round(center_sat_y)) - half_size_square :  int(round(center_sat_y)) + half_size_square +1,int(round(center_sat_x)) - half_size_square :  int(round(center_sat_x)) + half_size_square+1]
                    spot_square[np.where(crop_model_mask_rot == 1)] = np.nan
                    
                    # if there is a bright zone in the IFS slice, we also remove it
                    if sequence == '150403_K1_Spec' or sequence == '150403_K2_Spec' :
                        crop_mask_triangle1 = mask_triangle1[int(round(center_sat_y)) - half_size_square :  int(round(center_sat_y)) + half_size_square +1,int(round(center_sat_x)) - half_size_square :  int(round(center_sat_x)) + half_size_square+1]
                        crop_mask_triangle2 = mask_triangle2[int(round(center_sat_y)) - half_size_square :  int(round(center_sat_y)) + half_size_square +1,int(round(center_sat_x)) - half_size_square :  int(round(center_sat_x)) + half_size_square+1]
                        spot_square = spot_square*crop_mask_triangle1*crop_mask_triangle2

                    save_sat_spots[sat_spot_number,index_angle,index_wl,:,:] =spot_square

                # if one on the sat spots of one of the slice of a datacube intersects the disk, we will remove this datacube completely, 
                # so there is no use measuring the normalization for any of them
                if disk_intercept_sat_spot_bool == False:
                    # fits.writeto( "/Users/jmazoyer/Desktop/sat.fits",save_sat_spots[sat_spot_number,index_angle,index_wl,:,:] , overwrite=True)
                    
                    sat_spot_here = save_sat_spots[sat_spot_number,index_angle,index_wl,:,:]
                    
                    # # You can try to high pass filter the sat spots but this is usually very aggressive on the sat spots so I do not recommand it
                    # sat_spot_here = klip.high_pass_filter(sat_spot_here, 2*radius_aperture) 

                    # define the zoen for aperture and noise
                    wh_aperture_square = np.where( (rho2d_square < radius_aperture) & (sat_spot_here == sat_spot_here) )
                    wh_noise_square = np.where((rho2d_square < r_out_ring_noise) & (rho2d_square > r_in_ring_noise) & (sat_spot_here == sat_spot_here) )
                
                    # measure aperture flux for this sat spot.  Sum on the aperture - the noise 
                    ### we "np.clip" here to be sure that we do not include the negative first ring in the aperture
                    mean_noise = np.nanmean(sat_spot_here[wh_noise_square])
                    value_sat_spot_image_here[sat_spot_number] = np.nansum(np.clip(sat_spot_here[wh_aperture_square] - mean_noise,a_min = 0., a_max = None))                
                    
                    stdnoise = np.nanstd(sat_spot_here[wh_noise_square])
                    save_SNR_sat_spots[sat_spot_number, index_angle,index_wl] = np.nanmean(np.clip(sat_spot_here[wh_aperture_square] -mean_noise,a_min = 0., a_max = None))/stdnoise
                    
                else:
                    save_SNR_sat_spots[sat_spot_number, index_angle,index_wl] = np.nan
                    value_sat_spot_image_here[sat_spot_number] = np.nan

            # Now that we have the value, we make the mean for the 4 sat spots
            value_sat_spot[index_wl,index_angle] = np.nanmean(value_sat_spot_image_here)

            # we normalize the data 
            if value_sat_spot[index_wl,index_angle] > 0.:
                data_in_the_fits[index_wl] = data_in_the_fits[index_wl]/value_sat_spot[index_wl,index_angle]

            if sequence == '150403_K1_Spec' or sequence == '150403_K2_Spec' :
                data_in_the_fits[index_wl] = data_in_the_fits[index_wl]*mask_triangle1*mask_triangle2
            
            ### Just a test if you want to check where are the sat spots and the disk
            # data_in_the_fits[index_wl] = data_in_the_fits[index_wl]*(1-model_mask_rot)*(1-test_satspot)
            # fits.writeto("/Users/jmazoyer/Desktop/titi.fits", data_in_the_fits, overwrite=True)
                
        filenamewithouextension, extension = os.path.splitext(filenames[initial_wl_number * index_angle])
        if disk_intercept_sat_spot_bool == True:
            excluded_files.append(filenames[initial_wl_number * index_angle])
            
        # if disk_intercept_sat_spot_bool == False:
            
        #     header0[index_angle]['history'] = ''
        #     header0[index_angle]['history'] = 'To remove the PSF we remove {0} slices'.format(removed_slices)
        #     header0[index_angle]['history'] = ''
        #     header0[index_angle]['history'] = 'Normalized for disk with normalize_image_for_disk_spectra.py'
        #     header0[index_angle]['history'] = 'with a sat spot aperture of radius {0:.2f}*wl/1.6 pix'.format(radius_aperture_at_1point6)
        #     header0[index_angle]['history'] = 'we removed the noise measured on a ring of {0:.2f}*wl/1.6 to {1:.2f}*wl/1.6 pix'.format(r_in_ring_noise_at_1point6, r_out_ring_noise_at_1point6)
        #     header0[index_angle]['history'] = ''

        #     # dataset.input = data_in_the_fits
        #     # # PAs = dataset.PAs
        #     # # Wavelengths = dataset.wvs
        #     # # Starpos = dataset.centers
        #     # # filenames = dataset.filenames
        #     # # header0 = dataset.prihdrs
        #     # # header1 = dataset.exthdrs

        #     # dataset.output_wcs = dataset.wcs
        #     # dataset.output_centers = dataset.centers
        #     # dataset.savedata(filepath= filenamewithouextension +'_normalized.fits', data= dataset.input )


        #     new_fits = fits.HDUList()
        #     new_fits.append(fits.ImageHDU(data=None, header=header0[index_angle]))
        #     new_fits.append(fits.ImageHDU(data=data_in_the_fits, header=header1[index_angle]))
            
        #     new_fits.writeto(filenamewithouextension +'_normalized.fits', overwrite=True)
        #     new_fits.close()
        # else:
        #     if os.path.isfile(filenamewithouextension +'_normalized.fits'):
        #         os.remove(filenamewithouextension +'_normalized.fits')



    ## save all sat spots raw format

    if SaveAll == True:
        hdr = fits.Header()
        for excluded_filesi in excluded_files: 
            hdr['history'] = 'disk intersect sat spot: we remove in {0}'.format(excluded_filesi)
        
        fits.writeto(basedir+sequence +  "_sat_spots_all.fits",save_sat_spots ,hdr, overwrite=True)
    

    # four sat spots mean 
    mean_save_SNR_sat_spots = np.nanmean(save_SNR_sat_spots, axis = 0)
    mean_save_sat_spots = np.nanmean(save_sat_spots, axis = 0)
    ## mean over the sequence
    mean_save_SNR_sat_spots_per_wl = np.nanmean(mean_save_SNR_sat_spots, axis = 0)
    mean_save_sat_spots_per_wl = np.nanmean(mean_save_sat_spots, axis = 0)


    for index_wl in range(0, initial_wl_number):
        if (removed_slices is not None) and (index_wl in removed_slices):
            mean_save_sat_spots_per_wl[index_wl,:,:] = np.nan*mean_save_sat_spots_per_wl[0,:,:]
            mean_save_SNR_sat_spots_per_wl[index_wl] *= np.nan
        else:
            sat_spot_here = mean_save_sat_spots_per_wl[index_wl,:,:]

            ## just a small test to check the size of the aperture and clean the PSF a little
            radius_aperture = radius_aperture_at_1point6/(1.6)*np.unique(Wavelengths)[index_wl]
            r_in_ring_noise = r_in_ring_noise_at_1point6/(1.6)*np.unique(Wavelengths)[index_wl]
            r_out_ring_noise = r_out_ring_noise_at_1point6/(1.6)*np.unique(Wavelengths)[index_wl]
            wh_aperture_square = np.where( (rho2d_square < radius_aperture) )
            wh_noise_square = np.where((rho2d_square < r_out_ring_noise) & (rho2d_square > r_in_ring_noise))

            mask_aperture = np.zeros((2*half_size_square +1,2*half_size_square +1))
            mask_noise = np.zeros((2*half_size_square +1,2*half_size_square +1))
            mask_aperture[wh_aperture_square] = 1.
            mask_noise[wh_noise_square] = 1.

            sat_spot_here = np.clip(sat_spot_here - np.mean(sat_spot_here[wh_noise_square]),a_min = 0., a_max = None)
            # sat_spot_here = klip.high_pass_filter(sat_spot_here, 3*radius_aperture)
            # sat_spot_here -= np.nanmin(sat_spot_here) 

            smooth_mask = np.ones((2*half_size_square +1,2*half_size_square +1))
            smooth_mask[np.where(rho2d_square > r_out_ring_noise-1)] = 0.
            smooth_mask = scipy_filters.gaussian_filter(smooth_mask, 2.)
            smooth_mask[np.where(rho2d_square < r_in_ring_noise)] = 1.
            smooth_mask[np.where(smooth_mask < 0.01)] = 0.

            # sat_spot_here *= mask_aperture + mask_noise
            
            sat_spot_here = sat_spot_here*smooth_mask
            sat_spot_here = sat_spot_here/np.nanmax(sat_spot_here)
            mean_save_sat_spots_per_wl[index_wl,:,:] = sat_spot_here


    hdr = fits.Header()
    hdr['CD3_3'] = header_anglei['CD3_3']
    hdr['CRPIX3'] = header_anglei['CRPIX3']
    hdr['CRVAL3'] = header_anglei['CRVAL3']
    hdr['CTYPE3'] = header_anglei['CTYPE3']
    hdr['CUNIT3'] = header_anglei['CUNIT3']
    hdr['history'] = 'To measure the PSF we remove {0} slices'.format(removed_slices)
    if SavePSF == True:
        fits.writeto(basedir+ name_psf + ".fits",np.nanmean(mean_save_sat_spots_per_wl, axis = 0) ,hdr, overwrite=True)
    if SaveAll == True:
        fits.writeto(basedir+ name_psf + "_4spotaverage_Seqaveraged.fits",mean_save_sat_spots_per_wl ,hdr, overwrite=True)

    ## save the sat spots values
    # fits.writeto(basedir+sequence + "_value_sat_spot.fits",np.nanmean(value_sat_spot,axis =1) , overwrite=True)

    # filelist_end = glob.glob(basedir + "*_normalized.fits")
    nb_excluded = len(excluded_files)
    print(sequence + ': We remove ' + str(nb_excluded) +' files out of '+ str(nb_init) +' because sat spots intersected the disk')

    print('for the selected slices, the mean SNR for the sat spots are at each wl', mean_save_SNR_sat_spots_per_wl)
    if np.all(removed_slices) == None:
        print(sequence + ": All Wl used to measure the PSF")
    else:
        print(sequence + ": We cut WL {0} to measure the PSF ".format(removed_slices))

    return excluded_files




def gpi_satspots_emptydataset(basedir,SavePSF = True,  name_psf = 'psf_satspot', removed_slices = None, radius_aperture_at_1point6 = 2.4, r_in_ring_noise_at_1point6 = 9, r_out_ring_noise_at_1point6 = 12,SaveAll = False, quiet = True):

    excluded_files = list()
    sequence = basedir.split('/')[-2]
    filelist = glob.glob(basedir + "*_distorcorr.fits")
    nb_init = len(filelist)

    dataset = GPI.GPIData(filelist, quiet=True)

    PAs = dataset.PAs
    Wavelengths = dataset.wvs
    Starpos = dataset.centers
    filenames = dataset.filenames
    header0 = dataset.prihdrs
    header1 = dataset.exthdrs

    dim = dataset.input.shape[1]
    initial_wl_number = int(np.round(dataset.input.shape[0]/len(filelist)))

    ### Where is the disk
    # create nan and zeros masks for the disk
    mask_object_astro_ones = np.zeros((dim, dim))

    # for hr4796 disk
    estimPA = 27.
    estiminclin = 76.
    estimminr = 65.
    estimmaxr = 83.
    PA_rad = (90 + estimPA)*np.pi/180.

    x = np.arange(dim, dtype=np.float)[None,:] - (dim-1)/2
    y = np.arange(dim, dtype=np.float)[:,None] - (dim-1)/2
    rho2d = np.sqrt(x**2 + y**2)

    x1 = x * np.cos(PA_rad) + y * np.sin(PA_rad)
    y1 = -x * np.sin(PA_rad) + y * np.cos(PA_rad)
    x = x1
    y = y1 / np.cos(estiminclin*np.pi/180.)
    rho2dellip = np.sqrt(x**2 + y**2)
    mask_object_astro_ones[np.where(
        (rho2dellip > estimminr) & (rho2dellip < estimmaxr))] = 1.

    mask_object_astro_ones *=0

    # create a nan mask for the bright regions in 2015 probably due to the malfunctionning diode
    if sequence == '150403_K1_Spec' or sequence == '150403_K2_Spec' :
        x_image = np.arange(dim, dtype=np.float)[None,:] - 140
        y_image = np.arange(dim, dtype=np.float)[:,None] - 140
        triangle1 = 0.67*x_image + y_image - 114.5
        triangle2 = -3.2*x_image + y_image - 330

        mask_triangle1 = np.ones((dim, dim))
        mask_triangle2 = np.ones((dim, dim))

        mask_triangle1[np.where( (triangle1>0) )] = np.nan
        mask_triangle2[np.where( (triangle2>0) )] = np.nan

    #define the array for saving the satspot aperture flux
    value_sat_spot = np.zeros((initial_wl_number,len(filelist)))

    ## for the square to save the sat spots, we make it slighlty larger than the minimum size to include the noise of the rings
    half_size_square = int(np.round((r_out_ring_noise_at_1point6/(1.6)*np.max(Wavelengths)+4)/2.)*2)
    save_sat_spots = np.zeros((4,len(filelist),initial_wl_number,half_size_square*2 +1,half_size_square*2 +1))

    save_SNR_sat_spots = np.zeros((4,len(filelist),initial_wl_number))

    # create rho2D for the psf square
    x_square = np.arange(2*half_size_square +1, dtype=np.float)[None,:] - half_size_square
    y_square = np.arange(2*half_size_square +1, dtype=np.float)[:,None] - half_size_square
    rho2d_square = np.sqrt(x_square**2 + y_square**2)

    for index_angle in range(0, len(filelist)):
        
        disk_intercept_sat_spot_bool = False

        header_anglei = header1[index_angle]
        data_in_the_fits = dataset.input[initial_wl_number * index_angle + np.arange(initial_wl_number)]

        for index_wl in range(0, initial_wl_number):
            
            image_here = data_in_the_fits[index_wl]
            
            value_sat_spot_image_here = np.zeros(4)

            # Just a test
            test_satspot = np.zeros((dim,dim))

            radius_aperture = radius_aperture_at_1point6/(1.6)*Wavelengths[initial_wl_number * index_angle + index_wl]
            r_in_ring_noise = r_in_ring_noise_at_1point6/(1.6)*Wavelengths[initial_wl_number * index_angle + index_wl]
            r_out_ring_noise = r_out_ring_noise_at_1point6/(1.6)*Wavelengths[initial_wl_number * index_angle + index_wl]

            str_head_satspot = 'SATS'+str(index_wl)

            papath, filename_here =  os.path.split(filenames[initial_wl_number * index_angle + index_wl])

            model_mask_rot = np.round(np.abs(klip.rotate(mask_object_astro_ones, PAs[initial_wl_number * index_angle + index_wl], [140, 140], \
                                [Starpos[initial_wl_number * index_angle + index_wl, 0], Starpos[initial_wl_number * index_angle + index_wl, 1]])))
            # fits.writeto("/Users/jmazoyer/Desktop/model_mask_rot.fits",model_mask_rot, overwrite=True)



            model_mask_rot[np.where(model_mask_rot != model_mask_rot )] = 0
            # fits.writeto("/Users/jmazoyer/Desktop/model_mask_rot.fits", model_mask_rot, overwrite=True)

            for sat_spot_number in range(0, 4):

                ## find the center of this sat spot        
                satspotcenter = list(filter(None,header_anglei[str_head_satspot + '_'+str(sat_spot_number)].split(' ')))
                center_sat_x = float(satspotcenter[0])
                center_sat_y = float(satspotcenter[1])

                # create rho2D for this sat spot
                x_sat = np.arange(dim, dtype=np.float)[None,:] - center_sat_x
                y_sat = np.arange(dim, dtype=np.float)[:,None] - center_sat_y
                rho2d_sat = np.sqrt(x_sat**2 + y_sat**2)

                # create sat spot zone
                wh_aperture_sat = np.where( (rho2d_sat < radius_aperture))
                wh_noise_sat = np.where((rho2d_sat < r_out_ring_noise) & (rho2d_sat > r_in_ring_noise) & (image_here == image_here) )

                # Just a test
                test_satspot[wh_aperture_sat] = 1
                test_satspot[wh_noise_sat] = 1

                # exclude images where the disk is too close to the sat spot
                if np.sum(model_mask_rot[wh_aperture_sat]) >0:
                    disk_intercept_sat_spot_bool = True
                    if not quiet: 
                        print(filename_here, 'removed because of the sat spot #'+str(sat_spot_number))
                    save_sat_spots[sat_spot_number,index_angle,index_wl,:,:] *= np.nan
                else:
                    # save the sat spot centered on a square 

                    # Crop around the sat and center the sat spot on the square
                    spot_square = image_here[int(round(center_sat_y)) - half_size_square :  int(round(center_sat_y)) + half_size_square +1,int(round(center_sat_x)) - half_size_square :  int(round(center_sat_x)) + half_size_square+1]
                    
                    # sub-pixel shift to center the sat spot on [half_size_square, half_size_square] pixel
                    
                    # interpol.shift does not work well with nans, so if my square reach the edge I replace them by zeros, then shift, then put the nans back.
                    # This is ~ok because this is a sub pixel shift and because the nans are usually the edge of the IFS slice.
                    # this is not perfect, so for that reasons, avoid taking half_size_square too large
                    wh_spot_square_nan = np.where(spot_square != spot_square)
                    spot_square[wh_spot_square_nan] = 0.
                    spot_square = interpol.shift(spot_square, ( int(round(center_sat_y)) - center_sat_y,int(round(center_sat_x))- center_sat_x))
                    spot_square[wh_spot_square_nan] = np.nan
                    
                    # now that we know that the disk is not on this sat spot, we remove the disk
                    crop_model_mask_rot = model_mask_rot[int(round(center_sat_y)) - half_size_square :  int(round(center_sat_y)) + half_size_square +1,int(round(center_sat_x)) - half_size_square :  int(round(center_sat_x)) + half_size_square+1]
                    spot_square[np.where(crop_model_mask_rot == 1)] = np.nan
                    
                    # if there is a bright zone in the IFS slice, we also remove it
                    if sequence == '150403_K1_Spec' or sequence == '150403_K2_Spec' :
                        crop_mask_triangle1 = mask_triangle1[int(round(center_sat_y)) - half_size_square :  int(round(center_sat_y)) + half_size_square +1,int(round(center_sat_x)) - half_size_square :  int(round(center_sat_x)) + half_size_square+1]
                        crop_mask_triangle2 = mask_triangle2[int(round(center_sat_y)) - half_size_square :  int(round(center_sat_y)) + half_size_square +1,int(round(center_sat_x)) - half_size_square :  int(round(center_sat_x)) + half_size_square+1]
                        spot_square = spot_square*crop_mask_triangle1*crop_mask_triangle2

                    save_sat_spots[sat_spot_number,index_angle,index_wl,:,:] =spot_square

                # if one on the sat spots of one of the slice of a datacube intersects the disk, we will remove this datacube completely, 
                # so there is no use measuring the normalization for any of them
                if disk_intercept_sat_spot_bool == False:
                    # fits.writeto( "/Users/jmazoyer/Desktop/sat.fits",save_sat_spots[sat_spot_number,index_angle,index_wl,:,:] , overwrite=True)
                    
                    sat_spot_here = save_sat_spots[sat_spot_number,index_angle,index_wl,:,:]
                    
                    # # You can try to high pass filter the sat spots but this is usually very aggressive on the sat spots so I do not recommand it
                    # sat_spot_here = klip.high_pass_filter(sat_spot_here, 2*radius_aperture) 

                    # define the zoen for aperture and noise
                    wh_aperture_square = np.where( (rho2d_square < radius_aperture) & (sat_spot_here == sat_spot_here) )
                    wh_noise_square = np.where((rho2d_square < r_out_ring_noise) & (rho2d_square > r_in_ring_noise) & (sat_spot_here == sat_spot_here) )
                
                    # measure aperture flux for this sat spot.  Sum on the aperture - the noise 
                    ### we "np.clip" here to be sure that we do not include the negative first ring in the aperture
                    mean_noise = np.nanmean(sat_spot_here[wh_noise_square])
                    value_sat_spot_image_here[sat_spot_number] = np.nansum(np.clip(sat_spot_here[wh_aperture_square] - mean_noise,a_min = 0., a_max = None))                
                    
                    stdnoise = np.nanstd(sat_spot_here[wh_noise_square])
                    save_SNR_sat_spots[sat_spot_number, index_angle,index_wl] = np.nanmean(np.clip(sat_spot_here[wh_aperture_square] -mean_noise,a_min = 0., a_max = None))/stdnoise
                    
                else:
                    save_SNR_sat_spots[sat_spot_number, index_angle,index_wl] = np.nan
                    value_sat_spot_image_here[sat_spot_number] = np.nan

            # Now that we have the value, we make the mean for the 4 sat spots
            value_sat_spot[index_wl,index_angle] = np.nanmean(value_sat_spot_image_here)

            # we normalize the data 
            if value_sat_spot[index_wl,index_angle] > 0.:
                data_in_the_fits[index_wl] = data_in_the_fits[index_wl]/value_sat_spot[index_wl,index_angle]

            if sequence == '150403_K1_Spec' or sequence == '150403_K2_Spec' :
                data_in_the_fits[index_wl] = data_in_the_fits[index_wl]*mask_triangle1*mask_triangle2
            
            ### Just a test if you want to check where are the sat spots and the disk
            # data_in_the_fits[index_wl] = data_in_the_fits[index_wl]*(1-model_mask_rot)*(1-test_satspot)
            # fits.writeto("/Users/jmazoyer/Desktop/titi.fits", data_in_the_fits, overwrite=True)
                
        filenamewithouextension, extension = os.path.splitext(filenames[initial_wl_number * index_angle])
        if disk_intercept_sat_spot_bool == True:
            excluded_files.append(filenames[initial_wl_number * index_angle])
            
        # if disk_intercept_sat_spot_bool == False:
            
        #     header0[index_angle]['history'] = ''
        #     header0[index_angle]['history'] = 'To remove the PSF we remove {0} slices'.format(removed_slices)
        #     header0[index_angle]['history'] = ''
        #     header0[index_angle]['history'] = 'Normalized for disk with normalize_image_for_disk_spectra.py'
        #     header0[index_angle]['history'] = 'with a sat spot aperture of radius {0:.2f}*wl/1.6 pix'.format(radius_aperture_at_1point6)
        #     header0[index_angle]['history'] = 'we removed the noise measured on a ring of {0:.2f}*wl/1.6 to {1:.2f}*wl/1.6 pix'.format(r_in_ring_noise_at_1point6, r_out_ring_noise_at_1point6)
        #     header0[index_angle]['history'] = ''

        #     # dataset.input = data_in_the_fits
        #     # # PAs = dataset.PAs
        #     # # Wavelengths = dataset.wvs
        #     # # Starpos = dataset.centers
        #     # # filenames = dataset.filenames
        #     # # header0 = dataset.prihdrs
        #     # # header1 = dataset.exthdrs

        #     # dataset.output_wcs = dataset.wcs
        #     # dataset.output_centers = dataset.centers
        #     # dataset.savedata(filepath= filenamewithouextension +'_normalized.fits', data= dataset.input )


        #     new_fits = fits.HDUList()
        #     new_fits.append(fits.ImageHDU(data=None, header=header0[index_angle]))
        #     new_fits.append(fits.ImageHDU(data=data_in_the_fits, header=header1[index_angle]))
            
        #     new_fits.writeto(filenamewithouextension +'_normalized.fits', overwrite=True)
        #     new_fits.close()
        # else:
        #     if os.path.isfile(filenamewithouextension +'_normalized.fits'):
        #         os.remove(filenamewithouextension +'_normalized.fits')



    ## save all sat spots raw format

    if SaveAll == True:
        hdr = fits.Header()
        for excluded_filesi in excluded_files: 
            hdr['history'] = 'disk intersect sat spot: we remove in {0}'.format(excluded_filesi)
        
        fits.writeto(basedir+sequence +  "_sat_spots_all.fits",save_sat_spots ,hdr, overwrite=True)
    

    # four sat spots mean 
    mean_save_SNR_sat_spots = np.nanmean(save_SNR_sat_spots, axis = 0)
    mean_save_sat_spots = np.nanmean(save_sat_spots, axis = 0)
    ## mean over the sequence
    mean_save_SNR_sat_spots_per_wl = np.nanmean(mean_save_SNR_sat_spots, axis = 0)
    mean_save_sat_spots_per_wl = np.nanmean(mean_save_sat_spots, axis = 0)


    for index_wl in range(0, initial_wl_number):
        if (removed_slices is not None) and (index_wl in removed_slices):
            mean_save_sat_spots_per_wl[index_wl,:,:] = np.nan*mean_save_sat_spots_per_wl[0,:,:]
            mean_save_SNR_sat_spots_per_wl[index_wl] *= np.nan
        else:
            sat_spot_here = mean_save_sat_spots_per_wl[index_wl,:,:]

            ## just a small test to check the size of the aperture and clean the PSF a little
            radius_aperture = radius_aperture_at_1point6/(1.6)*np.unique(Wavelengths)[index_wl]
            r_in_ring_noise = r_in_ring_noise_at_1point6/(1.6)*np.unique(Wavelengths)[index_wl]
            r_out_ring_noise = r_out_ring_noise_at_1point6/(1.6)*np.unique(Wavelengths)[index_wl]
            wh_aperture_square = np.where( (rho2d_square < radius_aperture) )
            wh_noise_square = np.where((rho2d_square < r_out_ring_noise) & (rho2d_square > r_in_ring_noise))

            mask_aperture = np.zeros((2*half_size_square +1,2*half_size_square +1))
            mask_noise = np.zeros((2*half_size_square +1,2*half_size_square +1))
            mask_aperture[wh_aperture_square] = 1.
            mask_noise[wh_noise_square] = 1.

            sat_spot_here = np.clip(sat_spot_here - np.mean(sat_spot_here[wh_noise_square]),a_min = 0., a_max = None)
            # sat_spot_here = klip.high_pass_filter(sat_spot_here, 3*radius_aperture)
            # sat_spot_here -= np.nanmin(sat_spot_here) 

            smooth_mask = np.ones((2*half_size_square +1,2*half_size_square +1))
            smooth_mask[np.where(rho2d_square > r_out_ring_noise-1)] = 0.
            smooth_mask = scipy_filters.gaussian_filter(smooth_mask, 2.)
            smooth_mask[np.where(rho2d_square < r_in_ring_noise)] = 1.
            smooth_mask[np.where(smooth_mask < 0.01)] = 0.

            # sat_spot_here *= mask_aperture + mask_noise
            
            sat_spot_here = sat_spot_here*smooth_mask
            sat_spot_here = sat_spot_here/np.nanmax(sat_spot_here)
            mean_save_sat_spots_per_wl[index_wl,:,:] = sat_spot_here


    hdr = fits.Header()
    hdr['CD3_3'] = header_anglei['CD3_3']
    hdr['CRPIX3'] = header_anglei['CRPIX3']
    hdr['CRVAL3'] = header_anglei['CRVAL3']
    hdr['CTYPE3'] = header_anglei['CTYPE3']
    hdr['CUNIT3'] = header_anglei['CUNIT3']
    hdr['history'] = 'To measure the PSF we remove {0} slices'.format(removed_slices)
    if SavePSF == True:
        fits.writeto(basedir+ name_psf + ".fits",np.nanmean(mean_save_sat_spots_per_wl, axis = 0) ,hdr, overwrite=True)
    if SaveAll == True:
        fits.writeto(basedir+ name_psf + "_4spotaverage_Seqaveraged.fits",mean_save_sat_spots_per_wl ,hdr, overwrite=True)

    ## save the sat spots values
    # fits.writeto(basedir+sequence + "_value_sat_spot.fits",np.nanmean(value_sat_spot,axis =1) , overwrite=True)

    # filelist_end = glob.glob(basedir + "*_normalized.fits")
    nb_excluded = len(excluded_files)
    print(sequence + ': We remove ' + str(nb_excluded) +' files out of '+ str(nb_init) +' because sat spots intersected the disk')

    print('for the selected slices, the mean SNR for the sat spots are at each wl', mean_save_SNR_sat_spots_per_wl)
    if np.all(removed_slices) == None:
        print(sequence + ": All Wl used to measure the PSF")
    else:
        print(sequence + ": We cut WL {0} to measure the PSF ".format(removed_slices))

    return excluded_files
