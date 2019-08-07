"""Insert model into image
author: johan mazoyer
"""

import os
import glob
import distutils.dir_util
import numpy as np
import math as mt
import astropy.io.fits as fits

import scipy.ndimage.interpolation as interpol
from astropy.convolution import convolve
from scipy.integrate import quad


import pyklip.instruments.GPI as GPI
import pyklip.klip as klip
import pyklip.parallelized as parallelized


from check_gpi_satspots import check_gpi_satspots
from check_gpi_satspots import gpi_satspots_emptydataset




########################################################
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

    distance = 72 

    #generate the model
    model = mt.exp(norm)*gen_disk_dxdy_2g(R1=r1,R2=r2, beta=beta, aspect_ratio=0.01, g1=g1, g2=g2,alpha=alpha, inc=inc, pa=pa, distance=distance, dx=dx, dy=dy, mask = wheremask2generatedisk) #+ offset

    return model

########################################################
########################################################
def gen_disk_dxdy_2g(R1=74.42, R2=82.45, beta=1.0, aspect_ratio=0.1, g1=0.6, g2=-0.6, alpha=0.7, inc=76.49, pa=30, distance=72.8, psfcenx=140,psfceny=140, sampling=1, mask=None, dx=0, dy=0.):

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
    pixscale = 0.01414 #GPI's pixel scale
    dim=281. #for now the GPI dimensions are hardcoded
    
    max_fov = dim/2.*pixscale #maximum radial distance in AU from the center to the edge
    npts=int(np.floor(dim/sampling))
    xsize = max_fov*distance #maximum radial distance in AU from the center to the edge

    #The coordinate system here [x,y,z] is defined :
    # +ve x is the line of sight 
    # +ve y is going right from the center
    # +ve z is going up from the center

    # y = np.linspace(0,xsize,num=npts/2)
    y = np.linspace(-xsize,xsize,num=npts)
    z = np.linspace(-xsize,xsize,num=npts)

    #Only need to compute half the image
    # image =np.zeros((npts,npts/2+1))
    image =np.zeros((npts,npts))

    #Some things we can precompute ahead of time
    maxe = mt.log(np.finfo('f').max) #The log of the machine precision
    
    #Inclination Calculations
    incl = np.radians(90-inc)
    ci = mt.cos(incl) #Cosine of inclination
    si = mt.sin(incl) #Sine of inclination
    
    #Position angle calculations
    pa_rad=np.radians(90-pa) #The position angle in radians
    cos_pa=mt.cos(pa_rad) #Calculate these ahead of time
    sin_pa=mt.sin(pa_rad)
    
    #HG g value squared
    g1_2 = g1*g1 #First HG g squared
    g2_2 = g2*g2 #Second HG g squared
    #Constant for HG function
    k=1./(4*np.pi)

    #The aspect ratio
    a_r=aspect_ratio

    #Henyey Greenstein function at 90
    hg1_90=k*alpha*(1. - g1_2)/(1. + g1_2)**1.5
    hg2_90=k*(1-alpha)*(1. - g2_2)/(1. + g2_2)**1.5

    hg_90 = hg1_90+hg2_90

    #If there's no mask then calculate for the full image
    if len(np.shape(mask)) < 2:

        for i,yp in enumerate(y):
            for j,zp in enumerate(z):

                #This rotates the coordinates in the image frame
                yy=yp*cos_pa - zp * sin_pa  #Rotate the y coordinate by the PA
                zz=yp*sin_pa + zp * cos_pa  #Rotate the z coordinate by the PA

                #The distance from the center (in each coordinate) squared
                y2 = yy*yy
                z2 = zz*zz

                #This rotates the coordinates in and out of the sky
                zpci=zz*ci #Rotate the z coordinate by the inclination. 
                zpsi=zz*si
                #Subtract the offset
                zpsi_dx = zpsi - dx

                #The distance from the offset squared
                yy_dy=yy-dy
                yy_dy2=yy_dy*yy_dy

                image[j,i]=  quad(integrand_dxdy_2g, -R2, R2, epsrel=0.5e-3,limit=75,args=(yy_dy2,y2,zp,z2,zpsi_dx,zpci,R1,R2,beta,a_r,g1,g1_2,g2,g2_2, alpha,ci,si,maxe,dx,dy,k))[0]
            

    #If there is a mask then don't calculate disk there
    else: 
        hmask = mask
        # hmask = mask[:,140:] #Use only half the mask

        for i,yp in enumerate(y):
            for j,zp in enumerate(z):
                
                # if hmask[j,npts/2+i]: #This assumes that the input mask has is the same size as the desired image (i.e. ~ size / sampling)
                if hmask[j,i]:

                   image[j,i] = 0.#np.nan
                
                else:

                    #This rotates the coordinates in the image frame
                    yy=yp*cos_pa - zp * sin_pa  #Rotate the y coordinate by the PA
                    zz=yp*sin_pa + zp * cos_pa  #Rotate the z coordinate by the PA

                    #The distance from the center (in each coordinate) squared
                    y2 = yy*yy
                    z2 = zz*zz

                    #This rotates the coordinates in and out of the sky
                    zpci=zz*ci #Rotate the z coordinate by the inclination. 
                    zpsi=zz*si
                    #Subtract the offset
                    zpsi_dx = zpsi - dx

                    #The distance from the offset squared
                    yy_dy=yy-dy
                    yy_dy2=yy_dy*yy_dy

                    image[j,i]=  quad(integrand_dxdy_2g, -R2, R2, epsrel=0.5e-3,limit=75,args=(yy_dy2,y2,zp,z2,zpsi_dx,zpci,R1,R2,beta,a_r,g1,g1_2,g2,g2_2, alpha,ci,si,maxe,dx,dy,k))[0]

    # print("Running time: ", datetime.now()-starttime)

    # # normalize the HG function by the width
    image = image/a_r

    # normalize the HG function at the PA
    image = image/hg_90

    return image

########################################################
########################################################
def integrand_dxdy_2g(xp,yp_dy2,yp2,zp,zp2,zpsi_dx,zpci,R1,R2,beta,a_r,g1,g1_2,g2,g2_2, alpha, ci, si,maxe,dx, dy,k):

    # compute the scattering integrand
    # see analytic-disk.nb

    xx=(xp * ci + zpsi_dx)

    d1 = mt.sqrt((yp_dy2 + xx*xx))

    if (d1  < R1 or d1 > R2):
        return 0.0

    d2 = xp*xp + yp2 + zp2

    #The line of sight scattering angle
    cos_phi=xp/mt.sqrt(d2)
    # phi=np.arccos(cos_phi)

    #Henyey Greenstein function
    hg1=k*alpha*(1. - g1_2)/(1. + g1_2 - (2*g1*cos_phi))**1.5
    hg2=k*(1-alpha)*(1. - g2_2)/(1. + g2_2 - (2*g2*cos_phi))**1.5

    hg = hg1+hg2

    #Radial power low r propto -beta
    int1 = hg*(R1/d1)**beta

    #The scale height function
    zz = (zpci - xp*si)
    hh = (a_r*d1)
    expo = zz*zz/(hh*hh)

    # if expo > 2*maxe:   # cut off exponential after 28 e-foldings (~ 1E-06)
    #     return 0.0 

    int2 = np.exp(0.5*expo) 
    int3 = int2 * d2

    return int1/int3

########################################################
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


########################################################
########################################################


emptyStardir = '/Users/jmazoyer/Dropbox/STSCI/python/python_data/KlipFM_for_SPF/hd_48524/20180128_H_Spec_subset4injection/'


save_dir = '/Users/jmazoyer/Dropbox/ExchangeFolder/data_python/Aurora/20180128_H_Spec_injection/'

removed_slices = None
# measure the PSF from the satspots and identify angles where the disk intersect the satspots
excluded_files = gpi_satspots_emptydataset(emptyStardir, SavePSF=True, name_psf= 'emptydataset_SatSpotPSF', SaveAll=False)

psf = fits.getdata(emptyStardir +'emptydataset_SatSpotPSF.fits')


theta_here = (np.log(74.5),np.log(100),12.4, 0.825, -0.201, 0.298, np.cos(np.radians(76.8)), 26.64, -2.0,0.94, np.log(80))

#generate the model
model_non_convolved =call_gen_disk_2g(theta_here, None)

model_here_convolved = convolve(model_non_convolved,psf, boundary = 'wrap')
fits.writeto(save_dir+ 'Fake_Injected_hr4796_convolved.fits', model_here_convolved, overwrite=True)




filelist = glob.glob(emptyStardir + '*spdc_distorcorr.fits')

dataset = GPI.GPIData(filelist, quiet=True)
PAs = dataset.PAs
Starpos = dataset.centers
filenames = dataset.filenames
header0 = dataset.prihdrs
header1 = dataset.exthdrs
fluxspots = dataset.spot_flux



for index_angle in range(0, len(filelist)):

    new_fits = fits.HDUList()

    data_in_the_fits = fits.getdata(filenames[37 * index_angle], 1)
    for index_wl in range(0, 37):
        modelrot = klip.rotate(model_here_convolved, PAs[37 * index_angle + index_wl], [140, 140], \
                               [Starpos[37 * index_angle + index_wl, 0], Starpos[37 * index_angle + index_wl, 1]])
        modelrot[np.where(np.isnan(modelrot))] = 0.
        data_in_the_fits[index_wl] += modelrot

    new_fits.append(fits.ImageHDU(data=None, header=header0[index_angle]))
    new_fits.append(fits.ImageHDU(data=data_in_the_fits, header=header1[index_angle]))
    
    filenamewithouextension=os.path.basename(filenames[37 * index_angle])

    new_fits.writeto( save_dir+filenamewithouextension,overwrite=True)

    new_fits.close()


filelist = glob.glob(save_dir + "*_distorcorr.fits")

dataset = GPI.GPIData(filelist, quiet=True)
dataset.OWA = 98
mov_here = 6
KLMODE = [5] 

header0 = dataset.prihdrs
starname = header0[0]['OBJECT'].replace(" ", "_")

reduc = 'ADI_only/'
resultdir = save_dir + reduc
distutils.dir_util.mkpath(resultdir)


dataset.spectral_collapse(align_frames=True)

parallelized.klip_dataset(dataset, outputdir=resultdir,
                          fileprefix=starname + "_adi_injected_fakehr4796", mode='ADI', annuli=1, subsections=1,
                          minrot=mov_here, numbasis=KLMODE, calibrate_flux=False, highpass=False, lite=False,
                          save_aligned=False, aligned_center=[140, 140], maxnumbasis=100)

filename = starname + "_adi_injected_fakehr4796" + '-KLmodes-all.fits'

reduce_data = fits.getdata(resultdir + filename, 1)[0]


#measure the noise Wahhaj trick
dataset.PAs = -dataset.PAs
parallelized.klip_dataset(dataset, numbasis=KLMODE, maxnumbasis=len(filelist), annuli=1, subsections=1, mode='ADI',
                            outputdir=resultdir, fileprefix=starname + '_WahhajTrick',
                            aligned_center=[140, 140], highpass=False, minrot=mov_here, calibrate_flux=False)

reduced_data_wahhajtrick = fits.getdata(resultdir + starname + '_WahhajTrick-KLmodes-all.fits')[0]
noise = make_noise_map_no_mask(reduced_data_wahhajtrick, xcen=140, ycen=140, delta_raddii=3)
noise[np.where(noise == 0)] = np.nan

dataset.PAs = -dataset.PAs
os.remove(resultdir + starname + '_WahhajTrick-KLmodes-all.fits')

SNR_Map = np.zeros((281, 281))
wh_zero_noise = np.where(noise != 0)
SNR_Map[wh_zero_noise] = reduce_data[wh_zero_noise] / noise[wh_zero_noise]
SNR_Map[np.where(SNR_Map < 0)] = 0.

fits.writeto( resultdir + starname+ '_SNRmap.fits', SNR_Map, overwrite=True)


reduce_data = fits.getdata('/Users/jmazoyer/Dropbox/ExchangeFolder/data_python/Aurora/160318_H_Spec/klip_fm_files/Hband_hr4796_klipFM_KL5_Mov6_realPSF-klipped-KLmodes-all.fits', 1)[0]
noise = fits.getdata('/Users/jmazoyer/Dropbox/ExchangeFolder/data_python/Aurora/160318_H_Spec/klip_fm_files/Hband_hr4796_klipFM_KL5_Mov6_realPSF_noisemap.fits')

SNR_Map = np.zeros((281, 281))
wh_zero_noise = np.where(noise != 0)
SNR_Map[wh_zero_noise] = reduce_data[wh_zero_noise] / noise[wh_zero_noise]
SNR_Map[np.where(SNR_Map < 0)] = 0.

fits.writeto( resultdir + starname+ '_SNRmap_realdata.fits', SNR_Map, overwrite=True)



