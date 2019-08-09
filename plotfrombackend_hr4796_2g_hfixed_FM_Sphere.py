####### This is the MCMC plotting code for HR 4796 data #######

quality_plot = 4


import os, glob
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

import pyklip.instruments.Instrument as Instrument
import pyklip.parallelized as parallelized
from pyklip.fmlib.diskfm import DiskFM
import pyklip.fm as fm


from kowalsky import kowalsky

import matplotlib
matplotlib.use('Agg')  ### I dont use matplotlib but I notice there is a conflict when I import matplotlib with pyklip if I don't use this line

import corner
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# from emcee import EnsembleSampler
from emcee import backends



warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

########################################################
########################################################
#### Fonctions extraite de anadisk
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

def gen_disk_dxdy_2g(R1=74.42, R2=82.45, beta=1.0, aspect_ratio=0.1, g1=0.6, g2=-0.6, alpha=0.7, inc=76.49, pa=30, distance=72.8, sampling=1, mask=None, dx=0, dy=0.):

    # starttime=datetime.now()

    #SPHERE FOV stuff
    # pixscale = 0.01414 #GPI's pixel scale
    # pixscale  = 0.012255 #SPHERE's IRDIS pixel scale
    # nspaxels = 200 #The number of lenslets along one side 
    # fov_square = pixscale*nspaxels #The square FOV of GPI 
    # max_fov = mt.sqrt(2)*fov_square/2 #maximum FOV in arcseconds from the center to the edge
    
    #SPHERE FOV stuff
    pixscale  = 0.012255 #SPHERE's IRDIS pixel scale
    dim=281. 
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
def make_noise_map(reduced_data, PAs, mask_object_astro_zeros, xcen=140., ycen=140., delta_raddii=3):

    dim = mask_object_astro_zeros.shape[0]
    # create rho2D for the rings
    x = np.arange(dim, dtype=np.float)[None,:] - xcen
    y = np.arange(dim, dtype=np.float)[:,None] - ycen
    rho2d = np.sqrt(x**2 + y**2)

    total_paralax = np.max(PAs) - np.min(PAs)

    # create nan and zeros masks for the disk    
    mask_disk_all_angles_nans = np.ones((dim, dim))
    
    anglesrange = PAs - np.median(PAs)
    anglesrange = np.append(anglesrange,[0])

    for index_angle, PAshere in enumerate(anglesrange):
        rot_disk = np.round(interpol.rotate(mask_object_astro_zeros, PAshere, reshape=False, mode='wrap'))
        mask_disk_all_angles_nans[np.where(rot_disk == 0)] = np.nan
    
    
    noise_map = np.zeros((dim, dim))
    for i_ring in range(0, int(np.floor(xcen / delta_raddii)) - 2):
        image_masked = reduced_data * mask_disk_all_angles_nans
        wh_rings = np.where((rho2d >= i_ring * delta_raddii) &
                        (rho2d < (i_ring + 1) * delta_raddii))
        noise_map[wh_rings] = np.nanstd(image_masked[wh_rings])
    return noise_map

########################################################
def crop_center_even(img,crop):
    y,x = img.shape
    startx = x//2 - crop//2
    starty = y//2 - crop//2    
    return img[starty:starty+crop, startx:startx+crop]

def crop_center_odd(img,crop):
    y,x = img.shape
    startx = (x-1)//2 - crop//2
    starty = (y-1)//2 - crop//2    
    return img[starty:starty+crop, startx:startx+crop]

########################################################
def offset_2_RA_dec(dx_ml,dy_ml, inc_ml, pa_ml, dist_star):
    ## from offset in AU in the disk plane (returned by the MCMC), 
    ## return right ascension and declination of the ellipse centre with respect to the star location
    dist_star = 72.
    dx_disk_arcsec = dx_ml*np.cos(np.radians(inc_ml))/dist_star*1000
    dy_disk_arcsec= -dy_ml/dist_star*1000

    dx_sky = np.cos(np.radians(pa_ml))*dx_disk_arcsec - np.sin(np.radians(pa_ml))*dy_disk_arcsec
    dy_sky = np.sin(np.radians(pa_ml))*dx_disk_arcsec + np.cos(np.radians(pa_ml))*dy_disk_arcsec

    dAlpha = -dx_sky
    dDelta = dy_sky

    return dAlpha,dDelta


#####################################################
############### Start of the fitting ################
#####################################################

#MCMC Parameters
nwalkers = 192
ndim     = 11 #Number of dimensions

newdim = 281

#The PSF centers
xcen= newdim/2 #np.float(rstokes.extheader['PSFCENTX'])
ycen= newdim/2 #np.float(rstokes.extheader['PSFCENTY'])



if socket.gethostname()== 'MT-101942':
    datadir='/Users/jmazoyer/Dropbox/ExchangeFolder/data_python/Aurora/SPHERE_Hdata/' 
else:
    datadir='/home/jmazoyer/data_python/Aurora/SPHERE_Hdata/'   

file_prefix = 'Sphere_hr4796_klipFM_MaxiMaxiNoise'

klipdir = datadir + 'klip_fm_files/'
distutils.dir_util.mkpath(klipdir)
mcmcresultdir = datadir + 'results_MCMC/'
distutils.dir_util.mkpath(mcmcresultdir)


# read the backend in h5

name_h5 = file_prefix + 'backend_file_mcmc'
filename = mcmcresultdir + name_h5 +'.h5'

reader = backends.HDFBackend(filename)
tau = reader.get_autocorr_time(tol=0)

print("")
print("")
print(name_h5)
print("# of iteration in the backend chain initially: {0}".format(reader.iteration))
print("Max Tau times 50: {0}".format(50*np.max(tau)))
print("")

burnin = 1000
thin = 1



chain = reader.get_chain(discard=burnin, thin=thin)
chain_flat = reader.get_chain(discard=burnin, flat =True, thin=thin)
log_prob_samples_flat = reader.get_log_prob(discard=burnin,flat =True, thin=thin)


print("")
print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("chain shape: {0}".format(chain.shape))
print("chain flat shape: {0}".format(chain_flat.shape))
print("flat log prob shape: {0}".format(log_prob_samples_flat.shape))

print("best likelyhood: {0}".format(np.max(log_prob_samples_flat)))

#######################################################
#################### Make plot 1  #####################
#######################################################
#######################################################
# Plot the chain values

chain4plot = reader.get_chain(discard=0, thin=1)

## change log and arccos values to physical
chain4plot[:,:,0]=np.exp(chain4plot[:,:,0])
chain4plot[:,:,1]=np.exp(chain4plot[:,:,1])
chain4plot[:,:,6]=np.degrees(np.arccos(chain4plot[:,:,6]))
chain4plot[:,:,10]=np.exp(chain4plot[:,:,10])

## change g1, g2 and alpha to percentage
chain4plot[:,:,3]=100*chain4plot[:,:,3]
chain4plot[:,:,4]=100*chain4plot[:,:,4]
chain4plot[:,:,5]=100*chain4plot[:,:,5]


f, axarr=plt.subplots(ndim, sharex=True, figsize = (6.4*quality_plot, 4.8*quality_plot))

labels=["R1[AU]","R2[AU]",r"$\beta$", "g1[%]","g2[%]",r"$\alpha$[%]", r"$i[^{\circ}]$", r"$pa[^{\circ}]$",'dx[AU]', 'dy[AU]',"N[ADU]"]

for i in range(ndim):
    axarr[i].set_ylabel(labels[i],fontsize=5*quality_plot)
    axarr[i].tick_params(axis='y',   labelsize=4*quality_plot)
    

    for j in range(nwalkers):
        axarr[i].plot(chain4plot[:,j,i],linewidth=quality_plot)

    axarr[i].axvline(x=burnin, color = 'black',linewidth=1.5*quality_plot)

axarr[ndim -1].tick_params(axis='x',   labelsize=6*quality_plot)
axarr[ndim -1].set_xlabel('Iterations',fontsize=10*quality_plot)

plt.savefig(mcmcresultdir+name_h5+'_chains.jpg')


# ####################################################################
# ### #Plot the PDFs
# ####################################################################

chain[:,:,0]=np.exp(chain[:,:,0])
chain[:,:,1]=np.exp(chain[:,:,1])
chain[:,:,6]=np.degrees(np.arccos(chain[:,:,6]))
chain[:,:,10]=np.exp(chain[:,:,10])

## change g1, g2 and alpha to percentage
chain[:,:,3]=100*chain[:,:,3]
chain[:,:,4]=100*chain[:,:,4]
chain[:,:,5]=100*chain[:,:,5]

samples=chain[:,:].reshape(-1,ndim)

matplotlib.rcParams['axes.labelsize'] = 19
matplotlib.rcParams['axes.titlesize'] = 14

matplotlib.rcParams['xtick.labelsize'] = 13
matplotlib.rcParams['ytick.labelsize'] = 13

### Check truths = bests parameters
fig=corner.corner(samples, labels=labels, 
	quantiles=(0.159, 0.5, 0.841),show_titles=True, plot_datapoints=False, verbose=False)# levels=(1-np.exp(-0.5),) , ))
    
### cumulative percentiles 
### value at 50% is the center of the Normal law 
### value at 50% - value at 15.9% is -1 sigma
### value at 84.1%% - value at 50% is 1 sigma

fig.subplots_adjust(hspace=0)
fig.subplots_adjust(wspace=0)


fig.gca().annotate("SPHERE H2 band, {0:,} iterations (192 walkers): {1:,} models".format(reader.iteration, reader.iteration*192),
                      xy=(0.55, 0.99), xycoords="figure fraction",
                      xytext=(-20, -10), textcoords="offset points",
                      ha="center", va="top", fontsize=44)
                      
plt.savefig(mcmcresultdir+name_h5+'_pdfs.pdf')


####################################################################
### Best likelyhood model and MCMC errors
####################################################################

#### measure of 5 other parameters:  right ascension, declination, and Kowalsky (true_a, true_ecc, longnode)
dist_star = 72.
pixscale  = 0.012255 #SPHERE's IRDIS pixel scale

other_samples = np.zeros((samples.shape[0],6))

for j in range(samples.shape[0]):
    r1_here = samples[j,0]
    inc_here = samples[j,6]
    pa_here = samples[j,7]
    dx_here = samples[j,8]
    dy_here = samples[j,9]


    dAlpha, dDelta = offset_2_RA_dec(dx_here, dy_here, inc_here, pa_here, dist_star)
    
    
    other_samples[j,0] = dAlpha
    other_samples[j,1] = dDelta
    
    semimajoraxis = r1_here/dist_star*1000
    ecc = np.sin(np.radians(inc_here))

    true_a, true_ecc, argperi, inc, longnode = kowalsky(semimajoraxis,ecc, pa_here,dAlpha,dDelta)

    other_samples[j,2] = true_a
    other_samples[j,3] = true_ecc
    other_samples[j,4] = longnode
    other_samples[j,5] = argperi


MLval_mcmc_val_mcmc_err = np.zeros((ndim+6, 4))
names=["R1","R2","Beta", "g1","g2","Alpha", "inc", "PA",'dx', 'dy',\
    "N", "RA","Decl","a","ecc","Omega", "ARGPE" ]

comments = ['AU, inner radius', 'AU, outer radius', 'radial power law', '%, 1st HG param', '%, 2nd HG param', '%, relative HG weight', 'degree, inclination',\
    'degree, principal angle', 'AU, + -> NW offset disk plane Minor Axis','+ -> SW offset disk plane Major Axis', 'ADU, normalisation',\
    'mas, ->E right ascension (dalpha in Milli+2017)', 'mas, ->N declination (ddelta in Milli+2017)','mas, deproj. (true) semi major axis (Kowalsky)',\
    'deproj. (true) eccentricity (Kowalsky)', 'deg, longitude of the ascending node (Kowalsky)', 'deg, argument of pericentre (Kowalsky)']
wheremin = np.where(log_prob_samples_flat == np.max(log_prob_samples_flat))
wheremin0 = np.array(wheremin).flatten()[0]

for i in range(ndim):
    MLval_mcmc_val_mcmc_err[i,0] = samples[wheremin0, i]
    percent = np.percentile(samples[:, i], [15.9, 50, 84.1])
    MLval_mcmc_val_mcmc_err[i,1] = percent[1]
    MLval_mcmc_val_mcmc_err[i,2] = percent[0] - percent[1]
    MLval_mcmc_val_mcmc_err[i,3] = percent[2] - percent[1]



for i in range(6):
    MLval_mcmc_val_mcmc_err[i+ndim,0] = other_samples[wheremin0, i]
    percent = np.percentile(other_samples[:, i], [15.9, 50, 84.1])
    MLval_mcmc_val_mcmc_err[i+ndim,1] = percent[1]
    MLval_mcmc_val_mcmc_err[i+ndim,2] = percent[0] - percent[1]
    MLval_mcmc_val_mcmc_err[i+ndim,3] = percent[2] - percent[1]
print(" ")


for i in range(ndim +6):
    print(names[i]+' ML: {0:.3f}, MCMC {1:.3f}, -/+1sig: {2:.3f}/+{3:.3f} '.format(MLval_mcmc_val_mcmc_err[i,0], MLval_mcmc_val_mcmc_err[i,1], \
        MLval_mcmc_val_mcmc_err[i,2],MLval_mcmc_val_mcmc_err[i,3])+ comments[i])

    if i == 12:
        mas_2_pix = 1/(1000.*pixscale)
        print('RA_pix ML: {0:.3f}, MCMC {1:.3f}, -/+1sig: {2:.3f}/+{3:.3f} '.format(MLval_mcmc_val_mcmc_err[11,0]*mas_2_pix, MLval_mcmc_val_mcmc_err[11,1]*mas_2_pix, \
            MLval_mcmc_val_mcmc_err[11,2]*mas_2_pix,MLval_mcmc_val_mcmc_err[11,3]*mas_2_pix)+ 'pix, ->E right ascension')
        print('Decl_pix ML: {0:.3f}, MCMC {1:.3f}, -/+1sig: {2:.3f}/+{3:.3f} '.format(MLval_mcmc_val_mcmc_err[12,0]*mas_2_pix, MLval_mcmc_val_mcmc_err[12,1]*mas_2_pix, \
            MLval_mcmc_val_mcmc_err[12,2]*mas_2_pix,MLval_mcmc_val_mcmc_err[12,3]*mas_2_pix)+ 'pix, ->N Declination')


hdr = fits.Header()
hdr['COMMENT'] = 'Best model of the MCMC reduction'
hdr['COMMENT'] = 'PARAM_ML are the parameters producing the best LH'
hdr['COMMENT'] = 'PARAM_MM are the parameters at the 50% percentile in the MCMC'
hdr['COMMENT'] = 'PARAM_M and PARAM_P are the -/+ sigma error bars (16%, 84%)'
hdr['PKL_FILE'] = name_h5
hdr['FITSDATE'] = str(datetime.now())
hdr['BURNIN'] = burnin
hdr['THIN'] = thin

hdr['TOT_ITER'] = reader.iteration

hdr['n_walker'] = nwalkers
hdr['n_param'] = ndim


hdr['MAX_LH'] = (np.max(log_prob_samples_flat),'Max likelyhood, obtained for the ML parameters')

for i in range(ndim +6):
    hdr[names[i]+'_ML'] = (MLval_mcmc_val_mcmc_err[i,0], comments[i])
    hdr[names[i]+'_MC'] = MLval_mcmc_val_mcmc_err[i,1]
    hdr[names[i]+'_M'] = MLval_mcmc_val_mcmc_err[i,2]
    hdr[names[i]+'_P'] = MLval_mcmc_val_mcmc_err[i,3]

    if i == 12:
        mas_2_pix = 1/(1000.*pixscale)
        hdr['RAp_ML'] = (MLval_mcmc_val_mcmc_err[11,0]*mas_2_pix, 'pix, ->E right ascension')
        hdr['RAp_MC'] = MLval_mcmc_val_mcmc_err[11,1]*mas_2_pix
        hdr['RAp_M'] = MLval_mcmc_val_mcmc_err[11,2]*mas_2_pix
        hdr['RAp_P'] = MLval_mcmc_val_mcmc_err[11,3]*mas_2_pix

        hdr['Declp_ML'] = (MLval_mcmc_val_mcmc_err[12,0]*mas_2_pix, 'pix, ->N Declination')
        hdr['Declp_MC'] = MLval_mcmc_val_mcmc_err[12,1]*mas_2_pix
        hdr['Declp_M'] = MLval_mcmc_val_mcmc_err[12,2]*mas_2_pix
        hdr['Declp_P'] = MLval_mcmc_val_mcmc_err[12,3]*mas_2_pix


#######################################################
#### Display the fits, model and residuals
#######################################################

#Format the most likely values
theta_ml=[\
        np.log(MLval_mcmc_val_mcmc_err[0,0]),\
        np.log(MLval_mcmc_val_mcmc_err[1,0]),\
        MLval_mcmc_val_mcmc_err[2,0],\
        1/100.*MLval_mcmc_val_mcmc_err[3,0],\
        1/100.*MLval_mcmc_val_mcmc_err[4,0],\
        1/100.*MLval_mcmc_val_mcmc_err[5,0],\
        np.cos(np.radians(MLval_mcmc_val_mcmc_err[6,0])),\
        MLval_mcmc_val_mcmc_err[7,0],\
        MLval_mcmc_val_mcmc_err[8,0],\
        MLval_mcmc_val_mcmc_err[9,0],\
        np.log(MLval_mcmc_val_mcmc_err[10,0])\
        ]



psf = fits.getdata(datadir + 'psf_sphere_h2.fits')


mask2generatedisk = fits.getdata(klipdir + file_prefix + '_mask2generatedisk.fits')
mask2minimize = fits.getdata(klipdir + file_prefix + '_mask2minimize.fits')

mask2generatedisk[np.where(mask2generatedisk == 0.)] = np.nan
wheremask2generatedisk = (mask2generatedisk != mask2generatedisk)

mask2minimize[np.where(mask2minimize == 0.)] = np.nan
wheremask2minimize = (mask2minimize != mask2minimize)


# load the raw data
datacube_Julien = fits.getdata(datadir + "cube_H2.fits")/10.   ### we divide the data to keep the ~same prior as is GPI
parangs = fits.getdata(datadir + "parang.fits") 


datacube_Julien = np.delete(datacube_Julien, (72,81),0) ## 2 slices are bad
parangs = np.delete(parangs, (72,81),0) ## 2 slices are bad

datacube_Julien_newdim = np.zeros((datacube_Julien.shape[0], newdim,newdim))

for i in range(datacube_Julien.shape[0]):
    datacube_Julien_newdim[i,0:newdim-1,0:newdim-1] = crop_center_even(datacube_Julien[i,:,:], newdim-1)
    datacube_Julien_newdim[i,newdim-1, :] = datacube_Julien_newdim[i,newdim-2,0]
    datacube_Julien_newdim[i, :,newdim-1] = datacube_Julien_newdim[i,0,newdim-2]

datacube_Julien = datacube_Julien_newdim
size_datacube = datacube_Julien.shape


parangs = parangs  -135.99 +90 ## true north
centers = np.zeros((size_datacube[0],2)) + xcen
dataset = Instrument.GenericData(datacube_Julien, centers, parangs = parangs, wvs = None)

dataset.OWA = 116 #102 AU
mov_here = 8
KLMODE = [3] 

# load the data
reduced_data = fits.getdata(klipdir + file_prefix + '_Measurewithparallelized-KLmodes-all.fits')

# load the noise
noise = fits.getdata(klipdir + file_prefix + '_noisemap.fits')

#generate the best model
disk_ml=call_gen_disk_2g(theta_ml, wheremask2generatedisk)

new_fits = fits.HDUList()
new_fits.append(fits.ImageHDU(data=disk_ml, header=hdr))
# disk_ml[140,:] = np.max(disk_ml)
# disk_ml[:,140] = np.max(disk_ml)
# namefits = "Model_dx{0}_dy_{1}_PA{2}_inc{3}.fits".format(dx_ml,dy_ml,pa_ml,inc_ml)
# new_fits.writeto('/Users/jmazoyer/Desktop/' +namefits , clobber=True)

new_fits.writeto(mcmcresultdir+name_h5+'_BestModelBeforeConv.fits', clobber=True)


#convolve by the PSF
disk_ml_convolved = convolve(disk_ml,psf, boundary = 'wrap')

new_fits = fits.HDUList()
new_fits.append(fits.ImageHDU(data=disk_ml_convolved, header=hdr))
new_fits.writeto(mcmcresultdir+name_h5+'_BestModelAfterConv.fits', clobber=True)

if socket.gethostname() != 'MT-101942':
    diskobj = DiskFM([size_datacube[0], newdim, newdim], KLMODE,dataset, disk_ml_convolved, annuli=1, subsections=1,
                        basis_filename = klipdir + file_prefix+ '_KLbasis.pkl', save_basis = False, load_from_basis = True, numthreads =1)


    #do the FM
    diskobj.update_disk(disk_ml_convolved)
    disk_ml_FM = diskobj.fm_parallelized()[0] ### we take only the first KL modemode

    new_fits = fits.HDUList()
    new_fits.append(fits.ImageHDU(data=disk_ml_FM, header=hdr))
    new_fits.writeto(mcmcresultdir+name_h5+'_BestModelAfterFM.fits', clobber=True)


disk_ml_FM = fits.getdata(mcmcresultdir+name_h5+'_BestModelAfterFM.fits')

new_fits = fits.HDUList()
new_fits.append(fits.ImageHDU(data=np.abs(reduced_data-disk_ml_FM)/noise, header=hdr))
new_fits.writeto(mcmcresultdir+name_h5+'_BestModelResiduals.fits', clobber=True)


disk_ml_FM = fits.getdata(mcmcresultdir+name_h5+'_BestModelAfterFM.fits')

#Set the colormap
vmin=-2
vmax=12

reduced_data_crop = crop_center_odd(reduced_data,232)
disk_ml_FM_crop = crop_center_odd(disk_ml_FM,232)
disk_ml_convolved_crop = crop_center_odd(disk_ml_convolved,232)
noise_crop = crop_center_odd(noise,232)
disk_ml_crop = crop_center_odd(disk_ml,232)

# reduced_data_crop = reduced_data
# disk_ml_FM_crop = disk_ml_FM
# disk_ml_convolved_crop = disk_ml_convolved
# noise_crop = noise
# disk_ml_crop = disk_ml

caracsize = 40*quality_plot/2

fig=plt.figure(figsize = (6.4*2*quality_plot, 4.8*2*quality_plot))
#The data
ax1 = fig.add_subplot(235)
# cax = plt.imshow(reduced_data_crop+ vmin +0.1, origin='lower', norm=LogNorm(),vmin=vmin, vmax=vmax, cmap=plt.cm.get_cmap('summer'))
cax = plt.imshow(reduced_data_crop+0.1, origin='lower',vmin=vmin, vmax=vmax, cmap=plt.cm.get_cmap('hot'))
ax1.set_title("Original Data", fontsize=caracsize, pad=caracsize/3.)
cbar = fig.colorbar(cax,fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=caracsize*3/4)
plt.axis('off')

#The residuals
ax1 = fig.add_subplot(233)
# cax = plt.imshow(np.abs(reduced_data_crop-disk_ml_FM_crop) + vmin +0.1, origin='lower', norm=LogNorm(),vmin=vmin, vmax=vmax, cmap=plt.cm.get_cmap('summer'))
cax = plt.imshow(np.abs(reduced_data_crop-disk_ml_FM_crop) , origin='lower',vmin=0, vmax=vmax/3, cmap=plt.cm.get_cmap('hot'))
ax1.set_title("Residuals", fontsize=caracsize, pad=caracsize/3.)
cbar = fig.colorbar(cax,fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=caracsize*3/4)
plt.axis('off')

#The SNR of the residuals
ax1 = fig.add_subplot(236)
cax = plt.imshow(np.abs(reduced_data_crop-disk_ml_FM_crop)/noise_crop, origin='lower',vmin=0, vmax=2, cmap=plt.cm.get_cmap('hot'))
ax1.set_title("SNR Residuals", fontsize=caracsize, pad=caracsize/3.)
cbar = fig.colorbar(cax, ticks=[0, 1,2],fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=caracsize*3/4)
cbar.ax.set_yticklabels(['0', '1', '2'])  # vertically oriented colorbar
plt.axis('off')

# The model
ax1 = fig.add_subplot(231)
cax = plt.imshow(disk_ml_crop, origin='lower',vmin=-2, vmax=np.max(disk_ml_crop)/1.5, cmap=plt.cm.get_cmap('hot'))
ax1.set_title("Best Model", fontsize=caracsize, pad=caracsize/3.)
cbar = fig.colorbar(cax,fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=caracsize*3/4)
plt.axis('off')

#The convolved model
ax1 = fig.add_subplot(234)
# cax = plt.imshow(disk_ml_convolved_crop + vmin +0.1, origin='lower', norm=LogNorm(),vmin=vmin, vmax=vmax, cmap=plt.cm.get_cmap('summer'))
cax = plt.imshow(disk_ml_convolved_crop , origin='lower',vmin=vmin, vmax=vmax, cmap=plt.cm.get_cmap('hot'))

ax1.set_title("Model Convolved", fontsize=caracsize, pad=caracsize/3.)
cbar = fig.colorbar(cax,fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=caracsize*3/4)
plt.axis('off')

#The FM convolved model
ax1 = fig.add_subplot(232)
# cax = plt.imshow(disk_ml_FM_crop + vmin +0.1, origin='lower', norm=LogNorm(),vmin=vmin, vmax=vmax, cmap=plt.cm.get_cmap('summer'))
cax = plt.imshow(disk_ml_FM_crop, origin='lower', vmin=vmin, vmax=vmax, cmap=plt.cm.get_cmap('hot'))
ax1.set_title("Model Convolved + FM", fontsize=caracsize, pad=caracsize/3.)
cbar = fig.colorbar(cax,fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=caracsize*3/4)
plt.axis('off')

fig.subplots_adjust(hspace=-0.4, wspace=0.2)

fig.suptitle('SPHERE H2 band: Best Model and Residuals', fontsize=5/4.*caracsize, y = 0.985)

fig.tight_layout()

plt.savefig(mcmcresultdir+name_h5+'_PlotBestModel.jpg')


f1=open(mcmcresultdir+name_h5+'mcmcfit_geometrical_params.txt', 'w+')
f1.write("\n'{0} / {1}".format(reader.iteration, reader.iteration*192))
f1.write("\n")

f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(MLval_mcmc_val_mcmc_err[0,1]/dist_star*1000, MLval_mcmc_val_mcmc_err[0,2]/dist_star*1000,MLval_mcmc_val_mcmc_err[0,3]/dist_star*1000))
f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(MLval_mcmc_val_mcmc_err[1,1]/dist_star*1000, MLval_mcmc_val_mcmc_err[1,2]/dist_star*1000,MLval_mcmc_val_mcmc_err[1,3]/dist_star*1000))
f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(MLval_mcmc_val_mcmc_err[7,1], MLval_mcmc_val_mcmc_err[7,2],MLval_mcmc_val_mcmc_err[7,3]))

f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(MLval_mcmc_val_mcmc_err[11,1], MLval_mcmc_val_mcmc_err[11,2],MLval_mcmc_val_mcmc_err[11,3]))
f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(MLval_mcmc_val_mcmc_err[12,1], MLval_mcmc_val_mcmc_err[12,2],MLval_mcmc_val_mcmc_err[12,3]))


f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(MLval_mcmc_val_mcmc_err[8,1]/dist_star*1000, MLval_mcmc_val_mcmc_err[8,2]/dist_star*1000,MLval_mcmc_val_mcmc_err[8,3]/dist_star*1000))
f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(MLval_mcmc_val_mcmc_err[9,1]/dist_star*1000, MLval_mcmc_val_mcmc_err[9,2]/dist_star*1000,MLval_mcmc_val_mcmc_err[9,3]/dist_star*1000))



f1.write("\n")
f1.write("\n")
    
f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(MLval_mcmc_val_mcmc_err[13,1], MLval_mcmc_val_mcmc_err[13,2],MLval_mcmc_val_mcmc_err[13,3]))
f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(MLval_mcmc_val_mcmc_err[14,1], MLval_mcmc_val_mcmc_err[14,2],MLval_mcmc_val_mcmc_err[14,3]))
f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(MLval_mcmc_val_mcmc_err[6,1], MLval_mcmc_val_mcmc_err[6,2],MLval_mcmc_val_mcmc_err[6,3]))
f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(MLval_mcmc_val_mcmc_err[15,1], MLval_mcmc_val_mcmc_err[15,2],MLval_mcmc_val_mcmc_err[15,3]))
f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(MLval_mcmc_val_mcmc_err[16,1], MLval_mcmc_val_mcmc_err[16,2],MLval_mcmc_val_mcmc_err[16,3]))

f1.close()

