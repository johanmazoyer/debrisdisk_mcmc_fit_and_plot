# Analytic disk model
# Originally disk model by James Graham 
# Modified by Max Millar-Blanchaer

import numpy as np
import math as mt
import scipy.integrate as scipi
from datetime import datetime

def integrand(xp,yp,zp,yp2,zp2,zpci,zpsi,R1,R2,beta,a_r,g,g2,ci,si,maxe):

    # compute the scattering integrand
    # see analytic-disk.nb
    
    d1 = mt.sqrt(yp2 + (xp * ci + zpsi)**2)

    if (d1  < R1 or d1 > R2):
        return 0.0

    d2 = xp*xp + yp2 + zp2

    # Cos scattering angle
    mu = xp/mt.sqrt(d2)

    hgg = (1. - g2)*(1. + g2 - 2*g*mu )**(-1.5)

    int1 = hgg*(R1/d1)**beta

    expo = ((zpci - xp*si)/(a_r*d1))**2

    if expo > 2*maxe:   # cut off exponential after 28 e-foldings (~ 1E-06)
        return 0.0 

    int2 = mt.exp(0.5*expo) 
    int3 = int2 * d2 
    
    return int1/int3 

def integrand_dxdy(xp,yp_dy2,yp2,zp,zp2,zpsi_dx,zpci,R1,R2,beta,a_r,g,g2, ci, si,maxe,dx, dy,k):

    # compute the scattering integrand
    # see analytic-disk.nb

    xx=(xp * ci + zpsi_dx)

    d1 = mt.sqrt((yp_dy2 + xx*xx))

    if (d1  < R1 or d1 > R2):
        return 0.0

    d2 = xp*xp + yp2 + zp2

    if d2 == 0.:
        return 0.0

    #The line of sight scattering angle
    cos_phi=xp/mt.sqrt(d2)
    # phi=np.arccos(cos_phi)

    #Henyey Greenstein function
    hg=k*(1. - g2)/(1. + g2 - (2*g*cos_phi))**1.5

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


def integrand_dxdy_3g(xp,yp_dy2,yp2,zp,zp2,zpsi_dx,zpci,R1,R2,beta,a_r,g1,g1_2,g2,g2_2,g3,g3_2, alpha1, alpha2, ci, si,maxe,dx, dy,k):

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
    hg1=k*(1. - g1_2)/(1. + g1_2 - (2*g1*cos_phi))**1.5
    hg2=k*(1. - g2_2)/(1. + g2_2 - (2*g2*cos_phi))**1.5
    hg3=k*(1. - g3_2)/(1. + g3_2 - (2*g3*cos_phi))**1.5

    hg = alpha1*hg1 + alpha2 * hg2 + (1 - alpha1 -alpha2) * hg3

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

def integrand_dxdy_rayleigh(xp,yp_dy2,yp2,zp,zp2,zpsi_dx,zpci,R1,R2,beta,a_r,g,g2, ci, si,maxe,dx, dy,k):

    # compute the scattering integrand
    # see analytic-disk.nb

    xx=(xp * ci + zpsi_dx)

    d1 = mt.sqrt((yp_dy2 + xx*xx))

    if (d1  < R1 or d1 > R2):
        return 0.0

    d2 = xp*xp + yp2 + zp2

    #The line of sight scattering angle
    cos_phi=xp/mt.sqrt(d2)
    phi=np.arccos(cos_phi)

    #Henyey Greenstein function
    hg=k*(1. - g2)/(1. + g2 - (2*g*cos_phi))**1.5
    rayleigh=mt.sin(phi)*mt.sin(phi)/(1+cos_phi*cos_phi)

    #Radial power low r propto -beta
    int1 = hg*rayleigh*(R1/d1)**beta

    #The scale height function
    zz = (zpci - xp*si)
    hh = a_r * d1
    expo = zz*zz/((hh*hh))

    if expo > 2*maxe:   # cut off exponential after 28 e-foldings (~ 1E-06)
        return 0.0 

    int2 = np.exp(0.5*expo) 
    int3 = int2 * d2

    return int1/int3

def integrand_e(xp,yp,zp,incl,R1,R2,beta,a_r,g, ci, si,e, omega):

    # compute the scattering integrand
    # see analytic-disk.nb
    c=e*R1 #The location of the focus (For now we will force the to be along the x axis)
    cx=c*mt.cos(omega)
    cy=c*mt.sin(omega)

    # xx=(xp * ci + zp* si - cx)
    # yy=(yp-cy)

    xx=(xp * ci + zp* si)
    yy=yp

    #This is the distance from the focus [cx,cy]
    r = mt.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy,xx) + omega

    #The semi-major axis for a point at this focus
    a = r *(1+e*mt.cos(theta))/(1-e*e)

    d1 = mt.sqrt(yy**2 + xx**2)

    #If we're inside the inner radius then don't need to do anything
    if (a  < R1 or a > R2):
        return 0.0

    #The distance from the Star
    d2 = xp*xp + yp*yp + zp*zp

    #The line of sight scattering angle
    cos_phi=xp/mt.sqrt(d2)
    phi=np.arccos(cos_phi)

    hg=1./(4*np.pi)*(1. - g*g)/(1. + g*g - (2*g*cos_phi))**1.5
    rayleigh=mt.sin(phi)**2/(1+cos_phi**2)

    #Density goes as a^{-beta}
    int1 = hg*rayleigh*(R1/a)**beta
    # int1 = hg*(R1/d1)**beta

    hh=a_r*d1
    #The density as a function of height
    expo = (zp*ci - xp*si)**2/(hh*hh)

    if expo > 28:   # cut off exponential after 28 e-foldings (~ 1E-06)
        return 0.0 

    int2 = mt.exp(0.5*expo) 
    int3 = int2 * d2

    return int1/int3

#####################################################################
def gen_disk(R1=74.42, R2=82.45, beta=1.0, aspect_ratio=0.1, g=0.6, inc=76.49, pa=30, distance=72.8, psfcenx=140,psfceny=140, sampling=1, mask=None):
    
    # starttime=datetime.now()

    #The default  parameters -- similar to  HR 4796A ring geometry
    # R1 = 74.42              # inner hole [AU]
    # R2 = 82.45              # outer hole [AU]
    # beta = 1.0              # power law slope
    # g = 0.6                 # <cos theta>
    # inc=76.49                 # the disk inclination [degrees]
    incl = np.radians(90-inc)  # convert inclination [degrees] to radians and do the flip for this code 
    # pa = 30               # The Position angle of the disk [degrees]
    # distance=72.8            #Distance [parsecs]


    pa_rad=np.radians(pa) #The position angle in radians
    cos_pa=mt.cos(pa_rad) #Calculate these ahead of time
    sin_pa=mt.sin(pa_rad)

    #GPI FOV stuff
    pixscale = 0.01414 #GPI's pixel scale
    nspaxels = 200 #The number of lenslets along one side 
    fov_square = pixscale*nspaxels #The square FOV of GPI 
    max_fov = mt.sqrt(2)*fov_square/2 #maximum FOV in arcseconds from the center to the edge
    xsize = max_fov*distance #maximum radial distance in AU from the center to the edge

    #Get The number of resolution elements in the radial direction from the center
    # sampling = 1
    # npts = mt.ceil(nspaxels*mt.sqrt(2)/2/sampling)#The number of pixels to use from the center to the edge
    dim=281. #for now the GPI dimensions are hardcoded
    npts=int(np.floor(dim/sampling))

    #Set up the 

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
    ci = mt.cos(incl) #Cosine of inclination
    si = mt.sin(incl) #Sine of inclination

    #HG g value squared
    g2 = g**2

    a_r=aspect_ratio

    #If there's no mask then calculate for the full image
    if len(np.shape(mask)) < 2:

        for i,yp in enumerate(y):
            for j,zp in enumerate(z):


                #This rotates the coordinates in the image frame
                yy=yp*cos_pa - zp * sin_pa  #Rotate the y coordinate by the PA
                zz=yp*sin_pa + zp * cos_pa  #Rotate the z coordinate by the PA

                y2 = (yy)**2
                z2 = (zz)**2

                #This rotates the coordinates in and out of the sky
                zci=zz*ci #Rotate the z coordinate by the inclination. 
                zsi=zz*si

                # image[j,i]=  scipi.quad(integrand, -R2, R2, epsrel=1e-2,args=(yp,zp,y2[i],z2[j],zci[j],zsi[j],R1,R2,beta,h,g,g2,ci,si,maxe))[0]
                image[j,i]=  scipi.quad(integrand, -R2, R2, epsrel=1e-2,args=(yp,zp,y2,z2,zci,zsi,R1,R2,beta,a_r,g,g2,ci,si,maxe))[0]

    #If there is a mask then don't calculate disk there
    else: 
        hmask = mask
        # hmask = mask[:,140:] #Use only half the mask

        for i,yp in enumerate(y):
            for j,zp in enumerate(z):
                
                # if hmask[j,npts/2+i]: #This assumes that the input mask has is the same size as the desired image (i.e. ~ size / sampling)
                if hmask[j,i]:

                   image[j,i] = np.nan
                
                else:

                    yy=yp*cos_pa - zp * sin_pa
                    zz=yp*sin_pa + zp * cos_pa

                    y2 = yy**2
                    z2 = zz**2

                    zci=zz*ci
                    zsi=zz*si
                    # image[j,i]=  scipi.quad(integrand, -R2, R2, epsrel=1e-2,args=(yp,zp,y2[i],z2[j],zci[j],zsi[j],R1,R2,beta,h,g,g2,ci,si,maxe))[0]
                    image[j,i]=  scipi.quad(integrand, -R2, R2, epsrel=1e-2,args=(yp,zp,y2,z2,zci,zsi,R1,R2,beta,a_r,g,g2,ci,si,maxe))[0]


    return image

#####################################################################
def gen_disk_dxdy(R1=74.42, R2=82.45, beta=1.0, aspect_ratio=0.1, g=0.6, inc=76.49, pa=30, distance=72.8, psfcenx=140,psfceny=140, sampling=1, mask=None, dx=0, dy=0.):

    # starttime=datetime.now()

    #GPI FOV stuff
    pixscale = 0.01414 #GPI's pixel scale
    nspaxels = 200 #The number of lenslets along one side 
    fov_square = pixscale*nspaxels #The square FOV of GPI 
    max_fov = mt.sqrt(2)*fov_square/2 #maximum FOV in arcseconds from the center to the edge
    xsize = max_fov*distance #maximum radial distance in AU from the center to the edge

    #Get The number of resolution elements in the radial direction from the center
    # sampling = 1
    # npts = mt.ceil(nspaxels*mt.sqrt(2)/2/sampling)#The number of pixels to use from the center to the edge
    dim=281. #for now the GPI dimensions are hardcoded
    npts=int(np.floor(dim/sampling))

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
    g2 = g*g
    #Constant for HG function
    k=1./(4*np.pi)

    #The aspect ratio
    a_r=aspect_ratio

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

                # image[j,i]=  scipi.quad(integrand, -R2, R2, epsrel=1e-2,args=(yp,zp,y2[i],z2[j],zci[j],zsi[j],R1,R2,beta,h,g,g2,ci,si,maxe))[0]
                image[j,i]=  scipi.quad(integrand_dxdy, -R2, R2, epsrel=0.5e-3,limit=75,args=(yy_dy2,y2,zp,z2,zpsi_dx,zpci,R1,R2,beta,a_r,g,g2,ci,si,maxe,dx,dy,k))[0]
            

    #If there is a mask then don't calculate disk there
    else: 
        hmask = mask
        # hmask = mask[:,140:] #Use only half the mask

        for i,yp in enumerate(y):
            for j,zp in enumerate(z):
                
                # if hmask[j,npts/2+i]: #This assumes that the input mask has is the same size as the desired image (i.e. ~ size / sampling)
                if hmask[j,i]:

                   image[j,i] = np.nan
                
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

                    # image[j,i]=  scipi.quad(integrand, -R2, R2, epsrel=1e-2,args=(yp,zp,y2[i],z2[j],zci[j],zsi[j],R1,R2,beta,h,g,g2,ci,si,maxe))[0]
                    image[j,i]=  scipi.quad(integrand_dxdy, -R2, R2, epsrel=0.5e-3,limit=75,args=(yy_dy2,y2,zp,z2,zpsi_dx,zpci,R1,R2,beta,a_r,g,g2,ci,si,maxe,dx,dy,k))[0]

    # print("Running time: ", datetime.now()-starttime)



    return image
#####################################################################
def gen_disk_dxdy_2g(R1=74.42, R2=82.45, beta=1.0, aspect_ratio=0.1, g1=0.6, g2=-0.6, alpha=0.7, inc=76.49, pa=30, distance=72.8, psfcenx=140,psfceny=140, sampling=1, mask=None, dx=0, dy=0.):

    # starttime=datetime.now()

    #GPI FOV stuff
    pixscale = 0.01414 #GPI's pixel scale
    nspaxels = 200 #The number of lenslets along one side 
    fov_square = pixscale*nspaxels #The square FOV of GPI 
    max_fov = mt.sqrt(2)*fov_square/2 #maximum FOV in arcseconds from the center to the edge
    xsize = max_fov*distance #maximum radial distance in AU from the center to the edge

    #Get The number of resolution elements in the radial direction from the center
    # sampling = 1
    # npts = mt.ceil(nspaxels*mt.sqrt(2)/2/sampling)#The number of pixels to use from the center to the edge
    dim=281. #for now the GPI dimensions are hardcoded
    npts=int(np.floor(dim/sampling))

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

                # image[j,i]=  scipi.quad(integrand, -R2, R2, epsrel=1e-2,args=(yp,zp,y2[i],z2[j],zci[j],zsi[j],R1,R2,beta,h,g,g2,ci,si,maxe))[0]
                image[j,i]=  scipi.quad(integrand_dxdy_2g, -R2, R2, epsrel=0.5e-3,limit=75,args=(yy_dy2,y2,zp,z2,zpsi_dx,zpci,R1,R2,beta,a_r,g1,g1_2,g2,g2_2, alpha,ci,si,maxe,dx,dy,k))[0]
            

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

                    # image[j,i]=  scipi.quad(integrand, -R2, R2, epsrel=1e-2,args=(yp,zp,y2[i],z2[j],zci[j],zsi[j],R1,R2,beta,h,g,g2,ci,si,maxe))[0]
                    image[j,i]=  scipi.quad(integrand_dxdy_2g, -R2, R2, epsrel=0.5e-3,limit=75,args=(yy_dy2,y2,zp,z2,zpsi_dx,zpci,R1,R2,beta,a_r,g1,g1_2,g2,g2_2, alpha,ci,si,maxe,dx,dy,k))[0]

    # print("Running time: ", datetime.now()-starttime)

    # # normalize the HG function by the width
    image = image/a_r

    # normalize the HG function at the PA
    image = image/hg_90

    return image



#####################################################################
def gen_disk_dxdy_3g(R1=74.42, R2=82.45, beta=1.0, aspect_ratio=0.1, g1=0.6, g2=-0.6 , g3=-0.1, alpha1=0.7, alpha2=0., inc=76.49, pa=30, distance=72.8, psfcenx=140,psfceny=140, sampling=1, mask=None, dx=0, dy=0.):

    # starttime=datetime.now()

    #GPI FOV stuff
    pixscale = 0.01414 #GPI's pixel scale
    nspaxels = 200 #The number of lenslets along one side 
    fov_square = pixscale*nspaxels #The square FOV of GPI 
    max_fov = mt.sqrt(2)*fov_square/2 #maximum FOV in arcseconds from the center to the edge
    xsize = max_fov*distance #maximum radial distance in AU from the center to the edge

    #Get The number of resolution elements in the radial direction from the center
    # sampling = 1
    # npts = mt.ceil(nspaxels*mt.sqrt(2)/2/sampling)#The number of pixels to use from the center to the edge
    dim=281. #for now the GPI dimensions are hardcoded
    npts=int(np.floor(dim/sampling))

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
    g3_2 = g3*g3 #Second HG g squared
    
    #Constant for HG function
    k=1./(4*np.pi)

    #The aspect ratio
    a_r=aspect_ratio

    #Henyey Greenstein function at 90

    hg1_90=k*(1. - g1_2)/(1. + g1_2)**1.5
    hg2_90=k*(1. - g2_2)/(1. + g2_2)**1.5
    hg3_90=k*(1. - g3_2)/(1. + g3_2)**1.5

    hg_90 = alpha1*hg1_90 + alpha2 * hg2_90 + (1 - alpha1 -alpha2) * hg3_90


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

                # image[j,i]=  scipi.quad(integrand, -R2, R2, epsrel=1e-2,args=(yp,zp,y2[i],z2[j],zci[j],zsi[j],R1,R2,beta,h,g,g2,ci,si,maxe))[0]
                # image[j,i]=  scipi.quad(integrand_dxdy_2g, -R2, R2, epsrel=0.5e-3,limit=75,args=(yy_dy2,y2,zp,z2,zpsi_dx,zpci,R1,R2,beta,a_r,g1,g1_2,g2,g2_2, alpha,ci,si,maxe,dx,dy,k))[0]

                image[j,i]=  scipi.quad(integrand_dxdy_3g, -R2, R2, epsrel=0.5e-3,limit=75,args=(yy_dy2,y2,zp,z2,zpsi_dx,zpci,R1,R2,beta,a_r,g1,g1_2,g2,g2_2,g3,g3_2, alpha1, alpha2,ci,si,maxe,dx,dy,k))[0]

            

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

                    # image[j,i]=  scipi.quad(integrand, -R2, R2, epsrel=1e-2,args=(yp,zp,y2[i],z2[j],zci[j],zsi[j],R1,R2,beta,h,g,g2,ci,si,maxe))[0]
                    #image[j,i]=  scipi.quad(integrand_dxdy_2g, -R2, R2, epsrel=0.5e-3,limit=75,args=(yy_dy2,y2,zp,z2,zpsi_dx,zpci,R1,R2,beta,a_r,g1,g1_2,g2,g2_2, alpha,ci,si,maxe,dx,dy,k))[0]
                    image[j,i]=  scipi.quad(integrand_dxdy_3g, -R2, R2, epsrel=0.5e-3,limit=75,args=(yy_dy2,y2,zp,z2,zpsi_dx,zpci,R1,R2,beta,a_r,g1,g1_2,g2,g2_2, g3,g3_2, alpha1, alpha2,ci,si,maxe,dx,dy,k))[0]

    # print("Running time: ", datetime.now()-starttime)

    # # normalize the HG function by the width
    image = image/a_r

    # normalize the HG function at the PA
    image = image/hg_90

    return image


#####################################################################
def gen_disk_dxdy_rayleigh(R1=74.42, R2=82.45, beta=1.0, aspect_ratio=0.1, g=0.6, inc=76.49, pa=30, distance=72.8, psfcenx=140,psfceny=140, sampling=1, mask=None, dx=0, dy=0.):

    # starttime=datetime.now()

    #GPI FOV stuff
    pixscale = 0.01414 #GPI's pixel scale
    nspaxels = 200 #The number of lenslets along one side 
    fov_square = pixscale*nspaxels #The square FOV of GPI 
    max_fov = mt.sqrt(2)*fov_square/2 #maximum FOV in arcseconds from the center to the edge
    xsize = max_fov*distance #maximum radial distance in AU from the center to the edge

    #Get The number of resolution elements in the radial direction from the center
    # sampling = 1
    # npts = mt.ceil(nspaxels*mt.sqrt(2)/2/sampling)#The number of pixels to use from the center to the edge
    dim=281. #for now the GPI dimensions are hardcoded
    npts=int(np.floor(dim/sampling))

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
    g2 = g*g
    #Constant for HG function
    k=1./(4*np.pi)

    #The aspect ratio
    a_r=aspect_ratio

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

                # image[j,i]=  scipi.quad(integrand, -R2, R2, epsrel=1e-2,args=(yp,zp,y2[i],z2[j],zci[j],zsi[j],R1,R2,beta,h,g,g2,ci,si,maxe))[0]
                image[j,i]=  scipi.quad(integrand_dxdy_rayleigh, -R2, R2, epsrel=0.5e-3,limit=75,args=(yy_dy2,y2,zp,z2,zpsi_dx,zpci,R1,R2,beta,a_r,g,g2,ci,si,maxe,dx,dy,k))[0]

    #If there is a mask then don't calculate disk there
    else: 
        hmask = mask
        # hmask = mask[:,140:] #Use only half the mask

        for i,yp in enumerate(y):
            for j,zp in enumerate(z):
                
                # if hmask[j,npts/2+i]: #This assumes that the input mask has is the same size as the desired image (i.e. ~ size / sampling)
                if hmask[j,i]:

                   image[j,i] = np.nan
                
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

                    # image[j,i]=  scipi.quad(integrand, -R2, R2, epsrel=1e-2,args=(yp,zp,y2[i],z2[j],zci[j],zsi[j],R1,R2,beta,h,g,g2,ci,si,maxe))[0]
                    image[j,i]=  scipi.quad(integrand_dxdy_rayleigh, -R2, R2, epsrel=0.5e-3,limit=75,args=(yy_dy2,y2,zp,z2,zpsi_dx,zpci,R1,R2,beta,a_r,g,g2,ci,si,maxe,dx,dy,k))[0]

    # print("Running time: ", datetime.now()-starttime)

    return image

#####################################################################
def gen_disk_e(R1=74.42, R2=82.45, beta=1.0, h=6, g=0.6, inc=76.49, pa=30, distance=72.8, psfcenx=140,psfceny=140, sampling=1, mask=None, e=0, omega=0.):

    # R1 = 5.0                # inner hole
    # R2 = 50.0              # outer hole
    # beta = 1.0              # power law slope
    # h = 2.0                 # vertical scale height
    # g = 0.0                 # <cos theta>
    incl = np.radians(90-inc)  # convert inclination to radians
    omega = np.radians(omega)
    
    #GPI FOV stuff
    pixscale = 0.01414 #GPI's pixel scale
    nspaxels = 200 #The number of lenslets along one side 
    fov_square = pixscale*nspaxels #The square FOV of GPI 
    max_fov = mt.sqrt(2)*fov_square/2 #maximum FOV in arcseconds from the center to the edge
    xsize = max_fov*distance #maximum radial distance in AU from the center to the edge

    #Get The number of resolution elements in the radial direction from the center
    # sampling = 1
    # npts = mt.ceil(nspaxels*mt.sqrt(2)/2/sampling)#The number of pixels to use from the center to the edge
    dim=281. #for now the GPI dimensions are hardcoded
    npts=int(np.floor(dim/sampling))

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

    pa_rad=np.radians(90-pa) #The position angle in radians
    cos_pa=mt.cos(pa_rad) #Calculate these ahead of time
    sin_pa=mt.sin(pa_rad)

    #Precomputer some things ahead of time
    ci = np.cos(incl)
    si = np.sin(incl) 

    image =np.zeros((npts,npts))

    startTime = datetime.now()

    yp_array=np.zeros((npts,npts))
    zp_array=np.zeros((npts,npts))

    for i,yp in enumerate(y):
        for j,zp in enumerate(z):

            yy=yp*cos_pa - zp * sin_pa  #Rotate the y coordinate by the PA
            zz=yp*sin_pa + zp * cos_pa  #Rotate the z coordinate by the PA

            image[j,i]=  scipi.quad(integrand_e, -R2, R2, epsrel=1e-4,args=(yy,zz,incl,R1,R2,beta,h,g,ci,si,e,omega))[0]
    #        image[j,i]=  scipi.romberg(integrand, -R2, R2, rtol=1e-4,args=(yp,zp,incl,R1,R2,beta,h,g))

            
    # print(datetime.now()-startTime))

    return image