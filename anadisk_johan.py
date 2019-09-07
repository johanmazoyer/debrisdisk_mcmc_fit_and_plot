########################################################
########################################################
#### exctracted from anadisk_e.py
#### author Max Millar Blanchaer
#### modified by Johan
########################################################
########################################################

import math as mt
import numpy as np

from scipy.integrate import quad

import warnings
warnings.filterwarnings("ignore", category=UserWarning)



def integrand_dxdy_2g(xp, yp_dy2, yp2, zp, zp2, zpsi_dx, zpci, R1, R2, beta,
                      a_r, g1, g1_2, g2, g2_2, alpha, ci, si, maxe, dx, dy, k):
    # author : Max Millar Blanchaer
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
                     dx=0,
                     dy=0.,
                     mask=None,
                     sampling=1,
                     distance=72.8,
                     pixscale=0.01414):
    """ author : Max Millar Blanchaer
        create a 2g SPF disk model. The disk is normalized at 1 at 90degree
        (before star offset). also normalized by aspect_ratio


    Args:
        dim: dimension of the image in pixel assuming square image
        R1: inner radius of the disk
        R2: outer radius of the disk
        beta: radial power law of the disk between R1 and R2
        aspect_ratio=0.1 vertical width of the disk
        g1: %, 1st HG param
        g2: %, 2nd HG param
        Aplha: %, relative HG weight
        inc: degree, inclination
        pa: degree, principal angle
        dx: au, + -> NW offset disk plane Minor Axis
        dy: au, + -> SW offset disk plane Major Axis
        mask: a np.where result that give where the model should be
              measured (important to save a lot of time)
        sampling: increase this parameter to bin the model
                  and save time
        distance: distance of the star
        pixscale: pixel scale of the instrument

    Returns:
        a 2d model
    """
    # from datetime import datetime
    # starttime=datetime.now()

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

                # if hmask[j,npts/2+i]: #This assumes
                # that the input mask has is the same size as
                # the desired image (i.e. ~ size / sampling)
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


def integrand_dxdy_3g(xp, yp_dy2, yp2, zp, zp2, zpsi_dx, zpci, R1, R2, beta,
                      a_r, g1, g1_2, g2, g2_2, g3, g3_2, alpha1, alpha2, ci,
                      si, maxe, dx, dy, k):

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
    # hg1=k*alpha*     (1. - g1_2)/(1. + g1_2 - (2*g1*cos_phi))**1.5
    # hg2=k*(1-alpha)* (1. - g2_2)/(1. + g2_2 - (2*g2*cos_phi))**1.5

    #Henyey Greenstein function
    hg1 = k * (1. - g1_2) / (1. + g1_2 - (2 * g1 * cos_phi))**1.5
    hg2 = k * (1. - g2_2) / (1. + g2_2 - (2 * g2 * cos_phi))**1.5
    hg3 = k * (1. - g3_2) / (1. + g3_2 - (2 * g3 * cos_phi))**1.5

    hg = alpha1 * hg1 + alpha2 * hg2 + (1 - alpha1 - alpha2) * hg3
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




def gen_disk_dxdy_3g(dim,
                     R1=74.42,
                     R2=82.45,
                     beta=1.0,
                     aspect_ratio=0.1,
                     g1=0.6,
                     g2=-0.6,
                     g3=-0.1,
                     alpha1=0.7,
                     alpha2=0.,
                     inc=76.49,
                     pa=30,
                     dx=0,
                     dy=0.,
                     mask=None,
                     sampling=1,
                     distance=72.8,
                     pixscale=0.01414):

    """ author : Max Millar Blanchaer
        create a 3g SPF disk model. The disk is normalized at 1 at 90degree
        (before star offset). also normalized by aspect_ratio


    Args:
        dim: dimension of the image in pixel assuming square image
        R1: inner radius of the disk
        R2: outer radius of the disk
        beta: radial power law of the disk between R1 and R2
        aspect_ratio=0.1 vertical width of the disk
        g1: %, 1st HG param
        g2: %, 2nd HG param
        g3: %, 3rd HG param
        Aplha1: %, first relative HG weight
        Aplha2: %, second relative HG weight
        inc: degree, inclination
        pa: degree, principal angle
        dx: au, + -> NW offset disk plane Minor Axis
        dy: au, + -> SW offset disk plane Major Axis
        mask: a np.where result that give where the model should be
              measured (important to save a lot of time)
        sampling: increase this parameter to bin the model
                  and save time
        distance: distance of the star
        pixscale: pixel scale of the instrument

    Returns:
        a 2d model
    """
    # starttime=datetime.now()

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
    g3_2 = g3 * g3  #Second HG g squared

    #Constant for HG function
    k = 1. / (4 * np.pi) * 100
    ### we add a 100 multiplicateur to k avoid hg values to be too small, it makes the integral fail on certains points
    ### Since we normalize by hg90 at the end, this has no impact on the actual model

    #The aspect ratio
    a_r = aspect_ratio

    #Henyey Greenstein function at 90

    hg1_90 = k * (1. - g1_2) / (1. + g1_2)**1.5
    hg2_90 = k * (1. - g2_2) / (1. + g2_2)**1.5
    hg3_90 = k * (1. - g3_2) / (1. + g3_2)**1.5

    hg_90 = alpha1 * hg1_90 + alpha2 * hg2_90 + (1 - alpha1 - alpha2) * hg3_90

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

                # image[j,i]=  quad(integrand_dxdy_2g, -R2, R2, epsrel=0.5e-3,limit=75,args=(yy_dy2,y2,zp,z2,zpsi_dx,zpci,R1,R2,beta,a_r,g1,g1_2,g2,g2_2, alpha,ci,si,maxe,dx,dy,k))[0]
                image[j, i] = quad(integrand_dxdy_3g,
                                   -R2,
                                   R2,
                                   epsrel=0.5e-12,
                                   limit=75,
                                   args=(yy_dy2, y2, zp, z2, zpsi_dx, zpci, R1,
                                         R2, beta, a_r, g1, g1_2, g2, g2_2, g3,
                                         g3_2, alpha1, alpha2, ci, si, maxe,
                                         dx, dy, k))[0]

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

                    # image[j,i]=  quad(integrand_dxdy_2g, -R2, R2, epsrel=0.5e-3,limit=75,args=(yy_dy2,y2,zp,z2,zpsi_dx,zpci,R1,R2,beta,a_r,g1,g1_2,g2,g2_2, alpha,ci,si,maxe,dx,dy,k))[0]
                    image[j, i] = quad(integrand_dxdy_3g,
                                       -R2,
                                       R2,
                                       epsrel=0.5e-12,
                                       limit=75,
                                       args=(yy_dy2, y2, zp, z2, zpsi_dx, zpci,
                                             R1, R2, beta, a_r, g1, g1_2, g2,
                                             g2_2, g3, g3_2, alpha1, alpha2,
                                             ci, si, maxe, dx, dy, k))[0]

    # # normalize the HG function by the width
    image = image / a_r

    # normalize the HG function at the PA
    image = image / hg_90

    # print("Running time 3g: {0}".format(datetime.now()-starttime))

    return image




def integrand_dxdy_flat(xp, yp_dy2, yp2, zp, zp2, zpsi_dx, zpci, R1, R2, beta,
                      a_r, ci,
                      si, maxe, dx, dy):

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


    #Radial power low r propto -beta
    int1 = (R1 / d1)**beta

    #The scale height function
    zz = (zpci - xp * si)
    hh = (a_r * d1)
    expo = zz * zz / (hh * hh)

    # if expo > 2*maxe:   # cut off exponential after 28 e-foldings (~ 1E-06)
    #     return 0.0

    int2 = np.exp(0.5 * expo)
    int3 = int2 * d2

    return int1 / int3



def gen_disk_dxdy_flat(dim,
                     R1=74.42,
                     R2=82.45,
                     beta=1.0,
                     aspect_ratio=0.1,
                     inc=76.49,
                     pa=30,
                     dx=0,
                     dy=0.,
                     mask=None,
                     sampling=1,
                     distance=72.8,
                     pixscale=0.01414):

    """ author : Max Millar Blanchaer
        create a 3g SPF disk model. The disk is normalized at 1 at 90degree
        (before star offset). also normalized by aspect_ratio


    Args:
        dim: dimension of the image in pixel assuming square image
        R1: inner radius of the disk
        R2: outer radius of the disk
        beta: radial power law of the disk between R1 and R2
        aspect_ratio=0.1 vertical width of the disk
        g1: %, 1st HG param
        g2: %, 2nd HG param
        g3: %, 3rd HG param
        Aplha1: %, first relative HG weight
        Aplha2: %, second relative HG weight
        inc: degree, inclination
        pa: degree, principal angle
        dx: au, + -> NW offset disk plane Minor Axis
        dy: au, + -> SW offset disk plane Major Axis
        mask: a np.where result that give where the model should be
              measured (important to save a lot of time)
        sampling: increase this parameter to bin the model
                  and save time
        distance: distance of the star
        pixscale: pixel scale of the instrument

    Returns:
        a 2d model
    """
    # starttime=datetime.now()

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


    #The aspect ratio
    a_r = aspect_ratio


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

                # image[j,i]=  quad(integrand_dxdy_2g, -R2, R2, epsrel=0.5e-3,limit=75,args=(yy_dy2,y2,zp,z2,zpsi_dx,zpci,R1,R2,beta,a_r,g1,g1_2,g2,g2_2, alpha,ci,si,maxe,dx,dy,k))[0]
                image[j, i] = quad(integrand_dxdy_flat,
                                   -R2,
                                   R2,
                                   epsrel=0.5e-12,
                                   limit=75,
                                   args=(yy_dy2, y2, zp, z2, zpsi_dx, zpci, R1,
                                         R2, beta, a_r, ci, si, maxe,
                                         dx, dy))[0]

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

                    # image[j,i]=  quad(integrand_dxdy_2g, -R2, R2, epsrel=0.5e-3,limit=75,args=(yy_dy2,y2,zp,z2,zpsi_dx,zpci,R1,R2,beta,a_r,g1,g1_2,g2,g2_2, alpha,ci,si,maxe,dx,dy,k))[0]
                    image[j, i] = quad(integrand_dxdy_flat,
                                       -R2,
                                       R2,
                                       epsrel=0.5e-12,
                                       limit=75,
                                       args=(yy_dy2, y2, zp, z2, zpsi_dx, zpci,
                                             R1, R2, beta, a_r,
                                             ci, si, maxe, dx, dy))[0]

    # # normalize the HG function by the width
    image = image / a_r



    # print("Running time 3g: {0}".format(datetime.now()-starttime))

    return image

