import warnings

from scipy.ndimage import rotate

import math as mt
import numpy as np

from anadisk_johan import gen_disk_dxdy_flat

import astro_unit_conversion as convert
from kowalsky import kowalsky

import matplotlib.pyplot as plt
# import matplotlib.patches.Circle as circle

warnings.filterwarnings("ignore", category=RuntimeWarning)

DISTANCE_STAR = 72.0
PIXSCALE_INS = 0.012255
DIMENSION = 281

def call_gen_disk_flat(theta):
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
    inc = np.degrees(np.arccos(theta[3]))
    pa = theta[4]
    dx = theta[5]
    dy = theta[6]
    norm = mt.exp(theta[7])
    # offset = theta[11]



    #generate the model
    model = norm * gen_disk_dxdy_flat(DIMENSION,
                                    R1=r1,
                                    R2=r2,
                                    beta=beta,
                                    aspect_ratio=0.01,
                                    inc=inc,
                                    pa=pa,
                                    dx=dx,
                                    dy=dy,
                                    mask=WHEREMASK2GENERATEDISK,
                                    pixscale=PIXSCALE_INS,
                                    distance=DISTANCE_STAR)  #+ offset

    return model

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


def radius_ellip_mas(skyplaneangle_rad, radius_disk_mas,estimInc_rad):
        ### we measure the radius for every sky angle

        semi_minor_axis_mas= radius_disk_mas*np.cos(estimInc_rad)


        a = radius_disk_mas
        b = semi_minor_axis_mas

        excentricity = np.sqrt(1 - (b/a)**2)

        # print(excentricity) #you should find 0.97 for HR4796
        # print(semi_minor_axis_arcsec)

        ellispe_denom = np.sqrt(1 - excentricity**2 * np.cos(skyplaneangle_rad)**2 )

        radius_ellipse_mas = b/ellispe_denom

        return radius_ellipse_mas


rad1 = 74.3
rad2 = 85
incl = 76.44
pa =  0.

dx = 5.
dy = -10.


theta = (np.log(rad1), np.log(rad2), 13, np.cos(np.radians(incl)), pa, dx,dy, np.log(10))


mask_disk_zeros = make_disk_mask(
            DIMENSION,
            pa,
            incl,
            convert.au_to_pix(np.clip(rad1 - 10,1,None), PIXSCALE_INS, DISTANCE_STAR),
            convert.au_to_pix(rad2 + 10, PIXSCALE_INS, DISTANCE_STAR),
            xcen=DIMENSION//2,
            ycen=DIMENSION//2)

mask2generatedisk = 1 - mask_disk_zeros
WHEREMASK2GENERATEDISK = (mask2generatedisk == 0)

model = call_gen_disk_flat(theta)

# y_peri,x_peri = np.where(model == np.nanmax(model))

# x_peri = np.mean(x_peri)
# y_peri = np.mean(y_peri)

# circle1 = plt.Circle((x_peri,y_peri), 4, color='b',alpha = 0.8)



a = rad1
c = np.sqrt(dx**2 + dy**2)
eccentricity = c/a
argpe = np.degrees(np.arctan2(dx, dy))

# radius_argpe = rad1 #radius_ellip_mas(argpe, rad1, incl)

model_rot = np.clip(rotate(model, argpe + pa, mode='wrap', reshape = False), 0., None)
argpe_direction = model_rot[DIMENSION//2:,DIMENSION//2]
radius_argpe = np.where(argpe_direction == np.nanmax(argpe_direction))[0]

x_peri_true = radius_argpe*np.cos(np.radians(argpe + pa + 90)) + DIMENSION//2
y_peri_true = radius_argpe*np.sin(np.radians(argpe + pa + 90)) + DIMENSION//2
circle3 = plt.Circle((x_peri_true,y_peri_true), 3, color='g',alpha = 0.8)

print(eccentricity)
print(argpe)
# a few ideas before holydays
# to find the pericenter, replot with the same geom but with a flat SPF
# and just take the maximum of the image
# the pericenter is just the point the closest to the star.
# the eccentricity can be measure by realizing that the
# distance between the foyer-star and the center of the ellispe is
# 'c' in https://fr.wikipedia.org/wiki/Ellipse_(math%C3%A9matiques)
# and we have c = ae where a is R1.


center_image = plt.Circle((DIMENSION//2,DIMENSION//2), 3, color='r')
center_ellipse = plt.Circle((DIMENSION//2 + convert.au_to_pix(dx, PIXSCALE_INS, DISTANCE_STAR) ,DIMENSION//2 - convert.au_to_pix(dy, PIXSCALE_INS, DISTANCE_STAR)), 3, color='y')

fig, ax = plt.subplots()
ax.imshow(model,origin='lower')
ax.add_artist(center_ellipse)
ax.add_artist(center_image)
ax.add_artist(circle3)

plt.show()