import numpy as np
import math as mt


def kowalsky(a, ecc, pa, E_offset, N_offset):

    # Translated by Johan Mazoyer from IDL routine kowalsky.pro written by Christopher Stark 
    # Deprojects an ellipse using the Kowalsky method as detailed by Smart 1930.
    # Smart 1930 derives the following assuming the star/focus is at the origin.
    # The ellipse is assumed to have been centered at (0,0), rotated by an
    # angle, then shifted to a new center, i.e. rotation before
    # translation, NOT THE OTHER WAY AROUND.

    # INPUT: (Observed/projected ellipse parameters)
    # a = semi-major axis
    # ecc = eccentricity
    # pa = position angle in degrees measured E of N
    # E_offset = center of ellipse in the East direction (+ = Eastward)
    # N_offset = center of ellipse in the North direction (+ = Northward)

    # OUTPUT: (Deprojected ellipse parameters)
    # true_a = semi-major axis
    # true_ecc = eccentricity
    # argperi = argument of pericenter, the angle the true ellipse is
    #               rotated by prior to inclining (degrees)
    # inc = inclination (degrees)
    # longnode = longitude of the ascending node on the sky measured E of N (degrees)

    # The inputs are in terms of PA measured E of N, and E and N offsets.
    # Here we work in x and y coords (x = West, y = North)

    dx = - E_offset
    dy = N_offset
    temppa = pa + 90

    #Define some variables
    parad = np.radians(temppa)
    cpa = np.cos(parad)
    spa = np.sin(parad)
    cpa2 = cpa*cpa
    spa2 = spa*spa
    oneoa2 = 1./(a*a)
    b = a * np.sqrt(1.- ecc*ecc)
    oneob2 = 1./(b*b)

    #The general equation for an ellipse is given by
    # A0 * x**2 + 2 * H0 * x * y + B0 * y**2 + 2 * G0 * x + 2 * F0 * y + 1 = 0
    #For an ellipse rotated CCW by angle phi, then centered at (dx, dy), we have:
    # A*x**2 + 2*H*x*y + B*y**2 - 2*(A*dx+H*dy)*x - 2*(B*dy+H*dx)*y + A*dx**2+2*H*dx*dy+B*dy**2-1 = 0
    #where
    #A = ( cos(phi)**2 / a**2 + sin(phi)**2 / b**2 )
    #B = ( sin(phi)**2 / a**2 + cos(phi)**2 / b**2 )
    #H = cos(phi) * sin(phi) * (1/a**2 - 1/b**2)
    #With that in mind...
    A0 = ( cpa2 * oneoa2 + spa2 * oneob2 )
    B0 = ( spa2 * oneoa2 + cpa2 * oneob2 )
    H0 = cpa * spa * (oneoa2 - oneob2)
    F0 = - (B0 * dy + H0 * dx) 
    G0 = - (A0 * dx + H0 * dy)
    f = A0 * dx * dx + 2 * H0 * dx * dy + B0 * dy * dy - 1
    A0 /= f
    B0 /= f
    H0 /= f
    F0 /= f
    G0 /= f
    F02 = F0 * F0
    G02 = G0 * G0

    #First we calculate the longitude of the ascending node
    twolongnode = np.arctan2(-2*(F0*G0 - H0) , (F0**2 - G0**2 + A0 - B0)) #big Omega in Smart 1930
    if twolongnode < 0:
        twolongnode += 2*np.pi #long. of asc. node is between 0 and 180 according to Smart 1930
    stln = np.sin(twolongnode)
    fgmh = F0*G0-H0
    if (stln/abs(stln)) * (fgmh/abs(fgmh)) > 0: #if they are not opposite signs, add pi
        print("ERROR in kowalsky.py:  2*longnode must have opposite sign of F0*G0-H0.")
        raise
    longnode = twolongnode * 0.5

    #Now for the inclination
    tan2iop2 = (F02 - G02 + A0 - B0) / np.cos(twolongnode) #a quantity we need to calculate temporarily
    p = np.sqrt(2./(F02 + G02 - A0 - B0 - tan2iop2)) #another quantity
    inc = np.arctan(np.sqrt(tan2iop2 * p * p)) #i in Smart 1930   <---this is the abs value of inc

    #Now for the argument of periastron, the angle that determines the axis of inclination
    argperi = np.arctan2((G0*np.sin(longnode) - F0*np.cos(longnode)) * np.cos(inc) ,- (G0*np.cos(longnode) + F0*np.sin(longnode))) #little omega in Smart 1930

    #Now for the eccentricity
    true_ecc = - p * (G0*np.cos(longnode) + F0*np.sin(longnode)) / np.cos(argperi) #e in Smart 1930
    if true_ecc < 0:
        print('Adding pi to argperi...')
        argperi += np.pi
        true_ecc = abs(true_ecc)

    #Finally, the semi-major axis
    true_a = p / (1. - true_ecc*true_ecc) #a in Smart 1930

    #Convert angles to degrees
    rad2deg = 180./np.pi
    argperi *= rad2deg
    inc *= rad2deg
    longnode *= rad2deg

    longnode += 90. #right now longnode is measured N of W, we add 90 to get E of N
    if longnode > 180:
        longnode -= 180. #limit it to 0 - 180

    return true_a, true_ecc, argperi, inc, longnode