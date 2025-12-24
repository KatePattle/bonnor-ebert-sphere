x#/usr/bin/python3

import sys

import math
import numpy as np

import scipy.integrate as integrate
from scipy.optimize import fsolve, least_squares
from scipy.ndimage.filters import gaussian_filter

from astropy.io import fits
from astropy.io import ascii
from astropy import wcs
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.patches as patches

from random import randint, uniform

from copy import copy, deepcopy

from starlink import kappa

from reproject import reproject_interp, mosaicking, reproject_exact

import os.path

# Create BE spheres to inject into SCUBA-2 pipeline

# Jun 21 modification: I'm going to make all maps on the 850 grid and reproject to 450.
# They're smoothed to beam-size anyway, so why not?

#########################
# Constants

h = 6.63e-34
kb = 1.38e-23
c = 2.9979e8
G = 6.67e-11
mu = 2.8 
m_h = 1.67e-27
msun = 1.99e30

#########################
# Functions

# Planck function
def planck_nu(T, nu):
    return ((2*h*(nu**3.))/(c**2.))*(1./(np.exp((h*nu)/(kb*T))-1.))

# Dust opacity (Beckwith+1990 formulation)
def kappa_nu(kappa0, beta, nu):
    return kappa0*((nu/(1e12))**beta)

######## Lane-Emden solver for full radial profile

# This sets z=dphi/dxi, making this into two first-order equations rather than one second-order
def solver(U, xi):
    # "Here U is a vector such that y=U[0] and z=U[1]. This function should return [y', z']"
    if xi == 0:
        return [U[1], np.exp(-U[0])] # Am I allowed to do this??  It's singular if I don't
    else:
        return [U[1], np.exp(-U[0]) - 2.*(U[1]/xi)]

def lane_emden(xi): # First input xi **HAS** to be zero b/c that's where we know the limit
    
    # Lane-Emden equation, to solve:
    # (1./(xi**2.)*d/dxi((xi**2.)dpsi/dxi) = exp(-psi)
    # boundary conds: psi(xi=0)=0
    #                 dpsi/dxi|_xi=0 = 0
    
    return integrate.odeint(solver,[0,0],xi)[:,0]   # this returns psi as a function of xi

######## Get a specific value of psi for given xi
######## Makes a range that definitely includes it and interpolates

def psi(xi):

    xirange = np.linspace(0, 2.*xi, 10000)
    
    psirange = lane_emden(xirange)

    return np.interp(xi, xirange, psirange)

######## Bonnor-Ebert density profile
######## r: radius array
######## rho_c: central density
######## xi_0: dimensionless edge radius
######## outside: (optional) sets rho(xi > xi0) to a value of your choice.

def bonnor_ebert_density(r, rho_c, xi_0, T, outside=None):
    
    c_s = np.sqrt((kb*T)/(mu*m_h)) # sound speed
    
    xi = r * np.sqrt((4.*np.pi*G*rho_c)/(c_s**2.)) # dimensionless radius

    psi_func = lane_emden(xi) # calculate psi distribution

    rho = rho_c*np.exp(-psi_func) # turn psi into rho

    if outside == None:
        outrho = rho_c*np.exp(-psi(xi_0)) # set rho(xi > xi0) = rho(xi0)
    else:
        outrho = outside 

    exc = np.where(xi > xi_0)
    rho[exc] = outrho
    
    return rho

######## For given mass, calculate rho_c and xi0
######## This is really a pair of parametric equations in xi
######## want to solve for rho_c and xi0
######## Solving the mass equation in log space because the numbers are all either huge or tiny

def critical_be(invals, logMnorm):

    # Equations to solve:
    # logMnorm - np.log(integral)+0.5*logrho_c = 0
    # np.exp(psi(xi0)) - 14.1 = 0  <------ I'm just believing that rho/rho_c = 14.1 defines the boundary

    logrho_c, xi0 = invals
       
    integral = integrate.quad(lambda xi : np.exp(-psi(xi))*(xi**2.), 0, xi0)[0] # integration over xi
    
    return (logMnorm - np.log(integral) + 0.5*logrho_c, np.exp(psi(xi0)) - 14.1)

###############################################

# First, set up field (get Oph N6 maps)

path = '/export/caroline/kpattle/laptop/nthu/gbs_catalogue/'

mapset850 = ['20130204_00058_s850_extmask_snr3_IR3_cal_coll.fits', '20130205_00064_s850_extmask_snr3_IR3_cal_coll.fits',
             '20130702_00018_s850_extmask_snr3_IR3_cal_coll.fits', '20130702_00029_s850_extmask_snr3_IR3_cal_coll.fits',
             '20130714_00012_s850_extmask_snr3_IR3_cal_coll.fits',
             '20130714_00013_s850_extmask_snr3_IR3_cal_coll.fits'] # '20130702_00031_s850_extmask_snr3_IR3_cal_coll.fits'

mapset450 = ['20130204_00058_s450_extmask_snr3_IR3_cal_coll.fits', '20130205_00064_s450_extmask_snr3_IR3_cal_coll.fits',
             '20130702_00018_s450_extmask_snr3_IR3_cal_coll.fits', '20130702_00029_s450_extmask_snr3_IR3_cal_coll.fits',
             '20130714_00012_s450_extmask_snr3_IR3_cal_coll.fits',
             '20130714_00013_s450_extmask_snr3_IR3_cal_coll.fits'] # '20130702_00031_s450_extmask_snr3_IR3_cal_coll.fits', 


#####################################
# Set a distance for your sources to be placed at

dist_pc = 450. # distance to source in parsecs. Completely up to you.
dist = dist_pc*(3.086e16) # convert to metres


# Specify number of attempts to place each source

placeattempts = np.asarray([100, 100, 150, 200, 200])
#mustplace = np.asarray([10, 10, 20, 40, 40]) # 150 pc
#mustplace = np.asarray([20, 20, 40, 80, 80]) # 300 pc
mustplace = np.asarray([40, 40, 80, 160, 160]) # 450 pc
maxtries = np.asarray([2000, 2000, 8000, 16000, 16000])
#excl_multiplier = np.asarray([12., 10., 8., 6., 4.]) 
#excl_multiplier = np.asarray([18., 15., 12., 9., 3.]) 
excl_multiplier = np.asarray([24., 20., 16., 12., 8.]) 

#####################################
# frequencies

nu850 = (2.9979e8)/(850e-6)
nu450 = (2.9979e8)/(450e-6)

#####################################
# JCMT beam sizes and volumes
# from Dempsey et al. 2013

# 850um:

pri850 = 13.0 #arcsec
sec850 = 48.0 #arcsec

privol850 = 0.75
secvol850 = 0.25

# 450um:

pri450 = 7.9 #arcsec
sec450 = 25.0 #arcsec

privol450 = 0.6
secvol450 = 0.4

##################
# Set BE parameters

M = np.asarray([2.0, 1.0, 0.5, 0.2, 0.1])*msun # 0.1, 0.2, 0.5, 1.0, 2.0

T = 15. # temperature
    
c_s = np.sqrt((kb*T)/(mu*m_h)) # sound speed

prefactor = (1./np.sqrt(4*np.pi))*((c_s/np.sqrt(G))**3.) # annoying prefactor, let's divide it out later


#######################
# Dust opacity properties

kappa0 = 0.01 # m^2kg^-1
beta = 1.8 # Would anyone mind if I made this 1.8?

##################################
#
# What Helen did:
#
# "We constrain all fake sources to lie in angular separation at least
# three Gaussian σ away from the outer 3 arcmin of the map (where the local
# noise is significantly higher) and away from the zone of potential
# emission in the east of the mosaic,defined as two circles of 2.5-arcmin
# radius, with the centers set by eye."
#
# "For any given set of Gaussian parameters (i.e.,
# amplitude and width), we randomly placed 500 sources,
# eliminating those that landed in the edge or possible
# emission zones noted above or those located
# less than 6σ away from a previously placed source."
#
###################################



# 850um:


# Iterate over each mass.  We're going to place a certain number of each mass in the field
for j in range(0, len(M)):
    
    # First, calculate your BE properties
    
    wrapped_crit_be = lambda y : critical_be(y, np.log(M[j]/prefactor)) # lambda-ing the mass into the function
    logrho_c, xi_0 = fsolve(wrapped_crit_be, (-15., 200.)) # solving in log space
    
    # Central density
    rho_c = np.exp(logrho_c)
    
    # Edge radius
    r0 = xi_0 / np.sqrt((4.*np.pi*G*rho_c)/(c_s**2.))
    
    r0_ang = np.degrees(np.arctan2(r0,dist))*3600. # angular size in arcsec
    
    # Check here that BE values give back your input mass

    mcheck = prefactor*(rho_c**(-0.5))*integrate.quad(lambda xi : np.exp(-psi(xi))*(xi**2.), 0, xi_0)[0]

    print('Input mass: {0:.2f}'.format(M[j]/msun))
    print('Mass returned by BE calculator: {0:.2f}'.format(mcheck/msun))
    print('Ratio: {0:.5f}'.format(mcheck/M[j]))
    
    # Make a lookup table for rho

    rdummy = np.linspace(0, r0, 100000)
    rhodummy = bonnor_ebert_density(rdummy, rho_c, xi_0, T)

    ####################################################################################
    # Now we need to make BE profiles at each wavelength

    for w in range(0,2):
        
        for mno in range(0,1): # just do this once and then reproject for other maps

            print(mno)
            
            hdulist850 = fits.open(path+'ophn6_maps/'+mapset850[mno])
            
            image850 = hdulist850[0].data
            
            blank_im850 = deepcopy(image850)
            good = np.where(np.isfinite(image850) == 1)
            blank_im850[good] = 0.
            
            pixsz850 = float(round(3600.*abs(hdulist850[0].header['CDELT1']))) # pick up from header
            
            wcs850 = WCS(hdulist850[0].header) # WCS information
            
            tag = mapset850[mno][0:14]
            
            # 450um:
            
            hdulist450 = fits.open(path+'ophn6_maps/'+mapset450[mno])
            
            image450 = hdulist450[0].data
            
            blank_im450 = deepcopy(image450)
            good = np.where(np.isfinite(image450) == 1)
            blank_im450[good] = 0.
            
            pixsz450 = float(round(3600.*abs(hdulist450[0].header['CDELT1']))) # pick up from header
            
            wcs450 = WCS(hdulist450[0].header) # WCS information
            
            
            if w == 0:
                wl = 850
                image = image850
                wcs = wcs850
                hdulist = hdulist850
                pixsz = pixsz850
                plancknu = planck_nu(T,nu850)
                kappanu = kappa_nu(kappa0, beta, nu850)
                pri = pri850
                privol = privol850
                sec = sec850
                secvol = secvol850            
                blank_im = blank_im850
                mcruns = 1000 # changed from 1000 for 450um 0.1M_sun 300pc (to 10000) - change back later


            # for 450um, we're going to take the density structure that we made at 850
            #  -- which is wavelength-independent -- multiply it through by the factors
            # outside the integral -- smooth and reproject it
            if w == 1:
                wl = 450
                image = image450 # this is set this way to add onto the reprojected maps
                wcs = wcs850
                hdulist=hdulist850
                pixsz = pixsz850
                plancknu = planck_nu(T,nu450)
                kappanu = kappa_nu(kappa0, beta, nu450)
                pri = pri450
                privol = privol450
                sec = sec450
                secvol = secvol450
                blank_im = blank_im850
                mcruns = 1000 #round(((pixsz450/pixsz850)**2.)*mcruns)#

                # test if file already exists
                # if it does, don't bother
                if os.path.isfile(path+'completeness_may21/ophn6_'+tag+
                                  '_singleBE_{:.1f}'.format(M[j]/msun)+
                                  'Msun_{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+
                                  'K_beta{:.1f}'.format(beta)+'_{}'.format(wl)+'.fits'):
                    continue

                #  we need to pick up iarr_save and apply the prefactors etc
                iarr = iarr_save*plancknu*kappanu
                sqa_per_sr = ((180./np.pi)**2.)*(3600.**2.)
                jy = 1e-26
                iarr = (iarr/(sqa_per_sr))*(1000./jy)
                summap = np.nansum(iarr)*(jy/1000.)*(pixsz**2.)
                print('Returned {}'.format(wl)+'um mass after impact parameter calculation: {0:.2f}'.format(hildebrand/msun))
                print('Ratio: {0:.5f}'.format(hildebrand/M[j]))

                # now smooth
                pricomp = privol*gaussian_filter(iarr, (pri/pixsz)/np.sqrt(8.*np.log(2)))
                seccomp = secvol*gaussian_filter(iarr, (sec/pixsz)/np.sqrt(8.*np.log(2)))
            
                iarr_conv = pricomp + seccomp
            
                # Does this still sum to our input mass?
                summap_conv = np.nansum(iarr_conv)*(jy/1000.)*(pixsz**2.)
                hildebrand_conv = (summap_conv*(dist**2.))/(plancknu*kappanu)
                print('Returned {}'.format(wl)+'um mass after smoothing with SCUBA-2 beam: {0:.2f}'.format(hildebrand_conv/msun))
                print('Ratio: {0:.5f}'.format(hildebrand_conv/M[j]))

                # write out maps

                # 1) unconvolved map
                # for 450, immediately get back what we wrote out, reproject, write out again
                fits.writeto(path+'completeness_may21/ophn6_'+tag+'_singleBE_{:.1f}'.format(M[j]/msun)+
                             'Msun_{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+'K_beta{:.1f}'.format(beta)+'_{}'.format(wl)+'_850grid.fits',
                             data = iarr,
                             header = hdulist[0].header,
                             overwrite=True)
                hdulist_0 = fits.open(path+'completeness_may21/ophn6_'+tag+'_singleBE_{:.1f}'.format(M[j]/msun)+
                                      'Msun_{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+
                                      'K_beta{:.1f}'.format(beta)+'_{}'.format(wl)+'_850grid.fits')
                hdu_reproj, fp = reproject_exact(hdulist_0, hdulist450[0].header)
                fits.writeto(path+'completeness_may21/ophn6_'+tag+'_singleBE_{:.1f}'.format(M[j]/msun)+
                             'Msun_{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+'K_beta{:.1f}'.format(beta)+
                             '_{}'.format(wl)+'.fits',
                             data = hdu_reproj,
                             header = hdulist450[0].header,
                             overwrite=True)

                # 2) unconvolved map + image: pointless but never mind

                fits.writeto(path+'completeness_may21/ophn6_'+tag+'_singleBE_{:.1f}'.format(M[j]/msun)+
                                      'Msun_{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+'K_beta{:.1f}'.format(beta)+
                                      '_{}'.format(wl)+'_withS2.fits',
                             data = hdu_reproj+image,
                             header = hdulist450[0].header,
                             overwrite=True)

                # 3) convolved map: IMPORTANT ONE
                
                fits.writeto(path+'completeness_may21/ophn6_'+tag+'_singleBE_{:.1f}'.format(M[j]/msun)+
                             'Msun_{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+
                             'K_beta{:.1f}'.format(beta)+'_{}'.format(wl)+'_conv_850grid.fits',
                             data = iarr_conv,
                             header = hdulist[0].header,
                             overwrite=True)
                hdulist_0 = fits.open(path+'completeness_may21/ophn6_'+tag+'_singleBE_{:.1f}'.format(M[j]/msun)+
                                      'Msun_{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+'K_beta{:.1f}'.format(beta)+
                                      '_{}'.format(wl)+'_conv_850grid.fits')
                hdu_reproj, fp = reproject_exact(hdulist_0, hdulist450[0].header)
                fits.writeto(path+'completeness_may21/ophn6_'+tag+'_singleBE_{:.1f}'.format(M[j]/msun)+
                             'Msun_{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+'K_beta{:.1f}'.format(beta)+
                             '_{}'.format(wl)+'_conv.fits',
                             data = hdu_reproj,
                             header = hdulist450[0].header,
                             overwrite=True)

                # Convolved map + image
                fits.writeto(path+'completeness_may21/ophn6_'+tag+'_singleBE_{:.1f}'.format(M[j]/msun)+
                             'Msun_{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+'K_beta{:.1f}'.format(beta)+
                             '_{}'.format(wl)+'_conv_withS2.fits',
                             data = hdu_reproj+image,
                             header = hdulist450[0].header,
                             overwrite=True)

                print('450 single-sphere map written out')
                
                continue # skipping the rest of the loop for 450
                
            # Convert r0 to pixels
            
            r0_pix = r0_ang/pixsz
            
            # Set up blank arrays to output onto
            
            output = deepcopy(blank_im)
            output_conv = deepcopy(blank_im)
    
            # Image centre
            
            xlen = image.shape[1] # pixels
            ylen = image.shape[0] # pixels
                
            # Get the image centre
            # What we're going to do is take the central sky coordinates and go from there

            cenco = SkyCoord('16h21m13.760s -20d08m20.90s', frame='icrs') 
    
            xcen, ycen = wcs.wcs_world2pix(cenco.ra, cenco.dec, 0)
                    
            # Make BE flux density distribution

            # need to calculate an impact parameter for every pixel so need an mgrid
            # mgrid returns integers from 0 to length given
    
            yarr, xarr = np.mgrid[:image.shape[0], :image.shape[1]]
            
            yarr = yarr - np.floor(ycen)
            xarr = xarr - np.floor(xcen)
                
            # impact parameter array
            parr = np.sqrt((yarr**2.) + (xarr**2.)) # this is in pixels

            ## To skip loop, you'll want to remove this again later
            #continue

            # test if file already exists
            # if it does, don't bother
            if os.path.isfile(path+'completeness_may21/ophn6_'+tag+
                              '_singleBE_{:.1f}'.format(M[j]/msun)+
                              'Msun_{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+
                              'K_beta{:.1f}'.format(beta)+'_{}'.format(wl)+'.fits'):
                continue
            
            # Put in a tolerance; if the integration isn't accurate to within 1%, repeat
            tol = 1.0
            while tol > 0.01:
                # Intensity array, to be filled in
                iarr = deepcopy(parr)
                iarr[:] = np.nan
                # Check whether source is smaller than centre pixel
                # Calculate radii for all four corners of the pixel
                centrepix = np.where(parr == np.nanmin(parr))
                xcorners = np.asarray([xarr[centrepix] - 0.5, xarr[centrepix] - 0.5, xarr[centrepix] + 0.5, xarr[centrepix] + 0.5])
                ycorners = np.asarray([yarr[centrepix] + 0.5, yarr[centrepix] - 0.5, yarr[centrepix] + 0.5, yarr[centrepix] - 0.5])
                rcorners = dist*np.tan(np.radians((np.sqrt((xcorners**2.) + (ycorners**2.))*pixsz)/3600.))
                # If all corners are > r0 then skip this bit
                if all(rc < r0 for rc in rcorners):
                    # Integrate over impact parameter
                    for k in range(0, len(yarr[:,0])):
                        for l in range(0, len(xarr[0,:])):
                        
                            # Have settled on "Blunt-implement with frills" method:
                            # Blunt implement: Use trapezium rule to calculate the emission along the line of sight
                            # Frills: Monte Carlo positions inside pixel to get average value
                            # Frills are necessary to prevent over/underestimation, particularly for small sources
                        
                            # If the pixel hasn't already been filled by symmetry then start
                            if np.isfinite(iarr[k,l]) == 0:
                                # Calculate radii for all four corners of the pixel
                                xcorners = np.asarray([xarr[k,l] - 0.5, xarr[k,l] - 0.5, xarr[k,l] + 0.5, xarr[k,l] + 0.5])
                                ycorners = np.asarray([yarr[k,l] + 0.5, yarr[k,l] - 0.5, yarr[k,l] + 0.5, yarr[k,l] - 0.5])
                                rcorners = dist*np.tan(np.radians((np.sqrt((xcorners**2.) + (ycorners**2.))*pixsz)/3600.))
                                # If all corners are > r0 then set the pixel to zero
                                if all(rc > r0 for rc in rcorners):
                                    iarr[np.where(parr == parr[k,l])] = 0. # fills in all pixels with this impact parameter
                                else: # Otherwise start a Monte Carlo
                                    for n in range(0, mcruns): # not sure how big this needs to be: >= 1000 at 850um
                                        # Pick random position inside pixel
                                        xval = uniform(xarr[k,l] - 0.5, xarr[k,l] + 0.5)
                                        yval = uniform(yarr[k,l] - 0.5, yarr[k,l] + 0.5)
                                        # Calculate impact parameter
                                        pval = np.sqrt((xval**2.) + (yval**2.))
                                        imp = dist*np.tan(np.radians((pval*pixsz)/3600.))
                                        # If impact parameter > r0 set intensity to zero
                                        if imp > r0:
                                            integ = 0
                                        else: # Otherwise perform the integration
                                            # Make array of distances along the LOS from 0 to the edge of the sphere
                                            # (relative to the plane of the centre of the sphere)
                                            xvals = np.linspace(0, np.sqrt((r0**2.)-(imp**2.)), 10000)
                                            # Get radii for each of these distances
                                            rvals = np.sqrt((xvals**2.) + (imp**2.))
                                            # Get density values rho(x) by interpolating rho(r) lookup for r(x) values
                                            yvals = np.interp(rvals, rdummy, rhodummy)
                                            # Integrate under rho(x) curve using trapezium rule
                                            integ = 2.*np.trapz(yvals, xvals)
                                        if n == 0: # create/append to array of intensity values
                                            integ_mc = np.asarray([integ]) 
                                        else:
                                            integ_mc = np.append(integ_mc, integ)
                                    # intensity in this pixel is the mean of the intensity values found
                                    iarr[np.where(parr == parr[k,l])] = np.mean(integ_mc) # fills in all pixels with this impact parameter

                            if l == len(xarr[0,:])-1:
                                print('Integration {:.1f}'.format((float(k+1)/float(len(yarr[:,0])))*100.)+'% complete')
                else:
                    # Special case for if source is contained entirely in one pixel
                    #if iarr.size == iarr[np.where(iarr == 0.)].size:
                    print("The source is smaller than a pixel; let's do centre pixel only")
                    centrepix = np.where(parr == np.nanmin(parr))
                    sqa_per_sr = ((180./np.pi)**2.)*(3600.**2.)
                    iarr[:] = 0.
                    iarr[centrepix] = (M[j]/(dist**2.))*(sqa_per_sr/(pixsz**2.))


                ###########
                # What we have here is a density structure which is independent of wavelength
                # Let's save it
                iarr_save = deepcopy(iarr)
                               
                # Multiply by factors which can be taken out of the integral
                iarr = iarr*plancknu*kappanu
                
                # Convert output map units: Wm^-2Hz^-1sr^-1 -> mJy/arcsec^2
            
                sqa_per_sr = ((180./np.pi)**2.)*(3600.**2.)
                jy = 1e-26
                iarr = (iarr/(sqa_per_sr))*(1000./jy)
                #print(np.nansum(iarr))
            
                # Cross-check here: does this sum to our input mass?
                summap = np.nansum(iarr)*(jy/1000.)*(pixsz**2.)
                #print(summap/msun)
                hildebrand = (summap*(dist**2.))/(plancknu*kappanu)
                print('Returned {}'.format(wl)+'um mass after impact parameter calculation: {0:.2f}'.format(hildebrand/msun))
                print('Ratio: {0:.5f}'.format(hildebrand/M[j]))

                tol = abs(hildebrand/M[j] - 1.0) 
                if tol > 0.01:
                    print('Not good enough, retrying.')
                else:
                    print("Accepting, let's move on.")
                    

            # test again if file already exists
            # if it does, continue not to bother
            # FFS
            if os.path.isfile(path+'completeness_may21/ophn6_'+tag+
                              '_singleBE_{:.1f}'.format(M[j]/msun)+
                              'Msun_{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+
                              'K_beta{:.1f}'.format(beta)+'_{}'.format(wl)+'.fits'):
                continue
            
            # Convolve this with the SCUBA-2 beam
            
            pricomp = privol*gaussian_filter(iarr, (pri/pixsz)/np.sqrt(8.*np.log(2)))
            seccomp = secvol*gaussian_filter(iarr, (sec/pixsz)/np.sqrt(8.*np.log(2)))
            
            iarr_conv = pricomp + seccomp
            
            # Does this still sum to our input mass?
            summap_conv = np.nansum(iarr_conv)*(jy/1000.)*(pixsz**2.)
            hildebrand_conv = (summap_conv*(dist**2.))/(plancknu*kappanu)
            print('Returned {}'.format(wl)+'um mass after smoothing with SCUBA-2 beam: {0:.2f}'.format(hildebrand_conv/msun))
            print('Ratio: {0:.5f}'.format(hildebrand_conv/M[j]))

            # Save map for future use 
                
            # Write the output map to file
            print('Writing out maps...')
            
            if wl == 850:
                fits.writeto(path+'completeness_may21/ophn6_'+tag+'_singleBE_{:.1f}'.format(M[j]/msun)+
                             'Msun_{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+'K_beta{:.1f}'.format(beta)+'_{}'.format(wl)+'.fits',
                             data = iarr,
                             header = hdulist[0].header,
                             overwrite=True)
            
            # Add the output map to the SCUBA-2 data and write to file
            if wl == 850:
                fits.writeto(path+'completeness_may21/ophn6_'+tag+'_singleBE_{:.1f}'.format(M[j]/msun)+
                             'Msun_{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+'K_beta{:.1f}'.format(beta)+'_{}'.format(wl)+'_withS2.fits',
                             data = iarr+image,
                             header = hdulist[0].header,
                             overwrite=True)

            # Write the convolved output map to file: THIS IS THE IMPORTANT ONE
            if wl == 850:
                fits.writeto(path+'completeness_may21/ophn6_'+tag+'_singleBE_{:.1f}'.format(M[j]/msun)+
                             'Msun_{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+'K_beta{:.1f}'.format(beta)+'_{}'.format(wl)+'_conv.fits',
                             data = iarr_conv,
                             header = hdulist[0].header,
                             overwrite=True)
                print('850 single-sphere map written out')
                
            # Add the convolved output map to the SCUBA-2 data and write to file
            if wl == 850:
                fits.writeto(path+'completeness_may21/ophn6_'+tag+'_singleBE_{:.1f}'.format(M[j]/msun)+
                             'Msun_{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+'K_beta{:.1f}'.format(beta)+
                             '_{}'.format(wl)+'_conv_withS2.fits',
                             data = iarr_conv+image,
                             header = hdulist[0].header,
                             overwrite=True)



##############################################################################
# THIS IS WHERE WE START PLACING SOURCES


# For first 850um map, place sources

# Get map, WCS info

for w in range(0, 1):

    if w == 0:
        wl = 850
        image = image850
        wcs = wcs850
        hdulist = hdulist850
        pixsz = pixsz850
        blank_im = blank_im850
        mask = deepcopy(blank_im850)
        image450 = deepcopy(image)
    #if w == 1:
    #    wl = 450
    #    image = image450
    #    wcs = wcs450
    #    hdulist=hdulist450
    #    pixsz = pixsz450
    #    blank_im = blank_im450


    # Check if map exists already, if so don't bother
    if os.path.isfile(path+'completeness_may21/ophn6_'+tag+'_BEfield_'+
                      '{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+
                      'K_beta{:.1f}'.format(beta)+'_{}'.format(wl)+'_conv.fits'):
        continue

    mask[:] = 1

    # Approximate emission centres
    # Changed to exactly match Helen, previously 16h21m34.0s -20d00m19.3s
    emco1 = SkyCoord('16h21m35.251s -19d59m07.28s', frame='icrs')

    xcen, ycen = wcs.wcs_world2pix(cenco.ra, cenco.dec, 0)
    xlen = image.shape[1] # pixels
    ylen = image.shape[0] # pixels
    
    xem1, yem1 = wcs.wcs_world2pix(emco1.ra, emco1.dec, 0)
    xem1 = xem1 - xcen
    yem1 = yem1 - ycen
    
    # Changed to exactly match Helen, previously 16h21m48.9s -20d06m37.13s
    emco2 = SkyCoord('16h21m49.115s -20d06m37.12s', frame='icrs')
                
    xem2, yem2 = wcs.wcs_world2pix(emco2.ra, emco2.dec, 0)
    xem2 = xem2 - xcen
    yem2 = yem2 - ycen

    # 850um: create list of positions for sources. Do this for 450um simultaneously, regrid at the end
    if w == 0:
    
        none_placed = 1 # "Are we started yet?" flag
        
        for j in range(0, len(M)):
            
            print(M[j])
            
            # Source, etc. separations depend on larger of SCUBA-2 850um beam/BE size
            excl_radius = max((pri/pixsz)/np.sqrt(8.*np.log(2)), r0_pix)
            print(excl_radius)

            
            hdu_use = fits.open(path+'completeness_may21/ophn6_'+tag+'_singleBE_{:.1f}'.format(M[j]/msun)+
                                'Msun_{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+'K_beta{:.1f}'.format(beta)+
                                '_{}'.format(wl)+'.fits')

            iarr = hdu_use[0].data

            # Get 450um BE sphere on 850 grid
            hdu_use450 = fits.open(path+'completeness_may21/ophn6_'+tag+'_singleBE_{:.1f}'.format(M[j]/msun)+
                                   'Msun_{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+'K_beta{:.1f}'.format(beta)+
                                   '_450_850grid.fits')
            
            iarr450 = hdu_use450[0].data

            print('Comparing array sizes:')
            print(image.shape)
            print(iarr.shape)
            
            hdu_conv = fits.open(path+'completeness_may21/ophn6_'+tag+'_singleBE_{:.1f}'.format(M[j]/msun)+
                                 'Msun_{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+'K_beta{:.1f}'.format(beta)+
                                 '_{}'.format(wl)+'_conv.fits')

            iarr_conv = hdu_conv[0].data

            hdu450_conv = fits.open(path+'completeness_may21/ophn6_'+tag+'_singleBE_{:.1f}'.format(M[j]/msun)+
                                 'Msun_{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+'K_beta{:.1f}'.format(beta)+
                                 '_450_conv_850grid.fits')

            iarr450_conv = hdu450_conv[0].data            

            yarr1, xarr1 = np.mgrid[:iarr.shape[0], :iarr.shape[1]]

            xarr0 = xarr.astype(int)
            yarr0 = yarr.astype(int)
            
            if j == 0:
                output = deepcopy(iarr)
                output[:] = 0
                output_conv = deepcopy(iarr)
                output_conv[:] = 0
                output450 = deepcopy(iarr450)
                output450[:] = 0
                output450_conv = deepcopy(iarr450)
                output450_conv[:] = 0                
        
            # Place source at (semi-)random positions on the map
            #for i in range(0, placeattempts[j]):
            i = 0
            p = 0
            while p < mustplace[j] and i < maxtries[j]:

                i += 1
                #print(i)

                # Randomly select x and y coordinates
                xco = xarr0[0,randint(0, xlen-1)] # changed this from xarr to xarr1, 22/06/21
                yco = yarr0[randint(0, ylen-1),0] # ditto this
                    
                # Skip if too close to map edge
                centre_r = np.sqrt((xco**2.) + (yco**2.))                     
                if centre_r > (1800./(2*pixsz)) - (3.*excl_radius):
                    #print('Too close to edge')
                    continue

                # Skip if too close to emission
                emission_r1 = np.sqrt(((xco-xem1)**2.) + ((yco-yem1)**2.))
                if emission_r1 < ((2.5*60.)/pixsz) + (3.*excl_radius):
                    #print('Too close to emission zone 1')
                    continue
                emission_r2 = np.sqrt(((xco-xem2)**2.) + ((yco-yem2)**2.))
                if emission_r2 < ((2.5*60.)/pixsz) + (3.*excl_radius):
                    #print('Too close to emission zone 2')
                    continue

                ## Skip if too close to a source placed on a previous round
                #if np.isfinite(mask[yco, xco]) == 0.:
                #    print('Too close to a placed source')
                #    continue
                        
                # is this the first placed object?
                if none_placed == 1:
                    xes850 = np.asarray(xco)
                    ys850 = np.asarray(yco)
                    masses850 = np.asarray(M[j])
                    xclrd850 = np.asarray(excl_radius)
                    xclmult850 = np.asarray(excl_multiplier[j])
                    none_placed = 0
                    print('Placed!')
                    p += 1
                    print(p)
                else:
                    # compare x and y coordinates to list
                    crossmatch_r = np.sqrt(((xco-xes850)**2.) + ((yco-ys850)**2.))
                    # continue if necessary
                    # Conditions:
                    # within 6* excl_rad of same-mass source (this is probably overkill if r0 >> beam)
                    if xes850.size == 1:
                        crosscheck = np.asarray([crossmatch_r - (xclmult850*xclrd850) - (excl_multiplier[j]*excl_radius)])
                    else:
                        crosscheck = np.asarray(crossmatch_r - (xclmult850*xclrd850) - (excl_multiplier[j]*excl_radius)) #(25.*xclrd850)
                    #if np.min(crossmatch_r) < 6.*excl_radius:
                    if any(r < 0 for r in crosscheck):
                        #print('Too close to a placed source')
                        continue
                    #otherwise add x and y coords to list
                    xes850 = np.append(xes850, xco)
                    ys850 = np.append(ys850, yco)
                    masses850 = np.append(masses850, M[j])
                    xclrd850 = np.append(xclrd850, excl_radius)
                    xclmult850 = np.append(xclmult850, excl_multiplier[j])
                    print('Placed!')
                    p += 1
                    print(p)
                            
                iarr_shift = np.roll(deepcopy(iarr), (yco, xco), axis=(0,1)) # shift iarr to be centred on these coordinates
                iarr_conv_shift = np.roll(deepcopy(iarr_conv), (yco, xco), axis=(0,1))

                iarr450_shift = np.roll(deepcopy(iarr450), (yco, xco), axis=(0,1)) # shift iarr to be centred on these coordinates
                iarr450_conv_shift = np.roll(deepcopy(iarr450_conv), (yco, xco), axis=(0,1))
                        
                # Add this intensity distribution to the output maps
                output = output + iarr_shift
                output_conv = output_conv + iarr_conv_shift

                output450 = output450 + iarr450_shift
                output450_conv = output450_conv + iarr450_conv_shift

                # Blank out relevant bit of mask (3*excl_radius, let's say)
                #source_dist = ((yarr1 - yco)**2. + (xarr1 - xco)**2.)
                #mask[np.where(source_dist < 6.*excl_radius)] = np.nan

                # Blank out anywhere in mask where brightness > 0.0001*max(BE distribution)
                mask[np.where(output > 0.0001*np.nanmax(iarr))] = np.nan
            
        print('Sources placed: {}'.format(len(xes850)))

        raco, decco = wcs850.wcs_pix2world(xes850, ys850, 0)

        #######################################################
        # Here we need to write out a list of positions and masses
        f = open(path+'completeness_may21/source_positions_{}'.format(dist_pc)+
                 'pc_{:.1f}'.format(T)+'K_beta{:.1f}'.format(beta)+'.csv', "w")
        f.write('RA,Dec,X(850),Y(850),Mass(M_sun)\n')
        for i in range(0, len(xes850)):
            f.write('{0:.5f},{1:.5f},{2:.5f},{3:.5f},{4:.5f}\n'.format(raco[i],decco[i],xes850[i],ys850[i],masses850[i]))
        f.close()
            
        ###############################################################
        # Write the output maps to file
        fits.writeto(path+'completeness_may21/ophn6_'+tag+'_BEfield_'+
                     '{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+
                     'K_beta{:.1f}'.format(beta)+'_{}'.format(wl)+'.fits',
                     data = output,
                     header = hdulist[0].header,
                     overwrite=True)

        fits.writeto(path+'completeness_may21/ophn6_'+tag+'_BEfield_'+
                     '{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+
                     'K_beta{:.1f}'.format(beta)+'_450_850grid.fits',
                     data = output450,
                     header = hdulist[0].header,
                     overwrite=True)
            
        # Write the convolved output maps to file
        fits.writeto(path+'completeness_may21/ophn6_'+tag+'_BEfield_'+
                     '{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+
                     'K_beta{:.1f}'.format(beta)+'_{}'.format(wl)+'_conv.fits',
                     data = output_conv,
                     header = hdulist[0].header,
                     overwrite=True)

        fits.writeto(path+'completeness_may21/ophn6_'+tag+'_BEfield_'+
                     '{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+
                     'K_beta{:.1f}'.format(beta)+'_450_conv_850grid.fits',
                     data = output450_conv,
                     header = hdulist[0].header,
                     overwrite=True)

        #############
        # Regrid the 450um maps to the 450um grid

        hdulist_0 = fits.open(path+'completeness_may21/ophn6_'+tag+'_BEfield_'+
                              '{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+
                              'K_beta{:.1f}'.format(beta)+'_450_850grid.fits')
        
        hdu_reproj, fp = reproject_exact(hdulist_0, hdulist450[0].header)
        fits.writeto(path+'completeness_may21/ophn6_'+tag+'_BEfield_'+
                     '{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+
                     'K_beta{:.1f}'.format(beta)+'_450.fits',
                     data = hdu_reproj,
                     header = hdulist450[0].header,
                     overwrite=True)

        hdulist_0 = fits.open(path+'completeness_may21/ophn6_'+tag+'_BEfield_'+
                              '{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+
                              'K_beta{:.1f}'.format(beta)+'_450_conv_850grid.fits')
        
        hdu_reproj, fp = reproject_exact(hdulist_0, hdulist450[0].header)
        fits.writeto(path+'completeness_may21/ophn6_'+tag+'_BEfield_'+
                     '{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+
                     'K_beta{:.1f}'.format(beta)+'_450_conv.fits',
                     data = hdu_reproj,
                     header = hdulist450[0].header,
                     overwrite=True)        
            
            
#################################################
# Reproject made maps to other wavelengths

for mno in range(1, len(mapset850)):
    
    tag = mapset850[mno][0:14]
    
    for w in range(0, 2):

        if w == 0:
            wl = '850'
        else:
            wl = '450'

        if os.path.isfile(path+'completeness_may21/ophn6_'+tag+
                          '_BEfield_'+
                          '{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+
                          'K_beta{:.1f}'.format(beta)+'_{}'.format(wl)+'_conv.fits'):
            continue
                
        hdulist_0 = fits.open(path+'completeness_may21/ophn6_'+mapset850[0][0:14]+'_BEfield_'+
                              '{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+
                              'K_beta{:.1f}'.format(beta)+'_{}'.format(wl)+'_conv.fits')

        if w == 0:
            hdulist_1 = fits.open(path+'ophn6_maps/'+mapset850[mno])
        else:
            hdulist_1 = fits.open(path+'ophn6_maps/'+mapset450[mno])

        hdu_reproj, fp = reproject_exact(hdulist_0, hdulist_1[0].header)

        # Write the convolved output map to file
        fits.writeto(path+'completeness_may21/ophn6_'+tag+
                     '_BEfield_'+
                     '{}'.format(dist_pc)+'pc_{:.1f}'.format(T)+
                     'K_beta{:.1f}'.format(beta)+'_{}'.format(wl)+'_conv.fits',
                     data = hdu_reproj,
                     header = hdulist_1[0].header,
                     overwrite=True)

    
