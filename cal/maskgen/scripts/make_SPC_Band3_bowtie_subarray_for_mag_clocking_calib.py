# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate a clocked image of the two rotated SPC Band 3 FPMs in a row.

This will go into the Phase C PROPER model for functional tests of calibration
algorithms.

@author: ajriggs
"""
import os
import numpy as np
# from numpy.fft import fft2, fftshift, ifft2, ifftshift
from astropy.io import fits
# from scipy.interpolate import RectBivariateSpline
# from scipy.ndimage import rotate
import matplotlib.pyplot as plt

from cal.util.shapes import circle
from cal.pupilfit.util.insertinto import insertinto
from cal.maskgen.maskgen import rotate_shift_downsample_amplitude_mask
# from maskgen.mg_util import ceil_odd

localpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# %% INPUTS FOR GENERATING TEST DATA

clockAllDeg = -1.5  # clocking of the substrate [degrees]
mag = 1.01  # FPM magnification relative to the beam size expected there.

lam = 730e-9
pplFPM = 4.0*1024/1000.  # pixels per lam/D at FPAM in PROPER model. Constant with wavelength if being used with FFTs

rotBowtieDeg = 60
shapeOut = (1200, 1200)
padFac = 1.2
xcAll = 0  # x-offset of everything [output pixels]
ycAll = 0  # y-offset of everything [output pixels]

flagPlot = True  # whether to plot [boolean]

# %% Fixed properties of the optical system and masks

fnum = 32.22  # f-number at FPAM
aoiDeg = 5.5  # horizontal angle of incidence at FPAM [degrees]
DeltaXMicron = 3.50e3  # [microns]
DeltaYMicron = 3.05e3  # [microns]
diamPinholeBand4 = 0.5*0.825*fnum

# Note that the FPMs were made with a 1/cos(aoi)
# horizontal stretch to cancel out the cos(aoi) foreshortening from being
# mounted at an angle.

# %% Generate the mask array representation at the FPM plane

# Unit conversions
flamD = fnum*lam*1e6  # microns per lambda/D. grows with wavelength
umPerPix = flamD / pplFPM  # microns per pixel at FPM. grows with wavelength
clockAllRad = clockAllDeg*(np.pi/180.0)
aoiRad = aoiDeg*(np.pi/180.0)


# Read in the high-res bowtie
fnBowtie = os.path.join(localpath, 'maskdesigns', 'SPC_20200617_Spec',
                        'FPM_SPC_20200617_Spec_25pixPerLamD.fits')
bowtie = fits.getdata(fnBowtie)
pplFile = 25  # pixels per lambda/D in file

xcLeft = xcAll - mag*np.cos(clockAllRad)*DeltaXMicron/2/umPerPix*np.cos(aoiRad)   # x center of left mask [pixels]
ycLeft = ycAll - mag*np.sin(clockAllRad)*DeltaXMicron/2/umPerPix  # y center of left mask [pixels]
xcRight = xcAll + mag*np.cos(clockAllRad)*DeltaXMicron/2/umPerPix*np.cos(aoiRad)   # x center of right mask [pixels]
ycRight = ycAll + mag*np.sin(clockAllRad)*DeltaXMicron/2/umPerPix   # y center of right mask [pixels]
xcBottom = xcAll + mag*np.sin(clockAllRad)*DeltaYMicron/umPerPix*np.cos(aoiRad)   # x center of bottom mask [pixels]
ycBottom = ycAll - mag*np.cos(clockAllRad)*DeltaYMicron/umPerPix  # y center of left bottom [pixels]
xcTop = xcAll - mag*np.sin(clockAllRad)*DeltaYMicron/umPerPix*np.cos(aoiRad)   # x center of top mask [pixels]
ycTop = ycAll + mag*np.cos(clockAllRad)*DeltaYMicron/umPerPix  # y center of top bottom [pixels]

maskLeft = rotate_shift_downsample_amplitude_mask(bowtie,
                              rotBowtieDeg+clockAllDeg, mag*pplFPM/pplFile,
                              xcLeft, ycLeft, padFac)

maskRight = rotate_shift_downsample_amplitude_mask(bowtie,
                              rotBowtieDeg+clockAllDeg, mag*pplFPM/pplFile,
                              xcRight, ycRight, padFac)

maskBottom = rotate_shift_downsample_amplitude_mask(bowtie,
                              rotBowtieDeg+clockAllDeg, mag*pplFPM/pplFile,
                              xcBottom, ycBottom, padFac)

roiRadiusPix = diamPinholeBand4 / 2 / umPerPix
maskTop = circle(shapeOut[1], shapeOut[0], roiRadiusPix, xcTop, ycTop)


# Resize and rotate the SPMs and pupil
maskLeftPad = insertinto(maskLeft, shapeOut)
maskRightPad = insertinto(maskRight, shapeOut)
maskBottomPad = insertinto(maskBottom, shapeOut)
maskTopPad = insertinto(maskTop, shapeOut)

mask = maskLeftPad + maskRightPad + maskBottomPad + maskTopPad

if(flagPlot):
        plt.imshow(mask); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(3)

hdu = fits.PrimaryHDU(mask)
fn = ('fpm_alignment_masks_mag%.3f_rot%.2fdeg.fits' % (mag, clockAllDeg))
hdu.writeto(os.path.join(localpath, 'scripts', fn), overwrite=True)
