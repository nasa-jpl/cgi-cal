# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate clocked pupil images centered in between either row of SPMs

These images are used for calibrating the magnificition and clocking of the
SPMs. The output is at the pupil resolution of the PROPER model at the SPMs,
1000 pixels across.
@author: ajriggs
"""
import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

from cal.pupilfit.pf_util import rotate_shift_downsample_pupil_mask

from cal.util.shapes import circle
from cal.pupilfit.util.insertinto import insertinto

localpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# %% INPUTS FOR GENERATING TEST DATA

whichRow = 'top'  # 'top' or 'bottom' row of SPMs
mag = 1.02  # magnification of the SPMs relative to the beam size expected there.
clockAllDeg = 2.0  # [degrees]
xc = 0  # x-offset [meters]
yc = 0  # y-offset [meters]

nBeamNom = 1000.0  # number of pixels across the unmasked pupil diameter at SPAM in PROPER model
nArray = 2400  # output array width and height [pixels]

flagPlot = True  # whether to plot [boolean]

# %% Properties of the optical system and masks

dBeamNom = 17.0146e-3  # nominal diameter of the unmasked beam arriving at SPAM [meters]
dCircleM = 1.0e-3   # diameter of the fiducial circles [meters]
dyCircM = 7.5e-3  # half the vertical center-to-center distance between fiducial circles [meters]
dxSPM = 9.3e-3  # half the center-to-center distance between SPMs [meters]
aoiDeg = 7.5  # horizontal angle of incidence at SPAM [degrees]

# Note that the SPMs and fiducial circles were made with a 1/cos(aoi)
# horizontal stretch to cancel out the cos(aoi) foreshortening from being
# mounted at an angle.

# %% Generate the measured image

clockAllRad = clockAllDeg*(np.pi/180.0)
aoiRad = aoiDeg*(np.pi/180.0)
nBeam = nBeamNom*mag
dx = dBeamNom / nBeam  # meters per pixel at output

amp = np.zeros((nArray, nArray))

# Read in the high-res SPMs in the top or bottom row
if whichRow.lower() == 'top':
    fnLeft = os.path.join(localpath, 'maskdesigns', 'SPC_20200610_WFOV',
                        'SPM_SPC_20200610_WFOV_1000us3x_binary_uint8.fits')
    fnRight = os.path.join(localpath, 'maskdesigns', 'SPC_20200628_Spec_Rot60',
                    'SPM_SPC_20200628_Spec_Rot60_1000us3x_binary_uint8.fits')
    spmLeft = fits.getdata(fnLeft)
    spmRight = fits.getdata(fnRight)
    extraRotRightSPM = 60  # [degrees]
    resRightSPM = 3000  # [pixels]
elif whichRow.lower() == 'bottom':
    fnLeft = os.path.join(localpath, 'maskdesigns', 'SPC_20200617_Spec',
                        'SPM_SPC_20200617_Spec_1000us3x_binary_uint8.fits')
    fnRight = os.path.join(localpath, 'maskdesigns', 'SPC_20200623_MSWC',
                        'SPM_SPC_20200623_WFOV_982us3x_binary_uint8.fits')
    spmLeft = fits.getdata(fnLeft)
    spmRight = fits.getdata(fnRight)
    extraRotRightSPM = 0  # [degrees]
    resRightSPM = 3*982  # [pixels]
else:
    raise ValueError("whichRow must be 'top' or 'bottom'.")

# Resize and rotate the SPMs and pupil
spmLeftPad = insertinto(spmLeft, (3200, 3200))
spmRightPad = insertinto(spmRight, (3200, 3200))

xcLeft = -np.cos(clockAllRad)*dxSPM*np.cos(aoiRad)   # x center of left SPM [meters]
ycLeft = -np.sin(clockAllRad)*dxSPM*np.cos(aoiRad)  # y center of left SPM [meters]
xcRight = np.cos(clockAllRad)*dxSPM*np.cos(aoiRad)   # x center of right SPM [meters]
ycRight = np.sin(clockAllRad)*dxSPM*np.cos(aoiRad)   # y center of right SPM [meters]

spmLeftMoved = rotate_shift_downsample_pupil_mask(spmLeftPad, 3000, nBeam,
                        (xc+xcLeft)/dx, (yc+ycLeft)/dx, clockAllDeg)
spmRightMoved = rotate_shift_downsample_pupil_mask(spmRightPad, resRightSPM,
                        nBeam, (xc+xcRight)/dx, (yc+ycRight)/dx,
                        extraRotRightSPM+clockAllDeg)

spmLeftMoved = insertinto(spmLeftMoved, (nArray, nArray))
spmRightMoved = insertinto(spmRightMoved, (nArray, nArray))

amp += (spmLeftMoved + spmRightMoved)  # *pupilMoved

spms = spmLeftMoved + spmRightMoved

# Generate the alignment circles
circ1 = circle(nArray, nArray, dCircleM / dx/2.0,
                  (xc - np.sin(clockAllRad)*dyCircM)/dx,
                  (yc + np.cos(clockAllRad)*dyCircM)/dx,
                   nSubpixels=100)
circ2 = circle(nArray, nArray, dCircleM / dx/2.0,
                  (xc + np.sin(clockAllRad)*dyCircM)/dx,
                  (yc - np.cos(clockAllRad)*dyCircM)/dx,
                   nSubpixels=100)
amp += circ1 + circ2
spotMeas = np.abs(amp)**2


# xcImagePix = -120
# ycImagePix = 200
# imageShape = (3000, 801)
unnormFac = 897.2  # factor applied to un-normalize the image for testing

# # De-center the generated pupil image
# spotMeas = insertinto(spotMeas, imageShape)
# spotMeas = np.roll(spotMeas, (ycImagePix, xcImagePix), axis=(0, 1))
spotMeas *= unnormFac

if(flagPlot):
        plt.imshow(spotMeas); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(2)

hdu = fits.PrimaryHDU(spotMeas)
hdu.writeto(('./SPM_alignment_spots_image_%s_row_mag%.3f_rot%.2fdeg.fits' % (whichRow, mag, clockAllDeg)), overwrite=True)
