# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:01:59 2020

@author: ajriggs

Tests:
    1. Check that energy is conserved
"""
# import sys
# sys.path.insert(0,"/Users/ajriggs/Repos/falco-python/")

import numpy as np
# import yaml
# from astropy.io import fits  # for reading in FITS file test data
import matplotlib.pyplot as plt

import conjugate.conjugate as conjugate
from util.insertinto import insertinto as inin


# %%

windowWidth = 3

arrayIn = np.zeros((3, 3))
arrayIn[1, 1] = 1

arrayIn = np.zeros((12, 12))
arrayIn[1::3, 1::3] = 1

# arrayIn = np.ones((10, 10))
# arrayIn[5, 5] = 0

# arrayIn[::2,:] = 0
# arrayIn[:,1::2] = 0
# arrayIn = inin(arrayIn, (12,12))

arrayOut = conjugate.smooth_surface(arrayIn, windowWidth)

plt.figure(1); plt.imshow(arrayIn); plt.colorbar(); plt.title('Array In'); plt.pause(0.1);
plt.figure(2); plt.imshow(arrayOut); plt.colorbar(); plt.title('Array Out'); plt.pause(5);

np.sum(arrayIn)

# %%


# smooth_surface(arrayIn, windowWidth):
#     """
#     Smooth values in a 2-D array via convolution.
    
#     Function to smooth values in a 2-D array via convolution with box
#     kernel of width windowWidth.
    
#     Parameters
#     ----------
#     arrayIn : array_like
#         2-D, real-valued array to be smoothed
#     windowWidth : float
#         width of the convolution kernel for smoothing

#     Returns
#     -------
#     smoothedArray : numpy ndarray
#         Smoothed version of the input 2-D map. Same shape as input.
# """


# %% Various Inputs

# dm_fn = 'testdata/dm.yaml'
# lam = 575e-9  # used only for height-to-phase or vice-versa

# nBeam = 386.
# maxNollZern = 11

# flagUseFCM = True
# flagUseDM1 = True
# flagUseDM2 = True
# flagUseAboveZ11 = True

# %% This part should be replaced with a call to AMPTHRESH

# dm_fn = 'testdata/dm.yaml'
# dxPupil = 0
# dyPupil = 0
# wfeMeas = fits.getdata('./testdata/wfe_xc0.00_yc0.00.fits')
# pupil = fits.getdata('./testdata/amp_xc0.00_yc0.00.fits')

# dm_fn = 'testdata/dm_offset.yaml'
# dxPupil = 20.3  # pixels
# dyPupil = -10.5  # pixels
# wfeMeas = fits.getdata('./testdata/wfe_xc20.30_yc-10.50.fits')
# pupil = fits.getdata('./testdata/amp_xc20.30_yc-10.50.fits')

# thresh = 0.1
# usablePixelMap = pupil.copy()
# usablePixelMap[pupil > thresh] = 1
# usablePixelMap[pupil <= thresh] = 0
# usablePixelMap = usablePixelMap.astype(bool)

# %% Load DM Calibration Data to get "nact" and make total voltage maps
# try:
#     dmConfig = yaml.safe_load(open(dm_fn, "r"))
# except:
#     raise Exception('Error loading yaml file with DM data.')
# dm = dmConfig["dms"]
# dm1r = dm[0]['registration']
# dm2r = dm[1]['registration']

# Vdm1In = 50.*np.ones((dm1r['nact'], dm1r['nact']))
# Vdm2In = 50.*np.ones((dm2r['nact'], dm2r['nact']))

# %% Run CONJUGATE

# Vdm1Out, Vdm2Out, dV1, dV2, surfDesDM1, surfDesDM2 = \
# conjugate.conjugate(Vdm1In, Vdm2In, wfeMeas, usablePixelMap, nBeam, dxPupil,
#                     dyPupil, dm_fn, lam, maxNollZern, flagUseFCM, flagUseDM1,
#                     flagUseDM2, flagUseAboveZ11)

# %% Verify that the output conjugates the input

# flagPlot = True

# # Load DM Calibration Data
# try:
#     dmConfig = yaml.safe_load(open(dm_fn, "r"))
# except:
#     raise Exception('Error loading yaml file.')
# dm = dmConfig["dms"]
# dm1r = dm[0]['registration']
# dm1gainmap = fits.getdata(dm[0]['voltages']['gainfn'])
# dm2gainmap = fits.getdata(dm[1]['voltages']['gainfn'])
# dm2r = dm[1]['registration']
# dm1phase = conjugate.volts_to_dmh(dm1gainmap, dV1, lam)
# dm2phase = conjugate.volts_to_dmh(dm2gainmap, dV2, lam)

# dpad = 2
# flipx = False
# N = conjugate.ceil_even(1.4*nBeam)
# usablePixelMapCrop = insertinto.insertinto(usablePixelMap, (N, N))

# DM1SurfActual = dmhtoph.dmhtoph(dm1phase, dm1r['nact'], fits.getdata(dm1r['inffn']), dm1r['ppact_d'], dm1r['ppact_cx'], dm1r['ppact_cy'], dm1r['dx'], dm1r['dy'], dm1r['thact'], flipx=flipx, dpad=dpad)
# DM1SurfActual = insertinto.insertinto(DM1SurfActual, (N, N))

# DM2SurfActual = dmhtoph.dmhtoph(dm2phase, dm2r['nact'], fits.getdata(dm2r['inffn']), dm2r['ppact_d'], dm2r['ppact_cx'], dm2r['ppact_cy'], dm2r['dx'], dm2r['dy'], dm2r['thact'], flipx=flipx, dpad=dpad)
# DM2SurfActual = insertinto.insertinto(DM2SurfActual, (N, N))

# if flagPlot:
#     plt.figure(11); plt.imshow(wfeMeas); plt.colorbar(); plt.title('WFE'); plt.pause(0.1);
#     plt.figure(12); plt.imshow(usablePixelMap.astype(int)); plt.colorbar(); plt.title('Usable Pixel Map'); plt.pause(0.1);
    
#     plt.figure(21); plt.imshow(surfDesDM1); plt.colorbar(); plt.title('Desired DM1 Surface'); plt.pause(0.1);
#     plt.figure(22); plt.imshow(DM1SurfActual); plt.colorbar(); plt.title('Best Fit DM1 Surface'); plt.pause(0.1);
#     plt.figure(23); plt.imshow((surfDesDM1 - DM1SurfActual)*usablePixelMapCrop); plt.colorbar(); plt.title('Difference DM1'); plt.pause(0.1);

#     plt.figure(31); plt.imshow(surfDesDM2); plt.colorbar(); plt.title('Desired DM2 Surface'); plt.pause(0.1);
#     plt.figure(32); plt.imshow(DM2SurfActual); plt.colorbar(); plt.title('Best Fit DM2 Surface'); plt.pause(0.1);
#     plt.figure(33); plt.imshow((surfDesDM2 - DM2SurfActual)*usablePixelMapCrop); plt.colorbar(); plt.title('Difference DM2'); plt.pause(0.1);

# print('Flattening residual for DM1:\tP-V = %.3e \tRMS = %.3e' % (np.max((surfDesDM1 - DM1SurfActual)*usablePixelMapCrop)-np.min((surfDesDM1 - DM1SurfActual)*usablePixelMapCrop), conjugate.rms((surfDesDM1 - DM1SurfActual)*usablePixelMapCrop)) )
# print('Flattening residual for DM2:\tP-V = %.3e \tRMS = %.3e' % (np.max((surfDesDM2 - DM2SurfActual)*usablePixelMapCrop)-np.min((surfDesDM2 - DM2SurfActual)*usablePixelMapCrop), conjugate.rms((surfDesDM2 - DM2SurfActual)*usablePixelMapCrop)) )