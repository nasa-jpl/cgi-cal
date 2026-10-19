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
import util.nollzernikes as nollzernikes


# %%
nBeam = 100
wfeShape = (120, 130)
dxPupil = 0.0
dyPupil = 0.0
# zernCoefVec = np.arange(1, 13)
zernCoefVec = np.arange(1, 13)
maxNollZern = 11

x = (np.arange(wfeShape[1], dtype=np.float64) - wfeShape[1]//2)
y = (np.arange(wfeShape[0], dtype=np.float64) - wfeShape[0]//2)
xx, yy = np.meshgrid(x, y)

# Create the true WFE allocations
zernIndVec = np.arange(1, 13)
zernCube = nollzernikes.xyzern(xx, yy, nBeam/2., zernIndVec)
wfeZ1to3True = np.zeros(wfeShape, dtype=np.float64)
wfeZ4True = np.zeros(wfeShape, dtype=np.float64)
wfeZ4or5to11True = np.zeros(wfeShape, dtype=np.float64)
for iz in (0, 1, 2):
    wfeZ1to3True += zernCoefVec[iz]*np.squeeze(zernCube[iz, :, :])
for iz in (3,):
    wfeZ4True += zernCoefVec[iz]*np.squeeze(zernCube[iz, :, :])
# for iz in (4, 5, 6, 7, 8, 9, 10):
for iz in (4, 5, 6, 7, 8, 9, 10):
    wfeZ4or5to11True += zernCoefVec[iz]*np.squeeze(zernCube[iz, :, :])

flagUseFCM = True
wfeZ1to3, wfeZ4, wfeZ4or5to11 = conjugate.gen_split_wfe(
    flagUseFCM, maxNollZern, zernCoefVec, nBeam, wfeShape, dxPupil, dyPupil)

plt.figure(1); plt.imshow(wfeZ4or5to11True); plt.colorbar(); plt.title('True'); plt.pause(0.1);
plt.figure(2); plt.imshow(wfeZ4or5to11); plt.colorbar(); plt.title('Output'); plt.pause(0.1);
plt.figure(3); plt.imshow(wfeZ4or5to11True-wfeZ4or5to11); plt.colorbar(); plt.title('Difference'); plt.pause(1);

# self.assertTrue(isclose(np.sum(np.abs(wfeZ1to3 - wfeZ1to3True)), 0.0,
#                         abs_tol=1e-4))
# self.assertTrue(isclose(np.sum(np.abs(wfeZ4 - wfeZ4True)), 0.0,
#                         abs_tol=1e-4))
# self.assertTrue(isclose(np.sum(np.abs(wfeZ4or5to11 -
#                                       wfeZ4or5to11True)), 0.0,
#                         abs_tol=1e-4))

flagUseFCM = False
wfeZ1to3, wfeZ4, wfeZ4or5to11 = conjugate.gen_split_wfe(
    flagUseFCM, maxNollZern, zernCoefVec, nBeam, wfeShape, dxPupil, dyPupil)

# wfeZ4True = np.zeros(wfeShape, dtype=np.float64)
wfeZ4or5to11True = np.zeros(wfeShape, dtype=np.float64)
for iz in (3, 4, 5, 6, 7, 8, 9, 10):#, 5, 6, 7, 8, 9, 10):
    wfeZ4or5to11True += zernCoefVec[iz]*np.squeeze(zernCube[iz, :, :])
    
# self.assertTrue(isclose(np.sum(np.abs(wfeZ4)), 0.0, abs_tol=1e-4))
# self.assertTrue(isclose(np.sum(np.abs(wfeZ4or5to11 -
#                                       wfeZ4or5to11True)), 0.0,
#                         abs_tol=1e-4))

plt.figure(4); plt.imshow(wfeZ4or5to11True); plt.colorbar(); plt.title('True'); plt.pause(0.1);
plt.figure(5); plt.imshow(wfeZ4or5to11); plt.colorbar(); plt.title('Output'); plt.pause(0.1);
plt.figure(6); plt.imshow(wfeZ4or5to11True-wfeZ4or5to11); plt.colorbar(); plt.title('Difference'); plt.pause(5);


# plt.figure(3); plt.imshow(wfeZ1to3True); plt.colorbar(); plt.title('True'); plt.pause(0.1);
# plt.figure(4); plt.imshow(wfeZ1to3); plt.colorbar(); plt.title('Output'); plt.pause(0.1);

# for iz in np.arange(13):
#     plt.figure(); plt.imshow(zernCube[iz, :, :]); plt.colorbar(); plt.pause(0.1);

