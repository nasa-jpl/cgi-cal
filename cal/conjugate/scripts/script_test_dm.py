# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Created on Wed Jan 29 13:01:59 2020

@author: ajriggs

Tests:
    1. Check that energy is conserved
"""
import os
import numpy as np
import yaml
from astropy.io import fits  # for reading in FITS file test data
import matplotlib.pyplot as plt
import copy

import util.nollzernikes as nollzernikes
import util.dmhtoph as dmhtoph
from util.insertinto import insertinto as inin
import conjugate.conjugate as conjugate

localpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(localpath)
# dVROI = np.ones((48, 48))
dVROI = inin(np.ones((46, 46)), (48, 48))

lam = 575e-9
dm_fn = os.path.join(localpath, 'testdata', 'dm_offset.yaml')
# dm_fn = 'testdata/dm.yaml'

# %%

# Load DM Calibration Data
try:
    dmConfig = yaml.safe_load(open(dm_fn, "r"))
except:
    raise Exception('Error loading yaml file with DM data.')
dm = dmConfig["dms"]
dm1r = dm[0]['registration']


dm1r = dm[0]['registration']
dm1gainfn = os.path.join(localpath, dm[0]['voltages']['gainfn'])
dm1gainmap = fits.getdata(dm1gainfn)
dm1phase = conjugate.volts_to_dmh(dm1gainmap, dVROI, lam)

flipx = False
n = 512
# N = conjugate.ceil_even(1.4*nBeam)
# usablePixelMapCrop = inin(usablePixelMap, (N, N))

# %% Generate a map of the region the DM can control
dm1inffn = os.path.join(localpath, dm1r['inffn'])
dm1TestSurf = dmhtoph.dmhtoph(n, n, dm1phase, dm1r['nact'], fits.getdata(dm1inffn), dm1r['ppact_d'], dm1r['ppact_cx'], dm1r['ppact_cy'], dm1r['dx'], dm1r['dy'], dm1r['thact'], flipx)
# DM1SurfActual = inin(DM1SurfActual, (N, N))
# usablePixelMap = DM1SurfActual[DM1SurfActual >= 0.5*np.max(DM1SurfActual)]
usablePixelMap = copy.copy(dm1TestSurf)
usablePixelMap[dm1TestSurf >= 0.7*np.max(dm1TestSurf)] = 1
usablePixelMap[dm1TestSurf < 0.7*np.max(dm1TestSurf)] = 0


plt.figure(1); plt.imshow(dm1TestSurf); plt.colorbar(); plt.title('dm1TestSurf'); plt.pause(0.1);
plt.figure(2); plt.imshow(usablePixelMap); plt.colorbar(); plt.title('Controllable Surface'); plt.pause(0.1);

# %% Create the error map
nBeam = np.min(dm1TestSurf.shape)
zernIndVec = np.arange(4, 31)
zernCoefVec = np.random.rand(zernIndVec.size)

x = (np.arange(dm1TestSurf.shape[1], dtype=np.float64)
     - dm1TestSurf.shape[1]//2)
y = (np.arange(dm1TestSurf.shape[0], dtype=np.float64)
     - dm1TestSurf.shape[0]//2)
xx, yy = np.meshgrid(x, y)

# Create expected map using nollzernikes.xyzern
zernCube = nollzernikes.xyzern(xx, yy, nBeam/2., zernIndVec)
zernAbMap = np.zeros(dm1TestSurf.shape, dtype=np.float64)
for ii in range(zernIndVec.size):
    zernAbMap += zernCoefVec[ii]*np.squeeze(zernCube[ii, :, :])

plt.figure(3); plt.imshow(zernAbMap*usablePixelMap); plt.colorbar(); plt.title('Map to Correct'); plt.pause(0.1);

# %% Compute delta DM settings
hdm1 = conjugate.conv_surf_to_dm_cmd(zernAbMap, dm[0], flipx=flipx)

# dm1gainfn = dm[0]['voltages']['gainfn']

dV = conjugate.dmh_to_volts(dm1gainmap, hdm1, lam)

dm1phase = conjugate.volts_to_dmh(dm1gainmap, dV, lam)

dm1Surf = dmhtoph.dmhtoph(n, n, dm1phase, dm1r['nact'], fits.getdata(dm1inffn), dm1r['ppact_d'], dm1r['ppact_cx'], dm1r['ppact_cy'], dm1r['dx'], dm1r['dy'], dm1r['thact'], flipx)

plt.figure(4); plt.imshow(dm1Surf*usablePixelMap); plt.colorbar(); plt.title('DM Fit to WFE Map'); plt.pause(0.1);

plt.figure(5); plt.imshow((zernAbMap - dm1Surf)*usablePixelMap); plt.colorbar(); plt.title('Residual to Fit'); plt.pause(5);

diff = (zernAbMap - dm1Surf)*usablePixelMap;

# N = int(np.ceil(dm1r['ppact_cx']*(dm1r['nact'] - 2)))
# diff = inin(diff, (N, N))
# diffRMS = conjugate.rms(diff)
diffRMS = conjugate.rms(diff[usablePixelMap.astype(bool)])
print('RMS surface fitting error = %.4f' % diffRMS)
