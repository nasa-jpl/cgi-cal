# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to test mask rotation, translation, and downsampling with MASKGEN.
"""
import os
import numpy as np
from math import isclose
# import yaml
from astropy.io import fits  # for reading in FITS file test data
import matplotlib.pyplot as plt

from cal.maskgen.maskgen import rotate_shift_downsample_amplitude_mask\
    as changemask
from cal.util.insertinto import insertinto as inin

localpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# %% Check of Pupil Masks

fnMask = os.path.join(localpath, 'maskdesigns', 'HLC_20190210b_NFOV',
                      'LS_HLC_20190210b_NFOV_1000.fits')
mag = 386/1000.

# fnMask = os.path.join(localpath, 'maskdesigns', 'SPC_20200610_WFOV',
#                       'SPM_SPC_20200610_WFOV_1000us3x_binary_uint8.fits')
# mag = 386/3000.

maskIn = fits.getdata(fnMask)
rotDeg = 10  # degrees CCW

xOffset = 52
yOffset = -25
padFac = 1.2

maskOut = changemask(maskIn, rotDeg, mag, xOffset, yOffset, padFac)

plt.figure(1); plt.imshow(maskIn); plt.gca().invert_yaxis(); plt.colorbar(); plt.title('Mask In'); plt.pause(0.1);
plt.figure(2); plt.imshow(maskOut); plt.gca().invert_yaxis(); plt.colorbar(); plt.title('Mask Out'); plt.pause(4);


# %% Check of Focal Plane Masks

fnMask = os.path.join(localpath, 'maskdesigns', 'SPC_20200617_Spec',
                      'FPM_SPC_20200617_Spec_25pixPerLamD.fits')
mag = 4/25.

# fnMask = os.path.join(localpath, 'maskdesigns', 'SPC_20200610_WFOV',
#                       'SPM_SPC_20200610_WFOV_1000us3x_binary_uint8.fits')
# mag = 386/3000.

maskIn = fits.getdata(fnMask)
rotDeg = 90  # degrees CCW

xOffset = -15
yOffset = 50
padFac = 1.1

maskOut = changemask(maskIn, rotDeg, mag, xOffset, yOffset, padFac)
maskJustDemag = changemask(maskIn, 0, mag, 0, 0, padFac)
maskJustDemag = inin(maskJustDemag, (maskOut.shape))
maskRotRoll = np.roll(np.rot90(maskJustDemag[1::, 1::], 1), (yOffset, xOffset), axis=(0, 1))
maskRotRoll = inin(maskRotRoll, (maskOut.shape))
diff = maskOut - maskRotRoll

plt.figure(3); plt.imshow(maskIn); plt.gca().invert_yaxis(); plt.colorbar(); plt.title('Mask In'); plt.pause(0.1);
plt.figure(4); plt.imshow(maskOut); plt.gca().invert_yaxis(); plt.colorbar(); plt.title('Mask Out'); plt.pause(0.1);
plt.figure(5); plt.imshow(maskRotRoll); plt.gca().invert_yaxis(); plt.colorbar(); plt.title('Mask for Checking'); plt.pause(0.1);
plt.figure(6); plt.imshow(diff); plt.gca().invert_yaxis(); plt.colorbar(); plt.title('Diff'); plt.pause(0.1);
plt.pause(4);

# hdu = fits.PrimaryHDU(maskOut)
# hdu.writeto('/Users/ajriggs/Downloads/test_fpm.fits', overwrite=True)

# Test for correct downsampling by comparing the area before and after
ratioArea = np.sum(maskOut)/np.sum(maskIn)
isclose(mag**2, ratioArea, rel_tol=0.001)

# Test rotation and translation using np.rot90 and np.roll for comparison
np.sum(np.abs(diff)) < 1e-8*np.max(maskOut)