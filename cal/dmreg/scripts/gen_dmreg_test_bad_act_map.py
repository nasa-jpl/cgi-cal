# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
import os
import numpy as np
from astropy.io import fits
from scipy.ndimage import rotate
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from numpy.fft import fftshift, ifftshift
from skimage.morphology import binary_opening

from cal.dmreg.dmreg import calc_software_mask
from cal.util.remove_ptt import remove_ptt_directly

from cal.ampthresh.ampthresh import ampthresh
from cal.util.dmhtoph import dmhtoph
from cal.util.fresnelprop import fresnelprop
from cal.conjugate.conjugate import volts_to_dmh
from cal.maskgen.maskgen import rotate_shift_downsample_amplitude_mask
from cal.util.loadyaml import loadyaml
from cal.dmreg.old.unwrap import unwrap
from cal.util.insertinto import insertinto as inin

localpath = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

flagWrite = False  # whether to write unit test data

diamBeamPix = 300.0
diamBeamM = 46.34e-3  # meters
pitch = 0.9906e-3  # meters per actuator
aoi = 9.65  # degrees
ppact_cy_nom = pitch / (diamBeamM/diamBeamPix)  # (m/act)/(m/pix) = pix/act
ppact_cx_nom = ppact_cy_nom / np.cos(aoi*np.pi/180.)
pixpermeter = diamBeamPix / diamBeamM

xOffsetPupil = 90
yOffsetPupil = -33
rotPupilDeg = 0
magPupilIO = diamBeamPix/1000.
padFacPupil = 1.2

whichDM = 1
ppact_cx = 6.83  # 1.05*ppact_cx_nom
ppact_cy = 6.54  # 1.02*ppact_cy_nom
xOffsetDM = -3.8 + xOffsetPupil
yOffsetDM = 2.1 + yOffsetPupil
rotDM = 1.5  # degrees
flipxDM = False

lam = 575e-9
ppact_d = 10

deltaPhaseThreshRad = 5e-9 / lam * 2 * np.pi
nErode = 2

imageShape = (477, 561)
nrow = imageShape[0]
ncol = imageShape[1]
# imageShape = (500, 500)


# Generate the pupil amplitude
fnPupil = os.path.join(localpath, 'pupilfit', 'cgidata',
                       'pupil_template_D1000.00pixels.fits')
pupilRef = fits.getdata(fnPupil)
pupilAmp = rotate_shift_downsample_amplitude_mask(pupilRef, rotPupilDeg,
                                                  magPupilIO, xOffsetPupil,
                                                  yOffsetPupil, padFacPupil)
pupilAmp = inin(pupilAmp, imageShape)
pupilAmp[pupilAmp < 0] = 0

# imageShape = pupilAmp.shape
# plt.figure(4); plt.imshow(pupilAmp); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(1e-2)


# Generate the perturbed DM surface
# fninf = os.path.join(localpath, 'dmreg', 'testdata',
#                           'ut_influence_dm5v2_inffix.fits')
# inf_func = fits.getdata(fninf)


# Load DM Calibration Data
dm_fn = os.path.join(localpath, 'dmreg', 'testdata', 'ut_dm.yaml')
dmConfig = loadyaml(dm_fn)
if whichDM == 1:
    dm = dmConfig["dms"]["DM1"]
elif whichDM == 2:
    dm = dmConfig["dms"]["DM2"]
dmr = dm['registration']
gainfn = os.path.join(localpath, 'dmreg', dm['voltages']['gainfn'])
gainmap = fits.getdata(gainfn)
inffn = os.path.join(localpath, 'dmreg', dmr['inffn'])
inf_func = fits.getdata(inffn)
dpad = 2

nact = dmr['nact']
x = np.arange(-(nact-1)/2, (nact+1)/2)/nact*2
[X, Y] = np.meshgrid(x, x)
dV = X*X + Y*Y
dV = dV - np.mean(dV)
dV = 5*dV
# THETA = np.arctan2(Y, X)

dV = 5*np.ones((nact, nact))
dV[9, 10] = 0
dV[20, 35] = 0

dmrad = volts_to_dmh(gainmap, dV, lam)

dmSurfRad0 = dmhtoph(nrow, ncol, dmrad, nact, inf_func, ppact_d, ppact_cx,
                    ppact_cy,xOffsetDM, yOffsetDM, rotDM, flipxDM)
dmSurfRad0 = inin(dmSurfRad0, imageShape)
dmSurfRad = dmSurfRad0 * ampthresh(pupilAmp)

# plt.figure(13); plt.imshow(dmSurfRad0); plt.gca().invert_yaxis(); plt.pause(0.1)
# plt.figure(13); plt.imshow(dmSurfRad0.T); plt.gca().invert_yaxis(); plt.pause(0.1)
# plt.figure(14); plt.imshow(dmSurfRad0 - dmSurfRad0.T); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(0.1)
# plt.figure(15); plt.imshow(dmSurfRad0 - np.rot90(dmSurfRad0, 2)); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(0.1)
# plt.figure(16); plt.imshow(dmSurfRad0 - np.fliplr(dmSurfRad0)); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(0.1)
# plt.figure(17); plt.imshow(dmSurfRad0 - np.flipud(dmSurfRad0)); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(0.1)


# %
plt.figure(5); plt.imshow(dmSurfRad); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(0.1)
# plt.figure(6); plt.imshow(dmSurfRad+pupilAmp); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(5)
# %
# amp0 = pupilAmp
# amp1 = pupilAmp
# ph0 = np.zeros(imageShape)
# ph1 = dmSurfRad

dzDM1ToDM2 = 1.000
nArray = 1024

amp0 = pupilAmp
ph0 = np.zeros(imageShape)
if whichDM == 1:
    amp1 = pupilAmp
    ph1 = dmSurfRad
elif whichDM == 2:
    Edm2inc = fresnelprop(pupilAmp, lam, dzDM1ToDM2, nArray, pixpermeter)
    Edm2 = Edm2inc * np.exp(1j * dmSurfRad)
    EbackAtPupil0 = fresnelprop(Edm2inc, lam, -dzDM1ToDM2, nArray, pixpermeter)
    EbackAtPupil1 = fresnelprop(Edm2, lam, -dzDM1ToDM2, nArray, pixpermeter)
    amp1 = np.abs(EbackAtPupil1)
    [ph1, mask1] = unwrap(np.angle(EbackAtPupil1), amp0)
    [ph0, mask0] = unwrap(np.angle(EbackAtPupil0), amp0)
    # ph1 = unwrap(np.angle(pupilAmp))
    plt.imshow(amp1); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(1e-2)
    plt.imshow(ph1); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(1e-2)
    plt.imshow(ph0); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(1e-2)
    plt.imshow(ph1 - ph0); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(1e-2)

# fn_amp0 = os.path.join('dmreg', 'testdata', ('ut_sw_mask_dm%d_amp0.fits' % (whichDM)))
# fn_amp1 = os.path.join('dmreg', 'testdata', ('ut_sw_mask_dm%d_amp1.fits' % (whichDM)))
# fn_ph0 = os.path.join('dmreg', 'testdata', ('ut_sw_mask_dm%d_ph0.fits' % (whichDM)))
# fn_ph1 = os.path.join('dmreg', 'testdata', ('ut_sw_mask_dm%d_ph1.fits' % (whichDM)))

# if flagWrite:
#     hdu = fits.PrimaryHDU(amp0)
#     hdu.writeto(fn_amp0, overwrite=True)
#     hdu = fits.PrimaryHDU(amp1)
#     hdu.writeto(fn_amp1, overwrite=True)
#     hdu = fits.PrimaryHDU(ph0)
#     hdu.writeto(fn_ph0, overwrite=True)
#     hdu = fits.PrimaryHDU(ph1)
#     hdu.writeto(fn_ph1, overwrite=True)


# xOffsetEst, yOffsetEst = dm_register_translation(amp0, ph0, amp1, ph1, whichDM,
#                              searchRadiusPix, nSteps, lam, dzDM1ToDM2, nArray,
#                              pixpermeter)

phDiff = ph1 - ph0

# plt.figure(1); plt.imshow(ph0); plt.axis('off'); plt.clim(0, 0.22); plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig('ph0.png'); plt.pause(0.1)
# plt.figure(2); plt.imshow(ph1); plt.axis('off'); plt.clim(0, 0.22); plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig('ph1.png'); plt.pause(0.1)

# plt.figure(2); plt.imshow(ph1); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(0.1)

# if whichDM == 1:
#     phDiff = ph1 - ph0
# elif whichDM == 2:
#     # 1. Propagate the E-fields from the pupil to DM2.
#     E0 = amp0*np.exp(1j*ph0)
#     E1 = amp1*np.exp(1j*ph1)
#     E0atDM2 = fresnelprop(E0, lam, dzDM1ToDM2, nArray, pixpermeter)
#     E1atDM2 = fresnelprop(E1, lam, dzDM1ToDM2, nArray, pixpermeter)
#     # 2. Take their phases and unwrap.
#     [ph0atDM2, mask0] = unwrap(np.angle(E0atDM2), np.abs(E0))
#     [ph1atDM2, mask1] = unwrap(np.angle(E1atDM2), np.abs(E0))
#     # 3. Difference the phases.
#     phDiff = ph1atDM2 - ph0atDM2

usablePixelMap = ampthresh(pupilAmp)
phDiff = remove_ptt_directly(phDiff, usablePixelMap)
# plt.figure(3); plt.imshow(phDiff); plt.axis('off');  plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig('phDiffNoPTT.png'); plt.pause(0.1)


# phDiff = phDiff * usablePixelMap
# phDiff = phDiff - np.median(phDiff[usablePixelMap])
usablePixelMap[np.abs(phDiff) > deltaPhaseThreshRad] = False
usablePixelMap0 = usablePixelMap
# plt.figure(4); plt.imshow(usablePixelMap0); plt.axis('off');  plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig('phDiffThresh.png'); plt.pause(0.1)

# Erode the pupil aperture
paddedShape = (imageShape[0]+nErode, imageShape[1]+nErode)
maskPad0 = inin(usablePixelMap, paddedShape).astype(int)
# maskPad0 = inin(ampthresh(pupilAmp), paddedShape).astype(int)
maskPad = np.zeros_like(maskPad0)
for iy in np.arange(-nErode, nErode+1):
    for ix in np.arange(-nErode, nErode+1):
        maskPad += np.roll(maskPad0, (iy, ix), axis=(0, 1))
        pass
maskPad = np.floor(maskPad/(2*nErode+1)**2)
mask = inin(maskPad, imageShape)
usablePixelMap = mask.astype(bool)
# plt.figure(5); plt.imshow(usablePixelMap); plt.axis('off');  plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig('usablePixelMap.png'); plt.pause(0.1)

plt.figure(11); plt.imshow(usablePixelMap); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(0.1)
# plt.figure(12); plt.imshow(usablePixelMap + ampthresh(pupilAmp).astype(float)); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(0.1)


# maskBO = binary_opening(usablePixelMap0, np.ones((1+2*nErode, 1+2*nErode)))
# plt.figure(12); plt.imshow(maskBO); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(0.1)

from scipy.ndimage.morphology import binary_erosion

maskBE = binary_erosion(usablePixelMap0, structure=np.ones((1+2*nErode, 1+2*nErode)))
plt.figure(12); plt.imshow(maskBE); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(0.1)


# %

# phDiff = phDiff * usablePixelMap
# phDiff = phDiff - np.median(phDiff[usablePixelMap])
# usablePixelMap[(phDiff > deltaPhaseThreshRad) | (phDiff < -deltaPhaseThreshRad)] = False

plt.figure(11); plt.imshow(phDiff); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(0.1)
plt.figure(12); plt.imshow(usablePixelMap); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(0.1)


# %

arrayOut = calc_software_mask(amp0, ph0, amp1, ph1, whichDM, nErode, deltaPhaseThreshRad)

plt.figure(13); plt.imshow(arrayOut); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(0.1)

# Print out test data
localpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(localpath)
fn = os.path.join(localpath, 'testdata', ('ut_usablePixelMap_DM%d_2BadAct.fits' % (whichDM)))
hdu = fits.PrimaryHDU(arrayOut.astype(int))
if flagWrite:
    hdu.writeto(fn, overwrite=True)

# %% Code used in unit testing

deltaPhaseThresh = 0.1

nrow = 21
ncol = 30
amp0 = inin(np.ones((nrow-2, ncol-2)), (nrow, ncol))
ph0 = np.zeros((nrow, ncol))
amp1 = inin(np.ones((nrow-2, ncol-2)), (nrow, ncol))
ph1 = np.ones((nrow, ncol))
whichDM = 1
nErode = 1
deltaPhaseThreshRad = 0.1
boolMask = calc_software_mask(amp0, ph0, amp1, ph1, whichDM, nErode,
                              deltaPhaseThreshRad)
answer = inin(np.ones((nrow-4, ncol-4)), (nrow, ncol)).astype(int)
# self.assertTrue(np.sum(np.abs(boolMask.astype(int) - answer)) == 0)

# ph1 = inin(np.ones((nrow-3, ncol-3)), (nrow, ncol))
ph1 = np.ones((nrow, ncol))
ph1[9:11, 8] = 10
# answer = inin(np.ones((nrow-5, ncol-5)), (nrow, ncol)).astype(int)
# self.assertTrue(np.sum(np.abs(boolMask.astype(int) - answer)) == 0)


phDiff = ph1 - ph0

# Remove piston, tip, and tilt
usablePixelMap = ampthresh(amp0)
phDiff = remove_ptt_directly(phDiff, usablePixelMap)

# Threshold to get rid of large phase deviations near the pupil obscuration
# edges or at dead DM actuators.
usablePixelMap[(phDiff > deltaPhaseThresh) |
               (phDiff < -deltaPhaseThresh)] = False

imageShape = amp0.shape
paddedShape = (imageShape[0]+nErode, imageShape[1]+nErode)
maskPad0 = inin(usablePixelMap, paddedShape).astype(int)
maskPad = np.zeros_like(maskPad0)
for iy in np.arange(-nErode, nErode+1):
    for ix in np.arange(-nErode, nErode+1):
        maskPad += np.roll(maskPad0, (iy, ix), axis=(0, 1))
        pass
maskPad = np.floor(maskPad/(2*nErode+1)**2)
mask = inin(maskPad, imageShape)
usablePixelMap = mask.astype(bool)

plt.figure(2); plt.imshow(phDiff); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(0.1)
plt.figure(22); plt.imshow(answer); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(0.1)
plt.figure(23); plt.imshow(usablePixelMap); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(0.1)
