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

from cal.ampthresh.ampthresh import ampthresh
from cal.util.dmhtoph import dmhtoph, volts_to_dmh
from cal.util.findopt import find_optimum_2d
from cal.util.fresnelprop import fresnelprop
from cal.util.insertinto import insertinto as inin
from cal.util.loadyaml import loadyaml
from cal.dmreg.dmreg import calc_scale
from cal.dmreg.old.unwrap import unwrap
from cal.maskgen.maskgen import rotate_shift_downsample_amplitude_mask

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

whichDM = 1 #2

ppact_cx = 6.83  # 1.05*ppact_cx_nom
ppact_cy = 6.54  # 1.02*ppact_cy_nom
xOffsetDM = -3.8 + xOffsetPupil
yOffsetDM = 2.1+yOffsetPupil
rotDM = 1.5  # degrees
flipxDM = False

lam = 575e-9
ppact_d = 10

imageShape = (477, 561)
nrow = imageShape[0]
ncol = imageShape[1]
# imageShape = (500, 500)

fnBoolMap = os.path.join(localpath, 'dmreg', 'testdata',
                         ('ut_usablePixelMap_DM%d_2BadAct.fits' % whichDM))
usablePixelMap = fits.getdata(fnBoolMap).astype(bool)

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

nact = dmr['nact']
x = np.arange(-(nact-1)/2, (nact+1)/2)/nact*2
[X, Y] = np.meshgrid(x, x)
n0 = 26
deltaV = -np.ones((n0, n0))
deltaV = 1 + inin(deltaV, (n0+4, n0+4))
deltaV = inin(deltaV, (nact, nact))

# deltaV = np.zeros((nact, nact))
# deltaV[int(nact/2. - 1):int(nact/2.), :] = 1
# deltaV[:, int(nact/2. - 1):int(nact/2.)] = 1
# deltaV = deltaV - np.mean(deltaV)
deltaV = 5*deltaV

fn_deltaV = os.path.join(localpath, 'dmreg', 'cgidata',
                         'delta_DM_V_for_scale_calib.fits')
if flagWrite:
    hdu = fits.PrimaryHDU(deltaV)
    hdu.writeto(fn_deltaV, overwrite=True)

# %%


# THETA = np.arctan2(Y, X)

# deltaV = np.ones((nact, nact))

dmrad = volts_to_dmh(gainmap, deltaV, lam)

dmSurfRad0 = dmhtoph(nrow, ncol, dmrad, nact, inf_func, ppact_d, ppact_cx,
                    ppact_cy, xOffsetDM, yOffsetDM, rotDM, flipxDM)
# print(dmSurfRad0.shape)
# dmSurfRad0 = inin(dmSurfRad0, imageShape)
dmSurfRad = dmSurfRad0 * ampthresh(pupilAmp)

# plt.figure(13); plt.imshow(dmSurfRad0); plt.gca().invert_yaxis(); plt.pause(0.1)
# plt.figure(13); plt.imshow(dmSurfRad0.T); plt.gca().invert_yaxis(); plt.pause(0.1)
# plt.figure(14); plt.imshow(dmSurfRad0 - dmSurfRad0.T); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(0.1)
# plt.figure(15); plt.imshow(dmSurfRad0 - np.rot90(dmSurfRad0, 2)); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(0.1)
# plt.figure(16); plt.imshow(dmSurfRad0 - np.fliplr(dmSurfRad0)); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(0.1)
# plt.figure(17); plt.imshow(dmSurfRad0 - np.flipud(dmSurfRad0)); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(0.1)


# %
# plt.figure(5); plt.imshow(dmSurfRad); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(0.1)
# plt.figure(6); plt.imshow(dmSurfRad+pupilAmp); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(5)
# %
# amp0 = pupilAmp
# amp1 = pupilAmp
# ph0 = np.zeros(imageShape)
# ph1 = dmSurfRad
dzDM1ToDM2 = 1.000
nArray = 1024

amp0 = pupilAmp
amp0[amp0 < 0] = 0
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

fn_amp0 = os.path.join('dmreg', 'testdata', ('ut_scale_dm%d_amp0.fits' % (whichDM)))
fn_amp1 = os.path.join('dmreg', 'testdata', ('ut_scale_dm%d_amp1.fits' % (whichDM)))
fn_ph0 = os.path.join('dmreg', 'testdata', ('ut_scale_dm%d_ph0.fits' % (whichDM)))
fn_ph1 = os.path.join('dmreg', 'testdata', ('ut_scale_dm%d_ph1.fits' % (whichDM)))

if flagWrite:
    hdu = fits.PrimaryHDU(amp0)
    hdu.writeto(fn_amp0, overwrite=True)
    hdu = fits.PrimaryHDU(amp1)
    hdu.writeto(fn_amp1, overwrite=True)
    hdu = fits.PrimaryHDU(ph0)
    hdu.writeto(fn_ph0, overwrite=True)
    hdu = fits.PrimaryHDU(ph1)
    hdu.writeto(fn_ph1, overwrite=True)

# %
# xOffsetEst, yOffsetEst = dm_register_translation(amp0, ph0, amp1, ph1, whichDM,
#                              searchRadiusPix, nSteps, lam, dzDM1ToDM2, nArray,
#                              pixpermeter)

phDiff = ph1 - ph0
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

phDiffMeas = phDiff * usablePixelMap

plt.imshow(phDiffMeas); plt.gca().invert_yaxis(); plt.colorbar(); plt.pause(0.1)
# plt.figure(1); plt.imshow(phDiffMeas); plt.axis('off');  plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig('ScalePhDiffMeas.png'); plt.pause(0.1)

# Load DM Calibration Data
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
flipx = False  # DM registration pattern is flipx agnostic
dmrad = volts_to_dmh(gainmap, deltaV, lam)

# %%
nIter = 5
nScale = 7
maxDeltaScaleFac = 0.10 #0.05
shrinkFac = 0.75

xScaleVec = np.linspace(1-maxDeltaScaleFac, 1+maxDeltaScaleFac, nScale)
yScaleVec = np.linspace(1-maxDeltaScaleFac, 1+maxDeltaScaleFac, nScale)
costMat = np.zeros((nScale, nScale))
xScale = 1  # initialize
yScale = 1  # initialize
counter = 0
for _iter in range(nIter):
    for iy, yScaleTemp in enumerate(yScaleVec):
        for ix, xScaleTemp in enumerate(xScaleVec):
            phDiffSim = dmhtoph(nrow, ncol, dmrad, dmr['nact'], inf_func,
                                dmr['ppact_d'],
                                xScaleTemp*xScale*dmr['ppact_cx'],
                                yScaleTemp*yScale*dmr['ppact_cy'],
                                xOffsetDM, yOffsetDM,
                                rotDM, flipx)
            # phDiffSim = inin(phDiffSim, imageShape)
            costMat[iy, ix] = np.sum(usablePixelMap *
                                      np.abs(phDiffMeas - phDiffSim))
            # plt.figure(1); plt.imshow(phDiffSim*usablePixelMap); plt.axis('off');  plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig(('ScaleRef%d.png' % counter)); plt.pause(1e-2);
            # counter += 1

    # plt.figure(3); plt.imshow(costMat); plt.axis('off');  plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig(('ScaleCostMat%d.png' % _iter)); plt.pause(1e-2);

    [xScaleTemp, yScaleTemp] = \
        find_optimum_2d(xScaleVec, yScaleVec, costMat,
                                        np.ones_like(costMat))
    xScale = xScaleTemp*xScale
    yScale = yScaleTemp*yScale
    maxDeltaScaleFac = shrinkFac*maxDeltaScaleFac
    xScaleVec = np.linspace(1-maxDeltaScaleFac, 1+maxDeltaScaleFac, nScale)
    yScaleVec = np.linspace(1-maxDeltaScaleFac, 1+maxDeltaScaleFac, nScale)
    # xScaleVec = shrinkFac*xScaleVec
    # yScaleVec = shrinkFac*yScaleVec
    plt.figure(); plt.imshow(costMat); plt.gca().invert_yaxis(); plt.colorbar(); plt.pause(0.1)

    ppact_cx_est = xScale*dmr['ppact_cx']
    ppact_cy_est = yScale*dmr['ppact_cy']


    print('True:  (%.3f, %.3f)' % (ppact_cx, ppact_cy))
    print('Est.:  (%.3f, %.3f)' % (ppact_cx_est, ppact_cy_est))
    print('Error: (%.3f, %.3f)' % (ppact_cx-ppact_cx_est, ppact_cy-ppact_cy_est))
    print('Error: (%.3f%%, %.3f%%)\n' % (100*(ppact_cx-ppact_cx_est)/ppact_cx, 100*(ppact_cy-ppact_cy_est)/ppact_cy))

# %%
# dm_fn = 'testdata/ut_dm_for_dmreg.yaml'
# fn_fit_params = 'testdata/ut_scale_settings.yaml'
# ppact_cx_est, ppact_cy_est = \
#     calc_scale(amp0, ph0, amp1, ph1, usablePixelMap, whichDM, deltaV,
#                           dm_fn, fn_fit_params, lam,
#                           xOffsetDM, yOffsetDM, rotDM)

# print('True:  (%.3f, %.3f)' % (ppact_cx, ppact_cy))
# print('Est.:  (%.3f, %.3f)' % (ppact_cx_est, ppact_cy_est))
# print('Error: (%.3f, %.3f)' % (ppact_cx-ppact_cx_est, ppact_cy-ppact_cy_est))
# print('Error: (%.3f%%, %.3f%%)\n' % (100*(ppact_cx-ppact_cx_est)/ppact_cx, 100*(ppact_cy-ppact_cy_est)/ppact_cy))

# %% Test the exact code in the unit test
from math import isclose

localpath = (os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

# localpath = os.path.dirname(os.path.abspath(__file__))

whichDM = 1
fn_amp0 = os.path.join(localpath, 'testdata', 'ut_scale_dm1_amp0.fits')
fn_amp1 = os.path.join(localpath, 'testdata', 'ut_scale_dm1_amp1.fits')
fn_ph0 = os.path.join(localpath, 'testdata', 'ut_scale_dm1_ph0.fits')
fn_ph1 = os.path.join(localpath, 'testdata', 'ut_scale_dm1_ph1.fits')
amp0 = fits.getdata(fn_amp0)
amp1 = fits.getdata(fn_amp1)
ph0 = fits.getdata(fn_ph0)
ph1 = fits.getdata(fn_ph1)
fnBoolMap = os.path.join(localpath, 'testdata',
                         'ut_usablePixelMap_DM1_2BadAct.fits')
usablePixelMap = fits.getdata(fnBoolMap).astype(bool)

nact = 48
# x = np.arange(-(nact-1)/2, (nact+1)/2)/nact*2
# [X, Y] = np.meshgrid(x, x)
# deltaV = np.zeros((nact, nact))
# deltaV[int(nact/2. - 1):int(nact/2.), :] = 1
# deltaV[:, int(nact/2. - 1):int(nact/2.)] = 1
# deltaV = 5*deltaV

x = np.arange(-(nact-1)/2, (nact+1)/2)/nact*2
[X, Y] = np.meshgrid(x, x)
n0 = 26
deltaV = -np.ones((n0, n0))
deltaV = 1 + inin(deltaV, (n0+4, n0+4))
deltaV = 5*inin(deltaV, (nact, nact))

# plt.imshow(deltaV); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(1e-2)
# plt.imshow(deltaV); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(1e-2)

dm_fn = 'testdata/ut_dm_for_dmreg.yaml'
fn_fit_params = 'testdata/ut_scale_settings.yaml'

lam = 575e-9
xOffset = 86.2
yOffset = -30.9
clocking = 1.5

ppact_cx_est, ppact_cy_est = \
    calc_scale(amp0, ph0, amp1, ph1, usablePixelMap, whichDM, deltaV,
               dm_fn, fn_fit_params, lam, xOffset, yOffset, clocking)
ppact_cx_true = 6.83
ppact_cy_true = 6.54
print('ppact_cx_est, ppact_cy_est = %.3f, %.3f' % (ppact_cx_est, ppact_cy_est))
tolVal = 4e-3
print(isclose(ppact_cx_true, ppact_cx_est, rel_tol=tolVal))
print(isclose(ppact_cy_true, ppact_cy_est, rel_tol=tolVal))
