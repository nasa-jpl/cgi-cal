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

from cal.dmreg.dmreg import calc_translation_clocking

from cal.ampthresh.ampthresh import ampthresh
from cal.util.dmhtoph import dmhtoph, volts_to_dmh
from cal.util.findopt import find_optimum_1d, find_optimum_2d
from cal.util.fresnelprop import fresnelprop
from cal.util.insertinto import insertinto as inin
from cal.util.loadyaml import loadyaml
from cal.maskgen.maskgen import rotate_shift_downsample_amplitude_mask
from cal.dmreg.old.unwrap import unwrap

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
yOffsetDM = 2.1+yOffsetPupil
rotDM = 1.5  # degrees
flipxDM = False

lam = 575e-9
ppact_d = 10

dzDM1ToDM2 = 1.000
nArray = 1024

imageShape = (477, 561)
nrow = imageShape[0]
ncol = imageShape[1]

fnBoolMap = os.path.join(localpath, 'dmreg', 'testdata',
                         ('ut_usablePixelMap_DM%d_2BadAct.fits') % whichDM)
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
dm_fn = os.path.join(localpath, 'dmreg', 'testdata', 'ut_dm_for_dmreg.yaml')
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
deltaV = np.zeros((nact, nact))

# WORKS VERY WELL--BEST SO FAR
# lateral lines
deltaV[int(nact/2. - 3), :] = 1
deltaV[int(nact/2. + 2), :] = 1
# vertical lines
deltaV[:, int(nact/2. - 3)] = 1
deltaV[:, int(nact/2. + 2)] = 1

# # NOT GOOD:
# # lateral lines
# deltaV[int(nact/2. - 4), :] = 1
# deltaV[int(nact/2. + 3), :] = 1
# # vertical lines
# deltaV[:, int(nact/2. - 4)] = 1
# deltaV[:, int(nact/2. + 3)] = 1

# deltaV[:, int(nact/2. - 4):int(nact/2.-2)] = 1
# deltaV[:, int(nact/2. + 2):int(nact/2.+4)] = 1

# deltaV[int(nact/2. - 4):int(nact/2.-2), :] = 1
# deltaV[int(nact/2. + 2):int(nact/2.+4), :] = 1
# deltaV[:, int(nact/2. - 4):int(nact/2.-2)] = 1
# deltaV[:, int(nact/2. + 2):int(nact/2.+4)] = 1

# deltaV[int(nact/2. - 2):int(nact/2.+2), :] = 1
# deltaV[:, int(nact/2. - 1):int(nact/2.+1)] = 1
# deltaV = deltaV + (np.eye(nact) + np.fliplr(np.eye(nact)))
deltaV = 5*deltaV

# # Plus sign
# nact = dmr['nact']
# x = np.arange(-(nact-1)/2, (nact+1)/2)/nact*2
# [X, Y] = np.meshgrid(x, x)
# deltaV = np.zeros((nact, nact))
# deltaV[int(nact/2. - 1):int(nact/2.+1), :] = 1
# deltaV[:, int(nact/2. - 1):int(nact/2.+1)] = 1
# deltaV = deltaV + 1.8*(np.eye(nact) + np.fliplr(np.eye(nact)))
# deltaV = 5*deltaV

fn_deltaV = os.path.join('dmreg', 'cgidata',
                         'delta_DM_V_for_trans_rot_calib.fits')
if flagWrite:
    hdu = fits.PrimaryHDU(deltaV)
    hdu.writeto(fn_deltaV, overwrite=True)

plt.figure(13); plt.imshow(deltaV - np.rot90(deltaV, 2)); plt.gca().invert_yaxis(); plt.pause(0.1)

# %
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
plt.figure(5); plt.imshow(dmSurfRad); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(0.1)
# plt.figure(6); plt.imshow(dmSurfRad+pupilAmp); plt.colorbar(); plt.gca().invert_yaxis(); plt.pause(5)
# %

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

fn_amp0 = os.path.join('dmreg', 'testdata', ('ut_transrot_dm%d_amp0.fits' % (whichDM)))
fn_amp1 = os.path.join('dmreg', 'testdata', ('ut_transrot_dm%d_amp1.fits' % (whichDM)))
fn_ph0 = os.path.join('dmreg', 'testdata', ('ut_transrot_dm%d_ph0.fits' % (whichDM)))
fn_ph1 = os.path.join('dmreg', 'testdata', ('ut_transrot_dm%d_ph1.fits' % (whichDM)))

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
#     [ph0atDM2, mask0] = unwrap(np.angle(E0atDM2), np.abs(E0atDM2))
#     [ph1atDM2, mask1] = unwrap(np.angle(E1atDM2), np.abs(E1atDM2))
#     # 3. Difference the phases.
#     phDiff = ph1atDM2 - ph0atDM2

phDiffMeas = phDiff * usablePixelMap
plt.figure(1); plt.imshow(phDiffMeas); plt.axis('off');  plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig('TRphDiffMeas.png'); plt.pause(0.1)


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
dpad = 2
flipx = False  # DM registration pattern is flipx agnostic
dmrad = volts_to_dmh(gainmap, deltaV, lam)

nIterBoth = 1 #2

offsetMax = 10.0
nOffset = 7#13
nIterOffset = 3
shrinkFac = 0.9#0.75 #1 #0.75

clockMaxDeg = 3.0
nClock = 7
nIterClock = 1#5
# shrinkFac = 0.75

xOffsetVec = np.linspace(-offsetMax, offsetMax, nOffset)
yOffsetVec = np.linspace(-offsetMax, offsetMax, nOffset)
costMat = np.zeros((nOffset, nOffset))

plt.imshow(phDiffMeas); plt.gca().invert_yaxis(); plt.colorbar(); plt.pause(0.1)
# plt.figure(1); plt.imshow(phDiffMeas); plt.axis('off');  plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig('DMRphDiffMeasTransRot.png', bbox_inches='tight', pad_inches = 0); plt.pause(0.1)

# %
xOffsetEst = xOffsetPupil  # initialize
yOffsetEst = yOffsetPupil
clockEst = 0  # initialize

for _iter_both in range(nIterBoth):

    for ii in range(nIterOffset):
        for iy, yOffsetTemp in enumerate(yOffsetVec):
            for ix, xOffsetTemp in enumerate(xOffsetVec):
                phDiffSim = dmhtoph(nrow, ncol, dmrad, dmr['nact'], inf_func, dmr['ppact_d'],
                                    dmr['ppact_cx'], dmr['ppact_cy'],
                                    xOffsetEst+xOffsetTemp, yOffsetEst+yOffsetTemp, clockEst, flipx)
                # phDiffSim = inin(phDiffSim, imageShape)
                costMat[iy, ix] = np.sum(usablePixelMap *
                                          np.abs(phDiffMeas - phDiffSim))
                # plt.figure(21); plt.imshow(phDiffMeas - phDiffSim); plt.gca().invert_yaxis(); plt.colorbar(); plt.pause(0.1)
                # if iy == 0 and ix == 0:
                #     plt.figure(2); plt.imshow(usablePixelMap*phDiffSim); plt.axis('off');  plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig('DMRphDiffSimTrans.png', bbox_inches='tight', pad_inches = 0); plt.pause(0.1)


        [xOffsetTemp, yOffsetTemp] = \
            find_optimum_2d(xOffsetVec, yOffsetVec, costMat,
                                            np.ones_like(costMat))
        xOffsetEst += xOffsetTemp
        yOffsetEst += yOffsetTemp
        # offsetMax = shrinkFac*offsetMax
        xOffsetVec = (shrinkFac**ii)*np.linspace(-offsetMax, offsetMax, nOffset)
        yOffsetVec = (shrinkFac**ii)*np.linspace(-offsetMax, offsetMax, nOffset)
        plt.figure(22); plt.imshow(costMat); plt.gca().invert_yaxis(); plt.colorbar(); plt.pause(0.1)

        print('True:  (xc, yc) = (%.3f, %.3f) pixels' % (xOffsetDM, yOffsetDM))
        print('Est.:  (xc, yc) = (%.3f, %.3f) pixels' % (xOffsetEst, yOffsetEst))
        print('Error: (xc, yc) = (%.3f, %.3f) pixels\n' % (xOffsetDM-xOffsetEst, yOffsetDM-yOffsetEst))


    # Fit Rotation
    clockVec = np.linspace(-clockMaxDeg, clockMaxDeg, nClock)
    costVec = np.zeros((nClock,))
    for ii in range(nIterClock):
        for ic, deltaClock in enumerate((shrinkFac**ii)*clockVec):
            phDiffSim = dmhtoph(nrow, ncol, dmrad, dmr['nact'], inf_func, dmr['ppact_d'],
                                dmr['ppact_cx'], dmr['ppact_cy'],
                                xOffsetEst, yOffsetEst,
                                deltaClock+clockEst, flipx)
            phDiffSim = inin(phDiffSim, imageShape)
            costVec[ic] = np.sum(usablePixelMap*np.abs(phDiffMeas - phDiffSim))
            # plt.imshow(phDiffSim); plt.gca().invert_yaxis(); plt.colorbar(); plt.pause(0.1)
            # plt.imshow(usablePixelMap*(phDiffMeas - phDiffSim)); plt.gca().invert_yaxis(); plt.colorbar(); plt.pause(0.1)
            # if ic == 0:
                # plt.figure(3); plt.imshow(usablePixelMap*phDiffSim); plt.axis('off');  plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig('DMRphDiffSimRot.png', bbox_inches='tight', pad_inches = 0); plt.pause(0.1)

        clockEst = clockEst + find_optimum_1d(clockVec, costVec)
        # clockVec = shrinkFac*clockVec
        # plt.figure(4); plt.plot(clockVec, costVec, '-bo', linewidth=5, markersize=15); plt.tight_layout(); plt.savefig('DMRrotFit.png', bbox_inches='tight', pad_inches = 0); plt.pause(0.1)

        print('True:  %.3f degrees' % (rotDM))
        print('Est.:  %.3f degrees' % (clockEst))
        print('Error: %.3f degrees = %.2f mrad\n' % (clockEst-rotDM, (clockEst-rotDM)*np.pi/180*1000))


# dm_fn = 'testdata/ut_dm_for_dmreg.yaml'
# fn_fit_params = 'testdata/ut_trans_rot_settings.yaml'
# xOffsetEst, yOffsetEst, clockEst = calc_translation_clocking(amp0, ph0, amp1, ph1, usablePixelMap, whichDM,
#                          deltaV, xOffsetPupil, yOffsetPupil, dm_fn,
#                          fn_fit_params, lam)

# print('True:  (xc, yc) = (%.3f, %.3f) pixels' % (xOffsetDM, yOffsetDM))
# print('Est.:  (xc, yc) = (%.3f, %.3f) pixels' % (xOffsetEst, yOffsetEst))
# print('Error: (xc, yc) = (%.3f, %.3f) pixels\n' % (xOffsetDM-xOffsetEst, yOffsetDM-yOffsetEst))

# print('True:  %.3f degrees' % (rotDM))
# print('Est.:  %.3f degrees' % (clockEst))
# print('Error: %.3f degrees = %.2f mrad\n' % (clockEst-rotDM, (clockEst-rotDM)*np.pi/180*1000))
