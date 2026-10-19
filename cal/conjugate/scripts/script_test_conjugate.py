# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Script to check CONJUGATE.
"""
# import sys
# sys.path.insert(0,"/Users/ajriggs/Repos/falco-python/")
import os
import numpy as np
import yaml

from astropy.io import fits  # for reading in FITS file test data
import matplotlib.pyplot as plt

from ampthresh.ampthresh import ampthresh
from conjugate.conjugate import conjugate, rms, gen_split_wfe, smooth_surface,\
                                conv_surf_to_dm_cmd, ceil_odd, ceil_even,\
                                phase_cost_function, volts_to_dmh, dmh_to_volts
from util.insertinto import insertinto as inin
from util.nollzernikes import xyzern
from util.dmhtoph import dmhtoph
from conjugate.util.propcustom import fit_zernikes, gen_zernikes

localpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

flagPlot = True

# %% Various Inputs

cmdDM1 = 50*np.ones((48, 48))
cmdDM2 = 50*np.ones((48, 48))
cmdFCM = 123
flagUseDM1 = True
flagUseDM2 = True
flagUseFCM = True
flagCorrectHighZern = True
lam = 575e-9  # used only for height-to-phase or vice-versa
amp_fn = os.path.join(localpath, 'testdata',
                      'amp_cgi_entrance_phase_b_N386_radians_575nm.fits')
ph_fn = os.path.join(localpath, 'testdata',
                     'ph_cgi_entrance_phase_b_N386_radians_575nm.fits')
pupil_diam_fn = os.path.join(localpath, 'testdata',
                             'meas_pupil_diam_clocking.yaml')
pupil_offsets_fn = os.path.join(localpath, 'testdata',
                                'meas_pupil_lateral_offsets.yaml')
dm_fn = os.path.join(localpath, 'testdata', 'dm.yaml')
fcm_fn = os.path.join(localpath, 'testdata', 'fcm.yaml')
conjugate_params_fn = os.path.join(localpath, 'testdata',
                                   'fixed_inputs_for_conjugate.yaml')

# %% Load DM Calibration Data to get "nact" and make total voltage maps
try:
    dmConfig = yaml.safe_load(open(dm_fn, "r"))
except:
    raise Exception('Error loading yaml file with DM data.')
dm = dmConfig["dms"]
dm1r = dm["DM1"]['registration']
dm2r = dm["DM2"]['registration']

Vdm1In = 50.*np.ones((dm1r['nact'], dm1r['nact']))
Vdm2In = 50.*np.ones((dm2r['nact'], dm2r['nact']))

# %% For generating surface maps

dpad = 2
flipx = False

# Load DM Calibration Data
try:
    dmConfig = yaml.safe_load(open(dm_fn, "r"))
except:
    raise Exception('Error loading yaml file.')
dm = dmConfig["dms"]
dm1r = dm["DM1"]['registration']
dm2r = dm["DM2"]['registration']
dm1gainmap_fn = os.path.join(localpath, dm["DM1"]['voltages']['gainfn'])
dm1gainmap = fits.getdata(dm1gainmap_fn)
dm2gainmap_fn = os.path.join(localpath, dm["DM2"]['voltages']['gainfn'])
dm2gainmap = fits.getdata(dm2gainmap_fn)


# Load pupil calibration data
try:
    pupilDiam = yaml.safe_load(open(pupil_diam_fn, "r"))
except:
    raise Exception('Error loading yaml file with pupil diameter data.')
diamPupil = pupilDiam["diamPupil"]

try:
    pupilOffsets = yaml.safe_load(open(pupil_offsets_fn, "r"))
except:
    raise Exception('Error loading yaml file with pupil offset data.')
xOffsetPupil = pupilOffsets["xOffsetPupil"]
yOffsetPupil = pupilOffsets["yOffsetPupil"]
  
# Load DM Calibration Data
try:
    dmConfig = yaml.safe_load(open(dm_fn, "r"))
except:
    raise Exception('Error loading yaml file with DM data.')
dm = dmConfig["dms"]
dm1r = dm["DM1"]['registration']
dm2r = dm["DM2"]['registration']
# Determine the region of the array corresponding to the DM surface
# for use in the fitting.
dpad = 2
nxres1 = round(dpad*dm1r['nact']*dm1r['ppact_cx'] + 0.5)
nyres1 = round(dpad*dm1r['nact']*dm1r['ppact_cy'] + 0.5)
wfeShape1 = (nyres1, nxres1)
nxres2 = round(dpad*dm2r['nact']*dm2r['ppact_cx'] + 0.5)
nyres2 = round(dpad*dm2r['nact']*dm2r['ppact_cy'] + 0.5)
wfeShape2 = (nyres2, nxres2)
nxresMax = int(np.max((nxres1, nxres2)))
nyresMax = int(np.max((nyres1, nyres2)))
wfeShapeMax = (nyresMax, nxresMax)

# Load FCM Calibration Data
try:
    fcmConfig = yaml.safe_load(open(fcm_fn, "r"))
except:
    raise Exception('Error loading yaml file with FCM data.')
focusPerStroke = fcmConfig["focusPerStroke"]  # [m RMS per um of stroke]
strokePerStep = fcmConfig["strokePerStep"]  # [um stroke per coarse step]
# -          0.459 nm rms = 1 um stroke
# -          5.3um per coarse FCM step.

# Load fixed settings for CONJUGATE
try:
    conjConfig = yaml.safe_load(open(conjugate_params_fn, "r"))
except:
    raise Exception('Error loading yaml file with CONJUGATE parameters.')
maxNollZern = conjConfig["maxNollZern"]
nIterSmooth = conjConfig["nIterSmooth"]
nBin = conjConfig["nBin"]

# Load amplitude and phase maps from phase retrieval
try:
    amp = fits.getdata(amp_fn)
except:
    raise Exception('Error loading yaml file with PR amplitude data.')
try:
    ph = fits.getdata(ph_fn)  # Units of radians
except:
    raise Exception('Error loading yaml file with PR phase data.')

# Crop or pad the PR maps
N = 400  # ceil_even(1.4*diamPupil)
wfeIn = inin(ph, (N, N))
ampIn = inin(amp, (N, N))
# Use the amplitude map to define which pixels to use from the PR map.
usablePixelMap = ampthresh(ampIn, nBin=nBin)
usablePixelMapCrop = inin(usablePixelMap, (N, N))

# Divide the WFE into 4 parts: Z1-Z3, Z4, Z5-Z11, and >Z11
zernCoefVec = fit_zernikes(wfeIn, usablePixelMap, maxNollZern, diamPupil,
                                 xOffsetPupil, yOffsetPupil)
wfeZ1to3, wfeZ4, wfeZ4or5toZmax = gen_split_wfe(flagUseFCM, maxNollZern,
                                              zernCoefVec, diamPupil,
                                              (N, N),
                                              xOffsetPupil, yOffsetPupil)
wfeAboveZmax = (wfeIn - wfeZ4or5toZmax - wfeZ4 - wfeZ1to3)*usablePixelMap

if flagPlot:
    plt.figure(4); plt.imshow(wfeZ4*usablePixelMapCrop); plt.colorbar(); plt.title('WFE from Z4'); plt.pause(0.1);
    plt.figure(123); plt.imshow(wfeZ1to3*usablePixelMapCrop); plt.colorbar(); plt.title('WFE from Z1,Z2,Z3'); plt.pause(0.1);
    plt.figure(124); plt.imshow((wfeZ4+wfeZ1to3)*usablePixelMapCrop); plt.colorbar(); plt.title('WFE from Z1-Z4'); plt.pause(0.1);


# %% Run CONJUGATE

Vdm1Out, Vdm2Out, cmdFCMOut, zernCoef = \
conjugate(cmdDM1, cmdDM2, cmdFCM, flagUseDM1, flagUseDM2, flagUseFCM,
              flagCorrectHighZern, lam, amp_fn, ph_fn, pupil_diam_fn,
              pupil_offsets_fn, dm_fn, fcm_fn, conjugate_params_fn)

dV1 = Vdm1Out - cmdDM1
dV2 = Vdm2Out - cmdDM2


# %% Verify that the output conjugates the input

dm1phase = volts_to_dmh(dm1gainmap, dV1, lam)
dm2phase = volts_to_dmh(dm2gainmap, dV2, lam)

dm1inffn = os.path.join(localpath, dm1r['inffn'])
DM1SurfActual = dmhtoph(N, N, dm1phase, dm1r['nact'], fits.getdata(dm1inffn), dm1r['ppact_d'], dm1r['ppact_cx'], dm1r['ppact_cy'], dm1r['dx'], dm1r['dy'], dm1r['thact'], flipx)
# DM1SurfActual = inin(DM1SurfActual, (N, N))

dm2inffn = os.path.join(localpath, dm2r['inffn'])
DM2SurfActual = dmhtoph(N, N, dm2phase, dm2r['nact'], fits.getdata(dm2inffn), dm2r['ppact_d'], dm2r['ppact_cx'], dm2r['ppact_cy'], dm2r['dx'], dm2r['dy'], dm2r['thact'], flipx)
# DM2SurfActual = inin(DM2SurfActual, (N, N))

# if flagPlot:
#     plt.figure(11); plt.imshow(wfeIn); plt.colorbar(); plt.clim(-np.pi, np.pi); plt.title('WFE'); plt.pause(0.1);
#     plt.figure(12); plt.imshow(usablePixelMap.astype(int)); plt.colorbar(); plt.title('Usable Pixel Map'); plt.pause(0.1);
    
#     # plt.figure(21); plt.imshow(surfDesDM1); plt.colorbar(); plt.title('Desired DM1 Surface'); plt.pause(0.1);
#     plt.figure(22); plt.imshow(DM1SurfActual); plt.colorbar(); plt.title('Best Fit DM1 Surface'); plt.pause(0.1);
#     # plt.figure(23); plt.imshow((surfDesDM1 - DM1SurfActual)*usablePixelMapCrop); plt.colorbar(); plt.title('Difference DM1'); plt.pause(0.1);

#     # plt.figure(31); plt.imshow(surfDesDM2); plt.colorbar(); plt.title('Desired DM2 Surface'); plt.pause(0.1);
#     plt.figure(32); plt.imshow(DM2SurfActual); plt.colorbar(); plt.title('Best Fit DM2 Surface'); plt.pause(0.1);
#     # plt.figure(33); plt.imshow((surfDesDM2 - DM2SurfActual)*usablePixelMapCrop); plt.colorbar(); plt.title('Difference DM2'); plt.pause(0.1);

#     # plt.figure(124); plt.imshow((wfeZ4+wfeZ1to3)*usablePixelMapCrop); plt.colorbar(); plt.title('WFE from Z1-Z4'); plt.pause(0.1);
#     # plt.figure(41); plt.imshow(usablePixelMapCrop*(wfeIn + 2*(DM1SurfActual + DM2SurfActual))); plt.colorbar(); plt.title('WFE'); plt.pause(0.1);
#     # plt.figure(41); plt.imshow(insertinto.insertinto(usablePixelMap, (N, N))*(insertinto.insertinto(wfeMeas, (N, N)) - DM1SurfActual - DM2SurfActual - wfeZ4 - wfeZ1to3)); plt.colorbar(); plt.title('WFE'); plt.pause(0.1);

#     plt.figure(42); plt.imshow(usablePixelMapCrop*(wfeIn - wfeZ1to3 - wfeZ4)); plt.colorbar(); plt.title('Z5+ Residuals'); plt.pause(0.1);
#     plt.figure(43); plt.imshow(usablePixelMapCrop*(wfeIn - wfeZ1to3 - wfeZ4 + 2*DM1SurfActual + 2*DM2SurfActual)); plt.colorbar(); plt.title('WFE Residual'); plt.pause(1);

# Figures for printing
if flagPlot:
    plt.figure(11); plt.imshow(wfeIn); plt.colorbar(); plt.gca().invert_yaxis(); plt.axis('off');  plt.clim(-2.5, 4); plt.pause(0.1);
    plt.figure(12); plt.imshow(usablePixelMap.astype(int)); plt.gca().invert_yaxis(); plt.axis('off'); plt.pause(0.1);
    # plt.figure(4); plt.imshow(wfeIn); plt.clim(-0.5, 0.5); plt.colorbar(); plt.axis('off'); plt.clim(-2.5, 4); plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig('ConjWFEIn.png', bbox_inches='tight', pad_inches = 0); plt.pause(0.1)

    # plt.figure(21); plt.imshow(surfDesDM1); plt.colorbar(); plt.title('Desired DM1 Surface'); plt.pause(0.1);
    plt.figure(22); plt.imshow(2*DM1SurfActual); plt.colorbar(); plt.gca().invert_yaxis(); plt.axis('off'); plt.pause(0.1);
    # plt.figure(1); plt.imshow(2*DM1SurfActual); plt.colorbar(); plt.axis('off');  plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig('ConjDM1HighZern.png', bbox_inches='tight', pad_inches = 0); plt.pause(0.1)
    # plt.figure(1); plt.imshow(2*DM1SurfActual); plt.colorbar(); plt.axis('off');  plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig('ConjDM1MidZern.png', bbox_inches='tight', pad_inches = 0); plt.pause(0.1)
    plt.figure(1); plt.imshow(2*DM1SurfActual); plt.colorbar(); plt.axis('off'); plt.clim(-2.0, 1.3);  plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig('ConjDM1MidHighZern.png', bbox_inches='tight', pad_inches = 0); plt.pause(0.1)

    # plt.figure(23); plt.imshow((surfDesDM1 - DM1SurfActual)*usablePixelMapCrop); plt.colorbar(); plt.title('Difference DM1'); plt.pause(0.1);

    # plt.figure(31); plt.imshow(surfDesDM2); plt.colorbar(); plt.title('Desired DM2 Surface'); plt.pause(0.1);
    plt.figure(32); plt.imshow(2*DM2SurfActual); plt.colorbar(); plt.gca().invert_yaxis(); plt.axis('off'); plt.pause(0.1);
    # plt.figure(33); plt.imshow((surfDesDM2 - DM2SurfActual)*usablePixelMapCrop); plt.colorbar(); plt.title('Difference DM2'); plt.pause(0.1);
    plt.figure(2); plt.imshow(2*DM2SurfActual); plt.colorbar(); plt.axis('off');  plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig('ConjDM2MidZern.png', bbox_inches='tight', pad_inches = 0); plt.pause(0.1)

    # plt.figure(124); plt.imshow((wfeZ4+wfeZ1to3)*usablePixelMapCrop); plt.colorbar(); plt.title('WFE from Z1-Z4'); plt.pause(0.1);
    # plt.figure(41); plt.imshow(usablePixelMapCrop*(wfeIn + 2*(DM1SurfActual + DM2SurfActual))); plt.colorbar(); plt.title('WFE'); plt.pause(0.1);
    # plt.figure(41); plt.imshow(insertinto.insertinto(usablePixelMap, (N, N))*(insertinto.insertinto(wfeMeas, (N, N)) - DM1SurfActual - DM2SurfActual - wfeZ4 - wfeZ1to3)); plt.colorbar(); plt.title('WFE'); plt.pause(0.1);

    plt.figure(42); plt.imshow(usablePixelMapCrop*(wfeIn - wfeZ1to3 - wfeZ4)); plt.gca().invert_yaxis(); plt.axis('off'); plt.colorbar();  plt.pause(0.1); #plt.title('Z5+ Residuals');
    plt.figure(43); plt.imshow(usablePixelMapCrop*(wfeIn - wfeZ1to3 - wfeZ4 + 2*DM1SurfActual + 2*DM2SurfActual)); plt.clim(-0.5, 0.5); plt.colorbar(); plt.gca().invert_yaxis(); plt.axis('off'); plt.pause(1); #plt.title('WFE Residual');
    # plt.figure(3); plt.imshow(usablePixelMapCrop*(wfeIn - wfeZ1to3 - wfeZ4 + 2*DM1SurfActual + 2*DM2SurfActual)); plt.clim(-0.5, 0.5); plt.colorbar(); plt.axis('off');  plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig('ConjResidual.png', bbox_inches='tight', pad_inches = 0); plt.pause(0.1)

    # plt.figure(5); plt.imshow(-wfeZ4); plt.colorbar(); plt.axis('off');  plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig('ConjZ4.png', bbox_inches='tight', pad_inches = 0); plt.pause(0.1)
    # plt.figure(6); plt.imshow(wfeZ1to3); plt.colorbar(); plt.axis('off');  plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig('ConjZ123.png', bbox_inches='tight', pad_inches = 0); plt.pause(0.1)
