# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to test HLC occulter generation.
"""
import os
import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from math import isclose
# import yaml
from astropy.io import fits  # for reading in FITS file test data
import matplotlib.pyplot as plt

from cal.maskgen.maskgen import rotate_shift_downsample_amplitude_mask
from cal.maskgen.maskgen import ceil_odd
from cal.util.insertinto import insertinto as inin
from cal.maskgen.util.thinfilm import solver, calc_complex_occulter


localpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# %%

# # From PMGI_transmission_only.xlsx
# lam = 600e-9
# d0 = 4*lam
# aoi = 0
# t_Ti_base = 0
# t_Ni_vec = [0]
# t_PMGI_vec = [100e-9]
# pol = 2
# [tCoef, rCoef] = calc_complex_occulter(lam, aoi, t_Ti_base, t_Ni_vec,
#                                   t_PMGI_vec, d0, pol)
# T = np.abs(tCoef)**2
# print(T)

# %%

# User inputs --> Make a function input
lam = 575e-9
shapeOut = (19, 19)

# Data from Calibration --> Put in YAML file
resEnd = 3.0  # desired ending pixels per lambda_central/D
rotDeg = 10  # CCW rotation of the mask [degrees]
aoi = 5.5  # Angle of incidence at FPM in instrument [deg]

xOffset = 0  # [pixels]
yOffset = 0  # [pixels]

# Mask specific settings --> Put in YAML file
fnPMGI = 'maskdesigns/HLC_20190210b_NFOV/run432_theta6.69imthicks_PMGIfnum32.5676970504_lam5.75e-07_requestdx5e-07.fits'
lam0 = 575e-9
hTi = 3e-9
hNi = 1.09176961403441e-07
dx = 5e-7  # pixel width and height in design file [m]
FN = 32.5676970504  # F number assumed in Dwight's design file
polState = 2  # 0 for s, 1 for p, 2 for mean of s and p
hRefCoef = 4  # waves
padFac = 1.2
flagScaleWithWavelength = True

rOccLamD = 2.8  # radius of occulter at central wavelength. Units of lambda/D.
minOversizeFacForFFT = 8  # minimum factor of array oversizing compared to the occulter.

# %%

# High-res files for fabrication from Dwight
fnPMGIFull = os.path.join(localpath, fnPMGI)
pmgi = fits.getdata(fnPMGIFull)
nFPM0 = ceil_odd(np.max(pmgi.shape))
pmgi = inin(pmgi, (nFPM0, nFPM0))  # force to be square, odd-sized array
footprint = np.zeros_like(pmgi, dtype=int)
footprint[pmgi > 0] = 1
# plt.figure(1); plt.imshow(pmgi); plt.gca().invert_yaxis(); plt.colorbar(); plt.title('PMGI'); plt.pause(0.1);
# plt.figure(3); plt.imshow(footprint); plt.gca().invert_yaxis(); plt.colorbar(); plt.title('footprint'); plt.pause(0.1);

# Find indices of occulter pixels and substrate-only pixels
indOcc = np.nonzero(footprint.flatten() == 1)[0]
indSubstrate = np.nonzero(footprint.flatten() == 0)[0]
# nPixOcc = indOcc.size

# # Coordinates for starting occulter representation
# fLamD = FN * lam0
# x0 = np.arange(-nFPM0/2, nFPM0/2) * dx / fLamD
# [X0, Y0] = np.meshgrid(x0, x0)

hRef = hRefCoef*lam0
hTiVec = hTi*np.ones_like(indOcc)
hNiVec = hNi*np.ones_like(indOcc)
hPMGIVec = pmgi.flatten()[indOcc]
# [tCoef, rCoef] = calc_complex_occulter(lam, aoi, hTi, [hNi], pmgi.flatten()[indOcc], hRef, polState)
[tCoef, rCoef] = calc_complex_occulter(lam, aoi, hTiVec, hNiVec, hPMGIVec, hRef, polState)
[tCoef00, rCoef00] = calc_complex_occulter(lam, aoi, [0], [0], [0], hRef, polState)  #  pixel[0, 0]

# Tfpm = np.zeros((nFPM0, nFPM0), dtype=complex)
# Tfpm.flatten()[indSubstrate] = tCoef00
# Tfpm.flatten()[indOcc] = tCoef.reshape((tCoef.size,))

TfpmFlat = np.zeros(nFPM0*nFPM0, dtype=complex)
TfpmFlat[indSubstrate] = tCoef00[0]
TfpmFlat[indOcc] = tCoef
occStart = TfpmFlat.reshape((nFPM0, nFPM0))
occStart = occStart/occStart[0, 0]

plt.figure(4); plt.imshow(np.angle(occStart)); plt.gca().invert_yaxis(); plt.hsv(); plt.colorbar(); plt.title('Starting angle(Tfpm)'); plt.pause(0.1);
plt.figure(5); plt.imshow(np.abs(occStart)); plt.gca().invert_yaxis(); plt.magma(); plt.colorbar(); plt.title('Starting abs(Tfpm)'); plt.pause(0.1);

plt.figure(11); plt.imshow(hNi*footprint); plt.set_cmap('inferno'); plt.axis('off');  plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig('maskgen_hlc_nickel.png', bbox_inches='tight', pad_inches = 0); plt.pause(0.1)
plt.figure(12); plt.imshow(pmgi); plt.set_cmap('inferno'); plt.axis('off');  plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig('maskgen_hlc_pmgi.png', bbox_inches='tight', pad_inches = 0); plt.pause(0.1)

plt.figure(13); plt.imshow(np.angle(occStart)); plt.set_cmap('hsv'); plt.axis('off');  plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig('maskgen_hlc_occStartAngle.png', bbox_inches='tight', pad_inches = 0); plt.pause(0.1)
plt.figure(14); plt.imshow(np.abs(occStart)); plt.set_cmap('inferno'); plt.axis('off');  plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig('maskgen_hlc_occStartAbs.png', bbox_inches='tight', pad_inches = 0); plt.pause(0.1)

plt.figure(15); plt.imshow(np.real(occStart)); plt.set_cmap('inferno'); plt.axis('off');  plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig('maskgen_hlc_occStartReal.png', bbox_inches='tight', pad_inches = 0); plt.pause(0.1)
plt.figure(16); plt.imshow(np.imag(occStart)); plt.set_cmap('inferno'); plt.axis('off');  plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig('maskgen_hlc_occStartImag.png', bbox_inches='tight', pad_inches = 0); plt.pause(0.1)


# %% Compute intermediate resolution mask to be used for FFT downsampling.
resStart = (FN * lam0) / dx  # pixels per lambda_central/D starting from FITS file

nPadEnd = int(np.ceil(2*rOccLamD*resEnd*minOversizeFacForFFT))

nPadMiddle = int(np.floor((resStart/resEnd) * nPadEnd))

resMiddle = (nPadMiddle/nPadEnd) * resEnd


mag = resMiddle / resStart

occMiddleReal = rotate_shift_downsample_amplitude_mask(
                                        np.real(occStart - occStart[0, 0]),
                                        rotDeg, mag, xOffset, yOffset, padFac)
occMiddleImag = rotate_shift_downsample_amplitude_mask(
                                        np.imag(occStart - occStart[0, 0]),
                                        rotDeg, mag, xOffset, yOffset, padFac)

occMiddle = occMiddleReal + 1j*occMiddleImag

occMiddle = inin(occMiddle, (nPadMiddle, nPadMiddle)) + occStart[0, 0]

plt.figure(6); plt.imshow(np.angle(occMiddle)); plt.gca().invert_yaxis(); plt.hsv(); plt.colorbar(); plt.title('angle(occMiddle)'); plt.pause(0.1);
plt.figure(7); plt.imshow(np.abs(occMiddle)); plt.gca().invert_yaxis(); plt.magma(); plt.colorbar(); plt.title('abs(occMiddle)'); plt.pause(0.1);
# plt.pause(4);

n = 250
occMiddleCrop = inin(occMiddle, (n, n))

plt.figure(17); plt.imshow(np.real(occMiddleCrop)); plt.set_cmap('inferno'); plt.axis('off');  plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig('maskgen_hlc_occMiddleReal.png', bbox_inches='tight', pad_inches = 0); plt.pause(0.1)
plt.figure(18); plt.imshow(np.imag(occMiddleCrop)); plt.set_cmap('inferno'); plt.axis('off');  plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig('maskgen_hlc_occMiddleImag.png', bbox_inches='tight', pad_inches = 0); plt.pause(0.1)


# %% Fourier downsample

occFT = inin(fftshift(ifft2(ifftshift(occMiddle))), (nPadEnd, nPadEnd))
occEnd = fftshift(fft2(ifftshift(occFT)))


occEndPad = inin(occEnd - occStart[0, 0], shapeOut) + occStart[0, 0]



plt.figure(8); plt.imshow(np.angle(occEnd)); plt.gca().invert_yaxis(); plt.hsv(); plt.colorbar(); plt.title('angle(occEnd)'); plt.pause(0.1);
plt.figure(9); plt.imshow(np.abs(occEnd)); plt.gca().invert_yaxis(); plt.magma(); plt.colorbar(); plt.title('abs(occEnd)'); plt.pause(0.1);

plt.figure(10); plt.imshow(np.angle(occEndPad)); plt.gca().invert_yaxis(); plt.hsv(); plt.colorbar(); plt.title('angle(occEndPad)'); plt.pause(0.1);
plt.figure(11); plt.imshow(np.abs(occEndPad)); plt.gca().invert_yaxis(); plt.magma(); plt.colorbar(); plt.title('abs(occEndPad)'); plt.pause(0.1);

plt.figure(19); plt.imshow(np.real(occEndPad)); plt.set_cmap('inferno'); plt.axis('off');  plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig('maskgen_hlc_occEndReal.png', bbox_inches='tight', pad_inches = 0); plt.pause(0.1)
plt.figure(20); plt.imshow(np.imag(occEndPad)); plt.set_cmap('inferno'); plt.axis('off');  plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig('maskgen_hlc_occEndImag.png', bbox_inches='tight', pad_inches = 0); plt.pause(0.1)


# plt.pause(20);

# %%
# Compute low-res mask
# Currently adjust the x- and y-offsets to have a constant offset in meters,
# not in pixels.
# Have to subtract off Tfpm[0, 0] before resampling for the zero padding step
# not to cause problems. Tfpm[0, 0] is then added back after resampling.
# lam = 0.5*575e-9

# lam = 575e-9
# lam = 825e-9


if flagScaleWithWavelength:
    mag = resEnd / resStart * (lam0 / lam)
else:
    mag = resEnd / resStart

TfpmOutReal = rotate_shift_downsample_amplitude_mask(
                                            np.real(occStart - occStart[0, 0]), rotDeg,
                                            mag, xOffset, yOffset, padFac)
TfpmOutImag = rotate_shift_downsample_amplitude_mask(
                                            np.imag(occStart - occStart[0, 0]), rotDeg,
                                            mag, xOffset, yOffset, padFac)
# TfpmOutReal = rotate_shift_downsample_amplitude_mask(
#                                             np.real(Tfpm - Tfpm[0, 0]), rotDeg,
#                                             mag, xOffset*scaleFac, yOffset*scaleFac, padFac)
# TfpmOutImag = rotate_shift_downsample_amplitude_mask(
#                                             np.imag(Tfpm - Tfpm[0, 0]), rotDeg,
#                                             mag, xOffset*scaleFac, yOffset*scaleFac, padFac)
TfpmOut = TfpmOutReal + 1j*TfpmOutImag

TfpmOut = inin(TfpmOut, shapeOut) + occStart[0, 0]

plt.figure(16); plt.imshow(np.angle(TfpmOut)); plt.gca().invert_yaxis(); plt.hsv(); plt.colorbar(); plt.title('angle(TfpmOut)'); plt.pause(0.1);
plt.figure(17); plt.imshow(np.abs(TfpmOut)); plt.gca().invert_yaxis(); plt.magma(); plt.colorbar(); plt.title('abs(TfpmOut)'); plt.pause(0.1);
plt.pause(10);
