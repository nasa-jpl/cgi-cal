# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
# pylint: disable=maybe-no-member
"""Fit lateral offsets, magnification, and/or clocking of pupil masks."""
import os
import numpy as np
from astropy.io import fits
import scipy.ndimage as ImMorph

from cal.ampthresh.ampthresh import ampthresh
import cal.util.check as check
from cal.util.insertinto import insertinto
from cal.util.loadyaml import loadyaml
from cal.util.math import ceil_even
from cal.util.fit_shapes import fit_ellipse
from cal.maskgen.maskgen import rotate_shift_downsample_amplitude_mask
from cal.pupilfit.pf_util import compute_lateral_offsets

from cal.pupilfit.open_util import compute_clocking

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))


def fit_unmasked_pupil(pupil, fn_tuning, data_path=LOCAL_PATH):
    """
    Find diameter, clocking, and centering offsets of the unmasked pupil.

    All filenames may be absolute or relative paths.  If relative in input
    arguments, they will be relative to the current working directory, not to
    any particular location in Calibration.  If relative within YAML files
    (e.g. the fnMaskRefHighRes key in fn_tuning), this will be relative to the
    path in the data_path argument.

    Parameters
    ----------
    pupil : array_like
        2-D array containing the pupil amplitude or pupil intensity
    fn_tuning : str
        Name of the YAML file containing the tuning parameters and other
        data used for fitting the pupil.
    data_path : str
        Directory to serve as a base for relative paths with YAML files.
        If not supplied, defaults to directory containing this function. If
        using versions of the YAMLs delivered with the repository, the default
        will point to data files also delivered with the repository. In this
        case, it is the absolute path prefix for the high-res pupil reference
        file, fnMaskRefHighRes.

    Returns
    -------
    xOffset, yOffset: float
        Estimate of pupil's x- and y-axis offsets from the image center pixel.
        Units of ExCAM pixels.
    clockEst : float
        Estimate of pupil's clocking with respect to the array containing it.
        Units of degrees.
    diamEst : float
        Estimate of the pupil's diameter. Units of ExCAM pixels.

    Notes
    -----
    Works well for the unmasked Roman pupil. Does not work well for a shaped
    pupil probably because the number of edges is much higher, which can throw
    off how many pixels get rounded up or down by AMPTHRESH.
    """
    check.twoD_array(pupil, 'pupil', TypeError)
    check.real_array(pupil, 'pupil', TypeError)
    check.string(fn_tuning, 'fn_tuning', TypeError)
    check.string(data_path, 'data_path', TypeError)

    inp = loadyaml(fn_tuning)
    nIterOffsets = inp['nIterOffsets']
    nPadFFT = inp['nPadFFT']
    nFocusCrop = inp['nFocusCrop']
    nPhaseSteps = inp['nPhaseSteps']
    dPixel = inp['dPixel']
    clockDegMax = inp['clockDegMax']
    nClock = inp['nClock']
    fnMaskRefHighRes = inp['fnMaskRefHighRes']
    diamHighRes = inp['diamHighRes']
    rotRef = inp['rotRef']
    flipxRef = inp['flipxRef']
    padFacRef = inp['padFacRef']
    deltaAmpMax = inp['deltaAmpMax']
    percentileForImageNorm = inp['percentileForImageNorm']
    zeroPaddingForHighResMask = inp['zeroPaddingForHighResMask']
    n_binary_opening = inp['n_binary_opening']
    n_iter_dilate_erode = inp['n_iter_dilate_erode']
    pad_factor = inp['pad_factor']
    nBin = inp['nBin']

    check.positive_scalar_integer(nIterOffsets, 'nIterOffsets', TypeError)
    check.positive_scalar_integer(nPadFFT, 'nPadFFT', TypeError)
    check.positive_scalar_integer(nFocusCrop, 'nFocusCrop', TypeError)
    check.positive_scalar_integer(nPhaseSteps, 'nPhaseSteps', TypeError)
    check.real_positive_scalar(dPixel, 'dPixel', TypeError)
    check.real_positive_scalar(clockDegMax, 'clockDegMax', TypeError)
    check.positive_scalar_integer(nClock, 'nClock', TypeError)
    if not isinstance(fnMaskRefHighRes, str):
        raise TypeError('Wrong type for fn_prefix_high_res_sim_mask')
    check.real_positive_scalar(diamHighRes, 'diamHighRes', TypeError)
    check.real_scalar(rotRef, 'rotRef', TypeError)
    check.boolean(flipxRef, 'flipxRef', TypeError)
    check.real_positive_scalar(padFacRef, 'padFacRef', TypeError)
    check.real_nonnegative_scalar(deltaAmpMax, 'deltaAmpMax', TypeError)
    check.real_positive_scalar(percentileForImageNorm,
                               'percentileForImageNorm', TypeError)
    check.real_nonnegative_scalar(zeroPaddingForHighResMask,
                                  'zeroPaddingForHighResMask', TypeError)

    fnMaskRefHighRes = os.path.expandvars(fnMaskRefHighRes)
    data_path = os.path.expandvars(data_path)

    # Compute pupil diameter and coarse estimates of x- and y-offsets
    pupil = ampthresh(pupil, nBin=nBin)
    pupil = ImMorph.binary_opening(
        pupil, structure=np.ones((n_binary_opening,
                                  n_binary_opening))).astype(float)
    diamEst, xOffsetCoarse, yOffsetCoarse = fit_ellipse(
        pupil, n_iter_dilate_erode=n_iter_dilate_erode, pad_factor=pad_factor)

    # Compute fine estimates of x- and y-offsets
    xOffsetRef = 0
    yOffsetRef = 0
    maskRefHighRes = fits.getdata(os.path.join(data_path, fnMaskRefHighRes))
    pupilRef = rotate_shift_downsample_amplitude_mask(
        maskRefHighRes, rotRef, diamEst/diamHighRes, xOffsetRef, yOffsetRef,
        padFac=padFacRef, flipx=flipxRef)
    pupilRef = insertinto(pupilRef, (nPadFFT, nPadFFT))

    yOffset, xOffset = compute_lateral_offsets(
        pupil, pupilRef, diamEst, nPhaseSteps, dPixel, nPadFFT,
        nFocusCrop, nIter=nIterOffsets, useCoarse=False,
        xOffsetCoarse=xOffsetCoarse, yOffsetCoarse=yOffsetCoarse)

    # Compute clocking
    nPad = int(ceil_even(np.max(maskRefHighRes.shape) +
                         zeroPaddingForHighResMask))
    maskRefHighResPad = insertinto(maskRefHighRes, (nPad, nPad))
    clockEst = compute_clocking(
        pupil, diamEst, maskRefHighResPad, diamHighRes, xOffset, yOffset,
        clockDegMax, nClock, percentileForImageNorm, deltaAmpMax)

    return xOffset, yOffset, clockEst, diamEst
