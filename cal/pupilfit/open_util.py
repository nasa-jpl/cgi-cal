# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
# pylint: disable=maybe-no-member
"""Fit lateral offsets, magnification, and/or clocking of pupil masks."""
import numpy as np

from cal.ampthresh.ampthresh import ampthresh
from cal.util.insertinto import insertinto
from cal.util import check
from cal.maskgen.maskgen import rotate_shift_downsample_amplitude_mask
from cal.pupilfit.pf_util import compute_norm_factor


def compute_clocking(ampMeas, diamBeamAtMask, maskRefHighRes, diamHighRes,
                     xOffset, yOffset, clockDegMax, nClock, percentileForNorm,
                     deltaAmpMax, useAmpthresh=False, nBin=21):
    """
    Compute the clocking of a pupil or pupil mask compared to a reference file.

    Clocking is computed by rotating a reference mask by several values,
    summing the matching pixels for each value, and then performing a quadratic
    fit of the squared cost function to find the best clocking value. There are
    two options for what "matching" pixels means. The default is to normalize
    the mask representations and call pixels matching if they are within
    deltaAmpMax of each other. That option works well in simulation but might
    not work as well with real images (TBD). The non-default option is to use
    AMPTHRESH'ed (i.e., boolean) versions of the measured and reference pupils
    and call pixels matching if they have the same exact value (0 or 1).

    Parameters
    ----------
    ampMeas : array_like
        2-D measurement of pupil amplitude. Can be real- or complex-valued.
    diamBeamAtMask : float
        Diameter of beam arriving at mask to fit. Units of pixels.
    maskRefHighRes : array_like
        2-D reference mask amplitude at high resolution.
    diamHighRes : float
        Diameter of beam relative to the high-resolution mask. Units of pixels.
    xOffset, yOffset : float
        Known or estimated lateral offsets of the mask relative to the array
        center pixel. Units of pixels.
    clockDegMax : float
        Maximum +/- clocking values to try. Units of degrees. Somewhat larger
        values than the expected range can be useful for getting a good
        quadratic fit.
    nClock : int
        Number of uniformly-spaced clocking values to try between
        +/-clockDegMax and centered around zero.
    percentileForNorm : float
        Percentile at which to normalize the measured and reference masks if
        useAmpthresh == False. The pixels used in determining the percentile
        are found by running AMPTHRESH on the masks; the AMPTHRESH'ed masks are
        not used for later calculations, though.
    deltaAmpMax : float
        Maximum allowed amplitude difference allowed between a measured mask
        and a reference mask for that pixel to be counted as matching in value.
        Used only if useAmpthresh == False.
    useAmpthresh : bool, optional
        Whether to use AMPTHRESH'ed versions of the input arrays instead of
        percentile-normalized versions. Works better if False in simulation,
        but for real images it may be more accurate if set to True.
        The default is False.
    nBin : int
        Number of bins used when making a histogram of pupil values. Used
        by ampthresh(), so not used if useAmpthresh is False. Default is 21.


    Returns
    -------
    clockEst : float
        Estimate of mask clocking angle compared to the reference mask.
        Units of degrees.

    """
    check.twoD_array(ampMeas, 'ampMeas', TypeError)
    check.real_positive_scalar(diamBeamAtMask, 'diamBeamAtMask', TypeError)
    check.twoD_array(maskRefHighRes, 'maskRefHighRes', TypeError)
    check.real_positive_scalar(diamHighRes, 'diamHighRes', TypeError)
    check.real_scalar(xOffset, 'xOffset', TypeError)
    check.real_scalar(yOffset, 'yOffset', TypeError)
    check.real_scalar(clockDegMax, 'clockDegMax', TypeError)
    check.positive_scalar_integer(nClock, 'nClock', TypeError)
    check.real_nonnegative_scalar(percentileForNorm, 'percentileForNorm', TypeError)
    check.real_nonnegative_scalar(deltaAmpMax, 'deltaAmpMax', TypeError)
    if not isinstance(useAmpthresh, bool):
        raise TypeError('useAmpthresh')
    check.positive_scalar_integer(nBin, 'nBin', TypeError)

    # Line search for clocking
    clockVec = np.linspace(-clockDegMax, clockDegMax, nClock)
    padFac = 1  # zero padding should be done outside this function

    ampMeas = np.abs(ampMeas)
    if not useAmpthresh:  # Use edge values between 0 and 1 for fitting
        normFacMeas = compute_norm_factor(ampMeas,
                                          ampthresh(ampMeas, nBin=nBin),
                                          percentileForNorm)
        ampNorm = ampMeas/normFacMeas
        overlapVec = np.zeros((nClock,))
        for ic, clockDeg in enumerate(clockVec):

            pupilRef = rotate_shift_downsample_amplitude_mask(
                maskRefHighRes, clockDeg, diamBeamAtMask/diamHighRes, xOffset, yOffset, padFac)
            pupilRef = insertinto(pupilRef, ampMeas.shape)

            normFacRef = compute_norm_factor(pupilRef,
                                             ampthresh(pupilRef, nBin=nBin),
                                             percentileForNorm)
            pupilRefNorm = pupilRef/normFacRef
            compMatrix = np.abs(ampNorm - pupilRefNorm) < deltaAmpMax
            overlapVec[ic] = np.sum(compMatrix)

    else:  # Use only binary values (i.e., 0 and 1) for fitting
        ampBool = ampthresh(ampMeas, nBin=nBin)
        overlapVec = np.zeros((nClock,))
        for ic, clockDeg in enumerate(clockVec):

            pupilRef = rotate_shift_downsample_amplitude_mask(
                maskRefHighRes, clockDeg, diamBeamAtMask/diamHighRes, xOffset, yOffset, padFac)
            pupilRef = insertinto(pupilRef, ampMeas.shape)

            pupilRefBool = ampthresh(pupilRef, nBin=nBin)
            overlapVec[ic] = np.sum(ampBool == pupilRefBool)

    # Use quadratic fit to find best clocking estimate
    # 1) Subtract off the peak
    # 2) Square the data (because it looks like an absolute value function)
    # 3) Perform a fit to a parabola
    # parabola: polyCoefs[0]*clockVec**2 + polyCoefs[1]*clockVec + polyCoefs[2]
    overlapVec = overlapVec - np.max(overlapVec)
    polyCoefs = np.polyfit(clockVec, overlapVec**2, 2)
    if polyCoefs[0] == 0:  # Avoid divide by zero
        clockEst = 0.
    else:
        clockEst = -polyCoefs[1]/(2*polyCoefs[0])

    return clockEst
