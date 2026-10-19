# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
# pylint: disable=maybe-no-member
"""Fit lateral offsets, magnification, and/or clocking of pupil masks."""
import os
import numpy as np
from numpy.fft import fftshift, fft2
from astropy.io import fits

from cal.ampthresh.ampthresh import ampthresh
from cal.util import check, findopt, loadyaml
from cal.util.shapes import circle
from cal.util.insertinto import insertinto
from cal.maskgen.maskgen import rotate_shift_downsample_amplitude_mask
from cal.pupilfit.pf_util import (
    coarsely_locate_pupil_offset, compute_norm_factor, compute_lateral_offsets,
    offcenter_crop,
)
from cal.pupilfit.open_util import compute_clocking

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))


def fit_shaped_pupil_mag_clocking(spotMeas, fnShapedPupilCalib):
    """
    Compute the magnification and clocking of the shaped pupil masks.

    Uses two vertically aligned circle fiducials on the shared SPM substrate to
    compute the magnification and clocking of all the SPMs. For the image used
    as the input, the SPAM must have the beam centered halfway between the
    centers of either the top two or bottom two SPMs. Fiducials are
    needed because the algorithms that work well on the pupil and Lyot stops
    work poorly with the shaped pupils. The densely packed obscurations of the
    SPMs confuse the pixel-matching algorithms used for the other cases.

    If the spots in the image are horizontally aligned, then set rotation=0 in
    the config file. If they are vertically aligned, then set rotation=90.

    The alignment algorithm works by minimizing the squared difference between
    numerically generated fringes. The fringes are the squared FFT of either
    the estimated or measured pair of alignment spots. Squaring the FFTs is
    useful because then the measured spots don't have to be centered in the
    array--the tip/tilt in phase goes away and only the amplitude is left.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Parameters
    ----------
    spotMeas : array_like
        2-D pupil image with the pupil centered halfway between either the
        upper two or lower two SPMs' centers. The two circular fiducial spots
        must be fully visible.
    fnShapedPupilCalib : str
        Name of the YAML file containing fitting settings and the geometry of
        the fiducials.

    Returns
    -------
    magEst : float
        Estimated magnification (compared to the nominal telescope pupil) at
        the SPAM.
    clockEst : float
        Estimated clocking of the SPMs compared to being vertically oriented.
        Units of degrees.
    """
    check.twoD_array(spotMeas, 'spotMeas', TypeError)
    check.real_array(spotMeas, 'spotMeas', TypeError)

    inp = loadyaml.loadyaml(fnShapedPupilCalib)
    nIter = inp['nIter']
    clockMaxDeg = inp['clockMaxDeg']
    nClock = inp['nClock']
    DeltaMag = inp['DeltaMag']
    nMag = inp['nMag']
    dBeamNom = inp['dBeamNom']
    dCircleM = inp['dCircleM']
    dyCircM = inp['dyCircM']
    nBeamNom = inp['nBeamNom']
    radiusROI = inp['radiusROI']
    nPixFFT = inp['nPixFFT']
    nPixOut = inp['nPixOut']
    nSubpixels = inp['nSubpixels']
    nBin = inp['nBin']
    rotation = inp['rotation']

    # Find the approximate center of the pupil image.
    [xcPupEst, ycPupEst] = coarsely_locate_pupil_offset(spotMeas, nBeamNom,
                                                        shrink_factor=3.)
    spotMeasCrop = offcenter_crop(
        spotMeas, nPixFFT,
        int(np.floor(spotMeas.shape[0]/2.0)+ycPupEst),
        int(np.floor(spotMeas.shape[1]/2.0)+xcPupEst)
    )

    # Determine where the spots are; necessary if the pupil center is
    # shifted left-right of the spot centers
    # --> Not needed because PAM accuracy should put beam centered correctly
    # to within a few microns, and this algorithm needs an accuracy of a few
    # pixels.

    # Make regions of interest (ROIs) where the fiducial spots should be, and
    # zero out the rest of the image.
    dyCircPix = dyCircM/dBeamNom*nBeamNom
    rotRad = np.radians(rotation)
    rotMat = np.array([[np.cos(rotRad), -np.sin(rotRad)],
                       [np.sin(rotRad), np.cos(rotRad)]])
    xyRight = rotMat @ np.array([[1], [0]]) * dyCircPix/2  # pixels
    xyLeft = rotMat @ np.array([[-1], [0]]) * dyCircPix/2  # pixels
    roiRight = circle(nPixFFT, nPixFFT, radiusROI, xyRight[0][0],
                      xyRight[1][0], nSubpixels=nSubpixels)
    roiLeft = circle(nPixFFT, nPixFFT, radiusROI, xyLeft[0][0],
                     xyLeft[1][0], nSubpixels=nSubpixels)
    spotMeasCrop = spotMeasCrop * (roiRight + roiLeft)
    spotMeasCrop = ampthresh(spotMeasCrop, nBin).astype(float)

    spotMeasFT = fftshift(fft2(fftshift(spotMeasCrop)))/nPixFFT
    spotMeasFT2 = np.abs(spotMeasFT)**2
    spotMeasFT2 = spotMeasFT2/np.max(spotMeasFT2)  # normalize
    spotMeasFT2crop = insertinto(spotMeasFT2, (nPixOut, nPixOut))

    # Loop over mag and clocking values to make a grid of cost function values
    clockVec = np.linspace(-clockMaxDeg, clockMaxDeg, nClock)
    magVec = np.linspace(1-DeltaMag, 1+DeltaMag, nMag)
    costMat = np.zeros((nClock, nMag))
    deltaMag = 0  # initialize at 0
    deltaClock = 0  # initialize at 0

    for iter_ in range(nIter):

        # GENERATE THE MATRIX OF COST FUNCTION VALUES
        for iMag, magNew in enumerate((magVec-deltaMag)):
            for iClock, clockTempDeg in enumerate((clockVec-deltaClock)):
                clockTempRad = clockTempDeg*(np.pi/180.0)
                mPerPix = dBeamNom / (nBeamNom*magNew)  # meters per pixel
                rotTotal = clockTempRad + rotRad

                rotMat = np.array([[np.cos(rotTotal), -np.sin(rotTotal)],
                                   [np.sin(rotTotal), np.cos(rotTotal)]])
                xyRight = rotMat @ np.array([[1], [0]]) * dyCircM/mPerPix/2.0
                xyLeft = rotMat @ np.array([[-1], [0]]) * dyCircM/mPerPix/2.0

                # Generate the alignment circles
                circ1Temp = circle(nPixFFT, nPixFFT, dCircleM/mPerPix/2.0,
                                   xyRight[0][0], xyRight[1][0],
                                   nSubpixels=nSubpixels)
                circ2Temp = circle(nPixFFT, nPixFFT, dCircleM/mPerPix/2.0,
                                   xyLeft[0][0], xyLeft[1][0],
                                   nSubpixels=nSubpixels)
                spotTemp = np.abs(circ1Temp + circ2Temp)**2
                spotTemp = ampthresh(spotTemp, nBin).astype(float)

                spotTempPad = insertinto(spotTemp, (nPixFFT, nPixFFT))

                spotFT = fftshift(fft2(fftshift(spotTempPad)))/nPixFFT
                spotFT2 = np.abs(spotFT)**2
                spotFT2 = spotFT2/np.max(spotFT2)  # normalize

                spotFT2Crop = insertinto(spotFT2, (nPixOut, nPixOut))

                costMat[iClock, iMag] = \
                    np.sum((spotFT2Crop - spotMeasFT2crop)**2)

        # Find the mag and clocking that minimize the cost function.
        # At the first iteration, uses the minimum instead of a
        # quadratic fit in order to get a much larger capture range.
        if iter_ == 0 and nIter > 1:
            bestInds = np.unravel_index(np.argmin(costMat), costMat.shape)
            xSol = magVec[bestInds[1]] - deltaMag
            ySol = clockVec[bestInds[0]] - deltaClock
        else:
            # Fit to a 3-D paraboloid to find best (x,y) center
            xSol, ySol = findopt.find_optimum_2d(
                magVec-deltaMag, clockVec-deltaClock, costMat,
                np.ones_like(costMat))

        deltaMag = 1 - xSol
        deltaClock = -ySol

    magEst = xSol
    clockEst = ySol

    return magEst, clockEst


def fit_lyot_stop_mag_clocking(imageUnmasked, imageMasked,
                               xOffsetPupil, yOffsetPupil,
                               fnMaskParams, fnOffsetParams, fnLyotCalib,
                               data_path=LOCAL_PATH):
    """
    Compute the magnification and clocking of the SPC-Spec Lyot stop mask.

    The clocking of the other Lyot stops will be determined by this function's
    output combined with relative clocking measurements of all the Lyot stops
    after they are bonded onto the LSAM plate. Only the SPC-Spec Lyot stop is
    used by this function because the other required Lyot stops have struts
    that cause confusion when comparing clocking against the unmasked pupil,
    which also has struts.

    All filenames may be absolute or relative paths.  If relative in input
    arguments, they will be relative to the current working directory, not to
    any particular location in Calibration.  If relative within YAML files
    (e.g. the fnMaskRefHighRes key in fnMaskParams), this will be relative to
    the path in the data_path argument.

    Parameters
    ----------
    imageUnmasked : array_like
        Unmasked intensity image of the unmasked pupil.
    imageMasked : array_like
        Intensity image of the Lyot stop applied to the pupil (no SPM in
        place). Must have same array shape and centering as imageUnmasked.
    xOffsetPupil, yOffsetPupil : float
        Estimated lateral offsets of the underlying, nominal telescope pupil.
        Units of ExCAM pixels.
    fnMaskParams : str
        Name of YAML file containing the Lyot stop parameters needed for
        this fitting routine.
    fnOffsetParams : str
        Name of YAML file containing offset fitting parameters.
    fnLyotCalib : str
        Name of YAML file containing Lyot stop calibration parameters.
    data_path : str
        Directory to serve as a base for relative paths with YAML files.
        If not supplied, defaults to directory containing this function. If
        using versions of the YAMLs delivered with the repository, the default
        will point to data files also delivered with the repository.

    Returns
    -------
    magEst : float
        Estimated magnification (compared to the nominal telescope pupil) at
        the LSAM.
    clockEst : float
        Estimated clocking compared to the reference Lyot stop file.
        Units of degrees.

    Notes
    -----
    Variables loaded from the YAML file of tuning values are:
    clockMaxDeg : float
        Maximum +/- clocking values to try. Units of degrees. Somewhat larger
        values than the expected range can be useful for getting a good
        quadratic fit.
    nClock : int
        Number of uniformly-spaced clocking values to try.
    deltaAmpMax : float
        Maximum allowed amplitude difference allowed between a measured mask
        and a reference mask for that pixel to be counted as matching in value.
    diamPupil : float
        Estimated diameter of the nominal telescope pupil.
        Units of ExCAM pixels.
    magShrinkFac : float
        Factor by which to shrink deltaMag each loop iteration.
        Must be > 0 and < 1.
    deltaMag : float
        +/- amount of delta magnification to use in a line search of
        magnification values. Must be > 0.
    nMag : int
        Number of magnification values to use in the line search.
    nIterMag : int
        Number of iterations over which to refine the magnification estimate.
    """
    check.twoD_array(imageUnmasked, 'imageUnmasked', TypeError)
    check.twoD_array(imageMasked, 'imageMasked', TypeError)
    sh0 = imageUnmasked.shape
    sh1 = imageMasked.shape
    if (sh0[0] != sh1[0]) or (sh0[1] != sh1[1]):
        raise ValueError('imageUnmasked and imageMasked must have same shape')
    check.real_scalar(xOffsetPupil, 'xOffsetPupil', TypeError)
    check.real_scalar(yOffsetPupil, 'yOffsetPupil', TypeError)
    check.string(fnMaskParams, 'fnMaskParams', TypeError)
    check.string(fnOffsetParams, 'fnOffsetParams', TypeError)
    check.string(fnLyotCalib, 'fnLyotCalib', TypeError)
    check.string(data_path, 'data_path', TypeError)

    inp = loadyaml.loadyaml(fnLyotCalib)
    clockMaxDeg = inp['clockMaxDeg']
    nClock = inp['nClock']
    deltaAmpMax = inp['deltaAmpMax']
    diamPupil = inp['diamPupil']
    magShrinkFac = inp['magShrinkFac']
    deltaMag = inp['deltaMag']
    nMag = inp['nMag']
    nIterMag = inp['nIterMag']
    check.real_positive_scalar(clockMaxDeg, 'clockMaxDeg', TypeError)
    check.positive_scalar_integer(nClock, 'nClock', TypeError)
    check.real_positive_scalar(deltaAmpMax, 'deltaAmpMax', TypeError)
    check.real_positive_scalar(diamPupil, 'diamPupil', TypeError)
    if deltaAmpMax >= 1:
        raise ValueError('deltaAmpMax must be less than 1')
    check.real_positive_scalar(magShrinkFac, 'magShrinkFac', TypeError)
    if magShrinkFac >= 1:
        raise ValueError('magShrinkFac must be less than 1')
    check.real_positive_scalar(deltaMag, 'deltaMag', TypeError)
    check.positive_scalar_integer(nMag, 'nMag', TypeError)
    check.positive_scalar_integer(nIterMag, 'nIterMag', TypeError)

    inp = loadyaml.loadyaml(fnMaskParams)
    diamHighResMaskRef = inp['diamHighResMaskRef']
    fnMaskRefHighRes = os.path.expandvars(inp['fnMaskRefHighRes'])
    OD_LS = inp['OD_LS']

    inp = loadyaml.loadyaml(fnOffsetParams)
    nPadFFT = inp['nPadFFT']
    nFocusCrop = inp['nFocusCrop']
    nPhaseSteps = inp['nPhaseSteps']
    dPixel = inp['dPixel']
    percentileForImageNorm = inp['percentileForImageNorm']
    nBin = inp['nBin']

    data_path = os.path.expandvars(data_path)
    maskRefHighRes = fits.getdata(os.path.join(data_path, fnMaskRefHighRes))

    # Normalize and square root images
    # normFacMasked = compute_norm_factor(imageMasked, ampthresh(imageMasked),
    #                                     percentileForImageNorm)
    # imageMasked = imageMasked/normFacMasked
    # imageMasked[imageMasked < 0] = 0
    # ampMasked = np.sqrt(imageMasked)

    bMask = ampthresh(imageUnmasked, nBin=nBin)
    normFacUnmasked = compute_norm_factor(imageUnmasked,
                                          bMask,
                                          percentileForImageNorm)
    imageUnmasked = imageUnmasked/normFacUnmasked
    imageUnmasked[imageUnmasked < 0] = 0
    ampUnmasked = np.sqrt(imageUnmasked)

    normFacMasked = normFacUnmasked
    imageMasked = imageMasked/normFacMasked
    imageMasked[imageMasked < 0] = 0
    ampMasked = np.sqrt(imageMasked)

    # INITIAL MAGNIFICATION ESTIMATE
    xOffset0 = 0
    yOffset0 = 0
    clockDegEst = 0
    imageShape = np.shape(imageUnmasked)
    maskLyotRef0 = rotate_shift_downsample_amplitude_mask(
        maskRefHighRes, clockDegEst, diamPupil/diamHighResMaskRef,
        xOffset0+xOffsetPupil, yOffset0+yOffsetPupil,
    )
    maskLyotRef0 = insertinto(maskLyotRef0, imageShape)

    # Use ampthresh on the image instead of the amplitude because ampthresh
    # doesn't work well when on square-rooted noisy images with values less
    # than zero zeroed out. Otherwise, lots of pixels outside the pupil
    # get counted as well.
    magEst0 = np.sqrt(np.sum(ampMasked[bMask])
              / np.sum((ampUnmasked*maskLyotRef0)[bMask]))

    # LATERAL TRANSLATION
    maskLyotRef1 = rotate_shift_downsample_amplitude_mask(
        maskRefHighRes, clockDegEst, magEst0*diamPupil/diamHighResMaskRef,
        xOffset0+xOffsetPupil, yOffset0+yOffsetPupil,
    )
    maskLyotRef1 = insertinto(maskLyotRef1, imageShape)
    ampRef1 = maskLyotRef1*ampUnmasked

    diamBeamEst = magEst0*diamPupil
    diamMaskEst = magEst0*diamPupil*OD_LS
    xOffset1, yOffset1 = coarsely_locate_pupil_offset(ampMasked, diamMaskEst)

    ampMaskedCrop = offcenter_crop(ampMasked, int(np.max(imageShape)),
                                   yOffset1+imageShape[0]//2,
                                   xOffset1+imageShape[1]//2)
    ampRef1Crop = offcenter_crop(ampRef1, int(np.max(imageShape)),
                                 yOffset1+imageShape[0]//2,
                                 xOffset1+imageShape[1]//2)
    yOffset2, xOffset2 = compute_lateral_offsets(
        ampMaskedCrop, ampRef1Crop, diamBeamEst, nPhaseSteps, dPixel, nPadFFT,
        nFocusCrop, useCoarse=False, nIter=2)

    # CLOCKING
    clockEst = compute_clocking(ampMasked, diamBeamEst, maskRefHighRes,
                                diamHighResMaskRef, xOffset2+xOffsetPupil,
                                yOffset2+yOffsetPupil, clockMaxDeg, nClock,
                                percentileForImageNorm, deltaAmpMax)

    # MAGNIFICATION UPDATE: LINE SEARCH
    magEst = magEst0

    for iter_ in range(nIterMag):
        magVec = magEst + magShrinkFac*np.linspace(-deltaMag, deltaMag, nMag)
        overlapVec = np.zeros((nMag,))

        for im, mag in enumerate(magVec):

            pupilRef = rotate_shift_downsample_amplitude_mask(
                maskRefHighRes, clockEst, mag*diamPupil/diamHighResMaskRef,
                xOffset2+xOffsetPupil, yOffset2+yOffsetPupil,
            )
            pupilRef = insertinto(pupilRef, ampMasked.shape)
            pupilRef = pupilRef**2

            normFacRef = compute_norm_factor(pupilRef,
                                             ampthresh(pupilRef,nBin=nBin),
                                             percentileForImageNorm)
            pupilRefNorm = pupilRef/normFacRef
            compMatrix = np.abs(ampMasked - pupilRefNorm) < deltaAmpMax
            overlapVec[im] = np.sum(compMatrix)

        # Find the maximum of the overlap
        if iter_ == 0 and nIterMag > 1:
            bestInd = np.argmax(overlapVec)
            magEst = magVec[bestInd]
        else:
            # Use quadratic fit to find best clocking estimate
            # 1) Subtract off the peak
            # 2) Square the data (because it looks like an abs value function)
            # 3) Perform a fit to a parabola
            # polyCoefs[0]*clockVec**2 + polyCoefs[1]*clockVec + polyCoefs[2]
            overlapVec = overlapVec - np.max(overlapVec)
            polyCoefs = np.polyfit(magVec, overlapVec**2, 2)
            if polyCoefs[0] == 0:  # Avoid divide by zero
                magEst = 1.
            else:
                magEst = -polyCoefs[1]/(2*polyCoefs[0])

    return magEst, clockEst
