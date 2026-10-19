# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
# pylint: disable=maybe-no-member
"""Fit lateral offsets, magnification, and/or clocking of pupil masks."""
from astropy.io import fits
import numpy as np
import os
import scipy.ndimage

from cal.ampthresh.ampthresh import ampthresh
from cal.util.insertinto import insertinto
import cal.util.check as check
from cal.util.loadyaml import loadyaml
from cal.util.remove_ptt import remove_ptt_directly
from cal.maskgen.maskgen import rotate_shift_downsample_amplitude_mask
from cal.pupilfit.pf_util import (
    compute_norm_factor, offcenter_crop, compute_lateral_offsets,
)
from cal.psffit.psffit import psffit
import cal.pr.util.unwrap as pru


LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))


def fit_shaped_pupil_mask_offsets(
        imageUnmasked, imageMasked, offsetGain, fnMaskParams, fnTuningParams,
        xOffsetPupil, yOffsetPupil, diamPupil, magAtMask, clockDegMask,
        xOffsetSPM, yOffsetSPM, data_path=LOCAL_PATH):
    """
    Compute the lateral shaped pupil offsets compared to an unmasked image.

    This function is a wrapper for fit_pupil_mask_offsets() with one step
    performed beforehand, which is shifting the unmasked image. The shaped
    pupil mask (SPM) lateral offset is more difficult to estimate because the
    reflective SPM can be in a different plane than the fold mirror used
    during initial calibrations. This can cause a lateral shear of the
    underlying pupil image when the SPM is moved into the beam. This function
    shifts the underlying, unmasked pupil image by the estimated z-induced
    beam shear (separately found with fit_z_induced_spm_shear()) before
    calling fit_pupil_mask_offsets().

    All filenames may be absolute or relative paths.  If relative in input
    arguments, they will be relative to the current working directory, not to
    any particular location in Calibration.  If relative within YAML files
    (e.g. the fnMaskRefHighRes key in fnMaskParams), this will be relative to
    the path in the data_path argument.

    Parameters
    ----------
    imageUnmasked : array_like
        Unmasked intensity image of the pupil or mask to be masked.
    imageMasked : array_like
        Intensity image of the mask over the pupil or other mask. Must have
        same array shape and centering as imageUnmasked.
    offsetGain : float
        The gain used when iterating to compute the offset value. Leave as 1
        for most cases. Should be >1 for the SPC-WFOV Lyot stop to SPM case
        because the underlying shaped pupil hides the outer edge of the Lyot
         stop and makes each offset estimate a few times smaller than what
         it actually is.
    fnMaskParams : str
        Name of YAML file containing info about the high-resolution reference
        mask to use in the fitting.
    fnTuningParams : str
        Name of YAML file containing fitting parameter values.
    xOffsetPupil, yOffsetPupil : float
        Estimated lateral offsets of the underlying, nominal telescope pupil.
        Units of pixels.
    diamPupil : float
        Estimated diameter of the nominal, unmasked telescope pupil as
        seen at the detector. Units of pixels.
    magAtMask : float
        Estimated magnification (compared to the nominal telescope pupil) at
        the plane of the mask being aligned. Usually is close to 1.
    clockDegMask : float
        Estimated clocking of the Mask being aligned. Units of degrees.
    xOffsetSPM, yOffsetSPM : float
        Estimated lateral pupil beam shear that occurs when switching from
        the unmasked pupil to the shaped pupil mask. Units of pixels.
    data_path : str
        Directory to serve as a base for relative paths with YAML files.
        If not supplied, defaults to directory containing this function. If
        using versions of the YAMLs delivered with the repository, the default
        will point to data files also delivered with the repository.

    Returns
    -------
    xOffsetEst, yOffsetEst : float
        Lateral offset estimate of the mask compared to the underlying pupil
        or mask. Units of ExCAM pixels.
    """
    check.twoD_array(imageUnmasked, 'imageUnmasked', TypeError)
    check.twoD_array(imageMasked, 'imageMasked', TypeError)
    sh0 = imageUnmasked.shape
    sh1 = imageMasked.shape
    if (sh0[0] != sh1[0]) or (sh0[1] != sh1[1]):
        raise ValueError('imageUnmasked and imageMasked must have same shape')
    check.real_positive_scalar(offsetGain, 'offsetGain', TypeError)
    check.string(fnMaskParams, 'fnMaskParams', TypeError)
    check.string(fnTuningParams, 'fnTuningParams', TypeError)
    check.real_scalar(yOffsetPupil, 'yOffsetPupil', TypeError)
    check.real_scalar(xOffsetPupil, 'xOffsetPupil', TypeError)
    check.real_positive_scalar(diamPupil, 'diamPupil', TypeError)
    check.real_positive_scalar(magAtMask, 'magAtMask', TypeError)
    check.real_scalar(clockDegMask, 'clockDegMask', TypeError)
    check.real_scalar(xOffsetSPM, 'xOffsetSPM', TypeError)
    check.real_scalar(yOffsetSPM, 'yOffsetSPM', TypeError)
    check.string(data_path, 'data_path', TypeError)

    imageUnmasked = scipy.ndimage.shift(imageUnmasked,
                                        [yOffsetSPM, xOffsetSPM])

    # New beam center location with SPM in place
    xOffsetTotal = xOffsetPupil + xOffsetSPM
    yOffsetTotal = yOffsetPupil + yOffsetSPM

    xOffsetEst, yOffsetEst = fit_pupil_mask_offsets(
        imageUnmasked, imageMasked, offsetGain, fnMaskParams, fnTuningParams,
        xOffsetTotal, yOffsetTotal, diamPupil, magAtMask, clockDegMask,
        data_path=data_path,
    )

    return xOffsetEst, yOffsetEst


def fit_pupil_mask_offsets(
        imageUnmasked, imageMasked, offsetGain, fnMaskParams, fnTuningParams,
        xOffsetPupil, yOffsetPupil, diamPupil, magAtMask, clockDegMask,
        data_path=LOCAL_PATH):
    """
    Compute the lateral mask offsets compared to an unmasked image.

    Parameters
    ----------
    imageUnmasked : array_like
        Unmasked intensity image of the pupil or mask to be masked.
    imageMasked : array_like
        Intensity image of the mask over the pupil or other mask. Must have
        same array shape and centering as imageUnmasked.
    offsetGain : float
        The gain used when iterating to compute the offset value. Leave as 1
        for most cases. Should be >1 for the SPC-WFOV Lyot stop to SPM case
        because the underlying shaped pupil hides the outer edge of the Lyot
         stop and makes each offset estimate a few times smaller than what
         it actually is.
    fnMaskParams : str
        Name of YAML file containing info about the high-resolution reference
        mask to use in the fitting.
    fnTuningParams : str
        Name of YAML file containing fitting parameter values.
    xOffsetPupil, yOffsetPupil : float
        Estimated lateral offsets of the underlying, nominal telescope pupil.
        Units of pixels.
    diamPupil : float
        Estimated diameter of the nominal, unmasked telescope pupil as
        seen at the detector. Units of pixels.
    magAtMask : float
        Estimated magnification (compared to the nominal telescope pupil) at
        the plane of the mask being aligned. Usually is close to 1.
    clockDegMask : float
        Estimated clocking of the Mask being aligned. Units of degrees.
    data_path : str
        Directory to serve as a base for relative paths with YAML files.
        If not supplied, defaults to directory containing this function. If
        using versions of the YAMLs delivered with the repository, the default
        will point to data files also delivered with the repository.

    Returns
    -------
    xOffsetEst, yOffsetEst : float
        Lateral offset estimate of the mask compared to the underlying pupil
        or mask. Units of ExCAM pixels.
    """
    check.twoD_array(imageUnmasked, 'imageUnmasked', TypeError)
    check.twoD_array(imageMasked, 'imageMasked', TypeError)
    sh0 = imageUnmasked.shape
    sh1 = imageMasked.shape
    if (sh0[0] != sh1[0]) or (sh0[1] != sh1[1]):
        raise ValueError('imageUnmasked and imageMasked must have same shape')
    check.real_positive_scalar(offsetGain, 'offsetGain', TypeError)
    check.string(fnMaskParams, 'fnMaskParams', TypeError)
    check.string(fnTuningParams, 'fnTuningParams', TypeError)
    check.real_scalar(yOffsetPupil, 'yOffsetPupil', TypeError)
    check.real_scalar(xOffsetPupil, 'xOffsetPupil', TypeError)
    check.real_positive_scalar(diamPupil, 'diamPupil', TypeError)
    check.real_positive_scalar(magAtMask, 'magAtMask', TypeError)
    check.real_scalar(clockDegMask, 'clockDegMask', TypeError)
    check.string(data_path, 'data_path', TypeError)
    imageShape = np.shape(imageUnmasked)

    # Load tuning parameters from YAML file
    inp = loadyaml(fnTuningParams)
    nPadFFT = inp['nPadFFT']
    nFocusCrop = inp['nFocusCrop']
    nPhaseSteps = inp['nPhaseSteps']
    dPixel = inp['dPixel']
    nIterOuterLoop = inp['nIterOuterLoop']
    nIterInnerLoop = inp['nIterInnerLoop']
    percentileForImageNorm = inp['percentileForImageNorm']
    nBin = inp['nBin']

    # Load size and name of the high-res mask template from a YAML file
    inp = loadyaml(fnMaskParams)
    diamHighResMaskRef = inp['diamHighResMaskRef']
    fnMaskRefHighRes = os.path.expandvars(inp['fnMaskRefHighRes'])
    data_path = os.path.expandvars(data_path)
    maskRefHighRes = fits.getdata(os.path.join(data_path, fnMaskRefHighRes))

    # Normalize images and convert to amplitude
    normFactorUnmasked = compute_norm_factor(
        imageUnmasked, ampthresh(imageUnmasked, nBin=nBin),
        percentileForImageNorm)
    imageUnmasked = imageUnmasked/normFactorUnmasked
    imageUnmasked[imageUnmasked < 0] = 0
    ampUnmasked = np.sqrt(imageUnmasked)

    normFactorMasked = compute_norm_factor(imageMasked,
                                           ampthresh(imageMasked, nBin=nBin),
                                           percentileForImageNorm)
    imageMasked = imageMasked/normFactorMasked
    imageMasked[imageMasked < 0] = 0
    ampMasked = np.sqrt(imageMasked)

    # Not used because useCoarse=True, but still need a value assigned to the
    # input argument for compute_lateral_offsets()
    diamMask = diamPupil

    xOffsetEst = 0
    yOffsetEst = 0
    for iter_ in range(nIterOuterLoop):

        maskRef = rotate_shift_downsample_amplitude_mask(
            maskRefHighRes, clockDegMask,
            diamPupil*magAtMask/diamHighResMaskRef, xOffsetEst+xOffsetPupil,
            yOffsetEst+yOffsetPupil, padFac=1.2, flipx=False)
        maskRef = insertinto(maskRef, imageShape)
        arraySim = ampUnmasked*maskRef

        # Crop to be centered on the pupil
        xOffsetPupilRound = int(np.round(xOffsetPupil))
        yOffsetPupilRound = int(np.round(yOffsetPupil))
        xCenterPupil = imageShape[1]//2 + xOffsetPupilRound
        yCenterPupil = imageShape[0]//2 + yOffsetPupilRound
        ampMaskedRecenter = offcenter_crop(ampMasked, nPadFFT,
                                           yCenterPupil, xCenterPupil)
        arraySimRecenter = offcenter_crop(arraySim, nPadFFT,
                                          yCenterPupil, xCenterPupil)

        yOffsetDelta, xOffsetDelta = compute_lateral_offsets(
            ampMaskedRecenter, arraySimRecenter, diamMask, nPhaseSteps, dPixel,
            nPadFFT, nFocusCrop, nIter=nIterInnerLoop, useCoarse=False,
        )
        xOffsetEst += offsetGain*xOffsetDelta
        yOffsetEst += offsetGain*yOffsetDelta

    return xOffsetEst, yOffsetEst


def fit_z_induced_spm_shear(ampUnpoked0, phUnpoked0, phPoked0,
                            ampUnpoked1, phUnpoked1, phPoked1,
                            fnConfig):
    """
    Compute the z-induced lateral shear of the SPM relative to the pupil.

    Because the shaped pupil masks (SPMs) are reflective, they can lie in a
    different plane than the regular fold mirror used for HLC and DM
    registration. This can cause beam shear, which complicates the task of
    aligning an SPM to the moved underlying pupil. This function uses a poked
    phase pattern before and after moving the SPM (roughly) into place to
    determine how much the underlying pupil sheared. This estimated pupil shear
    needs to be included in both the assumed beam position and the DM
    registration for both DMs. Because the SPAM is after the DMs, the relative
    alignment between DM1 and DM2 is unchanged by the SPMs.

    The poked phase pattern on the DM should be performed in an area that is
    (minimally) unonbscured for both the pupil and SPM. A small "+" shape can
    be used.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Note: If the SPMs are not coplanar with the xy alignment stage holding
    them, then the beam shear can be different for each SPM. In that case,
    this calibration should be performed separately for each SPM.

    Parameters
    ----------
    ampUnpoked0 : array_like
        amplitude part of the pupil phase retrieval without the poke pattern
        on the deformable mirror.
    phUnpoked0 : array_like
        phase part of the pupil phase retrieval without the poke pattern
        on the deformable mirror.
    phPoked0 : array_like
        phase part of the pupil phase retrieval with the poke pattern
        on the deformable mirror.
    ampUnpoked1 : array_like
        amplitude part of the SPM phase retrieval without the poke pattern
        on the deformable mirror.
    phUnpoked1 : array_like
        phase part of the SPM phase retrieval without the poke pattern
        on the deformable mirror.
    phPoked1 : array_like
        phase part of the SPM phase retrieval with the poke pattern
        on the deformable mirror.
    fnConfig : str
        Name of file with tuning parameters for PSFFIT.

    Returns
    -------
    xOffset : float
        x shear of the beam in the second set of E-fields compared to the
        first set. Units of pixels.
    yOffset : float
        y shear of the beam in the second set of E-fields compared to the
        first set. Units of pixels.
    """
    # Type checks
    check.twoD_array(ampUnpoked0, 'ampUnpoked0', TypeError)
    check.twoD_array(phUnpoked0, 'phUnpoked0', TypeError)
    check.twoD_array(phPoked0, 'phPoked0', TypeError)
    check.twoD_array(ampUnpoked1, 'ampUnpoked0', TypeError)
    check.twoD_array(phUnpoked1, 'phUnpoked1', TypeError)
    check.twoD_array(phPoked1, 'phPoked1', TypeError)
    check.string(fnConfig, 'fnConfig', TypeError)

    # Array size checks
    shape0 = ampUnpoked0.shape
    if phUnpoked0.shape != shape0:
        raise ValueError('ampUnpoked0 and phUnpoked0 must have same shape')
    if phPoked0.shape != shape0:
        raise ValueError('ampUnpoked0 and phPoked0 must have same shape')
    shape1 = ampUnpoked1.shape
    if phUnpoked1.shape != shape1:
        raise ValueError('ampUnpoked1 and phUnpoked1 must have same shape')
    if phPoked1.shape != shape0:
        raise ValueError('ampUnpoked1 and phPoked1 must have same shape')

    phDiff0 = phPoked0 - phUnpoked0
    phDiff0, swMask0 = pru.unwrap(phDiff0, ampUnpoked0)
    phDiff0 = remove_ptt_directly(phDiff0, swMask0)

    phDiff1 = phPoked1 - phUnpoked1
    phDiff1, swMask1 = pru.unwrap(phDiff1, ampUnpoked1)
    phDiff1 = remove_ptt_directly(phDiff1, swMask1)
    phDiff1 = insertinto(phDiff1, shape0)
    swMask1 = insertinto(swMask1, shape0)

    shift_computed, _ = psffit(swMask0*phDiff0, swMask1*phDiff1, fnConfig)
    xOffset = shift_computed[1]
    yOffset = shift_computed[0]

    return xOffset, yOffset
