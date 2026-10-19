# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Module to standardize Zernike analysis of complex pupil fields."""

import numpy as np
import os
from scipy.ndimage import binary_erosion

from cal.ampthresh.ampthresh import ampthresh
import cal.pupilfit.pupilfit_open as pfo
from cal.conjugate.util.propcustom import fit_zernikes, gen_zernikes
import cal.util.check as check

PFO_PATH = os.path.dirname(os.path.abspath(pfo.__file__))

# some default values
SELEM_DEFAULT = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0],
])


def pupilzernikes(wfe, amp, fn_pupil_fit_params, bMask=None, Z_noll_max=11,
                  mask_selem=SELEM_DEFAULT, nBin=21, data_path=PFO_PATH):
    """
    Perform Zernike analysis of pupil wavefront.

    Given a wavefront phase and either its amplitude array or a binary pupil
    mask:
    1. compute pupil diameter and center from wavefront amplitude
    2. create a pupil binary mask if necessary
    3. calculate Zernike coefficients
    4. calculate residual wavefront phase (=wavefront - zernike fit)

    Parameters
    ----------
    wfe : 2-d numpy array, wavefront phase, units are arbitrary, but typically
     radians or nm
    amp : wavefront amplitude, numpy array same size as wfe,
        used to determine pupil center and normalization radius for zernike
        fitting. Used to create pupil mask if bMask is None
    fn_pupil_fit_params : str
        Passed directly to pupilfit_open.fit_unmasked_pupil().
        Name of the YAML file containing the tuning parameters and other
        data used for fitting the pupil.
    bMask : (optional) binary numpy array, same size as wfe
        pupil mask, only wfe pixels where bMask == 1 are used in the Zernike
        fit.  If bMask is not input, a pupil is created from amp
    Z_noll_max : (default = 11) max Noll Zernike coefficient to calculate,
        min = 3
    mask_selem : (default = SELEM_DEFAULT)
        if the binary pupil mask is calculated from amp, mask_selem is the
        kernel used to erode the edges of the mask. The default kernel erodes
        the mask edges by one pixel.

        If optional bMask is given, then edge erosion with mask_selem is NOT
        done.  To skip the erosion step, set mask_selem = None
    nBin : int
        Number of bins used when making a histogram of pupil values.
        Used by ampthresh(). Default is 21.
    data_path : str
        Directory to serve as a base for relative paths with YAML files.
        If not supplied, defaults to directory containing this function. If
        using versions of the YAMLs delivered with the repository, the default
        will point to data files also delivered with the repository.

    Returns
    -------
    Zvec : numpy ndarray
        1-D array of Zernike coefficients. Same units as the wfe map
    dictResults : dict
        'bMask': binary mask used for Zernike fit, so that it can be reused
         for next
        'wfe_res_ptt': 2-d array size(wfe) = wfe - zernike fit piston, tip,
         tilt
        'wfe_res' : 2-d array size(wfe) = wfe - zernike fit Zvec
        'diamEst' : diameter (pixels) used for Zernike normalization radius
        'xOffset' : float, x-offset from image center pixel to pupil center
            Units of EXCAM pixels.
        'yOffset' : float, y-offset from image center pixel to pupil center.
            Units of EXCAM pixels.
    """
    # Check inputs
    check.twoD_array(wfe, 'wfe', TypeError)
    check.real_array(wfe, 'wfe', TypeError)
    check.twoD_array(amp, 'amp', TypeError)
    check.real_array(amp, 'amp', TypeError)

    if bMask is not None:
        check.twoD_array(bMask, 'bMask', TypeError)
        check.real_array(bMask, 'bMask', TypeError)
        if np.logical_and(bMask != 0, bMask != 1).any():
            raise TypeError('bMask can only contain booleans')

    check.positive_scalar_integer(Z_noll_max, 'Z_noll_max', TypeError)
    if Z_noll_max < 3:
        print('warning: Z_noll_max < 3, setting Z_noll_max = 3')
        Z_noll_max = 3

    if mask_selem is not None:
        check.twoD_array(mask_selem, 'mask_selem', TypeError)
        check.real_array(mask_selem, 'mask_selem', TypeError)

    check.positive_scalar_integer(nBin, 'nBin', TypeError)

    if not isinstance(fn_pupil_fit_params, str):
        raise TypeError('fn_pupil_fit_params must be a string')

    fn_pupil_fit_params = os.path.expandvars(fn_pupil_fit_params)
    data_path = os.path.expandvars(data_path)

    # 1. compute pupil diameter
    xOffset, yOffset, _, diamEst = pfo.fit_unmasked_pupil(
        amp, fn_pupil_fit_params, data_path=data_path,
    )

    # 2. make a pupil mask (check against mask used for unwrapping = bMask
    # above?)
    if bMask is None:
        # create bMask from amp
        bMask = ampthresh(amp, nBin=nBin)
        if mask_selem is not None:
            bMask = binary_erosion(bMask, structure=mask_selem)

    # 3. zernike decomposition, return Zvec same units as wfe (radians)
    Zvec = fit_zernikes(wfe, bMask, Z_noll_max, diamEst, xOffset, yOffset)

    # 4. generate residual maps for analysis and convenience
    wfe_res_ptt = wfe - bMask*gen_zernikes(
        np.array([1, 2, 3]), Zvec[:3], xOffset, yOffset, diamEst,
        nArray=wfe.shape[0]
    )
    wfe_res = wfe - bMask*gen_zernikes(
        np.arange(1, 1+Z_noll_max), Zvec, xOffset, yOffset, diamEst,
        nArray=wfe.shape[0]
    )

    # return values
    return Zvec, {
        # binary mask used for Zernike fit, so that it can be reused for next
        'bMask': bMask,
        # residual = wfe - zernike fit piston,tip,tilt
        'wfe_res_ptt': wfe_res_ptt,
        # residual = wfe - zernike fit Zvec
        'wfe_res': wfe_res,
        # diameter (pixels) used for Zernike normalization radius
        'diamEst': diamEst,
        # x-offset from image center pixel to pupil center (EXCAM pixels)
        'xOffset': xOffset,
        # y-offset from image center pixel to pupil center (EXCAM pixels)
        'yOffset': yOffset,
    }
