# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Module to hold utility functions for the DMREG module."""
import numpy as np
from scipy.ndimage import binary_erosion

from cal.ampthresh.ampthresh import ampthresh
from cal.conjugate.util.propcustom import gen_zernikes,  fit_zernikes
from cal.util import check
from cal.util.dmhtoph import dmhtoph
from cal.util.insertinto import insertinto as inin


def calc_spm_delta_zernikes(
        ph0, amp1, ph1,
        diamPupil, xOffsetPupil, yOffsetPupil,
        Z_noll_max=11, nErode=1, nBin=21,
    ):
    """
    Compute the difference between two phase retrievals as low-order Zernikes.

    The main usage case is computing the change in low-order Zernikes when
    switching from the open-pupil case to having the apodizer inserted in
    order to know and model the low-order WFE at the apodizer.

    Parameters
    ----------
    ph0 : array_like
        2-D array of the unwrapped phase for the first phase retrieval,
        typically the reference one without the apodizer in the beam.
    amp1 : array_like
        2-D array of the amplitude for the second phase retrieval,
        typically the one with the apodizer inserted into the beam.
    ph1 : array_like
        2-D array of the unwrapped phase for the second phase retrieval,
        typically the one with the apodizer inserted into the beam.
    diamPupil : float
        Diameter of the unmasked beam in the phase retrievals. Units of pixels.
    xOffsetPupil, yOffsetPupil : float
        Lateral offsets of the unmasked pupil beam relative to the array
        containing the phase retrieval. Units of pixels.
    Z_noll_max : int
        Noll index of the maximum Zernike mode to include in the fit of the
        phase difference between ph1 and ph0. Must be >=4 because tip, tilt,
        and piston are thrown out. Default is 11.
    nErode : int
        Number of pixels to erode amp1 for making a software mask for the
        low-order fit. The phase estimates are usually incorrect right at the
        edge of apodizer blobs and should be masked out, but if this value is
        too large then the whole pupil could be masked out. Units of pixels.
        Default is 1.
    nBin : int
        Number of bins used when making a histogram of pupil values. Passed
        directly to ampthresh(). Default is 21.

    Returns
    -------
    Zvec : list
        List of the best-fit Zernike coefficients for Zernike Noll indices 1 to
        Z_noll_max over the computed software mask.
    zern_combo : array_like
        2-D array the same size as amp1 containing the sum of the Zernikes for
        Noll indices 4 to Z_noll_max using the coefficients from Zvec.
    wfe_actual : array_like
        2-D array the same size as amp1 containing the difference of ph1 and
        ph0 multiplied by the binary mask, bMask.
    bMask : array_like
        2-D array of the pixels to use for the phase fitting. Made by eroding
        the amp1 array.

    """
    # Check inputs
    check.twoD_array(ph0, 'ph0', TypeError)
    check.twoD_array(amp1, 'amp1', TypeError)
    check.twoD_array(ph1, 'ph1', TypeError)
    check.real_positive_scalar(diamPupil, 'diamPupil', TypeError)
    check.real_scalar(xOffsetPupil, 'xOffsetPupil', TypeError)
    check.real_scalar(yOffsetPupil, 'yOffsetPupil', TypeError)
    check.positive_scalar_integer(Z_noll_max, 'Z_noll_max', TypeError)
    if Z_noll_max < 4 and Z_noll_max > 0:
        raise ValueError('Z_noll_max must be >= 4.')
    check.positive_scalar_integer(nErode, 'nErode', TypeError)
    check.positive_scalar_integer(nBin, 'nBin', TypeError)

    # Enlarge (erode) the pupil obscurations to block out bad or large phase
    mask_selem = np.ones((1+2*nErode, 1+2*nErode))

    # 2. make a pupil mask from the amplitude of the second phase retrieval
    bMask = ampthresh(amp1, nBin=nBin)
    if mask_selem is not None:
        bMask = binary_erosion(bMask, structure=mask_selem)

    # 3. zernike decomposition, return Zvec same units as wfe (radians)
    Zvec = fit_zernikes(ph1-ph0, bMask, Z_noll_max, diamPupil, xOffsetPupil,
                        yOffsetPupil)

    # 4. generate residual maps for analysis and convenience
    zern_combo = gen_zernikes(
        np.arange(4, Z_noll_max+1), Zvec[3:], xOffsetPupil, yOffsetPupil,
        diamPupil, nArray=amp1.shape[0])

    wfe_actual = bMask * (ph1 - ph0)

    return Zvec, zern_combo, wfe_actual, bMask


def compute_phase_diff_model(opt_params, fixed_inputs):
    """
    Compute the cost and phase maps used for poke-grid DM registration.

    Parameters
    ----------
    opt_params : list or 1-D array
        List of the parameters to be optimized by scipy.optimize.minimize().
        They must be these DM 5 registration variables in this exact order:
            [ppact_cx, ppact_cy, dx, dy, thact]
    fixed_inputs : dict
        Dictionary of all the other input parameters needed for the
        DM registration optimization that are not changed by the optimizer.
        The dictionary keys that must be included are:
            'ph_diff_meas': 2-D array of the measured phase difference
            'ph_model_zern': 2-D array of the summed, low-order Zernike
                modes to subtract from the model-produced phase. Should be the
                best fit Zernikes to the starting phase model guess.
            'bmask': 2-D boolean array showing which pixels to use in the DM
                registration optimization. Should be an eroded version of the
                original pupil amplitude.
            'dh_map': 2-D nact x nact array of actuator heights in radians.
                This must already have any DM actuation limitations (e.g.,
                crosstalk, dead actuators, neighbor rule) applied because it
                is handed directly to dmhtoph.dmhtoph() in this function.
            'inf_func': 2-D array of the influence function for the chosen DM
            'ppact_d': Number of pixels per actuator for the given influence
                function.
            'nact': Number of actuators across both axes of the DM grid.
            'flipx': bool, whether to flip the DM surface map generated
                left-right.

    Returns
    -------
    ph_diff_model : numpy.ndarray
        2-D array of the model-based phase difference computed with dmhtoph().
    ph_diff_meas : numpy.ndarray
        2-D array of the measured phase difference. Same as the input of the
        same name except for being padded or cropped if necessary.
    bmask : numpy.ndarray
        2-D boolean array showing which pixels were used in the DM
            registration optimization. Should be an eroded version of the
            original pupil amplitude. Same as the input of the
            same name except for being padded or cropped if necessary.
    cost : float
        The RSS of the difference between ph_diff_meas and ph_diff_model for
        the True pixels in the boolean map bmask. This is used as the cost
        to be minimized by scipy.optimize.minimize().

    """
    # Unpack and check the input list of optimization variables
    opt_params = check.oneD_array(opt_params, 'opt_params', TypeError)
    if len(opt_params) != 5:
        raise ValueError('opt_params must contain five values.')
    ppact_cx, ppact_cy, dx, dy, thact = opt_params
    check.real_positive_scalar(ppact_cx, 'ppact_cx', TypeError)
    check.real_positive_scalar(ppact_cy, 'ppact_cy', TypeError)
    check.real_scalar(dx, 'dx', TypeError)
    check.real_scalar(dy, 'dy', TypeError)
    check.real_scalar(thact, 'thact', TypeError)
    
    # Check the values in the fixed_inputs dictionary
    check.dictionary(fixed_inputs, 'fixed_inputs', TypeError)
    check.twoD_array(fixed_inputs['ph_diff_meas'],
                     "fixed_inputs['ph_diff_meas']",
                     TypeError)
    check.twoD_array(fixed_inputs['ph_model_zern'],
                     "fixed_inputs['ph_model_zern']",
                     TypeError)
    check.twoD_array(fixed_inputs['bmask'],
                     "fixed_inputs['bmask']",
                     TypeError)
    check.twoD_array(fixed_inputs['dh_map'],
                     "fixed_inputs['dh_map']",
                     TypeError)

    # Make sure array shape is consistent for all the phase maps
    ph_diff_meas = fixed_inputs['ph_diff_meas']
    ph_shape = ph_diff_meas.shape
    ph_model_zern = inin(fixed_inputs['ph_model_zern'], ph_shape)
    bmask = inin(fixed_inputs['bmask'], ph_shape)

    # Compute the modeled phase change from the poke grid.
    ph_diff_model = 2 * dmhtoph(
        ph_shape[0], ph_shape[1], fixed_inputs['dh_map'], fixed_inputs['nact'],
        fixed_inputs['inf_func'], fixed_inputs['ppact_d'], ppact_cx,
        ppact_cy, dx, dy, thact, fixed_inputs['flipx'])

    # Subtract off all the same Zernike modes so the oversubtraction
    # residuals match and give a differenced background closer to zero.
    ph_diff_model = (ph_diff_model - ph_model_zern) * bmask

    # Define the cost function to minimize with scipy.optimize.minimize(). 
    cost = np.sum((ph_diff_meas - ph_diff_model)[bmask]**2)

    return ph_diff_model, ph_diff_meas, bmask, cost


def cost_func_dmreg(opt_params, fixed_inputs):
    """
    Return only the cost value for scipy.optimize.minimize().
    
    This is a thin wrapper for dr_util.compute_phase_diff_model() because
    scipy.optimize.minimize() requires the given function to return only
    the cost function value but we want some other diagnostic outputs from
    compute_phase_diff_model() as well.
    """
    _, _, _, cost = compute_phase_diff_model(opt_params, fixed_inputs)

    return cost
