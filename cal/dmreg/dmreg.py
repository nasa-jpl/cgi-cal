# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Module to hold methods for the DMREG package."""
import os
import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_erosion
from scipy.optimize import minimize
import warnings

from cal.ampthresh.ampthresh import ampthresh
from cal.conjugate.util.propcustom import fit_zernikes, gen_zernikes
from cal.dmreg import dr_util
import cal.pupilfit.pupilfit_open as pfo
from cal.pupilzernikes.pupilzernikes import pupilzernikes
from cal.util import check, remove_ptt
from cal.util.dmhtoph import dmhtoph, volts_to_dmh
from cal.util.findopt import find_optimum_1d, find_optimum_2d
from cal.util.insertinto import insertinto as inin
from cal.util.loadyaml import loadyaml

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
PFO_PATH = os.path.dirname(os.path.abspath(pfo.__file__))


def calc_dmreg_from_poke_grid_for_any_pupil(
        amp_unpoked, ph_diff_meas, delta_v, which_dm, lam, fn_dmreg_def,
        fn_dmreg_tuning_params, diamPupil, xOffsetPupil, yOffsetPupil,
    ):
    """
    Compute the DM registration by fitting a poked grid of actuators.

    This function can use any input pupil, such as an apodizer.

    This is also be used to compute the pupil-to-DM1 offset to meet
    REQ 822909: Given 1) a phase retrieval with no masks present and 2)
    a phase retrieval with no masks present but with a reference pattern
    overlaid on the DM setting in DM1, the CTC GSW shall compute the
    offset of the OTA pupil from the center of the DM1 in EXCAM pixels
    in pupil imaging mode to within 0.1% of the pupil diameter in
    EXCAM pixels.

    Parameters
    ----------
    amp_unpoked : array_like
        2-D array from phase retrieval of the retrieved pupil amplitude.
    ph_diff_meas : array_like
        2-D array of difference of the poked phase map minus the unpoked
        phase map. Poked means that the DM registration pattern such as a
        sparse grid of actuators has been actuated.
    delta_v : array_like
        2-D, nact x nact array of delta voltages corresponding to the
        difference between the poked and unpoked voltage maps used when
        generating ph_diff_meas.
    which_dm : int
        Which DM to use. Choices are 1 or 2.
    lam : float
        Central wavelength for the phase retrievals used.
    fn_dmreg_def : str
        YAML file containing the DM definition information and initial
        guess for the DM registration.
    fn_dmreg_tuning_params : str
        YAML file of tuning parameters specific to computing DM registration
        with this function.
    diamPupil : float
        Diameter of the unmasked beam in the phase retrievals. Units of pixels.
    xOffsetPupil, yOffsetPupil : float
        Lateral offsets of the unmasked pupil beam relative to the array
        containing the phase retrieval. These are used as the starting guesses
        for the DM registration lateral offsets. Units of pixels.

    Returns
    -------
    param_dict : dict
        Dictionary of the best fit DM registration parameters optimized by
        scipy.optimize.minimize(). The five parameters are:
            ppact_cx, ppact_cy, dx, dy, thact
    map_dict : dict
        Dictionary containing the all the outputs (output maps and cost) from
        dr_util.compute_phase_diff_model() as well as the pupil-to-array and
        DM-to-pupil offsets. The dictionary keys of map_dict
        have the following names:
            ph_diff_model, ph_diff_meas, bmask, cost, xOffsetPupil,
            yOffsetPupil, xOffsetPupilFromDM, yOffsetPupilFromDM
        xOffsetPupil and yOffsetPupil are the offsets of the pupil from the
        array center pixel and are directly output by pupilzernikes().
        xOffsetPupilFromDM and yOffsetPupilFromDM are computed by differencing
        the DM-to-array offset from the pupil-to-array offsets.

    """
    # Check inputs
    check.twoD_array(amp_unpoked, 'amp_unpoked', TypeError)
    check.twoD_array(ph_diff_meas, 'ph_diff_meas', TypeError)
    check.twoD_array(delta_v, 'delta_v', TypeError)
    check.positive_scalar_integer(which_dm, 'which_dm', TypeError)
    if which_dm not in (1, 2):
        raise TypeError('which_dm must be 1 or 2')
    check.real_positive_scalar(lam, 'lam', TypeError)
    if not isinstance(fn_dmreg_def, str):
        raise TypeError('fn_dmreg_def must be a string')
    if not isinstance(fn_dmreg_tuning_params, str):
        raise TypeError('fn_dmreg_tuning_params must be a string')
    check.real_positive_scalar(diamPupil, 'diamPupil', TypeError)
    check.real_scalar(xOffsetPupil, 'xOffsetPupil', TypeError)
    check.real_scalar(yOffsetPupil, 'yOffsetPupil', TypeError)

    # Unpack and check tuning parameters
    tuning_dict = loadyaml(fn_dmreg_tuning_params)
    opt_method = tuning_dict['opt_method']
    nErode = tuning_dict['nErode']
    Z_noll_max = tuning_dict['Z_noll_max']
    ppact_tol_pct = tuning_dict['ppact_tol_pct']
    dxdy_tol_pix = tuning_dict['dxdy_tol_pix']
    thact_tol_deg = tuning_dict['thact_tol_deg']
    nBin = tuning_dict['nBin']

    # Enlarge (erode) the pupil obscurations to block out bad or large phase
    mask_selem = np.ones((1+2*nErode, 1+2*nErode))

    # Make a pupil mask from the amplitude of the phase retrieval
    bmask = ampthresh(amp_unpoked, nBin=nBin)
    if mask_selem is not None:
        bmask = binary_erosion(bmask, structure=mask_selem)

    # Zernike decomposition, return Zvec same units as wfe (radians)
    ZvecMeas = fit_zernikes(ph_diff_meas, bmask, Z_noll_max, diamPupil,
                            xOffsetPupil, yOffsetPupil)

    # Subtract off the first Z_noll_max Zernikes from the eroded pupil phase
    # because phase retrieval repeatability errors are often low-order Zernike
    # modes and to get rid of any piston/tip/tilt.
    ph_diff_meas -= gen_zernikes(
        np.arange(1, 1+Z_noll_max), ZvecMeas, xOffsetPupil, yOffsetPupil,
        diamPupil, nArray=ph_diff_meas.shape[0])
    ph_diff_meas *= bmask

    # Load a bunch of DM definition info needed for the model and for some
    # starting values for the optimization.
    path_to_dmreg_yaml = os.path.dirname(os.path.expandvars(fn_dmreg_def))
    dmConfig = loadyaml(fn_dmreg_def)
    dm = dmConfig["dms"]

    if which_dm == 1:
        dmr = dm['DM1']['registration']
        gainmap = fits.getdata(os.path.join(
            path_to_dmreg_yaml,
            os.path.expandvars(dm['DM1']['voltages']['gainfn']),
        ))
        fn_crosstalk_base = dm['DM1']['voltages']['crosstalkfn']
    elif which_dm == 2:
        dmr = dm['DM2']['registration']
        gainmap = fits.getdata(os.path.join(
            path_to_dmreg_yaml,
            os.path.expandvars(dm['DM2']['voltages']['gainfn']),
        ))
        fn_crosstalk_base = dm['DM2']['voltages']['crosstalkfn']
    else:
        raise ValueError('which_dm must be either 1 or 2')

    # Cannot give None to os.path.join()
    if fn_crosstalk_base is not None:
        crosstalk_fn = os.path.join(path_to_dmreg_yaml,
                                    os.path.expandvars(fn_crosstalk_base),
                                    )
    else:
        crosstalk_fn = None

    # Convert voltage map to height map and account for crosstalk
    dh_map = volts_to_dmh(gainmap, delta_v, lam, crosstalk_fn=crosstalk_fn)

    # Gather starting values for the optimization. dx and dy are based on the
    # measured starting pupil position instead of the dmreg definition YAML.
    ppact_cx0 = dmr['ppact_cx']
    ppact_cy0 = dmr['ppact_cy']
    dx0 = xOffsetPupil
    dy0 = yOffsetPupil
    thact0 = dmr['thact']
    initial_guess = [ppact_cx0, ppact_cy0, dx0, dy0, thact0]

    # Define the bounds for the optimization variables.
    bounds = [
        [(1-ppact_tol_pct/100)*ppact_cx0, (1+ppact_tol_pct/100)*ppact_cx0],
        [(1-ppact_tol_pct/100)*ppact_cy0, (1+ppact_tol_pct/100)*ppact_cy0],
        [dx0-dxdy_tol_pix, dx0+dxdy_tol_pix],
        [dy0-dxdy_tol_pix, dy0+dxdy_tol_pix],
        [thact0-thact_tol_deg, thact0+thact_tol_deg],
    ]

    # Load more DM data from the DM definition file.
    ppact_d = dmr['ppact_d']
    nact = dmr['nact']
    flipx = dmr['flipx']
    inf_func = fits.getdata(os.path.join(path_to_dmreg_yaml,
                                         os.path.expandvars(dmr['inffn']),
                                         ))

    # Subtract off all the same modes so the oversubtraction residuals will be
    # the same. This turns out to change the answer by << 0.1% and could be
    # ignored, but it makes the plots look nicer by subtracting out a lot of
    # the piston and possibly other low-orders.
    ph_diff_model = 2 * dmhtoph(
        ph_diff_meas.shape[0], ph_diff_meas.shape[1], dh_map, nact,
        inf_func, ppact_d, ppact_cx0, ppact_cy0, dx0, dy0, thact0, flipx)
    # Zernike decomposition, return Zvec same units as wfe (radians)
    ZvecModel = fit_zernikes(ph_diff_model, bmask, Z_noll_max, diamPupil,
                             xOffsetPupil, yOffsetPupil)
    ph_model_zern = bmask * gen_zernikes(
        np.arange(1, 1+Z_noll_max), ZvecModel, xOffsetPupil, yOffsetPupil,
        diamPupil, nArray=ph_diff_model.shape[0])

    fixed_inputs = {
        'ph_diff_meas': ph_diff_meas,
        'ph_model_zern': ph_model_zern,
        'bmask': bmask,
        'dh_map': dh_map,
        'inf_func': inf_func,
        'ppact_d': ppact_d,
        'nact': nact,
        'flipx': flipx,
        }

    # Run the optimization
    result = minimize(dr_util.cost_func_dmreg,
                      initial_guess,
                      args=fixed_inputs,
                      bounds=bounds,
                      method=opt_method,
                      )
    if result.success:
        fitted_params = result.x
    else:
        raise ValueError(result.message)

    param_dict = {
        'ppact_cx': float(fitted_params[0]),
        'ppact_cy': float(fitted_params[1]),
        'dx': float(fitted_params[2]),
        'dy': float(fitted_params[3]),
        'thact': float(fitted_params[4]),
    }

    # Extra outputs for verifying quality of the fit such as with plots.
    out_maps = dr_util.compute_phase_diff_model(fitted_params, fixed_inputs)
    map_dict = {
        'ph_diff_model': out_maps[0],
        'ph_diff_meas': out_maps[1],
        'bmask': out_maps[2],
        'cost': out_maps[3],
        'xOffsetPupil': xOffsetPupil,
        'yOffsetPupil': yOffsetPupil,
        'xOffsetPupilFromDM': xOffsetPupil - param_dict['dx'],
        'yOffsetPupilFromDM': yOffsetPupil - param_dict['dy'],
    }

    return param_dict, map_dict


def calc_dmreg_from_poke_grid(
        amp_unpoked, ph_diff_meas, delta_v, which_dm, lam, fn_dmreg_def,
        fn_dmreg_tuning_params, fn_pupil_fit_params, data_path_pupil,
    ):
    """
    Compute the DM registration by fitting a poked grid of actuators.

    This function only works when the pupil is unmasked.

    This is also be used to compute the pupil-to-DM1 offset to meet
    REQ 822909: Given 1) a phase retrieval with no masks present and 2)
    a phase retrieval with no masks present but with a reference pattern
    overlaid on the DM setting in DM1, the CTC GSW shall compute the
    offset of the OTA pupil from the center of the DM1 in EXCAM pixels
    in pupil imaging mode to within 0.1% of the pupil diameter in
    EXCAM pixels.

    Parameters
    ----------
    amp_unpoked : array_like
        2-D array from phase retrieval of the retrieved pupil amplitude.
    ph_diff_meas : array_like
        2-D array of difference of the poked phase map minus the unpoked
        phase map. Poked means that the DM registration pattern such as a
        sparse grid of actuators has been actuated.
    delta_v : array_like
        2-D, nact x nact array of delta voltages corresponding to the
        difference between the poked and unpoked voltage maps used when
        generating ph_diff_meas.
    which_dm : int
        Which DM to use. Choices are 1 or 2.
    lam : float
        Central wavelength for the phase retrievals used.
    fn_dmreg_def : str
        YAML file containing the DM definition information and initial
        guess for the DM registration.
    fn_dmreg_tuning_params : str
        YAML file of tuning parameters specific to computing DM registration
        with this function.
    fn_pupil_fit_params : str
        YAML file of tuning parameters handed (via pupilzernikes()) to
        cal.pupilfit.pupilfit_open.fit_unmasked_pupil() for fitting the
        unmasked pupil in the provided pupil amplitude array.
    data_path_pupil : str
        Full path to the pupil reference file named in fn_pupil_fit_params.

    Returns
    -------
    param_dict : dict
        Dictionary of the best fit DM registration parameters optimized by
        scipy.optimize.minimize(). The five parameters are:
            ppact_cx, ppact_cy, dx, dy, thact
    map_dict : dict
        Dictionary containing the all the outputs (output maps and cost) from
        dr_util.compute_phase_diff_model() as well as the pupil-to-array and
        DM-to-pupil offsets. The dictionary keys of map_dict
        have the following names:
            ph_diff_model, ph_diff_meas, bmask, cost, xOffsetPupil,
            yOffsetPupil, xOffsetPupilFromDM, yOffsetPupilFromDM
        xOffsetPupil and yOffsetPupil are the offsets of the pupil from the
        array center pixel and are directly output by pupilzernikes().
        xOffsetPupilFromDM and yOffsetPupilFromDM are computed by differencing
        the DM-to-array offset from the pupil-to-array offsets.

    """
    # Check inputs
    check.twoD_array(amp_unpoked, 'amp_unpoked', TypeError)
    check.twoD_array(ph_diff_meas, 'ph_diff_meas', TypeError)
    check.twoD_array(delta_v, 'delta_v', TypeError)
    check.positive_scalar_integer(which_dm, 'which_dm', TypeError)
    if which_dm not in (1, 2):
        raise TypeError('which_dm must be 1 or 2')
    check.real_positive_scalar(lam, 'lam', TypeError)
    if not isinstance(fn_dmreg_def, str):
        raise TypeError('fn_dmreg_def must be a string')
    if not isinstance(fn_dmreg_tuning_params, str):
        raise TypeError('fn_dmreg_tuning_params must be a string')
    if not isinstance(fn_pupil_fit_params, str):
        raise TypeError('fn_pupil_fit_params must be a string')
    if not isinstance(data_path_pupil, str):
        raise TypeError('data_path_pupil must be a string')

    # Unpack and check tuning parameters
    tuning_dict = loadyaml(fn_dmreg_tuning_params)
    opt_method = tuning_dict['opt_method']
    nErode = tuning_dict['nErode']
    Z_noll_max = tuning_dict['Z_noll_max']
    ppact_tol_pct = tuning_dict['ppact_tol_pct']
    dxdy_tol_pix = tuning_dict['dxdy_tol_pix']
    thact_tol_deg = tuning_dict['thact_tol_deg']
    nBin = tuning_dict['nBin']

    # Enlarge (erode) the pupil obscurations to block out bad or large phase
    mask_selem = np.ones((1+2*nErode, 1+2*nErode))

    # Subtract off the first Z_noll_max Zernikes from the eroded pupil
    # because phase retrieval repeatability errors are often low-order Zernike
    # modes.
    _, out_dict = pupilzernikes(
        ph_diff_meas, amp_unpoked, fn_pupil_fit_params, bMask=None,
        Z_noll_max=Z_noll_max, mask_selem=mask_selem, nBin=nBin,
        data_path=data_path_pupil)
    bmask = out_dict['bMask']
    xOffsetPupil = out_dict['xOffset']
    yOffsetPupil = out_dict['yOffset']
    # Keep the same name, but ph_diff_meas now has the Zernikes subtracted out.
    ph_diff_meas = out_dict['wfe_res'] * bmask

    # Load a bunch of DM definition info needed for the model and for some
    # starting values for the optimization.
    path_to_dmreg_yaml = os.path.dirname(os.path.expandvars(fn_dmreg_def))
    dmConfig = loadyaml(fn_dmreg_def)
    dm = dmConfig["dms"]

    if which_dm == 1:
        dmr = dm['DM1']['registration']
        gainmap = fits.getdata(os.path.join(path_to_dmreg_yaml,
                      os.path.expandvars(dm['DM1']['voltages']['gainfn'])))
        fn_crosstalk_base = dm['DM1']['voltages']['crosstalkfn']
    elif which_dm == 2:
        dmr = dm['DM2']['registration']
        gainmap = fits.getdata(os.path.join(path_to_dmreg_yaml,
                      os.path.expandvars(dm['DM2']['voltages']['gainfn'])))
        fn_crosstalk_base = dm['DM2']['voltages']['crosstalkfn']
    else:
        raise ValueError('which_dm must be either 1 or 2')

    # Cannot give None to os.path.join()
    if fn_crosstalk_base is not None:
        crosstalk_fn = os.path.join(path_to_dmreg_yaml,
                                    os.path.expandvars(fn_crosstalk_base),
                                    )
    else:
        crosstalk_fn = None

    # Convert voltage map to height map and account for crosstalk
    dh_map = volts_to_dmh(gainmap, delta_v, lam, crosstalk_fn=crosstalk_fn)

    # Gather starting values for the optimization. dx and dy are based on the
    # measured starting pupil position instead of the dmreg definition YAML.
    ppact_cx0 = dmr['ppact_cx']
    ppact_cy0 = dmr['ppact_cy']
    dx0 = xOffsetPupil
    dy0 = yOffsetPupil
    thact0 = dmr['thact']
    initial_guess = [ppact_cx0, ppact_cy0, dx0, dy0, thact0]

    # Define the bounds for the optimization variables.
    bounds = [
        [(1-ppact_tol_pct/100)*ppact_cx0, (1+ppact_tol_pct/100)*ppact_cx0],
        [(1-ppact_tol_pct/100)*ppact_cy0, (1+ppact_tol_pct/100)*ppact_cy0],
        [dx0-dxdy_tol_pix, dx0+dxdy_tol_pix],
        [dy0-dxdy_tol_pix, dy0+dxdy_tol_pix],
        [thact0-thact_tol_deg, thact0+thact_tol_deg],
    ]

    # Load more DM data from the DM definition file.
    ppact_d = dmr['ppact_d']
    nact = dmr['nact']
    flipx = dmr['flipx']
    inf_func = fits.getdata(os.path.join(path_to_dmreg_yaml,
                                         os.path.expandvars(dmr['inffn']),
                                         ))

    # Subtract off all the same modes so the oversubtraction residuals will be
    # the same. This turns out to change the answer by << 0.1% and could be
    # ignored, but it makes the plots look nicer by subtracting out a lot of
    # the piston and possibly other low-orders.
    ph_diff_model = 2 * dmhtoph(
        ph_diff_meas.shape[0], ph_diff_meas.shape[1], dh_map, nact,
        inf_func, ppact_d, ppact_cx0, ppact_cy0, dx0, dy0, thact0, flipx)
    Zvec, out_dict = pupilzernikes(
        ph_diff_model, amp_unpoked, fn_pupil_fit_params, bMask=bmask,
        Z_noll_max=Z_noll_max, nBin=nBin, data_path=data_path_pupil)
    ph_diff_model = out_dict['wfe_res'] * bmask
    ph_model_zern = bmask * gen_zernikes(
        np.arange(1, 1+Z_noll_max), Zvec, out_dict['xOffset'],
        out_dict['yOffset'], out_dict['diamEst'],
        nArray=ph_diff_model.shape[0])

    fixed_inputs = {
        'ph_diff_meas': ph_diff_meas,
        'ph_model_zern': ph_model_zern,
        'bmask': bmask,
        'dh_map': dh_map,
        'inf_func': inf_func,
        'ppact_d': ppact_d,
        'nact': nact,
        'flipx': flipx,
        }

    # Run the optimization
    result = minimize(dr_util.cost_func_dmreg,
                      initial_guess,
                      args=fixed_inputs,
                      bounds=bounds,
                      method=opt_method,
                      )
    if result.success:
        fitted_params = result.x
    else:
        raise ValueError(result.message)

    param_dict = {
        'ppact_cx': float(fitted_params[0]),
        'ppact_cy': float(fitted_params[1]),
        'dx': float(fitted_params[2]),
        'dy': float(fitted_params[3]),
        'thact': float(fitted_params[4]),
    }

    # Extra outputs for verifying quality of the fit such as with plots.
    out_maps = dr_util.compute_phase_diff_model(fitted_params, fixed_inputs)
    map_dict = {
        'ph_diff_model': out_maps[0],
        'ph_diff_meas': out_maps[1],
        'bmask': out_maps[2],
        'cost': out_maps[3],
        'xOffsetPupil': xOffsetPupil,
        'yOffsetPupil': yOffsetPupil,
        'xOffsetPupilFromDM': xOffsetPupil - param_dict['dx'],
        'yOffsetPupilFromDM': yOffsetPupil - param_dict['dy'],
    }

    return param_dict, map_dict


def calc_software_mask(amp0, ph0, amp1, ph1, whichDM, nErode,
                       deltaPhaseThresh):
    """
    Compute which subset of pixels to use during DM registration.

    For DM registration, first compute which pixels to use later when computing
    translation, rotation, and scale. Using AMPTHRESH alone is not sufficient
    because that doesn't take into account dead actuators. This function
    expects an actuated piston between the two different phase retrievals, and
    throws out pixels where the phase exceeds some specified variation from
    that piston value (either because there is a non-moving actuator there, or
    the phase estimate was unreliable there). There is also an option to erode
    (pad) the pupil obscurations to eliminate any bad phase estimates at the
    edges of the pupil obscurations; this also enlarges the masked areas for
    dead actuators.

    Parameters
    ----------
    amp0 : array_like
        2-D amplitude map of the initial DM state.
    ph0 : array_like
        2-D unwrapped phase map of the initial DM state.
    amp1 : array_like
        2-D amplitude map of the perturbed DM state.
    ph1 : array_like
        2-D unwrapped phase map of the perturbed DM state.
    whichDM : {1, 2}
        Which DM to register. Can be 1 or 2 only.
    nErode : int
        Number of pixels to "erode" the AMPTHRESH'ed aperture by.
        Eroding means increasing the size of the obscurations.
        Units of ExCAM pixels.
    deltaPhaseThresh : float
        Minimum deviation from the median phase at which to threshold when
        making the usable pixel map. Must be positive. Units of radians.

    Returns
    -------
    usablePixelMap : array_like
        2-D boolean map of which pixels to use when fitting DM shapes for DM
        registration. Pixels near pupil obscurations and pixels at dead
        actuators are not included.
    """
    check.twoD_array(amp0, 'amp0', TypeError)
    check.twoD_array(ph0, 'ph0', TypeError)
    check.twoD_array(amp1, 'amp1', TypeError)
    check.twoD_array(ph1, 'ph1', TypeError)
    if amp0.shape != ph0.shape:
        raise TypeError('amp0 and ph0 must have same shape')
    if amp0.shape != amp1.shape:
        raise TypeError('amp0 and amp1 must have same shape')
    if ph0.shape != ph1.shape:
        raise TypeError('ph0 and ph1 must have same shape')
    check.positive_scalar_integer(whichDM, 'whichDM', TypeError)
    if whichDM not in (1, 2):
        raise ValueError('whichDM must have value of 1 or 2')
    check.nonnegative_scalar_integer(nErode, 'nErode', TypeError)
    check.real_positive_scalar(deltaPhaseThresh, 'deltaPhaseThresh',
                               TypeError)

    phDiff = ph1 - ph0

    # Remove piston, tip, and tilt
    usablePixelMap = ampthresh(amp0)
    phDiff = remove_ptt.remove_ptt_directly(phDiff, usablePixelMap)

    # Threshold to get rid of large phase deviations near the pupil obscuration
    # edges or at dead DM actuators.
    usablePixelMap[np.abs(phDiff) > deltaPhaseThresh] = False

    # Enlarge (erode) the pupil obscurations to block out bad or large phase
    # values near the edges
    usablePixelMap = binary_erosion(
        usablePixelMap, structure=np.ones((1+2*nErode, 1+2*nErode)))

    return usablePixelMap


def calc_translation_clocking(
        amp0, ph0, amp1, ph1, usablePixelMap, whichDM, deltaV,
        xOffsetPupil, yOffsetPupil, dm_fn, fn_fit_params, lam,
        data_path=LOCAL_PATH):
    """
    Compute the rotation and lateral offsets of the DM grid.

    Compute the rotation of the DM grid in degrees counterclockwise relative to
    the EXCAM x- and y-axes, and compute the translation of the DM grid from
    the center pixel of the array in EXCAM pixels, in both EXCAM x- and y-axes.

    Return np.nan's if the search runs into an exception.

    The recommended delta DM command for this function is a pair of uniformly
    actuated rows and a pair of uniformly actuated columns symmetric about the
    center of the DM. A simulation-tested example of this is in
    `Calibration/dmreg/cgidata/delta_DM_V_for_trans_rot_calib.fits`, but note
    that this command map should be adjusted to counteract non-uniform actuator
    gains. The symmetry is desired to make the DM shape insensitive to parity
    flips or 90-, 180-, and 270-degree rotations. The two actuated rows/columns
    are not neighboring because the Roman pupil struts would mostly block them
    along one dimension, and because the fitting seems to do better with
    actuated lines that are one (instead of two) actuators wide. The smoothness
    of the actuated lines seems to help as well--diagonal actuated lines are
    bumpier and give lower accuracy estimates.

    Other patterns may be used--no exception will be thrown--but the
    performance of this function has not been tested with them and cannot be
    guaranteed.

    For this function: all filenames must be relative paths.  This is
    different from most other functions in Calibration, and is true for
    both input arguments and filenames in YAML files.  They will be relative
    to the value supplied in data_path, which is the location of this module
    by default unless specified otherwise.

    Parameters
    ----------
    amp0 : array_like
        2-D amplitude map of the initial DM state.
    ph0 : array_like
        2-D unwrapped phase map of the initial DM state.
    amp1 : array_like
        2-D amplitude map of the perturbed DM state.
    ph1 : array_like
        2-D unwrapped phase map of the perturbed DM state.
    usablePixelMap : array_like
        2-D boolean array indicating which image pixels to use for DM
        registration.
    whichDM : {1, 2}
        Which DM to register. Can be 1 or 2 only.
    deltaV : array_like
        2-D array of delta DM commands for the translation and clocking
        registration. Units of volts.
    xOffsetPupil, yOffsetPupil : float
        Lateral offsets of the pupil relative to the ExCAM image. Units of
        ExCAM pixels.
    dm_fn : str
        Name of the YAML file containing pre-registration calibration data for
        DMs 1 and 2. The gainmap and influence function should be known ahead
        of DM registration. To make a model-based DM surface for comparison,
        approximate, a priori scaling values need to be in this file.
    fn_fit_params : str
        Name of the YAML file with the tuning parameters for translation and
        rotation.
    lam : float
        Center wavelength for the phase retrieval maps. Units of meters
    data_path : str
        Directory to serve as a base for relative paths with YAML files.
        If not supplied, defaults to directory containing this function. If
        using versions of the YAMLs delivered with the repository, the default
        will point to data files also delivered with the repository.


    Returns
    -------
    xOffsetEst, yOffsetEst : float
        Estimated x- and y-offsets of the DM grid center from the center pixel
        in the ExCAM image.
    clockEst : float
        Clocking estimate of the DM grid relative to the ExCAM image.
        Units of degrees.
    """
    check.twoD_array(amp0, 'amp0', TypeError)
    check.twoD_array(ph0, 'ph0', TypeError)
    check.twoD_array(amp1, 'amp1', TypeError)
    check.twoD_array(ph1, 'ph1', TypeError)
    check.twoD_array(usablePixelMap, 'usablePixelMap', TypeError)
    if amp0.shape != ph0.shape:
        raise TypeError('amp0 and ph0 must have same shape')
    if amp0.shape != amp1.shape:
        raise TypeError('amp0 and amp1 must have same shape')
    if amp0.shape != ph1.shape:
        raise TypeError('amp0 and ph1 must have same shape')
    if amp0.shape != usablePixelMap.shape:
        raise TypeError('amp0 and usablePixelMap must have same shape')
    check.positive_scalar_integer(whichDM, 'whichDM', TypeError)
    if whichDM not in (1, 2):
        raise ValueError('whichDM must have value of 1 or 2')
    check.real_scalar(xOffsetPupil, 'xOffsetPupil', TypeError)
    check.real_scalar(yOffsetPupil, 'yOffsetPupil', TypeError)
    check.twoD_square_array(deltaV, 'deltaV', TypeError)
    check.real_positive_scalar(lam, 'lam', TypeError)

    data_path = os.path.expandvars(data_path)

    # Load Fitting Parameter Values
    fn_fit_params_fullpath = os.path.join(data_path, fn_fit_params)
    fitParamDict = loadyaml(fn_fit_params_fullpath)
    nIterBoth = fitParamDict['nIterBoth']
    offsetMax = fitParamDict['offsetMax']
    nOffset = fitParamDict['nOffset']
    nIterOffset = fitParamDict['nIterOffset']
    shrinkFac = fitParamDict['shrinkFac']
    clockMaxDeg = fitParamDict['clockMaxDeg']
    nClock = fitParamDict['nClock']
    nIterClock = fitParamDict['nIterClock']
    # Check Fitting Parameters
    check.positive_scalar_integer(nIterBoth, 'nIterBoth', TypeError)
    check.real_scalar(offsetMax, 'offsetMax', TypeError)
    check.positive_scalar_integer(nOffset, 'nOffset', TypeError)
    check.positive_scalar_integer(nIterOffset, 'nIterOffset', TypeError)
    check.real_positive_scalar(shrinkFac, 'shrinkFac', TypeError)
    if shrinkFac > 1:
        raise TypeError('shrinkFac must be less than or equal to 1.')
    check.real_scalar(clockMaxDeg, 'clockMaxDeg', TypeError)
    check.positive_scalar_integer(nClock, 'nClock', TypeError)
    check.positive_scalar_integer(nIterClock, 'nIterClock', TypeError)

    # Load Initial DM Calibration Data
    dm_fn_fullpath = os.path.join(data_path, dm_fn)
    dmDict = loadyaml(dm_fn_fullpath)
    if whichDM == 1:
        dm = dmDict["dms"]["DM1"]
    elif whichDM == 2:
        dm = dmDict["dms"]["DM2"]
    dmr = dm['registration']
    flipx = dmr['flipx']  # DM pattern for trans and rot is flipx agnostic
    gainfn = os.path.join(data_path,
                          os.path.expandvars(dm['voltages']['gainfn']))
    gainmap = fits.getdata(gainfn)
    inffn = os.path.join(data_path,
                         os.path.expandvars(dmr['inffn']))
    inf_func = fits.getdata(inffn)
    dmrad = volts_to_dmh(gainmap, deltaV, lam)

    ph_diff_meas = ph1 - ph0
    ph_diff_meas = np.angle(np.exp(1j*ph_diff_meas))  # eliminate phase wrapping
    ph_diff_meas = remove_ptt.remove_ptt_directly(ph_diff_meas, usablePixelMap)

    # Initializations for the nested loops
    xOffsetVec = np.linspace(-offsetMax, offsetMax, nOffset)
    yOffsetVec = np.linspace(-offsetMax, offsetMax, nOffset)
    costMat = np.zeros((nOffset, nOffset))
    xOffsetEst = xOffsetPupil
    yOffsetEst = yOffsetPupil
    clockEst = dmr['thact']
    [nrow, ncol] = amp0.shape

    # Loop over translation and clocking
    for iterBoth in range(nIterBoth):

        try:
            # Fit Translation
            for iterOffset in range(nIterOffset):
                for iy, yOffsetTemp in enumerate(yOffsetVec):
                    for ix, xOffsetTemp in enumerate(xOffsetVec):
                        phDiffSim = dmhtoph(nrow, ncol, dmrad, dmr['nact'],
                                            inf_func, dmr['ppact_d'],
                                            dmr['ppact_cx'], dmr['ppact_cy'],
                                            xOffsetEst+xOffsetTemp,
                                            yOffsetEst+yOffsetTemp, clockEst,
                                            flipx)
                        phDiffSim *= 2  # surface to wavefront
                        phDiffSim = remove_ptt.remove_ptt_directly(
                            phDiffSim, usablePixelMap)

                        costMat[iy, ix] = np.sum(
                            usablePixelMap * np.abs(ph_diff_meas - phDiffSim))

                # Use argmin the first time only,
                # and then switch to a quadratic fit.
                if iterOffset == 0 and nIterOffset > 1:
                    bestInds = np.unravel_index(np.argmin(costMat),
                                                costMat.shape)
                    xOffsetTemp = xOffsetVec[bestInds[1]]
                    yOffsetTemp = yOffsetVec[bestInds[0]]
                else:
                    [xOffsetTemp, yOffsetTemp] = \
                        find_optimum_2d(xOffsetVec, yOffsetVec, costMat,
                                        np.ones_like(costMat))
                xOffsetEst += xOffsetTemp
                yOffsetEst += yOffsetTemp
                xOffsetVec = (shrinkFac**iterOffset) *\
                    np.linspace(-offsetMax, offsetMax, nOffset)
                yOffsetVec = (shrinkFac**iterOffset) *\
                    np.linspace(-offsetMax, offsetMax, nOffset)

            # Fit Rotation
            clockVec = np.linspace(-clockMaxDeg, clockMaxDeg, nClock)
            costVec = np.zeros((nClock,))
            for iterClock in range(nIterClock):
                for ic, deltaClock in enumerate((shrinkFac**iterClock)*\
                                                clockVec):
                    phDiffSim = dmhtoph(nrow, ncol, dmrad, dmr['nact'],
                                        inf_func,
                                        dmr['ppact_d'],
                                        dmr['ppact_cx'],
                                        dmr['ppact_cy'],
                                        xOffsetEst, yOffsetEst,
                                        deltaClock+clockEst, flipx)
                    costVec[ic] = np.sum(usablePixelMap *
                                         np.abs(ph_diff_meas - phDiffSim))

                # Use argmin the first time only,
                # and then switch to a quadratic fit.
                if iterClock == 0 and nIterClock > 1:
                    bestInd = np.argmin(costVec)
                    clockEst += clockVec[bestInd]
                else:
                    clockEst += find_optimum_1d(clockVec, costVec)

        except:  # Fail gracefully
            warnings.warn('Translation and clocking grid search failed.')
            xOffsetEst = np.nan
            yOffsetEst = np.nan
            clockEst = np.nan

    return xOffsetEst, yOffsetEst, clockEst


def calc_scale(amp0, ph0, amp1, ph1, usablePixelMap, whichDM, deltaV,
               dm_fn, fn_fit_params, lam, xOffset, yOffset, clocking,
               data_path=LOCAL_PATH):
    """
    Compute the scaling of the DM grid in x and y.

    Compute the scaling of the DM grid, in EXCAM pixels per actuator, along
    both x- and y-axes by using parabolic fits. Returns np.nan's if the search
    runs into an exception.

    The recommended delta DM command for the scaling function is a fully
    inscribed within the pupil, two-actuator-wide outline of a square
    centered on the DM. A simulation-tested example of this is in
    `Calibration/dmreg/cgidata/delta_DM_V_for_scale_calib.fits`, but note that
    this command map should be adjusted to counteract non-uniform actuator
    gains. The symmetry is desired to make the DM shape insensitive to parity
    flips or 90-, 180-, and 270-degree rotations. The outline is two actuators
    wide (instead of just one) so that the algorithm will work over a greater
    range of scale factors.

    Other patterns may be used--no exception will be thrown--but the
    performance of this function has not been tested with them and cannot be
    guaranteed.

    For this function: all filenames must be relative paths.  This is
    different from most other functions in Calibration, and is true for
    both input arguments and filenames in YAML files.  They will be relative
    to the value supplied in data_path, which is the location of this module
    by default unless specified otherwise.

    Parameters
    ----------
    amp0 : array_like
        2-D amplitude map of the initial DM state.
    ph0 : array_like
        2-D unwrapped phase map of the initial DM state.
    amp1 : array_like
        2-D amplitude map of the perturbed DM state.
    ph1 : array_like
        2-D unwrapped phase map of the perturbed DM state.
    usablePixelMap : array_like
        2-D boolean array indicating which image pixels to use for DM
        registration.
    whichDM : {1, 2}
        Which DM to register. Can be 1 or 2 only.
    deltaV : array_like
        2-D array of delta DM commands for the scale registration.
        Units of volts.
    dm_fn : str
        Name of the YAML file containing pre-registration calibration data for
        DMs 1 and 2. The gainmap and influence function should be known ahead
        of DM registration. To make a model-based DM surface for comparison,
        approximate, a priori scaling values need to be in this file.
    fn_fit_params : str
        Name of the YAML file with the scale tuning parameters.
    shrinkFac : float
        Factor by which to change maxDeltaScaleFac after each iteration of the
        fit. Must be >0.0 and <= 1.0.
    lam : float
        Center wavelength for the phase retrieval maps. Units of meters.
    xOffset, yOffset : float
        Lateral offsets of the DM grid relative to the ExCAM image center.
        Units of ExCAM pixels.
    clocking : float
        Counter-clockwise rotation of the DM grid with respect to the ExCAM
        pixel array. Units of degrees.
    data_path : str
        Directory to serve as a base for relative paths with YAML files.
        If not supplied, defaults to directory containing this function. If
        using versions of the YAMLs delivered with the repository, the default
        will point to data files also delivered with the repository.

    Returns
    -------
    ppact_cx_est, ppact_cy_est : float
        Estimates of the DM scaling along the x- and y-axes. Units of ExCAM
        pixels per actuator.
    """
    check.twoD_array(amp0, 'amp0', TypeError)
    check.twoD_array(ph0, 'ph0', TypeError)
    check.twoD_array(amp1, 'amp1', TypeError)
    check.twoD_array(ph1, 'ph1', TypeError)
    check.twoD_array(usablePixelMap, 'usablePixelMap', TypeError)
    if amp0.shape != ph0.shape:
        raise TypeError('amp0 and ph0 must have same shape')
    if amp0.shape != amp1.shape:
        raise TypeError('amp0 and amp1 must have same shape')
    if amp0.shape != ph1.shape:
        raise TypeError('amp0 and ph1 must have same shape')
    if amp0.shape != usablePixelMap.shape:
        raise TypeError('amp0 and usablePixelMap must have same shape')
    check.positive_scalar_integer(whichDM, 'whichDM', TypeError)
    if whichDM not in (1, 2):
        raise ValueError('whichDM must have value of 1 or 2')
    check.twoD_square_array(deltaV, 'deltaV', TypeError)
    check.real_positive_scalar(lam, 'lam', TypeError)
    check.real_scalar(xOffset, 'xOffset', TypeError)
    check.real_scalar(yOffset, 'yOffset', TypeError)
    check.real_scalar(clocking, 'clocking', TypeError)
    check.string(data_path, 'data_path', TypeError)

    data_path = os.path.expandvars(data_path)

    # Load fit parameters
    fn_fit_fullpath = os.path.join(data_path, fn_fit_params)
    fitDict = loadyaml(fn_fit_fullpath)
    maxDeltaScaleFac = fitDict['maxDeltaScaleFac']
    nScale = fitDict['nScale']
    nIter = fitDict['nIter']
    shrinkFac = fitDict['shrinkFac']
    check.real_scalar(maxDeltaScaleFac, 'maxDeltaScaleFac', TypeError)
    check.positive_scalar_integer(nScale, 'nScale', TypeError)
    check.positive_scalar_integer(nIter, 'nIter', TypeError)
    check.real_positive_scalar(shrinkFac, 'shrinkFac', TypeError)
    if shrinkFac > 1:
        raise TypeError('shrinkFac must be less than or equal to 1.')

    # Load DM Calibration Data
    dm_fn_fullpath = os.path.join(data_path, dm_fn)
    dmConfig = loadyaml(dm_fn_fullpath)
    if whichDM == 1:
        dm = dmConfig["dms"]["DM1"]
    elif whichDM == 2:
        dm = dmConfig["dms"]["DM2"]
    dmr = dm['registration']
    gainfn = os.path.join(data_path,
                          os.path.expandvars(dm['voltages']['gainfn']))
    gainmap = fits.getdata(gainfn)
    inffn = os.path.join(data_path,
                         os.path.expandvars(dmr['inffn']))
    inf_func = fits.getdata(inffn)
    dmrad = volts_to_dmh(gainmap, deltaV, lam)
    flipx = False  # DM registration pattern is flipx agnostic

    ph_diff_meas = ph1 - ph0
    ph_diff_meas = np.angle(np.exp(1j*ph_diff_meas))  # eliminate phase wrapping
    ph_diff_meas = remove_ptt.remove_ptt_directly(ph_diff_meas, usablePixelMap)

    xScaleVec = np.linspace(1-maxDeltaScaleFac, 1+maxDeltaScaleFac, nScale)
    yScaleVec = np.linspace(1-maxDeltaScaleFac, 1+maxDeltaScaleFac, nScale)
    costMat = np.zeros((nScale, nScale))
    xScale = 1  # initialize
    yScale = 1  # initialize
    try:
        for iter_ in range(nIter):
            for iy, yScaleTemp in enumerate(yScaleVec):
                for ix, xScaleTemp in enumerate(xScaleVec):
                    phDiffSim = dmhtoph(amp0.shape[1], amp0.shape[0], dmrad,
                                        dmr['nact'], inf_func, dmr['ppact_d'],
                                        xScaleTemp*xScale*dmr['ppact_cx'],
                                        yScaleTemp*yScale*dmr['ppact_cy'],
                                        xOffset, yOffset,
                                        clocking, flipx)
                    phDiffSim *= 2  # surface to wavefront
                    phDiffSim = inin(phDiffSim, amp0.shape)
                    phDiffSim = remove_ptt.remove_ptt_directly(phDiffSim,
                                                               usablePixelMap)

                    costMat[iy, ix] = np.sum(usablePixelMap *
                                             np.abs(ph_diff_meas - phDiffSim))

            if iter_ == 0 and nIter > 1:
                bestInds = np.unravel_index(np.argmin(costMat), costMat.shape)
                xScaleTemp = xScaleVec[bestInds[1]]
                yScaleTemp = yScaleVec[bestInds[0]]
            else:
                [xScaleTemp, yScaleTemp] = find_optimum_2d(
                    xScaleVec, yScaleVec, costMat, np.ones_like(costMat))
            xScale = xScaleTemp*xScale
            yScale = yScaleTemp*yScale
            maxDeltaScaleFac = shrinkFac*maxDeltaScaleFac
            xScaleVec = np.linspace(1-maxDeltaScaleFac, 1+maxDeltaScaleFac, nScale)
            yScaleVec = np.linspace(1-maxDeltaScaleFac, 1+maxDeltaScaleFac, nScale)

        ppact_cx_est = xScale*dmr['ppact_cx']
        ppact_cy_est = yScale*dmr['ppact_cy']

    except:  # Fail gracefully
        warnings.warn('Scaling grid search failed.')
        ppact_cx_est = np.nan
        ppact_cy_est = np.nan

    return ppact_cx_est, ppact_cy_est
