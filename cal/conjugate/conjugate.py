# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Functions for the CONJUGATE module."""
import os
import copy

import numpy as np
from scipy.ndimage import rotate, shift
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import fminbound
from astropy.convolution import Box2DKernel, convolve
from astropy.io import fits

from cal.ampthresh.ampthresh import ampthresh
from cal.gainmap import gainmap
from cal.util import check, shapes
from cal.util.constrain_dm import constrain_dm
from cal.util.dmhtoph import dmh_to_volts, dmhtoph
from cal.util.fit_surf_to_dm import fit_surf_to_dm, build_prefilter
from cal.util.insertinto import insertinto as inin
from cal.util.loadyaml import loadyaml
from cal.util.math import rms, ceil_odd
from cal.conjugate.util.propcustom import gen_zernikes, fit_zernikes

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))

def flatten(cmdDM1, cmdDM2, cmdFCM, flagUseDMs, flagUseFCM, flagHighZern,
            lam, amp, ph, diamPupil, xOffset, yOffset,
            dm_fn, fcm_fn, flatten_params_fn,
            fnRefCubeDM1=None, fnRefCommandVecDM1=None,
            fnRefCubeDM2=None, fnRefCommandVecDM2=None,
            data_path=LOCAL_PATH):
    """
    Compute commands to flatten the given WFE using the FCM and/or both DMs.

    Then apply those delta commands to the existing DM and FCM commands.

    Dead actuators (i.e., those stuck at 0 volts) are accounted for. Any
    wavefront error assigned to a dead DM1 actuator gets reassigned to DM2,
    and vice versa.

    All DM settings will be constrained to obey voltage caps and neighbor rules
    before returning.

    Cases:
    - FCM only
      - FCM corrects Z4 only.
    - DM1 and DM2 only
      - Only DM1 corrects >Zmax if flagHighZern == True.
      - Z4-Zmax correction split between DM1 and DM2 such that RMS for each DM
      phase applied is same.
    - DM1 and DM2 and FCM
      - FCM corrects Z4 only
      - Only DM1 corrects >Zmax if flagHighZern == True.
      - Z5-Zmax correction split between DM1 and DM2 such that RMS for each DM
      phase applied is same.

    All filenames may be absolute or relative paths.  If relative in input
    arguments, they will be relative to the current working directory, not to
    any particular location in Calibration.  If relative within YAML files
    (e.g. the inffn key in dm), this will be relative to
    the path in the data_path argument.

    Parameters
    ----------
    cmdDM1, cmdDM2 : numpy ndarray
        2-D square arrays of total commands currently applied to DM1 and DM2
    cmdFCM : int
        Absolute coarse step position of the FCM.
    flagUseDMs : bool or str or None
        Specify which DM(s) to use, if any. As a bool, choices are True
        (both DMs) or False (neither DM). As None, neither DM is used.
        As a str, choices are: 'DM1', 'DM2', 'BOTH', 'NONE'
    flagUseFCM : bool
        Boolean flag specifying whether to use the FCM here.
    flagHighZern : bool
        Boolean flag for correcting aberrations above Z-maxNollZern or
        not from the provided WFE map.
    lam : float
        Wavelength, used only for phase-to-height conversion. Units of meters.
    amp : numpy ndarray
        2-D array of phase retrieval amplitude data.
    ph : numpy ndarray
        2-D array of phase retrieval phase data in units of radians.
        Must have same shape as amp.
    diamPupil : float
        Pupil diameter. Units of EXCAM pixels. Used for defining Zernike
        polynomials.
    xOffset, yOffset : float
        x- and y-offsets of the pupil from the image center pixel. Units of
        EXCAM pixels. Used for defining Zernike polynominals.
    dm_fn : str
        Name of the YAML file containing calibration data for DMs 1 and 2.
    fcm_fn : str
        Name of the YAML file containing conversion factor values for the FCM.
    flatten_params_fn : str
        Name of file containing static input values needed for the flatten()
        algorithm.
    fnRefCubeDM1 : str
        Name of the FITS file containing the datacube with measured DM heights
        at the DM commands specified by fnRefCommandVecDM1. Used for nonlinear
        fitting of the DM1 voltages to the desired heights. If None, a linear
        fit using a constant gainmap is used instead. Default is None.
    fnRefCommandVecDM1 : str
        Name of the FITS file containing the DM1 command values
        (i.e., voltages) at which the heights in fnRefCubeDM1 were found.
        Used for nonlinear fitting of the DM voltages to the desired heights.
        If None, a linear fit using a constant gainmap is used instead.
        Default is None.
    fnRefCubeDM2 : str
        Name of the FITS file containing the datacube with measured DM heights
        at the DM commands specified by fnRefCommandVecDM2. Used for nonlinear
        fitting of the DM2 voltages to the desired heights. If None, a linear
        fit using a constant gainmap is used instead. Default is None.
    fnRefCommandVecDM2 : str
        Name of the FITS file containing the DM2 command values
        (i.e., voltages) at which the heights in fnRefCubeDM2 were found.
        Used for nonlinear fitting of the DM voltages to the desired heights.
        If None, a linear fit using a constant gainmap is used instead.
        Default is None.
    data_path : str
        Directory to serve as a base for relative paths with YAML files.
        If not supplied, defaults to directory containing this function. If
        using versions of the YAMLs delivered with the repository, the default
        will point to data files also delivered with the repository.

    Returns
    -------
    cmdDM1, cmdDM2 : numpy ndarray
        2-D array of total DM commands with the new deltas included.
    cmdFCM : int
        Absolute coarse step position of the FCM.
    zernCoefVec : array_like
        Zernike mode coefficients fitted to the input WFE map. Used only for
        verifying output in tests and not used in general operations.
    """

    check.twoD_square_array(cmdDM1, 'cmdDM1', TypeError)
    check.real_array(cmdDM1, 'cmdDM1', TypeError)
    check.twoD_square_array(cmdDM2, 'cmdDM2', TypeError)
    check.real_array(cmdDM2, 'cmdDM2', TypeError)
    check.scalar_integer(cmdFCM, 'cmdFCM', TypeError)
    check.boolean(flagUseFCM, 'flagUseFCM', TypeError)
    check.boolean(flagHighZern, 'flagHighZern', TypeError)
    check.real_positive_scalar(lam, 'lam', TypeError)
    check.twoD_array(amp, 'amp', TypeError)
    check.real_array(amp, 'amp', TypeError)
    check.twoD_array(ph, 'ph', TypeError)
    check.real_array(ph, 'ph', TypeError)
    check.real_positive_scalar(diamPupil, 'diamPupil', TypeError)
    check.real_scalar(xOffset, 'xOffset', TypeError)
    check.real_scalar(yOffset, 'yOffset', TypeError)
    check.string(dm_fn, 'dm_fn', TypeError)
    check.string(fcm_fn, 'fcm_fn', TypeError)
    check.string(flatten_params_fn, 'flatten_params_fn', TypeError)
    check.string(data_path, 'data_path', TypeError)
    if fnRefCubeDM1 is not None:
        check.string(fnRefCubeDM1, 'fnRefCubeDM1', TypeError)
    if fnRefCommandVecDM1 is not None:
        check.string(fnRefCommandVecDM1, 'fnRefCommandVecDM1', TypeError)
    if fnRefCubeDM2 is not None:
        check.string(fnRefCubeDM2, 'fnRefCubeDM2', TypeError)
    if fnRefCommandVecDM2 is not None:
        check.string(fnRefCommandVecDM2, 'fnRefCommandVecDM2', TypeError)

    data_path = os.path.expandvars(data_path)

    # flagUseDMs can be bool, str, or None. Convert to str if not already.
    if flagUseDMs is None:
        flagUseDMs = "NONE"
    elif isinstance(flagUseDMs, bool):
        if flagUseDMs:
            flagUseDMs = "BOTH"
        else:
            flagUseDMs = "NONE"
    elif isinstance(flagUseDMs, str):
        flagUseDMs = flagUseDMs.upper()
    else:
        raise TypeError('flagUseDMs must be bool, str, or None')

    allowed_dm_flags = ('BOTH', 'NONE', 'DM1', 'DM2')
    if flagUseDMs not in allowed_dm_flags:
        raise ValueError(
            'flagUseDMs must be one of ''{}'.format(allowed_dm_flags))

    cmdDM1orig = cmdDM1.copy()
    cmdDM2orig = cmdDM2.copy()

    # Load DM Calibration Data
    dmConfig = loadyaml(dm_fn)
    dm = dmConfig["dms"]
    crosstalk_fn_dm1 = dm['DM1']['voltages']['crosstalkfn']
    crosstalk_fn_dm2 = dm['DM2']['voltages']['crosstalkfn']
    dm1r = dm['DM1']['registration']
    dm2r = dm['DM2']['registration']
    inf_func_dm1 = fits.getdata(os.path.join(data_path,
                                os.path.expandvars(dm1r['inffn'])))
    inf_func_dm2 = fits.getdata(os.path.join(data_path,
                                os.path.expandvars(dm2r['inffn'])))

    # Determine the region of the array corresponding to the DM surface
    # for use in the fitting.
    dpad = 2
    nxres1 = round(dpad*dm1r['nact']*dm1r['ppact_cx'])
    nyres1 = round(dpad*dm1r['nact']*dm1r['ppact_cy'])
    wfeShape1 = (nyres1, nxres1)
    nxres2 = round(dpad*dm2r['nact']*dm2r['ppact_cx'])
    nyres2 = round(dpad*dm2r['nact']*dm2r['ppact_cy'])
    wfeShape2 = (nyres2, nxres2)
    nxresMax = int(np.max((nxres1, nxres2)))
    nyresMax = int(np.max((nyres1, nyres2)))
    wfeShapeMax = (nyresMax, nxresMax)

    # Load FCM Calibration Data
    fcmConfig = loadyaml(fcm_fn)
    focusPerStroke = fcmConfig["focusPerStroke"]  # [m RMS per um of stroke]
    strokePerStep = fcmConfig["strokePerStep"]  # [um stroke per coarse step]
    check.real_scalar(focusPerStroke, 'focusPerStroke', TypeError)
    check.real_scalar(strokePerStep, 'strokePerStep', TypeError)

    # Load fixed settings for CONJUGATE
    flattenDict = loadyaml(flatten_params_fn)
    maxNollZern = flattenDict["maxNollZern"]
    nIterSmooth = flattenDict["nIterSmooth"]
    nBin = flattenDict["nBin"]
    arrayExtFac = flattenDict["arrayExtFac"]
    bufferOuterRadiusAct = flattenDict["bufferOuterRadiusAct"]
    check.positive_scalar_integer(maxNollZern, 'maxNollZern', TypeError)
    check.positive_scalar_integer(nIterSmooth, 'nIterSmooth', TypeError)
    check.positive_scalar_integer(nBin, 'nBin', TypeError)
    check.real_positive_scalar(arrayExtFac, 'arrayExtFac', TypeError)
    check.real_nonnegative_scalar(bufferOuterRadiusAct,
                                  'bufferOuterRadiusAct',
                                  TypeError)

    # Crop or pad the PR maps
    wfeIn = inin(ph, wfeShapeMax)
    ampIn = inin(amp, wfeShapeMax)
    # Use the amplitude map to define which pixels to use from the PR map.
    usablePixelMap = ampthresh(ampIn, nBin=nBin)

    zernCoefVec = fit_zernikes(wfeIn, usablePixelMap, maxNollZern,
                               diamPupil, xOffset, yOffset)

    wfeZ1to3, wfeZ4, wfeZ4or5toZmax = gen_split_wfe(
        flagUseFCM, maxNollZern, zernCoefVec, diamPupil, wfeShapeMax,
        xOffset, yOffset)
    wfeAboveZmax = (wfeIn - wfeZ4or5toZmax - wfeZ4 - wfeZ1to3)*usablePixelMap

    # Threshold the low-order Zernikes so that they can't be higher or lower
    # outside the pupil than they are inside the pupil.
    minVal = np.min(wfeZ4or5toZmax[usablePixelMap])
    maxVal = np.max(wfeZ4or5toZmax[usablePixelMap])
    wfeZ4or5toZmax[wfeZ4or5toZmax < minVal] = minVal
    wfeZ4or5toZmax[wfeZ4or5toZmax > maxVal] = maxVal

    # Initialize delta DMs as zeros. Stay unchanged if flagUseDMs == 'NONE'.
    DM1Phase = np.zeros(wfeShape1)
    DM2Phase = np.zeros(wfeShape2)

    if flagUseDMs == 'DM1':

        DM1Phase = wfeZ4or5toZmax + flagHighZern*wfeAboveZmax

    elif flagUseDMs == 'DM2':

        DM2Phase = wfeZ4or5toZmax + flagHighZern*wfeAboveZmax

    elif flagUseDMs == 'BOTH':

        #  Equally Distribute RMS Phase Between DM1 and DM2
        rmsAboveZmax = rms(wfeAboveZmax[usablePixelMap])
        rmsZ4or5toZmax = rms(wfeZ4or5toZmax[usablePixelMap])

        if flagHighZern:
            if rmsZ4or5toZmax > rmsAboveZmax:
                alphaBest = fminbound(phase_cost_function, 0., 1.,
                                      args=(wfeZ4or5toZmax, wfeAboveZmax,
                                            usablePixelMap), disp=0)
            else:
                alphaBest = 0

            DM1Phase = alphaBest*wfeZ4or5toZmax + wfeAboveZmax
            DM2Phase = (1-alphaBest) * wfeZ4or5toZmax

        else:  # Split WFE evenly between DMs if just middle Zernikes
            DM1Phase = 0.5 * wfeZ4or5toZmax
            DM2Phase = 0.5 * wfeZ4or5toZmax

    # Create a mask to zero out the delta phase beyond a certain radius past
    # the outer pupil edge.
    nArraySmall = ceil_odd(arrayExtFac*diamPupil)
    ppact_max = np.max([dm1r['ppact_cx'], dm1r['ppact_cy'],
                        dm2r['ppact_cx'], dm2r['ppact_cy']])
    bufferOuterRadiusPix = bufferOuterRadiusAct * ppact_max
    outerActMask = shapes.circle(
        nArraySmall, nArraySmall, diamPupil/2 + bufferOuterRadiusPix, xOffset,
        yOffset, nSubpixels=1,
    ).astype(bool)

    # Create the Zernike Maps beyond the edges of the DM and limit
    #  phase to the range of the actual illuminated area
    usablePixelMapCrop = inin(usablePixelMap, (nArraySmall, nArraySmall))

    DM1Phase0 = inin(DM1Phase, (nArraySmall, nArraySmall))
    minPhase1 = np.min(DM1Phase0[usablePixelMapCrop == 1])
    maxPhase1 = np.max(DM1Phase0[usablePixelMapCrop == 1])
    DM1Phase0[DM1Phase0 < minPhase1] = minPhase1
    DM1Phase0[DM1Phase0 > maxPhase1] = maxPhase1
    DM1Phase0 *= outerActMask

    DM2Phase0 = inin(DM2Phase, (nArraySmall, nArraySmall))
    minPhase2 = np.min(DM2Phase0[usablePixelMapCrop == 1])
    maxPhase2 = np.max(DM2Phase0[usablePixelMapCrop == 1])
    DM2Phase0[DM2Phase0 < minPhase2] = minPhase2
    DM2Phase0[DM2Phase0 > maxPhase2] = maxPhase2
    DM2Phase0 *= outerActMask

    # Create phase values behind the struts to get a continuous phase. This
    # reduces WFE along the edges of the pupil obscurations when flattening
    # the phase.
    # Used only for aberrations >Zmax since Z1-Zmax are defined everywhere.
    surfDesDM1 = DM1Phase0.copy()/2.
    surfDesDM2 = DM2Phase0.copy()/2.

    if flagHighZern:
        windowWidth1 = round(np.max((dm1r['ppact_cx'], dm1r['ppact_cy'])))
        for _ in range(nIterSmooth):
            # smear out phase to fill in un-illuminated areas
            surfDesDM1 = smooth_surface(surfDesDM1, windowWidth1)
            # reset values within the pupil
            surfDesDM1[usablePixelMapCrop] = \
                DM1Phase0[usablePixelMapCrop].copy()/2.

        windowWidth2 = round(np.max((dm2r['ppact_cx'], dm2r['ppact_cy'])))
        for _ in range(nIterSmooth):
            # smear out phase to fill in un-illuminated areas
            surfDesDM2 = smooth_surface(surfDesDM2, windowWidth2)
            # reset values within the pupil
            surfDesDM2[usablePixelMapCrop] = \
                DM2Phase0[usablePixelMapCrop].copy()/2.

    # Get dead actuator maps for each DM
    dm1tiefn = os.path.join(data_path,
                            os.path.expandvars(dm['DM1']['voltages']['tiefn']))
    tiemap1 = fits.getdata(dm1tiefn)
    nact1 = dm1r['nact']
    isdeadmap1 = np.zeros((nact1, nact1))
    isdeadmap1[tiemap1 == -1] = 1

    dm2tiefn = os.path.join(data_path,
                            os.path.expandvars(dm['DM2']['voltages']['tiefn']))
    tiemap2 = fits.getdata(dm2tiefn)
    nact2 = dm2r['nact']
    isdeadmap2 = np.zeros((nact2, nact2))
    isdeadmap2[tiemap2 == -1] = 1

    # Compute new DM commands (in units of height at each actuator)
    hdm1 = conv_surf_to_dm_cmd(surfDesDM1, dm['DM1'],
                               data_path=data_path)  # radians
    hdm2 = conv_surf_to_dm_cmd(surfDesDM2, dm['DM2'],
                               data_path=data_path)  # radians

    # Reassign heights from dead actuators to the other DM,
    # but only if both DMs are used.
    if flagUseDMs == 'BOTH':
        hdm1_dead = np.zeros_like(hdm1)
        hdm1_dead[isdeadmap1 == 1] = hdm1[isdeadmap1 == 1]
        surf_dm1_dead_act = dmhtoph(
            nArraySmall, nArraySmall, hdm1_dead, dm1r['nact'],
            inf_func_dm1, dm1r['ppact_d'],
            dm1r['ppact_cx'], dm1r['ppact_cy'], dm1r['dx'],
            dm1r['dy'], dm1r['thact'], dm1r['flipx'])

        hdm2_dead = np.zeros_like(hdm2)
        hdm2_dead[isdeadmap2 == 1] = hdm2[isdeadmap2 == 1]
        surf_dm2_dead_act = dmhtoph(
            nArraySmall, nArraySmall, hdm2_dead, dm2r['nact'],
            inf_func_dm2, dm2r['ppact_d'],
            dm2r['ppact_cx'], dm2r['ppact_cy'], dm2r['dx'],
            dm2r['dy'], dm2r['thact'], dm2r['flipx'])

        # Compute the delta heights for just the dead actuators on the other DMs
        dhdm1 = conv_surf_to_dm_cmd(surf_dm2_dead_act, dm['DM1'],
                                    data_path=data_path)  # radians
        dhdm2 = conv_surf_to_dm_cmd(surf_dm1_dead_act, dm['DM2'],
                                    data_path=data_path)  # radians

        # Apply those delta heights
        hdm1 += dhdm1
        hdm2 += dhdm2

    hdm1[isdeadmap1 == 1] = 0  # Zero out delta commands for dead actuators
    hdm2[isdeadmap2 == 1] = 0  # Zero out delta commands for dead actuators

    # Only do the nonlinear fit if the necessary data is defined for both DMs.
    do_nonlin_dm_fit = ((fnRefCubeDM1 is not None) and
                        (fnRefCommandVecDM1 is not None) and
                        (fnRefCubeDM2 is not None) and
                        (fnRefCommandVecDM2 is not None)
                        )

    if do_nonlin_dm_fit:

        # Re-use gainmap.compute_starting_commands_for_flight() to add
        # in height first and then find the voltages using the nonlinear
        # height vs voltage relationship.

        # gainmap.compute_starting_commands_for_flight(..., deltaHeightMap, ...)
        # input argument deltaHeightMap is meters
        # need to convert hdm1, hdm2 to meters
        hdm1 *= lam/(2.0*np.pi)
        hdm2 *= lam/(2.0*np.pi)

        # Multiply hdm1 and hdm2 by -1 so that the input height is negated
        # byt the output command.
        if flagUseDMs in ('BOTH', 'DM1'):
            cmdDM1 = gainmap.compute_starting_commands_for_flight(
                cmdDM1, -1*hdm1, fnRefCubeDM1, fnRefCommandVecDM1, # cmdDM1 (volts), hdm1 (meters)
                dm['DM1']['voltages'], data_path=data_path)

            # cmdDM1 is Volts

        if flagUseDMs in ('BOTH', 'DM2'):
            cmdDM2 = gainmap.compute_starting_commands_for_flight(
                cmdDM2, -1*hdm2, fnRefCubeDM2, fnRefCommandVecDM2,
                dm['DM2']['voltages'], data_path=data_path)

    else:

        # Compute delta voltages and updated total voltages
        dm1gainfn = os.path.join(data_path,
                        os.path.expandvars(dm['DM1']['voltages']['gainfn']))
        dm1gainmap = fits.getdata(dm1gainfn)
        if crosstalk_fn_dm1 is not None:
             crosstalk_fn_dm1 = os.path.join(data_path,
                                    os.path.expandvars(crosstalk_fn_dm1))
        hdm1[dm1gainmap <= 0] = 0  # Zero out delta commands where gains are bad
        deltaCmdDM1 = dmh_to_volts(
            dm1gainmap, hdm1, lam, crosstalk_fn=crosstalk_fn_dm1,
        )

        dm2gainfn = os.path.join(data_path,
                        os.path.expandvars(dm['DM2']['voltages']['gainfn']))
        dm2gainmap = fits.getdata(dm2gainfn)
        if crosstalk_fn_dm2 is not None:
             crosstalk_fn_dm2 = os.path.join(data_path,
                                    os.path.expandvars(crosstalk_fn_dm2))
        hdm2[dm2gainmap <= 0] = 0  # Zero out delta commands where gains are bad
        deltaCmdDM2 = dmh_to_volts(
            dm2gainmap, hdm2, lam, crosstalk_fn=crosstalk_fn_dm2,
        )

        # Zero out NAN values in delta command maps
        deltaCmdDM1[np.isnan(deltaCmdDM1)] = 0
        deltaCmdDM2[np.isnan(deltaCmdDM2)] = 0
        # Zero out infinite values in delta command maps
        deltaCmdDM1[np.isinf(deltaCmdDM1)] = 0
        deltaCmdDM2[np.isinf(deltaCmdDM2)] = 0

        # To be extra sure, zero out delta commands to a DM if it isn't used.
        if flagUseDMs not in ('BOTH', 'DM1'):
            deltaCmdDM1 *= 0
        if flagUseDMs not in ('BOTH', 'DM2'):
            deltaCmdDM2 *= 0

        cmdDM1 = cmdDM1 - deltaCmdDM1
        cmdDM2 = cmdDM2 - deltaCmdDM2

    # Just in case, remove NAN values in nearly final command maps
    # Assumes input did not have any NAN values.
    cmdDM1[np.isnan(cmdDM1)] = cmdDM1orig[np.isnan(cmdDM1)]
    cmdDM2[np.isnan(cmdDM2)] = cmdDM2orig[np.isnan(cmdDM2)]

    # Just in case, infinite NAN values in nearly final command maps
    # Assumes input did not have any infinite values.
    cmdDM1[np.isinf(cmdDM1)] = cmdDM1orig[np.isinf(cmdDM1)]
    cmdDM2[np.isinf(cmdDM2)] = cmdDM2orig[np.isinf(cmdDM2)]

    cmdDM1 = constrain_dm(cmdDM1,
                          fits.getdata(
                              os.path.join(data_path,
                                           os.path.expandvars(
                                            dm['DM1']['voltages']['flatfn']))
                          ),
                          fits.getdata(
                              os.path.join(data_path,
                                           os.path.expandvars(
                                            dm['DM1']['voltages']['tiefn']))
                          ),
                          vmax=dm['DM1']['voltages']['vmax'],
                          vlat=dm['DM1']['voltages']['vneighbor'],
                          vdiag=dm['DM1']['voltages']['vcorner'],
                          vquant=dm['DM1']['voltages']['vquant'],
    )
    cmdDM2 = constrain_dm(cmdDM2,
                          fits.getdata(
                              os.path.join(data_path,
                                           os.path.expandvars(
                                            dm['DM2']['voltages']['flatfn']))
                          ),
                          fits.getdata(
                              os.path.join(data_path,
                                           os.path.expandvars(
                                            dm['DM2']['voltages']['tiefn']))
                          ),
                          vmax=dm['DM2']['voltages']['vmax'],
                          vlat=dm['DM2']['voltages']['vneighbor'],
                          vdiag=dm['DM2']['voltages']['vcorner'],
                          vquant=dm['DM2']['voltages']['vquant'],
    )

    # Compute new FCM command
    if flagUseFCM:
        z4_in_fcm_steps =\
            zernCoefVec[3]*lam/(2*np.pi)/focusPerStroke/strokePerStep
        deltaCmdFCM = -z4_in_fcm_steps
        cmdFCM = int(round(cmdFCM + deltaCmdFCM))

    return cmdDM1, cmdDM2, cmdFCM, zernCoefVec


def gen_split_wfe(flagUseFCM, maxNollZern, zernCoef, diamPupil, wfeShape,
                  xOffset, yOffset):
    """
    Generate WFE maps split into different sets of Zernike modes.

    Parameters
    ----------
    flagUseFCM: bool
        Boolean flag denoting whether Z4 will be corrected with the FCM
        or not (in which case the DM(s) will.)
    maxNollZern : int
        The maximum Noll Zernike mode to include as part of the low-
        order modes to split between DMs 1 and 2.
    zernCoefVec : numpy ndarray
        1-D array of Zernike coefficients. The coefficient for Noll mode 1 goes
        in position 0 of the array. The max number of Noll modes used is chosen
        as the size of this array.
    diamPupil : float
        (Unmasked) diameter of the pupil in the image. The Zernike maps will be
        generated to have this diameter. Units of pixels.
    wfeShape : array_like
        1-D array giving the shape of the WFE map arrays to produce.
    xOffset, yOffset : float
        x- and y-offsets of the pupil from the center pixel in the pupil
        image. Units of pixels.

    Returns
    -------
    wfeZ1to3 : numpy ndarray
        2-D WFE map for Zernike modes Z1-Z3. Used for verification only.
    wfeZ4 : numpy ndarray
        2-D WFE map for Zernike mode Z4. Used for verification only. Is zero
        if flagUseFCM == False.
    wfeZ4or5toZmax
        2-D WFE map for Zernike modes Z4-Zmax if flagUseFCM == False, or
        Z5-Zmax if flagUseFCM == True. Zmax is the largest Zernike mode
        specified by the length of the coefficients vector zernCoef.
    """
    check.boolean(flagUseFCM, 'flagUseFCM', TypeError)
    check.positive_scalar_integer(maxNollZern, 'maxNollZern', TypeError)
    check.oneD_array(zernCoef, 'zernCoef', TypeError)
    check.real_array(zernCoef, 'zernCoef', TypeError)
    # if not zernCoef.size >= 5:
    #     raise TypeError('zernCoef must have at least 5 values for correct' +
    #                      ' WFE map indexing.')
    check.real_positive_scalar(diamPupil, 'diamPupil', TypeError)
    check.oneD_array(wfeShape, 'wfeShape', TypeError)
    check.real_array(wfeShape, 'wfeShape', TypeError)
    if not np.array(wfeShape).size == 2:
        raise TypeError('wfeShape must contain two values')
    check.real_scalar(xOffset, 'xOffset', TypeError)
    check.real_scalar(yOffset, 'yOffset', TypeError)

    # Change zernCoef to length maxNollZern
    if zernCoef.size < maxNollZern:
        zernCoef0 = copy.copy(zernCoef)
        zernCoef = np.zeros((maxNollZern,))
        zernCoef[0:zernCoef0.size] = zernCoef0
    elif zernCoef.size > maxNollZern:
        zernCoef = zernCoef[0:maxNollZern]

    nArray = round(np.max(wfeShape))

    if flagUseFCM:
        zCoef = zernCoef[4::]
        Z4coef = zernCoef[3]  # FCM's component
        zSetDM1 = np.arange(5, maxNollZern+1)

        wfeZ4 = gen_zernikes(np.array([4]), np.array([Z4coef]), xOffset,
                             yOffset, diamPupil, nArray=nArray)
        wfeZ4 = inin(wfeZ4, wfeShape)
    else:
        zCoef = zernCoef[3::]
        Z4coef = 0.  # FCM's component
        zSetDM1 = np.arange(4, maxNollZern+1)

        wfeZ4 = np.zeros(wfeShape)

    wfeZ1to3 = gen_zernikes(np.array([1, 2, 3]), zernCoef[0:3], xOffset,
                            yOffset, diamPupil, nArray=nArray)
    wfeZ1to3 = inin(wfeZ1to3, wfeShape)

    wfeZ4or5toZmax = gen_zernikes(zSetDM1, zCoef, xOffset, yOffset,
                                  diamPupil, nArray=nArray)
    wfeZ4or5toZmax = inin(wfeZ4or5toZmax, wfeShape)

    return wfeZ1to3, wfeZ4, wfeZ4or5toZmax


def smooth_surface(arrayIn, windowWidth):
    """
    Smooth values in a 2-D array via convolution.

    The square convolution kernel has width "windowWidth".

    Parameters
    ----------
    arrayIn : array_like
        2-D, real-valued array to be smoothed
    windowWidth : float
        width of the square convolution kernel for smoothing

    Returns
    -------
    smoothedArray : numpy ndarray
        Smoothed version of the input 2-D map. Same shape as input.
    """
    check.twoD_array(arrayIn, 'arrayIn', TypeError)
    check.real_array(arrayIn, 'arrayIn', TypeError)
    check.real_positive_scalar(windowWidth, 'windowWidth', TypeError)

    smoothedArray = convolve(arrayIn, Box2DKernel(windowWidth))
    smoothedArray = inin(smoothedArray, arrayIn.shape)

    return smoothedArray


def conv_surf_to_dm_cmd(surfaceToFit, dm, data_path=LOCAL_PATH):
    """
    Compute the DM commands that recreate a given surface.

    This command should do the inverse of cal.util.dmhtoph.dmhtoph(), which
    does fliplr, rotation, and then translation. So this function should do
    translation, rotation, and then fliplr.

    NOTE: the DM command produced by this function does not have voltage caps
    or neighbor-rule compliance enforced.  Use the constrain_dm() function on
    the output if you wish to do so.

    All filenames may be absolute or relative paths.  If relative in input
    arguments, they will be relative to the current working directory, not to
    any particular location in Calibration.  If relative within YAML files
    (e.g. the inffn key in dm), this will be relative to
    the path in the data_path argument.

    Parameters
    ----------
    surfaceToFit : array_like, meters or radians
        2-D array of the surface heights for the DM to fit
    dm : dict
        Dictionary containing DM parameters and registration.
    data_path : str
        Directory to serve as a base for relative paths with YAML files.
        If not supplied, defaults to directory containing this function. If
        using versions of the YAMLs delivered with the repository, the default
        will point to data files also delivered with the repository.


    Returns
    -------
    dmhtout : numpy ndarray
        2-D array of DM commands, same units as input surfaceToFit (meters or radians)
    """
    # Check direct inputs
    check.twoD_array(surfaceToFit, 'surfaceToFit', TypeError)
    check.real_array(surfaceToFit, 'surfaceToFit', TypeError)
    if not isinstance(dm, dict):
        raise TypeError('dm must be a dict')
    data_path = os.path.expandvars(data_path)

    dmReg = dm['registration']
    actPitch = dm['pitch']
    dx = dmReg['dx']
    dy = dmReg['dy']
    ppact_cx = dmReg['ppact_cx']
    ppact_cy = dmReg['ppact_cy']
    thact = dmReg['thact']
    nact = dmReg['nact']
    ppact_d = dmReg['ppact_d']
    inffn = os.path.join(data_path, os.path.expandvars(dmReg['inffn']))
    flipx = dmReg['flipx']

    # Check values of the keys
    check.real_positive_scalar(actPitch, 'actPitch', TypeError)
    check.real_scalar(dx, 'dx', TypeError)
    check.real_scalar(dy, 'dy', TypeError)
    check.real_positive_scalar(ppact_cx, 'ppact_cx', TypeError)
    check.real_positive_scalar(ppact_cy, 'ppact_cy', TypeError)
    check.real_scalar(thact, 'thact', TypeError)
    check.positive_scalar_integer(nact, 'nact', TypeError)
    check.real_positive_scalar(ppact_d, 'ppact_d', TypeError)
    check.boolean(flipx, 'flipx', TypeError)
    if not isinstance(inffn, str):
        raise TypeError('inffn must be a string')
    try:
        infFunc = fits.getdata(inffn)
        check.twoD_square_array(infFunc, 'infFunc', TypeError)
    except OSError:
        raise OSError('Could not read influence function from %s' % inffn)

    dx_inf0 = actPitch/float(ppact_d)
    actres1 = actPitch/dx_inf0

    # First undo the translation
    surfaceToFit = shift(surfaceToFit, [-dy, -dx])

    # Undo rotation
    surfaceToFit = rotate(surfaceToFit, thact, reshape=False)

    # Last, undo flip that dmhtoph does first.
    if flipx:
        surfaceToFit = np.fliplr(surfaceToFit)


    # Define coordinates for the surface
    [nY, nX] = surfaceToFit.shape
    if nX % 2 == 0:
        xsA = np.linspace(-nX/2, (nX/2)-1, nX) / ppact_cx
    else:
        xsA = np.linspace(-(nX-1)/2, (nX-1)/2, nX)/ppact_cx

    if nY % 2 == 0:
        ysA = np.linspace(-nY/2, (nY/2)-1, nY) / ppact_cy
    else:
        ysA = np.linspace(-(nY-1)/2, (nY-1)/2, nY) / ppact_cy

    ppact_new = 1
    nB = nact*ppact_new
    xsB = np.linspace(-(nB-1)/2, (nB-1)/2, nB)/ppact_new  # centered on DM
    ysB = xsB
    # RectBivariateSpline is MUCH faster than interp2d
    interp_spline = RectBivariateSpline(ysA, xsA, surfaceToFit)
    gridDerotAtActRes = interp_spline(ysB, xsB)

    sparseMat = build_prefilter(gridDerotAtActRes.shape[0],
                                gridDerotAtActRes.shape[1], infFunc, actres1)
    dmhtout = fit_surf_to_dm(gridDerotAtActRes, sparseMat)

    return dmhtout # same units as surfaceToFit


def phase_cost_function(alpha, wfeZ4or5toZmax, wfeAboveZmax, usablePixelMap):
    """
    Compute the quadratic cost of the RMS phase difference for DMs 1 and 2.

    This function is used in evenly distributing the RMS phase between
    DMs 1 and 2. The weight alpha can vary between 0 and 1. The solver
    fminbound uses the cost function from this function to compute the best
    value of alpha.

    Parameters
    ----------
    alpha : float
        Weighting value between 0.0 and 1.0, inclusive. Determines how much of
        wfeZ4or5toZmax to allocate to DM1 on top of wfeAboveZmax.
        (1-alpha)*wfeZ4or5toZmax goes to DM2.
    wfeZ4or5toZmax : array_like
        2-D WFE map comprised of Zernike modes Z4 or Z5 to Zmax. Starts at Z4
        if FCM is not used or at Z5 if FCM is used.
    wfeAboveZmax : array_like
        2-D WFE map comprised of everything excluding Zernike terms Z1-Zmax
        from the original WFE map.
    usablePixelMap : array_like
        2-D boolean map of pixels to use in the WFE maps.

    Returns
    -------
    cost : float
        Quadratic cost of RMS phase difference for DMs 1 and 2.

    """
    check.real_scalar(alpha, 'alpha', TypeError)
    if alpha < 0 or alpha > 1:
        raise TypeError('alpha must be in range [0, 1].')
    check.twoD_array(wfeZ4or5toZmax, 'wfeZ4or5toZmax', TypeError)
    check.real_array(wfeZ4or5toZmax, 'wfeZ4or5toZmax', TypeError)
    check.twoD_array(wfeAboveZmax, 'wfeAboveZmax', TypeError)
    check.real_array(wfeAboveZmax, 'wfeAboveZmax', TypeError)
    check.twoD_array(usablePixelMap, 'usablePixelMap', TypeError)
    if not (usablePixelMap == usablePixelMap.astype(bool)).all():
        raise TypeError('usablePixelMap must be an array of booleans')
    if not wfeZ4or5toZmax.shape == usablePixelMap.shape:
        raise TypeError(('wfeZ4or5toZmax and usablePixelMap ' +
                          'must have same shape'))
    if not wfeAboveZmax.shape == usablePixelMap.shape:
        raise TypeError(('wfeAboveZmax and usablePixelMap ' +
                          'must have same shape'))

    cost = (rms(wfeAboveZmax[usablePixelMap] +
                alpha*wfeZ4or5toZmax[usablePixelMap]) -
            rms((1.-alpha)*wfeZ4or5toZmax[usablePixelMap]))**2

    return cost
