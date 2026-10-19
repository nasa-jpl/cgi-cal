# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Functions for computing gain maps or height maps of DM actuators."""
import os

import numpy as np
from scipy.interpolate import UnivariateSpline
from astropy.io import fits

from cal.util import check, extend_radially
from cal.util.constrain_dm import constrain_dm
import cal.gainmap.gm_util as gmu

vdict_req_keys = {'vmin', 'vmax', 'vneighbor', 'vcorner', 'vquant', 'tiefn',
                  'flatfn'}

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))

def compute_starting_commands_for_flight(flatCommandMap,
                                         deltaHeightMap,
                                         fnRefCube,
                                         fnRefCommandVec,
                                         voltageDict,
                                         data_path=LOCAL_PATH,
                                         ):
    """
    Compute starting commands for EFC from a flat map and a delta height map.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Note: this function will constrain computed voltages to obey all voltage
    caps and neighbor-rule constraints and to respect tied and dead actuators.

    Note regarding DM crosstalk: the functions
    gmu.compute_heights_for_command_map() and
    gmu.compute_commands_for_height_map() include DM crosstalk. Then
    deltaHeightMap must be "desired=actual" actuator heights, i.e. after
    including DM crosstalk.

    Parameters
    ----------
    flatCommandMap : array_like
        2-D array of DM actuator commands (i.e., voltages) for the flattened
        wavefront setting.
    deltaHeightMap : array_like (meters)
        2-D array of differenced actuator heights, not to be confused with
        surface heights. MUST already include any effects of DM crosstalk.
    fnRefCube : str
        Name of the FITS file containing the datacube with measured DM heights
        at the DM commands specified by fnRefCommandVec.
    fnRefCommandVec : str
        Name of the FITS file containing the DM command values
        (i.e., voltages) at which the heights in fnRefCube were found.
    voltageDict : dict
        Dictionary containing at least the following required keys:
          vmin: minimum permitted voltage, in volts
          vmax: maximum permitted voltage, in volts
          vneighbor: maximum lateral neighbor rule voltage, in volts
          vcorner: maximum diagonal neighbor rule voltage, in volts
          vquant: DAC step size, in volts
          tiefn: string with path to DM tiemap FITS file
          flatfn string with path to DM flatmap FITS file
        This dictionary may also contain the following key:
          crosstalkfn: string with path to DM crosstalk YAML file, or
           None/blank.  If this key is not present or is blank, None (i.e. no
           crosstalk) will be used as the default.
    data_path : str
        Directory to serve as a base for relative paths for flat, tie, and
        crosstalk filenames.  If not supplied, defaults to directory
        containing this function.

    Returns
    -------
    startingCommandMap : array_like
        2-D array of DM actuator commands (Volts) about which to start EFC.

    """
    check.twoD_array(flatCommandMap, 'flatCommandMap', TypeError)
    check.twoD_array(deltaHeightMap, 'deltaHeightMap', TypeError)
    check.string(fnRefCube, 'fnRefCube', TypeError)
    check.string(fnRefCommandVec, 'fnRefCommandVec', TypeError)
    check.string(data_path, 'data_path', TypeError)
    data_path = os.path.expandvars(data_path)
    if not np.allclose(np.array(flatCommandMap).shape,
                       np.array(deltaHeightMap).shape):
        raise ValueError('flatMap and deltaHeightMap must have same shape.')

    if not isinstance(voltageDict, dict):
        raise TypeError('voltageDict must be a dict')
    if vdict_req_keys - set(voltageDict.keys()) != set(): # missing keys
        raise ValueError('Missing required voltageDict keys')
    vmin = voltageDict['vmin']
    vmax = voltageDict['vmax']
    vlat = voltageDict['vneighbor']
    vdiag = voltageDict['vcorner']
    vquant = voltageDict['vquant']
    tiemap = fits.getdata(os.path.join(data_path,
                          os.path.expandvars(voltageDict['tiefn'])))
    flatmap = fits.getdata(os.path.join(data_path,
                           os.path.expandvars(voltageDict['flatfn'])))
    crosstalk_fn = voltageDict.get('crosstalkfn', None)
    if crosstalk_fn is not None:
        crosstalk_fn = os.path.join(data_path, crosstalk_fn)
        pass

    # flatHeightMap returned here is actual after including DM crosstalk, (meters) includes piston
    flatHeightMap = gmu.compute_heights_for_command_map(
        flatCommandMap, fnRefCube, fnRefCommandVec, crosstalk_fn=crosstalk_fn)

    # actual (desired) actuator heights including crosstalk
    startingHeightMap = flatHeightMap + deltaHeightMap # meters

    # accounts for DM crosstalk:
    startingCommandMap_V = gmu.compute_commands_for_height_map(
        startingHeightMap, fnRefCube, fnRefCommandVec, vmin, vmax,
        crosstalk_fn=crosstalk_fn)

    startingCommandMap_V = constrain_dm(
        startingCommandMap_V,
        flatmap,
        tiemap,
        vmax=vmax,
        vlat=vlat,
        vdiag=vdiag,
        vquant=vquant,
    )

    return startingCommandMap_V


def compute_gainmap(commandMap, fnRefCube, fnRefCommandVec, threshold=None,
                    min_gain=None):
    """
    Compute a gainmap for absolute DM commands based on reference data.

    Finds the gain map by interpolating DM actuator height as a function of
    DM command value and taking the derivative with respect to DM command
    voltage for each entry in commandMap. Any command values outside the bounds
    of the DM command vector from fnRefCommandVec will result in an
    exception.  Each actuator is independent and thus
    has a separate call to scipy.interpolate.UnivariateSpline().

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    NOTE: no enforcement of voltage caps, neighbor rule constraints, or tied
    and dead actuators are applied to the input DM setting.  Use the
    constrain_dm() function on the input before calling if you wish to do so.

    Non-ideal actuators can have a non-monotonic height vs voltage curve, and 
    the DM gain is the slope of this curve.
    However, utilizing zero or negative gain for an actuator may in practice
    halt the actuator or make it go in the wrong direction.  This function 
    sets any gain values that are at or below a user-specified threshold to be 
    a user-specified value. Actuators outside of the annulus are given values 
    that are smooth extensions of the annulus actuator values.

    Parameters
    ----------
    commandMap : array_like
        The array of absolute DM commands (Volts) about which to compute the
        gainmap.  2-D array of floats.
    fnRefCube : str
        Name of the FITS file containing the datacube with measured heights
        at the DM commands specified by fnRefCommandVec.
    fnRefCommandVec : str
        Name of the FITS file containing the DM command values at which the
        heights in fnRefCube were found.
    threshold : float, >= 0
        Threshold gain value at or below which the gain is unacceptable and 
        adjusted to the value from min_gain.  If None, 0 is used.  Defaults to 
        None.
    min_gain : array_like
        2-D array of minimum DM gain to be used for the actuators.  
        If None, an array of the smallest positive number allowed by machine 
        precision is used.  Defaults to None.
 
    Returns
    -------
    gainMap : array_like
        2-D array of DM actuator gains (nm/V)  about the provided command map.

    """
    check.twoD_array(commandMap, 'commandMap', ValueError)
    nRows = commandMap.shape[0]
    nCols = commandMap.shape[1]
    refHeightCube = fits.getdata(os.path.expandvars(fnRefCube))
    check.threeD_array(refHeightCube, 'refHeightCube', ValueError)
    if refHeightCube.shape[0] < 2:
        raise ValueError('Number of slices in height cube should be at '
                         'least 2.')
    if nRows != refHeightCube.shape[2]:
        raise ValueError('Number of actuator columns must be consistent.')
    if nCols != refHeightCube.shape[1]:
        raise ValueError('Number of actuator rows must be consistent.')

    refCommandVec = fits.getdata(os.path.expandvars(fnRefCommandVec))

    if refCommandVec.size != refHeightCube.shape[0]:
        raise ValueError('Number of height cube slices must equal the '
                            'number of reference commands.')
    # the two below check if refCommandVec is non-1-D array
    gmu.verify_vector_has_unique_values(refCommandVec)
    gmu.verify_vector_is_presorted(refCommandVec)

    gainMap = np.zeros((nRows, nCols))

    if threshold is not None:
        check.real_nonnegative_scalar(threshold, 'threshold', ValueError)
    if min_gain is not None:
        check.twoD_array(min_gain, 'min_gain', ValueError)
        if (min_gain <= 0).any():
            raise ValueError('min_gain must be an array consisting of values '
                             '> 0.')

    if threshold is None:
        threshold = 0
    if min_gain is None:
        min_gain = np.ones((nRows, nCols))*np.finfo(float).eps 

    for iCol in range(nCols):
        for iRow in range(nRows):
            # if all the same number, then set to 0. This should 
            # only be 
            # the case for the array entries that do not correspond to annulus
            # actuators.  If what's below applies to an actuator, then that 
            # actuator is not vey useful.  extend_radially will handle these 
            # and make them positive.
            if len(set(refHeightCube[:, iRow, iCol])) == 1:
                gainMap[iRow, iCol] = 0
            else:
                height = UnivariateSpline(refCommandVec,
                                        refHeightCube[:, iRow, iCol],
                                        k=min(3, len(refHeightCube)-1),
                                        ext='raise',
                                        )
                
                v = commandMap[iRow, iCol]

                gainMap[iRow, iCol] = height.derivative(1)(v)

    # Extend values out radially to fill in missing values
    if nRows % 2 == 0:
        x_offset = -0.5
        y_offset = -0.5
    else:
        x_offset = 0
        y_offset = 0

    gainMap = extend_radially.extend_radially(gainMap,
                                              x_offset=x_offset,
                                              y_offset=y_offset,
                                              atol=10*np.finfo(float).eps,
                                              )
    gainMap = extend_radially.extend_radially(gainMap,
                                              x_offset=x_offset,
                                              y_offset=y_offset,
                                              direction=-1,
                                              atol=10*np.finfo(float).eps,
                                              )
    
    # just in case any actuator is below some desired threshold:
    ind = np.where(gainMap <= threshold)
    gainMap[ind] = min_gain[ind]

    return gainMap


def compute_gainmap_from_gain_cube(commandMap, fnRefCube, fnRefCommandVec,
                                   threshold=None, min_gain=None):
    """
    Compute a gainmap for absolute DM voltages from a datacube of gain maps.

    Finds the gain map by linearly interpolating a data cube of DM gains. Each
    actuator is independent and thus has a separate call to
    scipy.interpolate.UnivariateSpline(). If the input voltage is outside the
    range of reference voltages, the returned gain is the same as for the
    nearest measured reference voltage.

    NOTE: no enforcement of voltage caps, neighbor rule constraints, or tied
    and dead actuators are applied to the input DM setting.  Use the
    constrain_dm() function on the input before calling if you wish to do so.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Non-ideal actuators can have a non-monotonic height vs voltage curve, and 
    the DM gain is the slope of this curve.
    However, utilizing zero or negative gain for an actuator may in practice
    halt the actuator or make it go in the wrong direction.  This function 
    sets any gain values that are at or below a user-specified threshold to be 
    a user-specified value.  Actuators outside of the annulus are given values 
    that are smooth extensions of the annulus actuator values.

    Parameters
    ----------
    commandMap : array_like
        The array of absolute DM commands (Volts) about which to compute the
        gainmap. 2-D array of floats.
    fnRefCube : str
        Name of the FITS file containing the datacube with measured DM gains
        at the DM commands specified by fnRefCommandVec.
    fnRefCommandVec : str
        Name of the FITS file containing the DM command values at which the
        gains in fnRefCube were found.
    threshold : float, >= 0
        Threshold gain value at or below which the gain is unacceptable and 
        adjusted to the value from min_gain.  If None, 0 is used.  Defaults to 
        None.
    min_gain : array_like
        2-D array of minimum DM gain to be used for the actuators.  
        If None, an array of the smallest positive number allowed by machine 
        precision is used.  Defaults to None.

    Returns
    -------
    gainMap : array_like
        2-D array of DM actuator gains (nm/V) about the provided command map.

    """
    _, _, Vec = gmu.compute_diff_and_mean_of_vector(fnRefCommandVec)

    check.twoD_array(commandMap, 'commandMap', ValueError)
    nRows = commandMap.shape[0]
    nCols = commandMap.shape[1]

    gainCube = fits.getdata(os.path.expandvars(fnRefCube))
    check.threeD_array(gainCube, 'gainCube', ValueError)
    if gainCube.shape[0] < 2:
        raise ValueError('Number of slices in height cube should be at '
                         'least 2.')
    if nRows != gainCube.shape[2]:
        raise ValueError('Number of actuator columns must be consistent.')
    if nCols != gainCube.shape[1]:
        raise ValueError('Number of actuator rows must be consistent.')
    if Vec.size != gainCube.shape[0]:
        raise ValueError('Number of gain cube slices must match the size of '
                         'the command value vector.')

    if gainCube.min() < 0:
        raise ValueError('Gain should not be negative.')

    gainMap = np.zeros((nRows, nCols))

    if threshold is not None:
        check.real_nonnegative_scalar(threshold, 'threshold', ValueError)
    if min_gain is not None:
        check.twoD_array(min_gain, 'min_gain', ValueError)
        if (min_gain <= 0).any():
            raise ValueError('min_gain must be an array consisting of values '
                             '> 0.')

    if threshold is None:
        threshold = 0
    if min_gain is None:
        min_gain = np.ones((nRows, nCols))*np.finfo(float).eps 

    for iCol in range(nCols):
        for iRow in range(nRows):
            if len(set(gainCube[:, iRow, iCol])) == 1:
                gainMap[iRow, iCol] = gainCube[0, iRow, iCol]
            else:
                # Set to the boundary gain values if the voltages are out of
                # range of the reference data. Otherwise, interpolate.
                interpolator = UnivariateSpline(Vec, # command Volts
                                            gainCube[:, iRow, iCol],
                                            k=min(3, len(gainCube)-1),
                                            ext='const', #set as boundary val
                                            )
                gainMap[iRow, iCol] = interpolator(commandMap[iRow, iCol])

    # Extend values out radially to fill in missing values
    if nRows % 2 == 0:
        x_offset = -0.5
        y_offset = -0.5
    else:
        x_offset = 0
        y_offset = 0

    gainMap = extend_radially.extend_radially(gainMap,
                                              x_offset=x_offset,
                                              y_offset=y_offset,
                                              atol=10*np.finfo(float).eps,
                                              )
    gainMap = extend_radially.extend_radially(gainMap,
                                              x_offset=x_offset,
                                              y_offset=y_offset,
                                              direction=-1,
                                              atol=10*np.finfo(float).eps,
                                              )
    
    # just in case any actuator is below some desired threshold:
    ind = np.where(gainMap <= threshold)
    gainMap[ind] = min_gain[ind]

    return gainMap
