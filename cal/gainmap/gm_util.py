# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Utility functions for computing gain maps or height maps of DM actuators."""
import os

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize, Bounds
import scipy.integrate as integrate
from astropy.io import fits

from cal.util import check
from cal.util.dm_crosstalk import CDMCrosstalk

BIGNUM = 1e12 # workaround for issues using bounds with 'powell' minimize

def compute_delta_height_map_from_command_maps(commandMapBefore,
                                               commandMapAfter,
                                               fnRefCube,
                                               fnRefCommandVec,
                                               crosstalk_fn=None):
    """
    Compute difference of actuator heights for two command maps.

    Computes the delta actuator heights (to add to a flight flat setting)
    from the difference of the ground dark hole setting and the ground flat
    setting.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Parameters
    ----------
    commandMapBefore : array_like
        2-D array of DM actuator commands at the earlier setting.
    commandMapAfter : array_like
        2-D array of DM actuator commands at the later setting.
    fnRefCube : str
        Name of the FITS file containing the datacube with measured DM heights
        at the DM commands specified by fnRefCommandVec.
    fnRefCommandVec : str
        Name of the FITS file containing the DM command values at which the
        gains in fnRefCube were found.
    crosstalk_fn : str
        Name of the YAML file containing the crosstalk mapping for the given
        DM. This is passed directly to gmu.compute_heights_for_command_map().
        Default is None.

    Returns
    -------
    deltaHeightMap : array_like (meters)
        2-D array of the difference in actuator heights for the two command
        maps provided.
    """
    check.twoD_array(commandMapBefore, 'commandMapBefore', ValueError)
    check.twoD_array(commandMapAfter, 'commandMapAfter', ValueError)
    if not np.allclose(np.array(commandMapBefore).shape,
                       np.array(commandMapAfter).shape):
        raise ValueError('Both command maps must have the same shape.')

    heightMapBefore = compute_heights_for_command_map(
        commandMapBefore, fnRefCube, fnRefCommandVec,
        crosstalk_fn=crosstalk_fn)

    heightMapAfter = compute_heights_for_command_map(
        commandMapAfter, fnRefCube, fnRefCommandVec,
        crosstalk_fn=crosstalk_fn)

    deltaHeightMap = heightMapAfter - heightMapBefore

    return deltaHeightMap


def compute_commands_for_height_map(heightMap, fnRefCube, fnRefCommandVec,
                                    vmin, vmax,
                                    crosstalk_fn=None,
                                    ):
    """
    Compute actuator commands for a 2-D array of actuator heights.

    This performs a cubic spline interpolation of data from the reference
    datacube of measured heights vs voltage. The interpolator function is
    simply scipy.interpolate.UnivariateSpline(), called separately for each
    actuator.  Any heights that cannot be achieved with DM commands ranging
    from the smallest and largest values in fnRefCommandVec will be assigned
    the voltage that gets closest.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Note regarding DM crosstalk: heightMap = desired actuator heights after
    accounting for crosstalk, then heightMap_applied =
    crosstalk_backward(heightMap) is the height map to compute the commands.

    NOTE: the DM command produced by this function does not have voltage caps
    or neighbor-rule compliance enforced.  Use the constrain_dm() function on
    the output if you wish to do so.

    Parameters
    ----------
    heightMap : array_like (meters)
        2-D array of actuator heights (not surface heights) to convert to
        actuator commands.
    fnRefCube : str
        Name of the FITS file containing the datacube with measured DM heights
        at the DM commands specified by fnRefCommandVec.
    fnRefCommandVec : str
        Name of the FITS file containing the DM command values at which the
        heights in fnRefCube were found.
    vmin : float
        Minimum voltage permitted on an actuator, in volts.
    vmax : float
        Maximum voltage permitted on an actuator, in volts.  Must be > vmin.
    crosstalk_fn : str
        Name of the YAML file containing the crosstalk mapping for the given
        DM. Default is None.

    Returns
    -------
    commandMap : array_like (Volts)
        2-D array of absolute actuator voltages corresponding to the
        given heights.

    """
    check.twoD_array(heightMap, 'heightMap', ValueError)
    nRows = heightMap.shape[0]
    nCols = heightMap.shape[1]

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
    verify_vector_has_unique_values(refCommandVec)
    verify_vector_is_presorted(refCommandVec)

    check.real_scalar(vmin, 'vmin', ValueError)
    check.real_scalar(vmax, 'vmax', ValueError)
    if not (vmax > vmin):
        raise ValueError('vmax must be greater than vmin.')
    vmid = (vmin + vmax)/2

    commandMap = np.zeros((nRows, nCols))

    # DM crosstalk object, if crosstalk_fn=None, default is identity
    dmxtalk = CDMCrosstalk(crosstalk_fn)
    # get applied heightmap from desired heightMap
    heightMap = dmxtalk.crosstalk_backward(heightMap)

    # can't simply interpolate a voltage command value for a given height
    # since voltage vs height may not be a function (may not pass the vertical
    # line test); this can happen if height vs voltage is not monotonic (due
    # to actuator crosstalk or some other issue).
    # Instead, interpolate height and solve for the voltage required.  If that
    # voltage doesn't exist (i.e., if that height can't be achieved), take
    # the voltage that gets closest.
    for iCol in range(nCols):
        for iRow in range(nRows):
            # if all the same number, then use the lowest voltage. Should only
            # apply to non-actuator entries of array (or a completely
            # unresponsive actuator)
            if len(set(refHeightCube[:, iRow, iCol])) == 1:
                voltage = refCommandVec.min()

            else:
                interpolator = UnivariateSpline(
                    refCommandVec,
                    refHeightCube[:, iRow, iCol],
                    k=min(3, len(refHeightCube)-1),
                    ext='extrapolate',
                    s=0,
                )

                def func(x):
                    if x < vmin or x > vmax:
                        out = BIGNUM
                    else:
                        out = np.abs(interpolator(x) - heightMap[iRow, iCol])
                    return out

                # Use Powell method because func may not have smooth gradient
                res  = minimize(func,
                                x0=np.array((vmid,)),
                                method="Powell",
                )
                voltage = res.x[0]

            commandMap[iRow, iCol] = voltage

    return commandMap # Volts


def compute_heights_for_command_map(commandMap, fnRefCube, fnRefCommandVec,
                                    crosstalk_fn=None):
    """
    Compute DM actuator heights for an array of DM commands.

    This performs a cubic spline interpolation of data from the reference
    datacube of measured heights vs voltage. The interpolator function is
    simply scipy.interpolate.UnivariateSpline(),
    called separately for each actuator.

    This function does not:
    - allow extrapolation; out-of-bounds voltages are used at the nearest
      boundary value instead.
    - multiply commands by a gain to get heights.
    - use the influence function to compute the heights.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Note regarding DM crosstalk: commandMap = applied voltage commands
    heightMap_applied is calculated from commandMap, then accounting for
    crosstalk, actual heightMap = crosstalk_forward(heightMap_applied) is
    returned.

    NOTE: no enforcement of voltage caps, neighbor rule constraints, or tied
    and dead actuators are applied to the input DM setting.  Use the
    constrain_dm() function on the input before calling if you wish to do so.

    Parameters
    ----------
    commandMap : array_like
        The array of absolute DM commands about which to compute the gainmap.
        2-D array of floats.
    fnRefCube : str
        Name of the FITS file containing the datacube with measured DM heights
        at the DM commands specified by fnRefCommandVec.
    fnRefCommandVec : str
        Name of the FITS file containing the DM command values at which the
        heights in fnRefCube were found.
    crosstalk_fn : str
        Name of the YAML file containing the crosstalk mapping for the given
        DM. Default is None.


    Returns
    -------
    heightMap : array_like (meters)
        2-D array of DM actuator heights corresponding to the provided
        command map. This is not the same as a surface height map, which
        would require use of the influence function.

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
    verify_vector_has_unique_values(refCommandVec)
    verify_vector_is_presorted(refCommandVec)

    heightMap = np.zeros((nRows, nCols))


    for iCol in range(nCols):
        for iRow in range(nRows):
            # if all the same number, then specify here and don't use the
            # interpolator. Should only
            # apply to non-actuator entries of array (or a completely
            # unresponsive actuator)
            if len(set(refHeightCube[:, iRow, iCol])) == 1:
                heightMap[iRow, iCol] = refHeightCube[0, iRow, iCol]
            else:
                interpolator = UnivariateSpline(refCommandVec,
                                            refHeightCube[:, iRow, iCol],
                                            k=min(3, len(refHeightCube)-1),
                                            ext='const',
                                                )

                heightMap[iRow, iCol] = interpolator(commandMap[iRow, iCol])

    # DM crosstalk object, if crosstalk_fn=None, default is identity
    dmxtalk = CDMCrosstalk(crosstalk_fn)
    # get actual heightmap from applied heightMap
    heightMap = dmxtalk.crosstalk_forward(heightMap)

    return heightMap


def compute_hcube_from_gcube(fnRefCube, fnRefCommandVec,
                             heightCommandVec=np.arange(0, 101, 10),
                             offsetMap=None, offsetVoltage=0):
    """
    Compute the cube of heights from stored cube of gains and an offset height
    map by interpolating a gain map and then integrating to get a height map.
    Then a height cube is made at DM command values for the cube slices as
    specified by the user.  Ideally, the user should specify the smallest and
    largest values of this DM command vector to correspond to the smallest and
    largest voltages possible for the DM actuators, especially if the output
    of this function is intended ultimately for making a gain map with
    compute_gainmap().

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Let h(v) be height as a function of voltage, and let g(v) be DM gain as a
    function of voltage:  h(v) = h0 + integral(g(v') dv', {v0, v}),
    where h(v0) = h0.  So given a gain cube, we interpolate g(v) and integrate,
    using constant of integration h0 (offsetMap) and v0 (offsetVoltage),
    to obtain heights for each slice of the height cube.

    NOTE: no enforcement of voltage caps, neighbor rule constraints, or tied
    and dead actuators are applied to the input DM setting.  Use the
    constrain_dm() function on the input before calling if you wish to do so.

    Parameters
    ----------
    fnRefCube : str
        Name of the FITS file containing the datacube with measured DM gains
        at the DM commands specified by fnRefCommandVec.
    fnRefCommandVec : str
        Name of the FITS file containing the DM command values at which the
        gains in fnRefCube were found.
    heightCommandVec : array, optional
        Array of DM command values desired for the height cube.  Defaults to
        arrray([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]), in Volts.
    offsetMap : array, optional
        2-D array which gives the offset height corresponding to the DM command
        value given by offsetVoltage.  This is the constant of integration 2-D
        array for finding the gain cube.  If None, defaults to an array of
        zeros for the voltage given by offsetVoltage.
    offsetVoltage : float, optional
        DM command value corresponding to offsetMap.  Defaults to 0 Volts.

    Returns
    -------
    heightCube : array_like
        Datacube of DM actuator heights

    """
    _, _, Vec = compute_diff_and_mean_of_vector(fnRefCommandVec)

    check.real_scalar(offsetVoltage, 'offsetVoltage', TypeError)
    if offsetMap is not None:
        check.twoD_array(offsetMap, 'offsetMap', ValueError)

    refGainCube = fits.getdata(os.path.expandvars(fnRefCube))
    check.threeD_array(refGainCube, 'refGainCube', ValueError)

    nSlicesGain = refGainCube.shape[0]
    if nSlicesGain < 2:
        raise ValueError('Number of slices in gain cube should be at '
                         'least 2.')
    nRows = refGainCube.shape[1]
    nCols = refGainCube.shape[2]
    if (Vec.size) != nSlicesGain:
        raise ValueError(('Number of reference slices must equal the number '
                          'of reference commands.'))

    heightCube = np.zeros((len(heightCommandVec), nRows, nCols))

    for iCol in range(nCols):
        for iRow in range(nRows):
            K=3
            if len(set(refGainCube[:, iRow, iCol])) == 1:
                K=1

            gain = UnivariateSpline(Vec,
                                    refGainCube[:, iRow, iCol],
                                    k=min(K, len(refGainCube)-1),
                                    ext='extrapolate',
                                    )

            val = []
            for hvolt in heightCommandVec:
                v, _ = integrate.quad(lambda x: gain(x), offsetVoltage,
                                      hvolt)
                val.append(v)

            heightCube[:, iRow, iCol] = val

    if offsetMap is not None:
        heightCube += offsetMap

    return heightCube


def compute_diff_and_mean_of_vector(fnVec):
    """
    Compute difference and mean of neighboring values in a vector.

    Because this is used on the stored DM commands used for gain
    calculations, a ValueError is raised if the values are not
    unique or if they do not monotonically increase.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Parameters
    ----------
    fnVec : str
        Name of the FITS file containing the vector (vecIn) of floats.


    Returns
    -------
    diffVec : array_like
        1-D array of the differences between neighboring values of vecIn.
    meanVec : array_like
        1-D array of the mean of neighboring values of vecIn.
    vecIn : array_like
        1-D array, vecIn.

    """
    vecIn = fits.getdata(os.path.expandvars(fnVec))

    # Checks on vecIn
    check.oneD_array(vecIn, 'vecIn', ValueError)
    verify_vector_has_unique_values(vecIn)
    verify_vector_is_presorted(vecIn)

    nOut = len(vecIn) - 1
    diffVec = np.zeros(nOut)
    meanVec = np.zeros(nOut)
    for ii in range(nOut):
        diffVec[ii] = vecIn[ii+1] - vecIn[ii]
        meanVec[ii] = (vecIn[ii+1] + vecIn[ii])/2

    return diffVec, meanVec, vecIn


def verify_vector_has_unique_values(vecIn):
    """Check that a vector has all unique values."""
    # Verify is a vector
    check.oneD_array(vecIn, 'vecIn', ValueError)

    # Verify unique values
    vecIn = np.array(vecIn)
    valuesAreUnique = (vecIn.size == np.unique(vecIn).size)
    if not valuesAreUnique:
        raise ValueError('Vector must have unique values.')

    return None


def verify_vector_is_presorted(vecIn):
    """Check that a vector has monotonically increasing values."""
    # Verify is a vector
    check.oneD_array(vecIn, 'vecIn', ValueError)

    # Verify sorted values
    vecIn = np.array(vecIn)
    valuesAreSorted = np.allclose(vecIn,
                                  np.sort(vecIn),
                                  atol=np.finfo(float).eps)
    if not valuesAreSorted:
        raise ValueError('Vector must have monotonically increasing values.')

    return None
