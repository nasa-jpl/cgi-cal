# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Module to hold functions modified from PROPER to be standalone for CGI."""

import numpy as np

import cal.util.check as check
from cal.util.nollzernikes import xyzern


def gen_zernikes(zernIndVec, zernCoefVec, x_offset, y_offset, nBeam,
                        nArray=int(1e6)):
    """Generate a shifted, 2-D WFE map composed of Zernike modes.

    Parameters
    ----------
    zernIndVec : numpy ndarray
        1-D array specifying which Zernike polynomials to include. Noll
        indexing used.
    zernCoefVec : numpy ndarray
        1-D array containing Zernike coefficients (in meters of RMS
        wavefront phase error or dimensionless RMS amplitude error) for
        Zernike polynomials indexed by "zernIndVec".
    x_offset, y_offset : float
        offsets in pixels of the Zernike mode from the center pixel
    nBeam : float
        Diameter of the beam containing the Zernikes. Units of pixels.
    nArray : float
        Optional value. Side length of square, 2-D array to contain the
        Zernike maps. If unset, chooses the smallest even value that
        fully contains the beam.

    Returns
    -------
    zernAbMap : numpy ndarray
        2-D, square aberration map comprised of the specified Zernikes
    """
    check.oneD_array(zernIndVec, 'zernIndVec', ValueError)
    check.oneD_array(zernCoefVec, 'zernCoefVec', ValueError)
    if not zernIndVec.size == zernCoefVec.size:
        raise ValueError('zernIndVec and zernCoefVec must have same size.')
    check.real_scalar(x_offset, 'x_offset', ValueError)
    check.real_scalar(y_offset, 'y_offset', ValueError)
    check.real_positive_scalar(nBeam, 'nBeam', ValueError)
    check.positive_scalar_integer(nArray, 'nArray', ValueError)

    if nArray == int(1e6):
        nArray = int(2*np.ceil(0.5*(nBeam+1+np.max((np.abs(x_offset),
                                                    np.abs(y_offset))))))

    x = (np.arange(nArray, dtype=np.float64) - nArray//2 - x_offset)
    y = (np.arange(nArray, dtype=np.float64) - nArray//2 - y_offset)
    xx, yy = np.meshgrid(x, y)

    zernCube = xyzern(xx, yy, nBeam/2., zernIndVec)
    zernAbMap = np.zeros([nArray, nArray], dtype=np.float64)
    for ii in range(zernIndVec.size):
        zernAbMap += zernCoefVec[ii]*np.squeeze(zernCube[ii, :, :])

    return zernAbMap


def fit_zernikes(wfe, usablePixelMap, maxNollZern, nBeam,
                            dxPupil, dyPupil):
    """Compute the coefficients of Zernike modes contained in given WFE.

    Decompose the provided WFE map in the specified pixels into the
    first maxNollZern Noll Zernike modes.

    Parameters
    ----------
    wfe : array_like
        2-D map of the wavefront error
    usablePixelMap : array_like
        2-D boolean map of which pixels to use from the WFE map
    maxNollZern : int
        The maximum Noll Zernike mode to compute a coefficient for.
    nBeam : float
        (Unmasked) diameter of the pupil in the image. Units of pixels.
    dxPupil, dYpupil : float
        x- and y-offsets of the pupil from the center pixel in the pupil
        image. Units of pixels.

    Returns
    -------
    zernCoefVec : numpy ndarray
        1-D array of Zernike coefficients. Same units as the wfe map.

    Notes
    -----
    This is a modified and extremely condensed version of PROPER's
    prop_fit_zernikes.py function. https://pypi.org/project/PyPROPER3/

    """
    # Check inputs
    check.twoD_array(wfe, 'wfe', ValueError)
    check.real_array(wfe, 'wfe', ValueError)
    check.twoD_array(usablePixelMap, 'usablePixelMap', ValueError)
    check.real_array(usablePixelMap, 'usablePixelMap', ValueError)
    if (wfe.shape[0] != usablePixelMap.shape[0]) or \
    (wfe.shape[1] != usablePixelMap.shape[1]):
        raise ValueError('wfe and usablePixelMap must have same shape')
    check.positive_scalar_integer(maxNollZern, 'maxNollZern', ValueError)
    check.real_positive_scalar(nBeam, 'nBeam', TypeError)
    check.real_scalar(dxPupil, 'dxPupil', TypeError)
    check.real_scalar(dyPupil, 'dyPupil', TypeError)

    nx = wfe.shape[1]
    ny = wfe.shape[0]

    xcFit = int(nx/2) + dxPupil
    ycFit = int(ny/2) + dyPupil

    x = np.arange(nx, dtype=np.float64) - xcFit
    y = np.arange(ny, dtype=np.float64) - ycFit
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2) / (nBeam/2.)
    t = np.arctan2(yy, xx)

    # During fitting stage, only use pixels in (possibly resized) pupil region
    # create a mask
    rVec = r.ravel()
    tVec = t.ravel()
    wfe = wfe.ravel()
    usablePixelMapRavel = usablePixelMap.ravel()
    usablePixelVec = (usablePixelMapRavel != 0)  # & (r < 1.0)
    nPix = int(nx * ny)
    rVec = rVec[usablePixelVec]
    tVec = tVec[usablePixelVec]
    wfe = wfe[usablePixelVec]

    nPix = len(rVec)
    ab = np.zeros([nPix, maxNollZern], dtype=np.float64)
    ab[:, 0] = 1.0

    orders = np.arange(maxNollZern)+1
    zernCube = xyzern(xx, yy, nBeam/2., orders)
    for iZern in range(zernCube.shape[0]):
        ab[:, iZern] = zernCube[iZern, :, :].ravel()[usablePixelVec]

    zernCoefVec = np.linalg.lstsq(ab, wfe, rcond=None)[0]

    return zernCoefVec
