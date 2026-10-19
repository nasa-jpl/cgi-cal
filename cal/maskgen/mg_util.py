# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Module to hold utility functions for MASKGEN."""
import numpy as np

from cal.util.check import real_scalar, positive_scalar_integer, \
    real_positive_scalar, real_nonnegative_scalar


def gen_annular_segments(nSeg, rInPix, rOutPix, nx, ny, xOffset, yOffset,
                        angOpen, angRot, nSubpixels=11, dTheta=5):
    """
    Generate a partial annulus with antialiased edges at specified location.

    Parameters
    ----------
    nSeg : {1, 2}
        Number of annular segments to include in the mask representation.
        Allowed values are 1 and 2.
    rInPix : float
        Inner radius of the annulus in pixels.
    rOutPix : float
        Outer radius of the annulus in pixels.
    nx, ny : array_like
        Dimensions of the 2-D array to create.
    xOffset, yOffset : float
        Lateral offsets in pixels of the annulus's center from the array's
        center pixel.
    angOpen : float
        Opening angle of the annular segment. Units of degrees.
    angRot : float
        Rotation angle of the annular segment. Units of degrees.
    nSubpixels : int
        Each edge pixel of the circle is subdivided into a square subarray
        nSubpixels across. The subarray is given binary values and then
        averaged to give the edge pixel a value between 0 and 1, inclusive.
        Default value is 11. Must be a positive scalar integer.
    dTheta : float
        Extra padding of the azimuthal region when determining where to compute
        the antialiased edge pixels. Units of degrees. Default value is 5.

    Returns
    -------
    mask : numpy ndarray
        2-D array containing the annular segment(s)

    """
    positive_scalar_integer(nSeg, 'nSeg', TypeError)
    if not (nSeg == 1 or nSeg == 2):
        raise TypeError('nSeg must be 1 or 2')
    real_positive_scalar(rInPix, 'rInPix', TypeError)
    real_positive_scalar(rOutPix, 'rOutPix', TypeError)
    positive_scalar_integer(nx, 'nx', TypeError)
    positive_scalar_integer(ny, 'ny', TypeError)
    real_scalar(xOffset, 'xOffset', TypeError)
    real_scalar(yOffset, 'yOffset', TypeError)
    real_nonnegative_scalar(angOpen, 'angOpen', TypeError)
    real_scalar(angRot, 'angRot', TypeError)
    positive_scalar_integer(nSubpixels, 'nSubpixels', TypeError)
    real_nonnegative_scalar(dTheta, 'dTheta', TypeError)

    # Convert degrees to radians
    angOpenRad = angOpen*np.pi/180
    angRotRad = angRot*np.pi/180
    dThetaRad = dTheta*np.pi/180

    # Coordinates in the output array
    if nx % 2 == 0:
        x = np.linspace(-nx/2., nx/2. - 1, nx) - xOffset
    elif nx % 2 == 1:
        x = np.linspace(-(nx-1)/2., (nx-1)/2., nx) - xOffset
    if ny % 2 == 0:
        y = np.linspace(-ny/2., ny/2. - 1, ny) - yOffset
    elif ny % 2 == 1:
        y = np.linspace(-(ny-1)/2., (ny-1)/2., ny) - yOffset
    dx = x[1] - x[0]
    [X, Y] = np.meshgrid(x, y)
    RHOS = np.sqrt(X*X + Y*Y)
    THETAS0 = np.arctan2(Y, X)
    THETAS = np.angle(np.exp(1j*(THETAS0 - angRotRad)))
    THETAS180 = np.angle(np.exp(1j*(THETAS0 - angRotRad - np.pi)))

    # Define which regions are =1, =0, or 0< and <1
    halfWindowWidth = np.sqrt(2.)*dx
    mask = -1*np.ones(RHOS.shape)
    if nSeg == 1:
        if angRot < 360:
            area1 = ((RHOS < rOutPix - halfWindowWidth) &
                     (RHOS > rInPix + halfWindowWidth) &
                     (THETAS < angOpenRad/2 - dThetaRad) &
                     (THETAS > -angOpenRad/2 + dThetaRad))
            mask[area1] = 1
            area0 = (((RHOS > rOutPix + halfWindowWidth) |
                     (RHOS < rInPix - halfWindowWidth))) |\
                    ((THETAS > angOpenRad/2 + dThetaRad) |
                     (THETAS < -angOpenRad/2 - dThetaRad))
            mask[area0] = 0
            grayInds = np.array(np.nonzero(mask == -1))

    elif nSeg == 2:
        if angRot < 360:
            area1 = ((RHOS < rOutPix - halfWindowWidth) &
                     (RHOS > rInPix + halfWindowWidth) &
                     (((THETAS < angOpenRad/2 - dThetaRad) &
                      (THETAS > -angOpenRad/2 + dThetaRad)) |
                     ((THETAS180 < angOpenRad/2 - dThetaRad) &
                     (THETAS180 > -angOpenRad/2 + dThetaRad))))
            mask[area1] = 1
            area0 = ((RHOS > rOutPix + halfWindowWidth) |
                     (RHOS < rInPix - halfWindowWidth)) |\
                    ((THETAS > angOpenRad/2 + dThetaRad) &
                     (THETAS < np.pi-angOpenRad/2 - dThetaRad)) |\
                    ((THETAS180 > angOpenRad/2 + dThetaRad) &
                     (THETAS180 < np.pi-angOpenRad/2 - dThetaRad))
            mask[area0] = 0
            grayInds = np.array(np.nonzero(mask == -1))

    # Nominal coordinates of the sub-pixels in a pixel
    dxHighRes = 1./float(nSubpixels)
    xUp = np.linspace(-(nSubpixels-1)/2., (nSubpixels-1)/2.,
                      nSubpixels)*dxHighRes
    [Xup, Yup] = np.meshgrid(xUp, xUp)

    # Compute the value between 0 and 1 of each edge pixel along the circle by
    # taking the mean of the binary subpixels.
    subpixelArray = np.zeros((nSubpixels, nSubpixels))
    for iInterior in range(grayInds.shape[1]):

        subpixelArray = 0*subpixelArray

        xCenter = X[grayInds[0, iInterior], grayInds[1, iInterior]]
        yCenter = Y[grayInds[0, iInterior], grayInds[1, iInterior]]
        RHOHighRes = np.sqrt((Xup+xCenter)**2 + (Yup+yCenter)**2)
        THETASHighRes0 = np.arctan2(Yup+yCenter, Xup+xCenter)
        THETASHighRes = np.angle(np.exp(1j*(THETASHighRes0 - angRotRad)))
        THETASHighRes180 = np.angle(np.exp(1j*(THETASHighRes0 -
                                               angRotRad - np.pi)))

        if nSeg == 1:
            edgeRegion = (RHOHighRes <= rOutPix) & (RHOHighRes >= rInPix) &\
                         (THETASHighRes <= angOpenRad/2) &\
                         (THETASHighRes >= -angOpenRad/2)

        elif nSeg == 2:
            edgeRegion = (RHOHighRes <= rOutPix) & (RHOHighRes >= rInPix) &\
                         (((THETASHighRes <= angOpenRad/2) &
                          (THETASHighRes >= -angOpenRad/2)) |
                          (THETASHighRes180 <= angOpenRad/2) &
                          (THETASHighRes180 >= -angOpenRad/2))

        subpixelArray[edgeRegion] = 1
        pixelValue = np.sum(subpixelArray)/float(nSubpixels*nSubpixels)
        mask[grayInds[0, iInterior], grayInds[1, iInterior]] = pixelValue

    return mask
