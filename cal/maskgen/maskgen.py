# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Module to hold functions for the MASKGEN package."""
import os
import numpy as np
from astropy.io import fits
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import rotate
from scipy.interpolate import RectBivariateSpline

from cal.maskgen.util.thinfilm import calc_complex_occulter
from cal.maskgen.mg_util import gen_annular_segments
from cal.util.findopt import find_optimum_2d
from cal.util.insertinto import insertinto as inin
from cal.util.loadyaml import loadyaml
from cal.util.math import ceil_odd
from cal.util.shapes import circle
import cal.util.check as check

from cal.pupilfit.pf_util import offcenter_crop

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))


def rotate_shift_downsample_amplitude_mask(
        maskIn, rotDeg, mag, xOffset, yOffset, padFac=1.2, flipx=False):
    """
    Flip, rotate, translate, and downsample a real-valued, 2-D mask.

    Parameters
    ----------
    maskIn : np.ndarray
        2-D array representing the amplitude mask to adjust.
    rotDeg : float
        Amount to rotate the array counterclockwise about the center pixel.
        Units of degrees.
    mag : float
        Magnification of the output mask compared to the input mask. Must be
        0 < mag <= 1. That is, the mask cannot be upsampled.
    xOffset, yOffset : float
        x- and y-offsets of the mask (after rotation) in output-sized pixels.
    padFac : float
        Factor by which to increase the array dimensions with zero padding.
        Array is zero padded in case the mask goes to the edge of the array;
        otherwise, artifacts can appear in the output array. padFac must be
        >= 1. Default value is 1.2.
    flipx : bool
        Flag to flip the mask left-right before doing the other operations
        (rotation, translation, and downsampling).

    Returns
    -------
    maskOut : np.ndarray
        2-D, even-sized, square array containing the altered mask.
    """
    check.twoD_array(maskIn, 'maskIn', TypeError)
    check.real_scalar(rotDeg, 'rotDeg', TypeError)
    check.real_positive_scalar(mag, 'mag', TypeError)
    if mag > 1:
        raise TypeError('This function does not perform upsampling.')
    check.real_scalar(xOffset, 'xOffset', TypeError)
    check.real_scalar(yOffset, 'yOffset', TypeError)
    check.real_positive_scalar(padFac, 'padFac', TypeError)
    if padFac < 1:
        raise TypeError('padFac must be >= 1.')
    check.boolean(flipx, 'flipx', TypeError)

    # Pad the input array to be a square with odd side lengths
    newSideLength = ceil_odd(padFac*np.max([maskIn.shape]))
    maskIn = inin(maskIn, (newSideLength, newSideLength))

    # Flip
    if flipx:
        maskIn = np.fliplr(maskIn)

    # Array sizes
    dxIn = 1.0
    dxOut = 1.0 / mag
    nMaskIn = maskIn.shape[0]
    nMaskOut = ceil_odd(nMaskIn*mag + 2 +
                        2.*np.max((np.abs(xOffset), np.abs(yOffset))))
    # 2 pixels added to guarantee the offset mask is fully contained in
    # the output array.

    # array-centered coordinates of input matrix [pupil diameters]
    x0 = np.arange(-(nMaskIn-1.)/2., (nMaskIn)/2., 1)*dxIn
    [X0, Y0] = np.meshgrid(x0, x0)
    R0 = np.sqrt(X0**2 + Y0**2)
    Window = 0*R0
    Window[R0 <= dxOut/2.] = 1
    Window = Window/np.sum(Window)

    # To get good grayscale edges, convolve with the correct window
    # before downsampling.
    f_window = ifft2(ifftshift(Window))*nMaskIn
    f_maskIn = ifft2(ifftshift(maskIn))*nMaskIn
    A = fftshift(fft2(f_window*f_maskIn))
    A = np.real(A)

    if not rotDeg == 0:
        A = rotate(A, -rotDeg, reshape=False)

    x1 = (np.arange(-(nMaskOut-1.)/2., nMaskOut/2., 1) - xOffset)*dxOut
    y1 = (np.arange(-(nMaskOut-1.)/2., nMaskOut/2., 1) - yOffset)*dxOut

    # RectBivariateSpline is faster in 2-D than interp2d
    interp_spline = RectBivariateSpline(x0, x0, A)
    Atemp = interp_spline(y1, x1)

    maskOut = np.zeros((nMaskOut+1, nMaskOut+1))
    maskOut[1::, 1::] = np.real(Atemp)

    return maskOut


def gen_hlc_occulter(lam, scaleWithWavelength, shapeOut, fnCalibData,
                     fnOccData, xOffset=0., yOffset=0., use_fourier=True,
                     data_path=LOCAL_PATH):
    """
    Generate a complex-valued HLC occulting mask.

    A complex-valued transmission of the HLC occulter is generated starting
    from a 2-D array giving the thickness of the PMGI layer and from scalars
    giving the thicknesses of the titanium and nickel layers. This assumes
    that there is no metal except underneath the PMGI.
    The order of operations is:
    1. compute complex transmission from material thicknesses
    1. rotate mask
    2. translate mask
    3. downsample mask

    To finish at the exact mask resolution desired, the downsampling is
    performed in two steps. First, rotate_shift_downsample_amplitude_mask is
    used to slightly downsample the starting, high-res mask representation.
    This intermediate resolution is chosen such that the ratio of the
    intermediate and final resolutions is exactly the ratio of two integers.
    Those two integers are used as the array side lengths (in pixels) so that
    the output of the FFT downsampling is exact. If use_fourier == True, then
    the Fourier downsampling is skipped and only regular interpolation is used
    for the entire downsampling operation.

    All filenames may be absolute or relative paths.  If relative in input
    arguments, they will be relative to the current working directory, not to
    any particular location in Calibration.  If relative within YAML files
    (e.g. the fnPMGI key in fnOccData), this will be relative to
    the path in the data_path argument.

    Parameters
    ----------
    lam : float
        Wavelength at which to generate the mask. Units of meters.
    scaleWithWavelength : bool
        Whether to have the FPM diameter shrink linearly with wavelength,
        as is needed for FFTs. If False, the diameter in pixels will be the
        same at each wavelength.
    shapeOut : array_like
        1-D array specifying the 2-D shape of the output array.
    fnCalibData : str
        Name of YAML file containing calibration data needed to generate
        the focal plane mask representation.
    fnOccData : str
        Name of YAML file containing data for the high-resolution occulter
        design file(s) from which to start.
    xOffset, yOffset : float
        x- and y-offsets of the mask (after rotation) in output-sized pixels.
    use_fourier: bool
        Flag whether to use Fourier downsampling as the final downsampling
        step. If False, use regular interpolation only. Default is True.
    data_path : str
        Directory to serve as a base for relative paths with YAML files.
        If not supplied, defaults to directory containing this function. If
        using versions of the YAMLs delivered with the repository, the default
        will point to data files also delivered with the repository.


    Returns
    -------
    occ : array_like
        complex-valued, 2-D array of the HLC occulter transmission

    """
    check.real_positive_scalar(lam, 'lam', TypeError)
    check.oneD_array(shapeOut, 'shapeOut', TypeError)
    if not len(shapeOut) == 2:
        raise TypeError('shapeOut must have length 2')
    check.positive_scalar_integer(shapeOut[0], 'shapeOut[0]', TypeError)
    check.positive_scalar_integer(shapeOut[1], 'shapeOut[1]', TypeError)
    check.boolean(scaleWithWavelength, 'scaleWithWavelength', TypeError)
    check.string(fnCalibData, 'fnCalibData', TypeError)
    check.string(fnOccData, 'fnOccData', TypeError)
    check.real_scalar(xOffset, 'xOffset', TypeError)
    check.real_scalar(yOffset, 'yOffset', TypeError)
    check.string(data_path, 'data_path', TypeError)
    check.boolean(use_fourier, 'use_fourier', TypeError)
    # All exceptions for the 2 YAML filenames are checked inside loadyaml

    # Load FPM calibtration data from YAML file
    inp = loadyaml(fnCalibData)
    ppl = inp['ppl']  # desired ending pixels per lambda_central/D
    rotDeg = inp['rotDeg']  # CCW rotation of the mask [degrees]
    aoi = inp['aoi']  # Angle of incidence of beam at FPM in instrument [deg]

    # Load occulter data from YAML file
    inp = loadyaml(fnOccData)
    rOccLamD = inp['rOccLamD']  # occulter radius in lambda_central/D
    fnPMGI = inp['fnPMGI']
    lam0 = inp['lam0']  # wavelength assumed in occulter design file [meters]
    hTi = inp['hTi']  # titanium height [meters]
    hNi = inp['hNi']  # nickel height [meters]
    dx = inp['dx']  # pixel width and height in occulter design file [meters]
    FN = inp['FN']  # F number assumed in occulter design file
    polState = inp['polState']  # 0 for s, 1 for p, 2 for mean of s and p
    # reference height away from substrate to use as the zero point for
    # phase calculation [waves]
    hRefCoef = inp['hRefCoef']

    padFac = inp['padFac']
    minOversizeFacForFFT = inp['minOversizeFacForFFT']

    # Checks on values from YAML files
    check.real_positive_scalar(ppl, 'ppl', TypeError)
    check.real_scalar(rotDeg, 'rotDeg', TypeError)
    check.real_scalar(aoi, 'aoi', TypeError)

    check.real_positive_scalar(lam0, 'lam0', TypeError)
    check.real_positive_scalar(hTi, 'hTi', TypeError)
    check.real_positive_scalar(hNi, 'hNi', TypeError)
    check.real_positive_scalar(dx, 'dx', TypeError)
    check.real_positive_scalar(FN, 'FN', TypeError)
    check.nonnegative_scalar_integer(polState, 'polState', TypeError)
    if polState not in (0, 1, 2):
        raise TypeError('polState must be 0, 1, or 2')
    check.real_positive_scalar(hRefCoef, 'hRefCoef', TypeError)
    check.real_nonnegative_scalar(aoi, 'aoi', TypeError)
    if aoi >= 90:
        raise TypeError('aoi must be <90 degrees.')
    check.real_positive_scalar(padFac, 'padFac', TypeError)
    if padFac < 1.0:
        raise TypeError('padFac must be >= 1.0')
    check.real_positive_scalar(minOversizeFacForFFT, 'minOversizeFacForFFT',
                               TypeError)
    if minOversizeFacForFFT < 2.0:
        raise TypeError('minOversizeFacForFFT must be >= 2.0')

    # Load high-res occulter PMGI thickness from Dwight
    pmgi = fits.getdata(os.path.join(data_path, fnPMGI))
    nFPM0 = ceil_odd(np.max(pmgi.shape))
    pmgi = inin(pmgi, (nFPM0, nFPM0))  # force to be square, odd-sized array
    footprint = np.zeros_like(pmgi, dtype=int)
    footprint[pmgi > 0] = 1

    # Find indices of occulter pixels and substrate-only pixels
    indOcc = np.nonzero(footprint.flatten() == 1)[0]
    indSubstrate = np.nonzero(footprint.flatten() == 0)[0]

    # Generate complex-valued occulter transmission at file resolution
    hRef = hRefCoef*lam0
    hTiVec = hTi*np.ones_like(indOcc)
    hNiVec = hNi*np.ones_like(indOcc)
    hPMGIVec = pmgi.flatten()[indOcc]
    tCoef, rCoef = calc_complex_occulter(lam, aoi, hTiVec, hNiVec, hPMGIVec,
                                         hRef, polState)
    tCoefSub, rCoefSub = calc_complex_occulter(lam, aoi, [0], [0], [0], hRef,
                                               polState)
    occStartFlat = np.zeros(nFPM0*nFPM0, dtype=complex)
    occStartFlat[indSubstrate] = tCoefSub[0]
    occStartFlat[indOcc] = tCoef
    occStart = occStartFlat.reshape((nFPM0, nFPM0))
    occStart = occStart/occStart[0, 0]

    # Compute intermediate resolution mask to be used for FFT downsampling.
    pplStart = (FN * lam0) / dx  # starting pixels per lambda_central/D
    if scaleWithWavelength:
        pplEnd = (lam0 / lam) * ppl
    else:
        pplEnd = ppl

    if use_fourier:

        nPadEnd = int(np.ceil(2*rOccLamD*pplEnd*minOversizeFacForFFT))
        nPadMiddle = int(np.floor((pplStart/pplEnd) * nPadEnd))
        pplMiddle = (nPadMiddle/nPadEnd) * pplEnd
        mag = pplMiddle / pplStart
        occMiddleReal = rotate_shift_downsample_amplitude_mask(
            np.real(occStart - occStart[0, 0]), rotDeg, mag,
            xOffset*(nPadMiddle/nPadEnd), yOffset*(nPadMiddle/nPadEnd), padFac)
        occMiddleImag = rotate_shift_downsample_amplitude_mask(
            np.imag(occStart - occStart[0, 0]), rotDeg, mag,
            xOffset*(nPadMiddle/nPadEnd), yOffset*(nPadMiddle/nPadEnd), padFac)
        occMiddle = occMiddleReal + 1j*occMiddleImag
        occMiddle = inin(occMiddle, (nPadMiddle, nPadMiddle)) + occStart[0, 0]

        # Fourier downsample
        occFT = inin(fftshift(ifft2(ifftshift(occMiddle))), (nPadEnd, nPadEnd))
        occEnd = fftshift(fft2(ifftshift(occFT)))
        occ = inin(occEnd - occStart[0, 0], shapeOut) + occStart[0, 0]

    else:

        # Interpolation downsampling only--no Fourier downsampling.
        mag = pplEnd / pplStart
        occReal = rotate_shift_downsample_amplitude_mask(
            np.real(occStart - occStart[0, 0]), rotDeg, mag,
            xOffset, yOffset, padFac)
        occImag = rotate_shift_downsample_amplitude_mask(
            np.imag(occStart - occStart[0, 0]), rotDeg, mag,
            xOffset, yOffset, padFac)
        occ = inin(occReal + 1j*occImag, shapeOut) + occStart[0, 0]

    return occ


def fit_focal_mask_mag_clocking(imageMeas,
                                xOffsetStarFromCenterPixel,
                                yOffsetStarFromCenterPixel,
                                fnOpticalParamsForFocalMaskCalib,
                                fnFitSettingsForFocalMaskCalib,
                                fnMaskDefForFocalMaskCalib):
    """
    Compute the magnification and clocking of the field stops or SPC FPMs.

    For field stops (on FSAM):
    Uses the two horizontally separated C-shaped field stops facing each other
    as fiducials. For the image used as the input, the FSAM must be moved to
    have the star centered halfway between the two C-shaped field stops. The
    C-shaped field stops are used because they are the only pair of stops in
    the FOV that are left-right symmetric, which helps avoid confusion when
    orienting the image.

    For shaped pupil bowtie focal plane masks (on FPAM):
    Uses two horizontally separated bowtie focal plane masks as fiducials. For
    the image used as the input, the FSAM must be moved to have the star
    halfway between the two bowtie masks. The bowtie masks are used because
    the pinholes are too small and the annular FPMs are too far apart. The
    bowties are rotated, so parity matters. The rotation angles of each bowtie
    mask are adjustable separately via the YAML file inputs.

    For hybrid Lyot focal plane masks (on FPAM):
    This algorithm does not work with the HLC focal plane masks.

    The alignment algorithm works by minimizing the squared difference between
    numerically generated fringes. The fringes are the squared FFT of either
    the estimated or measured pair of alignment stops. Squaring the FFTs is
    useful because then the pair of measured stops doesn't have to be centered
    in the array--the tip/tilt in phase goes away.

    Parameters
    ----------
    imageMeas : array_like
        2-D focal plane image with the star centered halfway between the two
        field stops or FPMs, which must be illuminated and visible.
    xOffsetStarFromCenterPixel, yOffsetStarFromCenterPixel : float
        Number of pixels in x and y that the star is offset from the center
        pixel of image.
    fnOpticalParamsForFocalMaskCalib : str
        Name of file containing CGI optical parameters needed for this
        calibration.
    fnFitSettingsForFocalMaskCalib : str
        Name of file containing tuning parameters for performing the clocking
        and magnification calibration of the specified mask array.
    fnMaskDefForFocalMaskCalib : str
        Name of file containing specifications of the imaged masks being used
        for calibration.

    Returns
    -------
    magEst : float
        Estimated magnification of the FPAM or FSAM image compared to what is
        expected.
    clockEst : float
        Estimated clocking of the FSs or FPMS compared to being horizontally
        oriented. Units of degrees.

    """
    check.twoD_array(imageMeas, 'spotMeas', TypeError)
    check.real_scalar(xOffsetStarFromCenterPixel, 'xOffsetStarFromCenterPixel',
                      TypeError)
    check.real_scalar(yOffsetStarFromCenterPixel, 'yOffsetStarFromCenterPixel',
                      TypeError)
    # Checks for the 3 filenames are all within loadyaml.

    inp = loadyaml(fnOpticalParamsForFocalMaskCalib)
    umPerPixExCAM = inp['umPerPixExCAM']
    magModel = inp['magModel']
    aoiDeg = inp['aoiDeg']

    inp = loadyaml(fnFitSettingsForFocalMaskCalib)
    nIter = inp['nIter']
    nClock = inp['nClock']
    clockMaxDeg = inp['clockMaxDeg']
    nMag = inp['nMag']
    DeltaMag = inp['DeltaMag']
    # nRow = inp['nRow']
    # nCol = inp['nCol']
    # nColToZero = inp['nColToZero']
    threshRatio = inp['threshRatio']
    nPixFFT = inp['nPixFFT']
    nPixOut = inp['nPixOut']
    radiusROI = inp['radiusROI']
    rotation = inp['rotation']

    inp = loadyaml(fnMaskDefForFocalMaskCalib)
    nSeg = inp['nSeg']
    rInUM = inp['rInUM']
    rOutUM = inp['rOutUM']
    angOpen = inp['angOpen']
    angRotRight = inp['angRotRight']
    angRotLeft = inp['angRotLeft']
    dTheta = inp['dTheta']
    lMaskSepUM = inp['lMaskSepUM']
    nSubpixels = inp['nSubpixels']

    # Crop out anything extra in the image
    # First re-center the image.
    # nMax = np.max([nRow, nCol])
    imageMeas = offcenter_crop(imageMeas, nPixFFT,
                               int(np.floor(imageMeas.shape[0]/2.0) +
                                   round(yOffsetStarFromCenterPixel)),
                               int(np.floor(imageMeas.shape[1]/2.0) +
                                   round(xOffsetStarFromCenterPixel)))

    # Make regions of interest (ROIs) where the masks should be, and
    # zero out the rest of the image.
    lMaskSepPix0 = (lMaskSepUM / umPerPixExCAM * magModel *
                    np.cos(np.radians(aoiDeg)))
    rotRad = np.radians(rotation)
    rotMat = np.array([[np.cos(rotRad), -np.sin(rotRad)],
                       [np.sin(rotRad), np.cos(rotRad)]])
    xyRight = rotMat @ np.array([[1], [0]]) * lMaskSepPix0/2  # pixels
    xyLeft = rotMat @ np.array([[-1], [0]]) * lMaskSepPix0/2  # pixels
    roiRight = circle(nPixFFT, nPixFFT, radiusROI, xyRight[0][0],
                      xyRight[1][0], nSubpixels=nSubpixels)
    roiLeft = circle(nPixFFT, nPixFFT, radiusROI, xyLeft[0][0],
                     xyLeft[1][0], nSubpixels=nSubpixels)
    imageMeas *= (roiRight + roiLeft)

    # Threshold the image to make it binary
    maxVal = np.max(imageMeas)
    thresh = maxVal*threshRatio
    imageMeas[imageMeas > thresh] = 1
    imageMeas[imageMeas <= thresh] = 0

    imageFT = fftshift(fft2(fftshift(imageMeas)))/nPixFFT
    imageFT2 = np.abs(imageFT)**2
    imageFT2 = imageFT2/np.max(imageFT2)  # normalize
    imageFT2crop = inin(imageFT2, (nPixOut, nPixOut))  # crop

    # Loop over mag and clocking values to make a grid of cost function values
    clockVec = np.linspace(-clockMaxDeg, clockMaxDeg, nClock)
    magVec = np.linspace(1-DeltaMag, 1+DeltaMag, nMag)
    deltaMag = 0  # initialize at 0
    deltaClock = 0  # initialize at 0

    for iter_ in range(nIter):
        costMat = np.zeros((nClock, nMag))

        # GENERATE THE MATRIX OF COST FUNCTION VALUES
        for iMag, magNew in enumerate((magVec-deltaMag)):

            lMaskSepPixTemp = lMaskSepPix0 * magNew
            rInPix = rInUM / umPerPixExCAM * magModel * magNew
            rOutPix = rOutUM / umPerPixExCAM * magModel * magNew

            for iClock, clockTempDeg in enumerate((clockVec-deltaClock)):

                # Mask centers
                rotTotal = np.radians(clockTempDeg + rotation)
                rotMat = np.array([[np.cos(rotTotal), -np.sin(rotTotal)],
                                   [np.sin(rotTotal), np.cos(rotTotal)]])
                xyRight = rotMat @ np.array([[1], [0]]) * lMaskSepPixTemp/2.0
                xyLeft = rotMat @ np.array([[-1], [0]]) * lMaskSepPixTemp/2.0

                # xMaskSepPix = np.cos(clockTempRad) * lMaskSepPixTemp
                # yMaskSepPix = np.sin(clockTempRad) * lMaskSepPixTemp

                # Generate the model-based masks.
                maskRight = gen_annular_segments(
                    nSeg, rInPix, rOutPix,
                    nPixFFT, nPixFFT, xyRight[0][0], xyRight[1][0],
                    angOpen, angRotRight+clockTempDeg+rotation,
                    nSubpixels=nSubpixels, dTheta=dTheta)

                maskLeft = gen_annular_segments(
                    nSeg, rInPix, rOutPix,
                    nPixFFT, nPixFFT, xyLeft[0][0], xyLeft[1][0],
                    angOpen, angRotLeft+clockTempDeg+rotation,
                    nSubpixels=nSubpixels, dTheta=dTheta)
                maskTemp = np.abs(maskRight + maskLeft)**2

                # Threshold the image to make it binary
                maxVal = np.max(maskTemp)
                thresh = maxVal*threshRatio
                maskTemp[maskTemp > thresh] = 1
                maskTemp[maskTemp <= thresh] = 0

                # Make cost function matrix
                maskFT = fftshift(fft2(fftshift(maskTemp)))/nPixFFT
                maskFT2 = np.abs(maskFT)**2
                maskFT2 = maskFT2/np.max(maskFT2)
                maskFT2Crop = inin(maskFT2, (nPixOut, nPixOut))

                costMat[iClock, iMag] = np.sum((maskFT2Crop - imageFT2crop)**2)

        # Find the mag and clocking that minimize the cost function.
        # At the first iteration, uses the minimum instead of a
        # quadratic fit in order to get a larger capture range.
        if iter_ == 0 and nIter > 1:
            bestInds = np.unravel_index(np.argmin(costMat), costMat.shape)
            xSol = magVec[bestInds[1]] - deltaMag
            ySol = clockVec[bestInds[0]] - deltaClock
        else:
            # Fit to a 3-D paraboloid to find best (x,y) center
            xSol, ySol = find_optimum_2d(magVec-deltaMag,
                                         clockVec-deltaClock, costMat,
                                         np.ones_like(costMat))

        deltaMag = 1 - xSol
        deltaClock = -ySol

    magEst = xSol
    clockEst = ySol

    return magEst, clockEst
