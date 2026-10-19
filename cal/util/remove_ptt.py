# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Functions to remove piston, tip, and tilt from a beam."""
import os

from astropy.io import fits
import numpy as np

from cal.ampthresh.ampthresh import ampthresh
from cal.psffit.psffit import psffit
from . import check, mft
from .insertinto import insertinto as inin

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FN_PSFFIT_TUNING = os.path.join(
    HERE, 'testdata','params_psffit_for_tip_tilt_removal.yaml')


def remove_ptt_directly(arrayToFit, mask):
    """
    Remove piston and x- and y-ramp terms from an array directly.

    Parameters
    ----------
    arrayToFit : array_like
        2-D array of values to fit.
    mask : array_like
        2-D boolean mask of which pixels to use in "arrayToFit".
        Must be same size as "arrayToFit".

    Returns
    -------
    arrayOut : array_like
        Input array with piston, tip, and tilt subtracted.
    """
    check.twoD_array(arrayToFit, 'arrayToFit', ValueError)
    check.twoD_array(mask, 'mask', ValueError)
    if arrayToFit.shape != mask.shape:
        raise ValueError('arrayToFit and mask must have same shape')

    maskBool = np.asarray(mask).astype(bool)
    nPix = int(np.sum(mask))
    ny = arrayToFit.shape[0]
    nx = arrayToFit.shape[1]
    xVec = np.arange(-nx/2., nx/2.)/nx
    yVec = np.arange(-ny/2., ny/2.)/ny

    #  Set the basis functions
    ONES = np.ones((ny, nx))
    [X, Y] = np.meshgrid(xVec, yVec)
    f0 = X
    f1 = Y
    f2 = ONES

    # Write as matrix equation and solve for coefficients
    A = np.concatenate((f0[maskBool].reshape((nPix, 1)),
                        f1[maskBool].reshape((nPix, 1)),
                        f2[maskBool].reshape((nPix, 1))), axis=1)
    y = arrayToFit[maskBool].flatten()
    temp = np.linalg.lstsq(A, y, rcond=None)
    coeffs = temp[0]
    a, b, c = coeffs[0:3]
    arrayOut = maskBool * (arrayToFit - (a*X + b*Y + c*ONES))

    return arrayOut


def remove_ptt_at_focus(
        amp, ph, ppl, diam_pupil_pix, ft_dir, fn_ref_psf,
        fn_psffit_tuning=DEFAULT_FN_PSFFIT_TUNING, nCrop=100, nBin=21, nIter=2,
        ):
    """
    Remove piston, tip, and tilt at the focus given a pupil-plane E-field.

    Remove tip and tilt from phase retrieval pupil data by centering the PSF
    of the pupil E-field's Fourier transform. Remove piston by making the
    angle of the mean E-field in the illuminated part zero.

    Parameters
    ----------
    amp : array_like
        2-D array of the amplitude at the pupil plane from a phase retrieval.
    ph : array_like
        2-D array of the phase at the pupil plane from a phase retrieval.
        This can be the original wrapped phase or the unwrapped phase; the
        algorithms used in this function operate on complex E-field, so either
        works.
    ppl : float
        Pixels per lambda/D to use in the focal plane. If just wanting the
        output ph_no_ptt, then use any value larger than Nyquist sampled that
        matches the reference PSF. But if using this function to get x_offset
        and y_offset, then make sure to use the same ppl value as is measured
        in the optical system so that it is consistent.
    diam_pupil_pix : float
        Diameter of the beam at the pupil plane (what is in amp). Units of
        pixels.
    ft_dir : str
        Direction of the Fourier transform. Options are 'forward' or 'reverse'.
        Forward means a Fourier transform is used and reverse means an inverse
        Fourier transform is used.
    fn_ref_psf : str
        Absolute path to the FITS file of the reference PSF.
        Should be at or very close to the same resolution as ppl.
    fn_psffit_tuning : str, optional
        Relative or absolute path to the tuning YAML file for the internal
        calls to psffit(). The default is in
        testdata/params_psffit_for_tip_tilt_removal.yaml.
    nCrop : int, optional
        Number of points across to keep in the focal plane after Fourier
        transforming to speed up PSFFIT. The default is 100.
    nBin : int, optional
        Number of bins used when making a histogram of pupil values. Passed to
        ampthresh() directly. The default is 21.
    nIter : int, optional
        Number of iterations to loop the PSF re-centering. The default is 2.

    Returns
    -------
    ph_no_ptt : array_like
        2-D array of the input phase map but with piston, tip, and tilt
        removed. No phase unwrapping is performed in this function, so the
        np.angle() calls performed within this function to strip the phase
        from the complex E-field may result in new phase wrapping.        
    bmask : array_like
        2-D array of the binary software mask used for isolating the beam
        location. Constructed by thresholding the input 2-D amplitude map, amp.
    x_offset : float
        The original, pre-correction estimated x-offset of the PSF at the
        given focal plane resolution ppl and for the given Fourier transform
        direction ft_dir.
    y_offset : float
        The original, pre-correction estimated y-offset of the PSF at the
        given focal plane resolution ppl and for the given Fourier transform
        direction ft_dir.

    """
    check.twoD_array(amp, 'amp', TypeError)
    check.twoD_array(ph, 'ph', TypeError)
    check.real_positive_scalar(ppl, 'ppl', TypeError)
    check.real_positive_scalar(diam_pupil_pix, 'diam_pupil_pix', TypeError)
    check.string(ft_dir, 'ft_dir', TypeError)
    check.string(fn_ref_psf, 'fn_ref_psf', TypeError)
    check.string(fn_psffit_tuning, 'fn_psffit_tuning', TypeError)
    check.positive_scalar_integer(nCrop, 'nCrop', TypeError)
    check.positive_scalar_integer(nBin, 'nBin', TypeError)
    check.positive_scalar_integer(nIter, 'nIter', TypeError)
    if not (ft_dir == 'forward' or ft_dir == 'reverse'):
        raise ValueError("ft_dir must be 'forward' or 'reverse'.")

    fn_ref_psf = os.path.expandvars(fn_ref_psf)
    fn_psffit_tuning = os.path.expandvars(fn_psffit_tuning)
    if not os.path.isabs(fn_ref_psf):
        raise ValueError('fn_ref_psf must be an absolute path')

    # Enforce same shape for amp and ph
    shape0 = ph.shape
    nx = shape0[1]
    ny = shape0[0]
    amp = inin(amp, shape0)

    epup0 = amp * np.exp(1j*ph)
    efoc0 = mft.do_mft(epup0, (nCrop, nCrop), ppl, diam_pupil_pix,
                       direction=ft_dir)

    ref_img = fits.getdata(fn_ref_psf)
    ref_img = inin(arr0=ref_img, outsize=(nCrop, nCrop))

    ifoc0 = np.abs(efoc0)**2
    ifoc0 = inin(arr0=ifoc0, outsize=(nCrop, nCrop))

    shifts, amp_ratio = psffit(ref_img, ifoc0, config_file=fn_psffit_tuning)
    x_offset = shifts[1]
    y_offset = shifts[0]

    xs = np.linspace(-0.5, 0.5, nx) * (nx/diam_pupil_pix)/ppl
    ys = np.linspace(-0.5, 0.5, ny) * (ny/diam_pupil_pix)/ppl
    XS, YS = np.meshgrid(xs, ys)

    for ii in range(nIter):

        if ft_dir == 'forward':
            epup1 = epup0 * np.exp(-2*np.pi*1j*(x_offset*XS + y_offset*YS))
        elif ft_dir == 'reverse':
            epup1 = epup0 * np.exp(2*np.pi*1j*(x_offset*XS + y_offset*YS))
        else:
            raise ValueError("ft_dir must be 'forward' or 'reverse'.")

        efoc1 = mft.do_mft(epup1, (nCrop, nCrop), ppl, diam_pupil_pix,
                           direction=ft_dir)
        ifoc1 = np.abs(efoc1)**2
    
        shifts, amp_ratio = psffit(ref_img, ifoc1, config_file=fn_psffit_tuning)
        x_offset_new = shifts[1]
        y_offset_new = shifts[0]

        x_offset += x_offset_new
        y_offset += y_offset_new

    # Use a software mask instead of amp to give equal weighting to all
    # illuminated pixels before computing the mean E-field for determining
    # the piston term to remove.
    bmask = ampthresh(pupilMap=amp, nBin=nBin)
    epupSynthetic = bmask.astype(float) * np.exp(1j*np.angle(epup1))
    mean_field = np.mean(epupSynthetic[bmask])
    epup2 = epup1 * np.exp(-1j*np.angle(mean_field))
    ph_no_ptt = np.angle(epup2)

    return ph_no_ptt, bmask, x_offset, y_offset
