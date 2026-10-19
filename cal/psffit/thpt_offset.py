# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Tools to process relative CFAM filter data
"""

import numpy as np
from scipy.ndimage import shift, zoom

import cal.util.check as check
from cal.psffit.psffit import psffit
from cal.util.insertinto import insertinto
from cal.util.math import ceil_odd

GAUSS_STD = 4 # sigma to use in the gaussian created as a centration reference

def thpt_offset(imlist, refindex, lamlist):
    """
    Compute relative throughputs and offsets for images taken with a set of
    filters

    Example:
        x 0 0 0 0 0       x 0 0 0 0 0
          0 0 0 0 0         0 0 0 0 0
    ref = 0 0 1 0 0  targ = 0 0 0 0 0
          0 0 0 0 0         0 0 0 0 3
          0 0 0 0 0,        0 0 0 0 0

    x next to the (0, 0) element, with rows increasing down and cols increasing
     right
    relative throughput of target => 3
    relative offset of target (row, col) => (1, 2)

    Arguments:
     imlist: list of ndarrays.  Each array should be a real-valued 2D array,
      and all arrays should be the same size.
     refindex: index of the imlist element which is the zero-point (no relative
      offset or throughput).  >= 0, and must be < len(imlist).
     lamlist: list of wavelengths (positive real scalars in nanometers).  Same
      length list as imlist; each wavelength is associated with the image of
      the same index.  Used to compensate for radial PSF scaling with
      wavelength.

    Returns:
     2 lists, of the same length of imlist:
      - list of relative throughputs.  The refindex element of this list will
        be 1.0.
      - list of relative offsets, as 2-tuples (row, col).  The refindex element
        of this list will be (0, 0).

    """

    # Check inputs
    if not isinstance(imlist, list):
        raise TypeError('imlist must be a list')
    for index, im in enumerate(imlist):
        check.twoD_array(im, 'im[' + str(index) + ']', TypeError)
        check.real_array(im, 'im[' + str(index) + ']', TypeError)
        if index == 0:
            s = im.shape
        else:
            if im.shape != s:
                raise TypeError('imlist arrays must be the same size')
            pass
        pass

    check.nonnegative_scalar_integer(refindex, 'refindex', TypeError)
    if refindex >= len(imlist):
        raise TypeError('refindex must be smaller than len(imlist)')

    if not isinstance(lamlist, list):
        raise TypeError('lamlist must be a list')
    for index, lam in enumerate(lamlist):
        check.real_positive_scalar(lam, 'lamlist[' + str(index) + ']',
                                   TypeError)
    if len(lamlist) != len(imlist):
        raise TypeError('lamlist and imlist must be the same length')

    nrow, ncol = imlist[refindex].shape

    # Common-mode for all wavelengths:
    #  - Simple centered gaussian template to serve as zero-point reference
    xx, yy = np.meshgrid(np.arange(nrow)-nrow//2, np.arange(ncol)-ncol//2)
    gauss = np.exp(-(xx**2 + yy**2)/(2*GAUSS_STD**2))
    ref_recenter, _ = psffit(gauss, imlist[refindex])
    im2ref = shift(imlist[refindex].astype('float64'),
                   (-ref_recenter[0], -ref_recenter[1]))

    thpt_list = [None]*len(imlist)
    offset_list = [None]*len(imlist)

    for index, im in enumerate(imlist):

        im = im.astype('float64')

        # Use psffit to extract offsets (but not amplitudes yet)
        shifts, _ = psffit(imlist[refindex].astype('float64'), im)
        offset_list[index] = tuple(shifts)

        # Co-center all PSFs
        im2 = shift(im, (-shifts[0]-ref_recenter[0],
                         -shifts[1]-ref_recenter[1]))

        # Rescale all PSFs to refindex sampling
        zscale = 1.0*lamlist[refindex]/lamlist[index]

        # Force odd so center is at center pixel for zoom
        im2_odd = insertinto(im2, (ceil_odd(im2.shape[0]),
                                   ceil_odd(im2.shape[1])))
        im3 = zoom(im2_odd, zscale)
        im3 /= np.sum(im3)
        im3 *= np.sum(im2) # preserve energy
        im3 = insertinto(im3, im2.shape)

        # Come back to psffit to get amplitude now that we rescaled
        _, thpt_list[index] = psffit(im2ref, im3)

        pass

    return thpt_list, offset_list
