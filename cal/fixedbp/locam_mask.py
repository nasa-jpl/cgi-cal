# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Utility to create a software mask for LOCAM."""
import numpy as np

import cal.util.check as check

def compute_locam_mask(masklist):
    """
    Compute a software mask for LOCAM to indicate which pixels to use.

    Given one or more predefined boolean arrays which indicate LOCAM
    pixels to keep or discard, this function will compute a single LOWFSC mask
    which incorporates all of them.

    True values in the output array are good pixels.

    Parameters
    ----------
    masklist : list
        list of 2D boolean arrays, all of the same dimension.  List length
        must be >= 1.  For each of these arrays, True means to use a pixel, and
        False means to not use a pixel.

    Returns
    -------
    locam_mask : array_like
        2-D boolean array indicating which pixels to use from LOCAM images.
        Array is same shape as all arrays in masklist. Pixels to use are True.

    """
    # Check inputs
    if not isinstance(masklist, list):
        raise TypeError('masklist must be a list')
    if len(masklist) == 0:
        raise TypeError('masklist cannot be empty')

    check.twoD_array(masklist[0], 'masklist[0]', TypeError)
    mshape = masklist[0].shape
    for index, mask in enumerate(masklist):
        check.twoD_array(mask, 'masklist[' + str(index) + ']', TypeError)
        if mask.dtype != bool:
            raise TypeError('masklist[' + str(index) + '] must be bool array')
        if mask.shape != mshape:
            raise TypeError('All elements of masklist must have same shape')
        pass

    # Want only common True, so use repeated AND
    locam_mask = np.ones(mshape, dtype=bool)
    for mask in masklist:
        locam_mask = np.logical_and(locam_mask, mask)
        pass

    return locam_mask
