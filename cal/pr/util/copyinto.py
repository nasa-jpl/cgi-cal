# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Utility routine for copying a subarray of one array into a different array
"""

import numpy as np

import cal.util.check as check

def copyinto(insertarr, xy, nout):
    """
    Insert a subarray from one matrix centered into another square array

    Output array is square and nout x nout.  If the input is smaller along an
    axis, it will be zero-padded; if it is larger, it will be truncated.  This
    function will always take as much of insertarr as it can, up to a max size
    region of nout x nout centered at the pixel defined by xy.

    Arguments:
     insertarr: 2D array containing raw image
     xy: two-element iterable of integers giving the location of the center
      pixel as (cols, rows). Must be within the size of insertarr.  The
      inverted convention is due to the legacy use where the image center was
      computed from a image display.  In these displays, x is the number of
      columns and y is the number of rows.
     nout: integer number of elements along one side of a square output array

    Returns:
     nout x nout array with a subregion centered at xy from insertarr

    """

    # Check inputs
    check.twoD_array(insertarr, 'insertarr', TypeError)
    try:
        if len(xy) != 2:
            raise TypeError('Output dimensions must have 2 elements')
        pass
    except TypeError: # not iterable
        raise TypeError('xy must be an iterable') # reraise with better msg
    check.nonnegative_scalar_integer(xy[0], 'xy[0]', TypeError)
    check.nonnegative_scalar_integer(xy[1], 'xy[1]', TypeError)
    check.positive_scalar_integer(nout, 'nout', TypeError)

    # upper range check (did < 0 already above)
    ccol, crow = xy
    if crow >= insertarr.shape[0] or ccol >= insertarr.shape[1]:
        raise TypeError('xy must correspond to an element in insertarr')

    # Identify start and end indices for a nout x nout area within insertarr
    # centered on (crow, ccol).  (May be out of range to start, that's fine.)
    #
    # These are meant to be uses as start:end, so end will be +1 to the
    # internal indices as end is exclusive.  (e.g. 0:5 -> 0, 1, 2, 3, 4)
    rout = [0, nout]
    cout = [0, nout]
    rinsert = [crow - nout//2, crow + nout - nout//2]
    cinsert = [ccol - nout//2, ccol + nout - nout//2]
    # Stupid bug tricks for a future reader: original was [-nout//2 + crow,
    # nout - nout//2 + crow]...but this got the wrong dimensions because
    # negation has higher precedence than floor division, subtraction has
    # lower precedence than floor division, and -(x//2) != (-x)//2 for odd x.

    # clip ranges so that rinsert remains within the bounds of insertarr
    # clip both together so the subarray remains the same size
    if rinsert[0] < 0:
        dr = 0 - rinsert[0]
        rinsert[0] += dr
        rout[0] += dr
        pass
    if rinsert[1] > insertarr.shape[0]:
        dr = rinsert[1] - insertarr.shape[0]
        rinsert[1] -= dr
        rout[1] -= dr
        pass

    if cinsert[0] < 0:
        dc = 0 - cinsert[0]
        cinsert[0] += dc
        cout[0] += dc
        pass
    if cinsert[1] > insertarr.shape[1]:
        dc = cinsert[1] - insertarr.shape[1]
        cinsert[1] -= dc
        cout[1] -= dc
        pass

    # Grab subregion and stuff it into a nout-sized matrix, zero-padding if
    # necessary
    basearr = np.zeros((nout, nout), dtype=insertarr.dtype)
    basearr[rout[0]:rout[1], cout[0]:cout[1]] = \
      insertarr[rinsert[0]:rinsert[1], cinsert[0]:cinsert[1]]
    return basearr
