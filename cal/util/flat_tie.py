# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Validation functions for flatmaps and tie matrices"""

import numpy as np

from . import check

def checktie(tie):
    """
    Check whether a tie matrix satisfies the format constraints
     - tie should be -1, 0, or the integer range 1-N for some N (with no gaps)

    Arguments:
     tie: 2D tie matrix to check

    Returns:
     True if valid, False if not

    """
    check.twoD_array(tie, 'tie', TypeError)

    tienumset = set(tie.ravel())
    tmp = tienumset - {0, -1} # -1/0 can be present or not
    vs = set(range(1, len(tmp)+1)) # +1 to len because it's exclusive at top
    if tmp == vs:
        return True
    else:
        return False


def checkflat(flatmap, vmin, vmax, tie):
    """
    Check whether a flatmap matrix satisfies the format constraints
     - flatmap should be >= vmin, <= vmax, have all ties at the same voltage,
      and have all dead actuators at 0V voltage.

    Arguments:
     flatmap: 2D flat matrix to check
     vmin: min voltage, used in check
     vmax: max voltage, used in check
     tie: 2D tie matrix to use in check, same size as flatmap

    Returns:
     True if valid, False if not

    """
    check.twoD_array(flatmap, 'flatmap', TypeError)
    # use real_scalar here to not encode CGI specifics in this simple fn
    check.real_scalar(vmin, 'vmin', TypeError)
    check.real_scalar(vmax, 'vmax', TypeError)
    check.twoD_array(tie, 'tie', TypeError)
    if tie.shape != flatmap.shape:
        raise TypeError('flatmap and tie must be the same shape')

    if not checktie(tie):
        raise ValueError('tie matrix contents do not match tie spec')

    if (flatmap < vmin).any():
        return False
    if (flatmap > vmax).any():
        return False
    if (flatmap[tie == -1] != 0).any():
        return False

    tienumset = set(tie.ravel())
    tmp = tienumset - {0, -1}
    for t in tmp:
        if len(np.unique(flatmap[tie == t])) != 1:
            # non-identical values implies more than one element
            return False
        pass

    return True
