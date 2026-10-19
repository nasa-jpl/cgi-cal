# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Contains validators to check DM settings for voltage bounds, neighbor rules,
and tied and dead actuators.  Checks only; does not fix.
"""

import numpy as np

from . import check
from .flat_tie import checktie, checkflat

MAXV = 110.0
BITFACTOR = 2**16
VMIN = 0.0

def check_valid_dn(dn, flatmap, tie, vmax=100.0, vlat=50.0, vdiag=75.0):
    """
    Check validity of a setting in DNs

    The setting is converted back to EUs before checking as per check_valid_eu,
    as all constraints are applied in EU space.

    The following items are checked:
      - EU voltages are >= 0V
      - EU voltages are <= vmax (default 100V)
      - EU voltage differences between laterally-adjacent actuators are <=
       vlat (default 50V) after a flatmap is subtracted
      - EU voltage differences between diagonally-adjacent actuators are <=
       vdiag (default 75V) after a flatmap is subtracted
      - All sets of actuators that are marked as tied together in the tie map
       are at the same voltage setting
      - All actuators that are marked as dead in the tie map are at 0V.

    If all pass, this function returns True, otherwise False.

    Arguments:
     dn: 2D array of 16-bit integers (data number or DNs)
     flatmap: a 2D array of floating-point voltages, of the same size as
      `volts`.  This array represents a physically-flat DM surface.
     tie: a 2D array of integers, of the same size as `volts`, which can take
      on the values 0, -1, or consecutive integers 1 -> N.

    Keyword Arguments:
     vmax: maximum commandable voltage, in volts.  Floating-point scalar, must
      be > 0. Defaults to 100V, which is the max voltage
      for CGI.
     vlat: maximum allowable voltage differential between laterally-adjacent
      actuators, in volts.  Floating-point scalar > 0.  Defaults to 50V, which
      is the CGI requirement.
     vdiag: maximum allowable voltage differential between diagonally-adjacent
      actuators, in volts.  Floating-point scalar > 0.  Defaults to 75V, which
      is the CGI requirement.

    Returns:
     True if the DN setting is consistent with all voltage constraints, False
     otherwise

    """
    # All inputs except this one are passthroughs, let _eu handle them
    check.twoD_array(dn, 'dn', TypeError)

    eu = dn_to_eu(dn)
    return check_valid_eu(eu, flatmap, tie, vmax=vmax, vlat=vlat, vdiag=vdiag)


def check_valid_eu(eu, flatmap, tie, vmax=100.0, vlat=50.0, vdiag=75.0):
    """
    Check validity of a setting in engineering units

    The following items are checked:
      - EU voltages are >= 0V
      - EU voltages are <= vmax (default 100V)
      - EU voltage differences between laterally-adjacent actuators are <=
       vlat (default 50V) after a flatmap is subtracted
      - EU voltage differences between diagonally-adjacent actuators are <=
       vdiag (default 75V) after a flatmap is subtracted
      - All sets of actuators that are marked as tied together in the tie map
       are at the same voltage setting
      - All actuators that are marked as dead in the tie map are at 0V.

    If all pass, this function returns True, otherwise False.

    Arguments:
     eu: 2D array of floating-point voltages (engineering units or EUs)
     flatmap: a 2D array of floating-point voltages, of the same size as
      `volts`.  This array represents a physically-flat DM surface.
     tie: a 2D array of integers, of the same size as `volts`, which can take
      on the values 0, -1, or consecutive integers 1 -> N.

    Keyword Arguments:
     vmax: maximum commandable voltage, in volts.  Floating-point scalar, must
      be > 0. Defaults to 100V, which is the max voltage
      for CGI.
     vlat: maximum allowable voltage differential between laterally-adjacent
      actuators, in volts.  Floating-point scalar > 0.  Defaults to 50V, which
      is the CGI requirement.
     vdiag: maximum allowable voltage differential between diagonally-adjacent
      actuators, in volts.  Floating-point scalar > 0.  Defaults to 75V, which
      is the CGI requirement.

    Returns:
     True if the EU setting is consistent with all voltage constraints, False
     otherwise

    """
    # Check inputs
    check.twoD_array(eu, 'eu', TypeError)
    check.twoD_array(flatmap, 'flatmap', TypeError)
    check.twoD_array(tie, 'tie', TypeError)
    if eu.shape != flatmap.shape:
        raise TypeError('eu and flatmap must have the same shape')
    if eu.shape != tie.shape:
        raise TypeError('eu and tie must have the same shape')

    check.real_positive_scalar(vmax, 'vmax', TypeError)
    check.real_positive_scalar(vlat, 'vlat', TypeError)
    check.real_positive_scalar(vdiag, 'vdiag', TypeError)

    # enforce tie and flat formatting
    if not checktie(tie):
        raise ValueError('tie must have values 0, -1, or consecutive ' +
                         'integers 1 -> N')
    if not checkflat(flatmap, VMIN, vmax, tie):
        raise ValueError('flatmap must be <= vmax, >= VMIN, have all tied ' +
                         'actuators tied already, and have all dead ' +
                         'actuators = 0V')

    # Validate input data array
    if not check_valid(array=eu, plus_limit=vlat, diag_limit=vdiag,
                       high_limit=vmax, low_limit=VMIN, dmflat=flatmap):
        return False
    if not check_tie_dead(volts=eu, tie=tie):
        return False

    return True


def dn_to_eu(dn):
    """
    Converts an array in DNs (uint16s) into floating-point volts (EUs or
    engineering units)

    Satisfies EU = DN*110.0/2**16.

    Arguments:
     dn: 2D array

    Returns:
     array of floating-point voltages of the same size as dn

    """
    check.twoD_array(dn, 'dn', TypeError)

    return dn*MAXV/BITFACTOR


def eu_to_dn(eu):
    """
    Converts an array in engineering units (EUs, floating-point volts)
    into uint16s

    Satisfies int(eu*2**16/110.0).  If greater than 110-1/2**16, or less than
    0, i.e. outside the range representable in a uint16 without overflow, will
    raise a ValueError.

    Arguments:
     eu: 2D array

    Returns:
     array of 16-bit unsigned integers of the same size as eu

    """
    check.twoD_array(eu, 'eu', TypeError)
    # check range before cast as overflows will happen silently
    if (eu > (MAXV - 1/BITFACTOR)).any() or (eu < 0).any():
        raise ValueError('eu inputs out of DN range')

    return np.uint16(eu*BITFACTOR/MAXV)


def check_valid(array, plus_limit, diag_limit, high_limit=None,
                low_limit=None, dmflat=None):
    """
    Checks if all array elements exceed the difference limits along the
    plus and diagonal axis.

    Voltage constraints (caps/neighbor rules) only; does not check ties.  This
    function is also used for dmsmooth checkout, so not worth trying to force
    them in.  check_tie_dead() will cover tied and dead actuators.

    Arguments:
     array: 2D voltage array of interest
     plus_limit: the limit on the absolute value difference of elements
      along the plus dimension.
     diag_limit: the limit on the absolute value difference of elements
      along the diagonal dimension.

    Keyword Arguments:
     high_limit: the maximum value of the arrays, or None.  If None, assumes
      there are no upper voltage caps
     low_limit: the minimum value of the arrays, or None.  If None, assumes
      there are no lower voltage caps
     dmflat: 2D voltage array of the same size as 'array' representing the set
      of voltages which make the DM surface flat, or None.  These may not
      necessarily be uniform.  The requirement is that neighbor rules are
      obeyed after the subtraction of a flatmap.  If None, the flatmap is
      treated as a uniform array of 0V.

    Returns:
     True if all elements pass the test, False otherwise

    """

    # Check input formats
    check.twoD_array(array, 'array', TypeError)
    check.real_positive_scalar(plus_limit, 'plus_limit', TypeError)
    check.real_positive_scalar(diag_limit, 'diag_limit', TypeError)

    if high_limit is not None:
        check.real_scalar(high_limit, 'high_limit', TypeError)
        pass
    if low_limit is not None:
        check.real_scalar(low_limit, 'low_limit', TypeError)
        pass

    if dmflat is not None:
        check.twoD_array(dmflat, 'dmflat', TypeError)
        pass
    else:
        dmflat = np.zeros_like(array)
        pass

    # Raise exceptions if out of bounds high/low
    if (high_limit is not None) and  (low_limit is not None):
        lims = _check_high_low_limit(array, high_limit=high_limit,
                                     low_limit=low_limit)
    elif high_limit is not None:
        lims = _check_high_low_limit(array, high_limit=high_limit,
                                     low_limit=-np.inf)
    elif low_limit is not None:
        lims = _check_high_low_limit(array, high_limit=np.inf,
                                     low_limit=low_limit)
    else:
        # No voltage bounds supplied, so they can't fail
        lims = True
        pass

    if not lims:
        return False

    fromflat = array - dmflat

    # no row violations
    rok = (np.abs(fromflat[1:, :] - fromflat[:-1, :]) <= plus_limit).all()
    # no col violations
    cok = (np.abs(fromflat[:, 1:] - fromflat[:, :-1]) <= plus_limit).all()
    # no diag violations
    d1ok = (np.abs(fromflat[1:, 1:] - fromflat[:-1, :-1]) <= diag_limit).all()
    d2ok = (np.abs(fromflat[1:, :-1] - fromflat[:-1, 1:]) <= diag_limit).all()

    isgood = rok and cok and d1ok and d2ok
    return isgood


def check_tie_dead(volts, tie):
    """
    Checks that all tied actuator groups are at the same voltage, and that all
     the dead actuators are at zero volts.

    Arguments:
     volts: 2D array of floating-point voltages to check
     tie: 2D array of integers of the same size as volts.  tie should be -1, 0,
      or the integer range 1-N for some N (with no gaps).  -1 indicates a dead
      actuator; integers > 0 denote tie groups.

    Returns:
     True if all tied groups are at the same voltage, and all dead actuators
      are 0V.  False otherwise.

    """

    check.twoD_array(volts, 'volts', TypeError)
    check.twoD_array(tie, 'tie', TypeError)
    if volts.shape != tie.shape:
        raise TypeError('volts and tie must be the same shape')

    # Get list of groups
    tienumset = set(tie.ravel())
    for t in tienumset:
        if t == -1:
            if not (volts[tie == t] == 0).all():
                return False
            continue
        elif t == 0:
            # not tied or dead, skip
            continue
        else:
            tmp = set(volts[tie == t])
            if len(tmp) != 1:
                return False
            continue
        pass

    return True




#----------------------------
# Internal utility functions
#----------------------------

def _check_high_low_limit(array, high_limit, low_limit):
    """
    Checks if an array is between two values, inclusive

    Inputs:
     array: 2D voltage array
     high_limit: maximum voltage to compare against
     low_limit: minimum voltage to compare against.  Must be lower than
      high_limit

    Outputs:
     Returns True if array is in bounds, otherwise returns False

    """

    check.twoD_array(array, 'array', TypeError)
    check.real_scalar(high_limit, 'high_limit', TypeError)
    check.real_scalar(low_limit, 'low_limit', TypeError)
    if low_limit > high_limit:
        raise ValueError('low_limit must be lower than high_limit')

    if (array >= low_limit).all() and (array <= high_limit).all():
        return True
    else:
        return False
