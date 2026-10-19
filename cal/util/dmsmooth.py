# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Routine to constrain DM voltages
"""

import numpy as np

from . import check

VMIN = 0 # define min voltage = 0V, also dead-actuator volts, also = DAC 0x0000

def _sign_array(x):
    """Fast scalar sign function for arrays"""
    out = np.zeros(x.shape, dtype=int)
    out[x > 0] = 1
    out[x < 0] = -1
    return out


def fix_nr(dmin, vneighbor, margin, r1=slice(None), c1=slice(None),
           r2=slice(None), c2=slice(None)):
    """
    Take a pair of columns and/or rows, check for neighbor rule violations
    between them, and fix them.
    """
    dmout = dmin.copy()

    # Subtract adjacent rows/cols
    diff = dmout[r2, c2] - dmout[r1, c1]

    # Find if any neighbor rule violations exist. Use mask to exclude good
    # neighbors from correction.
    diff_mask = np.abs(diff) > vneighbor - margin
    diff *= diff_mask

    # Split excess in half, with a little extra margin to keep
    # numerical errors from making this reappear.  2x margin to
    # overshoot the correction slightly so this doesn't pop up
    # again.
    delta = (diff - _sign_array(diff) * (vneighbor - 2*margin))/2.

    # Half on each side of the violation
    dmout[r1, c1] += delta
    dmout[r2, c2] -= delta

    # Return a mask for all the elements that were fixed
    fix_mask = dmout != dmin

    return dmout, fix_mask


def dmsmooth(dmin, vmax, vquant, vneighbor, vcorner, dmflat=None):
    """
    Modify a DM setting so that it obeys neighbor rules and voltages limits

    This implementation is conservative: it will produce settings that always
    obey neighbor rules after the floating-point DM specification has been
    quantized onto the n-bit DAC.  All neighbor-rule and voltage bounds will be
    at least 2 LSBs away from their threshold.

    This method will produce a result that is safe, but not necessarily
    optimal for any particular definition of optimality.

    This method is idempotent: running the output through the function a second
    time with identical settings will give an identical output to the original
    function.

    NOTE: this function assumes that the minimum commandable voltage is equal
    to the dead actuator voltage == 0V, and this also corresponds to 0x0000 in
    the DAC (so exactly the bottom value).  All of these things are true for
    CGI in its current implementation, but if this were to be revisited in
    another use case where these assumptions are not valid, then the logic here
    needs to be revisited.  (In particular, dead actuator voltage < minimum
    commandable voltage makes strange edge effects, as does the case where the
    voltage corresponding to the DAC value 0x0000 at the lower edge is not 0V
    but a dead actuator is.)

    Arguments:
     dmin: 2D square array of actuator settings to be fixed, in floating-point
      volts.
     vmax: Maximum voltage permitted, in floating-point volts.  The output
      will be no larger than the largest number <= vmax which is an integer
      multiple of vquant.  If vquant were allowed to be 0, this would be exact,
      but the solver can't be currently guaranteed to converge in that case.
     vquant: quantization step in the DAC (LSB).  Used to determine how close
      to the vmin/vmax/vneighbor/vcorner thresholds a setting can be before it
      triggers a correction.  Must be > 0; if vquant were permitted to be =0,
      thresholds would be exact to floating-point error, but the solver can't
      be currently guaranteed to converge.
     vneighbor: Permitted voltage difference between lateral neighbors, in
      floating-point volts.  Must be >= 0.
     vcorner: Permitted voltage difference between diagonal neighbors, in
      floating-point volts.  Must be >= 0.

    Keyword Arguments:
     dmflat: 2D array of the same size as dmin, or None. Defines the phase-flat
      voltages for neighbor rule checking.  If None, assumes 0V, consistent
      with unpowered polish (CGI baseline).  Neighbor rule must only be
      maintained with respect to a phase-flat array.  dmflat must be <= vmax,
      >= vmin at all points.

    Returns:
     a 2D array of the same size as dmin

    """

    # Check inputs
    check.twoD_array(dmin, 'dmin', TypeError)
    check.real_scalar(vmax, 'vmax', TypeError)
    check.real_positive_scalar(vquant, 'vquant', TypeError)
    check.real_nonnegative_scalar(vneighbor, 'vneighbor', TypeError)
    check.real_nonnegative_scalar(vcorner, 'vcorner', TypeError)
    if dmflat is not None:
        check.twoD_array(dmflat, 'dmflat', TypeError)
        pass
    else:
        dmflat = np.zeros_like(dmin) # 0V default
        pass

    if dmin.shape[0] != dmin.shape[1]:
        raise TypeError('dmin should be square')
    if vmax <= VMIN:
        raise ValueError('VMIN must be less than vmax')
    if dmflat.shape != dmin.shape:
        raise TypeError('dmflat should be same size as dmin')
    if (dmflat < VMIN).any():
        raise ValueError('dmflat should be >= VMIN')
    if (dmflat > vmax).any():
        raise ValueError('dmflat should be <= vmax')

    dmout = dmin.astype(float).copy() # copy as we change in place
    margin = 2*vquant

    # Leave this check.  A future enhancement is to refactor this code to
    # remove the while loop and replace it with something that is guaranteed to
    # be convergent even when vquant=0 (which is a useful test/model case).
    # Leaving this check hurts nothing and likely saves us a divide-by-zero
    # failure should we re-enable vquant==0 support.
    if vquant == 0:
        vmax_q = vmax
        pass
    else:
        vmax_q = vquant*(vmax // vquant)
        pass

    # Fix DM upper and lower bounds
    dmout = dmout.clip(min=VMIN, max=vmax_q)

    # If nside is even, exclude the last member in the second correction step.
    if dmout.shape[0] % 2 == 0:
        end = -1
    else:
        end = None

    # Fix internal neighbor rule violations
    nrdone = False
    dmout -= dmflat # subtract only while running
    while not nrdone:
        # Loop has four essentially similar sets of operations
        # Detailed comments only for first, but rest follow same pattern

        # Subtract adjacent columns (horizontal)
        # Fix the array in two steps.
        # First, subtract every other column so that difference is calculated
        # in pairs. For example, for [1, 2, 3, 4, 5, 6], take 2-1, 4-3, 6-5.
        c1x = slice(0, -1, 2)
        c2x = slice(1, None, 2)
        dmx = dmout.copy()
        dmx, maskx = fix_nr(dmx, vneighbor, margin, c1=c1x, c2=c2x)

        # Next, follow the same procedure but excluding the first column, and
        # if nside is even exclude the last column also.
        # For example, for [1, 2, 3, 4, 5, 6] take 3-2, 5-4.
        dmx_ = dmout.copy()
        dmx_[:, 1:end], maskx_ = fix_nr(dmx_[:, 1:end], vneighbor, margin,
                                        c1=c1x, c2=c2x)
        # Combine the two adjustements together by applying them one at a time.
        dmout[maskx] = dmx[maskx]
        # Only apply second step adjustments to elements that were not adjusted
        # in the first step. This will avoid adjusting the same elements twice.
        dmout[~maskx] = dmx_[~maskx]


        # Subtract adjacent rows (vertical)
        r1y = slice(0, -1, 2)
        r2y = slice(1, None, 2)
        dmy = dmout.copy()
        dmy, masky = fix_nr(dmy, vneighbor, margin, r1=r1y, r2=r2y)

        dmy_ = dmout.copy()
        dmy_[1:end], masky_ = fix_nr(dmy_[1:end], vneighbor, margin,
                                     r1=r1y, r2=r2y)
        dmout[masky] = dmy[masky]
        dmout[~masky] = dmy_[~masky]


        # Subtract adjacent rows (diagonal right)
        r1xy = slice(0, -1, 2)
        r2xy = slice(1, None, 2)
        c1xy = slice(0, -1)
        c2xy = slice(1, None)
        dmxy = dmout.copy()
        dmxy, maskxy = fix_nr(dmxy, vcorner, margin, r1=r1xy, c1=c1xy,
                              r2=r2xy, c2=c2xy)

        dmxy_ = dmout.copy()
        dmxy_[1:end], maskxy_ = fix_nr(dmxy_[1:end], vcorner, margin,
                                       r1=r1xy, c1=c1xy, r2=r2xy, c2=c2xy)
        dmout[maskxy] = dmxy[maskxy]
        dmout[~maskxy] = dmxy_[~maskxy]


        # Subtract adjacent rows (diagonal left)
        r1yx = slice(0, -1, 2)
        r2yx = slice(1, None, 2)
        c1yx = slice(1, None)
        c2yx = slice(0, -1)
        dmyx = dmout.copy()
        dmyx, maskyx = fix_nr(dmyx, vcorner, margin, r1=r1yx, c1=c1yx,
                              r2=r2yx, c2=c2yx)

        dmyx_ = dmout.copy()
        dmyx_[1:end], maskyx_ = fix_nr(dmyx_[1:end], vcorner, margin,
                                       r1=r1yx, c1=c1yx, r2=r2yx, c2=c2yx)
        dmout[maskyx] = dmyx[maskyx]
        dmout[~maskyx] = dmyx_[~maskyx]


        # If any of them had violations, go around again
        nrdone = (maskx.sum() == 0) and (maskx_.sum() == 0) and \
                 (masky.sum() == 0) and (masky_.sum() == 0) and \
                 (maskxy.sum() == 0) and (maskxy_.sum() == 0) and \
                 (maskyx.sum() == 0) and (maskyx_.sum() == 0)
        pass

    dmout += dmflat # reinsert
    # recheck clip, clip is applied without flat
    dmout = dmout.clip(min=VMIN, max=vmax_q)

    return dmout
