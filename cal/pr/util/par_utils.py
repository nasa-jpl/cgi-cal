# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Standalone utility functions specific to the parametric solver
"""

import numpy as np

from cal.util.nollzernikes import xyzern
import cal.util.check as check

def buildmodes(X, Y, prad, zernlist, freqtuplist):
    """
    Create a set of modes for the parametric-fitting to fit to

    Arguments:
     X: a 2D array of values giving the x-coordinate for each pixel in the
      reference plane, in pixels
     Y: a 2D array of values giving the y-coordinate for each pixel in the
      reference plane, in pixels.  Must be the same size as X.
     prad: radius of pupil, in pixels.  > 0. Used for Zernike normalization.
     zernlist: a list of Noll-ordered Zernikes, as integers, to be fit.  The
      list elements must be unique.  The mode order output will respect the
      mode order input.
     freqtuplist: a list of 2-tuples giving spatial frequencies to model,
       in cycles per aperture.  Each will be assigned a cosine and a sine
       term, in order.  The list of 2-tuples must be unique.

    Returns:
      a list of modes (reference field phase), assigned pixel-by-pixel in the
       reference plane. Zernike modes will be normalized to be orthonormal
       (Noll convention).  Sinusoids will have amplitude +/- 1.

    """

    # Check inputs
    check.twoD_array(X, 'X', TypeError)
    check.twoD_array(Y, 'Y', TypeError)
    if X.shape != Y.shape:
        raise TypeError('X and Y must be the same shape')
    check.real_positive_scalar(prad, 'prad', TypeError)

    for index, z in enumerate(zernlist): # if not iterable, also a TypeError
        check.positive_scalar_integer(z, 'zernlist[' + str(index) + ']',
                                      TypeError)
        pass

    for index, f in enumerate(freqtuplist):
        if len(f) != 2:
            raise TypeError('f must have two elements')
        check.real_scalar(f[0], 'freqtuplist[' + str(index) + '][0]',
                          TypeError)
        check.real_scalar(f[1], 'freqtuplist[' + str(index) + '][1]',
                          TypeError)
        pass

    if len(set(zernlist)) != len(zernlist):
        raise ValueError('zernlist elements must be unique')
    if len(set(freqtuplist)) != len(freqtuplist):
        raise ValueError('freqtuplist elements must be unique')


    mode = []

    # Add modes for Zernikes
    for z in zernlist:
        m = xyzern(X, Y, prad, [z])
        mode.append(np.squeeze(m))
        pass

    # Add spatial frequency modes (need both cos/sin components)
    for freqtup in freqtuplist:
        fx, fy = freqtup
        # /2r to be "per aperture"
        arg = 2.*np.pi*(X*fx + Y*fy)/(2.0*prad)
        mode.append(np.cos(arg))
        mode.append(np.sin(arg))
        pass

    return mode


def make_freqtuplist(freqmin, freqmax, freqspace):
    """
    Create a list of tuples for the mode building to use

    This will select all spatial frequencies with a given spacing along the x-
    and y-axes on a 2D grid within a given annular range.

    The list will be built to include positive and negative spatial
    frequencies across a positive-x half-plane.  It will not go full circle as
    sine/cosines make spots at (fx, fy) and (-fx, -fy).  The resulting
    grid will be cut by half an annulus.  No specification is made as to the
    order of frequencies in the list.

        |
        xx
        xxx
        | xx
    ----+-xx-
        | xx
        xxx
        xx
        |

    Each of these will be turned into sine and cosine phase surface terms to
    be fit by other functions in the parametric fitter.

    Using a freqspace == 0 or freqmin == freqmax will have the function return
    an empty list.  This can be useful when you want to always run the function
    but sometimes not specify any spatial frequencies.  freqmin and freqmax are
    inclusive.

    Arguments:
     freqmin: lowest spatial frequency to model in cycles per aperture.  Must
      be >= 0.
     freqmax: highest spatial frequency to model in cycles per aperture.  Must
      be >= freqmin. If freqmin == freqmax, the return list will be empty.
     freqspace: spacing between spatial frequencies, in cycles per aperture.
      Must be >= 0.  If freqspace == 0, the return list will be empty.

    Returns:
     a list of 2-tuples, each of which is a unique (fx, fy) pair.

    """

    # Check inputs
    check.real_scalar(freqmin, 'freqmin', TypeError)
    check.real_scalar(freqmax, 'freqmax', TypeError)
    check.real_scalar(freqspace, 'freqspace', TypeError)
    if freqmin < 0:
        raise ValueError('freqmin must be >= 0')
    if freqmax < 0:
        raise ValueError('freqmax must be >= 0')
    if freqspace < 0:
        raise ValueError('freqspace must be >= 0')
    if freqmin > freqmax:
        raise ValueError('freqmin must be <= freqmax')

    # Handle short-circuit cases
    if freqmin == freqmax:
        return []
    if freqspace == 0:
        return []

    # This list may have elements larger than freqmax, but the bound will be
    # enforced later.
    flist = np.arange(0, freqmax+freqspace, freqspace)

    ftup = set() # keep elements unique
    for fx in flist:
        for fy in flist:
            # sq to circ
            fr = np.sqrt(fx**2 + fy**2)
            if (fr > freqmax) or (fr < freqmin):
                continue

            ftup.add((fx, fy))
            ftup.add((fx, -fy))
            pass
        pass

    return list(ftup)
