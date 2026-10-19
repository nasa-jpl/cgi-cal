# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Utility tool to calculate a DM surface relative to flat
"""

import numpy as np

from . import check
from .dmhtoph import dmhtoph

def dmsurf(nrow, ncol, dmin, nact, inf_func, ppact_d, ppact_cx, ppact_cy, dx,
           dy, thact, flipx, gainmap, flatmap):
    """
    Compute an estimate of the DM surface relative to the phase flat and
    aligned to EXCAM pupil-plane coordinates.

    Surface creation uses dmhtoph(), and all inputs except dmin, gainmap, and
    flatmap are passed directly into a call to that.

    Note: this function produces the DM surface relative to flat in nm.  To get
    wavefront (OPD) relative to flat, multiply the output by 2.  To convert to
    radians surface at a particular wavelength 'lambda' in nm, multiply by
    2*pi/lambda.

    Arguments:
     nrow: number of rows in output array. Integer > 0.
     ncol: number of cols in output array. Integer > 0.
     dmin: a 2D ``nact`` x ``nact`` matrix of real numbers representing DM
      poke heights
     nact: number of actuators across the DM.  Should be an positve integer
      and match both ``dmin.shape[0]`` and ``dmin.shape[1]``.  This is not
      being taken directly from dmin to catch potential invalid inputs (as
      this function will happily work with whichever DM setting it gets).
     inf_func: 2D square array with the centered representation on what the
      facesheet deformation looks like for a single poke of unit height.  The
      function must be smaller than the DM (nact*ppact_d).   The edge size must
      be odd-sized so that the central pixel of the array coincides with the
      array peak.
     ppact_d: design pixels per actuator in ``inf_func``.  Expected to be a
      positive scalar integer. ``inf_func`` will in general cover several
      actuators, as influence functions are not perfectly localized;
      ``ppact_d`` is required to denote the underlying grid.  Must be sampled
      better than the camera (ppact_cx and ppact_cy).
     ppact_cx: pixels per actuator in the x-direction on the camera.  Expected
      to be a real positive scalar.  ``ppact_d`` will be downsampled to this.
       Measured by DM registration.
     ppact_cy: pixels per actuator in the y-direction on the camera.  Expected
      to be a real positive scalar.  ``ppact_d`` will be downsampled to this.
       Measured by DM registration.
     dx: number of pixels to shift the DM grid off center in the x-direction
      on the camera. Expected to be a real scalar.  Measured by DM
      registration.
     dy: number of pixels to shift the DM grid off center in the y-direction
      on the camera. Expected to be a real scalar.  Measured by DM
      registration.
     thact: number of degrees to rotate the grid in a counterclockwise
      direction about the center of the array.  Expected to be a real scalar.
      Measured by DM registration.
     flipx: boolean whether to flip the output in the x-direction, leaving the
      y-direction unchanged.  This will only be used if required to
      accommodate DM electronics being wired with a parity flip relative to
      the camera.
     gainmap: 2D array of floating point gain values per actuator, in nm/V.
      Same size as dmin.
     flatmap: a 2D array of floating-point voltages, of the same size as
      dmin.  This array represents a physically-flat DM surface.

    Returns:
     an nrow x ncol array, matched to camera orientation and representing a DM
       surface relative to flat in nm.

    """

    # Check inputs
    check.positive_scalar_integer(nrow, 'nrow', TypeError)
    check.positive_scalar_integer(ncol, 'ncol', TypeError)
    check.twoD_array(dmin, 'dmin', TypeError)
    check.positive_scalar_integer(nact, 'nact', TypeError)
    check.positive_scalar_integer(ppact_d, 'ppact_d', TypeError)
    check.twoD_array(inf_func, 'inf_func', TypeError)
    check.real_positive_scalar(ppact_cx, 'ppact_cx', TypeError)
    check.real_positive_scalar(ppact_cy, 'ppact_cy', TypeError)
    check.real_scalar(dx, 'dx', TypeError)
    check.real_scalar(dy, 'dy', TypeError)
    check.real_scalar(thact, 'thact', TypeError)
    check.boolean(flipx, 'flipx', TypeError)
    check.twoD_array(gainmap, 'gainmap', TypeError)
    check.twoD_array(flatmap, 'flatmap', TypeError)

    if not np.isreal(dmin).all():
        raise TypeError('DM settings must be real-valued')
    if (np.array(inf_func.shape) >= nact*ppact_d).any():
        raise TypeError('Influence function must be smaller than DM')
    if (np.array(dmin.shape) != nact).any():
        raise TypeError('dmin must be nact x nact')
    if (ppact_d < ppact_cx) or (ppact_d < ppact_cy):
        raise TypeError('Design influence function must be sampled '
                        'better than camera')
    if inf_func.shape[0] != inf_func.shape[1]:
        raise TypeError('inf_func must be square')
    if inf_func.shape[0] % 2 != 1:
        raise TypeError('inf_func must be odd-sized')
    if gainmap.shape != dmin.shape:
        raise TypeError('gainmap and dmin must be same size')
    if flatmap.shape != dmin.shape:
        raise TypeError('flatmap and dmin must be same size')

    # input = voltage
    # subtract flatmap (V - F)
    # multiply by gainmap in nm/V => G*(V -F)
    # dmhtoph says "radians" but is unit-maintaining.  Run G*(V-F) through
    #  dmhtoph and return the output to give surface in nm.
    return dmhtoph(nrow, ncol,
                   gainmap*(dmin-flatmap),
                   nact, inf_func, ppact_d, ppact_cx,
                   ppact_cy, dx, dy, thact, flipx)
