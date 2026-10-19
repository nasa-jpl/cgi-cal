# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Definition for Plane class and subclasses to store propagation information
"""

import numpy as np

import cal.util.check as check

class Plane(object):
    """
    A Plane class is used to define the optical contents of a plane used by a
    phase retrieval algorithm.  Stores values for a field following a lens and
    a free-space propagation.

    In addition to storing all of its input arguments, the Plane also stores
    five precalculated values that are used by the propagator and other code:
     xvec: a vector of length nx and central value 0 at index nx//2.  (For
      even-valued sizes, this will put the extra length on the lower end.)
      This is a vector in meters, and the difference between adjacent elements
      will be dpix.
     yvec: a vector of length ny and central value 0 at index ny//2.  (For
      even-valued sizes, this will put the extra length on the lower end.)
      This is a vector in meters, and the difference between adjacent elements
      will be dpix.
     X: this is a 2D grid of x at each point in the grid
     Y: this is a 2D grid of x at each point in the grid
     Rsq: this is a 2D grid of x**2 + y**2 at each point in the grid

    """

    def __init__(self, nx, ny, dpix, z, f, xshift, yshift, ceq=0.0):
        """

        Arguments:
         nx: number of *columns* in plane representation. Integer > 0.
         ny: number of *rows* in plane representation.  Integer > 0.
         dpix: size of a single pixel in plane, in meters.  Must be > 0.
         z: free-space distance after lens, in meters
         f: focal length of lens, in meters.  May be negative (diverging lens)
          but may not be zero.
         xshift: shift of image in *columns* from being centered in the array.
          Positive shift moves the image away from zero.  This is in pixels,
          and may be non-integer.
         yshift: shift of image in *rows* from being centered in the array.
          Positive shift moves the image away from zero.  This is in pixels,
          and may be non-integer.
         ceq: wavefront curvature correction, applied as an inner (before)
          quadratic phase term if this is the init plane, and as an outer (post)
          quadratic phase term if this is the final plane. This term is important
          when representing an air-spaced doublet (e.g. pupil image lens) as a
          an equivalent single lens and single propagation.

        """

        # Check inputs
        check.positive_scalar_integer(nx, 'nx', TypeError)
        check.positive_scalar_integer(ny, 'ny', TypeError)
        check.real_positive_scalar(dpix, 'dpix', TypeError)
        check.real_scalar(z, 'z', TypeError)
        check.real_scalar(f, 'f', TypeError)
        check.real_scalar(xshift, 'xshift', TypeError)
        check.real_scalar(yshift, 'yshift', TypeError)
        check.real_scalar(ceq, 'ceq', TypeError)
        if f == 0:
            raise ValueError('zero focal length is nonphysical')

        # Inputs stored in structure
        self.nx = nx
        self.ny = ny
        self.dpix = dpix
        self.z = z
        self.f = f
        self.xshift = xshift
        self.yshift = yshift
        self.ceq = ceq

        # Additional derived parameters

        # Vectors along each coordinate in physical units
        self.xvec = _fftinterval(self.nx)*self.dpix
        self.yvec = _fftinterval(self.ny)*self.dpix

        # r^2 grid, used for quadratic phases
        self.X, self.Y = np.meshgrid(self.xvec, self.yvec)
        self.Rsq = self.X**2 + self.Y**2

        pass


    def add_xshift(self, dx):
        """
        Add a delta shift to the existing x-shift

        If starting shift was x0, final shift will be x0 + dx.

        No returns.

        Arguments:
         dx: additional relative shift along columns, in pixels

        """

        check.real_scalar(dx, 'dx', TypeError)
        self.xshift += dx
        pass


    def add_yshift(self, dy):
        """
        Add a delta shift to the existing y-shift

        If starting shift was y0, final shift will be y0 + dy.

        No returns.

        Arguments:
         dy: additional relative shift along rows, in pixels

        """

        check.real_scalar(dy, 'dy', TypeError)
        self.yshift += dy
        pass


    def __eq__(self, other):
        """
        Overwrite the default object equality check so we can compare Planes
        """
        try:
            if (self.nx == other.nx and
                self.ny == other.ny and
                self.dpix == other.dpix and
                self.z == other.z and
                self.f == other.f and
                self.xshift == other.xshift and
                self.yshift == other.yshift and
                self.ceq == other.ceq):
                return True
        except AttributeError: # usually because other doesn't have these
            return False
        return False


    def __neq__(self, other):
        """
        Overwrite the default object inequality check so we can compare Planes
        """
        return not self.__eq__(other)


def _fftinterval(npts):
    """
    Returns a 1D array of integers of length npts, which places 0 at the center
    for odd-length arrays and on the right side of center for even-sized, e.g.:
     _fftinterval(3) -> [-1, 0, 1]
     _fftinterval(4) -> [-2, -1, 0, 1]
    """
    return np.arange(npts) - npts//2
