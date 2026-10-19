# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Unit tests for the Plane class and its methods
"""

import unittest

import numpy as np

from cal.pr.util.plane import Plane

class TestPlaneClass(unittest.TestCase):
    """
    x Test creation
    x Test invalid inputs
    x Check that internal parameters are set correctly

    x Also tests for x and y shift
    """

    def setUp(self):
        self.nx = 512
        self.ny = 511 # so we check even and odd
        self.dpix = 13e-6
        self.z = 0.300
        self.f = 0.150
        self.xshift = 0
        self.yshift = 0
        self.ceq = 0
        pass


    def test_success(self):
        """Class creation succeeds with valid parameters"""
        Plane(self.nx,
              self.ny,
              self.dpix,
              self.z,
              self.f,
              self.xshift,
              self.yshift,
              self.ceq,
        )
        pass


    def test_invalid_nx(self):
        """Invalid inputs are rejected"""
        for perr in [-1, 0, 512.5, 1j, (20,), 'txt', None]:
            with self.assertRaises(TypeError):
                Plane(perr,
                      self.ny,
                      self.dpix,
                      self.z,
                      self.f,
                      self.xshift,
                      self.yshift)
                pass
            pass
        pass


    def test_invalid_ny(self):
        """Invalid inputs are rejected"""
        for perr in [-1, 0, 512.5, 1j, (20,), 'txt', None]:
            with self.assertRaises(TypeError):
                Plane(self.nx,
                      perr,
                      self.dpix,
                      self.z,
                      self.f,
                      self.xshift,
                      self.yshift)
                pass
            pass
        pass


    def test_invalid_dpix(self):
        """Invalid inputs are rejected"""
        for perr in [-1, 0, 1j, (20,), 'txt', None]:
            with self.assertRaises(TypeError):
                Plane(self.nx,
                      self.ny,
                      perr,
                      self.z,
                      self.f,
                      self.xshift,
                      self.yshift)
                pass
            pass
        pass


    def test_invalid_z(self):
        """Invalid inputs are rejected"""
        for perr in [1j, (20,), 'txt', None]:
            with self.assertRaises(TypeError):
                Plane(self.nx,
                      self.ny,
                      self.dpix,
                      perr,
                      self.f,
                      self.xshift,
                      self.yshift)
                pass
            pass
        pass


    def test_invalid_f(self):
        """Invalid inputs are rejected"""
        for perr in [1j, (20,), 'txt', None]:
            with self.assertRaises(TypeError):
                Plane(self.nx,
                      self.ny,
                      self.dpix,
                      self.z,
                      perr,
                      self.xshift,
                      self.yshift)
                pass
            pass
        pass


    def test_invalid_xshift(self):
        """Invalid inputs are rejected"""
        for perr in [1j, (20,), 'txt', None]:
            with self.assertRaises(TypeError):
                Plane(self.nx,
                      self.ny,
                      self.dpix,
                      self.z,
                      self.f,
                      perr,
                      self.yshift)
                pass
            pass
        pass


    def test_invalid_yshift(self):
        """Invalid inputs are rejected"""
        for perr in [1j, (20,), 'txt', None]:
            with self.assertRaises(TypeError):
                Plane(self.nx,
                      self.ny,
                      self.dpix,
                      self.z,
                      self.f,
                      self.xshift,
                      perr)
                pass
            pass
        pass

    def test_invalid_ceq(self):
        """Invalid inputs are rejected"""
        for perr in [1j, (20,), 'txt', None]:
            with self.assertRaises(TypeError):
                Plane(self.nx,
                      self.ny,
                      self.dpix,
                      self.z,
                      self.f,
                      self.xshift,
                      self.yshift,
                      ceq=perr,
                )
                pass
            pass
        pass

    def test_zero_f(self):
        """Verify zero focal length caught"""
        with self.assertRaises(ValueError):
            Plane(self.nx,
                  self.ny,
                  self.dpix,
                  self.z,
                  0,
                  self.xshift,
                  self.yshift)
            pass
        pass


    def test_xvec(self):
        """xvec has size, center, scale as expected"""
        tol = 1e-13

        p = Plane(self.nx,
                  self.ny,
                  self.dpix,
                  self.z,
                  self.f,
                  self.xshift,
                  self.yshift)

        # size
        self.assertTrue(len(p.xvec) == p.nx)

        # zero-valued center
        self.assertTrue(p.xvec[p.nx//2] == 0)

        # spacing is all ~dpix
        dvec = np.diff(p.xvec)
        self.assertTrue(np.max(np.abs(dvec - p.dpix)) < tol)

        pass


    def test_yvec(self):
        """yvec has size, center, scale as expected"""
        tol = 1e-13

        p = Plane(self.nx,
                  self.ny,
                  self.dpix,
                  self.z,
                  self.f,
                  self.xshift,
                  self.yshift)

        # size
        self.assertTrue(len(p.yvec) == p.ny)

        # zero-valued center
        self.assertTrue(p.yvec[p.ny//2] == 0)

        # spacing is all ~dpix
        dvec = np.diff(p.yvec)
        self.assertTrue(np.max(np.abs(dvec - p.dpix)) < tol)

        pass


    def test_X(self):
        """X has consistent values with xvec"""

        tol = 1e-13

        # Use smaller nx/ny so this test doesn't take forever
        nx = 20
        ny = 21

        p = Plane(nx,
                  ny,
                  self.dpix,
                  self.z,
                  self.f,
                  self.xshift,
                  self.yshift)

        for row in range(ny):
            for col in range(nx):
                self.assertTrue(np.abs(p.X[row, col] - p.xvec[col]) < tol)
                pass
            pass
        pass


    def test_Y(self):
        """Y has consistent values with yvec"""

        tol = 1e-13

        # Use smaller nx/ny so this test doesn't take forever
        nx = 20
        ny = 21

        p = Plane(nx,
                  ny,
                  self.dpix,
                  self.z,
                  self.f,
                  self.xshift,
                  self.yshift)

        for row in range(ny):
            for col in range(nx):
                self.assertTrue(np.abs(p.Y[row, col] - p.yvec[row]) < tol)
                pass
            pass
        pass


    def test_rsq(self):
        """rsq has consistent values with xvec, yvec"""

        tol = 1e-13

        # Use smaller nx/ny so this test doesn't take forever
        nx = 20
        ny = 21

        p = Plane(nx,
                  ny,
                  self.dpix,
                  self.z,
                  self.f,
                  self.xshift,
                  self.yshift)

        for row in range(ny):
            for col in range(nx):
                self.assertTrue(np.abs(p.Rsq[row, col] -
                                       (p.xvec[col]**2 + p.yvec[row]**2))
                                       < tol)
                pass
            pass
        pass



class TestXYShift(unittest.TestCase):
    """
    Test success for good inputs
    test invalid inputs
    verify output matches expected
    verify in-place operation with successive calls and checks
    """

    def setUp(self):
        self.nx = 512
        self.ny = 511 # so we check even and odd
        self.dpix = 13e-6
        self.z = 0.300
        self.f = 0.150
        self.xshift = 0
        self.yshift = 0
        pass

    def test_x_success(self):
        """Good inputs run without throwing errors"""
        p = Plane(self.nx,
              self.ny,
              self.dpix,
              self.z,
              self.f,
              self.xshift,
              self.yshift)

        p.add_xshift(-0.5)
        pass


    def test_y_success(self):
        """Good inputs run without throwing errors"""
        p = Plane(self.nx,
              self.ny,
              self.dpix,
              self.z,
              self.f,
              self.xshift,
              self.yshift)

        p.add_yshift(-0.5)
        pass


    def test_invalid_x(self):
        """Invalid inputs are rejected and don't set shift"""
        p = Plane(self.nx,
              self.ny,
              self.dpix,
              self.z,
              self.f,
              self.xshift,
              self.yshift)
        xs0 = p.xshift

        for perr in [1j, (20,), 'txt', None]:
            with self.assertRaises(TypeError):
                p.add_xshift(perr)
                pass
            pass

        self.assertTrue(p.xshift == xs0)
        pass


    def test_invalid_y(self):
        """Invalid inputs are rejected and don't set shift"""
        p = Plane(self.nx,
              self.ny,
              self.dpix,
              self.z,
              self.f,
              self.xshift,
              self.yshift)
        ys0 = p.yshift

        for perr in [1j, (20,), 'txt', None]:
            with self.assertRaises(TypeError):
                p.add_yshift(perr)
                pass
            pass

        self.assertTrue(p.yshift == ys0)
        pass


    def test_x_sets(self):
        """Verify shift set correctly (as relative) over multiple calls"""
        tol = 1e-13

        p = Plane(self.nx,
              self.ny,
              self.dpix,
              self.z,
              self.f,
              self.xshift,
              self.yshift)

        xs0 = self.xshift
        dx = 0.1
        nx = 10

        out = xs0 + dx*np.arange(1, nx+1)
        for j in range(nx):
            p.add_xshift(dx)
            self.assertTrue(np.abs(p.xshift - out[j]) < tol)
            pass
        pass


    def test_y_sets(self):
        """Verify shift set correctly (as relative) over multiple calls"""
        tol = 1e-13

        p = Plane(self.nx,
              self.ny,
              self.dpix,
              self.z,
              self.f,
              self.xshift,
              self.yshift)

        ys0 = self.yshift
        dy = 0.1
        ny = 10

        out = ys0 + dy*np.arange(1, ny+1)
        for j in range(ny):
            p.add_yshift(dy)
            self.assertTrue(np.abs(p.yshift - out[j]) < tol)
            pass
        pass


class TestPlaneEquality(unittest.TestCase):
    """
    x Check planes are equal, not equal after creation
    x Check planes are equal, not equal after shifts
    x check eq and neq
    """

    def test_equal(self):
        """Verify equality works for equal planes"""
        p0 = Plane(5, 5, 5, 5, 5, 5, 5)
        p1 = Plane(5, 5, 5, 5, 5, 5, 5)
        self.assertTrue(p0 == p1)
        self.assertFalse(p0 != p1)
        pass


    def test_inequal(self):
        """Verify inequality works for equal planes"""
        p0 = Plane(5, 5, 5, 5, 5, 5, 5)
        p1 = Plane(4, 5, 5, 5, 5, 5, 5)
        p2 = Plane(5, 4, 5, 5, 5, 5, 5)
        p3 = Plane(5, 5, 4, 5, 5, 5, 5)
        p4 = Plane(5, 5, 5, 4, 5, 5, 5)
        p5 = Plane(5, 5, 5, 5, 4, 5, 5)
        p6 = Plane(5, 5, 5, 5, 5, 4, 5)
        p7 = Plane(5, 5, 5, 5, 5, 5, 4)

        for p in [p1, p2, p3, p4, p5, p6, p7]:
            self.assertTrue(p0 != p)
            self.assertFalse(p0 == p)
            pass
        pass

    def test_changing_equality(self):
        """Verify that equality status changes as values change"""

        p0 = Plane(5, 5, 5, 5, 5, 5, 5)
        p1 = Plane(5, 5, 5, 5, 5, 5, 5)
        self.assertTrue(p0 == p1)
        self.assertFalse(p0 != p1)

        p1.add_xshift(1)
        self.assertTrue(p0 != p1)
        self.assertFalse(p0 == p1)

        p1.add_xshift(-1)
        self.assertTrue(p0 == p1)
        self.assertFalse(p0 != p1)
        pass


if __name__ == '__main__':
    unittest.main()
