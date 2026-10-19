# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Unit tests for the out-and-back lens propagator
"""

import unittest
import os
import logging

import astropy.io.fits as pyfits
import numpy as np

from cal.util.insertinto import insertinto
from cal.pr.util.lens_fresnel import prop, tiltphases
from cal.pr.util.plane import Plane

logging.basicConfig(level=logging.WARNING) #, filename='log.txt')

def wrap_npi2pi(x):
    """Wrap an angle to be in [-pi, pi]"""
    return np.mod(x + np.pi, 2*np.pi) - np.pi


class TestProp(unittest.TestCase):
    """
    x Test success
    x Test invalid inputs

    x Check different output sizes, verify same in common regions
    x Check different zero-padded input sizes, verify no difference to output
    x Check rectangles
    x Check odd and even sizes

    x check a -> c == a -> b, b -> c (modulo truncation)
    x test with xshift/yshift on/off
    x propagate to same plane (pinit = pfinal) returns self

    x test exact case (pupil to image) from PDF
    x div by zero case handled appropriately
    x  - note plane handles f=0, so need a different source of zero.
    x no restrictions on e0 size wrt pinit
    """

    def setUp(self):
        localpath = os.path.dirname(os.path.abspath(__file__))
        r386 = pyfits.getdata(os.path.join(localpath, 'testdata',
                                          'pupil_template_D386.00pixels.fits'))
        self.e0 = insertinto(r386, [511, 512]) # get odd and even

        self.nx = 512
        self.ny = 511
        self.dpix = 13e-6
        self.z = 0.300
        self.fpup = self.z/2
        self.fim = self.z
        self.xshift = 0
        self.yshift = 0

        self.pinit = Plane(self.nx, self.ny, self.dpix, self.z, self.fpup,
                           self.xshift, self.yshift)
        self.pfinal = Plane(self.nx, self.ny, self.dpix, self.z, self.fim,
                           self.xshift, self.yshift)

        self.lam = 575e-9

        pass


    def test_success(self):
        """Completed successfully with good inputs"""
        prop(self.e0, self.pinit, self.pfinal, self.lam)
        pass


    def test_invalid_e0(self):
        """Verify invalid inputs caught"""
        for perr in [1, 0, 1j, -2.5, 'txt', [5, 5], None, # not arrays
                     np.zeros((4,)), np.zeros((4, 4, 4))]: # not 2D
            with self.assertRaises(TypeError):
                prop(perr, self.pinit, self.pfinal, self.lam)
                pass
            pass
        pass


    def test_invalid_pinit(self):
        """Verify invalid inputs caught"""
        for perr in [1, 0, 1j, -2.5, 'txt', [5, 5], None,
                     np.zeros((4,)), np.zeros((4, 4, 4))]:
            with self.assertRaises(TypeError):
                prop(self.e0, perr, self.pfinal, self.lam)
                pass
            pass
        pass


    def test_invalid_pfinal(self):
        """Verify invalid inputs caught"""
        for perr in [1, 0, 1j, -2.5, 'txt', [5, 5], None,
                     np.zeros((4,)), np.zeros((4, 4, 4))]:
            with self.assertRaises(TypeError):
                prop(self.e0, self.pinit, perr, self.lam)
                pass
            pass
        pass


    def test_invalid_lam(self):
        """Verify invalid inputs caught"""
        for perr in [0, 1j, -2.5, 'txt', [5, 5], None,
                     np.zeros((4,)), np.zeros((4, 4, 4))]:
            with self.assertRaises(TypeError):
                prop(self.e0, self.pinit, self.pfinal, perr)
                pass
            pass
        pass


    def test_zeropadding_changes_nothing(self):
        """
        Verify that using different zero padding on e0, beyond the extent of
        the region containing data, produces the same results

        Includes both padding and cropping with respect to pinit
        """
        # self.e0 is already zero-padded up from 386 pixels, so we can crop
        # a bit with no risk
        #
        # self.e0 is [511, 512]

        tol = 1e-12
        e1ref = prop(self.e0, self.pinit, self.pfinal, self.lam)

        for eshape in [[501, 500],
                       [600, 601],
                       [501, 531],
                       [530, 500]]:
            e1test = prop(insertinto(self.e0, eshape), self.pinit,
                          self.pfinal, self.lam)
            # Incidentally this also verifies that output shape is independent
            # of input shape
            self.assertTrue(np.max(np.abs(e1test - e1ref)) < tol)
            pass
        pass

    def test_output_size_does_not_affect_center(self):
        """
        Verify that the center region (in common with all) is independent of
        the size of the output
        """

        tol = 1e-12
        e1ref = prop(self.e0, self.pinit, self.pfinal, self.lam)

        # use ref as smallest and go larger; this does not lose generality
        # since all are checked against minimum anyway
        for ny, nx in [[self.ny+10, self.nx],
                       [self.ny, self.nx+15],
                       [self.ny+21, self.nx+37],
                       ]:
            pfin = Plane(nx, ny, self.dpix, self.z, self.fim,
                               self.xshift, self.yshift)
            e1tmp = prop(self.e0, self.pinit, pfin, self.lam)
            e1test = insertinto(e1tmp, [self.ny, self.nx])

            self.assertTrue(np.max(np.abs(e1test - e1ref)) < tol)
            pass
        pass


    def test_input_xshift_yshift(self):
        """
        Check that xshift and yshift on the input work as expected

        A shift on the input of N pixels should produce a tilt on the output,
        enough to shift the image by N pixels in the reverse direction on the
        way back
        """

        tol = 1e-12

        dx = 1.2
        dy = -3.4

        pinit = Plane(self.nx, self.ny, self.dpix, self.z, self.fpup,
                           0, 0)
        pinitx = Plane(self.nx, self.ny, self.dpix, self.z, self.fpup,
                           dx, 0)
        pinity = Plane(self.nx, self.ny, self.dpix, self.z, self.fpup,
                           0, dy)

        enom = prop(self.e0, pinit, self.pfinal, self.lam)
        ex = prop(self.e0, pinitx, self.pfinal, self.lam)
        ey = prop(self.e0, pinity, self.pfinal, self.lam)

        # Phase to shift 1 pixel when going back
        xph, yph = tiltphases(self.pfinal, pinit, self.lam)

        # Same amplitude, tilted phase
        self.assertTrue(np.max(np.abs(np.abs(ex) - np.abs(enom))) < tol)
        self.assertTrue(np.max(np.abs(np.abs(ey) - np.abs(enom))) < tol)
        dangx = wrap_npi2pi((np.angle(ex) - np.angle(enom)) + (dx*xph))
        dangy = wrap_npi2pi((np.angle(ey) - np.angle(enom)) + (dy*yph))
        self.assertTrue(np.max(np.abs(dangx)) < tol)
        self.assertTrue(np.max(np.abs(dangy)) < tol)

        pass


    def test_output_xshift_yshift(self):
        """Check that xshift and yshift on the output work as expected"""
        tol = 1e-12

        # Different initial and final sampling so we're not fooling ourselves
        # Same with dimensions
        pinit = Plane(self.nx+6, self.ny+6, self.dpix, self.z, self.fpup, 0, 0)
        for sc in [1, 0.5, 1.2]:
            pfinal = Plane(self.nx, self.ny, self.dpix*sc, self.z, self.fim,
                               0, 0)
            pfinalx = Plane(self.nx, self.ny, self.dpix*sc, self.z, self.fim,
                               1, 0)
            pfinaly = Plane(self.nx, self.ny, self.dpix*sc, self.z, self.fim,
                               0, 1)
            pfinalxn = Plane(self.nx, self.ny, self.dpix*sc, self.z, self.fim,
                               -1, 0)
            pfinalyn = Plane(self.nx, self.ny, self.dpix*sc, self.z, self.fim,
                               0, -1)

            enom = prop(self.e0, pinit, pfinal, self.lam)
            extarg = np.roll(enom, 1, axis=1)
            eytarg = np.roll(enom, 1, axis=0)
            extargn = np.roll(enom, -1, axis=1)
            eytargn = np.roll(enom, -1, axis=0)

            ex = prop(self.e0, self.pinit, pfinalx, self.lam)
            ey = prop(self.e0, self.pinit, pfinaly, self.lam)
            exn = prop(self.e0, self.pinit, pfinalxn, self.lam)
            eyn = prop(self.e0, self.pinit, pfinalyn, self.lam)

            # crop these down since roll is going to move garbage over to the
            # other side and we don't want to compare that
            extarg = insertinto(extarg, [self.ny-2, self.nx-2])
            eytarg = insertinto(eytarg, [self.ny-2, self.nx-2])
            extargn = insertinto(extargn, [self.ny-2, self.nx-2])
            eytargn = insertinto(eytargn, [self.ny-2, self.nx-2])
            ex = insertinto(ex, [self.ny-2, self.nx-2])
            ey = insertinto(ey, [self.ny-2, self.nx-2])
            exn = insertinto(exn, [self.ny-2, self.nx-2])
            eyn = insertinto(eyn, [self.ny-2, self.nx-2])

            self.assertTrue(np.max(np.abs(extarg-ex)) < tol)
            self.assertTrue(np.max(np.abs(eytarg-ey)) < tol)
            self.assertTrue(np.max(np.abs(extargn-exn)) < tol)
            self.assertTrue(np.max(np.abs(eytargn-eyn)) < tol)
            pass

        pass


    def test_out_and_back_to_shifted_plane(self):
        """
        Verify that propagation to and from a shifted plane returns the
        same field regardless of shift

        Use a gaussian so it's not dominated by edge effects
        """
        tol = 1e-12

        oneovere = 0.02 # make this small, so edge is e^(-50) ~ 1e-22
        x = (np.arange(self.nx) - self.nx//2)/(self.nx/2)
        y = (np.arange(self.ny) - self.ny//2)/(self.ny/2)
        X, Y = np.meshgrid(x, y)
        Rsq = X**2 + Y**2
        egauss = np.exp(-Rsq/oneovere)

        for dxi, dyi in [(0, 0),
                         (0, 2.1),
                         (3.5, 0),
                         (3.5, 2.1)]:
            for dxo, dyo in [(0, 0),
                             (1.2, 0),
                             (0, -3.5),
                             (1.2, -3.5)]:
                pinit = Plane(self.nx, self.ny, self.dpix, self.z, self.fpup,
                              dxi, dyi)
                pfinal = Plane(self.nx, self.ny, self.dpix, self.z, self.fim,
                               dxo, dyo)

                emid = prop(egauss, pinit, pfinal, self.lam)
                efin = prop(emid, pfinal, pinit, self.lam)

                self.assertTrue(np.max(np.abs(efin - egauss)) < tol)
                pass
        pass


    def test_return_self(self):
        """Verify that propagation from a plane to itself returns itself"""
        tol = 1e-12

        for p in [self.pinit, self.pfinal]:
            e1 = prop(self.e0, p, p, self.lam)
            self.assertTrue(np.max(np.abs(e1 - self.e0)) < tol)
            pass
        pass


    def test_two_step(self):
        """
        Verify that a -> c = a -> b then b -> c, at least to the level we
        can capture given the inherent truncation at b

        Use a gaussian (smooth falloff to zero) with a steep falloff, so the b
        truncation effects should be below floating-point noise
        """

        tol = 1e-12

        oneovere = 0.02 # make this small, so edge is e^(-50) ~ 1e-22
        x = (np.arange(self.nx) - self.nx//2)/(self.nx/2)
        y = (np.arange(self.ny) - self.ny//2)/(self.ny/2)
        X, Y = np.meshgrid(x, y)
        Rsq = X**2 + Y**2
        egauss = np.exp(-Rsq/oneovere)

        # need to choose smaller nx,ny for mid-plane to avoid DFT alias warning
        # nx, ny are chosen ad hoc
        pmid = Plane(340, 340, self.dpix, self.z,
                          (self.fpup + self.fim)/2,
                          self.xshift, self.yshift)

        etarg = prop(egauss, self.pinit, self.pfinal, self.lam)
        emid = prop(egauss, self.pinit, pmid, self.lam)
        etest = prop(emid, pmid, self.pfinal, self.lam)

        self.assertTrue(np.max(np.abs(etarg - etest)) < tol)

        pass


    def test_exact_case(self):
        """
        Check the worked exact case from the PDF documentation: pupil -> image
        with a rotation
        """

        tol = 1e-12

        # Choose pupil padding so it's exactly 2x
        # Choose focal sampling so it's exactly Nyquist
        # then we can use a 2D IFFT to cross-check
        localpath = os.path.dirname(os.path.abspath(__file__))
        r386 = pyfits.getdata(os.path.join(localpath, 'testdata',
                                          'pupil_template_D386.00pixels.fits'))
        npixp = 386 # pixels across pupil
        nx = npixp*2
        ny = npixp*2

        # pupil goes to edges, so this is 2x
        e0 = insertinto(r386, [nx, ny])
        e0 /= np.sqrt(np.sum(np.abs(e0)**2)) # normalize to unit energy

        # Nyquist: 1 pixel = 0.5*lam/D*f
        dpixI = 10e-6 # implies D = 3860e-6
        lam = 386e-9
        z = 0.2
        fpup = z/2
        fim = z
        dpixF = 0.5*(lam/(dpixI*npixp))*fim # values chosen so dpixF = dpixI

        xshift = 0
        yshift = 0

        pinit = Plane(nx, ny, dpixI, z, fpup, xshift, yshift)
        pfinal = Plane(nx, ny, dpixF, z, fim, xshift, yshift)

        etest = prop(e0, pinit, pfinal, lam)

        # ortho norm on fft/ifft preserves energy
        etarg = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(e0),
                                             norm='ortho'))
        etarg *= 1j

        self.assertTrue(np.max(np.abs(etarg-etest)) < tol)

        pass


    def test_zero_den_handled(self):
        """Verify the denominator=0 case is handled as in the PDF"""

        # 1/z0 - 1/f0 = 1/z1 - 1/f1
        # use integers for exactness
        z0 = 3
        f0 = 6
        z1 = 2
        f1 = 3

        pinit = Plane(self.nx, self.ny, self.dpix, z0, f0,
                      self.xshift, self.yshift)
        pfinal = Plane(self.nx, self.ny, self.dpix, z1, f1,
                       self.xshift, self.yshift)

        with self.assertRaises(ZeroDivisionError):
            prop(self.e0, pinit, pfinal, self.lam)
            pass

        pass

    def test_DFT_alias_warning_raised(self):
        """Verify too large pinit.nx or too small lam raises DFT alias warning"""

        # Define short forms for legibility
        z0 = self.pinit.z
        f0 = self.pinit.f
        z1 = self.pfinal.z
        f1 = self.pfinal.f

        # 1/zqi, 1/zqo, and 1/zft all have the same denominator, so...
        den = (z0*z1*f1 - z0*z1*f0 + z0*f0*f1 - z1*f0*f1)
        inv_zft = (-f0*f1)/den
        atmp = np.abs(self.pinit.nx*self.pinit.dpix*self.pfinal.dpix*inv_zft/self.lam)

        # alias warning when atmp > 1.0
        # increase nx to cause aliasing
        pinit = Plane(int(np.ceil(1.01/atmp))*self.nx, self.ny, self.dpix,
                        self.z, self.fpup, self.xshift, self.yshift)

        with self.assertWarns(Warning):
            prop(self.e0, pinit, self.pfinal, self.lam)

        # or reduce lam
        lam = 0.99*atmp*self.lam

        with self.assertWarns(Warning):
            prop(self.e0, self.pinit, self.pfinal, lam)

        pass


class TestTiltphases(unittest.TestCase):
    """
    For the function to make phase shapes for single-pixel tilt

    Note the fact that they make the correct shift is done by roll testing
    above, so don't need a separate validation (unless the two get somehow
    eventually decoupled)

    x Success
    x Test inputs
    check singular cases
    x Check output shapes
    """

    def setUp(self):
        # use prop settings
        self.nx = 512
        self.ny = 511
        self.dpix = 13e-6
        self.z = 0.300
        self.fpup = self.z/2
        self.fim = self.z
        self.xshift = 0
        self.yshift = 0

        self.pinit = Plane(self.nx, self.ny, self.dpix, self.z, self.fpup,
                           self.xshift, self.yshift)
        self.pfinal = Plane(self.nx, self.ny, self.dpix, self.z, self.fim,
                           self.xshift, self.yshift)

        self.lam = 575e-9

        pass


    def test_success(self):
        """Completed successfully with good inputs"""
        tiltphases(self.pinit, self.pfinal, self.lam)
        pass


    def test_invalid_pinit(self):
        """Verify invalid inputs caught"""
        for perr in [1, 0, 1j, -2.5, 'txt', [5, 5], None,
                     np.zeros((4,)), np.zeros((4, 4, 4))]:
            with self.assertRaises(TypeError):
                tiltphases(perr, self.pfinal, self.lam)
                pass
            pass
        pass


    def test_invalid_pfinal(self):
        """Verify invalid inputs caught"""
        for perr in [1, 0, 1j, -2.5, 'txt', [5, 5], None,
                     np.zeros((4,)), np.zeros((4, 4, 4))]:
            with self.assertRaises(TypeError):
                tiltphases(self.pinit, perr, self.lam)
                pass
            pass
        pass


    def test_invalid_lam(self):
        """Verify invalid inputs caught"""
        for perr in [0, 1j, -2.5, 'txt', [5, 5], None,
                     np.zeros((4,)), np.zeros((4, 4, 4))]:
            with self.assertRaises(TypeError):
                tiltphases(self.pinit, self.pfinal, perr)
                pass
            pass
        pass


    def test_output_size(self):
        """Check outputs have correct size"""
        # use input and output different sizes
        pinit = Plane(self.nx+6, self.ny+6, self.dpix, self.z, self.fpup, 0, 0)
        pfinal = Plane(self.nx, self.ny, self.dpix, self.z, self.fim, 0, 0)
        xph, yph = tiltphases(pinit, pfinal, self.lam)

        self.assertTrue((pinit.ny, pinit.nx) == xph.shape)
        self.assertTrue((pinit.ny, pinit.nx) == yph.shape)

        pass


    def test_zero_den_handled(self):
        """Verify the denominator=0 case is handled as in the PDF"""

        # 1/z0 - 1/f0 = 1/z1 - 1/f1
        # use integers for exactness
        z0 = 3
        f0 = 6
        z1 = 2
        f1 = 3

        pinit = Plane(self.nx, self.ny, self.dpix, z0, f0,
                      self.xshift, self.yshift)
        pfinal = Plane(self.nx, self.ny, self.dpix, z1, f1,
                       self.xshift, self.yshift)

        with self.assertRaises(ZeroDivisionError):
            tiltphases(pinit, pfinal, self.lam)
            pass

        pass


if __name__ == '__main__':
    unittest.main()
