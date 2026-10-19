# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
# pylint: disable=unsubscriptable-object
"""Unit tests for dmsurf.py."""

import unittest
import os

import numpy as np
from astropy.io import fits

from .dmsurf import dmsurf
from .insertinto import insertinto

# set up some defaults
localpath = os.path.dirname(os.path.abspath(__file__))
fninf_pist = os.path.join(localpath, 'testdata',
                          'ut_influence_dm5v2_inffix.fits')

p = fits.getdata(fninf_pist)
ppa = 10  # correct value for this shape

# Test shapes
nact = 48
dmin0 = np.ones((nact, nact))  # all acts
dmin1 = np.ones((nact, nact))
dmin1[1:-1, 1:-1] = 0  # outer acts
dmin2 = np.zeros((nact, nact))
dmin2[0, 0] = 1  # one act


class TestDmsurf(unittest.TestCase):
    """
    Unit test suite for dmsurf().

    Tests of surface creation are left to dmhtoph; we'll just get the scaling
    from wrapper pieces (gainmap, flatmap)

    invalid inputs
    outputs as expected (scaled right)
    outputs with right shape

    """

    def setUp(self):
        self.nmperV = 5
        self.gainmap = self.nmperV*np.ones((48, 48))
        self.flatV = 50
        self.flatmap = self.flatV*np.ones((48, 48))
        pass


    def test_flatmap_subtraction(self):
        """Test flatmap subtraction works as expected"""
        tol = 1e-10

        # use small nrow and ncol so we test the flatmap over a region where
        # it's uniform, and ignore the part outside the DM
        d50 = dmsurf(nrow=301, ncol=301,
                     dmin=dmin0, nact=nact, inf_func=p, ppact_d=ppa,
                     ppact_cx=ppa-1, ppact_cy=ppa-1,
                     dx=0, dy=0, thact=0, flipx=False,
                     gainmap=self.gainmap, flatmap=self.flatmap,
        )

        d0 = dmsurf(nrow=301, ncol=301,
                    dmin=dmin0, nact=nact, inf_func=p, ppact_d=ppa,
                    ppact_cx=ppa-1, ppact_cy=ppa-1,
                    dx=0, dy=0, thact=0, flipx=False,
                    gainmap=self.gainmap, flatmap=np.zeros_like(self.flatmap),
        )

        self.assertTrue(np.max(np.abs(d50 - d0)) - self.flatV*self.nmperV
                        < tol)
        pass


    def test_gainmap_multiplication(self):
        """Gainmap application works as expected"""
        tol = 1e-10

        d1x = dmsurf(nrow=601, ncol=601,
                     dmin=dmin0, nact=nact, inf_func=p, ppact_d=ppa,
                     ppact_cx=ppa-1, ppact_cy=ppa-1,
                     dx=0, dy=0, thact=0, flipx=False,
                     gainmap=self.gainmap, flatmap=self.flatmap,
        )

        d2x = dmsurf(nrow=601, ncol=601,
                    dmin=dmin0, nact=nact, inf_func=p, ppact_d=ppa,
                    ppact_cx=ppa-1, ppact_cy=ppa-1,
                    dx=0, dy=0, thact=0, flipx=False,
                    gainmap=2*self.gainmap, flatmap=self.flatmap,
        )

        self.assertTrue(np.max(np.abs(d2x - d1x*2)) < tol)

        pass


    def test_outputs(self):
        """Check outputs are nrow x ncol."""
        for nrow, ncol in [(600, 600),
                           (600, 300),
                           (300, 600),
                           (601, 601)]:
            d0 = dmsurf(nrow=nrow, ncol=ncol,
                        dmin=dmin0, nact=nact, inf_func=p, ppact_d=ppa,
                        ppact_cx=ppa-1, ppact_cy=ppa-1,
                        dx=0, dy=0, thact=0, flipx=False,
                        gainmap=self.gainmap, flatmap=self.flatmap,
            )
            self.assertTrue(d0.shape == (nrow, ncol))
            pass

        pass


    # Failure tests
    def test_nrow_positive_scalar_integer(self):
        """Check number-of-actuator type valid."""
        for perr in [[], -48, 0, (48,), 48.0, (6, 6), 1j, 'nact']:
            with self.assertRaises(TypeError):
                dmsurf(nrow=perr, ncol=601,
                       dmin=dmin0, nact=nact, inf_func=p, ppact_d=ppa,
                       ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                       flipx=False,
                       gainmap=self.gainmap, flatmap=self.flatmap)
                pass
            pass
        pass

    def test_ncol_positive_scalar_integer(self):
        """Check number-of-actuator type valid."""
        for perr in [[], -48, 0, (48,), 48.0, (6, 6), 1j, 'nact']:
            with self.assertRaises(TypeError):
                dmsurf(nrow=601, ncol=perr,
                       dmin=dmin0, nact=nact, inf_func=p, ppact_d=ppa,
                       ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                       flipx=False,
                       gainmap=self.gainmap, flatmap=self.flatmap)
                pass
            pass
        pass

    def test_dmin_real(self):
        """Check input DM real."""
        dminc = 1j*dmin0
        with self.assertRaises(TypeError):
            dmsurf(nrow=601, ncol=601,
                   dmin=dminc, nact=nact, inf_func=p, ppact_d=ppa,
                   ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                   flipx=False,
                   gainmap=self.gainmap, flatmap=self.flatmap)
            pass
        pass

    def test_dmin_2Darray(self):
        """Check input DM type valid."""
        for dmin in [np.ones((48,)), np.ones((48, 48, 48)), 'dmin', []]:
            with self.assertRaises(TypeError):
                dmsurf(nrow=601, ncol=601,
                       dmin=dmin, nact=nact, inf_func=p, ppact_d=ppa,
                       ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                       flipx=False,
                       gainmap=self.gainmap, flatmap=self.flatmap)
                pass
            pass
        pass

    def test_nact_positive_scalar_integer(self):
        """Check number-of-actuator type valid."""
        for nacte in [[], -48, 0, (48,), 48.0, (6, 6), 1j, 'nact']:
            with self.assertRaises(TypeError):
                dmsurf(nrow=601, ncol=601,
                       dmin=dmin0, nact=nacte, inf_func=p, ppact_d=ppa,
                       ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                       flipx=False,
                       gainmap=self.gainmap, flatmap=self.flatmap)
                pass
            pass
        pass

    def test_dmin_with_nonnact_sides(self):
        """Check DM size and nact are self-consistent."""
        for dmin in [np.ones((nact+1, nact+1)),
                     np.ones((nact, nact+1)),
                     np.ones((nact+1, nact))]:
            with self.assertRaises(TypeError):
                dmsurf(nrow=601, ncol=601,
                       dmin=dmin, nact=nact, inf_func=p, ppact_d=ppa,
                       ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                       flipx=False,
                       gainmap=self.gainmap, flatmap=self.flatmap)
                pass
            pass
        pass

    def test_inf_func_2Darray(self):
        """Check influence function array type valid."""
        for perr in [np.ones((20,)), np.ones((10, 10, 10)), [], 'inf']:
            with self.assertRaises(TypeError):
                dmsurf(nrow=601, ncol=601,
                       dmin=dmin0, nact=nact, inf_func=perr, ppact_d=ppa,
                       ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                       flipx=False,
                       gainmap=self.gainmap, flatmap=self.flatmap)
                pass
            pass
        pass

    def test_inf_func_square(self):
        """Check influence function array is square."""
        for perr in [np.ones((91, 90)), np.ones((90, 91))]:
            with self.assertRaises(TypeError):
                dmsurf(nrow=601, ncol=601,
                       dmin=dmin0, nact=nact, inf_func=perr, ppact_d=ppa,
                       ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                       flipx=False,
                       gainmap=self.gainmap, flatmap=self.flatmap)
                pass
            pass
        pass

    def test_inf_func_odd(self):
        """Check influence function array is odd-sized."""
        for perr in [np.ones((90, 90))]:
            with self.assertRaises(TypeError):
                dmsurf(nrow=601, ncol=601,
                       dmin=dmin0, nact=nact, inf_func=perr, ppact_d=ppa,
                       ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                       flipx=False,
                       gainmap=self.gainmap, flatmap=self.flatmap)
                pass
            pass
        pass

    def test_ppactd_realpositivescalar(self):
        """Check influence function scaling type valid."""
        for perr in [-1.5, -1, 0, 1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                dmsurf(nrow=601, ncol=601,
                       dmin=dmin0, nact=nact, inf_func=p, ppact_d=perr,
                       ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                       flipx=False,
                       gainmap=self.gainmap, flatmap=self.flatmap)
                pass
            pass
        pass

    def test_ppactcx_realpositivescalar(self):
        """Check output X scaling type valid."""
        for perr in [-1.5, -1, 0, 1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                dmsurf(nrow=601, ncol=601,
                       dmin=dmin0, nact=nact, inf_func=p, ppact_d=ppa,
                       ppact_cx=perr, ppact_cy=4, dx=0, dy=0, thact=0,
                       flipx=False,
                       gainmap=self.gainmap, flatmap=self.flatmap)
                pass
            pass
        pass

    def test_ppactcy_realpositivescalar(self):
        """Check output Y scaling type valid."""
        for perr in [-1.5, -1, 0, 1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                dmsurf(nrow=601, ncol=601,
                       dmin=dmin0, nact=nact, inf_func=p, ppact_d=ppa,
                       ppact_cx=4, ppact_cy=perr, dx=0, dy=0, thact=0,
                       flipx=False,
                       gainmap=self.gainmap, flatmap=self.flatmap)
                pass
            pass
        pass

    def test_dx_realscalar(self):
        """Check offset X scaling type valid."""
        for perr in [1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                dmsurf(nrow=601, ncol=601,
                       dmin=dmin0, nact=nact, inf_func=p, ppact_d=ppa,
                       ppact_cx=4, ppact_cy=4, dx=perr, dy=0, thact=0,
                       flipx=False,
                       gainmap=self.gainmap, flatmap=self.flatmap)
                pass
            pass
        pass

    def test_dy_realscalar(self):
        """Check offset Y scaling type valid."""
        for perr in [1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                dmsurf(nrow=601, ncol=601,
                       dmin=dmin0, nact=nact, inf_func=p, ppact_d=ppa,
                       ppact_cx=4, ppact_cy=4, dx=0, dy=perr, thact=0,
                       flipx=False,
                       gainmap=self.gainmap, flatmap=self.flatmap)
                pass
            pass
        pass

    def test_thact_realscalar(self):
        """Check rotation angle type valid."""
        for perr in [1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                dmsurf(nrow=601, ncol=601,
                       dmin=dmin0, nact=nact, inf_func=p, ppact_d=ppa,
                       ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=perr,
                       flipx=False,
                       gainmap=self.gainmap, flatmap=self.flatmap)
                pass
            pass
        pass

    def test_design_sampled_more_than_camera(self):
        """
        Should fail if your camera sampling is finer than your theoretical
        influence function (because you need a better sampling to downsample).
        """
        for ppactlist in [(10, 11, 11),
                          (10, 4, 11),
                          (10, 11, 4)]:
            pd, pcx, pcy = ppactlist
            with self.assertRaises(TypeError):
                dmsurf(nrow=601, ncol=601,
                       dmin=dmin0, nact=nact, inf_func=p, ppact_d=pd,
                       ppact_cx=pcx, ppact_cy=pcy, dx=0, dy=0, thact=0,
                       flipx=False,
                       gainmap=self.gainmap, flatmap=self.flatmap)
                pass
            pass
        pass

    def test_inf_func_smaller_than_DM_grid(self):
        """Check that the influence function fails if bigger than DMxo."""
        with self.assertRaises(TypeError):
            pwide = insertinto(p, (ppa*nact + 1, ppa*nact + 1))
            dmsurf(nrow=601, ncol=601,
                   dmin=dmin0, nact=nact, inf_func=pwide, ppact_d=ppa,
                   ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                   flipx=False,
                   gainmap=self.gainmap, flatmap=self.flatmap)
            pass
        pass

    def test_flipx_boolean(self):
        """Invalid inputs caught"""
        for perr in [None, (True, False, None)]:
            with self.assertRaises(TypeError):
                dmsurf(nrow=601, ncol=601,
                       dmin=dmin0, nact=nact, inf_func=p, ppact_d=ppa,
                       ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                       flipx=perr,
                       gainmap=self.gainmap, flatmap=self.flatmap)
                pass
            pass
        pass

    def test_gainmap_2Darray(self):
        """Check gainmap type valid."""
        for gainmap in [np.ones((48,)), np.ones((48, 48, 48)), 'dmin', []]:
            with self.assertRaises(TypeError):
                dmsurf(nrow=601, ncol=601,
                       dmin=dmin0, nact=nact, inf_func=p, ppact_d=ppa,
                       ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                       flipx=False,
                       gainmap=gainmap, flatmap=self.flatmap)
                pass
            pass
        pass


    def test_gainmap_not_same_size_as_dmin(self):
        """Check gainmap dim valid."""
        for gainmap in [np.ones((48, 47))]:
            with self.assertRaises(TypeError):
                dmsurf(nrow=601, ncol=601,
                       dmin=dmin0, nact=nact, inf_func=p, ppact_d=ppa,
                       ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                       flipx=False,
                       gainmap=gainmap, flatmap=self.flatmap)
                pass
            pass
        pass


    def test_flatmap_2Darray(self):
        """Check flatmap type valid."""
        for flatmap in [np.ones((48,)), np.ones((48, 48, 48)), 'dmin', []]:
            with self.assertRaises(TypeError):
                dmsurf(nrow=601, ncol=601,
                       dmin=dmin0, nact=nact, inf_func=p, ppact_d=ppa,
                       ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                       flipx=False,
                       gainmap=self.gainmap, flatmap=flatmap)
                pass
            pass
        pass


    def test_flatmap_not_same_size_as_dmin(self):
        """Check flatmap dim valid."""
        for flatmap in [np.ones((48, 47))]:
            with self.assertRaises(TypeError):
                dmsurf(nrow=601, ncol=601,
                       dmin=dmin0, nact=nact, inf_func=p, ppact_d=ppa,
                       ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                       flipx=False,
                       gainmap=self.gainmap, flatmap=flatmap)
                pass
            pass
        pass




if __name__ == '__main__':
    unittest.main()
