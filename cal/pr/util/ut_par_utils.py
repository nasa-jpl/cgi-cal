# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Unit tests for parametric-fitter standalone utilities
"""

import unittest

import numpy as np

from cal.pr.util.par_utils import buildmodes, make_freqtuplist

class TestBuildModes(unittest.TestCase):
    """
    Tests for building the PR parametric fitting modes
    """

    def setUp(self):
        """Set up a minimal case"""
        x = [-2, -1, 0, 1, 2]
        self.X, self.Y = np.meshgrid(x, x)
        self.prad = 2.
        self.zernlist = [4, 5]
        self.freqtuplist = [(0, 4), (0, -4)]
        pass


    def test_success(self):
        """Succeeds with good inputs"""
        buildmodes(self.X, self.Y, self.prad, self.zernlist, self.freqtuplist)
        pass


    def test_invalid_x(self):
        """Verify bad inputs caught as expected"""
        for perr in [1, 0, -1.5, 1j, [1, 1], 'txt', None, # not array
                     np.ones((2,)), np.ones((2, 2, 2)), # not 2D
                     ]:
            with self.assertRaises(TypeError):
                buildmodes(X=perr,
                           Y=self.Y,
                           prad=self.prad,
                           zernlist=self.zernlist,
                           freqtuplist=self.freqtuplist,
                           )
                pass
            pass
        pass


    def test_invalid_y(self):
        """Verify bad inputs caught as expected"""
        for perr in [1, 0, -1.5, 1j, [1, 1], 'txt', None, # not array
                     np.ones((2,)), np.ones((2, 2, 2)), # not 2D
                     ]:
            with self.assertRaises(TypeError):
                buildmodes(X=self.Y,
                           Y=perr,
                           prad=self.prad,
                           zernlist=self.zernlist,
                           freqtuplist=self.freqtuplist,
                           )
                pass
            pass
        pass


    def test_invalid_prad(self):
        """Verify bad inputs caught as expected"""
        for perr in [0, -1.5, 1j, [1, 1], 'txt', None,
                     np.ones((2,)), np.ones((2, 2, 2)),
                     ]:
            with self.assertRaises(TypeError):
                buildmodes(X=self.X,
                           Y=self.Y,
                           prad=perr,
                           zernlist=self.zernlist,
                           freqtuplist=self.freqtuplist,
                           )
                pass
            pass
        pass


    def test_invalid_zernlist_form(self):
        """Verify bad inputs caught as expected"""
        for perr in [0, -1.5, 1j, 'txt', None]:
            with self.assertRaises(TypeError):
                buildmodes(X=self.X,
                           Y=self.Y,
                           prad=self.prad,
                           zernlist=perr,
                           freqtuplist=self.freqtuplist,
                           )
                pass
            pass
        pass


    def test_invalid_zernlist_value(self):
        """Verify bad inputs caught as expected"""
        for perr in [
                [2.5, 3, 3.5],
                [1j, 1],
                [-5],
                [0],
                [(4, 1)],
                ]:
            with self.assertRaises(TypeError):
                buildmodes(X=self.X,
                           Y=self.Y,
                           prad=self.prad,
                           zernlist=perr,
                           freqtuplist=self.freqtuplist,
                           )
                pass
            pass
        pass


    def test_invalid_freqtuplist_form(self):
        """Verify bad inputs caught as expected"""
        for perr in [0, -1.5, 1j, 'txt', None]:
            with self.assertRaises(TypeError):
                buildmodes(X=self.X,
                           Y=self.Y,
                           prad=self.prad,
                           zernlist=self.zernlist,
                           freqtuplist=perr,
                           )
                pass
            pass
        pass


    def test_invalid_freqtuplist_value(self):
        """Verify bad inputs caught as expected"""
        for perr in [
                [2.5, 3, 3.5],
                [1j, 1],
                [-5],
                [0],
                [2, 3, 4],
                [(1j, 2)],
                [(1, 2, 3)],
                [(5, 5), (0,)]
                ]:
            with self.assertRaises(TypeError):
                buildmodes(X=self.X,
                           Y=self.Y,
                           prad=self.prad,
                           zernlist=self.zernlist,
                           freqtuplist=perr,
                           )
                pass
            pass
        pass


    def test_fail_at_x_and_y_different(self):
        """Check that different-sized X and Y fail"""

        badx = np.ones((5, 5))
        bady = np.ones((5, 6))

        with self.assertRaises(TypeError):
            buildmodes(X=badx,
                       Y=bady,
                       prad=self.prad,
                       zernlist=self.zernlist,
                       freqtuplist=self.freqtuplist,
                       )
            pass
        pass


    def test_repeated_zern(self):
        """Verify repeated list entries are rejected"""
        badlist = [4, 5, 6, 7, 4]

        with self.assertRaises(ValueError):
            buildmodes(X=self.X,
                       Y=self.Y,
                       prad=self.prad,
                       zernlist=badlist,
                       freqtuplist=self.freqtuplist,
                       )
            pass
        pass


    def test_repeated_freq(self):
        """Verify repeated list entries are rejected"""
        badlist = [(0, 4), (0, -4), (0, 4)]

        with self.assertRaises(ValueError):
            buildmodes(X=self.X,
                       Y=self.Y,
                       prad=self.prad,
                       zernlist=self.zernlist,
                       freqtuplist=badlist,
                       )
            pass
        pass


    def test_output_format(self):
        """Check the output format matches documentation"""
        modes = buildmodes(self.X, self.Y, self.prad, self.zernlist,
                           self.freqtuplist)
        for m in modes:
            self.assertTrue(m.shape == self.X.shape)
            pass

        # one mode per zernike, two modes per freq
        self.assertTrue(len(modes) == len(self.zernlist)
                        + 2*len(self.freqtuplist))
        pass


    def test_success_with_empty_arrays(self):
        """Check that empty zernlist and freqtuplist are fine"""

        for z, f in [([], self.freqtuplist),
                     (self.zernlist, []),
                     ([], [])]:
            modes = buildmodes(self.X, self.Y, self.prad, z, f)

            # one mode per zernike, two modes per freq
            self.assertTrue(len(modes) == len(z) + 2*len(f))
            pass
        pass


    def test_rearrange_xy(self):
        """Verify that a rearrangement of X and Y doesn't change per-pixel"""
        modes = buildmodes(self.X, self.Y, self.prad, self.zernlist,
                           self.freqtuplist)

        rx = np.roll(np.roll(self.X, 2, axis=0), 2, axis=1)
        ry = np.roll(np.roll(self.Y, 2, axis=0), 2, axis=1)

        rmodes = buildmodes(rx, ry, self.prad, self.zernlist,
                            self.freqtuplist)

        self.assertTrue(len(modes) == len(rmodes))

        for j, rm in enumerate(rmodes):
            # unroll
            rm_unroll = np.roll(np.roll(rm, -2, axis=0), -2, axis=1)
            self.assertTrue((modes[j] == rm_unroll).all())
            pass
        pass


    def test_exact_zern(self):
        """Check the outputs are as expected for exact cases"""

        tol = 1e-13

        # Let's do coma for its asymmetry
        # Noll 1976 says...
        # Z7 = sqrt(8)*(3r**3 - 2*r)*sin(theta)
        # Z8 = sqrt(8)*(3r**3 - 2*r)*cos(theta)
        x = [-2, -1, 0, 1, 2]
        X, Y = np.meshgrid(x, x)
        R = np.hypot(Y, X)
        T = np.arctan2(Y, X)
        prad = 1

        z7mode = np.sqrt(8)*(3*R**3 - 2*R)*np.sin(T)
        z8mode = np.sqrt(8)*(3*R**3 - 2*R)*np.cos(T)
        m = buildmodes(X, Y, prad, [7, 8], [])

        self.assertTrue(np.max(np.abs(m[0] - z7mode)) < tol)
        self.assertTrue(np.max(np.abs(m[1] - z8mode)) < tol)

        pass


    def test_exact_freq(self):
        """Check the outputs are as expected for exact cases"""

        tol = 1e-13

        x = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        X, Y = np.meshgrid(x, x)
        prad = 2

        # (2, 0) = 2 cycles per aperture in x (along cols), constant rows
        sin_2_0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]) # hits all the nodes
        cos_2_0 = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1]) # hits all peaks
        smode = np.ones((len(x), 1)) @ sin_2_0[np.newaxis, :]
        cmode = np.ones((len(x), 1)) @ cos_2_0[np.newaxis, :]

        m = buildmodes(X, Y, prad, [], [(2, 0)])

        self.assertTrue(np.max(np.abs(m[0] - cmode)) < tol)
        self.assertTrue(np.max(np.abs(m[1] - smode)) < tol)

        pass


class TestMakeFreqTupList(unittest.TestCase):
    """
    Tests for building the list of spatial frequencies

    # Success
    # failure by type
    # freqmin > freqmax
    # any input < 0
    empty list cases (freqspace == 0, freqmin == freqmax)
    exact return
    """

    def test_success(self):
        """Does not fail on good inputs"""
        make_freqtuplist(5, 10, 1)
        pass


    def test_invalid_freqmin_type(self):
        """Invalid inputs caught"""
        freqmax = 10
        freqspace = 1

        for perr in [1j, (5.,), 'txt', None]:
            with self.assertRaises(TypeError):
                make_freqtuplist(freqmin=perr,
                                 freqmax=freqmax,
                                 freqspace=freqspace)
                pass
            pass
        pass


    def test_invalid_freqmax_type(self):
        """Invalid inputs caught"""
        freqmin = 5
        freqspace = 1

        for perr in [1j, (10.,), 'txt', None]:
            with self.assertRaises(TypeError):
                make_freqtuplist(freqmin=freqmin,
                                 freqmax=perr,
                                 freqspace=freqspace)
                pass
            pass
        pass


    def test_invalid_freqspace_type(self):
        """Invalid inputs caught"""
        freqmin = 5
        freqmax = 10

        for perr in [1j, (1,), 'txt', None]:
            with self.assertRaises(TypeError):
                make_freqtuplist(freqmin=freqmin,
                                 freqmax=freqmax,
                                 freqspace=perr)
                pass
            pass
        pass


    def test_invalid_freqmin_value(self):
        """Invalid inputs caught"""
        freqmax = 10
        freqspace = 1

        for perr in [-1, -0.01, # >= 0
                    freqmax+1]: # <= freqmax
            with self.assertRaises(ValueError):
                make_freqtuplist(freqmin=perr,
                                 freqmax=freqmax,
                                 freqspace=freqspace)
                pass
            pass
        pass


    def test_invalid_freqmax_value(self):
        """Invalid inputs caught"""
        freqmin = 5
        freqspace = 1

        for perr in [-1, -0.01,  # >= 0
                     freqmin-1]: # >= freqmin
            with self.assertRaises(ValueError):
                make_freqtuplist(freqmin=freqmin,
                                 freqmax=perr,
                                 freqspace=freqspace)
                pass
            pass
        pass

    def test_invalid_freqspace_value(self):
        """Invalid inputs caught"""
        freqmin = 5
        freqmax = 10

        for perr in [-1, -0.01]:  # >= 0
            with self.assertRaises(ValueError):
                make_freqtuplist(freqmin=freqmin,
                                 freqmax=freqmax,
                                 freqspace=perr)
                pass
            pass
        pass


    def test_freqmin_eq_freqmax(self):
        """Test short-circuit"""
        freqmin = 5
        freqmax = freqmin
        freqspace = 1

        out = make_freqtuplist(freqmin, freqmax, freqspace)
        self.assertTrue(len(out) == 0)
        pass


    def test_freqspace_zero(self):
        """Test short-circuit"""
        freqmin = 5
        freqmax = 10
        freqspace = 0

        out = make_freqtuplist(freqmin, freqmax, freqspace)
        self.assertTrue(len(out) == 0)
        pass


    def test_bounded(self):
        """
        Check that freqmin/freqmax hold even when freqspace doesn't divide
        evenly
        """
        freqmin = 5
        freqmax = 10
        freqspace = 8/7

        out = make_freqtuplist(freqmin, freqmax, freqspace)
        for tup in out:
            r = np.sqrt(tup[0]**2 + tup[1]**2)
            self.assertTrue(r >= freqmin)
            self.assertTrue(r <= freqmax)
            pass
        pass


    def test_on_bounds(self):
        """
        Check that freqmin/freqmax are inclusive using freqspace that does
        divide evenly
        """
        tol = 1e-13

        freqmin = 5
        freqmax = 10
        freqspace = 1

        out = make_freqtuplist(freqmin, freqmax, freqspace)
        gotfreqmin = False
        gotfreqmax = False
        for tup in out:
            r = np.sqrt(tup[0]**2 + tup[1]**2)
            if np.abs(r - freqmin) < tol:
                gotfreqmin = True
                pass
            if np.abs(r - freqmax) < tol:
                gotfreqmax = True
                pass
            pass

        self.assertTrue(gotfreqmin)
        self.assertTrue(gotfreqmax)
        pass


    def test_exact(self):
        """Test exact solution"""
        freqmin = 1
        freqmax = 3
        freqspace = 1
        # tup    | dist | use?
        # (0, 0) | 0    |
        # (0, 1) | 1    |  x
        # (0, 2) | 2    |  x
        # (0, 3) | 3    |  x
        # (1, 0) | 1    |  x
        # (1, 1) | 1.41 |  x
        # (1, 2) | 2.24 |  x
        # (1, 3) | 3.16 |
        # (2, 0) | 2    |  x
        # (2, 1) | 2.24 |  x
        # (2, 2) | 2.83 |  x
        # (2, 3) | 3.61 |
        # (3, 0) | 3    |  x
        # (3, 1) | 3.16 |
        # (3, 2) | 3.61 |
        # (3, 3) | 4.24 |
        # plus -y versions of these

        # Use a set since there's no ordering promised by make_freqtuplist
        test_tup = {(0, 1), (0, -1),
                    (0, 2), (0, -2),
                    (0, 3), (0, -3),
                    (1, 0),
                    (1, 1), (1, -1),
                    (1, 2), (1, -2),
                    (2, 0),
                    (2, 1), (2, -1),
                    (2, 2), (2, -2),
                    (3, 0),
                    }

        out = make_freqtuplist(freqmin, freqmax, freqspace)
        sout = set(out)
        sym_diff = test_tup ^ sout # elements in one or the other but not both
        self.assertTrue(len(sym_diff) == 0)
        pass





if __name__ == '__main__':
    unittest.main()
