# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""This script contains unit tests for the module `change_dm_shape`"""


import unittest

import numpy as np

from .vdm_check import check_valid, check_tie_dead, dn_to_eu, eu_to_dn, \
     check_valid_eu, check_valid_dn
from .vdm_check import _check_high_low_limit
from .vdm_check import MAXV, BITFACTOR

def increasing_data():
    """[0  1  2  3  4
        5  6  7  8  9
        10 11 12 13 14
        15 16 17 18 19
        20 21 22 23 24]"""
    return np.arange(25).reshape((5, 5)).astype(int)

def data_to_fail():
    """returns a matrix of all ones (5x5)
    except 2's along the diagonal"""
    a = np.ones((5, 5))
    a = a + np.diag(np.repeat(1, 5))
    return a

def data_ones():
    """what it looks like"""
    return np.ones((5, 5))

def data_start():
    """a triangular data array, starts at 0 along the columns,
       each column rises by one until size//2, then back down
       to zero"""
    size = 12
    start_array = np.zeros((size, size))
    start_array[:, size//2-1:] = np.arange(size//2+1)[::-1]
    start_array[:, 0:size//2] = np.arange(size//2)
    return start_array

def data_end():
    size = 12
    return np.zeros((size, size))

class Test_check_high_low_limit(unittest.TestCase):
    """Tests the function that checks whether an array
       is within the right bounds"""
    def setUp(self):
        self.data = increasing_data()

    def test_toohigh(self):
        """Fails as expected at high limit"""
        self.assertFalse(
            _check_high_low_limit(self.data, high_limit=self.data.max()-1,
                                  low_limit=self.data.min()))

    def test_toolow(self):
        """Fails as expected at low limit"""
        self.assertFalse(
            _check_high_low_limit(self.data, high_limit=self.data.max(),
                                  low_limit=self.data.min()+1))

    def test_within_bounds(self):
        """checks if array is between high and low limits"""
        _check_high_low_limit(self.data, high_limit=self.data.max(),
                              low_limit=self.data.min())

    def test_bad_data(self):
        """Checks bad input fails correctly"""
        for perr in [np.ones((3, 3, 3)), None, 'txt', 1j, 5]:
            with self.assertRaises(TypeError):
                _check_high_low_limit(array=perr,
                                      high_limit=self.data.max(),
                                      low_limit=self.data.min())

    def test_bad_high_limit(self):
        """Checks bad input fails correctly"""
        for perr in [np.ones((3, 3, 3)), None, 'txt', 1j, np.eye(3)]:
            with self.assertRaises(TypeError):
                _check_high_low_limit(array=self.data,
                                      high_limit=perr,
                                      low_limit=self.data.min())


    def test_bad_low_limit(self):
        """Checks bad input fails correctly"""
        for perr in [np.ones((3, 3, 3)), None, 'txt', 1j, np.eye(3)]:
            with self.assertRaises(TypeError):
                _check_high_low_limit(array=self.data,
                                      high_limit=self.data.max(),
                                      low_limit=perr)


class Test_check_valid(unittest.TestCase):
    """A test to determine if the check_valid function passes and fails
    for various arrays"""
    def setUp(self):
        self.data = data_ones()
        self.bigeye = 5.0*np.eye(5)

    def test_equal_pass(self):
        """Uniform array passes as expected with no flatmap"""
        self.assertTrue(check_valid(self.data, plus_limit=1, diag_limit=1))


    def test_edge_pass(self):
        """Passes as expected if on edge of lateral and diagonal"""
        ok = np.array([[0, 0],
                       [1, 1]])
        self.assertTrue(check_valid(ok, plus_limit=1, diag_limit=1))
        pass


    def test_nr_plus_horizonatal_fail(self):
        """Fails as expected if lateral horizontal NR failure but no diag"""
        badplus = np.array([[0, 2],
                            [1, 1]])
        self.assertFalse(check_valid(badplus, plus_limit=1, diag_limit=1))
        pass

    def test_nr_plus_vertical_fail(self):
        """Fails as expected if lateral vertical NR failure but no diag"""
        badplus = np.array([[0, 1],
                            [2, 1]])
        self.assertFalse(check_valid(badplus, plus_limit=1, diag_limit=1))
        pass

    def test_nr_diag_right_fail(self):
        """Fails as expected if diagonal right NR failure but no plus"""
        baddiag = np.array([[0, 1],
                            [1, 2]])
        self.assertFalse(check_valid(baddiag, plus_limit=1, diag_limit=1))
        pass

    def test_nr_diag_left_fail(self):
        """Fails as expected if diagonal left NR failure but no plus"""
        baddiag = np.array([[1, 0],
                            [2, 1]])
        self.assertFalse(check_valid(baddiag, plus_limit=1, diag_limit=1))
        pass

    def test_nr_plus_diag_fail(self):
        """Fails as expected if diagonal and plus NR failure"""
        badboth = np.array([[0, 0],
                            [2, 4]])
        self.assertFalse(check_valid(badboth, plus_limit=1, diag_limit=1))
        pass

    def test_pass_dmflat(self):
        """
        An array that should fail NR checks passes if the dmflat is the
        same (i.e. no penalty for making surface phase-flat)
        """
        self.assertTrue(check_valid(self.bigeye,
                                    plus_limit=1,
                                    diag_limit=1,
                                    dmflat=self.bigeye))

    def test_equal_fail_bounds_high(self):
        self.assertFalse(
            check_valid(self.data, plus_limit=1, diag_limit=1, high_limit=0.5))

    def test_equal_fail_bounds_low(self):
        self.assertFalse(
            check_valid(self.data, plus_limit=1, diag_limit=1, low_limit=1.5))

    def test_bad_array(self):
        """Checks bad input fails correctly"""
        for perr in [np.ones((3, 3, 3)), None, 'txt', 1j, 5]:
            with self.assertRaises(TypeError):
                check_valid(array=perr, plus_limit=None,
                            diag_limit=None, high_limit=None,
                            low_limit=None)
                pass
            pass
        pass

    def test_bad_plus_limit(self):
        """Checks bad input fails correctly"""
        for perr in [np.ones((3, 3, 3)), None, 'txt', 1j, np.eye(3), 0, -1]:
            with self.assertRaises(TypeError):
                check_valid(array=self.data, plus_limit=perr,
                            diag_limit=None, high_limit=None,
                            low_limit=None)
                pass
            pass
        pass

    def test_bad_diag_limit(self):
        """Checks bad input fails correctly"""
        for perr in [np.ones((3, 3, 3)), None, 'txt', 1j, np.eye(3), 0, -1]:
            with self.assertRaises(TypeError):
                check_valid(array=self.data, plus_limit=None,
                            diag_limit=perr, high_limit=None,
                            low_limit=None)
                pass
            pass
        pass

    def test_bad_high_limit(self):
        """Checks bad input fails correctly"""
        for perr in [np.ones((3, 3, 3)), 'txt', 1j, np.eye(3)]:
            with self.assertRaises(TypeError):
                check_valid(array=self.data, plus_limit=None,
                            diag_limit=None, high_limit=perr,
                            low_limit=None)
                pass
            pass
        pass

    def test_bad_low_limit(self):
        """Checks bad input fails correctly"""
        for perr in [np.ones((3, 3, 3)), 'txt', 1j, np.eye(3)]:
            with self.assertRaises(TypeError):
                check_valid(array=self.data, plus_limit=None,
                            diag_limit=None, high_limit=None,
                            low_limit=perr)
                pass
            pass
        pass

    def test_bad_dmflat(self):
        """Checks bad input fails correctly"""
        for perr in [np.ones((3, 3, 3)), None, 'txt', 1j, 5]:
            with self.assertRaises(TypeError):
                check_valid(array=self.data, plus_limit=None,
                            diag_limit=None, high_limit=None,
                            low_limit=None, dmflat=perr)
                pass
            pass
        pass


class TestCheckTieDead(unittest.TestCase):
    """
    Test for function to do checking of tied/dead actuators
    """

    def setUp(self):
        # Use small arrays so we can write this in manually
        self.tie_no_tied_dead = np.array([[0, 0, 0],
                                          [0, 0, 0],
                                          [0, 0, 0]])
        self.volts_no_tied_dead = np.array([[1, 2, 3],
                                            [4, 5, 6],
                                            [7, 8, 9]])
        self.tie_tied = np.array([[0, 1, 0],
                                  [0, 1, 3],
                                  [2, 2, 0]])
        self.volts_tied = np.array([[1, 2, 3],
                                    [4, 2, 6],
                                    [7, 7, 9]])
        self.tie_dead = np.array([[-1, 0, -1],
                                  [0, 0, 0],
                                  [0, 0, 0]])
        self.volts_dead = np.array([[0, 2, 0],
                                    [4, 5, 6],
                                    [7, 8, 9]])
        self.tie_tied_dead = np.array([[-1, 1, -1],
                                       [0, 1, 3],
                                       [2, 2, 0]])
        self.volts_tied_dead = np.array([[0, 2, 0],
                                         [4, 2, 6],
                                         [7, 7, 9]])

        pass


    def test_success_no_tied_dead(self):
        """Succeeds for known-good data"""
        self.assertTrue(check_tie_dead(self.volts_no_tied_dead,
                                       self.tie_no_tied_dead))
        pass


    def test_success_tied(self):
        """Succeeds for known-good data"""
        self.assertTrue(check_tie_dead(self.volts_tied,
                                       self.tie_tied))
        pass


    def test_success_dead(self):
        """Succeeds for known-good data"""
        self.assertTrue(check_tie_dead(self.volts_dead,
                                       self.tie_dead))
        pass


    def test_success_tied_dead(self):
        """Succeeds for known-good data"""
        self.assertTrue(check_tie_dead(self.volts_tied_dead,
                                       self.tie_tied_dead))
        pass


    def test_failure_tied(self):
        """Fails when tied actuators do not match"""
        for v in [self.volts_no_tied_dead, self.volts_dead]:
            self.assertFalse(check_tie_dead(v, self.tie_tied))
            pass
        pass


    def test_failure_dead(self):
        """Fails when dead actuators are not zero"""
        for v in [self.volts_no_tied_dead, self.volts_tied]:
            self.assertFalse(check_tie_dead(v, self.tie_dead))
            pass
        pass


    def test_failure_tied_dead(self):
        """
        Fails when tied actuators do not match or dead actuators are not zero
        """
        for v in [self.volts_no_tied_dead, self.volts_dead, self.volts_tied]:
            self.assertFalse(check_tie_dead(v, self.tie_tied_dead))
            pass
        pass


    def test_invalid_volts(self):
        """Verifies invalid data caught"""
        for x in [np.ones((2, 2, 2)), np.ones((2,)), 'txt', None, 1.0]:
            with self.assertRaises(TypeError):
                check_tie_dead(x, self.tie_no_tied_dead)
                pass
            pass
        pass


    def test_invalid_tie(self):
        """Verifies invalid data caught"""
        for x in [np.ones((2, 2, 2)), np.ones((2,)), 'txt', None, 1.0]:
            with self.assertRaises(TypeError):
                check_tie_dead(self.volts_no_tied_dead, x)
                pass
            pass
        pass


    def test_volts_and_tie_different_shapes(self):
        """Verifies invalid data caught"""
        for x in [self.volts_no_tied_dead[:-1, :-1]]:
            with self.assertRaises(TypeError):
                check_tie_dead(x, self.tie_no_tied_dead)
                pass
            pass
        pass


class TestDNEU(unittest.TestCase):
    """
    Test functions to convert between DN and EU

    test success
    input validity for both
    check zero matches, 110-1/2**16 matches
    check idempotence after first iteration (but not zeroth), using dense grid
    """

    def test_dn_to_eu_success(self):
        """valid input succeeds"""
        inarr = np.array([[0, BITFACTOR-1]]).astype('uint16')
        dn_to_eu(inarr)
        pass


    def test_eu_to_dn_success(self):
        """valid input succeeds"""
        inarr = np.array([[0.0, MAXV - 1/BITFACTOR]])
        eu_to_dn(inarr)
        pass


    def test_invalid_dn(self):
        """invalid input caught"""
        for x in [np.ones((2, 2, 2)).astype('uint16'),
                  np.ones((2,)).astype('uint16'), 'txt', None, 1.0]:
            with self.assertRaises(TypeError):
                dn_to_eu(x)
                pass
            pass
        pass


    def test_invalid_eu(self):
        """invalid input caught"""
        for x in [np.ones((2, 2, 2)), np.ones((2,)), 'txt', None, 1.0]:
            with self.assertRaises(TypeError):
                eu_to_dn(x)
                pass
            pass
        pass


    def test_dn_exact(self):
        """Test endpoints match exactly"""
        inarr = np.array([[0, BITFACTOR-1]]).astype('uint16')
        targarr = np.array([[0.0, MAXV*(1 - 1/BITFACTOR)]])
        outarr = dn_to_eu(inarr)
        self.assertTrue((outarr - targarr == 0).all())
        pass


    def test_eu_exact(self):
        """Test endpoints match exactly"""
        inarr = np.array([[0.0, MAXV*(1 - 1/BITFACTOR)]])
        targarr = np.array([[0, BITFACTOR-1]]).astype('uint16')
        outarr = eu_to_dn(inarr)
        self.assertTrue((outarr - targarr == 0).all())
        pass


    def test_idempotence_after_one_cycle(self):
        """
        If an arbitrary EU set of values (in range) is converted to DN and
        back to EU, the values will not in general be the same unless they all
        just happened to line up with floating-point representations of the
        discrete steps.  However, if it's done a second time, they should be
        the same, since they were already matched to discrete steps.

        In either case the DN representations should always be the same.
        """
        inarr = np.linspace(0, MAXV*(1-1/BITFACTOR),
                            5*BITFACTOR)[:, np.newaxis]
        outdn1 = eu_to_dn(inarr)
        outeu1 = dn_to_eu(outdn1)
        outdn2 = eu_to_dn(outeu1)
        outeu2 = dn_to_eu(outdn2)

        self.assertFalse((inarr == outeu1).all())
        self.assertTrue((outdn1 == outdn2).all())
        self.assertTrue((outeu1 == outeu2).all())

        pass


class TestCheckValidEUDN(unittest.TestCase):
    """
    Tests for the upper-level checking functions, in EU and DN
    """

    def test_good(self):
        """
        Valid settings pass

        All voltage value constraints look alike when successfull, but need to
        test the combinations of dead and tied
        """

        flatmap = np.zeros((3, 3))

        # 0: no tied; no dead
        eu = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
        tie = np.zeros_like(eu)
        self.assertTrue(check_valid_eu(eu, flatmap, tie))
        dn = eu_to_dn(eu)
        self.assertTrue(check_valid_dn(dn, flatmap, tie))

        # 1: no tied; dead
        eu = np.array([[0, 2, 0],
                       [4, 5, 6],
                       [7, 8, 9]])
        tie = np.array([[-1, 0, -1],
                        [0, 0, 0],
                        [0, 0, 0]])
        self.assertTrue(check_valid_eu(eu, flatmap, tie))
        dn = eu_to_dn(eu)
        self.assertTrue(check_valid_dn(dn, flatmap, tie))

        # 2: tied; no dead
        eu = np.array([[1, 2, 3],
                       [4, 2, 6],
                       [7, 7, 9]])
        tie = np.array([[0, 1, 0],
                        [0, 1, 3],
                        [2, 2, 0]])
        self.assertTrue(check_valid_eu(eu, flatmap, tie))
        dn = eu_to_dn(eu)
        self.assertTrue(check_valid_dn(dn, flatmap, tie))

        # 3: tied; dead
        eu = np.array([[0, 2, 0],
                       [4, 2, 6],
                       [7, 7, 9]])
        tie = np.array([[-1, 1, -1],
                        [0, 1, 3],
                        [2, 2, 0]])
        self.assertTrue(check_valid_eu(eu, flatmap, tie))
        dn = eu_to_dn(eu)
        self.assertTrue(check_valid_dn(dn, flatmap, tie))

        pass


    def test_bad(self):
        """
        Settings that don't obey constraints are caught
        """
        # We'll create an array with six 3x3 regions separated by zeros, each
        # of which creates a different violation.  We'll toggle them on and off
        # and cycle through all 2^6 options

        flatmap = np.zeros((11, 7))
        vlat = 50
        vdiag = 75
        vmax = 100

        for high in [True, False]:
            for low in [True, False]:
                for lat in [True, False]:
                    for diag in [True, False]:
                        for ties in [True, False]:
                            for dead in [True, False]:
                                eu = np.zeros((11, 7))
                                tie = np.zeros((11, 7))

                                if high:
                                    eu[0:3, 0:3] = vmax + 1
                                    pass
                                if low:
                                    eu[4:7, 0:3] = -1
                                    pass
                                if lat:
                                    eu[8:11, 0:3] = np.array([[20, 20, 20],
                                                              [20, 80, 20],
                                                              [20, 20, 20]])
                                    pass
                                if diag:
                                    eu[0:3, 4:7] = np.array([[20, 50, 20],
                                                             [50, 100, 50],
                                                             [20, 50, 20]])
                                    pass
                                if ties:
                                    eu[4:7, 4:7] = np.array([[1, 2, 3],
                                                             [4, 5, 6],
                                                             [7, 8, 9]])
                                    tie[4:7, 4:7] = np.array([[1, 1, 1],
                                                              [1, 1, 1],
                                                              [1, 1, 1]])
                                    pass
                                if dead:
                                    eu[8:11, 4:7] = np.array([[1, 2, 3],
                                                              [4, 5, 6],
                                                              [7, 8, 9]])
                                    tie[8:11, 4:7] = np.array([[-1, -1, -1],
                                                               [-1, -1, -1],
                                                               [-1, -1, -1]])
                                    pass

                                if high or low or lat or diag or ties or dead:
                                    # i.e. if at least one thing is wrong with
                                    # it
                                    self.assertFalse(
                                        check_valid_eu(eu,
                                                       flatmap,
                                                       tie,
                                                       vlat=vlat,
                                                       vdiag=vdiag,
                                                       vmax=vmax,
                                                       ))

                                    if not high and not low:
                                        # High and low errors are not
                                        # representable in DNs
                                        dn = eu_to_dn(eu)
                                        self.assertFalse(
                                            check_valid_dn(dn,
                                                           flatmap,
                                                           tie,
                                                           vlat=vlat,
                                                           vdiag=vdiag,
                                                           vmax=vmax,
                                                           ))
                                        pass
                                    pass
                                pass
                            pass
                        pass
                    pass
                pass
            pass
        pass


    def test_invalid_dn(self):
        """invalid input caught"""
        flatmap = np.zeros((3, 3))
        tie = np.zeros((3, 3))

        for x in [np.ones((2, 2, 2)).astype('uint16'),
                  np.ones((2,)).astype('uint16'), 'txt', None, 1.0]:
            with self.assertRaises(TypeError):
                check_valid_dn(x, flatmap, tie)
                pass
            pass
        pass


    def test_invalid_eu(self):
        """invalid input caught"""
        flatmap = np.zeros((3, 3))
        tie = np.zeros((3, 3))

        for x in [np.ones((2, 2, 2)), np.ones((2,)), 'txt', None, 1.0]:
            with self.assertRaises(TypeError):
                check_valid_eu(x, flatmap, tie)
                pass
            pass
        pass


    def test_shape_mismatch(self):
        """Verify invalid input types caught"""
        dn = np.zeros((3, 3))
        eu = np.zeros((3, 3))
        flatmap = np.zeros((3, 3))
        tie = np.zeros((3, 3))

        wrong = np.zeros((4, 4))

        with self.assertRaises(TypeError):
            check_valid_eu(wrong, flatmap, tie)
        with self.assertRaises(TypeError):
            check_valid_eu(eu, wrong, tie)
        with self.assertRaises(TypeError):
            check_valid_eu(eu, flatmap, wrong)

        with self.assertRaises(TypeError):
            check_valid_dn(wrong, flatmap, tie)
        with self.assertRaises(TypeError):
            check_valid_dn(dn, wrong, tie)
        with self.assertRaises(TypeError):
            check_valid_dn(dn, flatmap, wrong)

        pass


    def test_invalid_vmax(self):
        """Verify invalid inputs caught"""
        dn = np.zeros((3, 3))
        eu = np.zeros((3, 3))
        flatmap = np.zeros((3, 3))
        tie = np.zeros((3, 3))

        for perr in [np.ones((2, 2, 2)), -1, 0, 1j, 'txt', (5,), None]:
            with self.assertRaises(TypeError):
                check_valid_eu(eu, flatmap, tie, vmax=perr)
                pass
            with self.assertRaises(TypeError):
                check_valid_dn(dn, flatmap, tie, vmax=perr)
                pass
            pass
        pass


    def test_invalid_vlat(self):
        """Verify invalid inputs caught"""
        dn = np.zeros((3, 3))
        eu = np.zeros((3, 3))
        flatmap = np.zeros((3, 3))
        tie = np.zeros((3, 3))

        for perr in [np.ones((2, 2, 2)), -1, 0, 1j, 'txt', (5,), None]:
            with self.assertRaises(TypeError):
                check_valid_eu(eu, flatmap, tie, vlat=perr)
                pass
            with self.assertRaises(TypeError):
                check_valid_dn(dn, flatmap, tie, vlat=perr)
                pass
            pass
        pass


    def test_invalid_vdiag(self):
        """Verify invalid inputs caught"""
        dn = np.zeros((3, 3))
        eu = np.zeros((3, 3))
        flatmap = np.zeros((3, 3))
        tie = np.zeros((3, 3))

        for perr in [np.ones((2, 2, 2)), -1, 0, 1j, 'txt', (5,), None]:
            with self.assertRaises(TypeError):
                check_valid_eu(eu, flatmap, tie, vdiag=perr)
                pass
            with self.assertRaises(TypeError):
                check_valid_dn(dn, flatmap, tie, vdiag=perr)
                pass
            pass
        pass




if __name__ == "__main__":
    unittest.main()
