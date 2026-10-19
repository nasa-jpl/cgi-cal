# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Unit tests for constrain_dm function"""

import unittest
from unittest.mock import patch
import os

import astropy.io.fits as pyfits
import numpy as np

from .constrain_dm import constrain_dm, tie_with_matrix
from .constrain_dm import ConstrainDMException
from .dmsmooth import dmsmooth
from .vdm_check import check_valid_eu

here = os.path.dirname(os.path.abspath(__file__))

class TestConstrainDM(unittest.TestCase):
    """Tests for the function that applies voltage constraints and ties"""

    def setUp(self):
        """Object setup for the rest of the tests"""
        rng = np.random.default_rng(2021)
        self.volts = rng.random((48, 48))

        self.tie = np.zeros_like(self.volts)
        self.tie[0:2, 0:2] = 1
        self.tie[5:10, 5:10] = 2
        self.tie[6:8, 0:2] = 3
        self.tie[18, 18] = 4 # single-element group should still work
        self.tie[20, 20] = -1

        # Set equal to defaults
        self.vmin = 0.
        self.vmax = 100.
        self.vlat = 50.
        self.vdiag = 75.
        self.vquant = 110./2**16
        self.maxiter = 1000

        self.flatmap_flat = np.zeros_like(self.volts)
        self.flatmap_nonflat = 50 + 1.0*rng.normal(0, 1, (48, 48))
        self.flatmap_nonflat[self.tie == 1] = 50
        self.flatmap_nonflat[self.tie == 2] = 51
        self.flatmap_nonflat[self.tie == 3] = 49
        self.flatmap_nonflat[self.tie == 4] = 50.5
        self.flatmap_nonflat[self.tie == -1] = 0

        self.checkerboard = np.array([i%2 for i in range(49*49)]) * self.vmax
        self.checkerboard = self.checkerboard.reshape((49, 49))
        self.checkerboard = self.checkerboard[:48, :48]


    def test_success_basic(self):
        """
        Verify a setting equal to its flatmap, with no ties, completes.
        """
        constrain_dm(self.flatmap_flat,
                     self.flatmap_flat,
                     np.zeros_like(self.tie))
        pass


    def test_success_flat(self):
        """
        Verify a setting with smoothing and tying completes with a uniform
        flatmap
        """
        constrain_dm(self.volts, self.flatmap_flat, self.tie)
        pass


    def test_success_nonflat(self):
        """
        Verify a setting with smoothing and tying completes with a non-uniform
        flatmap
        """
        constrain_dm(self.volts, self.flatmap_nonflat, self.tie)
        pass


    def test_nominal(self):
        """
        Verify method returns same array when input is uniform (except dead).
        """
        dmuniform = np.ones_like(self.volts)
        dmout = constrain_dm(dmuniform, self.flatmap_flat, self.tie)
        self.assertTrue((dmuniform[self.tie != -1]
                         == dmout[self.tie != -1]).all())


    def test_only_smooth(self):
        """Test dmsmooth by itself (no ties)."""

        # Made for 30V neighbor rules
        dmnr = np.array(
            [
                [40, 0, 20],
                [20, 20, 0],
                [20, 40, 20],
            ]
        )

        flatmap = np.zeros((3, 3))
        tie = np.zeros((3, 3)) # no ties

        dmout = constrain_dm(dmnr, flatmap, tie, vlat=30, vdiag=30)
        isgood = check_valid_eu(
            eu=dmout,
            flatmap=flatmap,
            tie=tie,
            vmax=self.vmax,
            vlat=30.,
            vdiag=30.,
        )

        self.assertTrue(isgood)
        pass


    def test_smooth_and_ties(self):
        """Analytically test dmsmooth used alongside tie_with_matrix."""

        # Made for 30V neighbor rules
        dmnr = np.array(
            [
                [40, 0, 20],
                [20, 20, 0],
                [20, 40, 20],
            ]
        )

        flatmap = np.zeros((3, 3))

        # Tie upper corners
        tie = np.zeros((3, 3))
        tie[0, 0] = 1
        tie[0, 2] = 1

        dmout = constrain_dm(dmnr, flatmap, tie, vlat=30, vdiag=30)
        isgood = check_valid_eu(
            eu=dmout,
            flatmap=flatmap,
            tie=tie,
            vmax=self.vmax,
            vlat=30.,
            vdiag=30.,
        )
        self.assertTrue(isgood)
        pass


    def test_tie_smooth_converge(self):
        """Verify that tie and smooth converge"""
        out = constrain_dm(self.checkerboard,
                           self.flatmap_flat,
                           self.tie,
                           vmax=self.vmax,
                           vlat=self.vlat,
                           vdiag=self.vdiag,
                           vquant=self.vquant,
                           maxiter=self.maxiter)

        # Check still tied (calling tie_with_matrix should change nothing)
        check_tie = tie_with_matrix(out.copy(), self.tie)
        self.assertTrue((out == check_tie).all())

        # Check still smoothed (calling dmsmooth should change nothing)
        check_smooth = dmsmooth(out.copy(), self.vmax,
                                self.vquant, self.vlat, self.vdiag,
                                dmflat=self.flatmap_flat)
        self.assertTrue((out == check_smooth).all())


    def test_stress(self):
        """
        Verify test completes even in the most stressing realistic case we've
        found, using default maxiter (to support use within higher-level
        functions)

        See PFR 218113
        """
        input_volts = (self.vmax+1)*np.ones((48, 48))
        flatmap = np.zeros((48, 48))
        tiemap = pyfits.getdata(os.path.join(here, 'testdata',
                                             'dm2_tied_actuator_map.fits'))
        constrain_dm(input_volts,
                     flatmap,
                     tiemap,
                     vmax=self.vmax,
                     vlat=self.vlat,
                     vdiag=self.vdiag,
                     vquant=self.vquant,
        )


    def test_full_bit_width_vquant(self):
        """
        Verify test completes even with a vquant uses the full bit width of
        a float64, to verify we don't still have accumulator issues.

        See PFR 218113
        """
        input_volts = (self.vmax+1)*np.ones((48, 48))
        flatmap = np.zeros((48, 48))
        tiemap = pyfits.getdata(os.path.join(here, 'testdata',
                                             'dm1_tied_actuator_map.fits'))
        vquant = 0.0016784668 # rounded from 0.001678466796875 = 110/2**16
        constrain_dm(input_volts,
                     flatmap,
                     tiemap,
                     vmax=self.vmax,
                     vlat=self.vlat,
                     vdiag=self.vdiag,
                     vquant=vquant,
        )


    def test_fuzz(self):
        """
        Run a fuzz test to see if we turn up any other failure modes

        see PFR 218113
        """
        dm1_ties = pyfits.getdata(os.path.join(here, 'testdata',
                                               'dm1_tied_actuator_map.fits'))
        dm2_ties = pyfits.getdata(os.path.join(here, 'testdata',
                                               'dm2_tied_actuator_map.fits'))
        tielist = [dm1_ties, dm2_ties]
        flatmap = np.zeros((48, 48))

        rng = np.random.default_rng(218113000)
        for _ in range(20):
            volts = -1 + 102*rng.random((48, 48))
            tie = tielist[rng.integers(2)]
            vmax = 100 + (2*rng.random() - 1)
            vlat = 50 + (2*rng.random() - 1)
            vdiag = 75 + (2*rng.random() - 1)
            vquant = 110/(2**16 + rng.integers(100) - 50)
            constrain_dm(volts, flatmap, tie, vmax, vlat, vdiag, vquant)
            pass
        pass


    def test_known_noquant_failure(self):
        """
        Verify test succeeds with inputs known to fail in the vquant=0 case

        Failure here == enter an infinite loop
        """
        # Example from A.J. Riggs/Garreth Ruane found in DST use
        flatmapInit = pyfits.getdata(os.path.join(
            here,
            'testdata',
            'AOX_48.4_flatmapVMU_20220509_maskA_1_noNRcon.fits'
        ))
        facesheetFlatmap = pyfits.getdata(os.path.join(
            here,
            'testdata',
            'AOX_48.4_flatvolts-20180902-45vbias-30vcorners.fits'
        )).T

        vlat = 16.
        vdiag = 37.
        tie = np.zeros_like(flatmapInit)
        maxiter = 1000
        vmax = 100
        vquant = 110./2**16 # nonzero

        flatmapFinal = constrain_dm(
            volts=flatmapInit,
            flatmap=facesheetFlatmap,
            tie=tie,
            vmax=vmax,
            vlat=vlat,
            vdiag=vdiag,
            vquant=vquant,
            maxiter=maxiter,
        )
        isgood = check_valid_eu(
            eu=flatmapFinal,
            flatmap=facesheetFlatmap,
            tie=tie,
            vmax=self.vmax,
            vlat=vlat,
            vdiag=vdiag,
        )
        self.assertTrue(isgood)
        pass


    @patch('cal.util.constrain_dm.tie_with_matrix')
    def test_maxiter(self, mock_tie_with_matrix):
        """Verify that method will not repeat indefinitely"""
        mock_tie_with_matrix.return_value = self.checkerboard # not smooth
        with self.assertRaises(ConstrainDMException):
            constrain_dm(volts=self.checkerboard,
                         flatmap=self.flatmap_flat,
                         tie=np.zeros_like(self.flatmap_flat),
                         maxiter=self.maxiter)
            pass
        pass


    @patch('cal.util.constrain_dm.tie_with_matrix')
    def test_maxiter_default(self, mock_tie_with_matrix):
        """Verify that method will not repeat indefinitely with default"""
        mock_tie_with_matrix.return_value = self.checkerboard # not smooth
        with self.assertRaises(ConstrainDMException):
            constrain_dm(volts=self.checkerboard,
                         flatmap=self.flatmap_flat,
                         tie=np.zeros_like(self.flatmap_flat))
            pass
        pass


    def test_invalid_volts(self):
        """Verify invalid data caught"""
        for volts in [np.ones((2, 2, 2)), np.ones((2,)), 'txt', None, 1.0]:
            with self.assertRaises(TypeError):
                constrain_dm(volts=volts,
                             flatmap=self.flatmap_flat,
                             tie=self.tie)
                pass
            pass
        pass


    def test_invalid_flatmap(self):
        """Verify invalid data caught"""
        for flat in [np.ones((2, 2, 2)), np.ones((2,)), 'txt', None, 1.0]:
            with self.assertRaises(TypeError):
                constrain_dm(volts=self.volts,
                             flatmap=flat,
                             tie=self.tie)
                pass
            pass
        pass


    def test_invalid_tie(self):
        """Verify invalid data caught"""
        for tie in [np.ones((2, 2, 2)), np.ones((2,)), 'txt', None, 1.0]:
            with self.assertRaises(TypeError):
                constrain_dm(volts=self.volts,
                             flatmap=self.flatmap_flat,
                             tie=tie)
                pass
            pass
        pass


    def test_volts_and_flatmap_different_shape(self):
        """Verify invalid data caught"""
        f0 = self.flatmap_flat[:-1, :-1].copy()
        with self.assertRaises(TypeError):
            constrain_dm(self.volts, f0, self.tie)
            pass
        pass


    def test_volts_and_tie_different_shape(self):
        """Verify invalid data caught"""
        t0 = self.tie[:-1, :-1].copy()
        with self.assertRaises(TypeError):
            constrain_dm(self.volts, self.flatmap_flat, t0)
            pass
        pass


    def test_invalid_vmax(self):
        """Verify invalid data caught"""
        for v in [np.ones((2, 2, 2)), 1j, 0, 'txt', (5,), None]:
            with self.assertRaises(TypeError):
                constrain_dm(self.volts, self.flatmap_flat, self.tie,
                             vmax=v)
                pass
            pass
        pass


    def test_invalid_vlat(self):
        """Verify invalid data caught"""
        for v in [np.ones((2, 2, 2)), -1, 0, 1j, 'txt', (5,), None]:
            with self.assertRaises(TypeError):
                constrain_dm(self.volts, self.flatmap_flat, self.tie,
                             vlat=v)
                pass
            pass
        pass


    def test_invalid_vdiag(self):
        """Verify invalid data caught"""
        for v in [np.ones((2, 2, 2)), -1, 0, 1j, 'txt', (5,), None]:
            with self.assertRaises(TypeError):
                constrain_dm(self.volts, self.flatmap_flat, self.tie,
                             vdiag=v)
                pass
            pass
        pass


    def test_invalid_vquant(self):
        """Verify invalid data caught"""
        for v in [np.ones((2, 2, 2)), -1, 1j, 'txt', (5,), None]:
            with self.assertRaises(TypeError):
                constrain_dm(self.volts, self.flatmap_flat, self.tie,
                             vquant=v)
                pass
            pass
        pass


    def test_invalid_maxiter(self):
        """Verify invalid data caught"""
        for m in [np.ones((2, 2, 2)), -1, 0, 1.5, 1j, 'txt', (5,), None]:
            with self.assertRaises(TypeError):
                constrain_dm(self.volts, self.flatmap_flat, self.tie,
                             maxiter=m)
                pass
            pass
        pass


    def test_flatmap_constraints_in_place(self):
        """Verify invalid values caught"""
        # all should be >= 0, <= vmax, all ties at same voltage, and all
        # dead actuator at 0V

        volts = np.zeros((3, 3))
        tie = np.array([[0, 1, 0],
                        [0, 0, 1],
                        [-1, 0, 0]])

        # vmin
        flat1 = np.array([[self.vmin-1, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]])
        with self.assertRaises(ValueError):
            constrain_dm(volts, flat1, tie)
            pass

        # vmax
        flat2 = np.array([[self.vmax+1, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]])
        with self.assertRaises(ValueError):
            constrain_dm(volts, flat2, tie)
            pass

        # ties same V
        flat3 = np.array([[0, 1, 0],
                          [0, 0, 0],
                          [0, 0, 0]])
        with self.assertRaises(ValueError):
            constrain_dm(volts, flat3, tie)
            pass

        # dead @ 0V
        flat4 = np.array([[0, 0, 0],
                          [0, 0, 0],
                          [1, 0, 0]])
        with self.assertRaises(ValueError):
            constrain_dm(volts, flat4, tie)
            pass

        pass


    def test_tie_constraints_in_place(self):
        """Verify invalid values are caught"""
        # integers which are 0, -1, or consecutive 1->N

        volts = np.zeros((3, 3))
        flatmap = np.zeros((3, 3))

        # not consecutive
        tie1 = np.array([[0, 1, 2],
                         [0, 0, 0],
                         [0, 0, 4]])
        with self.assertRaises(ValueError):
            constrain_dm(volts, flatmap, tie1)
            pass

        # integers beyond -1
        tie2 = np.array([[0, 1, 2],
                         [0, 0, 0],
                         [0, 0, -2]])
        with self.assertRaises(ValueError):
            constrain_dm(volts, flatmap, tie2)
            pass

        # non-integers
        tie1 = np.array([[0, 1, 2],
                         [0, 0, 0],
                         [0, 0, 0.5]])
        with self.assertRaises(ValueError):
            constrain_dm(volts, flatmap, tie1)
            pass

        pass



class TestTieWithMatrix(unittest.TestCase):
    """Tests for tie_with_matrix function."""

    def setUp(self):
        """Object setup for the rest of the tests"""
        rng = np.random.default_rng(2021)
        self.dmin = rng.random((48, 48))
        self.dmin_untied = self.dmin.copy()

        # tie everything in advance, dmin_untied should produce this
        self.dmin[0:2, 0:2] = np.mean(self.dmin[0:2, 0:2])
        self.dmin[5:10, 5:10] = np.mean(self.dmin[5:10, 5:10])
        self.dmin[6:8, 0:2] = np.mean(self.dmin[6:8, 0:2])
        self.dmin[20, 20] = 0 # dead

        self.tie = np.zeros_like(self.dmin)
        self.tie[0:2, 0:2] = 1
        self.tie[5:10, 5:10] = 2
        self.tie[6:8, 0:2] = 3
        self.tie[18, 18] = 4 # single-element group should still work
        self.tie[20, 20] = -1

        pass


    def test_success(self):
        """Basic call with good inputs completes successfully"""
        tie_with_matrix(self.dmin_untied, self.tie)
        pass


    def test_output_matches(self):
        """Verify known input matches known output"""
        out = tie_with_matrix(self.dmin_untied, self.tie)
        self.assertTrue((out == self.dmin).all())
        pass


    def test_idempotence(self):
        """Verify that a tied matrix produces itself (i.e. all tying done)"""
        out = tie_with_matrix(self.dmin, self.tie)
        self.assertTrue((out == self.dmin).all())
        pass


    def test_actually_tied_as_expected(self):
        """
        Verify that the tied regions are all identical and at the mean of the
        input

        Except dead regions are 0, and untied regions are unchanged
        """
        out = tie_with_matrix(self.dmin_untied, self.tie)

        # get a list of tie elements
        tienums = np.unique(self.tie)
        for t in tienums:
            if t == 0: # not tied
                # should not change
                self.assertTrue((self.dmin_untied[self.tie == t] ==
                                 out[self.tie == t]).all())
                pass
            elif t == -1: # dead
                # should be zeroed
                self.assertTrue((out[self.tie == t] == 0).all())
                pass
            else:
                # should be mean of input range
                m = np.mean(self.dmin_untied[self.tie == t])
                self.assertTrue((out[self.tie == t] == m).all())
                pass
            pass
        pass


    def test_invalid_volts(self):
        """Bad inputs caught as expected"""
        badlist = [np.ones((48,)), np.ones((2, 48, 48)),
                   5, (5,), 1j, 'txt', None]

        for b in badlist:
            with self.assertRaises(TypeError):
                tie_with_matrix(b, self.tie)
                pass
            pass
        pass


    def test_invalid_tie(self):
        """Bad inputs caught as expected"""
        badlist = [np.ones((48,)), np.ones((2, 48, 48)),
                   5, (5,), 1j, 'txt', None]

        for b in badlist:
            with self.assertRaises(TypeError):
                tie_with_matrix(self.dmin_untied, b)
                pass
            pass
        pass


    def test_volts_and_tie_not_same_size(self):
        """Bad inputs caught as expected"""
        with self.assertRaises(TypeError):
            tie_with_matrix(np.ones((48, 48)), np.ones((49, 49)))
            pass
        pass


    def test_tie_not_made_of_correct_values(self):
        """Bad inputs caught as expected"""
        dmin22 = np.ones((2, 2))
        # expect consecutive integers 1 -> N, along with 0 or -1
        badlist = [np.array([[0.5, 0.5], [0.5, 0.5]]), # not integers
                   np.array([[1, 2], [3, 5]]), # not consecutive
                   np.array([[0, 1], [2, 4]]), # not consecutive w/ 0
                   np.array([[-1, 1], [2, 4]]), # not consecutive w/ -1
                   np.array([[0, -1], [1, 3]]), # not consecutive w/ 0 + -1
                   np.array([[1, 2], [3, -2]]), # not consective (negative)
                   np.array([[2, 3], [4, 5]]), # consecutive but not from 0
                  ]

        for b in badlist:
            with self.assertRaises(ValueError):
                tie_with_matrix(dmin22, b)
                pass
            pass
        pass


    def test_tie_is_made_of_correct_values(self):
        """Good inputs allowed through as expected"""
        dmin22 = np.ones((2, 2))
        # expect consecutive integers 1 -> N, along with 0 or -1
        goodlist = [np.array([[0, 0], [0, 0]]), # all 0
                    np.array([[-1, -1], [-1, -1]]), # all -1
                    np.array([[0, -1], [-1, 0]]), # 0 and -1
                    np.array([[1, 1], [1, 1]]), # all 1
                    np.array([[1, 2], [1, 3]]), # consecutive with a group
                    np.array([[1, 2], [1, 2]]), # consecutive with groups
                    np.array([[1, 3], [2, 4]]), # consecutive but all individ.
                    np.array([[0, 1], [2, 2]]), # 1->N with 0
                    np.array([[-1, 1], [2, 2]]), # 1->N with -1
                    np.array([[0, 1], [-1, 2]]), # 1->N with 0 and -1
                   ]

        for g in goodlist:
            tie_with_matrix(dmin22, g)
            pass
        pass





if __name__ == '__main__':
    unittest.main()
