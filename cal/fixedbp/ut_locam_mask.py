# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Unit tests for making a LOCAM software mask."""
import unittest
import numpy as np

from cal.fixedbp.locam_mask import compute_locam_mask


class TestMaskForLOCAM(unittest.TestCase):
    """Tests for building the LOCAM software mask."""

    def setUp(self):
        """Set up."""
        shape = (10, 11)
        self.allgood = np.ones(shape, dtype=bool)
        self.allbad = np.zeros(shape, dtype=bool)
        self.badedge = np.ones(shape, dtype=bool)
        self.badedge[0, :] = False
        self.badedge[-1, :] = False
        self.badedge[:, 0] = False
        self.badedge[:, -1] = False

        self.badrows = np.ones(shape, dtype=bool)
        self.badrows[0, :] = False
        self.badrows[-1, :] = False

        self.badcols = np.ones(shape, dtype=bool)
        self.badcols[:, 0] = False
        self.badcols[:, -1] = False

        pass


    def test_success(self):
        """Valid inputs complete successfully."""
        masklist = [self.allgood, self.badedge]
        compute_locam_mask(masklist)
        pass


    def test_correct_outputs_allgood(self):
        """check various exact results"""
        # allgood should produce the same as if it's not present

        m0 = compute_locam_mask([self.badedge])
        m1 = compute_locam_mask([self.badedge, self.allgood])
        self.assertTrue((m0 == m1).all())

        m0 = compute_locam_mask([self.allbad])
        m1 = compute_locam_mask([self.allbad, self.allgood])
        self.assertTrue((m0 == m1).all())

        m0 = compute_locam_mask([self.allgood])
        m1 = compute_locam_mask([self.allgood, self.allgood])
        self.assertTrue((m0 == m1).all())

        pass

    def test_correct_outputs_allbad(self):
        """check various exact results"""
        # allbad should produce allbad regardless of what else is there

        m0 = compute_locam_mask([self.badedge])
        m1 = compute_locam_mask([self.badedge, self.allbad])
        self.assertFalse((m0 == m1).all())
        self.assertTrue((m1 == self.allbad).all())

        m0 = compute_locam_mask([self.allbad])
        m1 = compute_locam_mask([self.allbad, self.allgood])
        self.assertTrue((m1 == self.allbad).all())

        m0 = compute_locam_mask([self.allgood])
        m1 = compute_locam_mask([self.allgood, self.allbad])
        self.assertFalse((m0 == m1).all())
        self.assertTrue((m1 == self.allbad).all())

        pass


    def test_correct_outputs_badedge(self):
        """check various exact results"""
        # badedge should be buildable from badrows and badcols

        m0 = compute_locam_mask([self.badedge])
        m1 = compute_locam_mask([self.badcols, self.badrows])
        self.assertTrue((m0 == m1).all())

        m0 = compute_locam_mask([self.badedge])
        m1 = compute_locam_mask([self.badrows, self.badcols, self.allgood])
        self.assertTrue((m0 == m1).all())

        pass


    def test_invalid_masklist(self):
        """Invalid inputs caught"""
        perrlist = [
            {2, 3}, 5, np.ones((3, 3)), None, 'txt',  # not a list
            [], # empty list
            [np.ones((3, 3))], # not bools
            [np.ones((3, 3), dtype=bool), np.ones((3, 4), dtype=bool)], #sizes
        ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                compute_locam_mask(perr)
            pass
        pass


if __name__ == '__main__':
    unittest.main()
