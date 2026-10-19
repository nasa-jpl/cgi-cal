# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Unit tests for constrain_dm function"""

import unittest

import numpy as np

from .flat_tie import checktie, checkflat

class TestCheckTie(unittest.TestCase):
    """
    Tests for the function that validates tie matrices
    """

    def test_tie_not_made_of_correct_values(self):
        """Bad inputs caught as expected"""
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
            self.assertFalse(checktie(b))
            pass
        pass


    def test_tie_is_made_of_correct_values(self):
        """Good inputs allowed through as expected"""

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
            self.assertTrue(checktie(g))
            pass
        pass


    def test_tie_invalid_input(self):
        """Catches invalid type input"""

        for x in [np.ones((2, 2, 2)), np.ones((2,)), 'txt', None, 1.0]:
            with self.assertRaises(TypeError):
                checktie(x)
            pass
        pass


class TestCheckFlat(unittest.TestCase):
    """
    Tests for function to validate flatmaps

    invalid inputs for all four
    good flats go through
    bad flats caught
    """

    def test_good(self):
        """Verify good flats return True as expected"""
        vmin = 0.0
        vmax = 100.0
        tie = np.array([[0, 1, 0],
                        [0, 0, 1],
                        [-1, 0, 0]])

        flatmap = np.array([[0, 2, 100],
                            [4, 5, 2],
                            [0, 8, 9]])
        self.assertTrue(checkflat(flatmap, vmin, vmax, tie))


    def test_bad(self):
        """Verify bad flats return False as expected"""
        # all should be >= 0, <= vmax, all ties at same voltage, and all
        # dead actuator at 0V

        vmin = 0.0
        vmax = 100.0
        tie = np.array([[0, 1, 0],
                        [0, 0, 1],
                        [-1, 0, 0]])

        # vmin
        flat1 = np.array([[vmin-1, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]])
        self.assertFalse(checkflat(flat1, vmin, vmax, tie))


        # vmax
        flat2 = np.array([[0, 0, vmax+1],
                          [0, 0, 0],
                          [0, 0, 0]])
        self.assertFalse(checkflat(flat2, vmin, vmax, tie))

        # ties same V
        flat3 = np.array([[0, 1, 0],
                          [0, 0, 0],
                          [0, 0, 0]])
        self.assertFalse(checkflat(flat3, vmin, vmax, tie))

        # dead @ 0V
        flat4 = np.array([[0, 0, 0],
                          [0, 0, 0],
                          [1, 0, 0]])
        self.assertFalse(checkflat(flat4, vmin, vmax, tie))

        pass


    def test_invalid_flatmap(self):
        """Catches invalid type input"""
        tie = np.zeros((5, 5))
        vmin = 0
        vmax = 100

        for x in [np.ones((2, 2, 2)), np.ones((2,)), 'txt', None, 1.0]:
            with self.assertRaises(TypeError):
                checkflat(x, vmin, vmax, tie)
            pass
        pass


    def test_invalid_vmin(self):
        """Catches invalid type input"""
        flatmap = np.zeros((5, 5))
        tie = np.zeros((5, 5))
        vmax = 100

        for x in [np.ones((2, 2, 2)), 1j, 'txt', (5,), None]:
            with self.assertRaises(TypeError):
                checkflat(flatmap, x, vmax, tie)
            pass
        pass


    def test_invalid_vmax(self):
        """Catches invalid type input"""
        flatmap = np.zeros((5, 5))
        tie = np.zeros((5, 5))
        vmin = 100

        for x in [np.ones((2, 2, 2)), 1j, 'txt', (5,), None]:
            with self.assertRaises(TypeError):
                checkflat(flatmap, vmin, x, tie)
            pass
        pass


    def test_invalid_tie(self):
        """Catches invalid type input"""
        flatmap = np.zeros((5, 5))
        vmin = 0
        vmax = 100

        for x in [np.ones((2, 2, 2)), np.ones((2,)), 'txt', None, 1.0]:
            with self.assertRaises(TypeError):
                checkflat(flatmap, vmin, vmax, x)
            pass
        pass





if __name__ == '__main__':
    unittest.main()
