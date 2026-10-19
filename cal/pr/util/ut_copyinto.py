# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Unit tests for function to copy a subarray into a new array
"""

import unittest

import numpy as np

from cal.pr.util.copyinto import copyinto

class TestCopyinto(unittest.TestCase):
    """
    x Test success
    x Bad input type
    x output shape (square, nout x nout) for different input combos
    x Works for 1xN, Nx1 inputs (success, maybe exact?)
    x Works for nout >> insertarr, insertarr >> nout, nout == insertarr
    x insertarr rect
    x Catch xy out of range (do we? technically should work but not be useful)
    x Text exact (all nout sized for even and odd nout, a specific off-center
    pixel)
    x Verify xy col/row is right
    """

    def test_success(self):
        """Verify function completes with no errors on good data"""
        copyinto(np.eye(3), [1, 1], 5)
        pass


    def test_invalid_insertarr(self):
        """Verify bad inputs caught"""
        xy = [1, 1]
        nout = 5

        for perr in [0, 1, (5,), 'txt', -6.1, 1j,
                     np.ones((3,)), np.ones((3, 3, 3))]:
            with self.assertRaises(TypeError):
                copyinto(insertarr=perr,
                         xy=xy,
                         nout=nout)
            pass
        pass


    def test_invalid_xy(self):
        """Verify bad inputs caught"""
        insertarr = np.ones((3, 3))
        nout = 5

        for perr in [0, 1, 'txt', -6.1, 1j, np.ones((3,)), np.ones((3, 3, 3)),
                     (5,), (1, 1, 1),
                     (1j, 1), (6.0, 1)]:
            with self.assertRaises(TypeError):
                copyinto(insertarr=insertarr,
                         xy=perr,
                         nout=nout)
            pass
        pass


    def test_out_of_range_xy(self):
        """Verify out-of-range inputs caught"""
        insertarr = np.ones((3, 3))
        nout = 5

        for perr in [(-1, 1), (1, -1), (3, 1), (1, 3)]:
            with self.assertRaises(TypeError):
                copyinto(insertarr=insertarr,
                         xy=perr,
                         nout=nout)
            pass
        pass


    def test_invalid_nout(self):
        """Verify bad inputs caught"""
        insertarr = np.ones((3, 3))
        xy = [1, 1]

        for perr in [0, (5,), 'txt', -6.1, 1j, 5.0,
                     np.ones((3,)), np.ones((3, 3, 3))]:
            with self.assertRaises(TypeError):
                copyinto(insertarr=insertarr,
                         xy=xy,
                         nout=perr)
            pass
        pass

    def test_xy_colrow(self):
        """
        Verify that the xy parameter is input as (cols, rows) by putting in
        parameters that would otherwise fail (out of range)
        """
        # row vector
        insertarr = np.ones((1, 5))
        xy = [2, 0]
        nout = 5

        target = np.zeros((5, 5))
        target[2, :] = 1

        out = copyinto(insertarr=insertarr, xy=xy, nout=nout)
        self.assertTrue((out == target).all())

        # col vector
        insertarr = np.ones((5, 1))
        xy = [0, 2]
        nout = 5

        target = np.zeros((5, 5))
        target[:, 2] = 1

        out = copyinto(insertarr=insertarr, xy=xy, nout=nout)
        self.assertTrue((out == target).all())


        pass


    def test_output_nout_size(self):
        """Verify output is always nout x nout, regardless of input"""

        nout = 5

        iaxyvec = [(np.ones((3, 3)), [1, 1]),
                   (np.ones((1, 7)), [0, 0]),
                   (np.ones((7, 1)), [0, 0]),
                   (np.ones((8, 7)), [6, 6]),
                   (np.ones((1, 1)), [0, 0]),
                   (np.ones((25, 26)), [23, 21]),
                   (np.ones((5, 5)), [4, 0]),
                   ]

        for insertarr, xy in iaxyvec:
            out = copyinto(insertarr=insertarr, xy=xy, nout=nout)
            self.assertTrue(out.shape == (nout, nout))
            pass
        pass


    def test_self_recovery(self):
        """
        Verify inserting a nout x nout array with a co-aligned center returns
        the original array, for even and odd cases
        """

        # odd
        insertarr = np.eye(5)
        xy = [2, 2]
        nout = 5

        out = copyinto(insertarr=insertarr, xy=xy, nout=nout)
        self.assertTrue((out == insertarr).all())

        # even
        insertarr = np.eye(4)
        xy = [2, 2]
        nout = 4

        out = copyinto(insertarr=insertarr, xy=xy, nout=nout)
        self.assertTrue((out == insertarr).all())

        pass


    def test_off_center(self):
        """
        Verify we get the output we expect when xy is not aligned to
        a center pixel
        """
        insertarr = np.array([[0, 1, 2, 9],
                             [3, 4, 5, 9],
                             [6, 7, 8, 9]])
        xy = [2, 0] # col2 row0 -> 2

        # Case 1: smaller
        nout = 3
        target = np.array([[0, 0, 0],
                           [1, 2, 9],
                           [4, 5, 9]])
        out = copyinto(insertarr=insertarr, xy=xy, nout=nout)
        self.assertTrue((out == target).all())

        # TODO: should the output be centered about the xy point?  It's not
        # right now.  (This seems to be a literal edge case, because cutout
        # region is not nout x nout.  Should be able to handle this though.)

        # Case 2: larger
        nout = 5
        target = np.array([[0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 1, 2, 9, 0],
                           [3, 4, 5, 9, 0],
                           [6, 7, 8, 9, 0]])
        out = copyinto(insertarr=insertarr, xy=xy, nout=nout)
        self.assertTrue((out == target).all())


    def test_xy_in_center(self):
        """
        Expected output always is that the pixel at xy is in the center of the
        output array.  By convention center is defined as nout//2, even or odd.
        """

        # Odd nout, odd/even insertarr smaller than nout
        nrow = 3
        ncol = 4

        xylist = [[0, 0],
                  [0, nrow-1],
                  [ncol-1, 0],
                  [ncol-1, nrow-1],
                  [1, 1],
                  ]
        nout = 9

        center = nout//2
        xyval = 1.0

        for xy in xylist:
            insertarr = np.zeros((nrow, ncol))
            insertarr[xy[1], xy[0]] = xyval
            out = copyinto(insertarr=insertarr, xy=xy, nout=nout)
            self.assertTrue(out[center, center] == xyval)
            pass

        # Odd nout, odd/even insertarr larger than nout
        nrow = 8
        ncol = 9

        xylist = [[0, 0],
                  [0, nrow-1],
                  [ncol-1, 0],
                  [ncol-1, nrow-1],
                  [1, 1],
                  ]
        nout = 5

        center = nout//2
        xyval = 1.0

        for xy in xylist:
            insertarr = np.zeros((nrow, ncol))
            insertarr[xy[1], xy[0]] = xyval
            out = copyinto(insertarr=insertarr, xy=xy, nout=nout)
            self.assertTrue(out[center, center] == xyval)
            pass

        # Even nout, odd/even insertarr smaller than nout
        nrow = 4
        ncol = 3

        xylist = [[0, 0],
                  [0, nrow-1],
                  [ncol-1, 0],
                  [ncol-1, nrow-1],
                  [1, 1],
                  ]
        nout = 10

        center = nout//2
        xyval = 1.0

        for xy in xylist:
            insertarr = np.zeros((nrow, ncol))
            insertarr[xy[1], xy[0]] = xyval
            out = copyinto(insertarr=insertarr, xy=xy, nout=nout)
            self.assertTrue(out[center, center] == xyval)
            pass

        # Even nout, odd/even insertarr larger than nout
        nrow = 9
        ncol = 8

        xylist = [[0, 0],
                  [0, nrow-1],
                  [ncol-1, 0],
                  [ncol-1, nrow-1],
                  [1, 1],
                  ]
        nout = 4

        center = nout//2
        xyval = 1.0

        for xy in xylist:
            insertarr = np.zeros((nrow, ncol))
            insertarr[xy[1], xy[0]] = xyval
            out = copyinto(insertarr=insertarr, xy=xy, nout=nout)
            self.assertTrue(out[center, center] == xyval)
            pass

        pass


if __name__ == '__main__':
    unittest.main()
