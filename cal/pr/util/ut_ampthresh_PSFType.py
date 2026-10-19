# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Test class for the function PSFType

"""

import unittest

import numpy as np

from cal.pr.util.ampthresh_PSFType import PSFType

def _isequalto(a, b, tol):
    # works for lists of similar arrays, arrays, and scalars
    return np.max(np.abs(np.array(a) - np.array(b))) < tol

class TestPSFType(unittest.TestCase):
    """
    test inputs:
    test success
    """

    def setUp(self):
        """
        Predefine common variable inputs available for all tests
        """
        # create an image
        nx = 256
        w0 = 10 # pixels
        x = np.arange(nx) - nx//2
        X, Y = np.meshgrid(x, x)
        R = np.hypot(X, Y)
        self.image = np.exp(-(R/w0)**2)

        # lists of error inputs
        self.not_pos_scalar_int = [0, -1.2, 1.2, 1j, 'abc', np.ones((5,))]
        self.not_real_pos_scalar = [0, -1.2, 1j, 'abc', np.ones((5,))]

    def test_invalid_Image(self):
        """ invalid input image raises error """
        for perr in [0, 3.4, np.ones((5,)), 'abc']:
            with self.assertRaises(TypeError):
                PSFType(perr)

    def test_invalid_Nbins(self):
        """ invalid optional argument Nbins raises error """
        for perr in self.not_pos_scalar_int:
            with self.assertRaises(TypeError):
                PSFType(self.image, Nbins=perr)

    def test_invalid_Darkbins(self):
        """ invalid optional argument Darkbins raises error """
        for perr in self.not_pos_scalar_int:
            with self.assertRaises(TypeError):
                PSFType(self.image, Darkbins=perr)

    def test_invalid_Sigsig(self):
        """ invalid optional argument Sigsig raises error """
        for perr in self.not_real_pos_scalar:
            with self.assertRaises(TypeError):
                PSFType(self.image, Sigsig=perr)

    def test_invalid_NOpenSize(self):
        """ invalid optional argument NOpenSize raises error """
        for perr in self.not_pos_scalar_int:
            with self.assertRaises(TypeError):
                PSFType(self.image, NOpenSize=perr)

    def test_success(self):
        """ example that works and check answer """
        bMask, thresh = PSFType(self.image)

        # check thresh against saved value:
        self.assertTrue(_isequalto(thresh, 0.004156270184704752, 1e-12))

        # apply threshold and test mask is correct
        bMaskCheck = np.abs(self.image) >= thresh
        self.assertTrue(np.array_equal(bMask, bMaskCheck))

if __name__ == '__main__':
    unittest.main()
