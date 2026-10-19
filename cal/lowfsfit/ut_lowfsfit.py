# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Test suite for lowfsfit()"""
import unittest
import os
import numpy as np
from astropy.io import fits

import cal.lowfsfit.lowfsfit as lowfsfit

LIST_TEST_IMAGES_FN = [
    'Im_lowfs_reference.fits',
    'shift_dx_0_dy_0p3_img.fits',
    'shift_dx_0_dy_10_img.fits',
    'shift_dx_15_dy_0_img.fits',
    'shift_dx_15_dy_10_img.fits',
    'shift_dx_1_dy_5_img.fits',
    'shift_dx_1p2_dy_2p7_img.fits',
    'shift_dx_3p4_dy_0_img.fits',
]

class TestValidParameters(unittest.TestCase):
    """Test suite for valid function inputs."""

    def setUp(self):
        """parameters for valid function call"""
        localpath = os.path.dirname(os.path.abspath(__file__))

        self.list_moving_image = [
            fits.getdata(os.path.join(localpath, 'testdata', fn))
            for fn in LIST_TEST_IMAGES_FN
        ]

        self.reference_image_fn = os.path.join(
            localpath, 'testdata', 'Im_lowfs_reference.fits'
        )

        self.config = os.path.join(localpath, 'config_files', 'lowfsfit_parms.yaml')

    def test_reference_image_inputs(self):
        """test invalid reference image inputs"""
        for bad_fn in [1, [32.3, -2], {}, [], None]:
            with self.assertRaises(TypeError):
                lowfsfit.locate_img(
                    self.list_moving_image,
                    bad_fn,
                    self.config,
                )

    def test_moving_img_inputs(self):
        """test cases with problematic list_moving_image arguments"""

        # for convenience
        imgtmp = self.list_moving_image[0]

        # not a list
        for bad_list in [1, {'a':imgtmp}, imgtmp, 'abc']:
            with self.assertRaises(TypeError):
                lowfsfit.locate_img(bad_list, self.reference_image_fn, self.config)

        # list contains not-an-image
        for bad_img in [1, 'abc', np.zeros((10,)), np.zeros(imgtmp.shape + (3,))]:
            with self.assertRaises(TypeError):
                lowfsfit.locate_img([bad_img,], self.reference_image_fn, self.config)

        # list contains an image inconsistent size
        bad_img = self.list_moving_image[0][5:,5:] # cropped to different size
        with self.assertRaises(TypeError):
            lowfsfit.locate_img(self.list_moving_image + [bad_img,],
                                self.reference_image_fn, self.config)

    def test_config(self):
        """test cases with problematic config"""
        for bad_config in ['ab',]:
            with self.assertRaises(Exception): # loadyaml uses custom_exception
                lowfsfit.locate_img(
                    self.list_moving_image, self.reference_image_fn, bad_config
                )

    def test_correct_results(self):
        """test cases with various inputs, checking correct results"""

        # run prefabbed valid test cases
        list_shift = lowfsfit.locate_img(
            self.list_moving_image, self.reference_image_fn, self.config
        )

        # estimation error, shift is [row, col] == [y, x]
        list_truth = -1*np.array([
            [0.0, 0.0],
            [0.0, 0.3],
            [0.0, 10.0],
            [15.0, 0.0],
            [15.0, 10.0],
            [1.0, 5.0],
            [1.2, 2.7],
            [3.4, 0.0],
        ])

        # estimation error
        list_error = [sh - tr for sh, tr in zip(list_shift, list_truth)]

        # check that all cases pass
        self.assertTrue(np.all(np.abs(np.array(list_error)) < 0.2))


# check if this is run in main or not
if __name__ == '__main__':
    unittest.main()
