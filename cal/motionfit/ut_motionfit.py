# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Test suite for MOTIONFIT module."""
import unittest
import os
import pathlib
from numpy import isclose
import numpy as np
from astropy.io import fits

from cal.util.insertinto import insertinto
import cal.motionfit.motionfit as mf
import cal.pupilfit.pupilfit as pf
import cal.lowfsfit.lowfsfit as lf

PUPILFIT_PATH = pathlib.Path(pf.__file__).resolve().parent
LOWFSFIT_PATH = pathlib.Path(lf.__file__).resolve().parent
LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
TESTDATA_PATH = os.path.join(LOCAL_PATH, 'testdata')


class TestInputFailures(unittest.TestCase):
    """Test suite for valid function inputs."""

    def test_inputs_calc_trans_matrix(self):
        """Test the inputs of calc_trans_matrix."""
        radOverDeg = np.pi/180
        xyOffsetList = [[0, 0],
                        [1, 0],
                        [np.cos(120*radOverDeg), np.sin(120*radOverDeg)],
                        [np.cos(-120*radOverDeg), np.sin(-120*radOverDeg)]
                        ]
        commandList = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]

        # Check that standard inputs do not raise anything first
        _, _, _, _ = mf.calc_trans_matrix(xyOffsetList, commandList)

        # Not a list
        for xyListBad in ((1, 2), np.ones((3, 2)), 'asdf', 1):
            with self.assertRaises(TypeError):
                mf.calc_trans_matrix(xyListBad, commandList)
        # List entries aren't 1-D arrays
        for xyListBad in ([1, ], [np.ones((3, 3)), ], ['string', ]):
            with self.assertRaises(ValueError):
                mf.calc_trans_matrix(xyListBad, commandList)
        # xy Vectors must all be length 2
        for xyListBad in ([[1, 2], [1, 2, 3]], ):
            with self.assertRaises(ValueError):
                mf.calc_trans_matrix(xyListBad, commandList)
        # xy values must be real
        for xyListBad in ([[1, 2], [1, 1j]],):
            with self.assertRaises(TypeError):
                mf.calc_trans_matrix(xyListBad, commandList)

        # Not a list
        for commandListBad in ((1, 2), np.ones((3, 2)), 'asdf', 1):
            with self.assertRaises(TypeError):
                mf.calc_trans_matrix(xyOffsetList, commandListBad)
        # List entries aren't 1-D arrays
        for commandListBad in ([1, ], [np.ones((3, 3)), ], ['string', ]):
            with self.assertRaises(ValueError):
                mf.calc_trans_matrix(xyOffsetList, commandListBad)
        # Vectors within the list aren't the same length
        for commandListBad in ([[1, 2], [1, 2, 3]], ):
            with self.assertRaises(ValueError):
                mf.calc_trans_matrix(xyOffsetList, commandListBad)

        # xyOffsetList and commandList arguments don't have same length
        with self.assertRaises(ValueError):
            mf.calc_trans_matrix(xyOffsetList, commandList[0:3])

    def test_inputs_CtransAbsFwd_matmult(self):
        """Test the inputs of the overloaded matmult for CtransAbsFwd."""
        radOverDeg = np.pi/180
        xyOffsetList = [[0, 0],
                        [1, 0],
                        [np.cos(120*radOverDeg), np.sin(120*radOverDeg)],
                        [np.cos(-120*radOverDeg), np.sin(-120*radOverDeg)]
                        ]
        commandList = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]

        # first create a valid instance of the class
        _, _, tAbsFwd, _ = mf.calc_trans_matrix(xyOffsetList, commandList)
        tAbsFwd = mf.CtransAbsFwd(tAbsFwd)

        if not isinstance(tAbsFwd, mf.CtransAbsFwd):
            raise TypeError('return value is not type CtransAbsFwd')

        # Check that standard inputs do not raise anything first
        _ = tAbsFwd @ np.array(xyOffsetList)

        # Not a numpy array
        for xyNotArray in [xyOffsetList, ['asdf', 1], 'asdf', 1,]:
            with self.assertRaises(TypeError):
                _ = tAbsFwd @ xyNotArray

        # Not real values
        with self.assertRaises(TypeError):
            _ = tAbsFwd @ np.array([[1j, 0]])

        # must be 2-d array or 1-d shape (2,)
        for xyBad in (np.array([0, 1, 3]), np.zeros((3, 4, 5))):
            with self.assertRaises(TypeError):
                _ = tAbsFwd @ xyBad

        # XY are not len 2
        with self.assertRaises(TypeError):
            _ = tAbsFwd @ np.ones((3, 3))

    def test_inputs_CtransAbsRev_matmult(self):
        """Test the inputs of the overloaded matmult for CtransAbsRev."""
        radOverDeg = np.pi/180
        xyOffsetList = [[0, 0],
                        [1, 0],
                        [np.cos(120*radOverDeg), np.sin(120*radOverDeg)],
                        [np.cos(-120*radOverDeg), np.sin(-120*radOverDeg)]
                        ]
        commandList = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]

        # first create a valid instance of the class
        _, _, _, tAbsRev = mf.calc_trans_matrix(xyOffsetList, commandList)
        tAbsRev = mf.CtransAbsRev(tAbsRev)

        if not isinstance(tAbsRev, mf.CtransAbsRev):
            raise TypeError('return value is not type CtransAbsRev')

        # Check that standard inputs do not raise anything first
        _ = tAbsRev @ np.array(commandList)

        # Not a numpy array
        for cmdNotArray in [commandList, ['asdf', 1], 'asdf', 1,]:
            with self.assertRaises(TypeError):
                _ = tAbsRev @ cmdNotArray

        # Not real values
        with self.assertRaises(TypeError):
            _ = tAbsRev @ np.array([[1j, 0, 0]])

        # must be 2-d array, or 1-d not len M = 3
        for cmdBad in (np.array([0, 1, 0, 1]), np.zeros((3, 4, 5))):
            with self.assertRaises(TypeError):
                _ = tAbsRev @ cmdBad

        # commands are not len M = 3
        with self.assertRaises(TypeError):
            _ = tAbsRev @ np.ones((3, 2))

    def test_inputs_calc_unmasked_pupil_image_offsets(self):
        """
        Test the inputs of calc_unmasked_pupil_image_offsets.

        Filename input tests are already covered by the unit
        tests for util.loadyaml.loadyaml().
        """
        fn_pupil_fit_params = os.path.join(
            PUPILFIT_PATH, 'testdata', 'ut_fit_unmasked_pupil.yaml')
        fnImage0 = os.path.join(PUPILFIT_PATH, 'cgidata',
                                'pupil_template_D300.00pixels.fits')
        image0 = fits.getdata(fnImage0)
        image0 = insertinto(image0, (500, 501))
        imageList = [image0, ]

        # Check that standard inputs do not raise anything first
        _ = mf.calc_unmasked_pupil_image_offsets(imageList,
                                                 fn_pupil_fit_params)

        # Not a list
        for imageListBad in ((1, 2), np.ones((3, 2)), 'asdf', 1):
            with self.assertRaises(TypeError):
                mf.calc_unmasked_pupil_image_offsets(imageListBad,
                                                     fn_pupil_fit_params)
        # List entries aren't 2-D arrays
        for imageListBad in ([1, ], [np.ones(3), ], ['string', ]):
            with self.assertRaises(TypeError):
                mf.calc_unmasked_pupil_image_offsets(imageListBad,
                                                     fn_pupil_fit_params)
        # 2-D arrays within the list aren't the same shape
        imageListBad = [np.eye(3), np.eye(4)]
        with self.assertRaises(ValueError):
            mf.calc_unmasked_pupil_image_offsets(imageListBad,
                                                 fn_pupil_fit_params)

    def test_inputs_calc_unmasked_focal_image_offsets(self):
        """
        Test the inputs of calc_unmasked_focal_image_offsets.

        fnImageRef does not have tests because it is used by
        astropy.io.fits.getdata(), which has its own input checks.

        fnPSFFIT does not have tests because it is used by
        util.loadyaml.loadyaml(), which has its own input checks.
        """
        fnPSFFIT = os.path.join(TESTDATA_PATH, 'psffit_parms.yaml')
        fnImageRef = os.path.join(TESTDATA_PATH, 'ref_unmasked_psf_band1.fits')
        fnImage0 = os.path.join(TESTDATA_PATH, 'ref_unmasked_psf_band2.fits')
        image0 = fits.getdata(fnImage0)

        xyListIn = [[-10, 30], [14, -5], [1, 0]]
        imageA = 1e6*np.roll(image0, xyListIn[0], axis=(1, 0))
        imageB = 1e-6*np.roll(image0, xyListIn[1], axis=(1, 0))
        imageC = 1*np.roll(image0, xyListIn[2], axis=(1, 0))
        imageList = [imageA, imageB, imageC]

        # Check that standard inputs do not raise anything first
        _ = mf.calc_unmasked_focal_image_offsets(imageList,
                                                 fnImageRef,
                                                 fnPSFFIT)

        # Not a list
        for imageListBad in ((1, 2), np.ones((3, 2)), 'asdf', 1):
            with self.assertRaises(TypeError):
                mf.calc_unmasked_focal_image_offsets(imageListBad,
                                                     fnImageRef,
                                                     fnPSFFIT)
        # List entries aren't 2-D arrays
        for imageListBad in ([1, ], [np.ones(3), ], ['string', ]):
            with self.assertRaises(TypeError):
                mf.calc_unmasked_focal_image_offsets(imageListBad,
                                                     fnImageRef,
                                                     fnPSFFIT)
        # 2-D arrays within the list aren't the same shape
        imageListBad = [np.eye(3), np.eye(4)]
        with self.assertRaises(ValueError):
            mf.calc_unmasked_focal_image_offsets(imageListBad,
                                                 fnImageRef,
                                                 fnPSFFIT)

    def test_inputs_calc_circular_mask_offsets(self):
        """
        Test the inputs of calc_circular_mask_offsets().

        fnTuning does not have tests because it is used by
        util.loadyaml.loadyaml(), which has its own input checks.
        """
        fnTuning = os.path.join(LOCAL_PATH, 'testdata',
                                'ut_fit_hlc_band1_field_stop_offsets.yaml')

        image = fits.getdata(os.path.join(
            LOCAL_PATH, 'testdata',
            'fsam_motion_calib_offsets_x11.50lamD_y11.50lamD_pattern1.fits'
        ))
        imageList = [image, image]

        # Check that standard inputs do not raise anything first
        _ = mf.calc_circular_mask_offsets(imageList, fnTuning)

        # Not a list
        for imageListBad in ((1, 2), np.ones((3, 2)), 'asdf', 1):
            with self.assertRaises(TypeError):
                mf.calc_circular_mask_offsets(imageListBad, fnTuning)

        # List entries aren't 2-D arrays
        for imageListBad in ([1, ], [np.ones(3), ], ['string', ]):
            with self.assertRaises(TypeError):
                mf.calc_circular_mask_offsets(imageListBad, fnTuning)

        # 2-D arrays within the list aren't the same shape
        imageListBad = [np.eye(3), np.eye(4)]
        with self.assertRaises(ValueError):
            mf.calc_circular_mask_offsets(imageListBad, fnTuning)

    def test_inputs_convert_vector_list_into_two_dim_array(self):
        """Test the inputs of convert_vector_list_into_two_dim_array()."""
        vectorList = [[1, 0, -1], [0, 0, 1]]

        # Check that standard inputs do not raise anything first
        _ = mf.convert_vector_list_into_two_dim_array(vectorList)

        # Not a list
        for vectorListBad in ((1, 2), np.ones((3, 2)), 'asdf', 1):
            with self.assertRaises(TypeError):
                mf.convert_vector_list_into_two_dim_array(vectorListBad)
        # List entries aren't 1-D arrays
        for vectorListBad in ([1, ], [np.ones((3, 3)), ], ['string', ]):
            with self.assertRaises(TypeError):
                mf.convert_vector_list_into_two_dim_array(vectorListBad)
        # Vectors within the list aren't the same length
        for vectorListBad in ([[1, 2], [1, 2, 3]], ):
            with self.assertRaises(ValueError):
                mf.convert_vector_list_into_two_dim_array(vectorListBad)


# Wrapper inputs, crib from tests above

    def test_CTMfromPSF_inputs(self):
        """
        Tests for inputs to calc_trans_matrix_from_psf_images
        """
        fnPSFFIT = os.path.join(TESTDATA_PATH, 'psffit_parms.yaml')
        fnImageRef = os.path.join(TESTDATA_PATH, 'ref_unmasked_psf_band1.fits')
        fnImage0 = os.path.join(TESTDATA_PATH, 'ref_unmasked_psf_band2.fits')
        image0 = fits.getdata(fnImage0)

        xyListIn = [[-10, 30], [14, -5], [1, 0]]
        imageA = 1e6*np.roll(image0, xyListIn[0], axis=(1, 0))
        imageB = 1e-6*np.roll(image0, xyListIn[1], axis=(1, 0))
        imageC = 1*np.roll(image0, xyListIn[2], axis=(1, 0))
        imageList = [imageA, imageB, imageC]
        commandList = [[0, 1], [1, 0], [1, 1]]

        # Check that standard inputs do not raise anything first
        _ = mf.calc_trans_matrix_from_psf_images(imageList,
                                                 commandList,
                                                 fnImageRef,
                                                 fnPSFFIT)

        # Not a list
        for imageListBad in ((1, 2), np.ones((3, 2)), 'asdf', 1):
            with self.assertRaises(TypeError):
                mf.calc_trans_matrix_from_psf_images(imageListBad,
                                                     commandList,
                                                     fnImageRef,
                                                     fnPSFFIT)
        # List entries aren't 2-D arrays
        for imageListBad in ([1, ], [np.ones(3), ], ['string', ]):
            with self.assertRaises(TypeError):
                mf.calc_trans_matrix_from_psf_images(imageListBad,
                                                     commandList,
                                                     fnImageRef,
                                                     fnPSFFIT)
        # 2-D arrays within the list aren't the same shape
        imageListBad = [np.eye(3), np.eye(4)]
        with self.assertRaises(ValueError):
            mf.calc_trans_matrix_from_psf_images(imageListBad,
                                                 commandList,
                                                 fnImageRef,
                                                 fnPSFFIT)
        # Not a list
        for commandListBad in ((1, 2), np.ones((3, 2)), 'asdf', 1):
            with self.assertRaises(TypeError):
                mf.calc_trans_matrix_from_psf_images(imageList,
                                                     commandListBad,
                                                     fnImageRef,
                                                     fnPSFFIT)
        # List entries aren't 1-D arrays
        for commandListBad in ([1, ], [np.ones((3, 3)), ], ['string', ]):
            with self.assertRaises(TypeError):
                mf.calc_trans_matrix_from_psf_images(imageList,
                                                     commandListBad,
                                                     fnImageRef,
                                                     fnPSFFIT)
        # Vectors within the list aren't the same length
        for commandListBad in ([[1, 2], [1, 2, 3]], ):
            with self.assertRaises(ValueError):
                mf.calc_trans_matrix_from_psf_images(imageList,
                                                     commandListBad,
                                                     fnImageRef,
                                                     fnPSFFIT)
        # List arguments don't have same length
        with self.assertRaises(ValueError):
            mf.calc_trans_matrix_from_psf_images(imageList,
                                                 commandListBad,
                                                 fnImageRef,
                                                 fnPSFFIT)

    def test_CTMfromCirc_inputs(self):
        """
        Tests for calc_trans_matrix_from_circ_mask_images
        """
        fnTuning = os.path.join(LOCAL_PATH, 'testdata',
                                'ut_fit_hlc_band1_field_stop_offsets.yaml')

        image = fits.getdata(os.path.join(
            LOCAL_PATH, 'testdata',
            'fsam_motion_calib_offsets_x11.50lamD_y11.50lamD_pattern1.fits'
        ))
        imageList = [image, np.roll(image, 5, axis=0)]
        commandList = [[0, 1], [1, 0]]

        # Check that standard inputs do not raise anything first
        _ = mf.calc_trans_matrix_from_circ_mask_images(imageList,
                                                       commandList,
                                                       fnTuning)

        # Not a list
        for imageListBad in ((1, 2), np.ones((3, 2)), 'asdf', 1):
            with self.assertRaises(TypeError):
                mf.calc_trans_matrix_from_circ_mask_images(imageListBad,
                                                           commandList,
                                                           fnTuning)
        # List entries aren't 2-D arrays
        for imageListBad in ([1, ], [np.ones(3), ], ['string', ]):
            with self.assertRaises(TypeError):
                mf.calc_trans_matrix_from_circ_mask_images(imageListBad,
                                                           commandList,
                                                           fnTuning)
        # 2-D arrays within the list aren't the same shape
        imageListBad = [np.eye(3), np.eye(4)]
        with self.assertRaises(ValueError):
            mf.calc_trans_matrix_from_circ_mask_images(imageListBad,
                                                       commandList,
                                                       fnTuning)
        # Not a list
        for commandListBad in ((1, 2), np.ones((3, 2)), 'asdf', 1):
            with self.assertRaises(TypeError):
                mf.calc_trans_matrix_from_circ_mask_images(imageList,
                                                           commandListBad,
                                                           fnTuning)
        # List entries aren't 1-D arrays
        for commandListBad in ([1, ], [np.ones((3, 3)), ], ['string', ]):
            with self.assertRaises(TypeError):
                mf.calc_trans_matrix_from_circ_mask_images(imageList,
                                                           commandListBad,
                                                           fnTuning)
        # Vectors within the list aren't the same length
        for commandListBad in ([[1, 2], [1, 2, 3]], ):
            with self.assertRaises(ValueError):
                mf.calc_trans_matrix_from_circ_mask_images(imageList,
                                                           commandListBad,
                                                           fnTuning)
        # List arguments don't have same length
        with self.assertRaises(ValueError):
            mf.calc_trans_matrix_from_circ_mask_images(imageList,
                                                       commandListBad,
                                                       fnTuning)

    def test_CTMfromPupil_inputs(self):
        """
        Tests for calc_trans_matrix_from_pupil_images
        """
        fn_pupil_fit_params = os.path.join(
            PUPILFIT_PATH, 'testdata', 'ut_fit_unmasked_pupil.yaml')
        fnImage0 = os.path.join(PUPILFIT_PATH, 'cgidata',
                                'pupil_template_D300.00pixels.fits')
        image0 = fits.getdata(fnImage0)
        image0 = insertinto(image0, (500, 501))
        imageList = [image0, np.roll(image0, 5, axis=0)]
        commandList = [[1, 0], [0, 1]]

        # Check that standard inputs do not raise anything first
        _ = mf.calc_trans_matrix_from_pupil_images(imageList,
                                                   commandList,
                                                   fn_pupil_fit_params)

        # Not a list
        for imageListBad in ((1, 2), np.ones((3, 2)), 'asdf', 1):
            with self.assertRaises(TypeError):
                mf.calc_trans_matrix_from_pupil_images(imageListBad,
                                                       commandList,
                                                       fn_pupil_fit_params)
        # List entries aren't 2-D arrays
        for imageListBad in ([1, ], [np.ones(3), ], ['string', ]):
            with self.assertRaises(TypeError):
                mf.calc_trans_matrix_from_pupil_images(imageListBad,
                                                       commandList,
                                                       fn_pupil_fit_params)
        # 2-D arrays within the list aren't the same shape
        imageListBad = [np.eye(3), np.eye(4)]
        with self.assertRaises(ValueError):
            mf.calc_trans_matrix_from_pupil_images(imageListBad,
                                                   commandList,
                                                   fn_pupil_fit_params)
        # Not a list
        for commandListBad in ((1, 2), np.ones((3, 2)), 'asdf', 1):
            with self.assertRaises(TypeError):
                mf.calc_trans_matrix_from_pupil_images(imageList,
                                                       commandListBad,
                                                       fn_pupil_fit_params)
        # List entries aren't 1-D arrays
        for commandListBad in ([1, ], [np.ones((3, 3)), ], ['string', ]):
            with self.assertRaises(TypeError):
                mf.calc_trans_matrix_from_pupil_images(imageList,
                                                       commandListBad,
                                                       fn_pupil_fit_params)
        # Vectors within the list aren't the same length
        for commandListBad in ([[1, 2], [1, 2, 3]], ):
            with self.assertRaises(ValueError):
                mf.calc_trans_matrix_from_pupil_images(imageList,
                                                       commandListBad,
                                                       fn_pupil_fit_params)
        # List arguments don't have same length
        with self.assertRaises(ValueError):
            mf.calc_trans_matrix_from_pupil_images(imageList,
                                                   commandListBad,
                                                   fn_pupil_fit_params)

    def test_CTMfromLowfs_inputs(self):
        """
        Tests for calc_trans_matrix_from_lowfs_images
        """

        # location of reference image
        fn_im_lowfs_ref = os.path.join(
            LOWFSFIT_PATH, 'testdata', 'Im_lowfs_reference.fits')

        # create list of translated images
        image0 = fits.getdata(fn_im_lowfs_ref)
        imageList = [image0, np.roll(image0, 5, axis=0), np.roll(image0, 2, axis=1)]

        # make up command list to go with shifted images
        commandList = [[0, 0], [0, 1], [1, 0]]

        # config file
        config_fn = os.path.join(LOWFSFIT_PATH, 'config_files', 'lowfsfit_parms.yaml')

        # fixed value for bias+dark
        image_bias_dark = 0.7

        # add bias+dark to "measured" images
        for img in imageList:
            img += image_bias_dark

        # Check that standard inputs do not raise anything first
        _ = mf.calc_trans_matrix_from_lowfs_images(imageList,
                                                   commandList,
                                                   fn_im_lowfs_ref,
                                                   config_fn,
                                                   image_bias_dark=image_bias_dark,
                                                   )

        # Check that using default keyword arguments work ok
        _ = mf.calc_trans_matrix_from_lowfs_images(imageList, commandList, fn_im_lowfs_ref)

        # Not a list
        for imageListBad in ((1, 2), np.ones((3, 2)), 'asdf', 1):
            with self.assertRaises(TypeError):
                mf.calc_trans_matrix_from_lowfs_images(imageListBad,
                                                       commandList,
                                                       fn_im_lowfs_ref,
                                                       config_fn,
                                                       image_bias_dark=image_bias_dark)

        # List entries aren't 2-D arrays
        for imageListBad in ([1, ], [np.ones(3), ], ['string', ]):
            with self.assertRaises(TypeError):
                mf.calc_trans_matrix_from_lowfs_images(imageListBad,
                                                       commandList,
                                                       fn_im_lowfs_ref,
                                                       config_fn,
                                                       image_bias_dark=image_bias_dark)

        # 2-D arrays within the list aren't the same shape
        imageListBad = [np.eye(3), np.eye(4)]
        with self.assertRaises(ValueError):
            mf.calc_trans_matrix_from_lowfs_images(imageListBad,
                                                   commandList,
                                                   fn_im_lowfs_ref,
                                                   config_fn,
                                                   image_bias_dark=image_bias_dark)

        # Not a list
        for commandListBad in ((1, 2), np.ones((3, 2)), 'asdf', 1):
            with self.assertRaises(TypeError):
                mf.calc_trans_matrix_from_lowfs_images(imageList,
                                                       commandListBad,
                                                       fn_im_lowfs_ref,
                                                       config_fn,
                                                       image_bias_dark=image_bias_dark)

        # List entries aren't 1-D arrays
        for commandListBad in ([1, ], [np.ones((3, 3)), ], ['string', ]):
            with self.assertRaises(TypeError):
                mf.calc_trans_matrix_from_lowfs_images(imageList,
                                                       commandListBad,
                                                       fn_im_lowfs_ref,
                                                       config_fn,
                                                       image_bias_dark=image_bias_dark)

        # Vectors within the list aren't the same length
        for commandListBad in ([[1, 2], [1, 2, 3]], ):
            with self.assertRaises(ValueError):
                mf.calc_trans_matrix_from_lowfs_images(imageList,
                                                       commandListBad,
                                                       fn_im_lowfs_ref,
                                                       config_fn,
                                                       image_bias_dark=image_bias_dark)

        # List arguments don't have same length
        with self.assertRaises(ValueError):
            mf.calc_trans_matrix_from_lowfs_images(imageList,
                                                   commandListBad,
                                                   fn_im_lowfs_ref,
                                                   config_fn,
                                                   image_bias_dark=image_bias_dark)


class TestMotionfitPerformance(unittest.TestCase):
    """Test performance of functions in MOTIONFIT."""

    def test_calc_trans_matrix(self):
        """Test functionality of calc_trans_matrix."""

        # make up a problem:
        # 4 command sets => 4 (x,y) EXCAM pixels
        #
        # each set commands 3 actuators (FSM)
        #
        # commandList is list of 4 command sets (3 actuators):
        commandList = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        cmdMat = np.array(commandList)
        Ncmds, _  = cmdMat.shape  # shape = (4, 3)
        # add col of 1 to cmdMat for translation
        cmdMat_t = np.hstack((cmdMat, np.ones((Ncmds, 1))))

        # use these commands to create a set of 4 [x, y] EXCAM pixels, create a tform,
        # including translation.
        tx = 150.0
        ty = 75.0
        cos30 = 1.1*0.5*np.sqrt(3) # actuators are not a perfect equilateral triangle
        sin30 = 0.5
        TMinv = np.array([[1, -sin30, -sin30, tx], [0, -cos30, cos30, ty]]) # shape = (2, 4)

        # x, y position x 4 commands
        xyMat = TMinv @ cmdMat_t.T
        xyOffsetList = xyMat.T.tolist()

        # inputs are defined, calculate the transforms
        transRelFwd, transRelRev, transAbsFwd, transAbsRev \
            = mf.calc_trans_matrix(xyOffsetList, commandList)
        transAbsFwd = mf.CtransAbsFwd(transAbsFwd)
        transAbsRev = mf.CtransAbsRev(transAbsRev)

        # test 1: transAbsRev == TMinv
        self.assertTrue(np.all(isclose(transAbsRev.tMat, TMinv)))

        # test 2: xyOffsetList = transAbsRev @ commandList
        self.assertTrue(np.all(isclose(
            transAbsRev @ np.array(commandList), np.array(xyOffsetList)
        )))

        # test 3: CommandList = transAbsFwd @ xyOffsetList
        #   commandList doesn't match exactly because the original commandList
        #   has arbitrary piston
        cmd_test = transAbsFwd @ np.array(xyOffsetList)
        for c_test, cmd in zip(cmd_test, commandList):
            # check that c_test == cmd except for constant piston
            self.assertTrue(isclose(np.max(c_test - cmd), np.min(c_test - cmd)))
            # check that result cmd_test has zero piston
            self.assertTrue(isclose(np.sum(c_test), 0.0))

        # test 4: delta XY = transRelFwd @ delta commands
        #         delta cmds = transRelRev @ delta XY
        for xy, cmd in zip(xyOffsetList[1:], commandList[1:]):
            deltaxy = np.array(xy) - np.array(xyOffsetList[0])
            deltacmd = np.array(cmd) - np.array(commandList[0])
            deltacmdrel = deltacmd - deltacmd.mean() # remove piston
            self.assertTrue(np.all(isclose(transRelFwd @ deltaxy, deltacmdrel)))
            self.assertTrue(np.all(isclose(transRelRev @ deltacmd, deltaxy)))

        # test 5:
        #    transAbsFwd @ (xy + dxy) == transAbsFwd @ xy + transRelFwd @ dxy
        #    transMatReverse @ (cmd + dcmd) ==
        #         transMatReverse @ cmd + transRelRev @ dcmd
        # choose any arbitrary dxy
        dxy = np.array([100.3, -110.8])
        for xy in xyOffsetList:
            xy = np.array(xy)
            self.assertTrue(np.all(isclose(
                transAbsFwd @ (xy + dxy),
                transAbsFwd @ xy + transRelFwd @ dxy
            )))

        # choose any arbitrary dcmd
        dcmd = np.array([0.5, -0.5, 0.1])
        for cmd in commandList:
            cmd = np.array(cmd)
            self.assertTrue(np.all(isclose(
                transAbsRev @ (cmd + dcmd),
                transAbsRev @ np.array(cmd) + transRelRev @ dcmd
            )))

        # test 6: show that relative transforms are indpendent of translation
        # a. add offset to each xy and show relative transforms are unchanged
        xyshift = np.array([10.0, -21.0])
        xyOffList_test = [np.array(xy) + xyshift for xy in xyOffsetList]
        transRelFwd_test, transRelRev_test, transAbsFwd_test, transAbsRev_test = \
            mf.calc_trans_matrix(xyOffList_test, commandList)
        transAbsFwd_test = mf.CtransAbsFwd(transAbsFwd_test)
        transAbsRev_test = mf.CtransAbsRev(transAbsRev_test)

        self.assertTrue(np.all(np.isclose(transRelFwd, transRelFwd_test)))
        self.assertTrue(np.all(np.isclose(transRelRev, transRelRev_test)))
        self.assertTrue(np.all(np.isclose(
            transAbsRev_test.tMat[:, -1] - transAbsRev.tMat[:, -1], xyshift
        )))

        # b. add a translation of commandList, show get same relative transforms
        commandList_test = [np.array(cmd) + np.array([1.1, -0.2, 0.5])
                            for cmd in commandList]
        transRelFwd_test, transRelRev_test, transAbsFwd_test, transAbsRev_test = \
            mf.calc_trans_matrix(xyOffsetList, commandList_test)
        transAbsFwd_test = mf.CtransAbsFwd(transAbsFwd_test)
        transAbsRev_test = mf.CtransAbsRev(transAbsRev_test)

        self.assertTrue(np.all(np.isclose(transRelFwd, transRelFwd_test)))
        self.assertTrue(np.all(np.isclose(transRelRev, transRelRev_test)))
        self.assertFalse(np.all(np.isclose(transAbsFwd.tMat, transAbsFwd_test.tMat)))
        self.assertFalse(np.all(np.isclose(transAbsRev.tMat, transAbsRev_test.tMat)))

        # c. add a translation of both xyOffset list and commandList,
        #    show get same relative transforms
        xyOffList_test = [np.array(xy) + np.array([10.0, -21.0]) for xy in xyOffsetList]
        commandList_test = [np.array(cmd) + np.array([1.1, -0.2, 0.5])
                            for cmd in commandList]
        transRelFwd_test, transRelRev_test, transAbsFwd_test, transAbsRev_test = \
            mf.calc_trans_matrix(xyOffList_test, commandList_test)
        transAbsFwd_test = mf.CtransAbsFwd(transAbsFwd_test)
        transAbsRev_test = mf.CtransAbsRev(transAbsRev_test)

        self.assertTrue(np.all(np.isclose(transRelFwd, transRelFwd_test)))
        self.assertTrue(np.all(np.isclose(transRelRev, transRelRev_test)))
        self.assertFalse(np.all(np.isclose(transAbsFwd.tMat, transAbsFwd_test.tMat)))
        self.assertFalse(np.all(np.isclose(transAbsRev.tMat, transAbsRev_test.tMat)))

        # test 7: test transforms for points other than those used to
        #  calculate the transforms.
        #  make up new set of CommandList, xyOffsetList (chosen randomly)
        xyOffsetList = [[150.8116, 0.9412], [-0.7460, 0.9143], [0.8268, -0.0292],
                        [0.2647, 0.6006]]
        cmd_test = transAbsFwd @ np.array(xyOffsetList)
        xy_test = transAbsRev @ cmd_test
        self.assertTrue(np.all(isclose(xy_test, np.array(xyOffsetList))))

    def test_calc_trans_matrix_rank_deficiency(self):
        """Test that calc_trans_matrix fails when rank is too low."""

        # test calc_similarity_matrix for 2x2
        zeroList = [[1.0, 0.0], [1.0, 0.0]]
        moveList = [[0.0, 0.0], [1.0, 1.0]]
        # bad commandList:
        with self.assertRaises(ValueError):
            mf.calc_trans_matrix(moveList, zeroList)
        # bad xyOffsetList:
        with self.assertRaises(ValueError):
            mf.calc_trans_matrix(zeroList, moveList)

        # test with three actuators
        # valid xyOffsetList
        xyOffsetList = [[150.8116, 0.9412], [-0.7460, 0.9143], [0.8268, -0.0292],
                        [0.2647, 0.6006]]
        # xyOffsetList with repeating locations
        xyOffsetList_bad = [[0, 0],
                            [1, 0],
                            [2, 0],
                            [3, 0]]
        # valid command list
        commandList = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        commandList_bad = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]

        with self.assertRaises(ValueError):
            mf.calc_trans_matrix(xyOffsetList_bad, commandList)

        with self.assertRaises(ValueError):
            mf.calc_trans_matrix(xyOffsetList, commandList_bad)

    def test_calc_unmasked_pupil_image_offsets(self):
        """Test performance of calc_unmasked_pupil_image_offsets()."""
        fn_pupil_fit_params = os.path.join(
            PUPILFIT_PATH, 'testdata', 'ut_fit_unmasked_pupil.yaml')
        fnImage0 = os.path.join(
            PUPILFIT_PATH, 'cgidata', 'pupil_template_D300.00pixels.fits')
        image0 = fits.getdata(fnImage0)
        image0 = insertinto(image0, (500, 511))

        xyListIn = [[-10, 30], [14, -5]]
        imageA = 1e3*np.roll(image0, xyListIn[0], axis=(1, 0))
        imageB = 1e-3*np.roll(image0, xyListIn[1], axis=(1, 0))
        imageList = [imageA, imageB]

        xyListOut = mf.calc_unmasked_pupil_image_offsets(imageList,
                                                         fn_pupil_fit_params)

        tol = 0.3  # ExCAM pixels
        for index, xyIn in enumerate(xyListIn):
            xyOut = xyListOut[index]
            self.assertTrue(np.abs(xyIn[0] - xyOut[0]) < tol)
            self.assertTrue(np.abs(xyIn[1] - xyOut[1]) < tol)

    def test_calc_unmasked_focal_image_offsets(self):
        """Test performance of calc_unmasked_focal_image_offsets()."""
        fnPSFFIT = os.path.join(TESTDATA_PATH, 'psffit_parms.yaml')
        fnImageRef = os.path.join(TESTDATA_PATH, 'ref_unmasked_psf_band1.fits')
        fnImage0 = os.path.join(TESTDATA_PATH, 'ref_unmasked_psf_band2.fits')
        image0 = fits.getdata(fnImage0)

        xyListIn = [[-10, 30], [14, -5], [1, 0]]
        imageA = 1e6*np.roll(image0, xyListIn[0], axis=(1, 0))
        imageB = 1e-6*np.roll(image0, xyListIn[1], axis=(1, 0))
        imageC = 1*np.roll(image0, xyListIn[2], axis=(1, 0))
        imageList = [imageA, imageB, imageC]

        xyListOut = mf.calc_unmasked_focal_image_offsets(
            imageList, fnImageRef, fnPSFFIT)

        tol = 1e-3  # ExCAM pixels
        for index, xyIn in enumerate(xyListIn):
            xyOut = xyListOut[index]
            self.assertTrue(np.abs(xyIn[0] - xyOut[0]) < tol)
            self.assertTrue(np.abs(xyIn[1] - xyOut[1]) < tol)

    def test_convert_vector_list_into_two_dim_array(self):
        """Test functionality of convert_vector_list_into_two_dim_array()."""
        vectorList = [[1, 0, -1], [0, 0, 1]]
        matExpected = np.array([[1, 0, -1], [0, 0, 1]])
        matOut = mf.convert_vector_list_into_two_dim_array(vectorList)
        self.assertTrue(np.allclose(matExpected, matOut))

    def test_fpm_motion_calibration(self):
        """
        Test performance of calc_circular_mask_offsets() with FPMs.

        Run tests on both odd- and even-sized arrays.
        """
        fnTuning = os.path.join(LOCAL_PATH, 'testdata',
                                'ut_fit_hlc_band1_fpm_offsets.yaml')
        ppl = 2.3
        sep = 11.5  # [lambda/D]
        xOffsetMaskVec = sep*np.array([1, -1, -1, 1])  # [lambda/D]
        yOffsetMaskVec = sep*np.array([1, 1, -1, -1])  # [lambda/D]

        nOffsets = len(xOffsetMaskVec)
        nPatterns = 4
        imageListOdd = []
        imageListEven = []
        nOdd = 201
        nEven = 200
        xyOffsetTrueList = []
        for iOffset in range(nOffsets):

            xOffsetLamD = xOffsetMaskVec[iOffset]  # lambda/D
            yOffsetLamD = yOffsetMaskVec[iOffset]  # lambda/D
            xyOffsetTrueList.append([xOffsetLamD*ppl, yOffsetLamD*ppl])  # pix

            # Sum together several images with different white noise patterns
            # applied to a DM. Doing so smooths out the background and helps
            # in identifying the circular mask edge.
            imageSum = 0  # initialize
            for iPattern in range(nPatterns):

                fnIn = ('fpam_motion_calib_offsets_x'
                        '%.2flamD_y%.2flamD_pattern%d.fits' %
                        (xOffsetLamD, yOffsetLamD, iPattern))

                imageSum += fits.getdata(os.path.join(
                    LOCAL_PATH, 'testdata', fnIn), overwrite=True)

            imageSumOdd = insertinto(imageSum, (nOdd, nOdd))
            imageListOdd.append(imageSumOdd)

            imageSumEven = insertinto(imageSum, (nEven, nEven))
            imageListEven.append(imageSumEven)

        xyOffsetEstListOdd = mf.calc_circular_mask_offsets(
            imageListOdd, fnTuning)
        xyOffsetEstListEven = mf.calc_circular_mask_offsets(
            imageListEven, fnTuning)

        for iOffset in range(nOffsets):
            xOffsetEstOdd = xyOffsetEstListOdd[iOffset][0]
            yOffsetEstOdd = xyOffsetEstListOdd[iOffset][1]
            xOffsetEstEven = xyOffsetEstListEven[iOffset][0]
            yOffsetEstEven = xyOffsetEstListEven[iOffset][1]
            xOffsetTrue = xyOffsetTrueList[iOffset][0]
            yOffsetTrue = xyOffsetTrueList[iOffset][1]

            # Verify accuracy to within half a pixel
            self.assertTrue(isclose(xOffsetTrue, xOffsetEstOdd, atol=0.51))
            self.assertTrue(isclose(yOffsetTrue, yOffsetEstOdd, atol=0.51))
            self.assertTrue(isclose(xOffsetTrue, xOffsetEstEven, atol=0.51))
            self.assertTrue(isclose(yOffsetTrue, yOffsetEstEven, atol=0.51))

    def test_field_stop_motion_calibration(self):
        """
        Test performance of calc_circular_mask_offsets() with FSs.

        Run tests on both odd- and even-sized arrays.
        """
        fnTuning = os.path.join(LOCAL_PATH, 'testdata',
                                'ut_fit_hlc_band1_field_stop_offsets.yaml')
        ppl = 2.3
        sep = 11.5  # [lambda/D]
        xOffsetMaskVec = sep*np.array([1, -1, -1, 1])  # [lambda/D]
        yOffsetMaskVec = sep*np.array([1, 1, -1, -1])  # [lambda/D]

        nOffsets = len(xOffsetMaskVec)
        nPatterns = 4
        imageListOdd = []
        imageListEven = []
        nOdd = 201
        nEven = 200
        xyOffsetTrueList = []
        for iOffset in range(nOffsets):

            xOffsetLamD = xOffsetMaskVec[iOffset]  # lambda/D
            yOffsetLamD = yOffsetMaskVec[iOffset]  # lambda/D
            xyOffsetTrueList.append([xOffsetLamD*ppl, yOffsetLamD*ppl])  # pix

            # Sum together several images with different white noise patterns
            # applied to a DM. Doing so smooths out the background and helps
            # in identifying the circular mask edge.
            imageSum = 0  # initialize
            for iPattern in range(nPatterns):

                fnIn = ('fsam_motion_calib_offsets_x'
                        '%.2flamD_y%.2flamD_pattern%d.fits' %
                        (xOffsetLamD, yOffsetLamD, iPattern))

                imageSum += fits.getdata(os.path.join(
                    LOCAL_PATH, 'testdata', fnIn), overwrite=True)

            imageSumOdd = insertinto(imageSum, (nOdd, nOdd))
            imageListOdd.append(imageSumOdd)

            imageSumEven = insertinto(imageSum, (nEven, nEven))
            imageListEven.append(imageSumEven)

        xyOffsetEstListOdd = mf.calc_circular_mask_offsets(
            imageListOdd, fnTuning)
        xyOffsetEstListEven = mf.calc_circular_mask_offsets(
            imageListEven, fnTuning)

        for iOffset in range(nOffsets):
            xOffsetEstOdd = xyOffsetEstListOdd[iOffset][0]
            yOffsetEstOdd = xyOffsetEstListOdd[iOffset][1]
            xOffsetEstEven = xyOffsetEstListEven[iOffset][0]
            yOffsetEstEven = xyOffsetEstListEven[iOffset][1]
            xOffsetTrue = xyOffsetTrueList[iOffset][0]
            yOffsetTrue = xyOffsetTrueList[iOffset][1]

            # Verify accuracy to within half a pixel
            self.assertTrue(isclose(xOffsetTrue, xOffsetEstOdd, atol=0.51))
            self.assertTrue(isclose(yOffsetTrue, yOffsetEstOdd, atol=0.51))
            self.assertTrue(isclose(xOffsetTrue, xOffsetEstEven, atol=0.51))
            self.assertTrue(isclose(yOffsetTrue, yOffsetEstEven, atol=0.51))

    def test_calc_trans_matrix_from_lowfs_images(self):
        """
        test transforms calculated from shifted lowfs images reproduce
        the transform that created them
        """
        # make up an offset list and command list
        xyOffsetList = [[0, 0],
                        [1, 0],
                        [np.cos(np.radians(120)), np.sin(np.radians(120))],
                        [np.cos(np.radians(-120)), np.sin(np.radians(-120))]
        ]
        commandList = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]

        # read reference lowfs image
        fn_im_lowfs_ref = os.path.join(
            LOWFSFIT_PATH, 'testdata', 'Im_lowfs_reference.fits')
        ref_img = fits.getdata(fn_im_lowfs_ref)

        # create shifted images from the xy offset list
        list_moving_images = [
            fft_translate(ref_img, dx, dy) for dx, dy in xyOffsetList
        ]

        # fixed value for bias+dark
        image_bias_dark = 0.7

        # add bias+dark to "measured" images
        for img in list_moving_images:
            img += image_bias_dark

        # make the call to calculate the transforms from shifted
        # lowfs images to commandList
        config_fn = os.path.join(
            LOWFSFIT_PATH, 'config_files', 'lowfsfit_parms.yaml')

        tRelFwd, tRelRev, tAbsFwd, tAbsRev = mf.calc_trans_matrix_from_lowfs_images(
            list_moving_images, commandList, fn_im_lowfs_ref,
            config_fn, image_bias_dark=image_bias_dark
        )

        # check by comparing to transforms calculated directly
        tRelFwd_t, tRelRev_t, tAbsFwd_t, tAbsRev_t = mf.calc_trans_matrix(
            xyOffsetList, commandList
        )

        for t1, t2 in zip(
                [tRelFwd, tRelRev, tAbsFwd, tAbsRev],
                [tRelFwd_t, tRelRev_t, tAbsFwd_t, tAbsRev_t]
                ):

            self.assertTrue(np.all(isclose(t1, t2, atol=0.05)))


def fft_translate(img, dx, dy):
    """
    simple function to create translated reference images for
    test_CTMfromLowfs_inputs()
    """
    nr, nc = img.shape
    y = np.linspace(np.ceil(-nr/2), np.ceil(nr/2-1), nr)
    x = np.linspace(np.ceil(-nc/2), np.ceil(nc/2-1), nc)
    X, Y = np.meshgrid(x, y)
    Fimg = np.fft.fft2(img)
    Fimg_sh = Fimg * np.fft.fftshift(np.exp(1j*2*np.pi*(X*-dx/nc + Y*-dy/nr)))
    img_sh = np.fft.ifft2(Fimg_sh)
    return np.real(img_sh)


if __name__ == '__main__':
    unittest.main()
