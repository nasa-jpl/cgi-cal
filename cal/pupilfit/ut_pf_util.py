# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Test suite for PUPILFIT."""
import unittest
import os

from math import isclose
import numpy as np
from astropy.io import fits

from cal.util.insertinto import insertinto
from cal.ampthresh.ampthresh import ampthresh
import cal.pupilfit.pf_util as pfu

LOCALPATH = os.path.dirname(os.path.abspath(__file__))


class TestOffcenterCrop(unittest.TestCase):
    """Test suite for offcenter_crop()."""

    def setUp(self):
        """Define reused variables."""
        self.image = np.ones((15, 34))
        self.pixel_count_across = 20
        self.output_center_x = 4
        self.output_center_y = 3

    def test_offcenter_crop_input_0(self):
        """Test bad inputs to offcenter_crop."""
        for badVal in (-1, 0, 1, 1.5, (1, 2), np.ones((3, 3, 3)), 'asdf'):
            with self.assertRaises(TypeError):
                pfu.offcenter_crop(badVal,
                                   self.pixel_count_across,
                                   self.output_center_x,
                                   self.output_center_y)

    def test_offcenter_crop_input_1(self):
        """Test bad inputs to offcenter_crop."""
        for badVal in (-1, 0, 1.5, (1, 2), np.ones((3, 3, 3)), 'asdf'):
            with self.assertRaises(TypeError):
                pfu.offcenter_crop(self.image,
                                   badVal,
                                   self.output_center_x,
                                   self.output_center_y)

    def test_offcenter_crop_input_2(self):
        """Test bad inputs to offcenter_crop."""
        for badVal in (1j, (1, 2), np.ones((3, 3, 3)), 'asdf'):
            with self.assertRaises(TypeError):
                pfu.offcenter_crop(self.image,
                                   self.pixel_count_across,
                                   badVal,
                                   self.output_center_y)

    def test_offcenter_crop_input_3(self):
        """Test bad inputs to offcenter_crop."""
        for badVal in (1j, (1, 2), np.ones((3, 3, 3)), 'asdf'):
            with self.assertRaises(TypeError):
                pfu.offcenter_crop(self.image,
                                   self.pixel_count_across,
                                   self.output_center_x,
                                   badVal)

    def test_odd_in_odd_out_stay_centered(self):
        """Test a particular usage case."""
        image = np.zeros((5, 5))
        image[2::, 2::] = 1
        pupil_center_x = 3
        pupil_center_y = 3
        pixel_count_across = 7
        cropped_image = pfu.offcenter_crop(image, pixel_count_across,
                                           pupil_center_y, pupil_center_x)
        answer = insertinto(np.ones((3, 3)), (7, 7))
        self.assertTrue(np.sum(np.abs(cropped_image - answer)) == 0)

    def test_mixed_in_odd_out(self):
        """Test a particular usage case."""
        image = np.zeros((6, 5))
        image[0:3, 0:3] = 1
        pupil_center_x = 1
        pupil_center_y = 1
        pixel_count_across = 7
        cropped_image = pfu.offcenter_crop(image, pixel_count_across,
                                           pupil_center_y, pupil_center_x)
        answer = insertinto(np.ones((3, 3)), (7, 7))
        self.assertTrue(np.sum(np.abs(cropped_image - answer)) == 0)

    def test_mixed_in_odd_out_change_center(self):
        """Test a particular usage case."""
        image = np.zeros((5, 5))
        image[0:3, 2::] = 1
        pupil_center_x = 3
        pupil_center_y = 1
        pixel_count_across = 7
        cropped_image = pfu.offcenter_crop(image, pixel_count_across,
                                           pupil_center_y, pupil_center_x)
        answer = insertinto(np.ones((3, 3)), (7, 7))
        self.assertTrue(np.sum(np.abs(cropped_image - answer)) == 0)

    def test_even_in_odd_out(self):
        """Test a particular usage case."""
        image = np.zeros((6, 10))
        image[3, 3] = 1
        pupil_center_x = 3
        pupil_center_y = 3
        pixel_count_across = 7
        cropped_image = pfu.offcenter_crop(image, pixel_count_across,
                                           pupil_center_y, pupil_center_x)
        answer = insertinto(np.ones((1, 1)), (7, 7))
        self.assertTrue(np.sum(np.abs(cropped_image - answer)) == 0)

    def test_even_in_even_out(self):
        """Test a particular usage case."""
        image = np.zeros((6, 6))
        image[0:2, 0:2] = 1
        pupil_center_x = 1
        pupil_center_y = 1
        pixel_count_across = 4
        cropped_image = pfu.offcenter_crop(image, pixel_count_across,
                                           pupil_center_y, pupil_center_x)
        answer = insertinto(np.ones((2, 2)), (4, 4))
        self.assertTrue(np.sum(np.abs(cropped_image - answer)) == 0)

    def test_even_in_even_out_extend_past_input_array(self):
        """Test a particular usage case."""
        image = np.zeros((6, 6))
        image[4::, 4::] = 1
        pupil_center_x = 5
        pupil_center_y = 5
        pixel_count_across = 4
        cropped_image = pfu.offcenter_crop(image, pixel_count_across,
                                           pupil_center_y, pupil_center_x)
        answer = insertinto(np.ones((2, 2)), (4, 4))
        self.assertTrue(np.sum(np.abs(cropped_image - answer)) == 0)

    def fully_outside_input_array(self):
        """Test a particular usage case."""
        arrayIn = np.eye(4)
        cropped_image = pfu.offcenter_crop(arrayIn, 10, -20, 20)
        answer = np.zeros((10, 10))
        self.assertTrue(np.sum(np.abs(cropped_image - answer)) == 0)


class TestComputeNormFactor(unittest.TestCase):
    """Test suite for compute_norm_factor()."""

    def setUp(self):
        """Define reused variables."""
        self.nArray = 20
        self.unnorm_image = np.ones((self.nArray, self.nArray))
        self.software_mask = np.eye(self.nArray)
        self.percentile_for_norm = 55.0

    def test_compute_norm_factor_input_0(self):
        """Test bad inputs to compute_norm_factor."""
        for badVal in (-1, 0, 1, 1.5, (1, 2), np.ones((3, 3, 3)), 'asdf',
                       np.eye(self.nArray+1)):
            with self.assertRaises(TypeError):
                pfu.compute_norm_factor(badVal,
                                        self.software_mask,
                                        self.percentile_for_norm)

    def test_compute_norm_factor_input_1(self):
        """Test bad inputs to compute_norm_factor."""
        for badVal in (-1, 0, 1, 1.5, (1, 2), np.ones((3, 3, 3)), 'asdf',
                       2*np.eye(self.nArray), np.eye(self.nArray+1)):
            with self.assertRaises(TypeError):
                pfu.compute_norm_factor(self.unnorm_image,
                                        badVal,
                                        self.percentile_for_norm)

    def test_compute_norm_factor_input_2(self):
        """Test bad inputs to compute_norm_factor."""
        for badVal in (-1, 100.5, (1, 2), np.ones((3, 3, 3)), 'asdf'):
            with self.assertRaises(TypeError):
                pfu.compute_norm_factor(self.unnorm_image,
                                        self.software_mask,
                                        badVal)

    def test_pupil_image_normalization(self):
        """
        Scale pupil images and recover the correct normalization factor.

        This test function alters the normalization of the input images to
        check if pfu.compute_pupil_mask_offset() can re-normalize them
        correctly. One case with pre-normalized images is also included.
        Tests are performed for all allowed combinations of masks at several
        offsets.
        """
        scale_factor_vector = np.array([1.0, 2.3, 0.1, 3.0])

        percentile_for_norm = 50

        localpath = os.path.dirname(os.path.abspath(__file__))
        fn = os.path.join(localpath, 'testdata',
                          'imageUnmasked_SPCSPEC_LS2SPM.fits')
        image0 = fits.getdata(fn)

        for scale_factor in scale_factor_vector:
            imageNew = scale_factor*image0
            software_mask = ampthresh(imageNew)
            norm_factor_est = pfu.compute_norm_factor(imageNew,
                                                      software_mask,
                                                      percentile_for_norm)
            self.assertTrue(isclose(norm_factor_est, scale_factor,
                                    rel_tol=1e-3))


class TestCoarselyLocatePupilOffsetsOnce(unittest.TestCase):
    """Test suite for coarsely_locate_pupil_offset_once()."""

    def setUp(self):
        """Define reused variables."""
        self.pupil_image = np.ones((50, 50))
        self.n_points_beam = 10
        self.x_offset_start = 5
        self.y_offset_start = 3
        self.search_radius = 20
        self.search_step_size = 2

    def test_bad_input_0(self):
        """Test bad inputs to coarsely_locate_pupil_offset_once()."""
        for badVal in (-1, 0, 1, 1.5, 1j, (1, 2), np.ones((3, 3, 3)), 'asdf',
                       1j*np.eye(50)):
            with self.assertRaises(TypeError):
                pfu.coarsely_locate_pupil_offset_once(
                    badVal, self.n_points_beam,
                    self.x_offset_start, self.y_offset_start,
                    self.search_radius, self.search_step_size,
                )

    def test_bad_input_1(self):
        """Test bad inputs to coarsely_locate_pupil_offset_once()."""
        for badVal in (-1, 0, 1j, (1, 2), np.ones((3, 3, 3)), 'asdf',
                       1j*np.eye(50)):
            with self.assertRaises(TypeError):
                pfu.coarsely_locate_pupil_offset_once(
                    self.pupil_image, badVal,
                    self.x_offset_start, self.y_offset_start,
                    self.search_radius, self.search_step_size,
                )

    def test_bad_input_2(self):
        """Test bad inputs to coarsely_locate_pupil_offset_once()."""
        for badVal in (-1.5, 1.5, 1j, (1, 2), np.ones((3, 3, 3)), 'asdf',
                       1j*np.eye(50)):
            with self.assertRaises(TypeError):
                pfu.coarsely_locate_pupil_offset_once(
                    self.pupil_image, self.n_points_beam,
                    badVal, self.y_offset_start,
                    self.search_radius, self.search_step_size,
                )

    def test_bad_input_3(self):
        """Test bad inputs to coarsely_locate_pupil_offset_once()."""
        for badVal in (-1.5, 1.5, 1j, (1, 2), np.ones((3, 3, 3)), 'asdf',
                       1j*np.eye(50)):
            with self.assertRaises(TypeError):
                pfu.coarsely_locate_pupil_offset_once(
                    self.pupil_image, self.n_points_beam,
                    self.x_offset_start, badVal,
                    self.search_radius, self.search_step_size,
                )

    def test_bad_input_4(self):
        """Test bad inputs to coarsely_locate_pupil_offset_once()."""
        for badVal in (-1.5, 0, 1.5, 1j, (1, 2), np.ones((3, 3, 3)), 'asdf',
                       1j*np.eye(50)):
            with self.assertRaises(TypeError):
                pfu.coarsely_locate_pupil_offset_once(
                    self.pupil_image, self.n_points_beam,
                    self.x_offset_start, self.y_offset_start,
                    badVal, self.search_step_size,
                )

    def test_bad_input_5(self):
        """Test bad inputs to coarsely_locate_pupil_offset_once()."""
        for badVal in (-1.5, 0, 1.5, 1j, (1, 2), np.ones((3, 3, 3)), 'asdf',
                       1j*np.eye(50)):
            with self.assertRaises(TypeError):
                pfu.coarsely_locate_pupil_offset_once(
                    self.pupil_image, self.n_points_beam,
                    self.x_offset_start, self.y_offset_start,
                    self.search_radius, badVal,
                )

    def test_coarsely_locate_pupil_offset_once(self):
        """Test performance of coarsely_locate_pupil_offset_once()."""
        x_offset_start = 0
        y_offset_start = 0
        n_points_beam = 386.0
        magVec = [1+0.025, 1-0.025, 1]
        clockingVec = [-1., 1., 0.]
        xOffsetVec = [-123.1, 123.1, 0.]
        yOffsetVec = [205.5, -205.5, 0.]
        search_radius = 220
        search_step_size = 20

        for ind in range(len(xOffsetVec)):
            xOffset = xOffsetVec[ind]
            yOffset = yOffsetVec[ind]
            mag = magVec[ind]
            clocking = clockingVec[ind]

            localpath = os.path.dirname(os.path.abspath(__file__))
            unmasked_image_fn = os.path.join(localpath, 'testdata',
                'sim_pupil_meas_mag%.3f_clk%.3f_x%.1f_y%.1f.fits' %
                (mag, clocking, xOffset, yOffset))
            pupil_image = fits.getdata(unmasked_image_fn, ext=0)
            xOffsetEst, yOffsetEst, _ = \
                pfu.coarsely_locate_pupil_offset_once(
                    pupil_image, mag*n_points_beam, x_offset_start,
                    y_offset_start, search_radius, search_step_size)
            max_est_error = np.max(np.abs((xOffset - xOffsetEst,
                                           yOffset - yOffsetEst)))

            self.assertTrue(max_est_error <= search_step_size)


class TestCoarselyLocatePupilOffsets(unittest.TestCase):
    """Test suite for coarsely_locate_pupil_offset()."""

    def setUp(self):
        """Define reused variables."""
        self.pupil_image = 10*np.eye(20)
        self.n_points_beam = 5
        self.shrink_factor = 3.

    def test_bad_input_0(self):
        """Test bad inputs to coarsely_locate_pupil_offset()."""
        for badVal in (-1, 0, 1, 1.5, 1j, (1, 2), np.ones((3, 3, 3)), 'asdf',
                       1j*np.eye(50)):
            with self.assertRaises(TypeError):
                pfu.coarsely_locate_pupil_offset(
                    badVal, self.n_points_beam, self.shrink_factor)

    def test_bad_input_1(self):
        """Test bad inputs to coarsely_locate_pupil_offset()."""
        for badVal in (-1, 0, 1j, (1, 2), np.ones((3, 3, 3)), 'asdf',
                       1j*np.eye(50)):
            with self.assertRaises(TypeError):
                pfu.coarsely_locate_pupil_offset(
                    self.pupil_image, badVal, self.shrink_factor)

    def test_bad_input_2(self):
        """Test bad inputs to coarsely_locate_pupil_offset()."""
        for badVal in (-1, 0, 0.5, 1j, (1, 2), np.ones((3, 3, 3)), 'asdf',
                       1j*np.eye(50)):
            with self.assertRaises(TypeError):
                pfu.coarsely_locate_pupil_offset(
                    self.pupil_image, self.n_points_beam, badVal)

    def test_coarsely_locate_pupil_offset(self):
        """Test performance of coarsely_locate_pupil_offset()."""
        n_points_beam = 386.0
        magVec = [1+0.025, 1-0.025, 1]
        clockingVec = [-1., 1., 0.]
        xOffsetVec = [-123.1, 123.1, 0.]
        yOffsetVec = [205.5, -205.5, 0.]

        for ind in range(len(xOffsetVec)):
            xOffset = xOffsetVec[ind]
            yOffset = yOffsetVec[ind]
            mag = magVec[ind]
            clocking = clockingVec[ind]

            localpath = os.path.dirname(os.path.abspath(__file__))
            unmasked_image_fn = os.path.join(localpath, 'testdata',
                'sim_pupil_meas_mag%.3f_clk%.3f_x%.1f_y%.1f.fits' %
                (mag, clocking, xOffset, yOffset))
            pupil_image = fits.getdata(unmasked_image_fn, ext=0)
            xOffsetEst, yOffsetEst = \
                pfu.coarsely_locate_pupil_offset(pupil_image,
                                                 mag*n_points_beam)
            max_est_error = np.max(np.abs((xOffset - xOffsetEst,
                                           yOffset - yOffsetEst)))
            # 3 pixels is good enough for the coarse step
            self.assertTrue(max_est_error <= 3)


class TestComputeLateralOffsets(unittest.TestCase):
    """Test suite for compute_lateral_offsets."""

    def setUp(self):
        """Define reused variables."""
        self.arrayMeas = np.ones((10, 11))
        self.arrayRef = np.ones((10, 11))
        self.diamEst = 5.5
        self.nPhaseSteps = 29
        self.dPixel = 0.4
        self.nPadFFT = 512
        self.nFocusCrop = 15
        self.useCoarse = True
        self.nIter = 1

    def test_bad_input_0(self):
        """Test bad inputs to compute_lateral_offsets()."""
        for badVal in (-1, 0, 1, 1.5, 1j, (1, 2), np.ones((3, 3, 3)), 'asdf',
                       1j*np.eye(50)):
            with self.assertRaises(TypeError):
                pfu.compute_lateral_offsets(
                    badVal, self.arrayRef, self.diamEst,
                    self.nPhaseSteps, self.dPixel, self.nPadFFT,
                    self.nFocusCrop, useCoarse=self.useCoarse,
                    nIter=self.nIter)

    def test_bad_input_1(self):
        """Test bad inputs to compute_lateral_offsets()."""
        for badVal in (-1, 0, 1, 1.5, 1j, (1, 2), np.ones((3, 3, 3)), 'asdf',
                       1j*np.eye(50)):
            with self.assertRaises(TypeError):
                pfu.compute_lateral_offsets(
                    self.arrayMeas, badVal, self.diamEst,
                    self.nPhaseSteps, self.dPixel, self.nPadFFT,
                    self.nFocusCrop, useCoarse=self.useCoarse,
                    nIter=self.nIter)

    def test_bad_input_2(self):
        """Test bad inputs to compute_lateral_offsets()."""
        for badVal in (-1, 0, 1j, (1, 2), np.ones((3, 3, 3)), 'asdf',
                       1j*np.eye(50)):
            with self.assertRaises(TypeError):
                pfu.compute_lateral_offsets(
                    self.arrayMeas, self.arrayRef, badVal,
                    self.nPhaseSteps, self.dPixel, self.nPadFFT,
                    self.nFocusCrop, useCoarse=self.useCoarse,
                    nIter=self.nIter)

    def test_bad_input_3(self):
        """Test bad inputs to compute_lateral_offsets()."""
        for badVal in (-1, 0, 1.5, 1j, (1, 2), np.ones((3, 3, 3)), 'asdf',
                       1j*np.eye(50)):
            with self.assertRaises(TypeError):
                pfu.compute_lateral_offsets(
                    self.arrayMeas, self.arrayRef, self.diamEst,
                    badVal, self.dPixel, self.nPadFFT,
                    self.nFocusCrop, useCoarse=self.useCoarse,
                    nIter=self.nIter)

    def test_bad_input_4(self):
        """Test bad inputs to compute_lateral_offsets()."""
        for badVal in (-1, 0, 1j, (1, 2), np.ones((3, 3, 3)), 'asdf',
                       1j*np.eye(50)):
            with self.assertRaises(TypeError):
                pfu.compute_lateral_offsets(
                    self.arrayMeas, self.arrayRef, self.diamEst,
                    self.nPhaseSteps, badVal, self.nPadFFT,
                    self.nFocusCrop, useCoarse=self.useCoarse,
                    nIter=self.nIter)

    def test_bad_input_5(self):
        """Test bad inputs to compute_lateral_offsets()."""
        for badVal in (-1, 0, 1.5, 1j, (1, 2), np.ones((3, 3, 3)), 'asdf',
                       1j*np.eye(50)):
            with self.assertRaises(TypeError):
                pfu.compute_lateral_offsets(
                    self.arrayMeas, self.arrayRef, self.diamEst,
                    self.nPhaseSteps, self.dPixel, badVal,
                    self.nFocusCrop, useCoarse=self.useCoarse,
                    nIter=self.nIter)

    def test_bad_input_6(self):
        """Test bad inputs to compute_lateral_offsets()."""
        for badVal in (-1, 0, 1.5, 1j, (1, 2), np.ones((3, 3, 3)), 'asdf',
                       1j*np.eye(50)):
            with self.assertRaises(TypeError):
                pfu.compute_lateral_offsets(
                    self.arrayMeas, self.arrayRef, self.diamEst,
                    self.nPhaseSteps, self.dPixel, self.nPadFFT,
                    badVal, useCoarse=self.useCoarse,
                    nIter=self.nIter)

    def test_bad_input_7(self):
        """Test bad inputs to compute_lateral_offsets()."""
        for badVal in (-1, 0, 1, 1.5, 1j, (1, 2), np.ones((3, 3, 3)), 'asdf',
                       1j*np.eye(50)):
            with self.assertRaises(TypeError):
                pfu.compute_lateral_offsets(
                    self.arrayMeas, self.arrayRef, self.diamEst,
                    self.nPhaseSteps, self.dPixel, self.nPadFFT,
                    self.nFocusCrop, useCoarse=badVal,
                    nIter=self.nIter)

    def test_bad_input_8(self):
        """Test bad inputs to compute_lateral_offsets()."""
        for badVal in (-1, 0, 1.5, 1j, (1, 2), np.ones((3, 3, 3)), 'asdf',
                       1j*np.eye(50)):
            with self.assertRaises(TypeError):
                pfu.compute_lateral_offsets(
                    self.arrayMeas, self.arrayRef, self.diamEst,
                    self.nPhaseSteps, self.dPixel, self.nPadFFT,
                    self.nFocusCrop, useCoarse=self.useCoarse,
                    nIter=badVal)

    def test_compute_lateral_offsets(self):
        """Test performance of compute_lateral_offsets()."""
        xOffsetTrue = 5
        yOffsetTrue = 12
        arrayRef = insertinto(np.ones((200, 230)), (250, 250))
        arrayMeas = np.roll(arrayRef, (yOffsetTrue, xOffsetTrue), axis=(0, 1))
        diamEst = np.sqrt(200**2 + 230**2)

        for ii in range(3):

            if ii == 0:
                nPhaseSteps = 80
                dPixel = 0.5
                nPadFFT = 512
                nFocusCrop = 7
                useCoarse = True  # Default is true
                nIter = 2  # Default is 1
            elif ii == 1:
                nPhaseSteps = 30
                dPixel = 1
                nPadFFT = 1024
                nFocusCrop = 15
                useCoarse = False  # Default is true
                nIter = 2  # Default is 1
            elif ii == 2:
                nPhaseSteps = 50
                dPixel = 0.2
                nPadFFT = 512
                nFocusCrop = 7
                useCoarse = True  # Default is true
                nIter = 1  # Default is 1

            yOffsetEst, xOffsetEst = pfu.compute_lateral_offsets(arrayMeas,
                arrayRef, diamEst, nPhaseSteps, dPixel, nPadFFT, nFocusCrop,
                useCoarse=useCoarse, nIter=nIter)
            self.assertTrue(isclose(yOffsetTrue, yOffsetEst, abs_tol=0.01),
                            msg=('y true / est:  %.3f  %.3f' %
                                 (yOffsetTrue, yOffsetEst)))
            self.assertTrue(isclose(xOffsetTrue, xOffsetEst, abs_tol=0.01),
                            msg=('x true / est:  %.3f  %.3f' %
                                 (xOffsetTrue, xOffsetEst)))


if __name__ == '__main__':
    unittest.main()
