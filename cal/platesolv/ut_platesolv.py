# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Unit test suite for platesolv.py."""

import unittest
import os
import numpy as np
from astropy.io import fits

from cal.platesolv.platesolv import platesolv, PlateSolvException
from cal.util.mft import do_imft, do_mft

# sort out paths
LOCAL_PATH = os.path.dirname(os.path.realpath(__file__))
UT_FILE_PATH = os.path.join(LOCAL_PATH, 'ut_data')
UT_FITS_FILE = os.path.join(UT_FILE_PATH, 'ut_psf_fits.fits')
UT_FITS_IMAGE = (fits.getdata(UT_FITS_FILE))[0, :, :]


class TestPlateSolv(unittest.TestCase):
    """Unit test suite for platesolv()."""

    # test lambda/D/pixel estimate returned by platesolv
    def test_platescale_calc(self):
        """Check that platescale is recovered for sample image."""
        # discrepancy tolerance
        tol = 0.01   # 1% tolerance on lambda/d/pixel

        # reference pixel period
        # 'the answer'
        pixperiod_answer = 1.25  # reference lambda/d/pixel

        # 'the guess'
        pixperiod_guess = 1.32

        # pixperpupil value does not need to be accurate, just the same for all
        # mft and ifmt calls
        pixperpupil = np.max(UT_FITS_IMAGE.shape)

        # make fake image with the 'wrong' plate scale,
        # based on pupil image of correct scaling
        pupil_image_simulated = do_imft(UT_FITS_IMAGE,
                                        UT_FITS_IMAGE.shape,
                                        pixperiod_answer,
                                        pixperpupil)
        answer_image = np.real(do_mft(pupil_image_simulated,
                                      UT_FITS_IMAGE.shape,
                                      pixperiod_answer,
                                      pixperpupil))
        guess_image = np.real(do_mft(pupil_image_simulated,
                                     UT_FITS_IMAGE.shape,
                                     pixperiod_guess,
                                     pixperpupil))

        # assuming the simulated image was generated at the wrong plate scale
        # (pixperiod_guess), solve for the correct one
        best_fit, _, _ = platesolv(answer_image, guess_image, pixperiod_guess)

        # check that the computed place scale is the same as 'the answer'
        # print(np.abs(best_fit - pixperiod_answer) / pixperiod_answer)
        self.assertTrue(np.abs(best_fit - pixperiod_answer) /
                        pixperiod_answer <= tol)
        pass

    def test_snr_scaling(self):
        """Check that platescale is recovered in the presence of noise."""
        # discrepancy tolerance
        tol = 0.05   # 5% tolerance at low SNR

        # reference pixel period
        # 'the answer'
        pixperiod_answer = 1.25  # reference lambda/d/pixel

        # 'the guess'
        pixperiod_guess = 1.32

        # pixperpupil value does not need to be accurate, just the same for all
        # mft and ifmt calls
        pixperpupil = np.max(UT_FITS_IMAGE.shape)

        # snr value of 'recorded' image
        snr = 100.
        nfuzz = 50

        # make fake image with the 'wrong' plate scale,
        # based on pupil image of correct scaling
        pupil_image_simulated = do_imft(UT_FITS_IMAGE,
                                        UT_FITS_IMAGE.shape,
                                        pixperiod_answer,
                                        pixperpupil)
        answer_image = np.real(do_mft(pupil_image_simulated,
                                      UT_FITS_IMAGE.shape,
                                      pixperiod_answer,
                                      pixperpupil))
        guess_image = np.real(do_mft(pupil_image_simulated,
                                     UT_FITS_IMAGE.shape,
                                     pixperiod_guess,
                                     pixperpupil))

        # add noise to comparison image
        rng = np.random.default_rng(7654321)
        for dummy in range(nfuzz):
            noise_arr = rng.standard_normal(answer_image.shape) * snr
            im_noise = (answer_image / np.amax(answer_image)) * (snr ** 2.) + \
                noise_arr

            # assuming the simulated image was generated at the wrong plate
            # scale (pixperiod_guess), solve for the correct one
            best_fit, _, _ = platesolv(im_noise, guess_image, pixperiod_guess)

            # check that the computed shift of image is same as 'the answer'
            # print(np.abs(best_fit - pixperiod_answer) / pixperiod_answer)
            frac_error = np.abs(best_fit - pixperiod_answer)/pixperiod_answer
            self.assertTrue(frac_error <= tol)
            pass
        pass


    def test_pixperlod_platesolv(self):
        """Check pixel period is valid type for platesolv()."""
        for pixperlod in [-1.5, -1, 0, 1j, [], 'perr', (5,)]:
            with self.assertRaises(PlateSolvException):
                platesolv(UT_FITS_IMAGE, UT_FITS_IMAGE, pixperlod)
                pass
            pass
        pass

    def test_img_2Darray_platesolv(self):
        """Check input image type valid in platesolv()."""
        pixperlod = 1.2

        for e in [np.ones((6,)), np.ones((6, 6, 6)), [], 'fft']:
            with self.assertRaises(PlateSolvException):
                platesolv(e, e, pixperlod)
                pass
            pass
        pass

    def test_pixperlod_platesolv_input_sizes(self):
        """Check input images sizes are valid."""
        pixperlod = 1.2
        img1 = np.ones((512, 512))
        img2 = np.ones((256, 256))
        with self.assertRaises(PlateSolvException):
            platesolv(img1, img2, pixperlod)
            pass
        pass


if __name__ == '__main__':
    unittest.main()
