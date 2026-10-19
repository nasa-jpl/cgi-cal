# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Unit test suite for psffit.py."""

import unittest
import os
import numpy as np
from scipy import ndimage
from astropy.io import fits

from cal.psffit.psffit import psffit

# ignore divide by zero errors
np.seterr(divide='ignore', invalid='ignore')


class TestPsfFit(unittest.TestCase):
    """Unit tests for psffit."""

    # sort out paths
    local_path = os.path.dirname(os.path.realpath(__file__))
    ut_file_path = os.path.join(local_path, 'ut_data')
    ut_fits_file = os.path.join(ut_file_path, 'ut_psf_fits.fits')

    # example config file
    config_file = os.path.join(os.path.join(local_path, 'config_files'),
                               'psffit_parms.yaml')

    def test_offset_calc(self):
        """Test shift returned by psffit."""
        # discrepancy tolerance
        tol = 1/6   # 1/6 pixel accuracy relative to 'truth'

        # load in reference image
        img = (np.array(fits.getdata(self.ut_fits_file, 0)))[0, :, :]
        img /= np.amax(img)

        # shift reference image by known amount ()'the answer')
        the_answer = np.array([80.1335, -110.1312])
        img_shifted = ndimage.shift(img, the_answer)
        img_shifted /= np.amax(img_shifted)

        # use psffit to comput offset between two images
        shift_computed, _ = psffit(img, img_shifted, self.config_file)

        # check that the computed shift of image is the same as 'the answer'
        self.assertTrue(
            np.abs(np.amax(shift_computed - the_answer)) <= tol)


    def test_amplitude_calc(self):
        """Test amplitude estimate from PSF fit."""
        # discrepancy tolerance
        tol = 0.1   # 10% accuracy on amplitude recovery

        # load in reference image
        img = (np.array(fits.getdata(self.ut_fits_file, 0)))[0, :, :]

        # normalize
        img /= np.amax(img)

        # shift reference image by known amount
        shift_amt = [80.1335, -110.1312]

        # scale shifted image by known amplitude
        the_answer = 0.252
        img_shifted = ndimage.shift(img * the_answer, shift_amt)

        # use psffit to comput offset between two images
        _, amplitude = psffit(img, img_shifted, self.config_file)

        # check that the computed amplitude is the same as 'the answer'
        self.assertTrue(
            np.abs(np.amax(amplitude - the_answer)) / the_answer <= tol)
        pass


    def test_offset_calc_noise(self):
        """Test shift returned by psffit, including noise."""
        # discrepancy tolerance
        tol = 1/6  # 1/6 pixel accuracy relative to 'truth'
        nfuzz = 50

        # load in reference image
        img = (np.array(fits.getdata(self.ut_fits_file, 0)))[0, :, :]
        img /= np.amax(img)

        # scale image to specified SNR
        snr = 20.
        img *= (snr ** 2.)

        # shift reference image by known amount ()'the answer')
        the_answer = np.array([80.5335, -110.31312])
        img_shifted = ndimage.shift(img, the_answer)

        # add noise
        rng = np.random.default_rng(55555)
        for _ in range(nfuzz):
            noise_arr = rng.standard_normal(img.shape) * snr
            img_noise = img_shifted + noise_arr

            # use psffit to comput offset between two images
            shift_computed, _ = psffit(img, img_noise, self.config_file)

            # check that the computed shift of image is same as 'the answer'
            self.assertTrue(
                np.abs(np.amax(shift_computed - the_answer)) <= tol)
            pass
        pass


    def test_amplitude_calc_noise(self):
        """Test amplitude returned by psffit, including noise."""
        # discrepancy tolerance
        tol = 0.1   # 10% accuracy on amplitude recovery
        nfuzz = 50

        # load in reference image
        img = (np.array(fits.getdata(self.ut_fits_file, 0)))[0, :, :]
        img /= np.amax(img)

        # scale image to specified SNR
        snr = 40.
        the_answer = snr ** 2.

        # shift reference image by known amount
        shift_amt = [80.1335, -110.1312]

        # scale image by known offset, and scale up to 'the answer'
        img_shifted = ndimage.shift(img, shift_amt)
        img_shifted *= the_answer

        # add noise
        rng = np.random.default_rng(44444)
        for _ in range(nfuzz):
            noise_arr = rng.standard_normal(img.shape) * snr
            img_noise = img_shifted + noise_arr

            # use psffit to comput offset between two images
            _, amplitude = psffit(img, img_noise, self.config_file)

            # check that the computed amplitude is the same as 'the answer'
            self.assertTrue(
                np.abs(np.amax(amplitude - the_answer)) / the_answer <= tol)
            pass
        pass


# check if this is run in main or not
if __name__ == '__main__':
    unittest.main()
