# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Unit tests for pupilzernikes."""
import unittest
from math import isclose
import os

import numpy as np
import astropy.io.fits as pyfits

from cal.ampthresh.ampthresh import ampthresh
from cal.conjugate.util.propcustom import gen_zernikes
from .pupilzernikes import pupilzernikes

LOCALPATH = os.path.dirname(os.path.abspath(__file__))


class TestPupilzernikes(unittest.TestCase):
    """
    Test that pupilzernikes operates correctly.

    Since pupilzernikes only calls functions that have their own unit tests,
    we just run one generated example with offsets, etc and check for correct
    result.
    """

    def test_pupilzernikes(self):
        """Fabricate an example and test success for each return value."""
        # Generate a shifted, aberrated wavefront with Zernike modes and
        # standard amplitude mask
        # pupil template diam = 300 pixels, array size = (302, 302)
        amp_path = os.path.join(
            LOCALPATH, 'testdata', 'pupil_template_D300.00pixels.fits')
        amp_pup = pyfits.getdata(amp_path)
        fn_pupil_fit_params = os.path.join(
            LOCALPATH, 'testdata', 'ut_fit_unmasked_pupil.yaml')

        diamBeam = 300
        nArray = 302

        xOffset = 0.0
        yOffset = 0.0

        maxNollZern = 11
        zernCoefIn = 2*np.array([0.17181050, 0.79094532, 0.94230985,
                                 0.41422633, 0.53925223, 0.80700204,
                                 0.64446254, 0.87674567, 0.21717132,
                                 0.78011424, 0.63288971])

        mask = ampthresh(amp_pup)

        wfe = gen_zernikes(np.arange(1, 1+maxNollZern), zernCoefIn,
                           xOffset, yOffset, diamBeam, nArray=nArray)

        # make the call
        zernCoefOut, dictResults = pupilzernikes(
            wfe, amp_pup, fn_pupil_fit_params, bMask=mask)

        # test zernike coeff
        self.assertTrue(np.all(
            [isclose(a, b, abs_tol=0.05) for a, b in zip(zernCoefIn,
                                                         zernCoefOut)]))

        # test parameter returns
        self.assertTrue(
            isclose(dictResults['diamEst'], diamBeam, rel_tol=3e-3))
        self.assertTrue(isclose(dictResults['xOffset'], xOffset, abs_tol=0.3))
        self.assertTrue(isclose(dictResults['yOffset'], yOffset, abs_tol=0.3))


class TestPupilzernikesInputFailure(unittest.TestCase):
    """Test suite for valid function inputs."""

    def test_fit_zernikes_inputs(self):
        """Test inputs for fit_zernikes."""
        list_not_2drealarray = [5, np.ones((5,)), 2j*np.ones((5, 10)), 'asdf']
        list_not_pos_scalar_int = [-1, 0, 1.5, 1j, 'string',  np.ones((5,))]

        fn_pupil_fit_params = os.path.join(
            LOCALPATH, 'testdata', 'ut_fit_unmasked_pupil.yaml')

        # Generate a shifted aberration map of only Zernike modes
        diamBeam = 300
        nArray = 320

        xOffset = -15.55
        yOffset = 30.1

        maxNollZern = 11
        zernCoefIn = 20*np.random.rand(maxNollZern)

        # Generate a ROI mask
        x = (np.arange(-nArray/2., nArray/2.) - xOffset)/diamBeam
        y = (np.arange(-nArray/2., nArray/2.) - yOffset)/diamBeam
        [X, Y] = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        mask = np.zeros((nArray, nArray))
        mask[R <= 0.5] = 1
        amp_pup = mask.copy()

        wfe = gen_zernikes(np.arange(1, maxNollZern+1), zernCoefIn,
                           xOffset, yOffset, diamBeam, nArray=nArray)

        # Make sure it runs normally first.
        pupilzernikes(wfe, amp_pup, fn_pupil_fit_params)

        for wfe_bad in list_not_2drealarray:
            with self.assertRaises(TypeError):
                pupilzernikes(wfe_bad, amp_pup, fn_pupil_fit_params)

        for amp_bad in list_not_2drealarray:
            with self.assertRaises(TypeError):
                pupilzernikes(wfe, amp_bad, fn_pupil_fit_params)

        for mask_bad in list_not_2drealarray:
            with self.assertRaises(TypeError):
                pupilzernikes(wfe, amp_pup, fn_pupil_fit_params,
                              bMask=mask_bad)

        for maxNollZernBad in list_not_pos_scalar_int:
            with self.assertRaises(TypeError):
                pupilzernikes(wfe, amp_pup, fn_pupil_fit_params,
                              Z_noll_max=maxNollZernBad)

        for mask_selem_bad in list_not_2drealarray:
            with self.assertRaises(TypeError):
                pupilzernikes(wfe, amp_pup, fn_pupil_fit_params,
                              mask_selem=mask_selem_bad)

        for nBin_bad in list_not_pos_scalar_int:
            with self.assertRaises(TypeError):
                pupilzernikes(wfe, amp_pup, fn_pupil_fit_params,
                              nBin=nBin_bad)

        for fn_bad in [1j, np.ones((5,)), np.ones((5, 10))]:
            with self.assertRaises(TypeError):
                pupilzernikes(wfe, amp_pup, fn_bad)


if __name__ == '__main__':
    unittest.main()
