# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Test suite for pupilfit.pupilfit_open."""
import unittest
import os

import numpy as np
from astropy.io import fits

import cal.pupilfit.pupilfit_open as pfo

LOCALPATH = os.path.dirname(os.path.abspath(__file__))


class TestFitUnmaskedPupil(unittest.TestCase):
    """Test suite for fit_unmasked_pupil()."""

    def setUp(self):
        """Define reused variables."""
        self.fn_tuning = os.path.join(
            LOCALPATH, 'testdata', 'ut_fit_unmasked_pupil.yaml')

    def test_fit_unmasked_pupil(self):
        """Test that the unmasked pupil parameters are fitted correctly."""
        nBeamNom = 386.
        magVec = [1+0.025, 1-0.025, 1]
        clockingVec = [-1., 1., 0.]
        xOffsetVec = [-123.1, 123.1, 0.]
        yOffsetVec = [205.5, -205.5, 0.]

        for ind in range(len(xOffsetVec)):

            xOffsetTrue = xOffsetVec[ind]
            yOffsetTrue = yOffsetVec[ind]
            magTrue = magVec[ind]
            clockDegTrue = clockingVec[ind]

            pupilfn = os.path.join(LOCALPATH, 'testdata',
              'sim_pupil_meas_mag%.3f_clk%.3f_x%.1f_y%.1f.fits' %
              (magTrue, clockDegTrue, xOffsetTrue, yOffsetTrue))
            pupilMeas = fits.getdata(pupilfn)

            xOffsetEst, yOffsetEst, clockDegEst, diamEst = \
                pfo.fit_unmasked_pupil(pupilMeas, self.fn_tuning)

            self.assertTrue(np.abs(xOffsetEst-xOffsetTrue) < 0.2)  # pixels
            self.assertTrue(np.abs(yOffsetEst-yOffsetTrue) < 0.2)  # pixels
            self.assertTrue(np.abs(clockDegEst-clockDegTrue)*(1e3*np.pi/180.)
                            < 2.)  # mrad
            diamTrue = magTrue*nBeamNom
            self.assertTrue((diamEst - diamTrue)/diamTrue*100 < 0.3)  # percent

    def test_fit_unmasked_pupil_inputs(self):
        """Test incorrect inputs of fit_unmasked_pupil."""
        # YAML filename tests are handled separately in unit tests for
        # loadyaml.
        for badPupil in (-1, 0, 1, 1.5, 1j, 1j*np.eye(6), np.ones((3, 3, 3))):
            with self.assertRaises(TypeError):
                pfo.fit_unmasked_pupil(badPupil, self.fn_tuning)

    def test_invalid_fn_tuning(self):
        """Invalid inputs caught as expected"""
        pupil = np.ones((5, 5))

        perrlist = [-1, 0, 1.5, 1j, None, np.ones((2,)), np.ones((2, 2, 2))]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                pfo.fit_unmasked_pupil(pupil, perr)

    def test_invalid_data_path(self):
        """Invalid inputs caught as expected"""
        pupil = np.ones((5, 5))
        perrlist = [-1, 0, 1.5, 1j, None, np.ones((2,)), np.ones((2, 2, 2))]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                pfo.fit_unmasked_pupil(
                    pupil=pupil,
                    fn_tuning=self.fn_tuning,
                    data_path=perr,
                )


if __name__ == '__main__':
    unittest.main()
