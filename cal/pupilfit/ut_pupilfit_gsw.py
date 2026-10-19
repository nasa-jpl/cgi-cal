# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Test suite for pupilfit_gsw.py."""
import unittest
import os

import numpy as np
from astropy.io import fits

from cal.util.loadyaml import loadyaml
import cal.pupilfit.pupilfit_gsw as pg

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))


class TestShapedPupilMagClocking(unittest.TestCase):
    """Unit tests for SPAM calibration."""

    def setUp(self):
        """Define reused variables."""
        self.fnShapedPupilCalib = os.path.join(
            LOCAL_PATH, 'testdata', 'ut_fit_spam_mag_clocking_rot90.yaml')

    def test_fit_shaped_pupil_mag_clocking(self):
        """Functional test of fit_shaped_pupil_mag_clocking."""
        for case in np.arange(3):

            if case == 0:
                mag = 1.02
                clockAllDeg = 1.0  # 2.0  # [degrees]
                xcImagePix = -120
                ycImagePix = 200
                unnormFac = 897.2  # factor applied to un-normalize the image
                fn = self.fnShapedPupilCalib

            elif case == 1:
                mag = 0.98
                clockAllDeg = -0.92  # 2.0  # [degrees]
                xcImagePix = 175
                ycImagePix = -53
                unnormFac = 0.11  # factor applied to un-normalize the image
                fn = self.fnShapedPupilCalib

            elif case == 2:
                mag = 1.00
                clockAllDeg = 0.0  # 2.0  # [degrees]
                xcImagePix = 21
                ycImagePix = 30
                unnormFac = 1.0  # factor applied to un-normalize the image
                fn = os.path.join(LOCAL_PATH, 'testdata',
                                  'ut_fit_spam_mag_clocking_rot0.yaml')

            else:
                pass

            fnImage = os.path.join(
                LOCAL_PATH, 'testdata',
                'sim_spm_calib_spots_meas_mag%.3f_clk%.3f_x%d_y%d_norm%.2f'
                '.fits' %
                (mag, clockAllDeg, xcImagePix, ycImagePix, unnormFac))
            spotMeas = fits.getdata(fnImage)

            magEst, clockEst = \
                pg.fit_shaped_pupil_mag_clocking(spotMeas, fn)

            self.assertTrue((np.abs(magEst - mag)/mag*100)
                            <= 0.2)  # percent mag error
            self.assertTrue(np.abs(clockEst - clockAllDeg)*(np.pi/180.)*1e3
                            < 1.5)  # mrad

    def test_fit_shaped_pupil_mag_clocking_inputs(self):
        """Test the inputs of fit_shaped_pupil_mag_clocking."""
        # YAML filename tests are done separately using ut_loadyaml.py
        for arg0 in (np.ones((5,)), np.ones((3, 4, 5)), 2, 'asdf'):
            with self.assertRaises(TypeError):
                pg.fit_shaped_pupil_mag_clocking(arg0,
                                                 self.fnShapedPupilCalib)


class TestLyotStopMagClocking(unittest.TestCase):
    """Unit tests for LSAM calibration."""

    def test_fit_lyot_stop_mag_clocking(self):
        """Functional test of fit_lyot_stop_mag_clocking."""
        unmaskfn = os.path.join(LOCAL_PATH, 'testdata',
                                'imageUnmasked_SPCSPEC_LS2PUPIL.fits')
        imageUnmasked = fits.getdata(unmaskfn)
        maskfn = os.path.join(LOCAL_PATH, 'testdata',
                              'imageMasked_SPCSPEC_LS2PUPIL.fits')
        imageMasked = fits.getdata(maskfn)

        fnMaskParams = os.path.join(
            LOCAL_PATH, 'testdata',
            'ut_params_for_fitting_lyot_stop_mag_clocking.yaml'
        )
        fnOffsetParams = os.path.join(LOCAL_PATH, 'testdata',
                                      'fit_pupil_mask_offsets.yaml')
        fnPupilLateralOffsets = os.path.join(LOCAL_PATH, 'testdata',
                                             'meas_pupil_lateral_offsets.yaml')
        fnPupilMagClocking = os.path.join(LOCAL_PATH, 'testdata',
                                          'meas_pupil_diam_clocking.yaml')
        fnLyotStopMagClocking = os.path.join(
            LOCAL_PATH, 'testdata', 'meas_lyot_stop_mag_clocking.yaml')
        fnLyotCalib = os.path.join(LOCAL_PATH, 'testdata',
                                   'fit_lsam_mag_clocking.yaml')

        inp = loadyaml(fnPupilMagClocking)
        inp = loadyaml(fnPupilLateralOffsets)
        xOffsetPupil = inp['xOffsetPupil']
        yOffsetPupil = inp['yOffsetPupil']
        inp = loadyaml(fnLyotStopMagClocking)
        magAtLSAM = inp['magAtLSAM']
        clockDegLyot = inp['clockDegLyot']

        magEst, clockEst = pg.fit_lyot_stop_mag_clocking(
            imageUnmasked, imageMasked, xOffsetPupil, yOffsetPupil,
            fnMaskParams, fnOffsetParams, fnLyotCalib)

        self.assertTrue((np.abs(magEst - magAtLSAM)/magAtLSAM*100)
                        <= 0.2)  # percent mag error
        self.assertTrue(np.abs(clockEst - clockDegLyot)*(np.pi/180.)*1e3
                        < 1.5)  # mrad

    def test_fit_lyot_stop_mag_clocking_inputs(self):
        """Test the inputs of fit_lyot_stop_mag_clocking."""
        case = 'SPCSPEC_LS2PUPIL'

        unmaskfn = os.path.join(LOCAL_PATH, 'testdata',
                                'imageUnmasked_%s.fits' % (case))
        imageUnmasked = fits.getdata(unmaskfn)
        maskfn = os.path.join(LOCAL_PATH, 'testdata',
                              'imageMasked_%s.fits' % (case))
        imageMasked = fits.getdata(maskfn)

        fnMaskParams = os.path.join(
            LOCAL_PATH, 'testdata',
            'ut_params_for_fitting_lyot_stop_mag_clocking.yaml',
        )
        fnOffsetParams = os.path.join(LOCAL_PATH, 'testdata',
                                      'fit_pupil_mask_offsets.yaml')
        fnPupilLateralOffsets = os.path.join(
            LOCAL_PATH, 'testdata', 'meas_pupil_lateral_offsets.yaml')
        fnLyotCalib = os.path.join(
            LOCAL_PATH, 'testdata', 'fit_lsam_mag_clocking.yaml')

        inp = loadyaml(fnPupilLateralOffsets)
        xOffsetPupil = inp['xOffsetPupil']
        yOffsetPupil = inp['yOffsetPupil']

        # Check standard inputs do not raise anything first
        pg.fit_lyot_stop_mag_clocking(imageUnmasked, imageMasked,
                                      xOffsetPupil, yOffsetPupil,
                                      fnMaskParams, fnOffsetParams,
                                      fnLyotCalib)

        # Test with incorrect inputs
        # YAML filename tests are done separately using ut_loadyaml.py for
        # content; do type-checking (i.e. it's a string) here
        for imageUnmaskedBad in (1j, (10,), np.ones((10, 10, 10)), 'str'):
            with self.assertRaises(TypeError):
                pg.fit_lyot_stop_mag_clocking(imageUnmaskedBad,
                                              imageMasked,
                                              xOffsetPupil,
                                              yOffsetPupil,
                                              fnMaskParams,
                                              fnOffsetParams,
                                              fnLyotCalib)
        for imageMaskedBad in (1j, (10,), np.ones((10, 10, 10)), 'str'):
            with self.assertRaises(TypeError):
                pg.fit_lyot_stop_mag_clocking(imageUnmasked,
                                              imageMaskedBad,
                                              xOffsetPupil,
                                              yOffsetPupil,
                                              fnMaskParams,
                                              fnOffsetParams,
                                              fnLyotCalib)
        for xOffsetPupilBad in (1j, (10,), np.ones((10, 10, 10)), 'str'):
            with self.assertRaises(TypeError):
                pg.fit_lyot_stop_mag_clocking(imageUnmasked,
                                              imageMasked,
                                              xOffsetPupilBad,
                                              yOffsetPupil,
                                              fnMaskParams,
                                              fnOffsetParams,
                                              fnLyotCalib)
        for yOffsetPupilBad in (1j, (10,), np.ones((10, 10, 10)), 'str'):
            with self.assertRaises(TypeError):
                pg.fit_lyot_stop_mag_clocking(imageUnmasked,
                                              imageMasked,
                                              xOffsetPupil,
                                              yOffsetPupilBad,
                                              fnMaskParams,
                                              fnOffsetParams,
                                              fnLyotCalib)
        for bad in (1j, (10,), np.ones((10, 10, 10)), None):
            with self.assertRaises(TypeError):
                pg.fit_lyot_stop_mag_clocking(imageUnmasked,
                                              imageMasked,
                                              xOffsetPupil,
                                              yOffsetPupil,
                                              bad,
                                              fnOffsetParams,
                                              fnLyotCalib)
        for bad in (1j, (10,), np.ones((10, 10, 10)), None):
            with self.assertRaises(TypeError):
                pg.fit_lyot_stop_mag_clocking(imageUnmasked,
                                              imageMasked,
                                              xOffsetPupil,
                                              yOffsetPupil,
                                              fnMaskParams,
                                              bad,
                                              fnLyotCalib)
        for bad in (1j, (10,), np.ones((10, 10, 10)), None):
            with self.assertRaises(TypeError):
                pg.fit_lyot_stop_mag_clocking(imageUnmasked,
                                              imageMasked,
                                              xOffsetPupil,
                                              yOffsetPupil,
                                              fnMaskParams,
                                              fnOffsetParams,
                                              bad)
        for bad in (1j, (10,), np.ones((10, 10, 10)), None):
            with self.assertRaises(TypeError):
                pg.fit_lyot_stop_mag_clocking(imageUnmasked,
                                              imageMasked,
                                              xOffsetPupil,
                                              yOffsetPupil,
                                              fnMaskParams,
                                              fnOffsetParams,
                                              fnLyotCalib,
                                              data_path=bad)


if __name__ == '__main__':
    unittest.main()
