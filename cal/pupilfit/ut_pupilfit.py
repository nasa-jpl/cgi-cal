# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Test suite for PUPILFIT."""
import unittest
import os

import numpy as np
from astropy.io import fits

import cal.pupilfit.pupilfit as pupilfit

not_2d_list = [-1, 1.5, 1j, (1, ), np.ones((2, 3, 4)), 'asdf']


class TestShapedPupilInputFailure(unittest.TestCase):
    """Test suite for valid function inputs."""

    def test_fit_shaped_pupil_mask_offsets_inputs(self):
        """Test the inputs of fit_shaped_pupil_mask_offsets."""
        localpath = os.path.dirname(os.path.abspath(__file__))

        unmaskfn = os.path.join(localpath, 'testdata',
                                'imageUnmasked_SPCSPEC_SPM2PUPIL.fits')
        imageUnmasked = fits.getdata(unmaskfn)
        maskfn = os.path.join(localpath, 'testdata',
                              'imageMasked_SPCSPEC_SPM2PUPIL.fits')
        imageMasked = fits.getdata(maskfn)

        fnMaskParams = os.path.join(
            localpath, 'testdata',
            'ut_high_res_pupil_mask_info_SPCSPEC_SPM.yaml',
        )
        fnTuningParams = os.path.join(localpath, 'testdata',
                                      'fit_pupil_mask_offsets.yaml')

        offsetGain = 1.
        diamPupil = 386.0
        xOffsetPupil = 50.80
        yOffsetPupil = -21.37
        magAtSPAM = 1.01
        clockDegSPM = 0.4
        # magAtLSAM = 0.98
        # clockDegLyot = 0.1

        magAtMask = magAtSPAM
        clockDegMask = clockDegSPM

        xOffsetSPM = 0.0
        yOffsetSPM = 0.0

        # Check standard inputs do not raise anything first
        pupilfit.fit_shaped_pupil_mask_offsets(imageUnmasked,
                                               imageMasked,
                                               offsetGain,
                                               fnMaskParams,
                                               fnTuningParams,
                                               xOffsetPupil,
                                               yOffsetPupil,
                                               diamPupil,
                                               magAtMask,
                                               clockDegMask,
                                               xOffsetSPM,
                                               yOffsetSPM,
                                               )

        with self.assertRaises(TypeError):
            pupilfit.fit_shaped_pupil_mask_offsets(np.ones((5,)),
                                                   imageMasked,
                                                   offsetGain,
                                                   fnMaskParams,
                                                   fnTuningParams,
                                                   xOffsetPupil,
                                                   yOffsetPupil,
                                                   diamPupil,
                                                   magAtMask,
                                                   clockDegMask,
                                                   xOffsetSPM,
                                                   yOffsetSPM,
                                                   )

        with self.assertRaises(TypeError):
            pupilfit.fit_shaped_pupil_mask_offsets(imageUnmasked,
                                                   np.ones((5,)),
                                                   offsetGain,
                                                   fnMaskParams,
                                                   fnTuningParams,
                                                   xOffsetPupil,
                                                   yOffsetPupil,
                                                   diamPupil,
                                                   magAtMask,
                                                   clockDegMask,
                                                   xOffsetSPM,
                                                   yOffsetSPM,
                                                   )
        with self.assertRaises(TypeError):
            pupilfit.fit_shaped_pupil_mask_offsets(imageUnmasked,
                                                   imageMasked,
                                                   -0.5,
                                                   fnMaskParams,
                                                   fnTuningParams,
                                                   xOffsetPupil,
                                                   yOffsetPupil,
                                                   diamPupil,
                                                   magAtMask,
                                                   clockDegMask,
                                                   xOffsetSPM,
                                                   yOffsetSPM,
                                                   )

        with self.assertRaises(TypeError):
            pupilfit.fit_shaped_pupil_mask_offsets(imageUnmasked,
                                                   imageMasked,
                                                   offsetGain,
                                                   None,
                                                   fnTuningParams,
                                                   xOffsetPupil,
                                                   yOffsetPupil,
                                                   diamPupil,
                                                   magAtMask,
                                                   clockDegMask,
                                                   xOffsetSPM,
                                                   yOffsetSPM,
                                                   )

        with self.assertRaises(TypeError):
            pupilfit.fit_shaped_pupil_mask_offsets(imageUnmasked,
                                                   imageMasked,
                                                   offsetGain,
                                                   fnMaskParams,
                                                   None,
                                                   xOffsetPupil,
                                                   yOffsetPupil,
                                                   diamPupil,
                                                   magAtMask,
                                                   clockDegMask,
                                                   xOffsetSPM,
                                                   yOffsetSPM,
                                                   )


        with self.assertRaises(TypeError):
            pupilfit.fit_shaped_pupil_mask_offsets(imageUnmasked,
                                                   imageMasked,
                                                   offsetGain,
                                                   fnMaskParams,
                                                   fnTuningParams,
                                                   [1],
                                                   yOffsetPupil,
                                                   diamPupil,
                                                   magAtMask,
                                                   clockDegMask,
                                                   xOffsetSPM,
                                                   yOffsetSPM,
                                                   )
        with self.assertRaises(TypeError):
            pupilfit.fit_shaped_pupil_mask_offsets(imageUnmasked,
                                                   imageMasked,
                                                   offsetGain,
                                                   fnMaskParams,
                                                   fnTuningParams,
                                                   xOffsetPupil,
                                                   1j,
                                                   diamPupil,
                                                   magAtMask,
                                                   clockDegMask,
                                                   xOffsetSPM,
                                                   yOffsetSPM,
                                                   )
        with self.assertRaises(TypeError):
            pupilfit.fit_shaped_pupil_mask_offsets(imageUnmasked,
                                                   imageMasked,
                                                   offsetGain,
                                                   fnMaskParams,
                                                   fnTuningParams,
                                                   xOffsetPupil,
                                                   yOffsetPupil,
                                                   -100,
                                                   magAtMask,
                                                   clockDegMask,
                                                   xOffsetSPM,
                                                   yOffsetSPM,
                                                   )

        with self.assertRaises(TypeError):
            pupilfit.fit_shaped_pupil_mask_offsets(imageUnmasked,
                                                   imageMasked,
                                                   offsetGain,
                                                   fnMaskParams,
                                                   fnTuningParams,
                                                   xOffsetPupil,
                                                   yOffsetPupil,
                                                   diamPupil,
                                                   -1,
                                                   clockDegMask,
                                                   xOffsetSPM,
                                                   yOffsetSPM,
                                                   )
        with self.assertRaises(TypeError):
            pupilfit.fit_shaped_pupil_mask_offsets(imageUnmasked,
                                                   imageMasked,
                                                   offsetGain,
                                                   fnMaskParams,
                                                   fnTuningParams,
                                                   xOffsetPupil,
                                                   yOffsetPupil,
                                                   diamPupil,
                                                   magAtMask,
                                                   1j,
                                                   xOffsetSPM,
                                                   yOffsetSPM,
                                                   )
        with self.assertRaises(TypeError):
            pupilfit.fit_shaped_pupil_mask_offsets(imageUnmasked,
                                                   imageMasked,
                                                   offsetGain,
                                                   fnMaskParams,
                                                   fnTuningParams,
                                                   xOffsetPupil,
                                                   yOffsetPupil,
                                                   diamPupil,
                                                   magAtMask,
                                                   clockDegMask,
                                                   1j,
                                                   yOffsetSPM,
                                                   )
        with self.assertRaises(TypeError):
            pupilfit.fit_shaped_pupil_mask_offsets(imageUnmasked,
                                                   imageMasked,
                                                   offsetGain,
                                                   fnMaskParams,
                                                   fnTuningParams,
                                                   xOffsetPupil,
                                                   yOffsetPupil,
                                                   diamPupil,
                                                   magAtMask,
                                                   clockDegMask,
                                                   xOffsetSPM,
                                                   [10],
                                                   )

        with self.assertRaises(TypeError):
            pupilfit.fit_shaped_pupil_mask_offsets(imageUnmasked,
                                                   imageMasked,
                                                   offsetGain,
                                                   fnMaskParams,
                                                   fnTuningParams,
                                                   xOffsetPupil,
                                                   yOffsetPupil,
                                                   diamPupil,
                                                   magAtMask,
                                                   clockDegMask,
                                                   xOffsetSPM,
                                                   yOffsetSPM,
                                                   data_path=None,
                                                   )




class TestPupilfitInputFailure(unittest.TestCase):
    """Test suite for valid function inputs."""

    def test_fit_pupil_mask_offsets_inputs(self):
        """Test the inputs of fit_pupil_mask_offsets."""
        localpath = os.path.dirname(os.path.abspath(__file__))

        case = 'SPCSPEC_LS2PUPIL'
        unmaskfn = os.path.join(localpath, 'testdata',
                                'imageUnmasked_%s.fits' % (case))
        imageUnmasked = fits.getdata(unmaskfn)
        maskfn = os.path.join(localpath, 'testdata',
                              'imageMasked_%s.fits' % (case))
        imageMasked = fits.getdata(maskfn)

        fnMaskParams = os.path.join(
            localpath, 'testdata',
            'ut_high_res_pupil_mask_info_SPCSPEC_LS.yaml',
        )
        fnTuningParams = os.path.join(localpath, 'testdata',
                                      'fit_pupil_mask_offsets.yaml')

        offsetGain = 1.
        diamPupil = 386.0
        xOffsetPupil = 50.80
        yOffsetPupil = -21.37
        # magAtSPAM = 1.01
        # clockDegSPM = 0.4
        magAtLSAM = 0.98
        clockDegLyot = 0.1

        magAtMask = magAtLSAM
        clockDegMask = clockDegLyot

        # Check standard inputs do not raise anything first
        pupilfit.fit_pupil_mask_offsets(imageUnmasked,
                                        imageMasked,
                                        offsetGain,
                                        fnMaskParams,
                                        fnTuningParams,
                                        xOffsetPupil,
                                        yOffsetPupil,
                                        diamPupil,
                                        magAtMask,
                                        clockDegMask,
                                        )

        with self.assertRaises(TypeError):
            pupilfit.fit_pupil_mask_offsets(np.ones((5,)),
                                            imageMasked,
                                            offsetGain,
                                            fnMaskParams,
                                            fnTuningParams,
                                            xOffsetPupil,
                                            yOffsetPupil,
                                            diamPupil,
                                            magAtMask,
                                            clockDegMask,
                                            )
        with self.assertRaises(TypeError):
            pupilfit.fit_pupil_mask_offsets(imageUnmasked,
                                            np.ones((5,)),
                                            offsetGain,
                                            fnMaskParams,
                                            fnTuningParams,
                                            xOffsetPupil,
                                            yOffsetPupil,
                                            diamPupil,
                                            magAtMask,
                                            clockDegMask,
                                            )
        with self.assertRaises(ValueError):
            pupilfit.fit_pupil_mask_offsets(imageUnmasked,
                                            np.ones((5, 5)),
                                            offsetGain,
                                            fnMaskParams,
                                            fnTuningParams,
                                            xOffsetPupil,
                                            yOffsetPupil,
                                            diamPupil,
                                            magAtMask,
                                            clockDegMask,
                                            )
        with self.assertRaises(TypeError):
            pupilfit.fit_pupil_mask_offsets(imageUnmasked,
                                            imageMasked,
                                            -0.5,
                                            fnMaskParams,
                                            fnTuningParams,
                                            xOffsetPupil,
                                            yOffsetPupil,
                                            diamPupil,
                                            magAtMask,
                                            clockDegMask,
                                            )

        with self.assertRaises(TypeError):
            pupilfit.fit_pupil_mask_offsets(imageUnmasked,
                                            imageMasked,
                                            offsetGain,
                                            None,
                                            fnTuningParams,
                                            xOffsetPupil,
                                            yOffsetPupil,
                                            diamPupil,
                                            magAtMask,
                                            clockDegMask,
                                            )
        with self.assertRaises(TypeError):
            pupilfit.fit_pupil_mask_offsets(imageUnmasked,
                                            imageMasked,
                                            offsetGain,
                                            fnMaskParams,
                                            None,
                                            xOffsetPupil,
                                            yOffsetPupil,
                                            diamPupil,
                                            magAtMask,
                                            clockDegMask,
                                            )

        with self.assertRaises(TypeError):
            pupilfit.fit_pupil_mask_offsets(imageUnmasked,
                                            imageMasked,
                                            offsetGain,
                                            fnMaskParams,
                                            fnTuningParams,
                                            [1],
                                            yOffsetPupil,
                                            diamPupil,
                                            magAtMask,
                                            clockDegMask,
                                            )
        with self.assertRaises(TypeError):
            pupilfit.fit_pupil_mask_offsets(imageUnmasked,
                                            imageMasked,
                                            offsetGain,
                                            fnMaskParams,
                                            fnTuningParams,
                                            xOffsetPupil,
                                            1j,
                                            diamPupil,
                                            magAtMask,
                                            clockDegMask,
                                            )
        with self.assertRaises(TypeError):
            pupilfit.fit_pupil_mask_offsets(imageUnmasked,
                                            imageMasked,
                                            offsetGain,
                                            fnMaskParams,
                                            fnTuningParams,
                                            xOffsetPupil,
                                            yOffsetPupil,
                                            -100,
                                            magAtMask,
                                            clockDegMask,
                                            )
        with self.assertRaises(TypeError):
            pupilfit.fit_pupil_mask_offsets(imageUnmasked,
                                            imageMasked,
                                            offsetGain,
                                            fnMaskParams,
                                            fnTuningParams,
                                            xOffsetPupil,
                                            yOffsetPupil,
                                            diamPupil,
                                            -1,
                                            clockDegMask,
                                            )
        with self.assertRaises(TypeError):
            pupilfit.fit_pupil_mask_offsets(imageUnmasked,
                                            imageMasked,
                                            offsetGain,
                                            fnMaskParams,
                                            fnTuningParams,
                                            xOffsetPupil,
                                            yOffsetPupil,
                                            diamPupil,
                                            magAtMask,
                                            [2.5],
                                            )

        with self.assertRaises(TypeError):
            pupilfit.fit_pupil_mask_offsets(imageUnmasked,
                                            imageMasked,
                                            offsetGain,
                                            fnMaskParams,
                                            fnTuningParams,
                                            xOffsetPupil,
                                            yOffsetPupil,
                                            diamPupil,
                                            magAtMask,
                                            clockDegMask,
                                            data_path=None
                                            )



class TestShapedPupilLateralOffsets(unittest.TestCase):
    """Integration tests."""

    def test_fit_shaped_pupil_mask_offsets(self):
        """Integration test of fit_shaped_pupil_mask_offsets."""
        localpath = os.path.dirname(os.path.abspath(__file__))

        xOffsetSPMatSPAM = 5.5
        yOffsetSPMatSPAM = -6.8
        xOffsetPupil = -25
        yOffsetPupil = 40
        magAtMask = 1
        clockDegMask = 0
        offsetGain = 1
        fnTuningParams = os.path.join(localpath, 'testdata',
                                      'fit_pupil_mask_offsets_PhaseC.yaml')

        for case in ['SPCWFOV', 'SPCSPEC']:

            # convert from pixels at pupil mask plane to ExCAM pixels
            diamPupil = 300
            pupilFactor = diamPupil/1000
            xOffsetTrue = 7*pupilFactor
            yOffsetTrue = -7*pupilFactor

            xOffsetSPMatEXCAM = xOffsetSPMatSPAM*pupilFactor
            yOffsetSPMatEXCAM = yOffsetSPMatSPAM*pupilFactor

            fnMaskParams = os.path.join(
                localpath, 'testdata',
                'ut_high_res_pupil_mask_info_%s_SPM_PhaseC.yaml' % (case))

            fnImageUnmasked = os.path.join(
                localpath, 'testdata',
                'ut_%s_image_unmasked.fits' % (case),
            )
            imageUnmasked = fits.getdata(fnImageUnmasked)

            fnImageMasked = os.path.join(
                localpath, 'testdata',
                'ut_%s_image_masked_x%.3f_y%.3f.fits' %
                (case, xOffsetTrue, yOffsetTrue),
            )
            imageMasked = fits.getdata(fnImageMasked)

            xOffsetEst, yOffsetEst = \
                pupilfit.fit_shaped_pupil_mask_offsets(imageUnmasked,
                                                       imageMasked,
                                                       offsetGain,
                                                       fnMaskParams,
                                                       fnTuningParams,
                                                       xOffsetPupil,
                                                       yOffsetPupil,
                                                       diamPupil,
                                                       magAtMask,
                                                       clockDegMask,
                                                       xOffsetSPMatEXCAM,
                                                       yOffsetSPMatEXCAM,
                                                       )

            self.assertTrue(np.abs(xOffsetEst-xOffsetTrue) < 0.5)
            self.assertTrue(np.abs(yOffsetEst-yOffsetTrue) < 0.5)


class TestPupilfit(unittest.TestCase):
    """Unit and integration tests."""

    def test_fit_pupil_mask_offsets(self):
        """Integration test of fit_pupil_mask_offsets."""
        localpath = os.path.dirname(os.path.abspath(__file__))

        xOffsetMask = 4.90
        yOffsetMask = -3.72
        fnTuningParams = os.path.join(localpath, 'testdata',
                                      'fit_pupil_mask_offsets.yaml')
        xOffsetPupil = 50.80
        yOffsetPupil = -21.37
        diamPupil = 386.0

        magAtSPAM = 1.01
        clockDegSPM = 0.4
        magAtLSAM = 0.98
        clockDegLyot = 0.1

        valid_cases = [
            'SPCSPEC_SPM2PUPIL',
            'SPCWFOV_SPM2PUPIL',
            'HLCNFOV_LS2PUPIL',
            'SPCSPEC_LS2SPM',
            'SPCSPEC_LS2PUPIL',
            'SPCWFOV_LS2SPM',
        ]

        for case in valid_cases:

            if case in 'SPCWFOV_LS2SPM':
                offsetGain = 3
            else:
                offsetGain = 1

            if case in ('SPCSPEC_SPM2PUPIL', 'SPCWFOV_SPM2PUPIL'):
                magAtMask = magAtSPAM
                clockDegMask = clockDegSPM
            else:
                magAtMask = magAtLSAM
                clockDegMask = clockDegLyot

            if case == 'SPCSPEC_SPM2PUPIL':
                fnMaskParams = os.path.join(
                    localpath, 'testdata',
                    'ut_high_res_pupil_mask_info_SPCSPEC_SPM.yaml')

            elif case == 'SPCWFOV_SPM2PUPIL':
                fnMaskParams = os.path.join(
                    localpath, 'testdata',
                    'ut_high_res_pupil_mask_info_SPCWFOV_SPM.yaml')

            elif case == 'HLCNFOV_LS2PUPIL':
                fnMaskParams = os.path.join(
                    localpath, 'testdata',
                    'ut_high_res_pupil_mask_info_HLCNFOV_LS.yaml')

            elif case in ('SPCSPEC_LS2SPM', 'SPCSPEC_LS2PUPIL'):
                fnMaskParams = os.path.join(
                    localpath, 'testdata',
                    'ut_high_res_pupil_mask_info_SPCSPEC_LS.yaml')

            elif case in ('SPCWFOV_LS2SPM', 'SPCWFOV_LS2PUPIL'):
                fnMaskParams = os.path.join(
                    localpath, 'testdata',
                    'ut_high_res_pupil_mask_info_SPCWFOV_LS.yaml')

            unmaskfn = os.path.join(localpath, 'testdata',
                                    'imageUnmasked_%s.fits' % (case))
            imageUnmasked = fits.getdata(unmaskfn)
            maskfn = os.path.join(localpath, 'testdata',
                                  'imageMasked_%s.fits' % (case))
            imageMasked = fits.getdata(maskfn)

            xOffsetEst, yOffsetEst = \
                pupilfit.fit_pupil_mask_offsets(imageUnmasked,
                                                imageMasked,
                                                offsetGain,
                                                fnMaskParams,
                                                fnTuningParams,
                                                xOffsetPupil,
                                                yOffsetPupil,
                                                diamPupil,
                                                magAtMask,
                                                clockDegMask,
                                                )
            self.assertTrue(np.abs(yOffsetEst-yOffsetMask) < 0.2)
            self.assertTrue(np.abs(xOffsetEst-xOffsetMask) < 0.2)
        pass


class TestShapedPupilBeamShear(unittest.TestCase):
    """Unit tests for fit_z_induced_spm_shear."""

    def test_fit_z_induced_spm_shear(self):
        """Unit test of fit_z_induced_spm_shear performance."""
        localpath = os.path.dirname(os.path.abspath(__file__))

        fnConfig = os.path.join(localpath, 'testdata',
                                'ut_psffit_parms_spm_shear.yaml')

        # Data used to generate the FITS file inputs
        xOffsetTrueAct = -0.4
        yOffsetTrueAct = 0.0
        diamPupilMeters = 46.3e-3  # meters
        actPitch = 0.9906e-3  # meters
        aoiDeg = 9.65  # degrees

        for bandName in ('band3', 'band4'):

            if bandName == 'band3':
                diamPupilPixels = 302.3  # pixels
            elif bandName == 'band4':
                diamPupilPixels = 301.3  # pixels
            else:
                raise ValueError('Invalid value for bandName.')

            pixPerAct = diamPupilPixels/(diamPupilMeters /
                                         actPitch*np.cos(np.radians(aoiDeg)))
            xOffsetTrue = xOffsetTrueAct * pixPerAct
            yOffsetTrue = yOffsetTrueAct * pixPerAct

            prefix = ('ut_spm_shear_' + bandName + '_')
            ampUnpoked0 = fits.getdata(os.path.join(localpath, 'testdata',
                                                    prefix+'ampUnpoked0.fits'))
            ampUnpoked1 = fits.getdata(os.path.join(localpath, 'testdata',
                                                    prefix+'ampUnpoked1.fits'))
            phUnpoked0 = fits.getdata(os.path.join(localpath, 'testdata',
                                      prefix+'phUnpoked0.fits'))
            phUnpoked1 = fits.getdata(os.path.join(localpath, 'testdata',
                                      prefix+'phUnpoked1.fits'))
            phPoked0 = fits.getdata(os.path.join(localpath, 'testdata',
                                                 prefix+'phPoked0.fits'))
            phPoked1 = fits.getdata(os.path.join(localpath, 'testdata',
                                                 prefix+'phPoked1.fits'))

            xOffsetEst, yOffsetEst = pupilfit.fit_z_induced_spm_shear(
                ampUnpoked0, phUnpoked0, phPoked0,
                ampUnpoked1, phUnpoked1, phPoked1,
                fnConfig,
            )

            abs_tol = 0.1  # pixels
            self.assertTrue(np.abs(xOffsetTrue - xOffsetEst) < abs_tol)
            self.assertTrue(np.abs(yOffsetTrue - yOffsetEst) < abs_tol)


class TestShapedPupilShearInputFailure(unittest.TestCase):
    """Test suite for valid function inputs."""

    def setUp(self):
        """Test the inputs of fit_z_induced_spm_shear."""
        localpath = os.path.dirname(os.path.abspath(__file__))
        self.fnConfig = os.path.join(localpath, 'testdata',
                                     'ut_psffit_parms_spm_shear.yaml')

        self.badArray = np.zeros((10, 2))  # bad input with different shape

        bandName = 'band3'
        prefix = ('ut_spm_shear_' + bandName + '_')
        self.ampUnpoked0 = fits.getdata(os.path.join(
            localpath, 'testdata', prefix+'ampUnpoked0.fits'))
        self.ampUnpoked1 = fits.getdata(os.path.join(
            localpath, 'testdata', prefix+'ampUnpoked1.fits'))
        self.phUnpoked0 = fits.getdata(os.path.join(
            localpath, 'testdata', prefix+'phUnpoked0.fits'))
        self.phUnpoked1 = fits.getdata(os.path.join(
            localpath, 'testdata', prefix+'phUnpoked1.fits'))
        self.phPoked0 = fits.getdata(os.path.join(
            localpath, 'testdata', prefix+'phPoked0.fits'))
        self.phPoked1 = fits.getdata(os.path.join(localpath, 'testdata',
                                                  prefix+'phPoked1.fits'))

    def test_runs_without_error(self):
        """Make sure function runs without error before giving bad inputs."""
        xOffsetEst, yOffsetEst = pupilfit.fit_z_induced_spm_shear(
            self.ampUnpoked0, self.phUnpoked0, self.phPoked0,
            self.ampUnpoked1, self.phUnpoked1, self.phPoked1,
            self.fnConfig,
        )

    def test_bad_input_type_0(self):
        """Test with bad input type."""
        for badVal in not_2d_list:
            with self.assertRaises(TypeError):
                pupilfit.fit_z_induced_spm_shear(
                    badVal, self.phUnpoked0, self.phPoked0,
                    self.ampUnpoked1, self.phUnpoked1, self.phPoked1,
                    self.fnConfig,
                )

    def test_bad_input_type_1(self):
        """Test with bad input type."""
        for badVal in not_2d_list:
            with self.assertRaises(TypeError):
                pupilfit.fit_z_induced_spm_shear(
                    self.ampUnpoked0, badVal, self.phPoked0,
                    self.ampUnpoked1, self.phUnpoked1, self.phPoked1,
                    self.fnConfig,
                )

    def test_bad_input_type_2(self):
        """Test with bad input type."""
        for badVal in not_2d_list:
            with self.assertRaises(TypeError):
                pupilfit.fit_z_induced_spm_shear(
                    self.ampUnpoked0, self.phUnpoked0, badVal,
                    self.ampUnpoked1, self.phUnpoked1, self.phPoked1,
                    self.fnConfig,
                )

    def test_bad_input_type_3(self):
        """Test with bad input type."""
        for badVal in not_2d_list:
            with self.assertRaises(TypeError):
                pupilfit.fit_z_induced_spm_shear(
                    self.ampUnpoked0, self.phUnpoked0, self.phPoked0,
                    badVal, self.phUnpoked1, self.phPoked1,
                    self.fnConfig,
                )

    def test_bad_input_type_4(self):
        """Test with bad input type."""
        for badVal in not_2d_list:
            with self.assertRaises(TypeError):
                pupilfit.fit_z_induced_spm_shear(
                    self.ampUnpoked0, self.phUnpoked0, self.phPoked0,
                    self.ampUnpoked1, badVal, self.phPoked1,
                    self.fnConfig,
                )

    def test_bad_input_type_5(self):
        """Test with bad input type."""
        for badVal in not_2d_list:
            with self.assertRaises(TypeError):
                pupilfit.fit_z_induced_spm_shear(
                    self.ampUnpoked0, self.phUnpoked0, self.phPoked0,
                    self.ampUnpoked1, self.phUnpoked1, badVal,
                    self.fnConfig,
                )

    def test_bad_input_type_6(self):
        """Test with bad input type."""
        for badVal in [1, 1j, None, np.ones((3,))]:
            with self.assertRaises(TypeError):
                pupilfit.fit_z_induced_spm_shear(
                    self.ampUnpoked0, self.phUnpoked0, self.phPoked0,
                    self.ampUnpoked1, self.phUnpoked1, self.phPoked1,
                    badVal,
                )


    def test_bad_input_shape_0(self):
        """Test with bad input shape."""
        with self.assertRaises(ValueError):
            pupilfit.fit_z_induced_spm_shear(
                self.badArray, self.phUnpoked0, self.phPoked0,
                self.ampUnpoked1, self.phUnpoked1, self.phPoked1,
                self.fnConfig,
            )

    def test_bad_input_shape_1(self):
        """Test with bad input shape."""
        with self.assertRaises(ValueError):
            pupilfit.fit_z_induced_spm_shear(
                self.ampUnpoked0, self.badArray, self.phPoked0,
                self.ampUnpoked1, self.phUnpoked1, self.phPoked1,
                self.fnConfig,
            )

    def test_bad_input_shape_2(self):
        """Test with bad input shape."""
        with self.assertRaises(ValueError):
            pupilfit.fit_z_induced_spm_shear(
                self.ampUnpoked0, self.phUnpoked0, self.badArray,
                self.ampUnpoked1, self.phUnpoked1, self.phPoked1,
                self.fnConfig,
            )

    def test_bad_input_shape_3(self):
        """Test with bad input shape."""
        with self.assertRaises(ValueError):
            pupilfit.fit_z_induced_spm_shear(
                self.ampUnpoked0, self.phUnpoked0, self.phPoked0,
                self.badArray, self.phUnpoked1, self.phPoked1,
                self.fnConfig,
            )

    def test_bad_input_shape_4(self):
        """Test with bad input shape."""
        with self.assertRaises(ValueError):
            pupilfit.fit_z_induced_spm_shear(
                self.ampUnpoked0, self.phUnpoked0, self.phPoked0,
                self.ampUnpoked1, self.badArray, self.phPoked1,
                self.fnConfig,
            )

    def test_bad_input_shape_5(self):
        """Test with bad input shape."""
        with self.assertRaises(ValueError):
            pupilfit.fit_z_induced_spm_shear(
                self.ampUnpoked0, self.phUnpoked0, self.phPoked0,
                self.ampUnpoked1, self.phUnpoked1, self.badArray,
                self.fnConfig,
            )


if __name__ == '__main__':
    unittest.main()
