# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Test suite for PUPILFIT."""
import os
import unittest
from math import isclose
import numpy as np
from astropy.io import fits

import cal.pupilfit.open_util as open_util


class TestPupilfitInputFailure(unittest.TestCase):
    """Test suite for valid function inputs."""

    def test_compute_clocking_inputs(self):
        """Test the inputs of compute_clocking."""
        ampMeas = np.eye(100)
        diamBeamAtMask = 50.
        maskRefHighRes = np.ones((200, 200))
        diamHighRes = 100.
        xOffset = -2.5
        yOffset = 20
        clockDegMax = 4
        nClock = 11
        percentileForNorm = 63.
        deltaAmpMax = 0.7

        list_not_pos_scalar_int = [-1, 0, 1.5, 1j, 'string',  np.ones((5,))]

        # Check standard inputs do not raise anything first
        open_util.compute_clocking(ampMeas,
                                  diamBeamAtMask,
                                  maskRefHighRes,
                                  diamHighRes,
                                  xOffset,
                                  yOffset,
                                  clockDegMax,
                                  nClock,
                                  percentileForNorm,
                                  deltaAmpMax,
                                  useAmpthresh=False)

        with self.assertRaises(TypeError):
            open_util.compute_clocking(np.ones((20,)),
                                      diamBeamAtMask,
                                      maskRefHighRes,
                                      diamHighRes,
                                      xOffset,
                                      yOffset,
                                      clockDegMax,
                                      nClock,
                                      percentileForNorm,
                                      deltaAmpMax,
                                      useAmpthresh=False)
        with self.assertRaises(TypeError):
            open_util.compute_clocking(ampMeas,
                                      -10,
                                      maskRefHighRes,
                                      diamHighRes,
                                      xOffset,
                                      yOffset,
                                      clockDegMax,
                                      nClock,
                                      percentileForNorm,
                                      deltaAmpMax,
                                      useAmpthresh=False)
        with self.assertRaises(TypeError):
            open_util.compute_clocking(ampMeas,
                                      diamBeamAtMask,
                                      np.ones((30,)),
                                      diamHighRes,
                                      xOffset,
                                      yOffset,
                                      clockDegMax,
                                      nClock,
                                      percentileForNorm,
                                      deltaAmpMax,
                                      useAmpthresh=False)
        with self.assertRaises(TypeError):
            open_util.compute_clocking(ampMeas,
                                      diamBeamAtMask,
                                      maskRefHighRes,
                                      -40,
                                      xOffset,
                                      yOffset,
                                      clockDegMax,
                                      nClock,
                                      percentileForNorm,
                                      deltaAmpMax,
                                      useAmpthresh=False)
        with self.assertRaises(TypeError):
            open_util.compute_clocking(ampMeas,
                                      diamBeamAtMask,
                                      maskRefHighRes,
                                      diamHighRes,
                                      [2],
                                      yOffset,
                                      clockDegMax,
                                      nClock,
                                      percentileForNorm,
                                      deltaAmpMax,
                                      useAmpthresh=False)
        with self.assertRaises(TypeError):
            open_util.compute_clocking(ampMeas,
                                      diamBeamAtMask,
                                      maskRefHighRes,
                                      diamHighRes,
                                      xOffset,
                                      1j,
                                      clockDegMax,
                                      nClock,
                                      percentileForNorm,
                                      deltaAmpMax,
                                      useAmpthresh=False)
        with self.assertRaises(TypeError):
            open_util.compute_clocking(ampMeas,
                                      diamBeamAtMask,
                                      maskRefHighRes,
                                      diamHighRes,
                                      xOffset,
                                      yOffset,
                                      [2.3],
                                      nClock,
                                      percentileForNorm,
                                      deltaAmpMax,
                                      useAmpthresh=False)
        for badVal in list_not_pos_scalar_int:
            with self.assertRaises(TypeError):
                open_util.compute_clocking(ampMeas,
                                          diamBeamAtMask,
                                          maskRefHighRes,
                                          diamHighRes,
                                          xOffset,
                                          yOffset,
                                          clockDegMax,
                                          badVal,
                                          percentileForNorm,
                                          deltaAmpMax,
                                          useAmpthresh=False)
        with self.assertRaises(TypeError):
            open_util.compute_clocking(ampMeas,
                                      diamBeamAtMask,
                                      maskRefHighRes,
                                      diamHighRes,
                                      xOffset,
                                      yOffset,
                                      clockDegMax,
                                      nClock,
                                      'asdf',
                                      deltaAmpMax,
                                      useAmpthresh=False)
        with self.assertRaises(TypeError):
            open_util.compute_clocking(ampMeas,
                                      diamBeamAtMask,
                                      maskRefHighRes,
                                      diamHighRes,
                                      xOffset,
                                      yOffset,
                                      clockDegMax,
                                      nClock,
                                      101,
                                      deltaAmpMax,
                                      useAmpthresh=False)
        with self.assertRaises(TypeError):
            open_util.compute_clocking(ampMeas,
                                      diamBeamAtMask,
                                      maskRefHighRes,
                                      diamHighRes,
                                      xOffset,
                                      yOffset,
                                      clockDegMax,
                                      nClock,
                                      percentileForNorm,
                                      -0.1,
                                      useAmpthresh=False)
        with self.assertRaises(TypeError):
            open_util.compute_clocking(ampMeas,
                                      diamBeamAtMask,
                                      maskRefHighRes,
                                      diamHighRes,
                                      xOffset,
                                      yOffset,
                                      clockDegMax,
                                      nClock,
                                      percentileForNorm,
                                      deltaAmpMax,
                                      useAmpthresh='string')
            
        for badVal in list_not_pos_scalar_int:
            with self.assertRaises(TypeError):
                open_util.compute_clocking(ampMeas,
                                           diamBeamAtMask,
                                           maskRefHighRes,
                                           diamHighRes,
                                           xOffset,
                                           yOffset,
                                           clockDegMax,
                                           nClock,
                                           percentileForNorm,
                                           deltaAmpMax,
                                           nBin=badVal)


class TestComputeClocking(unittest.TestCase):
    """Unit and integration tests."""

    def test_compute_clocking(self):
        """Unit test of compute_clocking."""
        clockTrue = 0.5  # degrees

        localpath = os.path.dirname(os.path.abspath(__file__))

        fnMaskRefHighRes = os.path.join(localpath, 'testdata',
                                'pupil_template_D1000.00pixels.fits')
        maskRefHighRes = fits.getdata(fnMaskRefHighRes)

        fnPupil = os.path.join(localpath, 'testdata',
                            'clocking_test_pupil_N300_clk0.5_x3.5_y-10.2.fits')
        ampMeas = fits.getdata(fnPupil)

        diamBeamAtMask = 300
        diamHighRes = 1000
        xOffset = 3.5
        yOffset = -10.2
        clockDegMax = 2.5
        nClock = 7
        percentileForNorm = 50
        deltaAmpMax = 0.5
        useAmpthresh = False  # default is False

        clockEst = open_util.compute_clocking(ampMeas, diamBeamAtMask,
                        maskRefHighRes, diamHighRes, xOffset, yOffset,
                        clockDegMax, nClock, percentileForNorm,
                         deltaAmpMax, useAmpthresh=useAmpthresh)
        self.assertTrue(isclose(clockTrue, clockEst, abs_tol=0.1),
                        msg=('clocking true / est: %.3f %.3f degrees' %
                             (clockTrue, clockEst)))

        pass


if __name__ == '__main__':
    unittest.main()
