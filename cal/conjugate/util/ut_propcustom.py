# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Unit tests for propcustom."""
import unittest
from math import isclose
import numpy as np

from cal.util.nollzernikes import xyzern
from cal.util.insertinto import insertinto
from cal.conjugate.util.propcustom import gen_zernikes, fit_zernikes


class TestPropcustom(unittest.TestCase):
    """Test all the methods of conjugate."""

    # Sample assertTrue test
#    def test_DESCRIPTIVE_NAME_HERE(self):
#        self.assertTrue(1 == 1)
#        pass
#
#
#    # Sample assertRaises test
#    def test_DESCRIPTIVE_NAME_HERE_ALSO(self):
#        with self.assertRaises(ZeroDivisionError):
#            x = 1/0
#            pass
#        pass

    def test_gen_zernikes(self):
        """Test gen_zernikes."""
        nBeam = 100
        nArray = 110
        xOffset = 0
        yOffset = 0
        zernIndVec = np.array([1, 3, 5])
        zernCoefVec = np.array([0.5, -2.0, 3.1])

        x = (np.arange(nArray, dtype=np.float64) - nArray//2 - xOffset)
        y = (np.arange(nArray, dtype=np.float64) - nArray//2 - yOffset)
        xx, yy = np.meshgrid(x, y)

        # Create expected map
        zernCube = xyzern(xx, yy, nBeam/2., zernIndVec)
        zernAbMap0 = np.zeros([nArray, nArray], dtype=np.float64)
        for ii in range(zernIndVec.size):
            zernAbMap0 += zernCoefVec[ii]*np.squeeze(zernCube[ii, :, :])

        zernAbMap1 = gen_zernikes(zernIndVec, zernCoefVec, xOffset,
                                         yOffset, nBeam, nArray=nArray)

        diffSum = np.sum(np.abs(zernAbMap0-zernAbMap1))
        self.assertTrue(isclose(diffSum, 0, abs_tol=1e-7))
        pass

    def test_offset_zernikes(self):
        """Test that Zernikes translate as expected by comparing to np.roll."""
        nBeam = 100
        nArray = 200
        xOffset = -10
        yOffset = 30

        # Compute shifted Zernike and unshifted one for reference
        wfeZ4 = gen_zernikes(np.array([4]), np.array([1]), 0, 0, nBeam,
                                    nArray=nArray)
        wfeZ4shift = gen_zernikes(np.array([4]), np.array([1]), xOffset,
                                         yOffset, nBeam, nArray=nArray)

        # Roll the shifted one, crop both, and see if the difference is zero
        wfeZ4shiftRoll = np.roll(wfeZ4shift, (-yOffset, -xOffset), axis=(0, 1))
        nMax = int(nArray - 2*np.max(np.abs((yOffset, xOffset))))
        wfeZ4crop = insertinto(wfeZ4, (nMax, nMax))
        wfeZ4shiftRollCrop = insertinto(wfeZ4shiftRoll, (nMax, nMax))
        diffSum = np.sum(np.abs(wfeZ4crop-wfeZ4shiftRollCrop))

        self.assertTrue(isclose(diffSum, 0, abs_tol=1e-7))

    def test_fit_zernikes(self):
        """Test that known Zernike coefficients are returned from the fit."""
        # Generate a shifted aberration map of only Zernike modes
        nBeam = 100
        nArray = 120

        xOffset = -15.55
        yOffset = 30.1

        maxNollZern = 11
        zernCoefIn = 20*np.random.rand(maxNollZern)

        # Generate a ROI mask
        x = (np.arange(-nArray/2., nArray/2.) - xOffset)/nBeam
        y = (np.arange(-nArray/2., nArray/2.) - yOffset)/nBeam
        [X, Y] = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        mask = np.zeros((nArray, nArray))
        mask[R <= 0.5] = 1

        wfeShift = gen_zernikes(np.arange(1, maxNollZern+1), zernCoefIn,
                                       xOffset, yOffset, nBeam, nArray=nArray)
        zernCoefOut = fit_zernikes(wfeShift, mask, maxNollZern,
                                              nBeam, xOffset, yOffset)
        diffSum = np.sum(np.abs(zernCoefIn - zernCoefOut))

        self.assertTrue(isclose(diffSum, 0, abs_tol=1e-10))

    # def test_propcustom_fit_dm(self):
    #     """Test that."""
    #     pass


class TestPropcustomInputFailure(unittest.TestCase):
    """Test suite for valid function inputs."""

    def test_gen_zernikes_inputs(self):
        """Test inputs for gen_zernikes."""
        nBeam = 100
        nArray = 110
        xOffset = 0
        yOffset = 0
        zernIndVec = np.array([1, 3, 5])
        zernCoefVec = np.array([0.5, -2.0, 3.1])

        x = (np.arange(nArray, dtype=np.float64) - nArray//2 - xOffset)
        y = (np.arange(nArray, dtype=np.float64) - nArray//2 - yOffset)
        xx, yy = np.meshgrid(x, y)

        # Make sure it runs normally first.
        gen_zernikes(zernIndVec, zernCoefVec, xOffset,
                                         yOffset, nBeam, nArray=nArray)

        for zernIndVecBad in (1, np.ones((5, 10)), 'asdf'):
            with self.assertRaises(ValueError):
                gen_zernikes(zernIndVecBad, zernCoefVec, xOffset,
                                         yOffset, nBeam, nArray=nArray)

        for zernCoefVecBad in (1, np.ones((5, 10)), 'asdf'):
            with self.assertRaises(ValueError):
                gen_zernikes(zernIndVec, zernCoefVecBad, xOffset,
                                         yOffset, nBeam, nArray=nArray)

        zernIndVecBad = np.array([1, 3, 5])
        zernCoefVecBad = np.array([0.5, -2.0, 3.1, 1.1])
        with self.assertRaises(ValueError):
            gen_zernikes(zernIndVecBad, zernCoefVecBad, xOffset,
                                     yOffset, nBeam, nArray=nArray)

        for xOffsetBad in (1j, np.ones((5, 10)), 'asdf'):
            with self.assertRaises(ValueError):
                gen_zernikes(zernIndVec, zernCoefVec, xOffsetBad,
                                         yOffset, nBeam, nArray=nArray)

        for yOffsetBad in (1j, np.ones((5, 10)), 'asdf'):
            with self.assertRaises(ValueError):
                gen_zernikes(zernIndVec, zernCoefVec, xOffset,
                                         yOffsetBad, nBeam, nArray=nArray)

        for nBeamBad in (-100, 1j, np.ones((5, 10)), 'asdf'):
            with self.assertRaises(ValueError):
                gen_zernikes(zernIndVec, zernCoefVec, xOffset,
                                         yOffset, nBeamBad, nArray=nArray)

        for nArrayBad in (-100, 0, 1j, np.ones((5, 10)), 'asdf'):
            with self.assertRaises(ValueError):
                gen_zernikes(zernIndVec, zernCoefVec, xOffset,
                                         yOffset, nBeam, nArray=nArrayBad)
        pass

    def test_fit_zernikes_inputs(self):
        """Test inputs for fit_zernikes."""
        # Generate a shifted aberration map of only Zernike modes
        nBeam = 100
        nArray = 120

        xOffset = -15.55
        yOffset = 30.1

        maxNollZern = 11
        zernCoefIn = 20*np.random.rand(maxNollZern)

        # Generate a ROI mask
        x = (np.arange(-nArray/2., nArray/2.) - xOffset)/nBeam
        y = (np.arange(-nArray/2., nArray/2.) - yOffset)/nBeam
        [X, Y] = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        mask = np.zeros((nArray, nArray))
        mask[R <= 0.5] = 1

        wfe = gen_zernikes(np.arange(1, maxNollZern+1), zernCoefIn,
                                       xOffset, yOffset, nBeam, nArray=nArray)

        # Make sure it runs normally first.
        fit_zernikes(wfe, mask, maxNollZern, nBeam, xOffset, yOffset)

        for wfeBad in (5, 1j, np.ones((5,)), 2j*np.ones((5, 10)), 'asdf'):
            with self.assertRaises(ValueError):
                fit_zernikes(wfeBad, mask, maxNollZern, nBeam,
                             xOffset, yOffset)

        for maskBad in (5, 1j, np.ones((5,)), 2j*np.ones((5, 10)), 'asdf'):
            with self.assertRaises(ValueError):
                fit_zernikes(wfe, maskBad, maxNollZern, nBeam,
                             xOffset, yOffset)

        with self.assertRaises(ValueError):
            fit_zernikes(wfe, mask[2::, 2::], maxNollZern, nBeam,
                         xOffset, yOffset)

        for maxNollZernBad in (1j, np.ones((5,)), 2j*np.ones((5, 10)), 'asdf'):
            with self.assertRaises(ValueError):
                fit_zernikes(wfe, mask, maxNollZernBad, nBeam,
                             xOffset, yOffset)

        for nBeamBad in (-10, 1j, np.ones((5,)), np.ones((5, 10)), 'asdf'):
            with self.assertRaises(TypeError):
                fit_zernikes(wfe, mask, maxNollZern, nBeamBad,
                             xOffset, yOffset)

        for xOffsetBad in (1j, np.ones((5,)), np.ones((5, 10)), 'asdf'):
            with self.assertRaises(TypeError):
                fit_zernikes(wfe, mask, maxNollZern, nBeam,
                             xOffsetBad, yOffset)

        for yOffsetBad in (1j, np.ones((5,)), np.ones((5, 10)), 'asdf'):
            with self.assertRaises(TypeError):
                fit_zernikes(wfe, mask, maxNollZern, nBeam,
                             xOffset, yOffsetBad)
        pass

    # def test_propcustom_fit_dm_inputs(self):
    #     """Test that."""
    #     pass


if __name__ == '__main__':
    unittest.main()
