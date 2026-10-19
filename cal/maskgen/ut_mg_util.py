# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Unit tests for the MASKGEN utility functions."""
import unittest
import numpy as np
from math import isclose

from cal.maskgen.mg_util import gen_annular_segments


class TestMaskgenSupportFunctions(unittest.TestCase):
    """Test all the supporting methods of maskgen."""


class TestFocalMaskGeneration(unittest.TestCase):
    """Test that masks are generated as expected."""

    def test_number_of_segments(self):
        """Verify the 2-segment mask is composed of two 1-segment masks."""
        rInPix = 7.2
        rOutPix = 19.5
        nx = 60
        ny = 71
        xOffset = -5
        yOffset = 2
        angOpen = 65
        angRot = 15
        nsp = 100
        dTheta = 5

        nSeg = 1
        maskRight = gen_annular_segments(nSeg, rInPix, rOutPix, nx, ny,
                                        xOffset, yOffset, angOpen, angRot,
                                        nSubpixels=nsp, dTheta=dTheta)
        maskLeft = gen_annular_segments(nSeg, rInPix, rOutPix, nx, ny,
                                        xOffset, yOffset, angOpen, angRot+180,
                                        nSubpixels=nsp, dTheta=dTheta)
        nSeg = 2
        maskBoth = gen_annular_segments(nSeg, rInPix, rOutPix, nx, ny,
                                        xOffset, yOffset, angOpen, angRot,
                                        nSubpixels=nsp, dTheta=dTheta)

        sumAbsDiff = np.sum(np.abs(maskBoth - maskRight - maskLeft))
        self.assertTrue(isclose(sumAbsDiff, 0, abs_tol=1e-8))

    def test_summed_area(self):
        """Test that the summed area matches the analytical value."""
        nSeg = 1
        rInPix = 7.2
        rOutPix = 19.5
        nx = 60
        ny = 71
        xOffset = -5
        yOffset = 2
        angOpen = 65
        angRot = 15
        nsp = 100
        dTheta = 5

        mask1 = gen_annular_segments(nSeg, rInPix, rOutPix, nx, ny, xOffset,
                    yOffset, angOpen, angRot, nSubpixels=nsp, dTheta=dTheta)
        area1 = np.sum(mask1)
        nSeg = 2
        mask2 = gen_annular_segments(nSeg, rInPix, rOutPix, nx, ny, xOffset,
                    yOffset, angOpen, angRot, nSubpixels=nsp, dTheta=dTheta)
        area2 = np.sum(mask2)

        areaTrue = np.pi * (rOutPix**2 - rInPix**2) * (angOpen/360)
        self.assertTrue(isclose(areaTrue, area1, rel_tol=1e-5))
        self.assertTrue(isclose(2*areaTrue, area2, rel_tol=1e-5))

    def test_offsets(self):
        """Test the lateral offsets by comparing to np.roll."""
        nSeg = 1
        rInPix = 7.2
        rOutPix = 19.5
        nx = 60
        ny = 71
        xOffset = -5
        yOffset = 2
        angOpen = 65
        angRot = 15
        nsp = 100
        dTheta = 5

        maskOffset = gen_annular_segments(nSeg, rInPix, rOutPix, nx, ny,
            xOffset, yOffset, angOpen, angRot, nSubpixels=nsp, dTheta=dTheta)

        maskCentered = gen_annular_segments(nSeg, rInPix, rOutPix, nx, ny,
            0, 0, angOpen, angRot, nSubpixels=nsp, dTheta=dTheta)

        sumAbsDiff = np.sum(np.abs(maskCentered - np.roll(maskOffset,
                                                          (-yOffset, -xOffset),
                                                          axis=(0, 1))))

        self.assertTrue(isclose(sumAbsDiff, 0, abs_tol=1e-4))

    def test_rotation(self):
        """Test the rotation by comparing to np.rot90."""
        nSeg = 1
        rInPix = 7.2
        rOutPix = 19.5
        nx = 61
        ny = 61
        xOffset = 0
        yOffset = 0
        angOpen = 65
        angRot = 15
        nsp = 100
        dTheta = 5

        mask = gen_annular_segments(nSeg, rInPix, rOutPix, nx, ny,
            xOffset, yOffset, angOpen, angRot, nSubpixels=nsp, dTheta=dTheta)

        maskRot90 = gen_annular_segments(nSeg, rInPix, rOutPix, nx, ny,
            xOffset, yOffset, angOpen, angRot+90,
            nSubpixels=nsp, dTheta=dTheta)

        sumAbsDiff = np.sum(np.abs(maskRot90 - np.rot90(mask, -1)))

        self.assertTrue(isclose(sumAbsDiff, 0, abs_tol=1e-4))

    def test_return_zeros(self):
        """Test the lateral offsets by comparing to np.roll."""
        nSeg = 1
        rInPix = 7.2
        rOutPix = 19.5
        nx = 61
        ny = 61
        xOffset = 0
        yOffset = 0
        angOpen = 0
        angRot = 15
        nsp = 100
        dTheta = 5

        mask = gen_annular_segments(nSeg, rInPix, rOutPix, nx, ny,
            xOffset, yOffset, angOpen, angRot, nSubpixels=nsp, dTheta=dTheta)

        self.assertTrue(isclose(0, np.sum(mask), abs_tol=1e-8))


class TestInputFailure(unittest.TestCase):
    """Test suite for valid function inputs."""

    def test_gen_annular_segments_inputs(self):
        """Test the inputs of gen_annular_segments_inputs."""
        rInPix = 7.2
        rOutPix = 19.5
        nx = 60
        ny = 71
        xOffset = -5
        yOffset = 2
        angOpen = 65
        angRot = 15
        nsp = 100
        dTheta = 5
        nSeg = 1

        # Check standard inputs do not raise anything first
        gen_annular_segments(nSeg, rInPix, rOutPix, nx, ny,
                             xOffset, yOffset, angOpen, angRot,
                             nSubpixels=nsp, dTheta=dTheta)

        for nSegBad in (-1, 0, 3, 2.1, 1j, np.ones(5), np.ones((5, 2)), 'a'):
            with self.assertRaises(TypeError):
                gen_annular_segments(nSegBad, rInPix, rOutPix, nx, ny,
                                 xOffset, yOffset, angOpen, angRot,
                                 nSubpixels=nsp, dTheta=dTheta)

        for rInPixBad in (-1, 0, 1j, [5], np.ones(5), np.ones((5, 2)), 'a'):
            with self.assertRaises(TypeError):
                gen_annular_segments(nSeg, rInPixBad, rOutPix, nx, ny,
                                 xOffset, yOffset, angOpen, angRot,
                                 nSubpixels=nsp, dTheta=dTheta)

        for rOutPixBad in (-1, 0, 1j, [5], np.ones(5), np.ones((5, 2)), 'a'):
            with self.assertRaises(TypeError):
                gen_annular_segments(nSeg, rInPix, rOutPixBad, nx, ny,
                                 xOffset, yOffset, angOpen, angRot,
                                 nSubpixels=nsp, dTheta=dTheta)

        for nxBad in (-1, 0, 2.1, 1j, np.ones(5), np.ones((5, 2)), 'a'):
            with self.assertRaises(TypeError):
                gen_annular_segments(nSeg, rInPix, rOutPix, nxBad, ny,
                                 xOffset, yOffset, angOpen, angRot,
                                 nSubpixels=nsp, dTheta=dTheta)

        for nyBad in (-1, 0, 2.1, 1j, np.ones(5), np.ones((5, 2)), 'a'):
            with self.assertRaises(TypeError):
                gen_annular_segments(nSeg, rInPix, rOutPix, nx, nyBad,
                                 xOffset, yOffset, angOpen, angRot,
                                 nSubpixels=nsp, dTheta=dTheta)

        for xOffsetBad in (1j, [5], np.ones(5), np.ones((5, 2)), 'a'):
            with self.assertRaises(TypeError):
                gen_annular_segments(nSeg, rInPix, rOutPix, nx, ny,
                                 xOffsetBad, yOffset, angOpen, angRot,
                                 nSubpixels=nsp, dTheta=dTheta)

        for yOffsetBad in (1j, [5], np.ones(5), np.ones((5, 2)), 'a'):
            with self.assertRaises(TypeError):
                gen_annular_segments(nSeg, rInPix, rOutPix, nx, ny,
                                 xOffset, yOffsetBad, angOpen, angRot,
                                 nSubpixels=nsp, dTheta=dTheta)

        for angOpenBad in (-1, 1j, [5], np.ones(5), np.ones((5, 2)), 'a'):
            with self.assertRaises(TypeError):
                gen_annular_segments(nSeg, rInPix, rOutPix, nx, ny,
                                 xOffset, yOffset, angOpenBad, angRot,
                                 nSubpixels=nsp, dTheta=dTheta)

        for angRotBad in (1j, [5], np.ones(5), np.ones((5, 2)), 'a'):
            with self.assertRaises(TypeError):
                gen_annular_segments(nSeg, rInPix, rOutPix, nx, ny,
                                 xOffset, yOffset, angOpen, angRotBad,
                                 nSubpixels=nsp, dTheta=dTheta)

        for nspBad in (-1, 0, 2.1, 1j, [5], np.ones(5), np.ones((5, 2)), 'a'):
            with self.assertRaises(TypeError):
                gen_annular_segments(nSeg, rInPix, rOutPix, nx, ny,
                                 xOffset, yOffset, angOpen, angRot,
                                 nSubpixels=nspBad, dTheta=dTheta)

        for dThetaBad in (-1, 1j, [5], np.ones(5), np.ones((5, 2)), 'a'):
            with self.assertRaises(TypeError):
                gen_annular_segments(nSeg, rInPix, rOutPix, nx, ny,
                                 xOffset, yOffset, angOpen, angRot,
                                 nSubpixels=nsp, dTheta=dThetaBad)


if __name__ == '__main__':
    unittest.main()
