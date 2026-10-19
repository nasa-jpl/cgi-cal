# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Test suite for remove_ptt.py."""
from math import isclose
import os
import unittest

import numpy as np

from . import mft, remove_ptt, shapes


HERE = os.path.dirname(os.path.abspath(__file__))

not_a_2d_array = (True, -1, 1, 1.1, 1j, np.ones((5, 5, 4)), 'string', {'a': 2})
not_a_1d_array = (True, -1, 1, 1.1, 1j, np.ones((5, 5)), 'string', {'a': 2})
not_a_dict = (True, -1, 1, 1.1, 1j, np.ones(5), np.ones((5, 5)), 'string')
not_a_real_scalar = (1j, np.ones(5), 'string', {'a': 2})
not_a_real_positive_scalar = (-1, 0, 1j, np.ones(5), 'string', {'a': 2})
not_a_positive_scalar_integer = (-1, 0, 1.1, 1j, np.ones(5), np.ones((5, 5)),
                                 'string')
not_a_string = (True, -1, 0, 1.1, 1j, np.ones(5), np.ones((5, 5)),  {'a': 2})


class TestRemovePTTDirectly(unittest.TestCase):
    """Unit tests for remove_ptt.remove_ptt_directly()."""

    def test_remove_ptt_directly(self):
        """Test that piston, tip, and tilt are fully removed."""
        N = 100
        x = np.arange(-N/2, N/2)/N
        [X, Y] = np.meshgrid(x, x)

        arrayIn = 2*X - 3*Y + 0.8
        mask = np.ones_like(X)
        mask[X*X+Y*Y > 0.5**2] = 0

        arrayOut = remove_ptt.remove_ptt_directly(arrayIn, mask)

        self.assertTrue(isclose(np.mean(arrayOut), 0, abs_tol=1e-12))

    def test_remove_ptt_directly_inputs(self):
        """Test the inputs of remove_ptt_directly."""
        arrayToFit = np.ones((10, 11))
        mask = np.ones((10, 11))

        for bad_val in not_a_2d_array:
            with self.assertRaises(ValueError):
                remove_ptt.remove_ptt_directly(bad_val, mask)
        for bad_val in not_a_2d_array:
            with self.assertRaises(ValueError):
                remove_ptt.remove_ptt_directly(arrayToFit, bad_val)

        # Check shape equality
        with self.assertRaises(ValueError):
            remove_ptt.remove_ptt_directly(np.ones((10, 10)),
                                           np.ones((11, 12)))


class TestRemovePTTAtFocus(unittest.TestCase):
    """Unit tests for remove_ptt.remove_ptt_at_focus()."""

    def setUp(self):
        """Define re-used variables."""
        nx = 121
        ny = 140
        roiRadiusPix = 50.0
        self.diam_pupil_pix = 2 * roiRadiusPix
        xShear = 5.2
        yShear = 3.3
        self.amp = shapes.circle(nx, ny, roiRadiusPix, xShear, yShear,
                                 nSubpixels=101)
        self.ppl = 2.262

        xs = np.linspace(-0.5, 0.5, nx) * (nx/self.diam_pupil_pix)/self.ppl
        ys = np.linspace(-0.5, 0.5, ny) * (ny/self.diam_pupil_pix)/self.ppl
        XS, YS = np.meshgrid(xs, ys)

        self.x_offset_true = 4.91
        self.y_offset_true = 2.34
        epup0 = self.amp * np.exp(2 + 2*np.pi*1j*(self.x_offset_true*XS +
                                                  self.y_offset_true*YS))
        self.ph = np.angle(epup0)

        self.ft_dir = 'reverse'
        self.fn_ref_psf = os.path.join(HERE, 'testdata',
                                       'ut_psf_ref_for_remove_ptt.fits')

        # Make centered reference PSF
        efoc = mft.do_mft(self.amp, (100, 100), self.ppl, self.diam_pupil_pix)
        psf0 = np.abs(efoc)
        psf0 = psf0/np.max(psf0)

    def test_forward(self):
        """Test that a bright circle is correctly fitted."""
        ft_dir = 'forward'
        ph_no_ptt, bmask, x_offset, y_offset = remove_ptt.remove_ptt_at_focus(
                self.amp, self.ph, self.ppl, self.diam_pupil_pix, ft_dir,
                self.fn_ref_psf,
                )

        # Verify that the tip/tilt is removed
        abs_tol_offset = 1e-3  # pixels
        self.assertTrue(isclose(self.x_offset_true,
                                x_offset,
                                abs_tol=abs_tol_offset))
        self.assertTrue(isclose(self.y_offset_true,
                                y_offset,
                                abs_tol=abs_tol_offset))

        # Verify that the piston is removed
        abs_tol_piston = 1e-8  # radians
        self.assertTrue(isclose(np.mean(ph_no_ptt[bmask]),
                                0,
                                abs_tol=abs_tol_piston))

    def test_reverse(self):
        """Test that a bright circle is correctly fitted."""
        ft_dir = 'reverse'
        ph_no_ptt, bmask, x_offset, y_offset = remove_ptt.remove_ptt_at_focus(
                self.amp, self.ph, self.ppl, self.diam_pupil_pix, ft_dir,
                self.fn_ref_psf,
                )

        # Verify that the tip/tilt is removed
        abs_tol_offset = 1e-3  # pixels
        self.assertTrue(isclose(-1*self.x_offset_true,
                                x_offset,
                                abs_tol=abs_tol_offset))
        self.assertTrue(isclose(-1*self.y_offset_true,
                                y_offset,
                                abs_tol=abs_tol_offset))

        # Verify that the piston is removed
        abs_tol_piston = 1e-8  # radians
        self.assertTrue(isclose(np.mean(ph_no_ptt[bmask]),
                                0,
                                abs_tol=abs_tol_piston))

    def test_input_0(self):
        """Test a bad input to remove_ptt_at_focus()."""
        for bad_val in not_a_2d_array:
            with self.assertRaises(TypeError):
                remove_ptt.remove_ptt_at_focus(
                    bad_val, self.ph, self.ppl, self.diam_pupil_pix,
                    self.ft_dir, self.fn_ref_psf,
                    )

    def test_input_1(self):
        """Test a bad input to remove_ptt_at_focus()."""
        for bad_val in not_a_2d_array:
            with self.assertRaises(TypeError):
                remove_ptt.remove_ptt_at_focus(
                    self.amp, bad_val, self.ppl, self.diam_pupil_pix,
                    self.ft_dir, self.fn_ref_psf,
                    )

    def test_input_2(self):
        """Test a bad input to remove_ptt_at_focus()."""
        for bad_val in not_a_real_positive_scalar:
            with self.assertRaises(TypeError):
                remove_ptt.remove_ptt_at_focus(
                    self.amp, self.ph, bad_val, self.diam_pupil_pix,
                    self.ft_dir, self.fn_ref_psf,
                    )

    def test_input_3(self):
        """Test a bad input to remove_ptt_at_focus()."""
        for bad_val in not_a_real_positive_scalar:
            with self.assertRaises(TypeError):
                remove_ptt.remove_ptt_at_focus(
                    self.amp, self.ph, self.ppl, bad_val,
                    self.ft_dir, self.fn_ref_psf,
                    )

    def test_input_4(self):
        """Test a bad input to remove_ptt_at_focus()."""
        for bad_val in not_a_string:
            with self.assertRaises(TypeError):
                remove_ptt.remove_ptt_at_focus(
                    self.amp, self.ph, self.ppl, self.diam_pupil_pix,
                    bad_val, self.fn_ref_psf,
                    )

    def test_input_4b(self):
        """Test bad values for ft_dir given to remove_ptt_at_focus()."""
        for bad_val in ('string', 'REVERSE', 'FORWARD'):
            with self.assertRaises(ValueError):
                remove_ptt.remove_ptt_at_focus(
                    self.amp, self.ph, self.ppl, self.diam_pupil_pix,
                    bad_val, self.fn_ref_psf,
                    )

    def test_input_5(self):
        """Test a bad input to remove_ptt_at_focus()."""
        for bad_val in not_a_string:
            with self.assertRaises(TypeError):
                remove_ptt.remove_ptt_at_focus(
                    self.amp, self.ph, self.ppl, self.diam_pupil_pix,
                    self.ft_dir, bad_val,
                    )

    def test_input_5b(self):
        """Test a bad input to remove_ptt_at_focus()."""
        # Relative path (remove HERE from setUp path)
        relative_fn_ref_psf = os.path.join('testdata',
                                       'ut_psf_ref_for_remove_ptt.fits')
        with self.assertRaises(ValueError):
            remove_ptt.remove_ptt_at_focus(
                self.amp, self.ph, self.ppl, self.diam_pupil_pix,
                self.ft_dir, relative_fn_ref_psf,
                )

    def test_input_6(self):
        """Test a bad input to remove_ptt_at_focus()."""
        for bad_val in not_a_string:
            with self.assertRaises(TypeError):
                remove_ptt.remove_ptt_at_focus(
                    self.amp, self.ph, self.ppl, self.diam_pupil_pix,
                    self.ft_dir, self.fn_ref_psf, fn_psffit_tuning=bad_val,
                    )

    def test_input_7(self):
        """Test a bad input to remove_ptt_at_focus()."""
        for bad_val in not_a_positive_scalar_integer:
            with self.assertRaises(TypeError):
                remove_ptt.remove_ptt_at_focus(
                    self.amp, self.ph, self.ppl, self.diam_pupil_pix,
                    self.ft_dir, self.fn_ref_psf, nCrop=bad_val,
                    )

    def test_input_8(self):
        """Test a bad input to remove_ptt_at_focus()."""
        for bad_val in not_a_positive_scalar_integer:
            with self.assertRaises(TypeError):
                remove_ptt.remove_ptt_at_focus(
                    self.amp, self.ph, self.ppl, self.diam_pupil_pix,
                    self.ft_dir, self.fn_ref_psf, nBin=bad_val,
                    )

    def test_input_9(self):
        """Test a bad input to remove_ptt_at_focus()."""
        for bad_val in not_a_positive_scalar_integer:
            with self.assertRaises(TypeError):
                remove_ptt.remove_ptt_at_focus(
                    self.amp, self.ph, self.ppl, self.diam_pupil_pix,
                    self.ft_dir, self.fn_ref_psf, nIter=bad_val,
                    )


if __name__ == '__main__':
    unittest.main()
