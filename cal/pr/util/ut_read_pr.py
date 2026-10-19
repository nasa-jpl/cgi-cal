# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Test class for the functions for reading phase retrievals."""
import os
import unittest

from astropy.io import fits
import numpy as np

from cal.pr.util.read_pr import load_backend_pr, read_and_prop_pr
from cal.util.insertinto import insertinto as inin
from cal.util.loadyaml import loadyaml
from cal.util.remove_ptt import remove_ptt_directly

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND_PR_PATH = os.path.join(HERE, 'cgidata', 'prnum_000279')

NOT_REAL_2D_ARRAY = [
    False, -1, 0, 'txt', None, [1, 2, 3],  # not array
    np.ones((100,)), np.ones((5, 5, 5)),  # not 2D
    1j*np.eye(2),  # not real
]

NOT_DICT = [
    False, -1, 0, 1J, 'txt', None, [1, 2, 3], (1, 2, 3), np.ones((100,)),
]

NOT_BOOL = [
    -1, 1J, 'txt', None, [1, 2, 3], (1, 2, 3), np.ones((100,)),
]


class TestLoadBackendPR(unittest.TestCase):
    """
    X The default maps are loaded as expected.
    X Another folder can be specified as the source.
    X Folder with missing files fails.
    """

    def test_default_success(self):
        """Test that the default input works as expected."""
        amp, ph, xc, yc, pr_params = load_backend_pr()

        # Compare to expected maps
        amp2 = fits.getdata(os.path.join(BACKEND_PR_PATH, 'amp.fits'))
        ph2 = fits.getdata(os.path.join(BACKEND_PR_PATH, 'phwrap.fits'))
        pr_params2 = loadyaml(os.path.join(BACKEND_PR_PATH, 'parms.yaml'))

        self.assertTrue(np.array_equal(amp, amp2))
        self.assertTrue(np.array_equal(ph, ph2))
        self.assertTrue(pr_params == pr_params2)
        self.assertTrue(xc == pr_params2['XYcen_PIL'][0])
        self.assertTrue(yc == pr_params2['XYcen_PIL'][1])

    def test_alternate_success(self):
        """Test that another folder can be used for the inputs."""
        data_path = os.path.join(HERE, 'testdata', 'prnum_000277')
        amp, ph, xc, yc, pr_params = load_backend_pr(data_path=data_path)

        # Compare to expected maps
        amp2 = fits.getdata(os.path.join(data_path, 'amp.fits'))
        ph2 = fits.getdata(os.path.join(data_path, 'phwrap.fits'))
        pr_params2 = loadyaml(os.path.join(data_path, 'parms.yaml'))

        self.assertTrue(np.array_equal(amp, amp2))
        self.assertTrue(np.array_equal(ph, ph2))
        self.assertTrue(pr_params == pr_params2)
        self.assertTrue(xc == pr_params2['XYcen_PIL'][0])
        self.assertTrue(yc == pr_params2['XYcen_PIL'][1])

    def test_missing_files(self):
        """Test for an exception if specified folder is missing files."""
        data_path = os.path.join(HERE, 'testdata', 'prnum_missing_files')
        with self.assertRaises(FileNotFoundError):
            load_backend_pr(data_path=data_path)


class TestReadAndPropPR(unittest.TestCase):
    """
    X Valid inputs succeed.
    X Backend phase is subtracted when asked.
    X Phase is unchanged if backend is not subtracted.
    X Piston, tip, and tilt are removed from the phase.
    X Small blacked out regions are removed from the software mask.
    X Fresnel propagation occurs when asked.
    X Output phase is unwrapped.
    Bad inputs fail.
    """

    @classmethod
    def setUpClass(self):
        """Predefine common variable inputs available for all tests."""
        self.amp = fits.getdata(os.path.join(
            HERE, 'testdata', 'prnum_000277', 'amp.fits'))
        self.phwrap = fits.getdata(os.path.join(
            HERE, 'testdata', 'prnum_000277', 'phwrap.fits'))
        self.pr_params = loadyaml(os.path.join(
            HERE, 'testdata', 'prnum_000277', 'parms.yaml'))
        self.fn_pr_prop_params = os.path.join(
            HERE, 'cgidata', 'params_read_and_prop_pr.yaml')
        self.fn_pr_prop_params_zero_deltaz = os.path.join(
            HERE, 'testdata', 'params_read_and_prop_pr_zero_deltaz.yaml')
        self.fn_pr_prop_params_no_min = os.path.join(
            HERE, 'testdata', 'params_read_and_prop_pr_zero_min.yaml')
        self.subtract_backend = True

        diamBeam = 300  # approximate
        xs = 2*(np.arange(len(self.amp)) - len(self.amp)/2)/diamBeam
        XS, YS = np.meshgrid(xs, xs)
        RS2 = XS**2 + YS**2
        THETAS = np.arctan2(YS, XS)
        self.phwrapPTT = self.phwrap + 2*np.pi*(8*XS + 3*YS + 2)
        self.wavesZ5 = 0.1
        Z5 = np.sqrt(6)*RS2*np.sin(2*THETAS)
        self.phwrapZ5 = np.angle(np.exp(1j*(self.phwrap +
                                            2*np.pi*(self.wavesZ5*Z5))))
        self.phwrapPoints = self.phwrap
        self.range = 8*np.pi
        self.phwrapPoints[::4, ::4] += self.range
        self.ph_zeros = np.zeros_like(self.phwrap)

    def test_backend_subtraction(self):
        # Backend is subtracted
        amp_pupil, ph_pupil, bmask_pupil, x_offset, y_offset, lam = \
            read_and_prop_pr(
                self.amp,
                self.ph_zeros,
                self.pr_params,
                self.fn_pr_prop_params_zero_deltaz,
                subtract_backend=True,
                )

        pv_input = np.max(self.ph_zeros) - np.min(self.ph_zeros)
        pv_output = np.max(bmask_pupil*ph_pupil) - np.min(bmask_pupil*ph_pupil)

        self.assertTrue(pv_input < 10*np.finfo(float).eps)
        self.assertTrue(pv_output > 0.4)

    def test_no_change_if_no_backend_subtraction(self):
        # Backend not subtracted
        amp_pupil, ph_pupil, bmask_pupil, x_offset, y_offset, lam = \
            read_and_prop_pr(
                self.amp,
                self.phwrapZ5,
                self.pr_params,
                self.fn_pr_prop_params_zero_deltaz,
                subtract_backend=False,
            )

        # The input has some tip/tilt, so remove_ptt_directly is used to
        # remove that and make sure that the difference otherwise is zero.
        self.assertTrue((bmask_pupil*(remove_ptt_directly(
            self.phwrapZ5 - ph_pupil, bmask_pupil)) <
            100*np.finfo(float).eps).all())

    def test_ptt_removed(self):
        # Without PTT at input
        amp_pupil, ph_pupil, bmask_pupil, x_offset, y_offset, lam = \
            read_and_prop_pr(
                self.amp,
                self.phwrap,
                self.pr_params,
                self.fn_pr_prop_params,
                subtract_backend=self.subtract_backend,
                )

        # With PTT at input
        amp_pupil2, ph_pupil2, bmask_pupil2, x_offset2, y_offset2, lam2 = \
            read_and_prop_pr(
                self.amp,
                self.phwrapPTT,
                self.pr_params,
                self.fn_pr_prop_params,
                subtract_backend=self.subtract_backend,
                )

        self.assertTrue(np.array_equal(bmask_pupil, bmask_pupil2))
        self.assertTrue(np.allclose(bmask_pupil*ph_pupil,
                                    bmask_pupil2*ph_pupil2,
                                    atol=1e-3))
        self.assertTrue(np.allclose(amp_pupil, amp_pupil2, atol=1e-8))
        self.assertTrue(x_offset != x_offset2)
        self.assertTrue(y_offset != y_offset2)

    def test_ptt_removed_zero_deltaz(self):
        # Without PTT at input
        amp_pupil, ph_pupil, bmask_pupil, x_offset, y_offset, lam = \
            read_and_prop_pr(
                self.amp,
                self.phwrap,
                self.pr_params,
                self.fn_pr_prop_params_zero_deltaz,
                subtract_backend=self.subtract_backend,
                )

        # With PTT at input
        amp_pupil2, ph_pupil2, bmask_pupil2, x_offset2, y_offset2, lam2 = \
            read_and_prop_pr(
                self.amp,
                self.phwrapPTT,
                self.pr_params,
                self.fn_pr_prop_params_zero_deltaz,
                subtract_backend=self.subtract_backend,
                )

        self.assertTrue(np.array_equal(bmask_pupil, bmask_pupil2))
        self.assertTrue(np.allclose(bmask_pupil*ph_pupil,
                                    bmask_pupil2*ph_pupil2, atol=1e-3))
        self.assertTrue(np.allclose(amp_pupil, amp_pupil2, atol=1e-8))
        self.assertTrue(x_offset != x_offset2)
        self.assertTrue(y_offset != y_offset2)

    def test_small_areas_not_blacked_out(self):
        # No small areas allowed
        amp_pupil, ph_pupil, bmask_pupil, x_offset, y_offset, lam = \
            read_and_prop_pr(
                self.amp,
                self.phwrap,
                self.pr_params,
                self.fn_pr_prop_params,
                subtract_backend=self.subtract_backend,
                )

        # Small areas allowed
        amp_pupil2, ph_pupil2, bmask_pupil2, x_offset2, y_offset2, lam2 = \
            read_and_prop_pr(
                self.amp,
                self.phwrapPTT,
                self.pr_params,
                self.fn_pr_prop_params_no_min,
                subtract_backend=self.subtract_backend,
                )

        # One pixel should be different where the dead actuator is.
        self.assertTrue(np.sum(bmask_pupil.astype(int)) -
                        np.sum(bmask_pupil2.astype(int)) == 1)

    def test_prop_occurred(self):
        """Test that the phase and amplitude changed from propagation."""
        # No Fresnel propagation
        amp_pupil, ph_pupil, bmask_pupil, x_offset, y_offset, lam = \
            read_and_prop_pr(
                self.amp,
                self.phwrap,
                self.pr_params,
                self.fn_pr_prop_params_zero_deltaz,
                subtract_backend=self.subtract_backend,
                )

        # With Fresnel propagation
        amp_pupil2, ph_pupil2, bmask_pupil2, x_offset2, y_offset2, lam2 = \
            read_and_prop_pr(
                self.amp,
                self.phwrap,
                self.pr_params,
                self.fn_pr_prop_params,
                subtract_backend=self.subtract_backend,
                )

        self.assertTrue(not np.allclose(bmask_pupil*ph_pupil,
                                        bmask_pupil2*ph_pupil2,
                                        atol=1e-3))
        self.assertTrue(not np.allclose(amp_pupil, amp_pupil2, atol=1e-8))

    def test_phase_unwrapped(self):
        """Test that the output phase is unwrapped."""
        # No Fresnel propagation
        amp_pupil, ph_pupil, bmask_pupil, x_offset, y_offset, lam = \
            read_and_prop_pr(
                self.amp,
                self.phwrapPoints,
                self.pr_params,
                self.fn_pr_prop_params_zero_deltaz,
                subtract_backend=False,
                )

        pv_input = np.max(bmask_pupil*self.phwrapPoints) -\
            np.min(bmask_pupil*self.phwrapPoints)
        pv_output = np.max(bmask_pupil*ph_pupil) - np.min(bmask_pupil*ph_pupil)

        self.assertTrue(pv_input > self.range)
        self.assertTrue(pv_output < 2*np.pi)

    def test_invalid_amp(self):
        """ invalid input amplitude image raises error """
        for perr in NOT_REAL_2D_ARRAY:
            with self.assertRaises(TypeError):
                read_and_prop_pr(
                    perr,
                    self.phwrap,
                    self.pr_params,
                    self.fn_pr_prop_params_zero_deltaz,
                    subtract_backend=self.subtract_backend,
                )

    def test_invalid_ph(self):
        """ invalid input phase map raises error """
        for perr in NOT_REAL_2D_ARRAY:
            with self.assertRaises(TypeError):
                read_and_prop_pr(
                    self.amp,
                    perr,
                    self.pr_params,
                    self.fn_pr_prop_params_zero_deltaz,
                    subtract_backend=self.subtract_backend,
                )

    def test_invalid_ph_shape(self):
        """invalid shape for ph (mismatches amp)"""
        M = self.amp.shape[0] + 4
        N = self.amp.shape[1] + 4
        with self.assertRaises(ValueError):
            read_and_prop_pr(
                self.amp,
                inin(self.phwrap, (M, N)),
                self.pr_params,
                self.fn_pr_prop_params_zero_deltaz,
                subtract_backend=self.subtract_backend,
            )

    def test_invalid_pr_params(self):
        """ invalid input param dict raises error """
        for perr in NOT_DICT:
            with self.assertRaises(TypeError):
                read_and_prop_pr(
                    self.amp,
                    self.phwrap,
                    perr,
                    self.fn_pr_prop_params_zero_deltaz,
                    subtract_backend=self.subtract_backend,
                )

    def test_invalid_subtract_backend(self):
        """ invalid input subtract_backend raises error """
        for perr in NOT_BOOL:
            with self.assertRaises(TypeError):
                read_and_prop_pr(
                    self.amp,
                    self.phwrap,
                    self.pr_params,
                    self.fn_pr_prop_params_zero_deltaz,
                    subtract_backend=perr,
                )


if __name__ == '__main__':
    unittest.main()
