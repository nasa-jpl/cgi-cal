# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Unit tests for the DMREG utility functions."""
from math import isclose
import os
import unittest

from astropy.io import fits
import numpy as np

from cal.ampthresh.ampthresh import ampthresh
from cal.conjugate.util.propcustom import gen_zernikes
from cal.dmreg import dr_util
from cal.util.dmhtoph import dmhtoph, volts_to_dmh
from cal.util.insertinto import insertinto as inin
from cal.util.loadyaml import loadyaml
from cal.util import remove_ptt


DMREG_PATH = os.path.dirname(os.path.abspath(__file__))

not_a_2d_array = (True, -1, 1, 1.1, 1j, np.ones((5, 5, 2)), 'string', {'a': 2})
not_a_1d_array = (True, -1, 1, 1.1, 1j, np.ones((5, 5)), 'string', {'a': 2})
not_a_dict = (True, -1, 1, 1.1, 1j, np.ones(5), np.ones((5, 5)), 'string')
not_a_real_scalar = (1j, )  # (1j, np.ones(5), 'string', {'a': 2})
not_a_real_positive_scalar = (-1, 0, 1j) #, np.ones(5), 'string', {'a': 2})
not_a_positive_scalar_integer = (-1, 0, 1.1, 1j, np.ones((5, 5, 2)),
                                 'string', {'a': 2})


class TestCalcSPMDeltaZernikes(unittest.TestCase):
    """Test all the supporting methods of maskgen."""

    def setUp(self):
        self.ph0 = fits.getdata(os.path.join(
            DMREG_PATH, 'testdata', 'ut_calc_spm_delta_zernikes_ph0.fits'))
        self.ph1 = fits.getdata(os.path.join(
            DMREG_PATH, 'testdata', 'ut_calc_spm_delta_zernikes_ph1.fits'))
        self.amp1 = fits.getdata(os.path.join(
            DMREG_PATH, 'testdata', 'ut_calc_spm_delta_zernikes_amp1.fits'))
        self.diamPupil = 300.22955322265625
        self.xOffsetPupil = -0.5662876842292958
        self.yOffsetPupil = 0.378791850560172
        self.Z_noll_max=11
        self.nErode=1
        self.nBin=21
        self.lam = 575e-9

    def test_calc_spm_delta_zernikes_on_sim_data(self):
        """Test the fitting of low-order modes between PRs on sim data."""
        # 4. generate residual maps for analysis and convenience
        Zvec0 = np.array([0.3, 0.5, -0.2, 0.1, -0.36, 0.2, 0.42, -0.16, -0.32, -0.11, 0.23])
        ph0 = gen_zernikes(
            np.arange(4, self.Z_noll_max+1), Zvec0[3:], self.xOffsetPupil,
            self.yOffsetPupil, self.diamPupil, nArray=self.amp1.shape[0])
        
        Zvec1 = np.array([0.2, 0.1, -0.1, 0.3, 0.2, 0.1, 0.32, -0.36, -0.2, 0.21, -0.30])
        ph1 = gen_zernikes(
            np.arange(4, self.Z_noll_max+1), Zvec1[3:], self.xOffsetPupil,
            self.yOffsetPupil, self.diamPupil, nArray=self.amp1.shape[0])
        
        Zvec, zern_combo, wfe_actual, bMask = dr_util.calc_spm_delta_zernikes(
            ph0, self.amp1, ph1, self.diamPupil, self.xOffsetPupil,
            self.yOffsetPupil, Z_noll_max=self.Z_noll_max, nErode=self.nErode,
            nBin=self.nBin,
            )

        wfe_residual = bMask * (wfe_actual - zern_combo)
        wfe_residual = remove_ptt.remove_ptt_directly(wfe_residual, bMask)
        rms_residual_rad = np.sqrt(np.mean(wfe_residual[bMask]**2))

        self.assertTrue(isclose(rms_residual_rad, 0,
                                abs_tol=10*np.finfo(float).eps))

    def test_calc_spm_delta_zernikes_on_real_data(self):
        """Test the fitting of low-order modes between PRs on real data."""
        Zvec, zern_combo, wfe_actual, bMask = dr_util.calc_spm_delta_zernikes(
            self.ph0, self.amp1, self.ph1, self.diamPupil, self.xOffsetPupil,
            self.yOffsetPupil, Z_noll_max=self.Z_noll_max, nErode=self.nErode,
            nBin=self.nBin,
            )

        wfe_residual = bMask * (wfe_actual - zern_combo)
        wfe_residual = remove_ptt.remove_ptt_directly(wfe_residual, bMask)

        # RMS values in radians
        rms_residual = np.sqrt(np.mean(wfe_residual[bMask]**2))
        rms_actual = np.sqrt(np.mean(wfe_actual[bMask]**2))

        # Looks correct by eye but high-frequency residuals mean there is
        # only a factor of ~2.5 improvement in RMS WFE.
        self.assertTrue(rms_residual < 0.5*rms_actual)

    def test_calc_spm_delta_zernikes_inputs(self):
        """Test the inputs of calc_spm_delta_zernikes()."""
        for ph0 in not_a_2d_array:
            with self.assertRaises(TypeError):
                dr_util.calc_spm_delta_zernikes(
                    ph0, self.amp1, self.ph1, self.diamPupil,
                    self.xOffsetPupil, self.yOffsetPupil,
                    Z_noll_max=self.Z_noll_max, nErode=self.nErode,
                    nBin=self.nBin,
                    )

        for amp1 in not_a_2d_array:
            with self.assertRaises(TypeError):
                dr_util.calc_spm_delta_zernikes(
                    self.ph0, amp1, self.ph1, self.diamPupil,
                    self.xOffsetPupil, self.yOffsetPupil,
                    Z_noll_max=self.Z_noll_max, nErode=self.nErode,
                    nBin=self.nBin,
                    )

        for ph1 in not_a_2d_array:
            with self.assertRaises(TypeError):
                dr_util.calc_spm_delta_zernikes(
                    self.ph0, self.amp1, ph1, self.diamPupil,
                    self.xOffsetPupil, self.yOffsetPupil,
                    Z_noll_max=self.Z_noll_max, nErode=self.nErode,
                    nBin=self.nBin,
                    )

        for diamPupil in not_a_real_positive_scalar:
            with self.assertRaises(TypeError):
                dr_util.calc_spm_delta_zernikes(
                    self.ph0, self.amp1, self.ph1, diamPupil,
                    self.xOffsetPupil, self.yOffsetPupil,
                    Z_noll_max=self.Z_noll_max, nErode=self.nErode,
                    nBin=self.nBin,
                    )

        for xOffsetPupil in not_a_real_scalar:
            with self.assertRaises(TypeError):
                dr_util.calc_spm_delta_zernikes(
                    self.ph0, self.amp1, self.ph1, self.diamPupil,
                    xOffsetPupil, self.yOffsetPupil,
                    Z_noll_max=self.Z_noll_max, nErode=self.nErode,
                    nBin=self.nBin,
                    )

        for yOffsetPupil in not_a_real_scalar:
            with self.assertRaises(TypeError):
                dr_util.calc_spm_delta_zernikes(
                    self.ph0, self.amp1, self.ph1, self.diamPupil,
                    self.xOffsetPupil, yOffsetPupil,
                    Z_noll_max=self.Z_noll_max, nErode=self.nErode,
                    nBin=self.nBin,
                    )

        for Z_noll_max in not_a_positive_scalar_integer:
            with self.assertRaises(TypeError):
                dr_util.calc_spm_delta_zernikes(
                    self.ph0, self.amp1, self.ph1, self.diamPupil,
                    self.xOffsetPupil, self.yOffsetPupil,
                    Z_noll_max=Z_noll_max, nErode=self.nErode,
                    nBin=self.nBin,
                    )
        
        # Also test bad values of Z_noll_max
        for Z_noll_max in (1, 2, 3):
            with self.assertRaises(ValueError):
                dr_util.calc_spm_delta_zernikes(
                    self.ph0, self.amp1, self.ph1, self.diamPupil,
                    self.xOffsetPupil, self.yOffsetPupil,
                    Z_noll_max=Z_noll_max, nErode=self.nErode,
                    nBin=self.nBin,
                    )

        for nErode in not_a_positive_scalar_integer:
            with self.assertRaises(TypeError):
                dr_util.calc_spm_delta_zernikes(
                    self.ph0, self.amp1, self.ph1, self.diamPupil,
                    self.xOffsetPupil, self.yOffsetPupil,
                    Z_noll_max=self.Z_noll_max, nErode=nErode,
                    nBin=self.nBin,
                    )

        for nBin in not_a_positive_scalar_integer:
            with self.assertRaises(TypeError):
                dr_util.calc_spm_delta_zernikes(
                    self.ph0, self.amp1, self.ph1, self.diamPupil,
                    self.xOffsetPupil, self.yOffsetPupil,
                    Z_noll_max=self.Z_noll_max, nErode=self.nErode,
                    nBin=nBin,
                    )


class TestDMRegPokeGridSupportFunctions(unittest.TestCase):
    """Test support functions for DM poke grid registration."""

    def setUp(self):
        self.lam = 575e-9
        self.amp_unpoked = fits.getdata(os.path.join(
            DMREG_PATH, 'cgidata', 'amp_cgi_prnum_000168.fits'))
        self.bmask = ampthresh(self.amp_unpoked)

        self.fn_dmreg_def = os.path.join(
            DMREG_PATH, 'cgidata', 'ut_dmreg_def_sim_stress_tests.yaml')
        self.fn_dmreg_tuning_params = os.path.join(
            DMREG_PATH, 'cgidata', 'ut_dmreg_tuning_params.yaml')
        self.fn_pupil_fit_params = os.path.join(
            DMREG_PATH, 'cgidata', 'params_fit_unmasked_pupil_fft.yaml')
        self.data_path_pupil =  os.path.join(DMREG_PATH, 'cgidata')
        
        path_to_dmreg_yaml = os.path.dirname(self.fn_dmreg_def)
        dmConfig = loadyaml(self.fn_dmreg_def)
        dm = dmConfig["dms"]
        dmr = dm['DM1']['registration']
        inf_func = fits.getdata(os.path.join(path_to_dmreg_yaml, dmr['inffn']))

        self.ppact_cx0 = dmr['ppact_cx']
        self.ppact_cy0 = dmr['ppact_cy']
        self.dx0 = dmr['dx']
        self.dy0 = dmr['dy']
        self.thact0 = dmr['thact']
        self.opt_params = [self.ppact_cx0, self.ppact_cy0, self.dx0, self.dy0,
                           self.thact0]

        gainmap = fits.getdata(os.path.join(path_to_dmreg_yaml,
                                            dm['DM1']['voltages']['gainfn']))
        fn_crosstalk_base = dm['DM1']['voltages']['crosstalkfn']
        delta_v = fits.getdata(os.path.join(
            DMREG_PATH, 'cgidata', 'dmrel_grid_for_dmreg_dm1.fits'))

        # Cannot give None to os.path.join()
        if fn_crosstalk_base is not None:
            crosstalk_fn = os.path.join(path_to_dmreg_yaml, fn_crosstalk_base)
        else:
            crosstalk_fn = None

        # Convert voltage map to height map and account for crosstalk
        dh_map = volts_to_dmh(gainmap, delta_v, self.lam,
                              crosstalk_fn=crosstalk_fn)
        # Compute the modeled phase change from the poke grid.
        ph_diff = 2 * dmhtoph(
            self.bmask.shape[0], self.bmask.shape[1], dh_map, dmr['nact'],
            inf_func, dmr['ppact_d'], dmr['ppact_cx'],
            dmr['ppact_cy'], dmr['dx'], dmr['dy'], dmr['thact'], dmr['flipx'])
        ph_model_zern = np.zeros_like(ph_diff)
        
        # So that the fitted shape is equal and opposite the input to double
        # the error.
        ph_diff *= -1

        self.fixed_inputs = {
            'ph_diff_meas': ph_diff,
            'ph_model_zern': ph_model_zern,
            'bmask': self.bmask,
            'dh_map': dh_map,
            'inf_func': inf_func,
            'ppact_d': dmr['ppact_d'],
            'nact': dmr['nact'],
            'flipx': dmr['flipx'],
        }
    
    def test_compute_phase_diff_model_shapes(self):
        """Perform basic tests on the outputs of compute_phase_diff_model()."""
    
        ph_diff_model, ph_diff_meas, bmask, cost = \
            dr_util.compute_phase_diff_model(self.opt_params, self.fixed_inputs)
            
        # Verify ph_diff_model, ph_diff_meas, bmask all have the same shape
        self.assertTrue(np.all(ph_diff_model.shape == ph_diff_meas.shape))
        self.assertTrue(np.all(ph_diff_model.shape == bmask.shape))
        self.assertTrue(np.all(ph_diff_meas.shape == bmask.shape))

    def test_compute_phase_diff_cost_value(self):
        """Perform basic tests on an output of compute_phase_diff_model()."""
    
        ph_diff_model, ph_diff_meas, bmask, cost = \
            dr_util.compute_phase_diff_model(self.opt_params, self.fixed_inputs)
        
        cost_expected = \
            4 * np.sum(self.fixed_inputs['ph_diff_meas'][self.bmask]**2)

        self.assertTrue(isclose(cost, cost_expected, rel_tol=1e-8))

    def test_compute_phase_diff_ph_diff_meas_value(self):
        """Verify that ph_diff_meas is unchanged."""
    
        ph_diff_model, ph_diff_meas, bmask, cost = \
            dr_util.compute_phase_diff_model(self.opt_params, self.fixed_inputs)
        
        ph_diff_meas_in = self.fixed_inputs['ph_diff_meas']
        ph_diff_meas_out = inin(ph_diff_meas, ph_diff_meas_in.shape)

        self.assertTrue(np.allclose(ph_diff_meas_out,
                                    ph_diff_meas_in,
                                    atol=1e-8))

    def test_cost_func_dmreg(self):
        """Check the accuracy of dr_util.cost_func_dmreg()."""
        cost = dr_util.cost_func_dmreg(self.opt_params, self.fixed_inputs)
        cost_expected = \
            4 * np.sum(self.fixed_inputs['ph_diff_meas'][self.bmask]**2)

        self.assertTrue(isclose(cost, cost_expected, rel_tol=1e-8))

    def test_compute_phase_diff_model_inputs(self):
        """
        Perform checks on bad inputs to compute_phase_diff_model().
        
        Also counts for the thin wrapper function cost_func_dmreg(), which
        takes the exact same inputs.
        """
        # Wrong length of first input
        with self.assertRaises(ValueError):
            dr_util.compute_phase_diff_model(self.opt_params[0:2],
                                             self.fixed_inputs)

        for opt_params_bad in not_a_1d_array:
            with self.assertRaises(TypeError):
                dr_util.compute_phase_diff_model(opt_params_bad,
                                                 self.fixed_inputs)
        for fixed_inputs_bad in not_a_dict:
            with self.assertRaises(TypeError):
                dr_util.compute_phase_diff_model(self.opt_params,
                                                 fixed_inputs_bad)

        # Missing dictionary key
        required_keys = ['ph_diff_meas', 'ph_model_zern', 'bmask', 'dh_map',
                         'inf_func', 'ppact_d', 'nact', 'flipx']
        for key in required_keys:
            fixed_inputs_incomplete = self.fixed_inputs.copy()
            fixed_inputs_incomplete.pop(key)
            with self.assertRaises(KeyError):
                dr_util.compute_phase_diff_model(self.opt_params,
                                                 fixed_inputs_incomplete)

        for bad_value in not_a_real_positive_scalar:
            opt_params_bad = [bad_value, self.ppact_cy0,
                              self.dx0, self.dy0, self.thact0]
            with self.assertRaises(TypeError):
                dr_util.compute_phase_diff_model(opt_params_bad,
                                                  self.fixed_inputs)

        for bad_value in not_a_real_positive_scalar:
            opt_params_bad = [self.ppact_cx0, bad_value,
                              self.dx0, self.dy0, self.thact0]
            with self.assertRaises(TypeError):
                dr_util.compute_phase_diff_model(opt_params_bad,
                                                  self.fixed_inputs)

        for bad_value in not_a_real_scalar:
            opt_params_bad = [self.ppact_cx0, self.ppact_cy0,
                              bad_value, self.dy0, self.thact0]
            with self.assertRaises(TypeError):
                dr_util.compute_phase_diff_model(opt_params_bad,
                                                  self.fixed_inputs)

        for bad_value in not_a_real_scalar:
            opt_params_bad = [self.ppact_cx0, self.ppact_cy0,
                              self.dx0, bad_value, self.thact0]
            with self.assertRaises(TypeError):
                dr_util.compute_phase_diff_model(opt_params_bad,
                                                  self.fixed_inputs)

        for bad_value in not_a_real_scalar:
            opt_params_bad = [self.ppact_cx0, self.ppact_cy0,
                              self.dx0, self.dy0, bad_value]
            with self.assertRaises(TypeError):
                dr_util.compute_phase_diff_model(opt_params_bad,
                                                  self.fixed_inputs)


if __name__ == '__main__':
    unittest.main()
