# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Unit tests for the DMREG functions."""
import os
import unittest
import numpy as np
from astropy.io import fits

from cal.util.dmhtoph import dmhtoph, volts_to_dmh
from cal.util.insertinto import insertinto as inin
from cal.util.loadyaml import loadyaml
from cal.dmreg.dmreg import (
    calc_software_mask, calc_translation_clocking, calc_scale,
    calc_dmreg_from_poke_grid, calc_dmreg_from_poke_grid_for_any_pupil,
)

DMREG_PATH = os.path.dirname(os.path.abspath(__file__))

not_a_2d_array = (True, -1, 1, 1.1, 1j, np.ones((5, 5, 4)), 'string', {'a': 2})
not_a_1d_array = (True, -1, 1, 1.1, 1j, np.ones((5, 5)), 'string', {'a': 2})
not_a_dict = (True, -1, 1, 1.1, 1j, np.ones(5), np.ones((5, 5)), 'string')
not_a_real_scalar = (1j, )  # (1j, np.ones(5), 'string', {'a': 2})
not_a_real_positive_scalar = (-1, 0, 1j, np.ones(5), 'string', {'a': 2})
not_a_positive_scalar_integer = (-1, 0, 1.1, 1j, np.ones(5), np.ones((5, 5)),
                                 'string')
not_a_string = (True, -1, 0, 1.1, 1j, np.ones(5), np.ones((5, 5)),  {'a': 2})


class TestDMRegPokeGridForAnyPupil(unittest.TestCase):
    """Test calc_dmreg_from_poke_grid_for_any_pupil() performance in sim."""

    def setUp(self):
        self.lam = 575e-9
        self.amp_unpoked = fits.getdata(os.path.join(
            DMREG_PATH, 'cgidata', 'amp_cgi_prnum_000168.fits'))
        self.fn_dmreg_def = os.path.join(
            DMREG_PATH, 'cgidata', 'ut_dmreg_def_sim_stress_tests.yaml')
        self.fn_dmreg_tuning_params = os.path.join(
            DMREG_PATH, 'cgidata', 'ut_dmreg_tuning_params_spc.yaml')
        self.xOffsetPupil = -4.690501039060732
        self.yOffsetPupil = -5.787285016875911
        self.diamPupil = 300.23

    def test_dmreg_poke_grid_inputs(self):
        """Test bad inputs for calc_dmreg_from_poke_grid()."""
        ph_diff_meas = self.amp_unpoked  # for right size
        delta_v = np.eye(48)
        which_dm = 1

        for bad_val in not_a_2d_array:
            with self.assertRaises(TypeError):
                calc_dmreg_from_poke_grid_for_any_pupil(
                    bad_val, ph_diff_meas, delta_v, which_dm,
                    self.lam, self.fn_dmreg_def, self.fn_dmreg_tuning_params,
                    self.diamPupil, self.xOffsetPupil, self.yOffsetPupil,
                )
        for bad_val in not_a_2d_array:
            with self.assertRaises(TypeError):
                calc_dmreg_from_poke_grid_for_any_pupil(
                    self.amp_unpoked, bad_val, delta_v, which_dm,
                    self.lam, self.fn_dmreg_def, self.fn_dmreg_tuning_params,
                    self.diamPupil, self.xOffsetPupil, self.yOffsetPupil,
                )
        for bad_val in not_a_2d_array:
            with self.assertRaises(TypeError):
                calc_dmreg_from_poke_grid_for_any_pupil(
                    self.amp_unpoked, ph_diff_meas, bad_val, which_dm,
                    self.lam, self.fn_dmreg_def, self.fn_dmreg_tuning_params,
                    self.diamPupil, self.xOffsetPupil, self.yOffsetPupil,
                )
        for bad_val in not_a_positive_scalar_integer:
            with self.assertRaises(TypeError):
                calc_dmreg_from_poke_grid_for_any_pupil(
                    self.amp_unpoked, ph_diff_meas, delta_v, bad_val,
                    self.lam, self.fn_dmreg_def, self.fn_dmreg_tuning_params,
                    self.diamPupil, self.xOffsetPupil, self.yOffsetPupil,
                )
        for bad_val in not_a_real_positive_scalar:
            with self.assertRaises(TypeError):
                calc_dmreg_from_poke_grid_for_any_pupil(
                    self.amp_unpoked, ph_diff_meas, delta_v, which_dm,
                    bad_val, self.fn_dmreg_def, self.fn_dmreg_tuning_params,
                    self.diamPupil, self.xOffsetPupil, self.yOffsetPupil,
                )
        for bad_val in not_a_string:
            with self.assertRaises(TypeError):
                calc_dmreg_from_poke_grid_for_any_pupil(
                    self.amp_unpoked, ph_diff_meas, delta_v, which_dm,
                    self.lam, bad_val, self.fn_dmreg_tuning_params,
                    self.diamPupil, self.xOffsetPupil, self.yOffsetPupil,
                )
        for bad_val in not_a_string:
            with self.assertRaises(TypeError):
                calc_dmreg_from_poke_grid_for_any_pupil(
                    self.amp_unpoked, ph_diff_meas, delta_v, which_dm,
                    self.lam, self.fn_dmreg_def, bad_val,
                    self.diamPupil, self.xOffsetPupil, self.yOffsetPupil,
                )
        for bad_val in not_a_real_positive_scalar:
            with self.assertRaises(TypeError):
                calc_dmreg_from_poke_grid_for_any_pupil(
                    self.amp_unpoked, ph_diff_meas, delta_v, which_dm,
                    self.lam, self.fn_dmreg_def, self.fn_dmreg_tuning_params,
                    bad_val, self.xOffsetPupil, self.yOffsetPupil,
                )
        for bad_val in not_a_real_scalar:
            with self.assertRaises(TypeError):
                calc_dmreg_from_poke_grid_for_any_pupil(
                    self.amp_unpoked, ph_diff_meas, delta_v, which_dm,
                    self.lam, self.fn_dmreg_def, self.fn_dmreg_tuning_params,
                    self.diamPupil, bad_val, self.yOffsetPupil,
                )
        for bad_val in not_a_real_scalar:
            with self.assertRaises(TypeError):
                calc_dmreg_from_poke_grid_for_any_pupil(
                    self.amp_unpoked, ph_diff_meas, delta_v, which_dm,
                    self.lam, self.fn_dmreg_def, self.fn_dmreg_tuning_params,
                    self.diamPupil, self.xOffsetPupil, bad_val,
                )

    def test_dmreg_poke_grid_for_any_pupil_start_close(self):
        """Test calc_dmreg_from_poke_grid_for_any_pupil() near best fit."""
        which_dm = 1
        delta_v = fits.getdata(os.path.join(
            DMREG_PATH, 'cgidata', 'dmrel_grid_for_dmreg_dm1.fits'))

        # Load a bunch of DM definition info needed for the model and for some
        # starting values for the optimization.
        path_to_dmreg_yaml = os.path.dirname(self.fn_dmreg_def)
        dmConfig = loadyaml(self.fn_dmreg_def)
        dm = dmConfig["dms"]

        if which_dm == 1:
            dmr = dm['DM1']['registration']
            gainmap = fits.getdata(os.path.join(
                path_to_dmreg_yaml, dm['DM1']['voltages']['gainfn']))
            fn_crosstalk_base = dm['DM1']['voltages']['crosstalkfn']
        elif which_dm == 2:
            dmr = dm['DM2']['registration']
            gainmap = fits.getdata(os.path.join(
                path_to_dmreg_yaml, dm['DM2']['voltages']['gainfn']))
            fn_crosstalk_base = dm['DM2']['voltages']['crosstalkfn']
        else:
            raise ValueError('which_dm must be either 1 or 2')
        
        # Cannot give None to os.path.join()
        if fn_crosstalk_base is not None:
            crosstalk_fn = os.path.join(path_to_dmreg_yaml, fn_crosstalk_base)
        else:
            crosstalk_fn = None

        # Convert voltage map to height map and account for crosstalk
        dh_map = volts_to_dmh(gainmap, delta_v, self.lam,
                              crosstalk_fn=crosstalk_fn)

        inf_func = fits.getdata(os.path.join(path_to_dmreg_yaml, dmr['inffn']))

        # Originally found values used for the regression test. The true
        # values are not known because this is from real data.
        ppact_cx_true = dmr['ppact_cx'] * 1.01
        ppact_cy_true = dmr['ppact_cy'] * 0.99
        dx_true = dmr['dx'] - 1.0
        dy_true = dmr['dy'] + 2.0
        thact_true = dmr['thact'] - 0.2

        # Compute the modeled phase change from the poke grid.
        im_shape = self.amp_unpoked.shape
        ph_diff_meas = 2 * dmhtoph(
            im_shape[0], im_shape[1], dh_map,
            dmr['nact'], inf_func, dmr['ppact_d'], ppact_cx_true,
            ppact_cy_true, dx_true, dy_true, thact_true, dmr['flipx'],
        )

        param_dict, map_dict = calc_dmreg_from_poke_grid_for_any_pupil(
            self.amp_unpoked, ph_diff_meas, delta_v, which_dm, self.lam,
            self.fn_dmreg_def, self.fn_dmreg_tuning_params,
            self.diamPupil, self.xOffsetPupil, self.yOffsetPupil,
        )

        self.assertTrue(np.abs(ppact_cx_true - param_dict['ppact_cx']) /
                        ppact_cx_true * 100 < 0.3)
        self.assertTrue(np.abs(ppact_cy_true - param_dict['ppact_cy']) /
                        ppact_cy_true * 100 < 0.3)
        self.assertTrue(np.abs(dx_true - param_dict['dx']) < 0.15)
        self.assertTrue(np.abs(dy_true - param_dict['dy']) < 0.15)
        self.assertTrue((thact_true-param_dict['thact'])*np.pi/180*1000 < 0.25)

    def test_dmreg_poke_grid_for_any_pupil_near_lower_bounds(self):
        """Test DMREG performance near lower bounds and for REQ 822909."""
        which_dm = 1
        delta_v = fits.getdata(os.path.join(
            DMREG_PATH, 'cgidata', 'dmrel_grid_for_dmreg_dm1.fits'))

        # Load a bunch of DM definition info needed for the model and for some
        # starting values for the optimization.
        path_to_dmreg_yaml = os.path.dirname(self.fn_dmreg_def)
        dmConfig = loadyaml(self.fn_dmreg_def)
        dm = dmConfig["dms"]

        if which_dm == 1:
            dmr = dm['DM1']['registration']
            gainmap = fits.getdata(os.path.join(
                path_to_dmreg_yaml, dm['DM1']['voltages']['gainfn']))
            fn_crosstalk_base = dm['DM1']['voltages']['crosstalkfn']
        elif which_dm == 2:
            dmr = dm['DM2']['registration']
            gainmap = fits.getdata(os.path.join(
                path_to_dmreg_yaml, dm['DM2']['voltages']['gainfn']))
            fn_crosstalk_base = dm['DM2']['voltages']['crosstalkfn']
        else:
            raise ValueError('which_dm must be either 1 or 2')
        
        # Cannot give None to os.path.join()
        if fn_crosstalk_base is not None:
            crosstalk_fn = os.path.join(path_to_dmreg_yaml, fn_crosstalk_base)
        else:
            crosstalk_fn = None

        # Convert voltage map to height map and account for crosstalk
        dh_map = volts_to_dmh(gainmap, delta_v, self.lam,
                              crosstalk_fn=crosstalk_fn)

        inf_func = fits.getdata(os.path.join(path_to_dmreg_yaml, dmr['inffn']))

        # Originally found values used for the regression test. The true
        # values are not known because this is from real data.
        ppact_cx_true = dmr['ppact_cx'] * 0.95
        ppact_cy_true = dmr['ppact_cy'] * 0.95
        dx_true = dmr['dx'] - 11.5
        dy_true = dmr['dy'] - 11.5
        thact_true = dmr['thact'] - 1.9

        # Compute the modeled phase change from the poke grid.
        im_shape = self.amp_unpoked.shape
        ph_diff_meas = 2 * dmhtoph(
            im_shape[0], im_shape[1], dh_map,
            dmr['nact'], inf_func, dmr['ppact_d'], ppact_cx_true,
            ppact_cy_true, dx_true, dy_true, thact_true, dmr['flipx'],
        )

        param_dict, map_dict = calc_dmreg_from_poke_grid_for_any_pupil(
            self.amp_unpoked, ph_diff_meas, delta_v, which_dm, self.lam,
            self.fn_dmreg_def, self.fn_dmreg_tuning_params,
            self.diamPupil, self.xOffsetPupil, self.yOffsetPupil,
        )

        self.assertTrue(np.abs(ppact_cx_true - param_dict['ppact_cx']) /
                        ppact_cx_true * 100 < 0.3)
        self.assertTrue(np.abs(ppact_cy_true - param_dict['ppact_cy']) /
                        ppact_cy_true * 100 < 0.3)
        self.assertTrue(np.abs(dx_true - param_dict['dx']) < 0.15)
        self.assertTrue(np.abs(dy_true - param_dict['dy']) < 0.15)
        self.assertTrue((thact_true-param_dict['thact'])*np.pi/180*1000 < 0.25)

        # Accuracy value set by REQ 822909 (TTF Offset)
        diamPupil = 300.23  # pixels
        maxAllowedErrorPercent = 0.1
        xOffsetPupilPR168 = -4.69
        yOffsetPupilPR168 = -5.79
        xOffsetPupilFromDM1True = xOffsetPupilPR168 - dx_true
        yOffsetPupilFromDM1True = yOffsetPupilPR168 - dy_true
        abs_tol = diamPupil * maxAllowedErrorPercent/100  # pixels
        self.assertTrue(np.abs(xOffsetPupilFromDM1True -
                               map_dict['xOffsetPupilFromDM']) < abs_tol)
        self.assertTrue(np.abs(yOffsetPupilFromDM1True -
                               map_dict['yOffsetPupilFromDM']) < abs_tol)

    def test_dmreg_poke_grid_for_any_pupil_near_upper_bounds(self):
        """Test calc_dmreg_from_poke_grid_for_any_pupil() near upper bounds."""
        which_dm = 2

        # Load a bunch of DM definition info needed for the model and for some
        # starting values for the optimization.
        path_to_dmreg_yaml = os.path.dirname(self.fn_dmreg_def)
        dmConfig = loadyaml(self.fn_dmreg_def)
        dm = dmConfig["dms"]

        if which_dm == 1:
            dmr = dm['DM1']['registration']
            gainmap = fits.getdata(os.path.join(
                path_to_dmreg_yaml, dm['DM1']['voltages']['gainfn']))
            fn_crosstalk_base = dm['DM1']['voltages']['crosstalkfn']
            delta_v = fits.getdata(os.path.join(
                DMREG_PATH, 'cgidata', 'dmrel_grid_for_dmreg_dm1.fits'))
        elif which_dm == 2:
            dmr = dm['DM2']['registration']
            gainmap = fits.getdata(os.path.join(
                path_to_dmreg_yaml, dm['DM2']['voltages']['gainfn']))
            fn_crosstalk_base = dm['DM2']['voltages']['crosstalkfn']
            delta_v = fits.getdata(os.path.join(
                DMREG_PATH, 'cgidata', 'dmrel_grid_for_dmreg_dm2.fits'))
        else:
            raise ValueError('which_dm must be either 1 or 2')
        
        # Cannot give None to os.path.join()
        if fn_crosstalk_base is not None:
            crosstalk_fn = os.path.join(path_to_dmreg_yaml, fn_crosstalk_base)
        else:
            crosstalk_fn = None

        # Convert voltage map to height map and account for crosstalk
        dh_map = volts_to_dmh(gainmap, delta_v, self.lam,
                              crosstalk_fn=crosstalk_fn)

        inf_func = fits.getdata(os.path.join(path_to_dmreg_yaml, dmr['inffn']))

        # Originally found values used for the regression test. The true
        # values are not known because this is from real data.
        ppact_cx_true = dmr['ppact_cx'] * 1.05
        ppact_cy_true = dmr['ppact_cy'] * 1.05
        dx_true = dmr['dx'] + 11.5
        dy_true = dmr['dy'] + 11.5
        thact_true = dmr['thact'] + 1.9

        # Compute the modeled phase change from the poke grid.
        im_shape = self.amp_unpoked.shape
        ph_diff_meas = 2 * dmhtoph(
            im_shape[0], im_shape[1], dh_map,
            dmr['nact'], inf_func, dmr['ppact_d'], ppact_cx_true,
            ppact_cy_true, dx_true, dy_true, thact_true, dmr['flipx'],
        )

        param_dict, map_dict = calc_dmreg_from_poke_grid_for_any_pupil(
            self.amp_unpoked, ph_diff_meas, delta_v, which_dm, self.lam,
            self.fn_dmreg_def, self.fn_dmreg_tuning_params,
            self.diamPupil, self.xOffsetPupil, self.yOffsetPupil,
        )

        self.assertTrue(np.abs(ppact_cx_true - param_dict['ppact_cx']) /
                        ppact_cx_true * 100 < 0.3)
        self.assertTrue(np.abs(ppact_cy_true - param_dict['ppact_cy']) /
                        ppact_cy_true * 100 < 0.3)
        self.assertTrue(np.abs(dx_true - param_dict['dx']) < 0.15)
        self.assertTrue(np.abs(dy_true - param_dict['dy']) < 0.15)
        self.assertTrue((thact_true-param_dict['thact'])*np.pi/180*1000 < 0.25)


class TestDMRegPokeGridForAnyPupilWithDataFromTVAC(unittest.TestCase):
    """
    Test dmreg.calc_dmreg_from_poke_grid_for_any_pupil() with II&T TVAC data
    with a shaped pupil in place.
    """

    def setUp(self):
        self.lam = 575e-9
        self.amp_unpoked = fits.getdata(os.path.join(
            DMREG_PATH, 'cgidata', 'amp_cgi_prnum_000330.fits'))
        self.ph_unpoked = fits.getdata(os.path.join(
            DMREG_PATH, 'cgidata', 'ph_cgi_prnum_000330.fits'))
        self.fn_dmreg_def = os.path.join(
            DMREG_PATH, 'cgidata', 'ut_howfsc_optical_model_tvac_wfov.yaml')
        self.fn_dmreg_tuning_params = os.path.join(
            DMREG_PATH, 'cgidata', 'ut_dmreg_tuning_params_spc.yaml')
        self.xOffsetPupil = -0.57
        self.yOffsetPupil = 0.38
        self.diamPupil = 300.23

    def test_dmreg_poke_grid_with_tvac_dm1_data(self):
        """Regression test using measured data during II&T TVAC for DM1."""
        which_dm = 1
        delta_v = fits.getdata(os.path.join(
            DMREG_PATH, 'cgidata', 'dmrel_wfov_poke_grid.fits'))
        ph_poked = fits.getdata(os.path.join(
            DMREG_PATH, 'cgidata', 'ph_cgi_prnum_000329.fits'))
        ph_diff_meas = ph_poked - self.ph_unpoked

        param_dict, map_dict = calc_dmreg_from_poke_grid_for_any_pupil(
            self.amp_unpoked, ph_diff_meas, delta_v, which_dm, self.lam,
            self.fn_dmreg_def, self.fn_dmreg_tuning_params,
            self.diamPupil, self.xOffsetPupil, self.yOffsetPupil,
        )

        # Originally found values used for the regression test. The true
        # values are not known because this is from real data.
        ppact_cx_true = 6.261
        ppact_cy_true = 6.339
        dx_true = -0.356
        dy_true = 0.046
        thact_true = 89.942
        
        print('DM1 registration: ')
        print(param_dict)

        self.assertTrue(np.abs(ppact_cx_true - param_dict['ppact_cx']) /
                        ppact_cx_true * 100 < 0.3)
        self.assertTrue(np.abs(ppact_cy_true - param_dict['ppact_cy']) /
                        ppact_cy_true * 100 < 0.3)
        self.assertTrue(np.abs(dx_true - param_dict['dx']) < 0.15)
        self.assertTrue(np.abs(dy_true - param_dict['dy']) < 0.15)
        self.assertTrue((thact_true-param_dict['thact'])*np.pi/180*1000 < 0.25)

    def test_dmreg_poke_grid_with_tvac_dm2_data(self):
        """Regression test using measured data during II&T TVAC for DM2."""
        which_dm = 2
        delta_v = fits.getdata(os.path.join(
            DMREG_PATH, 'cgidata', 'dmrel_wfov_poke_grid.fits'))
        ph_poked = fits.getdata(os.path.join(
            DMREG_PATH, 'cgidata', 'ph_cgi_prnum_000331.fits'))
        ph_diff_meas = ph_poked - self.ph_unpoked

        param_dict, map_dict = calc_dmreg_from_poke_grid_for_any_pupil(
            self.amp_unpoked, ph_diff_meas, delta_v, which_dm, self.lam,
            self.fn_dmreg_def, self.fn_dmreg_tuning_params,
            self.diamPupil, self.xOffsetPupil, self.yOffsetPupil,
        )
        
        # Originally found values used for the regression test. The true
        # values are not known because this is from real data.
        ppact_cx_true = 6.259
        ppact_cy_true = 6.364
        dx_true = -0.836
        dy_true = 1.036
        thact_true = 89.88

        self.assertTrue(np.abs(ppact_cx_true - param_dict['ppact_cx']) /
                        ppact_cx_true * 100 < 0.3)
        self.assertTrue(np.abs(ppact_cy_true - param_dict['ppact_cy']) /
                        ppact_cy_true * 100 < 0.3)
        self.assertTrue(np.abs(dx_true - param_dict['dx']) < 0.15)
        self.assertTrue(np.abs(dy_true - param_dict['dy']) < 0.15)
        self.assertTrue((thact_true-param_dict['thact'])*np.pi/180*1000 < 0.25)


class TestDMRegPokeGridForAnyPupilWithDataFromFFT(unittest.TestCase):
    """
    Test dmreg.calc_dmreg_from_poke_grid_for_any_pupil() with II&T FFT data
    for an unmasked pupil.
    """

    def setUp(self):
        self.lam = 575e-9
        self.amp_unpoked = fits.getdata(os.path.join(
            DMREG_PATH, 'cgidata', 'amp_cgi_prnum_000168.fits'))
        self.ph_unpoked = fits.getdata(os.path.join(
            DMREG_PATH, 'cgidata', 'ph_cgi_prnum_000168.fits'))
        self.fn_dmreg_def = os.path.join(
            DMREG_PATH, 'cgidata', 'ut_howfsc_optical_model_fft.yaml')
        self.fn_dmreg_tuning_params = os.path.join(
            DMREG_PATH, 'cgidata', 'ut_dmreg_tuning_params_spc.yaml')
        self.xOffsetPupil = -4.690501039060732
        self.yOffsetPupil = -5.787285016875911
        self.diamPupil = 300.2


    def test_dmreg_poke_grid_with_fft_dm1_data(self):
        """Regression test using measured data during II&T FFT for DM1."""
        which_dm = 1
        delta_v = fits.getdata(os.path.join(
            DMREG_PATH, 'cgidata', 'dmrel_grid_for_dmreg_dm1.fits'))
        ph_poked = fits.getdata(os.path.join(
            DMREG_PATH, 'cgidata', 'ph_cgi_prnum_000167.fits'))
        ph_diff_meas = ph_poked - self.ph_unpoked

        param_dict, map_dict = calc_dmreg_from_poke_grid_for_any_pupil(
            self.amp_unpoked, ph_diff_meas, delta_v, which_dm, self.lam,
            self.fn_dmreg_def, self.fn_dmreg_tuning_params,
            self.diamPupil, self.xOffsetPupil, self.yOffsetPupil,
        )

        # Originally found values used for the regression test. The true
        # values are not known because this is from real data.
        ppact_cx_true = 6.2665
        ppact_cy_true = 6.3653
        dx_true = -4.2289
        dy_true = -5.0957
        thact_true = 89.9176

        self.assertTrue(np.abs(ppact_cx_true - param_dict['ppact_cx']) /
                        ppact_cx_true * 100 < 0.3)
        self.assertTrue(np.abs(ppact_cy_true - param_dict['ppact_cy']) /
                        ppact_cy_true * 100 < 0.3)
        self.assertTrue(np.abs(dx_true - param_dict['dx']) < 0.15)
        self.assertTrue(np.abs(dy_true - param_dict['dy']) < 0.15)
        self.assertTrue((thact_true-param_dict['thact'])*np.pi/180*1000 < 0.25)

    def test_dmreg_poke_grid_with_fft_dm2_data(self):
        """Regression test using measured data during II&T FFT for DM2."""
        which_dm = 2
        delta_v = fits.getdata(os.path.join(
            DMREG_PATH, 'cgidata', 'dmrel_grid_for_dmreg_dm2.fits'))
        ph_poked = fits.getdata(os.path.join(
            DMREG_PATH, 'cgidata', 'ph_cgi_prnum_000169.fits'))
        ph_diff_meas = ph_poked - self.ph_unpoked

        param_dict, map_dict = calc_dmreg_from_poke_grid_for_any_pupil(
            self.amp_unpoked, ph_diff_meas, delta_v, which_dm, self.lam,
            self.fn_dmreg_def, self.fn_dmreg_tuning_params,
            self.diamPupil, self.xOffsetPupil, self.yOffsetPupil,
        )
        
        # Originally found values used for the regression test. The true
        # values are not known because this is from real data.
        ppact_cx_true = 6.2671
        ppact_cy_true = 6.3505
        dx_true = -4.8128
        dy_true = -5.1984
        thact_true = 89.9026

        self.assertTrue(np.abs(ppact_cx_true - param_dict['ppact_cx']) /
                        ppact_cx_true * 100 < 0.3)
        self.assertTrue(np.abs(ppact_cy_true - param_dict['ppact_cy']) /
                        ppact_cy_true * 100 < 0.3)
        self.assertTrue(np.abs(dx_true - param_dict['dx']) < 0.15)
        self.assertTrue(np.abs(dy_true - param_dict['dy']) < 0.15)
        self.assertTrue((thact_true-param_dict['thact'])*np.pi/180*1000 < 0.25)


class TestDMRegPokeGrid(unittest.TestCase):
    """Test dmreg.calc_dmreg_from_poke_grid() performance in simulation."""

    def setUp(self):
        self.lam = 575e-9
        self.amp_unpoked = fits.getdata(os.path.join(
            DMREG_PATH, 'cgidata', 'amp_cgi_prnum_000168.fits'))

        self.fn_dmreg_def = os.path.join(
            DMREG_PATH, 'cgidata', 'ut_dmreg_def_sim_stress_tests.yaml')
        self.fn_dmreg_tuning_params = os.path.join(
            DMREG_PATH, 'cgidata', 'ut_dmreg_tuning_params_unmasked_pupil.yaml')
        self.fn_pupil_fit_params = os.path.join(
            DMREG_PATH, 'cgidata', 'params_fit_unmasked_pupil_fft.yaml')
        self.data_path_pupil =  os.path.join(DMREG_PATH, 'cgidata')

    def test_dmreg_poke_grid_inputs(self):
        """Test bad inputs for calc_dmreg_from_poke_grid()."""
        
        ph_diff_meas = self.amp_unpoked  # for right size
        delta_v = np.eye(48)
        which_dm = 1

        for bad_val in not_a_2d_array:
            with self.assertRaises(TypeError):
                calc_dmreg_from_poke_grid(
                    bad_val, ph_diff_meas, delta_v, which_dm,
                    self.lam, self.fn_dmreg_def, self.fn_dmreg_tuning_params,
                    self.fn_pupil_fit_params, self.data_path_pupil,
                )
        for bad_val in not_a_2d_array:
            with self.assertRaises(TypeError):
                calc_dmreg_from_poke_grid(
                    self.amp_unpoked, bad_val, delta_v, which_dm,
                    self.lam, self.fn_dmreg_def, self.fn_dmreg_tuning_params,
                    self.fn_pupil_fit_params, self.data_path_pupil,
                )
        for bad_val in not_a_2d_array:
            with self.assertRaises(TypeError):
                calc_dmreg_from_poke_grid(
                    self.amp_unpoked, ph_diff_meas, bad_val, which_dm,
                    self.lam, self.fn_dmreg_def, self.fn_dmreg_tuning_params,
                    self.fn_pupil_fit_params, self.data_path_pupil,
                )
        for bad_val in not_a_positive_scalar_integer:
            with self.assertRaises(TypeError):
                calc_dmreg_from_poke_grid(
                    self.amp_unpoked, ph_diff_meas, delta_v, bad_val,
                    self.lam, self.fn_dmreg_def, self.fn_dmreg_tuning_params,
                    self.fn_pupil_fit_params, self.data_path_pupil,
                )
        for bad_val in not_a_real_positive_scalar:
            with self.assertRaises(TypeError):
                calc_dmreg_from_poke_grid(
                    self.amp_unpoked, ph_diff_meas, delta_v, which_dm,
                    bad_val, self.fn_dmreg_def, self.fn_dmreg_tuning_params,
                    self.fn_pupil_fit_params, self.data_path_pupil,
                )
        for bad_val in not_a_string:
            with self.assertRaises(TypeError):
                calc_dmreg_from_poke_grid(
                    self.amp_unpoked, ph_diff_meas, delta_v, which_dm,
                    self.lam, bad_val, self.fn_dmreg_tuning_params,
                    self.fn_pupil_fit_params, self.data_path_pupil,
                )
        for bad_val in not_a_string:
            with self.assertRaises(TypeError):
                calc_dmreg_from_poke_grid(
                    self.amp_unpoked, ph_diff_meas, delta_v, which_dm,
                    self.lam, self.fn_dmreg_def, bad_val,
                    self.fn_pupil_fit_params, self.data_path_pupil,
                )
        for bad_val in not_a_string:
            with self.assertRaises(TypeError):
                calc_dmreg_from_poke_grid(
                    self.amp_unpoked, ph_diff_meas, delta_v, which_dm,
                    self.lam, self.fn_dmreg_def, self.fn_dmreg_tuning_params,
                    bad_val, self.data_path_pupil,
                )
        for bad_val in not_a_string:
            with self.assertRaises(TypeError):
                calc_dmreg_from_poke_grid(
                    self.amp_unpoked, ph_diff_meas, delta_v, which_dm,
                    self.lam, self.fn_dmreg_def, self.fn_dmreg_tuning_params,
                    self.fn_pupil_fit_params, bad_val,
                )

    def test_dmreg_poke_grid_start_close(self):
        """Test calc_dmreg_from_poke_grid() performance near best fit."""
        which_dm = 1
        delta_v = fits.getdata(os.path.join(
            DMREG_PATH, 'cgidata', 'dmrel_grid_for_dmreg_dm1.fits'))

        # Load a bunch of DM definition info needed for the model and for some
        # starting values for the optimization.
        path_to_dmreg_yaml = os.path.dirname(self.fn_dmreg_def)
        dmConfig = loadyaml(self.fn_dmreg_def)
        dm = dmConfig["dms"]

        if which_dm == 1:
            dmr = dm['DM1']['registration']
            gainmap = fits.getdata(os.path.join(
                path_to_dmreg_yaml, dm['DM1']['voltages']['gainfn']))
            fn_crosstalk_base = dm['DM1']['voltages']['crosstalkfn']
        elif which_dm == 2:
            dmr = dm['DM2']['registration']
            gainmap = fits.getdata(os.path.join(
                path_to_dmreg_yaml, dm['DM2']['voltages']['gainfn']))
            fn_crosstalk_base = dm['DM2']['voltages']['crosstalkfn']
        else:
            raise ValueError('which_dm must be either 1 or 2')
        
        # Cannot give None to os.path.join()
        if fn_crosstalk_base is not None:
            crosstalk_fn = os.path.join(path_to_dmreg_yaml, fn_crosstalk_base)
        else:
            crosstalk_fn = None

        # Convert voltage map to height map and account for crosstalk
        dh_map = volts_to_dmh(gainmap, delta_v, self.lam,
                              crosstalk_fn=crosstalk_fn)

        inf_func = fits.getdata(os.path.join(path_to_dmreg_yaml, dmr['inffn']))

        # Originally found values used for the regression test. The true
        # values are not known because this is from real data.
        ppact_cx_true = dmr['ppact_cx'] * 1.01
        ppact_cy_true = dmr['ppact_cy'] * 0.99
        dx_true = dmr['dx'] - 1.0
        dy_true = dmr['dy'] + 2.0
        thact_true = dmr['thact'] - 0.2

        # Compute the modeled phase change from the poke grid.
        im_shape = self.amp_unpoked.shape
        ph_diff_meas = 2 * dmhtoph(
            im_shape[0], im_shape[1], dh_map,
            dmr['nact'], inf_func, dmr['ppact_d'], ppact_cx_true,
            ppact_cy_true, dx_true, dy_true, thact_true, dmr['flipx'],
        )

        param_dict, map_dict = calc_dmreg_from_poke_grid(
            self.amp_unpoked, ph_diff_meas, delta_v, which_dm, self.lam,
            self.fn_dmreg_def, self.fn_dmreg_tuning_params,
            self.fn_pupil_fit_params, self.data_path_pupil,
        )

        self.assertTrue(np.abs(ppact_cx_true - param_dict['ppact_cx']) /
                        ppact_cx_true * 100 < 0.3)
        self.assertTrue(np.abs(ppact_cy_true - param_dict['ppact_cy']) /
                        ppact_cy_true * 100 < 0.3)
        self.assertTrue(np.abs(dx_true - param_dict['dx']) < 0.15)
        self.assertTrue(np.abs(dy_true - param_dict['dy']) < 0.15)
        self.assertTrue((thact_true-param_dict['thact'])*np.pi/180*1000 < 0.25)

    def test_dmreg_poke_grid_near_lower_bounds(self):
        """Test DMREG performance near lower bounds and for REQ 822909."""
        which_dm = 1
        delta_v = fits.getdata(os.path.join(
            DMREG_PATH, 'cgidata', 'dmrel_grid_for_dmreg_dm1.fits'))

        # Load a bunch of DM definition info needed for the model and for some
        # starting values for the optimization.
        path_to_dmreg_yaml = os.path.dirname(self.fn_dmreg_def)
        dmConfig = loadyaml(self.fn_dmreg_def)
        dm = dmConfig["dms"]

        if which_dm == 1:
            dmr = dm['DM1']['registration']
            gainmap = fits.getdata(os.path.join(
                path_to_dmreg_yaml, dm['DM1']['voltages']['gainfn']))
            fn_crosstalk_base = dm['DM1']['voltages']['crosstalkfn']
        elif which_dm == 2:
            dmr = dm['DM2']['registration']
            gainmap = fits.getdata(os.path.join(
                path_to_dmreg_yaml, dm['DM2']['voltages']['gainfn']))
            fn_crosstalk_base = dm['DM2']['voltages']['crosstalkfn']
        else:
            raise ValueError('which_dm must be either 1 or 2')
        
        # Cannot give None to os.path.join()
        if fn_crosstalk_base is not None:
            crosstalk_fn = os.path.join(path_to_dmreg_yaml, fn_crosstalk_base)
        else:
            crosstalk_fn = None

        # Convert voltage map to height map and account for crosstalk
        dh_map = volts_to_dmh(gainmap, delta_v, self.lam,
                              crosstalk_fn=crosstalk_fn)

        inf_func = fits.getdata(os.path.join(path_to_dmreg_yaml, dmr['inffn']))

        # Originally found values used for the regression test. The true
        # values are not known because this is from real data.
        ppact_cx_true = dmr['ppact_cx'] * 0.95
        ppact_cy_true = dmr['ppact_cy'] * 0.95
        dx_true = dmr['dx'] - 11.5
        dy_true = dmr['dy'] - 11.5
        thact_true = dmr['thact'] - 1.9

        # Compute the modeled phase change from the poke grid.
        im_shape = self.amp_unpoked.shape
        ph_diff_meas = 2 * dmhtoph(
            im_shape[0], im_shape[1], dh_map,
            dmr['nact'], inf_func, dmr['ppact_d'], ppact_cx_true,
            ppact_cy_true, dx_true, dy_true, thact_true, dmr['flipx'],
        )

        param_dict, map_dict = calc_dmreg_from_poke_grid(
            self.amp_unpoked, ph_diff_meas, delta_v, which_dm, self.lam,
            self.fn_dmreg_def, self.fn_dmreg_tuning_params,
            self.fn_pupil_fit_params, self.data_path_pupil,
        )

        self.assertTrue(np.abs(ppact_cx_true - param_dict['ppact_cx']) /
                        ppact_cx_true * 100 < 0.3)
        self.assertTrue(np.abs(ppact_cy_true - param_dict['ppact_cy']) /
                        ppact_cy_true * 100 < 0.3)
        self.assertTrue(np.abs(dx_true - param_dict['dx']) < 0.15)
        self.assertTrue(np.abs(dy_true - param_dict['dy']) < 0.15)
        self.assertTrue((thact_true-param_dict['thact'])*np.pi/180*1000 < 0.25)

        # Accuracy value set by REQ 822909 (TTF Offset)
        diamPupil = 300.2  # pixels
        maxAllowedErrorPercent = 0.1
        xOffsetPupilPR168 = -4.69
        yOffsetPupilPR168 = -5.79
        xOffsetPupilFromDM1True = xOffsetPupilPR168 - dx_true
        yOffsetPupilFromDM1True = yOffsetPupilPR168 - dy_true
        abs_tol = diamPupil * maxAllowedErrorPercent/100  # pixels
        self.assertTrue(np.abs(xOffsetPupilFromDM1True -
                               map_dict['xOffsetPupilFromDM']) < abs_tol)
        self.assertTrue(np.abs(yOffsetPupilFromDM1True -
                               map_dict['yOffsetPupilFromDM']) < abs_tol)


    def test_dmreg_poke_grid_near_upper_bounds(self):
        """Test calc_dmreg_from_poke_grid() performance near upper bounds."""
        which_dm = 2

        # Load a bunch of DM definition info needed for the model and for some
        # starting values for the optimization.
        path_to_dmreg_yaml = os.path.dirname(self.fn_dmreg_def)
        dmConfig = loadyaml(self.fn_dmreg_def)
        dm = dmConfig["dms"]

        if which_dm == 1:
            dmr = dm['DM1']['registration']
            gainmap = fits.getdata(os.path.join(
                path_to_dmreg_yaml, dm['DM1']['voltages']['gainfn']))
            fn_crosstalk_base = dm['DM1']['voltages']['crosstalkfn']
            delta_v = fits.getdata(os.path.join(
                DMREG_PATH, 'cgidata', 'dmrel_grid_for_dmreg_dm1.fits'))
        elif which_dm == 2:
            dmr = dm['DM2']['registration']
            gainmap = fits.getdata(os.path.join(
                path_to_dmreg_yaml, dm['DM2']['voltages']['gainfn']))
            fn_crosstalk_base = dm['DM2']['voltages']['crosstalkfn']
            delta_v = fits.getdata(os.path.join(
                DMREG_PATH, 'cgidata', 'dmrel_grid_for_dmreg_dm2.fits'))
        else:
            raise ValueError('which_dm must be either 1 or 2')
        
        # Cannot give None to os.path.join()
        if fn_crosstalk_base is not None:
            crosstalk_fn = os.path.join(path_to_dmreg_yaml, fn_crosstalk_base)
        else:
            crosstalk_fn = None

        # Convert voltage map to height map and account for crosstalk
        dh_map = volts_to_dmh(gainmap, delta_v, self.lam,
                              crosstalk_fn=crosstalk_fn)

        inf_func = fits.getdata(os.path.join(path_to_dmreg_yaml, dmr['inffn']))

        # Originally found values used for the regression test. The true
        # values are not known because this is from real data.
        ppact_cx_true = dmr['ppact_cx'] * 1.05
        ppact_cy_true = dmr['ppact_cy'] * 1.05
        dx_true = dmr['dx'] + 11.5
        dy_true = dmr['dy'] + 11.5
        thact_true = dmr['thact'] + 1.9

        # Compute the modeled phase change from the poke grid.
        im_shape = self.amp_unpoked.shape
        ph_diff_meas = 2 * dmhtoph(
            im_shape[0], im_shape[1], dh_map,
            dmr['nact'], inf_func, dmr['ppact_d'], ppact_cx_true,
            ppact_cy_true, dx_true, dy_true, thact_true, dmr['flipx'],
        )

        param_dict, map_dict = calc_dmreg_from_poke_grid(
            self.amp_unpoked, ph_diff_meas, delta_v, which_dm, self.lam,
            self.fn_dmreg_def, self.fn_dmreg_tuning_params,
            self.fn_pupil_fit_params, self.data_path_pupil,
        )

        self.assertTrue(np.abs(ppact_cx_true - param_dict['ppact_cx']) /
                        ppact_cx_true * 100 < 0.3)
        self.assertTrue(np.abs(ppact_cy_true - param_dict['ppact_cy']) /
                        ppact_cy_true * 100 < 0.3)
        self.assertTrue(np.abs(dx_true - param_dict['dx']) < 0.15)
        self.assertTrue(np.abs(dy_true - param_dict['dy']) < 0.15)
        self.assertTrue((thact_true-param_dict['thact'])*np.pi/180*1000 < 0.25)


class TestDMRegPokeGridIIT(unittest.TestCase):
    """Test dmreg.calc_dmreg_from_poke_grid() with II&T FFT data."""

    def setUp(self):
        self.lam = 575e-9
        self.amp_unpoked = fits.getdata(os.path.join(
            DMREG_PATH, 'cgidata', 'amp_cgi_prnum_000168.fits'))
        self.ph_unpoked = fits.getdata(os.path.join(
            DMREG_PATH, 'cgidata', 'ph_cgi_prnum_000168.fits'))

        self.fn_dmreg_def = os.path.join(
            DMREG_PATH, 'cgidata', 'ut_howfsc_optical_model_fft.yaml')
        self.fn_dmreg_tuning_params = os.path.join(
            DMREG_PATH, 'cgidata', 'ut_dmreg_tuning_params_unmasked_pupil.yaml')
        self.fn_pupil_fit_params = os.path.join(
            DMREG_PATH, 'cgidata', 'params_fit_unmasked_pupil_fft.yaml')
        self.data_path_pupil =  os.path.join(DMREG_PATH, 'cgidata')

    def test_dmreg_poke_grid_with_fft_dm1_data(self):
        """Regression test using measured data during II&T FFT for DM1."""
        which_dm = 1
        delta_v = fits.getdata(os.path.join(
            DMREG_PATH, 'cgidata', 'dmrel_grid_for_dmreg_dm1.fits'))
        ph_poked = fits.getdata(os.path.join(
            DMREG_PATH, 'cgidata', 'ph_cgi_prnum_000167.fits'))
        ph_diff_meas = ph_poked - self.ph_unpoked

        param_dict, map_dict = calc_dmreg_from_poke_grid(
            self.amp_unpoked, ph_diff_meas, delta_v, which_dm, self.lam,
            self.fn_dmreg_def, self.fn_dmreg_tuning_params,
            self.fn_pupil_fit_params, self.data_path_pupil,
        )
        
        # Originally found values used for the regression test. The true
        # values are not known because this is from real data.
        ppact_cx_true = 6.2665
        ppact_cy_true = 6.3653
        dx_true = -4.2289
        dy_true = -5.0957
        thact_true = 89.9176

        self.assertTrue(np.abs(ppact_cx_true - param_dict['ppact_cx']) /
                        ppact_cx_true * 100 < 0.3)
        self.assertTrue(np.abs(ppact_cy_true - param_dict['ppact_cy']) /
                        ppact_cy_true * 100 < 0.3)
        self.assertTrue(np.abs(dx_true - param_dict['dx']) < 0.15)
        self.assertTrue(np.abs(dy_true - param_dict['dy']) < 0.15)
        self.assertTrue((thact_true-param_dict['thact'])*np.pi/180*1000 < 0.25)

    def test_dmreg_poke_grid_with_fft_dm2_data(self):
        """Regression test using measured data during II&T FFT for DM2."""
        which_dm = 2
        delta_v = fits.getdata(os.path.join(
            DMREG_PATH, 'cgidata', 'dmrel_grid_for_dmreg_dm2.fits'))
        ph_poked = fits.getdata(os.path.join(
            DMREG_PATH, 'cgidata', 'ph_cgi_prnum_000169.fits'))
        ph_diff_meas = ph_poked - self.ph_unpoked

        param_dict, map_dict = calc_dmreg_from_poke_grid(
            self.amp_unpoked, ph_diff_meas, delta_v, which_dm, self.lam,
            self.fn_dmreg_def, self.fn_dmreg_tuning_params,
            self.fn_pupil_fit_params, self.data_path_pupil,
        )
        
        # Originally found values used for the regression test. The true
        # values are not known because this is from real data.
        ppact_cx_true = 6.2671
        ppact_cy_true = 6.3505
        dx_true = -4.8128
        dy_true = -5.1984
        thact_true = 89.9026

        self.assertTrue(np.abs(ppact_cx_true - param_dict['ppact_cx']) /
                        ppact_cx_true * 100 < 0.3)
        self.assertTrue(np.abs(ppact_cy_true - param_dict['ppact_cy']) /
                        ppact_cy_true * 100 < 0.3)
        self.assertTrue(np.abs(dx_true - param_dict['dx']) < 0.15)
        self.assertTrue(np.abs(dy_true - param_dict['dy']) < 0.15)
        self.assertTrue((thact_true-param_dict['thact'])*np.pi/180*1000 < 0.25)


class TestSoftwareMask(unittest.TestCase):
    """Test dm_software_mask."""

    def test_calc_software_mask_shape(self):
        """Test that output shape of calc_software_mask is correct."""
        nrow = 10
        ncol = 11
        amp0 = inin(np.ones((nrow-2, ncol-2)), (nrow, ncol))
        ph0 = np.zeros((nrow, ncol))
        amp1 = inin(np.ones((nrow-2, ncol-2)), (nrow, ncol))
        ph1 = np.ones((nrow, ncol))
        whichDM = 1
        nErode = 1
        deltaPhaseThreshRad = 0.5
        boolMask = calc_software_mask(amp0, ph0, amp1, ph1, whichDM, nErode,
                                      deltaPhaseThreshRad)
        self.assertEqual(boolMask.shape, amp0.shape)

    def test_calc_software_mask_output_values(self):
        """Test that output values of calc_software_mask are correct."""
        nrow = 21
        ncol = 30
        amp0 = inin(np.ones((nrow-2, ncol-2)), (nrow, ncol))
        ph0 = np.zeros((nrow, ncol))
        amp1 = inin(np.ones((nrow-2, ncol-2)), (nrow, ncol))
        ph1 = np.ones((nrow, ncol))
        whichDM = 1
        nErode = 1
        deltaPhaseThreshRad = 0.1
        boolMask = calc_software_mask(amp0, ph0, amp1, ph1, whichDM, nErode,
                                      deltaPhaseThreshRad)
        answer = inin(np.ones((nrow-4, ncol-4)), (nrow, ncol)).astype(int)
        self.assertTrue(np.sum(np.abs(boolMask.astype(int) - answer)) == 0)

        # Test again with a large phase deviation in one area.
        ph1[9:11, 8] = 10  # alter a 2x1 block, which eroded becomes 12 pixels
        boolMask = calc_software_mask(amp0, ph0, amp1, ph1, whichDM, nErode,
                                      deltaPhaseThreshRad)
        boolMask = boolMask.astype(int)
        self.assertTrue(np.sum(np.abs(boolMask - answer))-12 == 0)


class TestTranslationRotation(unittest.TestCase):
    """Test calc_translation_clocking."""

    def test_calc_translation_clocking(self):
        """Test clocking error <0.25 mrad & translation error <0.1 pixels."""
        localpath = os.path.dirname(os.path.abspath(__file__))

        dm_fn = 'testdata/ut_dm_for_dmreg.yaml'
        fn_fit_params = 'testdata/ut_trans_rot_settings.yaml'
        xOffsetPupil = 90
        yOffsetPupil = -33
        lam = 575e-9

        # This is the same DM command used to make amp1 and ph1: a 2-striped +
        fn_deltaV = os.path.join(localpath, 'cgidata',
                                 'delta_DM_V_for_trans_rot_calib.fits')
        deltaV = fits.getdata(fn_deltaV)

        for whichDM in (1, 2):
            fn_amp0 = os.path.join(localpath, 'testdata',
                                   ('ut_transrot_dm%d_amp0.fits' % whichDM))
            fn_amp1 = os.path.join(localpath, 'testdata',
                                   ('ut_transrot_dm%d_amp1.fits' % whichDM))
            fn_ph0 = os.path.join(localpath, 'testdata',
                                  ('ut_transrot_dm%d_ph0.fits' % whichDM))
            fn_ph1 = os.path.join(localpath, 'testdata',
                                  ('ut_transrot_dm%d_ph1.fits' % whichDM))
            amp0 = fits.getdata(fn_amp0)
            amp1 = fits.getdata(fn_amp1)
            # Test data were generated incorrectly as surface instead of
            # as wavefront, so multiply by 2.
            ph0 = 2*fits.getdata(fn_ph0)
            ph1 = 2*fits.getdata(fn_ph1)
            fnBoolMap = os.path.join(
                localpath, 'testdata', ('ut_usablePixelMap_DM%d_2BadAct.fits' %
                                        whichDM))
            usablePixelMap = fits.getdata(fnBoolMap).astype(bool)

            xOffsetEst, yOffsetEst, clockEst = calc_translation_clocking(
                amp0, ph0, amp1, ph1, usablePixelMap, whichDM, deltaV,
                xOffsetPupil, yOffsetPupil, dm_fn, fn_fit_params, lam
            )
            xOffsetTrue = 86.2
            yOffsetTrue = -30.9
            clockTrue = 1.5

            self.assertTrue(np.abs(xOffsetTrue - xOffsetEst) < 0.15)
            self.assertTrue(np.abs(yOffsetTrue - yOffsetEst) < 0.15)
            self.assertTrue((clockEst-clockTrue)*np.pi/180*1000 < 0.25)


class TestScale(unittest.TestCase):
    """Test calc_scale."""

    def test_calc_scale(self):
        """Test that the pixels per actuator estimates are correct to <0.3%."""
        localpath = os.path.dirname(os.path.abspath(__file__))

        # Same DM command used to make amp1 and ph1: a square outline
        fn_deltaV = os.path.join(localpath, 'cgidata',
                                 'delta_DM_V_for_scale_calib.fits')
        deltaV = fits.getdata(fn_deltaV)

        dm_fn = 'testdata/ut_dm_for_dmreg.yaml'
        fn_fit_params = 'testdata/ut_scale_settings.yaml'
        lam = 575e-9
        xOffset = 86.2
        yOffset = -30.9
        clocking = 1.5

        for whichDM in (1, 2):
            fn_amp0 = os.path.join(localpath, 'testdata',
                                   ('ut_scale_dm%d_amp0.fits' % whichDM))
            fn_amp1 = os.path.join(localpath, 'testdata',
                                   ('ut_scale_dm%d_amp1.fits' % whichDM))
            fn_ph0 = os.path.join(localpath, 'testdata',
                                  ('ut_scale_dm%d_ph0.fits' % whichDM))
            fn_ph1 = os.path.join(localpath, 'testdata',
                                  ('ut_scale_dm%d_ph1.fits' % whichDM))
            amp0 = fits.getdata(fn_amp0)
            amp1 = fits.getdata(fn_amp1)
            # Test data were generated incorrectly as surface instead of
            # as wavefront, so multiply by 2.
            ph0 = 2*fits.getdata(fn_ph0)
            ph1 = 2*fits.getdata(fn_ph1)
            fnBoolMap = os.path.join(
                localpath, 'testdata',
                ('ut_usablePixelMap_DM%d_2BadAct.fits' % whichDM))
            usablePixelMap = fits.getdata(fnBoolMap).astype(bool)

            ppact_cx_est, ppact_cy_est = \
                calc_scale(amp0, ph0, amp1, ph1, usablePixelMap, whichDM,
                           deltaV, dm_fn, fn_fit_params, lam,
                           xOffset, yOffset, clocking)
            ppact_cx_true = 6.83
            ppact_cy_true = 6.54
            self.assertTrue(np.abs(ppact_cx_true - ppact_cx_est) /
                            ppact_cx_true * 100 < 0.3)
            self.assertTrue(np.abs(ppact_cy_true - ppact_cy_est) /
                            ppact_cy_true * 100 < 0.3)


class TestInputFailure(unittest.TestCase):
    """Test suite for valid function inputs."""

    def test_inputs_calc_software_mask(self):
        """Test the inputs of calc_software_mask."""
        nrow = 10
        ncol = 11
        amp0 = inin(np.ones((nrow-2, ncol-2)), (nrow, ncol))
        ph0 = np.zeros((nrow, ncol))
        amp1 = inin(np.ones((nrow-2, ncol-2)), (nrow, ncol))
        ph1 = np.ones((nrow, ncol))
        whichDM = 1
        nErode = 1
        deltaPhaseThreshRad = 0.5

        # Check standard inputs do not raise anything first
        calc_software_mask(amp0, ph0, amp1, ph1, whichDM, nErode,
                           deltaPhaseThreshRad)

        for amp0Bad in (-1, 1, 1.1, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(TypeError):
                calc_software_mask(amp0Bad, ph0, amp1, ph1, whichDM, nErode,
                                   deltaPhaseThreshRad)
        for ph0Bad in (-1, 1, 1.1, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(TypeError):
                calc_software_mask(amp0, ph0Bad, amp1, ph1, whichDM, nErode,
                                   deltaPhaseThreshRad)
        for amp1Bad in (-1, 1, 1.1, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(TypeError):
                calc_software_mask(amp0, ph0, amp1Bad, ph1, whichDM, nErode,
                                   deltaPhaseThreshRad)
        for ph1Bad in (-1, 1, 1.1, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(TypeError):
                calc_software_mask(amp0, ph0, amp1, ph1Bad, whichDM, nErode,
                                   deltaPhaseThreshRad)
        for whichDMBad in (-1, 0, 1.1, 1j, np.ones((1, )), 'string'):
            with self.assertRaises(TypeError):
                calc_software_mask(amp0, ph0, amp1, ph1, whichDMBad, nErode,
                                   deltaPhaseThreshRad)
        for whichDMBad in (3, 4):
            with self.assertRaises(ValueError):
                calc_software_mask(amp0, ph0, amp1, ph1, whichDMBad, nErode,
                                   deltaPhaseThreshRad)
        for nErodeBad in (-1, 1.1, 1j, np.ones(1), np.ones((4, 5)), 'string'):
            with self.assertRaises(TypeError):
                calc_software_mask(amp0, ph0, amp1, ph1, whichDM, nErodeBad,
                                   deltaPhaseThreshRad)
        for deltaPhaseThreshRadBad in (-1, 0, 1j, np.ones(1),
                                       np.ones((4, 5)), 'string'):
            with self.assertRaises(TypeError):
                calc_software_mask(amp0, ph0, amp1, ph1, whichDM, nErode,
                                   deltaPhaseThreshRadBad)

        # Shape equivalency tests for amp0, ph0, amp1, ph1
        wrong2D = np.zeros((8, 9))
        with self.assertRaises(TypeError):
            calc_software_mask(wrong2D, ph0, amp1, ph1, whichDM, nErode,
                               deltaPhaseThreshRad)
        with self.assertRaises(TypeError):
            calc_software_mask(amp0, wrong2D, amp1, ph1, whichDM, nErode,
                               deltaPhaseThreshRad)
        with self.assertRaises(TypeError):
            calc_software_mask(amp0, ph0, wrong2D, ph1, whichDM, nErode,
                               deltaPhaseThreshRad)
        with self.assertRaises(TypeError):
            calc_software_mask(amp0, ph0, amp1, wrong2D, whichDM, nErode,
                               deltaPhaseThreshRad)

    def test_inputs_calc_translation_clocking(self):
        """Test the inputs of calc_translation_clocking."""
        localpath = os.path.dirname(os.path.abspath(__file__))

        whichDM = 1
        fn_amp0 = os.path.join(localpath, 'testdata', 'ut_scale_dm1_amp0.fits')
        fn_amp1 = os.path.join(localpath, 'testdata', 'ut_scale_dm1_amp1.fits')
        fn_ph0 = os.path.join(localpath, 'testdata', 'ut_scale_dm1_ph0.fits')
        fn_ph1 = os.path.join(localpath, 'testdata', 'ut_scale_dm1_ph1.fits')
        amp0 = fits.getdata(fn_amp0)
        amp1 = fits.getdata(fn_amp1)
        ph0 = fits.getdata(fn_ph0)
        ph1 = fits.getdata(fn_ph1)
        fnBoolMap = os.path.join(
            localpath, 'testdata', 'ut_usablePixelMap_DM1_2BadAct.fits')
        usablePixelMap = fits.getdata(fnBoolMap).astype(bool)
        dm_fn = 'testdata/ut_dm_for_dmreg.yaml'
        fn_fit_params = 'testdata/ut_trans_rot_settings_quick_run.yaml'
        xOffsetPupil = 0
        yOffsetPupil = 0
        lam = 575e-9

        nact = 48
        x = np.arange(-(nact-1)/2, (nact+1)/2)/nact*2
        [X, Y] = np.meshgrid(x, x)
        deltaV = np.zeros((nact, nact))
        deltaV[int(nact/2. - 1):int(nact/2.), :] = 1
        deltaV[:, int(nact/2. - 1):int(nact/2.)] = 1
        deltaV = 5*deltaV

        nact = 48
        x = np.arange(-(nact-1)/2, (nact+1)/2)/nact*2
        [X, Y] = np.meshgrid(x, x)
        deltaV = np.zeros((nact, nact))
        deltaV[int(nact/2. - 1):int(nact/2.), :] = 1
        deltaV[:, int(nact/2. - 1):int(nact/2.)] = 1
        deltaV = 5*deltaV

        # Check standard inputs do not raise anything first
        calc_translation_clocking(amp0, ph0, amp1, ph1, usablePixelMap,
                          whichDM, deltaV, xOffsetPupil, yOffsetPupil,
                          dm_fn, fn_fit_params, lam)

        # Bad input tests
        for amp0Bad in (-1, 1, 1.1, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(TypeError):
                calc_translation_clocking(amp0Bad, ph0, amp1, ph1,
                          usablePixelMap, whichDM, deltaV, xOffsetPupil,
                          yOffsetPupil, dm_fn, fn_fit_params, lam)
        for ph0Bad in (-1, 1, 1.1, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(TypeError):
                calc_translation_clocking(amp0, ph0Bad, amp1, ph1,
                          usablePixelMap, whichDM, deltaV, xOffsetPupil,
                          yOffsetPupil, dm_fn, fn_fit_params, lam)
        for amp1Bad in (-1, 1, 1.1, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(TypeError):
                calc_translation_clocking(amp0, ph0, amp1Bad, ph1,
                          usablePixelMap, whichDM, deltaV, xOffsetPupil,
                          yOffsetPupil, dm_fn, fn_fit_params, lam)
        for ph1Bad in (-1, 1, 1.1, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(TypeError):
                calc_translation_clocking(amp0, ph0, amp1, ph1Bad,
                          usablePixelMap, whichDM, deltaV, xOffsetPupil,
                          yOffsetPupil, dm_fn, fn_fit_params, lam)
        for usablePixelMapBad in (-1, 1, 1.1, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(TypeError):
                calc_translation_clocking(amp0, ph0, amp1, ph1,
                          usablePixelMapBad, whichDM, deltaV, xOffsetPupil,
                          yOffsetPupil, dm_fn, fn_fit_params, lam)

        for whichDMBad in (-1, 0, 1.1, 1j, np.ones((1, )), 'string'):
            with self.assertRaises(TypeError):
                calc_translation_clocking(amp0, ph0, amp1, ph1,
                          usablePixelMap, whichDMBad, deltaV, xOffsetPupil,
                          yOffsetPupil, dm_fn, fn_fit_params, lam)
        for whichDMBad in (3, 4):
            with self.assertRaises(ValueError):
                calc_translation_clocking(amp0, ph0, amp1, ph1,
                          usablePixelMap, whichDMBad, deltaV, xOffsetPupil,
                          yOffsetPupil, dm_fn, fn_fit_params, lam)

        for deltaVBad in (-1, 1, 1.1, 1j, np.ones(5),
                          np.ones((47, 48)), 'string'):
            with self.assertRaises(TypeError):
                calc_translation_clocking(amp0, ph0, amp1, ph1,
                          usablePixelMap, whichDM, deltaVBad, xOffsetPupil,
                          yOffsetPupil, dm_fn, fn_fit_params, lam)
        for xOffsetBad in (1j, np.ones(1), np.ones((4, 5)), 'string'):
            with self.assertRaises(TypeError):
                calc_translation_clocking(amp0, ph0, amp1, ph1,
                          usablePixelMap, whichDM, deltaV, xOffsetBad,
                          yOffsetPupil, dm_fn, fn_fit_params, lam)
        for yOffsetBad in (1j, np.ones((4, 5)), 'string'):
            with self.assertRaises(TypeError):
                calc_translation_clocking(amp0, ph0, amp1, ph1,
                          usablePixelMap, whichDM, deltaV, xOffsetPupil,
                          yOffsetBad, dm_fn, fn_fit_params, lam)
        for lamBad in (-1, 0, 1j, np.ones(1), np.ones((4, 5)), 'string'):
            with self.assertRaises(TypeError):
                calc_translation_clocking(amp0, ph0, amp1, ph1,
                          usablePixelMap, whichDM, deltaV, xOffsetPupil,
                          yOffsetPupil, dm_fn, fn_fit_params, lamBad)
        for pathbad in (-1, 0, 1.1, 1j, np.ones((1, ))):
            with self.assertRaises(TypeError):
                calc_translation_clocking(amp0, ph0, amp1, ph1,
                          usablePixelMap, whichDM, deltaV, xOffsetPupil,
                          yOffsetPupil, dm_fn, fn_fit_params, lam,
                          data_path=pathbad)

        # Shape equivalency tests for amp0, ph0, amp1, ph1, usablePixelMap
        wrong2D = np.zeros((8, 9))
        with self.assertRaises(TypeError):
            calc_translation_clocking(wrong2D, ph0, amp1, ph1,
                          usablePixelMap, whichDM, deltaV, xOffsetPupil,
                          yOffsetPupil, dm_fn, fn_fit_params, lam)
        with self.assertRaises(TypeError):
            calc_translation_clocking(amp0, wrong2D, amp1, ph1,
                          usablePixelMap, whichDM, deltaV, xOffsetPupil,
                          yOffsetPupil, dm_fn, fn_fit_params, lam)
        with self.assertRaises(TypeError):
            calc_translation_clocking(amp0, ph0, wrong2D, ph1,
                          usablePixelMap, whichDM, deltaV, xOffsetPupil,
                          yOffsetPupil, dm_fn, fn_fit_params, lam)
        with self.assertRaises(TypeError):
            calc_translation_clocking(amp0, ph0, amp1, wrong2D,
                          usablePixelMap, whichDM, deltaV, xOffsetPupil,
                          yOffsetPupil, dm_fn, fn_fit_params, lam)
        with self.assertRaises(TypeError):
            calc_translation_clocking(amp0, ph0, amp1, ph1,
                          wrong2D, whichDM, deltaV, xOffsetPupil,
                          yOffsetPupil, dm_fn, fn_fit_params, lam)

    def test_inputs_calc_scale(self):
        """Test the inputs of calc_scale."""
        localpath = os.path.dirname(os.path.abspath(__file__))

        whichDM = 1
        fn_amp0 = os.path.join(localpath, 'testdata', 'ut_scale_dm1_amp0.fits')
        fn_amp1 = os.path.join(localpath, 'testdata', 'ut_scale_dm1_amp1.fits')
        fn_ph0 = os.path.join(localpath, 'testdata', 'ut_scale_dm1_ph0.fits')
        fn_ph1 = os.path.join(localpath, 'testdata', 'ut_scale_dm1_ph1.fits')
        amp0 = fits.getdata(fn_amp0)
        amp1 = fits.getdata(fn_amp1)
        ph0 = fits.getdata(fn_ph0)
        ph1 = fits.getdata(fn_ph1)
        fnBoolMap = os.path.join(localpath, 'testdata',
                                 'ut_usablePixelMap_DM1_2BadAct.fits')
        usablePixelMap = fits.getdata(fnBoolMap).astype(bool)

        nact = 48
        x = np.arange(-(nact-1)/2, (nact+1)/2)/nact*2
        [X, Y] = np.meshgrid(x, x)
        deltaV = np.zeros((nact, nact))
        deltaV[int(nact/2. - 1):int(nact/2.), :] = 1
        deltaV[:, int(nact/2. - 1):int(nact/2.)] = 1
        deltaV = 5*deltaV

        dm_fn = 'testdata/ut_dm_for_dmreg.yaml'
        fn_fit_params = 'testdata/ut_scale_settings_quick_run.yaml'

        lam = 575e-9
        xOffset = 90
        yOffset = -33
        clocking = 1.5

        # Check standard inputs do not raise anything first
        calc_scale(amp0, ph0, amp1, ph1, usablePixelMap, whichDM, deltaV,
                   dm_fn, fn_fit_params, lam, xOffset, yOffset, clocking)

        # Bad input tests
        for amp0Bad in (-1, 1, 1.1, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(TypeError):
                calc_scale(amp0Bad, ph0, amp1, ph1, usablePixelMap, whichDM,
                           deltaV, dm_fn, fn_fit_params, lam,
                           xOffset, yOffset, clocking)
        for ph0Bad in (-1, 1, 1.1, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(TypeError):
                calc_scale(amp0, ph0Bad, amp1, ph1, usablePixelMap, whichDM,
                           deltaV, dm_fn, fn_fit_params, lam,
                        xOffset, yOffset, clocking)
        for amp1Bad in (-1, 1, 1.1, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(TypeError):
                calc_scale(amp0, ph0, amp1Bad, ph1, usablePixelMap, whichDM,
                           deltaV, dm_fn, fn_fit_params, lam,
                           xOffset, yOffset, clocking)
        for ph1Bad in (-1, 1, 1.1, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(TypeError):
                calc_scale(amp0, ph0, amp1, ph1Bad, usablePixelMap, whichDM,
                           deltaV, dm_fn, fn_fit_params, lam,
                           xOffset, yOffset, clocking)
        for usablePixelMapBad in (-1, 1, 1.1, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(TypeError):
                calc_scale(amp0, ph0, amp1, ph1, usablePixelMapBad, whichDM,
                           deltaV, dm_fn, fn_fit_params, lam,
                           xOffset, yOffset, clocking)
        for whichDMBad in (-1, 0, 1.1, 1j, np.ones((1, )), 'string'):
            with self.assertRaises(TypeError):
                calc_scale(amp0, ph0, amp1, ph1, usablePixelMap, whichDMBad,
                           deltaV, dm_fn, fn_fit_params, lam,
                           xOffset, yOffset, clocking)
        for whichDMBad in (3, 4):
            with self.assertRaises(ValueError):
                calc_scale(amp0, ph0, amp1, ph1, usablePixelMap, whichDMBad,
                           deltaV, dm_fn, fn_fit_params, lam,
                           xOffset, yOffset, clocking)
        for deltaVBad in (-1, 1, 1.1, 1j, np.ones(5),
                          np.ones((47, 48)), 'string'):
            with self.assertRaises(TypeError):
                calc_scale(amp0, ph0, amp1, ph1, usablePixelMap, whichDM,
                           deltaVBad, dm_fn, fn_fit_params, lam,
                           xOffset, yOffset, clocking)
        for lamBad in (-1, 0, 1j, np.ones(1), np.ones((4, 5)), 'string'):
            with self.assertRaises(TypeError):
                calc_scale(amp0, ph0, amp1, ph1, usablePixelMap, whichDM,
                           deltaV, dm_fn, fn_fit_params, lamBad,
                           xOffset, yOffset, clocking)
        for xOffsetBad in (1j, np.ones(1), np.ones((4, 5)), 'string'):
            with self.assertRaises(TypeError):
                calc_scale(amp0, ph0, amp1, ph1, usablePixelMap, whichDM,
                           deltaV, dm_fn, fn_fit_params, lam,
                           xOffsetBad, yOffset, clocking)
        for yOffsetBad in (1j, np.ones((4, 5)), 'string'):
            with self.assertRaises(TypeError):
                calc_scale(amp0, ph0, amp1, ph1, usablePixelMap, whichDM,
                           deltaV, dm_fn, fn_fit_params, lam,
                           xOffset, yOffsetBad, clocking)
        for clockingBad in (1j, np.ones(1), np.ones((4, 5)), 'string'):
            with self.assertRaises(TypeError):
                calc_scale(amp0, ph0, amp1, ph1, usablePixelMap, whichDM,
                           deltaV, dm_fn, fn_fit_params, lam,
                           xOffset, yOffset, clockingBad)
        for pathbad in (-1, 0, 1.1, 1j, np.ones((1, ))):
            with self.assertRaises(TypeError):
                calc_scale(amp0, ph0, amp1, ph1, usablePixelMap, whichDM,
                           deltaV, dm_fn, fn_fit_params, lam,
                           xOffset, yOffset, clocking, data_path=pathbad)

        # Shape equivalency tests for amp0, ph0, amp1, ph1, usablePixelMap
        wrong2D = np.zeros((8, 9))
        with self.assertRaises(TypeError):
            calc_scale(wrong2D, ph0, amp1, ph1, usablePixelMap, whichDM,
                       deltaV, dm_fn, fn_fit_params, lam,
                       xOffset, yOffset, clocking)
        with self.assertRaises(TypeError):
            calc_scale(amp0, wrong2D, amp1, ph1, usablePixelMap, whichDM,
                       deltaV, dm_fn, fn_fit_params, lam,
                       xOffset, yOffset, clocking)
        with self.assertRaises(TypeError):
            calc_scale(amp0, ph0, wrong2D, ph1, usablePixelMap, whichDM,
                       deltaV, dm_fn, fn_fit_params, lam,
                       xOffset, yOffset, clocking)
        with self.assertRaises(TypeError):
            calc_scale(amp0, ph0, amp1, wrong2D, usablePixelMap, whichDM,
                       deltaV, dm_fn, fn_fit_params, lam,
                       xOffset, yOffset, clocking)
        with self.assertRaises(TypeError):
            calc_scale(amp0, ph0, amp1, ph1, wrong2D, whichDM,
                       deltaV, dm_fn, fn_fit_params, lam,
                       xOffset, yOffset, clocking)


if __name__ == '__main__':
    unittest.main()
