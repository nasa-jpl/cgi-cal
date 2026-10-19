# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Unit tests for the CONJUGATE module."""
import unittest
import os
from math import isclose
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

from cal.gainmap import gm_util as gmu
from cal.util.nollzernikes import xyzern
from cal.util.insertinto import insertinto as inin
from cal.util.constrain_dm import constrain_dm
from cal.util.dm_crosstalk import CDMCrosstalk
from cal.util.dmhtoph import dmhtoph, volts_to_dmh, dmh_to_volts
from cal.util.loadyaml import loadyaml
from cal.util.math import ceil_even, rms
from cal.conjugate.conjugate import (
    flatten, gen_split_wfe, smooth_surface, conv_surf_to_dm_cmd,
    phase_cost_function
)
from cal.conjugate.util.propcustom import gen_zernikes
from cal.ampthresh.ampthresh import ampthresh

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
CAL_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NONLIN_PATH = os.path.join(CAL_PATH, 'gainmap', 'testdata', 'FFT_data')
DEBUG = False


class TestFlattenJsNonlin(unittest.TestCase):
    """Test functionality and inputs of flatten() with nonlinear DMs."""

    def setUp(self):
        """Initialize variables used in all the functional tests."""
        self.localpath = os.path.dirname(os.path.abspath(__file__))
        self.dm_fn = os.path.join(self.localpath, 'testdata', 'dm_rot_offset_scaled_fft.yaml')
        self.fcm_fn = os.path.join(self.localpath, 'testdata', 'fcm.yaml')
        self.flatten_params_fn = os.path.join(
            self.localpath, 'testdata', 'fixed_inputs_for_conjugate.yaml')

        # pupil calibration data
        self.diamPupil = 386.
        self.xOffset = 0.
        self.yOffset = 0.

        # Load FCM Calibration Data
        fcmConfig = loadyaml(self.fcm_fn)
        self.focusPerStroke = fcmConfig["focusPerStroke"]
        self.strokePerStep = fcmConfig["strokePerStep"]

        # Load fixed settings for CONJUGATE
        flattenDict = loadyaml(self.flatten_params_fn)
        arrayExtFac = flattenDict['arrayExtFac']
        self.nArray = ceil_even(arrayExtFac*self.diamPupil)

        # Load DM Calibration Data to get "nact" and make total voltage maps
        # self.dm_fn = os.path.join(self.localpath, 'testdata', 'dm.yaml')
        dmConfig = loadyaml(self.dm_fn)
        dm = dmConfig["dms"]
        self.dm = dm
        dm1r = dm['DM1']['registration']
        dm2r = dm['DM2']['registration']
        self.dm1r = dm1r
        self.dm2r = dm2r
        dm1gainmap_fn = os.path.join(self.localpath,
                                      dm['DM1']['voltages']['gainfn'])
        self.dm1gainmap = fits.getdata(dm1gainmap_fn)
        dm2gainmap_fn = os.path.join(self.localpath,
                                      dm['DM2']['voltages']['gainfn'])
        self.dm2gainmap = fits.getdata(dm2gainmap_fn)

        Vdm1In = 40.*np.ones((dm1r['nact'], dm1r['nact']))
        Vdm2In = 40.*np.ones((dm2r['nact'], dm2r['nact']))
        self.cmdFCMIn = 1000

        data_path = LOCAL_PATH
        self.Vdm1In = constrain_dm(
            Vdm1In, fits.getdata(os.path.join(
                data_path, dm['DM1']['voltages']['flatfn'])),
            fits.getdata(os.path.join(
                data_path, dm['DM1']['voltages']['tiefn'])),
            vmax=dm['DM1']['voltages']['vmax'],
            vlat=dm['DM1']['voltages']['vneighbor'],
            vdiag=dm['DM1']['voltages']['vcorner'],
            vquant=dm['DM1']['voltages']['vquant'],
        )
        
        self.Vdm2In = constrain_dm(Vdm2In,
            fits.getdata(
                os.path.join(data_path,
                         dm['DM2']['voltages']['flatfn'])
            ),
            fits.getdata(
                os.path.join(data_path,
                         dm['DM2']['voltages']['tiefn'])
            ),
            vmax=dm['DM2']['voltages']['vmax'],
            vlat=dm['DM2']['voltages']['vneighbor'],
            vdiag=dm['DM2']['voltages']['vcorner'],
            vquant=dm['DM2']['voltages']['vquant'],
        )
        

        self.lam = 575e-9  # used only for height-to-phase or vice-versa
        self.dm1inffn = os.path.join(self.localpath, dm1r['inffn'])
        self.dm2inffn = os.path.join(self.localpath, dm2r['inffn'])

        self.fnRefCubeDM1 = os.path.join(
            NONLIN_PATH, 'height_cube_integrated.fits')
        self.fnRefCommandVecDM1 = os.path.join(
            NONLIN_PATH, 'height_cube_command_vec.fits')
        self.fnRefCubeDM2 = os.path.join(
            NONLIN_PATH, 'height_cube_integrated_dm2.fits')
        self.fnRefCommandVecDM2 = os.path.join(
            NONLIN_PATH, 'height_cube_command_vec_dm2.fits')
            
            
        fn_delta_v = os.path.join(self.localpath, 'testdata',
                                  'dmrel_for_parity.fits')
        deltaV1 = fits.getdata(fn_delta_v)
        deltaV1 /= np.max(deltaV1)
        deltaV1 *= 27  # empirically found to normalize the peak phase
        
        dm = self.dm
        data_path = LOCAL_PATH
        Vdm1In = self.Vdm1In
        Vdm2In = self.Vdm2In

        Vdm10 = Vdm1In + deltaV1 # 40V + "Js"

        self.Vdm10 = constrain_dm(Vdm10,
            fits.getdata(
                os.path.join(data_path,
                             dm['DM1']['voltages']['flatfn'])
            ),
            fits.getdata(
                os.path.join(data_path,
                             dm['DM1']['voltages']['tiefn'])
            ),
            vmax=dm['DM1']['voltages']['vmax'],
            vlat=dm['DM1']['voltages']['vneighbor'],
            vdiag=dm['DM1']['voltages']['vcorner'],
            vquant=dm['DM1']['voltages']['vquant'],
        )

    def test_flatten_j_both_dms_nonlin(self):
        """Flatten the wavefront for an offset, rotated maps of J's."""
        flagUseFCM = False
        flagUseDMs = 'BOTH'
        flagHighZern = True

        Vdm1In = self.Vdm1In
        Vdm2In = self.Vdm2In
        Vdm10 = self.Vdm10

        # In: absolute DMV1 and DMV2, out: delta height(DMV2) - height(DMV1)
        # In: Volts, Out: meters
        dm1_h_in = gmu.compute_delta_height_map_from_command_maps(
            Vdm1In, Vdm10, self.fnRefCubeDM1,
            self.fnRefCommandVecDM1, crosstalk_fn=None)
        # meters to radians (surface height)
        dm1_h_in = 2.0*np.pi*dm1_h_in/self.lam
        # dm1_map_phase = volts_to_dmh(self.dm1gainmap, deltaV1, self.lam) # In: Volts, Out: radians

        nArray = 450

        # dmhtoph(): In: actuator pokes (radians), Out: surface (radians)
        dm1r = self.dm1r
        dm2r = self.dm2r
        DM1SurfIn = dmhtoph(
            nArray, nArray, dm1_h_in, dm1r['nact'],
            fits.getdata(self.dm1inffn), dm1r['ppact_d'],
            dm1r['ppact_cx'], dm1r['ppact_cy'], dm1r['dx'],
            dm1r['dy'], dm1r['thact'], dm1r['flipx'])
        dm1_phase = DM1SurfIn * 2
        dm1_amp = np.ones_like(dm1_phase)

        Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
            Vdm10, Vdm2In, self.cmdFCMIn, flagUseDMs, flagUseFCM,
            flagHighZern, self.lam, dm1_amp, dm1_phase,
            self.diamPupil, self.xOffset, self.yOffset,
            self.dm_fn, self.fcm_fn, self.flatten_params_fn,
            fnRefCubeDM1=self.fnRefCubeDM1,
            fnRefCommandVecDM1=self.fnRefCommandVecDM1,
            fnRefCubeDM2=self.fnRefCubeDM2,
            fnRefCommandVecDM2=self.fnRefCommandVecDM2,
        )

        # delta height DMV(flatten output) - DMV(flatten input) # In: Volts, Out: meters
        dm1_dht = gmu.compute_delta_height_map_from_command_maps(
            Vdm10, Vdm1Out, self.fnRefCubeDM1,
            self.fnRefCommandVecDM1, crosstalk_fn=None)
        dm2_dht = gmu.compute_delta_height_map_from_command_maps(
            Vdm2In, Vdm2Out, self.fnRefCubeDM2,
            self.fnRefCommandVecDM2, crosstalk_fn=None)

        # meters to radians (surface height)
        dm1_dht *= 2.0*np.pi/self.lam
        DM1SurfOut = dmhtoph(
            nArray, nArray, dm1_dht, dm1r['nact'],
            fits.getdata(self.dm1inffn), dm1r['ppact_d'],
            dm1r['ppact_cx'], dm1r['ppact_cy'], dm1r['dx'],
            dm1r['dy'], dm1r['thact'], dm1r['flipx'])

        dm2_dht *= 2.0*np.pi/self.lam
        DM2SurfOut = dmhtoph(
            nArray, nArray, dm2_dht, dm2r['nact'],
            fits.getdata(self.dm2inffn), dm2r['ppact_d'],
            dm2r['ppact_cx'], dm2r['ppact_cy'], dm2r['dx'],
            dm2r['dy'], dm2r['thact'], dm2r['flipx'])

        # Review:
        #   DM1SurfOut, DM2SurfOut (radians)
        #   DM1SurfIn (radians)
        # Surface out + Surface In = piston (radians)
        residual_dm1 = DM1SurfOut + DM1SurfIn
        rmsErrorDM1 = np.sqrt(np.mean(residual_dm1**2)) # rad
        rmsErrorDM2 = np.sqrt(np.mean(DM2SurfOut**2)) # rad

        if DEBUG:
            print('CONJUGATE: DM1 RMS WFE residual = %.3g rad' % rmsErrorDM1)
            print('CONJUGATE: DM2 RMS WFE residual = %.3g rad' % rmsErrorDM2)
    
            plt.figure(1)
            plt.imshow(DM1SurfIn)
            plt.title('Input DM1 Delta Surface (rad)')
            plt.colorbar()
            plt.gca().invert_yaxis()
            
            plt.figure(2)
            plt.imshow(DM1SurfOut)
            plt.title('Output DM1 Delta Surface (rad)')
            plt.colorbar()
            plt.gca().invert_yaxis()
            
            plt.figure(3)
            plt.imshow(DM1SurfIn + DM1SurfOut)
            plt.title('Surface Sum for DM1, rms residual = %.3e rad'%rmsErrorDM1)
            plt.colorbar()
            plt.gca().invert_yaxis()

            plt.figure(28)
            plt.imshow(Vdm1Out - Vdm1In)
            plt.title('Vdm1Out - Vdm1In Volts')
            plt.colorbar()
            plt.gca().invert_yaxis()

            plt.figure(29)
            plt.imshow(Vdm1Out - Vdm10)
            plt.title('Vdm1Out - Vdm10 Volts')
            plt.colorbar()
            plt.gca().invert_yaxis()
            
            plt.figure(30)
            plt.imshow(Vdm10 - Vdm1In)
            plt.title('Vdm10 - Vdm1In Volts')
            plt.colorbar()
            plt.gca().invert_yaxis()

            plt.figure(31)
            plt.imshow(Vdm1In)
            plt.title('Vdm1In Volts')
            plt.colorbar()
            plt.gca().invert_yaxis()
            
            plt.figure(32)
            plt.imshow(Vdm10)
            plt.title('Vdm10 Volts')
            plt.colorbar()
            plt.gca().invert_yaxis()

            plt.show()
    
        thresh = 5e-9  # meters
        self.assertTrue(rmsErrorDM1*self.lam/(2.0*np.pi) < thresh)
        self.assertTrue(rmsErrorDM2*self.lam/(2.0*np.pi) < thresh)


    def test_flatten_j_DM1_only_nonlin(self):
        """Flatten the wavefront for an offset, rotated maps of J's."""
        flagUseFCM = False
        flagUseDMs = 'DM1'
        flagHighZern = True

        fn_delta_v = os.path.join(self.localpath, 'testdata',
                                  'dmrel_for_parity.fits')
        deltaV1 = fits.getdata(fn_delta_v)
        deltaV1 /= np.max(deltaV1)
        deltaV1 *= 27  # empirically found to normalize the peak phase

        Vdm1In = self.Vdm1In
        Vdm2In = self.Vdm2In      
        Vdm10 = self.Vdm10

        # In: absolute DMV1 and DMV2, out: delta height(DMV2) - height(DMV1)
        # In: Volts, Out: meters
        dm1_h_in = gmu.compute_delta_height_map_from_command_maps(
            Vdm1In, Vdm10, self.fnRefCubeDM1,
            self.fnRefCommandVecDM1, crosstalk_fn=None)
        # meters to radians (surface height)
        dm1_h_in = 2.0*np.pi*dm1_h_in/self.lam
        # dm1_map_phase = volts_to_dmh(self.dm1gainmap, deltaV1, self.lam) # In: Volts, Out: radians

        nArray = 450

        # dmhtoph(): In: actuator pokes (radians), Out: surface (radians)
        dm1r = self.dm1r
        dm2r = self.dm2r
        DM1SurfIn = dmhtoph(
            nArray, nArray, dm1_h_in, dm1r['nact'],
            fits.getdata(self.dm1inffn), dm1r['ppact_d'],
            dm1r['ppact_cx'], dm1r['ppact_cy'], dm1r['dx'],
            dm1r['dy'], dm1r['thact'], dm1r['flipx'])
        dm1_phase = DM1SurfIn * 2
        dm1_amp = np.ones_like(dm1_phase)

        Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
            Vdm10, Vdm2In, self.cmdFCMIn, flagUseDMs, flagUseFCM,
            flagHighZern, self.lam, dm1_amp, dm1_phase,
            self.diamPupil, self.xOffset, self.yOffset,
            self.dm_fn, self.fcm_fn, self.flatten_params_fn,
            fnRefCubeDM1=self.fnRefCubeDM1,
            fnRefCommandVecDM1=self.fnRefCommandVecDM1,
            fnRefCubeDM2=self.fnRefCubeDM2,
            fnRefCommandVecDM2=self.fnRefCommandVecDM2,
        )

        # delta height DMV(flatten output) - DMV(flatten input) # In: Volts, Out: meters
        dm1_dht = gmu.compute_delta_height_map_from_command_maps(
            Vdm10, Vdm1Out, self.fnRefCubeDM1,
            self.fnRefCommandVecDM1, crosstalk_fn=None)
        dm2_dht = gmu.compute_delta_height_map_from_command_maps(
            Vdm2In, Vdm2Out, self.fnRefCubeDM2,
            self.fnRefCommandVecDM2, crosstalk_fn=None)

        # meters to radians (surface height)
        dm1_dht *= 2.0*np.pi/self.lam
        DM1SurfOut = dmhtoph(
            nArray, nArray, dm1_dht, dm1r['nact'],
            fits.getdata(self.dm1inffn), dm1r['ppact_d'],
            dm1r['ppact_cx'], dm1r['ppact_cy'], dm1r['dx'],
            dm1r['dy'], dm1r['thact'], dm1r['flipx'])

        dm2_dht *= 2.0*np.pi/self.lam
        DM2SurfOut = dmhtoph(
            nArray, nArray, dm2_dht, dm2r['nact'],
            fits.getdata(self.dm2inffn), dm2r['ppact_d'],
            dm2r['ppact_cx'], dm2r['ppact_cy'], dm2r['dx'],
            dm2r['dy'], dm2r['thact'], dm2r['flipx'])

        # Review:
        #   DM1SurfOut, DM2SurfOut (radians)
        #   DM1SurfIn (radians)
        # Surface out + Surface In = piston (radians)
        residual_dm1 = DM1SurfOut + DM1SurfIn
        rmsErrorDM1 = np.sqrt(np.mean(residual_dm1**2)) # rad
        rmsErrorDM2 = np.sqrt(np.mean(DM2SurfOut**2)) # rad
    
        thresh = 5e-9  # meters
        self.assertTrue(rmsErrorDM1*self.lam/(2.0*np.pi) < thresh)
        self.assertTrue(isclose(rmsErrorDM2, 0.0))


class TestFlattenJs(unittest.TestCase):
    """Test functionality and inputs of flatten()."""

    def setUp(self):
        """Initialize variables used in all the functional tests."""
        self.localpath = os.path.dirname(os.path.abspath(__file__))
        self.dm_fn = os.path.join(self.localpath, 'testdata', 'dm_rot_offset_scaled.yaml')
        self.fcm_fn = os.path.join(self.localpath, 'testdata', 'fcm.yaml')
        self.flatten_params_fn = os.path.join(
            self.localpath, 'testdata', 'fixed_inputs_for_conjugate.yaml')

        # pupil calibration data
        self.diamPupil = 386.
        self.xOffset = 0.
        self.yOffset = 0.

        # Load FCM Calibration Data
        fcmConfig = loadyaml(self.fcm_fn)
        self.focusPerStroke = fcmConfig["focusPerStroke"]
        self.strokePerStep = fcmConfig["strokePerStep"]

        # Load fixed settings for CONJUGATE
        flattenDict = loadyaml(self.flatten_params_fn)
        arrayExtFac = flattenDict['arrayExtFac']
        self.nArray = ceil_even(arrayExtFac*self.diamPupil)

        # Load DM Calibration Data to get "nact" and make total voltage maps
        # self.dm_fn = os.path.join(self.localpath, 'testdata', 'dm.yaml')
        dmConfig = loadyaml(self.dm_fn)
        dm = dmConfig["dms"]
        dm1r = dm['DM1']['registration']
        dm2r = dm['DM2']['registration']
        self.dm1r = dm1r
        self.dm2r = dm2r
        dm1gainmap_fn = os.path.join(self.localpath,
                                      dm['DM1']['voltages']['gainfn'])
        self.dm1gainmap = fits.getdata(dm1gainmap_fn)
        dm2gainmap_fn = os.path.join(self.localpath,
                                      dm['DM2']['voltages']['gainfn'])
        self.dm2gainmap = fits.getdata(dm2gainmap_fn)

        self.Vdm1In = 50.*np.ones((dm1r['nact'], dm1r['nact']))
        self.Vdm2In = 50.*np.ones((dm2r['nact'], dm2r['nact']))
        self.cmdFCMIn = 1000

        self.lam = 575e-9  # used only for height-to-phase or vice-versa
        self.dm1inffn = os.path.join(self.localpath, dm1r['inffn'])
        self.dm2inffn = os.path.join(self.localpath, dm2r['inffn'])

        ph_fn = os.path.join(self.localpath, 'testdata',
                        'ph_cgi_entrance_phase_b_N386_radians_575nm.fits')
        amp_fn = os.path.join(self.localpath, 'testdata',
                        'amp_cgi_entrance_phase_b_N386_radians_575nm.fits')
        self.ph = fits.getdata(ph_fn)
        self.amp = fits.getdata(amp_fn)

        self.flagUseFCM = True
        self.flagUseDMs = True
        self.flagHighZern = True

        self.dm2_real_crosstalk_fn =  os.path.join(
            self.localpath, 'testdata',
            'dm2_crosstalk_correct_orientation.yaml')

    def test_conv_surf_to_dm_cmd(self):
        """Test conv_surf_to_dm_cmd."""
        localpath = os.path.dirname(os.path.abspath(__file__))

        # The fit isn't as good at the last actuator, so don't include
        dVROI = inin(np.ones((46, 46)), (48, 48))
        lam = 575e-9

        fn_delta_v = os.path.join(self.localpath, 'testdata',
                                  'dmrel_for_parity.fits')
        deltaV1 = fits.getdata(fn_delta_v)
        deltaV1 /= np.max(deltaV1)
        deltaV1 *= 27  # empirically found to normalize the peak phase
        dm1_map_phase = volts_to_dmh(self.dm1gainmap, deltaV1, self.lam)

        nArray = 400

        dm1r = self.dm1r
        DM1SurfIn = dmhtoph(
            nArray, nArray, dm1_map_phase, dm1r['nact'],
            fits.getdata(self.dm1inffn), dm1r['ppact_d'],
            dm1r['ppact_cx'], dm1r['ppact_cy'], dm1r['dx'],
            dm1r['dy'], dm1r['thact'], dm1r['flipx'])

        # Load DM Calibration Data
        # dm_fn = os.path.join(localpath, 'testdata', 'dm_offset.yaml')
        dmConfig = loadyaml(self.dm_fn)
        dm = dmConfig["dms"]
        dm1r = dm['DM1']['registration']
        dm1gainmap_fn = os.path.join(localpath,
                                     dm['DM1']['voltages']['gainfn'])
        dm1gainmap = fits.getdata(dm1gainmap_fn)
        dm1phase = volts_to_dmh(dm1gainmap, dVROI, lam)
        dm1inffn = os.path.join(localpath, dm1r['inffn'])
        dm1r['inf_func'] = fits.getdata(dm1inffn)

        # Compute delta DM settings
        hdm1 = conv_surf_to_dm_cmd(DM1SurfIn, dm['DM1'])
        dV = dmh_to_volts(dm1gainmap, hdm1, lam)
        dm1phase = volts_to_dmh(dm1gainmap, dV, lam)
        dm1r.pop('inffn')
        DM1SurfOut = dmhtoph(DM1SurfIn.shape[0], DM1SurfIn.shape[1], dm1phase, **dm1r)

        # RMS error is dominated by the edges of the WFE map
        diff = (DM1SurfOut - DM1SurfIn)  # [radians]
        diffRMS = rms(diff)
        # print('conv_surf_to_dm_cmd: DM2 RMS WFE residual = %.3g rad' % diffRMS)
        self.assertTrue(diffRMS < 0.04)

    def test_flatten_j_both_dms(self):
        """Flatten the wavefront for an offset, rotated maps of J's."""
        flagUseFCM = False
        flagUseDMs = True
        flagHighZern = True

        fn_delta_v = os.path.join(self.localpath, 'testdata',
                                  'dmrel_for_parity.fits')
        deltaV1 = fits.getdata(fn_delta_v)
        deltaV1 /= np.max(deltaV1)
        deltaV1 *= 27  # empirically found to normalize the peak phase
        dm1_map_phase = volts_to_dmh(self.dm1gainmap, deltaV1, self.lam)

        nArray = 400

        dm1r = self.dm1r
        dm2r = self.dm2r
        DM1SurfIn = dmhtoph(
            nArray, nArray, dm1_map_phase, dm1r['nact'],
            fits.getdata(self.dm1inffn), dm1r['ppact_d'],
            dm1r['ppact_cx'], dm1r['ppact_cy'], dm1r['dx'],
            dm1r['dy'], dm1r['thact'], dm1r['flipx'])
        dm1_phase = DM1SurfIn * 2
        dm1_amp = np.ones_like(dm1_phase)

        Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
            self.Vdm1In, self.Vdm2In, self.cmdFCMIn, flagUseDMs, flagUseFCM,
            flagHighZern, self.lam, dm1_amp, dm1_phase,
            self.diamPupil, self.xOffset, self.yOffset,
            self.dm_fn, self.fcm_fn, self.flatten_params_fn)

        deltaV1 = Vdm1Out-self.Vdm1In
        deltaV2 = Vdm2Out-self.Vdm2In
        dm1phase = volts_to_dmh(self.dm1gainmap, deltaV1, self.lam)
        dm2phase = volts_to_dmh(self.dm2gainmap, deltaV2, self.lam)

        DM1SurfOut = dmhtoph(
            nArray, nArray, dm1phase, dm1r['nact'],
            fits.getdata(self.dm1inffn), dm1r['ppact_d'],
            dm1r['ppact_cx'], dm1r['ppact_cy'], dm1r['dx'],
            dm1r['dy'], dm1r['thact'], dm1r['flipx'])

        DM2SurfOut = dmhtoph(
            nArray, nArray, dm2phase, dm2r['nact'],
            fits.getdata(self.dm2inffn), dm2r['ppact_d'],
            dm2r['ppact_cx'], dm2r['ppact_cy'], dm2r['dx'],
            dm2r['dy'], dm2r['thact'], dm2r['flipx'])

        residual_dm1 = DM1SurfOut + DM1SurfIn
        rmsErrorDM1 = np.sqrt(np.mean(residual_dm1**2))
        rmsErrorDM2 = np.sqrt(np.mean(DM2SurfOut**2))

        # print('CONJUGATE: DM1 RMS WFE residual = %.3g rad' % rmsErrorDM1)
        # print('CONJUGATE: DM2 RMS WFE residual = %.3g rad' % rmsErrorDM2)
        self.assertTrue(rmsErrorDM1 < 0.05)
        self.assertTrue(rmsErrorDM2 < 0.05)

    def test_flatten_j_dm1_only(self):
        """Flatten the wavefront of J's with just DM1."""
        flagUseFCM = False
        flagUseDMs = 'DM1'
        flagHighZern = True

        fn_delta_v = os.path.join(self.localpath, 'testdata',
                                  'dmrel_for_parity.fits')
        deltaV1 = fits.getdata(fn_delta_v)
        deltaV1 /= np.max(deltaV1)
        deltaV1 *= 27  # empirically found to normalize the peak phase
        dm1_map_phase = volts_to_dmh(self.dm1gainmap, deltaV1, self.lam)

        nArray = 400

        dm1r = self.dm1r
        dm2r = self.dm2r
        DM1SurfIn = dmhtoph(
            nArray, nArray, dm1_map_phase, dm1r['nact'],
            fits.getdata(self.dm1inffn), dm1r['ppact_d'],
            dm1r['ppact_cx'], dm1r['ppact_cy'], dm1r['dx'],
            dm1r['dy'], dm1r['thact'], dm1r['flipx'])
        dm1_phase = DM1SurfIn * 2
        dm1_amp = np.ones_like(dm1_phase)

        Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
            self.Vdm1In, self.Vdm2In, self.cmdFCMIn, flagUseDMs, flagUseFCM,
            flagHighZern, self.lam, dm1_amp, dm1_phase,
            self.diamPupil, self.xOffset, self.yOffset,
            self.dm_fn, self.fcm_fn, self.flatten_params_fn)

        deltaV1 = Vdm1Out-self.Vdm1In
        deltaV2 = Vdm2Out-self.Vdm2In
        dm1phase = volts_to_dmh(self.dm1gainmap, deltaV1, self.lam)
        dm2phase = volts_to_dmh(self.dm2gainmap, deltaV2, self.lam)

        DM1SurfOut = dmhtoph(
            nArray, nArray, dm1phase, dm1r['nact'],
            fits.getdata(self.dm1inffn), dm1r['ppact_d'],
            dm1r['ppact_cx'], dm1r['ppact_cy'], dm1r['dx'],
            dm1r['dy'], dm1r['thact'], dm1r['flipx'])

        DM2SurfOut = dmhtoph(
            nArray, nArray, dm2phase, dm2r['nact'],
            fits.getdata(self.dm2inffn), dm2r['ppact_d'],
            dm2r['ppact_cx'], dm2r['ppact_cy'], dm2r['dx'],
            dm2r['dy'], dm2r['thact'], dm2r['flipx'])

        residual_dm1 = DM1SurfOut + DM1SurfIn
        rmsErrorDM1 = np.sqrt(np.mean(residual_dm1**2))
        rmsErrorDM2 = np.sqrt(np.mean(DM2SurfOut**2))

        self.assertTrue(rmsErrorDM1 < 0.05)
        self.assertTrue(rmsErrorDM2 < 0.05)

    def test_flatten_j_dm2_only(self):
        """Flatten the wavefront of J's with just DM2."""
        flagUseFCM = False
        flagUseDMs = 'DM2'
        flagHighZern = True

        fn_delta_v = os.path.join(self.localpath, 'testdata',
                                  'dmrel_for_parity.fits')
        deltaV2 = fits.getdata(fn_delta_v)
        deltaV2 /= np.max(deltaV2)
        deltaV2 *= 27  # empirically found to normalize the peak phase
        dm2_map_phase = volts_to_dmh(self.dm2gainmap, deltaV2, self.lam)

        nArray = 400

        dm1r = self.dm1r
        dm2r = self.dm2r
        DM2SurfIn = dmhtoph(
            nArray, nArray, dm2_map_phase, dm2r['nact'],
            fits.getdata(self.dm2inffn), dm2r['ppact_d'],
            dm2r['ppact_cx'], dm2r['ppact_cy'], dm2r['dx'],
            dm2r['dy'], dm1r['thact'], dm2r['flipx'])
        dm2_phase = DM2SurfIn * 2
        dm2_amp = np.ones_like(dm2_phase)

        Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
            self.Vdm1In, self.Vdm2In, self.cmdFCMIn, flagUseDMs, flagUseFCM,
            flagHighZern, self.lam, dm2_amp, dm2_phase,
            self.diamPupil, self.xOffset, self.yOffset,
            self.dm_fn, self.fcm_fn, self.flatten_params_fn)

        deltaV1 = Vdm1Out-self.Vdm1In
        deltaV2 = Vdm2Out-self.Vdm2In
        dm1phase = volts_to_dmh(self.dm1gainmap, deltaV1, self.lam)
        dm2phase = volts_to_dmh(self.dm2gainmap, deltaV2, self.lam)

        DM1SurfOut = dmhtoph(
            nArray, nArray, dm1phase, dm1r['nact'],
            fits.getdata(self.dm1inffn), dm1r['ppact_d'],
            dm1r['ppact_cx'], dm1r['ppact_cy'], dm1r['dx'],
            dm1r['dy'], dm1r['thact'], dm1r['flipx'])

        DM2SurfOut = dmhtoph(
            nArray, nArray, dm2phase, dm2r['nact'],
            fits.getdata(self.dm2inffn), dm2r['ppact_d'],
            dm2r['ppact_cx'], dm2r['ppact_cy'], dm2r['dx'],
            dm2r['dy'], dm2r['thact'], dm2r['flipx'])

        residual_dm2 = DM2SurfOut + DM2SurfIn
        rmsErrorDM1 = np.sqrt(np.mean(DM1SurfOut**2))
        rmsErrorDM2 = np.sqrt(np.mean(residual_dm2**2))

        self.assertTrue(rmsErrorDM1 < 0.05)
        self.assertTrue(rmsErrorDM2 < 0.05)


class TestFlattenWithDeadActs(unittest.TestCase):
    """Test functionality and inputs of flatten() with dead actuators."""

    def setUp(self):
        """Initialize variables used in all the functional tests."""
        self.localpath = os.path.dirname(os.path.abspath(__file__))
        self.dm_fn = os.path.join(self.localpath, 'testdata',
                             'dm_with_bad_act.yaml')
        self.fcm_fn = os.path.join(self.localpath, 'testdata', 'fcm.yaml')
        self.flatten_params_fn = os.path.join(
            self.localpath, 'testdata', 'fixed_inputs_for_conjugate.yaml')

        # pupil calibration data
        self.diamPupil = 386.
        self.xOffset = 0.
        self.yOffset = 0.

        # Load FCM Calibration Data
        fcmConfig = loadyaml(self.fcm_fn)
        self.focusPerStroke = fcmConfig["focusPerStroke"]
        self.strokePerStep = fcmConfig["strokePerStep"]

        # Load fixed settings for CONJUGATE
        flattenDict = loadyaml(self.flatten_params_fn)
        arrayExtFac = flattenDict['arrayExtFac']
        self.nArray = ceil_even(arrayExtFac*self.diamPupil)

        # Load DM Calibration Data to get "nact" and make total voltage maps
        dmConfig = loadyaml(self.dm_fn)
        dm = dmConfig["dms"]
        dm1r = dm['DM1']['registration']
        dm2r = dm['DM2']['registration']
        self.dm1r = dm1r
        self.dm2r = dm2r
        dm1gainmap_fn = os.path.join(self.localpath,
                                     dm['DM1']['voltages']['gainfn'])
        self.dm1gainmap = fits.getdata(dm1gainmap_fn)
        dm2gainmap_fn = os.path.join(self.localpath,
                                     dm['DM2']['voltages']['gainfn'])
        self.dm2gainmap = fits.getdata(dm2gainmap_fn)
        self.dm1tiemap = fits.getdata(os.path.join(
            self.localpath, dm['DM1']['voltages']['tiefn']))
        self.dm2tiemap = fits.getdata(os.path.join(
            self.localpath, dm['DM2']['voltages']['tiefn']))

        self.Vdm1In = 50.*np.ones((dm1r['nact'], dm1r['nact']))
        self.Vdm1In[self.dm1tiemap == -1] = 0
        self.Vdm2In = 50.*np.ones((dm2r['nact'], dm2r['nact']))
        self.Vdm2In[self.dm2tiemap == -1] = 0
        self.cmdFCMIn = 1000

        self.lam = 575e-9  # used only for height-to-phase or vice-versa
        self.dm1inffn = os.path.join(self.localpath, dm1r['inffn'])
        self.dm2inffn = os.path.join(self.localpath, dm2r['inffn'])

        # ph_fn = os.path.join(self.localpath, 'testdata',
        #                 'ph_cgi_entrance_phase_b_N386_radians_575nm.fits')
        amp_fn = os.path.join(self.localpath, 'testdata',
                        'amp_cgi_entrance_phase_b_N386_radians_575nm.fits')
        # self.ph = fits.getdata(ph_fn)
        self.amp = fits.getdata(amp_fn)

        self.flagUseFCM = True
        self.flagUseDMs = True
        self.flagHighZern = True

        self.dm2_real_crosstalk_fn =  os.path.join(
            self.localpath, 'testdata',
            'dm2_crosstalk_correct_orientation.yaml')

    def test_nonfunctional_actuators_both_dms_delta(self):
        """Test phase flattening with a different dead actuator on each DM."""
        peakSurfPoke = 1e-8  # meters
        maxAllowedError = 1e-10  # meters

        flagUseFCM = False
        flagUseDMs = True
        flagHighZern = True

        nArray = self.amp.shape[0]

        dmConfig = loadyaml(self.dm_fn)
        dm = dmConfig["dms"]
        dm1r = dm['DM1']['registration']
        dm2r = dm['DM2']['registration']
        dm1r = dm1r
        dm2r = dm2r

        dm1r.pop('inffn')
        dm1r['inf_func'] = fits.getdata(self.dm1inffn)
        peakInf1 = np.max(dm1r['inf_func'])

        dm2r.pop('inffn')
        dm2r['inf_func'] = fits.getdata(self.dm2inffn)
        peakInf2 = np.max(dm2r['inf_func'])

        nact1 = dm1r['nact']
        dmin1 = np.zeros((nact1, nact1))  # all acts
        dmin1[36, 24] = peakSurfPoke/peakInf1
        nact2 = dm2r['nact']
        dmin2 = np.zeros((nact2, nact2))  # all acts
        dmin2[11, 22] = peakSurfPoke/peakInf2

        d1 = dmhtoph(nrow=nArray, ncol=nArray, dmin=dmin1, **dm1r)
        d2 = dmhtoph(nrow=nArray, ncol=nArray, dmin=dmin2, **dm2r)
        dm1gain = 4e-9
        dm2gain = 4e-9
        ph = (d1*dm1gain + d2*dm2gain)/self.lam*(2.0*np.pi)  # radians

        Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
            self.Vdm1In, self.Vdm2In, self.cmdFCMIn, flagUseDMs, flagUseFCM,
            flagHighZern, self.lam, self.amp, ph,
            self.diamPupil, self.xOffset, self.yOffset,
            self.dm_fn, self.fcm_fn, self.flatten_params_fn)

        dV1 = Vdm1Out - self.Vdm1In
        dV2 = Vdm2Out - self.Vdm2In

        d1out = 2*dmhtoph(nrow=nArray, ncol=nArray, dmin=dV1, **dm1r)
        d2out = 2*dmhtoph(nrow=nArray, ncol=nArray, dmin=dV2, **dm2r)
        d12out = d1out + d2out

        # Verify that total WFE residual is small
        maxSurfResidual = np.max(d12out + (d1 + d2))
        # print('CONJUGATE: max residual = %.3g meters' % maxSurfResidual)
        self.assertTrue(maxSurfResidual < maxAllowedError)

        # Verify that each DM had equal contribution
        maxPeakDifference = np.abs(np.max(np.abs(d1out)) -
                                   np.max(np.abs(d2out)))
        # print('CONJUGATE: max DM1-DM2 WFE difference = %.3g meters' %
        #       maxPeakDifference)
        self.assertTrue(maxPeakDifference < maxAllowedError)

    def test_nonfunctional_actuators_both_dms_abs(self):
        """Test phase flattening with a different dead actuator on each DM."""
        maxAllowedError = 0.25  # volts

        flagUseFCM = False
        flagUseDMs = True
        flagHighZern = True

        nArray = self.amp.shape[0]
        # Have to crop because DM grids are shifted by a whole actuator for
        # this example.
        nArrayCrop = nArray - 40

        dmConfig = loadyaml(self.dm_fn)
        dm = dmConfig["dms"]
        dm1r = dm['DM1']['registration']
        dm2r = dm['DM2']['registration']
        dm1r = dm1r
        dm2r = dm2r

        dm1r.pop('inffn')
        dm1r['inf_func'] = fits.getdata(self.dm1inffn)

        dm2r.pop('inffn')
        dm2r['inf_func'] = fits.getdata(self.dm2inffn)

        nact1 = dm1r['nact']
        dmin1 = 40*np.ones((nact1, nact1))  # all acts
        dmin1[36, 24] = 0
        nact2 = dm2r['nact']
        dmin2 = 40*np.ones((nact2, nact2))  # all acts
        dmin2[11, 22] = 0

        d1 = dmhtoph(nrow=nArray, ncol=nArray, dmin=dmin1, **dm1r)
        d2 = dmhtoph(nrow=nArray, ncol=nArray, dmin=dmin2, **dm2r)
        dm1gain = 4e-9
        dm2gain = 4e-9
        ph = (d1*dm1gain + d2*dm2gain)/self.lam*(2.0*np.pi)  # radians

        Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
            self.Vdm1In, self.Vdm2In, self.cmdFCMIn, flagUseDMs, flagUseFCM,
            flagHighZern, self.lam, self.amp, ph,
            self.diamPupil, self.xOffset, self.yOffset,
            self.dm_fn, self.fcm_fn, self.flatten_params_fn)

        dV1 = Vdm1Out - self.Vdm1In
        dV2 = Vdm2Out - self.Vdm2In

        d1out = 2*dmhtoph(nrow=nArray, ncol=nArray, dmin=dV1, **dm1r)
        d2out = 2*dmhtoph(nrow=nArray, ncol=nArray, dmin=dV2, **dm2r)

        coadded_v = d1 + d2 + d1out + d2out
        usablePixelMap = ampthresh(self.amp).astype(bool)
        coadded_v = inin(coadded_v, (nArrayCrop, nArrayCrop))
        usablePixelMap = inin(usablePixelMap, (nArrayCrop, nArrayCrop))
        coadded_v -= np.mean(coadded_v[usablePixelMap])
        coadded_v[usablePixelMap == False] = 0
        rms_error = np.sqrt(np.mean(coadded_v[usablePixelMap]**2))

        # print('CONJUGATE: max residual = %.3g meters' % maxSurfResidual)
        self.assertTrue(rms_error < maxAllowedError)

    def test_nonfunctional_actuators_abs_dm1_only(self):
        """Test phase flattening with a dead DM2 actuator fixed by DM1."""
        maxAllowedError = 0.25  # volts

        flagUseFCM = False
        flagUseDMs = 'DM1'
        flagHighZern = True

        nArray = self.amp.shape[0]
        # Have to crop because DM grids are shifted by a whole actuator for
        # this example.
        nArrayCrop = nArray - 40

        dmConfig = loadyaml(self.dm_fn)
        dm = dmConfig["dms"]
        dm1r = dm['DM1']['registration']
        dm2r = dm['DM2']['registration']
        dm1r = dm1r
        dm2r = dm2r

        dm1r.pop('inffn')
        dm1r['inf_func'] = fits.getdata(self.dm1inffn)

        dm2r.pop('inffn')
        dm2r['inf_func'] = fits.getdata(self.dm2inffn)

        nact1 = dm1r['nact']
        dmin1 = 40*np.ones((nact1, nact1))  # all acts
        # dmin1[36, 24] = 0
        nact2 = dm2r['nact']
        dmin2 = 40*np.ones((nact2, nact2))  # all acts
        dmin2[11, 22] = 0

        d1 = dmhtoph(nrow=nArray, ncol=nArray, dmin=dmin1, **dm1r)
        d2 = dmhtoph(nrow=nArray, ncol=nArray, dmin=dmin2, **dm2r)
        dm1gain = 4e-9
        dm2gain = 4e-9
        ph = (d1*dm1gain + d2*dm2gain)/self.lam*(2.0*np.pi)  # radians

        Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
            self.Vdm1In, self.Vdm2In, self.cmdFCMIn, flagUseDMs, flagUseFCM,
            flagHighZern, self.lam, self.amp, ph,
            self.diamPupil, self.xOffset, self.yOffset,
            self.dm_fn, self.fcm_fn, self.flatten_params_fn)

        dV1 = Vdm1Out - self.Vdm1In
        dV2 = Vdm2Out - self.Vdm2In

        d1out = 2*dmhtoph(nrow=nArray, ncol=nArray, dmin=dV1, **dm1r)
        d2out = 2*dmhtoph(nrow=nArray, ncol=nArray, dmin=dV2, **dm2r)

        coadded_v = d1 + d2 + d1out + d2out
        usablePixelMap = ampthresh(self.amp).astype(bool)
        coadded_v = inin(coadded_v, (nArrayCrop, nArrayCrop))
        usablePixelMap = inin(usablePixelMap, (nArrayCrop, nArrayCrop))
        coadded_v -= np.mean(coadded_v[usablePixelMap])
        coadded_v[usablePixelMap == False] = 0
        rms_error = np.sqrt(np.mean(coadded_v[usablePixelMap]**2))

        # print('CONJUGATE: max residual = %.3g meters' % maxSurfResidual)
        self.assertTrue(rms_error < maxAllowedError)

    def test_nonfunctional_actuators_abs_dm2_only(self):
        """Test phase flattening with a dead DM1 actuator fixed by DM2."""
        maxAllowedError = 0.25  # volts

        flagUseFCM = False
        flagUseDMs = 'DM2'
        flagHighZern = True

        nArray = self.amp.shape[0]
        # Have to crop because DM grids are shifted by a whole actuator for
        # this example.
        nArrayCrop = nArray - 40

        dmConfig = loadyaml(self.dm_fn)
        dm = dmConfig["dms"]
        dm1r = dm['DM1']['registration']
        dm2r = dm['DM2']['registration']
        dm1r = dm1r
        dm2r = dm2r

        dm1r.pop('inffn')
        dm1r['inf_func'] = fits.getdata(self.dm1inffn)

        dm2r.pop('inffn')
        dm2r['inf_func'] = fits.getdata(self.dm2inffn)

        nact1 = dm1r['nact']
        dmin1 = 40*np.ones((nact1, nact1))  # all acts
        dmin1[36, 24] = 0
        nact2 = dm2r['nact']
        dmin2 = 40*np.ones((nact2, nact2))  # all acts
        # dmin2[11, 22] = 0

        d1 = dmhtoph(nrow=nArray, ncol=nArray, dmin=dmin1, **dm1r)
        d2 = dmhtoph(nrow=nArray, ncol=nArray, dmin=dmin2, **dm2r)
        dm1gain = 4e-9
        dm2gain = 4e-9
        ph = (d1*dm1gain + d2*dm2gain)/self.lam*(2.0*np.pi)  # radians

        Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
            self.Vdm1In, self.Vdm2In, self.cmdFCMIn, flagUseDMs, flagUseFCM,
            flagHighZern, self.lam, self.amp, ph,
            self.diamPupil, self.xOffset, self.yOffset,
            self.dm_fn, self.fcm_fn, self.flatten_params_fn)

        dV1 = Vdm1Out - self.Vdm1In
        dV2 = Vdm2Out - self.Vdm2In

        d1out = 2*dmhtoph(nrow=nArray, ncol=nArray, dmin=dV1, **dm1r)
        d2out = 2*dmhtoph(nrow=nArray, ncol=nArray, dmin=dV2, **dm2r)

        coadded_v = d1 + d2 + d1out + d2out
        usablePixelMap = ampthresh(self.amp).astype(bool)
        coadded_v = inin(coadded_v, (nArrayCrop, nArrayCrop))
        usablePixelMap = inin(usablePixelMap, (nArrayCrop, nArrayCrop))
        coadded_v -= np.mean(coadded_v[usablePixelMap])
        coadded_v[usablePixelMap == False] = 0
        rms_error = np.sqrt(np.mean(coadded_v[usablePixelMap]**2))

        # print('CONJUGATE: max residual = %.3g meters' % maxSurfResidual)
        self.assertTrue(rms_error < maxAllowedError)


class TestFlatten(unittest.TestCase):
    """Test functionality and inputs of flatten()."""

    def setUp(self):
        """Initialize variables used in all the functional tests."""
        self.localpath = os.path.dirname(os.path.abspath(__file__))
        self.dm_fn = os.path.join(self.localpath, 'testdata', 'dm.yaml')
        self.fcm_fn = os.path.join(self.localpath, 'testdata', 'fcm.yaml')
        self.flatten_params_fn = os.path.join(
            self.localpath, 'testdata', 'fixed_inputs_for_conjugate.yaml')

        # pupil calibration data
        self.diamPupil = 386.
        self.xOffset = 0.
        self.yOffset = 0.

        # Load FCM Calibration Data
        fcmConfig = loadyaml(self.fcm_fn)
        self.focusPerStroke = fcmConfig["focusPerStroke"]
        self.strokePerStep = fcmConfig["strokePerStep"]

        # Load fixed settings for CONJUGATE
        flattenDict = loadyaml(self.flatten_params_fn)
        arrayExtFac = flattenDict['arrayExtFac']
        self.nArray = ceil_even(arrayExtFac*self.diamPupil)

        # Load DM Calibration Data to get "nact" and make total voltage maps
        self.dm_fn = os.path.join(self.localpath, 'testdata', 'dm.yaml')
        dmConfig = loadyaml(self.dm_fn)
        dm = dmConfig["dms"]
        dm1r = dm['DM1']['registration']
        dm2r = dm['DM2']['registration']
        self.dm1r = dm1r
        self.dm2r = dm2r
        dm1gainmap_fn = os.path.join(self.localpath,
                                     dm['DM1']['voltages']['gainfn'])
        self.dm1gainmap = fits.getdata(dm1gainmap_fn)
        dm2gainmap_fn = os.path.join(self.localpath,
                                     dm['DM2']['voltages']['gainfn'])
        self.dm2gainmap = fits.getdata(dm2gainmap_fn)

        self.Vdm1In = 50.*np.ones((dm1r['nact'], dm1r['nact']))
        self.Vdm2In = 50.*np.ones((dm2r['nact'], dm2r['nact']))
        self.cmdFCMIn = 1000

        self.lam = 575e-9  # used only for height-to-phase or vice-versa
        self.dm1inffn = os.path.join(self.localpath, dm1r['inffn'])
        self.dm2inffn = os.path.join(self.localpath, dm2r['inffn'])

        ph_fn = os.path.join(self.localpath, 'testdata',
                        'ph_cgi_entrance_phase_b_N386_radians_575nm.fits')
        amp_fn = os.path.join(self.localpath, 'testdata',
                        'amp_cgi_entrance_phase_b_N386_radians_575nm.fits')
        self.ph = fits.getdata(ph_fn)
        self.amp = fits.getdata(amp_fn)

        self.flagUseFCM = True
        self.flagUseDMs = True
        self.flagHighZern = True

        self.dm2_real_crosstalk_fn =  os.path.join(
            self.localpath, 'testdata',
            'dm2_crosstalk_correct_orientation.yaml')

    def test_first_flattening_on_orbit(self):
        """Flatten the wavefront for the first time on orbit."""
        flagUseFCM = True
        flagUseDMs = True
        flagHighZern = True

        Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
            self.Vdm1In, self.Vdm2In, self.cmdFCMIn, flagUseDMs, flagUseFCM,
            flagHighZern, self.lam, self.amp, self.ph,
            self.diamPupil, self.xOffset, self.yOffset,
            self.dm_fn, self.fcm_fn, self.flatten_params_fn)

        # Make WFE maps for Z1,2,3 and Z4
        nArray = self.nArray
        wfeZ1to3 = gen_zernikes(np.array([1, 2, 3]), zernCoefVec[0:3],
                                self.xOffset, self.yOffset,
                                self.diamPupil, nArray=nArray)
        wfeZ4 = gen_zernikes(np.array([4]), np.array([zernCoefVec[3]],),
                             self.xOffset, self.yOffset,
                             self.diamPupil, nArray=nArray)

        # Verify that the output conjugates the input
        usablePixelMap = ampthresh(self.amp)
        usablePixelMapPad = inin(usablePixelMap, (nArray, nArray))
        wfeMeasPad = inin(self.ph, (nArray, nArray))

        deltaV1 = Vdm1Out-self.Vdm1In
        deltaV2 = Vdm2Out-self.Vdm2In
        dm1phase = volts_to_dmh(self.dm1gainmap, deltaV1, self.lam)
        dm2phase = volts_to_dmh(self.dm2gainmap, deltaV2, self.lam)

        dm1r = self.dm1r
        DM1SurfActual = dmhtoph(
            nArray, nArray, dm1phase, dm1r['nact'],
            fits.getdata(self.dm1inffn), dm1r['ppact_d'],
            dm1r['ppact_cx'], dm1r['ppact_cy'], dm1r['dx'],
            dm1r['dy'], dm1r['thact'], dm1r['flipx'])

        dm2r = self.dm2r
        DM2SurfActual = dmhtoph(
            nArray, nArray, dm2phase, dm2r['nact'],
            fits.getdata(self.dm2inffn), dm2r['ppact_d'],
            dm2r['ppact_cx'], dm2r['ppact_cy'], dm2r['dx'],
            dm2r['dy'], dm2r['thact'], dm2r['flipx'])

        wfeResidual = usablePixelMapPad*(wfeMeasPad - (wfeZ1to3 + wfeZ4) +
                        2*(DM1SurfActual + DM2SurfActual))
        rmsError = np.sqrt(np.mean(wfeResidual[usablePixelMapPad == 1]**2))
        # print('CONJUGATE: RMS WFE residual = %.3g rad' % rmsError)
        self.assertTrue(rmsError < 20e-9*(2*np.pi/self.lam))

    def test_first_flattening_on_orbit_flipx(self):
        """Flatten the wavefront for the first time on orbit."""
        flagUseFCM = True
        flagUseDMs = True
        flagHighZern = True

        dm_fn = os.path.join(self.localpath, 'testdata', 'dm_flipx.yaml')
        dmConfig = loadyaml(dm_fn)
        dm = dmConfig["dms"]
        dm1r = dm['DM1']['registration']
        dm2r = dm['DM2']['registration']
        dm1inffn = os.path.join(self.localpath, dm1r['inffn'])
        dm2inffn = os.path.join(self.localpath, dm2r['inffn'])
        dm1gainmap_fn = os.path.join(self.localpath,
                                     dm['DM1']['voltages']['gainfn'])
        dm1gainmap = fits.getdata(dm1gainmap_fn)
        dm2gainmap_fn = os.path.join(self.localpath,
                                     dm['DM2']['voltages']['gainfn'])
        dm2gainmap = fits.getdata(dm2gainmap_fn)

        Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
            self.Vdm1In, self.Vdm2In, self.cmdFCMIn, flagUseDMs, flagUseFCM,
            flagHighZern, self.lam, self.amp, self.ph,
            self.diamPupil, self.xOffset, self.yOffset,
            dm_fn, self.fcm_fn, self.flatten_params_fn)

        # Make WFE maps for Z1,2,3 and Z4
        nArray = self.nArray
        wfeZ1to3 = gen_zernikes(np.array([1, 2, 3]), zernCoefVec[0:3],
                                self.xOffset, self.yOffset,
                                self.diamPupil, nArray=nArray)
        wfeZ4 = gen_zernikes(np.array([4]), np.array([zernCoefVec[3]],),
                             self.xOffset, self.yOffset,
                             self.diamPupil, nArray=nArray)

        # Verify that the output conjugates the input
        usablePixelMap = ampthresh(self.amp)
        usablePixelMapPad = inin(usablePixelMap, (nArray, nArray))
        wfeMeasPad = inin(self.ph, (nArray, nArray))

        deltaV1 = Vdm1Out-self.Vdm1In
        deltaV2 = Vdm2Out-self.Vdm2In
        dm1phase = volts_to_dmh(dm1gainmap, deltaV1, self.lam)
        dm2phase = volts_to_dmh(dm2gainmap, deltaV2, self.lam)

        # dm1r = self.dm1r
        DM1SurfActual = dmhtoph(
            nArray, nArray, dm1phase, dm1r['nact'],
            fits.getdata(dm1inffn), dm1r['ppact_d'],
            dm1r['ppact_cx'], dm1r['ppact_cy'], dm1r['dx'],
            dm1r['dy'], dm1r['thact'], dm1r['flipx'])

        # dm2r = self.dm2r
        DM2SurfActual = dmhtoph(
            nArray, nArray, dm2phase, dm2r['nact'],
            fits.getdata(dm2inffn), dm2r['ppact_d'],
            dm2r['ppact_cx'], dm2r['ppact_cy'], dm2r['dx'],
            dm2r['dy'], dm2r['thact'], dm2r['flipx'])

        wfeResidual = usablePixelMapPad*(wfeMeasPad - (wfeZ1to3 + wfeZ4) +
                        2*(DM1SurfActual + DM2SurfActual))
        rmsError = np.sqrt(np.mean(wfeResidual[usablePixelMapPad == 1]**2))
        # print('CONJUGATE: RMS WFE residual = %.3g rad' % rmsError)
        self.assertTrue(rmsError < 20e-9*(2*np.pi/self.lam))

    def test_equal_dm1_dm2_allocations(self):
        """With >Z11 turned off, make sure DM1 and DM2 split the WFE evenly."""
        flagUseFCM = True
        flagUseDMs = True
        flagHighZern = False

        Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
            self.Vdm1In, self.Vdm2In, self.cmdFCMIn, flagUseDMs, flagUseFCM,
            flagHighZern, self.lam, self.amp, self.ph,
            self.diamPupil, self.xOffset, self.yOffset,
            self.dm_fn, self.fcm_fn, self.flatten_params_fn)

        deltaV1 = Vdm1Out-self.Vdm1In
        deltaV2 = Vdm2Out-self.Vdm2In

        # Undo the crosstalk in DM2 voltages so that we can use the voltages
        # as a proxy for surfaces (since the surfaces are not returned by the
        # function for checking).
        dmxtalk = CDMCrosstalk(self.dm2_real_crosstalk_fn)
        deltaV2 = dmxtalk.crosstalk_forward(deltaV2)

        maxAbsSum = np.max(np.abs(deltaV1 - deltaV2))
        print('CONJUGATE: Max difference in DM1 and DM2 commands = %.3g'
              % maxAbsSum)
        abs_tol = 1000*np.finfo(float).eps
        self.assertTrue(maxAbsSum < abs_tol)

    def test_equal_dm1_dm2_allocations_flipx(self):
        """With >Z11 turned off, make sure DM1 and DM2 split the WFE evenly."""
        flagUseFCM = True
        flagUseDMs = True
        flagHighZern = False

        dm_fn = os.path.join(self.localpath, 'testdata', 'dm_flipx.yaml')

        Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
            self.Vdm1In, self.Vdm2In, self.cmdFCMIn, flagUseDMs, flagUseFCM,
            flagHighZern, self.lam, self.amp, self.ph,
            self.diamPupil, self.xOffset, self.yOffset,
            dm_fn, self.fcm_fn, self.flatten_params_fn)

        deltaV1 = Vdm1Out-self.Vdm1In
        deltaV2 = Vdm2Out-self.Vdm2In

        # Undo the crosstalk in DM2 voltages so that we can use the voltages
        # as a proxy for surfaces (since the surfaces are not returned by the
        # function for checking).
        dmxtalk = CDMCrosstalk(self.dm2_real_crosstalk_fn)
        deltaV2 = dmxtalk.crosstalk_forward(deltaV2)

        maxAbsSum = np.max(np.abs(np.fliplr(deltaV1) - deltaV2))
        print('CONJUGATE: Max difference in DM1 and DM2 commands = %.3g'
              % maxAbsSum)
        abs_tol = 1000*np.finfo(float).eps
        self.assertTrue(maxAbsSum < abs_tol)

    def test_smallZ4_SPMs(self):
        """Flatten the WFE after moving in an SPM. Small focus change case."""
        flagUseFCM = False
        flagUseDMs = True
        flagHighZern = False

        wfeMeas = fits.getdata(os.path.join(
            self.localpath, 'testdata', 'ph_spm_smallZ4_N386.fits'))
        amp = fits.getdata(os.path.join(
            self.localpath, 'testdata', 'SPM_SPC-20200617_386.fits'))

        Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
            self.Vdm1In, self.Vdm2In, self.cmdFCMIn, flagUseDMs, flagUseFCM,
            flagHighZern, self.lam, amp, wfeMeas,
            self.diamPupil, self.xOffset, self.yOffset,
            self.dm_fn, self.fcm_fn, self.flatten_params_fn)

        # Make WFE maps for Z1,2,3
        nArray = self.nArray
        wfeZ1to3 = gen_zernikes(np.array([1, 2, 3]), zernCoefVec[0:3],
                                self.xOffset, self.yOffset,
                                self.diamPupil, nArray=nArray)

        # Verify that the output conjugates the input
        usablePixelMap = ampthresh(amp)
        usablePixelMapPad = inin(usablePixelMap, (nArray, nArray))
        wfeMeasPad = inin(wfeMeas, (nArray, nArray))

        deltaV1 = Vdm1Out-self.Vdm1In
        deltaV2 = Vdm2Out-self.Vdm2In
        dm1phase = volts_to_dmh(self.dm1gainmap, deltaV1, self.lam)
        dm2phase = volts_to_dmh(self.dm2gainmap, deltaV2, self.lam)

        dm1r = self.dm1r
        DM1SurfActual = dmhtoph(
            nArray, nArray, dm1phase, dm1r['nact'],
            fits.getdata(self.dm1inffn), dm1r['ppact_d'],
            dm1r['ppact_cx'], dm1r['ppact_cy'], dm1r['dx'],
            dm1r['dy'], dm1r['thact'], dm1r['flipx'])

        dm2r = self.dm2r
        DM2SurfActual = dmhtoph(
            nArray, nArray, dm2phase, dm2r['nact'],
            fits.getdata(self.dm2inffn), dm2r['ppact_d'],
            dm2r['ppact_cx'], dm2r['ppact_cy'], dm2r['dx'],
            dm2r['dy'], dm2r['thact'], dm2r['flipx'])

        wfeResidual = usablePixelMapPad*(wfeMeasPad - wfeZ1to3 +
                        2*(DM1SurfActual + DM2SurfActual))
        rmsError = np.sqrt(np.mean(wfeResidual[usablePixelMapPad]**2))
        # print('CONJUGATE: RMS WFE residual for small Z4 SPM case = %.3g rad'
        #       % rmsError)
        self.assertTrue(rmsError < 1e-9*(2*np.pi/self.lam))

    def test_largeZ4_SPMs(self):
        """Flatten the WFE after moving in an SPM. Large focus change case."""
        flagUseFCM = True
        flagUseDMs = True
        flagHighZern = False

        wfeMeas = fits.getdata(os.path.join(
            self.localpath, 'testdata', 'ph_spm_largeZ4_N386.fits'))
        amp = fits.getdata(os.path.join(
            self.localpath, 'testdata', 'SPM_SPC-20200610_386.fits'))

        Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
            self.Vdm1In, self.Vdm2In, self.cmdFCMIn, flagUseDMs, flagUseFCM,
            flagHighZern, self.lam, amp, wfeMeas,
            self.diamPupil, self.xOffset, self.yOffset,
            self.dm_fn, self.fcm_fn, self.flatten_params_fn)

        # Make WFE maps for Z1,2,3 and Z4
        nArray = self.nArray
        wfeZ1to3 = gen_zernikes(np.array([1, 2, 3]), zernCoefVec[0:3],
                                self.xOffset, self.yOffset,
                                self.diamPupil, nArray=nArray)
        wfeZ4 = gen_zernikes(np.array([4]), np.array([zernCoefVec[3]],),
                             self.xOffset, self.yOffset,
                             self.diamPupil, nArray=nArray)

        # Verify that the output conjugates the input
        usablePixelMap = ampthresh(amp)
        usablePixelMapPad = inin(usablePixelMap, (nArray, nArray))
        wfeMeasPad = inin(wfeMeas, (nArray, nArray))

        deltaV1 = Vdm1Out-self.Vdm1In
        deltaV2 = Vdm2Out-self.Vdm2In
        dm1phase = volts_to_dmh(self.dm1gainmap, deltaV1, self.lam)
        dm2phase = volts_to_dmh(self.dm2gainmap, deltaV2, self.lam)

        flipx = False
        dm1r = self.dm1r
        DM1SurfActual = dmhtoph(
            nArray, nArray, dm1phase, dm1r['nact'],
            fits.getdata(self.dm1inffn), dm1r['ppact_d'],
            dm1r['ppact_cx'], dm1r['ppact_cy'], dm1r['dx'],
            dm1r['dy'], dm1r['thact'], flipx)

        dm2r = self.dm2r
        DM2SurfActual = dmhtoph(
            nArray, nArray, dm2phase, dm2r['nact'],
            fits.getdata(self.dm2inffn), dm2r['ppact_d'],
            dm2r['ppact_cx'], dm2r['ppact_cy'], dm2r['dx'],
            dm2r['dy'], dm2r['thact'], flipx)

        wfeResidual = usablePixelMapPad*(wfeMeasPad - wfeZ1to3 - wfeZ4 +
                        2*(DM1SurfActual + DM2SurfActual))
        rmsError = np.sqrt(np.mean(wfeResidual[usablePixelMapPad]**2))
        # print('CONJUGATE: RMS WFE residual for large Z4 SPM case = %.3g rad'
        #       % rmsError)
        self.assertTrue(rmsError < 1e-9*(2*np.pi/self.lam))

    def test_minimize_focus_residual_when_setting_coarse_fcm(self):
        """Make sure coarse FCM calibration is off by <= half a step."""
        flagUseFCM = True
        flagUseDMs = False
        flagHighZern = False
        # Sorry, only reason it's not a one-shot is the coarse step size.
        # Probably don't need to do iteratively; could just show residual is
        # less than 1/2 FCM coarse step.

        wfe = fits.getdata(os.path.join(
            self.localpath, 'testdata', 'ph_Z4_1radianRMS_N386.fits'))
        amp = fits.getdata(os.path.join(
            self.localpath, 'testdata',
            'amp_cgi_entrance_phase_b_N386_radians_575nm.fits'))

        Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
            self.Vdm1In, self.Vdm2In, self.cmdFCMIn, flagUseDMs, flagUseFCM,
            flagHighZern, self.lam, amp, wfe,
            self.diamPupil, self.xOffset, self.yOffset,
            self.dm_fn, self.fcm_fn, self.flatten_params_fn)

        deltaCmdFCM = cmdFCMOut - self.cmdFCMIn
        stepSizeRadiansRMS = -(2*np.pi / self.lam * self.focusPerStroke *
                              self.strokePerStep)
        focusRadiansRMSEst = deltaCmdFCM*stepSizeRadiansRMS
        focusRadiansRMSTrue = 1.0
        absDiff = np.abs(focusRadiansRMSEst - focusRadiansRMSTrue)
        self.assertTrue(absDiff < 0.5)

    # Input Tests
    #
    # The last three inputs are YAML file names and are not tested because
    # they are directly passed to loadyaml(), which already has all its own
    # unit tests.
    def test_input_0(self):
        """Verify that an exception is raised for bad inputs."""
        for Vdm1InBad in (-1, 0.5, [2, ], 1j*np.ones((5, 10)), 'str'):
            with self.assertRaises(TypeError):
                Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
                    Vdm1InBad, self.Vdm2In, self.cmdFCMIn,
                    self.flagUseDMs, self.flagUseFCM, self.flagHighZern,
                    self.lam, self.amp, self.ph,
                    self.diamPupil, self.xOffset, self.yOffset,
                    self.dm_fn, self.fcm_fn, self.flatten_params_fn)

    def test_input_1(self):
        """Verify that an exception is raised for bad inputs."""
        for Vdm2InBad in (-1, 0.5, [2, ], 1j*np.ones((5, 10)), 'str'):
            with self.assertRaises(TypeError):
                Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
                    self.Vdm1In, Vdm2InBad, self.cmdFCMIn,
                    self.flagUseDMs, self.flagUseFCM, self.flagHighZern,
                    self.lam, self.amp, self.ph,
                    self.diamPupil, self.xOffset, self.yOffset,
                    self.dm_fn, self.fcm_fn, self.flatten_params_fn)

    def test_input_2(self):
        """Verify that an exception is raised for bad inputs."""
        for cmdFCMInBad in (-1.5, 2.5, 1j, [3, ], np.ones((5, 10)), 'str'):
            with self.assertRaises(TypeError):
                Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
                    self.Vdm1In, self.Vdm2In, cmdFCMInBad,
                    self.flagUseDMs, self.flagUseFCM, self.flagHighZern,
                    self.lam, self.amp, self.ph,
                    self.diamPupil, self.xOffset, self.yOffset,
                    self.dm_fn, self.fcm_fn, self.flatten_params_fn)

    def test_input_3(self):
        """Verify that an exception is raised for bad inputs."""
        for flagUseDMsBad in (-1, 0, 1, [True, ], np.ones((5, 10))):
            with self.assertRaises(TypeError):
                Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
                    self.Vdm1In, self.Vdm2In, self.cmdFCMIn,
                    flagUseDMsBad, self.flagUseFCM, self.flagHighZern,
                    self.lam, self.amp, self.ph,
                    self.diamPupil, self.xOffset, self.yOffset,
                    self.dm_fn, self.fcm_fn, self.flatten_params_fn)

    def test_input_3pt5(self):
        """Verify that an exception is raised for bad inputs."""
        for flagUseDMsBad in ('wrong_string', 'DM', '1', '2'):
            with self.assertRaises(ValueError):
                Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
                    self.Vdm1In, self.Vdm2In, self.cmdFCMIn,
                    flagUseDMsBad, self.flagUseFCM, self.flagHighZern,
                    self.lam, self.amp, self.ph,
                    self.diamPupil, self.xOffset, self.yOffset,
                    self.dm_fn, self.fcm_fn, self.flatten_params_fn)

    def test_input_4(self):
        """Verify that an exception is raised for bad inputs."""
        for flagUseFCMBad in (-1, 0, 1, [True, ], np.ones((5, 10)), 'str'):
            with self.assertRaises(TypeError):
                Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
                    self.Vdm1In, self.Vdm2In, self.cmdFCMIn,
                    self.flagUseDMs, flagUseFCMBad, self.flagHighZern,
                    self.lam, self.amp, self.ph,
                    self.diamPupil, self.xOffset, self.yOffset,
                    self.dm_fn, self.fcm_fn, self.flatten_params_fn)

    def test_input_5(self):
        """Verify that an exception is raised for bad inputs."""
        for cmdFCMInBad in (1.5, 1j, [True, ], np.ones((5, 10)), 'str'):
            with self.assertRaises(TypeError):
                Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
                    self.Vdm1In, self.Vdm2In, cmdFCMInBad,
                    self.flagUseDMs, self.flagUseFCM, self.flagHighZern,
                    self.lam, self.amp, self.ph,
                    self.diamPupil, self.xOffset, self.yOffset,
                    self.dm_fn, self.fcm_fn, self.flatten_params_fn)

    def test_input_6(self):
        """Verify that an exception is raised for bad inputs."""
        for lamBad in (-1, 0, 1j, [0.5, ], np.ones((5, 10)), 'str'):
            with self.assertRaises(TypeError):
                Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
                    self.Vdm1In, self.Vdm2In, self.cmdFCMIn,
                    self.flagUseDMs, self.flagUseFCM, self.flagHighZern,
                    lamBad, self.amp, self.ph,
                    self.diamPupil, self.xOffset, self.yOffset,
                    self.dm_fn, self.fcm_fn, self.flatten_params_fn)

    def test_input_7(self):
        """Verify that an exception is raised for bad inputs."""
        for ampBad in (-1, 0.5, [2, ], 1j*np.ones((5, 10)), 'str'):
            with self.assertRaises(TypeError):
                Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
                    self.Vdm1In, self.Vdm2In, self.cmdFCMIn,
                    self.flagUseDMs, self.flagUseFCM, self.flagHighZern,
                    self.lam, ampBad, self.ph,
                    self.diamPupil, self.xOffset, self.yOffset,
                    self.dm_fn, self.fcm_fn, self.flatten_params_fn)

    def test_input_8(self):
        """Verify that an exception is raised for bad inputs."""
        for phBad in (-1, 0.5, [2, ], 1j*np.ones((5, 10)), 'str'):
            with self.assertRaises(TypeError):
                Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
                    self.Vdm1In, self.Vdm2In, self.cmdFCMIn,
                    self.flagUseDMs, self.flagUseFCM, self.flagHighZern,
                    self.lam, self.amp, phBad,
                    self.diamPupil, self.xOffset, self.yOffset,
                    self.dm_fn, self.fcm_fn, self.flatten_params_fn)

    def test_input_9(self):
        """Verify that an exception is raised for bad inputs."""
        for diamPupilBad in (-1, 0, 1j, [0.5, ], np.ones((5, 10)), 'str'):
            with self.assertRaises(TypeError):
                Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
                    self.Vdm1In, self.Vdm2In, self.cmdFCMIn,
                    self.flagUseDMs, self.flagUseFCM, self.flagHighZern,
                    self.lam, self.amp, self.ph,
                    diamPupilBad, self.xOffset, self.yOffset,
                    self.dm_fn, self.fcm_fn, self.flatten_params_fn)

    def test_input_10(self):
        """Verify that an exception is raised for bad inputs."""
        for xOffsetBad in (1j, [2, ], np.ones((5, 10)), 'str'):
            with self.assertRaises(TypeError):
                Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
                    self.Vdm1In, self.Vdm2In, self.cmdFCMIn,
                    self.flagUseDMs, self.flagUseFCM, self.flagHighZern,
                    self.lam, self.amp, self.ph,
                    self.diamPupil, xOffsetBad, self.yOffset,
                    self.dm_fn, self.fcm_fn, self.flatten_params_fn)

    def test_input_11(self):
        """Verify that an exception is raised for bad inputs."""
        for yOffsetBad in (1j, [2, ], np.ones((5, 10)), 'str'):
            with self.assertRaises(TypeError):
                Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
                    self.Vdm1In, self.Vdm2In, self.cmdFCMIn,
                    self.flagUseDMs, self.flagUseFCM, self.flagHighZern,
                    self.lam, self.amp, self.ph,
                    self.diamPupil, self.xOffset, yOffsetBad,
                    self.dm_fn, self.fcm_fn, self.flatten_params_fn)


    def test_input_12(self):
        """Verify that an exception is raised for bad inputs."""
        for pathbad in (1j, [2, ], np.ones((5, 10))):
            with self.assertRaises(TypeError):
                Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
                    self.Vdm1In, self.Vdm2In, self.cmdFCMIn,
                    self.flagUseDMs, self.flagUseFCM, self.flagHighZern,
                    self.lam, self.amp, self.ph,
                    self.diamPupil, self.xOffset, self.yOffset,
                    pathbad, self.fcm_fn, self.flatten_params_fn)

    def test_input_13(self):
        """Verify that an exception is raised for bad inputs."""
        for pathbad in (1j, [2, ], np.ones((5, 10))):
            with self.assertRaises(TypeError):
                Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
                    self.Vdm1In, self.Vdm2In, self.cmdFCMIn,
                    self.flagUseDMs, self.flagUseFCM, self.flagHighZern,
                    self.lam, self.amp, self.ph,
                    self.diamPupil, self.xOffset, self.yOffset,
                    self.dm_fn, pathbad, self.flatten_params_fn)

    def test_input_14(self):
        """Verify that an exception is raised for bad inputs."""
        for pathbad in (1j, [2, ], np.ones((5, 10))):
            with self.assertRaises(TypeError):
                Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
                    self.Vdm1In, self.Vdm2In, self.cmdFCMIn,
                    self.flagUseDMs, self.flagUseFCM, self.flagHighZern,
                    self.lam, self.amp, self.ph,
                    self.diamPupil, self.xOffset, self.yOffset,
                    self.dm_fn, self.fcm_fn, pathbad)

    def test_input_15(self):
        """Verify that an exception is raised for bad inputs."""
        for pathbad in (1j, [2, ], np.ones((5, 10))):
            with self.assertRaises(TypeError):
                Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
                    self.Vdm1In, self.Vdm2In, self.cmdFCMIn,
                    self.flagUseDMs, self.flagUseFCM, self.flagHighZern,
                    self.lam, self.amp, self.ph,
                    self.diamPupil, self.xOffset, self.yOffset,
                    self.dm_fn, self.fcm_fn, self.flatten_params_fn,
                    data_path=pathbad)




class TestConjugateSupportFunctions(unittest.TestCase):
    """Test all the supporting methods of conjugate."""

    def test_gen_split_wfe(self):
        """Test that gen_split_wfe correctly generates WFE maps."""
        # Cases: flagUseFCM == True, flagUseFCM == False
        diamPupil = 100
        wfeShape = (120, 130)
        xOffset = 0.0
        yOffset = 0.0
        maxNollZern = 11
        zernCoefVec = np.arange(1, 13)

        x = (np.arange(wfeShape[1], dtype=np.float64) - wfeShape[1]//2)
        y = (np.arange(wfeShape[0], dtype=np.float64) - wfeShape[0]//2)
        xx, yy = np.meshgrid(x, y)

        # Create the true WFE allocations
        zernIndVec = np.arange(1, 13)
        zernCube = xyzern(xx, yy, diamPupil/2., zernIndVec)
        wfeZ1to3True = np.zeros(wfeShape, dtype=np.float64)
        wfeZ4True = np.zeros(wfeShape, dtype=np.float64)
        wfeZ4or5to11True = np.zeros(wfeShape, dtype=np.float64)
        for iz in (0, 1, 2):
            wfeZ1to3True += zernCoefVec[iz]*np.squeeze(zernCube[iz, :, :])
        for iz in (3,):
            wfeZ4True += zernCoefVec[iz]*np.squeeze(zernCube[iz, :, :])
        for iz in (4, 5, 6, 7, 8, 9, 10):
            wfeZ4or5to11True += zernCoefVec[iz]*np.squeeze(zernCube[iz, :, :])

        flagUseFCM = True
        wfeZ1to3, wfeZ4, wfeZ4or5to11 = gen_split_wfe(
            flagUseFCM, maxNollZern, zernCoefVec, diamPupil, wfeShape,
            xOffset, yOffset)

        self.assertTrue(isclose(np.sum(np.abs(wfeZ1to3 - wfeZ1to3True)), 0.0,
                                abs_tol=1e-4))
        self.assertTrue(isclose(np.sum(np.abs(wfeZ4 - wfeZ4True)), 0.0,
                                abs_tol=1e-4))
        self.assertTrue(isclose(np.sum(np.abs(wfeZ4or5to11 -
                                              wfeZ4or5to11True)), 0.0,
                                abs_tol=1e-4))

        flagUseFCM = False
        wfeZ1to3, wfeZ4, wfeZ4or5to11 = gen_split_wfe(
            flagUseFCM, maxNollZern, zernCoefVec, diamPupil, wfeShape,
            xOffset, yOffset)

        # wfeZ4True = np.zeros(wfeShape, dtype=np.float64)
        wfeZ4or5to11True = np.zeros(wfeShape, dtype=np.float64)
        for iz in (3, 4, 5, 6, 7, 8, 9, 10):
            wfeZ4or5to11True += zernCoefVec[iz]*np.squeeze(zernCube[iz, :, :])

        self.assertTrue(isclose(np.sum(np.abs(wfeZ4)), 0.0, abs_tol=1e-4))
        self.assertTrue(isclose(np.sum(np.abs(wfeZ4or5to11 -
                                              wfeZ4or5to11True)), 0.0,
                                abs_tol=1e-4))
        pass

    def test_smooth_surface(self):
        """Test that smooth_surface correctly smooths an array."""
        windowWidth = 3
        arrayIn = np.zeros((12, 12))
        arrayIn[1::3, 1::3] = 1
        arrayOut = smooth_surface(arrayIn, windowWidth)
        arrayExpected = np.ones((12, 12)) / 9.0
        diffSum = np.sum(np.abs(arrayOut - arrayExpected))
        self.assertTrue(isclose(diffSum, 0.0, abs_tol=1e-8))
        self.assertTrue(isclose(np.sum(arrayOut), np.sum(arrayExpected),
                                abs_tol=1e-8))
        pass

    def test_phase_cost_function(self):
        """Test phase_cost_function."""
        # If wfeZ4or5to11 and wfeAboveZ11 have same RMS,
        # then alpha=0 should give cost==0
        usablePixelMap = np.ones((2, 2), dtype=bool)

        alpha = 0.0
        wfeZ4or5to11 = np.array([[-1, 1], [1, -1]])
        wfeAboveZ11 = np.array([[1, 1], [-1, -1]])
        cost = phase_cost_function(alpha, wfeZ4or5to11, wfeAboveZ11,
                                   usablePixelMap)
        self.assertTrue(isclose(cost, 0.0, abs_tol=1e-8))

        # If rms(wfeZ4or5to11) < rms(wfeAboveZ11),
        # then alpha=0 should give cost > 0
        alpha = 0.0
        wfeZ4or5to11 = np.array([[-1, 1], [1, -1]])
        wfeAboveZ11 = 4*np.array([[-1, 1], [1, -1]])
        cost = phase_cost_function(alpha, wfeZ4or5to11, wfeAboveZ11,
                                   usablePixelMap)
        self.assertTrue(isclose(cost, (4.-1.)**2, abs_tol=1e-8))

        # If rms(wfeZ4or5to11) > rms(wfeAboveZ11)
        alpha = 1/3.0
        wfeZ4or5to11 = 3*np.array([[-1, 1], [1, -1]])
        wfeAboveZ11 = np.array([[-1, 1], [1, -1]])
        cost = phase_cost_function(alpha, wfeZ4or5to11, wfeAboveZ11,
                                   usablePixelMap)
        self.assertTrue(isclose(cost, 0.0, abs_tol=1e-8))
        pass


class TestSupportFunctionInputs(unittest.TestCase):
    """Test suite for valid function inputs."""

    def test_gen_split_wfe_inputs(self):
        """Test incorrect inputs of gen_split_wfe."""
        for flagUseFCMBad in (0, 1, 1.5, np.ones((5, 10)), 'string'):
            flagUseFCM = True
            maxNollZern = 9
            zernCoef = np.random.rand(11)
            diamPupil = 386.
            surfShape = (501, 485)
            xOffset = -0.1
            yOffset = 0.1

            for flagUseDM1Bad in (0, 1, 1.5, np.ones((5, 10)), 'string'):
                with self.assertRaises(TypeError):
                    gen_split_wfe(flagUseFCMBad, maxNollZern,
                                  zernCoef, diamPupil, surfShape,
                                  xOffset, yOffset)
            for maxNollZernBad in (-1, 0, 1.5, np.ones((5,)), 'string'):
                with self.assertRaises(TypeError):
                    gen_split_wfe(flagUseFCM, maxNollZernBad,
                                  zernCoef, diamPupil, surfShape,
                                  xOffset, yOffset)
            for zernCoefBad in (1, np.ones((5, 10)), 'string'):
                with self.assertRaises(TypeError):
                    gen_split_wfe(flagUseFCM, maxNollZern,
                                  zernCoefBad, diamPupil, surfShape,
                                  xOffset, yOffset)
            for diamPupilBad in (-1.5, 0, np.ones((5,)), 'string'):
                with self.assertRaises(TypeError):
                    gen_split_wfe(flagUseFCM, maxNollZern, zernCoef,
                                  diamPupilBad, surfShape,
                                  xOffset, yOffset)
            for surfShapeBad in (0, np.ones((5,)), (10, 10, 10), 'string'):
                with self.assertRaises(TypeError):
                    gen_split_wfe(flagUseFCM, maxNollZern, zernCoef,
                                  diamPupil, surfShapeBad,
                                  xOffset, yOffset)
            for xOffsetBad in (1.5+1j, np.ones((5,)), np.ones((10, 10)),
                               'string'):
                with self.assertRaises(TypeError):
                    gen_split_wfe(flagUseFCM, maxNollZern, zernCoef,
                                  diamPupil, surfShape,
                                  xOffsetBad, yOffset)
            for yOffsetBad in (1.5+1j, np.ones((5,)), np.ones((10, 10)),
                               'string'):
                with self.assertRaises(TypeError):
                    gen_split_wfe(flagUseFCM, maxNollZern, zernCoef,
                                  diamPupil, surfShape,
                                  xOffset, yOffsetBad)

    def test_smooth_surface_inputs(self):
        """Test incorrect inputs of smooth_surface."""
        with self.assertRaises(TypeError):
            smooth_surface(np.ones((3, 3, 4)), 3.2)
        with self.assertRaises(TypeError):
            smooth_surface(np.ones((10, 10)), np.array([2, 4.]))
        with self.assertRaises(TypeError):
            smooth_surface(np.ones((10, 10)), -4.2)

    def test_conv_surf_to_dm_cmd_inputs(self):
        """Test incorrect inputs of conv_surf_to_dm_cmd."""
        localpath = os.path.dirname(os.path.abspath(__file__))

        # Load DM Calibration Data
        dm_fn = os.path.join(localpath, 'testdata', 'dm_offset.yaml')
        dmConfig = loadyaml(dm_fn)
        dm = dmConfig["dms"]
        dm1 = dm['DM1']
        surfaceToFit = np.ones((500, 500))

        for surfaceToFitBad in (1, np.ones((5, 10, 20)), 'string'):
            with self.assertRaises(TypeError):
                conv_surf_to_dm_cmd(surfaceToFitBad, dm1)

        for dm1Bad in (1, np.ones((5, 10)), 'string'):
            with self.assertRaises(TypeError):
                conv_surf_to_dm_cmd(surfaceToFit, dm1Bad)

    def test_phase_cost_function_inputs(self):
        """Test incorrect inputs of phase_cost_function."""
        shape = (100, 102)
        alpha = 0.25
        wfeZ4or5to11 = 4*np.random.rand(shape[0], shape[1])
        wfeAboveZ11 = np.random.rand(shape[0], shape[1])
        usablePixelMap = np.round(np.random.rand(shape[0],
                                                 shape[1])).astype(bool)

        for alphaBad in (-0.1, 1.1, 1j, np.ones((5, 10)), 'string'):
            with self.assertRaises(TypeError):
                phase_cost_function(alphaBad, wfeZ4or5to11,
                                    wfeAboveZ11, usablePixelMap)

        for wfeZ4or5to11Bad in (-0.1, 1.1, 1j, [2, ], np.ones((5, 10, 3)),
                                np.ones((5, 10)), 1j*np.ones(shape), 'string'):
            with self.assertRaises(TypeError):
                phase_cost_function(alpha, wfeZ4or5to11Bad,
                                    wfeAboveZ11, usablePixelMap)

        for wfeAboveZ11Bad in (-0.1, 1.1, 1j, [2, ], np.ones((5, 10, 3)),
                               np.ones((5, 10)), 1j*np.ones(shape), 'string'):
            with self.assertRaises(TypeError):
                phase_cost_function(alpha, wfeZ4or5to11,
                                    wfeAboveZ11Bad, usablePixelMap)

        for usablePixelMapBad in (-0.1, 1.1, 1j, [2, ], np.ones((5, 10, 3)),
                                  np.ones((5, 10)), 1.5*np.ones(shape),
                                  'string'):
            with self.assertRaises(TypeError):
                phase_cost_function(alpha, wfeZ4or5to11,
                                    wfeAboveZ11Bad, usablePixelMap)


if __name__ == '__main__':
    unittest.main()
