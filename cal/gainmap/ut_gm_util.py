# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Unit test suite for gainmap module."""
import unittest
import os
import numpy as np
from astropy.io import fits

import cal.gainmap.gm_util as gmu
import cal.gainmap.gainmap as gm
import cal.util.ut_check as ut_check
from cal.util.dm_crosstalk import CDMCrosstalk
from cal.util.loadyaml import loadyaml

class TestGainmapUtils(unittest.TestCase):
    """Unit tests for gm_util.py."""

    def setUp(self):
        """Set up variables for communal use."""
        self.localpath = os.path.dirname(os.path.abspath(__file__))
        self.fnRefCubeLinear = os.path.join(self.localpath, 'testdata',
                                      'ut_refCubeLinear.fits')
        self.fnRefCube = os.path.join(self.localpath, 'testdata',
                                      'ut_refCubeQuadratic.fits')
        self.fnRefCubicCube = os.path.join(self.localpath, 'testdata',
                                      'ut_refCubeNegGain.fits')
        self.fnRefGainCube = os.path.join(self.localpath, 'testdata',
                                      'ut_refCubeQuadratic_Gain.fits')
        self.fnRefCommandVec = os.path.join(self.localpath, 'testdata',
                                            'ut_refCommandVec.fits')
        self.fnRefCommandVecGain = os.path.join(self.localpath, 'testdata',
                                            'ut_refCommandVec_Gain.fits')
        self.crosstalk_fn = os.path.join(self.localpath, 'testdata',
                                         'ut_dm_crosstalk.yaml')

        self.nActs = 48
        self.a = 1/100  # quadratic coefficient for ut_refCubeQuadratic.fits
        self.b = 3   # linear coefficient for ut_refCubeQuadratic.fits
        self.c = 0  # DC coefficient for ut_refCubeQuadratic.fits

        # height vs voltage curve that increases, dips down, then goes up again
        # for curve, for ut_refCubeNegGain.fits:
        # (v-z)**3 + b2*x*2 + c2
        self.z = 3
        self.b2 = -1
        self.c2 = 30


    def test_diff_of_vector_(self):
        """Test differencing part of compute_diff_and_mean_of_vector()."""
        diff, _, _ = gmu.compute_diff_and_mean_of_vector(self.fnRefCommandVec)
        expectedVec = 10*np.ones(10)

        self.assertTrue(np.allclose(diff,
                                    expectedVec,
                                    atol=np.finfo(float).eps,
                                    )
                        )

    def test_mean_of_vector_(self):
        """Test averaging part of compute_diff_and_mean_of_vector()."""
        _, mean, _ = gmu.compute_diff_and_mean_of_vector(self.fnRefCommandVec)
        expectedVec = np.arange(5, 96, 10)

        self.assertTrue(np.allclose(mean,
                                    expectedVec,
                                    atol=np.finfo(float).eps,
                                    )
                        )

    def test_vector_(self):
        """Test vector part of compute_diff_and_mean_of_vector()."""
        _, _, Vec = gmu.compute_diff_and_mean_of_vector(self.fnRefCommandVec)
        expectedVec = fits.getdata(self.fnRefCommandVec)

        self.assertTrue(np.allclose(Vec,
                                    expectedVec,
                                    atol=np.finfo(float).eps,
                                    )
                        )

    def test_compute_delta_height_map_from_command_maps(self):
        """Test accuracy of compute_delta_height_map_from_command_maps()."""
        commandBefore = 25.2
        commandAfter = 45.5
        commandMapBefore = commandBefore * np.ones((self.nActs, self.nActs))
        commandMapAfter = commandAfter * np.ones((self.nActs, self.nActs))
        deltaHeight = ((self.a*commandAfter**2 + self.b*commandAfter) -
                       (self.a*commandBefore**2 + self.b*commandBefore))
        deltaHeightMapTrue = deltaHeight * np.ones((self.nActs, self.nActs))

        deltaHeightMap = gmu.compute_delta_height_map_from_command_maps(
            commandMapBefore,
            commandMapAfter,
            self.fnRefCube,
            self.fnRefCommandVec)

        self.assertTrue(np.allclose(deltaHeightMapTrue,
                                    deltaHeightMap,
                                    atol=1e-4,
                                    )
                        )

    def test_compute_delta_height_map_from_command_maps_crosstalk(self):
        """Test accuracy of compute_delta_height_map_from_command_maps() with
        cross talk."""
        commandBefore = 25.2
        commandAfter = 45.5
        commandMapBefore = commandBefore * np.ones((self.nActs, self.nActs))
        commandMapAfter = commandAfter * np.ones((self.nActs, self.nActs))
        heightMapAfter = self.a*commandMapAfter**2 + self.b*commandMapAfter
        heightMapBefore = self.a*commandMapBefore**2 + self.b*commandMapBefore
        # apply DM crosstalk on height map to get true height map
        dmxtalk = CDMCrosstalk(self.crosstalk_fn)
        heightMapAfterTrue = dmxtalk.crosstalk_forward(heightMapAfter)
        heightMapBeforeTrue = dmxtalk.crosstalk_forward(heightMapBefore)
        deltaHeightMapTrue = heightMapAfterTrue - heightMapBeforeTrue


        deltaHeightMap = gmu.compute_delta_height_map_from_command_maps(
            commandMapBefore,
            commandMapAfter,
            self.fnRefCube,
            self.fnRefCommandVec,
            self.crosstalk_fn)

        self.assertTrue(np.allclose(deltaHeightMapTrue,
                                    deltaHeightMap,
                                    rtol=1e-4,
                                    )
                        )

    def test_compute_commands_for_height_map_neg(self):
        """Test accuracy of compute_commands_for_height_map(), with inherent
        negative gain and without cross talk."""
        nActs = self.nActs

        # For cubic map, max height (at max voltage vector value of 100V)
        # is 902,703.  All but the last entry in heightMap is < 902,703.
        # For
        # last few entries, the height evaluated at the found command map value
        # should be off a little, limited by max voltage allowed.
        heightMap = np.linspace(self.c2, 902704, nActs**2).reshape((nActs,
                                                                    nActs))

        # gain: g(v) = 3*(v-z)**2 + 2*b2*v.  The gain is first zero at
        # V = 1/3(-b+3z-sqrt(b^2-6bz)), less than zero for a stretch, then zero
        # again at V2 = 1/3(-b+3z+sqrt(b^2-6bz)).  Gain is positive after V2.
        # So no adjustment occurs before V, and there height difference b/w
        # V2 and some higher voltage is the same.  So the adjusted height at
        # v>V2:
        # adj_height(v) = height(V) + integral(min_gain dv', {V, V2}) +
        # height(v) - height(V2), and
        # integral(min_gain dv', {V, V2}) = min_gain*(V2-V)
        # Adjusted height for v<V2:
        # adj_height(v) = height(V) + min_gain*(v-V)
        V = 1/3*(-self.b2+3*self.z-np.sqrt(self.b2**2-6*self.b2*self.z))
        V2 = 1/3*(-self.b2+3*self.z+np.sqrt(self.b2**2-6*self.b2*self.z))
        def height(v):
            return (v-self.z)**3 + self.b2*v**2 + self.c2

        # # exact solution of height(v) = s for s > height(V2)
        def inverse_height(s):
            return  ((-self.b2 + 3*self.z)/3. - (2**(1/3)*(-self.b2**2 +
                                                        6*self.b2*self.z))/
            (3.*(-2*self.b2**3 - 27*self.c2 + 27*s + 18*self.b2**2*self.z -
             27*self.b2*self.z**2 + np.sqrt(4*(-self.b2**2 +
            6*self.b2*self.z)**3 + (-2*self.b2**3 - 27*self.c2 + 27*s +
            18*self.b2**2*self.z - 27*self.b2*self.z**2)**2))**
            (1/3)) + (-2*self.b2**3 - 27*self.c2 + 27*s + 18*self.b2**2*self.z
            - 27*self.b2*self.z**2 + np.sqrt(4*(-self.b2**2 +
            6*self.b2*self.z)**3 + (-2*self.b2**3 - 27*self.c2 + 27*s +
            18*self.b2**2*self.z - 27*self.b2*self.z**2)**2))**((1/3))
            /(3.*2**(1/3)))

        CommandMapTrue = np.zeros_like(heightMap).ravel()
        for ind, val in enumerate(heightMap.ravel()):
            volt = inverse_height(val)
            CommandMapTrue[ind] = volt
        CommandMapTrue = CommandMapTrue.reshape((nActs, nActs))
        vmin = np.min(CommandMapTrue)
        vmax = np.max(CommandMapTrue)

        commandMap = gmu.compute_commands_for_height_map(
            heightMap,
            self.fnRefCubicCube,
            self.fnRefCommandVec,
            vmin,
            vmax,
        )
        # for those below the cap and above the non-monotonic part, the
        # solution is single-valued for voltages
        ind1 = np.where(CommandMapTrue.ravel() < 100)
        ind2 = np.where(CommandMapTrue.ravel() > V2)
        ind = np.intersect1d(ind1, ind2)
        self.assertTrue(np.allclose(CommandMapTrue.ravel()[ind],
                                    commandMap.ravel()[ind], atol=1e-4))
        # just check that voltages found for heights in the
        # non-monotonic region give the expected height
        bw1 = np.where(heightMap.ravel() >= height(V2))
        bw2 = np.where(heightMap.ravel() <= height(V))
        bw = np.intersect1d(bw1, bw2)
        for i in bw:
            self.assertTrue(
            np.isclose(height(commandMap.ravel()[i]), heightMap.ravel()[i]))


    def test_compute_commands_for_height_map_with_crosstalk(self):
        """Test accuracy of compute_commands_for_height_map().
        For now, just test that it runs successfully with a real dm
        crosstalk  yaml"""
        nActs = self.nActs
        # For quadratic map, max height (at max voltage vector value of 100V)
        # is 400.  All but the last few entries in heightMap is <= 400.  For
        # last few entries, the height evaluated at the found command map value
        # should be off a little, limited by max voltage allowed.
        heightMap = np.linspace(0, 401, nActs**2).reshape((nActs, nActs))

        # undo DM crosstalk on height map to get true commandMap
        dmxtalk = CDMCrosstalk(self.crosstalk_fn)
        heightMapTrue = dmxtalk.crosstalk_backward(heightMap)

        commandMapTrue = (-self.b +
                          np.sqrt(self.b*self.b + 4*self.a*heightMapTrue)) \
            / (2*self.a)

        commandMap = gmu.compute_commands_for_height_map(
            heightMap,
            self.fnRefCube,
            self.fnRefCommandVec,
            vmin=np.min(commandMapTrue),
            vmax=np.max(commandMapTrue),
            crosstalk_fn=self.crosstalk_fn,
        )
        self.assertTrue(np.allclose(commandMapTrue,
                                    commandMap, atol=1e-4))



    def test_compute_heights_for_command_map(self):
        """Test accuracy of compute_heights_for_command_map()."""
        nActs = self.nActs
        commandMap = np.linspace(0, 101, nActs**2).reshape((nActs, nActs))
        heightMapTrue = self.a*commandMap**2 + self.b*commandMap + self.c

        heightMap = gmu.compute_heights_for_command_map(
            commandMap,
            self.fnRefCube,
            self.fnRefCommandVec,
        )
        ind = np.where(commandMap <= 100)
        self.assertTrue(np.allclose(heightMapTrue[ind], heightMap[ind],
                                    atol=1e-4))
        #for voltages out of range of vector, height should be what it is at
        # voltage cap; at 100V, height is 400
        ind2 = np.where(commandMap > 100)
        np.allclose(heightMap[ind2], 400.*np.ones_like(heightMap[ind2]),
                    atol=1e-9)


    def test_compute_commands_for_height_map(self):
        """Test accuracy of compute_commands_for_height_map()."""
        nActs = self.nActs
        commandMapTrue = np.linspace(0, 100, nActs**2).reshape((nActs, nActs))
        heightMap = self.a*commandMapTrue**2 + self.b*commandMapTrue + self.c
        commandMap = gmu.compute_commands_for_height_map(
            heightMap,
            self.fnRefCube,
            self.fnRefCommandVec,
            vmin=np.min(commandMapTrue),
            vmax=np.max(commandMapTrue),
        )
        self.assertTrue(np.allclose(commandMapTrue, commandMap, atol=1e-4))

    def test_compute_heights_for_command_map_with_dm_crosstalk(self):
        """Test accuracy of compute_heights_for_command_map().
        Include real non-identity crosstalk"""
        nActs = self.nActs
        commandMap = np.linspace(0, 100, nActs**2).reshape((nActs, nActs))
        heightMapTrue = self.a*commandMap**2 + self.b*commandMap + self.c

        dmxtalk = CDMCrosstalk(self.crosstalk_fn)
        heightMapTrue = dmxtalk.crosstalk_forward(heightMapTrue)

        heightMap = gmu.compute_heights_for_command_map(
            commandMap,
            self.fnRefCube,
            self.fnRefCommandVec,
            self.crosstalk_fn,
        )

        self.assertTrue(np.allclose(heightMapTrue, heightMap, atol=1e-4))

    def test_compute_hcube_from_gcube(self):
        """Test accuracy of compute_hcube_from_gcube()."""
        heightCube = gmu.compute_hcube_from_gcube(self.fnRefGainCube,
                                                  self.fnRefCommandVecGain)
        heightCubeTrue = fits.getdata(self.fnRefCube)

        self.assertTrue(np.allclose(heightCube, heightCubeTrue))

    def test_offset_compute_hcube_from_gcube(self):
        """Test accuracy of compute_hcube_from_gcube() when offset input given.
        Also, test optional inputs."""
        # For the quadratic map, h(0) = 0, so constant of integration is 0.
        # To test the input, we could also use, say, h(20) and v0=20.
        v0 = 20
        offsetMap = (self.a*v0**2+self.b*v0)*np.ones((self.nActs, self.nActs))
        heightCube = gmu.compute_hcube_from_gcube(self.fnRefGainCube,
                                                  self.fnRefCommandVecGain,
                                                  np.arange(0, 91, 10),
                                                  offsetMap, v0)

        heightCubeGen = fits.getdata(self.fnRefCube)
        # heightCubeGen has 11 slices, heightCube has 10:
        self.assertTrue(np.allclose(heightCube, heightCubeGen[:-1]))

    def test_fft_data_dm1(self):
        """Show compute_commands_for_height_map() works with FFT DM1 data.

        Also, the
        actuators outside the active annulus of actuators are typically at a
        height of 0.  If for a given actuator the height is the same for all
        voltage values, the minimizer could set any voltage to accomplish
        that height, so this test also confirms that the smallest voltage
        value (usually 0V) is selected for such actuators.
        """

        v_dm1_flat = fits.getdata(os.path.join(self.localpath,
                                    'testdata', 'FFT_data',
                                    'dm1_flat_synthesized.fits'))
        deltaHeightMapDM1 = fits.getdata(os.path.join(self.localpath,
                            'testdata', 'FFT_data',
                            'hlc_dm1_design_heights_meters_fft-17.fits'))
        fnRefCubeDM1 = os.path.join(self.localpath,
                                    'testdata', 'FFT_data',
                                    'height_cube_integrated.fits')
        fnRefCommandVecDM1 = os.path.join(self.localpath,
                                          'testdata', 'FFT_data',
                                          'height_cube_command_vec.fits')
        voltage_fn = os.path.join(self.localpath, 'testdata', 'dm.yaml')
        dm = loadyaml(voltage_fn)
        voltageDict = dm['dms']['DM1']['voltages']

        # below function utilizes compute_commands_for_height_map()
        v_dm1_hlc = gm.compute_starting_commands_for_flight(
            v_dm1_flat,
            deltaHeightMapDM1,
            fnRefCubeDM1,
            fnRefCommandVecDM1,
            voltageDict,
        )

        # verify that values other than the max and min values from the
        # reference voltage vector appear in the voltage command map:
        refVoltageVec = fits.getdata(fnRefCommandVecDM1)
        greater_min = v_dm1_hlc[v_dm1_hlc > refVoltageVec.min()]
        bw_min_max = greater_min[greater_min < refVoltageVec.max()]
        self.assertTrue(bw_min_max.size > 0)

        # [0,0] actuator has 0 height at all voltage values, so the starting
        # commanded voltage should be the minimum of the reference voltage
        # vector:

        self.assertTrue(v_dm1_hlc[0, 0] == refVoltageVec.min())

    def test_fft_data_dm2(self):
        """Show compute_commands_for_height_map() works with FFT DM2 data.

        Also, the
        actuators outside the active annulus of actuators are typically at a
        height of 0.  If for a given actuator the height is the same for all
        voltage values, the minimizer could set any voltage to accomplish
        that height, so this test also confirms that the smallest voltage
        value (usually 0V) is selected for such actuators.
        """
        crosstalk_fn = os.path.join(self.localpath,
                                    'testdata',
                                    'dm2_crosstalk_correct_orientation.yaml')
        commandMap = fits.getdata(os.path.join(self.localpath,
                                    'testdata', 'FFT_data',
                                    'dmabs_dm2_fft_flat_20231116.fits'))

        fnRefCube = os.path.join(self.localpath,
                                    'testdata', 'FFT_data',
                                    'height_cube_integrated_dm2.fits')
        fnRefCommandVec = os.path.join(self.localpath,
                                          'testdata', 'FFT_data',
                                          'height_cube_command_vec_dm2.fits')

        # below function utilizes compute_commands_for_height_map()
        v_dm1_hlc = gmu.compute_heights_for_command_map(
        commandMap, fnRefCube, fnRefCommandVec, crosstalk_fn)

        # verify that values other than the max and min values from the
        # reference voltage vector appear in the voltage command map:
        refVoltageVec = fits.getdata(fnRefCommandVec)
        greater_min = v_dm1_hlc[v_dm1_hlc > refVoltageVec.min()]
        bw_min_max = greater_min[greater_min < refVoltageVec.max()]
        self.assertTrue(bw_min_max.size > 0)

        # [0,0] actuator has 0 height at all voltage values, so the starting
        # commanded voltage should be the minimum of the reference voltage
        # vector:

        self.assertTrue(v_dm1_hlc[0, 0] == refVoltageVec.min())


class TestGainmapInputFailure(unittest.TestCase):
    """Unit tests for inputs failures of gainmap.py."""

    def setUp(self):
        """Set up variables for communal use."""
        self.localpath = os.path.dirname(os.path.abspath(__file__))
        self.fnRefCube = os.path.join(self.localpath, 'testdata',
                                      'ut_refCubeQuadratic.fits')
        self.fnRefCommandVec = os.path.join(self.localpath, 'testdata',
                                            'ut_refCommandVec.fits')
        self.fnRefGainCube = os.path.join(self.localpath, 'testdata',
                                      'ut_refCubeQuadratic_Gain.fits')
        self.fnRefCommandVecGain = os.path.join(self.localpath, 'testdata',
                                            'ut_refCommandVec_Gain.fits')
        # actual data from testbed
        self.fnRefGainCubeTestbed = os.path.join(self.localpath, 'testdata',
                                      'testbedDM1GainCube.fits')


    def test_compute_diff_and_mean_of_vector(self):
        """Test that error is raised for bad inputs. """
        # good input does not fail
        gmu.compute_diff_and_mean_of_vector(self.fnRefCommandVec)

        # Exception for non-1D array
        with self.assertRaises(ValueError):
            gmu.compute_diff_and_mean_of_vector(self.fnRefGainCube)


    def test_verify_vector_has_unique_values(self):
        """Test that error is raised if input is not a unique-valued vector."""
        # Check that good input does not fail
        gmu.verify_vector_has_unique_values((2, 3, 4))

        # Exception for non-1D array
        for badVec in (1j, 'string', np.ones((3, 3))):
            with self.assertRaises(ValueError):
                gmu.verify_vector_has_unique_values(badVec)

        # Exception for nonunique values
        for badVec in ((1, 1, 2),):
            with self.assertRaises(ValueError):
                gmu.verify_vector_has_unique_values(badVec)

    def test_verify_vector_is_presorted(self):
        """Test that error is raised if input not a sorted vector."""
        # Check that good input does not fail
        gmu.verify_vector_is_presorted((2, 3, 4))

        # Exception for non-1D array
        for badVec in (1j, 'string', np.ones((3, 3))):
            with self.assertRaises(ValueError):
                gmu.verify_vector_is_presorted(badVec)

        # Exception for unsorted array
        for badVec in ((2, 1), np.arange(0, -10, -1)):
            with self.assertRaises(ValueError):
                gmu.verify_vector_is_presorted(badVec)

    def test_compute_delta_height_map_from_command_maps_inputs(self):
        """Test bad inputs for compute_delta_height_map_from_command_maps()."""
        commandMapBefore = np.ones((48, 48))
        commandMapAfter = 2*np.ones((48, 48))

        # Check that no issues are raised with good inputs
        gmu.compute_delta_height_map_from_command_maps(
            commandMapBefore,
            commandMapAfter,
            self.fnRefCube,
            self.fnRefCommandVec
        )

        for commandMapBeforeBad in (1j, np.ones((48, 10)), np.ones((48, 10)),
                                    'string'):
            with self.assertRaises(ValueError):
                gmu.compute_delta_height_map_from_command_maps(
                    commandMapBeforeBad,
                    commandMapAfter,
                    self.fnRefCube,
                    self.fnRefCommandVec
                )
        for commandMapAfterBad in (1j, np.ones((48, 10)), np.ones((48, 10)),
                                   'string'):
            with self.assertRaises(ValueError):
                gmu.compute_delta_height_map_from_command_maps(
                    commandMapBefore,
                    commandMapAfterBad,
                    self.fnRefCube,
                    self.fnRefCommandVec
                )

    def test_compute_commands_for_height_map_inputs(self):
        """Test compute_commands_for_height_map() with bad inputs."""
        # Check that no issues are raised with good inputs
        gmu.compute_commands_for_height_map(np.ones((48, 48)),
                                            self.fnRefCube,
                                            self.fnRefCommandVec,
                                            vmin=0,
                                            vmax=100,
        )

        # Wrong type or shape for heightMap
        for heightMapBad in (1j, np.ones((48, 10)), np.ones((48, 10)), 's'):
            with self.assertRaises(ValueError):
                gmu.compute_commands_for_height_map(heightMapBad,
                                                    self.fnRefCube,
                                                    self.fnRefCommandVec,
                                                    0,
                                                    100,
                )

        # Exception for bad cube
        with self.assertRaises(ValueError):
            gmu.compute_commands_for_height_map(np.ones((48, 48)),
                                                self.fnRefCommandVec,
                                                self.fnRefCommandVec,
                                                0,
                                                100,
            )

        # cube with only 1 slice
        oneslicecube = os.path.join(self.localpath, 'testdata',
                                'ut_oneSliceCube.fits')
        lenOneVec = os.path.join(self.localpath, 'testdata',
                                'ut_lenOneVec.fits')
        with self.assertRaises(ValueError):
            gmu.compute_commands_for_height_map(np.ones((48, 48)),
                                                oneslicecube,
                                                lenOneVec,
                                                0,
                                                100,
            )

        # length of command vector should agree with number of cube slices
        fnLongVec = os.path.join(self.localpath, 'testdata',
                                'ut_vector_with_too_many_values.fits')
        with self.assertRaises(ValueError):
            gmu.compute_commands_for_height_map(np.ones((48, 48)),
                                                self.fnRefGainCube,
                                                fnLongVec,
                                                0,
                                                100,
            )

    def test_compute_commands_for_height_map_inputs_voltage(self):
        """Test compute_commands_for_height_map() with bad inputs."""

        # vmin not real scalar
        for perr in [None, 1j, (0,), 'txt']:
            with self.assertRaises(ValueError):
                gmu.compute_commands_for_height_map(np.ones((48, 48)),
                                                    self.fnRefCube,
                                                    self.fnRefCommandVec,
                                                    vmin=perr,
                                                    vmax=100,
                )

        # vmax not real scalar
        for perr in [None, 1j, (0,), 'txt']:
            with self.assertRaises(ValueError):
                gmu.compute_commands_for_height_map(np.ones((48, 48)),
                                                    self.fnRefCube,
                                                    self.fnRefCommandVec,
                                                    vmin=0,
                                                    vmax=perr,
                )

        # vmin > vmax
        with self.assertRaises(ValueError):
            gmu.compute_commands_for_height_map(np.ones((48, 48)),
                                                self.fnRefCube,
                                                self.fnRefCommandVec,
                                                vmin=100,
                                                vmax=0,
            )

        # vmin = vmax
        with self.assertRaises(ValueError):
            gmu.compute_commands_for_height_map(np.ones((48, 48)),
                                                self.fnRefCube,
                                                self.fnRefCommandVec,
                                                vmin=0,
                                                vmax=0,
            )



    def test_compute_heights_for_command_map_inputs(self):
        """Test compute_heights_for_command_maps() with bad inputs."""
        # Check that no issues are raised with good inputs
        gmu.compute_heights_for_command_map(np.ones((48, 48)),
                                            self.fnRefCube,
                                            self.fnRefCommandVec)

        # Wrong type or shape for commandMap
        for commandMapBad in (1j, np.ones((48, 10)), np.ones((48, 10)), 's'):
            with self.assertRaises(ValueError):
                gmu.compute_heights_for_command_map(commandMapBad,
                                                    self.fnRefCube,
                                                    self.fnRefCommandVec)

        # Exception for bad cube
        with self.assertRaises(ValueError):
            gmu.compute_heights_for_command_map(np.ones((48, 48)),
                                                self.fnRefCommandVec,
                                                self.fnRefCommandVec)

        # cube with only 1 slice
        oneslicecube = os.path.join(self.localpath, 'testdata',
                                'ut_oneSliceCube.fits')
        lenOneVec = os.path.join(self.localpath, 'testdata',
                                'ut_lenOneVec.fits')
        with self.assertRaises(ValueError):
            gmu.compute_heights_for_command_map(np.ones((48, 48)),
                                                oneslicecube,
                                                lenOneVec)

        # length of command vector should agree with number of cube slices
        fnLongVec = os.path.join(self.localpath, 'testdata',
                                'ut_vector_with_too_many_values.fits')
        with self.assertRaises(ValueError):
            gmu.compute_heights_for_command_map(np.ones((48, 48)),
                                                self.fnRefGainCube,
                                                fnLongVec)


    def test_compute_hcube_from_gcube_inputs(self):
        """Test compute_hcube_from_gcube() for bad inputs."""
        # good inputs fine
        gmu.compute_hcube_from_gcube(self.fnRefGainCubeTestbed,
                                     self.fnRefCommandVecGain,
                                     np.arange(10, 51, 5),
                                     np.ones((48, 48)),
                                     2.9)

        # check offsetVoltage
        for err in ut_check.rslist:
            with self.assertRaises(TypeError):
                gmu.compute_hcube_from_gcube(self.fnRefGainCube,
                                             self.fnRefCommandVecGain,
                                             np.arange(10, 51, 5),
                                             np.ones((48, 48)),
                                             err)

        # check offsetMap
        for err in ut_check.twoDlist:
            with self.assertRaises(ValueError):
                gmu.compute_hcube_from_gcube(self.fnRefGainCube,
                                             self.fnRefCommandVecGain,
                                             np.arange(10, 51, 5),
                                             err,
                                             2)

        # bad cube
        with self.assertRaises(ValueError):
            gmu.compute_hcube_from_gcube(self.fnRefCommandVec,
                                            self.fnRefCommandVec)

        # cube with only 1 slice
        oneslicecube = os.path.join(self.localpath, 'testdata',
                                'ut_oneSliceCube.fits')
        lenOneVec = os.path.join(self.localpath, 'testdata',
                                'ut_lenOneVec.fits')
        with self.assertRaises(ValueError):
            gmu.compute_hcube_from_gcube(oneslicecube, lenOneVec)

        # length of command vector should agree with number of cube slices
        fnLongVec = os.path.join(self.localpath, 'testdata',
                                'ut_vector_with_too_many_values.fits')
        with self.assertRaises(ValueError):
            gmu.compute_hcube_from_gcube(self.fnRefGainCube,
                                         fnLongVec)

if __name__ == '__main__':
    unittest.main()
