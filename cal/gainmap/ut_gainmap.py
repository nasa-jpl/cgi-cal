# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Unit test suite for gainmap module."""
import unittest
import os
import tempfile

import numpy as np
from astropy.io import fits

import cal.gainmap.gainmap as gm
from cal.gainmap.gm_util import compute_hcube_from_gcube
from cal.util.loadyaml import loadyaml
import cal.util.ut_check as ut_check


class TestGainmap(unittest.TestCase):
    """Unit tests for gainmap.py."""

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
        self.fnRefCubicGainCube = os.path.join(self.localpath, 'testdata',
                                      'ut_refCubeNegGain_Gain.fits')


        voltage_fn = os.path.join(self.localpath, 'testdata', 'dm.yaml')
        dm = loadyaml(voltage_fn)
        self.voltageDict = dm['dms']['DM1']['voltages']
        self.voltageDict_xtalk = dm['dms']['DM2']['voltages']

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

    def test_compute_gainmap(self):
        """Test the accuracy of compute_gainmap()."""
        nActs = self.nActs
        commandMap = np.linspace(0, 100, nActs**2).reshape((nActs, nActs))
        gainMapExpected = 2*self.a*commandMap + self.b
        gainMap = gm.compute_gainmap(commandMap,
                                     self.fnRefCube,
                                     self.fnRefCommandVec,
                                     )
        self.assertTrue(np.allclose(gainMap,
                                    gainMapExpected,
                                    atol=np.finfo(float).eps,)
                        )


    def test_compute_gainmap_zero(self):
        """Test compute_gainmap() for non-positive gain."""
        nActs = self.nActs
        commandMap = np.linspace(0, 100, nActs**2).reshape((nActs, nActs))
        #gain: g(v) = 3*(v-z)**2 + 2*b2*v
        gainMapExpected = 3*(commandMap-self.z)**2 + 2*self.b2*commandMap
        gainMap = gm.compute_gainmap(commandMap,
                                     self.fnRefCubicCube,
                                     self.fnRefCommandVec,
                                     )
        ind = np.where(gainMapExpected > 0)
        self.assertTrue(np.allclose(gainMap[ind],
                                    gainMapExpected[ind],
                                    atol=np.finfo(float).eps,)
                        )
        # where gain<=0 initially, it is set to the min_gain value, which is 
        # machine precision by default, if not superceded by extend_radially 
        # value
        ind2 = np.where(gainMapExpected <= 0)
        self.assertTrue((gainMap[ind2] >= np.finfo(float).eps).all())

    def test_compute_gainmap_zero_notdefault(self):
        """Test compute_gainmap() for non-positive gain and non-default 
        values for threshold and min_gain."""
        nActs = self.nActs
        threshold = 1e-9
        min_gain = 1e-8
        min_gain_map = min_gain*np.ones((nActs, nActs))
        commandMap = np.linspace(0, 100, nActs**2).reshape((nActs, nActs))
        #gain: g(v) = 3*(v-z)**2 + 2*b2*v
        gainMapExpected = 3*(commandMap-self.z)**2 + 2*self.b2*commandMap
        gainMap = gm.compute_gainmap(commandMap,
                                     self.fnRefCubicCube,
                                     self.fnRefCommandVec,
                                     threshold=threshold,
                                     min_gain=min_gain_map
                                     )
        ind = np.where(gainMapExpected > threshold)
        self.assertTrue(np.allclose(gainMap[ind],
                                    gainMapExpected[ind],
                                    atol=np.finfo(float).eps,)
                        )
        # where gain<=0 initially, it is set to the min_gain value
        ind2 = np.where(gainMapExpected <= threshold)
        self.assertTrue((gainMap[ind2] >= min_gain).all())
        

    def test_compute_gainmap_gcube(self):
        """Test the accuracy of compute_gainmap(). Start with provided gain
        cube, make height cube from it, then run it through
        compute_gainmap()."""
        nActs = self.nActs
        commandMap = np.linspace(0, 100, nActs**2).reshape((nActs, nActs))

        gainMapExpected = 2*self.a*commandMap + self.b

        heightCube = compute_hcube_from_gcube(self.fnRefGainCube,
                                              self.fnRefCommandVecGain
                                              )
        # make tmp file to save height cube and height vector
        tmp_fd, tmp_heightcube_fits = tempfile.mkstemp(suffix='.fits')

        hdr = fits.Header()
        prim = fits.PrimaryHDU(header=hdr)
        img = fits.ImageHDU(heightCube)
        hdul = fits.HDUList([prim, img])
        hdul.writeto(tmp_heightcube_fits)

        gainMap = gm.compute_gainmap(commandMap,
                                     tmp_heightcube_fits,
                                     self.fnRefCommandVec
                                     )

        self.assertTrue(np.allclose(gainMap,
                                    gainMapExpected
                                    )
                        )
        # now delete the file created
        os.close(tmp_fd)
        os.unlink(tmp_heightcube_fits)

    def test_compute_starting_commands_for_flight(self):
        """Test the accuracy of compute_starting_commands_for_flight()."""
        flatCommandMap = 50 * np.ones((self.nActs, self.nActs))
        deltaHeightMap = 100 * np.ones((self.nActs, self.nActs))

        heightMap = (self.a*flatCommandMap**2 + self.b*flatCommandMap +
                     deltaHeightMap)
        startingCommandMapTrue = ((-self.b + np.sqrt(self.b*self.b +
                                                     4*self.a*heightMap)) /
                                  (2*self.a))

        startingCommandMap = gm.compute_starting_commands_for_flight(
            flatCommandMap,
            deltaHeightMap,
            self.fnRefCube,
            self.fnRefCommandVec,
            self.voltageDict,
        )

        self.assertTrue(np.allclose(startingCommandMapTrue,
                                    startingCommandMap,
                                    atol=1e-1,
                                    )
                        )

    def test_compute_starting_commands_for_flight_func_xtalk(self):
        """
        Verify compute_starting_commands_for_flight() completes with
        crosstalk present without error.
        """
        flatCommandMap = 50 * np.ones((self.nActs, self.nActs))
        deltaHeightMap = 100 * np.ones((self.nActs, self.nActs))

        gm.compute_starting_commands_for_flight(
            flatCommandMap,
            deltaHeightMap,
            self.fnRefCube,
            self.fnRefCommandVec,
            self.voltageDict_xtalk,
        )


    def test_gain_to_height_starting_commands_for_flight(self):
        """Test the accuracy of compute_starting_commands_for_flight() when
        a gain cube is provided initially."""
        flatCommandMap = 50 * np.ones((self.nActs, self.nActs))
        deltaHeightMap = 100 * np.ones((self.nActs, self.nActs))

        heightMap = (self.a*flatCommandMap**2 + self.b*flatCommandMap +
                     deltaHeightMap)
        # solving heightMap = 2*a*x**2 + b*x for x
        startingCommandMapTrue = ((-self.b + np.sqrt(self.b*self.b +
                                    4*self.a*(heightMap))) /
                                   (2*self.a))

        # compute the height cube from the gain cube:
        heightCube = compute_hcube_from_gcube(
            self.fnRefGainCube, self.fnRefCommandVecGain,
            heightCommandVec=fits.getdata(self.fnRefCommandVec))

        # make tmp file to save height cube and height vector
        tmp_fd, tmp_heightcube_fits = tempfile.mkstemp(suffix='.fits')

        hdr = fits.Header()
        prim = fits.PrimaryHDU(header=hdr)
        img = fits.ImageHDU(heightCube)
        hdul = fits.HDUList([prim, img])
        hdul.writeto(tmp_heightcube_fits)

        startingCommandMap = gm.compute_starting_commands_for_flight(
            flatCommandMap,
            deltaHeightMap,
            tmp_heightcube_fits,
            self.fnRefCommandVec,
            self.voltageDict,
        )

        self.assertTrue(np.allclose(startingCommandMapTrue,
                                    startingCommandMap,
                                    atol=1e-1,
                                    )
                        )
        # now delete the file created
        os.close(tmp_fd)
        os.unlink(tmp_heightcube_fits)


    def test_compute_starting_commands_for_flight_lower_bound(self):
        """Test the lower bound of compute_starting_commands_for_flight()."""
        flatCommandMap = 50 * np.ones((self.nActs, self.nActs))
        deltaHeightMap = -1e3 * np.ones((self.nActs, self.nActs))

        startingCommandMapTrue = np.zeros((self.nActs, self.nActs))

        startingCommandMap = gm.compute_starting_commands_for_flight(
            flatCommandMap,
            deltaHeightMap,
            self.fnRefCube,
            self.fnRefCommandVec,
            self.voltageDict,
        )

        self.assertTrue(np.allclose(startingCommandMapTrue,
                                    startingCommandMap,
                                    atol=1e-4,
                                    )
                        )

    def test_compute_starting_commands_for_flight_upper_bound(self):
        """Test the upper bound of compute_starting_commands_for_flight()."""
        flatCommandMap = 50 * np.ones((self.nActs, self.nActs))
        deltaHeightMap = 1e3 * np.ones((self.nActs, self.nActs))

        startingCommandMapTrue = 100 * np.ones((self.nActs, self.nActs))

        startingCommandMap = gm.compute_starting_commands_for_flight(
            flatCommandMap,
            deltaHeightMap,
            self.fnRefCube,
            self.fnRefCommandVec,
            self.voltageDict,
        )

        self.assertTrue(np.allclose(startingCommandMapTrue,
                                    startingCommandMap,
                                    atol=1e-4,
                                    )
                        )

    def test_compute_gainmap_from_gain_cube(self):
        """Test the accuracy of compute_gainmap_from_gain_cube()."""
        nActs = self.nActs
        commandMap = np.linspace(0, 101, nActs**2).reshape((nActs, nActs))
        gainMapExpected = 2*self.a*(commandMap) + self.b
        gainMap = gm.compute_gainmap_from_gain_cube(commandMap,
                                                    self.fnRefGainCube,
                                                    self.fnRefCommandVecGain)
        v_ref_vec = fits.getdata(self.fnRefCommandVecGain)
        v_ref_min = np.min(v_ref_vec)
        v_ref_max = np.max(v_ref_vec)
        gainMapExpected[commandMap < v_ref_min] = 2*self.a*v_ref_min + self.b
        gainMapExpected[commandMap > v_ref_max] = 2*self.a*v_ref_max + self.b

        self.assertTrue(np.allclose(gainMap, gainMapExpected))
        # gets capped beyond voltage bounds
        self.assertTrue(np.allclose(gainMap[commandMap > v_ref_max], 
                        np.ones_like(gainMap[commandMap > v_ref_max])
                        *2*self.a*v_ref_max + self.b, atol=1e-4))

    def test_compute_gainmap_from_gain_cube_zero(self):
        """Test compute_gainmap_from_gain_cube() for non-positive gain."""
        nActs = self.nActs
        commandMap = np.linspace(10, 90, nActs**2).reshape((nActs, nActs))
        #gain: g(v) = 3*(v-z)**2 + 2*b2*v
        gainMapExpected = 3*(commandMap-self.z)**2 + 2*self.b2*commandMap
        gainMap = gm.compute_gainmap_from_gain_cube(commandMap,
                                                    self.fnRefCubicGainCube,
                                                    self.fnRefCommandVecGain,
                                                    )
        ind = np.where(gainMapExpected > 0)
        self.assertTrue(np.allclose(gainMap[ind],
                                    gainMapExpected[ind],
                                    atol=np.finfo(float).eps,)
                        )
        # where gain<=0 initially, it is set to the min_gain value, which is 
        # machine precision by default, if not superceded by extend_radially 
        # value
        ind2 = np.where(gainMapExpected <= 0)
        self.assertTrue((gainMap[ind2] >= np.finfo(float).eps).all())


    def test_compute_gainmap_from_gain_cube_zero_nondefault(self):
        """Test compute_gainmap_from_gain_cube() for non-positive gain and 
        non-default values for threshold and min_gain."""
        nActs = self.nActs
        commandMap = np.linspace(10, 90, nActs**2).reshape((nActs, nActs))
        threshold = 1e-9
        min_gain = 1e-8
        min_gain_map = min_gain*np.ones((nActs, nActs))
        #gain: g(v) = 3*(v-z)**2 + 2*b2*v
        gainMapExpected = 3*(commandMap-self.z)**2 + 2*self.b2*commandMap
        gainMap = gm.compute_gainmap_from_gain_cube(commandMap,
                                                    self.fnRefCubicGainCube,
                                                    self.fnRefCommandVecGain,
                                                    threshold=threshold,
                                                    min_gain=min_gain_map
                                                    )
        ind = np.where(gainMapExpected > threshold)
        self.assertTrue(np.allclose(gainMap[ind],
                                    gainMapExpected[ind],
                                    atol=np.finfo(float).eps,)
                        )
        # where gain<=0 initially, it is set to the min_gain value
        ind2 = np.where(gainMapExpected <= threshold)
        self.assertTrue((gainMap[ind2] >= min_gain).all())


class TestGainmapInputFailure(unittest.TestCase):
    """Unit tests for inputs failures of gainmap.py."""

    def setUp(self):
        """Set up variables for communal use."""
        self.localpath = os.path.dirname(os.path.abspath(__file__))
        self.fnRefCubeLinear = os.path.join(self.localpath, 'testdata',
                                      'ut_refCubeQuadratic.fits')
        self.fnRefCommandVec = os.path.join(self.localpath, 'testdata',
                                            'ut_refCommandVec.fits')
        self.fnRefGainCube = os.path.join(self.localpath, 'testdata',
                                      'ut_refCubeQuadratic_Gain.fits')
        self.fnRefCommandVecGain = os.path.join(self.localpath, 'testdata',
                                            'ut_refCommandVec_Gain.fits')


        voltage_fn = os.path.join(self.localpath, 'testdata', 'dm.yaml')
        dm = loadyaml(voltage_fn)
        self.voltageDict = dm['dms']['DM1']['voltages']


    def test_compute_gainmap_input(self):
        """Test compute_gainmap() with bad inputs."""
        # filename errors are handled directly by fits.getdata
        for commandMapBad in (1j, np.ones((48, 10)), np.ones((48, 10)), 's'):
            with self.assertRaises(ValueError):
                gm.compute_gainmap(commandMapBad,
                                   self.fnRefCubeLinear,
                                   self.fnRefCommandVec,
                                   )
        
        # command map with voltages outside of bounds of vector (not regulated
        # first with constrain_dm())
        badCommandMap = np.ones((48, 48))
        badCommandMap[2,2] = 101
        with self.assertRaises(ValueError):
            gm.compute_gainmap(badCommandMap,
                               self.fnRefCubeLinear,
                               self.fnRefCommandVec,
                               )

        # bad cube
        with self.assertRaises(ValueError):
            gm.compute_gainmap(np.ones((48, 48)),
                               self.fnRefCommandVec,
                               self.fnRefCommandVec,
                               )
            
       # cube with only 1 slice
        oneslicecube = os.path.join(self.localpath, 'testdata',
                                'ut_oneSliceCube.fits')
        lenOneVec = os.path.join(self.localpath, 'testdata',
                                'ut_lenOneVec.fits')
        with self.assertRaises(ValueError):
            gm.compute_gainmap(np.ones((48, 48)),
                                oneslicecube,
                                lenOneVec)

        # must have agreement b/w vector length and number of cube slices
        fnLongVec = os.path.join(self.localpath, 'testdata',
                                'ut_vector_with_too_many_values.fits')
        with self.assertRaises(ValueError):
            gm.compute_gainmap(np.ones((48, 48)),
                               self.fnRefCubeLinear,
                               fnLongVec,
                               )
        
        # threshold < 0
        for err in ut_check.rnslist:
            with self.assertRaises(ValueError):
                gm.compute_gainmap(np.ones((2, 48, 48)),
                                                    self.fnRefCubeLinear,
                                                    self.fnRefCommandVec,
                                                    threshold=err)
        
        # min_gain not a 2D array
        for err in ut_check.twoDlist:
            with self.assertRaises(ValueError):
                gm.compute_gainmap(np.ones((2, 48, 48)),
                                                    self.fnRefCubeLinear,
                                                    self.fnRefCommandVec,
                                                    min_gain=err)
                
        # min_gain with a non-positive value
        with self.assertRaises(ValueError):
            min_gain = np.ones((48,48))
            min_gain[2,3] = 0
            gm.compute_gainmap(np.ones((2, 48, 48)),
                                                self.fnRefCubeLinear,
                                                self.fnRefCommandVec,
                                                min_gain=min_gain)

    def test_compute_starting_commands_for_flight_inputs(self):
        """Test compute_starting_commands_for_flight() with bad inputs."""
        flatCommandMap = np.ones((48, 48))
        deltaHeightMap = np.ones((48, 48))

        for flatCommandMapBad in (1j, np.ones((48,)), np.ones((48, 10, 2)),
                                  None):
            with self.assertRaises(TypeError):
                gm.compute_starting_commands_for_flight(flatCommandMapBad,
                                                        deltaHeightMap,
                                                        self.fnRefCubeLinear,
                                                        self.fnRefCommandVec,
                                                        self.voltageDict,
                )

        for deltaHeightMapBad in (1j, np.ones((48,)), np.ones((48, 10, 2)),
                                  None):
            with self.assertRaises(TypeError):
                gm.compute_starting_commands_for_flight(flatCommandMap,
                                                        deltaHeightMapBad,
                                                        self.fnRefCubeLinear,
                                                        self.fnRefCommandVec,
                                                        self.voltageDict,
                )

        for fnRefCubeBad in (0, 1j, ('txt',), None, np.ones((48, 10))):
            with self.assertRaises(TypeError):
                gm.compute_starting_commands_for_flight(flatCommandMap,
                                                        deltaHeightMap,
                                                        fnRefCubeBad,
                                                        self.fnRefCommandVec,
                                                        self.voltageDict,
                )

        for fnRefCommandVecBad in (0, 1j, ('txt',), None, np.ones((48, 10))):
            with self.assertRaises(TypeError):
                gm.compute_starting_commands_for_flight(flatCommandMap,
                                                        deltaHeightMap,
                                                        self.fnRefCubeLinear,
                                                        fnRefCommandVecBad,
                                                        self.voltageDict,
                )

        for voltageDictBad in (0, 1j, ('txt',), None, np.ones((48, 10))):
            with self.assertRaises(TypeError):
                gm.compute_starting_commands_for_flight(flatCommandMap,
                                                        deltaHeightMap,
                                                        self.fnRefCubeLinear,
                                                        self.fnRefCommandVec,
                                                        voltageDictBad,
                )

        for dataPathBad in (0, 1j, ('txt',), None, np.ones((48, 10))):
            with self.assertRaises(TypeError):
                gm.compute_starting_commands_for_flight(flatCommandMap,
                                                        deltaHeightMap,
                                                        self.fnRefCubeLinear,
                                                        self.fnRefCommandVec,
                                                        self.voltageDict,
                                                        data_path=dataPathBad,
                )

    def test_compute_starting_commands_for_flight_inputs_diff_size(self):
        """Test compute_starting_commands_for_flight() with bad inputs."""
        flatCommandMap = np.ones((48, 48))
        deltaHeightMapBad = np.ones((48, 10))

        with self.assertRaises(ValueError):
            gm.compute_starting_commands_for_flight(flatCommandMap,
                                                    deltaHeightMapBad,
                                                    self.fnRefCubeLinear,
                                                    self.fnRefCommandVec,
                                                    self.voltageDict,
                                                    )


    def test_compute_gainmap_from_gain_cube_inputs(self):
        """Test compute_gainmap_from_gain_cube() with bad inputs."""

        # commandMap as non-2-D array and with bad shape
        for CommandMapBad in (1j, np.ones((48, 10)), np.ones((48, 10))):
            with self.assertRaises(ValueError):
                gm.compute_gainmap_from_gain_cube(CommandMapBad,
                                                  self.fnRefGainCube,
                                                  self.fnRefCommandVecGain
                                                  )

        # bad cube
        with self.assertRaises(ValueError):
            gm.compute_gainmap_from_gain_cube(np.ones((48, 48)),
                                              self.fnRefCommandVec,
                                              self.fnRefCommandVec,
                                              )

        # cube with only 1 slice
        oneslicecube = os.path.join(self.localpath, 'testdata',
                                'ut_oneSliceCube.fits')
        lenOneVec = os.path.join(self.localpath, 'testdata',
                                'ut_lenOneVec.fits')
        with self.assertRaises(ValueError):
            gm.compute_gainmap(np.ones((48, 48)),
                                oneslicecube,
                                lenOneVec)

        # must have agreement b/w vector length and number of cube slices
        fnLongVec = os.path.join(self.localpath, 'testdata',
                                'ut_vector_with_too_many_values.fits')
        with self.assertRaises(ValueError):
            gm.compute_gainmap_from_gain_cube(np.ones((48, 48)),
                                              self.fnRefCubeLinear,
                                              fnLongVec,
                                              )
            
        # threshold < 0
        for err in ut_check.rnslist:
            with self.assertRaises(ValueError):
                gm.compute_gainmap_from_gain_cube(np.ones((2, 48, 48)),
                                                    self.fnRefCubeLinear,
                                                    self.fnRefCommandVec,
                                                    threshold=err)
        
        # min_gain not a 2D array
        for err in ut_check.twoDlist:
            with self.assertRaises(ValueError):
                gm.compute_gainmap_from_gain_cube(np.ones((2, 48, 48)),
                                                    self.fnRefCubeLinear,
                                                    self.fnRefCommandVec,
                                                    min_gain=err)
                
        # min_gain with a non-positive value
        with self.assertRaises(ValueError):
            min_gain = np.ones((48,48))
            min_gain[2,3] = 0
            gm.compute_gainmap_from_gain_cube(np.ones((2, 48, 48)),
                                                self.fnRefCubeLinear,
                                                self.fnRefCommandVec,
                                                min_gain=min_gain)

if __name__ == '__main__':
    unittest.main()
