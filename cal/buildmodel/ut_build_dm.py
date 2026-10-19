# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Unit tests for assembling DM part of model."""
import unittest
import tempfile
import os
import yaml

from cal.util.loadyaml import loadyaml
from cal.util.writeyaml import writeyaml

from .build_dm import (
    add_pitch,
    add_z,
    add_influence_function,
    add_dm_registration,
    add_dm_voltage_bounds,
    add_dm_arrays,
)


class TestAddDMArrays(unittest.TestCase):
    """Tests for DM arrays (gain/flat/tie)."""

    def setUp(self):
        self.fd, self.name = tempfile.mkstemp()

        self.fd1, self.name1 = tempfile.mkstemp()
        mdict = dict()
        writeyaml(mdict, self.name1)

        self.fd2, self.name2 = tempfile.mkstemp()
        mdict['dms'] = dict()
        writeyaml(mdict, self.name2)

        self.fd3, self.name3 = tempfile.mkstemp()
        mdict['dms']['DM1'] = dict()
        mdict['dms']['DM2'] = dict()
        writeyaml(mdict, self.name3)

        self.fd4, self.name4 = tempfile.mkstemp()
        mdict['dms']['DM1']['voltages'] = dict()
        mdict['dms']['DM2']['voltages'] = dict()
        writeyaml(mdict, self.name4)
        pass

    def tearDown(self):
        # mkstemp() does not clean up after itself, but it does have a name
        os.close(self.fd)
        os.unlink(self.name)
        os.close(self.fd1)
        os.unlink(self.name1)
        os.close(self.fd2)
        os.unlink(self.name2)
        os.close(self.fd3)
        os.unlink(self.name3)
        os.close(self.fd4)
        os.unlink(self.name4)
        pass

    def test_success(self):
        """Good inputs complete without error."""
        for name in (self.name,
                     self.name1,
                     self.name2,
                     self.name3,
                     self.name4):
            add_dm_arrays(name, 'DM1', 'gain.fits', 'flat.fits',
                          'tie.fits', 'xtalk.yaml')
            add_dm_arrays(name, 'DM2', 'gain.fits', 'flat.fits',
                          'tie.fits', None)
        pass

    def test_readback(self):
        """Check that the output file has contents as expected."""
        # two elements, one to be skipped and one to be overwritten
        d0 = {'dms': {'DM1': {'foo': 0, 'voltages': {'gainfn': 'other.fits'}}}}

        for name in (self.name,
                     self.name1,
                     self.name2,
                     self.name3,
                     self.name4):
            writeyaml(d0, name)
            add_dm_arrays(name, 'DM1', 'gain.fits', 'flat.fits',
                          'tie.fits', 'xtalk.yaml')
            add_dm_arrays(name, 'DM2', 'gain.fits', 'flat.fits',
                          'tie.fits', None)
            d1 = loadyaml(name)

            # unrelated key unaffected
            self.assertTrue(d0['dms']['DM1']['foo'] ==
                            d1['dms']['DM1']['foo'])
            # overwritten key overwritten
            self.assertFalse(d0['dms']['DM1']['voltages']['gainfn'] ==
                             d1['dms']['DM1']['voltages']['gainfn'])
            # new keys present
            self.assertTrue('gainfn' in d1['dms']['DM2']['voltages'])
            self.assertTrue('flatfn' in d1['dms']['DM2']['voltages'])
            self.assertTrue('tiefn' in d1['dms']['DM2']['voltages'])
            self.assertTrue('crosstalkfn' in d1['dms']['DM2']['voltages'])

            self.assertTrue('flatfn' in d1['dms']['DM1']['voltages'])
            self.assertTrue('tiefn' in d1['dms']['DM1']['voltages'])
            self.assertTrue('crosstalkfn' in d1['dms']['DM1']['voltages'])
        pass

    def test_loaded_top_level_dict_not_a_dict(self):
        """Catch when something loaded isn't a dict when it should be."""
        fd, name = tempfile.mkstemp()
        mdict = [2, 3, 4]
        with open(name, 'w') as FILE:
            yaml.dump(mdict,
                      FILE,
                      default_flow_style=False,  # enforces block style
                      sort_keys=False,
                      )

        with self.assertRaises(TypeError):
            add_dm_arrays(name, 'DM1', 'gain.fits', 'flat.fits',
                          'tie.fits', 'xtalk.yaml')
        with self.assertRaises(TypeError):
            add_dm_arrays(name, 'DM2', 'gain.fits', 'flat.fits',
                          'tie.fits', None)

        # Cleanup
        os.close(fd)
        os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_DM1(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'dms': list()},
            {'dms': {'DM1': list()}},
            {'dms': {'DM1': {'voltages': list()}}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_dm_arrays(name, 'DM1', 'gain.fits', 'flat.fits',
                              'tie.fits', 'xtalk.yaml')

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_DM2(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'dms': list()},
            {'dms': {'DM2': list()}},
            {'dms': {'DM2': {'voltages': list()}}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_dm_arrays(name, 'DM2', 'gain.fits', 'flat.fits',
                              'tie.fits', None)

            # Cleanup
            os.close(fd)
            os.unlink(name)


    def test_invalid_fn(self):
        """Invalid inputs caught."""
        perrlist = [0, 0.5, 1j, None, (3, 3)]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_dm_arrays(perr, 'DM1', 'gain.fits', 'flat.fits',
                              'tie.fits', 'xtalk.yaml')

    def test_invalid_dm_id(self):
        """Invalid inputs caught."""
        perrlist = [0, 0.5, 1j, None, (3, 3), 'txt',
                    'DM0', 'DM3',
                    'dm1', 'dm2']

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_dm_arrays(self.name, perr, 'gain.fits', 'flat.fits',
                              'tie.fits', 'xtalk.yaml')

    def test_invalid_gainfn(self):
        """Invalid inputs caught."""
        perrlist = [0, 0.5, 1j, None, (3, 3)]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_dm_arrays(self.name, 'DM1', perr, 'flat.fits',
                              'tie.fits', 'xtalk.yaml')

    def test_invalid_flatfn(self):
        """Invalid inputs caught."""
        perrlist = [0, 0.5, 1j, None, (3, 3)]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_dm_arrays(self.name, 'DM1', 'gain.fits', perr,
                              'tie.fits', 'xtalk.yaml')

    def test_invalid_tiefn(self):
        """Invalid inputs caught."""
        perrlist = [0, 0.5, 1j, None, (3, 3)]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_dm_arrays(self.name, 'DM1', 'gain.fits', 'flat.fits',
                              perr, 'xtalk.yaml')

    def test_invalid_crosstalkfn(self):
        """Invalid inputs caught."""
        perrlist = [0, 0.5, 1j, (3, 3)]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_dm_arrays(self.name, 'DM1', 'gain.fits', 'flat.fits',
                              'tie.fits', perr)


class TestAddDMVoltageBounds(unittest.TestCase):
    """Tests for DM voltage constraint information."""

    def setUp(self):
        self.fd, self.name = tempfile.mkstemp()

        self.fd1, self.name1 = tempfile.mkstemp()
        mdict = dict()
        writeyaml(mdict, self.name1)

        self.fd2, self.name2 = tempfile.mkstemp()
        mdict['dms'] = dict()
        writeyaml(mdict, self.name2)

        self.fd3, self.name3 = tempfile.mkstemp()
        mdict['dms']['DM1'] = dict()
        mdict['dms']['DM2'] = dict()
        writeyaml(mdict, self.name3)

        self.fd4, self.name4 = tempfile.mkstemp()
        mdict['dms']['DM1']['voltages'] = dict()
        mdict['dms']['DM2']['voltages'] = dict()
        writeyaml(mdict, self.name4)

        pass

    def tearDown(self):
        # mkstemp() does not clean up after itself, but it does have a name
        os.close(self.fd)
        os.unlink(self.name)
        os.close(self.fd1)
        os.unlink(self.name1)
        os.close(self.fd2)
        os.unlink(self.name2)
        os.close(self.fd3)
        os.unlink(self.name3)
        os.close(self.fd4)
        os.unlink(self.name4)
        pass


    def test_success(self):
        """Good inputs complete without error."""
        for name in (self.name,
                     self.name1,
                     self.name2,
                     self.name3,
                     self.name4):
            add_dm_voltage_bounds(name, 'DM1', 0, 100, 50, 75, 0.001)
            add_dm_voltage_bounds(name, 'DM2', 0, 100, 50, 75, 0.001)
        pass


    def test_readback(self):
        """Check that the output file has contents as expected."""
        # two elements, one to be skipped and one to be overwritten
        d0 = {'dms': {'DM1': {'foo': 0, 'voltages': {'vmin': 5}}}}

        for name in (self.name,
                     self.name1,
                     self.name2,
                     self.name3,
                     self.name4):

            writeyaml(d0, name)
            add_dm_voltage_bounds(name, 'DM1', 0, 100, 50, 75, 0.001)
            add_dm_voltage_bounds(name, 'DM2', 0, 100, 50, 75, 0.001)
            d1 = loadyaml(name)

            # unrelated key unaffected
            self.assertTrue(d0['dms']['DM1']['foo'] ==
                            d1['dms']['DM1']['foo'])
            # overwritten key overwritten
            self.assertFalse(d0['dms']['DM1']['voltages']['vmin'] ==
                             d1['dms']['DM1']['voltages']['vmin'])
            # new keys present
            self.assertTrue('vmin' in d1['dms']['DM2']['voltages'])
            self.assertTrue('vmax' in d1['dms']['DM2']['voltages'])
            self.assertTrue('vneighbor' in d1['dms']['DM2']['voltages'])
            self.assertTrue('vcorner' in d1['dms']['DM2']['voltages'])
            self.assertTrue('vquant' in d1['dms']['DM2']['voltages'])

            self.assertTrue('vmax' in d1['dms']['DM1']['voltages'])
            self.assertTrue('vneighbor' in d1['dms']['DM1']['voltages'])
            self.assertTrue('vcorner' in d1['dms']['DM1']['voltages'])
            self.assertTrue('vquant' in d1['dms']['DM1']['voltages'])

        pass


    def test_loaded_top_level_dict_not_a_dict(self):
        """Catch when something loaded isn't a dict when it should be."""
        fd, name = tempfile.mkstemp()
        mdict = [2, 3, 4]
        with open(name, 'w') as FILE:
            yaml.dump(mdict,
                      FILE,
                      default_flow_style=False,  # enforces block style
                      sort_keys=False,
                      )

        with self.assertRaises(TypeError):
            add_dm_voltage_bounds(name, 'DM1', 0, 100, 50, 75, 0.001)
        with self.assertRaises(TypeError):
            add_dm_voltage_bounds(name, 'DM2', 0, 100, 50, 75, 0.001)

        # Cleanup
        os.close(fd)
        os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_DM1(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'dms': list()},
            {'dms': {'DM1': list()}},
            {'dms': {'DM1': {'voltages': list()}}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_dm_voltage_bounds(name, 'DM1', 0, 100, 50, 75, 0.001)

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_DM2(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'dms': list()},
            {'dms': {'DM2': list()}},
            {'dms': {'DM2': {'voltages': list()}}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_dm_voltage_bounds(name, 'DM2', 0, 100, 50, 75, 0.001)

            # Cleanup
            os.close(fd)
            os.unlink(name)


    def test_invalid_fn(self):
        """Invalid inputs caught."""
        perrlist = [0, 0.5, 1j, None, (3, 3)]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_dm_voltage_bounds(perr, 'DM1', 0, 100, 50, 75, 0.001)
            pass
        pass


    def test_invalid_dm_id(self):
        """Invalid inputs caught."""
        perrlist = [0, 0.5, 1j, None, (3, 3), 'txt',
                    'DM0', 'DM3',
                    'dm1', 'dm2']

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_dm_voltage_bounds(self.name, perr, 0, 100, 50, 75, 0.001)
            pass
        pass


    def test_invalid_vmin(self):
        """Invalid inputs caught."""
        perrlist = [1j, None, (3, 3)]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_dm_voltage_bounds(self.name, 'DM1',
                                      perr, 100, 50, 75, 0.001)
            pass
        pass


    def test_invalid_vmax(self):
        """Invalid inputs caught."""
        perrlist = [1j, None, (3, 3)]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_dm_voltage_bounds(self.name, 'DM1',
                                      0, perr, 50, 75, 0.001)
            pass
        pass


    def test_invalid_vneighbor(self):
        """Invalid inputs caught."""
        perrlist = [0, -0.5, 1j, None, (3, 3)]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_dm_voltage_bounds(self.name, 'DM1',
                                      0, 100, perr, 75, 0.001)
            pass
        pass


    def test_invalid_vcorner(self):
        """Invalid inputs caught."""
        perrlist = [0, -0.5, 1j, None, (3, 3)]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_dm_voltage_bounds(self.name, 'DM1',
                                      0, 100, 50, perr, 0.001)
            pass
        pass


    def test_invalid_vquant(self):
        """Invalid inputs caught."""
        perrlist = [0, -0.5, 1j, None, (3, 3)]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_dm_voltage_bounds(self.name, 'DM1',
                                      0, 100, 50, 75, perr)
            pass
        pass


class TestAddDMRegistration(unittest.TestCase):
    """Tests for DM registration data."""

    def setUp(self):
        self.fd, self.name = tempfile.mkstemp()

        self.fd1, self.name1 = tempfile.mkstemp()
        mdict = dict()
        writeyaml(mdict, self.name1)

        self.fd2, self.name2 = tempfile.mkstemp()
        mdict['dms'] = dict()
        writeyaml(mdict, self.name2)

        self.fd3, self.name3 = tempfile.mkstemp()
        mdict['dms']['DM1'] = dict()
        mdict['dms']['DM2'] = dict()
        writeyaml(mdict, self.name3)

        self.fd4, self.name4 = tempfile.mkstemp()
        mdict['dms']['DM1']['registration'] = dict()
        mdict['dms']['DM2']['registration'] = dict()
        writeyaml(mdict, self.name4)

    def tearDown(self):
        # mkstemp() does not clean up after itself, but it does have a name
        os.close(self.fd)
        os.unlink(self.name)
        os.close(self.fd1)
        os.unlink(self.name1)
        os.close(self.fd2)
        os.unlink(self.name2)
        os.close(self.fd3)
        os.unlink(self.name3)
        os.close(self.fd4)
        os.unlink(self.name4)

    def test_success(self):
        """Good inputs complete without error."""
        for name in (self.name,
                     self.name1,
                     self.name2,
                     self.name3,
                     self.name4):
            add_dm_registration(name, 'DM1', 48, 8, 8, 0, 0, 0, False)
            add_dm_registration(name, 'DM2', 48, 7, 7, 1, 1, 1, True)

    def test_readback(self):
        """Check that the output file has contents as expected."""
        # two elements, one to be skipped and one to be overwritten
        d0 = {'dms': {'DM1': {'foo': 0, 'registration': {'ppact_cx': 5}}}}

        for name in (self.name,
                     self.name1,
                     self.name2,
                     self.name3,
                     self.name4):

            writeyaml(d0, name)
            add_dm_registration(name, 'DM1', 48, 8, 8, 0, 0, 0, False)
            add_dm_registration(name, 'DM2', 48, 7, 7, 1, 1, 1, True)
            d1 = loadyaml(name)

            # unrelated key unaffected
            self.assertTrue(d0['dms']['DM1']['foo'] ==
                            d1['dms']['DM1']['foo'])
            # overwritten key overwritten
            self.assertFalse(d0['dms']['DM1']['registration']['ppact_cx'] ==
                             d1['dms']['DM1']['registration']['ppact_cx'])
            # new keys present
            self.assertTrue('nact' in d1['dms']['DM2']['registration'])
            self.assertTrue('ppact_cx' in d1['dms']['DM2']['registration'])
            self.assertTrue('ppact_cy' in d1['dms']['DM2']['registration'])
            self.assertTrue('dx' in d1['dms']['DM2']['registration'])
            self.assertTrue('dy' in d1['dms']['DM2']['registration'])
            self.assertTrue('thact' in d1['dms']['DM2']['registration'])
            self.assertTrue('flipx' in d1['dms']['DM2']['registration'])

            self.assertTrue('nact' in d1['dms']['DM1']['registration'])
            self.assertTrue('ppact_cy' in d1['dms']['DM1']['registration'])
            self.assertTrue('dx' in d1['dms']['DM1']['registration'])
            self.assertTrue('dy' in d1['dms']['DM1']['registration'])
            self.assertTrue('thact' in d1['dms']['DM1']['registration'])
            self.assertTrue('flipx' in d1['dms']['DM1']['registration'])

    def test_loaded_top_level_dict_not_a_dict(self):
        """Catch when something loaded isn't a dict when it should be."""
        fd, name = tempfile.mkstemp()
        mdict = [2, 3, 4]
        with open(name, 'w') as FILE:
            yaml.dump(mdict,
                      FILE,
                      default_flow_style=False,  # enforces block style
                      sort_keys=False,
                      )

        with self.assertRaises(TypeError):
            add_dm_registration(name, 'DM1', 48, 8, 8, 0, 0, 0, False)
        with self.assertRaises(TypeError):
            add_dm_registration(name, 'DM2', 48, 8, 8, 0, 0, 0, False)

        # Cleanup
        os.close(fd)
        os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_DM1(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'dms': list()},
            {'dms': {'DM1': list()}},
            {'dms': {'DM1': {'registration': list()}}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_dm_registration(name, 'DM1', 48, 8, 8, 0, 0, 0, False)

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_DM2(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'dms': list()},
            {'dms': {'DM2': list()}},
            {'dms': {'DM2': {'registration': list()}}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_dm_registration(name, 'DM2', 48, 8, 8, 0, 0, 0, False)

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_invalid_fn(self):
        """Invalid inputs caught."""
        perrlist = [0, 0.5, 1j, None, (3, 3)]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_dm_registration(perr, 'DM1', 48, 8, 8, 0, 0, 0, False)

    def test_invalid_dm_id(self):
        """Invalid inputs caught."""
        perrlist = [0, 0.5, 1j, None, (3, 3), 'txt',
                    'DM0', 'DM3',
                    'dm1', 'dm2']

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_dm_registration(self.name, perr, 48, 8, 8, 0, 0, 0, False)

    def test_invalid_nact(self):
        """Invalid inputs caught."""
        perrlist = [0, -0.5, 0.5, 1j, None, (3, 3)]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_dm_registration(
                    self.name, 'DM1', perr, 8, 8, 0, 0, 0, False)

    def test_invalid_ppact_cx(self):
        """Invalid inputs caught."""
        perrlist = [0, -0.5, 1j, None, (3, 3)]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_dm_registration(
                    self.name, 'DM1', 48, perr, 8, 0, 0, 0, False)

    def test_invalid_ppact_cy(self):
        """Invalid inputs caught."""
        perrlist = [0, -0.5, 1j, None, (3, 3)]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_dm_registration(
                    self.name, 'DM1', 48, 8, perr, 0, 0, 0, False)

    def test_invalid_dx(self):
        """Invalid inputs caught."""
        perrlist = [1j, None, (3, 3)]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_dm_registration(
                    self.name, 'DM1', 48, 8, 8, perr, 0, 0, False)

    def test_invalid_dy(self):
        """Invalid inputs caught."""
        perrlist = [1j, None, (3, 3)]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_dm_registration(
                    self.name, 'DM1', 48, 8, 8, 0, perr, 0, False)

    def test_invalid_thact(self):
        """Invalid inputs caught."""
        perrlist = [1j, None, (3, 3)]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_dm_registration(
                    self.name, 'DM1', 48, 8, 8, 0, 0, perr, False)

    def test_invalid_flipx(self):
        """Invalid inputs caught."""
        perrlist = [1j, 0, 1, (3, 3), 'string']

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_dm_registration(
                    self.name, 'DM1', 48, 8, 8, 0, 0, 0, perr)


class TestAddInfluenceFunction(unittest.TestCase):
    """Tests for DM influence function data."""

    def setUp(self):
        self.fd, self.name = tempfile.mkstemp()

        self.fd1, self.name1 = tempfile.mkstemp()
        mdict = dict()
        writeyaml(mdict, self.name1)

        self.fd2, self.name2 = tempfile.mkstemp()
        mdict['dms'] = dict()
        writeyaml(mdict, self.name2)

        self.fd3, self.name3 = tempfile.mkstemp()
        mdict['dms']['DM1'] = dict()
        mdict['dms']['DM2'] = dict()
        writeyaml(mdict, self.name3)

        self.fd4, self.name4 = tempfile.mkstemp()
        mdict['dms']['DM1']['registration'] = dict()
        mdict['dms']['DM2']['registration'] = dict()
        writeyaml(mdict, self.name4)

        pass

    def tearDown(self):
        # mkstemp() does not clean up after itself, but it does have a name
        os.close(self.fd)
        os.unlink(self.name)
        os.close(self.fd1)
        os.unlink(self.name1)
        os.close(self.fd2)
        os.unlink(self.name2)
        os.close(self.fd3)
        os.unlink(self.name3)
        os.close(self.fd4)
        os.unlink(self.name4)
        pass


    def test_success(self):
        """Good inputs complete without error."""
        for name in (self.name,
                     self.name1,
                     self.name2,
                     self.name3,
                     self.name4):
            add_influence_function(name, 'DM1', 'inf_dm1.fits', 10)
            add_influence_function(name, 'DM2', 'inf_dm2.fits', 10)
        pass


    def test_readback(self):
        """Check that the output file has contents as expected."""
        # two elements, one to be skipped and one to be overwritten
        d0 = {'dms': {'DM1': {'foo': 0, 'registration': {'ppact_d': 5}}}}

        for name in (self.name,
                     self.name1,
                     self.name2,
                     self.name3,
                     self.name4):

            writeyaml(d0, name)
            add_influence_function(name, 'DM1', 'inf_dm1.fits', 10)
            add_influence_function(name, 'DM2', 'inf_dm2.fits', 10)
            d1 = loadyaml(name)

            # unrelated key unaffected
            self.assertTrue(d0['dms']['DM1']['foo'] ==
                            d1['dms']['DM1']['foo'])
            # overwritten key overwritten
            self.assertFalse(d0['dms']['DM1']['registration']['ppact_d'] ==
                             d1['dms']['DM1']['registration']['ppact_d'])
            # new keys present
            self.assertTrue('ppact_d' in d1['dms']['DM2']['registration'])
            self.assertTrue('inffn' in d1['dms']['DM2']['registration'])
            self.assertTrue('inffn' in d1['dms']['DM1']['registration'])

        pass


    def test_loaded_top_level_dict_not_a_dict(self):
        """Catch when something loaded isn't a dict when it should be."""
        fd, name = tempfile.mkstemp()
        mdict = [2, 3, 4]
        with open(name, 'w') as FILE:
            yaml.dump(mdict,
                      FILE,
                      default_flow_style=False,  # enforces block style
                      sort_keys=False,
                      )

        with self.assertRaises(TypeError):
            add_influence_function(name, 'DM1', 'inf_dm1.fits', 10)
        with self.assertRaises(TypeError):
            add_influence_function(name, 'DM2', 'inf_dm2.fits', 10)

        # Cleanup
        os.close(fd)
        os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_DM1(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'dms': list()},
            {'dms': {'DM1': list()}},
            {'dms': {'DM1': {'registration': list()}}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_influence_function(name, 'DM1', 'inf_dm1.fits', 10)

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_DM2(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'dms': list()},
            {'dms': {'DM2': list()}},
            {'dms': {'DM2': {'registration': list()}}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_influence_function(name, 'DM2', 'inf_dm2.fits', 10)

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_invalid_fn(self):
        """Invalid inputs caught."""
        perrlist = [0, 0.5, 1j, None, (3, 3)]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_influence_function(perr, 'DM1', 'inf_dm1.fits', 10)
            pass
        pass


    def test_invalid_dm_id(self):
        """Invalid inputs caught."""
        perrlist = [0, 0.5, 1j, None, (3, 3), 'txt',
                    'DM0', 'DM3',
                    'dm1', 'dm2']

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_influence_function(self.name, perr, 'inf_dm1.fits', 10)
            pass
        pass


    def test_invalid_inffn(self):
        """Invalid inputs caught."""
        perrlist = [0, 0.5, 1j, None, (3, 3)]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_influence_function(self.name, 'DM1', perr, 10)
            pass
        pass


    def test_invalid_ppact_d(self):
        """Invalid inputs caught."""
        perrlist = [0, -0.5, 1j, None, (3, 3)]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_influence_function(self.name, 'DM1', 'inf_dm1.fits', perr)
            pass
        pass



class TestAddPitch(unittest.TestCase):
    """Tests for DM pitch."""

    def setUp(self):
        self.fd, self.name = tempfile.mkstemp()

        self.fd1, self.name1 = tempfile.mkstemp()
        mdict = dict()
        writeyaml(mdict, self.name1)

        self.fd2, self.name2 = tempfile.mkstemp()
        mdict['dms'] = dict()
        writeyaml(mdict, self.name2)

        self.fd3, self.name3 = tempfile.mkstemp()
        mdict['dms']['DM1'] = dict()
        mdict['dms']['DM2'] = dict()
        writeyaml(mdict, self.name3)

        pass

    def tearDown(self):
        # mkstemp() does not clean up after itself, but it does have a name
        os.close(self.fd)
        os.unlink(self.name)
        os.close(self.fd1)
        os.unlink(self.name1)
        os.close(self.fd2)
        os.unlink(self.name2)
        os.close(self.fd3)
        os.unlink(self.name3)
        pass


    def test_success(self):
        """Good inputs complete without error."""
        for name in (self.name,
                     self.name1,
                     self.name2,
                     self.name3):
            add_pitch(name, 'DM1', 0.001)
            add_pitch(name, 'DM2', 0.001)
        pass


    def test_readback(self):
        """Check that the output file has contents as expected."""
        # two elements, one to be skipped and one to be overwritten
        d0 = {'dms': {'DM1': {'foo': 0, 'pitch': 5}}}

        for name in (self.name,
                     self.name1,
                     self.name2,
                     self.name3):

            writeyaml(d0, name)
            add_pitch(name, 'DM1', 0.001)
            add_pitch(name, 'DM2', 0.001)
            d1 = loadyaml(name)

            # unrelated key unaffected
            self.assertTrue(d0['dms']['DM1']['foo'] ==
                            d1['dms']['DM1']['foo'])
            # overwritten key overwritten
            self.assertFalse(d0['dms']['DM1']['pitch'] ==
                             d1['dms']['DM1']['pitch'])
            # new key present
            self.assertTrue('pitch' in d1['dms']['DM2'])

        pass


    def test_loaded_top_level_dict_not_a_dict(self):
        """Catch when something loaded isn't a dict when it should be."""
        fd, name = tempfile.mkstemp()
        mdict = [2, 3, 4]
        with open(name, 'w') as FILE:
            yaml.dump(mdict,
                      FILE,
                      default_flow_style=False,  # enforces block style
                      sort_keys=False,
                      )

        with self.assertRaises(TypeError):
            add_pitch(name, 'DM1', 0.001)
        with self.assertRaises(TypeError):
            add_pitch(name, 'DM2', 0.001)

        # Cleanup
        os.close(fd)
        os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_DM1(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'dms': list()},
            {'dms': {'DM1': list()}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_pitch(name, 'DM1', 0.001)

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_DM2(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'dms': list()},
            {'dms': {'DM2': list()}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_pitch(name, 'DM2', 0.001)

            # Cleanup
            os.close(fd)
            os.unlink(name)


    def test_invalid_fn(self):
        """Invalid inputs caught."""
        perrlist = [0, 0.5, 1j, None, (3, 3)]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_pitch(perr, 'DM1', 0.001)
            pass
        pass


    def test_invalid_dm_id(self):
        """Invalid inputs caught."""
        perrlist = [0, 0.5, 1j, None, (3, 3), 'txt',
                    'DM0', 'DM3',
                    'dm1', 'dm2']

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_pitch(self.name, perr, 0.001)
            pass
        pass


    def test_invalid_pitch(self):
        """Invalid inputs caught."""
        perrlist = [0, -0.5, 1j, None, (3, 3)]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_pitch(self.name, 'DM1', perr)
            pass
        pass


class TestAddZ(unittest.TestCase):
    """Tests for DM propagation distance."""

    def setUp(self):
        self.fd, self.name = tempfile.mkstemp()


        self.fd1, self.name1 = tempfile.mkstemp()
        mdict = dict()
        writeyaml(mdict, self.name1)

        self.fd2, self.name2 = tempfile.mkstemp()
        mdict['dms'] = dict()
        writeyaml(mdict, self.name2)

        self.fd3, self.name3 = tempfile.mkstemp()
        mdict['dms']['DM1'] = dict()
        mdict['dms']['DM2'] = dict()
        writeyaml(mdict, self.name3)

        pass

    def tearDown(self):
        # mkstemp() does not clean up after itself, but it does have a name
        os.close(self.fd)
        os.unlink(self.name)
        os.close(self.fd1)
        os.unlink(self.name1)
        os.close(self.fd2)
        os.unlink(self.name2)
        os.close(self.fd3)
        os.unlink(self.name3)
        pass


    def test_success(self):
        """Good inputs complete without error."""
        for name in (self.name,
                     self.name1,
                     self.name2,
                     self.name3):
            add_z(name, 'DM1', 0.0)
            add_z(name, 'DM2', 1.0)
        pass


    def test_readback(self):
        """Check that the output file has contents as expected."""
        # two elements, one to be skipped and one to be overwritten
        d0 = {'dms': {'DM1': {'foo': 0, 'z': 5}}}

        for name in (self.name,
                     self.name1,
                     self.name2,
                     self.name3):

            writeyaml(d0, name)
            add_z(name, 'DM1', 0.0)
            add_z(name, 'DM2', 1.0)
            d1 = loadyaml(name)

            # unrelated key unaffected
            self.assertTrue(d0['dms']['DM1']['foo'] ==
                            d1['dms']['DM1']['foo'])
            # overwritten key overwritten
            self.assertFalse(d0['dms']['DM1']['z'] ==
                             d1['dms']['DM1']['z'])
            # new key present
            self.assertTrue('z' in d1['dms']['DM2'])

        pass


    def test_loaded_top_level_dict_not_a_dict(self):
        """Catch when something loaded isn't a dict when it should be."""
        fd, name = tempfile.mkstemp()
        mdict = [2, 3, 4]
        with open(name, 'w') as FILE:
            yaml.dump(mdict,
                      FILE,
                      default_flow_style=False,  # enforces block style
                      sort_keys=False,
                      )

        with self.assertRaises(TypeError):
            add_z(name, 'DM1', 0.0)
        with self.assertRaises(TypeError):
            add_z(name, 'DM2', 1.0)

        # Cleanup
        os.close(fd)
        os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_DM1(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'dms': list()},
            {'dms': {'DM1': list()}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_z(name, 'DM1', 0.0)

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_DM2(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'dms': list()},
            {'dms': {'DM2': list()}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_z(name, 'DM2', 1.0)

            # Cleanup
            os.close(fd)
            os.unlink(name)


    def test_invalid_fn(self):
        """Invalid inputs caught."""
        perrlist = [0, 0.5, 1j, None, (3, 3)]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_z(perr, 'DM1', 0.001)
            pass
        pass


    def test_invalid_dm_id(self):
        """Invalid inputs caught."""
        perrlist = [0, 0.5, 1j, None, (3, 3), 'txt',
                    'DM0', 'DM3',
                    'dm1', 'dm2']

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_z(self.name, perr, 0.001)
            pass
        pass


    def test_invalid_z(self):
        """Invalid inputs caught."""
        perrlist = [1j, None, (3, 3)]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                add_z(self.name, 'DM1', perr)
            pass
        pass


if __name__ == '__main__':
    unittest.main()
