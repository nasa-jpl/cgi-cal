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

from .build_sls import (
    add_dminit,
    add_dh,
    add_ft_dir,
    add_lam,
    add_epup,
    add_fpm,
    add_fpm_real_imag,
    add_fs,
    add_lyot_mask,
    add_lyot_tip_tilt,
    add_sp,
)

invalid_real_scalar_list = [1j, (3, 3), 'txt']
invalid_real_positive_scalar_list = [-0.5, 0, 1j, (3, 3), 'txt']
invalid_fn_list = [0, 0.5, 1j, None, (3, 3)]
invalid_ft_dir_list = [0, 0.5, 1j, None, (3, 3), 'txt', 'REV']
invalid_lam_index_list = [-1, 0.5, 1j, None, (3, 3), 'txt']
invalid_bool_list = [0, 0.5, 1, 1j, None, (3, 3), 'txt']


class TestAddDmInit(unittest.TestCase):
    """Tests for adding the the initial DM command map filename, dminit."""

    def setUp(self):
        self.fd, self.name = tempfile.mkstemp()

        self.fd1, self.name1 = tempfile.mkstemp()
        mdict = dict()
        writeyaml(mdict, self.name1)

        self.fd2, self.name2 = tempfile.mkstemp()
        mdict['init'] = dict()
        writeyaml(mdict, self.name2)

        self.fd3, self.name3 = tempfile.mkstemp()
        mdict['init']['DM1'] = dict()
        mdict['init']['DM2'] = dict()
        writeyaml(mdict, self.name3)

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

    def test_success(self):
        """Good inputs complete without error."""
        for name in (self.name,
                     self.name1,
                     self.name2,
                     self.name3):
            add_dminit(name, 'DM1', 'dmabs.fits')
            add_dminit(name, 'DM2', 'dmabs.fits')

    def test_readback(self):
        """Check that the output file has contents as expected."""
        # two elements, one to be skipped and one to be overwritten
        d0 = {'init': {'DM1': {'foo': 0.1, 'dminit': 'string'}}}

        for name in (self.name,
                     self.name1,
                     self.name2,
                     self.name3):

            writeyaml(d0, name)
            add_dminit(name, 'DM1', 'dmabs.fits')
            add_dminit(name, 'DM2', 'dmabs.fits')
            d1 = loadyaml(name)

            # unrelated key unaffected
            self.assertTrue(d0['init']['DM1']['foo'] ==
                            d1['init']['DM1']['foo'])
            # overwritten key overwritten
            self.assertFalse(d0['init']['DM1']['dminit'] ==
                             d1['init']['DM1']['dminit'])
            # new key present
            self.assertTrue('dminit' in d1['init']['DM2'])

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
            add_dminit(name, 'DM1', 'dmabs.fits')
        with self.assertRaises(TypeError):
            add_dminit(name, 'DM2', 'dmabs.fits')

        # Cleanup
        os.close(fd)
        os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_0(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'init': list()},
            {'init': {'DM1': list()}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_dminit(name, 'DM1', 'dmabs.fits')

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_1(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'init': list()},
            {'init': {'DM2': list()}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_dminit(name, 'DM2', 'darkhole1.fits')

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_invalid_fn(self):
        """Invalid inputs caught."""
        for perr in invalid_fn_list:
            with self.assertRaises(TypeError):
                add_dminit(perr, 'DM1', 'darkhole.fits')

    def test_invalid_lam_index(self):
        """Invalid inputs caught."""
        for perr in invalid_lam_index_list:
            with self.assertRaises(TypeError):
                add_dminit(self.name, perr, 'darkhole.fits')

    def test_invalid_dh(self):
        """Invalid inputs caught."""
        for perr in invalid_fn_list:
            with self.assertRaises(TypeError):
                add_dminit(self.name, 'DM1', perr)


class TestAddDH(unittest.TestCase):
    """Tests for adding the the dark hole filename, dh."""

    def setUp(self):
        self.fd, self.name = tempfile.mkstemp()

        self.fd1, self.name1 = tempfile.mkstemp()
        mdict = dict()
        writeyaml(mdict, self.name1)

        self.fd2, self.name2 = tempfile.mkstemp()
        mdict['sls'] = dict()
        writeyaml(mdict, self.name2)

        self.fd3, self.name3 = tempfile.mkstemp()
        mdict['sls'][0] = dict()
        mdict['sls'][1] = dict()
        writeyaml(mdict, self.name3)

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

    def test_success(self):
        """Good inputs complete without error."""
        for name in (self.name,
                     self.name1,
                     self.name2,
                     self.name3):
            add_dh(name, 0, 'darkhole0.fits')
            add_dh(name, 1, 'darkhole1.fits')

    def test_readback(self):
        """Check that the output file has contents as expected."""
        # two elements, one to be skipped and one to be overwritten
        d0 = {'sls': {0: {'foo': 0.1, 'dh': 'string'}}}

        for name in (self.name,
                     self.name1,
                     self.name2,
                     self.name3):

            writeyaml(d0, name)
            add_dh(name, 0, 'darkhole0.fits')
            add_dh(name, 1, 'darkhole1.fits')
            d1 = loadyaml(name)

            # unrelated key unaffected
            self.assertTrue(d0['sls'][0]['foo'] ==
                            d1['sls'][0]['foo'])
            # overwritten key overwritten
            self.assertFalse(d0['sls'][0]['dh'] ==
                             d1['sls'][0]['dh'])
            # new key present
            self.assertTrue('dh' in d1['sls'][1])

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
            add_dh(name, 0, 'darkhole0.fits')
        with self.assertRaises(TypeError):
            add_dh(name, 1, 'darkhole1.fits')

        # Cleanup
        os.close(fd)
        os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_0(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'sls': list()},
            {'sls': {0: list()}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_dh(name, 0, 'darkhole0.fits')

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_1(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'sls': list()},
            {'sls': {1: list()}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_dh(name, 1, 'darkhole1.fits')

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_invalid_fn(self):
        """Invalid inputs caught."""
        for perr in invalid_fn_list:
            with self.assertRaises(TypeError):
                add_dh(perr, 0, 'darkhole.fits')

    def test_invalid_lam_index(self):
        """Invalid inputs caught."""
        for perr in invalid_lam_index_list:
            with self.assertRaises(TypeError):
                add_dh(self.name, perr, 'darkhole.fits')

    def test_invalid_dh(self):
        """Invalid inputs caught."""
        for perr in invalid_fn_list:
            with self.assertRaises(TypeError):
                add_dh(self.name, 0, perr)


class TestAddFtDir(unittest.TestCase):
    """Tests for adding the the Fourier transform direction, ft_dir."""

    def setUp(self):
        self.fd, self.name = tempfile.mkstemp()

        self.fd1, self.name1 = tempfile.mkstemp()
        mdict = dict()
        writeyaml(mdict, self.name1)

        self.fd2, self.name2 = tempfile.mkstemp()
        mdict['sls'] = dict()
        writeyaml(mdict, self.name2)

        self.fd3, self.name3 = tempfile.mkstemp()
        mdict['sls'][0] = dict()
        mdict['sls'][1] = dict()
        writeyaml(mdict, self.name3)

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

    def test_success(self):
        """Good inputs complete without error."""
        for name in (self.name,
                     self.name1,
                     self.name2,
                     self.name3):
            add_ft_dir(name, 0, 'reverse')
            add_ft_dir(name, 1, 'reverse')

    def test_readback(self):
        """Check that the output file has contents as expected."""
        # two elements, one to be skipped and one to be overwritten
        d0 = {'sls': {0: {'foo': 0.1, 'ft_dir': 'forward'}}}

        for name in (self.name,
                     self.name1,
                     self.name2,
                     self.name3):

            writeyaml(d0, name)
            add_ft_dir(name, 0, 'reverse')
            add_ft_dir(name, 1, 'reverse')
            d1 = loadyaml(name)

            # unrelated key unaffected
            self.assertTrue(d0['sls'][0]['foo'] ==
                            d1['sls'][0]['foo'])
            # overwritten key overwritten
            self.assertFalse(d0['sls'][0]['ft_dir'] ==
                             d1['sls'][0]['ft_dir'])
            # new key present
            self.assertTrue('ft_dir' in d1['sls'][1])

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
            add_ft_dir(name, 0, 'reverse')
        with self.assertRaises(TypeError):
            add_ft_dir(name, 1, 'reverse')

        # Cleanup
        os.close(fd)
        os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_0(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'sls': list()},
            {'sls': {0: list()}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_ft_dir(name, 0, 'reverse')

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_1(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'sls': list()},
            {'sls': {1: list()}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_ft_dir(name, 1, 'reverse')

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_invalid_fn(self):
        """Invalid inputs caught."""
        for perr in invalid_fn_list:
            with self.assertRaises(TypeError):
                add_ft_dir(perr, 0, 'reverse')

    def test_invalid_lam_index(self):
        """Invalid inputs caught."""
        for perr in invalid_lam_index_list:
            with self.assertRaises(TypeError):
                add_ft_dir(self.name, perr, 'reverse')

    def test_invalid_dh(self):
        """Invalid inputs caught."""
        for perr in invalid_ft_dir_list:
            with self.assertRaises(TypeError):
                add_ft_dir(self.name, 0, perr)


class TestAddLam(unittest.TestCase):
    """Tests for adding the wavelength, lam."""

    def setUp(self):
        self.fd, self.name = tempfile.mkstemp()

        self.fd1, self.name1 = tempfile.mkstemp()
        mdict = dict()
        writeyaml(mdict, self.name1)

        self.fd2, self.name2 = tempfile.mkstemp()
        mdict['sls'] = dict()
        writeyaml(mdict, self.name2)

        self.fd3, self.name3 = tempfile.mkstemp()
        mdict['sls'][0] = dict()
        mdict['sls'][1] = dict()
        writeyaml(mdict, self.name3)

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

    def test_success(self):
        """Good inputs complete without error."""
        for name in (self.name,
                     self.name1,
                     self.name2,
                     self.name3):
            add_lam(name, 0, 575e-9)
            add_lam(name, 1, 600.0e-9)

    def test_readback(self):
        """Check that the output file has contents as expected."""
        # two elements, one to be skipped and one to be overwritten
        d0 = {'sls': {0: {'foo': 0.1, 'lam': 400e-9}}}

        for name in (self.name,
                     self.name1,
                     self.name2,
                     self.name3):

            writeyaml(d0, name)
            add_lam(name, 0, 575e-9)
            add_lam(name, 1, 600e-9)
            d1 = loadyaml(name)

            # unrelated key unaffected
            self.assertTrue(d0['sls'][0]['foo'] ==
                            d1['sls'][0]['foo'])
            # overwritten key overwritten
            self.assertFalse(d0['sls'][0]['lam'] ==
                             d1['sls'][0]['lam'])
            # new key present
            self.assertTrue('lam' in d1['sls'][1])

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
            add_lam(name, 0, 0.001)
        with self.assertRaises(TypeError):
            add_lam(name, 1, 0.001)

        # Cleanup
        os.close(fd)
        os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_0(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'sls': list()},
            {'sls': {0: list()}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_lam(name, 0, 0.001)

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_1(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'sls': list()},
            {'sls': {1: list()}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_lam(name, 1, 0.001)

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_invalid_fn(self):
        """Invalid inputs caught."""
        for perr in invalid_fn_list:
            with self.assertRaises(TypeError):
                add_lam(perr, 0, 0.001)

    def test_invalid_lam_index(self):
        """Invalid inputs caught."""
        for perr in invalid_lam_index_list:
            with self.assertRaises(TypeError):
                add_lam(self.name, perr, 0.001)

    def test_invalid_lam(self):
        """Invalid inputs caught."""
        for perr in invalid_real_positive_scalar_list:
            with self.assertRaises(TypeError):
                add_lam(self.name, 0, perr)


class TestAddEpup(unittest.TestCase):
    """Tests for adding the epup dictionary."""

    def setUp(self):
        self.fd, self.name = tempfile.mkstemp()

        self.fd1, self.name1 = tempfile.mkstemp()
        mdict = dict()
        writeyaml(mdict, self.name1)

        self.fd2, self.name2 = tempfile.mkstemp()
        mdict['sls'] = dict()
        writeyaml(mdict, self.name2)

        self.fd3, self.name3 = tempfile.mkstemp()
        mdict['sls'][0] = dict()
        mdict['sls'][1] = dict()
        writeyaml(mdict, self.name3)

        self.fd4, self.name4 = tempfile.mkstemp()
        mdict['sls'][0]['epup'] = dict()
        mdict['sls'][1]['epup'] = dict()
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
            add_epup(name, 0, 'amp0.fits', 300.1, 'ph0.fits', 0.2, 0.3)
            add_epup(name, 1, 'amp1.fits', 300.1, 'ph1.fits', 0.4, 0.5)

    def test_readback(self):
        """Check that the output file has contents as expected."""
        # two elements, one to be skipped and one to be overwritten
        d0 = {'sls': {0: {'foo': 0.1, 'epup': {'pdp': 123}}}}

        for name in (self.name,
                      self.name1,
                      self.name2,
                      self.name3,
                      self.name4):
            writeyaml(d0, name)
            add_epup(name, 0, 'amp0.fits', 300.1, 'ph0.fits', 0.2, 0.3)
            add_epup(name, 1, 'amp1.fits', 300.1, 'ph1.fits', 0.4, 0.5)
            d1 = loadyaml(name)

            # unrelated key unaffected
            self.assertTrue(d0['sls'][0]['foo'] ==
                            d1['sls'][0]['foo'])
            # overwritten key overwritten
            self.assertFalse(d0['sls'][0]['epup']['pdp'] ==
                             d1['sls'][0]['epup']['pdp'])
            # new keys present
            self.assertTrue('pdp' in d1['sls'][1]['epup'])
            self.assertTrue('afn' in d1['sls'][1]['epup'])
            self.assertTrue('pfn' in d1['sls'][1]['epup'])
            self.assertTrue('tip' in d1['sls'][1]['epup'])
            self.assertTrue('tilt' in d1['sls'][1]['epup'])

            self.assertTrue('afn' in d1['sls'][0]['epup'])
            self.assertTrue('pfn' in d1['sls'][0]['epup'])
            self.assertTrue('tip' in d1['sls'][0]['epup'])
            self.assertTrue('tilt' in d1['sls'][0]['epup'])

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
            add_epup(name, 0, 'amp0.fits', 300.1, 'ph0.fits', 0.2, 0.3)
        with self.assertRaises(TypeError):
            add_epup(name, 1, 'amp1.fits', 300.1, 'ph1.fits', 0.4, 0.5)

        # Cleanup
        os.close(fd)
        os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_sls_0(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'sls': list()},
            {'sls': {0: list()}},
            {'sls': {0: {'epup': list()}}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_epup(name, 0, 'amp0.fits', 300.1, 'ph0.fits', 0.2, 0.3)

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_sls_1(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'sls': list()},
            {'sls': {1: list()}},
            {'sls': {1: {'epup': list()}}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_epup(name, 1, 'amp0.fits', 300.1, 'ph0.fits', 0.2, 0.3)

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_invalid_fn_model(self):
        """Invalid inputs caught."""
        for perr in invalid_fn_list:
            with self.assertRaises(TypeError):
                add_epup(perr, 0, 'amp0.fits', 300.1, 'ph0.fits', 0.2, 0.3)

    def test_invalid_lam_index(self):
        """Invalid inputs caught."""
        for perr in invalid_lam_index_list:
            with self.assertRaises(TypeError):
                add_epup(
                    self.name, perr, 'amp0.fits', 300.1, 'ph0.fits', 0.2, 0.3)

    def test_invalid_afn(self):
        """Invalid inputs caught."""
        for perr in invalid_fn_list:
            with self.assertRaises(TypeError):
                add_epup(self.name, 0, perr, 300.1, 'ph0.fits', 0.2, 0.3)

    def test_invalid_pdp(self):
        """Invalid inputs caught."""
        for perr in invalid_real_scalar_list:
            with self.assertRaises(TypeError):
                add_epup(self.name, 0, 'amp0.fits', perr, 'ph0.fits', 0.2, 0.3)

    def test_invalid_pfn(self):
        """Invalid inputs caught."""
        for perr in invalid_fn_list:
            with self.assertRaises(TypeError):
                add_epup(self.name, 0, 'amp0.fits', 300.1, perr, 0.2, 0.3)

    def test_invalid_tip(self):
        """Invalid inputs caught."""
        for perr in invalid_real_scalar_list:
            with self.assertRaises(TypeError):
                add_epup(
                    self.name, 0, 'amp0.fits', 300.1, 'ph0.fits', perr, 0.3)

    def test_invalid_tilt(self):
        """Invalid inputs caught."""
        for perr in invalid_real_scalar_list:
            with self.assertRaises(TypeError):
                add_epup(
                    self.name, 0, 'amp0.fits', 300.1, 'ph0.fits', 0.2, perr)


class TestAddLyotMask(unittest.TestCase):
    """Tests for adding the lyot dictionary's mask related keys."""

    def setUp(self):
        self.fd, self.name = tempfile.mkstemp()

        self.fd1, self.name1 = tempfile.mkstemp()
        mdict = dict()
        writeyaml(mdict, self.name1)

        self.fd2, self.name2 = tempfile.mkstemp()
        mdict['sls'] = dict()
        writeyaml(mdict, self.name2)

        self.fd3, self.name3 = tempfile.mkstemp()
        mdict['sls'][0] = dict()
        mdict['sls'][1] = dict()
        writeyaml(mdict, self.name3)

        self.fd4, self.name4 = tempfile.mkstemp()
        mdict['sls'][0]['lyot'] = dict()
        mdict['sls'][1]['lyot'] = dict()
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
            add_lyot_mask(name, 0, 'amp0.fits', 300.1, 'ph0.fits')
            add_lyot_mask(name, 1, 'amp1.fits', 300.1, 'ph1.fits')

    def test_readback(self):
        """Check that the output file has contents as expected."""
        # two elements, one to be skipped and one to be overwritten
        d0 = {'sls': {0: {'foo': 0.1, 'lyot': {'pdp': 123}}}}

        for name in (self.name,
                      self.name1,
                      self.name2,
                      self.name3,
                      self.name4):
            writeyaml(d0, name)
            add_lyot_mask(name, 0, 'amp0.fits', 300.1, 'ph0.fits')
            add_lyot_mask(name, 1, 'amp1.fits', 300.1, 'ph1.fits')
            d1 = loadyaml(name)

            # unrelated key unaffected
            self.assertTrue(d0['sls'][0]['foo'] ==
                            d1['sls'][0]['foo'])
            # overwritten key overwritten
            self.assertFalse(d0['sls'][0]['lyot']['pdp'] ==
                             d1['sls'][0]['lyot']['pdp'])
            # new keys present
            self.assertTrue('pdp' in d1['sls'][1]['lyot'])
            self.assertTrue('afn' in d1['sls'][1]['lyot'])
            self.assertTrue('pfn' in d1['sls'][1]['lyot'])

            self.assertTrue('afn' in d1['sls'][0]['lyot'])
            self.assertTrue('pfn' in d1['sls'][0]['lyot'])

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
            add_lyot_mask(name, 0, 'amp0.fits', 300.1, 'ph0.fits')
        with self.assertRaises(TypeError):
            add_lyot_mask(name, 1, 'amp1.fits', 300.1, 'ph1.fits')

        # Cleanup
        os.close(fd)
        os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_sls_0(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'sls': list()},
            {'sls': {0: list()}},
            {'sls': {0: {'lyot': list()}}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_lyot_mask(name, 0, 'amp0.fits', 300.1, 'ph0.fits')

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_sls_1(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'sls': list()},
            {'sls': {1: list()}},
            {'sls': {1: {'lyot': list()}}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_lyot_mask(name, 1, 'amp0.fits', 300.1, 'ph0.fits')

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_invalid_fn_model(self):
        """Invalid inputs caught."""
        for perr in invalid_fn_list:
            with self.assertRaises(TypeError):
                add_lyot_mask(perr, 0, 'amp0.fits', 300.1, 'ph0.fits')

    def test_invalid_lam_index(self):
        """Invalid inputs caught."""
        for perr in invalid_lam_index_list:
            with self.assertRaises(TypeError):
                add_lyot_mask(self.name, perr, 'amp0.fits', 300.1, 'ph0.fits')

    def test_invalid_afn(self):
        """Invalid inputs caught."""
        for perr in invalid_fn_list:
            with self.assertRaises(TypeError):
                add_lyot_mask(self.name, 0, perr, 300.1, 'ph0.fits')

    def test_invalid_pdp(self):
        """Invalid inputs caught."""
        for perr in invalid_real_scalar_list:
            with self.assertRaises(TypeError):
                add_lyot_mask(self.name, 0, 'amp0.fits', perr, 'ph0.fits')

    def test_invalid_pfn(self):
        """Invalid inputs caught."""
        for perr in invalid_fn_list:
            with self.assertRaises(TypeError):
                add_lyot_mask(self.name, 0, 'amp0.fits', 300.1, perr)


class TestAddLyotTipTilt(unittest.TestCase):
    """Tests for adding the lyot dictionary's tip and tilt."""

    def setUp(self):
        self.fd, self.name = tempfile.mkstemp()

        self.fd1, self.name1 = tempfile.mkstemp()
        mdict = dict()
        writeyaml(mdict, self.name1)

        self.fd2, self.name2 = tempfile.mkstemp()
        mdict['sls'] = dict()
        writeyaml(mdict, self.name2)

        self.fd3, self.name3 = tempfile.mkstemp()
        mdict['sls'][0] = dict()
        mdict['sls'][1] = dict()
        writeyaml(mdict, self.name3)

        self.fd4, self.name4 = tempfile.mkstemp()
        mdict['sls'][0]['lyot'] = dict()
        mdict['sls'][1]['lyot'] = dict()
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
            add_lyot_tip_tilt(name, 0, 0.2, 0.3)
            add_lyot_tip_tilt(name, 1, 0.4, 0.5)

    def test_readback(self):
        """Check that the output file has contents as expected."""
        # two elements, one to be skipped and one to be overwritten
        d0 = {'sls': {0: {'foo': 0.1, 'lyot': {'tip': 123}}}}

        for name in (self.name,
                      self.name1,
                      self.name2,
                      self.name3,
                      self.name4):
            writeyaml(d0, name)
            add_lyot_tip_tilt(name, 0, 0.2, 0.3)
            add_lyot_tip_tilt(name, 1, 0.4, 0.5)
            d1 = loadyaml(name)

            # unrelated key unaffected
            self.assertTrue(d0['sls'][0]['foo'] ==
                            d1['sls'][0]['foo'])
            # overwritten key overwritten
            self.assertFalse(d0['sls'][0]['lyot']['tip'] ==
                             d1['sls'][0]['lyot']['tip'])
            # new keys present
            self.assertTrue('tip' in d1['sls'][1]['lyot'])
            self.assertTrue('tilt' in d1['sls'][1]['lyot'])

            self.assertTrue('tilt' in d1['sls'][0]['lyot'])

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
            add_lyot_tip_tilt(name, 0, 0.2, 0.3)
        with self.assertRaises(TypeError):
            add_lyot_tip_tilt(name, 1, 0.4, 0.5)

        # Cleanup
        os.close(fd)
        os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_sls_0(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'sls': list()},
            {'sls': {0: list()}},
            {'sls': {0: {'lyot': list()}}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_lyot_tip_tilt(name, 0, 0.2, 0.3)

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_sls_1(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'sls': list()},
            {'sls': {1: list()}},
            {'sls': {1: {'lyot': list()}}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_lyot_tip_tilt(name, 1, 0.2, 0.3)

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_invalid_fn_model(self):
        """Invalid inputs caught."""
        for perr in invalid_fn_list:
            with self.assertRaises(TypeError):
                add_lyot_tip_tilt(perr, 0, 0.2, 0.3)

    def test_invalid_lam_index(self):
        """Invalid inputs caught."""
        for perr in invalid_lam_index_list:
            with self.assertRaises(TypeError):
                add_lyot_tip_tilt(self.name, perr, 0.2, 0.3)

    def test_invalid_tip(self):
        """Invalid inputs caught."""
        for perr in invalid_real_scalar_list:
            with self.assertRaises(TypeError):
                add_lyot_tip_tilt(self.name, 0, perr, 0.3)

    def test_invalid_tilt(self):
        """Invalid inputs caught."""
        for perr in invalid_real_scalar_list:
            with self.assertRaises(TypeError):
                add_lyot_tip_tilt(self.name, 0, 0.2, perr)


class TestAddSp(unittest.TestCase):
    """Tests for adding the sp dictionary."""

    def setUp(self):
        self.fd, self.name = tempfile.mkstemp()

        self.fd1, self.name1 = tempfile.mkstemp()
        mdict = dict()
        writeyaml(mdict, self.name1)

        self.fd2, self.name2 = tempfile.mkstemp()
        mdict['sls'] = dict()
        writeyaml(mdict, self.name2)

        self.fd3, self.name3 = tempfile.mkstemp()
        mdict['sls'][0] = dict()
        mdict['sls'][1] = dict()
        writeyaml(mdict, self.name3)

        self.fd4, self.name4 = tempfile.mkstemp()
        mdict['sls'][0]['sp'] = dict()
        mdict['sls'][1]['sp'] = dict()
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
            add_sp(name, 0, 'amp0.fits', 300.1, 'ph0.fits')
            add_sp(name, 1, 'amp1.fits', 300.1, 'ph1.fits')

    def test_readback(self):
        """Check that the output file has contents as expected."""
        # two elements, one to be skipped and one to be overwritten
        d0 = {'sls': {0: {'foo': 0.1, 'sp': {'pdp': 123}}}}

        for name in (self.name,
                      self.name1,
                      self.name2,
                      self.name3,
                      self.name4):
            writeyaml(d0, name)
            add_sp(name, 0, 'amp0.fits', 300.1, 'ph0.fits')
            add_sp(name, 1, 'amp1.fits', 300.1, 'ph1.fits')
            d1 = loadyaml(name)

            # unrelated key unaffected
            self.assertTrue(d0['sls'][0]['foo'] ==
                            d1['sls'][0]['foo'])
            # overwritten key overwritten
            self.assertFalse(d0['sls'][0]['sp']['pdp'] ==
                             d1['sls'][0]['sp']['pdp'])
            # new keys present
            self.assertTrue('pdp' in d1['sls'][1]['sp'])
            self.assertTrue('afn' in d1['sls'][1]['sp'])
            self.assertTrue('pfn' in d1['sls'][1]['sp'])

            self.assertTrue('afn' in d1['sls'][0]['sp'])
            self.assertTrue('pfn' in d1['sls'][0]['sp'])

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
            add_sp(name, 0, 'amp0.fits', 300.1, 'ph0.fits')
        with self.assertRaises(TypeError):
            add_sp(name, 1, 'amp1.fits', 300.1, 'ph1.fits')

        # Cleanup
        os.close(fd)
        os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_sls_0(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'sls': list()},
            {'sls': {0: list()}},
            {'sls': {0: {'sp': list()}}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_sp(name, 0, 'amp0.fits', 300.1, 'ph0.fits')

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_sls_1(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'sls': list()},
            {'sls': {1: list()}},
            {'sls': {1: {'sp': list()}}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_sp(name, 1, 'amp0.fits', 300.1, 'ph0.fits')

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_invalid_fn_model(self):
        """Invalid inputs caught."""
        for perr in invalid_fn_list:
            with self.assertRaises(TypeError):
                add_sp(perr, 0, 'amp0.fits', 300.1, 'ph0.fits')

    def test_invalid_lam_index(self):
        """Invalid inputs caught."""
        for perr in invalid_lam_index_list:
            with self.assertRaises(TypeError):
                add_sp(self.name, perr, 'amp0.fits', 300.1, 'ph0.fits')

    def test_invalid_afn(self):
        """Invalid inputs caught."""
        for perr in invalid_fn_list:
            with self.assertRaises(TypeError):
                add_sp(self.name, 0, perr, 300.1, 'ph0.fits')

    def test_invalid_pdp(self):
        """Invalid inputs caught."""
        for perr in invalid_real_scalar_list:
            with self.assertRaises(TypeError):
                add_sp(self.name, 0, 'amp0.fits', perr, 'ph0.fits')

    def test_invalid_pfn(self):
        """Invalid inputs caught."""
        for perr in invalid_fn_list:
            with self.assertRaises(TypeError):
                add_sp(self.name, 0, 'amp0.fits', 300.1, perr)


class TestAddFs(unittest.TestCase):
    """Tests for adding the fs dictionary."""

    def setUp(self):
        self.fd, self.name = tempfile.mkstemp()

        self.fd1, self.name1 = tempfile.mkstemp()
        mdict = dict()
        writeyaml(mdict, self.name1)

        self.fd2, self.name2 = tempfile.mkstemp()
        mdict['sls'] = dict()
        writeyaml(mdict, self.name2)

        self.fd3, self.name3 = tempfile.mkstemp()
        mdict['sls'][0] = dict()
        mdict['sls'][1] = dict()
        writeyaml(mdict, self.name3)

        self.fd4, self.name4 = tempfile.mkstemp()
        mdict['sls'][0]['fs'] = dict()
        mdict['sls'][1]['fs'] = dict()
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
            add_fs(name, 0, 'amp0.fits', 'ph0.fits', 2.3)
            add_fs(name, 1, 'amp1.fits', 'ph1.fits', 2.3)

    def test_readback(self):
        """Check that the output file has contents as expected."""
        # two elements, one to be skipped and one to be overwritten
        d0 = {'sls': {0: {'foo': 0.1, 'fs': {'ppl': 10.0}}}}

        for name in (self.name,
                      self.name1,
                      self.name2,
                      self.name3,
                      self.name4):
            writeyaml(d0, name)
            add_fs(name, 0, 'amp0.fits', 'ph0.fits', 2.3)
            add_fs(name, 1, 'amp1.fits', 'ph1.fits', 3.3)
            d1 = loadyaml(name)

            # unrelated key unaffected
            self.assertTrue(d0['sls'][0]['foo'] ==
                            d1['sls'][0]['foo'])
            # overwritten key overwritten
            self.assertFalse(d0['sls'][0]['fs']['ppl'] ==
                             d1['sls'][0]['fs']['ppl'])
            # new keys present
            self.assertTrue('ppl' in d1['sls'][1]['fs'])
            self.assertTrue('afn' in d1['sls'][1]['fs'])
            self.assertTrue('pfn' in d1['sls'][1]['fs'])

            self.assertTrue('afn' in d1['sls'][0]['fs'])
            self.assertTrue('pfn' in d1['sls'][0]['fs'])

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
            add_fs(name, 0, 'amp0.fits', 'ph0.fits', 2.3)
        with self.assertRaises(TypeError):
            add_fs(name, 1, 'amp1.fits', 'ph1.fits', 2.3)

        # Cleanup
        os.close(fd)
        os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_sls_0(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'sls': list()},
            {'sls': {0: list()}},
            {'sls': {0: {'fs': list()}}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_fs(name, 0, 'amp0.fits', 'ph0.fits', 2.3)

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_sls_1(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'sls': list()},
            {'sls': {1: list()}},
            {'sls': {1: {'fs': list()}}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_fs(name, 1, 'amp0.fits', 'ph0.fits', 2.3)

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_invalid_fn_model(self):
        """Invalid inputs caught."""
        for perr in invalid_fn_list:
            with self.assertRaises(TypeError):
                add_fs(perr, 0, 'amp0.fits', 'ph0.fits', 2.3)

    def test_invalid_lam_index(self):
        """Invalid inputs caught."""
        for perr in invalid_lam_index_list:
            with self.assertRaises(TypeError):
                add_fs(self.name, perr, 'amp0.fits', 'ph0.fits', 2.3)

    def test_invalid_afn(self):
        """Invalid inputs caught."""
        for perr in invalid_fn_list:
            with self.assertRaises(TypeError):
                add_fs(self.name, 0, perr, 'ph0.fits', 2.3)

    def test_invalid_pfn(self):
        """Invalid inputs caught."""
        for perr in invalid_fn_list:
            with self.assertRaises(TypeError):
                add_fs(self.name, 0, 'amp0.fits', perr, 2.3)

    def test_invalid_ppl(self):
        """Invalid inputs caught."""
        for perr in invalid_real_positive_scalar_list:
            with self.assertRaises(TypeError):
                add_fs(self.name, 0, 'amp0.fits','ph0.fits', perr)


class TestAddFpm(unittest.TestCase):
    """Tests for adding the fpm dictionary."""

    def setUp(self):
        self.fd, self.name = tempfile.mkstemp()

        self.fd1, self.name1 = tempfile.mkstemp()
        mdict = dict()
        writeyaml(mdict, self.name1)

        self.fd2, self.name2 = tempfile.mkstemp()
        mdict['sls'] = dict()
        writeyaml(mdict, self.name2)

        self.fd3, self.name3 = tempfile.mkstemp()
        mdict['sls'][0] = dict()
        mdict['sls'][1] = dict()
        writeyaml(mdict, self.name3)

        self.fd4, self.name4 = tempfile.mkstemp()
        mdict['sls'][0]['fpm'] = dict()
        mdict['sls'][1]['fpm'] = dict()
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
            add_fpm(name, 0, 'amp.fits', True, 'ph.fits', 6.6)
            add_fpm(name, 1, 'amp.fits', True, 'ph.fits', 6.6)

    def test_readback(self):
        """Check that the output file has contents as expected."""
        # two elements, one to be skipped and one to be overwritten
        d0 = {'sls': {0: {'foo': 0.1, 'fpm': {'ppl': 10.0}}}}

        for name in (self.name,
                      self.name1,
                      self.name2,
                      self.name3,
                      self.name4):
            writeyaml(d0, name)
            add_fpm(name, 0, 'amp.fits', True, 'ph.fits', 6.6)
            add_fpm(name, 1, 'amp.fits', True, 'ph.fits', 7.0)
            d1 = loadyaml(name)

            # unrelated key unaffected
            self.assertTrue(d0['sls'][0]['foo'] ==
                            d1['sls'][0]['foo'])
            # overwritten key overwritten
            self.assertFalse(d0['sls'][0]['fpm']['ppl'] ==
                             d1['sls'][0]['fpm']['ppl'])
            # new keys present
            self.assertTrue('ppl' in d1['sls'][1]['fpm'])
            self.assertTrue('afn' in d1['sls'][1]['fpm'])
            self.assertTrue('pfn' in d1['sls'][1]['fpm'])
            self.assertTrue('isopen' in d1['sls'][1]['fpm'])

            self.assertTrue('afn' in d1['sls'][0]['fpm'])
            self.assertTrue('pfn' in d1['sls'][0]['fpm'])
            self.assertTrue('isopen' in d1['sls'][0]['fpm'])

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
            add_fpm(name, 0, 'amp.fits', True, 'ph.fits', 6.6)
        with self.assertRaises(TypeError):
            add_fpm(name, 1, 'amp.fits', True, 'ph.fits', 6.6)

        # Cleanup
        os.close(fd)
        os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_sls_0(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'sls': list()},
            {'sls': {0: list()}},
            {'sls': {0: {'fpm': list()}}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_fpm(name, 0, 'amp.fits', True, 'ph.fits', 6.6)

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_sls_1(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'sls': list()},
            {'sls': {1: list()}},
            {'sls': {1: {'fpm': list()}}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_fpm(name, 1, 'amp.fits', True, 'ph.fits', 6.6)

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_invalid_fn_model(self):
        """Invalid inputs caught."""
        for perr in invalid_fn_list:
            with self.assertRaises(TypeError):
                add_fpm(perr, 0, 'amp.fits', True, 'ph.fits', 6.6)

    def test_invalid_lam_index(self):
        """Invalid inputs caught."""
        for perr in invalid_lam_index_list:
            with self.assertRaises(TypeError):
                add_fpm(self.name, perr, 'amp.fits', True, 'ph.fits', 6.6)

    def test_invalid_rfn(self):
        """Invalid inputs caught."""
        for perr in invalid_fn_list:
            with self.assertRaises(TypeError):
                add_fpm(self.name, 0, perr, True, 'ph.fits', 6.6)

    def test_invalid_isopen(self):
        """Invalid inputs caught."""
        for perr in invalid_bool_list:
            with self.assertRaises(TypeError):
                add_fpm(self.name, 0, 'amp.fits', perr, 'ph.fits', 6.6)

    def test_invalid_ifn(self):
        """Invalid inputs caught."""
        for perr in invalid_fn_list:
            with self.assertRaises(TypeError):
                add_fpm(self.name, 0, 'amp.fits', True, perr, 6.6)

    def test_invalid_ppl(self):
        """Invalid inputs caught."""
        for perr in invalid_real_positive_scalar_list:
            with self.assertRaises(TypeError):
                add_fpm(self.name, 0, 'amp.fits', True, 'ph.fits', perr)


class TestAddFpmRealImag(unittest.TestCase):
    """Tests for adding the fpm dictionary."""

    def setUp(self):
        self.fd, self.name = tempfile.mkstemp()

        self.fd1, self.name1 = tempfile.mkstemp()
        mdict = dict()
        writeyaml(mdict, self.name1)

        self.fd2, self.name2 = tempfile.mkstemp()
        mdict['sls'] = dict()
        writeyaml(mdict, self.name2)

        self.fd3, self.name3 = tempfile.mkstemp()
        mdict['sls'][0] = dict()
        mdict['sls'][1] = dict()
        writeyaml(mdict, self.name3)

        self.fd4, self.name4 = tempfile.mkstemp()
        mdict['sls'][0]['fpm'] = dict()
        mdict['sls'][1]['fpm'] = dict()
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
            add_fpm_real_imag(name, 0, 'real.fits', True, 'imag.fits', 6.6)
            add_fpm_real_imag(name, 1, 'real.fits', True, 'imag.fits', 6.6)

    def test_readback(self):
        """Check that the output file has contents as expected."""
        # two elements, one to be skipped and one to be overwritten
        d0 = {'sls': {0: {'foo': 0.1, 'fpm': {'ppl': 10.0}}}}

        for name in (self.name,
                      self.name1,
                      self.name2,
                      self.name3,
                      self.name4):
            writeyaml(d0, name)
            add_fpm_real_imag(name, 0, 'real.fits', True, 'imag.fits', 6.6)
            add_fpm_real_imag(name, 1, 'real.fits', True, 'imag.fits', 7.0)
            d1 = loadyaml(name)

            # unrelated key unaffected
            self.assertTrue(d0['sls'][0]['foo'] ==
                            d1['sls'][0]['foo'])
            # overwritten key overwritten
            self.assertFalse(d0['sls'][0]['fpm']['ppl'] ==
                             d1['sls'][0]['fpm']['ppl'])
            # new keys present
            self.assertTrue('ppl' in d1['sls'][1]['fpm'])
            self.assertTrue('rfn' in d1['sls'][1]['fpm'])
            self.assertTrue('ifn' in d1['sls'][1]['fpm'])
            self.assertTrue('isopen' in d1['sls'][1]['fpm'])

            self.assertTrue('rfn' in d1['sls'][0]['fpm'])
            self.assertTrue('ifn' in d1['sls'][0]['fpm'])
            self.assertTrue('isopen' in d1['sls'][0]['fpm'])

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
            add_fpm_real_imag(name, 0, 'real.fits', True, 'imag.fits', 6.6)
        with self.assertRaises(TypeError):
            add_fpm_real_imag(name, 1, 'real.fits', True, 'imag.fits', 6.6)

        # Cleanup
        os.close(fd)
        os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_sls_0(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'sls': list()},
            {'sls': {0: list()}},
            {'sls': {0: {'fpm': list()}}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_fpm_real_imag(name, 0, 'real.fits', True, 'imag.fits', 6.6)

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_loaded_nested_dict_not_a_dict_sls_1(self):
        """Catch when something loaded isn't a dict when it should be."""
        badDictList = [
            {'sls': list()},
            {'sls': {1: list()}},
            {'sls': {1: {'fpm': list()}}},
        ]

        for mdict in badDictList:

            fd, name = tempfile.mkstemp()

            writeyaml(mdict, name)
            with self.assertRaises(TypeError):
                add_fpm_real_imag(name, 1, 'real.fits', True, 'imag.fits', 6.6)

            # Cleanup
            os.close(fd)
            os.unlink(name)

    def test_invalid_fn_model(self):
        """Invalid inputs caught."""
        for perr in invalid_fn_list:
            with self.assertRaises(TypeError):
                add_fpm_real_imag(perr, 0, 'real.fits', True, 'imag.fits', 6.6)

    def test_invalid_lam_index(self):
        """Invalid inputs caught."""
        for perr in invalid_lam_index_list:
            with self.assertRaises(TypeError):
                add_fpm_real_imag(
                    self.name, perr, 'real.fits', True, 'imag.fits', 6.6)

    def test_invalid_rfn(self):
        """Invalid inputs caught."""
        for perr in invalid_fn_list:
            with self.assertRaises(TypeError):
                add_fpm_real_imag(self.name, 0, perr, True, 'imag.fits', 6.6)

    def test_invalid_isopen(self):
        """Invalid inputs caught."""
        for perr in invalid_bool_list:
            with self.assertRaises(TypeError):
                add_fpm_real_imag(
                    self.name, 0, 'real.fits', perr, 'imag.fits', 6.6)

    def test_invalid_ifn(self):
        """Invalid inputs caught."""
        for perr in invalid_fn_list:
            with self.assertRaises(TypeError):
                add_fpm_real_imag(self.name, 0, 'real.fits', True, perr, 6.6)

    def test_invalid_ppl(self):
        """Invalid inputs caught."""
        for perr in invalid_real_positive_scalar_list:
            with self.assertRaises(TypeError):
                add_fpm_real_imag(
                    self.name, 0, 'real.fits', True, 'imag.fits', perr)


if __name__ == '__main__':
    unittest.main()
