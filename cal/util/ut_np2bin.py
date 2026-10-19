# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Unit tests for writing a numpy scalar or array to a binary file."""
import os
import stat
from tempfile import NamedTemporaryFile
import unittest

import numpy as np

from .np2bin import np2bin


NACT = 48

not_a_real_array = (True, -1, 1, 1.1, 1j, 'string', {'a': 2})
not_a_2d_array = (True, -1, 1, 1.1, 1j, np.ones((5, 5, 4)), 'string', {'a': 2})
not_a_1d_array = (True, -1, 1, 1.1, 1j, np.ones((5, 5)), 'string', {'a': 2})
not_a_dict = (True, -1, 1, 1.1, 1j, np.ones(5), np.ones((5, 5)), 'string')
not_a_real_scalar = (1j, np.ones(5), 'string', {'a': 2})
not_a_real_positive_scalar = (-1, 0, 1j, np.ones(5), 'string', {'a': 2})
not_a_positive_scalar_integer = (-1, 0, 1.1, 1j, np.ones(5), np.ones((5, 5)),
                                 'string')
not_a_string = (True, -1, 0, 1.1, 1j, np.ones(5), np.ones((5, 5)),  {'a': 2})
not_a_boolean = (-1, 0, 1.1, 1j, np.ones(5), np.ones((5, 5)),  {'a': 2})


class TestInputErrors(unittest.TestCase):
    """Test successful file writing with np2bin.py."""

    def test_input_0_type(self):
        
        datatype = 'float32'
        f = NamedTemporaryFile(suffix='.bin')

        for bad_val in not_a_real_array:
            with self.assertRaises(TypeError):
                np2bin(bad_val, datatype, f.name)

    def test_input_1_type(self):

        f = NamedTemporaryFile(suffix='.bin')
        data = np.arange(NACT**2).reshape((NACT, NACT))

        for bad_val in not_a_string:
            with self.assertRaises(TypeError):
                np2bin(data, bad_val, f.name)

    def test_input_2_type(self):

        datatype = 'float32'
        data = np.arange(NACT**2).reshape((NACT, NACT))

        for bad_val in not_a_string:
            with self.assertRaises(TypeError):
                np2bin(data, datatype, bad_val)

    def test_input_3_type(self):

        datatype = 'float32'
        f = NamedTemporaryFile(suffix='.bin')
        data = np.arange(NACT**2).reshape((NACT, NACT))

        for bad_val in not_a_boolean:
            with self.assertRaises(TypeError):
                np2bin(data, datatype, f.name, allownan=bad_val)

    def test_input_0_value_inf(self):
        
        datatype = 'float32'
        f = NamedTemporaryFile(suffix='.bin')
        
        data = np.arange(NACT**2, dtype='>f4').reshape((NACT, NACT))
        data[3, 2] = np.inf

        with self.assertRaises(ValueError):
            np2bin(data, datatype, f.name)
            
    def test_input_0_value_nan(self):
        
        datatype = 'float32'
        f = NamedTemporaryFile(suffix='.bin')
        
        data = np.arange(NACT**2, dtype='>f4').reshape((NACT, NACT))
        data[3, 2] = np.nan

        with self.assertRaises(ValueError):
            np2bin(data, datatype, f.name, allownan=False)

    def test_input_0_value_nan_allowed(self):
        
        datatype = 'float32'
        f = NamedTemporaryFile(suffix='.bin')
        
        data = np.arange(NACT**2, dtype='>f4').reshape((NACT, NACT))
        data[3, 2] = np.nan

        # No error should occur
        np2bin(data, datatype, f.name, allownan=True)

    def test_input_1_value(self):
        
        data = np.arange(NACT**2).reshape((NACT, NACT))
        f = NamedTemporaryFile(suffix='.bin')
    
        for bad_val in ('<f4', 'string', 'float', 'int'):
            with self.assertRaises(ValueError):
                np2bin(data, bad_val, f.name)


class TestRemoveSubnormals(unittest.TestCase):
    """Tests for stripping subnormals out of numpy arrays."""

    def setUp(self):
        self.rng = np.random.default_rng(17273747)
        self.shape = (5, 5)
        self.data = self.rng.random(25).reshape(self.shape)
        # random in [0, 1], random + 2 in [2, 3] and can't be subnormal or -0
        self.data += 2

    def test_valid(self):
        """Correct answer with valid inputs"""
        # nbyte = 4  # for this data type
        dtype = '>f4'
        datatype = dtype

        # Refer to https://echorand.me/posts/named_temporary_file/ for help
        # with the temporary file usage.
        f = NamedTemporaryFile(suffix='.bin')
        
        np2bin(self.data, datatype, f.name)
        # file_size = os.path.getsize(f.name)

        self.assertTrue(os.path.isfile(f.name))
        
        data1 = np.fromfile(f.name, dtype=dtype)
        self.assertTrue((self.data.flatten().astype(dtype) == data1).all())

    def test_subnormals_float32(self):
        """Correct answer with subnormals, float32 edition"""
        dtype = '>f4'
        datatype = 'float32'
        
        dsub = self.data.copy().astype(datatype)
        dsub *= np.finfo(dsub.dtype).tiny
        dsub *= 0.1 # for margin
        target = np.zeros_like(dsub)
        
        # Refer to https://echorand.me/posts/named_temporary_file/ for help
        # with the temporary file usage.
        f = NamedTemporaryFile(suffix='.bin')
        np2bin(dsub, datatype, f.name)
        self.assertTrue(os.path.isfile(f.name))
        
        output = np.fromfile(f.name, dtype=dtype).reshape(self.shape)
        self.assertTrue((output == target).all())

    def test_subnormals_float64(self):
        """Correct answer with subnormals, float64 edition"""
        dtype = '>f8'
        datatype = 'float64'
        
        dsub = self.data.copy().astype(datatype)
        dsub *= np.finfo(dsub.dtype).tiny
        dsub *= 0.1 # for margin
        target = np.zeros_like(dsub)
        
        # Refer to https://echorand.me/posts/named_temporary_file/ for help
        # with the temporary file usage.
        f = NamedTemporaryFile(suffix='.bin')
        np2bin(dsub, datatype, f.name)
        self.assertTrue(os.path.isfile(f.name))
        
        output = np.fromfile(f.name, dtype=dtype).reshape(self.shape)
        self.assertTrue((output == target).all())

    def test_negzero_float32(self):
        """Correct answer with negative zero, float32 edition"""
        dtype = '>f4'
        datatype = 'float32'

        nrow = 2
        ncol = 2
        # don't astype from float64 to avoid overflow (default is float64)
        dneg = np.empty((nrow, ncol), dtype=datatype)
        for j in range(nrow):
            for k in range(ncol):
                dneg[j, k] = -0

        target = np.zeros((nrow, ncol)).astype(dneg.dtype)

        # Refer to https://echorand.me/posts/named_temporary_file/ for help
        # with the temporary file usage.
        f = NamedTemporaryFile(suffix='.bin')
        np2bin(dneg, datatype, f.name)
        self.assertTrue(os.path.isfile(f.name))

        output = np.fromfile(f.name, dtype=dtype).reshape((nrow, ncol))
        self.assertTrue((output == target).all())

    def test_negzero_float64(self):
        """Correct answer with negative zero, float64 edition"""
        dtype = '>f8'
        datatype = 'float64'

        nrow = 2
        ncol = 2
        dneg = np.empty((nrow, ncol), dtype=datatype)
        for j in range(nrow):
            for k in range(ncol):
                dneg[j, k] = -0

        target = np.zeros((nrow, ncol)).astype(dneg.dtype)

        # Refer to https://echorand.me/posts/named_temporary_file/ for help
        # with the temporary file usage.
        f = NamedTemporaryFile(suffix='.bin')
        np2bin(dneg, datatype, f.name)
        self.assertTrue(os.path.isfile(f.name))

        output = np.fromfile(f.name, dtype=dtype).reshape((nrow, ncol))
        self.assertTrue((output == target).all())


class TestSuccessfulWrite(unittest.TestCase):
    """Test successful file writing with np2bin.py."""

    def test_write_all_data_types(self):
        """Test successful file writing for all data types."""
        list_datatype = [
            'float32', '>f4', 'float64', '>f8', 'int16', '>i2', 'int32',
            '>i4', 'uint8', '>u1', 'uint16', '>u2', 'uint32', '>u4']
        list_dtype = [
            '>f4', '>f4', '>f8', '>f8', '>i2', '>i2', '>i4', '>i4',
            '>u1', '>u1', '>u2', '>u2', '>u4', '>u4']
        list_nbyte = [4, 4, 8, 8, 2, 2, 4, 4, 1, 1, 2, 2, 4, 4]

        for index, dtype in enumerate(list_dtype):

            nbyte = list_nbyte[index]
            datatype = list_datatype[index]

            # Refer to https://echorand.me/posts/named_temporary_file/ for help
            # with the temporary file usage.
            f = NamedTemporaryFile(suffix='.bin')

            data = np.arange(NACT**2).reshape((NACT, NACT))
            np2bin(data, datatype, f.name)
            file_size = os.path.getsize(f.name)

            self.assertTrue(os.path.isfile(f.name))
            self.assertTrue(file_size == NACT*NACT*nbyte)

            data1 = np.fromfile(f.name, dtype=dtype)
            self.assertTrue(np.array_equal(data.flatten().astype(dtype), data1))

    def test_permissions(self):

        dtype = '>u4'
        datatype = dtype
        f = NamedTemporaryFile(suffix='.bin')
        
        data = np.arange(NACT**2).reshape((NACT, NACT))
        np2bin(data, datatype, f.name)

        self.assertTrue(os.path.isfile(f.name))

        # Cannot change permissions on a temp file, so check only user
        # permissions.
        st = os.stat(f.name)
        self.assertTrue(bool(st.st_mode & stat.S_IRUSR))
        self.assertTrue(bool(st.st_mode & stat.S_IWUSR))
        # self.assertTrue(bool(st.st_mode & stat.S_IRGRP))
        # self.assertTrue(bool(st.st_mode & stat.S_IWGRP))
        # self.assertTrue(bool(st.st_mode & stat.S_IROTH))
        # self.assertTrue(bool(st.st_mode & stat.S_IWOTH))


if __name__ == '__main__':
    unittest.main()
