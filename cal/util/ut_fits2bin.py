# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Unit tests for writing a numpy scalar or array to a binary file."""
import os
import stat
from tempfile import NamedTemporaryFile
import unittest

from astropy.io import fits
import numpy as np

from .fits2bin import fits2bin
from .np2bin import np2bin

HERE = os.path.dirname(os.path.abspath(__file__))


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
    """Test successful file writing with fits2bin.py."""

    def test_input_0_type(self):
        
        # Create FITS file first
        dtype = '>f4'
        # Refer to https://echorand.me/posts/named_temporary_file/ for help
        # with the temporary file usage.
        f = NamedTemporaryFile(suffix='.fits')
        data = np.arange(NACT**2, dtype=dtype).reshape((NACT, NACT))
        fits.writeto(f.name, data, overwrite=True)
        self.assertTrue(os.path.isfile(f.name))
        
        for bad_val in not_a_string:
            with self.assertRaises(TypeError):
                fits2bin(bad_val, f.name, fn_out=None)

    def test_input_1_type(self):

        # Create .bin file first
        datatype = 'float32'
        # Refer to https://echorand.me/posts/named_temporary_file/ for help
        # with the temporary file usage.
        f = NamedTemporaryFile(suffix='.bin')
        data = np.arange(NACT**2).reshape((NACT, NACT))
        np2bin(data, datatype, f.name)
        self.assertTrue(os.path.isfile(f.name))

        for bad_val in not_a_string:
            with self.assertRaises(TypeError):
                fits2bin(datatype, bad_val, fn_out=None)

    def test_input_2_type(self):

        # Create FITS file first
        dtype = '>f4'
        # Refer to https://echorand.me/posts/named_temporary_file/ for help
        # with the temporary file usage.
        f = NamedTemporaryFile(suffix='.fits')
        data = np.arange(NACT**2, dtype=dtype).reshape((NACT, NACT))
        fits.writeto(f.name, data, overwrite=True)
        self.assertTrue(os.path.isfile(f.name))

        for bad_val in not_a_string:
            with self.assertRaises(TypeError):
                fits2bin(bad_val, f.name, fn_out=bad_val)

    def test_input_1_value(self):
        
        # Create FITS file first
        dtype = '>f4'
        # Refer to https://echorand.me/posts/named_temporary_file/ for help
        # with the temporary file usage.
        f = NamedTemporaryFile(suffix='.fits')
        data = np.arange(NACT**2, dtype=dtype).reshape((NACT, NACT))
        fits.writeto(f.name, data, overwrite=True)
        self.assertTrue(os.path.isfile(f.name))

        for bad_val in ('<f4', 'string', 'float', 'int'):
            with self.assertRaises(ValueError):
                print(bad_val)
                fits2bin(bad_val, f.name)


class TestSuccessfulWrite(unittest.TestCase):
    """Test successful file writing with fits2bin.py."""
    def setUp(self):
        self.fn_fits = os.path.join(HERE, 'testdata', 'ut_arange_48x48_float32.fits')
        self.data0 = fits.getdata(self.fn_fits)

    def test_automatic_out_filename(self):
        """Test that no output filename makes the expected one."""

        # Create FITS file first
        dtype = '>f4'
        datatype = 'float32'
        # Refer to https://echorand.me/posts/named_temporary_file/ for help
        # with the temporary file usage.
        f = NamedTemporaryFile(suffix='.fits')
        data = np.arange(NACT**2, dtype=dtype).reshape((NACT, NACT))
        fits.writeto(f.name, data, overwrite=True)
        self.assertTrue(os.path.isfile(f.name))
        
        fn_bin = os.path.splitext(f.name)[0] + '.bin'
        
        fits2bin(datatype, f.name)
        self.assertTrue(os.path.isfile(fn_bin))

        data1 = np.fromfile(fn_bin, dtype=dtype)
        print(data.flatten().astype(dtype).shape)
        print(data1.astype(dtype).shape)
        
        self.assertTrue(np.array_equal(data.flatten().astype(dtype),
                                       data1.astype(dtype)))

    def test_write_all_data_types(self):
        """Test for a dataype using an existing FITS file."""

        list_datatype = [
            'float32', '>f4', 'float64', '>f8', 'int16', '>i2', 'int32',
            '>i4', 'uint8', '>u1', 'uint16', '>u2', 'uint32', '>u4']
        list_dtype = [
            '>f4', '>f4', '>f8', '>f8', '>i2', '>i2', '>i4', '>i4',
            '>u1', '>u1', '>u2', '>u2', '>u4', '>u4']

        for index, dtype in enumerate(list_dtype):

            datatype = list_datatype[index]
            # dtype = '>f4'
            # datatype = 'float32'
            # Refer to https://echorand.me/posts/named_temporary_file/ for help
            # with the temporary file usage.
            f = NamedTemporaryFile(suffix='.bin')

            fits2bin(datatype, self.fn_fits, fn_out=f.name)
            self.assertTrue(os.path.isfile(f.name))

            # Verify the written values
            data1 = np.fromfile(f.name, dtype=dtype)
            self.assertTrue(np.array_equal(self.data0.flatten().astype(dtype),
                                           data1.astype(dtype)))


if __name__ == '__main__':
    unittest.main()