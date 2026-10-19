# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Convert a numpy array into a binary stream and save it."""
import os
import stat
import numpy as np

from . import check


def np2bin(data, dtype, outfn, allownan=False):
    """
    Convert input data (scalar, ndarray) into a single stream of binary data.

    All large data structures are big-endian and row-major.

    Arguments:
     data: a Python scalar or numpy ndarray to convert to a byte stream.  Note
      that this function does not check any sort of dimensions against a spec.
      It will convert whatever it's given.
     dtype: string with one of the following values: float32, float64, int16,
      int32, uint8, uint16, uint32.  Any other input will raise a ValueError.
     outfn: string with absolute path of file to write to.  This function will
      overwrite any existing file at that location.

    Keyword arguments:
     allownan: permits NaNs to be in the output binary.  Only relevant for
      floating-point types.  Use should be limited to objects like Hall maps
      which are designed to handle missing data with NaNs.

    No returns, but writes a file to the location specified in outfn.

    """
    check.real_array(data, 'data', TypeError)
    check.string(dtype, 'dtype', TypeError)
    check.string(outfn, 'outfn', TypeError)
    check.boolean(allownan, 'allownan', TypeError)

    isfloat = False
    if dtype in ('float32', '>f4'):
        typestr = '>f4'
        isfloat = True
    elif dtype in ('float64', '>f8'):
        typestr = '>f8'
        isfloat = True
    elif dtype in ('int16', '>i2'):
        typestr = '>i2'
    elif dtype in ('int32', '>i4'):
        typestr = '>i4'
    elif dtype in('uint8', '>u1'):
        typestr = '>u1'
    elif dtype in ('uint16', '>u2'):
        typestr = '>u2'
    elif dtype in ('uint32', '>u4'):
        typestr = '>u4'
    else:
        raise ValueError('%s is an unsupported data type' % dtype)

    outfn = os.path.expandvars(outfn)

    # Check floats for infinite, NaN, or subnormal values.
    if isfloat:
        tmp = np.asarray(data).astype(typestr)

        # Force subnormals (and negative 0) to 0
        smallestnorm = np.finfo(tmp.dtype).tiny
        tmp[(tmp > -1*smallestnorm) & (tmp < smallestnorm)] = 0

        # Catch infinities
        if np.isinf(tmp).any():
            raise ValueError('Infinite values present in data')

        # Catch NaNs if desired
        if np.isnan(tmp).any() and not allownan:
            raise ValueError('NaN values present in data and allownan=False')

        # Write file. Give r+w permissions to all, but only if the file is new.
        file_already_existed = os.path.isfile(outfn)
        # print('Writing file    %s' % outfn)
        tmp.tofile(outfn)  
        if not file_already_existed:
            os.chmod(outfn,
                     stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP |
                     stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)

    else:
        # Write file. Give r+w permissions to all, but only if the file is new.
        file_already_existed = os.path.isfile(outfn)
        # print('Writing file    %s' % outfn)
        np.asarray(data).astype(typestr).tofile(outfn)
        if not file_already_existed:
            os.chmod(outfn,
                     stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP |
                     stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)    

    return None
