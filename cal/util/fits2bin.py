# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Convert the contents of a FITS file into a binary stream and save it."""
import argparse
import os
import stat

from astropy.io import fits

from . import check, np2bin


def fits2bin(dtype, fn_in, fn_out=None):
    """
    Convert data in a FITS file into a file of a single stream of binary data.

    All large data structures are big-endian and row-major.
     
    No returns, but writes a file to the location specified by fn_out.

    Parameters
    ----------
    dtype : str
        String with one of the following values: float32, float64, int16,
        int32, uint8, uint16, uint32.  Any other input will raise a ValueError.
    fn_in : str
        String with absolute path of FITS file to read in. This function
        will convert whatever is in that file without any checks.
    fn_out : str
        String with absolute path of file to write to.  This function will
        overwrite any existing file at that location. If None, reuses the path
        and base filename of the input filename. Default is None.

    Returns
    -------
    None.

    """
    check.string(dtype, 'dtype', TypeError)
    check.string(fn_in, 'fn_in', TypeError)

    # If fn_out is not given, just use the same path and filename but
    # change the file extension from FITS to .bin
    if fn_out is None:
        fn_out = os.path.splitext(fn_in)[0] + '.bin'
    check.string(fn_out, 'fn_out', TypeError)

    fn_in = os.path.expandvars(fn_in)
    fn_out = os.path.expandvars(fn_out)

    # Check if file already exists. If yes and made by a different user,
    # we cannot change permissions
    file_already_existed = os.path.isfile(fn_out)

    # Write the bin file
    np2bin.np2bin(fits.getdata(fn_in), dtype, fn_out)

    # Give r+w permissions to all, but only if the file is new
    if not file_already_existed:
        os.chmod(fn_out, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP |
                 stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)
    
    return None


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        prog='Convert .fits to .bin.',
        description='Convert data in a FITS file into a file of a single '
        'stream of binary data.',
        )
    parser.add_argument(
        "dtype",
        type=str,
        help="string with one of the following values: float32, float64, "
        "int16, int32, uint8, uint16, uint32.  Any other input will raise a "
        "ValueError.",
        )
    parser.add_argument(
        "fn_in",
        type=str,
        help="Full path and name of fits ""file to convert.",
        )
    parser.add_argument(
        "--fn_out",
        type=str,
        default=None,
        help="Full path and name of output .bin file to write. If None, "
        "reuses the path and base filename of the given FITS file.")

    args = parser.parse_args()

    fits2bin(
        args.dtype,
        args.fn_in,
        fn_out=args.fn_out,
    )
