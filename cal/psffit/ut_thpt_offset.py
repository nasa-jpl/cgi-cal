# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Unit tests for CFAM filter relative throughput/offset calcs
"""

import unittest
import os

import numpy as np
import astropy.io.fits as pyfits
import scipy

from cal.psffit.thpt_offset import thpt_offset

class TestThptOffset(unittest.TestCase):
    """
    x success
    x bad inputs fail
    x outputs are right size and type (and order)
    x functionally, this does the right thing for offset and amp, including
     example in docstring
    x refindex used correctly (switch order with same arrays)
    does the right thing given band 1a and band 4c data

    """

    def setUp(self):
        self.ref = np.array([[0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]])
        self.targ0 = np.array([[0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 3],
                               [0, 0, 0, 0, 0]])
        self.ref_thpt = 1
        self.ref_offset = (0, 0)
        self.targ0_thpt = 3
        self.targ0_offset = (1, 2)

        self.lamlist = [600, 600]

        pass


    def test_success(self):
        """Good inputs return successfully"""
        thpt_offset([self.ref, self.targ0], 0, self.lamlist)
        pass


    def test_right_outputs(self):
        """expected values return from expected inputs"""
        thpts, offsets = thpt_offset([self.ref, self.targ0], 0, self.lamlist)

        tol = 1e-13

        self.assertTrue(np.max(np.abs(np.array(thpts)
                        - np.array([self.ref_thpt, self.targ0_thpt]))) < tol)
        self.assertTrue(np.max(np.abs(np.array(offsets)
                        - np.array([self.ref_offset,
                                    self.targ0_offset]))) < tol)
        pass


    def test_right_outputs_non_first(self):
        """nonzero refindex handled correctly"""
        thpts, offsets = thpt_offset([self.targ0, self.targ0, self.ref], 2,
                                     [600, 600, 600])

        tol = 1e-13

        self.assertTrue(np.max(np.abs(np.array(thpts)
                        - np.array([self.targ0_thpt,
                                    self.targ0_thpt, self.ref_thpt]))) < tol)
        self.assertTrue(np.max(np.abs(np.array(offsets)
                        - np.array([self.targ0_offset,
                                    self.targ0_offset,
                                    self.ref_offset]))) < tol)
        pass


    def test_invalid_imlist(self):
        """invalid inputs caught"""
        perrlist = [(self.ref, self.targ0), self.targ0, 0, 'self.ref', None]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                thpt_offset(imlist=perr, refindex=0, lamlist=self.lamlist)
                pass
            pass
        pass


    def test_invalid_imlist_elements(self):
        """invalid inputs caught"""
        perrlist = [
            [self.ref[None, :, :], self.targ0[None, :, :]], # not 2D
            [1j*self.ref, 1j*self.targ0], # not real
            [self.ref, self.targ0[:-1, :]], # not same size
        ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                thpt_offset(imlist=perr, refindex=0, lamlist=self.lamlist)
                pass
            pass
        pass


    def test_invalid_refindex_type(self):
        """invalid inputs caught"""
        imlist = [self.ref, self.targ0]
        perrlist = [0.5, 1j, 'txt', (0,), None]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                thpt_offset(imlist=imlist, refindex=perr, lamlist=self.lamlist)
                pass
            pass
        pass


    def test_refindex_bounds(self):
        """invalid inputs caught but valid ones aren't"""
        imlist = [self.ref, self.ref, self.targ0, self.targ0]
        nim = len(imlist)

        valid_list = np.arange(nim)
        invalid_list = [-1, nim, nim+1]

        for v in valid_list:
            thpt_offset(imlist=imlist, refindex=v,
                        lamlist=[600, 600, 600, 600])

        for perr in invalid_list:
            with self.assertRaises(TypeError):
                thpt_offset(imlist=imlist, refindex=perr,
                            lamlist=[600, 600, 600, 600])
                pass
            pass
        pass


    def test_invalid_lamlist(self):
        """invalid inputs caught"""
        perrlist = [(self.ref, self.targ0), self.targ0, 0, 'self.ref', None]

        imlist = [self.ref, self.targ0]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                thpt_offset(imlist=imlist, refindex=0, lamlist=perr)
                pass
            pass
        pass


    def test_invalid_lamlist_elements(self):
        """invalid inputs caught"""
        perrlist = [
            [600, None],
            [600, (600,)],
            [600, 'txt'],
            [1j, 600],
            [0, 600],
            [-600, 600],
        ]
        imlist = [self.ref, self.targ0]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                thpt_offset(imlist=imlist, refindex=0, lamlist=perr)
                pass
            pass
        pass


    def test_lamlist_imlist_len(self):
        """invalid inputs caught"""
        imlist = [self.ref, self.targ0]
        badlamlist = [600]*(len(imlist)+1)

        with self.assertRaises(TypeError):
            thpt_offset(imlist=imlist, refindex=0, lamlist=badlamlist)
            pass
        pass


    def test_1b_vs_4b(self):
        """
        Test that this tool works across bands to capture shift

        use two that are far away: 1b (550nm) and 4b (825nm)
        """

        tol = 1e-13

        # psf1b and psf4b are cgisim outputs courtesy D. Marx for the thpt vs.
        # filter sequence (CGI_SEQ_THROUGHPUT_VS_FILTER)

        localpath = os.path.dirname(os.path.abspath(__file__))
        psf1bfn = os.path.join(localpath, 'ut_data', 'ut_psf_1b.fits')
        psf4bfn = os.path.join(localpath, 'ut_data', 'ut_psf_4b.fits')

        psf1b = pyfits.getdata(psf1bfn)
        psf4b = pyfits.getdata(psf4bfn)

        _, offsets = thpt_offset([psf1b, psf4b], 0, [575, 825])

        # use a controlled relative shift, to reflect the fact that cgisim
        # internals do not co-center all PSFs in all bands (nor should they,
        # this is a property of internal polarization aberration)

        delta = (2.5, -0.67)
        psf4b_shift = scipy.ndimage.shift(psf4b, delta)

        _, delta_offsets = thpt_offset([psf1b, psf4b_shift], 0, [575, 825])

        self.assertTrue(np.abs(delta_offsets[1][0]
                               - (offsets[1][0] + delta[0])) < tol)
        self.assertTrue(np.abs(delta_offsets[1][1]
                               - (offsets[1][1] + delta[1])) < tol)
        pass



if __name__ == '__main__':
    unittest.main()
