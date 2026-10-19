# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Test class for the functions for reading phase retrievals."""
import os
import unittest

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

from cal.pr.util.read_pr import load_backend_pr, read_and_prop_pr
from cal.util.insertinto import insertinto as inin
from cal.util.loadyaml import loadyaml
from cal.util.remove_ptt import remove_ptt_directly

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND_PR_PATH = os.path.join(HERE, 'cgidata', 'prnum_000279')

NOT_REAL_2D_ARRAY = [
    False, -1, 0, 'txt', None, [1, 2, 3],  # not array
    np.ones((100,)), np.ones((5, 5, 5)),  # not 2D
    1j*np.eye(2),  # not real
]

NOT_DICT = [
    False, -1, 0, 1J, 'txt', None, [1, 2, 3], (1, 2, 3), np.ones((100,)),
]

NOT_BOOL = [
    -1, 1J, 'txt', None, [1, 2, 3], (1, 2, 3), np.ones((100,)),
]


class TestReadAndPropPR(unittest.TestCase):
    """
    X Valid inputs succeed.
    X Backend phase is subtracted when asked.
    X Phase is unchanged if backend is not subtracted.
    X Piston, tip, and tilt are removed from the phase.
    X Small blacked out regions are removed from the software mask.
    X Fresnel propagation occurs when asked.
    X Output phase is unwrapped.
    Bad inputs fail.
    """

#     @classmethod
#     def setUpClass(self):
#         """Predefine common variable inputs available for all tests."""
#         self.amp = fits.getdata(os.path.join(
#             HERE, 'testdata', 'prnum_000259', 'amp.fits'))
#         self.phwrap = fits.getdata(os.path.join(
#             HERE, 'testdata', 'prnum_000259', 'phwrap.fits'))
#         self.pr_params = loadyaml(os.path.join(
#             HERE, 'testdata', 'prnum_000259', 'parms.yaml'))
#         self.fn_pr_prop_params = os.path.join(
#             HERE, 'cgidata', 'params_read_and_prop_pr.yaml')
#         self.fn_pr_prop_params_zero_deltaz = os.path.join(
#             HERE, 'testdata', 'params_read_and_prop_pr_zero_deltaz.yaml')
#         self.fn_pr_prop_params_no_min = os.path.join(
#             HERE, 'testdata', 'params_read_and_prop_pr_zero_min.yaml')
#         self.subtract_backend = True



    def test_259(self):
        """PR before flattening."""
        self.amp = fits.getdata(os.path.join(
            HERE, 'testdata', 'prnum_000259', 'amp.fits'))
        self.phwrap = fits.getdata(os.path.join(
            HERE, 'testdata', 'prnum_000259', 'phwrap.fits'))
        self.pr_params = loadyaml(os.path.join(
            HERE, 'testdata', 'prnum_000259', 'parms.yaml'))
        self.fn_pr_prop_params = os.path.join(
            HERE, 'cgidata', 'params_read_and_prop_pr.yaml')
        self.fn_pr_prop_params_zero_deltaz = os.path.join(
            HERE, 'testdata', 'params_read_and_prop_pr_zero_deltaz.yaml')
        self.fn_pr_prop_params_no_min = os.path.join(
            HERE, 'testdata', 'params_read_and_prop_pr_zero_min.yaml')
        self.subtract_backend = True
        
        amp_pupil, ph_pupil, bmask_pupil, x_offset, y_offset, lam = \
            read_and_prop_pr(
                self.amp,
                self.phwrap,
                self.pr_params,
                self.fn_pr_prop_params,
                subtract_backend=self.subtract_backend,
                )     
        ph_pre = ph_pupil   

#     def test_264(self):
        """PR before flattening."""
        self.amp = fits.getdata(os.path.join(
            HERE, 'testdata', 'prnum_000264', 'amp.fits'))
        self.phwrap = fits.getdata(os.path.join(
            HERE, 'testdata', 'prnum_000264', 'phwrap.fits'))
        self.pr_params = loadyaml(os.path.join(
            HERE, 'testdata', 'prnum_000264', 'parms.yaml'))
        self.fn_pr_prop_params = os.path.join(
            HERE, 'cgidata', 'params_read_and_prop_pr.yaml')
        self.fn_pr_prop_params_zero_deltaz = os.path.join(
            HERE, 'testdata', 'params_read_and_prop_pr_zero_deltaz.yaml')
        self.fn_pr_prop_params_no_min = os.path.join(
            HERE, 'testdata', 'params_read_and_prop_pr_zero_min.yaml')
        self.subtract_backend = True
        
        amp_pupil, ph_pupil, bmask_pupil, x_offset, y_offset, lam = \
            read_and_prop_pr(
                self.amp,
                self.phwrap,
                self.pr_params,
                self.fn_pr_prop_params,
                subtract_backend=self.subtract_backend,
                )
        ph_post = ph_pupil
        
        ph_pre[ph_pre == 0] = np.nan
        ph_post[ph_post == 0] = np.nan
        sz = 12
        
        plt.figure(1)
        plt.imshow(ph_pre[100:150, 100:150])
        cb1 = plt.colorbar()
        cb1.set_label('Radians', size=sz)
        plt.set_cmap('magma')
        plt.gca().invert_yaxis()
        plt.clim(-4, -1)
#         plt.gca().axis_off()
#         ax.set_axis_off()
        fn_png = os.path.join(HERE, 'figures', 'fig_dead_act_pr_pre_flatten.png')
        plt.savefig(fn_png, bbox_inches='tight', pad_inches = 0.1)
        
        plt.figure(2)
        plt.imshow(ph_post[100:150, 100:150])
        cb2 = plt.colorbar()
        cb2.set_label('Radians', size=sz)
        plt.set_cmap('magma')
        plt.gca().invert_yaxis()
        plt.clim(-2, 1)
#         plt.set_axis_off()
        fn_png = os.path.join(HERE, 'figures', 'fig_dead_act_pr_post_flatten.png')
        plt.savefig(fn_png, bbox_inches='tight', pad_inches = 0.1)


        plt.show()


# 
#     def test_prop_occurred(self):
#         """Test that the phase and amplitude changed from propagation."""
#         # No Fresnel propagation
#         amp_pupil, ph_pupil, bmask_pupil, x_offset, y_offset, lam = \
#             read_and_prop_pr(
#                 self.amp,
#                 self.phwrap,
#                 self.pr_params,
#                 self.fn_pr_prop_params_zero_deltaz,
#                 subtract_backend=self.subtract_backend,
#                 )
# 
#         # With Fresnel propagation
#         amp_pupil2, ph_pupil2, bmask_pupil2, x_offset2, y_offset2, lam2 = \
#             read_and_prop_pr(
#                 self.amp,
#                 self.phwrap,
#                 self.pr_params,
#                 self.fn_pr_prop_params,
#                 subtract_backend=self.subtract_backend,
#                 )
# 
#         self.assertTrue(not np.allclose(bmask_pupil*ph_pupil,
#                                         bmask_pupil2*ph_pupil2,
#                                         atol=1e-3))
#         self.assertTrue(not np.allclose(amp_pupil, amp_pupil2, atol=1e-8))
# 
#     def test_phase_unwrapped(self):
#         """Test that the output phase is unwrapped."""
#         # No Fresnel propagation
#         amp_pupil, ph_pupil, bmask_pupil, x_offset, y_offset, lam = \
#             read_and_prop_pr(
#                 self.amp,
#                 self.phwrapPoints,
#                 self.pr_params,
#                 self.fn_pr_prop_params_zero_deltaz,
#                 subtract_backend=False,
#                 )
# 
#         pv_input = np.max(bmask_pupil*self.phwrapPoints) -\
#             np.min(bmask_pupil*self.phwrapPoints)
#         pv_output = np.max(bmask_pupil*ph_pupil) - np.min(bmask_pupil*ph_pupil)
# 
#         self.assertTrue(pv_input > self.range)
#         self.assertTrue(pv_output < 2*np.pi)
# 


if __name__ == '__main__':
    unittest.main()
