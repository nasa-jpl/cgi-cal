# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Test class for dmsatspots.ddmv_satspots
"""

import unittest

import numpy as np
import scipy.ndimage as ndimage

from cal.util.insertinto import insertinto
from cal.util.dmsatspots import ddmv_satspots
from cal.pr.ut_gaussprop import creategrid

# functions to calculate the focal plane intensity from return ddmv
def fFocus(Ep, Nf):
    """Return focal plane field from pupil, of size Nf x Nf"""
    Etmp = insertinto(Ep, (Nf, Nf))
    Ef = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(Etmp)))
    return np.abs(Ef)**2

def NI_spots_focal_plane(ddmv, D, lam, dm_gain_m_per_v):
    """
    given the ddmv array returned from dmsatspots, calculate the focal plane
    and NI of the spots in the focal plane

    return lists of NI, x, y for each found spot. x and y are (lam/D),
       and pix_f, pixel size in focal plane
    """
    #
    N = ddmv.shape[0]

    # fft interpolated oversampling in focal plane
    Nf = 8*N
    pix_f = D/Nf # pixel size in focal plane in lam/D

    # create Epup in DM space, low pass filter the edges
    _, _, _, _, R = creategrid((N, N))
    h = np.outer(np.hanning(5), np.hanning(5))
    Eref = np.ones((N, N)) * (R < D/2)
    Eref = ndimage.convolve(Eref, h)

    # focal plane of reference
    Iref = fFocus(Eref, Nf)
    I0 = np.max(Iref)

    # focal plane with speckle spot, use +/- ddmv and subtract Iref
    ddm_m = ddmv * dm_gain_m_per_v
    wfe_m = 2*ddm_m * 2*np.pi/lam
    Espot_p = Eref * np.exp(1j*wfe_m)
    Espot_m = Eref * np.exp(-1j*wfe_m)
    Ispot = 0.5*(fFocus(Espot_p, Nf) + fFocus(Espot_m, Nf)) - Iref
    Ispot[Ispot < 0.0] = 1e-9*I0
    Ispot = Ispot / I0

    # focus plane, grid in lam/D
    _, _, Xf, Yf, _ = creategrid((Nf, Nf), pix_f)

    # how many sat spots?
    Ispot_max = np.max(Ispot)
    img_bw = np.ones(Ispot.shape) * (Ispot > 0.95*Ispot_max)
    label, num_features = ndimage.label(img_bw,
                                        [[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    x_spot = np.zeros((num_features,))
    y_spot = np.zeros((num_features,))
    NI_spot = np.zeros((num_features,))
    for ispot, ilabel in enumerate(range(1, 1+num_features)):
        NI_spot[ispot] = np.max(Ispot[label == ilabel])
        ii = np.argmax(Ispot[label == ilabel])
        x_spot[ispot] = Xf[label == ilabel][ii]
        y_spot[ispot] = Yf[label == ilabel][ii]

    return NI_spot, x_spot, y_spot, pix_f

class Test_ddmv_satspots(unittest.TestCase):
    """
    Unit test for function to generate ddmv for satellite spots
    """

    def setUp(self):
        """
        Predefine OK parameters
        """
        self.NI_spot = 1e-5
        self.radius_spot = 10 # lam/D
        self.theta_spot = np.array([0, 45, 90, 135])
        self.sine_phase = 0

        self.N = 48
        self.D = 46.0

        self.lam = 575e-9
        self.dm_gain_m_per_v = 4e-9*np.ones((self.N, self.N)) # m/V

    # test inputs
    def test_success(self):
        """
        Run with valid inputs should not throw anything
        """
        ddmv_satspots(self.radius_spot, self.theta_spot, self.NI_spot,
                      self.sine_phase, self.D, self.N, self.lam,
                      self.dm_gain_m_per_v)

    def test_radius_spot_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, -1.5, 0, [1.0, 2.0]]:
            with self.assertRaises(TypeError):
                ddmv_satspots(perr, self.theta_spot, self.NI_spot,
                              self.sine_phase, self.D, self.N, self.lam,
                              self.dm_gain_m_per_v)


    def test_radius_spot_warning(self):
        """Test questionable values warn as documented."""
        with self.assertWarns(UserWarning):
            ddmv_satspots(5.999, self.theta_spot, self.NI_spot,
                          self.sine_phase, self.D, self.N, self.lam,
                          self.dm_gain_m_per_v)
            pass


    def test_theta_spot_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, [[-1.0, 1.0], [-1.0, 1.0]]]:
            with self.assertRaises(TypeError):
                ddmv_satspots(self.radius_spot, perr, self.NI_spot,
                              self.sine_phase, self.D, self.N, self.lam,
                              self.dm_gain_m_per_v)

    def test_NI_spot_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, -1.0, [[-1.0, 1.0], [-1.0, 1.0]]]:
            with self.assertRaises(TypeError):
                ddmv_satspots(self.radius_spot, self.theta_spot, perr,
                              self.sine_phase, self.D, self.N, self.lam,
                              self.dm_gain_m_per_v)


    def test_NI_spot_warning(self):
        """Test questionable values warn as documented."""
        with self.assertWarns(UserWarning):
            ddmv_satspots(self.radius_spot, self.theta_spot, 0.999e-7,
                          self.sine_phase, self.D, self.N, self.lam,
                          self.dm_gain_m_per_v)
            pass


    def test_sine_phase_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, [[-1.0, 1.0], [-1.0, 1.0]]]:
            with self.assertRaises(TypeError):
                ddmv_satspots(self.radius_spot, self.theta_spot, self.NI_spot,
                              perr, self.D, self.N, self.lam,
                              self.dm_gain_m_per_v)

    def test_D_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, -1.0, [[46.3, 46.3], [46.3, 46.3]]]:
            with self.assertRaises(TypeError):
                ddmv_satspots(self.radius_spot, self.theta_spot, self.NI_spot,
                              self.sine_phase, perr, self.N, self.lam,
                              self.dm_gain_m_per_v)

    def test_N_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, -1.0, [[48, 47], [48, 47]]]:
            with self.assertRaises(TypeError):
                ddmv_satspots(self.radius_spot, self.theta_spot, self.NI_spot,
                              self.sine_phase, self.D, perr, self.lam,
                              self.dm_gain_m_per_v)

    def test_lam_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, -1.0, [[5e-7, 5e-7], [5e-7, 5e-7]]]:
            with self.assertRaises(TypeError):
                ddmv_satspots(self.radius_spot, self.theta_spot, self.NI_spot,
                              self.sine_phase, self.D, self.N, perr,
                              self.dm_gain_m_per_v)

    def test_dm_gain_m_per_v_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, -1.0, [[5e-7, 5e-7], [5e-7, 5e-7]]]:
            with self.assertRaises(TypeError):
                ddmv_satspots(self.radius_spot, self.theta_spot, self.NI_spot,
                              self.sine_phase, self.D, self.N, self.lam,
                              perr)

    def test_x_aoi_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, -1.0, (1,)]:
            with self.assertRaises(TypeError):
                ddmv_satspots(self.radius_spot, self.theta_spot, self.NI_spot,
                              self.sine_phase, self.D, self.N, self.lam,
                              self.dm_gain_m_per_v,
                              x_aoi=perr,
                )


    def test_y_aoi_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, -1.0, (1,)]:
            with self.assertRaises(TypeError):
                ddmv_satspots(self.radius_spot, self.theta_spot, self.NI_spot,
                              self.sine_phase, self.D, self.N, self.lam,
                              self.dm_gain_m_per_v,
                              y_aoi=perr,
                )


    def test_theta_0_xaxis(self):
        """
        From the TDD FDD requirement:
        1) spot location in lam/D
        2) CCW rotation angle about center relative to the DM x-axis
        3) fractional amplitude of spot relative to the PSF peak (we test NI
           instead of amplitude)
        4) spot phase

        This tests "relative to the x-axis". It means that theta_spot = 0 puts
        the spots on the x-axis, as well as spot location (radius) and NI.
        """
        theta_spot = 0.0
        ddmv_spot = ddmv_satspots(self.radius_spot, theta_spot, self.NI_spot,
                                  self.sine_phase, self.D, self.N, self.lam,
                                  self.dm_gain_m_per_v)

        # calculate focal plane and check spots on x-axis
        NI_spot, x_spot, y_spot, pix_f = NI_spots_focal_plane(
            ddmv_spot, self.D, self.lam, self.dm_gain_m_per_v
        )

        # check two spots (one pair)
        self.assertTrue(len(NI_spot) == 2)
        # check on x-axis (y_spot == 0)
        self.assertTrue(np.max(np.abs(y_spot)) < 0.5*pix_f)
        # check spot radius (x_spot == +/- radius_spot
        self.assertTrue(
            np.max(np.abs(np.abs(x_spot) - self.radius_spot)) < pix_f)
        # check spot intensity
        self.assertTrue(np.min(NI_spot/self.NI_spot) > 0.99)
        self.assertTrue(np.max(NI_spot/self.NI_spot) < 1.01)


    def test_theta_CCW(self):
        """
        From the TDD FDD requirement:
        1) spot location in lam/D
        2) CCW rotation angle about center relative to the DM x-axis
        3) fractional amplitude of spot relative to the PSF peak (we test NI
           instead of amplitude)
        4) spot phase

        This tests "relative to the x-axis". It means that theta_spot = 45 puts
        the spots 1st and 3rd quadrants.

        """
        theta_spot = 45
        ddmv_spot = ddmv_satspots(self.radius_spot, theta_spot, self.NI_spot,
                                  self.sine_phase, self.D, self.N, self.lam,
                                  self.dm_gain_m_per_v)

        # calculate focal plane and check spots on x-axis
        NI_spot, x_spot, y_spot, pix_f = NI_spots_focal_plane(
            ddmv_spot, self.D, self.lam, self.dm_gain_m_per_v
        )

        # check two spots (one pair)
        self.assertTrue(len(NI_spot) == 2)
        # check spots are on the x=y line, which proves 45deg and 1st and 3rd
        # quadrants
        self.assertTrue(np.max(np.abs(x_spot - y_spot)) < 1e-3)
        # check spot radius
        self.assertTrue(
           np.max(np.abs(np.hypot(x_spot, y_spot) - self.radius_spot)) < pix_f)
        # check spot intensity
        self.assertTrue(np.min(NI_spot/self.NI_spot) > 0.99)
        self.assertTrue(np.max(NI_spot/self.NI_spot) < 1.01)


    def test_x_aoi_exact(self):
        """
        Check that AOI terms reduce spatial frequency as expected.
        """

        rlamD = 12
        # cos(60 degrees) = 1/2
        tol = 1e-13

        ddmv_normal = ddmv_satspots(rlamD/2, 0, self.NI_spot,
                                  self.sine_phase, self.D, self.N, self.lam,
                                  self.dm_gain_m_per_v)
        ddmv_aoi = ddmv_satspots(rlamD, 0, self.NI_spot,
                                  self.sine_phase, self.D, self.N, self.lam,
                                  self.dm_gain_m_per_v, x_aoi=60)

        self.assertTrue(np.max(np.abs(ddmv_normal - ddmv_aoi)) < tol)


    def test_y_aoi_exact(self):
        """
        Check that AOI terms reduce spatial frequency as expected.
        """

        rlamD = 12
        # cos(60 degrees) = 1/2
        tol = 1e-13

        ddmv_normal = ddmv_satspots(rlamD/2, 90, self.NI_spot,
                                  self.sine_phase, self.D, self.N, self.lam,
                                  self.dm_gain_m_per_v)
        ddmv_aoi = ddmv_satspots(rlamD, 90, self.NI_spot,
                                  self.sine_phase, self.D, self.N, self.lam,
                                  self.dm_gain_m_per_v, y_aoi=60)

        self.assertTrue(np.max(np.abs(ddmv_normal - ddmv_aoi)) < tol)


    def test_x_aoi_cross(self):
        """
        Check that AOI terms do not affect spatial frequencies on the
        orthogonal axis.
        """

        rlamD = 8
        tol = 1e-13

        ddmv_normal = ddmv_satspots(rlamD, 0, self.NI_spot,
                                  self.sine_phase, self.D, self.N, self.lam,
                                  self.dm_gain_m_per_v)
        ddmv_aoi = ddmv_satspots(rlamD, 0, self.NI_spot,
                                  self.sine_phase, self.D, self.N, self.lam,
                                  self.dm_gain_m_per_v, y_aoi=60)

        self.assertTrue(np.max(np.abs(ddmv_normal - ddmv_aoi)) < tol)


    def test_y_aoi_cross(self):
        """
        Check that AOI terms do not affect spatial frequencies on the
        orthogonal axis.
        """

        rlamD = 8
        tol = 1e-13

        ddmv_normal = ddmv_satspots(rlamD, 90, self.NI_spot,
                                  self.sine_phase, self.D, self.N, self.lam,
                                  self.dm_gain_m_per_v)
        ddmv_aoi = ddmv_satspots(rlamD, 90, self.NI_spot,
                                  self.sine_phase, self.D, self.N, self.lam,
                                  self.dm_gain_m_per_v, x_aoi=60)

        self.assertTrue(np.max(np.abs(ddmv_normal - ddmv_aoi)) < tol)



    def test_multiple_spots(self):
        """
        From the TDD FDD requirement:
        1) spot location in lam/D
        2) CCW rotation angle about center relative to the DM x-axis
        3) fractional amplitude of spot relative to the PSF peak (we test NI
           instead of amplitude)
        4) spot phase

        This tests "comprised of a sum of sinusoids" by creating four pairs of
        spots

        """
        ddmv_spot = ddmv_satspots(self.radius_spot, self.theta_spot,
                                  self.NI_spot,
                                  self.sine_phase, self.D, self.N, self.lam,
                                  self.dm_gain_m_per_v)

        # calculate focal plane and check spots on x-axis
        NI_spot, x_spot, y_spot, pix_f = NI_spots_focal_plane(
            ddmv_spot, self.D, self.lam, self.dm_gain_m_per_v
        )

        # check four pairs (8 spots)
        self.assertTrue(len(NI_spot) == 8)
        # check spot radius
        self.assertTrue(
           np.max(np.abs(np.hypot(x_spot, y_spot) - self.radius_spot)) < pix_f)

        # check spot intensity
        self.assertTrue(np.min(NI_spot/self.NI_spot) > 0.95)
        self.assertTrue(np.max(NI_spot/self.NI_spot) < 1.01)

if __name__ == '__main__':
    unittest.main()
