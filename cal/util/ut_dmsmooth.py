# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Test suite for routine to alter a DM setting to obey voltage constraints
and neighbor rules
"""

import unittest

import numpy as np

from .dmsmooth import dmsmooth, VMIN
from .vdm_check import check_valid


# @unittest.skipUnless(__name__ == '__main__', 'Slow')
class TestDmSmooth(unittest.TestCase):
    """
    Predefine a few useful edge cases
    """
    def setUp(self):
        self.vmin = VMIN
        self.vmax = 100.
        self.vquant = 100./(2**16 - 1.)
        self.vneighbor = 30.
        self.vcorner = 30.
        self.nact = 48

        self.vmid = (self.vmin + self.vmax)/2.

        # Uniform
        self.dm0 = self.vmid*np.ones((self.nact, self.nact))

        # Checkerboard
        self.dmcheck = self.vmin*np.ones((self.nact, self.nact))
        self.dmcheck[::2, ::2] = self.vmax
        self.dmcheck[1::2, 1::2] = self.vmax

        # Up-down checkerboard
        self.dmud = self.vmid*np.ones((self.nact, self.nact))
        self.dmud[::2, ::2] = self.vmin
        self.dmud[1::2, 1::2] = self.vmax

        # Out-of-range checkerboard
        self.dmoor = (self.vmin-1.)*np.ones((self.nact, self.nact))
        self.dmoor[::2, ::2] = self.vmax+1.
        self.dmoor[1::2, 1::2] = self.vmax+1.

        # Horizonatl stripes
        self.dmhstripes = self.vmin*np.ones((self.nact, self.nact))
        self.dmhstripes[::2] = self.vmax

        # Vertical stripes
        self.dmvstripes = self.vmin*np.ones((self.nact, self.nact))
        self.dmvstripes[:, ::2] = self.vmax

        # Uniform and non-uniform flatmaps (which don't violate NR themselves)
        self.uniform_flat = self.vmid*np.ones((self.nact, self.nact))
        self.nonuniform_flat = self.vmid*np.ones((self.nact, self.nact)) + \
          np.min([self.vneighbor, self.vcorner])/10.*np.eye(self.nact)

        pass

    def test_success_flat(self):
        """
        Runs with no issues on a flat input DM, and returns the same output
        """
        dm0a = dmsmooth(dmin=self.dm0, vmax=self.vmax,
                        vquant=self.vquant,
                        vneighbor=self.vneighbor, vcorner=self.vcorner)
        self.assertTrue(check_valid(dm0a,
                                    plus_limit=self.vneighbor,
                                    diag_limit=self.vcorner,
                                    high_limit=self.vmax,
                                    low_limit=self.vmin))
        self.assertTrue((dm0a == self.dm0).all())
        pass

    def test_success_flat_with_uniform_flatmap(self):
        """
        Runs with no issues on a flat input DM, and returns the same output
        with a uniform flatmap
        """
        dm0a = dmsmooth(dmin=self.dm0, vmax=self.vmax,
                        vquant=self.vquant,
                        vneighbor=self.vneighbor, vcorner=self.vcorner,
                        dmflat=self.uniform_flat)
        self.assertTrue(check_valid(dm0a,
                                    plus_limit=self.vneighbor,
                                    diag_limit=self.vcorner,
                                    high_limit=self.vmax,
                                    low_limit=self.vmin,
                                    dmflat=self.uniform_flat))
        self.assertTrue((dm0a == self.dm0).all())
        pass

    def test_success_flat_with_nonuniform_flatmap(self):
        """
        Runs with no issues on a flat input DM, and returns the same output
        with a non-uniform flatmap which doesn't violate any neighbor rules
        """
        dm0a = dmsmooth(dmin=self.dm0, vmax=self.vmax,
                        vquant=self.vquant,
                        vneighbor=self.vneighbor, vcorner=self.vcorner,
                        dmflat=self.nonuniform_flat)
        self.assertTrue(check_valid(dm0a,
                                    plus_limit=self.vneighbor,
                                    diag_limit=self.vcorner,
                                    high_limit=self.vmax,
                                    low_limit=self.vmin,
                                    dmflat=self.nonuniform_flat))
        self.assertTrue((dm0a == self.dm0).all())
        pass


    def test_success_stress_fixed(self):
        """
        Runs with no failure for two stressing cases (either alternating
        between bounds or going upward and downward from a midpoint level)
        """
        dmc = dmsmooth(dmin=self.dmcheck, vmax=self.vmax,
                       vquant=self.vquant,
                       vneighbor=self.vneighbor, vcorner=self.vcorner)
        self.assertTrue(check_valid(dmc,
                                    plus_limit=self.vneighbor,
                                    diag_limit=self.vcorner,
                                    high_limit=self.vmax,
                                    low_limit=self.vmin))

        dmu = dmsmooth(dmin=self.dmud, vmax=self.vmax,
                       vquant=self.vquant,
                       vneighbor=self.vneighbor, vcorner=self.vcorner)
        self.assertTrue(check_valid(dmu,
                                    plus_limit=self.vneighbor,
                                    diag_limit=self.vcorner,
                                    high_limit=self.vmax,
                                    low_limit=self.vmin))
        pass

    def test_success_stress_fixed_with_nonuniform_flatmap(self):
        """
        Runs with no failure for two stressing cases (either alternating
        between bounds or going upward and downward from a midpoint level) in
        the presence of a nonuniform flatmap
        """
        dmc = dmsmooth(dmin=self.dmcheck, vmax=self.vmax,
                       vquant=self.vquant,
                       vneighbor=self.vneighbor, vcorner=self.vcorner,
                       dmflat=self.nonuniform_flat)
        self.assertTrue(check_valid(dmc,
                                    plus_limit=self.vneighbor,
                                    diag_limit=self.vcorner,
                                    high_limit=self.vmax,
                                    low_limit=self.vmin,
                                    dmflat=self.nonuniform_flat))

        dmu = dmsmooth(dmin=self.dmud, vmax=self.vmax,
                       vquant=self.vquant,
                       vneighbor=self.vneighbor, vcorner=self.vcorner,
                       dmflat=self.nonuniform_flat)
        self.assertTrue(check_valid(dmu,
                                    plus_limit=self.vneighbor,
                                    diag_limit=self.vcorner,
                                    high_limit=self.vmax,
                                    low_limit=self.vmin,
                                    dmflat=self.nonuniform_flat))
        pass


    # @unittest.skipUnless(__name__ == '__main__', 'Very slow (~30 sec)')
    def test_success_stress_fuzz(self):
        """
        Runs with no failure for a set of stressing cases made from uniformly
        distributing actuator heights between vmin and vmax

        Checks both that operation succeeds and that output is valid

        Seed is fixed, so these are reproducible; intent is write explicit
        checks for the edge cases we can think of and use a randomly-derived
        set to see if we come across anything we didn't think of...
        """

        # Fuzz
        np.random.seed(3621)
        nfuzz = 200
        dmfuzz = self.vmin + (self.vmax-self.vmin)*np.random.rand(nfuzz, \
                                            self.nact, self.nact)

        for n in range(dmfuzz.shape[0]):
            dmf = dmfuzz[n]
            dmout = dmsmooth(dmin=dmf, vmax=self.vmax,
                             vquant=self.vquant,
                             vneighbor=self.vneighbor, vcorner=self.vcorner)
            self.assertTrue(check_valid(dmout,
                                        plus_limit=self.vneighbor,
                                        diag_limit=self.vcorner,
                                        high_limit=self.vmax,
                                        low_limit=self.vmin))
            pass
        pass


    def test_success_stress_fuzz_with_nonuniform_flat(self):
        """
        Runs with no failure for a set of stressing cases made from uniformly
        distributing actuator heights between vmin and vmax

        Checks both that operation succeeds and that output is valid

        Seed is fixed, so these are reproducible; intent is write explicit
        checks for the edge cases we can think of and use a randomly-derived
        set to see if we come across anything we didn't think of...
        """

        # Fuzz
        np.random.seed(5599)
        nfuzz = 200
        dmfuzz = self.vmin + (self.vmax-self.vmin)*np.random.rand(nfuzz, \
                                            self.nact, self.nact)

        for n in range(dmfuzz.shape[0]):
            dmf = dmfuzz[n]
            dmout = dmsmooth(dmin=dmf, vmax=self.vmax,
                             vquant=self.vquant,
                             vneighbor=self.vneighbor, vcorner=self.vcorner,
                             dmflat=self.nonuniform_flat)
            self.assertTrue(check_valid(dmout,
                                        plus_limit=self.vneighbor,
                                        diag_limit=self.vcorner,
                                        high_limit=self.vmax,
                                        low_limit=self.vmin,
                                        dmflat=self.nonuniform_flat))
            pass
        pass


    ## Comment this out for now, as we are disallowing vquant=0.  Leave it here
    ## though so we have it if there is a future refactor of the solver to
    ## enable the vquant=0 case.
    # def test_success_stress_fixed_noquant(self):
    #     """
    #     Runs with no failure for two stressing cases (either alternating
    #     between bounds or going upward and downward from a midpoint level)
    #     when vquant goes to zero
    #     """
    #     dmc = dmsmooth(dmin=self.dmcheck, vmax=self.vmax,
    #                    vquant=0,
    #                    vneighbor=self.vneighbor, vcorner=self.vcorner)
    #     self.assertTrue(check_valid(dmc,
    #                                 plus_limit=self.vneighbor,
    #                                 diag_limit=self.vcorner,
    #                                 high_limit=self.vmax,
    #                                 low_limit=self.vmin))

    #     dmu = dmsmooth(dmin=self.dmud, vmax=self.vmax,
    #                    vquant=0,
    #                    vneighbor=self.vneighbor, vcorner=self.vcorner)
    #     self.assertTrue(check_valid(dmu,
    #                                 plus_limit=self.vneighbor,
    #                                 diag_limit=self.vcorner,
    #                                 high_limit=self.vmax,
    #                                 low_limit=self.vmin))
    #     pass


    def test_idempotence_fixed(self):
        """
        For fixed stressing cases, the output from the first run,
        when fed as input to a second run, produces the same result
        """
        dmc = dmsmooth(dmin=self.dmcheck, vmax=self.vmax,
                       vquant=self.vquant,
                       vneighbor=self.vneighbor, vcorner=self.vcorner)
        dmc2 = dmsmooth(dmin=dmc, vmax=self.vmax,
                        vquant=self.vquant,
                        vneighbor=self.vneighbor, vcorner=self.vcorner)
        self.assertTrue((dmc == dmc2).all())

        dmu = dmsmooth(dmin=self.dmud, vmax=self.vmax,
                       vquant=self.vquant,
                       vneighbor=self.vneighbor, vcorner=self.vcorner)
        dmu2 = dmsmooth(dmin=dmu, vmax=self.vmax,
                        vquant=self.vquant,
                        vneighbor=self.vneighbor, vcorner=self.vcorner)
        self.assertTrue((dmu == dmu2).all())
        pass


    def test_idempotence_fixed_with_nonuniform_flat(self):
        """
        For fixed stressing cases, the output from the first run,
        when fed as input to a second run, produces the same result
        """
        dmc = dmsmooth(dmin=self.dmcheck, vmax=self.vmax,
                       vquant=self.vquant,
                       vneighbor=self.vneighbor, vcorner=self.vcorner,
                       dmflat=self.nonuniform_flat)
        dmc2 = dmsmooth(dmin=dmc, vmax=self.vmax,
                        vquant=self.vquant,
                        vneighbor=self.vneighbor, vcorner=self.vcorner,
                        dmflat=self.nonuniform_flat)
        self.assertTrue((dmc == dmc2).all())

        dmu = dmsmooth(dmin=self.dmud, vmax=self.vmax,
                       vquant=self.vquant,
                       vneighbor=self.vneighbor, vcorner=self.vcorner,
                       dmflat=self.nonuniform_flat)
        dmu2 = dmsmooth(dmin=dmu, vmax=self.vmax,
                        vquant=self.vquant,
                        vneighbor=self.vneighbor, vcorner=self.vcorner,
                        dmflat=self.nonuniform_flat)
        self.assertTrue((dmu == dmu2).all())
        pass


    def test_idempotence_fuzz(self):
        """
        For randomly-derived stressing cases, the output from the first run,
        when fed as input to a second run, produces the same result
        """
        # Fuzz
        np.random.seed(3621)
        nfuzz = 200
        dmfuzz = self.vmin + (self.vmax-self.vmin)*np.random.rand(nfuzz, \
                                            self.nact, self.nact)

        for n in range(dmfuzz.shape[0]):
            dmf = dmfuzz[n]
            dmout = dmsmooth(dmin=dmf, vmax=self.vmax,
                             vquant=self.vquant,
                             vneighbor=self.vneighbor, vcorner=self.vcorner)
            dmout2 = dmsmooth(dmin=dmout, vmax=self.vmax,
                              vquant=self.vquant,
                              vneighbor=self.vneighbor, vcorner=self.vcorner)
            self.assertTrue((dmout == dmout2).all())
            pass
        pass

    def test_margin_fixed(self):
        """
        Verify that all outputs are at least 2 LSB from their min/max, and all
        all NR gaps are at least 2 LSB beyond the accompanying rule.
        """
        margin = 2.*self.vquant

        dmc = dmsmooth(dmin=self.dmcheck, vmax=self.vmax,
                       vquant=self.vquant,
                       vneighbor=self.vneighbor, vcorner=self.vcorner)
        self.assertTrue(np.max(dmc) < self.vmax-margin)
        self.assertTrue(np.min(dmc) > self.vmin+margin)
        self.assertTrue(np.max(np.abs(dmc[1:, :] - dmc[:-1, :])) <
                        self.vneighbor-margin)
        self.assertTrue(np.max(np.abs(dmc[:, 1:] - dmc[:, :-1])) <
                        self.vneighbor-margin)
        self.assertTrue(np.max(np.abs(dmc[1:, 1:] - dmc[:-1, :-1])) <
                        self.vcorner-margin)
        self.assertTrue(np.max(np.abs(dmc[1:, :-1] - dmc[:-1, 1:])) <
                        self.vcorner-margin)

        dmc = dmsmooth(dmin=self.dmud, vmax=self.vmax,
                       vquant=self.vquant,
                       vneighbor=self.vneighbor, vcorner=self.vcorner)
        self.assertTrue(np.max(dmc) < self.vmax-margin)
        self.assertTrue(np.min(dmc) > self.vmin+margin)
        self.assertTrue(np.max(np.abs(dmc[1:, :] - dmc[:-1, :])) <
                        self.vneighbor-margin)
        self.assertTrue(np.max(np.abs(dmc[:, 1:] - dmc[:, :-1])) <
                        self.vneighbor-margin)
        self.assertTrue(np.max(np.abs(dmc[1:, 1:] - dmc[:-1, :-1])) <
                        self.vcorner-margin)
        self.assertTrue(np.max(np.abs(dmc[1:, :-1] - dmc[:-1, 1:])) <
                        self.vcorner-margin)

        dmc = dmsmooth(dmin=self.dmoor, vmax=self.vmax,
                       vquant=self.vquant,
                       vneighbor=self.vneighbor, vcorner=self.vcorner)
        self.assertTrue(np.max(dmc) < self.vmax-margin)
        self.assertTrue(np.min(dmc) > self.vmin+margin)
        self.assertTrue(np.max(np.abs(dmc[1:, :] - dmc[:-1, :])) <
                        self.vneighbor-margin)
        self.assertTrue(np.max(np.abs(dmc[:, 1:] - dmc[:, :-1])) <
                        self.vneighbor-margin)
        self.assertTrue(np.max(np.abs(dmc[1:, 1:] - dmc[:-1, :-1])) <
                        self.vcorner-margin)
        self.assertTrue(np.max(np.abs(dmc[1:, :-1] - dmc[:-1, 1:])) <
                        self.vcorner-margin)

        dmc = dmsmooth(dmin=self.dmvstripes, vmax=self.vmax,
                       vquant=self.vquant,
                       vneighbor=self.vneighbor, vcorner=self.vcorner)
        self.assertTrue(np.max(dmc) < self.vmax-margin)
        self.assertTrue(np.min(dmc) > self.vmin+margin)
        self.assertTrue(np.max(np.abs(dmc[1:, :] - dmc[:-1, :])) <
                        self.vneighbor-margin)
        self.assertTrue(np.max(np.abs(dmc[:, 1:] - dmc[:, :-1])) <
                        self.vneighbor-margin)
        self.assertTrue(np.max(np.abs(dmc[1:, 1:] - dmc[:-1, :-1])) <
                        self.vcorner-margin)
        self.assertTrue(np.max(np.abs(dmc[1:, :-1] - dmc[:-1, 1:])) <
                        self.vcorner-margin)

        dmc = dmsmooth(dmin=self.dmhstripes, vmax=self.vmax,
                       vquant=self.vquant,
                       vneighbor=self.vneighbor, vcorner=self.vcorner)
        self.assertTrue(np.max(dmc) < self.vmax-margin)
        self.assertTrue(np.min(dmc) > self.vmin+margin)
        self.assertTrue(np.max(np.abs(dmc[1:, :] - dmc[:-1, :])) <
                        self.vneighbor-margin)
        self.assertTrue(np.max(np.abs(dmc[:, 1:] - dmc[:, :-1])) <
                        self.vneighbor-margin)
        self.assertTrue(np.max(np.abs(dmc[1:, 1:] - dmc[:-1, :-1])) <
                        self.vcorner-margin)
        self.assertTrue(np.max(np.abs(dmc[1:, :-1] - dmc[:-1, 1:])) <
                        self.vcorner-margin)

        pass

    def test_margin_fuzz(self):
        """
        Verify that all outputs are at least 2 LSB from their min/max, and all
        all NR gaps are at least 2 LSB beyond the accompanying rule.
        """
        margin = 2.*self.vquant

        # Fuzz
        np.random.seed(3621)
        nfuzz = 200
        dmfuzz = self.vmin + (self.vmax-self.vmin)*np.random.rand(nfuzz, \
                                            self.nact, self.nact)

        for n in range(dmfuzz.shape[0]):
            dmf = dmfuzz[n]
            dmc = dmsmooth(dmin=dmf, vmax=self.vmax,
                           vquant=self.vquant,
                           vneighbor=self.vneighbor, vcorner=self.vcorner)
            self.assertTrue(np.max(dmc) < self.vmax-margin)
            self.assertTrue(np.min(dmc) > self.vmin+margin)
            self.assertTrue(np.max(np.abs(dmc[1:, :] - dmc[:-1, :])) <
                            self.vneighbor-margin)
            self.assertTrue(np.max(np.abs(dmc[:, 1:] - dmc[:, :-1])) <
                            self.vneighbor-margin)
            self.assertTrue(np.max(np.abs(dmc[1:, 1:] - dmc[:-1, :-1])) <
                            self.vcorner-margin)
            self.assertTrue(np.max(np.abs(dmc[1:, :-1] - dmc[:-1, 1:])) <
                            self.vcorner-margin)
            pass
        pass



    def test_vneighbor_vcorner_differ(self):
        """
        Check checkerboard returns successfully if the neighbor rules are
        different between diagonal and lateral
        """
        dmc = dmsmooth(dmin=self.dmcheck, vmax=self.vmax,
                       vquant=self.vquant,
                       vneighbor=self.vneighbor, vcorner=self.vneighbor+1)
        self.assertTrue(check_valid(dmc,
                                    plus_limit=self.vneighbor,
                                    diag_limit=self.vneighbor+1,
                                    high_limit=self.vmax,
                                    low_limit=self.vmin))

        dmc = dmsmooth(dmin=self.dmcheck, vmax=self.vmax,
                       vquant=self.vquant,
                       vneighbor=self.vcorner+1, vcorner=self.vcorner)
        self.assertTrue(check_valid(dmc,
                                    plus_limit=self.vcorner+1,
                                    diag_limit=self.vcorner,
                                    high_limit=self.vmax,
                                    low_limit=self.vmin))
        pass

    def test_nonr(self):
        """
        Check checkerboard returns successfully if the neighbor rules are
        all larger than the valid voltage range (i.e. disabled)
        """
        rangeplus = self.vmax - self.vmin + 1.
        dmc = dmsmooth(dmin=self.dmcheck, vmax=self.vmax,
                       vquant=self.vquant,
                       vneighbor=rangeplus, vcorner=rangeplus)
        self.assertTrue(check_valid(dmc,
                                    plus_limit=rangeplus,
                                    diag_limit=rangeplus,
                                    high_limit=self.vmax,
                                    low_limit=self.vmin))
        pass


    def test_cap_high(self):
        """Verify high voltage is capped as expected"""
        dmhigh = (self.vmax+1.)*np.ones((self.nact, self.nact))
        dmcap = (self.vmax)*np.ones((self.nact, self.nact))

        dmout = dmsmooth(dmin=dmhigh, vmax=self.vmax,
                         vquant=self.vquant,
                         vneighbor=self.vneighbor, vcorner=self.vcorner)
        self.assertTrue(check_valid(dmout,
                                    plus_limit=self.vneighbor,
                                    diag_limit=self.vcorner,
                                    high_limit=self.vmax,
                                    low_limit=self.vmin))
        self.assertTrue((dmout <= dmcap).all())
        pass

    def test_cap_low(self):
        """Verify low voltage is capped as expected"""
        dmlow = (self.vmin-1.)*np.ones((self.nact, self.nact))
        dmcap = (self.vmin)*np.ones((self.nact, self.nact))

        dmout = dmsmooth(dmin=dmlow, vmax=self.vmax,
                         vquant=self.vquant,
                         vneighbor=self.vneighbor, vcorner=self.vcorner)
        self.assertTrue(check_valid(dmout,
                                    plus_limit=self.vneighbor,
                                    diag_limit=self.vcorner,
                                    high_limit=self.vmax,
                                    low_limit=self.vmin))
        self.assertTrue((dmout >= dmcap).all())
        pass

    def test_oor_checker(self):
        """
        Check checkerboard that extends out-of-range completes successfully
        """
        dmc = dmsmooth(dmin=self.dmoor, vmax=self.vmax,
                       vquant=self.vquant,
                       vneighbor=self.vneighbor, vcorner=self.vcorner)
        self.assertTrue(check_valid(dmc,
                                    plus_limit=self.vneighbor,
                                    diag_limit=self.vcorner,
                                    high_limit=self.vmax,
                                    low_limit=self.vmin))
        pass



    # Input checking
    def test_bad_dmin(self):
        """Checks bad input fails correctly"""
        for perr in [np.ones((3, 3, 3)), None, 'txt', 1j, 5]:
            with self.assertRaises(TypeError):
                dmsmooth(dmin=perr, vmax=self.vmax,
                         vquant=self.vquant,
                         vneighbor=self.vneighbor, vcorner=self.vcorner)
                pass
            pass
        pass


    def test_bad_vmax_type(self):
        """Checks bad input fails correctly"""
        for perr in [np.ones((3, 3, 3)), None, 'txt', 1j, np.eye(3)]:
            with self.assertRaises(TypeError):
                dmsmooth(dmin=self.dm0, vmax=perr,
                         vquant=self.vquant,
                         vneighbor=self.vneighbor, vcorner=self.vcorner)
                pass
            pass
        pass

    def test_bad_vmax_value(self):
        """Checks bad input fails correctly"""
        for perr in [VMIN, VMIN-1]:
            with self.assertRaises(ValueError):
                dmsmooth(dmin=self.dm0, vmax=perr,
                         vquant=self.vquant,
                         vneighbor=self.vneighbor, vcorner=self.vcorner)
                pass
            pass
        pass


    def test_bad_vquant(self):
        """Checks bad input fails correctly"""
        for perr in [np.ones((3, 3, 3)), None, 'txt', 1j, np.eye(3), -1]:
            with self.assertRaises(TypeError):
                dmsmooth(dmin=self.dm0, vmax=self.vmax,
                         vquant=perr,
                         vneighbor=self.vneighbor, vcorner=self.vcorner)
                pass
            pass
        pass


    def test_bad_vneighbor(self):
        """Checks bad input fails correctly"""
        for perr in [np.ones((3, 3, 3)), None, 'txt', 1j, np.eye(3), -1]:
            with self.assertRaises(TypeError):
                dmsmooth(dmin=self.dm0, vmax=self.vmax,
                         vquant=self.vquant,
                         vneighbor=perr, vcorner=self.vcorner)
                pass
            pass
        pass


    def test_bad_vcorner(self):
        """Checks bad input fails correctly"""
        for perr in [np.ones((3, 3, 3)), None, 'txt', 1j, np.eye(3), -1]:
            with self.assertRaises(TypeError):
                dmsmooth(dmin=self.dm0, vmax=self.vmax,
                         vquant=self.vquant,
                         vneighbor=self.vneighbor, vcorner=perr)
                pass
            pass
        pass

    def test_bad_dmflat(self):
        """Checks bad input fails correctly"""
        for perr in [np.ones((3, 3, 3)), 'txt', 1j, 5]:
            with self.assertRaises(TypeError):
                dmsmooth(dmin=self.dm0, vmax=self.vmax,
                         vquant=self.vquant,
                         vneighbor=self.vneighbor, vcorner=self.vcorner,
                         dmflat=perr)
                pass
            pass
        pass


    def test_fail_nonsquare_dmin(self):
        """Expect input DM to be square"""
        with self.assertRaises(TypeError):
            dmsmooth(dmin=self.dm0[:, :-1], vmax=self.vmax,
                     vquant=self.vquant,
                     vneighbor=self.vneighbor, vcorner=self.vcorner)
            pass
        pass

    def test_fail_not_samesize_dmflat(self):
        """Expect dmflat to be the same size as input DM"""
        with self.assertRaises(TypeError):
            dmsmooth(dmin=self.dm0, vmax=self.vmax,
                     vquant=self.vquant,
                     vneighbor=self.vneighbor, vcorner=self.vcorner,
                     dmflat=self.uniform_flat[:, :-1])
            pass
        pass

    def test_dmflat_greater_than_vmax(self):
        """Expect dmflat to be <= vmax"""
        with self.assertRaises(ValueError):
            dmsmooth(dmin=self.dm0, vmax=self.vmax,
                     vquant=self.vquant,
                     vneighbor=self.vneighbor, vcorner=self.vcorner,
                     dmflat=np.ones_like(self.dm0)*(self.vmax + 1))
            pass
        pass


    def test_dmflat_less_than_vmin(self):
        """Expect dmflat to be >= vmin"""
        with self.assertRaises(ValueError):
            dmsmooth(dmin=self.dm0, vmax=self.vmax,
                     vquant=self.vquant,
                     vneighbor=self.vneighbor, vcorner=self.vcorner,
                     dmflat=np.ones_like(self.dm0)*(self.vmin - 1))
            pass
        pass



    # Output checks
    #
    # Verifies a particular discrepancy is corrected in the output array.  Does
    # not guarantee anything about the values in the output array.
    def test_horizontal_noquant(self):
        """Verify result for horizontal NR violations"""
        dmnr = np.array([[20, 20, 20], [20, 40, 0], [20, 20, 20]])
        dmout = dmsmooth(
            dmin=dmnr,
            vmax=self.vmax,
            vquant=self.vquant,
            vneighbor=self.vneighbor,
            vcorner=self.vcorner,
        )
        isgood = check_valid(
            array=dmout,
            plus_limit=self.vneighbor,
            diag_limit=self.vcorner,
            high_limit=self.vmax,
            low_limit=self.vmin,
            dmflat=None,
        )

        self.assertTrue(isgood)
        pass


    def test_vertical(self):
        """Verify result for vertical NR violations"""
        dmnr = np.array([[20, 20, 20], [20, 40, 20], [20, 0, 20]])
        dmout = dmsmooth(
            dmin=dmnr,
            vmax=self.vmax,
            vquant=self.vquant,
            vneighbor=self.vneighbor,
            vcorner=self.vcorner,
        )
        isgood = check_valid(
            array=dmout,
            plus_limit=self.vneighbor,
            diag_limit=self.vcorner,
            high_limit=self.vmax,
            low_limit=self.vmin,
            dmflat=None,
        )

        self.assertTrue(isgood)
        pass


    def test_diag1(self):
        """Verify result for 1st diag NR violations"""
        dmnr = np.array([[20, 20, 20], [20, 40, 20], [20, 20, 0]])
        dmout = dmsmooth(
            dmin=dmnr,
            vmax=self.vmax,
            vquant=self.vquant,
            vneighbor=self.vneighbor,
            vcorner=self.vcorner,
        )
        isgood = check_valid(
            array=dmout,
            plus_limit=self.vneighbor,
            diag_limit=self.vcorner,
            high_limit=self.vmax,
            low_limit=self.vmin,
            dmflat=None,
        )

        self.assertTrue(isgood)
        pass


    def test_diag2(self):
        """Verify result for 2nd diag NR violations"""
        dmnr = np.array([[20, 20, 20], [20, 40, 20], [0, 20, 20]])
        dmout = dmsmooth(
            dmin=dmnr,
            vmax=self.vmax,
            vquant=self.vquant,
            vneighbor=self.vneighbor,
            vcorner=self.vcorner,
        )
        isgood = check_valid(
            array=dmout,
            plus_limit=self.vneighbor,
            diag_limit=self.vcorner,
            high_limit=self.vmax,
            low_limit=self.vmin,
            dmflat=None,
        )

        self.assertTrue(isgood)
        pass


    def test_multiNR_noquant(self):
        """
        Verify result when more than one NR violation is present
        """
        dmnr = np.array([[40, 0, 20], [20, 20, 0], [20, 40, 20]])
        dmout = dmsmooth(
            dmin=dmnr,
            vmax=self.vmax,
            vquant=self.vquant,
            vneighbor=self.vneighbor,
            vcorner=self.vcorner,
        )
        isgood = check_valid(
            array=dmout,
            plus_limit=self.vneighbor,
            diag_limit=self.vcorner,
            high_limit=self.vmax,
            low_limit=self.vmin,
            dmflat=None,
        )

        self.assertTrue(isgood)
        pass


    def test_check_odd(self):
        """
        Verify result is correct for checkerboard pattern (nside odd)
        """

        dmnr = np.array(
            [
                [0, 100, 0],
                [100, 0, 100],
                [0, 100, 0],
            ]
        )
        dmout = dmsmooth(
            dmin=dmnr,
            vmax=self.vmax,
            vquant=self.vquant,
            vneighbor=self.vneighbor,
            vcorner=self.vcorner,
        )
        isgood = check_valid(
            array=dmout,
            plus_limit=self.vneighbor,
            diag_limit=self.vcorner,
            high_limit=self.vmax,
            low_limit=self.vmin,
            dmflat=None,
        )

        self.assertTrue(isgood)
        pass


    def test_check_even(self):
        """
        Verify result is correct for checkerboard pattern (nside even)
        """

        dmnr = np.array(
            [
                [0, 100, 0, 100],
                [100, 0, 100, 0],
                [0, 100, 0, 100],
                [100, 0, 100, 0],
            ]
        )
        dmout = dmsmooth(
            dmin=dmnr,
            vmax=self.vmax,
            vquant=self.vquant,
            vneighbor=self.vneighbor,
            vcorner=self.vcorner,
        )
        isgood = check_valid(
            array=dmout,
            plus_limit=self.vneighbor,
            diag_limit=self.vcorner,
            high_limit=self.vmax,
            low_limit=self.vmin,
            dmflat=None,
        )

        self.assertTrue(isgood)
        pass


    def test_hstripe_odd(self):
        """
        Verify result is correct for horizontal stripe pattern (nside odd)
        """

        dmnr = np.array(
            [
                [0, 0, 0],
                [100, 100, 100],
                [0, 0, 0],
            ]
        )
        dmout = dmsmooth(
            dmin=dmnr,
            vmax=self.vmax,
            vquant=self.vquant,
            vneighbor=self.vneighbor,
            vcorner=self.vcorner,
        )
        isgood = check_valid(
            array=dmout,
            plus_limit=self.vneighbor,
            diag_limit=self.vcorner,
            high_limit=self.vmax,
            low_limit=self.vmin,
            dmflat=None,
        )

        self.assertTrue(isgood)
        pass


    def test_hstripe_even(self):
        """
        Verify result is correct for horizonatal stripe pattern (nside even)
        """

        dmnr = np.array(
            [
                [0, 0, 0, 0],
                [100, 100, 100, 100],
                [0, 0, 0, 0],
                [100, 100, 100, 100],
            ]
        )
        dmout = dmsmooth(
            dmin=dmnr,
            vmax=self.vmax,
            vquant=self.vquant,
            vneighbor=self.vneighbor,
            vcorner=self.vcorner,
        )
        isgood = check_valid(
            array=dmout,
            plus_limit=self.vneighbor,
            diag_limit=self.vcorner,
            high_limit=self.vmax,
            low_limit=self.vmin,
            dmflat=None,
        )

        self.assertTrue(isgood)
        pass


    def test_vstripe_odd(self):
        """
        Verify result is correct for vertical stripe pattern (nside odd)
        """
        dmnr = np.array(
            [
                [0, 100, 0],
                [0, 100, 0],
                [0, 100, 0],
            ]
        )
        dmout = dmsmooth(
            dmin=dmnr,
            vmax=self.vmax,
            vquant=self.vquant,
            vneighbor=self.vneighbor,
            vcorner=self.vcorner,
        )
        isgood = check_valid(
            array=dmout,
            plus_limit=self.vneighbor,
            diag_limit=self.vcorner,
            high_limit=self.vmax,
            low_limit=self.vmin,
            dmflat=None,
        )

        self.assertTrue(isgood)
        pass


    def test_vstripe_even(self):
        """
        Verify result is correct for vertical stripe pattern (nside even)
        """

        dmnr = np.array(
            [
                [0, 100, 0, 100],
                [0, 100, 0, 100],
                [0, 100, 0, 100],
                [0, 100, 0, 100],
            ]
        )
        dmout = dmsmooth(
            dmin=dmnr,
            vmax=self.vmax,
            vquant=self.vquant,
            vneighbor=self.vneighbor,
            vcorner=self.vcorner,
        )
        isgood = check_valid(
            array=dmout,
            plus_limit=self.vneighbor,
            diag_limit=self.vcorner,
            high_limit=self.vmax,
            low_limit=self.vmin,
            dmflat=None,
        )

        self.assertTrue(isgood)
        pass

    def test_vcorner_less_vneighbor(self):
        """
        Verify result is correct when vcorner is less than vneighbor
        """
        vneighbor = 30.
        vcorner = 25.

        dmnr = np.array(
            [
                [20, 40, 0],
                [30, 20, 20],
                [20, 0, 20],
            ]
        )
        dmout = dmsmooth(
            dmin=dmnr,
            vmax=self.vmax,
            vquant=self.vquant,
            vneighbor=vneighbor,
            vcorner=vcorner,
        )
        isgood = check_valid(
            array=dmout,
            plus_limit=vneighbor,
            diag_limit=vcorner,
            high_limit=self.vmax,
            low_limit=self.vmin,
            dmflat=None,
        )

        self.assertTrue(isgood)
        pass


    def test_vcorner_greater_vneighbor(self):
        """
        Verify result is correct when vcorner is greater than vneighbor
        """
        vneighbor = 25.
        vcorner = 30.

        dmnr = np.array(
            [
                [20, 30, 0],
                [40, 20, 20],
                [20, 0, 20],
            ]
        )
        dmout = dmsmooth(
            dmin=dmnr,
            vmax=self.vmax,
            vquant=self.vquant,
            vneighbor=vneighbor,
            vcorner=vcorner,
        )
        isgood = check_valid(
            array=dmout,
            plus_limit=vneighbor,
            diag_limit=vcorner,
            high_limit=self.vmax,
            low_limit=self.vmin,
            dmflat=None,
        )

        self.assertTrue(isgood)
        pass


    def test_nonuniform_flatmap(self):
        """
        Verify result is correct with nonuniform flatmap
        """
        vneighbor = 50.
        vcorner = 50.

        dmnr = np.array(
            [
                [100, 0, 100],
                [0, 100, 0],
                [100, 0, 100],
            ]
        )

        dmflat = np.array(
            [
                [48, 0, 48],
                [0, 48, 0],
                [48, 0, 48],
            ]
        )


        dmout = dmsmooth(
            dmin=dmnr,
            vmax=self.vmax,
            vquant=self.vquant,
            vneighbor=vneighbor,
            vcorner=vcorner,
            dmflat=dmflat,
        )
        isgood = check_valid(
            array=dmout,
            plus_limit=vneighbor,
            diag_limit=vcorner,
            high_limit=self.vmax,
            low_limit=self.vmin,
            dmflat=dmflat,
        )

        self.assertTrue(isgood)
        pass



if __name__ == '__main__':
    unittest.main()
