# Purpose of DMREG

DMREG takes in phase retrieval outputs and computes the translation, clocking, and scale of the DM grid relative to the phase retrieval output array. The determination of parity is not part of DMREG.

# Usage

DM registration is now done with just one high-level function, **dmreg.calc_dmreg_from_poke_grid()**. (See the Section "Deprecated Usage" for a description of the old way.) Using the unpoked pupil amplitude as well as the difference between the poked pupil phase and unpoked pupil phase, it performs a bounded optimization to determine the translation, clocking, and scaling of the DM grid relative to the given array size.

There are many possible delta voltages that can be used for generating the delta phase pattern. However, it is recommended to use a sparse grid of isolated actuator pokes, e.g. just one out of each 4x4 sub-array in the DM grid, and to avoid actuating any near or behind a pupil obscuration. These standalone, high-spatial frequency features are easier to disambiguate from low-order phase variations in different phase retrievals, which are subtracted out (as the first 21 Zernikes by default) before performing the fit.


# DM1 vs DM2

All the above functions rely on comparing the expected phase and measured phase difference between phase retrievals. DM1 is conjugate to a pupil, so we expect a DM poke at DM1 to be purely in phase. DM2 is slightly out of pupil (1-meter away for a 46.3mm-diameter beam diameter) and therefore some of the phase change is converted to amplitude, but that amount is small enough that all the algorithms are still sufficiently accurate if the DM2 E-field change at the pupil is treated as though it were purely phase as well.


# Deprecated Usage

The original--but now deprecated--DM registration procedure was performed in several consecutive steps run in this order:
1. **dmreg.calc_software_mask**: First compute which pixels to use later when computing translation, rotation, and scale. Using AMPTHRESH alone is not sufficient because that doesn't take into account dead actuators. This function expects an actuated piston between the two different phase retrievals, and throws out pixels where the phase exceeds some specified variation from that piston value (either because there is a non-moving actuator there, or the phase estimate was unreliable there). There is also an option to erode (pad) the pupil obscurations to eliminate any bad phase estimates at the edges of the pupil obscurations; this also enlarges the masked areas for dead actuators.

2. **dmreg.calc_translation_clocking**: This function computes the translation of the DM grid, then the rotation, and then repeats both after initial values are found. The reasons that the translation and rotation fits are combined into one function are that 1) the same phase retrieval inputs work well for both and 2) the translation estimate improves once the clocking estimate is known, which requires iterating fits for both parameters.

  The recommended delta DM command for this function is a pair of uniformly actuated rows and a pair of uniformly actuated columns symmetric about the center of the DM. A simulation-tested example of this is in `Calibration/dmreg/cgidata/delta_DM_V_for_trans_rot_calib.fits`, but note that this command map should be adjusted to counteract non-uniform actuator gains. The symmetry is desired to make the DM shape insensitive to parity flips or 90-, 180-, and 270-degree rotations. The two actuated rows/columns are not neighboring because the Roman pupil struts would mostly block them along one dimension, and because the fitting seems to do better with actuated lines that are one (instead of two) actuators wide. The smoothness of the actuated lines seems to help as well--diagonal actuated lines are bumpier and give lower accuracy estimates.

  Other patterns may be used--no exception will be thrown--but the performance of this function has not been tested with them and cannot be guaranteed.

3. **dmreg.calc_scale**: After the translation and rotation are known, fit the x- and y-scales with this function.

  The recommended delta DM command for the scaling function is a fully-inscribed-within-the-pupil, two-actuator-wide outline of a square centered on the DM. A simulation-tested example of this is in `Calibration/dmreg/cgidata/delta_DM_V_for_scale_calib.fits`, but note that this command map should be adjusted to counteract non-uniform actuator gains. The symmetry is desired to make the DM shape insensitive to parity flips or 90-, 180-, and 270-degree rotations. The outline is two actuators wide (instead of just one) so that the algorithm will work over a greater range of scale factors.

  Other patterns may be used--no exception will be thrown--but the performance of this function has not been tested with them and cannot be guaranteed.
