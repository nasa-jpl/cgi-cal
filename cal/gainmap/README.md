# Purpose of *gainmap*

The *gainmap* module has two main functions:

- `compute_gainmap()` computes the gain of each actuator about the provided 2-D DM command map. The units of the gains depend on the units of the inputs.
- `compute_starting_commands_for_flight()` computes the DM commands at which to start EFC on orbit given an on-orbit flat map and a delta height (not to be confused with a delta surface map). The delta height map can either be computed in simulation (by differencing the DM actuator heights before and after running EFC) or from V&V data (by differencing the actuator heights computed from actuator maps of the ground flat setting and ground dark hole setting.)

This module uses two reference FITS files to compute gains at DM commands, heights at DM commands, or DM commands at heights:
1. a datacube of DM actuator heights for several DM command values (the DM command is assumed to be the same for all actuators in a given slice of the datacube)
2. a vector of DM command values corresponding to each slice of the datacube.

In the event that a datacube of DM gains with a corresponding vector of DM command values is given instead of #1 and #2 above for heights, `gm_util.compute_hcube_from_gcube()` can be used to compute #1 and #2. To compute accurately, this function requires an offset DM actuator height 2-D map and the corresponding offset DM command voltage. The offset map acts as a constant of integration used to integrate the DM gain (which is interpolated and extrapolated from the gain datacube) with respect to DM command voltage to find DM actuator height as a function of DM command voltage.  The height datacube is produced by integrating to the DM command values found in the DM command vector specified by the user.

Alternatively, one can use `compute_gainmap_from_gain_cube()` to use the DM gain datacube directly to compute the gain for a given 2-D DM command map via interpolation and extrapolation.  It is assumed that the DM command values lie within the allowed range of voltages for the DM actuators.

The vector of reference DM command values must be sorted in increasing value, and no two values can be the same. Otherwise, an exception will be raised because these properties are needed for the finite difference calculations of the gains. (Note: If the reference vector has to be sorted, then the associated height datacube slices must also be sorted the same way. That would all be done manually before calling the *gainmap* module.)

Because reference data is available at only a few DM commands, **cubic spline interpolation** is used to compute heights or gains from a given datacube for given DM commands. An exception is raised if a provided DM command is outside the DM command range for the reference height datacube since the module assumes the DM command range for height datacube corresponds to the full range of voltages allowed for the DM actuators.  The full voltage range, however, is not assumed to be spanned for any input gain datacube since the DM gain is not typically found at the boundaries of the range of allowed voltages.  Each actuator is independent and thus has a separate call to `scipy.interpolate.UnivariateSpline()`.
