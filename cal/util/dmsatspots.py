# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Generate relative DM Volts array to produce a set of satellite spots
around the star PSF
"""
import warnings
import numbers
import numpy as np

from cal.util.dmshapes import sine
from . import check

def ddmv_satspots(radius_spot, theta_spot, NI_spot, sine_phase,
                  Dact, Nact, wavelength, dm_gain_m_per_v, x_aoi=0, y_aoi=0):
    """
    Create a DM voltage array (delta or relative voltage) that generates pairs
    of spots in the focal plane. Each pair is a spot at the given radius,
    theta, and normalized intensity, and a second spot 180 degress around the
    central PSF.

    NOTE: the DM command produced by this function does not have voltage caps,
    neighbor-rule compliance, or tied and dead actuators enforced.  Use the
    constrain_dm() function on the output if you wish to do so.  Note that
    while tied and dead actuators have a meaningful application on a
    relative DM setting, voltage caps and neighbor rules do not, and care
    should be taken if constraining not to over-enforce constraints.

    Arguments - Spot Parameters:
      radius_spot: radial distance from the PSF peak (lam/D), scalar or 1-d
       array, real values > 0.  Note that values of radius_spot < 6 L/D will
       raise a warning; spots < 6 L/D in radius may be too close and may
       negatively impact the spots' location estimation
      theta_spot: azimuthal position, CCW from the x-axis (degrees), scalar or
       1-d array
      NI_spot: intensity of spots, normalized to the PSF peak, scalar or 1-d
       array, real values > 0.  Note that values of NI_spot < 1e-7 will raise a
       warning; the algorithm may not succeed for spots which are too faint if
       a dark hole is not present.
      sine_phase: phase angle offset for the sinusoids on the DM array.
       (degrees) scalar or 1-d array

    Arguments - System Parameters:
      Dact: diameter of the pupil in DM actuators, i.e. "D" for lam/D
      Nact: number of actuators along one side of the DM (assumes square DM)
      wavelength: operating wavelength (m), for converting wavefront to DM
       surface height
      dm_gain_m_per_v: DM actuator gain (m/V), array size (Nact, Nact)

    Keyword Arguments:
      x_aoi: angle of incidence along the x-axis of the output array, not
       necessarily the optical x-axis.  Used to compensate for non-normal DM
       incidence.  Floating point number, in degrees, >= 0.  0 is the default
       and = normal incidence on that axis.
      y_aoi: angle of incidence along the y-axis of the output array, not
       necessarily the optical y-axis.  Used to compensate for non-normal DM
       incidence.  Floating point number, in degrees, >= 0.  0 is the default
       and = normal incidence on that axis.

    Returns:
      ddmv = array of relative DM voltage (V), size (Nact, Nact)

    The return ddmv creates one pair of spots if all the spot parameters are
    scalar.  If any of the spot parameters are 1-d arrays, size (nspots,), then
    the return ddmv generates nspots pairs of spots.

    The general form of ddmv to generate a pair of spots is:
       ddmv ~ A sin( 2pi radius (cos(theta*pi/180) xact/D
                                 + sin(theta*pi/180) yact/D) + sine_phase)

    """

    # check inputs
    # check spot parameters, first force spot parameters to numpy arrays
    # future refactor note: use np.atleast_1d() here
    radius_spot = np.array(radius_spot)
    theta_spot = np.array(theta_spot)
    NI_spot = np.array(NI_spot)
    sine_phase = np.array(sine_phase)

    # if spot parameters are a mix of scalars and arrays, and all the arrays
    # are the same size, this simple math operation will be ok, otherwise,
    # throw error:
    try:
        dum_array = radius_spot * theta_spot * NI_spot * sine_phase

    except:
        raise TypeError('spot parameters must be scalar or arrays of ' +
                        'same shape')

    # if all parameters are scalars, then dum_array is a scalar
    if isinstance(dum_array, numbers.Number):
        dum_array = dum_array*np.ones((1,))

    # all spot parameters must be real, then dum_array must be real
    check.real_array(dum_array, 'spot parameters', TypeError)

    # make each parameter 1-d array of same size
    radius_spot = radius_spot * np.ones(dum_array.shape)
    theta_spot = theta_spot * np.ones(dum_array.shape)
    NI_spot = NI_spot * np.ones(dum_array.shape)
    sine_phase = sine_phase * np.ones(dum_array.shape)

    # all spot parameters are real arrays, some must be positive
    if not np.all(radius_spot > 0.0):
        raise TypeError('radius_spot must be positive')
    if not np.all(NI_spot > 0.0):
        raise TypeError('NI_spot must be positive')

    # Warn about numerical ranges where the algorithm may or may not work
    if np.any(radius_spot < 6.0):
        warnings.warn('All values of radius_spot should be >= 6.0 lambda/D ' +
                      'or the spots may be too close for location estimation')
    if np.any(NI_spot < 1.0e-7):
        warnings.warn('All values of NI_spot should be >= 1e-7 or the spots ' +
                      'may be too faint for the algorithm to succeed ' +
                      'if a dark hole is not present')

    # check System Parameters
    # Dact, Nact, wavelength, dm_gain_m_per_v):
    check.real_positive_scalar(Dact, 'Dact', TypeError)
    check.positive_scalar_integer(Nact, 'Nact', TypeError)
    check.real_positive_scalar(wavelength, 'wavelength', TypeError)
    check.twoD_square_array(dm_gain_m_per_v, 'dm_gain_m_per_v', TypeError)
    check.real_array(dm_gain_m_per_v, 'dm_gain_m_per_v', TypeError)
    dm_gain_m_per_v = np.array(dm_gain_m_per_v)
    if dm_gain_m_per_v.shape != (Nact, Nact):
        raise TypeError('dm_gain_m_per_v.shape must be (Nact,Nact)')

    check.real_nonnegative_scalar(x_aoi, 'x_aoi', TypeError)
    check.real_nonnegative_scalar(y_aoi, 'y_aoi', TypeError)

    # ddmv is sum of sinusoids, one for each pair of spots
    ddmv = np.zeros((Nact, Nact))
    for r, t, NI, phi in zip(radius_spot, theta_spot, NI_spot, sine_phase):

        freqx = r*np.cos(t*np.pi/180)*np.cos(x_aoi*np.pi/180)
        freqy = r*np.sin(t*np.pi/180)*np.cos(y_aoi*np.pi/180)

        # factor of 2.0 because F.T. of sine is
        #  (1/2)delta(f-r) + (1/2)delta(f+r)
        amplitude = 2.0*np.sqrt(NI)

        ddm_wavefront_radians = sine(Nact, Dact, amplitude, phi, freqx, freqy)

        # wavefront to dm surface (m)
        ddm_surface_m = 0.5*wavelength*ddm_wavefront_radians/(2.0*np.pi)

        # surface (m) to dm Volts
        ddmv += ddm_surface_m / dm_gain_m_per_v


    return ddmv
