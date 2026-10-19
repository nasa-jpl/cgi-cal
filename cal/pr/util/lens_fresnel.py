# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Propagation function for phase retrieval with lenses

environment variable LENS_FRESNEL_PROP_FAST
If set to anything other than '0', store propagation matrices for
faster execution time. Speed up is approximately factor of 2.
"""

import os
import warnings
import logging

import numpy as np

from cal.util.insertinto import insertinto
import cal.util.check as check
from cal.pr.util.plane import Plane

_listAliasPlanesMFT = [] # list of plane pairs that fail alias check for MFT
_listAliasPlanesQ = [] # list of plane pairs that fail alias check for quad phase
# list of pre-calc'd prop matrices, one set for each
# pair of (pinit, pfinal)
_listStoredArrays = []

logging.captureWarnings(True)
log = logging.getLogger(__name__)

def prop(e0, pinit, pfinal, lam):
    """
    Propagate from one plane back a certain distance and through a lens, and
    then forward again through another lens and distance

    Used to move from one plane to another phase retrieval for both
    lens-swapping and camera-shifting approaches while maintaining camera
    sampling (no interpolation).

    Note only shifts on the output plane will be applied.

    See 'Fresnel Transforms for phase retrieval' for the math, and Soummer et
    al. for the matrix Fourier transform.

    For fast execution time, set environment variable LENS_FRESNEL_PROP_FAST
    If set to anything other than '0', propagation matrices are stored for
    faster execution time. Speed up is approximately factor of 2. On some 
    systems, the required memory might be a burden.

    Arguments:
     e0: 2D complex-valued array.  If this is not the same size as (pinit.ny,
      pinit.nx), then it will be cropped or zero-padded to make this true.
      Each pixel on e0 should physically correspond to a pinit.dpix x
      pinit.dpix size.
     pinit: a Plane object, corresponding to the plane where e0 is located
     pfinal: a Plane object, corresponding to the desired output plane
     lam: wavelength in meters

    Returns:
     2D complex-valued array of size (pfinal.ny, pfinal.nx) with the field in
      the pfinal plane

    """

    # Check inputs, Planes check their data internally
    check.twoD_array(e0, 'e0', TypeError)
    if not isinstance(pinit, Plane):
        raise TypeError('pinit must be a Plane object')
    if not isinstance(pfinal, Plane):
        raise TypeError('pfinal must be a Plane object')
    check.real_positive_scalar(lam, 'lam', TypeError)

    # Enforce shape by pad/crop
    if e0.shape != (pinit.ny, pinit.nx):
        e0 = insertinto(e0, (pinit.ny, pinit.nx))

    # Handle common special case, from feeding in a list of planes to propagate
    # to which includes itself, that causes den == 0
    if pinit == pfinal:
        return e0

    # get prop matrices
    q_prop_in, q_condiv_in, xmat, ymat, q_prop_out, q_condiv_out \
        = _StoreRecallPropMatrices(pinit, pfinal, lam)

    # Define short forms for legibility
    z0 = pinit.z
    f0 = pinit.f
    z1 = pfinal.z
    f1 = pfinal.f

    # 1/zqi, 1/zqo, and 1/zft all have the same denominator, so...
    den = (z0*z1*f1 - z0*z1*f0 + z0*f0*f1 - z1*f0*f1)
    if den == 0:
        raise ZeroDivisionError('Distance/lens combo is singular')

    inv_zft = (-f0*f1)/den

    # Tilt at first plane to create a shift at the second
    xph_in, yph_in = tiltphases(pinit=pinit, pfinal=pfinal, lam=lam)
    tt_in = np.exp(1j*((pfinal.xshift)*xph_in
                       + (pfinal.yshift)*yph_in))
    xph_out, yph_out = tiltphases(pinit=pfinal, pfinal=pinit, lam=lam)
    tt_out = np.exp(-1j*((pinit.xshift)*xph_out
                         + (pinit.yshift)*yph_out))

    # Propagation:
    #  - premultiply with quadratics and tilt
    #  - scaled FT with MFT
    #  - post-multiply with quadratics
    e0in = e0*q_prop_in*q_condiv_in*tt_in
    e1out = np.dot(np.dot(ymat, e0in), xmat)*(inv_zft/(1j*lam))*pinit.dpix*pinit.dpix
    e1 = e1out*q_prop_out*q_condiv_out*tt_out

    # z/lam is a large number, use mod before 2pi mult to keep errors down
    return e1*np.exp(1j*2*np.pi*((z1-z0)/lam % 1))

def tiltphases(pinit, pfinal, lam):
    """
    Compute the phase to create a 1-pixel shift by the Fourier shift theorem

    These are phase terms in radians and should be used as e^{i n phase} for
    some coefficient n.  They are normalized such that e^{i n phase} applied to
    the initial field will produce an n pixel shift in the final field.
    Fourier shift theorem is that a tilt on one side of a Fourier transform
    creates a shift on the other side.

    Of all the pieces of the propagation calculation, we pull this one out in
    particular because the parametric fitting code will also use it to fit the
    shift magnitudes when computes derivatives:
     E = E0 e^{i n phase}
     dE/dn = i phase E0 e^{i n phase} = i phase E

    Arguments:
     pinit: a Plane object, corresponding to the plane where e0 is located
     pfinal: a Plane object, corresponding to the desired output plane
     lam: wavelength in meters

    Returns:
     two phase grids, for respectively x and y, with size (pinit.ny, pinit.nx)

    """

    # Check inputs, Planes check their data internally
    if not isinstance(pinit, Plane):
        raise TypeError('pinit must be a Plane object')
    if not isinstance(pfinal, Plane):
        raise TypeError('pfinal must be a Plane object')
    check.real_positive_scalar(lam, 'lam', TypeError)

    # Define short forms for legibility
    z0 = pinit.z
    f0 = pinit.f
    z1 = pfinal.z
    f1 = pfinal.f

    den = (z0*z1*f1 - z0*z1*f0 + z0*f0*f1 - z1*f0*f1)
    if den == 0:
        raise ZeroDivisionError('Distance/lens combo is singular')
    inv_zft = (-f0*f1)/den  # (1/m)

    #
    onepix = 2*np.pi*pfinal.dpix*(inv_zft/lam) # (rad/m)

    # Want final sampling but initial size
    return pinit.X*onepix, pinit.Y*onepix # (rad)

def _StoreRecallPropMatrices(pinit, pfinal, lam):
    """
    These propagation matrices have no dependence on e0, xshift, yshift
    Then only need to calculate once per (pinit, pfinal, lam) pair
    and store for future propagations:
    q_prop_in
    q_condiv_in
    xmat
    ymat
    q_prop_out
    q_condiv_out

    Note: in common usage, lam will not vary.

    all the arrays are stored in _listStoredArrays, which is a list
    of dicts. Each dict has keys:
    d_arr = {
        'pinit':pinit,
        'pfinal':pfinal,
        'lam':lam,
        'prop_matrices':(
            q_prop_in, q_condiv_in, xmat, ymat, q_prop_out, q_condiv_out
        ),
    }

    """
    
    # search the stored list for this pair
    # prop arrays are stored only if environment variable
    # LENS_FRESNEL_PROP_FAST exists and is not '0'
    dpair = [d_arr for d_arr in _listStoredArrays \
             if (pinit, pfinal, lam) == \
             (d_arr['pinit'], d_arr['pfinal'], d_arr['lam'])]

    # if this pair is already stored, just return the stored arrays:
    if len(dpair) > 0:

        # return the propagation matrices
        return dpair[0]['prop_matrices']

    # else (pinit, pfinal, lam) not in the list, so let's calculate

    # Define short forms for legibility
    z0 = pinit.z
    f0 = pinit.f
    z1 = pfinal.z
    f1 = pfinal.f

    # 1/zqi, 1/zqo, and 1/zft all have the same denominator, so...
    den = (z0*z1*f1 - z0*z1*f0 + z0*f0*f1 - z1*f0*f1)
    if den == 0:
        raise ZeroDivisionError('Distance/lens combo is singular')

    inv_zqi = (-z1*f1 + z1*f0 - f0*f1)/den
    inv_zqo = (z0*f1 - z0*f0 - f0*f1)/den
    inv_zft = (-f0*f1)/den

    # integration is over init plane, assumes square pixels, differential elements:
    dxp = pinit.dpix
    dyp = pinit.dpix
    
    # Build phase terms to matrix-multiply in MFT
    # xprod and xmat shape is (pinit.nx, pfinal.nx)
    xprod = np.outer(pinit.xvec, pfinal.xvec)
    yprod = np.outer(pfinal.yvec, pinit.yvec)
    xmat = np.exp(-1j*2.*np.pi*xprod*(inv_zft/lam))
    ymat = np.exp(-1j*2.*np.pi*yprod*(inv_zft/lam))

    # DFT alias check: (note: this check occurs only once per (pinit, pfinal))
    # if arg xmat or ymat > 2pi
    if np.abs(pinit.nx*pinit.dpix*pfinal.dpix*inv_zft/lam) > 1.0:

        warnings.warn('DFT aliased:')
        warnings.warn('   pinit  z, fl: %.6f, %.6f'%(pinit.z, pinit.f))
        warnings.warn('   pfinal z, fl: %.6f, %.6f'%(pfinal.z, pfinal.f))
        warnings.warn('   pinit.nx*pinit.dpix*pfinal.dpix*inv_zft/lam = %.3e'%(
            pinit.nx*pinit.dpix*pfinal.dpix*inv_zft/lam
        ))

        # if we reduce pinit.nx s.t. no aliasing, how much energy is removed?
        nx_noalias = int(np.floor(np.abs(1.0/(pinit.dpix*pfinal.dpix*inv_zft/lam))))
        warnings.warn('   pinit.nx: %d'%pinit.nx)
        warnings.warn('   nx for no aliasing: %d'%nx_noalias)

    # Build inner quadratic phases
    if z0 == f0:
        q_condiv_in = np.ones((pinit.ny, pinit.nx))
    else:
        q_condiv_in = np.exp(1j*np.pi/(lam*(z0-f0))*pinit.Rsq) # Rsq (m) = (nx*dpix)^2

    # additional wavefront curvature correction, if not zero
    if np.abs(pinit.ceq) > 1e-6:
        q_condiv_in *= np.exp(-1j*(np.pi/lam)*pinit.ceq*pinit.Rsq)

    # propagation quadratic phase
    q_prop_in = np.exp(1j*np.pi*(inv_zqi/lam)*pinit.Rsq)

    # test quadratic phase Nyquist: A/(lam*z) <= 1/dpix
    # can use A = diameter of region with amplitude > 0, or whole array N*dpix
    if np.abs((inv_zqi/lam)*pinit.dpix*pinit.yvec[-1]) > 1.0:

        warnings.warn('quadratic phase Nyquist exceeded')
        warnings.warn('  init  plane: z = %.3e, f = %.3e, nx = %d, dpix = %.3e'%(
            z0, f0, pinit.nx, pinit.dpix
        ))
        warnings.warn('  inv_zqi = %.3e, lam = %.3e'%(inv_zqi, lam))
        warnings.warn('  final plane: z = %.3e, f = %.3e'%(z1, f1))

    # Build outer quadratic phases
    if z1 == f1:
        q_condiv_out = np.ones((pfinal.ny, pfinal.nx))
    else:
        q_condiv_out = np.exp(-1j*np.pi/(lam*(z1-f1))*pfinal.Rsq)

    # additional wavefront curvature correction at final plane, if not zero
    if np.abs(pfinal.ceq) > 1e-6:
        q_condiv_out *= np.exp(1j*(np.pi/lam)*pfinal.ceq*pfinal.Rsq)

    q_prop_out = np.exp(1j*np.pi*(inv_zqo/lam)*pfinal.Rsq)

    # test quadratic phase Nyquist: A/(lam*z) <= 1/dpix
    # can use A = diameter of region with amplitude > 0, or whole array N*dpix
    if np.abs((inv_zqo/lam)*pfinal.dpix*pfinal.yvec[-1]) > 1.0:

        warnings.warn('quadratic phase q_prop_out Nyquist exceeded')
        warnings.warn('  init  plane: z = %.3e, f = %.3e'%(z0, f0))
        warnings.warn('  inv_zq0 = %.3e, lam = %.3e'%(inv_zqo, lam))
        warnings.warn('  final plane: z = %.3e, f = %.3e, nx = %d, dpix = %.3e'%(
            z1, f1, pfinal.nx, pfinal.dpix
        ))

    if 'LENS_FRESNEL_PROP_FAST' in os.environ:
        if os.environ['LENS_FRESNEL_PROP_FAST'] != '0':
            d_arr = {
                'pinit':pinit,
                'pfinal':pfinal,
                'lam':lam,
                'prop_matrices':(
                    q_prop_in, q_condiv_in, xmat, ymat, q_prop_out, q_condiv_out
                ),
            }
            _listStoredArrays.append(d_arr)

    return q_prop_in, q_condiv_in, xmat, ymat, q_prop_out, q_condiv_out
