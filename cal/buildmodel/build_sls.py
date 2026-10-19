# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Functions to build init and SLS portions of a HOWFSC optical model def'n file
"""
from cal.util import check
from cal.util.loadyaml import loadyaml
from cal.util.writeyaml import writeyaml


def add_dminit(fn_model, dm_id, dminit):
    """
    Add a DM init name to a HOWFSC optical model definition file in progress.

    No returns; this will write out to the file in `fn_model`, overwriting any
    existing contents.  Used to incrementally build a model definition file
    as calibration data becomes available.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Parameters
    ----------
    fn_model : str
        Name of YAML file to be read from and written to.
    dm_id : str
        Either "DM1" or "DM2".
    dminit : str
        Name of the FITS file containing the initial absolute DM voltage map.

    Returns
    -------
    None.

    """
    # Validate inputs
    check.string(fn_model, 'fn_model', TypeError)
    if (dm_id != 'DM1') and (dm_id != 'DM2'):
        raise TypeError('dm_id must be either "DM1" or "DM2"')
    check.string(dminit, 'dminit', TypeError)

    mdict = loadyaml(fn_model)

    # Build supporting structure if necessary
    if mdict is None:
        mdict = dict()
    else:
        check.dictionary(mdict, "mdict", TypeError)
    if 'init' not in mdict:
        mdict['init'] = dict()
    else:
        check.dictionary(mdict['init'], "mdict['init']", TypeError)
    if dm_id not in mdict['init']:
        mdict['init'][dm_id] = dict()
    else:
        check.dictionary(mdict['init'][dm_id],
                         "mdict['init'][" + str(dm_id) + "]",
                         TypeError)

    mdict['init'][dm_id]['dminit'] = dminit
    
    # Sort for readability in the output file
    mdict['init'] = dict(sorted(mdict['init'] .items()))
    
    writeyaml(mdict, fn_model)

    return None


def add_dh(fn_model, lam_index, dh):
    """
    Add a dark hole map fn to a HOWFSC optical model def'n file in progress.

    No returns; this will write out to the file in `fn_model`, overwriting any
    existing contents.  Used to incrementally build a model definition file
    as calibration data becomes available.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Parameters
    ----------
    fn_model : str
        Name of YAML file to be read from and written to.
    lam_index : int
        Index of the SLS object (i.e., subbandpass) to assign this data to.
        Must be a nonnegative scalar integer.
    dh : str
        Name of the FITS file containing the 0-and-1 map of pixels that
        define the dark hole region.

    Returns
    -------
    None.

    """
    # Validate inputs
    check.string(fn_model, 'fn_model', TypeError)
    check.nonnegative_scalar_integer(lam_index, 'lam_index', TypeError)
    check.string(dh, 'dh', TypeError)

    mdict = loadyaml(fn_model)

    # Build supporting structure if necessary
    if mdict is None:
        mdict = dict()
    else:
        check.dictionary(mdict, "mdict", TypeError)
    if 'sls' not in mdict:
        mdict['sls'] = dict()
    else:
        check.dictionary(mdict['sls'], "mdict['sls']", TypeError)
    if lam_index not in mdict['sls']:
        mdict['sls'][lam_index] = dict()
    else:
        check.dictionary(mdict['sls'][lam_index],
                         "mdict['sls'][" + str(lam_index) + "]",
                         TypeError)

    mdict['sls'][lam_index]['dh'] = dh

    # Sort for readability in the output file
    mdict['sls'] = dict(sorted(mdict['sls'] .items()))

    writeyaml(mdict, fn_model)

    return None


def add_ft_dir(fn_model, lam_index, ft_dir):
    """
    Add the Fourier transform direction to a HOWFSC optical model file.

    No returns; this will write out to the file in `fn_model`, overwriting any
    existing contents.  Used to incrementally build a model definition file
    as calibration data becomes available.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Parameters
    ----------
    fn_model : str
        Name of YAML file to be read from and written to.
    lam_index : int
        Index of the SLS object (i.e., subbandpass) to assign this data to.
        Must be a nonnegative scalar integer.
    ft_dir : str
        The direction of the Fourier transform. Must be 'forward' or 'reverse'.

    Returns
    -------
    None.

    """
    # Validate inputs
    check.string(fn_model, 'fn_model', TypeError)
    check.nonnegative_scalar_integer(lam_index, 'lam_index', TypeError)
    check.string(ft_dir, 'ft_dir', TypeError)
    if ft_dir not in ('forward', 'reverse'):
        raise TypeError("ft_dir must be 'forward' or 'reverse'.")

    mdict = loadyaml(fn_model)

    # Build supporting structure if necessary
    if mdict is None:
        mdict = dict()
    else:
        check.dictionary(mdict, "mdict", TypeError)
    if 'sls' not in mdict:
        mdict['sls'] = dict()
    else:
        check.dictionary(mdict['sls'], "mdict['sls']", TypeError)
    if lam_index not in mdict['sls']:
        mdict['sls'][lam_index] = dict()
    else:
        check.dictionary(mdict['sls'][lam_index],
                         "mdict['sls'][" + str(lam_index) + "]",
                         TypeError)

    mdict['sls'][lam_index]['ft_dir'] = ft_dir

    # Sort for readability in the output file
    mdict['sls'] = dict(sorted(mdict['sls'] .items()))

    writeyaml(mdict, fn_model)

    return None


def add_lam(fn_model, lam_index, lam):
    """
    Add wavelength to a HOWFSC optical model definition file in progress.

    The following two rules must be met in order for the final HOWFSC optical
    model definition file to be valid, but no explicit checks for them are
    performed during the construction of this file.
    2cii. Each of the N collections in sls shall be arranged in wavelength
          order, from shortest wavelength to longest.
    2ciii. There shall be no more than one collection per wavelength. That is,
           do not use the same wavelength in multiple SLS indices.

    No returns; this will write out to the file in `fn_model`, overwriting any
    existing contents.  Used to incrementally build a model definition file
    as calibration data becomes available.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Parameters
    ----------
    fn_model : str
        Name of YAML file to be read from and written to.
    lam_index : int
        Index of the SLS object (i.e., subbandpass) to assign this data to.
        Must be a nonnegative scalar integer.
    lam : float
        Wavelength of the SLS (i.e., subbandpass). Units of meters.

    Returns
    -------
    None.

    """
    # Validate inputs
    check.string(fn_model, 'fn_model', TypeError)
    check.nonnegative_scalar_integer(lam_index, 'lam_index', TypeError)
    check.real_positive_scalar(lam, 'lam', TypeError)

    mdict = loadyaml(fn_model)

    # Build supporting structure if necessary
    if mdict is None:
        mdict = dict()
    else:
        check.dictionary(mdict, "mdict", TypeError)
    if 'sls' not in mdict:
        mdict['sls'] = dict()
    else:
        check.dictionary(mdict['sls'], "mdict['sls']", TypeError)
    if lam_index not in mdict['sls']:
        mdict['sls'][lam_index] = dict()
    else:
        check.dictionary(mdict['sls'][lam_index],
                         "mdict['sls'][" + str(lam_index) + "]",
                         TypeError)

    mdict['sls'][lam_index]['lam'] = lam

    # Sort for readability in the output file
    mdict['sls'] = dict(sorted(mdict['sls'] .items()))

    writeyaml(mdict, fn_model)

    return None


def add_epup(fn_model, lam_index, afn, pdp, pfn, tip, tilt):
    """
    Add epup dict to a HOWFSC optical model definition file in progress.

    No returns; this will write out to the file in `fn_model`, overwriting any
    existing contents.  Used to incrementally build a model definition file
    as calibration data becomes available.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Parameters
    ----------
    fn_model : str
        Name of YAML file to be read from and written to.
    lam_index : int
        Index of the SLS object (i.e., subbandpass) to assign this data to.
        Must be a nonnegative scalar integer.
    afn : str
        Name of the FITS file containing the pupil amplitude map.
    pdp : float
        Pupil diameter. Units of pixels.
    pfn : str
        Name of the FITS file containing the pupil phase map.
    tip : float
        Amount of upstream tip to apply to the wavefront. Units of detector
        pixels.
    tilt : float
        Amount of upstream tilt to apply to the wavefront. Units of detector
        pixels.

    Returns
    -------
    None.

    """
    # Validate inputs
    check.string(fn_model, 'fn_model', TypeError)
    check.nonnegative_scalar_integer(lam_index, 'lam_index', TypeError)
    check.string(afn, 'afn', TypeError)
    check.real_positive_scalar(pdp, 'pdp', TypeError)
    check.string(pfn, 'pfn', TypeError)
    check.real_scalar(tip, 'tip', TypeError)
    check.real_scalar(tilt, 'tilt', TypeError)

    mdict = loadyaml(fn_model)

    # Build supporting structure if necessary
    if mdict is None:
        mdict = dict()
    else:
        check.dictionary(mdict, "mdict", TypeError)

    if 'sls' not in mdict:
        mdict['sls'] = dict()
    else:
        check.dictionary(mdict['sls'], "mdict['sls']", TypeError)

    if lam_index not in mdict['sls']:
        mdict['sls'][lam_index] = dict()
    else:
        check.dictionary(mdict['sls'][lam_index],
                         "mdict['sls'][" + str(lam_index) + "]",
                         TypeError)

    if 'epup' not in mdict['sls'][lam_index]:
        mdict['sls'][lam_index]['epup'] = dict()
    else:
        check.dictionary(mdict['sls'][lam_index]['epup'],
                         "mdict['sls'][" + str(lam_index) + "]['epup']",
                         TypeError)

    mdict['sls'][lam_index]['epup']['afn'] = afn
    mdict['sls'][lam_index]['epup']['pdp'] = pdp
    mdict['sls'][lam_index]['epup']['pfn'] = pfn
    mdict['sls'][lam_index]['epup']['tip'] = tip
    mdict['sls'][lam_index]['epup']['tilt'] = tilt

    # Sort for readability in the output file
    mdict['sls'] = dict(sorted(mdict['sls'] .items()))

    writeyaml(mdict, fn_model)

    return None


def add_fpm(fn_model, lam_index, afn, isopen, pfn, ppl):
    """
    Add fpm dict to a HOWFSC optical model definition file in progress.

    Uses amplitude and phase breakdown of the mask.

    No returns; this will write out to the file in `fn_model`, overwriting any
    existing contents.  Used to incrementally build a model definition file
    as calibration data becomes available.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Parameters
    ----------
    fn_model : str
        Name of YAML file to be read from and written to.
    lam_index : int
        Index of the SLS object (i.e., subbandpass) to assign this data to.
        Must be a nonnegative scalar integer.
    afn : str
        Name of the FITS file containing the map of FPM amplitude values.
    isopen : bool
        Whether the FPM has an open or closed outer edge. True means Babinet's
        principle is used during propagation. False means there is a finite
        outer edge and a direct MFT is used for propagation to/from the FPM.
    pfn : str
        Name of the FITS file containing the map of FPM phase values.
    ppl : float
        Resolution of the FPM in the files named by rfn and ifn. Units of
        pixels per lambda/D.

    Returns
    -------
    None.

    """
    # Validate inputs
    check.string(fn_model, 'fn_model', TypeError)
    check.nonnegative_scalar_integer(lam_index, 'lam_index', TypeError)
    check.string(afn, 'afn', TypeError)
    check.boolean(isopen, 'isopen', TypeError)
    check.string(pfn, 'pfn', TypeError)
    check.real_positive_scalar(ppl, 'ppl', TypeError)

    mdict = loadyaml(fn_model)

    # Build supporting structure if necessary
    if mdict is None:
        mdict = dict()
    else:
        check.dictionary(mdict, "mdict", TypeError)

    if 'sls' not in mdict:
        mdict['sls'] = dict()
    else:
        check.dictionary(mdict['sls'], "mdict['sls']", TypeError)

    if lam_index not in mdict['sls']:
        mdict['sls'][lam_index] = dict()
    else:
        check.dictionary(mdict['sls'][lam_index],
                         "mdict['sls'][" + str(lam_index) + "]",
                         TypeError)

    if 'fpm' not in mdict['sls'][lam_index]:
        mdict['sls'][lam_index]['fpm'] = dict()
    else:
        check.dictionary(mdict['sls'][lam_index]['fpm'],
                         "mdict['sls'][" + str(lam_index) + "]['fpm']",
                         TypeError)

    mdict['sls'][lam_index]['fpm']['afn'] = afn
    mdict['sls'][lam_index]['fpm']['isopen'] = isopen
    mdict['sls'][lam_index]['fpm']['pfn'] = pfn
    mdict['sls'][lam_index]['fpm']['ppl'] = ppl

    # Sort for readability in the output file
    mdict['sls'] = dict(sorted(mdict['sls'] .items()))

    writeyaml(mdict, fn_model)

    return None


def add_fpm_real_imag(fn_model, lam_index, rfn, isopen, ifn, ppl):
    """
    Add fpm dict to a HOWFSC optical model definition file in progress.

    Uses real and imaginary breakdown of the mask. For an amplitude and
    phase breakdown, use add_fpm() instead.

    No returns; this will write out to the file in `fn_model`, overwriting any
    existing contents.  Used to incrementally build a model definition file
    as calibration data becomes available.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Parameters
    ----------
    fn_model : str
        Name of YAML file to be read from and written to.
    lam_index : int
        Index of the SLS object (i.e., subbandpass) to assign this data to.
        Must be a nonnegative scalar integer.
    rfn : str
        Name of the FITS file containing the map of real FPM values.
    isopen : bool
        Whether the FPM has an open or closed outer edge. True means Babinet's
        principle is used during propagation. False means there is a finite
        outer edge and a direct MFT is used for propagation to/from the FPM.
    ifn : str
        Name of the FITS file containing the map of imaginary FPM values.
    ppl : float
        Resolution of the FPM in the files named by rfn and ifn. Units of
        pixels per lambda/D.

    Returns
    -------
    None.

    """
    # Validate inputs
    check.string(fn_model, 'fn_model', TypeError)
    check.nonnegative_scalar_integer(lam_index, 'lam_index', TypeError)
    check.string(rfn, 'rfn', TypeError)
    check.boolean(isopen, 'isopen', TypeError)
    check.string(ifn, 'ifn', TypeError)
    check.real_positive_scalar(ppl, 'ppl', TypeError)

    mdict = loadyaml(fn_model)

    # Build supporting structure if necessary
    if mdict is None:
        mdict = dict()
    else:
        check.dictionary(mdict, "mdict", TypeError)

    if 'sls' not in mdict:
        mdict['sls'] = dict()
    else:
        check.dictionary(mdict['sls'], "mdict['sls']", TypeError)

    if lam_index not in mdict['sls']:
        mdict['sls'][lam_index] = dict()
    else:
        check.dictionary(mdict['sls'][lam_index],
                         "mdict['sls'][" + str(lam_index) + "]",
                         TypeError)

    if 'fpm' not in mdict['sls'][lam_index]:
        mdict['sls'][lam_index]['fpm'] = dict()
    else:
        check.dictionary(mdict['sls'][lam_index]['fpm'],
                         "mdict['sls'][" + str(lam_index) + "]['fpm']",
                         TypeError)

    mdict['sls'][lam_index]['fpm']['rfn'] = rfn
    mdict['sls'][lam_index]['fpm']['isopen'] = isopen
    mdict['sls'][lam_index]['fpm']['ifn'] = ifn
    mdict['sls'][lam_index]['fpm']['ppl'] = ppl

    # Sort for readability in the output file
    mdict['sls'] = dict(sorted(mdict['sls'] .items()))

    writeyaml(mdict, fn_model)

    return None


def add_fs(fn_model, lam_index, afn, pfn, ppl):
    """
    Add field stop dict to a HOWFSC optical model definition file in progress.

    No returns; this will write out to the file in `fn_model`, overwriting any
    existing contents.  Used to incrementally build a model definition file
    as calibration data becomes available.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Parameters
    ----------
    fn_model : str
        Name of YAML file to be read from and written to.
    lam_index : int
        Index of the SLS object (i.e., subbandpass) to assign this data to.
        Must be a nonnegative scalar integer.
    afn : str
        Name of the FITS file containing the field stop amplitude map.
    pfn : str
        Name of the FITS file containing the field stop phase map.
    ppl : float
        Resolution of the field stop in the files named by afn and pfn.
        Must match the detector resolution. Units of pixels per
        lambda_central/D.

    Returns
    -------
    None.

    """
    # Validate inputs
    check.string(fn_model, 'fn_model', TypeError)
    check.nonnegative_scalar_integer(lam_index, 'lam_index', TypeError)
    check.string(afn, 'afn', TypeError)
    check.string(pfn, 'pfn', TypeError)
    check.real_positive_scalar(ppl, 'ppl', TypeError)

    mdict = loadyaml(fn_model)

    # Build supporting structure if necessary
    if mdict is None:
        mdict = dict()
    else:
        check.dictionary(mdict, "mdict", TypeError)

    if 'sls' not in mdict:
        mdict['sls'] = dict()
    else:
        check.dictionary(mdict['sls'], "mdict['sls']", TypeError)

    if lam_index not in mdict['sls']:
        mdict['sls'][lam_index] = dict()
    else:
        check.dictionary(mdict['sls'][lam_index],
                         "mdict['sls'][" + str(lam_index) + "]",
                         TypeError)

    if 'fs' not in mdict['sls'][lam_index]:
        mdict['sls'][lam_index]['fs'] = dict()
    else:
        check.dictionary(mdict['sls'][lam_index]['fs'],
                         "mdict['sls'][" + str(lam_index) + "]['fs']",
                         TypeError)

    mdict['sls'][lam_index]['fs']['afn'] = afn
    mdict['sls'][lam_index]['fs']['pfn'] = pfn
    mdict['sls'][lam_index]['fs']['ppl'] = ppl

    # Sort for readability in the output file
    mdict['sls'] = dict(sorted(mdict['sls'] .items()))

    writeyaml(mdict, fn_model)

    return None


def add_lyot_mask(fn_model, lam_index, afn, pdp, pfn):
    """
    Add mask part of lyot dict to a HOWFSC optical model defn file in progress.

    No returns; this will write out to the file in `fn_model`, overwriting any
    existing contents.  Used to incrementally build a model definition file
    as calibration data becomes available.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Parameters
    ----------
    fn_model : str
        Name of YAML file to be read from and written to.
    lam_index : int
        Index of the SLS object (i.e., subbandpass) to assign this data to.
        Must be a nonnegative scalar integer.
    afn : str
        Name of the FITS file containing the Lyot stop amplitude map.
    pdp : float
        Pupil diameter. Units of pixels.
    pfn : str
        Name of the FITS file containing the Lyot stop phase map. This is
        where any downstream phase aberrations (e.g., between the FPM and
        the detector) should be assigned in the model.

    Returns
    -------
    None.

    """
    # Validate inputs
    check.string(fn_model, 'fn_model', TypeError)
    check.nonnegative_scalar_integer(lam_index, 'lam_index', TypeError)
    check.string(afn, 'afn', TypeError)
    check.real_positive_scalar(pdp, 'pdp', TypeError)
    check.string(pfn, 'pfn', TypeError)

    mdict = loadyaml(fn_model)

    # Build supporting structure if necessary
    if mdict is None:
        mdict = dict()
    else:
        check.dictionary(mdict, "mdict", TypeError)

    if 'sls' not in mdict:
        mdict['sls'] = dict()
    else:
        check.dictionary(mdict['sls'], "mdict['sls']", TypeError)

    if lam_index not in mdict['sls']:
        mdict['sls'][lam_index] = dict()
    else:
        check.dictionary(mdict['sls'][lam_index],
                         "mdict['sls'][" + str(lam_index) + "]",
                         TypeError)

    if 'lyot' not in mdict['sls'][lam_index]:
        mdict['sls'][lam_index]['lyot'] = dict()
    else:
        check.dictionary(mdict['sls'][lam_index]['lyot'],
                         "mdict['sls'][" + str(lam_index) + "]['lyot']",
                         TypeError)

    mdict['sls'][lam_index]['lyot']['afn'] = afn
    mdict['sls'][lam_index]['lyot']['pdp'] = pdp
    mdict['sls'][lam_index]['lyot']['pfn'] = pfn

    # Sort for readability in the output file
    mdict['sls'] = dict(sorted(mdict['sls'] .items()))

    writeyaml(mdict, fn_model)

    return None


def add_lyot_tip_tilt(fn_model, lam_index, tip, tilt):
    """
    Add tip/tilt part of lyot dict to a HOWFSC optical model def'n file.

    No returns; this will write out to the file in `fn_model`, overwriting any
    existing contents.  Used to incrementally build a model definition file
    as calibration data becomes available.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Parameters
    ----------
    fn_model : str
        Name of YAML file to be read from and written to.
    lam_index : int
        Index of the SLS object (i.e., subbandpass) to assign this data to.
        Must be a nonnegative scalar integer.
    tip : float
        Amount of downstream tip to apply to the wavefront. This is purely to
        account for beam offsets on the detector away from the detector center
        pixel and does not affect how the beam lands on any intermediate
        planes, unlike the upstream tip at epup. Units of detector pixels.
    tilt : float
        Amount of downstream tilt to apply to the wavefront. This is purely to
        account for beam offsets on the detector away from the detector center
        pixel and does not affect how the beam lands on any intermediate
        planes, unlike the upstream tilt at epup. Units of detector pixels.

    Returns
    -------
    None.

    """
    # Validate inputs
    check.string(fn_model, 'fn_model', TypeError)
    check.nonnegative_scalar_integer(lam_index, 'lam_index', TypeError)
    check.real_scalar(tip, 'tip', TypeError)
    check.real_scalar(tilt, 'tilt', TypeError)

    mdict = loadyaml(fn_model)

    # Build supporting structure if necessary
    if mdict is None:
        mdict = dict()
    else:
        check.dictionary(mdict, "mdict", TypeError)

    if 'sls' not in mdict:
        mdict['sls'] = dict()
    else:
        check.dictionary(mdict['sls'], "mdict['sls']", TypeError)

    if lam_index not in mdict['sls']:
        mdict['sls'][lam_index] = dict()
    else:
        check.dictionary(mdict['sls'][lam_index],
                         "mdict['sls'][" + str(lam_index) + "]",
                         TypeError)

    if 'lyot' not in mdict['sls'][lam_index]:
        mdict['sls'][lam_index]['lyot'] = dict()
    else:
        check.dictionary(mdict['sls'][lam_index]['lyot'],
                         "mdict['sls'][" + str(lam_index) + "]['lyot']",
                         TypeError)

    mdict['sls'][lam_index]['lyot']['tip'] = tip
    mdict['sls'][lam_index]['lyot']['tilt'] = tilt

    # Sort for readability in the output file
    mdict['sls'] = dict(sorted(mdict['sls'] .items()))

    writeyaml(mdict, fn_model)

    return None


def add_sp(fn_model, lam_index, afn, pdp, pfn):
    """
    Add sp dict to a HOWFSC optical model definition file in progress.

    No returns; this will write out to the file in `fn_model`, overwriting any
    existing contents.  Used to incrementally build a model definition file
    as calibration data becomes available.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Parameters
    ----------
    fn_model : str
        Name of YAML file to be read from and written to.
    lam_index : int
        Index of the SLS object (i.e., subbandpass) to assign this data to.
        Must be a nonnegative scalar integer.
    afn : str
        Name of the FITS file containing the shaped pupil amplitude map.
    pdp : float
        Pupil diameter. Units of pixels.
    pfn : str
        Name of the FITS file containing the shaped pupil phase map. This is
        only for delta phase attributable only to the shaped pupil mask
        itself since epup's phase accounts for the rest.

    Returns
    -------
    None.

    """
    # Validate inputs
    check.string(fn_model, 'fn_model', TypeError)
    check.nonnegative_scalar_integer(lam_index, 'lam_index', TypeError)
    check.string(afn, 'afn', TypeError)
    check.real_positive_scalar(pdp, 'pdp', TypeError)
    check.string(pfn, 'pfn', TypeError)

    mdict = loadyaml(fn_model)

    # Build supporting structure if necessary
    if mdict is None:
        mdict = dict()
    else:
        check.dictionary(mdict, "mdict", TypeError)

    if 'sls' not in mdict:
        mdict['sls'] = dict()
    else:
        check.dictionary(mdict['sls'], "mdict['sls']", TypeError)

    if lam_index not in mdict['sls']:
        mdict['sls'][lam_index] = dict()
    else:
        check.dictionary(mdict['sls'][lam_index],
                         "mdict['sls'][" + str(lam_index) + "]",
                         TypeError)

    if 'sp' not in mdict['sls'][lam_index]:
        mdict['sls'][lam_index]['sp'] = dict()
    else:
        check.dictionary(mdict['sls'][lam_index]['sp'],
                         "mdict['sls'][" + str(lam_index) + "]['sp']",
                         TypeError)

    mdict['sls'][lam_index]['sp']['afn'] = afn
    mdict['sls'][lam_index]['sp']['pdp'] = pdp
    mdict['sls'][lam_index]['sp']['pfn'] = pfn

    # Sort for readability in the output file
    mdict['sls'] = dict(sorted(mdict['sls'] .items()))

    writeyaml(mdict, fn_model)

    return None
