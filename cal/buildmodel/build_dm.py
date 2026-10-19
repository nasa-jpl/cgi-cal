# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Functions to build DM portions of a HOWFSC optical model definition file
"""

from cal.util.loadyaml import loadyaml
from cal.util.writeyaml import writeyaml
import cal.util.check as check

def add_dm_arrays(fn, dm_id, gainfn, flatfn, tiefn, crosstalkfn):
    """
    Adds information about arrays of DM voltage data to a HOWFSC
    optical model definition file in progress

    No returns; this will write out to the file in `fn`, overwriting any
    existing contents.  Used to incrementally build a model definition file as
    calibration data becomes available.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Arguments:
     fn: name of YAML file to be read from and written to (string)
     dm_id: either "DM1" or "DM2"
     gainfn: string containing a path to a FITS filename from which the DM
      gainmap will be read.  Should either be an absolute path or
      a path relative to the directory where the YAML file will be placed.  No
      validation of the content (beyond it being a string) is done here.
     flatfn: string containing a path to a FITS filename from which the DM
      flatmap will be read.  Should either be an absolute path or
      a path relative to the directory where the YAML file will be placed.  No
      validation of the content (beyond it being a string) is done here.
     tiefn: string containing a path to a FITS filename from which the DM
      tiemap will be read.  Should either be an absolute path or
      a path relative to the directory where the YAML file will be placed.  No
      validation of the content (beyond it being a string) is done here.
     crosstalkfn: None or string containing a path to a YAML filename from
      which the crosstalk mapping will be read. Should either be an absolute
      path or a path relative to the directory where the YAML file will be
      placed. No validation of the content (beyond it being a string) is done
      here. None indicates that no crosstalk will be used.

    """
    # Validate inputs
    check.string(fn, 'fn', TypeError)
    if (dm_id != 'DM1') and (dm_id != 'DM2'):
        raise TypeError('dm_id must be either "DM1" or "DM2"')
    check.string(gainfn, 'gainfn', TypeError)
    check.string(flatfn, 'flatfn', TypeError)
    check.string(tiefn, 'tiefn', TypeError)
    if crosstalkfn is not None:
        check.string(crosstalkfn, 'crosstalkfn', TypeError)

    mdict = loadyaml(fn)

    # build supporting structure if necessary
    if mdict is None:
        mdict = dict()
    else:
        check.dictionary(mdict, "mdict", TypeError)
    if 'dms' not in mdict:
        mdict['dms'] = dict()
    else:
        check.dictionary(mdict['dms'], "mdict['dms']", TypeError)
    if dm_id not in mdict['dms']:
        mdict['dms'][dm_id] = dict()
    else:
        check.dictionary(mdict['dms'][dm_id],
                         "mdict['dms'][" + str(dm_id) + "]",
                         TypeError)
    if 'voltages' not in mdict['dms'][dm_id]:
        mdict['dms'][dm_id]['voltages'] = dict()
    else:
        check.dictionary(mdict['dms'][dm_id]['voltages'],
                         "mdict['dms'][" + str(dm_id) + "]['voltages']",
                         TypeError)

    mdict['dms'][dm_id]['voltages']['gainfn'] = gainfn
    mdict['dms'][dm_id]['voltages']['flatfn'] = flatfn
    mdict['dms'][dm_id]['voltages']['tiefn'] = tiefn
    mdict['dms'][dm_id]['voltages']['crosstalkfn'] = crosstalkfn

    writeyaml(mdict, fn)

    pass


def add_dm_voltage_bounds(fn, dm_id, vmin, vmax, vneighbor, vcorner, vquant):
    """
    Adds information about DM electrical constraints (e.g. caps, neighbor
    rules) to a HOWFSC optical model definition file in progress

    No returns; this will write out to the file in `fn`, overwriting any
    existing contents.  Used to incrementally build a model definition file as
    calibration data becomes available.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Arguments:
     fn: name of YAML file to be read from and written to (string)
     dm_id: either "DM1" or "DM2"
     vmin: minimum permitted voltage on a DM command (floating-point value)
     vmax: maximum permitted voltage on a DM command (floating-point value)
     vneighbor: maximum allowed voltage between laterally-adjacent actuators
      (floating-point value > 0)
     vcorner: maximum allowed voltage between diagonally-adjacent actuators
      (floating-point value > 0)
     vquant: minimum bit size in DM electronics DACs (floating-point value > 0)

    """
    # Validate inputs
    check.string(fn, 'fn', TypeError)
    if (dm_id != 'DM1') and (dm_id != 'DM2'):
        raise TypeError('dm_id must be either "DM1" or "DM2"')
    check.real_scalar(vmin, 'vmin', TypeError)
    check.real_scalar(vmax, 'vmax', TypeError)
    check.real_positive_scalar(vneighbor, 'vneighbor', TypeError)
    check.real_positive_scalar(vcorner, 'vcorner', TypeError)
    check.real_positive_scalar(vquant, 'vquant', TypeError)

    mdict = loadyaml(fn)

    # build supporting structure if necessary
    if mdict is None:
        mdict = dict()
    else:
        check.dictionary(mdict, "mdict", TypeError)
    if 'dms' not in mdict:
        mdict['dms'] = dict()
    else:
        check.dictionary(mdict['dms'], "mdict['dms']", TypeError)
    if dm_id not in mdict['dms']:
        mdict['dms'][dm_id] = dict()
    else:
        check.dictionary(mdict['dms'][dm_id],
                         "mdict['dms'][" + str(dm_id) + "]",
                         TypeError)
    if 'voltages' not in mdict['dms'][dm_id]:
        mdict['dms'][dm_id]['voltages'] = dict()
    else:
        check.dictionary(mdict['dms'][dm_id]['voltages'],
                         "mdict['dms'][" + str(dm_id) + "]['voltages']",
                         TypeError)

    mdict['dms'][dm_id]['voltages']['vmin'] = vmin
    mdict['dms'][dm_id]['voltages']['vmax'] = vmax
    mdict['dms'][dm_id]['voltages']['vneighbor'] = vneighbor
    mdict['dms'][dm_id]['voltages']['vcorner'] = vcorner
    mdict['dms'][dm_id]['voltages']['vquant'] = vquant

    writeyaml(mdict, fn)

    pass


def add_dm_registration(fn, dm_id, nact, ppact_cx, ppact_cy, dx, dy, thact,
                        flipx):
    """
    Adds information about DM registration (translation/rotation/shift/parity)
    to a HOWFSC optical model definition file in progress

    For DM registration, this should represent a registration done in the
    central subband.  The pupil imaging lens (PIL) has wavelength-dependent
    pupil magnification that would create different apparent registrations at
    different bands, but this is not a relevant effect (the DM does not change
    size relative to OTA pupil).

    No returns; this will write out to the file in `fn`, overwriting any
    existing contents.  Used to incrementally build a model definition file as
    calibration data becomes available.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Arguments:
     fn: name of YAML file to be read from and written to (string)
     dm_id: either "DM1" or "DM2"
     nact: number of actuators along an edge of the DM.  Integer > 0; will be
      48 for CGI
     ppact_cx: pixels per actuator in the x-direction on the camera.  Expected
      to be a real positive scalar.
     ppact_cy: pixels per actuator in the y-direction on the camera.  Expected
      to be a real positive scalar.
     dx: number of pixels to shift the DM grid off center in the x-direction
      on the camera. Expected to be a real scalar.
     dy: number of pixels to shift the DM grid off center in the y-direction
      on the camera. Expected to be a real scalar.
     thact: number of degrees to rotate the grid in a counterclockwise
      direction about the center of the array.  Expected to be a real scalar.
     flipx: boolean whether to flip the output in the x-direction, leaving the
      y-direction unchanged.

    """
    # Validate inputs
    check.string(fn, 'fn', TypeError)
    if (dm_id != 'DM1') and (dm_id != 'DM2'):
        raise TypeError('dm_id must be either "DM1" or "DM2"')
    check.positive_scalar_integer(nact, 'nact', TypeError)
    check.real_positive_scalar(ppact_cx, 'ppact_cx', TypeError)
    check.real_positive_scalar(ppact_cy, 'ppact_cy', TypeError)
    check.real_scalar(dx, 'dx', TypeError)
    check.real_scalar(dy, 'dy', TypeError)
    check.real_scalar(thact, 'thact', TypeError)
    check.boolean(flipx, 'flipx', TypeError)

    mdict = loadyaml(fn)

    # build supporting structure if necessary
    if mdict is None:
        mdict = dict()
    else:
        check.dictionary(mdict, "mdict", TypeError)
    if 'dms' not in mdict:
        mdict['dms'] = dict()
    else:
        check.dictionary(mdict['dms'], "mdict['dms']", TypeError)
    if dm_id not in mdict['dms']:
        mdict['dms'][dm_id] = dict()
    else:
        check.dictionary(mdict['dms'][dm_id],
                         "mdict['dms'][" + str(dm_id) + "]",
                         TypeError)
    if 'registration' not in mdict['dms'][dm_id]:
        mdict['dms'][dm_id]['registration'] = dict()
    else:
        check.dictionary(mdict['dms'][dm_id]['registration'],
                         "mdict['dms'][" + str(dm_id) + "]['registration']",
                         TypeError)

    mdict['dms'][dm_id]['registration']['nact'] = nact
    mdict['dms'][dm_id]['registration']['ppact_cx'] = ppact_cx
    mdict['dms'][dm_id]['registration']['ppact_cy'] = ppact_cy
    mdict['dms'][dm_id]['registration']['dx'] = dx
    mdict['dms'][dm_id]['registration']['dy'] = dy
    mdict['dms'][dm_id]['registration']['thact'] = thact
    mdict['dms'][dm_id]['registration']['flipx'] = flipx

    writeyaml(mdict, fn)

    pass


def add_influence_function(fn, dm_id, inffn, ppact_d):
    """
    Adds information about the average DM influence function to a HOWFSC
    optical model definition file in progress

    No returns; this will write out to the file in `fn`, overwriting any
    existing contents.  Used to incrementally build a model definition file as
    calibration data becomes available.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Arguments:
     fn: name of YAML file to be read from and written to (string)
     dm_id: either "DM1" or "DM2"
     inffn: string containing a path to a FITS filename from which the
      influence function will be read.  Should either be an absolute path or
      a path relative to the directory where the YAML file will be placed.  No
      validation of the content (beyond it being a string) is done here.
     ppact_d: design pixels per actuator in the array in inffn.  Floating-point
      value > 0.

    """
    # Validate inputs
    check.string(fn, 'fn', TypeError)
    if (dm_id != 'DM1') and (dm_id != 'DM2'):
        raise TypeError('dm_id must be either "DM1" or "DM2"')
    check.string(inffn, 'inffn', TypeError)
    check.real_positive_scalar(ppact_d, 'ppact_d', TypeError)

    mdict = loadyaml(fn)

    # build supporting structure if necessary
    if mdict is None:
        mdict = dict()
    else:
        check.dictionary(mdict, "mdict", TypeError)
    if 'dms' not in mdict:
        mdict['dms'] = dict()
    else:
        check.dictionary(mdict['dms'], "mdict['dms']", TypeError)
    if dm_id not in mdict['dms']:
        mdict['dms'][dm_id] = dict()
    else:
        check.dictionary(mdict['dms'][dm_id],
                         "mdict['dms'][" + str(dm_id) + "]",
                         TypeError)
    if 'registration' not in mdict['dms'][dm_id]:
        mdict['dms'][dm_id]['registration'] = dict()
    else:
        check.dictionary(mdict['dms'][dm_id]['registration'],
                         "mdict['dms'][" + str(dm_id) + "]['registration']",
                         TypeError)

    mdict['dms'][dm_id]['registration']['inffn'] = inffn
    mdict['dms'][dm_id]['registration']['ppact_d'] = ppact_d

    writeyaml(mdict, fn)

    pass


def add_pitch(fn, dm_id, pitch):
    """
    Adds a DM pitch to a HOWFSC optical model definition file in progress

    No returns; this will write out to the file in `fn`, overwriting any
    existing contents.  Used to incrementally build a model definition file as
    calibration data becomes available.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Arguments:
     fn: name of YAML file to be read from and written to (string)
     dm_id: either "DM1" or "DM2"
     pitch: floating-point value > 0 representing the actuator pitch in meters.
      Expected to be between 0.00099 and 0.00100 meters as per DNG 338450

    """
    # Validate inputs
    check.string(fn, 'fn', TypeError)
    if (dm_id != 'DM1') and (dm_id != 'DM2'):
        raise TypeError('dm_id must be either "DM1" or "DM2"')
    check.real_positive_scalar(pitch, 'pitch', TypeError)

    mdict = loadyaml(fn)

    # build supporting structure if necessary
    if mdict is None:
        mdict = dict()
    else:
        check.dictionary(mdict, "mdict", TypeError)
    if 'dms' not in mdict:
        mdict['dms'] = dict()
    else:
        check.dictionary(mdict['dms'], "mdict['dms']", TypeError)
    if dm_id not in mdict['dms']:
        mdict['dms'][dm_id] = dict()
    else:
        check.dictionary(mdict['dms'][dm_id],
                         "mdict['dms'][" + str(dm_id) + "]",
                         TypeError)

    mdict['dms'][dm_id]['pitch'] = pitch

    writeyaml(mdict, fn)

    pass


def add_z(fn, dm_id, z):
    """
    Adds a DM z (propagation distance) to a HOWFSC optical model definition
    file in progress

    No returns; this will write out to the file in `fn`, overwriting any
    existing contents.  Used to incrementally build a model definition file as
    calibration data becomes available.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Arguments:
     fn: name of YAML file to be read from and written to (string)
     dm_id: either "DM1" or "DM2"
     z: floating-point value representing the propagation distance from a pupil
      plane to the DM surface in meters.  May in principle be negative if the
      DM precedes the pupil.  Expected to be approximately 0.0 for DM1 and 1.0
      for DM2.

    """
    # Validate inputs
    check.string(fn, 'fn', TypeError)
    if (dm_id != 'DM1') and (dm_id != 'DM2'):
        raise TypeError('dm_id must be either "DM1" or "DM2"')
    check.real_scalar(z, 'z', TypeError)

    mdict = loadyaml(fn)

    # build supporting structure if necessary
    if mdict is None:
        mdict = dict()
    else:
        check.dictionary(mdict, "mdict", TypeError)
    if 'dms' not in mdict:
        mdict['dms'] = dict()
    else:
        check.dictionary(mdict['dms'], "mdict['dms']", TypeError)
    if dm_id not in mdict['dms']:
        mdict['dms'][dm_id] = dict()
    else:
        check.dictionary(mdict['dms'][dm_id],
                         "mdict['dms'][" + str(dm_id) + "]",
                         TypeError)

    mdict['dms'][dm_id]['z'] = z

    writeyaml(mdict, fn)

    pass
