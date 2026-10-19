# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Functions to load and modify phase retrieval data."""
import os

from astropy.io import fits
import numpy as np
import skimage.morphology

from cal.ampthresh.ampthresh import ampthresh
from cal.util import check
from cal.util.fresnelprop import fresnelprop
from cal.util.insertinto import insertinto as inin
from cal.util.loadyaml import loadyaml
from cal.util import remove_ptt
from . import wavelength_lut
from .unwrap import unwrap_segments


HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND_PR_PATH = os.path.join(HERE, 'cgidata', 'prnum_000279')


def load_backend_pr(data_path=BACKEND_PR_PATH):
    """
    Load a back-end phase retrieval.

    Parameters
    ----------
    None

    Keyword Parameters
    ------------------
    data_path : str
        Path of the folder containing the backend phase retrieval output files:
        - amp.fits : the 2-D amplitude map.
        - phwrap.fits : the wrapped 2-D phase map.
        - parms.yaml : the phase retrieval tuning parameters used.
        The default folder is in the repo and contains data taken during TVAC.

    Returns
    -------
    amp : array_like
        2-D amplitude map for the backend phase retrieval
    ph : array_like
        2-D unwrapped map of the phase
    xc, yc : int
        The x and y center pixels used to crop the returned arrays from the
        original 1024x1024 cleaned frame size.
    pr_params : dict
        Dictionary of the tuning parameters used in the phase retrieval

    """
    amp = fits.getdata(os.path.join(data_path, 'amp.fits'))
    ph = fits.getdata(os.path.join(data_path, 'phwrap.fits'))
    pr_params = loadyaml(os.path.join(data_path, 'parms.yaml'))
    xc = pr_params['XYcen_PIL'][0]
    yc = pr_params['XYcen_PIL'][1]

    return amp, ph, xc, yc, pr_params


def read_and_prop_pr(amp, phwrap, pr_params, fn_pr_prop_params,
                     subtract_backend=True,
                     data_path_backend=BACKEND_PR_PATH):
    """
    Load a back-end phase retrieval from data stored in the repo.

    Parameters
    ----------
    amp : array_like
        2-D amplitude map for the end-to-end phase retrieval
    phwrap : array_like
        2-D wrapped map of the phase. Preference is for the raw, wrapped phase
        instead of the unwrapped phase to avoid spuriously masked regions that
        will cause issues.
    pr_params : dict
        Dictionary of the tuning parameters used in the phase retrieval
    fn_pr_prop_params : str
        Path and filename of the YAML file containing a dictionary of tuning
        parameters used for Fresnel propagating the phase retrieved E-field
        back to the pupil from the PIL plane.

    Keyword Parameters
    ------------------
    subtract_backend : bool
        Whether to subtract off the measured back-end phase of the instrument.
        Default is True.
    data_path_backend : str
        Path of the folder containing the backend phase retrieval output files.
        The default folder is in the repo and contains data taken during TVAC.

    Returns
    -------
    amp_pupil : array_like
        2-D map of the amplitude of the instrument front-end
    ph_pupil : array_like
        2-D map of the unwrapped phase of the instrument front-end.
    bmask_pupil : array_like
        2-D mask of which pixels are assumed to be illuminated parts of the
        pupil.
    x_offset, y_offset : float
        The original, pre-correction estimated x- and y-offsets of the PSF at
        the given focal plane resolution ppl and for the given Fourier
        transform direction ft_dir.

    Notes
    -----
    Steps for PFAC preparation of phase retrieval data:
    0. Receive wrapped end-to-end PR (not unwrapped to avoid fake masked
       regions).
    1. Load unwrapped back-end PR (unwrapped is fine here).
    2. Subtract back-end phase from end-to-end phase (in E-field to avoid
       issues with unwrapping making artifacts before Fresnel propagation) to
       get front-end E-field at PIL. This looks like:
       E_PIL = amp_end2end*exp(1j*(ph_end2end - ph_backend))
    3. Run remove_ptt_at_focus() on the front-end E-field at PIL.
    4. Fresnel propagate from the PIL plane to the DM1 (i.e., pupil) plane.
    5. Convert E-field back to amplitude and unwrapped phase.
       (Both of these will be outputs.)
    6. (Continued in cal/pfac/pfac.py's prep_pfac_data())
       Run pupilzernikes and DM registration on the unwrapped phases to get
       the focus term and the shear between the pupil and DM1.

    """
    check.twoD_array(amp, 'amp', TypeError)
    check.real_array(amp, 'amp', TypeError)
    check.twoD_array(phwrap, 'phwrap', TypeError)
    check.real_array(phwrap, 'phwrap', TypeError)
    if phwrap.shape != amp.shape:
        raise ValueError('phwrap must have same shape as amp')
    check.dictionary(pr_params, 'pr_params', TypeError)
    check.boolean(subtract_backend, 'subtract_backend', TypeError)

    lam_dict = wavelength_lut.get()
    lam = lam_dict[pr_params['bandpass'].lower()]

    PROP_PARAMS_DIR = os.path.dirname(os.path.abspath(fn_pr_prop_params))
    input_dict = loadyaml(fn_pr_prop_params)
    ppl = input_dict['ppl']
    fn_ref_psf = input_dict['fn_ref_psf']
    if not os.path.isabs(fn_ref_psf):
        fn_ref_psf = os.path.join(PROP_PARAMS_DIR, fn_ref_psf)
    deltaz = input_dict['deltaz']
    nxfresnel = input_dict['nxfresnel']
    diam_pupil_pix = input_dict['diam_pupil_pix']
    diam_pupil_m = input_dict['diam_pupil_m']
    ft_dir = input_dict['ft_dir']
    fn_psffit_tuning = input_dict['fn_psffit_tuning']
    if not os.path.isabs(fn_psffit_tuning):
        fn_psffit_tuning = os.path.join(PROP_PARAMS_DIR, fn_psffit_tuning)
    nCrop = input_dict['nCrop']
    nBin = input_dict['nBin']
    nIter = input_dict['nIter']
    min_size_zeros = input_dict['min_size_zeros']

    # 1. Load and modify the backend phase map
    if subtract_backend:

        # Load back-end PR data
        _, ph_backend, _, _, pr_params_backend = load_backend_pr(
            data_path=data_path_backend)
        lam_dict = wavelength_lut.get()
        lam_backend = lam_dict[pr_params_backend['bandpass'].lower()]

        # Scale backend phase with wavelength because it was taken in Band 1B
        # only.
        ph_backend *= lam_backend/lam

        # Translate (i.e., roll) backend phase to same center pixel as the
        # new data.
        deltax = pr_params['XYcen_PIL'][0] - pr_params_backend['XYcen_PIL'][0]
        deltay = pr_params['XYcen_PIL'][1] - pr_params_backend['XYcen_PIL'][1]
        ph_backend = np.roll(ph_backend, (deltay, deltax), axis=(0, 1))

        # Make back-end phase array the same size
        ph_backend = inin(ph_backend, phwrap.shape)

    else:
        ph_backend = np.zeros_like(phwrap)

    # 2. Subtract back-end phase from end-to-end phase (in E-field to avoid
    # issues with unwrapping making artifacts before Fresnel propagation)
    # to get front-end E-field at PIL.
    efield_pil = amp * np.exp(1j*(phwrap - ph_backend))

    # 3. Remove piston/tip, and tilt from the front-end E-field at PIL plane.
    ph_no_ptt, bmask, x_offset, y_offset = remove_ptt.remove_ptt_at_focus(
        np.abs(efield_pil),
        np.angle(efield_pil),
        ppl,
        diam_pupil_pix,
        ft_dir,
        fn_ref_psf,
        fn_psffit_tuning=fn_psffit_tuning,
        nCrop=nCrop,
        nBin=nBin,
        nIter=nIter,
    )

    # 4. Fresnel propagate from the PIL plane to the DM1 (i.e., pupil) plane.
    # Fresnel propagate
    # diam_pup_pix = 300
    # diam_pup_m = 46.3e-3
    pixpermeter = diam_pupil_pix/diam_pupil_m
    if deltaz != 0:
        efield_pil = amp * np.exp(1j*ph_no_ptt)
        efield_pupil = fresnelprop(
            e=efield_pil,
            lam=lam,
            z=deltaz,
            nxfresnel=nxfresnel,
            pixpermeter=pixpermeter,
        )

        # 5. Convert E-field back to amplitude and unwrapped phase.
        phwrap_pupil = np.angle(efield_pupil)
        amp_pupil = np.abs(efield_pupil)
        bmask_pupil = ampthresh(pupilMap=amp, nBin=nBin)
    else:
        phwrap_pupil = ph_no_ptt
        amp_pupil = amp
        bmask_pupil = bmask

    # Remove isolated islands of zeroes in the software mask, which usually
    # happen from dead actuators where we want to keep the phase.
    bmask_pupil = np.logical_not(skimage.morphology.remove_small_objects(
        np.logical_not(bmask_pupil), min_size=min_size_zeros))

    # Unwrap the phase only after propagating back to the pupil to avoid
    # incorrectly masked spots from dead DM actuators.
    ph_pupil, _ = unwrap_segments(
        phwrap_pupil, amp_pupil, bMask=bmask_pupil, selem=None)

    return amp_pupil, ph_pupil, bmask_pupil, x_offset, y_offset, lam


def get_backend_phase(
        lyot, phwrap, pr_params, fn_pr_prop_params,
        subtract_backend=True,
        data_path_backend=BACKEND_PR_PATH):
    """
    Process and return the phase for the model to use at the Lyot plane.

    Parameters
    ----------
    lyot : array_like
        2-D amplitude map for the Lyot stop as made already for the HOWFSC
        model.
    phwrap : array_like
        2-D wrapped map of the phase. Preference is for the raw, wrapped phase
        instead of the unwrapped phase to avoid spuriously masked regions that
        will cause issues.
    pr_params : dict
        Dictionary of the tuning parameters used in the phase retrieval
    fn_pr_prop_params : str
        Path and filename of the YAML file containing a dictionary of tuning
        parameters used for Fresnel propagating the phase retrieved E-field
        back to the pupil from the PIL plane.

    Keyword Parameters
    ------------------
    subtract_backend : bool
        Whether to subtract off the measured back-end phase of the instrument.
        Default is True.
    data_path_backend : str
        Path of the folder containing the backend phase retrieval output files.
        The default folder is in the repo and contains data taken during TVAC.

    Returns
    -------
    ph_pupil : array_like
        2-D map of the unwrapped backend phase.

    """
    check.twoD_array(amp, 'amp', TypeError)
    check.real_array(amp, 'amp', TypeError)
    check.twoD_array(phwrap, 'phwrap', TypeError)
    check.real_array(phwrap, 'phwrap', TypeError)
    if phwrap.shape != amp.shape:
        raise ValueError('phwrap must have same shape as amp')
    check.dictionary(pr_params, 'pr_params', TypeError)
    check.boolean(subtract_backend, 'subtract_backend', TypeError)

    lam_dict = wavelength_lut.get()
    lam = lam_dict[pr_params['bandpass'].lower()]

    PROP_PARAMS_DIR = os.path.dirname(os.path.abspath(fn_pr_prop_params))
    input_dict = loadyaml(fn_pr_prop_params)
    ppl = input_dict['ppl']
    fn_ref_psf = input_dict['fn_ref_psf']
    if not os.path.isabs(fn_ref_psf):
        fn_ref_psf = os.path.join(PROP_PARAMS_DIR, fn_ref_psf)
    deltaz = input_dict['deltaz']
    nxfresnel = input_dict['nxfresnel']
    diam_pupil_pix = input_dict['diam_pupil_pix']
    diam_pupil_m = input_dict['diam_pupil_m']
    ft_dir = input_dict['ft_dir']
    fn_psffit_tuning = input_dict['fn_psffit_tuning']
    if not os.path.isabs(fn_psffit_tuning):
        fn_psffit_tuning = os.path.join(PROP_PARAMS_DIR, fn_psffit_tuning)
    nCrop = input_dict['nCrop']
    nBin = input_dict['nBin']
    nIter = input_dict['nIter']
    min_size_zeros = input_dict['min_size_zeros']

    # 1. Load and modify the backend phase map

    # Load back-end PR data
    _, ph_backend, _, _, pr_params_backend = load_backend_pr(
        data_path=data_path_backend)
    lam_dict = wavelength_lut.get()
    lam_backend = lam_dict[pr_params_backend['bandpass'].lower()]

    # Scale backend phase with wavelength because it was taken in Band 1B
    # only.
    ph_backend *= lam_backend/lam

    # Translate (i.e., roll) backend phase to same center pixel as the
    # new data.
    deltax = pr_params['XYcen_PIL'][0] - pr_params_backend['XYcen_PIL'][0]
    deltay = pr_params['XYcen_PIL'][1] - pr_params_backend['XYcen_PIL'][1]
    ph_backend = np.roll(ph_backend, (deltay, deltax), axis=(0, 1))

    # Make back-end phase array the same size
    amp_backend = inin(lyot, phwrap.shape)
    ph_backend = inin(ph_backend, phwrap.shape)

    # 2. 
    # efield_lyot = lyot * np.exp(1j*ph_backend)

    # 3. Remove piston/tip, and tilt from the Lyot plane's complex
    # transmission.
    ph_no_ptt, bmask, x_offset, y_offset = remove_ptt.remove_ptt_at_focus(
        amp_backend,
        ph_backend,
        ppl,
        diam_pupil_pix,
        ft_dir,
        fn_ref_psf,
        fn_psffit_tuning=fn_psffit_tuning,
        nCrop=nCrop,
        nBin=nBin,
        nIter=nIter,
    )

    # Remove isolated islands of zeroes in the software mask, which usually
    # happen from dead actuators where we want to keep the phase.
    bmask_pupil = np.logical_not(skimage.morphology.remove_small_objects(
        np.logical_not(bmask), min_size=min_size_zeros))

    # Unwrap the phase only after propagating back to the pupil to avoid
    # incorrectly masked spots from dead DM actuators.
    ph_pupil, _ = unwrap_segments(
        ph_no_ptt, amp, bMask=bmask, selem=None)

    return ph_pupil
#     return ph_pupil, bmask, x_offset, y_offset, lam


