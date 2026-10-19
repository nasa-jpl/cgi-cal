# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
function to find an amplitude threshold and a binary mask for
source and defocused source images. Regular ampthresh is for pupil type
images where there is a clear separation between image intensity and noise.
For PSF type images, the intensity histogram has no obvious separation.

Algorithm: assume most image pixels are dark+noise since the source image
takes up a small part of the sensor. Then peak of intensity histogram is
approx mean dark intensity, and standard deviation around histogram peak is
noise. Then choose as signal pixels with intensity > mean_dark +
Sigsig*sigma_noise, where Sigsig is a user parameter.
"""

import numpy as np
import scipy.ndimage as ImMorph

import cal.util.check as check

def PSFType(Im, **kwds):
    """
    Arguments:
      Im is a 2-d real array of image intenstiy

    Returns:
      bMask is a boolean mask = True where image intensity is > threshold
      thresh is the derived threshold

    Optional arguments:
      Nbins = 256, # of histogram bins
      Darkbins = 10,
      Sigsig = 2.0, # define signal = intensity > Sigsig * noise std
      NOpenSize = 5, # for morphological opening as a final step
    """

    Nbins = kwds.get('Nbins', 256)
    Darkbins = kwds.get('Darkbins', 10)
    Sigsig = kwds.get('Sigsig', 2.0)
    NOpenSize = kwds.get('NOpenSize', 5)

    check.twoD_array(Im, 'Image', TypeError)
    check.positive_scalar_integer(Nbins, 'Nbins', TypeError)
    check.positive_scalar_integer(Darkbins, 'Darkbins', TypeError)
    check.real_positive_scalar(Sigsig, 'Sigsig', TypeError)
    check.positive_scalar_integer(NOpenSize, 'NOpenSize', TypeError)


    cnt, Ibinedges = np.histogram(np.abs(Im), bins=Nbins)
    # assume most pixels are dark + noise
    imax = np.argmax(cnt)

    # assign dark = only pixels where Intensity is < something
    bV = np.abs(Im) < Ibinedges[imax+Darkbins]
    Idk = Im[bV]

    # # mean, std of dark pixels
    mu = np.mean(np.abs(Idk))
    sig = np.std(np.abs(Idk))

    # anything above dark + noise is signal
    thresh = mu + Sigsig*sig

    bMask = np.abs(Im) >= thresh

    # morphological opening to eliminate outlier 'salt' noise,
    # sort of assumes speckle lambda/D is more than 1 pixel
    # Matlab:
    #     bMask = imopen(bMask, strel('disk',3));
    N = NOpenSize
    bMask = ImMorph.binary_opening(bMask, structure=np.ones((N, N)))

    return bMask, thresh
