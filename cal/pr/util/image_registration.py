# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
wrapper for skimage.registration.phase_cross_correlation()
because function prototype and option 'normalization' are
different for version 0.19.2 compared to prior releases
"""

import skimage

try: # 0.17.2 and later
    from skimage.registration import phase_cross_correlation

except ModuleNotFoundError: # try pre-0.17.2
    from skimage.feature import register_translation as phase_cross_correlation


# shift_dydx, _, _ = phase_cross_correlation(
#                 immeas, imcalc, upsample_factor=self.upsample_factor)

def cross_correlation(reference_image, moving_image, upsample_factor=100):
    """
    wrapper for:
       skimage.registration.phase_cross_correlation()
    or for skimage versions earlier than 0.17.2:
       skimage.feature.register_translation()

    for skimage versions 0.19. and later, must specify normalization=None

    inputs:
       reference_image, 2-d array of real values, typically intensity
       moving_image, 2-d array of real values, typically intensity
       upsample_factor, sub-pixel resolution to estimate image shift from correlation

    returns:
       shift_dydx = [dy, dx] (pixels) is the amount moving_image has to shift to match
         reference_image
    """

    # which version of skimage
    ver = skimage.__version__     # e.g. ver = '0.17.2'
    v0, v1, v2 = ver.split('.')   # e.g. v0 = '0', ...

    if int(v0) == 0 and int(v1) <= 18:
        # no 'normalization' keyword
        shift_dydx, _, _ = phase_cross_correlation(
            reference_image, moving_image,
            upsample_factor=upsample_factor
        )

    else:
        # need 'normalization' keyword
        shift_dydx, _, _ = phase_cross_correlation(
            reference_image, moving_image,
            upsample_factor=upsample_factor, normalization=None,
        )

    return shift_dydx
