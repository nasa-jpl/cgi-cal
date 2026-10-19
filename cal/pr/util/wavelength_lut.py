# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Function to return the bandpass center dictionary."""


def get():
    """Subband wavelength table from cgisim_v3.0.7a.pdf."""

    wavelength_lut = {
            '1': 575e-9,
            '1a': 556e-9,
            '1b': 575e-9,
            '1c': 594e-9,
            '2': 660e-9,
            '2a': 615e-9,
            '2b': 638e-9,
            '2c': 656e-9,
            '3': 730e-9,
            '3a': 681e-9,
            '3b': 704e-9,
            '3c': 727e-9,
            '3d': 754e-9,
            '3e': 778e-9,
            '4': 825e-9,
            '4a': 792e-9,
            '4b': 825e-9,
            '4c': 857e-9,
        }

    return wavelength_lut
