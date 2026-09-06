"""Regression-Test fuer chamber_obb.preprocessing.normalize_array (Story 5).

Deckt den verifizierten Bug ab: die Kanal-Stapel-Bedingung in
``load_and_normalize``/``normalize_array`` war invertiert (``if img.ndim != 2``),
wodurch ein 2D-Graustufen-Array (der Normalfall nach Midframe-Extraktion aus
einem Multi-Frame-TIFF) still als 2D zurueckgegeben wurde statt als das von
CAP-1 geforderte 3-Kanal-RGB-Bild. Story 5 fixt die Bedingung zu
``if img.ndim == 2`` -- dieser Test reproduziert genau den Fehlerfall.
"""

import numpy as np

from chamber_obb.preprocessing import normalize_array


def test_normalize_array_stacks_2d_uint16_to_3_channel():
    """2D uint16-Array (post-Midframe-Extraktion-Form) -> 3-Kanal-Output."""
    img = np.random.default_rng(0).integers(0, 65535, size=(480, 640), dtype=np.uint16)

    result = normalize_array(img)

    assert result.ndim == 3
    assert result.shape[-1] == 3
    assert result.dtype == np.uint8


def test_normalize_array_leaves_already_3_channel_uint8_untouched_in_shape():
    """Ein bereits 3-kanaliges uint8-Bild bleibt 3-kanalig (kein erneutes Stapeln)."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    result = normalize_array(img)

    assert result.ndim == 3
    assert result.shape == (480, 640, 3)
    assert result.dtype == np.uint8
