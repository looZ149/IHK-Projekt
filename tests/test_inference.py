"""Tests fuer die pre-load Validierungs-Zweige von chamber_obb.inference.run_inference (Story 6).

Deckt die drei retypten raise-Stellen direkt ab, ohne ueber ``analyze()`` zu
gehen -- alle drei liegen vor dem eigentlichen Modell-Ladevorgang, also kein
echtes Modell/Netzwerkzugriff noetig.
"""

import numpy as np
import pytest

from chamber_obb.errors import InvalidParameterError, ModelLoadError
from chamber_obb.inference import run_inference


def test_run_inference_raises_model_load_error_for_non_str_path_checkpoint():
    """checkpoint_path ist weder str noch Path -> ModelLoadError."""
    array = np.zeros((640, 640, 3), dtype=np.uint8)

    with pytest.raises(ModelLoadError, match="int"):
        run_inference(array, checkpoint_path=12345)


def test_run_inference_raises_model_load_error_for_nonexistent_checkpoint(tmp_path):
    """checkpoint_path zeigt auf keine existierende Datei -> ModelLoadError."""
    array = np.zeros((640, 640, 3), dtype=np.uint8)
    missing = tmp_path / "does-not-exist.pt"

    with pytest.raises(ModelLoadError, match=str(missing)):
        run_inference(array, checkpoint_path=missing)


def test_run_inference_raises_invalid_parameter_error_for_out_of_range_conf(tmp_path):
    """conf ausserhalb von [0, 1] -> InvalidParameterError."""
    array = np.zeros((640, 640, 3), dtype=np.uint8)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not a real checkpoint")

    with pytest.raises(InvalidParameterError, match="1.5"):
        run_inference(array, checkpoint_path=checkpoint, conf=1.5)
