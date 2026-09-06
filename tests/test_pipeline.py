"""Integrationstest fuer chamber_obb.pipeline.analyze (Story 5).

``run_inference`` wird per ``monkeypatch`` ersetzt -- kein echtes Modell, kein
Checkpoint-Download, kein Netzwerkzugriff. Deckt die I/O-Matrix der Story ab:
Array-Input (mit Normalisierungsnachweis), Pfad-Input, fehlender Checkpoint,
ungueltiger Bildtyp, null Detektionen.
"""

import numpy as np
import pytest
import tifffile

import chamber_obb
from chamber_obb.inference import DEFAULT_CONFIDENCE
from chamber_obb.pipeline import analyze


class _FakeOBB:
    """Minimaler Stand-in fuer ultralytics' ``Results.obb`` -- kein Modell/GPU noetig."""

    def __init__(self, xyxyxyxy, conf, cls):
        self.xyxyxyxy = xyxyxyxy
        self.conf = conf
        self.cls = cls

    def __len__(self):
        return len(self.cls)


class _FakeResults:
    def __init__(self, obb, names):
        self.obb = obb
        self.names = names


def _one_detection_results():
    # Corners placed inside the unpadded letterbox region for a 600x1200 (h x w)
    # original: scale=8/15, new_width=640, new_height=320, pad_top=160 -- so a
    # box between y=200 and y=300 lands within the real (non-padding) content band.
    corners = np.array([[[100.0, 200.0], [300.0, 200.0], [300.0, 300.0], [100.0, 300.0]]])
    return _FakeResults(
        obb=_FakeOBB(corners, np.array([0.9]), np.array([0])),
        names={0: "chamber_region", 1: "channel_region"},
    )


def _zero_detection_results():
    return _FakeResults(
        obb=_FakeOBB(np.zeros((0, 4, 2)), np.zeros((0,)), np.zeros((0,))),
        names={0: "chamber_region", 1: "channel_region"},
    )


def test_analyze_array_input_normalizes_before_inference(monkeypatch):
    """16-bit 2D-Array + checkpoint_path -> Normalisierung laeuft, Ergebnis ist list[Detection]."""
    captured = {}

    def fake_run_inference(image, checkpoint_path, conf):
        captured["image"] = image
        captured["checkpoint_path"] = checkpoint_path
        captured["conf"] = conf
        return _one_detection_results()

    monkeypatch.setattr("chamber_obb.pipeline.run_inference", fake_run_inference)

    array = np.random.default_rng(0).integers(0, 65535, size=(600, 1200), dtype=np.uint16)
    detections = analyze(array, checkpoint_path="/fake/checkpoint.pt")

    # Proves normalization ran: the image handed to run_inference is 640x640x3 uint8,
    # not the raw 2D 16-bit input.
    assert captured["image"].shape == (640, 640, 3)
    assert captured["image"].dtype == np.uint8
    assert captured["checkpoint_path"] == "/fake/checkpoint.pt"
    assert captured["conf"] == DEFAULT_CONFIDENCE

    assert len(detections) == 1
    det = detections[0]
    assert det.cls_name == "chamber_region"
    # Hand-computed inverse-letterbox: 600x1200 (h x w) original -> scale=8/15,
    # pad_left=0, pad_top=160. orig = (letterboxed - pad) / scale, applied to
    # the synthetic box [[100,200],[300,200],[300,300],[100,300]] from
    # _one_detection_results().
    expected_corners = np.array(
        [[187.5, 75.0], [562.5, 75.0], [562.5, 262.5], [187.5, 262.5]]
    )
    assert det.corners == pytest.approx(expected_corners)


def test_analyze_path_input_reads_checkpoint_from_env_var(monkeypatch, tmp_path):
    """.tiff-Pfad + CHAMBER_OBB_CHECKPOINT env var (kein Param) -> gleiche Pipeline/Rueckgabeform."""
    monkeypatch.setattr("chamber_obb.pipeline.run_inference", lambda *a, **kw: _one_detection_results())
    monkeypatch.setenv("CHAMBER_OBB_CHECKPOINT", "/fake/env-checkpoint.pt")

    tiff_path = tmp_path / "frame.tiff"
    image = np.random.default_rng(1).integers(0, 255, size=(480, 640), dtype=np.uint8)
    tifffile.imwrite(tiff_path, image)

    detections = analyze(tiff_path)

    assert len(detections) == 1
    assert detections[0].cls_name == "chamber_region"


def test_analyze_checkpoint_path_param_takes_precedence_over_env_var(monkeypatch):
    """Beide gesetzt -> der explizite Param gewinnt, nicht die Umgebungsvariable."""
    monkeypatch.setenv("CHAMBER_OBB_CHECKPOINT", "/env/checkpoint.pt")

    captured = {}

    def fake_run_inference(image, checkpoint_path, conf):
        captured["checkpoint_path"] = checkpoint_path
        return _one_detection_results()

    monkeypatch.setattr("chamber_obb.pipeline.run_inference", fake_run_inference)

    array = np.zeros((640, 640, 3), dtype=np.uint8)
    analyze(array, checkpoint_path="/param/checkpoint.pt")

    assert captured["checkpoint_path"] == "/param/checkpoint.pt"


def test_analyze_raises_value_error_when_no_checkpoint_anywhere(monkeypatch):
    """Weder Param noch env var gesetzt -> ValueError, bevor inference.py beruehrt wird."""
    monkeypatch.delenv("CHAMBER_OBB_CHECKPOINT", raising=False)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("run_inference must not be called when no checkpoint is resolved")

    monkeypatch.setattr("chamber_obb.pipeline.run_inference", fail_if_called)

    array = np.zeros((640, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        analyze(array)


def test_analyze_raises_value_error_for_invalid_image_type(monkeypatch):
    """image=12345 -> ValueError, die den ungueltigen Typ benennt; keine Verarbeitung."""

    def fail_if_called(*args, **kwargs):
        raise AssertionError("run_inference must not be called for an invalid image type")

    monkeypatch.setattr("chamber_obb.pipeline.run_inference", fail_if_called)

    with pytest.raises(ValueError, match="int"):
        analyze(12345, checkpoint_path="/fake/checkpoint.pt")


def test_analyze_returns_empty_list_for_zero_detections(monkeypatch):
    """Mocked run_inference liefert null Zeilen -> analyze() gibt [] zurueck."""
    monkeypatch.setattr("chamber_obb.pipeline.run_inference", lambda *a, **kw: _zero_detection_results())

    array = np.zeros((640, 640, 3), dtype=np.uint8)
    detections = analyze(array, checkpoint_path="/fake/checkpoint.pt")

    assert detections == []


def test_analyze_already_3_channel_uint8_array_passes_through_end_to_end(monkeypatch):
    """Ein bereits 3-kanaliges uint8-Array durchlaeuft die volle Pipeline unveraendert-normalisiert.

    Beweist den idempotenten Normalisierungspfad end-to-end (nicht nur isoliert
    in ``normalize_array``): kein erneutes Stapeln, kein erneutes 8-bit-Clipping.
    """
    captured = {}

    def fake_run_inference(image, checkpoint_path, conf):
        captured["image"] = image
        return _one_detection_results()

    monkeypatch.setattr("chamber_obb.pipeline.run_inference", fake_run_inference)

    array = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = analyze(array, checkpoint_path="/fake/checkpoint.pt")

    assert captured["image"].shape == (640, 640, 3)
    assert captured["image"].dtype == np.uint8
    assert len(detections) == 1


def test_public_namespace_exposes_only_analyze_and_frame_metadata():
    """AD-2: ``__all__`` names exactly the public API -- not preprocessing/inference/postprocessing.

    Python's import machinery also binds submodules (``chamber_obb.pipeline`` etc.)
    as attributes of the package -- normal for any Python package (numpy, requests, ...)
    and not something ``__all__`` needs to suppress; ``__all__`` is the idiomatic,
    ``from chamber_obb import *``-respected contract for what's public (AD-2).
    """
    assert chamber_obb.__all__ == ["FrameMetadata", "analyze"]
    assert chamber_obb.analyze is analyze
    assert chamber_obb.FrameMetadata is not None
