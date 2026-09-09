"""Integrationstest fuer chamber_obb.dashboard.open_annotated_dashboard.

Deckt die I/O-Matrix der Spec ab: Chambers-/Channels-Happy-Path, null
Detektionen an einer Position, ``analyze()``-Fehler an einer Position
(isoliert, AD-11), acia-Oeffnungsfehler (gewrappt, AD-7), fehlender
Checkpoint (unveraendert propagiert, AD-7), sowie CAP-1s Rundreise-Test
und die AD-3-Regression fuer ``chamber_obb.__init__``.

``analyze()`` wird per ``monkeypatch.setattr("chamber_obb.dashboard.analyze",
...)`` gestubbt -- kein echtes YOLO-Modell, kein Checkpoint-Download. Die
synthetische Multi-Positions-Akquisition wird als ``tiff_folder`` (ein
Unterordner je Position, ein TIFF je Zeitpunkt) gebaut und ueber das echte
``acia.segm.open.open_sequence`` geoeffnet, damit die acia-Haelfte der
Integration (Frame-Lesen, ``RotatedCropSpec``, ``SequenceDashboard.resume()``)
real durchlaufen wird.
"""

import logging
import math

import numpy as np
import pytest
import tifffile

pytest.importorskip("acia")
pytest.importorskip("anywidget")

import chamber_obb
import chamber_obb.dashboard
import chamber_obb.inference
from chamber_obb.dashboard import _corners_to_rotated_crop_spec, open_annotated_dashboard
from chamber_obb.errors import (
    ChamberOBBError,
    DashboardIntegrationError,
    InvalidImageError,
    InvalidParameterError,
    ModelLoadError,
)
from chamber_obb.postprocessing import Detection

_H, _W = 16, 20


def _make_acquisition(tmp_path, n_positions: int, n_timepoints: int):
    """Builds ``<tmp_path>/acquisition/pos<i>/t<j>.tif`` for every position/frame.

    Opened via ``acia.segm.open.open_sequence`` as a nested ``tiff_folder``
    (one subfolder per position) -- the same generic interface real .nd2/.czi
    acquisitions expose, per the architecture's "never .nd2-specific" rule.
    """
    root = tmp_path / "acquisition"
    rng = np.random.default_rng(0)
    for p in range(n_positions):
        pos_dir = root / f"pos{p}"
        pos_dir.mkdir(parents=True)
        for t in range(n_timepoints):
            image = rng.integers(0, 255, size=(_H, _W), dtype=np.uint8)
            tifffile.imwrite(pos_dir / f"t{t}.tif", image)
    return str(root)


def _detection(cls_name="chamber_region", offset=0):
    corners = np.array(
        [[2.0 + offset, 2.0], [10.0 + offset, 2.0], [10.0 + offset, 9.0], [2.0 + offset, 9.0]]
    )
    return Detection(cls_id=0, cls_name=cls_name, confidence=0.9, corners=corners)


def _by_position(selections):
    by_pos: dict[int, list] = {}
    for sel in selections:
        by_pos.setdefault(sel["position"], []).append(sel)
    return by_pos


def test_chambers_happy_path_one_selection_per_position(monkeypatch, tmp_path):
    """N Positionen, 1 Detektion/Position -> .selections hat genau 1 Eintrag je
    Position, jeweils mit anchor_frame == der aufgeloeste Frame-Index (AD-5)."""
    path = _make_acquisition(tmp_path, n_positions=3, n_timepoints=3)  # num_t//2 == 1

    monkeypatch.setattr("chamber_obb.dashboard.analyze", lambda *a, **kw: [_detection()])

    dash = open_annotated_dashboard(path, checkpoint_path="/fake/checkpoint.pt", roi_mode="single")

    by_pos = _by_position(dash.selections)
    assert set(by_pos) == {0, 1, 2}
    for sels in by_pos.values():
        assert len(sels) == 1
        assert sels[0]["anchor_frame"] == 1


def test_channels_happy_path_unfiltered_counts_per_position(monkeypatch, tmp_path):
    """3-5 Detektionen/Position -> .selections hat exakt diese Anzahl je
    Position, ohne Filterung (AD-10)."""
    path = _make_acquisition(tmp_path, n_positions=3, n_timepoints=1)
    expected_counts = [3, 4, 5]
    calls = {"n": 0}

    def fake_analyze(*args, **kwargs):
        count = expected_counts[calls["n"]]
        calls["n"] += 1
        return [_detection(cls_name="channel_region", offset=i) for i in range(count)]

    monkeypatch.setattr("chamber_obb.dashboard.analyze", fake_analyze)

    dash = open_annotated_dashboard(path, checkpoint_path="/fake/checkpoint.pt", roi_mode="multi")

    by_pos = _by_position(dash.selections)
    assert [len(by_pos[p]) for p in sorted(by_pos)] == expected_counts


def test_zero_detections_at_a_position_yields_no_selections(monkeypatch, tmp_path):
    """Modell findet an einer Position nichts oberhalb conf -> null
    RoiSelections dort, kein Fehler (AD-10)."""
    path = _make_acquisition(tmp_path, n_positions=2, n_timepoints=1)
    responses = [[], [_detection()]]
    calls = {"n": 0}

    def fake_analyze(*args, **kwargs):
        response = responses[calls["n"]]
        calls["n"] += 1
        return response

    monkeypatch.setattr("chamber_obb.dashboard.analyze", fake_analyze)

    dash = open_annotated_dashboard(path, checkpoint_path="/fake/checkpoint.pt")

    by_pos = _by_position(dash.selections)
    assert 0 not in by_pos
    assert len(by_pos[1]) == 1


def test_analyze_failure_for_one_position_is_isolated(monkeypatch, tmp_path, caplog):
    """analyze() wirft fuer eine Position -> geloggte Warnung, null
    Detektionen dort, Batch/Dashboard laeuft fuer den Rest weiter (AD-11)."""
    path = _make_acquisition(tmp_path, n_positions=2, n_timepoints=1)
    calls = {"n": 0}

    def fake_analyze(*args, **kwargs):
        i = calls["n"]
        calls["n"] += 1
        if i == 0:
            raise InvalidImageError("corrupt frame")
        return [_detection()]

    monkeypatch.setattr("chamber_obb.dashboard.analyze", fake_analyze)

    with caplog.at_level(logging.WARNING, logger="chamber_obb.dashboard"):
        dash = open_annotated_dashboard(path, checkpoint_path="/fake/checkpoint.pt")

    by_pos = _by_position(dash.selections)
    assert 0 not in by_pos
    assert len(by_pos[1]) == 1
    assert any("position 0" in record.message for record in caplog.records)


def test_acia_open_failure_raises_dashboard_integration_error(tmp_path):
    """open_sequence() scheitert (kein Reader fuer die Endung) ->
    DashboardIntegrationError, gefangen von ``except ChamberOBBError:`` (AD-7)."""
    bad_path = str(tmp_path / "not_a_sequence.xyz")

    with pytest.raises(DashboardIntegrationError):
        open_annotated_dashboard(bad_path, checkpoint_path="/fake/checkpoint.pt")

    # AD-7's contract: the guaranteed-complete catch-all still works.
    try:
        open_annotated_dashboard(bad_path, checkpoint_path="/fake/checkpoint.pt")
        raise AssertionError("expected DashboardIntegrationError")
    except ChamberOBBError:
        pass


def test_no_checkpoint_configured_propagates_model_load_error_unchanged(monkeypatch, tmp_path):
    """Kein checkpoint_path, keine env var -> analyze()'s eigener
    ModelLoadError propagiert unveraendert, wird nicht zu
    DashboardIntegrationError gewrappt und nicht pro Position verschluckt (AD-7)."""
    path = _make_acquisition(tmp_path, n_positions=1, n_timepoints=1)
    monkeypatch.delenv("CHAMBER_OBB_CHECKPOINT", raising=False)

    # Real chamber_obb.pipeline.analyze() -- no mocking -- it raises
    # ModelLoadError itself, before any (nonexistent) checkpoint is touched.
    with pytest.raises(ModelLoadError):
        open_annotated_dashboard(path, checkpoint_path=None)


def test_corners_to_rotated_crop_spec_round_trip():
    """CAP-1: ein bekannter Eckpunkt-Satz -> RotatedCropSpec -> rekonstruierte
    Eckpunkte stimmen (innerhalb Toleranz) mit dem Original ueberein."""
    center = (50.0, 40.0)
    width, height, angle_deg = 12, 8, 30.0
    theta = math.radians(angle_deg)
    u = np.array([math.cos(theta), math.sin(theta)])
    v = np.array([-math.sin(theta), math.cos(theta)])
    corners = np.array(
        [
            center - width / 2 * u - height / 2 * v,
            center + width / 2 * u - height / 2 * v,
            center + width / 2 * u + height / 2 * v,
            center - width / 2 * u + height / 2 * v,
        ]
    )

    spec = _corners_to_rotated_crop_spec(corners)

    assert spec.center == pytest.approx(center)
    assert spec.size == (width, height)
    assert spec.angle == pytest.approx(angle_deg)

    cx, cy = spec.center
    w, h = spec.size
    rtheta = math.radians(spec.angle)
    ru = np.array([math.cos(rtheta), math.sin(rtheta)])
    rv = np.array([-math.sin(rtheta), math.cos(rtheta)])
    reconstructed = np.array(
        [
            [cx, cy] - w / 2 * ru - h / 2 * rv,
            [cx, cy] + w / 2 * ru - h / 2 * rv,
            [cx, cy] + w / 2 * ru + h / 2 * rv,
            [cx, cy] - w / 2 * ru + h / 2 * rv,
        ]
    )
    assert reconstructed == pytest.approx(corners, abs=1e-6)


def test_negative_preview_frame_raises_invalid_parameter_error(monkeypatch, tmp_path):
    """Negatives preview_frame wird abgelehnt statt still auf 0 geklemmt zu
    werden -- wie das echte SequenceDashboard.__init__ (traitlets.TraitError),
    das ebenfalls nicht klemmt."""
    path = _make_acquisition(tmp_path, n_positions=1, n_timepoints=3)
    monkeypatch.setattr("chamber_obb.dashboard.analyze", lambda *a, **kw: [_detection()])

    with pytest.raises(InvalidParameterError):
        open_annotated_dashboard(path, checkpoint_path="/fake/checkpoint.pt", preview_frame=-1)


def test_default_confidence_matches_inference_default():
    """dashboard.py haengt einwaerts nur von pipeline/errors ab (Architecture
    Spine) und darf chamber_obb.inference nicht importieren, weshalb
    _DEFAULT_CONFIDENCE von Hand dupliziert ist -- dieser Test haelt die
    Dopplung synchron, statt sie unbemerkt auseinanderlaufen zu lassen."""
    assert chamber_obb.dashboard._DEFAULT_CONFIDENCE == chamber_obb.inference.DEFAULT_CONFIDENCE


def test_public_namespace_still_exposes_only_analyze_and_frame_metadata():
    """AD-3 Regression: chamber_obb/__init__.py ist von dieser Aenderung
    unberuehrt -- weiterhin exakt {analyze, FrameMetadata}."""
    assert chamber_obb.__all__ == ["FrameMetadata", "analyze"]
