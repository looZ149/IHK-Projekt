"""Tests fuer chamber_obb.dataset.build_dataset (Story 12: CAP-7, AD-12, AD-20, AD-23)."""

import math
import subprocess
import sys

import cv2
import pytest
from training_helpers import fake_open_sequence, make_dataset, tree_fingerprint, write_selection

import chamber_obb
from chamber_obb import locks
from chamber_obb.dataset import (
    assign_splits,
    build_dataset,
    propagate_roi_across_frames,
    rotated_box_to_corners,
    select_target_frames,
)
from chamber_obb.errors import DatasetBuildError, DatasetDescriptorError, InvalidParameterError
from chamber_obb.manifest import write_manifest


def _chips(root, *names, **kw):
    for name in names:
        write_selection(root, name, **kw)
    return root


def _build(tmp_path, **kw):
    kw.setdefault("open_sequence", fake_open_sequence())
    return build_dataset(tmp_path / "data", tmp_path / "new", **kw)


def test_rotated_box_corners_are_normalized_and_clipped():
    corners = rotated_box_to_corners(50, 40, 20, 20, 0, 100, 80)
    assert corners == pytest.approx([0.4, 0.375, 0.6, 0.375, 0.6, 0.625, 0.4, 0.625])
    assert all(0.0 <= c <= 1.0 for c in rotated_box_to_corners(99, 79, 60, 60, 33, 100, 80))


SHAPE = (80, 100)  # (H, W) des Fake-Frames


def test_propagation_same_frame_and_reference_frame_are_identity():
    roi = {"center": [50.0, 40.0], "size": [20, 10], "angle": 5.0}
    transforms = {3: {"dx": 5.0, "dy": -2.0, "theta": 10.0}}
    result = propagate_roi_across_frames(roi, 0, [0, 3, 9], transforms, 0, SHAPE)
    assert set(result) == {0, 3}  # Frame 9 hat keine Registrierung
    assert result[0]["center"] == pytest.approx([50.0, 40.0]) and result[0]["angle"] == pytest.approx(5.0)
    back = propagate_roi_across_frames({**roi, "center": result[3]["center"], "angle": result[3]["angle"]},
                                       3, [0], transforms, 0, SHAPE)
    assert back[0]["center"] == pytest.approx([50.0, 40.0], abs=1e-6)  # Rundreise Referenz -> 3 -> Referenz
    assert back[0]["angle"] == pytest.approx(5.0, abs=1e-6)
    assert result[3]["size"] == [20, 10]
    with pytest.raises(KeyError):  # Anker ohne Transform
        propagate_roi_across_frames(roi, 7, [0], transforms, 0, SHAPE)


def test_translation_only_transform_shifts_the_center_by_dx_dy():
    roi = {"center": [50.0, 40.0], "size": [20, 10], "angle": 0.0}
    transforms = {4: {"dx": 6.0, "dy": -3.0, "theta": 0.0}}
    moved = propagate_roi_across_frames(roi, 0, [4], transforms, 0, SHAPE)[4]
    assert moved["center"] == pytest.approx([56.0, 37.0]) and moved["angle"] == 0.0  # Referenz -> Frame 4 = +(dx, dy)


def test_geometry_matches_acia_apply_correction_to_spec():
    """Rundreise gegen acia selbst: Anker A -> Referenz (acia) == Ziel B -> Referenz (acia) fuer unsere Ziel-ROI."""
    registration = pytest.importorskip("acia.registration")
    from acia.base import RotatedCropSpec
    t_a = registration.FrameTransform(dx=3.0, dy=-2.0, theta=1.5)
    t_b = registration.FrameTransform(dx=-4.0, dy=5.0, theta=-2.0)
    roi = {"center": [50.0, 40.0], "size": [20, 10], "angle": 12.0}
    expected = registration.apply_correction_to_spec(RotatedCropSpec((50.0, 40.0), (20, 10), 12.0), t_a, shape=SHAPE)
    got = propagate_roi_across_frames(roi, 5, [7], {5: t_a.to_dict(), 7: t_b.to_dict()}, 0, SHAPE)[7]
    mapped = registration.apply_correction_to_spec(
        RotatedCropSpec(tuple(got["center"]), (20, 10), got["angle"]), t_b, shape=SHAPE)
    assert mapped.center == pytest.approx(expected.center, abs=1e-6)
    assert mapped.angle == pytest.approx(expected.angle, abs=1e-6)


@pytest.mark.parametrize("angle", [0.0, 25.0, -40.0, 90.0])
def test_label_corners_match_acia_crop_convention(angle):
    """Die Label-Ecken einer ROI muessen genau die Region sein, die acia's crop_rotated begradigt."""
    base = pytest.importorskip("acia.base")
    import cv2
    import numpy as np
    img_w, img_h, center, size = 200, 160, (100.0, 80.0), (60, 24)
    corners = np.array(rotated_box_to_corners(*center, *size, angle, img_w, img_h)).reshape(4, 2) * (img_w, img_h)
    canvas = np.zeros((img_h, img_w), np.uint8)
    cv2.fillPoly(canvas, [corners.round().astype(np.int32)], 255)
    crop = base.RotatedCropSequenceSource(None, base.RotatedCropSpec(center, size, angle))._warp(canvas)
    assert crop.shape == (24, 60)
    assert (crop > 127).mean() > 0.95  # Ecken und acia-Crop decken dieselbe Region ab (falsches Vorzeichen: << 0.9)


def test_select_target_frames_is_reproducible_and_keeps_anchor_frames():
    frames = set(range(20))
    a = select_target_frames(frames, [3, 9], 5, 42, "chip_pos0")
    assert a == select_target_frames(frames, [3, 9], 5, 42, "chip_pos0")
    assert 3 in a and 9 in a and len(a) == 5
    assert select_target_frames(frames, [0], None, 42) == sorted(frames)
    assert select_target_frames({0, 1}, [7], None, 42) == [0, 1, 7]  # Anker zaehlt auch ohne eigenen Transform


def test_assign_splits_is_deterministic_and_rejects_conflicts():
    names = [f"c{i}" for i in range(10)]
    a = assign_splits(names, 0.2, 0.2, 7, set(), set(), set())
    assert a == assign_splits(names, 0.2, 0.2, 7, set(), set(), set())
    assert sorted(set(a.values())) == ["test", "train", "val"]
    assert sum(1 for s in a.values() if s == "val") == 2
    forced = assign_splits(names, 0.2, 0.2, 7, {"c0"}, set(), {"c1"})
    assert forced["c0"] == "val" and forced["c1"] == "train"
    with pytest.raises(InvalidParameterError):
        assign_splits(names, 0.2, 0.2, 7, {"c0"}, {"c0"}, set())


def test_build_writes_session_level_splits_labels_and_descriptor(tmp_path):
    _chips(tmp_path / "data", *[f"chip{i}" for i in range(6)],
           rois=[{"center": [50.0, 40.0], "size": [20.0, 20.0], "angle": 0.0},
                 {"center": [50.0, 40.0], "size": [10.0, 40.0], "angle": 0.0}], positions=(0, 1))
    report = _build(tmp_path, val_ratio=0.34, test_ratio=0.17)
    assert set(report.sessions) == {f"chip{i}" for i in range(6)}
    assert sorted(set(report.sessions.values())) == ["test", "train", "val"]
    descriptor = locks.load_descriptor(tmp_path / "new")
    assert locks.check_consistency(tmp_path / "new", descriptor) == []
    on_disk, unattributable = locks.scan_sessions_on_disk(tmp_path / "new")
    assert not unattributable
    for split, names in on_disk.items():  # nie eine Session in zwei Splits
        assert all(report.sessions[n] == split for n in names)
    assert sum(len(v) for v in on_disk.values()) == 6
    assert descriptor["locked_train_sessions"] == sorted(n for n, s in report.sessions.items() if s == "train")
    split = report.sessions["chip0"]
    label = (tmp_path / "new" / "labels" / split / "chip0_pos1_frame000.txt").read_text().splitlines()
    assert [line.split()[0] for line in label] == ["0", "1"]  # Kammer + Kanal per Seitenverhaeltnis
    assert all(len(line.split()) == 9 for line in label)
    image = cv2.imread(str(tmp_path / "new" / "images" / split / "chip0_pos1_frame000.png"), cv2.IMREAD_UNCHANGED)
    assert image.dtype.name == "uint8" and image.shape == (80, 100)
    assert report.images_written[split] >= 2


def test_same_seed_gives_same_split(tmp_path):
    _chips(tmp_path / "data", *[f"chip{i}" for i in range(8)])
    first = _build(tmp_path, val_ratio=0.25, test_ratio=0.25)
    second = build_dataset(tmp_path / "data", tmp_path / "new2", open_sequence=fake_open_sequence(),
                           val_ratio=0.25, test_ratio=0.25)
    assert first.sessions == second.sessions


def test_additive_build_keeps_existing_splits_and_rebuild_keeps_locks(tmp_path):
    _chips(tmp_path / "data", "a", "b", "c", "d")
    first = _build(tmp_path, chips=["a", "b", "c"], val_ratio=0.34, test_ratio=0.34)
    write_selection(tmp_path / "data", "e")
    second = _build(tmp_path, chips=["a", "e"])
    assert second.sessions["a"] == first.sessions["a"]  # bestehende Session behaelt den Split
    descriptor = locks.load_descriptor(tmp_path / "new")
    assert set(descriptor["sessions"]) == {"a", "b", "c", "e"}
    ever_train = set(descriptor["locked_train_sessions"])
    third = _build(tmp_path, chips=["b", "d"], rebuild=True, val_ratio=0.5, test_ratio=0.0)
    assert third.rebuilt
    after = locks.load_descriptor(tmp_path / "new")
    assert set(after["sessions"]) == {"b", "d"}  # rebuild leert die Session-Tabelle ...
    assert ever_train <= set(after["locked_train_sessions"])  # ... aber nie die Sperrliste
    if "b" in ever_train:
        assert after["sessions"]["b"]["split"] == "train"


def test_old_dataset_locks_and_val_test_rules_and_read_only(tmp_path):
    old = make_dataset(tmp_path / "old", {"seen": "train", "oldval": "val", "oldtest": "test"})
    _chips(tmp_path / "data", "seen", "oldval", "fresh1", "fresh2", "fresh3")
    before = tree_fingerprint(old)
    report = _build(tmp_path, old_dataset=old, val_ratio=0.5, test_ratio=0.0)
    assert tree_fingerprint(old) == before  # old_dataset unveraendert (AD-20)
    assert report.sessions["seen"] == "train"  # bereits trainiert -> gesperrt
    assert "oldval" in report.skipped and "Val/Test" in report.skipped["oldval"]
    assert "oldval" not in locks.load_descriptor(tmp_path / "new")["sessions"]
    assert "seen" in report.locked


def test_all_trained_old_dataset_locks_its_monitor_sessions_instead_of_rejecting_them(tmp_path):
    old = make_dataset(tmp_path / "old", {"seen": "train", "monitor": "val", "monitor2": "test"}, all_trained=True)
    _chips(tmp_path / "data", "monitor", "monitor2", "fresh1", "fresh2", "fresh3")
    before = tree_fingerprint(old)
    report = _build(tmp_path, old_dataset=old, val_ratio=0.34, test_ratio=0.0)
    assert tree_fingerprint(old) == before
    assert report.sessions["monitor"] == "train" and report.sessions["monitor2"] == "train"  # schon trainiert
    assert not report.skipped
    assert "val" in set(report.sessions[n] for n in ("fresh1", "fresh2", "fresh3"))


def test_lineage_from_base_checkpoint_manifest_forces_train(tmp_path):
    run = tmp_path / "run"
    (run / "fit" / "weights").mkdir(parents=True)
    ckpt = run / "fit" / "weights" / "best.pt"
    ckpt.write_bytes(b"x")
    write_manifest(run / "manifest.json", {"schema_version": 1, "trained_sessions": ["c1", "c2"]})
    _chips(tmp_path / "data", "c1", "c2", "c3", "c4")
    report = _build(tmp_path, base_checkpoint=ckpt, val_ratio=0.5, test_ratio=0.25)
    assert report.sessions["c1"] == "train" and report.sessions["c2"] == "train"
    explicit = build_dataset(tmp_path / "data", tmp_path / "new2", open_sequence=fake_open_sequence(),
                             prior_manifest=run / "manifest.json", val_ratio=0.5, test_ratio=0.25)
    assert explicit.sessions["c1"] == "train"


def test_identity_conflicts_are_rejected(tmp_path):
    old = make_dataset(tmp_path / "old", {"known": "train"})
    write_selection(tmp_path / "data", "known", source="/acq/OTHER.nd2")  # gleicher Name, andere Quelle
    write_selection(tmp_path / "data", "renamed", source="/acq/known.nd2")  # gleiche Quelle, anderer Name
    write_selection(tmp_path / "data", "ok")
    report = _build(tmp_path, old_dataset=old)
    assert "andere Quelle" in report.skipped["known"]
    assert "gleiche Quelle" in report.skipped["renamed"]
    assert list(report.sessions) == ["ok"]
    second = _build(tmp_path, chips=["ok"])  # gleiche Quelle, gleicher Name: kein Konflikt
    assert "ok" in second.sessions and not second.skipped
    write_selection(tmp_path / "data", "ok2", source="/acq/ok.nd2")
    third = _build(tmp_path, chips=["ok2"])
    assert "gleiche Quelle" in third.skipped["ok2"]


def test_failed_chip_is_skipped_and_leaves_no_files(tmp_path):
    _chips(tmp_path / "data", "good", "bad")
    report = _build(tmp_path, open_sequence=fake_open_sequence(fail_for=("bad",)), val_ratio=0, test_ratio=0)
    assert list(report.sessions) == ["good"]
    assert "bad" in report.skipped and "OSError" in report.skipped["bad"]
    on_disk, _ = locks.scan_sessions_on_disk(tmp_path / "new")
    assert on_disk["train"] == {"good"}
    assert "bad" not in locks.load_descriptor(tmp_path / "new")["sessions"]


def test_unreadable_selection_and_missing_requested_chip_are_reported(tmp_path):
    (tmp_path / "data" / "broken").mkdir(parents=True)
    (tmp_path / "data" / "broken" / "selection.json").write_text("{nope")
    write_selection(tmp_path / "data", "fine")
    report = _build(tmp_path, chips=["broken", "fine", "ghost"], val_ratio=0, test_ratio=0)
    assert "unlesbar" in report.skipped["broken"]
    assert "ghost" in report.skipped
    assert list(report.sessions) == ["fine"]


def test_new_dataset_may_not_nest_with_old_dataset(tmp_path):
    old = make_dataset(tmp_path / "old", {"a": "train"})
    _chips(tmp_path / "data", "x")
    before = tree_fingerprint(old)
    for new in (old, old / "inner", tmp_path):
        with pytest.raises(InvalidParameterError, match="ineinander|gleich"):
            build_dataset(tmp_path / "data", new, old, open_sequence=fake_open_sequence())
    assert tree_fingerprint(old) == before and not (old / "inner").exists()


def test_old_dataset_without_descriptor_fails_closed(tmp_path):
    old = make_dataset(tmp_path / "old", {"a": "train"})
    (old / locks.DESCRIPTOR_FILENAME).unlink()
    _chips(tmp_path / "data", "x")
    with pytest.raises(DatasetDescriptorError):
        _build(tmp_path, old_dataset=old)


def test_existing_new_dataset_inconsistency_and_threshold_mismatch(tmp_path):
    _chips(tmp_path / "data", "a", "b")
    _build(tmp_path, chips=["a"])
    with pytest.raises(InvalidParameterError, match="channel_aspect_threshold"):
        _build(tmp_path, chips=["b"], channel_aspect_threshold=3.0)
    cv2.imwrite(str(tmp_path / "new" / "images" / "train" / "stray_pos0_frame000.png"), cv2.imread(
        str(next((tmp_path / "new" / "images").rglob("*.png"))), cv2.IMREAD_UNCHANGED))
    with pytest.raises(DatasetDescriptorError, match="inkonsistent"):
        _build(tmp_path, chips=["b"])


def _registration(frames, position=0, reference_frame=0, shift=1.0):
    return {"records": [{"position": position, "reference_frame": reference_frame, "transforms": {
        str(i): {"dx": shift * i, "dy": 0.0, "theta": 0.0} for i in frames}}]}


def test_registration_exports_multiple_frames(tmp_path):
    write_selection(tmp_path / "data", "reg", registration=_registration(range(4)))
    report = _build(tmp_path, use_registration=True, frames_per_position=None, val_ratio=0, test_ratio=0)
    assert report.images_written["train"] == 4
    label1 = (tmp_path / "new" / "labels" / "train" / "reg_pos0_frame001.txt").read_text().split()
    label0 = (tmp_path / "new" / "labels" / "train" / "reg_pos0_frame000.txt").read_text().split()
    assert float(label1[1]) == pytest.approx(float(label0[1]) + 1 / 100)  # x um dx=1 px verschoben


def test_roi_drawn_on_a_middle_frame_is_exported_from_that_frame(tmp_path):
    """Das Dashboard oeffnet auf dem MITTLEREN Frame: die ROI gehoert zu ihrem anchor_frame, nicht zu Frame 0."""
    from training_helpers import _Position
    write_selection(tmp_path / "data", "mid", anchor_frame=53)
    _Position.requested.clear()
    report = _build(tmp_path, val_ratio=0, test_ratio=0)
    assert _Position.requested == [53]
    assert (tmp_path / "new" / "images" / "train" / "mid_pos0_frame053.png").is_file()
    assert not (tmp_path / "new" / "images" / "train" / "mid_pos0_frame000.png").exists()
    assert report.images_written["train"] == 1 and not report.skipped


def test_missing_anchor_field_means_frame_zero(tmp_path):
    write_selection(tmp_path / "data", "old_format")  # ohne anchor_frame, wie acia bei Frame 0
    _build(tmp_path, val_ratio=0, test_ratio=0)
    assert (tmp_path / "new" / "images" / "train" / "old_format_pos0_frame000.png").is_file()


def test_registration_carries_a_mid_frame_roi_to_other_frames(tmp_path):
    write_selection(tmp_path / "data", "mid", anchor_frame=2, registration=_registration(range(5), shift=2.0))
    _build(tmp_path, use_registration=True, frames_per_position=None, val_ratio=0, test_ratio=0)
    labels = {f: (tmp_path / "new" / "labels" / "train" / f"mid_pos0_frame{f:03d}.txt").read_text().split() for f in (0, 2, 4)}
    assert float(labels[2][1]) == pytest.approx(0.4)  # auf dem Anker-Frame unveraendert (Mitte 50, Breite 20 -> 0.4)
    assert float(labels[4][1]) == pytest.approx(0.4 + 2 * 2.0 / 100)  # Frame 4 ist +4 px gegenueber dem Anker (Frame 2)
    assert float(labels[0][1]) == pytest.approx(0.4 - 2 * 2.0 / 100)


def test_mixed_anchor_frames_need_registration(tmp_path):
    rois = [{"center": [30.0, 40.0], "size": [20.0, 20.0], "angle": 0.0}, {"center": [70.0, 40.0], "size": [20.0, 20.0], "angle": 0.0}]
    write_selection(tmp_path / "data", "mixed", rois=rois, anchor_frame=[10, 20])
    write_selection(tmp_path / "data", "fine")
    report = _build(tmp_path, val_ratio=0, test_ratio=0)
    assert "mixed" in report.skipped and "Registrierung" in report.skipped["mixed"]  # sonst waeren ROIs unbeschriftet
    assert list(report.sessions) == ["fine"]


def test_anchor_without_transform_skips_the_position(tmp_path):
    write_selection(tmp_path / "data", "gap", anchor_frame=9, registration=_registration(range(4)))
    report = _build(tmp_path, use_registration=True, val_ratio=0, test_ratio=0)
    assert "gap" in report.skipped and "Transform" in report.skipped["gap"]


def test_missing_registration_falls_back_to_the_anchor_frame(tmp_path):
    write_selection(tmp_path / "data", "noreg", anchor_frame=7)
    report = _build(tmp_path, use_registration=True, frames_per_position=None, val_ratio=0, test_ratio=0)
    assert report.images_written["train"] == 1
    assert (tmp_path / "new" / "images" / "train" / "noreg_pos0_frame007.png").is_file()


def test_small_dataset_warns_about_missing_val_and_test(tmp_path):
    _chips(tmp_path / "data", "only")
    report = _build(tmp_path)
    assert any("val" in w for w in report.warnings) and any("test" in w for w in report.warnings)


def test_paths_fall_back_to_environment_variables(tmp_path, monkeypatch):
    old = make_dataset(tmp_path / "old", {"a": "train"})
    _chips(tmp_path / "data", "x")
    monkeypatch.setenv("CHAMBER_OBB_NEW_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("CHAMBER_OBB_NEW_DATASET", str(tmp_path / "new"))
    monkeypatch.setenv("CHAMBER_OBB_OLD_DATASET", str(old))
    report = build_dataset(open_sequence=fake_open_sequence(), val_ratio=0, test_ratio=0)
    assert report.new_dataset == (tmp_path / "new").resolve() and "x" in report.sessions
    monkeypatch.delenv("CHAMBER_OBB_NEW_DATA")
    with pytest.raises(InvalidParameterError, match="new_data"):
        build_dataset(new_dataset=tmp_path / "new", open_sequence=fake_open_sequence())


@pytest.mark.parametrize("kwargs", [{"val_ratio": 0.7, "test_ratio": 0.5}, {"channel_aspect_threshold": 1.0}])
def test_invalid_parameters(tmp_path, kwargs):
    _chips(tmp_path / "data", "x")
    with pytest.raises(InvalidParameterError):
        _build(tmp_path, **kwargs)


def test_acia_is_lazy_and_missing_acia_gives_a_clear_error(tmp_path, monkeypatch):
    code = ("import sys; sys.modules['acia']=None; import chamber_obb; "
            "assert not any(m.startswith('acia') and sys.modules[m] for m in sys.modules); print('ok')")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.stdout.strip() == "ok", result.stderr
    _chips(tmp_path / "data", "x")
    monkeypatch.setitem(sys.modules, "acia", None)
    monkeypatch.setitem(sys.modules, "acia.segm", None)
    monkeypatch.setitem(sys.modules, "acia.segm.open", None)
    with pytest.raises(DatasetBuildError, match="acia"):
        build_dataset(tmp_path / "data", tmp_path / "new")
    assert chamber_obb.build_dataset is build_dataset
