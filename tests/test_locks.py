"""Tests fuer classes.py, locks.py und manifest.py (Story 11: AD-12, AD-13, AD-18, AD-23, AD-24)."""

import json

import pytest
from training_helpers import make_dataset

from chamber_obb import locks
from chamber_obb.classes import CLASS_NAMES, class_list, classify_roi_by_shape
from chamber_obb.errors import ChamberOBBError, DatasetDescriptorError, ManifestError
from chamber_obb.manifest import find_manifest, load_manifest, trained_sessions_of, write_manifest


def test_class_list_is_ordered_and_copied():
    assert class_list() == ["chamber_region", "channel_region"] == list(CLASS_NAMES)
    class_list().append("x")
    assert class_list() == list(CLASS_NAMES)


def test_classify_by_aspect_ratio_with_overridable_threshold():
    assert classify_roi_by_shape(20, 20) == 0
    assert classify_roi_by_shape(10, 40) == 1
    assert classify_roi_by_shape(40, 10) == 1
    assert classify_roi_by_shape(20, 40) == 0  # 2.0 < 2.5
    assert classify_roi_by_shape(20, 40, threshold=1.5) == 1


def test_new_errors_share_the_base():
    for name in ("DatasetDescriptorError", "DatasetBuildError", "ClassListMismatchError", "LeakageError",
                 "EmptySplitError", "TrainingError", "ManifestError", "InferenceError"):
        import chamber_obb.errors as errors
        assert issubclass(getattr(errors, name), ChamberOBBError)


def test_descriptor_round_trip_and_hash(tmp_path):
    make_dataset(tmp_path / "ds", {"a": "train", "b": "val"})
    descriptor = locks.load_descriptor(tmp_path / "ds")
    assert descriptor["class_list"] == class_list()
    assert descriptor["sessions"]["b"] == {"split": "val", "source_path": "/acq/b.nd2"}
    assert descriptor["locked_train_sessions"] == ["a"]
    assert len(locks.descriptor_hash(tmp_path / "ds")) == 64


def test_empty_dir_has_no_descriptor_but_images_without_one_fail_closed(tmp_path):
    (tmp_path / "empty").mkdir()
    assert locks.load_descriptor(tmp_path / "empty") is None
    make_dataset(tmp_path / "ds", {"a": "train"})
    (tmp_path / "ds" / locks.DESCRIPTOR_FILENAME).unlink()
    with pytest.raises(DatasetDescriptorError, match="keine dataset.json"):
        locks.load_descriptor(tmp_path / "ds")


@pytest.mark.parametrize("payload", ["{not json", "[]", json.dumps({"schema_version": 99}),
                                     json.dumps({"schema_version": 1})])
def test_corrupt_descriptor_is_an_error_never_empty(tmp_path, payload):
    make_dataset(tmp_path / "ds", {"a": "train"})
    (tmp_path / "ds" / locks.DESCRIPTOR_FILENAME).write_text(payload)
    with pytest.raises(DatasetDescriptorError):
        locks.load_descriptor(tmp_path / "ds")


def test_write_is_atomic_and_leaves_no_temp_files(tmp_path):
    make_dataset(tmp_path / "ds", {"a": "train"})
    descriptor = locks.load_descriptor(tmp_path / "ds")
    descriptor["locked_train_sessions"] = ["z", "a", "a"]
    with locks.dataset_write_lock(tmp_path / "ds"):
        locks.write_descriptor(tmp_path / "ds", descriptor)
    assert locks.load_descriptor(tmp_path / "ds")["locked_train_sessions"] == ["a", "z"]
    assert not list((tmp_path / "ds").glob(".dataset-*.tmp"))
    assert not (tmp_path / "ds" / locks.LOCK_FILENAME).exists()


def test_write_lock_is_exclusive(tmp_path):
    root = tmp_path / "ds"
    with locks.dataset_write_lock(root):
        with pytest.raises(DatasetDescriptorError, match="existiert bereits"):
            with locks.dataset_write_lock(root, timeout=0.2):
                pass
    with locks.dataset_write_lock(root, timeout=0.2):  # nach Freigabe wieder moeglich
        pass


def test_lock_set_unions_registry_and_train_folder_on_disk(tmp_path):
    root = make_dataset(tmp_path / "ds", {"a": "train", "b": "val"}, locked=("ghost",))
    descriptor = locks.load_descriptor(root)
    descriptor["locked_train_sessions"] = ["ghost"]  # 'a' fehlt in der Registry, liegt aber in train/
    assert locks.effective_lock_set(root, descriptor) == {"a", "ghost"}


def test_chip_name_containing_pos_is_recovered(tmp_path):
    from pathlib import Path
    assert locks.chip_of_image(Path("sample_pos_test_pos3_frame012.png")) == "sample_pos_test"
    assert locks.chip_of_image(Path("weird.png")) is None


def test_consistency_reports_split_mismatch_missing_and_unattributable(tmp_path):
    root = make_dataset(tmp_path / "ds", {"a": "train", "b": "val"})
    (root / "images" / "test").mkdir()
    import cv2
    import numpy as np
    cv2.imwrite(str(root / "images" / "test" / "c_pos0_frame000.png"), np.zeros((4, 4), np.uint8))
    cv2.imwrite(str(root / "images" / "test" / "junk.png"), np.zeros((4, 4), np.uint8))
    descriptor = locks.load_descriptor(root)
    descriptor["sessions"]["b"]["split"] = "test"
    problems = " | ".join(locks.check_consistency(root, descriptor))
    assert "'c'" in problems and "fehlt" in problems
    assert "'b'" in problems and "val/" in problems
    assert "junk.png" in problems


def _legacy_dataset(tmp_path):
    root = make_dataset(tmp_path / "ds", {"a": "train", "b": "val"})
    (root / locks.DESCRIPTOR_FILENAME).unlink()
    import cv2
    import numpy as np
    cv2.imwrite(str(root / "images" / "val" / "a_pos0_frame009.png"), np.zeros((4, 4), np.uint8))  # a in train UND val
    legacy = tmp_path / "train_locked_chips.txt"
    legacy.write_text("old_only\n\na\n")
    return root, legacy


def test_migrate_marks_all_sessions_trained_and_uses_monitor_images_as_training_data(tmp_path):
    root, legacy = _legacy_dataset(tmp_path)
    report = locks.migrate_legacy_dataset(root, legacy)  # Standard: das aktuelle Modell sah ALLE Sessions
    assert report["straddling"] == ["a"]  # nur Information
    assert report["locked"] == ["a", "b", "old_only"]
    descriptor = locks.load_descriptor(root)
    assert descriptor["all_sessions_trained"] is True
    assert {n: e["split"] for n, e in descriptor["sessions"].items()} == {"a": "train", "b": "train"}
    assert descriptor["sessions"]["a"]["source_path"] is None
    assert locks.check_consistency(root, descriptor) == []  # val/-Dateien sind erlaubte Monitoring-Ausschnitte
    images = locks.images_by_role(root, descriptor)
    assert images["val"] == [] and images["test"] == []
    assert len(images["train"]) == 2 + 1 + 2  # a: 2 train + 1 val, b: 2 val
    assert locks.effective_lock_set(root, descriptor) == {"a", "b", "old_only"}
    with pytest.raises(DatasetDescriptorError, match="bereits"):
        locks.migrate_legacy_dataset(root)


def test_migrate_with_independent_val_test_reports_straddling_and_fails_consistency(tmp_path):
    root, legacy = _legacy_dataset(tmp_path)
    report = locks.migrate_legacy_dataset(root, legacy, all_sessions_trained=False)
    assert report["straddling"] == ["a"]
    assert report["locked"] == ["a", "old_only"]
    descriptor = locks.load_descriptor(root)
    assert not descriptor.get("all_sessions_trained")
    assert descriptor["sessions"]["a"]["split"] == "train" and descriptor["sessions"]["b"]["split"] == "val"
    assert any("'a'" in p for p in locks.check_consistency(root, descriptor))  # bleibt fuer train() gesperrt
    assert locks.images_by_role(root, descriptor)["val"]


def test_all_sessions_trained_flag_must_be_boolean(tmp_path):
    root = make_dataset(tmp_path / "ds", {"a": "train"})
    data = json.loads((root / locks.DESCRIPTOR_FILENAME).read_text())
    data["all_sessions_trained"] = "yes"
    (root / locks.DESCRIPTOR_FILENAME).write_text(json.dumps(data))
    with pytest.raises(DatasetDescriptorError, match="Boolean"):
        locks.load_descriptor(root)


def test_migrate_refuses_files_that_cannot_be_attributed(tmp_path):
    root = make_dataset(tmp_path / "ds", {"a": "train"})
    (root / locks.DESCRIPTOR_FILENAME).unlink()
    import cv2
    import numpy as np
    cv2.imwrite(str(root / "images" / "train" / "weird.png"), np.zeros((4, 4), np.uint8))
    with pytest.raises(DatasetDescriptorError, match="Session"):
        locks.migrate_legacy_dataset(root)


def test_manifest_discovery_and_validation(tmp_path):
    run = tmp_path / "run"
    (run / "fit" / "weights").mkdir(parents=True)
    ckpt = run / "fit" / "weights" / "best.pt"
    ckpt.write_bytes(b"x")
    assert find_manifest(ckpt) is None
    assert trained_sessions_of(ckpt) is None
    write_manifest(run / "manifest.json", {"schema_version": 1, "trained_sessions": ["a", "b"]})
    assert find_manifest(ckpt) == run / "manifest.json"
    assert trained_sessions_of(ckpt) == {"a", "b"}
    copied = tmp_path / "copied" / "best.pt"
    copied.parent.mkdir()
    copied.write_bytes(b"x")
    (copied.parent / "best.manifest.json").write_text(json.dumps({"schema_version": 1, "trained_sessions": ["c"]}))
    assert trained_sessions_of(copied) == {"c"}


@pytest.mark.parametrize("payload", ["nope", json.dumps({"schema_version": 2, "trained_sessions": []}),
                                     json.dumps({"schema_version": 1})])
def test_bad_manifest_raises(tmp_path, payload):
    path = tmp_path / "manifest.json"
    path.write_text(payload)
    with pytest.raises(ManifestError):
        load_manifest(path)
