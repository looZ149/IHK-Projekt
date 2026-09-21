"""Tests fuer chamber_obb.training.train (Story 13: CAP-7, AD-15, AD-20..AD-24).

``ultralytics.YOLO`` wird durch ``FakeYOLO`` ersetzt -- kein GPU, kein Download.
"""

import json
import os

import pytest
import yaml
from training_helpers import FakeYOLO, make_dataset, tree_fingerprint

import chamber_obb
from chamber_obb import inference, locks, training
from chamber_obb.classes import class_list
from chamber_obb.errors import (
    ClassListMismatchError,
    DatasetDescriptorError,
    EmptySplitError,
    InvalidParameterError,
    LeakageError,
    ModelLoadError,
    TrainingError,
)
from chamber_obb.manifest import write_manifest
from chamber_obb.training import train


@pytest.fixture(autouse=True)
def fake_yolo(monkeypatch):
    FakeYOLO.instances = []
    FakeYOLO.names_for_ref = {}
    FakeYOLO.fail_train = False
    monkeypatch.setattr(training, "YOLO", FakeYOLO)
    monkeypatch.delenv("CHAMBER_OBB_CHECKPOINT", raising=False)


@pytest.fixture
def world(tmp_path):
    """Alt-Datensatz (mit val/test), neuer Datensatz (mit val), Basis-Checkpoint ohne Manifest."""
    old = make_dataset(tmp_path / "old", {"o1": "train", "o2": "train", "ov": "val", "ot": "test"})
    new = make_dataset(tmp_path / "new", {"n1": "train", "nv": "val"})
    ckpt = tmp_path / "base" / "best.pt"
    ckpt.parent.mkdir()
    ckpt.write_bytes(b"base-weights")
    return {"old": old, "new": new, "ckpt": ckpt, "runs": tmp_path / "runs", "tmp": tmp_path}


def _train(world, **kw):
    kw.setdefault("runs_dir", world["runs"])
    kw.setdefault("hyperparameters", {"epochs": 1})
    return train(world["new"], world["old"], world["ckpt"], **kw)


def test_public_surface_exports_train():
    assert chamber_obb.train is train and "TrainResult" in chamber_obb.__all__


def test_happy_path_both_datasets(world):
    old_before, ckpt_before = tree_fingerprint(world["old"]), world["ckpt"].read_bytes()
    result = _train(world)
    assert result.run_dir.parent == world["runs"].resolve()
    assert result.best_weights.is_file() and result.best_weights.is_relative_to(result.run_dir)
    assert result.val.map50 == 0.9 and result.test.map50_95 == 0.7 and result.test.precision == 0.8
    assert world["ckpt"].read_bytes() == ckpt_before  # Basis-Checkpoint unveraendert
    assert tree_fingerprint(world["old"]) == old_before  # old_dataset unveraendert
    cfg = yaml.safe_load((result.run_dir / "data.yaml").read_text())
    assert cfg["names"] == {0: "chamber_region", 1: "channel_region"}
    assert cfg["train"] == "lists/train.txt" and cfg["val"] == "lists/val.txt" and cfg["test"] == "lists/test.txt"
    train_list = (result.run_dir / "lists" / "train.txt").read_text().split()
    assert len(train_list) == 6 and all(os.path.isabs(p) for p in train_list)  # o1,o2,n1 je 2 Frames
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["schema_version"] == 1 and manifest["mode"] == "both"
    assert manifest["trained_sessions"] == ["n1", "o1", "o2"]
    assert manifest["val_sessions"] == ["nv", "ov"] and manifest["test_sessions"] == ["ot"]
    assert manifest["metrics"]["test"]["map50"] == 0.9
    assert {d["role"] for d in manifest["datasets"]} == {"old", "new"}
    assert all(d["descriptor_sha256"] for d in manifest["datasets"])
    assert manifest["hyperparameters"]["epochs"] == 1 and manifest["hyperparameters"]["degrees"] == 180


def test_fit_arguments_follow_the_spine(world):
    _train(world)
    fit = FakeYOLO.instances[0]
    kw = fit.train_kwargs
    assert isinstance(kw["data"], str) and kw["data"].endswith("data.yaml")  # ein yaml, nie eine Liste
    assert kw["exist_ok"] is False and kw["resume"] is False and kw["task"] == "obb" and kw["name"] == "fit"
    assert kw["project"].endswith(os.sep + os.path.basename(kw["project"])) and "runs" in kw["project"]
    evaluator = FakeYOLO.instances[-1]
    assert [c["split"] for c in evaluator.val_calls] == ["val", "test"]
    assert all(c["project"] == kw["project"] and c["exist_ok"] is False for c in evaluator.val_calls)


def test_every_call_builds_a_fresh_model_and_a_unique_run_dir(world):
    first = _train(world)
    second = _train(world)
    assert first.run_dir != second.run_dir and first.run_dir.is_dir() and second.run_dir.is_dir()
    fits = [m for m in FakeYOLO.instances if m.train_kwargs]
    assert len(fits) == 2 and fits[0] is not fits[1]
    assert all(m.ref == str(world["ckpt"].resolve()) for m in fits)  # beide ab base_checkpoint, nie ab Vorlauf


def test_modes_new_old_both(world):
    only_new = _train(world, mode="new")
    lists = (only_new.run_dir / "lists" / "train.txt").read_text()
    assert "n1_pos0" in lists and "o1_pos0" not in lists
    assert json.loads(only_new.manifest_path.read_text())["trained_sessions"] == ["n1", "o1", "o2"]  # Historie via old_dataset
    only_old = _train(world, mode="old")
    assert "n1_pos0" not in (only_old.run_dir / "lists" / "train.txt").read_text()
    with pytest.raises(InvalidParameterError):
        train(new_dataset=None, old_dataset=world["old"], base_checkpoint=world["ckpt"], mode="new", runs_dir=world["runs"])
    with pytest.raises(InvalidParameterError):
        _train(world, mode="everything")
    with pytest.raises(InvalidParameterError, match="weder"):
        train(base_checkpoint=world["ckpt"], runs_dir=world["runs"])


def test_only_the_checkpoint_falls_back_to_the_environment(world, monkeypatch):
    monkeypatch.setenv("CHAMBER_OBB_CHECKPOINT", str(world["ckpt"]))
    result = train(world["new"], world["old"], runs_dir=world["runs"], hyperparameters={"epochs": 1})
    assert result.run_dir.parent == world["runs"].resolve()
    assert os.environ["CHAMBER_OBB_CHECKPOINT"] == str(world["ckpt"])  # AD-15: nie umgesetzt
    # Datensatz- und Run-Pfade haben keine Umgebungsvariablen: sie muessen uebergeben werden
    monkeypatch.setenv("CHAMBER_OBB_NEW_DATASET", str(world["new"]))
    monkeypatch.setenv("CHAMBER_OBB_OLD_DATASET", str(world["old"]))
    with pytest.raises(InvalidParameterError, match="weder"):
        train(base_checkpoint=world["ckpt"], runs_dir=world["runs"])


def test_missing_or_unloadable_base_checkpoint(world, monkeypatch):
    with pytest.raises(ModelLoadError, match="kein base_checkpoint"):
        train(world["new"], world["old"], runs_dir=world["runs"])
    with pytest.raises(ModelLoadError, match="existiert nicht"):
        train(world["new"], world["old"], world["tmp"] / "nope.pt", runs_dir=world["runs"])
    with pytest.raises(ModelLoadError, match="existiert nicht"):  # Stock-Name ohne reinit_head
        train(world["new"], world["old"], "yolo26n-obb.pt", runs_dir=world["runs"])

    class Broken(FakeYOLO):
        def __init__(self, ref):
            raise TypeError("corrupt checkpoint")

    monkeypatch.setattr(training, "YOLO", Broken)
    with pytest.raises(ModelLoadError, match="liess sich nicht laden"):
        _train(world)
    assert not world["runs"].exists() or not list(world["runs"].iterdir())


def test_class_guard_and_reinit_head(world):
    FakeYOLO.names_for_ref[str(world["ckpt"].resolve())] = ["person", "car", "boat"]
    with pytest.raises(ClassListMismatchError, match="reinit_head"):
        _train(world)
    assert not world["runs"].exists() or not list(world["runs"].iterdir())  # kein Run-Dir vor dem Gate
    result = _train(world, reinit_head=True)
    assert json.loads(result.manifest_path.read_text())["reinit_head"] is True
    FakeYOLO.names_for_ref[str(world["ckpt"].resolve())] = list(reversed(class_list()))  # gleiche Namen, andere Reihenfolge
    with pytest.raises(ClassListMismatchError):
        _train(world)


def test_stock_start_needs_reinit_head_and_has_known_empty_lineage(world):
    FakeYOLO.names_for_ref["yolo26n-obb.pt"] = ["a", "b", "c"]
    result = train(world["new"], None, "yolo26n-obb.pt", reinit_head=True, runs_dir=world["runs"],
                   hyperparameters={"epochs": 1})
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["base_is_stock"] and manifest["trained_sessions"] == ["n1"]
    assert FakeYOLO.instances[0].ref == "yolo26n-obb.pt"


def test_dataset_class_list_mismatch_is_rejected(world):
    descriptor = locks.load_descriptor(world["new"])
    descriptor["class_list"] = ["chamber_region"]
    with locks.dataset_write_lock(world["new"]):
        locks.write_descriptor(world["new"], descriptor)
    with pytest.raises(ClassListMismatchError, match="Datensatz"):
        _train(world)


def test_dataset_without_or_with_inconsistent_descriptor_is_rejected(world):
    (world["new"] / "dataset.json").unlink()
    with pytest.raises(DatasetDescriptorError):
        _train(world)
    make_dataset(world["tmp"] / "fresh", {"a": "train", "v": "val"})
    import cv2
    import numpy as np
    cv2.imwrite(str(world["tmp"] / "fresh" / "images" / "val" / "a_pos0_frame005.png"), np.zeros((4, 4), np.uint8))
    with pytest.raises(DatasetDescriptorError, match="inkonsistent"):
        train(world["tmp"] / "fresh", None, world["ckpt"], acknowledge_unknown_lineage=True, runs_dir=world["runs"])
    with pytest.raises(InvalidParameterError, match="existiert nicht"):
        train(world["tmp"] / "missing", None, world["ckpt"], runs_dir=world["runs"])


def test_leakage_old_val_session_reappearing_in_new_train(world):
    make_dataset(world["tmp"] / "bad", {"ov": "train", "bv": "val"})  # 'ov' ist im Alt-Datensatz Val
    with pytest.raises(LeakageError, match="Val und Train"):
        train(world["tmp"] / "bad", world["old"], world["ckpt"], runs_dir=world["runs"])
    make_dataset(world["tmp"] / "bad2", {"x": "train", "ot": "train", "y": "val"})
    with pytest.raises(LeakageError, match="Test und Train"):
        train(world["tmp"] / "bad2", world["old"], world["ckpt"], runs_dir=world["runs"])
    assert not world["runs"].exists() or not list(world["runs"].iterdir())


def test_empty_val_or_train_raises_without_fallback(world):
    make_dataset(world["tmp"] / "noval", {"a": "train"})
    with pytest.raises(EmptySplitError, match="Val"):
        train(world["tmp"] / "noval", None, world["ckpt"], acknowledge_unknown_lineage=True, runs_dir=world["runs"])
    make_dataset(world["tmp"] / "notrain", {"v": "val"})
    with pytest.raises(EmptySplitError, match="Train"):
        train(world["tmp"] / "notrain", None, world["ckpt"], acknowledge_unknown_lineage=True, runs_dir=world["runs"])


def test_no_test_split_trains_with_warning_and_no_test_metrics(world):
    make_dataset(world["tmp"] / "notest", {"a": "train", "v": "val"})
    result = train(world["tmp"] / "notest", None, world["ckpt"], acknowledge_unknown_lineage=True,
                   runs_dir=world["runs"], hyperparameters={"epochs": 1})
    assert result.test is None and result.warnings
    assert json.loads(result.manifest_path.read_text())["metrics"]["test"] is None
    assert [c["split"] for c in FakeYOLO.instances[-1].val_calls] == ["val"]
    assert "test" not in yaml.safe_load((result.run_dir / "data.yaml").read_text())


def test_lineage_from_manifest_blocks_val_sessions_the_checkpoint_saw(world):
    run = world["tmp"] / "prev"
    (run / "fit" / "weights").mkdir(parents=True)
    (run / "fit" / "weights" / "best.pt").write_bytes(b"prev")
    write_manifest(run / "manifest.json", {"schema_version": 1, "trained_sessions": ["nv", "z"]})
    with pytest.raises(LeakageError, match="nv"):
        train(world["new"], None, run / "fit" / "weights" / "best.pt", runs_dir=world["runs"])


def test_lineage_from_old_dataset_when_checkpoint_has_no_manifest(world):
    make_dataset(world["tmp"] / "cheat", {"n": "train", "o1": "val"})  # o1 ist im old_dataset Train (gesperrt)
    with pytest.raises(LeakageError, match="o1"):
        train(world["tmp"] / "cheat", world["old"], world["ckpt"], mode="new", runs_dir=world["runs"])


def test_unknown_lineage_raises_unless_acknowledged(world):
    with pytest.raises(LeakageError, match="unbekannt"):
        train(world["new"], None, world["ckpt"], runs_dir=world["runs"])
    result = train(world["new"], None, world["ckpt"], acknowledge_unknown_lineage=True, runs_dir=world["runs"],
                   hyperparameters={"epochs": 1})
    assert json.loads(result.manifest_path.read_text())["acknowledge_unknown_lineage"] is True


def test_manifest_lineage_is_cumulative_across_generations(world):
    first = _train(world)
    promoted = world["tmp"] / "promoted"
    promoted.mkdir()
    (promoted / "best.pt").write_bytes(first.best_weights.read_bytes())
    (promoted / "best.manifest.json").write_text(first.manifest_path.read_text())  # wie in RELEASE.md beschrieben
    gen2_ds = make_dataset(world["tmp"] / "gen2", {"g2": "train", "g2v": "val"})
    second = train(gen2_ds, None, promoted / "best.pt", mode="new", runs_dir=world["runs"], hyperparameters={"epochs": 1})
    manifest = json.loads(second.manifest_path.read_text())
    assert manifest["trained_sessions"] == ["g2", "n1", "o1", "o2"]  # transitiv: Vorgaenger + neu
    assert manifest["parent_manifest"] == str(promoted / "best.manifest.json")
    leaky = make_dataset(world["tmp"] / "gen3", {"g3": "train", "n1": "val"})  # n1 hat schon ein Vorgaenger gesehen
    with pytest.raises(LeakageError, match="n1"):
        train(leaky, None, second.best_weights, mode="new", runs_dir=world["runs"])


def test_ultralytics_failures_are_wrapped(world):
    FakeYOLO.fail_train = True
    with pytest.raises(TrainingError, match="CUDA"):
        _train(world)


def test_reserved_hyperparameters_are_rejected(world):
    with pytest.raises(InvalidParameterError, match="data"):
        _train(world, hyperparameters={"data": "elsewhere.yaml", "project": "x"})


def test_train_never_touches_the_inference_cache_or_promotes(world, monkeypatch):
    sentinel = object()
    monkeypatch.setitem(inference._MODEL_CACHE, "sentinel", sentinel)
    monkeypatch.setattr(inference, "_get_model", lambda *a, **k: pytest.fail("Inferenz-Cache benutzt"))
    _train(world)
    assert inference._MODEL_CACHE == {"sentinel": sentinel}
    assert "CHAMBER_OBB_CHECKPOINT" not in os.environ


def test_device_and_seed_are_passed_through(world):
    _train(world, device="cpu", seed=7)
    kw = FakeYOLO.instances[0].train_kwargs
    assert kw["device"] == "cpu" and kw["seed"] == 7
    assert FakeYOLO.instances[-1].val_calls[0]["device"] == "cpu"


# --- Alt-Datensatz ohne unabhaengiges Val/Test (aktuelles Modell auf ALLEN Sessions trainiert) ---------------

@pytest.fixture
def legacy_world(world):
    old = make_dataset(world["tmp"] / "legacy_old", {"o1": "train", "o2": "train", "mon": "val", "mon2": "test"},
                       all_trained=True)
    return {**world, "old": old}


def test_all_trained_old_dataset_is_pure_training_data(legacy_world):
    result = _train(legacy_world)
    train_list = (result.run_dir / "lists" / "train.txt").read_text()
    assert "mon_pos0" in train_list and "mon2_pos0" in train_list and "o1_pos0" in train_list  # Monitoring-Bilder = Train
    val_list = (result.run_dir / "lists" / "val.txt").read_text()
    assert "nv_pos0" in val_list and "mon" not in val_list.replace("nv_pos0", "")
    assert not (result.run_dir / "lists" / "test.txt").exists()  # nirgends ein Test-Split
    assert result.test is None and result.warnings
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["val_sessions"] == ["nv"] and manifest["test_sessions"] == []
    assert manifest["trained_sessions"] == ["mon", "mon2", "n1", "o1", "o2"]


def test_all_trained_old_dataset_alone_or_without_new_val_has_no_validation(legacy_world):
    with pytest.raises(EmptySplitError, match="Val"):
        _train(legacy_world, mode="old")
    make_dataset(legacy_world["tmp"] / "noval", {"a": "train"})
    with pytest.raises(EmptySplitError, match="all_sessions_trained"):
        train(legacy_world["tmp"] / "noval", legacy_world["old"], legacy_world["ckpt"], runs_dir=legacy_world["runs"])


def test_new_val_session_that_the_legacy_model_trained_on_is_leakage(legacy_world):
    make_dataset(legacy_world["tmp"] / "cheat", {"x": "train", "mon": "val"})  # 'mon' war im Alt-Modell schon Train
    with pytest.raises(LeakageError, match="mon"):
        train(legacy_world["tmp"] / "cheat", legacy_world["old"], legacy_world["ckpt"], runs_dir=legacy_world["runs"])


# --- Baseline-Evaluation des Basis-Checkpoints -------------------------------------------------------------

def test_base_checkpoint_is_evaluated_as_baseline_on_the_same_val_test_data(world):
    result = _train(world)
    assert result.baseline_val == result.val and result.baseline_test == result.test  # Fake liefert gleiche Zahlen
    baseline_model = next(m for m in FakeYOLO.instances if m.val_calls and not m.train_kwargs and m.ref == str(world["ckpt"].resolve()))
    assert [c["split"] for c in baseline_model.val_calls] == ["val", "test"]
    assert [c["name"] for c in baseline_model.val_calls] == ["baseline-val", "baseline-test"]
    assert all(c["data"] == str(result.run_dir / "data.yaml") for c in baseline_model.val_calls)
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["baseline_metrics"]["val"]["map50"] == 0.9
    # Baseline laeuft VOR dem Fitten: sie darf nicht die frisch trainierten Gewichte bewerten
    assert baseline_model is not [m for m in FakeYOLO.instances if m.train_kwargs][0]


def test_baseline_is_skipped_for_stock_reinit_or_when_switched_off(world):
    assert _train(world, evaluate_baseline=False).baseline_val is None
    FakeYOLO.names_for_ref[str(world["ckpt"].resolve())] = ["a", "b", "c"]
    assert _train(world, reinit_head=True).baseline_val is None
    FakeYOLO.names_for_ref["yolo26n-obb.pt"] = ["a"]
    stock = train(world["new"], None, "yolo26n-obb.pt", reinit_head=True, runs_dir=world["runs"])
    assert stock.baseline_val is None and stock.baseline_test is None


def test_failing_baseline_evaluation_stops_before_training(world, monkeypatch):
    class BrokenVal(FakeYOLO):
        def val(self, **kwargs):
            raise RuntimeError("bad val data")

    monkeypatch.setattr(training, "YOLO", BrokenVal)
    with pytest.raises(TrainingError, match="Baseline"):
        _train(world)
    assert not any(m.train_kwargs for m in FakeYOLO.instances)  # kein Fit ausgefuehrt
