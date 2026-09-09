"""Regression tests for train_lock.discover_filesystem_train_chips (Story 9,
AD-18) and the assign_splits() forced_train guarantee it feeds into (AD-12).

Two import hazards this file works around, both pre-existing and documented in
AGENTS.md, unrelated to this story:
- `training/` is not an installed package (unlike `chamber_obb`, wired in via
  the uv workspace) - its modules are imported by prepending `src/training` to
  `sys.path`, the same approach this story's own spec verification commands use.
- `train_lock.py` itself has zero external dependencies, so it imports cleanly.
  `build_trainingsdata.py`, however, imports `acia` (not installed in this
  stripped repo) and `training/config.py`, which eagerly loads real YOLO
  checkpoints at import time - one of them (`YOLO("yolo26n-obb.pt")`) auto-
  downloads from the internet if not already cached, the other raises
  FileNotFoundError for a path that only exists on the remote KIT training box.
  To test assign_splits() without triggering either, `build_trainingsdata_module`
  below installs minimal stub modules for `acia.segm.open` and `config` in
  sys.modules before importing build_trainingsdata, then reverts them.
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "training"))

from train_lock import discover_filesystem_train_chips  # noqa: E402


def test_discover_recovers_chip_name_from_realistic_filenames(tmp_path):
    train_dir = tmp_path / "images" / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "SessionA_pos0_frame000.png").touch()
    (train_dir / "SessionA_pos1_frame005.png").touch()

    assert discover_filesystem_train_chips(tmp_path) == {"SessionA"}


def test_discover_skips_non_matching_filename_but_logs_a_warning(tmp_path, capsys):
    train_dir = tmp_path / "images" / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "not_matching_at_all.png").touch()

    result = discover_filesystem_train_chips(tmp_path)

    assert result == set()
    captured = capsys.readouterr()
    assert "not_matching_at_all.png" in captured.out


def test_discover_chip_name_containing_pos_substring_roundtrips(tmp_path):
    """A chip folder literally named 'sample_pos_test' must not be truncated
    by a naive '_pos' split - only the trailing _pos<N>_frame<M> suffix is
    stripped."""
    train_dir = tmp_path / "images" / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "sample_pos_test_pos0_frame001.png").touch()

    assert discover_filesystem_train_chips(tmp_path) == {"sample_pos_test"}


def test_discover_missing_directory_returns_empty_set_without_raising(tmp_path):
    missing = tmp_path / "does_not_exist"

    assert discover_filesystem_train_chips(missing) == set()


def test_discover_degenerate_empty_prefix_is_not_added(tmp_path):
    """A filename whose stem is EXACTLY '_pos<N>_frame<M>' (no chip-name
    prefix) must not add an empty string to the result."""
    train_dir = tmp_path / "images" / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "_pos0_frame000.png").touch()

    assert discover_filesystem_train_chips(tmp_path) == set()


def test_discover_locked_list_and_filesystem_agreement_is_a_no_op(tmp_path):
    """Both sources naming the same chip is fine - union is idempotent."""
    train_dir = tmp_path / "images" / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "AlreadyLocked_pos0_frame000.png").touch()

    discovered = discover_filesystem_train_chips(tmp_path)
    locked = {"AlreadyLocked"}

    assert discovered | locked == {"AlreadyLocked"}


@pytest.fixture
def build_trainingsdata_module(monkeypatch):
    """Imports build_trainingsdata with acia/config stubbed out so the real,
    network-/filesystem-touching config.py never executes. See module
    docstring above for why this is necessary in this environment."""
    fake_acia = types.ModuleType("acia")
    fake_acia_segm = types.ModuleType("acia.segm")
    fake_acia_segm_open = types.ModuleType("acia.segm.open")

    def _unused_open_sequence(*_args, **_kwargs):
        raise NotImplementedError("stub open_sequence - not exercised by this test")

    fake_acia_segm_open.open_sequence = _unused_open_sequence
    fake_acia_segm.open = fake_acia_segm_open
    fake_acia.segm = fake_acia_segm

    fake_config = types.ModuleType("config")
    fake_config.CHIPS_PATH = "/nonexistent"
    fake_config.SEED = 42

    for name, mod in {
        "acia": fake_acia,
        "acia.segm": fake_acia_segm,
        "acia.segm.open": fake_acia_segm_open,
        "config": fake_config,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    # Falls ein frueherer Test/Import bereits ein echtes (oder anders gestubtes)
    # build_trainingsdata geladen hat: frischen Import unter diesen Stubs erzwingen.
    monkeypatch.delitem(sys.modules, "build_trainingsdata", raising=False)
    import build_trainingsdata

    yield build_trainingsdata

    monkeypatch.delitem(sys.modules, "build_trainingsdata", raising=False)


def test_assign_splits_forced_train_member_survives_random_draw(build_trainingsdata_module):
    """The actual leakage-prevention guarantee this story protects: a name
    passed via forced_train must land in 'train' in the returned assignment
    even when the random draw would otherwise place it elsewhere."""
    assign_splits = build_trainingsdata_module.assign_splits
    folder_names = [f"session_{i}" for i in range(20)]
    forced = {"session_0"}

    for seed in range(10):
        assignment = assign_splits(
            folder_names, val_ratio=0.5, test_ratio=0.5, seed=seed, forced_train=forced,
        )
        assert assignment["session_0"] == "train"
