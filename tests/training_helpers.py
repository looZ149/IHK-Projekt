"""Gemeinsame Test-Helfer fuer die Trainings-API (Stories 11-13).

Kein GPU, kein echtes ND2, kein acia: eine Fake-Sequenz und ein Fake-YOLO.
"""

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from chamber_obb import locks
from chamber_obb.classes import class_list


def write_selection(
    root: Path, name: str, source: str | None = None, rois=None, positions=(0,), registration=None,
    anchor_frame: int | list[int] | None = None,
) -> Path:
    """Legt ``root/<name>/selection.json`` (+ optional Registrierung) an.

    ``anchor_frame``: ``None`` = Feld weglassen (wie acia bei Frame 0), ``int`` = fuer alle ROIs, Liste = je ROI
    (pro Position dieselbe Reihenfolge wie ``rois``).
    """
    folder = root / name
    folder.mkdir(parents=True)
    rois = rois or [{"center": [50.0, 40.0], "size": [20.0, 20.0], "angle": 0.0}]
    selection = {
        "source": {"path": source or f"/acq/{name}.nd2", "sizes": {"X": 100, "Y": 80, "T": 5}},
        "selections": [],
    }
    for p in positions:
        for i, roi in enumerate(rois):
            entry = {"position": p, "roi": roi}
            anchor = anchor_frame[i] if isinstance(anchor_frame, list) else anchor_frame
            if anchor:  # acia schreibt anchor_frame nur, wenn es != 0 ist
                entry["anchor_frame"] = anchor
            selection["selections"].append(entry)
    (folder / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    if registration is not None:
        (folder / "registration_transforms.json").write_text(json.dumps(registration), encoding="utf-8")
    return folder


class _Frame:
    def __init__(self, raw):
        self.raw = raw


class _Position:
    requested: list[int] = []

    def get_frame(self, frame_idx):
        _Position.requested.append(frame_idx)
        rng = np.random.default_rng(frame_idx)
        return _Frame(rng.integers(0, 4000, size=(80, 100), dtype=np.uint16))


class _Sequence:
    def position(self, pos):
        return _Position()


def fake_open_sequence(fail_for: tuple[str, ...] = ()):
    """Fake-Opener fuer ``build_dataset(open_sequence=...)``; wirft fuer Quellen mit ``fail_for``."""

    @contextmanager
    def opener(path):
        if any(marker in str(path) for marker in fail_for):
            raise OSError(f"kaputte Aufnahme {path}")
        yield _Sequence()

    return opener


def make_dataset(
    root: Path, sessions: dict[str, str], locked: tuple[str, ...] = (), frames: int = 2, all_trained: bool = False
) -> Path:
    """Baut direkt einen gueltigen Datensatz (Bilder + Labels + dataset.json).

    ``all_trained=True`` bildet den migrierten Alt-Datensatz nach: alle Sessions sind Train, die Bilder
    unter val/ und test/ (``sessions``-Angabe) sind nur Monitoring-Ausschnitte bereits trainierter Sessions.
    """
    descriptor = locks.new_descriptor()
    descriptor["all_sessions_trained"] = all_trained
    for name, split in sessions.items():
        for kind in ("images", "labels"):
            (root / kind / split).mkdir(parents=True, exist_ok=True)
        for frame in range(frames):
            stem = f"{name}_pos0_frame{frame:03d}"
            cv2.imwrite(str(root / "images" / split / f"{stem}.png"), np.zeros((16, 16), np.uint8))
            (root / "labels" / split / f"{stem}.txt").write_text("0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n")
        descriptor["sessions"][name] = {"split": "train" if all_trained else split, "source_path": f"/acq/{name}.nd2"}
        if split == "train" or all_trained:
            descriptor["locked_train_sessions"].append(name)
    descriptor["locked_train_sessions"] += list(locked)
    with locks.dataset_write_lock(root):
        locks.write_descriptor(root, descriptor)
    return root


def tree_fingerprint(root: Path, ignore_suffixes: tuple[str, ...] = (".cache",)) -> dict[str, str]:
    """Hash-Abbild aller Dateien unter ``root`` (fuer 'unveraendert'-Pruefungen)."""
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix not in ignore_suffixes:
            result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


class FakeYOLO:
    """Stand-in fuer ``ultralytics.YOLO``: schreibt nur eine best.pt, keine echte Optimierung."""

    instances: list["FakeYOLO"] = []
    names_for_ref: dict[str, list[str]] = {}
    fail_train = False

    def __init__(self, ref):
        self.ref = str(ref)
        names = self.names_for_ref.get(self.ref, class_list())
        self.names = dict(enumerate(names))
        self.train_kwargs = None
        self.val_calls = []
        self.trainer = None
        FakeYOLO.instances.append(self)

    def train(self, **kwargs):
        if FakeYOLO.fail_train:
            raise RuntimeError("CUDA out of memory")
        self.train_kwargs = kwargs
        save_dir = Path(kwargs["project"]) / kwargs["name"]
        (save_dir / "weights").mkdir(parents=True)
        best = save_dir / "weights" / "best.pt"
        best.write_bytes(b"fake-weights:" + self.ref.encode())
        self.trainer = type("T", (), {"save_dir": save_dir, "best": best})()

    def val(self, **kwargs):
        self.val_calls.append(kwargs)
        box = type("B", (), {"map50": 0.9, "map": 0.7, "mp": 0.8, "mr": 0.6})()
        return type("R", (), {"box": box})()
