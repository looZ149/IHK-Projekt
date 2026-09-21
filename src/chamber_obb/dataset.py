"""``build_dataset()``: kuratierte Chip-Ordner -> neuer YOLO-OBB-Datensatz (CAP-7, AD-20).

Portiert die wiederverwendbare Logik aus ``training/build_trainingsdata.py``
(ROI -> OBB-Ecken, Registrierungs-Propagation, Frame-Auswahl, Session-Split),
ohne ``training/`` zu importieren oder zu veraendern (AD-4). Schreibt nur in
``new_dataset``; ``old_dataset`` wird ausschliesslich gelesen (AD-20).

``acia`` wird erst beim tatsaechlichen Oeffnen einer Aufnahme importiert
(AD-4) -- ``import chamber_obb`` und ``train()`` brauchen es nicht.
"""

import hashlib
import json
import logging
import math
import os
import random
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from chamber_obb.classes import DEFAULT_CHANNEL_ASPECT_RATIO_THRESHOLD, class_list, classify_roi_by_shape
from chamber_obb.errors import (
    DatasetBuildError,
    DatasetDescriptorError,
    InvalidParameterError,
)
from chamber_obb.locks import (
    SPLITS,
    Descriptor,
    check_consistency,
    chip_of_image,
    all_sessions_trained,
    dataset_write_lock,
    effective_lock_set,
    load_descriptor,
    new_descriptor,
    scan_sessions_on_disk,
    write_descriptor,
)
from chamber_obb.manifest import trained_sessions_of
from chamber_obb.preprocessing import normalize_to_8bit

logger = logging.getLogger(__name__)

OLD_DATASET_ENV_VAR = "CHAMBER_OBB_OLD_DATASET"
NEW_DATA_ENV_VAR = "CHAMBER_OBB_NEW_DATA"
NEW_DATASET_ENV_VAR = "CHAMBER_OBB_NEW_DATASET"
_CHECKPOINT_ENV_VAR = "CHAMBER_OBB_CHECKPOINT"

REGISTRATION_FILENAME = "registration_transforms.json"
SELECTION_FILENAME = "selection.json"
# Projektweit fester Annotations-Frame (= reference_frame der Registrierung).
ANNOT_FRAME = 0
DEFAULT_SEED = 42


@dataclass(frozen=True)
class BuildReport:
    """Ergebnis von ``build_dataset()`` -- nur was in DIESEM Aufruf passiert ist."""

    new_dataset: Path
    sessions: dict[str, str]  # Session -> Split (in diesem Aufruf verarbeitet)
    images_written: dict[str, int]  # Split -> Anzahl Bild/Label-Paare
    skipped: dict[str, str]  # Chip-Ordner -> Grund
    locked: list[str]  # Sessions, die auf Train gesperrt waren
    warnings: list[str] = field(default_factory=list)
    rebuilt: bool = False


# --------------------------------------------------------------------------
# Pfade
# --------------------------------------------------------------------------

def _resolve_path(value: str | Path | None, env_var: str, label: str, required: bool = True) -> Path | None:
    raw = value if value is not None else os.environ.get(env_var)
    if not raw:
        if required:
            raise InvalidParameterError(f"{label} nicht angegeben und {env_var} nicht gesetzt")
        return None
    return Path(raw).expanduser().resolve()


def _is_within(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents


def check_dataset_paths(new_dataset: Path, old_dataset: Path | None) -> None:
    """``new_dataset`` darf ``old_dataset`` weder gleichen noch enthalten noch darin liegen (AD-20)."""
    if old_dataset is None:
        return
    if _is_within(new_dataset, old_dataset) or _is_within(old_dataset, new_dataset):
        raise InvalidParameterError(
            f"new_dataset ({new_dataset}) und old_dataset ({old_dataset}) duerfen weder gleich "
            f"sein noch ineinander liegen"
        )


# --------------------------------------------------------------------------
# Geometrie und Registrierung (portiert aus build_trainingsdata.py)
# --------------------------------------------------------------------------

def rotated_box_to_corners(
    cx: float, cy: float, w: float, h: float, angle_deg: float, img_w: int, img_h: int
) -> list[float]:
    """Wandelt center/size/angle in vier normalisierte Eckpunkte um (YOLO-OBB-Format)."""
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    dx, dy = w / 2, h / 2
    corners: list[float] = []
    for x, y in [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]:
        x_rot = x * cos_a - y * sin_a + cx
        y_rot = x * sin_a + y * cos_a + cy
        corners.append(min(max(x_rot / img_w, 0.0), 1.0))
        corners.append(min(max(y_rot / img_h, 0.0), 1.0))
    return corners


def _rotation_matrix(theta_rad: float) -> np.ndarray:
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return np.array([[c, -s], [s, c]])


def _invert_rigid(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R_inv = R.T
    return R_inv, -R_inv @ t


def transform_point(
    point: tuple[float, float],
    theta_from_rad: float, t_from: tuple[float, float],
    theta_to_rad: float, t_to: tuple[float, float],
) -> tuple[float, float]:
    """Ueberfuehrt einen Punkt von den Koordinaten eines Frames in die eines anderen.

    Die gespeicherte Transformation (dx, dy, theta) eines Frames bildet Punkte
    vom reference_frame IN dieses Frame ab: ``p_frame = R(theta) @ p_ref + t``.
    """
    R_from_inv, t_from_inv = _invert_rigid(_rotation_matrix(theta_from_rad), np.array(t_from))
    p_ref = R_from_inv @ np.array(point) + t_from_inv
    p_to = _rotation_matrix(theta_to_rad) @ p_ref + np.array(t_to)
    return float(p_to[0]), float(p_to[1])


def propagate_roi_across_frames(
    roi: dict, annot_frame: int, target_frames: list[int], transforms: dict[int, dict],
) -> dict[int, dict]:
    """Berechnet center/angle einer auf ``annot_frame`` gezeichneten ROI fuer jeden Ziel-Frame.

    Die Groesse bleibt unveraendert (starre Registrierung, keine Skalierung).
    Frames ohne Transform werden ausgelassen.
    """
    if annot_frame not in transforms:
        raise ValueError(
            f"Kein Registrierungs-Transform fuer den Annotations-Frame {annot_frame} vorhanden"
        )
    t_annot = transforms[annot_frame]
    cx, cy = roi["center"]
    result: dict[int, dict] = {}
    for frame_idx in target_frames:
        if frame_idx not in transforms:
            continue
        t_target = transforms[frame_idx]
        new_cx, new_cy = transform_point(
            (cx, cy),
            t_annot["theta"], (t_annot["dx"], t_annot["dy"]),
            t_target["theta"], (t_target["dx"], t_target["dy"]),
        )
        result[frame_idx] = {
            "center": [new_cx, new_cy],
            "size": roi["size"],
            "angle": roi["angle"] + math.degrees(t_target["theta"] - t_annot["theta"]),
        }
    return result


def load_registration(registration_path: Path) -> dict[int, dict]:
    """Laedt ``registration_transforms.json`` -> Position -> {transforms, reference_frame}."""
    data = json.loads(Path(registration_path).read_text(encoding="utf-8"))
    by_position: dict[int, dict] = {}
    for record in data.get("records", []):
        transforms = {
            int(frame_idx): {"dx": t["dx"], "dy": t["dy"], "theta": t["theta"]}
            for frame_idx, t in record.get("transforms", {}).items()
        }
        by_position[record["position"]] = {
            "transforms": transforms,
            "reference_frame": record.get("reference_frame", 0),
        }
    return by_position


def select_target_frames(
    valid_frames: set[int], annot_frame: int, count: int | None, seed: int, seed_key: str = ""
) -> list[int]:
    """Waehlt reproduzierbar (seed + seed_key) Ziel-Frames aus den registrierten Frames.

    ``count=None``: alle. Der Annotations-Frame ist immer dabei. Die Auswahl
    betrifft nur, WELCHE Frames exportiert werden -- die Box wird pro Frame
    korrekt ueber die Registrierung berechnet.
    """
    frames = sorted(valid_frames)
    if count is None or count >= len(frames):
        return frames
    count = max(1, count)
    remaining = [f for f in frames if f != annot_frame]
    digest = hashlib.sha256(f"{seed}:{seed_key}".encode()).hexdigest()
    random.Random(int(digest, 16)).shuffle(remaining)
    chosen = set(remaining[: max(0, count - 1)])
    chosen.add(annot_frame)
    return sorted(chosen)


# --------------------------------------------------------------------------
# Split-Zuordnung (portiert aus assign_splits, AD-12)
# --------------------------------------------------------------------------

def assign_splits(
    names: list[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    forced_val: set[str],
    forced_test: set[str],
    forced_train: set[str],
) -> dict[str, str]:
    """Weist jede Session reproduzierbar (per seed) einem Split zu -- nie pro Einzelbild."""
    for label, overlap in (
        ("val_chips und test_chips", forced_val & forced_test),
        ("val_chips und train/gesperrt", forced_val & forced_train),
        ("test_chips und train/gesperrt", forced_test & forced_train),
    ):
        if overlap:
            raise InvalidParameterError(f"Sessions duerfen nicht gleichzeitig in {label} stehen: {sorted(overlap)}")

    remaining = sorted(n for n in names if n not in forced_val | forced_test | forced_train)
    random.Random(seed).shuffle(remaining)
    n_total = len(names)
    n_val = max(0, round(n_total * val_ratio) - len(forced_val))
    n_test = max(0, round(n_total * test_ratio) - len(forced_test))
    val_random = set(remaining[:n_val])
    test_random = set(remaining[n_val:n_val + n_test])

    assignment: dict[str, str] = {}
    for name in names:
        if name in forced_train:
            assignment[name] = "train"
        elif name in forced_val or name in val_random:
            assignment[name] = "val"
        elif name in forced_test or name in test_random:
            assignment[name] = "test"
        else:
            assignment[name] = "train"
    return assignment


# --------------------------------------------------------------------------
# Verarbeitung eines Chip-Ordners
# --------------------------------------------------------------------------

def _default_open_sequence() -> Callable:
    try:
        from acia.segm.open import open_sequence  # lazy, AD-4
    except ImportError as exc:
        raise DatasetBuildError(
            "acia ist nicht installiert -- zum Bauen eines Datensatzes aus ND2-Aufnahmen "
            "wird das Extra 'chamber_obb[dashboard]' benoetigt"
        ) from exc
    return open_sequence


def _remove_chip_files(dataset_root: Path, chip: str) -> None:
    """Entfernt alle Bild/Label-Dateien eines Chips (alle Splits) vor dem Neuschreiben."""
    for kind in ("images", "labels"):
        for split in SPLITS:
            split_dir = dataset_root / kind / split
            if not split_dir.is_dir():
                continue
            for path in split_dir.iterdir():
                if chip_of_image(path) == chip:
                    path.unlink()


def _process_chip(
    folder: Path,
    selection: dict,
    images_dir: Path,
    labels_dir: Path,
    open_sequence: Callable,
    use_registration: bool,
    frames_per_position: int | None,
    threshold: float,
    seed: int,
) -> int:
    """Schreibt Bild/Label-Paare eines Chip-Ordners; gibt die Anzahl zurueck."""
    source = selection["source"]
    img_w, img_h = source["sizes"]["X"], source["sizes"]["Y"]

    rois_by_position: dict[int, list[dict]] = {}
    for sel in selection.get("selections", []):
        rois_by_position.setdefault(sel["position"], []).append(sel["roi"])
    if not rois_by_position:
        raise DatasetBuildError("keine Annotationen in selection.json")

    registration_by_position: dict[int, dict] = {}
    if use_registration:
        registration_path = folder / REGISTRATION_FILENAME
        if registration_path.exists():
            registration_by_position = load_registration(registration_path)
        else:
            logger.warning(
                "%s: use_registration gesetzt, aber %s fehlt -- nur Frame %d (kein unkorrigierter "
                "Ersatz-Frame, um keine falschen Labels zu erzeugen)",
                folder.name, REGISTRATION_FILENAME, ANNOT_FRAME,
            )

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with open_sequence(source["path"]) as seq:
        for pos, rois in sorted(rois_by_position.items()):
            pos_source = seq.position(pos)
            reg = registration_by_position.get(pos)
            # Ohne Registrierung wird NUR der annotierte Frame exportiert: ein anderer
            # Frame mit unveraenderter Box waere falsch beschriftet.
            target_frames = [ANNOT_FRAME]
            if reg is not None:
                try:
                    target_frames = select_target_frames(
                        set(reg["transforms"]), ANNOT_FRAME, frames_per_position, seed,
                        seed_key=f"{folder.name}_pos{pos}",
                    )
                except Exception:  # defekte Registrierung: bei Frame 0 bleiben
                    target_frames = [ANNOT_FRAME]

            for frame_idx in target_frames:
                if reg is not None and frame_idx != ANNOT_FRAME:
                    try:
                        frame_rois = [
                            propagate_roi_across_frames(roi, ANNOT_FRAME, [frame_idx], reg["transforms"])[frame_idx]
                            for roi in rois
                        ]
                    except (ValueError, KeyError) as exc:
                        logger.warning("%s Position %s: %s -- Propagation uebersprungen", folder.name, pos, exc)
                        continue
                else:
                    frame_rois = rois

                raw = pos_source.get_frame(frame_idx).raw
                frame = raw[..., 0] if raw.ndim == 3 else raw
                img_8bit = normalize_to_8bit(frame)

                # Namensschema synchron mit locks._IMAGE_NAME_RE (AD-18).
                name = f"{folder.name}_pos{pos}_frame{frame_idx:03d}"
                if not cv2.imwrite(str(images_dir / f"{name}.png"), img_8bit):
                    raise DatasetBuildError(f"Bild {name}.png konnte nicht geschrieben werden")

                lines = []
                for roi in frame_rois:
                    cx, cy = roi["center"]
                    w, h = roi["size"]
                    class_id = classify_roi_by_shape(w, h, threshold)
                    corners = rotated_box_to_corners(cx, cy, w, h, roi["angle"], img_w, img_h)
                    lines.append(f"{class_id} " + " ".join(f"{c:.6f}" for c in corners))
                (labels_dir / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
                written += 1
    return written


# --------------------------------------------------------------------------
# Oeffentlicher Einstiegspunkt
# --------------------------------------------------------------------------

def _old_dataset_view(old_dataset: Path | None) -> tuple[Descriptor | None, set[str], set[str], dict[str, str | None]]:
    """(Descriptor, Lock-Set, Val/Test-Sessions, Quellpfade) des Alt-Datensatzes; nur lesend."""
    if old_dataset is None:
        return None, set(), set(), {}
    descriptor = load_descriptor(old_dataset)
    if descriptor is None:
        raise DatasetDescriptorError(f"old_dataset {old_dataset} enthaelt weder Bilder noch {'dataset.json'}")
    locked = effective_lock_set(old_dataset, descriptor)
    val_test = {n for n, e in descriptor["sessions"].items() if e["split"] in ("val", "test")}
    if not all_sessions_trained(descriptor):  # sonst: nur Monitoring-Val/Test bereits trainierter Sessions
        on_disk, _ = scan_sessions_on_disk(old_dataset)
        val_test |= on_disk["val"] | on_disk["test"]
    sources = {n: e.get("source_path") for n, e in descriptor["sessions"].items()}
    return descriptor, locked, val_test, sources


def build_dataset(
    new_data: str | Path | None = None,
    new_dataset: str | Path | None = None,
    old_dataset: str | Path | None = None,
    *,
    chips: Iterable[str] | None = None,
    base_checkpoint: str | Path | None = None,
    prior_manifest: str | Path | None = None,
    rebuild: bool = False,
    val_ratio: float = 0.175,
    test_ratio: float = 0.175,
    val_chips: Iterable[str] = (),
    test_chips: Iterable[str] = (),
    train_chips: Iterable[str] = (),
    use_registration: bool = False,
    frames_per_position: int | None = 1,
    channel_aspect_threshold: float = DEFAULT_CHANNEL_ASPECT_RATIO_THRESHOLD,
    seed: int = DEFAULT_SEED,
    open_sequence: Callable | None = None,
) -> BuildReport:
    """Baut aus kuratierten Chip-Ordnern einen neuen YOLO-OBB-Datensatz (CAP-7).

    ``new_data``: Ordner mit Chip-Unterordnern (``selection.json`` + Aufnahme
    [+ ``registration_transforms.json``]). ``new_dataset``: Ausgabeordner, der
    einzige Ort, in den geschrieben wird. ``old_dataset``: geteilter Alt-Datensatz,
    nur lesend -- seine Sperrliste und Val/Test-Sessions schuetzen den neuen
    Datensatz vor Leakage (AD-12). Alle drei fallen auf
    ``CHAMBER_OBB_NEW_DATA`` / ``CHAMBER_OBB_NEW_DATASET`` / ``CHAMBER_OBB_OLD_DATASET``
    zurueck (AD-8).

    Die Trainings-Historie des Basis-Checkpoints (``base_checkpoint``, sonst
    ``CHAMBER_OBB_CHECKPOINT``, oder ein explizites ``prior_manifest``) sperrt
    zusaetzlich Sessions auf Train. Additiv: nur die uebergebenen Chip-Ordner
    werden hinzugefuegt/ueberschrieben, bestehende Sessions behalten ihren Split;
    ``rebuild=True`` leert Bilder/Labels und teilt neu auf (die Sperrliste bleibt).
    Fehlerhafte Chip-Ordner werden uebersprungen und im Bericht genannt.

    Raises:
        InvalidParameterError: fehlende/ungueltige Pfade oder Parameter.
        DatasetDescriptorError: ``dataset.json`` fehlt/ist beschaedigt/inkonsistent.
        DatasetBuildError: ``acia`` fehlt, obwohl eine Aufnahme geoeffnet werden muss.
    """
    new_data_path = _resolve_path(new_data, NEW_DATA_ENV_VAR, "new_data")
    new_dataset_path = _resolve_path(new_dataset, NEW_DATASET_ENV_VAR, "new_dataset")
    old_dataset_path = _resolve_path(old_dataset, OLD_DATASET_ENV_VAR, "old_dataset", required=False)
    assert new_data_path is not None and new_dataset_path is not None
    check_dataset_paths(new_dataset_path, old_dataset_path)
    if not new_data_path.is_dir():
        raise InvalidParameterError(f"new_data ist kein Ordner: {new_data_path}")
    if not (0 <= val_ratio < 1 and 0 <= test_ratio < 1 and val_ratio + test_ratio < 1):
        raise InvalidParameterError("val_ratio/test_ratio muessen in [0, 1) liegen und zusammen < 1 sein")
    if channel_aspect_threshold <= 1:
        raise InvalidParameterError("channel_aspect_threshold muss > 1 sein")

    forced_val, forced_test, forced_train = set(val_chips), set(test_chips), set(train_chips)
    warnings: list[str] = []

    _, old_locked, old_val_test, old_sources = _old_dataset_view(old_dataset_path)

    checkpoint = base_checkpoint if base_checkpoint is not None else os.environ.get(_CHECKPOINT_ENV_VAR)
    lineage = trained_sessions_of(checkpoint, prior_manifest) or set()

    skipped: dict[str, str] = {}
    with dataset_write_lock(new_dataset_path):
        descriptor = load_descriptor(new_dataset_path)
        if descriptor is None:
            descriptor = new_descriptor(channel_aspect_threshold)
        else:
            if descriptor["class_list"] != class_list():
                raise DatasetDescriptorError(
                    f"{new_dataset_path}: Klassenliste {descriptor['class_list']} passt nicht zur "
                    f"Paket-Klassenliste {class_list()}"
                )
            if descriptor["channel_aspect_ratio_threshold"] != float(channel_aspect_threshold):
                raise InvalidParameterError(
                    f"{new_dataset_path} wurde mit channel_aspect_threshold="
                    f"{descriptor['channel_aspect_ratio_threshold']} gebaut, nicht {channel_aspect_threshold}"
                )
            if not rebuild:
                problems = check_consistency(new_dataset_path, descriptor)
                if problems:
                    raise DatasetDescriptorError(
                        f"{new_dataset_path} ist inkonsistent: " + "; ".join(problems[:5])
                    )

        if rebuild:
            for kind in ("images", "labels"):
                shutil.rmtree(new_dataset_path / kind, ignore_errors=True)
            descriptor["sessions"] = {}
            logger.info("rebuild: images/ und labels/ in %s geleert, Sperrliste bleibt", new_dataset_path)

        new_lock = effective_lock_set(new_dataset_path, descriptor) if not rebuild else set(descriptor["locked_train_sessions"])
        locked = old_locked | lineage | new_lock

        # Kandidaten einlesen
        requested = set(chips) if chips is not None else None
        candidates: dict[str, tuple[Path, dict]] = {}
        for folder in sorted(p for p in new_data_path.iterdir() if p.is_dir()):
            if requested is not None and folder.name not in requested:
                continue
            selection_path = folder / SELECTION_FILENAME
            if not selection_path.exists():
                continue
            try:
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
                selection["source"]["path"], selection["source"]["sizes"]["X"]  # Pflichtfelder pruefen
            except (OSError, ValueError, KeyError, TypeError) as exc:
                skipped[folder.name] = f"selection.json unlesbar oder unvollstaendig: {exc}"
                continue
            candidates[folder.name] = (folder, selection)
        if requested is not None:
            for missing in sorted(requested - set(candidates) - set(skipped)):
                skipped[missing] = f"kein Chip-Ordner mit {SELECTION_FILENAME} unter {new_data_path}"

        # Identitaets- und Leakage-Regeln (AD-23, AD-12)
        new_sources = {n: e.get("source_path") for n, e in descriptor["sessions"].items()}
        seen_sources: dict[str, str] = {}
        for other, src in {**old_sources, **new_sources}.items():
            if src:
                seen_sources[src] = other
        for name in sorted(candidates):
            source_path = candidates[name][1]["source"]["path"]
            if name in old_val_test:
                skipped[name] = "Session liegt im Val/Test des old_dataset und darf nicht in new_data auftauchen"
            elif name in new_sources and new_sources[name] and new_sources[name] != source_path:
                skipped[name] = f"gleicher Name, andere Quelle (bisher {new_sources[name]})"
            elif name in old_sources and old_sources[name] and old_sources[name] != source_path:
                skipped[name] = f"gleicher Name wie eine old_dataset-Session, aber andere Quelle ({old_sources[name]})"
            elif source_path in seen_sources and seen_sources[source_path] != name:
                skipped[name] = f"gleiche Quelle wie Session '{seen_sources[source_path]}' unter anderem Namen"
            else:
                seen_sources[source_path] = name
        for name in skipped:
            candidates.pop(name, None)

        # Split-Zuordnung: bestehende Sessions behalten ihren Split (additiv)
        assignment: dict[str, str] = {}
        to_assign: list[str] = []
        for name in sorted(candidates):
            existing = descriptor["sessions"].get(name)
            if existing is not None and not rebuild:
                assignment[name] = existing["split"]
                if name in forced_val | forced_test | forced_train and assignment[name] != (
                    "val" if name in forced_val else "test" if name in forced_test else "train"
                ):
                    raise InvalidParameterError(
                        f"Session '{name}' liegt bereits im Split {assignment[name]} und kann nicht umgehaengt werden"
                    )
            else:
                to_assign.append(name)
        locked_new = {n for n in to_assign if n in locked}
        assignment.update(
            assign_splits(
                to_assign, val_ratio, test_ratio, seed,
                forced_val & set(to_assign), forced_test & set(to_assign),
                (forced_train | locked_new) & set(to_assign),
            )
        )

        # Verarbeitung
        opener = open_sequence
        if opener is None and assignment:
            opener = _default_open_sequence()  # lazy: acia nur, wenn wirklich etwas zu verarbeiten ist
        images_written = {split: 0 for split in SPLITS}
        processed: dict[str, str] = {}
        for name in sorted(assignment):
            folder, selection = candidates[name]
            split = assignment[name]
            _remove_chip_files(new_dataset_path, name)
            try:
                count = _process_chip(
                    folder, selection,
                    new_dataset_path / "images" / split, new_dataset_path / "labels" / split,
                    opener, use_registration, frames_per_position, channel_aspect_threshold, seed,
                )
            except Exception as exc:  # pro Chip-Ordner isolieren (AD-10-Geist)
                _remove_chip_files(new_dataset_path, name)
                skipped[name] = f"{type(exc).__name__}: {exc}"
                logger.error("chip %s uebersprungen: %s", name, exc)
                continue
            processed[name] = split
            images_written[split] += count
            descriptor["sessions"][name] = {"split": split, "source_path": selection["source"]["path"]}
            if split == "train":
                descriptor["locked_train_sessions"] = sorted(set(descriptor["locked_train_sessions"]) | {name})
            logger.info("chip %s -> %s (%d Bild/Label-Paare)", name, split, count)

        new_dataset_path.mkdir(parents=True, exist_ok=True)
        write_descriptor(new_dataset_path, descriptor)

    counts = {s: sum(1 for e in descriptor["sessions"].values() if e["split"] == s) for s in SPLITS}
    for split in ("val", "test"):
        if counts[split] == 0:
            warnings.append(
                f"new_dataset hat keine {split}-Session: fuer Training nur auf new_data "
                f"val_chips/test_chips angeben, sonst zusammen mit old_dataset trainieren"
            )
    for message in warnings:
        logger.warning(message)
    return BuildReport(
        new_dataset=new_dataset_path,
        sessions=processed,
        images_written=images_written,
        skipped=skipped,
        locked=sorted(locked & set(candidates)),
        warnings=warnings,
        rebuilt=rebuild,
    )
