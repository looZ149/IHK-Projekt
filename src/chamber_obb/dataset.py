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
from chamber_obb.util import resolve_path

logger = logging.getLogger(__name__)

REGISTRATION_FILENAME = "registration_transforms.json"
SELECTION_FILENAME = "selection.json"
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


def _forward_matrix(shape: tuple[int, int], transform: dict) -> np.ndarray:
    """2x3-Matrix Referenz-Frame -> Frame N (acia-Konvention, ``acia.registration._correction_matrix``).

    ``transform`` = ``{"dx", "dy", "theta"}``: Rotation um die Mitte des GANZEN Frames
    (``cv2.getRotationMatrix2D``, ``theta`` in GRAD, gegen den Uhrzeigersinn), danach Verschiebung um
    ``(dx, dy)``. ``shape`` ist ``(H, W)`` des vollen Frames.
    """
    height, width = int(shape[0]), int(shape[1])
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), float(transform["theta"]), 1.0)
    matrix[0, 2] += float(transform["dx"])
    matrix[1, 2] += float(transform["dy"])
    return matrix


_IDENTITY = {"dx": 0.0, "dy": 0.0, "theta": 0.0}


def _transform_of(frame: int, transforms: dict[int, dict], reference_frame: int) -> dict:
    """Transform eines Frames; der Referenz-Frame ist exakt die Identitaet (wie ``acia``-Export)."""
    if frame == reference_frame:
        return _IDENTITY
    return transforms[frame]


def propagate_roi_across_frames(
    roi: dict,
    anchor_frame: int,
    target_frames: list[int],
    transforms: dict[int, dict],
    reference_frame: int,
    shape: tuple[int, int],
) -> dict[int, dict]:
    """Traegt eine auf ``anchor_frame`` gezeichnete ROI in die Roh-Koordinaten der Ziel-Frames.

    Geometrie exakt wie ``acia`` (``apply_correction_to_spec``): die gespeicherten Transformationen
    gehen Referenz -> Frame N (Rotation um die Frame-Mitte, ``theta`` in Grad, dann ``(dx, dy)``).
    Punkt: ``p_ref = M_anchor^-1 p_anchor``, ``p_ziel = M_ziel p_ref``; Winkel:
    ``angle_ziel = angle + theta_anchor - theta_ziel`` (in Referenz-Koordinaten gilt
    ``angle_ref = angle + theta``). Die Groesse bleibt unveraendert (starre Registrierung).
    Ziel-Frames ohne Transform werden ausgelassen; fehlt der Anker-Transform, wirft die Funktion ``KeyError``.

    ``roi``: ``{"center": [x, y], "size": [w, h], "angle": Grad}``.
    """
    anchor_t = _transform_of(anchor_frame, transforms, reference_frame)
    to_reference = cv2.invertAffineTransform(_forward_matrix(shape, anchor_t))
    point = np.asarray(roi["center"], dtype=np.float64)
    p_ref = to_reference[:, :2] @ point + to_reference[:, 2]
    result: dict[int, dict] = {}
    for frame in target_frames:
        if frame != reference_frame and frame not in transforms:
            continue
        target_t = _transform_of(frame, transforms, reference_frame)
        forward = _forward_matrix(shape, target_t)
        p_target = forward[:, :2] @ p_ref + forward[:, 2]
        result[frame] = {
            "center": [float(p_target[0]), float(p_target[1])],
            "size": roi["size"],
            "angle": float(roi["angle"]) + float(anchor_t["theta"]) - float(target_t["theta"]),
        }
    return result


def load_registration(registration_path: Path) -> dict[int, dict]:
    """Laedt ``registration_transforms.json`` -> Position -> {transforms, reference_frame}.

    ``theta`` bleibt in Grad (acia-Konvention); ``confidence`` wird nicht gebraucht.
    """
    data = json.loads(Path(registration_path).read_text(encoding="utf-8"))
    by_position: dict[int, dict] = {}
    for record in data.get("records", []):
        transforms = {
            int(frame_idx): {"dx": t["dx"], "dy": t["dy"], "theta": t.get("theta", 0.0)}
            for frame_idx, t in record.get("transforms", {}).items()
        }
        by_position[record["position"]] = {
            "transforms": transforms,
            "reference_frame": int(record.get("reference_frame", 0)),
        }
    return by_position


def select_target_frames(
    valid_frames: set[int], must_include: Iterable[int], count: int | None, seed: int, seed_key: str = ""
) -> list[int]:
    """Waehlt reproduzierbar (seed + seed_key) Ziel-Frames aus den registrierten Frames.

    ``count=None``: alle. Die Anker-Frames (``must_include``) sind immer dabei. Die Auswahl betrifft
    nur, WELCHE Frames exportiert werden -- die Boxen werden pro Frame korrekt ueber die
    Registrierung berechnet.
    """
    frames = sorted(valid_frames | set(must_include))
    required = sorted(set(must_include))
    if count is None or count >= len(frames):
        return frames
    remaining = [f for f in frames if f not in required]
    digest = hashlib.sha256(f"{seed}:{seed_key}".encode()).hexdigest()
    random.Random(int(digest, 16)).shuffle(remaining)
    chosen = set(required) | set(remaining[: max(0, max(1, count) - len(required))])
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
) -> tuple[int, list[str]]:
    """Schreibt Bild/Label-Paare eines Chip-Ordners; gibt (Anzahl, Warnungen) zurueck.

    Jede ROI gehoert zu dem Frame, auf dem sie gezeichnet wurde (``anchor_frame`` in
    ``selection.json``; fehlt das Feld, ist es Frame 0, wie bei aelteren Dateien). Das Dashboard
    oeffnet standardmaessig auf dem MITTLEREN Frame -- ein Bild wird deshalb immer vom Anker-Frame
    exportiert, nie stillschweigend von Frame 0.

    - Ohne Registrierung: nur Anker-Frames. Liegen die ROIs einer Position auf verschiedenen
      Frames, wuerde jedes Bild ROIs der anderen Frames unbeschriftet lassen (falsche Negative) --
      diese Position wird uebersprungen und gemeldet.
    - Mit Registrierung: Anker-Frames plus (``frames_per_position``) weitere registrierte Frames;
      alle ROIs werden per ``propagate_roi_across_frames`` in jeden Ziel-Frame getragen.
    """
    source = selection["source"]
    img_w, img_h = source["sizes"]["X"], source["sizes"]["Y"]
    shape = (img_h, img_w)

    entries_by_position: dict[int, list[tuple[dict, int]]] = {}
    for sel in selection.get("selections", []):
        entries_by_position.setdefault(sel["position"], []).append((sel["roi"], int(sel.get("anchor_frame", 0))))
    if not entries_by_position:
        raise DatasetBuildError("keine Annotationen in selection.json")

    registration_by_position: dict[int, dict] = {}
    if use_registration:
        registration_path = folder / REGISTRATION_FILENAME
        if registration_path.exists():
            registration_by_position = load_registration(registration_path)
        else:
            logger.warning(
                "%s: use_registration gesetzt, aber %s fehlt -- nur Anker-Frames (kein unkorrigierter "
                "Ersatz-Frame, um keine falschen Labels zu erzeugen)",
                folder.name, REGISTRATION_FILENAME,
            )

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    warnings: list[str] = []
    with open_sequence(source["path"]) as seq:
        for pos, entries in sorted(entries_by_position.items()):
            pos_source = seq.position(pos)
            anchors = sorted({anchor for _, anchor in entries})
            reg = registration_by_position.get(pos)
            transforms = reg["transforms"] if reg else {}
            reference_frame = reg["reference_frame"] if reg else 0

            if reg is None and len(anchors) > 1:
                warnings.append(
                    f"{folder.name} Position {pos}: ROIs auf verschiedenen Frames {anchors} brauchen "
                    f"eine Registrierung -- Position uebersprungen"
                )
                continue
            missing = [a for a in anchors if reg is not None and a != reference_frame and a not in transforms]
            if missing:
                warnings.append(
                    f"{folder.name} Position {pos}: kein Registrierungs-Transform fuer Anker-Frame(s) "
                    f"{missing} -- Position uebersprungen"
                )
                continue

            target_frames = anchors
            if reg is not None:
                try:
                    target_frames = select_target_frames(
                        set(transforms) | {reference_frame}, anchors, frames_per_position, seed,
                        seed_key=f"{folder.name}_pos{pos}",
                    )
                except Exception:  # defekte Registrierung: bei den Anker-Frames bleiben
                    target_frames = anchors

            for frame_idx in target_frames:
                if reg is None:
                    frame_rois = [roi for roi, _ in entries]
                else:
                    frame_rois = [
                        propagate_roi_across_frames(
                            roi, anchor, [frame_idx], transforms, reference_frame, shape
                        )[frame_idx]
                        for roi, anchor in entries
                    ]

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
    if written == 0:
        raise DatasetBuildError("keine verwertbare Position: " + "; ".join(warnings or ["keine Frames exportiert"]))
    return written, warnings


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
    new_data: str | Path,
    new_dataset: str | Path,
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
    einzige Ort, in den geschrieben wird. ``old_dataset`` (optional): geteilter Alt-Datensatz,
    nur lesend -- seine Sperrliste und Val/Test-Sessions schuetzen den neuen
    Datensatz vor Leakage (AD-12). Alle Pfade werden explizit uebergeben (AD-8).

    Die Trainings-Historie des Basis-Checkpoints (``base_checkpoint`` oder ein
    explizites ``prior_manifest``) sperrt zusaetzlich Sessions auf Train. Additiv: nur die uebergebenen Chip-Ordner
    werden hinzugefuegt/ueberschrieben, bestehende Sessions behalten ihren Split;
    ``rebuild=True`` leert Bilder/Labels und teilt neu auf (die Sperrliste bleibt).
    Fehlerhafte Chip-Ordner werden uebersprungen und im Bericht genannt.

    Raises:
        InvalidParameterError: fehlende/ungueltige Pfade oder Parameter.
        DatasetDescriptorError: ``dataset.json`` fehlt/ist beschaedigt/inkonsistent.
        DatasetBuildError: ``acia`` fehlt, obwohl eine Aufnahme geoeffnet werden muss.
    """
    new_data_path = resolve_path(new_data, "new_data")
    new_dataset_path = resolve_path(new_dataset, "new_dataset")
    old_dataset_path = resolve_path(old_dataset, "old_dataset", required=False)
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

    lineage = trained_sessions_of(base_checkpoint, prior_manifest) or set()

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
                count, chip_warnings = _process_chip(
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
            warnings.extend(chip_warnings)
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
                f"new_dataset hat keine {split}-Session: val_chips/test_chips aus den neuen Chips "
                f"angeben oder mehr neue Chips liefern (ein Alt-Datensatz mit all_sessions_trained "
                f"liefert kein Val/Test)"
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
