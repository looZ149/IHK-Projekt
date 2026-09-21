"""Dataset-Descriptor (``dataset.json``) und Lock-Registry (AD-12, AD-18, AD-23).

Jeder Datensatz-Ordner traegt seine eigene Registry: die Klassenliste, den
Seitenverhaeltnis-Schwellwert, die Session-Tabelle (Chip-Ordnername ->
Split + Quellpfad) und die Menge aller Sessions, die JEMALS in Train waren.
Die Datei liegt im Datensatz-Ordner -- nie im cwd, nie in einem Repo-Root.

Schreibzugriffe sind atomar (Temp-Datei + ``os.replace``) und laufen unter einer
exklusiven ``O_EXCL``-Lockdatei. Ein fehlender oder beschaedigter Descriptor in
einem Ordner mit Bildern ist ein Fehler (fail closed), nie eine leere Registry.
"""

import contextlib
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

from chamber_obb.classes import DEFAULT_CHANNEL_ASPECT_RATIO_THRESHOLD, class_list
from chamber_obb.errors import DatasetDescriptorError

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
DESCRIPTOR_FILENAME = "dataset.json"
LOCK_FILENAME = ".dataset.lock"
SCHEMA_VERSION = 1

# Muss synchron bleiben mit dataset.py (Bild-Schreibstelle):
# f"{chip}_pos{pos}_frame{frame:03d}" -- AD-18 parst genau dieses Format.
_IMAGE_NAME_RE = re.compile(r"^(?P<chip>.+)_pos\d+_frame\d+$")
_IMAGE_SUFFIXES = {".png"}


class SessionEntry(TypedDict):
    split: str
    source_path: str | None


class Descriptor(TypedDict, total=False):
    schema_version: int
    class_list: list[str]
    channel_aspect_ratio_threshold: float
    sessions: dict[str, SessionEntry]
    locked_train_sessions: list[str]
    # Alt-Datensatz eines Modells, das auf ALLEN Sessions trainiert wurde: alle Sessions sind
    # Train, die Bilder unter val/ und test/ sind nur Early-Stopping-Ausschnitte derselben
    # Sessions (kein unabhaengiges Val/Test) und zaehlen als Trainingsdaten.
    all_sessions_trained: bool


def new_descriptor(
    threshold: float = DEFAULT_CHANNEL_ASPECT_RATIO_THRESHOLD,
) -> Descriptor:
    return {
        "schema_version": SCHEMA_VERSION,
        "class_list": class_list(),
        "channel_aspect_ratio_threshold": float(threshold),
        "sessions": {},
        "locked_train_sessions": [],
    }


def list_images(dataset_root: Path, split: str) -> list[Path]:
    split_dir = dataset_root / "images" / split
    if not split_dir.is_dir():
        return []
    return sorted(p for p in split_dir.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)


def dataset_has_images(dataset_root: Path) -> bool:
    return any(list_images(dataset_root, split) for split in SPLITS)


def chip_of_image(path: Path) -> str | None:
    """Chip-Ordnername aus einem Bilddateinamen; ``None`` wenn das Schema nicht passt."""
    match = _IMAGE_NAME_RE.match(path.stem)
    return match.group("chip") if match else None


def scan_sessions_on_disk(dataset_root: Path) -> tuple[dict[str, set[str]], list[Path]]:
    """Sessions je Split aus den Bilddateinamen (AD-18) plus nicht zuordenbare Dateien."""
    by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    unattributable: list[Path] = []
    for split in SPLITS:
        for image in list_images(dataset_root, split):
            chip = chip_of_image(image)
            if chip is None:
                unattributable.append(image)
            else:
                by_split[split].add(chip)
    return by_split, unattributable


def _validate(data: object, path: Path) -> Descriptor:
    if not isinstance(data, dict):
        raise DatasetDescriptorError(f"{path}: Descriptor ist kein JSON-Objekt")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise DatasetDescriptorError(
            f"{path}: schema_version {data.get('schema_version')!r} wird nicht unterstuetzt "
            f"(erwartet {SCHEMA_VERSION})"
        )
    for key in ("class_list", "channel_aspect_ratio_threshold", "sessions", "locked_train_sessions"):
        if key not in data:
            raise DatasetDescriptorError(f"{path}: Pflichtfeld '{key}' fehlt")
    sessions = data["sessions"]
    if not isinstance(sessions, dict):
        raise DatasetDescriptorError(f"{path}: 'sessions' muss ein Objekt sein")
    for name, entry in sessions.items():
        if not isinstance(entry, dict) or entry.get("split") not in SPLITS:
            raise DatasetDescriptorError(f"{path}: Session '{name}' hat keinen gueltigen Split")
    if not isinstance(data.get("all_sessions_trained", False), bool):
        raise DatasetDescriptorError(f"{path}: 'all_sessions_trained' muss ein Boolean sein")
    if not isinstance(data["class_list"], list) or not isinstance(data["locked_train_sessions"], list):
        raise DatasetDescriptorError(f"{path}: 'class_list'/'locked_train_sessions' muessen Listen sein")
    return data  # type: ignore[return-value]


def load_descriptor(dataset_root: Path) -> Descriptor | None:
    """Liest ``dataset.json``.

    Gibt ``None`` nur zurueck, wenn der Ordner weder Descriptor noch Bilder
    enthaelt (leerer, neuer Datensatz). Fehlt der Descriptor bei vorhandenen
    Bildern, oder ist er unlesbar, wird ein ``DatasetDescriptorError`` geworfen
    (AD-23, fail closed) -- niemals eine leere Registry angenommen.
    """
    path = dataset_root / DESCRIPTOR_FILENAME
    if not path.exists():
        if dataset_has_images(dataset_root):
            raise DatasetDescriptorError(
                f"{dataset_root} enthaelt Bilder, aber keine {DESCRIPTOR_FILENAME}. "
                f"Einmalig migrieren: chamber_obb.locks.migrate_legacy_dataset() (siehe Anwenderdokumentation)."
            )
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DatasetDescriptorError(f"{path} ist nicht lesbar: {exc}") from exc
    return _validate(data, path)


def descriptor_hash(dataset_root: Path) -> str | None:
    path = dataset_root / DESCRIPTOR_FILENAME
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


@contextlib.contextmanager
def dataset_write_lock(dataset_root: Path, timeout: float = 30.0) -> Iterator[None]:
    """Exklusive Schreibsperre (``O_EXCL``) fuer einen Datensatz-Ordner (AD-23)."""
    dataset_root.mkdir(parents=True, exist_ok=True)
    lock_path = dataset_root / LOCK_FILENAME
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise DatasetDescriptorError(
                    f"{lock_path} existiert bereits (anderer build_dataset()-Lauf?). Ist sie "
                    f"verwaist, kann sie nach Pruefung manuell geloescht werden."
                ) from None
            time.sleep(0.05)
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            lock_path.unlink()


def write_descriptor(dataset_root: Path, descriptor: Descriptor) -> None:
    """Schreibt ``dataset.json`` atomar. Aufrufer haelt ``dataset_write_lock``."""
    descriptor["locked_train_sessions"] = sorted(set(descriptor["locked_train_sessions"]))
    payload = json.dumps(descriptor, indent=2, sort_keys=True, ensure_ascii=False)
    fd, tmp_name = tempfile.mkstemp(dir=dataset_root, prefix=".dataset-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, dataset_root / DESCRIPTOR_FILENAME)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def all_sessions_trained(descriptor: Descriptor | None) -> bool:
    """True bei einem Alt-Datensatz, dessen Modell auf ALLEN Sessions trainiert wurde."""
    return bool(descriptor and descriptor.get("all_sessions_trained"))


def effective_lock_set(dataset_root: Path, descriptor: Descriptor | None) -> set[str]:
    """Locked-Set = Descriptor-Registry + aktuell unter images/train vorhandene Sessions (AD-18).

    Bei ``all_sessions_trained`` zaehlen auch Sessions unter val/ und test/ (dort liegen nur
    Early-Stopping-Ausschnitte bereits trainierter Sessions).
    """
    on_disk, _ = scan_sessions_on_disk(dataset_root)
    locked = set(on_disk["train"])
    if descriptor is not None:
        locked |= set(descriptor["locked_train_sessions"])
        if all_sessions_trained(descriptor):
            locked |= on_disk["val"] | on_disk["test"] | set(descriptor["sessions"])
    return locked


def images_by_role(dataset_root: Path, descriptor: Descriptor) -> dict[str, list[Path]]:
    """Bilder je Rolle (train/val/test). Bei ``all_sessions_trained`` sind ALLE Bilder Trainingsdaten
    und der Datensatz liefert weder Val noch Test -- er hat keine unabhaengigen Daten."""
    images = {split: list_images(dataset_root, split) for split in SPLITS}
    if all_sessions_trained(descriptor):
        images = {"train": images["train"] + images["val"] + images["test"], "val": [], "test": []}
    return images


def migrate_legacy_dataset(
    dataset_root: str | Path,
    legacy_lock_file: str | Path | None = None,
    all_sessions_trained: bool = True,
) -> dict[str, list[str]]:
    """Einmalige Migration eines Alt-Datensatzes (ohne ``dataset.json``) -- AD-23.

    Braucht Schreibrechte im Datensatz-Ordner; wird bewusst manuell von jemandem
    mit diesen Rechten ausgefuehrt, nie von ``build_dataset()``/``train()``.
    Leitet die Session-Tabelle aus den Bilddateinamen ab (Quellpfade sind
    unbekannt, ``None``) und vereinigt die Sperrliste mit dem Inhalt der alten
    ``train_locked_chips.txt`` (falls angegeben) und den Train-Sessions auf Platte.

    ``all_sessions_trained=True`` (Standard) gilt fuer einen Datensatz, mit dem das
    aktuelle Modell auf ALLEN Sessions trainiert wurde: jede Session wird als ``train``
    gefuehrt und gesperrt, die Bilder unter val/ und test/ (nur Early-Stopping-Ausschnitte
    derselben Sessions) zaehlen als Trainingsdaten -- der Datensatz liefert kein Val/Test.
    Mit ``False`` wird ein Datensatz mit echtem, unabhaengigem Val/Test migriert; Sessions mit
    Dateien in mehreren Splits werden dann als ``train`` gefuehrt und von ``train()`` per
    Konsistenzpruefung verweigert, bis sie bereinigt sind.

    Gibt einen Bericht zurueck: ``straddling`` = Sessions mit Dateien in mehr als einem Split
    (bei ``all_sessions_trained`` nur Information: das Monitoring-Val des alten ``--all-train``).
    """
    root = Path(dataset_root)
    if (root / DESCRIPTOR_FILENAME).exists():
        raise DatasetDescriptorError(f"{root} hat bereits eine {DESCRIPTOR_FILENAME}")
    on_disk, unattributable = scan_sessions_on_disk(root)
    if unattributable:
        names = ", ".join(p.name for p in unattributable[:5])
        raise DatasetDescriptorError(
            f"{len(unattributable)} Bild(er) unter {root} passen nicht zum Schema "
            f"<chip>_pos<N>_frame<M>.png und lassen sich keiner Session zuordnen: {names}"
        )
    descriptor = new_descriptor()
    descriptor["all_sessions_trained"] = bool(all_sessions_trained)
    seen_in: dict[str, set[str]] = {}
    for split in SPLITS:
        for chip in on_disk[split]:
            seen_in.setdefault(chip, set()).add(split)
    straddling = sorted(chip for chip, splits in seen_in.items() if len(splits) > 1)
    for chip, splits in seen_in.items():
        split = "train" if (all_sessions_trained or "train" in splits) else sorted(splits)[0]
        descriptor["sessions"][chip] = {"split": split, "source_path": None}
    locked = set(seen_in) if all_sessions_trained else set(on_disk["train"])
    if legacy_lock_file is not None:
        legacy = Path(legacy_lock_file)
        locked |= {ln.strip() for ln in legacy.read_text(encoding="utf-8").splitlines() if ln.strip()}
    descriptor["locked_train_sessions"] = sorted(locked)
    with dataset_write_lock(root):
        write_descriptor(root, descriptor)
    logger.info("migrated %s: %d sessions, %d straddling", root, len(descriptor["sessions"]), len(straddling))
    return {"sessions": sorted(descriptor["sessions"]), "locked": sorted(locked), "straddling": straddling}


def check_consistency(dataset_root: Path, descriptor: Descriptor) -> list[str]:
    """Findet Widersprueche zwischen Bildern auf Platte und der Session-Tabelle (AD-23).

    Nicht zuordenbare Dateien und Sessions, deren Dateien in einem anderen Split
    liegen als die Tabelle sagt, sind Fehler -- sie wuerden die Lock-Garantie
    still aushebeln.
    """
    problems: list[str] = []
    on_disk, unattributable = scan_sessions_on_disk(dataset_root)
    for image in unattributable:
        problems.append(f"Datei '{image.name}' passt nicht zum Schema <chip>_pos<N>_frame<M>.png")
    trained_all = all_sessions_trained(descriptor)
    for split in SPLITS:
        for chip in sorted(on_disk[split]):
            entry = descriptor["sessions"].get(chip)
            if entry is None:
                problems.append(f"Session '{chip}' hat Dateien in {split}/, fehlt aber in der Session-Tabelle")
            elif trained_all:
                continue  # Monitoring-Val/Test-Dateien bereits trainierter Sessions sind erlaubt
            elif entry["split"] != split:
                problems.append(
                    f"Session '{chip}' hat Dateien in {split}/, die Tabelle fuehrt sie als {entry['split']}"
                )
    return problems
