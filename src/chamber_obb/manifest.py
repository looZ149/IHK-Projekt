"""Run-Manifest eines ``train()``-Laufs (AD-9, AD-22, AD-24).

Das Manifest ist die Provenienz des Laufs UND die Quelle der Trainings-Historie
fuer das Leakage-Gate: ``trained_sessions`` ist kumulativ (transitiv ueber alle
Vorgaenger-Laeufe), sodass ein spaeterer Lauf nur das Manifest seines
Basis-Checkpoints lesen muss.
"""

import json
import logging
import os
import tempfile
from pathlib import Path

from chamber_obb.errors import ManifestError

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1


def find_manifest(checkpoint: str | Path) -> Path | None:
    """Sucht das Manifest neben den Gewichten (AD-24).

    Kandidaten: ``<ckpt-stem>.manifest.json`` (kopierter Checkpoint), dann
    ``manifest.json`` im Gewichte-Ordner und bis zu zwei Ebenen darueber
    (``<run>/fit/weights/best.pt`` -> ``<run>/manifest.json``).
    """
    ckpt = Path(checkpoint)
    candidates = [ckpt.with_name(f"{ckpt.stem}.manifest.json")]
    candidates += [parent / MANIFEST_FILENAME for parent in list(ckpt.parents)[:3]]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_manifest(path: str | Path) -> dict:
    """Liest und validiert ein Manifest (Schema-Version, ``trained_sessions``)."""
    manifest_path = Path(path)
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(f"{manifest_path} ist nicht lesbar: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(f"{manifest_path}: unbekannte oder fehlende schema_version")
    if not isinstance(data.get("trained_sessions"), list):
        raise ManifestError(f"{manifest_path}: 'trained_sessions' fehlt oder ist keine Liste")
    return data


def trained_sessions_of(checkpoint: str | Path | None, manifest: str | Path | None = None) -> set[str] | None:
    """Trainings-Historie eines Checkpoints; ``None`` wenn sie unbekannt ist."""
    path = Path(manifest) if manifest is not None else (find_manifest(checkpoint) if checkpoint else None)
    if path is None:
        return None
    return set(load_manifest(path)["trained_sessions"])


def write_manifest(path: Path, data: dict) -> None:
    """Schreibt das Manifest atomar (Temp-Datei + ``os.replace``)."""
    payload = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".manifest-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
