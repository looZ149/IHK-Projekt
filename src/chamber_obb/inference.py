"""Inferenz-Stufe der chamber_obb Bibliothek (CAP-2).

Fuehrt OBB-Detektion (YOLO) auf einem einzelnen, bereits vorverarbeiteten
(640x640) Bild aus. Laedt das Modell lazy und cached es je aufgeloestem
Checkpoint-Pfad (AD-3); besitzt als einzige Stelle den Confidence-Threshold
und die NMS-Einstellungen (AD-17). Kein Batching (AD-2, AD-10), keine
Disk-Nebenwirkungen.
"""

import logging
import threading
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

logger = logging.getLogger(__name__)

# Einziger benannter Confidence-Threshold der Bibliothek (AD-17). Entspricht
# dem in SPEC.md festgelegten Wert aus training/model.py::inference() (0.8),
# nicht den 0.25-Werten aus train()/checkUp().
DEFAULT_CONFIDENCE = 0.8

# Modell-Cache: aufgeloester Checkpoint-Pfad -> geladenes YOLO-Modell (AD-3).
_MODEL_CACHE: dict[str, YOLO] = {}
# Schuetzt nur den (seltenen, teuren) Erstladevorgang je Checkpoint, nicht
# jeden Cache-Treffer -- siehe Design Notes in der Story fuer die Begruendung.
_CACHE_LOCK = threading.Lock()


def _get_model(checkpoint_path: str | Path) -> YOLO:
    """Laedt das YOLO-Modell fuer ``checkpoint_path`` lazy und cached es.

    Double-checked locking: der ungesicherte Fast-Path-Read ist unter der
    GIL fuer ein einfaches dict.get() sicher; der Lock wird nur fuer den
    tatsaechlichen Ladevorgang beim ersten Aufruf je Pfad genommen, damit
    nebenlaeufige Erstaufrufe nicht doppelt laden oder sich ueberschreiben.

    Raises:
        ValueError: wenn die Datei zwischen dem Existenz-Check in
            ``run_inference`` und diesem Ladevorgang verschwunden ist oder
            ultralytics sie nicht als Checkpoint laden kann (TOCTOU).
    """
    key = str(Path(checkpoint_path).resolve())

    model = _MODEL_CACHE.get(key)
    if model is not None:
        logger.debug("cache hit for checkpoint %s", key)
        return model

    with _CACHE_LOCK:
        model = _MODEL_CACHE.get(key)  # re-check innerhalb des Locks
        if model is not None:
            logger.debug("cache hit for checkpoint %s", key)
            return model
        logger.info("loading checkpoint %s", key)
        try:
            model = YOLO(key)
        except (OSError, TypeError) as exc:
            # OSError: Datei wurde nach dem Existenz-Check geloescht/ersetzt
            # (TOCTOU). TypeError: ultralytics' eigene, konkrete Fehlerklasse
            # fuer eine leere/abgeschnittene/korrupte .pt-Datei (siehe
            # ultralytics.nn.tasks.torch_safe_load).
            logger.error("failed to load checkpoint %s: %s", key, exc)
            raise ValueError(f"failed to load checkpoint {key}: {exc}") from exc
        _MODEL_CACHE[key] = model
        return model


def run_inference(
    image: np.ndarray, checkpoint_path: str | Path, conf: float = DEFAULT_CONFIDENCE
) -> Results:
    """Fuehrt OBB-Detektion auf einem einzelnen, vorverarbeiteten Bild aus.

    ``image`` ist ein bereits letterbox-resiztes 640x640 numpy-Array (Story 2
    ``letterbox_resize``-Output). ``checkpoint_path`` muss existieren -- eine
    zusaetzliche Pruefung, da ultralytics selbst nicht-existente lokale Pfade
    sonst als Download-Alias missinterpretieren koennte. Gibt das rohe
    Einzelbild-``Results``-Objekt zurueck; Koordinatentransformation und
    Filterung sind Sache von ``postprocessing.py`` (Story 4).

    Raises:
        ValueError: wenn ``checkpoint_path`` kein str/Path ist, nicht auf eine
            Datei zeigt, oder ``conf`` keine Zahl in [0, 1] ist.
    """
    if not isinstance(checkpoint_path, (str, Path)):
        logger.error("run_inference: checkpoint_path is not a str or Path: %r", checkpoint_path)
        raise ValueError(f"checkpoint_path must be a str or Path, got {type(checkpoint_path).__name__}")

    if not isinstance(conf, (int, float)) or not (0 <= conf <= 1):
        logger.error("run_inference: conf is not a number in [0, 1]: %r", conf)
        raise ValueError(f"conf must be a number in [0, 1], got {conf!r}")

    resolved = Path(checkpoint_path)
    if not resolved.is_file():
        logger.error("checkpoint path does not exist: %s", checkpoint_path)
        raise ValueError(f"checkpoint path does not exist: {checkpoint_path}")

    model = _get_model(checkpoint_path)
    results = model.predict(image, conf=conf, save=False, verbose=False)
    return results[0]
