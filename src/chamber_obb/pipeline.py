"""Oeffentlicher Einstiegspunkt der chamber_obb Bibliothek (CAP-2, CAP-4, AD-2).

Orchestriert preprocess -> infer -> postprocess hinter einem einzigen Aufruf,
``analyze()``. Das ist die einzige Funktion, die ``chamber_obb/__init__.py``
als oeffentliche API exportiert -- ``preprocessing``/``inference``/
``postprocessing`` bleiben Implementierungsdetail (AD-2).
"""

import logging
import os
from pathlib import Path
from typing import TypedDict

import numpy as np

from chamber_obb.inference import DEFAULT_CONFIDENCE, run_inference
from chamber_obb.postprocessing import Detection, postprocess
from chamber_obb.preprocessing import letterbox_resize, load_and_normalize, normalize_array

logger = logging.getLogger(__name__)

_CHECKPOINT_ENV_VAR = "CHAMBER_OBB_CHECKPOINT"


class FrameMetadata(TypedDict, total=False):
    """Optionaler, typisierter Metadaten-Payload fuer ``analyze()`` (AD-19).

    Startet bewusst mit null Feldern -- das Schema soll wachsen koennen,
    sobald das reale Dashboard-Notebook gesehen wurde, ohne eine der beiden
    Seiten zu brechen. ``analyze()`` verzweigt niemals auf ein Feld hiervon.
    """


def analyze(
    image: np.ndarray | str | Path,
    metadata: FrameMetadata | None = None,
    checkpoint_path: str | Path | None = None,
    conf: float = DEFAULT_CONFIDENCE,
) -> list[Detection]:
    """Fuehrt die vollstaendige Detection-Pipeline auf einem einzelnen Frame aus.

    ``image`` ist entweder ein numpy-Array (ein bereits extrahiertes Frame,
    das dieselbe 8-bit/3-Kanal-Normalisierung wie ein TIFF-Pfad noch braucht)
    oder ein ``.tif``/``.tiff``-Pfad (``str``/``Path``) (AD-6). Beide Formen
    durchlaufen dieselbe Normalisierung -- ein Array gilt nicht als bereits
    normalisiert.

    ``checkpoint_path`` ist optional; fehlt er, wird die Umgebungsvariable
    ``CHAMBER_OBB_CHECKPOINT`` gelesen (AD-8, AD-15). Ist keins von beiden
    gesetzt, wird ein ``ValueError`` geworfen, bevor ``inference.py``
    ueberhaupt beruehrt wird -- kein stiller Default.

    ``metadata`` wird entgegengenommen, aber (noch) nirgends verwendet oder
    verzweigt (AD-19) -- die Signatur ist bereits stabil fuer eine spaetere
    Dashboard-Integration.

    Einzelnes Frame, synchron, kein Batching (AD-7, AD-10). Ausnahmen aus
    ``load_and_normalize``/``normalize_array``/``letterbox_resize``/
    ``run_inference``/``postprocess`` propagieren unveraendert als deren
    aktuelle einfache ``ValueError``s (kein neues Fehler-Typing, das ist
    Story 6).

    Raises:
        ValueError: wenn ``image`` weder ein numpy-Array noch ein str/Path
            ist, wenn ein Array-``image`` eine Form hat, die nicht Graustufen
            (2D) oder 3-Kanal (``ndim == 3`` mit letzter Achse ``== 3``) ist,
            oder wenn weder ``checkpoint_path`` noch die Umgebungsvariable
            ``CHAMBER_OBB_CHECKPOINT`` gesetzt sind.
    """
    if isinstance(image, (str, Path)):
        normalized = load_and_normalize(image)
    elif isinstance(image, np.ndarray):
        if (image.ndim == 3 and image.shape[-1] != 3) or image.ndim >= 4:
            logger.error("analyze: array image has an unsupported shape: %r", image.shape)
            raise ValueError(
                f"image array must be 2D grayscale or 3D with 3 channels, got shape {image.shape}"
            )
        normalized = normalize_array(image)
    else:
        logger.error("analyze: image is neither a numpy array nor a str/Path: %r", type(image))
        raise ValueError(f"image must be a numpy array or str/Path, got {type(image).__name__}")

    resolved_checkpoint = checkpoint_path or os.environ.get(_CHECKPOINT_ENV_VAR)
    if not resolved_checkpoint:
        logger.error(
            "analyze: no checkpoint_path given and %s is not set", _CHECKPOINT_ENV_VAR
        )
        raise ValueError(
            f"No checkpoint path given and {_CHECKPOINT_ENV_VAR} is not set"
        )

    resized, letterbox_info = letterbox_resize(normalized)
    results = run_inference(resized, resolved_checkpoint, conf=conf)
    detections = postprocess(results, letterbox_info)
    logger.info("analyze: %d detection(s)", len(detections))
    return detections
