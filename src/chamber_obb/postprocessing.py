"""Postprocessing-Stufe der chamber_obb Bibliothek (CAP-2, AD-14).

Einzige Stelle, an der Koordinaten aus dem 640x640-letterbox-Raum (Story 2)
zurueck in den Pixel-Raum des urspruenglich uebergebenen Bildes transformiert
werden (AD-14). ``inference.py`` (Story 3) fuehrt selbst keine
Koordinatentransformation durch -- das bleibt ausschliesslich hier.
"""

import logging
from typing import NamedTuple

import numpy as np
from ultralytics.engine.results import Results

from chamber_obb.preprocessing import LetterboxInfo

logger = logging.getLogger(__name__)


class Detection(NamedTuple):
    """Eine einzelne Detektion in Original-Pixel-Raum-Koordinaten."""

    cls_id: int
    cls_name: str
    confidence: float
    corners: np.ndarray


def invert_letterbox_coords(coords: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    """Invertiert Story 2s Letterbox-Transform: ``orig = (letterboxed - pad) / scale``.

    Arbeitet auf jedem Array mit einer abschliessenden ``(..., 2)``-Achse
    (ein einzelner Punkt, eine achsparallele Box oder die 4 Eckpunkte eines
    OBB), ohne separate Codepfade je Form. Eckpunkte, die legitim im
    Padding-Bereich liegen, werden bewusst NICHT auf die Originalbild-Grenzen
    geclippt -- das ist korrekte Information, kein zu verbergender Fehler.
    """
    coords = np.asarray(coords, dtype=float).copy()
    coords[..., 0] = (coords[..., 0] - info.pad_left) / info.scale
    coords[..., 1] = (coords[..., 1] - info.pad_top) / info.scale
    return coords


def _to_numpy(value: object) -> np.ndarray:
    """Normalisiert einen ultralytics-Tensor (torch oder bereits numpy) zu numpy."""
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return np.asarray(value)


def postprocess(results: Results, letterbox_info: LetterboxInfo) -> list[Detection]:
    """Stage-3 Entry-Point: baut aus rohem ``Results`` die oeffentliche Detection-Liste.

    Gibt bei null Detektionen eine leere Liste zurueck (kein Fehler). Klassennamen
    stammen ausschliesslich aus ``results.names`` -- niemals hartkodiert (AD-13).
    Keine Best-of-Class-Filterung (CAP-2): jede Detektion oberhalb des in
    ``inference.py`` angewandten Confidence-Thresholds wird durchgereicht.
    """
    obb = results.obb
    if obb is None:
        logger.debug("postprocess: zero detections (results.obb is None)")
        return []
    if len(obb) == 0:
        logger.debug("postprocess: zero detections (results.obb has 0 rows)")
        return []

    corners = _to_numpy(obb.xyxyxyxy)
    confidences = _to_numpy(obb.conf)
    class_ids = _to_numpy(obb.cls)

    detections: list[Detection] = []
    for i in range(len(obb)):
        cls_id = int(class_ids[i])
        detections.append(
            Detection(
                cls_id=cls_id,
                cls_name=results.names[cls_id],
                confidence=float(confidences[i]),
                corners=invert_letterbox_coords(corners[i], letterbox_info),
            )
        )

    logger.info("postprocess: %d detection(s)", len(detections))
    return detections
