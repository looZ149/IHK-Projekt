"""acia-Dashboard-Integration fuer chamber_obb (CAP-1/2/3, AD-1/2/3/4/5/6/7/10/11).

Einzige oeffentliche Funktion: ``open_annotated_dashboard()``. Sie fuehrt
``chamber_obb.pipeline.analyze()`` auf einem repraesentativen Frame je Position
einer ``acia``-Sequenz aus, wandelt jede Detektion in eine ``acia.selection.
RoiSelection`` um, baut daraus ein ``SelectionManifest`` und oeffnet damit
``acia.notebook.SequenceDashboard`` ueber ``.resume()`` -- der Nutzer startet
mit den vom Modell vorgeschlagenen ROIs statt einer leeren Leinwand.

``acia`` wird ausschliesslich hier importiert (AD-2) -- nie in
``chamber_obb/__init__.py`` oder einem der Core-Module
(``pipeline``/``preprocessing``/``inference``/``postprocessing``). Dieses
Modul haengt einwaerts nur von ``chamber_obb.pipeline``/``chamber_obb.errors``
ab und niemals umgekehrt (Core kennt ``acia`` nicht).
"""

from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING

import numpy as np

from acia.base import RotatedCropSpec
from acia.notebook import SequenceDashboard
from acia.segm.open import open_sequence
from acia.selection import RoiSelection, SelectionManifest, make_source_block

from chamber_obb.errors import (
    ChamberOBBError,
    DashboardIntegrationError,
    InvalidParameterError,
    ModelLoadError,
)
from chamber_obb.pipeline import analyze

if TYPE_CHECKING:
    from chamber_obb.postprocessing import Detection

logger = logging.getLogger(__name__)

__all__ = ["open_annotated_dashboard"]

# Deckt sich mit chamber_obb.inference.DEFAULT_CONFIDENCE (AD-4's Signatur
# nennt den Wert explizit, statt von inference.py zu importieren -- dashboard.py
# haengt einwaerts nur von pipeline/errors ab, siehe Architecture Spine).
_DEFAULT_CONFIDENCE = 0.8


def _resolve_frame_index(source, preview_frame: int | None) -> int:
    """Ein einziger Frame-Index fuer die gesamte Sequenz (AD-5).

    Identische Formel wie ``SequenceDashboard.__init__``: die mittlere
    Zeitaufnahme, sofern ``preview_frame`` nicht explizit gesetzt ist, geklemmt
    auf ``[0, num_timepoints - 1]``. Dieser eine Wert steuert Detektion,
    ``anchor_frame`` jeder erzeugten ``RoiSelection`` und das an
    ``SequenceDashboard.resume()`` weitergereichte ``preview_frame`` -- nie
    dreimal unabhaengig berechnet.

    Raises:
        InvalidParameterError: wenn ``preview_frame`` negativ ist -- wie
            ``SequenceDashboard.__init__`` selbst (dort ``traitlets.
            TraitError``), das ebenfalls nicht auf 0 klemmt, sondern ablehnt.
    """
    num_t = max(1, int(source.metadata.num_timepoints or 1))
    chosen = num_t // 2 if preview_frame is None else int(preview_frame)
    if chosen < 0:
        raise InvalidParameterError(f"preview_frame must be >= 0, got {chosen}")
    return min(num_t - 1, chosen)


def _corners_to_rotated_crop_spec(corners: np.ndarray) -> RotatedCropSpec:
    """Wandelt die 4 Eckpunkte eines ``Detection.corners``-OBB in eine
    ``acia.base.RotatedCropSpec`` um (CAP-1).

    Keine Kanonisierung -- Eckpunkte werden exakt in der von ``chamber_obb``
    zurueckgegebenen Reihenfolge verwendet (Consistency Conventions):
    ``center`` ist der Mittelwert aller 4 Punkte; eine ``size``-Dimension
    stammt aus ``|corners[1]-corners[0]|``, die andere aus
    ``|corners[2]-corners[1]|``; ``angle`` aus ``atan2`` der Kante
    ``corners[1]-corners[0]``, in Grad, passend zu OpenCVs
    ``getRotationMatrix2D``-Konvention (gegen den Uhrzeigersinn).
    """
    pts = np.asarray(corners, dtype=float)
    center = pts.mean(axis=0)
    edge_01 = pts[1] - pts[0]
    edge_12 = pts[2] - pts[1]
    width = float(np.linalg.norm(edge_01))
    height = float(np.linalg.norm(edge_12))
    angle = math.degrees(math.atan2(edge_01[1], edge_01[0]))
    return RotatedCropSpec(
        center=(float(center[0]), float(center[1])),
        size=(int(round(width)), int(round(height))),
        angle=angle,
    )


def _detect_rois_for_frames(
    position: int,
    frame_indices: list[int],
    frames: list[np.ndarray],
    *,
    checkpoint_path,
    conf: float,
) -> list[RoiSelection]:
    """Detektiert + konvertiert fuer eine Liste von Frames einer Position.

    Nimmt bewusst eine *Liste* von Frames entgegen, auch wenn v1 diese Liste
    immer mit genau einem Eintrag aufruft (AD-6) -- eine spaetere Erweiterung
    auf Mehrframe-Aggregation ueber die Zeitreihe einer Position wird damit
    additiv statt eines Redesigns.

    Wirft ``analyze()`` fuer einen Frame, wird das (ausser bei
    ``ModelLoadError``, das einen systemischen Konfigurationsfehler anzeigt
    und unveraendert propagieren muss, AD-7) als Warnung geloggt und als null
    Detektionen fuer diesen Frame behandelt -- der Batch laeuft weiter (AD-11).
    Jede Detektion oberhalb ``conf`` wird zu genau einer ``RoiSelection``, ohne
    Filterung/Padding (AD-10).
    """
    selections: list[RoiSelection] = []
    for frame_index, frame in zip(frame_indices, frames):
        try:
            detections: list[Detection] = analyze(frame, checkpoint_path=checkpoint_path, conf=conf)
        except ModelLoadError:
            raise
        except ChamberOBBError as exc:
            logger.warning(
                "open_annotated_dashboard: analyze() failed for position %d "
                "(frame %d): %s",
                position,
                frame_index,
                exc,
            )
            continue

        for detection in detections:
            selections.append(
                RoiSelection(
                    position=position,
                    roi=_corners_to_rotated_crop_spec(detection.corners),
                    label=detection.cls_name,
                    anchor_frame=frame_index,
                )
            )
    return selections


def _flatten_resumed_selections(dash: SequenceDashboard) -> None:
    """Workaround fuer einen acia-core-Bug (siehe ``docs/issues.md``):
    ``SequenceDashboard.resume()`` setzt ``dash.selections`` im
    verschachtelten On-Disk-Format (``{"roi": {"center", "size", "angle"}}``),
    aber das Widget-JS liest beim Rendern ausschliesslich das flache Format
    (``x``/``y``/``w``/``h``/``angle`` direkt auf jedem Eintrag), wie es
    interaktiv gezeichnete ROIs verwenden -- nichts konvertiert zwischen
    beiden. Eine resumte Box wird dadurch nur als toter Punkt gerendert und
    stuerzt beim Anklicken ab (``sel.angle.toFixed`` auf ``undefined``).

    Laeuft, bevor das Widget jemals angezeigt wird (Jupyter loest anywidgets
    JS-``render()`` erst beim Anzeigen von ``dash`` aus, nicht bei dieser
    Zuweisung) -- die Korrektur hier reicht deshalb aus, kein erneutes
    Rendern noetig. Ein No-Op fuer jeden bereits flachen Eintrag, damit
    dieser Workaround harmlos bleibt, sobald der Upstream-Bug behoben ist.
    """
    fixed = []
    for sel in dash.selections:
        roi = sel.get("roi")
        if roi is None:
            fixed.append(sel)
            continue
        sel = dict(sel)
        roi = sel.pop("roi")
        sel["x"], sel["y"] = roi["center"]
        sel["w"], sel["h"] = roi["size"]
        sel["angle"] = roi["angle"]
        fixed.append(sel)
    dash.selections = fixed


def open_annotated_dashboard(
    source,
    checkpoint_path: str | os.PathLike | None = None,
    conf: float = _DEFAULT_CONFIDENCE,
    roi_mode: str = "multi",
    save_dir=None,
    preview_frame: int | None = None,
) -> SequenceDashboard:
    """Fuehrt Detektion auf jeder Position aus und oeffnet das vorbefuellte
    Dashboard (CAP-1, CAP-2, CAP-3).

    Fuehrt ``chamber_obb.analyze()`` auf einem repraesentativen Frame je
    Position aus (die mittlere Zeitaufnahme, AD-5), wandelt jede Detektion in
    eine ``RoiSelection`` um (CAP-1), baut daraus ein ``SelectionManifest`` und
    oeffnet ``acia.notebook.SequenceDashboard`` ueber ``.resume()`` (CAP-3).
    Synchron -- der Aufruf blockiert, bis das Dashboard bereit ist.

    Args:
        source: Ein Pfad/str (wird via ``acia.segm.open.open_sequence``
            geoeffnet) oder eine bereits geoeffnete ``acia``-Sequenz --
            dieselbe Konvention wie ``SequenceDashboard.resume()`` selbst,
            damit die Datei nicht doppelt geoeffnet wird.
        checkpoint_path: Wie bei ``chamber_obb.analyze()``; fehlt er, wird
            ``CHAMBER_OBB_CHECKPOINT`` gelesen.
        conf: Confidence-Schwelle, an ``analyze()`` weitergereicht.
        roi_mode: ``"single"`` (<=1 ROI/Position, Chambers-Chips) oder
            ``"multi"`` (Channels-Chips, Default -- der permissive Modus, der
            jede Anzahl inklusive null akzeptiert). Wird niemals aus der
            Detektionsanzahl abgeleitet/ueberschrieben (AD-10).
        save_dir: Standard-Ausgabeverzeichnis fuer ``SequenceDashboard.save``.
        preview_frame: Wie bei ``SequenceDashboard`` -- ``None`` waehlt die
            mittlere Zeitaufnahme (AD-5).

    Returns:
        Ein ``SequenceDashboard``, dessen Galerie einen Punkt-Marker auf jeder
        Position mit Detektionen zeigt und dessen Editor die vorbefuellten
        Boxen beim Oeffnen rendert.

    Raises:
        DashboardIntegrationError: fuer jeden von ``acia`` ausgehenden Fehler
            beim Oeffnen der Sequenz, beim Bauen des Manifests oder bei
            ``SequenceDashboard.resume()``, der das Oeffnen des Dashboards
            insgesamt verhindert (AD-7).
        ChamberOBBError: propagiert unveraendert -- z.B. ``InvalidParameterError``
            fuer ein negatives ``preview_frame``, oder ``ModelLoadError`` bei
            fehlendem Checkpoint (AD-7), da beides kein isolierbarer Fehler
            einer einzelnen Position ist -- ein Fehler fuer eine einzelne
            Position wird stattdessen geloggt und uebersprungen (AD-11).
    """
    opened_here = isinstance(source, (str, os.PathLike))
    try:
        seqfile = open_sequence(source) if opened_here else source
    except Exception as exc:
        raise DashboardIntegrationError(f"failed to open acquisition: {exc}") from exc

    # Nur diese Ausnahme-Behandlung dient dem Schliessen von seqfile im
    # Fehlerfall (nur, wenn diese Funktion die Datei selbst geoeffnet hat --
    # ein vom Aufrufer uebergebenes source bleibt unangetastet) -- sie wrappt
    # nichts in DashboardIntegrationError; das macht ausschliesslich der
    # innere try-Block unten fuer die tatsaechlich acia-seitigen Aufrufe.
    try:
        frame_index = _resolve_frame_index(seqfile, preview_frame)

        selections: list[RoiSelection] = []
        for position_index in range(seqfile.num_positions):
            frame = seqfile.position(position_index).get_frame(frame_index).get_channel(0)
            selections.extend(
                _detect_rois_for_frames(
                    position_index,
                    [frame_index],
                    [frame],
                    checkpoint_path=checkpoint_path,
                    conf=conf,
                )
            )

        try:
            manifest = SelectionManifest(
                source=make_source_block(seqfile),
                selections=selections,
                roi_mode=roi_mode,
            )
            dash = SequenceDashboard.resume(
                manifest,
                source=seqfile,
                save_dir=save_dir,
                preview_frame=frame_index,
            )
            _flatten_resumed_selections(dash)
            return dash
        except ChamberOBBError:
            raise
        except Exception as exc:
            raise DashboardIntegrationError(f"failed to open dashboard: {exc}") from exc
    except Exception:
        if opened_here:
            seqfile.close()
        raise
