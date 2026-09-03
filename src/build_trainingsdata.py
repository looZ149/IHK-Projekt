"""Pipeline: Chip-Ordner (selection.json + ND2 [+ registration_transforms.json])
-> trainingsfertiger YOLO-OBB-Datensatz.

Der Annotations-Frame ist projektweit fest auf Frame 0 (ANNOT_FRAME) gesetzt -
alle ROIs werden auf diesem Frame gezeichnet, kein Raten und keine CLI-Angabe
mehr noetig. Frame 0 ist zugleich der reference_frame der Registrierung, daher
ist dort die Transformation immer die Identitaet (dx=0, dy=0, theta=0).

Registrierungsmodus (--use-registration): nutzt registration_transforms.json,
um die auf Frame 0 gezeichnete ROI korrekt auf beliebig viele andere Frames
derselben Position zu uebertragen - inklusive Ausgleich der Stage-Drift
(Kammer verschiebt sich leicht zwischen Frames).

Grundmodus (keine Registrierung fuer eine Position): es wird ausschliesslich
Frame 0 exportiert - der tatsaechlich annotierte Frame, garantiert korrekt.
Kein zufaelliger Ersatz-Frame, da dessen Box ohne Registrierungsdaten falsch
positioniert waere. Fuer mehrere, positionsgenaue Frames pro Position
--use-registration nutzen (setzt registration_transforms.json voraus).

Vor jedem Lauf werden images/{train,val,test} und labels/{train,val,test}
komplett geleert und neu angelegt, damit der Datensatz immer konsistent zur
aktuellen Split-Zuordnung ist.

Verwendung:
    python build_training_dataset.py --dataset-root dataset
    python build_training_dataset.py --dataset-root dataset \
        --use-registration --frames-per-position 20

Abhaengigkeiten: acia, pillow, numpy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image
from acia.segm.open import open_sequence  # Pfad ggf. anpassen, falls Modul woanders liegt

from config import CHIPS_PATH, SEED
from train_lock import load_locked_chips, update_locked_chips

SPLITS = ("train", "val", "test")
MIN_FOLDERS_PER_SPLIT = {"val": 1, "test": 2}
REGISTRATION_FILENAME = "registration_transforms.json"

# Zwei ROI-Bedeutungen kommen im Datensatz vor und muessen als getrennte
# Klassen gelernt werden - sie werden automatisch anhand des Seitenverhaeltnisses
# der Box unterschieden (per ROI, nicht per Chip-Ordner, da beide Typen auch
# INNERHALB eines Ordners gemischt vorkommen koennen, z.B. Judith-Arg-20231127).
CLASS_NAMES = {0: "chamber_region", 1: "channel_region"}
CHANNEL_ASPECT_RATIO_THRESHOLD = 2.5  # Breite:Hoehe (oder umgekehrt) ab diesem Wert -> Kanal

# Fester Annotations-Frame fuer das gesamte Projekt: alle ROIs werden auf
# diesem Frame gezeichnet. Frame 0 = reference_frame der Registrierung,
# daher dort immer Identitaets-Transformation (kein Rate-Risiko mehr).
ANNOT_FRAME = 0


# --------------------------------------------------------------------------
# Bildverarbeitung
# --------------------------------------------------------------------------

def normalize_to_8bit(img: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    """Skaliert ein Bild per Perzentil-Clipping auf 8-bit.

    Robuster als Min-Max-Normalisierung gegenueber Ausreisser-Pixeln und
    besser geeignet fuer den engen Wertebereich von Phasenkontrast-
    Mikroskopiebildern.
    """
    lo, hi = np.percentile(img, [p_low, p_high])
    clipped = np.clip(img, lo, hi)
    return ((clipped - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)


def rotated_box_to_corners(
    cx: float, cy: float, w: float, h: float, angle_deg: float, img_w: int, img_h: int
) -> list[float]:
    """Wandelt center/size/angle in vier normalisierte Eckpunkte um (YOLO-OBB-Format)."""
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    dx, dy = w / 2, h / 2
    local_corners = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]

    corners: list[float] = []
    for x, y in local_corners:
        x_rot = x * cos_a - y * sin_a + cx
        y_rot = x * sin_a + y * cos_a + cy
        corners.append(min(max(x_rot / img_w, 0.0), 1.0))
        corners.append(min(max(y_rot / img_h, 0.0), 1.0))
    return corners


def classify_roi_by_shape(w: float, h: float) -> int:
    """Bestimmt automatisch die Klasse einer ROI anhand ihres Seitenverhaeltnisses.

    Kammer-Boxen sind ungefaehr quadratisch (Seitenverhaeltnis nahe 1).
    Kanal-Boxen sind stark laenglich (schmale, lange Streifen). Damit laesst
    sich die Klasse pro einzelner ROI bestimmen, ohne dass ein ganzer
    Chip-Ordner nur eine Klasse haben muesste - einzelne Positionen innerhalb
    eines Ordners koennen durchaus die jeweils andere Bedeutung haben.
    """
    aspect_ratio = max(w, h) / max(min(w, h), 1e-6)
    return 1 if aspect_ratio >= CHANNEL_ASPECT_RATIO_THRESHOLD else 0


def pick_random_frame(total_frames: int, seed_key: str) -> int:
    """Waehlt einen zufaelligen, aber ueber SEED + seed_key reproduzierbaren
    Frame-Index. Nur fuer explizite Anwendungsfaelle ohne Box-Bezug (z.B.
    Preview-Zwecke) - fuer Trainingsdaten NICHT verwenden, wenn dafuer keine
    Registrierung zur Korrektur der Box-Position vorliegt (siehe
    process_selection_file: ohne Registrierung wird bewusst nur Frame 0
    exportiert, um keine falsch positionierten Boxen zu erzeugen)."""
    digest = hashlib.sha256(f"{SEED}:{seed_key}".encode()).hexdigest()
    rng = random.Random(int(digest, 16))
    return rng.randrange(total_frames)


# --------------------------------------------------------------------------
# Registrierung: ROI zwischen Frames derselben Position uebertragen
# --------------------------------------------------------------------------

def _rotation_matrix(theta_rad: float) -> np.ndarray:
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return np.array([[c, -s], [s, c]])


def _invert_rigid(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Invertiert eine starre Transformation p' = R @ p + t."""
    R_inv = R.T  # Rotationsmatrizen sind orthogonal, Inverse = Transponierte
    return R_inv, -R_inv @ t


def transform_point(
    point: tuple[float, float],
    theta_from_rad: float, t_from: tuple[float, float],
    theta_to_rad: float, t_to: tuple[float, float],
) -> tuple[float, float]:
    """Ueberfuehrt einen Punkt von den Koordinaten eines Frames ('from') in
    die eines anderen Frames ('to'), beide relativ zum selben reference_frame.

    Verifiziert (siehe verify_registration.py): die gespeicherte Transformation
    (dx, dy, theta) eines Frames bildet Punkte vom reference_frame IN dieses
    Frame ab: p_frame = R(theta) @ p_ref + t.
    """
    R_from = _rotation_matrix(theta_from_rad)
    t_from_arr = np.array(t_from)
    R_from_inv, t_from_inv = _invert_rigid(R_from, t_from_arr)

    p = np.array(point)
    p_ref = R_from_inv @ p + t_from_inv  # zurueck in reference_frame-Koordinaten

    R_to = _rotation_matrix(theta_to_rad)
    p_to = R_to @ p_ref + np.array(t_to)
    return float(p_to[0]), float(p_to[1])


def propagate_roi_across_frames(
    roi: dict, annot_frame: int, target_frames: list[int], transforms: dict[int, dict],
) -> dict[int, dict]:
    """Berechnet fuer eine auf annot_frame gezeichnete ROI die entsprechenden
    center/angle-Werte fuer jeden Frame in target_frames.

    transforms: dict[frame_idx] -> {"dx": float, "dy": float, "theta": float (rad)}
    Gibt dict[frame_idx] -> roi-artiges dict (center, size, angle) zurueck.
    Groesse (size) bleibt unveraendert, da die Registrierung starr ist
    (keine Skalierung).
    """
    if annot_frame not in transforms:
        raise ValueError(
            f"Kein Registrierungs-Transform fuer den Annotations-Frame {annot_frame} "
            f"vorhanden - Propagation nicht moeglich fuer diese Position."
        )

    t_annot = transforms[annot_frame]
    cx, cy = roi["center"]
    angle_annot_deg = roi["angle"]

    result: dict[int, dict] = {}
    for frame_idx in target_frames:
        if frame_idx not in transforms:
            continue  # Frame ohne verlaessliche Registrierung wird ausgelassen

        t_target = transforms[frame_idx]
        new_cx, new_cy = transform_point(
            (cx, cy),
            t_annot["theta"], (t_annot["dx"], t_annot["dy"]),
            t_target["theta"], (t_target["dx"], t_target["dy"]),
        )
        # Rotationswinkel addieren sich in 2D exakt, keine Naeherung noetig.
        angle_delta_deg = math.degrees(t_target["theta"] - t_annot["theta"])

        result[frame_idx] = {
            "center": [new_cx, new_cy],
            "size": roi["size"],
            "angle": angle_annot_deg + angle_delta_deg,
        }

    return result


def load_registration(registration_path: Path) -> dict[int, dict]:
    """Laedt registration_transforms.json und liefert dict[position] -> {
        'transforms': dict[frame_idx] -> {'dx', 'dy', 'theta'},
        'reference_frame': int,
    }. Nur Frames MIT Eintrag in 'transforms' gelten als verlaesslich
    registriert (failed_frames sind hier bereits ausgeschlossen)."""
    with open(registration_path, "r", encoding="utf-8") as f:
        data = json.load(f)

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
    valid_frames: set[int], annot_frame: int, count: int | None, seed_key: str = ""
) -> list[int]:
    """Waehlt Ziel-Frames aus den registrierten (validen) Frames einer Position.

    count=None (bzw. 'all'): alle validen Frames.
    count=int: 'count' zufaellig (aber ueber SEED + seed_key reproduzierbar)
    gewaehlte Frames, annot_frame ist immer mit dabei. Fuer jeden gewaehlten
    Frame wird die Box anschliessend ueber propagate_roi_across_frames korrekt
    aus der Registrierung berechnet - die zufaellige Auswahl betrifft nur,
    WELCHE Frames als Bilder exportiert werden, nicht die Korrektheit der Box.
    """
    frames = sorted(valid_frames)
    if count is None or count >= len(frames):
        return frames

    count = max(1, count)
    remaining = [f for f in frames if f != annot_frame]
    digest = hashlib.sha256(f"{SEED}:{seed_key}".encode()).hexdigest()
    random.Random(int(digest, 16)).shuffle(remaining)

    chosen = set(remaining[:max(0, count - 1)])
    chosen.add(annot_frame)
    return sorted(chosen)


# --------------------------------------------------------------------------
# Selection-Dateien finden und verarbeiten
# --------------------------------------------------------------------------

def find_selection_jsons(chips_path: Path) -> Iterator[tuple[Path, Path]]:
    """Liefert (Chip-Ordner, selection.json-Pfad) fuer jeden Unterordner mit einer selection.json."""
    for folder in sorted(chips_path.iterdir()):
        if not folder.is_dir():
            continue
        selection_path = folder / "selection.json"
        if selection_path.exists():
            yield folder, selection_path


def process_selection_file(
    folder: Path,
    selection_path: Path,
    images_dir: Path,
    labels_dir: Path,
    use_registration: bool,
    frames_per_position: int | None,
) -> int:
    """Verarbeitet eine selection.json: extrahiert Frames und schreibt Bild/Label-Paare.

    Die Klasse jeder ROI wird automatisch anhand ihres Seitenverhaeltnisses
    bestimmt (siehe classify_roi_by_shape) - nicht mehr ordnerweit vorgegeben,
    da Kammer- und Kanal-ROIs auch innerhalb desselben Chip-Ordners gemischt
    vorkommen koennen.

    Returns:
        Anzahl der geschriebenen Bild/Label-Paare.
    """
    with open(selection_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    source_path = data.get("source", {}).get("path")
    if source_path is None:
        print(f"Warnung: {selection_path} hat kein source.path Feld, uebersprungen.", flush=True)
        return 0

    img_w = data["source"]["sizes"]["X"]
    img_h = data["source"]["sizes"]["Y"]
    total_frames = data["source"]["sizes"]["T"]

    rois_by_position: dict[int, list[dict]] = {}
    for sel in data.get("selections", []):
        rois_by_position.setdefault(sel["position"], []).append(sel["roi"])

    if not rois_by_position:
        print(f"Keine Annotationen in {selection_path}, uebersprungen.", flush=True)
        return 0

    registration_by_position: dict[int, dict] = {}
    if use_registration:
        registration_path = folder / REGISTRATION_FILENAME
        if registration_path.exists():
            registration_by_position = load_registration(registration_path)
        else:
            print(f"Warnung: --use-registration gesetzt, aber {registration_path} "
                  f"nicht gefunden - {folder.name} nutzt nur Frame 0 (kein "
                  f"unkorrigierter Ersatz-Frame, um keine falschen Labels zu erzeugen).", flush=True)

    print(f"[{folder.name}] Oeffne Sequenz: {source_path} "
          f"(Annotations-Frame {ANNOT_FRAME}/{total_frames - 1})", flush=True)

    written = 0
    with open_sequence(source_path) as seq:
        for pos, rois in sorted(rois_by_position.items()):
            pos_source = seq.position(pos)
            reg = registration_by_position.get(pos)

            if reg is not None:
                # Registrierung vorhanden: zufaellige Ziel-Frames waehlen, Box fuer
                # jeden davon korrekt ueber die Registrierung berechnen (siehe unten).
                try:
                    target_frames = select_target_frames(
                        set(reg["transforms"]), ANNOT_FRAME, frames_per_position,
                        seed_key=f"{folder.name}_pos{pos}",
                    )
                except Exception:
                    target_frames = [ANNOT_FRAME]
            else:
                # Keine Registrierung fuer diese Position: NUR der tatsaechlich
                # annotierte Frame 0 wird exportiert. Ein anderer Frame mit der
                # unveraenderten Frame-0-Box waere schlicht falsch beschriftet -
                # ohne Korrekturmoeglichkeit lieber ein korrektes Beispiel als
                # ein zusaetzliches falsches.
                target_frames = [ANNOT_FRAME]

            for frame_idx in target_frames:
                if reg is not None and frame_idx != ANNOT_FRAME:
                    try:
                        frame_rois = [
                            propagate_roi_across_frames(
                                roi, ANNOT_FRAME, [frame_idx], reg["transforms"]
                            )[frame_idx]
                            for roi in rois
                        ]
                    except ValueError as exc:
                        print(f"  Position {pos}: {exc} - ueberspringe Propagation.", flush=True)
                        continue
                else:
                    frame_rois = rois

                # get_frame() liest lazy genau diesen einen Frame von der Platte -
                # kein Multi-GB-Ladeproblem, kein manuelles Achsen-Indexieren noetig.
                raw = pos_source.get_frame(frame_idx).raw  # (H, W, C) oder (H, W)
                frame = raw[..., 0] if raw.ndim == 3 else raw
                img_8bit = normalize_to_8bit(frame)

                name = f"{folder.name}_pos{pos}_frame{frame_idx:03d}"
                Image.fromarray(img_8bit).save(images_dir / f"{name}.png")

                lines = []
                for roi in frame_rois:
                    cx, cy = roi["center"]
                    w, h = roi["size"]
                    roi_class_id = classify_roi_by_shape(w, h)
                    corners = rotated_box_to_corners(cx, cy, w, h, roi["angle"], img_w, img_h)
                    lines.append(f"{roi_class_id} " + " ".join(f"{c:.6f}" for c in corners))
                (labels_dir / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

                written += 1

            mode = "registriert" if reg is not None else "nur Frame 0 (keine Registrierung)"
            print(f"  Position {pos}: {len(rois)} ROI(s) x {len(target_frames)} Frame(s) ({mode})", flush=True)

    return written


# --------------------------------------------------------------------------
# Train/Val/Test-Split
# --------------------------------------------------------------------------

def assign_splits(
    folder_names: list[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    forced_val: set[str] | None = None,
    forced_test: set[str] | None = None,
    forced_train: set[str] | None = None,
) -> dict[str, str]:
    """Weist jeden Chip-Ordner reproduzierbar (per seed) einem Split zu.

    forced_val/forced_test/forced_train legen bestimmte Ordner manuell fest;
    der Rest wird zufaellig auf die verbleibenden Prozentsaetze verteilt.
    Splits erfolgen auf ganzen Chip-Ordnern (Sessions), nicht auf
    Einzelbildern, um Data Leakage zwischen aehnlichen Aufnahmen derselben
    Session zu vermeiden.
    """
    forced_val = forced_val or set()
    forced_test = forced_test or set()
    forced_train = forced_train or set()

    overlaps = [
        ("--val-chips und --test-chips", forced_val & forced_test),
        ("--val-chips und --train-chips", forced_val & forced_train),
        ("--test-chips und --train-chips", forced_test & forced_train),
    ]
    for label, overlap in overlaps:
        if overlap:
            raise ValueError(f"Ordner duerfen nicht gleichzeitig in {label} stehen: {overlap}")

    remaining = sorted(
        n for n in folder_names
        if n not in forced_val and n not in forced_test and n not in forced_train
    )
    random.Random(seed).shuffle(remaining)

    n_total = len(folder_names)
    n_val = max(0, round(n_total * val_ratio) - len(forced_val))
    n_test = max(0, round(n_total * test_ratio) - len(forced_test))

    val_random = set(remaining[:n_val])
    test_random = set(remaining[n_val:n_val + n_test])

    assignment = {}
    for name in folder_names:
        if name in forced_train:
            assignment[name] = "train"
        elif name in forced_val or name in val_random:
            assignment[name] = "val"
        elif name in forced_test or name in test_random:
            assignment[name] = "test"
        else:
            assignment[name] = "train"

    for split, minimum in MIN_FOLDERS_PER_SPLIT.items():
        count = sum(1 for s in assignment.values() if s == split)
        if count < minimum:
            print(f"Warnung: nur {count} Chip-Ordner im {split}-Split "
                  f"(empfohlenes Minimum: {minimum}) - Ergebnis ggf. wenig robust.", flush=True)

    return assignment


def clean_split_dirs(dataset_root: Path) -> None:
    """Leert images/{train,val,test} und labels/{train,val,test} und legt sie neu an.

    Verhindert, dass Dateien aus frueheren Laeufen (z.B. mit anderem Split
    oder anderer Frame-Wahl) im Datensatz zurueckbleiben. Gibt Fortschritt
    pro Ordner aus, da das Loeschen bei vielen Dateien (z.B. Netzwerk-Mounts)
    spuerbar dauern kann - ohne das sieht ein langsamer Lauf wie ein Haenger aus.
    """
    for kind in ("images", "labels"):
        for split in SPLITS:
            split_dir = dataset_root / kind / split
            if split_dir.exists():
                existing = sum(1 for _ in split_dir.iterdir())
                if existing:
                    print(f"Loesche {existing} bestehende Datei(en) in {split_dir} ...", flush=True)
                    start = time.monotonic()
                    shutil.rmtree(split_dir)
                    print(f"  fertig nach {time.monotonic() - start:.1f}s", flush=True)
            split_dir.mkdir(parents=True, exist_ok=True)
    print(f"Bestehende Inhalte unter '{dataset_root}/images' und '{dataset_root}/labels' "
          f"geleert - starte mit sauberen Ordnern.\n", flush=True)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chip-Ordner (selection.json + ND2) -> trainingsfertiger YOLO-OBB-Datensatz"
    )
    parser.add_argument("--dataset-root", default="dataset",
                         help="Basisordner fuer images/<split> und labels/<split>")
    parser.add_argument("--chips", default="",
                         help="Optional: kommagetrennte Chip-Ordnernamen, auf die die "
                              "Verarbeitung beschraenkt wird (Standard: alle Ordner unter "
                              "CHIPS_PATH). Nuetzlich zum gezielten Testen einzelner Ordner.")
    parser.add_argument("--val-ratio", type=float, default=0.175,
                         help="Anteil der Chip-Ordner fuer Validation (Standard: 0.175)")
    parser.add_argument("--test-ratio", type=float, default=0.175,
                         help="Anteil der Chip-Ordner fuer Test (Standard: 0.175)")
    parser.add_argument("--val-chips", default="",
                         help="Kommagetrennte Chip-Ordner, die fest Validation zugewiesen werden")
    parser.add_argument("--test-chips", default="",
                         help="Kommagetrennte Chip-Ordner, die fest Test zugewiesen werden")
    parser.add_argument("--train-chips", default="",
                         help="Kommagetrennte Chip-Ordner, die fest Training zugewiesen werden "
                              "(garantiert, landen nie in Val/Test, unabhaengig vom Zufalls-Split)")
    parser.add_argument("--split", choices=SPLITS, default=None,
                         help="Nur zusammen mit --chips: erzwingt genau diesen Split fuer die "
                              "angegebenen Ordner, OHNE den restlichen Datensatz zu loeschen oder "
                              "neu zu verteilen. Fuer gezieltes Nachgenerieren einzelner Ordner "
                              "(z.B. nach einem Fehler beim ND2-Oeffnen).")
    parser.add_argument("--channel-aspect-threshold", type=float, default=CHANNEL_ASPECT_RATIO_THRESHOLD,
                         help=f"Seitenverhaeltnis, ab dem eine ROI als 'channel_region' statt "
                              f"'chamber_region' klassifiziert wird (Standard: {CHANNEL_ASPECT_RATIO_THRESHOLD})")
    parser.add_argument("--use-registration", action="store_true",
                         help="ROI mithilfe von registration_transforms.json auf mehrere Frames "
                              "derselben Position uebertragen (Stage-Drift-Korrektur). Ohne dieses "
                              "Flag wird pro Position ein zufaelliger, unkorrigierter Frame genutzt.")
    parser.add_argument("--frames-per-position", default="1",
                         help="Nur mit --use-registration relevant: Anzahl Ziel-Frames pro "
                              "Position ('all' fuer alle registrierten Frames, Standard: 1 = "
                              "nur der Annotations-Frame, keine Propagation)")
    parser.add_argument("--all-train", action="store_true",
                         help="FINALER Modus: ALLE gefundenen Ordner werden Train zugewiesen, "
                              "kein echtes Val/Test mehr. Fuer das finale Modell, NACHDEM die "
                              "Generalisierung bereits ueber ein separates Val/Test bestaetigt "
                              "wurde. Ein kleiner Anteil wird trotzdem als 'Monitoring-Val' "
                              "abgezweigt (siehe --monitor-val-fraction) - das ist NUR fuer "
                              "Ultralytics' Early-Stopping-Mechanik, KEINE echte "
                              "Generalisierungsmessung (stammt aus denselben Sessions wie Train).")
    parser.add_argument("--monitor-val-fraction", type=float, default=0.1,
                         help="Nur mit --all-train: Anteil der Bilder, der als reines "
                              "Trainings-Monitoring nach Val verschoben wird (Standard: 0.1)")
    return parser.parse_args()


def carve_out_monitor_val(dataset_root: Path, fraction: float, seed: int) -> None:
    """Verschiebt einen zufaelligen (reproduzierbaren) Anteil der Train-Bilder
    nach Val - NUR als technische Notwendigkeit fuer Ultralytics' Early-Stopping,
    KEINE echte Generalisierungsmessung. Stammt aus denselben Chip-Ordnern/
    Sessions wie Train, ist also nicht unabhaengig."""
    train_images = dataset_root / "images" / "train"
    train_labels = dataset_root / "labels" / "train"
    val_images = dataset_root / "images" / "val"
    val_labels = dataset_root / "labels" / "val"
    val_images.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)

    all_images = sorted(train_images.glob("*"))
    n_val = max(1, round(len(all_images) * fraction))
    rng = random.Random(seed)
    chosen = rng.sample(all_images, k=min(n_val, len(all_images)))

    for img_path in chosen:
        label_path = train_labels / f"{img_path.stem}.txt"
        img_path.rename(val_images / img_path.name)
        if label_path.exists():
            label_path.rename(val_labels / label_path.name)

    print(f"\nACHTUNG: {len(chosen)} Bilder als Monitoring-Val nach '{val_images}' verschoben.\n"
          f"Das ist KEINE echte Generalisierungsmessung (gleiche Sessions wie Train) - "
          f"dient nur der Early-Stopping-Mechanik von Ultralytics. Fuer die Projekt-"
          f"dokumentation gilt weiterhin die zuletzt ermittelte Test-Metrik als "
          f"Generalisierungsnachweis, nicht diese Val-Zahlen.", flush=True)


def main() -> None:
    global CHANNEL_ASPECT_RATIO_THRESHOLD
    args = parse_args()
    CHANNEL_ASPECT_RATIO_THRESHOLD = args.channel_aspect_threshold

    chips_path = Path(CHIPS_PATH)
    dataset_root = Path(args.dataset_root)
    forced_val = {name.strip() for name in args.val_chips.split(",") if name.strip()}
    forced_test = {name.strip() for name in args.test_chips.split(",") if name.strip()}
    forced_train = {name.strip() for name in args.train_chips.split(",") if name.strip()}

    # Ordner, die JEMALS schon in Train waren, bleiben fuer immer in Train -
    # sonst koennte ein spaeteres Weitertrainieren von einem bestehenden
    # Checkpoint aus (der diese Ordner schon "kennt") gegen einen Val/Test-Split
    # auswerten, der eigentlich nicht mehr unabhaengig ist.
    locked_chips = load_locked_chips()
    if locked_chips:
        print(f"Aus train_locked_chips.txt dauerhaft auf Train gesperrt: "
              f"{', '.join(sorted(locked_chips))}", flush=True)
    forced_train |= locked_chips

    frames_per_position = None if args.frames_per_position == "all" else int(args.frames_per_position)

    if args.split and not args.chips.strip():
        raise SystemExit("--split kann nur zusammen mit --chips verwendet werden - "
                          "sonst waere unklar, welche Ordner betroffen sein sollen.")

    found = list(find_selection_jsons(chips_path))

    if args.chips.strip():
        requested = {name.strip() for name in args.chips.split(",") if name.strip()}
        available = {folder.name for folder, _ in found}
        missing = requested - available
        if missing:
            print(f"Warnung: folgende --chips Ordner wurden unter {chips_path} nicht "
                  f"gefunden und werden ignoriert: {', '.join(sorted(missing))}", flush=True)
        found = [(folder, path) for folder, path in found if folder.name in requested]
        if not found:
            raise SystemExit(f"Keiner der angegebenen --chips Ordner wurde gefunden. "
                              f"Verfuegbar unter {chips_path}: {', '.join(sorted(available))}")
        print(f"Beschraenkt auf {len(found)} Chip-Ordner: {', '.join(f.name for f, _ in found)}\n", flush=True)

    folder_names = [folder.name for folder, _ in found]

    if args.all_train:
        # Finaler Modus: ALLE Ordner nach Train, kein echtes Val/Test mehr.
        assignment = {name: "train" for name in folder_names}
        print(f"FINALER MODUS (--all-train): alle {len(folder_names)} Ordner werden "
              f"Train zugewiesen: {', '.join(sorted(folder_names))}\n", flush=True)
        update_locked_chips(set(folder_names))
        clean_split_dirs(dataset_root)
    elif args.split:
        # Gezielter Modus: alle ausgewaehlten Ordner bekommen denselben, fest
        # vorgegebenen Split. Kein assign_splits, kein clean_split_dirs - der
        # bestehende Datensatz bleibt komplett unangetastet, es werden nur
        # die angegebenen Ordner ergaenzt/aktualisiert.
        assignment = {name: args.split for name in folder_names}
        print(f"Gezielter Modus: {len(folder_names)} Ordner werden alle dem Split "
              f"'{args.split}' zugewiesen. Bestehender Datensatz bleibt unangetastet, "
              f"es wird nur ergaenzt/ueberschrieben.\n", flush=True)
        if args.split == "train":
            update_locked_chips(set(folder_names))
    else:
        assignment = assign_splits(folder_names, args.val_ratio, args.test_ratio, SEED,
                                    forced_val, forced_test, forced_train)

        print(f"Split-Zuordnung (seed={SEED}):", flush=True)
        for split in SPLITS:
            names = [n for n, s in assignment.items() if s == split]
            print(f"  {split}: {', '.join(names) if names else '(keine)'}", flush=True)
        print(flush=True)

        update_locked_chips({n for n, s in assignment.items() if s == "train"})

        clean_split_dirs(dataset_root)


    counts = {split: 0 for split in SPLITS}
    folder_counts = {split: 0 for split in SPLITS}
    failed_folders: list[str] = []

    for folder, selection_path in found:
        split = assignment[folder.name]
        images_dir = dataset_root / "images" / split
        labels_dir = dataset_root / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        print(f"--- {folder.name} -> {split} ---", flush=True)
        try:
            written = process_selection_file(
                folder, selection_path, images_dir, labels_dir,
                args.use_registration, frames_per_position,
            )
        except Exception as exc:
            print(f"FEHLER bei Chip-Ordner '{folder.name}': {exc}", flush=True)
            print(f"  -> Ordner wird uebersprungen, restlicher Datensatz bleibt unberuehrt. "
                  f"Nach Behebung des Fehlers separat nachziehen mit:\n"
                  f"     --chips \"{folder.name}\" --split {split}"
                  + (" --use-registration" if args.use_registration else "")
                  + (f" --frames-per-position {args.frames_per_position}" if args.use_registration else ""),
                  flush=True)
            failed_folders.append(folder.name)
            continue

        counts[split] += written
        folder_counts[split] += 1

    print("\nFertig:", flush=True)
    for split in SPLITS:
        print(f"  {split}: {counts[split]} Bild/Label-Paar(e) aus {folder_counts[split]} Chip-Ordner(n)", flush=True)
    if failed_folders:
        print(f"\nFehlgeschlagene Ordner ({len(failed_folders)}): {', '.join(failed_folders)}", flush=True)

    if args.all_train:
        carve_out_monitor_val(dataset_root, args.monitor_val_fraction, SEED)


if __name__ == "__main__":
    main()