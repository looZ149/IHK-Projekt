"""Visuelle Verifikation der Registrierungs-Korrektur.

Zeichnet die (ueber die Registrierung propagierte) ROI-Box auf zwei weit
auseinanderliegenden Frames derselben Position und speichert beide als PNG.
Manuell vergleichen: sitzt die Box auf beiden Bildern noch korrekt auf der
Kolonie/dem Objekt, oder ist sie verschoben?

Falls die Box in die falsche Richtung driftet, in build_trainingsdata.py
in transform_point() die Rollen von R_from/R_to bzw. die Vorzeichen von t
vertauschen (siehe Kommentar dort) und diesen Test wiederholen.

Verwendung:
    python verify_registration.py --chip-folder /pfad/zu/AM002 --position 0 \
        --annot-frame last --compare-frame 0

    (--compare-frame moeglichst weit vom Annotations-Frame entfernt waehlen,
    idealerweise eine Position mit grosser bekannter Drift - je groesser der
    Unterschied, desto eindeutiger ist ein Richtungsfehler zu erkennen.)
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
from acia.segm.open import open_sequence  # Pfad ggf. anpassen

from chamber_obb.dataset import rotated_box_to_corners
from chamber_obb.preprocessing import normalize_to_8bit

from build_trainingsdata import (
    REGISTRATION_FILENAME,
    load_registration,
    propagate_roi_across_frames,
)


def resolve_frame_index(spec: str, total_frames: int) -> int:
    """'last'/'mid'/int in einen konkreten Frame-Index aufloesen."""
    if spec == "last":
        return total_frames - 1
    if spec == "mid":
        return total_frames // 2
    return int(spec)


def draw_box(img_8bit, roi: dict, img_w: int, img_h: int, color: str = "red") -> Image.Image:
    img = Image.fromarray(img_8bit).convert("RGB")
    draw = ImageDraw.Draw(img)
    cx, cy = roi["center"]
    w, h = roi["size"]
    corners_norm = rotated_box_to_corners(cx, cy, w, h, roi["angle"], img_w, img_h)
    points = [(corners_norm[i] * img_w, corners_norm[i + 1] * img_h) for i in range(0, 8, 2)]
    draw.polygon(points, outline=color, width=4)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Visuelle Verifikation der Registrierungs-Korrektur")
    parser.add_argument("--chip-folder", required=True, help="Pfad zum Chip-Ordner (enthaelt selection.json)")
    parser.add_argument("--position", type=int, required=True)
    parser.add_argument("--annot-frame", default="last", help="'last', 'mid' oder fester Index")
    parser.add_argument("--compare-frame", type=int, default=0,
                         help="Frame, mit dem verglichen wird - moeglichst weit vom Annotations-Frame entfernt")
    parser.add_argument("--output-dir", default="verify_output")
    args = parser.parse_args()

    folder = Path(args.chip_folder)
    selection_path = folder / "selection.json"
    registration_path = folder / REGISTRATION_FILENAME

    with open(selection_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    source_path = data["source"]["path"]
    img_w = data["source"]["sizes"]["X"]
    img_h = data["source"]["sizes"]["Y"]
    total_frames = data["source"]["sizes"]["T"]

    rois = [sel["roi"] for sel in data["selections"] if sel["position"] == args.position]
    if not rois:
        raise SystemExit(f"Keine ROI fuer Position {args.position} in {selection_path} gefunden.")
    roi = rois[0]

    annot_frame = resolve_frame_index(args.annot_frame, total_frames)
    registration_by_position = load_registration(registration_path)
    reg = registration_by_position.get(args.position)
    if reg is None:
        raise SystemExit(f"Keine Registrierungsdaten fuer Position {args.position} gefunden.")

    propagated = propagate_roi_across_frames(
        roi, annot_frame, [annot_frame, args.compare_frame], reg["transforms"]
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open_sequence(source_path) as seq:
        pos_source = seq.position(args.position)
        for frame_idx, label in [(annot_frame, "annot"), (args.compare_frame, "compare")]:
            if frame_idx not in propagated:
                print(f"Frame {frame_idx} nicht registriert, uebersprungen.")
                continue

            raw = pos_source.get_frame(frame_idx).raw
            frame = raw[..., 0] if raw.ndim == 3 else raw
            img_8bit = normalize_to_8bit(frame)

            img = draw_box(img_8bit, propagated[frame_idx], img_w, img_h)
            out_path = output_dir / f"{folder.name}_pos{args.position}_frame{frame_idx:03d}_{label}.png"
            img.save(out_path)
            print(f"Gespeichert: {out_path}")

    print("\nManuell pruefen: sitzt die rote Box auf beiden Bildern auf der Kolonie?")
    print("Falls nicht (Box driftet weg statt mit dem Objekt mitzuwandern):")
    print("  -> in build_trainingsdata.py, Funktion transform_point(),")
    print("     R_from/R_to bzw. Vorzeichen von t vertauschen, dann erneut testen.")


if __name__ == "__main__":
    main()