"""
Konvertiert acia.selection/v1 JSON-Annotationen (center, size, angle) in
YOLO-OBB-Label-Format (class x1 y1 x2 y2 x3 y3 x4 y4, normalisiert).

Erzeugt eine Label-Datei pro Position, benannt nach dem gewaehlten Naming-Schema,
damit sie zum zugehoerigen Bild (z.B. letzter Frame dieser Position) passt.

Verwendung:
    python convert_selection_to_yolo_obb.py selection.json -o labels/train --prefix EH01

Erzeugt z.B.:
    labels/train/EH01_pos0.txt
    labels/train/EH01_pos1.txt
"""

import argparse
import json
import math
from pathlib import Path


def rotated_box_to_corners(cx, cy, w, h, angle_deg, img_w, img_h):
    """Wandelt center+size+angle in vier normalisierte Eckpunkte um (YOLO-OBB-Format)."""
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    dx, dy = w / 2, h / 2
    corners = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]

    yolo_corners = []
    for x, y in corners:
        x_rot = x * cos_a - y * sin_a + cx
        y_rot = x * sin_a + y * cos_a + cy
        # normalisieren und in [0, 1] clampen (kleine Rundungsfehler an den Raendern)
        x_norm = min(max(x_rot / img_w, 0.0), 1.0)
        y_norm = min(max(y_rot / img_h, 0.0), 1.0)
        yolo_corners.extend([x_norm, y_norm])

    return yolo_corners


def convert(json_path, output_dir, prefix, class_id=0):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = data["source"]["sizes"]["X"]
    img_h = data["source"]["sizes"]["Y"]

    labels_by_position = {}
    for sel in data["selections"]:
        pos = sel["position"]
        cx, cy = sel["roi"]["center"]
        w, h = sel["roi"]["size"]
        angle = sel["roi"]["angle"]

        if angle != 0.0:
            print(f"Hinweis: ROI '{sel['label']}' (Position {pos}) ist rotiert "
                  f"({angle}Grad) - wird korrekt als OBB umgerechnet.")

        corners = rotated_box_to_corners(cx, cy, w, h, angle, img_w, img_h)
        line = f"{class_id} " + " ".join(f"{c:.6f}" for c in corners)
        labels_by_position.setdefault(pos, []).append(line)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written_files = []
    for pos, lines in sorted(labels_by_position.items()):
        out_path = output_dir / f"{prefix}_pos{pos}.txt"
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written_files.append(out_path)
        print(f"Geschrieben: {out_path}  ({len(lines)} ROI(s))")

    return written_files


def main():
    parser = argparse.ArgumentParser(description="acia.selection/v1 JSON -> YOLO-OBB Labels")
    parser.add_argument("json_path", help="Pfad zur selection.json Datei")
    parser.add_argument("-o", "--output-dir", default="labels", help="Zielordner fuer .txt Label-Dateien")
    parser.add_argument("--prefix", default=None,
                         help="Dateiname-Praefix (Standard: Dateiname der JSON ohne Endung)")
    parser.add_argument("--class-id", type=int, default=0, help="Klassen-ID (Standard: 0, eine Klasse)")
    args = parser.parse_args()

    prefix = args.prefix or Path(args.json_path).stem
    convert(args.json_path, args.output_dir, prefix, args.class_id)


if __name__ == "__main__":
    main()