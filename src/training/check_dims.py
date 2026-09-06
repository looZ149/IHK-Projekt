"""Prueft alle Bilder in einem Ordner auf ungueltige Groesse (Breite/Hoehe 0
oder nicht lesbar) sowie auf EXTREME Seitenverhaeltnisse - genau die Art
Datei, die cv2.resize mit 'inv_scale_x > 0'-Fehlern zum Absturz bringt (beim
internen Letterboxing kann die kuerzere Seite eines sehr laenglichen Bildes
nach dem Skalieren auf 0 gerundet werden, auch wenn das Original selbst
nicht 0 gross war).

Verwendung:
    python check_image_shapes.py --source-dir inference_check
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# Ab diesem Seitenverhaeltnis wird ein Bild als verdaechtig markiert - bei
# Standard imgsz=640 kann die kurze Seite bei sehr extremen Verhaeltnissen
# beim internen Letterboxing auf 0 gerundet werden.
ASPECT_RATIO_WARNING = 50.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--aspect-warning", type=float, default=ASPECT_RATIO_WARNING)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    problems = []
    all_shapes = []
    checked = 0

    for img_path in sorted(source_dir.glob("*")):
        if img_path.suffix.lower() not in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
            continue
        checked += 1
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                arr = np.array(img)
        except Exception as exc:
            problems.append((img_path.name, f"nicht lesbar: {exc}"))
            continue

        if w <= 0 or h <= 0:
            problems.append((img_path.name, f"ungueltige Groesse: {w}x{h}"))
            continue
        if arr.size == 0:
            problems.append((img_path.name, "leeres Array"))
            continue

        ratio = max(w, h) / min(w, h)
        all_shapes.append((img_path.name, w, h, ratio))
        if ratio >= args.aspect_warning:
            problems.append((img_path.name, f"extremes Seitenverhaeltnis: {w}x{h} (ratio={ratio:.1f})"))

    print(f"{checked} Bild(er) geprueft.")

    if all_shapes:
        ratios = [r for _, _, _, r in all_shapes]
        widths = [w for _, w, _, _ in all_shapes]
        heights = [h for _, _, h, _ in all_shapes]
        print(f"\nBreite: min={min(widths)} max={max(widths)}")
        print(f"Hoehe:  min={min(heights)} max={max(heights)}")
        print(f"Seitenverhaeltnis: min={min(ratios):.1f} max={max(ratios):.1f}")

        top5 = sorted(all_shapes, key=lambda x: -x[3])[:5]
        print(f"\nDie 5 extremsten Seitenverhaeltnisse:")
        for name, w, h, ratio in top5:
            print(f"  {name}: {w}x{h} (ratio={ratio:.1f})")

    if problems:
        print(f"\n{len(problems)} problematische Datei(en) gefunden:")
        for name, reason in problems:
            print(f"  {name}: {reason}")
        print("\nDiese Dateien vor der Inferenz aus dem Ordner entfernen/ersetzen.")
    else:
        print("\nKeine Probleme gefunden - die Ursache liegt vermutlich woanders "
              "(z.B. eine Nicht-Bild-Datei im Quellordner, oder ein Ladeproblem bei "
              "einem bestimmten TIFF-Codec).")


if __name__ == "__main__":
    main()