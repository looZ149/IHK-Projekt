"""Fasst beschreibende Statistiken aus reinen Inferenz-Ergebnissen zusammen -
KEINE Genauigkeits-Metriken (mAP/Precision/Recall), da dafuer Ground-Truth-
Labels noetig waeren, die bei frischen, unannotierten Bildern nicht existieren.

Nuetzlich als schnelle Plausibilitaetspruefung: erkennt das Modell ueberhaupt
was, wie sicher ist es sich, welche Klassen kommen vor - und ob eine Klasse
ODER ein bestimmter Chip auf neuen Daten auffaellig selten/unsicher erkannt
wird. Die Chip-Art wird aus dem Dateinamen abgeleitet
('<ChipName>_posX_frameYYY.tiff', wie von
export_random_frames_for_inference.py erzeugt).
"""

from collections import Counter, defaultdict
from pathlib import Path

from train_lock import CHIP_IMAGE_NAME_RE


def summarize_predictions(results, class_names: dict[int, str]) -> None:
    """Gibt eine Zusammenfassung ueber eine Liste von Ultralytics-Results aus."""
    total_images = len(results)
    images_with_detection = 0
    detections_per_image = []
    class_counts = Counter()
    class_confidences: dict[int, list[float]] = defaultdict(list)

    # Pro Chip-Art (aus dem Dateinamen abgeleitet)
    chip_total = Counter()
    chip_with_detection = Counter()
    chip_class_counts: dict[str, Counter] = defaultdict(Counter)

    for r in results:
        n_boxes = len(r.obb) if r.obb is not None else 0
        detections_per_image.append(n_boxes)

        image_path = getattr(r, "path", None)
        chip_name = "<unbekannt>"
        if image_path:
            match = CHIP_IMAGE_NAME_RE.match(Path(image_path).name)
            chip_name = match.group(1) if match else Path(image_path).stem
        chip_total[chip_name] += 1

        if n_boxes > 0:
            images_with_detection += 1
            chip_with_detection[chip_name] += 1
            for cls_id, conf in zip(r.obb.cls.tolist(), r.obb.conf.tolist()):
                class_counts[int(cls_id)] += 1
                class_confidences[int(cls_id)].append(conf)
                chip_class_counts[chip_name][int(cls_id)] += 1

    print(f"Bilder insgesamt: {total_images}")
    print(f"Bilder MIT mind. 1 Erkennung: {images_with_detection} "
          f"({images_with_detection / total_images * 100:.1f}%)")
    print(f"Bilder OHNE Erkennung: {total_images - images_with_detection}")

    if detections_per_image:
        print(f"\nErkennungen pro Bild: min={min(detections_per_image)}  "
              f"avg={sum(detections_per_image) / len(detections_per_image):.2f}  "
              f"max={max(detections_per_image)}")

    print("\nErkennungen pro Klasse (mit durchschnittlicher Confidence):")
    for cls_id in sorted(class_counts.keys()):
        name = class_names.get(cls_id, f"Klasse {cls_id}")
        confs = class_confidences[cls_id]
        avg_conf = sum(confs) / len(confs)
        print(f"  {name}: {class_counts[cls_id]} Erkennungen, "
              f"Confidence min={min(confs):.2f} avg={avg_conf:.2f} max={max(confs):.2f}")

    missing_classes = set(class_names.keys()) - set(class_counts.keys())
    if missing_classes:
        missing_names = [class_names[c] for c in missing_classes]
        print(f"\nHinweis: folgende Klasse(n) wurden auf KEINEM Bild erkannt: "
              f"{', '.join(missing_names)}")

    print("\nErgebnis pro Chip-Art:")
    header = f"  {'Chip-Art':<40} {'Bilder':>7} {'Erkannt':>8} {'Rate':>7}  Klassen"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for chip_name in sorted(chip_total.keys()):
        total = chip_total[chip_name]
        detected = chip_with_detection[chip_name]
        rate = detected / total * 100
        class_breakdown = ", ".join(
            f"{class_names.get(cid, f'Klasse {cid}')}: {count}"
            for cid, count in sorted(chip_class_counts[chip_name].items())
        )
        print(f"  {chip_name:<40} {total:>7} {detected:>8} {rate:>6.1f}%  {class_breakdown}")


# Beispiel:
if __name__ == "__main__":
    from ultralytics import YOLO

    model = YOLO("runs/final/final_model/weights/best.pt")
    results = model("inference_check", stream=False, conf=0.25)

    summarize_predictions(results, class_names={0: "chamber_region", 1: "channel_region"})