"""Fuehrt Inferenz auf 'inference_check' aus und liefert BEIDES in einem Lauf:
1. Annotierte Bilder (Boxen eingezeichnet) zum visuellen Anschauen
2. Eine Zahlen-Zusammenfassung (Erkennungsrate, Confidence, Klassenverteilung)

Verwendung:
    python check_inference.py
"""

from ultralytics import YOLO
from summarize_inference import summarize_predictions

from build_trainingsdata import CLASS_NAMES

MODEL_PATH = "/home/ubuntu/work/workspace/Projekt/runs/obb/train/weights/best.pt"  # <- ggf. anpassen
SOURCE_DIR = "/home/ubuntu/work/workspace/Projekt/src/inference_check"
CONF_THRESHOLD = 0.25


def main() -> None:
    model = YOLO(MODEL_PATH)

    # Nur EIN Inferenz-Durchlauf - das Ergebnis wird fuer beides genutzt
    results = model(SOURCE_DIR, stream=False, save=True, conf=CONF_THRESHOLD)

    print(f"Annotierte Bilder gespeichert in: {results[0].save_dir}\n")
    print("=" * 60)
    summarize_predictions(results, class_names=CLASS_NAMES)


if __name__ == "__main__":
    main()