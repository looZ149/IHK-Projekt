"""Leave-One-Chip-Out Cross-Validation.

Haelt bei jedem Fold GENAU EINEN Chip-Ordner komplett als Test zurueck, baut
den Datensatz aus allen anderen Ordnern, trainiert ein frisches Modell und
wertet auf dem zurueckgehaltenen Ordner aus. Am Ende: Tabelle mit
Generalisierungs-Metriken PRO Ordner + Durchschnitt.

Ordner werden automatisch unter CHIPS_PATH entdeckt (kein manuelles Pflegen
einer Design-Gruppen-Liste mehr) - Namenspraefixe sind KEIN verlaesslicher
Hinweis auf die tatsaechliche Struktur (z.B. koennen zwei Ordner mit
aehnlichem Namen trotzdem unterschiedliche Kammer-/Kanal-Struktur haben).
Jeder Ordner ist intern konsistent (nur Kammer- ODER nur Kanal-ROIs), aber
Ordner untereinander koennen sich unterscheiden - daher ein eigener Fold
pro Ordner statt einer riskanten Gruppierung.

Zwei Klassen werden automatisch pro ROI erkannt (siehe build_trainingsdata.py
-> classify_roi_by_shape): chamber_region (0) und channel_region (1).

Verwendung:
    python leave_design_out_cv.py

Setzt voraus: build_trainingsdata.py liegt im selben Verzeichnis und
config.py definiert CHIPS_PATH/SEED wie gewohnt.
"""

import shutil
import subprocess
import sys
from pathlib import Path

from ultralytics import YOLO

import config
from build_trainingsdata import CLASS_NAMES, find_selection_jsons
from train_lock import LOCK_FILE

LOCK_FILE_BACKUP = Path(f"{LOCK_FILE}.cv_backup")

SCRIPT_DIR = Path(__file__).resolve().parent
BUILD_SCRIPT = SCRIPT_DIR / "build_trainingsdata.py"

BASE_MODEL = "yolo26n-obb.pt"
USE_REGISTRATION = True
FRAMES_PER_POSITION = "20"
EPOCHS = 100
PATIENCE = 40


def discover_chip_folders() -> list[str]:
    """Findet automatisch alle Chip-Ordner unter CHIPS_PATH (mit selection.json)."""
    chips_path = Path(config.CHIPS_PATH)
    return sorted(folder.name for folder, _ in find_selection_jsons(chips_path))


def suspend_train_lock() -> None:
    """Verschiebt eine bestehende train_locked_chips.txt kurzzeitig weg, damit
    die Cross-Validation-Folds sich nicht gegenseitig ueber die Sperrliste
    beeinflussen (ein Fold wuerde sonst Ordner fest auf Train sperren, die im
    naechsten Fold eigentlich als Test zurueckgehalten werden sollen)."""
    if LOCK_FILE.exists():
        shutil.move(str(LOCK_FILE), str(LOCK_FILE_BACKUP))
        print(f"Hinweis: bestehende {LOCK_FILE} temporaer beiseite gelegt "
              f"(wird nach der Cross-Validation wiederhergestellt).")


def restore_train_lock() -> None:
    """Stellt die urspruengliche train_locked_chips.txt wieder her und
    entfernt alles, was waehrend der Cross-Validation-Folds neu entstanden
    ist - die Folds selbst sollen nichts an der echten Sperrliste aendern."""
    if LOCK_FILE.exists():
        LOCK_FILE.unlink()
    if LOCK_FILE_BACKUP.exists():
        shutil.move(str(LOCK_FILE_BACKUP), str(LOCK_FILE))
        print(f"\n{LOCK_FILE} wiederhergestellt - unveraendert wie vor der Cross-Validation.")


def build_fold_dataset(held_out_folder: str, dataset_root: Path) -> None:
    """Baut den Datensatz fuer einen Fold: held_out_folder komplett als Test,
    alle anderen Ordner normal (zufaellig, aber nur Train/Val) verteilt."""
    cmd = [
        sys.executable, str(BUILD_SCRIPT),
        "--dataset-root", str(dataset_root),
        "--test-chips", held_out_folder,
        "--test-ratio", "0",  # kein zusaetzlicher zufaelliger Test-Ordner noetig
        "--val-ratio", "0.2",
    ]
    if USE_REGISTRATION:
        cmd += ["--use-registration", "--frames-per-position", FRAMES_PER_POSITION]

    print(f"\n{'='*70}\nFold '{held_out_folder}': Datensatz wird gebaut (Test = {held_out_folder})\n{'='*70}")
    subprocess.run(cmd, check=True)


def write_fold_data_yaml(dataset_root: Path) -> Path:
    yaml_path = dataset_root / "data.yaml"
    abs_root = dataset_root.resolve()
    names_block = "\n".join(f"  {cid}: {name}" for cid, name in sorted(CLASS_NAMES.items()))
    yaml_path.write_text(
        f"path: {abs_root}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )
    return yaml_path


def train_and_evaluate_fold(held_out_folder: str, data_yaml: Path) -> dict:
    print(f"\n--- Training fuer Fold '{held_out_folder}' ---")
    model = YOLO(BASE_MODEL)
    model.train(
        data=str(data_yaml),
        task="obb",
        epochs=EPOCHS,
        patience=PATIENCE,
        seed=config.SEED,
        batch=config.BATCH_SIZE,
        hsv_v=getattr(config, "HSV_V", 0.0),
        degrees=getattr(config, "DEGREES", 0.0),
        translate=getattr(config, "TRANSLATE", 0.0),
        scale=getattr(config, "SCALE", 0.0),
        flipud=getattr(config, "FLIPUD", 0.0),
        fliplr=getattr(config, "FLIPLR", 0.0),
        mosaic=0.0,
        project="runs/leave_one_chip_out",
        name=held_out_folder,
    )

    best_weights = Path(model.trainer.save_dir) / "weights" / "best.pt"
    eval_model = YOLO(str(best_weights))
    metrics = eval_model.val(data=str(data_yaml), split="test")

    return {
        "folder": held_out_folder,
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.p.mean()),
        "recall": float(metrics.box.r.mean()),
    }


def main() -> None:
    folders = discover_chip_folders()
    print(f"Gefundene Chip-Ordner ({len(folders)}): {', '.join(folders)}\n")
    print(f"Das ergibt {len(folders)} Folds (Leave-One-Chip-Out) - "
          f"je nach Trainingszeit kann das eine Weile dauern.\n")

    suspend_train_lock()
    results = []

    try:
        for held_out_folder in folders:
            dataset_root = Path(f"dataset_leaveout_{held_out_folder}")
            build_fold_dataset(held_out_folder, dataset_root)
            data_yaml = write_fold_data_yaml(dataset_root)
            result = train_and_evaluate_fold(held_out_folder, data_yaml)
            results.append(result)

            print(f"\nZwischenergebnis '{held_out_folder}': "
                  f"mAP50={result['mAP50']:.3f}  mAP50-95={result['mAP50-95']:.3f}  "
                  f"P={result['precision']:.3f}  R={result['recall']:.3f}")
    finally:
        # Sperrliste IMMER wiederherstellen, auch wenn ein Fold fehlschlaegt -
        # sonst verliert der normale Produktions-Workflow seinen alten Stand.
        restore_train_lock()

    if not results:
        print("Keine Folds erfolgreich abgeschlossen.")
        return

    print(f"\n{'='*70}\nLeave-One-Chip-Out Ergebnisse\n{'='*70}")
    header = f"{'Chip-Ordner':<25} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['folder']:<25} {r['mAP50']:>8.3f} {r['mAP50-95']:>10.3f} "
              f"{r['precision']:>10.3f} {r['recall']:>8.3f}")

    n = len(results)
    avg = {k: sum(r[k] for r in results) / n for k in ("mAP50", "mAP50-95", "precision", "recall")}
    print("-" * len(header))
    print(f"{'Durchschnitt':<25} {avg['mAP50']:>8.3f} {avg['mAP50-95']:>10.3f} "
          f"{avg['precision']:>10.3f} {avg['recall']:>8.3f}")

    print("\nDer Durchschnitt ist deine beste Schaetzung, wie gut das Modell auf "
          "einen KOMPLETT unbekannten Chip-Ordner reagiert. Grosse Unterschiede "
          "zwischen den Zeilen zeigen, bei welchem konkreten Ordner die "
          "Generalisierung besonders schwach ist.")


if __name__ == "__main__":
    main()