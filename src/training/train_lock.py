"""Verwaltet eine dauerhafte Liste von Chip-Ordnern, die JEMALS in Training
waren - und daher fuer alle Zeit im Train-Split bleiben MUESSEN.

Grund: Wird spaeter mit einem bestehenden Checkpoint weitertrainiert (statt
komplett neu ab yolo26n-obb.pt), stecken in den Gewichten bereits
Informationen ueber alle bisher trainierten Ordner. Wuerde so ein Ordner in
einem spaeteren Rebuild zufaellig in Val/Test landen, waere die Auswertung
verfaelscht (das Modell "kennt" ihn ja schon aus einem frueheren Lauf) - auch
wenn der AKTUELLE Datensatz selbst sauber getrennt ist.

Verwendung in build_training_dataset.py:
    from train_lock import discover_filesystem_train_chips, load_locked_chips, update_locked_chips
    ...
    locked = load_locked_chips()
    locked |= discover_filesystem_train_chips(dataset_root)  # AD-18: Textdatei allein reicht nicht
    forced_train |= locked   # zusaetzlich zu manuell via --train-chips
    ...
    # nach erfolgreichem Lauf:
    update_locked_chips(forced_train | {n for n, s in assignment.items() if s == "train"})
"""

import re
from pathlib import Path

LOCK_FILE = Path("train_locked_chips.txt")

# Namensschema der Chip-Bilddateien (<chip>_pos<N>_frame<M>.<ext>), geschrieben
# von build_trainingsdata.py's process_selection_file. Beide Muster leben
# bewusst HIER (eine Stelle) und nicht noch einmal im Builder: train_lock hat
# null externe Abhaengigkeiten und build_trainingsdata importiert train_lock,
# nie umgekehrt (Kreisimport). Das Schreib-Format (f-String) bleibt im Builder;
# die Regexes dazu sind der Desync-Schutz an dieser einen Stelle.
_CHIP_NAME_SUFFIX_RE = re.compile(r"_pos\d+_frame\d+$")
CHIP_IMAGE_NAME_RE = re.compile(r"^(.+)_pos\d+_frame\d+\.\w+$")


def load_locked_chips() -> set[str]:
    """Liest die bisher gesperrten (immer-Train-)Ordnernamen, eine pro Zeile."""
    if not LOCK_FILE.exists():
        return set()
    return {line.strip() for line in LOCK_FILE.read_text().splitlines() if line.strip()}


def discover_filesystem_train_chips(dataset_root: Path) -> set[str]:
    """Findet Chip-Ordnernamen, die AKTUELL unter dataset_root/images/train/
    vorhanden sind (AD-18) -- niemals train_locked_chips.txt allein vertrauen,
    da CAP-5s Annotationsworkflow Sessions direkt dort ablegen kann, ohne
    train_lock.py ueberhaupt zu kennen. Muss VOR clean_split_dirs() aufgerufen
    werden, sonst sieht diese Funktion nur den eigenen, frisch geschriebenen
    Output dieses Laufs.
    """
    train_images_dir = dataset_root / "images" / "train"
    if not train_images_dir.is_dir():
        return set()
    names: set[str] = set()
    for f in train_images_dir.glob("*.png"):
        name = _CHIP_NAME_SUFFIX_RE.sub("", f.stem)
        if name and name != f.stem:  # nicht-Treffer UND einen entarteten Leer-Praefix-Treffer ausschliessen
            names.add(name)
        elif name == f.stem:
            print(f"Warnung: Datei '{f.name}' unter {train_images_dir} entspricht nicht dem "
                  f"erwarteten Namensschema (<chip>_pos<N>_frame<M>.png) - wird bei der "
                  f"Sperrlisten-Erkennung ignoriert.", flush=True)
    return names


def update_locked_chips(train_folder_names: set[str]) -> None:
    """Ergaenzt die Sperrliste um alle aktuell in Train befindlichen Ordner
    (bestehende Eintraege bleiben erhalten - nur Hinzufuegen, nie Entfernen)."""
    current = load_locked_chips()
    updated = current | train_folder_names
    LOCK_FILE.write_text("\n".join(sorted(updated)) + "\n", encoding="utf-8")
    new_entries = updated - current
    if new_entries:
        print(f"Neu zur Train-Sperrliste hinzugefuegt: {', '.join(sorted(new_entries))}")