"""Verwaltet eine dauerhafte Liste von Chip-Ordnern, die JEMALS in Training
waren - und daher fuer alle Zeit im Train-Split bleiben MUESSEN.

Grund: Wird spaeter mit einem bestehenden Checkpoint weitertrainiert (statt
komplett neu ab yolo26n-obb.pt), stecken in den Gewichten bereits
Informationen ueber alle bisher trainierten Ordner. Wuerde so ein Ordner in
einem spaeteren Rebuild zufaellig in Val/Test landen, waere die Auswertung
verfaelscht (das Modell "kennt" ihn ja schon aus einem frueheren Lauf) - auch
wenn der AKTUELLE Datensatz selbst sauber getrennt ist.

Verwendung in build_training_dataset.py:
    from train_lock import load_locked_chips, update_locked_chips
    ...
    locked = load_locked_chips()
    forced_train |= locked   # zusaetzlich zu manuell via --train-chips
    ...
    # nach erfolgreichem Lauf:
    update_locked_chips(forced_train | {n for n, s in assignment.items() if s == "train"})
"""

from pathlib import Path

LOCK_FILE = Path("train_locked_chips.txt")


def load_locked_chips() -> set[str]:
    """Liest die bisher gesperrten (immer-Train-)Ordnernamen, eine pro Zeile."""
    if not LOCK_FILE.exists():
        return set()
    return {line.strip() for line in LOCK_FILE.read_text().splitlines() if line.strip()}


def update_locked_chips(train_folder_names: set[str]) -> None:
    """Ergaenzt die Sperrliste um alle aktuell in Train befindlichen Ordner
    (bestehende Eintraege bleiben erhalten - nur Hinzufuegen, nie Entfernen)."""
    current = load_locked_chips()
    updated = current | train_folder_names
    LOCK_FILE.write_text("\n".join(sorted(updated)) + "\n", encoding="utf-8")
    new_entries = updated - current
    if new_entries:
        print(f"Neu zur Train-Sperrliste hinzugefuegt: {', '.join(sorted(new_entries))}")