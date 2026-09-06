"""Zaehlt, wie viele Bild/Label-Paare pro Chip-Ordner in jedem Split liegen -
hilft zu pruefen, ob ein bestimmter Kammertyp (Chip-Ordner-Praefix) im
Trainingsdatensatz unterrepraesentiert ist.

Verwendung:
    python check_split_balance.py --dataset-root dataset
"""

import argparse
import re
from collections import Counter
from pathlib import Path

# Erkennt den Chip-Ordnernamen als Praefix vor "_posN_frameNNN.png"
NAME_PATTERN = re.compile(r"^(.+)_pos\d+_frame\d+\.\w+$")


def count_by_folder(images_dir: Path) -> Counter:
    counts = Counter()
    for img_path in images_dir.glob("*"):
        match = NAME_PATTERN.match(img_path.name)
        if match:
            counts[match.group(1)] += 1
        else:
            counts["<unbekanntes Format>"] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="dataset")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    all_folders = set()
    per_split = {}

    for split in ("train", "val", "test"):
        images_dir = root / "images" / split
        if not images_dir.exists():
            continue
        counts = count_by_folder(images_dir)
        per_split[split] = counts
        all_folders |= set(counts.keys())

    header = f"{'Chip-Ordner':<35} {'Train':>8} {'Val':>8} {'Test':>8} {'Gesamt':>8}"
    print(header)
    print("-" * len(header))
    for folder in sorted(all_folders):
        row = [per_split.get(s, Counter()).get(folder, 0) for s in ("train", "val", "test")]
        total = sum(row)
        print(f"{folder:<35} {row[0]:>8} {row[1]:>8} {row[2]:>8} {total:>8}")

    print("-" * len(header))
    for split in ("train", "val", "test"):
        total = sum(per_split.get(split, Counter()).values())
        print(f"Gesamt {split}: {total}")


if __name__ == "__main__":
    main()