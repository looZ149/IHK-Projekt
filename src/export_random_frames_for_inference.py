"""Exportiert zufaellige Positionen/Frames aus NEUEN, bisher unbenutzten
ND2-Dateien als einzelne 8-bit TIFFs - fuer den visuellen Test, ob das
trainierte Modell auch auf echten, nie gesehenen Rohdaten funktioniert.

Bewusst UNABHAENGIG von der Trainings-Pipeline (selection.json,
registration_transforms.json, acia): diese Dateien sollen noch nirgends
annotiert oder sonst irgendwo verwendet worden sein - genau deshalb wird
hier direkt mit dem 'nd2'-Package gelesen, nicht ueber build_trainingsdata.py.

Zwei Modi:
  --nd2 PFAD           Eine konkrete Datei
  --search-dir PFAD    Durchsucht den Ordner (inkl. Unterordner) rekursiv
                        nach *.nd2 Dateien und waehlt zufaellig --num-files
                        davon aus

Verwendung:
    python export_random_frames_for_inference.py \
        --search-dir /pfad/zu/irgendeinem/rohdaten_ordner \
        --num-files 2 --num-positions 5 --frames-per-position 3

Direkt danach Inferenz testen:
    from ultralytics import YOLO
    model = YOLO("runs/final/final_model/weights/best.pt")
    model.predict("inference_check", save=True, conf=0.25)

Benoetigt: pip install nd2 pillow numpy
"""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image


def normalize_to_8bit(img: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    """Perzentil-basiertes Kontrast-Stretching statt Min-Max - siehe
    build_trainingsdata.py fuer die ausfuehrliche Begruendung. Muss identisch
    zur Trainings-Pipeline bleiben, sonst 'sieht' das Modell hier einen
    anderen Kontrast-Stil als beim Training."""
    lo, hi = np.percentile(img, [p_low, p_high])
    clipped = np.clip(img, lo, hi)
    return ((clipped - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)


def find_nd2_files(search_dir: Path) -> list[Path]:
    """Durchsucht search_dir rekursiv (inkl. aller Unterordner) nach *.nd2 Dateien.

    Nutzt os.walk statt Path.rglob, um live Fortschritt anzuzeigen - bei
    breiten/tiefen Verzeichnisbaeumen (v.a. auf Netzwerk-Mounts) kann die
    Suche sonst laenger dauern, ohne dass sichtbar ist, dass sie noch laeuft.
    """
    print(f"Durchsuche '{search_dir}' nach .nd2 Dateien ...", flush=True)
    start = time.monotonic()

    found: list[Path] = []
    dirs_visited = 0
    for root, dirs, files in os.walk(search_dir):
        dirs_visited += 1
        for name in files:
            if name.lower().endswith(".nd2"):
                found.append(Path(root) / name)

        if dirs_visited % 50 == 0:
            elapsed = time.monotonic() - start
            print(f"  ... {dirs_visited} Ordner durchsucht, {len(found)} .nd2 Datei(en) "
                  f"gefunden bisher ({elapsed:.0f}s)", flush=True)

    elapsed = time.monotonic() - start
    print(f"Fertig: {dirs_visited} Ordner durchsucht, {len(found)} .nd2 Datei(en) "
          f"gefunden, {elapsed:.1f}s benoetigt.", flush=True)
    return sorted(found)


def export_from_nd2(nd2_path: Path, output_dir: Path, num_positions: int,
                     frames_per_position: int, rng: random.Random) -> int:
    """Exportiert zufaellige Positionen/Frames aus einer einzelnen ND2-Datei.
    Gibt die Anzahl exportierter Bilder zurueck.

    Bei einem Fehler mitten in der Datei (z.B. Positions-/Frame-Anzahl in den
    Metadaten passt nicht zur tatsaechlichen Array-Form - kommt bei
    uneinheitlich aufgenommenen ND2-Dateien durchaus vor) werden bereits
    geschriebene Teilexporte dieser Datei geloescht und die Exception wird
    weitergereicht, damit main() die Datei komplett verwirft und stattdessen
    eine andere zieht.
    """
    import nd2

    written_paths: list[Path] = []

    try:
        with nd2.ND2File(str(nd2_path)) as f:
            sizes = f.sizes  # z.B. {'T': 115, 'P': 39, 'Y': 1608, 'X': 1608}
            total_frames = sizes.get("T", 1)
            total_positions = sizes.get("P", 1)
            print(f"\n{nd2_path.name}: {total_positions} Positionen, {total_frames} Frames/Position", flush=True)

            # to_dask() statt asarray(): laedt nichts vorab in den RAM, wichtig bei
            # grossen Dateien (siehe build_trainingsdata.py fuer die ausfuehrliche
            # Begruendung - dasselbe OOM-Risiko besteht hier genauso).
            arr = f.to_dask()  # ueblich (T, P, Y, X) - bei Bedarf Achsenreihenfolge pruefen

            n_pos = min(num_positions, total_positions)
            chosen_positions = rng.sample(range(total_positions), k=n_pos)
            stem = nd2_path.stem

            for pos in sorted(chosen_positions):
                n_frames = min(frames_per_position, total_frames)
                chosen_frames = rng.sample(range(total_frames), k=n_frames)
                for frame_idx in sorted(chosen_frames):
                    raw = arr[frame_idx, pos].compute()

                    # Manche Dateien haben eine zusaetzliche Achse (z.B. ein
                    # Fluoreszenz-Kanal wie GFP neben Phasenkontrast), die
                    # unsere (T, P, Y, X)-Annahme nicht kennt. Ohne diese
                    # Pruefung wuerde dann ein degeneriertes "Bild" (z.B. 1px
                    # breit) OHNE Fehler gespeichert - erst bei der Inferenz
                    # spaeter wuerde das mit einem kryptischen cv2-Fehler
                    # auffallen. Stattdessen hier sofort erkennen und als
                    # Fehler werfen, damit die Datei automatisch uebersprungen
                    # und durch eine andere ersetzt wird (siehe main()).
                    if raw.ndim != 2:
                        raise ValueError(
                            f"Unerwartete Array-Form nach Indexierung: {raw.shape} "
                            f"(erwartet 2D (Y,X)) - vermutlich eine zusaetzliche Achse "
                            f"(z.B. Fluoreszenz-Kanal) in dieser Datei; die Achsenannahme "
                            f"(T,P,Y,X) passt hier nicht."
                        )
                    if min(raw.shape) < 10:
                        raise ValueError(
                            f"Verdaechtig kleine Dimension nach Indexierung: {raw.shape} "
                            f"- Achsenreihenfolge passt vermutlich nicht fuer diese Datei."
                        )

                    img_8bit = normalize_to_8bit(raw)

                    name = f"{stem}_pos{pos}_frame{frame_idx:03d}.tiff"
                    out_path = output_dir / name
                    Image.fromarray(img_8bit).save(out_path)
                    written_paths.append(out_path)
                    print(f"  Exportiert: {name}", flush=True)
    except Exception:
        for path in written_paths:
            path.unlink(missing_ok=True)
        raise

    return len(written_paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zufaellige Frames aus (einer oder mehreren) neuen ND2-Datei(en) "
                     "fuer den Inferenz-Test exportieren"
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--nd2", help="Pfad zu EINER konkreten ND2-Datei")
    source_group.add_argument("--search-dir",
                               help="Ordner, der (auch in Unterordnern) nach *.nd2 "
                                    "Dateien durchsucht wird - zufaellige Auswahl daraus")

    parser.add_argument("--num-files", type=int, default=1,
                         help="Nur mit --search-dir: wie viele ND2-Dateien zufaellig "
                              "ausgewaehlt werden (Standard: 1)")
    parser.add_argument("--output-dir", default="inference_check")
    parser.add_argument("--num-positions", type=int, default=5,
                         help="Wie viele zufaellige Positionen pro Datei (Standard: 5)")
    parser.add_argument("--frames-per-position", type=int, default=3,
                         help="Wie viele zufaellige Frames pro gewaehlter Position (Standard: 3)")
    parser.add_argument("--seed", type=int, default=None,
                         help="Optional, fuer reproduzierbare Auswahl (Standard: echter Zufall)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    total_exported = 0
    failed_files: list[str] = []

    if args.nd2:
        # Einzelne, explizit angegebene Datei - kein Ersatz bei Fehler moeglich.
        try:
            total_exported += export_from_nd2(
                Path(args.nd2), output_dir, args.num_positions, args.frames_per_position, rng
            )
        except Exception as exc:
            raise SystemExit(f"Fehler beim Verarbeiten von '{args.nd2}': {exc}")
    else:
        search_dir = Path(args.search_dir)
        all_nd2 = find_nd2_files(search_dir)
        if not all_nd2:
            raise SystemExit(f"Keine .nd2 Dateien unter '{search_dir}' gefunden "
                              f"(auch nicht in Unterordnern).")
        print(f"{len(all_nd2)} .nd2 Datei(en) unter '{search_dir}' gefunden.", flush=True)

        candidates = all_nd2.copy()
        rng.shuffle(candidates)

        succeeded: list[str] = []
        while candidates and len(succeeded) < args.num_files:
            nd2_path = candidates.pop()
            try:
                total_exported += export_from_nd2(
                    nd2_path, output_dir, args.num_positions, args.frames_per_position, rng
                )
                succeeded.append(nd2_path.name)
            except Exception as exc:
                print(f"Warnung: '{nd2_path.name}' uebersprungen wegen Fehler "
                      f"({exc}) - versuche eine andere Datei ...", flush=True)
                failed_files.append(nd2_path.name)

        if len(succeeded) < args.num_files:
            print(f"\nHinweis: nur {len(succeeded)} von {args.num_files} gewuenschten "
                  f"Dateien erfolgreich verarbeitet - keine weiteren Kandidaten mehr "
                  f"uebrig.", flush=True)

    if failed_files:
        print(f"\nUebersprungene Datei(en) wegen Fehler ({len(failed_files)}): "
              f"{', '.join(failed_files)}", flush=True)

    print(f"\n{total_exported} Bild(er) insgesamt in '{output_dir}' gespeichert.", flush=True)
    print("\nJetzt Inferenz testen, z.B.:", flush=True)
    print("  from ultralytics import YOLO", flush=True)
    print("  model = YOLO('runs/final/final_model/weights/best.pt')", flush=True)
    print(f"  model.predict('{output_dir}', save=True, conf=0.25)", flush=True)


if __name__ == "__main__":
    main()