"""Gemeinsame Primitive ohne eigene Domaene.

Kleine Helfer, die in mehr als einem Top-Level-Modul vorkamen und deshalb
eine einzige Heimat brauchen statt Kopien: Pfad-Aufloesung (``train()``,
``build_dataset()``) und das atomare Ersetzen einer Textdatei
(``dataset.json``-Descriptor in ``locks.py``, Run-Manifest in ``manifest.py``).
"""

import contextlib
import os
import tempfile
from pathlib import Path

from chamber_obb.errors import InvalidParameterError


def resolve_path(value: str | Path | None, label: str = "Pfad", *, required: bool = True) -> Path | None:
    """Loesst ``value`` zu einem absoluten Pfad auf (``expanduser`` + ``resolve``).

    Ein fehlender Wert ist ein ``InvalidParameterError``, wenn ``required``;
    sonst ``None``. Einzige Stelle dieser Aufloesungsregel fuer die explizit
    zu uebergebenden Pfade der oeffentlichen API (AD-8).
    """
    if not value:
        if required:
            raise InvalidParameterError(f"{label} nicht angegeben")
        return None
    return Path(value).expanduser().resolve()


def atomic_write_text(target: Path, text: str, *, prefix: str, fsync: bool = False) -> None:
    """Ersetzt den Inhalt von ``target`` atomar: Temp-Datei im selben Ordner + ``os.replace``.

    Gleicher Ordner garantiert, dass Ersetzen und Ziel auf einem Dateisystem
    liegen; beim Scheitern von Schreiben oder Ersetzen wird die Temp-Datei
    aufgeraeumt. ``fsync=True`` zwingt den Inhalt auf die Platte, bevor
    ersetzt wird (Pflicht fuer ``dataset.json``, schneller Weg fuer Manifeste).
    """
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=prefix, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            if fsync:
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
