"""Ausnahme-Hierarchie der chamber_obb Bibliothek (AD-16).

Alle Ausnahmen, die irgendeine interne Stufe der Bibliothek werfen kann,
erben von ``ChamberOBBError`` -- ``except ChamberOBBError:`` ist damit der
garantiert vollstaendige Catch-All fuer alles, was ``analyze()`` werfen kann
(abgesehen von echten Programmierfehlern wie einer falschen Aufrufsignatur).

Dieses Modul haengt von keinem anderen ``chamber_obb``-Modul ab -- die drei
Module mit tatsaechlichen raise-Stellen (``preprocessing.py``, ``inference.py``,
``pipeline.py``) importieren hiervon, nie umgekehrt.
"""


class ChamberOBBError(Exception):
    """Basisklasse aller chamber_obb-Ausnahmen (AD-16). ``except ChamberOBBError:``
    ist der garantiert vollstaendige Catch-All fuer alles, was analyze() werfen kann."""


class InvalidImageError(ChamberOBBError):
    """Ungueltiges, fehlendes oder korruptes Bild-Input."""


class ModelLoadError(ChamberOBBError):
    """Checkpoint-Pfad ungueltig, nicht gefunden, oder Ladevorgang fehlgeschlagen."""


class InvalidParameterError(ChamberOBBError):
    """Ein Parameterwert liegt ausserhalb seines gueltigen Bereichs/Typs."""
