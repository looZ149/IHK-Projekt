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


class DashboardIntegrationError(ChamberOBBError):
    """Von acia ausgehender Fehler beim Oeffnen der Sequenz oder des Dashboards."""


class DatasetDescriptorError(ChamberOBBError):
    """``dataset.json`` fehlt, ist beschaedigt oder passt nicht zum Datensatz-Inhalt (AD-23)."""


class DatasetBuildError(ChamberOBBError):
    """``build_dataset()`` kann nicht starten oder einen Chip-Ordner nicht verarbeiten (AD-20)."""


class ClassListMismatchError(ChamberOBBError):
    """Klassenliste von Checkpoint oder Datensatz weicht von der Paket-Klassenliste ab (AD-21)."""


class LeakageError(ChamberOBBError):
    """Val/Test-Sessions ueberlappen mit Train oder mit der Trainings-Historie (AD-22)."""


class EmptySplitError(ChamberOBBError):
    """Der Val-Split ist leer -- kein Ersatz durch Train, kein stilles Weitertrainieren (AD-22)."""


class TrainingError(ChamberOBBError):
    """Trainings- oder Evaluationslauf ist fehlgeschlagen (auch Fehler aus ultralytics, AD-16/AD-20)."""


class ManifestError(ChamberOBBError):
    """Run-Manifest fehlt in einem erwarteten Ort oder ist beschaedigt (AD-24)."""


class InferenceError(ChamberOBBError):
    """Die eigentliche Vorhersage (``model.predict``) ist fehlgeschlagen (AD-16)."""
