"""chamber_obb -- installierbare OBB-Detektions- und Trainings-Bibliothek (CAP-2, CAP-4, CAP-7).

Oeffentliche Oberflaeche (AD-2): ``analyze()`` fuer die Detektion sowie
``build_dataset()`` und ``train()`` fuer das Weitertrainieren auf neuen Daten,
jeweils mit ihren typisierten Ein-/Ausgaben. ``preprocessing``/``inference``/
``postprocessing`` und die Session-/Lock-Logik sind Implementierungsdetail. Einzige Ausnahme:
``migrate_legacy_dataset()``, der einmalige Administrator-Schritt fuer den Alt-Datensatz.
Der optionale ``chamber_obb.dashboard`` (Extra ``dashboard``) wird bewusst
nicht hier importiert -- ``import chamber_obb`` braucht kein ``acia``.
"""

from chamber_obb.dataset import BuildReport, build_dataset
from chamber_obb.locks import migrate_legacy_dataset
from chamber_obb.pipeline import FrameMetadata, analyze
from chamber_obb.training import Metrics, TrainResult, train

__all__ = [
    "BuildReport",
    "FrameMetadata",
    "Metrics",
    "TrainResult",
    "analyze",
    "build_dataset",
    "migrate_legacy_dataset",
    "train",
]
