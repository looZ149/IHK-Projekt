"""Checkpoint-Aufloesung fuer ``analyze()`` und ``train()`` (AD-8, AD-15).

Einzige Stelle, an der der Name der Umgebungsvariablen ``CHAMBER_OBB_CHECKPOINT``
vorkommt. ``pipeline.analyze()`` und ``training.train()`` fallen beide auf sie
zurueck, wenn kein Checkpoint-Pfad explizit uebergeben wurde. Hier findet
kein Modell-Laden und kein Caching statt -- ausschliesslich die Aufloesung,
woher ein Aufrufer seinen Checkpoint-Pfad bezieht.
"""

import os
from pathlib import Path

CHECKPOINT_ENV_VAR = "CHAMBER_OBB_CHECKPOINT"


def resolve_checkpoint(value: str | Path | None) -> str | None:
    """Expliziter Checkpoint-Pfad oder Env-Var-Fallback; ``None`` wenn beides fehlt."""
    if value:
        return str(value)
    return os.environ.get(CHECKPOINT_ENV_VAR)
