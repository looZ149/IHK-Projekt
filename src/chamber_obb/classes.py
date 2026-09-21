"""Die eine Klassenliste der Bibliothek und die ROI-zu-Klasse-Regel (AD-13).

Einzige Stelle, an der ``chamber_region``/``channel_region`` beim Namen
vorkommen. ``build_dataset()`` und ``train()`` lesen beide von hier; die
Detektions-Pipeline (``analyze()``) bleibt davon unabhaengig und liest
Klassennamen ausschliesslich aus ``results.names`` des geladenen Checkpoints.
Eine Klasse hinzuzufuegen heisst: hier ergaenzen + Regel anpassen (AD-21
schuetzt vor unbemerktem Index-Drift).
"""

# Reihenfolge = Klassen-Index. Nur am Ende erweitern, nie umsortieren.
CLASS_NAMES: tuple[str, ...] = ("chamber_region", "channel_region")

# Breite:Hoehe (oder umgekehrt) ab diesem Wert gilt eine ROI als Kanal. Pro
# Aufruf ueberschreibbar; der verwendete Wert steht im Dataset-Descriptor (AD-21).
DEFAULT_CHANNEL_ASPECT_RATIO_THRESHOLD = 2.5

_CHANNEL_CLASS_ID = 1


def class_list() -> list[str]:
    """Gibt die Klassenliste als neue Liste zurueck (Index = Position)."""
    return list(CLASS_NAMES)


def classify_roi_by_shape(
    w: float, h: float, threshold: float = DEFAULT_CHANNEL_ASPECT_RATIO_THRESHOLD
) -> int:
    """Bestimmt die Klasse einer ROI anhand ihres Seitenverhaeltnisses.

    Kammer-Boxen sind ungefaehr quadratisch, Kanal-Boxen stark laenglich. Die
    Klasse wird pro ROI bestimmt, nicht pro Chip-Ordner, weil beide Typen auch
    innerhalb eines Ordners gemischt vorkommen koennen.
    """
    aspect_ratio = max(w, h) / max(min(w, h), 1e-6)
    return _CHANNEL_CLASS_ID if aspect_ratio >= threshold else 0
