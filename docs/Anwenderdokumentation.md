# Anwenderdokumentation

Diese Dokumentation richtet sich an alle, die die `chamber_obb`-Bibliothek
dieses Repos installieren und eine Analyse durchführen wollen, ohne den Code
vorher gelesen zu haben. Sie beschreibt ausschließlich das, was in diesem
(stripped-down) Repo heute tatsächlich funktioniert.

## Installation

**Voraussetzungen:** Python ≥ 3.12 und [`uv`](https://docs.astral.sh/uv/).

1. Dieses Repo klonen und in das Repo-Verzeichnis wechseln.
2. Abhängigkeiten installieren:

   ```bash
   uv sync
   ```

   Das installiert unter anderem `chamber_obb` selbst (als Workspace-Mitglied,
   siehe `pyproject.toml`), `ultralytics` und `torch`.

3. Installation prüfen:

   ```bash
   uv run python -c "import chamber_obb"
   ```

   Der Befehl darf keine Ausgabe/Fehler erzeugen. Damit ist die Bibliothek
   einsatzbereit.

Es gibt aktuell kein `pip install git+...`-Paket für dieses Repo — die
Installation beschränkt sich bewusst auf "Repo klonen, `uv sync`".

## Analyse durchführen

Der einzige öffentliche Einstiegspunkt der Bibliothek ist
`chamber_obb.analyze()` (`src/chamber_obb/pipeline.py`). Alles andere in
`chamber_obb` (`preprocessing`, `inference`, `postprocessing`) ist
Implementierungsdetail und nicht Teil der öffentlichen API.

```python
import chamber_obb

detections = chamber_obb.analyze(
    image,                    # np.ndarray ODER Pfad (str/Path) zu einer .tif/.tiff-Datei
    metadata=None,             # optional, aktuell ohne Wirkung (siehe unten)
    checkpoint_path=None,      # optional, siehe "Checkpoint" unten
    conf=0.8,                  # optionaler Confidence-Threshold, Standard 0.8
)
```

**Parameter im Detail:**

- `image` (`np.ndarray | str | Path`, Pflicht): entweder ein bereits
  extrahiertes Frame als numpy-Array, oder ein Pfad zu einer `.tif`/`.tiff`-
  Datei. Beide Formen durchlaufen dieselbe Normalisierung auf 8-bit/3-Kanal
  — ein übergebenes Array gilt **nicht** bereits als normalisiert. Ein
  Array-`image` muss 2D (Graustufen) oder 3D mit letzter Achse `== 3` sein,
  sonst wirft `analyze()` einen `InvalidImageError`.
- `metadata` (optional): ein `FrameMetadata`-Dict (aktuell ohne definierte
  Felder). Wird entgegengenommen, aber von `analyze()` nirgends ausgewertet —
  auch nicht von der Dashboard-Integration (`dashboard.py`) weiter unten,
  die `analyze()` ohne `metadata` aufruft.
- `checkpoint_path` (optional, `str | Path`): Pfad zu einem trainierten
  YOLO-OBB-Checkpoint (`.pt`-Datei). Fehlt er, liest `analyze()` stattdessen
  die Umgebungsvariable `CHAMBER_OBB_CHECKPOINT`. Ist **keins** von beiden
  gesetzt, wirft `analyze()` einen `ModelLoadError`, bevor überhaupt ein
  Modell geladen wird.
- `conf` (optional, `float`, Standard `0.8`): Confidence-Threshold für die
  Detektion. Muss eine Zahl in `[0, 1]` sein, sonst wirft `analyze()` einen
  `InvalidParameterError`.

**Checkpoint:** Dieses stripped-down Repo enthält keinen trainierten
YOLO-OBB-Checkpoint. Ein echter End-to-End-Lauf setzt voraus, dass zuerst
irgendwo ein Modell trainiert und als Checkpoint (`best.pt`) abgelegt wurde —
wie ein neuer Checkpoint als "der aktuelle" markiert wird (per
`CHAMBER_OBB_CHECKPOINT`), ist in `RELEASE.md` beschrieben und wird hier
nicht dupliziert. Ohne gesetzten Checkpoint zeigt `analyze()` lediglich den
oben gezeigten Aufruf-Shape — es bricht mit `ModelLoadError` ab, sobald es
versucht, das Modell zu laden.

**Rückgabewert:** eine Liste von `Detection`-Objekten (leer, wenn nichts
erkannt wurde) — siehe "Ergebnisse interpretieren" unten.

**Fehlerbehandlung:** Alle Ausnahmen, die `analyze()` werfen kann, erben von
`chamber_obb.errors.ChamberOBBError`. `except ChamberOBBError:` ist damit
der garantiert vollständige Catch-All für alles, was aus `analyze()`
herauskommen kann (echte Programmierfehler wie eine falsche Aufrufsignatur
ausgenommen). Die drei konkreten Unterklassen:

- `InvalidImageError` — Bild-Input ungültig, fehlend oder korrupt.
- `ModelLoadError` — Checkpoint-Pfad ungültig/nicht gefunden, oder
  Ladevorgang fehlgeschlagen.
- `InvalidParameterError` — ein Parameterwert liegt außerhalb seines
  gültigen Bereichs/Typs (z. B. `conf` nicht in `[0, 1]`).

```python
from chamber_obb.errors import ChamberOBBError

try:
    detections = chamber_obb.analyze(image_path, checkpoint_path=checkpoint)
except ChamberOBBError as exc:
    ...  # eigene Fehlerbehandlung
```

## Dashboard-Integration

Neben dem direkten `analyze()`-Aufruf gibt es eine zweite, ebenfalls reale
Schnittstelle: `chamber_obb.dashboard.open_annotated_dashboard()`
(`src/chamber_obb/dashboard.py`). Sie führt `analyze()` auf einem
repräsentativen Frame (die mittlere Zeitaufnahme) je Position einer
`acia`-Sequenz (nd2/czi) aus, wandelt jede Detektion in eine
`acia.selection.RoiSelection` um und öffnet damit
`acia.notebook.SequenceDashboard` über `.resume()` — vorbefüllt mit den vom
Modell vorgeschlagenen ROIs statt einer leeren Leinwand. Der Aufruf ist
synchron und blockiert, bis das Dashboard bereit ist.

```python
from acia.segm.open import open_sequence
import chamber_obb.dashboard

f = open_sequence("/pfad/zur/aufnahme.nd2")
dash = chamber_obb.dashboard.open_annotated_dashboard(
    f,
    checkpoint_path=None,   # wie bei analyze(): optional, sonst CHAMBER_OBB_CHECKPOINT
    conf=0.8,
    roi_mode="multi",        # "single" (<=1 ROI/Position) oder "multi" (Default)
    save_dir=None,            # Standard-Ausgabeverzeichnis für SequenceDashboard.save
    preview_frame=None,       # None = mittlere Zeitaufnahme
)
dash
```

Diese Funktion ist es auch, die die produktive Notebook-Zelle in
`../01_SequencePreprocessing/SequencePreprocessing_v2.ipynb` aufruft.

**Wichtig — nur in `dashboard.py`, nicht im Core:** `acia` wird
ausschließlich von `dashboard.py` importiert (optionale
`chamber_obb[dashboard]`-Extra). `import chamber_obb` und der oben
beschriebene `analyze()`-Pfad funktionieren weiterhin vollständig ohne
`acia` installiert.

**Fehlerbehandlung:** Ein Fehler bei der Detektion auf einer einzelnen
Position wird geloggt und wie null Detektionen für diese Position behandelt
— der Lauf über die übrigen Positionen läuft weiter. Ein `ModelLoadError`
(fehlender/ungültiger Checkpoint) ist davon ausgenommen und propagiert
unverändert, da er kein auf eine Position isolierbarer Fehler ist, sondern
ein systemischer Konfigurationsfehler. Jeder von `acia` selbst ausgehende
Fehler beim Öffnen der Sequenz, Bauen des Manifests oder Öffnen des
Dashboards wird als `chamber_obb.errors.DashboardIntegrationError`
geworfen (erbt ebenfalls von `ChamberOBBError`).

**Installation der `acia`-Abhängigkeit:** `acia==0.4.0` ist nicht auf
öffentlichem PyPI verfügbar. In diesem Repo kommt es beim Entwickeln/Testen
über einen lokalen, editierbaren Checkout aus `../acia-core`
(`pyproject.toml` → `tool.uv.sources`). Für eine eigenständige Installation
(z. B. auf einer anderen Maschine, ohne diesen uv-Workspace) `acia` stattdessen
direkt aus dem privaten GitLab-Repo installieren — der genaue
`pip install`-Befehl mit Zugangsdaten steht in
`../01_SequencePreprocessing/00_installation.ipynb`.

## Ergebnisse interpretieren

`analyze()` liefert eine Liste von `Detection`-Objekten
(`chamber_obb/postprocessing.py`), jeweils bereits zurücktransformiert in die
Pixel-Koordinaten des ursprünglich übergebenen Bildes (nicht die internen
640×640-Letterbox-Koordinaten). Jede `Detection` hat vier Felder:

| Feld | Typ | Bedeutung |
| --- | --- | --- |
| `cls_id` | `int` | Numerische Klassen-ID der Detektion. |
| `cls_name` | `str` | Klassenname (`"chamber_region"` oder `"channel_region"`), gelesen aus dem Modell selbst — niemals hartkodiert. |
| `confidence` | `float` | Confidence-Score der Detektion (immer ≥ dem an `analyze()` übergebenen `conf`-Threshold). |
| `corners` | `np.ndarray` | Die vier Eckpunkte der orientierten Bounding-Box (OBB) im Original-Pixel-Raum. |

Wichtig für die Interpretation:

- Eine leere Liste bedeutet "keine Detektion oberhalb des Thresholds" — kein
  Fehler.
- Es gibt **keine** Best-of-Class-Filterung: Jede Detektion oberhalb des
  Confidence-Thresholds wird durchgereicht, auch mehrere pro Klasse im
  selben Bild.
- Eckpunkte, die im ursprünglichen Padding-Bereich der Letterbox-Transform
  liegen würden, werden nicht auf die Originalbild-Grenzen geclippt — das
  ist korrekte, absichtlich unveränderte Information.

## Bekannte Einschränkungen

- **Kein trainierter Checkpoint enthalten:** Dieses stripped-down Repo
  liefert kein `best.pt`. Ohne einen selbst bereitgestellten, trainierten
  YOLO-OBB-Checkpoint kann kein echter End-to-End-Lauf durchgeführt werden —
  `analyze()` bricht dann mit `ModelLoadError` ab.
- **Dashboard-Notebook nicht Teil dieses Repos:** `open_annotated_dashboard()`
  selbst ist fertig (siehe "Dashboard-Integration" oben), aber das
  konsumierende Notebook (`SequencePreprocessing_v2.ipynb`) und `acia` liegen
  außerhalb dieses Repos, in den nicht von hier aus versionierten
  Geschwisterordnern `../01_SequencePreprocessing` und `../acia-core`.
