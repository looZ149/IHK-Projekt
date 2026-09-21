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

Der öffentliche Einstiegspunkt für die Detektion ist
`chamber_obb.analyze()` (`src/chamber_obb/pipeline.py`). Für das Weitertrainieren
gibt es zusätzlich `build_dataset()` und `train()` (siehe "Training auf neuen
Daten"). Alles andere in `chamber_obb` (`preprocessing`, `inference`,
`postprocessing`, Session-/Lock-Logik) ist Implementierungsdetail und nicht Teil
der öffentlichen API.

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

## Training auf neuen Daten

Wenn neue Chip-Designs oder neue Experimente dazukommen, kann das Modell
weitertrainiert werden. Dafür gibt es zwei öffentliche Funktionen:
`chamber_obb.build_dataset()` und `chamber_obb.train()`. Geführt durch den
Ablauf wird man vom Notebook `Training.ipynb` (im Notebook-Ordner
`image_analysis_V3`). Die Funktionen selbst brauchen kein Notebook.

**Ohne Daten im Paket:** `pip install` liefert nur Code — keine Bilder, Labels
oder Gewichte. Der bisherige Trainingsdatensatz liegt in einem gemeinsamen
Ordner auf dem KIT-Rechner (`old_dataset`) und wird **nur gelesen**.

### Pfade

| Name | Bedeutung | Umgebungsvariable |
| --- | --- | --- |
| `old_dataset` | Gemeinsamer, bisheriger YOLO-Datensatz (nur lesend) | `CHAMBER_OBB_OLD_DATASET` |
| `new_data` | Neue, kuratierte Chip-Ordner (`selection.json` + Aufnahme, optional `registration_transforms.json`) aus dem Notebook `SequencePreprocessing` | `CHAMBER_OBB_NEW_DATA` |
| `new_dataset` | Ausgabe von `build_dataset()`: der neue Datensatz | `CHAMBER_OBB_NEW_DATASET` |
| `base_checkpoint` | Gewichte, ab denen weitertrainiert wird (dasselbe wie für `analyze()`) | `CHAMBER_OBB_CHECKPOINT` |
| `runs_dir` | Ordner für die Trainingsläufe | `CHAMBER_OBB_RUNS_DIR` |

### Ablauf

```python
import chamber_obb

report = chamber_obb.build_dataset(new_data, new_dataset, old_dataset)   # 1. Datensatz bauen
print(report.sessions, report.skipped, report.warnings)                   # 2. Split prüfen

result = chamber_obb.train(new_dataset, old_dataset, base_checkpoint,     # 3. trainieren
                           mode="both", runs_dir=runs_dir)
print(result.val, result.test, result.best_weights)                       # 4. Metriken prüfen
```

- **`build_dataset()`** wandelt die Chip-Ordner in einen YOLO-OBB-Datensatz
  um. Jede ROI wird von dem Frame exportiert, auf dem sie gezeichnet wurde
  (`anchor_frame` in `selection.json`; das Dashboard öffnet auf dem mittleren Frame).
  Ohne Registrierung entsteht ein Bild pro Position und Anker-Frame; liegen die
  ROIs einer Position auf verschiedenen Frames, wird sie ohne Registrierung
  übersprungen. Mit `use_registration=True` und einer
  `registration_transforms.json` werden die ROIs auf weitere Frames übertragen. Der Split erfolgt **pro Chip-Ordner (Session)**, nie pro Bild. Es wird
  nur in `new_dataset` geschrieben und standardmäßig **additiv** gearbeitet:
  Bestehende Sessions behalten ihren Split; `rebuild=True` leert und teilt neu
  auf. Sessions, die bereits im Training waren, bleiben für immer in Train.
  Der `old_dataset` liefert **kein unabhängiges Val/Test** (siehe unten); liegt eine
  Session dort in Val/Test, wird sie aus `new_data` abgelehnt. Fehlerhafte Chip-Ordner werden übersprungen und in
  `report.skipped` genannt. Für ND2-Aufnahmen wird `acia` gebraucht
  (`pip install "chamber_obb[dashboard]"`).
- **`train()`** wählt mit `mode` die Trainingsdaten: `"new"` (nur neue),
  `"old"` (nur alte) oder `"both"` (alte + neue — vermeidet, dass alte
  Chip-Designs vergessen werden). Jeder Aufruf legt ein **neues Run-Verzeichnis**
  an; der Basis-Checkpoint und der alte Datensatz werden nie verändert. Im Run-
  Verzeichnis liegen `manifest.json` (Herkunft des Laufs), `data.yaml`, die
  Bildlisten und die Gewichte (`fit/weights/best.pt`). Ein Checkpoint wird
  **nie automatisch** übernommen (siehe `RELEASE.md`).
- **Ergebnis:** `result.val` und `result.test` sind `Metrics` (`map50`,
  `map50_95`, `precision`, `recall`). `result.test` ist `None`, wenn kein
  Test-Split vorhanden war. `result.baseline_val` / `result.baseline_test` sind
  dieselben Zahlen für den **Basis-Checkpoint** auf **denselben** Val/Test-Sessions,
  vor dem Training gemessen (`evaluate_baseline=False` schaltet das ab; bei
  Stock-Start und `reinit_head` gibt es keine Baseline). Nur dieser Vergleich sagt,
  ob der neue Checkpoint besser ist als der aktuelle.

### Der Alt-Datensatz liefert kein Val/Test

Das aktuelle Modell wurde auf dem **gesamten** Alt-Datensatz trainiert. Die Bilder
unter `val/` und `test/` dort sind nur Ausschnitte bereits trainierter Sessions
(Early-Stopping-Monitoring) und keine unabhängige Prüfung. Deshalb:

- Der Alt-Datensatz ist **reine Trainingsdaten** (alle seine Bilder, auch die aus
  `val/` und `test/`); Val und Test kommen **ausschließlich aus dem `new_dataset`**.
- Das `new_dataset` braucht mindestens eine Val-Session (besser auch 1–2 Test-Sessions):
  neue Chips, die weder das aktuelle noch das neue Modell gesehen hat. Dafür beim Bauen
  `val_chips=[...]` und `test_chips=[...]` angeben oder genug neue Chips liefern.
  Sonst wirft `train()` einen `EmptySplitError`.
- Die Metriken auf so wenigen Sessions sind verrauscht — als Hinweis lesen, nicht als Beweis.

### Schutz vor Leakage (Fehler, die `train()` vor dem Training wirft)

- `LeakageError`: Val/Test-Sessions überschneiden sich mit Train, oder eine
  Val/Test-Session wurde vom Basis-Checkpoint (oder einem Vorgänger) bereits
  trainiert. Die Historie steht in `manifest.json` neben den Gewichten — beim
  Kopieren eines Checkpoints die Datei als `best.manifest.json` daneben legen.
  Ist die Historie unbekannt (kein Manifest, kein `old_dataset`), wirft
  `train()` ebenfalls; mit `acknowledge_unknown_lineage=True` bewusst bestätigen.
- `EmptySplitError`: Val ist leer. Es gibt keinen Ersatz durch Train, und der
  Alt-Datensatz liefert kein Val. Beim Bauen `val_chips`/`test_chips` aus den neuen
  Chips angeben.
- `ClassListMismatchError`: Die Klassen des Basis-Checkpoints oder eines
  Datensatzes passen nicht zur Paket-Klassenliste. Für den Start von
  `yolo26n-obb.pt` oder eine neue Klasse `reinit_head=True` setzen.

### Einmalig vor dem ersten Lauf (Administrator mit Schreibrechten)

1. **Alt-Datensatz migrieren:** Der bisherige Datensatz hat noch keine
   `dataset.json`; ohne sie verweigern `build_dataset()` und `train()` ihn
   (fail closed). Mit Schreibrechten im gemeinsamen Ordner:

   ```python
   import chamber_obb
   report = chamber_obb.migrate_legacy_dataset(
       "/pfad/zum/dataset", "/pfad/zur/train_locked_chips.txt")
   print(report["sessions"], report["straddling"])
   ```

   `train_locked_chips.txt` liegt dort, wo das alte Skript zuletzt ausgeführt
   wurde (bisher das aktuelle Arbeitsverzeichnis). Die Migration markiert den
   Datensatz als `all_sessions_trained`: alle Sessions sind Train und für immer
   gesperrt, die Bilder in `val/` und `test/` zählen als Trainingsdaten.
   (`all_sessions_trained=False` gibt es nur für einen Datensatz mit echtem,
   unabhängigem Val/Test.)
2. **Bericht prüfen:** `report["sessions"]` sollte alle bekannten Chip-Ordner enthalten.
   `report["straddling"]` listet Sessions mit Dateien in mehr als einem Split — bei
   dem `--all-train`-Datensatz erwartet (das Monitoring-Val) und nur zur Information.
3. **Neue Daten in den gemeinsamen Datensatz übernehmen** ist bewusst ein
   manueller Schritt (Kopieren von `new_dataset` samt `dataset.json` durch
   jemanden mit Schreibrechten); eine `promote`-Funktion gibt es noch nicht.
   Ein neu übernommener Datensatz hat dann wieder ein echtes Val/Test — beim
   Zusammenführen die Sperrlisten (`locked_train_sessions`) vereinigen und nichts entfernen.
4. Das alte CLI unter `training/` darf nicht in einen Datensatz-Ordner
   schreiben, der eine `dataset.json` hat.

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

- **Training nur mit Stub-Tests geprüft:** `build_dataset()` und `train()` sind mit
  Fake-Aufnahmen und einem Fake-YOLO getestet (kein GPU, kein ND2). Ein echter
  Trainingslauf auf dem KIT-Rechner mit echten Chips steht noch aus.
- **`.cache`-Dateien im gemeinsamen Datensatz:** ultralytics legt beim Training `labels/*.cache` neben den Labels an — auch im (sonst nur gelesenen) `old_dataset`. Ist der Ordner nicht beschreibbar, gibt es nur eine Warnung.
- **Kein trainierter Checkpoint enthalten:** Dieses stripped-down Repo
  liefert kein `best.pt`. Ohne einen selbst bereitgestellten, trainierten
  YOLO-OBB-Checkpoint kann kein echter End-to-End-Lauf durchgeführt werden —
  `analyze()` bricht dann mit `ModelLoadError` ab.
- **Dashboard-Notebook nicht Teil dieses Repos:** `open_annotated_dashboard()`
  selbst ist fertig (siehe "Dashboard-Integration" oben), aber das
  konsumierende Notebook (`SequencePreprocessing_v2.ipynb`) und `acia` liegen
  außerhalb dieses Repos, in den nicht von hier aus versionierten
  Geschwisterordnern `../01_SequencePreprocessing` und `../acia-core`.
