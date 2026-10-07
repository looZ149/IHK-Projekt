"""``train()``: Fine-Tuning ab einem Basis-Checkpoint auf alten und/oder neuen Daten (CAP-7).

Schreibt ausschliesslich in ein eigenes, eindeutiges Run-Verzeichnis (AD-20,
AD-24): ``old_dataset`` und ``base_checkpoint`` bleiben unberuehrt, der
Inferenz-Modell-Cache aus ``inference.py`` wird nie benutzt (frisches
``YOLO(base_checkpoint)`` je Aufruf). Vor dem Fitten laufen drei Gates:

1. Klassenlisten-Guard (AD-21),
2. Leakage-Gate ueber die Session-Splits und die Trainings-Historie (AD-22),
3. Konsistenz jedes Datensatzes mit seinem Descriptor (AD-23).

Ein Checkpoint wird hier nie "befoerdert" (AD-15): weder wird
``CHAMBER_OBB_CHECKPOINT`` gesetzt noch sonst etwas umgeleitet.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Literal

import yaml
from ultralytics import YOLO

from chamber_obb.checkpoints import CHECKPOINT_ENV_VAR, resolve_checkpoint
from chamber_obb.classes import class_list
from chamber_obb.errors import (
    ChamberOBBError,
    ClassListMismatchError,
    DatasetDescriptorError,
    EmptySplitError,
    InvalidParameterError,
    LeakageError,
    ModelLoadError,
    TrainingError,
)
from chamber_obb.locks import (
    SPLITS,
    Descriptor,
    check_consistency,
    descriptor_hash,
    effective_lock_set,
    images_by_role,
    load_descriptor,
)
from chamber_obb.manifest import MANIFEST_FILENAME, MANIFEST_SCHEMA_VERSION, find_manifest, load_manifest, write_manifest
from chamber_obb.util import resolve_path

logger = logging.getLogger(__name__)

# Defaults aus training/config.py (legacy) -- pro Aufruf ueber ``hyperparameters`` ueberschreibbar.
DEFAULT_HYPERPARAMETERS: dict[str, object] = {
    "epochs": 150,
    "batch": 32,
    "imgsz": 640,
    "multi_scale": 0.5,
    "hsv_v": 0.6,
    "degrees": 180,
    "translate": 0.3,
    "scale": 0.3,
    "flipud": 0.5,
    "fliplr": 0.5,
    "mosaic": 1,
    "erasing": 0.5,
}
_RESERVED_KEYS = {"data", "project", "name", "exist_ok", "resume", "task", "model", "split"}


@dataclass(frozen=True)
class Metrics:
    map50: float
    map50_95: float
    precision: float
    recall: float


@dataclass(frozen=True)
class TrainResult:
    """Ergebnis von ``train()``. ``test`` ist ``None``, wenn kein Test-Split vorhanden war."""

    run_dir: Path
    best_weights: Path
    manifest_path: Path
    val: Metrics
    test: Metrics | None
    warnings: list[str] = field(default_factory=list)
    # Basis-Checkpoint auf DENSELBEN Val/Test-Sessions (None bei Stock-Start/reinit_head/abgeschaltet):
    # der Vergleichswert fuer die Freigabe-Entscheidung (RELEASE.md).
    baseline_val: Metrics | None = None
    baseline_test: Metrics | None = None


@dataclass
class _Dataset:
    role: str
    root: Path
    descriptor: Descriptor
    sessions: dict[str, set[str]]
    images: dict[str, list[Path]]


def _load_dataset(role: str, root: Path) -> _Dataset:
    """Laedt Descriptor + Bilder eines Datensatzes; Widersprueche sind Fehler (AD-23)."""
    if not root.is_dir():
        raise InvalidParameterError(f"{role}-Datensatz existiert nicht: {root}")
    descriptor = load_descriptor(root)
    if descriptor is None:
        raise DatasetDescriptorError(f"{role}-Datensatz {root} ist leer (keine Bilder, keine dataset.json)")
    problems = check_consistency(root, descriptor)
    if problems:
        raise DatasetDescriptorError(f"{root} ist inkonsistent: " + "; ".join(problems[:5]))
    if descriptor["class_list"] != class_list():
        raise ClassListMismatchError(
            f"{role}-Datensatz {root}: Klassenliste {descriptor['class_list']} != Paket-Klassenliste {class_list()} "
            f"(Datensatz mit der aktuellen Klassenliste neu bauen)"
        )
    sessions: dict[str, set[str]] = {s: set() for s in SPLITS}
    for name, entry in descriptor["sessions"].items():
        sessions[entry["split"]].add(name)
    images = images_by_role(root, descriptor)  # Alt-Datensatz 'alles trainiert': nur Train-Bilder
    return _Dataset(role, root, descriptor, sessions, images)


def _model_class_list(model: YOLO) -> list[str]:
    names = model.names
    return [names[i] for i in sorted(names)]


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for dist in ("chamber_obb", "ultralytics", "torch"):
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            versions[dist] = None
    return versions


def _new_run_dir(runs_dir: Path, run_name: str) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for attempt in range(1, 1000):
        suffix = "" if attempt == 1 else f"-{attempt}"
        candidate = runs_dir / f"{run_name}-{stamp}{suffix}"
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise TrainingError(f"kein freies Run-Verzeichnis unter {runs_dir}")


def _metrics_of(result: object) -> Metrics:
    box = result.box  # type: ignore[attr-defined]
    return Metrics(
        map50=float(box.map50), map50_95=float(box.map), precision=float(box.mp), recall=float(box.mr)
    )


def train(
    new_dataset: str | Path | None = None,
    old_dataset: str | Path | None = None,
    base_checkpoint: str | Path | None = None,
    *,
    mode: Literal["new", "old", "both"] | None = None,
    runs_dir: str | Path | None = None,
    run_name: str = "chamber_obb",
    hyperparameters: dict[str, object] | None = None,
    reinit_head: bool = False,
    acknowledge_unknown_lineage: bool = False,
    evaluate_baseline: bool = True,
    device: str | int | None = None,
    seed: int = 42,
) -> TrainResult:
    """Fine-tunt ``base_checkpoint`` auf ``old_dataset`` und/oder ``new_dataset`` (CAP-7).

    ``mode`` waehlt die Trainingsdaten: ``"new"``, ``"old"`` oder ``"both"``;
    ohne Angabe gilt ``"both"``, wenn beide Pfade vorliegen, sonst der eine.
    Die Pfade werden explizit uebergeben (AD-8); nur ``base_checkpoint`` faellt wie bei
    ``analyze()`` auf ``CHAMBER_OBB_CHECKPOINT`` zurueck. Run-Verzeichnisse liegen unter
    ``runs_dir`` (sonst ``./runs/chamber_obb``).

    Gates vor dem Fitten -- jede Verletzung wirft, bevor GPU-Zeit verbraucht wird:
    Klassenliste (AD-21; ``reinit_head=True`` erlaubt einen Basis-Checkpoint mit
    anderen Klassen bzw. den Stock-Start ``yolo26n-obb.pt``), Val/Test-Sessions
    disjunkt zu Train, Val nicht leer, kein Val/Test-Session in der
    Trainings-Historie des Basis-Checkpoints (AD-22). Ist die Historie unbekannt
    (kein Manifest neben dem Checkpoint, kein ``old_dataset``), wirft ``train()``,
    sofern nicht ``acknowledge_unknown_lineage=True``.

    Ein Alt-Datensatz mit ``all_sessions_trained`` (das aktuelle Modell wurde auf allen
    Sessions trainiert) liefert nur Trainingsdaten -- Val/Test kommen dann ausschliesslich
    aus ``new_dataset``, und ohne Val-Session dort wirft ``train()`` ``EmptySplitError``.
    ``evaluate_baseline`` (Standard an) bewertet zusaetzlich den Basis-Checkpoint VOR dem Training
    auf denselben Val/Test-Sessions (``baseline_val``/``baseline_test``); bei Stock-Start oder
    ``reinit_head`` entfaellt das.

    Raises:
        InvalidParameterError, ModelLoadError, DatasetDescriptorError,
        ClassListMismatchError, LeakageError, EmptySplitError, TrainingError.
    """
    new_root = resolve_path(new_dataset, "new_dataset", required=False)
    old_root = resolve_path(old_dataset, "old_dataset", required=False)
    if mode not in (None, "new", "old", "both"):
        raise InvalidParameterError(f"mode muss 'new', 'old' oder 'both' sein, nicht {mode!r}")
    chosen = mode or ("both" if new_root and old_root else "new" if new_root else "old" if old_root else None)
    if chosen is None:
        raise InvalidParameterError("weder new_dataset noch old_dataset angegeben")
    if chosen in ("new", "both") and new_root is None:
        raise InvalidParameterError(f"mode={chosen!r} braucht new_dataset")
    if chosen in ("old", "both") and old_root is None:
        raise InvalidParameterError(f"mode={chosen!r} braucht old_dataset")

    overrides = dict(hyperparameters or {})
    reserved = _RESERVED_KEYS & overrides.keys()
    if reserved:
        raise InvalidParameterError(f"hyperparameters darf {sorted(reserved)} nicht setzen (gehoert train())")
    params = {**DEFAULT_HYPERPARAMETERS, **overrides}

    # --- Basis-Checkpoint ---------------------------------------------------
    raw_base = resolve_checkpoint(base_checkpoint)
    if not raw_base:
        raise ModelLoadError(f"kein base_checkpoint angegeben und {CHECKPOINT_ENV_VAR} nicht gesetzt")
    base_path = Path(raw_base).expanduser()
    is_stock = False
    if base_path.is_file():
        base_ref = str(base_path.resolve())
    elif reinit_head and base_path.name == str(raw_base) and str(raw_base).endswith(".pt"):
        is_stock, base_ref = True, str(raw_base)  # z.B. "yolo26n-obb.pt": ultralytics laedt die Stock-Gewichte
    else:
        raise ModelLoadError(f"base_checkpoint existiert nicht: {raw_base}")

    # --- Datensaetze laden (AD-23) ------------------------------------------
    included: list[_Dataset] = []
    if chosen in ("old", "both"):
        included.append(_load_dataset("old", old_root))  # type: ignore[arg-type]
    if chosen in ("new", "both"):
        included.append(_load_dataset("new", new_root))  # type: ignore[arg-type]
    if len(included) == 2 and (
        included[0].root == included[1].root
        or included[0].root in included[1].root.parents
        or included[1].root in included[0].root.parents
    ):
        raise InvalidParameterError("old_dataset und new_dataset duerfen weder gleich sein noch ineinander liegen")

    # --- Gate 1: Klassenliste (AD-21) ---------------------------------------
    try:
        model = YOLO(base_ref)  # frisch je Aufruf, nie der Inferenz-Cache (AD-20)
    except Exception as exc:
        raise ModelLoadError(f"base_checkpoint {raw_base} liess sich nicht laden: {exc}") from exc
    model_classes = _model_class_list(model)
    if model_classes != class_list():
        if not reinit_head:
            raise ClassListMismatchError(
                f"Klassen des Basis-Checkpoints {model_classes} != Paket-Klassenliste {class_list()}. "
                f"Fuer einen Stock-Start oder eine neue Klasse reinit_head=True setzen."
            )
        logger.warning("reinit_head=True: Klassen des Basis-Checkpoints %s werden durch %s ersetzt", model_classes, class_list())

    # --- Gate 2: Leakage (AD-22) --------------------------------------------
    train_sessions = set().union(*(d.sessions["train"] for d in included))
    val_sessions = set().union(*(d.sessions["val"] for d in included))
    test_sessions = set().union(*(d.sessions["test"] for d in included))
    for label, overlap in (
        ("Val und Train", val_sessions & train_sessions),
        ("Test und Train", test_sessions & train_sessions),
        ("Val und Test", val_sessions & test_sessions),
    ):
        if overlap:
            raise LeakageError(f"Sessions liegen gleichzeitig in {label}: {sorted(overlap)}")
    n_images = {s: sum(len(d.images[s]) for d in included) for s in SPLITS}
    if n_images["train"] == 0:
        raise EmptySplitError("der Train-Split ist leer")
    if n_images["val"] == 0:
        raise EmptySplitError(
            "der Val-Split ist leer -- kein Fallback auf Train. Ein Alt-Datensatz ohne unabhaengiges "
            "Val/Test (all_sessions_trained) liefert keinen; im new_dataset muss mindestens eine "
            "Val-Session liegen (beim Bauen val_chips/test_chips angeben)"
        )

    manifest_path = None if is_stock else find_manifest(base_ref)
    parent = load_manifest(manifest_path) if manifest_path else None
    lineage: set[str] = set(parent["trained_sessions"]) if parent else set()
    old_locked: set[str] = set()
    if old_root is not None:
        old_view = next((d for d in included if d.role == "old"), None) or _load_dataset("old", old_root)
        old_locked = effective_lock_set(old_view.root, old_view.descriptor)
    lineage_known = is_stock or parent is not None or old_root is not None
    if not lineage_known and not acknowledge_unknown_lineage:
        raise LeakageError(
            "Trainings-Historie des Basis-Checkpoints unbekannt (kein manifest.json neben den Gewichten, "
            "kein old_dataset). acknowledge_unknown_lineage=True setzen, wenn das bewusst ist."
        )
    if parent is None:
        lineage |= old_locked  # Legacy-Checkpoint ohne Manifest: Historie = Sperrliste des Alt-Datensatzes
    leaked = (val_sessions | test_sessions) & lineage
    if leaked:
        raise LeakageError(
            f"Val/Test-Sessions wurden vom Basis-Checkpoint (oder einem Vorgaenger) bereits trainiert: {sorted(leaked)}"
        )

    warnings: list[str] = []
    if n_images["test"] == 0:
        warnings.append("kein Test-Split vorhanden -- es gibt keine Test-Metriken")
        logger.warning(warnings[-1])

    # --- Run-Verzeichnis, Bildlisten, data.yaml (AD-24) ----------------------
    runs_root = resolve_path(runs_dir, "runs_dir", required=False) or (Path.cwd() / "runs" / "chamber_obb").resolve()
    run_dir = _new_run_dir(runs_root, run_name)
    lists_dir = run_dir / "lists"
    lists_dir.mkdir()
    data_cfg: dict[str, object] = {"path": str(run_dir), "names": dict(enumerate(class_list()))}
    for split in SPLITS:
        images = [str(p.resolve()) for d in included for p in d.images[split]]
        if images:
            (lists_dir / f"{split}.txt").write_text("\n".join(images) + "\n", encoding="utf-8")
            data_cfg[split] = f"lists/{split}.txt"
    data_yaml = run_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    extra = {"device": device} if device is not None else {}

    # --- Baseline: Basis-Checkpoint auf denselben Val/Test-Sessions (vor dem Fitten) ---------
    baseline: dict[str, Metrics | None] = {"val": None, "test": None}
    if evaluate_baseline and not is_stock and not reinit_head:
        try:
            base_eval = YOLO(base_ref)
            for split in ("val", "test"):
                if n_images[split] == 0:
                    continue
                baseline[split] = _metrics_of(base_eval.val(
                    data=str(data_yaml), split=split, project=str(run_dir), name=f"baseline-{split}",
                    exist_ok=False, **extra,
                ))
        except ChamberOBBError:
            raise
        except Exception as exc:
            raise TrainingError(f"Baseline-Evaluation des Basis-Checkpoints fehlgeschlagen ({run_dir}): {exc}") from exc

    # --- Fitten -------------------------------------------------------------
    logger.info("train: %s auf %s -> %s", raw_base, [d.role for d in included], run_dir)
    try:
        model.train(
            data=str(data_yaml), project=str(run_dir), name="fit", exist_ok=False, resume=False,
            task="obb", seed=seed, **params, **extra,
        )
        trainer = model.trainer
        save_dir = Path(trainer.save_dir)
        best = Path(getattr(trainer, "best", save_dir / "weights" / "best.pt"))
    except ChamberOBBError:
        raise
    except Exception as exc:
        raise TrainingError(f"Training fehlgeschlagen ({run_dir}): {exc}") from exc
    if not best.is_file():
        raise TrainingError(f"Training lieferte keine best.pt unter {save_dir}")

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "best_weights": str(best),
        "base_checkpoint": base_ref,
        "base_is_stock": is_stock,
        "parent_manifest": str(manifest_path) if manifest_path else None,
        "mode": chosen,
        "reinit_head": reinit_head,
        "acknowledge_unknown_lineage": acknowledge_unknown_lineage,
        "class_list": class_list(),
        "trained_sessions": sorted(lineage | train_sessions),
        "train_sessions": sorted(train_sessions),
        "val_sessions": sorted(val_sessions),
        "test_sessions": sorted(test_sessions),
        "datasets": [
            {"role": d.role, "path": str(d.root), "descriptor_sha256": descriptor_hash(d.root)} for d in included
        ],
        "hyperparameters": {**params, "seed": seed},
        "package_versions": _package_versions(),
        "metrics": None,
        "baseline_metrics": {s: (vars(m) if m is not None else None) for s, m in baseline.items()},
    }
    manifest_file = run_dir / MANIFEST_FILENAME
    write_manifest(manifest_file, manifest)

    # --- Evaluieren ---------------------------------------------------------
    results: dict[str, Metrics | None] = {"val": None, "test": None}
    try:
        evaluator = YOLO(str(best))
        for split in ("val", "test"):
            if n_images[split] == 0:
                continue
            outcome = evaluator.val(
                data=str(data_yaml), split=split, project=str(run_dir), name=f"eval-{split}",
                exist_ok=False, **extra,
            )
            results[split] = _metrics_of(outcome)
    except ChamberOBBError:
        raise
    except Exception as exc:
        raise TrainingError(f"Evaluation fehlgeschlagen ({run_dir}): {exc}") from exc

    manifest["metrics"] = {
        s: (vars(m) if m is not None else None) for s, m in results.items()
    }
    write_manifest(manifest_file, manifest)
    assert results["val"] is not None
    return TrainResult(
        run_dir=run_dir, best_weights=best, manifest_path=manifest_file,
        val=results["val"], test=results["test"], warnings=warnings,
        baseline_val=baseline["val"], baseline_test=baseline["test"],
    )
