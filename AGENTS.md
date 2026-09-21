<!-- bmad:context -->
<!-- Verified 2026-09-09 against a131812. Managed by bmad-project-context; edits inside this block are replaced on refresh. Keep anything you want preserved outside the markers. -->

## IHK-Projekt

KI-gestützte Bildverarbeitungsanwendung for automated analysis of bioprocess-engineering image data (phase-contrast microscopy) via YOLO-OBB detection (`chamber_region`, `channel_region`). Python + PyTorch (`ultralytics`). Stripped-down copy of a larger work repo (KIT Karlsruhe / Bioverfahrenstechnik), prepared for an IHK Abschlussprojekt.

## Policy

- Don't modify `training/` — legacy remote-box CLI, out of scope; the ported logic lives in `chamber_obb` (`training/` must not write into a dataset dir that has a `dataset.json`).

## Where things are

- `chamber_obb` library (`src/chamber_obb/`): the public surface is `analyze()` (`pipeline.py`, detection) plus `build_dataset()` (`dataset.py`), `train()` (`training.py`) and the admin helper `migrate_legacy_dataset()` (`locks.py`) for continuing training on new data (CAP-7, AD-2/AD-20..24). `preprocessing`/`inference`/`postprocessing` and the session/lock logic (`locks.py`, `classes.py`, `manifest.py`) are implementation detail — never import them directly (exception: `chamber_obb.migrate_legacy_dataset()`, the public one-time admin step for the old dataset, see `docs/Anwenderdokumentation.md`).
- Training data lives outside the repo and the pip package (code only): `old_dataset` = shared read-only YOLO dataset on the KIT machine, `new_data` = raw chip folders, `new_dataset` = `build_dataset()` output; paths are parameters with `CHAMBER_OBB_*` env fallbacks. Each dataset dir carries its own `dataset.json` (class list, session table, ever-in-train lock set) — the lock registry no longer lives in the cwd/repo root; a dataset with images but no `dataset.json` fails closed. Each `train()` run writes a unique run dir with a cumulative `manifest.json` (the lineage source for the leakage gate).
- Dashboard integration (built, `src/chamber_obb/dashboard.py`): `open_annotated_dashboard()` runs `analyze()` per position of an acia sequence and opens `acia.notebook.SequenceDashboard` pre-populated with the detections — see "Dashboard-Integration" in `docs/Anwenderdokumentation.md`. `acia` is only ever imported inside `dashboard.py` (an optional `chamber_obb[dashboard]` extra) — the core `analyze()` path stays acia-free. The consuming notebook lives in the sibling folder `../01_SequencePreprocessing` (`SequencePreprocessing_v2.ipynb`), which depends on the `acia` library — full `acia` source is in the sibling folder `../acia-core`. Neither sibling folder is a git repo tracked by this one.
- Planning/analysis (marimo notebooks, German): `Analysephase/` (model research), `Planungsphase/` (architecture, data structure, training/validation concepts, tooling). Not frozen — update them to stay accurate as the project evolves.

## Running and verifying

- `uv sync` installs `chamber_obb` (workspace member) + `ultralytics`/`torch`, plus `pytest` and `acia[nd2,widget]` as dev-dependencies (resolved from the local `../acia-core` checkout via `tool.uv.sources`). `uv run pytest` runs the 123-test suite (all pass; the training API is tested with a fake YOLO and a fake acquisition opener, no GPU/ND2/acia); no CI configured.
- `acia` is a dev-dependency here only (so `uv run pytest` exercises the real dashboard integration) — never a main dependency of `chamber_obb` itself, so `import chamber_obb` works without it. Outside this uv workspace (e.g. a standalone notebook environment on another machine), install it separately: see "Dashboard-Integration" in `docs/Anwenderdokumentation.md`, or `../01_SequencePreprocessing/00_installation.ipynb` for the exact command.
- No trained checkpoint ships in this repo — `analyze()` raises `ModelLoadError` until `CHAMBER_OBB_CHECKPOINT` (env var) or `checkpoint_path` is set. Promotion is a manual, human-executed step — never automatic — full checklist in `RELEASE.md`.
- `training/`: does not run on this machine. Hardcoded paths in `config.py`/`data.yaml` (`/home/ubuntu/work/workspace/Projekt/...`) point at the remote KIT box on purpose — never "fix" them to local paths. `train_lock.py` enforces that chip folders ever used in training stay in the train split forever, to avoid val/test leakage.

## Conventions that differ from defaults

- Comments and docstrings are written in German — match the existing language when editing `chamber_obb/`, `training/`, or the notebooks.

## Known pitfalls

- All exceptions `analyze()` can raise inherit from `chamber_obb.errors.ChamberOBBError` — `except ChamberOBBError:` is the guaranteed-complete catch-all.
- The old dataset (the current model was trained on ALL of it) has no independent val/test: `migrate_legacy_dataset()` marks it `all_sessions_trained`, `train()` uses all its images as training data, and val/test must come from held-out sessions in the new dataset; `train()` also evaluates the base checkpoint on the same val/test as a baseline (promotion bar in `RELEASE.md`).
- Training on new data was only verified with stubs — no real `train()` run on the KIT box (GPU, real chips) has happened yet; `build_dataset()` needs `acia` (`chamber_obb[dashboard]`) for real ND2 sources.
- Training data volume may be insufficient for the model to generalize well — treat accuracy/robustness claims cautiously until validated on held-out data.
- Suspected pitfalls noticed while planning: flag them for a `bmad-project-context` `record` pass rather than letting them go unrecorded.

<!-- /bmad:context -->
