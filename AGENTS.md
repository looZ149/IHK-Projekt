<!-- bmad:context -->
<!-- Verified 2026-09-21 against bcbe9b4. Managed by bmad-project-context; edits inside this block are replaced on refresh. Keep anything you want preserved outside the markers. -->

## IHK-Projekt

KI-gestützte Bildverarbeitungsanwendung for automated analysis of bioprocess-engineering image data (phase-contrast microscopy) via YOLO-OBB detection (`chamber_region`, `channel_region`). Python + PyTorch (`ultralytics`). Stripped-down copy of a larger work repo (KIT Karlsruhe / Bioverfahrenstechnik), prepared for an IHK Abschlussprojekt. Spec and architecture rules (AD-n) live in `_bmad-output/` (`specs/spec-bioprocess-image-analysis/`, `planning-artifacts/architecture/`), which is gitignored and exists only locally.

## Policy

- `src/training/` is feature-frozen legacy remote-box CLI — don't build new features there; the maintained logic lives in `chamber_obb`. A user-approved cleanup (2026-10) deduplicated its verbatim helper copies: `normalize_to_8bit`/`rotated_box_to_corners` now come from `chamber_obb.preprocessing`/`chamber_obb.dataset`, the chip-image name regexes + `LOCK_FILE` are single-sourced in `train_lock.py`, and `verify_registration.py`'s dead `resolve_frame_index` import was repaired. It must not write into a dataset dir that has a `dataset.json`.

## Where things are

- `chamber_obb` library (`src/chamber_obb/`): the public surface is `analyze()` (`pipeline.py`, detection), `build_dataset()` (`dataset.py`) and `train()` (`training.py`) for continuing training on new data, plus `migrate_legacy_dataset()` (`locks.py`), the one-time admin step for the old dataset. `preprocessing`/`inference`/`postprocessing`, `classes.py`, `manifest.py`, `checkpoints.py`, `util.py` and the rest of `locks.py` are implementation detail — never import them directly.
- Training data lives outside the repo and the code-only pip package: `old_dataset` (shared, read-only YOLO dataset on the KIT machine), `new_data` (raw chip folders), `new_dataset` (`build_dataset()` output); paths are plain parameters (only the checkpoint also falls back to `CHAMBER_OBB_CHECKPOINT`, like in `analyze()`). Every dataset dir carries a `dataset.json` (class list, session table, ever-in-train lock set); a dir with images but none fails closed. Each `train()` run writes a unique run dir with a cumulative `manifest.json`, the lineage source for the leakage gate. User flow: "Training auf neuen Daten" in `docs/Anwenderdokumentation.md`.
- Dashboard integration (built, `src/chamber_obb/dashboard.py`): `open_annotated_dashboard()` runs `analyze()` per position of an acia sequence and opens `acia.notebook.SequenceDashboard` pre-populated with the detections — see "Dashboard-Integration" in `docs/Anwenderdokumentation.md`. `acia` is only ever imported inside `dashboard.py` and, lazily, in `build_dataset()` (optional `chamber_obb[dashboard]` extra) — `import chamber_obb`, `analyze()` and `train()` stay acia-free. The consuming notebooks are outside this repo, in `~/Downloads/image_analysis_V3` (`SequencePreprocessing.ipynb`, `Training.ipynb`, `installation.ipynb`); they depend on `acia`, whose full source is in the sibling folder `../acia-core`. Neither is tracked by this repo.
- Planning/analysis (marimo notebooks, German): `Analysephase/` (model research), `Planungsphase/` (architecture, data structure, training/validation concepts, tooling). Not frozen — update them to stay accurate as the project evolves.

## Running and verifying

- `uv sync` installs `chamber_obb` (workspace member) + `ultralytics`/`torch`, plus `pytest` and `acia[nd2,widget]` as dev-dependencies (resolved from the local `../acia-core` checkout via `tool.uv.sources`). `uv run pytest` runs the suite (all pass; the training API is tested with a fake YOLO and a fake acquisition opener — no GPU/ND2/acia); no CI configured.
- `acia` is a dev-dependency here only (so `uv run pytest` exercises the real dashboard integration) — never a main dependency of `chamber_obb` itself. Outside this uv workspace (e.g. a standalone notebook environment on another machine), install it separately: see "Dashboard-Integration" in `docs/Anwenderdokumentation.md`, or `installation.ipynb` in `~/Downloads/image_analysis_V3` for the exact command.
- After changing dependencies run plain `uv lock` (needs network); `uv lock --offline` fails resolving other platforms' packages.
- No trained checkpoint ships in this repo — `analyze()` raises `ModelLoadError` until `CHAMBER_OBB_CHECKPOINT` (env var) or `checkpoint_path` is set. Promotion is a manual, human-executed step — never automatic — full checklist in `RELEASE.md`.
- `src/training/`: does not run on this machine. Hardcoded paths in `config.py`/`data.yaml` (`/home/ubuntu/work/workspace/Projekt/...`) point at the remote KIT box on purpose — never "fix" them to local paths. Its `train_lock.py` (cwd-relative `train_locked_chips.txt`) keeps chip folders ever used in training in the train split; the library's equivalent is `dataset.json` — don't mix the two.

## Conventions that differ from defaults

- Comments and docstrings are written in German — match the existing language when editing `chamber_obb/`, `training/`, or the notebooks.

## Known pitfalls

- All exceptions `analyze()` can raise inherit from `chamber_obb.errors.ChamberOBBError` — `except ChamberOBBError:` is the guaranteed-complete catch-all.
- The old dataset (the current model was trained on ALL of it) has no independent val/test: `migrate_legacy_dataset()` marks it `all_sessions_trained`, `train()` uses all its images as training data, and val/test must come from held-out sessions in the new dataset; `train()` also evaluates the base checkpoint on the same val/test as a baseline (promotion bar in `RELEASE.md`).
- ROIs in a chip's `selection.json` belong to their `anchor_frame` (the dashboard opens on the middle frame), and `registration_transforms.json` uses acia's convention (reference→frame, rotation about the frame centre, `theta` in degrees). The legacy `src/training/build_trainingsdata.py` assumes frame 0 and radians/origin pivot, so its `--use-registration` labels are wrong for rotated frames — never copy its geometry; use `chamber_obb.dataset` (tested against acia).
- Real ultralytics writes `labels/*.cache` next to the labels of every dataset it reads, including the read-only `old_dataset` — expected, not a bug in `train()`.
- Training on new data was only verified with stubs and one tiny CPU run — no real `train()` run on the KIT box (GPU, real chips) has happened yet; `build_dataset()` needs `acia` (`chamber_obb[dashboard]`) for real ND2 sources.
- Training data volume may be insufficient for the model to generalize well — treat accuracy/robustness claims cautiously until validated on held-out data.
- Suspected pitfalls noticed while planning: flag them for a `bmad-project-context` `record` pass rather than letting them go unrecorded.

<!-- /bmad:context -->
