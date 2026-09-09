<!-- bmad:context -->
<!-- Verified 2026-09-09 against a131812. Managed by bmad-project-context; edits inside this block are replaced on refresh. Keep anything you want preserved outside the markers. -->

## IHK-Projekt

KI-gestützte Bildverarbeitungsanwendung for automated analysis of bioprocess-engineering image data (phase-contrast microscopy) via YOLO-OBB detection (`chamber_region`, `channel_region`). Python + PyTorch (`ultralytics`). Stripped-down copy of a larger work repo (KIT Karlsruhe / Bioverfahrenstechnik), prepared for an IHK Abschlussprojekt.

## Policy

- Don't modify `training/` — legacy dev tooling, out of scope for current work.

## Where things are

- `chamber_obb` library (`src/chamber_obb/`): the only public entry point is `analyze()` in `pipeline.py`. `preprocessing`/`inference`/`postprocessing` are implementation detail — never import them directly.
- Planned dashboard integration (not yet built): `analyze()` results are meant to feed an existing private dashboard Jupyter notebook as a background process — see "Dashboard-Integration" in `docs/Anwenderdokumentation.md`. The notebooks live in the sibling folder `../01_SequencePreprocessing`, which depends on the `acia` library — full `acia` source is in the sibling folder `../acia-core`.
- Planning/analysis (marimo notebooks, German): `Analysephase/` (model research), `Planungsphase/` (architecture, data structure, training/validation concepts, tooling). Not frozen — update them to stay accurate as the project evolves.

## Running and verifying

- `uv sync` installs `chamber_obb` (workspace member) + `ultralytics`/`torch`. `uv run pytest` runs the 38-test suite (all pass); no CI configured.
- `acia` is intentionally not declared in `pyproject.toml` (see its description) — install it separately via `pip` from the sibling `../acia-core` checkout when working with acia-dependent code.
- No trained checkpoint ships in this repo — `analyze()` raises `ModelLoadError` until `CHAMBER_OBB_CHECKPOINT` (env var) or `checkpoint_path` is set. Promotion is a manual, human-executed step — never automatic — full checklist in `RELEASE.md`.
- `training/`: does not run on this machine. Hardcoded paths in `config.py`/`data.yaml` (`/home/ubuntu/work/workspace/Projekt/...`) point at the remote KIT box on purpose — never "fix" them to local paths. `train_lock.py` enforces that chip folders ever used in training stay in the train split forever, to avoid val/test leakage.

## Conventions that differ from defaults

- Comments and docstrings are written in German — match the existing language when editing `chamber_obb/`, `training/`, or the notebooks.

## Known pitfalls

- All exceptions `analyze()` can raise inherit from `chamber_obb.errors.ChamberOBBError` — `except ChamberOBBError:` is the guaranteed-complete catch-all.
- Training data volume may be insufficient for the model to generalize well — treat accuracy/robustness claims cautiously until validated on held-out data.
- Suspected pitfalls noticed while planning: flag them for a `bmad-project-context` `record` pass rather than letting them go unrecorded.

<!-- /bmad:context -->
