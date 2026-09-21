# Release checklist

## Promoting a new checkpoint (AD-15)

Training writes each run to its own directory — nothing here happens
automatically beyond that:

- the legacy CLI (`training/model.py::train()`) writes to `runs/obb/train*/weights/best.pt`;
- `chamber_obb.train()` (CAP-7) writes to a unique `<runs_dir>/<run_name>-<timestamp>/`
  with the weights in `fit/weights/best.pt`, the run's `manifest.json` (base
  checkpoint, sessions trained on, val/test sessions, dataset hashes,
  hyperparameters, versions, metrics) and `eval-val/`, `eval-test/`.

"The current checkpoint" — the one `chamber_obb.analyze()` actually loads — is
decided by exactly one thing: the `CHAMBER_OBB_CHECKPOINT` environment
variable (or an explicit `checkpoint_path` argument passed to `analyze()`).
Neither is ever updated automatically.

**After a training run is accepted:**

1. Check the promotion bar. The current checkpoint was trained on the whole old
   dataset, so the old dataset has no independent val/test data. The bar is therefore
   comparative: the new checkpoint's test metrics (`TrainResult.test`) must be **not
   worse than the base checkpoint's** (`TrainResult.baseline_test`, measured by
   `train()` on the same held-out new sessions, which neither model trained on),
   and the Projektverantwortlicher signs off. No absolute threshold exists yet; a
   few held-out sessions give noisy numbers, so treat them as evidence, not proof.
   (Legacy CLI runs: `training/model.py::test()` has no such baseline.)
2. Manually set `CHAMBER_OBB_CHECKPOINT` to the accepted run's weights path,
   e.g.:
   ```
   export CHAMBER_OBB_CHECKPOINT=/path/to/runs/obb/train7/weights/best.pt
   ```
3. **Copy the manifest with the weights.** If `best.pt` is copied to another
   location or machine, copy the run's `manifest.json` next to it as
   `best.manifest.json` (or keep it as `manifest.json` in the weights folder's
   parent). The next `train()` and `build_dataset()` read the sessions this
   checkpoint has already seen from it (leakage gate); without it the lineage is
   unknown and `train()` refuses unless `old_dataset` is given or
   `acknowledge_unknown_lineage=True`.
4. Restart/reload whatever process calls `chamber_obb.analyze()` so it picks
   up the new environment variable.

This is a deliberate, human-executed step — `chamber_obb.train()` never sets
`CHAMBER_OBB_CHECKPOINT` — not an implicit part of running
`train()`, and not triggered by a new run directory simply appearing. A
symlink/pointer-file mechanism (e.g. `runs/obb/current -> train7/weights/best.pt`)
is a reasonable future upgrade once this manual step proves tedious, but it
is not implemented yet — this checklist is the floor.

## Promoting new data into the shared dataset

Manual for now (a `promote` function is deferred): someone with write access
copies the `new_dataset` folder's images, labels and `dataset.json` into the
shared `old_dataset` — merging the `dataset.json` session tables and lock sets,
never dropping lock entries — and re-runs the disjointness check described in
`docs/Anwenderdokumentation.md` ("Training auf neuen Daten").
