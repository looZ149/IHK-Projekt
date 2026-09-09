# Release checklist

## Promoting a new checkpoint (AD-15)

Training (`training/model.py::train()`) writes each run to its own timestamped
directory (`runs/obb/train*/weights/best.pt`) — nothing here happens
automatically beyond that.

"The current checkpoint" — the one `chamber_obb.analyze()` actually loads — is
decided by exactly one thing: the `CHAMBER_OBB_CHECKPOINT` environment
variable (or an explicit `checkpoint_path` argument passed to `analyze()`).
Neither is ever updated automatically.

**After a training run is accepted:**

1. Confirm the new run's metrics (via `training/model.py::test()`) meet the
   bar for promotion.
2. Manually set `CHAMBER_OBB_CHECKPOINT` to the accepted run's weights path,
   e.g.:
   ```
   export CHAMBER_OBB_CHECKPOINT=/path/to/runs/obb/train7/weights/best.pt
   ```
3. Restart/reload whatever process calls `chamber_obb.analyze()` so it picks
   up the new environment variable.

This is a deliberate, human-executed step — not an implicit part of running
`train()`, and not triggered by a new run directory simply appearing. A
symlink/pointer-file mechanism (e.g. `runs/obb/current -> train7/weights/best.pt`)
is a reasonable future upgrade once this manual step proves tedious, but it
is not implemented yet — this checklist is the floor.
