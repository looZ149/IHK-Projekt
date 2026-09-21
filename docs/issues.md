# Upstream issues to file

Bugs found in dependencies while integrating `chamber_obb` that are out of this
repo's authority to fix — tracked here until filed as an issue in the owning
repository.

---

## acia-core: `SequenceDashboard.resume()` sends selections in the wrong shape, crashing/hiding the editor

**Repo:** `gitlab.kit.edu/kit/blt/mbvt/research/image-analysis/acia` (local
read-only reference clone used to trace this: `~/Documents/acia-core`, at
`acia==0.4.0` / commit `49fb47b` "Bump version: 0.3.2 → 0.4.0")

**Found:** 2026-09-15, while debugging `chamber_obb.dashboard.open_annotated_dashboard()`
against a real `.nd2` acquisition. `dash.selections` (Python side) correctly
listed all detected ROIs with sane `center`/`size`/`angle` values, but the
editor rendered nothing beyond a single small dot per position, and clicking
a selection in the list crashed the widget with:

```
Uncaught TypeError: Cannot read properties of undefined (reading 'toFixed')
    at renderSpec (...:258:101)
    at frameImg.onload (...:158:56)
```

### Root cause

Two incompatible shapes exist for a "selection" and nothing converts between
them at load time:

- **On-disk / Python-native shape** — what `RoiSelection.roi.to_dict()`
  produces (`acia/base.py:1104-1114`): nested —
  `{"center": [x, y], "size": [w, h], "angle": ...}`.
- **JS runtime shape** — what the widget actually renders from. Every
  interactively-drawn ROI is created flat, e.g. in `addRoi()`
  (`acia/notebook.py:1531`): `{id, position, x, y, w, h, angle, label, ci,
  anchor_frame}` — no nesting.

`SequenceDashboard.resume()` (`acia/notebook.py:3139-3148`) builds the
**nested, on-disk shape** and assigns it directly to `dash.selections`:

```python
restored.append({
    "id": i + 1, "position": sel.position, "label": sel.label,
    "ci": ci, "anchor_frame": sel.anchor_frame,
    "roi": sel.roi.to_dict(),   # nested: {center, size, angle}
})
dash.selections = restored
```

The widget's JS startup code (`acia/notebook.py:1266`) shallow-copies
whatever it receives with **no flattening step**:

```js
let selections = (model.get("selections") || []).map((x) => Object.assign({}, x));
```

So every selection loaded via `.resume()` ends up with a `roi: {...}` key and
no top-level `x`/`y`/`w`/`h`/`angle` at all. That explains both symptoms:

- **Invisible/dot rendering** — `renderEditor()` builds the box's CSS from
  `sel.x`/`sel.w`/`sel.angle` (`acia/notebook.py:1399-1401`), all `undefined`
  → the box collapses to nothing.
- **Crash on click** — `renderSpec()` does `sel.angle.toFixed(1)`
  (`acia/notebook.py:1423`) → `undefined.toFixed` → the exact error above.

### Impact

Not specific to `chamber_obb`. This breaks **any** `.resume()` call, which is
also the documented "save a session, continue it later" workflow for a human
curator (`dash2 = SequenceDashboard.resume(output_dir)`, per
`SequencePreprocessing_v2.ipynb`'s own markdown). It just hadn't been
exercised end-to-end before — a curator normally starts from an empty
dashboard and only ever creates ROIs via `addRoi()`/dragging, which is
self-consistent (flat in, flat out) and never hits this path.

### Proposed fix

Build the flat shape in `resume()` instead of the nested one:

```python
restored.append({
    "id": i + 1, "position": sel.position, "label": sel.label,
    "ci": ci, "anchor_frame": sel.anchor_frame,
    "x": sel.roi.center[0], "y": sel.roi.center[1],
    "w": sel.roi.size[0], "h": sel.roi.size[1],
    "angle": sel.roi.angle,
})
```

### Reproduction

1. Build a `SelectionManifest` with at least one `RoiSelection` (e.g. via
   `chamber_obb.dashboard.open_annotated_dashboard()`, or by hand).
2. `SequenceDashboard.resume(manifest, source=seqfile)`.
3. Click into any position that has a detection/selection — no box is drawn,
   only a small dot marker.
4. Click the selection's row in the selections list — widget throws
   `Cannot read properties of undefined (reading 'toFixed')` in
   `renderSpec`, editor becomes unresponsive.

### Status: worked around downstream, not fixed upstream

**Still open in `acia-core`** — the fix above hasn't been applied there, and
this issue hasn't been filed on GitLab yet. A human curator's own
save-then-`.resume()` workflow is therefore still broken today.

**Worked around in `chamber_obb`**, since `open_annotated_dashboard()`
controls its own `dash` before ever returning it to the caller (and the
widget's JS only runs once `dash` is actually displayed, not when
`.resume()` returns in Python — so a Python-side correction in between is
enough, no re-render needed):

- `src/chamber_obb/dashboard.py::_patched_widget_esm()` / `_AnnotatedDashboard` —
  a `SequenceDashboard` subclass whose widget JS flattens each nested
  `roi: {center, size, angle}` entry into the flat `x`/`y`/`w`/`h`/`angle`
  shape at startup. No-op if the startup line no longer exists in acia.
- **Do not flatten `dash.selections` in Python** (the first workaround did, and
  it was wrong): the nested shape is the Python-side contract. The widget writes
  it back on every edit (`pushSelections`) and `SequenceDashboard.manifest` /
  `.save()` read `item["roi"]`, so a Python-flattened list made "Save
  selection.json" and `dash.manifest` fail with `KeyError: 'roi'` until the user
  edited something. The fix proposed above ("build the flat shape in `resume()`")
  has the same flaw — the upstream fix belongs in the JS init line (flatten
  there), not in `resume()`.
- Test: `tests/test_dashboard.py::test_resumed_selections_stay_nested_so_manifest_and_save_work`
  and `test_widget_js_flattens_nested_selections_on_startup`.
- Commit: [`4e433ed`](https://github.com/looZ149/IHK-Projekt/commit/4e433ed62b3feaf92272fa9db6e2e0ea3a7972fc)
  (tag `dashboard-fix-1`), branch `fix/dashboard-resume-flatten-workaround`.

This unblocks `open_annotated_dashboard()` specifically. It does **not**
help a manual `SequenceDashboard.resume()` call outside chamber_obb (e.g. a
curator resuming their own saved `selection.json`) — that still needs the
real fix in `acia-core` itself.
