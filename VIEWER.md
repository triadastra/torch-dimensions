# torch-dimensions viewer — Design and Plan

Companion to [DESIGN.md](DESIGN.md) and [PLAN.md](PLAN.md). This one covers the GUI: what it shows, what it deliberately does not, and how it gets built.

**Status:** V0 and V1 built. The architecture spec (`model.to_spec()`) and the static viewer (`viewer/` — Vite + React + react-three-fiber; sample specs, file loading, layer stepping, sweep animation, ranks 1–3 literal, rank 4 stacked) are working. V2 (`td.viz.show`, wheel bundling) is next.

---

## 1. What this is for

The library knows something no general tool can see: **which axis each layer sweeps, in which direction, over what lattice, with which cells absent.** TensorBoard's graph view renders ops, so an N-D model appears there as an undifferentiated pile of permutes and matmuls — technically complete, useless for the question people actually have.

That question is almost always *"is my model actually doing what I think along each axis?"* The bidirectional-aliasing bug this project already hit is the canonical example: a schedule that pinned every axis to one direction forever, invisible in code review, invisible in the loss curve, and instantly obvious in a view where you watch the sweep arrows never flip.

So the viewer's job is to make an N-D architecture *legible*, not to be a better plotting library.

---

## 2. Deliberately out of scope

- **No trainer.** [PLAN.md](PLAN.md) rules one out permanently and the GUI does not get to smuggle it back. The viewer is a **preflight and a monitor**, never a driver: it shows what will happen, and your own loop stays yours.
- **No metric plotting in v1.** Reaching parity with TensorBoard or W&B on scalars and run comparison is weeks of work against mature free tools. The version worth building is loss *annotated by architecture* — which layer and axis were active at a spike — and that is only possible once the architecture view exists. It is v2 for that reason, not because metrics do not matter.
- **No npm for users.** See §5.

---

## 3. Architecture: the library emits, the viewer consumes

```
model.to_spec()  ──►  architecture JSON  ──►  React + Vite + three.js
   (pure Python)         (versioned)              (reads, renders)
```

The Python side never imports anything JS-adjacent and never opens a socket. It produces a versioned JSON document; the viewer reads it. That boundary means the spec is independently useful — for config diffing, for debugging, for docs — and the renderer can be rewritten without touching the library.

The spec is implemented and tested (`td.spec`, `src/torch_dimensions/spec.py`). It carries the lattice with a run-length-encoded presence mask, the resolved per-layer sweep schedule, mixer identities and parameter counts, the `nd_method`, and symbolic input/output shapes.

---

## 4. Rendering N dimensions

**Ranks 1–3: literal.** A rank-3 lattice *is* a 3-D grid. Cells are instanced cubes; absent cells are not drawn. A sweep animates as a wavefront travelling along the swept axis, in the swept direction, one layer at a time. Direction is the thing people get wrong, so it is the thing the animation makes loudest.

**Ranks 4–5: dimensional stacking.** Each axis above the third becomes a spatial arrangement of the blocks below it, separated by a gap larger than the gap at the level beneath. Rank 4 is a grid of cubes; rank 5 is a grid of those grids. Every cell stays visible — no slider hides part of the model — and the gap hierarchy is what encodes which level you are looking at.

**Rank 6+: slicing.** Not because it cannot be drawn but because it cannot be *read*: the gap hierarchy needs each level's spacing to be visibly larger than the last, and perceptually that runs out around five. Above it, three axes render and the rest become index selectors.

**Budget.** Rendered cells are `∏shape`, so a rank-5 lattice of size 8 is 32,768 instances — fine for `InstancedMesh`, and sparse lattices draw only present cells. The viewer caps total instances and falls back to slicing past the cap rather than freezing the tab.

---

## 5. Packaging: `pip install torch-dimensions[gui]`

The viewer lives in `viewer/` in this repo and ships **prebuilt** inside the wheel. Users never see npm, never run a dev server, never install node. `td.viz.show(model)` opens a browser at a small local static server.

The costs are real and accepted: the release pipeline must build the bundle before building the wheel, CI grows a JS job, and the wheel gets meaningfully larger. In exchange the install experience is one line, which for a visualization tool is most of whether it gets used at all.

The base `pip install torch-dimensions` stays exactly as it is — torch and nothing else. The `[gui]` extra adds only the server dependency; the bundle itself is a data file.

---

## 6. Build order

**V0 — spec.** `model.to_spec()`, versioned, tested against the library's own models. *Done.*

**V1 — static architecture viewer.** Vite + React + react-three-fiber. Load a spec (file or paste), render the lattice, step through layers, animate the sweep. Rank 1–3 literal, 4–5 stacked. No Python server yet — it reads a JSON file, which makes it developable and testable entirely standalone. *Built:* `viewer/` — `npm install && npm run dev`. Three bundled sample specs (2-D sparse LSTM, 3-D paired-schedule Mamba-ND, 4-D S4D exercising the stacking); a time sweep breathes uniformly (no lattice direction to draw), the kernel family flashes all axes at once rather than pretending to sweep (§7 Q1 answered by idiom, not schema), and the never-swept-axes warning renders red in the sidebar. One hard-won rule: the WebGL canvas is never remounted — swap specs by rebuilding the instanced mesh and recentering the camera, or the context is lost and the scene goes permanently black.

**V2 — `td.viz.show(model)`.** The static server, browser launch, and the wheel-bundling pipeline. This is the packaging work, deliberately separated from the rendering work so neither blocks the other.

**V3 — expected output and shape flow.** Per-stage shapes, parameter counts, memory estimate, and the preflight "here is what a forward pass will do" panel. The natural place for a *"this plan never sweeps axis w"* warning to appear visually rather than as a `UserWarning` nobody reads.

**V4 — live metrics.** A streaming channel, and loss annotated by which layer and axis were active. Only worth building on top of V1–V3.

---

## 7. Open questions

1. **Does the spec need to describe the kernel family?** `axial_scan` has a per-layer sweep, which animates naturally. A Kronecker contraction does not sweep — it contracts all axes at once. That is a different visual idiom and the spec schema should not assume the scan one.
2. **How is a hybrid `nd_method` shown**, where a kernel operator handles the lattice and a mixer handles time? Two idioms in one frame.
3. **Should the spec round-trip?** A spec that can rebuild its model would double as a checkpoint format, which overlaps Phase 8's `save`/`load`. Attractive, and a reason to keep the two schemas aware of each other rather than divergent.
