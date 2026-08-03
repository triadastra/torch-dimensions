import { useEffect, useRef } from "react";

import { fmtParams, latticeAxisOf } from "../spec.js";

// One column tried to hold the model summary, the sweep directions, the run
// metrics and every layer at once, so everything was cramped and the layer
// list — the part that changes as you watch — sat below the fold. Tabs put one
// concern on screen at a time; the run metrics moved out entirely, to the dock
// under the scene where a training curve has room to be a curve.

const S = {
  panel: {
    width: 320,
    height: "100%",
    display: "flex",
    flexDirection: "column",
    background: "linear-gradient(180deg, #131926 0%, #10141d 42%, #0e121a 100%)",
    borderRight: "1px solid #1e2635",
    flexShrink: 0,
  },
  head: { padding: "16px 18px 0" },
  body: { flex: 1, overflowY: "auto", padding: "4px 18px 18px" },
  h1: { fontSize: 15, fontWeight: 650, letterSpacing: 0.2, marginBottom: 2 },
  sub: { color: "#8b95a8", fontSize: 12, marginBottom: 12 },
  tabs: {
    display: "flex",
    gap: 2,
    padding: "0 12px",
    borderBottom: "1px solid #1e2635",
  },
  tab: (active) => ({
    flex: 1,
    background: "none",
    border: "none",
    borderBottom: active ? "2px solid #ffc061" : "2px solid transparent",
    color: active ? "#e8eef8" : "#7d8aa3",
    fontSize: 12.5,
    fontWeight: active ? 650 : 500,
    padding: "9px 4px",
    cursor: "pointer",
    letterSpacing: 0.3,
  }),
  section: {
    color: "#5d6b84",
    fontSize: 10.5,
    fontWeight: 700,
    letterSpacing: 1.4,
    textTransform: "uppercase",
    margin: "16px 2px 6px",
  },
  card: {
    background: "#151b28",
    border: "1px solid #232d40",
    borderRadius: 8,
    padding: "10px 12px",
  },
  row: {
    display: "flex",
    justifyContent: "space-between",
    padding: "2px 0",
    fontSize: 13,
  },
  key: { color: "#8b95a8" },
  warn: {
    background: "#2a1418",
    border: "1px solid #7f1d1d",
    color: "#fca5a5",
    borderRadius: 8,
    padding: "8px 12px",
    marginTop: 10,
    fontSize: 13,
  },
  layer: (active) => ({
    position: "relative",
    overflow: "hidden",
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "6px 10px",
    borderRadius: 6,
    cursor: "pointer",
    fontSize: 13,
    background: active ? "#22304c" : "transparent",
    border: active ? "1px solid #41568a" : "1px solid transparent",
    boxShadow: active ? "inset 3px 0 0 #ffc061" : "none",
  }),
  // The sweep's progress *through the active layer*, drawn behind its row.
  layerFill: {
    position: "absolute",
    inset: 0,
    transformOrigin: "left center",
    background:
      "linear-gradient(90deg, rgba(255,192,97,0.20), rgba(255,192,97,0.05))",
    pointerEvents: "none",
  },
  chip: {
    fontSize: 11,
    padding: "1px 7px",
    borderRadius: 99,
    background: "#233049",
    color: "#9fb3d9",
  },
  controls: { display: "flex", gap: 8, marginTop: 10 },
  btn: {
    background: "#1c2536",
    color: "#d6dbe4",
    border: "1px solid #2b3850",
    borderRadius: 6,
    padding: "7px 12px",
    cursor: "pointer",
    fontSize: 13,
    flex: 1,
  },
  select: {
    background: "#1c2536",
    color: "#d6dbe4",
    border: "1px solid #2b3850",
    borderRadius: 6,
    padding: "6px 8px",
    fontSize: 13,
    width: "100%",
  },
  switch: (on) => ({
    display: "flex",
    alignItems: "center",
    gap: 10,
    width: "100%",
    background: on ? "#1d2b21" : "#1c2536",
    color: on ? "#9ece6a" : "#d6dbe4",
    border: `1px solid ${on ? "#2f6e42" : "#2b3850"}`,
    borderRadius: 8,
    padding: "10px 12px",
    cursor: "pointer",
    fontSize: 13.5,
    fontWeight: 600,
    textAlign: "left",
  }),
  knob: (on) => ({
    width: 30,
    height: 17,
    borderRadius: 99,
    background: on ? "#2f6e42" : "#2b3850",
    position: "relative",
    flexShrink: 0,
    transition: "background 160ms ease",
  }),
  dot: (on) => ({
    position: "absolute",
    top: 2,
    left: on ? 15 : 2,
    width: 13,
    height: 13,
    borderRadius: 99,
    background: on ? "#9ece6a" : "#7d8aa3",
    transition: "left 160ms ease",
  }),
  note: { color: "#5d6b84", fontSize: 11.5, marginTop: 8, lineHeight: 1.5 },
};

// Progress through the active layer, written straight to the DOM node. Going
// through React state here would re-render the whole panel sixty times a
// second to move one bar.
function LayerFill({ anim }) {
  const ref = useRef();
  useEffect(() => {
    let raf;
    const tick = () => {
      const el = ref.current;
      if (el) el.style.transform = `scaleX(${anim.current.progress})`;
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [anim]);
  return <div ref={ref} style={S.layerFill} />;
}

function ModelTab({ parsed }) {
  const { spec } = parsed;
  const m = spec.model;
  const cells = spec.lattice.cells;
  return (
    <>
      <div style={S.section}>model</div>
      <div style={S.card} className="td-card">
        <div style={S.row}>
          <span style={S.key}>model</span>
          <b>{m.kind}</b>
        </div>
        <div style={S.row}>
          <span style={S.key}>d_model</span>
          <span>{m.d_model}</span>
        </div>
        <div style={S.row}>
          <span style={S.key}>layers</span>
          <span>{m.n_layers}</span>
        </div>
        <div style={S.row}>
          <span style={S.key}>parameters</span>
          <span>{fmtParams(m.n_params)}</span>
        </div>
        <div style={S.row}>
          <span style={S.key}>method</span>
          <span>{spec.nd_method.name}</span>
        </div>
        <div style={S.row}>
          <span style={S.key}>lattice</span>
          <span>
            {spec.lattice.shape.join(" × ") || "—"}
            {spec.lattice.time ? " + time" : ""}
          </span>
        </div>
        <div style={S.row}>
          <span style={S.key}>cells</span>
          <span>
            {cells.present} / {cells.total}
            {cells.dense ? "" : " (sparse)"}
          </span>
        </div>
      </div>

      {spec.sweeps.unswept_axes.length > 0 && (
        <div style={S.warn}>
          <b>never swept:</b> {spec.sweeps.unswept_axes.join(", ")} — these axes
          get no mixing.
        </div>
      )}

      <div style={S.section}>directions</div>
      <div style={S.card} className="td-card">
        {Object.entries(spec.sweeps.directions).map(([axis, dir]) => (
          <div key={axis} style={S.row}>
            <span style={S.key}>{axis}</span>
            <span>
              {dir === "both"
                ? "→ and ←"
                : dir === "forward"
                  ? "→ only"
                  : "← only"}
            </span>
          </div>
        ))}
        {(spec.sweeps.joint_axes ?? []).map((axis) => (
          <div key={axis} style={S.row}>
            <span style={S.key}>{axis}</span>
            <span style={{ color: "#5eead4" }}>⊕ joint</span>
          </div>
        ))}
        {(spec.sweeps.contracted_axes ?? []).map((axis) => (
          <div key={axis} style={S.row}>
            <span style={S.key}>{axis}</span>
            {/* A contraction has no direction: the whole axis is mixed at
                once, every layer. Listing it as "→ only" would invent a
                property the model does not have. */}
            <span style={{ color: "#c39bd8" }}>⊗ contracted</span>
          </div>
        ))}
      </div>
    </>
  );
}

function LayersTab({
  parsed,
  layerIndex,
  playing,
  onTogglePlay,
  onSelectLayer,
  anim,
}) {
  const { spec } = parsed;
  const dirGlyph = (r) => (r ? "←" : "→");
  return (
    <>
      <div style={S.section}>sweep animation</div>
      <div style={{ ...S.controls, marginTop: 0, marginBottom: 8 }}>
        <button style={S.btn} className="td-btn" onClick={onTogglePlay}>
          {playing ? "⏸ pause" : "▶ play"}
        </button>
        <button
          style={S.btn}
          className="td-btn"
          onClick={() => onSelectLayer(layerIndex - 1)}
        >
          ◀ prev
        </button>
        <button
          style={S.btn}
          className="td-btn"
          onClick={() => onSelectLayer(layerIndex + 1)}
        >
          next ▶
        </button>
      </div>

      {spec.layers.map((l, i) => {
        // A kernel-family layer contracts every spatial axis at once and, in
        // the hybrid form, sweeps time. Rendering it with the scan family's
        // "one axis, one arrow" was drawing sweeps that never happen — the
        // spec used to claim them (DEBUG.md #26).
        const kernel = l.kind === "kernel";
        const joint = l.kind === "flatten";
        const isTime = !kernel && !joint && latticeAxisOf(l, spec) === null;
        return (
          <div
            key={i}
            style={S.layer(i === layerIndex)}
            className="td-row"
            onClick={() => onSelectLayer(i)}
          >
            {i === layerIndex && anim && <LayerFill anim={anim} />}
            <span style={{ color: "#8b95a8", width: 28, position: "relative" }}>
              L{i}
            </span>
            {joint ? (
              <>
                <b style={{ color: "#5eead4" }}>{(l.axes ?? []).join(" ⊕ ")}</b>
                <span style={S.key}>{l.tokens ? `${l.tokens} tokens` : ""}</span>
              </>
            ) : kernel ? (
              <>
                <b style={{ color: "#c39bd8" }}>
                  {(l.contracted ?? []).join(" ⊗ ")}
                </b>
                <span style={S.key}>{l.axis ? `+ ${l.axis} →` : ""}</span>
              </>
            ) : (
              <>
                <b>{l.axis}</b>
                <span>{isTime ? "→ (causal)" : dirGlyph(l.reverse)}</span>
              </>
            )}
            <span style={{ flex: 1 }} />
            {l.mixer && <span style={S.chip}>{l.mixer}</span>}
          </div>
        );
      })}
    </>
  );
}

function DataTab({ parsed, dataShow, onToggleData }) {
  const { spec } = parsed;
  const cells = spec.lattice.cells;
  const names = (spec.lattice.names ?? []).slice(spec.lattice.time ? 1 : 0);
  return (
    <>
      <div style={S.section}>cell data</div>
      <button style={S.switch(dataShow)} className="td-btn" onClick={onToggleData}>
        <span style={S.knob(dataShow)}>
          <span style={S.dot(dataShow)} />
        </span>
        data_show — label every cell
      </button>
      <div style={S.note}>
        Each cell is labelled with its axis names and indices, and tinted by
        which side of the wavefront it is on. Only the cells nearest the camera
        get one: a label per cell across a rank-4 lattice is thousands of text
        meshes and an unreadable thicket besides. Zoom in to name more of them.
      </div>

      <div style={S.section}>axes</div>
      <div style={S.card} className="td-card">
        {parsed.shape.map((size, i) => (
          <div key={i} style={S.row}>
            <span style={S.key}>{names[i] ?? `dim${i}`}</span>
            <span>{size}</span>
          </div>
        ))}
        {spec.lattice.time && (
          <div style={S.row}>
            <span style={S.key}>time</span>
            <span style={{ color: "#8b95a8" }}>dynamic</span>
          </div>
        )}
      </div>

      <div style={S.section}>presence</div>
      <div style={S.card} className="td-card">
        <div style={S.row}>
          <span style={S.key}>present</span>
          <span style={{ color: "#9ece6a" }}>{cells.present}</span>
        </div>
        <div style={S.row}>
          <span style={S.key}>absent</span>
          <span style={{ color: cells.dense ? "#8b95a8" : "#e0af68" }}>
            {cells.total - cells.present}
          </span>
        </div>
        <div style={S.row}>
          <span style={S.key}>total</span>
          <span>{cells.total}</span>
        </div>
      </div>
      {!cells.dense && (
        <div style={S.note}>
          Absent cells are not drawn at all, and their values can never reach an
          output — that is tested bitwise, not asserted.
        </div>
      )}
    </>
  );
}

const TABS = ["model", "layers", "data"];

export default function Sidebar({
  parsed,
  layerIndex,
  playing,
  onTogglePlay,
  onSelectLayer,
  samples,
  sampleKey,
  onSample,
  onFile,
  anim,
  tab,
  onTab,
  dataShow,
  onToggleData,
}) {
  const { spec } = parsed;

  return (
    <div style={S.panel}>
      <div style={S.head}>
        <div style={S.h1}>torch-dimensions</div>
        <div style={S.sub}>architecture viewer &middot; spec v{spec.version}</div>

        <select
          style={S.select}
          value={sampleKey}
          onChange={(e) => onSample(e.target.value)}
        >
          {Object.keys(samples).map((k) => (
            <option key={k} value={k}>
              {k}
            </option>
          ))}
        </select>
        <div style={{ margin: "8px 0 12px" }}>
          <label
            style={{ ...S.btn, display: "inline-block", flex: "none" }}
            className="td-btn"
          >
            open spec JSON
            <input
              type="file"
              accept=".json"
              style={{ display: "none" }}
              onChange={(e) => e.target.files[0] && onFile(e.target.files[0])}
            />
          </label>
        </div>
      </div>

      <div style={S.tabs}>
        {TABS.map((t) => (
          <button key={t} style={S.tab(tab === t)} onClick={() => onTab(t)}>
            {t}
          </button>
        ))}
      </div>

      <div style={S.body} className="td-panel">
        {tab === "model" && <ModelTab parsed={parsed} />}
        {tab === "layers" && (
          <LayersTab
            parsed={parsed}
            layerIndex={layerIndex}
            playing={playing}
            onTogglePlay={onTogglePlay}
            onSelectLayer={onSelectLayer}
            anim={anim}
          />
        )}
        {tab === "data" && (
          <DataTab
            parsed={parsed}
            dataShow={dataShow}
            onToggleData={onToggleData}
          />
        )}
      </div>
    </div>
  );
}
