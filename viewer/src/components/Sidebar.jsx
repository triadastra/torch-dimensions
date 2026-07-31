import { fmtParams, latticeAxisOf } from "../spec.js";

const S = {
  panel: {
    width: 320,
    height: "100%",
    overflowY: "auto",
    padding: "18px 20px",
    background: "#10141d",
    borderRight: "1px solid #1e2635",
    flexShrink: 0,
  },
  h1: { fontSize: 15, fontWeight: 650, letterSpacing: 0.2, marginBottom: 2 },
  sub: { color: "#8b95a8", fontSize: 12, marginBottom: 14 },
  card: {
    background: "#151b28",
    border: "1px solid #232d40",
    borderRadius: 8,
    padding: "10px 12px",
    marginBottom: 12,
  },
  row: { display: "flex", justifyContent: "space-between", padding: "2px 0", fontSize: 13 },
  key: { color: "#8b95a8" },
  warn: {
    background: "#2a1418",
    border: "1px solid #7f1d1d",
    color: "#fca5a5",
    borderRadius: 8,
    padding: "8px 12px",
    marginBottom: 12,
    fontSize: 13,
  },
  layer: (active) => ({
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "6px 10px",
    borderRadius: 6,
    cursor: "pointer",
    fontSize: 13,
    background: active ? "#23304a" : "transparent",
    border: active ? "1px solid #3b4d75" : "1px solid transparent",
  }),
  chip: {
    fontSize: 11,
    padding: "1px 7px",
    borderRadius: 99,
    background: "#233049",
    color: "#9fb3d9",
  },
  controls: { display: "flex", gap: 8, margin: "10px 0 16px" },
  btn: {
    background: "#1c2536",
    color: "#d6dbe4",
    border: "1px solid #2b3850",
    borderRadius: 6,
    padding: "6px 12px",
    cursor: "pointer",
    fontSize: 13,
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
};

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
}) {
  const { spec } = parsed;
  const m = spec.model;
  const cells = spec.lattice.cells;
  const dirGlyph = (r) => (r ? "←" : "→");

  return (
    <div style={S.panel}>
      <div style={S.h1}>torch-dimensions</div>
      <div style={S.sub}>architecture viewer &middot; spec v{spec.version}</div>

      <select style={S.select} value={sampleKey} onChange={(e) => onSample(e.target.value)}>
        {Object.keys(samples).map((k) => (
          <option key={k} value={k}>
            sample: {k}
          </option>
        ))}
      </select>
      <div style={{ margin: "8px 0 14px" }}>
        <label style={{ ...S.btn, display: "inline-block" }}>
          open spec JSON
          <input
            type="file"
            accept=".json"
            style={{ display: "none" }}
            onChange={(e) => e.target.files[0] && onFile(e.target.files[0])}
          />
        </label>
      </div>

      <div style={S.card}>
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
          <b>never swept:</b> {spec.sweeps.unswept_axes.join(", ")} — these axes get no mixing.
        </div>
      )}

      <div style={S.card}>
        {Object.entries(spec.sweeps.directions).map(([axis, dir]) => (
          <div key={axis} style={S.row}>
            <span style={S.key}>{axis}</span>
            <span>
              {dir === "both" ? "→ and ←" : dir === "forward" ? "→ only" : "← only"}
            </span>
          </div>
        ))}
      </div>

      <div style={S.controls}>
        <button style={S.btn} onClick={onTogglePlay}>
          {playing ? "pause" : "play"}
        </button>
        <button style={S.btn} onClick={() => onSelectLayer(layerIndex - 1)}>
          ◀ prev
        </button>
        <button style={S.btn} onClick={() => onSelectLayer(layerIndex + 1)}>
          next ▶
        </button>
      </div>

      {spec.layers.map((l, i) => {
        const isTime = latticeAxisOf(l, spec) === null;
        return (
          <div key={i} style={S.layer(i === layerIndex)} onClick={() => onSelectLayer(i)}>
            <span style={{ color: "#8b95a8", width: 28 }}>L{i}</span>
            <b>{l.axis}</b>
            <span>{isTime ? "→ (causal)" : dirGlyph(l.reverse)}</span>
            <span style={{ flex: 1 }} />
            <span style={S.chip}>{l.mixer}</span>
          </div>
        );
      })}
    </div>
  );
}
