import { useCallback, useEffect, useRef, useState } from "react";

import { fmtParams, latticeAxisOf } from "../spec.js";

const S = {
  panel: {
    width: 320,
    height: "100%",
    overflowY: "auto",
    padding: "18px 20px",
    background: "linear-gradient(180deg, #131926 0%, #10141d 42%, #0e121a 100%)",
    borderRight: "1px solid #1e2635",
    flexShrink: 0,
  },
  h1: { fontSize: 15, fontWeight: 650, letterSpacing: 0.2, marginBottom: 2 },
  sub: { color: "#8b95a8", fontSize: 12, marginBottom: 14 },
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
  // The 3-D wavefront already shows this, but only for whichever axis is on
  // screen; here it is legible even when the camera is pointed elsewhere.
  layerFill: {
    position: "absolute",
    inset: 0,
    transformOrigin: "left center",
    background: "linear-gradient(90deg, rgba(255,192,97,0.20), rgba(255,192,97,0.05))",
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
  btnStart: {
    background: "#173423",
    color: "#9ece6a",
    border: "1px solid #2f6e42",
    borderRadius: 6,
    padding: "9px 12px",
    cursor: "pointer",
    fontSize: 13.5,
    fontWeight: 650,
    width: "100%",
  },
  btnStop: {
    background: "#2a1418",
    color: "#fca5a5",
    border: "1px solid #7f1d1d",
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
  dot: (color) => ({
    display: "inline-block",
    width: 8,
    height: 8,
    borderRadius: 99,
    background: color,
    marginRight: 7,
  }),
  barOuter: {
    height: 6,
    borderRadius: 99,
    background: "#0c1018",
    border: "1px solid #232d40",
    margin: "8px 0 2px",
    overflow: "hidden",
  },
  barInner: (pct, color) => ({
    height: "100%",
    width: `${pct}%`,
    background: color,
    transition: "width 0.4s",
  }),
};

const STATUS = {
  waiting: { color: "#e0af68", label: "waiting for start" },
  training: { color: "#e8963a", label: "training" },
  paused: { color: "#7aa2f7", label: "paused" },
  done: { color: "#9ece6a", label: "done" },
  stopped: { color: "#8b95a8", label: "stopped" },
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


function LossChart({ metrics }) {
  const w = 276;
  const h = 84;
  const pad = 4;
  const logs = metrics.map((m) => Math.log10(Math.max(m.loss, 1e-8)));
  const lo = Math.min(...logs);
  const hi = Math.max(...logs, lo + 1e-6);
  const x = (i) => pad + (i * (w - 2 * pad)) / Math.max(metrics.length - 1, 1);
  const y = (v) => pad + ((hi - v) * (h - 2 * pad)) / (hi - lo);
  const line = logs
    .map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`)
    .join(" ");
  const held = metrics
    .map((m, i) =>
      m.held_out == null ? null : [i, Math.log10(Math.max(m.held_out, 1e-8))],
    )
    .filter(Boolean);
  const heldLine = held
    .map(([i, v]) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`)
    .join(" ");
  return (
    <svg width={w} height={h} style={{ display: "block", marginTop: 8 }}>
      <polyline points={line} fill="none" stroke="#e8963a" strokeWidth="1.6" />
      {held.length > 1 && (
        <polyline
          points={heldLine}
          fill="none"
          stroke="#7aa2f7"
          strokeWidth="1.4"
          strokeDasharray="4 3"
        />
      )}
    </svg>
  );
}

function RunPanel({ live }) {
  const [sendError, setSendError] = useState(null);
  const post = useCallback(
    (action) => {
      setSendError(null);
      fetch(`${live.control}/control`, {
        method: "POST",
        headers: { "Content-Type": "text/plain" },
        body: JSON.stringify({ action }),
      }).catch(() =>
        setSendError(
          "control server unreachable — is the training script running?",
        ),
      );
    },
    [live.control],
  );

  const st = STATUS[live.status] ?? STATUS.stopped;
  const last = live.metrics?.length
    ? live.metrics[live.metrics.length - 1]
    : null;
  const lastHeld = live.metrics
    ? [...live.metrics].reverse().find((e) => e.held_out != null)
    : null;
  const pct =
    live.total_steps && last ? ((last.step + 1) / live.total_steps) * 100 : 0;

  return (
    <div style={{ ...S.card, borderColor: "#2c3b58" }}>
      <div style={S.row}>
        <span style={S.key}>
          <span style={S.dot(st.color)} />
          {st.label}
        </span>
        <span style={S.chip}>{live.device}</span>
      </div>
      <div style={S.row}>
        <span style={S.key}>task</span>
        <span style={{ fontSize: 11, color: "#8b95a8", textAlign: "right" }}>
          {live.task}
        </span>
      </div>

      {live.status === "waiting" && (
        <div style={{ marginTop: 10 }}>
          <button style={S.btnStart} onClick={() => post("start")}>
            ▶ start training
          </button>
          <div style={{ fontSize: 11.5, color: "#8b95a8", marginTop: 6 }}>
            the model is built and idle — nothing runs until you press start
          </div>
        </div>
      )}

      {(live.status === "training" || live.status === "paused") && (
        <div style={S.controls}>
          {live.status === "training" ? (
            <button style={S.btn} onClick={() => post("pause")}>
              ⏸ pause
            </button>
          ) : (
            <button style={S.btn} onClick={() => post("resume")}>
              ▶ resume
            </button>
          )}
          <button style={S.btnStop} onClick={() => post("stop")}>
            ■ stop
          </button>
        </div>
      )}

      {last && (
        <>
          <div style={S.barOuter}>
            <div style={S.barInner(pct, st.color)} />
          </div>
          <div style={S.row}>
            <span style={S.key}>step</span>
            <span>
              {last.step + 1} / {live.total_steps}
            </span>
          </div>
          <div style={S.row}>
            <span style={S.key}>train loss</span>
            <span style={{ color: "#e8963a" }}>{last.loss.toFixed(5)}</span>
          </div>
          {lastHeld && (
            <div style={S.row}>
              <span style={S.key}>held-out</span>
              <span style={{ color: "#7aa2f7" }}>
                {lastHeld.held_out.toFixed(5)}
              </span>
            </div>
          )}
          <LossChart metrics={live.metrics} />
        </>
      )}

      {sendError && <div style={S.warn}>{sendError}</div>}
    </div>
  );
}

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
  live,
  anim,
}) {
  const { spec } = parsed;
  const m = spec.model;
  const cells = spec.lattice.cells;
  const dirGlyph = (r) => (r ? "←" : "→");

  return (
    <div style={S.panel} className="td-panel">
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
      <div style={{ margin: "8px 0 0" }}>
        <label style={{ ...S.btn, display: "inline-block", flex: "none" }}>
          open spec JSON
          <input
            type="file"
            accept=".json"
            style={{ display: "none" }}
            onChange={(e) => e.target.files[0] && onFile(e.target.files[0])}
          />
        </label>
      </div>

      {live && (
        <>
          <div style={S.section}>run</div>
          <RunPanel live={live} />
        </>
      )}

      <div style={S.section}>model</div>
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
          <b>never swept:</b> {spec.sweeps.unswept_axes.join(", ")} — these axes
          get no mixing.
        </div>
      )}

      <div style={S.section}>directions</div>
      <div style={S.card}>
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

      <div style={S.section}>sweep animation</div>
      <div style={{ ...S.controls, marginTop: 0, marginBottom: 6 }}>
        <button style={S.btn} className="td-btn" onClick={onTogglePlay}>
          {playing ? "⏸ pause" : "▶ play"}
        </button>
        <button style={S.btn} className="td-btn" onClick={() => onSelectLayer(layerIndex - 1)}>
          ◀ prev
        </button>
        <button style={S.btn} className="td-btn" onClick={() => onSelectLayer(layerIndex + 1)}>
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
                <span style={S.key}>
                  {l.tokens ? `${l.tokens} tokens` : ""}
                </span>
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
    </div>
  );
}
