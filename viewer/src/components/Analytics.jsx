import { useCallback, useEffect, useRef, useState } from "react";

// The run panel used to be one card in a 320px column, with an 84px chart
// squeezed under the model summary. Training curves are the thing you stare at
// for minutes at a time; they get the width here.

const STATUS = {
  waiting: { color: "#e0af68", label: "waiting for start" },
  training: { color: "#e8963a", label: "training" },
  paused: { color: "#7aa2f7", label: "paused" },
  done: { color: "#9ece6a", label: "done" },
  stopped: { color: "#8b95a8", label: "stopped" },
};

const S = {
  dock: {
    position: "relative",
    flexShrink: 0,
    background: "linear-gradient(180deg, rgba(13,17,26,0.72) 0%, rgba(11,14,20,0.95) 42%)",
    borderTop: "1px solid #1e2635",
    backdropFilter: "blur(10px)",
    display: "flex",
    alignItems: "stretch",
    gap: 18,
    padding: "12px 18px",
  },
  stat: { minWidth: 92 },
  statLabel: {
    color: "#5d6b84",
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: 1.2,
    textTransform: "uppercase",
  },
  statValue: {
    fontSize: 19,
    fontWeight: 600,
    fontVariantNumeric: "tabular-nums",
    marginTop: 2,
  },
  chip: {
    fontSize: 11,
    padding: "1px 7px",
    borderRadius: 99,
    background: "#233049",
    color: "#9fb3d9",
  },
  btn: {
    background: "#1c2536",
    color: "#d6dbe4",
    border: "1px solid #2b3850",
    borderRadius: 6,
    padding: "7px 14px",
    cursor: "pointer",
    fontSize: 13,
  },
  btnStart: {
    background: "#173423",
    color: "#9ece6a",
    border: "1px solid #2f6e42",
    borderRadius: 6,
    padding: "8px 16px",
    cursor: "pointer",
    fontSize: 13.5,
    fontWeight: 650,
  },
  btnStop: {
    background: "#2a1418",
    color: "#fca5a5",
    border: "1px solid #7f1d1d",
    borderRadius: 6,
    padding: "7px 14px",
    cursor: "pointer",
    fontSize: 13,
  },
  toggle: {
    position: "absolute",
    right: 14,
    top: -34,
    background: "rgba(16,20,29,0.9)",
    color: "#9fb3d9",
    border: "1px solid #2b3850",
    borderRadius: 6,
    padding: "4px 10px",
    cursor: "pointer",
    fontSize: 12,
  },
};

// Log-scale loss curve. Wide, with a filled area under the training line and a
// held-out overlay; the axis is labelled because an unlabelled log axis is a
// decoration rather than a measurement.
function LossChart({ metrics, width, height }) {
  const pad = { l: 46, r: 10, t: 8, b: 16 };
  const w = Math.max(160, width);
  const h = height;
  const logs = metrics.map((m) => Math.log10(Math.max(m.loss, 1e-9)));
  const held = metrics
    .map((m, i) =>
      m.held_out == null ? null : [i, Math.log10(Math.max(m.held_out, 1e-9))],
    )
    .filter(Boolean);
  const all = logs.concat(held.map((d) => d[1]));
  const lo = Math.min(...all);
  const hi = Math.max(...all, lo + 1e-6);
  const x = (i) =>
    pad.l + (i * (w - pad.l - pad.r)) / Math.max(metrics.length - 1, 1);
  const y = (v) => pad.t + ((hi - v) * (h - pad.t - pad.b)) / (hi - lo);

  const line = logs.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const area = `${pad.l},${y(lo)} ${line} ${x(logs.length - 1)},${y(lo)}`;
  const heldLine = held
    .map(([i, v]) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`)
    .join(" ");
  const ticks = [hi, (hi + lo) / 2, lo];

  return (
    <svg width={w} height={h} style={{ display: "block" }}>
      <defs>
        <linearGradient id="td-loss" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#e8963a" stopOpacity="0.34" />
          <stop offset="100%" stopColor="#e8963a" stopOpacity="0.02" />
        </linearGradient>
      </defs>
      {ticks.map((v, i) => (
        <g key={i}>
          <line
            x1={pad.l}
            x2={w - pad.r}
            y1={y(v)}
            y2={y(v)}
            stroke="#1c2437"
            strokeWidth="1"
          />
          <text x={4} y={y(v) + 3.5} fill="#5d6b84" fontSize="9.5">
            {(10 ** v).toExponential(0)}
          </text>
        </g>
      ))}
      <polygon points={area} fill="url(#td-loss)" />
      <polyline points={line} fill="none" stroke="#e8963a" strokeWidth="1.8" />
      {held.length > 1 && (
        <polyline
          points={heldLine}
          fill="none"
          stroke="#7aa2f7"
          strokeWidth="1.5"
          strokeDasharray="4 3"
        />
      )}
      <text x={pad.l} y={h - 3} fill="#5d6b84" fontSize="9.5">
        step 0
      </text>
      <text x={w - pad.r} y={h - 3} fill="#5d6b84" fontSize="9.5" textAnchor="end">
        loss (log)
      </text>
    </svg>
  );
}

// The chart takes whatever width the dock has left. A fixed one made the dock
// wider than its container, and a flex row that cannot shrink pushes its
// siblings out of the window — which is how the sidebar ended up at x = -96.
function useElementWidth() {
  const ref = useRef(null);
  const [width, setWidth] = useState(320);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return undefined;
    const ro = new ResizeObserver(([entry]) => {
      setWidth(Math.max(160, Math.floor(entry.contentRect.width)));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, width];
}


function Stat({ label, value, color }) {
  return (
    <div style={S.stat}>
      <div style={S.statLabel}>{label}</div>
      <div style={{ ...S.statValue, color: color ?? "#d6dbe4" }}>{value}</div>
    </div>
  );
}

export default function Analytics({ live, parsed, open, onToggle, stepRate }) {
  const [sendError, setSendError] = useState(null);
  const [chartRef, chartWidth] = useElementWidth();
  const post = useCallback(
    (action) => {
      setSendError(null);
      fetch(`${live.control}/control`, {
        method: "POST",
        headers: { "Content-Type": "text/plain" },
        body: JSON.stringify({ action }),
      }).catch(() =>
        setSendError("control server unreachable — is the training script running?"),
      );
    },
    [live],
  );

  const toggle = (
    <button style={S.toggle} className="td-btn" onClick={onToggle}>
      {open ? "▾ hide analytics" : "▴ analytics"}
    </button>
  );

  if (!open) return <div style={{ ...S.dock, padding: 0, background: "none", border: "none" }}>{toggle}</div>;

  // With no run attached the dock still earns its space: it says so plainly
  // and shows what the *model* is, rather than rendering an empty chart frame.
  if (!live) {
    const m = parsed.spec.model;
    const cells = parsed.spec.lattice.cells;
    return (
      <div style={S.dock}>
        {toggle}
        <Stat label="model" value={m.kind} />
        <Stat label="parameters" value={m.n_params.toLocaleString()} />
        <Stat label="layers" value={m.n_layers} />
        <Stat
          label="cells"
          value={`${cells.present}/${cells.total}`}
          color={cells.dense ? "#d6dbe4" : "#e0af68"}
        />
        <div style={{ flex: 1 }} />
        <div style={{ alignSelf: "center", color: "#5d6b84", fontSize: 12.5 }}>
          no run attached — start one with{" "}
          <code style={{ color: "#8b95a8" }}>examples/viewer_live.py</code> and
          metrics stream in here
        </div>
      </div>
    );
  }

  const st = STATUS[live.status] ?? STATUS.stopped;
  const metrics = live.metrics ?? [];
  const last = metrics.length ? metrics[metrics.length - 1] : null;
  const lastHeld = [...metrics].reverse().find((e) => e.held_out != null);
  const pct = live.total_steps && last ? ((last.step + 1) / live.total_steps) * 100 : 0;

  return (
    <div style={S.dock}>
      {toggle}
      <div style={{ minWidth: 168 }}>
        <div style={S.statLabel}>
          <span className={live.status === "training" ? "td-live" : undefined}>
            ● </span>
          {st.label}
        </div>
        <div style={{ ...S.statValue, color: st.color, fontSize: 15 }}>
          {last ? `${last.step + 1} / ${live.total_steps}` : "—"}
        </div>
        <div
          style={{
            height: 5,
            borderRadius: 99,
            background: "#0c1018",
            border: "1px solid #232d40",
            overflow: "hidden",
            marginTop: 6,
          }}
        >
          <div
            style={{
              height: "100%",
              width: `${pct}%`,
              background: st.color,
              transition: "width 0.4s",
            }}
          />
        </div>
        <div style={{ marginTop: 8, display: "flex", gap: 6, flexWrap: "wrap" }}>
          <span style={S.chip}>{live.device}</span>
          {live.task && (
            <span style={{ ...S.chip, background: "#1b2437" }}>{live.task}</span>
          )}
        </div>
      </div>

      <Stat
        label="train loss"
        value={last ? last.loss.toFixed(5) : "—"}
        color="#e8963a"
      />
      <Stat
        label="held-out"
        value={lastHeld ? lastHeld.held_out.toFixed(5) : "—"}
        color="#7aa2f7"
      />
      {/* Throughput, and the clock the sweep animation runs on: one pass over
          every layer is one training step, so the wavefront moves at the rate
          the model is actually training. */}
      <Stat
        label="steps/s"
        value={stepRate ? stepRate.toFixed(1) : "—"}
        color="#9ece6a"
      />

      <div ref={chartRef} style={{ flex: 1, minWidth: 0 }}>
        {metrics.length > 1 ? (
          <LossChart metrics={metrics} width={chartWidth} height={92} />
        ) : (
          <div style={{ color: "#5d6b84", fontSize: 12.5, paddingTop: 30 }}>
            waiting for the first steps…
          </div>
        )}
      </div>

      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 7,
          justifyContent: "center",
        }}
      >
        {live.status === "waiting" && (
          <button style={S.btnStart} className="td-btn" onClick={() => post("start")}>
            ▶ start training
          </button>
        )}
        {(live.status === "training" || live.status === "paused") && (
          <>
            <button
              style={S.btn}
              className="td-btn"
              onClick={() => post(live.status === "training" ? "pause" : "resume")}
            >
              {live.status === "training" ? "⏸ pause" : "▶ resume"}
            </button>
            <button style={S.btnStop} className="td-btn" onClick={() => post("stop")}>
              ■ stop
            </button>
          </>
        )}
        {sendError && (
          <div style={{ color: "#fca5a5", fontSize: 11, maxWidth: 150 }}>
            {sendError}
          </div>
        )}
      </div>
    </div>
  );
}
