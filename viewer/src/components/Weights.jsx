import { useMemo } from "react";

// The weights, drawn as the mechanism rather than as a heatmap of everything.
// A linear map is a bipartite graph of units; a convolution is a set of taps
// over a receptive field; an SSM is a bank of states with a decay each, an
// input map and an output map. Those are three different pictures because they
// are three different mechanisms, and rendering all of them as coloured
// rectangles would throw away the only thing the diagram is for.

const POS = "#ffc061"; // a positive weight
const NEG = "#6d9eff"; // a negative one
const DIM = "#4a5878";

const S = {
  card: {
    background: "#101725",
    border: "1px solid #1f2a3d",
    borderRadius: 8,
    padding: "9px 11px",
    marginBottom: 9,
  },
  head: {
    display: "flex",
    alignItems: "baseline",
    gap: 7,
    flexWrap: "wrap",
    marginBottom: 6,
  },
  name: { fontSize: 12, color: "#c3cdde", fontWeight: 600 },
  role: {
    fontSize: 10,
    padding: "1px 6px",
    borderRadius: 99,
    background: "#1c2740",
    color: "#9fb3d9",
    letterSpacing: 0.4,
  },
  shape: { fontSize: 11, color: "#6b7a96", fontVariantNumeric: "tabular-nums" },
  note: { fontSize: 10.5, color: "#5d6b84", marginTop: 5, lineHeight: 1.45 },
  empty: {
    color: "#5d6b84",
    fontSize: 12.5,
    padding: "18px 4px",
    lineHeight: 1.5,
  },
};

const colour = (v, absmax) => {
  const t = absmax ? Math.min(1, Math.abs(v) / absmax) : 0;
  return {
    stroke: v >= 0 ? POS : NEG,
    opacity: 0.08 + 0.9 * t ** 0.7,
    width: 0.4 + 1.6 * t,
  };
};

// --- a matrix, drawn as a matrix --------------------------------------------
// A dense map's structure is "every entry is free", and the way to show that is
// every entry. Drawing it as a bipartite graph means hundreds of crossing
// lines: a hairball that reads as one grey smudge, which is neither pretty nor
// informative. Lines are kept only for maps small enough that the wires are
// individually visible.
function Matrix({ values, absmax, width = 250, maxCell = 12, gamma = 0.6 }) {
  const rows = values.length;
  const cols = values[0]?.length ?? 0;
  const cell = Math.max(
    2,
    Math.min(maxCell, Math.floor(width / Math.max(cols, 1))),
  );
  const w = cols * cell;
  const h = rows * cell;
  return (
    <svg
      width="100%"
      viewBox={`0 0 ${w} ${h}`}
      shapeRendering="crispEdges"
      style={{ display: "block", maxHeight: 260 }}
    >
      {values.map((row, r) =>
        row.map((v, c) => {
          const t = absmax ? Math.min(1, Math.abs(v) / absmax) : 0;
          return (
            <rect
              key={`${r}-${c}`}
              x={c * cell}
              y={r * cell}
              width={cell}
              height={cell}
              fill={v >= 0 ? POS : NEG}
              opacity={0.04 + 0.96 * t ** gamma}
            />
          );
        }),
      )}
    </svg>
  );
}

// Small enough that individual wires are legible — the picture "everything is
// connected to everything" only lands when you can count the wires.
function WireDiagram({ tensor }) {
  const rows = tensor.values.length; // output units
  const cols = tensor.values[0]?.length ?? 0; // input units
  const absmax = Math.max(1e-9, tensor.stats.absmax);
  const h = Math.max(120, Math.max(rows, cols) * 9);
  const w = 250;
  const x0 = 34;
  const x1 = w - 34;
  const yOf = (i, n) => (n <= 1 ? h / 2 : 12 + (i * (h - 24)) / (n - 1));

  const edges = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = tensor.values[r][c];
      const { stroke, opacity, width } = colour(v, absmax);
      if (opacity < 0.12) continue; // below this it is a smudge, not a weight
      edges.push(
        <line
          key={`${r}-${c}`}
          x1={x0}
          y1={yOf(c, cols)}
          x2={x1}
          y2={yOf(r, rows)}
          stroke={stroke}
          strokeOpacity={opacity}
          strokeWidth={width}
        />,
      );
    }
  }

  return (
    <svg width="100%" viewBox={`0 0 ${w} ${h}`} style={{ display: "block" }}>
      {edges}
      {Array.from({ length: cols }, (_, c) => (
        <circle key={`i${c}`} cx={x0} cy={yOf(c, cols)} r={2.6} fill={DIM} />
      ))}
      {Array.from({ length: rows }, (_, r) => (
        <circle
          key={`o${r}`}
          cx={x1}
          cy={yOf(r, rows)}
          r={2.6}
          fill="#8fa3c8"
        />
      ))}
      <text x={x0} y={h - 1} fill="#5d6b84" fontSize="8.5" textAnchor="middle">
        in
      </text>
      <text x={x1} y={h - 1} fill="#5d6b84" fontSize="8.5" textAnchor="middle">
        out
      </text>
    </svg>
  );
}

// --- a convolution: the same taps applied everywhere -------------------------
function ConvDiagram({ tensor }) {
  const taps = tensor.values[0]?.length ?? 0;
  const absmax = Math.max(1e-9, tensor.stats.absmax);
  const row = tensor.values[0] ?? [];
  const w = 250;
  const h = 108;
  const y0 = 22;
  const y1 = h - 30;
  const xOf = (i) => (taps <= 1 ? w / 2 : 26 + (i * (w - 52)) / (taps - 1));

  return (
    <svg width="100%" viewBox={`0 0 ${w} ${h}`} style={{ display: "block" }}>
      {row.map((v, i) => {
        const { stroke, opacity, width } = colour(v, absmax);
        return (
          <line
            key={i}
            x1={xOf(i)}
            y1={y0}
            x2={w / 2}
            y2={y1}
            stroke={stroke}
            strokeOpacity={Math.max(0.18, opacity)}
            strokeWidth={width}
          />
        );
      })}
      {row.map((v, i) => (
        <circle key={`t${i}`} cx={xOf(i)} cy={y0} r={3} fill={DIM} />
      ))}
      <circle cx={w / 2} cy={y1} r={4.5} fill="#8fa3c8" />
      <text x={w / 2} y={12} fill="#5d6b84" fontSize="8.5" textAnchor="middle">
        receptive field — {tensor.shape[tensor.shape.length - 1]} taps
      </text>
      <text
        x={w / 2}
        y={h - 6}
        fill="#5d6b84"
        fontSize="8.5"
        textAnchor="middle"
      >
        one output position
      </text>
    </svg>
  );
}

// --- an SSM: states that decay, with an input and an output map --------------
// Assembled from whichever of the roles the layer actually has, so an S4D
// (decay + C + D) and a Mamba (decay + D + projections) each draw the parts
// they really own instead of a common denominator that flatters both.
function SSMDiagram({ decay, input, output, skip }) {
  const src = decay ?? input ?? output;
  if (!src) return null;
  const flat = src.values.flat();
  const n = Math.min(12, flat.length);
  const w = 250;
  const h = 132;
  const cx = w / 2;
  const yOf = (i) => (n <= 1 ? h / 2 : 26 + (i * (h - 62)) / (n - 1));

  // A decay parameter is stored as a log; what matters visually is how long a
  // state remembers, so it is mapped to a 0..1 retention rather than drawn raw.
  const decays = decay
    ? decay.values
        .flat()
        .slice(0, n)
        .map((v) => Math.exp(-Math.exp(Math.min(6, v))))
    : new Array(n).fill(0.5);
  const inVals = input ? input.values.flat() : null;
  const outVals = output ? output.values.flat() : null;
  const inMax = input ? Math.max(1e-9, input.stats.absmax) : 1;
  const outMax = output ? Math.max(1e-9, output.stats.absmax) : 1;

  return (
    <svg width="100%" viewBox={`0 0 ${w} ${h}`} style={{ display: "block" }}>
      <text x={22} y={12} fill="#5d6b84" fontSize="8.5" textAnchor="middle">
        x
      </text>
      <text x={w - 22} y={12} fill="#5d6b84" fontSize="8.5" textAnchor="middle">
        y
      </text>
      {Array.from({ length: n }, (_, i) => {
        const y = yOf(i);
        const ret = decays[i];
        const bIn = inVals ? colour(inVals[i % inVals.length], inMax) : null;
        const cOut = outVals
          ? colour(outVals[i % outVals.length], outMax)
          : null;
        return (
          <g key={i}>
            <line
              x1={22}
              y1={h / 2}
              x2={cx - 9}
              y2={y}
              stroke={bIn ? bIn.stroke : DIM}
              strokeOpacity={bIn ? bIn.opacity : 0.35}
              strokeWidth={bIn ? bIn.width : 0.7}
            />
            <line
              x1={cx + 9}
              y1={y}
              x2={w - 22}
              y2={h / 2}
              stroke={cOut ? cOut.stroke : DIM}
              strokeOpacity={cOut ? cOut.opacity : 0.35}
              strokeWidth={cOut ? cOut.width : 0.7}
            />
            {/* the self-loop: how much of this state survives a step */}
            <path
              d={`M ${cx - 7} ${y - 4} A 8 8 0 1 1 ${cx + 7} ${y - 4}`}
              fill="none"
              stroke="#9b7cff"
              strokeOpacity={0.2 + 0.8 * ret}
              strokeWidth={0.6 + 1.8 * ret}
            />
            <circle cx={cx} cy={y} r={3.4} fill="#8fa3c8" />
          </g>
        );
      })}
      <circle cx={22} cy={h / 2} r={4.5} fill={DIM} />
      <circle cx={w - 22} cy={h / 2} r={4.5} fill="#8fa3c8" />
      {skip && (
        <>
          <path
            d={`M 22 ${h / 2 + 8} Q ${cx} ${h - 6} ${w - 22} ${h / 2 + 8}`}
            fill="none"
            stroke={POS}
            strokeOpacity={0.5}
            strokeWidth={1.1}
            strokeDasharray="3 3"
          />
          <text
            x={cx}
            y={h - 2}
            fill="#6b7a96"
            fontSize="8"
            textAnchor="middle"
          >
            D skip
          </text>
        </>
      )}
      <text x={cx} y={12} fill="#5d6b84" fontSize="8.5" textAnchor="middle">
        {n} of {src.shape.reduce((a, b) => a * b, 1)} states · loop = retention
      </text>
    </svg>
  );
}

// --- anything with no shape worth drawing ------------------------------------
function StripDiagram({ tensor }) {
  const flat = tensor.values.flat();
  const absmax = Math.max(1e-9, tensor.stats.absmax);
  const n = Math.min(flat.length, 48);
  const w = 250;
  const h = 26;
  const bw = w / n;
  return (
    <svg width="100%" viewBox={`0 0 ${w} ${h}`} style={{ display: "block" }}>
      {flat.slice(0, n).map((v, i) => {
        const t = Math.min(1, Math.abs(v) / absmax);
        return (
          <rect
            key={i}
            x={i * bw}
            y={h / 2 - (t * h) / 2}
            width={Math.max(1, bw - 0.8)}
            height={Math.max(1.5, t * h)}
            fill={v >= 0 ? POS : NEG}
            opacity={0.25 + 0.7 * t}
          />
        );
      })}
    </svg>
  );
}

function LinearDiagram({ tensor }) {
  const rows = tensor.values.length;
  const cols = tensor.values[0]?.length ?? 0;
  if (Math.max(rows, cols) <= 12) return <WireDiagram tensor={tensor} />;
  return (
    <Matrix
      values={tensor.values}
      absmax={Math.max(1e-9, tensor.stats.absmax)}
    />
  );
}

// --- the one picture every family shares ------------------------------------
// Each family is a map over positions along the swept axis; what differs is the
// structure that map is *forced* to have. Side by side on the same axes, the
// difference is the picture: a convolution is a narrow band repeated down every
// diagonal, an SSM is lower-triangular and fading, a dense map fills the
// square. Nothing here is inferred from the class name — it is measured off the
// mixer's own impulse response.
function OperatorPanel({ operator, mixer }) {
  if (!operator) return null;
  const { causal, bandwidth, tied, size } = operator;

  const notes = [];
  if (causal >= 0.99) {
    notes.push("causal — nothing reaches backwards in the sweep");
  } else if (causal <= 0.6) {
    notes.push(
      `bidirectional — ${Math.round(100 * (1 - causal))}% of the influence runs backwards`,
    );
  } else {
    notes.push(`mostly causal (${Math.round(100 * causal)}% of the influence)`);
  }
  if (bandwidth === 0) {
    notes.push(
      /attention/i.test(mixer ?? "")
        ? "no fixed off-diagonal structure: attention's mixing is computed from the data, so it is not in the parameters to draw"
        : "diagonal — this layer maps each position to itself",
    );
  } else if (bandwidth < size - 1) {
    notes.push(`banded ±${bandwidth} — a position only reaches that far`);
  } else {
    notes.push("dense — every position reaches every other");
  }
  if (tied >= 0.9 && bandwidth > 0) {
    notes.push(
      "weights tied along each diagonal: the same kernel, repeated at every position",
    );
  } else if (tied < 0.6) {
    notes.push("untied — every position pair has its own weight");
  }

  return (
    <div style={S.card}>
      <div style={S.head}>
        <span style={S.name}>position → position</span>
        <span style={S.role}>impulse response</span>
        <span style={S.shape}>
          {size} × {size}
        </span>
      </div>
      {/* A steep curve on purpose: an SSM's causal tail is a couple of percent
          of its local term, so a linear ramp renders the one structure worth
          seeing as black. Stated below rather than left as a flattering
          default. */}
      <Matrix
        values={operator.values}
        absmax={Math.max(1e-9, operator.absmax)}
        maxCell={14}
        gamma={0.35}
      />
      <div style={S.note}>
        Row <i>i</i> is what an impulse at position <i>j</i> does to position{" "}
        <i>i</i>. {notes.join(" · ")}.
        {operator.reach != null && (
          <>
            {" "}
            Influence away from the diagonal is{" "}
            <b style={{ color: "#9fb0cc" }}>
              {(100 * operator.reach).toFixed(1)}%
            </b>{" "}
            of the local term. Shading is |value|<sup>0.35</sup>, so a small
            tail stays visible.
          </>
        )}
      </div>
    </div>
  );
}

function TensorCard({ tensor }) {
  const body =
    tensor.role === "linear" ? (
      <LinearDiagram tensor={tensor} />
    ) : tensor.role === "conv" ? (
      <ConvDiagram tensor={tensor} />
    ) : (
      <StripDiagram tensor={tensor} />
    );

  return (
    <div style={S.card}>
      <div style={S.head}>
        <span style={S.name}>{tensor.name}</span>
        <span style={S.role}>{tensor.role}</span>
        <span style={S.shape}>{tensor.shape.join(" × ")}</span>
      </div>
      {body}
      <div style={S.note}>
        {tensor.sampled
          ? `drawn as a ${tensor.rows}×${tensor.cols} stride-${tensor.stride.join("/")} sample of ${tensor.stats.n.toLocaleString()} weights`
          : `all ${tensor.stats.n.toLocaleString()} weights drawn`}
        {" · "}μ {tensor.stats.mean} · σ {tensor.stats.std} · |max|{" "}
        {tensor.stats.absmax}
      </div>
    </div>
  );
}

export default function Weights({ weights, layerIndex }) {
  const layer = useMemo(() => {
    if (!weights?.layers?.length) return null;
    return (
      weights.layers.find((l) => l.layer === layerIndex) ??
      weights.layers[Math.min(layerIndex, weights.layers.length - 1)]
    );
  }, [weights, layerIndex]);

  if (!weights) {
    return (
      <div style={S.empty}>
        No weights to draw. The viewer was opened on a spec — a saved
        architecture, not a model — so there are no parameters to read. Open it
        with <code style={{ color: "#8b95a8" }}>td.viz.show(model)</code> to see
        them.
      </div>
    );
  }
  if (!layer) return <div style={S.empty}>this layer has no parameters</div>;

  // The SSM parts are one mechanism, so they are drawn as one figure; the rest
  // of the tensors keep their own cards.
  const by = (role) => layer.tensors.find((t) => t.role === role);
  const ssm = {
    decay: by("ssm_decay"),
    input: by("ssm_in"),
    output: by("ssm_out"),
    skip: by("skip"),
  };
  const hasSSM = !!(ssm.decay || ssm.input || ssm.output);
  const drawn = new Set(
    hasSSM
      ? [ssm.decay, ssm.input, ssm.output, ssm.skip]
          .filter(Boolean)
          .map((t) => t.name)
      : [],
  );

  return (
    <div>
      <OperatorPanel operator={layer.operator} mixer={layer.mixer} />
      {hasSSM && (
        <div style={S.card}>
          <div style={S.head}>
            <span style={S.name}>state-space recurrence</span>
            <span style={S.role}>ssm</span>
          </div>
          <SSMDiagram {...ssm} />
          <div style={S.note}>
            Each circle is a state with its own decay: the loop shows how much
            of it survives one step. Edges in are B, edges out are C.
          </div>
        </div>
      )}
      {layer.tensors
        .filter((t) => !drawn.has(t.name))
        .map((t) => (
          <TensorCard key={t.name} tensor={t} />
        ))}
    </div>
  );
}
