import { useEffect, useMemo, useRef } from "react";

import { fmtParams, latticeAxisOf } from "../spec.js";

// The right half: what the model *is*, next to what it does to the lattice.
// The 3-D view answers "which cells, in what order"; this answers "through
// what, with how many weights". Same spec, same active layer — the two halves
// are one picture cut down the middle, not two tools sharing a window.

const FAMILY = {
  scan: { color: "#ffc061", glyph: "→" },
  kernel: { color: "#c39bd8", glyph: "⊗" },
  flatten: { color: "#5eead4", glyph: "⊕" },
};

const S = {
  wrap: {
    width: "44%",
    minWidth: 300,
    display: "flex",
    flexDirection: "column",
    borderLeft: "1px solid #1e2635",
    background: "linear-gradient(180deg, #0e1320 0%, #0b0f18 100%)",
  },
  head: { padding: "12px 16px 10px", borderBottom: "1px solid #1a2233" },
  title: {
    color: "#5d6b84",
    fontSize: 10.5,
    fontWeight: 700,
    letterSpacing: 1.4,
    textTransform: "uppercase",
  },
  io: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    marginTop: 7,
    fontSize: 12,
    color: "#9fb0cc",
    fontVariantNumeric: "tabular-nums",
    flexWrap: "wrap",
  },
  pill: {
    background: "#151d2e",
    border: "1px solid #253048",
    borderRadius: 6,
    padding: "3px 8px",
  },
  body: { flex: 1, overflowY: "auto", padding: "10px 16px 16px" },
  node: (active, color) => ({
    position: "relative",
    display: "flex",
    alignItems: "center",
    gap: 9,
    padding: "7px 10px",
    borderRadius: 8,
    fontSize: 12.5,
    cursor: "pointer",
    background: active ? "#1a2438" : "#121926",
    border: `1px solid ${active ? color : "#1f2a3d"}`,
    boxShadow: active ? `0 0 0 1px ${color}55, inset 2px 0 0 ${color}` : "none",
  }),
  bar: {
    position: "relative",
    height: 4,
    borderRadius: 99,
    background: "#0c1018",
    overflow: "hidden",
    marginTop: 5,
  },
  rail: {
    width: 1,
    height: 12,
    background: "#243049",
    margin: "0 auto",
  },
  chip: {
    fontSize: 10.5,
    padding: "1px 6px",
    borderRadius: 99,
    background: "#1c2740",
    color: "#9fb3d9",
    whiteSpace: "nowrap",
  },
  mech: {
    display: "flex",
    gap: 6,
    flexWrap: "wrap",
    marginTop: 8,
  },
};

function Node({ layer, i, spec, active, maxParams, onSelect }) {
  const ref = useRef();
  const kind = layer.kind ?? "scan";
  const fam = FAMILY[kind] ?? FAMILY.scan;
  const isTime = kind === "scan" && latticeAxisOf(layer, spec) === null;

  // Keep the active layer in view as the animation advances, but never yank
  // the panel while the user is reading somewhere else — only scroll when the
  // node is actually outside the viewport.
  useEffect(() => {
    if (active && ref.current) {
      const el = ref.current;
      const box = el.getBoundingClientRect();
      const par = el.parentElement.parentElement.getBoundingClientRect();
      if (box.top < par.top || box.bottom > par.bottom) {
        el.scrollIntoView({ block: "nearest", behavior: "smooth" });
      }
    }
  }, [active]);

  const label =
    kind === "flatten"
      ? (layer.axes ?? []).join(" ⊕ ")
      : kind === "kernel"
        ? (layer.contracted ?? []).join(" ⊗ ")
        : layer.axis;
  const detail =
    kind === "flatten"
      ? layer.tokens
        ? `${layer.tokens} tokens`
        : ""
      : kind === "kernel"
        ? layer.axis
          ? `+ ${layer.axis} →`
          : ""
        : isTime
          ? "→ causal"
          : layer.reverse
            ? "←"
            : "→";

  const pct = maxParams ? (100 * (layer.n_params ?? 0)) / maxParams : 0;

  return (
    <div>
      <div
        ref={ref}
        className="td-row"
        style={S.node(active, fam.color)}
        onClick={() => onSelect(i)}
      >
        <span style={{ color: "#6b7a96", width: 26, flexShrink: 0 }}>L{i}</span>
        <span style={{ color: fam.color, width: 14, flexShrink: 0 }}>{fam.glyph}</span>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div
            style={{
              display: "flex",
              alignItems: "baseline",
              gap: 6,
              whiteSpace: "nowrap",
              overflow: "hidden",
            }}
          >
            <b style={{ color: active ? "#e8eef8" : "#c3cdde" }}>{label}</b>
            <span style={{ color: "#6b7a96", fontSize: 11.5 }}>{detail}</span>
          </div>
          <div style={S.bar}>
            <div
              style={{
                height: "100%",
                width: `${pct}%`,
                background: fam.color,
                opacity: active ? 0.95 : 0.45,
                transition: "opacity 160ms ease",
              }}
            />
          </div>
        </div>
        <div style={{ textAlign: "right", flexShrink: 0 }}>
          {layer.mixer && <div style={S.chip}>{layer.mixer}</div>}
          <div
            style={{
              color: "#6b7a96",
              fontSize: 11,
              marginTop: 3,
              fontVariantNumeric: "tabular-nums",
            }}
          >
            {fmtParams(layer.n_params ?? 0)}
          </div>
        </div>
      </div>
      <div style={S.rail} />
    </div>
  );
}

export default function ModelView({ parsed, layerIndex, onSelectLayer }) {
  const { spec } = parsed;
  const m = spec.model;
  const io = spec.io ?? {};

  const maxParams = useMemo(
    () => Math.max(1, ...spec.layers.map((l) => l.n_params ?? 0)),
    [spec],
  );
  // Which distinct mechanisms this model is actually built from — the mixers
  // in play plus the composition, rather than a class name that hides both.
  const mechanisms = useMemo(() => {
    const mixers = [...new Set(spec.layers.map((l) => l.mixer).filter(Boolean))];
    const kinds = [...new Set(spec.layers.map((l) => l.kind ?? "scan"))];
    return { mixers, kinds };
  }, [spec]);

  const swept = spec.layers.reduce((a, l) => a + (l.n_params ?? 0), 0);

  return (
    <div style={S.wrap}>
      <div style={S.head}>
        <div style={S.title}>model</div>
        <div style={S.io}>
          <span style={S.pill}>{(io.input ?? []).join(" × ") || "—"}</span>
          <span style={{ color: "#41568a" }}>▶</span>
          <b style={{ color: "#e8eef8" }}>{m.kind}</b>
          <span style={{ color: "#41568a" }}>▶</span>
          <span style={S.pill}>{(io.output ?? []).join(" × ") || "—"}</span>
        </div>
        <div style={S.mech}>
          <span style={S.chip}>{spec.nd_method.name}</span>
          {mechanisms.mixers.map((x) => (
            <span key={x} style={{ ...S.chip, color: "#ffc9a0" }}>
              {x}
            </span>
          ))}
          {mechanisms.kinds.map((k) => (
            <span key={k} style={{ ...S.chip, color: FAMILY[k]?.color ?? "#9fb3d9" }}>
              {FAMILY[k]?.glyph} {k}
            </span>
          ))}
          <span style={S.chip}>{fmtParams(m.n_params)} total</span>
          {swept !== m.n_params && (
            // Layer weights rarely add up to the whole model: projections,
            // norms and the head live outside the swept stack. Saying so beats
            // letting the two numbers quietly disagree.
            <span style={{ ...S.chip, color: "#7d8aa3" }}>
              {fmtParams(m.n_params - swept)} outside layers
            </span>
          )}
        </div>
      </div>

      <div style={S.body} className="td-panel">
        {spec.layers.map((l, i) => (
          <Node
            key={i}
            layer={l}
            i={i}
            spec={spec}
            active={i === layerIndex}
            maxParams={maxParams}
            onSelect={onSelectLayer}
          />
        ))}
      </div>
    </div>
  );
}
