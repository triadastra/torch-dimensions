import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import Scene from "./components/Scene.jsx";
import Sidebar from "./components/Sidebar.jsx";
import lstm2d from "./samples/lstm_2d_sparse.json";
import mamba3d from "./samples/mamba_3d.json";
import s4d4d from "./samples/s4d_4d.json";
import { parseSpec } from "./spec.js";

const SAMPLES = {
  "LSTM · 2-D sparse lattice": lstm2d,
  "Mamba-ND · 3-D, paired schedule": mamba3d,
  "S4D · 4-D (dimensional stacking)": s4d4d,
};

const LAYER_SECONDS = 2.2;
export const LIVE_KEY = "● live run";
export const SHOWN_KEY = "● this model";

export default function App() {
  const [sampleKey, setSampleKey] = useState(Object.keys(SAMPLES)[0]);
  const [parsed, setParsed] = useState(() => parseSpec(SAMPLES[sampleKey]));
  const [layerIndex, setLayerIndex] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [error, setError] = useState(null);
  const [live, setLive] = useState(null);
  const [shown, setShown] = useState(null);
  const liveRunId = useRef(null);

  // progress is animation state, not UI state: it lives in a ref the render
  // loop mutates, so 60fps never re-renders the React tree.
  const anim = useRef({ layer: 0, progress: 0 });
  anim.current.layer = layerIndex;

  const nLayers = parsed.spec.layers.length;

  useEffect(() => {
    if (!playing) return undefined;
    let raf;
    let last = performance.now();
    const tick = (now) => {
      const dt = (now - last) / 1000;
      last = now;
      anim.current.progress += dt / LAYER_SECONDS;
      if (anim.current.progress >= 1) {
        anim.current.progress = 0;
        setLayerIndex((i) => (i + 1) % nLayers);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, nLayers]);

  const selectLayer = useCallback(
    (i) => {
      anim.current.progress = 0;
      setLayerIndex(((i % nLayers) + nLayers) % nLayers);
    },
    [nLayers],
  );

  const loadSpec = useCallback((json) => {
    try {
      const p = parseSpec(json);
      setParsed(p);
      setLayerIndex(0);
      anim.current.progress = 0;
      setError(null);
    } catch (e) {
      setError(String(e.message ?? e));
    }
  }, []);

  const onSample = useCallback(
    (key) => {
      setSampleKey(key);
      if (key === SHOWN_KEY) loadSpec(shown);
      else if (key !== LIVE_KEY) loadSpec(SAMPLES[key]);
    },
    [loadSpec, shown],
  );

  // `td.viz.show(model)` serves the bundle with the model's spec at /spec.json.
  // Fetched once at startup: when it is there, it is what the user asked to
  // see, so it selects itself. When it is not — the dev server, a static host —
  // the samples are the whole app and nothing about this path is visible.
  useEffect(() => {
    let cancelled = false;
    fetch("/spec.json", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((json) => {
        if (cancelled || !json) return;
        setShown(json);
        setSampleKey(SHOWN_KEY);
        loadSpec(json);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [loadSpec]);

  // Live-run mode: a training script (examples/viewer_live.py) writes
  // public/run.json after every optimizer step. Poll it; a new `started`
  // stamp means a new run, which auto-selects the live entry and loads its
  // spec once — metrics keep flowing without re-parsing the architecture.
  useEffect(() => {
    const id = setInterval(async () => {
      try {
        const r = await fetch(`/run.json?t=${Date.now()}`, { cache: "no-store" });
        if (!r.ok) return;
        const j = await r.json();
        setLive(j);
        if (j.started !== liveRunId.current) {
          liveRunId.current = j.started;
          setSampleKey(LIVE_KEY);
          loadSpec(j.spec);
        }
      } catch {
        /* no run.json yet — the dropdown simply has no live entry */
      }
    }, 700);
    return () => clearInterval(id);
  }, [loadSpec]);

  const onFile = useCallback(
    (file) => {
      file.text().then((t) => {
        try {
          loadSpec(JSON.parse(t));
        } catch (e) {
          setError(String(e.message ?? e));
        }
      });
    },
    [loadSpec],
  );

  const scene = useMemo(() => <Scene parsed={parsed} anim={anim} />, [parsed]);

  return (
    <div style={{ display: "flex", height: "100%" }}>
      <Sidebar
        parsed={parsed}
        layerIndex={layerIndex}
        playing={playing}
        onTogglePlay={() => setPlaying((p) => !p)}
        onSelectLayer={selectLayer}
        samples={{
          ...(live ? { [LIVE_KEY]: null } : {}),
          ...(shown ? { [SHOWN_KEY]: null } : {}),
          ...SAMPLES,
        }}
        sampleKey={sampleKey}
        onSample={onSample}
        onFile={onFile}
        live={sampleKey === LIVE_KEY ? live : null}
      />
      <div style={{ flex: 1, position: "relative" }}>
        {scene}
        {error && (
          <div
            style={{
              position: "absolute",
              top: 14,
              left: 14,
              background: "#2a1418",
              border: "1px solid #7f1d1d",
              color: "#fca5a5",
              borderRadius: 8,
              padding: "8px 14px",
            }}
          >
            {error}
          </div>
        )}
      </div>
    </div>
  );
}
