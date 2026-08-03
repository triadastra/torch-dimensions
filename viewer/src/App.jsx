import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import Analytics from "./components/Analytics.jsx";
import ModelView from "./components/ModelView.jsx";
import Scene from "./components/Scene.jsx";
import Sidebar from "./components/Sidebar.jsx";
import cafaHybrid from "./samples/cafa_hybrid.json";
import lstm2d from "./samples/lstm_2d_sparse.json";
import mamba3d from "./samples/mamba_3d.json";
import s4d4d from "./samples/s4d_4d.json";
import vitJoint from "./samples/vit_joint.json";
import { parseSpec } from "./spec.js";

// Regenerate with `python viewer/make_samples.py`.
const SAMPLES = {
  "LSTM · 2-D sparse lattice": lstm2d,
  "Mamba-ND · 3-D, paired schedule": mamba3d,
  "S4D · 4-D (dimensional stacking)": s4d4d,
  "CaFA · kernel family (no wavefront)": cafaHybrid,
  "ViT · joint attention over patches": vitJoint,
};

// Seconds per layer with no run attached. With one, the sweep is clocked to
// the training itself — see `layerSeconds` below.
const LAYER_SECONDS = 2.2;
// A full pass over the layers is one training step, so the animation can run
// at the training's own rate. Clamped at both ends: a fast step rate turns the
// sweep into a strobe that shows nothing, and a slow one into a still image.
const MIN_LAYER = 0.05;
const MAX_LAYER = 4.0;
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
  const [tab, setTab] = useState("model");
  const [dataShow, setDataShow] = useState(false);
  const [dockOpen, setDockOpen] = useState(true);
  // Measured throughput, shown in the dock. Also what the sweep is clocked to.
  const [stepRate, setStepRate] = useState(null);
  // Served by td.viz.show(model); absent when the viewer was opened on a
  // spec, which has no parameters to read.
  const [weights, setWeights] = useState(null);
  // A run streams its weights as they train; those win over the snapshot
  // taken when the page was opened, which is stale the moment training starts.
  const liveWeights =
    sampleKey === LIVE_KEY && live?.weights ? live.weights : null;
  const liveRunId = useRef(null);

  // progress is animation state, not UI state: it lives in a ref the render
  // loop mutates, so 60fps never re-renders the React tree.
  const anim = useRef({ layer: 0, progress: 0 });
  anim.current.layer = layerIndex;

  // Measured seconds per training step, and the per-layer duration derived
  // from it. Both live in refs: the render loop reads them every frame, and
  // re-running the effect on every metrics poll would restart the animation.
  const stepClock = useRef({ step: -1, at: 0, seconds: null });
  const layerSeconds = useRef(LAYER_SECONDS);

  const nLayers = parsed.spec.layers.length;
  const parsedRef = useRef(parsed);
  parsedRef.current = parsed;

  useEffect(() => {
    if (!playing) return undefined;
    let raf;
    let last = performance.now();
    const tick = (now) => {
      const dt = (now - last) / 1000;
      last = now;
      anim.current.progress += dt / layerSeconds.current;
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
    fetch("/weights.json", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((w) => !cancelled && w && setWeights(w))
      .catch(() => {});
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
        const r = await fetch(`/run.json?t=${Date.now()}`, {
          cache: "no-store",
        });
        if (!r.ok) return;
        const j = await r.json();
        setLive(j);

        // Clock the sweep to the run. One pass over every layer is one
        // training step, so the wavefront moves at the rate the model is
        // actually being trained rather than at a decorative constant.
        const last = j.metrics?.length ? j.metrics[j.metrics.length - 1] : null;
        const clock = stepClock.current;
        if (last && j.status === "training") {
          const now = performance.now();
          if (clock.step >= 0 && last.step > clock.step) {
            const per = (now - clock.at) / 1000 / (last.step - clock.step);
            // Smoothed: a single slow poll should not visibly jerk the sweep.
            clock.seconds = clock.seconds
              ? clock.seconds * 0.6 + per * 0.4
              : per;
          }
          clock.step = last.step;
          clock.at = now;
        } else {
          clock.step = -1;
          clock.seconds = null;
        }
        const layers = Math.max(1, parsedRef.current.spec.layers.length);
        layerSeconds.current = clock.seconds
          ? Math.min(MAX_LAYER, Math.max(MIN_LAYER, clock.seconds / layers))
          : LAYER_SECONDS;
        setStepRate(clock.seconds ? 1 / clock.seconds : null);
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

  const scene = useMemo(
    () => (
      <Scene
        parsed={parsed}
        anim={anim}
        dataShow={dataShow}
        cellData={sampleKey === LIVE_KEY ? live?.cells : null}
      />
    ),
    [parsed, dataShow, sampleKey, live],
  );

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
        anim={anim}
        tab={tab}
        onTab={setTab}
        dataShow={dataShow}
        onToggleData={() => setDataShow((d) => !d)}
      />
      {/* The main area is split: what the model does to the lattice on the
          left, what the model is on the right, and the run's analytics across
          the bottom of both. */}
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          minWidth: 0,
        }}
      >
        <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
          <div
            style={{
              flex: 1,
              position: "relative",
              minWidth: 0,
              // The canvas is alpha; this is the sky behind it. A flat fill made
              // the far cubes sit on nothing once fog took their contrast away.
              background:
                "radial-gradient(120% 90% at 62% 32%, #16203a 0%, #0d1220 45%, #070a10 100%)",
            }}
          >
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
          <ModelView
            parsed={parsed}
            layerIndex={layerIndex}
            onSelectLayer={selectLayer}
            weights={liveWeights ?? weights}
          />
        </div>
        <Analytics
          live={sampleKey === LIVE_KEY ? live : null}
          parsed={parsed}
          stepRate={stepRate}
          open={dockOpen}
          onToggle={() => setDockOpen((o) => !o)}
        />
      </div>
    </div>
  );
}
