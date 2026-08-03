import { Text } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";

import { latticeAxisOf } from "../spec.js";

// How many labels exist at once. Not "how many cells have labels" — every cell
// has one, but only the nearest POOL of them are drawn. A 4-D lattice can hold
// thousands of cells and each label is a text mesh with its own glyph layout;
// building one per cell drops the frame rate to single digits and produces an
// unreadable thicket besides. Distance decides who gets one.
const POOL = 56;
const RECHECK = 0.12; // seconds between reassignments
const CAM_STEP = 0.4; // or immediately, once the camera has moved this far
const MIN_SEP = 0.115; // NDC; closest two labels may sit before one is dropped

const _v = new THREE.Vector3();

export default function CellLabels({ parsed, layout, anim, enabled, cellData }) {
  const refs = useRef([]);
  const lastCam = useRef(new THREE.Vector3(Infinity, 0, 0));
  const since = useRef(0);
  const shown = useRef([]);

  // Axis names line up with `shape`, which is spatial only — `names` carries
  // the time axis in front of them when the lattice has one.
  const names = useMemo(() => {
    const all = parsed.spec.lattice.names ?? [];
    const spatial = all.slice(parsed.spec.lattice.time ? 1 : 0);
    return parsed.shape.map((_, i) => spatial[i] ?? `dim${i}`);
  }, [parsed]);

  const positions = useMemo(
    () => parsed.cells.map((c) => layout.position(c)),
    [parsed, layout],
  );

  const coordText = useMemo(
    () => parsed.cells.map((c) => names.map((n, i) => `${n} ${c[i]}`).join("\n")),
    [parsed, names],
  );

  // With a run attached the label carries what the cell actually holds this
  // step, under its coordinates. `cellData` arrives present-only and in the
  // same flat order parseSpec built `cells`, so the index is shared; a length
  // that disagrees means the run and the spec are out of step, and showing
  // numbers against the wrong cells would be worse than showing none.
  const aligned =
    cellData &&
    cellData.pred &&
    cellData.pred.length === parsed.cells.length;

  useFrame((state, dt) => {
    const pool = refs.current;
    if (!pool.length) return;

    if (!enabled) {
      // Hide once, then stop doing work every frame.
      if (shown.current.length) {
        pool.forEach((t) => t && (t.visible = false));
        shown.current = [];
      }
      return;
    }

    const cam = state.camera;
    since.current += dt;
    const moved = cam.position.distanceTo(lastCam.current);
    const restack = since.current > RECHECK || moved > CAM_STEP;

    if (restack) {
      since.current = 0;
      lastCam.current.copy(cam.position);
      // Nearest first...
      const scored = [];
      for (let i = 0; i < positions.length; i++) {
        const p = positions[i];
        _v.set(p[0], p[1], p[2]);
        scored.push([_v.distanceTo(cam.position), i]);
      }
      scored.sort((a, b) => a[0] - b[0]);

      // ...then thinned in *screen* space. Taking the nearest N by distance
      // alone piles them all into one corner of the frame and the result is a
      // thicket you cannot read a single line of. Skipping any candidate that
      // lands too close to one already placed keeps every drawn label legible,
      // and means zooming in reveals more of them — cells spread apart on
      // screen, so more of them clear the spacing test. Behind the camera is
      // skipped outright: a label back there is one spent on nothing.
      const kept = [];
      for (const [dist, i] of scored) {
        if (kept.length >= POOL) break;
        const p = positions[i];
        _v.set(p[0], p[1], p[2]).project(cam);
        if (_v.z > 1) continue;
        let clear = true;
        for (const k of kept) {
          if (Math.hypot((_v.x - k.x) * 0.55, _v.y - k.y) < MIN_SEP) {
            clear = false;
            break;
          }
        }
        if (clear) kept.push({ x: _v.x, y: _v.y, i, dist });
      }
      shown.current = kept.map((k) => [k.dist, k.i]);
    }

    // The layer's swept axis decides each label's tint, so a label says which
    // side of the wavefront its cell is on rather than only where it sits.
    const { spec } = parsed;
    const layer = spec.layers[anim.current.layer];
    const family = spec.nd_method.family;
    const axis =
      layer && family !== "kernel" && family !== "flatten"
        ? latticeAxisOf(layer, spec)
        : null;
    const size = axis === null ? 0 : parsed.shape[axis];
    const front =
      layer && layer.reverse
        ? (1 - anim.current.progress) * (size - 1)
        : anim.current.progress * (size - 1);

    for (let k = 0; k < POOL; k++) {
      const t = pool[k];
      if (!t) continue;
      const entry = shown.current[k];
      if (!entry) {
        t.visible = false;
        continue;
      }
      const [dist, i] = entry;
      const p = positions[i];
      t.visible = true;
      t.position.set(p[0], p[1] + 0.52, p[2]);
      t.quaternion.copy(cam.quaternion); // face the camera
      const label = aligned
        ? `${coordText[i]}\npred ${cellData.pred[i]}\ntrue ${cellData.true[i]}`
        : coordText[i];
      if (t.text !== label) t.text = label;

      // Fade with distance instead of popping in and out at the pool edge.
      const near = layout.radius * 1.1;
      const far = layout.radius * 2.6;
      t.fillOpacity = 1 - THREE.MathUtils.clamp((dist - near) / (far - near), 0, 0.92);

      if (axis === null) {
        t.color = "#9fb0cc";
      } else {
        const d = parsed.cells[i][axis] - front;
        const behind = layer.reverse ? d > 0 : d < 0;
        t.color = Math.abs(d) < 0.75 ? "#ffd79a" : behind ? "#9fc4ff" : "#7d8aa3";
      }
    }
  });

  // Scaled to the model for the same reason the axis labels are: the camera
  // pulls back in proportion to the lattice, so a fixed world size is not a
  // fixed readable size.
  const fontSize = Math.max(0.15, layout.radius * 0.032);

  return (
    <group>
      {Array.from({ length: POOL }, (_, i) => (
        <Text
          key={i}
          ref={(el) => (refs.current[i] = el)}
          visible={false}
          fontSize={fontSize}
          lineHeight={1.25}
          anchorX="center"
          anchorY="bottom"
          outlineWidth={0.014}
          outlineColor="#080b11"
        >
          {""}
        </Text>
      ))}
    </group>
  );
}
