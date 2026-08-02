import { OrbitControls, Text } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

import { CUBE, makeLayout } from "../layout.js";
import { latticeAxisOf } from "../spec.js";

const BASE = new THREE.Color("#3d4d74");
const SWEPT = new THREE.Color("#6d8ac9");
const FRONT = new THREE.Color("#ffc061");
const KERNEL = new THREE.Color("#9b7cff");
const JOINT = new THREE.Color("#45e3cd");

// How the wavefront reads. The front is a gaussian rather than a hard stripe,
// and what it leaves behind decays with distance instead of switching state:
// a sweep is a continuous process, and drawing it as two flat regions with a
// bright band between them said "three categories of cell" when the truth is
// one moving front and a fading memory of it.
// Sigma of the leading glow, in cells. Deliberately under one cell: a short
// axis has only a handful of slabs, so a sigma of ~1 lit three of four at
// once and the whole lattice read as "front". The wavefront has to stay
// narrower than the thing it moves through.
const FRONT_WIDTH = 0.5;
const TRAIL = 4.5; // cells; e-folding length of the wake
const SWELL = 0.34; // peak scale bump at the front

const _m = new THREE.Matrix4();
const _q = new THREE.Quaternion();
const _p = new THREE.Vector3();
const _s = new THREE.Vector3();
const _c = new THREE.Color();

// Per-frame wavefront colouring and scale. anim is a mutable ref shared with
// the app: { layer: int, progress: 0..1 } — progress lives outside React state
// so the animation never re-renders the tree.
function Lattice({ parsed, layout, anim }) {
  const meshRef = useRef();
  const count = parsed.cells.length;

  const positions = useMemo(
    () => parsed.cells.map((c) => layout.position(c)),
    [parsed, layout],
  );

  // Seed the matrices once so a paused first frame is not a pile of cubes at
  // the origin; the frame loop overwrites both matrix and colour anyway.
  useEffect(() => {
    const mesh = meshRef.current;
    positions.forEach((p, i) => {
      _m.compose(_p.set(p[0], p[1], p[2]), _q.identity(), _s.setScalar(1));
      mesh.setMatrixAt(i, _m);
      mesh.setColorAt(i, BASE);
    });
    mesh.instanceMatrix.needsUpdate = true;
    mesh.instanceColor.needsUpdate = true;
  }, [positions]);

  useFrame(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const { spec } = parsed;
    const layer = spec.layers[anim.current.layer];
    const progress = anim.current.progress;
    // The family, not the class name: a second kernel-family method would
    // have sniffed as a scan and been drawn with a travelling wavefront.
    const family = spec.nd_method.family;
    const atOnce = family === "kernel" || family === "flatten";
    const tint = family === "flatten" ? JOINT : KERNEL;
    const axis = layer ? latticeAxisOf(layer, spec) : null;

    // Both directionless cases get one *uniform* pulse. A travelling ripple
    // would look better and would be a lie: these families have no order
    // across cells, and the picture must not imply one.
    const pulse = Math.sin(progress * Math.PI) ** 1.5;
    const breathe = 0.5 - 0.5 * Math.cos(progress * 2 * Math.PI);

    const size = axis === null ? 0 : parsed.shape[axis];
    const front = layer && layer.reverse ? (1 - progress) * (size - 1) : progress * (size - 1);

    for (let i = 0; i < count; i++) {
      let swell = 0;
      if (atOnce) {
        _c.copy(BASE).lerp(tint, 0.72 * pulse);
        swell = 0.5 * pulse;
      } else if (axis === null) {
        // A time sweep has no lattice direction; the whole grid breathes.
        _c.copy(BASE).lerp(SWEPT, breathe);
        swell = 0.4 * breathe;
      } else {
        const c = layout.axisCoord(parsed.cells[i], axis);
        const d = c - front;
        const behind = layer.reverse ? d > 0 : d < 0;
        const glow = Math.exp(-((d / FRONT_WIDTH) ** 2));
        // Behind the front stays *visibly* swept however far back it is —
        // "which part of the axis is done" is the question the picture
        // answers, and a wake that decays to nothing erases the answer. The
        // decay rides on top of a floor instead of replacing it.
        const wake = behind ? 0.55 + 0.45 * Math.exp(-Math.abs(d) / TRAIL) : 0;
        _c.copy(BASE).lerp(SWEPT, wake).lerp(FRONT, glow);
        swell = glow;
      }
      const p = positions[i];
      _m.compose(
        _p.set(p[0], p[1], p[2]),
        _q.identity(),
        _s.setScalar(1 + SWELL * swell),
      );
      mesh.setMatrixAt(i, _m);
      mesh.setColorAt(i, _c);
    }
    mesh.instanceMatrix.needsUpdate = true;
    mesh.instanceColor.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[null, null, count]} key={count}>
      <boxGeometry args={[CUBE, CUBE, CUBE]} />
      <meshStandardMaterial
        roughness={0.38}
        metalness={0.22}
        envMapIntensity={0.6}
      />
    </instancedMesh>
  );
}

// Direction arrow for the active layer's swept axis, flipped when reverse. It
// rides *with* the wavefront rather than sitting still, so the arrow and the
// glow are one gesture instead of two things to reconcile.
function SweepArrow({ parsed, layout, anim }) {
  const groupRef = useRef();

  useFrame(() => {
    const g = groupRef.current;
    if (!g) return;
    const { spec } = parsed;
    const layer = spec.layers[anim.current.layer];
    const axis = layer ? latticeAxisOf(layer, spec) : null;
    const family = spec.nd_method.family;
    if (axis === null || family === "kernel" || family === "flatten") {
      g.visible = false;
      return;
    }
    g.visible = true;
    const sd = layout.dimOf[axis];
    const size = parsed.shape[axis];
    const margin = 1.35;
    const dir = layer.reverse ? -1 : 1;
    // axis 1 renders downward, so its on-screen direction flips
    const screenDir = sd === 1 ? -dir : dir;

    // Ride the front, in the cells' own coordinates. Scaling the screen extent
    // instead put the arrow at one end while the glow was at the other,
    // because above rank 3 two lattice axes share a screen dimension.
    const progress = anim.current.progress;
    const front = (layer.reverse ? 1 - progress : progress) * (size - 1);
    const along = layout.screenOffset(axis, front);

    // Travel along the swept dimension, stand off along a different one, so
    // the arrow never flies through the cells it is pointing at.
    const p = [0, 0, 0];
    p[sd] = along;
    if (sd === 1) p[0] = layout.blockExtent[0] / 2 + margin;
    else p[1] = layout.blockExtent[1] / 2 + margin;

    g.position.set(p[0], p[1], p[2]);
    g.rotation.set(0, 0, 0);
    if (sd === 0) g.rotation.z = screenDir > 0 ? -Math.PI / 2 : Math.PI / 2;
    if (sd === 1) g.rotation.z = screenDir > 0 ? 0 : Math.PI;
    if (sd === 2) g.rotation.x = screenDir > 0 ? Math.PI / 2 : -Math.PI / 2;
    // Sized to the model, not to the axis: a stride-sized arrow vanished on
    // the inner axes, where the stride is one cell.
    g.scale.setScalar(Math.max(2.2, layout.radius * 0.34) / 2.4);
  });

  return (
    <group ref={groupRef}>
      <mesh>
        <cylinderGeometry args={[0.055, 0.055, 1.5, 12]} />
        <meshBasicMaterial color="#ffc061" toneMapped={false} transparent opacity={0.9} />
      </mesh>
      <mesh position={[0, 1.05, 0]}>
        <coneGeometry args={[0.24, 0.62, 20]} />
        <meshBasicMaterial color="#ffd79a" toneMapped={false} />
      </mesh>
    </group>
  );
}

function AxisLabels({ parsed, layout }) {
  const names = parsed.spec.lattice.names.filter((n) => n !== "time");
  const labels = [];
  // The camera pulls back in proportion to the model, so a fixed world-space
  // size is not a fixed *apparent* size: 0.52 was a shout on a 5x7 lattice and
  // unreadable on a rank-4 stack. Scaling with the radius keeps it constant.
  const fontSize = Math.max(0.3, layout.radius * 0.075);
  for (let i = 0; i < Math.min(3, layout.rank); i++) {
    const sd = layout.dimOf[i];
    const p = [0, 0, 0];
    p[sd] = layout.blockExtent[sd] / 2 + 1.15;
    if (sd === 1) p[sd] = -p[sd];
    labels.push(
      <Text
        key={i}
        position={p}
        fontSize={fontSize}
        color="#7f8ca6"
        outlineWidth={0.012}
        outlineColor="#0b0e14"
      >
        {names[i] ?? `dim${i}`}
      </Text>,
    );
  }
  return labels;
}

// Depth cues. Fog fades the far side of a rank-4 stack so the near blocks read
// as near — without it the dimensional stacking is a flat wall of cubes — and
// the floor gives the whole thing somewhere to stand.
function Depth({ layout }) {
  const { scene } = useThree();
  useEffect(() => {
    const r = layout.radius;
    scene.fog = new THREE.Fog("#080b11", r * 2.0, r * 6.2);
    return () => {
      scene.fog = null;
    };
  }, [scene, layout]);

  const y = -layout.blockExtent[1] / 2 - 1.6;
  return (
    <gridHelper
      args={[layout.radius * 7, 22, "#1b2436", "#141b29"]}
      position={[0, y, 0]}
    />
  );
}

// Reposition the camera when the spec changes. The Canvas itself is never
// remounted — tearing down the WebGL context per sample is how you get
// "THREE.WebGLRenderer: Context Lost" and a permanently black scene.
function Recenter({ layout }) {
  const camera = useThree((s) => s.camera);
  const controls = useThree((s) => s.controls);
  useEffect(() => {
    const dist = Math.max(6, layout.radius * 2.4);
    camera.position.set(dist * 0.8, dist * 0.55, dist * 0.85);
    camera.lookAt(0, 0, 0);
    if (controls) {
      controls.target.set(0, 0, 0);
      controls.update();
    }
  }, [layout, camera, controls]);
  return null;
}

export default function Scene({ parsed, anim }) {
  const layout = useMemo(() => makeLayout(parsed.shape), [parsed]);
  const dist = Math.max(6, layout.radius * 2.4);
  const [touched, setTouched] = useState(false);

  return (
    <Canvas
      camera={{ position: [dist * 0.8, dist * 0.55, dist * 0.85], fov: 42 }}
      gl={{ antialias: true, alpha: true }}
      dpr={[1, 2]}
    >
      {/* Key, fill and rim. The rim separates a cube from the cube behind it
          once fog has flattened the value range — but it stays cool and weak
          on purpose: warm light is the wavefront's job, and an amber rim at
          any strength turns every resting cell gold and destroys the one
          colour distinction the picture exists to make. */}
      <ambientLight intensity={0.72} />
      <hemisphereLight args={["#93b4ff", "#0b0e14", 0.35]} />
      <directionalLight position={[6, 10, 8]} intensity={1.0} />
      <directionalLight position={[-8, -3, -6]} intensity={0.28} color="#7aa2ff" />
      <directionalLight position={[-4, 6, -9]} intensity={0.16} color="#cfe0ff" />
      <Recenter layout={layout} />
      <Depth layout={layout} />
      <Lattice
        key={
          parsed.spec.model.kind + parsed.shape.join("x") + parsed.cells.length
        }
        parsed={parsed}
        layout={layout}
        anim={anim}
      />
      <SweepArrow parsed={parsed} layout={layout} anim={anim} />
      <Suspense fallback={null}>
        <AxisLabels parsed={parsed} layout={layout} />
      </Suspense>
      {/* Drifts on its own so a still model still reads as 3-D, and stops
          for good the moment the user grabs it — a viewer that keeps pulling
          the camera out from under you is worse than one that never moved. */}
      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.12}
        autoRotate={!touched}
        autoRotateSpeed={0.35}
        onStart={() => setTouched(true)}
      />
    </Canvas>
  );
}
