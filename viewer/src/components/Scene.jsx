import { OrbitControls, Text } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Suspense, useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

import { CUBE, makeLayout } from "../layout.js";
import { latticeAxisOf } from "../spec.js";

const BASE = new THREE.Color("#33415e");
const SWEPT = new THREE.Color("#4a5b80");
const FRONT = new THREE.Color("#f5a623");
const KERNEL = new THREE.Color("#7c5cff");

// Per-frame wavefront colouring. anim is a mutable ref shared with the app:
// { layer: int, progress: 0..1 } — progress lives outside React state so the
// animation never re-renders the tree.
function Lattice({ parsed, layout, anim }) {
  const meshRef = useRef();
  const count = parsed.cells.length;

  const positions = useMemo(
    () => parsed.cells.map((c) => layout.position(c)),
    [parsed, layout],
  );

  useEffect(() => {
    const mesh = meshRef.current;
    const m = new THREE.Matrix4();
    positions.forEach((p, i) => {
      m.setPosition(p[0], p[1], p[2]);
      mesh.setMatrixAt(i, m);
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
    const isKernel = spec.nd_method.name === "AxialKernel";
    const axis = layer ? latticeAxisOf(layer, spec) : null;
    const color = new THREE.Color();

    for (let i = 0; i < count; i++) {
      if (isKernel) {
        // The kernel family contracts every spatial axis at once: one
        // simultaneous flash per layer, not a travelling front.
        const pulse = Math.sin(progress * Math.PI);
        color.copy(BASE).lerp(KERNEL, 0.65 * pulse);
      } else if (axis === null) {
        // A time sweep has no lattice direction; the whole grid breathes.
        const pulse = 0.5 - 0.5 * Math.cos(progress * 2 * Math.PI);
        color.copy(BASE).lerp(SWEPT, pulse);
      } else {
        const size = parsed.shape[axis];
        const c = layout.axisCoord(parsed.cells[i], axis);
        const front = (layer.reverse ? 1 - progress : progress) * (size - 1);
        const d = c - front;
        const behind = layer.reverse ? d > 0 : d < 0;
        if (Math.abs(d) < 0.75) {
          color.copy(FRONT);
        } else if (behind) {
          color.copy(SWEPT);
        } else {
          color.copy(BASE);
        }
      }
      mesh.setColorAt(i, color);
    }
    mesh.instanceColor.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[null, null, count]} key={count}>
      <boxGeometry args={[CUBE, CUBE, CUBE]} />
      <meshStandardMaterial roughness={0.55} metalness={0.1} />
    </instancedMesh>
  );
}

// Direction arrow for the active layer's swept axis, flipped when reverse.
function SweepArrow({ parsed, layout, anim }) {
  const groupRef = useRef();

  useFrame(() => {
    const g = groupRef.current;
    if (!g) return;
    const { spec } = parsed;
    const layer = spec.layers[anim.current.layer];
    const axis = layer ? latticeAxisOf(layer, spec) : null;
    const isKernel = spec.nd_method.name === "AxialKernel";
    if (axis === null || isKernel) {
      g.visible = false;
      return;
    }
    g.visible = true;
    const sd = layout.dimOf[axis];
    const len = layout.blockExtent[sd];
    const margin = 1.1;
    const offY = layout.blockExtent[1] / 2 + margin;
    const dir = layer.reverse ? -1 : 1;
    // axis 1 renders downward, so its on-screen direction flips
    const screenDir = sd === 1 ? -dir : dir;

    g.position.set(0, offY, 0);
    if (sd === 0) g.rotation.set(0, 0, screenDir > 0 ? -Math.PI / 2 : Math.PI / 2);
    if (sd === 1) g.rotation.set(0, 0, screenDir > 0 ? 0 : Math.PI);
    if (sd === 2) g.rotation.set(screenDir > 0 ? Math.PI / 2 : -Math.PI / 2, 0, 0);
    g.scale.setScalar(Math.max(2, len * 0.35) / 2.4);
  });

  return (
    <group ref={groupRef}>
      <mesh position={[0, 0, 0]}>
        <cylinderGeometry args={[0.1, 0.1, 1.6, 12]} />
        <meshStandardMaterial color="#f5a623" />
      </mesh>
      <mesh position={[0, 1.2, 0]}>
        <coneGeometry args={[0.32, 0.8, 16]} />
        <meshStandardMaterial color="#f5a623" />
      </mesh>
    </group>
  );
}

function AxisLabels({ parsed, layout }) {
  const names = parsed.spec.lattice.names.filter((n) => n !== "time");
  const labels = [];
  for (let i = 0; i < Math.min(3, layout.rank); i++) {
    const sd = layout.dimOf[i];
    const p = [0, 0, 0];
    p[sd] = layout.blockExtent[sd] / 2 + 1.0;
    if (sd === 1) p[sd] = -p[sd];
    labels.push(
      <Text key={i} position={p} fontSize={0.55} color="#8b95a8">
        {names[i] ?? `dim${i}`}
      </Text>,
    );
  }
  return labels;
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

  return (
    <Canvas camera={{ position: [dist * 0.8, dist * 0.55, dist * 0.85], fov: 42 }}>
      <color attach="background" args={["#0b0e14"]} />
      <ambientLight intensity={0.85} />
      <directionalLight position={[6, 10, 8]} intensity={1.1} />
      <directionalLight position={[-8, -4, -6]} intensity={0.35} />
      <Recenter layout={layout} />
      <Lattice
        key={parsed.spec.model.kind + parsed.shape.join("x") + parsed.cells.length}
        parsed={parsed}
        layout={layout}
        anim={anim}
      />
      <SweepArrow parsed={parsed} layout={layout} anim={anim} />
      <Suspense fallback={null}>
        <AxisLabels parsed={parsed} layout={layout} />
      </Suspense>
      <OrbitControls makeDefault enableDamping dampingFactor={0.12} />
    </Canvas>
  );
}
