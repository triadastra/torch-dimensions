// Parsing the architecture spec (see src/torch_dimensions/spec.py).
// The viewer consumes; it never invents. Anything not in the spec is not shown.

export const SPEC_FORMAT = "torch-dimensions/architecture";

export function decodeRle(runs, total) {
  // Run-length encoding over the flattened lattice, starting with a False run.
  const flags = new Uint8Array(total);
  let pos = 0;
  let val = 0;
  for (const run of runs) {
    if (val) flags.fill(1, pos, pos + run);
    pos += run;
    val ^= 1;
  }
  return flags;
}

export function parseSpec(json) {
  if (!json || json.format !== SPEC_FORMAT) {
    throw new Error(`not a ${SPEC_FORMAT} document`);
  }
  const shape = json.lattice.shape;
  const total = Math.max(
    1,
    shape.reduce((a, b) => a * b, 1),
  );
  const flags = decodeRle(json.lattice.cells.present_rle, total);
  const cells = [];
  for (let f = 0; f < total; f++) {
    if (!flags[f]) continue;
    const coord = new Array(shape.length);
    let rem = f;
    for (let i = shape.length - 1; i >= 0; i--) {
      coord[i] = rem % shape[i];
      rem = Math.floor(rem / shape[i]);
    }
    cells.push(coord);
  }
  return { spec: json, shape, cells };
}

// Which lattice axis (index into shape) a layer sweeps, or null when it
// sweeps none. `axis_index` is null for the joint family, which sweeps
// nothing at all — without this guard `null - 0` is 0 and the scene draws a
// travelling wavefront along the first axis for a model that has no direction.
export function latticeAxisOf(layer, spec) {
  if (layer.axis_index == null) return null;
  if (spec.lattice.time && layer.axis_index === 0) return null;
  return layer.axis_index - (spec.lattice.time ? 1 : 0);
}

export function fmtParams(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
  return String(n);
}
