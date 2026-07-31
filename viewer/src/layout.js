// Dimensional stacking (VIEWER.md §4). Axes 0-2 are literal x/y/z; each axis
// above the third arranges the blocks below it along x/y/z again, with a gap
// visibly larger than the level beneath — rank 4 is a grid of cubes, rank 5 a
// grid of those grids. Every cell stays visible; the gap hierarchy encodes
// which level you are looking at.

export const CUBE = 0.72;
const PITCH = 1.0;

export function makeLayout(shape) {
  const rank = shape.length;
  const strides = new Array(rank);
  const dimOf = new Array(rank);
  const blockExtent = [CUBE, CUBE, CUBE]; // grows as levels accumulate

  for (let i = 0; i < rank; i++) {
    const sd = i % 3;
    const level = Math.floor(i / 3);
    const gap = level === 0 ? PITCH - CUBE : PITCH * (0.8 + 1.4 * level);
    strides[i] = blockExtent[sd] + gap;
    dimOf[i] = sd;
    blockExtent[sd] = strides[i] * (shape[i] - 1) + blockExtent[sd];
  }

  const center = blockExtent.map((e) => (e - CUBE) / 2);

  function position(coord) {
    const p = [0, 0, 0];
    for (let i = 0; i < rank; i++) p[dimOf[i]] += coord[i] * strides[i];
    return [p[0] - center[0], center[1] - p[1], p[2] - center[2]]; // axis 1 reads downward
  }

  // Wavefront position of a cell along one lattice axis, in [0, shape[axis]).
  function axisCoord(coord, axis) {
    return coord[axis];
  }

  const radius = Math.hypot(...blockExtent) / 2;
  return { position, axisCoord, strides, dimOf, blockExtent, radius, rank };
}
