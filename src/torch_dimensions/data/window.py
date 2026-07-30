"""Windowing over the time axis. Pure index arithmetic — no tensors, no I/O.

Kept free of data so it can be reasoned about and tested on its own: an
off-by-one here leaks future timesteps into the input window, which is the
quietest possible way to produce excellent and meaningless results.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import NamedTuple

__all__ = ["LatticeWindow", "Window"]


class Window(NamedTuple):
    """Half-open index ranges: inputs ``[x0, x1)``, targets ``[y0, y1)``."""

    x0: int
    x1: int
    y0: int
    y1: int


class LatticeWindow:
    """The set of windows tiling a time axis.

    Args:
        n_time: length of the time axis.
        input_len: timesteps fed to the model.
        horizon: timesteps to predict. ``0`` means no target range.
        stride: step between consecutive window starts.

    Targets begin exactly where inputs end, so no window ever sees its own
    target.
    """

    __slots__ = ("windows", "input_len", "horizon", "stride", "n_time")

    def __init__(
        self,
        n_time: int,
        input_len: int,
        horizon: int = 0,
        stride: int = 1,
        *,
        _windows: Sequence[Window] | None = None,
    ) -> None:
        if input_len < 1:
            raise ValueError(f"input_len must be >= 1; got {input_len}")
        if horizon < 0:
            raise ValueError(f"horizon must be >= 0; got {horizon}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1; got {stride}")
        self.n_time, self.input_len, self.horizon, self.stride = (
            n_time,
            input_len,
            horizon,
            stride,
        )
        if _windows is not None:
            self.windows = tuple(_windows)
            return
        span = input_len + horizon
        if span > n_time:
            raise ValueError(
                f"input_len + horizon = {span} exceeds the {n_time} available timesteps"
            )
        self.windows = tuple(
            Window(i, i + input_len, i + input_len, i + span)
            for i in range(0, n_time - span + 1, stride)
        )

    def _derive(self, windows: Sequence[Window]) -> LatticeWindow:
        return LatticeWindow(
            self.n_time, self.input_len, self.horizon, self.stride, _windows=windows
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __iter__(self) -> Iterator[Window]:
        return iter(self.windows)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return self._derive(self.windows[i])
        return self.windows[i]

    def split(self, at: int) -> tuple[LatticeWindow, LatticeWindow]:
        """Split into windows ending at or before ``at``, and those starting at
        or after it.

        Windows straddling the boundary are dropped by both sides rather than
        assigned to one. That gap is deliberate: keeping a straddling window
        would put timesteps from after the cut inside a training input.
        """
        before = [w for w in self.windows if w.y1 <= at]
        after = [w for w in self.windows if w.x0 >= at]
        return self._derive(before), self._derive(after)

    def split_at_time(self, times: Sequence, value) -> tuple[LatticeWindow, LatticeWindow]:
        """:meth:`split` by an actual timestamp rather than an index.

        ``times`` must be sorted — a scan for the first timestamp past the cut
        is meaningless otherwise, and unsorted input previously produced a
        silently nonsensical split rather than an error.
        """
        for prev, cur in zip(times, times[1:], strict=False):
            if cur < prev:
                raise ValueError(f"times must be sorted; got {cur!r} after {prev!r}")
        for i, t in enumerate(times):
            if t >= value:
                return self.split(i)
        return self.split(len(times))

    def __repr__(self) -> str:
        return (
            f"LatticeWindow({len(self)} windows, input_len={self.input_len}, "
            f"horizon={self.horizon}, stride={self.stride})"
        )
