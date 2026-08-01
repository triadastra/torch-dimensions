# Documentation

- **[Adding a mixer](adding-a-mixer.md)** — the extension point that matters
  most: a 1-D sequence model becomes an N-D one by satisfying one shape
  contract. Worked example, conformance report, and the mistakes to expect.
- **[Adding an nd_method](adding-a-method.md)** — changing *how* the axes are
  handled rather than what happens along one, including the two bugs the
  conformance suite found in this project's own example strategy.
- **[CUDA verification checklist](cuda-checklist.md)** — the procedure for the
  device the library has never actually run on. Fifteen minutes on a free
  Colab T4.

Elsewhere in the repo: [DESIGN.md](../DESIGN.md) (the architecture),
[PLAN.md](../PLAN.md) (what is built and what is next),
[RESULTS.md](../RESULTS.md) (reproductions),
[BENCHMARKS.md](../BENCHMARKS.md) (measured costs),
[DEBUG.md](../DEBUG.md) (every bug found, and what caught it),
[VIEWER.md](../VIEWER.md) (the GUI).
