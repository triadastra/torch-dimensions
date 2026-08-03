# Documentation

- **[Adding a mixer](adding-a-mixer.md)** — the extension point that matters
  most: a 1-D sequence model becomes an N-D one by satisfying one shape
  contract. Worked example, conformance report, and the mistakes to expect.
- **[Adding an nd_method](adding-a-method.md)** — changing *how* the axes are
  handled rather than what happens along one, including the two bugs the
  conformance suite found in this project's own example strategy.
- **[CUDA verification checklist](cuda-checklist.md)** — every CUDA claim as
  one runnable file, and what it reported on an RTX 5090: 13 passed, 0 failed,
  plus the one gap that a Blackwell card could not close. Fifteen minutes on a
  free Colab T4 to re-run.

- **[Hugging Face model card](hf-card.md)** — the source for
  [`Celsia/torch-dimensions`](https://huggingface.co/Celsia/torch-dimensions),
  where the benchmark checkpoints and agreement runs are published. Edited
  here, uploaded as that repo's `README.md`.

Elsewhere in the repo: [DESIGN.md](../DESIGN.md) (the architecture),
[PLAN.md](../PLAN.md) (what is built and what is next),
[RESULTS.md](../RESULTS.md) (reproductions),
[BENCHMARKS.md](../BENCHMARKS.md) (measured costs),
[DEBUG.md](../DEBUG.md) (every bug found, and what caught it),
[VIEWER.md](../VIEWER.md) (the GUI).
