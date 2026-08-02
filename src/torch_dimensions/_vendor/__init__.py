"""Upstream reference implementations, redistributed verbatim.

This package contains the original authors' code for S4/S4D and Mamba,
shipped so that anyone can run — and diff — the exact reference
implementations rather than trusting our portable rewrites:

- ``s4/``     from state-spaces/s4 (Gu, Goel, Ré et al.), Apache-2.0
- ``mamba/``  from state-spaces/mamba (Gu, Dao), Apache-2.0

**How to verify it is really their code.** Every vendored module sits next to
a ``.orig`` file that is byte-identical to the upstream file at the commit
recorded in ``MANIFEST.json``. The only differences between a module and its
``.orig`` are import-path and optional-dependency patches, every changed line
tagged ``torch-dimensions patch``. ``tests/test_vendored.py`` enforces all of
that offline on every CI run: the ``.orig`` hashes must match the manifest,
and any unmarked difference fails the build. ``dossier/verify_vendored.py``
additionally re-fetches upstream and compares the ``.orig`` files byte-for-
byte against the pinned commits.

Both licenses are included in this directory (``s4/LICENSE``,
``mamba/LICENSE``); attribution and the statement of changes required by
Apache-2.0 section 4 are in the project-level NOTICE file.

Nothing here is imported by ``torch_dimensions`` itself — the core library
stays pure-torch. The adapters in ``torch_dimensions.mixers.upstream`` load
these modules lazily and need the ``[upstream]`` extra (einops, numpy, scipy).
"""
