"""state-spaces/mamba reference modules, redistributed verbatim (Apache-2.0).

See ``torch_dimensions._vendor`` for the verification scheme. The selective
scan runs on any device: ``selective_scan_fn`` dispatches to the authors' own
``selective_scan_ref`` when the CUDA extension is absent (the one functional
patch, tagged in ``selective_scan_interface.py``).
"""
