# System dependencies

Use CPython 3.12 and a UTF-8 locale. Native scientific wheels normally avoid a
local compiler, but source builds require a C/C++ toolchain and `make`.

## Required by feature

- **Graphviz rendering:** install the Graphviz system package and verify
  `dot -V`. The Python `graphviz` package alone does not provide `dot`.
- **Pandoc production conversion (optional):** install a reviewed Pandoc build
  at `/usr/bin/pandoc` or `/usr/local/bin/pandoc`, or set `ZH_PANDOC` to an
  absolute executable path. The build never selects an arbitrary PATH entry.
  Without Pandoc, the deterministic internal converter is used and rejects
  syntax it cannot preserve safely.
- **Compiled package fallback:** GCC/G++ (or platform equivalents), Python
  headers, and `make` may be needed when wheels are unavailable.
- **PyTensor native C backend:** requires the CPython development headers
  (`/usr/include/python3.<minor>/Python.h`, e.g. the `python3.12-dev` package
  on Debian/Ubuntu). Without them, `pm.sample(...)` and any other PyTensor
  compilation raise `CompileError: ... fatal error: Python.h: No such file or
  directory`. Until those headers are installed, execute chapter builders with
  a non-native PyTensor fallback, e.g.:
  `PYTENSOR_FLAGS='cxx=,mode=FAST_COMPILE' python zh/tools/build_book.py ...`
  or `PYTENSOR_FLAGS='mode=NUMBA,cxx=' python zh/tools/build_book.py ...`
  (the latter additionally requires `numba` in the execution environment).
  Do not claim native-C execution was verified unless the matching headers
  were actually installed and `pm.sample` was run without either flag.
- **JAX/TensorFlow accelerators:** CUDA/cuDNN versions are platform-specific;
  resolve the corresponding optional stack on the actual runner rather than
  copying CPU pins.

On the 2026-08-20 verification host, GCC/G++ were present but the CPython 3.12
development headers (`/usr/include/python3.12/Python.h`) and `make` were
absent, so PyTensor native compilation fails; `dot` and Pandoc were also
absent. Smoke execution on this host therefore requires the `PYTENSOR_FLAGS`
fallback above. These observations are not package-version promises for other
platforms.

**Known effect on smoke timeouts:** the pure-Python `FAST_COMPILE`/`NUMBA`
fallbacks above are dramatically slower than native compilation for
multi-parameter `pm.sample(...)` calls. On the 2026-08-20 verification host,
several individual PyMC-heavy cells in Chapter 3 (e.g. penguin-mass regression
models) took 3-5+ minutes each under `FAST_COMPILE`, which exceeds
`profiles.smoke.timeout` (300s, an `nbclient` per-cell ceiling) in
`zh/book.toml` and produces `error: ... A cell timed out while it was being
executed, after 300 seconds`. This was confirmed to be purely an artifact of
the missing native compiler on this specific host, not a content or logic
defect: the same notebook, run cell-by-cell with a larger per-cell timeout
(900s), executed every code cell successfully with zero errors. Do not
"fix" a smoke timeout on this class of host by raising
`profiles.smoke.timeout` globally — that ceiling is sized for a host with a
working native/Numba PyTensor backend (e.g. GitHub Actions' `ubuntu-latest`
via `actions/setup-python`, which ships CPython headers), and inflating it
would just mask real hangs there. On a header-less host, either install
`python3.<minor>-dev` or pass a larger `--timeout` to a direct
`nbclient`/`build_book.py` invocation for local diagnosis only.
