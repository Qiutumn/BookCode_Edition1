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

**Same effect on Chapter 7's release-scale timeout.** Chapter 7's single
heaviest cell (`ch07-fit-interaction-bart`, a `pymc-bart` model with `m=200`
trees, release budget 1000 draws x 4 chains) took 707s in isolation under
`FAST_COMPILE` on the 2026-08-20/21 verification host -- comfortably under
`profiles.release.timeout` (1200s) on its own, but a full `release
--promote` run of the whole notebook hit `CellTimeoutError: ... after 1200
seconds` on exactly this cell, evidently from memory/CPU pressure
accumulated by the ~9 other BART model fits earlier in the same notebook
and kernel session. Isolated, single-model reruns of this cell completed
cleanly and produced the expected posterior shape every time it was tried.
This is the same class of host-speed artifact as the Chapter 3 note above,
not a content or logic defect -- but unlike Chapter 3, it means a full
`release --promote` of Chapter 7 could not be completed end-to-end on this
specific host within the standard profile timeout, even though every cell
is individually correct (verified both via a full smoke-profile run, 25/25
code cells with zero errors, and via this isolated release-scale rerun).
Chapter 7 is included in `book.toml`'s `smoke`/`release` profiles and the
CI matrix on the expectation that a host with native PyTensor compilation
(e.g. CI's `ubuntu-latest`) completes this cell in a small fraction of
707s, comfortably inside 1200s even with the earlier cells' accumulated
overhead; its live `.ipynb`/`.org` were not promoted from this host.

**Chapter 6: a memory-bound version of the same class of issue.** Chapter
6's `ch06-gam-inference-and-forecast` cell (a TFP `windowed_adaptive_nuts`
GAM fit) originally failed release-scale `release --promote` with
`CellTimeoutError` under the shared helper's hardcoded `jit_compile=False`.
Isolated diagnostics on this host confirmed (a) the GAM model itself has no
`LinearGaussianStateSpaceModel` component, so `jit_compile=True` is
numerically safe for it and ~4x faster (90.2s vs 356.1s at smoke scale, both
well under budget), while (b) every other model in the chapter that *does*
use an LGSSM (latent-AR, ARMA, BSTS) hard-fails under `jit_compile=True`
with `InvalidArgumentError: XLA compilation requires a fixed tensor list
size` -- confirmed by direct reproduction, not inferred. `run_windowed_nuts`
was accordingly given an opt-in `jit_compile` parameter (default `False`,
unchanged everywhere except the GAM call site) rather than a global flip.
This fix is real and correct, but it did not fully unblock promotion on
this host, across two further `release --promote` attempts:

- A second attempt (with the fix applied, but run concurrently with other
  heavy promotion jobs on this host) got past XLA compilation for the GAM
  cell (confirmed in the log, `Compiled cluster using XLA!`) and then the
  kernel died outright (`Kernel died`, not a timeout) partway through
  sampling -- consistent with this host's ~7.7GB total RAM being exhausted
  by concurrent jobs plus the accumulated state of the heavier models fit
  earlier in the same kernel session.
- A third attempt, run in isolation with no concurrent jobs (confirming the
  memory pressure above was at least partly cross-job contention, not
  purely this notebook alone), got further still: the GAM cell itself no
  longer bottlenecked, but `ch06-gam-with-latent-ar-errors`'s
  `gam_latent_ar_model` fit -- an LGSSM-based model that must keep
  `jit_compile=False` (see the confirmed `InvalidArgumentError` above) --
  then hit `CellTimeoutError` after 1200s on its own. Unlike the GAM cell,
  there is no available fix here: this model's XLA incompatibility is
  hard, so it cannot be sped up the same way, and its release-scale budget
  (larger draws/chains than smoke scale) is simply slow under the
  `FAST_COMPILE`/pure-Python TFP sampling path this header-less host is
  limited to.

Both are host-capacity artifacts, not logic or content defects -- the
chapter passed a full smoke-profile run with zero errors, and the
`jit_compile=True` GAM change is independently verified correct in
isolation. Chapter 6 remains in `book.toml`'s `smoke`/`release` profiles
and the CI matrix (CI runners have more headroom and, on `ubuntu-latest`,
native PyTensor/XLA compilation rather than this host's `FAST_COMPILE`
fallback); its live `.ipynb`/`.org` were not promoted from this host.
