# Chinese-edition environments

The files in this directory are grouped by execution stack rather than by a
single global freeze.

- `core.in` / `core.lock.txt`: notebook validation, execution, and conversion.
- `pymc.in` / `pymc.lock.txt`: the verified PyMC scientific stack (numpy,
  scipy, pandas, matplotlib, arviz, xarray, statsmodels, graphviz -- pymc
  itself lives in `pytensor-pymc.lock.txt` below).
- `pytensor-pymc.lock.txt`: `pytensor`/`pymc` themselves, installed together
  with `--no-deps` (see the install recipe below for why).
- `pymc-extras.lock.txt`: `pytensor`'s and `pymc`'s own actual runtime
  dependencies (minus numba/llvmlite), installed normally alongside
  `core.lock.txt`/`pymc.lock.txt`.
- `tfp.in` / `tfp.lock.txt`: the verified TensorFlow Probability stack.
- `bart.in` / `bart.lock.txt`: Chapter 7's PyMC-BART extension
  (`pymc-bart==0.11.0` plus `arviz-stats`/`arviz-base`). Verified on
  2026-08-21 to install cleanly with `--no-deps` alongside the exact
  `pymc.lock.txt` stack, with no version changes to pymc/pytensor/numpy/arviz.
- `ml.in` / `ml.lock.txt`: Chapter 8's scikit-learn dependency for its
  random-forest ABC model selector (`scikit-learn==1.9.0` plus its
  `joblib`/`narwhals` dependencies -- `threadpoolctl` is already satisfied by
  `pymc.lock.txt`). This is the exact version the chapter was executed
  against (37/37 code cells, zero errors).
- `ppl.in` / `ppl.lock.txt`: Chapter 10's JAX/NumPyro extension
  (`jax==0.4.38`, within the TFP-0.25-tested `<0.5.0` bound, plus `numpyro`
  and its `tqdm` dependency). Verified on 2026-08-21 to install cleanly with
  `--no-deps` alongside the exact `pymc.lock.txt`/`tfp.lock.txt` stack.
- `optional-jax.in`: unresolved JAX/NumPyro input, kept as a general-purpose
  unresolved stub; `ppl.in`/`ppl.lock.txt` above is what Chapter 10 actually
  declares and uses.
- `optional-modeling.in`: unresolved Bambi/seaborn input; it is not
  release-ready. Installing `bambi` was tried on 2026-08-21 and pulled in
  PyMC 6.x, ArviZ 1.x, and PyTensor 3.x, overwriting the verified pinned
  stack every other unit depends on -- do not install it into the shared
  environment described below; it needs its own isolated environment before
  the Preface's one Bambi-dependent cell can be executed. (PyMC-BART has its
  own dedicated, version-pinned `bart.in`/`bart.lock.txt` group above and was
  removed from here to avoid two conflicting specifications for the same
  package.)

The lock-style files contain exact, locally verified **direct** package
versions, not a fabricated transitive resolution. Install the required groups
in a clean Python 3.12 environment, record the platform, and run:

```sh
# PyTensor 2.38.x declares numba<=0.65.1 (no Python 3.12 wheel below 0.59,
# and conflicts with numpy==2.5.2 across the whole 0.59-0.65.1 range), and
# pymc declares cachetools<7. A normal resolve of the full stack in one
# pass tries to "fix" both bounds: for numba this walks down to a
# source-only build that fails outright on 3.12; for cachetools it silently
# downgrades an already-verified-compatible version. Nothing in this
# project uses PyTensor's numba backend (see system-dependencies.md), so
# stage the verified numba/llvmlite pair first, then install pytensor and
# pymc together with --no-deps to skip re-checking those bounds. Their own
# actual runtime dependencies (everything except numba/llvmlite) are pinned
# in pymc-extras.lock.txt and installed normally alongside
# core.lock.txt/pymc.lock.txt in the same pass.
python -m pip install numba==0.67.0 llvmlite==0.49.0
python -m pip install --no-deps -r zh/environments/pytensor-pymc.lock.txt
python -m pip install -r zh/environments/core.lock.txt -r zh/environments/pymc.lock.txt -r zh/environments/pymc-extras.lock.txt  # add tfp.lock.txt as needed
python -m pip check
```

Optional groups (`bart`, `ml`, `ppl`) declare their own top-level packages
with real version pins but were resolved by hand rather than by a plain
`pip install -r *.in`; install each of their `*.lock.txt` with its own
separate `--no-deps` pass (never folded into the normal-resolve command
above) -- each file says exactly what it needs beyond what
`core`/`pymc`/`tfp` already provide. Folding one into the normal pass lets
its own solver "fix" numpy/numba/llvmlite by silently downgrading them to
satisfy its unrelated, hand-resolved transitive chain (confirmed: asking
pip to resolve `pymc-bart` normally downgrades numpy 2.5.2->2.4.6 and
llvmlite 0.49.0->0.47.0 to find a self-consistent combination).

A chapter release must not use an unresolved optional input. Resolve it on
the target CPU/GPU platform, pin versions that actually install, add a
tested lock file, and record the exact command sequence above (or its
group-specific `--no-deps` equivalent) before adding the unit to
`book.toml`'s `smoke`/`release` profiles. See `system-dependencies.md` for
non-Python requirements.
