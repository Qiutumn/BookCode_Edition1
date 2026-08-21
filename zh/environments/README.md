# Chinese-edition environments

The files in this directory are grouped by execution stack rather than by a
single global freeze.

- `core.in` / `core.lock.txt`: notebook validation, execution, and conversion.
- `pymc.in` / `pymc.lock.txt`: the verified PyMC scientific stack.
- `tfp.in` / `tfp.lock.txt`: the verified TensorFlow Probability stack.
- `bart.in`: Chapter 7's PyMC-BART extension. The pin (`pymc-bart==0.11.0`) is
  researched, not locally verified — `pymc-bart` could not be installed in
  either Python environment available this session, so Chapter 7 has only
  ever been statically validated, never actually executed, on this host.
- `ml.in` / (no lock file yet): Chapter 8's scikit-learn dependency for its
  random-forest ABC model selector. `scikit-learn==1.9.0` is the exact
  version the chapter was executed against (37/37 code cells, zero errors).
- `ppl.in` / (no lock file yet): Chapter 10's JAX/NumPyro extension,
  version-bounded to match the JAX substrate TFP 0.25 was tested against.
  JAX is unavailable in every environment reachable this session, so the
  chapter's one JAX-only cell has never been executed here.
- `optional-jax.in`: unresolved JAX/NumPyro input; it is not release-ready.
- `optional-modeling.in`: unresolved Bambi/seaborn input; it is not
  release-ready. (PyMC-BART now has its own dedicated, version-pinned
  `bart.in` group above and was removed from here to avoid two conflicting
  specifications for the same package.)

The current lock-style files contain exact, locally verified **direct** package
versions, not a fabricated transitive resolution. Install the required groups
in a clean Python 3.12 environment, record the platform, and run:

```sh
python -m pip install -r zh/environments/core.lock.txt
python -m pip install -r zh/environments/pymc.lock.txt  # as needed
python -m pip check
```

A chapter release must not use an unresolved optional input. Resolve it on the
target CPU/GPU platform, pin versions that actually install, and add a tested
lock file first. See `system-dependencies.md` for non-Python requirements.
