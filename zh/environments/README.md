# Chinese-edition environments

The files in this directory are grouped by execution stack rather than by a
single global freeze.

- `core.in` / `core.lock.txt`: notebook validation, execution, and conversion.
- `pymc.in` / `pymc.lock.txt`: the verified PyMC scientific stack.
- `tfp.in` / `tfp.lock.txt`: the verified TensorFlow Probability stack.
- `optional-jax.in`: unresolved JAX/NumPyro input; it is not release-ready.
- `optional-modeling.in`: unresolved Bambi/PyMC-BART/seaborn input; it is not release-ready.

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
