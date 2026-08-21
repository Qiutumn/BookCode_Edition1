"""第八章随机森林选择器的局部单元/冒烟测试。"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

CHAPTER_DIR = Path(__file__).resolve().parents[1]
if str(CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIR))

SCIENTIFIC_STACK_AVAILABLE = all(
    importlib.util.find_spec(name) is not None for name in ("arviz", "pymc", "xarray")
)
SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None


@unittest.skipUnless(
    SCIENTIFIC_STACK_AVAILABLE and SKLEARN_AVAILABLE,
    "行为测试需要 arviz、pymc、xarray 与可选依赖 scikit-learn",
)
class RandomForestSelectorBehaviorTests(unittest.TestCase):
    def test_selector_prefers_matching_model_and_is_deterministic(self):
        import arviz as az
        import xarray as xr

        from scripts import rf_selector

        class FakeModel:
            def __init__(self, name: str, location: float):
                self.observed_RVs = [SimpleNamespace(name=name)]
                self.location = location

        model_0 = FakeModel("y", -3.0)
        model_1 = FakeModel("y", 3.0)
        posterior = az.from_dict(
            posterior={"theta": np.zeros((2, 40))},
        )

        def fake_posterior_predictive(
            posterior_subset,
            *,
            model,
            sample_dims,
            var_names,
            return_inferencedata,
            progressbar,
            random_seed,
        ):
            self.assertEqual(sample_dims, ["sample"])
            self.assertEqual(var_names, ["y"])
            self.assertTrue(return_inferencedata)
            self.assertFalse(progressbar)
            n_samples = posterior_subset.sizes["sample"]
            rng = np.random.default_rng(random_seed)
            values = rng.normal(model.location, 0.2, size=(n_samples, 12))
            return az.InferenceData(
                posterior_predictive=xr.Dataset(
                    {"y": (("sample", "obs_id"), values)}
                )
            )

        kwargs = dict(
            models=[(model_0, posterior), (model_1, posterior)],
            statistics=[np.mean, np.std],
            observations=np.full(12, 3.0),
            n_samples=80,
            n_trees=40,
            f_max_features=0.5,
            random_seed=814,
        )
        with mock.patch.object(
            rf_selector.pm,
            "sample_posterior_predictive",
            side_effect=fake_posterior_predictive,
        ):
            first = rf_selector.select_model(**kwargs)
            second = rf_selector.select_model(**kwargs)

        self.assertEqual(first, second)
        self.assertEqual(first[0], 1)
        self.assertGreaterEqual(first[1], 0.0)
        self.assertLessEqual(first[1], 1.0)

@unittest.skipUnless(
    SCIENTIFIC_STACK_AVAILABLE,
    "输入验证测试需要 arviz、pymc 与 xarray",
)
class RandomForestSelectorValidationTests(unittest.TestCase):
    def test_rejects_obsolete_size_argument(self):
        from scripts.rf_selector import select_model

        with self.assertRaisesRegex(ValueError, "不再接受 size"):
            select_model(
                models=[object()],
                statistics=[np.mean],
                observations=np.arange(3),
                size=10,
            )


if __name__ == "__main__":
    unittest.main()
