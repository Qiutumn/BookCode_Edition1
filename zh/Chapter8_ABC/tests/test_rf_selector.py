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
        import pymc as pm
        import xarray as xr

        from scripts import rf_selector

        # 中文版现代化说明：早先用一个只带 observed_RVs/location 属性的轻量占位对象
        # （FakeModel）代替真实 pm.Model；但 select_model 会把它直接传给
        # pm.sample_prior_predictive，而当前 PyMC 内部无条件访问 model.potentials，
        # 占位对象没有这个属性就会报 AttributeError。这里改用真实的最小 pm.Model，
        # 既满足 pm.sample_prior_predictive 的实际接口要求，也让测试更贴近真实用法；
        # `location` 只用于 fake_posterior_predictive 里生成对应模型的伪造后验预测。
        def make_fake_model(name: str, location: float):
            with pm.Model() as model:
                pm.Normal(name, mu=location, sigma=0.2, observed=np.zeros(12))
            model.location = location
            return model

        model_0 = make_fake_model("y", -3.0)
        model_1 = make_fake_model("y", 3.0)
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
