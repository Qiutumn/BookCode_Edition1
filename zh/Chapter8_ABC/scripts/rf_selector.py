"""用先验模拟参考表和随机森林执行 ABC 模型选择。

本文件是 ``notebooks/scripts/rf_selector.py`` 的章节局部现代化版本。为保留
原书调用结构，``select_model`` 仍接受 ``(pm.Model, InferenceData)`` 对；但参考
表严格由每个模型的先验与模拟器独立生成，第二项不会用于条件化模拟。这样避免
把同一份观测 ``y_obs`` 已经更新过的后验再次用于“模型选择”。

分类森林使用袋外（out-of-bag, OOB）预测产生逐行 0/1 误分类目标；第二座回归
森林再估计观测摘要邻域的局部误分类概率。该返回值不是后验模型概率。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import arviz as az
import numpy as np
import pymc as pm

ArrayStatistic = Callable[[np.ndarray], Any]
ModelFit = tuple[Any, az.InferenceData | None]


def _as_feature_vector(value: Any, *, statistic_index: int) -> np.ndarray:
    """把一个标量或数组摘要统计量展平成有限的一维特征。"""

    feature = np.asarray(value, dtype=float).reshape(-1)
    if feature.size == 0:
        raise ValueError(f"第 {statistic_index} 个摘要统计量返回了空结果")
    if not np.all(np.isfinite(feature)):
        raise ValueError(f"第 {statistic_index} 个摘要统计量返回了非有限值")
    return feature


def _summarize_one(sample: np.ndarray, statistics: Sequence[ArrayStatistic]) -> np.ndarray:
    pieces = [
        _as_feature_vector(statistic(sample), statistic_index=index)
        for index, statistic in enumerate(statistics)
    ]
    return np.concatenate(pieces)


def _summary_matrix(
    samples: np.ndarray,
    statistics: Sequence[ArrayStatistic],
    *,
    expected_width: int | None = None,
) -> np.ndarray:
    """逐行计算摘要统计量，并验证所有行的特征宽度一致。"""

    values = np.asarray(samples)
    if values.ndim < 2:
        raise ValueError("先验模拟样本至少应有“样本”和“观测”两个维度")

    rows = [_summarize_one(np.asarray(row), statistics) for row in values]
    widths = {row.size for row in rows}
    if len(widths) != 1:
        raise ValueError("摘要统计量在不同模拟样本上返回了不一致的特征宽度")

    matrix = np.vstack(rows)
    if expected_width is not None and matrix.shape[1] != expected_width:
        raise ValueError(
            "模拟数据与观测数据的摘要统计量宽度不一致："
            f"{matrix.shape[1]} != {expected_width}"
        )
    return matrix


def _prior_predictive_rows(idata: az.InferenceData, observed_name: str) -> np.ndarray:
    """把 prior_predictive 中的样本维移到首轴，其余维展平。"""

    if not isinstance(idata, az.InferenceData) or not hasattr(idata, "prior_predictive"):
        raise TypeError("PyMC 先验预测必须返回带 prior_predictive 组的 InferenceData")
    if observed_name not in idata.prior_predictive:
        raise KeyError(f"prior_predictive 中不存在观测变量 {observed_name!r}")

    data_array = idata.prior_predictive[observed_name]
    sample_dims = [dim for dim in ("chain", "draw", "sample") if dim in data_array.dims]
    if not sample_dims:
        raise ValueError("先验预测变量缺少 sample 或 chain/draw 采样维")

    if sample_dims == ["sample"]:
        normalized = data_array
    else:
        normalized = data_array.stack(sample=sample_dims)

    event_dims = [dim for dim in normalized.dims if dim != "sample"]
    normalized = normalized.transpose("sample", *event_dims)
    return np.asarray(normalized.values).reshape(normalized.sizes["sample"], -1)


def _oob_misclassification_targets(classifier: Any, labels: np.ndarray) -> np.ndarray:
    """由分类森林的袋外类别预测构造严格的 0/1 误分类目标。"""

    if not hasattr(classifier, "oob_decision_function_"):
        raise ValueError("分类森林没有 OOB 决策结果；必须设置 oob_score=True")

    decision = np.asarray(classifier.oob_decision_function_, dtype=float)
    labels = np.asarray(labels)
    if decision.ndim != 2 or decision.shape[0] != labels.size:
        raise ValueError("OOB 决策矩阵与参考表标签形状不一致")

    row_totals = decision.sum(axis=1)
    valid = np.all(np.isfinite(decision), axis=1) & (row_totals > 0)
    if not np.all(valid):
        missing = int(np.count_nonzero(~valid))
        raise ValueError(
            f"有 {missing} 行没有有效 OOB 预测；请增加 n_trees 后重试"
        )

    predicted = np.asarray(classifier.classes_)[np.argmax(decision, axis=1)]
    return (predicted != labels).astype(float)


def select_model(
    models: Sequence[ModelFit],
    statistics: Sequence[ArrayStatistic],
    observations: np.ndarray,
    n_samples: int = 1000,
    size: Any = None,
    n_trees: int = 100,
    f_max_features: float = 0.5,
    random_seed: int | None = None,
) -> tuple[int, float]:
    """选择最符合观测摘要统计量的模型。

    Parameters
    ----------
    models
        ``(pm.Model, InferenceData)`` 对的非空序列。为保持原书公共调用结构，第二项
        可以是模型拟合所得 ``InferenceData``，但不会用于参考表；每个模型必须恰有
        一个观测随机变量。约束必须编码进规范化的生成式先验，不能依赖 ``Potential``，
        因为 PyMC 的先验预测不会应用 Potential。
    statistics
        应用于每个先验模拟数据集与观测数据集的摘要统计函数。函数可返回标量或
        一维/多维数组；返回值会按固定顺序展平。
    observations
        单个观测数据集。
    n_samples
        每个候选模型从先验和模拟器独立生成的参考表行数。
    size
        为保留旧版调用签名而存在。当前 PyMC 的先验预测形状由模型观测变量决定，
        因此非 ``None`` 值会触发明确错误。
    n_trees
        分类森林和回归森林各自的树数。
    f_max_features
        每次分裂考虑的特征比例，取值范围为 ``(0, 1]``；换算后的整数至少为 1。
    random_seed
        控制各模型先验预测与两座森林的随机种子。

    Returns
    -------
    (best_model, uncertainty)
        从零开始的模型索引，以及回归森林估计的局部误分类概率。后者不是后验
        模型概率。
    """

    if not models:
        raise ValueError("models 不能为空")
    if not statistics:
        raise ValueError("statistics 不能为空")
    if not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError("n_samples 必须是正整数")
    if not isinstance(n_trees, (int, np.integer)) or n_trees <= 0:
        raise ValueError("n_trees 必须是正整数")
    if not (0 < float(f_max_features) <= 1):
        raise ValueError("f_max_features 必须位于 (0, 1] 区间")
    if size is not None:
        raise ValueError("当前 PyMC 先验预测 API 不再接受 size；请在模型中定义观测形状")

    # 延迟导入使章节静态验证不依赖可选的 scikit-learn；只要环境中存在 sklearn，
    # 章节行为测试就必须执行而不能静默跳过。
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    obs_summary = _summarize_one(np.asarray(observations), statistics)
    root_sequence = np.random.SeedSequence(random_seed)
    child_sequences = root_sequence.spawn(len(models) + 2)

    reference_blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for model_index, model_fit in enumerate(models):
        if not isinstance(model_fit, tuple) or len(model_fit) != 2:
            raise TypeError("models 中每一项都必须是 (pm.Model, InferenceData) 对")
        model, _fit_idata = model_fit
        if len(model.observed_RVs) != 1:
            raise ValueError("每个候选模型必须恰有一个观测随机变量")
        if getattr(model, "potentials", ()):
            raise ValueError(
                "先验参考表不能可靠应用 Potential；请把约束改写为规范化的生成式先验"
            )
        observed_name = model.observed_RVs[0].name
        predictive_seed = int(
            np.random.default_rng(child_sequences[model_index]).integers(
                0, np.iinfo(np.int32).max
            )
        )
        predictive = pm.sample_prior_predictive(
            samples=int(n_samples),
            model=model,
            var_names=[observed_name],
            return_inferencedata=True,
            random_seed=predictive_seed,
        )
        rows = _prior_predictive_rows(predictive, observed_name)
        block = _summary_matrix(rows, statistics, expected_width=obs_summary.size)
        reference_blocks.append(block)
        labels.append(np.full(block.shape[0], model_index, dtype=int))

    reference_table = np.vstack(reference_blocks)
    model_labels = np.concatenate(labels)
    n_features = reference_table.shape[1]
    max_features = max(1, min(n_features, int(np.ceil(f_max_features * n_features))))

    classifier_seed = int(
        np.random.default_rng(child_sequences[-2]).integers(0, np.iinfo(np.int32).max)
    )
    classifier = RandomForestClassifier(
        n_estimators=int(n_trees),
        max_features=max_features,
        oob_score=True,
        random_state=classifier_seed,
        n_jobs=1,
    )
    classifier.fit(reference_table, model_labels)
    classification_error = _oob_misclassification_targets(classifier, model_labels)

    regressor_seed = int(
        np.random.default_rng(child_sequences[-1]).integers(0, np.iinfo(np.int32).max)
    )
    regressor = RandomForestRegressor(
        n_estimators=int(n_trees),
        max_features=max_features,
        random_state=regressor_seed,
        n_jobs=1,
    )
    regressor.fit(reference_table, classification_error)

    observed_row = obs_summary.reshape(1, -1)
    best_model = int(classifier.predict(observed_row)[0])
    uncertainty = float(regressor.predict(observed_row)[0])
    return best_model, float(np.clip(uncertainty, 0.0, 1.0))


__all__ = ["select_model"]
