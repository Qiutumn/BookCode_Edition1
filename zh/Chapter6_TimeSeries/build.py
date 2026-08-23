#!/usr/bin/env python3
r"""第 6 章《时间序列》中文版的唯一规范源。

本文件保存经过完整翻译和 TensorFlow Probability 0.25 现代化的单元格源。
它不会在导入时执行模型，也不会把最终 Notebook 或 Org 文件写入本目录。

常用命令：

    python build.py --check
    python build.py --runtime-check
    python build.py --write-notebook /tmp/Ch6_TimeSeries_zh.ipynb

`--check` 会验证清单、稳定单元格 ID、锚点、数据哈希，并静态编译每个代码
单元格；`--runtime-check` 只执行轻量导入和模型构造，不运行完整 MCMC。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tomllib
from typing import Any, Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
TOOLS_DIR = HERE.parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import nb_tools

SOURCE_MARKDOWN = REPO_ROOT / "markdown" / "chp_06.md"
MIGRATION_NOTEBOOK = REPO_ROOT / "notebooks_updated" / "chp_06.ipynb"
MANIFEST_PATH = HERE / "manifest.toml"

SOURCE_HASHES = {
    "markdown/chp_06.md": "59e8ec1336c88160b534f840ab3618eb14751d6626d1e0ba1a6c157b5602e1c2",
    "notebooks_updated/chp_06.ipynb": "19abaef490c651c09560a95a901f3350fcf07cb8e58e254b9633a1acddda6870",
    "data/monthly_mauna_loa_co2.csv": "ba3b354bb28832f3f2df3b52d356475523e83adefe7eceb0d106be3c1c11a6f4",
    "data/monthly_birth_usa.csv": "fdb36a2e564f63c5d72e03d622b81be2e5a4a2e5b404c39c9cb7528a97a7d0bf",
}

# 这些锚点来自 markdown/chp_06.md。错误拼写 chap4 与 kalman_fitler 属于原书
# 公共引用接口，中文版保留它们以避免破坏跨章节链接。
REQUIRED_ANCHORS = {
    "chap4",
    "an-overview-of-time-series-problems",
    "time-series-analysis-as-a-regression-problem",
    "design-matrices-for-time-series",
    "chp4_gam",
    "chap4_ar",
    "latent-ar-process-and-smoothing",
    "sarimax",
    "state-space-models",
    "lgssm_time_series",
    "arima-expressed-as-a-state-space-model",
    "bayesian-structural-time-series",
    "other-time-series-models",
    "model-criticism-and-choosing-priors",
    "priors-for-time-series-models",
    "exercises6",
}

REQUIRED_EQUATION_ANCHORS = {
    "eq:generic_time_series",
    "eq:regression_model",
    "eq:step_linear_function",
    "eq:Fourier_basis_functions",
    "eq:ar1",
    "eq:gw_formulation1",
    "eq:gw_formulation2",
    "eq:arma",
    "eq:sarma",
    "eq:state_space_model",
    "eq:lgssm",
    "eq:lgssm_generative",
    "eq:kalman_fitler",
    "eq:kalman_fitler_preddict_step",
    "eq:kalman_fitler_update_step",
    "eq:linear_growth_state",
    "eq:linear_growth_observed_state",
    "eq:arma_pre_lgssm",
    "eq:arma_lgssm_state_fn",
    "eq:arma_lgssm_state",
    "eq:arma_lgssm_state_full",
    "eq:arima_lgssm_state_transition",
    "eq:combining_lgssm",
    "eq:mape",
    "eq:horse_shoe",
    "eq:step_linear",
}

REQUIRED_FIGURE_ANCHORS = {f"fig:fig{i}_{suffix}" for i, suffix in [
    (1, "co2_by_month"),
    (2, "sparse_design_matrix"),
    (3, "prior_predictive1"),
    (4, "posterior_predictive_components1"),
    (5, "posterior_predictive1"),
    (6, "step_linear_function"),
    (7, "fourier_basis"),
    (8, "prior_predictive2"),
    (9, "posterior_predictive2"),
    (10, "ar1_process"),
    (11, "ar1_likelihood_rho"),
    (12, "posterior_predictive_ar1"),
    (13, "ar1_likelihood_rho2"),
    (14, "smoothing_with_gw"),
    (15, "birth_by_month"),
    (16, "linear_growth_lgssm"),
    (17, "arma_lgssm_inference_result"),
    (18, "bsts_lgssm"),
    (19, "bsts_lgssm_result"),
]}

REQUIRED_TABLE_ANCHORS = {"tab:loo_sarima", "table:ts_model_type"}


def _metadata(
    *,
    lines: str,
    anchors: Sequence[str] = (),
    code_block: str | None = None,
    figures: Sequence[str] = (),
    kind: str,
    notes: str | None = None,
) -> dict[str, Any]:
    zh: dict[str, Any] = {
        "chapter": 6,
        "kind": kind,
        "source_authority": "markdown/chp_06.md",
        "source_lines": lines,
        "migration_start": "notebooks_updated/chp_06.ipynb",
    }
    if anchors:
        zh["anchors"] = list(anchors)
    if code_block:
        zh["code_block"] = code_block
    if figures:
        zh["figures"] = list(figures)
        zh["output_name"] = figures[0].replace(":", "-")
    if notes:
        zh["modernization_notes"] = notes
    provenance: dict[str, Any] = {
        "source": "markdown/chp_06.md" if kind == "translated-prose" else "notebooks_updated/chp_06.ipynb",
        "locator": lines,
        "authority": "complete-prose" if kind == "translated-prose" else "code-modernization-starting-point",
        "prose_authority": "markdown/chp_06.md",
    }
    if code_block:
        provenance["code_block"] = code_block
    metadata: dict[str, Any] = {
        "kind": kind,
        "provenance": provenance,
        "zh": zh,
    }
    if anchors:
        metadata["anchors"] = list(anchors)
    if figures:
        metadata["figures"] = list(figures)
        metadata["output_name"] = figures[0].replace(":", "-")
    return metadata


def md(
    cell_id: str,
    source: str,
    *,
    lines: str,
    anchors: Sequence[str] = (),
    notes: str | None = None,
) -> dict[str, Any]:
    return {
        "type": "markdown",
        "id": cell_id,
        "source": source.strip("\n") + "\n",
        "metadata": _metadata(
            lines=lines, anchors=anchors, kind="translated-prose", notes=notes
        ),
    }


def code(
    cell_id: str,
    source: str,
    *,
    lines: str,
    code_block: str | None = None,
    figures: Sequence[str] = (),
    notes: str | None = None,
) -> dict[str, Any]:
    return {
        "type": "code",
        "id": cell_id,
        "source": source.strip("\n") + "\n",
        "metadata": _metadata(
            lines=lines,
            code_block=code_block,
            figures=figures,
            kind="modernized-code",
            notes=notes,
        ),
    }


cells: list[dict[str, Any]] = [
    md(
        "ch06-title-introduction",
        r"""
(chap4)=

# 第 6 章 时间序列

> “预测很困难，尤其是预测未来。”

据说，荷兰政治家 Karl Kristian Steincke 在 20 世纪 40 年代的某个时候说过这句话[^1]。时至今日，这句话依然成立；对从事时间序列与预测问题的人来说尤其如此。时间序列分析有许多用途：既可以通过预测推断未来，也可以帮助我们理解历史趋势背后潜在的驱动因素。

本章讨论这个问题的若干贝叶斯方法。我们先把时间序列建模视为一个回归问题，从时间戳信息中解析出设计矩阵；随后探索如何用自回归成分描述时间相关性。这些模型可以推广到更宽泛、更一般的状态空间模型与贝叶斯结构时间序列（Bayesian Structural Time Series，BSTS）。在线性高斯情形下，我们还会介绍一种专用推断方法——卡尔曼滤波。最后，本章将简要讨论模型比较，以及为时间序列模型选择先验时应考虑的问题。

> **中文版现代化说明**：本章代码以原书更新版 Notebook 为迁移起点，但统一改写为本地环境中 TensorFlow 2.21 与 TensorFlow Probability 0.25 可用的公共 API。所有完整轨迹自回归示例均使用公开的 TensorFlow 运算或公开分布实现；不再导入 `tensorflow_probability.python.internal`。原迁移 Notebook 中损坏的 `joint_distribution` 行也已重建为明确的 `tfp.distributions` 公共别名。
        """,
        lines="L1-L21",
        anchors=("chap4",),
        notes="新增明确标注的 TFP 0.25 现代化说明。",
    ),
    md(
        "ch06-overview",
        r"""
(an-overview-of-time-series-problems)=

## 时间序列问题概览

在许多现实应用中，我们按时间顺序连续观测数据；每次得到观测值时，也同时生成一个时间戳。除观测值本身之外，时间戳在以下情况下也可能包含大量信息：

- 存在随时间变化的**趋势（trend）**。例如地区人口、全球 GDP、美国年度 CO₂ 排放量。我们通常会凭直觉把这种总体模式称为“增长”或“下降”。
- 存在与时间相关的重复模式，称为**季节性（seasonality）**[^2]。例如月度气温（夏季较高、冬季较低）、月度降雨量（世界许多地区冬季较少、夏季较多）、某办公楼每天的咖啡消费量（工作日较高、周末较低），以及第 [5](chap3_5) 章见过的逐小时自行车租赁量（白天高于夜间）。
- 当前数据点会以某种方式为下一个数据点提供信息。换句话说，噪声或**残差（residuals）**在时间上相关[^3]。例如服务台每天解决的工单数、股票价格、逐小时气温和逐小时降雨量。

因此，把时间序列自然地分解为以下三部分通常很有用：

<a id="eq:generic_time_series"></a>
$$
y_t = \text{趋势}_t + \text{季节性}_t + \text{残差}_t
$$

许多经典时间序列模型都建立在这种分解上。本章将讨论如何为具有某种时间趋势和季节性的序列建模：既要捕捉这些规则模式，也要捕捉不那么规则的模式，例如随时间相关的残差。
        """,
        lines="L23-L64",
        anchors=("an-overview-of-time-series-problems", "eq:generic_time_series"),
    ),
    code(
        "ch06-runtime-setup",
        r"""
from __future__ import annotations

import gc
import hashlib
import os
from collections.abc import Mapping
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd
from scipy import linalg, stats
import tensorflow as tf
import tensorflow_probability as tfp

# 只使用公开命名空间；本章不导入 tensorflow_probability.python.internal。
tfd = tfp.distributions
tfb = tfp.bijectors

EXECUTION_PROFILE = os.environ.get("BMCP_EXECUTION_PROFILE", "release").lower()
if EXECUTION_PROFILE not in {"smoke", "release"}:
    raise ValueError("BMCP_EXECUTION_PROFILE 必须是 smoke 或 release")

BUDGETS = {
    "smoke": {
        "draws": 32,
        "adapt": 32,
        "chains": 2,
        "prior_samples": 24,
        "posterior_predictive_samples": 24,
        "sts_parameter_samples": 16,
    },
    "release": {
        "draws": 1000,
        "adapt": 1000,
        "chains": 4,
        "prior_samples": 100,
        "posterior_predictive_samples": 200,
        "sts_parameter_samples": 100,
    },
}
BUDGET = BUDGETS[EXECUTION_PROFILE]
BASE_SEED = 20210606
tf.random.set_seed(BASE_SEED)

az.style.use("arviz-grayscale")
plt.rcParams.update({"figure.dpi": 144, "axes.grid": False})


def split_seed(salt: str, n: int = 2):
    '''通过公开 tfp.random API 生成可复现的无状态种子。'''
    seed = tfp.random.sanitize_seed(BASE_SEED, salt=salt)
    return tfp.random.split_seed(seed, n=n, salt=f"{salt}-split")


def cell_numpy_rng(cell_id: str) -> np.random.Generator:
    '''为每个代码单元派生互不共享状态的确定性 NumPy RNG。'''
    digest = hashlib.sha256(f"{BASE_SEED}:{cell_id}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def flatten_draw_chain(value, *, name: str):
    '''把 TFP 的 [draw, chain, ...] 结果确定性展平为 [sample, ...]。'''
    tensor = tf.convert_to_tensor(value)
    if tensor.shape.rank is None or tensor.shape.rank < 2:
        raise ValueError(f"{name} 必须至少具有 draw 与 chain 两个轴")
    return tf.reshape(
        tensor,
        tf.concat([[tf.shape(tensor)[0] * tf.shape(tensor)[1]], tf.shape(tensor)[2:]], axis=0),
    )


def select_sample_budget(value, sample_count: int, *, name: str):
    '''从已展平样本中等距、无重复地选择恰好 sample_count 个样本。'''
    tensor = tf.convert_to_tensor(value)
    total = tensor.shape[0]
    if total is None:
        total = int(tf.shape(tensor)[0].numpy())
    else:
        total = int(total)
    if total < sample_count:
        raise ValueError(f"{name} 只有 {total} 个样本，少于预算 {sample_count}")
    indices = np.linspace(0, total - 1, num=sample_count, dtype=np.int64)
    if len(np.unique(indices)) != sample_count:
        raise AssertionError(f"{name} 的确定性样本索引出现重复")
    selected = tf.gather(tensor, indices, axis=0)
    if selected.shape[0] != sample_count:
        raise AssertionError(f"{name} 未选择恰好 {sample_count} 个样本")
    return selected


def flatten_and_select(value, sample_count: int, *, name: str):
    return select_sample_budget(
        flatten_draw_chain(value, name=name), sample_count, name=name
    )


def release_resources(*names: str) -> None:
    '''在重型章节之间释放图形、Python 对象与 TensorFlow 会话缓存。'''
    namespace = globals()
    for name in names:
        namespace.pop(name, None)
    plt.close("all")
    tf.keras.backend.clear_session()
    gc.collect()


def run_windowed_nuts(model, *, seed, jit_compile=False, **pins):
    '''用一致的 smoke/release 预算运行 TFP 的公共自适应 NUTS。

    中文版现代化说明：默认 jit_compile=False，因为本章几个模型内部用
    LinearGaussianStateSpaceModel（latent-AR、ARMA、BSTS）构造，其卡尔曼
    滤波递推依赖动态长度的 TensorArray/while_loop，XLA 编译会报错
    "XLA compilation requires a fixed tensor list size"——这是已验证的
    硬性限制，不是本地环境问题。不含 LGSSM 的模型（如下方 GAM）经验证在
    jit_compile=True 下结果一致，且在缺少 PyTensor/TF 原生加速的宿主上快
    约 4 倍（356s → 90s，相同 smoke 预算），因此按调用点显式开启。
    '''
    sampler = tf.function(
        tfp.experimental.mcmc.windowed_adaptive_nuts,
        autograph=False,
        jit_compile=jit_compile,
    )
    return sampler(
        BUDGET["draws"],
        model,
        n_chains=BUDGET["chains"],
        num_adaptation_steps=BUDGET["adapt"],
        seed=seed,
        **pins,
    )


def tfp_draws_to_idata(draws, stats_dict=None):
    '''把 TFP 的 (draw, chain, ...) 结果转换成 ArviZ 的 (chain, draw, ...)。'''
    if hasattr(draws, "_asdict"):
        items = draws._asdict().items()
    elif isinstance(draws, Mapping):
        items = draws.items()
    else:
        raise TypeError("draws 必须是 namedtuple 或映射")
    posterior = {name: np.swapaxes(np.asarray(value), 0, 1) for name, value in items}
    sample_stats = None
    if stats_dict:
        sample_stats = {
            name: np.swapaxes(np.asarray(stats_dict[name]), 0, 1)
            for name in ("target_log_prob", "diverging", "accept_ratio", "n_steps")
            if name in stats_dict
        }
    return az.from_dict(posterior=posterior, sample_stats=sample_stats)


def ar1_scale_tril(rho, innovation_scale, num_timesteps):
    '''构造完整 AR(1) 轨迹的公开 MVN Cholesky 因子。

    该自包含实现把独立创新映射为 x[t] = rho*x[t-1] + eps[t]，
    从而保留所有时间步，而不是只保留 tfd.Autoregressive 的末状态。
    '''
    rho = tf.convert_to_tensor(rho)
    innovation_scale = tf.convert_to_tensor(innovation_scale, dtype=rho.dtype)
    row = tf.range(num_timesteps)[:, None]
    col = tf.range(num_timesteps)[None, :]
    lag = row - col
    mask = lag >= 0
    safe_lag = tf.maximum(lag, 0)
    powers = tf.where(
        mask,
        tf.pow(rho[..., None, None], tf.cast(safe_lag, rho.dtype)),
        tf.zeros([], rho.dtype),
    )
    return innovation_scale[..., None, None] * powers


def ar1_trajectory_distribution(rho, innovation_scale, num_timesteps, name="ar1"):
    scale_tril = ar1_scale_tril(rho, innovation_scale, num_timesteps)
    loc = tf.zeros(tf.concat([tf.shape(rho), [num_timesteps]], axis=0), dtype=rho.dtype)
    return tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril, name=name)


def ar1_lgssm(rho, ar_sigma, noise_sigma, num_timesteps, name="ar1_lgssm"):
    '''把标量 AR(1) 潜在过程构造成 LinearGaussianStateSpaceModel。

    与 ``ar1_trajectory_distribution``（稠密 num_timesteps×num_timesteps
    协方差的 MVN）不同，这里通过卡尔曼滤波递推解析边缘化潜在轨迹：
    对数密度与梯度的计算量随 num_timesteps 线性增长，而不是平方甚至立方增长。
    对较长序列（数百步以上），联合采样整条稠密协方差路径会让 NUTS 的每一次
    梯度求值都很昂贵，实践中可能耗尽内存或使内核挂起；用本函数配合
    ``TransformedDistribution(lgssm, tfb.Shift(baseline))`` 把已知的
    均值分量（如 GAM 的趋势加季节性）加到潜在 AR 分量上，只对全局参数
    （rho、ar_sigma 及噪声尺度）做 NUTS 采样，再用 ``posterior_sample``
    或 ``posterior_marginals`` 事后恢复潜在轨迹，比联合采样整条路径快得多。

    观测/状态维度恒为 1；调用方需要给观测数据附加长度为 1 的末尾维度。
    '''
    rho = tf.convert_to_tensor(rho)
    dtype = rho.dtype
    ar_sigma = tf.convert_to_tensor(ar_sigma, dtype=dtype)
    noise_sigma = tf.convert_to_tensor(noise_sigma, dtype=dtype)
    transition_matrix = tf.linalg.LinearOperatorFullMatrix(rho[..., None, None])
    transition_noise = tfd.MultivariateNormalDiag(scale_diag=ar_sigma[..., None])
    observation_matrix = tf.linalg.LinearOperatorIdentity(1, dtype=dtype)
    observation_noise = tfd.MultivariateNormalDiag(scale_diag=noise_sigma[..., None])
    # 平稳分布的方差 ar_sigma**2 / (1 - rho**2) 作为初始状态先验，避免链条起点
    # 依赖任意选择、且与后续步骤的平稳分布不一致。
    stationary_scale = ar_sigma / tf.sqrt(1.0 - tf.square(rho))
    initial_state_prior = tfd.MultivariateNormalDiag(
        loc=tf.zeros_like(rho)[..., None], scale_diag=stationary_scale[..., None]
    )
    return tfd.LinearGaussianStateSpaceModel(
        num_timesteps=num_timesteps,
        transition_matrix=transition_matrix,
        transition_noise=transition_noise,
        observation_matrix=observation_matrix,
        observation_noise=observation_noise,
        initial_state_prior=initial_state_prior,
        name=name,
    )


assert tfp.__version__.startswith("0.25"), tfp.__version__
assert int(tf.__version__.split(".")[0]) >= 2
        """,
        lines="migration-notebook setup cells",
        code_block="chapter_runtime_setup",
        notes="确定性预算、公共 TFP 随机 API、无跳过路径、重型章节资源清理。",
    ),
    md(
        "ch06-regression-introduction",
        r"""
(time-series-analysis-as-a-regression-problem)=

## 把时间序列分析视为回归问题

我们先用线性回归模型分析一个广泛用于教程的演示数据集。PyMC、TensorFlow Probability 等教程都用过它，Rasmussen 与 Williams 的《Gaussian Processes for Machine Learning》也以它为例 {cite:t}`Rasmussen2005`。自 20 世纪 50 年代末以来，夏威夷莫纳罗亚观测站一直以逐小时频率定期测量大气 CO₂ 浓度；许多示例会把这些观测聚合成月平均值。

下面的代码加载月度数据，并把最后十年留作测试集。模型只用训练集拟合，再把预测与测试集比较。
        """,
        lines="L66-L82",
        anchors=("time-series-analysis-as-a-regression-problem",),
    ),
    code(
        "ch06-load-co2-data",
        r"""
co2_by_month = pd.read_csv("data/monthly_mauna_loa_co2.csv")
co2_by_month["date_month"] = pd.to_datetime(co2_by_month["date_month"])
co2_by_month["CO2"] = co2_by_month["CO2"].astype(np.float32)
co2_by_month.set_index("date_month", drop=True, inplace=True)

num_forecast_steps = 12 * 10  # 用此前数据预测最后十年
co2_training = co2_by_month.iloc[:-num_forecast_steps].copy()
co2_testing = co2_by_month.iloc[-num_forecast_steps:].copy()

assert len(co2_training) + len(co2_testing) == len(co2_by_month)
assert co2_training.index.max() < co2_testing.index.min()
assert np.isfinite(co2_by_month["CO2"]).all()

fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
ax.plot(co2_training.index, co2_training["CO2"], color="0.15", lw=1.8, label="训练集")
ax.plot(co2_testing.index, co2_testing["CO2"], color="0.55", lw=1.8, ls="--", label="测试集")
ax.set(xlabel="年份", ylabel="CO₂（ppm）")
ax.legend(frameon=False)
fig.autofmt_xdate()
        """,
        lines="L83-L104",
        code_block="load_co2_data",
        figures=("fig:fig1_co2_by_month",),
        notes="路径改为章节本地 data；使用 iloc 避免未来 pandas 索引语义变化。",
    ),
    md(
        "ch06-co2-figure-and-regression-equation",
        r"""
<a id="fig:fig1_co2_by_month"></a>
**图 6.1**　1966 年 1 月至 2019 年 1 月莫纳罗亚的月度 CO₂ 测量值。黑色实线是训练集，灰色虚线是测试集。数据同时呈现强烈的上升趋势和季节性模式。

这里有一个月度大气 CO₂ 浓度观测向量 $y_t$，其中 $t=[0,\dots,636]$，每个元素都对应一个时间戳。可以把一年中的月份解析成 $[1,2,3,\dots,12,1,2,\dots]$。回顾线性回归，其似然可以写成：

<a id="eq:regression_model"></a>
$$
Y \sim \mathcal{N}(\mathbf{X}\beta,\sigma)
$$

为表示季节性效应，我们可以直接用月份预测变量索引一组回归系数。下面把月份虚拟编码成形状为 `(637, 12)` 的设计矩阵，再增加一个线性时间预测变量，以捕捉数据中的上升趋势。
        """,
        lines="L83-L127",
        anchors=("fig:fig1_co2_by_month", "eq:regression_model"),
    ),
    code(
        "ch06-generate-design-matrix",
        r"""
trend_all = np.linspace(0.0, 1.0, len(co2_by_month), dtype=np.float32)[:, None]
trend = trend_all[:-num_forecast_steps]

seasonality_all = pd.get_dummies(co2_by_month.index.month, dtype=np.float32).to_numpy()
seasonality = seasonality_all[:-num_forecast_steps]

X_training = np.concatenate([trend, seasonality], axis=-1)
assert X_training.shape == (len(co2_training), 13)
assert np.allclose(seasonality.sum(axis=1), 1.0)

fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
X_subset = X_training[-50:]
image = ax.imshow(X_subset.T, aspect="auto", cmap="Greys", interpolation="nearest")
ax.set(xlabel="最近 50 个训练时间点", ylabel="特征索引")
fig.colorbar(image, ax=ax, shrink=0.8, label="设计矩阵取值")
        """,
        lines="L140-L155",
        code_block="generate_design_matrix",
        figures=("fig:fig2_sparse_design_matrix",),
        notes="pd.get_dummies 显式指定 float32；增加形状与独热编码语义断言。",
    ),
    md(
        "ch06-design-matrix-caption-admonition",
        r"""
<a id="fig:fig2_sparse_design_matrix"></a>
**图 6.2**　用于简单时间序列回归的设计矩阵：包含一个线性成分和一年中月份成分。为了便于观察，图中把设计矩阵转置成“特征 × 时间戳”。第 0 行是 0 到 1 之间的连续值，表示时间和线性增长；其余第 1–12 行是月份的虚拟编码。黑色表示 1，浅灰表示 0。

> **把时间戳解析为设计矩阵**
>
> 处理时间戳可能既繁琐又容易出错，尤其当问题涉及时区时。常见的周期信息按时间分辨率由细到粗包括：
>
> - 一小时中的秒（1, 2, …, 60）；
> - 一天中的小时（1, 2, …, 24）；
> - 一周中的星期几（星期一、星期二、…、星期日）；
> - 一个月中的第几天（1, 2, …, 31）；
> - 节假日效应（元旦、复活节、国际劳动节、圣诞节等）；
> - 一年中的月份（1, 2, …, 12）。
>
> 这些信息都可以用虚拟编码解析成设计矩阵。星期几和每月日期等效应通常与人类活动密切相关。例如，公共交通客流通常有很强的工作日效应；消费者支出可能在发薪日之后较高，而发薪日常在月底附近。本章主要考虑按规则时间间隔记录的时间戳。
        """,
        lines="L128-L183",
        anchors=("fig:fig2_sparse_design_matrix",),
    ),
    md(
        "ch06-regression-model-prose",
        r"""
现在可以把第一个时间序列模型写成回归问题。我们沿用第 [3](chap2) 章介绍的 `tfd.JointDistributionCoroutine` 思路，但使用 0.25 版中公开的自动批处理变体，以减少手工处理批形状的样板代码。

TFP 比 PyMC 提供更低层的 API。低层模块与组件更灵活，例如可以组合定制的推断方法；代价是通常需要写更多样板代码，并更仔细地处理形状。下面的模型仍显式使用 `einsum`，使线性预测能正确广播到任意链或样本批形状。关于批形状的更多说明参见 {ref}`shape_ppl`。
        """,
        lines="L185-L234",
    ),
    code(
        "ch06-regression-model",
        r"""
@tfd.JointDistributionCoroutineAutoBatched
def ts_regression_model():
    intercept = yield tfd.Normal(0.0, 100.0, name="intercept")
    trend_coeff = yield tfd.Normal(0.0, 10.0, name="trend_coeff")
    seasonality_coeff = yield tfd.Sample(
        tfd.Normal(0.0, 1.0),
        sample_shape=seasonality.shape[-1],
        name="seasonality_coeff",
    )
    noise_sigma = yield tfd.HalfCauchy(0.0, 5.0, name="noise_sigma")
    y_hat = (
        intercept[..., None]
        + tf.einsum("ij,...->...i", trend, trend_coeff)
        + tf.einsum("ij,...j->...i", seasonality, seasonality_coeff)
    )
    yield tfd.Independent(
        tfd.Normal(y_hat, noise_sigma[..., None]),
        reinterpreted_batch_ndims=1,
        name="observed",
    )

regression_shapes = ts_regression_model.sample_distributions(seed=split_seed("regression-shape")[0])[0]
assert ts_regression_model.event_shape.observed[-1] == len(co2_training)
        """,
        lines="L190-L213",
        code_block="regression_model_for_timeseries",
        notes="修复损坏 joint_distribution 行；改用公开 JointDistributionCoroutineAutoBatched。",
    ),
    md(
        "ch06-regression-prior-predictive-prose",
        r"""
生成的 `ts_regression_model` 和其他 `tfd.Distribution` 一样，可以用于贝叶斯工作流。调用 `.sample(.)` 即可抽取先验与先验预测样本。这里的先验刻意较宽，因此先验预测的范围也很宽。
        """,
        lines="L226-L258",
    ),
    code(
        "ch06-regression-prior-predictive",
        r"""
prior_samples = ts_regression_model.sample(
    BUDGET["prior_samples"], seed=split_seed("regression-prior")[0]
)
prior_predictive_timeseries = np.asarray(prior_samples.observed)
assert prior_predictive_timeseries.shape[-1] == len(co2_training)

fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
ax.plot(
    co2_training.index,
    prior_predictive_timeseries.T,
    color="0.45",
    alpha=0.20,
    lw=0.8,
)
ax.set(xlabel="年份", ylabel="CO₂（ppm）")
fig.autofmt_xdate()
        """,
        lines="L236-L249",
        code_block="prior_predictive",
        figures=("fig:fig3_prior_predictive1",),
    ),
    md(
        "ch06-regression-prior-caption-inference",
        r"""
<a id="fig:fig3_prior_predictive1"></a>
**图 6.3**　简单回归时间序列模型对莫纳罗亚月度 CO₂ 的先验预测样本。每条线是一条模拟时间序列。由于先验信息量很弱，先验预测覆盖的范围相当宽。

接下来对回归模型运行推断，并把结果整理为 `az.InferenceData`。`smoke` 与 `release` 两种配置会走完全相同的推断路径，只改变确定性的抽样、适应与链数预算；不会跳过任何代码分支。
        """,
        lines="L251-L286",
        anchors=("fig:fig3_prior_predictive1",),
    ),
    code(
        "ch06-regression-inference",
        r"""
regression_seed = split_seed("regression-mcmc")[0]
regression_draws, regression_stats = run_windowed_nuts(
    ts_regression_model,
    seed=regression_seed,
    observed=tf.convert_to_tensor(co2_training["CO2"].to_numpy(), tf.float32),
)
regression_idata = tfp_draws_to_idata(regression_draws, regression_stats)

assert set(regression_idata.posterior.data_vars) >= {
    "intercept", "trend_coeff", "seasonality_coeff", "noise_sigma"
}
        """,
        lines="L265-L286 plus migration notebook inference cell",
        code_block="inference_of_regression_model",
        notes="公共 windowed_adaptive_nuts；预算化但无跳过路径；显式种子。",
    ),
    md(
        "ch06-regression-posterior-predictive-prose",
        r"""
要从推断结果得到后验预测，可以按后验抽样构造预测分布。这里还希望分别画出趋势和季节性成分，并同时覆盖训练集和测试集，因此直接构造各个成分。
        """,
        lines="L288-L320",
    ),
    code(
        "ch06-regression-posterior-components",
        r"""
regression_posterior = regression_idata.posterior.stack(sample=("chain", "draw"))
intercept_draws = tf.convert_to_tensor(regression_posterior["intercept"].values, tf.float32)
trend_coeff_draws = tf.convert_to_tensor(regression_posterior["trend_coeff"].values, tf.float32)
seasonality_coeff_values = np.asarray(
    regression_posterior["seasonality_coeff"].values
)
if seasonality_coeff_values.shape[0] != seasonality_all.shape[1]:
    seasonality_coeff_values = np.moveaxis(seasonality_coeff_values, -1, 0)
seasonality_coeff_draws = tf.convert_to_tensor(seasonality_coeff_values, tf.float32)
noise_draws = tf.convert_to_tensor(regression_posterior["noise_sigma"].values, tf.float32)

trend_posterior = (
    intercept_draws[None, :]
    + tf.convert_to_tensor(trend_all, tf.float32) * trend_coeff_draws[None, :]
)
seasonality_posterior = tf.einsum(
    "ij,jk->ik", seasonality_all, seasonality_coeff_draws
)
y_hat_regression = trend_posterior + seasonality_posterior
regression_pp_dist = tfd.Normal(y_hat_regression, noise_draws[None, :])
regression_pp = regression_pp_dist.sample(seed=split_seed("regression-pp")[0])

assert trend_posterior.shape[0] == len(co2_by_month)
assert seasonality_posterior.shape == trend_posterior.shape
assert regression_pp.shape == trend_posterior.shape
        """,
        lines="L301-L319",
        code_block="posterior_predictive_with_component",
        notes="使用 xarray stack 后显式整理维度；所有预测随机性都有无状态种子。",
    ),
    code(
        "ch06-regression-component-plot",
        r"""
def plot_interval(ax, dates, draws_time_by_sample, label, *, linestyle="-"):
    values = np.asarray(draws_time_by_sample)
    mean = values.mean(axis=-1)
    low, high = np.quantile(values, [0.03, 0.97], axis=-1)
    line = ax.plot(dates, mean, color="0.15", lw=1.8, ls=linestyle, label=label)[0]
    ax.fill_between(dates, low, high, color="0.55", alpha=0.25, linewidth=0)
    return line

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
plot_interval(axes[0], co2_by_month.index, trend_posterior, "趋势后验均值")
axes[0].set_ylabel("趋势（ppm）")
axes[0].legend(frameon=False)
plot_interval(axes[1], co2_by_month.index, seasonality_posterior, "季节性后验均值")
axes[1].set(xlabel="年份", ylabel="季节性（ppm）")
axes[1].legend(frameon=False)

fig2, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
plot_interval(ax, co2_by_month.index, regression_pp, "后验预测")
ax.plot(co2_training.index, co2_training["CO2"], color="0.05", lw=1.3, label="训练集")
ax.plot(co2_testing.index, co2_testing["CO2"], color="0.45", lw=1.3, ls="--", label="测试集")
ax.axvline(co2_testing.index[0], color="0.25", ls=":", lw=1)
ax.set(xlabel="年份", ylabel="CO₂（ppm）")
ax.legend(frameon=False, ncol=3)
        """,
        lines="migration notebook figure cells for source L321-L336",
        figures=("fig:fig4_posterior_predictive_components1", "fig:fig5_posterior_predictive1"),
    ),
    md(
        "ch06-regression-component-captions",
        r"""
<a id="fig:fig4_posterior_predictive_components1"></a>
**图 6.4**　时间序列回归模型中趋势成分与季节性成分的后验预测。实线表示后验均值，灰色带表示 94% 区间。

<a id="fig:fig5_posterior_predictive1"></a>
**图 6.5**　简单时间序列回归的后验预测：预测区间为灰色，实际训练数据为黑色，测试数据为灰色虚线。模型对训练集的总体拟合尚可，但样本外预测较差，因为真实趋势的加速程度超过线性增长。

观察图 6.5 的样本外预测，可以发现：

1. 预测到更远未来时，线性趋势表现不佳，预测系统性低于实际观测。大气 CO₂ 并不是多年始终以恒定斜率线性增长[^4]。
2. 不确定性范围几乎不变（有时也称“预测锥”），但直觉上，预测越远，不确定性应该越大。
        """,
        lines="L321-L349",
        anchors=(
            "fig:fig4_posterior_predictive_components1",
            "fig:fig5_posterior_predictive1",
        ),
    ),
    md(
        "ch06-design-matrices-trend",
        r"""
(design-matrices-for-time-series)=

### 时间序列的设计矩阵

上面的回归模型使用了相当简化的设计矩阵。把额外信息加入设计矩阵，可以让模型更好地表达我们对观测序列的认识。

一般来说，更好的趋势成分是改善预测表现最重要的一环：季节性成分*通常*是平稳的[^5]，其参数也较容易估计。换句话说，季节模式会重复出现，形成某种重复测量。因此，时间序列建模的大量工作都在设计一个能真实捕捉趋势非平稳性的潜在过程。

一种相当成功的方法，是把趋势建模为局部线性过程：在某个局部范围内近似线性，但截距和斜率会在观测时间跨度内缓慢变化或漂移。Facebook Prophet[^6] 是典型例子；它用半平滑的分段线性函数表示趋势 {cite:p}`TaylorLetham2018`。允许斜率在指定变点发生变化后，趋势线比单一直线更能捕捉长期趋势。这与 {ref}`expanding_feature_space` 一节的指示函数思想相似。

<a id="eq:step_linear_function"></a>
$$
g(t)=(k+\mathbf{A}\delta)t+(m+\mathbf{A}\gamma)
$$

其中 $k$ 是全局增长率，$\delta$ 是每个变点处的速率调整向量，$m$ 是全局截距。$\mathbf{A}$ 的形状为 `(n_t, n_s)`，$n_s$ 是变点数；在时间 $t$，它累积斜率漂移 $\delta$ 的影响。令 $\gamma_j=-s_j\delta_j$（$s_j$ 是第 $j$ 个变点的时间位置），可以使趋势线保持连续。通常为 $\delta$ 选择 Laplace 等正则化先验，以表达我们不期待斜率突然或大幅变化。
        """,
        lines="L350-L397",
        anchors=("design-matrices-for-time-series", "eq:step_linear_function"),
    ),
    code(
        "ch06-step-linear-function",
        r"""
n_changepoints = 8
n_timepoints_demo = 500
t_demo = np.linspace(0.0, 1.0, n_timepoints_demo)
changepoints_demo = np.linspace(0.0, 1.0, n_changepoints + 2)[1:-1]
A_demo = (t_demo[:, None] > changepoints_demo).astype(np.float32)

k_demo, m_demo = 2.5, 40.0
rng = cell_numpy_rng("ch06-step-linear-function")
delta_demo = rng.laplace(loc=0.0, scale=0.1, size=n_changepoints)
growth_demo = (k_demo + A_demo @ delta_demo) * t_demo
offset_demo = m_demo + A_demo @ (-changepoints_demo * delta_demo)
trend_demo = growth_demo + offset_demo

assert np.all(np.isfinite(trend_demo))
assert np.max(np.abs(np.diff(trend_demo))) < 1.0

fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), constrained_layout=True)
axes[0].imshow(A_demo.T, aspect="auto", cmap="Greys", interpolation="nearest")
axes[0].set(title=r"$\mathbf{A}$", xlabel="时间", ylabel="变点")
axes[1].plot(t_demo, growth_demo, color="0.15", lw=1.8)
axes[1].set_title("增长项")
axes[2].plot(t_demo, offset_demo, color="0.15", lw=1.8, ls="--")
axes[2].set_title("偏移项")
axes[3].plot(t_demo, trend_demo, color="0.05", lw=2.0)
axes[3].set_title(r"$g(t)$")
for ax in axes[1:]:
    ax.set_xlabel("缩放后的时间")
        """,
        lines="L398-L413 plus migration notebook plot cell",
        code_block="step_linear_function_for_trend",
        figures=("fig:fig6_step_linear_function",),
    ),
    md(
        "ch06-step-linear-caption-changepoints",
        r"""
<a id="fig:fig6_step_linear_function"></a>
**图 6.6**　分段线性趋势函数。第一个面板是设计矩阵 $\mathbf{A}$，黑色为 1、浅灰为 0；中间两个面板分别展示式 {eq}`eq:step_linear_function` 的增长项和偏移项；最后一个面板是可用于时间序列趋势的 $g(t)$。两部分组合后，最终趋势保持连续。

实践中，我们通常预先指定变点数量，使 $\mathbf{A}$ 可以静态生成。一种常见策略是指定比序列实际呈现更多的变点，再为 $\delta$ 设置更稀疏的先验，把后验正则化到 0 附近。也可以进行自动变点检测 {cite:p}`adams2007bayesian`。
        """,
        lines="L415-L435",
        anchors=("fig:fig6_step_linear_function",),
    ),
    md(
        "ch06-basis-functions-gam",
        r"""
(chp4_gam)=

### 基函数与广义加性模型

在前面的回归模型中，我们用稀疏索引矩阵表示季节性。另一种方法是使用 B 样条等基函数（见第 [5](chap3_5) 章），或 Facebook Prophet 所用的 Fourier 基函数。基函数形成的设计矩阵可能具有正交性等良好性质（见“设计矩阵的数学性质”），因而更容易稳定地求解线性方程 {cite:p}`strang09`。

Fourier 基函数是一组正弦与余弦函数，可以逼近任意平滑的季节效应 {cite:p}`109876`：

<a id="eq:Fourier_basis_functions"></a>
$$
s(t)=\sum_{n=1}^{N}\left[a_n\cos\left(\frac{2\pi nt}{P}\right)+b_n\sin\left(\frac{2\pi nt}{P}\right)\right]
$$

$P$ 是序列的规则周期。例如，以“天”为时间单位时，年度数据可取 $P=365.25$，周度数据可取 $P=7$。下面静态生成这些基函数。
        """,
        lines="L437-L467",
        anchors=("chp4_gam", "eq:Fourier_basis_functions"),
    ),
    code(
        "ch06-fourier-basis",
        r"""
def gen_fourier_basis(t, p=365.25, n=3):
    t = np.asarray(t, dtype=np.float64)
    x = 2.0 * np.pi * (np.arange(n) + 1) * t[:, None] / p
    return np.concatenate((np.cos(x), np.sin(x)), axis=1).astype(np.float32)

n_timepoints_fourier = 500
period_months = 12
t_monthly = np.asarray([i % period_months for i in range(n_timepoints_fourier)])
monthly_X = gen_fourier_basis(t_monthly, p=period_months, n=3)
assert monthly_X.shape == (n_timepoints_fourier, 6)

fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
for idx, series in enumerate(monthly_X.T):
    ax.plot(
        series[:72],
        color="0.10" if idx == 0 else "0.65",
        alpha=1.0 if idx == 0 else 0.45,
        lw=2.0 if idx == 0 else 1.0,
        label="第一基函数" if idx == 0 else None,
    )
ax.set(xlabel="月份索引", ylabel="基函数取值")
ax.legend(frameon=False)
        """,
        lines="L468-L481",
        code_block="fourier_basis_as_seasonality",
        figures=("fig:fig7_fourier_basis",),
    ),
    md(
        "ch06-fourier-caption-design-properties",
        r"""
<a id="fig:fig7_fourier_basis"></a>
**图 6.7**　$n=3$ 的 Fourier 基函数，共有 6 个预测变量。第一条曲线用深色强调，其余曲线以半透明浅灰表示。

用上述 Fourier 基函数设计矩阵拟合季节性，需要估计 $2N$ 个参数 $\beta=[a_1,b_1,\dots,a_N,b_N]$。Facebook Prophet 一类回归模型也常被称为**广义加性模型（Generalized Additive Model，GAM）**，因为响应变量 $Y_t$ 线性依赖于未知的平滑基函数[^7]。第 [5](chap3_5) 章也讨论过其他 GAM。

> **设计矩阵的数学性质**
>
> 在线性最小二乘问题中，设计矩阵的数学性质受到广泛研究：我们希望针对 $\beta$ 最小化 $\lVert Y-\mathbf{X}\beta\rVert^2$。检查 $\mathbf{X}^{T}\mathbf{X}$ 的性质，往往能判断 $\beta$ 的解是否稳定，甚至判断解是否存在。
>
> 一个重要性质是条件数，它能提示 $\beta$ 的解是否容易产生很大的数值误差。例如，若设计矩阵含有高度相关的列（多重共线性），条件数会很大，$\mathbf{X}^{T}\mathbf{X}$ 就是病态的。类似原则也适用于贝叶斯建模。无论采用哪种正式建模方法，深入的探索性数据分析都很有价值。由基函数组成的设计矩阵通常具有较好的条件数。
        """,
        lines="L483-L515",
        anchors=("fig:fig7_fourier_basis",),
    ),
    md(
        "ch06-gam-model-prose",
        r"""
下面为月度 CO₂ 构造一个类似 Facebook Prophet 的 GAM。我们为 `k` 与 `m` 设置弱信息先验，表达月度观测总体向上增长的知识。这样，先验预测会落在与实际观测相近的量级，而不再像图 6.3 那样过度宽泛。

> **中文版现代化说明**：变点位置 `s_gam` 在完整序列（含训练集与测试集）的 $[0,1]$ 归一化时间轴上等距排列；由于测试集占最后约 19% 的时间跨度，12 个变点中有 2 个落在训练区间之外。训练阶段这两个变点对应的 `A_gam` 列在训练数据上恒为 0，相应的 `delta` 分量因此完全由先验主导、不受训练似然约束——在使用合理先验的贝叶斯模型下这不会导致数值问题，但意味着这两个变点实际上不参与训练期间的趋势拟合，只在预测阶段外推增长率时才可能起作用。
        """,
        lines="L517-L523",
        notes="声明变点位置按完整序列而非仅训练区间等距排列，其中 2 个变点落在测试区间。",
    ),
    code(
        "ch06-gam-design-and-model",
        r"""
# 趋势设计矩阵
n_gam_changepoints = 12
n_gam_timepoints = seasonality_all.shape[0]
t_gam = np.linspace(0.0, 1.0, n_gam_timepoints, dtype=np.float32)
s_gam = np.linspace(0.0, 1.0, n_gam_changepoints + 2, dtype=np.float32)[1:-1]
A_gam = (t_gam[:, None] > s_gam).astype(np.float32)

# 季节性设计矩阵；n=6 产生 12 列，与月份独热编码列数一致。
month_index = co2_by_month.index.month.to_numpy() - 1
X_pred = gen_fourier_basis(month_index, p=12, n=6)
n_pred = X_pred.shape[-1]


def gam_components():
    beta = yield tfd.Sample(tfd.Normal(0.0, 1.0), n_pred, name="beta")
    seasonal_component = tf.einsum("ij,...j->...i", X_pred, beta)

    k = yield tfd.HalfNormal(10.0, name="k")
    m = yield tfd.Normal(float(co2_training["CO2"].mean()), 5.0, name="m")
    tau = yield tfd.HalfNormal(1.0, name="tau")
    delta = yield tfd.Sample(tfd.Laplace(0.0, tau), n_gam_changepoints, name="delta")

    growth_rate = k[..., None] + tf.einsum("ij,...j->...i", A_gam, delta)
    offset = m[..., None] + tf.einsum("ij,...j->...i", A_gam, -s_gam * delta)
    trend_component = growth_rate * t_gam + offset
    noise_sigma = yield tfd.HalfNormal(5.0, name="noise_sigma")
    return seasonal_component, trend_component, noise_sigma


def make_gam(training=True):
    @tfd.JointDistributionCoroutineAutoBatched
    def model():
        seasonal_component, trend_component, noise_sigma = yield from gam_components()
        y_hat = seasonal_component + trend_component
        if training:
            y_hat = y_hat[..., :len(co2_training)]
        yield tfd.Independent(
            tfd.Normal(y_hat, noise_sigma[..., None]),
            reinterpreted_batch_ndims=1,
            name="observed",
        )
    return model


gam_model = make_gam(training=True)
gam_forecast_model = make_gam(training=False)
assert gam_model.event_shape.observed[-1] == len(co2_training)
assert gam_forecast_model.event_shape.observed[-1] == len(co2_by_month)
        """,
        lines="L524-L566 and L720-L761",
        code_block="gam",
        notes="合并原 gam 与 gam_alternative 的模块化写法；AutoBatched 公共 API。",
    ),
    code(
        "ch06-gam-prior-predictive",
        r"""
gam_prior = gam_forecast_model.sample(
    BUDGET["prior_samples"], seed=split_seed("gam-prior")[0]
)
gam_prior_observed = np.asarray(gam_prior.observed)
assert gam_prior_observed.shape[-1] == len(co2_by_month)

fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
ax.plot(
    co2_by_month.index,
    gam_prior_observed.T,
    color="0.45",
    alpha=0.20,
    lw=0.8,
)
ax.plot(co2_training.index, co2_training["CO2"], color="0.05", lw=1.5, label="训练数据")
ax.set(xlabel="年份", ylabel="CO₂（ppm）")
ax.legend(frameon=False)
        """,
        lines="L568-L577 plus migration notebook prior plot",
        figures=("fig:fig8_prior_predictive2",),
    ),
    md(
        "ch06-gam-prior-caption-forecast-note",
        r"""
<a id="fig:fig8_prior_predictive2"></a>
**图 6.8**　类似 Facebook Prophet 的 GAM 先验预测。趋势参数使用弱信息先验，每条线是一条模拟时间序列。与图 6.3 相比，预测范围已经更接近实际观测。

完成推断后即可生成后验预测。其样本外表现比简单线性回归更好。需要注意，{cite:t}`TaylorLetham2018` 的预测生成过程并不与拟合时的生成模型完全相同：拟合时分段线性函数的变点等距且预先确定；预测时，建议先按“预设变点数 ÷ 总观测数”的概率判断一个新时间点是否成为变点，再从后验 $\delta_{new}\sim\text{Laplace}(0,\tau)$ 生成新的斜率变化。为简化生成过程，下面沿用最后一个区间的线性趋势。
        """,
        lines="L568-L592",
        anchors=("fig:fig8_prior_predictive2",),
    ),
    code(
        "ch06-gam-inference-and-forecast",
        r"""
# 中文版现代化说明：gam_model 不含 LinearGaussianStateSpaceModel，已验证可用
# jit_compile=True；相同 smoke 预算下比默认 jit_compile=False 快约 4 倍。
gam_draws, gam_stats = run_windowed_nuts(
    gam_model,
    seed=split_seed("gam-mcmc")[0],
    jit_compile=True,
    observed=tf.convert_to_tensor(co2_training["CO2"].to_numpy(), tf.float32),
)
gam_idata = tfp_draws_to_idata(gam_draws, gam_stats)

# 让全长度模型按训练后验参数正向生成预测。
gam_values = {
    name: value for name, value in gam_draws._asdict().items()
    if name != "observed"
}
gam_conditioned_model = gam_forecast_model.experimental_pin(**gam_values)
gam_conditioned_samples = gam_conditioned_model.sample_unpinned(
    seed=split_seed("gam-forecast")[0]
)
gam_pp = np.asarray(gam_conditioned_samples.observed)
assert gam_pp.shape[-1] == len(co2_by_month)

# 合并 draw 与 chain 轴后绘图。
gam_pp_flat = gam_pp.reshape((-1, gam_pp.shape[-1])).T
fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
plot_interval(ax, co2_by_month.index, gam_pp_flat, "GAM 后验预测")
ax.plot(co2_training.index, co2_training["CO2"], color="0.05", lw=1.3, label="训练集")
ax.plot(co2_testing.index, co2_testing["CO2"], color="0.45", lw=1.3, ls="--", label="测试集")
ax.axvline(co2_testing.index[0], color="0.25", ls=":", lw=1)
ax.set(xlabel="年份", ylabel="CO₂（ppm）")
ax.legend(frameon=False, ncol=3)
        """,
        lines="L579-L600 plus migration notebook GAM inference/forecast cells",
        figures=("fig:fig9_posterior_predictive2",),
        notes="后验预测通过公开 sample_distributions 条件生成；显式全长度预测模型。",
    ),
    md(
        "ch06-gam-forecast-caption-cleanup",
        r"""
<a id="fig:fig9_posterior_predictive2"></a>
**图 6.9**　类似 Facebook Prophet 的 GAM 后验预测。灰色带表示预测不确定性，黑色为训练数据，灰色虚线为测试数据。分段线性趋势比单一直线更能追随加速增长。

> **中文版补充**：分段线性趋势并不会自动保证长期外推正确。预测质量仍依赖变点生成机制、先验以及未来是否延续历史机制；因此必须保留测试集或使用留未来交叉验证来评估。
        """,
        lines="L594-L600",
        anchors=("fig:fig9_posterior_predictive2",),
        notes="补充强调外推假设与时间顺序验证。",
    ),
    code(
        "ch06-cleanup-regression-gam",
        r"""
release_resources(
    "regression_shapes",
    "prior_samples",
    "prior_predictive_timeseries",
    "regression_posterior",
    "intercept_draws",
    "trend_coeff_draws",
    "seasonality_coeff_draws",
    "noise_draws",
    "regression_pp",
    "gam_prior",
    "gam_prior_observed",
    "gam_pp_flat",
)
        """,
        lines="中文版现代化资源管理单元",
        code_block="cleanup_regression_gam",
        notes="重型章节之间明确关闭图形、释放对象并清理 TensorFlow 会话缓存。",
    ),
]

cells.extend([
    md(
        "ch06-autoregressive-models",
        r"""
(chap4_ar)=

## 自回归模型

到目前为止，我们一直把时间序列视为传统的线性回归问题，其中时间戳被解析为设计矩阵。这样可以捕捉总体趋势和季节性，但不能描述**当前观测能够预测下一观测**的动态。最简单的例子是**一阶自回归过程**，简称 AR(1)：

<a id="eq:ar1"></a>
$$
y_t\sim\mathcal{N}(\alpha+\rho y_{t-1},\sigma)
$$

下标 $t$ 表示当前时间点，$t-1$ 表示前一个时间点。每个观测 $y_t$ 都以 $\rho$ 倍的前一观测为均值，再加上截距 $\alpha$ 与标准差为 $\sigma$ 的随机扰动。直接用循环生成 AR(1) 最容易理解。
        """,
        lines="L602-L629",
        anchors=("chap4_ar", "eq:ar1"),
    ),
    code(
        "ch06-ar1-loop",
        r"""
def ar1_with_python_loop(rho, sigma, n, *, seed):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.float32)
    for t in range(1, n):
        y[t] = rho * y[t - 1] + rng.normal(0.0, sigma)
    return y

rho_values = [-1.1, -0.5, 0.0, 0.5, 1.1]
ar1_loop_samples = {
    rho: ar1_with_python_loop(rho, 1.0, 200, seed=BASE_SEED + i)
    for i, rho in enumerate(rho_values)
}
assert all(values.shape == (200,) for values in ar1_loop_samples.values())

fig, axes = plt.subplots(len(rho_values), 1, figsize=(10, 8), sharex=True, constrained_layout=True)
for ax, (rho, values) in zip(axes, ar1_loop_samples.items()):
    ax.plot(values, color="0.15", lw=1.2, label=fr"$\rho={rho}$")
    ax.legend(frameon=False, loc="upper left")
axes[-1].set_xlabel("时间")
fig.supylabel("AR(1) 过程")
        """,
        lines="L631-L663",
        code_block="ar1_with_forloop",
        figures=("fig:fig10_ar1_process",),
        notes="为每个 rho 使用独立且确定的 NumPy Generator。",
    ),
    md(
        "ch06-ar1-caption-stationarity",
        r"""
<a id="fig:fig10_ar1_process"></a>
**图 6.10**　$\sigma=1$、不同 $\rho$ 下 AR(1) 过程的随机样本。$|\rho|<1$ 时，序列围绕稳定水平波动；$\rho$ 接近 1 时呈现较长的持续性；$\rho$ 为负时相邻值倾向于交替；$|\rho|>1$ 时过程不平稳并会发散。

循环实现清楚地展示了生成过程，却不能充分利用现代计算硬件的向量化能力。在概率编程中，我们通常改写联合对数概率。定义**后移算子** $\mathbf B y_t=y_{t-1}$，则条件 AR(1) 似然可以写为

$$
Y_{1:T}\sim\mathcal{N}(\rho\mathbf B Y_{1:T},\sigma),
$$

对应代码是 `Normal(rho * y[:-1], sigma).log_prob(y[1:])`。这里把第一个观测视为给定初始条件。
        """,
        lines="L657-L691",
        anchors=("fig:fig10_ar1_process",),
    ),
    code(
        "ch06-ar1-public-trajectory",
        r"""
def ar1_with_scan(rho, sigma, n, *, seed):
    '''用公开 tf.scan 保留 AR(1) 的完整轨迹。'''
    innovation_seed = tfp.random.sanitize_seed(seed, salt="ar1-scan")
    innovations = tfd.Normal(0.0, sigma).sample(n, seed=innovation_seed)
    return tf.scan(
        lambda previous, innovation: rho * previous + innovation,
        innovations,
        initializer=tf.zeros([], dtype=innovations.dtype),
    )

ar1_scan_sample = ar1_with_scan(
    tf.constant(0.8, tf.float32),
    tf.constant(1.0, tf.float32),
    200,
    seed=split_seed("ar1-scan-example")[0],
)
ar1_mvn = ar1_trajectory_distribution(
    tf.constant(0.8, tf.float32), tf.constant(1.0, tf.float32), 200
)
ar1_mvn_sample = ar1_mvn.sample(seed=split_seed("ar1-mvn-example")[0])

assert ar1_scan_sample.shape == (200,)
assert ar1_mvn_sample.shape == (200,)
assert bool(tf.reduce_all(tf.math.is_finite(ar1_mvn.log_prob(ar1_mvn_sample))))
        """,
        lines="L665-L713",
        code_block="ar1_without_forloop",
        notes="以公开 tf.scan 和 MultivariateNormalTriL 替代私有 TFP 采样器；保留完整轨迹。",
    ),
    md(
        "ch06-ar1-regression-relationship",
        r"""
从回归角度看，自回归模型只是把响应变量自己的滞后值加入设计矩阵：第 $i$ 行预测变量取 $x_i=y_{i-1}$，对应回归系数为 $\rho$。名称“自回归”也正是“对自身做回归”的意思[^8]。

这种写法有两种重要用途。第一，可以把 AR 结构直接放在观测似然中，对回归均值之外的残差建模；第二，可以把 AR 过程作为一个不可观测的潜在成分，与趋势和季节性相加。两者的参数解释并不完全相同。
        """,
        lines="L715-L779",
    ),
    code(
        "ch06-gam-ar-likelihood",
        r"""
def make_gam_with_ar_likelihood(previous_observed):
    previous_observed = tf.convert_to_tensor(previous_observed, tf.float32)

    @tfd.JointDistributionCoroutineAutoBatched
    def model():
        seasonal_component, trend_component, noise_sigma = yield from gam_components()
        rho = yield tfd.Uniform(-1.0, 1.0, name="rho")
        baseline = seasonal_component[..., :len(co2_training)] + trend_component[..., :len(co2_training)]
        residual_previous = previous_observed - baseline[..., :-1]
        conditional_mean = baseline[..., 1:] + rho[..., None] * residual_previous
        yield tfd.Independent(
            tfd.Normal(conditional_mean, noise_sigma[..., None]),
            reinterpreted_batch_ndims=1,
            name="observed",
        )

    return model


gam_ar_likelihood_model = make_gam_with_ar_likelihood(
    co2_training["CO2"].to_numpy()[:-1]
)
assert gam_ar_likelihood_model.event_shape.observed[-1] == len(co2_training) - 1

gam_ar_draws, gam_ar_stats = run_windowed_nuts(
    gam_ar_likelihood_model,
    seed=split_seed("gam-ar-likelihood-mcmc")[0],
    observed=tf.convert_to_tensor(co2_training["CO2"].to_numpy()[1:], tf.float32),
)
gam_ar_idata = tfp_draws_to_idata(gam_ar_draws, gam_ar_stats)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
az.plot_posterior(gam_idata, var_names=["noise_sigma"], ax=axes[0])
axes[0].set_title("普通似然 $\\sigma$")
az.plot_posterior(gam_ar_idata, var_names=["noise_sigma"], ax=axes[1])
axes[1].set_title("AR 似然 $\\sigma$")
az.plot_posterior(gam_ar_idata, var_names=["rho"], ax=axes[2])
axes[2].set_title("AR 系数 $\\rho$")
        """,
        lines="L781-L833",
        code_block="gam_with_ar_likelihood",
        figures=("fig:fig11_ar1_likelihood_rho",),
        notes="条件 AR 残差似然只依赖公开分布；rho 用 Uniform(-1,1) 保证平稳区间。",
    ),
    md(
        "ch06-ar-likelihood-caption",
        r"""
<a id="fig:fig11_ar1_likelihood_rho"></a>
**图 6.11**　普通正态似然的 $\sigma$、AR 似然的 $\sigma$ 和 $\rho$ 后验。两个噪声尺度的估计相近，而 $\rho$ 的后验集中在 0 附近；这表明，在趋势与季节性设计已经解释主要结构后，剩余的一阶相关性很弱。

上面的模型让前一时刻的**观测残差**进入当前似然。另一种构造则引入完整的潜在 AR 轨迹，把它与 GAM 均值相加。此时观测噪声和 AR 创新尺度是两个不同参数。
        """,
        lines="L815-L838",
        anchors=("fig:fig11_ar1_likelihood_rho",),
    ),
    code(
        "ch06-gam-latent-ar",
        r"""
def make_gam_with_latent_ar(training=True):
    n_steps = len(co2_training) if training else len(co2_by_month)

    @tfd.JointDistributionCoroutineAutoBatched
    def model():
        seasonal_component, trend_component, noise_sigma = yield from gam_components()
        rho = yield tfd.Uniform(-0.99, 0.99, name="rho")
        ar_sigma = yield tfd.HalfNormal(2.0, name="ar_sigma")
        baseline = seasonal_component[..., :n_steps] + trend_component[..., :n_steps]
        lgssm = ar1_lgssm(rho, ar_sigma, noise_sigma, n_steps)
        yield tfd.TransformedDistribution(
            lgssm, tfb.Shift(baseline[..., None]), name="observed"
        )

    return model


gam_latent_ar_model = make_gam_with_latent_ar(training=True)
assert gam_latent_ar_model.event_shape.observed == [len(co2_training), 1]

latent_ar_draws, latent_ar_stats = run_windowed_nuts(
    gam_latent_ar_model,
    seed=split_seed("latent-ar-mcmc")[0],
    observed=tf.convert_to_tensor(co2_training["CO2"].to_numpy(), tf.float32)[:, None],
)
latent_ar_idata = tfp_draws_to_idata(latent_ar_draws, latent_ar_stats)
        """,
        lines="L840-L880",
        code_block="gam_with_latent_ar",
        notes=(
            "中文版现代化说明：原始联合采样整条稠密协方差潜在 AR 轨迹（"
            "num_timesteps≈517）在本地无原生编译器的执行环境下会导致单个梯度求值"
            "过慢、内核在实际运行中挂起。改用 ar1_lgssm 通过卡尔曼滤波解析边缘化"
            "潜在轨迹：NUTS 只对 rho、ar_sigma 等全局参数采样，观测分布用 "
            "TransformedDistribution(lgssm, tfb.Shift(baseline)) 把已知的 GAM "
            "趋势加季节性均值叠加到潜在 AR 分量上；潜在轨迹本身在下一单元中"
            "用 posterior_sample 事后恢复，而不是作为联合采样的一部分。"
        ),
    ),
    code(
        "ch06-latent-ar-plots",
        r"""
latent_posterior = latent_ar_idata.posterior

# 中文版现代化说明：latent_ar 已被解析边缘化，不再是采样变量；
# 用后验中的 GAM 与 AR 参数重建 baseline，再用 posterior_sample 恢复潜在轨迹。
n_steps_latent_ar = len(co2_training)
seasonal_posterior = tf.einsum(
    "ij,...j->...i", X_pred, tf.convert_to_tensor(latent_posterior["beta"].values)
)
growth_rate_posterior = tf.convert_to_tensor(
    latent_posterior["k"].values
)[..., None] + tf.einsum(
    "ij,...j->...i", A_gam, tf.convert_to_tensor(latent_posterior["delta"].values)
)
offset_posterior = tf.convert_to_tensor(
    latent_posterior["m"].values
)[..., None] + tf.einsum(
    "ij,...j->...i", A_gam, -s_gam * tf.convert_to_tensor(latent_posterior["delta"].values)
)
trend_posterior_full = growth_rate_posterior * t_gam + offset_posterior
baseline_posterior = (seasonal_posterior + trend_posterior_full)[..., :n_steps_latent_ar]

rho_posterior = tf.convert_to_tensor(latent_posterior["rho"].values)
ar_sigma_posterior = tf.convert_to_tensor(latent_posterior["ar_sigma"].values)
noise_sigma_posterior = tf.convert_to_tensor(latent_posterior["noise_sigma"].values)
residual_posterior = (
    tf.convert_to_tensor(co2_training["CO2"].to_numpy(), tf.float32) - baseline_posterior
)[..., None]

lgssm_posterior = ar1_lgssm(
    rho_posterior, ar_sigma_posterior, noise_sigma_posterior, n_steps_latent_ar
)
latent_ar_sample = lgssm_posterior.posterior_sample(
    residual_posterior, sample_shape=(), seed=split_seed("latent-ar-recover")[0]
)
latent_ar_array = np.asarray(latent_ar_sample)[..., 0]  # (chain, draw, n_steps)
latent_ar_component = np.moveaxis(
    latent_ar_array.reshape(-1, latent_ar_array.shape[-1]), -1, 0
)
assert latent_ar_component.shape[0] == n_steps_latent_ar

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
# 趋势与季节性可由相同后验参数和设计矩阵重建；这里突出潜在 AR 成分。
axes[0].plot(co2_training.index, co2_training["CO2"], color="0.10", lw=1.2)
axes[0].set_ylabel("观测 CO₂")
plot_interval(axes[1], co2_training.index, latent_ar_component, "潜在 AR(1)")
axes[1].set_ylabel("AR 成分")
axes[1].legend(frameon=False)
axes[2].plot(
    co2_training.index,
    co2_training["CO2"].to_numpy() - latent_ar_component.mean(axis=1),
    color="0.15",
    lw=1.2,
    label="移除 AR 成分后的序列",
)
axes[2].set(xlabel="年份", ylabel="校正值")
axes[2].legend(frameon=False)

fig2, axes2 = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
az.plot_posterior(latent_ar_idata, var_names=["noise_sigma", "ar_sigma", "rho"], ax=axes2)
        """,
        lines="L882-L917 plus migration notebook plot cells",
        figures=("fig:fig12_posterior_predictive_ar1", "fig:fig13_ar1_likelihood_rho2"),
        notes="中文版现代化说明：用 LGSSM 的 posterior_sample 事后恢复潜在 AR 轨迹，取代原先直接读取联合采样变量。",
    ),
    md(
        "ch06-latent-ar-captions",
        r"""
<a id="fig:fig12_posterior_predictive_ar1"></a>
**图 6.12**　带潜在 AR(1) 成分的 GAM 后验结果。各面板展示观测、潜在 AR 成分，以及移除该成分后的序列；实线为后验均值，灰色带为 94% 区间。

<a id="fig:fig13_ar1_likelihood_rho2"></a>
**图 6.13**　潜在 AR 模型中观测噪声 $\sigma_{noise}$、AR 创新尺度 $\sigma_{AR}$ 与 $\rho$ 的后验。请勿把这里的两个尺度与图 6.11 中条件 AR 似然的单一尺度混淆。

潜在 AR 模型通常更灵活，但也更难识别：平滑趋势、季节性、潜在 AR 轨迹和独立噪声都可能解释相似的变化。先验与模型检验因而至关重要。
        """,
        lines="L890-L917",
        anchors=("fig:fig12_posterior_predictive_ar1", "fig:fig13_ar1_likelihood_rho2"),
    ),
    md(
        "ch06-latent-ar-smoothing",
        r"""
(latent-ar-process-and-smoothing)=

### 潜在 AR 过程与平滑

当 $\rho=1$ 时，AR(1) 退化为**高斯随机游走**（Gaussian Random Walk，GRW）：

<a id="eq:gw_formulation1"></a>
$$
\begin{aligned}
z_i &\sim \mathcal{N}(z_{i-1},\sigma_z^2), && i=1,\ldots,N,\\
y_i &\sim \mathcal{N}(z_i,\sigma_y^2).
\end{aligned}
$$

其中 $z$ 是潜在时间序列，$y$ 是观测。两种噪声的相对大小控制平滑程度。定义

$$
\alpha=\frac{\sigma_y^2}{\sigma_z^2+\sigma_y^2},\qquad \alpha\in[0,1],
$$

并令总方差为 $\sigma^2=\sigma_z^2+\sigma_y^2$，可重新写为：

<a id="eq:gw_formulation2"></a>
$$
\begin{aligned}
z_i &\sim \mathcal{N}\!\left(z_{i-1},(1-\alpha)\sigma^2\right), && i=1,\ldots,N,\\
y_i &\sim \mathcal{N}\!\left(z_i,\alpha\sigma^2\right).
\end{aligned}
$$

$\alpha$ 越接近 1，观测噪声相对越大，潜在序列越平滑；$\alpha$ 越接近 0，潜在序列越贴近每个观测。因此，“平滑”可以理解为在状态创新与观测噪声之间分配变化。
        """,
        lines="L919-L967",
        anchors=("latent-ar-process-and-smoothing", "eq:gw_formulation1", "eq:gw_formulation2"),
    ),
    code(
        "ch06-gaussian-random-walk-model",
        r"""
x_smooth = np.linspace(0.0, 30.0, 150, dtype=np.float32)
f_smooth = np.exp(1.0 + np.sqrt(x_smooth) - np.exp(x_smooth / 15.0)).astype(np.float32)
rng = cell_numpy_rng("ch06-gaussian-random-walk-model")
y_smooth = f_smooth + rng.normal(0.0, 1.0, size=len(x_smooth)).astype(np.float32)

@tfd.JointDistributionCoroutineAutoBatched
def gaussian_random_walk_model():
    total_scale = yield tfd.HalfNormal(2.0, name="total_scale")
    alpha = yield tfd.Beta(2.0, 2.0, name="alpha")
    state_scale = total_scale * tf.sqrt(1.0 - alpha)
    observation_scale = total_scale * tf.sqrt(alpha)
    latent = yield ar1_trajectory_distribution(
        tf.constant(1.0, tf.float32), state_scale, len(y_smooth), name="latent"
    )
    yield tfd.Independent(
        tfd.Normal(latent, observation_scale[..., None]),
        reinterpreted_batch_ndims=1,
        name="observed",
    )

random_walk_draws, random_walk_stats = run_windowed_nuts(
    gaussian_random_walk_model,
    seed=split_seed("random-walk-mcmc")[0],
    observed=tf.convert_to_tensor(y_smooth),
)
random_walk_idata = tfp_draws_to_idata(random_walk_draws, random_walk_stats)

latent_smooth = np.asarray(
    random_walk_idata.posterior["latent"].stack(sample=("chain", "draw")).values
)
if latent_smooth.shape[0] != len(x_smooth):
    latent_smooth = np.moveaxis(latent_smooth, -1, 0)
assert latent_smooth.shape[0] == len(x_smooth)

fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
ax.scatter(x_smooth, y_smooth, s=12, facecolors="none", edgecolors="0.45", label="观测")
plot_interval(ax, x_smooth, latent_smooth, "潜在随机游走")
ax.plot(x_smooth, f_smooth, color="0.10", ls="--", lw=1.5, label="真实函数")
ax.set(xlabel="$x$", ylabel="$y$")
ax.legend(frameon=False, ncol=3)
        """,
        lines="L968-L991",
        code_block="gw_tfp",
        figures=("fig:fig14_smoothing_with_gw",),
        notes="公开 MVN 轨迹替代私有循环工具；alpha 显式控制状态/观测噪声比。",
    ),
    md(
        "ch06-random-walk-caption-gp",
        r"""
<a id="fig:fig14_smoothing_with_gw"></a>
**图 6.14**　从 $y\sim\operatorname{Normal}(f(x),1)$ 模拟的数据，其中 $f(x)=\exp(1+\sqrt{x}-\exp(x/15))$。圆点为观测，虚线为真实函数，实线与灰色带分别为潜在 GRW 的后验均值和 94% 区间。

随机游走可以近似复杂的非线性函数，但它并不知道横轴上“距离”的几何信息：这里只利用了观测顺序。若观测间隔不规则，或希望相关性随实际时间距离连续变化，高斯过程通常更合适。事实上，特定 AR 过程本身就是高斯过程[^9]，二者可用协方差函数联系起来 {cite:p}`Rasmussen2005`。

这种灵活性同时造成缩放与可识别性问题。状态创新尺度、观测噪声以及其他平滑成分可能互相补偿。建模者应通过先验预测、后验预测和灵敏度分析确认尺度具有合理解释。
        """,
        lines="L983-L1008",
        anchors=("fig:fig14_smoothing_with_gw",),
    ),
    code(
        "ch06-cleanup-ar-smoothing",
        r"""
release_resources(
    "ar1_loop_samples",
    "ar1_scan_sample",
    "ar1_mvn_sample",
    "gam_ar_draws",
    "gam_ar_idata",
    "latent_ar_component",
    "latent_ar_draws",
    "latent_ar_idata",
    "latent_posterior",
    "seasonal_posterior",
    "growth_rate_posterior",
    "offset_posterior",
    "trend_posterior_full",
    "baseline_posterior",
    "rho_posterior",
    "ar_sigma_posterior",
    "noise_sigma_posterior",
    "residual_posterior",
    "lgssm_posterior",
    "latent_ar_sample",
    "latent_ar_array",
    "random_walk_draws",
    "random_walk_idata",
    "latent_smooth",
)
        """,
        lines="中文版现代化资源管理单元",
        code_block="cleanup_ar_smoothing",
        notes="在 SARIMA 与状态空间章节前释放后验轨迹和绘图对象。",
    ),
    md(
        "ch06-sarimax-concepts",
        r"""
(sarimax)=

### (S)AR(I)MA(X)

金融时间序列常表现出**波动率聚集**：大幅变化之后往往仍是大幅变化，小幅变化之后往往仍是小幅变化。ARCH/GARCH 一类模型让当前方差依赖过去的平方残差。另一类模型则让当前值依赖过去的随机冲击，即**移动平均**（Moving Average，MA）成分。

把 $p$ 阶 AR 与 $q$ 阶 MA 组合，就得到 ARMA$(p,q)$：

<a id="eq:arma"></a>
$$
\begin{aligned}
y_t &= \alpha+\sum_{i=1}^{p}\phi_i y_{t-i}+\sum_{j=1}^{q}\theta_j\epsilon_{t-j}+\epsilon_t,\\
\epsilon_t&\sim\mathcal{N}(0,\sigma^2).
\end{aligned}
$$

若序列有周期为 `period` 的季节结构，也可以在季节滞后上使用 AR 与 MA 项：

<a id="eq:sarma"></a>
$$
y_t=\alpha+\sum_{i=1}^{p}\phi_i y_{t-\text{period}\,i}
+\sum_{j=1}^{q}\theta_j\epsilon_{t-\text{period}\,j}+\epsilon_t.
$$

真实序列常有趋势而不平稳。差分算子 $I(d)$ 连续做 $d$ 次 `delta_y[i] = y[i] - y[i-1]`，使序列更接近平稳；这得到 ARIMA$(p,d,q)$ {cite:t}`box2008time`。还可以把截距 $\alpha$ 换成外生设计矩阵 $\mathbf X\beta$，形成 ARIMAX。季节差分、季节 ARMA 与外生预测变量也可以按建模需要共同出现。
        """,
        lines="L1010-L1070",
        anchors=("sarimax", "eq:arma", "eq:sarma"),
    ),
    md(
        "ch06-sarimax-notation-admonition",
        r"""
> **(S)AR(I)MA(X) 的记号**
>
> - **ARIMA$(p,d,q)$**：非季节 AR 阶数 $p$、差分阶数 $d$、MA 阶数 $q$。
> - **SARIMA$(p,d,q)(P,D,Q)_s$**：再增加季节 AR 阶数 $P$、季节差分阶数 $D$、季节 MA 阶数 $Q$ 与季节周期 $s$。也常写成 SARIMA$(p,d,q)(P,D,Q,s)$。
> - **ARIMAX$(p,d,q)\mathbf X[k]$**：再加入有 $k$ 列的外生设计矩阵 $\mathbf X$。
>
> 文献与软件包的符号约定可能不同；使用实现时必须确认系数符号、差分顺序、截距以及初始状态的定义。
        """,
        lines="L1072-L1084",
    ),
    md(
        "ch06-birth-data-introduction",
        r"""
下面用美国月度活产数说明 SARIMA。数据覆盖 1948 年 1 月至 1979 年 1 月，单位为千人 {cite:p}`shumway2019time`。原书代码注释写成 372 个观测，但随附 CSV 实际包含 373 个；中文版按文件内容读取，并保留这一来源差异。
        """,
        lines="L1086-L1100",
        notes="明确记录源数据 373 行与原注释 372 的不一致。",
    ),
    code(
        "ch06-birth-data-preprocess",
        r"""
us_monthly_birth = pd.read_csv("data/monthly_birth_usa.csv")
us_monthly_birth["date_month"] = pd.to_datetime(us_monthly_birth["date_month"])
us_monthly_birth.set_index("date_month", inplace=True)
births = us_monthly_birth["birth_in_thousands"].to_numpy(dtype=np.float32)

assert births.shape == (373,)
assert np.isfinite(births).all()
assert us_monthly_birth.index.is_monotonic_increasing

fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
ax.plot(us_monthly_birth.index, births, color="0.10", lw=1.3)
ax.set(xlabel="年份", ylabel="活产数（千人）")
        """,
        lines="L1091-L1120",
        code_block="sarima_preprocess",
        figures=("fig:fig15_birth_by_month",),
        notes="按实际 CSV 断言 373 个观测；不沿用原书过时的 372 注释。",
    ),
    md(
        "ch06-birth-caption-sarima-likelihood",
        r"""
<a id="fig:fig15_birth_by_month"></a>
**图 6.15**　美国月度活产数（1948–1979）。纵轴单位为千人。序列同时呈现趋势、年度季节性和随时间变化的局部结构。

原书实现通过自定义递推计算 SARIMA 似然。TFP 0.25 没有一个与该教学公式完全对应的专用公共 `SARIMA` 分布；因此，下面用公开 TensorFlow 的 `tf.while_loop` 和 `TensorArray` 给出自包含实现，而不是导入 TFP 私有循环或种子工具。这个选择保留原书“理解递推似然”的教学目的[^10]。把可调用的对数后验传给 TFP MCMC 的一般方式参见 {cite:p}`lao2020tfpmcmc`。

> **中文版现代化说明**：下面 `sarima_residuals` 把常规 AR/MA 项与季节 AR/MA 项分别相加，是真正乘法 SARIMA 的一种加法近似——完整的 $\operatorname{SARIMA}(p,d,q)(P,D,Q)_s$ 需要把常规与季节自回归/移动平均多项式相乘，因此还应包含常规项与季节项的交叉滞后（例如 $\operatorname{SARIMA}(1,\cdot,\cdot)(1,\cdot,\cdot)_{12}$ 的 AR 多项式 $(1-\phi B)(1-\Phi B^{12})$ 展开后还有一项 $\phi\Phi B^{13}$，本实现未包含这一交叉项）。这一简化足以支撑本节关于自包含似然递推与 LOO 比较的教学目的，但产出的系数不能等同于标准乘法 SARIMA 的估计值。
        """,
        lines="L1091-L1137",
        anchors=("fig:fig15_birth_by_month",),
        notes="说明 TFP 0.25 无专用公共 SARIMA 分布及自包含等价实现；并声明当前递推是加法近似而非完整乘法 SARIMA。",
    ),
    code(
        "ch06-sarima-likelihood",
        r"""
def difference_series(series, d=0, D=0, period=12):
    result = tf.convert_to_tensor(series, tf.float32)
    for _ in range(d):
        result = result[1:] - result[:-1]
    for _ in range(D):
        result = result[period:] - result[:-period]
    return result


def sarima_residuals(series, ar, ma, seasonal_ar, seasonal_ma, period=12):
    '''用公开 TensorFlow 运算计算条件 SARIMA 创新。'''
    y = tf.convert_to_tensor(series, tf.float32)
    ar = tf.convert_to_tensor(ar, tf.float32)
    ma = tf.convert_to_tensor(ma, tf.float32)
    seasonal_ar = tf.convert_to_tensor(seasonal_ar, tf.float32)
    seasonal_ma = tf.convert_to_tensor(seasonal_ma, tf.float32)
    n = tf.shape(y)[0]
    batch_shape = tf.shape(ar)[:-1]
    residuals = tf.TensorArray(
        y.dtype,
        size=n,
        clear_after_read=False,
        element_shape=ar.shape[:-1],
    )

    def previous(array, index):
        return tf.cond(
            index >= 0,
            lambda: array.read(index),
            lambda: tf.zeros(batch_shape, y.dtype),
        )

    def body(t, eps):
        mean = tf.zeros(batch_shape, y.dtype)
        for lag, coefficient in enumerate(tf.unstack(ar, axis=-1), start=1):
            mean += coefficient * tf.cond(t >= lag, lambda lag=lag: y[t - lag], lambda: 0.0)
        for lag, coefficient in enumerate(tf.unstack(ma, axis=-1), start=1):
            mean += coefficient * previous(eps, t - lag)
        for order, coefficient in enumerate(tf.unstack(seasonal_ar, axis=-1), start=1):
            lag = period * order
            mean += coefficient * tf.cond(t >= lag, lambda lag=lag: y[t - lag], lambda: 0.0)
        for order, coefficient in enumerate(tf.unstack(seasonal_ma, axis=-1), start=1):
            lag = period * order
            mean += coefficient * previous(eps, t - lag)
        return t + 1, eps.write(t, y[t] - mean)

    _, residuals = tf.while_loop(
        lambda t, _: t < n,
        body,
        (tf.constant(0), residuals),
        parallel_iterations=1,
    )
    return residuals.stack()


def sarima_log_likelihood(
    series, ar, ma, seasonal_ar, seasonal_ma, sigma, *, d=1, D=1, period=12
):
    differenced = difference_series(series, d=d, D=D, period=period)
    residuals = sarima_residuals(differenced, ar, ma, seasonal_ar, seasonal_ma, period)
    orders = [ar.shape[-1], ma.shape[-1], seasonal_ar.shape[-1], seasonal_ma.shape[-1]]
    if any(order is None for order in orders):
        raise ValueError("SARIMA 阶数必须在图追踪时静态已知")
    warmup = max(orders[0], orders[1], period * orders[2], period * orders[3])
    return tf.reduce_sum(
        tfd.Normal(0.0, sigma).log_prob(residuals[warmup:]), axis=0
    )

# 一个有限值检查，也验证差分与递推长度。
sarima_demo_ll = sarima_log_likelihood(
    births,
    ar=tf.constant([0.2]),
    ma=tf.constant([0.1]),
    seasonal_ar=tf.constant([0.2]),
    seasonal_ma=tf.constant([0.1]),
    sigma=tf.constant(5.0),
)
assert bool(tf.math.is_finite(sarima_demo_ll))
        """,
        lines="L1138-L1181",
        code_block="sarima_likelihood",
        notes="自包含 tf.while_loop/TensorArray；无 tensorflow_probability.python.internal 导入。",
    ),
    code(
        "ch06-sarima-posterior-and-mcmc",
        r"""
# 与正文 SARIMA(1,1,1)(1,1,1)_12 对应的先验。
sarima_prior = tfd.JointDistributionNamedAutoBatched({
    "ar": tfd.Sample(tfd.Normal(0.0, 0.5), 1),
    "ma": tfd.Sample(tfd.Normal(0.0, 0.5), 1),
    "seasonal_ar": tfd.Sample(tfd.Normal(0.0, 0.5), 1),
    "seasonal_ma": tfd.Sample(tfd.Normal(0.0, 0.5), 1),
    "sigma": tfd.HalfNormal(10.0),
})


def sarima_target_log_prob(ar, ma, seasonal_ar, seasonal_ma, sigma):
    state = {
        "ar": ar,
        "ma": ma,
        "seasonal_ar": seasonal_ar,
        "seasonal_ma": seasonal_ma,
        "sigma": sigma,
    }
    return sarima_prior.log_prob(state) + sarima_log_likelihood(
        births, ar, ma, seasonal_ar, seasonal_ma, sigma, d=1, D=1, period=12
    )


def run_sarima_chain():
    initial_state = [
        tf.zeros([BUDGET["chains"], 1], tf.float32),
        tf.zeros([BUDGET["chains"], 1], tf.float32),
        tf.zeros([BUDGET["chains"], 1], tf.float32),
        tf.zeros([BUDGET["chains"], 1], tf.float32),
        tf.ones([BUDGET["chains"]], tf.float32),
    ]
    nuts = tfp.mcmc.NoUTurnSampler(sarima_target_log_prob, step_size=0.05)
    transformed = tfp.mcmc.TransformedTransitionKernel(
        nuts,
        bijector=[tfb.Identity(), tfb.Identity(), tfb.Identity(), tfb.Identity(), tfb.Softplus()],
    )
    adaptive = tfp.mcmc.DualAveragingStepSizeAdaptation(
        transformed,
        num_adaptation_steps=BUDGET["adapt"],
        target_accept_prob=0.8,
    )
    return tfp.mcmc.sample_chain(
        num_results=BUDGET["draws"],
        num_burnin_steps=BUDGET["adapt"],
        current_state=initial_state,
        kernel=adaptive,
        trace_fn=lambda _, pkr: pkr,
        seed=split_seed("sarima-mcmc")[0],
    )

# 完整执行 Notebook 时，这一调用会得到原书正文省略的 MCMC 结果；
# smoke 与 release 都执行同一路径，只改变链数、抽样数和适应预算。
sarima_chain, sarima_kernel_results = run_sarima_chain()
        """,
        lines="L1183-L1217 plus accompanying migration notebook sampler",
        code_block="sarima_posterior",
        notes="以公开 NoUTurnSampler、TransformedTransitionKernel 与 DualAveraging 实现完整推断。",
    ),
    md(
        "ch06-sarima-comparison",
        r"""
原书正文为简洁而省略了 MCMC 细节，并把读者引向随附 Notebook[^11]；上面的规范源已把公共 API 版本保留在教学顺序中。得到后验逐点对数似然后，可以用 ArviZ 比较两个候选模型。

<a id="tab:loo_sarima"></a>
**表 6.1　不同 SARIMA 模型的 LOO（对数尺度）比较汇总**

| 模型 | 排名 | loo | p_loo | d_loo | 权重 | se | dse |
|---|---:|---:|---:|---:|---:|---:|---:|
| $\operatorname{SARIMA}(0,1,2)(1,1,1)_{12}$ | 0 | -1235.60 | 7.51 | 0.00 | 0.5 | 15.41 | 0.00 |
| $\operatorname{SARIMA}(1,1,1)(1,1,1)_{12}$ | 1 | -1235.97 | 8.30 | 0.37 | 0.5 | 15.47 | 6.29 |

两者的差异远小于其不确定性，因此没有证据明确偏好其中一个。LOO 的一般使用方法见第 [2](chap1) 章；不过时间序列数据不满足普通逐点 LOO 所需的可交换性，后文还会讨论留未来交叉验证。
        """,
        lines="L1183-L1245",
        anchors=("tab:loo_sarima",),
    ),
    code(
        "ch06-cleanup-sarima",
        r"""
release_resources(
    "sarima_chain",
    "sarima_kernel_results",
    "sarima_demo_ll",
)
        """,
        lines="中文版现代化资源管理单元",
        code_block="cleanup_sarima",
        notes="进入 LGSSM 前释放 SARIMA 链与内核轨迹。",
    ),
])

cells.extend([
    md(
        "ch06-state-space-introduction",
        r"""
(state-space-models)=

## 状态空间模型

前面的回归与自回归模型都可以放进更一般的**状态空间模型**框架。状态空间模型区分两类量：不可直接观测的潜在状态 $X_t$，以及由状态生成的观测 $Y_t$[^12]。其生成过程是：

<a id="eq:state_space_model"></a>
$$
\begin{aligned}
X_0 &\sim p(X_0),\\
&\text{对 } t=0,\ldots,T:\\
Y_t &\sim p^\psi(Y_t\mid X_t),\\
X_{t+1} &\sim p^\theta(X_{t+1}\mid X_t).
\end{aligned}
$$

$p^\theta$ 称为**状态转移模型**，描述系统如何从当前状态演化到下一状态；$p^\psi$ 称为**观测模型**，描述状态如何产生测量值。参数 $\theta$ 与 $\psi$ 可以固定，也可以具有先验并从数据中推断。

状态空间表示的优势在于模块化：趋势、季节性、自回归和外生回归都可以成为潜在状态的不同部分。它也明确区分“过程自身的不确定性”和“测量过程的不确定性”。
        """,
        lines="L1248-L1282",
        anchors=("state-space-models", "eq:state_space_model"),
    ),
    md(
        "ch06-state-space-implementation-admonition",
        r"""
> **高效实现状态空间模型**
>
> 状态空间模型的生成与似然计算通常需要沿时间维递推。在 TensorFlow 中，应优先使用 `tf.while_loop` 或 `tf.scan`，而不是在图追踪期间展开很长的 Python 循环。循环体应写成纯函数：输入前一状态与当前数据，返回下一状态和需要保存的结果。
>
> 还要明确哪个轴是时间轴、哪个轴是批轴。例如，输入可能是 `[N, T, ...]`，其中 `N` 是样本或链、`T` 是时间；而 `tf.scan` 默认把扫描轴放在输出最前，得到 `[T, N, ...]`。在进入和离开递推时应显式转置，并用形状断言检查。
>
> 初始转移与初始观测放在循环内还是循环外，会改变索引语义。实现前应先写出 $X_0$、$Y_0$、$X_1$ 的生成顺序，再对应到代码。
        """,
        lines="L1284-L1316",
    ),
    md(
        "ch06-filter-predict-smooth",
        r"""
给定状态空间模型后，常见的三个推断问题是：

- **滤波（filtering）**：在时间 $k$，只使用截至当前的观测，计算 $p(X_k\mid y_{0:k})$。
- **预测（prediction）**：使用截至 $k$ 的观测，计算未来 $n$ 步状态 $p(X_{k+n}\mid y_{0:k})$。
- **平滑（smoothing）**：在已经看完整段数据后，回头计算 $p(X_k\mid y_{0:T})$，其中 $k<T$。

滤波适合在线更新；预测面向未来决策；平滑则用未来观测改善对历史状态的估计。三者不能混为一谈。
        """,
        lines="L1318-L1342",
    ),
    md(
        "ch06-lgssm-kalman",
        r"""
(lgssm_time_series)=

### 线性高斯状态空间模型与卡尔曼滤波

若状态转移和观测都是线性的，所有噪声都是高斯的，就得到**线性高斯状态空间模型**（Linear Gaussian State Space Model，LGSSM）：

<a id="eq:lgssm"></a>
$$
\begin{aligned}
Y_t &= \mathbf H_tX_t+\epsilon_t,\\
X_t &= \mathbf F_tX_{t-1}+\eta_t,
\end{aligned}
$$

其中 $\epsilon_t\sim\mathcal N(0,\mathbf R_t)$，$\eta_t\sim\mathcal N(0,\mathbf Q_t)$。$\mathbf F_t$ 是状态转移矩阵，$\mathbf H_t$ 是观测矩阵，$\mathbf Q_t$ 与 $\mathbf R_t$ 分别是状态噪声和观测噪声协方差。

线性变换保持高斯性，高斯分布又具有共轭性，因此所有滤波与平滑分布都有闭式高斯形式。这正是卡尔曼滤波（Kalman, 1960）的基础。把生成式写成条件分布：

<a id="eq:lgssm_generative"></a>
$$
\begin{aligned}
X_t\sim p(X_t\mid X_{t-1})&\equiv\mathcal N(\mathbf F_tX_{t-1},\mathbf Q_t),\\
Y_t\sim p(Y_t\mid X_t)&\equiv\mathcal N(\mathbf H_tX_t,\mathbf R_t).
\end{aligned}
$$
        """,
        lines="L1344-L1396",
        anchors=("lgssm_time_series", "eq:lgssm", "eq:lgssm_generative"),
    ),
    md(
        "ch06-kalman-distributions",
        r"""
为递推滤波定义以下分布：

<a id="eq:kalman_fitler"></a>
$$
\begin{aligned}
X_0\sim p(X_0\mid m_0,\mathbf P_0)&\equiv\mathcal N(m_0,\mathbf P_0),\\
X_{t\mid t-1}\sim p(X_{t\mid t-1}\mid Y_{0:t-1})&\equiv\mathcal N(m_{t\mid t-1},\mathbf P_{t\mid t-1}),\\
X_{t\mid t}\sim p(X_{t\mid t}\mid Y_{0:t})&\equiv\mathcal N(m_{t\mid t},\mathbf P_{t\mid t}),\\
Y_t\sim p(Y_t\mid Y_{0:t-1})&\equiv\mathcal N(\mathbf H_tm_{t\mid t-1},\mathbf S_t).
\end{aligned}
$$

每一步先**预测**状态：

<a id="eq:kalman_fitler_preddict_step"></a>
$$
\begin{aligned}
m_{t\mid t-1}&=\mathbf F_tm_{t-1\mid t-1},\\
\mathbf P_{t\mid t-1}&=\mathbf F_t\mathbf P_{t-1\mid t-1}\mathbf F_t^T+\mathbf Q_t.
\end{aligned}
$$

再用新观测**更新**：

<a id="eq:kalman_fitler_update_step"></a>
$$
\begin{aligned}
z_t&=Y_t-\mathbf H_tm_{t\mid t-1},\\
\mathbf S_t&=\mathbf H_t\mathbf P_{t\mid t-1}\mathbf H_t^T+\mathbf R_t,\\
\mathbf K_t&=\mathbf P_{t\mid t-1}\mathbf H_t^T\mathbf S_t^{-1},\\
m_{t\mid t}&=m_{t\mid t-1}+\mathbf K_tz_t,\\
\mathbf P_{t\mid t}&=\mathbf P_{t\mid t-1}-\mathbf K_t\mathbf S_t\mathbf K_t^T.
\end{aligned}
$$

$z_t$ 是创新（预测误差），$\mathbf S_t$ 是其协方差，$\mathbf K_t$ 是卡尔曼增益。实际实现应避免显式求逆，并用 Cholesky 分解或线性求解提高数值稳定性 {cite:p}`westharrison1997`。

> **中文版现代化说明**：原书的 `fitler` 与 `preddict` 拼写错误已经成为跨引用锚点，故上面保留其机器标识，但中文正文使用正确术语。
        """,
        lines="L1398-L1465",
        anchors=(
            "eq:kalman_fitler",
            "eq:kalman_fitler_preddict_step",
            "eq:kalman_fitler_update_step",
        ),
        notes="保留原书错误拼写的公共锚点。",
    ),
    md(
        "ch06-linear-growth-model",
        r"""
考虑最简单的线性增长模型 {cite:p}`sarkka2013bayesian`。潜在状态包含截距和斜率：

<a id="eq:linear_growth_state"></a>
$$
X_t=\begin{bmatrix}\theta_0\\\theta_1\end{bmatrix}.
$$

如果它们随时间保持不变，状态转移矩阵就是单位矩阵。观测矩阵则随时间变化[^13]：

<a id="eq:linear_growth_observed_state"></a>
$$
y_t=\theta_0+\theta_1t
=\begin{bmatrix}1&t\end{bmatrix}
\begin{bmatrix}\theta_0\\\theta_1\end{bmatrix}.
$$
        """,
        lines="L1467-L1520",
        anchors=("eq:linear_growth_state", "eq:linear_growth_observed_state"),
    ),
    code(
        "ch06-linear-growth-simulate",
        r"""
num_linear_steps = 100
linear_time = np.arange(num_linear_steps, dtype=np.float32)
true_intercept, true_slope = 0.5, 0.08
rng = cell_numpy_rng("ch06-linear-growth-simulate")
linear_observed = (
    true_intercept
    + true_slope * linear_time
    + rng.normal(0.0, 0.5, num_linear_steps)
).astype(np.float32)
assert linear_observed.shape == (num_linear_steps,)
        """,
        lines="L1473-L1484",
        code_block="linear_growth_model",
    ),
    code(
        "ch06-linear-growth-lgssm",
        r"""
def linear_observation_matrix(t):
    t = tf.cast(t, tf.float32)
    return tf.linalg.LinearOperatorFullMatrix(tf.reshape(tf.stack([1.0, t]), [1, 2]))

linear_growth_lgssm = tfd.LinearGaussianStateSpaceModel(
    num_timesteps=num_linear_steps,
    transition_matrix=tf.linalg.LinearOperatorIdentity(2, dtype=tf.float32),
    transition_noise=tfd.MultivariateNormalDiag(
        loc=tf.zeros(2), scale_diag=tf.fill([2], tf.constant(1e-4, tf.float32))
    ),
    observation_matrix=linear_observation_matrix,
    observation_noise=tfd.MultivariateNormalDiag(scale_diag=[0.5]),
    initial_state_prior=tfd.MultivariateNormalDiag(
        loc=[0.0, 0.0], scale_diag=[5.0, 1.0]
    ),
)

linear_observed_column = linear_observed[:, None]
linear_log_prob = linear_growth_lgssm.log_prob(linear_observed_column)
assert bool(tf.math.is_finite(linear_log_prob))
        """,
        lines="L1522-L1549",
        code_block="tfd_lgssm_linear_growth",
        notes="以 1e-4 状态噪声近似概念上的确定性转移，避免奇异协方差。",
    ),
    code(
        "ch06-linear-growth-filter",
        r"""
(
    linear_log_likelihoods,
    filtered_means,
    filtered_covs,
    predicted_means,
    predicted_covs,
    observation_means,
    observation_covs,
) = linear_growth_lgssm.forward_filter(linear_observed_column)

assert filtered_means.shape == (num_linear_steps, 2)
assert predicted_means.shape == (num_linear_steps, 2)
assert np.isfinite(np.asarray(filtered_means)).all()

filtered_sd = np.sqrt(np.diagonal(np.asarray(filtered_covs), axis1=-2, axis2=-1))
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
axes[0].scatter(linear_time, linear_observed, s=12, facecolors="none", edgecolors="0.45", label="观测")
axes[0].plot(linear_time, np.asarray(observation_means).squeeze(-1), color="0.10", lw=1.5, label="一步预测")
axes[0].set(xlabel="时间", ylabel="$y_t$")
axes[0].legend(frameon=False)
for i, label in enumerate(["截距", "斜率"]):
    axes[i + 1].plot(linear_time, np.asarray(filtered_means)[:, i], color="0.10", lw=1.5, label="滤波均值")
    axes[i + 1].fill_between(
        linear_time,
        np.asarray(filtered_means)[:, i] - 1.88 * filtered_sd[:, i],
        np.asarray(filtered_means)[:, i] + 1.88 * filtered_sd[:, i],
        color="0.55",
        alpha=0.25,
    )
    axes[i + 1].axhline([true_intercept, true_slope][i], color="0.35", ls="--", label="真实值")
    axes[i + 1].set(xlabel="时间", ylabel=label)
    axes[i + 1].legend(frameon=False)
        """,
        lines="L1554-L1581",
        code_block="tfd_lgssm_linear_growth_filter",
        figures=("fig:fig16_linear_growth_lgssm",),
    ),
    md(
        "ch06-linear-growth-caption",
        r"""
<a id="fig:fig16_linear_growth_lgssm"></a>
**图 6.16**　线性增长 LGSSM 的卡尔曼滤波结果。左图显示观测与一步预测 $\mathbf H_tm_{t\mid t-1}$；中图和右图显示截距、斜率的逐步滤波后验及 94% 区间，虚线为真实值。随着观测累积，状态不确定性逐渐收缩。
        """,
        lines="L1571-L1581",
        anchors=("fig:fig16_linear_growth_lgssm",),
    ),
    md(
        "ch06-arima-as-state-space",
        r"""
(arima-expressed-as-a-state-space-model)=

### 用状态空间模型表示 ARIMA

ARMA 模型也可以写成 LGSSM，而且表示并不唯一[^14]。先考虑

<a id="eq:arma_pre_lgssm"></a>
$$
y_t=\sum_{i=1}^{r}\phi_i y_{t-i}+\sum_{i=1}^{r-1}\theta_i\epsilon_{t-i}+\epsilon_t,
$$

其中 $r=\max(p,q+1)$，不足的系数用 0 填充。令

<a id="eq:arma_lgssm_state_fn"></a>
$$
\mathbf F=
\begin{bmatrix}
\phi_1&1&\cdots&0\\
\vdots&\vdots&\ddots&\vdots\\
\phi_{r-1}&0&\cdots&1\\
\phi_r&0&\cdots&0
\end{bmatrix},\qquad
\mathbf A=\begin{bmatrix}1\\\theta_1\\\vdots\\\theta_{r-1}\end{bmatrix},
$$

其中 $\eta'_{t+1}\sim\mathcal N(0,\sigma^2)$、$\eta_t=\mathbf A\eta'_{t+1}$。状态向量可写为

<a id="eq:arma_lgssm_state"></a>
$$
X_t=\begin{bmatrix}
y_t\\
\phi_2y_{t-1}+\cdots+\phi_ry_{t-r+1}+\theta_1\eta'_t+\cdots+\theta_{r-1}\eta'_{t-r+2}\\
\phi_3y_{t-1}+\cdots+\phi_ry_{t-r+2}+\theta_2\eta'_t+\cdots+\theta_{r-1}\eta'_{t-r+3}\\
\vdots\\
\phi_ry_{t-1}+\theta_{r-1}\eta'_t
\end{bmatrix}.
$$

观测矩阵为 $\mathbf H_t=[1,0,\ldots,0]$，故 $y_t=\mathbf H_tX_t$。
        """,
        lines="L1583-L1647",
        anchors=(
            "arima-expressed-as-a-state-space-model",
            "eq:arma_pre_lgssm",
            "eq:arma_lgssm_state_fn",
            "eq:arma_lgssm_state",
        ),
    ),
    md(
        "ch06-arma21-equation",
        r"""
以 ARMA(2,1) 为例：

<a id="eq:arma_lgssm_state_full"></a>
$$
\begin{aligned}
\begin{bmatrix}y_{t+1}\\\phi_2y_t+\theta_1\eta'_{t+1}\end{bmatrix}
&=
\begin{bmatrix}\phi_1&1\\\phi_2&0\end{bmatrix}
\begin{bmatrix}y_t\\\phi_2y_{t-1}+\theta_1\eta'_t\end{bmatrix}
+
\begin{bmatrix}1\\\theta_1\end{bmatrix}\eta'_{t+1},\\
\eta'_{t+1}&\sim\mathcal N(0,\sigma^2).
\end{aligned}
$$

因此状态噪声协方差是 $\mathbf Q_t=\mathbf A\sigma^2\mathbf A^T$，它只有秩 1。下面用 $\phi=[-0.1,0.5]$、$\theta=-0.25$、$\sigma=1.25$ 模拟。
        """,
        lines="L1649-L1680",
        anchors=("eq:arma_lgssm_state_full",),
    ),
    code(
        "ch06-arma-lgssm-simulate",
        r"""
def make_arma21_lgssm(phi, theta, sigma, num_timesteps):
    phi = tf.convert_to_tensor(phi, tf.float32)
    theta = tf.convert_to_tensor(theta, tf.float32)
    sigma = tf.convert_to_tensor(sigma, tf.float32)
    transition_matrix = tf.stack(
        [tf.stack([phi[..., 0], tf.ones_like(phi[..., 0])], axis=-1),
         tf.stack([phi[..., 1], tf.zeros_like(phi[..., 1])], axis=-1)],
        axis=-2,
    )
    innovation_loading = tf.stack([tf.ones_like(theta), theta], axis=-1)
    covariance = sigma[..., None, None] ** 2 * (
        innovation_loading[..., :, None] * innovation_loading[..., None, :]
    )
    # Q 理论上秩为 1；极小 jitter 只用于现代 TFP 的 Cholesky 数值稳定性。
    jitter = tf.eye(2, batch_shape=tf.shape(covariance)[:-2], dtype=tf.float32) * 1e-6
    transition_scale = tf.linalg.cholesky(covariance + jitter)
    return tfd.LinearGaussianStateSpaceModel(
        num_timesteps=num_timesteps,
        transition_matrix=tf.linalg.LinearOperatorFullMatrix(transition_matrix),
        transition_noise=tfd.MultivariateNormalTriL(
            loc=tf.zeros_like(innovation_loading), scale_tril=transition_scale
        ),
        observation_matrix=tf.linalg.LinearOperatorFullMatrix([[1.0, 0.0]]),
        observation_noise=tfd.MultivariateNormalDiag(scale_diag=[1e-4]),
        initial_state_prior=tfd.MultivariateNormalDiag(
            loc=tf.zeros_like(innovation_loading),
            scale_diag=tf.ones_like(innovation_loading) * 2.0,
        ),
    )

true_arma_phi = tf.constant([-0.1, 0.5], tf.float32)
true_arma_theta = tf.constant(-0.25, tf.float32)
true_arma_sigma = tf.constant(1.25, tf.float32)
arma_lgssm = make_arma21_lgssm(true_arma_phi, true_arma_theta, true_arma_sigma, 200)
arma_observed = arma_lgssm.sample(seed=split_seed("arma-lgssm-simulate")[0])
assert arma_observed.shape == (200, 1)
assert bool(tf.math.is_finite(arma_lgssm.log_prob(arma_observed)))
        """,
        lines="L1682-L1720",
        code_block="tfd_lgssm_arma_simulate",
        notes="用公开 LGSSM；对理论秩一 Q 和零观测噪声加入明确记录的微小 jitter。",
    ),
    code(
        "ch06-arma-lgssm-inference",
        r"""
@tfd.JointDistributionCoroutineAutoBatched
def arma21_model():
    phi = yield tfd.Sample(tfd.Normal(0.0, 0.5), 2, name="phi")
    theta = yield tfd.Normal(0.0, 0.5, name="theta")
    sigma = yield tfd.HalfNormal(2.0, name="sigma")
    yield make_arma21_lgssm(phi, theta, sigma, 200).copy(name="observed")

arma_draws, arma_stats = run_windowed_nuts(
    arma21_model,
    seed=split_seed("arma-lgssm-mcmc")[0],
    observed=arma_observed,
)
arma_idata = tfp_draws_to_idata(arma_draws, arma_stats)

fig = az.plot_trace(arma_idata, var_names=["phi", "theta", "sigma"], compact=True)
        """,
        lines="L1734-L1788",
        code_block="tfd_lgssm_arma_with_prior",
        figures=("fig:fig17_arma_lgssm_inference_result",),
        notes="公共 windowed NUTS；固定无状态种子；观测事件保留 T×1 形状。",
    ),
    md(
        "ch06-arma-inference-caption-arima",
        r"""
<a id="fig:fig17_arma_lgssm_inference_result"></a>
**图 6.17**　对模拟 ARMA(2,1) LGSSM 的 MCMC 结果。轨迹图与边际密度显示 $\phi_1,\phi_2,\theta_1,\sigma$ 的后验；参考线标出生成数据所用的真实值。

对积分阶数 $d=1$ 的 ARIMA，可把累积水平与差分同时纳入状态。令 $\Delta y_t=y_t-y_{t-1}$、$y_t=y_{t-1}+\Delta y_t$，并取 $\mathbf H_t=[1,1,0]$，一种 ARIMA(2,1,1) 转移是：

<a id="eq:arima_lgssm_state_transition"></a>
$$
\begin{aligned}
\begin{bmatrix}
y_{t-1}+\Delta y_t\\
\phi_1\Delta y_t+\phi_2\Delta y_{t-1}+\eta'_{t+1}+\theta_1\eta'_t\\
\phi_2\Delta y_t+\theta_1\eta'_{t+1}
\end{bmatrix}
&=
\begin{bmatrix}1&1&0\\0&\phi_1&1\\0&\phi_2&0\end{bmatrix}
\begin{bmatrix}y_{t-1}\\\Delta y_t\\\phi_2\Delta y_{t-1}+\theta_1\eta'_t\end{bmatrix}
+
\begin{bmatrix}0\\1\\\theta_1\end{bmatrix}\eta'_{t+1}.
\end{aligned}
$$

更高阶差分可以继续扩展状态向量。初始状态的选择会显著影响短序列的似然与预测；系统处理参见 {cite:t}`durbin2012time`。
        """,
        lines="L1778-L1837",
        anchors=("fig:fig17_arma_lgssm_inference_result", "eq:arima_lgssm_state_transition"),
    ),
    code(
        "ch06-cleanup-lgssm",
        r"""
release_resources(
    "filtered_means",
    "filtered_covs",
    "predicted_means",
    "predicted_covs",
    "observation_means",
    "observation_covs",
    "arma_draws",
    "arma_idata",
)
        """,
        lines="中文版现代化资源管理单元",
        code_block="cleanup_lgssm",
        notes="在 BSTS 之前释放滤波协方差和 ARMA 链。",
    ),
    md(
        "ch06-bsts-introduction",
        r"""
(bayesian-structural-time-series)=

### 贝叶斯结构时间序列

复杂时间序列可以把多个 LGSSM 成分组合起来。例如两个独立成分的块对角组合为：

<a id="eq:combining_lgssm"></a>
$$
\begin{aligned}
\mathbf F_t&=\begin{bmatrix}\mathbf F_{\mathbf1,t}&0\\0&\mathbf F_{\mathbf2,t}\end{bmatrix},
&\mathbf Q_t&=\begin{bmatrix}\mathbf Q_{\mathbf1,t}&0\\0&\mathbf Q_{\mathbf2,t}\end{bmatrix},
&X_t&=\begin{bmatrix}X_{1,t}\\X_{2,t}\end{bmatrix},\\
\mathbf H_t&=\begin{bmatrix}\mathbf H_{\mathbf1,t}&\mathbf H_{\mathbf2,t}\end{bmatrix},
&\mathbf R_t&=\mathbf R_{\mathbf1,t}+\mathbf R_{\mathbf2,t}.
\end{aligned}
$$

如果还存在非高斯或外生回归成分，其均值也可以加入观测模型，例如 $\epsilon_t\sim N(\hat\mu_t+\hat\psi_t,R_t)$。这类以趋势、季节性、回归和自回归等结构成分构成的模型，通常称为**贝叶斯结构时间序列**（BSTS）。TFP 的 `tfp.sts` 模块提供了一组可组合的公共结构成分。
        """,
        lines="L1839-L1892",
        anchors=("bayesian-structural-time-series", "eq:combining_lgssm"),
    ),
    code(
        "ch06-bsts-data-model",
        r"""
observed_births = tf.convert_to_tensor(births, tf.float32)

birth_trend = tfp.sts.LocalLinearTrend(
    observed_time_series=observed_births,
    name="local_linear_trend",
)
birth_seasonal = tfp.sts.Seasonal(
    num_seasons=12,
    observed_time_series=observed_births,
    name="annual_seasonal",
)
birth_model = tfp.sts.Sum(
    [birth_trend, birth_seasonal],
    observed_time_series=observed_births,
    name="birth_model",
)

assert len(birth_model.parameters) >= 3
        """,
        lines="L1900-L1921",
        code_block="tfp_sts_example2",
    ),
    code(
        "ch06-bsts-joint-distribution",
        r"""
# 迁移 Notebook 的损坏文本是：
# birth_model.joint_distribu2.15.0.post1tion(...)
# 这里恢复 TFP 0.25 的公开方法。
birth_model_jd = birth_model.joint_distribution(
    observed_time_series=observed_births
)

# 构造并计算一个先验样本的有限对数概率；不依赖弃用的 joint_log_prob。
birth_prior_sample = birth_model_jd.sample_unpinned(
    seed=split_seed("bsts-prior-check")[0]
)
birth_prior_log_prob = birth_model_jd.unnormalized_log_prob(birth_prior_sample)
assert bool(tf.reduce_all(tf.math.is_finite(birth_prior_log_prob)))
        """,
        lines="migration notebook cell 127 and source L1925-L1935",
        code_block="tfp_sts_model",
        notes="精确修复 joint_distribu2.15.0.post1tion 腐坏行，使用公开 joint_distribution。",
    ),
    code(
        "ch06-bsts-components",
        r"""
component_names = [component.name for component in birth_model.components]
parameter_names = [parameter.name for parameter in birth_model.parameters]
assert len(component_names) == 2
assert any("trend" in name for name in component_names)
assert any("seasonal" in name for name in component_names)

print("结构成分：", component_names)
print("超参数：", parameter_names)
        """,
        lines="L1943-L1954",
        code_block="tfp_sts_model_component",
    ),
    md(
        "ch06-bsts-latent-vs-parameters",
        r"""
结构成分的**潜在状态**与其**超参数**需要区分。以局部线性趋势为例，每个时间点都有水平与斜率状态，而状态创新尺度只是少数几个全局参数。MCMC 通常先对这些超参数抽样；给定参数后，LGSSM 可以通过卡尔曼算法高效地边缘化或恢复整条潜在轨迹。

图 6.18 的概念关系可以概括为：BSTS 是组合模型；其局部线性趋势、季节性和自回归成分可各自表示成 LGSSM；组合后仍可由卡尔曼算法高效计算条件似然。外部回归或非高斯成分则可能需要额外近似或 MCMC。
        """,
        lines="L1956-L1988",
        anchors=("fig:fig18_bsts_lgssm",),
    ),
    code(
        "ch06-bsts-lgssm-diagram",
        r"""
fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

outer = patches.FancyBboxPatch(
    (0.4, 0.5), 9.2, 5.0,
    boxstyle="round,pad=0.15",
    facecolor="0.96",
    edgecolor="0.15",
    linewidth=1.5,
)
ax.add_patch(outer)
ax.text(0.75, 5.05, "贝叶斯结构时间序列（BSTS）", fontsize=12, weight="bold")

components = [
    ("局部线性趋势", 1.0),
    ("年度季节性", 2.65),
    ("自回归成分", 4.30),
]
for label, y in components:
    box = patches.FancyBboxPatch(
        (1.0, y), 3.4, 1.0,
        boxstyle="round,pad=0.08",
        facecolor="white",
        edgecolor="0.25",
        linewidth=1.2,
    )
    ax.add_patch(box)
    ax.text(2.7, y + 0.63, label, ha="center", va="center", fontsize=10)
    ax.text(2.7, y + 0.30, "LGSSM 模块", ha="center", va="center", fontsize=8, color="0.35")
    ax.annotate("", xy=(6.0, 3.0), xytext=(4.45, y + 0.5), arrowprops={"arrowstyle": "->", "color": "0.25"})

sum_box = patches.FancyBboxPatch(
    (6.0, 2.35), 2.8, 1.3,
    boxstyle="round,pad=0.1",
    facecolor="0.88",
    edgecolor="0.15",
    linewidth=1.3,
)
ax.add_patch(sum_box)
ax.text(7.4, 3.0, "组合状态空间\n与观测模型", ha="center", va="center", fontsize=10)
        """,
        lines="L1970-L1977 plus migration notebook conceptual figure",
        figures=("fig:fig18_bsts_lgssm",),
        notes="用中文 Matplotlib 矢量图重建原英文栅格概念图。",
    ),
    code(
        "ch06-bsts-inference-forecast",
        r"""
birth_parameter_draws, birth_sampler_stats = run_windowed_nuts(
    birth_model_jd,
    seed=split_seed("bsts-mcmc")[0],
)

if hasattr(birth_parameter_draws, "_asdict"):
    birth_parameter_mapping = dict(birth_parameter_draws._asdict())
elif isinstance(birth_parameter_draws, Mapping):
    birth_parameter_mapping = dict(birth_parameter_draws)
else:
    birth_parameter_mapping = {
        parameter.name: value
        for parameter, value in zip(birth_model.parameters, birth_parameter_draws)
    }

# 中文版现代化说明：windowed_adaptive_nuts 返回的每个参数张量前两轴分别是
# draw、chain；tfp.sts.decompose_by_component/forecast 期望一条展平后的样本轴，
# 而不是把 chain 轴当成模型自身的批量轴。用 flatten_and_select 展平并按
# BUDGET["sts_parameter_samples"] 确定性抽取子样本，而不是把原始
# [draw, chain, ...] 数组直接传入。
birth_parameter_samples = [
    flatten_and_select(
        birth_parameter_mapping[parameter.name],
        BUDGET["sts_parameter_samples"],
        name=parameter.name,
    )
    for parameter in birth_model.parameters
]

birth_component_dists = tfp.sts.decompose_by_component(
    birth_model,
    observed_time_series=observed_births,
    parameter_samples=birth_parameter_samples,
)

n_birth_forecast = 36
birth_forecast_dist = tfp.sts.forecast(
    birth_model,
    observed_time_series=observed_births,
    parameter_samples=birth_parameter_samples,
    num_steps_forecast=n_birth_forecast,
)

birth_forecast_mean = np.asarray(birth_forecast_dist.mean()).squeeze()
birth_forecast_sd = np.asarray(birth_forecast_dist.stddev()).squeeze()
birth_forecast_dates = pd.date_range(
    start=us_monthly_birth.index[-1] + pd.offsets.MonthBegin(1),
    periods=n_birth_forecast,
    freq="MS",
)
assert len(birth_forecast_dates) == n_birth_forecast
assert birth_forecast_dates[0].day == 1
assert birth_forecast_mean.shape[-1] == n_birth_forecast
        """,
        lines="L1995-L2045 plus migration notebook BSTS sampler",
        code_block="tfp_sts_example2_result",
        notes=(
            "使用 MonthBegin/freq=MS 替代 np.timedelta64('M') 与 pandas freq='M'；"
            "并用 flatten_and_select 把 [draw, chain, ...] 参数样本展平为一条样本轴、"
            "按 sts_parameter_samples 预算抽取子样本，避免 chain 轴被 tfp.sts 误当成"
            "模型自身的批量轴而导致成分分解/预测形状错误。"
        ),
    ),
    code(
        "ch06-bsts-result-plot",
        r"""
fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
axes[0].plot(us_monthly_birth.index, births, color="0.10", lw=1.2, label="观测")
axes[0].plot(birth_forecast_dates, birth_forecast_mean, color="0.20", ls="--", lw=1.5, label="36 个月预测")
axes[0].fill_between(
    birth_forecast_dates,
    birth_forecast_mean - 1.88 * birth_forecast_sd,
    birth_forecast_mean + 1.88 * birth_forecast_sd,
    color="0.55",
    alpha=0.25,
)
axes[0].set(ylabel="活产数（千人）")
axes[0].legend(frameon=False)

for ax, (component, distribution) in zip(axes[1:], birth_component_dists.items()):
    component_mean = np.asarray(distribution.mean()).squeeze()
    component_sd = np.asarray(distribution.stddev()).squeeze()
    ax.plot(us_monthly_birth.index, component_mean, color="0.10", lw=1.2, label=component.name)
    ax.fill_between(
        us_monthly_birth.index,
        component_mean - 1.88 * component_sd,
        component_mean + 1.88 * component_sd,
        color="0.55",
        alpha=0.25,
    )
    ax.legend(frameon=False)
axes[-1].set_xlabel("年份")
        """,
        lines="L1995-L2055",
        figures=("fig:fig19_bsts_lgssm_result",),
        notes="重绘中文轴标签与图例；单轴小多图，不依赖颜色区分。",
    ),
    md(
        "ch06-bsts-captions",
        r"""
<a id="fig:fig18_bsts_lgssm"></a>
**图 6.18**　BSTS 与 LGSSM 的关系。BSTS 可以由局部线性趋势、季节性和自回归等结构模块组成；每个线性高斯模块都能表示为 LGSSM，组合后形成更大的块状态空间。

<a id="fig:fig19_bsts_lgssm_result"></a>
**图 6.19**　美国月度活产数的 BSTS 推断与预测。上图显示 36 个月预测及 94% 区间；下方两个面板分别展示局部线性趋势和年度季节性结构分解。
        """,
        lines="L1970-L1977 and L2047-L2055",
        anchors=("fig:fig18_bsts_lgssm", "fig:fig19_bsts_lgssm_result"),
    ),
    code(
        "ch06-cleanup-bsts",
        r"""
release_resources(
    "birth_prior_sample",
    "birth_parameter_draws",
    "birth_sampler_stats",
    "birth_component_dists",
    "birth_forecast_dist",
)
        """,
        lines="中文版现代化资源管理单元",
        code_block="cleanup_bsts",
        notes="在概念总结与练习前释放 STS 参数链和分解分布。",
    ),
    md(
        "ch06-other-time-series-models",
        r"""
(other-time-series-models)=

## 其他时间序列模型

LGSSM 之所以高效，是因为线性与高斯假设使分布保持闭式。但现实系统可能非线性或非高斯。常见扩展包括：

- **扩展卡尔曼滤波**：在当前估计附近对非线性函数做一阶线性化 {cite:p}`grewal2014kalman`；
- **无迹卡尔曼滤波**：用一组确定性的 sigma 点传播均值与协方差 {cite:p}`grewal2014kalman`；
- **粒子滤波**：用带权粒子近似任意状态分布 {cite:p}`Chopin2020`；
- **隐马尔可夫模型**：潜在状态离散，适合状态切换；
- **常微分方程与随机微分方程**：在连续时间描述确定性或随机动力系统。

<a id="table:ts_model_type"></a>
**表 6.2　按随机性与时间处理方式分类的时间序列模型**

|  | 确定性动力学 | 随机动力学 |
|---|---|---|
| 离散时间 | 自动机 / 离散化 ODE | 状态空间模型 |
| 连续时间 | ODE | SDE |

选择模型时，应从数据生成机制、观测频率、计算预算与决策目标出发，而不是只按软件中最容易调用的类来选择。
        """,
        lines="L2057-L2097",
        anchors=("other-time-series-models", "table:ts_model_type"),
    ),
])

cells.extend([
    md(
        "ch06-model-criticism",
        r"""
(model-criticism-and-choosing-priors)=

## 模型批判与先验选择

经典时间序列著作 {cite:t}`box2008time`[^15] 列出了五类重要的实际问题：

- 预测；
- 传递函数估计；
- 分析异常干预事件对系统的影响；
- 多元时间序列分析；
- 离散控制系统。

实践中的大多数问题都包含某种预测；也可能是**即时预测（nowcasting）**，即在时刻 $t$ 推断因测量延迟而尚未获得的当前量。预测任务为模型批判提供了自然标准。虽然本章不专门展开贝叶斯决策理论，但 {cite:t}`westharrison1997` 的话值得牢记：

> 好的建模需要艰苦思考，而好的预测则要求把预测在决策系统中的作用视为一个整体。

因此，时间序列推断的批判、预测评价与决策过程应紧密结合，尤其要明确不确定性怎样进入决策。当然，也可以单独评价预测表现：等待新数据，或像 CO₂ 示例那样预留测试集，再用标准指标比较观测与预测。

一个常见指标是平均绝对百分比误差（MAPE）：

<a id="eq:mape"></a>
$$
MAPE=\frac{1}{n}\sum_{i=1}^{n}
\frac{|\text{forecast}_i-\text{observed}_i|}{\text{observed}_i}.
$$

MAPE 有已知偏差：观测值较小时，同样的绝对误差会造成极大的百分比误差；若多个序列的尺度和取值范围差异很大，MAPE 也难以横向比较。若观测可能为 0 或负数，它甚至可能无定义或难以解释。
        """,
        lines="L2099-L2146",
        anchors=("model-criticism-and-choosing-priors", "eq:mape"),
    ),
    code(
        "ch06-forecast-metrics",
        r"""
def mean_absolute_percentage_error(observed, forecast, epsilon=1e-8):
    observed = np.asarray(observed, dtype=np.float64)
    forecast = np.asarray(forecast, dtype=np.float64)
    if observed.shape != forecast.shape:
        raise ValueError("observed 与 forecast 的形状必须相同")
    denominator = np.maximum(np.abs(observed), epsilon)
    return np.mean(np.abs(forecast - observed) / denominator)

# 对简单回归预测的测试区间做语义检查；MAPE 保持比例值而非乘 100。
if "regression_pp_dist" in globals():
    regression_forecast_mean = np.asarray(regression_pp_dist.mean())[-num_forecast_steps:].mean(axis=-1)
    regression_test_mape = mean_absolute_percentage_error(
        co2_testing["CO2"].to_numpy(), regression_forecast_mean
    )
    assert np.isfinite(regression_test_mape) and regression_test_mape >= 0.0
        """,
        lines="L2131-L2146",
        code_block="mape_semantic_check",
        notes="增加零分母保护与形状检查；保持原式的比例定义。",
    ),
    md(
        "ch06-leave-future-out",
        r"""
基于交叉验证的模型评价仍适用于时间序列，也值得推荐；但若目标是未来预测，对单条时间序列直接做普通 LOO 会有问题。逐点删去 $t$ 后，用其余数据预测它，不仅用了过去的 $t_{-1},t_{-2},\ldots$，也用了未来的 $t_{+1},t_{+2},\ldots$，相当于用未来预测过去。即使能够计算 LOO，其数值解释也会误导。

更合适的是某种**留未来交叉验证**（Leave-Future-Out Cross-Validation，LFO-CV；参见 {cite:t}`Burkner2020`）。粗略地说：先在初始时间窗拟合模型，再沿预留的未来序列迭代，评价一步或多步对数预测密度；当 Pareto $k$ 等诊断超过阈值时，把新时间点纳入数据并重新拟合[^16]。LFO-CV 不是唯一固定算法，而是一族尊重时间箭头、用过去预测未来的验证方案。
        """,
        lines="L2148-L2169",
    ),
    md(
        "ch06-priors-time-series",
        r"""
(priors-for-time-series-models)=

### 时间序列模型的先验

在 {ref}`chp4_gam` 中，分段线性趋势的斜率变化使用了 Laplace 正则化先验，以表达“斜率变化通常较小并接近 0”的知识，使潜在趋势更平滑。稀疏先验也常用于节假日或特殊日期效应：每个节假日有自己的系数，但我们相信少数节假日可能影响很大，大多数则与普通日期相似。

马蹄先验可以形式化这种直觉 {cite:p}`carvalho2010horseshoe, piironen2017sparsity`：

<a id="eq:horse_shoe"></a>
$$
\begin{aligned}
\lambda_t^2&\sim\mathcal H\mathrm C(1),\\
\beta_t&\sim\mathcal N(0,\lambda_t^2\tau^2).
\end{aligned}
$$

全局参数 $\tau$ 把所有节假日系数整体拉向 0；局部尺度 $\lambda_t$ 的重尾允许少数效应逃离收缩。$\tau$ 越接近 0，整体越稀疏；$\tau$ 越大，先验越分散 {cite:p}`piironen2017hyperprior`[^17]。例如，{cite:t}`riutort2020practical` 的案例为一年中的每一天（包括闰日，共 366 天）设置特殊日期效应，并用马蹄先验正则化。
        """,
        lines="L2171-L2206",
        anchors=("priors-for-time-series-models", "eq:horse_shoe"),
    ),
    md(
        "ch06-observation-noise-priors",
        r"""
观测噪声先验同样重要。大多数时间序列观测本质上不可重复：我们无法回到同一时刻，在完全相同条件下再次测量，因此不能从重复实验中直接量化**偶然不确定性（aleatoric uncertainty）**。模型需要借助先验在“测量噪声”和“潜在过程不确定性（epistemic uncertainty）”之间分配变化。

例如，在包含潜在 AR 成分或局部线性趋势的模型中，更强地把观测噪声约束到较小值，会迫使趋势或 AR 成分贴合更多短期波动，短期预测看起来可能更准确；代价是对潜在趋势过度自信，长期预测反而可能更差。现实时间序列往往非平稳，因此应根据应用与灵敏度分析调整这些先验，而不能把噪声分解当成完全由数据识别的事实。
        """,
        lines="L2208-L2224",
    ),
    md(
        "ch06-exercises-heading",
        r"""
(exercises6)=

## 练习
        """,
        lines="L2226-L2228",
        anchors=("exercises6",),
    ),
    md(
        "ch06-exercises-e1-e4",
        r"""
**6E1.** 正如上文“把时间戳解析为设计矩阵”所述，日期信息可以格式化为回归设计矩阵，以表示时间序列的周期模式。请为 2021 年生成以下设计矩阵。可用代码块 `timerange_2021` 生成这一年的全部时间戳：

- 一个月中第几天的效应；
- 工作日与周末效应；
- G 公司每月 25 日发薪；若 25 日是周末，则提前到前一个星期五。请编码 2021 年的发薪日；
- 2021 年美国联邦节假日效应[^18]，每个节假日各有一个独立系数。

**6E2.** 上一题把每个节假日分别处理。如果认为所有节假日共享同一效应，设计矩阵的形状是什么？请说明这种部分池化程度的变化会怎样影响时间序列回归拟合。

**6E3.** 对 `monthly_mauna_loa_co2.csv` 拟合以下线性回归：

- 只有截距和线性时间斜率的普通回归；
- 类似第 [4](chap3) 章代码块 `babies_transformed` 中平方根预测变量的协变量变换回归。

说明与代码块 `regression_model_for_timeseries` 相比，这些模型遗漏了什么。

**6E4.** 用自己的话解释回归、自回归和状态空间三种架构的差异，并各举一种特别适用的情境。
        """,
        lines="L2230-L2277",
    ),
    code(
        "ch06-exercise-2021-time-range",
        r"""
datetime_index = pd.date_range(
    start="2021-01-01", end="2021-12-31", freq="D"
)
assert len(datetime_index) == 365
assert datetime_index[0].day_name() == "Friday"
        """,
        lines="L2237-L2242",
        code_block="timerange_2021",
        notes="保留日频语义并增加闰年/起始日断言。",
    ),
    md(
        "ch06-exercises-m5-m6",
        r"""
**6M5.** 用基函数构造的设计矩阵是否真的比稀疏矩阵有更好的条件数？使用 `numpy.linalg.cond` 比较以下同形状设计矩阵（注意它们未必同秩——`X_pred` 恰好在 Nyquist 频率处有一列 $\sin(\pi t)$ 对整数月份取值恒为 0，秩比列数少 1；这本身就是条件数比较中值得注意的一点）：

- 代码块 `generate_design_matrix` 中虚拟编码的 `seasonality_all`；
- 代码块 `gam` 中 Fourier 基函数矩阵 `X_pred`；
- 与 `seasonality_all` 同形状、元素来自正态分布的数组；
- 同样的正态随机数组，但让其中一列与另一列完全相同。

**6M6.** 代码块 `fourier_basis_as_seasonality` 中的 `gen_fourier_basis` 把时间索引 `t` 作为第一个输入。对于从 2019 年 1 月开始、共 36 个月的月度观测，可用连续月份编号，也可用一年中的月份编号。运行代码块 `exercise_chap4_e6`，再考虑日频观测：

- 让 `time_index` 表示一年中的第几天而不是第几月；
- 修改两次 `gen_fourier_basis` 调用的参数，使设计矩阵编码一年中月份的效应；
- 新的 `design_matrix0` 与 `design_matrix1` 有何差异？会怎样影响拟合？可让两者乘以同一组随机回归系数验证推理。
        """,
        lines="L2279-L2336",
    ),
    code(
        "ch06-exercise-fourier-index",
        r"""
nmonths = 36
day0 = pd.Timestamp("2019-01-01")
time_index = pd.date_range(start=day0, periods=nmonths, freq="MS")

t0 = np.arange(len(time_index))
design_matrix0 = gen_fourier_basis(t0, p=12, n=6)
t1 = time_index.month.to_numpy() - 1
design_matrix1 = gen_fourier_basis(t1, p=12, n=6)

np.testing.assert_array_almost_equal(design_matrix0, design_matrix1)
        """,
        lines="L2305-L2321",
        code_block="exercise_chap4_e6",
        notes="用 periods/freq='MS' 替代不支持的 np.timedelta64(...,'M') 与 freq='M'。",
    ),
    md(
        "ch06-exercises-e7-e11",
        r"""
**6E7.** 在 {ref}`chap4_ar` 中，我们介绍了后移算子 $\mathbf B$。对时间序列应用 $\mathbf B$ 等价于矩阵乘法。修改代码块 `ar1_without_forloop`，用 NumPy 或 TensorFlow 显式构造的 $\mathbf B$ 实现。

**6E8.** 式 {eq}`eq:step_linear_function` 与代码块 `step_linear_function_for_trend` 的分段线性函数依赖关键系数 $\delta$。把它改写成与线性回归相似的形式：

<a id="eq:step_linear"></a>
$$
g(t)=\mathbf A'\delta'.
$$

求设计矩阵 $\mathbf A'$ 和系数 $\delta'$ 的适当表达式。

**6E9.** 写出生成过程是理解数据的好方法。令基础线性趋势为 `y = 2*x`、`x = np.arange(90)`，每个时间点再加独立同分布的 $\mathcal N(0,1)$ 噪声。假设序列从 2021 年 6 月 6 日（星期日）开始，生成四个合成数据集：

1. 加入周末效应：周末的量为工作日的 2 倍（请明确你采用加法还是乘法定义）；
2. 加入 $\sin(2x)$ 的正弦效应；
3. 加入潜在 AR(1) 过程，自选自回归系数，噪声尺度 $\sigma=0.2$；
4. 同时包含第 1、2 项，并让与第 3 项相同系数的 AR(1) 过程作用于时间序列均值。

**6E10.** 调整代码块 `gam_with_ar_likelihood`，为 **6E9** 第 4 个合成序列建模。

**6E11.** 用 ArviZ 检查本章各模型的 MCMC 轨迹与诊断，包括轨迹图、秩图和后验摘要。哪些模型有发散、有效样本量低或 $\hat R$ 偏大等问题？能否通过重参数化、更合理先验、调整步长或增加适应预算来改善？
        """,
        lines="L2338-L2397",
        anchors=("eq:step_linear",),
        notes="6E9 原文把“加法”与“2 倍”混用；译文要求解题者明确采用的定义。",
    ),
    md(
        "ch06-exercises-m12-m16",
        r"""
**6M12.** 用 Python 生成 200 个时间点的正弦时间序列，并拟合 AR(2) 模型。在 TFP 中修改代码块 `ar1_without_forloop` 完成；再用 PyMC 的公共 `pm.AR` API 完成一次，对比两种参数化与结果。

**6M13.** 对上一题的 AR(2) 模型做后验预测检验。对每个时间步 $t$ 生成预测分布，并只条件于截至 $t-1$ 的全部观测。一步预测分布是否与观测序列相符？

> **中文版现代化说明**：原文的 6M13 和 6M14 都引用“练习 6M11”，但 AR(2) 构造实际编号为 **6M12**。这里修正叙述性引用，同时保留题号与来源记录。

**6M14.** 使用 6M12 的 AR(2) 模型向未来预测 50 个时间步。预测是否仍像正弦信号？说明预测均值衰减或不确定性扩张与 AR 根之间的关系。

**6H15.** 实现 $\operatorname{SARIMA}(1,1,1)(1,1,1)_{12}$ 的生成过程，并进行预测。

**6M16.** 对本章月度活产数据实现并推断 $\operatorname{ARIMAX}(1,1,1)\mathbf X[4]$，其中设计矩阵由 $N=2$ 的 Fourier 基函数生成。
        """,
        lines="L2399-L2421",
        notes="明确说明原书 6M11/6M12 编号不一致。",
    ),
    md(
        "ch06-exercises-h17-m20",
        r"""
**6H17.** 推导卡尔曼滤波方程。提示：先求 $X_t$ 与 $X_{t-1}$ 的联合分布，再求 $Y_t$ 与 $X_t$ 的联合分布。若仍有困难，参见 Särkkä 的书第 4 章 {cite:p}`sarkka2013bayesian`。

**6M18.** 索引 `linear_growth_lgssm.forward_filter` 在某个给定时间步的输出：

- 找出一次卡尔曼滤波步骤的输入与输出；
- 用这些输入手工计算一次预测和更新；
- 用断言确认手工结果与索引得到的输出一致。

**6M19.** 阅读 `tfp.sts.Seasonal` 的文档与实现，并回答：

- 季节性 SSM 有多少个超参数？
- 它如何参数化潜在状态？先验产生什么正则化效果？提示：联系第 [5](chap3_5) 章的高斯随机游走先验。

**6M20.** 阅读 `tfp.sts.LinearRegression` 与 `tfp.sts.Seasonal` 的文档与实现。若用它们表示星期几模式，请比较：

- 星期几系数如何表示？它们是否属于潜在状态？
- 两个 SSM 的拟合方式有何不同？用模拟验证推理。
        """,
        lines="L2423-L2456",
    ),
    md(
        "ch06-footnotes",
        r"""
## 脚注

[^1]: 这句话的出处与归属讨论见 <https://quoteinvestigator.com/2013/10/20/no-predict/>。

[^2]: 并非时间序列中所有周期模式都应称为季节性。区分循环行为与季节行为很有用；简要总结见 <https://robjhyndman.com/hyndsight/cyclicts/>。

[^3]: 这使观测不再独立同分布，也不再可交换。残差的定义还可参见第 [4](chap3) 章。

[^4]: 这既不利于我们的模型，也不利于我们的星球。

[^5]: 如果均值、协方差等特征性质随时间平移保持不变，就称序列为平稳序列。

[^6]: <https://facebook.github.io/prophet/>。

[^7]: PyMCon 2020 演讲提供了 Facebook Prophet 设计矩阵演示：<http://prophet.mbrouns.com>。

[^8]: 这就是“自回归”名称的由来：对自身做线性回归。因此，它与 {ref}`autocorr_plot` 中介绍的自相关诊断名称相似。

[^9]: 严格说来，本节的 AR 示例本身就是一个高斯过程。

[^10]: SARIMA 的 Stan 实现可参见 <https://github.com/asael697/bayesforecast>。

[^11]: 原书为简洁省略了 MCMC 抽样代码，并说明细节在随附 Jupyter Notebook 中；本中文版规范源已将现代化公共 API 路径补回。

[^12]: 可以先把这里的“空间”理解为多维欧氏空间；在 Python 计算中，$X_t$ 与 $Y_t$ 就是多维数组或张量。

[^13]: 这也给出了观测矩阵 $\mathbf H$ 非平稳的一个清晰例子。

[^14]: 这不是把 ARMA 写成状态空间形式的唯一方法。更多细节见讲义 <http://www-stat.wharton.upenn.edu/~stine/stat910/lectures/14_state_space.pdf>。

[^15]: 阅读 Box 的经典著作并亲自处理预测问题，最能体会 George E. P. Box 的名言：“所有模型都是错的，但有些是有用的。”

[^16]: 演示见 <https://mc-stan.org/loo/articles/loo2-lfo.html>。

[^17]: 实践中通常会对式 {eq}`eq:horse_shoe` 使用略有不同、数值更稳定的参数化。

[^18]: 美国联邦节假日列表见 <https://en.wikipedia.org/wiki/Federal_holidays_in_the_United_States#List_of_federal_holidays>。
        """,
        lines="L2458-L2512",
    ),
    md(
        "ch06-closing-provenance",
        r"""
## 中文版来源与执行说明

本章中文文本的完整性权威是 `markdown/chp_06.md`；`notebooks_updated/chp_06.ipynb` 仅作为代码迁移起点。两个数据文件按字节复制到本章 `data/`，哈希记录在 `manifest.toml`。本规范源不读取答案目录，也不在默认检查中生成最终 Notebook/Org 或运行完整 MCMC。

> **中文版补充**：源 CSV 的 CO₂ 最后日期是 2019 年 1 月，而原图注写到 2019 年 2 月；月度活产 CSV 有 373 条观测，而原代码注释写 372。本章以数据文件为准，并在清单中保留差异说明。
        """,
        lines="中文版补充；源差异见 L83-L104、L1091-L1120",
        notes="记录数据与原图注/注释不一致，不静默改写来源。",
    ),
])

REQUIRED_CODE_BLOCKS = {
    "load_co2_data",
    "generate_design_matrix",
    "regression_model_for_timeseries",
    "prior_predictive",
    "inference_of_regression_model",
    "posterior_predictive_with_component",
    "step_linear_function_for_trend",
    "fourier_basis_as_seasonality",
    "gam",
    "ar1_with_forloop",
    "ar1_without_forloop",
    "gam_alternative",
    "gam_with_ar_likelihood",
    "gam_with_latent_ar",
    "gw_tfp",
    "sarima_preprocess",
    "sarima_likelihood",
    "sarima_posterior",
    "linear_growth_model",
    "tfd_lgssm_linear_growth",
    "tfd_lgssm_linear_growth_filter",
    "tfd_lgssm_arma_simulate",
    "tfd_lgssm_arma_with_prior",
    "tfp_sts_example2",
    "tfp_sts_model",
    "tfp_sts_model_component",
    "tfp_sts_example2_result",
    "timerange_2021",
    "exercise_chap4_e6",
}

# gam_components/make_gam 是原 gam_alternative 的现代化模块化实现。
CODE_BLOCK_ALIASES = {"gam_alternative": "ch06-gam-design-and-model"}

REQUIRED_EXERCISES = {
    "6E1", "6E2", "6E3", "6E4", "6M5", "6M6", "6E7", "6E8", "6E9",
    "6E10", "6E11", "6M12", "6M13", "6M14", "6H15", "6M16", "6H17",
    "6M18", "6M19", "6M20",
}

REQUIRED_CITATIONS = {
    "Rasmussen2005",
    "TaylorLetham2018",
    "adams2007bayesian",
    "strang09",
    "109876",
    "box2008time",
    "shumway2019time",
    "lao2020tfpmcmc",
    "westharrison1997",
    "sarkka2013bayesian",
    "durbin2012time",
    "grewal2014kalman",
    "Chopin2020",
    "Burkner2020",
    "carvalho2010horseshoe",
    "piironen2017sparsity",
    "piironen2017hyperprior",
    "riutort2020practical",
}

ALLOWED_TOP_LEVEL_MANIFEST_KEYS = {
    "schema_version", "unit", "notebook", "execution", "assets"
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell_anchor_inventory() -> set[str]:
    result: set[str] = set()
    for cell in cells:
        result.update(cell.get("metadata", {}).get("zh", {}).get("anchors", []))
        result.update(cell.get("metadata", {}).get("zh", {}).get("figures", []))
    return result


def _code_block_inventory() -> set[str]:
    result: set[str] = set()
    for cell in cells:
        value = cell.get("metadata", {}).get("zh", {}).get("code_block")
        if value:
            result.add(value)
    result.update(CODE_BLOCK_ALIASES)
    return result


def _manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open("rb") as handle:
        return tomllib.load(handle)


def validate_source() -> dict[str, Any]:
    """验证规范源；不会导入 TensorFlow、运行 MCMC 或写出 Notebook/Org。"""
    normalized = nb_tools.validate_cells(cells)
    ids = [cell["id"] for cell in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("单元格 ID 不唯一")
    undescriptive = [cell_id for cell_id in ids if not cell_id.startswith("ch06-")]
    if undescriptive:
        raise ValueError(f"单元格 ID 必须以 ch06- 开头：{undescriptive}")

    compiled_code_cells = 0
    forbidden_imports: list[str] = []
    for cell in normalized:
        cell_metadata = cell.get("metadata", {})
        metadata = cell_metadata.get("zh", {})
        if not cell_metadata.get("kind"):
            raise ValueError(f"单元格 {cell['id']} 缺少直接 metadata.kind")
        provenance = cell_metadata.get("provenance")
        if not isinstance(provenance, dict) or not provenance.get("source"):
            raise ValueError(f"单元格 {cell['id']} 缺少直接 metadata.provenance")
        if not metadata.get("source_lines"):
            raise ValueError(f"单元格 {cell['id']} 缺少 source_lines 来源信息")
        if cell["type"] != "code":
            continue
        compiled_code_cells += 1
        try:
            compile(cell["source"], f"<{cell['id']}>", "exec")
        except SyntaxError as exc:
            raise SyntaxError(f"代码单元格 {cell['id']} 静态编译失败：{exc}") from exc
        for line in cell["source"].splitlines():
            compact = line.strip().replace(" ", "")
            if compact.startswith("fromtensorflow_probability.python.internal"):
                forbidden_imports.append(f"{cell['id']}: {line.strip()}")
            if compact.startswith("importtensorflow_probability.python.internal"):
                forbidden_imports.append(f"{cell['id']}: {line.strip()}")
            if "solutions/" in compact or compact.startswith("fromsolutions"):
                forbidden_imports.append(f"{cell['id']}: {line.strip()}")
    if forbidden_imports:
        raise ValueError("检测到禁止的私有/答案导入：\n" + "\n".join(forbidden_imports))

    found_anchors = _cell_anchor_inventory()
    expected_anchors = (
        REQUIRED_ANCHORS
        | REQUIRED_EQUATION_ANCHORS
        | REQUIRED_FIGURE_ANCHORS
        | REQUIRED_TABLE_ANCHORS
    )
    missing_anchors = expected_anchors - found_anchors
    if missing_anchors:
        raise ValueError(f"缺少锚点：{sorted(missing_anchors)}")

    missing_code_blocks = REQUIRED_CODE_BLOCKS - _code_block_inventory()
    if missing_code_blocks:
        raise ValueError(f"缺少源代码块映射：{sorted(missing_code_blocks)}")

    combined_markdown = "\n".join(
        cell["source"] for cell in normalized if cell["type"] == "markdown"
    )
    missing_exercises = {
        exercise for exercise in REQUIRED_EXERCISES
        if not re.search(rf"\*\*{re.escape(exercise)}\.\*\*", combined_markdown)
    }
    if missing_exercises:
        raise ValueError(f"缺少练习：{sorted(missing_exercises)}")

    missing_footnotes = {
        str(number) for number in range(1, 19)
        if f"[^{number}]:" not in combined_markdown
    }
    if missing_footnotes:
        raise ValueError(f"缺少脚注定义：{sorted(missing_footnotes)}")

    missing_citations = {
        key for key in REQUIRED_CITATIONS if key not in combined_markdown
    }
    if missing_citations:
        raise ValueError(f"缺少引用：{sorted(missing_citations)}")

    checked_hashes: dict[str, str] = {}
    for relative, expected in SOURCE_HASHES.items():
        path = HERE / relative if relative.startswith("data/") else REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"哈希不匹配：{path}，期望 {expected}，实际 {actual}")
        checked_hashes[relative] = actual

    manifest = _manifest()
    unexpected_top_level = set(manifest) - ALLOWED_TOP_LEVEL_MANIFEST_KEYS
    if unexpected_top_level:
        raise ValueError(f"manifest.toml 顶层字段不兼容标准清单：{sorted(unexpected_top_level)}")
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest.toml schema_version 必须为 1")

    canonical_fields = {
        "unit": {"id", "title", "builder", "status", "environment"},
        "notebook": {"file", "org_file"},
        "execution": {"cwd", "kernel", "timeout", "allow_errors"},
        "assets": {"directory"},
    }
    for table_name, expected_fields in canonical_fields.items():
        actual_fields = set(manifest.get(table_name, {}))
        if actual_fields != expected_fields:
            raise ValueError(
                f"manifest [{table_name}] 字段不符合共享 schema；"
                f"期望={sorted(expected_fields)}，实际={sorted(actual_fields)}"
            )
    if manifest["unit"]["id"] != "chapter-6":
        raise ValueError("manifest unit.id 必须为 chapter-6")

    # 共享 schema 不允许额外清单字段，因此详细库存以结构化 TOML 注释保存；
    # 验证器仍逐项确认注释没有遗漏或陈旧。
    manifest_text = MANIFEST_PATH.read_text(encoding="utf-8")
    manifest_tokens = (
        expected_anchors
        | REQUIRED_CODE_BLOCKS
        | REQUIRED_EXERCISES
        | REQUIRED_CITATIONS
        | set(SOURCE_HASHES)
        | set(SOURCE_HASHES.values())
    )
    missing_manifest_tokens = {
        token for token in manifest_tokens if token not in manifest_text
    }
    if missing_manifest_tokens:
        raise ValueError(
            f"manifest 详细注释清单缺少：{sorted(missing_manifest_tokens)}"
        )
    if f"cells={len(normalized)}" not in manifest_text:
        raise ValueError("manifest 注释中的单元格计数与 build.py 不一致")

    # 本次交付明确不在章节目录中生成最终产物。
    generated = [
        HERE / manifest["notebook"]["file"],
        HERE / manifest["notebook"]["org_file"],
    ]
    unexpected_generated = [str(path) for path in generated if path.exists()]
    if unexpected_generated:
        raise ValueError(f"本次源交付不应包含生成文件：{unexpected_generated}")

    return {
        "cells": len(normalized),
        "code_cells_compiled": compiled_code_cells,
        "anchors": len(found_anchors & expected_anchors),
        "sections": len(REQUIRED_ANCHORS),
        "equations": len(REQUIRED_EQUATION_ANCHORS),
        "figures": len(REQUIRED_FIGURE_ANCHORS),
        "tables": len(REQUIRED_TABLE_ANCHORS),
        "exercises": len(REQUIRED_EXERCISES),
        "footnotes": 18,
        "hashes": checked_hashes,
    }


def runtime_api_check() -> dict[str, Any]:
    """只做轻量导入、有限 log_prob 和模型构造；绝不运行 MCMC。"""
    import numpy as np
    import tensorflow as tf
    import tensorflow_probability as tfp

    tfd = tfp.distributions
    versions = {
        "tensorflow": tf.__version__,
        "tensorflow_probability": tfp.__version__,
    }
    if not tfp.__version__.startswith("0.25"):
        raise RuntimeError(f"需要 TensorFlow Probability 0.25，实际为 {tfp.__version__}")

    @tfd.JointDistributionCoroutineAutoBatched
    def regression_check():
        scale = yield tfd.HalfNormal(1.0, name="scale")
        yield tfd.Independent(
            tfd.Normal(tf.zeros(8), scale[..., None]),
            reinterpreted_batch_ndims=1,
            name="observed",
        )

    jd_seed = tfp.random.sanitize_seed(606, salt="chapter6-runtime-jd")
    jd_sample = regression_check.sample(seed=jd_seed)
    jd_log_prob = regression_check.log_prob(jd_sample)
    if not bool(tf.reduce_all(tf.math.is_finite(jd_log_prob))):
        raise RuntimeError("JointDistribution 轻量 log_prob 非有限")

    # 公开 tf.scan 的完整 AR 轨迹。
    innovations = tf.ones(8, tf.float32) * 0.1
    ar_path = tf.scan(
        lambda previous, innovation: 0.7 * previous + innovation,
        innovations,
        initializer=tf.constant(0.0),
    )
    if ar_path.shape != (8,) or not bool(tf.reduce_all(tf.math.is_finite(ar_path))):
        raise RuntimeError("公开 tf.scan AR 轨迹检查失败")

    # 公开 tf.while_loop 的 SARIMA 类递推骨架。
    residual_array = tf.TensorArray(tf.float32, size=8, clear_after_read=False)
    values = tf.linspace(0.0, 1.0, 8)

    def recurrence_body(index, residuals):
        previous = tf.cond(index > 0, lambda: residuals.read(index - 1), lambda: 0.0)
        return index + 1, residuals.write(index, values[index] - 0.2 * previous)

    _, residual_array = tf.while_loop(
        lambda index, _: index < 8,
        recurrence_body,
        (tf.constant(0), residual_array),
        parallel_iterations=1,
    )
    if not bool(tf.reduce_all(tf.math.is_finite(residual_array.stack()))):
        raise RuntimeError("公开 tf.while_loop 递推检查失败")

    lgssm = tfd.LinearGaussianStateSpaceModel(
        num_timesteps=6,
        transition_matrix=tf.linalg.LinearOperatorIdentity(2, dtype=tf.float32),
        transition_noise=tfd.MultivariateNormalDiag(scale_diag=[0.1, 0.1]),
        observation_matrix=tf.linalg.LinearOperatorFullMatrix([[1.0, 0.0]]),
        observation_noise=tfd.MultivariateNormalDiag(scale_diag=[0.2]),
        initial_state_prior=tfd.MultivariateNormalDiag(
            loc=[0.0, 0.0], scale_diag=[1.0, 1.0]
        ),
    )
    lgssm_sample = lgssm.sample(seed=tfp.random.sanitize_seed(607, salt="lgssm"))
    if not bool(tf.math.is_finite(lgssm.log_prob(lgssm_sample))):
        raise RuntimeError("LinearGaussianStateSpaceModel 轻量 log_prob 非有限")

    observed = tf.linspace(10.0, 12.0, 12)
    sts_model = tfp.sts.Sum(
        [
            tfp.sts.LocalLevel(observed_time_series=observed, name="level"),
            tfp.sts.Seasonal(
                num_seasons=4,
                observed_time_series=observed,
                name="seasonal",
            ),
        ],
        observed_time_series=observed,
        name="runtime_sts",
    )
    sts_jd = sts_model.joint_distribution(observed_time_series=observed)
    sts_sample = sts_jd.sample_unpinned(
        seed=tfp.random.sanitize_seed(608, salt="sts")
    )
    sts_log_prob = sts_jd.unnormalized_log_prob(sts_sample)
    if not bool(tf.reduce_all(tf.math.is_finite(sts_log_prob))):
        raise RuntimeError("tfp.sts joint_distribution 轻量 log_prob 非有限")

    required_public = {
        "windowed_adaptive_nuts": tfp.experimental.mcmc.windowed_adaptive_nuts,
        "NoUTurnSampler": tfp.mcmc.NoUTurnSampler,
        "TransformedTransitionKernel": tfp.mcmc.TransformedTransitionKernel,
        "DualAveragingStepSizeAdaptation": tfp.mcmc.DualAveragingStepSizeAdaptation,
        "forecast": tfp.sts.forecast,
        "decompose_by_component": tfp.sts.decompose_by_component,
    }
    if not all(callable(value) for value in required_public.values()):
        raise RuntimeError("一个或多个所需公共 TFP API 不可调用")

    return {
        "versions": versions,
        "joint_distribution": "ok",
        "ar_scan": "ok",
        "sarima_public_loop": "ok",
        "lgssm": "ok",
        "structural_time_series": "ok",
        "public_api_count": len(required_public),
        "mcmc_executed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="验证源、哈希、清单并编译所有代码单元格")
    parser.add_argument(
        "--runtime-check",
        action="store_true",
        help="轻量导入并构造 TFP 模型；不运行 MCMC",
    )
    parser.add_argument("--write-notebook", type=Path, metavar="PATH", help="显式写出 Notebook 到指定路径")
    parser.add_argument("--write-org", type=Path, metavar="PATH", help="显式写出 Org 到指定路径")
    args = parser.parse_args(argv)

    requested = any((args.check, args.runtime_check, args.write_notebook, args.write_org))
    report: dict[str, Any] = {}
    if args.check or not requested or args.write_notebook or args.write_org:
        report["source"] = validate_source()
    if args.runtime_check:
        report["runtime"] = runtime_api_check()
    if args.write_notebook:
        nb_tools.write_ipynb(cells, args.write_notebook, kernelspec_name="python3")
        report["notebook"] = str(args.write_notebook.resolve())
    if args.write_org:
        nb_tools.write_org(cells, args.write_org, title="第六章：时间序列")
        report["org"] = str(args.write_org.resolve())

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
