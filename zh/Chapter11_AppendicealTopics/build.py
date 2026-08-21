#!/usr/bin/env python3
r"""第 11 章《附录主题》中文版的规范源文件。

完整性依据：``markdown/chp_11.md``。
代码迁移依据：``notebooks_updated/chp_11.ipynb``。
本文件只定义带稳定 ID 与来源元数据的单元格；生成的 Notebook/Org 不是源文件。
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parent
SOURCE_AUTHORITY = "markdown/chp_11.md"
CODE_MIGRATION_SOURCE = "notebooks_updated/chp_11.ipynb"
RANDOM_SEED = 14067

# 默认采用 smoke，避免直接执行源文件时意外触发发布级优化；发布构建应显式设置 release。
EXECUTION_PROFILE = os.environ.get("BMCP_EXECUTION_PROFILE", "smoke").lower()
if EXECUTION_PROFILE not in {"smoke", "release"}:
    raise ValueError("BMCP_EXECUTION_PROFILE 必须是 smoke 或 release")

if EXECUTION_PROFILE == "smoke":
    PIT_DRAWS = 2_000
    MONTE_CARLO_DRAWS = 5_000
    HIGHDIM_DRAWS = 5_000
    POSTERIOR_DRAWS = 400
    VI_STEPS = 60
    FLOW_VI_STEPS = 90
    VI_SAMPLE_SIZE = 8
    VI_DRAWS = 512
else:
    PIT_DRAWS = 100_000
    MONTE_CARLO_DRAWS = 100_000
    HIGHDIM_DRAWS = 100_000
    POSTERIOR_DRAWS = 2_000
    VI_STEPS = 500
    FLOW_VI_STEPS = 800
    VI_SAMPLE_SIZE = 32
    VI_DRAWS = 10_000

SOURCE_COMPLETENESS = {
    "anchors": 48,
    # 粗略按行首 # 计数为 51，其中 3 行其实位于 Python 代码块。
    "headings": 51,
    "rendered_headings": 48,
    # 119 个 math 围栏中有 2 个位于 HTML 注释内，是矩阵的替代排版。
    "display_math": 119,
    "active_display_math": 117,
    "commented_display_math": 2,
    "equation_labels": 110,
    "figures": 33,
    "authoritative_code_blocks": 13,
    "citation_occurrences": 34,
    "footnotes": 31,
    "exercises": 0,
}

ASSET_SHA256 = {
    "KL_heatmap.png": "48067b8cb338d3b2d105a39342f72d119cf3d951706333e8a94361526e9d9e9e",
    "beta_pdf_cdf.png": "70350bc0aa5e1ef923af6a108d06e0d709caa29fdf2b59cb752415fa586b959b",
    "binomial_pmf_cdf.png": "994cbed0263629fb9a9476a6384e7c7626a8ff129ccf331617ee21c1f024e71a",
    "central_limit.png": "04c63fab77ea706dcb360fedb1e8d59ac1cc915e891e96ba394cebc825aa724c",
    "cmf_pdf_pmf.png": "525774ff5369f9c02a3398abdc12c83cdd510142623ce561d638e2985eb5b900",
    "colin_joint_marginals.png": "bd0124b8abf91df3ef47fcf01d181d8057828442944ed676df1dbdfa42cc49e3",
    "cond.png": "eda4b50b2bf4693e68faa0aae5e332d2e5a12b1cc3adae02eb6568e5ca4c5349",
    "dice_distribution.png": "e7818107bd4aa8ea155f5bd70a32fed18ade2ebd66012a8d58c0dd2b761f502c",
    "discrete_uniform_pmf_cdf.png": "681d1daab10d84faa346619e2120b94d04bbdc1685c69f3b7bfdee1e4cfcef67",
    "distance_to_mode.png": "779ac0e884d33cdac463839da7e3cc52cc5b7dfebec5602b8f001f2c6302ce9d",
    "entropy.png": "9bc74ae782d38e434fb5b27ccbf4a06c6d84ddd0e91d264733c63f943330334b",
    "entropy_T.png": "efc70e8415166cda8f29f6eaa1cf2d65e9819f13fdb41c9db6f918631ae2ba5a",
    "funnel_leapfrog.png": "6660beed4e7ca2ef66e0a91bedd01d14b508ddaf2e71eaa68e6ad6b94e364c3c",
    "grid_method.png": "5d866e640a3c4374afca937ab419b0906e8cb21d1f20c8c969235b3da9bf9ade",
    "harmonic_mean_heatmap.png": "98be21c67ba544744cce41e3e38071e73ba23687bcaa641efd58085b76182d0c",
    "importance_sampling.png": "1e4f97a18093227fe2874e3fe48a5ab3a0badb53d674f44421453601e9f69445",
    "inside_out.png": "731471f600f4e36d1e20e33d93d6c579457730079fc5feb6cbe599a47f80f7cd",
    "joint_dist_conditional.png": "ad7cac9b20eb6f051efd1d61f71d3d92a2f08e78621a230f7f0b73855c83cce1",
    "joint_dist_marginal.png": "7bc977258df83ba16b9fee19e4b06d8de6b8139406c49cdb054ff4e683859388",
    "joint_marginal_cond_continuous.png": "5191201f848265fa2771f83a4f65aedb6f749f62a608a2f78a003adfc6e4ba05",
    "law_of_large_numbers.png": "a1efdb72a2a3b138e4e0af20ad88d5e1f5246429ab92ceaafba745fb27da5dea",
    "markov_chains_graph.png": "4c4bf8d5ed45166fb272d436ac62b9a7bb2db4bf5802205f8e987c0dd9b52f24",
    "mix_joint.png": "fa28ea53c2435d39354bdbd0c61ee618873edd8ab3c67b4c50d0ae923c706606",
    "ml_waic_loo.png": "9b9502640ff818c279c4a6698d23175c8bb148246284fc84f2a3e12fc8a3758a",
    "monte_carlo.png": "7c2406f0b5164cb992b3d8f3d24b9755bc868b6d3e06f1fad3b2b7c38af4e730",
    "normal_leapfrog.png": "caf2e1578b29ff07e51f0ed4cb1dc77cd8447b0efd4f4a30a1943b62e3a0c186",
    "normal_pdf_cdf.png": "397bf01732e8a697a201a5e772eb037623d59ea77cd37ccc76ee8e25dd3c5941",
    "pit.png": "019fee7ca71fe7d1b721181c8233906953c92e1d1f501b7f986ce67e03c174ba",
    "poisson_pmf_cdf.png": "7683b2ed705c66bf9d9f54c5d00c966befcff324549f22d561669aed3aa2b862",
    "student_t_pdf_cdf.png": "2b5ebb981b8ce1288e4c634ac87cf1d6e4651761802f694bd100b8313b8ed688",
    "sum_dice_distribution.png": "bd826b7d6c376666e7cc16dc408d65fb90995e26025791c379359f4496a80c04",
    "uniform_pdf_cdf.png": "12c4507b6f05de546a080b69869ff94136e9efdfe04a2c9a343a312ad7f30b22",
    "vi_in_tfp.png": "d6fdbe317f2ba6a62c8713ceffb679e8d94691ff253b10693ccafccffcaff7ab",
}


def validate_assets() -> None:
    """确认本章只携带并引用已登记的静态资源。"""
    asset_dir = HERE / "img"
    actual = {path.name for path in asset_dir.iterdir() if path.is_file()}
    expected = set(ASSET_SHA256)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(f"资源集合不匹配；缺少={missing}，多余={extra}")
    for name, expected_hash in ASSET_SHA256.items():
        digest = hashlib.sha256((asset_dir / name).read_bytes()).hexdigest()
        if digest != expected_hash:
            raise RuntimeError(f"资源哈希不匹配：{name}")


def _zh_metadata(
    *,
    kind: str,
    source_lines: str,
    source_anchor: str | None = None,
    labels: Iterable[str] = (),
    provenance: str = "translation",
    modernization: str | None = None,
) -> dict:
    label_list = list(labels)
    zh = {
        "language": "zh-CN",
        "authority": SOURCE_AUTHORITY,
        "source_lines": source_lines,
        "provenance": provenance,
        "labels": label_list,
    }
    direct_provenance = {
        "classification": provenance,
        "authority": SOURCE_AUTHORITY,
        "source_lines": source_lines,
        "labels": label_list,
    }
    if source_anchor is not None:
        zh["source_anchor"] = source_anchor
        direct_provenance["source_anchor"] = source_anchor
    if modernization is not None:
        zh["modernization"] = modernization
        direct_provenance["modernization"] = modernization
    return {
        "kind": kind,
        "provenance": direct_provenance,
        "zh": zh,
    }


cells: list[dict] = []


def add_md(
    cell_id: str,
    source: str,
    source_lines: str,
    *,
    anchor: str | None = None,
    labels: Iterable[str] = (),
    provenance: str = "translation",
    modernization: str | None = None,
) -> None:
    cells.append(
        {
            "type": "markdown",
            "id": cell_id,
            "source": source.strip(),
            "metadata": _zh_metadata(
                kind=(
                    "supplement"
                    if "addition" in provenance
                    else "translation-correction"
                    if "correction" in provenance
                    else "translation"
                ),
                source_lines=source_lines,
                source_anchor=anchor,
                labels=labels,
                provenance=provenance,
                modernization=modernization,
            ),
        }
    )


def add_code(
    cell_id: str,
    source: str,
    source_lines: str,
    *,
    labels: Iterable[str] = (),
    migration_cells: Iterable[int] = (),
    provenance: str = "code-migration",
    modernization: str | None = None,
    tags: Iterable[str] = (),
) -> None:
    metadata = _zh_metadata(
        kind=(
            "supplement-code"
            if "addition" in provenance
            else "modernized-code"
        ),
        source_lines=source_lines,
        labels=labels,
        provenance=provenance,
        modernization=modernization,
    )
    migration_cell_list = list(migration_cells)
    metadata["zh"]["migration_source"] = CODE_MIGRATION_SOURCE
    metadata["zh"]["migration_cells"] = migration_cell_list
    metadata["provenance"]["migration_source"] = CODE_MIGRATION_SOURCE
    metadata["provenance"]["migration_cells"] = migration_cell_list
    if tags:
        metadata["tags"] = list(tags)
    cells.append(
        {
            "type": "code",
            "id": cell_id,
            "source": source.strip(),
            "metadata": metadata,
        }
    )


add_md(
    "app",
    r"""
# 第 11 章　附录主题

本章与其他章节不同，它并不围绕某一个特定主题展开，而是汇集若干彼此不同、但能为全书其余内容提供支撑的专题，用来补充其他章节已经讨论过的方法与概念。对于希望进一步理解各类方法及其理论基础的读者，这些材料会很有帮助。相应地，本章的写作风格也会比其他章节更偏理论、更抽象一些。

> **中文版来源与现代化说明**：正文完整性以 `markdown/chp_11.md` 为准；代码以 `notebooks_updated/chp_11.ipynb` 为迁移输入。原文中的 PyMC3 统一更新为当前 PyMC；ArviZ 示例使用 `InferenceData` 与当前返回字段；TFP 变分推断与正规化流示例只使用公开 API。所有现代化补充都在单元格元数据中标明，未把新增内容伪装成原书正文。源文件没有习题，本章也不凭空增补习题。
""",
    "1-11",
    anchor="app",
    modernization="新增中文版来源、API 版本与零习题说明。",
)

add_code(
    "imports-and-profiles",
    r"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from scipy import special, stats

EXECUTION_PROFILE = os.environ.get("BMCP_EXECUTION_PROFILE", "smoke").lower()
if EXECUTION_PROFILE not in {"smoke", "release"}:
    raise ValueError("BMCP_EXECUTION_PROFILE 必须是 smoke 或 release")
RANDOM_SEED = 14067
if EXECUTION_PROFILE == "smoke":
    PIT_DRAWS, MONTE_CARLO_DRAWS, HIGHDIM_DRAWS = 2_000, 5_000, 5_000
    POSTERIOR_DRAWS, VI_STEPS, FLOW_VI_STEPS = 400, 60, 90
    VI_SAMPLE_SIZE, VI_DRAWS = 8, 512
    HMC_DEMO_SAMPLES = 20
else:
    PIT_DRAWS, MONTE_CARLO_DRAWS, HIGHDIM_DRAWS = 100_000, 100_000, 100_000
    POSTERIOR_DRAWS, VI_STEPS, FLOW_VI_STEPS = 2_000, 500, 800
    VI_SAMPLE_SIZE, VI_DRAWS = 32, 10_000
    HMC_DEMO_SAMPLES = 200

az.style.use("arviz-grayscale")
plt.rcParams["figure.dpi"] = 150 if EXECUTION_PROFILE == "smoke" else 300
rng = np.random.default_rng(RANDOM_SEED)
""",
    "notebook cells 1-3",
    migration_cells=(1, 2, 3),
    modernization="用 numpy Generator 取代全局随机状态；显式设置 TFP 所需的 legacy Keras 开关；导入当前 PyMC/ArviZ。",
)

add_md(
    "probability-background",
    r"""
## 概率论背景

西班牙语中的 *azahar*（某些柑橘类植物的花）与 *azar*（随机、偶然）之所以相似，并非纯属巧合：二者都源自阿拉伯语[^1]。古代乃至今天，一些机会游戏会使用一种有两个平面的骨头，形似硬币或两面骰。为了区分两面，至少一面会画上特殊记号；古代阿拉伯人常画一朵花。后来，西班牙语用 *azahar* 指某类花朵，用 *azar* 表示随机性。

概率论发展的动机之一，可以追溯到人们试图理解机会游戏——也许还想顺便发一笔小财。下面对概率论若干核心概念作简短介绍[^2]，先想象一枚六面骰。每次掷骰只可能得到 1 到 6 的某个整数，并且我们暂时假定各面没有偏好。用 Python 可以这样编写一个骰子：
""",
    "13-31",
    anchor="probability-background",
)

add_code(
    "die",
    r"""
def die(rng=rng):
    '''返回一次公平六面骰结果。'''
    return int(rng.choice(np.arange(1, 7)))

assert 1 <= die() <= 6
""",
    "33-40",
    labels=("die",),
    migration_cells=(6,),
    modernization="注入 numpy Generator，使随机结果可复现并避免依赖全局状态。",
)

add_md(
    "probability-background-data",
    r"""
假设我们怀疑这枚骰子有偏。怎样评估这种可能性？一种科学的做法是收集并分析数据。下面的代码模拟了数据收集过程。
""",
    "42-46",
)

add_code(
    "experiment",
    r"""
def experiment(N=10, rng=rng):
    '''掷骰 N 次并返回每个点数的经验频率。'''
    if N <= 0:
        raise ValueError("N 必须为正整数")
    sample = np.array([die(rng) for _ in range(N)], dtype=int)
    frequencies = np.bincount(sample, minlength=7)[1:] / N
    for face, frequency in enumerate(frequencies, start=1):
        print(f"{face}: {frequency:.2g}")
    return frequencies

frequencies = experiment()
assert np.isclose(frequencies.sum(), 1.0)
""",
    "47-58",
    labels=("experiment",),
    migration_cells=(8,),
    modernization="返回频率数组并加入概率和为 1 的语义断言；保留原示例的打印行为。",
)

add_md(
    "probability-background-observations",
    r"""
原书一次运行得到如下输出；具体数值会随随机样本而变化：

```text
1: 0
2: 0.1
3: 0.4
4: 0.1
5: 0.4
6: 0
```

第一列是所有可能结果，第二列是各点数出现的频率。频率等于该结果出现的次数除以总掷骰次数 `N`。

这个例子至少有两点值得注意。第一，多次执行 `experiment()` 会得到不同结果。这也正是骰子用于机会游戏的原因：每次掷出什么点数无法预先确定。第二，即使重复掷同一枚骰很多次，我们预测下一次单独结果的能力也不会提高；然而，收集和分析数据确实能帮助我们估计各结果的**频率列表**，而且 `N` 越大，估计通常越好。把 `N` 设为 10000，会看到各频率大约是 $0.17$，而 $0.17 \approx \frac{1}{6}$，正是六个点数机会相等时的预期。

这两点并不限于骰子和机会游戏。假如每天称体重，我们会得到不同数值，因为体重与摄入的食物和水、排泄次数、秤的精度、穿着以及许多其他因素有关。因此，一次测量未必能**代表**我们的体重。变化也许很小，小到不值得关心，但那是后话。此处最重要的是：数据的测量与收集伴随着不确定性。

统计学基本上研究如何在实际问题中处理不确定性，而概率论是统计学的理论支柱之一。概率论让我们能把刚才的讨论形式化，并从骰子推广到其他对象，从而更好地提出和回答有关预期结果的问题，例如实验次数增加后会怎样、哪个事件比另一个事件更可能发生，等等。
""",
    "60-103",
)

add_md(
    "probability",
    r"""
### 概率

概率是一种数学工具，使我们能够以有原则的方式量化不确定性。和其他数学对象与理论一样，概率可以完全从纯数学角度得到论证；但从实践角度看，它也会从实验、观察性数据收集乃至计算模拟中**自然地**产生。为简洁起见，下面统一称为“实验”，但要知道这里采用的是非常宽泛的含义。

可以用集合来理解概率。**样本空间** $\mathcal{X}$ 是一次**实验**所有可能结果的集合；**事件** $A$ 是 $\mathcal{X}$ 的一个子集。进行实验并得到属于 $A$ 的结果时，我们就说事件 $A$ 发生。对典型六面骰可写为：

$$
\mathcal{X} = \{1, 2, 3, 4, 5, 6\}
\tag{eq:sample_space_dice}
$$

事件 $A$ 可以是 $\mathcal{X}$ 的任意子集，例如“掷出偶数”可写为 $A=\{2,4,6\}$。我们可以给事件分配概率。事件 $A$ 的概率写成 $P(A=\{2,4,6\})$，或简写为 $P(A)$。概率函数 $P$ 以事件 $A\subseteq\mathcal{X}$ 为输入，返回 $P(A)$。

概率取值属于闭区间 $[0,1]$。事件永不发生时概率为 0，例如 $P(A=-1)=0$；事件必然发生时概率为 1，例如 $P(A=\{1,2,3,4,5,6\})=1$。如果两个事件不能同时发生，就称它们互斥。例如 $A_1$ 表示奇数、$A_2$ 表示偶数，则同时掷出 $A_1$ 与 $A_2$ 的概率为 0。如果 $A_1,A_2,\ldots,A_n$ 两两互斥并穷尽样本空间，则 $\sum_i^n P(A_i)=1$。任何满足这些性质的函数都是有效的概率函数。可以把概率看成一种分配在所有可能事件之间、非负且守恒的量[^3]。

概率有清楚的数学定义，但如何解释概率则存在不同学派。贝叶斯学派倾向于把概率解释为不确定程度。例如，对公平骰而言，掷出奇数的概率为 $0.5$，可理解为我们有一半把握会得到奇数。另一种解释是：无限次掷骰时，一半结果是奇数、另一半是偶数。这是频率学派的解释，也是理解概率的有用方式。若不想掷无限次，只需掷很多次，并说奇数大约占一半；这正是前面的 `experiment` 所做的。公平骰每个单独点数的概率都是 $\frac{1}{6}$，但非公平骰可以不同。等概率只是一个特例。

如果概率反映不确定性，那么我们自然会问：火星质量恰为 $6.39\times10^{23}$ kg 的概率是多少？赫尔辛基 5 月 1 日下雨的概率是多少？未来三十年资本主义被另一种社会经济制度取代的概率是多少？这种概率定义称为**认识论的**，因为它描述的不是“真实世界”（无论那究竟是什么）的属性，而是我们关于世界的知识状态。我们收集并分析数据，是因为相信外部信息能够更新内部知识状态。

真实世界中会发生什么，由实验的全部细节决定，包括我们不能控制或根本没有意识到的细节。相反，样本空间是由我们显式或隐式定义的数学对象。例如，把骰子的样本空间定义为上式，就排除了骰子立在棱上的可能性；在不平整表面掷骰时，这其实可能发生。某些结果可能被有意排除，例如规定不断重掷，直到得到 $\{1,2,3,4,5,6\}$ 中的整数；也可能因疏忽被排除。例如问卷询问性别，却只给“女性”和“男性”两个选项，会迫使一些人选择不合适的答案，或者干脆放弃填写后续问卷。包含所有数学概念的理念世界与真实世界并不相同；统计建模不断在这两个世界之间来回切换。
""",
    "104-195",
    anchor="probability",
    labels=("eq:sample_space_dice",),
)

add_md(
    "conditional_probability",
    r"""
### 条件概率

给定两个事件 $A$ 与 $B$，且 $P(B)>0$，在 $B$ 条件下 $A$ 的概率写作 $P(A\mid B)$，定义为：

$$
P(A \mid B) = \frac{P(A, B)}{P(B)}
\tag{eq:conditional_probability}
$$

$P(A,B)$ 表示事件 $A$ 与 $B$ 同时发生的概率，也常写成 $P(A\cap B)$；符号 $\cap$ 表示集合的交集。

$P(A\mid B)$ 称为条件概率：在已知（或假设、设想、提出假说认为）$B$ 已发生的前提下，事件 $A$ 发生的概率。例如，人行道湿润的概率，与已知正在下雨时人行道湿润的概率并不相同。

条件概率可以看作对样本空间的缩减或限制。下图左侧是在样本空间 $\mathcal{X}$ 中的事件 $A$ 与 $B$；右侧则把 $B$ 重新当作样本空间，只留下与 $B$ 相容的那部分 $A$。说“$B$ 已发生”不一定是在谈过去，它只是“已经以 $B$ 为条件”或“已经把样本空间限制到与证据 $B$ 一致”的通俗表达。

![条件化就是重新定义样本空间。左图每个圆点代表样本空间中的一种可能结果，并标出事件 A 与 B；右图表示 P(A|B)：知道 B 后可排除所有不属于 B 的结果。本图改编自 Introduction to Probability {cite:p}`blitzstein_2019`。](img/cond.png)

条件概率是统计学的核心，也是思考“面对新数据，应当怎样更新对事件的认识”的基础。所有概率都以某些假设或模型为条件；即便没有明确写出来，也不存在脱离上下文的概率。
""",
    "197-243",
    anchor="conditional_probability",
    labels=("eq:conditional_probability", "fig:cond"),
)

add_md(
    "probability-distribution",
    r"""
### 概率分布

与其只计算一次掷骰得到 5 的概率，我们也许更想知道骰子所有点数的**概率列表**。得到这个列表后，可以展示它，也可以用它计算其他量，例如得到 5 的概率，或得到不小于 5 的点数的概率。这个“列表”的正式名称就是**概率分布**。

前面的 `experiment` 给出了骰子的经验概率分布，即由数据计算出来的分布。除此之外还有理论分布；它们在统计学中极为重要，其中一个原因是可以用来构造概率模型。

理论概率分布具有精确的数学公式，正如圆有精确的数学定义：圆是平面上与一个称为圆心的点等距的所有点。给定半径这一参数，圆就被完全确定[^4]。也可以说，不是只有一个圆，而是有一个圆族；各成员只在半径参数上不同，一旦半径确定，具体的圆也随之确定。

类似地，概率分布也组成分布族，族中成员由一个或多个参数完全确定。参数常用希腊字母表示，但并非总是如此。下图展示了一个可用来表示不均匀骰子的离散分布族。两个参数 $\alpha$ 与 $\beta$ 控制分布：改变它们就会改变分布的**形状**，可以使分布平坦、偏向一侧、把概率质量推向两端，或集中在中间。正如圆的半径必须为正，分布参数也有约束；这里 $\alpha$ 与 $\beta$ 都必须为正。

![由参数 α、β 控制的离散分布族中的四个成员。柱高表示每个 x 的概率；未画出的 x 位于分布支撑集之外，概率为 0。](img/dice_distribution.png)
""",
    "245-292",
    anchor="probability-distribution",
    labels=("fig:dice_distribution",),
)

add_md(
    "discrete-random-variables",
    r"""
### 离散随机变量与分布

随机变量是把样本空间映射到实数 $\mathbb{R}$ 的函数。继续以骰子为例，如果关心的是点数，映射非常直接：把 $\LARGE\unicode{x2680}$ 映射到 1，把 $\LARGE\unicode{x2681}$ 映射到 2，依此类推。若同时掷两枚骰，可以定义随机变量 $S$ 为两枚骰结果之和。于是 $S$ 的取值域为 $\{2,3,4,5,6,7,8,9,10,11,12\}$；若两枚骰都公平，其概率分布如下图所示。

![若样本空间是两枚骰的所有结果，随机变量 S 是点数之和，则 S 是离散随机变量；每个结果的概率由柱高表示。本图改编自 Wikimedia Commons 的 Dice Distribution 图。](img/sum_dice_distribution.png)

也可以定义随机变量 $C$，其样本空间为 $\{\text{红},\text{绿},\text{蓝}\}$，并映射到 $\mathbb{R}$：

$$
\begin{aligned}
C(\text{红}) &= 0 \\
C(\text{绿}) &= 1 \\
C(\text{蓝}) &= 2
\end{aligned}
$$

这种编码很有用，因为无论用纸笔做模拟计算，还是用计算机做数字计算，处理数字通常都比处理字符串容易。

随机变量既然是函数，而样本空间到 $\mathbb{R}$ 的映射又是确定性的，“随机”究竟来自哪里并不显然。这里所谓随机，是指进行实验、向变量“索取”一个值时，会在没有确定性模式的情况下得到不同结果。例如连续三次询问 $C$ 的值，可能得到红、红、蓝，也可能得到蓝、绿、蓝，等等。

若存在有限列表 $a_1,\ldots,a_n$ 或无限可数列表 $a_1,a_2,\ldots$，使总概率 $\sum_jP(X=a_j)=1$，就称随机变量 $X$ 是离散的。对离散随机变量 $X$，所有满足 $P(X=x)>0$ 的有限或可数无限个 $x$ 构成 $X$ 的**支撑集**。

概率分布可看作把概率与每个事件对应起来的列表，而随机变量也有与之关联的概率分布。对离散随机变量，这个分布又称**概率质量函数**（PMF）。PMF 是返回概率的函数；$X$ 的 PMF 是定义在 $x\in\mathbb{R}$ 上的 $P(X=x)$。有效 PMF 必须非负且总和为 1。

“随机”并不意味着任何值都允许出现，只能出现样本空间中的值。例如 $C$ 不可能得到橙色，$S$ 不可能得到 13。另一个常见误解是“随机”意味着等概率；其实每个事件的概率由 PMF 给出。例如可以有
$P(C=\text{红})=\frac12$、$P(C=\text{绿})=\frac14$、$P(C=\text{蓝})=\frac14$。等概率仍然只是特例。

还可以用**累积分布函数**（CDF）定义离散随机变量。随机变量 $X$ 的 CDF 是 $F_X(x)=P(X\le x)$。有效 CDF 必须单调不减[^5]、右连续[^6]，在 $x\to-\infty$ 时趋于 0，在 $x\to\infty$ 时趋于 1。

原则上没有什么阻止我们自定义概率分布，但已经有许多分布极其常用，因而拥有专门名称。熟悉它们很有价值：本书大多数模型都是预定义概率分布的组合，只有少数使用自定义分布。例如第 8 章 ABC 移动平均示例用均匀分布和两个势函数定义了二维三角分布。

后面的离散均匀、二项和泊松分布图同时展示 PMF 与 CDF。左侧 PMF 中，柱或竖线高度表示每个 $x$ 的概率；右侧 CDF 中，在某个 $x$ 处两条水平线之间的“跳跃”高度表示该 $x$ 的概率。图中还列出了分布的均值和标准差。必须强调：它们是理论分布的性质，就像圆周长一样，不是由有限样本计算出来的量；详见“期望”一节。

还可以用故事描述随机变量。关于 $X$ 的故事，是一个能够产生与 $X$ 同分布随机变量的实验。故事不是形式化工具，却仍然有用。人类数千年来一直借助故事理解周围世界；今天即使在统计学中也不例外。Blitzstein 与 Hwang 的 *Introduction to Probability* {cite:p}`blitzstein_2019` 大量使用这种方法，甚至广泛采用比形式证明更直观的“故事证明”。故事也有助于创建统计模型：先思考数据可能怎样生成，再把过程写成统计记号或代码。本书第 9 章航班延误示例就是这样做的。
""",
    "294-415",
    anchor="discrete-random-variables-and-distributions",
    labels=("fig:sum_dice_distribution",),
)

add_md(
    "discrete-uniform-distribution",
    r"""
#### 离散均匀分布

离散均匀分布把相同概率分配给闭区间 $[a,b]$ 中有限个连续整数。其 PMF 为：

$$
P(X=x)=\frac{1}{b-a+1}=\frac{1}{n}
\tag{eq:pmf_uniform}
$$

当 $x\in[a,b]$ 时使用上式，否则 $P(X=x)=0$；其中 $n=b-a+1$ 是 $x$ 可取值的总数。

例如，可以用这个分布建模公平骰。下面用 SciPy 定义分布，并计算 PMF、CDF 与矩。
""",
    "417-437",
    anchor="discrete-uniform-distribution",
    labels=("eq:pmf_uniform",),
)

add_code(
    "scipy-unif",
    r"""
a = 1
b = 6
rv = stats.randint(a, b + 1)
x = np.arange(a, b + 1)

x_pmf = rv.pmf(x)  # 在 x 上计算 PMF
x_cdf = rv.cdf(x)  # 在 x 上计算 CDF
mean, variance = rv.stats(moments="mv")

assert np.isclose(x_pmf.sum(), 1.0)
assert np.all(np.diff(x_cdf) >= 0)
assert np.isclose(x_cdf[-1], 1.0)
""",
    "438-450",
    labels=("scipy_unif",),
    migration_cells=(9,),
    modernization="保留当前 SciPy 参数化并加入 PMF/CDF 有效性断言。",
)

add_md(
    "discrete-uniform-figure",
    r"""
配合少量 Matplotlib 代码即可得到下图。左图是 PMF，点和虚线强调了离散性；右图是 CDF，每个 $x$ 处的跳跃高度就是该点概率。

![参数为 (1,6) 的离散均匀分布。左：PMF，竖线高度表示各 x 的概率。右：CDF，每次跳跃的高度表示该点概率；实心点表示包含该 x 处的 CDF 值，空心点表示不包含。支撑集之外的值未画出。](img/discrete_uniform_pmf_cdf.png)

这个例子定义在 $[1,6]$ 上，因此小于 1 或大于 6 的值概率为 0。因为是均匀分布，六个点的高度都等于 $\frac16$。参数 $a$、$b$ 分别是下界与上界。

改变分布参数，就会改变分布的具体形状。例如把 `stats.randint(1, 7)` 换成 `stats.randint(1, 4)`。因此我们通常谈论一个**分布族**：每个成员对应一组特定且有效的参数。只要 $a<b$ 且二者都是整数，上面的公式就定义了离散均匀分布族。

在应用概率分布构建统计模型时，通常会把参数与具有物理意义的量联系起来。例如六面骰取 $a=1,b=6$ 很合理。在概率论问题中，参数值往往已知；在统计学问题中，参数通常未知，需要借助数据推断。
""",
    "452-493",
    labels=("fig:discrete_uniform_pmf_cdf",),
)

add_md(
    "binomial-distribution",
    r"""
#### 二项分布

伯努利试验是只有两种可能结果的实验：是/否、成功/失败、快乐/悲伤、患病/健康，等等。假设独立地[^7]进行 $n$ 次伯努利试验，每次成功概率均为 $p$，并令 $X$ 为成功次数，则 $X$ 服从参数为 $n,p$ 的二项分布。其中 $n$ 是正整数，$p\in[0,1]$。记作 $X\sim\operatorname{Bin}(n,p)$，其 PMF 为：

$$
P(X=x)=\frac{n!}{x!(n-x)!}p^x(1-p)^{n-x}
\tag{eq:binomial_pmf}
$$

$p^x(1-p)^{n-x}$ 对应 $n$ 次试验中有 $x$ 次成功的某一类结果，只计成功总数而不区分精确顺序，例如 $(0,1)$ 与 $(1,0)$ 都是在两次试验中成功一次。前面的系数称为**二项系数**，计算从 $n$ 个元素中取 $x$ 个元素的所有组合数。

书写二项 PMF 时经常省略返回 0 的取值，也就是支撑集之外的值。但为了避免错误，必须明确随机变量的支撑集。检查 PMF 是否有效是一项好习惯；若提出的是新 PMF，而不是直接使用现成分布，这一点尤其重要。

当 $n=1$ 时，二项分布也称伯努利分布。许多分布是其他分布的特例，或能通过某种方式从其他分布得到。

![Bin(n=4,p=0.5)。左：PMF，竖线高度表示各 x 的概率；右：CDF，跳跃高度表示相应概率。支撑集之外的值未画出。](img/binomial_pmf_cdf.png)
""",
    "495-540",
    anchor="binomial-distribution",
    labels=("eq:binomial_pmf", "fig:binomial_pmf_cdf"),
)

add_md(
    "poisson-distribution",
    r"""
#### 泊松分布

若若干事件以平均速率 $\mu$ 相互独立地发生，泊松分布给出固定时间区间（或空间区间）内发生 $x$ 次事件的概率。它通常用于试验次数很多、而单次成功概率很小的场景，例如：

- 放射性衰变：一块材料中的原子数极其庞大，而真正发生核裂变的原子数相对很少。
- 一座城市每天的交通事故数：即使这个数字高得令人不快，相对于驾驶者每天完成的转弯、停车、等信号灯等大量独立操作，事故仍属于低概率事件。

泊松分布的 PMF 定义为：

$$
P(X=x)=\frac{\mu^xe^{-\mu}}{x!},\qquad x=0,1,2,\ldots
\tag{eq:poisson_pmf}
$$

这个 PMF 的支撑集是全部非负整数，是无限集合，因此“概率列表”的类比需要谨慎：无限级数的求和可能很微妙。上式确实是有效 PMF，因为泰勒级数 $\sum_{x=0}^{\infty}\frac{\mu^x}{x!}=e^\mu$。

泊松分布的均值与方差都等于 $\mu$。随着 $\mu$ 增大，泊松分布会近似正态分布，尽管前者离散、后者连续。泊松分布也与二项分布紧密相关：在 $n$ 很大而 $p$ 很小、并保持 $np$ 不变的极限下[^8]，有 $\operatorname{Pois}(\mu=np)\approx\operatorname{Bin}(n,p)$。因此泊松分布也称“小数定律”或“稀有事件定律”。这并不要求 $\mu$ 本身很小，而是要求 $p$ 相对于 $n$ 很小。

![Pois(2.3)。左：PMF，竖线高度表示各 x 的概率；右：CDF，跳跃高度表示相应概率。支撑集之外的值未画出。](img/poisson_pmf_cdf.png)
""",
    "542-596",
    anchor="poisson-distribution",
    labels=("eq:poisson_pmf", "fig:poisson_pmf_cdf"),
)

add_md(
    "continuous-random-variables",
    r"""
### 连续随机变量与分布

到目前为止，我们讨论的都是离散随机变量。另一类广泛使用的随机变量叫作连续随机变量，其支撑集可以取 $\mathbb{R}$ 中的值。离散与连续随机变量最重要的区别是：连续随机变量可在一个区间内取任意 $x$，但任何单点 $x$ 的概率都恰好为 0。这样介绍也许会让人觉得连续分布毫无用处；其实问题在于，把概率分布当成有限列表的类比非常有限，遇到连续随机变量就彻底失效[^9]。

表示离散变量的 PMF 时，我们用竖线高度表示每个事件的概率，所有高度相加总是 1。连续分布没有一根根竖线，而是一条连续曲线。曲线高度不是概率，而是**概率密度**；相应地，我们使用概率密度函数（PDF），而不是 PMF。$\operatorname{PDF}(x)$ 的高度甚至可以大于 1，因为它表示密度而非概率。要从 PDF 得到概率，必须在某个区间上积分：

$$
P(a<X<b)=\int_a^b pdf(x)\,dx
\tag{eq:pdf_to_prob}
$$

因此，PDF 曲线下的面积——不是像 PMF 那样的高度——才给出概率。对整个支撑集积分得到的总面积必须为 1。若只想比较 $x_1$ 与 $x_2$ 哪个更可能，可以计算密度比 $pdf(x_1)/pdf(x_2)$。

包括本书在内的许多文本，会笼统地用符号 $p$ 表示 PMF 或 PDF。这是为了保持一般性，也为了避免在上下文已经足够清楚时让严苛记号成为负担。

离散随机变量的 CDF 在支撑集每一点发生跳跃，其他地方保持平坦，因此使用起来有些别扭：其导数在跳点处无定义，在其他地方又等于 0。对哈密顿蒙特卡洛等基于梯度的采样方法而言，这是一个问题。相反，连续随机变量的 CDF 往往很方便，其导数正是 PDF。

下图总结了 CDF、PDF 与 PMF 的关系。离散 CDF 与 PMF 之间、连续 CDF 与 PDF 之间的变换有严格定义，所以用实线箭头表示；离散与连续变量之间的变换则更多是数值近似。离散到连续通常用平滑方法，例如以连续分布近似离散分布；连续到离散则可通过离散化或分箱。比如 $\mu$ 较大的泊松分布尽管仍是离散的，却近似高斯分布[^10]，某些实际场景下二者可以互换。

在 ArviZ 中，可对离散数据调用 `az.plot_kde` 近似连续函数，效果取决于数据与平滑设置；对较大 $\mu$ 的泊松分布通常看起来不错。对离散变量调用当前 API `az.plot_bpv(..., kind="u_value")` 时，ArviZ 会通过随机化 PIT 的方式处理离散性，因为普通概率积分变换只直接适用于连续变量。

![CDF、PDF 与 PMF 的关系。改编自 Think Stats {cite:p}`Downey2014`。](img/cmf_pdf_pmf.png)

下面像讨论离散随机变量那样，考察几种常见连续随机变量及其 PDF 和 CDF。
""",
    "598-684",
    anchor="cont_rvs",
    labels=("eq:pdf_to_prob", "fig:cmf_pdf_pmf"),
    modernization="把原文拼写错误 az.plot_pbv 更新为当前 ArviZ 的 az.plot_bpv，并说明离散 PIT 的处理。",
)

add_md(
    "continuous-uniform-distribution",
    r"""
#### 连续均匀分布

若连续随机变量在区间 $(a,b)$ 上服从均匀分布，其 PDF 为：

$$
p(x\mid a,b)=
\begin{cases}
\frac{1}{b-a}, & a\le x\le b,\\
0, & \text{其他情况}。
\end{cases}
\tag{eq:uniform_pdf}
$$

![U(0,1)。左：PDF，黑线表示概率密度，灰色区域表示 P(0.25<X<0.75)=0.5；右：CDF，灰色连续线段的高度差表示同一概率。支撑集之外的值未画出。](img/uniform_pdf_cdf.png)

统计学中最常用的是 $\mathcal{U}(0,1)$，又称标准均匀分布。它的 PDF 与 CDF 分别是 $p(x)=1$ 和 $F_X(x)=x$。上图同时展示了二者，也说明怎样从 PDF 与 CDF 计算概率。
""",
    "685-713",
    anchor="continuous-uniform-distribution",
    labels=("eq:uniform_pdf", "fig:uniform_pdf_cdf"),
)

add_md(
    "gaussian-normal-distribution",
    r"""
#### 高斯分布或正态分布

这也许是最著名的分布[^11]。一方面，许多现象都能用它近似描述，这与后文的中心极限定理有关；另一方面，它具有若干便于解析计算的数学性质。

高斯分布由均值 $\mu$ 与标准差 $\sigma$ 两个参数定义：

$$
p(x\mid\mu,\sigma)=\frac{1}{\sigma\sqrt{2\pi}}
\exp\left[-\frac{(x-\mu)^2}{2\sigma^2}\right]
\tag{eq:gaussian_pdf}
$$

$\mu=0,\sigma=1$ 的高斯分布称为**标准高斯分布**。下图左侧为 PDF，右侧为 CDF。图只画出区间 $[-4,4]$，但高斯分布的支撑集是整条实数轴。

![N(0,1) 的表示。左为 PDF，右为 CDF；支撑集为整条实数轴。](img/normal_pdf_cdf.png)
""",
    "714-748",
    anchor="gaussian-or-normal-distribution",
    labels=("eq:gaussian_pdf", "fig:normal_pdf_cdf"),
)

add_md(
    "students-t-distribution",
    r"""
#### Student t 分布

历史上，这个分布源于在样本量较小时估计正态总体均值的问题[^12]。在贝叶斯统计中，一个常见用途是构建对异常数据更稳健的模型，正如“稳健回归”一节所讨论的那样。

$$
p(x\mid\nu,\mu,\sigma)=
\frac{\Gamma\left(\frac{\nu+1}{2}\right)}
{\Gamma\left(\frac{\nu}{2}\right)\sqrt{\pi\nu}\,\sigma}
\left[1+\frac{1}{\nu}\left(\frac{x-\mu}{\sigma}\right)^2\right]^{-\frac{\nu+1}{2}}
\tag{eq:student_t_pdf}
$$

其中 $\Gamma$ 是 Gamma 函数[^13]，$\nu$ 通常称为自由度。也可以把它叫作“正态程度”：$\nu$ 越大，t 分布越接近高斯分布；当 $\nu\to\infty$ 时，它恰好收敛到均值相同、标准差等于 $\sigma$ 的高斯分布。

当 $\nu=1$ 时得到柯西分布[^14]。它形似高斯分布，但尾部衰减极慢，慢到均值和方差都没有定义。我们当然能从一组数据算出样本均值；但若数据来自柯西分布，均值周围的离散程度会很大，而且不会随着样本量增加而缩小。原因在于，柯西等分布由尾部行为主导，这与高斯分布等情形不同。

在 t 分布中，$\sigma$ 是尺度而不一定是标准差——标准差甚至可能不存在。随着 $\nu$ 增大，这个尺度会趋近相应高斯分布的标准差。

下图左侧为 PDF、右侧为 CDF。把 $\mathcal{T}(\nu=4,\mu=0,\sigma=1)$ 与标准正态分布比较，可以看到 t 分布的尾部更重。

![T(ν=4,μ=0,σ=1)。左为 PDF，右为 CDF；支撑集为整条实数轴。](img/student_t_pdf_cdf.png)
""",
    "749-798",
    anchor="students-t-distribution",
    labels=("eq:student_t_pdf", "fig:student_t_pdf_cdf"),
)

add_md(
    "beta-distribution",
    r"""
#### Beta 分布

Beta 分布定义在区间 $[0,1]$ 上，可用于描述被限制在有限区间内的随机变量，例如比例或百分比。

$$
p(x\mid\alpha,\beta)=
\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}
 x^{\alpha-1}(1-x)^{\beta-1}
\tag{eq:beta_pdf}
$$

第一项是归一化常数，保证 PDF 的积分为 1；$\Gamma$ 是 Gamma 函数。当 $\alpha=1,\beta=1$ 时，Beta 分布退化为标准均匀分布。下图展示 $\operatorname{Beta}(\alpha=5,\beta=2)$。

![Beta(α=5,β=2)。左为 PDF，右为 CDF；支撑集为 0 至 1。](img/beta_pdf_cdf.png)

若想用均值与围绕均值的离散程度来参数化 Beta 分布，可以令 $\alpha=\mu\kappa$、$\beta=(1-\mu)\kappa$。其中 $\mu$ 是均值，$\kappa$ 称为集中度；$\kappa$ 越大，离散程度越小，并且 $\kappa=\alpha+\beta$。
""",
    "799-830",
    anchor="beta-distribution",
    labels=("eq:beta_pdf", "fig:beta_pdf_cdf"),
)

add_md(
    "joint-conditional-marginal",
    r"""
### 联合分布、条件分布与边缘分布

假设随机变量 $X,Y$ 都有 PMF $\operatorname{Bin}(1,0.5)$，它们相互依赖还是独立？若 $X$ 表示第一次抛硬币是否为正面，$Y$ 表示另一次抛硬币是否为正面，它们独立；若二者分别表示同一次抛硬币的正面与反面，它们就相互依赖。因此，一元 PMF/PDF 虽能完全刻画单个随机变量，却不包含它与其他变量之间关系的信息。回答这种问题需要知道**联合分布**，也称多元分布。

如果 $p(X)$ 提供关于 $X$ 在实数轴上位置概率的全部信息，那么 $p(X,Y)$——$X,Y$ 的联合分布——就提供二元组 $(X,Y)$ 在平面上位置概率的全部信息。联合分布描述同一实验产生的多个随机变量。例如，把模型以观测数据为条件后，后验分布就是所有参数的联合分布。

联合 PMF 为

$$
p_{X,Y}(x,y)=P(X=x,Y=y)
\tag{eq:joint_pmf}
$$

$n$ 个离散随机变量的定义类似，只需包含 $n$ 项。和一元 PMF 一样，有效联合 PMF 必须非负，并在所有可能取值上求和为 1：

$$
\sum_x\sum_yP(X=x,Y=y)=1
\tag{eq:joint_pmf_sum1}
$$

$X,Y$ 的联合 CDF 为

$$
F_{X,Y}(x,y)=P(X\le x,Y\le y)
\tag{eq:joint_cdf}
$$

给定联合分布，可通过对 $Y$ 的所有可能值求和得到 $X$ 的分布：

$$
P(X=x)=\sum_yP(X=x,Y=y)
\tag{eq:joint_to_marginal}
$$

![黑线表示 x、y 的联合分布；沿 y 轴对每个 x 的竖线高度求和，就得到蓝色的 x 边缘分布。](img/joint_dist_marginal.png)

此前把 $P(X=x)$ 称为 $X$ 的 PMF 或分布；讨论联合分布时，常称它为 $X$ 的**边缘分布**，强调只谈单独的 $X$ 而不再涉及 $Y$。对 $Y$ 所有取值求和，相当于“消去 $Y$”，正式名称是**边缘化掉 $Y$**。类似地，对 $X$ 求和可得到 $Y$ 的 PMF。若联合分布含两个以上变量，就对“其他所有变量”求和。

从联合分布得到边缘分布很直接；反过来通常不可能，除非再引入假设。上图中，沿轴相加只有一种方式，而逆向操作需要把柱子拆分，拆法却有无穷多种。

前文已经介绍条件分布：条件化就是重新定义样本空间。下图把这一思想放到 $X,Y$ 的联合分布中。要以 $Y=y$ 为条件，就取联合分布在该 $y$ 值上的切片，忽略所有 $Y\ne y$ 的部分。这类似于索引二维数组并选取某一行或列。剩下的 $X$ 概率必须重新归一化，使其和为 1，也就是除以 $P(Y=y)$。

![左：x、y 的联合分布，蓝线表示条件分布 p(x|y=3)；右：把该条件分布单独画出。每个 y 都对应一个 x 的条件 PMF，反之亦然；图中只突出一种情况。](img/joint_dist_conditional.png)

连续联合 CDF 的定义与离散情形相同，联合 PDF 则是 CDF 对 $x,y$ 的导数。有效联合 PDF 必须非负且积分为 1。连续变量的边缘化与离散变量类似，只是把求和换成积分：

$$
pdf_X(x)=\int pdf_{X,Y}(x,y)\,dy
\tag{eq:marginal_pdf}
$$

![中央用灰度表示联合密度 p(x,y)，颜色越深密度越高；上边与右边分别是边缘分布 p(x)、p(y)。虚线表示三个不同 y 值下的条件分布 p(x|y)，可把它们理解为联合密度在固定 y 处经过重新归一化的切片。](img/joint_marginal_cond_continuous.png)

下一幅图再次展示联合分布及其边缘分布，也清楚说明从联合到边缘有唯一方式，而从边缘到联合若无额外假设则不可能。联合分布还可以混合离散与连续变量，随后一图就是例子。

![把 PyMC3 标志视作一个联合分布的样本，并画出边缘分布。图由 imcmc 生成：<https://github.com/ColCarroll/imcmc>。这里保留原图历史名称；当前项目名称为 PyMC。](img/colin_joint_marginals.png)

![黑色表示一个混合型联合分布，蓝色表示边缘分布：X 服从高斯分布，Y 服从泊松分布。对 Y 的每个取值，都有一个高斯条件分布。](img/mix_joint.png)
""",
    "831-985",
    anchor="joint-conditional-and-marginal-distributions",
    labels=(
        "eq:joint_pmf", "eq:joint_pmf_sum1", "eq:joint_cdf",
        "eq:joint_to_marginal", "eq:marginal_pdf", "fig:joint_dist_marginal",
        "fig:joint_dist_conditional", "fig:joint_marginal_cond_continuous",
        "fig:colin_joint_marginals", "fig:mix_joint",
    ),
    modernization="图注保留原图的 PyMC3 历史来源，同时注明当前名称 PyMC。",
)

add_md(
    "probability-integral-transform",
    r"""
### 概率积分变换（PIT）

概率积分变换（PIT），又称均匀分布的普适性，指出：若随机变量 $X$ 服从连续分布，累积分布函数为 $F_X$，则

$$
Y=F_X(X)
\tag{eq:pit}
$$

服从标准均匀分布。证明如下。按 $Y$ 的 CDF 定义，

$$
F_Y(y)=P(Y\le y)
\tag{eq:pit1}
$$

代入 $Y=F_X(X)$：

$$
P(F_X(X)\le y)
\tag{eq:pit2}
$$

在不等式两边应用 $F_X$ 的逆函数：

$$
P(X\le F_X^{-1}(y))
\tag{eq:pit3}
$$

再由 CDF 定义得到

$$
F_X(F_X^{-1}(y))
\tag{eq:pit4}
$$

化简后就是标准均匀分布 $\mathcal{U}(0,1)$ 的 CDF：

$$
F_Y(y)=y
\tag{eq:pit5}
$$

若不知道 $F_X$，但有来自 $X$ 的样本，可以用经验 CDF 近似。下面的代码演示这一性质。
""",
    "986-1039",
    anchor="probability-integral-transform-pit",
    labels=("eq:pit", "eq:pit1", "eq:pit2", "eq:pit3", "eq:pit4", "eq:pit5"),
)

add_code(
    "pit",
    r"""
xs = (
    np.linspace(0, 20, 200),
    np.linspace(0, 1, 200),
    np.linspace(-4, 4, 200),
)
dists = (stats.expon(scale=5), stats.beta(0.5, 0.5), stats.norm(0, 1))

fig, axes = plt.subplots(3, 3, figsize=(9, 8), constrained_layout=True)
pit_samples = []
for idx, (dist, x_grid) in enumerate(zip(dists, xs)):
    draws = dist.rvs(size=PIT_DRAWS, random_state=rng)
    transformed = dist.cdf(draws)
    pit_samples.append(transformed)
    axes[idx, 0].plot(x_grid, dist.pdf(x_grid), color="black", linewidth=2)
    axes[idx, 1].plot(
        np.sort(transformed),
        np.linspace(0, 1, len(transformed)),
        color="black",
        linewidth=2,
    )
    az.plot_kde(transformed, ax=axes[idx, 2], plot_kwargs={"color": "black"})

for transformed in pit_samples:
    assert 0.45 < transformed.mean() < 0.55
    assert 0.06 < transformed.var() < 0.11
plt.close(fig)
""",
    "1040-1059",
    labels=("pit",),
    migration_cells=(11,),
    modernization="使用 profile 控制样本量、numpy Generator 与有界均匀矩断言；避免固定 100000 次烟雾测试。",
)

add_md(
    "pit-uses",
    r"""
![第一列是三种不同分布的 PDF；中间列先从相应分布抽样，再计算这些样本的 CDF，得到标准均匀分布的经验 CDF；最后一列以核密度估计近似同一均匀 PDF。](img/pit.png)

PIT 可用于检验给定数据集是否可视为来自某个指定分布或概率模型。本书中的 `az.plot_loo_pit()`，以及贝叶斯 p 值图等校准工具，都运用了相关思想。

PIT 也可用于从分布抽样。若 $X\sim\mathcal{U}(0,1)$，则 $Y=F^{-1}(X)$ 服从分布 $F$。因此，只要有像 `rng.random()` 这样的伪随机数生成器和目标分布的逆 CDF，就能生成目标样本。它未必总是最高效，却很难在一般性与简洁性上被超越。
""",
    "1061-1087",
    labels=("fig:pit",),
    modernization="把全局 np.random.rand 表述更新为 Generator.random，并避免绑定已变化的 ArviZ 次级参数名。",
)

add_md(
    "expectations",
    r"""
### 期望

期望是概括分布质量中心的单个数。若 $X$ 是离散随机变量，其期望为：

$$
\mathbb{E}(X)=\sum_x xP(X=x)
\tag{eq:expectation}
$$

统计分析还常需要描述分布的扩散或离散程度，例如表达点估计周围的不确定性。方差可完成这一任务，而方差本身也是一种期望：

$$
\mathbb{V}(X)=\mathbb{E}\left[(X-\mathbb{E}X)^2\right]
=\mathbb{E}(X^2)-(\mathbb{E}X)^2
\tag{eq:var_as_expectation}
$$

方差会在许多计算中自然出现；但报告结果时常取其平方根，也就是标准差，因为标准差与随机变量使用相同单位。

前面离散均匀、二项、泊松、连续均匀、正态、Student t 与 Beta 分布的图都给出了期望和标准差。请注意，这些不是由样本计算的数，而是理论数学对象的性质。

期望具有线性：

$$
\mathbb{E}(cX)=c\mathbb{E}(X)
\tag{eq:expectation_linear}
$$

其中 $c$ 是常数，并且

$$
\mathbb{E}(X+Y)=\mathbb{E}(X)+\mathbb{E}(Y)
\tag{eq:expectation_sums}
$$

即便 $X,Y$ 相互依赖，上式也成立。方差则不具有线性：

$$
\mathbb{V}(cX)=c^2\mathbb{V}(X)
\tag{eq:expectation_varc}
$$

一般而言，

$$
\mathbb{V}(X+Y)\ne\mathbb{V}(X)+\mathbb{V}(Y)
\tag{eq:expectation_ineq}
$$

但 $X,Y$ 独立等特殊情形除外。

随机变量 $X$ 的第 $n$ 阶矩记为 $\mathbb{E}(X^n)$，所以期望值和方差与分布的一阶、二阶矩相关。三阶矩——偏度——描述分布的不对称性。均值为 $\mu$、方差为 $\sigma^2$ 的随机变量 $X$，其偏度是三阶标准化矩：

$$
\operatorname{skew}(X)=\mathbb{E}\left(\frac{X-\mu}{\sigma}\right)^3
\tag{eq:skewness}
$$

减去均值再除以标准差，是为了让偏度不依赖 $X$ 的位置与尺度；这些信息已经由均值与方差提供。标准化也让偏度不依赖单位，便于比较。

例如 $\operatorname{Beta}(2,2)$ 的偏度为 0，$\operatorname{Beta}(2,5)$ 为正，$\operatorname{Beta}(5,2)$ 为负。对单峰分布，正偏通常表示右尾更长，负偏相反；但这并非永远成立。偏度为 0 只要求两侧尾部的**总质量**平衡，也可能是一侧又长又薄，另一侧较短却较厚。

四阶矩对应峰度，用来描述尾部或**极端值**的行为 {cite:p}`westfall2014`：

$$
\operatorname{Kurtosis}(X)=
\mathbb{E}\left(\frac{X-\mu}{\sigma}\right)^4-3
\tag{kurtosis}
$$

减去 3 是为了让高斯分布的峰度为 0，因为峰度经常以高斯分布为基准讨论；有时定义中不减 3，所以若有疑问，应查明具体采用的定义。上式本质上是在计算标准化数据四次方的期望。绝对值小于 1 的标准化数据贡献很小，真正主导峰度的是极端值。

Student t 分布的 $\nu$ 越大，峰度越低，并趋向高斯分布的 0；$\nu$ 越小，峰度越高。峰度只在 $\nu>4$ 时有定义。一般地，t 分布的第 $n$ 阶矩只在 $\nu>n$ 时存在。

SciPy 的分布对象提供 `stats(moments=...)` 方法来计算矩，前面的离散均匀示例就用它求均值与方差。本节谈的是从概率分布计算期望与矩，即理论分布的性质；实践中常从数据估计这些量，因此统计学家研究了很多估计量，例如样本均值和样本中位数都是 $\mathbb{E}(X)$ 的估计量。
""",
    "1088-1219",
    anchor="expectations",
    labels=(
        "eq:expectation", "eq:var_as_expectation", "eq:expectation_linear",
        "eq:expectation_sums", "eq:expectation_varc", "eq:expectation_ineq",
        "eq:skewness", "kurtosis",
    ),
)

add_md(
    "transformations",
    r"""
### 变换

若对随机变量 $X$ 应用函数 $g$，会得到另一个随机变量 $Y=g(X)$。已知 $X$ 的分布时，怎样求 $Y$ 的分布？一种简单办法是从 $X$ 抽样、应用变换，再画出结果；当然也有形式化方法，其中之一是**变量替换**。

若 $X$ 是连续随机变量，$Y=g(X)$，且 $g$ 可微并严格单调递增或递减，则 $Y$ 的 PDF 为：

$$
p_Y(y)=p_X(x)\left|\frac{dx}{dy}\right|
\tag{eq:changeofvariable}
$$

证明如下。先设 $g$ 严格递增，则 $Y$ 的 CDF 为：

$$
\begin{aligned}
F_Y(y)&=P(Y\le y)\\
&=P(g(X)\le y)\\
&=P(X\le g^{-1}(y))\\
&=F_X(g^{-1}(y))\\
&=F_X(x)
\end{aligned}
\tag{eq:changeofvariable_proof0}
$$

再由链式法则，从 $X$ 的 PDF 得到 $Y$ 的 PDF：

$$
p_Y(y)=p_X(x)\frac{dx}{dy}
\tag{eq:changeofvariable_proof1}
$$

$g$ 严格递减时证明类似，但右侧会多一个负号，因此一般公式要取绝对值。

对多元随机变量，也就是更高维情形，需要用雅可比行列式代替一维导数。因此，即使在一维情形，人们也常把 $\left|dx/dy\right|$ 称为雅可比项。函数 $g$ 在点 $p$ 附近使体积扩大或缩小的倍数，由雅可比行列式的绝对值给出；这一解释同样适用于概率密度。若 $g$ 非线性，变换后的分布会在某些区域收缩、另一些区域扩张，计算 $Y$ 的密度时必须正确补偿这些形变。

把变量替换公式稍作改写也很有帮助：

$$
p_Y(y)\,dy=p_X(x)\,dx
\tag{eq:changeofvariable2}
$$

它说明，在 $y$ 附近极小区间中找到 $Y$ 的概率，等于在相应 $x$ 区间中找到 $X$ 的概率。雅可比项告诉我们，怎样把 $X$ 所在空间中的概率重新映射到 $Y$ 所在空间。
""",
    "1220-1290",
    anchor="transformations",
    labels=(
        "eq:changeofvariable", "eq:changeofvariable_proof0",
        "eq:changeofvariable_proof1", "eq:changeofvariable2",
    ),
)

add_md(
    "limits",
    r"""
### 极限

概率论中最著名、使用最广的两个定理是大数定律与中心极限定理。它们都说明样本量增大时样本均值会怎样变化，也都可以放在重复实验的语境中理解：每次实验结果都可视为来自某个底层分布的样本。
""",
    "1291-1301",
    anchor="limits",
)

add_md(
    "law-of-large-numbers",
    r"""
#### 大数定律

大数定律告诉我们：独立同分布随机变量的样本均值，会随着样本数增加而收敛到该随机变量的期望值。对柯西分布等没有均值或有限方差的分布，这一结论并不成立。

大数定律经常被误解，从而导致赌徒谬误。例如，人们可能认为彩票中很久没出现的号码“更值得下注”。错误之处在于相信：若某个号码一段时间未出现，就会有某种力量提高它在下一次抽取中的概率，以恢复号码的等可能性和宇宙的“自然秩序”。独立抽取不会以这种方式补偿过去。

![来自 U(0,1) 分布的运行均值。0.5 处的虚线表示期望值；随着抽取次数增加，经验均值趋近该值。每条线代表一个不同样本序列。](img/law_of_large_numbers.png)
""",
    "1302-1328",
    anchor="the-law-of-large-numbers",
    labels=("fig:law_of_large_numbers",),
)

add_md(
    "central-limit-theorem",
    r"""
#### 中心极限定理

中心极限定理指出：若从任意分布中独立抽取 $n$ 个值，那么当 $n\to\infty$ 时，这些值的均值 $\bar X$ 近似服从高斯分布：

$$
\bar X_n\mathrel{\dot\sim}\mathcal{N}\left(\mu,\frac{\sigma^2}{n}\right)
\tag{eq:central_limit}
$$

其中 $\mu$ 与 $\sigma^2$ 是原分布的均值和方差。

中心极限定理通常要求：

- 各值独立抽取；
- 各值来自同一分布；
- 该分布的均值与标准差有限。

前两个条件可以放宽很多，结果仍往往近似高斯；第三个条件却无法绕过。柯西分布没有定义良好的均值与方差，定理不适用：$N$ 个柯西变量的平均仍服从柯西分布，而不是高斯分布。

中心极限定理解释了高斯分布在自然界中的普遍性。许多现象可以理解为围绕均值的波动，或许多不同因素之和。

下图对 $\operatorname{Pois}(2.3)$、$\mathcal{U}(0,1)$ 与 $\operatorname{Beta}(1,10)$ 三个分布展示中心极限定理随 $n$ 增大的效果。

![左侧标明原始分布。每幅直方图都由 1000 个模拟的 X̄_n 值构成；随着 n 增大，X̄_n 的分布趋近正态。黑色曲线是中心极限定理给出的高斯近似。](img/central_limit.png)
""",
    "1329-1381",
    anchor="appendix_clt",
    labels=("eq:central_limit", "fig:central_limit"),
)

add_md(
    "markov-chains",
    r"""
### 马尔可夫链

马尔可夫链是随机变量序列 $X_0,X_1,\ldots$，在给定当前状态后，未来状态与所有过去状态条件独立。换句话说，只要知道当前状态，就足以确定所有未来状态的概率。这称为马尔可夫性质：

$$
P(X_{n+1}=j\mid X_n=i,X_{n-1}=i_{n-1},\ldots,X_0=i_0)
=P(X_{n+1}=j\mid X_n=i)
\tag{markov_property}
$$

一种很有效的可视化方式，是想象你或某个物体在空间中移动[^15]。有限空间更容易理解，例如棋盘上的棋子或访问不同城市的销售员。由此可以问：访问某一状态（特定方格、城市等）的可能性有多大？如果不断在状态间移动，长期来看会在每个状态停留多久？

下图给出四个马尔可夫链例子。第一个是经典但极度简化的天气模型，状态为晴天或雨天；第二个是确定性骰子；后两个更抽象，没有为状态指定具体含义。

![马尔可夫链示例。(a) 极简天气模型：箭头表示晴、雨状态间的转移，并标注转移概率。(b) 周期马尔可夫链。(c) 不连通链：状态 1、2、3 与 A、B 不连通；从任一组出发都无法到达另一组，图中省略转移概率。(d) 赌徒破产问题：赌徒 A、B 分别有 i 与 N−i 单位资金，每次下注 1 单位；A 以概率 p 获胜、以 q=1−p 失败。若 X_n 是时刻 n 时 A 的资金，则 X_0,X_1,… 构成图示马尔可夫链。](img/markov_chains_graph.png)

研究马尔可夫链的一种方便方法，是把一步内各状态之间的转移概率收集到转移矩阵 $\mathbf T=(t_{ij})$ 中。例如，上图 (a) 的转移矩阵为

$$
\begin{bmatrix}
0.9 & 0.1\\
0.8 & 0.2
\end{bmatrix}
$$

上图 (b) 的源文矩阵写作

$$
\begin{bmatrix}
0 & 0 & 1 & 0 & 0 & 0 & 0\\
1 & 0 & 0 & 1 & 0 & 0 & 0\\
2 & 0 & 0 & 0 & 1 & 0 & 0\\
3 & 0 & 0 & 0 & 0 & 1 & 0\\
4 & 0 & 0 & 0 & 0 & 0 & 1\\
5 & 1 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}
$$

这里保留了权威 Markdown 中把状态编号列并入矩阵的排版；忽略最左侧编号列后，右侧 $6\times6$ 部分就是确定性循环的转移概率矩阵。

转移矩阵第 $i$ 行表示从状态 $X_n=i$ 到下一状态 $X_{n+1}$ 的条件概率分布，即 $p(X_{n+1}\mid X_n=i)$。例如，当前为晴天时，以 0.9 的概率仍为晴天，以 0.1 的概率转为雨天；从晴天转向某处的总概率为 1，符合 PMF 的要求。

由于马尔可夫性质，连续 $n$ 步的转移概率可由 $\mathbf T^n$ 得到。还可以指定初始条件 $s_i=P(X_0=i)$，并令 $\mathbf s=(s_1,\ldots,s_M)$；这样，$X_n$ 的边缘 PMF 就是 $\mathbf s\mathbf T^n$。

研究马尔可夫链时，可以定义单个状态或整条链的性质。如果链反复回到某一状态，就称它为**常返状态**；链最终永远离开的状态称为**暂态**。图 (d) 中，除 0 与 $N$ 外所有状态都是暂态。若能在有限步内从任一状态到达任一其他状态，则称整条链**不可约**；图 (c) 不是不可约的，因为 1、2、3 与 A、B 互不连通。

马尔可夫链的长期行为很重要。安德烈·马尔可夫引入它，正是为了说明大数定律也可应用于非独立随机变量。常返与暂态概念有助于理解长期行为：若一条链同时有暂态和常返状态，它可能先在暂态停留，但最终会永远待在常返状态中。每个状态长期占据多少时间，由链的**平稳分布**回答。

对有限马尔可夫链，平稳分布 $\mathbf s$ 是满足 $\mathbf s\mathbf T=\mathbf s$ 的 PMF[^16]，即经转移矩阵作用后不改变的分布。这并不意味着链停止移动，而是说链移动时在各状态花费的长期比例由 $\mathbf s$ 给出。可类比密封但未装满水的杯子：液态水分子蒸发到空气中，空气中的水分子也回到液体。系统最终达到动态平衡；局部仍在运动，全局却不再变化[^17]。稳态也是平稳分布的另一个名称。

在若干条件下，有限马尔可夫链的平稳分布存在且唯一，并且当 $n\to\infty$ 时 $X_n$ 的 PMF 收敛到 $\mathbf s$。图 (d) 没有唯一平稳分布：到达 0 或 $N$ 后就永久停留，所以 $s_0=(1,0,\ldots,0)$ 与 $s_N=(0,0,\ldots,1)$ 都是平稳分布。相反，图 (b) 虽然转移是确定性的，却有唯一平稳分布 $s=(1/6,1/6,1/6,1/6,1/6,1/6)$。

如果 PMF $\mathbf s$ 满足可逆条件（也称细致平衡）$s_it_{ij}=s_jt_{ji}$ 对所有 $i,j$ 成立，那么可以保证 $\mathbf s$ 是转移矩阵 $\mathbf T=(t_{ij})$ 对应马尔可夫链的平稳分布。这样的链称为可逆链。“推断方法”一节将利用这一性质解释 Metropolis–Hastings 为什么能在渐近意义下工作。

马尔可夫链也满足与式 {eq}`eq:central_limit` 类似的中心极限定理，只是需要用有效样本量（ESS）替代 $n$。本书 {ref}`ess` 一节讨论了如何从马尔可夫链估计 ESS，并用它诊断链的质量。$\sqrt{\sigma^2/\mathrm{ESS}}$ 就是 {ref}`Monte_Carlo_standard_error` 一节介绍的蒙特卡洛标准误（MCSE）。
""",
    "1382-1559",
    anchor="markov_chains",
    labels=("markov_property", "fig:markov_chains_graph"),
    modernization="保留源矩阵的排版并明确指出最左列是状态编号，避免把 6×7 展示误读成转移矩阵维度。",
)

add_md(
    "entropy",
    r"""
## 熵

在维也纳中央公墓（Zentralfriedhof），路德维希·玻尔兹曼的墓碑刻着 $S=k\log W$。这是一种优美的说法：热力学第二定律是概率定律的结果。玻尔兹曼借此推动了现代物理支柱之一——统计力学——的发展。统计力学解释温度等宏观观测如何与分子的微观世界相联系。想象一杯水：感官所见基本上是杯中海量水分子的平均行为[^18]。在给定温度下，与该温度相容的水分子排列方式有一定数量；温度越低，可能排列越少，直到只剩一种，这就是 0 K——宇宙中可能的最低温度。反向升温时，分子可处于越来越多的排列。

![粒子可能采取的排列数与系统温度有关。图示含 3 个等价粒子的离散系统，可用单元格（灰色高线）数表示可能排列；升温等价于增加可用单元格。T=0 时只有一种排列，温度上升时粒子可占据越来越多状态。](img/entropy_T.png)

可用不确定性分析这个思想实验。若系统处于 0 K，它只能有一种排列，我们对其状态绝对确定[^19]。升温后可能排列增多，指出“此刻水分子恰好采用这一排列”会越来越困难。我们仍可用温度、体积等平均量刻画系统，但对具体微观排列的确定性会下降。因此，可以把熵理解为不确定性的度量。

熵并不只适用于分子，也适用于像素排列、文本字符、音符、袜子、酸面包中的气泡等。它之所以灵活，是因为它量化对象的排列方式，是底层分布的性质。分布熵越大，信息越少，给各事件分配的概率越均匀。“42”比“$42\pm5$”更确定，后者又比“任意实数”更确定；熵能把这种定性观察转换成数字。

熵适用于连续与离散分布，但用离散状态更容易思考，所以下文采用离散例子；同样思想也适用于连续情形。

若概率分布 $p$ 有 $n$ 个不同事件，第 $i$ 个事件的概率为 $p_i$，则熵定义为：

$$
H(p)=-\mathbb E[\log p]= -\sum_i^n p_i\log p_i
\tag{eq:entropy}
$$

这只是玻尔兹曼墓碑公式的另一种写法：这里用 $H$ 而非 $S$ 表示熵，并取 $k=1$。玻尔兹曼公式中的重数 $W$ 是不同结果可能出现方式的总数：

$$
W=\frac{N!}{n_1!n_2!\cdots n_t!}
\tag{eq:degeneracy}
$$

可将其想成把一个 $t$ 面骰掷 $N$ 次，$n_i$ 是第 $i$ 面出现次数。当 $N$ 很大时，用斯特林近似 $x!\approx(x/e)^x$：

$$
W=\frac{N^N}{n_1^{n_1}n_2^{n_2}\cdots n_t^{n_t}}
 e^{\left(\sum_i n_i-N\right)}
\tag{eq:Stirling0}
$$

因为 $n_1+n_2+\cdots+n_t=N$，指数项 $\sum_i n_i-N$ 恒为 0，$e^0=1$，指数因子直接消失。注意 $p_i=n_i/N$，可写成：

$$
W=\frac{1}{p_1^{n_1}p_2^{n_2}\cdots p_t^{n_t}}
\tag{eq:p_as_frac}
$$

最后取对数，并用 $n_i=Np_i$ 代入，得到

$$
\log W=-\sum_i^n n_i\log p_i=-N\sum_i^n p_i\log p_i
\tag{eq:entropy_W}
$$

也就是说 $\dfrac1N\log W=-\sum_i^n p_i\log p_i=H(p)$：熵是重数对数按系统规模 $N$ 归一化后的极限，这正是玻尔兹曼墓碑公式与熵定义之间的对应关系。下面用 Python 计算几个分布的熵。
""",
    "1560-1670",
    anchor="entropy",
    labels=(
        "fig:entropy_T", "eq:entropy", "eq:degeneracy", "eq:Stirling0",
        "eq:p_as_frac", "eq:entropy_W",
    ),
    provenance="translation-with-correction",
    modernization=(
        "原文推导中，eq:Stirling0 的指数项被误写成 n_1、n_2…、n_t 的连乘再减 N"
        "（应为它们的求和 \\sum_i n_i 减 N，且由于 \\sum_i n_i=N 该项恒为 0，"
        "指数因子应直接消失而非保留在式中）；eq:entropy_W 把 log W 直接等同于"
        "熵 H(p)，遗漏了系统规模 N 这个因子——正确关系是 log W = N·H(p)，"
        "即 (1/N)log W = H(p)。这里已按正确推导改写这两处公式并据此调整了衔接文字。"
    ),
)

add_code(
    "entropy-dist",
    r"""
x_entropy = np.arange(26)
q_pmf = stats.binom(10, 0.75).pmf(x_entropy)
qu_pmf = stats.randint(0, np.flatnonzero(q_pmf)[-1] + 1).pmf(x_entropy)
r_pmf = (q_pmf + np.roll(q_pmf, 12)) / 2
ru_pmf = stats.randint(0, np.flatnonzero(r_pmf)[-1] + 1).pmf(x_entropy)
s_pmf = (q_pmf + np.roll(q_pmf, 15)) / 2
su_pmf = (qu_pmf + np.roll(qu_pmf, 15)) / 2

entropy_dists = [q_pmf, qu_pmf, r_pmf, ru_pmf, s_pmf, su_pmf]
entropy_names = ["q", "qu", "r", "ru", "s", "su"]
fig, axes = plt.subplots(
    3, 2, figsize=(10, 6), sharex=True, sharey=True, constrained_layout=True
)
for axis, dist, label in zip(axes.ravel(), entropy_dists, entropy_names):
    axis.vlines(x_entropy, 0, dist, color="black", linewidth=1.5)
    axis.set_title(label)
    axis.text(0.98, 0.92, f"H = {stats.entropy(dist):.2f}",
              transform=axis.transAxes, ha="right", va="top")
    assert np.isclose(dist.sum(), 1.0)
    assert np.isfinite(stats.entropy(dist))
plt.close(fig)
""",
    "1671-1693",
    labels=("entropy_dist",),
    migration_cells=(13,),
    modernization="改用稳定的 flatnonzero 支撑集计算、显式灰度样式与归一化/有限熵断言。",
)

add_md(
    "entropy-examples",
    r"""
![代码单元格 entropy-dist 定义的六个离散分布及其熵值 H。](img/entropy.png)

上图内容较多。最尖、最集中的 $q$ 在六个分布中熵最低。$q\sim\operatorname{Binom}(n=10,p=0.75)$，因此有 11 个可能事件；$qu$ 是同样拥有 11 个事件的均匀分布，其熵更大。事实上，对 $n=10$、不同 $p$ 的二项分布，熵都不会超过 $qu$；需要把 $n$ 增大约 3 倍，才会遇到第一个熵比 $qu$ 更大的二项分布。

$r$ 由 $q$ 与右移后的 $q$ 各占一半构成。因为 $r$ 比 $q$ 更分散，所以熵更大。$ru$ 是覆盖 $r$ 全部 23 个位置的均匀分布，其中包括两峰之间“谷底”的位置；均匀版本仍具有更大的熵。

不过，熵并不简单地与方差成正比。$s$ 基本上与 $r$ 相同，只是两峰之间的谷更宽，但二者熵相同，因为熵不关心概率为零的谷底事件，只关心具有正概率的可能事件。$su$ 用两个 $qu$ 峰替换 $s$ 的两峰并归一化。尽管 $su$ 看起来更分散，它的熵仍低于 $ru$：$su$ 把总概率分在 22 个事件上，而 $ru$ 覆盖 23 个事件。
""",
    "1695-1734",
    labels=("fig:entropy",),
)

add_md(
    "kl-divergence",
    r"""
## Kullback–Leibler 散度

统计学中常用一个概率分布 $q$ 表示另一个分布 $p$：或许不知道 $p$，只能用 $q$ 近似；或许 $p$ 很复杂，希望寻找更简单、更方便的 $q$。这时可以问：用 $q$ 表示 $p$ 损失了多少信息，等价地说，引入了多少额外不确定性？直觉上，我们希望某个量只在 $q=p$ 时为 0，其他时候为正。沿用熵的定义，可取 $\log p$ 与 $\log q$ 之差的期望，这就是 Kullback–Leibler（KL）散度：

$$
\mathbb{KL}(p\parallel q)=\mathbb E_p[\log p-\log q]
\tag{eq:kl_divergence}
$$

因此，$\mathbb{KL}(p\parallel q)$ 是用 $q$ 近似 $p$ 时对数概率差的平均值。事件按照 $p$ 出现，所以期望必须相对于 $p$ 计算。离散分布下：

$$
\mathbb{KL}(p\parallel q)=\sum_i^n p_i(\log p_i-\log q_i)
\tag{eq:kl_divergence_discrete}
$$

利用对数性质，可写成最常见的形式：

$$
\mathbb{KL}(p\parallel q)=\sum_i^n p_i\log\frac{p_i}{q_i}
\tag{eq:kl_divergence_log}
$$

也可重新排列为：

$$
\mathbb{KL}(p\parallel q)=-\sum_i^n p_i(\log q_i-\log p_i)
\tag{eq:kl_divergence_log_dif}
$$

展开后得到：

$$
\mathbb{KL}(p\parallel q)=
\overbrace{-\sum_i^n p_i\log q_i}^{H(p,q)}-
\overbrace{\left(-\sum_i^n p_i\log p_i\right)}^{H(p)}
\tag{eq:kl_divergence_cross_entropy}
$$

$H(p)$ 是 $p$ 的熵；$H(p,q)=-\mathbb E_p[\log q]$ 是按照 $p$ 的取值来评价 $q$ 的交叉熵。重新排列得：

$$
H(p,q)=H(p)+D_{\mathrm{KL}}(p\parallel q)
\tag{eq:cross_entropy}
$$

这表明，用 $q$ 表示 $p$ 时，KL 散度可以解释为相对于 $H(p)$ 增加的额外熵。下面对上一节的六个分布计算两两 KL 散度。
""",
    "1735-1808",
    anchor="DKL",
    labels=(
        "eq:kl_divergence", "eq:kl_divergence_discrete",
        "eq:kl_divergence_log", "eq:kl_divergence_log_dif",
        "eq:kl_divergence_cross_entropy", "eq:cross_entropy",
    ),
)

add_code(
    "kl-varies-dist",
    r"""
kl_matrix = np.zeros((len(entropy_dists), len(entropy_dists)))
for row, dist_p in enumerate(entropy_dists):
    for col, dist_q in enumerate(entropy_dists):
        kl_matrix[row, col] = stats.entropy(dist_p, dist_q)

masked_kl = np.ma.masked_invalid(kl_matrix)
gray_map = plt.colormaps["Greys"].copy()
gray_map.set_bad("white")
fig, axis = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
image = axis.imshow(masked_kl, cmap=gray_map)
axis.set_xticks(range(len(entropy_names)), entropy_names)
axis.set_yticks(range(len(entropy_names)), entropy_names)
axis.set_xlabel("近似分布 q")
axis.set_ylabel("目标分布 p")
fig.colorbar(image, ax=axis, label=r"$D_{KL}(p\parallel q)$")

assert np.allclose(np.diag(kl_matrix), 0.0, atol=1e-12)
assert np.all((kl_matrix >= 0) | np.isinf(kl_matrix))
finite_pairs = np.isfinite(kl_matrix) & np.isfinite(kl_matrix.T)
assert np.any(np.abs(kl_matrix[finite_pairs] - kl_matrix.T[finite_pairs]) > 1e-6)
plt.close(fig)
""",
    "1809-1823",
    labels=("kl_varies_dist",),
    migration_cells=(15,),
    modernization="用 Matplotlib 公共 Greys 色图替代可选 colorcet 名称，屏蔽无穷值，并添加非负、对角为零与非对称性断言。",
)

add_md(
    "kl-properties-log-score",
    r"""
上面代码的结果如下图。两点很醒目。第一，图不对称，因为 $\mathbb{KL}(p\parallel q)$ 通常不等于 $\mathbb{KL}(q\parallel p)$。第二，许多白色单元格表示 $\infty$。KL 散度采用以下约定 {cite:p}`Cover_Thomas`：

$$
0\log\frac00=0,\qquad
0\log\frac{0}{q(\boldsymbol x)}=0,\qquad
p(\boldsymbol x)\log\frac{p(\boldsymbol x)}0=\infty
\tag{eq:kl_divergence_conventions}
$$

![分布 q、qu、r、ru、s、su 的所有两两 KL 散度；白色表示无穷值。各分布见上一幅熵图。](img/KL_heatmap.png)

还可以用 KL 散度说明，为什么计算期望逐点对数预测密度时采用对数评分。设有 $k$ 个模型后验 $\{q_{M_1},q_{M_2},\ldots,q_{M_k}\}$，并暂且假设知道真实模型 $M_0$，则：

$$
\begin{aligned}
\mathbb{KL}(p_{M_0}\parallel q_{M_1})&=\mathbb E[\log p_{M_0}]-\mathbb E[\log q_{M_1}],\\
\mathbb{KL}(p_{M_0}\parallel q_{M_2})&=\mathbb E[\log p_{M_0}]-\mathbb E[\log q_{M_2}],\\
&\ \vdots\\
\mathbb{KL}(p_{M_0}\parallel q_{M_k})&=\mathbb E[\log p_{M_0}]-\mathbb E[\log q_{M_k}].
\end{aligned}
\tag{eq:kl_divergence_log_score}
$$

现实中当然不知道真实模型 $M_0$。关键在于，$p_{M_0}$ 对所有比较都相同，所以按 KL 散度排序，等价于按对数评分排序。
""",
    "1825-1870",
    labels=(
        "eq:kl_divergence_conventions", "fig:KL_heatmap",
        "eq:kl_divergence_log_score",
    ),
)

add_md(
    "information-criterion",
    r"""
## 信息准则

信息准则用于衡量统计模型的预测准确性：既考虑模型对数据的拟合程度，也惩罚模型复杂度。不同准则以不同方式计算这两项。其中最著名的——尤其在非贝叶斯统计中——是 Akaike 信息准则（AIC）{cite:p}`akaike_1973`：

$$
\mathrm{AIC}=-2\sum_i^n\log p(y_i\mid\hat\theta_{\mathrm{mle}})+2p_{\mathrm{AIC}}
\tag{eq:aic}
$$

$\hat\theta_{\mathrm{mle}}$ 是 $\boldsymbol\theta$ 的最大似然估计，$p_{\mathrm{AIC}}$ 就是模型参数个数。第一项衡量拟合，惩罚项则补偿“用同一数据拟合又评价模型”造成的乐观偏差。

AIC 在非贝叶斯场景很流行，却不足以处理贝叶斯模型的一般性。它不利用完整后验，因而丢弃潜在有用信息。从平坦先验转向弱信息或信息性先验，或在层级模型中加入更多结构时，AIC 的平均表现会越来越差。AIC 还假设后验至少在渐近意义下可由高斯分布良好表示；层级模型、混合模型、神经网络等常不满足这一点，所以需要更合适的替代方案。

广泛适用信息准则（WAIC[^20]）{cite:p}`watanabe2010asymptotic` 可视为 AIC 的完全贝叶斯扩展。它也包含含义大致相同的拟合与惩罚项，但二者都使用完整后验分布计算：

$$
\mathrm{WAIC}=
\sum_i^n\log\left(\frac1s\sum_j^S p(y_i\mid\boldsymbol\theta^j)\right)
-\sum_i^n\left(\mathbb V_j^s\log p(Y_i\mid\boldsymbol\theta^j)\right)
\tag{eq:waic}
$$

第一项与 AIC 的对数似然类似，但在每个观测点 $i$ 上逐点求值；对 $s$ 个后验样本取平均，使后验不确定性进入计算。它是理论期望逐点对数预测密度（ELPD，见式 {eq}`eq:elpd`）及其实际近似（式 {eq}`eq:elpd_practice`）的一种计算方式。

第二项是每个观测点上、跨 $s$ 个后验样本的对数似然方差。若某一观测在整个后验中的对数似然相近，方差较低；若对后验“细节”敏感，方差较高，惩罚也更大。也可以从灵活性理解：既容纳直线又容纳向上弯曲曲线的模型，比只容纳直线的模型更灵活；前者在后验上评价观测时，对数似然平均会有更高方差。若灵活模型提高的估计 ELPD 不足以抵消惩罚，简单模型就会排名更高。因此，WAIC 的方差项通过惩罚过度复杂模型来抑制过拟合，可以宽泛地解释为 AIC 中参数数目的贝叶斯对应物。

AIC 与 WAIC 都不试图判断模型是否“为真”，它们只是比较备选模型的相对量。贝叶斯观点下先验属于模型的一部分，但 WAIC 在后验上评价，先验仅通过影响后验而间接进入。BIC、WBIC 等其他信息准则试图处理不同问题，并可看成边缘似然的近似，本书不再讨论。
""",
    "1871-1954",
    anchor="information_criterion",
    labels=("eq:aic", "eq:waic"),
)

add_md(
    "loo-in-depth",
    r"""
## 深入理解 LOO

如本书 {ref}`CV_and_LOO` 一节所述，这里的 LOO 特指用 Pareto 平滑重要性采样近似留一交叉验证的方法，即 PSIS-LOO-CV。本节讨论其中几个细节。

LOO 是 WAIC 的替代方案；二者可证明在渐近意义下收敛到同一个数值 {cite:p}`watanabe2010asymptotic,vehtari_practical_2017`。但 LOO 对实践者有两个重要优势：它在有限样本下更稳健，并在计算过程中提供有用诊断 {cite:p}`vehtari_practical_2017,gabry_visualization_2017`。

在 LOO-CV 下，新数据集的期望逐点对数预测密度是：

$$
\operatorname{ELPD}_{\mathrm{LOO-CV}}=
\sum_{i=1}^n\log\int p(y_i\mid\boldsymbol\theta)
 p(\boldsymbol\theta\mid y_{-i})\,d\boldsymbol\theta
\tag{eq:elpd_loo_cv}
$$

其中 $y_{-i}$ 表示移除第 $i$ 个观测的数据集。实践中不知道 $\boldsymbol\theta$，可用从相应后验抽取的 $s$ 个样本近似：

$$
\sum_i^n\log\left(\frac1s\sum_j^s
 p(y_i\mid\boldsymbol\theta_{-i}^j)\right)
\tag{eq:loo_cv_naive}
$$

它类似 WAIC 第一项，不同之处是每次移除一个观测，共计算 $n$ 个后验，因此不再需要额外惩罚项。但这样直接计算非常昂贵。若 $n$ 个观测条件独立，则可以用下式近似 {cite:p}`gilks1995markov,vehtari_practical_2017`：

$$
\operatorname{ELPD}_{\mathrm{psis-loo}}=
\sum_i^n\log\sum_j^s w_i^j p(y_i\mid\boldsymbol\theta^j)
\tag{eq:loo}
$$

其中 $w$ 是归一化权重向量。

计算 $w$ 使用重要性采样。它在只有另一个分布 $g$ 的样本时，用来估计目标分布 $f$ 的性质；当从 $g$ 抽样比从 $f$ 抽样容易时尤其有意义。若有随机变量 $X$ 的样本，并能逐点评价 $g,f$，重要性权重为：

$$
w_i=\frac{f(x_i)}{g(x_i)}
\tag{eq:importance_weights}
$$

计算过程如下：

1. 从 $g$ 抽取 $N$ 个样本 $x_i$；
2. 计算每个样本的密度 $g(x_i)$；
3. 在同一批样本上计算 $f(x_i)$；
4. 计算权重 $w_i=f(x_i)/g(x_i)$；
5. 返回带权样本 $(x_i,w_i)$，供估计量使用。

下图使用两个不同提议分布近似同一个目标分布（虚线）。第一行的提议比目标更宽，第二行更窄；第一种近似明显更好，这是重要性采样的一般特征。

![重要性采样。左列为提议分布 g 样本的 KDE；右列中虚线是目标分布，实线是用式 eq:importance_weights 的权重重新加权提议样本后得到的近似。](img/importance_sampling.png)

回到 LOO：已有样本来自完整数据后验，而评价模型需要来自留一后验的样本，因此希望计算的权重是：

$$
w_i^j=\frac{p(\theta^j\mid y_{-i})}{p(\theta^j\mid y)}
\propto\frac{1}{p(y_i\mid\theta^j)}
\tag{eq:loocv_weights}
$$

这个比例关系使权重几乎可以免费得到。不过，完整后验的尾部很可能比留一后验更薄；如上图所示，这会造成很差的估计。数学上，重要性权重可能具有很高、甚至无穷的方差。LOO 为控制方差，会用估计的 Pareto 分布值替换最大的若干重要性权重，从而显著提升稳健性 {cite:p}`vehtari_practical_2017`。估计出的 Pareto 参数 $\hat\kappa$ 还能检测高影响观测，即移除后会显著改变预测分布的观测。一般而言，较高的 $\hat\kappa$ 暗示数据或模型可能有问题，特别是 $\hat\kappa>0.7$ 时 {cite:p}`vehtari_pareto_2019,gabry_visualization_2017`。
""",
    "1955-2081",
    anchor="loo_depth",
    labels=(
        "eq:elpd_loo_cv", "eq:loo_cv_naive", "eq:loo",
        "eq:importance_weights", "fig:importance_sampling", "eq:loocv_weights",
    ),
    modernization="把源文失效的 LaTeX ref/tag 组合恢复为可引用标签 eq:elpd_loo_cv。",
)

add_md(
    "jeffreys-prior-derivation",
    r"""
## Jeffreys 先验推导

本节推导二项似然的 Jeffreys 先验：先以成功概率 $\theta$ 参数化，再以优势比 $\kappa=\theta/(1-\theta)$ 参数化。

回顾第一章，一维参数 $\theta$ 的 Jeffreys 先验定义为：

$$
p(\theta)\propto\sqrt{I(\theta)}
$$

其中 $I(\theta)$ 是 Fisher 信息：

$$
I(\theta)=-\mathbb E_Y\left[\frac{d^2}{d\theta^2}
\log p(Y\mid\theta)\right].
$$
""",
    "2082-2103",
    anchor="Jeffreys_prior_derivation",
)

add_md(
    "jeffreys-binomial-theta",
    r"""
### 以 $\theta$ 表示的二项似然 Jeffreys 先验

二项似然可写为：

$$
p(Y\mid\theta)\propto\theta^y(1-\theta)^{n-y}
\tag{eq:binomial_kernel}
$$

其中 $y$ 是成功次数，$n$ 是总试验数，$n-y$ 是失败次数。之所以写成正比，是因为似然中的二项系数不依赖 $\theta$。

为计算 Fisher 信息，先取似然的对数：

$$
\ell=\log p(Y\mid\theta)
\propto y\log\theta+(n-y)\log(1-\theta)
\tag{eq:JP_0}
$$

再计算一阶与二阶导数：

$$
\begin{aligned}
\frac{d\ell}{d\theta}&=\frac y\theta-\frac{n-y}{1-\theta},\\
\frac{d^2\ell}{d\theta^2}&=-\frac y{\theta^2}-\frac{n-y}{(1-\theta)^2}.
\end{aligned}
\tag{eq:JP_1}
$$

Fisher 信息是负二阶导数的期望：

$$
I(\theta)=-\mathbb E_Y\left[-\frac y{\theta^2}
-\frac{n-y}{(1-\theta)^2}\right]
\tag{eq:JP_2}
$$

因为 $\mathbb E[y]=n\theta$，所以

$$
I(\theta)=\frac{n\theta}{\theta^2}
+\frac{n-n\theta}{(1-\theta)^2}
\tag{eq:JP_3}
$$

可重写为

$$
I(\theta)=\frac n\theta+\frac{n(1-\theta)}{(1-\theta)^2}
=\frac n\theta+\frac n{1-\theta}
\tag{eq:JP_4}
$$

通分得到

$$
I(\theta)=n\left[
\frac{1-\theta}{\theta(1-\theta)}+
\frac{\theta}{\theta(1-\theta)}\right]
\tag{eq:JP_5}
$$

合并后：

$$
I(\theta)=n\frac1{\theta(1-\theta)}
\tag{eq:JP_6}
$$

忽略与 $\theta$ 无关的 $n$：

$$
I(\theta)\propto\frac1{\theta(1-\theta)}
=\theta^{-1}(1-\theta)^{-1}
\tag{eq:fisher_info}
$$

最后取平方根，得到以 $\theta$ 参数化的二项似然 Jeffreys 先验：

$$
p(\theta)\propto\theta^{-0.5}(1-\theta)^{-0.5}
\tag{eq:alice_prior}
$$

> **译注与勘误。** 权威 Markdown 的式 `eq:JP_2` 至 `eq:JP_5` 在 $(n-y)$ 项上有连锁符号错误，与其正确终式 `eq:JP_6` 矛盾。这里保留所有原标签，并显式修正这些符号；这是数学勘误，不是新增推导。
""",
    "2104-2194",
    anchor="jeffreys-prior-for-the-binomial-likelihood-in-terms-of-theta",
    labels=(
        "eq:binomial_kernel", "eq:JP_0", "eq:JP_1", "eq:JP_2",
        "eq:JP_3", "eq:JP_4", "eq:JP_5", "eq:JP_6",
        "eq:fisher_info", "eq:alice_prior",
    ),
    provenance="translation-with-correction",
    modernization="修正源文 eq:JP_2 至 eq:JP_5 的符号错误，同时保留原标签并显式标注勘误。",
)

add_md(
    "jeffreys-binomial-kappa",
    r"""
### 以 $\kappa$ 表示的二项似然 Jeffreys 先验

现在推导以优势比 $\kappa$ 参数化的 Jeffreys 先验。把 $\theta=\kappa/(\kappa+1)$ 代入式 {eq}`eq:binomial_kernel`：

$$
p(Y\mid\kappa)\propto
\left(\frac\kappa{\kappa+1}\right)^y
\left(1-\frac\kappa{\kappa+1}\right)^{n-y}
\tag{eq:JP_7}
$$

也可写为：

$$
p(Y\mid\kappa)\propto
\kappa^y(\kappa+1)^{-y}(\kappa+1)^{-n+y}
\tag{eq:JP_8}
$$

进一步化简：

$$
p(Y\mid\kappa)\propto\kappa^y(\kappa+1)^{-n}
\tag{eq_likelihood_binom_odds}
$$

取对数：

$$
\ell=\log p(Y\mid\kappa)
\propto y\log\kappa-n\log(\kappa+1)
\tag{eq:JP_9}
$$

再求导：

$$
\begin{aligned}
\frac{d\ell}{d\kappa}&=\frac y\kappa-\frac n{\kappa+1},\\
\frac{d^2\ell}{d\kappa^2}&=-\frac y{\kappa^2}+\frac n{(\kappa+1)^2}.
\end{aligned}
\tag{eq:JP_10}
$$

Fisher 信息为：

$$
I(\kappa)=-\mathbb E_Y\left[-\frac y{\kappa^2}
+\frac n{(\kappa+1)^2}\right]
\tag{eq:JP_11}
$$

因为 $\mathbb E[y]=n\theta=n\kappa/(\kappa+1)$，有

$$
I(\kappa)=\frac n{\kappa(\kappa+1)}-
\frac n{(\kappa+1)^2}
\tag{eq:JP_12}
$$

通分：

$$
I(\kappa)=\frac{n(\kappa+1)}{\kappa(\kappa+1)^2}
-\frac{n\kappa}{\kappa(\kappa+1)^2}
\tag{eq:JP_13}
$$

合并：

$$
I(\kappa)=\frac{n(\kappa+1)-n\kappa}{\kappa(\kappa+1)^2}
\tag{eq:JP_14}
$$

化简得：

$$
I(\kappa)=\frac n{\kappa(\kappa+1)^2}
\tag{eq:JP_15}
$$

取平方根，得到：

$$
p(\kappa)\propto\kappa^{-0.5}(1+\kappa)^{-1}
\tag{eq:bob_prior}
$$
""",
    "2195-2287",
    anchor="jeffreys-prior-for-the-binomial-likelihood-in-terms-of-kappa",
    labels=(
        "eq:JP_7", "eq:JP_8", "eq_likelihood_binom_odds", "eq:JP_9",
        "eq:JP_10", "eq:JP_11", "eq:JP_12", "eq:JP_13",
        "eq:JP_14", "eq:JP_15", "eq:bob_prior",
    ),
)

add_md(
    "jeffreys-binomial-posterior",
    r"""
### 二项似然的 Jeffreys 后验

以 $\theta$ 参数化时，把式 {eq}`eq:binomial_kernel` 与式 {eq}`eq:alice_prior` 相乘：

$$
p(\theta\mid Y)\propto
\theta^y(1-\theta)^{n-y}\theta^{-0.5}(1-\theta)^{-0.5}
=\theta^{y-0.5}(1-\theta)^{n-y-0.5}
\tag{eq:alice_posterior}
$$

以 $\kappa$ 参数化时，把式 {eq}`eq_likelihood_binom_odds` 与式 {eq}`eq:bob_prior` 相乘：

$$
p(\kappa\mid Y)\propto
\kappa^y(\kappa+1)^{-n}\kappa^{-0.5}(1+\kappa)^{-1}
=\kappa^{y-0.5}(\kappa+1)^{-n-1}
\tag{eq:bob_posterior}
$$
""",
    "2288-2311",
    anchor="jeffreys-posterior-for-the-binomial-likelihood",
    labels=("eq:alice_posterior", "eq:bob_posterior"),
    modernization="移除源文 eq:bob_posterior 末尾多余右括号，不改变数学内容。",
)

add_md(
    "marginal-likelihood",
    r"""
## 边缘似然

对共轭先验等模型，边缘似然可以解析计算；其他模型中，对通常很复杂且变化剧烈的函数做高维积分是公认的困难任务 {cite:p}`Friel_2011`。本节试图解释它为什么通常难算。

在低维情况下，可以在网格上计算先验与似然的乘积，再使用梯形法或类似数值方法求边缘似然。然而，“高维空间”一节将说明网格不能随维度良好扩展：变量增加时，所需网格点数迅速增长，超过少数几个变量便不切实际。最朴素的蒙特卡洛积分也可能有问题。为此，人们提出了许多专门计算边缘似然的方法 {cite:p}`Friel_2011`。这里仅讨论其中一种，目的不是教授实务计算，而是说明困难来源。
""",
    "2312-2336",
    anchor="marginal_likelihood",
)

add_md(
    "harmonic-mean-estimator",
    r"""
### 调和均值估计量

调和均值估计量是一个声名不佳的边缘似然估计量 {cite:p}`Neal_1994`。它诱人的特点是只需 $s$ 个后验样本：

$$
p(Y)\approx\left(\frac1s\sum_{i=1}^s
\frac1{p(Y\mid\boldsymbol\theta_i)}\right)^{-1}
\tag{eq:harmonic_mean_approx}
$$

也就是先对后验样本处似然的倒数求平均，再对结果取倒数。原则上，这是下列期望的有效蒙特卡洛估计：

$$
\mathbb E\left[\frac1{p(Y\mid\boldsymbol\theta)}\right]
=\int_{\boldsymbol\Theta}\frac1{p(Y\mid\boldsymbol\theta)}
 p(\boldsymbol\theta\mid Y)\,d\boldsymbol\theta
\tag{eq:harmonic_mean_expectation}
$$

它是式 {eq}`eq:posterior_expectation` 的一个特例，看起来颇具贝叶斯意味。展开后验项：

$$
\mathbb E\left[\frac1{p(Y\mid\boldsymbol\theta)}\right]
=\int_{\boldsymbol\Theta}\frac1{p(Y\mid\boldsymbol\theta)}
\frac{p(Y\mid\boldsymbol\theta)p(\boldsymbol\theta)}{p(Y)}
\,d\boldsymbol\theta
\tag{eq:harmonic_mean_approx1}
$$

化简为：

$$
\mathbb E\left[\frac1{p(Y\mid\boldsymbol\theta)}\right]
=\frac1{p(Y)}
\underbrace{\int_{\boldsymbol\Theta}p(\boldsymbol\theta)
\,d\boldsymbol\theta}_{=1}
=\frac1{p(Y)}
\tag{eq:harmonic_mean_approx2}
$$

这里假设先验是正则的（可归一化），因而积分为 1；所以第一式确实近似边缘似然。

坏消息很快出现：为接近正确答案，所需样本数通常大得令估计量在实践中几乎无用 {cite:p}`Neal_1994,Friel_2011`。和式会被似然极低的样本主导，而且估计量可能有无穷方差。无穷方差意味着增大 $s$ 也不保证改善答案，即使样本量巨大也可能远远不够。另一个问题是它对先验变化相当不敏感，而精确边缘似然实际上对先验非常敏感。当似然远比先验集中，或二者集中在参数空间不同区域时，这些问题都会加剧。

从比先验尖锐得多的后验抽样，会漏掉先验中后验密度很低的区域。粗略地说，贝叶斯推断用数据把先验更新为后验；只有当数据信息很少时，先验与后验才会相近。即便简单的一维 Beta–二项模型，调和均值估计也可能灾难性失败：

![调和均值估计量近似 Beta–二项模型边缘似然时的相对误差热图。各行对应不同先验，各列对应不同观测情景；括号内是成功数与失败数。](img/harmonic_mean_heatmap.png)

模型维数增大时，后验会越来越集中在薄超壳层中。近似后验时，抽取壳层外样本没有意义；但计算边缘似然时，只从这个薄壳层抽样又不够，需要覆盖整个先验分布，而正确完成这一点非常困难。

有些方法更适合计算边缘似然，但也不是万无一失。第八章讨论的序贯蒙特卡洛（SMC）主要用于近似贝叶斯计算，也能计算边缘似然。它之所以可行，是因为用一系列中间分布桥接先验与后验，从而缓解“从宽先验抽样、在高度集中后验中评价”的问题。
""",
    "2337-2442",
    anchor="harmonic_mean",
    labels=(
        "eq:harmonic_mean_approx", "eq:harmonic_mean_expectation",
        "eq:harmonic_mean_approx1", "eq:harmonic_mean_approx2",
        "fig:harmonic_mean_heatmap",
    ),
)

add_code(
    "harmonic-mean-stable",
    r"""
def beta_binomial_log_marginal(alpha, beta, successes, trials):
    '''解析计算 Beta–二项模型的对数边缘似然。'''
    if not (alpha > 0 and beta > 0 and 0 <= successes <= trials):
        raise ValueError("参数必须满足 alpha,beta>0 且 0<=successes<=trials")
    log_choose = (
        special.gammaln(trials + 1)
        - special.gammaln(successes + 1)
        - special.gammaln(trials - successes + 1)
    )
    return (
        log_choose
        + special.betaln(alpha + successes, beta + trials - successes)
        - special.betaln(alpha, beta)
    )


def beta_binomial_log_harmonic(
    alpha, beta, successes, trials, draws=MONTE_CARLO_DRAWS, seed=RANDOM_SEED + 31
):
    '''在对数空间中计算调和均值估计；仅用于展示其不稳定性。'''
    local_rng = np.random.default_rng(seed)
    posterior_draws = stats.beta(
        alpha + successes, beta + trials - successes
    ).rvs(size=draws, random_state=local_rng)
    log_likelihood = stats.binom.logpmf(successes, trials, posterior_draws)
    return -(special.logsumexp(-log_likelihood) - np.log(draws))

log_ml_exact = beta_binomial_log_marginal(2, 2, 5, 10)
log_ml_harmonic = beta_binomial_log_harmonic(2, 2, 5, 10)
assert np.isfinite(log_ml_exact) and np.isfinite(log_ml_harmonic)
assert abs(log_ml_harmonic - log_ml_exact) < 0.5
""",
    "2337-2442",
    migration_cells=(16, 17, 18, 19),
    modernization="修正迁移笔记中把对数似然倒数求平均的错误；使用 logsumexp 实现数值稳定的调和均值，并以宽松误差界作教学性烟雾断言。",
)

add_md(
    "marginal-likelihood-model-comparison",
    r"""
### 边缘似然与模型比较

推断时，边缘似然通常只是归一化常数，经常可以省略或约去；模型比较时，它却常被视为关键量 {cite:p}`Gronau2017,Navarro2020,Schad2021`。把贝叶斯定理写成显式依赖模型的形式：

$$
p(\boldsymbol\theta\mid Y,M)=
\frac{p(Y\mid\boldsymbol\theta,M)p(\boldsymbol\theta\mid M)}{p(Y\mid M)}
\tag{eq:bayes_theorem_M}
$$

其中 $Y$ 是数据，$\boldsymbol\theta$ 是模型 $M$ 的参数。

若有 $k$ 个模型且目标是只选一个，在这 $k$ 个模型具有离散均匀先验的假设下，选择边缘似然 $p(Y\mid M)$ 最大的模型可由贝叶斯定理直接得到：

$$
p(M\mid Y)\propto p(Y\mid M)p(M)
\tag{eq:posterior_model}
$$

若所有模型先验概率相同，比较 $p(Y\mid M)$ 就等价于比较 $p(M\mid Y)$。这里的 $p(M)$ 是赋给模型的先验概率，不是每个模型内部参数的先验 $p(\theta\mid M)$。

单个 $p(Y\mid M_k)$ 本身不容易解释，所以实践中常取两个边缘似然之比，即 Bayes 因子：

$$
\mathrm{BF}=\frac{p(Y\mid M_0)}{p(Y\mid M_1)}
\tag{eq:bayes_factor}
$$

$\mathrm{BF}>1$ 表示相对于 $M_1$，$M_0$ 更能解释数据。实践中常用经验规则区分“小”“大”等程度[^21]。

Bayes 因子很有吸引力，因为它直接应用贝叶斯定理；但调和均值估计量同样来自合法的贝叶斯恒等式，这并不自动保证它是好方法。另一个吸引点是：不同于最大化似然，边缘似然不会必然随模型复杂度增加。参数越多，先验相对于似然通常越分散；更分散的先验把更多数据集视为合理，因此在整个先验上平均似然后，会得到更小的边缘似然，形成内建复杂度惩罚。

除计算困难外，边缘似然还有一个常被视为缺陷的性质：它对先验选择**极其敏感**。即使先验变化对后验推断几乎无关紧要，也可能实质改变边缘似然。考虑模型：

$$
\begin{aligned}
\mu&\sim\mathcal N(0,\sigma_0),\\
Y&\sim\mathcal N(\mu,\sigma_1).
\end{aligned}
\tag{eq:normal_normal}
$$

这个模型的边缘对数似然可解析计算。单个观测 $y=0$ 且 $\sigma_0=\sigma_1=1$ 时，结果约为 $-1.2655$。把先验尺度 $\sigma_0$ 从 1 改成 2.5，边缘似然约缩小一半；改成 10 时约缩小到七分之一。下面的当前 API 示例同时构造 PyMC 模型，并在不运行 MCMC 的前提下用解析后验样本计算 ArviZ WAIC 与 PSIS-LOO。
""",
    "2443-2543",
    anchor="Bayes_factors",
    labels=(
        "eq:bayes_theorem_M", "eq:posterior_model", "eq:bayes_factor",
        "eq:normal_normal",
    ),
)

add_code(
    "normal-model-waic-loo",
    r"""
def make_normal_model(prior_sigma, observation_sigma, observations):
    '''只构造当前 PyMC 模型；本单元格不运行 MCMC。'''
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=prior_sigma)
        pm.Normal("y", mu=mu, sigma=observation_sigma, observed=observations)
    return model


def posterior_ml_ic_normal(
    prior_sigma, observation_sigma, observations, seed=RANDOM_SEED + 37
):
    '''用共轭解析后验和 ArviZ 公共 API 计算比较量。'''
    observations = np.asarray(observations, dtype=float)
    n_obs = observations.size
    posterior_variance = 1.0 / (
        1.0 / prior_sigma**2 + n_obs / observation_sigma**2
    )
    posterior_mean = (
        posterior_variance * observations.sum() / observation_sigma**2
    )
    posterior_sd = np.sqrt(posterior_variance)

    local_rng = np.random.default_rng(seed)
    mu_draws = local_rng.normal(
        posterior_mean, posterior_sd, size=(2, POSTERIOR_DRAWS)
    )
    log_likelihood = stats.norm.logpdf(
        observations[None, None, :],
        loc=mu_draws[:, :, None],
        scale=observation_sigma,
    )
    idata = az.from_dict(
        posterior={"mu": mu_draws},
        log_likelihood={"y": log_likelihood},
    )
    waic = az.waic(idata, var_name="y", scale="log", pointwise=True)
    loo = az.loo(idata, var_name="y", reff=1.0, scale="log", pointwise=True)

    covariance = (
        observation_sigma**2 * np.eye(n_obs)
        + prior_sigma**2 * np.ones((n_obs, n_obs))
    )
    log_marginal = stats.multivariate_normal.logpdf(
        observations, mean=np.zeros(n_obs), cov=covariance
    )
    return {
        "mean": posterior_mean,
        "sd": posterior_sd,
        "log_marginal": float(log_marginal),
        "waic": float(waic.elpd_waic),
        "loo": float(loo.elpd_loo),
        "idata": idata,
    }

normal_observations = np.array([
    0.65225338, -0.06122589, 0.27745188, 1.38026371, -0.72751008,
    -1.10323829, 2.07122286, -0.52652711, 0.51528113, 0.71297661,
])
normal_models = [
    make_normal_model(scale, 1.0, normal_observations)
    for scale in (1.0, 10.0, 100.0)
]
normal_comparisons = [
    posterior_ml_ic_normal(scale, 1.0, normal_observations, RANDOM_SEED + idx)
    for idx, scale in enumerate((1.0, 10.0, 100.0), start=40)
]

assert all({"mu", "y"}.issubset(model.named_vars) for model in normal_models)
# 中文版补充：不只检查变量名是否存在，还实际编译并在若干点上求值每个模型的
# logp，与手写的正态-正态解析对数密度比较，确认 PyMC 模型结构（尤其是先验/
# 似然尺度参数）与上面用于 WAIC/LOO/边缘似然计算的解析公式完全一致。
for model, scale in zip(normal_models, (1.0, 10.0, 100.0)):
    compiled_logp = model.compile_logp()
    for mu_probe in (0.0, 0.5, -0.3):
        point = {"mu": mu_probe}
        manual_logp = stats.norm.logpdf(mu_probe, loc=0.0, scale=scale) + float(
            stats.norm.logpdf(normal_observations, loc=mu_probe, scale=1.0).sum()
        )
        assert np.isclose(compiled_logp(point), manual_logp, atol=1e-6)
for result in normal_comparisons:
    assert result["sd"] > 0
    assert np.isfinite([result["log_marginal"], result["waic"], result["loo"]]).all()
    assert abs(result["waic"] - result["loo"]) < 5.0
assert normal_comparisons[0]["log_marginal"] > normal_comparisons[-1]["log_marginal"]
""",
    "2523-2600",
    migration_cells=(21, 22),
    modernization="迁移为 PyMC 5 模型构造和 ArviZ 公共 waic/loo API；用共轭解析后验确定性抽样，修正多观测共享均值模型的边缘似然协方差，完全避免 MCMC；另外实际编译各模型 logp 并与手写解析对数密度比对，而不只检查变量是否存在。",
    tags=("model-construction", "no-mcmc"),
)

add_md(
    "bayes-factor-vs-waic-loo",
    r"""
### Bayes 因子与 WAIC、LOO

本书不用 Bayes 因子比较模型，而更偏好 LOO。忽略细节，可以概括为：

- WAIC 是在后验上平均的对数似然；
- LOO 是在留一后验上平均的对数似然；
- 边缘似然是在先验上平均的（对数）似然[^22]。

三者都以对数评分衡量拟合，但计算方式不同。WAIC 使用从后验方差得到的显式惩罚项；LOO 与边缘似然都不需要显式惩罚。LOO 近似留一交叉验证，即用一份数据拟合、用不同数据评价。边缘似然的惩罚来自在整个先验上平均：先验相对于似然越分散，内建惩罚越强。它在某种程度上类似 WAIC 的惩罚，但 WAIC 使用后验方差，因而更接近交叉验证。因为更分散的先验把更多数据集视为合理，边缘似然也可理解为隐式地在先验容许的所有数据集上平均。

等价地，边缘似然就是在特定数据集 $Y$ 处评价先验预测分布，所以它说明完整模型——包括先验和似然——认为这组数据有多可能。

WAIC 与 LOO 中先验只通过后验间接起作用。数据相对先验越有信息，或者先验与后验差异越大，二者就越不敏感于先验细节。边缘似然则直接在先验上平均似然。从概念上说，Bayes 因子关注识别“最佳模型”（先验属于模型），WAIC 与 LOO 更关注哪个拟合后的模型及参数给出最佳预测。

下图展示式 {eq}`eq:normal_normal` 在 $\sigma_0=1,10,100$ 时的三个后验。它们很接近，尤其后两个；WAIC 与 LOO 只略微变化，而对数边缘似然对先验选择非常敏感。图中后验与边缘似然解析计算，WAIC 与 LOO 来自后验样本。

![式 eq:normal_normal 模型的先验（灰线）和后验（蓝线）。WAIC 与 LOO 反映三个后验几乎相同，而边缘似然反映先验差异。](img/ml_waic_loo.png)

这解释了为何一些领域广泛使用 Bayes 因子，另一些领域却不喜欢它。如果先验较接近某个底层“真实”模型，边缘似然对先验规范的敏感性不那么令人担忧；如果先验主要用于正则化，并尽可能承载少量背景知识，这种敏感性就可能成为问题。

因此，我们认为 WAIC、尤其是 LOO 更具实践价值：计算通常更稳健，不需要特殊推断方法，而且 LOO 还提供良好诊断。
""",
    "2544-2623",
    anchor="bayes-factor-vs-waic-and-loo",
    labels=("fig:posterior_ML",),
)

add_md(
    "moving-out-of-flatland",
    r"""
## 走出平面国

埃德温·阿博特的《平面国：多维空间传奇》{cite:p}`AbbottFlatland` 描述一个正方形生活在二维世界“平面国”：居民是多边形，边数决定地位；女性只是线段，祭司虽只是高阶多边形却坚称自己是圆。这部小说初版于 1884 年，既是社会讽刺，也描绘了理解超出日常经验之观念的困难。就像故事中的正方形一样，我们现在要面对高维空间的奇异之处。

> **译注与勘误。** 权威 Markdown 把小说初版年份误写为 1984；此处据作品史实修正为 1884。

假设要估计 $\pi$。在正方形内切一个圆，在正方形中均匀生成 $N$ 个点，再计算落入圆内的比例。严格说，这是蒙特卡洛积分：用（伪）随机数生成器计算定积分。

圆与正方形面积之比，对应圆内点数与总点数之比。若正方形边长为 $2R$，面积是 $(2R)^2$，内切圆面积是 $\pi R^2$，于是：

$$
\frac{\text{inside}}N\propto\frac{\pi R^2}{(2R)^2}
\tag{eq:circ_mc}
$$

整理后得到：

$$
\hat\pi=4\frac{\operatorname{Count}_{\mathrm{inside}}}{N}
\tag{eq:pi_mc}
$$
""",
    "2624-2668",
    anchor="high_dimensions",
    labels=("eq:circ_mc", "eq:pi_mc"),
    provenance="translation-with-correction",
    modernization="把源文误写的《平面国》初版年份 1984 修正为 1884，并显式标注。",
)

add_code(
    "monte-carlo-pi",
    r"""
mc_rng = np.random.default_rng(RANDOM_SEED + 41)
xy = mc_rng.uniform(-1.0, 1.0, size=(2, MONTE_CARLO_DRAWS))
inside_circle = np.square(xy).sum(axis=0) <= 1.0
pi_estimate = 4.0 * inside_circle.mean()
pi_relative_error = abs(pi_estimate - np.pi) / np.pi

assert 2.8 < pi_estimate < 3.5
assert pi_relative_error < 0.12
""",
    "2669-2678",
    labels=("montecarlo",),
    migration_cells=(24,),
    modernization="使用独立 numpy Generator 与 smoke/release 样本预算，并加入宽松、确定性的估计误差断言。",
)

add_md(
    "monte-carlo-geometry",
    r"""
![用蒙特卡洛样本估计 π；图例给出估计值与百分比误差。](img/monte_carlo.png)

因为抽样独立同分布，可应用中心极限定理，误差按 $1/\sqrt N$ 的速率下降；每多获得一位十进制精度，抽取数 $N$ 大约要增大 100 倍。

这就是蒙特卡洛方法[^23]的例子：泛指用（伪）随机样本进行计算的方法。更具体地，这里用样本计算定积分（面积），所以是蒙特卡洛积分。蒙特卡洛方法在统计学中无处不在。

贝叶斯统计需要积分来得到后验或计算后验期望。遗憾的是，把上述朴素方法推广到更有趣的问题后，维度增加时通常表现很差。下面从 2 到 14 维，在超立方体中抽样并计算落入内切超球的点数比例。即使超球“接触”超立方体各面，内部点比例仍随维度迅速下降；某种意义上，高维超立方体几乎所有体积都在角落[^24]。
""",
    "2680-2712",
    labels=("fig:monte_carlo",),
)

add_code(
    "inside-out",
    r"""
highdim_rng = np.random.default_rng(RANDOM_SEED + 43)
dimensions = np.arange(2, 15)
inside_proportions = []
for dimension in dimensions:
    samples = highdim_rng.random(size=(dimension, HIGHDIM_DRAWS))
    inside_proportions.append(np.mean(np.square(samples).sum(axis=0) < 1.0))
inside_proportions = np.asarray(inside_proportions)

assert np.all((0.0 <= inside_proportions) & (inside_proportions <= 1.0))
assert inside_proportions[0] > 0.70
assert inside_proportions[-1] < 0.01
assert inside_proportions[0] > inside_proportions[5] > inside_proportions[-1]
""",
    "2713-2726",
    labels=("inside_out",),
    migration_cells=(26,),
    modernization="使用 profile 控制高维样本预算和独立 Generator，并对几何趋势作有界而非逐点单调断言。",
)

add_md(
    "typical-set",
    r"""
![维度增加时，在超立方体内均匀抽到内切超球内部点的概率趋近 0，说明高维超立方体几乎所有体积都位于角落。](img/inside_out.png)

多元高斯也展示同样的高维反直觉性质。下图说明：维度越高，大多数概率质量离众数越远；事实上，它主要集中在距众数约 $\sqrt d$ 的“环壳”附近。维度增加时，众数越来越不典型；在高维中，同时也是均值的众数实际上可能像一个异常值，因为一个点在所有维度上都恰好平均极不寻常。

换个角度看，众数始终是密度最高的单点，但它只有一个。远离众数的各点密度分别较低，数量却极多。连续随机变量的概率是密度在体积上的积分，所以寻找概率质量所在区域时，必须同时平衡密度与体积。高维高斯中，最可能抽到的是排除众数的环壳区域。

包含概率分布大部分质量的空间区域称为**典型集**。贝叶斯统计很关心典型集：用样本近似高维后验时，只要样本来自典型集即可。

![高斯分布维度增加时，大部分概率质量分布在离众数越来越远的位置。](img/distance_to_mode.png)
""",
    "2728-2769",
    labels=("fig:inside_out", "fig:distance_to_mode"),
)

add_md(
    "inference-methods",
    r"""
## 推断方法

计算后验的方法多种多样。若排除第一章讨论共轭先验时已经见过的精确解析解，大体可分为三类：

1. 确定性积分方法，本书此前尚未介绍，下一节就会看到；
2. 模拟方法，第一章已经引入，也是全书主要采用的方法；
3. 近似方法，例如第八章讨论的 ABC，适用于似然没有闭式表达的情形。

有些方法横跨多个类别，但这一分类仍有助于整理众多可用方法。若希望按时间顺序了解过去两个半世纪的贝叶斯计算，尤其是那些改变了贝叶斯推断的方法，推荐阅读 *Computing Bayes: Bayesian Computation from 1763 to the 21st Century* {cite:p}`Martin2020`。
""",
    "2770-2796",
    anchor="inference_methods",
)

add_md(
    "grid-method",
    r"""
### 网格法

网格法是一种简单的穷举方法。为了使用后验分布——寻找最大值、计算期望等——我们希望知道它在定义域上的值。即使不能整体求出后验，往往仍能逐点评价先验和似然密度。对单参数模型，网格近似步骤是：

1. 为参数寻找合理区间，先验通常能提供线索；
2. 在区间上定义一组通常等距的网格点；
3. 在每个网格点把似然乘以先验；可选地，再除以所有点之和，使离散近似后验总和为 1。

下面计算 Beta–二项模型的网格后验。
""",
    "2797-2821",
    anchor="grid-method",
)

add_code(
    "posterior-grid",
    r"""
def posterior_grid(ngrid=10, alpha=1.0, beta=1.0, heads=6, trials=9):
    '''以数值稳定的网格法近似 Beta–二项后验。'''
    if ngrid < 2 or alpha <= 0 or beta <= 0 or not (0 <= heads <= trials):
        raise ValueError("网格数、先验参数或观测计数无效")
    grid = (np.arange(ngrid, dtype=float) + 0.5) / ngrid
    log_unnormalized = (
        stats.beta.logpdf(grid, alpha, beta)
        + stats.binom.logpmf(heads, trials, grid)
    )
    posterior = np.exp(log_unnormalized - special.logsumexp(log_unnormalized))
    return grid, posterior

grid, grid_posterior = posterior_grid(ngrid=100)
assert np.isclose(grid_posterior.sum(), 1.0)
assert np.all(np.isfinite(grid_posterior))
assert 0.5 < grid[np.argmax(grid_posterior)] < 0.8
""",
    "2822-2833",
    labels=("grid_method",),
    migration_cells=(27,),
    modernization="使用单元中心避免端点奇异性，在对数空间归一化，并返回网格以便正确解释结果。",
)

add_md(
    "grid-method-result",
    r"""
![在网格上逐点评价先验与似然，从而近似后验。](img/grid_method.png)

增加网格点数可以改善近似；若使用无限多个点，理论上会得到精确后验，代价是无限计算资源。网格法最大的限制，是它会像“走出平面国”一节所述那样随参数个数迅速失去可扩展性。
""",
    "2835-2848",
    labels=("fig:grid_method",),
)

add_md(
    "metropolis-hastings",
    r"""
### Metropolis–Hastings

本书在 {ref}`sampling_methods_intro` 一节很早就介绍了 Metropolis–Hastings 算法 {cite:p}`Metropolis1953,Hastings1970,Rosenbluth2003`，并在代码块 `metropolis_hastings` 中给出简单 Python 实现。这里用前面介绍的马尔可夫链语言，更详细地说明它为何成立。

Metropolis–Hastings 是一种通用方法：从目标状态空间上任意不可约马尔可夫链出发，把它修改成以目标分布为平稳分布的新链。也就是说，从容易抽样的分布（如多元正态）提出样本，再有选择地接受其中一些、拒绝另一些，使结果成为目标分布的样本。新提议的接受概率为：

$$
p_a(x_{i+1}\mid x_i)=\min\left(1,
\frac{p(x_{i+1})q(x_i\mid x_{i+1})}
{p(x_i)q(x_{i+1}\mid x_i)}\right)
$$

简写为：

$$
a_{ij}=\min\left(1,\frac{p_jq_{ji}}{p_iq_{ij}}\right)
\tag{eq:acceptance_prob1}
$$

我们以概率 $q_{ij}$ 从 $i$ 提议到 $j$，再以概率 $a_{ij}$ 接受。一个重要优点是，目标分布的归一化常数在 $p_j/p_i$ 中约去；贝叶斯推断等许多问题的归一化常数（边缘似然）都很难计算。

下面证明该链可逆，并以 $p$ 为平稳分布。设 $\mathbf T$ 是转移矩阵，只需证明所有 $i,j$ 都满足细致平衡 $p_it_{ij}=p_jt_{ji}$。$i=j$ 时显然成立，故设 $i\ne j$：

$$
t_{ij}=q_{ij}a_{ij}
\tag{eq:transition}
$$

即从 $i$ 到 $j$ 的转移概率等于提出该移动的概率乘以接受概率。先看接受概率小于 1 的情形，即 $p_jq_{ji}\le p_iq_{ij}$。此时

$$
a_{ij}=\frac{p_jq_{ji}}{p_iq_{ij}}
\tag{eq:acceptance_ij}
$$

且

$$
a_{ji}=1.
\tag{eq:acceptance_ji}
$$

由式 {eq}`eq:transition`：

$$
p_it_{ij}=p_iq_{ij}a_{ij}
\tag{eq:transition2}
$$

代入 $a_{ij}$：

$$
p_it_{ij}=p_iq_{ij}\frac{p_jq_{ji}}{p_iq_{ij}}
\tag{eq:transition3}
$$

化简：

$$
p_it_{ij}=p_jq_{ji}
\tag{eq:transition4}
$$

因为 $a_{ji}=1$，可写成：

$$
p_it_{ij}=p_jq_{ji}a_{ji}
\tag{eq:transition5}
$$

最终得到：

$$
p_it_{ij}=p_jt_{ji}.
\tag{eq:transition6}
$$

当 $p_jq_{ji}>p_iq_{ij}$ 时，由对称性得到同一结果。因此可逆条件成立，$p$ 是转移矩阵 $\mathbf T$ 对应链的平稳分布。

这给出理论保证：Metropolis–Hastings 原则上几乎能从任意分布抽样，却没有告诉我们如何选择提议分布。实践中效率高度依赖提议：跳得太远，接受概率很低，链长时间拒绝并停在原处；跳得太近，接受率虽高，探索却局限于旧状态附近。理想提议既能到达远处又有高接受率，但在未知后验几何时很难做到，而后验几何恰好又是我们想了解的对象。

实用的 Metropolis–Hastings 方法通常具有自适应机制 {cite:p}`Haario2001,Andrieu2008,Roberts2009,Sejdinovic2014`。例如，可用多元高斯作提议，在调优阶段从后验样本估计经验协方差，再调整其尺度以趋近预设平均接受率 {cite:p}`Roberts1997,Roberts2001,Bedard2008`。有研究表明，在特定条件和高维极限下，最优接受率趋近 0.234 {cite:p}`Roberts1997`；实践中 0.234 附近或略高通常表现相似，但这一结果的一般有效性与实用性也受到质疑 {cite:p}`Sherlock2013,Potter2015`。

下一节讨论一种更巧妙的提议机制，以缓解基本 Metropolis–Hastings 的大部分问题。
""",
    "2849-2997",
    anchor="sec_metropolis_hastings",
    labels=(
        "eq:acceptance_prob1", "eq:transition", "eq:acceptance_ij",
        "eq:acceptance_ji", "eq:transition2", "eq:transition3",
        "eq:transition4", "eq:transition5", "eq:transition6",
    ),
)

add_md(
    "hamiltonian-monte-carlo",
    r"""
### 哈密顿蒙特卡洛

哈密顿蒙特卡洛（HMC）[^25] {cite:p}`Duane1987,Brooks2011,Betancourt2017` 是利用梯度生成新提议状态的 MCMC 方法。在某个状态评价后验对数概率的梯度，会提供后验密度几何的信息。HMC 借助梯度，从当前位置提出相距较远但接受概率仍高的新位置，从而避免 Metropolis–Hastings 常见的随机游走行为，也更能扩展到高维和复杂几何。

简单说，哈密顿量描述物理系统的总能量，可分为动能与势能。以小球滚下山坡为例，位置越高，势能越大；动能由速度——更准确说是兼顾速度与质量的动量——决定。假设总能量守恒，动能增加就对应同量势能减少：

$$
H(\mathbf q,\mathbf p)=K(\mathbf p,\mathbf q)+V(\mathbf q)
\tag{eq:hamiltonian}
$$

$K$ 是动能，$V$ 是势能。系统出现在特定位置和动量的密度与下式成正比：

$$
p(\mathbf q,\mathbf p)\propto e^{-H(\mathbf q,\mathbf p)}
\tag{eq:canonical}
$$

模拟系统需要求解哈密顿方程：

$$
\begin{aligned}
\frac{d\mathbf q}{dt}
&=\frac{\partial H}{\partial\mathbf p}
=\frac{\partial K}{\partial\mathbf p}+
  \frac{\partial V}{\partial\mathbf p},\\
\frac{d\mathbf p}{dt}
&=-\frac{\partial H}{\partial\mathbf q}
=-\frac{\partial K}{\partial\mathbf q}-
  \frac{\partial V}{\partial\mathbf q}.
\end{aligned}
\tag{eq:hamiltonian_equations}
$$

注意 $\partial V/\partial\mathbf p=\mathbf0$。

我们关心的不是理想小球，而是沿后验分布运动的理想粒子。势能由目标密度 $p(\mathbf q)$ 决定；动量则引入一个辅助变量 $\mathbf p$。若选择 $p(\mathbf p\mid\mathbf q)$，则

$$
p(\mathbf q,\mathbf p)=p(\mathbf p\mid\mathbf q)p(\mathbf q)
\tag{eq:auxiliary}
$$

边缘化动量便能恢复目标分布。把它代入式 {eq}`eq:canonical`：

$$
H(\mathbf q,\mathbf p)=
\overbrace{-\log p(\mathbf p\mid\mathbf q)}^{K(\mathbf p,\mathbf q)}
+\overbrace{-\log p(\mathbf q)}^{V(\mathbf q)}
\tag{eq:hamiltonian_KV}
$$

势能由目标后验密度给定，动能可以选择。若令动量为高斯分布，则在保留与 $\mathbf p$ 无关的项时：

$$
K(\mathbf p,\mathbf q)=
\frac12\mathbf p^T M^{-1}\mathbf p+
\frac12\log|M|+\mathrm{const}
\tag{eq:kinetic_energy}
$$

这里 $M$ 是 HMC 文献中的**质量矩阵**，也可视为动量高斯的协方差矩阵；$M^{-1}$ 才是精度矩阵。若选 $M=I$ 并略去常数：

$$
K(\mathbf p,\mathbf q)=\frac12\mathbf p^T\mathbf p
\tag{eq:kinetic_energy2}
$$

于是

$$
\frac{\partial K}{\partial\mathbf p}=\mathbf p
\tag{eq:kinetic_momentum}
$$

且

$$
\frac{\partial K}{\partial\mathbf q}=\mathbf0.
\tag{eq:kinetic_position}
$$

哈密顿方程简化为：

$$
\begin{aligned}
\frac{d\mathbf q}{dt}&=\mathbf p,\\
\frac{d\mathbf p}{dt}&=-\frac{\partial V}{\partial\mathbf q}.
\end{aligned}
\tag{eq:hamiltonian_equations2}
$$

HMC 算法可概括为：

1. 抽取 $\mathbf p\sim\mathcal N(0,I)$；
2. 把 $\mathbf q_t,\mathbf p_t$ 模拟一段时间 $T$；
3. 以 $\mathbf q_T$ 作为新提议；
4. 用 Metropolis 接受准则决定接受还是拒绝。

仍需 Metropolis 准则，一方面因为 HMC 可视为采用更好提议的 Metropolis–Hastings；另一方面，它能纠正数值求解哈密顿方程引入的离散化误差。

> **数学勘误。** 权威 Markdown 把 $M$ 称为精度矩阵，并漏掉高斯归一化项的系数 $1/2$；这里按现代 HMC 常用约定改称质量（协方差）矩阵，并修正式 {eq}`eq:kinetic_energy`。式 {eq}`eq:canonical` 也改用正比号以显式容纳归一化常数。
""",
    "2998-3139",
    anchor="hmc",
    labels=(
        "eq:hamiltonian", "eq:canonical", "eq:hamiltonian_equations",
        "eq:auxiliary", "eq:hamiltonian_KV", "eq:kinetic_energy",
        "eq:kinetic_energy2", "eq:kinetic_momentum", "eq:kinetic_position",
        "eq:hamiltonian_equations2",
    ),
    provenance="translation-with-correction",
    modernization="修正质量/精度矩阵术语、动量高斯归一化系数与未归一化密度等号，并显式标注勘误。",
)

add_md(
    "leapfrog-explanation",
    r"""
计算哈密顿轨迹需要在一个状态与下一个状态之间执行一系列小积分步。最常用的数值积分器是蛙跳法：位置与动量在交错的时间点更新，仿佛彼此“蛙跳”。

下面实现蛙跳积分器[^26]。`position` 与 `momentum` 是初始位置和动量；`gradient_potential` 返回目标势能在当前位置的梯度 $\partial V/\partial\mathbf q$；`path_length` 是积分总长度，`step_size` 是每步大小。源文借助 JAX 自动微分生成梯度 {cite:p}`jax2018github`；中文版把梯度函数显式作为参数传入，因此不依赖尚未发布锁定环境的可选 JAX 栈。
""",
    "3140-3158",
)

add_code(
    "leapfrog",
    r"""
def leapfrog(position, momentum, gradient_potential, path_length, step_size):
    '''用单位质量矩阵执行可逆、保体积的蛙跳积分。'''
    if path_length <= 0 or step_size <= 0:
        raise ValueError("path_length 与 step_size 必须为正")
    n_steps = max(1, int(np.ceil(path_length / step_size)))
    epsilon = path_length / n_steps
    q = np.asarray(position, dtype=float).copy()
    p = np.asarray(momentum, dtype=float).copy()

    p -= 0.5 * epsilon * np.asarray(gradient_potential(q))
    for step in range(n_steps):
        q += epsilon * p
        if step != n_steps - 1:
            p -= epsilon * np.asarray(gradient_potential(q))
    p -= 0.5 * epsilon * np.asarray(gradient_potential(q))
    return q, -p

# 只验证一条标准高斯轨迹的近似能量守恒；不运行 MCMC。
q0 = np.array([0.4, -0.7])
p0 = np.array([0.3, 0.2])
q1, p1 = leapfrog(q0, p0, lambda q: q, path_length=0.5, step_size=0.05)
energy0 = 0.5 * (q0 @ q0 + p0 @ p0)
energy1 = 0.5 * (q1 @ q1 + p1 @ p1)
assert q1.shape == q0.shape and p1.shape == p0.shape
assert abs(energy1 - energy0) < 0.01
""",
    "3159-3177",
    labels=("leapfrog",),
    modernization="移除未锁定 JAX 自动微分依赖，显式传入梯度；复制输入避免原地副作用，并只做单轨迹能量守恒检查。",
    tags=("lightweight-check", "no-mcmc"),
)

add_md(
    "hmc-implementation-explanation",
    r"""
函数末尾翻转输出动量符号，这是让数值积分提议可逆的最简单方式，相当于给积分补上负向步骤。

现在具备实现教学版 HMC 的全部要素。下面函数的参数为：返回样本数 `n_samples`、目标负对数概率 `negative_log_prob`、其梯度 `gradient_potential`、初始位置、路径长度和步长。与第一章的基础 Metropolis–Hastings 实现一样，它只用于展示原理，不应替代成熟 PPL 的采样器。
""",
    "3174-3188",
)

add_code(
    "hamiltonian-monte-carlo-code",
    r"""
def hamiltonian_monte_carlo(
    n_samples,
    negative_log_prob,
    gradient_potential,
    initial_position,
    path_length,
    step_size,
    seed=RANDOM_SEED + 53,
):
    '''教学版 HMC；生产推断应使用经过调优与诊断的 PPL 实现。'''
    if n_samples <= 0:
        raise ValueError("n_samples 必须为正")
    local_rng = np.random.default_rng(seed)
    current = np.asarray(initial_position, dtype=float).copy()
    samples = []
    accepted = 0
    for _ in range(n_samples):
        momentum0 = local_rng.normal(size=current.shape)
        proposal, momentum1 = leapfrog(
            current, momentum0, gradient_potential, path_length, step_size
        )
        energy0 = negative_log_prob(current) + 0.5 * np.square(momentum0).sum()
        energy1 = negative_log_prob(proposal) + 0.5 * np.square(momentum1).sum()
        if np.log(local_rng.uniform()) < energy0 - energy1:
            current = proposal
            accepted += 1
        samples.append(current.copy())
    return np.asarray(samples), accepted / n_samples
""",
    "3189-3223",
    labels=("hamiltonian_mc",),
    modernization="以显式梯度和 numpy Generator 取代 JAX/global RNG，修正数组别名风险，并返回接受率；定义函数但验证阶段不运行采样。",
    tags=("definition-only", "no-validation-execution"),
)

add_code(
    "hamiltonian-monte-carlo-demo",
    r"""
# 中文版补充：教学版 hamiltonian_monte_carlo 此前只定义、从未实际运行；这里用与
# leapfrog 演示相同的标准二维正态目标（负对数概率 0.5*q@q，梯度 q）轻量运行几步，
# 展示函数确实可用，而不只是停留在定义阶段。样本数按 smoke/release 预算区分，
# 但两种配置走同一条代码路径。
hmc_demo_samples, hmc_demo_accept_rate = hamiltonian_monte_carlo(
    n_samples=HMC_DEMO_SAMPLES,
    negative_log_prob=lambda q: 0.5 * float(q @ q),
    gradient_potential=lambda q: q,
    initial_position=np.array([0.4, -0.7]),
    path_length=0.5,
    step_size=0.05,
)
assert hmc_demo_samples.shape == (HMC_DEMO_SAMPLES, 2)
assert np.all(np.isfinite(hmc_demo_samples))
assert 0.0 <= hmc_demo_accept_rate <= 1.0
print(f"教学版 HMC 接受率：{hmc_demo_accept_rate:.2f}")
""",
    "中文版补充：hamiltonian_monte_carlo 的最小可运行演示",
    labels=("hamiltonian_mc_demo",),
    provenance="addition",
    modernization="新增对 hamiltonian_monte_carlo 的实际调用，验证函数可运行而不只是定义。",
    tags=("lightweight-check",),
)

add_md(
    "hmc-trajectories-tuning",
    r"""
下图展示同一个二维正态分布周围的三条轨迹。实际抽样不希望轨迹绕一圈回到起点，而希望尽可能远离起点，例如避免轨迹掉头；最常用的动态 HMC 方法之一因此得名“无掉头采样器”（NUTS）。

![二维多元正态周围的三条 HMC 轨迹。箭头大小与方向表示动量，小箭头表示较低动能。这些示例都完成椭圆轨迹并回到起点。](img/normal_leapfrog.png)

下图展示 Neal 漏斗周围的三条轨迹。这类几何常见于居中参数化的层级模型，图中数值轨迹都未能正确追随目标分布，称为发散轨迹或简称发散。发散是很有用的诊断，见 {ref}`divergences`。蛙跳等辛积分器即使在长轨迹上通常也很准确：小误差倾向于围绕正确轨迹振荡，并可由 Metropolis 接受步骤精确纠正。但当精确轨迹位于高曲率区域时，数值轨迹可能迅速偏向目标分布边界，产生无法轻易纠正的发散。

![二维 Neal 漏斗周围的三条 HMC 轨迹；这种几何常见于居中层级模型。三条轨迹都发生错误，这类发散可作为 HMC 采样器诊断。](img/funnel_leapfrog.png)

两图强调：高效 HMC 需要恰当调优三个超参数：

- 时间离散化，即蛙跳步长；
- 积分时间，即蛙跳步数；
- 参数化动能的质量矩阵 $M$。

步长过大，蛙跳不准确且大量提议被拒；过小则浪费计算。步数过少，轨迹太短，抽样退化为随机游走；过多则可能绕圈并浪费资源。若估计协方差（质量矩阵）与后验协方差差异太大，各维的位置移动就可能过大或过小。

PyMC、Stan 等 PPL 默认使用的自适应动态 HMC，可在预热或调优阶段自动适配这些量。步长通过逼近预设接受率目标来学习；在当前 PyMC 中可向采样接口传递 `target_accept`[^27]。质量矩阵可从预热样本估计，NUTS 则在每个 MCMC 步动态选择积分长度 {cite:p}`Hoffman2014`。为避免轨迹过长又靠近起点，NUTS 双向扩展轨迹直到满足掉头准则，并从生成的轨迹点中进行多项式抽样，以提升目标探索效率。
""",
    "3225-3307",
    labels=("fig:normal_leapgrog", "fig:funnel_leapgrog"),
    modernization="把 PyMC3 更新为 PyMC，并按前文修正后的质量矩阵术语描述调优。",
)

add_md(
    "sequential-monte-carlo",
    r"""
### 序贯蒙特卡洛

序贯蒙特卡洛（SMC）是一族也称粒子滤波器的蒙特卡洛方法，广泛用于静态模型和动态模型的贝叶斯推断，例如序列时间序列推断与信号处理 {cite:p}`delmoral2006,Ching2007,Naesseth2019,Chopin2020`。同名或近似名称下有许多变体和实现，文献有时会令人困惑。这里简述 PyMC 与 TFP 所采用的一类 SMC/SMC-ABC 方法；统一框架下的详尽讨论可参见 *An Introduction to Sequential Monte Carlo* {cite:p}`Chopin2020`。

后验可写为幂后验：

$$
p(\boldsymbol\theta\mid Y)_\beta\propto
p(Y\mid\boldsymbol\theta)^\beta p(\boldsymbol\theta)
\tag{eq:powered_posterior}
$$

$\beta=0$ 时它就是先验，$\beta=1$ 时则是“真正”的后验[^28]。SMC 在 $s$ 个连续阶段中增大 $\beta$：
$\{\beta_0=0<\beta_1<\cdots<\beta_s=1\}$。

可用两种相关类比理解。第一是踏脚石：不直接从后验抽样，而从通常更容易的先验开始，再加入若干中间分布，直到后验。第二是温度：$\beta$ 类似物理系统的逆温度；$\beta$ 较小时系统可访问更多状态，逐渐把 $\beta$ 提高到 1，则系统“冷却”并冻结到后验[^29]。把温度或逆温度作为辅助参数称为温度化，也常称退火[^30]。

PyMC 与 TFP 中的这类 SMC 可概括为：

1. 初始化 $\beta=0$；
2. 从温度化后验生成 $N$ 个样本 $s_\beta$；
3. 增大 $\beta$，使有效样本量[^31]保持在预设水平；
4. 根据新旧温度化后验计算 $N$ 个重要性权重 $W$；
5. 按 $W$ 对 $s_\beta$ 重采样，得到 $s_w$；
6. 从 $s_w$ 的不同样本出发，运行 $N$ 条、每条 $k$ 步的 MCMC 链，只保留最后一步；
7. 从第 3 步重复，直到 $\beta=1$。

重采样会移除低概率粒子，用高概率粒子替代，因而降低样本多样性；随后的 MCMC 步扰动粒子，希望恢复多样性并帮助探索参数空间。任何有效 MCMC 转移核都可用于 SMC，不同问题适用的方法不同。ABC 模拟器通常不可微，所以一般依赖随机游走 Metropolis–Hastings 等无梯度方法。

温度化方法的效率高度依赖中间 $\beta$ 值。连续 $\beta$ 差越小，相邻温度化后验越接近、阶段转换越容易；但步长过小会需要大量中间阶段，超过一定程度只浪费计算而不改善精度。另一个关键因素是增加粒子多样性的 MCMC 转移核效率。PyMC 与 TFP 会利用上一阶段样本调节当前阶段的提议分布和 MCMC 步数，各粒子链使用相同步数。
""",
    "3308-3396",
    anchor="smc_details",
    labels=("eq:powered_posterior",),
    modernization="把历史实现名称 PyMC3 更新为当前 PyMC；不在本章验证中执行 SMC。",
)

add_md(
    "variational-inference",
    r"""
### 变分推断

虽然本书主体不使用变分推断（VI），了解它仍很有价值。与 MCMC 相比，VI 往往更容易扩展到大数据，计算也更快，但收敛的理论保证较弱 {cite:p}`yao2018yes`。

前面已经看到，可以用一个分布近似另一个分布，并用 KL 散度衡量近似质量。变分推断把这一思想用于贝叶斯推断 {cite:p}`Blei_2017`：用替代分布 $q(\boldsymbol\theta)$ 近似目标后验 $p(\boldsymbol\theta\mid Y)$。通常选择比目标更简单的分布族，再通过优化找到该族中 KL 散度意义下最接近目标的成员。稍改写式 {eq}`eq:kl_divergence`：

$$
\mathbb{KL}\left(q(\boldsymbol\theta)\parallel
p(\boldsymbol\theta\mid Y)\right)
=\mathbb E_q\left[\log q(\boldsymbol\theta)-
\log p(\boldsymbol\theta\mid Y)\right]
\tag{eq:kl_divergence2}
$$

这个目标难以直接计算，因为它需要边缘似然 $p(Y)$。展开可见：

$$
\begin{aligned}
\mathbb{KL}\left(q(\boldsymbol\theta)\parallel
p(\boldsymbol\theta\mid Y)\right)
&=\mathbb E[\log q(\boldsymbol\theta)]-
  \mathbb E[\log p(\boldsymbol\theta\mid Y)]\\
&=\mathbb E[\log q(\boldsymbol\theta)]-
  \mathbb E[\log p(\boldsymbol\theta,Y)]+\log p(Y).
\end{aligned}
\tag{eq:kl_divergence2_expanded}
$$

幸运的是，$\log p(Y)$ 对 $q$ 是常数，优化时可以省略。因此实践中最大化证据下界（ELBO），等价于最小化 KL 散度：

$$
\operatorname{ELBO}(q)=
\mathbb E[\log p(\boldsymbol\theta,Y)]-
\mathbb E[\log q(\boldsymbol\theta)]
\tag{eq:elbo_vi}
$$

式中期望不必通过昂贵积分求解，可以从替代分布 $q$ 抽取蒙特卡洛样本并取平均。

VI 表现取决于许多因素，包括替代分布族。更具表达力的替代分布能捕捉目标后验各分量间更复杂、非线性的依赖，通常给出更好近似。自动选择合适分布族并高效优化仍是活跃研究领域。下面用 TFP 的公开 API，以均值场高斯和满秩高斯近似二维香蕉形目标。
""",
    "3397-3466",
    anchor="vi_details",
    labels=("eq:kl_divergence2", "eq:kl_divergence2_expanded", "eq:elbo_vi"),
)

add_code(
    "tfp-public-surrogates",
    r"""
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

tf.keras.utils.set_random_seed(RANDOM_SEED)
try:
    tf.config.experimental.enable_op_determinism()
except (AttributeError, RuntimeError):
    pass


def banana_target_log_prob(theta):
    '''二维香蕉形未归一化目标的对数密度。'''
    theta = tf.convert_to_tensor(theta, dtype=tf.float32)
    x = theta[..., 0]
    y = theta[..., 1]
    return -tf.square(1.0 - x) - 1.5 * tf.square(y - tf.square(x))


mean_field_loc = tf.Variable(tf.zeros(2), name="mean_field_loc")
mean_field_scale = tfp.util.TransformedVariable(
    tf.ones(2), bijector=tfb.Softplus(), name="mean_field_scale"
)
mean_field_surrogate = tfd.MultivariateNormalDiag(
    loc=mean_field_loc, scale_diag=mean_field_scale
)

full_rank_loc = tf.Variable(tf.zeros(2), name="full_rank_loc")
full_rank_scale = tfp.util.TransformedVariable(
    tf.eye(2),
    bijector=tfb.FillScaleTriL(
        diag_bijector=tfb.Softplus(), diag_shift=tf.constant(1e-4, tf.float32)
    ),
    name="full_rank_scale",
)
full_rank_surrogate = tfd.MultivariateNormalTriL(
    loc=full_rank_loc, scale_tril=full_rank_scale
)

assert mean_field_surrogate.event_shape == tf.TensorShape([2])
assert full_rank_surrogate.event_shape == tf.TensorShape([2])
assert mean_field_surrogate.trainable_variables
assert full_rank_surrogate.trainable_variables
""",
    "3467-3492",
    migration_cells=(29, 30),
    modernization="用稳定的向量事件和公共 MultivariateNormalDiag/TriL、TransformedVariable、Softplus、FillScaleTriL API 替代 experimental.vi 构造器。",
    tags=("tfp", "public-api", "construction"),
)

add_code(
    "tfp-fit-gaussian-surrogates",
    r"""
def fit_surrogate_checked(surrogate, steps, seed):
    '''确定性优化替代分布，并作有界数值与近似行为检查。'''
    losses = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn=banana_target_log_prob,
        surrogate_posterior=surrogate,
        optimizer=tf.optimizers.Adam(learning_rate=0.03),
        num_steps=steps,
        sample_size=VI_SAMPLE_SIZE,
        trainable_variables=surrogate.trainable_variables,
        seed=seed,
    )
    samples = surrogate.sample(VI_DRAWS, seed=seed + 1)
    losses_np = np.asarray(losses)
    samples_np = np.asarray(samples)
    energy_np = np.asarray(-banana_target_log_prob(samples))

    assert losses_np.shape == (steps,)
    assert np.isfinite(losses_np).all()
    window = max(5, steps // 5)
    first_loss = float(np.median(losses_np[:window]))
    last_loss = float(np.median(losses_np[-window:]))
    assert last_loss <= first_loss + max(10.0, 0.5 * abs(first_loss))
    assert samples_np.shape == (VI_DRAWS, 2)
    assert np.isfinite(samples_np).all() and np.isfinite(energy_np).all()
    assert np.all(np.abs(samples_np.mean(axis=0)) < 5.0)
    assert np.all((samples_np.std(axis=0) > 0.02) & (samples_np.std(axis=0) < 6.0))
    assert np.median(energy_np) < 20.0
    return losses_np, samples_np, energy_np


gaussian_vi_results = {}
for offset, (name, surrogate) in enumerate(
    (("mean-field", mean_field_surrogate), ("full-rank", full_rank_surrogate)),
    start=61,
):
    gaussian_vi_results[name] = fit_surrogate_checked(
        surrogate, VI_STEPS, RANDOM_SEED + offset
    )

mean_field_energy = np.median(gaussian_vi_results["mean-field"][2])
full_rank_energy = np.median(gaussian_vi_results["full-rank"][2])
assert full_rank_energy <= mean_field_energy + 10.0
""",
    "3467-3492",
    labels=("vi_in_tfp",),
    migration_cells=(30, 31, 33),
    modernization="采用 fit_surrogate_posterior 公共 API、profile 优化预算和显式 trainable_variables；断言有限损失、尾段趋势、样本形状/尺度与有界目标能量，不要求随机损失逐步单调。",
    tags=("tfp", "public-api", "optimization"),
)

add_md(
    "tfp-vi-result",
    r"""
![用变分推断近似二维香蕉形目标密度。左图是均值场高斯：每一维各有可训练位置和尺度；右图是具有可训练均值与完整协方差的二维满秩高斯 {cite:p}`kucukelbir2016automatic`。点表示优化后的近似样本，等高线表示目标。两者都不能完整捕捉弯曲形状，但满秩高斯凭借更丰富结构通常更接近目标。](img/vi_in_tfp.png)

> **中文版现代化说明。** 上图是原书哈希锁定的静态资源，由已废弃的私有 `tfp.experimental.vi` 构造器生成；本章上方 `tfp-public-surrogates`/`tfp-fit-gaussian-surrogates` 单元格改用公开 API 重新实现了等价的均值场/满秩高斯拟合，并已通过独立断言验证收敛。两者定性结论一致（满秩高斯通常更贴合弯曲目标），但优化轨迹的随机性来源不同，实际重新运行代码得到的散点位置不会与静态图逐点相同。
""",
    "3494-3509",
    labels=("fig:vi_in_tfp",),
    modernization="补充说明：静态图仍来自原书私有 API 输出，与上方已迁移到公开 API 的代码定性一致但不逐点相同。",
)

add_md(
    "normalizing-flow-addition",
    r"""
#### 现代化补充：正规化流替代分布

以下正规化流“深入 VI”示例只存在于 `notebooks_updated/chp_11.ipynb`，不属于权威 Markdown 正文。这里明确作为现代化补充保留：它使用公开的 `AutoregressiveNetwork`、`MaskedAutoregressiveFlow` 与 `TransformedDistribution`，避免迁移笔记中脆弱的嵌套 `JointMap`/`Split`/`Reshape` 事件结构，也不调用任何私有 TFP 内部接口。
""",
    "not-in-authority; migration cells 34-38",
    provenance="notebook-only-modernization-addition",
    modernization="明确标注更新笔记独有的正规化流扩展，不把它伪装成原书 Markdown 内容。",
)

add_code(
    "tfp-normalizing-flow",
    r"""
flow_base_loc = tf.Variable(tf.zeros(2), name="flow_base_loc")
flow_base_scale = tfp.util.TransformedVariable(
    tf.ones(2), bijector=tfb.Softplus(), name="flow_base_scale"
)
flow_base = tfd.MultivariateNormalDiag(
    loc=flow_base_loc, scale_diag=flow_base_scale
)
flow_network = tfb.AutoregressiveNetwork(
    params=2, hidden_units=[16, 16], activation="tanh", name="flow_network"
)
flow_bijector = tfb.MaskedAutoregressiveFlow(
    shift_and_log_scale_fn=flow_network, name="masked_autoregressive_flow"
)
flow_surrogate = tfd.TransformedDistribution(
    distribution=flow_base, bijector=flow_bijector, name="flow_surrogate"
)

# 首次调用只用于构建网络变量，不执行优化。
_ = flow_surrogate.sample(1, seed=RANDOM_SEED + 70)
assert flow_surrogate.event_shape == tf.TensorShape([2])
assert flow_surrogate.trainable_variables

flow_losses, flow_samples, flow_energy = fit_surrogate_checked(
    flow_surrogate, FLOW_VI_STEPS, RANDOM_SEED + 71
)
assert np.median(flow_energy) <= mean_field_energy + 12.0
assert np.quantile(flow_energy, 0.95) < 100.0
""",
    "not-in-authority; migration cells 34-38",
    migration_cells=(34, 35, 36, 37, 38),
    provenance="notebook-only-modernization-addition",
    modernization="以单一二维向量事件构建公共 MAF，不使用私有 API 或脆弱 JointMap/Split/Reshape 链；采用独立 flow 预算与有界能量断言。",
    tags=("tfp", "public-api", "normalizing-flow", "optimization"),
)

add_md(
    "programming-references",
    r"""
## 编程参考

计算贝叶斯的一部分，当然是计算机和现有软件工具。现代贝叶斯实践者借助这些工具共享模型、减少错误，并加速模型构建与推断。要让计算机替我们工作，就必须编程；但说起来容易，真正有效使用仍需要思考与理解。本章最后一节对几个主要概念给出高层建议。
""",
    "3511-3522",
    anchor="programming_ref",
)

add_md(
    "which-programming-language",
    r"""
### 选择哪种编程语言？

编程语言很多。本书主要使用 Python，但 Julia、R、C/C++ 等流行语言也有专门的贝叶斯计算应用。应该选哪一种？没有普适的对错答案，应考察完整生态系统。本书使用 Python，是因为 ArviZ、Matplotlib、Pandas 等包让数据处理和展示更方便；这些能力并非 Python 独有。贝叶斯实践者尤其应考虑该语言可用的概率编程语言（PPL）：如果一种也没有，可能需要重新考虑。还要考虑希望合作的社群使用什么语言。本书一位作者住在南加州，懂英语和一些西班牙语很实用，因为两者足以应对常见交流；编程语言也是如此，如果未来实验室使用 R，学习 R 就很合理。

计算纯粹主义者或许会强调有些语言运行更快。这当然是真的，但不要过度纠缠“哪种 PPL 最快”。现实中不同模型运行时间不同，还要区分“人的时间”——迭代并构思模型所需时间——与“模型运行时间”——计算机返回有用结果所需时间。二者并不相同，重要性也随场景变化。所以，不必一开始就过度担心选中“正确”语言；真正学会一种后，概念通常可以迁移到另一种。
""",
    "3523-3555",
    anchor="which-programming-language",
)

add_md(
    "version-control",
    r"""
### 版本控制

版本控制不是必需品，却非常值得推荐，正确使用会带来巨大收益。独自工作时，它让你放心迭代模型设计，不必担心丢失代码，也不怕某次修改或实验破坏模型；你能更快、更有信心地迭代，并在不同模型定义之间切换。多人协作时，版本控制的快照和比较功能支持代码共享与协作，否则这些工作会很困难甚至不可能。Mercurial、SVN、Perforce 等系统很多，目前最流行的是 Git。版本控制通常不绑定特定编程语言。
""",
    "3556-3572",
    anchor="version-control",
)

add_md(
    "dependency-management",
    r"""
### 依赖管理与软件包仓库

几乎所有代码都依赖其他代码才能运行——层层相叠，永无尽头。PPL 尤其依赖大量库。强烈建议熟悉一种依赖管理工具，用来查看、列出并冻结分析所依赖的软件包；软件包仓库则是获取这些依赖的位置。它们通常与语言相关：Python 可用 `pip` 管理依赖，PyPI 是常用公共仓库；Scala 可用 `sbt` 管理依赖，Maven 是流行仓库。成熟语言都有相应工具，但必须主动选择并使用它们。
""",
    "3573-3588",
    anchor="dependency-management-and-package-repositories",
)

add_md(
    "environment-management",
    r"""
### 环境管理

所有代码都在某个环境中执行。很多人直到代码突然失效，或换一台计算机就无法运行时，才想起环境的存在。环境管理是一组用于创建可复现计算环境的工具；贝叶斯建模者已经需要处理模型中的随机性，更不希望计算机额外增加一层不可控差异，因此它尤其重要。

遗憾的是，环境管理也是编程中最令人困惑的部分之一。大体有语言特定与语言无关两类方案：Python 的 `virtualenv` 是语言特定的环境管理器，容器化与虚拟化则与语言无关。这里不给出唯一建议，因为选择很大程度取决于你对工具的熟悉程度和计划在哪里运行代码；但强烈建议有意识地作出选择，以确保结果可复现。
""",
    "3589-3608",
    anchor="environment-management",
)

add_md(
    "development-environments",
    r"""
### 文本编辑器、集成开发环境还是 Notebook？

写代码总得有个地方。面向数据工作的常见界面大体有三类。

第一类、也最简单的是文本编辑器。最基础的编辑器让你编辑并保存文本：写 Python 程序、保存、运行。它们通常很“轻”，除查找替换等基础功能外几乎不附加其他能力。可以把文本编辑器想成自行车：界面基本只有车把与踏板，能把你从这里送到那里，但主要工作由你完成。

相比之下，集成开发环境（IDE）像现代飞机：功能和按钮极多，自动化程度很高。IDE 核心仍是编辑文本，但也集成运行代码、单元测试、代码检查、版本控制、版本差异比较等开发环节。编写跨越许多模块的大量复杂代码时，IDE 通常最有用。

如今文本编辑器与 IDE 的边界很模糊。建议初学者先从偏文本编辑器的一端开始，熟悉代码工作方式后再转向 IDE；否则很难判断 IDE 在“幕后”替你做了什么。

Notebook 是完全不同的界面。它把代码、输出与文档混合在一起，还允许非线性执行。本书大部分代码和图都在 Jupyter Notebook 中，也提供云端 Notebook 环境 Google Colab 的链接。Notebook 通常最适合探索性数据分析和本书这样的解释性场景，不太适合运行生产代码。

对 Notebook 的建议与 IDE 类似：若刚接触统计计算，先学会从单独文件运行代码，再转向云端 Google Colab、Binder，或本地 Jupyter Notebook 环境。
""",
    "3609-3654",
    anchor="dev_environment",
)

add_md(
    "book-specific-tools",
    r"""
### 本书使用的具体工具

下面是本书采用的工具；这并不意味着它们是唯一选择，只是作者实际使用的组合。

- **编程语言**：Python；
- **概率编程语言**：PyMC、TensorFlow Probability；书中也简要展示 Stan 与 NumPyro；
- **版本控制**：Git；
- **依赖管理**：`pip` 与 `conda`；
- **软件包仓库**：PyPI、conda-forge；
- **环境管理**：`conda`；
- **通用文档**：LaTeX（图书写作）、Markdown（代码包）与 Jupyter Notebook。

> **现代化说明。** 原文工具清单写的是历史名称 PyMC3；中文版统一使用当前项目名称 PyMC，但不改变原书采用该工具族的事实。
""",
    "3655-3677",
    anchor="the-specific-tools-used-for-this-book",
    modernization="把历史名称 PyMC3 更新为 PyMC，并规范 PyPI、NumPyro 的当前拼写。",
)

add_md(
    "chapter-footnotes",
    r"""
[^1]: 今天称为西班牙与葡萄牙的大片领土曾属于安达卢斯及阿拉伯政权，这对西班牙/葡萄牙文化产生了巨大影响，包括食物、音乐、语言乃至遗传构成。

[^2]: 若想深入学习概率论，推荐 Joseph K. Blitzstein 与 Jessica Hwang 的 *Introduction to Probability* {cite:p}`blitzstein_2019`。

[^3]: John K. Kruschke 据此精彩地概括：贝叶斯推断就是在各种可能性之间重新分配可信度（概率）{cite:p}`Kruschke2014`。

[^4]: 若还要确定圆周相对于平面中其他物体的位置，就需要圆心坐标；这里暂时忽略这一细节。

[^5]: 增加或保持不变，但绝不减少。

[^6]: 粗略说，右连续函数从右侧趋近极限点时不会发生跳跃。

[^7]: 一次结果不会影响其他结果。

[^8]: 更准确地说，令二项分布 $\operatorname{Bin}(n,p)$ 中 $n\to\infty$、$p\to0$，同时保持 $np$ 不变，其极限就是泊松分布。

[^9]: 要避免看似荒谬的表述，需要正式讨论测度论；这里绕开这一要求。

[^10]: 可借助 SciPy 自行检验这一说法。

[^11]: 它不仅在地球上著名；从我们观察到的高斯形 UFO 来看，其他星球似乎也一样——当然只是玩笑，正如所谓 UFO 学一样。

[^12]: William Gosset 为改进一家啤酒厂的质量控制方法而发现了这个分布。公司允许员工发表科学论文，条件是不能出现“啤酒”、公司名称或员工自己的姓氏，因此 Gosset 以 “Student” 为笔名发表。

[^13]: 参见 <https://en.wikipedia.org/wiki/Gamma_function>。

[^14]: $\nu$ 也可以取小于 1 的值。

[^15]: 例如参见 <https://www.youtube.com/watch?v=i5oND7rHtFs>。

[^16]: 熟悉特征向量与特征值的读者应该会觉得似曾相识。

[^17]: 另一个类比来自政治：政治人物或政府不断更替，不平等、气候变化等紧迫问题却仍未得到妥善处理。

[^18]: 严格说还应包括玻璃分子和空气中的分子等等，不过这里专注于水。

[^19]: 别让海森堡和他的不确定性原理扫了兴。

[^20]: 通常逐字母读作 W-A-I-C，尽管连读会省力一些。

[^21]: 我们不喜欢这种经验分级，但例如可参见 <https://en.wikipedia.org/wiki/Bayes_factor#Interpretation>。

[^22]: 实践中，为保证数值稳定，边缘似然通常在对数尺度计算；此时两个模型的对数边缘似然之差对应对数 Bayes 因子。

[^23]: 名称来自摩纳哥公国著名的蒙特卡洛赌场。

[^24]: 这个视频以非常从容、清晰的方式展示了一个密切相关的例子：<https://www.youtube.com/watch?v=zwAD6dRSVyI>。

[^25]: 也称混合蒙特卡洛，因为它最初被构想为分子力学与 Metropolis–Hastings 的混合方法；分子力学是分子系统中广泛使用的模拟技术。

[^26]: 代码原型来自好友 Colin Carroll 关于 HMC 的博客文章：<https://colindcarroll.com/2019/04/11/hamiltonian-monte-carlo-from-scratch/>。中文版为避免未锁定 JAX 依赖，对接口作了明确标注的现代化修改。

[^27]: 该值位于 $[0,1]$，原文所述默认值为 0.8；另见 {ref}`divergences`。具体默认值可能随 PyMC 版本变化，应以当前 API 文档为准。

[^28]: 这里的“真正”纯粹指数学意义，不评价该后验是否适合某个实际问题。

[^29]: 关于这一物理系统类比的更多细节，见 {ref}`entropy`。

[^30]: 这些术语借自冶金学，特指通过加热和冷却合金以获得特定分子结构的过程。

[^31]: 这里的有效样本量由重要性权重计算，不同于诊断 MCMC 采样器时依据样本自相关计算的 ESS。
""",
    "3678-3783",
    labels=tuple(f"footnote:{idx}" for idx in range(1, 32)),
    modernization="脚注 26 说明 HMC 接口现代化；脚注 27 避免把历史 PyMC 默认值表述为永久 API 保证。",
)

def validate_canonical_source() -> dict[str, int]:
    """对权威 Markdown、翻译单元格、来源元数据和代码做静态完整性检查。"""
    authority_path = HERE.parents[1] / SOURCE_AUTHORITY
    source = authority_path.read_text(encoding="utf-8")
    source_without_fences = re.sub(r"```.*?```", "", source, flags=re.DOTALL)
    active_source = re.sub(r"<!---.*?-->", "", source, flags=re.DOTALL)
    source_counts = {
        "anchors": len(re.findall(r"^\([^)]+\)=\s*$", source, re.MULTILINE)),
        "headings": len(re.findall(r"^#{1,6}\s+", source, re.MULTILINE)),
        "rendered_headings": len(
            re.findall(r"^#{1,6}\s+", source_without_fences, re.MULTILINE)
        ),
        "display_math": len(re.findall(r"^```\{math\}", source, re.MULTILINE)),
        "active_display_math": len(
            re.findall(r"^```\{math\}", active_source, re.MULTILINE)
        ),
        "commented_display_math": len(
            re.findall(r"^```\{math\}", source, re.MULTILINE)
        )
        - len(re.findall(r"^```\{math\}", active_source, re.MULTILINE)),
        "equation_labels": len(re.findall(r"^:label:\s*\S+", source, re.MULTILINE)),
        "figures": len(re.findall(r"^```\{figure\}", source, re.MULTILINE)),
        "authoritative_code_blocks": len(
            re.findall(
                r"^```\s*(?:python|\{code-block\}\s+python)", source, re.MULTILINE
            )
        ),
        "citation_occurrences": len(
            re.findall(r"\{cite(?::p)?\}`[^`]+`", source)
        ),
        "footnotes": len(re.findall(r"^\[\^\d+\]:", source, re.MULTILINE)),
        "exercises": len(
            re.findall(r"^#{1,6}\s+Exercises?\b", source_without_fences, re.MULTILINE)
        ),
    }
    if source_counts != SOURCE_COMPLETENESS:
        raise RuntimeError(
            f"权威源结构计数变化：expected={SOURCE_COMPLETENESS}, actual={source_counts}"
        )

    ids = [cell["id"] for cell in cells]
    if len(ids) != len(set(ids)):
        raise RuntimeError("单元格 ID 重复")
    if not all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cell_id) for cell_id in ids):
        raise RuntimeError("单元格 ID 不符合共享构建器约束")

    for cell in cells:
        metadata = cell.get("metadata", {})
        if not isinstance(metadata.get("kind"), str):
            raise RuntimeError(f"单元格 {cell['id']} 缺少 metadata.kind")
        if not isinstance(metadata.get("provenance"), dict):
            raise RuntimeError(f"单元格 {cell['id']} 缺少 metadata.provenance")
        if cell["type"] == "code":
            compile(cell["source"], f"<cell:{cell['id']}>", "exec")

    translated_markdown = "\n".join(
        cell["source"] for cell in cells if cell["type"] == "markdown"
    )
    translated_all = "\n".join(cell["source"] for cell in cells)
    source_anchors = set(
        re.findall(r"^\(([^)]+)\)=\s*$", source, re.MULTILINE)
    )
    built_anchors = {
        cell["metadata"]["zh"].get("source_anchor") for cell in cells
    } - {None}
    if built_anchors != source_anchors:
        raise RuntimeError(
            f"翻译锚点集合不匹配：缺少={sorted(source_anchors - built_anchors)}，"
            f"多余={sorted(built_anchors - source_anchors)}"
        )

    source_labels = set(
        re.findall(r"^:(?:label|name):\s*(\S+)", source, re.MULTILINE)
    )
    translated_labels = {
        label
        for cell in cells
        for label in cell["metadata"]["zh"].get("labels", [])
    }
    if not source_labels.issubset(translated_labels):
        raise RuntimeError(
            f"缺少源标签：{sorted(source_labels - translated_labels)}"
        )

    translated_counts = {
        "authoritative_headings": len(
            re.findall(r"^#{1,6}\s+", translated_markdown, re.MULTILINE)
        )
        - 1,  # 明确标注的正规化流补充标题不属于权威 Markdown。
        "active_display_math": translated_markdown.count("$$") // 2,
        "figures": len(re.findall(r"\]\(img/[^)]+\)", translated_markdown)),
        "citation_occurrences": len(
            re.findall(r"\{cite(?::p)?\}`[^`]+`", translated_all)
        ),
        "footnotes": len(
            re.findall(r"^\[\^\d+\]:", translated_markdown, re.MULTILINE)
        ),
    }
    expected_translated = {
        "authoritative_headings": SOURCE_COMPLETENESS["rendered_headings"],
        "active_display_math": SOURCE_COMPLETENESS["active_display_math"],
        "figures": SOURCE_COMPLETENESS["figures"],
        "citation_occurrences": SOURCE_COMPLETENESS["citation_occurrences"],
        "footnotes": SOURCE_COMPLETENESS["footnotes"],
    }
    if translated_counts != expected_translated:
        raise RuntimeError(
            f"翻译结构计数不匹配：expected={expected_translated}, actual={translated_counts}"
        )

    source_images = set(
        re.findall(r"^```\{figure\}\s+figures/(\S+)", source, re.MULTILINE)
    )
    translated_images = set(
        re.findall(r"\]\(img/([^)]+)\)", translated_markdown)
    )
    if translated_images != source_images:
        raise RuntimeError(
            f"图像引用集合不匹配：缺少={sorted(source_images - translated_images)}，"
            f"多余={sorted(translated_images - source_images)}"
        )

    def citation_keys(text: str) -> list[str]:
        keys: list[str] = []
        for group in re.findall(r"\{cite(?::p)?\}`([^`]+)`", text):
            keys.extend(key.strip() for key in group.split(","))
        return sorted(keys)

    if citation_keys(translated_all) != citation_keys(source):
        raise RuntimeError("翻译引用键或重复次数与权威源不一致")
    source_footnote_ids = set(re.findall(r"\[\^(\d+)\]", source))
    translated_footnote_ids = set(re.findall(r"\[\^(\d+)\]", translated_markdown))
    if translated_footnote_ids != source_footnote_ids:
        raise RuntimeError("翻译脚注编号集合与权威源不一致")

    code_text = "\n".join(
        cell["source"] for cell in cells if cell["type"] == "code"
    )
    forbidden_tfp_tokens = (
        "tfp.experimental.vi",
        "tensorflow_probability.python",
    )
    if any(token in code_text for token in forbidden_tfp_tokens):
        raise RuntimeError("发现不允许的实验性或私有 TFP 内部 API")

    validate_assets()
    return {**source_counts, "cells": len(cells)}


def _write_notebook() -> Path:
    import sys

    sys.path.insert(0, str(HERE.parent / "tools"))
    import nb_tools

    validate_canonical_source()
    nb_tools.validate_cells(cells)
    output = HERE / "Ch11_AppendicealTopics_zh.ipynb"
    nb_tools.write_ipynb(cells, output)
    return output


if __name__ == "__main__":
    print(f"wrote {_write_notebook()}")
