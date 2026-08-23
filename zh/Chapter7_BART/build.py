"""Canonical Chinese source builder for Chapter 7: Bayesian Additive Regression Trees.

The whole-book orchestrator imports the module-level ``cells`` list. Direct
execution validates source completeness, provenance, assets, syntax, and the
runtime contract; it intentionally does not emit a notebook or Org file.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import re
import shutil
import sysconfig
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

CHAPTER_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = CHAPTER_DIR.parents[1]
PROSE_AUTHORITY = REPOSITORY_ROOT / "markdown" / "chp_07.md"
CODE_AUTHORITY = REPOSITORY_ROOT / "notebooks_updated" / "chp_07.ipynb"
MANIFEST_PATH = CHAPTER_DIR / "manifest.toml"

SOURCE_SHA256 = {
    "markdown/chp_07.md": "51e768e66549844e2508939c6e8222bfb82cd2fe0f252b1ca296347dc59ba24e",
    "notebooks_updated/chp_07.ipynb": "cbc8f4250cbd2330675bc3fb08a04669a4539c324850f0454c17c9069fb6f207",
}
DATA_SHA256 = {
    "bikes_hour.csv": "58a5b55b8a7617b669b79414bdc11893d33228e14e89e702512917558059812d",
    "space_influenza.csv": "52e1f5205ddd8a87bf49ad1f68c97301f8ededf645eb50a1e6547b205aaed3a7",
}
STATIC_FIGURE_SHA256 = {
    "decision_tree.png": "41cf5d03b155298177a1f018c91c3402c95bb5855786ef8d08265d931ee0cecb",
    "decision_tree_reg.png": "c284dc5222c1c18c5e741bba9dff08131a369c6601abfe75fc1aed14882ec249",
    "decision_tree_overfitting.png": "54c0885d8c8e9a84be26ab4de9289863b106c2369b726f93358d9693023a620b",
    "BART_bikes_samples.png": "6edc08419af6431473d0e183b00bdea4b65a2aed41dc6705012bc0d50bdc05c6",
    "BART_bikes.png": "a0f9e58f7c861db516523d68f37d70cba5d52d5e6c48337660871df90740c81f",
    "BART_space_flu_comp.png": "87f376b74f7e060fcee472b9010a0dbe94aee037cd30516d660e8fc6bf372b6c",
    "BART_space_flu_fit.png": "2b5a287a480426426608aed3b0a0e86980bf101c769fa46456a892a48c4bb5cb",
    "partial_dependence_plot.png": "62e47e3b352d32285ce28942ce3354219d2fbc68b50d6ce4812c01bdc82c8744",
    "partial_dependence_plot_bikes.png": "98657922f82f3ba0b1709324fdbe9bcac26197a1c9430c47be09f0b2cf846d6a",
    "individual_conditional_expectation_plot_bikes.png": "6f0e25e85e6966b49849daa68c5204350b09476ba1a50c573f571a0c780a0eac",
    "pdp_vs_ice_toy.png": "5dc081d6d72a86e4cdc3eb4b7ef9f7cbf9171fbfcc12cf540a32d128c3834758",
    "bart_vi_toy.png": "1f34d4ad3ca314f7b1a01f72e24a51ab6792cd56d1bb6ff3c96fb13f14d3fcdb",
    "bart_vi_bikes.png": "6b053deb9df0bcf6488bc0cb16e92d35806bc74480f38ab95421eb494bfe5c4a",
}
GENERATED_FIGURES = {
    "BART_bikes.png",
    "BART_space_flu_comp.png",
    "BART_space_flu_fit.png",
    "partial_dependence_plot.png",
    "partial_dependence_plot_bikes.png",
    "individual_conditional_expectation_plot_bikes.png",
    "pdp_vs_ice_toy.png",
    "bart_vi_toy.png",
    "bart_vi_bikes.png",
}
API_COMPATIBILITY = {
    "python": ">=3.12",
    "numpy": ">=2.0",
    "pymc": ">=5.24,<6 (validated target: 5.28.5)",
    "arviz": "validated target: 0.23.4",
    "pymc-bart": ">=0.11.0,<0.12.0",
    "arviz-stats[xarray]": ">=0.6.0",
}
PUBLIC_BART_APIS = {
    "pmb.BART",
    "pmb.plot_pdp",
    "pmb.plot_ice",
    "pmb.compute_variable_importance",
    "pmb.plot_variable_importance",
}

EXPECTED_ANCHORS = {
    "chap7",
    "chap6",
    "decision-trees",
    "ensembles-of-decision-trees",
    "the-bart-model",
    "priors-for-bart",
    "prior-independence",
    "prior-for-the-tree-structure-mathcalt_j",
    "bart_mu_m_priors",
    "fitting-bayesian-additive-regression-trees",
    "bart_bike",
    "generalized-bart-models",
    "interpretability-of-barts",
    "partial-dependence-plots",
    "individual-conditional-expectation",
    "sec:variable_selection",
    "priors-for-bart-in-pymc3",
    "exercises7",
}
EXPECTED_EQUATIONS = {
    "eq:bart",
    "eq:bart_gaussian",
    "eq:bart_poisson",
    "eq:bart_student",
    "eq:bart_bikes_model",
    "eq:partial_dependence",
}
EXPECTED_FIGURE_IDS = {
    "fig:decision_tree",
    "fig:decision_tree_reg",
    "fig:decision_tree_overfitting",
    "fig:bart_bikes_samples",
    "fig:bart_bikes",
    "fig:BART_space_flu_comp",
    "fig:BART_space_flu_fit",
    "fig:pdp_fake_example",
    "fig:partial_dependence_plot_bikes",
    "fig:individual_conditional_expectation_plot_bikes",
    "fig:pdp_vs_ice_toy",
    "fig:bart_vi_toy",
    "fig:bart_vi_bikes",
}
EXPECTED_CITATIONS = {
    "breiman2001",
    "BreimanForests2001",
    "ZhouEnsembleMethods2012",
    "ChipmanBARTBayesianadditive2010",
    "Rockova2018",
    "Lakshminarayanan",
    "Molnarbook",
    "Molnar2020",
    "Friedman2001",
    "Goldstein2014",
    "Liu2020",
    "Carlson2020",
    "Bleich2014",
    "BalogMondrianProcessMachine2015",
    "royMondrianProcess",
}
EXPECTED_EXERCISES = tuple(
    [f"7E{index}" for index in range(1, 9)]
    + [f"7M{index}" for index in range(9, 13)]
)
EXPECTED_CODE_BLOCKS = {"bart_model_gauss", "bart_model_bern"}
PROSE_SOURCE_RANGES = (
    "1-39",
    "40-198",
    "199-227",
    "228-278",
    "279-340",
    "341-387",
    "388-477",
    "478-550",
    "551-575",
    "576-690",
    "691-750",
    "751-889",
    "890-944",
    "945-1037",
    "1038-1076",
)
CODE_SOURCE_CELLS = (
    "cells 1-2",
    "cell 4",
    "cell 6",
    "cell 8",
    "cell 10",
    "cell 11",
    "cell 14",
    "cell 15",
    "cell 16",
    "cell 17",
    "cell 20",
    "cell 21",
    "cell 22",
    "cell 24",
    "cell 25",
    "cell 26",
    "cell 27",
    "cell 28",
    "cell 29",
    "cell 30",
    "cell 32",
    "cell 33",
    "cell 35",
    "cells 36-37",
    "cell 39",
)

PROSE_CELLS = [
    # Source: markdown/chp_07.md lines 1-39.
    (
        "ch07-introduction-prose",
        r"""
(chap6)=

(chap7)=

> **中文版补充**：原稿在本章开头使用了遗留锚点 `(chap6)=`。为保持既有交叉引用兼容性，
> 这里原样保留该锚点；同时新增正确的第 7 章锚点 `(chap7)=`。后者是对章号的明确修正，
> 前者不应再被理解为本章实际编号。

# 贝叶斯加性回归树

在[第 5 章](chap3_5)中，我们看到如何通过一系列（简单）基函数之和来近似一个函数，
并展示了 B 样条作为基函数时所具有的一些良好性质。本章将讨论一种相似的方法，不过我们将用
**决策树**代替 B 样条。决策树是表示分段常数函数（也就是阶梯函数）的另一种灵活方式；我们在
[第 5 章](chap3_5)已经见过这类函数。本章尤其关注贝叶斯加性回归树（Bayesian
Additive Regression Trees，BART）：这是一类用决策树之和获得灵活模型的贝叶斯非参数模型
[^1]。人们讨论它时，使用的语言往往更接近机器学习而不是统计学 {cite:p}`breiman2001`。
从某种意义上说，与本书其他章节精心手工构建的模型相比，BART 更像一种“设定好以后就让它自己
运行”的模型。

在 BART 文献中，人们通常不谈基函数，而是谈**学习器**（learners），但整体思想非常相似：
用多个简单函数——也称为学习器——的组合去近似复杂函数，并施加足够的正则化，从而在获得灵活性
的同时不过度增加模型复杂度，也就是避免过拟合。用多个学习器共同解决同一个问题的方法称为集成
方法。在这一语境下，学习器可以是你能想到的任何统计模型或数据算法。集成方法建立在这样一种观察
之上：组合多个**弱学习器**通常比试图使用单个非常**强的学习器**更好。一般认为，为了在准确率
与泛化能力上取得良好结果，基础学习器既要尽可能准确，也要尽可能多样
{cite:p}`ZhouEnsembleMethods2012`。BART 所采用的主要贝叶斯思想是：决策树很容易过拟合，
因此我们加入正则化先验（或收缩先验），让每棵树都表现为一个**弱学习器**。

为了把这段总体描述变成更容易理解和应用的内容，我们应当先讨论决策树。如果你已经熟悉决策树，
可以直接跳过下一节。
""",
    ),
    # Source: markdown/chp_07.md lines 40-198.
    (
        "ch07-decision-trees-prose",
        r"""
(decision-trees)=

## 决策树

假设我们有两个变量 $X_1$ 和 $X_2$，并希望用它们把对象分到两个类别中：⬤ 或 ▲。为了完成这个
任务，可以使用 {numref}`fig:decision_tree` 左图所示的树结构。树就是节点的集合，其中任意两个
节点之间至多由一条线或边连接。{numref}`fig:decision_tree` 中的树称为二叉树，因为每个节点
最多有两个子节点。没有子节点的节点称为叶节点或终端节点。本例有两个内部节点（也称内节点，图中
用矩形表示）和 3 个终端节点（用圆角矩形表示）。每个内部节点都有一条与之关联的决策规则。沿着
这些决策规则前进，最终会到达唯一的叶节点，由它给出决策问题的答案。例如，如果变量 $X_1$ 的
某个实例大于 $c_1$，决策树会让我们把该实例归入类别 ⬤。反之，如果观测到 $x_{1i}$ 小于
$c_1$，并且 $x_{2i}$ 小于 $c_2$，就应将其归入类别 ▲。从算法角度看，可以把一棵树理解为
一组 if-else 语句，我们依次遵循这些语句完成分类之类的任务。从几何角度看，也可以把二叉树理解
为一种把样本空间划分成若干**区块**的方法，如 {numref}`fig:decision_tree` 右图所示。每个
区块都由垂直于坐标轴的**分割**线定义，因此样本空间中的每次分割都会与某个协变量（或特征）
坐标轴对齐。

从数学上说，一棵决策树 $g$ 完全由两个集合定义：

-   $\mathcal{T}$：边和节点的集合（即 {numref}`fig:decision_tree` 中的矩形、圆角矩形及
    连接它们的线），以及与内部节点关联的决策规则。

-   $\mathcal{M} = \{\mu_1, \mu_2, \dots, \mu_b\}$：与 $\mathcal{T}$ 的每个终端节点
    关联的一组参数值。

于是，$g(X; \mathcal{T}, \mathcal{M})$ 是把 $\mu_i \in M$ 赋给 $X$ 的函数。例如，在
{numref}`fig:decision_tree` 中，$\mu_i$ 的取值为（⬤、⬤ 和 ▲）。函数 $g$ 把满足
$X_1>c_1$ 的情形赋值为 ⬤；把满足 $X_1<c_1$ 且 $X_2>c_2$ 的情形赋值为 ⬤；把满足
$X_1<c_1$ 且 $X_2<c_2$ 的情形赋值为 ▲。

把树抽象地定义为两个集合组成的元组 $g(\mathcal{T}, \mathcal{M})$，等我们稍后讨论树上的
先验时会非常有用。

```{figure} img/chp07/decision_tree.png
:name: fig:decision_tree
:width: 8.00in
一棵二叉树（左）及其对应的样本空间划分（右）。树的内部节点是那些拥有子节点的节点，它们都由
一条边连接到下方节点，并带有相应的分割规则。终端节点（即叶节点）没有子节点，包含最终返回的值；
本例中为 ⬤ 或 ▲。决策树把样本空间划分成由垂直于坐标轴的分割线界定的区块。这意味着样本空间
中的每次分割都会与某个协变量轴对齐。
```

{numref}`fig:decision_tree` 展示的是如何用决策树解决分类问题，其中 $\mathcal{M}_j$ 包含类别
或标签值；树也可以用于回归。在回归情形下，我们不再把终端节点与类别标签关联，而是把它与一个
实数关联，例如某个区块内数据点的均值。{numref}`fig:decision_tree_reg` 展示了只有一个协变量
的回归例子。左边是一棵与 {numref}`fig:decision_tree` 相似的二叉树，主要区别在于：
{numref}`fig:decision_tree_reg` 中每个叶节点返回的不是类别，而是一个实数。再把这棵树与右侧
近似正弦形的数据对照起来看：这里并没有得到连续函数近似，而是把数据划分为三个区块，再用各区块
的均值来近似该区块。

```{figure} img/chp07/decision_tree_reg.png
:name: fig:decision_tree_reg
:width: 8.00in
一棵二叉树（左）及其对应的样本空间划分（右）。树的内部节点是那些拥有子节点的节点（它们由一条
边连接到下方节点），每个内部节点都有相应的分割规则。终端节点（即叶节点）没有子节点，包含最终
返回的值（本例中为 1.1、1.9 和 0.1）。由此可以看出，树是表示分段函数的一种方式，正如
[第 5 章](chap3_5)讨论过的那些函数。
```

回归树并不局限于返回区块内数据点的均值，也可以采用其他选择。例如，可以把叶节点与数据点的
中位数关联；可以对每个区块中的数据点拟合线性回归；甚至可以拟合更复杂的函数。尽管如此，均值
大概仍是回归树最常见的选择。

需要注意，回归树的输出不是光滑函数，而是分段阶梯函数。这并不意味着回归树一定不适合拟合光滑
函数。原则上，我们可以用阶梯函数近似任意连续函数；实践中，这种近似也可能已经足够好。

决策树一个很吸引人的特点是可解释性：你真的可以“阅读”一棵树，沿着解决特定问题所需的步骤走
下去。因此，你能透明地理解方法在做什么、为什么会有当前表现、为什么某些类别没有得到正确分类，
或者为什么某些数据拟合得不好。此外，也很容易用简单语言向非技术受众解释结果。

遗憾的是，决策树的灵活性也意味着它们很容易过拟合，因为总能找到一棵足够复杂、让每个数据点都
单独占据一个分区的树。{numref}`fig:decision_tree_overfitting` 展示了分类问题的一个过度复杂
解。你也可以亲自验证这一点：拿一张纸，画几个数据点，然后构造一个把每个点单独隔离开的划分。
做这个练习时，你可能还会发现，事实上不止一棵树能够同样好地拟合这些数据。

```{figure} img/chp07/decision_tree_overfitting.png
:name: fig:decision_tree_overfitting
:width: 4.5in
样本空间的一个过度复杂划分。每个数据点都被分到单独的区块中。我们称它为**过度复杂**的划分，
因为用 {numref}`fig:decision_tree` 那样简单得多的划分，也能以同等准确度解释和预测这些数据。
最简单的划分通常比更复杂的划分更可能实现泛化，也就是更可能预测并解释新数据。
```

如果像分析线性模型那样，从主效应和交互效应的角度思考树（见[第 4 章](chap3)），会发现树还有
一个有趣性质。注意，$\mathbb{E}(Y \mid \boldsymbol{X})$ 等于所有叶节点参数 $\mu_{ij}$
之和，因此：

-   当一棵树只依赖单个变量时（如 {numref}`fig:decision_tree_reg`），每个这样的
    $\mu_{ij}$ 都表示一个主效应。

-   当一棵树依赖多个变量时（如 {numref}`fig:decision_tree`），每个这样的 $\mu_{ij}$ 都表示
    一个交互效应。例如，要返回三角形，需要 $X_1$ 与 $X_2$ 发生交互，因为子节点的条件
   （$X_2>c_2$）以父节点的条件（$X_1>c_1$）为前提。

由于树的大小可变，我们可以用树来表示不同阶数的交互效应。树越深，更多变量进入树的机会就越大，
表示高阶交互的潜力也随之增加。此外，因为我们使用的是树的集成，所以实际上可以构造几乎任意组合
的主效应与交互效应。
""",
    ),
    # Source: markdown/chp_07.md lines 199-227.
    (
        "ch07-tree-ensembles-prose",
        r"""
(ensembles-of-decision-trees)=

### 决策树集成

考虑到过度复杂的树往往不擅长预测新数据，人们通常会引入一些机制来降低决策树的复杂度，使拟合
能够更好地适应当前数据自身的复杂程度。一种解决方案是拟合树的集成，并对其中每一棵树进行正则化，
让它保持较浅。结果是，每棵树单独只能解释数据的一小部分；只有把许多这样的树组合起来，才能给出
合适的答案。这可以说是数据科学版本的“团结就是力量”。贝叶斯方法（如 BART）和非贝叶斯方法
（如随机森林）都采用这种集成策略。总体而言，集成模型既能保持对给定数据集的灵活拟合能力，又能
降低泛化误差。

使用集成还有助于减轻输出的“阶梯感”。尽管多棵树组合后的输出仍然是阶梯函数，但它拥有更多阶梯，
因而可以形成某种程度上更光滑的近似；前提是我们要确保各棵树足够多样。

树集成的一个缺点是：单棵决策树的可解释性会丢失。现在，要得到答案，不能只沿着一棵树走，而要
同时考虑许多棵树，这通常会遮蔽任何简单解释。我们用可解释性换取了灵活性和泛化能力。
""",
    ),
    # Source: markdown/chp_07.md lines 228-278.
    (
        "ch07-bart-model-prose",
        r"""
(the-bart-model)=

## BART 模型

如果假设公式 [eq:bfr](eq:bfr) 中的 $B_i$ 函数都是决策树，就可以写成：

```{math}
:label: eq:bart
\mathbb{E}[Y] = \phi \left(\sum_{j=0}^m g_j(\boldsymbol{X}; \mathcal{T}_j, \mathcal{M}_j), \theta \right)
```

其中，每个 $g_j$ 都是形如 $g(\boldsymbol{X}; \mathcal{T}_j, \mathcal{M}_j)$ 的树；
$\mathcal{T}_j$ 表示一棵二叉树的结构，也就是内部节点及其决策规则的集合，以及终端节点的集合。
$\mathcal{M}_j = \{\mu_{1,j}, \mu_{2,j}, \cdots, \mu_{b,j}\}$ 表示 $b_j$ 个终端节点处的
取值；$\phi$ 表示模型似然所采用的任意概率分布；$\theta$ 则表示 $\phi$ 中没有被建模成树之和的
其他参数。

例如，可以令 $\phi$ 为高斯分布，于是得到：

```{math}
:label: eq:bart_gaussian
Y = \mathcal{N}\left(\mu = \sum_{j=0}^m g_j(\boldsymbol{X}; \mathcal{T}_j, \mathcal{M}_j), \sigma \right)
```

也可以像[第 3 章](chap2)讨论广义线性模型时那样尝试其他分布。例如，如果 $\phi$ 是泊松分布，
就得到：

```{math}
:label: eq:bart_poisson
Y = \text{Pois}\left(\lambda = \sum_{j}^m g_j(\boldsymbol{X}; \mathcal{T}_j, \mathcal{M}_j)\right)
```

> **中文版补充**：上面的 `eq:bart_poisson` 按原书公式保留，但泊松分布的率参数必须满足
> $\lambda>0$，而树之和可以为负。因此实际模型必须使用值域为正的逆链接函数，例如
> $\lambda=\exp\!\left(\sum_j^m g_j(\boldsymbol X;\mathcal T_j,\mathcal M_j)\right)$；
> 也可以选用其他始终为正的映射。这是对原式取值域约束的修正，而不是对原式的静默替换。

又或者，令 $\phi$ 为 Student $t$ 分布：

```{math}
:label: eq:bart_student
Y = \text{T}\left(\mu = \sum_{j}^m g_j(\boldsymbol{X}; \mathcal{T}_j, \mathcal{M}_j), \sigma, \nu \right)
```

和往常一样，要完整指定一个 BART 模型，还需要选择先验。我们已经熟悉高斯似然中 $\sigma$ 的先验，
以及 Student $t$ 似然中 $\sigma$ 和 $\nu$ 的先验，因此接下来将重点讨论 BART 模型特有的先验。
""",
    ),
    # Source: markdown/chp_07.md lines 279-340.
    (
        "ch07-bart-priors-prose",
        r"""
(priors-for-bart)=

## BART 的先验

原始 BART 论文 {cite:p}`ChipmanBARTBayesianadditive2010` 以及之后的大多数改进和实现都依赖
共轭先验。PyMC3 中的 BART 实现不使用共轭先验，而且在其他方面也有所不同。这里不逐项讨论这些
差异，而把重点放在 PyMC3 的实现上，因为本章示例将使用这一实现。

> **中文版现代化说明**：上一段描述的是原书写作时的历史实现。当前 PyMC 中的 BART 已移到独立
> 包 `pymc_bart`；在 PyMC-BART 0.11 中应先 `import pymc_bart as pmb`，再使用
> `pmb.BART(...)`，而不是 `pm.BART(...)`。正文中继续出现的“PyMC3”均保留其历史语境。

(prior-independence)=

### 先验独立性

为了简化先验的指定，我们假设树结构 $\mathcal{T}_j$ 与叶节点取值 $\mathcal{M}_j$ 相互独立。
此外，这些先验还与其余参数——公式 {eq}`eq:bart` 中的 $\theta$——相互独立。独立性假设允许我们
把先验拆成几个部分分别指定；否则，就必须设计一种方法，在整个树空间上指定一个单一先验 [^2]。

(prior-for-the-tree-structure-mathcalt_j)=

### 树结构 $\mathcal{T}_j$ 的先验

树结构 $\mathcal{T}_j$ 的先验由三个方面指定：

-   深度为 $d=(0,1,2,\dots)$ 的节点为非终端节点的概率，由 $\alpha^d$ 给出。建议把
    $\alpha$ 设在 $[0,0.5)$ 内 {cite:p}`Rockova2018` [^3]。

-   分割变量上的分布，也就是选择让哪个协变量进入树（{numref}`fig:decision_tree` 中的
    $X_i$）。最常见的做法是在所有可用协变量上使用均匀分布。

-   分割规则上的分布，也就是选定分割变量后，用哪个值作出决策（{numref}`fig:decision_tree`
    中的 $c_i$）。通常在可用取值上使用均匀分布。

> **中文版现代化说明**：上面第一条的 $\alpha^d$ 及 $\alpha\in[0,0.5)$ 是原书所述旧实现的
> 语义。PyMC-BART 0.11 的公开 API 同时使用 `alpha` 与 `beta`，深度 $d$ 处发生分割的先验概率为
> $\alpha(1+d)^{-\beta}$；默认值分别是 `alpha=0.95`、`beta=2.0`。其中较大的 `alpha`
> 提高分割倾向，较大的 `beta` 更快地惩罚深层分割。二者的现代语义不能与原书旧版 `alpha=0.25`
> 的单参数语义直接等同。

(bart_mu_m_priors)=

### 叶节点取值 $\mu_{ij}$ 与树数量 $m$ 的先验

默认情况下，PyMC3 不为叶节点取值设定一个先验值；相反，采样算法在每次迭代中返回残差的均值。

至于集成中的树数量 $m$，它通常也由用户预先设定。实践观察表明，把 $m$ 设为 200，甚至低至 10，
通常都能得到良好结果；推断也可能对 $m$ 的精确取值相当稳健。因此，一个通用经验法则是尝试几个
不同的 $m$，再通过交叉验证为具体问题选择最合适的值 [^4]。
""",
    ),
    # Source: markdown/chp_07.md lines 341-387.
    (
        "ch07-bart-fitting-prose",
        r"""
(fitting-bayesian-additive-regression-trees)=

## 拟合贝叶斯加性回归树

到目前为止，我们讨论了如何用决策树编码分段函数，以处理回归或分类问题，也讨论了如何为决策树
指定先验。下面将讨论如何高效地对树进行采样，从而针对给定数据集得到树的后验分布。完成这一任务
有许多策略，具体细节超出了本书范围，因此这里只描述主要组成部分。

拟合 BART 模型时，不能使用哈密顿蒙特卡洛这样的基于梯度的采样器，因为树空间是离散的，对梯度
并不友好。因此，研究者开发了专门适配树的 MCMC 与序贯蒙特卡洛（SMC）变体。PyMC3 中实现的
BART 采样器以序贯、迭代的方式工作。简要地说，我们从一棵树开始，用响应变量 $Y$ 拟合它，然后
计算残差
$R=Y-g_0(\boldsymbol{X};\mathcal{T}_0,\mathcal{M}_0)$。第二棵树拟合的是 $R$，而不是
$Y$。接着，根据目前已拟合树的总和更新残差；原文把这一步写成
$R-g_1(\boldsymbol{X};\mathcal{T}_0,\mathcal{M}_0)+g_0(\boldsymbol{X};\mathcal{T}_1,\mathcal{M}_1)$，
如此继续，直到拟合完 $m$ 棵树。

> **中文版补充**：上一段保留了原书的残差表达式，但其中存在明确的**残差索引笔误**：
> $g_1$ 错配了 $(\mathcal T_0,\mathcal M_0)$，$g_0$ 又错配了
> $(\mathcal T_1,\mathcal M_1)$，且符号也不一致。拟合完前两棵树后，正确的残差应为
> $R_1=Y-g_0(\boldsymbol X;\mathcal T_0,\mathcal M_0)-g_1(\boldsymbol X;\mathcal T_1,\mathcal M_1)$；
> 若把第一棵树后的残差记为 $R_0$，则更新式是
> $R_1=R_0-g_1(\boldsymbol X;\mathcal T_1,\mathcal M_1)$。

这一过程会产生后验分布的一个样本，其中包含 $m$ 棵树。注意，第一次迭代很容易产生次优树，主要
原因包括：最先拟合的树往往会比必要的更复杂；树可能陷入局部最小值；后拟合树的结果会受到前面各树
的影响。随着采样继续，这些影响往往会逐渐消失，因为采样方法会多次重新访问先前拟合的树，让它们
有机会根据更新后的残差重新调整。事实上，拟合 BART 时常见的现象是：最初几轮中的树往往较深，
之后会“坍缩”为更浅的树。

在文献中，具体 BART 模型通常会利用共轭性而与特定采样器配套，因此带高斯似然的 BART 模型会
不同于带泊松似然的模型。PyMC3 使用一种基于粒子 Gibbs 采样器 {cite:p}`Lakshminarayanan`、
且专门为树设计的采样器。PyMC3 会自动把该采样器分配给 `pm.BART` 分布；如果模型里还有其他
随机变量，则会为它们分配 NUTS 等其他采样器。

> **中文版现代化说明**：在 PyMC-BART 0.11 中，模型变量写作 `pmb.BART`，对应的专用步进方法
> 是 `pmb.PGBART`。调用 `pm.sample()` 时，PyMC 会为 BART 变量自动选择 PGBART，并可同时为适合
> 梯度采样的其他连续变量选择 NUTS；原文的 `pm.BART`/PyMC3 写法不再是当前公共 API。
""",
    ),
    # Source: markdown/chp_07.md lines 388-477.
    (
        "ch07-bikes-gaussian-prose",
        r"""
(bart_bike)=

## BART 自行车租赁示例

来看 BART 如何拟合我们曾在[第 5 章](chap3_5)研究过的自行车租赁数据集。模型为：

```{math}
:label: eq:bart_bikes_model
\begin{aligned}
\begin{split}
    \mu \sim& \; \text{BART}(m=50) \\
    \sigma \sim& \; \mathcal{HN}(1) \\
    Y \sim& \; \mathcal{N}(\mu, \sigma)
\end{split}\end{aligned}
```

在 PyMC3 中构建 BART 模型与构建其他类型模型非常相似。一个区别是，在指定随机变量 `pm.BART`
时，需要同时提供自变量和因变量。主要原因是，正如上一节所解释的，用来拟合 BART 的采样方法会
根据残差提出一棵新树。

有了这些说明，原模型用现代 PyMC-BART 0.11 公共 API 可以写成：

> **中文版现代化说明**：原代码使用 `pm.BART("μ", X, Y, m=50)` 和 PyMC3 的
> `return_inferencedata=True`。下面明确改为独立包中的 `pmb.BART`，并使用当前 PyMC 默认返回的
> `InferenceData`；需要先执行 `import pymc_bart as pmb`。

```{code-block} python
:name: bart_model_gauss
:caption: bart_model_gauss

with pm.Model() as bart_g:
    σ = pm.HalfNormal("σ", sigma=Y.std())
    μ = pmb.BART("μ", X=X, Y=Y, m=50)
    y = pm.Normal("y", mu=μ, sigma=σ, observed=Y)
    idata_bart_g = pm.sample(2000, random_seed=42)
```

在展示拟合模型的最终结果之前，我们先稍微探索一下中间步骤，以便更直观地理解 BART 的工作方式。
{numref}`fig:bart_bikes_samples` 展示了代码块 [bart_model_gauss](bart_model_gauss) 所定义模型
的后验树样本。上方是 `m=50` 棵树中的 3 棵单独的树。树实际返回的值用实心圆点表示，连接它们的
线只是视觉辅助。数据范围（每小时租出的自行车数量）大约是 0 到 800。因此，即使图中省略了数据，
也可以看出单棵树的拟合相当粗糙，而且在数据尺度上，这些分段函数大体是平坦的。这符合前面关于
树作为**弱学习器**的讨论。由于这里使用高斯似然，模型允许出现负的计数值。

下方则是后验样本，每个样本都是 $m$ 棵树之和。

```{figure} img/chp07/BART_bikes_samples.png
:name: fig:bart_bikes_samples
:width: 8.00in
后验树实现。上图：从后验中抽取的 3 棵单独的树。下图：3 个后验样本，每个样本都是 $m$ 棵树之和。
BART 实际采样到的取值用圆点表示，虚线只是视觉辅助；小点（仅见于下图）表示观测到的自行车租赁数。
```

> **中文版现代化说明**：PyMC-BART 0.11 的稳定公共 API 可以从 `InferenceData` 取得树之和的
> 后验，但**不能公开、稳定地暴露每一棵单独树**。更新版笔记本曾通过
> `μ.owner.op.all_trees` 访问内部实现来绘制上图；这不是受支持的公共接口，可能随版本变化。
> 因此，使用 0.11 公共 API 可以直接复现下方“树之和”面板，而上方单树面板只能依赖私有内部
> 结构或预先生成的图，不能把该私有访问写成现代公共 API 示例。

{numref}`fig:bart_bikes` 展示了用 BART 拟合自行车数据集（一天中每个小时与自行车租赁数之间的
关系）的结果。与用样条创建的 {numref}`fig:bikes_data2` 相比，这幅图给出了相似的拟合。最明显
的差别是，BART 的拟合比样条拟合更为锯齿状；当然，这并不是说两者不存在其他差别，例如 HDI 的
宽度也可能不同。

```{figure} img/chp07/BART_bikes.png
:name: fig:bart_bikes
:width: 8.00in
用 BART（具体为 `bart_model`）拟合自行车数据（黑点）的结果。阴影带表示均值的 94% HDI，
蓝线表示均值趋势。请与 {numref}`fig:bikes_data2` 比较。
```

> **中文版补充**：图注中的 `bart_model` 是原文名称，而本节实际带标签的高斯模型代码块名为
> `bart_model_gauss`；该图注应理解为指向本节的高斯 BART 拟合。

BART 文献常强调，它通常无需调参就能给出有竞争力的结果 [^5]。例如，与拟合样条相比，我们不必
操心手动设置结点，也不必选择一个用来正则化结点的先验。当然，也有人会认为，在某些问题中能够
调整结点对当前任务有益；这种看法同样合理。
""",
    ),
    # Source: markdown/chp_07.md lines 478-550.
    (
        "ch07-generalized-bart-prose",
        r"""
(generalized-bart-models)=

## 广义 BART 模型

PyMC3 的 BART 实现试图让不同似然的使用变得简单 [^6]，这与我们在[第 3 章](chap2)看到的
广义线性模型相似。下面看看如何把 Bernoulli 似然与 BART 结合。这个例子使用“太空流感”数据集。
这种疾病主要影响年轻人和老年人，而不太影响中年人。幸运的是，太空流感并不值得真正担忧，因为它
完全是虚构的。数据集中记录了接受太空流感检测的人是否患病——患病为 1，健康为 0——以及他们的
年龄。以代码块 [bart_model_gauss](bart_model_gauss) 中带高斯似然的 BART 模型为参照，可以
看到二者的差别很小。

> **中文版现代化说明**：原代码把逆链接作为
> `pm.BART(..., inv_link="logistic")` 传入。PyMC-BART 0.11 的 `pmb.BART` 没有这个
> `inv_link` 参数；现代写法是先让 BART 表示实数尺度上的 $\mu$，再显式计算
> `p = pm.math.sigmoid(μ)`，把概率约束到 $[0,1]$。下面的代码对这一变化作了明确标注，而没有
> 静默沿用旧参数。

```{code-block} python
:name: bart_model_bern
:caption: bart_model_bern

with pm.Model() as model:
    μ = pmb.BART("μ", X=X, Y=Y, m=50)
    p = pm.Deterministic("p", pm.math.sigmoid(μ))
    y = pm.Bernoulli("y", p=p, observed=Y)
    idata_bart_b = pm.sample(2000, random_seed=42)
```

首先，不再需要定义参数 $\sigma$，因为 Bernoulli 分布只有一个参数 `p`。原版 BART 定义多了一个
参数 `inv_link`，也就是逆链接函数，用于把 $\mu$ 的取值限制在区间 $[0,1]$ 内。原书为此让
PyMC3 使用逻辑函数，正如[第 3 章](chap2)逻辑回归中的做法；在上面的现代代码中，同一变换由
显式的 `pm.math.sigmoid` 完成。

{numref}`fig:BART_space_flu_comp` 使用 LOO 比较了代码块
[bart_model_bern](bart_model_bern) 在 4 个 $m$ 值——2、10、20、50——下的结果。
{numref}`fig:BART_space_flu_fit` 则展示数据、拟合函数与 94% HDI 带。按照原书该次 LOO 结果，
$m=10$ 和 $m=20$ 都拟合得较好。这与视觉检查在定性上相符：$m=2$ 明显欠拟合（ELPD 较低，
但样本内与样本外 ELPD 的差别不算大）；$m=50$ 看起来过拟合（ELPD 较低，而且样本内与样本外
ELPD 差别很大）。

> **中文版现代化说明**：LOO 排名不是一个脱离具体运行而固定不变的事实。后验采样、随机种子、
> 软件版本以及数据预处理都可能让相近模型的排名发生变化；更新版笔记本当前一次运行甚至把
> $m=50$ 排在第一。因此，正文中“$m=10$ 最佳”的说法只描述原书那次随机拟合，不应当被当作
> 确定性结论。复现时应检查本次运行的标准误、差值标准误和 Pareto $\hat k$ 诊断。

```{figure} img/chp07/BART_space_flu_comp.png
:name: fig:BART_space_flu_comp
:width: 8.00in
用 LOO 比较代码块 [bart_model_bern](bart_model_bern) 在 $m$ 分别为 2、10、20、50 时的结果。
按照原书该次 LOO 计算，$m=10$ 给出了最佳拟合。
```

```{figure} img/chp07/BART_space_flu_fit.png
:name: fig:BART_space_flu_fit
:width: 8.00in
BART 对太空流感数据集的拟合，分别使用 4 个 $m$ 值（2、10、20、50）。与 LOO 结果一致，
$m$ 的模型欠拟合，而 $m=50$ 的模型过拟合。
```

> **中文版补充**：上一图注中的“$m$ 的模型欠拟合”是原文遗漏具体数值的笔误；结合正文和四个
> 面板，应为“$m=2$ 的模型欠拟合”。这是对图 7.7 缺失 `m=2` 的明确修正。

到目前为止，为了简单起见，我们讨论的都是单协变量回归。不过，也可以拟合包含更多协变量的数据集。
从 PyMC3 实现角度看，这很直接：只需传入一个含有多个协变量的二维数组 $X$。但这也会引出一些
有趣的统计问题，例如：怎样轻松解释包含许多协变量的 BART 模型？怎样判断每个协变量对结果贡献了
多少？接下来几节将展示相应做法。
""",
    ),
    # Source: markdown/chp_07.md lines 551-575.
    (
        "ch07-interpretability-prose",
        r"""
(interpretability-of-barts)=

## BART 的可解释性

单棵决策树通常很容易解释，但把许多棵树相加以后就不再如此。有人可能以为，这是因为树相加会产生
某种奇怪、无法辨认或难以刻画的对象；实际上，树的和仍然是一棵树。这个**组装后**的树之所以难以
解释，是因为面对复杂问题时，它的决策规则很难把握。这有点像在钢琴上演奏乐曲：弹出单个音符相当
容易，但要把许多音符组合成悦耳的音乐，既会带来声音的丰富性，也会增加逐项解释的复杂度。

直接检查树的和仍可能提供一些有用信息（见 {ref}`sec:variable_selection` 一节），但它不会像
一棵简单的单树那样透明、有用。因此，为了解释 BART 模型的结果，我们通常依赖模型诊断工具
{cite:p}`Molnarbook, Molnar2020`，例如那些也用于多元线性回归和其他非参数方法的工具。下面讨论
两个相互关联的工具：**部分依赖图**（Partial Dependence Plot，PDP）
{cite:p}`Friedman2001`，以及**个体条件期望图**（Individual Conditional Expectation，ICE）
{cite:p}`Goldstein2014`。
""",
    ),
    # Source: markdown/chp_07.md lines 576-690.
    (
        "ch07-partial-dependence-prose",
        r"""
(partial-dependence-plots)=

### 部分依赖图

BART 文献中一种非常常见的方法称为部分依赖图（PDP）{cite:p}`Friedman2001`，参见
{numref}`fig:pdp_fake_example`。PDP 展示的是：改变某个协变量时，预测变量的取值如何变化，
同时对其余协变量的边缘分布取平均。也就是说，我们计算并绘制：

```{math}
:label: eq:partial_dependence
\tilde{Y}_{\boldsymbol{X}_i}= \mathbb{E}_{\boldsymbol{X}_{-i}}[\tilde{Y}(\boldsymbol{X}_i, \boldsymbol{X}_{-i})] \approx \frac{1}{n}\sum_{j=1}^{n} \tilde{Y}(\boldsymbol{X}_i, \boldsymbol{X}_{-ij})
```

其中，$\tilde{Y}_{\boldsymbol{X}_i}$ 是把除 $i$ 以外的所有变量
（$\boldsymbol{X}_{-i}$）边缘化之后，预测变量关于 $\boldsymbol{X}_i$ 的函数。一般来说，
$X_i$ 会是由 1 个或 2 个变量组成的子集，因为更高维的图通常很难绘制。

如公式 {eq}`eq:partial_dependence` 所示，可以通过对“以观测到的 $\boldsymbol{X}_{-i}$ 为条件
所得到的预测值”取平均，来数值近似这个期望。但要注意，这意味着
$\boldsymbol{X}_i,\boldsymbol{X}_{-ij}$ 中的一些组合可能并不对应任何真实观测到的组合，甚至可能
是根本不可能观测到的组合。这和[第 3 章](chap2)介绍反事实图时讨论的问题相似；事实上，部分
依赖图就是一种反事实工具。

```{figure} img/chp07/partial_dependence_plot.png
:name: fig:pdp_fake_example
:width: 8.00in
部分依赖图。在把其余变量（$X_{-i}$）的贡献边缘化之后，每个变量 $X_i$ 对 $Y$ 的部分贡献。
灰色带表示 94% HDI。均值与 HDI 带都经过平滑处理（见 `plot_ppd` 函数）。每幅子图底部由黑色
短线组成的 rug plot 表示该协变量的观测值。
```

> **中文版现代化说明**：上述原图注以及本节后面的自行车图注都把函数名误写成了 `plot_ppd`；
> 正确名称是 `plot_pdp`。在 PyMC-BART 0.11 中，公共函数写作 `pmb.plot_pdp(...)`。

{numref}`fig:pdp_fake_example` 是在用 BART 拟合一组合成数据后得到的 PDP。数据生成过程为：
$Y\sim\mathcal{N}(0,1)$，$X_0\sim\mathcal{N}(Y,0.1)$，
$X_1\sim\mathcal{N}(Y,0.2)$，$X_2\sim\mathcal{N}(0,1)$。可以看到，$X_0$ 与 $X_1$ 都像生成
过程所预期的那样，与 $Y$ 呈线性关系。还可以看到，$X_0$ 对 $Y$ 的影响强于 $X_1$，因为
$X_0$ 的斜率更陡。由于协变量服从高斯分布，其尾部数据更稀疏，因此尾部区域表现出更高的不确定性；
这是合理的。最后，$X_2$ 在其整个取值范围内的贡献几乎可以忽略不计。

现在回到自行车租赁数据集。这一次，我们用 4 个协变量来建模自行车租赁数（预测变量）：一天中的
小时、温度、湿度与风速。{numref}`fig:partial_dependence_plot_bikes` 展示了拟合模型后的部分
依赖图。一天中小时数的 PDP 与 {numref}`fig:bart_bikes` 很相似，后者是在没有其他变量时单独
拟合小时数得到的。随着温度升高，自行车租赁数也增加，但到某一点后趋势趋于平缓。借助外部领域
知识，可以推测这一模式是合理的：温度过低时，人们不太愿意骑车；但温度**过高**时，骑车同样不
那么有吸引力。湿度先呈平坦趋势，随后表现出负贡献；同样不难想象，较高湿度会降低人们骑车的意愿。
风速的贡献更加平坦，但仍然能看到影响：风越大，愿意租车的人似乎越少。

```{figure} img/chp07/partial_dependence_plot_bikes.png
:name: fig:partial_dependence_plot_bikes
:width: 8.00in
部分依赖图。在把其余变量（$X_{-i}$）的贡献边缘化之后，小时、温度、湿度和风速对自行车租赁数
的部分贡献。灰色带表示 94% HDI。均值与 HDI 带都经过平滑处理（见 `plot_ppd` 函数）。每幅
子图底部由黑色短线组成的 rug plot 表示该协变量的观测值。
```

计算部分依赖图时，一个假设是 $X_i$ 与 $X_{-i}$ 不相关，因此可以跨边缘分布取平均。在大多数
真实问题中，这个假设很难成立，于是部分依赖图可能会隐藏数据中的关系。尽管如此，如果所选变量
子集之间的依赖不太强，PDP 仍然可以作为有用的汇总 {cite:p}`Friedman2001`。

::: {admonition} 部分依赖的计算成本

计算部分依赖图的代价很高。每当我们想在某一点对变量 $X_i$ 求值时，都需要计算 $n$ 次预测，其中
$n$ 是样本量。对 BART 而言，为了得到一个预测 $\tilde Y$，首先要对 $m$ 棵树求和来获得 $Y$ 的
点估计；随后还要在“树之和”的整个后验分布上取平均，以得到可信区间。最终会需要相当大量的计算！
如果有必要，一种降低计算量的方法是在 $p$ 个点上对 $X_i$ 求值，并令 $p\ll n$。可以选择
$p$ 个等间距点，也可以选取若干分位数。另一种可以大幅加速的方案是：不对
$\boldsymbol{X}_{-ij}$ 边缘化，而把它固定在均值上。当然，这会损失信息，而且均值未必真正代表
底层分布。还有一种选择是对 $\boldsymbol{X}_{-ij}$ 做子采样，这对大型数据集尤其有用。
:::
""",
    ),
    # Source: markdown/chp_07.md lines 691-750.
    (
        "ch07-ice-prose",
        r"""
(individual-conditional-expectation)=

### 个体条件期望

个体条件期望（ICE）图与 PDP 紧密相关。区别在于，我们不再绘制目标协变量对预测响应的平均部分
效应，而是绘制 $n$ 条估计条件期望曲线。也就是说，ICE 图中的每条曲线都表示：把
$\boldsymbol{X}_{-ij}$ 固定在某个值时，部分预测响应如何随协变量 $\boldsymbol{X}_i$ 变化。
例子见 {numref}`fig:individual_conditional_expectation_plot_bikes`。如果在每个
$\boldsymbol{X}_{ij}$ 取值处对所有灰色曲线取平均，就得到蓝色曲线；它应当与
{numref}`fig:partial_dependence_plot_bikes` 中计算的平均部分依赖曲线相同。

```{figure} img/chp07/individual_conditional_expectation_plot_bikes.png
:name: fig:individual_conditional_expectation_plot_bikes
:width: 8.00in
个体条件期望图。把其余变量（$X_{-i}$）固定在某个观测值时，小时、温度、湿度和风速对自行车租赁
数的部分贡献。蓝色曲线是灰色曲线的平均。所有曲线都经过平滑处理（见 `plot_ice` 函数）。每幅
子图底部由黑色短线组成的 rug plot 表示该协变量的观测值。
```

ICE 图最适合变量之间存在强交互的问题；如果交互不强，PDP 与 ICE 图传达的信息相同。
{numref}`fig:pdp_vs_ice_toy` 展示了一个例子：PDP 隐藏了数据中的关系，而 ICE 图能够更好地
揭示它。该图通过用 BART 拟合以下合成数据生成：
$Y=0.2X_0-5X_1+10X_1\unicode{x1D7D9}_{X_2<0}+\epsilon$，其中
$X\sim\mathcal{U}(-1,1)$，$\epsilon\sim\mathcal{N}(0,0.1)$。注意 $X_1$ 的效应如何依赖
$X_2$ 的取值。

> **中文版现代化说明**（勘误）：权威 Markdown 正文在这里写作
> $\unicode{x1D7D9}_{X_2\geq0}$ 且 $\epsilon\sim\mathcal{N}(0,0.5)$，但更新后笔记本中生成
> 图 7.11 的代码实际使用 $\unicode{x1D7D9}_{X_2<0}$ 与标准差 $0.1$。为了让正文、静态图和
> 本章可执行代码保持一致，上式采用更新后笔记本的 `<0` / `0.1` 约定；原稿差异在本说明中完整保留，
> 而不是被静默改写。

```{figure} img/chp07/pdp_vs_ice_toy.png
:name: fig:pdp_vs_ice_toy
:width: 8.00in
部分依赖图与个体条件期望图的比较；该图采用上文勘误后、与更新笔记本一致的
$\unicode{x1D7D9}_{X_2<0}$ 和噪声标准差 $0.1$。第一幅：$X_1$ 与 $Y$ 的散点图；中间：
部分依赖图；最后：个体条件期望图。
```

在 {numref}`fig:pdp_vs_ice_toy` 第一幅图中，我们绘制 $X_1$ 与 $Y$。由于存在交互效应，$Y$
可以在给定 $X_2$ 取值后随 $X_1$ 线性增加或减少，因此图中呈现 **X 形**模式。中间的 PDP 表明
这种关系是平坦的；从**平均意义**上说这没有错，却隐藏了交互效应。相反，最后一幅 ICE 图有助于
揭示这一关系，因为每条灰色曲线都代表 $X_{0,2}$ 的一个取值 [^7]。蓝色曲线是灰色曲线的平均；
虽然它与 PDP 的均值曲线并不完全相同，但传达的是同一信息 [^8]。
""",
    ),
    # Source: markdown/chp_07.md lines 751-889.
    (
        "ch07-variable-selection-prose",
        r"""
(sec:variable_selection)=

## 变量选择

在拟合含有多个预测变量的回归时，我们经常希望知道哪些预测变量最重要。在一些情形下，我们确实
想更深入地理解不同变量如何促成特定结果，例如哪些饮食和环境因素会促发结肠癌。在另一些情形下，
收集包含许多协变量的数据在经济上可能难以负担，或者耗时太久、流程太复杂。例如在医学研究中，
从人体测量大量变量可能昂贵、耗时、令人不适，甚至给患者带来风险。因此，我们也许能在先导研究中
测量大量变量，但若要把分析扩展到更大人群，就可能需要减少变量数量。这时，我们希望保留最小的
（最便宜、最容易获得的）变量集合，同时仍维持相当高的预测能力。

原书所述 BART 模型提供了一种非常简单、几乎不增加计算成本的启发式变量重要性估计：记录一个
协变量被用作分割变量的次数。例如，{numref}`fig:decision_tree` 有两个分割节点，一个使用变量
$X_1$，另一个使用 $X_2$；仅根据这棵树，两者同等重要。如果某棵树中 $X_1$ 出现两次、$X_2$
出现一次，就会说 $X_1$ 的重要性是 $X_2$ 的两倍。对 BART 模型，变量重要性需要在 $m$ 棵树以及
全部后验样本上取平均。注意，用这种简单启发式只能报告相对重要性，没有简单办法断言“这个变量
重要，而另一个变量不重要”。

为了让解释更容易，可以把数值归一化，使每个值都落在 $[0,1]$ 内，并使总重要性为 1。人们很容易
把这些数解释成后验概率，但必须记住，这只是一种缺少强理论支持的简单启发式；更委婉地说，我们
对它的理解还不充分 {cite:p}`Liu2020`。

> **中文版现代化说明**：上面两段描述的是原书使用的**按分割次数计数并归一化**的历史方法。
> PyMC-BART 0.11 的当前公共工作流是先调用
> `pmb.compute_variable_importance(idata, bart_rv, X)`，再把返回结果交给
> `pmb.plot_variable_importance(...)`。当前图中报告的量是：完整模型的后验预测与逐步加入变量的
> 子模型预测之间的平方 Pearson 相关系数
> $R^2=\operatorname{corr}(\hat y_{\mathrm{full}},\hat y_{\mathrm{subset}})^2$，不是相对于观测
> $Y$ 计算的传统残差型决定系数。默认 `method="VI"` 仍可用后验树中的分割计数来**排序**变量，
> 但最终展示和比较的是累计子模型的 $R^2$ 及其区间；因此，原书“归一化分割次数就是重要性数值”
> 的解释不能直接套到 0.11 的图上。

{numref}`fig:bart_vi_toy` 展示了来自 3 个已知生成过程的数据集的相对变量重要性：

-   $Y\sim\mathcal{N}(0,1)$，$X_0\sim\mathcal{N}(Y,0.1)$，
    $X_1\sim\mathcal{N}(Y,0.2)$，$\boldsymbol{X}_{2:9}\sim\mathcal{N}(0,1)$。只有前两个
    自变量与响应有关，而且第一个变量的关联更强。

-   $Y=10\sin(\pi X_0X_1)+20(X_2-0.5)^2+10X_3+5X_4+\epsilon$，其中
    $\epsilon\sim\mathcal{N}(0,1)$，$\boldsymbol{X}_{0:9}\sim\mathcal{U}(0,1)$。这通常称为
    Friedman 五维测试函数 {cite:p}`Friedman2001`。前 5 个随机变量都以不同程度与 $Y$ 有关，
    后 5 个则无关。

-   $\boldsymbol{X}_{0:9}\sim\mathcal{N}(0,1)$，$Y\sim\mathcal{N}(0,1)$。所有变量都与响应
    变量无关。

```{figure} img/chp07/bart_vi_toy.png
:name: fig:bart_vi_toy
:width: 8.00in
按 `pmb.compute_variable_importance` 的重要性排序，逐步累加变量后子模型与完整模型预测的
平方相关系数 $R^2$（`m=50`）。左图：前两个输入变量迅速把 $R^2$ 推高，其余变量再加入几乎
没有提升，符合“只有前两个变量与响应有关”的生成过程。中图：前 5 个变量依次加入时 $R^2$
持续上升，之后趋于平坦，符合 Friedman 五维函数中后 5 个变量与响应无关的设定。右图：所有
10 个变量都与响应无关，因此曲线整体低平、逐个加入变量对 $R^2$ 几乎没有贡献。
```

> **中文版现代化说明**：原书这里进一步讨论了“增大树数量 $m$ 会让相对重要性变得更平坦”
> 以及“比较不同 $m$ 下的重要性排序”这一诊断思路；这依赖对同一数据集用多个 $m$ 值分别拟合
> 模型。当前 {numref}`fig:bart_vi_toy` 的三个面板都只用单一 `m=50` 拟合（见代码块
> [ch07-fit-variable-importance-models](ch07-fit-variable-importance-models)），
> 因此图中看不出随 $m$ 变化的效果；下面这段方法性讨论请理解为一般性建议，读者可自行
> 用不同 `m` 重新运行该代码块来验证，而不是对照本图直接得出结论。

一种诊断思路是观察从较小 $m$ 增加到较大 $m$ 时会发生什么{cite:p}`ChipmanBARTBayesianadditive2010, Carlson2020`：
$m$ 越大，对每棵树预测能力的要求越低，相关性较弱的特征也更有机会进入某棵树，使累计 $R^2$ 曲线
整体变得更平缓；$m$ 越小，变量之间的“竞争”越激烈，只有真正重要的变量才会进入最终的树。若某个
变量在增大 $m$ 后累计 $R^2$ 的提升变得更不明显，就说明它相对**更重要**；反之则**不太重要**。

这种评估变量重要性的方法可能有用，但也很棘手。在某些情况下，我们希望得到变量重要性的置信区间，
而不只是点估计。可以用相同参数和数据多次运行 BART 来做到这一点。不过，缺少一个把重要与不重要
变量明确分开的阈值，也可能被视为问题。已有研究提出了一些替代方法
{cite:p}`Carlson2020, Bleich2014`，其中一种可以概括为：

1.  使用较小的 $m$（例如 25）多次拟合模型（约 50 次）[^9]，记录均方根误差。

2.  根据全部 50 次运行，删除信息量最低的变量。

3.  重复步骤 1 和 2，每次让模型少一个变量；当模型达到预先指定的协变量数量时停止（不一定要
    减到只剩 1 个）。

4.  最后，选择平均均方根误差最低的模型。

Carlson {cite:p}`Carlson2020` 指出，这一过程似乎几乎总能返回与直接绘制
{numref}`fig:bart_vi_toy` 类似图形相同的结果。不过，也可以认为它更自动化——自动决策既有优点
也有缺点。当然，也完全可以先执行自动过程，再用图形做视觉检查。

现在回到有 4 个协变量的自行车租赁例子：小时、温度、湿度和风速。从
{numref}`fig:bart_vi_bikes` 可以看出，小时和温度对预测自行车租赁数比湿度或风速更重要。变量
重要性的排序也与部分依赖图（{numref}`fig:partial_dependence_plot_bikes`）和个体条件期望图
（{numref}`fig:individual_conditional_expectation_plot_bikes`）的结果在定性上相符。

```{figure} img/chp07/bart_vi_bikes.png
:name: fig:bart_vi_bikes
:width: 8.00in
按重要性排序、逐步累加协变量后子模型与完整模型（`m=50`）预测的平方相关系数 $R^2$。小时
是最重要的协变量，加入后 $R^2$ 提升最大；其次是温度；湿度和风速加入后 $R^2$ 几乎不再提升，
说明二者相关性较弱。
```
""",
    ),
    # Source: markdown/chp_07.md lines 890-944.
    (
        "ch07-pymc-bart-priors-prose",
        r"""
(priors-for-bart-in-pymc3)=

## PyMC3 中 BART 的先验

与本书其他模型相比，BART 是最像“黑箱”的模型。我们不能随意指定任何想要的先验来生成 BART
模型，而只能通过少数参数控制预定义的先验。原书中的 PyMC3 允许通过 3 个参数控制 BART 的先验：

-   树的数量 $m$；

-   树深度参数 $\alpha$；

-   分割变量上的分布。

前文已经看到改变树数量的影响；研究表明，在 50 到 200 的范围内，预测往往相当稳健。也有许多
例子说明，用交叉验证确定这个数量可能有益。我们还看到，通过扫描相对较小的 $m$，例如 25 到 100，
可以评估变量重要性。原书没有尝试改变默认值 $\alpha=0.25$，因为这一变化看起来影响更小，尽管仍
需要更多研究来理解该先验 {cite:p}`Rockova2018`。与 $m$ 一样，也可以用交叉验证调节它，以提高
效率。最后，PyMC3 允许传入一个权重向量，使不同变量具有不同的先验入选概率。当用户有证据认为
某些变量比其他变量更重要时，这会很有用；否则，最好保留均匀分布。为实现这一目标，并在需要诱导
稀疏性时改善推断，已有研究提出更复杂的 Dirichlet 型先验 [^10]。当协变量很多、只有少数协变量
可能真正有贡献、而我们事先又不知道是哪一些时，这尤其有用。遗传学研究就是常见例子：测量数百个
乃至更多基因的活性相对容易，但这些基因之间如何关联不仅尚不清楚，往往正是研究目标。

> **中文版现代化说明**：原书的“树深度参数 $\alpha$”及默认值 `0.25` 属于旧版单参数语义。
> PyMC-BART 0.11 使用
> $P(\text{在深度 }d\text{ 分割})=\alpha(1+d)^{-\beta}$，公开参数默认是
> `alpha=0.95`、`beta=2.0`；变量的固定先验选择权重通过 `split_prior` 传入。因此当前可直接
> 控制的项目至少包括 `m`、`alpha`、`beta`、`split_prior`，以及按列指定的 `split_rules` 等，
> 不能把旧版 `alpha=0.25` 与 0.11 的默认值或含义混为一谈。

大多数 BART 实现都存在于独立软件包中，有些甚至面向特定学科领域；它们通常不是概率编程语言的
一部分，因此一般也不期望用户大幅改造 BART 模型。即使原则上可以直接给树数量设置先验，实践中
通常也不这样做。相反，BART 文献常赞扬默认参数的良好表现，同时承认交叉验证还能再榨取一些性能。
PyMC3 的 BART 实现稍微偏离这一传统，允许额外的灵活性，但与高斯、泊松等分布，乃至高斯过程
这样的非参数分布在概率编程语言中的用法相比，仍然非常受限。原书作者设想，这种情况可能在不太
遥远的未来发生变化；其中一个原因是，他们希望探索更灵活的 BART 实现，让用户能够像概率编程语言
通常允许的那样，构建灵活、针对具体问题定制的模型。
""",
    ),
    # Source: markdown/chp_07.md lines 945-1037.
    (
        "ch07-exercises-prose",
        r"""
(exercises7)=

## 习题

**7E1.** 解释下列各点：

1. BART 与线性回归和样条有何不同？

2. 什么情况下你可能更愿意使用线性回归而不是 BART？

3. 什么情况下你可能更愿意使用样条而不是 BART？

**7E2.** 至少再画两棵能够解释 {numref}`fig:decision_tree` 中数据的树。

**7E3.** 画一棵比 {numref}`fig:decision_tree` 多一个内部节点、但能同样好地解释数据的树。

**7E4.** 画一棵表示你每天早上如何决定穿什么衣服的决策树，并给叶节点和根节点加上标签。

**7E5.** BART 需要哪些先验？解释先验在 BART 模型中的作用，以及这种作用与前几章讨论过的模型
有何相似、又有何不同。

**7E6.** 用自己的话解释：为什么多棵小树有时能比一棵大树更好地拟合模式？这两种方法有什么
区别？各自需要作出哪些权衡？

**7E7.** 下面给出几组数据。对每组数据拟合一个 $m=50$ 的 BART 模型，把数据和拟合结果画在
同一幅图上，并描述拟合效果。

1. `x = np.linspace(-1, 1., 200)`，`y = np.random.normal(2*x, 0.25)`；

2. `x = np.linspace(-1, 1., 200)`，`y = np.random.normal(x**2, 0.25)`；

3. 选择一个你喜欢的函数；

4. 与[第 5 章](chap3_5)习题 **5E4.** 的结果比较。

**7E8.** 对生成 {numref}`fig:bart_vi_toy` 所用的数据集计算 PDP。比较变量重要性指标与 PDP
分别提供的信息。

**7M9.** 在自行车租赁例子中，我们使用高斯分布作为似然。当计数很大时，这可以视为合理近似，
但仍会带来一些问题，例如预测出负的自行车租赁数（比如夜间的观测租赁数接近零时）。为了修复这一
问题并改进模型，可以尝试其他似然：

1. 使用泊松似然（提示：需要使用逆链接函数，请查阅 `pm.Bart` 的 docstring）。拟合结果与书中
   示例有何不同？这是更好的拟合吗？在哪种意义上更好？

2. 使用负二项似然。拟合结果与前两种有何不同？你能解释这一结果吗？

3. 这个结果与[第 5 章](chap3_5)中的结果有何不同？你能解释差异吗？

> **中文版现代化说明**：本题原文的 `pm.Bart` 同时存在大小写与包位置问题。PyMC-BART 0.11
> 应写作 `pmb.BART`。此外，泊松率参数必须为正；0.11 不使用旧式 `inv_link` 参数，应显式把
> BART 的实数输出映射到正数，例如 `λ = pm.Deterministic("λ", pm.math.exp(μ))`，再令
> `pm.Poisson(..., mu=λ)`。这里保留原题文字，并单独标出修正。

**7M10.** 用 BART 重做 {ref}`classifying_penguins` 一节中的第一个企鹅分类例子，也就是把
`"bill_length_mm"` 作为协变量，把物种 `"Adelie"` 和 `"Chistrap"` 作为响应。尝试不同的
`m`，例如 4、10、20 和 50，并像正文那样选择一个合适的值。把结果与
{numref}`fig:Logistic_bill_length` 中的拟合做视觉比较。你认为哪个模型表现最好？

> **中文版补充**：本题原文的企鹅物种名 `"Chistrap"` 是拼写错误，正确名称是
> `"Chinstrap"`。这里保留原题拼写，并明确给出修正。

**7M11.** 用 BART 重做 {ref}`classifying_penguins` 一节中的企鹅分类。设 `m=50`，使用协变量
`"bill_length_mm"`、`"bill_depth_mm"`、`"flipper_length_mm"` 和 `"body_mass_g"`。

使用部分依赖图与个体条件期望图，判断不同协变量如何影响把物种识别为 `"Adelie"` 和
`"Chinstrap"` 的概率。

重新拟合模型，但这次只使用 3 个协变量：`"bill_depth_m"`、`"flipper_length_mm"` 和
`"body_mass_g"`。结果与使用 4 个协变量时有何不同？说明理由。

> **中文版补充**：本题最后一段的 `"bill_depth_m"` 是字段名笔误，正确字段名是
> `"bill_depth_mm"`。这里没有静默改写原题，而是单独标出修正。

**7M12.** 用 BART 重做 {ref}`classifying_penguins` 一节中的企鹅分类。使用协变量
`"bill_length_mm"`、`"bill_depth_mm"`、`"flipper_length_mm"` 和 `"body_mass_g"` 构建模型，
并评估它们的相对变量重要性。将结果与上一题的 PDP 比较。
""",
    ),
    # Source: markdown/chp_07.md lines 1038-1076.
    (
        "ch07-footnotes-prose",
        r"""
## 脚注

[^1]: 你也许听说过它的非贝叶斯“表亲”：随机森林 {cite:p}`BreimanForests2001`。

[^2]: 其他方案见 {cite:p}`BalogMondrianProcessMachine2015, royMondrianProcess`。

[^3]: 节点深度定义为该节点到根节点的距离。因此，根节点本身的深度为 0，它的第一层子节点深度
    为 1，依此类推。

[^4]: 原则上可以采用完全贝叶斯的方法，从数据中估计树数量 $m$；但有报告表明，这并不总是最佳
    做法。这个方向可能仍需更多研究。

[^5]: 同一批文献通常也表明，用交叉验证调节树数量和/或树深度先验，还能带来进一步收益。

[^6]: 其他实现灵活性较低，或者需要在底层作出调整才能更换似然。

[^7]: 这个记号表示变量（$X_0$，$X_2$），也就是排除 $X_1$。

[^8]: ICE 曲线的均值与平均部分依赖曲线略有不同。这是绘图内部细节造成的，包括先在后验样本上
    取平均还是先在观测上取平均。真正重要的是整体特征；例如在本例中，两条曲线基本都是平坦的。
    此外，为了加速计算，部分依赖图只在 $X_1$ 的 10 个等间距点上求值；计算个体条件期望图时，
    则对 $X_{0,2}$ 做了子采样。

[^9]: 原始方案建议使用 10，但根据作者使用 PyMC3 BART 实现的经验，$m$ 低于 20 或 25 时可能
    出现问题。

[^10]: 原书认为，这项功能很可能会在 PyMC3 的未来版本中加入。

> **中文版现代化说明**：脚注 [^10] 是 2021 年写作时的未来展望，如今已经过时，不能当作当前
> 路线图。PyMC-BART 0.11 已是独立包，并提供固定的 `split_prior` 正权重来设置各预测变量的先验
> 分割偏好；但其 `pmb.BART` 构造函数并没有把“用户自定义的 Dirichlet 超先验”作为同等层次的
> 公共参数。因此，应把原脚注理解为历史说明，而不是声称 0.11 已完整实现了原文设想的
> Dirichlet 型稀疏先验。
""",
    ),
]

CODE_CELLS = [
    (
        "ch07-imports-runtime",
        r'''from __future__ import annotations

import hashlib
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from cycler import cycler

try:
    import pymc_bart as pmb
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "本章需要 pymc-bart 0.11.x 及 arviz-stats[xarray] >= 0.6.0；"
        f"当前缺少模块 {exc.name!r}。"
    ) from exc

BASE_SEED = 5453
PROFILE = os.environ.get(
    "BMCP_EXECUTION_PROFILE",
    os.environ.get("BMC_BOOK_PROFILE", "smoke"),
).strip().lower()
BUDGETS = {
    "smoke": {
        "draws": 50,
        "tune": 50,
        "chains": 1,
        "cores": 1,
        "progressbar": False,
        "compute_convergence_checks": False,
        "pdp_samples": 20,
        "ice_samples": 20,
        "ice_instances": 12,
        "vi_samples": 10,
    },
    "release": {
        "draws": 1_000,
        "tune": 1_000,
        "chains": 4,
        "cores": 4,
        "progressbar": True,
        "compute_convergence_checks": True,
        "pdp_samples": 200,
        "ice_samples": 100,
        "ice_instances": 30,
        "vi_samples": 50,
    },
}
if PROFILE not in BUDGETS:
    raise ValueError(
        f"BMCP_EXECUTION_PROFILE 必须是 {tuple(BUDGETS)} 之一，实际为 {PROFILE!r}"
    )
BUDGET = BUDGETS[PROFILE]

CHAPTER_DIR = Path.cwd().resolve()
DATA_DIR = CHAPTER_DIR / "data"
STATIC_FIGURE_DIR = CHAPTER_DIR / "img" / "chp07"
GENERATED_FIGURE_DIR = CHAPTER_DIR / "generated"
if not DATA_DIR.is_dir() or not STATIC_FIGURE_DIR.is_dir():
    raise FileNotFoundError("请从 zh/Chapter7_BART 目录启动本笔记本。")
GENERATED_FIGURE_DIR.mkdir(exist_ok=True)

EXPECTED_DATA_SHA256 = {
    "bikes_hour.csv": "58a5b55b8a7617b669b79414bdc11893d33228e14e89e702512917558059812d",
    "space_influenza.csv": "52e1f5205ddd8a87bf49ad1f68c97301f8ededf645eb50a1e6547b205aaed3a7",
}

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

for filename, expected_hash in EXPECTED_DATA_SHA256.items():
    actual_hash = sha256_file(DATA_DIR / filename)
    assert actual_hash == expected_hash, (filename, actual_hash)


def seed_for(label: str) -> int:
    payload = f"chapter-7:{BASE_SEED}:{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def rng_for(label: str) -> np.random.Generator:
    return np.random.default_rng(seed_for(label))


def sample_bart(*, seed_label: str, log_likelihood: bool = False):
    idata_kwargs = {"log_likelihood": True} if log_likelihood else None
    return pm.sample(
        draws=BUDGET["draws"],
        tune=BUDGET["tune"],
        chains=BUDGET["chains"],
        cores=BUDGET["cores"],
        random_seed=seed_for(seed_label),
        progressbar=BUDGET["progressbar"],
        compute_convergence_checks=BUDGET["compute_convergence_checks"],
        idata_kwargs=idata_kwargs,
        return_inferencedata=True,
    )


def posterior_matrix(idata, variable: str) -> np.ndarray:
    """Return posterior draws as (sample, observation), using named dimensions."""
    values = idata.posterior[variable].stack(sample=("chain", "draw"))
    observation_dims = [dim for dim in values.dims if dim != "sample"]
    if len(observation_dims) != 1:
        raise ValueError(f"{variable!r} 预期只有一个观测维，实际为 {values.dims}")
    matrix = values.transpose("sample", observation_dims[0]).to_numpy()
    if not np.isfinite(matrix).all():
        raise AssertionError(f"{variable!r} 后验样本包含非有限值")
    return matrix


def assert_prediction_shape(samples: np.ndarray, n_observations: int) -> None:
    assert samples.ndim == 2
    assert samples.shape[1] == n_observations
    assert samples.shape[0] == BUDGET["draws"] * BUDGET["chains"]


def assert_vi_result(result: dict, n_features: int) -> None:
    required = {"indices", "labels", "r2_mean", "r2_hdi", "preds", "preds_all"}
    assert required.issubset(result), required - set(result)
    indices = np.asarray(result["indices"], dtype=int)
    assert indices.size == n_features
    assert set(indices.ravel()) == set(range(n_features))
    r2_mean = np.asarray(result["r2_mean"], dtype=float)
    assert np.isfinite(r2_mean).all()
    assert ((0.0 <= r2_mean) & (r2_mean <= 1.0)).all()

try:
    PYMC_BART_VERSION = version("pymc-bart")
except PackageNotFoundError as exc:
    raise RuntimeError(
        "本章需要 pymc-bart 0.11.x；当前环境未安装该可选依赖。"
    ) from exc
if not PYMC_BART_VERSION.startswith("0.11."):
    raise RuntimeError(
        f"本章按 pymc-bart 0.11.x 公共 API 编写，当前版本为 {PYMC_BART_VERSION}。"
    )

az.style.use("arviz-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.sans-serif": ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.25,
        "lines.linewidth": 2.0,
        "lines.markersize": 6.0,
    }
)
# 已验证的参考分类配色；固定顺序对应 m=2、10、20、50。
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
plt.rcParams["axes.prop_cycle"] = cycler(color=SERIES_COLORS)
print(f"PyMC {pm.__version__}; ArviZ {az.__version__}; PyMC-BART {PYMC_BART_VERSION}; profile={PROFILE}")''',
        {"tags": ["setup", "imports", "deterministic", "smoke-release-budget"]},
    ),
    (
        "ch07-load-bikes-hour",
        r'''bikes = pd.read_csv(DATA_DIR / "bikes_hour.csv")
BIKE_COLUMNS = ["hour", "count", "temperature", "humidity", "windspeed"]
assert set(BIKE_COLUMNS).issubset(bikes.columns)
assert not bikes[BIKE_COLUMNS].isna().any().any()

# 与本章全部共享单车示例统一：先按源数据每 50 行抽取，再按小时排序。
bikes_sample = bikes.iloc[::50].sort_values("hour").reset_index(drop=True)
X_bike_hour = bikes_sample[["hour"]].to_numpy(dtype=float)
y_bikes = bikes_sample["count"].to_numpy(dtype=float)
assert X_bike_hour.ndim == 2 and X_bike_hour.shape[1] == 1
assert X_bike_hour.shape[0] == y_bikes.size
assert np.isfinite(X_bike_hour).all() and np.isfinite(y_bikes).all()
assert (y_bikes >= 0).all()

bikes_sample.head()''',
        {"tags": ["data", "semantic-check"]},
    ),
    (
        "ch07-fit-bikes-gaussian",
        r'''with pm.Model() as bart_bikes_gaussian:
    sigma_bikes = pm.HalfNormal("sigma_bikes", sigma=float(y_bikes.std()))
    mu_bikes = pmb.BART("mu_bikes", X_bike_hour, y_bikes, m=50)
    y_bikes_obs = pm.Normal(
        "y_bikes_obs", mu=mu_bikes, sigma=sigma_bikes, observed=y_bikes
    )
    idata_bikes_gaussian = sample_bart(seed_label="bikes-gaussian-m50")

mu_bikes_samples = posterior_matrix(idata_bikes_gaussian, "mu_bikes")
assert_prediction_shape(mu_bikes_samples, len(y_bikes))''',
        {"tags": ["model", "bart", "smoke", "release"], "budget_model": "bikes-gaussian-m50"},
    ),
    (
        "ch07-public-posterior-sums",
        r'''# 现代化说明：pymc-bart 0.11.x 的公共 API 不公开单棵树预测。
# 因而不再访问旧笔记本中的私有单棵树内部状态；
# 图 7.4 的原始历史图作为带哈希的静态资产保留。这里仅展示公共 API
# 可稳定获得的三条“树之和”后验样本，不覆盖该静态图。
fig, ax = plt.subplots(figsize=(10, 4))
x_hour = X_bike_hour[:, 0]
for sample_index, color in zip(
    np.linspace(0, mu_bikes_samples.shape[0] - 1, 3, dtype=int),
    SERIES_COLORS[:3],
):
    ax.plot(
        x_hour,
        mu_bikes_samples[sample_index],
        color=color,
        marker="o",
        markeredgecolor="white",
        markeredgewidth=1.2,
        label=f"后验样本 {sample_index}",
    )
ax.scatter(x_hour, y_bikes, color="0.15", alpha=0.35, s=24, label="观测值", zorder=3)
ax.set(xlabel="小时", ylabel="租车数量", title="BART 树之和的后验样本（公共 API）")
ax.legend(frameon=False, ncols=2)
plt.show()''',
        {"tags": ["figure", "modernization", "public-api-only"], "historical_figure": "fig:bart_bikes_samples"},
    ),
    (
        "ch07-figure-bikes-fit",
        r'''fig, ax = plt.subplots(figsize=(12, 4))
order = np.argsort(x_hour)
x_sorted = x_hour[order]
mu_sorted = mu_bikes_samples[:, order]
ax.scatter(x_hour, y_bikes, color="0.15", alpha=0.35, s=24, label="观测值", zorder=3)
az.plot_hdi(
    x_sorted,
    mu_sorted[None, ...],
    hdi_prob=0.94,
    smooth=True,
    color=SERIES_COLORS[0],
    fill_kwargs={"alpha": 0.12, "label": "94% HDI"},
    ax=ax,
)
ax.plot(x_sorted, mu_sorted.mean(axis=0), color=SERIES_COLORS[0], label="后验均值")
ax.set(xlabel="小时", ylabel="租车数量")
ax.legend(frameon=False)
fig.savefig(GENERATED_FIGURE_DIR / "BART_bikes.png", bbox_inches="tight")''',
        {"tags": ["figure", "expected-output"], "figure_id": "fig:bart_bikes"},
    ),
    (
        "ch07-load-space-influenza",
        r'''space_influenza = pd.read_csv(DATA_DIR / "space_influenza.csv")
assert {"age", "sick"}.issubset(space_influenza.columns)
assert not space_influenza[["age", "sick"]].isna().any().any()
X_space = space_influenza[["age"]].to_numpy(dtype=float)
y_space = space_influenza["sick"].to_numpy(dtype=int)
assert X_space.ndim == 2 and X_space.shape[0] == y_space.size
assert set(np.unique(y_space)) <= {0, 1}
space_rng = rng_for("space-jitter")
y_space_jittered = y_space + space_rng.normal(0.0, 0.02, size=y_space.size)

fig, ax = plt.subplots(figsize=(8, 3))
ax.scatter(X_space[:, 0], y_space_jittered, color="0.2", alpha=0.45, s=24)
ax.set(xlabel="年龄", ylabel="太空流感", yticks=[0, 1], yticklabels=["健康", "患病"])
plt.show()''',
        {"tags": ["data", "semantic-check", "figure"]},
    ),
    (
        "ch07-fit-space-models",
        r'''SPACE_TREE_COUNTS = (2, 10, 20, 50)
space_idatas = {}
space_models = {}
space_mu_rvs = {}

for m in SPACE_TREE_COUNTS:
    with pm.Model() as space_model:
        mu_space = pmb.BART("mu_space", X_space, y_space, m=m)
        p_space = pm.Deterministic("p_space", pm.math.sigmoid(mu_space))
        pm.Bernoulli("y_space_obs", p=p_space, observed=y_space)
        idata_space = sample_bart(seed_label=f"space-bernoulli-m{m}", log_likelihood=True)
    space_models[m] = space_model
    space_mu_rvs[m] = mu_space
    space_idatas[f"m={m}"] = idata_space

for m, idata_space in zip(SPACE_TREE_COUNTS, space_idatas.values()):
    probability_samples = posterior_matrix(idata_space, "p_space")
    assert_prediction_shape(probability_samples, len(y_space))
    assert ((0.0 <= probability_samples) & (probability_samples <= 1.0)).all()
    assert "log_likelihood" in idata_space.groups()''',
        {"tags": ["model", "bart", "bernoulli", "smoke", "release", "semantic-check"], "budget_model": "space-model-grid"},
    ),
    (
        "ch07-compare-space-models",
        r'''space_compare = az.compare(
    space_idatas,
    ic="loo",
    method="stacking",
    seed=seed_for("space-loo-comparison"),
)
assert set(space_compare.index) == {f"m={m}" for m in SPACE_TREE_COUNTS}
# 不把任何特定随机 LOO 排名写成构建断言。
space_compare''',
        {"tags": ["loo", "semantic-check", "stochastic-ranking"]},
    ),
    (
        "ch07-figure-space-compare",
        r'''ax = az.plot_compare(space_compare, figsize=(10, 3), legend=False)
ax.set_title("不同树数的太空流感 BART：LOO 比较")
ax.set_xlabel("期望对数逐点预测密度")
ax.figure.savefig(GENERATED_FIGURE_DIR / "BART_space_flu_comp.png", bbox_inches="tight")''',
        {"tags": ["figure", "expected-output"], "figure_id": "fig:BART_space_flu_comp"},
    ),
    (
        "ch07-figure-space-fits",
        r'''fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
space_order = np.argsort(X_space[:, 0])
x_space_sorted = X_space[:, 0][space_order]

for color, ax, m in zip(SERIES_COLORS, axes.ravel(), SPACE_TREE_COUNTS):
    p_samples = posterior_matrix(space_idatas[f"m={m}"], "p_space")[:, space_order]
    ax.scatter(
        X_space[:, 0],
        y_space_jittered,
        color="0.2",
        alpha=0.35,
        s=20,
        label="观测值",
    )
    az.plot_hdi(
        x_space_sorted,
        p_samples[None, ...],
        hdi_prob=0.94,
        smooth=False,
        color=color,
        fill_kwargs={"alpha": 0.12, "label": "94% HDI"},
        ax=ax,
    )
    ax.plot(x_space_sorted, p_samples.mean(axis=0), color=color, label="后验均值")
    ax.set_title(f"m={m}")
    ax.set_yticks([0, 1], labels=["健康", "患病"])
    ax.legend(frameon=False, fontsize=8)
fig.supxlabel("年龄")
fig.supylabel("太空流感")
fig.tight_layout()
fig.savefig(GENERATED_FIGURE_DIR / "BART_space_flu_fit.png", bbox_inches="tight")''',
        {"tags": ["figure", "expected-output"], "figure_id": "fig:BART_space_flu_fit"},
    ),
    (
        "ch07-synthetic-pdp-data",
        r'''pdp_rng = rng_for("synthetic-pdp")
y_pdp = pdp_rng.normal(0.0, 1.0, size=250)
X_pdp = pdp_rng.normal(0.0, 1.0, size=(250, 3))
X_pdp[:, 0] = pdp_rng.normal(y_pdp, 0.1)
X_pdp[:, 1] = pdp_rng.normal(y_pdp, 0.2)
assert X_pdp.shape == (y_pdp.size, 3)
assert np.isfinite(X_pdp).all() and np.isfinite(y_pdp).all()

fig, ax = plt.subplots(figsize=(8, 4))
for index, color in enumerate(SERIES_COLORS[:3]):
    ax.scatter(X_pdp[:, index], y_pdp, s=18, alpha=0.35, color=color, label=fr"$X_{index}$")
ax.set(xlabel="协变量", ylabel="响应 Y")
ax.legend(frameon=False, ncols=3)
plt.show()''',
        {"tags": ["synthetic-data", "deterministic", "semantic-check", "figure"]},
    ),
    (
        "ch07-fit-synthetic-pdp",
        r'''with pm.Model() as bart_pdp_model:
    mu_pdp = pmb.BART("mu_pdp", X_pdp, y_pdp, m=50)
    sigma_pdp = pm.HalfNormal("sigma_pdp", sigma=1.0)
    pm.Normal("y_pdp_obs", mu=mu_pdp, sigma=sigma_pdp, observed=y_pdp)
    idata_pdp = sample_bart(seed_label="synthetic-pdp-m50")

mu_pdp_samples = posterior_matrix(idata_pdp, "mu_pdp")
assert_prediction_shape(mu_pdp_samples, len(y_pdp))''',
        {"tags": ["model", "bart", "smoke", "release"], "budget_model": "synthetic-pdp-m50"},
    ),
    (
        "ch07-figure-synthetic-pdp",
        r'''pdp_axes = pmb.plot_pdp(
    mu_pdp,
    X_pdp,
    Y=y_pdp,
    grid="long",
    samples=BUDGET["pdp_samples"],
    random_seed=seed_for("plot-synthetic-pdp"),
    color=SERIES_COLORS[0],
    color_mean=SERIES_COLORS[0],
    alpha=0.12,
)
for index, ax in enumerate(np.asarray(pdp_axes).ravel()):
    ax.set_xlabel(fr"$X_{index}$")
    ax.set_ylabel("Y 的偏依赖")
plt.gcf().savefig(GENERATED_FIGURE_DIR / "partial_dependence_plot.png", bbox_inches="tight")
assert np.asarray(pdp_axes).size == X_pdp.shape[1]''',
        {"tags": ["figure", "expected-output", "semantic-check"], "figure_id": "fig:pdp_fake_example"},
    ),
    (
        "ch07-prepare-bikes-multivariate",
        r'''BIKE_FEATURES = ["hour", "temperature", "humidity", "windspeed"]
BIKE_FEATURE_LABELS_ZH = ["小时", "温度", "湿度", "风速"]
X_bikes_multi = bikes_sample[BIKE_FEATURES].copy()
y_bikes_multi = bikes_sample["count"].to_numpy(dtype=float)
assert X_bikes_multi.shape == (y_bikes_multi.size, len(BIKE_FEATURES))
assert np.isfinite(X_bikes_multi.to_numpy(dtype=float)).all()''',
        {"tags": ["data", "semantic-check"]},
    ),
    (
        "ch07-fit-bikes-multivariate",
        r'''with pm.Model() as bart_bikes_multivariate:
    sigma_bikes_multi = pm.HalfNormal("sigma_bikes_multi", sigma=float(y_bikes_multi.std()))
    mu_bikes_multi = pmb.BART("mu_bikes_multi", X_bikes_multi, y_bikes_multi, m=50)
    pm.Normal(
        "y_bikes_multi_obs",
        mu=mu_bikes_multi,
        sigma=sigma_bikes_multi,
        observed=y_bikes_multi,
    )
    idata_bikes_multi = sample_bart(seed_label="bikes-multivariate-m50")

mu_bikes_multi_samples = posterior_matrix(idata_bikes_multi, "mu_bikes_multi")
assert_prediction_shape(mu_bikes_multi_samples, len(y_bikes_multi))''',
        {"tags": ["model", "bart", "smoke", "release"], "budget_model": "bikes-multivariate-m50"},
    ),
    (
        "ch07-figure-bikes-pdp",
        r'''bike_pdp_axes = pmb.plot_pdp(
    mu_bikes_multi,
    X_bikes_multi,
    Y=y_bikes_multi,
    grid=(2, 2),
    figsize=(12, 6),
    sharey=True,
    samples=BUDGET["pdp_samples"],
    random_seed=seed_for("plot-bikes-pdp"),
    color=SERIES_COLORS[0],
    color_mean=SERIES_COLORS[0],
    alpha=0.12,
)
for label, ax in zip(BIKE_FEATURE_LABELS_ZH, np.asarray(bike_pdp_axes).ravel()):
    ax.set_xlabel(label)
    ax.set_ylabel("租车数量的偏依赖")
plt.gcf().savefig(GENERATED_FIGURE_DIR / "partial_dependence_plot_bikes.png", bbox_inches="tight")
assert np.asarray(bike_pdp_axes).size == len(BIKE_FEATURES)''',
        {"tags": ["figure", "expected-output", "semantic-check"], "figure_id": "fig:partial_dependence_plot_bikes"},
    ),
    (
        "ch07-figure-bikes-ice",
        r'''bike_ice_axes = pmb.plot_ice(
    mu_bikes_multi,
    X_bikes_multi,
    Y=y_bikes_multi,
    grid=(2, 2),
    smooth=True,
    centered=True,
    samples=BUDGET["ice_samples"],
    instances=min(BUDGET["ice_instances"], len(X_bikes_multi)),
    random_seed=seed_for("plot-bikes-ice"),
    color="0.65",
    color_mean=SERIES_COLORS[0],
    alpha=0.18,
)
for label, ax in zip(BIKE_FEATURE_LABELS_ZH, np.asarray(bike_ice_axes).ravel()):
    ax.set_xlabel(label)
    ax.set_ylabel("租车数量的条件期望")
plt.gcf().savefig(
    GENERATED_FIGURE_DIR / "individual_conditional_expectation_plot_bikes.png",
    bbox_inches="tight",
)
assert np.asarray(bike_ice_axes).size == len(BIKE_FEATURES)''',
        {"tags": ["figure", "expected-output", "semantic-check"], "figure_id": "fig:individual_conditional_expectation_plot_bikes"},
    ),
    (
        "ch07-synthetic-interaction-data",
        r'''interaction_rng = rng_for("synthetic-interaction")
X_interaction = interaction_rng.uniform(-1.0, 1.0, size=(250, 3))
interaction_indicator = (X_interaction[:, 2] < 0).astype(float)
interaction_noise = interaction_rng.normal(0.0, 0.1, size=250)
y_interaction = (
    0.2 * X_interaction[:, 0]
    - 5.0 * X_interaction[:, 1]
    + 10.0 * X_interaction[:, 1] * interaction_indicator
    + interaction_noise
)
assert X_interaction.shape == (y_interaction.size, 3)
assert np.isfinite(y_interaction).all()
# pymc-bart 0.11.0 的 plot_pdp 对非零 var_idx 存在位置索引缺陷。
# 将目标变量 X_1 放在第 0 列，可让 PDP 与 ICE 都通过公共 API 正确评估它。
X_interaction_bart = X_interaction[:, [1, 0, 2]]
assert np.array_equal(X_interaction_bart[:, 0], X_interaction[:, 1])''',
        {"tags": ["synthetic-data", "deterministic", "semantic-check"]},
    ),
    (
        "ch07-fit-interaction-bart",
        r'''# 中文版现代化说明：在 release 预算（1000 draws x 4 chains）下，本模型的
# sigma_interaction 偶尔报告 r_hat > 1.01；这不影响 mu_interaction 的形状断言，
# 也不改变下方 PDP/ICE 对比演示的教学结论，但读者若在自己环境中复现，看到
# 该收敛提示不必意外。m=200 也是本章计算量最大的单个模型：在缺少 PyTensor
# 原生编译后端的宿主上，仅这一个单元格就可能需要十余分钟。
with pm.Model() as bart_interaction_model:
    mu_interaction = pmb.BART(
        "mu_interaction", X_interaction_bart, y_interaction, m=200
    )
    sigma_interaction = pm.HalfNormal("sigma_interaction", sigma=1.0)
    pm.Normal(
        "y_interaction_obs",
        mu=mu_interaction,
        sigma=sigma_interaction,
        observed=y_interaction,
    )
    idata_interaction = sample_bart(seed_label="synthetic-interaction-m200")

mu_interaction_samples = posterior_matrix(idata_interaction, "mu_interaction")
assert_prediction_shape(mu_interaction_samples, len(y_interaction))''',
        {"tags": ["model", "bart", "smoke", "release"], "budget_model": "synthetic-interaction-m200"},
    ),
    (
        "ch07-figure-pdp-vs-ice",
        r'''fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
axes[0].scatter(X_interaction[:, 1], y_interaction, s=20, alpha=0.4, color="0.2")
axes[0].set(xlabel=r"$X_1$", ylabel="观测 Y", title="观测关系")
pmb.plot_pdp(
    mu_interaction,
    X_interaction_bart,
    Y=y_interaction,
    var_idx=[0],
    smooth=True,
    samples=BUDGET["pdp_samples"],
    random_seed=seed_for("plot-interaction-pdp"),
    color=SERIES_COLORS[0],
    color_mean=SERIES_COLORS[0],
    alpha=0.12,
    ax=axes[1],
)
axes[1].set(xlabel=r"$X_1$", ylabel="偏依赖", title="PDP")
pmb.plot_ice(
    mu_interaction,
    X_interaction_bart,
    Y=y_interaction,
    var_idx=[0],
    centered=False,
    smooth=False,
    samples=BUDGET["ice_samples"],
    instances=min(BUDGET["ice_instances"], len(X_interaction)),
    random_seed=seed_for("plot-interaction-ice"),
    color="0.65",
    color_mean=SERIES_COLORS[0],
    alpha=0.18,
    ax=axes[2],
)
axes[2].set(xlabel=r"$X_1$", ylabel="条件期望", title="ICE")
fig.tight_layout()
fig.savefig(GENERATED_FIGURE_DIR / "pdp_vs_ice_toy.png", bbox_inches="tight")''',
        {"tags": ["figure", "expected-output"], "figure_id": "fig:pdp_vs_ice_toy"},
    ),
    (
        "ch07-variable-importance-data",
        r'''vi_Xs: list[np.ndarray] = []
vi_ys: list[np.ndarray] = []

vi_rng_1 = rng_for("vi-signal-two")
y_vi_1 = vi_rng_1.normal(0.0, 1.0, size=100)
X_vi_1 = vi_rng_1.normal(0.0, 1.0, size=(100, 10))
X_vi_1[:, 0] = vi_rng_1.normal(y_vi_1, 0.1)
X_vi_1[:, 1] = vi_rng_1.normal(y_vi_1, 0.2)
vi_Xs.append(X_vi_1)
vi_ys.append(y_vi_1)

vi_rng_2 = rng_for("vi-friedman")
X_vi_2 = vi_rng_2.uniform(0.0, 1.0, size=(100, 10))
f_vi_2 = (
    10.0 * np.sin(np.pi * X_vi_2[:, 0] * X_vi_2[:, 1])
    + 20.0 * (X_vi_2[:, 2] - 0.5) ** 2
    + 10.0 * X_vi_2[:, 3]
    + 5.0 * X_vi_2[:, 4]
)
y_vi_2 = vi_rng_2.normal(f_vi_2, 1.0)
vi_Xs.append(X_vi_2)
vi_ys.append(y_vi_2)

vi_rng_3 = rng_for("vi-all-noise")
X_vi_3 = vi_rng_3.normal(0.0, 1.0, size=(100, 10))
y_vi_3 = vi_rng_3.normal(0.0, 1.0, size=100)
vi_Xs.append(X_vi_3)
vi_ys.append(y_vi_3)

assert len(vi_Xs) == len(vi_ys) == 3
for X_vi, y_vi in zip(vi_Xs, vi_ys):
    assert X_vi.shape == (y_vi.size, 10)
    assert np.isfinite(X_vi).all() and np.isfinite(y_vi).all()''',
        {"tags": ["synthetic-data", "deterministic", "semantic-check"]},
    ),
    (
        "ch07-fit-variable-importance-models",
        r'''vi_idatas = []
vi_bart_rvs = []
vi_models = []

for index, (X_vi, y_vi) in enumerate(zip(vi_Xs, vi_ys), start=1):
    with pm.Model() as vi_model:
        sigma_vi = pm.HalfNormal(f"sigma_vi_{index}", sigma=float(y_vi.std()))
        mu_vi = pmb.BART(f"mu_vi_{index}", X_vi, y_vi, m=50)
        pm.Normal(f"y_vi_obs_{index}", mu=mu_vi, sigma=sigma_vi, observed=y_vi)
        idata_vi = sample_bart(seed_label=f"vi-model-{index}-m50")
    vi_models.append(vi_model)
    vi_bart_rvs.append(mu_vi)
    vi_idatas.append(idata_vi)

for index, (idata_vi, y_vi) in enumerate(zip(vi_idatas, vi_ys), start=1):
    vi_mu_samples = posterior_matrix(idata_vi, f"mu_vi_{index}")
    assert_prediction_shape(vi_mu_samples, len(y_vi))''',
        {"tags": ["model", "bart", "smoke", "release"], "budget_model": "variable-importance-grid"},
    ),
    (
        "ch07-figure-variable-importance-toy",
        r'''vi_results = []
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)
for index, (idata_vi, mu_vi, X_vi, model_vi, ax) in enumerate(
    zip(vi_idatas, vi_bart_rvs, vi_Xs, vi_models, axes),
    start=1,
):
    result = pmb.compute_variable_importance(
        idata_vi,
        mu_vi,
        X_vi,
        model=model_vi,
        method="VI",
        samples=BUDGET["vi_samples"],
        random_seed=seed_for(f"compute-vi-{index}"),
    )
    assert_vi_result(result, X_vi.shape[1])
    vi_results.append(result)
    importance_order = np.asarray(result["indices"], dtype=int)
    ordered_labels = np.asarray(
        [fr"$X_{column}$" for column in range(X_vi.shape[1])],
        dtype=object,
    )[importance_order].tolist()
    # 中文版现代化说明：自定义 labels 会覆盖 pmb.plot_variable_importance 自带的
    # 累计前缀；这里手动补上 "+ "，标明每个点是在前面已选变量基础上再加入该变量的
    # 累计子模型 R²，而不是该变量单独的重要性。
    ordered_labels = [
        label if position == 0 else f"+ {label}"
        for position, label in enumerate(ordered_labels)
    ]
    pmb.plot_variable_importance(
        result,
        labels=ordered_labels,
        ax=ax,
        plot_kwargs={"color_r2": SERIES_COLORS[0], "color_ref": "0.35"},
    )
    ax.set_title(("两个信号变量", "Friedman 五维函数", "全部为噪声")[index - 1])
    ax.set_ylabel(r"预测相关性 $R^2$")
fig.tight_layout()
fig.savefig(GENERATED_FIGURE_DIR / "bart_vi_toy.png", bbox_inches="tight")''',
        {"tags": ["figure", "expected-output", "semantic-check", "modernized-method"], "figure_id": "fig:bart_vi_toy"},
    ),
    (
        "ch07-fit-bikes-variable-importance",
        r'''with pm.Model() as bart_bikes_vi_model:
    sigma_bikes_vi = pm.HalfNormal("sigma_bikes_vi", sigma=float(y_bikes_multi.std()))
    mu_bikes_vi = pmb.BART("mu_bikes_vi", X_bikes_multi, y_bikes_multi, m=50)
    pm.Normal(
        "y_bikes_vi_obs",
        mu=mu_bikes_vi,
        sigma=sigma_bikes_vi,
        observed=y_bikes_multi,
    )
    idata_bikes_vi = sample_bart(seed_label="bikes-variable-importance-m50")

mu_bikes_vi_samples = posterior_matrix(idata_bikes_vi, "mu_bikes_vi")
assert_prediction_shape(mu_bikes_vi_samples, len(y_bikes_multi))''',
        {"tags": ["model", "bart", "smoke", "release"], "budget_model": "bikes-variable-importance-m50"},
    ),
    (
        "ch07-figure-bikes-variable-importance",
        r'''bikes_vi_result = pmb.compute_variable_importance(
    idata_bikes_vi,
    mu_bikes_vi,
    X_bikes_multi,
    model=bart_bikes_vi_model,
    method="VI",
    samples=BUDGET["vi_samples"],
    random_seed=seed_for("compute-bikes-vi"),
)
assert_vi_result(bikes_vi_result, len(BIKE_FEATURES))
bikes_importance_order = np.asarray(bikes_vi_result["indices"], dtype=int)
ordered_bike_labels = np.asarray(
    BIKE_FEATURE_LABELS_ZH,
    dtype=object,
)[bikes_importance_order].tolist()
fig, ax = plt.subplots(figsize=(10, 3.5))
pmb.plot_variable_importance(
    bikes_vi_result,
    labels=ordered_bike_labels,
    ax=ax,
    plot_kwargs={"color_r2": SERIES_COLORS[0], "color_ref": "0.35"},
)
ax.set_ylabel(r"预测相关性 $R^2$")
ax.set_title("共享单车模型的变量重要性")
fig.tight_layout()
fig.savefig(GENERATED_FIGURE_DIR / "bart_vi_bikes.png", bbox_inches="tight")''',
        {"tags": ["figure", "expected-output", "semantic-check", "modernized-method"], "figure_id": "fig:bart_vi_bikes"},
    ),
]


def _prose_cell(index: int) -> dict[str, Any]:
    cell_id, source = PROSE_CELLS[index]
    return {
        "type": "markdown",
        "source": source,
        "id": cell_id,
        "metadata": {
            "kind": "translated-source",
            "provenance": {
                "source_path": "markdown/chp_07.md",
                "source_range": PROSE_SOURCE_RANGES[index],
                "note": "完整简体中文翻译；更正与现代化内容均以固定标签显式标注。",
            },
            "zh": {
                "authority": "markdown/chp_07.md",
                "language": "zh-CN",
            },
        },
    }


def _code_cell(index: int) -> dict[str, Any]:
    cell_id, source, extra_metadata = CODE_CELLS[index]
    metadata = dict(extra_metadata)
    metadata["kind"] = "modernized-code"
    metadata["provenance"] = {
        "source_path": "notebooks_updated/chp_07.ipynb",
        "source_range": CODE_SOURCE_CELLS[index],
        "note": "按 PyMC 5.28 / pymc-bart 0.11 公共 API 现代化，保持原代码顺序。",
    }
    zh_metadata = dict(metadata.get("zh", {}))
    zh_metadata.update(
        {
            "authority": "notebooks_updated/chp_07.ipynb",
            "output_name": cell_id,
        }
    )
    metadata["zh"] = zh_metadata
    return {"type": "code", "source": source, "id": cell_id, "metadata": metadata}


RUNTIME_CONTRACT_CELL = {
    "type": "markdown",
    "id": "ch07-runtime-contract",
    "source": r'''> **中文版现代化说明**
>
> 本章代码以当前仓库的 Python 3.12、PyMC 5.28.x 与 ArviZ 0.23.x 为基础，
> 并明确锁定与之兼容的 `pymc-bart 0.11.x` 公共 API。运行时还需要
> `arviz-stats[xarray] >= 0.6.0`。代码不访问单棵树私有状态；PDP、ICE 与
> 变量重要性分别使用 `plot_pdp`、`plot_ice`、`compute_variable_importance`
> 和 `plot_variable_importance`。`pymc-bart 0.11.0` 的 `plot_pdp` 对非零
> `var_idx` 有位置索引缺陷，因此交互示例把目标变量重排到第 0 列后再调用
> 公共 API。
>
> `BMCP_EXECUTION_PROFILE=smoke|release` 选择两套显式预算；缺省为 `smoke`，
> 以避免误触发整章采样。当前共享编排器不会自动导出该变量，因此发布运行
> 必须显式设置 `BMCP_EXECUTION_PROFILE=release`。历史笔记本输出只用于
> 溯源，不能作为当前环境执行成功或收敛通过的证据。''',
    "metadata": {
        "kind": "modernization-note",
        "provenance": {
            "source_path": "markdown/chp_07.md; notebooks_updated/chp_07.ipynb",
            "source_range": "chapter-wide API and execution contract",
            "note": "中文版新增的兼容性、复现性与执行预算说明。",
        },
        "zh": {"language": "zh-CN", "role": "runtime-contract"},
    },
}

# Interleave translated prose with the updated notebook code while preserving the
# exact relative order of all 25 code cells.
cells = [
    _prose_cell(0),
    RUNTIME_CONTRACT_CELL,
    _code_cell(0),
    *(_prose_cell(index) for index in range(1, 7)),
    *(_code_cell(index) for index in range(1, 5)),
    _prose_cell(7),
    *(_code_cell(index) for index in range(5, 10)),
    _prose_cell(8),
    _prose_cell(9),
    *(_code_cell(index) for index in range(10, 16)),
    _prose_cell(10),
    *(_code_cell(index) for index in range(16, 20)),
    _prose_cell(11),
    *(_code_cell(index) for index in range(20, 25)),
    _prose_cell(12),
    _prose_cell(13),
    _prose_cell(14),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_exact_hashes(directory: Path, expected: dict[str, str]) -> None:
    actual_names = {path.name for path in directory.iterdir() if path.is_file()}
    if actual_names != set(expected):
        raise AssertionError(
            f"{directory}: missing={sorted(set(expected) - actual_names)}, "
            f"extra={sorted(actual_names - set(expected))}"
        )
    for filename, expected_hash in expected.items():
        actual_hash = _sha256(directory / filename)
        if actual_hash != expected_hash:
            raise AssertionError(f"{directory / filename}: {actual_hash} != {expected_hash}")


def validate_assets() -> None:
    for relative_path, expected_hash in SOURCE_SHA256.items():
        path = REPOSITORY_ROOT / relative_path
        if not path.is_file() or _sha256(path) != expected_hash:
            raise AssertionError(f"权威源缺失或哈希不符: {relative_path}")
    _validate_exact_hashes(CHAPTER_DIR / "data", DATA_SHA256)
    _validate_exact_hashes(CHAPTER_DIR / "img" / "chp07", STATIC_FIGURE_SHA256)


def _citation_keys(markdown: str) -> set[str]:
    keys: set[str] = set()
    for group in re.findall(r"\{cite:p\}`([^`]+)`", markdown):
        keys.update(key.strip() for key in group.split(",") if key.strip())
    return keys


def validate_prose() -> None:
    if len(PROSE_CELLS) != len(PROSE_SOURCE_RANGES):
        raise AssertionError((len(PROSE_CELLS), len(PROSE_SOURCE_RANGES)))
    markdown = "\n\n".join(source for _, source in PROSE_CELLS)
    for control in ("\x00", "\x08", "\x0c", "\r"):
        if control in markdown:
            raise AssertionError(f"Markdown contains control character {control!r}")

    anchors = set(re.findall(r"^\(([^)]+)\)=$", markdown, flags=re.MULTILINE))
    if not EXPECTED_ANCHORS <= anchors:
        raise AssertionError(f"缺少锚点: {sorted(EXPECTED_ANCHORS - anchors)}")
    equations = set(re.findall(r"^:label:\s*(\S+)$", markdown, flags=re.MULTILINE))
    if equations != EXPECTED_EQUATIONS:
        raise AssertionError(f"公式标签不符: {sorted(equations)}")
    figures = set(re.findall(r"^:name:\s*(fig:\S+)$", markdown, flags=re.MULTILINE))
    if figures != EXPECTED_FIGURE_IDS:
        raise AssertionError(f"图锚点不符: {sorted(figures)}")
    if _citation_keys(markdown) != EXPECTED_CITATIONS:
        raise AssertionError(
            f"引文键不符: got={sorted(_citation_keys(markdown))}"
        )
    footnotes = set(re.findall(r"^\[\^(\d+)\]:", markdown, flags=re.MULTILINE))
    if footnotes != {str(index) for index in range(1, 11)}:
        raise AssertionError(f"脚注定义不符: {sorted(footnotes)}")
    for exercise in EXPECTED_EXERCISES:
        if f"**{exercise}.**" not in markdown and f"**{exercise}**" not in markdown:
            raise AssertionError(f"缺少习题 {exercise}")
    for block_name in EXPECTED_CODE_BLOCKS:
        if f":name: {block_name}" not in markdown:
            raise AssertionError(f"缺少代码块锚点 {block_name}")
    if "{admonition}" not in markdown or "部分依赖的计算成本" not in markdown:
        raise AssertionError("缺少已翻译的部分依赖计算成本告诫框")
    if "figures/" in markdown:
        raise AssertionError("图路径尚未迁移到章内 img/chp07")
    for filename in STATIC_FIGURE_SHA256:
        if f"img/chp07/{filename}" not in markdown:
            raise AssertionError(f"正文未引用图像 {filename}")
    if markdown.count("中文版现代化说明") < 10:
        raise AssertionError("现代化说明数量不足，可能发生静默改写")
    if "中文版补充" not in markdown:
        raise AssertionError("缺少中文版补充标签")
    required_modernization_terms = (
        "pmb.BART",
        "pm.math.sigmoid",
        "alpha",
        "beta",
        r"R^2",
        "m=2",
        "plot_pdp",
        "Chinstrap",
        "bill_depth_mm",
        "逆链接",
        "单棵树",
        "随机",
        "图 7.11 的代码实际使用",
        "`<0` / `0.1` 约定",
        r"\unicode{x1D7D9}_{X_2<0}",
    )
    for term in required_modernization_terms:
        if term not in markdown:
            raise AssertionError(f"缺少现代化语义标记 {term!r}")
    for english_heading in (
        "# Bayesian Additive Regression Trees",
        "## Decision Trees",
        "## The BART Model",
        "## Exercises",
    ):
        if english_heading in markdown:
            raise AssertionError(f"发现未翻译标题: {english_heading}")
    cjk_count = len(re.findall(r"[㐀-鿿]", markdown))
    if cjk_count < 8_000:
        raise AssertionError(f"中文字符数过低: {cjk_count}")


def validate_code() -> None:
    if len(CODE_CELLS) != len(CODE_SOURCE_CELLS):
        raise AssertionError((len(CODE_CELLS), len(CODE_SOURCE_CELLS)))
    code_ids = []
    for cell_id, source, _ in CODE_CELLS:
        compile(source, f"<{cell_id}>", "exec")
        code_ids.append(cell_id)
    if len(code_ids) != len(set(code_ids)):
        raise AssertionError("代码单元 ID 重复")
    joined = "\n\n".join(source for _, source, _ in CODE_CELLS)
    for api in PUBLIC_BART_APIS:
        if api not in joined:
            raise AssertionError(f"缺少公共 API {api}")
    for forbidden in (
        "%matplotlib",
        "pm.BART(",
        "pmb.utils.",
        "inv_link=",
        ".owner.op",
        ".sel(draw=",
        "np.random.seed(",
        "plot_dependence",
    ):
        if forbidden in joined:
            raise AssertionError(f"发现过时或私有 API: {forbidden}")
    for required in (
        "BMCP_EXECUTION_PROFILE",
        '"smoke"',
        '"release"',
        "random_seed=",
        "pm.math.sigmoid",
        'idata_kwargs = {"log_likelihood": True}',
        "idata_kwargs=idata_kwargs",
        "interaction_indicator = (X_interaction[:, 2] < 0).astype(float)",
        "interaction_rng.normal(0.0, 0.1",
        "X_interaction_bart",
        "var_idx=[0]",
        "GENERATED_FIGURE_DIR",
    ):
        if required not in joined:
            raise AssertionError(f"缺少代码语义 {required}")
    for filename in GENERATED_FIGURES:
        if f' / "{filename}"' not in joined:
            raise AssertionError(f"缺少生成图保存步骤 {filename}")
    actual_code_order = [cell["id"] for cell in cells if cell["type"] == "code"]
    if actual_code_order != code_ids:
        raise AssertionError("代码单元未保持 notebooks_updated/chp_07.ipynb 的顺序")


def validate_cell_contract() -> None:
    allowed_fields = {"type", "source", "id", "metadata"}
    id_pattern = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
    seen: set[str] = set()
    for index, cell in enumerate(cells):
        if set(cell) != allowed_fields:
            raise AssertionError(f"cell {index} fields={sorted(cell)}")
        if cell["type"] not in {"markdown", "code"}:
            raise AssertionError(cell["type"])
        if not isinstance(cell["source"], str):
            raise AssertionError(f"cell {index} source is not str")
        cell_id = cell["id"]
        if not id_pattern.fullmatch(cell_id) or cell_id in seen:
            raise AssertionError(f"invalid or duplicate cell id {cell_id!r}")
        seen.add(cell_id)
        metadata = cell["metadata"]
        if not isinstance(metadata, dict):
            raise AssertionError(f"cell {cell_id} metadata is not dict")
        if not isinstance(metadata.get("kind"), str) or not metadata["kind"]:
            raise AssertionError(f"cell {cell_id} lacks direct metadata.kind")
        provenance = metadata.get("provenance")
        if not isinstance(provenance, dict):
            raise AssertionError(f"cell {cell_id} lacks direct metadata.provenance")
        for field in ("source_path", "source_range", "note"):
            if not isinstance(provenance.get(field), str) or not provenance[field]:
                raise AssertionError(f"cell {cell_id} provenance lacks {field}")


def validate_manifest() -> None:
    manifest = tomllib.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if set(manifest) != {"schema_version", "unit", "notebook", "execution", "assets"}:
        raise AssertionError(f"manifest fields={sorted(manifest)}")
    if manifest["schema_version"] != 1:
        raise AssertionError(manifest["schema_version"])
    if set(manifest["unit"]) != {"id", "title", "status", "builder", "environment"}:
        raise AssertionError(manifest["unit"])
    if set(manifest["notebook"]) != {"file", "org_file"}:
        raise AssertionError(manifest["notebook"])
    if set(manifest["execution"]) != {"cwd", "kernel", "timeout", "allow_errors"}:
        raise AssertionError(manifest["execution"])
    if set(manifest["assets"]) != {"directory"}:
        raise AssertionError(manifest["assets"])
    expected = {
        "id": "chapter-7",
        "status": "available",
        "builder": "Chapter7_BART/build.py",
        "environment": "bart",
    }
    for key, value in expected.items():
        if manifest["unit"].get(key) != value:
            raise AssertionError((key, manifest["unit"].get(key), value))
    if manifest["assets"]["directory"] != "generated":
        raise AssertionError(manifest["assets"])


def runtime_blockers() -> list[str]:
    blockers: list[str] = []
    if importlib.util.find_spec("pymc_bart") is None:
        blockers.append("缺少 pymc_bart；需要 pymc-bart >=0.11.0,<0.12.0")
    if importlib.util.find_spec("arviz_stats") is None:
        blockers.append("缺少 arviz_stats；需要 arviz-stats[xarray] >=0.6.0")
    python_header = Path(sysconfig.get_path("include")) / "Python.h"
    if not python_header.is_file():
        blockers.append(f"缺少原生 PyTensor 编译头文件: {python_header}")
    if importlib.util.find_spec("pip") is None and shutil.which("uv") is None:
        blockers.append("当前虚拟环境没有 pip，系统也没有 uv；无法就地安装缺失可选依赖")
    return blockers


def validate_source() -> dict[str, int]:
    validate_assets()
    validate_prose()
    validate_code()
    validate_cell_contract()
    validate_manifest()
    return {
        "cells": len(cells),
        "markdown_cells": sum(cell["type"] == "markdown" for cell in cells),
        "code_cells": sum(cell["type"] == "code" for cell in cells),
        "data_files": len(DATA_SHA256),
        "static_figures": len(STATIC_FIGURE_SHA256),
        "generated_figures": len(GENERATED_FIGURES),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="also report required runtime blockers (never samples models)",
    )
    args = parser.parse_args()
    summary = validate_source()
    print("Chapter 7 source validation passed:", summary)
    if args.runtime:
        blockers = runtime_blockers()
        if blockers:
            print("Runtime blockers:")
            for blocker in blockers:
                print(f"- {blocker}")
            return 2
        versions = {}
        for distribution in ("numpy", "pymc", "arviz", "pymc-bart", "arviz-stats"):
            try:
                versions[distribution] = version(distribution)
            except PackageNotFoundError:
                versions[distribution] = "missing"
        print("Runtime contract available:", versions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
