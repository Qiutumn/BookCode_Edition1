"""Cells-list source for the Simplified Chinese foreword.

Importing this module performs no file I/O and generates no outputs.
"""

from zh.FrontMatter._shared import attribution_cell, cell


SOURCE_PATH = "markdown/foreword.md"

cells = [
    attribution_cell(SOURCE_PATH),
    cell(
        "foreword-translation",
        "markdown",
        r"""
(foreword)=

# 序言

```{epigraph}
贝叶斯建模(Bayesian modeling)为许多数据科学和决策问题提供了一种优雅的方法。
然而,要让它在实践中良好运作并不容易。尤其是,虽然 Stan、PyMC3、
TensorFlow Probability(TFP)和 Pyro 等许多软件包都能让复杂层级模型的指定
变得简单,用户仍然需要额外的工具来诊断计算结果是否正确。当事情确实出错时,
他们可能也需要有关如何处理问题的建议。

本书重点介绍 ArviZ 库。它让用户能够对贝叶斯模型进行探索性分析
(exploratory analysis of Bayesian models),例如诊断由任意推断方法生成的
后验样本。这些工具可用于诊断贝叶斯推断中的多种失效模式。本书还讨论了各种
建模策略(例如中心化,centering),用来消除许多最常见的问题。书中的大多数
示例使用 PyMC3,也有一些使用 TFP;书中还简要比较了其他概率编程语言
(probabilistic programming languages, PPLs)。

三位作者都是贝叶斯软件领域的专家,也是 PyMC3、ArviZ 和 TFP 库的主要贡献者。
他们在实践中应用贝叶斯数据分析方面也拥有丰富经验,这一点体现在本书所采用的
实践导向方法中。总的来说,我认为本书为相关文献增添了宝贵内容,并希望它能
进一步推动贝叶斯方法的采用。


-- Kevin P. Murphy
```
""",
        kind="translated-source",
        source_path=SOURCE_PATH,
        source_range="1-34",
    ),
]
