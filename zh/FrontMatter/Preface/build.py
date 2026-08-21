"""Cells-list source for the Simplified Chinese preface.

Importing this module performs no file I/O and generates no outputs.
"""

from zh.FrontMatter._shared import attribution_cell, cell


SOURCE_PATH = "markdown/preface.md"

cells = [
    attribution_cell(SOURCE_PATH),
    cell(
        "preface-opening",
        "markdown",
        r"""
(preface)=

# 前言

"贝叶斯统计"这一名称源自 Thomas Bayes(1702--1761)。他是一位长老会牧师和业余
数学家,首次推导出了我们今天所说的贝叶斯定理(Bayes' theorem);这一定理在他去世后
于 1763 年发表。不过,真正最早发展贝叶斯方法的人之一是 Pierre-Simon Laplace
(1749--1827),所以称其为"拉普拉斯统计"或许更准确一些。尽管如此,我们将遵循
Stigler's law of eponymy,也遵循传统,在本书余下部分继续使用"贝叶斯方法"这一说法。
从 Bayes、Laplace(以及许多其他先驱)的时代到今天,发生了很多事情:新的思想不断
出现,其中许多思想由计算机推动,或者因计算机而成为可能。本书旨在提供这一主题的
现代视角:从基础知识出发,建立扎实根基,进而应用现代贝叶斯工作流与工具。

我们写作本书,是为了帮助贝叶斯初学实践者成长为中级建模者。我们并不声称你读完
本书后这一转变就会自动发生,但希望本书能引导你走向富有成效的方向,尤其是在你
认真通读全书、完成练习、把书中的思想应用于自己的问题,并继续向他人学习的情况下。

更具体地说,本书面向有兴趣运用贝叶斯模型解决数据分析问题的贝叶斯实践者。人们
经常区分学术界和工业界,但本书不作这种区分:它对大学里的学生和公司的机器学习
工程师同样有用。

我们希望你在读完本书后,不仅熟悉**贝叶斯推断(Bayesian Inference)**,也能够从容地
开展**贝叶斯模型的探索性分析(Exploratory Analysis of Bayesian Models)**,其中包括
模型比较、诊断、评估以及结果沟通。我们还希望从现代计算视角教授所有这些内容。
在我们看来,采用**计算(computational)**方法,能够更好地理解和应用贝叶斯统计。例如,
与其试图证明假设是正确的,我们更关心经验证地检查假设会以何种方式被违背。这也意味着
我们会使用大量可视化(如果没有更多,只是为了避免把本书写到 1000 页)。随着阅读推进,
这种建模方法的其他含义也会逐渐明朗。

最后,正如书名所示,本书使用 Python 编程语言。更具体地说,我们主要使用 PyMC3
{cite:p}`Salvatier2016` 和 TensorFlow Probability(TFP)
{cite:p}`dillon2017tensorflow` 作为构建模型和进行推断的主要概率编程语言
(probabilistic programming languages, PPLs),并使用 ArviZ 作为贝叶斯模型探索性分析
的主要库 {cite:p}`Kumar2019`。我们无意对所有 Python PPL 作穷尽式调查与比较,因为
可选工具很多,而且演进迅速。相反,我们关注贝叶斯分析的实践层面。编程语言和库只不过是
帮助我们抵达目的地的桥梁。

虽然本书选择使用 Python 以及少数几个库,但书中涉及的统计与建模概念并不依赖特定语言
或库,在 R、Julia 和 Scala 等许多编程语言中同样可用。熟悉这些语言但不熟悉 Python 的
读者,只要找到支持等价功能的合适软件包或自行编写代码以获得动手实践,仍然能从本书中
受益。此外,作者鼓励其他人把本书的代码示例翻译到其他语言或框架中。如果你愿意这样做,
请与我们联系。
""",
        kind="translated-source",
        source_path=SOURCE_PATH,
        source_range="1-68",
    ),
    cell(
        "preface-prior-knowledge",
        "markdown",
        r"""
(prior-knowledge)=

## 预备知识

本书旨在帮助初学者成长为中级实践者,因此我们假定读者接触过贝叶斯统计的一些基本思想,
但不要求熟练掌握,例如先验、似然和后验;我们也假定读者了解随机变量、概率分布、期望等
基本统计概念。如果你对这些内容有些生疏,第 [11](app) 章中有一整个小节回顾基础统计概念。
对这些概念作更深入讲解的两本好书是 Understanding Advanced Statistical Methods
{cite:p}`WestfallUnderstandingAdvancedStatistical2013` 和 Introduction to
Probability {cite:p}`blitzstein_2019`。后者的理论性稍强,但两本书都着眼于应用。

如果你通过实践或正规训练很好地掌握了统计学,但从未接触过贝叶斯统计,仍然可以把本书
作为这一主题的入门读物。不过开篇部分(主要是前两章)节奏较快,可能需要多读几遍。

我们希望你熟悉积分、导数和对数性质等数学概念。本书使用的数学水平大致相当于技术类
高中,或者科学、技术、工程和数学专业大学一年级通常教授的水平。需要复习这些数学概念的
读者,可以观看 3Blue1Brown 的系列视频 [^1]。我们不会要求你解答很多数学练习;相反,
主要会要求你使用代码和交互式计算环境来理解并解决问题。全书中的数学公式只在有助于
更好理解贝叶斯统计建模时使用。

本书假定读者具备一些科学计算编程知识。除了 Python 语言,我们还会使用多个专用软件包,
尤其是概率编程语言。阅读本书前若曾用某种概率编程语言拟合过至少一个模型会有帮助,但
并非必要条件。有关如何搭建本书所需计算环境的参考说明,请阅读[环境安装](https://github.com/BayesianModelingandComputationInPython/BookCode_Edition1#environment-installation)。
""",
        kind="translated-source",
        source_path=SOURCE_PATH,
        source_range="70-111",
    ),
    cell(
        "preface-how-to-read",
        "markdown",
        r"""
(how-to-read-this-book)=

## 如何阅读本书

我们会先用玩具模型来理解重要概念,避免数据掩盖核心思想;然后使用真实数据集来近似真实的
实践问题,例如采样问题、重参数化、先验/后验校准等。我们鼓励你在阅读时,在交互式代码环境中
运行这些模型。

我们强烈建议你阅读并使用各个库的在线文档。虽然我们会尽力让本书自成体系,但这些工具在
网上还有大量文档;查阅它们既有助于学习本书,也有助于你独立使用这些工具。

[第 1 章](chap1)回顾或简要介绍贝叶斯推断中最基础、最核心的概念。本章内容会在全书其余
章节中反复出现并得到应用。

[第 2 章](chap1bis)介绍贝叶斯模型的探索性分析。具体而言,它会介绍贝叶斯工作流中许多
并不属于推断本身的概念。本章内容也会在全书其余章节中得到应用和再次讨论。

[第 3 章](chap2)是第一章专门讨论某种具体模型架构的章节。它介绍线性回归模型,并为接下来
5 章奠定基础。第 3 章还会完整介绍本书使用的主要概率编程语言 PyMC3 和 TensorFlow Probability。

[第 4 章](chap3)扩展线性回归模型,讨论稳健回归、层级模型和模型重参数化等更高级主题。
本章使用 PyMC3 和 TensorFlow Probability。

[第 5 章](chap3_5)介绍基函数,尤其是样条(splines)。样条是线性模型的一种扩展,使我们
能够构建更灵活的模型。本章使用 PyMC3。

[第 6 章](chap4)重点介绍时间序列模型,从把时间序列建模为回归问题,到 ARIMA 和线性高斯
状态空间模型等更复杂的模型。本章使用 TensorFlow Probability。

[第 7 章](chap6)介绍贝叶斯加法回归树(Bayesian additive regression trees, BART),
这是一种非参数模型。我们会讨论该模型的可解释性和变量重要性。本章使用 PyMC3。

[第 8 章](chap8)把注意力转向近似贝叶斯计算(Approximate Bayesian Computation, ABC)
框架。对于无法显式写出似然的问题,这一框架十分有用。本章使用 PyMC3。

[第 9 章](chap9)概述端到端贝叶斯工作流,既展示商业场景中的观察性研究,也展示科研场景中的
实验研究。本章使用 PyMC3。

[第 10 章](chap10)深入讨论概率编程语言,并展示多种不同的概率编程语言。

[第 11 章](app)在你阅读其他章节时提供辅助。由于其中的主题彼此关系较松散,你可能不需要
按顺序阅读。
""",
        kind="translated-source",
        source_path=SOURCE_PATH,
        source_range="113-175",
    ),
    cell(
        "preface-text-highlights",
        "markdown",
        r"""
(text-highlights)=

### 文本强调

本书使用**粗体**或*斜体*来强调文字。**粗体文字**用于突出新概念或强调某个概念。
*斜体文字*表示口语化或不严格的表达。提到具体代码时也会突出显示,例如 `pm.sample`。
""",
        kind="translated-source",
        source_path=SOURCE_PATH,
        source_range="177-185",
    ),
    cell(
        "preface-code-conventions",
        "markdown",
        r"""
(code)=

### 代码

本书中的代码块显示在带阴影的方框中,左侧带有行号。引用代码块时,先写章号,再写该代码块
的编号。例如:
""",
        kind="translated-source",
        source_path=SOURCE_PATH,
        source_range="186-193",
    ),
    cell(
        "preface-loop-example",
        "code",
        r"""
for i in range(3):
    print(i**2)
""",
        kind="preserved-code",
        source_path=SOURCE_PATH,
        source_range="194-197",
    ),
    cell(
        "preface-loop-output-and-repository",
        "markdown",
        r"""
```none
0
1
4
```

每当看到代码块时,请寻找对应的结果。结果通常是一幅图、一个数字、一段代码输出或一张表。
反过来说,书中的大多数图也有相应代码块。有时为了节省篇幅,我们不会在书中展示代码块,
但你仍然可以在 [GitHub 仓库](https://github.com/BayesianModelingandComputationInPython)
中访问它们。该仓库还包含部分练习的附加材料。仓库中的 notebooks 也可能包含书中没有出现的
额外图形、代码或输出,但这些内容曾用于开发书中展示的模型。GitHub 上还提供了说明,介绍如何
在你拥有的任何设备上创建标准计算环境。
""",
        kind="translated-source",
        source_path=SOURCE_PATH,
        source_range="199-215",
    ),
    cell(
        "preface-boxes",
        "markdown",
        r"""
(boxes)=

### 边注框

我们使用边注框,快速说明你需要掌握的重要统计、数学或(Python)编程概念。我们还会提供参考资料,
供你继续学习相关主题。

:::{admonition} 中心极限定理(Central Limit Theorem)

在概率论中,中心极限定理说明:在某些情况下,把相互独立的随机变量相加时,即便原始变量本身
并不服从正态分布,经过适当归一化的和也会趋向正态分布。

设 $X_1, X_2, X_3, ...$ 独立同分布(i.i.d.),均值为 $\mu$,标准差为 $\sigma$。
当 $n \rightarrow \infty$ 时,有:

```{math}
\sqrt{n} \left(\frac{\bar{X}-\mu}{\sigma} \right) \xrightarrow{\text{d}} \mathcal{N}(0, 1)
```

Introduction to Probability {cite:p}`blitzstein_2019` 是学习概率论诸多理论内容的
优秀资料,而这些理论在实践中也很有用。
:::
""",
        kind="translated-source",
        source_path=SOURCE_PATH,
        source_range="217-243",
    ),
    cell(
        "preface-code-imports-intro",
        "markdown",
        r"""
(code-imports)=

### 代码导入约定

本书导入 Python 软件包时采用以下约定。

:::{admonition} 中文版现代化说明
原版代码使用 `pymc3`、`theano` 和 `theano.tensor as tt`。本中文版的可执行示例改用当前
公开命名 `pymc`、`pytensor` 和 `pytensor.tensor as pt`;ArviZ、TensorFlow Probability
及其他导入名称保持其当前公开 API。为了让后续构建工具能够分别进行快速冒烟检查和发布级执行,
示例还通过环境变量 `BMCP_EXECUTION_PROFILE` 提供 `smoke` 与 `release` 两种可配置参数组。
此处只定义配置,不会在导入时运行采样。
:::
""",
        kind="modernization-note",
        source_path=SOURCE_PATH,
        source_range="245-250",
        note="Explains PyMC3/Theano to PyMC/PyTensor API changes.",
    ),
    cell(
        "preface-modern-imports",
        "code",
        r"""
import os

# Basic
import numpy as np
from scipy import stats
import pandas as pd
from patsy import bs, dmatrix
import matplotlib.pyplot as plt

# Exploratory Analysis of Bayesian Models
import arviz as az

# Probabilistic programming languages
import bambi as bmb
import pymc as pm
import tensorflow_probability as tfp

tfd = tfp.distributions

# Computational Backend
import pytensor
import pytensor.tensor as pt
import tensorflow as tf

EXECUTION_PROFILE = os.environ.get("BMCP_EXECUTION_PROFILE", "release").lower()
if EXECUTION_PROFILE not in {"smoke", "release"}:
    raise ValueError("BMCP_EXECUTION_PROFILE must be 'smoke' or 'release'")

if EXECUTION_PROFILE == "smoke":
    DRAWS, TUNE, CHAINS = 100, 100, 2
else:
    DRAWS, TUNE, CHAINS = 1000, 1000, 4
""",
        kind="modernized-code",
        source_path=SOURCE_PATH,
        source_range="252-274",
        note="PyMC3/Theano imports modernized to PyMC/PyTensor; execution profile added.",
    ),
    cell(
        "preface-arviz-style",
        "markdown",
        r"""
我们还使用 ArviZ 样式 `az.style.use("arviz-grayscale")`。
""",
        kind="translated-source",
        source_path=SOURCE_PATH,
        source_range="276-276",
    ),
    cell(
        "preface-interaction",
        "markdown",
        r"""
(how-to-interact-with-this-book)=

### 如何与本书互动

我们的读者并非只想成为*贝叶斯方法的阅读者*,而是要成为贝叶斯实践者。因此,我们会提供材料,
让你练习贝叶斯推断和贝叶斯模型的探索性分析。运用计算和代码是现代贝叶斯实践者必备的核心技能,
所以我们会提供可以反复尝试的示例,帮助你逐步建立直觉。我们希望你阅读、执行并修改本书代码,
然后一次又一次地重新执行。一本书能展示的例子数量有限,但你可以利用计算机为自己生成无限多的
例子。这样你不仅会学到统计概念,也会学会如何使用计算机从这些概念中创造价值。

计算机还会帮助你摆脱印刷文本的局限,例如缺少颜色、缺少动画以及难以并排比较。现代贝叶斯
实践者会利用显示器提供的灵活性和快速的计算"双重检查",而我们也特意把示例设计成能够提供
同等程度的交互性。每章末尾还包含检验学习效果并提供额外练习的习题。习题标为简单(E)、
中等(M)和困难(H)。可按要求获取解答。
""",
        kind="translated-source",
        source_path=SOURCE_PATH,
        source_range="278-301",
    ),
    cell(
        "preface-acknowledgments",
        "markdown",
        r"""
(acknowledgments)=

## 致谢

我们感谢朋友和同事们慷慨投入时间与精力,阅读早期草稿、提出建议并提供有益反馈。这些反馈
帮助我们改进本书,也帮助我们修复了书中的许多错误。感谢:

Oriol Abril-Pla,Alex Andorra,Paul Anzel,Dan Becker,Tomás Capretto,
Allen Downey,Christopher Fonnesbeck,Meenal Jhajharia,Will Kurt,Asael
Matamoros,Kevin Murphy,以及 Aki Vehtari。

[^1]: <https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw>,即使你不需要复习,
    我们也推荐观看这些视频。
""",
        kind="translated-source",
        source_path=SOURCE_PATH,
        source_range="303-317",
    ),
]
