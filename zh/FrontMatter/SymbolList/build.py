"""Cells-list source for the Simplified Chinese symbol list.

Importing this module performs no file I/O and generates no outputs.
"""

from zh.FrontMatter._shared import attribution_cell, cell


SOURCE_PATH = "markdown/symbollist.md"

cells = [
    attribution_cell(SOURCE_PATH),
    cell(
        "symbol-list-translation",
        "markdown",
        r"""
(symbols)=

# 符号表


$\log(x)$ x 的自然对数

$\mathbb{R}$ 实数

$\mathbb{R}^n$ n 维实向量空间

$\mathcal{A, S}$ 集合

$x \in A$ 集合成员关系。$x$ 是集合 $A$ 的一个元素

$\unicode{x1D7D9}_A$ 指示函数(indicator function)。若 $x \in A$,返回 1;否则返回 0

$a \propto b$ a 与 b 成正比

$a \underset{\sim}{\propto}  b$ a 与 b 近似成正比

$a \approx b$ a 近似等于 b
%\symbolentry{$\forall x$ 对所有 $x$} 未使用\
$a, c, \alpha, \gamma$ 标量用小写字母表示

$\mathbf{x, y}$ 向量用粗体小写字母表示,因此列向量写作 $\mathbf{x}=[x_1,\dots,x_n]^T$

$\mathbf{X, Y}$ 矩阵用粗体大写字母表示

$X, Y$ 随机变量用大写罗马字母表示

$x, y$ 随机变量的结果通常用小写罗马字母表示

$\boldsymbol{X, Y}$ 随机向量用大写倾斜粗体表示,$\boldsymbol{X} = [X_1,\dots,X_n]^T$

$\boldsymbol{\theta}$ 希腊小写字母通常用于表示模型参数。请注意,作为贝叶斯主义者,我们通常把参数视为随机变量

$\hat \theta$ $\boldsymbol{\theta}$ 的点估计(point estimate)

$\mathbb{E}_{X}[X]$ 关于 $X$ 的 $X$ 的期望,通常简写为 $\mathbb{E}[X]$

$\mathbb{V}_{X}[X]$ 关于 $X$ 的 $X$ 的方差,通常简写为 $\mathbb{V}[X]$

$X \sim p$ 随机变量 $X$ 服从分布 p

$p(\cdot)$ 概率密度函数(probability density function)或概率质量函数(probability mass function)

$p(y \mid \boldsymbol{x})$ 给定 $\boldsymbol{x}$ 时 $y$ 的概率(密度)。这是 $p(Y=y \mid \boldsymbol{X}=\boldsymbol{x})$ 的简写

$f(x)$ 关于 x 的任意函数

$f(\boldsymbol{X}; \theta, \gamma)$ $f$ 是以 $\theta$ 和 $\gamma$ 为参数、关于 $\boldsymbol{X}$ 的函数。我们用这种记法强调:$\boldsymbol{X}$ 是传给函数或模型的数据,$\theta$ 和 $\gamma$ 是参数

$\mathcal{N}(\mu, \sigma)$ 均值为 $\mu$、标准差为 $\sigma$ 的高斯分布(Gaussian distribution,也称正态分布)

$\mathcal{HN}(\sigma)$ 标准差为 $\sigma$ 的半高斯分布(Half-Gaussian distribution,也称半正态分布)

Beta$(\alpha, \beta)$ 形状参数为 $\alpha$、$\beta$ 的 Beta 分布

Expo$(\lambda)$ 率参数为 $\lambda$ 的指数分布(Exponential distribution)

$\mathcal{U}(a, b)$ 下界为 $a$、上界为 $b$ 的均匀分布(Uniform distribution)

T$(\nu, \mu, \sigma)$ 正态性等级为 $\nu$(也称自由度)、位置参数为 $\mu$(当 $\nu > 1$ 时为均值)、尺度参数为 $\sigma$(当 $\lim_{\nu\to\infty}$ 时为标准差)的 Student's t 分布。

$\mathcal{H}\text{T}( \nu \sigma)$ 正态性等级为 $\nu$(也称自由度)、尺度参数为 $\sigma$ 的半 Student's t 分布

Cauchy$(\alpha, \beta)$ 位置参数为 $\alpha$、尺度参数为 $\beta$ 的 Cauchy 分布

$\mathcal{H}\text{C}(\beta)$ 尺度参数为 $\beta$ 的半 Cauchy 分布

$\text{Laplace}(\mu, \tau)$ 均值为 $\mu$、尺度为 $\tau$ 的 Laplace 分布

Bin$(n, p)$ 试验次数为 $n$、成功概率为 $p$ 的二项分布(Binomial distribution)

Pois($\mu)$ 均值(也是方差)为 $\mu$ 的 Poisson 分布

NB($\mu, \alpha)$ Poisson 参数为 $\mu$、Gamma 分布参数为 $\alpha$ 的负二项分布(Negative Binomial distribution)

$\mathcal{G}RW(\mu, \sigma)$ 创新漂移为 $\mu$、创新标准差为 $\sigma$ 的高斯随机游走分布(Gaussian random walk distribution)

$\mathbb{KL}(p \parallel q)$ 从 $p$ 到 $q$ 的 Kullback-Leibler 散度
""",
        kind="translated-source",
        source_path=SOURCE_PATH,
        source_range="1-82",
    ),
]
