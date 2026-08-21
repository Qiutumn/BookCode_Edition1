r"""《Python 贝叶斯建模与计算》中文版术语表的规范源。"""

SOURCE = "markdown/glossary.md"


def entry(cell_id, english, chinese, definition, source_term=None, kind="translation"):
    source_term = source_term or english
    return {
        "id": cell_id,
        "type": "markdown",
        "metadata": {
            "kind": kind,
            "provenance": [f"{SOURCE}#{source_term}"],
        },
        "source": f"## {chinese} ({english})\n\n{definition}\n",
    }


cells = [
    {
        "id": "glossary-title",
        "type": "markdown",
        "metadata": {
            "kind": "translation",
            "provenance": [f"{SOURCE}#glossary"],
        },
        "source": r"""# 术语表

本术语表按英文术语排列。中文译名后保留英文原词与常用缩写,便于读者检索英文文献和软件文档。
""",
    },
    entry(
        "glossary-autocorrelation", "Autocorrelation", "自相关",
        "自相关是一个信号与其滞后副本之间的相关性。直观地说,它描述观测值的相似程度怎样随两者之间的滞后变化。MCMC 样本中较大的自相关值得警惕,因为它会降低有效样本量。",
    ),
    entry(
        "glossary-aleatoric-uncertainty", "Aleatoric Uncertainty", "偶然不确定性",
        "偶然不确定性(也称随机不确定性)来自测量或观测过程中本质上不可知或随机的因素。例如,即使我们精确复现用弓射箭时的方向、高度和力度,箭仍不会每次命中同一点,因为大气波动、箭杆振动等未受控制的条件具有随机性。",
    ),
    entry(
        "glossary-bayesian-inference", "Bayesian Inference", "贝叶斯推断",
        r"贝叶斯推断是一种通过组合概率分布而得到其他概率分布的统计推断形式。换言之,它研究条件概率或概率密度的建模与计算: $p(\boldsymbol{\theta}\mid\boldsymbol{Y}) \propto p(\boldsymbol{Y}\mid\boldsymbol{\theta})p(\boldsymbol{\theta})$。",
    ),
    entry(
        "glossary-bayesian-workflow", "Bayesian workflow", "贝叶斯工作流",
        "为给定问题设计一个足够好的模型,需要统计知识和领域知识。这样的设计通常通过称为贝叶斯工作流的迭代过程完成。该过程包括模型构建的三个环节 {cite:p}`Gelman2020`:推断、模型检查与改进、模型比较。在这里,模型比较的目的不一定只是选出所谓“最佳”模型,更重要的是加深对各个模型的理解。",
    ),
    entry(
        "glossary-causal-inference", "Causal inference", "因果推断",
        "这里主要指观测性因果推断:在不实际实施处理或干预的情况下,根据观测数据而非实验数据,估计某项处理或干预对系统的影响所使用的过程与工具。",
    ),
    entry(
        "glossary-covariance-precision", "Covariance Matrix and Precision Matrix", "协方差矩阵与精度矩阵",
        "协方差矩阵是一个方阵,包含一组随机变量中每一对变量之间的协方差;其对角线元素是各随机变量的方差。精度矩阵是协方差矩阵的逆矩阵。",
    ),
    entry(
        "glossary-design-matrix", "Design Matrix", "设计矩阵",
        "在回归分析中,设计矩阵由解释变量的取值组成。每一行代表一个观测对象,各列依次对应变量及该观测的具体取值。矩阵既可以包含表示组别成员身份的指示变量(0 和 1),也可以包含连续值。",
    ),
    entry(
        "glossary-decision-tree", "Decision tree", "决策树",
        "决策树是一种类似流程图的结构。每个内部节点表示对某项属性的“检验”(例如抛硬币是正面还是反面),每条分支表示检验结果,每个叶节点表示在计算完相关属性后得到的类别标签。从根节点到叶节点的路径构成分类规则。如果决策树用于回归,叶节点上的值也可以是连续量。",
    ),
    entry(
        "glossary-dse", "dse", "差值标准误",
        "`dse` 是两个模型逐观测 `elpd_loo` 差值的标准误。它通常小于单个模型的标准误(`az.compare` 中的 `se`),因为某些观测对所有模型而言都同样容易或同样难以预测,从而在模型结果之间引入相关性。",
    ),
    entry(
        "glossary-d-loo", "d_loo", "LOO 差值",
        "`d_loo` 是两个模型的 `elpd_loo` 之差。比较两个以上模型时,差值相对于 `elpd_loo` 最高的模型计算。",
    ),
    entry(
        "glossary-epistemic-uncertainty", "Epistemic Uncertainty", "认知不确定性",
        "**中文版现代化说明**: 原书术语表误拼为 `Epistimic`;这里采用标准拼写 `Epistemic`。认知不确定性来自观察者对系统状态缺乏知识。它涉及原则上可以知道、但实践中尚不知道的信息,而不是自然界本质上不可知的随机量(参见偶然不确定性)。例如,手边没有秤时只能用手估计物体重量,或秤的精度只能达到千克;又如设计实验或计算时忽略收费站停留、天气和道路状况等因素。认知不确定性源于无知,原则上可以通过取得更多信息而减小。",
        source_term="Epistimic Uncertainty",
        kind="modernization",
    ),
    entry(
        "glossary-statistic", "Statistic", "统计量",
        "统计量(这里是单数概念)或样本统计量,是由样本计算得到的任意量。统计量可用于估计总体(或数据生成过程)参数、描述样本或检验假设。样本均值(经验均值)和样本方差(经验方差)都是统计量。当统计量用于估计总体参数时,称为估计量;样本均值可以是估计量,后验均值也可以是另一种估计量。",
    ),
    entry(
        "glossary-elpd", "ELPD", "期望逐点对数预测密度",
        "ELPD 是 Expected Log-pointwise Predictive Density 的缩写;对离散模型也可理解为期望逐点对数预测概率。它通常通过交叉验证,或 WAIC (`elpd_waic`) 与 LOO (`elpd_loo`) 等方法估计。连续变量的概率密度可以小于或大于 1,所以其 ELPD 可以为负或为正;离散变量的对数预测概率不大于 0,相应符号应结合软件采用的报告约定解释。",
    ),
    entry(
        "glossary-exchangeability", "Exchangeability", "可交换性",
        "如果改变一列随机变量在序列中的位置不会改变它们的联合概率分布,则称该序列可交换。可交换随机变量不一定独立同分布(iid),但独立同分布随机变量一定可交换。",
    ),
    entry(
        "glossary-eabm", "Exploratory Analysis of Bayesian Models", "贝叶斯模型的探索性分析",
        "贝叶斯模型的探索性分析是成功完成贝叶斯数据分析所需、但不等同于推断本身的一组任务,包括:诊断数值推断结果的质量;模型批判,即评估模型假设和模型预测;模型选择或模型平均等模型比较;以及为特定受众准备分析结果。",
    ),
    entry(
        "glossary-hmc", "Hamiltonian Monte Carlo", "哈密顿蒙特卡洛",
        "哈密顿蒙特卡洛(HMC)是一类利用梯度高效探索概率分布的马尔可夫链蒙特卡洛(MCMC)方法,在贝叶斯统计中最常用于从后验分布获取样本。HMC 是 Metropolis--Hastings 算法的一类实例,其候选点由哈密顿动力系统产生,因此能以较高接受概率移动到离当前状态较远的位置。系统演化由时间可逆且保持体积的数值积分器模拟,最常见的是蛙跳积分器。HMC 的效率高度依赖若干超参数,所以实用方法通常采用自适应动力学,在预热或调优阶段自动调整这些超参数。",
    ),
    entry(
        "glossary-heteroscedasticity", "Heteroscedasticity", "异方差性",
        "如果一列随机变量的方差并不相同,即不具有同方差性,则称其具有异方差性,也称方差异质性。",
    ),
    entry(
        "glossary-homoscedasticity", "Homoscedasticity", "同方差性",
        "如果一列随机变量都具有相同且有限的方差,则称其具有同方差性,也称方差齐性。与之互补的概念是异方差性。",
    ),
    entry(
        "glossary-iid", "iid", "独立同分布",
        "iid 是 independent and identically distributed 的缩写。如果一组随机变量具有相同概率分布且彼此相互独立,则称它们独立同分布。独立同分布蕴含可交换性,但反过来不一定成立。",
    ),
    entry(
        "glossary-ice", "Individual Conditional Expectation", "个体条件期望",
        "个体条件期望图(ICE)展示响应变量对某个感兴趣协变量的依赖关系。它对每个样本分别计算,每个样本画一条线。与此相对,部分依赖图(PDP)展示的是协变量的平均效应。",
    ),
    entry(
        "glossary-inference", "Inference", "推断",
        "日常语言中的推断,是根据证据和推理得出结论。本书谈到推断时通常特指定义更严格的贝叶斯推断:用现有数据对模型做条件化并得到后验分布。要真正根据证据和推理得出结论,还需要完成贝叶斯推断之外的步骤,因此应从贝叶斯模型探索性分析乃至完整贝叶斯工作流的角度理解分析过程。",
    ),
    entry(
        "glossary-imputation", "Imputation", "插补",
        "插补是用某种选定方法替换缺失数据值。常见做法包括用出现最频繁的值替换,或根据其他已观测数据进行插值或模型化预测。",
    ),
    entry(
        "glossary-kde", "KDE", "核密度估计",
        "核密度估计(Kernel Density Estimation, KDE)是一种根据有限样本估计随机变量概率密度函数的非参数方法。实际使用中,KDE 也常指由该方法得到的估计密度本身。",
    ),
    entry(
        "glossary-loo", "LOO", "留一法交叉验证",
        "本书中的 LOO 通常是 Pareto 平滑重要性采样留一法交叉验证(PSIS-LOO-CV)的简称;其他文献有时用 LOO 泛指任何留一法交叉验证。",
    ),
    entry(
        "glossary-map", "Maximum a Posteriori (MAP)", "最大后验估计",
        "最大后验估计(MAP)是一种未知量估计量,取后验分布的众数。MAP 需要优化后验,而后验均值需要积分。在先验平坦或样本量趋于无穷的极限条件下,MAP 与最大似然估计等价。",
    ),
    entry(
        "glossary-odds", "Odds", "几率",
        "几率是衡量某个结果相对可能性的量,等于产生该结果的事件数与不产生该结果的事件数之比。几率在博彩和逻辑回归中都很常见。",
    ),
    entry(
        "glossary-overfitting", "Overfitting", "过拟合",
        "当模型的预测过度贴合用于拟合的数据,以至于不能很好适应新数据时,就发生了过拟合。从参数数量看,过拟合模型包含的数据无法充分支持的过多参数。任意复杂的模型不仅会拟合信号,还会拟合噪声,因而导致较差的样本外预测。",
    ),
    entry(
        "glossary-pdp", "Partial Dependence Plots", "部分依赖图",
        "部分依赖图(Partial Dependence Plot, PDP)通过对其他协变量的取值做边缘化,展示响应变量对一组感兴趣协变量的依赖关系。直观上,部分依赖可解释为响应变量的期望值如何随这些协变量变化。",
    ),
    entry(
        "glossary-pareto-k", "Pareto k estimates", "Pareto k 估计",
        r"Pareto $\hat{k}$ 是 LOO 所用 Pareto 平滑重要性采样(PSIS)的诊断量,衡量单个留一观测相对于完整后验分布有多远。如果删去一个观测会使后验改变过大,重要性采样便无法可靠估计。一般而言,$\hat{k}<0.5$ 时对应的 `elpd_loo` 分量精度较高;$0.5<\hat{k}<0.7$ 时精度下降但通常仍可使用;$\hat{k}>0.7$ 时该观测的近似往往不可靠。$\hat{k}$ 也反映观测的影响力;很高的值常提示模型设定错误、离群值或数据处理错误。",
    ),
    entry(
        "glossary-point-estimate", "Point estimate", "点估计",
        "点估计是用单个值概括某个未知量的“最佳估计”,它通常但不一定处于参数空间。点估计可与最高密度区间等给出范围的区间估计对照,也可与后验分布及其边缘分布等分布式估计对照。",
    ),
    entry(
        "glossary-p-loo", "p_loo", "LOO 有效复杂度",
        "`p_loo` 是非交叉验证的对数后验预测密度与 `elpd_loo` 之间的差,描述预测未来数据比预测已观测数据困难多少。在一定正则条件下的渐近意义上,它可解释为有效参数个数。表现良好时,`p_loo` 应小于模型参数数目和观测数;否则可能说明模型预测能力很弱或存在严重设定错误,通常还应结合较高的 Pareto k 值检查。",
    ),
    entry(
        "glossary-ppl", "Probabilistic Programming Language", "概率编程语言",
        "概率编程语言(PPL)是一套由建模基元组成的编程语法,允许使用者定义贝叶斯模型并自动执行推断。典型 PPL 还提供先验预测和后验预测采样,以及分析推断结果的功能。",
    ),
    entry(
        "glossary-prior-predictive", "Prior predictive distribution", "先验预测分布",
        "先验预测分布是模型(先验与似然)在看到数据之前对数据分布的预期,也就是模型预期会观察到什么。参见公式 `eq:prior_pred_dist`。它可用于先验构建,因为人们通常比起抽象模型参数,更容易在可观测数据的尺度上思考合理性。",
    ),
    entry(
        "glossary-posterior-predictive", "Posterior predictive distribution", "后验预测分布",
        "后验预测分布是给定后验之后未来数据的分布;后验又由模型(先验和似然)与已观测数据共同决定。换言之,它给出模型的预测。参见公式 `eq:post_pred_dist`。除了生成预测,还可通过将其与观测数据比较来评估模型拟合。",
    ),
    entry(
        "glossary-residuals", "Residuals", "残差",
        "残差是观测值与目标量估计值之差。如果模型假设所有残差的方差有限且相同,则具有同方差性;如果方差可以变化,则具有异方差性。",
    ),
    entry(
        "glossary-sufficient-statistics", "Sufficient statistics", "充分统计量",
        r"如果同一样本计算出的其他任何统计量都不能为某个模型参数提供额外信息,则称该统计量关于此参数是充分的。也就是说,它足以在不损失相关信息的情况下概括样本。例如,对来自均值为 $\mu$、已知有限方差的正态分布的一组独立样本,样本均值是关于 $\mu$ 的充分统计量;但均值不描述离散程度,所以只对 $\mu$ 充分。对 iid 数据,只有指数族分布具有维数与参数 $\theta$ 维数相同的充分统计量;其他分布的充分统计量维数通常随样本量增加。",
    ),
    entry(
        "glossary-synthetic-data", "Synthetic data", "合成数据",
        "合成数据也称模拟数据,指由模型生成而不是通过实验或观测收集的数据。先验预测分布和后验预测分布的样本都是合成数据。",
    ),
    entry(
        "glossary-timestamp", "Timestamp", "时间戳",
        "时间戳是一种用于标识某个事件发生时间的编码信息,通常包含日期与当日时间,必要时还包含更精细的秒的小数部分。",
    ),
    entry(
        "glossary-turing-complete", "Turing-complete", "图灵完备",
        "在日常技术语境中,图灵完备通常表示任意现实世界的通用计算机或通用编程语言,都能近似模拟其他任意通用计算机或语言的计算过程。",
    ),
]

OUTPUT_NOTEBOOK = "Glossary_zh.ipynb"
OUTPUT_ORG = "Glossary_zh.org"
