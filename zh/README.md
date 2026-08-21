# 中文版构建说明

本目录保存《Bayesian Modeling and Computation in Python》（Martin、Kumar、Lao
著，Chapman & Hall/CRC，2021）的中文翻译与现代化代码。构建系统以可审查、
可复现和“不覆盖源文件”为原则。

## 哪些文件可以编辑

每章的 `build.py` 中的 `cells` 列表是该章 Notebook 与 Org 的可编辑单一来源。
单元格至少包含 `type` 和 `source`；需要长期引用输出时可显式增加稳定的 `id`
和 `metadata`。章目录中的 `.ipynb`、`.org` 及 `generated/` 输出均为生成物，
不要把它们当作独立源文件手工修补。

共享配置位于：

- `book.toml`：全书顺序、可用/规划单元、执行配置；
- `manifests/*.toml`：单元级输出、内核和工作目录；
- `tools/nb_tools.py`：Notebook、输出提取和 Org 转换；
- `tools/build_book.py`：分阶段验证、构建、检查和发布。

规划中的单元可以先登记为 `status = "planned"`，此时静态验证不要求尚不存在
的目录或 builder。

## 常用命令

在仓库根目录、Python 3.12 环境中运行：

```sh
# 只检查 TOML、builder 可加载性和 cell schema；不运行 Notebook
python zh/tools/build_book.py validate

# 在 zh/.build/ 下生成并检查，不写回章目录
python zh/tools/build_book.py build --profile static --unit chapter-3

# 运行一个 smoke 单元；仍只写 staging
python zh/tools/build_book.py check --profile smoke --unit chapter-1

# 完整执行与检查，但不发布
python zh/tools/build_book.py release --unit chapter-1 --keep-staging

# 只有显式 --promote 才原子替换已检查的生成文件
python zh/tools/build_book.py release --unit chapter-1 --promote
```

`validate` 和 `static` 不执行代码。`smoke` 用较短超时验证单章执行路径；
`release` 使用较长超时并要求所有非空代码单元具有有效执行计数、且默认不允许
错误输出。可用 `--execute` / `--no-execute` 明确覆盖 profile。请勿把
`--promote` 用于仅验证 CI；工作流也不会自动提交生成文件。

生产转换优先使用绝对固定路径 `/usr/bin/pandoc`、`/usr/local/bin/pandoc`，
或 `ZH_PANDOC` 指定的绝对可执行路径。Pandoc 不存在时会使用确定性的内部
转换器；遇到不能安全表达的 Markdown 会报错，而不是静默改坏内容。

## 工作目录与数据策略

Notebook 由 `nbclient` 在对应章目录中执行，因此代码中的 `data/...`、
`img/...` 等相对路径必须以章目录为基准。数据和静态资源应提交在明确的章级
路径中；不要依赖调用者当前目录、用户主目录、Notebook 启动目录或未声明的
网络下载。构建器本身应只声明 `cells`，所有有副作用的生成逻辑必须保留在
`if __name__ == "__main__":` 保护之下。

禁止将 `solution`、`solutions`、`answer`、`answers` 目录作为构建输入，
也禁止路径越过 `zh/` 配置根目录。

## 中文版说明标签

需要与原文清楚区分的内容采用以下固定标签：

- **中文版补充**：为中文读者增加的背景、解释或本地化信息；
- **中文版现代化说明**：因当前 PyMC、ArviZ、NumPy 等 API 或行为变化而作的
  代码与结果说明。

标签应出现在 Markdown callout 或正文开头，不应伪装成原书内容。

## 环境分组

依赖按执行栈保存在 `environments/`：

- `core`：Notebook 构建、验证与执行工具，全书各单元共用；
- `pymc`：科学计算/PyMC 基础栈，第 1、2、3–9、10、11 章及大部分前后
  辅文都依赖它；
- `tfp`：已在本机验证的 TensorFlow Probability 栈（第 3、4、6、10、11 章）；
- `bart`：第 7 章专用的 PyMC-BART 扩展。`bart.in` 中的版本是研究后固定的
  组合，但本机两个可用 Python 环境都未能安装 `pymc-bart`，因此第 7 章至今
  只完成了静态校验，从未在本机真正执行过；
- `ml`：第 8 章随机森林 ABC 模型选择所需的 scikit-learn，已在本机验证
  （`scikit-learn==1.9.0`，37/37 代码单元零错误执行）；
- `ppl`：第 10 章 JAX/NumPyro 扩展，与已验证的 TensorFlow Probability 0.25
  搭配使用；
- `optional-jax`、`optional-modeling`：尚未在本机安装验证，只有未解析输入，
  在相应章节发布前必须在目标平台解析、固定版本并测试。

默认 `requirements.txt` 只组合 `core` 与 `pymc` 直接依赖，用于最小化安装第
1–2 章。精确版本来自 2026-08-20 的 Python 3.12.3 项目环境，不声称是完整的
传递依赖冻结。安装后必须运行 `python -m pip check`。系统包及
Pandoc/Graphviz 说明见 `environments/system-dependencies.md`。

## 许可

许可按仓库 `welcome.md` 的约定拆分：

- 书籍正文与翻译内容：Creative Commons Attribution-NonCommercial-ShareAlike
  4.0 International（CC BY-NC-SA 4.0）；
- 代码、代码块与 Notebook：仓库 `LICENSE` 中的 GNU GPL v2.0。

贡献时请根据内容类型保留这一许可边界。
