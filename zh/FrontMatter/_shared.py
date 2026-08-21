"""Pure cells-list helpers shared by the Chinese front-matter builders.

Importing this module performs no file I/O and generates no outputs.
"""

from __future__ import annotations

from typing import Any


CELL_SCHEMA_VERSION = 1


def cell(
    cell_id: str,
    cell_type: str,
    source: str,
    *,
    kind: str,
    source_path: str,
    source_range: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Return one authored cell with stable metadata and provenance."""

    provenance: dict[str, str] = {"source_path": source_path}
    if source_range is not None:
        provenance["source_range"] = source_range
    if note is not None:
        provenance["note"] = note
    return {
        "id": cell_id,
        "type": cell_type,
        "source": source.strip("\n"),
        "metadata": {
            "kind": kind,
            "provenance": provenance,
        },
    }


def attribution_cell(unit_source_path: str) -> dict[str, Any]:
    """Return the shared Chinese-edition attribution and licensing notice."""

    return cell(
        "frontmatter-attribution",
        "markdown",
        rf"""
### 中文版补充: 中文版说明与授权

原著作者: Osvaldo A. Martin, Ravin Kumar, Junpeng Lao

原书: *Bayesian Modeling and Computation in Python* (Chapman & Hall/CRC, 2021), ISBN 978-0-367-89436-8

原始仓库: [BayesianModelingandComputationInPython/BookCode_Edition1](https://github.com/BayesianModelingandComputationInPython/BookCode_Edition1)

本单元译自 `{unit_source_path}`。本中文版在保留原意、结构、引用、公式与代码的基础上完成简体中文翻译,并在确有需要时将代码更新为当前公开的 PyMC、PyTensor、ArviZ 等 API。新增内容统一标为“中文版补充”,行为或 API 变化统一标为“中文版现代化说明”。

授权范围严格遵循原仓库 `welcome.md`:

- 本书内容中除代码以外的部分采用 [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/) (CC BY-NC-SA 4.0) 授权。
- 代码(包括代码块与 Jupyter notebooks)采用 [GNU GENERAL PUBLIC LICENSE Version 2](https://github.com/BayesianModelingandComputationInPython/BookCode_Edition1/blob/main/LICENSE) (GPL-2.0) 授权。

因此,本中文版不声称全部内容均采用 GPL 授权。翻译与现代化改动不改变原内容各自适用的授权范围。
""",
        kind="chinese-edition-addition",
        source_path="welcome.md",
        source_range="68-76",
        note=f"Shared attribution for {unit_source_path}",
    )
