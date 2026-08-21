from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nbformat
from nbformat.v4 import new_output

from zh.tools import nb_tools


class CellSchemaTests(unittest.TestCase):
    def test_legacy_two_field_cells_remain_valid_and_stable(self) -> None:
        legacy = [
            {"type": "markdown", "source": "# 标题"},
            {"type": "code", "source": "print('ok')"},
        ]
        normalized = nb_tools.validate_cells(legacy)
        self.assertEqual([cell["type"] for cell in normalized], ["markdown", "code"])
        self.assertTrue(all(cell["id"] for cell in normalized))

        with_insert = nb_tools.validate_cells(
            [{"type": "markdown", "source": "前言"}, *legacy]
        )
        self.assertEqual(normalized[0]["id"], with_insert[1]["id"])
        self.assertEqual(normalized[1]["id"], with_insert[2]["id"])

    def test_optional_id_and_metadata_are_preserved(self) -> None:
        notebook = nb_tools.cells_to_notebook(
            [
                {
                    "type": "code",
                    "source": "1 + 1",
                    "id": "sum-cell",
                    "metadata": {"tags": ["smoke"], "zh": {"output_name": "sum"}},
                }
            ]
        )
        self.assertEqual(notebook.cells[0].id, "sum-cell")
        self.assertEqual(notebook.cells[0].metadata.tags, ["smoke"])
        nb_tools.validate_notebook(notebook)

        with self.assertRaises(nb_tools.NotebookValidationError):
            nb_tools.validate_notebook(notebook, require_executed=True)
        notebook.cells[0].execution_count = 1
        nb_tools.validate_notebook(notebook, require_executed=True)

    def test_execution_order_and_errors_are_validated(self) -> None:
        notebook = nb_tools.cells_to_notebook(
            [
                {"type": "code", "source": "first()", "id": "first"},
                {"type": "code", "source": "second()", "id": "second"},
            ]
        )
        notebook.cells[0].execution_count = 2
        notebook.cells[1].execution_count = 1
        with self.assertRaises(nb_tools.NotebookValidationError):
            nb_tools.validate_notebook(notebook, require_executed=True)

        notebook.cells[0].execution_count = 1
        notebook.cells[1].execution_count = 2
        notebook.cells[1].outputs = [
            new_output("error", ename="ValueError", evalue="bad", traceback=[])
        ]
        with self.assertRaises(nb_tools.NotebookValidationError):
            nb_tools.validate_notebook(notebook, require_executed=True)
        nb_tools.validate_notebook(notebook, require_executed=True, allow_errors=True)

    def test_schema_rejects_unknown_fields_and_duplicate_ids(self) -> None:
        with self.assertRaises(nb_tools.CellValidationError):
            nb_tools.validate_cells(
                [{"type": "code", "source": "pass", "unexpected": True}]
            )
        with self.assertRaises(nb_tools.CellValidationError):
            nb_tools.validate_cells(
                [
                    {"type": "code", "source": "1", "id": "same"},
                    {"type": "code", "source": "2", "id": "same"},
                ]
            )

    def test_legacy_notebook_without_ids_gets_deterministic_in_memory_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.ipynb"
            payload = {
                "nbformat": 4,
                "nbformat_minor": 4,
                "metadata": {},
                "cells": [
                    {"cell_type": "markdown", "metadata": {}, "source": "# Legacy"},
                    {
                        "cell_type": "code",
                        "metadata": {},
                        "source": "1 + 1",
                        "execution_count": None,
                        "outputs": [],
                    },
                ],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            first = nb_tools.read_notebook(path)
            second = nb_tools.read_notebook(path)
            self.assertEqual(
                [cell.id for cell in first.cells], [cell.id for cell in second.cells]
            )
            self.assertTrue(all(cell.id for cell in first.cells))
            # Migration is inspection-only; the legacy source is not rewritten.
            self.assertNotIn('"id"', path.read_text(encoding="utf-8"))


class OutputExtractionTests(unittest.TestCase):
    @staticmethod
    def _notebook(path: Path, *, shifted: bool, cell_id: str = "plot-cell") -> None:
        cells = []
        if shifted:
            cells.append({"type": "markdown", "source": "Unrelated introduction"})
        cells.append(
            {
                "type": "code",
                "source": "make_plot()",
                "id": cell_id,
                "metadata": {"zh": {"output_name": "posterior"}},
            }
        )
        notebook = nb_tools.cells_to_notebook(cells)
        code_cell = notebook.cells[-1]
        code_cell.execution_count = 1
        code_cell.outputs = [
            new_output("stream", name="stdout", text="made plot\n"),
            new_output(
                "display_data",
                data={
                    "image/png": base64.b64encode(b"png-bytes").decode("ascii"),
                    "image/svg+xml": "<svg xmlns='http://www.w3.org/2000/svg'></svg>",
                    "image/jpeg": base64.b64encode(b"jpeg-bytes").decode("ascii"),
                    "text/plain": "<posterior figure>",
                },
                metadata={},
            ),
        ]
        nbformat.write(notebook, str(path))

    def test_mime_bundle_selects_one_preferred_stable_representation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_notebook = root / "first.ipynb"
            shifted_notebook = root / "shifted.ipynb"
            self._notebook(first_notebook, shifted=False)
            self._notebook(shifted_notebook, shifted=True)

            first = nb_tools.extract_outputs(first_notebook, root / "one/generated", "ch3")
            shifted = nb_tools.extract_outputs(
                shifted_notebook, root / "two/generated", "ch3"
            )
            first_images = [
                item for item in first["plot-cell"] if item["kind"] == "image"
            ]
            shifted_images = [
                item for item in shifted["plot-cell"] if item["kind"] == "image"
            ]
            self.assertEqual(len(first_images), 1)
            self.assertEqual(first_images[0]["mime"], "image/svg+xml")
            self.assertEqual(
                Path(first_images[0]["path"]).name,
                Path(shifted_images[0]["path"]).name,
            )
            self.assertTrue(Path(first_images[0]["path"]).name.endswith("-figure.svg"))
            self.assertTrue((root / "one" / first_images[0]["path"]).is_file())
            self.assertEqual(
                [item["text"] for item in first["plot-cell"] if item.get("stream")],
                ["made plot\n"],
            )
            self.assertFalse(list((root / "one/generated").glob("*.png")))
            self.assertFalse(list((root / "one/generated").glob("*.jpg")))
            self.assertFalse(list((root / "one/generated").glob("*.txt")))

    def test_case_and_truncation_collisions_get_distinct_digests(self) -> None:
        self.assertNotEqual(nb_tools._slug("Plot"), nb_tools._slug("plot"))
        shared = "x" * 80
        self.assertNotEqual(
            nb_tools._slug(shared + "A"), nb_tools._slug(shared + "B")
        )

    def test_existing_empty_directory_is_allowed_but_files_are_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notebook = root / "notebook.ipynb"
            destination = root / "generated"
            destination.mkdir()
            self._notebook(notebook, shifted=False)
            nb_tools.extract_outputs(notebook, destination, "chapter")
            generated = list(destination.iterdir())
            self.assertEqual(len(generated), 1)
            original = generated[0].read_bytes()
            with self.assertRaisesRegex(
                nb_tools.NotebookValidationError, "refusing to overwrite"
            ):
                nb_tools.extract_outputs(notebook, destination, "chapter")
            self.assertEqual(generated[0].read_bytes(), original)

    def test_legacy_extract_images_keeps_index_mapping_but_not_index_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notebook = root / "notebook.ipynb"
            self._notebook(notebook, shifted=True)
            images = nb_tools.extract_images(notebook, root / "generated", "chapter")
            self.assertIn(1, images)
            self.assertTrue(all("plot-cell" in path for path in images[1]))
            self.assertFalse(list((root / "generated").glob("*.txt")))


class OrgConversionTests(unittest.TestCase):
    def test_footnotes_fences_callouts_links_math_and_results(self) -> None:
        cells = [
            {
                "type": "markdown",
                "source": (
                    "## 小节\n\n"
                    "参见 [项目](https://example.test)、<https://xkcd.com/2117/> 与公式 $x^2$。[^why]\n\n"
                    "**粗体**、*斜体* 和 `x=1`。\n\n"
                    "软换行代码 `az.plot_bpv(\nkind=\"u_value\")`。\n\n"
                    "> [!NOTE] 中文版补充\n"
                    "> 保留说明。\n"
                    "> ```python\n> print('fenced')\n> ````\n\n"
                    "[^why]: 原生脚注。"
                ),
                "id": "intro",
            },
            {"type": "code", "source": "print('结果')", "id": "run-cell"},
        ]
        outputs = {
            "run-cell": [
                {
                    "kind": "text",
                    "mime": "text/plain",
                    "stream": "stdout",
                    "text": "结果\n",
                },
                {
                    "kind": "image",
                    "mime": "image/svg+xml",
                    "path": "generated/run-cell-figure.svg",
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chapter.org"
            nb_tools.write_org(
                cells,
                path,
                "测试",
                outputs_by_cell=outputs,
                markdown_converter="internal",
            )
            org = path.read_text(encoding="utf-8")

        self.assertIn("** 小节", org)
        self.assertIn("[[https://example.test][项目]]", org)
        self.assertIn("[[https://xkcd.com/2117/]]", org)
        self.assertIn("$x^2$", org)
        self.assertIn("*粗体*、/斜体/ 和 ~x=1~。", org)
        self.assertIn('软换行代码 ~az.plot_bpv( kind="u_value")~。', org)
        self.assertIn("[fn:why]", org)
        self.assertIn("[fn:why] 原生脚注。", org)
        self.assertIn("#+ATTR_ORG: :callout NOTE", org)
        self.assertIn("*NOTE* — 中文版补充", org)
        self.assertIn("#+BEGIN_SRC python\nprint('fenced')\n#+END_SRC", org)
        self.assertIn("#+RESULTS: stdout", org)
        self.assertIn("#+BEGIN_EXAMPLE\n结果\n#+END_EXAMPLE", org)
        self.assertIn("[[file:generated/run-cell-figure.svg]]", org)

    def test_setext_myst_roles_directives_and_anchors_are_preserved(self) -> None:
        markdown = """Main title
==========

Subsection
----------

(eq:posterior)=

参见 {cite:p}`Gelman2020,Vehtari2017`、{cite:t}`McElreath2020`、{ref}`eq:posterior` 与 {numref}`fig:one`。

:::{admonition} 中文版补充
:class: warning

正文含有 `code`。
:::
"""
        org = nb_tools.md_to_org(markdown, converter="internal")
        self.assertIn("* Main title", org)
        self.assertIn("** Subsection", org)
        self.assertIn("<<eq:posterior>>", org)
        self.assertIn("[cite:@Gelman2020; @Vehtari2017]", org)
        self.assertIn("[cite/t:@McElreath2020]", org)
        self.assertIn("[[eq:posterior]]", org)
        self.assertIn("[[fig:one]]", org)
        self.assertIn("#+ATTR_ORG: :callout WARNING", org)
        self.assertIn("*WARNING* — 中文版补充", org)
        self.assertIn("正文含有 =code=。", org)

    def test_empty_html_anchor_becomes_org_anchor_but_real_html_still_rejected(self) -> None:
        markdown = '<a id="eq:example"></a>\n\n参见 {eq}`eq:example`。'
        org = nb_tools.md_to_org(markdown, converter="internal")
        self.assertIn("<<eq:example>>", org)
        self.assertIn("[[eq:example]]", org)
        with self.assertRaises(nb_tools.UnsupportedMarkdownError):
            nb_tools.md_to_org('<a id="x">text</a>', converter="internal")
        with self.assertRaises(nb_tools.UnsupportedMarkdownError):
            nb_tools.md_to_org("<div>x</div>", converter="internal")

    def test_myst_math_blocks_equation_refs_and_named_source_aliases(self) -> None:
        markdown = """```{math}
:label: eq:normal
p(x) = \\frac{1}{\\sqrt{2\\pi}} e^{-x^2/2}
```

Use {eq}`eq:normal` and inline {math}`x^2`.

:::{math} eq:variance
\\operatorname{Var}(X)=1
:::
"""
        converted = nb_tools.md_to_org(markdown, converter="internal")
        self.assertIn("<<eq:normal>>\n#+NAME: eq:normal\n\\[", converted)
        self.assertIn("p(x) = \\frac{1}{\\sqrt{2\\pi}}", converted)
        self.assertIn("Use [[eq:normal]] and inline $x^2$.", converted)
        self.assertIn("<<eq:variance>>\n#+NAME: eq:variance\n\\[", converted)
        self.assertNotIn("#+BEGIN_SRC math", converted)

        cells = [
            {
                "type": "code",
                "source": "x = 1",
                "id": "stable-cell-id",
                "metadata": {
                    "name": "modern-name",
                    "source_names": ["original_name", "legacy-alias"],
                },
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "named.org"
            nb_tools.write_org(cells, path, "Named", markdown_converter="internal")
            org = path.read_text(encoding="utf-8")
        self.assertIn("<<stable-cell-id>>", org)
        self.assertIn("<<original_name>>", org)
        self.assertIn("<<legacy-alias>>", org)
        self.assertIn("#+NAME: modern-name\n#+BEGIN_SRC python", org)

    def test_nested_inline_placeholders_and_backslash_escapes_resolve(self) -> None:
        markdown = (
            r"[use `x`](https://example.test) and **bold `y`**; "
            r"literal \*stars\*, \_underscores\_, and \[brackets\]."
        )
        org = nb_tools.md_to_org(markdown, converter="internal")
        self.assertIn("[[https://example.test][use =x=]]", org)
        self.assertIn("*bold =y=*", org)
        self.assertIn("literal *stars*, _underscores_, and [brackets].", org)
        self.assertNotIn("\x00", org)

    def test_auto_never_discovers_host_pandoc_and_explicit_mode_requires_pin(self) -> None:
        expected = nb_tools.md_to_org("# 标题", converter="internal")
        with patch.dict(
            os.environ,
            {"ZH_PANDOC": "", "ZH_PANDOC_VERSION": "", "PATH": "/untrusted"},
            clear=False,
        ):
            self.assertEqual(nb_tools.md_to_org("# 标题", converter="auto"), expected)
            with self.assertRaises(FileNotFoundError):
                nb_tools.md_to_org("# 标题", converter="pandoc")

    def test_myst_figure_directive_becomes_captioned_image_not_source_block(self) -> None:
        markdown = (
            "```{figure} figures/example.png\n"
            ":name: fig:example\n"
            ":width: 8.00in\n"
            "示例*图注*。\n"
            "```\n"
            "\n"
            "参见 {numref}`fig:example`。"
        )
        org = nb_tools.md_to_org(markdown, converter="internal")
        self.assertIn("<<fig:example>>", org)
        self.assertIn("#+NAME: fig:example", org)
        self.assertIn("#+CAPTION: 示例/图注/。", org)
        self.assertIn("[[file:figures/example.png]]", org)
        self.assertIn("[[fig:example]]", org)
        self.assertNotIn("#+BEGIN_SRC figure", org)

    def test_myst_code_block_directive_uses_declared_language_not_directive_name(self) -> None:
        markdown = (
            "```{code-block} python\n"
            ":name: example_code\n"
            ":caption: example_code\n"
            "\n"
            "x = 1\n"
            "```"
        )
        org = nb_tools.md_to_org(markdown, converter="internal")
        self.assertIn("#+BEGIN_SRC python", org)
        self.assertIn("x = 1", org)
        self.assertNotIn("#+BEGIN_SRC code-block", org)
        self.assertNotIn(":name:", org)

    def test_myst_list_table_directive_becomes_org_table(self) -> None:
        markdown = (
            "```{list-table} 摘要\n"
            ":name: tab:example\n"
            ":header-rows: 1\n"
            "\n"
            "* - **rank**\n"
            "  - **loo**\n"
            "* - 0\n"
            "  - -377.67\n"
            "```"
        )
        org = nb_tools.md_to_org(markdown, converter="internal")
        self.assertIn("<<tab:example>>", org)
        self.assertIn("#+CAPTION: 摘要", org)
        self.assertIn("| *rank* | *loo* |", org)
        self.assertIn("| 0 | -377.67 |", org)
        self.assertNotIn("#+BEGIN_SRC list-table", org)

    def test_myst_epigraph_directive_becomes_quote_with_attribution(self) -> None:
        markdown = "```{epigraph}\n献给读者。\n\n--- 作者\n```"
        org = nb_tools.md_to_org(markdown, converter="internal")
        self.assertIn("#+BEGIN_QUOTE", org)
        self.assertIn("献给读者。", org)
        self.assertIn("--- /作者/", org)
        self.assertIn("#+END_QUOTE", org)
        self.assertNotIn("#+BEGIN_SRC epigraph", org)

    def test_gfm_pipe_tables_convert_to_org_tables(self) -> None:
        markdown = (
            "| 列 A | 列 B |\n"
            "| --- | :---: |\n"
            "| 1 | *斜体* |\n"
            "| 22 | 33 |\n"
            "\n"
            "后文段落，包含一个竖线 | 不是表格。"
        )
        org = nb_tools.md_to_org(markdown, converter="internal")
        self.assertIn("| 列 A | 列 B |", org)
        self.assertIn("|---+---|", org)
        self.assertIn("| 1 | /斜体/ |", org)
        self.assertIn("| 22 | 33 |", org)
        self.assertIn("后文段落，包含一个竖线 | 不是表格。", org)
        self.assertNotIn("\x00", org)

    def test_same_line_math_comparisons_are_not_mistaken_for_html(self) -> None:
        # A `<` opening one inline-math span and a `>` closing a later,
        # unrelated span on the same line must not be read as a single HTML
        # tag spanning both spans.
        markdown = r"满足 $X_1<c_1$ 且 $X_2>c_2$ 的情形。"
        org = nb_tools.md_to_org(markdown, converter="internal")
        self.assertIn(r"$X_1<c_1$", org)
        self.assertIn(r"$X_2>c_2$", org)
        with self.assertRaises(nb_tools.UnsupportedMarkdownError):
            nb_tools.md_to_org("a <div>b</div> c", converter="internal")

    def test_unsupported_internal_syntax_fails_loudly(self) -> None:
        with self.assertRaises(nb_tools.UnsupportedMarkdownError):
            nb_tools.md_to_org("```python\nprint(1)", converter="internal")
        with self.assertRaises(nb_tools.UnsupportedMarkdownError):
            nb_tools.md_to_org("`x=1~2`", converter="internal")
        with self.assertRaises(nb_tools.UnsupportedMarkdownError):
            nb_tools.md_to_org(":::{figure} x.png\n:::", converter="internal")


if __name__ == "__main__":
    unittest.main()
