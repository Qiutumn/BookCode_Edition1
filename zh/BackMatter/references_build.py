r"""中文版参考文献的规范源,从根目录 ``bibtex.bib`` 确定性生成。"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

SOURCE = "markdown/references.md"
BIBTEX_SOURCE = "bibtex.bib"
ROOT = Path(__file__).resolve().parents[2]


def _split_entries(text: str) -> list[str]:
    starts = [match.start() for match in re.finditer(r"(?m)^@", text)]
    return [text[start:end].strip() for start, end in zip(starts, starts[1:] + [len(text)])]


def _split_top_level(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    brace_depth = 0
    quote = False
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"' and brace_depth == 0:
            quote = not quote
        elif not quote:
            if char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
            elif char == "," and brace_depth == 0:
                parts.append(text[start:index].strip())
                start = index + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _unwrap(value: str) -> str:
    value = value.strip().rstrip(",").strip()
    while len(value) >= 2 and ((value[0] == "{" and value[-1] == "}") or (value[0] == '"' and value[-1] == '"')):
        value = value[1:-1].strip()
    return value


def _clean_latex(value: str) -> str:
    value = _unwrap(value)
    value = value.replace("\\&", "&").replace("--", "–")
    value = re.sub(r"\\url\{([^}]+)\}", r"\1", value)
    # 保留公式和带重音的人名所需的 LaTeX,只去掉用于保护大小写的普通外层花括号。
    value = re.sub(r"\{([A-Za-z0-9][^{}]*)\}", r"\1", value)
    return " ".join(value.split())


def parse_entry(raw: str) -> dict[str, str]:
    header = re.match(r"@(?P<type>[^\s{]+)\s*\{\s*(?P<key>[^,]+),", raw, re.S)
    if not header:
        raise ValueError(f"无法解析 BibTeX 条目头: {raw[:80]!r}")
    body = raw[header.end():].strip()
    if body.endswith("}"):
        body = body[:-1]
    fields: dict[str, str] = {
        "entry_type": header.group("type").lower(),
        "key": header.group("key").strip(),
    }
    for part in _split_top_level(body):
        if not part or "=" not in part:
            continue
        name, value = part.split("=", 1)
        fields[name.strip().lower()] = _clean_latex(value)
    return fields


def reference_cell_id(key: str) -> str:
    """Return a deterministic nbformat-safe cell ID for a BibTeX key."""
    direct = f"reference-{key}"
    if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", direct):
        return direct
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", key).strip("-_")[:42] or "entry"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
    return f"reference-{slug}-{digest}"


def render_entry(entry: dict[str, str]) -> str:
    authors = entry.get("author", entry.get("editor", "作者信息缺失")).replace(" and ", "; ")
    title = entry.get("title", "题名缺失")
    year = entry.get("year", "年份不详")
    venue = (
        entry.get("journal")
        or entry.get("booktitle")
        or entry.get("publisher")
        or entry.get("howpublished")
        or ""
    )
    details = []
    if entry.get("volume"):
        details.append(f"vol. {entry['volume']}")
    if entry.get("number"):
        details.append(f"no. {entry['number']}")
    if entry.get("pages"):
        details.append(f"pp. {entry['pages']}")
    if entry.get("doi"):
        details.append(f"DOI: {entry['doi']}")
    elif entry.get("url"):
        details.append(entry["url"])
    suffix = ", ".join(item for item in [venue, *details] if item)
    if suffix:
        suffix = f" {suffix}."
    return f"**{entry['key']}** — {authors}. *{title}*. {year}.{suffix}"


_raw_bibliography = (ROOT / BIBTEX_SOURCE).read_text(encoding="utf-8")
entries = [parse_entry(raw) for raw in _split_entries(_raw_bibliography)]
_entry_keys = [entry["key"] for entry in entries]
if len(entries) != 174:
    raise ValueError(f"预期 174 条参考文献,实际解析到 {len(entries)} 条")
if len(_entry_keys) != len(set(_entry_keys)):
    raise ValueError("bibtex.bib 包含重复引用键")

cells = [
    {
        "id": "references-title",
        "type": "markdown",
        "metadata": {
            "kind": "translation",
            "provenance": [f"{SOURCE}#References", BIBTEX_SOURCE],
        },
        "source": r"""# 参考文献

以下条目由仓库根目录的 `bibtex.bib` 确定性生成。论文、书籍和会议录的正式题名保留其发表时的
语言,以便准确检索;作者、出版物、年份、DOI 等元数据同样保持原貌。章节中的引用键会在构建时
与本列表交叉校验。
""",
    }
]

for entry_data in entries:
    cells.append({
        "id": reference_cell_id(entry_data["key"]),
        "type": "markdown",
        "metadata": {
            "kind": "translation",
            "provenance": [f"{BIBTEX_SOURCE}#{entry_data['key']}"],
        },
        "source": render_entry(entry_data),
    })

OUTPUT_NOTEBOOK = "References_zh.ipynb"
OUTPUT_ORG = "References_zh.org"
