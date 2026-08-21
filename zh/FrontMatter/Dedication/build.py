"""Cells-list source for the Simplified Chinese dedication.

Importing this module performs no file I/O and generates no outputs.
"""

from zh.FrontMatter._shared import attribution_cell, cell


SOURCE_PATH = "markdown/dedication.md"

cells = [
    attribution_cell(SOURCE_PATH),
    cell(
        "dedication-translation",
        "markdown",
        r"""
(dedication)=

# 题献

```{epigraph}
献给 Romina 和 Abril,也献给她们充满关怀的爱。献给每一位帮助我
走到这里的人。


-- Osvaldo Martin
```

```{epigraph}
献给那些教育者,无论他们来自正规教育还是非正规教育;他们无条件地分享
知识与智慧,塑造了今天的我。按照他们在我生命中出现的先后顺序,他们是
Tim Pegg,Mr. Michael Collins,Mrs. Sara LaFramboise Saadeh,
Professor Mehrdad Haghi,Professor Winny Dong,Professor Dixon Davis,
Jason Errington,Chris Lopez,John Norman,Professor Ananth Krishnamurthy,
以及 Kurt Campbell。

谢谢你们


-- Ravin Kumar
```

```{epigraph}
献给 Yuli。

-- Junpeng Lao
```
""",
        kind="translated-source",
        source_path=SOURCE_PATH,
        source_range="1-31",
    ),
]
