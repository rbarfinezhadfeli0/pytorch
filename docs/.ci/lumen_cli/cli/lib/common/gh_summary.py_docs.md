# Documentation: `.ci/lumen_cli/cli/lib/common/gh_summary.py`

## File Metadata

- **Path**: `.ci/lumen_cli/cli/lib/common/gh_summary.py`
- **Size**: 3,671 bytes (3.58 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from __future__ import annotations

import logging
import os
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from cli.lib.common.utils import get_wheels
from jinja2 import Template


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


logger = logging.getLogger(__name__)

_TPL_CONTENT = Template(
    textwrap.dedent("""\
    ## {{ title }}

    ```{{ lang }}
    {{ content }}
    ```
""")
)

_TPL_LIST_ITEMS = Template(
    textwrap.dedent("""\
    ## {{ title }}
    {% for it in items %}
    - {{ it.pkg }}: {{ it.relpath }}
    {% else %}
    _(no item found)_
    {% endfor %}
    """)
)

_TPL_TABLE = Template(
    textwrap.dedent("""\
    {%- if rows %}
    | {{ cols | join(' | ') }} |
    |{%- for _ in cols %} --- |{%- endfor %}
    {%- for r in rows %}
    | {%- for c in cols %} {{ r.get(c, "") }} |{%- endfor %}
    {%- endfor %}
    {%- else %}
    _(no data)_
    {%- endif %}
""")
)


def gh_summary_path() -> Path | None:
    """Return the Path to the GitHub step summary file, or None if not set."""
    p = os.environ.get("GITHUB_STEP_SUMMARY")
    return Path(p) if p else None


def write_gh_step_summary(md: str, *, append_content: bool = True) -> bool:
    """
    Write Markdown content to the GitHub Step Summary file if GITHUB_STEP_SUMMARY is set.
    append_content: default true, if True, append to the end of the file, else overwrite the whole file

    Returns:
        True if written successfully (in GitHub Actions environment),
        False if skipped (e.g., running locally where the variable is not set).
    """
    sp = gh_summary_path()
    if not sp:
        logger.info("[gh-summary] GITHUB_STEP_SUMMARY not set, skipping write.")
        return False

    md_clean = textwrap.dedent(md).strip() + "\n"

    mode = "a" if append_content else "w"
    with sp.open(mode, encoding="utf-8") as f:
        f.write(md_clean)
    return True


def md_heading(text: str, level: int = 2) -> str:
    """Generate a Markdown heading string with the given level (1-6)."""
    return f"{'#' * max(1, min(level, 6))} {text}\n"


def md_details(summary: str, content: str) -> str:
    """Generate a collapsible <details> block with a summary and inner content."""
    return f"<details>\n<summary>{summary}</summary>\n\n{content}\n\n</details>\n"


def summarize_content_from_file(
    output_dir: Path,
    freeze_file: str,
    title: str = "Content from file",
    code_lang: str = "",  # e.g. "text" or "ini"
) -> bool:
    f = Path(output_dir) / freeze_file
    if not f.exists():
        return False
    content = f.read_text(encoding="utf-8").strip()
    md = render_content(content, title=title, lang=code_lang)
    return write_gh_step_summary(md)


def summarize_wheels(path: Path, title: str = "Wheels", max_depth: int = 3):
    items = get_wheels(path, max_depth=max_depth)
    if not items:
        return False
    md = render_list(items, title=title)
    return write_gh_step_summary(md)


def md_kv_table(rows: Iterable[Mapping[str, str | int | float]]) -> str:
    """
    Render a list of dicts as a Markdown table using Jinja template.
    """
    rows = list(rows)
    cols = list({k for r in rows for k in r.keys()})
    md = _TPL_TABLE.render(cols=cols, rows=rows).strip() + "\n"
    return md


def render_list(
    items: Iterable[str],
    *,
    title: str = "List",
) -> str:
    tpl = _TPL_LIST_ITEMS
    md = tpl.render(title=title, items=items)
    return md


def render_content(
    content: str,
    *,
    title: str = "Content",
    lang: str = "text",
) -> str:
    tpl = _TPL_CONTENT
    md = tpl.render(title=title, content=content, lang=lang)
    return md

```



## High-Level Overview

textwrap.dedent("""\    ## {{ title }}    ```{{ lang }}    {{ content }}    ```

This Python file contains 0 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `gh_summary_path`, `write_gh_step_summary`, `md_heading`, `md_details`, `summarize_content_from_file`, `summarize_wheels`, `md_kv_table`, `render_list`, `render_content`

**Key imports**: annotations, logging, os, textwrap, Path, TYPE_CHECKING, get_wheels, Template, Iterable, Mapping


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.ci/lumen_cli/cli/lib/common`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `logging`
- `os`
- `textwrap`
- `pathlib`: Path
- `typing`: TYPE_CHECKING
- `cli.lib.common.utils`: get_wheels
- `jinja2`: Template
- `collections.abc`: Iterable, Mapping


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`.ci/lumen_cli/cli/lib/common`):

- [`docker_helper.py_docs.md`](./docker_helper.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`git_helper.py_docs.md`](./git_helper.py_docs.md)
- [`envs_helper.py_docs.md`](./envs_helper.py_docs.md)
- [`pip_helper.py_docs.md`](./pip_helper.py_docs.md)
- [`cli_helper.py_docs.md`](./cli_helper.py_docs.md)
- [`logger.py_docs.md`](./logger.py_docs.md)
- [`path_helper.py_docs.md`](./path_helper.py_docs.md)


## Cross-References

- **File Documentation**: `gh_summary.py_docs.md`
- **Keyword Index**: `gh_summary.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
