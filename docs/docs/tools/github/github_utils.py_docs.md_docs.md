# Documentation: `docs/tools/github/github_utils.py_docs.md`

## File Metadata

- **Path**: `docs/tools/github/github_utils.py_docs.md`
- **Size**: 5,843 bytes (5.71 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/github/github_utils.py`

## File Metadata

- **Path**: `tools/github/github_utils.py`
- **Size**: 2,612 bytes (2.55 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
"""GitHub Utilities"""

from __future__ import annotations

import json
import os
from typing import Any, cast, TYPE_CHECKING
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


if TYPE_CHECKING:
    from collections.abc import Callable


def gh_fetch_url_and_headers(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    data: dict[str, Any] | None = None,
    method: str | None = None,
    reader: Callable[[Any], Any] = lambda x: x.read(),
) -> tuple[Any, Any]:
    if headers is None:
        headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token is not None and url.startswith("https://api.github.com/"):
        headers["Authorization"] = f"token {token}"
    data_ = json.dumps(data).encode() if data is not None else None
    try:
        with urlopen(Request(url, headers=headers, data=data_, method=method)) as conn:
            return conn.headers, reader(conn)
    except HTTPError as err:
        if err.code == 403 and all(
            key in err.headers for key in ["X-RateLimit-Limit", "X-RateLimit-Used"]
        ):
            print(
                f"""Rate limit exceeded:
                Used: {err.headers["X-RateLimit-Used"]}
                Limit: {err.headers["X-RateLimit-Limit"]}
                Remaining: {err.headers["X-RateLimit-Remaining"]}
                Resets at: {err.headers["x-RateLimit-Reset"]}"""
            )
        raise


def gh_fetch_url(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    data: dict[str, Any] | None = None,
    method: str | None = None,
    reader: Callable[[Any], Any] = lambda x: x.read(),
) -> Any:
    return gh_fetch_url_and_headers(
        url, headers=headers, data=data, reader=json.load, method=method
    )[1]


def _gh_fetch_json_any(
    url: str,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> Any:
    headers = {"Accept": "application/vnd.github.v3+json"}
    if params is not None and len(params) > 0:
        url += "?" + "&".join(
            f"{name}={quote(str(val))}" for name, val in params.items()
        )
    return gh_fetch_url(url, headers=headers, data=data, reader=json.load)


def gh_fetch_json_dict(
    url: str,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return cast(dict[str, Any], _gh_fetch_json_any(url, params, data))


def gh_fetch_commit(org: str, repo: str, sha: str) -> dict[str, Any]:
    return gh_fetch_json_dict(
        f"https://api.github.com/repos/{org}/{repo}/commits/{sha}"
    )

```



## High-Level Overview

"""GitHub Utilities"""from __future__ import annotationsimport jsonimport osfrom typing import Any, cast, TYPE_CHECKINGfrom urllib.error import HTTPErrorfrom urllib.parse import quotefrom urllib.request import Request, urlopenif TYPE_CHECKING:    from collections.abc import Callabledef gh_fetch_url_and_headers(    url: str,    *,    headers: dict[str, str] | None = None,    data: dict[str, Any] | None = None,    method: str | None = None,    reader: Callable[[Any], Any] = lambda x: x.read(),) -> tuple[Any, Any]:    if headers is None:        headers = {}    token = os.environ.get("GITHUB_TOKEN")    if token is not None and url.startswith("https://api.github.com/"):        headers["Authorization"] = f"token {token}"    data_ = json.dumps(data).encode() if data is not None else None    try:        with urlopen(Request(url, headers=headers, data=data_, method=method)) as conn:            return conn.headers, reader(conn)    except HTTPError as err:        if err.code == 403 and all(            key in err.headers for key in ["X-RateLimit-Limit", "X-RateLimit-Used"]        ):            print(

This Python file contains 0 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `gh_fetch_url_and_headers`, `gh_fetch_url`, `_gh_fetch_json_any`, `gh_fetch_json_dict`, `gh_fetch_commit`

**Key imports**: annotations, json, os, Any, cast, TYPE_CHECKING, HTTPError, quote, Request, urlopen, Callable


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/github`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `json`
- `os`
- `typing`: Any, cast, TYPE_CHECKING
- `urllib.error`: HTTPError
- `urllib.parse`: quote
- `urllib.request`: Request, urlopen
- `collections.abc`: Callable


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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

Files in the same folder (`tools/github`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)


## Cross-References

- **File Documentation**: `github_utils.py_docs.md`
- **Keyword Index**: `github_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/tools/github`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/tools/github`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs/tools/github`):

- [`github_utils.py_kw.md_docs.md`](./github_utils.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `github_utils.py_docs.md_docs.md`
- **Keyword Index**: `github_utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
