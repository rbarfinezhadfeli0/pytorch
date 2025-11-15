# Documentation: `torch/_dynamo/funcname_cache.py`

## File Metadata

- **Path**: `torch/_dynamo/funcname_cache.py`
- **Size**: 2,548 bytes (2.49 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""
This module provides functionality for caching and looking up fully qualified function
and class names from Python source files by line number.

It uses Python's tokenize module to parse source files and tracks function/class
definitions along with their nesting to build fully qualified names (e.g. 'class.method'
or 'module.function'). The results are cached in a two-level dictionary mapping:

    filename -> (line_number -> fully_qualified_name)

Example usage:
    name = get_funcname("myfile.py", 42)  # Returns name of function/class at line 42
    clearcache()  # Clear the cache if file contents have changed

The parsing is done lazily when a file is first accessed. Invalid Python files or
IO errors are handled gracefully by returning empty cache entries.
"""

import tokenize
from typing import Optional


cache: dict[str, dict[int, str]] = {}


def clearcache() -> None:
    cache.clear()


def _add_file(filename: str) -> None:
    try:
        with tokenize.open(filename) as f:
            tokens = list(tokenize.generate_tokens(f.readline))
    except (OSError, tokenize.TokenError):
        cache[filename] = {}
        return

    # NOTE: undefined behavior if file is not valid Python source,
    # since tokenize will have undefined behavior.
    result: dict[int, str] = {}
    # current full funcname, e.g. xxx.yyy.zzz
    cur_name = ""
    cur_indent = 0
    significant_indents: list[int] = []

    for i, token in enumerate(tokens):
        if token.type == tokenize.INDENT:
            cur_indent += 1
        elif token.type == tokenize.DEDENT:
            cur_indent -= 1
            # possible end of function or class
            if significant_indents and cur_indent == significant_indents[-1]:
                significant_indents.pop()
                # pop the last name
                cur_name = cur_name.rpartition(".")[0]
        elif (
            token.type == tokenize.NAME
            and i + 1 < len(tokens)
            and tokens[i + 1].type == tokenize.NAME
            and (token.string == "class" or token.string == "def")
        ):
            # name of class/function always follows class/def token
            significant_indents.append(cur_indent)
            if cur_name:
                cur_name += "."
            cur_name += tokens[i + 1].string
        result[token.start[0]] = cur_name

    cache[filename] = result


def get_funcname(filename: str, lineno: int) -> Optional[str]:
    if filename not in cache:
        _add_file(filename)
    return cache[filename].get(lineno, None)

```



## High-Level Overview

"""This module provides functionality for caching and looking up fully qualified functionand class names from Python source files by line number.It uses Python's tokenize module to parse source files and tracks function/classdefinitions along with their nesting to build fully qualified names (e.g. 'class.method'or 'module.function'). The results are cached in a two-level dictionary mapping:    filename -> (line_number -> fully_qualified_name)Example usage:    name = get_funcname("myfile.py", 42)  # Returns name of function/class at line 42    clearcache()  # Clear the cache if file contents have changedThe parsing is done lazily when a file is first accessed. Invalid Python files orIO errors are handled gracefully by returning empty cache entries.

This Python file contains 4 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `clearcache`, `_add_file`, `get_funcname`

**Key imports**: tokenize, Optional


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dynamo`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `tokenize`
- `typing`: Optional


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

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

Files in the same folder (`torch/_dynamo`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`side_effects.py_docs.md`](./side_effects.py_docs.md)
- [`package.py_docs.md`](./package.py_docs.md)
- [`types.py_docs.md`](./types.py_docs.md)
- [`hooks.py_docs.md`](./hooks.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`graph_break_hints.py_docs.md`](./graph_break_hints.py_docs.md)
- [`device_interface.py_docs.md`](./device_interface.py_docs.md)
- [`graph_break_registry.json_docs.md`](./graph_break_registry.json_docs.md)
- [`current_scope_id.py_docs.md`](./current_scope_id.py_docs.md)


## Cross-References

- **File Documentation**: `funcname_cache.py_docs.md`
- **Keyword Index**: `funcname_cache.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
