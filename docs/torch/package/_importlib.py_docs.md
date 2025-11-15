# Documentation: `torch/package/_importlib.py`

## File Metadata

- **Path**: `torch/package/_importlib.py`
- **Size**: 2,998 bytes (2.93 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import _warnings
import os.path


# note: implementations
# copied from cpython's import code


# _zip_searchorder defines how we search for a module in the Zip
# archive: we first search for a package __init__, then for
# non-package .pyc, and .py entries. The .pyc entries
# are swapped by initzipimport() if we run in optimized mode. Also,
# '/' is replaced by path_sep there.

_zip_searchorder = (
    ("/__init__.py", True),
    (".py", False),
)


# Replace any occurrences of '\r\n?' in the input string with '\n'.
# This converts DOS and Mac line endings to Unix line endings.
def _normalize_line_endings(source):
    source = source.replace(b"\r\n", b"\n")
    source = source.replace(b"\r", b"\n")
    return source


def _resolve_name(name, package, level):
    """Resolve a relative module name to an absolute one."""
    bits = package.rsplit(".", level - 1)
    if len(bits) < level:
        raise ValueError("attempted relative import beyond top-level package")
    base = bits[0]
    return f"{base}.{name}" if name else base


def _sanity_check(name, package, level):
    """Verify arguments are "sane"."""
    if not isinstance(name, str):
        raise TypeError(f"module name must be str, not {type(name)}")
    if level < 0:
        raise ValueError("level must be >= 0")
    if level > 0:
        if not isinstance(package, str):
            raise TypeError("__package__ not set to a string")
        elif not package:
            raise ImportError("attempted relative import with no known parent package")
    if not name and level == 0:
        raise ValueError("Empty module name")


def _calc___package__(globals):
    """Calculate what __package__ should be.

    __package__ is not guaranteed to be defined or could be set to None
    to represent that its proper value is unknown.

    """
    package = globals.get("__package__")
    spec = globals.get("__spec__")
    if package is not None:
        if spec is not None and package != spec.parent:
            _warnings.warn(  # noqa: G010
                f"__package__ != __spec__.parent ({package!r} != {spec.parent!r})",  # noqa: G004
                ImportWarning,
                stacklevel=3,
            )
        return package
    elif spec is not None:
        return spec.parent
    else:
        _warnings.warn(  # noqa: G010
            "can't resolve package from __spec__ or __package__, "
            "falling back on __name__ and __path__",
            ImportWarning,
            stacklevel=3,
        )
        package = globals["__name__"]
        if "__path__" not in globals:
            package = package.rpartition(".")[0]
    return package


def _normalize_path(path):
    """Normalize a path by ensuring it is a string.

    If the resulting string contains path separators, an exception is raised.
    """
    parent, file_name = os.path.split(path)
    if parent:
        raise ValueError(f"{path!r} must be only a file name")
    else:
        return file_name

```



## High-Level Overview

"""Resolve a relative module name to an absolute one."""    bits = package.rsplit(".", level - 1)    if len(bits) < level:        raise ValueError("attempted relative import beyond top-level package")    base = bits[0]    return f"{base}.{name}" if name else basedef _sanity_check(name, package, level):

This Python file contains 0 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_normalize_line_endings`, `_resolve_name`, `_sanity_check`, `_calc___package__`, `_normalize_path`

**Key imports**: _warnings, os.path, code, beyond top, with no known parent package


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/package`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `_warnings`
- `os.path`
- `code`
- `beyond top`
- `with no known parent package`


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

Files in the same folder (`torch/package`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`package_exporter.py_docs.md`](./package_exporter.py_docs.md)
- [`_package_pickler.py_docs.md`](./_package_pickler.py_docs.md)
- [`glob_group.py_docs.md`](./glob_group.py_docs.md)
- [`file_structure_representation.py_docs.md`](./file_structure_representation.py_docs.md)
- [`_directory_reader.py_docs.md`](./_directory_reader.py_docs.md)
- [`_mangling.py_docs.md`](./_mangling.py_docs.md)
- [`mangling.md_docs.md`](./mangling.md_docs.md)
- [`package_importer.py_docs.md`](./package_importer.py_docs.md)
- [`find_file_dependencies.py_docs.md`](./find_file_dependencies.py_docs.md)


## Cross-References

- **File Documentation**: `_importlib.py_docs.md`
- **Keyword Index**: `_importlib.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
