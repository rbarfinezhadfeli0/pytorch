# Documentation: `docs/torch/package/glob_group.py_docs.md`

## File Metadata

- **Path**: `docs/torch/package/glob_group.py_docs.md`
- **Size**: 7,965 bytes (7.78 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/package/glob_group.py`

## File Metadata

- **Path**: `torch/package/glob_group.py`
- **Size**: 3,665 bytes (3.58 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import re
from collections.abc import Iterable
from typing import Union


GlobPattern = Union[str, Iterable[str]]


class GlobGroup:
    """A set of patterns that candidate strings will be matched against.

    A candidate is composed of a list of segments separated by ``separator``, e.g. "foo.bar.baz".

    A pattern contains one or more segments. Segments can be:
        - A literal string (e.g. "foo"), which matches exactly.
        - A string containing a wildcard (e.g. "torch*", or "foo*baz*"). The wildcard matches
          any string, including the empty string.
        - A double wildcard ("**"). This matches against zero or more complete segments.

    Examples:
        ``torch.**``: matches ``torch`` and all its submodules, e.g. ``torch.nn`` and ``torch.nn.functional``.
        ``torch.*``: matches ``torch.nn`` or ``torch.functional``, but not ``torch.nn.functional``.
        ``torch*.**``: matches ``torch``, ``torchvision``, and all their submodules.

    A candidates will match the ``GlobGroup`` if it matches any of the ``include`` patterns and
    none of the ``exclude`` patterns.

    Args:
        include (Union[str, Iterable[str]]): A string or list of strings,
            each representing a pattern to be matched against. A candidate
            will match if it matches *any* include pattern
        exclude (Union[str, Iterable[str]]): A string or list of strings,
            each representing a pattern to be matched against. A candidate
            will be excluded from matching if it matches *any* exclude pattern.
        separator (str): A string that delimits segments in candidates and
            patterns. By default this is "." which corresponds to how modules are
            named in Python. Another common value for this is "/", which is
            the Unix path separator.
    """

    def __init__(
        self, include: GlobPattern, *, exclude: GlobPattern = (), separator: str = "."
    ):
        self._dbg = f"GlobGroup(include={include}, exclude={exclude})"
        self.include = GlobGroup._glob_list(include, separator)
        self.exclude = GlobGroup._glob_list(exclude, separator)
        self.separator = separator

    def __str__(self):
        return self._dbg

    def __repr__(self):
        return self._dbg

    def matches(self, candidate: str) -> bool:
        candidate = self.separator + candidate
        return any(p.fullmatch(candidate) for p in self.include) and all(
            not p.fullmatch(candidate) for p in self.exclude
        )

    @staticmethod
    def _glob_list(elems: GlobPattern, separator: str = "."):
        if isinstance(elems, str):
            return [GlobGroup._glob_to_re(elems, separator)]
        else:
            return [GlobGroup._glob_to_re(e, separator) for e in elems]

    @staticmethod
    def _glob_to_re(pattern: str, separator: str = "."):
        # to avoid corner cases for the first component, we prefix the candidate string
        # with '.' so `import torch` will regex against `.torch`, assuming '.' is the separator
        def component_to_re(component):
            if "**" in component:
                if component == "**":
                    return "(" + re.escape(separator) + "[^" + separator + "]+)*"
                else:
                    raise ValueError("** can only appear as an entire path segment")
            else:
                return re.escape(separator) + ("[^" + separator + "]*").join(
                    re.escape(x) for x in component.split("*")
                )

        result = "".join(component_to_re(c) for c in pattern.split(separator))
        return re.compile(result)

```



## High-Level Overview

"""A set of patterns that candidate strings will be matched against.    A candidate is composed of a list of segments separated by ``separator``, e.g. "foo.bar.baz".    A pattern contains one or more segments. Segments can be:        - A literal string (e.g. "foo"), which matches exactly.        - A string containing a wildcard (e.g. "torch*", or "foo*baz*"). The wildcard matches          any string, including the empty string.        - A double wildcard ("**"). This matches against zero or more complete segments.    Examples:        ``torch.**``: matches ``torch`` and all its submodules, e.g. ``torch.nn`` and ``torch.nn.functional``.        ``torch.*``: matches ``torch.nn`` or ``torch.functional``, but not ``torch.nn.functional``.        ``torch*.**``: matches ``torch``, ``torchvision``, and all their submodules.    A candidates will match the ``GlobGroup`` if it matches any of the ``include`` patterns and    none of the ``exclude`` patterns.    Args:        include (Union[str, Iterable[str]]): A string or list of strings,            each representing a pattern to be matched against. A candidate            will match if it matches *any* include pattern        exclude (Union[str, Iterable[str]]): A string or list of strings,            each representing a pattern to be matched against. A candidate            will be excluded from matching if it matches *any* exclude pattern.        separator (str): A string that delimits segments in candidates and            patterns. By default this is "." which corresponds to how modules are            named in Python. Another common value for this is "/", which is            the Unix path separator.

This Python file contains 1 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `GlobGroup`

**Functions defined**: `__init__`, `__str__`, `__repr__`, `matches`, `_glob_list`, `_glob_to_re`, `component_to_re`

**Key imports**: re, Iterable, Union, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/package`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `re`
- `collections.abc`: Iterable
- `typing`: Union
- `torch`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

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
- [`file_structure_representation.py_docs.md`](./file_structure_representation.py_docs.md)
- [`_directory_reader.py_docs.md`](./_directory_reader.py_docs.md)
- [`_mangling.py_docs.md`](./_mangling.py_docs.md)
- [`mangling.md_docs.md`](./mangling.md_docs.md)
- [`package_importer.py_docs.md`](./package_importer.py_docs.md)
- [`find_file_dependencies.py_docs.md`](./find_file_dependencies.py_docs.md)


## Cross-References

- **File Documentation**: `glob_group.py_docs.md`
- **Keyword Index**: `glob_group.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/package`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/package`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/package`):

- [`importer.py_docs.md_docs.md`](./importer.py_docs.md_docs.md)
- [`file_structure_representation.py_kw.md_docs.md`](./file_structure_representation.py_kw.md_docs.md)
- [`_directory_reader.py_docs.md_docs.md`](./_directory_reader.py_docs.md_docs.md)
- [`_package_unpickler.py_kw.md_docs.md`](./_package_unpickler.py_kw.md_docs.md)
- [`_digraph.py_kw.md_docs.md`](./_digraph.py_kw.md_docs.md)
- [`_directory_reader.py_kw.md_docs.md`](./_directory_reader.py_kw.md_docs.md)
- [`mangling.md_docs.md_docs.md`](./mangling.md_docs.md_docs.md)
- [`mangling.md_kw.md_docs.md`](./mangling.md_kw.md_docs.md)
- [`package_importer.py_docs.md_docs.md`](./package_importer.py_docs.md_docs.md)
- [`package_importer.py_kw.md_docs.md`](./package_importer.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `glob_group.py_docs.md_docs.md`
- **Keyword Index**: `glob_group.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
