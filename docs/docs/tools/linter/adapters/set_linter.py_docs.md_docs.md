# Documentation: `docs/tools/linter/adapters/set_linter.py_docs.md`

## File Metadata

- **Path**: `docs/tools/linter/adapters/set_linter.py_docs.md`
- **Size**: 5,693 bytes (5.56 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/linter/adapters/set_linter.py`

## File Metadata

- **Path**: `tools/linter/adapters/set_linter.py`
- **Size**: 2,802 bytes (2.74 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING


_PARENT = Path(__file__).parent.absolute()
_PATH = [Path(p).absolute() for p in sys.path]

if TYPE_CHECKING or _PARENT not in _PATH:
    from . import _linter
else:
    import _linter

if TYPE_CHECKING:
    from collections.abc import Iterator


ERROR = "Builtin `set` is deprecated"
IMPORT_LINE = "from torch.utils._ordered_set import OrderedSet\n\n"

DESCRIPTION = """`set_linter` is a lintrunner linter which finds usages of the
Python built-in class `set` in Python code, and optionally replaces them with
`OrderedSet`.
"""

EPILOG = """
`lintrunner` operates on whole commits. If you want to remove uses of `set`
from existing files not part of a commit, call `set_linter` directly:

    python tools/linter/adapters/set_linter.py --fix [... python files ...]

---

To omit a line of Python code from `set_linter` checking, append a comment:

    s = set()  # noqa: set_linter
    t = {  # noqa: set_linter
       "one",
       "two",
    }

---

Running set_linter in fix mode (though either `lintrunner -a` or `--fix`
should not significantly change the behavior of working code, but will still
usually needs some manual intervention:

1. Replacing `set` with `OrderedSet` will sometimes introduce new typechecking
errors because `OrderedSet` is imperfectly generic. Find a common type for its
elements (in the worst case, `typing.Any` always works), and use
`OrderedSet[YourCommonTypeHere]`.

2. The fix mode doesn't recognize generator expressions, so it replaces:

    s = {i for i in range(3)}

with

    s = OrderedSet([i for i in range(3)])

You can and should delete the square brackets in every such case.

3. There is a common pattern of set usage where a set is created and then only
used for testing inclusion. For small collections, up to around 12 elements, a
tuple is more time-efficient than an OrderedSet and also has less visual clutter
(see https://github.com/rec/test/blob/master/python/time_access.py).
"""


class SetLinter(_linter.FileLinter):
    linter_name = "set_linter"
    description = DESCRIPTION
    epilog = EPILOG
    report_column_numbers = True

    def _lint(self, pf: _linter.PythonFile) -> Iterator[_linter.LintResult]:
        if (pf.sets or pf.braced_sets) and (ins := pf.insert_import_line) is not None:
            yield _linter.LintResult(
                "Add import for OrderedSet", ins, 0, IMPORT_LINE, 0
            )
        for b in pf.braced_sets:
            yield _linter.LintResult(ERROR, *b[0].start, "OrderedSet([", 1)
            yield _linter.LintResult(ERROR, *b[-1].start, "])", 1)

        for s in pf.sets:
            yield _linter.LintResult(ERROR, *s.start, "OrderedSet", 3)


if __name__ == "__main__":
    SetLinter.run()

```



## High-Level Overview

DESCRIPTION = """`set_linter` is a lintrunner linter which finds usages of thePython built-in class `set` in Python code, and optionally replaces them with`OrderedSet`.

This Python file contains 1 class(es) and 1 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `SetLinter`

**Functions defined**: `_lint`

**Key imports**: annotations, sys, Path, TYPE_CHECKING, _linter, _linter, Iterator, OrderedSet, for OrderedSet


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/linter/adapters`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `sys`
- `pathlib`: Path
- `typing`: TYPE_CHECKING
- `.`: _linter
- `_linter`
- `collections.abc`: Iterator
- `torch.utils._ordered_set`: OrderedSet
- `for OrderedSet`


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

Files in the same folder (`tools/linter/adapters`):

- [`grep_linter.py_docs.md`](./grep_linter.py_docs.md)
- [`import_linter.py_docs.md`](./import_linter.py_docs.md)
- [`gha_linter.py_docs.md`](./gha_linter.py_docs.md)
- [`actionlint_linter.py_docs.md`](./actionlint_linter.py_docs.md)
- [`pyfmt_linter.py_docs.md`](./pyfmt_linter.py_docs.md)
- [`mypy_linter.py_docs.md`](./mypy_linter.py_docs.md)
- [`no_merge_conflict_csv_linter.py_docs.md`](./no_merge_conflict_csv_linter.py_docs.md)
- [`no_workflows_on_fork.py_docs.md`](./no_workflows_on_fork.py_docs.md)
- [`bazel_linter.py_docs.md`](./bazel_linter.py_docs.md)
- [`test_device_bias_linter.py_docs.md`](./test_device_bias_linter.py_docs.md)


## Cross-References

- **File Documentation**: `set_linter.py_docs.md`
- **Keyword Index**: `set_linter.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/tools/linter/adapters`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/tools/linter/adapters`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`docs/tools/linter/adapters`):

- [`pyrefly_linter.py_kw.md_docs.md`](./pyrefly_linter.py_kw.md_docs.md)
- [`codespell_linter.py_kw.md_docs.md`](./codespell_linter.py_kw.md_docs.md)
- [`no_workflows_on_fork.py_kw.md_docs.md`](./no_workflows_on_fork.py_kw.md_docs.md)
- [`bazel_linter.py_kw.md_docs.md`](./bazel_linter.py_kw.md_docs.md)
- [`mypy_linter.py_docs.md_docs.md`](./mypy_linter.py_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`exec_linter.py_kw.md_docs.md`](./exec_linter.py_kw.md_docs.md)
- [`clangformat_linter.py_docs.md_docs.md`](./clangformat_linter.py_docs.md_docs.md)
- [`pip_init.py_kw.md_docs.md`](./pip_init.py_kw.md_docs.md)
- [`testowners_linter.py_docs.md_docs.md`](./testowners_linter.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `set_linter.py_docs.md_docs.md`
- **Keyword Index**: `set_linter.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
