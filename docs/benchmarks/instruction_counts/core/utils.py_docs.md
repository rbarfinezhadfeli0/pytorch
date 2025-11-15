# Documentation: `benchmarks/instruction_counts/core/utils.py`

## File Metadata

- **Path**: `benchmarks/instruction_counts/core/utils.py`
- **Size**: 3,589 bytes (3.50 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```python
# mypy: ignore-errors
import atexit
import re
import shutil
import textwrap
from typing import Optional

from core.api import GroupedBenchmark, TimerArgs
from core.types import Definition, FlatIntermediateDefinition, Label

from torch.utils.benchmark.utils.common import _make_temp_dir


_TEMPDIR: Optional[str] = None


def get_temp_dir() -> str:
    global _TEMPDIR
    if _TEMPDIR is None:
        _TEMPDIR = _make_temp_dir(
            prefix="instruction_count_microbenchmarks", gc_dev_shm=True
        )
        atexit.register(shutil.rmtree, path=_TEMPDIR)
    return _TEMPDIR


def _flatten(
    key_prefix: Label, sub_schema: Definition, result: FlatIntermediateDefinition
) -> None:
    for k, value in sub_schema.items():
        if isinstance(k, tuple):
            assert all(isinstance(ki, str) for ki in k)
            key_suffix: Label = k
        elif k is None:
            key_suffix = ()
        else:
            assert isinstance(k, str)
            key_suffix = (k,)

        key: Label = key_prefix + key_suffix
        if isinstance(value, (TimerArgs, GroupedBenchmark)):
            assert key not in result, f"duplicate key: {key}"
            result[key] = value
        else:
            assert isinstance(value, dict)
            _flatten(key_prefix=key, sub_schema=value, result=result)


def flatten(schema: Definition) -> FlatIntermediateDefinition:
    """See types.py for an explanation of nested vs. flat definitions."""
    result: FlatIntermediateDefinition = {}
    _flatten(key_prefix=(), sub_schema=schema, result=result)

    # Ensure that we produced a valid flat definition.
    for k, v in result.items():
        assert isinstance(k, tuple)
        assert all(isinstance(ki, str) for ki in k)
        assert isinstance(v, (TimerArgs, GroupedBenchmark))
    return result


def parse_stmts(stmts: str) -> tuple[str, str]:
    """Helper function for side-by-side Python and C++ stmts.

    For more complex statements, it can be useful to see Python and C++ code
    side by side. To this end, we provide an **extremely restricted** way
    to define Python and C++ code side-by-side. The schema should be mostly
    self explanatory, with the following non-obvious caveats:
      - Width for the left (Python) column MUST be 40 characters.
      - The column separator is " | ", not "|". Whitespace matters.
    """
    stmts = textwrap.dedent(stmts).strip()
    lines: list[str] = stmts.splitlines(keepends=False)
    assert len(lines) >= 3, f"Invalid string:\n{stmts}"

    column_header_pattern = r"^Python\s{35}\| C\+\+(\s*)$"
    signature_pattern = r"^: f\((.*)\)( -> (.+))?\s*$"  # noqa: F841
    separation_pattern = r"^[-]{40} | [-]{40}$"
    code_pattern = r"^(.{40}) \|($| (.*)$)"

    column_match = re.search(column_header_pattern, lines[0])
    if column_match is None:
        raise ValueError(
            f"Column header `{lines[0]}` "
            f"does not match pattern `{column_header_pattern}`"
        )

    assert re.search(separation_pattern, lines[1])

    py_lines: list[str] = []
    cpp_lines: list[str] = []
    for l in lines[2:]:
        l_match = re.search(code_pattern, l)
        if l_match is None:
            raise ValueError(f"Invalid line `{l}`")
        py_lines.append(l_match.groups()[0])
        cpp_lines.append(l_match.groups()[2] or "")

        # Make sure we can round trip for correctness.
        l_from_stmts = f"{py_lines[-1]:<40} | {cpp_lines[-1]:<40}".rstrip()
        assert l_from_stmts == l.rstrip(), f"Failed to round trip `{l}`"

    return "\n".join(py_lines), "\n".join(cpp_lines)

```



## High-Level Overview

"""See types.py for an explanation of nested vs. flat definitions."""

This Python file contains 0 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `get_temp_dir`, `_flatten`, `flatten`, `parse_stmts`

**Key imports**: atexit, re, shutil, textwrap, Optional, GroupedBenchmark, TimerArgs, Definition, FlatIntermediateDefinition, Label, _make_temp_dir


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/instruction_counts/core`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `atexit`
- `re`
- `shutil`
- `textwrap`
- `typing`: Optional
- `core.api`: GroupedBenchmark, TimerArgs
- `core.types`: Definition, FlatIntermediateDefinition, Label
- `torch.utils.benchmark.utils.common`: _make_temp_dir


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

Files in the same folder (`benchmarks/instruction_counts/core`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`types.py_docs.md`](./types.py_docs.md)
- [`api.py_docs.md`](./api.py_docs.md)
- [`expand.py_docs.md`](./expand.py_docs.md)


## Cross-References

- **File Documentation**: `utils.py_docs.md`
- **Keyword Index**: `utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
