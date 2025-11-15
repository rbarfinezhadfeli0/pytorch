# Documentation: `docs/tools/gdb/pytorch-gdb.py_docs.md`

## File Metadata

- **Path**: `docs/tools/gdb/pytorch-gdb.py_docs.md`
- **Size**: 5,585 bytes (5.45 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/gdb/pytorch-gdb.py`

## File Metadata

- **Path**: `tools/gdb/pytorch-gdb.py`
- **Size**: 3,433 bytes (3.35 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
import textwrap
from typing import Any

import gdb  # type: ignore[import]


class DisableBreakpoints:
    """
    Context-manager to temporarily disable all gdb breakpoints, useful if
    there is a risk to hit one during the evaluation of one of our custom
    commands
    """

    def __enter__(self) -> None:
        self.disabled_breakpoints = []
        for b in gdb.breakpoints():
            if b.enabled:
                b.enabled = False
                self.disabled_breakpoints.append(b)

    def __exit__(self, etype: Any, evalue: Any, tb: Any) -> None:
        for b in self.disabled_breakpoints:
            b.enabled = True


class TensorRepr(gdb.Command):  # type: ignore[misc, no-any-unimported]
    """
    Print a human readable representation of the given at::Tensor.
    Usage: torch-tensor-repr EXP

    at::Tensor instances do not have a C++ implementation of a repr method: in
    pytorch, this is done by pure-Python code. As such, torch-tensor-repr
    internally creates a Python wrapper for the given tensor and call repr()
    on it.
    """

    # pyrefly: ignore [bad-argument-type]
    __doc__ = textwrap.dedent(__doc__).strip()

    def __init__(self) -> None:
        gdb.Command.__init__(
            self, "torch-tensor-repr", gdb.COMMAND_USER, gdb.COMPLETE_EXPRESSION
        )

    def invoke(self, args: str, from_tty: bool) -> None:
        args = gdb.string_to_argv(args)
        if len(args) != 1:
            print("Usage: torch-tensor-repr EXP")
            return
        name = args[0]
        with DisableBreakpoints():
            res = gdb.parse_and_eval(f"torch::gdb::tensor_repr({name})")
            print(f"Python-level repr of {name}:")
            print(res.string())
            # torch::gdb::tensor_repr returns a malloc()ed buffer, let's free it
            gdb.parse_and_eval(f"(void)free({int(res)})")


class IntArrayRefRepr(gdb.Command):  # type: ignore[misc, no-any-unimported]
    """
    Print human readable representation of c10::IntArrayRef
    """

    def __init__(self) -> None:
        gdb.Command.__init__(
            self, "torch-int-array-ref-repr", gdb.COMMAND_USER, gdb.COMPLETE_EXPRESSION
        )

    def invoke(self, args: str, from_tty: bool) -> None:
        args = gdb.string_to_argv(args)
        if len(args) != 1:
            print("Usage: torch-int-array-ref-repr EXP")
            return
        name = args[0]
        with DisableBreakpoints():
            res = gdb.parse_and_eval(f"torch::gdb::int_array_ref_string({name})")
            res = str(res)
            print(res[res.find('"') + 1 : -1])


class DispatchKeysetRepr(gdb.Command):  # type: ignore[misc, no-any-unimported]
    """
    Print human readable representation of c10::DispatchKeyset
    """

    def __init__(self) -> None:
        gdb.Command.__init__(
            self,
            "torch-dispatch-keyset-repr",
            gdb.COMMAND_USER,
            gdb.COMPLETE_EXPRESSION,
        )

    def invoke(self, args: str, from_tty: bool) -> None:
        args = gdb.string_to_argv(args)
        if len(args) != 1:
            print("Usage: torch-dispatch-keyset-repr EXP")
            return
        keyset = args[0]
        with DisableBreakpoints():
            res = gdb.parse_and_eval(f"torch::gdb::dispatch_keyset_string({keyset})")
            res = str(res)
            print(res[res.find('"') + 1 : -1])


TensorRepr()
IntArrayRefRepr()
DispatchKeysetRepr()

```



## High-Level Overview

"""    Context-manager to temporarily disable all gdb breakpoints, useful if    there is a risk to hit one during the evaluation of one of our custom    commands

This Python file contains 4 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `DisableBreakpoints`, `TensorRepr`, `IntArrayRefRepr`, `DispatchKeysetRepr`

**Functions defined**: `__enter__`, `__exit__`, `__init__`, `invoke`, `__init__`, `invoke`, `__init__`, `invoke`

**Key imports**: textwrap, Any, gdb  


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/gdb`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `textwrap`
- `typing`: Any
- `gdb  `


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Context Manager**: Implements context manager protocol


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`tools/gdb`):



## Cross-References

- **File Documentation**: `pytorch-gdb.py_docs.md`
- **Keyword Index**: `pytorch-gdb.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/tools/gdb`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/tools/gdb`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Context Manager**: Implements context manager protocol


## Performance Considerations

### Performance Notes

- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/tools/gdb`):

- [`pytorch-gdb.py_kw.md_docs.md`](./pytorch-gdb.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `pytorch-gdb.py_docs.md_docs.md`
- **Keyword Index**: `pytorch-gdb.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
