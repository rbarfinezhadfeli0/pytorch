# Documentation: `tools/lldb/pytorch_lldb.py`

## File Metadata

- **Path**: `tools/lldb/pytorch_lldb.py`
- **Size**: 3,443 bytes (3.36 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
from typing import Any

import lldb  # type: ignore[import]


def get_target() -> Any:
    target = lldb.debugger.GetSelectedTarget()
    if not target:
        print("[-] error: no target available. please add a target to lldb.")
        return None
    return target


class DisableBreakpoints:
    """
    Context-manager to temporarily disable all lldb breakpoints, useful if
    there is a risk to hit one during the evaluation of one of our custom
    commands
    """

    def __enter__(self) -> None:
        target = get_target()

        if target.DisableAllBreakpoints() is False:
            print("[-] error: failed to disable all breakpoints.")

    def __exit__(self, etype: Any, evalue: Any, tb: Any) -> None:
        target = get_target()

        if target.EnableAllBreakpoints() is False:
            print("[-] error: failed to enable all breakpoints.")


def IntArrayRef_summary(valobj: Any, internal_dict: Any, options: Any) -> str:
    """Print human readable representation of c10::IntArrayRef"""
    with DisableBreakpoints():
        target = get_target()
        tensor = valobj.GetName()
        result = target.EvaluateExpression(
            f"torch::gdb::int_array_ref_string({tensor})"
        )
        str_result = str(result)
        str_result = str_result[str_result.find('"') + 1 : -1]
        return str_result


def DispatchKeyset_summary(valobj: Any, internal_dict: Any, options: Any) -> str:
    """Print human readable representation of c10::DispatchKeyset"""
    with DisableBreakpoints():
        target = get_target()
        keyset = valobj.GetName()
        result = target.EvaluateExpression(
            f"torch::gdb::dispatch_keyset_string({keyset})"
        )
        str_result = str(result)
        str_result = str_result[str_result.find('"') + 1 : -1]
        return str_result


def Tensor_summary(valobj: Any, internal_dict: Any, options: Any) -> str:
    """Print a human readable representation of the given at::Tensor.

    at::Tensor instances do not have a C++ implementation of a repr method: in
    pytorch, this is done by pure-Python code. As such, print <tensor>
    internally creates a Python wrapper for the given tensor and call repr()
    on it.
    Usage:
        print self
    """
    with DisableBreakpoints():
        target = get_target()
        tensor = valobj.GetName()
        result = target.EvaluateExpression(f"torch::gdb::tensor_repr({tensor})")
        str_result = str(result)
        target.EvaluateExpression(f"(void)free({result.GetValue()})")
        str_result = "\n" + str_result[str_result.find("tensor") : -1]
        return str_result


# And the initialization code to add your commands
def __lldb_init_module(debugger: Any, internal_dict: Any) -> Any:
    debugger.HandleCommand(
        "type summary add c10::IntArrayRef -F pytorch_lldb.IntArrayRef_summary -w torch"
    )
    debugger.HandleCommand(
        "type summary add c10::DispatchKeySet -F pytorch_lldb.DispatchKeyset_summary -w torch"
    )
    debugger.HandleCommand(
        "type summary add at::Tensor -F pytorch_lldb.Tensor_summary -w torch"
    )
    print(
        "Pretty Printing lldb summary for PyTorch AT types has been installed and is ready for use. "
        "This category is enabled by default. To disable run: `type category disable torch`"
    )
    print(
        "Usage:\n\tprint <at::tensor>\n\tprint <c10::IntArrayRef>\n\tprint <c10::DispatchKeySet>"
    )

```



## High-Level Overview

"""    Context-manager to temporarily disable all lldb breakpoints, useful if    there is a risk to hit one during the evaluation of one of our custom    commands

This Python file contains 1 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `DisableBreakpoints`

**Functions defined**: `get_target`, `__enter__`, `__exit__`, `IntArrayRef_summary`, `DispatchKeyset_summary`, `Tensor_summary`, `__lldb_init_module`

**Key imports**: Any, lldb  


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/lldb`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Any
- `lldb  `


## Code Patterns & Idioms

### Common Patterns

- **Context Manager**: Implements context manager protocol


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

Files in the same folder (`tools/lldb`):



## Cross-References

- **File Documentation**: `pytorch_lldb.py_docs.md`
- **Keyword Index**: `pytorch_lldb.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
