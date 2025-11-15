# Documentation: `docs/torch/backends/mkldnn/__init__.py_docs.md`

## File Metadata

- **Path**: `docs/torch/backends/mkldnn/__init__.py_docs.md`
- **Size**: 6,570 bytes (6.42 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This is a **Python package initialization file**.

## Original Source

```markdown
# Documentation: `torch/backends/mkldnn/__init__.py`

## File Metadata

- **Path**: `torch/backends/mkldnn/__init__.py`
- **Size**: 4,392 bytes (4.29 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a **Python package initialization file**.

## Original Source

```python
# mypy: allow-untyped-defs
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
from torch.backends import (
    __allow_nonbracketed_mutation,
    _FP32Precision,
    _get_fp32_precision_getter,
    _set_fp32_precision_setter,
    ContextProp,
    PropModule,
)


def is_available():
    r"""Return whether PyTorch is built with MKL-DNN support."""
    return torch._C._has_mkldnn


def is_acl_available():
    r"""Return whether PyTorch is built with MKL-DNN + ACL support."""
    # pyrefly: ignore [missing-attribute]
    return torch._C._has_mkldnn_acl


VERBOSE_OFF = 0
VERBOSE_ON = 1
VERBOSE_ON_CREATION = 2


class verbose:
    """
    On-demand oneDNN (former MKL-DNN) verbosing functionality.

    To make it easier to debug performance issues, oneDNN can dump verbose
    messages containing information like kernel size, input data size and
    execution duration while executing the kernel. The verbosing functionality
    can be invoked via an environment variable named `DNNL_VERBOSE`. However,
    this methodology dumps messages in all steps. Those are a large amount of
    verbose messages. Moreover, for investigating the performance issues,
    generally taking verbose messages for one single iteration is enough.
    This on-demand verbosing functionality makes it possible to control scope
    for verbose message dumping. In the following example, verbose messages
    will be dumped out for the second inference only.

    .. highlight:: python
    .. code-block:: python

        import torch

        model(data)
        with torch.backends.mkldnn.verbose(torch.backends.mkldnn.VERBOSE_ON):
            model(data)

    Args:
        level: Verbose level
            - ``VERBOSE_OFF``: Disable verbosing
            - ``VERBOSE_ON``:  Enable verbosing
            - ``VERBOSE_ON_CREATION``: Enable verbosing, including oneDNN kernel creation
    """

    def __init__(self, level):
        self.level = level

    def __enter__(self):
        if self.level == VERBOSE_OFF:
            return
        st = torch._C._verbose.mkldnn_set_verbose(self.level)
        assert st, (
            "Failed to set MKLDNN into verbose mode. Please consider to disable this verbose scope."
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch._C._verbose.mkldnn_set_verbose(VERBOSE_OFF)
        return False


def set_flags(
    _enabled=None, _deterministic=None, _allow_tf32=None, _fp32_precision="none"
):
    orig_flags = (
        torch._C._get_mkldnn_enabled(),
        torch._C._get_mkldnn_deterministic(),
        torch._C._get_onednn_allow_tf32(),
        torch._C._get_fp32_precision_getter("mkldnn", "all"),
    )
    if _enabled is not None:
        torch._C._set_mkldnn_enabled(_enabled)
    if _deterministic is not None:
        torch._C._set_mkldnn_deterministic(_deterministic)
    if _allow_tf32 is not None:
        torch._C._set_onednn_allow_tf32(_allow_tf32)
    if _fp32_precision is not None:
        torch._C._set_fp32_precision_setter("mkldnn", "all", _fp32_precision)
    return orig_flags


@contextmanager
def flags(enabled=False, deterministic=False, allow_tf32=True, fp32_precision="none"):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(enabled, deterministic, allow_tf32, fp32_precision)
    try:
        yield
    finally:
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)


class MkldnnModule(PropModule):
    def __init__(self, m, name):
        super().__init__(m, name)

    def is_available(self):
        return is_available()

    enabled = ContextProp(torch._C._get_mkldnn_enabled, torch._C._set_mkldnn_enabled)
    deterministic = ContextProp(
        torch._C._get_mkldnn_deterministic, torch._C._set_mkldnn_deterministic
    )
    allow_tf32 = ContextProp(
        torch._C._get_onednn_allow_tf32, torch._C._set_onednn_allow_tf32
    )
    matmul = _FP32Precision("mkldnn", "matmul")
    conv = _FP32Precision("mkldnn", "conv")
    rnn = _FP32Precision("mkldnn", "rnn")
    fp32_precision = ContextProp(
        _get_fp32_precision_getter("mkldnn", "all"),
        _set_fp32_precision_setter("generic", "all"),
    )


if TYPE_CHECKING:
    enabled: ContextProp
    deterministic: ContextProp
    allow_tf32: ContextProp

sys.modules[__name__] = MkldnnModule(sys.modules[__name__], __name__)

```



## High-Level Overview

r"""Return whether PyTorch is built with MKL-DNN support."""    return torch._C._has_mkldnndef is_acl_available():

This Python file contains 2 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `verbose`, `MkldnnModule`

**Functions defined**: `is_available`, `is_acl_available`, `__init__`, `__enter__`, `__exit__`, `set_flags`, `flags`, `__init__`, `is_available`

**Key imports**: sys, contextmanager, TYPE_CHECKING, torch, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/backends/mkldnn`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `contextlib`: contextmanager
- `typing`: TYPE_CHECKING
- `torch`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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

Files in the same folder (`torch/backends/mkldnn`):



## Cross-References

- **File Documentation**: `__init__.py_docs.md`
- **Keyword Index**: `__init__.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/backends/mkldnn`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/backends/mkldnn`, which is part of the **core PyTorch library**.



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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/backends/mkldnn`):

- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `__init__.py_docs.md_docs.md`
- **Keyword Index**: `__init__.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
