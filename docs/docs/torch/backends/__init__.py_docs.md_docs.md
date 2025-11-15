# Documentation: `docs/torch/backends/__init__.py_docs.md`

## File Metadata

- **Path**: `docs/torch/backends/__init__.py_docs.md`
- **Size**: 5,874 bytes (5.74 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This is a **Python package initialization file**.

## Original Source

```markdown
# Documentation: `torch/backends/__init__.py`

## File Metadata

- **Path**: `torch/backends/__init__.py`
- **Size**: 3,654 bytes (3.57 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a **Python package initialization file**.

## Original Source

```python
# mypy: allow-untyped-defs
import sys
import types
from contextlib import contextmanager

import torch


# The idea for this parameter is that we forbid bare assignment
# to torch.backends.<cudnn|mkldnn>.enabled and friends when running our
# test suite, where it's very easy to forget to undo the change
# later.
__allow_nonbracketed_mutation_flag = True


def disable_global_flags():
    global __allow_nonbracketed_mutation_flag
    __allow_nonbracketed_mutation_flag = False


def flags_frozen():
    return not __allow_nonbracketed_mutation_flag


@contextmanager
def __allow_nonbracketed_mutation():
    global __allow_nonbracketed_mutation_flag
    old = __allow_nonbracketed_mutation_flag
    __allow_nonbracketed_mutation_flag = True
    try:
        yield
    finally:
        __allow_nonbracketed_mutation_flag = old


class ContextProp:
    def __init__(self, getter, setter):
        self.getter = getter
        self.setter = setter

    def __get__(self, obj, objtype):
        return self.getter()

    def __set__(self, obj, val):
        if not flags_frozen():
            self.setter(val)
        else:
            raise RuntimeError(
                f"not allowed to set {obj.__name__} flags "
                "after disable_global_flags; please use flags() context manager instead"
            )


class PropModule(types.ModuleType):
    def __init__(self, m, name):
        super().__init__(name)
        self.m = m

    def __getattr__(self, attr):
        return self.m.__getattribute__(attr)


class _FP32Precision:
    def __init__(self, backend, op):
        self.backend = backend
        self.op = op

    def __setattr__(self, name, value):
        if name == "fp32_precision":
            torch._C._set_fp32_precision_setter(self.backend, self.op, value)
        elif name in ("backend", "op"):
            super().__setattr__(name, value)
        else:
            raise AttributeError("Unknown attribute " + name)

    def __getattr__(self, name):
        if name == "fp32_precision":
            return torch._C._get_fp32_precision_getter(self.backend, self.op)
        else:
            raise AttributeError("Unknown attribute " + name)


def set_flags(_fp32_precision="none"):
    orig_flags = (torch._C._get_fp32_precision_getter("generic", "all"),)
    if _fp32_precision is not None:
        torch._C._set_fp32_precision_setter("generic", "all", _fp32_precision)
    return orig_flags


@contextmanager
def flags(fp32_precision="none"):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(fp32_precision)
    try:
        yield
    finally:
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)


def _get_fp32_precision_getter(backend, op):
    def inner():
        return torch._C._get_fp32_precision_getter(backend, op)

    return inner


def _set_fp32_precision_setter(backend, op):
    def inner(precision):
        return torch._C._set_fp32_precision_setter(backend, op, precision)

    return inner


class GenericModule(PropModule):
    def __init__(self, m, name):
        super().__init__(m, name)

    fp32_precision = ContextProp(
        _get_fp32_precision_getter("generic", "all"),
        _set_fp32_precision_setter("generic", "all"),
    )


sys.modules[__name__] = GenericModule(sys.modules[__name__], __name__)

from torch.backends import (
    cpu as cpu,
    cuda as cuda,
    cudnn as cudnn,
    cusparselt as cusparselt,
    kleidiai as kleidiai,
    mha as mha,
    miopen as miopen,
    mkl as mkl,
    mkldnn as mkldnn,
    mps as mps,
    nnpack as nnpack,
    openmp as openmp,
    opt_einsum as opt_einsum,
    quantized as quantized,
)

```



## High-Level Overview


This Python file contains 4 class(es) and 18 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ContextProp`, `PropModule`, `_FP32Precision`, `GenericModule`

**Functions defined**: `disable_global_flags`, `flags_frozen`, `__allow_nonbracketed_mutation`, `__init__`, `__get__`, `__set__`, `__init__`, `__getattr__`, `__init__`, `__setattr__`, `__getattr__`, `set_flags`, `flags`, `_get_fp32_precision_getter`, `inner`, `_set_fp32_precision_setter`, `inner`, `__init__`

**Key imports**: sys, types, contextmanager, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/backends`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `types`
- `contextlib`: contextmanager
- `torch`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`torch/backends`):



## Cross-References

- **File Documentation**: `__init__.py_docs.md`
- **Keyword Index**: `__init__.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/backends`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/backends`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/torch/backends`):

- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `__init__.py_docs.md_docs.md`
- **Keyword Index**: `__init__.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
