# Documentation: `docs/torch/testing/_internal/logging_tensor.py_docs.md`

## File Metadata

- **Path**: `docs/torch/testing/_internal/logging_tensor.py_docs.md`
- **Size**: 9,866 bytes (9.63 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/testing/_internal/logging_tensor.py`

## File Metadata

- **Path**: `torch/testing/_internal/logging_tensor.py`
- **Size**: 6,646 bytes (6.49 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```python
# mypy: ignore-errors

import torch
from torch.utils._pytree import tree_map
from typing import Optional
from collections.abc import Iterator
import logging
import contextlib
import itertools
from torch.utils._dtype_abbrs import dtype_abbrs as _dtype_abbrs
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.weak import WeakTensorKeyDictionary
import functools
from torch._C._profiler import gather_traceback, symbolize_tracebacks

logger = logging.getLogger("LoggingTensor")

# How the chain of calls works for LoggingTensor:
# 1. Call torch.sin
# 2. Attempt __torch_function__. In LoggingTensor torch function is disabled so we bypass it entirely
# 3. Enter dispatcher, wind your way through Autograd
# 4. Hit Python dispatch key, call __torch_dispatch__

# This Tensor can work with autograd in two ways:
#  - The wrapped Tensor does not require gradients. In that case, the LoggingTensor
#    can require gradients if the user asks for it as a constructor kwarg.
#  - The wrapped Tensor can require gradients. In that case autograd will be tracked
#    for the wrapped Tensor and the LoggingTensor itself cannot require gradients.
# WARNING: We allow these two possibilities for testing purposes. You should NEVER use both in a single
# test or you might get surprising behavior.

# TODO: TensorBase should work
class LoggingTensor(torch.Tensor):
    elem: torch.Tensor

    __slots__ = ['elem']

    context = contextlib.nullcontext

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        # The wrapping tensor (LoggingTensor) shouldn't hold any
        # memory for the class in question, but it should still
        # advertise the same device as before
        r = torch.Tensor._make_wrapper_subclass(
            cls, elem.size(),
            strides=elem.stride(), storage_offset=elem.storage_offset(),
            # TODO: clone storage aliasing
            dtype=elem.dtype, layout=elem.layout,
            device=elem.device, requires_grad=kwargs.get("requires_grad", False)
        )
        # ...the real tensor is held as an element on the tensor.
        r.elem = elem.detach() if r.requires_grad else elem
        return r

    def __repr__(self):
        return super().__repr__(tensor_contents=f"{self.elem}")

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(e):
            return e.elem if isinstance(e, cls) else e

        def wrap(e):
            return cls(e) if isinstance(e, torch.Tensor) else e

        with cls.context():
            rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))
        logging.getLogger("LoggingTensor").info(f"{func.__module__}.{func.__name__}", args, kwargs, rs)  # noqa: G004
        return rs

class LoggingTensorMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        rs = func(*args, **kwargs)
        logging.getLogger("LoggingTensor").info(f"{func.__module__}.{func.__name__}", args, kwargs, rs)  # noqa: G004
        return rs

class LoggingTensorReentrant(LoggingTensor):
    context = torch.overrides.enable_reentrant_dispatch

# https://stackoverflow.com/questions/36408496/python-logging-handler-to-append-to-list
class LoggingTensorHandler(logging.Handler):
    def __init__(
            self, log_list: list[str], use_shortid_for_all_tensors: bool,
            with_type: bool, tracebacks_list: Optional[list]) -> None:
        logging.Handler.__init__(self)
        self.log_list = log_list
        self.use_shortid_for_all_tensors = use_shortid_for_all_tensors
        self.tracebacks_list = tracebacks_list
        self.memo = WeakTensorKeyDictionary()
        self.next_id = 0
        self.with_type = with_type

    def _shortid(self, t: torch.Tensor) -> int:
        if t not in self.memo:
            self.memo[t] = self.next_id
            self.next_id += 1
        return self.memo[t]

    def _fmt(self, a: object, with_type: bool = False) -> str:
        cond_cls = torch.Tensor if self.use_shortid_for_all_tensors else LoggingTensor
        if isinstance(a, cond_cls):
            maybe_type = ""
            if with_type and self.with_type:
                maybe_type = f": {_dtype_abbrs[a.dtype]}[{', '.join(map(str, a.shape))}]"
            x = f"${self._shortid(a)}{maybe_type}"
            return x
        else:
            return repr(a)

    def emit(self, record):
        fmt_args = ", ".join(
            itertools.chain(
                (str(tree_map(self._fmt, a)) for a in record.args[0]),
                (f"{k}={str(tree_map(self._fmt, v))}" for k, v in record.args[1].items()),
            )
        )
        fmt_rets = tree_map(functools.partial(self._fmt, with_type=True), record.args[2])
        self.log_list.append(f'{fmt_rets} = {record.msg}({fmt_args})')
        if self.tracebacks_list is not None:
            self.tracebacks_list.append(record.traceback)

def log_input(name: str, var: object) -> None:
    logger.info("input", (name,), {}, var)  # noqa: PLE1205

class GatherTraceback(logging.Filter):
    def __init__(self, python=True, script=True, cpp=False):
        self.python = python
        self.script = script
        self.cpp = cpp

    def filter(self, record):
        record.traceback = gather_traceback(python=self.python, script=self.script, cpp=self.cpp)
        return True

@contextlib.contextmanager
def capture_logs(is_mode=False, python_tb=False, script_tb=False, cpp_tb=False) -> Iterator[list[str]]:
    collect_traceback = python_tb or script_tb or cpp_tb
    log_list: list[str] = []
    tracebacks_list: list[str] = []
    handler = LoggingTensorHandler(
        log_list,
        with_type=True,
        use_shortid_for_all_tensors=is_mode,
        tracebacks_list=tracebacks_list if collect_traceback else None
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if collect_traceback:
        logger.addFilter(GatherTraceback(python=python_tb, script=script_tb, cpp=cpp_tb))
    try:
        if collect_traceback:
            yield log_list, tracebacks_list
        else:
            yield log_list
    finally:
        symbolized_tracebacks = symbolize_tracebacks(tracebacks_list)
        tracebacks_list.clear()
        tracebacks_list.extend(symbolized_tracebacks)
        logger.removeHandler(handler)

@contextlib.contextmanager
def capture_logs_with_logging_tensor_mode(python_tb=False, script_tb=False, cpp_tb=False):
    with LoggingTensorMode(), capture_logs(True, python_tb, script_tb, cpp_tb) as logs:
        yield logs

```



## High-Level Overview


This Python file contains 6 class(es) and 15 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LoggingTensor`, `LoggingTensorMode`, `LoggingTensorReentrant`, `LoggingTensorHandler`, `GatherTraceback`

**Functions defined**: `__new__`, `__repr__`, `__torch_dispatch__`, `unwrap`, `wrap`, `__torch_dispatch__`, `__init__`, `_shortid`, `_fmt`, `emit`, `log_input`, `__init__`, `filter`, `capture_logs`, `capture_logs_with_logging_tensor_mode`

**Key imports**: torch, tree_map, Optional, Iterator, logging, contextlib, itertools, dtype_abbrs as _dtype_abbrs, TorchDispatchMode, WeakTensorKeyDictionary


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/testing/_internal`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.utils._pytree`: tree_map
- `typing`: Optional
- `collections.abc`: Iterator
- `logging`
- `contextlib`
- `itertools`
- `torch.utils._dtype_abbrs`: dtype_abbrs as _dtype_abbrs
- `torch.utils._python_dispatch`: TorchDispatchMode
- `torch.utils.weak`: WeakTensorKeyDictionary
- `functools`
- `torch._C._profiler`: gather_traceback, symbolize_tracebacks


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python torch/testing/_internal/logging_tensor.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/testing/_internal`):

- [`common_jit.py_docs.md`](./common_jit.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`autograd_function_db.py_docs.md`](./autograd_function_db.py_docs.md)
- [`custom_op_db.py_docs.md`](./custom_op_db.py_docs.md)
- [`subclasses.py_docs.md`](./subclasses.py_docs.md)
- [`logging_utils.py_docs.md`](./logging_utils.py_docs.md)
- [`two_tensor.py_docs.md`](./two_tensor.py_docs.md)
- [`autocast_test_lists.py_docs.md`](./autocast_test_lists.py_docs.md)
- [`hypothesis_utils.py_docs.md`](./hypothesis_utils.py_docs.md)
- [`common_mkldnn.py_docs.md`](./common_mkldnn.py_docs.md)


## Cross-References

- **File Documentation**: `logging_tensor.py_docs.md`
- **Keyword Index**: `logging_tensor.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/testing/_internal`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/testing/_internal`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/torch/testing/_internal/logging_tensor.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/testing/_internal`):

- [`hypothesis_utils.py_kw.md_docs.md`](./hypothesis_utils.py_kw.md_docs.md)
- [`fake_config_module2.py_kw.md_docs.md`](./fake_config_module2.py_kw.md_docs.md)
- [`static_module.py_kw.md_docs.md`](./static_module.py_kw.md_docs.md)
- [`common_pruning.py_kw.md_docs.md`](./common_pruning.py_kw.md_docs.md)
- [`composite_compliance.py_kw.md_docs.md`](./composite_compliance.py_kw.md_docs.md)
- [`common_mkldnn.py_docs.md_docs.md`](./common_mkldnn.py_docs.md_docs.md)
- [`triton_utils.py_docs.md_docs.md`](./triton_utils.py_docs.md_docs.md)
- [`common_dtype.py_docs.md_docs.md`](./common_dtype.py_docs.md_docs.md)
- [`common_methods_invocations.py_docs.md_docs.md`](./common_methods_invocations.py_docs.md_docs.md)
- [`hypothesis_utils.py_docs.md_docs.md`](./hypothesis_utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `logging_tensor.py_docs.md_docs.md`
- **Keyword Index**: `logging_tensor.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
