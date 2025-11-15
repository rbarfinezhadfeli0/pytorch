# Documentation: `docs/torch/distributed/rpc/rref_proxy.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/rpc/rref_proxy.py_docs.md`
- **Size**: 5,411 bytes (5.28 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/rpc/rref_proxy.py`

## File Metadata

- **Path**: `torch/distributed/rpc/rref_proxy.py`
- **Size**: 2,705 bytes (2.64 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from functools import partial

import torch
from torch.futures import Future

from . import functions, rpc_async
from .constants import UNSET_RPC_TIMEOUT


def _local_invoke(rref, func_name, args, kwargs):
    return getattr(rref.local_value(), func_name)(*args, **kwargs)


@functions.async_execution
def _local_invoke_async_execution(rref, func_name, args, kwargs):
    return getattr(rref.local_value(), func_name)(*args, **kwargs)


def _invoke_rpc(rref, rpc_api, func_name, timeout, *args, **kwargs):
    def _rref_type_cont(rref_fut):
        rref_type = rref_fut.value()

        _invoke_func = _local_invoke
        # Bypass ScriptModules when checking for async function attribute.
        bypass_type = issubclass(rref_type, torch.jit.ScriptModule) or issubclass(
            rref_type, torch._C.ScriptModule
        )
        if not bypass_type:
            func = getattr(rref_type, func_name)
            if hasattr(func, "_wrapped_async_rpc_function"):
                _invoke_func = _local_invoke_async_execution

        return rpc_api(
            rref.owner(),
            _invoke_func,
            args=(rref, func_name, args, kwargs),
            timeout=timeout,
        )

    rref_fut = rref._get_type(timeout=timeout, blocking=False)

    if rpc_api is not rpc_async:
        rref_fut.wait()
        return _rref_type_cont(rref_fut)
    else:
        # A little explanation on this.
        # rpc_async returns a Future pointing to the return value of `func_name`, it returns a `Future[T]`
        # Calling _rref_type_cont from the `then` lambda causes Future wrapping. IOW, `then` returns a `Future[Future[T]]`
        # To address that, we return a Future that is completed with the result of the async call.
        result: Future = Future()

        def _wrap_rref_type_cont(fut):
            try:
                _rref_type_cont(fut).then(_complete_op)
            except BaseException as ex:  # noqa: B036
                result.set_exception(ex)

        def _complete_op(fut):
            try:
                result.set_result(fut.value())
            except BaseException as ex:  # noqa: B036
                result.set_exception(ex)

        rref_fut.then(_wrap_rref_type_cont)
        return result


# This class manages proxied RPC API calls for RRefs. It is entirely used from
# C++ (see python_rpc_handler.cpp).
class RRefProxy:
    def __init__(self, rref, rpc_api, timeout=UNSET_RPC_TIMEOUT):
        self.rref = rref
        self.rpc_api = rpc_api
        self.rpc_timeout = timeout

    def __getattr__(self, func_name):
        return partial(
            _invoke_rpc, self.rref, self.rpc_api, func_name, self.rpc_timeout
        )

```



## High-Level Overview


This Python file contains 2 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `RRefProxy`

**Functions defined**: `_local_invoke`, `_local_invoke_async_execution`, `_invoke_rpc`, `_rref_type_cont`, `_wrap_rref_type_cont`, `_complete_op`, `__init__`, `__getattr__`

**Key imports**: partial, torch, Future, functions, rpc_async, UNSET_RPC_TIMEOUT


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/rpc`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`: partial
- `torch`
- `torch.futures`: Future
- `.`: functions, rpc_async
- `.constants`: UNSET_RPC_TIMEOUT


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


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

Files in the same folder (`torch/distributed/rpc`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`server_process_global_profiler.py_docs.md`](./server_process_global_profiler.py_docs.md)
- [`internal.py_docs.md`](./internal.py_docs.md)
- [`_utils.py_docs.md`](./_utils.py_docs.md)
- [`functions.py_docs.md`](./functions.py_docs.md)
- [`backend_registry.py_docs.md`](./backend_registry.py_docs.md)
- [`api.py_docs.md`](./api.py_docs.md)
- [`constants.py_docs.md`](./constants.py_docs.md)
- [`options.py_docs.md`](./options.py_docs.md)


## Cross-References

- **File Documentation**: `rref_proxy.py_docs.md`
- **Keyword Index**: `rref_proxy.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/rpc`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/rpc`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/distributed/rpc`):

- [`api.py_kw.md_docs.md`](./api.py_kw.md_docs.md)
- [`internal.py_kw.md_docs.md`](./internal.py_kw.md_docs.md)
- [`server_process_global_profiler.py_kw.md_docs.md`](./server_process_global_profiler.py_kw.md_docs.md)
- [`functions.py_kw.md_docs.md`](./functions.py_kw.md_docs.md)
- [`server_process_global_profiler.py_docs.md_docs.md`](./server_process_global_profiler.py_docs.md_docs.md)
- [`options.py_docs.md_docs.md`](./options.py_docs.md_docs.md)
- [`options.py_kw.md_docs.md`](./options.py_kw.md_docs.md)
- [`_utils.py_kw.md_docs.md`](./_utils.py_kw.md_docs.md)
- [`_utils.py_docs.md_docs.md`](./_utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `rref_proxy.py_docs.md_docs.md`
- **Keyword Index**: `rref_proxy.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
