# Documentation: `docs/torch/utils/_device.py_docs.md`

## File Metadata

- **Path**: `docs/torch/utils/_device.py_docs.md`
- **Size**: 6,760 bytes (6.60 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/utils/_device.py`

## File Metadata

- **Path**: `torch/utils/_device.py`
- **Size**: 3,996 bytes (3.90 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import functools

import torch
from torch._C import _len_torch_function_stack
from torch.overrides import _pop_mode, _push_mode, TorchFunctionMode
from torch.utils._contextlib import context_decorator


CURRENT_DEVICE: torch.device | None = None


@functools.lru_cache(1)
def _device_constructors():
    return {
        # standard ones
        torch.empty,
        torch.empty_permuted,
        torch.empty_strided,
        torch.empty_quantized,
        torch.ones,
        torch.arange,
        torch.bartlett_window,
        torch.blackman_window,
        torch.eye,
        torch.fft.fftfreq,
        torch.fft.rfftfreq,
        torch.full,
        torch.hamming_window,
        torch.hann_window,
        torch.kaiser_window,
        torch.linspace,
        torch.logspace,
        torch.nested.nested_tensor,
        # This function doesn't actually take a device argument
        # torch.normal,
        torch.rand,
        torch.randn,
        torch.randint,
        torch.randperm,
        torch.range,
        torch.sparse_coo_tensor,
        torch.sparse_compressed_tensor,
        torch.sparse_csr_tensor,
        torch.sparse_csc_tensor,
        torch.sparse_bsr_tensor,
        torch.sparse_bsc_tensor,
        torch.tril_indices,
        torch.triu_indices,
        torch.zeros,
        torch.asarray,
        # weird ones
        torch.tensor,
        torch.as_tensor,
        torch.scalar_tensor,
    }


# NB: This is directly called from C++ in torch/csrc/Device.cpp
class DeviceContext(TorchFunctionMode):
    def __init__(self, device) -> None:
        # pyrefly: ignore [read-only]
        self.device = torch.device(device)

    def __enter__(self):
        global CURRENT_DEVICE
        self.old_device = CURRENT_DEVICE
        CURRENT_DEVICE = self.device
        # We need to put the device at the bottom of the stack
        # If we set default device within a function mode context
        # exiting that context mode will pop the device function mode off
        # of the stack incorrectly
        cur_stack = [_pop_mode() for _ in range(_len_torch_function_stack())]

        _push_mode(self)

        for mode in reversed(cur_stack):
            _push_mode(mode)

    def __exit__(self, exc_type, exc_val, exc_tb):
        global CURRENT_DEVICE
        CURRENT_DEVICE = self.old_device
        cur_stack = []
        # Invariant: there should only be one DeviceContext on the stack at any time
        # (At the bottom), pop all modes until we hit the bottom, assert it's a DeviceContext
        # or else someone else has popped it!
        for _ in range(_len_torch_function_stack() - 1):
            mode = _pop_mode()
            if isinstance(mode, DeviceContext):
                raise AssertionError(
                    "Found nested DeviceContext on the mode stack where none expected"
                )
            cur_stack.append(mode)

        if _len_torch_function_stack() > 0:
            mode = _pop_mode()
            if not isinstance(mode, DeviceContext):
                raise AssertionError(
                    "Expected a DeviceContext at the bottom of the mode stack"
                )

        for mode in reversed(cur_stack):
            _push_mode(mode)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _device_constructors() and kwargs.get("device") is None:
            kwargs["device"] = self.device
        return func(*args, **kwargs)


# NB: This is directly called from C++ in torch/csrc/Device.cpp
def device_decorator(device, func):
    return context_decorator(lambda: device, func)


def set_device(device):
    """
    Set the default device inside of the wrapped function by decorating it with this function.

    If you would like to use this as a context manager, use device as a
    context manager directly, e.g., ``with torch.device(device)``.
    """
    return lambda func: device_decorator(torch.device(device), func)

```



## High-Level Overview


This Python file contains 1 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `DeviceContext`

**Functions defined**: `_device_constructors`, `__init__`, `__enter__`, `__exit__`, `__torch_function__`, `device_decorator`, `set_device`

**Key imports**: functools, torch, _len_torch_function_stack, _pop_mode, _push_mode, TorchFunctionMode, context_decorator


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `torch`
- `torch._C`: _len_torch_function_stack
- `torch.overrides`: _pop_mode, _push_mode, TorchFunctionMode
- `torch.utils._contextlib`: context_decorator


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Context Manager**: Implements context manager protocol


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

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

Files in the same folder (`torch/utils`):

- [`_zip.py_docs.md`](./_zip.py_docs.md)
- [`weak.py_docs.md`](./weak.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_cpp_embed_headers.py_docs.md`](./_cpp_embed_headers.py_docs.md)
- [`_cpp_extension_versioner.py_docs.md`](./_cpp_extension_versioner.py_docs.md)
- [`module_tracker.py_docs.md`](./module_tracker.py_docs.md)
- [`hooks.py_docs.md`](./hooks.py_docs.md)
- [`_content_store.py_docs.md`](./_content_store.py_docs.md)
- [`_triton.py_docs.md`](./_triton.py_docs.md)
- [`file_baton.py_docs.md`](./file_baton.py_docs.md)


## Cross-References

- **File Documentation**: `_device.py_docs.md`
- **Keyword Index**: `_device.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/utils`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Context Manager**: Implements context manager protocol


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/torch/utils`):

- [`show_pickle.py_docs.md_docs.md`](./show_pickle.py_docs.md_docs.md)
- [`file_baton.py_docs.md_docs.md`](./file_baton.py_docs.md_docs.md)
- [`_filelock.py_kw.md_docs.md`](./_filelock.py_kw.md_docs.md)
- [`_config_module.py_docs.md_docs.md`](./_config_module.py_docs.md_docs.md)
- [`cpp_extension.py_docs.md_docs.md`](./cpp_extension.py_docs.md_docs.md)
- [`checkpoint.py_docs.md_docs.md`](./checkpoint.py_docs.md_docs.md)
- [`module_tracker.py_kw.md_docs.md`](./module_tracker.py_kw.md_docs.md)
- [`dlpack.py_docs.md_docs.md`](./dlpack.py_docs.md_docs.md)
- [`_import_utils.py_kw.md_docs.md`](./_import_utils.py_kw.md_docs.md)
- [`_traceback.py_kw.md_docs.md`](./_traceback.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `_device.py_docs.md_docs.md`
- **Keyword Index**: `_device.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
