# Documentation: _device.py

## File Metadata
- **Path**: `torch/utils/_device.py`
- **Size**: 3996 bytes
- **Lines**: 124
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 1 class(es): DeviceContext

### Functions
This file defines 7 function(s): _device_constructors, __init__, __enter__, __exit__, __torch_function__, device_decorator, set_device


## Key Components

The file contains 367 words across 124 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 3996 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
