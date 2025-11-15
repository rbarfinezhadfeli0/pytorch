# Documentation: common_mkldnn.py

## File Metadata
- **Path**: `torch/testing/_internal/common_mkldnn.py`
- **Size**: 3899 bytes
- **Lines**: 113
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# mypy: ignore-errors

import contextlib
import functools
import inspect

import torch


def bf32_is_not_fp32():
    if not torch.backends.mkldnn.is_available():
        return False
    if not torch.ops.mkldnn._is_mkldnn_bf16_supported():
        return False
    return True


def tf32_is_not_fp32():
    if not torch.backends.mkldnn.is_available():
        return False
    if not torch._C._cpu._is_amx_fp16_supported():
        return False
    return True


@contextlib.contextmanager
def reduced_f32_off():
    old_matmul_precision = torch.backends.mkldnn.matmul.fp32_precision
    old_conv_precision = torch.backends.mkldnn.conv.fp32_precision
    try:
        torch.backends.mkldnn.matmul.fp32_precision = "ieee"
        torch.backends.mkldnn.conv.fp32_precision = "ieee"
        yield
    finally:
        torch.backends.mkldnn.matmul.fp32_precision = old_matmul_precision
        torch.backends.mkldnn.conv.fp32_precision = old_conv_precision


@contextlib.contextmanager
def bf32_on(self, bf32_precision=1e-2):
    old_matmul_precision = torch.backends.mkldnn.matmul.fp32_precision
    old_conv_precision = torch.backends.mkldnn.conv.fp32_precision
    old_precision = self.precision
    try:
        torch.backends.mkldnn.matmul.fp32_precision = "bf16"
        torch.backends.mkldnn.conv.fp32_precision = "bf16"
        self.precision = bf32_precision
        yield
    finally:
        torch.backends.mkldnn.matmul.fp32_precision = old_matmul_precision
        torch.backends.mkldnn.conv.fp32_precision = old_conv_precision
        self.precision = old_precision


@contextlib.contextmanager
def tf32_on(self, tf32_precision=1e-5):
    old_matmul_precision = torch.backends.mkldnn.matmul.fp32_precision
    old_conv_precision = torch.backends.mkldnn.conv.fp32_precision
    old_precision = self.precision
    try:
        torch.backends.mkldnn.matmul.fp32_precision = "tf32"
        torch.backends.mkldnn.conv.fp32_precision = "tf32"
        self.precision = tf32_precision
        yield
    finally:
        torch.backends.mkldnn.matmul.fp32_precision = old_matmul_precision
        torch.backends.mkldnn.conv.fp32_precision = old_conv_precision
        self.precision = old_precision


# This is a wrapper that wraps a test to run this test three times, one with
# reduced_f32 OFF, the others with reduced_f32 ON (including bf32 ON and tf32
# ON). When running with reduced_f32 ON, it will use reduced precision (bf16/
# tf32) as specified by the argument.
def reduced_f32_on_and_off(bf32_precision=1e-2, tf32_precision=1e-5):
    def with_reduced_f32_disabled(self, function_call):
        with reduced_f32_off():
            function_call()

    def with_bf32_enabled(self, function_call):
        with bf32_on(self, bf32_precision):
            function_call()

    def with_tf32_enabled(self, function_call):
        with tf32_on(self, tf32_precision):
            function_call()

    def wrapper(f):
        params = inspect.signature(f).parameters
        arg_names = tuple(params.keys())

        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            kwargs.update(zip(arg_names, args, strict=False))
            cond = True
            if "device" in kwargs:
                cond = cond and (torch.device(kwargs["device"]).type == "cpu")
            if "dtype" in kwargs:
                cond = cond and (kwargs["dtype"] == torch.float)
            bf32_cond = cond and bf32_is_not_fp32()
            tf32_cond = cond and tf32_is_not_fp32()
            if bf32_cond or tf32_cond:
                with_reduced_f32_disabled(kwargs["self"], lambda: f(**kwargs))
                if bf32_cond:
                    with_bf32_enabled(kwargs["self"], lambda: f(**kwargs))
                if tf32_cond:
                    with_tf32_enabled(kwargs["self"], lambda: f(**kwargs))
            else:
                f(**kwargs)

        return wrapped

    return wrapper

```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Functions
This file defines 11 function(s): bf32_is_not_fp32, tf32_is_not_fp32, reduced_f32_off, bf32_on, tf32_on, reduced_f32_on_and_off, with_reduced_f32_disabled, with_bf32_enabled, with_tf32_enabled, wrapper, wrapped


## Key Components

The file contains 277 words across 113 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 3899 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
