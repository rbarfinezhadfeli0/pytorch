# Documentation: `docs/test/lazy/test_extract_compiled_graph.py_docs.md`

## File Metadata

- **Path**: `docs/test/lazy/test_extract_compiled_graph.py_docs.md`
- **Size**: 9,681 bytes (9.45 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/lazy/test_extract_compiled_graph.py`

## File Metadata

- **Path**: `test/lazy/test_extract_compiled_graph.py`
- **Size**: 6,185 bytes (6.04 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]

import unittest

from torch._lazy.ts_backend import init as init_ts_backend


init_ts_backend()
import copy
import dis
import inspect
import re
from contextlib import contextmanager

import torch
from torch import fx, nn
from torch._lazy import config
from torch._lazy.extract_compiled_graph import extract_compiled_graph


class ModuleConstScale(nn.Module):
    def forward(self, a):
        return a * 2


class ModuleSub(nn.Module):
    def forward(self, a, b):
        return a - b


class ModuleAddcmul(nn.Module):
    """
    addcmul function takes a at::Scalar which results in a special TSData containing a Scalar rather than a Tensor.
    """

    def forward(self, a, b, c):
        return torch.addcmul(a, b, c, value=5)


class ModuleReturnMulti(nn.Module):
    def forward(self, a, b):
        return (b + 1, a - 1)


# The default fx tracer will convert torch.randn to a constant.. We may need
# a custom tracer.
# class ModuleEagerTensor(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#
#     def forward(self, a):
#         b = torch.randn(2, 3, device="cpu") # eager device
#         return a + b

# The module was planned to cover the case that a Fx graph return an eager
# tensor on the default device. It's harder than ModuleEagerTensor because
# we can not just override the device argument to Lazy since there is no
# explicit device argument.
#
# Unfortunately, the default fx tracer convert the return value of the forward
# method to a constant.. Comment out for now
# class ModuleReturnEagerTensorOnDefaultDevice(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#
#     def forward(self):
#         return torch.tensor((2, 3), dtype=torch.float32)


class ModuleReturnDupTensor(nn.Module):
    """
    Handle the corner case that the same tensor appears multiple times in the
    returned tuple. torchbench like drq will hit this corner case when running
    thru torchdynamo..
    """

    def forward(self, a, b):
        c = a + b
        return a - b, c, a + 1, c


class ModuleInplaceUpdate(nn.Module):
    def forward(self, a, b):
        a.sub_(b)
        return b - 1, b + 1


@contextmanager
def force_fallback_ctx_mgr(fallback_op):
    oldconfig = config.get_force_fallback()
    config.set_force_fallback(fallback_op)
    try:
        yield None
    finally:
        config.set_force_fallback(oldconfig)


@contextmanager
def nop_ctx_mgr():
    try:
        yield None
    finally:
        pass


def gen_rand_args(mod):
    args = []
    for _ in range(len(inspect.signature(mod.forward).parameters)):
        args.append(torch.randn(2, 3))
    return args


def allclose(expected, actual):
    def unwrap(cont):
        if isinstance(cont, (list, tuple)) and len(cont) == 1:
            return cont[0]
        return cont

    expected = unwrap(expected)
    actual = unwrap(actual)

    if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
        return torch.allclose(expected, actual)
    elif isinstance(expected, (tuple, list)) and isinstance(actual, (tuple, list)):
        return len(expected) == len(actual) and all(
            torch.allclose(a, b) for a, b in zip(expected, actual)
        )
    else:
        raise RuntimeError("Unexpected types")


def verify_reusing_compiled_graph(mod, exception_msg_pattern, ncase=10):
    args = gen_rand_args(mod)
    mod(*args)

    dis.dis(mod.forward)

    try:
        optimized_mod = extract_compiled_graph(fx.symbolic_trace(mod), args)
    except RuntimeError as e:
        if exception_msg_pattern is None:
            raise e  # reraise the exception
        exception_message = str(e)
        if not re.search(exception_msg_pattern, exception_message):
            raise RuntimeError(
                f"Exception message does not match the required pattern: {exception_message}"
            ) from e
        else:
            # We are done for the test case that expects an exception
            return

    if exception_msg_pattern is not None:
        raise RuntimeError(
            f"Expect an exception matching pattern {exception_msg_pattern}"
        )
    print("return value of optimized_mod", optimized_mod(*args))

    # check correctness
    failed_index = []
    for i in range(ncase):
        rand_args = gen_rand_args(mod)
        rand_args_copy = copy.deepcopy(rand_args)
        expected = mod(*rand_args)
        actual = optimized_mod(*rand_args_copy)

        if not allclose(expected, actual):
            print(f"Incorrect results. expected {expected}, actual {actual}")
            failed_index.append(i)
            continue

        # make sure arguments match after calling the model forward method to handle inplace
        # updates.
        if not allclose(rand_args, rand_args_copy):
            print(
                f"Incorrect updated arguments. expected {rand_args}, actual {rand_args_copy}"
            )
            failed_index.append(i)
            continue

    if len(failed_index) > 0:
        raise RuntimeError(f"Failed {len(failed_index)}/{ncase} cases")


def maketest(module_cls, exception_msg_pattern=None, ctxmgr=None):
    def wrapper(self):
        nonlocal ctxmgr
        if not ctxmgr:
            ctxmgr = nop_ctx_mgr()
        with ctxmgr:
            verify_reusing_compiled_graph(module_cls(), exception_msg_pattern)

    return wrapper


class OptimizeTest(unittest.TestCase):
    test_sub = maketest(ModuleSub)
    # Same as test_sub but force aten::sub to fallback
    # We expect an exception caught because of LTC fallback.
    test_ltc_fallback = maketest(
        ModuleSub,
        exception_msg_pattern="fallback.*aten::sub",
        ctxmgr=force_fallback_ctx_mgr("aten::sub"),
    )
    test_const_scale = maketest(ModuleConstScale)
    test_addcmul = maketest(ModuleAddcmul)
    test_return_multi = maketest(ModuleReturnMulti)
    test_return_dup_tensor = maketest(ModuleReturnDupTensor)
    test_inplace_update = maketest(ModuleInplaceUpdate)


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )

```



## High-Level Overview

"""    addcmul function takes a at::Scalar which results in a special TSData containing a Scalar rather than a Tensor.

This Python file contains 9 class(es) and 18 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ModuleConstScale`, `ModuleSub`, `ModuleAddcmul`, `ModuleReturnMulti`, `ModuleEagerTensor`, `ModuleReturnEagerTensorOnDefaultDevice`, `ModuleReturnDupTensor`, `ModuleInplaceUpdate`, `OptimizeTest`

**Functions defined**: `forward`, `forward`, `forward`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `forward`, `forward`, `force_fallback_ctx_mgr`, `nop_ctx_mgr`, `gen_rand_args`, `allclose`, `unwrap`, `verify_reusing_compiled_graph`, `maketest`, `wrapper`

**Key imports**: unittest, init as init_ts_backend, copy, dis, inspect, re, contextmanager, torch, fx, nn, config


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/lazy`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`
- `torch._lazy.ts_backend`: init as init_ts_backend
- `copy`
- `dis`
- `inspect`
- `re`
- `contextlib`: contextmanager
- `torch`
- `torch._lazy`: config
- `torch._lazy.extract_compiled_graph`: extract_compiled_graph


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

This is a test file. Run it with:

```bash
python test/lazy/test_extract_compiled_graph.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/lazy`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_ts_opinfo.py_docs.md`](./test_ts_opinfo.py_docs.md)
- [`test_meta_kernel.py_docs.md`](./test_meta_kernel.py_docs.md)
- [`test_functionalization.py_docs.md`](./test_functionalization.py_docs.md)
- [`test_generator.py_docs.md`](./test_generator.py_docs.md)
- [`test_bindings.py_docs.md`](./test_bindings.py_docs.md)
- [`test_reuse_ir.py_docs.md`](./test_reuse_ir.py_docs.md)
- [`test_step_closures.py_docs.md`](./test_step_closures.py_docs.md)
- [`test_debug_util.py_docs.md`](./test_debug_util.py_docs.md)


## Cross-References

- **File Documentation**: `test_extract_compiled_graph.py_docs.md`
- **Keyword Index**: `test_extract_compiled_graph.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/lazy`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/lazy`, which is part of the **testing infrastructure**.



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

This is a test file. Run it with:

```bash
python docs/test/lazy/test_extract_compiled_graph.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/lazy`):

- [`test_reuse_ir.py_kw.md_docs.md`](./test_reuse_ir.py_kw.md_docs.md)
- [`test_extract_compiled_graph.py_kw.md_docs.md`](./test_extract_compiled_graph.py_kw.md_docs.md)
- [`test_reuse_ir.py_docs.md_docs.md`](./test_reuse_ir.py_docs.md_docs.md)
- [`test_step_closures.py_kw.md_docs.md`](./test_step_closures.py_kw.md_docs.md)
- [`test_step_closures.py_docs.md_docs.md`](./test_step_closures.py_docs.md_docs.md)
- [`test_ts_opinfo.py_kw.md_docs.md`](./test_ts_opinfo.py_kw.md_docs.md)
- [`test_meta_kernel.py_kw.md_docs.md`](./test_meta_kernel.py_kw.md_docs.md)
- [`test_bindings.py_kw.md_docs.md`](./test_bindings.py_kw.md_docs.md)
- [`test_ts_opinfo.py_docs.md_docs.md`](./test_ts_opinfo.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_extract_compiled_graph.py_docs.md_docs.md`
- **Keyword Index**: `test_extract_compiled_graph.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
