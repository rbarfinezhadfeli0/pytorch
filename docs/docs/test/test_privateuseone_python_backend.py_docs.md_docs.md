# Documentation: `docs/test/test_privateuseone_python_backend.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_privateuseone_python_backend.py_docs.md`
- **Size**: 7,195 bytes (7.03 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_privateuseone_python_backend.py`

## File Metadata

- **Path**: `test/test_privateuseone_python_backend.py`
- **Size**: 4,093 bytes (4.00 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: PrivateUse1"]
import numpy as np

import torch
import torch._C
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils.backend_registration import _setup_privateuseone_for_python_backend


_setup_privateuseone_for_python_backend("npy")

aten = torch.ops.aten


# NOTE: From https://github.com/albanD/subclass_zoo/blob/main/new_device.py
# but using torch.library instead of `__torch_dispatch__`
class MyDeviceTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, size, dtype, raw_data=None, requires_grad=False):
        # Use a meta Tensor here to be used as the wrapper
        res = torch._C._acc.create_empty_tensor(size, dtype)
        res.__class__ = MyDeviceTensor
        return res

    def __init__(self, size, dtype, raw_data=None, requires_grad=False):
        # Store any provided user raw_data
        self.raw_data = raw_data

    def __repr__(self):
        return "MyDeviceTensor" + str(self.raw_data)

    __str__ = __repr__


def wrap(arr, shape, dtype):
    # hard code float32 for tests
    return MyDeviceTensor(shape, dtype, arr)


def unwrap(arr):
    return arr.raw_data


# Add some ops
@torch.library.impl("aten::add.Tensor", "privateuseone")
def add(t1, t2):
    out = unwrap(t1) + unwrap(t2)
    return wrap(out, out.shape, torch.float32)


@torch.library.impl("aten::mul.Tensor", "privateuseone")
def mul(t1, t2):
    # If unsure what should be the result's properties, you can
    # use the super_fn (can be useful for type promotion)
    out = unwrap(t1) * unwrap(t2)
    return wrap(out, out.shape, torch.float32)


@torch.library.impl("aten::detach", "privateuseone")
def detach(self):
    out = unwrap(self)
    return wrap(out, out.shape, torch.float32)


@torch.library.impl("aten::empty_strided", "privateuseone")
def empty_strided(
    size, stride, *, dtype=None, layout=None, device=None, pin_memory=None
):
    out = np.empty(size)
    return wrap(out, out.shape, torch.float32)


@torch.library.impl("aten::_copy_from", "privateuseone")
def _copy_from(a, b):
    if a.device.type == "npy":
        npy_data = unwrap(a)
    else:
        npy_data = a.numpy()
    b.raw_data = npy_data


@torch.library.impl("aten::view", "privateuseone")
def _view(a, b):
    ans = unwrap(a)
    return wrap(ans, a.shape, a.dtype)


@torch.library.impl("aten::empty.memory_format", "privateuseone")
def empty_memory_format(
    size, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    ans = np.empty(size)
    return wrap(ans, ans.shape, torch.float32)


@torch.library.impl("aten::sum", "privateuseone")
def sum_int_list(*args, **kwargs):
    ans = unwrap(args[0]).sum()
    return wrap(ans, ans.shape, torch.float32)


@torch.library.impl("aten::ones_like", "privateuseone")
def ones_like(
    self, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    ans = np.ones_like(unwrap(self))
    return wrap(ans, ans.shape, torch.float32)


@torch.library.impl("aten::expand", "privateuseone")
def expand(self, size, *, implicit=False):
    ans = np.broadcast_to(self.raw_data, size)
    return wrap(ans, ans.shape, torch.float32)


@torch.library.impl("aten::as_strided", "privateuseone")
def as_strided(self, size, stride, storage_offset=None):
    ans = np.lib.stride_tricks.as_strided(self.raw_data, size, stride)
    return wrap(ans, ans.shape, torch.float32)


class PrivateUse1BackendTest(TestCase):
    @classmethod
    def setupClass(cls):
        pass

    def test_accessing_is_pinned(self):
        a_cpu = torch.randn((2, 2))
        # Assert this don't throw:
        _ = a_cpu.is_pinned()

    def test_backend_simple(self):
        a_cpu = torch.randn((2, 2))
        b_cpu = torch.randn((2, 2))
        # Assert this don't throw:
        a = a_cpu.to("privateuseone")
        b = b_cpu.to("privateuseone")

        a.requires_grad = True
        b.requires_grad = True
        c = (a + b).sum()
        c.backward()
        self.assertTrue(np.allclose(a.grad.raw_data, np.ones((2, 2))))


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 19 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MyDeviceTensor`, `PrivateUse1BackendTest`

**Functions defined**: `__new__`, `__init__`, `__repr__`, `wrap`, `unwrap`, `add`, `mul`, `detach`, `empty_strided`, `_copy_from`, `_view`, `empty_memory_format`, `sum_int_list`, `ones_like`, `expand`, `as_strided`, `setupClass`, `test_accessing_is_pinned`, `test_backend_simple`

**Key imports**: numpy as np, torch, torch._C, run_tests, TestCase, _setup_privateuseone_for_python_backend


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `numpy as np`
- `torch`
- `torch._C`
- `torch.testing._internal.common_utils`: run_tests, TestCase
- `torch.utils.backend_registration`: _setup_privateuseone_for_python_backend


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
python test/test_privateuseone_python_backend.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test`):

- [`test_file_check.py_docs.md`](./test_file_check.py_docs.md)
- [`test_jit_simple.py_docs.md`](./test_jit_simple.py_docs.md)
- [`test_mkldnn.py_docs.md`](./test_mkldnn.py_docs.md)
- [`test_expanded_weights.py_docs.md`](./test_expanded_weights.py_docs.md)
- [`test_overrides.py_docs.md`](./test_overrides.py_docs.md)
- [`test_decomp.py_docs.md`](./test_decomp.py_docs.md)
- [`test_show_pickle.py_docs.md`](./test_show_pickle.py_docs.md)
- [`test_utils_config_module.py_docs.md`](./test_utils_config_module.py_docs.md)
- [`test_mobile_optimizer.py_docs.md`](./test_mobile_optimizer.py_docs.md)
- [`test_type_info.py_docs.md`](./test_type_info.py_docs.md)


## Cross-References

- **File Documentation**: `test_privateuseone_python_backend.py_docs.md`
- **Keyword Index**: `test_privateuseone_python_backend.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test`, which is part of the **testing infrastructure**.



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

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/test_privateuseone_python_backend.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test`):

- [`test_ops.py_docs.md_docs.md`](./test_ops.py_docs.md_docs.md)
- [`test_tensorexpr.py_docs.md_docs.md`](./test_tensorexpr.py_docs.md_docs.md)
- [`pytest_shard_custom.py_docs.md_docs.md`](./pytest_shard_custom.py_docs.md_docs.md)
- [`test_weak.py_kw.md_docs.md`](./test_weak.py_kw.md_docs.md)
- [`test_view_ops.py_kw.md_docs.md`](./test_view_ops.py_kw.md_docs.md)
- [`test_varlen_attention.py_kw.md_docs.md`](./test_varlen_attention.py_kw.md_docs.md)
- [`test_namedtensor.py_docs.md_docs.md`](./test_namedtensor.py_docs.md_docs.md)
- [`test_binary_ufuncs.py_docs.md_docs.md`](./test_binary_ufuncs.py_docs.md_docs.md)
- [`test_ops_gradients.py_kw.md_docs.md`](./test_ops_gradients.py_kw.md_docs.md)
- [`test_torchfuzz_repros.py_docs.md_docs.md`](./test_torchfuzz_repros.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_privateuseone_python_backend.py_docs.md_docs.md`
- **Keyword Index**: `test_privateuseone_python_backend.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
