# Documentation: `torch/testing/_internal/subclasses.py`

## File Metadata

- **Path**: `torch/testing/_internal/subclasses.py`
- **Size**: 2,530 bytes (2.47 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```python
# mypy: ignore-errors
from typing import Any, Optional

import torch
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import is_fake
from torch.testing._internal.two_tensor import TwoTensor
from torch.utils._python_dispatch import return_and_correct_aliasing


class WrapperSubclass(torch.Tensor):
    @staticmethod
    def __new__(cls, a, outer_size=None, outer_stride=None):
        if outer_size is None:
            outer_size = a.size()
        if outer_stride is None:
            outer_stride = a.stride()

        kwargs = {}
        kwargs["strides"] = outer_stride
        kwargs["storage_offset"] = a.storage_offset()
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = a.requires_grad
        kwargs["dtype"] = a.dtype
        out = torch.Tensor._make_wrapper_subclass(cls, outer_size, **kwargs)

        return out

    def __init__(self, a, outer_size=None, outer_stride=None):
        self.a = a

    def __repr__(self):
        return f"WrapperSubclass({repr(self.a)})"

    def __tensor_flatten__(self):
        return ["a"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert meta is None
        a = inner_tensors["a"]
        if is_fake(a):
            assert outer_size is not None
            assert outer_stride is not None
        return WrapperSubclass(a, outer_size, outer_stride)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        args_a = pytree.tree_map_only(WrapperSubclass, lambda x: x.a, args)

        kwargs_a = pytree.tree_map_only(WrapperSubclass, lambda x: x.a, kwargs)

        out_a = func(*args_a, **kwargs_a)
        out_a_flat, spec = pytree.tree_flatten(out_a)
        out_flat = [
            WrapperSubclass(o_a) if isinstance(o_a, torch.Tensor) else o_a
            for o_a in out_a_flat
        ]
        out = pytree.tree_unflatten(out_flat, spec)
        from torch._higher_order_ops.cond import cond_op

        if func is cond_op:
            return out
        else:
            return return_and_correct_aliasing(func, args, kwargs, out)

    def __coerce_same_metadata_as_tangent__(
        self, expected_metadata: Any, expected_type: Optional[type] = None
    ):
        if expected_type is type(self.a):
            return self.a
        elif expected_type is TwoTensor:
            return TwoTensor(self.a, self.a.clone())

        return None

```



## High-Level Overview


This Python file contains 1 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `WrapperSubclass`

**Functions defined**: `__new__`, `__init__`, `__repr__`, `__tensor_flatten__`, `__tensor_unflatten__`, `__torch_dispatch__`, `__coerce_same_metadata_as_tangent__`

**Key imports**: Any, Optional, torch, torch.utils._pytree as pytree, is_fake, TwoTensor, return_and_correct_aliasing, cond_op


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/testing/_internal`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Any, Optional
- `torch`
- `torch.utils._pytree as pytree`
- `torch._subclasses.fake_tensor`: is_fake
- `torch.testing._internal.two_tensor`: TwoTensor
- `torch.utils._python_dispatch`: return_and_correct_aliasing
- `torch._higher_order_ops.cond`: cond_op


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
python torch/testing/_internal/subclasses.py
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
- [`logging_utils.py_docs.md`](./logging_utils.py_docs.md)
- [`two_tensor.py_docs.md`](./two_tensor.py_docs.md)
- [`autocast_test_lists.py_docs.md`](./autocast_test_lists.py_docs.md)
- [`hypothesis_utils.py_docs.md`](./hypothesis_utils.py_docs.md)
- [`common_mkldnn.py_docs.md`](./common_mkldnn.py_docs.md)


## Cross-References

- **File Documentation**: `subclasses.py_docs.md`
- **Keyword Index**: `subclasses.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
