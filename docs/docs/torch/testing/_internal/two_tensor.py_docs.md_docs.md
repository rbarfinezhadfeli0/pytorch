# Documentation: `docs/torch/testing/_internal/two_tensor.py_docs.md`

## File Metadata

- **Path**: `docs/torch/testing/_internal/two_tensor.py_docs.md`
- **Size**: 6,510 bytes (6.36 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/testing/_internal/two_tensor.py`

## File Metadata

- **Path**: `torch/testing/_internal/two_tensor.py`
- **Size**: 3,677 bytes (3.59 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```python
# mypy: ignore-errors

import torch
import torch.utils._pytree as pytree
from torch._export.wrappers import mark_subclass_constructor_exportable_experimental
from torch.utils._python_dispatch import return_and_correct_aliasing


# A simple tensor subclass that holds two tensors internally, and runs every op on both tensors.
class TwoTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, a, b, outer_size=None, outer_stride=None, *, requires_grad=None):
        if outer_size is None:
            outer_size = a.size()
        if outer_stride is None:
            outer_stride = a.stride()

        assert (
            a.device == b.device
            and a.layout == b.layout
            and a.requires_grad == b.requires_grad
            and a.dtype == b.dtype
        )
        # I guess it would be more accurate to represent the shape as torch.cat(a, b).shape
        shape = outer_size
        kwargs = {}
        kwargs["strides"] = outer_stride
        kwargs["storage_offset"] = a.storage_offset()
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = requires_grad or a.requires_grad
        kwargs["dtype"] = a.dtype
        out = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

        assert a.shape == b.shape
        assert a.stride() == b.stride()
        assert a.storage_offset() == b.storage_offset()
        return out

    @torch._disable_dynamo
    @mark_subclass_constructor_exportable_experimental
    def __init__(self, a, b, outer_size=None, outer_stride=None, *, requires_grad=None):
        self.a = a
        self.b = b

    def __repr__(self):
        a_repr = repr(self.a)
        b_repr = repr(self.b)
        return f"TwoTensor({a_repr}, {b_repr})"

    def __tensor_flatten__(self):
        return ["a", "b"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert meta is None
        a, b = inner_tensors["a"], inner_tensors["b"]
        if type(a) is torch.Tensor:
            assert outer_size is not None
            assert outer_stride is not None
        return TwoTensor(a, b, outer_size, outer_stride)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        args_a = pytree.tree_map_only(TwoTensor, lambda x: x.a, args)
        args_b = pytree.tree_map_only(TwoTensor, lambda x: x.b, args)

        kwargs_a = pytree.tree_map_only(TwoTensor, lambda x: x.a, kwargs)
        kwargs_b = pytree.tree_map_only(TwoTensor, lambda x: x.b, kwargs)

        out_a = func(*args_a, **kwargs_a)
        out_b = func(*args_b, **kwargs_b)
        out_a_flat, spec = pytree.tree_flatten(out_a)
        out_b_flat = pytree.tree_leaves(out_b)
        # for aten ops that return non-tensors, just assume that
        # our two inner tensors return the same value
        out_flat = [
            cls(o_a, o_b) if isinstance(o_a, torch.Tensor) else o_a
            for o_a, o_b in zip(out_a_flat, out_b_flat, strict=True)
        ]
        out = pytree.tree_unflatten(out_flat, spec)
        from torch._higher_order_ops.cond import cond_op

        if func is cond_op:
            return out
        else:
            return return_and_correct_aliasing(func, args, kwargs, out)

    def get_elem_a(self):
        return self.a


class TwoTensorMode(torch.utils._python_dispatch.TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **kwargs)
        if torch._subclasses.fake_tensor._is_tensor_constructor(func):
            out = TwoTensor(out, out.clone())
        return out

```



## High-Level Overview


This Python file contains 3 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TwoTensor`, `TwoTensorMode`

**Functions defined**: `__new__`, `__init__`, `__repr__`, `__tensor_flatten__`, `__tensor_unflatten__`, `__torch_dispatch__`, `get_elem_a`, `__torch_dispatch__`

**Key imports**: torch, torch.utils._pytree as pytree, mark_subclass_constructor_exportable_experimental, return_and_correct_aliasing, cond_op


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/testing/_internal`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.utils._pytree as pytree`
- `torch._export.wrappers`: mark_subclass_constructor_exportable_experimental
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
python torch/testing/_internal/two_tensor.py
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
- [`autocast_test_lists.py_docs.md`](./autocast_test_lists.py_docs.md)
- [`hypothesis_utils.py_docs.md`](./hypothesis_utils.py_docs.md)
- [`common_mkldnn.py_docs.md`](./common_mkldnn.py_docs.md)


## Cross-References

- **File Documentation**: `two_tensor.py_docs.md`
- **Keyword Index**: `two_tensor.py_kw.md`
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
python docs/torch/testing/_internal/two_tensor.py_docs.md
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

- **File Documentation**: `two_tensor.py_docs.md_docs.md`
- **Keyword Index**: `two_tensor.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
