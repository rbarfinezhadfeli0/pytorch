# Documentation: `docs/torch/_inductor/shape_propagation.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/shape_propagation.py_docs.md`
- **Size**: 7,648 bytes (7.47 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/shape_propagation.py`

## File Metadata

- **Path**: `torch/_inductor/shape_propagation.py`
- **Size**: 4,565 bytes (4.46 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import functools
from collections.abc import Callable, Sequence
from typing import Optional, Protocol, Union

import sympy

import torch

from .virtualized import OpsValue, V


BlockShapeType = Optional[Sequence[Union[int, str]]]


class ShapeVar(Protocol):
    @property
    def shape(self) -> BlockShapeType: ...


ShapeArg = Union[ShapeVar, torch.types.Number, str, OpsValue, torch.dtype]

# Inputs need to be cacheable (e.g., not a CSEVar) in order for the cache to be effective
# So first decompose CSEVars -> tuple before calling this


@functools.lru_cache(None)
def get_broadcasted_shape(a: BlockShapeType, b: BlockShapeType) -> BlockShapeType:
    assert isinstance(a, Sequence)
    assert isinstance(b, Sequence)
    if len(a) > len(b):
        return get_broadcasted_shape(a, (*[1] * (len(a) - len(b)), *b))
    elif len(a) < len(b):
        b, a = a, b
        return get_broadcasted_shape(a, (*[1] * (len(a) - len(b)), *b))
    else:

        def _get_broadcasted_dim(
            d1: Union[int, str], d2: Union[int, str]
        ) -> Union[int, str]:
            if str(d1) == "1":
                return d2
            elif str(d2) == "1":
                return d1
            assert str(d1) == str(d2)
            return d1

        return tuple(_get_broadcasted_dim(d1, d2) for d1, d2 in zip(a, b))


def broadcast_shapes_for_args(args: Sequence[ShapeArg]) -> BlockShapeType:
    result_shape: BlockShapeType = None

    for arg in args:
        if hasattr(arg, "shape"):
            shape = arg.shape
            if shape is None:
                return None
            elif result_shape is None:
                result_shape = tuple(shape)
            else:
                result_shape = get_broadcasted_shape(result_shape, tuple(shape))
        elif isinstance(arg, (int, float)):
            if result_shape is None:
                result_shape = ()
        elif isinstance(arg, torch.dtype):
            continue
        else:
            from torch._inductor.loop_body import LoopBody, LoopBodyBlock

            if isinstance(arg, (LoopBodyBlock, LoopBody, OpsValue)):
                # TODO: fix me
                return None
            raise TypeError(f"Unknown type: {type(arg)}")

    return result_shape


class ShapePropagationOpsHandler:
    """
    Propagate shape from args to output
    """

    @staticmethod
    def constant(value: torch.types.Number, dtype: torch.dtype) -> BlockShapeType:
        # See implementation of constant for triton for the reason
        from torch._inductor.codegen.triton import triton_compute_type, TritonKernel

        triton_type = triton_compute_type(dtype)

        if isinstance(V.kernel, TritonKernel) and triton_type != "tl.float32":
            ndim = V.kernel.triton_tensor_ndim()
            return tuple([1] * ndim)
        else:
            return ()

    @staticmethod
    def store_reduction(name: str, index: int, value: ShapeArg) -> None:
        return None

    @staticmethod
    def reduction(
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: str,
        value: Union[ShapeArg, tuple[ShapeArg, ...]],
    ) -> Union[BlockShapeType, tuple[BlockShapeType, ...]]:
        raise NotImplementedError

    @staticmethod
    def store(
        name: str, index: int, value: ShapeArg, mode: Optional[str] = None
    ) -> None:
        return None

    @staticmethod
    def to_dtype(
        value: ShapeVar,
        dtype: torch.dtype,
        src_dtype: Optional[torch.dtype] = None,
        use_compute_types: bool = True,
    ) -> BlockShapeType:
        return value.shape

    @staticmethod
    def dot(a: sympy.Expr, b: sympy.Expr) -> BlockShapeType:
        from torch._inductor.codegen.triton import TritonKernel

        assert isinstance(V.kernel, TritonKernel), "dot supports Triton only"
        return ("YBLOCK", "XBLOCK")

    @staticmethod
    def index_expr(expr: sympy.Expr, dtype: torch.dtype) -> BlockShapeType:
        # shape is implicitly embedded in expr.
        return None

    @staticmethod
    def load_seed(name: str, offset: int) -> BlockShapeType:
        return ()

    @staticmethod
    def indirect_indexing(
        var: ShapeArg,
        size: Union[sympy.Expr, int],
        check: bool = True,
        wrap_neg: bool = True,
    ) -> None:
        return None

    def __getattr__(self, name: str) -> Callable[..., BlockShapeType]:
        return lambda *args, **kwargs: broadcast_shapes_for_args(args)

    @staticmethod
    def device_assert_async(cond: ShapeArg, msg: str) -> None:
        return None

```



## High-Level Overview


This Python file contains 2 class(es) and 15 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ShapeVar`, `ShapePropagationOpsHandler`

**Functions defined**: `shape`, `get_broadcasted_shape`, `_get_broadcasted_dim`, `broadcast_shapes_for_args`, `constant`, `store_reduction`, `reduction`, `store`, `to_dtype`, `dot`, `index_expr`, `load_seed`, `indirect_indexing`, `__getattr__`, `device_assert_async`

**Key imports**: functools, Callable, Sequence, Optional, Protocol, Union, sympy, torch, OpsValue, V, LoopBody, LoopBodyBlock, triton_compute_type, TritonKernel, TritonKernel


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `collections.abc`: Callable, Sequence
- `typing`: Optional, Protocol, Union
- `sympy`
- `torch`
- `.virtualized`: OpsValue, V
- `torch._inductor.loop_body`: LoopBody, LoopBodyBlock
- `torch._inductor.codegen.triton`: triton_compute_type, TritonKernel


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`torch/_inductor`):

- [`freezing_utils.py_docs.md`](./freezing_utils.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`mkldnn_ir.py_docs.md`](./mkldnn_ir.py_docs.md)
- [`async_compile.py_docs.md`](./async_compile.py_docs.md)
- [`invert_expr_analysis.py_docs.md`](./invert_expr_analysis.py_docs.md)
- [`extern_node_serializer.py_docs.md`](./extern_node_serializer.py_docs.md)
- [`loop_body.py_docs.md`](./loop_body.py_docs.md)
- [`debug.py_docs.md`](./debug.py_docs.md)
- [`freezing.py_docs.md`](./freezing.py_docs.md)
- [`optimize_indexing.py_docs.md`](./optimize_indexing.py_docs.md)


## Cross-References

- **File Documentation**: `shape_propagation.py_docs.md`
- **Keyword Index**: `shape_propagation.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/torch/_inductor`):

- [`freezing.py_docs.md_docs.md`](./freezing.py_docs.md_docs.md)
- [`lowering.py_kw.md_docs.md`](./lowering.py_kw.md_docs.md)
- [`quantized_lowerings.py_docs.md_docs.md`](./quantized_lowerings.py_docs.md_docs.md)
- [`select_algorithm.py_docs.md_docs.md`](./select_algorithm.py_docs.md_docs.md)
- [`kernel_inputs.py_kw.md_docs.md`](./kernel_inputs.py_kw.md_docs.md)
- [`compile_fx_ext.py_kw.md_docs.md`](./compile_fx_ext.py_kw.md_docs.md)
- [`extern_node_serializer.py_docs.md_docs.md`](./extern_node_serializer.py_docs.md_docs.md)
- [`mkldnn_lowerings.py_kw.md_docs.md`](./mkldnn_lowerings.py_kw.md_docs.md)
- [`ops_handler.py_docs.md_docs.md`](./ops_handler.py_docs.md_docs.md)
- [`test_operators.py_docs.md_docs.md`](./test_operators.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `shape_propagation.py_docs.md_docs.md`
- **Keyword Index**: `shape_propagation.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
