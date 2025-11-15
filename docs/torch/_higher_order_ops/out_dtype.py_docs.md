# Documentation: `torch/_higher_order_ops/out_dtype.py`

## File Metadata

- **Path**: `torch/_higher_order_ops/out_dtype.py`
- **Size**: 5,566 bytes (5.44 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._prims_common import elementwise_dtypes, ELEMENTWISE_TYPE_PROMOTION_KIND
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    maybe_handle_decomp,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


# TODO to figure out a more generic approach
ALLOWABLE_OPS = [
    torch.ops.aten.linear.default,
    torch.ops.aten.mm.default,
    torch.ops.aten.conv2d.default,
    torch.ops.aten.convolution.default,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.mul.Scalar,
    torch.ops.aten.div.Tensor,
    torch.ops.aten.div.Scalar,
]


class OutDtypeOperator(HigherOrderOperator):
    """
    The out_dtype operator takes an existing ATen functional operator, an
    `out_dtype` argument, and arguments to the original operator, and executes
    the original operator and returns a Tensor with the `out_dtype` precision.
    This operator does not mandate a compute precision so it allows the
    representation to not be opinionated about the exact implementation.

    The general implementation for all operators will be the following:
        1. Promote inputs dtypes based on default PyTorch dtype promotion rules,
            using the dtypes of all input Tensors/Scalars and the `out_dtype`
            arugument.
        2. Execute the operator
        3. Cast the output to `out_dtype`
    """

    def __init__(self) -> None:
        super().__init__("out_dtype")

    def __call__(self, op, output_dtype, *args):
        if not isinstance(op, torch._ops.OpOverload):
            raise ValueError("out_dtype's first argument must be an OpOverload")
        if op._schema.is_mutable:
            raise ValueError(
                "out_dtype's first argument needs to be a functional operator"
            )
        if not (
            len(op._schema.returns) == 1
            and isinstance(op._schema.returns[0].type, torch.TensorType)
        ):
            raise ValueError(
                "out_dtype's can only apply to ops that return a single tensor"
                f"Instead got {[r.type for r in op._schema.returns]}"
            )

        if op not in ALLOWABLE_OPS:
            raise ValueError(
                f"out_dtype only allows the following operators: {ALLOWABLE_OPS}."
            )

        res = super().__call__(op, output_dtype, *args)

        return res


out_dtype = OutDtypeOperator()


def trace_out_dtype(proxy_mode, func_overload, op, output_dtype, *args):
    # NB: Long-term we should put the decomposition logic into
    # ProxyTorchDispatchMode so that people do not need to call maybe_handle_decomp
    # in all HigherOrderOp proxy implementations.
    r = maybe_handle_decomp(proxy_mode, func_overload, (op, output_dtype, *args), {})
    if r is not NotImplemented:
        return r

    with disable_proxy_modes_tracing():
        # This is a simplified implementation of this operator just for tracing.
        # Actual implementation may also first promote the arguments
        out = op(*args).to(dtype=output_dtype)

    node_args = (op, output_dtype, *args)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="out_dtype"
    )
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@out_dtype.py_impl(DispatchKey.CompositeExplicitAutograd)
def out_dtype_dense(op: torch._ops.OpOverload, output_dtype: torch.dtype, *args):
    if is_int_mm(op, output_dtype, args):
        return torch._int_mm(*args)
    return out_dtype_fallback(op, output_dtype, *args)


def is_int_mm(op, output_dtype, args):
    return (
        op is torch.ops.aten.mm.default
        and output_dtype == torch.int32
        and len(args) == 2
        and args[0].dtype == torch.int8
        and args[1].dtype == torch.int8
        and (args[0].is_cuda or args[0].is_xpu)
        and (args[1].is_cuda or args[1].is_xpu)
    )


def out_dtype_fallback(op, output_dtype, *args):
    flat_inputs = pytree.arg_tree_leaves(*args) + [torch.ones(1, dtype=output_dtype)]
    promote_dtype: torch.dtype = elementwise_dtypes(
        *flat_inputs,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    )[0]

    casted_args = pytree.tree_map_only(
        torch.Tensor, lambda arg: arg.to(dtype=promote_dtype), args
    )
    res = op(*casted_args).to(dtype=output_dtype)
    return res


out_dtype.py_autograd_impl(autograd_not_implemented(out_dtype, deferred_error=True))


@out_dtype.py_impl(ProxyTorchDispatchMode)
def out_dtype_proxy(
    mode: ProxyTorchDispatchMode,
    op: torch._ops.OpOverload,
    output_dtype: torch.dtype,
    *args,
):
    return trace_out_dtype(mode, out_dtype, op, output_dtype, *args)


@out_dtype.py_impl(FakeTensorMode)
def out_dtype_fake_tensor_mode(
    mode: FakeTensorMode,
    op: torch._ops.OpOverload,
    output_dtype: torch.dtype,
    *args,
):
    with mode:
        return out_dtype_dense(op, output_dtype, *args)


@out_dtype.py_functionalize_impl
def out_dtype_func(ctx, op, output_dtype, *args):
    unwrapped_args = tuple(ctx.unwrap_tensors(arg) for arg in args)

    with ctx.redispatch_to_next():
        res = out_dtype(op, output_dtype, *unwrapped_args)
    return ctx.wrap_tensors(res)

```



## High-Level Overview

"""    The out_dtype operator takes an existing ATen functional operator, an    `out_dtype` argument, and arguments to the original operator, and executes    the original operator and returns a Tensor with the `out_dtype` precision.    This operator does not mandate a compute precision so it allows the    representation to not be opinionated about the exact implementation.    The general implementation for all operators will be the following:        1. Promote inputs dtypes based on default PyTorch dtype promotion rules,            using the dtypes of all input Tensors/Scalars and the `out_dtype`            arugument.        2. Execute the operator        3. Cast the output to `out_dtype`

This Python file contains 1 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `OutDtypeOperator`

**Functions defined**: `__init__`, `__call__`, `trace_out_dtype`, `out_dtype_dense`, `is_int_mm`, `out_dtype_fallback`, `out_dtype_proxy`, `out_dtype_fake_tensor_mode`, `out_dtype_func`

**Key imports**: torch, torch.utils._pytree as pytree, DispatchKey, autograd_not_implemented, HigherOrderOperator, elementwise_dtypes, ELEMENTWISE_TYPE_PROMOTION_KIND, FakeTensorMode


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_higher_order_ops`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.utils._pytree as pytree`
- `torch._C`: DispatchKey
- `torch._higher_order_ops.utils`: autograd_not_implemented
- `torch._ops`: HigherOrderOperator
- `torch._prims_common`: elementwise_dtypes, ELEMENTWISE_TYPE_PROMOTION_KIND
- `torch._subclasses.fake_tensor`: FakeTensorMode


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`torch/_higher_order_ops`):

- [`associative_scan.py_docs.md`](./associative_scan.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`effects.py_docs.md`](./effects.py_docs.md)
- [`foreach_map.py_docs.md`](./foreach_map.py_docs.md)
- [`strict_mode.py_docs.md`](./strict_mode.py_docs.md)
- [`torchbind.py_docs.md`](./torchbind.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`run_const_graph.py_docs.md`](./run_const_graph.py_docs.md)
- [`_invoke_quant.py_docs.md`](./_invoke_quant.py_docs.md)
- [`wrap.py_docs.md`](./wrap.py_docs.md)


## Cross-References

- **File Documentation**: `out_dtype.py_docs.md`
- **Keyword Index**: `out_dtype.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
