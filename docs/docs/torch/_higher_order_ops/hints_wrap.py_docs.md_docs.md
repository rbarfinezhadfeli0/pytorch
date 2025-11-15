# Documentation: `docs/torch/_higher_order_ops/hints_wrap.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_higher_order_ops/hints_wrap.py_docs.md`
- **Size**: 8,221 bytes (8.03 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_higher_order_ops/hints_wrap.py`

## File Metadata

- **Path**: `torch/_higher_order_ops/hints_wrap.py`
- **Size**: 4,818 bytes (4.71 KB)
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
from torch._higher_order_ops.utils import (
    autograd_not_implemented,
    reenter_make_fx,
    unique_graph_id,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree


# used for wrapping a function/op with context hints
class HintsWrapper(HigherOrderOperator):
    def __init__(self):
        super().__init__("hints_wrapper")

    def __call__(self, body_fn, args, kwargs, hints):
        r"""
        Call implementation of hints_wrapper

        Args:
            body_fn (Callable): A callable function that is within the scope
             that is being traced.

            args (Tuple of torch.Tensor/int/float/bool): A tuple of inputs to
             body_fn.

            kwargs (dict): Keyword argument to the body_fn.

            hints (dict): A dict of context hints which could be passed to
             backend compiler.
        """
        if not isinstance(args, tuple):
            raise RuntimeError(f"args must be a tuple, got {type(args)}")

        if not all(isinstance(t, (torch.Tensor, int, float, bool)) for t in args):
            raise RuntimeError(
                f"args must be a tuple of tensors, ints, floats, or bools, got {args}"
            )

        if not isinstance(kwargs, dict):
            raise RuntimeError(f"kwargs must be a dict, got {type(kwargs)}")

        if len(kwargs) > 0:
            raise RuntimeError(
                f"kwargs except for hints are not supported, got {kwargs}"
            )

        if not isinstance(hints, dict):
            raise RuntimeError(f"hints must be a dict, got {type(hints)}")

        for k, v in hints.items():
            if not isinstance(k, str):
                raise RuntimeError(f"hints key must be a str, got {k}.")

            if not isinstance(v, (int, float, bool, str)):
                raise RuntimeError(
                    "hints must be a dict containing int, float, bool or str "
                    f"value, got value {v} for key {k}."
                )

        return super().__call__(body_fn, args, kwargs, hints)


hints_wrapper = HintsWrapper()


@hints_wrapper.py_impl(DispatchKey.CompositeExplicitAutograd)
def hints_wrapper_dense(body_fn, args, kwargs, hints):
    return body_fn(*args, **kwargs)


hints_wrapper.py_autograd_impl(
    autograd_not_implemented(hints_wrapper, deferred_error=True)
)


@hints_wrapper.py_impl(FakeTensorMode)
def hints_wrapper_fake_tensor_mode(mode, body_func, args, kwargs, hints):
    flat_args = pytree.tree_leaves(args)
    with mode:
        return body_func(*flat_args, **kwargs)


@hints_wrapper.py_functionalize_impl
def hints_wrapper_functionalize(ctx, body_fn, args, kwargs, hints):
    from torch._higher_order_ops.utils import _check_alias_and_mutation

    unwrapped_args = ctx.unwrap_tensors(args)
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)
    unwrapped_hints = ctx.unwrap_tensors(hints)
    with ctx.redispatch_to_next():
        functional_body_fn = ctx.functionalize(body_fn)
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        _check_alias_and_mutation(
            body_fn, unwrapped_args, "hints_wrapper", pre_dispatch
        )

        outputs = hints_wrapper(
            functional_body_fn,
            unwrapped_args,
            unwrapped_kwargs,
            unwrapped_hints,
        )
        return ctx.wrap_tensors(outputs)


def trace_hints_wrapper(proxy_mode, hints_wrapper, body_fn, args, kwargs, hints):
    flat_args = tuple(pytree.tree_leaves(args))
    body_graph = reenter_make_fx(body_fn)(*flat_args, **kwargs)

    _, body_graph_name = unique_graph_id(proxy_mode, prefix="hints_wrapper_body_graph")
    proxy_mode.tracer.root.register_module(body_graph_name, body_graph)

    new_args: tuple = (body_graph, flat_args, {})
    # merge hints into kwargs
    new_kwargs = {}
    new_kwargs["hints"] = hints

    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, new_args)
    proxy_kwargs = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, new_kwargs)

    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", hints_wrapper, proxy_args, proxy_kwargs, name="hints_wrapper"
    )

    out = body_fn(*flat_args, **kwargs)
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@hints_wrapper.py_impl(ProxyTorchDispatchMode)
def inner(proxy_mode, body_fn, args, kwargs, hints):
    if proxy_mode.enable_tracing:
        return trace_hints_wrapper(
            proxy_mode, hints_wrapper, body_fn, args, kwargs, hints
        )
    else:
        return hints_wrapper(body_fn, args, kwargs, hints)

```



## High-Level Overview

r"""        Call implementation of hints_wrapper        Args:            body_fn (Callable): A callable function that is within the scope             that is being traced.            args (Tuple of torch.Tensor/int/float/bool): A tuple of inputs to             body_fn.            kwargs (dict): Keyword argument to the body_fn.            hints (dict): A dict of context hints which could be passed to             backend compiler.

This Python file contains 1 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `HintsWrapper`

**Functions defined**: `__init__`, `__call__`, `hints_wrapper_dense`, `hints_wrapper_fake_tensor_mode`, `hints_wrapper_functionalize`, `trace_hints_wrapper`, `inner`

**Key imports**: torch, torch.utils._pytree as pytree, DispatchKey, HigherOrderOperator, FakeTensorMode, ProxyTorchDispatchMode, track_tensor_tree, _check_alias_and_mutation


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
- `torch._ops`: HigherOrderOperator
- `torch._subclasses.fake_tensor`: FakeTensorMode
- `torch.fx.experimental.proxy_tensor`: ProxyTorchDispatchMode, track_tensor_tree
- `torch._higher_order_ops.utils`: _check_alias_and_mutation


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

- **File Documentation**: `hints_wrap.py_docs.md`
- **Keyword Index**: `hints_wrap.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_higher_order_ops`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_higher_order_ops`, which is part of the **core PyTorch library**.



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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/_higher_order_ops`):

- [`schema.py_docs.md_docs.md`](./schema.py_docs.md_docs.md)
- [`run_const_graph.py_docs.md_docs.md`](./run_const_graph.py_docs.md_docs.md)
- [`effects.py_kw.md_docs.md`](./effects.py_kw.md_docs.md)
- [`partitioner.py_docs.md_docs.md`](./partitioner.py_docs.md_docs.md)
- [`strict_mode.py_docs.md_docs.md`](./strict_mode.py_docs.md_docs.md)
- [`out_dtype.py_kw.md_docs.md`](./out_dtype.py_kw.md_docs.md)
- [`wrap.py_docs.md_docs.md`](./wrap.py_docs.md_docs.md)
- [`while_loop.py_kw.md_docs.md`](./while_loop.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`invoke_subgraph.py_docs.md_docs.md`](./invoke_subgraph.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `hints_wrap.py_docs.md_docs.md`
- **Keyword Index**: `hints_wrap.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
