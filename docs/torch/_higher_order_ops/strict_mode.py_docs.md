# Documentation: `torch/_higher_order_ops/strict_mode.py`

## File Metadata

- **Path**: `torch/_higher_order_ops/strict_mode.py`
- **Size**: 3,831 bytes (3.74 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from typing import Any, TYPE_CHECKING, Union

import torch
import torch._subclasses.functional_tensor
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._functorch.utils import exposed_in
from torch._higher_order_ops.utils import _set_compilation_env, autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    _temp_remove_metadata_torch_function_mode,
    _temp_remove_pre_dispatch_torch_function_mode,
    disable_proxy_modes_tracing,
    make_fx,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.utils._python_dispatch import _get_current_dispatch_mode


if TYPE_CHECKING:
    from collections.abc import Callable


@exposed_in("torch")
def strict_mode(callable, operands):
    from torch._dynamo.backends.debugging import (
        make_eager_backend_with_torch_function_modes,
    )

    if torch.compiler.is_dynamo_compiling():
        return strict_mode_op(callable, operands)

    with _set_compilation_env():
        with _temp_remove_metadata_torch_function_mode() as metadata_mode:
            with _temp_remove_pre_dispatch_torch_function_mode() as predispatch_mode:
                modes = [metadata_mode, predispatch_mode]
                modes = [mode for mode in modes if mode is not None]
                if modes:
                    backend: Union[str, Callable[..., Any]] = (
                        make_eager_backend_with_torch_function_modes(modes)
                    )
                else:
                    backend = "eager"
                with torch._dynamo.utils.disable_cache_limit():
                    return torch.compile(
                        strict_mode_op, backend=backend, fullgraph=True
                    )(callable, operands)


class StrictMode(HigherOrderOperator):
    def __init__(self):
        super().__init__("strict_mode")

    def __call__(self, callable, operands):
        return super().__call__(callable, operands)


strict_mode_op = StrictMode()


@strict_mode_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def strict_mode_op_dense(callable, operands):
    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    return callable(*operands)


strict_mode_op.py_autograd_impl(
    autograd_not_implemented(strict_mode_op, deferred_error=True)
)


@strict_mode_op.py_impl(ProxyTorchDispatchMode)
def inner(mode, callable, operands):
    return trace_strict_mode(mode, strict_mode_op, callable, operands)


def trace_strict_mode(mode, strict_mode_op, callable, operands):
    pre_dispatch = getattr(mode, "pre_dispatch", False)

    with disable_proxy_modes_tracing():
        graph = make_fx(callable, pre_dispatch=pre_dispatch)(*operands)

    graph_name = mode.tracer.get_fresh_qualname("strict_graph_")
    mode.tracer.root.register_module(graph_name, graph)

    args = (graph, operands)

    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, args)

    out_proxy = mode.tracer.create_proxy(
        "call_function", strict_mode_op, proxy_args, {}, name="strict_mode"
    )

    out = graph(*operands)
    return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)


@strict_mode_op.py_impl(FakeTensorMode)
def strict_mode_fake_tensor_mode(mode, callable, operands):
    with mode:
        true_outs = callable(*operands)
    return true_outs


@strict_mode_op.py_functionalize_impl
def strict_mode_func(ctx, callable, inputs):
    unwrapped_inputs = ctx.unwrap_tensors(inputs)
    with ctx.redispatch_to_next():
        functional_callable = ctx.functionalize(callable)

        cond_return = strict_mode_op(functional_callable, unwrapped_inputs)
        return ctx.wrap_tensors(cond_return)

```



## High-Level Overview


This Python file contains 1 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `StrictMode`

**Functions defined**: `strict_mode`, `__init__`, `__call__`, `strict_mode_op_dense`, `inner`, `trace_strict_mode`, `strict_mode_fake_tensor_mode`, `strict_mode_func`

**Key imports**: Any, TYPE_CHECKING, Union, torch, torch._subclasses.functional_tensor, torch.utils._pytree as pytree, DispatchKey, exposed_in, _set_compilation_env, autograd_not_implemented, HigherOrderOperator, FakeTensorMode, _get_current_dispatch_mode


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_higher_order_ops`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Any, TYPE_CHECKING, Union
- `torch`
- `torch._subclasses.functional_tensor`
- `torch.utils._pytree as pytree`
- `torch._C`: DispatchKey
- `torch._functorch.utils`: exposed_in
- `torch._higher_order_ops.utils`: _set_compilation_env, autograd_not_implemented
- `torch._ops`: HigherOrderOperator
- `torch._subclasses.fake_tensor`: FakeTensorMode
- `torch.utils._python_dispatch`: _get_current_dispatch_mode
- `collections.abc`: Callable


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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
- [`torchbind.py_docs.md`](./torchbind.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`run_const_graph.py_docs.md`](./run_const_graph.py_docs.md)
- [`_invoke_quant.py_docs.md`](./_invoke_quant.py_docs.md)
- [`wrap.py_docs.md`](./wrap.py_docs.md)


## Cross-References

- **File Documentation**: `strict_mode.py_docs.md`
- **Keyword Index**: `strict_mode.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
