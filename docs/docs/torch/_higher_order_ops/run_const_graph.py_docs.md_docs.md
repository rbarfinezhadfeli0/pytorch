# Documentation: `docs/torch/_higher_order_ops/run_const_graph.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_higher_order_ops/run_const_graph.py_docs.md`
- **Size**: 5,450 bytes (5.32 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_higher_order_ops/run_const_graph.py`

## File Metadata

- **Path**: `torch/_higher_order_ops/run_const_graph.py`
- **Size**: 2,462 bytes (2.40 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from typing import Any, TYPE_CHECKING

import torch
from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode


if TYPE_CHECKING:
    from torch._subclasses.functional_tensor import BaseFunctionalizeAPI

from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.utils import _pytree as pytree


class RunConstGraph(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("run_const_graph")

    def __call__(self, graph: torch.fx.GraphModule, args: tuple[object, ...]) -> object:
        return super().__call__(graph, args)


run_const_graph = RunConstGraph()


@run_const_graph.py_impl(ProxyTorchDispatchMode)
def run_const_graph_dispatch_mode(
    mode: ProxyTorchDispatchMode, graph: torch.fx.GraphModule, args: tuple[object, ...]
) -> object:
    const_gm, weights = graph, args
    p_args = pytree.tree_map(mode.tracer.unwrap_proxy, (graph, args))  # type: ignore[union-attr]
    assert isinstance(const_gm, torch.fx.GraphModule)
    assert not hasattr(mode.tracer.root, "_const_graph")  # type: ignore[union-attr]
    mode.tracer.root.register_module("_const_graph", const_gm)  # type: ignore[union-attr]

    proxy = mode.tracer.create_proxy("call_function", run_const_graph, p_args, {})

    out = const_gm(*weights)
    return track_tensor_tree(out, proxy, constant=None, tracer=mode.tracer)


@run_const_graph.py_functionalize_impl
def run_const_graph_functional(
    ctx: "BaseFunctionalizeAPI", graph: torch.fx.GraphModule, args: tuple[Any, ...]
) -> Any:
    unwrapped_args = ctx.unwrap_tensors(args)

    with ctx.redispatch_to_next():
        out = run_const_graph(graph, unwrapped_args)
        return ctx.wrap_tensors(out)  # type: ignore[arg-type]


run_const_graph.py_autograd_impl(
    autograd_not_implemented(run_const_graph, deferred_error=True)
)


@run_const_graph.py_impl(FakeTensorMode)
def run_const_graph_fake_tensor_mode(
    mode: FakeTensorMode, graph: torch.fx.GraphModule, args: tuple[object, ...]
) -> object:
    assert isinstance(graph, torch.fx.GraphModule)
    with mode:
        return graph(*args)


@run_const_graph.py_impl(DispatchKey.CPU)
def run_const_graph_cpu(
    graph: torch.fx.GraphModule, args: tuple[object, ...]
) -> object:
    assert isinstance(graph, torch.fx.GraphModule)
    return graph(*args)

```



## High-Level Overview


This Python file contains 1 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `RunConstGraph`

**Functions defined**: `__init__`, `__call__`, `run_const_graph_dispatch_mode`, `run_const_graph_functional`, `run_const_graph_fake_tensor_mode`, `run_const_graph_cpu`

**Key imports**: Any, TYPE_CHECKING, torch, DispatchKey, autograd_not_implemented, HigherOrderOperator, FakeTensorMode, BaseFunctionalizeAPI, ProxyTorchDispatchMode, track_tensor_tree, _pytree as pytree


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_higher_order_ops`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Any, TYPE_CHECKING
- `torch`
- `torch._C`: DispatchKey
- `torch._higher_order_ops.utils`: autograd_not_implemented
- `torch._ops`: HigherOrderOperator
- `torch._subclasses.fake_tensor`: FakeTensorMode
- `torch._subclasses.functional_tensor`: BaseFunctionalizeAPI
- `torch.fx.experimental.proxy_tensor`: ProxyTorchDispatchMode, track_tensor_tree
- `torch.utils`: _pytree as pytree


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
- [`_invoke_quant.py_docs.md`](./_invoke_quant.py_docs.md)
- [`wrap.py_docs.md`](./wrap.py_docs.md)


## Cross-References

- **File Documentation**: `run_const_graph.py_docs.md`
- **Keyword Index**: `run_const_graph.py_kw.md`
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
- [`effects.py_kw.md_docs.md`](./effects.py_kw.md_docs.md)
- [`partitioner.py_docs.md_docs.md`](./partitioner.py_docs.md_docs.md)
- [`strict_mode.py_docs.md_docs.md`](./strict_mode.py_docs.md_docs.md)
- [`out_dtype.py_kw.md_docs.md`](./out_dtype.py_kw.md_docs.md)
- [`wrap.py_docs.md_docs.md`](./wrap.py_docs.md_docs.md)
- [`while_loop.py_kw.md_docs.md`](./while_loop.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`invoke_subgraph.py_docs.md_docs.md`](./invoke_subgraph.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `run_const_graph.py_docs.md_docs.md`
- **Keyword Index**: `run_const_graph.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
