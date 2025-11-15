# Documentation: `docs/torch/_inductor/subgraph_lowering.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/subgraph_lowering.py_docs.md`
- **Size**: 11,277 bytes (11.01 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/subgraph_lowering.py`

## File Metadata

- **Path**: `torch/_inductor/subgraph_lowering.py`
- **Size**: 7,313 bytes (7.14 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""Utilities for lowering subgraphs used by higher order operators"""

import functools
import operator
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional, TypeVar, Union
from typing_extensions import ParamSpec

import torch
from torch.utils._ordered_set import OrderedSet

from . import ir
from .exc import SubgraphLoweringException
from .graph import GraphLowering
from .ops_handler import SimpleCSEHandler
from .virtualized import ops, V, WrapperHandler


T = TypeVar("T")
_P = ParamSpec("_P")

OpOverload = torch._ops.OpOverload
LoweringDict = dict[Union[OpOverload, str], Callable[..., Any]]
TargetType = Union[Callable[..., Any], str]


class PointwiseSubgraphLowering(torch.fx.Interpreter):
    """
    Lowers a pointwise subgraph to a single set of buffers with a separate
    lowering object. Errors if buffers are created unexpectedly
    """

    graph_outputs: Optional[list[ir.IRNode]]
    root_graph: GraphLowering
    _current_op: Optional[TargetType]
    # For backwards of buffer_grads with scatters we allow mutations
    allowed_mutations: Optional[OrderedSet[OpOverload]]
    additional_lowerings: Optional[LoweringDict]
    buffers: list[ir.Buffer]
    mutated_buffers: OrderedSet[str]

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        root_graph_lowering: GraphLowering,
        allowed_mutations: Optional[OrderedSet[OpOverload]] = None,
        additional_lowerings: Optional[LoweringDict] = None,
    ) -> None:
        super().__init__(gm)
        self.graph_outputs = None
        self.root_graph = root_graph_lowering
        self.allowed_mutations = allowed_mutations
        self.additional_lowerings = additional_lowerings
        self._current_op = None

        # Used to track buffers created during lowering
        self.mutated_buffers = OrderedSet()
        self.buffers = []

    @contextmanager
    def _op_context(self, op: TargetType) -> Generator[None, None, None]:
        """Set which op is being processed in call function to know if we can mutate buffers"""
        previous = self._current_op
        self._current_op = op
        try:
            yield
        finally:
            self._current_op = previous

    def _approved_mutator(self) -> bool:
        return (
            self.allowed_mutations is not None
            and self._current_op in self.allowed_mutations
        )

    def mark_buffer_mutated(self, name: str) -> None:
        if self._approved_mutator():
            self.mutated_buffers.add(name)
        else:
            raise SubgraphLoweringException(
                f"Buffer mutation detected during lowering of {self._current_op}. "
                "Buffer mutations are only allowed in approved mutation ops. "
                "This is an error in the lowering of the subgraph, please file a bug report."
            )

    def register_buffer(self, buffer: ir.Buffer, *, set_name: bool = False) -> str:
        if self._approved_mutator():
            name = self.root_graph.register_buffer(buffer, set_name=set_name)
            return name
        else:
            raise SubgraphLoweringException(
                "Buffers cannot be created while lowering a pointwise subgraph. "
                "This could be for a good reason (e.g. you're calling an op we can't codegen as a pointwise op), "
                "but it could also be a bug. Please file a bug report if you think this should be supportable."
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.root_graph, name)

    def call_function(
        self,
        target: TargetType,
        args: Any,
        kwargs: dict[str, Any],
    ) -> Any:
        from .lowering import lowerings

        with self._op_context(target):
            if target is operator.getitem and isinstance(args[0], (list, tuple, dict)):
                return super().call_function(target, args, kwargs)

            # These takes precedence over the main lowerings
            if self.additional_lowerings is not None:
                if target in self.additional_lowerings:
                    assert isinstance(target, OpOverload)
                    return self.additional_lowerings[target](*args, **kwargs)

            if target not in lowerings:
                raise SubgraphLoweringException(
                    f"{target} not supported in subgraph, (missing lowering)"
                )

            return lowerings[target](*args, **kwargs)

    def output(self, target: str, args: tuple[Any], kwargs: dict[str, Any]) -> None:  # type: ignore[override]
        assert len(args) == 1
        self.graph_outputs = args[0]


@dataclass
class InputDescriptor:
    dtype: torch.dtype
    device: torch.device


class TracingOpsHandler(WrapperHandler):
    def __init__(self, tracer: torch.fx.Tracer, num_inputs: int) -> None:
        parent = tracer.create_proxy("placeholder", "ops", (), {})
        super().__init__(parent)
        self.tracer = tracer

        self.placeholders = [
            self.tracer.create_proxy("placeholder", f"input{i}", (), {})
            for i in range(num_inputs)
        ]

    def placeholder(self, idx: int) -> torch.fx.Proxy:
        return self.placeholders[idx]

    def output(self, *args: tuple[object]) -> None:
        self.tracer.create_node(
            "output", "output", (tuple(self.tracer.create_arg(a) for a in args),), {}
        )


def lower_pointwise_subgraph(
    subgraph: ir.Subgraph, inputs: list[InputDescriptor]
) -> Callable[_P, Any]:
    # Lower subgraph to ir.Pointwise nodes
    def fake_inner_fn(
        loop_idx: int, input_idx: int
    ) -> Union[ir.Expr, ir.TensorBox, None]:
        return ops.placeholder(input_idx)

    graph_inputs = [
        ir.Pointwise.create(
            device=desc.device,
            dtype=desc.dtype,
            inner_fn=functools.partial(fake_inner_fn, input_idx=i),
            ranges=[],
        )
        for i, desc in enumerate(inputs)
    ]
    gm = subgraph.graph_module
    pw_subgraph = PointwiseSubgraphLowering(gm, root_graph_lowering=V.graph)
    with V.set_graph_handler(pw_subgraph):  # type: ignore[arg-type]
        pw_subgraph.run(*graph_inputs)

    # Combine multiple pointwise computations into a single graph module
    # Do this by tracing through each individually and doing CSE
    tracer = torch.fx.Tracer()
    tracer.graph = torch.fx.Graph(tracer_cls=tracer.__class__)
    trace_ops = SimpleCSEHandler(TracingOpsHandler(tracer, len(inputs)))
    assert pw_subgraph.graph_outputs is not None

    with V.set_ops_handler(trace_ops):
        output_irs = []

        for out_var in pw_subgraph.graph_outputs:
            assert isinstance(out_var, ir.TensorBox), type(out_var)
            assert out_var.get_size() == []
            assert isinstance(out_var.data, ir.StorageBox)
            assert isinstance(out_var.data.data, ir.Pointwise)

            idx = ()
            ir_out = out_var.data.data.inner_fn(idx)

            output_irs.append(ir_out)

        ops.output(*output_irs)

    lowered_gm = torch.fx.GraphModule({}, tracer.graph)

    def inner_fn(*args: _P.args, **kwargs: _P.kwargs) -> Any:
        return lowered_gm(V.get_ops_handler(), *args, **kwargs)

    return inner_fn

```



## High-Level Overview

"""Utilities for lowering subgraphs used by higher order operators"""import functoolsimport operatorfrom collections.abc import Callable, Generatorfrom contextlib import contextmanagerfrom dataclasses import dataclassfrom typing import Any, Optional, TypeVar, Unionfrom typing_extensions import ParamSpecimport torchfrom torch.utils._ordered_set import OrderedSetfrom . import irfrom .exc import SubgraphLoweringExceptionfrom .graph import GraphLoweringfrom .ops_handler import SimpleCSEHandlerfrom .virtualized import ops, V, WrapperHandlerT = TypeVar("T")_P = ParamSpec("_P")OpOverload = torch._ops.OpOverloadLoweringDict = dict[Union[OpOverload, str], Callable[..., Any]]TargetType = Union[Callable[..., Any], str]class PointwiseSubgraphLowering(torch.fx.Interpreter):

This Python file contains 4 class(es) and 14 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `PointwiseSubgraphLowering`, `InputDescriptor`, `TracingOpsHandler`

**Functions defined**: `__init__`, `_op_context`, `_approved_mutator`, `mark_buffer_mutated`, `register_buffer`, `__getattr__`, `call_function`, `output`, `__init__`, `placeholder`, `output`, `lower_pointwise_subgraph`, `fake_inner_fn`, `inner_fn`

**Key imports**: functools, operator, Callable, Generator, contextmanager, dataclass, Any, Optional, TypeVar, Union, ParamSpec, torch, OrderedSet, ir


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `operator`
- `collections.abc`: Callable, Generator
- `contextlib`: contextmanager
- `dataclasses`: dataclass
- `typing`: Any, Optional, TypeVar, Union
- `typing_extensions`: ParamSpec
- `torch`
- `torch.utils._ordered_set`: OrderedSet
- `.`: ir
- `.exc`: SubgraphLoweringException
- `.graph`: GraphLowering
- `.ops_handler`: SimpleCSEHandler
- `.virtualized`: ops, V, WrapperHandler
- `.lowering`: lowerings


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

- **File Documentation**: `subgraph_lowering.py_docs.md`
- **Keyword Index**: `subgraph_lowering.py_kw.md`
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

- **File Documentation**: `subgraph_lowering.py_docs.md_docs.md`
- **Keyword Index**: `subgraph_lowering.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
