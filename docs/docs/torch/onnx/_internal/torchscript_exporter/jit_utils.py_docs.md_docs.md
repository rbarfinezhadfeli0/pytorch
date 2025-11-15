# Documentation: `docs/torch/onnx/_internal/torchscript_exporter/jit_utils.py_docs.md`

## File Metadata

- **Path**: `docs/torch/onnx/_internal/torchscript_exporter/jit_utils.py_docs.md`
- **Size**: 17,569 bytes (17.16 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `torch/onnx/_internal/torchscript_exporter/jit_utils.py`

## File Metadata

- **Path**: `torch/onnx/_internal/torchscript_exporter/jit_utils.py`
- **Size**: 13,986 bytes (13.66 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
# mypy: allow-untyped-defs
"""Utilities for manipulating the torch.Graph object and the torchscript."""

from __future__ import annotations

import dataclasses
import re
import typing
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import _C
from torch.onnx._internal.torchscript_exporter import registration
from torch.onnx._internal.torchscript_exporter._globals import GLOBALS


_ATTR_PATTERN = re.compile("^(.+)_(([ifstgz])|(ty))$")
_SKIP_NODE_ATTRIBUTES = {"inplace", "aten"}


@dataclasses.dataclass
class GraphContext:
    """Extra context for symbolic functions with all methods from torch.Graph.

    NOTE: This class is not meant for external consumption. Please do not depend on
    it outside of torch.onnx as the interface may evolve.

    Attributes:
        graph: The _C.Graph being constructed.
        block: The current _C.Block being constructed.
        opset: The opset version.
        original_node: Current node that is being converted from.
        params_dict: Mapping from graph initializer name to IValue.
        env: Mapping from Torch domain graph Value to ONNX domain graph Value.
        values_in_env: Set of all values in env, for constant-time lookups.
        new_nodes: List that tracks all new nodes that are added (used to make
            sure metadata is propagated to all new nodes).
    """

    graph: _C.Graph
    block: _C.Block
    opset: int
    original_node: _C.Node
    params_dict: dict[str, _C.IValue]
    env: dict[_C.Value, _C.Value]
    values_in_env: set[_C.Value]
    new_nodes: list[_C.Node] = dataclasses.field(default_factory=list)

    # Relay methods from _C.Graph for compatibility with symbolic functions that expect
    # a _C.Graph
    def __getattr__(self, name: str) -> Any:
        return getattr(self.graph, name)

    def op(
        self,
        opname: str,
        *raw_args: torch.Tensor | _C.Value,
        outputs: int = 1,
        **kwargs,
    ):
        """Creates an ONNX operator "opname", taking "raw_args" as inputs and "kwargs" as attributes.

        The set of operators and the inputs/attributes they take
        is documented at https://github.com/onnx/onnx/blob/master/docs/Operators.md

        Args:
            opname: The ONNX operator name, e.g., `Abs` or `Add`, or an operator qualified
                with a namespace, e.g., `aten::add`.
            raw_args: The inputs to the operator; usually provided
                as arguments to the `symbolic` definition.
            outputs: The number of outputs this operator returns.
                By default an operator is assumed to return a single output.
                If `outputs` is greater than one, this functions returns a tuple
                of output `Value`, representing each output of the ONNX operator
                in order.
            kwargs: The attributes of the ONNX operator, whose keys are named
                according to the following convention: `alpha_f` indicates
                the `alpha` attribute with type `f`.  The valid type specifiers are
                `f` (float), `i` (int), `s` (string) or `t` (Tensor).  An attribute
                specified with type float accepts either a single float, or a
                list of floats (e.g., you would say `dims_i` for a `dims` attribute
                that takes a list of integers).

        Returns:
            The value representing the single output of this operator (see the `outputs`
            keyword argument for multi-return nodes).
        """
        return _add_op(self, opname, *raw_args, outputs=outputs, **kwargs)

    def aten_op(self, operator: str, *args, overload_name: str = "", **kwargs):
        """Generates an ONNX ATen op node.

        This function is for backward compatibility with the old symbolic functions.
        """
        return self.op(
            "aten::ATen",
            *args,
            operator_s=operator,
            overload_name_s=overload_name,
            **kwargs,
        )

    # NOTE: For backward compatibility with the old symbolic functions.
    # We are probably going to remove this only after the fx exporter is established.
    at = aten_op

    def onnxscript_op(
        self,
        onnx_fn,
        *raw_args: torch.Tensor | _C.Value,
        outputs: int = 1,
        **kwargs,
    ):
        """Creates an ONNX operator from onnx-script function, taking "raw_args" as inputs and "kwargs" as attributes.

        onnx-script repository: https://github.com/microsoft/onnx-script

        Args:
            onnx_fn: ONNXFunction from onnx-script; An example can be found at
                https://github.com/microsoft/onnx-script#example
            raw_args: The inputs to the operator; usually provided
                as arguments to the `symbolic` definition.
            outputs: The number of outputs this operator returns.
                By default an operator is assumed to return a single output.
                If `outputs` is greater than one, this functions returns a tuple
                of output `Value`, representing each output of the ONNX operator
                in order.
            kwargs: The attributes of the ONNX operator, whose keys are named
                according to the following convention: `alpha_f` indicates
                the `alpha` attribute with type `f`.  The valid type specifiers are
                `f` (float), `i` (int), `s` (string) or `t` (Tensor).  An attribute
                specified with type float accepts either a single float, or a
                list of floats (e.g., you would say `dims_i` for a `dims` attribute
                that takes a list of integers).

        Returns:
            The value representing the single output of this operator (see the `outputs`
            keyword argument for multi-return nodes).
        """
        # NOTE(titaiwang): This is using class attributes, and it needs to be updated
        # if onnx-script makes any change on these.
        symbolic_name = f"{onnx_fn.opset.domain}::{onnx_fn.name}"
        opset_version = onnx_fn.opset.version

        registration.custom_onnx_symbolic(symbolic_name, opset_version)(onnx_fn)

        return _add_op(self, symbolic_name, *raw_args, outputs=outputs, **kwargs)


def add_op_with_blocks(
    graph_context: GraphContext,
    opname: str,
    *inputs: _C.Value,
    outputs: int = 1,
    n_blocks: int = 1,
    **attributes,
) -> tuple[Any, tuple[GraphContext, ...], _C.Node]:
    """Creates an ONNX operator "opname", taking inputs and attributes.

    Args:
        graph_context: The context for the current graph.
        opname: The ONNX operator name, e.g., `Abs` or `Add`, or an operator qualified
            with a namespace, e.g., `aten::add`.
        inputs: The inputs to the operator.
        outputs: The number of outputs this operator returns.
            By default an operator is assumed to return a single output.
            If `outputs` is greater than one, this functions returns a tuple
            of output `Value`, representing each output of the ONNX operator
            in order.
        n_blocks: The number of sub-blocks to create in the node.
        attributes: The attributes of the ONNX operator.

    Returns:
        A tuple of (output_values, new_contexts, node) where:
            output_values: One or more output value of this operator
                (see the `outputs` keyword argument for multi-return nodes).
            new_contexts: A tuple of new graph contexts for each sub-block.
            node: The node representing the operator.
    """

    output_values = graph_context.op(opname, *inputs, outputs=outputs, **attributes)
    if isinstance(output_values, Sequence):
        node = output_values[0].node()
    else:
        node = output_values.node()

    new_contexts = []
    for _ in range(n_blocks):
        new_block = node.addBlock()
        # Create shallow copy of the graph context and update the block
        new_context = dataclasses.replace(graph_context, block=new_block)
        new_contexts.append(new_context)

    return output_values, tuple(new_contexts), node


def _add_op(
    graph_context: GraphContext,
    opname: str,
    *args: torch.Tensor | _C.Value,
    outputs: int = 1,
    **kwargs,
):
    """Creates an ONNX operator "opname", taking "args" as inputs and attributes "kwargs".

    The set of operators and the inputs/attributes they take
    is documented at https://github.com/onnx/onnx/blob/master/docs/Operators.md

    Args:
        graph_context: The Torch Graph or Block.
        opname: The ONNX operator name, e.g., `Abs` or `Add`, or an operator qualified
            with a namespace, e.g., `aten::add`.
        args: The inputs to the operator; usually provided
            as arguments to the `symbolic` definition.
        outputs: The number of outputs this operator returns.
            By default an operator is assumed to return a single output.
            If `outputs` is greater than one, this functions returns a tuple
            of output `Value`, representing each output of the ONNX operator
            in order.
        kwargs: The attributes of the ONNX operator, whose keys are named
            according to the following convention: `alpha_f` indicates
            the `alpha` attribute with type `f`.  The valid type specifiers are
            `f` (float), `i` (int), `s` (string) or `t` (Tensor).  An attribute
            specified with type float accepts either a single float, or a
            list of floats (e.g., you would say `dims_i` for a `dims` attribute
            that takes a list of integers).

    Returns:
        (Union[_C.Value, Tuple[_C.Value, ...]])
        The value representing the single output of this operator (see the `outputs`
        keyword argument for multi-return nodes).
    """
    inputs = [_const_if_tensor(graph_context, arg) for arg in args]
    # Filter out None attributes, this can be convenient client side because
    # now they can pass through None attributes, and have them not show up
    attributes = {k: v for k, v in kwargs.items() if v is not None}

    if "::" not in opname:
        opname = "onnx::" + opname

    node = _create_node(
        graph_context.block,
        opname,
        inputs,
        attributes,
        params_dict=graph_context.params_dict,
        opset_version=graph_context.opset,
        n_outputs=outputs,
        shape_inference=GLOBALS.onnx_shape_inference,
    )
    graph_context.new_nodes.append(node)

    if outputs == 1:
        return node.output()
    return tuple(node.outputs())


def _const_if_tensor(graph_context: GraphContext, arg):
    if arg is None:
        return arg
    if isinstance(arg, _C.Value):
        return arg

    return _add_op(graph_context, "onnx::Constant", value_z=arg)


def _create_node(
    graph_or_block: _C.Graph | _C.Block,
    domain_op: str,
    inputs: Sequence,
    attributes: dict,
    params_dict: dict,
    opset_version: int,
    n_outputs: int,
    shape_inference: bool = True,
) -> _C.Node:
    """Creates an node 'domain_op', taking inputs and attributes."""
    if isinstance(graph_or_block, _C.Graph):
        graph = graph_or_block
        node = graph.create(domain_op, inputs, n_outputs)
        node = graph.insertNode(node)
    elif isinstance(graph_or_block, _C.Block):
        block = graph_or_block
        node = block.addNode(domain_op, inputs)

        # Block does not have create defined, so we need to add outputs manually
        if n_outputs > 1:
            for _ in range(1, n_outputs):
                node.addOutput()

    node_outputs = tuple(node.outputs())  # type: ignore[possibly-undefined]
    assert len(node_outputs) == n_outputs

    aten = domain_op.startswith("aten::")

    # Add all attributes
    for key, value in sorted(attributes.items()):
        if key in _SKIP_NODE_ATTRIBUTES:
            continue
        # pyrefly: ignore [unbound-name]
        _add_attribute(node, key, value, aten=aten)
    if shape_inference:
        # pyrefly: ignore [unbound-name]
        _C._jit_pass_onnx_node_shape_type_inference(node, params_dict, opset_version)
    # pyrefly: ignore [unbound-name]
    return node


def _is_onnx_list(value):
    return isinstance(value, Iterable) and not isinstance(
        value, (str, bytes, torch.Tensor)
    )


def _scalar(x: torch.Tensor):
    """Convert a scalar tensor into a Python value."""
    assert x.numel() == 1
    return x[0]


def _add_attribute(node: _C.Node, key: str, value: Any, aten: bool):
    r"""Initializes the right attribute based on type of value."""
    m = _ATTR_PATTERN.match(key)
    if m is None:
        raise ValueError(
            f"Invalid attribute specifier '{key}' names "
            "must be suffixed with type, e.g. 'dim_i' or 'dims_i'"
        )
    name, kind = m.group(1), m.group(2)
    if _is_onnx_list(value):
        kind += "s"

    return getattr(node, f"{kind}_")(name, value)


def _is_tensor(x: _C.Value) -> bool:
    return x.type().isSubtypeOf(_C.TensorType.get())


def get_device_from_value(value: _C.Value) -> torch.device | None:
    if not _is_tensor(value):
        return None
    tensor_type = typing.cast(_C.TensorType, value.type())
    return tensor_type.device()


def parse_node_kind(kind: str) -> tuple[str, str]:
    """Parse node kind into domain and Op name."""
    if "::" not in kind:
        raise ValueError(f"Node kind: {kind} is invalid. '::' is not in node kind.")
    domain, opname = kind.split("::", 1)
    if "::" in opname:
        raise ValueError(f"Node kind: {kind} is invalid. '::' should only appear once.")
    return domain, opname


def is_aten(domain: str) -> bool:
    """Check if the domain is official."""
    return domain == "aten"


def is_prim(domain: str) -> bool:
    """Check if the domain is official."""
    return domain == "prim"


def is_onnx(domain: str) -> bool:
    """Check if the domain is official."""
    return domain == "onnx"

```



## High-Level Overview

"""Utilities for manipulating the torch.Graph object and the torchscript."""from __future__ import annotationsimport dataclassesimport reimport typingfrom collections.abc import Iterable, Sequencefrom typing import Anyimport torchfrom torch import _Cfrom torch.onnx._internal.torchscript_exporter import registrationfrom torch.onnx._internal.torchscript_exporter._globals import GLOBALS_ATTR_PATTERN = re.compile("^(.+)_(([ifstgz])|(ty))$")_SKIP_NODE_ATTRIBUTES = {"inplace", "aten"}@dataclasses.dataclassclass GraphContext:

This Python file contains 3 class(es) and 17 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `GraphContext`

**Functions defined**: `__getattr__`, `op`, `aten_op`, `onnxscript_op`, `add_op_with_blocks`, `_add_op`, `_const_if_tensor`, `_create_node`, `_is_onnx_list`, `_scalar`, `_add_attribute`, `_is_tensor`, `get_device_from_value`, `parse_node_kind`, `is_aten`, `is_prim`, `is_onnx`

**Key imports**: annotations, dataclasses, re, typing, Iterable, Sequence, Any, torch, _C, registration, GLOBALS


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/onnx/_internal/torchscript_exporter`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `dataclasses`
- `re`
- `typing`
- `collections.abc`: Iterable, Sequence
- `torch`
- `torch.onnx._internal.torchscript_exporter`: registration
- `torch.onnx._internal.torchscript_exporter._globals`: GLOBALS


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`torch/onnx/_internal/torchscript_exporter`):

- [`symbolic_opset7.py_docs.md`](./symbolic_opset7.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`symbolic_opset14.py_docs.md`](./symbolic_opset14.py_docs.md)
- [`symbolic_opset11.py_docs.md`](./symbolic_opset11.py_docs.md)
- [`verification.py_docs.md`](./verification.py_docs.md)
- [`symbolic_opset12.py_docs.md`](./symbolic_opset12.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`_experimental.py_docs.md`](./_experimental.py_docs.md)
- [`symbolic_opset20.py_docs.md`](./symbolic_opset20.py_docs.md)
- [`symbolic_opset9.py_docs.md`](./symbolic_opset9.py_docs.md)


## Cross-References

- **File Documentation**: `jit_utils.py_docs.md`
- **Keyword Index**: `jit_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/onnx/_internal/torchscript_exporter`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/onnx/_internal/torchscript_exporter`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`docs/torch/onnx/_internal/torchscript_exporter`):

- [`symbolic_opset14.py_docs.md_docs.md`](./symbolic_opset14.py_docs.md_docs.md)
- [`symbolic_opset18.py_kw.md_docs.md`](./symbolic_opset18.py_kw.md_docs.md)
- [`_experimental.py_kw.md_docs.md`](./_experimental.py_kw.md_docs.md)
- [`onnx_proto_utils.py_docs.md_docs.md`](./onnx_proto_utils.py_docs.md_docs.md)
- [`symbolic_opset13.py_kw.md_docs.md`](./symbolic_opset13.py_kw.md_docs.md)
- [`symbolic_opset12.py_docs.md_docs.md`](./symbolic_opset12.py_docs.md_docs.md)
- [`symbolic_opset16.py_docs.md_docs.md`](./symbolic_opset16.py_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`symbolic_helper.py_kw.md_docs.md`](./symbolic_helper.py_kw.md_docs.md)
- [`symbolic_opset8.py_docs.md_docs.md`](./symbolic_opset8.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `jit_utils.py_docs.md_docs.md`
- **Keyword Index**: `jit_utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
