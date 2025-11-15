# Documentation: `docs/torch/onnx/_internal/exporter/_torchlib/ops/hop.py_docs.md`

## File Metadata

- **Path**: `docs/torch/onnx/_internal/exporter/_torchlib/ops/hop.py_docs.md`
- **Size**: 8,441 bytes (8.24 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/onnx/_internal/exporter/_torchlib/ops/hop.py`

## File Metadata

- **Path**: `torch/onnx/_internal/exporter/_torchlib/ops/hop.py`
- **Size**: 5,309 bytes (5.18 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""Implementation for higher-order operators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.onnx._internal._lazy_import import onnxscript_ir as ir
from torch.onnx._internal.exporter import _core
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


if TYPE_CHECKING:
    from collections.abc import Sequence


def call_op(
    op_type: str,
    *args: ir.Value,
    _num_outputs: int = 1,
    _domain: str = "",
    **kwargs: int | float | str | bool | ir.Graph | ir.TensorProtocol | Sequence[int],
) -> Sequence[ir.Value]:
    """Call an operator with the given arguments and keyword arguments.

    Arguments are always inputs, while keyword arguments are attributes.
    """
    # This is a wrapper around the IR node creation that hooks into the _builder.OpRecorder
    # tracer so that all nodes created are recorded the same way as if we were to use
    # onnxscript ops directly.
    from onnxscript.ir import convenience as ir_convenience

    assert _core.current_tracer is not None
    tracer = _core.current_tracer

    inputs = list(args)

    # If final inputs are None, strip them from the node inputs
    for input in reversed(inputs):
        if input is not None:
            break
        inputs.pop()

    # Construct and filter out None attributes
    attributes = [
        attr
        for attr in ir_convenience.convert_attributes(kwargs)
        if attr.value is not None  # type: ignore[union-attr]
    ]
    tracer.nodes.append(
        node := ir.Node(
            _domain,
            op_type,
            inputs=inputs,
            attributes=attributes,
            num_outputs=_num_outputs,
            version=tracer.opset.version,
        )
    )
    return node.outputs


@onnx_impl(torch.ops.higher_order.cond, no_compile=True)
def higher_order_cond(
    cond: ir.Value,
    true_func: ir.Function,
    false_func: ir.Function,
    inputs: Sequence[ir.Value],
) -> Sequence[ir.Value]:
    then_node = ir.Node(
        true_func.domain, true_func.name, inputs, num_outputs=len(true_func.outputs)
    )
    else_node = ir.Node(
        false_func.domain, false_func.name, inputs, num_outputs=len(false_func.outputs)
    )

    # ONNX Runtime complains about duplicate output names if we don't rename them.
    # But the doesn't seem to be an actual violation of SSA form without renaming.
    for func_out, out in zip(true_func.outputs, then_node.outputs):
        out.name = f"{func_out.name}_{true_func.name}"
    for func_out, out in zip(false_func.outputs, else_node.outputs):
        out.name = f"{func_out.name}_{false_func.name}"

    return call_op(
        "If",
        cond,
        _num_outputs=len(true_func.outputs),
        then_branch=ir.Graph(
            (), then_node.outputs, nodes=[then_node], name=true_func.name
        ),
        else_branch=ir.Graph(
            (), else_node.outputs, nodes=[else_node], name=false_func.name
        ),
    )


@onnx_impl(torch.ops.higher_order.scan, no_compile=True)
def higher_order_scan(
    body_func: ir.Function,
    scan_inits: Sequence[ir.Value],
    scan_inputs: Sequence[ir.Value],
    additional_inputs: Sequence[ir.Value] | None,
    reverse: bool = False,
) -> Sequence[ir.Value]:
    """https://github.com/pytorch/pytorch/blob/66ac724b56e6c37a534f3e066423ef2f41d7477f/torch/_higher_order_ops/scan.py#L109"""
    subgraph_inputs = [
        *[
            ir.Value(
                name=f"{inp.name}_{body_func.name}__subgraph_in",
                shape=inp.shape,
                type=ir.TensorType(inp.dtype),  # type: ignore[arg-type]
            )
            for inp in scan_inits
        ],
        *[
            ir.Value(
                name=f"{inp.name}_{body_func.name}__subgraph_in",
                # The iterated element passed to the body subgraph does not have a sequence axis.
                # It will have a rank one less than the rank of the corresponding scan_input.
                shape=ir.Shape(inp.shape[1:]),  # type: ignore[index]
                type=ir.TensorType(inp.dtype),  # type: ignore[arg-type]
            )
            for inp in scan_inputs
        ],
    ]
    # The one and only node in the Scan subgraph that calls the body_func
    body_node = ir.Node(
        body_func.domain,
        body_func.name,
        [
            *subgraph_inputs,
            *(additional_inputs or []),
        ],
        num_outputs=len(body_func.outputs),
    )

    # ONNX Runtime complains about duplicate output names if we don't rename them.
    # But the doesn't seem to be an actual violation of SSA form without renaming.
    for func_out, out in zip(body_func.outputs, body_node.outputs):
        out.name = f"{func_out.name}_{body_func.name}"

    n_outputs = len(body_func.outputs) - len(scan_inits)
    return call_op(
        "Scan",
        *scan_inits,
        *scan_inputs,
        _num_outputs=len(body_func.outputs),
        body=ir.Graph(
            subgraph_inputs,
            body_node.outputs,
            nodes=[body_node],
            name=body_func.name,
        ),
        num_scan_inputs=len(scan_inputs),
        scan_input_directions=[(1 if reverse else 0) for _ in scan_inputs],
        scan_output_directions=[(1 if reverse else 0) for _ in range(n_outputs)],
    )

```



## High-Level Overview

"""Implementation for higher-order operators."""from __future__ import annotationsfrom typing import TYPE_CHECKINGimport torchfrom torch.onnx._internal._lazy_import import onnxscript_ir as irfrom torch.onnx._internal.exporter import _corefrom torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_implif TYPE_CHECKING:    from collections.abc import Sequencedef call_op(    op_type: str,    *args: ir.Value,    _num_outputs: int = 1,    _domain: str = "",    **kwargs: int | float | str | bool | ir.Graph | ir.TensorProtocol | Sequence[int],) -> Sequence[ir.Value]:

This Python file contains 0 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `call_op`, `higher_order_cond`, `higher_order_scan`

**Key imports**: annotations, TYPE_CHECKING, torch, onnxscript_ir as ir, _core, onnx_impl, Sequence, convenience as ir_convenience


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/onnx/_internal/exporter/_torchlib/ops`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `typing`: TYPE_CHECKING
- `torch`
- `torch.onnx._internal._lazy_import`: onnxscript_ir as ir
- `torch.onnx._internal.exporter`: _core
- `torch.onnx._internal.exporter._torchlib._torchlib_registry`: onnx_impl
- `collections.abc`: Sequence
- `onnxscript.ir`: convenience as ir_convenience


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

Files in the same folder (`torch/onnx/_internal/exporter/_torchlib/ops`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`nn.py_docs.md`](./nn.py_docs.md)
- [`core.py_docs.md`](./core.py_docs.md)
- [`symops.py_docs.md`](./symops.py_docs.md)
- [`symbolic.py_docs.md`](./symbolic.py_docs.md)


## Cross-References

- **File Documentation**: `hop.py_docs.md`
- **Keyword Index**: `hop.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/onnx/_internal/exporter/_torchlib/ops`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/onnx/_internal/exporter/_torchlib/ops`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/onnx/_internal/exporter/_torchlib/ops`):

- [`symops.py_docs.md_docs.md`](./symops.py_docs.md_docs.md)
- [`nn.py_docs.md_docs.md`](./nn.py_docs.md_docs.md)
- [`core.py_docs.md_docs.md`](./core.py_docs.md_docs.md)
- [`symops.py_kw.md_docs.md`](./symops.py_kw.md_docs.md)
- [`nn.py_kw.md_docs.md`](./nn.py_kw.md_docs.md)
- [`hop.py_kw.md_docs.md`](./hop.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`core.py_kw.md_docs.md`](./core.py_kw.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `hop.py_docs.md_docs.md`
- **Keyword Index**: `hop.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
