# Documentation: `docs/torch/onnx/_internal/exporter/_torchlib/ops/symbolic.py_docs.md`

## File Metadata

- **Path**: `docs/torch/onnx/_internal/exporter/_torchlib/ops/symbolic.py_docs.md`
- **Size**: 8,127 bytes (7.94 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/onnx/_internal/exporter/_torchlib/ops/symbolic.py`

## File Metadata

- **Path**: `torch/onnx/_internal/exporter/_torchlib/ops/symbolic.py`
- **Size**: 4,722 bytes (4.61 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""Implementation for higher-order operators."""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript.ir import convenience as ir_convenience

import torch
from torch.onnx._internal._lazy_import import onnxscript_ir as ir
from torch.onnx._internal.exporter import _core
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl
from torch.onnx.ops import _symbolic_impl


if TYPE_CHECKING:
    from collections.abc import Sequence


def _call_symbolic_op(
    op_type: str,
    domain: str,
    args: Sequence[ir.Value | None],
    kwargs: dict[str, int | float | str | bool | list[int] | list[float] | list[str]],
    dtypes: Sequence[int],
    version: int | None,
    metadata_props: dict[str, str] | None,
) -> Sequence[ir.Value]:
    """Call an operator with the given arguments and keyword arguments.

    Arguments are always inputs, while keyword arguments are attributes.
    """
    # This is a wrapper around the IR node creation that hooks into the _builder.OpRecorder
    # tracer so that all nodes created are recorded the same way as if we were to use
    # onnxscript ops directly.

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
        for attr in ir_convenience.convert_attributes(kwargs)  # type: ignore[arg-type]
        if attr.value is not None  # type: ignore[union-attr]
    ]
    tracer.nodes.append(
        node := ir.Node(
            domain,
            op_type,
            inputs=inputs,
            attributes=attributes,
            num_outputs=len(dtypes),
            version=version,
            metadata_props=metadata_props,
        )
    )
    # Set the dtypes for the outputs. We set them here because the graph builder
    # Uses PyTorch types which are sometimes inaccurate when they are ONNX only
    # types like float4e2m1.
    for value, dtype in zip(node.outputs, dtypes):
        value.dtype = ir.DataType(dtype)
        # The shape is set by the graph builder. We don't need to set it here.
    return node.outputs


@onnx_impl(torch.ops.onnx_symbolic._symbolic.default, no_compile=True)
def onnx_symbolic_symbolic(
    inputs: Sequence[ir.Value | None],
    op_type: str,
    onnx_dtype: int,
    *,
    shape: Sequence[int | ir.Value],
    attr_keys: Sequence[str],
    attr_types: Sequence[str],
    attr_pos: Sequence[tuple[int, int]],
    attr_ints: Sequence[int],
    attr_floats: Sequence[float],
    attr_strs: Sequence[str],
    metadata_props_keys: Sequence[str] = (),
    metadata_props_values: Sequence[str] = (),
    domain: str = "",
    version: int | None = None,
) -> ir.Value:
    del shape  # Unused. The shapes are set by the graph builder
    encoded = _symbolic_impl.EncodedAttrs(
        attr_keys=list(attr_keys),
        attr_types=list(attr_types),
        attr_pos=list(attr_pos),
        attr_ints=list(attr_ints),
        attr_floats=list(attr_floats),
        attr_strs=list(attr_strs),
    )
    attrs = encoded.to_dict()
    return _call_symbolic_op(
        op_type,
        domain,
        inputs,
        attrs,
        dtypes=[onnx_dtype],
        version=version,
        metadata_props=dict(zip(metadata_props_keys, metadata_props_values)),
    )[0]


@onnx_impl(torch.ops.onnx_symbolic._symbolic_multi_out.default, no_compile=True)
def onnx_symbolic_symbolic_multi_out(
    inputs: Sequence[ir.Value | None],
    op_type: str,
    onnx_dtypes: Sequence[int],
    *,
    shapes: Sequence[Sequence[int | ir.Value]],
    attr_keys: Sequence[str],
    attr_types: Sequence[str],
    attr_pos: Sequence[tuple[int, int]],
    attr_ints: Sequence[int],
    attr_floats: Sequence[float],
    attr_strs: Sequence[str],
    metadata_props_keys: Sequence[str] = (),
    metadata_props_values: Sequence[str] = (),
    domain: str = "",
    version: int | None = None,
) -> Sequence[ir.Value]:
    del shapes  # Unused. The shapes are set by the graph builder
    encoded = _symbolic_impl.EncodedAttrs(
        attr_keys=list(attr_keys),
        attr_types=list(attr_types),
        attr_pos=list(attr_pos),
        attr_ints=list(attr_ints),
        attr_floats=list(attr_floats),
        attr_strs=list(attr_strs),
    )
    attrs = encoded.to_dict()
    return _call_symbolic_op(
        op_type,
        domain,
        inputs,
        attrs,
        dtypes=onnx_dtypes,
        version=version,
        metadata_props=dict(zip(metadata_props_keys, metadata_props_values)),
    )

```



## High-Level Overview

"""Implementation for higher-order operators."""from __future__ import annotationsfrom typing import TYPE_CHECKINGfrom onnxscript.ir import convenience as ir_convenienceimport torchfrom torch.onnx._internal._lazy_import import onnxscript_ir as irfrom torch.onnx._internal.exporter import _corefrom torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_implfrom torch.onnx.ops import _symbolic_implif TYPE_CHECKING:    from collections.abc import Sequencedef _call_symbolic_op(    op_type: str,    domain: str,    args: Sequence[ir.Value | None],    kwargs: dict[str, int | float | str | bool | list[int] | list[float] | list[str]],    dtypes: Sequence[int],    version: int | None,    metadata_props: dict[str, str] | None,) -> Sequence[ir.Value]:

This Python file contains 0 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_call_symbolic_op`, `onnx_symbolic_symbolic`, `onnx_symbolic_symbolic_multi_out`

**Key imports**: annotations, TYPE_CHECKING, convenience as ir_convenience, torch, onnxscript_ir as ir, _core, onnx_impl, _symbolic_impl, Sequence


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/onnx/_internal/exporter/_torchlib/ops`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `typing`: TYPE_CHECKING
- `onnxscript.ir`: convenience as ir_convenience
- `torch`
- `torch.onnx._internal._lazy_import`: onnxscript_ir as ir
- `torch.onnx._internal.exporter`: _core
- `torch.onnx._internal.exporter._torchlib._torchlib_registry`: onnx_impl
- `torch.onnx.ops`: _symbolic_impl
- `collections.abc`: Sequence


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
- [`hop.py_docs.md`](./hop.py_docs.md)


## Cross-References

- **File Documentation**: `symbolic.py_docs.md`
- **Keyword Index**: `symbolic.py_kw.md`
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
- [`hop.py_docs.md_docs.md`](./hop.py_docs.md_docs.md)
- [`nn.py_kw.md_docs.md`](./nn.py_kw.md_docs.md)
- [`hop.py_kw.md_docs.md`](./hop.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`core.py_kw.md_docs.md`](./core.py_kw.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `symbolic.py_docs.md_docs.md`
- **Keyword Index**: `symbolic.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
