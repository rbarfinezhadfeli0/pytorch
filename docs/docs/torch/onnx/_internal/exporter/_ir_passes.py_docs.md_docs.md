# Documentation: `docs/torch/onnx/_internal/exporter/_ir_passes.py_docs.md`

## File Metadata

- **Path**: `docs/torch/onnx/_internal/exporter/_ir_passes.py_docs.md`
- **Size**: 8,143 bytes (7.95 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/onnx/_internal/exporter/_ir_passes.py`

## File Metadata

- **Path**: `torch/onnx/_internal/exporter/_ir_passes.py`
- **Size**: 4,940 bytes (4.82 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from torch.onnx._internal._lazy_import import onnxscript_ir as ir
from torch.onnx._internal.exporter import _constants


if TYPE_CHECKING:
    from collections.abc import Sequence


# The opset domain for ONNX operators
_ONNX_DOMAIN = ""

logger = logging.getLogger(__name__)


def rename_inputs(model: ir.Model, new_names: Sequence[str]) -> None:
    # TODO: Ensure the names do not have duplicates
    for input, new_name in zip(model.graph.inputs, new_names):
        input.metadata_props["pkg.torch.onnx.original_node_name"] = str(input.name)
        input.name = new_name


def rename_outputs(model: ir.Model, new_names: Sequence[str]) -> None:
    for output, new_name in zip(model.graph.outputs, new_names):
        output.metadata_props["pkg.torch.onnx.original_node_name"] = str(output.name)
        output.name = new_name


def _all_values(model: ir.Model):
    """Yield all values in a model."""
    # Yield all values in the model
    yield from model.graph.inputs
    yield from model.graph.initializers.values()
    for node in ir.traversal.RecursiveGraphIterator(model.graph):
        yield from node.outputs
    # Yield all values in functions
    for function in model.functions.values():
        yield from function.inputs
        for node in ir.traversal.RecursiveGraphIterator(function):
            yield from node.outputs


def _replace_names(shape_expr: str, rename_mapping: dict[str, str]) -> str:
    """Replace all known names in a shape expression with new names."""
    for old_name, new_name in rename_mapping.items():
        shape_expr = re.sub(
            rf"(?<!\w){re.escape(old_name)}(?!\w)", new_name, shape_expr
        )
    return shape_expr


def rename_axis(model: ir.Model, rename_mapping: dict[str, str]) -> None:
    """Rename dynamic axes in a model according to the specified dynamic_axes names."""

    # NOTE: Mapping needs to be srted by length because the shape expression
    # could have multiple ways to be expressed, for example,
    # {"s1": sequence_length, "s11": "past_sequence_length", "s1 + s11": "masked_sequence_length"}
    # We prefer the replacement starts from the longest match.
    sorted_rename_mapping = dict(
        sorted(rename_mapping.items(), key=lambda item: len(item[0]), reverse=True)
    )
    for value in _all_values(model):
        if value.shape is None:
            continue
        new_shape = []
        changed = False
        for dim in value.shape:
            if not isinstance(dim, ir.SymbolicDim):
                new_shape.append(dim)
                continue
            dim_name = dim.value
            if dim_name in sorted_rename_mapping:
                # pyrefly: ignore
                new_shape.append(sorted_rename_mapping[dim_name])
                changed = True
            elif dim_name is not None:
                # For example: "2*s1", "s1+1", "s1-1", "s1*s2", "s1/s2"
                new_name = _replace_names(dim_name, sorted_rename_mapping)
                new_shape.append(new_name)
                if new_name != dim_name:
                    changed = True
            else:
                new_shape.append(None)
        if changed:
            value.shape = ir.Shape(new_shape)


def _maybe_set_opset_version(
    opset_imports: dict[str, int], domain: str, version: int | None
) -> None:
    """Set the opset version for the domain."""
    if domain in opset_imports and opset_imports[domain] != 1:
        # Already set
        return
    if domain == _ONNX_DOMAIN:
        opset_imports[domain] = _constants.TORCHLIB_OPSET
        return
    if version is None:
        # We don't know the opset version, so set it to 1
        # This is valid for the custom function domains like "pkg.torch.__subgraph__"
        opset_imports[domain] = 1
        return
    # Set the known opset version for the domain
    opset_imports[domain] = version


def add_opset_imports(model: ir.Model) -> None:
    """Collect all opsets used and add opset imports to the model and functions."""
    for node in ir.traversal.RecursiveGraphIterator(model.graph):
        domain = node.domain
        _maybe_set_opset_version(model.opset_imports, domain, node.version)

    for function in model.functions.values():
        for node in ir.traversal.RecursiveGraphIterator(function):
            domain = node.domain
            _maybe_set_opset_version(function.opset_imports, domain, node.version)
        for domain, version in function.opset_imports.items():
            # Add all opsets used in the function to the model, because ONNX Runtime
            # does not handle adding the opset imports to the model after inlining during inference.
            # This should happen after all opsets are collected for the function from its nodes.
            _maybe_set_opset_version(model.opset_imports, domain, version)

```



## High-Level Overview

"""Yield all values in a model."""    # Yield all values in the model    yield from model.graph.inputs    yield from model.graph.initializers.values()    for node in ir.traversal.RecursiveGraphIterator(model.graph):        yield from node.outputs    # Yield all values in functions    for function in model.functions.values():        yield from function.inputs        for node in ir.traversal.RecursiveGraphIterator(function):            yield from node.outputsdef _replace_names(shape_expr: str, rename_mapping: dict[str, str]) -> str:

This Python file contains 0 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `rename_inputs`, `rename_outputs`, `_all_values`, `_replace_names`, `rename_axis`, `_maybe_set_opset_version`, `add_opset_imports`

**Key imports**: annotations, logging, re, TYPE_CHECKING, onnxscript_ir as ir, _constants, Sequence


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/onnx/_internal/exporter`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `logging`
- `re`
- `typing`: TYPE_CHECKING
- `torch.onnx._internal._lazy_import`: onnxscript_ir as ir
- `torch.onnx._internal.exporter`: _constants
- `collections.abc`: Sequence


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`torch/onnx/_internal/exporter`):

- [`_registration.py_docs.md`](./_registration.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_flags.py_docs.md`](./_flags.py_docs.md)
- [`_building.py_docs.md`](./_building.py_docs.md)
- [`_analysis.py_docs.md`](./_analysis.py_docs.md)
- [`_verification.py_docs.md`](./_verification.py_docs.md)
- [`_capture_strategies.py_docs.md`](./_capture_strategies.py_docs.md)
- [`_tensors.py_docs.md`](./_tensors.py_docs.md)
- [`_dispatching.py_docs.md`](./_dispatching.py_docs.md)


## Cross-References

- **File Documentation**: `_ir_passes.py_docs.md`
- **Keyword Index**: `_ir_passes.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/onnx/_internal/exporter`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/onnx/_internal/exporter`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`docs/torch/onnx/_internal/exporter`):

- [`_onnx_program.py_docs.md_docs.md`](./_onnx_program.py_docs.md_docs.md)
- [`_decomp.py_docs.md_docs.md`](./_decomp.py_docs.md_docs.md)
- [`_testing.py_docs.md_docs.md`](./_testing.py_docs.md_docs.md)
- [`_flags.py_docs.md_docs.md`](./_flags.py_docs.md_docs.md)
- [`_verification.py_docs.md_docs.md`](./_verification.py_docs.md_docs.md)
- [`_dispatching.py_docs.md_docs.md`](./_dispatching.py_docs.md_docs.md)
- [`_errors.py_kw.md_docs.md`](./_errors.py_kw.md_docs.md)
- [`_schemas.py_kw.md_docs.md`](./_schemas.py_kw.md_docs.md)
- [`_ir_passes.py_kw.md_docs.md`](./_ir_passes.py_kw.md_docs.md)
- [`_compat.py_kw.md_docs.md`](./_compat.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `_ir_passes.py_docs.md_docs.md`
- **Keyword Index**: `_ir_passes.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
