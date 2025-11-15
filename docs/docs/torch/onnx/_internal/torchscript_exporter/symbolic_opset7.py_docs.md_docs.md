# Documentation: `docs/torch/onnx/_internal/torchscript_exporter/symbolic_opset7.py_docs.md`

## File Metadata

- **Path**: `docs/torch/onnx/_internal/torchscript_exporter/symbolic_opset7.py_docs.md`
- **Size**: 4,934 bytes (4.82 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `torch/onnx/_internal/torchscript_exporter/symbolic_opset7.py`

## File Metadata

- **Path**: `torch/onnx/_internal/torchscript_exporter/symbolic_opset7.py`
- **Size**: 2,192 bytes (2.14 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
# mypy: allow-untyped-defs
"""
Note [ONNX operators that are added/updated from opset 7 to opset 8]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
New operators:
  Expand

Updated operators:
  Min, Max, Sum, Mean: supports multidirectional broadcasting.
  MaxPool: added optional indices output.
  Scan
"""

import functools
import warnings

from torch.onnx._internal.torchscript_exporter import (
    jit_utils,
    registration,
    symbolic_helper,
    symbolic_opset9 as opset9,
)


_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=7)

block_listed_operators = (
    "scan",
    "expand",
    "expand_as",
    "meshgrid",
    "adaptive_max_pool1d",
    "adaptive_max_pool2d",
    "adaptive_max_pool3d",
    "max_pool1d_with_indices",
    "max_pool2d_with_indices",
    "max_pool3d_with_indices",
)


# NOTE: max, min, sum, mean: broadcasting is not supported in opset 7.
# torch.max (same for torch.min) actually has two interfaces smashed together:
# torch.max(x, dim, keepdim) and torch.max(x, y)
@_onnx_symbolic("aten::max")
def max(g: jit_utils.GraphContext, self, dim_or_y=None, keepdim=None):
    # torch.max(input, other)
    if keepdim is None and dim_or_y is not None:
        warnings.warn(
            "Multidirectional broadcasting is not supported in opset 7. "
            "This might cause the onnx model to be incorrect, if inputs to max operators "
            "have different shapes",
            stacklevel=2,
        )
    return opset9.max(g, self, dim_or_y, keepdim)


@_onnx_symbolic("aten::min")
def min(g: jit_utils.GraphContext, self, dim_or_y=None, keepdim=None):
    # torch.min(input, other)
    if keepdim is None and dim_or_y is not None:
        warnings.warn(
            "Multidirectional broadcasting is not supported in opset 7. "
            "This might cause the onnx model to be incorrect, if inputs to min operators "
            "have different shapes",
            stacklevel=2,
        )
    return opset9.min(g, self, dim_or_y, keepdim)


for block_listed_op in block_listed_operators:
    _onnx_symbolic(f"aten::{block_listed_op}")(
        symbolic_helper._block_list_in_opset(block_listed_op)
    )

```



## High-Level Overview

"""Note [ONNX operators that are added/updated from opset 7 to opset 8]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~New operators:  ExpandUpdated operators:  Min, Max, Sum, Mean: supports multidirectional broadcasting.  MaxPool: added optional indices output.  Scan

This Python file contains 0 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `max`, `min`

**Key imports**: functools, warnings


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/onnx/_internal/torchscript_exporter`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `warnings`


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

- **File Documentation**: `symbolic_opset7.py_docs.md`
- **Keyword Index**: `symbolic_opset7.py_kw.md`
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

- **File Documentation**: `symbolic_opset7.py_docs.md_docs.md`
- **Keyword Index**: `symbolic_opset7.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
