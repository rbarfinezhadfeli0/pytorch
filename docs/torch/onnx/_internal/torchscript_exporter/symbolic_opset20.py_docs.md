# Documentation: `torch/onnx/_internal/torchscript_exporter/symbolic_opset20.py`

## File Metadata

- **Path**: `torch/onnx/_internal/torchscript_exporter/symbolic_opset20.py`
- **Size**: 2,462 bytes (2.40 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
# mypy: allow-untyped-defs
"""This file exports ONNX ops for opset 20.

Note [ONNX Operators that are added/updated in opset 20]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-20-of-the-default-onnx-operator-set
New operators:
    AffineGrid
    ConstantOfShape
    DFT
    Gelu
    GridSample
    ImageDecoder
    IsInf
    IsNaN
    ReduceMax
    ReduceMin
    RegexFullMatch
    StringConcat
    StringSplit
"""

import functools

import torch.nn.functional as F
from torch import _C
from torch.onnx._internal.torchscript_exporter import (
    jit_utils,
    registration,
    symbolic_helper,
)


# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

__all__ = ["_grid_sampler", "_affine_grid_generator", "gelu"]


def convert_grid_sample_mode(mode_s):
    return (
        "linear" if mode_s == "bilinear" else "cubic" if mode_s == "bicubic" else mode_s
    )


_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=20)


@_onnx_symbolic("aten::grid_sampler")
@symbolic_helper.parse_args("v", "v", "i", "i", "b")
def _grid_sampler(
    g: jit_utils.GraphContext,
    input: _C.Value,
    grid: _C.Value,
    mode_enum: int,
    padding_mode_enum: int,
    align_corners: bool,
):
    mode_s = {v: k for k, v in F.GRID_SAMPLE_INTERPOLATION_MODES.items()}[mode_enum]  # type: ignore[call-arg, index]
    # mode string changes at https://onnx.ai/onnx/operators/text_diff_GridSample_16_20.html
    mode_s = convert_grid_sample_mode(mode_s)
    padding_mode_s = {v: k for k, v in F.GRID_SAMPLE_PADDING_MODES.items()}[  # type: ignore[call-arg, index]
        padding_mode_enum  # type: ignore[index]
    ]
    return g.op(
        "GridSample",
        input,
        grid,
        align_corners_i=int(align_corners),
        mode_s=mode_s,
        padding_mode_s=padding_mode_s,
    )


@_onnx_symbolic("aten::affine_grid_generator")
@symbolic_helper.parse_args("v", "v", "b")
def _affine_grid_generator(
    g: jit_utils.GraphContext,
    theta: _C.Value,
    size: _C.Value,
    align_corners: bool,
):
    return g.op(
        "AffineGrid",
        theta,
        size,
        align_corners_i=int(align_corners),
    )


@_onnx_symbolic("aten::gelu")
@symbolic_helper.parse_args("v", "s")
def gelu(g: jit_utils.GraphContext, self: _C.Value, approximate: str = "none"):
    return g.op("Gelu", self, approximate_s=approximate)

```



## High-Level Overview

"""This file exports ONNX ops for opset 20.Note [ONNX Operators that are added/updated in opset 20]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-20-of-the-default-onnx-operator-setNew operators:    AffineGrid    ConstantOfShape    DFT    Gelu    GridSample    ImageDecoder    IsInf    IsNaN    ReduceMax    ReduceMin    RegexFullMatch    StringConcat    StringSplit

This Python file contains 0 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `convert_grid_sample_mode`, `_grid_sampler`, `_affine_grid_generator`, `gelu`

**Key imports**: functools, torch.nn.functional as F, _C


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/onnx/_internal/torchscript_exporter`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `torch.nn.functional as F`
- `torch`: _C


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


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
- [`symbolic_opset9.py_docs.md`](./symbolic_opset9.py_docs.md)


## Cross-References

- **File Documentation**: `symbolic_opset20.py_docs.md`
- **Keyword Index**: `symbolic_opset20.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
