# Documentation: `docs/torch/onnx/_internal/torchscript_exporter/symbolic_opset15.py_docs.md`

## File Metadata

- **Path**: `docs/torch/onnx/_internal/torchscript_exporter/symbolic_opset15.py_docs.md`
- **Size**: 6,320 bytes (6.17 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `torch/onnx/_internal/torchscript_exporter/symbolic_opset15.py`

## File Metadata

- **Path**: `torch/onnx/_internal/torchscript_exporter/symbolic_opset15.py`
- **Size**: 2,892 bytes (2.82 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
# mypy: allow-untyped-defs
"""This file exports ONNX ops for opset 15.

Note [ONNX operators that are added/updated in opset 15]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/master/docs/Changelog.md#version-15-of-the-default-onnx-operator-set
New operators:
    Bernoulli
    CastLike
    Optional
    OptionalGetElement
    OptionalHasElement

Updated operators:
    BatchNormalization https://github.com/onnx/onnx/pull/3545
                        Backwards compatible
                        TODO: test coverage for mixed types inputs.
    Pow                https://github.com/onnx/onnx/pull/3412
                        Backwards compatible
                        TODO: bfloat16 support.
    Shape              https://github.com/onnx/onnx/pull/3580
                        Backwards compatible
                        TODO: optional start/end attribute.
"""

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in README.md

import functools

import torch
from torch import _C
from torch.onnx._internal.torchscript_exporter import (
    jit_utils,
    registration,
    symbolic_helper,
    symbolic_opset9 as opset9,
)


_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=15)


@_onnx_symbolic("aten::__is_")
def aten__is_(g: jit_utils.GraphContext, self, other):
    if symbolic_helper._is_none(other):
        if isinstance(self.type(), _C.OptionalType):
            none = g.op("OptionalHasElement", self)
            return g.op("Not", none)
        else:
            return g.op("Constant", value_t=torch.BoolTensor([0]))
    return opset9.eq(g, self, other)


@_onnx_symbolic("aten::__isnot_")
@opset9.wrap_logical_op_with_negation  # type: ignore[has-type]
def aten__isnot_(g: jit_utils.GraphContext, self, other):
    return aten__is_(g, self, other)


@_onnx_symbolic("aten::bernoulli")
def bernoulli(g: jit_utils.GraphContext, input, p=None, generator=None, out=None):
    if out is not None and not symbolic_helper._is_none(out):
        symbolic_helper._unimplemented(
            "Bernoulli", "out parameter is not supported for bernoulli", input
        )
    if generator is not None and not symbolic_helper._is_none(generator):
        symbolic_helper._unimplemented(
            "Bernoulli", "generator is not supported for bernoulli", input
        )
    if p is None or symbolic_helper._is_none(p):
        return g.op("Bernoulli", input)
    return opset9.bernoulli(g, input, p, generator, out)


@_onnx_symbolic("prim::unchecked_cast")
def prim_unchecked_cast(g: jit_utils.GraphContext, self):
    # exists to refine the type of the Value
    # if x is Optional[Tensor], unchecked_cast will cast
    # x to Tensor, so the rest of the graph knows that x is a Tensor.
    if isinstance(self.type(), _C.OptionalType):
        return g.op("OptionalGetElement", self)

    return self

```



## High-Level Overview

"""This file exports ONNX ops for opset 15.Note [ONNX operators that are added/updated in opset 15]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~https://github.com/onnx/onnx/blob/master/docs/Changelog.md#version-15-of-the-default-onnx-operator-setNew operators:    Bernoulli    CastLike    Optional    OptionalGetElement    OptionalHasElementUpdated operators:    BatchNormalization https://github.com/onnx/onnx/pull/3545                        Backwards compatible                        TODO: test coverage for mixed types inputs.    Pow                https://github.com/onnx/onnx/pull/3412                        Backwards compatible                        TODO: bfloat16 support.    Shape              https://github.com/onnx/onnx/pull/3580                        Backwards compatible                        TODO: optional start/end attribute.

This Python file contains 0 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `aten__is_`, `aten__isnot_`, `bernoulli`, `prim_unchecked_cast`

**Key imports**: functools, torch, _C


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/onnx/_internal/torchscript_exporter`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `torch`


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

- **File Documentation**: `symbolic_opset15.py_docs.md`
- **Keyword Index**: `symbolic_opset15.py_kw.md`
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

- **File Documentation**: `symbolic_opset15.py_docs.md_docs.md`
- **Keyword Index**: `symbolic_opset15.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
