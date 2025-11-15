# Documentation: `torch/onnx/_internal/exporter/_tensors.py`

## File Metadata

- **Path**: `torch/onnx/_internal/exporter/_tensors.py`
- **Size**: 2,613 bytes (2.55 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""Subclass of ir.Value that supports Python operators."""

# mypy: allow-untyped-defs
from __future__ import annotations

import onnxscript
from onnxscript import ir


class SymbolicTensor(ir.Value):
    """A subclass of ir.Value that supports Python operators."""

    def __init__(
        self,
        opset: onnxscript.values.Opset,
        name: str | None = None,
        shape: ir.Shape | None = None,
        type: ir.TypeProtocol | None = None,
        doc_string: str | None = None,
        const_value: ir.TensorProtocol | None = None,
    ) -> None:
        super().__init__(
            name=name,
            shape=shape,
            type=type,
            doc_string=doc_string,
            const_value=const_value,
        )
        self._opset = opset

    @property
    def rank(self) -> int | None:
        # pyrefly: ignore [missing-attribute]
        if self.shape is None:
            return None
        # pyrefly: ignore [bad-argument-type]
        return len(self.shape)

    # TODO: Implement indexing

    def __mod__(self, other):
        # pyrefly: ignore [missing-attribute]
        if self.dtype in {
            ir.DataType.FLOAT,
            ir.DataType.DOUBLE,
            ir.DataType.FLOAT16,
            ir.DataType.BFLOAT16,
        }:
            return self._opset.Mod(self, other, fmod=1)
        return self._opset.Mod(self, other)

    def __ne__(self, other):
        return self._opset.Not(self._opset.Equal(self, other))

    def __neg__(self):
        return self._opset.Neg(self)

    def __add__(self, other):
        return self._opset.Add(self, other)

    def __radd__(self, other):
        return self._opset.Add(other, self)

    def __rand__(self, other):
        return self._opset.And(other, self)

    def __mul__(self, other):
        return self._opset.Mul(self, other)

    def __rmul__(self, other):
        return self._opset.Mul(other, self)

    def __matmul__(self, other):
        return self._opset.MatMul(self, other)

    def __pow__(self, other):
        return self._opset.Pow(self, other)

    def __sub__(self, other):
        return self._opset.Sub(self, other)

    def __rsub__(self, other):
        return self._opset.Sub(other, self)

    def __truediv__(self, other):
        return self._opset.Div(self, other)

    def __lt__(self, other):
        return self._opset.Less(self, other)

    def __le__(self, other):
        return self._opset.LessOrEqual(self, other)

    def __ge__(self, other):
        return self._opset.GreaterOrEqual(self, other)

    def __gt__(self, other):
        return self._opset.Greater(self, other)

```



## High-Level Overview

"""Subclass of ir.Value that supports Python operators."""# mypy: allow-untyped-defsfrom __future__ import annotationsimport onnxscriptfrom onnxscript import irclass SymbolicTensor(ir.Value):

This Python file contains 3 class(es) and 19 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `SymbolicTensor`

**Functions defined**: `__init__`, `rank`, `__mod__`, `__ne__`, `__neg__`, `__add__`, `__radd__`, `__rand__`, `__mul__`, `__rmul__`, `__matmul__`, `__pow__`, `__sub__`, `__rsub__`, `__truediv__`, `__lt__`, `__le__`, `__ge__`, `__gt__`

**Key imports**: annotations, onnxscript, ir


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/onnx/_internal/exporter`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `onnxscript`


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

Files in the same folder (`torch/onnx/_internal/exporter`):

- [`_registration.py_docs.md`](./_registration.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_flags.py_docs.md`](./_flags.py_docs.md)
- [`_building.py_docs.md`](./_building.py_docs.md)
- [`_ir_passes.py_docs.md`](./_ir_passes.py_docs.md)
- [`_analysis.py_docs.md`](./_analysis.py_docs.md)
- [`_verification.py_docs.md`](./_verification.py_docs.md)
- [`_capture_strategies.py_docs.md`](./_capture_strategies.py_docs.md)
- [`_dispatching.py_docs.md`](./_dispatching.py_docs.md)


## Cross-References

- **File Documentation**: `_tensors.py_docs.md`
- **Keyword Index**: `_tensors.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
