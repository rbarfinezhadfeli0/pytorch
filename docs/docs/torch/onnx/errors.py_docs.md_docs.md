# Documentation: `docs/torch/onnx/errors.py_docs.md`

## File Metadata

- **Path**: `docs/torch/onnx/errors.py_docs.md`
- **Size**: 6,329 bytes (6.18 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/onnx/errors.py`

## File Metadata

- **Path**: `torch/onnx/errors.py`
- **Size**: 3,478 bytes (3.40 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""ONNX exporter exceptions."""

from __future__ import annotations


__all__ = [
    "OnnxExporterWarning",
    "SymbolicValueError",
    "UnsupportedOperatorError",
]

import textwrap
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from torch import _C


class OnnxExporterWarning(UserWarning):
    """Warnings in the ONNX exporter."""


class OnnxExporterError(RuntimeError):
    """Errors raised by the ONNX exporter. This is the base class for all exporter errors."""


class UnsupportedOperatorError(OnnxExporterError):
    """Raised when an operator is unsupported by the exporter."""

    # NOTE: This is legacy and is only used by the torchscript exporter
    # Clean up when the torchscript exporter is removed
    def __init__(self, name: str, version: int, supported_version: int | None) -> None:
        if supported_version is not None:
            msg = (
                f"Exporting the operator '{name}' to ONNX opset version {version} "
                "is not supported. Support for this operator was added in version "
                f"{supported_version}, try exporting with this version"
            )
        elif name.startswith(("aten::", "prim::", "quantized::")):
            msg = (
                f"Exporting the operator '{name}' to ONNX opset version {version} "
                "is not supported"
            )
        else:
            msg = (
                f"ONNX export failed on an operator with unrecognized namespace {name}. "
                "If you are trying to export a custom operator, make sure you registered it with "
                "the right domain and version."
            )

        super().__init__(msg)


class SymbolicValueError(OnnxExporterError):
    """Errors around TorchScript values and nodes."""

    # NOTE: This is legacy and is only used by the torchscript exporter
    # Clean up when the torchscript exporter is removed
    def __init__(self, msg: str, value: _C.Value) -> None:
        message = (
            f"{msg}  [Caused by the value '{value}' (type '{value.type()}') in the "
            f"TorchScript graph. The containing node has kind '{value.node().kind()}'.] "
        )

        code_location = value.node().sourceRange()
        if code_location:
            message += f"\n    (node defined in {code_location})"

        try:
            # Add its input and output to the message.
            message += "\n\n"
            message += textwrap.indent(
                (
                    "Inputs:\n"
                    + (
                        "\n".join(
                            f"    #{i}: {input_}  (type '{input_.type()}')"
                            for i, input_ in enumerate(value.node().inputs())
                        )
                        or "    Empty"
                    )
                    + "\n"
                    + "Outputs:\n"
                    + (
                        "\n".join(
                            f"    #{i}: {output}  (type '{output.type()}')"
                            for i, output in enumerate(value.node().outputs())
                        )
                        or "    Empty"
                    )
                ),
                "    ",
            )
        except AttributeError:
            message += (
                " Failed to obtain its input and output for debugging. "
                "Please refer to the TorchScript graph for debugging information."
            )

        super().__init__(message)

```



## High-Level Overview

"""ONNX exporter exceptions."""from __future__ import annotations__all__ = [    "OnnxExporterWarning",    "SymbolicValueError",    "UnsupportedOperatorError",]import textwrapfrom typing import TYPE_CHECKINGif TYPE_CHECKING:    from torch import _Cclass OnnxExporterWarning(UserWarning):

This Python file contains 5 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `OnnxExporterWarning`, `OnnxExporterError`, `UnsupportedOperatorError`, `SymbolicValueError`

**Functions defined**: `__init__`, `__init__`

**Key imports**: annotations, textwrap, TYPE_CHECKING, _C


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/onnx`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `textwrap`
- `typing`: TYPE_CHECKING
- `torch`: _C


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


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

Files in the same folder (`torch/onnx`):

- [`symbolic_opset7.py_docs.md`](./symbolic_opset7.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_flags.py_docs.md`](./_flags.py_docs.md)
- [`symbolic_opset14.py_docs.md`](./symbolic_opset14.py_docs.md)
- [`symbolic_opset11.py_docs.md`](./symbolic_opset11.py_docs.md)
- [`verification.py_docs.md`](./verification.py_docs.md)
- [`symbolic_opset12.py_docs.md`](./symbolic_opset12.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`symbolic_opset20.py_docs.md`](./symbolic_opset20.py_docs.md)
- [`symbolic_opset9.py_docs.md`](./symbolic_opset9.py_docs.md)


## Cross-References

- **File Documentation**: `errors.py_docs.md`
- **Keyword Index**: `errors.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/onnx`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/onnx`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


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

Files in the same folder (`docs/torch/onnx`):

- [`symbolic_opset14.py_docs.md_docs.md`](./symbolic_opset14.py_docs.md_docs.md)
- [`symbolic_opset18.py_kw.md_docs.md`](./symbolic_opset18.py_kw.md_docs.md)
- [`_flags.py_docs.md_docs.md`](./_flags.py_docs.md_docs.md)
- [`symbolic_opset13.py_kw.md_docs.md`](./symbolic_opset13.py_kw.md_docs.md)
- [`symbolic_opset12.py_docs.md_docs.md`](./symbolic_opset12.py_docs.md_docs.md)
- [`symbolic_opset16.py_docs.md_docs.md`](./symbolic_opset16.py_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`symbolic_helper.py_kw.md_docs.md`](./symbolic_helper.py_kw.md_docs.md)
- [`symbolic_opset8.py_docs.md_docs.md`](./symbolic_opset8.py_docs.md_docs.md)
- [`symbolic_opset20.py_docs.md_docs.md`](./symbolic_opset20.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `errors.py_docs.md_docs.md`
- **Keyword Index**: `errors.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
