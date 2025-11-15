# Documentation: `docs/torch/onnx/_internal/torchscript_exporter/_globals.py_docs.md`

## File Metadata

- **Path**: `docs/torch/onnx/_internal/torchscript_exporter/_globals.py_docs.md`
- **Size**: 5,732 bytes (5.60 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `torch/onnx/_internal/torchscript_exporter/_globals.py`

## File Metadata

- **Path**: `torch/onnx/_internal/torchscript_exporter/_globals.py`
- **Size**: 2,801 bytes (2.74 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
"""Globals used internally by the ONNX exporter.

Do not use this module outside of `torch.onnx` and its tests.

Be very judicious when adding any new global variables. Do not create new global
variables unless they are absolutely necessary.
"""

import torch._C._onnx as _C_onnx

# This module should only depend on _constants and nothing else in torch.onnx to keep
# dependency direction clean.
from torch.onnx import _constants


class _InternalGlobals:
    """Globals used internally by ONNX exporter.

    NOTE: Be very judicious when adding any new variables. Do not create new
    global variables unless they are absolutely necessary.
    """

    def __init__(self) -> None:
        self._export_onnx_opset_version = _constants.ONNX_DEFAULT_OPSET
        self._training_mode: _C_onnx.TrainingMode = _C_onnx.TrainingMode.EVAL
        self._in_onnx_export: bool = False
        # Whether the user's model is training during export
        self.export_training: bool = False
        self.operator_export_type: _C_onnx.OperatorExportTypes = (
            _C_onnx.OperatorExportTypes.ONNX
        )
        self.onnx_shape_inference: bool = True
        self._autograd_inlining: bool = True

    @property
    def training_mode(self) -> _C_onnx.TrainingMode:
        """The training mode for the exporter."""
        return self._training_mode

    @training_mode.setter
    def training_mode(self, training_mode: _C_onnx.TrainingMode) -> None:
        if not isinstance(training_mode, _C_onnx.TrainingMode):
            raise TypeError(
                "training_mode must be of type 'torch.onnx.TrainingMode'. This is "
                "likely a bug in torch.onnx."
            )
        self._training_mode = training_mode

    @property
    def export_onnx_opset_version(self) -> int:
        """Opset version used during export."""
        return self._export_onnx_opset_version

    @export_onnx_opset_version.setter
    def export_onnx_opset_version(self, value: int) -> None:
        self._export_onnx_opset_version = value

    @property
    def in_onnx_export(self) -> bool:
        """Whether it is in the middle of ONNX export."""
        return self._in_onnx_export

    @in_onnx_export.setter
    def in_onnx_export(self, value: bool) -> None:
        if type(value) is not bool:
            raise TypeError("in_onnx_export must be a boolean")
        self._in_onnx_export = value

    @property
    def autograd_inlining(self) -> bool:
        """Whether Autograd must be inlined."""
        return self._autograd_inlining

    @autograd_inlining.setter
    def autograd_inlining(self, value: bool) -> None:
        if type(value) is not bool:
            raise TypeError("autograd_inlining must be a boolean")
        self._autograd_inlining = value


GLOBALS = _InternalGlobals()

```



## High-Level Overview

"""Globals used internally by the ONNX exporter.Do not use this module outside of `torch.onnx` and its tests.Be very judicious when adding any new global variables. Do not create new globalvariables unless they are absolutely necessary.

This Python file contains 1 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_InternalGlobals`

**Functions defined**: `__init__`, `training_mode`, `training_mode`, `export_onnx_opset_version`, `export_onnx_opset_version`, `in_onnx_export`, `in_onnx_export`, `autograd_inlining`, `autograd_inlining`

**Key imports**: torch._C._onnx as _C_onnx, _constants


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/onnx/_internal/torchscript_exporter`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch._C._onnx as _C_onnx`
- `torch.onnx`: _constants


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

- **File Documentation**: `_globals.py_docs.md`
- **Keyword Index**: `_globals.py_kw.md`
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

- **Object-Oriented Design**: Uses classes and constructors


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

- **File Documentation**: `_globals.py_docs.md_docs.md`
- **Keyword Index**: `_globals.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
