# Documentation: `docs/torch/onnx/_internal/exporter/_testing.py_docs.md`

## File Metadata

- **Path**: `docs/torch/onnx/_internal/exporter/_testing.py_docs.md`
- **Size**: 7,139 bytes (6.97 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `torch/onnx/_internal/exporter/_testing.py`

## File Metadata

- **Path**: `torch/onnx/_internal/exporter/_testing.py`
- **Size**: 3,993 bytes (3.90 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This appears to be a **test file**.

## Original Source

```python
"""Test utilities for ONNX export."""

from __future__ import annotations


__all__ = ["assert_onnx_program"]

from typing import Any, Literal, TYPE_CHECKING

import torch
from torch.utils import _pytree


if TYPE_CHECKING:
    from torch.onnx._internal.exporter import _onnx_program


def assert_onnx_program(
    program: _onnx_program.ONNXProgram,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    strategy: str | None = "TorchExportNonStrictStrategy",
    backend: Literal["onnxruntime", "reference"] = "onnxruntime",
) -> None:
    """Assert that the ONNX model produces the same output as the PyTorch ExportedProgram.

    Args:
        program: The ``ONNXProgram`` to verify.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        args: The positional arguments to pass to the program.
            If None, the default example inputs in the ExportedProgram will be used.
        kwargs: The keyword arguments to pass to the program.
            If None, the default example inputs in the ExportedProgram will be used.
        strategy: Assert the capture strategy used to export the program. Values can be
            class names like "TorchExportNonStrictStrategy".
            If None, the strategy is not asserted.
        backend: The backend to use for evaluating the ONNX program.
            Supported values are "onnxruntime" and "reference".
    """
    if strategy is not None:
        if program._capture_strategy != strategy:
            raise ValueError(
                f"Expected strategy '{strategy}' is used to capture the exported program, "
                f"but got '{program._capture_strategy}'."
            )
    exported_program = program.exported_program
    if exported_program is None:
        raise ValueError(
            "The ONNXProgram does not contain an ExportedProgram. "
            "To verify the ONNX program, initialize ONNXProgram with an ExportedProgram, "
            "or assign the ExportedProgram to the ONNXProgram.exported_program attribute."
        )
    if args is None and kwargs is None:
        # User did not provide example inputs, use the default example inputs
        if exported_program.example_inputs is None:
            raise ValueError(
                "No example inputs provided and the exported_program does not contain example inputs. "
                "Please provide arguments to verify the ONNX program."
            )
        args, kwargs = exported_program.example_inputs
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    torch_module = exported_program.module()
    torch_outputs, _ = _pytree.tree_flatten(torch_module(*args, **kwargs))
    # ONNX outputs are always real, so we need to convert torch complex outputs to real representations
    torch_outputs_adapted = []
    for output in torch_outputs:
        # ONNX graph does not support None outputs, so we skip them
        if output is None:
            continue
        if not isinstance(output, torch.Tensor):
            torch_outputs_adapted.append(torch.tensor(output))
        elif torch.is_complex(output):
            torch_outputs_adapted.append(torch.view_as_real(output))
        else:
            torch_outputs_adapted.append(output)

    # Obtain the ONNX outputs using the specified backend
    if backend == "onnxruntime":
        onnx_outputs = program(*args, **kwargs)
    elif backend == "reference":
        onnx_outputs = program.call_reference(*args, **kwargs)
    else:
        raise ValueError(
            f"Unsupported backend '{backend}'. Supported backends are 'onnxruntime' and 'reference'."
        )

    # TODO(justinchuby): Include output names in the error message
    torch.testing.assert_close(
        tuple(onnx_outputs),
        tuple(torch_outputs_adapted),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
        check_device=False,
    )

```



## High-Level Overview

"""Test utilities for ONNX export."""from __future__ import annotations__all__ = ["assert_onnx_program"]from typing import Any, Literal, TYPE_CHECKINGimport torchfrom torch.utils import _pytreeif TYPE_CHECKING:    from torch.onnx._internal.exporter import _onnx_programdef assert_onnx_program(    program: _onnx_program.ONNXProgram,    *,    rtol: float | None = None,    atol: float | None = None,    args: tuple[Any, ...] | None = None,    kwargs: dict[str, Any] | None = None,    strategy: str | None = "TorchExportNonStrictStrategy",    backend: Literal["onnxruntime", "reference"] = "onnxruntime",) -> None:

This Python file contains 1 class(es) and 1 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `assert_onnx_program`

**Key imports**: annotations, Any, Literal, TYPE_CHECKING, torch, _pytree, _onnx_program


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/onnx/_internal/exporter`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `typing`: Any, Literal, TYPE_CHECKING
- `torch`
- `torch.utils`: _pytree
- `torch.onnx._internal.exporter`: _onnx_program


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

This is a test file. Run it with:

```bash
python torch/onnx/_internal/exporter/_testing.py
```

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
- [`_tensors.py_docs.md`](./_tensors.py_docs.md)
- [`_dispatching.py_docs.md`](./_dispatching.py_docs.md)


## Cross-References

- **File Documentation**: `_testing.py_docs.md`
- **Keyword Index**: `_testing.py_kw.md`
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

This is a test file. Run it with:

```bash
python docs/torch/onnx/_internal/exporter/_testing.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/onnx/_internal/exporter`):

- [`_onnx_program.py_docs.md_docs.md`](./_onnx_program.py_docs.md_docs.md)
- [`_decomp.py_docs.md_docs.md`](./_decomp.py_docs.md_docs.md)
- [`_flags.py_docs.md_docs.md`](./_flags.py_docs.md_docs.md)
- [`_verification.py_docs.md_docs.md`](./_verification.py_docs.md_docs.md)
- [`_dispatching.py_docs.md_docs.md`](./_dispatching.py_docs.md_docs.md)
- [`_errors.py_kw.md_docs.md`](./_errors.py_kw.md_docs.md)
- [`_schemas.py_kw.md_docs.md`](./_schemas.py_kw.md_docs.md)
- [`_ir_passes.py_kw.md_docs.md`](./_ir_passes.py_kw.md_docs.md)
- [`_compat.py_kw.md_docs.md`](./_compat.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `_testing.py_docs.md_docs.md`
- **Keyword Index**: `_testing.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
