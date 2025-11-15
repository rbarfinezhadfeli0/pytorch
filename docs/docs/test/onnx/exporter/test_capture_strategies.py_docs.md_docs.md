# Documentation: `docs/test/onnx/exporter/test_capture_strategies.py_docs.md`

## File Metadata

- **Path**: `docs/test/onnx/exporter/test_capture_strategies.py_docs.md`
- **Size**: 6,661 bytes (6.50 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/onnx/exporter/test_capture_strategies.py`

## File Metadata

- **Path**: `test/onnx/exporter/test_capture_strategies.py`
- **Size**: 2,202 bytes (2.15 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: onnx"]
"""Unit tests for the _capture_strategies module."""

from __future__ import annotations

import torch
from torch.onnx._internal.exporter import _capture_strategies
from torch.testing._internal import common_utils


@common_utils.instantiate_parametrized_tests
class ExportStrategiesTest(common_utils.TestCase):
    @common_utils.parametrize(
        "strategy_cls",
        [
            _capture_strategies.TorchExportStrictStrategy,
            _capture_strategies.TorchExportNonStrictStrategy,
            _capture_strategies.TorchExportDraftExportStrategy,
        ],
        name_fn=lambda strategy_cls: strategy_cls.__name__,
    )
    def test_jit_isinstance(self, strategy_cls):
        class Model(torch.nn.Module):
            def forward(self, a, b):
                if torch.jit.isinstance(a, torch.Tensor):
                    return a.cos()
                return b.sin()

        model = Model()
        a = torch.tensor(0.0)
        b = torch.tensor(1.0)

        result = strategy_cls()(model, (a, b), kwargs=None, dynamic_shapes=None)
        ep = result.exported_program
        assert ep is not None
        torch.testing.assert_close(ep.module()(a, b), model(a, b))

    def test_draft_export_on_data_dependent_model(self):
        class Model(torch.nn.Module):
            def forward(self, a, b):
                if a.sum() > 0:
                    return a.cos()
                # The branch is expected to be specialized and a warning is logged
                return b.sin()

        model = Model()
        a = torch.tensor(0.0)
        b = torch.tensor(1.0)

        strategy = _capture_strategies.TorchExportDraftExportStrategy()
        with self.assertLogs("torch.export", level="WARNING") as cm:
            result = strategy(model, (a, b), kwargs=None, dynamic_shapes=None)
            expected_warning = "1 issue(s) found during export, and it was not able to soundly produce a graph."
            self.assertIn(expected_warning, str(cm.output))
        ep = result.exported_program
        assert ep is not None
        torch.testing.assert_close(ep.module()(a, b), model(a, b))


if __name__ == "__main__":
    common_utils.run_tests()

```



## High-Level Overview

"""Unit tests for the _capture_strategies module."""from __future__ import annotationsimport torchfrom torch.onnx._internal.exporter import _capture_strategiesfrom torch.testing._internal import common_utils@common_utils.instantiate_parametrized_testsclass ExportStrategiesTest(common_utils.TestCase):    @common_utils.parametrize(        "strategy_cls",        [            _capture_strategies.TorchExportStrictStrategy,            _capture_strategies.TorchExportNonStrictStrategy,            _capture_strategies.TorchExportDraftExportStrategy,        ],        name_fn=lambda strategy_cls: strategy_cls.__name__,    )    def test_jit_isinstance(self, strategy_cls):        class Model(torch.nn.Module):            def forward(self, a, b):                if torch.jit.isinstance(a, torch.Tensor):                    return a.cos()                return b.sin()        model = Model()        a = torch.tensor(0.0)        b = torch.tensor(1.0)        result = strategy_cls()(model, (a, b), kwargs=None, dynamic_shapes=None)        ep = result.exported_program        assert ep is not None        torch.testing.assert_close(ep.module()(a, b), model(a, b))    def test_draft_export_on_data_dependent_model(self):        class Model(torch.nn.Module):            def forward(self, a, b):                if a.sum() > 0:                    return a.cos()                # The branch is expected to be specialized and a warning is logged                return b.sin()        model = Model()        a = torch.tensor(0.0)        b = torch.tensor(1.0)        strategy = _capture_strategies.TorchExportDraftExportStrategy()

This Python file contains 3 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ExportStrategiesTest`, `Model`, `Model`

**Functions defined**: `test_jit_isinstance`, `forward`, `test_draft_export_on_data_dependent_model`, `forward`

**Key imports**: annotations, torch, _capture_strategies, common_utils


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/onnx/exporter`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `torch`
- `torch.onnx._internal.exporter`: _capture_strategies
- `torch.testing._internal`: common_utils


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

This is a test file. Run it with:

```bash
python test/onnx/exporter/test_capture_strategies.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/onnx/exporter`):

- [`test_building.py_docs.md`](./test_building.py_docs.md)
- [`test_hf_models_e2e.py_docs.md`](./test_hf_models_e2e.py_docs.md)
- [`test_verification.py_docs.md`](./test_verification.py_docs.md)
- [`test_dynamic_shapes.py_docs.md`](./test_dynamic_shapes.py_docs.md)
- [`test_small_models_e2e.py_docs.md`](./test_small_models_e2e.py_docs.md)
- [`test_core.py_docs.md`](./test_core.py_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`test_ir_passes.py_docs.md`](./test_ir_passes.py_docs.md)
- [`test_tensors.py_docs.md`](./test_tensors.py_docs.md)


## Cross-References

- **File Documentation**: `test_capture_strategies.py_docs.md`
- **Keyword Index**: `test_capture_strategies.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/onnx/exporter`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/onnx/exporter`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


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

This is a test file. Run it with:

```bash
python docs/test/onnx/exporter/test_capture_strategies.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/onnx/exporter`):

- [`test_tensors.py_docs.md_docs.md`](./test_tensors.py_docs.md_docs.md)
- [`test_hf_models_e2e.py_kw.md_docs.md`](./test_hf_models_e2e.py_kw.md_docs.md)
- [`test_ir_passes.py_kw.md_docs.md`](./test_ir_passes.py_kw.md_docs.md)
- [`test_dynamic_shapes.py_kw.md_docs.md`](./test_dynamic_shapes.py_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`test_api.py_docs.md_docs.md`](./test_api.py_docs.md_docs.md)
- [`test_dynamic_shapes.py_docs.md_docs.md`](./test_dynamic_shapes.py_docs.md_docs.md)
- [`test_building.py_docs.md_docs.md`](./test_building.py_docs.md_docs.md)
- [`test_core.py_docs.md_docs.md`](./test_core.py_docs.md_docs.md)
- [`test_ir_passes.py_docs.md_docs.md`](./test_ir_passes.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_capture_strategies.py_docs.md_docs.md`
- **Keyword Index**: `test_capture_strategies.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
