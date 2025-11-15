# Documentation: `docs/test/onnx/test_symbolic_helper.py_docs.md`

## File Metadata

- **Path**: `docs/test/onnx/test_symbolic_helper.py_docs.md`
- **Size**: 6,867 bytes (6.71 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/onnx/test_symbolic_helper.py`

## File Metadata

- **Path**: `test/onnx/test_symbolic_helper.py`
- **Size**: 2,323 bytes (2.27 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: onnx"]
"""Unit tests on `torch.onnx.symbolic_helper`."""

import torch
from torch.onnx import symbolic_helper
from torch.onnx._internal.torchscript_exporter._globals import GLOBALS
from torch.testing._internal import common_utils


class TestHelperFunctions(common_utils.TestCase):
    def setUp(self):
        super().setUp()
        self._initial_training_mode = GLOBALS.training_mode

    def tearDown(self):
        GLOBALS.training_mode = self._initial_training_mode

    @common_utils.parametrize(
        "op_train_mode,export_mode",
        [
            common_utils.subtest(
                [1, torch.onnx.TrainingMode.PRESERVE], name="export_mode_is_preserve"
            ),
            common_utils.subtest(
                [0, torch.onnx.TrainingMode.EVAL],
                name="modes_match_op_train_mode_0_export_mode_eval",
            ),
            common_utils.subtest(
                [1, torch.onnx.TrainingMode.TRAINING],
                name="modes_match_op_train_mode_1_export_mode_training",
            ),
        ],
    )
    def test_check_training_mode_does_not_warn_when(
        self, op_train_mode: int, export_mode: torch.onnx.TrainingMode
    ):
        GLOBALS.training_mode = export_mode
        self.assertNotWarn(
            lambda: symbolic_helper.check_training_mode(op_train_mode, "testop")
        )

    @common_utils.parametrize(
        "op_train_mode,export_mode",
        [
            common_utils.subtest(
                [0, torch.onnx.TrainingMode.TRAINING],
                name="modes_do_not_match_op_train_mode_0_export_mode_training",
            ),
            common_utils.subtest(
                [1, torch.onnx.TrainingMode.EVAL],
                name="modes_do_not_match_op_train_mode_1_export_mode_eval",
            ),
        ],
    )
    def test_check_training_mode_warns_when(
        self,
        op_train_mode: int,
        export_mode: torch.onnx.TrainingMode,
    ):
        with self.assertWarnsRegex(
            UserWarning, f"ONNX export mode is set to {export_mode}"
        ):
            GLOBALS.training_mode = export_mode
            symbolic_helper.check_training_mode(op_train_mode, "testop")


common_utils.instantiate_parametrized_tests(TestHelperFunctions)


if __name__ == "__main__":
    common_utils.run_tests()

```



## High-Level Overview

"""Unit tests on `torch.onnx.symbolic_helper`."""import torchfrom torch.onnx import symbolic_helperfrom torch.onnx._internal.torchscript_exporter._globals import GLOBALSfrom torch.testing._internal import common_utilsclass TestHelperFunctions(common_utils.TestCase):    def setUp(self):        super().setUp()        self._initial_training_mode = GLOBALS.training_mode    def tearDown(self):        GLOBALS.training_mode = self._initial_training_mode    @common_utils.parametrize(        "op_train_mode,export_mode",        [            common_utils.subtest(                [1, torch.onnx.TrainingMode.PRESERVE], name="export_mode_is_preserve"            ),            common_utils.subtest(                [0, torch.onnx.TrainingMode.EVAL],                name="modes_match_op_train_mode_0_export_mode_eval",            ),            common_utils.subtest(                [1, torch.onnx.TrainingMode.TRAINING],                name="modes_match_op_train_mode_1_export_mode_training",            ),        ],    )    def test_check_training_mode_does_not_warn_when(        self, op_train_mode: int, export_mode: torch.onnx.TrainingMode    ):        GLOBALS.training_mode = export_mode        self.assertNotWarn(            lambda: symbolic_helper.check_training_mode(op_train_mode, "testop")        )    @common_utils.parametrize(        "op_train_mode,export_mode",        [            common_utils.subtest(                [0, torch.onnx.TrainingMode.TRAINING],                name="modes_do_not_match_op_train_mode_0_export_mode_training",            ),            common_utils.subtest(                [1, torch.onnx.TrainingMode.EVAL],

This Python file contains 1 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestHelperFunctions`

**Functions defined**: `setUp`, `tearDown`, `test_check_training_mode_does_not_warn_when`, `test_check_training_mode_warns_when`

**Key imports**: torch, symbolic_helper, GLOBALS, common_utils


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/onnx`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.onnx`: symbolic_helper
- `torch.onnx._internal.torchscript_exporter._globals`: GLOBALS
- `torch.testing._internal`: common_utils


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
python test/onnx/test_symbolic_helper.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/onnx`):

- [`test_lazy_import.py_docs.md`](./test_lazy_import.py_docs.md)
- [`onnx_test_common.py_docs.md`](./onnx_test_common.py_docs.md)
- [`pytorch_test_common.py_docs.md`](./pytorch_test_common.py_docs.md)
- [`test_pytorch_onnx_shape_inference.py_docs.md`](./test_pytorch_onnx_shape_inference.py_docs.md)
- [`test_onnxscript_no_runtime.py_docs.md`](./test_onnxscript_no_runtime.py_docs.md)
- [`test_models_onnxruntime.py_docs.md`](./test_models_onnxruntime.py_docs.md)
- [`test_custom_ops.py_docs.md`](./test_custom_ops.py_docs.md)
- [`test_models.py_docs.md`](./test_models.py_docs.md)
- [`test_onnxscript_runtime.py_docs.md`](./test_onnxscript_runtime.py_docs.md)
- [`test_pytorch_onnx_onnxruntime_cuda.py_docs.md`](./test_pytorch_onnx_onnxruntime_cuda.py_docs.md)


## Cross-References

- **File Documentation**: `test_symbolic_helper.py_docs.md`
- **Keyword Index**: `test_symbolic_helper.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/onnx`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/onnx`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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
python docs/test/onnx/test_symbolic_helper.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/onnx`):

- [`test_pytorch_onnx_onnxruntime.py_docs.md_docs.md`](./test_pytorch_onnx_onnxruntime.py_docs.md_docs.md)
- [`test_models_onnxruntime.py_docs.md_docs.md`](./test_models_onnxruntime.py_docs.md_docs.md)
- [`test_utility_funs.py_kw.md_docs.md`](./test_utility_funs.py_kw.md_docs.md)
- [`test_autograd_funs.py_kw.md_docs.md`](./test_autograd_funs.py_kw.md_docs.md)
- [`test_fx_type_promotion.py_docs.md_docs.md`](./test_fx_type_promotion.py_docs.md_docs.md)
- [`test_onnx_opset.py_docs.md_docs.md`](./test_onnx_opset.py_docs.md_docs.md)
- [`verify.py_docs.md_docs.md`](./verify.py_docs.md_docs.md)
- [`pytorch_test_common.py_kw.md_docs.md`](./pytorch_test_common.py_kw.md_docs.md)
- [`test_models_quantized_onnxruntime.py_kw.md_docs.md`](./test_models_quantized_onnxruntime.py_kw.md_docs.md)
- [`test_models_onnxruntime.py_kw.md_docs.md`](./test_models_onnxruntime.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_symbolic_helper.py_docs.md_docs.md`
- **Keyword Index**: `test_symbolic_helper.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
