# Documentation: `docs/test/onnx/exporter/test_ir_passes.py_docs.md`

## File Metadata

- **Path**: `docs/test/onnx/exporter/test_ir_passes.py_docs.md`
- **Size**: 7,700 bytes (7.52 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/onnx/exporter/test_ir_passes.py`

## File Metadata

- **Path**: `test/onnx/exporter/test_ir_passes.py`
- **Size**: 3,158 bytes (3.08 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: onnx"]
"""Unit tests for the _ir_passes module."""

from __future__ import annotations

import torch
from torch.onnx._internal._lazy_import import onnxscript_ir as ir
from torch.onnx._internal.exporter import _ir_passes
from torch.testing._internal import common_utils


@common_utils.instantiate_parametrized_tests
class ONNXIRPassesTest(common_utils.TestCase):
    @common_utils.parametrize(
        "shape_expr, expected_shape_expr",
        [
            ("2*s1", "batch_size*sequence_length"),
            ("s11/s1", "past_sequence_length/sequence_length"),
            ("(s1 + s11)*2", "(masked_sequence_length)*batch_size"),
        ],
    )
    def test__replace_names_in_rename_axis(self, shape_expr, expected_shape_expr):
        rename_mapping = {
            "s1 + s11": "masked_sequence_length",
            "s11": "past_sequence_length",
            "s1": "sequence_length",
            "2": "batch_size",
        }
        new_shape_expr = _ir_passes._replace_names(shape_expr, rename_mapping)
        self.assertEqual(new_shape_expr, expected_shape_expr)

    def test_rename_axis_succeeds_when_mapping_is_not_sorted_and_contains_the_str_not_in_the_model(
        self,
    ):
        model = ir.Model(
            ir.Graph(
                inputs=[
                    ir.Value(
                        name="input_0",
                        type=ir.DataType.FLOAT,
                        shape=ir.Shape(["s0", "s1"]),
                    ),
                    ir.Value(
                        name="input_1",
                        type=ir.DataType.FLOAT,
                        shape=ir.Shape(["s0 + s2", "s1 + s2"]),
                    ),
                    ir.Value(
                        name="input_2",
                        type=ir.DataType.FLOAT,
                        shape=ir.Shape(["s1/(s1 + s2)*2", "(s1 + s2)*2"]),
                    ),
                ],
                outputs=[
                    ir.Value(
                        name="output", type=ir.DataType.FLOAT, shape=ir.Shape("s99")
                    )
                ],
                nodes=[],
            ),
            ir_version=9,
            producer_name="pytorch",
            producer_version=torch.__version__,
        )

        mapping = {
            "s1": "sequence_length",
            "s2": "past_sequence_length",
            "s0": "batch_size",
            "s1 + s2": "masked_sequence_length",
            "s3": "extra_sequence_length",
        }
        _ir_passes.rename_axis(model, mapping)

        self.assertEqual(
            model.graph.inputs[0].shape, ir.Shape(["batch_size", "sequence_length"])
        )
        self.assertEqual(
            model.graph.inputs[1].shape,
            ir.Shape(["batch_size + past_sequence_length", "masked_sequence_length"]),
        )
        self.assertEqual(
            model.graph.inputs[2].shape,
            ir.Shape(
                [
                    "sequence_length/(masked_sequence_length)*2",
                    "(masked_sequence_length)*2",
                ]
            ),
        )


if __name__ == "__main__":
    common_utils.run_tests()

```



## High-Level Overview

"""Unit tests for the _ir_passes module."""from __future__ import annotationsimport torchfrom torch.onnx._internal._lazy_import import onnxscript_ir as irfrom torch.onnx._internal.exporter import _ir_passesfrom torch.testing._internal import common_utils@common_utils.instantiate_parametrized_testsclass ONNXIRPassesTest(common_utils.TestCase):    @common_utils.parametrize(        "shape_expr, expected_shape_expr",        [            ("2*s1", "batch_size*sequence_length"),            ("s11/s1", "past_sequence_length/sequence_length"),            ("(s1 + s11)*2", "(masked_sequence_length)*batch_size"),        ],    )    def test__replace_names_in_rename_axis(self, shape_expr, expected_shape_expr):        rename_mapping = {            "s1 + s11": "masked_sequence_length",            "s11": "past_sequence_length",            "s1": "sequence_length",            "2": "batch_size",        }        new_shape_expr = _ir_passes._replace_names(shape_expr, rename_mapping)        self.assertEqual(new_shape_expr, expected_shape_expr)    def test_rename_axis_succeeds_when_mapping_is_not_sorted_and_contains_the_str_not_in_the_model(        self,    ):        model = ir.Model(            ir.Graph(                inputs=[                    ir.Value(                        name="input_0",                        type=ir.DataType.FLOAT,                        shape=ir.Shape(["s0", "s1"]),                    ),                    ir.Value(                        name="input_1",                        type=ir.DataType.FLOAT,                        shape=ir.Shape(["s0 + s2", "s1 + s2"]),                    ),                    ir.Value(                        name="input_2",                        type=ir.DataType.FLOAT,

This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ONNXIRPassesTest`

**Functions defined**: `test__replace_names_in_rename_axis`, `test_rename_axis_succeeds_when_mapping_is_not_sorted_and_contains_the_str_not_in_the_model`

**Key imports**: annotations, torch, onnxscript_ir as ir, _ir_passes, common_utils


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/onnx/exporter`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `torch`
- `torch.onnx._internal._lazy_import`: onnxscript_ir as ir
- `torch.onnx._internal.exporter`: _ir_passes
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
python test/onnx/exporter/test_ir_passes.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/onnx/exporter`):

- [`test_capture_strategies.py_docs.md`](./test_capture_strategies.py_docs.md)
- [`test_building.py_docs.md`](./test_building.py_docs.md)
- [`test_hf_models_e2e.py_docs.md`](./test_hf_models_e2e.py_docs.md)
- [`test_verification.py_docs.md`](./test_verification.py_docs.md)
- [`test_dynamic_shapes.py_docs.md`](./test_dynamic_shapes.py_docs.md)
- [`test_small_models_e2e.py_docs.md`](./test_small_models_e2e.py_docs.md)
- [`test_core.py_docs.md`](./test_core.py_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`test_tensors.py_docs.md`](./test_tensors.py_docs.md)


## Cross-References

- **File Documentation**: `test_ir_passes.py_docs.md`
- **Keyword Index**: `test_ir_passes.py_kw.md`
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
python docs/test/onnx/exporter/test_ir_passes.py_docs.md
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


## Cross-References

- **File Documentation**: `test_ir_passes.py_docs.md_docs.md`
- **Keyword Index**: `test_ir_passes.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
