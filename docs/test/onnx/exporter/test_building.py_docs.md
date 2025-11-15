# Documentation: `test/onnx/exporter/test_building.py`

## File Metadata

- **Path**: `test/onnx/exporter/test_building.py`
- **Size**: 6,425 bytes (6.27 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: onnx"]
"""Unit tests for the _building module."""

from __future__ import annotations

import numpy as np
import onnxscript
from onnxscript import ir

import torch
from torch.onnx._internal.exporter import _building, _tensors
from torch.testing._internal import common_utils


class TestOpRecorder(common_utils.TestCase):
    def setUp(self):
        super().setUp()
        self.opset_version = 17
        self.opset = onnxscript.values.Opset("", self.opset_version)
        self.recorder = _building.OpRecorder(opset=self.opset, constant_farm={})

        self.model = ir.Model(
            graph=ir.Graph(
                [],
                [],
                nodes=[],
                opset_imports={
                    "": self.opset_version,
                },
                name="main_graph",
            ),
            ir_version=9,
            producer_name="pytorch",
            producer_version=torch.__version__,
        )

    def test_skippable_castlike_is_ommited(self):
        input_x = _tensors.SymbolicTensor(opset=self.opset, name="input_x")
        input_x.dtype = ir.DataType.FLOAT

        input_y = _tensors.SymbolicTensor(opset=self.opset, name="input_y")
        input_y.dtype = ir.DataType.FLOAT

        with onnxscript.evaluator.default_as(tracer := self.recorder):
            cast = self.opset.CastLike(input_y, input_x)
            _ = self.opset.Add(input_x, cast)

        self.assertEqual(len(tracer.nodes), 1)
        self.assertEqual(tracer.nodes[0].op_type, "Add")

    def test_castlike_is_replaced_with_cast_when_it_is_traced(self):
        input_x = _tensors.SymbolicTensor(opset=self.opset, name="input_x")
        input_x.dtype = ir.DataType.FLOAT

        input_y = _tensors.SymbolicTensor(opset=self.opset, name="input_y")
        input_y.dtype = ir.DataType.INT64

        with onnxscript.evaluator.default_as(tracer := self.recorder):
            cast = self.opset.CastLike(input_y, input_x)
            _ = self.opset.Add(input_x, cast)

        self.assertEqual(len(tracer.nodes), 2)
        self.assertEqual(tracer.nodes[0].op_type, "Cast")
        self.assertEqual(tracer.nodes[1].op_type, "Add")

    def test_python_constant_added_as_constant_nodes(self):
        input_x = _tensors.SymbolicTensor(
            opset=self.opset, name="input_x", shape=ir.Shape([2, 3, 4])
        )
        new_shape = [3, 2, 4]

        with onnxscript.evaluator.default_as(tracer := self.recorder):
            _ = self.opset.Reshape(input_x, new_shape)

        self.assertEqual(len(tracer.nodes), 2)
        self.assertEqual(tracer.nodes[0].op_type, "Constant")
        self.assertEqual(
            tracer.nodes[0].attributes["value"].value.numpy(), np.array(new_shape)
        )
        self.assertEqual(tracer.nodes[1].op_type, "Reshape")

    def test_process_python_sequence_with_allowed_sequence_type(self):
        input_x = _tensors.SymbolicTensor(
            opset=self.opset, name="input_x", shape=ir.Shape([2, 3])
        )
        input_y = _tensors.SymbolicTensor(
            opset=self.opset, name="input_y", shape=ir.Shape([2, 4])
        )
        input_z = _tensors.SymbolicTensor(
            opset=self.opset, name="input_z", shape=ir.Shape([1, 3])
        )

        with onnxscript.evaluator.default_as(tracer := self.recorder):
            _ = self.opset.SequenceAt([input_x, input_y, input_z], 1)

        self.assertEqual(len(tracer.nodes), 3)
        self.assertEqual(tracer.nodes[1].op_type, "SequenceConstruct")

    def test_process_python_sequence_with_variadic_input(self):
        input_x = _tensors.SymbolicTensor(
            opset=self.opset, name="input_x", shape=ir.Shape([2, 3])
        )
        input_y = _tensors.SymbolicTensor(
            opset=self.opset, name="input_y", shape=ir.Shape([2, 4])
        )
        input_z = _tensors.SymbolicTensor(
            opset=self.opset, name="input_z", shape=ir.Shape([1, 3])
        )

        with onnxscript.evaluator.default_as(tracer := self.recorder):
            _ = self.opset.Max(input_x, input_y, 0, input_z)

        self.assertEqual(len(tracer.nodes), 2)
        self.assertEqual(tracer.nodes[0].op_type, "Constant")

    def test_process_python_sequence_creates_extra_concat(self):
        # Elements in the list must be 0D tensors
        input_x = _tensors.SymbolicTensor(
            opset=self.opset, name="input_x", shape=ir.Shape([])
        )
        input_y = _tensors.SymbolicTensor(
            opset=self.opset, name="input_y", shape=ir.Shape([])
        )
        input_z = _tensors.SymbolicTensor(
            opset=self.opset, name="input_z", shape=ir.Shape([4, 3])
        )

        with onnxscript.evaluator.default_as(tracer := self.recorder):
            _ = self.opset.Add([input_x, input_y], input_z)

        self.assertEqual(len(tracer.nodes), 6)
        self.assertEqual(tracer.nodes[-2].op_type, "Concat")
        self.assertEqual(tracer.nodes[-2].attributes["axis"].value, 0)

    def test_process_python_sequence_mix_symbolic_constant_creates_extra_concat(self):
        # Elements in the list must be 0D tensors
        input_x = _tensors.SymbolicTensor(
            opset=self.opset, name="input_x", shape=ir.Shape([])
        )
        input_z = _tensors.SymbolicTensor(
            opset=self.opset, name="input_z", shape=ir.Shape([4, 3])
        )

        with onnxscript.evaluator.default_as(tracer := self.recorder):
            _ = self.opset.Add([input_x, 42], input_z)

        self.assertEqual(len(tracer.nodes), 5)
        self.assertEqual(tracer.nodes[-2].op_type, "Concat")
        self.assertEqual(tracer.nodes[-2].attributes["axis"].value, 0)

    def test_process_python_sequence_mix_constant_symbolic_creates_extra_concat(self):
        # Elements in the list must be 0D tensors
        input_x = _tensors.SymbolicTensor(
            opset=self.opset, name="input_x", shape=ir.Shape([])
        )
        input_z = _tensors.SymbolicTensor(
            opset=self.opset, name="input_z", shape=ir.Shape([4, 3])
        )

        with onnxscript.evaluator.default_as(tracer := self.recorder):
            # Constant first
            _ = self.opset.Add([42, input_x], input_z)

        self.assertEqual(len(tracer.nodes), 5)
        self.assertEqual(tracer.nodes[-2].op_type, "Concat")
        self.assertEqual(tracer.nodes[-2].attributes["axis"].value, 0)


if __name__ == "__main__":
    common_utils.run_tests()

```



## High-Level Overview

"""Unit tests for the _building module."""from __future__ import annotationsimport numpy as npimport onnxscriptfrom onnxscript import irimport torchfrom torch.onnx._internal.exporter import _building, _tensorsfrom torch.testing._internal import common_utilsclass TestOpRecorder(common_utils.TestCase):    def setUp(self):        super().setUp()        self.opset_version = 17        self.opset = onnxscript.values.Opset("", self.opset_version)        self.recorder = _building.OpRecorder(opset=self.opset, constant_farm={})        self.model = ir.Model(            graph=ir.Graph(                [],                [],                nodes=[],                opset_imports={                    "": self.opset_version,                },                name="main_graph",            ),            ir_version=9,            producer_name="pytorch",            producer_version=torch.__version__,        )    def test_skippable_castlike_is_ommited(self):        input_x = _tensors.SymbolicTensor(opset=self.opset, name="input_x")        input_x.dtype = ir.DataType.FLOAT        input_y = _tensors.SymbolicTensor(opset=self.opset, name="input_y")        input_y.dtype = ir.DataType.FLOAT        with onnxscript.evaluator.default_as(tracer := self.recorder):            cast = self.opset.CastLike(input_y, input_x)            _ = self.opset.Add(input_x, cast)        self.assertEqual(len(tracer.nodes), 1)        self.assertEqual(tracer.nodes[0].op_type, "Add")

This Python file contains 1 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestOpRecorder`

**Functions defined**: `setUp`, `test_skippable_castlike_is_ommited`, `test_castlike_is_replaced_with_cast_when_it_is_traced`, `test_python_constant_added_as_constant_nodes`, `test_process_python_sequence_with_allowed_sequence_type`, `test_process_python_sequence_with_variadic_input`, `test_process_python_sequence_creates_extra_concat`, `test_process_python_sequence_mix_symbolic_constant_creates_extra_concat`, `test_process_python_sequence_mix_constant_symbolic_creates_extra_concat`

**Key imports**: annotations, numpy as np, onnxscript, ir, torch, _building, _tensors, common_utils


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/onnx/exporter`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `numpy as np`
- `onnxscript`
- `torch`
- `torch.onnx._internal.exporter`: _building, _tensors
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
python test/onnx/exporter/test_building.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/onnx/exporter`):

- [`test_capture_strategies.py_docs.md`](./test_capture_strategies.py_docs.md)
- [`test_hf_models_e2e.py_docs.md`](./test_hf_models_e2e.py_docs.md)
- [`test_verification.py_docs.md`](./test_verification.py_docs.md)
- [`test_dynamic_shapes.py_docs.md`](./test_dynamic_shapes.py_docs.md)
- [`test_small_models_e2e.py_docs.md`](./test_small_models_e2e.py_docs.md)
- [`test_core.py_docs.md`](./test_core.py_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`test_ir_passes.py_docs.md`](./test_ir_passes.py_docs.md)
- [`test_tensors.py_docs.md`](./test_tensors.py_docs.md)


## Cross-References

- **File Documentation**: `test_building.py_docs.md`
- **Keyword Index**: `test_building.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
