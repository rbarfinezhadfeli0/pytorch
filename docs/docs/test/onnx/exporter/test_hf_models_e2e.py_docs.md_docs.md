# Documentation: `docs/test/onnx/exporter/test_hf_models_e2e.py_docs.md`

## File Metadata

- **Path**: `docs/test/onnx/exporter/test_hf_models_e2e.py_docs.md`
- **Size**: 12,756 bytes (12.46 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/onnx/exporter/test_hf_models_e2e.py`

## File Metadata

- **Path**: `test/onnx/exporter/test_hf_models_e2e.py`
- **Size**: 8,278 bytes (8.08 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: onnx"]
"""Unit LLM tests for the onnx dynamo exporter."""

from __future__ import annotations

from typing import Any

import transformers

import torch
from torch.onnx._internal.exporter import _testing as onnx_testing
from torch.testing._internal import common_utils


class DynamoExporterHfModelsTest(common_utils.TestCase):
    def export(self, model, args=(), kwargs=None, **options) -> torch.onnx.ONNXProgram:
        onnx_program = torch.onnx.export(
            model,
            args,
            kwargs=kwargs,
            dynamo=True,
            fallback=False,
            verbose=False,
            **options,
        )
        assert onnx_program is not None
        return onnx_program

    def test_onnx_export_huggingface_llm_models_with_kv_cache(self):
        model, kwargs, dynamic_axes, input_names, output_names = (
            _prepare_llm_model_gptj_to_test()
        )
        onnx_program = self.export(
            model,
            kwargs=kwargs,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        onnx_testing.assert_onnx_program(onnx_program)

    def test_onnx_export_with_custom_axis_names_in_dynamic_shapes(self):
        model, kwargs, _, input_names, output_names = _prepare_llm_model_gptj_to_test()

        dynamic_shapes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "past_key_values": [
                (
                    {0: "batch_size", 2: "past_sequence_length"},
                    {0: "batch_size", 2: "past_sequence_length"},
                ),
                (
                    {0: "batch_size", 2: "past_sequence_length"},
                    {0: "batch_size", 2: "past_sequence_length"},
                ),
                (
                    {0: "batch_size", 2: "past_sequence_length"},
                    {0: "batch_size", 2: "past_sequence_length"},
                ),
                (
                    {0: "batch_size", 2: "past_sequence_length"},
                    {0: "batch_size", 2: "past_sequence_length"},
                ),
                (
                    {0: "batch_size", 2: "past_sequence_length"},
                    {0: "batch_size", 2: "past_sequence_length"},
                ),
            ],
            "attention_mask": {0: "batch_size", 1: "masked_sequence_length"},
            "position_ids": {0: "batch_size", 1: "sequence_length"},
        }

        onnx_program = self.export(
            model,
            kwargs=kwargs,
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=dynamic_shapes,
            optimize=False,
        )
        onnx_testing.assert_onnx_program(onnx_program)

        # Check that the dynamic axes are correctly set in the ONNX model
        for dim, custom_name in zip(
            onnx_program.model.graph.inputs[0].shape,
            dynamic_shapes["input_ids"].values(),
        ):
            self.assertEqual(dim.value, custom_name)
        for idx in range(1, 11):
            shape_value = [
                dim if isinstance(dim, int) else dim.value
                for dim in onnx_program.model.graph.inputs[idx].shape
            ]
            self.assertEqual(shape_value, ["batch_size", 4, "past_sequence_length", 8])
        for dim, custom_name in zip(
            onnx_program.model.graph.inputs[11].shape,
            dynamic_shapes["attention_mask"].values(),
        ):
            self.assertEqual(dim.value, custom_name)
        for dim, custom_name in zip(
            onnx_program.model.graph.inputs[12].shape,
            dynamic_shapes["position_ids"].values(),
        ):
            self.assertEqual(dim.value, custom_name)


def _prepare_llm_model_gptj_to_test() -> tuple[
    torch.nn.Module,
    dict[str, Any],
    dict[str, dict[int, str]],
    list[str],
    list[str],
]:
    model = transformers.GPTJForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-gptj"
    )

    batch_size = 2
    input_seq_len = 16
    mask_seq_len = 32
    active_prob = 0.5
    vocab_size = 1000

    # Generate random input_ids with values between 0 and vocab_size-1
    input_ids = torch.randint(100, vocab_size, (batch_size, input_seq_len))
    # Generate random attention_mask with values 0 or 1, where 1 indicates an active token
    attention_mask = torch.bernoulli(
        torch.full((batch_size, mask_seq_len), active_prob)
    ).int()
    position_ids = torch.tensor(
        [
            [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        ]
    )
    past_key_values = [
        (torch.randn(2, 4, 16, 8), torch.randn(2, 4, 16, 8)),
        (torch.randn(2, 4, 16, 8), torch.randn(2, 4, 16, 8)),
        (torch.randn(2, 4, 16, 8), torch.randn(2, 4, 16, 8)),
        (torch.randn(2, 4, 16, 8), torch.randn(2, 4, 16, 8)),
        (torch.randn(2, 4, 16, 8), torch.randn(2, 4, 16, 8)),
    ]
    kwargs = {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "past_key_values.0.key": {0: "batch_size", 2: "past_sequence_length"},
        "past_key_values.0.value": {0: "batch_size", 2: "past_sequence_length"},
        "past_key_values.1.key": {0: "batch_size", 2: "past_sequence_length"},
        "past_key_values.1.value": {0: "batch_size", 2: "past_sequence_length"},
        "past_key_values.2.key": {0: "batch_size", 2: "past_sequence_length"},
        "past_key_values.2.value": {0: "batch_size", 2: "past_sequence_length"},
        "past_key_values.3.key": {0: "batch_size", 2: "past_sequence_length"},
        "past_key_values.3.value": {0: "batch_size", 2: "past_sequence_length"},
        "past_key_values.4.key": {0: "batch_size", 2: "past_sequence_length"},
        "past_key_values.4.value": {0: "batch_size", 2: "past_sequence_length"},
        "attention_mask": {
            0: "batch_size",
            1: "past_sequence_length + sequence_length",
        },
        "position_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
        "present.0.key": {0: "batch_size", 2: "past_sequence_length + sequence_length"},
        "present.0.value": {
            0: "batch_size",
            2: "past_sequence_length + sequence_length",
        },
        "present.1.key": {0: "batch_size", 2: "past_sequence_length + sequence_length"},
        "present.1.value": {
            0: "batch_size",
            2: "past_sequence_length + sequence_length",
        },
        "present.2.key": {0: "batch_size", 2: "past_sequence_length + sequence_length"},
        "present.2.value": {
            0: "batch_size",
            2: "past_sequence_length + sequence_length",
        },
        "present.3.key": {0: "batch_size", 2: "past_sequence_length + sequence_length"},
        "present.3.value": {
            0: "batch_size",
            2: "past_sequence_length + sequence_length",
        },
        "present.4.key": {0: "batch_size", 2: "past_sequence_length + sequence_length"},
        "present.4.value": {
            0: "batch_size",
            2: "past_sequence_length + sequence_length",
        },
    }
    input_names = [
        "input_ids",
        "past_key_values.0.key",
        "past_key_values.0.value",
        "past_key_values.1.key",
        "past_key_values.1.value",
        "past_key_values.2.key",
        "past_key_values.2.value",
        "past_key_values.3.key",
        "past_key_values.3.value",
        "past_key_values.4.key",
        "past_key_values.4.value",
        "attention_mask",
        "position_ids",
    ]
    output_names = [
        "logits",
        "present.0.key",
        "present.0.value",
        "present.1.key",
        "present.1.value",
        "present.2.key",
        "present.2.value",
        "present.3.key",
        "present.3.value",
        "present.4.key",
        "present.4.value",
    ]

    return model, kwargs, dynamic_axes, input_names, output_names


if __name__ == "__main__":
    common_utils.run_tests()

```



## High-Level Overview

"""Unit LLM tests for the onnx dynamo exporter."""from __future__ import annotationsfrom typing import Anyimport transformersimport torchfrom torch.onnx._internal.exporter import _testing as onnx_testingfrom torch.testing._internal import common_utilsclass DynamoExporterHfModelsTest(common_utils.TestCase):    def export(self, model, args=(), kwargs=None, **options) -> torch.onnx.ONNXProgram:        onnx_program = torch.onnx.export(            model,            args,            kwargs=kwargs,            dynamo=True,            fallback=False,            verbose=False,            **options,        )        assert onnx_program is not None        return onnx_program    def test_onnx_export_huggingface_llm_models_with_kv_cache(self):        model, kwargs, dynamic_axes, input_names, output_names = (            _prepare_llm_model_gptj_to_test()        )        onnx_program = self.export(            model,            kwargs=kwargs,            input_names=input_names,            output_names=output_names,            dynamic_axes=dynamic_axes,        )        onnx_testing.assert_onnx_program(onnx_program)    def test_onnx_export_with_custom_axis_names_in_dynamic_shapes(self):        model, kwargs, _, input_names, output_names = _prepare_llm_model_gptj_to_test()        dynamic_shapes = {            "input_ids": {0: "batch_size", 1: "sequence_length"},            "past_key_values": [                (                    {0: "batch_size", 2: "past_sequence_length"},                    {0: "batch_size", 2: "past_sequence_length"},

This Python file contains 1 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `DynamoExporterHfModelsTest`

**Functions defined**: `export`, `test_onnx_export_huggingface_llm_models_with_kv_cache`, `test_onnx_export_with_custom_axis_names_in_dynamic_shapes`, `_prepare_llm_model_gptj_to_test`

**Key imports**: annotations, Any, transformers, torch, _testing as onnx_testing, common_utils


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/onnx/exporter`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `typing`: Any
- `transformers`
- `torch`
- `torch.onnx._internal.exporter`: _testing as onnx_testing
- `torch.testing._internal`: common_utils


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/onnx/exporter/test_hf_models_e2e.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/onnx/exporter`):

- [`test_capture_strategies.py_docs.md`](./test_capture_strategies.py_docs.md)
- [`test_building.py_docs.md`](./test_building.py_docs.md)
- [`test_verification.py_docs.md`](./test_verification.py_docs.md)
- [`test_dynamic_shapes.py_docs.md`](./test_dynamic_shapes.py_docs.md)
- [`test_small_models_e2e.py_docs.md`](./test_small_models_e2e.py_docs.md)
- [`test_core.py_docs.md`](./test_core.py_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`test_ir_passes.py_docs.md`](./test_ir_passes.py_docs.md)
- [`test_tensors.py_docs.md`](./test_tensors.py_docs.md)


## Cross-References

- **File Documentation**: `test_hf_models_e2e.py_docs.md`
- **Keyword Index**: `test_hf_models_e2e.py_kw.md`
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

- Implements or uses **caching** mechanisms.
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
python docs/test/onnx/exporter/test_hf_models_e2e.py_docs.md
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

- **File Documentation**: `test_hf_models_e2e.py_docs.md_docs.md`
- **Keyword Index**: `test_hf_models_e2e.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
