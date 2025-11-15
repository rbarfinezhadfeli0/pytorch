# Documentation: `docs/test/ao/sparsity/test_qlinear_packed_params.py_docs.md`

## File Metadata

- **Path**: `docs/test/ao/sparsity/test_qlinear_packed_params.py_docs.md`
- **Size**: 13,425 bytes (13.11 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/ao/sparsity/test_qlinear_packed_params.py`

## File Metadata

- **Path**: `test/ao/sparsity/test_qlinear_packed_params.py`
- **Size**: 10,352 bytes (10.11 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```python
#!/usr/bin/env python3
# Owner(s): ["oncall: mobile"]

import tempfile

import torch
from torch.ao.nn.sparse.quantized.dynamic.linear import Linear
from torch.testing._internal.common_quantization import skipIfNoFBGEMM, skipIfNoQNNPACK
from torch.testing._internal.common_quantized import (
    override_cpu_allocator_for_qnnpack,
    override_quantized_engine,
    qengine_is_qnnpack,
)
from torch.testing._internal.common_utils import TestCase


class TestQlinearPackedParams(TestCase):
    def qlinear_packed_params_test(self, allow_non_zero_zero_points=False):
        # copied from https://pytorch.org/docs/stable/sparse.html#csr-tensor-operations,
        # so row/col block indices match that example, but with blocks and
        # scaled rows
        weight_fp32 = torch.Tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0],
                [6, 6, 6, 6, 12, 12, 12, 12, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        row_block_size = 1
        col_block_size = 4
        out_features = weight_fp32.shape[0]

        scales = [2.0, 6.0, 12.0]
        zero_points = [
            ((i + 1) if allow_non_zero_zero_points else 0) for i in range(out_features)
        ]
        dtype = torch.qint8

        wide_weight_fp32 = torch.zeros((3, 4008))  # 4000 is tile width for Fbgemm
        wide_weight_fp32[0][0] = 4
        wide_weight_fp32[0][4004] = 6
        wide_weight_fp32[1][0] = 8

        per_tensor_small = (
            torch.quantize_per_tensor(weight_fp32, scales[0], zero_points[0], dtype),
            True,
            [0, 1, 3, 3],
            [2, 0, 1],
            [
                x + (1 if allow_non_zero_zero_points else 0)
                for x in [1, 1, 1, 1, 3, 3, 3, 3, 6, 6, 6, 6]
            ],
        )

        per_channel_small = (
            torch.quantize_per_channel(
                weight_fp32,
                torch.Tensor(scales),
                torch.Tensor(zero_points).to(torch.int),
                0,  # axis = 0
                dtype,
            ),
            False,
            [0, 1, 3, 3],
            [2, 0, 1],
            [
                x + ([1, 2, 2][i // 4] if allow_non_zero_zero_points else 0)
                for (i, x) in enumerate([1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
            ],
        )

        per_tensor_large = (
            torch.quantize_per_tensor(
                wide_weight_fp32,
                scales[0],
                zero_points[0],
                dtype,
            ),
            True,
            [0, 2, 3, 3],
            [0, 1001, 0],
            [
                x + (1 if allow_non_zero_zero_points else 0)
                for x in [2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0]
            ],
        )

        for (
            weight,
            is_per_tensor_quantized,
            expected_row_block_indices,
            expected_col_block_indices,
            expected_weights,
        ) in [per_tensor_small, per_channel_small, per_tensor_large]:
            lin = Linear(
                out_features=weight.shape[0],
                in_features=weight.shape[1],
                row_block_size=row_block_size,
                col_block_size=col_block_size,
                bias=True,
                dtype=dtype,
            )

            bias = torch.ones(size=(weight.shape[0],))

            lin.set_weight_bias(weight, bias, row_block_size, col_block_size)

            serialized = lin._packed_params._packed_params.__getstate__()

            (
                _,  # version
                bias_,
                out_features_block_size_,
                in_features_block_size_,
                weight_scales_,
                weight_zero_points_,
                quantization_scheme_,
                row_block_indices_,
                col_block_indices_,
                weights_,
                output_channels_,
                input_channels_,
            ) = serialized[0]

            # Test Serialization
            self.assertEqual(bias_, bias)
            self.assertEqual(out_features_block_size_, row_block_size)
            self.assertEqual(in_features_block_size_, col_block_size)
            self.assertEqual(
                weight_scales_, [scales[0]] if is_per_tensor_quantized else scales
            )
            self.assertEqual(
                weight_zero_points_,
                [zero_points[0]] if is_per_tensor_quantized else zero_points,
            )
            self.assertEqual(quantization_scheme_, is_per_tensor_quantized)
            self.assertEqual(row_block_indices_, expected_row_block_indices)
            self.assertEqual(col_block_indices_, expected_col_block_indices)
            self.assertEqual(
                weights_.tolist(), [v + 128 for v in expected_weights]
            )  # weights are serialized as +128
            self.assertEqual(output_channels_, weight.shape[0])
            self.assertEqual(input_channels_, weight.shape[1])

            # Test Unpacking
            (
                weights_,
                bias_,
                out_features_block_size_,
                in_features_block_size_,
            ) = lin._weight_bias()
            self.assertEqual(torch.dequantize(weights_), torch.dequantize(weight))
            self.assertEqual(bias_, bias)
            self.assertEqual(out_features_block_size_, row_block_size)
            self.assertEqual(in_features_block_size_, col_block_size)

            # Test Deserialization
            with tempfile.TemporaryFile() as file_buff:
                torch.save(lin, file_buff)
                file_buff.seek(0)
                lin2 = torch.load(file_buff)
                self.assertEqual(lin._weight_bias(), lin2._weight_bias())
                # Serialize -> Deserialize -> Serialize should match Serialize
                self.assertEqual(
                    serialized, lin2._packed_params._packed_params.__getstate__()
                )

                # Test that op output is preserved by serialize -> deserialize
                if qengine_is_qnnpack():
                    x = torch.rand(size=(1, weight.shape[1]))
                    y1 = lin(x)
                    y2 = lin2(x)
                    self.assertEqual(y1, y2)

    @skipIfNoFBGEMM
    def test_qlinear_packed_params_fbgemm(self):
        torch.manual_seed(0)
        with override_quantized_engine("fbgemm"):
            self.qlinear_packed_params_test(allow_non_zero_zero_points=False)

    @skipIfNoQNNPACK
    def test_qlinear_packed_params_qnnpack(self):
        torch.manual_seed(0)
        with override_quantized_engine("qnnpack"):
            with override_cpu_allocator_for_qnnpack(qengine_is_qnnpack()):
                self.qlinear_packed_params_test(allow_non_zero_zero_points=True)

    def test_qlinear_packed_params_fbgemm_qnnpack_cross_compatibility(self):
        torch.manual_seed(0)

        weight_fp32 = torch.Tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0],
                [6, 6, 6, 6, 12, 12, 12, 12, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        row_block_size = 1
        col_block_size = 4
        out_features = weight_fp32.shape[0]

        scales = [2.0, 3.0, 7.0]
        zero_points = [0 for _ in range(out_features)]
        dtype = torch.qint8

        def make_lin_get_state_weight_bias_and_save():
            weight = torch.quantize_per_tensor(
                weight_fp32,
                scales[0],
                zero_points[0],
                dtype,
            )
            lin = Linear(
                out_features=weight.shape[0],
                in_features=weight.shape[1],
                row_block_size=row_block_size,
                col_block_size=col_block_size,
                bias=True,
                dtype=dtype,
            )
            bias = torch.ones(size=(weight.shape[0],))
            lin.set_weight_bias(weight, bias, row_block_size, col_block_size)

            state = lin._packed_params._packed_params.__getstate__()
            weight_bias = lin._weight_bias()

            file_buff = tempfile.TemporaryFile()
            torch.save(lin, file_buff)
            file_buff.seek(0)

            return ((state, weight_bias), file_buff)

        def load_get_state_weight_bias(f_b):
            lin2 = torch.load(f_b)
            state = lin2._packed_params._packed_params.__getstate__()
            weight_bias = lin2._weight_bias()
            f_b.close()
            return (state, weight_bias)

        def packed_params_data_with_int32_indices(data_as_state_and_weight_bias):
            (st, weight_bias) = data_as_state_and_weight_bias
            (s0, s1) = st
            s0_updated = tuple(
                [
                    # 7 and 8 are row and col block indices respectively
                    v if (i != 7 and i != 8) else v.to(torch.int32)
                    for (i, v) in enumerate(list(s0))
                ]
            )
            return ((s0_updated, s1), weight_bias)

        # Test Fbgemm -> Qnnpack
        with override_quantized_engine("fbgemm"):
            (
                packed_params_data_1a,
                file_buff_1,
            ) = make_lin_get_state_weight_bias_and_save()

        with override_quantized_engine("qnnpack"):
            with override_cpu_allocator_for_qnnpack(qengine_is_qnnpack()):
                packed_params_data_1b = load_get_state_weight_bias(file_buff_1)

        self.assertEqual(
            packed_params_data_with_int32_indices(packed_params_data_1a),
            packed_params_data_with_int32_indices(packed_params_data_1b),
        )

        # Test Qnnpack -> Fbgemm
        with override_quantized_engine("qnnpack"):
            with override_cpu_allocator_for_qnnpack(qengine_is_qnnpack()):
                (
                    packed_params_data_2a,
                    file_buff_2,
                ) = make_lin_get_state_weight_bias_and_save()

        with override_quantized_engine("fbgemm"):
            packed_params_data_2b = load_get_state_weight_bias(file_buff_2)

        self.assertEqual(
            packed_params_data_with_int32_indices(packed_params_data_2a),
            packed_params_data_with_int32_indices(packed_params_data_2b),
        )

```



## High-Level Overview


This Python file contains 1 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestQlinearPackedParams`

**Functions defined**: `qlinear_packed_params_test`, `test_qlinear_packed_params_fbgemm`, `test_qlinear_packed_params_qnnpack`, `test_qlinear_packed_params_fbgemm_qnnpack_cross_compatibility`, `make_lin_get_state_weight_bias_and_save`, `load_get_state_weight_bias`, `packed_params_data_with_int32_indices`

**Key imports**: tempfile, torch, Linear, skipIfNoFBGEMM, skipIfNoQNNPACK, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/ao/sparsity`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `tempfile`
- `torch`
- `torch.ao.nn.sparse.quantized.dynamic.linear`: Linear
- `torch.testing._internal.common_quantization`: skipIfNoFBGEMM, skipIfNoQNNPACK
- `torch.testing._internal.common_utils`: TestCase


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
python test/ao/sparsity/test_qlinear_packed_params.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/ao/sparsity`):

- [`test_kernels.py_docs.md`](./test_kernels.py_docs.md)
- [`test_activation_sparsifier.py_docs.md`](./test_activation_sparsifier.py_docs.md)
- [`test_data_scheduler.py_docs.md`](./test_data_scheduler.py_docs.md)
- [`test_parametrization.py_docs.md`](./test_parametrization.py_docs.md)
- [`test_scheduler.py_docs.md`](./test_scheduler.py_docs.md)
- [`test_sparsity_utils.py_docs.md`](./test_sparsity_utils.py_docs.md)
- [`test_data_sparsifier.py_docs.md`](./test_data_sparsifier.py_docs.md)
- [`test_structured_sparsifier.py_docs.md`](./test_structured_sparsifier.py_docs.md)
- [`test_sparsifier.py_docs.md`](./test_sparsifier.py_docs.md)


## Cross-References

- **File Documentation**: `test_qlinear_packed_params.py_docs.md`
- **Keyword Index**: `test_qlinear_packed_params.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/ao/sparsity`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/ao/sparsity`, which is part of the **testing infrastructure**.



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
python docs/test/ao/sparsity/test_qlinear_packed_params.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/ao/sparsity`):

- [`test_parametrization.py_kw.md_docs.md`](./test_parametrization.py_kw.md_docs.md)
- [`test_data_sparsifier.py_kw.md_docs.md`](./test_data_sparsifier.py_kw.md_docs.md)
- [`test_activation_sparsifier.py_docs.md_docs.md`](./test_activation_sparsifier.py_docs.md_docs.md)
- [`test_data_scheduler.py_kw.md_docs.md`](./test_data_scheduler.py_kw.md_docs.md)
- [`test_sparsity_utils.py_kw.md_docs.md`](./test_sparsity_utils.py_kw.md_docs.md)
- [`test_structured_sparsifier.py_docs.md_docs.md`](./test_structured_sparsifier.py_docs.md_docs.md)
- [`test_composability.py_kw.md_docs.md`](./test_composability.py_kw.md_docs.md)
- [`test_kernels.py_kw.md_docs.md`](./test_kernels.py_kw.md_docs.md)
- [`test_structured_sparsifier.py_kw.md_docs.md`](./test_structured_sparsifier.py_kw.md_docs.md)
- [`test_data_sparsifier.py_docs.md_docs.md`](./test_data_sparsifier.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_qlinear_packed_params.py_docs.md_docs.md`
- **Keyword Index**: `test_qlinear_packed_params.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
