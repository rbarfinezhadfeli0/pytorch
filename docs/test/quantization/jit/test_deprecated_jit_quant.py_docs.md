# Documentation: `test/quantization/jit/test_deprecated_jit_quant.py`

## File Metadata

- **Path**: `test/quantization/jit/test_deprecated_jit_quant.py`
- **Size**: 7,538 bytes (7.36 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: quantization"]
# ruff: noqa: F841

import torch
from torch.testing._internal.common_quantization import skipIfNoFBGEMM
from torch.testing._internal.jit_utils import JitTestCase


class TestDeprecatedJitQuantized(JitTestCase):
    @skipIfNoFBGEMM
    def test_rnn_cell_quantized(self):
        d_in, d_hid = 2, 2

        for cell in [
            torch.nn.LSTMCell(d_in, d_hid).float(),
            torch.nn.GRUCell(d_in, d_hid).float(),
            torch.nn.RNNCell(d_in, d_hid).float(),
        ]:
            if isinstance(cell, torch.nn.LSTMCell):
                num_chunks = 4
            elif isinstance(cell, torch.nn.GRUCell):
                num_chunks = 3
            elif isinstance(cell, torch.nn.RNNCell):
                num_chunks = 1

            # Replace parameter values s.t. the range of values is exactly
            # 255, thus we will have 0 quantization error in the quantized
            # GEMM call. This i s for testing purposes.
            #
            # Note that the current implementation does not support
            # accumulation values outside of the range representable by a
            # 16 bit integer, instead resulting in a saturated value. We
            # must take care that in our test we do not end up with a dot
            # product that overflows the int16 range, e.g.
            # (255*127+255*127) = 64770. So, we hardcode the test values
            # here and ensure a mix of signedness.
            vals = [
                [100, -155],
                [100, -155],
                [-155, 100],
                [-155, 100],
                [100, -155],
                [-155, 100],
                [-155, 100],
                [100, -155],
            ]
            vals = vals[: d_hid * num_chunks]
            cell.weight_ih = torch.nn.Parameter(
                torch.tensor(vals, dtype=torch.float), requires_grad=False
            )
            cell.weight_hh = torch.nn.Parameter(
                torch.tensor(vals, dtype=torch.float), requires_grad=False
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "quantize_rnn_cell_modules function is no longer supported",
            ):
                cell = torch.jit.quantized.quantize_rnn_cell_modules(cell)

    @skipIfNoFBGEMM
    def test_rnn_quantized(self):
        d_in, d_hid = 2, 2

        for cell in [
            torch.nn.LSTM(d_in, d_hid).float(),
            torch.nn.GRU(d_in, d_hid).float(),
        ]:
            # Replace parameter values s.t. the range of values is exactly
            # 255, thus we will have 0 quantization error in the quantized
            # GEMM call. This i s for testing purposes.
            #
            # Note that the current implementation does not support
            # accumulation values outside of the range representable by a
            # 16 bit integer, instead resulting in a saturated value. We
            # must take care that in our test we do not end up with a dot
            # product that overflows the int16 range, e.g.
            # (255*127+255*127) = 64770. So, we hardcode the test values
            # here and ensure a mix of signedness.
            vals = [
                [100, -155],
                [100, -155],
                [-155, 100],
                [-155, 100],
                [100, -155],
                [-155, 100],
                [-155, 100],
                [100, -155],
            ]
            if isinstance(cell, torch.nn.LSTM):
                num_chunks = 4
            elif isinstance(cell, torch.nn.GRU):
                num_chunks = 3
            vals = vals[: d_hid * num_chunks]
            cell.weight_ih_l0 = torch.nn.Parameter(
                torch.tensor(vals, dtype=torch.float), requires_grad=False
            )
            cell.weight_hh_l0 = torch.nn.Parameter(
                torch.tensor(vals, dtype=torch.float), requires_grad=False
            )

            with self.assertRaisesRegex(
                RuntimeError, "quantize_rnn_modules function is no longer supported"
            ):
                cell_int8 = torch.jit.quantized.quantize_rnn_modules(
                    cell, dtype=torch.int8
                )

            with self.assertRaisesRegex(
                RuntimeError, "quantize_rnn_modules function is no longer supported"
            ):
                cell_fp16 = torch.jit.quantized.quantize_rnn_modules(
                    cell, dtype=torch.float16
                )

    if "fbgemm" in torch.backends.quantized.supported_engines:

        def test_quantization_modules(self):
            K1, N1 = 2, 2

            class FooBar(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.linear1 = torch.nn.Linear(K1, N1).float()

                def forward(self, x):
                    x = self.linear1(x)
                    return x

            fb = FooBar()
            fb.linear1.weight = torch.nn.Parameter(
                torch.tensor([[-150, 100], [100, -150]], dtype=torch.float),
                requires_grad=False,
            )
            fb.linear1.bias = torch.nn.Parameter(
                torch.zeros_like(fb.linear1.bias), requires_grad=False
            )

            x = (torch.rand(1, K1).float() - 0.5) / 10.0
            value = torch.tensor([[100, -150]], dtype=torch.float)

            y_ref = fb(value)

            with self.assertRaisesRegex(
                RuntimeError, "quantize_linear_modules function is no longer supported"
            ):
                fb_int8 = torch.jit.quantized.quantize_linear_modules(fb)

            with self.assertRaisesRegex(
                RuntimeError, "quantize_linear_modules function is no longer supported"
            ):
                fb_fp16 = torch.jit.quantized.quantize_linear_modules(fb, torch.float16)

    @skipIfNoFBGEMM
    def test_erase_class_tensor_shapes(self):
        class Linear(torch.nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                qweight = torch._empty_affine_quantized(
                    [out_features, in_features],
                    scale=1,
                    zero_point=0,
                    dtype=torch.qint8,
                )
                self._packed_weight = torch.ops.quantized.linear_prepack(qweight)

            @torch.jit.export
            def __getstate__(self):
                return (
                    torch.ops.quantized.linear_unpack(self._packed_weight)[0],
                    self.training,
                )

            def forward(self):
                return self._packed_weight

            @torch.jit.export
            def __setstate__(self, state):
                self._packed_weight = torch.ops.quantized.linear_prepack(state[0])
                self.training = state[1]

            @property
            def weight(self):
                return torch.ops.quantized.linear_unpack(self._packed_weight)[0]

            @weight.setter
            def weight(self, w):
                self._packed_weight = torch.ops.quantized.linear_prepack(w)

        with torch._jit_internal._disable_emit_hooks():
            x = torch.jit.script(Linear(10, 10))
            torch._C._jit_pass_erase_shape_information(x.graph)


if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_quantization.py TESTNAME\n\n"
        "instead."
    )

```



## High-Level Overview


This Python file contains 3 class(es) and 12 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestDeprecatedJitQuantized`, `FooBar`, `Linear`

**Functions defined**: `test_rnn_cell_quantized`, `test_rnn_quantized`, `test_quantization_modules`, `__init__`, `forward`, `test_erase_class_tensor_shapes`, `__init__`, `__getstate__`, `forward`, `__setstate__`, `weight`, `weight`

**Key imports**: torch, skipIfNoFBGEMM, JitTestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/quantization/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.testing._internal.common_quantization`: skipIfNoFBGEMM
- `torch.testing._internal.jit_utils`: JitTestCase


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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
python test/quantization/jit/test_deprecated_jit_quant.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/quantization/jit`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_quantize_jit.py_docs.md`](./test_quantize_jit.py_docs.md)
- [`test_ondevice_quantization.py_docs.md`](./test_ondevice_quantization.py_docs.md)
- [`test_fusion_passes.py_docs.md`](./test_fusion_passes.py_docs.md)


## Cross-References

- **File Documentation**: `test_deprecated_jit_quant.py_docs.md`
- **Keyword Index**: `test_deprecated_jit_quant.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
