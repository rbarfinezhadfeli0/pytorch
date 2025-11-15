# Documentation: `docs/test/quantization/core/experimental/test_fake_quantize.py_docs.md`

## File Metadata

- **Path**: `docs/test/quantization/core/experimental/test_fake_quantize.py_docs.md`
- **Size**: 7,120 bytes (6.95 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/quantization/core/experimental/test_fake_quantize.py`

## File Metadata

- **Path**: `test/quantization/core/experimental/test_fake_quantize.py`
- **Size**: 3,793 bytes (3.70 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# Owner(s): ["oncall: quantization"]

import torch
import unittest
from torch.ao.quantization.experimental.observer import APoTObserver
from torch.ao.quantization.experimental.quantizer import quantize_APoT, dequantize_APoT
from torch.ao.quantization.experimental.fake_quantize import APoTFakeQuantize
from torch.ao.quantization.experimental.fake_quantize_function import fake_quantize_function
forward_helper = fake_quantize_function.forward
backward = fake_quantize_function.backward
from torch.autograd import gradcheck

class TestFakeQuantize(unittest.TestCase):
    r""" Tests fake quantize calculate_qparams() method
         by comparing with result from observer calculate_qparams.
         Uses hard-coded values: alpha=1.0, b=4, k=2.
    """
    def test_fake_calc_qparams(self):
        apot_fake = APoTFakeQuantize(b=4, k=2)
        apot_fake.activation_post_process.min_val = torch.tensor([0.0])
        apot_fake.activation_post_process.max_val = torch.tensor([1.0])

        alpha, gamma, quantization_levels, level_indices = apot_fake.calculate_qparams(signed=False)

        observer = APoTObserver(b=4, k=2)
        observer.min_val = torch.tensor([0.0])
        observer.max_val = torch.tensor([1.0])

        qparams_expected = observer.calculate_qparams(signed=False)

        self.assertEqual(alpha, qparams_expected[0])
        self.assertTrue(torch.equal(gamma, qparams_expected[1]))
        self.assertTrue(torch.equal(quantization_levels, qparams_expected[2]))
        self.assertTrue(torch.equal(level_indices, qparams_expected[3]))

    r""" Tests fake quantize forward() method
         by comparing result with expected
         quant_dequant_APoT mapping of input tensor.
         Uses input tensor with random values from 0 -> 1000
         and APoT observer with hard-coded values b=4, k=2
    """
    def test_forward(self):
        # generate a tensor of size 20 with random values
        # between 0 -> 1000 to quantize -> dequantize
        X = 1000 * torch.rand(20)

        observer = APoTObserver(b=4, k=2)
        observer.forward(X)
        alpha, gamma, quantization_levels, level_indices = observer.calculate_qparams(signed=False)

        apot_fake = APoTFakeQuantize(b=4, k=2)
        apot_fake.enable_observer()
        apot_fake.enable_fake_quant()

        X_reduced_precision_fp = apot_fake.forward(torch.clone(X), False)

        # get X_expected by converting fp -> apot -> fp to simulate quantize -> dequantize
        X_to_apot = quantize_APoT(X, alpha, gamma, quantization_levels, level_indices)
        X_expected = dequantize_APoT(X_to_apot)

        self.assertTrue(torch.equal(X_reduced_precision_fp, X_expected))

    r""" Tests fake quantize forward() method
         throws error when qparams are None
    """
    def test_forward_exception(self):
        # generate a tensor of size 20 with random values
        # between 0 -> 1000 to quantize -> dequantize
        X = 1000 * torch.rand(20)

        apot_fake = APoTFakeQuantize(b=4, k=2)
        # disable observer so qparams not set, qparams are all None
        apot_fake.disable_observer()
        apot_fake.enable_fake_quant()

        with self.assertRaises(Exception):
            apot_fake.forward(torch.clone(X), False)

    r""" Tests fake quantize helper backward() method
         using torch.autograd.gradcheck function.
    """
    def test_backward(self):
        input = torch.randn(20, dtype=torch.double, requires_grad=True)

        observer = APoTObserver(b=4, k=2)
        observer(input)
        alpha, gamma, quantization_levels, level_indices = observer.calculate_qparams(signed=False)

        gradcheck(fake_quantize_function.apply, (input, alpha, gamma, quantization_levels, level_indices), atol=1e-4)

if __name__ == '__main__':
    unittest.main()

```



## High-Level Overview

r""" Tests fake quantize calculate_qparams() method         by comparing with result from observer calculate_qparams.         Uses hard-coded values: alpha=1.0, b=4, k=2.

This Python file contains 1 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestFakeQuantize`

**Functions defined**: `test_fake_calc_qparams`, `test_forward`, `test_forward_exception`, `test_backward`

**Key imports**: torch, unittest, APoTObserver, quantize_APoT, dequantize_APoT, APoTFakeQuantize, fake_quantize_function, gradcheck


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/quantization/core/experimental`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `unittest`
- `torch.ao.quantization.experimental.observer`: APoTObserver
- `torch.ao.quantization.experimental.quantizer`: quantize_APoT, dequantize_APoT
- `torch.ao.quantization.experimental.fake_quantize`: APoTFakeQuantize
- `torch.ao.quantization.experimental.fake_quantize_function`: fake_quantize_function
- `torch.autograd`: gradcheck


## Code Patterns & Idioms

### Common Patterns

- **Automatic Differentiation**: Uses autograd for gradient computation


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
python test/quantization/core/experimental/test_fake_quantize.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/quantization/core/experimental`):

- [`test_adaround_eager.py_docs.md`](./test_adaround_eager.py_docs.md)
- [`test_floatx.py_docs.md`](./test_floatx.py_docs.md)
- [`test_quantizer.py_docs.md`](./test_quantizer.py_docs.md)
- [`test_bits.py_docs.md`](./test_bits.py_docs.md)
- [`apot_fx_graph_mode_qat.py_docs.md`](./apot_fx_graph_mode_qat.py_docs.md)
- [`apot_fx_graph_mode_ptq.py_docs.md`](./apot_fx_graph_mode_ptq.py_docs.md)
- [`test_quantized_tensor.py_docs.md`](./test_quantized_tensor.py_docs.md)
- [`test_nonuniform_observer.py_docs.md`](./test_nonuniform_observer.py_docs.md)
- [`test_linear.py_docs.md`](./test_linear.py_docs.md)


## Cross-References

- **File Documentation**: `test_fake_quantize.py_docs.md`
- **Keyword Index**: `test_fake_quantize.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/quantization/core/experimental`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/quantization/core/experimental`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Automatic Differentiation**: Uses autograd for gradient computation


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
python docs/test/quantization/core/experimental/test_fake_quantize.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/quantization/core/experimental`):

- [`test_bits.py_docs.md_docs.md`](./test_bits.py_docs.md_docs.md)
- [`test_quantizer.py_docs.md_docs.md`](./test_quantizer.py_docs.md_docs.md)
- [`test_adaround_eager.py_docs.md_docs.md`](./test_adaround_eager.py_docs.md_docs.md)
- [`apot_fx_graph_mode_qat.py_kw.md_docs.md`](./apot_fx_graph_mode_qat.py_kw.md_docs.md)
- [`test_quantized_tensor.py_kw.md_docs.md`](./test_quantized_tensor.py_kw.md_docs.md)
- [`apot_fx_graph_mode_ptq.py_kw.md_docs.md`](./apot_fx_graph_mode_ptq.py_kw.md_docs.md)
- [`test_fake_quantize.py_kw.md_docs.md`](./test_fake_quantize.py_kw.md_docs.md)
- [`test_nonuniform_observer.py_kw.md_docs.md`](./test_nonuniform_observer.py_kw.md_docs.md)
- [`apot_fx_graph_mode_qat.py_docs.md_docs.md`](./apot_fx_graph_mode_qat.py_docs.md_docs.md)
- [`test_floatx.py_docs.md_docs.md`](./test_floatx.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_fake_quantize.py_docs.md_docs.md`
- **Keyword Index**: `test_fake_quantize.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
