# Documentation: `test/quantization/core/experimental/test_quantizer.py`

## File Metadata

- **Path**: `test/quantization/core/experimental/test_quantizer.py`
- **Size**: 9,213 bytes (9.00 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# Owner(s): ["oncall: quantization"]

import torch
from torch import quantize_per_tensor
from torch.ao.quantization.observer import MinMaxObserver
from torch.ao.quantization.experimental.observer import APoTObserver
from torch.ao.quantization.experimental.quantizer import APoTQuantizer, quantize_APoT, dequantize_APoT
import unittest
import random

class TestQuantizer(unittest.TestCase):
    r""" Tests quantize_APoT result on random 1-dim tensor
        and hardcoded values for b, k by comparing to uniform quantization
        (non-uniform quantization reduces to uniform for k = 1)
        quantized tensor (https://pytorch.org/docs/stable/generated/torch.quantize_per_tensor.html)
        * tensor2quantize: Tensor
        * b: 8
        * k: 1
    """
    def test_quantize_APoT_rand_k1(self):
        # generate random size of tensor2quantize between 1 -> 20
        size = random.randint(1, 20)

        # generate tensor with random fp values between 0 -> 1000
        tensor2quantize = 1000 * torch.rand(size, dtype=torch.float)

        apot_observer = APoTObserver(b=8, k=1)
        apot_observer(tensor2quantize)
        alpha, gamma, quantization_levels, level_indices = apot_observer.calculate_qparams(signed=False)

        # get apot quantized tensor result
        qtensor = quantize_APoT(tensor2quantize=tensor2quantize,
                                alpha=alpha,
                                gamma=gamma,
                                quantization_levels=quantization_levels,
                                level_indices=level_indices)

        # get uniform quantization quantized tensor result
        uniform_observer = MinMaxObserver()
        uniform_observer(tensor2quantize)
        scale, zero_point = uniform_observer.calculate_qparams()

        uniform_quantized = quantize_per_tensor(input=tensor2quantize,
                                                scale=scale,
                                                zero_point=zero_point,
                                                dtype=torch.quint8).int_repr()

        qtensor_data = qtensor.data.int()
        uniform_quantized_tensor = uniform_quantized.data.int()

        self.assertTrue(torch.equal(qtensor_data, uniform_quantized_tensor))

    r""" Tests quantize_APoT for k != 1.
        Tests quantize_APoT result on random 1-dim tensor and hardcoded values for
        b=4, k=2 by comparing results to hand-calculated results from APoT paper
        https://arxiv.org/pdf/1909.13144.pdf
        * tensor2quantize: Tensor
        * b: 4
        * k: 2
    """
    def test_quantize_APoT_k2(self):
        r"""
        given b = 4, k = 2, alpha = 1.0, we know:
        (from APoT paper example: https://arxiv.org/pdf/1909.13144.pdf)

        quantization_levels = tensor([0.0000, 0.0208, 0.0417, 0.0625, 0.0833, 0.1250, 0.1667,
        0.1875, 0.2500, 0.3333, 0.3750, 0.5000, 0.6667, 0.6875, 0.7500, 1.0000])

        level_indices = tensor([ 0, 3, 12, 15,  2, 14,  8, 11, 10, 1, 13,  9,  4,  7,  6,  5]))
        """

        # generate tensor with random fp values
        tensor2quantize = torch.tensor([0, 0.0215, 0.1692, 0.385, 1, 0.0391])

        observer = APoTObserver(b=4, k=2)
        observer.forward(tensor2quantize)
        alpha, gamma, quantization_levels, level_indices = observer.calculate_qparams(signed=False)

        # get apot quantized tensor result
        qtensor = quantize_APoT(tensor2quantize=tensor2quantize,
                                alpha=alpha,
                                gamma=gamma,
                                quantization_levels=quantization_levels,
                                level_indices=level_indices)

        qtensor_data = qtensor.data.int()

        # expected qtensor values calculated based on
        # corresponding level_indices to nearest quantization level
        # for each fp value in tensor2quantize
        # e.g.
        # 0.0215 in tensor2quantize nearest 0.0208 in quantization_levels -> 3 in level_indices
        expected_qtensor = torch.tensor([0, 3, 8, 13, 5, 12], dtype=torch.int32)

        self.assertTrue(torch.equal(qtensor_data, expected_qtensor))

    r""" Tests dequantize_apot result on random 1-dim tensor
        and hardcoded values for b, k.
        Dequant -> quant an input tensor and verify that
        result is equivalent to input
        * tensor2quantize: Tensor
        * b: 4
        * k: 2
    """
    def test_dequantize_quantize_rand_b4(self):
        # make observer
        observer = APoTObserver(4, 2)

        # generate random size of tensor2quantize between 1 -> 20
        size = random.randint(1, 20)

        # make tensor2quantize: random fp values between 0 -> 1000
        tensor2quantize = 1000 * torch.rand(size, dtype=torch.float)

        observer.forward(tensor2quantize)

        alpha, gamma, quantization_levels, level_indices = observer.calculate_qparams(signed=False)

        # make mock apot_tensor
        original_apot = quantize_APoT(tensor2quantize=tensor2quantize,
                                      alpha=alpha,
                                      gamma=gamma,
                                      quantization_levels=quantization_levels,
                                      level_indices=level_indices)

        original_input = torch.clone(original_apot.data).int()

        # dequantize apot_tensor
        dequantize_result = dequantize_APoT(apot_tensor=original_apot)

        # quantize apot_tensor
        final_apot = quantize_APoT(tensor2quantize=dequantize_result,
                                   alpha=alpha,
                                   gamma=gamma,
                                   quantization_levels=quantization_levels,
                                   level_indices=level_indices)

        result = final_apot.data.int()

        self.assertTrue(torch.equal(original_input, result))

    r""" Tests dequantize_apot result on random 1-dim tensor
        and hardcoded values for b, k.
        Dequant -> quant an input tensor and verify that
        result is equivalent to input
        * tensor2quantize: Tensor
        * b: 12
        * k: 4
    """
    def test_dequantize_quantize_rand_b6(self):
        # make observer
        observer = APoTObserver(12, 4)

        # generate random size of tensor2quantize between 1 -> 20
        size = random.randint(1, 20)

        # make tensor2quantize: random fp values between 0 -> 1000
        tensor2quantize = 1000 * torch.rand(size, dtype=torch.float)

        observer.forward(tensor2quantize)

        alpha, gamma, quantization_levels, level_indices = observer.calculate_qparams(signed=False)

        # make mock apot_tensor
        original_apot = quantize_APoT(tensor2quantize=tensor2quantize,
                                      alpha=alpha,
                                      gamma=gamma,
                                      quantization_levels=quantization_levels,
                                      level_indices=level_indices)

        original_input = torch.clone(original_apot.data).int()

        # dequantize apot_tensor
        dequantize_result = dequantize_APoT(apot_tensor=original_apot)

        # quantize apot_tensor
        final_apot = quantize_APoT(tensor2quantize=dequantize_result,
                                   alpha=alpha,
                                   gamma=gamma,
                                   quantization_levels=quantization_levels,
                                   level_indices=level_indices)

        result = final_apot.data.int()

        self.assertTrue(torch.equal(original_input, result))

    r""" Tests for correct dimensions in dequantize_apot result
         on random 3-dim tensor with random dimension sizes
         and hardcoded values for b, k.
         Dequant an input tensor and verify that
         dimensions are same as input.
         * tensor2quantize: Tensor
         * b: 4
         * k: 2
    """
    def test_dequantize_dim(self):
        # make observer
        observer = APoTObserver(4, 2)

        # generate random size of tensor2quantize between 1 -> 20
        size1 = random.randint(1, 20)
        size2 = random.randint(1, 20)
        size3 = random.randint(1, 20)

        # make tensor2quantize: random fp values between 0 -> 1000
        tensor2quantize = 1000 * torch.rand(size1, size2, size3, dtype=torch.float)

        observer.forward(tensor2quantize)

        alpha, gamma, quantization_levels, level_indices = observer.calculate_qparams(signed=False)

        # make mock apot_tensor
        original_apot = quantize_APoT(tensor2quantize=tensor2quantize,
                                      alpha=alpha,
                                      gamma=gamma,
                                      quantization_levels=quantization_levels,
                                      level_indices=level_indices)

        # dequantize apot_tensor
        dequantize_result = dequantize_APoT(apot_tensor=original_apot)

        self.assertEqual(original_apot.data.size(), dequantize_result.size())

    def test_q_apot_alpha(self):
        with self.assertRaises(NotImplementedError):
            APoTQuantizer.q_apot_alpha(self)

if __name__ == '__main__':
    unittest.main()

```



## High-Level Overview

r""" Tests quantize_APoT result on random 1-dim tensor        and hardcoded values for b, k by comparing to uniform quantization        (non-uniform quantization reduces to uniform for k = 1)        quantized tensor (https://pytorch.org/docs/stable/generated/torch.quantize_per_tensor.html)        * tensor2quantize: Tensor        * b: 8        * k: 1

This Python file contains 1 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestQuantizer`

**Functions defined**: `test_quantize_APoT_rand_k1`, `test_quantize_APoT_k2`, `test_dequantize_quantize_rand_b4`, `test_dequantize_quantize_rand_b6`, `test_dequantize_dim`, `test_q_apot_alpha`

**Key imports**: torch, quantize_per_tensor, MinMaxObserver, APoTObserver, APoTQuantizer, quantize_APoT, dequantize_APoT, unittest, random


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/quantization/core/experimental`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.ao.quantization.observer`: MinMaxObserver
- `torch.ao.quantization.experimental.observer`: APoTObserver
- `torch.ao.quantization.experimental.quantizer`: APoTQuantizer, quantize_APoT, dequantize_APoT
- `unittest`
- `random`


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
python test/quantization/core/experimental/test_quantizer.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/quantization/core/experimental`):

- [`test_adaround_eager.py_docs.md`](./test_adaround_eager.py_docs.md)
- [`test_fake_quantize.py_docs.md`](./test_fake_quantize.py_docs.md)
- [`test_floatx.py_docs.md`](./test_floatx.py_docs.md)
- [`test_bits.py_docs.md`](./test_bits.py_docs.md)
- [`apot_fx_graph_mode_qat.py_docs.md`](./apot_fx_graph_mode_qat.py_docs.md)
- [`apot_fx_graph_mode_ptq.py_docs.md`](./apot_fx_graph_mode_ptq.py_docs.md)
- [`test_quantized_tensor.py_docs.md`](./test_quantized_tensor.py_docs.md)
- [`test_nonuniform_observer.py_docs.md`](./test_nonuniform_observer.py_docs.md)
- [`test_linear.py_docs.md`](./test_linear.py_docs.md)


## Cross-References

- **File Documentation**: `test_quantizer.py_docs.md`
- **Keyword Index**: `test_quantizer.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
