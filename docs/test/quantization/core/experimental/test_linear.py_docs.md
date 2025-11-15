# Documentation: `test/quantization/core/experimental/test_linear.py`

## File Metadata

- **Path**: `test/quantization/core/experimental/test_linear.py`
- **Size**: 2,346 bytes (2.29 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# Owner(s): ["oncall: quantization"]

import torch
from torch.ao.quantization.experimental.linear import LinearAPoT
from torch.nn.modules.linear import Linear
import unittest

class TestNonUniformObserver(unittest.TestCase):
    """
        Test linear_APoT_fn by comparing to uniform linear
        for 2d tensors with size (4,4) and k=1
    """
    def test_linear_APoT_k1(self):
        # weight: fp tensor
        weight = 1000 * torch.rand(4, 4)

        # activation: fp32 tensor with ~ integer values
        activation = torch.randint(low=0, high=255, size=(4, 4), dtype=torch.float)

        # calculate result from calling linear forward method
        apot_linear = LinearAPoT(weight, 8, 1)
        apot_linear_result = apot_linear(activation)

        # calculate expected results
        fp_linear = Linear(4, 4, bias=False)

        # set weight for fp linear
        apot_quantized_weight_float = apot_linear.weight.type(torch.FloatTensor)
        fp_linear_weight = torch.nn.parameter.Parameter(data=apot_quantized_weight_float)
        fp_linear.weight = fp_linear_weight

        fp_linear_result = fp_linear(activation).data

        self.assertTrue(torch.equal(apot_linear_result, fp_linear_result))

    """
        Test linear_APoT_fn by comparing to uniform linear
        for 2d tensors with size (5,3), (3, 5) and k=2
    """
    def test_linear_APoT_k2(self):
        # weight: fp tensor
        weight = 1000 * torch.rand(5, 3)

        # activation: fp32 tensor with ~ integer values
        # note: transpose of activation matrix will have dimension (3, 5)
        activation = torch.randint(low=0, high=255, size=(5, 3), dtype=torch.float)

        # calculate result from calling linear forward method
        apot_linear = LinearAPoT(weight, 8, 2)
        apot_linear_result = apot_linear(activation)

        # calculate expected results
        fp_linear = Linear(4, 4, bias=False)

        # set weight for fp linear
        apot_quantized_weight_float = apot_linear.weight.type(torch.FloatTensor)
        fp_linear_weight = torch.nn.parameter.Parameter(data=apot_quantized_weight_float)
        fp_linear.weight = fp_linear_weight

        fp_linear_result = fp_linear(activation).data

        self.assertTrue(torch.equal(apot_linear_result, fp_linear_result))

if __name__ == '__main__':
    unittest.main()

```



## High-Level Overview

"""        Test linear_APoT_fn by comparing to uniform linear        for 2d tensors with size (4,4) and k=1

This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestNonUniformObserver`

**Functions defined**: `test_linear_APoT_k1`, `test_linear_APoT_k2`

**Key imports**: torch, LinearAPoT, Linear, unittest


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/quantization/core/experimental`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.ao.quantization.experimental.linear`: LinearAPoT
- `torch.nn.modules.linear`: Linear
- `unittest`


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


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
python test/quantization/core/experimental/test_linear.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/quantization/core/experimental`):

- [`test_adaround_eager.py_docs.md`](./test_adaround_eager.py_docs.md)
- [`test_fake_quantize.py_docs.md`](./test_fake_quantize.py_docs.md)
- [`test_floatx.py_docs.md`](./test_floatx.py_docs.md)
- [`test_quantizer.py_docs.md`](./test_quantizer.py_docs.md)
- [`test_bits.py_docs.md`](./test_bits.py_docs.md)
- [`apot_fx_graph_mode_qat.py_docs.md`](./apot_fx_graph_mode_qat.py_docs.md)
- [`apot_fx_graph_mode_ptq.py_docs.md`](./apot_fx_graph_mode_ptq.py_docs.md)
- [`test_quantized_tensor.py_docs.md`](./test_quantized_tensor.py_docs.md)
- [`test_nonuniform_observer.py_docs.md`](./test_nonuniform_observer.py_docs.md)


## Cross-References

- **File Documentation**: `test_linear.py_docs.md`
- **Keyword Index**: `test_linear.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
