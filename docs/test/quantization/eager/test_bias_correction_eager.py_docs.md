# Documentation: `test/quantization/eager/test_bias_correction_eager.py`

## File Metadata

- **Path**: `test/quantization/eager/test_bias_correction_eager.py`
- **Size**: 4,429 bytes (4.33 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: quantization"]

import copy

import torch
import torch.ao.ns._numeric_suite as ns
import torch.nn as nn
from torch.ao.quantization import default_qconfig, QuantWrapper
from torch.ao.quantization._correct_bias import (
    _supported_modules,
    _supported_modules_quantized,
    bias_correction,
    get_module,
    get_param,
    parent_child_names,
)
from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    skipIfNoFBGEMM,
)
from torch.testing._internal.common_utils import raise_on_run_directly


class TestBiasCorrectionEager(QuantizationTestCase):
    def compute_sqnr(self, x, y):
        Ps = torch.norm(x)
        Pn = torch.norm(x - y)
        return 20 * torch.log10(Ps / Pn)

    def correct_artificial_bias_quantize(self, float_model, img_data):
        """Adding artificial bias and testing if bias persists after bias
        correction. This test case changes the bias of a quantized submodule
        """
        artificial_model = copy.deepcopy(float_model)
        artificial_model.qconfig = default_qconfig
        torch.ao.quantization.prepare(artificial_model, inplace=True)
        for data in img_data:
            artificial_model(data[0])
        torch.ao.quantization.convert(artificial_model, inplace=True)

        # manually changing bias
        for submodule in artificial_model.modules():
            if type(submodule) in _supported_modules:
                x = get_param(submodule, "bias")
                weight = get_param(submodule, "weight")
                if x is not None:
                    submodule.set_weight_bias(weight, x.data * 3)

        bias_correction(
            float_model,
            artificial_model,
            img_data,
            target_modules=_supported_modules_quantized,
        )

        # Trims off the shadow module,
        for name, submodule in artificial_model.named_modules():
            if isinstance(submodule, ns.Shadow):
                parent_name, child_name = parent_child_names(name)
                parent = get_module(artificial_model, parent_name)
                parent._modules[child_name] = submodule.orig_module

        for name, artificial_submodule in artificial_model.named_modules():
            if type(artificial_submodule) in _supported_modules_quantized:
                submodule = get_module(float_model, name)
                float_bias = get_param(submodule, "bias")
                artificial_bias = get_param(artificial_submodule, "bias")

                self.assertTrue(
                    self.compute_sqnr(float_bias, artificial_bias) > 30,
                    "Correcting quantized bias produced too much noise, sqnr score too low",
                )

    @skipIfNoFBGEMM
    def test_linear_chain(self):
        class LinearChain(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = nn.Linear(3, 4)
                self.linear2 = nn.Linear(4, 5)
                self.linear3 = nn.Linear(5, 6)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                return x

        float_model = QuantWrapper(LinearChain())
        img_data = [
            (
                torch.rand(10, 3, dtype=torch.float),
                torch.randint(0, 1, (2,), dtype=torch.long),
            )
            for _ in range(50)
        ]
        self.correct_artificial_bias_quantize(float_model, img_data)

    @skipIfNoFBGEMM
    def test_conv_chain(self):
        class ConvChain(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv2d1 = nn.Conv2d(3, 4, 5, 5)
                self.conv2d2 = nn.Conv2d(4, 5, 5, 5)
                self.conv2d3 = nn.Conv2d(5, 6, 5, 5)

            def forward(self, x):
                x = self.conv2d1(x)
                x = self.conv2d2(x)
                x = self.conv2d3(x)
                return x

        float_model = QuantWrapper(ConvChain())
        img_data = [
            (
                torch.rand(10, 3, 125, 125, dtype=torch.float),
                torch.randint(0, 1, (2,), dtype=torch.long),
            )
            for _ in range(50)
        ]
        self.correct_artificial_bias_quantize(float_model, img_data)


if __name__ == "__main__":
    raise_on_run_directly("test/test_quantization.py")

```



## High-Level Overview

"""Adding artificial bias and testing if bias persists after bias        correction. This test case changes the bias of a quantized submodule

This Python file contains 3 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestBiasCorrectionEager`, `LinearChain`, `ConvChain`

**Functions defined**: `compute_sqnr`, `correct_artificial_bias_quantize`, `test_linear_chain`, `__init__`, `forward`, `test_conv_chain`, `__init__`, `forward`

**Key imports**: copy, torch, torch.ao.ns._numeric_suite as ns, torch.nn as nn, default_qconfig, QuantWrapper, raise_on_run_directly


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/quantization/eager`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `torch`
- `torch.ao.ns._numeric_suite as ns`
- `torch.nn as nn`
- `torch.ao.quantization`: default_qconfig, QuantWrapper
- `torch.testing._internal.common_utils`: raise_on_run_directly


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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
python test/quantization/eager/test_bias_correction_eager.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/quantization/eager`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_quantize_eager_ptq.py_docs.md`](./test_quantize_eager_ptq.py_docs.md)
- [`test_quantize_eager_qat.py_docs.md`](./test_quantize_eager_qat.py_docs.md)
- [`test_equalize_eager.py_docs.md`](./test_equalize_eager.py_docs.md)
- [`test_fuse_eager.py_docs.md`](./test_fuse_eager.py_docs.md)
- [`test_model_numerics.py_docs.md`](./test_model_numerics.py_docs.md)
- [`test_numeric_suite_eager.py_docs.md`](./test_numeric_suite_eager.py_docs.md)


## Cross-References

- **File Documentation**: `test_bias_correction_eager.py_docs.md`
- **Keyword Index**: `test_bias_correction_eager.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
