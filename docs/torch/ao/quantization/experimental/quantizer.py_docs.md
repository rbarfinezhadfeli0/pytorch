# Documentation: `torch/ao/quantization/experimental/quantizer.py`

## File Metadata

- **Path**: `torch/ao/quantization/experimental/quantizer.py`
- **Size**: 5,597 bytes (5.47 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import numpy as np

import torch
from torch import Tensor
from torch.ao.quantization.experimental.apot_utils import (
    apot_to_float,
    float_to_apot,
    quant_dequant_util,
)


# class to store APoT quantizer and
# implement quantize and dequantize
class APoTQuantizer:
    alpha: torch.Tensor
    gamma: torch.Tensor
    quantization_levels: torch.Tensor
    level_indices: torch.Tensor

    def __init__(
        self,
        alpha: torch.Tensor,
        gamma: torch.Tensor,
        quantization_levels: torch.Tensor,
        level_indices: torch.Tensor,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.quantization_levels = quantization_levels
        self.level_indices = level_indices

    r""" Quantizes fp Tensor to integer APoT representation.
    Conversion is based on the qparams from a specified APoT non-uniform observer.
    The approach follows the method outlined in the APoT paper: https://arxiv.org/pdf/1909.13144.pdf.
    Args:
        tensor2quantize: fp Tensor
    Returns:
        result: APoT Tensor representation of tensor2quantize
    """

    def quantize(self, tensor2quantize: Tensor):
        result = torch.tensor([])

        # map float_to_apot over tensor2quantize elements
        tensor2quantize = tensor2quantize.detach().apply_(
            lambda x: float_to_apot(
                x, self.quantization_levels, self.level_indices, self.alpha
            )
        )

        # convert to APoT int representation for dtype
        tensor2quantize = tensor2quantize.int()

        from torch.ao.quantization.experimental.APoT_tensor import TensorAPoT

        result = TensorAPoT(self, tensor2quantize)  # type: ignore[assignment]

        return result

    r""" Dequantizes integer Tensor to floating point (fp) representation
    based on the calculated quantization levels from a specified APoT non-uniform observer.
    The approach follows the method outlined in the APoT paper: https://arxiv.org/pdf/1909.13144.pdf.
    Args:
        tensor2quantize: fp Tensor
    Returns:
        result: fp reduced precision representation of input Tensor
    """

    def dequantize(self, apot_tensor) -> Tensor:
        orig_size = apot_tensor.data.size()
        apot_tensor_data = apot_tensor.data.flatten()

        print(apot_tensor_data)

        # map apot_to_float over tensor2quantize elements
        result_temp = np.empty(shape=apot_tensor_data.size())
        for i in range(len(apot_tensor_data)):
            new_ele = apot_to_float(
                apot_tensor_data[i], self.quantization_levels, self.level_indices
            )
            result_temp[i] = new_ele

        result = torch.from_numpy(result_temp).reshape(orig_size)

        return result

    r""" Returns result of quantize -> dequantize on a fp Tensor (reduced precision)
    based on the calculated quantization levels from a specified APoT non-uniform observer.
    The approach follows the method outlined in the APoT paper: https://arxiv.org/pdf/1909.13144.pdf.
    Args:
        apot_tensor: quantized APoT Tensor to dequantize
    Returns:
        result: fp representation of input Tensor
    """

    def quant_dequant(self, tensor2quantize: Tensor) -> Tensor:
        levels_lst = list(self.quantization_levels)

        result = tensor2quantize.apply_(lambda x: quant_dequant_util(x, levels_lst))  # type: ignore[call-arg]

        return result

    def q_apot_alpha(self) -> float:
        raise NotImplementedError


r""" Global method to create quantizer and call quantizer quantize_APoT
    Args:
        tensor2quantize: fp Tensor to quantize
        alpha: Tensor qparam alpha (clipping level)
        gamma: Tensor qparam gamma (scale factor for quantization levels)
        quantization levels: Tensor with fp quantization levels
        level indices: Tensor with integer quantization level indices
    Returns:
        result: ApoT Tensor representation of tensor2quantize
"""


def quantize_APoT(
    tensor2quantize: Tensor,
    alpha: Tensor,
    gamma: Tensor,
    quantization_levels: Tensor,
    level_indices: Tensor,
):
    quantizer = APoTQuantizer(
        alpha=alpha,
        gamma=gamma,
        quantization_levels=quantization_levels,
        level_indices=level_indices,
    )
    result = quantizer.quantize(tensor2quantize)
    return result


r""" Global method to create quantizer and call quantizer dequantize_APoT
    Args:
        apot_tensor: APoT Tensor to dequantize
    Returns:
        result: fp Tensor dequantized from apot_tensor
"""


def dequantize_APoT(apot_tensor) -> Tensor:
    quantizer = apot_tensor.quantizer
    result = quantizer.dequantize(apot_tensor)
    return result


r""" Global method to create quantizer and call quantizer quant_dequant
    Args:
        tensor2quantize: fp Tensor to quantize
        alpha: Tensor qparam alpha (clipping level)
        gamma: Tensor qparam gamma (scale factor for quantization levels)
        quantization levels: Tensor with fp quantization levels
        level indices: Tensor with integer quantization level indices
    Returns:
        result: fp reduced precision Tensor from tensor2quantize
"""


def quant_dequant_APoT(
    tensor2quantize: Tensor,
    alpha: Tensor,
    gamma: Tensor,
    quantization_levels: Tensor,
    level_indices: Tensor,
) -> Tensor:
    quantizer = APoTQuantizer(
        alpha=alpha,
        gamma=gamma,
        quantization_levels=quantization_levels,
        level_indices=level_indices,
    )
    result = quantizer.quant_dequant(tensor2quantize)
    return result

```



## High-Level Overview

r""" Quantizes fp Tensor to integer APoT representation.    Conversion is based on the qparams from a specified APoT non-uniform observer.    The approach follows the method outlined in the APoT paper: https://arxiv.org/pdf/1909.13144.pdf.    Args:        tensor2quantize: fp Tensor    Returns:        result: APoT Tensor representation of tensor2quantize

This Python file contains 2 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `APoTQuantizer`

**Functions defined**: `__init__`, `quantize`, `dequantize`, `quant_dequant`, `q_apot_alpha`, `quantize_APoT`, `dequantize_APoT`, `quant_dequant_APoT`

**Key imports**: numpy as np, torch, Tensor, TensorAPoT


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/quantization/experimental`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `numpy as np`
- `torch`
- `torch.ao.quantization.experimental.APoT_tensor`: TensorAPoT


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/ao/quantization/experimental`):

- [`fake_quantize.py_docs.md`](./fake_quantize.py_docs.md)
- [`adaround_fake_quantize.py_docs.md`](./adaround_fake_quantize.py_docs.md)
- [`linear.py_docs.md`](./linear.py_docs.md)
- [`adaround_loss.py_docs.md`](./adaround_loss.py_docs.md)
- [`observer.py_docs.md`](./observer.py_docs.md)
- [`fake_quantize_function.py_docs.md`](./fake_quantize_function.py_docs.md)
- [`APoT_tensor.py_docs.md`](./APoT_tensor.py_docs.md)
- [`qconfig.py_docs.md`](./qconfig.py_docs.md)
- [`adaround_optimization.py_docs.md`](./adaround_optimization.py_docs.md)


## Cross-References

- **File Documentation**: `quantizer.py_docs.md`
- **Keyword Index**: `quantizer.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
