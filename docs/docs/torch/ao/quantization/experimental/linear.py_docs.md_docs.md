# Documentation: `docs/torch/ao/quantization/experimental/linear.py_docs.md`

## File Metadata

- **Path**: `docs/torch/ao/quantization/experimental/linear.py_docs.md`
- **Size**: 9,671 bytes (9.44 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/ao/quantization/experimental/linear.py`

## File Metadata

- **Path**: `torch/ao/quantization/experimental/linear.py`
- **Size**: 5,933 bytes (5.79 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import numpy as np
import numpy.typing as npt

import torch
from torch.ao.nn.quantized.modules.utils import WeightedQuantizedModule
from torch.ao.quantization.experimental.observer import APoTObserver
from torch.ao.quantization.experimental.quantizer import quantize_APoT


class LinearAPoT(WeightedQuantizedModule):
    r"""
    A quantized linear module with quantized tensor as inputs and outputs
    to support APoT quantization.
    We adopt the same interface as `torch.nn.Linear`, see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear for documentation.

    Similar to :class:`~torch.nn.Linear`, attributes will be randomly
    initialized at module creation time and will be overwritten later

    Attributes:
        alpha: `alpha` qparam of output Quantized Tensor, type: Tensor
        gamma: `gamma` qparam of output Quantized Tensor, type: Tensor
        quantization_levels: `quantization_levels` qparam of output Quantized Tensor, type: Tensor
        level_indices: `level_indices` qparam of output Quantized Tensor, type: Tensor
        weight: APoT quantized tensor from weight2quantize
        weight_transposed: transposed weight tensor, used in linear transformation calculation (y = x * A^T + b)
    """

    def __init__(self, weight2quantize: torch.Tensor, b: int, k: int):
        if weight2quantize.dim() != 2:
            raise AssertionError(
                f"weight2quantize must be a 2-D tensor, got dim={weight2quantize.dim()}"
            )
        if b % k != 0:
            raise AssertionError(f"b must be divisible by k, got b={b}, k={k}")

        super().__init__()

        self.b = b
        self.k = k
        self.n = self.b // self.k

        observer = APoTObserver(b=self.b, k=self.k)

        observer(weight2quantize)

        (
            self.alpha,
            self.gamma,
            self.quantization_levels,
            self.level_indices,
        ) = observer.calculate_qparams(signed=False)

        quantized_weight = quantize_APoT(
            weight2quantize,
            self.alpha,
            self.gamma,
            self.quantization_levels,
            self.level_indices,
        )
        self.weight = quantized_weight.data
        self.weight_transposed = torch.transpose(self.weight, 0, 1)

    def decompose_APoT(self, x):
        r"""
        Decompose binary representation of APoT values into list of k-sized blocks
        Args:
            x (Tensor): binary representation of APoT quantized tensor
        """
        # remove "0b" prefix from binary representation
        x = x[2:]

        # initialize list of blocks
        blocks = []

        while x:
            blocks.append(x[0 : self.k])
            x = x[self.k :]

        return blocks

    def bitshift_mul(self, weight_val, r):
        r"""
        Compute multiplication of weight_val * r using bitshifting
        method discussed in APoT paper: https://arxiv.org/pdf/1909.13144.pdf
        Args:
            weight_val: list of binary digits representing APoT quantized weight value
            r: int representing uniformly quantized activation value
        """
        product = 0

        idx = len(weight_val) - 1
        place = 0

        while idx >= 0:
            block = weight_val[idx]

            # reverse digits in block
            block = block[::-1]

            curr_block_result = 0

            for ele in block:
                if int(ele):
                    curr_block_result += r << place
                place += 1

            idx -= 1
            product += curr_block_result

        return product

    def matmul(self, decomposed_weight, activation):
        r"""
        Perform matrix multiplication between decomposed_weight and
        activation by calling bitshift_mul function for each value
        Args:
            decomposed_weight (Tensor): APoT quantized weight decomposed into binary
            activation (Tensor): uniformly quantized activation
        """
        rows1 = activation.size(dim=0)
        rows2 = decomposed_weight.shape[0]
        cols2 = decomposed_weight.shape[1]

        result = torch.zeros(rows1, cols2)

        # compute matrix multiplication with bitshifts
        for i in range(rows1):
            for j in range(cols2):
                for k in range(rows2):
                    weight_val = decomposed_weight[k][j]
                    r = int(activation[i][k])

                    product = self.bitshift_mul(weight_val, r)

                    result[i][j] += product

        return result

    def forward(self, activation: torch.Tensor) -> torch.FloatTensor:
        r"""
        Multiply APoT quantized weight and uniformly quantized activation (dtype: quint8)
        with bitshifting instead of matrix multiplication.
        Result has dtype torch.float32
        Args:
            activation (Tensor): uniformly quantized activation tensor
        """
        if activation.dim() != 2:
            raise AssertionError(
                f"activation must be a 2-D tensor, got dim={activation.dim()}"
            )

        weight_rows = self.weight_transposed.size()[0]
        weight_cols = self.weight_transposed.size()[1]

        decomposed_weight: npt.NDArray = np.empty(
            shape=(weight_rows, weight_cols), dtype=object
        )
        for row in range(weight_rows):
            for col in range(weight_cols):
                decomposed_weight[row][col] = self.decompose_APoT(
                    bin(self.weight_transposed[row][col])
                )

        result = self.matmul(decomposed_weight, activation).type(torch.FloatTensor)

        return result

    @classmethod
    def from_reference(  # type: ignore[override]
        cls,
        ref_qlinear,
        alpha: torch.Tensor,
        gamma: torch.Tensor,
        quantization_levels: torch.Tensor,
        level_indices: torch.Tensor,
    ):
        raise NotImplementedError

```



## High-Level Overview

r"""    A quantized linear module with quantized tensor as inputs and outputs    to support APoT quantization.    We adopt the same interface as `torch.nn.Linear`, see    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear for documentation.    Similar to :class:`~torch.nn.Linear`, attributes will be randomly    initialized at module creation time and will be overwritten later    Attributes:        alpha: `alpha` qparam of output Quantized Tensor, type: Tensor        gamma: `gamma` qparam of output Quantized Tensor, type: Tensor        quantization_levels: `quantization_levels` qparam of output Quantized Tensor, type: Tensor        level_indices: `level_indices` qparam of output Quantized Tensor, type: Tensor        weight: APoT quantized tensor from weight2quantize        weight_transposed: transposed weight tensor, used in linear transformation calculation (y = x * A^T + b)

This Python file contains 1 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LinearAPoT`

**Functions defined**: `__init__`, `decompose_APoT`, `bitshift_mul`, `matmul`, `forward`, `from_reference`

**Key imports**: numpy as np, numpy.typing as npt, torch, WeightedQuantizedModule, APoTObserver, quantize_APoT


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/quantization/experimental`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `numpy as np`
- `numpy.typing as npt`
- `torch`
- `torch.ao.nn.quantized.modules.utils`: WeightedQuantizedModule
- `torch.ao.quantization.experimental.observer`: APoTObserver
- `torch.ao.quantization.experimental.quantizer`: quantize_APoT


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/ao/quantization/experimental`):

- [`fake_quantize.py_docs.md`](./fake_quantize.py_docs.md)
- [`adaround_fake_quantize.py_docs.md`](./adaround_fake_quantize.py_docs.md)
- [`adaround_loss.py_docs.md`](./adaround_loss.py_docs.md)
- [`observer.py_docs.md`](./observer.py_docs.md)
- [`quantizer.py_docs.md`](./quantizer.py_docs.md)
- [`fake_quantize_function.py_docs.md`](./fake_quantize_function.py_docs.md)
- [`APoT_tensor.py_docs.md`](./APoT_tensor.py_docs.md)
- [`qconfig.py_docs.md`](./qconfig.py_docs.md)
- [`adaround_optimization.py_docs.md`](./adaround_optimization.py_docs.md)


## Cross-References

- **File Documentation**: `linear.py_docs.md`
- **Keyword Index**: `linear.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/ao/quantization/experimental`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/ao/quantization/experimental`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/ao/quantization/experimental`):

- [`qconfig.py_docs.md_docs.md`](./qconfig.py_docs.md_docs.md)
- [`apot_utils.py_docs.md_docs.md`](./apot_utils.py_docs.md_docs.md)
- [`APoT_tensor.py_kw.md_docs.md`](./APoT_tensor.py_kw.md_docs.md)
- [`fake_quantize_function.py_kw.md_docs.md`](./fake_quantize_function.py_kw.md_docs.md)
- [`adaround_loss.py_docs.md_docs.md`](./adaround_loss.py_docs.md_docs.md)
- [`apot_utils.py_kw.md_docs.md`](./apot_utils.py_kw.md_docs.md)
- [`adaround_fake_quantize.py_docs.md_docs.md`](./adaround_fake_quantize.py_docs.md_docs.md)
- [`APoT_tensor.py_docs.md_docs.md`](./APoT_tensor.py_docs.md_docs.md)
- [`observer.py_kw.md_docs.md`](./observer.py_kw.md_docs.md)
- [`observer.py_docs.md_docs.md`](./observer.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `linear.py_docs.md_docs.md`
- **Keyword Index**: `linear.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
