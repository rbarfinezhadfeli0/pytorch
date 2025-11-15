# Documentation: `docs/torch/ao/quantization/experimental/adaround_loss.py_docs.md`

## File Metadata

- **Path**: `docs/torch/ao/quantization/experimental/adaround_loss.py_docs.md`
- **Size**: 6,067 bytes (5.92 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/ao/quantization/experimental/adaround_loss.py`

## File Metadata

- **Path**: `torch/ao/quantization/experimental/adaround_loss.py`
- **Size**: 3,264 bytes (3.19 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import numpy as np

import torch
from torch.nn import functional as F


ADAROUND_ZETA: float = 1.1
ADAROUND_GAMMA: float = -0.1


class AdaptiveRoundingLoss(torch.nn.Module):
    """
    Adaptive Rounding Loss functions described in https://arxiv.org/pdf/2004.10568.pdf
    rounding regularization is eq [24]
    reconstruction loss is eq [25] except regularization term
    """

    def __init__(
        self,
        max_iter: int,
        warm_start: float = 0.2,
        beta_range: tuple[int, int] = (20, 2),
        reg_param: float = 0.001,
    ) -> None:
        super().__init__()
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.beta_range = beta_range
        self.reg_param = reg_param

    def rounding_regularization(
        self,
        V: torch.Tensor,
        curr_iter: int,
    ) -> torch.Tensor:
        """
        Major logics copied from official Adaround Implementation.
        Apply rounding regularization to the input tensor V.
        """
        if curr_iter >= self.max_iter:
            raise AssertionError("Current iteration strictly les sthan max iteration")
        if curr_iter < self.warm_start * self.max_iter:
            return torch.tensor(0.0)
        else:
            start_beta, end_beta = self.beta_range
            warm_start_end_iter = self.warm_start * self.max_iter

            # compute relative iteration of current iteration
            rel_iter = (curr_iter - warm_start_end_iter) / (
                self.max_iter - warm_start_end_iter
            )
            beta = end_beta + 0.5 * (start_beta - end_beta) * (
                1 + np.cos(rel_iter * np.pi)
            )

            # A rectified sigmoid for soft-quantization as formulated [23] in https://arxiv.org/pdf/2004.10568.pdf
            h_alpha = torch.clamp(
                torch.sigmoid(V) * (ADAROUND_ZETA - ADAROUND_GAMMA) + ADAROUND_GAMMA,
                min=0,
                max=1,
            )

            # Apply rounding regularization
            # This regularization term helps out term to converge into binary solution either 0 or 1 at the end of optimization.
            inner_term = torch.add(2 * h_alpha, -1).abs().pow(beta)
            regularization_term = torch.add(1, -inner_term).sum()
            return regularization_term * self.reg_param

    def reconstruction_loss(
        self,
        soft_quantized_output: torch.Tensor,
        original_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the reconstruction loss between the soft quantized output and the original output.
        """
        return F.mse_loss(
            soft_quantized_output, original_output, reduction="none"
        ).mean()

    def forward(
        self,
        soft_quantized_output: torch.Tensor,
        original_output: torch.Tensor,
        V: torch.Tensor,
        curr_iter: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the asymmetric reconstruction formulation as eq [25]
        """
        regularization_term = self.rounding_regularization(V, curr_iter)
        reconstruction_term = self.reconstruction_loss(
            soft_quantized_output, original_output
        )
        return regularization_term, reconstruction_term

```



## High-Level Overview

"""    Adaptive Rounding Loss functions described in https://arxiv.org/pdf/2004.10568.pdf    rounding regularization is eq [24]    reconstruction loss is eq [25] except regularization term

This Python file contains 1 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `AdaptiveRoundingLoss`

**Functions defined**: `__init__`, `rounding_regularization`, `reconstruction_loss`, `forward`

**Key imports**: numpy as np, torch, functional as F


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/quantization/experimental`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `numpy as np`
- `torch`
- `torch.nn`: functional as F


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
- [`linear.py_docs.md`](./linear.py_docs.md)
- [`observer.py_docs.md`](./observer.py_docs.md)
- [`quantizer.py_docs.md`](./quantizer.py_docs.md)
- [`fake_quantize_function.py_docs.md`](./fake_quantize_function.py_docs.md)
- [`APoT_tensor.py_docs.md`](./APoT_tensor.py_docs.md)
- [`qconfig.py_docs.md`](./qconfig.py_docs.md)
- [`adaround_optimization.py_docs.md`](./adaround_optimization.py_docs.md)


## Cross-References

- **File Documentation**: `adaround_loss.py_docs.md`
- **Keyword Index**: `adaround_loss.py_kw.md`
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
- [`apot_utils.py_kw.md_docs.md`](./apot_utils.py_kw.md_docs.md)
- [`adaround_fake_quantize.py_docs.md_docs.md`](./adaround_fake_quantize.py_docs.md_docs.md)
- [`APoT_tensor.py_docs.md_docs.md`](./APoT_tensor.py_docs.md_docs.md)
- [`observer.py_kw.md_docs.md`](./observer.py_kw.md_docs.md)
- [`observer.py_docs.md_docs.md`](./observer.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `adaround_loss.py_docs.md_docs.md`
- **Keyword Index**: `adaround_loss.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
