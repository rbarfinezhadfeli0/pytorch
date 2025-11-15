# Documentation: `docs/torch/ao/nn/quantized/modules/__init__.py_docs.md`

## File Metadata

- **Path**: `docs/torch/ao/nn/quantized/modules/__init__.py_docs.md`
- **Size**: 7,442 bytes (7.27 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This is a **Python package initialization file**.

## Original Source

```markdown
# Documentation: `torch/ao/nn/quantized/modules/__init__.py`

## File Metadata

- **Path**: `torch/ao/nn/quantized/modules/__init__.py`
- **Size**: 4,521 bytes (4.42 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a **Python package initialization file**.

## Original Source

```python
# mypy: allow-untyped-defs
import torch

# The quantized modules use `torch.nn` and `torch.ao.nn.quantizable`
# packages. However, the `quantizable` package uses "lazy imports"
# to avoid circular dependency.
# Hence we need to include it here to make sure it is resolved before
# they are used in the modules.
import torch.ao.nn.quantizable
from torch.nn.modules.pooling import MaxPool2d

from .activation import (
    ELU,
    Hardswish,
    LeakyReLU,
    MultiheadAttention,
    PReLU,
    ReLU6,
    Sigmoid,
    Softmax,
)
from .batchnorm import BatchNorm2d, BatchNorm3d
from .conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)
from .dropout import Dropout
from .embedding_ops import Embedding, EmbeddingBag
from .functional_modules import FloatFunctional, FXFloatFunctional, QFunctional
from .linear import Linear
from .normalization import (
    GroupNorm,
    InstanceNorm1d,
    InstanceNorm2d,
    InstanceNorm3d,
    LayerNorm,
)
from .rnn import LSTM


__all__ = [
    "BatchNorm2d",
    "BatchNorm3d",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "DeQuantize",
    "ELU",
    "Embedding",
    "EmbeddingBag",
    "GroupNorm",
    "Hardswish",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "LayerNorm",
    "LeakyReLU",
    "Linear",
    "LSTM",
    "MultiheadAttention",
    "Quantize",
    "ReLU6",
    "Sigmoid",
    "Softmax",
    "Dropout",
    "PReLU",
    # Wrapper modules
    "FloatFunctional",
    "FXFloatFunctional",
    "QFunctional",
]


class Quantize(torch.nn.Module):
    r"""Quantizes an incoming tensor

    Args:
     `scale`: scale of the output Quantized Tensor
     `zero_point`: zero_point of output Quantized Tensor
     `dtype`: data type of output Quantized Tensor
     `factory_kwargs`: Dictionary of kwargs used for configuring initialization
         of internal buffers. Currently, `device` and `dtype` are supported.
         Example: `factory_kwargs={'device': 'cuda', 'dtype': torch.float64}`
         will initialize internal buffers as type `torch.float64` on the current CUDA device.
         Note that `dtype` only applies to floating-point buffers.

    Examples::
        >>> t = torch.tensor([[1., -1.], [1., -1.]])
        >>> scale, zero_point, dtype = 1.0, 2, torch.qint8
        >>> qm = Quantize(scale, zero_point, dtype)
        >>> # xdoctest: +SKIP
        >>> qt = qm(t)
        >>> print(qt)
        tensor([[ 1., -1.],
                [ 1., -1.]], size=(2, 2), dtype=torch.qint8, scale=1.0, zero_point=2)
    """

    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(self, scale, zero_point, dtype, factory_kwargs=None):
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        super().__init__()
        self.register_buffer("scale", torch.tensor([scale], **factory_kwargs))
        self.register_buffer(
            "zero_point",
            torch.tensor(
                [zero_point],
                dtype=torch.long,
                **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
            ),
        )
        self.dtype = dtype

    def forward(self, X):
        return torch.quantize_per_tensor(
            X, float(self.scale), int(self.zero_point), self.dtype
        )

    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        assert hasattr(mod, "activation_post_process")
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return Quantize(
            scale.float().item(),
            zero_point.long().item(),
            mod.activation_post_process.dtype,
        )

    def extra_repr(self):
        return f"scale={self.scale}, zero_point={self.zero_point}, dtype={self.dtype}"


class DeQuantize(torch.nn.Module):
    r"""Dequantizes an incoming tensor

    Examples::
        >>> input = torch.tensor([[1., -1.], [1., -1.]])
        >>> scale, zero_point, dtype = 1.0, 2, torch.qint8
        >>> qm = Quantize(scale, zero_point, dtype)
        >>> # xdoctest: +SKIP
        >>> quantized_input = qm(input)
        >>> dqm = DeQuantize()
        >>> dequantized = dqm(quantized_input)
        >>> print(dequantized)
        tensor([[ 1., -1.],
                [ 1., -1.]], dtype=torch.float32)
    """

    def forward(self, Xq):
        return Xq.dequantize()

    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        return DeQuantize()

```



## High-Level Overview


This Python file contains 2 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Quantize`, `DeQuantize`

**Functions defined**: `__init__`, `forward`, `from_float`, `extra_repr`, `forward`, `from_float`

**Key imports**: torch, torch.ao.nn.quantizable, MaxPool2d, BatchNorm2d, BatchNorm3d, Dropout, Embedding, EmbeddingBag, FloatFunctional, FXFloatFunctional, QFunctional, Linear, LSTM


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/nn/quantized/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.ao.nn.quantizable`
- `torch.nn.modules.pooling`: MaxPool2d
- `.batchnorm`: BatchNorm2d, BatchNorm3d
- `.dropout`: Dropout
- `.embedding_ops`: Embedding, EmbeddingBag
- `.functional_modules`: FloatFunctional, FXFloatFunctional, QFunctional
- `.linear`: Linear
- `.rnn`: LSTM


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`torch/ao/nn/quantized/modules`):

- [`rnn.py_docs.md`](./rnn.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`embedding_ops.py_docs.md`](./embedding_ops.py_docs.md)
- [`conv.py_docs.md`](./conv.py_docs.md)
- [`linear.py_docs.md`](./linear.py_docs.md)
- [`functional_modules.py_docs.md`](./functional_modules.py_docs.md)
- [`normalization.py_docs.md`](./normalization.py_docs.md)
- [`activation.py_docs.md`](./activation.py_docs.md)
- [`dropout.py_docs.md`](./dropout.py_docs.md)


## Cross-References

- **File Documentation**: `__init__.py_docs.md`
- **Keyword Index**: `__init__.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/ao/nn/quantized/modules`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/ao/nn/quantized/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/torch/ao/nn/quantized/modules`):

- [`embedding_ops.py_kw.md_docs.md`](./embedding_ops.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`functional_modules.py_docs.md_docs.md`](./functional_modules.py_docs.md_docs.md)
- [`functional_modules.py_kw.md_docs.md`](./functional_modules.py_kw.md_docs.md)
- [`embedding_ops.py_docs.md_docs.md`](./embedding_ops.py_docs.md_docs.md)
- [`rnn.py_docs.md_docs.md`](./rnn.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`linear.py_docs.md_docs.md`](./linear.py_docs.md_docs.md)
- [`conv.py_kw.md_docs.md`](./conv.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `__init__.py_docs.md_docs.md`
- **Keyword Index**: `__init__.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
