# Documentation: `docs/torch/nn/quantized/modules/__init__.py_docs.md`

## File Metadata

- **Path**: `docs/torch/nn/quantized/modules/__init__.py_docs.md`
- **Size**: 4,846 bytes (4.73 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This is a **Python package initialization file**.

## Original Source

```markdown
# Documentation: `torch/nn/quantized/modules/__init__.py`

## File Metadata

- **Path**: `torch/nn/quantized/modules/__init__.py`
- **Size**: 2,101 bytes (2.05 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a **Python package initialization file**.

## Original Source

```python
r"""Quantized Modules.

Note::
    The `torch.nn.quantized` namespace is in the process of being deprecated.
    Please, use `torch.ao.nn.quantized` instead.
"""

# The following imports are needed in case the user decides
# to import the files directly,
# s.a. `from torch.nn.quantized.modules.conv import ...`.
# No need to add them to the `__all__`.
from torch.ao.nn.quantized.modules import (
    activation,
    batchnorm,
    conv,
    DeQuantize,
    dropout,
    embedding_ops,
    functional_modules,
    linear,
    MaxPool2d,
    normalization,
    Quantize,
    rnn,
    utils,
)
from torch.ao.nn.quantized.modules.activation import (
    ELU,
    Hardswish,
    LeakyReLU,
    MultiheadAttention,
    PReLU,
    ReLU6,
    Sigmoid,
    Softmax,
)
from torch.ao.nn.quantized.modules.batchnorm import BatchNorm2d, BatchNorm3d
from torch.ao.nn.quantized.modules.conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)
from torch.ao.nn.quantized.modules.dropout import Dropout
from torch.ao.nn.quantized.modules.embedding_ops import Embedding, EmbeddingBag
from torch.ao.nn.quantized.modules.functional_modules import (
    FloatFunctional,
    FXFloatFunctional,
    QFunctional,
)
from torch.ao.nn.quantized.modules.linear import Linear
from torch.ao.nn.quantized.modules.normalization import (
    GroupNorm,
    InstanceNorm1d,
    InstanceNorm2d,
    InstanceNorm3d,
    LayerNorm,
)
from torch.ao.nn.quantized.modules.rnn import LSTM


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

```



## High-Level Overview

r"""Quantized Modules.Note::    The `torch.nn.quantized` namespace is in the process of being deprecated.    Please, use `torch.ao.nn.quantized` instead.

This Python file contains 0 class(es) and 0 function(s).

## Detailed Analysis

### Code Structure

**Key imports**: the files directly,, ..., BatchNorm2d, BatchNorm3d, Dropout, Embedding, EmbeddingBag, Linear, LSTM


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nn/quantized/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `the files directly,`
- `...`
- `torch.ao.nn.quantized.modules.batchnorm`: BatchNorm2d, BatchNorm3d
- `torch.ao.nn.quantized.modules.dropout`: Dropout
- `torch.ao.nn.quantized.modules.embedding_ops`: Embedding, EmbeddingBag
- `torch.ao.nn.quantized.modules.linear`: Linear
- `torch.ao.nn.quantized.modules.rnn`: LSTM


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/nn/quantized/modules`):

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

This file is part of the PyTorch framework located at `docs/torch/nn/quantized/modules`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/nn/quantized/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

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

Files in the same folder (`docs/torch/nn/quantized/modules`):

- [`activation.py_kw.md_docs.md`](./activation.py_kw.md_docs.md)
- [`batchnorm.py_docs.md_docs.md`](./batchnorm.py_docs.md_docs.md)
- [`embedding_ops.py_kw.md_docs.md`](./embedding_ops.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`functional_modules.py_docs.md_docs.md`](./functional_modules.py_docs.md_docs.md)
- [`dropout.py_docs.md_docs.md`](./dropout.py_docs.md_docs.md)
- [`functional_modules.py_kw.md_docs.md`](./functional_modules.py_kw.md_docs.md)
- [`embedding_ops.py_docs.md_docs.md`](./embedding_ops.py_docs.md_docs.md)
- [`rnn.py_docs.md_docs.md`](./rnn.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `__init__.py_docs.md_docs.md`
- **Keyword Index**: `__init__.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
