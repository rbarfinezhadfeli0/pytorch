# Documentation: `docs/torch/ao/nn/quantized/reference/modules/sparse.py_docs.md`

## File Metadata

- **Path**: `docs/torch/ao/nn/quantized/reference/modules/sparse.py_docs.md`
- **Size**: 7,442 bytes (7.27 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/ao/nn/quantized/reference/modules/sparse.py`

## File Metadata

- **Path**: `torch/ao/nn/quantized/reference/modules/sparse.py`
- **Size**: 4,683 bytes (4.57 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from typing import Any

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import ReferenceQuantizedModule


__all__ = ["Embedding", "EmbeddingBag"]


class Embedding(nn.Embedding, ReferenceQuantizedModule):
    """A reference quantized Embedding module that fits into the
    FX Graph Mode Quantization workflow, activation will be floating point Tensor,
    we will store floating point weight as well in the module, but in forward we'll
    quantize and dequantize the weight before running the floating point functional
    embedding operator.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Tensor | None = None,
        device=None,
        dtype=None,
        weight_qparams: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            # pyrefly: ignore [bad-argument-type]
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams, device)

    def _get_name(self):
        return "QuantizedEmbedding(Reference)"

    def forward(self, input: Tensor) -> Tensor:
        weight_quant_dequant = self.get_weight()
        return F.embedding(
            input,
            weight_quant_dequant,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    @classmethod
    def from_float(cls, mod, weight_qparams):
        return cls(
            mod.num_embeddings,
            mod.embedding_dim,
            mod.padding_idx,
            mod.max_norm,
            mod.norm_type,
            mod.scale_grad_by_freq,
            mod.sparse,
            mod.weight,
            mod.weight.device,
            mod.weight.dtype,
            weight_qparams,
        )


class EmbeddingBag(nn.EmbeddingBag, ReferenceQuantizedModule):
    """A reference quantized EmbeddingBag module that fits into the
    FX Graph Mode Quantization workflow, activation will be floating point Tensor,
    we will store floating point weight as well in the module, but in forward we'll
    quantize and dequantize the weight before running the floating point functional
    embedding operator.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        mode: str = "mean",
        sparse: bool = False,
        _weight: Tensor | None = None,
        include_last_offset: bool = False,
        padding_idx: int | None = None,
        device=None,
        dtype=None,
        weight_qparams: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            mode,
            sparse,
            _weight,
            include_last_offset,
            padding_idx,
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams, device)

    def _get_name(self):
        return "QuantizedEmbedding(Reference)"

    def forward(
        self,
        input: Tensor,
        offsets: Tensor | None = None,
        per_sample_weights: Tensor | None = None,
    ) -> Tensor:
        weight_quant_dequant = self.get_weight()
        return F.embedding_bag(
            input,
            weight_quant_dequant,
            offsets,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.mode,
            self.sparse,
            per_sample_weights,
            self.include_last_offset,
            self.padding_idx,
        )

    @classmethod
    def from_float(cls, mod, weight_qparams, use_precomputed_fake_quant=False):
        return cls(
            mod.num_embeddings,
            mod.embedding_dim,
            mod.max_norm,
            mod.norm_type,
            mod.scale_grad_by_freq,
            mod.mode,
            mod.sparse,
            mod.weight,
            mod.include_last_offset,
            mod.padding_idx,
            mod.weight.device,
            mod.weight.dtype,
            weight_qparams,
        )

```



## High-Level Overview

"""A reference quantized Embedding module that fits into the    FX Graph Mode Quantization workflow, activation will be floating point Tensor,    we will store floating point weight as well in the module, but in forward we'll    quantize and dequantize the weight before running the floating point functional    embedding operator.

This Python file contains 2 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Embedding`, `EmbeddingBag`

**Functions defined**: `__init__`, `_get_name`, `forward`, `from_float`, `__init__`, `_get_name`, `forward`, `from_float`

**Key imports**: Any, torch.nn as nn, torch.nn.functional as F, Tensor, ReferenceQuantizedModule


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/nn/quantized/reference/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Any
- `torch.nn as nn`
- `torch.nn.functional as F`
- `torch`: Tensor
- `.utils`: ReferenceQuantizedModule


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

Files in the same folder (`torch/ao/nn/quantized/reference/modules`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`rnn.py_docs.md`](./rnn.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`conv.py_docs.md`](./conv.py_docs.md)
- [`linear.py_docs.md`](./linear.py_docs.md)


## Cross-References

- **File Documentation**: `sparse.py_docs.md`
- **Keyword Index**: `sparse.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/ao/nn/quantized/reference/modules`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/ao/nn/quantized/reference/modules`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/ao/nn/quantized/reference/modules`):

- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`sparse.py_kw.md_docs.md`](./sparse.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`rnn.py_docs.md_docs.md`](./rnn.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`linear.py_docs.md_docs.md`](./linear.py_docs.md_docs.md)
- [`conv.py_kw.md_docs.md`](./conv.py_kw.md_docs.md)
- [`linear.py_kw.md_docs.md`](./linear.py_kw.md_docs.md)
- [`conv.py_docs.md_docs.md`](./conv.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `sparse.py_docs.md_docs.md`
- **Keyword Index**: `sparse.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
