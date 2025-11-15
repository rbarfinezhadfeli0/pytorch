# Documentation: `torch/ao/nn/qat/modules/embedding_ops.py`

## File Metadata

- **Path**: `torch/ao/nn/qat/modules/embedding_ops.py`
- **Size**: 7,867 bytes (7.68 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


__all__ = ["Embedding", "EmbeddingBag"]


class Embedding(nn.Embedding):
    r"""
    An embedding bag module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.Embedding`, please see
    https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding
    for documentation.

    Similar to `torch.nn.Embedding`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight: fake quant module for weight
    """

    _FLOAT_MODULE = nn.Embedding

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        device=None,
        dtype=None,
        qconfig=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
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
            **factory_kwargs,
        )
        assert qconfig, "qconfig must be provided for QAT module"
        assert qconfig.weight().qscheme == torch.per_channel_affine_float_qparams, (
            "Embedding weights requires a qscheme of torch.per_channel_affine_float_qparams Got "
            + str(qconfig.weight().qscheme)
        )
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)

    def forward(self, input) -> Tensor:
        return F.embedding(
            input,
            self.weight_fake_quant(self.weight),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Create a qat module from a float module

        Args: `mod` a float module, either produced by torch.ao.quantization utilities
        or directly from user
        """
        assert type(mod) is cls._FLOAT_MODULE, (
            " qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"
        weight_qscheme = mod.qconfig.weight().qscheme  # type: ignore[union-attr, operator]
        assert weight_qscheme == torch.per_channel_affine_float_qparams, (
            "Embedding weights requires a qscheme of torch.per_channel_affine_float_qparams Got "
            + str(weight_qscheme)
        )

        qconfig = mod.qconfig
        qat_embedding_bag = cls(
            mod.num_embeddings,
            mod.embedding_dim,
            mod.padding_idx,
            mod.max_norm,
            mod.norm_type,
            mod.scale_grad_by_freq,
            mod.sparse,
            mod.weight,
            qconfig=qconfig,
        )

        return qat_embedding_bag

    def to_float(self):
        embedding_bag = torch.nn.Embedding(
            self.num_embeddings,
            self.embedding_dim,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
            None,
        )
        embedding_bag.weight = torch.nn.Parameter(self.weight.detach())
        embedding_bag.train(self.training)
        return embedding_bag


class EmbeddingBag(nn.EmbeddingBag):
    r"""
    An embedding bag module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.EmbeddingBag`, please see
    https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag
    for documentation.

    Similar to `torch.nn.EmbeddingBag`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight: fake quant module for weight
    """

    _FLOAT_MODULE = nn.EmbeddingBag

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        mode="mean",
        sparse=False,
        _weight=None,
        include_last_offset=False,
        padding_idx=None,
        qconfig=None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
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
            **factory_kwargs,
        )
        assert qconfig, "qconfig must be provided for QAT module"
        assert qconfig.weight().qscheme == torch.per_channel_affine_float_qparams, (
            "Embedding Bag weights requires a qscheme of torch.per_channel_affine_float_qparams Got "
            + str(qconfig.weight().qscheme)
        )
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)

    def forward(self, input, offsets=None, per_sample_weights=None) -> Tensor:
        return F.embedding_bag(
            input,
            self.weight_fake_quant(self.weight),
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
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Create a qat module from a float module

        Args: `mod` a float module, either produced by torch.ao.quantization utilities
        or directly from user
        """
        assert type(mod) is cls._FLOAT_MODULE, (
            " qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"
        weight_qscheme = mod.qconfig.weight().qscheme  # type: ignore[union-attr, operator]
        assert weight_qscheme == torch.per_channel_affine_float_qparams, (
            "Embedding Bag weights requires a qscheme of torch.per_channel_affine_float_qparams Got "
            + str(weight_qscheme)
        )

        qconfig = mod.qconfig
        qat_embedding_bag = cls(
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
            qconfig=qconfig,
        )

        return qat_embedding_bag

    def to_float(self):
        embedding_bag = torch.nn.EmbeddingBag(
            self.num_embeddings,
            self.embedding_dim,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.mode,
            self.sparse,
            None,
            self.include_last_offset,
            self.padding_idx,
        )
        embedding_bag.weight = torch.nn.Parameter(self.weight.detach())
        embedding_bag.train(self.training)
        return embedding_bag

```



## High-Level Overview

r"""    An embedding bag module attached with FakeQuantize modules for weight,    used for quantization aware training.    We adopt the same interface as `torch.nn.Embedding`, please see    https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding    for documentation.    Similar to `torch.nn.Embedding`, with FakeQuantize modules initialized to    default.    Attributes:        weight: fake quant module for weight

This Python file contains 2 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Embedding`, `EmbeddingBag`

**Functions defined**: `__init__`, `forward`, `from_float`, `to_float`, `__init__`, `forward`, `from_float`, `to_float`

**Key imports**: torch, torch.nn as nn, torch.nn.functional as F, Tensor


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/nn/qat/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.nn as nn`
- `torch.nn.functional as F`


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

Files in the same folder (`torch/ao/nn/qat/modules`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`conv.py_docs.md`](./conv.py_docs.md)
- [`linear.py_docs.md`](./linear.py_docs.md)


## Cross-References

- **File Documentation**: `embedding_ops.py_docs.md`
- **Keyword Index**: `embedding_ops.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
