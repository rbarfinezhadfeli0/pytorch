# Documentation: `docs/torch/ao/nn/intrinsic/qat/modules/linear_fused.py_docs.md`

## File Metadata

- **Path**: `docs/torch/ao/nn/intrinsic/qat/modules/linear_fused.py_docs.md`
- **Size**: 9,643 bytes (9.42 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/ao/nn/intrinsic/qat/modules/linear_fused.py`

## File Metadata

- **Path**: `torch/ao/nn/intrinsic/qat/modules/linear_fused.py`
- **Size**: 6,613 bytes (6.46 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import torch
import torch.ao.nn.intrinsic as nni
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.utils.fusion import fuse_linear_bn_weights


__all__ = [
    "LinearBn1d",
]


class LinearBn1d(nn.modules.linear.Linear, nni._FusedModule):
    r"""
    A LinearBn1d module is a module fused from Linear and BatchNorm1d, attached
    with FakeQuantize modules for weight, used in quantization aware training.

    We combined the interface of :class:`torch.nn.Linear` and
    :class:torch.nn.BatchNorm1d`.

    Similar to :class:`torch.nn.Linear`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight

    """

    def __init__(
        self,
        # Linear args
        in_features,
        out_features,
        bias=True,
        # BatchNorm1d args
        # num_features: out_features
        eps=1e-05,
        momentum=0.1,
        # affine: True
        # track_running_stats: True
        # Args for this module
        freeze_bn=False,
        qconfig=None,
    ):
        nn.modules.linear.Linear.__init__(self, in_features, out_features, bias)
        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
        self.freeze_bn = freeze_bn if self.training else True
        self.bn = nn.BatchNorm1d(out_features, eps, momentum, True, True)
        self.weight_fake_quant = self.qconfig.weight()
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_bn_parameters()

        # this needs to be called after reset_bn_parameters,
        # as they modify the same state
        if self.training:
            if freeze_bn:
                self.freeze_bn_stats()
            else:
                self.update_bn_stats()
        else:
            self.freeze_bn_stats()

    def reset_running_stats(self):
        self.bn.reset_running_stats()

    def reset_bn_parameters(self):
        self.bn.reset_running_stats()
        init.uniform_(self.bn.weight)
        init.zeros_(self.bn.bias)

    def reset_parameters(self):
        super().reset_parameters()

    def update_bn_stats(self):
        self.freeze_bn = False
        self.bn.training = True
        return self

    def freeze_bn_stats(self):
        self.freeze_bn = True
        self.bn.training = False
        return self

    def forward(self, input):
        assert self.bn.running_var is not None

        # Scale the linear weights by BN's running statistics to reduce
        # weight jitter, see https://arxiv.org/pdf/1806.08342.pdf, page 18
        # for motivation.
        #
        # Instead of
        #
        #   x1 = F.linear(x0, fq(w), b)
        #   x2 = self.bn(x1)
        #
        # We have
        #
        #   # scale the weight by previous batch's running statistics
        #   scale_factor = bn.w / bn.running_std_from_prev_batch
        #   # do the linear transformation without bias
        #   x1_scaled = F.linear(x0, fq(w * scale_factor), 0)
        #   # reverse the scaling and add original bias
        #   x1_orig = x1_scaled / scale_factor + b
        #   x2 = self.bn(x1_orig)

        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1
        scaled_weight = self.weight_fake_quant(
            self.weight * scale_factor.reshape(weight_shape)
        )
        if self.bias is not None:
            zero_bias = torch.zeros_like(self.bias)
        else:
            zero_bias = torch.zeros(self.out_features, device=scaled_weight.device)
        linear_out = F.linear(input, scaled_weight, zero_bias)
        linear_out_orig = linear_out / scale_factor.reshape(bias_shape)
        if self.bias is not None:
            linear_out_orig = linear_out_orig + self.bias.reshape(bias_shape)
        bn_out = self.bn(linear_out_orig)
        return bn_out

    def train(self, mode=True):
        """
        Batchnorm's training behavior is using the self.training flag. Prevent
        changing it if BN is frozen. This makes sure that calling `model.train()`
        on a model with a frozen BN will behave properly.
        """
        self.training = mode
        if not self.freeze_bn:
            for module in self.children():
                module.train(mode)
        return self

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Create a qat module from a float module or qparams_dict

        Args: `mod' a float module, either produced by torch.ao.quantization
        utilities or directly from user
        """
        assert type(mod) is nni.LinearBn1d, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + nni.LinearBn1d.__name__
        )
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid config"
        qconfig = mod.qconfig
        linear, bn = mod[0], mod[1]
        qat_linearbn = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            bn.eps,
            bn.momentum,
            False,
            qconfig,
        )
        qat_linearbn.weight = linear.weight  # type: ignore[assignment]
        qat_linearbn.bias = linear.bias  # type: ignore[assignment]
        qat_linearbn.bn.weight = bn.weight  # type: ignore[assignment]
        qat_linearbn.bn.bias = bn.bias  # type: ignore[assignment]
        qat_linearbn.bn.running_mean = bn.running_mean  # type: ignore[assignment]
        qat_linearbn.bn.running_var = bn.running_var  # type: ignore[assignment]
        qat_linearbn.bn.num_batches_tracked = bn.num_batches_tracked  # type: ignore[assignment]
        return qat_linearbn

    def to_float(self):
        linear = torch.nn.Linear(self.in_features, self.out_features)
        assert self.bn.running_var is not None and self.bn.running_mean is not None
        linear.weight, linear.bias = fuse_linear_bn_weights(
            self.weight,
            self.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.eps,
            self.bn.weight,
            self.bn.bias,
        )
        return linear

```



## High-Level Overview

r"""    A LinearBn1d module is a module fused from Linear and BatchNorm1d, attached    with FakeQuantize modules for weight, used in quantization aware training.    We combined the interface of :class:`torch.nn.Linear` and    :class:torch.nn.BatchNorm1d`.    Similar to :class:`torch.nn.Linear`, with FakeQuantize modules initialized    to default.    Attributes:        freeze_bn:        weight_fake_quant: fake quant module for weight

This Python file contains 1 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LinearBn1d`

**Functions defined**: `__init__`, `reset_running_stats`, `reset_bn_parameters`, `reset_parameters`, `update_bn_stats`, `freeze_bn_stats`, `forward`, `train`, `from_float`, `to_float`

**Key imports**: torch, torch.ao.nn.intrinsic as nni, torch.nn as nn, torch.nn.functional as F, init, Parameter, fuse_linear_bn_weights


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/nn/intrinsic/qat/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.ao.nn.intrinsic as nni`
- `torch.nn as nn`
- `torch.nn.functional as F`
- `torch.nn`: init
- `torch.nn.parameter`: Parameter
- `torch.nn.utils.fusion`: fuse_linear_bn_weights


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`torch/ao/nn/intrinsic/qat/modules`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`linear_relu.py_docs.md`](./linear_relu.py_docs.md)
- [`conv_fused.py_docs.md`](./conv_fused.py_docs.md)


## Cross-References

- **File Documentation**: `linear_fused.py_docs.md`
- **Keyword Index**: `linear_fused.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/ao/nn/intrinsic/qat/modules`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/ao/nn/intrinsic/qat/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/torch/ao/nn/intrinsic/qat/modules`):

- [`linear_relu.py_kw.md_docs.md`](./linear_relu.py_kw.md_docs.md)
- [`conv_fused.py_kw.md_docs.md`](./conv_fused.py_kw.md_docs.md)
- [`conv_fused.py_docs.md_docs.md`](./conv_fused.py_docs.md_docs.md)
- [`linear_fused.py_kw.md_docs.md`](./linear_fused.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`linear_relu.py_docs.md_docs.md`](./linear_relu.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `linear_fused.py_docs.md_docs.md`
- **Keyword Index**: `linear_fused.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
