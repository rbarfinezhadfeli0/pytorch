# Documentation: `torch/ao/nn/quantized/modules/batchnorm.py`

## File Metadata

- **Path**: `torch/ao/nn/quantized/modules/batchnorm.py`
- **Size**: 4,519 bytes (4.41 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import torch
import torch.ao.nn.intrinsic as nni


__all__ = ["BatchNorm2d", "BatchNorm3d"]


class _BatchNorm(torch.nn.modules.batchnorm._BatchNorm):
    def __init__(
        self, num_features, eps=1e-5, momentum=0.1, device=None, dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(num_features, eps, momentum, True, True, **factory_kwargs)
        # pyrefly: ignore [bad-argument-type]
        self.register_buffer("scale", torch.tensor(1.0, **factory_kwargs))
        # pyrefly: ignore [bad-argument-type]
        self.register_buffer("zero_point", torch.tensor(0, **factory_kwargs))

    @staticmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        activation_post_process = mod.activation_post_process
        if type(mod) is cls._NNI_BN_RELU_MODULE:
            mod = mod[0]
        scale, zero_point = activation_post_process.calculate_qparams()
        new_mod = cls(mod.num_features, mod.eps)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        new_mod.running_mean = mod.running_mean
        new_mod.running_var = mod.running_var
        new_mod.scale = scale
        new_mod.zero_point = zero_point
        return new_mod

    @classmethod
    def from_reference(cls, bn, output_scale, output_zero_point):
        qbn = cls(
            bn.num_features,
            bn.eps,
            bn.momentum,
            device=bn.weight.device,
            dtype=bn.weight.dtype,
        )
        qbn.weight = bn.weight
        qbn.bias = bn.bias
        qbn.running_mean = bn.running_mean
        qbn.running_var = bn.running_var
        qbn.scale = output_scale
        qbn.zero_point = output_zero_point
        return qbn


class BatchNorm2d(_BatchNorm):
    r"""This is the quantized version of :class:`~torch.nn.BatchNorm2d`."""

    _NNI_BN_RELU_MODULE = nni.BNReLU2d

    def __init__(
        self, num_features, eps=1e-5, momentum=0.1, device=None, dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(num_features, eps, momentum, **factory_kwargs)

    def _get_name(self):
        return "QuantizedBatchNorm2d"

    def _check_input_dim(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # disabling this since this is not symbolically traceable
        # self._check_input_dim(input)
        return torch.ops.quantized.batch_norm2d(
            input,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.eps,
            self.scale,
            self.zero_point,
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        return _BatchNorm.from_float(
            cls, mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )


class BatchNorm3d(_BatchNorm):
    r"""This is the quantized version of :class:`~torch.nn.BatchNorm3d`."""

    _NNI_BN_RELU_MODULE = nni.BNReLU3d

    def __init__(self, num_features, eps=1e-5, momentum=0.1, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(num_features, eps, momentum, **factory_kwargs)

    def _get_name(self):
        return "QuantizedBatchNorm3d"

    def _check_input_dim(self, input):
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, H, W)`!")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # disabling this since this is not symbolically traceable
        # self._check_input_dim(input)
        return torch.ops.quantized.batch_norm3d(
            input,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.eps,
            self.scale,
            self.zero_point,
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        return _BatchNorm.from_float(
            cls, mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )

```



## High-Level Overview


This Python file contains 3 class(es) and 13 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_BatchNorm`, `BatchNorm2d`, `BatchNorm3d`

**Functions defined**: `__init__`, `from_float`, `from_reference`, `__init__`, `_get_name`, `_check_input_dim`, `forward`, `from_float`, `__init__`, `_get_name`, `_check_input_dim`, `forward`, `from_float`

**Key imports**: torch, torch.ao.nn.intrinsic as nni


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/nn/quantized/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.ao.nn.intrinsic as nni`


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

Files in the same folder (`torch/ao/nn/quantized/modules`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
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

- **File Documentation**: `batchnorm.py_docs.md`
- **Keyword Index**: `batchnorm.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
