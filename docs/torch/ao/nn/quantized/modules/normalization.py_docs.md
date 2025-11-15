# Documentation: `torch/ao/nn/quantized/modules/normalization.py`

## File Metadata

- **Path**: `torch/ao/nn/quantized/modules/normalization.py`
- **Size**: 10,049 bytes (9.81 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import torch


__all__ = [
    "LayerNorm",
    "GroupNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
]


class LayerNorm(torch.nn.LayerNorm):
    r"""This is the quantized version of :class:`~torch.nn.LayerNorm`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """

    def __init__(
        self,
        normalized_shape,
        weight,
        bias,
        scale,
        zero_point,
        eps=1e-5,
        elementwise_affine=True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            # pyrefly: ignore [bad-argument-type]
            **factory_kwargs,
        )
        self.weight = weight
        self.bias = bias
        # pyrefly: ignore [bad-argument-type]
        self.register_buffer("scale", torch.tensor(scale, **factory_kwargs))
        # pyrefly: ignore [bad-argument-type]
        self.register_buffer("zero_point", torch.tensor(zero_point, **factory_kwargs))

    def forward(self, input):
        return torch.ops.quantized.layer_norm(
            input,
            self.normalized_shape,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
            output_scale=self.scale,
            output_zero_point=self.zero_point,
        )

    def _get_name(self):
        return "QuantizedLayerNorm"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = cls(
            mod.normalized_shape,
            mod.weight,
            mod.bias,
            float(scale),
            int(zero_point),
            mod.eps,
            mod.elementwise_affine,
        )
        return new_mod

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(
            mod.normalized_shape,
            mod.weight,
            mod.bias,
            float(scale),
            int(zero_point),
            mod.eps,
            mod.elementwise_affine,
        )


class GroupNorm(torch.nn.GroupNorm):
    r"""This is the quantized version of :class:`~torch.nn.GroupNorm`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """

    __constants__ = ["num_groups", "num_channels", "eps", "affine"]

    def __init__(
        self,
        num_groups,
        num_channels,
        weight,
        bias,
        scale,
        zero_point,
        eps=1e-5,
        affine=True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(num_groups, num_channels, eps, affine, **factory_kwargs)
        self.weight = weight
        self.bias = bias
        # pyrefly: ignore [bad-argument-type]
        self.register_buffer("scale", torch.tensor(scale, **factory_kwargs))
        # pyrefly: ignore [bad-argument-type]
        self.register_buffer("zero_point", torch.tensor(zero_point, **factory_kwargs))

    def forward(self, input):
        return torch.ops.quantized.group_norm(
            input,
            self.num_groups,
            self.weight,
            self.bias,
            self.eps,
            self.scale,
            self.zero_point,
        )

    def _get_name(self):
        return "QuantizedGroupNorm"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = cls(
            mod.num_groups,
            mod.num_channels,
            mod.weight,
            mod.bias,
            float(scale),
            int(zero_point),
            mod.eps,
            mod.affine,
        )
        return new_mod


class InstanceNorm1d(torch.nn.InstanceNorm1d):
    r"""This is the quantized version of :class:`~torch.nn.InstanceNorm1d`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """

    def __init__(
        self,
        num_features,
        weight,
        bias,
        scale,
        zero_point,
        eps=1e-5,
        momentum=0.1,
        affine=False,
        track_running_stats=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        self.weight = weight
        self.bias = bias
        # pyrefly: ignore [bad-argument-type]
        self.register_buffer("scale", torch.tensor(scale, **factory_kwargs))
        # pyrefly: ignore [bad-argument-type]
        self.register_buffer("zero_point", torch.tensor(zero_point, **factory_kwargs))

    def forward(self, input):
        return torch.ops.quantized.instance_norm(
            input, self.weight, self.bias, self.eps, self.scale, self.zero_point
        )

    def _get_name(self):
        return "QuantizedInstanceNorm1d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = cls(
            mod.num_features,
            mod.weight,
            mod.bias,
            float(scale),
            int(zero_point),
            mod.eps,
            mod.affine,
        )
        return new_mod

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(
            mod.num_features,
            mod.weight,
            mod.bias,
            float(scale),
            int(zero_point),
            mod.eps,
            mod.affine,
        )


class InstanceNorm2d(torch.nn.InstanceNorm2d):
    r"""This is the quantized version of :class:`~torch.nn.InstanceNorm2d`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """

    def __init__(
        self,
        num_features,
        weight,
        bias,
        scale,
        zero_point,
        eps=1e-5,
        momentum=0.1,
        affine=False,
        track_running_stats=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        self.weight = weight
        self.bias = bias
        # pyrefly: ignore [bad-argument-type]
        self.register_buffer("scale", torch.tensor(scale, **factory_kwargs))
        # pyrefly: ignore [bad-argument-type]
        self.register_buffer("zero_point", torch.tensor(zero_point, **factory_kwargs))

    def forward(self, input):
        return torch.ops.quantized.instance_norm(
            input, self.weight, self.bias, self.eps, self.scale, self.zero_point
        )

    def _get_name(self):
        return "QuantizedInstanceNorm2d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = cls(
            mod.num_features,
            mod.weight,
            mod.bias,
            float(scale),
            int(zero_point),
            mod.eps,
            mod.affine,
        )
        return new_mod

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(
            mod.num_features,
            mod.weight,
            mod.bias,
            float(scale),
            int(zero_point),
            mod.eps,
            mod.affine,
        )


class InstanceNorm3d(torch.nn.InstanceNorm3d):
    r"""This is the quantized version of :class:`~torch.nn.InstanceNorm3d`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """

    def __init__(
        self,
        num_features,
        weight,
        bias,
        scale,
        zero_point,
        eps=1e-5,
        momentum=0.1,
        affine=False,
        track_running_stats=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        self.weight = weight
        self.bias = bias
        # pyrefly: ignore [bad-argument-type]
        self.register_buffer("scale", torch.tensor(scale, **factory_kwargs))
        # pyrefly: ignore [bad-argument-type]
        self.register_buffer("zero_point", torch.tensor(zero_point, **factory_kwargs))

    def forward(self, input):
        return torch.ops.quantized.instance_norm(
            input, self.weight, self.bias, self.eps, self.scale, self.zero_point
        )

    def _get_name(self):
        return "QuantizedInstanceNorm3d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = cls(
            mod.num_features,
            mod.weight,
            mod.bias,
            float(scale),
            int(zero_point),
            mod.eps,
            mod.affine,
        )
        return new_mod

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(
            mod.num_features,
            mod.weight,
            mod.bias,
            float(scale),
            int(zero_point),
            mod.eps,
            mod.affine,
        )

```



## High-Level Overview

r"""This is the quantized version of :class:`~torch.nn.LayerNorm`.    Additional args:        * **scale** - quantization scale of the output, type: double.        * **zero_point** - quantization zero point of the output, type: long.

This Python file contains 5 class(es) and 24 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LayerNorm`, `GroupNorm`, `InstanceNorm1d`, `InstanceNorm2d`, `InstanceNorm3d`

**Functions defined**: `__init__`, `forward`, `_get_name`, `from_float`, `from_reference`, `__init__`, `forward`, `_get_name`, `from_float`, `__init__`, `forward`, `_get_name`, `from_float`, `from_reference`, `__init__`, `forward`, `_get_name`, `from_float`, `from_reference`, `__init__`

**Key imports**: torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/nn/quantized/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`


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

Files in the same folder (`torch/ao/nn/quantized/modules`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`rnn.py_docs.md`](./rnn.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`embedding_ops.py_docs.md`](./embedding_ops.py_docs.md)
- [`conv.py_docs.md`](./conv.py_docs.md)
- [`linear.py_docs.md`](./linear.py_docs.md)
- [`functional_modules.py_docs.md`](./functional_modules.py_docs.md)
- [`activation.py_docs.md`](./activation.py_docs.md)
- [`dropout.py_docs.md`](./dropout.py_docs.md)


## Cross-References

- **File Documentation**: `normalization.py_docs.md`
- **Keyword Index**: `normalization.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
