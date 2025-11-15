# Documentation: `docs/torch/ao/nn/intrinsic/quantized/modules/linear_relu.py_docs.md`

## File Metadata

- **Path**: `docs/torch/ao/nn/intrinsic/quantized/modules/linear_relu.py_docs.md`
- **Size**: 9,795 bytes (9.57 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/ao/nn/intrinsic/quantized/modules/linear_relu.py`

## File Metadata

- **Path**: `torch/ao/nn/intrinsic/quantized/modules/linear_relu.py`
- **Size**: 6,884 bytes (6.72 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.quantized as nnq
from torch.ao.nn.quantized.modules.utils import _quantize_weight


__all__ = [
    "LinearReLU",
    "LinearLeakyReLU",
    "LinearTanh",
]


class LinearReLU(nnq.Linear):
    r"""
    A LinearReLU module fused from Linear and ReLU modules

    We adopt the same interface as :class:`torch.ao.nn.quantized.Linear`.

    Attributes:
        Same as torch.ao.nn.quantized.Linear

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.intrinsic.LinearReLU(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    _FLOAT_MODULE = nni.LinearReLU  # type: ignore[assignment]

    def __init__(self, in_features, out_features, bias=True, dtype=torch.qint8):
        super().__init__(in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.quantized.linear_relu(
            x, self._packed_params._packed_params, self.scale, self.zero_point
        )

    def _get_name(self):
        return "QuantizedLinearReLU"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(mod, use_precomputed_fake_quant)

    @classmethod
    def from_reference(cls, ref_linear_relu, output_scale, output_zero_point):
        return super().from_reference(
            ref_linear_relu[0], output_scale, output_zero_point
        )


class LinearLeakyReLU(nnq.Linear):
    r"""
    For onednn backend only
    A LinearLeakyReLU module fused from Linear and LeakyReLU modules
    We adopt the same interface as :class:`torch.ao.nn.quantized.Linear`.
    Attributes:
        Same as torch.ao.nn.quantized.Linear
        + negative_slope
    Examples::
        >>> # xdoctest: +SKIP
        >>> m = nn.intrinsic.LinearLeakyReLU(20, 30, 0.01)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    _FLOAT_MODULE = nni.LinearLeakyReLU  # type: ignore[assignment]

    def __init__(
        self, in_features, out_features, negative_slope, bias=True, dtype=torch.qint8
    ):
        super().__init__(in_features, out_features, bias, dtype)
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.quantized.linear_leaky_relu(
            x,
            self._packed_params._packed_params,
            self.scale,
            self.zero_point,
            self.negative_slope,
        )

    def _get_name(self):
        return "QuantizedLinearLeakyReLU"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        assert type(mod) is nni.LinearLeakyReLU, (
            "Input float module should be LinearLeakyReLU"
        )
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"
        activation_post_process = mod.activation_post_process
        leaky_relu = mod[1]
        mod = mod[0]
        weight_post_process = mod.qconfig.weight()  # type: ignore[union-attr, operator]
        weight_post_process(mod.weight)
        dtype = weight_post_process.dtype
        act_scale, act_zp = activation_post_process.calculate_qparams()  # type: ignore[union-attr,operator]
        assert dtype == torch.qint8, "Weight observer must have dtype torch.qint8"
        qweight = _quantize_weight(mod.weight.float(), weight_post_process)
        qlinear_leaky_relu = cls(
            mod.in_features, mod.out_features, leaky_relu.negative_slope, dtype=dtype
        )
        qlinear_leaky_relu.set_weight_bias(qweight, mod.bias)  # type: ignore[arg-type]
        qlinear_leaky_relu.scale = float(act_scale)
        qlinear_leaky_relu.zero_point = int(act_zp)
        return qlinear_leaky_relu

    @classmethod
    def from_reference(cls, ref_mod, output_scale, output_zero_point):
        linear = ref_mod[0]
        leaky_relu = ref_mod[1]
        qlinear_leaky_relu = cls(
            linear.in_features, linear.out_features, leaky_relu.negative_slope
        )
        qweight = linear.get_quantized_weight()
        qlinear_leaky_relu.set_weight_bias(qweight, linear.bias)
        qlinear_leaky_relu.scale = float(output_scale)
        qlinear_leaky_relu.zero_point = int(output_zero_point)
        return qlinear_leaky_relu


class LinearTanh(nnq.Linear):
    r"""
    A LinearTanh module fused from Linear and Tanh modules

    We adopt the same interface as :class:`torch.ao.nn.quantized.Linear`.

    Attributes:
        Same as torch.ao.nn.quantized.Linear

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.intrinsic.LinearTanh(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    _FLOAT_MODULE = nni.LinearTanh  # type: ignore[assignment]

    def __init__(self, in_features, out_features, bias=True, dtype=torch.qint8):
        super().__init__(in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.quantized.linear_tanh(
            x, self._packed_params._packed_params, self.scale, self.zero_point
        )

    def _get_name(self):
        return "QuantizedLinearTanh"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        assert type(mod) is nni.LinearTanh, "Input float module should be LinearTanh"
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"
        activation_post_process = mod.activation_post_process
        mod = mod[0]
        weight_post_process = mod.qconfig.weight()  # type: ignore[union-attr,operator]
        weight_post_process(mod.weight)
        dtype = weight_post_process.dtype
        act_scale, act_zp = activation_post_process.calculate_qparams()  # type: ignore[union-attr,operator]
        assert dtype == torch.qint8, "Weight observer must have dtype torch.qint8"
        qweight = _quantize_weight(mod.weight.float(), weight_post_process)
        qlinear_tanh = cls(mod.in_features, mod.out_features, dtype=dtype)
        qlinear_tanh.set_weight_bias(qweight, mod.bias)  # type: ignore[arg-type]
        qlinear_tanh.scale = float(act_scale)
        qlinear_tanh.zero_point = int(act_zp)
        return qlinear_tanh

    @classmethod
    def from_reference(cls, ref_mod, output_scale, output_zero_point):
        linear = ref_mod[0]
        qlinear_tanh = cls(linear.in_features, linear.out_features)
        qweight = linear.get_quantized_weight()
        qlinear_tanh.set_weight_bias(qweight, linear.bias)
        qlinear_tanh.scale = float(output_scale)
        qlinear_tanh.zero_point = int(output_zero_point)
        return qlinear_tanh

```



## High-Level Overview

r"""    A LinearReLU module fused from Linear and ReLU modules    We adopt the same interface as :class:`torch.ao.nn.quantized.Linear`.    Attributes:        Same as torch.ao.nn.quantized.Linear    Examples::        >>> # xdoctest: +SKIP        >>> m = nn.intrinsic.LinearReLU(20, 30)        >>> input = torch.randn(128, 20)        >>> output = m(input)        >>> print(output.size())        torch.Size([128, 30])

This Python file contains 3 class(es) and 15 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LinearReLU`, `LinearLeakyReLU`, `LinearTanh`

**Functions defined**: `__init__`, `forward`, `_get_name`, `from_float`, `from_reference`, `__init__`, `forward`, `_get_name`, `from_float`, `from_reference`, `__init__`, `forward`, `_get_name`, `from_float`, `from_reference`

**Key imports**: torch, torch.ao.nn.intrinsic as nni, torch.ao.nn.quantized as nnq, _quantize_weight


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/nn/intrinsic/quantized/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.ao.nn.intrinsic as nni`
- `torch.ao.nn.quantized as nnq`
- `torch.ao.nn.quantized.modules.utils`: _quantize_weight


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

Files in the same folder (`torch/ao/nn/intrinsic/quantized/modules`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`conv_add.py_docs.md`](./conv_add.py_docs.md)
- [`bn_relu.py_docs.md`](./bn_relu.py_docs.md)
- [`conv_relu.py_docs.md`](./conv_relu.py_docs.md)


## Cross-References

- **File Documentation**: `linear_relu.py_docs.md`
- **Keyword Index**: `linear_relu.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/ao/nn/intrinsic/quantized/modules`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/ao/nn/intrinsic/quantized/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Files in the same folder (`docs/torch/ao/nn/intrinsic/quantized/modules`):

- [`linear_relu.py_kw.md_docs.md`](./linear_relu.py_kw.md_docs.md)
- [`conv_relu.py_docs.md_docs.md`](./conv_relu.py_docs.md_docs.md)
- [`conv_relu.py_kw.md_docs.md`](./conv_relu.py_kw.md_docs.md)
- [`conv_add.py_docs.md_docs.md`](./conv_add.py_docs.md_docs.md)
- [`bn_relu.py_kw.md_docs.md`](./bn_relu.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`bn_relu.py_docs.md_docs.md`](./bn_relu.py_docs.md_docs.md)
- [`conv_add.py_kw.md_docs.md`](./conv_add.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `linear_relu.py_docs.md_docs.md`
- **Keyword Index**: `linear_relu.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
