# Documentation: `docs/torch/ao/nn/intrinsic/quantized/modules/conv_add.py_docs.md`

## File Metadata

- **Path**: `docs/torch/ao/nn/intrinsic/quantized/modules/conv_add.py_docs.md`
- **Size**: 7,192 bytes (7.02 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/ao/nn/intrinsic/quantized/modules/conv_add.py`

## File Metadata

- **Path**: `torch/ao/nn/intrinsic/quantized/modules/conv_add.py`
- **Size**: 4,434 bytes (4.33 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import torch
import torch.ao.nn.intrinsic
import torch.ao.nn.intrinsic.qat
import torch.ao.nn.quantized as nnq
import torch.nn.functional as F


_reverse_repeat_padding = nnq.modules.conv._reverse_repeat_padding


class ConvAdd2d(nnq.Conv2d):
    r"""
    A ConvAdd2d module is a fused module of Conv2d and Add

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv2d`.

    Attributes:
        Same as torch.ao.nn.quantized.Conv2d

    """

    _FLOAT_MODULE = torch.ao.nn.intrinsic.ConvAdd2d  # type: ignore[assignment]

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, input, extra_input):  # type: ignore[override]
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        if self.padding_mode != "zeros":
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(
                input, _reversed_padding_repeated_twice, mode=self.padding_mode
            )
        return torch.ops.quantized.conv2d_add(
            input, extra_input, self._packed_params, self.scale, self.zero_point
        )

    def _get_name(self):
        return "QuantizedConvAdd2d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        return super().from_float(
            mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )

    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point):
        return super().from_reference(ref_qconv[0], output_scale, output_zero_point)


class ConvAddReLU2d(nnq.Conv2d):
    r"""
    A ConvAddReLU2d module is a fused module of Conv2d, Add and Relu

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv2d`.

    Attributes:
        Same as torch.ao.nn.quantized.Conv2d

    """

    _FLOAT_MODULE = torch.ao.nn.intrinsic.ConvAddReLU2d  # type: ignore[assignment]

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, input, extra_input):  # type: ignore[override]
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        if self.padding_mode != "zeros":
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(
                input, _reversed_padding_repeated_twice, mode=self.padding_mode
            )
        return torch.ops.quantized.conv2d_add_relu(
            input, extra_input, self._packed_params, self.scale, self.zero_point
        )

    def _get_name(self):
        return "QuantizedConvAddReLU2d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        return super().from_float(
            mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )

    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point):
        return super().from_reference(ref_qconv[0], output_scale, output_zero_point)

```



## High-Level Overview

r"""    A ConvAdd2d module is a fused module of Conv2d and Add    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv2d`.    Attributes:        Same as torch.ao.nn.quantized.Conv2d

This Python file contains 2 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ConvAdd2d`, `ConvAddReLU2d`

**Functions defined**: `__init__`, `forward`, `_get_name`, `from_float`, `from_reference`, `__init__`, `forward`, `_get_name`, `from_float`, `from_reference`

**Key imports**: torch, torch.ao.nn.intrinsic, torch.ao.nn.intrinsic.qat, torch.ao.nn.quantized as nnq, torch.nn.functional as F


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/nn/intrinsic/quantized/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.ao.nn.intrinsic`
- `torch.ao.nn.intrinsic.qat`
- `torch.ao.nn.quantized as nnq`
- `torch.nn.functional as F`


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

Files in the same folder (`torch/ao/nn/intrinsic/quantized/modules`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`bn_relu.py_docs.md`](./bn_relu.py_docs.md)
- [`linear_relu.py_docs.md`](./linear_relu.py_docs.md)
- [`conv_relu.py_docs.md`](./conv_relu.py_docs.md)


## Cross-References

- **File Documentation**: `conv_add.py_docs.md`
- **Keyword Index**: `conv_add.py_kw.md`
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

Files in the same folder (`docs/torch/ao/nn/intrinsic/quantized/modules`):

- [`linear_relu.py_kw.md_docs.md`](./linear_relu.py_kw.md_docs.md)
- [`conv_relu.py_docs.md_docs.md`](./conv_relu.py_docs.md_docs.md)
- [`conv_relu.py_kw.md_docs.md`](./conv_relu.py_kw.md_docs.md)
- [`bn_relu.py_kw.md_docs.md`](./bn_relu.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`bn_relu.py_docs.md_docs.md`](./bn_relu.py_docs.md_docs.md)
- [`conv_add.py_kw.md_docs.md`](./conv_add.py_kw.md_docs.md)
- [`linear_relu.py_docs.md_docs.md`](./linear_relu.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `conv_add.py_docs.md_docs.md`
- **Keyword Index**: `conv_add.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
