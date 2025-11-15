# Documentation: `torch/ao/nn/intrinsic/modules/fused.py`

## File Metadata

- **Path**: `torch/ao/nn/intrinsic/modules/fused.py`
- **Size**: 10,317 bytes (10.08 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import torch
from torch.nn import (
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    Conv1d,
    Conv2d,
    Conv3d,
    Linear,
    ReLU,
)
from torch.nn.utils.parametrize import type_before_parametrizations


__all__ = [
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
    "LinearReLU",
    "ConvBn1d",
    "ConvBn2d",
    "ConvBnReLU1d",
    "ConvBnReLU2d",
    "ConvBn3d",
    "ConvBnReLU3d",
    "BNReLU2d",
    "BNReLU3d",
    "LinearBn1d",
    "LinearLeakyReLU",
    "LinearTanh",
    "ConvAdd2d",
    "ConvAddReLU2d",
]


# Used for identifying intrinsic modules used in quantization
class _FusedModule(torch.nn.Sequential):
    pass


class ConvReLU1d(_FusedModule):
    r"""This is a sequential container which calls the Conv1d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, relu):
        assert (
            type_before_parametrizations(conv) == Conv1d
            and type_before_parametrizations(relu) == ReLU
        ), (
            f"Incorrect types for input modules{type_before_parametrizations(conv)}"
            f"{type_before_parametrizations(relu)}"
        )
        super().__init__(conv, relu)


class ConvReLU2d(_FusedModule):
    r"""This is a sequential container which calls the Conv2d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, relu):
        assert (
            type_before_parametrizations(conv) == Conv2d
            and type_before_parametrizations(relu) == ReLU
        ), (
            f"Incorrect types for input modules{type_before_parametrizations(conv)}"
            f"{type_before_parametrizations(relu)}"
        )
        super().__init__(conv, relu)


class ConvReLU3d(_FusedModule):
    r"""This is a sequential container which calls the Conv3d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, relu):
        assert (
            type_before_parametrizations(conv) == Conv3d
            and type_before_parametrizations(relu) == ReLU
        ), (
            f"Incorrect types for input modules{type_before_parametrizations(conv)}"
            f"{type_before_parametrizations(relu)}"
        )
        super().__init__(conv, relu)


class LinearReLU(_FusedModule):
    r"""This is a sequential container which calls the Linear and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, linear, relu):
        assert (
            type_before_parametrizations(linear) == Linear
            and type_before_parametrizations(relu) == ReLU
        ), (
            f"Incorrect types for input modules{type_before_parametrizations(linear)}"
            f"{type_before_parametrizations(relu)}"
        )
        super().__init__(linear, relu)


class ConvBn1d(_FusedModule):
    r"""This is a sequential container which calls the Conv 1d and Batch Norm 1d modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn):
        assert (
            type_before_parametrizations(conv) == Conv1d
            and type_before_parametrizations(bn) == BatchNorm1d
        ), (
            f"Incorrect types for input modules{type_before_parametrizations(conv)}"
            f"{type_before_parametrizations(bn)}"
        )
        super().__init__(conv, bn)


class ConvBn2d(_FusedModule):
    r"""This is a sequential container which calls the Conv 2d and Batch Norm 2d modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn):
        assert (
            type_before_parametrizations(conv) == Conv2d
            and type_before_parametrizations(bn) == BatchNorm2d
        ), (
            f"Incorrect types for input modules{type_before_parametrizations(conv)}"
            f"{type_before_parametrizations(bn)}"
        )
        super().__init__(conv, bn)


class ConvBnReLU1d(_FusedModule):
    r"""This is a sequential container which calls the Conv 1d, Batch Norm 1d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn, relu):
        assert (
            type_before_parametrizations(conv) == Conv1d
            and type_before_parametrizations(bn) == BatchNorm1d
            and type_before_parametrizations(relu) == ReLU
        ), (
            f"Incorrect types for input modules{type_before_parametrizations(conv)}"
            f"{type_before_parametrizations(bn)}"
            f"{type_before_parametrizations(relu)}"
        )
        super().__init__(conv, bn, relu)


class ConvBnReLU2d(_FusedModule):
    r"""This is a sequential container which calls the Conv 2d, Batch Norm 2d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn, relu):
        assert (
            type_before_parametrizations(conv) == Conv2d
            and type_before_parametrizations(bn) == BatchNorm2d
            and type_before_parametrizations(relu) == ReLU
        ), (
            f"Incorrect types for input modules{type_before_parametrizations(conv)}"
            f"{type_before_parametrizations(bn)}"
            f"{type_before_parametrizations(relu)}"
        )
        super().__init__(conv, bn, relu)


class ConvBn3d(_FusedModule):
    r"""This is a sequential container which calls the Conv 3d and Batch Norm 3d modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn):
        assert (
            type_before_parametrizations(conv) == Conv3d
            and type_before_parametrizations(bn) == BatchNorm3d
        ), (
            f"Incorrect types for input modules{type_before_parametrizations(conv)}"
            f"{type_before_parametrizations(bn)}"
        )
        super().__init__(conv, bn)


class ConvBnReLU3d(_FusedModule):
    r"""This is a sequential container which calls the Conv 3d, Batch Norm 3d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn, relu):
        assert (
            type_before_parametrizations(conv) == Conv3d
            and type_before_parametrizations(bn) == BatchNorm3d
            and type_before_parametrizations(relu) == ReLU
        ), (
            f"Incorrect types for input modules{type_before_parametrizations(conv)}"
            f"{type_before_parametrizations(bn)}"
            f"{type_before_parametrizations(relu)}"
        )
        super().__init__(conv, bn, relu)


class BNReLU2d(_FusedModule):
    r"""This is a sequential container which calls the BatchNorm 2d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, batch_norm, relu):
        assert (
            type_before_parametrizations(batch_norm) == BatchNorm2d
            and type_before_parametrizations(relu) == ReLU
        ), (
            f"Incorrect types for input modules{type_before_parametrizations(batch_norm)}"
            f"{type_before_parametrizations(relu)}"
        )
        super().__init__(batch_norm, relu)


class BNReLU3d(_FusedModule):
    r"""This is a sequential container which calls the BatchNorm 3d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, batch_norm, relu):
        assert (
            type_before_parametrizations(batch_norm) == BatchNorm3d
            and type_before_parametrizations(relu) == ReLU
        ), (
            f"Incorrect types for input modules{type_before_parametrizations(batch_norm)}"
            f"{type_before_parametrizations(relu)}"
        )
        super().__init__(batch_norm, relu)


class LinearBn1d(_FusedModule):
    r"""This is a sequential container which calls the Linear and BatchNorm1d modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, linear, bn):
        assert (
            type_before_parametrizations(linear) == Linear
            and type_before_parametrizations(bn) == BatchNorm1d
        ), (
            f"Incorrect types for input modules{type_before_parametrizations(linear)}"
            f"{type_before_parametrizations(bn)}"
        )
        super().__init__(linear, bn)


class LinearLeakyReLU(_FusedModule):
    r"""This is a sequential container which calls the Linear and LeakyReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, linear, leaky_relu):
        assert type(linear) is Linear and type(leaky_relu) is torch.nn.LeakyReLU, (
            f"Incorrect types for input modules{type(linear)}{type(leaky_relu)}"
        )
        super().__init__(linear, leaky_relu)


class LinearTanh(_FusedModule):
    r"""This is a sequential container which calls the Linear and Tanh modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, linear, tanh):
        assert type(linear) is Linear and type(tanh) is torch.nn.Tanh, (
            f"Incorrect types for input modules{type(linear)}{type(tanh)}"
        )
        super().__init__(linear, tanh)


class ConvAdd2d(_FusedModule):
    r"""This is a sequential container which calls the Conv2d modules with extra Add.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, add):
        super().__init__(conv)
        self.add = add

    def forward(self, x1, x2):  # type: ignore[override]
        return self.add(self[0](x1), x2)


class ConvAddReLU2d(_FusedModule):
    r"""This is a sequential container which calls the Conv2d, add, Relu.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, add, relu):
        super().__init__(conv)
        self.add = add
        self.relu = relu

    def forward(self, x1, x2):  # type: ignore[override]
        return self.relu(self.add(self[0](x1), x2))

```



## High-Level Overview

r"""This is a sequential container which calls the Conv1d and ReLU modules.

This Python file contains 18 class(es) and 19 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_FusedModule`, `ConvReLU1d`, `ConvReLU2d`, `ConvReLU3d`, `LinearReLU`, `ConvBn1d`, `ConvBn2d`, `ConvBnReLU1d`, `ConvBnReLU2d`, `ConvBn3d`, `ConvBnReLU3d`, `BNReLU2d`, `BNReLU3d`, `LinearBn1d`, `LinearLeakyReLU`, `LinearTanh`, `ConvAdd2d`, `ConvAddReLU2d`

**Functions defined**: `__init__`, `__init__`, `__init__`, `__init__`, `__init__`, `__init__`, `__init__`, `__init__`, `__init__`, `__init__`, `__init__`, `__init__`, `__init__`, `__init__`, `__init__`, `__init__`, `forward`, `__init__`, `forward`

**Key imports**: torch, type_before_parametrizations


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/ao/nn/intrinsic/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.nn.utils.parametrize`: type_before_parametrizations


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

Files in the same folder (`torch/ao/nn/intrinsic/modules`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)


## Cross-References

- **File Documentation**: `fused.py_docs.md`
- **Keyword Index**: `fused.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
