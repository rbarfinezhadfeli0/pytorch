# Documentation: `torch/nn/modules/pixelshuffle.py`

## File Metadata

- **Path**: `torch/nn/modules/pixelshuffle.py`
- **Size**: 3,948 bytes (3.86 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import torch.nn.functional as F
from torch import Tensor

from .module import Module


__all__ = ["PixelShuffle", "PixelUnshuffle"]


class PixelShuffle(Module):
    r"""Rearrange elements in a tensor according to an upscaling factor.

    Rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)`
    to a tensor of shape :math:`(*, C, H \times r, W \times r)`, where r is an upscale factor.

    This is useful for implementing efficient sub-pixel convolution
    with a stride of :math:`1/r`.

    See the paper:
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network`_
    by Shi et al. (2016) for more details.

    Args:
        upscale_factor (int): factor to increase spatial resolution by

    Shape:
        - Input: :math:`(*, C_{in}, H_{in}, W_{in})`, where * is zero or more batch dimensions
        - Output: :math:`(*, C_{out}, H_{out}, W_{out})`, where

    .. math::
        C_{out} = C_{in} \div \text{upscale\_factor}^2

    .. math::
        H_{out} = H_{in} \times \text{upscale\_factor}

    .. math::
        W_{out} = W_{in} \times \text{upscale\_factor}

    Examples::

        >>> pixel_shuffle = nn.PixelShuffle(3)
        >>> input = torch.randn(1, 9, 4, 4)
        >>> output = pixel_shuffle(input)
        >>> print(output.size())
        torch.Size([1, 1, 12, 12])

    .. _Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network:
        https://arxiv.org/abs/1609.05158
    """

    __constants__ = ["upscale_factor"]
    upscale_factor: int

    def __init__(self, upscale_factor: int) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.pixel_shuffle(input, self.upscale_factor)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"upscale_factor={self.upscale_factor}"


class PixelUnshuffle(Module):
    r"""Reverse the PixelShuffle operation.

    Reverses the :class:`~torch.nn.PixelShuffle` operation by rearranging elements
    in a tensor of shape :math:`(*, C, H \times r, W \times r)` to a tensor of shape
    :math:`(*, C \times r^2, H, W)`, where r is a downscale factor.

    See the paper:
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network`_
    by Shi et al. (2016) for more details.

    Args:
        downscale_factor (int): factor to decrease spatial resolution by

    Shape:
        - Input: :math:`(*, C_{in}, H_{in}, W_{in})`, where * is zero or more batch dimensions
        - Output: :math:`(*, C_{out}, H_{out}, W_{out})`, where

    .. math::
        C_{out} = C_{in} \times \text{downscale\_factor}^2

    .. math::
        H_{out} = H_{in} \div \text{downscale\_factor}

    .. math::
        W_{out} = W_{in} \div \text{downscale\_factor}

    Examples::

        >>> pixel_unshuffle = nn.PixelUnshuffle(3)
        >>> input = torch.randn(1, 1, 12, 12)
        >>> output = pixel_unshuffle(input)
        >>> print(output.size())
        torch.Size([1, 9, 4, 4])

    .. _Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network:
        https://arxiv.org/abs/1609.05158
    """

    __constants__ = ["downscale_factor"]
    downscale_factor: int

    def __init__(self, downscale_factor: int) -> None:
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.pixel_unshuffle(input, self.downscale_factor)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"downscale_factor={self.downscale_factor}"

```



## High-Level Overview

r"""Rearrange elements in a tensor according to an upscaling factor.    Rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)`    to a tensor of shape :math:`(*, C, H \times r, W \times r)`, where r is an upscale factor.    This is useful for implementing efficient sub-pixel convolution    with a stride of :math:`1/r`.    See the paper:    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network`_    by Shi et al. (2016) for more details.    Args:        upscale_factor (int): factor to increase spatial resolution by    Shape:        - Input: :math:`(*, C_{in}, H_{in}, W_{in})`, where * is zero or more batch dimensions        - Output: :math:`(*, C_{out}, H_{out}, W_{out})`, where    .. math::        C_{out} = C_{in} \div \text{upscale\_factor}^2    .. math::        H_{out} = H_{in} \times \text{upscale\_factor}    .. math::        W_{out} = W_{in} \times \text{upscale\_factor}    Examples::        >>> pixel_shuffle = nn.PixelShuffle(3)        >>> input = torch.randn(1, 9, 4, 4)        >>> output = pixel_shuffle(input)        >>> print(output.size())        torch.Size([1, 1, 12, 12])    .. _Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network:        https://arxiv.org/abs/1609.05158

This Python file contains 2 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `PixelShuffle`, `PixelUnshuffle`

**Functions defined**: `__init__`, `forward`, `extra_repr`, `__init__`, `forward`, `extra_repr`

**Key imports**: torch.nn.functional as F, Tensor, Module


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nn/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch.nn.functional as F`
- `torch`: Tensor
- `.module`: Module


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

Files in the same folder (`torch/nn/modules`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`fold.py_docs.md`](./fold.py_docs.md)
- [`rnn.py_docs.md`](./rnn.py_docs.md)
- [`channelshuffle.py_docs.md`](./channelshuffle.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`adaptive.py_docs.md`](./adaptive.py_docs.md)
- [`conv.py_docs.md`](./conv.py_docs.md)
- [`distance.py_docs.md`](./distance.py_docs.md)
- [`linear.py_docs.md`](./linear.py_docs.md)
- [`normalization.py_docs.md`](./normalization.py_docs.md)


## Cross-References

- **File Documentation**: `pixelshuffle.py_docs.md`
- **Keyword Index**: `pixelshuffle.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
