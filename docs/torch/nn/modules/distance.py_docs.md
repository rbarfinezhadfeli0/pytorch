# Documentation: `torch/nn/modules/distance.py`

## File Metadata

- **Path**: `torch/nn/modules/distance.py`
- **Size**: 3,372 bytes (3.29 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import torch.nn.functional as F
from torch import Tensor

from .module import Module


__all__ = ["PairwiseDistance", "CosineSimilarity"]


class PairwiseDistance(Module):
    r"""
    Computes the pairwise distance between input vectors, or between columns of input matrices.

    Distances are computed using ``p``-norm, with constant ``eps`` added to avoid division by zero
    if ``p`` is negative, i.e.:

    .. math ::
        \mathrm{dist}\left(x, y\right) = \left\Vert x-y + \epsilon e \right\Vert_p,

    where :math:`e` is the vector of ones and the ``p``-norm is given by.

    .. math ::
        \Vert x \Vert _p = \left( \sum_{i=1}^n  \vert x_i \vert ^ p \right) ^ {1/p}.

    Args:
        p (real, optional): the norm degree. Can be negative. Default: 2
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-6
        keepdim (bool, optional): Determines whether or not to keep the vector dimension.
            Default: False
    Shape:
        - Input1: :math:`(N, D)` or :math:`(D)` where `N = batch dimension` and `D = vector dimension`
        - Input2: :math:`(N, D)` or :math:`(D)`, same shape as the Input1
        - Output: :math:`(N)` or :math:`()` based on input dimension.
          If :attr:`keepdim` is ``True``, then :math:`(N, 1)` or :math:`(1)` based on input dimension.

    Examples:
        >>> pdist = nn.PairwiseDistance(p=2)
        >>> input1 = torch.randn(100, 128)
        >>> input2 = torch.randn(100, 128)
        >>> output = pdist(input1, input2)
    """

    __constants__ = ["norm", "eps", "keepdim"]
    norm: float
    eps: float
    keepdim: bool

    def __init__(
        self, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False
    ) -> None:
        super().__init__()
        self.norm = p
        self.eps = eps
        self.keepdim = keepdim

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.pairwise_distance(x1, x2, self.norm, self.eps, self.keepdim)


class CosineSimilarity(Module):
    r"""Returns cosine similarity between :math:`x_1` and :math:`x_2`, computed along `dim`.

    .. math ::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}.

    Args:
        dim (int, optional): Dimension where cosine similarity is computed. Default: 1
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-8
    Shape:
        - Input1: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`
        - Input2: :math:`(\ast_1, D, \ast_2)`, same number of dimensions as x1, matching x1 size at dimension `dim`,
          and broadcastable with x1 at other dimensions.
        - Output: :math:`(\ast_1, \ast_2)`

    Examples:
        >>> input1 = torch.randn(100, 128)
        >>> input2 = torch.randn(100, 128)
        >>> cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        >>> output = cos(input1, input2)
    """

    __constants__ = ["dim", "eps"]
    dim: int
    eps: float

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.cosine_similarity(x1, x2, self.dim, self.eps)

```



## High-Level Overview

r"""    Computes the pairwise distance between input vectors, or between columns of input matrices.    Distances are computed using ``p``-norm, with constant ``eps`` added to avoid division by zero    if ``p`` is negative, i.e.:    .. math ::        \mathrm{dist}\left(x, y\right) = \left\Vert x-y + \epsilon e \right\Vert_p,    where :math:`e` is the vector of ones and the ``p``-norm is given by.    .. math ::        \Vert x \Vert _p = \left( \sum_{i=1}^n  \vert x_i \vert ^ p \right) ^ {1/p}.    Args:        p (real, optional): the norm degree. Can be negative. Default: 2        eps (float, optional): Small value to avoid division by zero.            Default: 1e-6        keepdim (bool, optional): Determines whether or not to keep the vector dimension.            Default: False    Shape:        - Input1: :math:`(N, D)` or :math:`(D)` where `N = batch dimension` and `D = vector dimension`        - Input2: :math:`(N, D)` or :math:`(D)`, same shape as the Input1        - Output: :math:`(N)` or :math:`()` based on input dimension.          If :attr:`keepdim` is ``True``, then :math:`(N, 1)` or :math:`(1)` based on input dimension.    Examples:        >>> pdist = nn.PairwiseDistance(p=2)        >>> input1 = torch.randn(100, 128)        >>> input2 = torch.randn(100, 128)        >>> output = pdist(input1, input2)

This Python file contains 2 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `PairwiseDistance`, `CosineSimilarity`

**Functions defined**: `__init__`, `forward`, `__init__`, `forward`

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
- [`linear.py_docs.md`](./linear.py_docs.md)
- [`normalization.py_docs.md`](./normalization.py_docs.md)


## Cross-References

- **File Documentation**: `distance.py_docs.md`
- **Keyword Index**: `distance.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
