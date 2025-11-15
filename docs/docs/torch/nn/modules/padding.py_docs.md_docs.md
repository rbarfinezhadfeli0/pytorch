# Documentation: `docs/torch/nn/modules/padding.py_docs.md`

## File Metadata

- **Path**: `docs/torch/nn/modules/padding.py_docs.md`
- **Size**: 34,923 bytes (34.10 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nn/modules/padding.py`

## File Metadata

- **Path**: `torch/nn/modules/padding.py`
- **Size**: 31,660 bytes (30.92 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from collections.abc import Sequence

import torch.nn.functional as F
from torch import Tensor
from torch.nn.common_types import _size_2_t, _size_4_t, _size_6_t

from .module import Module
from .utils import _ntuple, _pair, _quadruple


# TODO: grad_output size asserts in THNN

__all__ = [
    "CircularPad1d",
    "CircularPad2d",
    "CircularPad3d",
    "ConstantPad1d",
    "ConstantPad2d",
    "ConstantPad3d",
    "ReflectionPad1d",
    "ReflectionPad2d",
    "ReflectionPad3d",
    "ReplicationPad1d",
    "ReplicationPad2d",
    "ReplicationPad3d",
    "ZeroPad1d",
    "ZeroPad2d",
    "ZeroPad3d",
]


class _CircularPadNd(Module):
    __constants__ = ["padding"]
    padding: Sequence[int]

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)
        return F.pad(input, self.padding, "circular")

    def extra_repr(self) -> str:
        return f"{self.padding}"


class CircularPad1d(_CircularPadNd):
    r"""Pads the input tensor using circular padding of the input boundary.

    Tensor values at the beginning of the dimension are used to pad the end,
    and values at the end are used to pad the beginning. If negative padding is
    applied then the ends of the tensor get removed.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 2-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`)
            Note that padding size should be less than or equal to the corresponding input dimension.

    Shape:
        - Input: :math:`(C, W_{in})` or :math:`(N, C, W_{in})`.
        - Output: :math:`(C, W_{out})` or :math:`(N, C, W_{out})`, where

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("not sure why xdoctest is choking on this")
        >>> m = nn.CircularPad1d(2)
        >>> input = torch.arange(8, dtype=torch.float).reshape(1, 2, 4)
        >>> input
        tensor([[[0., 1., 2., 3.],
                 [4., 5., 6., 7.]]])
        >>> m(input)
        tensor([[[2., 3., 0., 1., 2., 3., 0., 1.],
                 [6., 7., 4., 5., 6., 7., 4., 5.]]])
        >>> # using different paddings for different sides
        >>> m = nn.CircularPad1d((3, 1))
        >>> m(input)
        tensor([[[1., 2., 3., 0., 1., 2., 3., 0.],
                 [5., 6., 7., 4., 5., 6., 7., 4.]]])
    """

    # pyrefly: ignore [bad-override]
    padding: tuple[int, int]

    def __init__(self, padding: _size_2_t) -> None:
        super().__init__()
        self.padding = _pair(padding)

    def _check_input_dim(self, input) -> None:
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(f"expected 2D or 3D input (got {input.dim()}D input)")


class CircularPad2d(_CircularPadNd):
    r"""Pads the input tensor using circular padding of the input boundary.

    Tensor values at the beginning of the dimension are used to pad the end,
    and values at the end are used to pad the beginning. If negative padding is
    applied then the ends of the tensor get removed.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 4-`tuple`, uses (:math:`\text{padding\_left}`,
            :math:`\text{padding\_right}`, :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`)
            Note that padding size should be less than or equal to the corresponding input dimension.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> m = nn.CircularPad2d(2)
        >>> input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
        >>> input
        tensor([[[[0., 1., 2.],
                  [3., 4., 5.],
                  [6., 7., 8.]]]])
        >>> m(input)
        tensor([[[[4., 5., 3., 4., 5., 3., 4.],
                  [7., 8., 6., 7., 8., 6., 7.],
                  [1., 2., 0., 1., 2., 0., 1.],
                  [4., 5., 3., 4., 5., 3., 4.],
                  [7., 8., 6., 7., 8., 6., 7.],
                  [1., 2., 0., 1., 2., 0., 1.],
                  [4., 5., 3., 4., 5., 3., 4.]]]])
        >>> # using different paddings for different sides
        >>> m = nn.CircularPad2d((1, 1, 2, 0))
        >>> m(input)
        tensor([[[[5., 3., 4., 5., 3.],
                  [8., 6., 7., 8., 6.],
                  [2., 0., 1., 2., 0.],
                  [5., 3., 4., 5., 3.],
                  [8., 6., 7., 8., 6.]]]])
    """

    # pyrefly: ignore [bad-override]
    padding: tuple[int, int, int, int]

    def __init__(self, padding: _size_4_t) -> None:
        super().__init__()
        self.padding = _quadruple(padding)

    def _check_input_dim(self, input) -> None:
        if input.dim() != 3 and input.dim() != 4:
            raise ValueError(f"expected 3D or 4D input (got {input.dim()}D input)")


class CircularPad3d(_CircularPadNd):
    r"""Pads the input tensor using circular padding of the input boundary.

    Tensor values at the beginning of the dimension are used to pad the end,
    and values at the end are used to pad the beginning. If negative padding is
    applied then the ends of the tensor get removed.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 6-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`,
            :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`,
            :math:`\text{padding\_front}`, :math:`\text{padding\_back}`)
            Note that padding size should be less than or equal to the corresponding input dimension.

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or :math:`(C, D_{out}, H_{out}, W_{out})`,
          where

          :math:`D_{out} = D_{in} + \text{padding\_front} + \text{padding\_back}`

          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = nn.CircularPad3d(3)
        >>> input = torch.randn(16, 3, 8, 320, 480)
        >>> output = m(input)
        >>> # using different paddings for different sides
        >>> m = nn.CircularPad3d((3, 3, 6, 6, 1, 1))
        >>> output = m(input)
    """

    # pyrefly: ignore [bad-override]
    padding: tuple[int, int, int, int, int, int]

    def __init__(self, padding: _size_6_t) -> None:
        super().__init__()
        self.padding = _ntuple(6)(padding)

    def _check_input_dim(self, input) -> None:
        if input.dim() != 4 and input.dim() != 5:
            raise ValueError(f"expected 4D or 5D input (got {input.dim()}D input)")


class _ConstantPadNd(Module):
    __constants__ = ["padding", "value"]
    value: float
    padding: Sequence[int]

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, input: Tensor) -> Tensor:
        return F.pad(input, self.padding, "constant", self.value)

    def extra_repr(self) -> str:
        return f"padding={self.padding}, value={self.value}"


class ConstantPad1d(_ConstantPadNd):
    r"""Pads the input tensor boundaries with a constant value.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in both boundaries. If a 2-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`)

    Shape:
        - Input: :math:`(C, W_{in})` or :math:`(N, C, W_{in})`.
        - Output: :math:`(C, W_{out})` or :math:`(N, C, W_{out})`, where

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = nn.ConstantPad1d(2, 3.5)
        >>> input = torch.randn(1, 2, 4)
        >>> input
        tensor([[[-1.0491, -0.7152, -0.0749,  0.8530],
                 [-1.3287,  1.8966,  0.1466, -0.2771]]])
        >>> m(input)
        tensor([[[ 3.5000,  3.5000, -1.0491, -0.7152, -0.0749,  0.8530,  3.5000,
                   3.5000],
                 [ 3.5000,  3.5000, -1.3287,  1.8966,  0.1466, -0.2771,  3.5000,
                   3.5000]]])
        >>> m = nn.ConstantPad1d(2, 3.5)
        >>> input = torch.randn(1, 2, 3)
        >>> input
        tensor([[[ 1.6616,  1.4523, -1.1255],
                 [-3.6372,  0.1182, -1.8652]]])
        >>> m(input)
        tensor([[[ 3.5000,  3.5000,  1.6616,  1.4523, -1.1255,  3.5000,  3.5000],
                 [ 3.5000,  3.5000, -3.6372,  0.1182, -1.8652,  3.5000,  3.5000]]])
        >>> # using different paddings for different sides
        >>> m = nn.ConstantPad1d((3, 1), 3.5)
        >>> m(input)
        tensor([[[ 3.5000,  3.5000,  3.5000,  1.6616,  1.4523, -1.1255,  3.5000],
                 [ 3.5000,  3.5000,  3.5000, -3.6372,  0.1182, -1.8652,  3.5000]]])
    """

    # pyrefly: ignore [bad-override]
    padding: tuple[int, int]

    def __init__(self, padding: _size_2_t, value: float) -> None:
        super().__init__(value)
        self.padding = _pair(padding)


class ConstantPad2d(_ConstantPadNd):
    r"""Pads the input tensor boundaries with a constant value.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 4-`tuple`, uses (:math:`\text{padding\_left}`,
            :math:`\text{padding\_right}`, :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`)

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = nn.ConstantPad2d(2, 3.5)
        >>> input = torch.randn(1, 2, 2)
        >>> input
        tensor([[[ 1.6585,  0.4320],
                 [-0.8701, -0.4649]]])
        >>> m(input)
        tensor([[[ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
                 [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
                 [ 3.5000,  3.5000,  1.6585,  0.4320,  3.5000,  3.5000],
                 [ 3.5000,  3.5000, -0.8701, -0.4649,  3.5000,  3.5000],
                 [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
                 [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000,  3.5000]]])
        >>> # using different paddings for different sides
        >>> m = nn.ConstantPad2d((3, 0, 2, 1), 3.5)
        >>> m(input)
        tensor([[[ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
                 [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000],
                 [ 3.5000,  3.5000,  3.5000,  1.6585,  0.4320],
                 [ 3.5000,  3.5000,  3.5000, -0.8701, -0.4649],
                 [ 3.5000,  3.5000,  3.5000,  3.5000,  3.5000]]])
    """

    __constants__ = ["padding", "value"]
    # pyrefly: ignore [bad-override]
    padding: tuple[int, int, int, int]

    def __init__(self, padding: _size_4_t, value: float) -> None:
        super().__init__(value)
        self.padding = _quadruple(padding)


class ConstantPad3d(_ConstantPadNd):
    r"""Pads the input tensor boundaries with a constant value.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 6-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`,
            :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`,
            :math:`\text{padding\_front}`, :math:`\text{padding\_back}`)

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or
          :math:`(C, D_{out}, H_{out}, W_{out})`, where

          :math:`D_{out} = D_{in} + \text{padding\_front} + \text{padding\_back}`

          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> m = nn.ConstantPad3d(3, 3.5)
        >>> input = torch.randn(16, 3, 10, 20, 30)
        >>> output = m(input)
        >>> # using different paddings for different sides
        >>> m = nn.ConstantPad3d((3, 3, 6, 6, 0, 1), 3.5)
        >>> output = m(input)
    """

    # pyrefly: ignore [bad-override]
    padding: tuple[int, int, int, int, int, int]

    def __init__(self, padding: _size_6_t, value: float) -> None:
        super().__init__(value)
        self.padding = _ntuple(6)(padding)


class _ReflectionPadNd(Module):
    __constants__ = ["padding"]
    padding: Sequence[int]

    def forward(self, input: Tensor) -> Tensor:
        return F.pad(input, self.padding, "reflect")

    def extra_repr(self) -> str:
        return f"{self.padding}"


class ReflectionPad1d(_ReflectionPadNd):
    r"""Pads the input tensor using the reflection of the input boundary.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 2-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`)
            Note that padding size should be less than the corresponding input dimension.

    Shape:
        - Input: :math:`(C, W_{in})` or :math:`(N, C, W_{in})`.
        - Output: :math:`(C, W_{out})` or :math:`(N, C, W_{out})`, where

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> m = nn.ReflectionPad1d(2)
        >>> # xdoctest: +IGNORE_WANT("other tests seem to modify printing styles")
        >>> input = torch.arange(8, dtype=torch.float).reshape(1, 2, 4)
        >>> input
        tensor([[[0., 1., 2., 3.],
                 [4., 5., 6., 7.]]])
        >>> m(input)
        tensor([[[2., 1., 0., 1., 2., 3., 2., 1.],
                 [6., 5., 4., 5., 6., 7., 6., 5.]]])
        >>> # using different paddings for different sides
        >>> m = nn.ReflectionPad1d((3, 1))
        >>> m(input)
        tensor([[[3., 2., 1., 0., 1., 2., 3., 2.],
                 [7., 6., 5., 4., 5., 6., 7., 6.]]])
    """

    # pyrefly: ignore [bad-override]
    padding: tuple[int, int]

    def __init__(self, padding: _size_2_t) -> None:
        super().__init__()
        self.padding = _pair(padding)


class ReflectionPad2d(_ReflectionPadNd):
    r"""Pads the input tensor using the reflection of the input boundary.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 4-`tuple`, uses (:math:`\text{padding\_left}`,
            :math:`\text{padding\_right}`, :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`)
            Note that padding size should be less than the corresponding input dimension.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})` where

          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("not sure why xdoctest is choking on this")
        >>> m = nn.ReflectionPad2d(2)
        >>> input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
        >>> input
        tensor([[[[0., 1., 2.],
                  [3., 4., 5.],
                  [6., 7., 8.]]]])
        >>> m(input)
        tensor([[[[8., 7., 6., 7., 8., 7., 6.],
                  [5., 4., 3., 4., 5., 4., 3.],
                  [2., 1., 0., 1., 2., 1., 0.],
                  [5., 4., 3., 4., 5., 4., 3.],
                  [8., 7., 6., 7., 8., 7., 6.],
                  [5., 4., 3., 4., 5., 4., 3.],
                  [2., 1., 0., 1., 2., 1., 0.]]]])
        >>> # using different paddings for different sides
        >>> m = nn.ReflectionPad2d((1, 1, 2, 0))
        >>> m(input)
        tensor([[[[7., 6., 7., 8., 7.],
                  [4., 3., 4., 5., 4.],
                  [1., 0., 1., 2., 1.],
                  [4., 3., 4., 5., 4.],
                  [7., 6., 7., 8., 7.]]]])
    """

    # pyrefly: ignore [bad-override]
    padding: tuple[int, int, int, int]

    def __init__(self, padding: _size_4_t) -> None:
        super().__init__()
        self.padding = _quadruple(padding)


class ReflectionPad3d(_ReflectionPadNd):
    r"""Pads the input tensor using the reflection of the input boundary.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 6-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`,
            :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`,
            :math:`\text{padding\_front}`, :math:`\text{padding\_back}`)
            Note that padding size should be less than the corresponding input dimension.

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or :math:`(C, D_{out}, H_{out}, W_{out})`,
          where

          :math:`D_{out} = D_{in} + \text{padding\_front} + \text{padding\_back}`

          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("not sure why xdoctest is choking on this")
        >>> m = nn.ReflectionPad3d(1)
        >>> input = torch.arange(8, dtype=torch.float).reshape(1, 1, 2, 2, 2)
        >>> m(input)
        tensor([[[[[7., 6., 7., 6.],
                   [5., 4., 5., 4.],
                   [7., 6., 7., 6.],
                   [5., 4., 5., 4.]],
                  [[3., 2., 3., 2.],
                   [1., 0., 1., 0.],
                   [3., 2., 3., 2.],
                   [1., 0., 1., 0.]],
                  [[7., 6., 7., 6.],
                   [5., 4., 5., 4.],
                   [7., 6., 7., 6.],
                   [5., 4., 5., 4.]],
                  [[3., 2., 3., 2.],
                   [1., 0., 1., 0.],
                   [3., 2., 3., 2.],
                   [1., 0., 1., 0.]]]]])
    """

    # pyrefly: ignore [bad-override]
    padding: tuple[int, int, int, int, int, int]

    def __init__(self, padding: _size_6_t) -> None:
        super().__init__()
        self.padding = _ntuple(6)(padding)


class _ReplicationPadNd(Module):
    __constants__ = ["padding"]
    padding: Sequence[int]

    def forward(self, input: Tensor) -> Tensor:
        return F.pad(input, self.padding, "replicate")

    def extra_repr(self) -> str:
        return f"{self.padding}"


class ReplicationPad1d(_ReplicationPadNd):
    r"""Pads the input tensor using replication of the input boundary.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 2-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`)
            Note that the output dimensions must remain positive.

    Shape:
        - Input: :math:`(C, W_{in})` or :math:`(N, C, W_{in})`.
        - Output: :math:`(C, W_{out})` or :math:`(N, C, W_{out})`, where

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("not sure why xdoctest is choking on this")
        >>> m = nn.ReplicationPad1d(2)
        >>> input = torch.arange(8, dtype=torch.float).reshape(1, 2, 4)
        >>> input
        tensor([[[0., 1., 2., 3.],
                 [4., 5., 6., 7.]]])
        >>> m(input)
        tensor([[[0., 0., 0., 1., 2., 3., 3., 3.],
                 [4., 4., 4., 5., 6., 7., 7., 7.]]])
        >>> # using different paddings for different sides
        >>> m = nn.ReplicationPad1d((3, 1))
        >>> m(input)
        tensor([[[0., 0., 0., 0., 1., 2., 3., 3.],
                 [4., 4., 4., 4., 5., 6., 7., 7.]]])
    """

    # pyrefly: ignore [bad-override]
    padding: tuple[int, int]

    def __init__(self, padding: _size_2_t) -> None:
        super().__init__()
        self.padding = _pair(padding)


class ReplicationPad2d(_ReplicationPadNd):
    r"""Pads the input tensor using replication of the input boundary.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 4-`tuple`, uses (:math:`\text{padding\_left}`,
            :math:`\text{padding\_right}`, :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`)
            Note that the output dimensions must remain positive.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> m = nn.ReplicationPad2d(2)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
        >>> input
        tensor([[[[0., 1., 2.],
                  [3., 4., 5.],
                  [6., 7., 8.]]]])
        >>> m(input)
        tensor([[[[0., 0., 0., 1., 2., 2., 2.],
                  [0., 0., 0., 1., 2., 2., 2.],
                  [0., 0., 0., 1., 2., 2., 2.],
                  [3., 3., 3., 4., 5., 5., 5.],
                  [6., 6., 6., 7., 8., 8., 8.],
                  [6., 6., 6., 7., 8., 8., 8.],
                  [6., 6., 6., 7., 8., 8., 8.]]]])
        >>> # using different paddings for different sides
        >>> m = nn.ReplicationPad2d((1, 1, 2, 0))
        >>> m(input)
        tensor([[[[0., 0., 1., 2., 2.],
                  [0., 0., 1., 2., 2.],
                  [0., 0., 1., 2., 2.],
                  [3., 3., 4., 5., 5.],
                  [6., 6., 7., 8., 8.]]]])
    """

    # pyrefly: ignore [bad-override]
    padding: tuple[int, int, int, int]

    def __init__(self, padding: _size_4_t) -> None:
        super().__init__()
        self.padding = _quadruple(padding)


class ReplicationPad3d(_ReplicationPadNd):
    r"""Pads the input tensor using replication of the input boundary.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 6-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`,
            :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`,
            :math:`\text{padding\_front}`, :math:`\text{padding\_back}`)
            Note that the output dimensions must remain positive.

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or :math:`(C, D_{out}, H_{out}, W_{out})`,
          where

          :math:`D_{out} = D_{in} + \text{padding\_front} + \text{padding\_back}`

          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = nn.ReplicationPad3d(3)
        >>> input = torch.randn(16, 3, 8, 320, 480)
        >>> output = m(input)
        >>> # using different paddings for different sides
        >>> m = nn.ReplicationPad3d((3, 3, 6, 6, 1, 1))
        >>> output = m(input)
    """

    # pyrefly: ignore [bad-override]
    padding: tuple[int, int, int, int, int, int]

    def __init__(self, padding: _size_6_t) -> None:
        super().__init__()
        self.padding = _ntuple(6)(padding)


class ZeroPad1d(ConstantPad1d):
    r"""Pads the input tensor boundaries with zero.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in both boundaries. If a 2-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`)

    Shape:
        - Input: :math:`(C, W_{in})` or :math:`(N, C, W_{in})`.
        - Output: :math:`(C, W_{out})` or :math:`(N, C, W_{out})`, where

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = nn.ZeroPad1d(2)
        >>> input = torch.randn(1, 2, 4)
        >>> input
        tensor([[[-1.0491, -0.7152, -0.0749,  0.8530],
                 [-1.3287,  1.8966,  0.1466, -0.2771]]])
        >>> m(input)
        tensor([[[ 0.0000,  0.0000, -1.0491, -0.7152, -0.0749,  0.8530,  0.0000,
                   0.0000],
                 [ 0.0000,  0.0000, -1.3287,  1.8966,  0.1466, -0.2771,  0.0000,
                   0.0000]]])
        >>> m = nn.ZeroPad1d(2)
        >>> input = torch.randn(1, 2, 3)
        >>> input
        tensor([[[ 1.6616,  1.4523, -1.1255],
                 [-3.6372,  0.1182, -1.8652]]])
        >>> m(input)
        tensor([[[ 0.0000,  0.0000,  1.6616,  1.4523, -1.1255,  0.0000,  0.0000],
                 [ 0.0000,  0.0000, -3.6372,  0.1182, -1.8652,  0.0000,  0.0000]]])
        >>> # using different paddings for different sides
        >>> m = nn.ZeroPad1d((3, 1))
        >>> m(input)
        tensor([[[ 0.0000,  0.0000,  0.0000,  1.6616,  1.4523, -1.1255,  0.0000],
                 [ 0.0000,  0.0000,  0.0000, -3.6372,  0.1182, -1.8652,  0.0000]]])
    """

    padding: tuple[int, int]

    def __init__(self, padding: _size_2_t) -> None:
        super().__init__(padding, 0.0)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"{self.padding}"


class ZeroPad2d(ConstantPad2d):
    r"""Pads the input tensor boundaries with zero.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 4-`tuple`, uses (:math:`\text{padding\_left}`,
            :math:`\text{padding\_right}`, :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`)

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = nn.ZeroPad2d(2)
        >>> input = torch.randn(1, 1, 3, 3)
        >>> input
        tensor([[[[-0.1678, -0.4418,  1.9466],
                  [ 0.9604, -0.4219, -0.5241],
                  [-0.9162, -0.5436, -0.6446]]]])
        >>> m(input)
        tensor([[[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                  [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                  [ 0.0000,  0.0000, -0.1678, -0.4418,  1.9466,  0.0000,  0.0000],
                  [ 0.0000,  0.0000,  0.9604, -0.4219, -0.5241,  0.0000,  0.0000],
                  [ 0.0000,  0.0000, -0.9162, -0.5436, -0.6446,  0.0000,  0.0000],
                  [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                  [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]]])
        >>> # using different paddings for different sides
        >>> m = nn.ZeroPad2d((1, 1, 2, 0))
        >>> m(input)
        tensor([[[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                  [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                  [ 0.0000, -0.1678, -0.4418,  1.9466,  0.0000],
                  [ 0.0000,  0.9604, -0.4219, -0.5241,  0.0000],
                  [ 0.0000, -0.9162, -0.5436, -0.6446,  0.0000]]]])
    """

    padding: tuple[int, int, int, int]

    def __init__(self, padding: _size_4_t) -> None:
        super().__init__(padding, 0.0)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"{self.padding}"


class ZeroPad3d(ConstantPad3d):
    r"""Pads the input tensor boundaries with zero.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 6-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`,
            :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`,
            :math:`\text{padding\_front}`, :math:`\text{padding\_back}`)

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or
          :math:`(C, D_{out}, H_{out}, W_{out})`, where

          :math:`D_{out} = D_{in} + \text{padding\_front} + \text{padding\_back}`

          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> m = nn.ZeroPad3d(3)
        >>> input = torch.randn(16, 3, 10, 20, 30)
        >>> output = m(input)
        >>> # using different paddings for different sides
        >>> m = nn.ZeroPad3d((3, 3, 6, 6, 0, 1))
        >>> output = m(input)
    """

    padding: tuple[int, int, int, int, int, int]

    def __init__(self, padding: _size_6_t) -> None:
        super().__init__(padding, 0.0)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"{self.padding}"

```



## High-Level Overview

r"""Pads the input tensor using circular padding of the input boundary.

This Python file contains 19 class(es) and 31 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_CircularPadNd`, `CircularPad1d`, `CircularPad2d`, `CircularPad3d`, `_ConstantPadNd`, `ConstantPad1d`, `ConstantPad2d`, `ConstantPad3d`, `_ReflectionPadNd`, `ReflectionPad1d`, `ReflectionPad2d`, `ReflectionPad3d`, `_ReplicationPadNd`, `ReplicationPad1d`, `ReplicationPad2d`, `ReplicationPad3d`, `ZeroPad1d`, `ZeroPad2d`, `ZeroPad3d`

**Functions defined**: `_check_input_dim`, `forward`, `extra_repr`, `__init__`, `_check_input_dim`, `__init__`, `_check_input_dim`, `__init__`, `_check_input_dim`, `__init__`, `forward`, `extra_repr`, `__init__`, `__init__`, `__init__`, `forward`, `extra_repr`, `__init__`, `__init__`, `__init__`

**Key imports**: Sequence, torch.nn.functional as F, Tensor, _size_2_t, _size_4_t, _size_6_t, Module, _ntuple, _pair, _quadruple


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nn/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Sequence
- `torch.nn.functional as F`
- `torch`: Tensor
- `torch.nn.common_types`: _size_2_t, _size_4_t, _size_6_t
- `.module`: Module
- `.utils`: _ntuple, _pair, _quadruple


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

- **File Documentation**: `padding.py_docs.md`
- **Keyword Index**: `padding.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/nn/modules`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/nn/modules`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/nn/modules`):

- [`sparse.py_docs.md_docs.md`](./sparse.py_docs.md_docs.md)
- [`instancenorm.py_kw.md_docs.md`](./instancenorm.py_kw.md_docs.md)
- [`activation.py_kw.md_docs.md`](./activation.py_kw.md_docs.md)
- [`container.py_docs.md_docs.md`](./container.py_docs.md_docs.md)
- [`distance.py_kw.md_docs.md`](./distance.py_kw.md_docs.md)
- [`pixelshuffle.py_kw.md_docs.md`](./pixelshuffle.py_kw.md_docs.md)
- [`module.py_docs.md_docs.md`](./module.py_docs.md_docs.md)
- [`batchnorm.py_docs.md_docs.md`](./batchnorm.py_docs.md_docs.md)
- [`transformer.py_kw.md_docs.md`](./transformer.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `padding.py_docs.md_docs.md`
- **Keyword Index**: `padding.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
