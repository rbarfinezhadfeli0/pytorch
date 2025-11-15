# Documentation: `torch/_tensor_docs.py`

## File Metadata

- **Path**: `torch/_tensor_docs.py`
- **Size**: 145,257 bytes (141.85 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
"""Adds docstrings to Tensor functions"""

import torch._C
from torch._C import _add_docstr as add_docstr
from torch._torch_docs import parse_kwargs, reproducibility_notes


def add_docstr_all(method: str, docstr: str) -> None:
    add_docstr(getattr(torch._C.TensorBase, method), docstr)


common_args = parse_kwargs(
    """
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.preserve_format``.
"""
)

new_common_args = parse_kwargs(
    """
    size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
        shape of the output tensor.
    dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
        Default: if None, same :class:`torch.dtype` as this tensor.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if None, same :class:`torch.device` as this tensor.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    pin_memory (bool, optional): If set, returned tensor would be allocated in
        the pinned memory. Works only for CPU tensors. Default: ``False``.
    layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
        Default: ``torch.strided``.
"""
)

add_docstr_all(
    "new_tensor",
    """
new_tensor(data, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, \
pin_memory=False) -> Tensor
"""
    + r"""

Returns a new Tensor with :attr:`data` as the tensor data.
By default, the returned Tensor has the same :class:`torch.dtype` and
:class:`torch.device` as this tensor.

.. warning::

    :func:`new_tensor` always copies :attr:`data`. If you have a Tensor
    ``data`` and want to avoid a copy, use :func:`torch.Tensor.requires_grad_`
    or :func:`torch.Tensor.detach`.
    If you have a numpy array and want to avoid a copy, use
    :func:`torch.from_numpy`.

.. warning::

    When data is a tensor `x`, :func:`new_tensor()` reads out 'the data' from whatever it is passed,
    and constructs a leaf variable. Therefore ``tensor.new_tensor(x)`` is equivalent to ``x.detach().clone()``
    and ``tensor.new_tensor(x, requires_grad=True)`` is equivalent to ``x.detach().clone().requires_grad_(True)``.
    The equivalents using ``detach()`` and ``clone()`` are recommended.

Args:
    data (array_like): The returned Tensor copies :attr:`data`.

Keyword args:
    {dtype}
    {device}
    {requires_grad}
    {layout}
    {pin_memory}

Example::

    >>> tensor = torch.ones((2,), dtype=torch.int8)
    >>> data = [[0, 1], [2, 3]]
    >>> tensor.new_tensor(data)
    tensor([[ 0,  1],
            [ 2,  3]], dtype=torch.int8)

""".format(**new_common_args),
)

add_docstr_all(
    "new_full",
    """
new_full(size, fill_value, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, \
pin_memory=False) -> Tensor
"""
    + r"""

Returns a Tensor of size :attr:`size` filled with :attr:`fill_value`.
By default, the returned Tensor has the same :class:`torch.dtype` and
:class:`torch.device` as this tensor.

Args:
    fill_value (scalar): the number to fill the output tensor with.

Keyword args:
    {dtype}
    {device}
    {requires_grad}
    {layout}
    {pin_memory}

Example::

    >>> tensor = torch.ones((2,), dtype=torch.float64)
    >>> tensor.new_full((3, 4), 3.141592)
    tensor([[ 3.1416,  3.1416,  3.1416,  3.1416],
            [ 3.1416,  3.1416,  3.1416,  3.1416],
            [ 3.1416,  3.1416,  3.1416,  3.1416]], dtype=torch.float64)

""".format(**new_common_args),
)

add_docstr_all(
    "new_empty",
    """
new_empty(size, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, \
pin_memory=False) -> Tensor
"""
    + r"""

Returns a Tensor of size :attr:`size` filled with uninitialized data.
By default, the returned Tensor has the same :class:`torch.dtype` and
:class:`torch.device` as this tensor.

Args:
    size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
        shape of the output tensor.

Keyword args:
    {dtype}
    {device}
    {requires_grad}
    {layout}
    {pin_memory}

Example::

    >>> tensor = torch.ones(())
    >>> tensor.new_empty((2, 3))
    tensor([[ 5.8182e-18,  4.5765e-41, -1.0545e+30],
            [ 3.0949e-41,  4.4842e-44,  0.0000e+00]])

""".format(**new_common_args),
)

add_docstr_all(
    "new_empty_strided",
    """
new_empty_strided(size, stride, dtype=None, device=None, requires_grad=False, layout=torch.strided, \
pin_memory=False) -> Tensor
"""
    + r"""

Returns a Tensor of size :attr:`size` and strides :attr:`stride` filled with
uninitialized data. By default, the returned Tensor has the same
:class:`torch.dtype` and :class:`torch.device` as this tensor.

Args:
    size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
        shape of the output tensor.

Keyword args:
    {dtype}
    {device}
    {requires_grad}
    {layout}
    {pin_memory}

Example::

    >>> tensor = torch.ones(())
    >>> tensor.new_empty_strided((2, 3), (3, 1))
    tensor([[ 5.8182e-18,  4.5765e-41, -1.0545e+30],
            [ 3.0949e-41,  4.4842e-44,  0.0000e+00]])

""".format(**new_common_args),
)

add_docstr_all(
    "new_ones",
    """
new_ones(size, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, \
pin_memory=False) -> Tensor
"""
    + r"""

Returns a Tensor of size :attr:`size` filled with ``1``.
By default, the returned Tensor has the same :class:`torch.dtype` and
:class:`torch.device` as this tensor.

Args:
    size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
        shape of the output tensor.

Keyword args:
    {dtype}
    {device}
    {requires_grad}
    {layout}
    {pin_memory}

Example::

    >>> tensor = torch.tensor((), dtype=torch.int32)
    >>> tensor.new_ones((2, 3))
    tensor([[ 1,  1,  1],
            [ 1,  1,  1]], dtype=torch.int32)

""".format(**new_common_args),
)

add_docstr_all(
    "new_zeros",
    """
new_zeros(size, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, \
pin_memory=False) -> Tensor
"""
    + r"""

Returns a Tensor of size :attr:`size` filled with ``0``.
By default, the returned Tensor has the same :class:`torch.dtype` and
:class:`torch.device` as this tensor.

Args:
    size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
        shape of the output tensor.

Keyword args:
    {dtype}
    {device}
    {requires_grad}
    {layout}
    {pin_memory}

Example::

    >>> tensor = torch.tensor((), dtype=torch.float64)
    >>> tensor.new_zeros((2, 3))
    tensor([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.]], dtype=torch.float64)

""".format(**new_common_args),
)

add_docstr_all(
    "abs",
    r"""
abs() -> Tensor

See :func:`torch.abs`
""",
)

add_docstr_all(
    "abs_",
    r"""
abs_() -> Tensor

In-place version of :meth:`~Tensor.abs`
""",
)

add_docstr_all(
    "absolute",
    r"""
absolute() -> Tensor

Alias for :func:`abs`
""",
)

add_docstr_all(
    "absolute_",
    r"""
absolute_() -> Tensor

In-place version of :meth:`~Tensor.absolute`
Alias for :func:`abs_`
""",
)

add_docstr_all(
    "acos",
    r"""
acos() -> Tensor

See :func:`torch.acos`
""",
)

add_docstr_all(
    "acos_",
    r"""
acos_() -> Tensor

In-place version of :meth:`~Tensor.acos`
""",
)

add_docstr_all(
    "arccos",
    r"""
arccos() -> Tensor

See :func:`torch.arccos`
""",
)

add_docstr_all(
    "arccos_",
    r"""
arccos_() -> Tensor

In-place version of :meth:`~Tensor.arccos`
""",
)

add_docstr_all(
    "acosh",
    r"""
acosh() -> Tensor

See :func:`torch.acosh`
""",
)

add_docstr_all(
    "acosh_",
    r"""
acosh_() -> Tensor

In-place version of :meth:`~Tensor.acosh`
""",
)

add_docstr_all(
    "arccosh",
    r"""
acosh() -> Tensor

See :func:`torch.arccosh`
""",
)

add_docstr_all(
    "arccosh_",
    r"""
acosh_() -> Tensor

In-place version of :meth:`~Tensor.arccosh`
""",
)

add_docstr_all(
    "add",
    r"""
add(other, *, alpha=1) -> Tensor

Add a scalar or tensor to :attr:`self` tensor. If both :attr:`alpha`
and :attr:`other` are specified, each element of :attr:`other` is scaled by
:attr:`alpha` before being used.

When :attr:`other` is a tensor, the shape of :attr:`other` must be
:ref:`broadcastable <broadcasting-semantics>` with the shape of the underlying
tensor

See :func:`torch.add`
""",
)

add_docstr_all(
    "add_",
    r"""
add_(other, *, alpha=1) -> Tensor

In-place version of :meth:`~Tensor.add`
""",
)

add_docstr_all(
    "addbmm",
    r"""
addbmm(batch1, batch2, *, beta=1, alpha=1) -> Tensor

See :func:`torch.addbmm`
""",
)

add_docstr_all(
    "addbmm_",
    r"""
addbmm_(batch1, batch2, *, beta=1, alpha=1) -> Tensor

In-place version of :meth:`~Tensor.addbmm`
""",
)

add_docstr_all(
    "addcdiv",
    r"""
addcdiv(tensor1, tensor2, *, value=1) -> Tensor

See :func:`torch.addcdiv`
""",
)

add_docstr_all(
    "addcdiv_",
    r"""
addcdiv_(tensor1, tensor2, *, value=1) -> Tensor

In-place version of :meth:`~Tensor.addcdiv`
""",
)

add_docstr_all(
    "addcmul",
    r"""
addcmul(tensor1, tensor2, *, value=1) -> Tensor

See :func:`torch.addcmul`
""",
)

add_docstr_all(
    "addcmul_",
    r"""
addcmul_(tensor1, tensor2, *, value=1) -> Tensor

In-place version of :meth:`~Tensor.addcmul`
""",
)

add_docstr_all(
    "addmm",
    r"""
addmm(mat1, mat2, *, beta=1, alpha=1) -> Tensor

See :func:`torch.addmm`
""",
)

add_docstr_all(
    "addmm_",
    r"""
addmm_(mat1, mat2, *, beta=1, alpha=1) -> Tensor

In-place version of :meth:`~Tensor.addmm`
""",
)

add_docstr_all(
    "addmv",
    r"""
addmv(mat, vec, *, beta=1, alpha=1) -> Tensor

See :func:`torch.addmv`
""",
)

add_docstr_all(
    "addmv_",
    r"""
addmv_(mat, vec, *, beta=1, alpha=1) -> Tensor

In-place version of :meth:`~Tensor.addmv`
""",
)

add_docstr_all(
    "sspaddmm",
    r"""
sspaddmm(mat1, mat2, *, beta=1, alpha=1) -> Tensor

See :func:`torch.sspaddmm`
""",
)

add_docstr_all(
    "smm",
    r"""
smm(mat) -> Tensor

See :func:`torch.smm`
""",
)

add_docstr_all(
    "addr",
    r"""
addr(vec1, vec2, *, beta=1, alpha=1) -> Tensor

See :func:`torch.addr`
""",
)

add_docstr_all(
    "addr_",
    r"""
addr_(vec1, vec2, *, beta=1, alpha=1) -> Tensor

In-place version of :meth:`~Tensor.addr`
""",
)

add_docstr_all(
    "align_as",
    r"""
align_as(other) -> Tensor

Permutes the dimensions of the :attr:`self` tensor to match the dimension order
in the :attr:`other` tensor, adding size-one dims for any new names.

This operation is useful for explicit broadcasting by names (see examples).

All of the dims of :attr:`self` must be named in order to use this method.
The resulting tensor is a view on the original tensor.

All dimension names of :attr:`self` must be present in ``other.names``.
:attr:`other` may contain named dimensions that are not in ``self.names``;
the output tensor has a size-one dimension for each of those new names.

To align a tensor to a specific order, use :meth:`~Tensor.align_to`.

Examples::

    # Example 1: Applying a mask
    >>> mask = torch.randint(2, [127, 128], dtype=torch.bool).refine_names('W', 'H')
    >>> imgs = torch.randn(32, 128, 127, 3, names=('N', 'H', 'W', 'C'))
    >>> imgs.masked_fill_(mask.align_as(imgs), 0)


    # Example 2: Applying a per-channel-scale
    >>> def scale_channels(input, scale):
    >>>    scale = scale.refine_names('C')
    >>>    return input * scale.align_as(input)

    >>> num_channels = 3
    >>> scale = torch.randn(num_channels, names=('C',))
    >>> imgs = torch.rand(32, 128, 128, num_channels, names=('N', 'H', 'W', 'C'))
    >>> more_imgs = torch.rand(32, num_channels, 128, 128, names=('N', 'C', 'H', 'W'))
    >>> videos = torch.randn(3, num_channels, 128, 128, 128, names=('N', 'C', 'H', 'W', 'D'))

    # scale_channels is agnostic to the dimension order of the input
    >>> scale_channels(imgs, scale)
    >>> scale_channels(more_imgs, scale)
    >>> scale_channels(videos, scale)

.. warning::
    The named tensor API is experimental and subject to change.

""",
)

add_docstr_all(
    "all",
    r"""
all(dim=None, keepdim=False) -> Tensor

See :func:`torch.all`
""",
)

add_docstr_all(
    "allclose",
    r"""
allclose(other, rtol=1e-05, atol=1e-08, equal_nan=False) -> Tensor

See :func:`torch.allclose`
""",
)

add_docstr_all(
    "angle",
    r"""
angle() -> Tensor

See :func:`torch.angle`
""",
)

add_docstr_all(
    "any",
    r"""
any(dim=None, keepdim=False) -> Tensor

See :func:`torch.any`
""",
)

add_docstr_all(
    "apply_",
    r"""
apply_(callable) -> Tensor

Applies the function :attr:`callable` to each element in the tensor, replacing
each element with the value returned by :attr:`callable`.

.. note::

    This function only works with CPU tensors and should not be used in code
    sections that require high performance.
""",
)

add_docstr_all(
    "asin",
    r"""
asin() -> Tensor

See :func:`torch.asin`
""",
)

add_docstr_all(
    "asin_",
    r"""
asin_() -> Tensor

In-place version of :meth:`~Tensor.asin`
""",
)

add_docstr_all(
    "arcsin",
    r"""
arcsin() -> Tensor

See :func:`torch.arcsin`
""",
)

add_docstr_all(
    "arcsin_",
    r"""
arcsin_() -> Tensor

In-place version of :meth:`~Tensor.arcsin`
""",
)

add_docstr_all(
    "asinh",
    r"""
asinh() -> Tensor

See :func:`torch.asinh`
""",
)

add_docstr_all(
    "asinh_",
    r"""
asinh_() -> Tensor

In-place version of :meth:`~Tensor.asinh`
""",
)

add_docstr_all(
    "arcsinh",
    r"""
arcsinh() -> Tensor

See :func:`torch.arcsinh`
""",
)

add_docstr_all(
    "arcsinh_",
    r"""
arcsinh_() -> Tensor

In-place version of :meth:`~Tensor.arcsinh`
""",
)

add_docstr_all(
    "as_strided",
    r"""
as_strided(size, stride, storage_offset=None) -> Tensor

See :func:`torch.as_strided`
""",
)

add_docstr_all(
    "as_strided_",
    r"""
as_strided_(size, stride, storage_offset=None) -> Tensor

In-place version of :meth:`~Tensor.as_strided`
""",
)

add_docstr_all(
    "atan",
    r"""
atan() -> Tensor

See :func:`torch.atan`
""",
)

add_docstr_all(
    "atan_",
    r"""
atan_() -> Tensor

In-place version of :meth:`~Tensor.atan`
""",
)

add_docstr_all(
    "arctan",
    r"""
arctan() -> Tensor

See :func:`torch.arctan`
""",
)

add_docstr_all(
    "arctan_",
    r"""
arctan_() -> Tensor

In-place version of :meth:`~Tensor.arctan`
""",
)

add_docstr_all(
    "atan2",
    r"""
atan2(other) -> Tensor

See :func:`torch.atan2`
""",
)

add_docstr_all(
    "atan2_",
    r"""
atan2_(other) -> Tensor

In-place version of :meth:`~Tensor.atan2`
""",
)

add_docstr_all(
    "arctan2",
    r"""
arctan2(other) -> Tensor

See :func:`torch.arctan2`
""",
)

add_docstr_all(
    "arctan2_",
    r"""
atan2_(other) -> Tensor

In-place version of :meth:`~Tensor.arctan2`
""",
)

add_docstr_all(
    "atanh",
    r"""
atanh() -> Tensor

See :func:`torch.atanh`
""",
)

add_docstr_all(
    "atanh_",
    r"""
atanh_(other) -> Tensor

In-place version of :meth:`~Tensor.atanh`
""",
)

add_docstr_all(
    "arctanh",
    r"""
arctanh() -> Tensor

See :func:`torch.arctanh`
""",
)

add_docstr_all(
    "arctanh_",
    r"""
arctanh_(other) -> Tensor

In-place version of :meth:`~Tensor.arctanh`
""",
)

add_docstr_all(
    "baddbmm",
    r"""
baddbmm(batch1, batch2, *, beta=1, alpha=1) -> Tensor

See :func:`torch.baddbmm`
""",
)

add_docstr_all(
    "baddbmm_",
    r"""
baddbmm_(batch1, batch2, *, beta=1, alpha=1) -> Tensor

In-place version of :meth:`~Tensor.baddbmm`
""",
)

add_docstr_all(
    "bernoulli",
    r"""
bernoulli(*, generator=None) -> Tensor

Returns a result tensor where each :math:`\texttt{result[i]}` is independently
sampled from :math:`\text{Bernoulli}(\texttt{self[i]})`. :attr:`self` must have
floating point ``dtype``, and the result will have the same ``dtype``.

See :func:`torch.bernoulli`
""",
)

add_docstr_all(
    "bernoulli_",
    r"""
bernoulli_(p=0.5, *, generator=None) -> Tensor

Fills each location of :attr:`self` with an independent sample from
:math:`\text{Bernoulli}(\texttt{p})`. :attr:`self` can have integral
``dtype``.

:attr:`p` should either be a scalar or tensor containing probabilities to be
used for drawing the binary random number.

If it is a tensor, the :math:`\text{i}^{th}` element of :attr:`self` tensor
will be set to a value sampled from
:math:`\text{Bernoulli}(\texttt{p\_tensor[i]})`. In this case `p` must have
floating point ``dtype``.

See also :meth:`~Tensor.bernoulli` and :func:`torch.bernoulli`
""",
)

add_docstr_all(
    "bincount",
    r"""
bincount(weights=None, minlength=0) -> Tensor

See :func:`torch.bincount`
""",
)

add_docstr_all(
    "bitwise_not",
    r"""
bitwise_not() -> Tensor

See :func:`torch.bitwise_not`
""",
)

add_docstr_all(
    "bitwise_not_",
    r"""
bitwise_not_() -> Tensor

In-place version of :meth:`~Tensor.bitwise_not`
""",
)

add_docstr_all(
    "bitwise_and",
    r"""
bitwise_and() -> Tensor

See :func:`torch.bitwise_and`
""",
)

add_docstr_all(
    "bitwise_and_",
    r"""
bitwise_and_() -> Tensor

In-place version of :meth:`~Tensor.bitwise_and`
""",
)

add_docstr_all(
    "bitwise_or",
    r"""
bitwise_or() -> Tensor

See :func:`torch.bitwise_or`
""",
)

add_docstr_all(
    "bitwise_or_",
    r"""
bitwise_or_() -> Tensor

In-place version of :meth:`~Tensor.bitwise_or`
""",
)

add_docstr_all(
    "bitwise_xor",
    r"""
bitwise_xor() -> Tensor

See :func:`torch.bitwise_xor`
""",
)

add_docstr_all(
    "bitwise_xor_",
    r"""
bitwise_xor_() -> Tensor

In-place version of :meth:`~Tensor.bitwise_xor`
""",
)

add_docstr_all(
    "bitwise_left_shift",
    r"""
bitwise_left_shift(other) -> Tensor

See :func:`torch.bitwise_left_shift`
""",
)

add_docstr_all(
    "bitwise_left_shift_",
    r"""
bitwise_left_shift_(other) -> Tensor

In-place version of :meth:`~Tensor.bitwise_left_shift`
""",
)

add_docstr_all(
    "bitwise_right_shift",
    r"""
bitwise_right_shift(other) -> Tensor

See :func:`torch.bitwise_right_shift`
""",
)

add_docstr_all(
    "bitwise_right_shift_",
    r"""
bitwise_right_shift_(other) -> Tensor

In-place version of :meth:`~Tensor.bitwise_right_shift`
""",
)

add_docstr_all(
    "broadcast_to",
    r"""
broadcast_to(shape) -> Tensor

See :func:`torch.broadcast_to`.
""",
)

add_docstr_all(
    "logical_and",
    r"""
logical_and() -> Tensor

See :func:`torch.logical_and`
""",
)

add_docstr_all(
    "logical_and_",
    r"""
logical_and_() -> Tensor

In-place version of :meth:`~Tensor.logical_and`
""",
)

add_docstr_all(
    "logical_not",
    r"""
logical_not() -> Tensor

See :func:`torch.logical_not`
""",
)

add_docstr_all(
    "logical_not_",
    r"""
logical_not_() -> Tensor

In-place version of :meth:`~Tensor.logical_not`
""",
)

add_docstr_all(
    "logical_or",
    r"""
logical_or() -> Tensor

See :func:`torch.logical_or`
""",
)

add_docstr_all(
    "logical_or_",
    r"""
logical_or_() -> Tensor

In-place version of :meth:`~Tensor.logical_or`
""",
)

add_docstr_all(
    "logical_xor",
    r"""
logical_xor() -> Tensor

See :func:`torch.logical_xor`
""",
)

add_docstr_all(
    "logical_xor_",
    r"""
logical_xor_() -> Tensor

In-place version of :meth:`~Tensor.logical_xor`
""",
)

add_docstr_all(
    "bmm",
    r"""
bmm(batch2) -> Tensor

See :func:`torch.bmm`
""",
)

add_docstr_all(
    "cauchy_",
    r"""
cauchy_(median=0, sigma=1, *, generator=None) -> Tensor

Fills the tensor with numbers drawn from the Cauchy distribution:

.. math::

    f(x) = \dfrac{1}{\pi} \dfrac{\sigma}{(x - \text{median})^2 + \sigma^2}

.. note::
  Sigma (:math:`\sigma`) is used to denote the scale parameter in Cauchy distribution.
""",
)

add_docstr_all(
    "ceil",
    r"""
ceil() -> Tensor

See :func:`torch.ceil`
""",
)

add_docstr_all(
    "ceil_",
    r"""
ceil_() -> Tensor

In-place version of :meth:`~Tensor.ceil`
""",
)

add_docstr_all(
    "cholesky",
    r"""
cholesky(upper=False) -> Tensor

See :func:`torch.cholesky`
""",
)

add_docstr_all(
    "cholesky_solve",
    r"""
cholesky_solve(input2, upper=False) -> Tensor

See :func:`torch.cholesky_solve`
""",
)

add_docstr_all(
    "cholesky_inverse",
    r"""
cholesky_inverse(upper=False) -> Tensor

See :func:`torch.cholesky_inverse`
""",
)

add_docstr_all(
    "clamp",
    r"""
clamp(min=None, max=None) -> Tensor

See :func:`torch.clamp`
""",
)

add_docstr_all(
    "clamp_",
    r"""
clamp_(min=None, max=None) -> Tensor

In-place version of :meth:`~Tensor.clamp`
""",
)

add_docstr_all(
    "clip",
    r"""
clip(min=None, max=None) -> Tensor

Alias for :meth:`~Tensor.clamp`.
""",
)

add_docstr_all(
    "clip_",
    r"""
clip_(min=None, max=None) -> Tensor

Alias for :meth:`~Tensor.clamp_`.
""",
)

add_docstr_all(
    "clone",
    r"""
clone(*, memory_format=torch.preserve_format) -> Tensor

See :func:`torch.clone`
""".format(**common_args),
)

add_docstr_all(
    "coalesce",
    r"""
coalesce() -> Tensor

Returns a coalesced copy of :attr:`self` if :attr:`self` is an
:ref:`uncoalesced tensor <sparse-uncoalesced-coo-docs>`.

Returns :attr:`self` if :attr:`self` is a coalesced tensor.

.. warning::
  Throws an error if :attr:`self` is not a sparse COO tensor.
""",
)

add_docstr_all(
    "contiguous",
    r"""
contiguous(memory_format=torch.contiguous_format) -> Tensor

Returns a contiguous in memory tensor containing the same data as :attr:`self` tensor. If
:attr:`self` tensor is already in the specified memory format, this function returns the
:attr:`self` tensor.

Args:
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.contiguous_format``.
""",
)

add_docstr_all(
    "copy_",
    r"""
copy_(src, non_blocking=False) -> Tensor

Copies the elements from :attr:`src` into :attr:`self` tensor and returns
:attr:`self`.

The :attr:`src` tensor must be :ref:`broadcastable <broadcasting-semantics>`
with the :attr:`self` tensor. It may be of a different data type or reside on a
different device.

Args:
    src (Tensor): the source tensor to copy from
    non_blocking (bool, optional): if ``True`` and this copy is between CPU and GPU,
        the copy may occur asynchronously with respect to the host. For other
        cases, this argument has no effect. Default: ``False``
""",
)

add_docstr_all(
    "conj",
    r"""
conj() -> Tensor

See :func:`torch.conj`
""",
)

add_docstr_all(
    "conj_physical",
    r"""
conj_physical() -> Tensor

See :func:`torch.conj_physical`
""",
)

add_docstr_all(
    "conj_physical_",
    r"""
conj_physical_() -> Tensor

In-place version of :meth:`~Tensor.conj_physical`
""",
)

add_docstr_all(
    "resolve_conj",
    r"""
resolve_conj() -> Tensor

See :func:`torch.resolve_conj`
""",
)

add_docstr_all(
    "resolve_neg",
    r"""
resolve_neg() -> Tensor

See :func:`torch.resolve_neg`
""",
)

add_docstr_all(
    "copysign",
    r"""
copysign(other) -> Tensor

See :func:`torch.copysign`
""",
)

add_docstr_all(
    "copysign_",
    r"""
copysign_(other) -> Tensor

In-place version of :meth:`~Tensor.copysign`
""",
)

add_docstr_all(
    "cos",
    r"""
cos() -> Tensor

See :func:`torch.cos`
""",
)

add_docstr_all(
    "cos_",
    r"""
cos_() -> Tensor

In-place version of :meth:`~Tensor.cos`
""",
)

add_docstr_all(
    "cosh",
    r"""
cosh() -> Tensor

See :func:`torch.cosh`
""",
)

add_docstr_all(
    "cosh_",
    r"""
cosh_() -> Tensor

In-place version of :meth:`~Tensor.cosh`
""",
)

add_docstr_all(
    "cpu",
    r"""
cpu(memory_format=torch.preserve_format) -> Tensor

Returns a copy of this object in CPU memory.

If this object is already in CPU memory,
then no copy is performed and the original object is returned.

Args:
    {memory_format}

""".format(**common_args),
)

add_docstr_all(
    "count_nonzero",
    r"""
count_nonzero(dim=None) -> Tensor

See :func:`torch.count_nonzero`
""",
)

add_docstr_all(
    "cov",
    r"""
cov(*, correction=1, fweights=None, aweights=None) -> Tensor

See :func:`torch.cov`
""",
)

add_docstr_all(
    "corrcoef",
    r"""
corrcoef() -> Tensor

See :func:`torch.corrcoef`
""",
)

add_docstr_all(
    "cross",
    r"""
cross(other, dim=None) -> Tensor

See :func:`torch.cross`
""",
)

add_docstr_all(
    "cuda",
    r"""
cuda(device=None, non_blocking=False, memory_format=torch.preserve_format) -> Tensor

Returns a copy of this object in CUDA memory.

If this object is already in CUDA memory and on the correct device,
then no copy is performed and the original object is returned.

Args:
    device (:class:`torch.device`, optional): The destination GPU device.
        Defaults to the current CUDA device.
    non_blocking (bool, optional): If ``True`` and the source is in pinned memory,
        the copy will be asynchronous with respect to the host.
        Otherwise, the argument has no effect. Default: ``False``.
    {memory_format}
""".format(**common_args),
)

add_docstr_all(
    "mtia",
    r"""
mtia(device=None, non_blocking=False, memory_format=torch.preserve_format) -> Tensor

Returns a copy of this object in MTIA memory.

If this object is already in MTIA memory and on the correct device,
then no copy is performed and the original object is returned.

Args:
    device (:class:`torch.device`, optional): The destination MTIA device.
        Defaults to the current MTIA device.
    non_blocking (bool, optional): If ``True`` and the source is in pinned memory,
        the copy will be asynchronous with respect to the host.
        Otherwise, the argument has no effect. Default: ``False``.
    {memory_format}
""".format(**common_args),
)

add_docstr_all(
    "ipu",
    r"""
ipu(device=None, non_blocking=False, memory_format=torch.preserve_format) -> Tensor

Returns a copy of this object in IPU memory.

If this object is already in IPU memory and on the correct device,
then no copy is performed and the original object is returned.

Args:
    device (:class:`torch.device`, optional): The destination IPU device.
        Defaults to the current IPU device.
    non_blocking (bool, optional): If ``True`` and the source is in pinned memory,
        the copy will be asynchronous with respect to the host.
        Otherwise, the argument has no effect. Default: ``False``.
    {memory_format}
""".format(**common_args),
)

add_docstr_all(
    "xpu",
    r"""
xpu(device=None, non_blocking=False, memory_format=torch.preserve_format) -> Tensor

Returns a copy of this object in XPU memory.

If this object is already in XPU memory and on the correct device,
then no copy is performed and the original object is returned.

Args:
    device (:class:`torch.device`, optional): The destination XPU device.
        Defaults to the current XPU device.
    non_blocking (bool, optional): If ``True`` and the source is in pinned memory,
        the copy will be asynchronous with respect to the host.
        Otherwise, the argument has no effect. Default: ``False``.
    {memory_format}
""".format(**common_args),
)

add_docstr_all(
    "logcumsumexp",
    r"""
logcumsumexp(dim) -> Tensor

See :func:`torch.logcumsumexp`
""",
)

add_docstr_all(
    "cummax",
    r"""
cummax(dim) -> (Tensor, Tensor)

See :func:`torch.cummax`
""",
)

add_docstr_all(
    "cummin",
    r"""
cummin(dim) -> (Tensor, Tensor)

See :func:`torch.cummin`
""",
)

add_docstr_all(
    "cumprod",
    r"""
cumprod(dim, dtype=None) -> Tensor

See :func:`torch.cumprod`
""",
)

add_docstr_all(
    "cumprod_",
    r"""
cumprod_(dim, dtype=None) -> Tensor

In-place version of :meth:`~Tensor.cumprod`
""",
)

add_docstr_all(
    "cumsum",
    r"""
cumsum(dim, dtype=None) -> Tensor

See :func:`torch.cumsum`
""",
)

add_docstr_all(
    "cumsum_",
    r"""
cumsum_(dim, dtype=None) -> Tensor

In-place version of :meth:`~Tensor.cumsum`
""",
)

add_docstr_all(
    "data_ptr",
    r"""
data_ptr() -> int

Returns the address of the first element of :attr:`self` tensor.
""",
)

add_docstr_all(
    "dequantize",
    r"""
dequantize() -> Tensor

Given a quantized Tensor, dequantize it and return the dequantized float Tensor.
""",
)

add_docstr_all(
    "dense_dim",
    r"""
dense_dim() -> int

Return the number of dense dimensions in a :ref:`sparse tensor <sparse-docs>` :attr:`self`.

.. note::
  Returns ``len(self.shape)`` if :attr:`self` is not a sparse tensor.

See also :meth:`Tensor.sparse_dim` and :ref:`hybrid tensors <sparse-hybrid-coo-docs>`.
""",
)

add_docstr_all(
    "diag",
    r"""
diag(diagonal=0) -> Tensor

See :func:`torch.diag`
""",
)

add_docstr_all(
    "diag_embed",
    r"""
diag_embed(offset=0, dim1=-2, dim2=-1) -> Tensor

See :func:`torch.diag_embed`
""",
)

add_docstr_all(
    "diagflat",
    r"""
diagflat(offset=0) -> Tensor

See :func:`torch.diagflat`
""",
)

add_docstr_all(
    "diagonal",
    r"""
diagonal(offset=0, dim1=0, dim2=1) -> Tensor

See :func:`torch.diagonal`
""",
)

add_docstr_all(
    "diagonal_scatter",
    r"""
diagonal_scatter(src, offset=0, dim1=0, dim2=1) -> Tensor

See :func:`torch.diagonal_scatter`
""",
)

add_docstr_all(
    "as_strided_scatter",
    r"""
as_strided_scatter(src, size, stride, storage_offset=None) -> Tensor

See :func:`torch.as_strided_scatter`
""",
)

add_docstr_all(
    "fill_diagonal_",
    r"""
fill_diagonal_(fill_value, wrap=False) -> Tensor

Fill the main diagonal of a tensor that has at least 2-dimensions.
When dims>2, all dimensions of input must be of equal length.
This function modifies the input tensor in-place, and returns the input tensor.

Arguments:
    fill_value (Scalar): the fill value
    wrap (bool, optional): the diagonal 'wrapped' after N columns for tall matrices. Default: ``False``

Example::

    >>> a = torch.zeros(3, 3)
    >>> a.fill_diagonal_(5)
    tensor([[5., 0., 0.],
            [0., 5., 0.],
            [0., 0., 5.]])
    >>> b = torch.zeros(7, 3)
    >>> b.fill_diagonal_(5)
    tensor([[5., 0., 0.],
            [0., 5., 0.],
            [0., 0., 5.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]])
    >>> c = torch.zeros(7, 3)
    >>> c.fill_diagonal_(5, wrap=True)
    tensor([[5., 0., 0.],
            [0., 5., 0.],
            [0., 0., 5.],
            [0., 0., 0.],
            [5., 0., 0.],
            [0., 5., 0.],
            [0., 0., 5.]])

""",
)

add_docstr_all(
    "floor_divide",
    r"""
floor_divide(value) -> Tensor

See :func:`torch.floor_divide`
""",
)

add_docstr_all(
    "floor_divide_",
    r"""
floor_divide_(value) -> Tensor

In-place version of :meth:`~Tensor.floor_divide`
""",
)

add_docstr_all(
    "diff",
    r"""
diff(n=1, dim=-1, prepend=None, append=None) -> Tensor

See :func:`torch.diff`
""",
)

add_docstr_all(
    "digamma",
    r"""
digamma() -> Tensor

See :func:`torch.digamma`
""",
)

add_docstr_all(
    "digamma_",
    r"""
digamma_() -> Tensor

In-place version of :meth:`~Tensor.digamma`
""",
)

add_docstr_all(
    "dim",
    r"""
dim() -> int

Returns the number of dimensions of :attr:`self` tensor.
""",
)

add_docstr_all(
    "dist",
    r"""
dist(other, p=2) -> Tensor

See :func:`torch.dist`
""",
)

add_docstr_all(
    "div",
    r"""
div(value, *, rounding_mode=None) -> Tensor

See :func:`torch.div`
""",
)

add_docstr_all(
    "div_",
    r"""
div_(value, *, rounding_mode=None) -> Tensor

In-place version of :meth:`~Tensor.div`
""",
)

add_docstr_all(
    "divide",
    r"""
divide(value, *, rounding_mode=None) -> Tensor

See :func:`torch.divide`
""",
)

add_docstr_all(
    "divide_",
    r"""
divide_(value, *, rounding_mode=None) -> Tensor

In-place version of :meth:`~Tensor.divide`
""",
)

add_docstr_all(
    "dot",
    r"""
dot(other) -> Tensor

See :func:`torch.dot`
""",
)

add_docstr_all(
    "element_size",
    r"""
element_size() -> int

Returns the size in bytes of an individual element.

Example::

    >>> torch.tensor([]).element_size()
    4
    >>> torch.tensor([], dtype=torch.uint8).element_size()
    1

""",
)

add_docstr_all(
    "eq",
    r"""
eq(other) -> Tensor

See :func:`torch.eq`
""",
)

add_docstr_all(
    "eq_",
    r"""
eq_(other) -> Tensor

In-place version of :meth:`~Tensor.eq`
""",
)

add_docstr_all(
    "equal",
    r"""
equal(other) -> bool

See :func:`torch.equal`
""",
)

add_docstr_all(
    "erf",
    r"""
erf() -> Tensor

See :func:`torch.erf`
""",
)

add_docstr_all(
    "erf_",
    r"""
erf_() -> Tensor

In-place version of :meth:`~Tensor.erf`
""",
)

add_docstr_all(
    "erfc",
    r"""
erfc() -> Tensor

See :func:`torch.erfc`
""",
)

add_docstr_all(
    "erfc_",
    r"""
erfc_() -> Tensor

In-place version of :meth:`~Tensor.erfc`
""",
)

add_docstr_all(
    "erfinv",
    r"""
erfinv() -> Tensor

See :func:`torch.erfinv`
""",
)

add_docstr_all(
    "erfinv_",
    r"""
erfinv_() -> Tensor

In-place version of :meth:`~Tensor.erfinv`
""",
)

add_docstr_all(
    "exp",
    r"""
exp() -> Tensor

See :func:`torch.exp`
""",
)

add_docstr_all(
    "exp_",
    r"""
exp_() -> Tensor

In-place version of :meth:`~Tensor.exp`
""",
)

add_docstr_all(
    "exp2",
    r"""
exp2() -> Tensor

See :func:`torch.exp2`
""",
)

add_docstr_all(
    "exp2_",
    r"""
exp2_() -> Tensor

In-place version of :meth:`~Tensor.exp2`
""",
)

add_docstr_all(
    "expm1",
    r"""
expm1() -> Tensor

See :func:`torch.expm1`
""",
)

add_docstr_all(
    "expm1_",
    r"""
expm1_() -> Tensor

In-place version of :meth:`~Tensor.expm1`
""",
)

add_docstr_all(
    "exponential_",
    r"""
exponential_(lambd=1, *, generator=None) -> Tensor

Fills :attr:`self` tensor with elements drawn from the PDF (probability density function):

.. math::

    f(x) = \lambda e^{-\lambda x}, x > 0

.. note::
  In probability theory, exponential distribution is supported on interval [0, :math:`\inf`) (i.e., :math:`x >= 0`)
  implying that zero can be sampled from the exponential distribution.
  However, :func:`torch.Tensor.exponential_` does not sample zero,
  which means that its actual support is the interval (0, :math:`\inf`).

  Note that :func:`torch.distributions.exponential.Exponential` is supported on the interval [0, :math:`\inf`) and can sample zero.
""",
)

add_docstr_all(
    "fill_",
    r"""
fill_(value) -> Tensor

Fills :attr:`self` tensor with the specified value.
""",
)

add_docstr_all(
    "floor",
    r"""
floor() -> Tensor

See :func:`torch.floor`
""",
)

add_docstr_all(
    "flip",
    r"""
flip(dims) -> Tensor

See :func:`torch.flip`
""",
)

add_docstr_all(
    "fliplr",
    r"""
fliplr() -> Tensor

See :func:`torch.fliplr`
""",
)

add_docstr_all(
    "flipud",
    r"""
flipud() -> Tensor

See :func:`torch.flipud`
""",
)

add_docstr_all(
    "roll",
    r"""
roll(shifts, dims) -> Tensor

See :func:`torch.roll`
""",
)

add_docstr_all(
    "floor_",
    r"""
floor_() -> Tensor

In-place version of :meth:`~Tensor.floor`
""",
)

add_docstr_all(
    "fmod",
    r"""
fmod(divisor) -> Tensor

See :func:`torch.fmod`
""",
)

add_docstr_all(
    "fmod_",
    r"""
fmod_(divisor) -> Tensor

In-place version of :meth:`~Tensor.fmod`
""",
)

add_docstr_all(
    "frac",
    r"""
frac() -> Tensor

See :func:`torch.frac`
""",
)

add_docstr_all(
    "frac_",
    r"""
frac_() -> Tensor

In-place version of :meth:`~Tensor.frac`
""",
)

add_docstr_all(
    "frexp",
    r"""
frexp(input) -> (Tensor mantissa, Tensor exponent)

See :func:`torch.frexp`
""",
)

add_docstr_all(
    "flatten",
    r"""
flatten(start_dim=0, end_dim=-1) -> Tensor

See :func:`torch.flatten`
""",
)

add_docstr_all(
    "gather",
    r"""
gather(dim, index) -> Tensor

See :func:`torch.gather`
""",
)

add_docstr_all(
    "gcd",
    r"""
gcd(other) -> Tensor

See :func:`torch.gcd`
""",
)

add_docstr_all(
    "gcd_",
    r"""
gcd_(other) -> Tensor

In-place version of :meth:`~Tensor.gcd`
""",
)

add_docstr_all(
    "ge",
    r"""
ge(other) -> Tensor

See :func:`torch.ge`.
""",
)

add_docstr_all(
    "ge_",
    r"""
ge_(other) -> Tensor

In-place version of :meth:`~Tensor.ge`.
""",
)

add_docstr_all(
    "greater_equal",
    r"""
greater_equal(other) -> Tensor

See :func:`torch.greater_equal`.
""",
)

add_docstr_all(
    "greater_equal_",
    r"""
greater_equal_(other) -> Tensor

In-place version of :meth:`~Tensor.greater_equal`.
""",
)

add_docstr_all(
    "geometric_",
    r"""
geometric_(p, *, generator=None) -> Tensor

Fills :attr:`self` tensor with elements drawn from the geometric distribution:

.. math::

    P(X=k) = (1 - p)^{k - 1} p, k = 1, 2, ...

.. note::
  :func:`torch.Tensor.geometric_` `k`-th trial is the first success hence draws samples in :math:`\{1, 2, \ldots\}`, whereas
  :func:`torch.distributions.geometric.Geometric` :math:`(k+1)`-th trial is the first success
  hence draws samples in :math:`\{0, 1, \ldots\}`.
""",
)

add_docstr_all(
    "geqrf",
    r"""
geqrf() -> (Tensor, Tensor)

See :func:`torch.geqrf`
""",
)

add_docstr_all(
    "ger",
    r"""
ger(vec2) -> Tensor

See :func:`torch.ger`
""",
)

add_docstr_all(
    "inner",
    r"""
inner(other) -> Tensor

See :func:`torch.inner`.
""",
)

add_docstr_all(
    "outer",
    r"""
outer(vec2) -> Tensor

See :func:`torch.outer`.
""",
)

add_docstr_all(
    "hypot",
    r"""
hypot(other) -> Tensor

See :func:`torch.hypot`
""",
)

add_docstr_all(
    "hypot_",
    r"""
hypot_(other) -> Tensor

In-place version of :meth:`~Tensor.hypot`
""",
)

add_docstr_all(
    "i0",
    r"""
i0() -> Tensor

See :func:`torch.i0`
""",
)

add_docstr_all(
    "i0_",
    r"""
i0_() -> Tensor

In-place version of :meth:`~Tensor.i0`
""",
)

add_docstr_all(
    "igamma",
    r"""
igamma(other) -> Tensor

See :func:`torch.igamma`
""",
)

add_docstr_all(
    "igamma_",
    r"""
igamma_(other) -> Tensor

In-place version of :meth:`~Tensor.igamma`
""",
)

add_docstr_all(
    "igammac",
    r"""
igammac(other) -> Tensor
See :func:`torch.igammac`
""",
)

add_docstr_all(
    "igammac_",
    r"""
igammac_(other) -> Tensor
In-place version of :meth:`~Tensor.igammac`
""",
)

add_docstr_all(
    "indices",
    r"""
indices() -> Tensor

Return the indices tensor of a :ref:`sparse COO tensor <sparse-coo-docs>`.

.. warning::
  Throws an error if :attr:`self` is not a sparse COO tensor.

See also :meth:`Tensor.values`.

.. note::
  This method can only be called on a coalesced sparse tensor. See
  :meth:`Tensor.coalesce` for details.
""",
)

add_docstr_all(
    "get_device",
    r"""
get_device() -> Device ordinal (Integer)

For CUDA tensors, this function returns the device ordinal of the GPU on which the tensor resides.
For CPU tensors, this function returns `-1`.

Example::

    >>> x = torch.randn(3, 4, 5, device='cuda:0')
    >>> x.get_device()
    0
    >>> x.cpu().get_device()
    -1
""",
)

add_docstr_all(
    "values",
    r"""
values() -> Tensor

Return the values tensor of a :ref:`sparse COO tensor <sparse-coo-docs>`.

.. warning::
  Throws an error if :attr:`self` is not a sparse COO tensor.

See also :meth:`Tensor.indices`.

.. note::
  This method can only be called on a coalesced sparse tensor. See
  :meth:`Tensor.coalesce` for details.
""",
)

add_docstr_all(
    "gt",
    r"""
gt(other) -> Tensor

See :func:`torch.gt`.
""",
)

add_docstr_all(
    "gt_",
    r"""
gt_(other) -> Tensor

In-place version of :meth:`~Tensor.gt`.
""",
)

add_docstr_all(
    "greater",
    r"""
greater(other) -> Tensor

See :func:`torch.greater`.
""",
)

add_docstr_all(
    "greater_",
    r"""
greater_(other) -> Tensor

In-place version of :meth:`~Tensor.greater`.
""",
)

add_docstr_all(
    "has_names",
    r"""
Is ``True`` if any of this tensor's dimensions are named. Otherwise, is ``False``.
""",
)

add_docstr_all(
    "hardshrink",
    r"""
hardshrink(lambd=0.5) -> Tensor

See :func:`torch.nn.functional.hardshrink`
""",
)

add_docstr_all(
    "heaviside",
    r"""
heaviside(values) -> Tensor

See :func:`torch.heaviside`
""",
)

add_docstr_all(
    "heaviside_",
    r"""
heaviside_(values) -> Tensor

In-place version of :meth:`~Tensor.heaviside`
""",
)

add_docstr_all(
    "histc",
    r"""
histc(bins=100, min=0, max=0) -> Tensor

See :func:`torch.histc`
""",
)

add_docstr_all(
    "histogram",
    r"""
histogram(input, bins, *, range=None, weight=None, density=False) -> (Tensor, Tensor)

See :func:`torch.histogram`
""",
)

add_docstr_all(
    "index_add_",
    r"""
index_add_(dim, index, source, *, alpha=1) -> Tensor

Accumulate the elements of :attr:`alpha` times ``source`` into the :attr:`self`
tensor by adding to the indices in the order given in :attr:`index`. For example,
if ``dim == 0``, ``index[i] == j``, and ``alpha=-1``, then the ``i``\ th row of
``source`` is subtracted from the ``j``\ th row of :attr:`self`.

The :attr:`dim`\ th dimension of ``source`` must have the same size as the
length of :attr:`index` (which must be a vector), and all other dimensions must
match :attr:`self`, or an error will be raised.

For a 3-D tensor the output is given as::

    self[index[i], :, :] += alpha * src[i, :, :]  # if dim == 0
    self[:, index[i], :] += alpha * src[:, i, :]  # if dim == 1
    self[:, :, index[i]] += alpha * src[:, :, i]  # if dim == 2

Note:
    {forward_reproducibility_note}

Args:
    dim (int): dimension along which to index
    index (Tensor): indices of ``source`` to select from,
            should have dtype either `torch.int64` or `torch.int32`
    source (Tensor): the tensor containing values to add

Keyword args:
    alpha (Number): the scalar multiplier for ``source``

Example::

    >>> x = torch.ones(5, 3)
    >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
    >>> index = torch.tensor([0, 4, 2])
    >>> x.index_add_(0, index, t)
    tensor([[  2.,   3.,   4.],
            [  1.,   1.,   1.],
            [  8.,   9.,  10.],
            [  1.,   1.,   1.],
            [  5.,   6.,   7.]])
    >>> x.index_add_(0, index, t, alpha=-1)
    tensor([[  1.,   1.,   1.],
            [  1.,   1.,   1.],
            [  1.,   1.,   1.],
            [  1.,   1.,   1.],
            [  1.,   1.,   1.]])
""".format(**reproducibility_notes),
)

add_docstr_all(
    "index_copy_",
    r"""
index_copy_(dim, index, tensor) -> Tensor

Copies the elements of :attr:`tensor` into the :attr:`self` tensor by selecting
the indices in the order given in :attr:`index`. For example, if ``dim == 0``
and ``index[i] == j``, then the ``i``\ th row of :attr:`tensor` is copied to the
``j``\ th row of :attr:`self`.

The :attr:`dim`\ th dimension of :attr:`tensor` must have the same size as the
length of :attr:`index` (which must be a vector), and all other dimensions must
match :attr:`self`, or an error will be raised.

.. note::
    If :attr:`index` contains duplicate entries, multiple elements from
    :attr:`tensor` will be copied to the same index of :attr:`self`. The result
    is nondeterministic since it depends on which copy occurs last.

Args:
    dim (int): dimension along which to index
    index (LongTensor): indices of :attr:`tensor` to select from
    tensor (Tensor): the tensor containing values to copy

Example::

    >>> x = torch.zeros(5, 3)
    >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
    >>> index = torch.tensor([0, 4, 2])
    >>> x.index_copy_(0, index, t)
    tensor([[ 1.,  2.,  3.],
            [ 0.,  0.,  0.],
            [ 7.,  8.,  9.],
            [ 0.,  0.,  0.],
            [ 4.,  5.,  6.]])
""",
)

add_docstr_all(
    "index_fill_",
    r"""
index_fill_(dim, index, value) -> Tensor

Fills the elements of the :attr:`self` tensor with value :attr:`value` by
selecting the indices in the order given in :attr:`index`.

Args:
    dim (int): dimension along which to index
    index (LongTensor): indices of :attr:`self` tensor to fill in
    value (float): the value to fill with

Example::

    >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
    >>> index = torch.tensor([0, 2])
    >>> x.index_fill_(1, index, -1)
    tensor([[-1.,  2., -1.],
            [-1.,  5., -1.],
            [-1.,  8., -1.]])
""",
)

add_docstr_all(
    "index_put_",
    r"""
index_put_(indices, values, accumulate=False) -> Tensor

Puts values from the tensor :attr:`values` into the tensor :attr:`self` using
the indices specified in :attr:`indices` (which is a tuple of Tensors). The
expression ``tensor.index_put_(indices, values)`` is equivalent to
``tensor[indices] = values``. Returns :attr:`self`.

If :attr:`accumulate` is ``True``, the elements in :attr:`values` are added to
:attr:`self`. If accumulate is ``False``, the behavior is undefined if indices
contain duplicate elements.

Args:
    indices (tuple of LongTensor): tensors used to index into `self`.
    values (Tensor): tensor of same dtype as `self`.
    accumulate (bool): whether to accumulate into self
""",
)

add_docstr_all(
    "index_put",
    r"""
index_put(indices, values, accumulate=False) -> Tensor

Out-place version of :meth:`~Tensor.index_put_`.
""",
)

add_docstr_all(
    "index_reduce_",
    r"""
index_reduce_(dim, index, source, reduce, *, include_self=True) -> Tensor

Accumulate the elements of ``source`` into the :attr:`self`
tensor by accumulating to the indices in the order given in :attr:`index`
using the reduction given by the ``reduce`` argument. For example, if ``dim == 0``,
``index[i] == j``, ``reduce == prod`` and ``include_self == True`` then the ``i``\ th
row of ``source`` is multiplied by the ``j``\ th row of :attr:`self`. If
:obj:`include_self="True"`, the values in the :attr:`self` tensor are included
in the reduction, otherwise, rows in the :attr:`self` tensor that are accumulated
to are treated as if they were filled with the reduction identities.

The :attr:`dim`\ th dimension of ``source`` must have the same size as the
length of :attr:`index` (which must be a vector), and all other dimensions must
match :attr:`self`, or an error will be raised.

For a 3-D tensor with :obj:`reduce="prod"` and :obj:`include_self=True` the
output is given as::

    self[index[i], :, :] *= src[i, :, :]  # if dim == 0
    self[:, index[i], :] *= src[:, i, :]  # if dim == 1
    self[:, :, index[i]] *= src[:, :, i]  # if dim == 2

Note:
    {forward_reproducibility_note}

.. note::

    This function only supports floating point tensors.

.. warning::

    This function is in beta and may change in the near future.

Args:
    dim (int): dimension along which to index
    index (Tensor): indices of ``source`` to select from,
        should have dtype either `torch.int64` or `torch.int32`
    source (FloatTensor): the tensor containing values to accumulate
    reduce (str): the reduction operation to apply
        (:obj:`"prod"`, :obj:`"mean"`, :obj:`"amax"`, :obj:`"amin"`)

Keyword args:
    include_self (bool): whether the elements from the ``self`` tensor are
        included in the reduction

Example::

    >>> x = torch.empty(5, 3).fill_(2)
    >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.float)
    >>> index = torch.tensor([0, 4, 2, 0])
    >>> x.index_reduce_(0, index, t, 'prod')
    tensor([[20., 44., 72.],
            [ 2.,  2.,  2.],
            [14., 16., 18.],
            [ 2.,  2.,  2.],
            [ 8., 10., 12.]])
    >>> x = torch.empty(5, 3).fill_(2)
    >>> x.index_reduce_(0, index, t, 'prod', include_self=False)
    tensor([[10., 22., 36.],
            [ 2.,  2.,  2.],
            [ 7.,  8.,  9.],
            [ 2.,  2.,  2.],
            [ 4.,  5.,  6.]])
""".format(**reproducibility_notes),
)

add_docstr_all(
    "index_select",
    r"""
index_select(dim, index) -> Tensor

See :func:`torch.index_select`
""",
)

add_docstr_all(
    "sparse_mask",
    r"""
sparse_mask(mask) -> Tensor

Returns a new :ref:`sparse tensor <sparse-docs>` with values from a
strided tensor :attr:`self` filtered by the indices of the sparse
tensor :attr:`mask`. The values of :attr:`mask` sparse tensor are
ignored. :attr:`self` and :attr:`mask` tensors must have the same
shape.

.. note::

  The returned sparse tensor might contain duplicate values if :attr:`mask`
  is not coalesced. It is therefore advisable to pass ``mask.coalesce()``
  if such behavior is not desired.

.. note::

  The returned sparse tensor has the same indices as the sparse tensor
  :attr:`mask`, even when the corresponding values in :attr:`self` are
  zeros.

Args:
    mask (Tensor): a sparse tensor whose indices are used as a filter

Example::

    >>> nse = 5
    >>> dims = (5, 5, 2, 2)
    >>> I = torch.cat([torch.randint(0, dims[0], size=(nse,)),
    ...                torch.randint(0, dims[1], size=(nse,))], 0).reshape(2, nse)
    >>> V = torch.randn(nse, dims[2], dims[3])
    >>> S = torch.sparse_coo_tensor(I, V, dims).coalesce()
    >>> D = torch.randn(dims)
    >>> D.sparse_mask(S)
    tensor(indices=tensor([[0, 0, 0, 2],
                           [0, 1, 4, 3]]),
           values=tensor([[[ 1.6550,  0.2397],
                           [-0.1611, -0.0779]],

                          [[ 0.2326, -1.0558],
                           [ 1.4711,  1.9678]],

                          [[-0.5138, -0.0411],
                           [ 1.9417,  0.5158]],

                          [[ 0.0793,  0.0036],
                           [-0.2569, -0.1055]]]),
           size=(5, 5, 2, 2), nnz=4, layout=torch.sparse_coo)
""",
)

add_docstr_all(
    "inverse",
    r"""
inverse() -> Tensor

See :func:`torch.inverse`
""",
)

add_docstr_all(
    "isnan",
    r"""
isnan() -> Tensor

See :func:`torch.isnan`
""",
)

add_docstr_all(
    "isinf",
    r"""
isinf() -> Tensor

See :func:`torch.isinf`
""",
)

add_docstr_all(
    "isposinf",
    r"""
isposinf() -> Tensor

See :func:`torch.isposinf`
""",
)

add_docstr_all(
    "isneginf",
    r"""
isneginf() -> Tensor

See :func:`torch.isneginf`
""",
)

add_docstr_all(
    "isfinite",
    r"""
isfinite() -> Tensor

See :func:`torch.isfinite`
""",
)

add_docstr_all(
    "isclose",
    r"""
isclose(other, rtol=1e-05, atol=1e-08, equal_nan=False) -> Tensor

See :func:`torch.isclose`
""",
)

add_docstr_all(
    "isreal",
    r"""
isreal() -> Tensor

See :func:`torch.isreal`
""",
)

add_docstr_all(
    "is_coalesced",
    r"""
is_coalesced() -> bool

Returns ``Tru
```



## High-Level Overview

"""Adds docstrings to Tensor functions"""import torch._Cfrom torch._C import _add_docstr as add_docstrfrom torch._torch_docs import parse_kwargs, reproducibility_notesdef add_docstr_all(method: str, docstr: str) -> None:    add_docstr(getattr(torch._C.TensorBase, method), docstr)common_args = parse_kwargs(

This Python file contains 2 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `add_docstr_all`, `scale_channels`, `callable`

**Key imports**: torch._C, _add_docstr as add_docstr, parse_kwargs, reproducibility_notes


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch._C`
- `torch._torch_docs`: parse_kwargs, reproducibility_notes


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`torch`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_classes.py_docs.md`](./_classes.py_docs.md)
- [`types.py_docs.md`](./types.py_docs.md)
- [`_meta_registrations.py_docs.md`](./_meta_registrations.py_docs.md)
- [`_appdirs.py_docs.md`](./_appdirs.py_docs.md)
- [`_tensor.py_docs.md`](./_tensor.py_docs.md)
- [`_streambase.py_docs.md`](./_streambase.py_docs.md)
- [`_lowrank.py_docs.md`](./_lowrank.py_docs.md)
- [`_size_docs.py_docs.md`](./_size_docs.py_docs.md)


## Cross-References

- **File Documentation**: `_tensor_docs.py_docs.md`
- **Keyword Index**: `_tensor_docs.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
