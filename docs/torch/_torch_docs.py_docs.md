# Documentation: _torch_docs.py

## File Metadata
- **Path**: `torch/_torch_docs.py`
- **Size**: 435608 bytes
- **Lines**: 14392
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# mypy: allow-untyped-defs
"""Adds docstrings to functions defined in the torch._C module."""

import re

import torch._C
from torch._C import _add_docstr as add_docstr


def parse_kwargs(desc):
    r"""Map a description of args to a dictionary of {argname: description}.

    Input:
        ('    weight (Tensor): a weight tensor\n' +
         '        Some optional description')
    Output: {
        'weight': \
        'weight (Tensor): a weight tensor\n        Some optional description'
    }
    """
    # Split on exactly 4 spaces after a newline
    regx = re.compile(r"\n\s{4}(?!\s)")
    kwargs = [section.strip() for section in regx.split(desc)]
    kwargs = [section for section in kwargs if len(section) > 0]
    return {desc.split(" ")[0]: desc for desc in kwargs}


def merge_dicts(*dicts):
    """Merge dictionaries into a single dictionary."""
    return {x: d[x] for d in dicts for x in d}


common_args = parse_kwargs(
    """
    input (Tensor): the input tensor.
    generator (:class:`torch.Generator`, optional): a pseudorandom number generator for sampling
    out (Tensor, optional): the output tensor.
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned tensor. Default: ``torch.preserve_format``.
"""
)

reduceops_common_args = merge_dicts(
    common_args,
    parse_kwargs(
        """
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.
"""
    ),
    {
        "opt_keepdim": """
    keepdim (bool, optional): whether the output tensor has :attr:`dim` retained or not. Default: ``False``.
"""
    },
)

multi_dim_common = merge_dicts(
    reduceops_common_args,
    parse_kwargs(
        """
    dim (int or tuple of ints): the dimension or dimensions to reduce.
"""
    ),
    {
        "keepdim_details": """
If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
output tensor having 1 (or ``len(dim)``) fewer dimension(s).
"""
    },
    {
        "opt_dim": """
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
"""
    },
    {
        "opt_dim_all_reduce": """
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
        If ``None``, all dimensions are reduced.
"""
    },
)

single_dim_common = merge_dicts(
    reduceops_common_args,
    parse_kwargs(
        """
    dim (int): the dimension to reduce.
"""
    ),
    {
        "opt_dim": """
    dim (int, optional): the dimension to reduce.
"""
    },
    {
        "opt_dim_all_reduce": """
    dim (int, optional): the dimension to reduce.
        If ``None``, all dimensions are reduced.
"""
    },
    {
        "opt_dim_without_none": """
    dim (int, optional): the dimension to reduce. If omitted, all dimensions are reduced. Explicit ``None`` is not supported.
"""
    },
    {
        "keepdim_details": """If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in
the output tensor having 1 fewer dimension than :attr:`input`."""
    },
)

factory_common_args = merge_dicts(
    common_args,
    parse_kwargs(
        """
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
    layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
        Default: ``torch.strided``.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_device`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    pin_memory (bool, optional): If set, returned tensor would be allocated in
        the pinned memory. Works only for CPU tensors. Default: ``False``.
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.contiguous_format``.
    check_invariants (bool, optional): If sparse tensor invariants are checked.
        Default: as returned by :func:`torch.sparse.check_sparse_tensor_invariants.is_enabled`,
        initially False.
"""
    ),
    {
        "sparse_factory_device_note": """\
.. note::

   If the ``device`` argument is not specified the device of the given
   :attr:`values` and indices tensor(s) must match. If, however, the
   argument is specified the input Tensors will be converted to the
   given device and in turn determine the device of the constructed
   sparse tensor."""
    },
)

factory_like_common_args = parse_kwargs(
    """
    input (Tensor): the size of :attr:`input` will determine size of the output tensor.
    layout (:class:`torch.layout`, optional): the desired layout of returned tensor.
        Default: if ``None``, defaults to the layout of :attr:`input`.
    generator (:class:`torch.Generator`, optional): a pseudorandom number generator for sampling.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned Tensor.
        Default: if ``None``, defaults to the dtype of :attr:`input`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, defaults to the device of :attr:`input`.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    pin_memory (bool, optional): If set, returned tensor would be allocated in
        the pinned memory. Works only for CPU tensors. Default: ``False``.
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.preserve_format``.
"""
)

factory_data_common_args = parse_kwargs(
    """
    data (array_like): Initial data for the tensor. Can be a list, tuple,
        NumPy ``ndarray``, scalar, and other types.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, infers data type from :attr:`data`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_device`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    pin_memory (bool, optional): If set, returned tensor would be allocated in
        the pinned memory. Works only for CPU tensors. Default: ``False``.
"""
)

tf32_notes = {
    "tf32_note": """This operator supports :ref:`TensorFloat32<tf32_on_ampere>`."""
}

rocm_fp16_notes = {
    "rocm_fp16_note": """On certain ROCm devices, when using float16 inputs this module will use \
:ref:`different precision<fp16_on_mi200>` for backward."""
}

reproducibility_notes: dict[str, str] = {
    "forward_reproducibility_note": """This operation may behave nondeterministically when given tensors on \
a CUDA device. See :doc:`/notes/randomness` for more information.""",
    "backward_reproducibility_note": """This operation may produce nondeterministic gradients when given tensors on \
a CUDA device. See :doc:`/notes/randomness` for more information.""",
    "cudnn_reproducibility_note": """In some circumstances when given tensors on a CUDA device \
and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is \
undesirable, you can try to make the operation deterministic (potentially at \
a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. \
See :doc:`/notes/randomness` for more information.""",
}

sparse_support_notes = {
    "sparse_beta_warning": """
.. warning::
    Sparse support is a beta feature and some layout(s)/dtype/device combinations may not be supported,
    or may not have autograd support. If you notice missing functionality please
    open a feature request.""",
}

add_docstr(
    torch.abs,
    r"""
abs(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Computes the absolute value of each element in :attr:`input`.

.. math::
    \text{out}_{i} = |\text{input}_{i}|
"""
    + r"""
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> torch.abs(torch.tensor([-1, -2, 3]))
    tensor([ 1,  2,  3])
""".format(**common_args),
)

add_docstr(
    torch.absolute,
    r"""
absolute(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Alias for :func:`torch.abs`
""",
)

add_docstr(
    torch.acos,
    r"""
acos(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Returns a new tensor with the arccosine (in radians) of each element in :attr:`input`.

.. math::
    \text{out}_{i} = \cos^{-1}(\text{input}_{i})
"""
    + r"""
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.3348, -0.5889,  0.2005, -0.1584])
    >>> torch.acos(a)
    tensor([ 1.2294,  2.2004,  1.3690,  1.7298])
""".format(**common_args),
)

add_docstr(
    torch.arccos,
    r"""
arccos(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Alias for :func:`torch.acos`.
""",
)

add_docstr(
    torch.acosh,
    r"""
acosh(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Returns a new tensor with the inverse hyperbolic cosine of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \cosh^{-1}(\text{input}_{i})

Note:
    The domain of the inverse hyperbolic cosine is `[1, inf)` and values outside this range
    will be mapped to ``NaN``, except for `+ INF` for which the output is mapped to `+ INF`.
"""
    + r"""
Args:
    {input}

Keyword arguments:
    {out}

Example::

    >>> a = torch.randn(4).uniform_(1, 2)
    >>> a
    tensor([ 1.3192, 1.9915, 1.9674, 1.7151 ])
    >>> torch.acosh(a)
    tensor([ 0.7791, 1.3120, 1.2979, 1.1341 ])
""".format(**common_args),
)

add_docstr(
    torch.arccosh,
    r"""
arccosh(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Alias for :func:`torch.acosh`.
""",
)

add_docstr(
    torch.index_add,
    r"""
index_add(input: Tensor, dim: int, index: Tensor, source: Tensor, *, alpha: Union[Number, _complex] = 1, out: Optional[Tensor]) -> Tensor # noqa: B950

See :meth:`~Tensor.index_add_` for function description.
""",
)

add_docstr(
    torch.index_copy,
    r"""
index_copy(input: Tensor, dim: int, index: Tensor, source: Tensor, *, out: Optional[Tensor]) -> Tensor

See :meth:`~Tensor.index_add_` for function description.
""",
)

add_docstr(
    torch.index_reduce,
    r"""
index_reduce(input: Tensor, dim: int, index: Tensor, source: Tensor, reduce: str, *, include_self: bool = True, out: Optional[Tensor]) -> Tensor # noqa: B950

See :meth:`~Tensor.index_reduce_` for function description.
""",
)

add_docstr(
    torch.add,
    r"""
add(input, other, *, alpha=1, out=None) -> Tensor

Adds :attr:`other`, scaled by :attr:`alpha`, to :attr:`input`.

.. math::
    \text{{out}}_i = \text{{input}}_i + \text{{alpha}} \times \text{{other}}_i
"""
    + r"""

Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
:ref:`type promotion <type-promotion-doc>`, and integer, float, and complex inputs.

Args:
    {input}
    other (Tensor or Number): the tensor or number to add to :attr:`input`.

Keyword arguments:
    alpha (Number): the multiplier for :attr:`other`.
    {out}

Examples::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.0202,  1.0985,  1.3506, -0.6056])
    >>> torch.add(a, 20)
    tensor([ 20.0202,  21.0985,  21.3506,  19.3944])

    >>> b = torch.randn(4)
    >>> b
    tensor([-0.9732, -0.3497,  0.6245,  0.4022])
    >>> c = torch.randn(4, 1)
    >>> c
    tensor([[ 0.3743],
            [-1.7724],
            [-0.5811],
            [-0.8017]])
    >>> torch.add(b, c, alpha=10)
    tensor([[  2.7695,   3.3930,   4.3672,   4.1450],
            [-18.6971, -18.0736, -17.0994, -17.3216],
            [ -6.7845,  -6.1610,  -5.1868,  -5.4090],
            [ -8.9902,  -8.3667,  -7.3925,  -7.6147]])
""".format(**common_args),
)

add_docstr(
    torch.addbmm,
    r"""
addbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None) -> Tensor

Performs a batch matrix-matrix product of matrices stored
in :attr:`batch1` and :attr:`batch2`,
with a reduced add step (all matrix multiplications get accumulated
along the first dimension).
:attr:`input` is added to the final result.

:attr:`batch1` and :attr:`batch2` must be 3-D tensors each containing the
same number of matrices.

If :attr:`batch1` is a :math:`(b \times n \times m)` tensor, :attr:`batch2` is a
:math:`(b \times m \times p)` tensor, :attr:`input` must be
:ref:`broadcastable <broadcasting-semantics>` with a :math:`(n \times p)` tensor
and :attr:`out` will be a :math:`(n \times p)` tensor.

.. math::
    out = \beta\ \text{input} + \alpha\ (\sum_{i=0}^{b-1} \text{batch1}_i \mathbin{@} \text{batch2}_i)

If :attr:`beta` is 0, then the content of :attr:`input` will be ignored, and `nan` and `inf` in
it will not be propagated.
"""
    + r"""
For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and :attr:`alpha`
must be real numbers, otherwise they should be integers.

{tf32_note}

{rocm_fp16_note}

Args:
    input (Tensor): matrix to be added
    batch1 (Tensor): the first batch of matrices to be multiplied
    batch2 (Tensor): the second batch of matrices to be multiplied

Keyword args:
    beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
    alpha (Number, optional): multiplier for `batch1 @ batch2` (:math:`\alpha`)
    {out}

Example::

    >>> M = torch.randn(3, 5)
    >>> batch1 = torch.randn(10, 3, 4)
    >>> batch2 = torch.randn(10, 4, 5)
    >>> torch.addbmm(M, batch1, batch2)
    tensor([[  6.6311,   0.0503,   6.9768, -12.0362,  -2.1653],
            [ -4.8185,  -1.4255,  -6.6760,   8.9453,   2.5743],
            [ -3.8202,   4.3691,   1.0943,  -1.1109,   5.4730]])
""".format(**common_args, **tf32_notes, **rocm_fp16_notes),
)

add_docstr(
    torch.addcdiv,
    r"""
addcdiv(input, tensor1, tensor2, *, value=1, out=None) -> Tensor

Performs the element-wise division of :attr:`tensor1` by :attr:`tensor2`,
multiplies the result by the scalar :attr:`value` and adds it to :attr:`input`.

.. warning::
    Integer division with addcdiv is no longer supported, and in a future
    release addcdiv will perform a true division of tensor1 and tensor2.
    The historic addcdiv behavior can be implemented as
    (input + value * torch.trunc(tensor1 / tensor2)).to(input.dtype)
    for integer inputs and as (input + value * tensor1 / tensor2) for float inputs.
    The future addcdiv behavior is just the latter implementation:
    (input + value * tensor1 / tensor2), for all dtypes.

.. math::
    \text{out}_i = \text{input}_i + \text{value} \times \frac{\text{tensor1}_i}{\text{tensor2}_i}
"""
    + r"""

The shapes of :attr:`input`, :attr:`tensor1`, and :attr:`tensor2` must be
:ref:`broadcastable <broadcasting-semantics>`.

For inputs of type `FloatTensor` or `DoubleTensor`, :attr:`value` must be
a real number, otherwise an integer.

Args:
    input (Tensor): the tensor to be added
    tensor1 (Tensor): the numerator tensor
    tensor2 (Tensor): the denominator tensor

Keyword args:
    value (Number, optional): multiplier for :math:`\text{{tensor1}} / \text{{tensor2}}`
    {out}

Example::

    >>> t = torch.randn(1, 3)
    >>> t1 = torch.randn(3, 1)
    >>> t2 = torch.randn(1, 3)
    >>> torch.addcdiv(t, t1, t2, value=0.1)
    tensor([[-0.2312, -3.6496,  0.1312],
            [-1.0428,  3.4292, -0.1030],
            [-0.5369, -0.9829,  0.0430]])
""".format(**common_args),
)

add_docstr(
    torch.addcmul,
    r"""
addcmul(input, tensor1, tensor2, *, value=1, out=None) -> Tensor

Performs the element-wise multiplication of :attr:`tensor1`
by :attr:`tensor2`, multiplies the result by the scalar :attr:`value`
and adds it to :attr:`input`.

.. math::
    \text{out}_i = \text{input}_i + \text{value} \times \text{tensor1}_i \times \text{tensor2}_i
"""
    + r"""
The shapes of :attr:`tensor`, :attr:`tensor1`, and :attr:`tensor2` must be
:ref:`broadcastable <broadcasting-semantics>`.

For inputs of type `FloatTensor` or `DoubleTensor`, :attr:`value` must be
a real number, otherwise an integer.

Args:
    input (Tensor): the tensor to be added
    tensor1 (Tensor): the tensor to be multiplied
    tensor2 (Tensor): the tensor to be multiplied

Keyword args:
    value (Number, optional): multiplier for :math:`tensor1 .* tensor2`
    {out}

Example::

    >>> t = torch.randn(1, 3)
    >>> t1 = torch.randn(3, 1)
    >>> t2 = torch.randn(1, 3)
    >>> torch.addcmul(t, t1, t2, value=0.1)
    tensor([[-0.8635, -0.6391,  1.6174],
            [-0.7617, -0.5879,  1.7388],
            [-0.8353, -0.6249,  1.6511]])
""".format(**common_args),
)

add_docstr(
    torch.addmm,
    r"""
addmm(input, mat1, mat2, out_dtype=None, *, beta=1, alpha=1, out=None) -> Tensor

Performs a matrix multiplication of the matrices :attr:`mat1` and :attr:`mat2`.
The matrix :attr:`input` is added to the final result.

If :attr:`mat1` is a :math:`(n \times m)` tensor, :attr:`mat2` is a
:math:`(m \times p)` tensor, then :attr:`input` must be
:ref:`broadcastable <broadcasting-semantics>` with a :math:`(n \times p)` tensor
and :attr:`out` will be a :math:`(n \times p)` tensor.

:attr:`alpha` and :attr:`beta` are scaling factors on matrix-vector product between
:attr:`mat1` and :attr:`mat2` and the added matrix :attr:`input` respectively.

.. math::
    \text{out} = \beta\ \text{input} + \alpha\ (\text{mat1}_i \mathbin{@} \text{mat2}_i)

If :attr:`beta` is 0, then the content of :attr:`input` will be ignored, and `nan` and `inf` in
it will not be propagated.
"""
    + r"""
For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and
:attr:`alpha` must be real numbers, otherwise they should be integers.

This operation has support for arguments with :ref:`sparse layouts<sparse-docs>`. If
:attr:`input` is sparse the result will have the same layout and if :attr:`out`
is provided it must have the same layout as :attr:`input`.

{sparse_beta_warning}

{tf32_note}

{rocm_fp16_note}

Args:
    input (Tensor): matrix to be added
    mat1 (Tensor): the first matrix to be matrix multiplied
    mat2 (Tensor): the second matrix to be matrix multiplied
    out_dtype (dtype, optional): the dtype of the output tensor,
        Supported only on CUDA and for torch.float32 given
        torch.float16/torch.bfloat16 input dtypes

Keyword args:
    beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
    alpha (Number, optional): multiplier for :math:`mat1 @ mat2` (:math:`\alpha`)
    {out}

Example::

    >>> M = torch.randn(2, 3)
    >>> mat1 = torch.randn(2, 3)
    >>> mat2 = torch.randn(3, 3)
    >>> torch.addmm(M, mat1, mat2)
    tensor([[-4.8716,  1.4671, -1.3746],
            [ 0.7573, -3.9555, -2.8681]])
""".format(**common_args, **tf32_notes, **rocm_fp16_notes, **sparse_support_notes),
)

add_docstr(
    torch.adjoint,
    r"""
adjoint(input: Tensor) -> Tensor
Returns a view of the tensor conjugated and with the last two dimensions transposed.

``x.adjoint()`` is equivalent to ``x.transpose(-2, -1).conj()`` for complex tensors and
to ``x.transpose(-2, -1)`` for real tensors.

Args:
    {input}

Example::

    >>> x = torch.arange(4, dtype=torch.float)
    >>> A = torch.complex(x, x).reshape(2, 2)
    >>> A
    tensor([[0.+0.j, 1.+1.j],
            [2.+2.j, 3.+3.j]])
    >>> A.adjoint()
    tensor([[0.-0.j, 2.-2.j],
            [1.-1.j, 3.-3.j]])
    >>> (A.adjoint() == A.mH).all()
    tensor(True)
""",
)

add_docstr(
    torch.sspaddmm,
    r"""
sspaddmm(input, mat1, mat2, *, beta=1, alpha=1, out=None) -> Tensor

Matrix multiplies a sparse tensor :attr:`mat1` with a dense tensor
:attr:`mat2`, then adds the sparse tensor :attr:`input` to the result.

Note: This function is equivalent to :func:`torch.addmm`, except
:attr:`input` and :attr:`mat1` are sparse.

Args:
    input (Tensor): a sparse matrix to be added
    mat1 (Tensor): a sparse matrix to be matrix multiplied
    mat2 (Tensor): a dense matrix to be matrix multiplied

Keyword args:
    beta (Number, optional): multiplier for :attr:`mat` (:math:`\beta`)
    alpha (Number, optional): multiplier for :math:`mat1 @ mat2` (:math:`\alpha`)
    {out}
""".format(**common_args),
)

add_docstr(
    torch.smm,
    r"""
smm(input, mat) -> Tensor

Performs a matrix multiplication of the sparse matrix :attr:`input`
with the dense matrix :attr:`mat`.

Args:
    input (Tensor): a sparse matrix to be matrix multiplied
    mat (Tensor): a dense matrix to be matrix multiplied
""",
)

add_docstr(
    torch.addmv,
    r"""
addmv(input, mat, vec, *, beta=1, alpha=1, out=None) -> Tensor

Performs a matrix-vector product of the matrix :attr:`mat` and
the vector :attr:`vec`.
The vector :attr:`input` is added to the final result.

If :attr:`mat` is a :math:`(n \times m)` tensor, :attr:`vec` is a 1-D tensor of
size `m`, then :attr:`input` must be
:ref:`broadcastable <broadcasting-semantics>` with a 1-D tensor of size `n` and
:attr:`out` will be 1-D tensor of size `n`.

:attr:`alpha` and :attr:`beta` are scaling factors on matrix-vector product between
:attr:`mat` and :attr:`vec` and the added tensor :attr:`input` respectively.

.. math::
    \text{out} = \beta\ \text{input} + \alpha\ (\text{mat} \mathbin{@} \text{vec})

If :attr:`beta` is 0, then the content of :attr:`input` will be ignored, and `nan` and `inf` in
it will not be propagated.
"""
    + r"""
For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and
:attr:`alpha` must be real numbers, otherwise they should be integers.

Args:
    input (Tensor): vector to be added
    mat (Tensor): matrix to be matrix multiplied
    vec (Tensor): vector to be matrix multiplied

Keyword args:
    beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
    alpha (Number, optional): multiplier for :math:`mat @ vec` (:math:`\alpha`)
    {out}

Example::

    >>> M = torch.randn(2)
    >>> mat = torch.randn(2, 3)
    >>> vec = torch.randn(3)
    >>> torch.addmv(M, mat, vec)
    tensor([-0.3768, -5.5565])
""".format(**common_args),
)

add_docstr(
    torch.addr,
    r"""
addr(input, vec1, vec2, *, beta=1, alpha=1, out=None) -> Tensor

Performs the outer-product of vectors :attr:`vec1` and :attr:`vec2`
and adds it to the matrix :attr:`input`.

Optional values :attr:`beta` and :attr:`alpha` are scaling factors on the
outer product between :attr:`vec1` and :attr:`vec2` and the added matrix
:attr:`input` respectively.

.. math::
    \text{out} = \beta\ \text{input} + \alpha\ (\text{vec1} \otimes \text{vec2})

If :attr:`beta` is 0, then the content of :attr:`input` will be ignored, and `nan` and `inf` in
it will not be propagated.
"""
    + r"""
If :attr:`vec1` is a vector of size `n` and :attr:`vec2` is a vector
of size `m`, then :attr:`input` must be
:ref:`broadcastable <broadcasting-semantics>` with a matrix of size
:math:`(n \times m)` and :attr:`out` will be a matrix of size
:math:`(n \times m)`.

Args:
    input (Tensor): matrix to be added
    vec1 (Tensor): the first vector of the outer product
    vec2 (Tensor): the second vector of the outer product

Keyword args:
    beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
    alpha (Number, optional): multiplier for :math:`\text{{vec1}} \otimes \text{{vec2}}` (:math:`\alpha`)
    {out}

Example::

    >>> vec1 = torch.arange(1., 4.)
    >>> vec2 = torch.arange(1., 3.)
    >>> M = torch.zeros(3, 2)
    >>> torch.addr(M, vec1, vec2)
    tensor([[ 1.,  2.],
            [ 2.,  4.],
            [ 3.,  6.]])
""".format(**common_args),
)

add_docstr(
    torch.allclose,
    r"""
allclose(input: Tensor, other: Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) -> bool

This function checks if :attr:`input` and :attr:`other` satisfy the condition:

.. math::
    \lvert \text{input}_i - \text{other}_i \rvert \leq \texttt{atol} + \texttt{rtol} \times \lvert \text{other}_i \rvert
"""
    + r"""
elementwise, for all elements of :attr:`input` and :attr:`other`. The behaviour of this function is analogous to
`numpy.allclose <https://numpy.org/doc/stable/reference/generated/numpy.allclose.html>`_

Args:
    input (Tensor): first tensor to compare
    other (Tensor): second tensor to compare
    atol (float, optional): absolute tolerance. Default: 1e-08
    rtol (float, optional): relative tolerance. Default: 1e-05
    equal_nan (bool, optional): if ``True``, then two ``NaN`` s will be considered equal. Default: ``False``

Example::

    >>> torch.allclose(torch.tensor([10000., 1e-07]), torch.tensor([10000.1, 1e-08]))
    False
    >>> torch.allclose(torch.tensor([10000., 1e-08]), torch.tensor([10000.1, 1e-09]))
    True
    >>> torch.allclose(torch.tensor([1.0, float('nan')]), torch.tensor([1.0, float('nan')]))
    False
    >>> torch.allclose(torch.tensor([1.0, float('nan')]), torch.tensor([1.0, float('nan')]), equal_nan=True)
    True
""",
)

add_docstr(
    torch.all,
    r"""
all(input: Tensor, *, out=None) -> Tensor

Tests if all elements in :attr:`input` evaluate to `True`.

.. note:: This function matches the behaviour of NumPy in returning
          output of dtype `bool` for all supported dtypes except `uint8`.
          For `uint8` the dtype of output is `uint8` itself.

Args:
    {input}

Keyword args:
    {out}

Example::

    >>> a = torch.rand(1, 2).bool()
    >>> a
    tensor([[False, True]], dtype=torch.bool)
    >>> torch.all(a)
    tensor(False, dtype=torch.bool)
    >>> a = torch.arange(0, 3)
    >>> a
    tensor([0, 1, 2])
    >>> torch.all(a)
    tensor(False)

.. function:: all(input, dim, keepdim=False, *, out=None) -> Tensor
   :noindex:

For each row of :attr:`input` in the given dimension :attr:`dim`,
returns `True` if all elements in the row evaluate to `True` and `False` otherwise.

{keepdim_details}

Args:
    {input}
    {opt_dim_all_reduce}
    {opt_keepdim}

Keyword args:
    {out}

Example::

    >>> a = torch.rand(4, 2).bool()
    >>> a
    tensor([[True, True],
            [True, False],
            [True, True],
            [True, True]], dtype=torch.bool)
    >>> torch.all(a, dim=1)
    tensor([ True, False,  True,  True], dtype=torch.bool)
    >>> torch.all(a, dim=0)
    tensor([ True, False], dtype=torch.bool)
""".format(**multi_dim_common),
)

add_docstr(
    torch.any,
    r"""
any(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Tests if any element in :attr:`input` evaluates to `True`.

.. note:: This function matches the behaviour of NumPy in returning
          output of dtype `bool` for all supported dtypes except `uint8`.
          For `uint8` the dtype of output is `uint8` itself.

Args:
    {input}

Keyword args:
    {out}

Example::

    >>> a = torch.rand(1, 2).bool()
    >>> a
    tensor([[False, True]], dtype=torch.bool)
    >>> torch.any(a)
    tensor(True, dtype=torch.bool)
    >>> a = torch.arange(0, 3)
    >>> a
    tensor([0, 1, 2])
    >>> torch.any(a)
    tensor(True)

.. function:: any(input, dim, keepdim=False, *, out=None) -> Tensor
   :noindex:

For each row of :attr:`input` in the given dimension :attr:`dim`,
returns `True` if any element in the row evaluate to `True` and `False` otherwise.

{keepdim_details}

Args:
    {input}
    {opt_dim_all_reduce}
    {opt_keepdim}

Keyword args:
    {out}

Example::

    >>> a = torch.randn(4, 2) < 0
    >>> a
    tensor([[ True,  True],
            [False,  True],
            [ True,  True],
            [False, False]])
    >>> torch.any(a, 1)
    tensor([ True,  True,  True, False])
    >>> torch.any(a, 0)
    tensor([True, True])
""".format(**multi_dim_common),
)

add_docstr(
    torch.angle,
    r"""
angle(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Computes the element-wise angle (in radians) of the given :attr:`input` tensor.

.. math::
    \text{out}_{i} = angle(\text{input}_{i})
"""
    + r"""
Args:
    {input}

Keyword args:
    {out}

.. note:: Starting in PyTorch 1.8, angle returns pi for negative real numbers,
          zero for non-negative real numbers, and propagates NaNs. Previously
          the function would return zero for all real numbers and not propagate
          floating-point NaNs.

Example::

    >>> torch.angle(torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j]))*180/3.14159
    tensor([ 135.,  135,  -45])
""".format(**common_args),
)

add_docstr(
    torch.as_strided,
    r"""
as_strided(input, size, stride, storage_offset=None) -> Tensor

Create a view of an existing `torch.Tensor` :attr:`input` with specified
:attr:`size`, :attr:`stride` and :attr:`storage_offset`.

.. warning::
    Prefer using other view functions, like :meth:`torch.Tensor.view` or
    :meth:`torch.Tensor.expand`, to setting a view's strides manually with
    `as_strided`, as this function will throw an error on non-standard Pytorch
    backends (that do not have a concept of stride) and the result will depend
    on the current layout in memory. The constructed view must only refer to
    elements within the Tensor's storage or a runtime error will be thrown.
    If the generated view is "overlapped" (with multiple indices referring to
    the same element in memory), the behavior of inplace operations on this view
    is undefined (and might not throw runtime errors).

Args:
    {input}
    size (tuple or ints): the shape of the output tensor
    stride (tuple or ints): the stride of the output tensor
    storage_offset (int, optional): the offset in the underlying storage of the output tensor.
        If ``None``, the storage_offset of the output tensor will match the input tensor.

Example::

    >>> x = torch.randn(3, 3)
    >>> x
    tensor([[ 0.9039,  0.6291,  1.0795],
            [ 0.1586,  2.1939, -0.4900],
            [-0.1909, -0.7503,  1.9355]])
    >>> t = torch.as_strided(x, (2, 2), (1, 2))
    >>> t
    tensor([[0.9039, 1.0795],
            [0.6291, 0.1586]])
    >>> t = torch.as_strided(x, (2, 2), (1, 2), 1)
    tensor([[0.6291, 0.1586],
            [1.0795, 2.1939]])
""".format(**common_args),
)

add_docstr(
    torch.as_tensor,
    r"""
as_tensor(data: Any, dtype: Optional[dtype] = None, device: Optional[DeviceLikeType]) -> Tensor

Converts :attr:`data` into a tensor, sharing data and preserving autograd
history if possible.

If :attr:`data` is already a tensor with the requested dtype and device
then :attr:`data` itself is returned, but if :attr:`data` is a
tensor with a different dtype or device then it's copied as if using
`data.to(dtype=dtype, device=device)`.

If :attr:`data` is a NumPy array (an ndarray) with the same dtype and device then a
tensor is constructed using :func:`torch.from_numpy`.

If :attr:`data` is a CuPy array, the returned tensor will be located on the same device as the CuPy array unless
specifically overwritten by :attr:`device` or a default device. The device of the CuPy array is inferred from the
pointer of the array using `cudaPointerGetAttributes` unless :attr:`device` is provided with an explicit device index.

.. seealso::

    :func:`torch.tensor` never shares its data and creates a new "leaf tensor" (see :doc:`/notes/autograd`).


Args:
    {data}
    {dtype}
    device (:class:`torch.device`, optional): the device of the constructed tensor. If None and data is a tensor
        then the device of data is used. If None and data is not a tensor then
        the result tensor is constructed on the current device.


Example::

    >>> a = numpy.array([1, 2, 3])
    >>> t = torch.as_tensor(a)
    >>> t
    tensor([ 1,  2,  3])
    >>> t[0] = -1
    >>> a
    array([-1,  2,  3])

    >>> a = numpy.array([1, 2, 3])
    >>> t = torch.as_tensor(a, device=torch.device('cuda'))
    >>> t
    tensor([ 1,  2,  3])
    >>> t[0] = -1
    >>> a
    array([1,  2,  3])
""".format(**factory_data_common_args),
)

add_docstr(
    torch.asin,
    r"""
asin(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Returns a new tensor with the arcsine of the elements (in radians) in the :attr:`input` tensor.

.. math::
    \text{out}_{i} = \sin^{-1}(\text{input}_{i})
"""
    + r"""
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.5962,  1.4985, -0.4396,  1.4525])
    >>> torch.asin(a)
    tensor([-0.6387,     nan, -0.4552,     nan])
""".format(**common_args),
)

add_docstr(
    torch.arcsin,
    r"""
arcsin(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Alias for :func:`torch.asin`.
""",
)

add_docstr(
    torch.asinh,
    r"""
asinh(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Returns a new tensor with the inverse hyperbolic sine of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \sinh^{-1}(\text{input}_{i})
"""
    + r"""
Args:
    {input}

Keyword arguments:
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.1606, -1.4267, -1.0899, -1.0250 ])
    >>> torch.asinh(a)
    tensor([ 0.1599, -1.1534, -0.9435, -0.8990 ])
""".format(**common_args),
)

add_docstr(
    torch.arcsinh,
    r"""
arcsinh(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Alias for :func:`torch.asinh`.
""",
)

add_docstr(
    torch.atan,
    r"""
atan(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Returns a new tensor with the arctangent of the elements (in radians) in the :attr:`input` tensor.

.. math::
    \text{out}_{i} = \tan^{-1}(\text{input}_{i})
"""
    + r"""
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.2341,  0.2539, -0.6256, -0.6448])
    >>> torch.atan(a)
    tensor([ 0.2299,  0.2487, -0.5591, -0.5727])
""".format(**common_args),
)

add_docstr(
    torch.arctan,
    r"""
arctan(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Alias for :func:`torch.atan`.
""",
)

add_docstr(
    torch.atan2,
    r"""
atan2(input: Tensor, other: Tensor, *, out: Optional[Tensor]) -> Tensor

Element-wise arctangent of :math:`\text{{input}}_{{i}} / \text{{other}}_{{i}}`
with consideration of the quadrant. Returns a new tensor with the signed angles
in radians between vector :math:`(\text{{other}}_{{i}}, \text{{input}}_{{i}})`
and vector :math:`(1, 0)`. (Note that :math:`\text{{other}}_{{i}}`, the second
parameter, is the x-coordinate, while :math:`\text{{input}}_{{i}}`, the first
parameter, is the y-coordinate.)

The shapes of ``input`` and ``other`` must be
:ref:`broadcastable <broadcasting-semantics>`.

Args:
    input (Tensor): the first input tensor
    other (Tensor): the second input tensor

Keyword args:
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.9041,  0.0196, -0.3108, -2.4423])
    >>> torch.atan2(a, torch.randn(4))
    tensor([ 0.9833,  0.0811, -1.9743, -1.4151])
""".format(**common_args),
)

add_docstr(
    torch.arctan2,
    r"""
arctan2(input: Tensor, other: Tensor, *, out: Optional[Tensor]) -> Tensor
Alias for :func:`torch.atan2`.
""",
)

add_docstr(
    torch.atanh,
    r"""
atanh(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Returns a new tensor with the inverse hyperbolic tangent of the elements of :attr:`input`.

Note:
    The domain of the inverse hyperbolic tangent is `(-1, 1)` and values outside this range
    will be mapped to ``NaN``, except for the values `1` and `-1` for which the output is
    mapped to `+/-INF` respectively.

.. math::
    \text{out}_{i} = \tanh^{-1}(\text{input}_{i})
"""
    + r"""
Args:
    {input}

Keyword arguments:
    {out}

Example::

    >>> a = torch.randn(4).uniform_(-1, 1)
    >>> a
    tensor([ -0.9385, 0.2968, -0.8591, -0.1871 ])
    >>> torch.atanh(a)
    tensor([ -1.7253, 0.3060, -1.2899, -0.1893 ])
""".format(**common_args),
)

add_docstr(
    torch.arctanh,
    r"""
arctanh(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Alias for :func:`torch.atanh`.
""",
)

add_docstr(
    torch.asarray,
    r"""
asarray(obj: Any, *, dtype: Optional[dtype], device: Optional[DeviceLikeType], copy: Optional[bool] = None, requires_grad: bool = False) -> Tensor # noqa: B950

Converts :attr:`obj` to a tensor.

:attr:`obj` can be one of:

1. a tensor
2. a NumPy array or a NumPy scalar
3. a DLPack capsule
4. an object that implements Python's buffer protocol
5. a scalar
6. a sequence of scalars

When :attr:`obj` is a tensor, NumPy array, or DLPack capsule the returned tensor will,
by default, not require a gradient, have the same datatype as :attr:`obj`, be on the
same device, and share memory with it. These properties can be controlled with the
:attr:`dtype`, :attr:`device`, :attr:`copy`, and :attr:`requires_grad` keyword arguments.
If the returned tensor is of a different datatype, on a different device, or a copy is
requested then it will not share its memory with :attr:`obj`. If :attr:`requires_grad`
is ``True`` then the returned tensor will require a gradient, and if :attr:`obj` is
also a tensor with an autograd history then the returned tensor will have the same history.

When :attr:`obj` is not a tensor, NumPy array, or DLPack capsule but implements Python's
buffer protocol then the buffer is interpreted as an array of bytes grouped according to
the size of the datatype passed to the :attr:`dtype` keyword argument. (If no datatype is
passed then the default floating point datatype is used, instead.) The returned tensor
will have the specified datatype (or default floating point datatype if none is specified)
and, by default, be on the CPU device and share memory with the buffer.

When :attr:`obj` is a NumPy scalar, the returned tensor will be a 0-dimensional tensor on
the CPU and that doesn't share its memory (i.e. ``copy=True``). By default datatype will
be the PyTorch datatype corresponding to the NumPy's scalar's datatype.

When :attr:`obj` is none of the above but a scalar, or a sequence of scalars then the
returned tensor will, by default, infer its datatype from the scalar values, be on the
current default device, and not share its memory.

.. seealso::

    :func:`torch.tensor` creates a tensor that always copies the data from the input object.
    :func:`torch.from_numpy` creates a tensor that always shares memory from NumPy arrays.
    :func:`torch.frombuffer` creates a tensor that always shares memory from objects that
    implement the buffer protocol.
    :func:`torch.from_dlpack` creates a tensor that always shares memory from
    DLPack capsules.

Args:
    obj (object): a tensor, NumPy array, DLPack Capsule, object that implements Python's
           buffer protocol, scalar, or sequence of scalars.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the datatype of the returned tensor.
           Default: ``None``, which causes the datatype of the returned tensor to be
           inferred from :attr:`obj`.
    copy (bool, optional): controls whether the returned tensor shares memory with :attr:`obj`.
           Default: ``None``, which causes the returned tensor to share memory with :attr:`obj`
           whenever possible. If ``True`` then the returned tensor does not share its memory.
           If ``False`` then the returned tensor shares its memory with :attr:`obj` and an
           error is thrown if it cannot.
    device (:class:`torch.device`, optional): the device of the returned tensor.
           Default: ``None``, which causes the device of :attr:`obj` to be used. Or, if
           :attr:`obj` is a Python sequence, the current default device will be used.
    requires_grad (bool, optional): whether the returned tensor requires grad.
           Default: ``False``, which causes the returned tensor not to require a gradient.
           If ``True``, then the returned tensor will require a gradient, and if :attr:`obj`
           is also a tensor with an autograd history then the returned tensor will have
           the same history.

Example::

    >>> a = torch.tensor([1, 2, 3])
    >>> # Shares memory with tensor 'a'
    >>> b = torch.asarray(a)
    >>> a.data_ptr() == b.data_ptr()
    True
    >>> # Forces memory copy
    >>> c = torch.asarray(a, copy=True)
    >>> a.data_ptr() == c.data_ptr()
    False

    >>> a = torch.tensor([1., 2., 3.], requires_grad=True)
    >>> b = a + 2
    >>> b
    tensor([3., 4., 5.], grad_fn=<AddBackward0>)
    >>> # Shares memory with tensor 'b', with no grad
    >>> c = torch.asarray(b)
    >>> c
    tensor([3., 4., 5.])
    >>> # Shares memory with tensor 'b', retaining autograd history
    >>> d = torch.asarray(b, requires_grad=True)
    >>> d
    tensor([3., 4., 5.], grad_fn=<AddBackward0>)

    >>> array = numpy.array([1, 2, 3])
    >>> # Shares memory with array 'array'
    >>> t1 = torch.asarray(array)
    >>> array.__array_interface__['data'][0] == t1.data_ptr()
    True
    >>> # Copies memory due to dtype mismatch
    >>> t2 = torch.asarray(array, dtype=torch.float32)
    >>> array.__array_interface__['data'][0] == t2.data_ptr()
    False

    >>> scalar = numpy.float64(0.5)
    >>> torch.asarray(scalar)
    tensor(0.5000, dtype=torch.float64)
""",
)

add_docstr(
    torch.baddbmm,
    r"""
baddbmm(input, batch1, batch2, out_dtype=None, *, beta=1, alpha=1, out=None) -> Tensor

Performs a batch matrix-matrix product of matrices in :attr:`batch1`
and :attr:`batch2`.
:attr:`input` is added to the final result.

:attr:`batch1` and :attr:`batch2` must be 3-D tensors each containing the same
number of matrices.

If :attr:`batch1` is a :math:`(b \times n \times m)` tensor, :attr:`batch2` is a
:math:`(b \times m \times p)` tensor, then :attr:`input` must be
:ref:`broadcastable <broadcasting-semantics>` with a
:math:`(b \times n \times p)` tensor and :attr:`out` will be a
:math:`(b \times n \times p)` tensor. Both :attr:`alpha` and :attr:`beta` mean the
same as the scaling factors used in :meth:`torch.addbmm`.

.. math::
    \text{out}_i = \beta\ \text{input}_i + \alpha\ (\text{batch1}_i \mathbin{@} \text{batch2}_i)

If :attr:`beta` is 0, then the content of :attr:`input` will be ignored, and `nan` and `inf` in
it will not be propagated.
"""
    + r"""
For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and
:attr:`alpha` must be real numbers, otherwise they should be integers.

{tf32_note}

{rocm_fp16_note}

Args:
    input (Tensor): the tensor to be added
    batch1 (Tensor): the first batch of matrices to be multiplied
    batch2 (Tensor): the second batch of matrices to be multiplied
    out_dtype (dtype, optional): the dtype of the output tensor,
        Supported only on CUDA and for torch.float32 given
        torch.float16/torch.bfloat16 input dtypes

Keyword args:
    beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
    alpha (Number, optional): multiplier for :math:`\text{{batch1}} \mathbin{{@}} \text{{batch2}}` (:math:`\alpha`)
    {out}

Example::

    >>> M = torch.randn(10, 3, 5)
    >>> batch1 = torch.randn(10, 3, 4)
    >>> batch2 = torch.randn(10, 4, 5)
    >>> torch.baddbmm(M, batch1, batch2).size()
    torch.Size([10, 3, 5])
""".format(**common_args, **tf32_notes, **rocm_fp16_notes),
)

add_docstr(
    torch.bernoulli,
    r"""
bernoulli(input: Tensor, *, generator: Optional[Generator], out: Optional[Tensor]) -> Tensor

Draws binary random numbers (0 or 1) from a Bernoulli distribution.

The :attr:`input` tensor should be a tensor containing probabilities
to be used for drawing the binary random number.
Hence, all values in :attr:`input` have to be in the range:
:math:`0 \leq \text{input}_i \leq 1`.

The :math:`\text{i}^{th}` element of the output tensor will draw a
value :math:`1` according to the :math:`\text{i}^{th}` probability value given
in :attr:`input`.

.. math::
    \text{out}_{i} \sim \mathrm{Bernoulli}(p = \text{input}_{i})
"""
    + r"""
The returned :attr:`out` tensor only has values 0 or 1 and is of the same
shape as :attr:`input`.

:attr:`out` can have integral ``dtype``, but :attr:`input` must have floating
point ``dtype``.

Args:
    input (Tensor): the input tensor of probability values for the Bernoulli distribution

Keyword args:
    {generator}
    {out}

Example::

    >>> a = torch.empty(3, 3).uniform_(0, 1)  # generate a uniform random matrix with range [0, 1]
    >>> a
    tensor([[ 0.1737,  0.0950,  0.3609],
            [ 0.7148,  0.0289,  0.2676],
            [ 0.9456,  0.8937,  0.7202]])
    >>> torch.bernoulli(a)
    tensor([[ 1.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 1.,  1.,  1.]])

    >>> a = torch.ones(3, 3) # probability of drawing "1" is 1
    >>> torch.bernoulli(a)
    tensor([[ 1.,  1.,  1.],
            [ 1.,  1.,  1.],
            [ 1.,  1.,  1.]])
    >>> a = torch.zeros(3, 3) # probability of drawing "1" is 0
    >>> torch.bernoulli(a)
    tensor([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.]])
""".format(**common_args),
)

add_docstr(
    torch.bincount,
    r"""
bincount(input, weights=None, minlength=0) -> Tensor

Count the frequency of each value in an array of non-negative ints.

The number of bins (size 1) is one larger than the largest value in
:attr:`input` unless :attr:`input` is empty, in which case the result is a
tensor of size 0. If :attr:`minlength` is specified, the number of bins is at least
:attr:`minlength` and if :attr:`input` is empty, then the result is tensor of size
:attr:`minlength` filled with zeros. If ``n`` is the value at position ``i``,
``out[n] += weights[i]`` if :attr:`weights` is specified else
``out[n] += 1``.

Note:
    {backward_reproducibility_note}

Arguments:
    input (Tensor): 1-d int tensor
    weights (Tensor): optional, weight for each value in the input tensor.
        Should be of same size as input tensor.
    minlength (int): optional, minimum number of bins. Should be non-negative.

Returns:
    output (Tensor): a tensor of shape ``Size([max(input) + 1])`` if
    :attr:`input` is non-empty, else ``Size(0)``

Example::

    >>> input = torch.randint(0, 8, (5,), dtype=torch.int64)
    >>> weights = torch.linspace(0, 1, steps=5)
    >>> input, weights
    (tensor([4, 3, 6, 3, 4]),
     tensor([ 0.0000,  0.2500,  0.5000,  0.7500,  1.0000])

    >>> torch.bincount(input)
    tensor([0, 0, 0, 2, 2, 0, 1])

    >>> input.bincount(weights)
    tensor([0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 0.5000])
""".format(**reproducibility_notes),
)

add_docstr(
    torch.bitwise_not,
    r"""
bitwise_not(input, *, out=None) -> Tensor

Computes the bitwise NOT of the given input tensor. The input tensor must be of
integral or Boolean types. For bool tensors, it computes the logical NOT.

Args:
    {input}

Keyword args:
    {out}

Example::

    >>> torch.bitwise_not(torch.tensor([-1, -2, 3], dtype=torch.int8))
    tensor([ 0,  1, -4], dtype=torch.int8)
""".format(**common_args),
)

add_docstr(
    torch.bmm,
    r"""
bmm(input, mat2, out_dtype=None, *, out=None) -> Tensor

Performs a batch matrix-matrix product of matrices stored in :attr:`input`
and :attr:`mat2`.

:attr:`input` and :attr:`mat2` must be 3-D tensors each containing
the same number of matrices.

If :attr:`input` is a :math:`(b \times n \times m)` tensor, :attr:`mat2` is a
:math:`(b \times m \times p)` tensor, :attr:`out` will be a
:math:`(b \times n \times p)` tensor.

.. math::
    \text{out}_i = \text{input}_i \mathbin{@} \text{mat2}_i
"""
    + r"""
{tf32_note}

{rocm_fp16_note}

.. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.
          For broadcasting matrix products, see :func:`torch.matmul`.

Args:
    input (Tensor): the first batch of matrices to be multiplied
    mat2 (Tensor): the second batch of matrices to be multiplied
    out_dtype (dtype, optional): the dtype of the output tensor,
        Supported only on CUDA and for torch.float32 given
        torch.float16/torch.bfloat16 input dtypes

Keyword Args:
    {out}

Example::

    >>> input = torch.randn(10, 3, 4)
    >>> mat2 = torch.randn(10, 4, 5)
    >>> res = torch.bmm(input, mat2)
    >>> res.size()
    torch.Size([10, 3, 5])
""".format(**common_args, **tf32_notes, **rocm_fp16_notes),
)

add_docstr(
    torch.bitwise_and,
    r"""
bitwise_and(input, other, *, out=None) -> Tensor

Computes the bitwise AND of :attr:`input` and :attr:`other`. The input tensor must be of
integral or Boolean types. For bool tensors, it computes the logical AND.

Args:
    input: the first input tensor
    other: the second input tensor

Keyword args:
    {out}

Example::

    >>> torch.bitwise_and(torch.tensor([-1, -2, 3], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))
    tensor([1, 0,  3], dtype=torch.int8)
    >>> torch.bitwise_and(torch.tensor([True, True, False]), torch.tensor([False, True, False]))
    tensor([ False, True, False])
""".format(**common_args),
)

add_docstr(
    torch.bitwise_or,
    r"""
bitwise_or(input: Tensor, other: Tensor, *, out: Optional[Tensor]) -> Tensor

Computes the bitwise OR of :attr:`input` and :attr:`other`. The input tensor must be of
integral or Boolean types. For bool tensors, it computes the logical OR.

Args:
    input: the first input tensor
    other: the second input tensor

Keyword args:
    {out}

Example::

    >>> torch.bitwise_or(torch.tensor([-1, -2, 3], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))
    tensor([-1, -2,  3], dtype=torch.int8)
    >>> torch.bitwise_or(torch.tensor([True, True, False]), torch.tensor([False, True, False]))
    tensor([ True, True, False])
""".format(**common_args),
)

add_docstr(
    torch.bitwise_xor,
    r"""
bitwise_xor(input, other, *, out=None) -> Tensor

Computes the bitwise XOR of :attr:`input` and :attr:`other`. The input tensor must be of
integral or Boolean types. For bool tensors, it computes the logical XOR.

Args:
    input: the first input tensor
    other: the second input tensor

Keyword args:
    {out}

Example::

    >>> torch.bitwise_xor(torch.tensor([-1, -2, 3], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))
    tensor([-2, -2,  0], dtype=torch.int8)
    >>> torch.bitwise_xor(torch.tensor([True, True, False]), torch.tensor([False, True, False]))
    tensor([ True, False, False])
""".format(**common_args),
)

add_docstr(
    torch.bitwise_left_shift,
    r"""
bitwise_left_shift(input, other, *, out=None) -> Tensor

Computes the left arithmetic shift of :attr:`input` by :attr:`other` bits.
The input tensor must be of integral type. This operator supports
:ref:`broadcasting to a common shape <broadcasting-semantics>` and
:ref:`type promotion <type-promotion-doc>`.

The operation applied is:

.. math::
    \text{{out}}_i = \text{{input}}_i << \text{{other}}_i

Args:
    input (Tensor or Scalar): the first input tensor
    other (Tensor or Scalar): the second input tensor

Keyword args:
    {out}

Example::

    >>> torch.bitwise_left_shift(torch.tensor([-1, -2, 3], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))
    tensor([-2, -2, 24], dtype=torch.int8)
""".format(**common_args),
)

add_docstr(
    torch.bitwise_right_shift,
    r"""
bitwise_right_shift(input, other, *, out=None) -> Tensor

Computes the right arithmetic shift of :attr:`input` by :attr:`other` bits.
The input tensor must be of integral type. This operator supports
:ref:`broadcasting to a common shape <broadcasting-semantics>` and
:ref:`type promotion <type-promotion-doc>`.
In any case, if the value of the right operand is negative or is greater
or equal to the number of bits in the promoted left operand, the behavior is undefined.

The operation applied is:

.. math::
    \text{{out}}_i = \text{{input}}_i >> \text{{other}}_i

Args:
    input (Tensor or Scalar): the first input tensor
    other (Tensor or Scalar): the second input tensor

Keyword args:
    {out}

Example::

    >>> torch.bitwise_right_shift(torch.tensor([-2, -7, 31], dtype=torch.int8), torch.tensor([1, 0, 3], dtype=torch.int8))
    tensor([-1, -7,  3], dtype=torch.int8)
""".format(**common_args),
)

add_docstr(
    torch.broadcast_to,
    r"""
broadcast_to(input, shape) -> Tensor

Broadcasts :attr:`input` to the shape :attr:`\shape`.
Equivalent to calling ``input.expand(shape)``. See :meth:`~Tensor.expand` for details.

Args:
    {input}
    shape (list, tuple, or :class:`torch.Size`): the new shape.

Example::

    >>> x = torch.tensor([1, 2, 3])
    >>> torch.broadcast_to(x, (3, 3))
    tensor([[1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]])
""".format(**common_args),
)

add_docstr(
    torch.stack,
    r"""
stack(tensors, dim=0, *, out=None) -> Tensor

Concatenates a sequence of tensors along a new dimension.

All tensors need to be of the same size.

.. seealso::

    :func:`torch.cat` concatenates the given sequence along an existing dimension.

Arguments:
    tensors (sequence of Tensors): sequence of tensors to concatenate
    dim (int, optional): dimension to insert. Has to be between 0 and the number
        of dimensions of concatenated tensors (inclusive). Default: 0

Keyword args:
    {out}

Example::

    >>> x = torch.randn(2, 3)
    >>> x
    tensor([[ 0.3367,  0.1288,  0.2345],
            [ 0.2303, -1.1229, -0.1863]])
    >>> torch.stack((x, x)) # same as torch.stack((x, x), dim=0)
    tensor([[[ 0.3367,  0.1288,  0.2345],
             [ 0.2303, -1.1229, -0.1863]],

            [[ 0.3367,  0.1288,  0.2345],
             [ 0.2303, -1.1229, -0.1863]]])
    >>> torch.stack((x, x)).size()
    torch.Size([2, 2, 3])
    >>> torch.stack((x, x), dim=1)
    tensor([[[ 0.3367,  0.1288,  0.2345],
             [ 0.3367,  0.1288,  0.2345]],

            [[ 0.2303, -1.1229, -0.1863],
             [ 0.2303, -1.1229, -0.1863]]])
    >>> torch.stack((x, x), dim=2)
    tensor([[[ 0.3367,  0.3367],
             [ 0.1288,  0.1288],
             [ 0.2345,  0.2345]],

            [[ 0.2303,  0.2303],
             [-1.1229, -1.1229],
             [-0.1863, -0.1863]]])
    >>> torch.stack((x, x), dim=-1)
    tensor([[[ 0.3367,  0.3367],
             [ 0.1288,  0.1288],
             [ 0.2345,  0.2345]],

            [[ 0.2303,  0.2303],
             [-1.1229, -1.1229],
             [-0.1863, -0.1863]]])
""".format(**common_args),
)

add_docstr(
    torch.hstack,
    r"""
hstack(tensors, *, out=None) -> Tensor

Stack tensors in sequence horizontally (column wise).

This is equivalent to concatenation along the first axis for 1-D tensors, and along the second axis for all other tensors.

Args:
    tensors (sequence of Tensors): sequence of tensors to concatenate

Keyword args:
    {out}

Example::

    >>> a = torch.tensor([1, 2, 3])
    >>> b = torch.tensor([4, 5, 6])
    >>> torch.hstack((a,b))
    tensor([1, 2, 3, 4, 5, 6])
    >>> a = torch.tensor([[1],[2],[3]])
    >>> b = torch.tensor([[4],[5],[6]])
    >>> torch.hstack((a,b))
    tensor([[1, 4],
            [2, 5],
            [3, 6]])

""".format(**common_args),
)

add_docstr(
    torch.vstack,
    r"""
vstack(tensors, *, out=None) -> Tensor

Stack tensors in sequence vertically (row wise).

This is equivalent to concatenation along the first axis after all 1-D tensors have been reshaped by :func:`torch.atleast_2d`.

Args:
    tensors (sequence of Tensors): sequence of tensors to concatenate

Keyword args:
    {out}

Example::

    >>> a = torch.tensor([1, 2, 3])
    >>> b = torch.tensor([4, 5, 6])
    >>> torch.vstack((a,b))
    tensor([[1, 2, 3],
            [4, 5, 6]])
    >>> a = torch.tensor([[1],[2],[3]])
    >>> b = torch.tensor([[4],[5],[6]])
    >>> torch.vstack((a,b))
    tensor([[1],
            [2],
            [3],
            [4],
            [5],
            [6]])


""".format(**common_args),
)

add_docstr(
    torch.dstack,
    r"""
dstack(tensors, *, out=None) -> Tensor

Stack tensors in sequence depthwise (along third axis).

This is equivalent to concatenation along the third axis after 1-D and 2-D tensors have been reshaped by :func:`torch.atleast_3d`.

Args:
    tensors (sequence of Tensors): sequence of tensors to concatenate

Keyword args:
    {out}

Example::

    >>> a = torch.tensor([1, 2, 3])
    >>> b = torch.tensor([4, 5, 6])
    >>> torch.dstack((a,b))
    tensor([[[1, 4],
             [2, 5],
             [3, 6]]])
    >>> a = torch.tensor([[1],[2],[3]])
    >>> b = torch.tensor([[4],[5],[6]])
    >>> torch.dstack((a,b))
    tensor([[[1, 4]],
            [[2, 5]],
            [[3, 6]]])


""".format(**common_args),
)

add_docstr(
    torch.tensor_split,
    r"""
tensor_split(input, indices_or_sections, dim=0) -> List of Tensors

Splits a tensor into multiple sub-tensors, all of which are views of :attr:`input`,
along dimension :attr:`dim` according to the indices or number of sections specified
by :attr:`indices_or_sections`. This function is based on NumPy's
:func:`numpy.array_split`.

Args:
    input (Tensor): the tensor to split
    indices_or_sections (Tensor, int or list or tuple of ints):
        If :attr:`indices_or_sections` is an integer ``n`` or a zero dimensional long tensor
        with value ``n``, :attr:`input` is split into ``n`` sections along dimension :attr:`dim`.
        If :attr:`input` is divisible by ``n`` along dimension :attr:`dim`, each
        section will be of equal size, :code:`input.size(dim) / n`. If :attr:`input`
        is not divisible by ``n``, the sizes of the first :code:`int(input.size(dim) % n)`
        sections will have size :code:`int(input.size(dim) / n) + 1`, and the rest will
        have size :code:`int(input.size(dim) / n)`.

        If :attr:`indices_or_sections` is a list or tuple of ints, or a one-dimensional long
        tensor, then :attr:`input` is split along dimension :attr:`dim` at each of the indices
        in the list, tuple or tensor. For instance, :code:`indices_or_sections=[2, 3]` and :code:`dim=0`
        would result in the tensors :code:`input[:2]`, :code:`input[2:3]`, and :code:`input[3:]`.

        If :attr:`indices_or_sections` is a tensor, it must be a zero-dimensional or one-dimensional
        long tensor on the CPU.

    dim (int, optional): dimension along which to split the tensor. Default: ``0``

Example::

    >>> x = torch.arange(8)
    >>> torch.tensor_split(x, 3)
    (tensor([0, 1, 2]), tensor([3, 4, 5]), tensor([6, 7]))

    >>> x = torch.arange(7)
    >>> torch.tensor_split(x, 3)
    (tensor([0, 1, 2]), tensor([3, 4]), tensor([5, 6]))
    >>> torch.tensor_split(x, (1, 6))
    (tensor([0]), tensor([1, 2, 3, 4, 5]), tensor([6]))

    >>> x = torch.arange(14).reshape(2, 7)
    >>> x
    tensor([[ 0,  1,  2,  3,  4,  5,  6],
            [ 7,  8,  9, 10, 11, 12, 13]])
    >>> torch.tensor_split(x, 3, dim=1)
    (tensor([[0, 1, 2],
            [7, 8, 9]]),
     tensor([[ 3,  4],
            [10, 11]]),
     tensor([[ 5,  6],
            [12, 13]]))
    >>> torch.tensor_split(x, (1, 6), dim=1)
    (tensor([[0],
            [7]]),
     tensor([[ 1,  2,  3,  4,  5],
            [ 8,  9, 10, 11, 12]]),
     tensor([[ 6],
            [13]]))
""",
)

add_docstr(
    torch.chunk,
    r"""
chunk(input: Tensor, chunks: int, dim: int = 0) -> Tuple[Tensor, ...]

Attempts to split a tensor into the specified number of chunks. Each chunk is a view of
the input tensor.


.. note::

    This function may return fewer than the specified number of chunks!

.. seealso::

    :func:`torch.tensor_split` a function that always returns exactly the specified number of chunks

If the tensor size along the given dimension :attr:`dim` is divisible by :attr:`chunks`,
all returned chunks will be the same size.
If the tensor size along the given dimension :attr:`dim` is not divisible by :attr:`chunks`,
all returned chunks will be the same size, except the last one.
If such division is not possible, this function may return fewer
than the specified number of chunks.

Arguments:
    input (Tensor): the tensor to split
    chunks (int): number of chunks to return
    dim (int): dimension along which to split the tensor

Example:
    >>> torch.arange(11).chunk(6)
    (tensor([0, 1]),
     tensor([2, 3]),
     tensor([4, 5]),
     tensor([6, 7]),
     tensor([8, 9]),
     tensor([10]))
    >>> torch.arange(12).chunk(6)
    (tensor([0, 1]),
     tensor([2, 3]),
     tensor([4, 5]),
     tensor([6, 7]),
     tensor([8, 9]),
     tensor([10, 11]))
    >>> torch.arange(13).chunk(6)
    (tensor([0, 1, 2]),
     tensor([3, 4, 5]),
     tensor([6, 7, 8]),
     tensor([ 9, 10, 11]),
     tensor([12]))
""",
)

add_docstr(
    torch.unsafe_chunk,
    r"""
unsafe_chunk(input, chunks, dim=0) -> List of Tensors

Works like :func:`torch.chunk` but without enforcing the autograd restrictions
on inplace modification of the outputs.

.. warning::
    This function is safe to use as long as only the input, or only the outputs
    are modified inplace after calling this function. It is user's
    responsibility to ensure that is the case. If both the input and one or more
    of the outputs are modified inplace, gradients computed by autograd will be
    silently incorrect.
""",
)

add_docstr(
    torch.unsafe_split,
    r"""
unsafe_split(tensor, split_size_or_sections, dim=0) -> List of Tensors

Works like :func:`torch.split` but without enforcing the autograd restrictions
on inplace modification of the outputs.

.. warning::
    This function is safe to use as long as only the input, or only the outputs
    are modified inplace after calling this function. It is user's
    responsibility to ensure that is the case. If both the input and one or more
    of the outputs are modified inplace, gradients computed by autograd will be
    silently incorrect.
""",
)

add_docstr(
    torch.hsplit,
    r"""
hsplit(input, indices_or_sections) -> List of Tensors

Splits :attr:`input`, a tensor with one or more dimensions, into multiple tensors
horizontally according to :attr:`indices_or_sections`. Each split is a view of
:attr:`input`.

If :attr:`input` is one dimensional this is equivalent to calling
torch.tensor_split(input, indices_or_sections, dim=0) (the split dimension is
zero), and if :attr:`input` has two or more dimensions it's equivalent to calling
torch.tensor_split(input, indices_or_sections, dim=1) (the split dimension is 1),
except that if :attr:`indices_or_sections` is an integer it must evenly divide
the split dimension or a runtime error will be thrown.

This function is based on NumPy's :func:`numpy.hsplit`.

Args:
    input (Tensor): tensor to split.
    indices_or_sections (int or list or tuple of ints): See argument in :func:`torch.tensor_split`.

Example::

    >>> t = torch.arange(16.0).reshape(4,4)
    >>> t
    tensor([[ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.],
            [12., 13., 14., 15.]])
    >>> torch.hsplit(t, 2)
    (tensor([[ 0.,  1.],
             [ 4.,  5.],
             [ 8.,  9.],
             [12., 13.]]),
     tensor([[ 2.,  3.],
             [ 6.,  7.],
             [10., 11.],
             [14., 15.]]))
    >>> torch.hsplit(t, [3, 6])
    (tensor([[ 0.,  1.,  2.],
             [ 4.,  5.,  6.],
             [ 8.,  9., 10.],
             [12., 13., 14.]]),
     tensor([[ 3.],
             [ 7.],
             [11.],
             [15.]]),
     tensor([], size=(4, 0)))

""",
)

add_docstr(
    torch.vsplit,
    r"""
vsplit(input, indices_or_sections) -> List of Tensors

Splits :attr:`input`, a tensor with two or more dimensions, into multiple tensors
vertically according to :attr:`indices_or_sections`. Each split is a view of
:attr:`input`.

This is equivalent to calling torch.tensor_split(input, indices_or_sections, dim=0)
(the split dimension is 0), except that if :attr:`indices_or_sections` is an integer
it must evenly divide the split dimension or a runtime error will be thrown.

This function is based on NumPy's :func:`numpy.vsplit`.

Args:
    input (Tensor): tensor to split.
    indices_or_sections (int or list or tuple of ints): See argument in :func:`torch.tensor_split`.

Example::

    >>> t = torch.arange(16.0).reshape(4,4)
    >>> t
    tensor([[ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.],
            [12., 13., 14., 15.]])
    >>> torch.vsplit(t, 2)
    (tensor([[0., 1., 2., 3.],
             [4., 5., 6., 7.]]),
     tensor([[ 8.,  9., 10., 11.],
             [12., 13., 14., 15.]]))
    >>> torch.vsplit(t, [3, 6])
    (tensor([[ 0.,  1.,  2.,  3.],
             [ 4.,  5.,  6.,  7.],
             [ 8.,  9., 10., 11.]]),
     tensor([[12., 13., 14., 15.]]),
     tensor([], size=(0, 4)))

""",
)

add_docstr(
    torch.dsplit,
    r"""
dsplit(input, indices_or_sections) -> List of Tensors

Splits :attr:`input`, a tensor with three or more dimensions, into multiple tensors
depthwise according to :attr:`indices_or_sections`. Each split is a view of
:attr:`input`.

This is equivalent to calling torch.tensor_split(input, indices_or_sections, dim=2)
(the split dimension is 2), except that if :attr:`indices_or_sections` is an integer
it must evenly divide the split dimension or a runtime error will be thrown.

This function is based on NumPy's :func:`numpy.dsplit`.

Args:
    input (Tensor): tensor to split.
    indices_or_sections (int or list or tuple of ints): See argument in :func:`torch.tensor_split`.

Example::

    >>> t = torch.arange(16.0).reshape(2, 2, 4)
    >>> t
    tensor([[[ 0.,  1.,  2.,  3.],
             [ 4.,  5.,  6.,  7.]],
            [[ 8.,  9., 10., 11.],
             [12., 13., 14., 15.]]])
    >>> torch.dsplit(t, 2)
    (tensor([[[ 0.,  1.],
            [ 4.,  5.]],
           [[ 8.,  9.],
            [12., 13.]]]),
     tensor([[[ 2.,  3.],
              [ 6.,  7.]],
             [[10., 11.],
              [14., 15.]]]))

    >>> torch.dsplit(t, [3, 6])
    (tensor([[[ 0.,  1.,  2.],
              [ 4.,  5.,  6.]],
             [[ 8.,  9., 10.],
              [12., 13., 14.]]]),
     tensor([[[ 3.],
              [ 7.]],
             [[11.],
              [15.]]]),
     tensor([], size=(2, 2, 0)))

""",
)

add_docstr(
    torch.can_cast,
    r"""
can_cast(from_, to) -> bool

Determines if a type conversion is allowed under PyTorch casting rules
described in the type promotion :ref:`documentation <type-promotion-doc>`.

Args:
    from\_ (dtype): The original :class:`torch.dtype`.
    to (dtype): The target :class:`torch.dtype`.

Example::

    >>> torch.can_cast(torch.double, torch.float)
    True
    >>> torch.can_cast(torch.float, torch.int)
    False
""",
)

add_docstr(
    torch.corrcoef,
    r"""
corrcoef(input) -> Tensor

Estimates the Pearson product-moment correlation coefficient matrix of the variables given by the :attr:`input` matrix,
where rows are the variables and columns are the observations.

.. note::

    The correlation coefficient matrix R is computed using the covariance matrix C as given by
    :math:`R_{ij} = \frac{ C_{ij} } { \sqrt{ C_{ii} * C_{jj} } }`

.. note::

    Due to floating point rounding, the resulting array may not be Hermitian and its diagonal elements may not be 1.
    The real and imaginary values are clipped to the interval [-1, 1] in an attempt to improve this situation.

Args:
    input (Tensor): A 2D matrix containing multiple variables and observations, or a
        Scalar or 1D vector representing a single variable.

Returns:
    (Tensor) The correlation coefficient matrix of the variables.

.. seealso::

        :func:`torch.cov` covariance matrix.

Example::

    >>> x = torch.tensor([[0, 1, 2], [2, 1, 0]])
    >>> torch.corrcoef(x)
    tensor([[ 1., -1.],
            [-1.,  1.]])
    >>> x = torch.randn(2, 4)
    >>> x
    tensor([[-0.2678, -0.0908, -0.3766,  0.2780],
            [-0.5812,  0.1535,  0.2387,  0.2350]])
    >>> torch.corrcoef(x)
    tensor([[1.0000, 0.3582],
            [0.3582, 1.0000]])
    >>> torch.corrcoef(x[0])
    tensor(1.)
""",
)

add_docstr(
    torch.cov,
    r"""
cov(input, *, correction=1, fweights=None, aweights=None) -> Tensor

Estimates the covariance matrix of the variables given by the :attr:`input` matrix, where rows are
the variables and columns are the observations.

A covariance matrix is a square matrix giving the covariance of each pair of variables. The diagonal contains
the variance of each variable (covariance of a variable with itself). By definition, if :attr:`input` represents
a single variable (Scalar or 1D) then its variance is returned.

The sample covariance of the variables :math:`x` and :math:`y` is given by:

.. math::
    \text{cov}(x,y) = \frac{\sum^{N}_{i = 1}(x_{i} - \bar{x})(y_{i} - \bar{y})}{\max(0,~N~-~\delta N)}

where :math:`\bar{x}` and :math:`\bar{y}` are the simple means of the :math:`x` and :math:`y` respectively, and
:math:`\delta N` is the :attr:`correction`.

If :attr:`fweights` and/or :attr:`aweights` are provided, the weighted covariance
is calculated, which is given by:

.. math::
    \text{cov}_w(x,y) = \frac{\sum^{N}_{i = 1}w_i(x_{i} - \mu_x^*)(y_{i} - \mu_y^*)}
    {\max(0,~\sum^{N}_{i = 1}w_i~-~\frac{\sum^{N}_{i = 1}w_ia_i}{\sum^{N}_{i = 1}w_i}~\delta N)}

where :math:`w` denotes :attr:`fweights` or :attr:`aweights` (``f`` and ``a`` for brevity) based on whichever is
provided, or :math:`w = f \times a` if both are provided, and
:math:`\mu_x^* = \frac{\sum^{N}_{i = 1}w_ix_{i} }{\sum^{N}_{i = 1}w_i}` is the weighted mean of the variable. If not
provided, ``f`` and/or ``a`` can be seen as a :math:`\mathbb{1}` vector of appropriate size.

Args:
    input (Tensor): A 2D matrix containing multiple variables and observations, or a
        Scalar or 1D vector representing a single variable.

Keyword Args:
    correction (int, optional): difference between the sample size and sample degrees of freedom.
        Defaults to Bessel's correction, ``correction = 1`` which returns the unbiased estimate,
        even if both :attr:`fweights` and :attr:`aweights` are specified. ``correction = 0``
        will return the simple average. Defaults to ``1``.
    fweights (tensor, optional): A Scalar or 1D tensor of observation vector frequencies representing the number of
        times each observation should be repeated. Its numel must equal the number of columns of :attr:`input`.
        Must have integral dtype. Ignored if ``None``. Defaults to ``None``.
    aweights (tensor, optional): A Scalar or 1D array of observation vector weights.
        These relative weights are typically large for observations considered "important" and smaller for
        observations considered less "important". Its numel must equal the number of columns of :attr:`input`.
        Must have floating point dtype. Ignored if ``None``. Defaults to ``None``.

Returns:
    (Tensor) The covariance matrix of the variables.

.. seealso::

        :func:`torch.corrcoef` normalized covariance matrix.

Example::

    >>> x = torch.tensor([[0, 2], [1, 1], [2, 0]]).T
    >>> x
    tensor([[0, 1, 2],
            [2, 1, 0]])
    >>> torch.cov(x)
    tensor([[ 1., -1.],
            [-1.,  1.]])
    >>> torch.cov(x, correction=0)
    tensor([[ 0.6667, -0.6667],
            [-0.6667,  0.6667]])
    >>> fw = torch.randint(1, 10, (3,))
    >>> fw
    tensor([1, 6, 9])
    >>> aw = torch.rand(3)
    >>> aw
    tensor([0.4282, 0.0255, 0.4144])
    >>> torch.cov(x, fweights=fw, aweights=aw)
    tensor([[ 0.4169, -0.4169],
            [-0.4169,  0.4169]])
""",
)

add_docstr(
    torch.cat,
    r"""
cat(tensors, dim=0, *, out=None) -> Tensor

Concatenates the given sequence of tensors in :attr:`tensors` in the given dimension.
All tensors must either have the same shape (except in the concatenating
dimension) or be a 1-D empty tensor with size ``(0,)``.

:func:`torch.cat` can be seen as an inverse operation for :func:`torch.split`
and :func:`torch.chunk`.

:func:`torch.cat` can be best understood via examples.

.. seealso::

    :func:`torch.stack` concatenates the given sequence along a new dimension.

Args:
    tensors (sequence of Tensors): Non-empty tensors provided must have the same shape,
        except in the cat dimension.

    dim (int, optional): the dimension over which the tensors are concatenated

Keyword args:
    {out}

Example::

    >>> x = torch.randn(2, 3)
    >>> x
    tensor([[ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497]])
    >>> torch.cat((x, x, x), 0)
    tensor([[ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497],
            [ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497],
            [ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497]])
    >>> torch.cat((x, x, x), 1)
    tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
             -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
             -0.5790,  0.1497]])
""".format(**common_args),
)

add_docstr(
    torch.concat,
    r"""
concat(tensors, dim=0, *, out=None) -> Tensor

Alias of :func:`torch.cat`.
""",
)

add_docstr(
    torch.concatenate,
    r"""
concatenate(tensors, axis=0, out=None) -> Tensor

Alias of :func:`torch.cat`.
""",
)

add_docstr(
    torch.ceil,
    r"""
ceil(input, *, out=None) -> Tensor

Returns a new tensor with the ceil of the elements of :attr:`input`,
the smallest integer greater than or equal to each element.

For integer inputs, follows the array-api convention of returning a
copy of the input tensor.

.. math::
    \text{out}_{i} = \left\lceil \text{input}_{i} \right\rceil
"""
    + r"""
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.6341, -1.4208, -1.0900,  0.5826])
    >>> torch.ceil(a)
    tensor([-0., -1., -1.,  1.])
""".format(**common_args),
)

add_docstr(
    torch.real,
    r"""
real(input) -> Tensor

Returns a new tensor containing real values of the :attr:`self` tensor.
The returned tensor and :attr:`self` share the same underlying storage.

Args:
    {input}

Example::

    >>> x=torch.randn(4, dtype=torch.cfloat)
    >>> x
    tensor([(0.3100+0.3553j), (-0.5445-0.7896j), (-1.6492-0.0633j), (-0.0638-0.8119j)])
    >>> x.real
    tensor([ 0.3100, -0.5445, -1.6492, -0.0638])

""".format(**common_args),
)

add_docstr(
    torch.imag,
    r"""
imag(input) -> Tensor

Returns a new tensor containing imaginary values of the :attr:`self` tensor.
The returned tensor and :attr:`self` share the same underlying storage.

.. warning::
    :func:`imag` is only supported for tensors with complex dtypes.

Args:
    {input}

Example::

    >>> x=torch.randn(4, dtype=torch.cfloat)
    >>> x
    tensor([(0.3100+0.3553j), (-0.5445-0.7896j), (-1.6492-0.0633j), (-0.0638-0.8119j)])
    >>> x.imag
    tensor([ 0.3553, -0.7896, -0.0633, -0.8119])

""".format(**common_args),
)

add_docstr(
    torch.view_as_real,
    r"""
view_as_real(input) -> Tensor

Returns a view of :attr:`input` as a real tensor. For an input complex tensor of
:attr:`size` :math:`m1, m2, \dots, mi`, this function returns a new
real tensor of size :math:`m1, m2, \dots, mi, 2`, where the last dimension of size 2
represents the real and imaginary components of complex numbers.

.. warning::
    :func:`view_as_real` is only supported for tensors with ``complex dtypes``.

Args:
    {input}

Example::

    >>> x=torch.randn(4, dtype=torch.cfloat)
    >>> x
    tensor([(0.4737-0.3839j), (-0.2098-0.6699j), (0.3470-0.9451j), (-0.5174-1.3136j)])
    >>> torch.view_as_real(x)
    tensor([[ 0.4737, -0.3839],
            [-0.2098, -0.6699],
            [ 0.3470, -0.9451],
            [-0.5174, -1.3136]])
""".format(**common_args),
)

add_docstr(
    torch.view_as_complex,
    r"""
view_as_complex(input) -> Tensor

Returns a view of :attr:`input` as a complex tensor. For an input complex
tensor of :attr:`size` :math:`m1, m2, \dots, mi, 2`, this function returns a
new complex tensor of :attr:`size` :math:`m1, m2, \dots, mi` where the last
dimension of the input tensor is expected to represent the real and imaginary
components of complex numbers.

.. warning::
    :func:`view_as_complex` is only supported for tensors with
    :class:`torch.dtype` ``torch.float64`` and ``torch.float32``.  The input is
    expected to have the last dimension of :attr:`size` 2. In addition, the
    tensor must have a `stride` of 1 for its last dimension. The strides of all
    other dimensions must be even numbers.

Args:
    {input}

Example::

    >>> x=torch.randn(4, 2)
    >>> x
    tensor([[ 1.6116, -0.5772],
            [-1.4606, -0.9120],
            [ 0.0786, -1.7497],
            [-0.6561, -1.6623]])
    >>> torch.view_as_complex(x)
    tensor([(1.6116-0.5772j), (-1.4606-0.9120j), (0.0786-1.7497j), (-0.6561-1.6623j)])
""".format(**common_args),
)

add_docstr(
    torch.reciprocal,
    r"""
reciprocal(input, *, out=None) -> Tensor

Returns a new tensor with the reciprocal of the elements of :attr:`input`

.. math::
    \text{out}_{i} = \frac{1}{\text{input}_{i}}

.. note::
    Unlike NumPy's reciprocal, torch.reciprocal supports integral inputs. Integral
    inputs to reciprocal are automatically :ref:`promoted <type-promotion-doc>` to
    the default scalar type.
"""
    + r"""
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.4595, -2.1219, -1.4314,  0.7298])
    >>> torch.reciprocal(a)
    tensor([-2.1763, -0.4713, -0.6986,  1.3702])
""".format(**common_args),
)

add_docstr(
    torch.cholesky,
    r"""
cholesky(input, upper=False, *, out=None) -> Tensor

Computes the Cholesky decomposition of a symmetric positive-definite
matrix :math:`A` or for batches of symmetric positive-definite matrices.

If :attr:`upper` is ``True``, the returned matrix ``U`` is upper-triangular, and
the decomposition has the form:

.. math::

  A = U^TU

If :attr:`upper` is ``False``, the returned matrix ``L`` is lower-triangular, and
the decomposition has the form:

.. math::

    A = LL^T

If :attr:`upper` is ``True``, and :math:`A` is a batch of symmetric positive-definite
matrices, then the returned tensor will be composed of upper-triangular Cholesky factors
of each of the individual matrices. Similarly, when :attr:`upper` is ``False``, the returned
tensor will be composed of lower-triangular Cholesky factors of each of the individual
matrices.

.. warning::

    :func:`torch.cholesky` is deprecated in favor of :func:`torch.linalg.cholesky`
    and will be removed in a future PyTorch release.

    ``L = torch.cholesky(A)`` should be replaced with

    .. code:: python

        L = torch.linalg.cholesky(A)

    ``U = torch.cholesky(A, upper=True)`` should be replaced with

    .. code:: python

        U = torch.linalg.cholesky(A).mH

    This transform will produce equivalent results for all valid (symmetric positive definite) inputs.

Args:
    input (Tensor): the input tensor :math:`A` of size :math:`(*, n, n)` where `*` is zero or more
                batch dimensions consisting of symmetric positive-definite matrices.
    upper (bool, optional): flag that indicates whether to return a
                            upper or lower triangular matrix. Default: ``False``

Keyword args:
    out (Tensor, optional): the output matrix

Example::

    >>> a = torch.randn(3, 3)
    >>> a = a @ a.mT + 1e-3 # make symmetric positive-definite
    >>> l = torch.cholesky(a)
    >>> a
    tensor([[ 2.4112, -0.7486,  1.4551],
            [-0.7486,  1.3544,  0.1294],
            [ 1.4551,  0.1294,  1.6724]])
    >>> l
    tensor([[ 1.5528,  0.0000,  0.0000],
            [-0.4821,  1.0592,  0.0000],
            [ 0.9371,  0.5487,  0.7023]])
    >>> l @ l.mT
    tensor([[ 2.4112, -0.7486,  1.4551],
            [-0.7486,  1.3544,  0.1294],
            [ 1.4551,  0.1294,  1.6724]])
    >>> a = torch.randn(3, 2, 2) # Example for batched input
    >>> a = a @ a.mT + 1e-03 # make symmetric positive-definite
    >>> l = torch.cholesky(a)
    >>> z = l @ l.mT
    >>> torch.dist(z, a)
    tensor(2.3842e-07)
""",
)

add_docstr(
    torch.cholesky_solve,
    r"""
cholesky_solve(B, L, upper=False, *, out=None) -> Tensor

Computes the solution of a system of linear equations with complex Hermitian
or real symmetric positive-definite lhs given its Cholesky decomposition.

Let :math:`A` be a complex Hermitian or real symmetric positive-definite matrix,
and :math:`L` its Cholesky decomposition such that:

.. math::

    A = LL^{\text{H}}

where :math:`L^{\text{H}}` is the conjugate transpose when :math:`L` is complex,
and the transpose when :math:`L` is real-valued.

Returns the solution :math:`X` of the following linear system:

.. math::

    AX = B

Supports inputs of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :math:`A` or :math:`B` is a batch of matrices
then the output has the same batch dimensions.

Args:
    B (Tensor): right-hand side tensor of shape `(*, n, k)`
        where :math:`*` is zero or more batch dimensions
    L (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
        consisting of lower or upper triangular Cholesky decompositions of
        symmetric or Hermitian positive-definite matrices.
    upper (bool, optional): flag that indicates whether :math:`L` is lower triangular
        or upper triangular. Default: ``False``.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Example::

    >>> A = torch.randn(3, 3)
    >>> A = A @ A.T + torch.eye(3) * 1e-3 # Creates a symmetric positive-definite matrix
    >>> L = torch.linalg.cholesky(A) # Extract Cholesky decomposition
    >>> B = torch.randn(3, 2)
    >>> torch.cholesky_solve(B, L)
    tensor([[ -8.1625,  19.6097],
            [ -5.8398,  14.2387],
            [ -4.3771,  10.4173]])
    >>> A.inverse() @  B
    tensor([[ -8.1626,  19.6097],
            [ -5.8398,  14.2387],
            [ -4.3771,  10.4173]])

    >>> A = torch.randn(3, 2, 2, dtype=torch.complex64)
    >>> A = A @ A.mH + torch.eye(2) * 1e-3 # Batch of Hermitian positive-definite matrices
    >>> L = torch.linalg.cholesky(A)
    >>> B = torch.randn(2, 1, dtype=torch.complex64)
    >>> X = torch.cholesky_solve(B, L)
    >>> torch.dist(X, A.inverse() @ B)
    tensor(1.6881e-5)
""",
)

add_docstr(
    torch.cholesky_inverse,
    r"""
cholesky_inverse(L, upper=False, *, out=None) -> Tensor

Computes the inverse of a complex Hermitian or real symmetric
positive-definite matrix given its Cholesky decomposition.

Let :math:`A` be a complex Hermitian or real symmetric positive-definite matrix,
and :math:`L` its Cholesky decomposition such that:

.. math::

    A = LL^{\text{H}}

where :math:`L^{\text{H}}` is the conjugate transpose when :math:`L` is complex,
and the transpose when :math:`L` is real-valued.

Computes the inverse matrix :math:`A^{-1}`.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :math:`A` is a batch of matrices
then the output has the same batch dimensions.

Args:
    L (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
        consisting of lower or upper triangular Cholesky decompositions of
        symmetric or Hermitian positive-definite matrices.
    upper (bool, optional): flag that indicates whether :math:`L` is lower triangular
        or upper triangular. Default: ``False``

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Example::

    >>> A = torch.randn(3, 3)
    >>> A = A @ A.T + torch.eye(3) * 1e-3 # Creates a symmetric positive-definite matrix
    >>> L = torch.linalg.cholesky(A) # Extract Cholesky decomposition
    >>> torch.cholesky_inverse(L)
    tensor([[ 1.9314,  1.2251, -0.0889],
            [ 1.2251,  2.4439,  0.2122],
            [-0.0889,  0.2122,  0.1412]])
    >>> A.inverse()
    tensor([[ 1.9314,  1.2251, -0.0889],
            [ 1.2251,  2.4439,  0.2122],
            [-0.0889,  0.2122,  0.1412]])

    >>> A = torch.randn(3, 2, 2, dtype=torch.complex64)
    >>> A = A @ A.mH + torch.eye(2) * 1e-3 # Batch of Hermitian positive-definite matrices
    >>> L = torch.linalg.cholesky(A)
    >>> torch.dist(torch.inverse(A), torch.cholesky_inverse(L))
    tensor(5.6358e-7)
""",
)

add_docstr(
    torch.clone,
    r"""
clone(input, *, memory_format=torch.preserve_format) -> Tensor

Returns a copy of :attr:`input`.

.. note::

    This function is differentiable, so gradients will flow back from the
    result of this operation to :attr:`input`. To create a tensor without an
    autograd relationship to :attr:`input` see :meth:`~Tensor.detach`.

    In addition, when ``torch.preserve_format`` is used:
    If the input tensor is dense (i.e., non-overlapping strided),
    its memory format (including strides) is retained.
    Otherwise (e.g., a non-dense view like a stepped slice),
    the output is converted to the dense (contiguous) format.

Args:
    {input}

Keyword args:
    {memory_format}
""".format(**common_args),
)

add_docstr(
    torch.clamp,
    r"""
clamp(input, min=None, max=None, *, out=None) -> Tensor

Clamps all elements in :attr:`input` into the range `[` :attr:`min`, :attr:`max` `]`.
Letting min_value and max_value be :attr:`min` and :attr:`max`, respectively, this returns:

.. math::
    y_i = \min(\max(x_i, \text{min\_value}_i), \text{max\_value}_i)

If :attr:`min` is ``None``, there is no lower bound.
Or, if :attr:`max` is ``None`` there is no upper bound.
"""
    + r"""

.. note::
    If :attr:`min` is greater than :attr:`max` :func:`torch.clamp(..., min, max) <torch.clamp>`
    sets all elements in :attr:`input` to the value of :attr:`max`.

Args:
    {input}
    min (Number or Tensor, optional): lower-bound of the range to be clamped to
    max (Number or Tensor, optional): upper-bound of the range to be clamped to

Keyword args:
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-1.7120,  0.1734, -0.0478, -0.0922])
    >>> torch.clamp(a, min=-0.5, max=0.5)
    tensor([-0.5000,  0.1734, -0.0478, -0.0922])

    >>> min = torch.linspace(-1, 1, steps=4)
    >>> torch.clamp(a, min=min)
    tensor([-1.0000,  0.1734,  0.3333,  1.0000])

""".format(**common_args),
)

add_docstr(
    torch.clip,
    r"""
clip(input, min=None, max=None, *, out=None) -> Tensor

Alias for :func:`torch.clamp`.
""",
)

add_docstr(
    torch.column_stack,
    r"""
column_stack(tensors, *, out=None) -> Tensor

Creates a new tensor by horizontally stacking the tensors in :attr:`tensors`.

Equivalent to ``torch.hstack(tensors)``, except each zero or one dimensional tensor ``t``
in :attr:`tensors` is first reshaped into a ``(t.numel(), 1)`` column before being stacked horizontally.

Args:
    tensors (sequence of Tensors): sequence of tensors to concatenate

Keyword args:
    {out}

Example::

    >>> a = torch.tensor([1, 2, 3])
    >>> b = torch.tensor([4, 5, 6])
    >>> torch.column_stack((a, b))
    tensor([[1, 4],
        [2, 5],
        [3, 6]])
    >>> a = torch.arange(5)
    >>> b = torch.arange(10).reshape(5, 2)
    >>> torch.column_stack((a, b, b))
    tensor([[0, 0, 1, 0, 1],
            [1, 2, 3, 2, 3],
            [2, 4, 5, 4, 5],
            [3, 6, 7, 6, 7],
            [4, 8, 9, 8, 9]])

""".format(**common_args),
)

add_docstr(
    torch.complex,
    r"""
complex(real, imag, *, out=None) -> Tensor

Constructs a complex tensor with its real part equal to :attr:`real` and its
imaginary part equal to :attr:`imag`.

Args:
    real (Tensor): The real part of the complex tensor. Must be half, float or double.
    imag (Tensor): The imaginary part of the complex tensor. Must be same dtype
        as :attr:`real`.

Keyword args:
    out (Tensor): If the inputs are ``torch.float32``, must be
        ``torch.complex64``. If the inputs are ``torch.float64``, must be
        ``torch.complex128``.

Example::

    >>> real = torch.tensor([1, 2], dtype=torch.float32)
    >>> imag = torch.tensor([3, 4], dtype=torch.float32)
    >>> z = torch.complex(real, imag)
    >>> z
    tensor([(1.+3.j), (2.+4.j)])
    >>> z.dtype
    torch.complex64

""",
)

add_docstr(
    torch.polar,
    r"""
polar(abs, angle, *, out=None) -> Tensor

Constructs a complex tensor whose elements are Cartesian coordinates
corresponding to the polar coordinates with absolute value :attr:`abs` and angle
:attr:`angle`.

.. math::
    \text{out} = \text{abs} \cdot \cos(\text{angle}) + \text{abs} \cdot \sin(\text{angle}) \cdot j

.. note::
    `torch.polar` is similar to
    `std::polar <https://en.cppreference.com/w/cpp/numeric/complex/polar>`_
    and does not compute the polar decomposition
    of a complex tensor like Python's `cmath.polar` and SciPy's `linalg.polar` do.
    The behavior of this function is undefined if `abs` is negative or NaN, or if `angle` is
    infinite.

"""
    + r"""
Args:
    abs (Tensor): The absolute value the complex tensor. Must be float or double.
    angle (Tensor): The angle of the complex tensor. Must be same dtype as
        :attr:`abs`.

Keyword args:
    out (Tensor): If the inputs are ``torch.float32``, must be
        ``torch.complex64``. If the inputs are ``torch.float64``, must be
        ``torch.complex128``.

Example::

    >>> import numpy as np
    >>> abs = torch.tensor([1, 2], dtype=torch.float64)
    >>> angle = torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64)
    >>> z = torch.polar(abs, angle)
    >>> z
    tensor([(0.0000+1.0000j), (-1.4142-1.4142j)], dtype=torch.complex128)
""",
)

add_docstr(
    torch.conj_physical,
    r"""
conj_physical(input, *, out=None) -> Tensor

Computes the element-wise conjugate of the given :attr:`input` tensor.
If :attr:`input` has a non-complex dtype, this function just returns :attr:`input`.

.. note::
   This performs the conjugate operation regardless of the fact conjugate bit is set or not.

.. warning:: In the future, :func:`torch.conj_physical` may return a non-writeable view for an :attr:`input` of
             non-complex dtype. It's recommended that programs not modify the tensor returned by :func:`torch.conj_physical`
             when :attr:`input` is of non-complex dtype to be compatible with this change.

.. math::
    \text{out}_{i} = conj(\text{input}_{i})
"""
    + r"""
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> torch.conj_physical(torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j]))
    tensor([-1 - 1j, -2 - 2j, 3 + 3j])
""".format(**common_args),
)

add_docstr(
    torch.conj,
    r"""
conj(input) -> Tensor

Returns a view of :attr:`input` with a flipped conjugate bit. If :attr:`input` has a non-complex dtype,
this function just returns :attr:`input`.

.. note::
    :func:`torch.conj` performs a lazy conjugation, but the actual conjugated tensor can be materialized
    at any time using :func:`torch.resolve_conj`.

.. warning:: In the future, :func:`torch.conj` may return a non-writeable view for an :attr:`input` of
             non-complex dtype. It's recommended that programs not modify the tensor returned by :func:`torch.conj_physical`
             when :attr:`input` is of non-complex dtype to be compatible with this change.

Args:
    {input}

Example::

    >>> x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
    >>> x.is_conj()
    False
    >>> y = torch.conj(x)
    >>> y.is_conj()
    True
""".format(**common_args),
)

add_docstr(
    torch.resolve_conj,
    r"""
resolve_conj(input) -> Tensor

Returns a new tensor with materialized conjugation if :attr:`input`'s conjugate bit is set to `True`,
else returns :attr:`input`. The output tensor will always have its conjugate bit set to `False`.

Args:
    {input}

Example::

    >>> x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
    >>> y = x.conj()
    >>> y.is_conj()
    True
    >>> z = y.resolve_conj()
    >>> z
    tensor([-1 - 1j, -2 - 2j, 3 + 3j])
    >>> z.is_conj()
    False
""".format(**common_args),
)

add_docstr(
    torch.resolve_neg,
    r"""
resolve_neg(input) -> Tensor

Returns a new tensor with materialized negation if :attr:`input`'s negative bit is set to `True`,
else returns :attr:`input`. The output tensor will always have its negative bit set to `False`.

Args:
    {input}

Example::

    >>> x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
    >>> y = x.conj()
    >>> z = y.imag
    >>> z.is_neg()
    True
    >>> out = z.resolve_neg()
    >>> out
    tensor([-1., -2., 3.])
    >>> out.is_neg()
    False
""".format(**common_args),
)

add_docstr(
    torch.copysign,
    r"""
copysign(input, other, *, out=None) -> Tensor

Create a new floating-point tensor with the magnitude of :attr:`input` and the sign of :attr:`other`, elementwise.

.. math::
    \text{out}_{i} = \begin{cases}
        -|\text{input}_{i}| & \text{if } \text{other}_{i} \leq -0.0 \\
         |\text{input}_{i}| & \text{if } \text{other}_{i} \geq 0.0 \\
    \end{cases}
"""
    + r"""

Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
and integer and float inputs.

Args:
    input (Tensor): magnitudes.
    other (Tensor or Number): contains value(s) whose signbit(s) are
        applied to the magnitudes in :attr:`input`.

Keyword args:
    {out}

Example::

    >>> a = torch.randn(5)
    >>> a
    tensor([-1.2557, -0.0026, -0.5387,  0.4740, -0.9244])
    >>> torch.copysign(a, 1)
    tensor([1.2557, 0.0026, 0.5387, 0.4740, 0.9244])
    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.7079,  0.2778, -1.0249,  0.5719],
            [-0.0059, -0.2600, -0.4475, -1.3948],
            [ 0.3667, -0.9567, -2.5757, -0.1751],
            [ 0.2046, -0.0742,  0.2998, -0.1054]])
    >>> b = torch.randn(4)
    tensor([ 0.2373,  0.3120,  0.3190, -1.1128])
    >>> torch.copysign(a, b)
    tensor([[ 0.7079,  0.2778,  1.0249, -0.5719],
            [ 0.0059,  0.2600,  0.4475, -1.3948],
            [ 0.3667,  0.9567,  2.5757, -0.1751],
            [ 0.2046,  0.0742,  0.2998, -0.1054]])
    >>> a = torch.tensor([1.])
    >>> b = torch.tensor([-0.])
    >>> torch.copysign(a, b)
    tensor([-1.])

.. note::
    copysign handles signed zeros. If the other argument has a negative zero (-0),
    the corresponding output value will be negative.

""".format(**common_args),
)

add_docstr(
    torch.cos,
    r"""
cos(input, *, out=None) -> Tensor

Returns a new tensor with the cosine of the elements of :attr:`input` given in radians.

.. math::
    \text{out}_{i} = \cos(\text{input}_{i})
"""
    + r"""
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 1.4309,  1.2706, -0.8562,  0.9796])
    >>> torch.cos(a)
    tensor([ 0.1395,  0.2957,  0.6553,  0.5574])
""".format(**common_args),
)

add_docstr(
    torch.cosh,
    r"""
cosh(input, *, out=None) -> Tensor

Returns a new tensor with the hyperbolic cosine  of the elements of
:attr:`input`.

.. math::
    \text{out}_{i} = \cosh(\text{input}_{i})
"""
    + r"""
Args:
    {input}

Keyword args:
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.1632,  1.1835, -0.6979, -0.7325])
    >>> torch.cosh(a)
    tensor([ 1.0133,  1.7860,  1.2536,  1.2805])

.. note::
   When :attr:`input` is on the CPU, the implementation of torch.cosh may use
   the Sleef library, which rounds very large results to infinity or negative
   infinity. See `here <https://sleef.org/purec.xhtml>`_ for details.
""".format(**common_args),
)

add_docstr(
    torch.cross,
    r"""
cross(input, other, dim=None, *, out=None) -> Tensor


Returns the cross product of vectors in dimension :attr:`dim` of :attr:`input`
and :attr:`other`.

Supports input of float, double, cfloat and cdouble dtypes. Also supports batches
of vectors, for which it computes the product along the dimension :attr:`dim`.
In this case, the output has the same batch dimensions as the inputs.

.. warning::
    If :attr:`dim` is not given, it defaults to the first dimension found
    with the size 3. Note that this might be unexpected.

    This behavior is deprecated and will be changed to match that of :func:`torch.linalg.cross`
    in a future release.

.. seealso::
        :func:`torch.linalg.cross` which has dim=-1 as default.


Args:
    {input}
    other (Tensor): the second input tensor
    dim  (int, optional): the dimension to take the cross-product in.

Keyword args:
    {out}

Example::

    >>> a = torch.randn(4, 3)
    >>> a
    tensor([[-0.3956,  1.1455,  1.6895],
            [-0.5849,  1.3672,  0.3599],
            [-1.1626,  0.7180, -0.0521],
            [-0.1339,  0.9902, -2.0225]])
    >>> b = torch.randn(4, 3)
    >>> b
    tensor([[-0.0257, -1.4725, -1.2251],
            [-1.1479, -0.7005, -1.9757],
            [-1.3904,  0.3726, -1.1836],
            [-0.9688, -0.7153,  0.2159]])
    >>> torch.cross(a, b, dim=1)
    tensor([[ 1.0844, -0.5281,  0.6120],
            [-2.4490, -1.5687,  1.9792],
            [-0.8304, -1.3037,  0.5650],
            [-1.2329,  1.9883,  1.0551]])
    >>> torch.cross(a, b)
    tensor([[ 1.0844, -0.5281,  0.6120],
            [-2.4490, -1.5687,  1.9792],
            [-0.8304, -1.3037,  0.5650],
            [-1.2329,  1.9883,  1.0551]])
""".format(**common_args),
)

add_docstr(
    torch.logcumsumexp,
    r"""
logcumsumexp(input, dim, *, out=None) -> Tensor
Returns the logarithm of the cumulative summation of the exponentiation of
elements of :attr:`input` in the dimension :attr:`dim`.

For summation index :math:`j` given by `dim` and other indices :math:`i`, the result is

    .. math::
        \text{{logcumsumexp}}(x)_{{ij}} = \log \sum\limits_{{k=0}}^{{j}} \exp(x_{{ik}})

Args:
    {input}
    dim  (int): the dimension to do the operation over

Keyword args:
    {out}

Example::

    >>> a = torch.randn(10)
    >>> torch.logcumsumexp(a, dim=0)
    tensor([-0.42296738, -0.04462666,  0.86278635,  0.94622083,  1.05277811,
             1.39202815,  1.83525007,  1.84492621,  2.06084887,  2.06844475]))
""".format(**reduceops_common_args),
)

add_docstr(
    torch.cummax,
    r"""
cummax(input, dim, *, out=None) -> (Tensor, LongTensor)
Returns a namedtuple ``(values, indices)`` where ``values`` is the cumulative maximum of
elements of :attr:`input` in the dimension :attr:`dim`. And ``indices`` is the index
location of each maximum value found in the dimension :attr:`dim`.

.. math::
    y_i = max(x_1, x_2, x_3, \dots, x_i)

Args:
    {input}
    dim  (int): the dimension to do the operation over

Keyword args:
    out (tuple, optional): the result tuple of two output tensors (values, indices)

Example::

    >>> a = torch.randn(10)
    >>> a
    tensor([-0.3449, -1.5447,  0.0685, -1.5104, -1.1706,  0.2259,  1.4696, -1.3284,
         1.9946, -0.8209])
    >>> torch.cummax(a, dim=0)
    torch.return_types.cummax(
        values=tensor([-0.3449, -0.3449,  0.0685,  0.0685,  0.0685,  0.2259,  1.4696,  1.4696,
         1.9946,  1.9946]),
        indices=tensor([0, 0, 2, 2, 2, 5, 6, 6, 8, 8]))
""".format(**reduceops_common_args),
)

add_docstr(
    torch.cummin,
    r"""
cummin(input, dim, *, out=None) -> (Tensor, LongTensor)
Returns a namedtuple ``(values, indices)`` where ``values`` is the cumulative minimum of
elements of :attr:`input` in the dimension :attr:`dim`. And ``indices`` is the index
location of each maximum value found in the dimension :attr:`dim`.

.. math::
    y_i = min(x_1, x_2, x_3, \dots, x_i)

Args:
    {input}
    dim  (int): the dimension to do the operation over

Keyword args:
    out (tuple, optional): the result tuple of two output tensors (values, indices)

Example::

    >>> a = torch.randn(10)
    >>> a
    tensor([-0.2284, -0.6628,  0.0975,  0.2680, -1.3298, -0.4220

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Functions
This file defines 2 function(s): parse_kwargs, merge_dicts


## Key Components

The file contains 53354 words across 14392 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 435608 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
