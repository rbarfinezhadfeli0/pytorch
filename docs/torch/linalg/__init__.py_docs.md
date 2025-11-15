# Documentation: `torch/linalg/__init__.py`

## File Metadata

- **Path**: `torch/linalg/__init__.py`
- **Size**: 114,965 bytes (112.27 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a **Python package initialization file**.

## Original Source

```python
from torch._C import (  # type: ignore[attr-defined]  # pyrefly: ignore  # missing-module-attribute
    _add_docstr,
    _linalg,
    _LinAlgError as LinAlgError,  # pyrefly: ignore  # missing-module-attribute
)


common_notes = {
    "experimental_warning": """This function is "experimental" and it may change in a future PyTorch release.""",
    "sync_note": "When inputs are on a CUDA device, this function synchronizes that device with the CPU.",
    "sync_note_ex": r"When the inputs are on a CUDA device, this function synchronizes only when :attr:`check_errors`\ `= True`.",
    "sync_note_has_ex": (
        "When inputs are on a CUDA device, this function synchronizes that device with the CPU. "
        "For a version of this function that does not synchronize, see :func:`{}`."
    ),
}


# Note: This not only adds doc strings for functions in the linalg namespace, but
# also connects the torch.linalg Python namespace to the torch._C._linalg builtins.

cross = _add_docstr(
    _linalg.linalg_cross,
    r"""
linalg.cross(input, other, *, dim=-1, out=None) -> Tensor


Computes the cross product of two 3-dimensional vectors.

Supports input of float, double, cfloat and cdouble dtypes. Also supports batches
of vectors, for which it computes the product along the dimension :attr:`dim`.
It broadcasts over the batch dimensions.

Args:
    input (Tensor): the first input tensor.
    other (Tensor): the second input tensor.
    dim  (int, optional): the dimension along which to take the cross-product. Default: `-1`.

Keyword args:
    out (Tensor, optional): the output tensor. Ignored if `None`. Default: `None`.

Example:
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
    >>> torch.linalg.cross(a, b)
    tensor([[ 1.0844, -0.5281,  0.6120],
            [-2.4490, -1.5687,  1.9792],
            [-0.8304, -1.3037,  0.5650],
            [-1.2329,  1.9883,  1.0551]])
    >>> a = torch.randn(1, 3)  # a is broadcast to match shape of b
    >>> a
    tensor([[-0.9941, -0.5132,  0.5681]])
    >>> torch.linalg.cross(a, b)
    tensor([[ 1.4653, -1.2325,  1.4507],
            [ 1.4119, -2.6163,  0.1073],
            [ 0.3957, -1.9666, -1.0840],
            [ 0.2956, -0.3357,  0.2139]])
""",
)

cholesky = _add_docstr(
    _linalg.linalg_cholesky,
    r"""
linalg.cholesky(A, *, upper=False, out=None) -> Tensor

Computes the Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **Cholesky decomposition** of a complex Hermitian or real symmetric positive-definite matrix
:math:`A \in \mathbb{K}^{n \times n}` is defined as

.. math::

    A = LL^{\text{H}}\mathrlap{\qquad L \in \mathbb{K}^{n \times n}}

where :math:`L` is a lower triangular matrix with real positive diagonal (even in the complex case) and
:math:`L^{\text{H}}` is the conjugate transpose when :math:`L` is complex, and the transpose when :math:`L` is real-valued.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

"""
    + rf"""
.. note:: {common_notes["sync_note_has_ex"].format("torch.linalg.cholesky_ex")}
"""
    + r"""

.. seealso::

        :func:`torch.linalg.cholesky_ex` for a version of this operation that
        skips the (slow) error checking by default and instead returns the debug
        information. This makes it a faster way to check if a matrix is
        positive-definite.

        :func:`torch.linalg.eigh` for a different decomposition of a Hermitian matrix.
        The eigenvalue decomposition gives more information about the matrix but it
        slower to compute than the Cholesky decomposition.

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                consisting of symmetric or Hermitian positive-definite matrices.

Keyword args:
    upper (bool, optional): whether to return an upper triangular matrix.
        The tensor returned with upper=True is the conjugate transpose of the tensor
        returned with upper=False.
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Raises:
    RuntimeError: if the :attr:`A` matrix or any matrix in a batched :attr:`A` is not Hermitian
                  (resp. symmetric) positive-definite. If :attr:`A` is a batch of matrices,
                  the error message will include the batch index of the first matrix that fails
                  to meet this condition.

Examples::

    >>> A = torch.randn(2, 2, dtype=torch.complex128)
    >>> A = A @ A.T.conj() + torch.eye(2) # creates a Hermitian positive-definite matrix
    >>> A
    tensor([[2.5266+0.0000j, 1.9586-2.0626j],
            [1.9586+2.0626j, 9.4160+0.0000j]], dtype=torch.complex128)
    >>> L = torch.linalg.cholesky(A)
    >>> L
    tensor([[1.5895+0.0000j, 0.0000+0.0000j],
            [1.2322+1.2976j, 2.4928+0.0000j]], dtype=torch.complex128)
    >>> torch.dist(L @ L.T.conj(), A)
    tensor(4.4692e-16, dtype=torch.float64)

    >>> A = torch.randn(3, 2, 2, dtype=torch.float64)
    >>> A = A @ A.mT + torch.eye(2)  # batch of symmetric positive-definite matrices
    >>> L = torch.linalg.cholesky(A)
    >>> torch.dist(L @ L.mT, A)
    tensor(5.8747e-16, dtype=torch.float64)
""",
)

cholesky_ex = _add_docstr(
    _linalg.linalg_cholesky_ex,
    r"""
linalg.cholesky_ex(A, *, upper=False, check_errors=False, out=None) -> (Tensor, Tensor)

Computes the Cholesky decomposition of a complex Hermitian or real
symmetric positive-definite matrix.

This function skips the (slow) error checking and error message construction
of :func:`torch.linalg.cholesky`, instead directly returning the LAPACK
error codes as part of a named tuple ``(L, info)``. This makes this function
a faster way to check if a matrix is positive-definite, and it provides an
opportunity to handle decomposition errors more gracefully or performantly
than :func:`torch.linalg.cholesky` does.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

If :attr:`A` is not a Hermitian positive-definite matrix, or if it's a batch of matrices
and one or more of them is not a Hermitian positive-definite matrix,
then ``info`` stores a positive integer for the corresponding matrix.
The positive integer indicates the order of the leading minor that is not positive-definite,
and the decomposition could not be completed.
``info`` filled with zeros indicates that the decomposition was successful.
If ``check_errors=True`` and ``info`` contains positive integers, then a RuntimeError is thrown.

"""
    + rf"""
.. note:: {common_notes["sync_note_ex"]}

.. warning:: {common_notes["experimental_warning"]}
"""
    + r"""

.. seealso::
        :func:`torch.linalg.cholesky` is a NumPy compatible variant that always checks for errors.

Args:
    A (Tensor): the Hermitian `n \times n` matrix or the batch of such matrices of size
                    `(*, n, n)` where `*` is one or more batch dimensions.

Keyword args:
    upper (bool, optional): whether to return an upper triangular matrix.
        The tensor returned with upper=True is the conjugate transpose of the tensor
        returned with upper=False.
    check_errors (bool, optional): controls whether to check the content of ``infos``. Default: `False`.
    out (tuple, optional): tuple of two tensors to write the output to. Ignored if `None`. Default: `None`.

Examples::

    >>> A = torch.randn(2, 2, dtype=torch.complex128)
    >>> A = A @ A.t().conj()  # creates a Hermitian positive-definite matrix
    >>> L, info = torch.linalg.cholesky_ex(A)
    >>> A
    tensor([[ 2.3792+0.0000j, -0.9023+0.9831j],
            [-0.9023-0.9831j,  0.8757+0.0000j]], dtype=torch.complex128)
    >>> L
    tensor([[ 1.5425+0.0000j,  0.0000+0.0000j],
            [-0.5850-0.6374j,  0.3567+0.0000j]], dtype=torch.complex128)
    >>> info
    tensor(0, dtype=torch.int32)

""",
)

inv = _add_docstr(
    _linalg.linalg_inv,
    r"""
linalg.inv(A, *, out=None) -> Tensor

Computes the inverse of a square matrix if it exists.
Throws a `RuntimeError` if the matrix is not invertible.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
for a matrix :math:`A \in \mathbb{K}^{n \times n}`,
its **inverse matrix** :math:`A^{-1} \in \mathbb{K}^{n \times n}` (if it exists) is defined as

.. math::

    A^{-1}A = AA^{-1} = \mathrm{I}_n

where :math:`\mathrm{I}_n` is the `n`-dimensional identity matrix.

The inverse matrix exists if and only if :math:`A` is `invertible`_. In this case,
the inverse is unique.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices
then the output has the same batch dimensions.

"""
    + rf"""
.. note:: {common_notes["sync_note_has_ex"].format("torch.linalg.inv_ex")}
"""
    + r"""

.. note::
    Consider using :func:`torch.linalg.solve` if possible for multiplying a matrix on the left by
    the inverse, as::

        linalg.solve(A, B) == linalg.inv(A) @ B  # When B is a matrix

    It is always preferred to use :func:`~solve` when possible, as it is faster and more
    numerically stable than computing the inverse explicitly.

.. seealso::

        :func:`torch.linalg.pinv` computes the pseudoinverse (Moore-Penrose inverse) of matrices
        of any shape.

        :func:`torch.linalg.solve` computes :attr:`A`\ `.inv() @ \ `:attr:`B` with a
        numerically stable algorithm.

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                consisting of invertible matrices.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Raises:
    RuntimeError: if the matrix :attr:`A` or any matrix in the batch of matrices :attr:`A` is not invertible.

Examples::

    >>> A = torch.randn(4, 4)
    >>> Ainv = torch.linalg.inv(A)
    >>> torch.dist(A @ Ainv, torch.eye(4))
    tensor(1.1921e-07)

    >>> A = torch.randn(2, 3, 4, 4)  # Batch of matrices
    >>> Ainv = torch.linalg.inv(A)
    >>> torch.dist(A @ Ainv, torch.eye(4))
    tensor(1.9073e-06)

    >>> A = torch.randn(4, 4, dtype=torch.complex128)  # Complex matrix
    >>> Ainv = torch.linalg.inv(A)
    >>> torch.dist(A @ Ainv, torch.eye(4))
    tensor(7.5107e-16, dtype=torch.float64)

.. _invertible:
    https://en.wikipedia.org/wiki/Invertible_matrix#The_invertible_matrix_theorem
""",
)

solve_ex = _add_docstr(
    _linalg.linalg_solve_ex,
    r"""
linalg.solve_ex(A, B, *, left=True, check_errors=False, out=None) -> (Tensor, Tensor)

A version of :func:`~solve` that does not perform error checks unless :attr:`check_errors`\ `= True`.
It also returns the :attr:`info` tensor returned by `LAPACK's getrf`_.

"""
    + rf"""
.. note:: {common_notes["sync_note_ex"]}

.. warning:: {common_notes["experimental_warning"]}
"""
    + r"""

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions.

Keyword args:
    left (bool, optional): whether to solve the system :math:`AX=B` or :math:`XA = B`. Default: `True`.
    check_errors (bool, optional): controls whether to check the content of ``infos`` and raise
                                   an error if it is non-zero. Default: `False`.
    out (tuple, optional): tuple of two tensors to write the output to. Ignored if `None`. Default: `None`.

Returns:
    A named tuple `(result, info)`.

Examples::

    >>> A = torch.randn(3, 3)
    >>> Ainv, info = torch.linalg.solve_ex(A)
    >>> torch.dist(torch.linalg.inv(A), Ainv)
    tensor(0.)
    >>> info
    tensor(0, dtype=torch.int32)

.. _LAPACK's getrf:
    https://www.netlib.org/lapack/explore-html-3.6.1/dd/d9a/group__double_g_ecomputational_ga0019443faea08275ca60a734d0593e60.html
""",
)

inv_ex = _add_docstr(
    _linalg.linalg_inv_ex,
    r"""
linalg.inv_ex(A, *, check_errors=False, out=None) -> (Tensor, Tensor)

Computes the inverse of a square matrix if it is invertible.

Returns a namedtuple ``(inverse, info)``. ``inverse`` contains the result of
inverting :attr:`A` and ``info`` stores the LAPACK error codes.

If :attr:`A` is not an invertible matrix, or if it's a batch of matrices
and one or more of them is not an invertible matrix,
then ``info`` stores a positive integer for the corresponding matrix.
The positive integer indicates the diagonal element of the LU decomposition of
the input matrix that is exactly zero.
``info`` filled with zeros indicates that the inversion was successful.
If ``check_errors=True`` and ``info`` contains positive integers, then a RuntimeError is thrown.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

"""
    + rf"""
.. note:: {common_notes["sync_note_ex"]}

.. warning:: {common_notes["experimental_warning"]}
"""
    + r"""

.. seealso::

        :func:`torch.linalg.inv` is a NumPy compatible variant that always checks for errors.

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                    consisting of square matrices.
    check_errors (bool, optional): controls whether to check the content of ``info``. Default: `False`.

Keyword args:
    out (tuple, optional): tuple of two tensors to write the output to. Ignored if `None`. Default: `None`.

Examples::

    >>> A = torch.randn(3, 3)
    >>> Ainv, info = torch.linalg.inv_ex(A)
    >>> torch.dist(torch.linalg.inv(A), Ainv)
    tensor(0.)
    >>> info
    tensor(0, dtype=torch.int32)

""",
)

det = _add_docstr(
    _linalg.linalg_det,
    r"""
linalg.det(A, *, out=None) -> Tensor

Computes the determinant of a square matrix.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

.. seealso::

        :func:`torch.linalg.slogdet` computes the sign and natural logarithm of the absolute
        value of the determinant of square matrices.

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Examples::

    >>> A = torch.randn(3, 3)
    >>> torch.linalg.det(A)
    tensor(0.0934)

    >>> A = torch.randn(3, 2, 2)
    >>> torch.linalg.det(A)
    tensor([1.1990, 0.4099, 0.7386])
""",
)

slogdet = _add_docstr(
    _linalg.linalg_slogdet,
    r"""
linalg.slogdet(A, *, out=None) -> (Tensor, Tensor)

Computes the sign and natural logarithm of the absolute value of the determinant of a square matrix.

For complex :attr:`A`, it returns the sign and the natural logarithm of the modulus of the
determinant, that is, a logarithmic polar decomposition of the determinant.

The determinant can be recovered as `sign * exp(logabsdet)`.
When a matrix has a determinant of zero, it returns `(0, -inf)`.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

.. seealso::

        :func:`torch.linalg.det` computes the determinant of square matrices.

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions.

Keyword args:
    out (tuple, optional): output tuple of two tensors. Ignored if `None`. Default: `None`.

Returns:
    A named tuple `(sign, logabsdet)`.

    `sign` will have the same dtype as :attr:`A`.

    `logabsdet` will always be real-valued, even when :attr:`A` is complex.

Examples::

    >>> A = torch.randn(3, 3)
    >>> A
    tensor([[ 0.0032, -0.2239, -1.1219],
            [-0.6690,  0.1161,  0.4053],
            [-1.6218, -0.9273, -0.0082]])
    >>> torch.linalg.det(A)
    tensor(-0.7576)
    >>> torch.logdet(A)
    tensor(nan)
    >>> torch.linalg.slogdet(A)
    torch.return_types.linalg_slogdet(sign=tensor(-1.), logabsdet=tensor(-0.2776))
""",
)

eig = _add_docstr(
    _linalg.linalg_eig,
    r"""
linalg.eig(A, *, out=None) -> (Tensor, Tensor)

Computes the eigenvalue decomposition of a square matrix if it exists.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **eigenvalue decomposition** of a square matrix
:math:`A \in \mathbb{K}^{n \times n}` (if it exists) is defined as

.. math::

    A = V \operatorname{diag}(\Lambda) V^{-1}\mathrlap{\qquad V \in \mathbb{C}^{n \times n}, \Lambda \in \mathbb{C}^n}

This decomposition exists if and only if :math:`A` is `diagonalizable`_.
This is the case when all its eigenvalues are different.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

The returned eigenvalues are not guaranteed to be in any specific order.

.. note:: The eigenvalues and eigenvectors of a real matrix may be complex.

"""
    + rf"""
.. note:: {common_notes["sync_note"]}
"""
    + r"""

.. warning:: This function assumes that :attr:`A` is `diagonalizable`_ (for example, when all the
             eigenvalues are different). If it is not diagonalizable, the returned
             eigenvalues will be correct but :math:`A \neq V \operatorname{diag}(\Lambda)V^{-1}`.

.. warning:: The returned eigenvectors are normalized to have norm `1`.
             Even then, the eigenvectors of a matrix are not unique, nor are they continuous with respect to
             :attr:`A`. Due to this lack of uniqueness, different hardware and software may compute
             different eigenvectors.

             This non-uniqueness is caused by the fact that multiplying an eigenvector by
             by :math:`e^{i \phi}, \phi \in \mathbb{R}` produces another set of valid eigenvectors
             of the matrix.  For this reason, the loss function shall not depend on the phase of the
             eigenvectors, as this quantity is not well-defined.
             This is checked when computing the gradients of this function. As such,
             when inputs are on a CUDA device, the computation of the gradients
             of this function synchronizes that device with the CPU.


.. warning:: Gradients computed using the `eigenvectors` tensor will only be finite when
             :attr:`A` has distinct eigenvalues.
             Furthermore, if the distance between any two eigenvalues is close to zero,
             the gradient will be numerically unstable, as it depends on the eigenvalues
             :math:`\lambda_i` through the computation of
             :math:`\frac{1}{\min_{i \neq j} \lambda_i - \lambda_j}`.

.. seealso::

        :func:`torch.linalg.eigvals` computes only the eigenvalues.
        Unlike :func:`torch.linalg.eig`, the gradients of :func:`~eigvals` are always
        numerically stable.

        :func:`torch.linalg.eigh` for a (faster) function that computes the eigenvalue decomposition
        for Hermitian and symmetric matrices.

        :func:`torch.linalg.svd` for a function that computes another type of spectral
        decomposition that works on matrices of any shape.

        :func:`torch.linalg.qr` for another (much faster) decomposition that works on matrices of
        any shape.

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                consisting of diagonalizable matrices.

Keyword args:
    out (tuple, optional): output tuple of two tensors. Ignored if `None`. Default: `None`.

Returns:
    A named tuple `(eigenvalues, eigenvectors)` which corresponds to :math:`\Lambda` and :math:`V` above.

    `eigenvalues` and `eigenvectors` will always be complex-valued, even when :attr:`A` is real. The eigenvectors
    will be given by the columns of `eigenvectors`.

Examples::

    >>> A = torch.randn(2, 2, dtype=torch.complex128)
    >>> A
    tensor([[ 0.9828+0.3889j, -0.4617+0.3010j],
            [ 0.1662-0.7435j, -0.6139+0.0562j]], dtype=torch.complex128)
    >>> L, V = torch.linalg.eig(A)
    >>> L
    tensor([ 1.1226+0.5738j, -0.7537-0.1286j], dtype=torch.complex128)
    >>> V
    tensor([[ 0.9218+0.0000j,  0.1882-0.2220j],
            [-0.0270-0.3867j,  0.9567+0.0000j]], dtype=torch.complex128)
    >>> torch.dist(V @ torch.diag(L) @ torch.linalg.inv(V), A)
    tensor(7.7119e-16, dtype=torch.float64)

    >>> A = torch.randn(3, 2, 2, dtype=torch.float64)
    >>> L, V = torch.linalg.eig(A)
    >>> torch.dist(V @ torch.diag_embed(L) @ torch.linalg.inv(V), A)
    tensor(3.2841e-16, dtype=torch.float64)

.. _diagonalizable:
    https://en.wikipedia.org/wiki/Diagonalizable_matrix#Definition
""",
)

eigvals = _add_docstr(
    _linalg.linalg_eigvals,
    r"""
linalg.eigvals(A, *, out=None) -> Tensor

Computes the eigenvalues of a square matrix.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **eigenvalues** of a square matrix :math:`A \in \mathbb{K}^{n \times n}` are defined
as the roots (counted with multiplicity) of the polynomial `p` of degree `n` given by

.. math::

    p(\lambda) = \operatorname{det}(A - \lambda \mathrm{I}_n)\mathrlap{\qquad \lambda \in \mathbb{C}}

where :math:`\mathrm{I}_n` is the `n`-dimensional identity matrix.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

The returned eigenvalues are not guaranteed to be in any specific order.

.. note:: The eigenvalues of a real matrix may be complex, as the roots of a real polynomial may be complex.

          The eigenvalues of a matrix are always well-defined, even when the matrix is not diagonalizable.

"""
    + rf"""
.. note:: {common_notes["sync_note"]}
"""
    + r"""

.. seealso::

        :func:`torch.linalg.eig` computes the full eigenvalue decomposition.

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Returns:
    A complex-valued tensor containing the eigenvalues even when :attr:`A` is real.

Examples::

    >>> A = torch.randn(2, 2, dtype=torch.complex128)
    >>> L = torch.linalg.eigvals(A)
    >>> L
    tensor([ 1.1226+0.5738j, -0.7537-0.1286j], dtype=torch.complex128)

    >>> torch.dist(L, torch.linalg.eig(A).eigenvalues)
    tensor(2.4576e-07)
""",
)

eigh = _add_docstr(
    _linalg.linalg_eigh,
    r"""
linalg.eigh(A, UPLO='L', *, out=None) -> (Tensor, Tensor)

Computes the eigenvalue decomposition of a complex Hermitian or real symmetric matrix.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **eigenvalue decomposition** of a complex Hermitian or real symmetric matrix
:math:`A \in \mathbb{K}^{n \times n}` is defined as

.. math::

    A = Q \operatorname{diag}(\Lambda) Q^{\text{H}}\mathrlap{\qquad Q \in \mathbb{K}^{n \times n}, \Lambda \in \mathbb{R}^n}

where :math:`Q^{\text{H}}` is the conjugate transpose when :math:`Q` is complex, and the transpose when :math:`Q` is real-valued.
:math:`Q` is orthogonal in the real case and unitary in the complex case.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

:attr:`A` is assumed to be Hermitian (resp. symmetric), but this is not checked internally, instead:

- If :attr:`UPLO`\ `= 'L'` (default), only the lower triangular part of the matrix is used in the computation.
- If :attr:`UPLO`\ `= 'U'`, only the upper triangular part of the matrix is used.

The eigenvalues are returned in ascending order.

"""
    + rf"""
.. note:: {common_notes["sync_note"]}
"""
    + r"""

.. note:: The eigenvalues of real symmetric or complex Hermitian matrices are always real.

.. warning:: The eigenvectors of a symmetric matrix are not unique, nor are they continuous with
             respect to :attr:`A`. Due to this lack of uniqueness, different hardware and
             software may compute different eigenvectors.

             This non-uniqueness is caused by the fact that multiplying an eigenvector by
             `-1` in the real case or by :math:`e^{i \phi}, \phi \in \mathbb{R}` in the complex
             case produces another set of valid eigenvectors of the matrix.
             For this reason, the loss function shall not depend on the phase of the eigenvectors, as
             this quantity is not well-defined.
             This is checked for complex inputs when computing the gradients of this function. As such,
             when inputs are complex and are on a CUDA device, the computation of the gradients
             of this function synchronizes that device with the CPU.

.. warning:: Gradients computed using the `eigenvectors` tensor will only be finite when
             :attr:`A` has distinct eigenvalues.
             Furthermore, if the distance between any two eigenvalues is close to zero,
             the gradient will be numerically unstable, as it depends on the eigenvalues
             :math:`\lambda_i` through the computation of
             :math:`\frac{1}{\min_{i \neq j} \lambda_i - \lambda_j}`.

.. warning:: User may see pytorch crashes if running `eigh` on CUDA devices with CUDA versions before 12.1 update 1
             with large ill-conditioned matrices as inputs.
             Refer to :ref:`Linear Algebra Numerical Stability<Linear Algebra Stability>` for more details.
             If this is the case, user may (1) tune their matrix inputs to be less ill-conditioned,
             or (2) use :func:`torch.backends.cuda.preferred_linalg_library` to
             try other supported backends.

.. seealso::

        :func:`torch.linalg.eigvalsh` computes only the eigenvalues of a Hermitian matrix.
        Unlike :func:`torch.linalg.eigh`, the gradients of :func:`~eigvalsh` are always
        numerically stable.

        :func:`torch.linalg.cholesky` for a different decomposition of a Hermitian matrix.
        The Cholesky decomposition gives less information about the matrix but is much faster
        to compute than the eigenvalue decomposition.

        :func:`torch.linalg.eig` for a (slower) function that computes the eigenvalue decomposition
        of a not necessarily Hermitian square matrix.

        :func:`torch.linalg.svd` for a (slower) function that computes the more general SVD
        decomposition of matrices of any shape.

        :func:`torch.linalg.qr` for another (much faster) decomposition that works on general
        matrices.

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                consisting of symmetric or Hermitian matrices.
    UPLO ('L', 'U', optional): controls whether to use the upper or lower triangular part
                               of :attr:`A` in the computations. Default: `'L'`.

Keyword args:
    out (tuple, optional): output tuple of two tensors. Ignored if `None`. Default: `None`.

Returns:
    A named tuple `(eigenvalues, eigenvectors)` which corresponds to :math:`\Lambda` and :math:`Q` above.

    `eigenvalues` will always be real-valued, even when :attr:`A` is complex.
    It will also be ordered in ascending order.

    `eigenvectors` will have the same dtype as :attr:`A` and will contain the eigenvectors as its columns.

Examples::
    >>> A = torch.randn(2, 2, dtype=torch.complex128)
    >>> A = A + A.T.conj()  # creates a Hermitian matrix
    >>> A
    tensor([[2.9228+0.0000j, 0.2029-0.0862j],
            [0.2029+0.0862j, 0.3464+0.0000j]], dtype=torch.complex128)
    >>> L, Q = torch.linalg.eigh(A)
    >>> L
    tensor([0.3277, 2.9415], dtype=torch.float64)
    >>> Q
    tensor([[-0.0846+-0.0000j, -0.9964+0.0000j],
            [ 0.9170+0.3898j, -0.0779-0.0331j]], dtype=torch.complex128)
    >>> torch.dist(Q @ torch.diag(L.cdouble()) @ Q.T.conj(), A)
    tensor(6.1062e-16, dtype=torch.float64)

    >>> A = torch.randn(3, 2, 2, dtype=torch.float64)
    >>> A = A + A.mT  # creates a batch of symmetric matrices
    >>> L, Q = torch.linalg.eigh(A)
    >>> torch.dist(Q @ torch.diag_embed(L) @ Q.mH, A)
    tensor(1.5423e-15, dtype=torch.float64)
""",
)

eigvalsh = _add_docstr(
    _linalg.linalg_eigvalsh,
    r"""
linalg.eigvalsh(A, UPLO='L', *, out=None) -> Tensor

Computes the eigenvalues of a complex Hermitian or real symmetric matrix.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **eigenvalues** of a complex Hermitian or real symmetric  matrix :math:`A \in \mathbb{K}^{n \times n}`
are defined as the roots (counted with multiplicity) of the polynomial `p` of degree `n` given by

.. math::

    p(\lambda) = \operatorname{det}(A - \lambda \mathrm{I}_n)\mathrlap{\qquad \lambda \in \mathbb{R}}

where :math:`\mathrm{I}_n` is the `n`-dimensional identity matrix.
The eigenvalues of a real symmetric or complex Hermitian matrix are always real.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

The eigenvalues are returned in ascending order.

:attr:`A` is assumed to be Hermitian (resp. symmetric), but this is not checked internally, instead:

- If :attr:`UPLO`\ `= 'L'` (default), only the lower triangular part of the matrix is used in the computation.
- If :attr:`UPLO`\ `= 'U'`, only the upper triangular part of the matrix is used.

"""
    + rf"""
.. note:: {common_notes["sync_note"]}
"""
    + r"""

.. seealso::

        :func:`torch.linalg.eigh` computes the full eigenvalue decomposition.

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                consisting of symmetric or Hermitian matrices.
    UPLO ('L', 'U', optional): controls whether to use the upper or lower triangular part
                               of :attr:`A` in the computations. Default: `'L'`.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Returns:
    A real-valued tensor containing the eigenvalues even when :attr:`A` is complex.
    The eigenvalues are returned in ascending order.

Examples::

    >>> A = torch.randn(2, 2, dtype=torch.complex128)
    >>> A = A + A.T.conj()  # creates a Hermitian matrix
    >>> A
    tensor([[2.9228+0.0000j, 0.2029-0.0862j],
            [0.2029+0.0862j, 0.3464+0.0000j]], dtype=torch.complex128)
    >>> torch.linalg.eigvalsh(A)
    tensor([0.3277, 2.9415], dtype=torch.float64)

    >>> A = torch.randn(3, 2, 2, dtype=torch.float64)
    >>> A = A + A.mT  # creates a batch of symmetric matrices
    >>> torch.linalg.eigvalsh(A)
    tensor([[ 2.5797,  3.4629],
            [-4.1605,  1.3780],
            [-3.1113,  2.7381]], dtype=torch.float64)
""",
)

householder_product = _add_docstr(
    _linalg.linalg_householder_product,
    r"""
householder_product(A, tau, *, out=None) -> Tensor

Computes the first `n` columns of a product of Householder matrices.

Let :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`, and
let :math:`A \in \mathbb{K}^{m \times n}` be a matrix with columns :math:`a_i \in \mathbb{K}^m`
for :math:`i=1,\ldots,m` with :math:`m \geq n`. Denote by :math:`b_i` the vector resulting from
zeroing out the first :math:`i-1` components of :math:`a_i` and setting to `1` the :math:`i`-th.
For a vector :math:`\tau \in \mathbb{K}^k` with :math:`k \leq n`, this function computes the
first :math:`n` columns of the matrix

.. math::

    H_1H_2 ... H_k \qquad\text{with}\qquad H_i = \mathrm{I}_m - \tau_i b_i b_i^{\text{H}}

where :math:`\mathrm{I}_m` is the `m`-dimensional identity matrix and :math:`b^{\text{H}}` is the
conjugate transpose when :math:`b` is complex, and the transpose when :math:`b` is real-valued.
The output matrix is the same size as the input matrix :attr:`A`.

See `Representation of Orthogonal or Unitary Matrices`_ for further details.

Supports inputs of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if the inputs are batches of matrices then
the output has the same batch dimensions.

.. seealso::

        :func:`torch.geqrf` can be used together with this function to form the `Q` from the
        :func:`~qr` decomposition.

        :func:`torch.ormqr` is a related function that computes the matrix multiplication
        of a product of Householder matrices with another matrix.
        However, that function is not supported by autograd.

.. warning::
    Gradient computations are only well-defined if :math:`\tau_i \neq \frac{1}{||a_i||^2}`.
    If this condition is not met, no error will be thrown, but the gradient produced may contain `NaN`.

Args:
    A (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
    tau (Tensor): tensor of shape `(*, k)` where `*` is zero or more batch dimensions.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Raises:
    RuntimeError: if :attr:`A` doesn't satisfy the requirement `m >= n`,
                  or :attr:`tau` doesn't satisfy the requirement `n >= k`.

Examples::

    >>> A = torch.randn(2, 2)
    >>> h, tau = torch.geqrf(A)
    >>> Q = torch.linalg.householder_product(h, tau)
    >>> torch.dist(Q, torch.linalg.qr(A).Q)
    tensor(0.)

    >>> h = torch.randn(3, 2, 2, dtype=torch.complex128)
    >>> tau = torch.randn(3, 1, dtype=torch.complex128)
    >>> Q = torch.linalg.householder_product(h, tau)
    >>> Q
    tensor([[[ 1.8034+0.4184j,  0.2588-1.0174j],
            [-0.6853+0.7953j,  2.0790+0.5620j]],

            [[ 1.4581+1.6989j, -1.5360+0.1193j],
            [ 1.3877-0.6691j,  1.3512+1.3024j]],

            [[ 1.4766+0.5783j,  0.0361+0.6587j],
            [ 0.6396+0.1612j,  1.3693+0.4481j]]], dtype=torch.complex128)

.. _Representation of Orthogonal or Unitary Matrices:
    https://www.netlib.org/lapack/lug/node128.html
""",
)

ldl_factor = _add_docstr(
    _linalg.linalg_ldl_factor,
    r"""
linalg.ldl_factor(A, *, hermitian=False, out=None) -> (Tensor, Tensor)

Computes a compact representation of the LDL factorization of a Hermitian or symmetric (possibly indefinite) matrix.

When :attr:`A` is complex valued it can be Hermitian (:attr:`hermitian`\ `= True`)
or symmetric (:attr:`hermitian`\ `= False`).

The factorization is of the form the form :math:`A = L D L^T`.
If :attr:`hermitian` is `True` then transpose operation is the conjugate transpose.

:math:`L` (or :math:`U`) and :math:`D` are stored in compact form in ``LD``.
They follow the format specified by `LAPACK's sytrf`_ function.
These tensors may be used in :func:`torch.linalg.ldl_solve` to solve linear systems.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

"""
    + rf"""
.. note:: {common_notes["sync_note_has_ex"].format("torch.linalg.ldl_factor_ex")}
"""
    + r"""

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                consisting of symmetric or Hermitian matrices.

Keyword args:
    hermitian (bool, optional): whether to consider the input to be Hermitian or symmetric.
                                For real-valued matrices, this switch has no effect. Default: `False`.
    out (tuple, optional): tuple of two tensors to write the output to. Ignored if `None`. Default: `None`.

Returns:
    A named tuple `(LD, pivots)`.

Examples::

    >>> A = torch.randn(3, 3)
    >>> A = A @ A.mT # make symmetric
    >>> A
    tensor([[7.2079, 4.2414, 1.9428],
            [4.2414, 3.4554, 0.3264],
            [1.9428, 0.3264, 1.3823]])
    >>> LD, pivots = torch.linalg.ldl_factor(A)
    >>> LD
    tensor([[ 7.2079,  0.0000,  0.0000],
            [ 0.5884,  0.9595,  0.0000],
            [ 0.2695, -0.8513,  0.1633]])
    >>> pivots
    tensor([1, 2, 3], dtype=torch.int32)

.. _LAPACK's sytrf:
    https://www.netlib.org/lapack/explore-html-3.6.1/d3/db6/group__double_s_ycomputational_gad91bde1212277b3e909eb6af7f64858a.html
""",
)

ldl_factor_ex = _add_docstr(
    _linalg.linalg_ldl_factor_ex,
    r"""
linalg.ldl_factor_ex(A, *, hermitian=False, check_errors=False, out=None) -> (Tensor, Tensor, Tensor)

This is a version of :func:`~ldl_factor` that does not perform error checks unless :attr:`check_errors`\ `= True`.
It also returns the :attr:`info` tensor returned by `LAPACK's sytrf`_.
``info`` stores integer error codes from the backend library.
A positive integer indicates the diagonal element of :math:`D` that is zero.
Division by 0 will occur if the result is used for solving a system of linear equations.
``info`` filled with zeros indicates that the factorization was successful.
If ``check_errors=True`` and ``info`` contains positive integers, then a `RuntimeError` is thrown.

"""
    + rf"""
.. note:: {common_notes["sync_note_ex"]}

.. warning:: {common_notes["experimental_warning"]}
"""
    + r"""

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                consisting of symmetric or Hermitian matrices.

Keyword args:
    hermitian (bool, optional): whether to consider the input to be Hermitian or symmetric.
                                For real-valued matrices, this switch has no effect. Default: `False`.
    check_errors (bool, optional): controls whether to check the content of ``info`` and raise
                                   an error if it is non-zero. Default: `False`.
    out (tuple, optional): tuple of three tensors to write the output to. Ignored if `None`. Default: `None`.

Returns:
    A named tuple `(LD, pivots, info)`.

Examples::

    >>> A = torch.randn(3, 3)
    >>> A = A @ A.mT # make symmetric
    >>> A
    tensor([[7.2079, 4.2414, 1.9428],
            [4.2414, 3.4554, 0.3264],
            [1.9428, 0.3264, 1.3823]])
    >>> LD, pivots, info = torch.linalg.ldl_factor_ex(A)
    >>> LD
    tensor([[ 7.2079,  0.0000,  0.0000],
            [ 0.5884,  0.9595,  0.0000],
            [ 0.2695, -0.8513,  0.1633]])
    >>> pivots
    tensor([1, 2, 3], dtype=torch.int32)
    >>> info
    tensor(0, dtype=torch.int32)

.. _LAPACK's sytrf:
    https://www.netlib.org/lapack/explore-html-3.6.1/d3/db6/group__double_s_ycomputational_gad91bde1212277b3e909eb6af7f64858a.html
""",
)

ldl_solve = _add_docstr(
    _linalg.linalg_ldl_solve,
    r"""
linalg.ldl_solve(LD, pivots, B, *, hermitian=False, out=None) -> Tensor

Computes the solution of a system of linear equations using the LDL factorization.

:attr:`LD` and :attr:`pivots` are the compact representation of the LDL factorization and
are expected to be computed by :func:`torch.linalg.ldl_factor_ex`.
:attr:`hermitian` argument to this function should be the same
as the corresponding arguments in :func:`torch.linalg.ldl_factor_ex`.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

"""
    + rf"""
.. warning:: {common_notes["experimental_warning"]}
"""
    + r"""

Args:
    LD (Tensor): the `n \times n` matrix or the batch of such matrices of size
                      `(*, n, n)` where `*` is one or more batch dimensions.
    pivots (Tensor): the pivots corresponding to the LDL factorization of :attr:`LD`.
    B (Tensor): right-hand side tensor of shape `(*, n, k)`.

Keyword args:
    hermitian (bool, optional): whether to consider the decomposed matrix to be Hermitian or symmetric.
                                For real-valued matrices, this switch has no effect. Default: `False`.
    out (tuple, optional): output tensor. `B` may be passed as `out` and the result is computed in-place on `B`.
                           Ignored if `None`. Default: `None`.

Examples::

    >>> A = torch.randn(2, 3, 3)
    >>> A = A @ A.mT # make symmetric
    >>> LD, pivots, info = torch.linalg.ldl_factor_ex(A)
    >>> B = torch.randn(2, 3, 4)
    >>> X = torch.linalg.ldl_solve(LD, pivots, B)
    >>> torch.linalg.norm(A @ X - B)
    >>> tensor(0.0001)
""",
)

lstsq = _add_docstr(
    _linalg.linalg_lstsq,
    r"""
torch.linalg.lstsq(A, B, rcond=None, *, driver=None) -> (Tensor, Tensor, Tensor, Tensor)

Computes a solution to the least squares problem of a system of linear equations.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **least squares problem** for a linear system :math:`AX = B` with
:math:`A \in \mathbb{K}^{m \times n}, B \in \mathbb{K}^{m \times k}` is defined as

.. math::

    \min_{X \in \mathbb{K}^{n \times k}} \|AX - B\|_F

where :math:`\|-\|_F` denotes the Frobenius norm.

Supports inputs of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if the inputs are batches of matrices then
the output has the same batch dimensions.

:attr:`driver` chooses the backend function that will be used.
For CPU inputs the valid values are `'gels'`, `'gelsy'`, `'gelsd`, `'gelss'`.
To choose the best driver on CPU consider:

- If :attr:`A` is well-conditioned (its `condition number`_ is not too large), or you do not mind some precision loss.

  - For a general matrix: `'gelsy'` (QR with pivoting) (default)
  - If :attr:`A` is full-rank: `'gels'` (QR)

- If :attr:`A` is not well-conditioned.

  - `'gelsd'` (tridiagonal reduction and SVD)
  - But if you run into memory issues: `'gelss'` (full SVD).

For CUDA input, the only valid driver is `'gels'`, which assumes that :attr:`A` is full-rank.

See also the `full description of these drivers`_

:attr:`rcond` is used to determine the effective rank of the matrices in :attr:`A`
when :attr:`driver` is one of (`'gelsy'`, `'gelsd'`, `'gelss'`).
In this case, if :math:`\sigma_i` are the singular values of `A` in decreasing order,
:math:`\sigma_i` will be rounded down to zero if :math:`\sigma_i \leq \text{rcond} \cdot \sigma_1`.
If :attr:`rcond`\ `= None` (default), :attr:`rcond` is set to the machine precision of the dtype of :attr:`A` times `max(m, n)`.

This function returns the solution to the problem and some extra information in a named tuple of
four tensors `(solution, residuals, rank, singular_values)`. For inputs :attr:`A`, :attr:`B`
of shape `(*, m, n)`, `(*, m, k)` respectively, it contains

- `solution`: the least squares solution. It has shape `(*, n, k)`.
- `residuals`: the squared residuals of the solutions, that is, :math:`\|AX - B\|_F^2`.
  It has shape `(*, k)`.
  It is computed when `m > n` and every matrix in :attr:`A` is full-rank,
  otherwise, it is an empty tensor.
  If :attr:`A` is a batch of matrices and any matrix in the batch is not full rank,
  then an empty tensor is returned. This behavior may change in a future PyTorch release.
- `rank`: tensor of ranks of the matrices in :attr:`A`.
  It has shape equal to the batch dimensions of :attr:`A`.
  It is computed when :attr:`driver` is one of (`'gelsy'`, `'gelsd'`, `'gelss'`),
  otherwise it is an empty tensor.
- `singular_values`: tensor of singular values of the matrices in :attr:`A`.
  It has shape `(*, min(m, n))`.
  It is computed when :attr:`driver` is one of (`'gelsd'`, `'gelss'`),
  otherwise it is an empty tensor.

.. note::
    This function computes `X = \ `:attr:`A`\ `.pinverse() @ \ `:attr:`B` in a faster and
    more numerically stable way than performing the computations separately.

.. warning::
    The default value of :attr:`rcond` may change in a future PyTorch release.
    It is therefore recommended to use a fixed value to avoid potential
    breaking changes.

Args:
    A (Tensor): lhs tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
    B (Tensor): rhs tensor of shape `(*, m, k)` where `*` is zero or more batch dimensions.
    rcond (float, optional): used to determine the effective rank of :attr:`A`.
                             If :attr:`rcond`\ `= None`, :attr:`rcond` is set to the machine
                             precision of the dtype of :attr:`A` times `max(m, n)`. Default: `None`.

Keyword args:
    driver (str, optional): name of the LAPACK/MAGMA method to be used.
        If `None`, `'gelsy'` is used for CPU inputs and `'gels'` for CUDA inputs.
        Default: `None`.

Returns:
    A named tuple `(solution, residuals, rank, singular_values)`.

Examples::

    >>> A = torch.randn(1,3,3)
    >>> A
    tensor([[[-1.0838,  0.0225,  0.2275],
         [ 0.2438,  0.3844,  0.5499],
         [ 0.1175, -0.9102,  2.0870]]])
    >>> B = torch.randn(2,3,3)
    >>> B
    tensor([[[-0.6772,  0.7758,  0.5109],
         [-1.4382,  1.3769,  1.1818],
         [-0.3450,  0.0806,  0.3967]],
        [[-1.3994, -0.1521, -0.1473],
         [ 1.9194,  1.0458,  0.6705],
         [-1.1802, -0.9796,  1.4086]]])
    >>> X = torch.linalg.lstsq(A, B).solution # A is broadcasted to shape (2, 3, 3)
    >>> torch.dist(X, torch.linalg.pinv(A) @ B)
    tensor(1.5152e-06)

    >>> S = torch.linalg.lstsq(A, B, driver='gelsd').singular_values
    >>> torch.dist(S, torch.linalg.svdvals(A))
    tensor(2.3842e-07)

    >>> A[:, 0].zero_()  # Decrease the rank of A
    >>> rank = torch.linalg.lstsq(A, B).rank
    >>> rank
    tensor([2])

.. _condition number:
    https://pytorch.org/docs/main/linalg.html#torch.linalg.cond
.. _full description of these drivers:
    https://www.netlib.org/lapack/lug/node27.html
""",
)

matrix_power = _add_docstr(
    _linalg.linalg_matrix_power,
    r"""
matrix_power(A, n, *, out=None) -> Tensor

Computes the `n`-th power of a square matrix for an integer `n`.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

If :attr:`n`\ `= 0`, it returns the identity matrix (or batch) of the same shape
as :attr:`A`. If :attr:`n` is negative, it returns the inverse of each matrix
(if invertible) raised to the power of `abs(n)`.

.. note::
    Consider using :func:`torch.linalg.solve` if possible for multiplying a matrix on the left by
    a negative power as, if :attr:`n`\ `> 0`::

        torch.linalg.solve(matrix_power(A, n), B) == matrix_power(A, -n)  @ B

    It is always preferred to use :func:`~solve` when possible, as it is faster and more
    numerically stable than computing :math:`A^{-n}` explicitly.

.. seealso::

        :func:`torch.linalg.solve` computes :attr:`A`\ `.inverse() @ \ `:attr:`B` with a
        numerically stable algorithm.

Args:
    A (Tensor): tensor of shape `(*, m, m)` where `*` is zero or more batch dimensions.
    n (int): the exponent.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Raises:
    RuntimeError: if :attr:`n`\ `< 0` and the matrix :attr:`A` or any matrix in the
                  batch of matrices :attr:`A` is not invertible.

Examples::

    >>> A = torch.randn(3, 3)
    >>> torch.linalg.matrix_power(A, 0)
    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])
    >>> torch.linalg.matrix_power(A, 3)
    tensor([[ 1.0756,  0.4980,  0.0100],
            [-1.6617,  1.4994, -1.9980],
            [-0.4509,  0.2731,  0.8001]])
    >>> torch.linalg.matrix_power(A.expand(2, -1, -1), -2)
    tensor([[[ 0.2640,  0.4571, -0.5511],
            [-1.0163,  0.3491, -1.5292],
            [-0.4899,  0.0822,  0.2773]],
            [[ 0.2640,  0.4571, -0.5511],
            [-1.0163,  0.3491, -1.5292],
            [-0.4899,  0.0822,  0.2773]]])
""",
)

matrix_rank = _add_docstr(
    _linalg.linalg_matrix_rank,
    r"""
linalg.matrix_rank(A, *, atol=None, rtol=None, hermitian=False, out=None) -> Tensor

Computes the numerical rank of a matrix.

The matrix rank is computed as the number of singular values
(or eigenvalues in absolute value when :attr:`hermitian`\ `= True`)
that are greater than :math:`\max(\text{atol}, \sigma_1 * \text{rtol})` threshold,
where :math:`\sigma_1` is the largest singular value (or eigenvalue).

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

If :attr:`hermitian`\ `= True`, :attr:`A` is assumed to be Hermitian if complex or
symmetric if real, but this is not checked internally. Instead, just the lower
triangular part of the matrix is used in the computations.

If :attr:`rtol` is not specified and :attr:`A` is a matrix of dimensions `(m, n)`,
the relative tolerance is set to be :math:`\text{rtol} = \max(m, n) \varepsilon`
and :math:`\varepsilon` is the epsilon value for the dtype of :attr:`A` (see :class:`.finfo`).
If :attr:`rtol` is not specified and :attr:`atol` is specified to be larger than zero then
:attr:`rtol` is set to zero.

If :attr:`atol` or :attr:`rtol` is a :class:`torch.Tensor`, its shape must be broadcastable to that
of the singular values of :attr:`A` as returned by :func:`torch.linalg.svdvals`.

.. note::
    This function has NumPy compatible variant `linalg.matrix_rank(A, tol, hermitian=False)`.
    However, use of the positional argument :attr:`tol` is deprecated in favor of :attr:`atol` and :attr:`rtol`.

"""
    + rf"""
.. note:: The matrix rank is computed using a singular value decomposition
          :func:`torch.linalg.svdvals` if :attr:`hermitian`\ `= False` (default) and the eigenvalue
          decomposition :func:`torch.linal
```



## High-Level Overview

"experimental_warning": """This function is "experimental" and it may change in a future PyTorch release.""",    "sync_note": "When inputs are on a CUDA device, this function synchronizes that device with the CPU.",    "sync_note_ex": r"When the inputs are on a CUDA device, this function synchronizes only when :attr:`check_errors`\ `= True`.",    "sync_note_has_ex": (        "When inputs are on a CUDA device, this function synchronizes that device with the CPU. "        "For a version of this function that does not synchronize, see :func:`{}`."    ),}# Note: This not only adds doc strings for functions in the linalg namespace, but# also connects the torch.linalg Python namespace to the torch._C._linalg builtins.cross = _add_docstr(    _linalg.linalg_cross,

This Python file contains 0 class(es) and 0 function(s).

## Detailed Analysis

### Code Structure

**Key imports**: linalg as LA, linalg as LA, linalg as LA, multi_dot, math


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/linalg`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`: linalg as LA
- `torch.linalg`: multi_dot
- `math`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`torch/linalg`):



## Cross-References

- **File Documentation**: `__init__.py_docs.md`
- **Keyword Index**: `__init__.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
