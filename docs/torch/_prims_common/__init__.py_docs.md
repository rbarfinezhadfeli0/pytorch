# Documentation: `torch/_prims_common/__init__.py`

## File Metadata

- **Path**: `torch/_prims_common/__init__.py`
- **Size**: 72,047 bytes (70.36 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a **Python package initialization file**.

## Original Source

```python
# mypy: allow-untyped-defs
from __future__ import annotations

import operator
import typing
import warnings
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from enum import Enum
from functools import reduce
from typing import (
    Any,
    cast,
    NamedTuple,
    Optional,
    overload,
    TYPE_CHECKING,
    TypeAlias,
    TypeGuard,
    TypeVar,
    Union,
)
from typing_extensions import deprecated

import torch
from torch import sym_float, sym_int, sym_max


if TYPE_CHECKING:
    # Import the following modules during type checking to enable code intelligence features,
    # such as auto-completion in tools like pylance, even when these modules are not explicitly
    # imported in user code.

    import sympy

    class _WorksWithInt(typing.Protocol):
        def __add__(self, other: Any) -> typing.Self: ...

        def __radd__(self, other: Any) -> typing.Self: ...

        def __mul__(self, other: Any) -> typing.Self: ...

        def __rmul__(self, other: Any) -> typing.Self: ...

    _IntLikeT = TypeVar("_IntLikeT", bound=_WorksWithInt)


ShapeType: TypeAlias = Union[torch.Size, list[int], tuple[int, ...]]
StrideType: TypeAlias = Union[list[int], tuple[int, ...]]
DimsType: TypeAlias = Union[int, list[int], tuple[int, ...]]
DimsSequenceType: TypeAlias = Union[list[int], tuple[int, ...]]
# TODO: Type[torch.SymInt], Type[torch.SymFloat]
NumberTypeType: TypeAlias = Union[type[bool], type[int], type[float], type[complex]]
# TODO: This needs a lot more type annotations
# NumberType = Union[bool, int, float, complex, torch.SymInt, torch.SymFloat]
NumberType: TypeAlias = Union[bool, int, float, complex]
RealNumberType: TypeAlias = Union[bool, int, float]

Number = (bool, int, float, complex, torch.SymInt, torch.SymFloat, torch.SymBool)
# I don't call it Integral because numbers.Integral includes bool, but IntLike
# does not
Dim = int
IntLike = (int, torch.SymInt)
FloatLike = (float, torch.SymFloat)
BoolLike = (bool, torch.SymBool)
IntWithoutSymInt = int
FloatWithoutSymFloat = float
DeviceLikeType: TypeAlias = Union[str, torch.device, int]
Tensor = torch.Tensor


torch_function_passthrough = {
    torch.device,
    torch.sym_not,
    torch.sym_float,
    torch.sym_int,
    torch.sym_max,
    torch.sym_min,
    torch._sym_sqrt,  # type: ignore[attr-defined]
    torch.sym_ite,
    torch.Tensor.dim,
    torch.Tensor.ndim.__get__,  # type: ignore[attr-defined]
    torch.Tensor.numel,
    torch.Tensor.size,
    torch.Tensor.storage_offset,
    torch.Tensor.stride,
    torch.Tensor.dtype.__get__,  # type: ignore[attr-defined]
    torch.Tensor.is_sparse.__get__,  # type: ignore[attr-defined]
    torch.Tensor.shape.__get__,  # type: ignore[attr-defined]
    torch.Tensor.device.__get__,  # type: ignore[attr-defined]
    torch.Tensor.requires_grad.__get__,  # type: ignore[attr-defined]
    torch.Tensor.layout.__get__,  # type: ignore[attr-defined]
    torch.Tensor.is_contiguous,
    # For TorchRefsMode only
    torch.Tensor.__format__,
    torch.Tensor.__repr__,
    torch.Tensor.requires_grad.__get__,  # type: ignore[attr-defined]
    torch.Tensor.__getitem__,
}


TensorLikeType = torch.Tensor
TensorLike = torch.Tensor
TensorSequenceType: TypeAlias = Union[list[TensorLikeType], tuple[TensorLikeType, ...]]
TensorOrNumberLikeType: TypeAlias = Union[TensorLikeType, NumberType]

CustomOutParamAnnotation = "__custom_out_param__"


def same_shape(a: ShapeType, b: ShapeType, *, allow_rhs_unbacked=False) -> bool:
    from torch.fx.experimental.symbolic_shapes import guard_or_true

    if len(a) != len(b):
        return False

    for x, y in zip(a, b):
        if allow_rhs_unbacked:
            if isinstance(y, torch.SymInt):
                continue

        # if we do not know, then they are not the same.
        if guard_or_true(x != y):
            return False

    return True


def _maybe_get_pytype(t):
    if t is torch.SymFloat:
        return float
    elif t is torch.SymInt:
        return int
    elif t is torch.SymBool:
        return bool
    else:
        return t


# TODO: look at using torch.testing.assert_close instead with an option
#   to just compare metadata
def compare_tensor_meta(
    a: TensorLikeType,
    b: TensorLikeType,
    check_sizes=True,
    check_strides=False,
    *,
    allow_rhs_unbacked=False,
    check_conj=True,
):
    """
    Checks that two tensor likes have the same shape,
    dtype and device.

    In the future this will validate additional metadata, like
    strides.
    """
    from torch._subclasses.fake_tensor import MetadataMismatchError

    assert isinstance(a, TensorLike)
    assert isinstance(b, TensorLike)

    if check_sizes and not same_shape(
        a.shape, b.shape, allow_rhs_unbacked=allow_rhs_unbacked
    ):
        msg = f"Shapes {a.shape} and {b.shape} are not equal!"
        raise MetadataMismatchError(msg)

    if a.dtype != b.dtype:
        msg = f"Dtypes {a.dtype} and {b.dtype} are not equal!"
        raise MetadataMismatchError(msg)

    if a.device != b.device:
        # Handles special cuda:0 vs cuda case
        # TODO: we should review why this happens and see about fixing it
        if (str(a.device) == "cuda:0" or str(a.device) == "cuda") and (
            str(b.device) == "cuda:0" or str(b.device) == "cuda"
        ):
            pass
        else:
            msg = f"Devices {a.device} and {b.device} are not equal!"
            raise MetadataMismatchError(msg)

    # Stride checking is currently disabled, see https://github.com/pytorch/pytorch/issues/78050
    if check_strides:
        same_strides, idx = check_significant_strides(
            a, b, allow_rhs_unbacked=allow_rhs_unbacked
        )
        if not same_strides:
            msg = f"Stride mismatch! Strides are {a.stride()} and {b.stride()} (mismatched at {idx})!"
            raise MetadataMismatchError(msg)

        if a.storage_offset() != b.storage_offset():
            msg = f"Storage offset mismatch! Storage offsets are {a.storage_offset()} and {b.storage_offset()}!"
            raise MetadataMismatchError(msg)

    if check_conj:
        if a.is_conj() != b.is_conj():
            raise MetadataMismatchError(
                f"Conj mismatch! is_conj is set to {a.is_conj()} and {b.is_conj()}"
            )

    if a.is_neg() != b.is_neg():
        raise MetadataMismatchError(
            f"Neg mismatch! is_neg is set to {a.is_neg()} and {b.is_neg()}"
        )


def _check_strides_helper(
    a: TensorLikeType,
    b: TensorLikeType,
    *,
    only_cuda=True,
    significant_only=True,
    allow_rhs_unbacked=False,
) -> tuple[bool, Optional[int]]:
    # NOTE: only on CUDA because CPU elementwise strides are incorrect in PyTorch
    # See https://github.com/pytorch/pytorch/issues/77553
    # Only compares strides that are "meaningful" -- strides for dimensions with length > 1
    # and for tensors with more than one element
    if (
        not only_cuda or a.device.type == "cuda" or b.device.type == "cuda"
    ) and a.numel() > 0:
        for idx in range(a.ndim):
            check = not significant_only or a.shape[idx] > 1
            # TODO: Check the symbols are consistent with each other
            if isinstance(b.stride()[idx], torch.SymInt):
                continue
            if a.stride()[idx] != b.stride()[idx] and check:
                return False, idx

    return True, None


def check_significant_strides(
    a: TensorLikeType, b: TensorLikeType, *, only_cuda=True, allow_rhs_unbacked=False
) -> tuple[bool, Optional[int]]:
    return _check_strides_helper(
        a,
        b,
        only_cuda=only_cuda,
        significant_only=True,
        allow_rhs_unbacked=allow_rhs_unbacked,
    )


def check_all_strides(
    a: TensorLikeType, b: TensorLikeType, *, only_cuda=True
) -> tuple[bool, Optional[int]]:
    return _check_strides_helper(a, b, only_cuda=only_cuda, significant_only=False)


def check_contiguous_sizes_strides(sizes, strides, false_if_dde=False):
    """
    Performs an equality check between actual stride & expected stride (based on composed sizes),
    handling contiguous stride representations:
    e.g. torch.empty(u0, u1, u2).contiguous().stride() -> (Max(1, u1) * Max(1, u2), Max(1, u2), 1)
    and we'd like to treat this equal to (u1 * u2, u2, 1) for comparison purposes.
    """

    from torch.fx.experimental.symbolic_shapes import (
        guard_or_false,
        guard_or_true,
        is_nested_int,
    )

    def eval_eager(x):
        return bool(x)

    maybe_guard_or_false = guard_or_false if false_if_dde else eval_eager
    maybe_guard_or_true = guard_or_true if false_if_dde else eval_eager

    expected_stride = 1
    expected_stride_max = 1

    for x, y in reversed(tuple(zip(sizes, strides))):
        # Skips checking strides when a dimension has length 1.
        if maybe_guard_or_false(x == 1):
            continue

        if maybe_guard_or_true(y != expected_stride) and maybe_guard_or_true(
            y != expected_stride_max
        ):
            return False

        #  We symbolically check both paths to maximize the cases where this function
        #  returns true. This is because make_contiguous_strides_for adds the max
        #  symbolically, and in some other situations the max might not be there.
        #  And we want to ensure we return true in both cases.
        expected_stride_max *= x if is_nested_int(x) else sym_max(x, 1)  # type:ignore[assignment]

        expected_stride *= x

    return True


# This function is equivalent to compute_contiguous() from TensorImpl.cpp
def is_contiguous(a: TensorLikeType, false_if_dde=False) -> bool:
    """
    Tests whether a tensor is contiguous or not.

    Tensors are contiguous when they have no elements,
    one element, or when they have "nested" strides.
    """
    from torch.fx.experimental.symbolic_shapes import (
        guard_or_false,
        guard_size_oblivious,
    )

    def eval_eager(x):
        return bool(x)

    maybe_guard_or_false = guard_or_false if false_if_dde else eval_eager

    if maybe_guard_or_false(a.numel() < 2):
        return True

    return check_contiguous_sizes_strides(
        a.shape, a.stride(), false_if_dde=false_if_dde
    )


# This function is equivalent to compute_channels_last_contiguous_2d() in TensorImpl.cpp
def is_channels_last_contiguous_2d(a: Tensor, false_if_dde=False) -> bool:
    # NHWC or not channels last 2D contiguous
    if a.ndim != 4:
        return False

    from torch.fx.experimental.symbolic_shapes import guard_or_false, guard_or_true

    def eval_eager(x):
        return bool(x)

    maybe_guard_or_false = guard_or_false if false_if_dde else eval_eager
    maybe_guard_or_true = guard_or_true if false_if_dde else eval_eager

    expected_stride = 1
    for idx in (1, 3, 2, 0):
        length = a.shape[idx]
        if maybe_guard_or_false(length == 1):
            continue

        stride = a.stride()[idx]
        if maybe_guard_or_true(stride != expected_stride):
            return False

        expected_stride *= length

    return True


def is_channels_last_contiguous_3d(a: Tensor, false_if_dde=False) -> bool:
    # NDHWC or not channels last 3D contiguous
    if a.ndim != 5:
        return False

    from torch.fx.experimental.symbolic_shapes import guard_or_false, guard_or_true

    def eval_eager(x):
        return bool(x)

    maybe_guard_or_false = guard_or_false if false_if_dde else eval_eager
    maybe_guard_or_true = guard_or_true if false_if_dde else eval_eager

    expected_stride = 1
    for idx in (1, 4, 3, 2, 0):
        length = a.shape[idx]
        if maybe_guard_or_false(length == 1):
            continue

        stride = a.stride()[idx]
        if maybe_guard_or_true(stride != expected_stride):
            return False

        expected_stride *= length

    return True


_memory_formats = {
    torch.contiguous_format,
    torch.preserve_format,
    torch.channels_last,
    torch.channels_last_3d,
}


def validate_memory_format(memory_format: torch.memory_format):
    torch._check(
        memory_format in _memory_formats,
        lambda: f"Received unknown memory format {memory_format}!",
    )


def is_contiguous_for_memory_format(  # type: ignore[return]
    a: Tensor,
    *,
    memory_format: torch.memory_format,
    false_if_dde=False,
    # pyrefly: ignore [bad-return]
) -> bool:
    validate_memory_format(memory_format)

    if memory_format == torch.contiguous_format:
        return is_contiguous(a, false_if_dde)
    if memory_format == torch.channels_last:
        return is_channels_last_contiguous_2d(a, false_if_dde)
    if memory_format == torch.channels_last_3d:
        return is_channels_last_contiguous_3d(a, false_if_dde)

    torch._check(
        False,
        lambda: f"is_contiguous received unsupported memory format {memory_format}",
    )


def is_contiguous_or_false(a: TensorLikeType) -> bool:
    return is_contiguous(a, false_if_dde=True)


# similar to is_channels_last_contiguous_2d but return false on data dependency.
def is_channels_last_contiguous_or_false_2d(a: Tensor) -> bool:
    return is_channels_last_contiguous_2d(a, false_if_dde=True)


# similar to is_channels_last_contiguous_3d but return false on data dependency.
def is_channels_last_contiguous_or_false_3d(a: Tensor) -> bool:
    return is_channels_last_contiguous_3d(a, false_if_dde=True)


# similar to is_contiguous_for_memory_format but return false on data dependency.
def is_contiguous_for_memory_format_or_false(  # type: ignore[return]
    a: Tensor, *, memory_format: torch.memory_format
) -> bool:
    return is_contiguous_for_memory_format(
        a, memory_format=memory_format, false_if_dde=True
    )


# NOTE: that tensors with no elements and channels last is ???
def is_channels_last_contiguous(a: Tensor) -> bool:
    """
    True when a tensor is channels-last contiguous.

    This requires that:

      - the tensor is conceptually either 4 (NHWC) or 5 (NDHWC) dimensions
      - if we name the tensor's dimensions NCHW or NCDHW, then the strides are such that the
        stride of the 'C' dimension (Cs) is 1 and the strides corresponding to
        each dimension (Xs) can be ordered Cs <= Ws <= Hs <= (Ds) <= Ns and are
        "nested" -- so Ws = Cs * Cl, where Cl is the length of the 'C' dimension,
        for example.
    """
    return is_channels_last_contiguous_2d(a) or is_channels_last_contiguous_3d(a)


# similar to is_channels_last_contiguous but return false on data dependency.
def is_channels_last_contiguous_or_false(a: Tensor) -> bool:
    return is_channels_last_contiguous_or_false_2d(
        a
    ) or is_channels_last_contiguous_or_false_3d(a)


def _is_non_overlapping_and_dense_or_false(sizes, strides) -> bool:
    """
    Helper function for is_non_overlapping_and_dense.
    For unbacked sizes & strides, returns True only if symbolically non-overlapping & dense,
    and False otherwise.

    e.g. sizes: [u0, u1], strides: [u2, u3]
    this may be non-overlapping & dense at runtime, for values {u0: 4, u1: 4, u2: 4, u3: 1},
    but isn't true for all values.
    """
    from torch.fx.experimental.symbolic_shapes import guard_or_false, guard_or_true
    from torch.utils._sympy.functions import Max

    # Short-circuits for 0/1-element tensors
    if guard_or_false(prod(sizes) < 2):  # type: ignore[operator]
        return True

    # Short-circuits for tensors of rank one, which are
    # non-overlapping and "dense" if their stride is one
    if len(sizes) == 1:
        return guard_or_false(strides[0] == 1)

    # Checks that there exists a permutation of the strides s.t. the tensor would be contiguous
    # Sorts (length, stride) pairs by stride
    #
    # This sort is done in a size-oblivious way, which helps if we do a
    # comparison like 2048*u0 > u0; we just want this to return True
    # (and not worry about what if u0 is zero).
    class K(NamedTuple):
        size: int
        stride: int

        def __lt__(self, other):
            # for backed symbols, this is practically a < operation
            # for unbacked, we return True if < is statically known,
            # then try to answer this symbolically, with stride ordering semantics
            # (e.g. u0 < u0 is False, u0 < u1 is False with no axioms, u0 < 2 * u0 is True)
            return (
                guard_or_false(
                    self.stride < other.stride
                )  # checks statically known inequality
                or (
                    (
                        guard_or_false(self.stride == 0)
                        or guard_or_false(other.stride % self.stride == 0)
                    )
                    and guard_or_true(self.stride != other.stride)
                )  # checks symbolic inequality (e.g. u0 < 2048 * u0)
            )

    lengths_and_strides = sorted(map(K, sizes, strides))

    # verify actual strides match the expected (composed sizes)
    sizes = [x.size for x in lengths_and_strides][::-1]
    strides = [x.stride for x in lengths_and_strides][::-1]
    return check_contiguous_sizes_strides(sizes, strides, false_if_dde=True)


def is_non_overlapping_and_dense(a: Tensor) -> bool:
    """
    True when a tensor is non-overlapping and dense.

    A tensor is non-overlapping and dense when there exists a permutation of
    its dimensions that is contiguous.
    """
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    if a.is_sparse:
        return False

    return _is_non_overlapping_and_dense_or_false(a.shape, a.stride())


# NOTE: Based on the implementation in TensorIterator.cpp, but note that
# the note [Computing output strides] is incorrect, because it
# says that strides will be preserved even if they are not
# "non overlapping and dense", but this is incorrect. The
# output of elementwise operations are always given
# non overlapping and dense strides.
# This is also INCORRECT because it does not model TensorIterator's
# short-circuit, which can cause different strides.
def compute_elementwise_output_logical_to_physical_perm(
    *tensors, _skip_checks=False, ambiguity_check=False
) -> tuple[list[int], bool]:
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    if not _skip_checks and len(tensors) == 0:
        msg = "Can't compute elementwise output strides for zero tensors!"
        raise ValueError(msg)

    if not _skip_checks:
        check_same_shape(*tensors, allow_cpu_scalar_tensors=True)

    # Filters the tensors to actual tensors
    if not _skip_checks:
        tensors = tuple(
            a
            for a in tensors
            if isinstance(a, TensorLike) and not is_cpu_scalar_tensor(a)
        )

    # Short-circuits for CPU scalar case
    if len(tensors) == 0:
        return [], False

    # Short-circuits for shapes with zero or one dimensions
    # TODO: are these necessary?
    ndim = tensors[0].ndim
    if ndim == 0:
        return [], False
    if ndim == 1:
        return [0], False

    # Short-circuits if contiguous or channels last, following the fake fast path.
    # This reduces the number of guards we end up making
    is_contiguous = True
    is_channels_last = True
    for t in tensors:
        is_contiguous = is_contiguous and is_contiguous_for_memory_format_or_false(
            t, memory_format=torch.contiguous_format
        )
        is_channels_last = (
            is_channels_last
            and is_contiguous_for_memory_format_or_false(
                t, memory_format=torch.channels_last
            )
        )

    if is_contiguous and not is_channels_last:
        return list(range(ndim)), False

    if is_channels_last and not is_contiguous:
        return [0, *list(range(2, ndim)), 1], False

    shape = tensors[0].shape

    def should_swap(idx_a, idx_b):
        def ge(a, b):
            """
            Returns true if a is symbolically greater than or equal to b, assuming a >= 0, b >= 0.
            """
            if guard_or_false(b == 0):
                return True
            elif guard_or_false(a == 0):
                return False
            return guard_or_false(a >= b) or guard_or_false(a % b == 0)

        for tensor in tensors:
            stride_a = tensor.stride()[idx_a]
            stride_b = tensor.stride()[idx_b]

            if guard_or_false(stride_a == 0) or guard_or_false(stride_b == 0):
                continue

            if guard_or_false(stride_a == stride_b):
                if ge(shape[idx_b], shape[idx_a]):
                    continue
                return 1

            if ge(stride_b, stride_a):
                return -1

            if ge(stride_a, stride_b):
                return 1

        # Note: this case is hit if all strides are zero,
        # or all strides are equal and all dimensions have the same length
        return 0

    # The "sort" order for the permutation is back-to-front, but
    # the natural order for permutations is front-to-back.  Do the
    # sorting back-to-front and then reverse it on output.
    #
    # also, note this returns the logical to physical shape permutation
    perm = list(reversed(range(ndim)))

    # insertion sort with support for ambiguous comparisons
    for i in range(1, ndim):
        dim1 = i
        for dim0 in reversed(range(i)):
            comparison = should_swap(perm[dim0], perm[dim1])
            if comparison > 0:
                perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
                dim1 = dim0
            elif comparison < 0:
                break

    # verify we've imposed ordering if ambiguity_check=True
    raise_ambiguous = False
    if ambiguity_check:
        for i, j in zip(range(ndim - 1), range(1, ndim)):
            order = should_swap(perm[i], perm[j])
            if order != -1:
                raise_ambiguous = True
                break

    return list(reversed(perm)), raise_ambiguous


def compute_elementwise_output_strides(*tensors) -> tuple[int, ...]:
    """
    Computes the output strides for elementwise operations.
    """
    if len(tensors) == 0:
        msg = "Can't compute elementwise output strides for zero tensors!"
        raise ValueError(msg)

    check_same_shape(*tensors, allow_cpu_scalar_tensors=True)

    # Filters the tensors to actual tensors
    tensors = tuple(
        a for a in tensors if isinstance(a, TensorLike) and not is_cpu_scalar_tensor(a)
    )

    # Short-circuits for CPU scalar case
    if len(tensors) == 0:
        return ()

    ndim = tensors[0].ndim
    shape = tensors[0].shape

    if ndim == 0:
        return ()
    if ndim == 1:
        return (1,)

    logical_to_physical_perm, _ = compute_elementwise_output_logical_to_physical_perm(
        *tensors, _skip_checks=True
    )
    permuted_shape = apply_perm(shape, logical_to_physical_perm)  # to physical

    new_strides = make_contiguous_strides_for(permuted_shape)
    permuted_strides = apply_perm(
        new_strides, invert_perm(logical_to_physical_perm)
    )  # to logical

    return tuple(permuted_strides)


# Identity permutation is [0, 1, 2]
def apply_perm(inp, perm):
    ndim = len(inp)
    permuted_inp = [-1] * ndim
    for idx, x in enumerate(perm):
        permuted_inp[idx] = inp[x]
    return permuted_inp


def invert_perm(perm):
    ndim = len(perm)
    new_perm = [-1] * ndim
    for idx, x in enumerate(perm):
        new_perm[x] = idx
    return new_perm


#
# Common helper functions
#


def validate_dim_length(length: int):
    """
    Validates that an object represents a valid
    dimension length.
    """

    if isinstance(length, (int, torch.SymInt)):
        torch._check(length >= 0)
    else:
        # sometimes called with sympy expression by inductor
        assert length >= 0


def validate_shape(shape: ShapeType):
    """
    Validates that a sequence represents a valid shape.
    """

    assert isinstance(shape, Sequence), type(shape)
    for l in shape:
        validate_dim_length(l)


def validate_strides(strides: StrideType):
    """
    Verifies the object specifies valid strides.
    """

    assert isinstance(strides, Sequence)
    for stride in strides:
        assert stride >= 0


def validate_idx(rank: int, idx: int):
    """
    Validates that idx is a valid index for the given shape.
    Assumes the index is already canonicalized.
    """

    assert isinstance(idx, Dim)
    assert isinstance(rank, Dim)

    assert idx >= 0 and idx < rank or idx == 0


def validate_dimension_indices(rank: int, indices: DimsSequenceType):
    for idx in indices:
        validate_idx(rank, idx)


def validate_exclusive_idx(rank: int, ex_idx: int):
    """
    Validates that ex_idx is a valid exclusive index
    for the given shape.
    """

    assert isinstance(ex_idx, Dim)
    assert isinstance(rank, Dim)
    assert ex_idx > 0 and ex_idx <= rank


# "Wraps" a dim (up to one time) for the given rank, allowing dims to be
# specified using negative indices. If `wrap_scalar` is true then scalar
# tensors of rank 0 will allow dimensions in the range [-1, 0]. Otherwise,
# idx should be in the range [-rank, rank-1].
def canonicalize_dim(rank: int, idx: int, wrap_scalar: bool = True) -> int:
    if rank < 0:
        msg = f"Rank cannot be negative but got {rank}"
        raise IndexError(msg)

    if rank == 0:
        if not wrap_scalar:
            msg = f"Dimension specified as {idx} but tensor has no dimensions"
            raise IndexError(msg)
        rank = 1

    if idx >= 0 and idx < rank:
        return idx

    if idx < 0:
        _idx = idx + rank
    else:
        _idx = idx

    if _idx < 0 or _idx >= rank:
        # Same error message as in aten/src/ATen/WrapDimUtils.h:49
        msg = f"Dimension out of range (expected to be in range of [{-rank}, {rank - 1}], but got {idx})"
        raise IndexError(msg)

    return _idx


# Takes a dimension or sequence of dimensions and "wraps" them,
# mapping negative offsets to positive ones
@overload
def canonicalize_dims(
    rank: int,
    indices: Sequence[int],
    wrap_scalar: bool = True,
    # pyrefly: ignore [bad-return]
) -> tuple[int, ...]:
    pass


@overload
# pyrefly: ignore [bad-return]
def canonicalize_dims(rank: int, indices: int, wrap_scalar: bool = True) -> int:
    pass


def canonicalize_dims(rank, indices, wrap_scalar=True):
    if isinstance(indices, Dim):
        return canonicalize_dim(rank, indices, wrap_scalar)

    return tuple(canonicalize_dim(rank, x, wrap_scalar) for x in indices)


def is_valid_permutation(rank: int, perm: DimsSequenceType) -> bool:
    """
    Validates that perm is a permutation of length rank.
    """

    return isinstance(perm, Sequence) and sorted(perm) == list(range(rank))


def is_same_shape(a: Sequence, b: Sequence) -> bool:
    """
    Compares two shapes a and b, returning True if they are the same
    (their ranks and corresponding lengths match) and False otherwise.
    """

    return tuple(a) == tuple(b)


def is_cpu_scalar_tensor(a: object) -> TypeGuard[TensorLike]:
    return isinstance(a, TensorLike) and a.ndim == 0 and a.device.type == "cpu"


def check_same_device(*args, allow_cpu_scalar_tensors):
    """
    Checks that all Tensors in args have the same device.

    Raises a RuntimeError when:
      - args contains an object whose type is not Tensor or Number
      - two Tensor objects in args have different devices, unless one is a CPU scalar tensor and allow_cpu_scalar_tensors is True
    """
    # Short-circuits if all (one or fewer) arguments are trivially on the same device
    if len(args) <= 1:
        return

    # Note: cannot initialize device to the first arg's device (it may not have one)
    device = None
    # pyrefly: ignore [bad-assignment]
    for arg in args:
        if isinstance(arg, Number):
            continue
        elif isinstance(arg, TensorLike):
            if allow_cpu_scalar_tensors and is_cpu_scalar_tensor(arg):
                continue

            if device is None:
                device = arg.device

            if device != arg.device:
                msg = (
                    "Tensor on device "
                    + str(arg.device)
                    + " is not on the expected device "
                    + str(device)
                    + "!"
                )
                raise RuntimeError(msg)
        else:
            msg = (
                "Unexpected type when checking for same device, " + str(type(arg)) + "!"
            )
            raise RuntimeError(msg)


def canonicalize_device(device: DeviceLikeType) -> torch.device:
    if isinstance(device, torch.device):
        return device

    assert isinstance(device, str)
    return torch.device(device)


# Asserts if any of the following are true:
#   - a non-scalar or non-Tensor is given
#   - the shape of any tensors is distinct
def check_same_shape(*args, allow_cpu_scalar_tensors: bool):
    """
    Checks that all Tensors in args have the same shape.

    Raises a RuntimeError when:
      - args contains an object whose type is not Tensor or Number
      - two Tensor objects in args have different devices
    """
    shape = None

    # pyrefly: ignore [bad-assignment]
    for arg in args:
        if isinstance(arg, Number):
            continue
        elif isinstance(arg, TensorLike):
            if allow_cpu_scalar_tensors and is_cpu_scalar_tensor(arg):
                continue

            if shape is None:
                shape = arg.shape

            if not is_same_shape(shape, arg.shape):
                msg = f"Shape {arg.shape} is not the expected shape {shape}!"
                raise RuntimeError(msg)
        else:
            msg = (
                "Unexpected type when checking for same shape, " + str(type(arg)) + "!"
            )
            raise RuntimeError(msg)


# Acquires a common shape, if it exists, from one or more tensor arguments,
# filtering number arguments
def extract_shape(*args, allow_cpu_scalar_tensors: bool) -> Optional[ShapeType]:
    shape = None
    scalar_shape = None

    # pyrefly: ignore [bad-assignment]
    for arg in args:
        if isinstance(arg, Number):
            continue
        elif isinstance(arg, TensorLike):
            if allow_cpu_scalar_tensors and is_cpu_scalar_tensor(arg):
                scalar_shape = arg.shape
                continue

            if shape is None:
                shape = arg.shape

            if not is_same_shape(shape, arg.shape):
                return None
        else:
            return None

    return shape if shape is not None else scalar_shape


# Extracts dimensions that might be passed either as a list/tuple or as varargs.
# A typical case is Tensor.permute .
def extract_dims_from_varargs(
    dims: Union[DimsSequenceType, tuple[DimsSequenceType, ...]],
) -> DimsSequenceType:
    if dims and isinstance(dims[0], Sequence):
        assert len(dims) == 1
        dims = cast(tuple[DimsSequenceType], dims)
        return dims[0]
    else:
        return cast(DimsSequenceType, dims)


def extract_shape_from_varargs(
    shape: Union[ShapeType, tuple[ShapeType]],
    validate=True,
) -> tuple[int, ...]:
    """
    Returns a shape from varargs.

    In PyTorch, operations that accept shapes often accept them as varargs, like
    foo(*shape). However a user can pass the shape as a sequence of integers,
    like this:

      foo(1, 2, 3)

    or as a sequence of integers

      foo((1, 2, 3))

    In the first case shape will be a tuple of integers, and in the second case it's a tuple
    containing a tuple of integers. This validates those inputs and canonicalizes them
    to a tuple of integers.
    """

    # Handles tuple unwrapping
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        # pyrefly: ignore [bad-assignment]
        shape = shape[0]

    if validate:
        validate_shape(shape)  # type: ignore[arg-type]
    return shape  # type: ignore[return-value]


def infer_size_shapes(a: ShapeType, b: ShapeType) -> tuple[int, ...]:
    ndim = max(len(a), len(b))
    expandedSizes = [0] * ndim

    for i in range(ndim - 1, -1, -1):
        offset = ndim - 1 - i
        dimA = len(a) - 1 - offset
        dimB = len(b) - 1 - offset
        sizeA = a[dimA] if dimA >= 0 else 1
        sizeB = b[dimB] if dimB >= 0 else 1

        torch._check(
            (sizeA == sizeB) or (sizeA == 1) or (sizeB == 1),
            lambda: (
                f"The size of tensor a ({sizeA}) must match the size of "
                f"tensor b ({sizeB}) at non-jagged dimension {i}"
            ),
        )

        # 1s map to the other size (even 0)
        expandedSizes[i] = sizeB if sizeA == 1 else sizeA

    return tuple(expandedSizes)


def infer_size(shape: ShapeType, numel: int) -> tuple[int, ...]:
    """
    Infers the size of a dim with size -1, if it exists.
    Also checks that new shape is compatible with the number of elements.
    """
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    dim = None
    newsize = 1
    for i, d in enumerate(shape):
        if guard_or_false(d == -1):
            torch._check(dim is None, lambda: "only one dimension can be inferred")
            dim = i
        else:
            torch._check(
                d >= 0,
                lambda: (
                    f"invalid shape dimension {d}. If this was symbolic, it was assumed to not be -1."
                    "If this was meant to be inferred, please explicitly pass in -1."
                ),
            )
            newsize *= d
    if dim is None:
        torch._check(
            numel == newsize,
            lambda: f"shape '{list(shape)}' is invalid for input of size {numel}",
        )
    else:
        torch._check(
            newsize != 0,
            lambda: (
                f"cannot reshape tensor of 0 elements into shape {list(shape)} because the "
                f"unspecified dimension size -1 can be any value and is ambiguous"
                if guard_or_false(numel == 0)
                else f"shape '{list(shape)}' is invalid for input of size {numel}"
            ),
        )
        torch._check(
            numel % newsize == 0,
            lambda: f"shape '{list(shape)}' is invalid for input of size {numel}",
        )
        # Convert to list to produce a compatible error message with core
        # PyTorch, which prints sequences in square brackets.
        shape = list(shape)
        shape[dim] = numel // newsize
        torch._check(shape[dim] >= 0)
    return tuple(shape)


_integer_dtypes = (
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)
_low_precision_dtypes = (torch.float16, torch.bfloat16, torch.complex32)
_complex_dtypes = (torch.complex32, torch.complex64, torch.complex128)


def is_boolean_dtype(dtype: torch.dtype) -> bool:
    assert isinstance(dtype, torch.dtype)
    return dtype is torch.bool


def is_integer_dtype(dtype: torch.dtype) -> bool:
    assert isinstance(dtype, torch.dtype)
    return dtype in _integer_dtypes


def is_low_precision_dtype(dtype: torch.dtype) -> bool:
    assert isinstance(dtype, torch.dtype)
    return dtype in _low_precision_dtypes


def is_float_dtype(dtype: torch.dtype) -> bool:
    assert isinstance(dtype, torch.dtype)
    return dtype.is_floating_point


def is_complex_dtype(dtype: torch.dtype) -> bool:
    assert isinstance(dtype, torch.dtype)
    return dtype in _complex_dtypes


def is_grad_dtype(dtype: torch.dtype) -> bool:
    """
    Checks if the dtype can require a gradient.
    """
    return dtype.is_floating_point or is_complex_dtype(dtype)


_complex_to_real_dtype_map = {
    torch.complex128: torch.float64,
    torch.complex64: torch.float32,
    torch.complex32: torch.float16,
}

_real_to_complex_dtype_map = {
    torch.float16: torch.complex32,
    torch.bfloat16: torch.complex64,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}


def corresponding_real_dtype(dtype: torch.dtype) -> torch.dtype:
    return _complex_to_real_dtype_map[dtype]


def corresponding_complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return _real_to_complex_dtype_map[dtype]


def dtype_to_type(dtype: torch.dtype) -> type:
    """
    Computes the corresponding Python type (AKA "type kind") for the
    given dtype.
    """
    assert isinstance(dtype, torch.dtype)

    if dtype is torch.bool:
        return bool
    if dtype in _integer_dtypes:
        return int
    if dtype.is_floating_point:
        return float
    if dtype in _complex_dtypes:
        return complex

    raise ValueError("Invalid dtype!")


def dtype_to_type_ctor(dtype: torch.dtype) -> Callable[[NumberType], NumberType]:
    """
    Computes the corresponding Python type constructor for the
    given dtype.
    """
    assert isinstance(dtype, torch.dtype)

    if dtype is torch.bool:
        return lambda x: bool(x)
    if dtype in _integer_dtypes:
        return sym_int
    if dtype.is_floating_point:
        return sym_float
    if dtype in _complex_dtypes:
        # TODO: type error here is real, replace with sym_complex
        return lambda x: complex(x)  # type: ignore[arg-type]

    raise ValueError("Invalid dtype!")


def type_to_dtype(typ: type) -> torch.dtype:
    """
    Computes the corresponding dtype for a Number type.
    """

    assert isinstance(typ, type)

    if typ in (bool, torch.SymBool):
        return torch.bool
    if typ in (int, torch.SymInt):
        return torch.long
    if typ in (float, torch.SymFloat):
        return torch.get_default_dtype()
    # TODO: sym_complex_float?
    if typ is complex:
        return corresponding_complex_dtype(torch.get_default_dtype())

    raise ValueError(f"Invalid type {typ}!")


def get_dtype(x: Union[torch.Tensor, NumberType]):
    if isinstance(x, torch.Tensor):
        return x.dtype
    else:
        return type_to_dtype(type(x))


_ordered_types = (bool, int, float, complex)


def check_fp_or_complex(
    dtype: torch.dtype, fn_name: str, allow_low_precision_dtypes: bool = True
):
    """
    Checks whether the input is floating point or complex.
    If allow_low_precision_dtypes is True, it allows having float16, bfloat16, and complex32
    """
    torch._check(
        is_float_dtype(dtype) or is_complex_dtype(dtype),
        lambda: f"{fn_name}: Expected a floating point or complex tensor as input. Got {dtype}",
    )
    torch._check(
        allow_low_precision_dtypes or not is_low_precision_dtype(dtype),
        lambda: f"{fn_name}: Half precision dtypes not supported. Got {dtype}",
    )


def check_is_matrix(A: TensorLikeType, f_name: str, arg_name: str = "A"):
    torch._check(
        len(A.shape) >= 2,
        lambda: f"{f_name}: The input tensor {arg_name} must have at least 2 dimensions.",
    )


def get_higher_type(a: type, b: type) -> type:
    """
    Returns the higher of the two given Number types.

    The types are ordered bool -> int -> float -> complex.
    """
    a, b = _maybe_get_pytype(a), _maybe_get_pytype(b)
    # Type checking
    if a not in _ordered_types or b not in _ordered_types:
        raise RuntimeError(f"Expected builtin numeric types, found {a}, {b}")

    if a is b:
        return a

    for typ in _ordered_types:
        if a is typ:
            return b
        if b is typ:
            return a

    raise ValueError("Unknown Python scalar type!")


# Returns the higher of two torch datatypes a and b or, if the two
#   are not ordered relative to each other, the next
#   higher datatype
def get_higher_dtype(
    a: Optional[Union[torch.dtype, TensorLikeType, NumberType]],
    b: Optional[Union[torch.dtype, TensorLikeType, NumberType]],
) -> Optional[torch.dtype]:
    """
    Computes the "lowest" datatype that is weakly
    "higher" than both a and b.
    """

    # Type checking
    assert a is None or isinstance(a, (torch.dtype, TensorLike, Number))
    assert b is None or isinstance(b, (torch.dtype, TensorLike, Number))

    def _extract_dtype(
        x: Optional[Union[torch.dtype, TensorLikeType, NumberType]],
    ) -> Optional[torch.dtype]:
        if x is None:
            return None
        if isinstance(x, torch.dtype):
            return x
        if isinstance(x, TensorLike):
            return x.dtype
        if isinstance(x, Number):
            return type_to_dtype(type(x))

        raise RuntimeError("Unexpected type given to _extract_dtype!")

    # pyrefly: ignore [bad-argument-type]
    a, b = _extract_dtype(a), _extract_dtype(b)

    if a is b:
        return a

    if a is None:
        return b

    if b is None:
        return a

    ordered_datatypes = (
        (torch.bool,),
        (torch.uint8, torch.int8),
        (torch.int16,),
        (torch.int32,),
        (torch.int64,),
        (torch.float16, torch.bfloat16),
        (torch.float32,),
        (torch.float64,),
        (torch.complex32,),
        (torch.complex64,),
        (torch.complex128,),
    )

    for idx, dtypes in enumerate(ordered_datatypes):
        if a in dtypes and b in dtypes:
            return ordered_datatypes[idx + 1][0]
        if a in dtypes:
            return b
        if b in dtypes:
            return a

    raise RuntimeError("Unexpected termination!")


def check_pin_memory(pin_memory: bool):
    torch._check_not_implemented(
        not pin_memory, lambda: "PrimTorch does not support pinned memory"
    )


def check_layout(layout: torch.layout):
    torch._check_not_implemented(
        layout == torch.strided, lambda: f"PrimTorch doesn't support layout={layout}"
    )


# TODO: maybe unify with can_cast_to?
def is_weakly_lesser_type(a: type, b: type) -> bool:
    """
    Compares two types, a and b, returning True if a is weakly "less" than b.

    The comparison is determined by the following type ordering: bool, int, float, complex.
    """

    a, b = _maybe_get_pytype(a), _maybe_get_pytype(b)

    if a not in _ordered_types or b not in _ordered_types:
        raise RuntimeError(f"Expected builtin numeric types, found {a}, {b}")

    for typ in _ordered_types:
        if a == typ:
            return True
        if b == typ:
            return False

    raise RuntimeError("Unexpected termination!")


def can_safe_cast_to(*, cast_to: torch.dtype, cast_from: torch.dtype) -> bool:
    for fn in (is_complex_dtype, is_float_dtype, is_integer_dtype, is_boolean_dtype):
        if fn(cast_to):
            return True
        if fn(cast_from):
            return False

    raise ValueError(f"Received unknown dtypes {cast_to}, {cast_from}!")


def check_same_dtype(*args):
    """
    Checks that all Tensors in args have the same device and that all Numbers have the
    same corresponding Python type.

    Raises a RuntimeError when:
      - args contains an object whose type is not Tensor or Number
      - two Tensors objects in args have different dtypes
      - two Number objects in args have different types
      - there are Tensors and Numbers in args, and one of those Tensors corresponding
          Python types is different from the type of one of those Numbers
    """
    full_dtype = None
    scalar_type = None

    # pyrefly: ignore [bad-assignment]
    for arg in args:
        if isinstance(arg, Number):
            # Scalar type checking is disabled (and may be removed in the future)
            continue
            # if scalar_type is None:
            #     scalar_type = type(arg)

            # if scalar_type is not type(arg):
            #     msg = (
            #         "Scalar of type "
            #         + str(type(arg))
            #         + " is not the expected type of "
            #         + str(scalar_type)
            #         + "!"
            #     )
            #     raise RuntimeError(msg)
        elif isinstance(arg, TensorLike):
            if full_dtype is None:
                full_dtype = arg.dtype
            if scalar_type is None:
                scalar_type = dtype_to_type(arg.dtype)

            if full_dtype is not arg.dtype:
                msg = (
                    "Tensor with dtype "
                    + str(arg.dtype)
                    + " is not the expected dtype of "
                    + str(full_dtype)
                    + "!"
                )
                raise RuntimeError(msg)

            arg_type = dtype_to_type(arg.dtype)
            if arg_type is not scalar_type:
                msg = (
                    "Tensor with corresponding Python type "
                    + str(arg_type)
                    + " is not the expected type of "
                    + str(scalar_type)
                    + "!"
                )
                raise RuntimeError(msg)
        else:
            msg = (
                "Unexpected type when checking for same dtype, " + str(type(arg)) + "!"
            )
            raise RuntimeError(msg)


# Maps datatypes to their computation types for elementwise operations
_computation_dtype_map = {
    torch.bfloat16: torch.float32,
    torch.float16: torch.float32,
    torch.complex32: torch.complex64,
}


def get_computation_dtype(dtype: torch.dtype) -> torch.dtype:
    return _computation_dtype_map.get(dtype, dtype)


_cpu_acc_type_map = {
    torch.bfloat16: torch.float64,
    torch.float16: torch.float64,
    torch.float32: torch.float64,
    torch.complex32: torch.complex128,
    torch.complex64: torch.complex128,
}


def get_acc_type(dtype: torch.dtype, device: torch.device) -> torch.dtype:
    # Equivalent to at::toAccumulateType, prefer computation_dtype where possible
    if device.type == "cpu":
        return _cpu_acc_type_map.get(dtype, dtype)
    else:
        return get_computation_dtype(dtype)


class ELEMENTWISE_TYPE_PROMOTION_KIND(Enum):
    DEFAULT = (0,)
    NO_OPMATH = (1,)
    INT_TO_FLOAT = (2,)
    ALWAYS_BOOL = (3,)
    COMPLEX_TO_FLOAT = (4,)
    BOOL_TO_LONG = (5,)


class REDUCTION_OUTPUT_TYPE_KIND(Enum):
    SAME = (0,)
    COMPLEX_TO_FLOAT = (1,)  # for complex types outputs corresponding real type
    KEEP_PROMOTED_TYPE = (2,)  # keep output in opmath type, needed for mean
    ALWAYS_BOOL = (3,)


# Describes the return type of the primitive:
#
#   - NEW, a new tensor is created
#   - VIEW, a view of an input tensor is returned
#   - INPLACE, one or more input tensors is modified
#
# these descriptors are mututally exclusive and exhaustive.
class RETURN_TYPE(Enum):
    NEW = (0,)
    VIEW = (1,)
    INPLACE = (2,)
    NONE = (3,)


# TODO: when NumberType contains the sym types, can simplify this
def number_type(
    x: Union[NumberType, torch.SymInt, torch.SymFloat, torch.SymBool],
) -> type:
    if isinstance(x, torch.SymInt):
        return int
    elif isinstance(x, torch.SymFloat):
        return float
    elif isinstance(x, torch.SymBool):
        return bool
    else:
        return type(x)


def expr_type(x: sympy.Basic) -> type:
    import sympy

    if x.kind is sympy.core.kind.BooleanKind:
        return bool
    elif x.is_integer:  # type: ignore[attr-defined]
        return int
    else:
        # NB: Not strictly correct, but we don't support SymPy complex or bool.
        return float


# TODO: document type promotion kinds
def elementwise_dtypes(
    *_args,
    type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND,
) -> tuple[torch.dtype, torch.dtype]:
    """
    Computes the computation and result dtypes for elementwise type promotion
    on the given arguments and with the given elementwise type promotion kind.

    Note that not all inputs to an elementwise operation necessarily participate in type promotion.
    For example, the "alpha" parameter of torch.add does not participate in type promotion,
    although it may be cast to the Python type corresponding to the computation dtype that
    the type promotion algorithm determines.

    Default elementwise type promotion, which all other type promotion kinds tweak (see below),
    first decides which of four ordered types to use:

    bool -> integer -> floating point -> complex

    The selected type is the "lowest" type in the above list such that all number arguments
    have a weakly "lower" type and all tensor arguments have a weakly lower corresponding
    type for their dtype.

    Once the type is determined, the particular result dtype is found. The dtypes are
    partially ordered as follows:

    bool -> uint8, int8 -> int16 -> int32 -> int64 ->
      float16, bfloat16 -> float32 -> float64 -> complex32 -> complex64 -> complex128

    The result dtype is selected by:
      - if no tensor's dtype has the same corresponding type as the one selected,
          then the result dtype is the (default) dtype corresponding to the selected type
          (for example, 1.5 + an integer tensor has a result dtype of the default floating point dtype)
      - if the result type is complex then the dtype is:
        -  the default complex dtype if there are no floating point or complex tensors
        -  if there are floating point or complex tensors with one or more dimensions, then
            the complex dtype corresponding to the highest corresponding complex dtype among those tensors
            (for example, double + cfloat -> cdouble)
        -  if there are only floating point or complex tensors with zero dimensions, then
            the complex dtype corresponding to the highest corresponding complex dtype among those tensors
      - if the first two cases do not apply, the result dtype is the highest dtype among
          all tensors with one or more dimensions of the output type, and if there are no such
          tensors then it's the highest dtype among all tensors with zero dimensions of the output type
          (for example, long + half -> half, even if the half tensor has zero dimensions)

    The "corresponding complex dtypes" are:
      float16    -> complex32
      bfloat16   -> complex64
      float32    -> complex64
      float64    -> complex128
      complex32  -> complex32
      complex64  
```



## High-Level Overview


This Python file contains 6 class(es) and 111 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_WorksWithInt`, `K`, `ELEMENTWISE_TYPE_PROMOTION_KIND`, `REDUCTION_OUTPUT_TYPE_KIND`, `RETURN_TYPE`, `CUDARngStateHelper`

**Functions defined**: `__add__`, `__radd__`, `__mul__`, `__rmul__`, `same_shape`, `_maybe_get_pytype`, `compare_tensor_meta`, `_check_strides_helper`, `check_significant_strides`, `check_all_strides`, `check_contiguous_sizes_strides`, `eval_eager`, `is_contiguous`, `eval_eager`, `is_channels_last_contiguous_2d`, `eval_eager`, `is_channels_last_contiguous_3d`, `eval_eager`, `validate_memory_format`, `is_contiguous_for_memory_format`

**Key imports**: annotations, operator, typing, warnings, Callable, Sequence, AbstractContextManager, nullcontext, Enum, reduce, deprecated, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_prims_common`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `operator`
- `typing`
- `warnings`
- `collections.abc`: Callable, Sequence
- `contextlib`: AbstractContextManager, nullcontext
- `enum`: Enum
- `functools`: reduce
- `typing_extensions`: deprecated
- `torch`
- `sympy`
- `torch.fx.experimental.symbolic_shapes`: guard_or_true
- `torch._subclasses.fake_tensor`: MetadataMismatchError
- `torch.utils._sympy.functions`: Max


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

Files in the same folder (`torch/_prims_common`):

- [`wrappers.py_docs.md`](./wrappers.py_docs.md)


## Cross-References

- **File Documentation**: `__init__.py_docs.md`
- **Keyword Index**: `__init__.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
