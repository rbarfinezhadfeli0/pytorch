# Documentation: `torch/_meta_registrations.py`

## File Metadata

- **Path**: `torch/_meta_registrations.py`
- **Size**: 270,455 bytes (264.12 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import math
from collections.abc import Callable, Sequence
from enum import Enum
from functools import wraps
from typing import Optional, TypeVar, Union
from typing_extensions import ParamSpec

import torch
import torch._prims_common as utils
from torch import SymBool, SymFloat, Tensor
from torch._decomp import (
    _add_op_to_registry,
    _convert_out_params,
    global_decomposition_table,
    meta_table,
)
from torch._ops import OpOverload
from torch._prims import _prim_elementwise_meta, ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND
from torch._prims_common import (
    BoolLike,
    corresponding_complex_dtype,
    corresponding_real_dtype,
    elementwise_dtypes,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    FloatLike,
    IntLike,
    make_contiguous_strides_for,
    Number,
    suggest_memory_format,
    TensorLike,
)
from torch._prims_common.wrappers import (
    _maybe_convert_to_dtype,
    _maybe_resize_out,
    _resize_output_check,
    _safe_copy_out,
    out_wrapper,
)
from torch._refs import _broadcast_shapes, _maybe_broadcast
from torch.fx.experimental import _config as exp_config
from torch.nn.functional import ScalingType, SwizzleType
from torch.utils import _pytree as pytree


_T = TypeVar("_T")
_P = ParamSpec("_P")

aten = torch.ops.aten

_meta_lib_dont_use_me_use_register_meta = torch.library.Library("aten", "IMPL", "Meta")
MODE_SUM, MODE_MEAN, MODE_MAX = range(3)


def ceil_div(a, b):
    return (a + b - 1) // b


def round_up(x, y):
    """Rounds up x to nearest multiple of y"""
    return ((x + y - 1) // y) * y


def register_meta(op) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    def wrapper(fn):
        fn = _convert_out_params(fn)

        def register(op):
            _add_op_to_registry(meta_table, op, fn)

        pytree.tree_map_(register, op)
        return fn

    return wrapper


def elementwise_meta(
    *args,
    type_promotion: ELEMENTWISE_TYPE_PROMOTION_KIND,
):
    # Perform type promotion, as this is expected from prim_metafunction
    _, result_dtype = utils.elementwise_dtypes(
        *args,
        type_promotion_kind=type_promotion,
    )
    args = [_maybe_convert_to_dtype(x, result_dtype) for x in args]

    # Broadcast
    args = _maybe_broadcast(*args)

    # Perform prim checks
    return _prim_elementwise_meta(
        *args, type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT
    )


def toRealValueType(dtype):
    from_complex = {
        torch.complex32: torch.half,
        torch.cfloat: torch.float,
        torch.cdouble: torch.double,
    }
    return from_complex.get(dtype, dtype)


def check_inplace_broadcast(self_shape, *args_shape):
    broadcasted_shape = tuple(_broadcast_shapes(self_shape, *args_shape))
    torch._check(
        broadcasted_shape == self_shape,
        lambda: f"output with shape {self_shape} doesn't match the broadcast shape {broadcasted_shape}",
    )


@register_meta([aten.linspace, aten.logspace])
@out_wrapper()
def meta_linspace_logspace(
    start,
    end,
    steps,
    base=None,
    dtype=None,
    device=None,
    layout=torch.strided,
    pin_memory=False,
    requires_grad=False,
):
    if isinstance(start, torch.Tensor):
        torch._check(
            start.dim() == 0,
            lambda: "linspace only supports 0-dimensional start and end tensors",
        )
    if isinstance(end, torch.Tensor):
        torch._check(
            end.dim() == 0,
            lambda: "linspace only supports 0-dimensional start and end tensors",
        )

    if any(isinstance(arg, complex) for arg in (start, end, steps)):
        default_complex_dtype = utils.corresponding_complex_dtype(
            torch.get_default_dtype()
        )
        if dtype is None:
            dtype = default_complex_dtype
        else:
            torch._check(
                utils.is_complex_dtype(dtype),
                lambda: f"linspace(): inferred dtype {default_complex_dtype} can't be safely cast to passed dtype {dtype}",
            )
    else:
        dtype = dtype or torch.get_default_dtype()
    assert isinstance(dtype, torch.dtype)

    # steps does not participate in the computation of the dtype
    torch._check_type(
        isinstance(steps, IntLike),
        lambda: f"received an invalid combination of arguments - got \
({type(start).__name__}, {type(end).__name__}, {type(steps).__name__})",
    )
    assert isinstance(steps, IntLike)  # for mypy
    torch._check(steps >= 0, lambda: "number of steps must be non-negative")

    return torch.empty(
        (steps,),  # type: ignore[arg-type]
        dtype=dtype,
        layout=layout,
        device="meta",
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


@register_meta([aten.take.default, aten.take.out])
@out_wrapper()
def meta_take(self, index):
    # Type and device checks
    torch._check(
        index.dtype == torch.long,
        lambda: f"take(): Expected a long tensor for index, but got {index.dtype}",
    )
    # Index checks
    torch._check_index(
        not (self.numel() == 0 and index.numel() != 0),
        lambda: "take(): tried to take from an empty tensor",
    )
    return self.new_empty(index.shape)


@register_meta([aten.linalg_cross.default, aten.linalg_cross.out])
@out_wrapper()
def linalg_cross(self, other, *, dim=-1):
    x_d = self.ndim
    y_d = other.ndim
    torch._check(
        x_d == y_d,
        lambda: "linalg.cross: inputs must have the same number of dimensions.",
    )
    torch._check(
        self.size(dim) == 3 and other.size(dim) == 3,
        lambda: (
            f"linalg.cross: inputs dimension {dim} must have length 3. "
            f"Got {self.size(dim)} and {other.size(dim)}"
        ),
    )
    out_shape = _broadcast_shapes(self.shape, other.shape)
    return self.new_empty(out_shape)


@register_meta(aten.linalg_matrix_exp)
@out_wrapper()
def linalg_matrix_exp(self):
    squareCheckInputs(self, "linalg.matrix_exp")
    checkFloatingOrComplex(self, "linalg.matrix_exp")
    return torch.empty_like(self, memory_format=torch.contiguous_format)


@register_meta(
    [aten.cummax.default, aten.cummax.out, aten.cummin.default, aten.cummin.out]
)
@out_wrapper("values", "indices")
def cummaxmin(self, dim):
    values = torch.empty(self.shape, device=self.device, dtype=self.dtype)
    indices = torch.empty(self.shape, device=self.device, dtype=torch.int64)
    if self.numel() != 0 and self.ndim != 0:
        # Checks that dim is within bounds
        maybe_wrap_dim(dim, self.ndim)
    return values, indices


@register_meta([aten.logcumsumexp.default, aten.logcumsumexp.out])
@out_wrapper()
def logcumsumexp(self, dim):
    # Checks that dim is within bounds
    maybe_wrap_dim(dim, self.ndim)
    return torch.empty_like(self, memory_format=torch.contiguous_format)


# Stride-related code from _exec_fft in aten/src/ATen/native/mkl/SpectralOps.cpp
# and aten/src/ATen/cuda/SpectralOps.cpp
#
# Although the actual FFT launch is different, all the permuting code appears
# to be the same
def _exec_fft(out, self, out_sizes, dim, *, forward):
    ndim = self.ndim
    signal_ndim = len(dim)
    batch_dims = ndim - signal_ndim

    # Permute dimensions so batch dimensions come first, and in stride order
    dim_permute = list(range(ndim))

    is_transformed_dim = [False for _ in range(ndim)]
    for d in dim:
        is_transformed_dim[d] = True

    # std::partition
    left, right = [], []
    for d in dim_permute:
        if not is_transformed_dim[d]:
            left.append(d)
        else:
            right.append(d)
    dim_permute = left + right
    batch_end = len(left)

    self_strides = self.stride()
    tmp = dim_permute[:batch_end]
    tmp.sort(key=lambda x: self_strides[x], reverse=True)
    dim_permute = tmp + dim_permute[batch_end:]
    input = self.permute(dim_permute)

    # Collapse batch dimensions into a single dimension
    batched_sizes = [-1] + list(input.shape[batch_dims:])
    input = input.reshape(batched_sizes)

    batch_size = input.size(0)
    batched_sizes[0] = batch_size
    batched_out_sizes = list(batched_sizes)
    for i in range(len(dim)):
        batched_out_sizes[i + 1] = out_sizes[dim[i]]
    out.resize_(batched_out_sizes, memory_format=torch.contiguous_format)

    # Inplace reshaping to original batch shape and inverting the dimension permutation
    out_strides = [0 for _ in range(ndim)]
    batch_numel = 1
    i = batch_dims - 1
    while i >= 0:
        out_strides[dim_permute[i]] = batch_numel * out.stride(0)
        batch_numel *= out_sizes[dim_permute[i]]
        i -= 1
    for i in range(batch_dims, ndim):
        out_strides[dim_permute[i]] = out.stride(1 + (i - batch_dims))
    out.as_strided_(out_sizes, out_strides, out.storage_offset())

    return out


def _sort_dims(self: Tensor, dim: list[int], exclude_last: bool = False):
    sorted_dims = list(dim)
    self_strides = self.stride()
    sorted_dims[: len(sorted_dims) - int(exclude_last)].sort(
        key=lambda i: self_strides[i]
    )
    return sorted_dims


# See _fft_c2c_cufft in aten/src/ATen/native/cuda/SpectralOps.cpp
# and _fft_c2c_mkl in aten/src/ATen/native/mkl/SpectralOps.cpp
@register_meta([aten._fft_c2c.default, aten._fft_c2c.out])
@out_wrapper()
def meta_fft_c2c(self, dim, normalization, forward):
    torch._check(self.dtype.is_complex)
    if not dim:
        return self.clone()

    sorted_dims = _sort_dims(self, dim)
    out = self.new_empty(self.size())
    return _exec_fft(out, self, self.size(), sorted_dims, forward=forward)


cufft_max_ndim = 3


def use_optimized_cufft_path(dim: list[int]):
    if len(dim) > cufft_max_ndim or (len(dim) >= 2 and dim[0] == 0 and dim[1] == 1):
        return False
    else:
        return True


@register_meta([aten._fft_r2c.default, aten._fft_r2c.out])
@out_wrapper()
def meta_fft_r2c(self, dim, normalization, onesided):
    torch._check(self.dtype.is_floating_point)
    input_sizes = list(self.size())
    out_sizes = list(input_sizes)
    last_dim = dim[-1]
    last_dim_halfsize = input_sizes[last_dim] // 2 + 1
    onesided_sizes = list(input_sizes)
    onesided_sizes[last_dim] = last_dim_halfsize

    if onesided:
        out_sizes[last_dim] = last_dim_halfsize

    if device_hint(self) == "cuda" or device_hint(self) == "xpu":
        # _fft_r2c_cufft in aten/src/ATen/native/cuda/SpectralOps.cpp
        # _fft_r2c_xpu in torch-xpu-ops/src/ATen/native/xpu/SpectralOps.cpp
        output = self.new_empty(
            out_sizes, dtype=utils.corresponding_complex_dtype(self.dtype)
        )

        working_tensor = self
        if device_hint(self) == "cuda" and use_optimized_cufft_path(dim):
            _exec_fft(output, working_tensor, out_sizes, dim, forward=True)
        else:
            # First do the R2C transform on the last dimension
            target_sizes = out_sizes if len(dim) == 1 else onesided_sizes
            _exec_fft(output, working_tensor, target_sizes, [last_dim], forward=True)
            if len(dim) > 1:
                working_tensor = self.new_empty(
                    out_sizes, dtype=utils.corresponding_complex_dtype(self.dtype)
                )

            # Then any remaining C2C transforms
            sorted_dims = dim[:-1]
            while sorted_dims:
                output, working_tensor = working_tensor, output
                strides = working_tensor.stride()
                sorted_dims.sort(
                    key=lambda i: strides[i], reverse=True
                )  # NB reverse!  Not sure if this is og bug
                max_dims = min(cufft_max_ndim, len(sorted_dims))
                last_dims = sorted_dims[len(sorted_dims) - max_dims :]
                _exec_fft(
                    output, working_tensor, onesided_sizes, last_dims, forward=True
                )
                sorted_dims = sorted_dims[: len(sorted_dims) - max_dims]

        if not onesided:
            if output.size(last_dim) != out_sizes[last_dim]:
                working_tensor.resize_(out_sizes, memory_format=torch.contiguous_format)
                output = working_tensor

        return output

    else:
        return self.new_empty(
            out_sizes, dtype=utils.corresponding_complex_dtype(self.dtype)
        )


@register_meta(aten.randperm.generator_out)
def meta_randperm(n, *, generator=None, out):
    return _maybe_resize_out(out, torch.Size([n]))


@register_meta(aten.randperm.default)
def meta_randperm_default(
    n,
    *,
    dtype=torch.long,
    layout=None,
    device=None,
    pin_memory=None,
):
    return torch.empty(
        n, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


@register_meta([aten.randint.default, aten.randint.out])
@out_wrapper()
def meta_randint(
    high,
    size,
    *,
    dtype=torch.long,
    layout=None,
    device=None,
    pin_memory=None,
):
    low = 0
    torch._check(
        high > low,
        lambda: f"random_ expects 'from' to be less than 'to', but got from={low} >= to={high}",
    )
    return torch.empty(
        size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


@register_meta([aten.randint.low, aten.randint.low_out])
@out_wrapper()
def meta_randint_low(
    low,
    high,
    size,
    *,
    dtype=torch.long,
    layout=None,
    device=None,
    pin_memory=None,
):
    torch._check(
        high > low,
        lambda: f"random_ expects 'from' to be less than 'to', but got from={low} >= to={high}",
    )
    return torch.empty(
        size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


@register_meta([aten.rand.default, aten.rand.out])
@out_wrapper()
def meta_rand_default(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    return torch.empty(
        size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


@register_meta([aten._fft_c2r.default, aten._fft_c2r.out])
@out_wrapper()
def meta_fft_c2r(self: Tensor, dim: list[int], normalization: int, lastdim: int):
    # _fft_c2r_mkl
    torch._check(self.dtype.is_complex)

    if device_hint(self) == "cuda":
        out_sizes = list(self.size())
        out_sizes[dim[-1]] = lastdim

        output = self.new_empty(out_sizes, dtype=toRealValueType(self.dtype))

        if use_optimized_cufft_path(dim):
            return _exec_fft(
                output,
                self.clone(memory_format=torch.contiguous_format),
                out_sizes,
                dim,
                forward=False,
            )
        else:
            # First complete any C2C transforms
            if len(dim) > 1:
                temp = meta_fft_c2c(self, dim[:-1], 0, lastdim)  # fft_norm_mode::none
            else:
                temp = self.clone(memory_format=torch.contiguous_format)
            return _exec_fft(output, temp, out_sizes, [dim[-1]], forward=False)

    else:
        input = self
        if len(dim) > 1:
            c2c_dims = dim[:-1]
            input = meta_fft_c2c(self, c2c_dims, normalization, forward=False)
            dim = dim[-1:]

        out_sizes = list(input.size())
        out_sizes[dim[-1]] = lastdim
        out = self.new_empty(out_sizes, dtype=toRealValueType(self.dtype))
        return _exec_fft(out, input, out_sizes, dim, forward=False)


@register_meta(aten.copy_.default)
def meta_copy_(self, src, non_blocking=False):
    # This code simulates the original decomp from inductor,
    # which runs most of the meta checks that we care about.
    # In theory, we should make this more robust by carefully
    # auditing our C++ copy_() kernel and copying the checks here.
    from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

    # TODO: Ideally, we'd insert a deferred runtime assert here, but if we are
    # calling an actual copy_, you'll get that automatically
    # https://github.com/pytorch/pytorch/issues/122477
    if (
        not free_unbacked_symbols(self) and torch._debug_has_internal_overlap(self) == 1
    ):  # 1 == MemOverlap::Yes
        raise RuntimeError(
            "more than one element of the written-to tensor refers to a single memory location"
        )

    if isinstance(src, Tensor):
        intermediate = src.to(self, non_blocking)
        if self.size() != intermediate.size():
            aten.expand_copy.default(intermediate, self.size())
    return self


def inferUnsqueezeGeometry(tensor, dim):
    result_sizes = list(tensor.size())
    result_strides = list(tensor.stride())
    # pyrefly: ignore [unsupported-operation]
    new_stride = 1 if dim >= tensor.dim() else result_sizes[dim] * result_strides[dim]
    # pyrefly: ignore [bad-argument-type]
    result_sizes.insert(dim, 1)
    # pyrefly: ignore [bad-argument-type]
    result_strides.insert(dim, new_stride)
    return result_sizes, result_strides


@register_meta(aten.unsqueeze_.default)
def meta_unsqueeze_(self, dim):
    dim = maybe_wrap_dim(dim, self.dim() + 1)
    g_sizes, g_strides = inferUnsqueezeGeometry(self, dim)
    self.as_strided_(g_sizes, g_strides)
    return self


@register_meta(aten._sparse_semi_structured_linear)
def meta_sparse_structured_linear(
    input: Tensor,
    weight: Tensor,
    _meta: Tensor,
    bias: Optional[Tensor] = None,
    _activation_opt: Optional[str] = None,
    out_dtype: Optional[torch.dtype] = None,
):
    output_sizes = list(input.shape)
    if bias is not None:
        assert weight.size(0) == bias.size(0), "output size mismatch"
    assert weight.size(1) == input.size(-1) / 2
    output_sizes[-1] = weight.size(0)

    # see: https://github.com/pytorch/pytorch/pull/114477#issuecomment-1830121375
    # We assume that we have already squashed the inputs into a 2-D tensor
    # Then, as the output is transposed, we need to propagate the transposed
    # stride information to the output tensor
    assert len(input.shape) == 2, "we can only handle the squashed input case"
    transposed_strides = (1, input.size(0))

    if out_dtype is not None:
        assert input.dtype == torch.int8 and out_dtype == torch.int32, (
            "out_dtype is only supported for i8i8->i32 linear operator"
        )
    output = input.new_empty(
        output_sizes,
        dtype=input.dtype if out_dtype is None else out_dtype,
    ).as_strided(output_sizes, transposed_strides)

    return output


@register_meta(aten._sparse_semi_structured_mm)
def meta_sparse_structured_mm(
    mat1: Tensor,
    mat1_meta: Tensor,
    mat2: Tensor,
    out_dtype: Optional[torch.dtype] = None,
):
    assert len(mat1.shape) == 2
    assert len(mat1_meta.shape) == 2
    assert len(mat2.shape) == 2
    assert mat1.size(1) == mat2.size(0) / 2
    output_sizes = [mat1.size(0), mat2.size(1)]

    if out_dtype is not None:
        assert mat2.dtype == torch.int8 and out_dtype == torch.int32, (
            "out_dtype is only supported for i8i8->i32 linear operator"
        )
    output = mat2.new_empty(
        output_sizes,
        dtype=mat2.dtype if out_dtype is None else out_dtype,
    )

    return output


@register_meta(aten._sparse_semi_structured_addmm)
def meta_sparse_structured_addmm(
    input: Tensor,
    mat1: Tensor,
    mat1_meta: Tensor,
    mat2: Tensor,
    *,
    alpha=1,
    beta=1,
    out_dtype: Optional[torch.dtype] = None,
):
    assert len(input.shape) == 1, (
        "only input broadcasted to columns of mat1 * mat2 product is supported"
    )
    assert len(mat1.shape) == 2
    assert len(mat1_meta.shape) == 2
    assert len(mat2.shape) == 2
    assert input.size(0) == mat1.size(0), (
        "only input broadcasted to columns of mat1 * mat2 product is supported"
    )
    assert mat1.size(1) == mat2.size(0) / 2
    output_sizes = [mat1.size(0), mat2.size(1)]

    if out_dtype is not None:
        assert mat2.dtype == torch.int8 and out_dtype == torch.int32, (
            "out_dtype is only supported for i8i8->i32 linear operator"
        )
    output = mat2.new_empty(
        output_sizes,
        dtype=mat2.dtype if out_dtype is None else out_dtype,
    )

    return output


@register_meta(aten._cslt_sparse_mm)
def meta__cslt_sparse_mm(
    compressed_A: torch.Tensor,
    dense_B: torch.Tensor,
    bias: Optional[Tensor] = None,
    alpha: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    transpose_result: bool = False,
    alg_id: int = 0,
    split_k: int = 1,
    split_k_mode: int = -1,
):
    assert dense_B.dtype in {
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int8,
        torch.float8_e4m3fn,
    }, "_cslt_sparse_mm only supports fp16, bf16, int8, and fp8e4m3"
    assert compressed_A.dtype == dense_B.dtype, "inputs must have the same dtype"
    assert len(dense_B.shape) == 2, "_cslt_sparse_mm only supports 2d inputs"

    is_8bit_input_type = compressed_A.dtype in [torch.int8, torch.float8_e4m3fn]

    if is_8bit_input_type:
        assert not dense_B.is_contiguous(), (
            "dense input must be transposed for 8bit dtypes"
        )

    n = dense_B.size(1)
    m = compressed_A.size(0)
    if bias is not None:
        assert m == bias.size(0)

    if out_dtype is not None:
        assert is_8bit_input_type and out_dtype in {
            torch.float16,
            torch.bfloat16,
            torch.int32,
            torch.float8_e4m3fn,
        }, (
            f"out_dtype is not supported for {compressed_A.dtype} x {dense_B.dtype} -> {out_dtype} matmul!"
        )
    output_shape = (n, m) if transpose_result else (m, n)
    return dense_B.new_empty(output_shape, dtype=out_dtype)


@register_meta(aten.index_reduce.default)
def meta_index_reduce(
    self: Tensor,
    dim: int,
    index: Tensor,
    source: torch.Tensor,
    reduce: str,
    *,
    include_self: bool = True,
) -> Tensor:
    return torch.empty_like(self, memory_format=torch.contiguous_format)


@register_meta(aten.index_reduce_.default)
def meta_index_reduce_(
    self: Tensor,
    dim: int,
    index: Tensor,
    source: torch.Tensor,
    reduce: str,
    *,
    include_self: bool = True,
) -> Tensor:
    return self


# Implementations below are taken from https://github.com/albanD/subclass_zoo/blob/main/python_meta_tensor.py
@out_wrapper()
@register_meta(aten.index_select.default)
def meta_index_select(self, dim, index):
    result_size = list(self.size())
    if self.dim() > 0:
        result_size[dim] = index.numel()
    return self.new_empty(result_size)


@register_meta(aten.segment_reduce.default)
def meta_segment_reduce(
    data: Tensor,
    reduce: str,
    *,
    lengths: Optional[Tensor] = None,
    indices: Optional[Tensor] = None,
    offsets: Optional[Tensor] = None,
    axis: int = 0,
    unsafe: bool = False,
    initial=None,
) -> Tensor:
    if indices is not None:
        raise NotImplementedError(
            "segment_reduce(): indices based reduction is not supported yet."
        )

    def segment_reduce_lengths_tensor(lengths_shape):
        return torch.empty(
            lengths_shape + data.shape[axis + 1 :],
            dtype=data.dtype,
            device="meta",
            memory_format=torch.contiguous_format,
        )

    if lengths is not None:
        return segment_reduce_lengths_tensor(lengths.shape)
    # FIXME should probably check that lengths and offset aren't both set, but
    # the ATen implementation neglects this too
    if offsets is not None:
        # lengths == torch.diff(offsets)
        lengths_shape = offsets.shape[:-1] + (offsets.shape[-1] - 1,)
        return segment_reduce_lengths_tensor(lengths_shape)
    raise RuntimeError("segment_reduce(): Either lengths or offsets must be defined.")


@register_meta([aten.max.default, aten.max.unary_out])
@out_wrapper()
def meta_max(self):
    return self.new_empty(())


@register_meta(aten.max.dim)
def meta_max_dim(self, dim, keepdim=False):
    dim = utils.reduction_dims(self.shape, (dim,))
    output_shape = _compute_reduction_shape(self, dim, keepdim)
    return (
        self.new_empty(output_shape),
        self.new_empty(output_shape, dtype=torch.long),
    )


@register_meta([aten.min.default, aten.min.unary_out])
@out_wrapper()
def meta_min(self):
    return self.new_empty(())


@register_meta(aten.min.dim)
def meta_min_dim(self, dim, keepdim=False):
    dim = utils.reduction_dims(self.shape, (dim,))
    output_shape = _compute_reduction_shape(self, dim, keepdim)
    return (
        self.new_empty(output_shape),
        self.new_empty(output_shape, dtype=torch.long),
    )


@register_meta(aten.angle.default)
def meta_angle(self):
    if self.is_complex():
        result_dtype = corresponding_real_dtype(self.dtype)
    else:
        _, result_dtype = elementwise_dtypes(
            self,
            type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        )
    return torch.empty_like(self, dtype=result_dtype)


@register_meta(aten.angle.out)
def meta_angle_out(self, out):
    torch._resize_output_(out, self.size(), self.device)
    return out.copy_(torch.angle(self))


@register_meta(aten._assert_async.default)
def assert_async(val):
    return


@register_meta(aten._assert_async.msg)
def assert_async_meta(val, assert_msg):
    return


@register_meta(aten._print.default)
def print_meta(s):
    return


@register_meta(aten._make_dep_token.default)
def make_dep_token(
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    return torch.empty(0, device="meta")


@register_meta(aten.sym_constrain_range.default)
def sym_constrain_range(size, min=None, max=None):
    # Avoid importing sympy at a module level
    from torch.fx.experimental.symbolic_shapes import constrain_range

    if isinstance(size, (SymFloat, SymBool)):
        raise ValueError("Constraining SymFloat or Symbool is nyi")
    constrain_range(size, min=min, max=max)


@register_meta(aten._functional_sym_constrain_range.default)
def functional_sym_constrain_range(size, min=None, max=None, dep_token=None):
    aten.sym_constrain_range(size, min=min, max=max)
    return dep_token


@register_meta(aten.sym_constrain_range_for_size.default)
def sym_constrain_range_for_size(size, min=None, max=None):
    # Avoid importing sympy at a module level
    from torch.fx.experimental.symbolic_shapes import _constrain_range_for_size

    if min is None and max is None:
        torch._check(size >= 0)
        return

    if isinstance(size, (SymFloat, SymBool)):
        raise ValueError("Constraining SymFloat or Symbool is nyi")
    if type(size) is int:
        if min is not None:
            torch._check(size >= min)
        if max is not None:
            torch._check(size <= max)
        return
    _constrain_range_for_size(size, min=min, max=max)


@register_meta(aten._functional_sym_constrain_range_for_size.default)
def functional_sym_constrain_range_for_size(size, min, max, dep_token):
    aten.sym_constrain_range_for_size(size, min=min, max=max)
    return dep_token


@register_meta(aten._functional_assert_async.msg)
def functional_assert_async_meta(val, assert_msg, dep_token):
    return dep_token


# From aten/src/ATen/native/LinearAlgebraUtils.h
def squareCheckInputs(self: Tensor, f_name: str):
    assert self.dim() >= 2, (
        f"{f_name}: The input tensor must have at least 2 dimensions."
    )
    assert self.size(-1) == self.size(-2), (
        f"{f_name}: A must be batches of square matrices, but they are {self.size(-2)} by {self.size(-1)} matrices"
    )


# Validates input shapes and devices
# for linear solve methods (solve, cholesky_solve, lu_solve, triangular_solve)
# From aten/src/ATen/native/LinearAlgebraUtils.h
def linearSolveCheckInputs(self: Tensor, A: Tensor, name: str):
    torch._check(
        self.device == A.device,
        lambda: (
            f"Expected b and A to be on the same device, but found b on "
            f"{self.device} and A on {A.device} instead."
        ),
    )

    torch._check(
        self.dtype == A.dtype,
        lambda: (
            f"Expected b and A to have the same dtype, but found b of type "
            f"{self.dtype} and A of type {A.dtype} instead."
        ),
    )

    torch._check(
        A.size(-1) == A.size(-2),
        lambda: (
            f"A must be batches of square matrices, "
            f"but they are {A.size(-2)} by {A.size(-1)} matrices"
        ),
    )

    torch._check(
        A.size(-1) == self.size(-2),
        lambda: (
            f"Incompatible matrix sizes for {name}: each A "
            f"matrix is {A.size(-1)} by {A.size(-1)}"
            f" but each b matrix is {self.size(-2)} by {self.size(-1)}"
        ),
    )


# From aten/src/ATen/native/LinearAlgebraUtils.h
def checkFloatingOrComplex(
    t: Tensor,
    f_name: str,
    allow_low_precision_dtypes: bool = True,
):
    dtype = t.dtype
    torch._check(
        t.is_floating_point() or t.is_complex(),
        lambda: f"{f_name}: Expected a floating point or complex tensor as input. Got {dtype}",
    )
    if not allow_low_precision_dtypes:
        torch._check(
            dtype in (torch.float, torch.double, torch.cfloat, torch.cdouble),
            lambda: f"{f_name}: Low precision dtypes not supported. Got {dtype}",
        )


# From aten/src/ATen/native/LinearAlgebraUtils.h
def checkIsMatrix(A: Tensor, f_name: str, arg_name: str = "A"):
    torch._check(
        A.dim() >= 2,
        lambda: f"{f_name}: The input tensor {arg_name} must have at least 2 dimensions.",
    )


def checkInputsSolver(A: Tensor, B: Tensor, left: bool, f_name: str):
    squareCheckInputs(A, f_name)
    checkIsMatrix(B, f_name)
    torch._check(
        A.size(-2) == B.size(-2) if left else A.size(-1) == B.size(-1),
        lambda: (
            f"{f_name}: Incompatible shapes of A and B for the equation "
            f"{'AX = B' if left else 'XA = B'}"
            f" ({A.size(-2)}x{A.size(-1)} and {B.size(-2)}x{B.size(-1)})"
        ),
    )


def checkSameDevice(
    fn_name: str,
    result: Tensor,
    input: Tensor,
    result_name: str = "result",
):
    torch._check(
        result.device == input.device,
        lambda: (
            f"{fn_name}: Expected {result_name} and input tensors to be on the same device, but got "
            f"{result_name} on {result.device} and input on {input.device}"
        ),
    )


def checkUplo(UPLO: str):
    UPLO_uppercase = UPLO.upper()
    torch._check(
        len(UPLO) == 1 and (UPLO_uppercase == "U" or UPLO_uppercase == "L"),
        lambda: f"Expected UPLO argument to be 'L' or 'U', but got {UPLO}",
    )


@register_meta([aten._linalg_eigh.default, aten._linalg_eigh.eigenvalues])
@out_wrapper("eigenvalues", "eigenvectors")
def meta__linalg_eigh(A: Tensor, UPLO: str = "L", compute_v: bool = True):
    squareCheckInputs(A, "linalg.eigh")
    checkUplo(UPLO)

    shape = list(A.shape)
    if compute_v:
        vecs = A.new_empty(shape)
        vecs.as_strided_(shape, make_contiguous_strides_for(shape, row_major=False))
    else:
        vecs = A.new_empty([0])

    shape.pop()
    vals = A.new_empty(shape, dtype=toRealValueType(A.dtype))

    return vals, vecs


@register_meta([aten._linalg_eigvals.default, aten.linalg_eigvals.out])
@out_wrapper()
def meta__linalg_eigvals(input: Tensor) -> Tensor:
    squareCheckInputs(input, "linalg.eigvals")
    complex_dtype = (
        input.dtype
        if utils.is_complex_dtype(input.dtype)
        else utils.corresponding_complex_dtype(input.dtype)
    )
    return input.new_empty(input.shape[:-1], dtype=complex_dtype)


@register_meta([aten.linalg_eig])
@out_wrapper("eigenvalues", "eigenvectors")
def meta_linalg_eig(input: Tensor):
    squareCheckInputs(input, "linalg.eig")
    complex_dtype = (
        input.dtype
        if utils.is_complex_dtype(input.dtype)
        else utils.corresponding_complex_dtype(input.dtype)
    )
    values = input.new_empty(input.shape[:-1], dtype=complex_dtype)
    vectors = input.new_empty(input.shape, dtype=complex_dtype)
    is_cuda = device_hint(input) == "cuda"
    vectors.as_strided_(
        input.shape, make_contiguous_strides_for(input.shape, row_major=is_cuda)
    )
    return values, vectors


def cloneBatchedColumnMajor(src: Tensor) -> Tensor:
    return src.mT.clone(memory_format=torch.contiguous_format).transpose(-2, -1)


@register_meta(aten._cholesky_solve_helper)
@out_wrapper()
def _cholesky_solve_helper(self: Tensor, A: Tensor, upper: bool) -> Tensor:
    return cloneBatchedColumnMajor(self)


@register_meta(aten.cholesky_solve)
@out_wrapper()
def cholesky_solve(self: Tensor, A: Tensor, upper: bool = False) -> Tensor:
    torch._check(
        self.ndim >= 2,
        lambda: f"b should have at least 2 dimensions, but has {self.ndim} dimensions instead",
    )
    torch._check(
        A.ndim >= 2,
        lambda: f"u should have at least 2 dimensions, but has {A.ndim} dimensions instead",
    )
    self_broadcasted, A_broadcasted = _linalg_broadcast_batch_dims_name(
        self, A, "cholesky_solve"
    )
    return _cholesky_solve_helper(self_broadcasted, A_broadcasted, upper)


@register_meta(aten.cholesky)
@out_wrapper()
def cholesky(self: Tensor, upper: bool = False) -> Tensor:
    if self.numel() == 0:
        return torch.empty_like(self, memory_format=torch.legacy_contiguous_format)
    squareCheckInputs(self, "cholesky")
    return cloneBatchedColumnMajor(self)


@register_meta(aten.cholesky_inverse)
@out_wrapper()
def cholesky_inverse(self: Tensor, upper: bool = False) -> Tensor:
    squareCheckInputs(self, "cholesky_inverse")
    return cloneBatchedColumnMajor(self)


# From aten/src/ATen/native/BatchLinearAlgebra.cpp
@register_meta(aten.linalg_cholesky_ex.default)
def linalg_cholesky_ex(A: Tensor, upper: bool = False, check_errors: bool = False):
    squareCheckInputs(A, "linalg.cholesky")
    checkFloatingOrComplex(A, "linalg.cholesky")

    A_shape = A.shape
    ndim = len(A_shape)

    # L
    L_strides = make_contiguous_strides_for(A_shape, False)
    L = A.new_empty(A_shape)
    L.as_strided_(A_shape, L_strides)

    # infos
    infos = A.new_empty(A_shape[0 : ndim - 2], dtype=torch.int32)
    return L, infos


@register_meta(
    [aten.linalg_householder_product.default, aten.linalg_householder_product.out]
)
@out_wrapper()
def linalg_householder_product(input: Tensor, tau: Tensor) -> Tensor:
    torch._check(
        input.ndim >= 2,
        lambda: "torch.linalg.householder_product: input must have at least 2 dimensions.",
    )
    torch._check(
        input.size(-2) >= input.size(-1),
        lambda: "torch.linalg.householder_product: input.shape[-2] must be greater than or equal to input.shape[-1]",
    )
    torch._check(
        input.size(-1) >= tau.size(-1),
        lambda: "torch.linalg.householder_product: input.shape[-1] must be greater than or equal to tau.shape[-1]",
    )

    torch._check(
        input.ndim - tau.ndim == 1,
        lambda: (
            f"torch.linalg.householder_product: Expected tau to have one dimension less than input, "
            f"but got tau.ndim equal to {tau.ndim} and input.ndim is equal to {input.ndim}"
        ),
    )
    if input.ndim > 2:
        expected_batch_tau_shape = input.shape[:-2]
        actual_batch_tau_shape = tau.shape[:-1]
        torch._check(
            actual_batch_tau_shape == expected_batch_tau_shape,
            lambda: (
                f"torch.linalg.householder_product: Expected batch dimensions of tau to be "
                f"equal to input.shape[:-2], but got {actual_batch_tau_shape}"
            ),
        )

    torch._check(
        tau.dtype == input.dtype,
        lambda: (
            f"torch.linalg.householder_product: tau dtype {tau.dtype}"
            f" does not match input dtype {input.dtype}"
        ),
    )
    checkSameDevice("torch.linalg.householder_product", tau, input, "tau")

    return torch.empty_strided(
        size=input.shape,
        stride=make_contiguous_strides_for(input.shape, row_major=False),
        dtype=input.dtype,
        device=input.device,
    )


# From aten/src/ATen/native/BatchLinearAlgebra.cpp
@register_meta(aten.linalg_inv_ex.default)
def linalg_inv_ex_meta(A: Tensor, check_errors: bool = False):
    squareCheckInputs(A, "linalg.inv_ex")
    checkFloatingOrComplex(A, "linalg.inv_ex", allow_low_precision_dtypes=False)

    L = A.new_empty(A.shape)
    L.as_strided_(A.shape, make_contiguous_strides_for(A.shape, row_major=False))

    infos = A.new_empty(A.shape[:-2], dtype=torch.int32)
    return L, infos


@register_meta([aten.linalg_ldl_factor_ex.default, aten.linalg_ldl_factor_ex.out])
@out_wrapper("LD", "pivots", "info")
def linalg_ldl_factor_ex_meta(
    self: Tensor,
    *,
    hermitian: bool = False,
    check_errors: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    squareCheckInputs(self, "torch.linalg.ldl_factor_ex")
    checkFloatingOrComplex(self, "torch.linalg.ldl_factor_ex")
    LD = torch.empty_strided(
        size=self.shape,
        stride=make_contiguous_strides_for(self.shape, row_major=False),
        dtype=self.dtype,
        device=self.device,
    )
    pivots = self.new_empty(self.shape[:-1], dtype=torch.int)
    info = self.new_empty(self.shape[:-2], dtype=torch.int)
    return LD, pivots, info


@register_meta([aten.linalg_ldl_solve.default, aten.linalg_ldl_solve.out])
@out_wrapper()
def linalg_ldl_solve_meta(
    LD: Tensor,
    pivots: Tensor,
    B: Tensor,
    *,
    hermitian: bool = False,
) -> Tensor:
    squareCheckInputs(LD, "torch.linalg.ldl_solve")
    checkFloatingOrComplex(LD, "torch.linalg.ldl_solve")
    linearSolveCheckInputs(B, LD, "torch.linalg.ldl_solve")
    torch._check(
        B.ndim >= 2,
        lambda: (
            f"torch.linalg.ldl_solve: Expected B to have at least 2 dimensions, "
            f"but it has {B.ndim} dimensions instead"
        ),
    )
    expected_pivots_shape = LD.shape[:-1]
    torch._check(
        expected_pivots_shape == pivots.shape,
        lambda: (
            f"torch.linalg.ldl_solve: Expected LD.shape[:-1] and pivots.shape to be the same, "
            f"but got pivots with shape {pivots.shape} instead"
        ),
    )
    torch._check(
        utils.is_integer_dtype(pivots.dtype),
        lambda: f"torch.linalg.ldl_solve: Expected pivots to be integers. Got {pivots.dtype}",
    )
    torch._check(
        LD.dtype == B.dtype,
        lambda: f"torch.linalg.ldl_solve: LD dtype {LD.dtype} does not match b dtype {B.dtype}",
    )
    B_broadcast_size, _ = _linalg_broadcast_batch_dims(B, LD)
    return torch.empty_strided(
        size=B_broadcast_size,
        stride=make_contiguous_strides_for(B_broadcast_size, row_major=False),
        dtype=B.dtype,
        device=B.device,
    )


@register_meta([aten.linalg_lu.default, aten.linalg_lu.out])
@out_wrapper("P", "L", "U")
def linalg_lu_meta(A: Tensor, *, pivot: bool = True) -> tuple[Tensor, Tensor, Tensor]:
    torch._check(
        A.ndim >= 2,
        lambda: f"linalg.lu: Expected tensor with 2 or more dimensions. Got size: {A.shape} instead",
    )

    sizes = list(A.shape)
    m = sizes[-2]
    n = sizes[-1]
    k = min(m, n)

    sizes[-1] = m
    if pivot:
        P = A.new_empty(sizes)
    else:
        P = A.new_empty([0])

    sizes[-1] = k
    L = A.new_empty(sizes)

    sizes[-2] = k
    sizes[-1] = n
    U = A.new_empty(sizes)
    return P, L, U


@register_meta([aten.linalg_lu_factor_ex.default, aten.linalg_lu_factor_ex.out])
@out_wrapper("LU", "pivots", "info")
def linalg_lu_factor_ex_meta(
    A: Tensor,
    *,
    pivot: bool = True,
    check_errors: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    torch._check(
        A.ndim >= 2,
        lambda: f"torch.lu_factor: Expected tensor with 2 or more dimensions. Got size: {A.shape} instead",
    )

    sizes = list(A.shape)
    m = sizes[-2]
    n = sizes[-1]

    LU = torch.empty_strided(
        size=sizes,
        stride=make_contiguous_strides_for(sizes, row_major=False),
        dtype=A.dtype,
        device=A.device,
    )

    # Sets sizes to the size of pivots
    sizes.pop()
    sizes[-1] = min(m, n)
    pivots = A.new_empty(sizes, dtype=torch.int)

    # Sets sizes to the size of info
    sizes.pop()
    info = A.new_empty(sizes, dtype=torch.int)

    return LU, pivots, info


@register_meta([aten.linalg_lu_solve.default, aten.linalg_lu_solve.out])
@out_wrapper()
def linalg_lu_solve_meta(
    LU: Tensor,
    pivots: Tensor,
    B: Tensor,
    *,
    left: bool = True,
    adjoint: bool = False,
) -> Tensor:
    # dtype
    checkFloatingOrComplex(LU, "torch.linalg.lu_solve")
    torch._check(
        LU.dtype == B.dtype,
        lambda: (
            f"linalg.lu_solve: Expected LU and B to have the same dtype, "
            f"but found LU of type {LU.dtype} and B of type {B.dtype} instead"
        ),
    )
    torch._check(
        pivots.dtype == torch.int,
        lambda: "linalg.lu_solve: pivots should be a Tensor of scalar type torch.int32",
    )

    # matrix shapes
    squareCheckInputs(LU, "torch.linalg.lu_solve")
    checkInputsSolver(LU, B, left, "linalg.lu_solve")
    torch._check(
        LU.size(-1) == pivots.size(-1),
        lambda: "linalg.lu_solve: Number of pivots per batch should be same as the dimension of the matrix",
    )

    # batches
    torch._check(
        LU.shape[:-1] == pivots.shape,
        lambda: (
            f"linalg.lu_solve: Expected LU.shape[:-1] and pivots.shape to be the same, "
            f"but got pivots with shape {pivots.shape} instead"
        ),
    )

    B_broadcast_size, _ = _linalg_broadcast_batch_dims(B, LU)

    result = torch.empty_strided(
        size=B_broadcast_size,
        stride=make_contiguous_strides_for(B_broadcast_size, row_major=not left),
        dtype=B.dtype,
        device=B.device,
    )

    if result.numel() != 0 and not left:
        if result.is_complex():
            result = result.conj()

    return result


@register_meta(aten.lu_unpack)
@out_wrapper("P", "L", "U")
def lu_unpack_meta(
    LU: Tensor,
    pivots: Tensor,
    unpack_data: bool = True,
    unpack_pivots: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    torch._check(
        LU.ndim >= 2,
        lambda: f"torch.lu_unpack: Expected tensor with 2 or more dimensions. Got size: {LU.shape} instead",
    )
    if unpack_pivots:
        torch._check(
            pivots.dtype == torch.int32,
            lambda: (
                "torch.lu_unpack: LU_pivots is expected to be a contiguous tensor of torch.int32 dtype.\n"
                "Note: this function is intended to be used with the output produced by torch.linalg.lu_factor"
            ),
        )
    sizes = list(LU.shape)
    m = sizes[-2]
    n = sizes[-1]
    k = min(m, n)
    sizes[-1] = m
    if unpack_pivots:
        P = LU.new_empty(sizes)
    else:
        P = LU.new_empty([0])
    if unpack_data:
        sizes[-1] = k
        L = LU.new_empty(sizes)
        sizes[-2] = k
        sizes[-1] = n
        U = LU.new_empty(sizes)
    else:
        L = LU.new_empty([0])
        U = LU.new_empty([0])
    return P, L, U


# parse the "mode" param in linalg_qr: return a tuple of bools (compute_q, reduced)
def _parse_qr_mode(mode: str) -> tuple[bool, bool]:
    if mode == "reduced":
        compute_q = True
        reduced = True
    elif mode == "complete":
        compute_q = True
        reduced = False
    elif mode == "r":
        compute_q = False
        reduced = True  # this is actually irrelevant in this mode
    else:
        torch._check(
            False,
            lambda: (
                f"qr received unrecognized mode '{mode}' "
                f"but expected one of 'reduced' (default), 'r', or 'complete'"
            ),
        )
    return compute_q, reduced  # type: ignore[possibly-undefined]


@register_meta([aten.linalg_qr.default, aten.linalg_qr.out])
@out_wrapper("Q", "R")
def linalg_qr_meta(A: Tensor, mode: str = "reduced") -> tuple[Tensor, Tensor]:
    checkIsMatrix(A, "linalg.qr")
    checkFloatingOrComplex(A, "linalg.qr")

    compute_q, reduced_mode = _parse_qr_mode(mode)

    m = A.shape[-2]
    n = A.shape[-1]
    k = min(m, n)

    if compute_q:
        Q_shape = list(A.shape)
        Q_shape[-1] = k if reduced_mode else m
        Q = A.new_empty(Q_shape)
        Q.as_strided_(Q_shape, make_contiguous_strides_for(Q_shape, row_major=False))
    else:
        Q = A.new_empty([0])

    # For readability
    R_shape = list(A.shape)
    R_shape[-2] = k if reduced_mode or not compute_q else m
    R = A.new_empty(R_shape)
    R.as_strided_(R_shape, make_contiguous_strides_for(R_shape, row_major=False))
    return Q, R


@register_meta([aten._linalg_slogdet.default, aten._linalg_slogdet.sign])
@out_wrapper("sign", "logabsdet", "LU", "pivots")
def _linalg_slogdet(A: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    squareCheckInputs(A, "linalg.slogdet")
    checkFloatingOrComplex(A, "linalg.slogdet", False)
    shape = A.shape
    sign = A.new_empty(shape[:-2])
    logabsdet = A.new_empty(shape[:-2], dtype=toRealValueType(A.dtype))
    LU = torch.empty_strided(
        size=shape,
        stride=make_contiguous_strides_for(shape, False),
        dtype=A.dtype,
        device=A.device,
    )
    pivots = A.new_empty(shape[:-1], dtype=torch.int32)
    return sign, logabsdet, LU, pivots


# From aten/src/ATen/native/BatchLinearAlgebra.cpp
# NOTE: matching defaults in aten/src/ATen/native/native_functions.yaml
@register_meta(aten._linalg_svd.default)
def _linalg_svd_meta(
    A: Tensor,
    full_matrices: bool = False,
    compute_uv: bool = True,
    driver: Optional[str] = None,
):
    checkIsMatrix(A, "linalg.svd")
    checkFloatingOrComplex(A, "linalg.svd")

    batch_dims = list(A.shape[:-2])
    m = A.shape[-2]
    n = A.shape[-1]
    k = min(m, n)

    if compute_uv:
        U_shape = batch_dims + [m, m if full_matrices else k]
        U = A.new_empty(U_shape)
        U.as_strided_(U_shape, make_contiguous_strides_for(U_shape, row_major=False))

        V_shape = batch_dims + [n if full_matrices else k, n]
        V = A.new_empty(V_shape)
        # NB: This checks for CUDA since there is no way to check for cuSolver.
        # Also, this might not work correctly on CPU when fake_device is not
        # available as device_hint just defaults to CUDA in that case. See
        # _linalg_svd meta in core.
        is_cuda = device_hint(A) == "cuda"
        V.as_strided_(V_shape, make_contiguous_strides_for(V_shape, row_major=is_cuda))
    else:
        # doesn't matter
        U = A.new_empty([0])
        V = A.new_empty([0])

    # S is always real, even when A is complex.
    S = A.new_empty(batch_dims + [k], dtype=toRealValueType(A.dtype))
    return U, S, V


def _linalg_broadcast_batch_dims(
    arg1: Tensor,
    arg2: Tensor,
) -> tuple[list[int], list[int]]:
    # broadcast the batch dimensions of arg1 and arg2.
    arg1_batch_sizes = arg1.shape[:-2]
    arg2_batch_sizes = arg2.shape[:-2]
    expand_batch_portion = _broadcast_shapes(arg1_batch_sizes, arg2_batch_sizes)

    arg1_expand_size = list(expand_batch_portion)
    arg1_expand_size += [arg1.size(-2), arg1.size(-1)]

    arg2_expand_size = list(expand_batch_portion)
    arg2_expand_size += [arg2.size(-2), arg2.size(-1)]
    return arg1_expand_size, arg2_expand_size


def _linalg_broadcast_batch_dims_name(
    arg1: Tensor,
    arg2: Tensor,
    name: Optional[str],
) -> tuple[Tensor, Tensor]:
    # If there's no name we assume we don't want to check the errors
    if name:
        linearSolveCheckInputs(arg1, arg2, name)

    arg1_expand_size, arg2_expand_size = _linalg_broadcast_batch_dims(arg1, arg2)

    arg1_broadcasted = (
        arg1 if arg1_expand_size == arg1.shape else arg1.expand(arg1_expand_size)
    )
    arg2_broadcasted = (
        arg2 if arg2_expand_size == arg2.shape else arg2.expand(arg2_expand_size)
    )
    return arg1_broadcasted, arg2_broadcasted


def linalg_solve_is_vector_rhs(input: Tensor, other: Tensor) -> bool:
    expected_batched_rhs_shape = input.shape[:-1]
    vector_case = other.ndim == 1 or (
        input.ndim - 1 == other.ndim and other.shape == expected_batched_rhs_shape
    )
    return vector_case


@register_meta(aten._linalg_solve_ex)
def _linalg_solve_ex(
    A: Tensor,
    B: Tensor,
    *,
    left: bool = True,
    check_errors: bool = False,
    result: Optional[Tensor] = None,
    LU: Optional[Tensor] = None,
    pivots: Optional[Tensor] = None,
    info: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    checkFloatingOrComplex(A, "linalg.solve")
    torch._check(
        A.dtype == B.dtype,
        lambda: (
            f"linalg.solve: Expected A and B to have the same dtype, but found A of type "
            f"{A.dtype} and B of type {B.dtype} instead"
        ),
    )
    vector_case = linalg_solve_is_vector_rhs(A, B)
    B_ = B.unsqueeze(-1) if vector_case else B
    checkInputsSolver(A, B_, left, "linalg.solve")
    B_broad_shape, _ = _linalg_broadcast_batch_dims(B_, A)
    torch._check(
        left or not vector_case,
        lambda: (
            "linalg.solve: Vector broadcasting of the left hand side is not supported for left=False. "
            "In this case linalg.solve is equivalent to B / A.squeeze(-1)"
        ),
    )
    result_shape = B_broad_shape[:-1] if vector_case else B_broad_shape
    result_ = torch.empty_strided(
        size=result_shape,
        stride=make_contiguous_strides_for(result_shape, not left),
        dtype=B.dtype,
        device=B.device,
    )
    shape = A.shape
    LU_ = torch.empty_strided(
        size=shape,
        stride=make_contiguous_strides_for(shape, False),
        dtype=A.dtype,
        device=A.device,
    )
    pivots_ = A.new_empty(shape[:-1], dtype=torch.int32)
    info_ = A.new_empty(shape[:-2], dtype=torch.int32)
    out = (result, LU, pivots, info)
    res = (result_, LU_, pivots_, info_)
    if all(x is not None for x in out):
        for r, o in zip(res, out):
            # resize and copy operations are done in-place
            _maybe_resize_out(o, r.shape)  # type:
```



## High-Level Overview


This Python file contains 1 class(es) and 368 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `GridSamplerInterpolation`

**Functions defined**: `ceil_div`, `round_up`, `register_meta`, `wrapper`, `register`, `elementwise_meta`, `toRealValueType`, `check_inplace_broadcast`, `meta_linspace_logspace`, `meta_take`, `linalg_cross`, `linalg_matrix_exp`, `cummaxmin`, `logcumsumexp`, `_exec_fft`, `_sort_dims`, `meta_fft_c2c`, `use_optimized_cufft_path`, `meta_fft_r2c`, `meta_randperm`

**Key imports**: math, Callable, Sequence, Enum, wraps, Optional, TypeVar, Union, ParamSpec, torch, torch._prims_common as utils, SymBool, SymFloat, Tensor, OpOverload


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `math`
- `collections.abc`: Callable, Sequence
- `enum`: Enum
- `functools`: wraps
- `typing`: Optional, TypeVar, Union
- `typing_extensions`: ParamSpec
- `torch`
- `torch._prims_common as utils`
- `torch._ops`: OpOverload
- `torch._prims`: _prim_elementwise_meta, ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND
- `torch._refs`: _broadcast_shapes, _maybe_broadcast
- `torch.fx.experimental`: _config as exp_config
- `torch.nn.functional`: ScalingType, SwizzleType
- `torch.utils`: _pytree as pytree
- `torch.fx.experimental.symbolic_shapes`: free_unbacked_symbols
- `torch._refs as refs  `
- `cycle in mypy`
- `torch._refs.nn.functional`


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`torch`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_tensor_docs.py_docs.md`](./_tensor_docs.py_docs.md)
- [`_classes.py_docs.md`](./_classes.py_docs.md)
- [`types.py_docs.md`](./types.py_docs.md)
- [`_appdirs.py_docs.md`](./_appdirs.py_docs.md)
- [`_tensor.py_docs.md`](./_tensor.py_docs.md)
- [`_streambase.py_docs.md`](./_streambase.py_docs.md)
- [`_lowrank.py_docs.md`](./_lowrank.py_docs.md)
- [`_size_docs.py_docs.md`](./_size_docs.py_docs.md)


## Cross-References

- **File Documentation**: `_meta_registrations.py_docs.md`
- **Keyword Index**: `_meta_registrations.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
