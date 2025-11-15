# Documentation: `torch/_decomp/decompositions.py`

## File Metadata

- **Path**: `torch/_decomp/decompositions.py`
- **Size**: 181,990 bytes (177.72 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import functools
import itertools
import numbers
import operator
import sys
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from enum import Enum
from functools import partial, reduce
from itertools import chain, product
from typing import Any, cast, Optional, Union

import torch
import torch._meta_registrations
import torch._prims as prims
import torch._prims_common as utils
import torch.nn.functional as F
from torch import sym_float, sym_int, Tensor
from torch._decomp import register_decomposition
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import (
    IntLike,
    NumberType,
    suggest_memory_format,
    TensorLike,
    TensorSequenceType,
)
from torch._prims_common.wrappers import (
    _maybe_convert_to_dtype,
    _maybe_resize_out,
    _safe_copy_out,
    out_wrapper,
)
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map


DispatchKey = torch._C.DispatchKey  # type: ignore[attr-defined]

# None of these functions are publicly accessible; get at them
# from torch._decomps
__all__: list[str] = []

aten = torch._ops.ops.aten


class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2


# This wraps a decomposition and performs various type promotion logic within it, depending on the strategy provided
# We're currently reusing ELEMENTWISE_TYPE_PROMOTION_KIND, although some of the usages are on non-elementwise ops
# Will need to validate the non-elementwise uses
def type_casts(
    f: Callable,
    type_promotion: utils.ELEMENTWISE_TYPE_PROMOTION_KIND,
    compute_dtype_only: bool = False,
    include_non_tensor_args: bool = False,
):
    @functools.wraps(f)
    def inner(*args, **kwargs):
        allowed_types = (
            (Tensor, torch.types._Number) if include_non_tensor_args else (Tensor,)
        )  # type: ignore[arg-type]
        flat_args = [
            x
            for x in pytree.arg_tree_leaves(*args, **kwargs)
            if isinstance(x, allowed_types)
        ]
        computation_dtype, result_dtype = utils.elementwise_dtypes(
            *flat_args, type_promotion_kind=type_promotion
        )

        # TODO: pretty sure this is not quite right
        def increase_prec(x):
            if isinstance(x, Tensor):
                return x.to(computation_dtype)
            else:
                return x

        def decrease_prec(x):
            if isinstance(x, Tensor):
                return x.to(result_dtype)
            else:
                return x

        r = f(*tree_map(increase_prec, args), **tree_map(increase_prec, kwargs))
        if compute_dtype_only:
            return r
        else:
            return tree_map(decrease_prec, r)

    return inner


compute_only_pw_cast_for_opmath = partial(
    type_casts,
    type_promotion=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    compute_dtype_only=True,
)
pw_cast_for_opmath = partial(
    type_casts, type_promotion=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
)
pw_cast_for_opmath_non_tensor_args = partial(
    type_casts,
    type_promotion=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    include_non_tensor_args=True,
)
pw_cast_for_int_to_real = partial(
    type_casts, type_promotion=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)


# This expands x until x.dim() == dim. Might be useful as an operator
def _unsqueeze_to_dim(x: Tensor, dim: int) -> Tensor:
    for _ in range(dim - x.dim()):
        x = x.unsqueeze(-1)
    return x


@register_decomposition(aten.tanh_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def tanh_backward(out_grad: Tensor, y: Tensor):
    return out_grad * (1 - y * y).conj_physical()


@register_decomposition(aten.sigmoid_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def sigmoid_backward(out_grad: Tensor, y: Tensor):
    return out_grad * (y * (1 - y)).conj_physical()


@register_decomposition(aten.softplus_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def softplus_backward(out_grad: Tensor, x: Tensor, beta: float, threshold: float):
    z = (x * beta).exp()
    return torch.where((x * beta) > threshold, out_grad, out_grad * z / (z + 1.0))


@register_decomposition(aten.elu_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def elu_backward(
    grad_output: Tensor,
    alpha: float,
    scale: float,
    input_scale: float,
    is_result: bool,
    self_or_result: Tensor,
):
    negcoef = alpha * scale
    poscoef = scale
    negiptcoef = input_scale
    if is_result:
        return torch.where(
            self_or_result <= 0,
            grad_output * negiptcoef * (self_or_result + negcoef),
            grad_output * poscoef,
        )
    else:
        return torch.where(
            self_or_result <= 0,
            grad_output * negiptcoef * negcoef * torch.exp(self_or_result * negiptcoef),
            grad_output * poscoef,
        )


@register_decomposition([aten.fill.Scalar])
def fill_scalar(self, value):
    return torch.full_like(self, value)


@register_decomposition([aten.fill.Tensor])
def fill_tensor(self, value: Tensor):
    torch._check(
        value.dim() == 0,
        lambda: f"fill only supports 0-dimension value tensor but got tensor with {value.dim()} dimensions",
    )
    return aten.copy(self, value)


@register_decomposition(aten.hardsigmoid)
@out_wrapper()
@pw_cast_for_opmath
def hardsigmoid(self: Tensor) -> Tensor:
    return torch.clamp(torch.clamp(self + 3, min=0), max=6) / 6


@register_decomposition(aten.hardsigmoid_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def hardsigmoid_backward(grad_output: Tensor, self: Tensor):
    return torch.where(
        (self > -3.0) & (self < 3.0),
        grad_output * (1.0 / 6.0),
        0.0,
    )


@register_decomposition(aten.hardtanh_backward)
@out_wrapper("grad_input")
def hardtanh_backward(
    grad_output: Tensor, self: Tensor, min_val: float, max_val: float
):
    return torch.where((self <= min_val) | (self >= max_val), 0.0, grad_output)


@register_decomposition(aten.hardswish)
@out_wrapper()
@pw_cast_for_opmath
def hardswish(self: Tensor) -> Tensor:
    return self * torch.clamp(torch.clamp(self + 3, min=0), max=6) / 6


@register_decomposition(aten.hardswish_backward)
@out_wrapper()
@pw_cast_for_opmath
def hardswish_backward(grad_output: Tensor, self: Tensor) -> Tensor:
    return torch.where(
        self <= -3,
        0.0,
        torch.where(self < 3, grad_output * ((self / 3) + 0.5), grad_output),
    )


@register_decomposition(aten.threshold_backward)
@out_wrapper("grad_input")
def threshold_backward(grad_output: Tensor, self: Tensor, threshold: float):
    return torch.where(self <= threshold, 0, grad_output)


@register_decomposition(aten.leaky_relu_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def leaky_relu_backward(
    grad_output: Tensor, self: Tensor, negative_slope: float, self_is_result: bool
):
    return torch.where(self > 0, grad_output, grad_output * negative_slope)


@register_decomposition(aten.gelu_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def gelu_backward(grad: Tensor, self: Tensor, approximate: str = "none"):
    M_SQRT2 = 1.41421356237309504880
    M_SQRT1_2 = 0.70710678118654752440
    M_2_SQRTPI = 1.12837916709551257390
    if approximate == "tanh":
        kBeta = M_SQRT2 * M_2_SQRTPI * 0.5
        kKappa = 0.044715
        x_sq = self * self
        x_cube = x_sq * self
        inner = kBeta * (self + kKappa * x_cube)
        tanh_inner = torch.tanh(inner)

        left = 0.5 * self
        right = 1 + tanh_inner

        left_derivative = 0.5 * right

        tanh_derivative = 1 - tanh_inner * tanh_inner
        inner_derivative = kBeta * (1 + 3 * kKappa * x_sq)
        right_derivative = left * tanh_derivative * inner_derivative

        return grad * (left_derivative + right_derivative)
    else:
        kAlpha = M_SQRT1_2
        kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5
        cdf = 0.5 * (1 + torch.erf(self * kAlpha))
        pdf = kBeta * torch.exp(self * self * -0.5)
        return grad * (cdf + self * pdf)


@register_decomposition(aten.mish_backward)
@pw_cast_for_opmath
def mish_backward(grad_output: Tensor, input: Tensor):
    input_tanh_softplus = torch.tanh(F.softplus(input))
    input_sigmoid = torch.sigmoid(input)
    out = input * input_sigmoid * (1 - input_tanh_softplus * input_tanh_softplus)
    return grad_output * (input_tanh_softplus + out)


@register_decomposition(aten.silu)
@out_wrapper()
@pw_cast_for_opmath
def silu(self: Tensor) -> Tensor:
    return self * torch.sigmoid(self)


@register_decomposition(aten.silu_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def silu_backward(grad_output: Tensor, self: Tensor) -> Tensor:
    sigmoid = 1 / (1 + torch.exp(-self))
    return grad_output * sigmoid * (1 + self * (1 - sigmoid))


@register_decomposition(aten._prelu_kernel)
def _prelu_kernel(self: Tensor, weight: Tensor) -> Tensor:
    return torch.where(self > 0, self, weight * self)


@register_decomposition(aten._prelu_kernel_backward)
def _prelu_kernel_backward(
    grad_output: Tensor,
    self: Tensor,
    weight: Tensor,
) -> tuple[Tensor, Tensor]:
    input_grad = torch.where(self > 0, grad_output, weight * grad_output)
    weight_grad = torch.where(self > 0, 0.0, self * grad_output)
    return (input_grad, weight_grad)


@register_decomposition(aten.rrelu_with_noise_backward)
@out_wrapper()
@pw_cast_for_opmath
def rrelu_with_noise_backward(
    grad_output: Tensor,
    self: Tensor,
    noise: Tensor,
    lower: float,
    upper: float,
    training: bool,
    self_is_result: bool,
) -> Tensor:
    if training and upper - lower > 1e-6:
        return grad_output.mul(noise)
    else:
        negative_slope = (lower + upper) / 2
        return aten.leaky_relu_backward(
            grad_output, self, negative_slope, self_is_result
        )


@register_decomposition(aten.log_sigmoid_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def log_sigmoid_backward(grad_output: Tensor, self: Tensor, buffer: Tensor) -> Tensor:
    in_negative = self < 0
    max_deriv = torch.where(in_negative, 1, 0)
    sign = torch.where(in_negative, 1, -1)
    z = torch.exp(-torch.abs(self))
    return grad_output * (max_deriv - sign * (z / (1 + z)))
    # CPU has a special formula that uses buffer, but disabled for convenience sake
    # return (max_deriv - sign * (buffer / (1 + buffer))) * grad_output


def apply_loss_reduction(loss: Tensor, reduction: int):
    if reduction == Reduction.MEAN.value:
        return torch.mean(loss)
    elif reduction == Reduction.SUM.value:
        return torch.sum(loss)
    else:
        return loss


def to_real_dtype(dtype: torch.dtype):
    if dtype == torch.complex32:
        return torch.float16
    elif dtype == torch.complex64:
        return torch.float32
    elif dtype == torch.complex128:
        return torch.float64


# TODO: None of these loss castings are quite correct, see
# https://github.com/pytorch/pytorch/issues/76870. Also, the ATen kernels
# perform the pointwise portion in opmath, but don't maintain it between the
# pointwise portion and the reduction


@register_decomposition(aten.mse_loss)
@out_wrapper()
@pw_cast_for_opmath
def mse_loss(
    self: Tensor, target: Tensor, reduction: int = Reduction.MEAN.value
) -> Tensor:
    # pyrefly: ignore [unsupported-operation]
    loss = (self - target) ** 2
    return apply_loss_reduction(loss, reduction)


@register_decomposition(aten.mse_loss_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def mse_loss_backward(
    grad_output: Tensor, input: Tensor, target: Tensor, reduction: int
):
    norm = 2.0 / input.numel() if reduction == Reduction.MEAN.value else 2.0
    return norm * (input - target) * grad_output


@register_decomposition(aten._safe_softmax)
def safe_softmax(self, dim, dtype=None):
    out = torch.softmax(self, dim=dim, dtype=dtype)
    masked = self.eq(float("-inf"))
    masked_rows = torch.all(masked, dim=dim, keepdim=True)
    zeros = torch.zeros_like(out)
    return torch.where(masked_rows, zeros, out)


@register_decomposition(aten.smooth_l1_loss)
@out_wrapper()
@pw_cast_for_opmath
def smooth_l1_loss(
    self: Tensor,
    target: Tensor,
    reduction: int = Reduction.MEAN.value,
    beta: float = 1.0,
):
    loss = (self - target).abs()
    # pyrefly: ignore [unsupported-operation]
    loss = torch.where(loss < beta, 0.5 * loss**2 / beta, loss - 0.5 * beta)
    return apply_loss_reduction(loss, reduction)


@register_decomposition(aten.smooth_l1_loss_backward.default)
@pw_cast_for_opmath
def smooth_l1_loss_backward(
    grad_output: Tensor, self: Tensor, target: Tensor, reduction: int, beta: float
):
    norm = 1.0 / self.numel() if reduction == Reduction.MEAN.value else 1.0
    x = self - target
    abs_x = torch.abs(x)
    norm_grad = norm * grad_output
    return torch.where(
        abs_x < beta,
        norm_grad * x / beta,
        norm_grad * torch.sign(x),
    )


@register_decomposition(aten.smooth_l1_loss_backward.grad_input)
@pw_cast_for_opmath
def smooth_l1_loss_backward_out(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    reduction: int,
    beta: float,
    grad_input: Tensor,
):
    result = smooth_l1_loss_backward(grad_output, self, target, reduction, beta)
    _maybe_resize_out(grad_input, result.shape)
    return _safe_copy_out(copy_from=result, copy_to=grad_input, exact_dtype=True)


@register_decomposition(aten.huber_loss_backward.default)
@pw_cast_for_opmath
def huber_loss_backward(
    grad_output: Tensor, self: Tensor, target: Tensor, reduction: int, delta: float
):
    norm = 1.0 / self.numel() if reduction == Reduction.MEAN.value else 1.0
    x = self - target
    return torch.where(
        x < -delta,
        -norm * grad_output * delta,
        torch.where(x > delta, norm * grad_output * delta, norm * x * grad_output),
    )


# We cannot use @out_wrapper() here, because the output tensor is not named 'out', it's 'grad_input'
@register_decomposition(aten.huber_loss_backward.out)
@pw_cast_for_opmath
def huber_loss_backward_out(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    reduction: int,
    delta: float,
    grad_input: Tensor,
):
    result = huber_loss_backward(grad_output, self, target, reduction, delta)
    _maybe_resize_out(grad_input, result.shape)
    return _safe_copy_out(copy_from=result, copy_to=grad_input, exact_dtype=True)


def _nll_loss_backward(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    reduction: int,
    ignore_index: int,
    total_weight: Tensor,
) -> Tensor:
    channel_dim = 0 if self.dim() < 2 else 1
    if reduction == Reduction.MEAN.value:
        grad_output = grad_output / total_weight

    target = target.unsqueeze(channel_dim)
    safe_target = torch.where(target != ignore_index, target, 0)
    grad_input = torch.zeros_like(self)
    grad_input = torch.scatter(grad_input, channel_dim, safe_target, -1.0)

    if grad_input.dim() > grad_output.dim() > 0:
        grad_output = grad_output.unsqueeze(channel_dim)

    if weight is not None:
        new_shape = [1 for _ in range(self.dim())]
        new_shape[channel_dim] = weight.shape[0]
        weight = weight.reshape(new_shape)
        grad_output = grad_output * weight

    grad_output = torch.where(target != ignore_index, grad_output, 0)

    return grad_input * grad_output


@register_decomposition(aten.glu_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def glu_backward(grad_output: Tensor, self: Tensor, dim: int) -> Tensor:
    assert self.dim() > 0, "glu does not support 0-dimensional tensors"
    wrap_dim = utils.canonicalize_dim(self.dim(), dim)
    nIn = self.size(wrap_dim)
    assert nIn % 2 == 0, (
        f"Halving dimension must be even, but dimension {wrap_dim} is size {nIn}"
    )
    inputSize = nIn // 2
    firstHalf = self.narrow(wrap_dim, 0, inputSize)
    secondHalf = self.narrow(wrap_dim, inputSize, inputSize)
    gradInputFirstHalf = torch.sigmoid(secondHalf)
    gradInputSecondHalf = (
        (1.0 - gradInputFirstHalf) * gradInputFirstHalf * firstHalf * grad_output
    )
    gradInputFirstHalf = gradInputFirstHalf * grad_output
    return torch.cat([gradInputFirstHalf, gradInputSecondHalf], dim=wrap_dim)


@register_decomposition(aten.nll_loss_backward)
@out_wrapper("grad_input")
def nll_loss_backward(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    reduction: int,
    ignore_index: int,
    total_weight: Tensor,
) -> Tensor:
    assert 0 <= self.dim() <= 2, "input tensor should be 1D or 2D"
    assert target.dim() <= 1, (
        "0D or 1D target tensor expected, multi-target not supported"
    )

    no_batch_dim = self.dim() == 1 and target.dim() == 0
    assert no_batch_dim or (self.shape[0] == target.shape[0]), (
        f"size mismatch (got input: {self.shape}, target: {target.shape})"
    )
    assert total_weight.numel() == 1, (
        "expected total_weight to be a single element tensor, got: ",
        f"{total_weight.shape} ({total_weight.numel()} elements)",
    )

    assert weight is None or weight.numel() == self.shape[-1], (
        "weight tensor should be defined either for all or no classes"
    )

    if reduction == Reduction.NONE.value and self.dim() == 2:
        assert grad_output.dim() == 1 and grad_output.shape[0] == self.shape[0], (
            f"Expected a tensor of dimension 1 and tensor.size[0] == {self.shape[0]} but "
            f"got: dimension {grad_output.dim()} and tensor.size[0] == {grad_output.shape[0]}"
        )
    else:
        assert grad_output.dim() <= 1 and grad_output.numel() == 1, (
            f"Expected a single element grad_output tensor, but got: {grad_output.shape}"
        )

    return _nll_loss_backward(
        grad_output, self, target, weight, reduction, ignore_index, total_weight
    )


@register_decomposition(aten.nll_loss2d_backward)
@out_wrapper("grad_input")
def nll_loss2d_backward(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    reduction: int,
    ignore_index: int,
    total_weight: Tensor,
) -> Tensor:
    assert self.dim() == 4, (
        f"only batches of spatial inputs supported (4D tensors), but got input of dimension: {self.dim()}"
    )

    assert target.dim() == 3, (
        f"only batches of spatial targets supported (3D tensors) but got targets of dimension: {target.dim()}"
    )

    assert (
        self.shape[0] == target.shape[0]
        and self.shape[2] == target.shape[1]
        and self.shape[3] == target.shape[2]
    ), f"size mismatch (got input: {self.shape}, target: {target.shape}"

    assert total_weight.numel() == 1, (
        "expected total_weight to be a single element tensor, "
        f"got: {total_weight.shape} ( {total_weight.numel()}, elements)"
    )

    return _nll_loss_backward(
        grad_output, self, target, weight, reduction, ignore_index, total_weight
    )


@register_decomposition(aten.binary_cross_entropy)
@out_wrapper()
@pw_cast_for_opmath
def binary_cross_entropy(
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: int = Reduction.MEAN.value,
) -> Tensor:
    # We cannot currently model this without introducing data-dependent control flow
    # TORCH_CHECK(
    #     (input_val >= 0) && (input_val <= 1),
    #     "all elements of input should be between 0 and 1"
    # )
    loss = (target - 1) * torch.maximum(
        torch.log1p(-self), self.new_full((), -100)
    ) - target * torch.maximum(torch.log(self), self.new_full((), -100))
    if weight is not None:
        loss = loss * weight
    return apply_loss_reduction(loss, reduction)


@register_decomposition(aten.binary_cross_entropy_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def binary_cross_entropy_backward(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: int = Reduction.MEAN.value,
) -> Tensor:
    EPSILON = 1e-12
    result = grad_output * (self - target) / torch.clamp(self * (1 - self), min=EPSILON)
    if weight is not None:
        result = result * weight
    if reduction == Reduction.MEAN.value:
        result = result / self.numel()
    return result


@register_decomposition(aten.soft_margin_loss)
@out_wrapper()
@pw_cast_for_opmath
def soft_margin_loss(
    input: Tensor,
    target: Tensor,
    reduction: int = Reduction.MEAN.value,
) -> Tensor:
    loss = torch.log1p(torch.exp(-input * target))
    return apply_loss_reduction(loss, reduction)


@register_decomposition(aten.soft_margin_loss_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def soft_margin_loss_backward(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    reduction: int = Reduction.MEAN.value,
) -> Tensor:
    grad_input = target * grad_output * (torch.sigmoid(target * self) - 1)
    if reduction == Reduction.MEAN.value:
        grad_input = grad_input / self.numel()
    return grad_input


@register_decomposition(aten.dist)
@out_wrapper()
def dist(input: Tensor, other: Tensor, p: float = 2):
    return aten.norm(input - other, p=p)


@register_decomposition(aten._euclidean_dist)
@out_wrapper()
def _euclidean_dist(x1: Tensor, x2: Tensor) -> Tensor:
    x1_norm = x1.pow(2).sum(-1, True)
    x1_pad = torch.ones_like(x1_norm, memory_format=torch.contiguous_format)
    x2_norm = x2.pow(2).sum(-1, True)
    x2_pad = torch.ones_like(x2_norm, memory_format=torch.contiguous_format)
    x1_ = torch.cat([x1.mul(-2), x1_norm, x1_pad], -1)
    x2_ = torch.cat([x2, x2_pad, x2_norm], -1)
    result = x1_.matmul(x2_.mT)
    return result.clamp_min(0).sqrt()


@register_decomposition(aten.slice_backward)
@out_wrapper()
def slice_backward(
    grad_output: Tensor,
    input_sizes: list[int],
    dim: int,
    start: int,
    end: int,
    step: int,
):
    grad_input = grad_output.new_zeros(input_sizes)
    return torch.slice_scatter(grad_input, grad_output, dim, start, end, step)


@register_decomposition(aten.slice.Tensor)
def slice_forward(
    # Tensor(a) self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1
    self: Tensor,
    dim: int = 0,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
):
    from torch.fx.experimental.symbolic_shapes import statically_known_true

    ndim = self.dim()
    if ndim == 0:
        raise RuntimeError("slice() cannot be applied to a 0-dim tensor.")
    dim = utils.canonicalize_dim(self.dim(), dim)
    sizes = list(self.size())
    strides = list(self.stride())

    if step <= 0:
        raise RuntimeError("slice step must be positive")

    start_val = start if start is not None else 0
    end_val = end if end is not None else sys.maxsize  # 2^63 - 1

    if start_val < 0:
        start_val += sizes[dim]

    if end_val < 0:
        end_val += sizes[dim]

    if start_val < 0:
        start_val = 0
    elif start_val > sizes[dim]:
        start_val = sizes[dim]

    if statically_known_true(end_val == sys.maxsize):
        end_val = sizes[dim]
    elif end_val < start_val:
        end_val = start_val
    elif end_val > sizes[dim]:
        end_val = sizes[dim]

    storage_offset = self.storage_offset() + start_val * strides[dim]
    len = end_val - start_val
    sizes[dim] = (len + step - 1) // step
    strides[dim] *= step

    if self.is_quantized:
        raise NotImplementedError(
            "Slice decomposition for quantized tensors aren't implemented"
        )
    else:
        return self.as_strided(sizes, strides, storage_offset)


def _normalize_start_end(
    x: Tensor, dim: int, start: Optional[int], end: Optional[int]
) -> tuple[int, int]:
    """
    Normalize start and end such that both are in the range
    [0, x.get_size()[dim]] and start <= end.
    """
    dim_size = x.shape[dim]

    def clamp_wrap(val, lower, upper, default) -> int:
        if val is None:
            return default
        if val < 0:
            val = val + dim_size
        return min(max(val, lower), upper)

    start = clamp_wrap(start, 0, dim_size, 0)
    end = clamp_wrap(end, start, dim_size, dim_size)
    return start, end


# This is not in torch._refs because aten.index used by
# aten._unsafe_masked_index does not have a decomposition.
@register_decomposition(aten.slice_scatter)
@out_wrapper()
def slice_scatter(
    input: Tensor,
    src: Tensor,
    dim: int = 0,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
):
    dim = utils.canonicalize_dim(input.ndim, dim)
    dim_size = input.shape[dim]
    start, end = _normalize_start_end(input, dim, start, end)

    src_size = list(input.shape)
    src_size[dim] = (end - start + (step - 1)) // step
    src = src.expand(src_size)

    if start == 0 and end == dim_size and step == 1:
        return src.clone()

    indices: list[Optional[Tensor]] = [None] * input.dim()
    idx = torch.arange(dim_size, device=input.device)
    indices[dim] = (idx - start) // step

    mask = torch.ones(dim_size, device=input.device, dtype=torch.bool)
    if start != 0:
        mask = torch.logical_and(mask, idx >= start)

    if end != dim_size:
        mask = torch.logical_and(mask, idx < end)

    if step != 1:
        mask = torch.logical_and(mask, (idx - start) % step == 0)

    mask_shape = [1] * input.dim()
    mask_shape[dim] = -1
    mask = mask.view(mask_shape)
    return aten.where(mask, aten._unsafe_masked_index(src, mask, indices, 0), input)


@register_decomposition(aten.select_backward)
@out_wrapper()
def select_backward(grad_output: Tensor, input_sizes: list[int], dim: int, index: int):
    grad_input = grad_output.new_zeros(input_sizes)
    return torch.select_scatter(grad_input, grad_output, dim, index)


@register_decomposition(aten.diagonal_backward)
@out_wrapper()
def diagonal_backward(
    grad_output: Tensor, input_sizes: list[int], offset: int, dim1: int, dim2: int
):
    grad_input = grad_output.new_zeros(input_sizes)
    return torch.diagonal_scatter(grad_input, grad_output, offset, dim1, dim2)


def _cast_grad_to_input_dtype(
    grad_output: Tensor, grad_input: Tensor, input_dtype: torch.dtype
):
    if grad_output.dtype != input_dtype:
        grad_input = grad_input.to(input_dtype)
    return grad_input


@register_decomposition(aten._softmax_backward_data)
@out_wrapper("grad_input")
@compute_only_pw_cast_for_opmath
def _softmax_backward_data(
    grad_output: Tensor, output: Tensor, dim: int, input_dtype: torch.dtype
):
    new_grad_output = grad_output * output
    grad_input = new_grad_output - output * torch.sum(
        new_grad_output, dim=dim, keepdim=True
    )

    # CPU kernel doesn't respect input_dtype, but following check doesn't work for meta tensor
    # if grad_output.device == torch.device("cpu"):
    #     return grad_input.contiguous()

    return _cast_grad_to_input_dtype(grad_output, grad_input, input_dtype).contiguous()


@register_decomposition(aten._log_softmax_backward_data)
@out_wrapper()
@compute_only_pw_cast_for_opmath
def _log_softmax_backward_data(
    grad_output: Tensor, output: Tensor, dim: int, input_dtype: torch.dtype
):
    grad_input = grad_output - torch.exp(output) * torch.sum(
        grad_output, dim=dim, keepdim=True
    )
    return _cast_grad_to_input_dtype(grad_output, grad_input, input_dtype)


def _im2col_col2im_indices_along_dim(
    input_d, kernel_d, dilation_d, padding_d, stride_d, device
):
    """Utility function to implement im2col and col2im"""
    blocks_d = input_d + padding_d * 2 - dilation_d * (kernel_d - 1)

    arange_kw = partial(torch.arange, dtype=torch.int64, device=device)

    # Stride kernel over input and find starting indices along dim d
    blocks_d_indices = arange_kw(0, blocks_d, stride_d).unsqueeze(0)

    # Apply dilation on kernel and find its indices along dim d
    kernel_grid = arange_kw(0, kernel_d * dilation_d, dilation_d).unsqueeze(-1)

    # Broadcast and add kernel starting positions (indices) with
    # kernel_grid along dim d, to get block indices along dim d
    return blocks_d_indices + kernel_grid


@register_decomposition(aten.im2col)
@out_wrapper()
def im2col(
    input: Tensor,
    kernel_size: list[int],
    dilation: list[int],
    padding: list[int],
    stride: list[int],
) -> Tensor:
    torch._check(len(kernel_size) == 2, lambda: "im2col(): only 2D kernel supported")
    torch._check(len(dilation) == 2, lambda: "im2col(): only 2D dilation supported")
    torch._check(len(padding) == 2, lambda: "im2col(): only 2D padding supported")
    torch._check(len(stride) == 2, lambda: "im2col(): only 2D stride supported")

    def check_positive(param, param_name, strict=True):
        cond = all(p > 0 for p in param) if strict else all(p >= 0 for p in param)
        torch._check(
            cond, lambda: f"{param_name} should be greater than zero, but got {param}"
        )

    check_positive(kernel_size, "kernel_size")
    check_positive(dilation, "dilation")
    check_positive(dilation, "padding", strict=False)
    check_positive(stride, "stride")

    shape = input.shape
    ndim = len(shape)
    torch._check(
        ndim in (3, 4) and all(d != 0 for d in shape[-3:]),
        lambda: "Expected 3D or 4D (batch mode) tensor for input with possible 0 batch size "
        f"and non-zero dimensions, but got: {tuple(shape)}",
    )
    output_size = tuple(
        1 + (out + 2 * pad - dil * (ker - 1) - 1) // st
        for out, pad, dil, ker, st in zip(
            shape[-2:], padding, dilation, kernel_size, stride
        )
    )
    torch._check(
        all(c > 0 for c in output_size),
        lambda: f"Given an input with spatial size {tuple(shape[-2:])}, "
        f"kernel_size={kernel_size}, dilation={dilation}, "
        f"padding={padding}, stride={stride}, "
        "the calculated shape of the array of sliding blocks "
        f"is {output_size}, but its components must be at least one.",
    )
    batched_input = ndim == 4
    if not batched_input:
        input = input.unsqueeze(0)

    batch_dim, channel_dim, input_h, input_w = input.shape

    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    kernel_h, kernel_w = kernel_size

    blocks_row_indices = _im2col_col2im_indices_along_dim(
        input_h, kernel_h, dilation_h, padding_h, stride_h, input.device
    )
    blocks_col_indices = _im2col_col2im_indices_along_dim(
        input_w, kernel_w, dilation_w, padding_w, stride_w, input.device
    )

    # Note that F.pad takes (padding_left, padding_right, padding_top, padding_bottom)
    # ugh
    padded_input = F.pad(input, (padding_w, padding_w, padding_h, padding_h))

    blocks_row_indices = blocks_row_indices.unsqueeze(-1).unsqueeze(-1)
    output = padded_input[:, :, blocks_row_indices, blocks_col_indices]
    output = output.permute(0, 1, 2, 4, 3, 5)
    num_blocks_row = blocks_row_indices.size(1)
    num_blocks_col = blocks_col_indices.size(1)
    output = output.reshape(
        batch_dim, channel_dim * kernel_h * kernel_w, num_blocks_row * num_blocks_col
    )

    if not batched_input:
        output = output.squeeze(0)
    return output


@register_decomposition(aten.col2im)
@out_wrapper()
@pw_cast_for_opmath
def col2im(
    input: Tensor,
    output_size: list[int],
    kernel_size: list[int],
    dilation: list[int],
    padding: list[int],
    stride: list[int],
) -> Tensor:
    torch._check(len(output_size) == 2, lambda: "only 2D output_size supported")
    torch._check(len(kernel_size) == 2, lambda: "only 2D kernel supported")
    torch._check(len(dilation) == 2, lambda: "only 2D dilation supported")
    torch._check(len(padding) == 2, lambda: "only 2D padding supported")
    torch._check(len(stride) == 2, lambda: "only 2D stride supported")

    def check_positive(param, param_name, strict=True):
        cond = all(p > 0 for p in param) if strict else all(p >= 0 for p in param)
        torch._check(
            cond, lambda: f"{param_name} should be greater than zero, but got {param}"
        )

    check_positive(kernel_size, "kernel_size")
    check_positive(dilation, "dilation")
    check_positive(padding, "padding", strict=False)
    check_positive(stride, "stride")
    check_positive(output_size, "output_size")

    shape = input.shape
    ndim = len(shape)
    torch._check(
        ndim in (2, 3) and all(d != 0 for d in shape[-2:]),
        lambda: "Expected 2D or 3D (batch mode) tensor for input with possible 0 batch size "
        f"and non-zero dimensions, but got: {tuple(shape)}",
    )
    prod_kernel_size = kernel_size[0] * kernel_size[1]
    torch._check(
        shape[-2] % prod_kernel_size == 0,
        lambda: "Expected size of input's first non-batch dimension to be divisible by the "
        f"product of kernel_size, but got input.shape[-2] = {shape[-2]} and "
        f"kernel_size={kernel_size}",
    )
    col = [
        1 + (out + 2 * pad - dil * (ker - 1) - 1) // st
        for out, pad, dil, ker, st in zip(
            output_size, padding, dilation, kernel_size, stride
        )
    ]
    L = col[0] * col[1]
    torch._check(
        shape[-1] == L,
        lambda: f"Given output_size={output_size}, kernel_size={kernel_size}, "
        f"dilation={dilation}, padding={padding}, stride={stride}, "
        f"expected input.size(-1) to be {L} but got {shape[-1]}.",
    )
    torch._check(
        L > 0,
        lambda: f"Given output_size={output_size}, kernel_size={kernel_size}, "
        f"dilation={dilation}, padding={padding}, stride={stride}, "
        f"expected input.size(-1) to be {L} but got {shape[-1]}.",
    )
    batched_input = ndim == 3
    if not batched_input:
        input = input.unsqueeze(0)

    shape = input.shape

    out_h, out_w = output_size
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    kernel_h, kernel_w = kernel_size

    # col2im is defined as the backwards of im2col, so we differentiate its decomposition by hand
    input = input.reshape([shape[0], shape[1] // prod_kernel_size] + kernel_size + col)
    input = input.permute(0, 1, 2, 4, 3, 5)

    indices_row = _im2col_col2im_indices_along_dim(
        out_h, kernel_h, dilation_h, padding_h, stride_h, input.device
    )
    indices_row = _unsqueeze_to_dim(indices_row, 4)
    indices_col = _im2col_col2im_indices_along_dim(
        out_w, kernel_w, dilation_w, padding_w, stride_w, input.device
    )

    output_padded_size = [o + 2 * p for o, p in zip(output_size, padding)]
    output = input.new_zeros(
        [shape[0], shape[1] // prod(kernel_size)] + output_padded_size
    )
    idx = (None, None, indices_row, indices_col)
    output = aten._unsafe_index_put(output, idx, input, accumulate=True)
    output = F.pad(output, (-padding_w, -padding_w, -padding_h, -padding_h))

    if not batched_input:
        output = output.squeeze(0)
    return output


@register_decomposition(aten.native_dropout_backward)
@out_wrapper()
def native_dropout_backward(grad_output: Tensor, mask: Tensor, scale: float):
    # According to the CUDA kernel implementation we should have this test;
    # but it seems to fail tests!
    # torch._check(mask.dtype == torch.bool, lambda: f"Mask should be Bool Scalar Type {mask.dtype}")

    # Mimicking CUDA kernel's behavior for output stride: output follow input's memory format
    # This different from TensorIterator's behavior
    r = (grad_output * (mask.type_as(grad_output) * scale)).clone(
        memory_format=utils.suggest_memory_format(grad_output)
    )
    return r


@register_decomposition(aten.unfold_backward)
@out_wrapper()
def unfold_backward(
    grad: Tensor, input_size: list[int], dimension: int, size: int, step: int
) -> Tensor:
    if len(input_size) == 0:
        return torch.squeeze_copy(grad, 0)
    dim = utils.canonicalize_dim(len(input_size), dimension)
    idx = torch.arange(input_size[dim], device=grad.device, dtype=torch.int32)
    idx = idx.unfold(0, size, step).flatten()
    grad = grad.movedim(-1, dim + 1).flatten(dim, dim + 1)
    # nb. At the moment this generates two kernels in triton
    # It could potentially be fused into one call to scatter_reduce,
    # in the case step <= size provided scatter_reduce generates 1 kernel
    grad_input = grad.new_zeros(input_size)
    index = (None,) * dim + (idx,)
    return aten._unsafe_index_put(grad_input, index, grad, accumulate=True).contiguous()


@register_decomposition(aten.logit_backward.default)
@pw_cast_for_opmath
def logit_backward(
    grad_output: Tensor, self: Tensor, eps: Optional[float] = None
) -> Tensor:
    if eps is not None:
        lo = eps
        hi = 1.0 - lo
        return torch.where(
            torch.logical_and(self >= lo, self <= hi),
            grad_output / (self * (1.0 - self)),
            0.0,
        )
    else:
        return torch.where(
            torch.logical_and(self >= 0.0, self <= 1.0),
            grad_output / (self * (1.0 - self)),
            self.new_full((), float("nan")),
        )


@register_decomposition(aten.dropout)
@aten.dropout.default.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.dropout.default.py_impl(DispatchKey.Autograd)
def dropout(input: Tensor, p: float, train: Optional[bool]):
    if train and p != 0:
        return aten.native_dropout(input, p, train)[0]
    else:
        return input.clone()


@register_decomposition(aten.native_dropout)
@out_wrapper("out0", "out1")
def native_dropout(input: Tensor, p: float, train: Optional[bool]):
    if train and p != 0:
        if p == 1:
            return (torch.zeros_like(input), torch.zeros_like(input, dtype=torch.bool))
        if not input.dtype.is_floating_point:
            raise RuntimeError(
                "result type Float can't be cast to the desired output type Long"
            )
        bool_mask = torch.rand_like(input) > p
        res = bool_mask * input * float(1.0 / (1.0 - p))
        return (res, bool_mask)
    else:
        return (input, torch.ones_like(input, dtype=torch.bool))


@register_decomposition(aten._softmax)
@out_wrapper()
def _softmax(x: Tensor, dim: int, half_to_float: bool):
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    # eager softmax returns a contiguous tensor. Ensure that decomp also returns
    # a contiguous tensor.
    x = x.contiguous()
    if half_to_float:
        assert x.dtype == torch.half
    computation_dtype, result_dtype = utils.elementwise_dtypes(
        x, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    x = x.to(computation_dtype)
    if guard_or_false(x.numel() == 0):
        unnormalized = torch.exp(x)
    else:
        x_max = torch.amax(x, dim, keepdim=True)
        unnormalized = torch.exp(x - x_max)
    result = unnormalized / torch.sum(unnormalized, dim, keepdim=True)
    if not half_to_float:
        result = result.to(result_dtype)
    return result


@register_decomposition(aten._log_softmax)
@out_wrapper(exact_dtype=True)
def _log_softmax(x: Tensor, dim: int, half_to_float: bool):
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    # eager log_softmax returns a contiguous tensor. Ensure that decomp also
    # returns a contiguous tensor.
    x = x.contiguous()
    if half_to_float:
        assert x.dtype == torch.half
    computation_dtype, result_dtype = utils.elementwise_dtypes(
        x, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    x = x.to(computation_dtype)
    if guard_or_false(x.numel() == 0):
        shifted = x
    else:
        x_max = torch.amax(x, dim, keepdim=True)
        shifted = x - x_max
    shifted_logsumexp = torch.log(torch.sum(torch.exp(shifted), dim, keepdim=True))
    result = shifted - shifted_logsumexp
    if not half_to_float:
        result = result.to(result_dtype)
    return result


@register_decomposition(aten.embedding)
@out_wrapper()
def embedding(
    weight: Tensor,
    indices: Tensor,
    padding_idx: int = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tensor:
    assert weight.dim() == 2, "'weight' must be 2-D"
    # Nb. scale_grad_by_freq is not used in the forward
    if indices.ndim <= 1:
        # We need this one as weight[indices] calls item() in these cases
        out = weight.index_select(0, indices)
        if indices.ndim == 0:
            out = out.squeeze(0)
        return out
    else:
        return weight[indices]


@register_decomposition(aten.embedding_dense_backward)
@out_wrapper()
def embedding_dense_backward(
    grad_output: Tensor,
    indices: Tensor,
    num_weights: int,
    padding_idx: int,
    scale_grad_by_freq: bool,
):
    computation_dtype, result_dtype = utils.elementwise_dtypes(
        grad_output, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    grad_output = grad_output.to(computation_dtype)
    indices = _maybe_convert_to_dtype(indices, torch.long)  # type: ignore[assignment]
    if scale_grad_by_freq:
        counts = indices.new_zeros((num_weights,))
        ones = torch.ones_like(indices)
        counts = aten._unsafe_index_put(counts, [indices], ones, accumulate=True)
        grad_weights_scale = counts[indices]
        grad_output = grad_output / grad_weights_scale.unsqueeze(-1)

    mask = _unsqueeze_to_dim(indices == padding_idx, grad_output.ndim)
    grad = grad_output.masked_fill(mask, 0)
    grad_weight = grad_output.new_zeros(
        (num_weights,) + grad_output.shape[indices.ndim :]
    )
    return aten._unsafe_index_put(grad_weight, [indices], grad, accumulate=True).to(
        result_dtype
    )


def prod(x: list[int]):
    r = 1
    for i in x:
        r *= i
    return r


def _pad_chunk(
    tensors: list[Tensor],
    dim: int,
    num_chunks: int,
) -> list[Tensor]:
    padded_tensors = []
    for tensor in tensors:
        tensor_size = tensor.size()
        pad_along_dim = (tensor_size[dim] + num_chunks - 1) // num_chunks * num_chunks
        if pad_along_dim != tensor_size[dim]:
            # Use aten.constant_pad_nd instead of copy_ for functionalization
            pad = [0] * 2 * (tensor.ndim - dim - 1) + [
                0,
                pad_along_dim - tensor_size[dim],
            ]
            tensor = aten.constant_pad_nd(tensor, pad, 0)
        view_size = tensor_size[:dim] + torch.Size([num_chunks, -1])
        padded_tensors.append(tensor.reshape(view_size))
    return padded_tensors


def have_same_ndims(tensors: list[Tensor]):
    ndim = tensors[0].ndim
    for tensor in tensors:
        if tensor.ndim != ndim:
            return False
    return True


def leading_dimension_matches(tensors: list[Tensor], dim: int):
    leading_dim_sizes = tensors[0].size()[:dim]
    for tensor in tensors:
        torch._check(
            tensor.size()[:dim] == leading_dim_sizes,
            lambda: "_chunk_cat expects same sizes of 0,...,dim-1 dimensions for all tensors",
        )


def _preprocess_chunk_cat_inputs(
    tensors: list[Tensor],
    dim: int,
    num_chunks: int,
):
    torch._check(num_chunks >= 1, lambda: "_chunk_cat expects positive num_chunks")
    torch._check(
        len(tensors) > 0, lambda: "_chunk_cat expects a non-empty input tensor list"
    )
    expected_dtype = tensors[0].dtype
    expected_device = tensors[0].device
    for tensor in tensors:
        torch._check(tensor.numel() > 0, lambda: "_chunk_cat expects non-empty tensor")
        torch._check(
            tensor.dtype == expected_dtype,
            lambda: "_chunk_cat expects all input tensors with the same dtype",
        )
        torch._check(
            tensor.device == expected_device,
            lambda: "_chunk_cat expects all inputs tensors on the same device",
        )
    if have_same_ndims(tensors):
        dim = utils.canonicalize_dim(tensors[0].dim(), dim)
    else:
        torch._check(
            dim >= 0,
            lambda: "_chunk_cat expects non-negative dim when input tensors have different ndims",
        )
        for tensor in tensors:
            torch._check(
                dim < tensor.ndim,
                lambda: "_chunk_cat expects dim < ndim for all input tensors",
            )
    leading_dimension_matches(tensors, dim)
    return dim


@register_decomposition([aten._chunk_cat.default, aten._chunk_cat.out])
def _chunk_cat(
    tensors: list[Tensor],
    dim: int,
    num_chunks: int,
    out: Optional[Tensor] = None,
) -> Tensor:
    dim = _preprocess_chunk_cat_inputs(tensors, dim, num_chunks)
    padded_tensors = _pad_chunk(tensors, dim, num_chunks)
    if out is None:
        return torch.cat(padded_tensors, dim + 1)
    else:
        torch.cat(padded_tensors, dim + 1, out=out)
        return out


# out_wrapper currently does not allow optional outputs
@register_decomposition(
    [aten.split_with_sizes_copy.default, aten.split_with_sizes_copy.out]
)
def split_with_sizes_copy(
    self: Tensor,
    split_sizes: list[int],
    dim: int = 0,
    out: Optional[list[Tensor]] = None,
) -> Optional[list[Tensor]]:
    splits = aten.split_with_sizes(self, split_sizes, dim=dim)
    if out is None:
        return [s.clone(memory_format=torch.contiguous_format) for s in splits]
    else:
        for output, split in zip(out, splits):
            _maybe_resize_out(output, split.shape)
            _safe_copy_out(copy_from=split, copy_to=output, exact_dtype=True)
        return None


@register_decomposition(aten.unsafe_split.Tensor)
def unsafe_split(input: Tensor, split_size: int, dim: int = 0) -> tuple[Tensor, ...]:
    return aten.split.Tensor(input, split_size, dim)


@register_decomposition(aten.unsafe_split_with_sizes.default)
def unsafe_split_with_sizes(
    input: Tensor, split_sizes: list[int], dim: int = 0
) -> tuple[Tensor, ...]:
    return aten.split_with_sizes.default(input, split_sizes, dim)


@register_decomposition(aten.split.Tensor)
def split(self: Tensor, split_size: int, dim: int = 0) -> tuple[Tensor, ...]:
    input_sizes = self.shape
    dim_size = input_sizes[dim]
    if split_size == 0:
        assert dim_size == 0
        return (self.detach(),)
    chunks = (dim_size + split_size - 1) // split_size

    # Avoid importing sympy at a module level
    from torch.fx.experimental.symbolic_shapes import guard_int

    chunks = guard_int(chunks)
    split_sizes = [split_size for i in range(chunks)]
    split_sizes[-1] = split_size - (split_size * chunks - dim_size)
    return torch.split(self, split_sizes, dim)


@aten.tensor_split.tensor_indices_or_sections.py_impl(
    DispatchKey.CompositeImplicitAutograd
)
def tensor_split_tensor_indices_or_sections_py_impl(
    self: Tensor,
    tensor_indices_or_sections: Tensor,
    dim: int = 0,
) -> tuple[Tensor, ...]:
    assert tensor_indices_or_sections.device.type == "cpu"
    assert tensor_indices_or_sections.dtype == torch.int64
    split_dim = tensor_indices_or_sections.dim()
    torch._check(
        split_dim == 1 or split_dim == 0,
        lambda: "tensor_split expected tensor_indices_or_sections to be a zero-dimensional "
        f"or one-dimensional tensor, but got a tensor with {split_dim} dims",
    )
    if split_dim == 0:
        sections = tensor_indices_or_sections.item()
        assert isinstance(sections, IntLike)
        return self.tensor_split(sections, dim)
    else:
        ctx = nullcontext
        if (fake_mode := torch._guards.detect_fake_mode()) and (
            shape_env := fake_mode.shape_env
        ):
            ctx = shape_env.ignore_fresh_unbacked_symbols  # type: ignore[assignment]
        # In fake tensor prop, we end up calling slice() with these unbacked indices.
        # Because slice has flexible semantics, the unbacked handling generates new output sizes
        # for each slice, effectively clobbering over these index symbols.
        # To avoid PendingUnbackedSymbolNotFound errors, we tell the compiler it's fine to not bind these.
        with ctx():
            indices = [i.item() for i in tensor_indices_or_sections]
        # WARNING: Tempted to torch._check(x>0) on the indices here?  You
        # can't: tensor_split works with negative values in indices:
        #
        # >>> torch.tensor_split(torch.randn(10), torch.tensor([-5, 5]))
        # (tensor([ 0.3540,  2.1074, -0.8507,  1.1639,  0.3055]), tensor([]),
        # tensor([-0.4285,  1.0692, -0.1776,  0.9362,  1.6143]))
        #
        # Sorry, I don't make the rules.  Explicitly do the item call in user
        # code if you KNOW that they are non-negative.
        return self.tensor_split(indices, dim)


# TODO: this doesn't appear to have enough precision in bfloat16
@register_decomposition(aten.addmm)
@out_wrapper(exact_dtype=True)
@pw_cast_for_opmath
def addmm(self: Tensor, mat1: Tensor, mat2: Tensor, beta: int = 1, alpha: int = 1):
    if not self.is_floating_point() and not self.is_complex():
        beta = int(beta)
        alpha = int(alpha)
    out = alpha * torch.mm(mat1, mat2)
    if beta == 0:
        return out

    # The output of aten.addmm is contiguous, we need to match this behavior in the decomposition.
    # The original implementation 'beta * self + out' would return a strided tensor if `self` is strided.
    # We thus use `out`, the output of torch.mm, which is always contiguous, as the first argument for addition.
    # This is relying on TensorIterator's behavior that it takes higher precedence on the stride of first input.
    # Alternative, we can write `(beta * self + out).contiguous()`, but it introduces another copy in some cases.
    # This implementation is not ideal, and we should revisit this when we have a better solution.
    return out + beta * self


@register_decomposition(aten._addmm_activation)
@out_wrapper()
@pw_cast_for_opmath
def _addmm_activation(
    self: Tensor,
    mat1: Tensor,
    mat2: Tensor,
    beta: int = 1,
    alpha: int = 1,
    use_gelu: bool = False,
)
```



## High-Level Overview


This Python file contains 1 class(es) and 251 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Reduction`

**Functions defined**: `type_casts`, `inner`, `increase_prec`, `decrease_prec`, `_unsqueeze_to_dim`, `tanh_backward`, `sigmoid_backward`, `softplus_backward`, `elu_backward`, `fill_scalar`, `fill_tensor`, `hardsigmoid`, `hardsigmoid_backward`, `hardtanh_backward`, `hardswish`, `hardswish_backward`, `threshold_backward`, `leaky_relu_backward`, `gelu_backward`, `mish_backward`

**Key imports**: functools, itertools, numbers, operator, sys, Callable, Iterable, nullcontext, Enum, partial, reduce, chain, product


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_decomp`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `itertools`
- `numbers`
- `operator`
- `sys`
- `collections.abc`: Callable, Iterable
- `contextlib`: nullcontext
- `enum`: Enum
- `typing`: Any, cast, Optional, Union
- `torch`
- `torch._meta_registrations`
- `torch._prims as prims`
- `torch._prims_common as utils`
- `torch.nn.functional as F`
- `torch._decomp`: register_decomposition
- `torch._higher_order_ops.out_dtype`: out_dtype
- `torch.utils`: _pytree as pytree
- `torch.utils._pytree`: tree_map
- `torch.fx.experimental.symbolic_shapes`: statically_known_true


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

Files in the same folder (`torch/_decomp`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`decompositions_for_jvp.py_docs.md`](./decompositions_for_jvp.py_docs.md)
- [`decompositions_for_rng.py_docs.md`](./decompositions_for_rng.py_docs.md)


## Cross-References

- **File Documentation**: `decompositions.py_docs.md`
- **Keyword Index**: `decompositions.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
