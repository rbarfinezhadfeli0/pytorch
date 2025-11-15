# Documentation: decompositions.py

## File Metadata
- **Path**: `torch/_decomp/decompositions.py`
- **Size**: 181990 bytes
- **Lines**: 5376
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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
):
    out = addmm(self, mat1, mat2, beta, alpha)
    if use_gelu:
        if self.is_cuda:
            return aten.gelu(out, approximate="tanh")
        else:
            return aten.gelu(out)
    return aten.relu(out)


@register_decomposition(aten.addmv)
@out_wrapper(exact_dtype=True)
@pw_cast_for_opmath
def addmv(self: Tensor, mat1: Tensor, vec: Tensor, beta: int = 1, alpha: int = 1):
    if not self.is_floating_point() and not self.is_complex():
        beta = int(beta)
        alpha = int(alpha)
    out = alpha * torch.mv(mat1, vec)
    if beta == 0:
        return out
    if out.numel() == 0:  # handle empty matrix
        return beta * self
    return out + beta * self


@register_decomposition(aten.native_group_norm_backward.default)
@pw_cast_for_opmath
def native_group_norm_backward(
    grad_output: Tensor,
    input: Tensor,
    mean: Tensor,
    rstd: Tensor,
    gamma: Optional[Tensor],
    N: int,
    C: int,
    HxW: int,
    group: int,
    output_mask: list[bool],
) -> tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    utils.check_same_device(
        grad_output, input, mean, rstd, allow_cpu_scalar_tensors=False
    )
    utils.check_same_shape(input, grad_output, allow_cpu_scalar_tensors=False)
    utils.check_same_shape(mean, rstd, allow_cpu_scalar_tensors=False)
    torch._check(
        input.numel() == N * C * HxW,
        lambda: f"Expect input to have {N * C * HxW} elements",
    )
    torch._check(
        mean.shape == (N, group),
        lambda: f"Expect mean to have shape ({N}, {group}, but got {mean.shape}",
    )
    torch._check(
        gamma is None or gamma.numel() == C,
        lambda: f"Expect gamma to have {C} elements but got {gamma.numel() if gamma is not None else -1}",
    )

    cpg = C // group
    torch._check(
        C == cpg * group,
        lambda: f"Expect number of channels {C} to be evenly-divisible by number of groups {group}",
    )

    # Compute Internal gradients
    ds = torch.mul(grad_output, input).view(N, C, HxW).sum(dim=[2])
    db = grad_output.view(N, C, HxW).sum(dim=[2])

    d_input: Optional[Tensor] = None
    d_gamma: Optional[Tensor] = None
    d_bias: Optional[Tensor] = None
    if output_mask[0]:
        s = 1.0 / (HxW * cpg)
        if gamma is not None:
            ds_val = torch.mul(ds, gamma.unsqueeze(0)).reshape(N, group, cpg).sum(2)
            db_val = torch.mul(db, gamma.unsqueeze(0)).reshape(N, group, cpg).sum(2)
            c1 = torch.mul(
                rstd.unsqueeze(-1),
                gamma.reshape(1, group, cpg),
            )
        else:
            ds_val = ds.reshape(N, group, cpg).sum(2)
            db_val = db.reshape(N, group, cpg).sum(2)
            c1 = torch.mul(
                rstd.unsqueeze(-1),
                torch.ones((1, group, cpg), device=rstd.device),
            )
        c2 = (db_val * mean - ds_val) * rstd * rstd * rstd * s
        c3 = -c2 * mean - db_val * rstd * s

        c1 = c1.unsqueeze(-1)
        c2 = _unsqueeze_to_dim(c2, 4)
        c3 = _unsqueeze_to_dim(c3, 4)
        d_input = (
            torch.mul(grad_output.reshape(N, group, cpg, HxW), c1)
            + torch.mul(input.reshape(N, group, cpg, HxW), c2)
            + c3
        )
        d_input = d_input.reshape(input.shape).to(input.dtype)
    if output_mask[1]:
        d_gamma = (
            (
                (ds.view(N, group, cpg) - db.view(N, group, cpg) * mean.unsqueeze(-1))
                * rstd.unsqueeze(-1)
            )
            .sum(dim=[0])
            .reshape(C)
        )
    if output_mask[2]:
        d_bias = db.sum(dim=[0])

    return (d_input, d_gamma, d_bias)


# out_wrapper currently does not allow optional outputs
@register_decomposition(aten.native_group_norm_backward.out)
def native_group_norm_backward_out(
    grad_output: Tensor,
    input: Tensor,
    mean: Tensor,
    rstd: Tensor,
    gamma: Optional[Tensor],
    N: int,
    C: int,
    HxW: int,
    group: int,
    output_mask: list[bool],
    *,
    out0: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
) -> tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    result = native_group_norm_backward(
        grad_output, input, mean, rstd, gamma, N, C, HxW, group, output_mask
    )
    grad_input = (out0, out1, out2)
    for i, r in enumerate(result):
        if r is not None:
            _maybe_resize_out(grad_input[i], r.shape)
            _safe_copy_out(copy_from=r, copy_to=grad_input[i], exact_dtype=True)

    return grad_input


def _maybe_cast(x: Optional[Tensor], dtype) -> Optional[Tensor]:
    if x is not None:
        return x.to(dtype)
    return x


# TODO: Take a closer look at the type promotion semantics
@register_decomposition(aten.native_layer_norm_backward.default)
def native_layer_norm_backward(
    grad_out: Tensor,
    input: Tensor,
    normalized_shape: list[int],
    mean: Tensor,
    rstd: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    output_mask: list[bool],
) -> tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    input_shape = input.shape
    input_ndim = input.dim()
    computation_dtype = utils.get_computation_dtype(input.dtype)
    grad_out_cast, input_cast, weight_cast, bias_cast = (
        x.to(computation_dtype, memory_format=torch.contiguous_format)
        if x is not None
        else x
        for x in (grad_out, input, weight, bias)
    )
    assert grad_out_cast is not None

    axis = input_ndim - len(normalized_shape)
    inner_dims = input_shape[axis:]
    outer_dims = input_shape[:axis]
    inner_dim_indices: list[int] = []
    outer_dim_indices: list[int] = []
    for i in range(input_ndim):
        if i >= axis:
            inner_dim_indices.append(i)
        else:
            outer_dim_indices.append(i)

    N = prod(inner_dims)  # type: ignore[arg-type]
    M = prod(outer_dims)  # type: ignore[arg-type]
    from torch.fx.experimental.symbolic_shapes import statically_known_true

    if statically_known_true(M == 0) or statically_known_true(N == 0):
        return (
            input.new_zeros(input_shape) if output_mask[0] else None,
            input.new_zeros(input_shape[axis:]) if output_mask[1] else None,
            input.new_zeros(input_shape[axis:]) if output_mask[2] else None,
        )
    mean = _unsqueeze_to_dim(mean, input_cast.dim())  # type: ignore[union-attr]
    rstd = _unsqueeze_to_dim(rstd, input_cast.dim())  # type: ignore[union-attr]
    assert input_cast is not None
    x_hat = (input_cast - mean) * rstd
    if weight_cast is not None:
        grad_x_hat = grad_out_cast * weight_cast
    else:
        grad_x_hat = grad_out_cast
    a = grad_x_hat * N
    b = torch.sum(grad_x_hat, inner_dim_indices, True)
    c1 = torch.mul(grad_x_hat, x_hat)
    c2 = torch.sum(c1, inner_dim_indices, True)
    c3 = torch.mul(x_hat, c2)

    inner = a - b - c3
    d_input: Optional[Tensor] = None
    d_weight: Optional[Tensor] = None
    d_bias: Optional[Tensor] = None
    if output_mask[0]:
        d_input = (rstd / N) * inner

    if output_mask[1] and weight_cast is not None:
        if len(outer_dim_indices) > 0:
            d_weight = torch.sum(grad_out_cast * x_hat, outer_dim_indices, False)
        else:
            d_weight = grad_out_cast * x_hat

    if output_mask[2] and bias_cast is not None:
        if len(outer_dim_indices) > 0:
            d_bias = torch.sum(grad_out_cast, outer_dim_indices, False)
        else:
            d_bias = grad_out_cast.clone()

    return (
        _maybe_cast(d_input, input.dtype),
        _maybe_cast(d_weight, weight.dtype if weight is not None else None),
        _maybe_cast(d_bias, bias.dtype if bias is not None else None),
    )


# out_wrapper currently does not allow optional outputs
@register_decomposition(aten.native_layer_norm_backward.out)
def native_layer_norm_backward_out(
    grad_out: Tensor,
    input: Tensor,
    normalized_shape: list[int],
    mean: Tensor,
    rstd: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    output_mask: list[bool],
    *,
    out0: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
) -> tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    result = native_layer_norm_backward(
        grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask
    )
    grad_input = (out0, out1, out2)
    for i, r in enumerate(result):
        if r is not None:
            _maybe_resize_out(grad_input[i], r.shape)
            _safe_copy_out(copy_from=r, copy_to=grad_input[i], exact_dtype=True)

    return grad_input


@register_decomposition(aten._fused_rms_norm.default)
def _fused_rms_norm(
    input: Tensor,
    normalized_shape: list[int],
    weight: Optional[Tensor],
    eps: Optional[float],
) -> tuple[Tensor, Tensor]:
    dims_to_reduce: list[int] = []
    for i in range(len(normalized_shape)):
        dims_to_reduce.append(input.dim() - i - 1)

    # upcast is needed for fp16 and bf16
    computation_dtype = utils.get_computation_dtype(input.dtype)
    upcasted_input = input.to(computation_dtype)

    # computation_dtype would be one of [Double, Float, ComplexFloat, ComplexDouble]
    if eps is None:
        if computation_dtype in (torch.float32, torch.complex64):
            eps_val = torch.finfo(torch.float32).eps
        else:
            eps_val = torch.finfo(torch.float64).eps
    else:
        eps_val = eps

    rqrst_input = torch.rsqrt(
        # NB: don't inplace here, will violate functional IR invariant
        # NB: carefully use the Scalar overload of add to ensure compatibility with the C++ decomp
        torch.ops.aten.add.Scalar(
            torch.pow(upcasted_input, 2).mean(dim=dims_to_reduce, keepdim=True), eps_val
        )
    )

    upcasted_result = upcasted_input.mul(rqrst_input)

    if weight is not None:
        upcasted_result = upcasted_result.mul(weight)

    # NB: nested should be dead here, just here for fidelity
    is_nested = input.is_nested or (weight is not None and weight.is_nested)
    memory_format = utils.suggest_memory_format(input)
    is_channels_last = memory_format in (
        torch.channels_last,
        torch.channels_last_3d,
    )

    if not is_nested and not is_channels_last:
        upcasted_result = upcasted_result.contiguous()
        rqrst_input = rqrst_input.contiguous()

    # Cast normalized result back to original input type
    result = upcasted_result.type_as(input)

    return result, rqrst_input


@register_decomposition(aten._fused_rms_norm_backward.default)
def _fused_rms_norm_backward(
    grad_out: Tensor,
    input: Tensor,
    normalized_shape: list[int],
    rstd: Tensor,
    weight: Optional[Tensor],
    output_mask: list[bool],
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    input_shape = input.shape
    input_ndim = input.dim()
    computation_dtype = utils.get_computation_dtype(input.dtype)

    grad_out_cast = grad_out.to(
        computation_dtype, memory_format=torch.contiguous_format
    )
    input_cast = input.to(computation_dtype, memory_format=torch.contiguous_format)
    weight_cast = (
        weight.to(computation_dtype, memory_format=torch.contiguous_format)
        if weight is not None
        else None
    )
    assert grad_out_cast is not None

    axis = input_ndim - len(normalized_shape)
    inner_dims = input_shape[axis:]
    outer_dims = input_shape[:axis]
    inner_dim_indices: list[int] = []
    outer_dim_indices: list[int] = []
    for i in range(input_ndim):
        if i >= axis:
            inner_dim_indices.append(i)
        else:
            outer_dim_indices.append(i)

    N = prod(inner_dims)  # type: ignore[arg-type]
    M = prod(outer_dims)  # type: ignore[arg-type]
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    if guard_or_false(M == 0) or guard_or_false(N == 0):
        return (
            input.new_zeros(input_shape) if output_mask[0] else None,
            input.new_zeros(input_shape[axis:]) if output_mask[1] else None,
        )

    rstd = _unsqueeze_to_dim(rstd, input_cast.dim())  # type: ignore[union-attr]
    if weight_cast is not None:
        grad_x_hat = grad_out_cast * weight_cast
    else:
        grad_x_hat = grad_out_cast

    d_input: Optional[Tensor] = None
    d_weight: Optional[Tensor] = None

    x_hat = input_cast * rstd

    if output_mask[0]:
        sum_val = torch.sum(x_hat * grad_x_hat, dim=inner_dim_indices, keepdim=True)
        d_input = (grad_x_hat - (x_hat / N) * sum_val) * rstd

    if output_mask[1] and weight_cast is not None:
        d_weight_full_shape = grad_out_cast * x_hat
        if len(outer_dim_indices) > 0:
            d_weight = torch.sum(
                d_weight_full_shape, dim=outer_dim_indices, keepdim=False
            )
        else:
            d_weight = d_weight_full_shape

    return (
        _maybe_cast(d_input, input.dtype),
        _maybe_cast(d_weight, input.dtype),
    )


def native_batch_norm_helper(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    training: bool,
    momentum: float,
    eps: float,
    functional: bool,
) -> tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    reduction_dims = [0] + list(range(2, input.dim()))
    computation_dtype = utils.get_computation_dtype(input.dtype)
    new_running_mean = running_mean
    new_running_var = running_var
    if training:
        computation_dtype = utils.get_computation_dtype(input.dtype)
        input_acc = input.to(dtype=computation_dtype)
        biased_var, mean = torch.var_mean(
            input_acc, dim=reduction_dims, correction=0, keepdim=True
        )
        rstd = torch.rsqrt(biased_var + eps)

        output = (input - mean) * rstd

        save_mean = torch.squeeze(mean, reduction_dims)
        save_rstd = torch.squeeze(rstd, reduction_dims)
        if running_mean is not None:
            new_running_mean = momentum * save_mean + (1 - momentum) * running_mean
            if not functional:
                running_mean.copy_(new_running_mean)
        if running_var is not None:
            n = input.numel() / input.shape[1]
            # This doesn't strictly match eager's numerics, which accumulates var sum and then directly applies the correction
            # But... that would require re-implementing var here, for negligible numerics gain on a tensor whose
            # numerics probably don't matter.
            squeezed_var = torch.squeeze(biased_var, reduction_dims)
            unbiased_var = squeezed_var * (n / (n - 1))
            new_running_var = momentum * unbiased_var + (1 - momentum) * running_var
            if not functional:
                running_var.copy_(new_running_var)
    else:
        assert running_mean is not None and running_var is not None
        running_mean = running_mean.to(dtype=computation_dtype, copy=True)
        new_running_mean = running_mean
        running_var = running_var.to(dtype=computation_dtype, copy=True)
        new_running_var = running_var
        mean = running_mean
        invstd = 1 / (torch.sqrt(running_var + eps))
        # Very annoying inconsistency where CPU and CUDA give different shapes
        if input.device.type != "cpu":
            save_mean = running_mean
            save_rstd = invstd
        else:
            save_mean = input.new_zeros((0,))
            save_rstd = input.new_zeros((0,))
        mean = _unsqueeze_to_dim(mean, input.dim() - 1)
        invstd = _unsqueeze_to_dim(invstd, input.dim() - 1)
        output = (input - mean) * invstd

    if weight is not None:
        weight = weight.flatten()
        weight = _unsqueeze_to_dim(weight, input.dim() - 1)
        output = output * weight

    if bias is not None:
        bias = bias.flatten()
        bias = _unsqueeze_to_dim(bias, input.dim() - 1)
        output = output + bias

    if input.device.type == "cpu":
        save_mean = save_mean.to(dtype=input.dtype)
        save_rstd = save_rstd.to(dtype=input.dtype)
    return (
        output.to(dtype=input.dtype),
        save_mean,
        save_rstd,
        new_running_mean,
        new_running_var,
    )


@register_decomposition(aten.native_batch_norm)
@out_wrapper("out", "save_mean", "save_invstd")
def native_batch_norm(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    training: bool,
    momentum: float,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    output, save_mean, save_rstd, _, _ = native_batch_norm_helper(
        input, weight, bias, running_mean, running_var, training, momentum, eps, False
    )
    return output, save_mean, save_rstd


# TODO: this decomposition is NOT here to stay. We would much prefer replacing native_batch_norm
# with our new correctly schema'd _native_batch_norm_legit and its variants, but
# we cannot do that immediately in the C++ because it would be forwards incompatible
# with some mobile use cases.
#
# Since this change is most impactful for aot autograd/functionalization, we simply
# register this decomposition on the Autograd key for the python dispatcher (which is
# currently only used by aot autograd/functionalization and no one else, really).
# In two weeks or so, we should remove this decomposition and phase out the current native_batch_norm
# to be _native_batch_norm_legit and have the right schema (stating that there are input mutations).
@aten.native_batch_norm.default.py_impl(DispatchKey.Autograd)
@aten.native_batch_norm.default.py_impl(DispatchKey.CompositeImplicitAutograd)
def native_batch_norm_decomposition(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    training: bool,
    momentum: float,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    if running_mean is None and running_var is None:
        return aten._native_batch_norm_legit(
            input, weight, bias, training, momentum, eps
        )
    if running_mean is None:
        raise RuntimeError(
            "running_mean is None, but running_var is provided. "
            "They should both be None or both be provided."
        )
    if running_var is None:
        raise RuntimeError(
            "running_var is None, but running_mean is provided. "
            "They should both be None or both be provided."
        )
    if training:
        # HACK: batch norm consolidation should clean this up so this op doesn't take in a training arg.
        return aten._native_batch_norm_legit(
            input, weight, bias, running_mean, running_var, training, momentum, eps
        )
    else:
        return aten._native_batch_norm_legit_no_training(
            input, weight, bias, running_mean, running_var, momentum, eps
        )


@aten.unsafe_chunk.default.py_impl(DispatchKey.CompositeImplicitAutograd)
def unsafe_chunk_py_impl(tensor, chunks, dim=0) -> list[Tensor]:
    dim_size = tensor.size(dim)
    split_size = (dim_size + chunks - 1) // chunks

    if split_size == 0 and dim_size == 0:
        split_sizes = [split_size for _ in chunks]
        split_sizes[chunks - 1] = split_size - (split_size * chunks - dim_size)
        return torch.ops.aten.unsafe_split_with_sizes.default(tensor, split_sizes, dim)
    return torch.ops.aten.unsafe_split.Tensor(tensor, split_size, dim)


@register_decomposition(aten._native_batch_norm_legit_no_training.default)
def _native_batch_norm_legit_no_training(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    momentum: float,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    return aten._native_batch_norm_legit.default(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        False,  # training
        momentum,
        eps,
    )


@register_decomposition(aten._native_batch_norm_legit.default)
def _native_batch_norm_legit(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    training: bool,
    momentum: float,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    output, save_mean, save_rstd, _, _ = native_batch_norm_helper(
        input, weight, bias, running_mean, running_var, training, momentum, eps, False
    )
    return output, save_mean, save_rstd


@register_decomposition(aten._native_batch_norm_legit.no_stats)
def _native_batch_norm_legit_no_stats(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    training: bool,
    momentum: float,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    output, save_mean, save_rstd, _, _ = native_batch_norm_helper(
        input, weight, bias, None, None, training, momentum, eps, False
    )
    return output, save_mean, save_rstd


@register_decomposition(aten._native_batch_norm_legit_functional.default)
def _native_batch_norm_legit_functional(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    training: bool,
    momentum: float,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    (
        output,
        save_mean,
        save_rstd,
        new_running_mean,
        new_running_var,
    ) = native_batch_norm_helper(
        input, weight, bias, running_mean, running_var, training, momentum, eps, True
    )
    assert new_running_mean is not None, "new_running_mean should not be None"
    assert new_running_var is not None, "new_running_var should not be None"
    return output, save_mean, save_rstd, new_running_mean, new_running_var


def _get_batch_norm_reserve_tensor(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    eps: float,
    training: bool,
) -> Tensor:
    """
    Return a reserve tensor for batch norm, used only by cudnn to pass forward state to the
    backward pass. This is needed for `_batch_norm_with_update` and `_batch_norm_no_update`,
    which support a variety of backends including cudnn. We create this tensor here to get
    the correct shape in the traced graph if we detect that will call the cudnn kernel,
    and rely on DCE to avoid materializing this tensor.
    """
    backend = torch._C._select_batch_norm_backend(  # type: ignore[attr-defined]
        input, weight, bias, running_mean, running_var, True, eps
    )
    reserve_size = 0
    if backend == torch._C._BatchNormBackend.Cudnn:  # type: ignore[attr-defined]
        reserve_size = torch._C._get_cudnn_batch_norm_reserve_space_size(  # type: ignore[attr-defined]
            input, training
        )
    return torch.empty(
        reserve_size, dtype=torch.uint8, layout=input.layout, device=input.device
    )


@register_decomposition(aten._batch_norm_with_update.default)
def _batch_norm_with_update(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    momentum: float,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    output, save_mean, save_rstd, _, _ = native_batch_norm_helper(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        True,  # training
        momentum,
        eps,
        False,  # functional
    )
    reserve = _get_batch_norm_reserve_tensor(
        input, weight, bias, running_mean, running_var, eps, training=True
    )
    return output, save_mean, save_rstd, reserve


@register_decomposition(aten._batch_norm_with_update_functional.default)
def _batch_norm_with_update_functional(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    momentum: float,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    (
        output,
        save_mean,
        save_rstd,
        new_rm,
        new_rv,
    ) = native_batch_norm_helper(
        input, weight, bias, running_mean, running_var, True, momentum, eps, True
    )
    reserve = _get_batch_norm_reserve_tensor(
        input, weight, bias, running_mean, running_var, eps, training=True
    )
    assert new_rm is not None, "new_running_mean should not be None"
    assert new_rv is not None, "new_running_var should not be None"
    return (output, save_mean, save_rstd, reserve, new_rm, new_rv)


@register_decomposition(aten._batch_norm_no_update.default)
def _batch_norm_no_update(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    momentum: float,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    output, save_mean, save_rstd, _, _ = native_batch_norm_helper(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        False,  # training
        momentum,
        eps,
        False,  # functional
    )
    reserve = _get_batch_norm_reserve_tensor(
        input, weight, bias, running_mean, running_var, eps, training=False
    )
    return output, save_mean, save_rstd, reserve


@register_decomposition(aten._fused_dropout)
@out_wrapper("out0", "out1")
@pw_cast_for_opmath
def _fused_dropout_decomposition(input, p, generator=None):
    assert generator is None
    mask = (torch.rand_like(input) < p).to(dtype=torch.uint8)
    res = mask.type_as(input) * input * (1.0 / p)
    return (res, mask)


@register_decomposition(aten._to_copy)
@out_wrapper()
def _to_copy(
    x: Union[Tensor, NumberType],
    *,
    dtype: Optional[torch.dtype] = None,
    layout=None,
    device: Optional[torch.device] = None,
    pin_memory: bool = False,
    non_blocking: bool = False,
    memory_format: Optional[torch.memory_format] = None,
):
    assert not layout or layout == torch.strided, "TODO"
    assert not pin_memory, "TODO"
    assert isinstance(x, (torch.Tensor, int, float, bool, complex))
    if device is None and dtype is None and memory_format is None:
        if isinstance(x, torch.Tensor):
            return x.clone()
        else:
            return x
    dtype_converted = False

    if isinstance(x, torch.Tensor):
        x_tensor = x
    else:
        x_tensor = torch.scalar_tensor(x)

    if device is not None and device != x_tensor.device:
        # avoid conversions on cpu
        if dtype is not None and device.type == "cpu":
            x_tensor = torch._prims.convert_element_type(x_tensor, dtype)
            dtype_converted = True
        x_tensor = torch._prims.device_put(x_tensor, device, non_blocking)

    if dtype is not None and not dtype_converted:
        x_tensor = torch._prims.convert_element_type(x_tensor, dtype)
        dtype_converted = True

    if memory_format is not None:  # no ref/prim for memory format
        return torch.clone(x_tensor, memory_format=memory_format)
    return x_tensor


# Questionable decompositions
# This is only valid if we're running the graph without autograd, such as if the backward pass has been traced.
# Note that this decomposition causes issues with in-place ops
@register_decomposition([aten.detach, aten.lift, aten.lift_fresh])
@out_wrapper()
def nop_decomposition(x):
    return aten.alias(x)


# Also register to the Autograd dispatch key, so this decomp can run above autograd.
# native_batch_norm needs to decompose into other ops before autograd.
@aten.cudnn_batch_norm.default.py_impl(DispatchKey.Autograd)
@register_decomposition(aten.cudnn_batch_norm)
@out_wrapper("out0", "out1", "out2", "out3")
def cudnn_batch_norm(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    training: bool,
    exponential_average_factor: float,
    epsilon: float,
):
    a, b, c = aten.native_batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        exponential_average_factor,
        epsilon,
    )
    # Cudnn return running mean and variance when training is True
    if training:
        return (a, b, c, input.new_zeros((0,), dtype=torch.uint8))
    return (
        a,
        weight.new_zeros((0,)),
        weight.new_zeros((0,)),
        input.new_zeros((0,), dtype=torch.uint8),
    )


def _broadcast_batch_norm_backward(x, broadcast_mask):
    for axis, mask in enumerate(broadcast_mask):
        if mask == 1 and not (axis < x.ndim and x.shape[axis] == mask):
            x = x.unsqueeze(axis)
    return x


@register_decomposition(aten.batch_norm_backward.default)
def batch_norm_backward(
    grad_out: Tensor,
    input: Tensor,
    weight: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    save_mean: Optional[Tensor],
    save_invstd: Optional[Tensor],
    train: bool,
    eps: float,
    output_mask: list[bool],
    reserve: Tensor,
) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    return native_batch_norm_backward(
        grad_out,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        train,
        eps,
        output_mask,
    )


@register_decomposition(aten.native_batch_norm_backward.default)
def native_batch_norm_backward(
    grad_out: Tensor,
    input: Tensor,
    weight: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    save_mean: Optional[Tensor],
    save_invstd: Optional[Tensor],
    train: bool,
    eps: float,
    output_mask: list[bool],
) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    input_dtype = input.dtype
    if weight is not None:
        weight_dtype = weight.dtype
    else:
        weight_dtype = input_dtype
    computation_dtype = utils.get_computation_dtype(input.dtype)
    (
        grad_out_cast,
        input_cast,
        weight_cast,
        running_mean_cast,
        running_var_cast,
        save_mean_cast,
        save_invstd_cast,
    ) = (
        x.to(computation_dtype) if x is not None else x
        for x in (
            grad_out,
            input,
            weight,
            running_mean,
            running_var,
            save_mean,
            save_invstd,
        )
    )
    input_shape = input.shape
    input_rank = input.dim()
    assert input_rank >= 2, "rank of the input must be at least 2"

    axis = 1
    num_features = prod(list(input_shape)) / input_shape[axis]
    mean = save_mean_cast
    invstd = save_invstd_cast
    if train:
        assert mean is not None and invstd is not None

    else:
        assert running_mean_cast is not None and running_var_cast is not None
        mean = running_mean_cast
        invstd = torch.rsqrt(running_var_cast + eps)

    broadcast_mask: list[int] = [1] * input_rank
    broadcast_mask[axis] = input_shape[axis]

    reduction_axes: list[int] = []
    for i in range(input_rank):
        if i != axis:
            reduction_axes.append(i)

    mean = _broadcast_batch_norm_backward(mean, broadcast_mask)  # type: ignore[arg-type]
    norm = 1.0 / num_features
    grad_output_sum = torch.sum(grad_out_cast, reduction_axes)  # type: ignore[arg-type]
    dot_p = torch.sum(grad_out_cast * (input_cast - mean), reduction_axes)  # type: ignore[operator]

    grad_mean = _broadcast_batch_norm_backward(grad_output_sum * norm, broadcast_mask)
    proj_scale = _broadcast_batch_norm_backward(
        torch.mul(dot_p * norm, invstd * invstd),  # type: ignore[operator]
        broadcast_mask,
    )

    if weight_cast is None:
        grad_scale = _broadcast_batch_norm_backward(invstd, broadcast_mask) * 1.0  # type: ignore[arg-type]
    else:
        grad_scale = _broadcast_batch_norm_backward(
            invstd * weight_cast, broadcast_mask
        )

    if train:
        proj = (input_cast - mean) * proj_scale  # type: ignore[operator]
        grad_input = ((grad_out_cast - proj) - grad_mean) * grad_scale
    else:
        grad_input = grad_out_cast * grad_scale

    if output_mask[1]:
        grad_weight = dot_p * invstd
    else:
        grad_weight = None  # "None" doesn't work with vjp, should use zeros for vjp

    if output_mask[2]:
        grad_bias = grad_output_sum
    else:
        grad_bias = None  # "None" doesn't work with vjp, should use zeros for vjp

    return (
        grad_input.to(input_dtype),
        _maybe_cast(grad_weight, weight_dtype),
        _maybe_cast(grad_bias, weight_dtype),
    )


# out_wrapper currently does not allow optional outputs
@register_decomposition(aten.native_batch_norm_backward.out)
def native_batch_norm_backward_out(
    grad_out: Tensor,
    input: Tensor,
    weight: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    save_mean: Optional[Tensor],
    save_invstd: Optional[Tensor],
    train: bool,
    eps: float,
    output_mask: list[bool],
    *,
    out0: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    result = native_batch_norm_backward(
        grad_out,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        train,
        eps,
        output_mask,
    )
    grad_input = (out0, out1, out2)
    for i, r in enumerate(result):
        if r is not None:
            _maybe_resize_out(grad_input[i], r.shape)
            _safe_copy_out(copy_from=r, copy_to=grad_input[i], exact_dtype=True)

    return grad_input


@register_decomposition(aten.miopen_batch_norm_backward)
@out_wrapper("out0", "out1", "out2")
def miopen_batch_norm_backward(
    input: Tensor,
    grad_output: Tensor,
    weight: Tensor,
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    save_mean: Optional[Tensor],
    save_var: Optional[Tensor],
    epsilon: float,
):
    return aten.native_batch_norm_backward(
        grad_output,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_var,
        True,
        epsilon,
        [True, True, True],
    )


@register_decomposition(aten.cudnn_batch_norm_backward)
@out_wrapper("out0", "out1", "out2")
def cudnn_batch_norm_backward(
    input: Tensor,
    grad_output: Tensor,
    weight: Tensor,
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    save_mean: Optional[Tensor],
    save_var: Optional[Tensor],
    epsilon: float,
    reserveSpace: Tensor,
):
    return aten.native_batch_norm_backward(
        grad_output,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_var,
        True,
        epsilon,
        [True, True, True],
    )


@register_decomposition(aten._adaptive_avg_pool2d)
@out_wrapper()
@pw_cast_for_opmath
def adaptive_avg_pool2d(input: Tensor, output_size: tuple[int, int]):
    # Preconditions
    device = input.device
    shape = input.shape
    ndim = len(shape)
    torch._check(
        ndim in (3, 4),
        lambda: f"adaptive_avg_pool2d(): Expected 3D or 4D tensor, but got {ndim}",
    )
    for d in input.shape[-2:]:
        torch._check(
            d != 0,
            lambda: "adaptive_avg_pool2d(): Expected input to have non-zero size for "
            f"non-batch dimensions, but input has shape {tuple(shape)}.",
        )

    # Optimisation (we should also do this in the kernel implementation)
    if shape[-2] % output_size[-2] == 0 and shape[-1] % output_size[-1] == 0:
        stride = tuple(i // o for i, o in zip(shape[-2:], output_size))
        kernel = tuple(
            i - (o - 1) * s for i, o, s in zip(shape[-2:], output_size, stride)
        )
        return torch.nn.functional.avg_pool2d(input, kernel, stride)

    def start_index(a, b, c):
        return torch.div(a * c, b, rounding_mode="trunc")

    def end_index(a, b, c):
        return torch.div((a + 1) * c + b - 1, b, rounding_mode="trunc")

    def compute_idx(in_size, out_size):
        orange = torch.arange(out_size, device=device, dtype=torch.int64)
        i0 = start_index(orange, out_size, in_size)
        # Let length = end_index - start_index, i.e. the length of the pooling kernels
        # length.max() can be computed analytically as follows:
        maxlength = in_size // out_size + 1
        in_size_mod = in_size % out_size
        # adaptive = True iff there are kernels with different lengths
        adaptive = not (in_size_mod == 0 or out_size % in_size_mod == 0)
        if adaptive:
            maxlength += 1
        elif in_size_mod == 0:
            maxlength -= 1

        range_max = torch.arange(maxlength, device=device, dtype=torch.int64)
        idx = i0.unsqueeze(-1) + range_max
        if adaptive:
            # Need to clamp to avoid accessing out-of-bounds memory
            # TODO make minimum accept scalars
            maxval = torch.scalar_tensor(
                in_size - 1, dtype=idx.dtype, device=idx.device
            )
            idx = torch.minimum(idx, maxval)

            # Compute the length
            i1 = end_index(orange, out_size, in_size)
            length = i1 - i0
        else:
            length = maxlength
        return idx, length, range_max, adaptive

    # length is not None if it's constant, otherwise we'll need to compute it
    idxh, length_h, range_max_h, adaptive_h = compute_idx(shape[-2], output_size[-2])
    idxw, length_w, range_max_w, adaptive_w = compute_idx(shape[-1], output_size[-1])

    vals = input[..., _unsqueeze_to_dim(idxh, 4), idxw]
    # Shortcut for the simpler case
    if not adaptive_h and not adaptive_w:
        return torch.mean(vals, dim=(-3, -1))

    def maybe_mask(vals, length, range_max, adaptive, dim):
        if isinstance(length, IntLike):
            return vals, length
        else:
            # zero-out the things we didn't really want to select
            assert dim < 0
            # hack
            mask = range_max >= length.unsqueeze(-1)
            if dim == -2:
                mask = _unsqueeze_to_dim(mask, 4)
            vals = torch.masked_fill(vals, mask, 0.0)
            # Compute the length of each window
            length = _unsqueeze_to_dim(length, -dim)
            return vals, length

    vals, length_h = maybe_mask(
        vals, length_h, range_max_h, adaptive=adaptive_h, dim=-2
    )
    vals, length_w = maybe_mask(
        vals, length_w, range_max_w, adaptive=adaptive_w, dim=-1
    )

    # We unroll the sum as we assume that the kernels are going to be small
    ret = None
    for i, j in product(range(vals.shape[-3]), range(vals.shape[-1])):
        if ret is None:
            ret = vals[..., i, :, j]
        else:
            ret = ret + vals[..., i, :, j]
    return ret / (length_h * length_w)


def _max_unpoolnd(
    self: TensorLike, indices: TensorLike, output_size: list[int], dim: int
):
    # If the input tensors self and indices came from max_pool call as
    # required by the documentation, this operation is deterministic
    # because that ensures that if there are two entries in `indices`
    # tensor that are equal, the corresponding values in `self` are also
    # equal. If this condition is not satisfied, the operation is
    # non-deterministic as one of the different values in `self` 'wins'.
    utils.alert_not_deterministic(f"max_unpooling{dim}d_forward_out")
    nc = reduce(operator.mul, self.shape[:-dim])
    hw = reduce(operator.mul, output_size)
    indices_nc_shape = [1] * self.ndim
    indices_nc_shape[:-dim] = self.shape[:-dim]
    indices_flat = (
        indices + aten.arange(nc, device=self.device).view(indices_nc_shape) * hw
    ).reshape(-1)

    output = self.new_zeros(list(self.shape[:-dim]) + list(output_size))
    return aten._unsafe_index_put(
        output.reshape(-1), [indices_flat], self.reshape(-1), accumulate=False
    ).view(output.shape)


@register_decomposition(aten.max_unpool2d)
@out_wrapper()
def max_unpool2d(
    self: TensorLike,
    indices: TensorLike,
    output_size: list[int],
):
    torch._check(
        indices.dtype == torch.int64,
        lambda: f"elements in indices should be type int64 but got: {indices.dtype}",
    )
    torch._check(
        len(output_size) == 2,
        lambda: (
            f"There should be exactly two elements (height, width) in output_size, "
            f"but got {len(output_size)} elements."
        ),
    )

    torch._check(
        self.ndim in (3, 4),
        lambda: (
            f"Input to max_unpooling2d should be a 3d or 4d Tensor, "
            f"but got a tensor with {self.ndim} dimensions."
        ),
    )
    torch._check(
        self.shape == indices.shape,
        lambda: (
            f"Expected shape of indices to be same as that of the input tensor ({self.shape}) "
            f"but got indices tensor with shape: {indices.shape}"
        ),
    )

    for i in range(1, self.ndim):
        torch._check(
            self.size(i) > 0,
            lambda: (
                f"max_unpooling2d(): "
                f"Expected input to have non-zero size for non-batch dimensions, "
                f"but got {self.shape} with dimension {i} being empty."
            ),
        )

    return _max_unpoolnd(self, indices, output_size, 2)


@register_decomposition(aten.max_unpool3d)
@out_wrapper()
def max_unpool3d(
    input: TensorLike,
    indices: TensorLike,
    output_size: list[int],
    stride: list[int],
    padding: list[int],
):
    torch._check(
        indices.dtype == torch.int64, lambda: "elements in indices should be type int64"
    )
    torch._check(
        input.ndim in (4, 5),
        lambda: f"Input to max_unpooling3d should be a 4d or 5d Tensor, but got a tensor with {input.ndim} dimensions.",
    )
    torch._check(
        len(output_size) == 3,
        lambda: (
            f"There should be exactly three elements (depth, height, width) in output_size, "
            f"but got {len(output_size)} elements."
        ),
    )
    torch._check(
        len(stride) == 3,
        lambda: f"There should be exactly three elements (depth, height, width) in stride, but got: {len(stride)} elements.",
    )
    torch._check(
        len(padding) == 3,
        lambda: f"There should be exactly three elements (depth, height, width) in padding, but got: {len(padding)} elements.",
    )
    torch._check(
        input.shape == indices.shape,
        lambda: (
            f"Expected shape of indices to be same as that of the input tensor ({input.shape}) "
            f"but got indices tensor with shape: {indices.shape}"
        ),
    )

    for i in range(1, input.ndim):
        torch._check(
            input.size(i) > 0,
            lambda: (
                f"max_unpooling3d(): "
                f"Expected input to have non-zero size for non-batch dimensions, "
                f"but got {input.shape} with dimension {i} being empty."
            ),
        )

    torch._check(
        stride[0] > 0 and stride[1] > 0 and stride[2] > 0,
        lambda: f"strides should be greater than zero, but got stride: {stride}",
    )

    return _max_unpoolnd(input, indices, output_size, 3)


@register_decomposition(aten.index_add_)
def index_add_(
    x: TensorLike,
    dim: int,
    index: TensorLike,
    tensor: TensorLike,
    *,
    alpha: NumberType = 1,
):
    return _index_add(x, dim, index, tensor, inplace=True, alpha=alpha)


@register_decomposition(aten.index_add)
@out_wrapper()
def index_add(
    x: TensorLike,
    dim: int,
    index: TensorLike,
    tensor: TensorLike,
    *,
    alpha: NumberType = 1,
):
    return _index_add(x, dim, index, tensor, inplace=False, alpha=alpha)


def _index_add(
    x: TensorLike,
    dim: int,
    index: TensorLike,
    tensor: TensorLike,
    *,
    inplace: bool,
    alpha: NumberType = 1,
):
    dim = utils.canonicalize_dims(x.ndim, dim)
    torch._check(
        index.ndim <= 1,
        lambda: f"Index should have dimension 1 or 0 (got {index.ndim})",
    )
    index_size = index.size(0) if index.ndim == 1 else 1
    tensor_size = tensor.size(dim) if tensor.ndim > 0 else 1
    torch._check(
        tensor_size == index_size,
        lambda: f"Number of indices ({index_size}) should be equal to tensor.size(dim) ({tensor_size}), for {dim=}",
    )
    if alpha != 1:
        python_type = utils.dtype_to_type(x.dtype)
        torch._check(
            python_type is bool
            or utils.is_weakly_lesser_type(type(alpha), python_type),
            lambda: f"alpha argument of type {type(alpha)} cannot be safely cast to type {python_type}!",
        )
        tensor = tensor * alpha
    # Treat scalars as elements of \R^1
    zero_dim = x.ndim == 0
    x1 = x.unsqueeze(0) if zero_dim else x
    idx = (None,) * dim + (index,)
    index_put = aten.index_put_ if inplace else aten.index_put
    out = index_put(x1, idx, tensor, accumulate=True)
    if inplace:
        return x
    else:
        return out.squeeze(0) if zero_dim else out.contiguous()


@register_decomposition(aten.pad_sequence.default)
@aten.pad_sequence.default.py_impl(DispatchKey.CompositeImplicitAutograd)
def pad_sequence(sequences, batch_first=False, padding_value=0.0):
    torch._check(len(sequences) > 0, lambda: "received an empty list of sequences")
    sequences_size = len(sequences)
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(x.size(0) for x in sequences)
    if batch_first:
        out_dims = (sequences_size, max_len)
    else:
        out_dims = (max_len, sequences_size)
    out_dims = out_dims + trailing_dims
    out = sequences[0].new_full(out_dims, padding_value)
    dim_paddings = (0, 0) * len(trailing_dims)
    for i in range(sequences_size):
        currseq = sequences[i]
        row = aten.constant_pad_nd(
            currseq, dim_paddings + (0, max_len - currseq.size(0)), padding_value
        )
        if batch_first:
            out = aten.select_scatter(out, row, dim=0, index=i)
        else:
            out = aten.select_scatter(out, row, dim=1, index=i)
    return out


@register_decomposition(aten.index_copy_)
def index_copy_(x: TensorLike, dim: int, index: TensorLike, tensor: TensorLike):
    return _index_copy(x, dim, index, tensor, inplace=True)


@register_decomposition(aten.index_copy)
@out_wrapper()
def index_copy(x: TensorLike, dim: int, index: TensorLike, tensor: TensorLike):
    return _index_copy(x, dim, index, tensor, inplace=False)


def _index_copy(
    x: TensorLike, dim: int, index: TensorLike, tensor: TensorLike, *, inplace: bool
):
    dim = utils.canonicalize_dims(x.ndim, dim)
    torch._check(
        index.ndim <= 1,
        lambda: f"Index should have dimension 1 or 0 (got {index.ndim})",
    )
    # Treat scalars as elements of \R^1
    zero_dim = x.ndim == 0
    x1 = x.unsqueeze(0) if zero_dim else x
    index = index.unsqueeze(0) if index.ndim == 0 else index
    idx = (None,) * dim + (index,)
    index_put = aten.index_put_ if inplace else aten.index_put
    out = index_put(x1, idx, tensor)
    if inplace:
        return x
    else:
        return out.squeeze(0) if zero_dim else out.contiguous()


# nb: Should use acc_t, not op_math
@register_decomposition(aten.log_sigmoid_forward)
@out_wrapper("output", "buffer")
@pw_cast_for_opmath
def log_sigmoid_forward(self: Tensor) -> tuple[Tensor, Tensor]:
    min = torch.minimum(self.new_zeros(()), self)
    z = torch.exp(-torch.abs(self))
    if self.is_cuda or self.is_xpu:
        buffer = self.new_zeros((0,))
    else:
        buffer = z
    return min - torch.log1p(z), buffer


@register_decomposition(aten.uniform)
@out_wrapper()
def uniform(
    x: Tensor,
    low: Union[bool, int, float] = 0.0,
    high: Union[bool, int, float] = 1.0,
    generator: Optional[torch.Generator] = None,
):
    return prims._uniform_helper(
        x.shape,
        low=sym_float(low),
        high=sym_float(high),
        dtype=x.dtype,
        device=x.device,
        generator=generator,
    )


@register_decomposition(aten.uniform_)
def uniform_(self, low=0, high=1, generator=None):
    return self.copy_(uniform(self, low, high, generator))


# aten/src/ATen/native/UpSample.cpp compute_output_size
def upsample_compute_output_size(input_size, output_size, scale_factors):
    spatial_dimensions = len(input_size) - 2
    if output_size is not None:
        torch._check(
            scale_factors is None,
            lambda: "Must specify exactly one of output_size and scale_factors",
        )
        torch._check(len(output_size) == spatial_dimensions, lambda: "")
        return output_size
    if scale_factors is not None:
        # NB: this isn't necessary lol
        torch._check(
            output_size is None,
            lambda: "Must specify exactly one of output_size and scale_factors",
        )
        torch._check(len(scale_factors) == spatial_dimensions, lambda: "")
        output_size = []
        for i, s in enumerate(scale_factors):
            if int(s) == s:
                output_size.append(input_size[i + 2] * int(s))
            else:
                output_size.append(sym_int(input_size[i + 2] * s))
        return output_size
    torch._check(
        False, lambda: "Must specify exactly one of output_size and scale_factors"
    )


def get_scale_value(scales, idx):
    if scales is None:
        return None
    return scales[idx]


@register_decomposition(aten.upsample_nearest1d.vec)
@register_decomposition(aten.upsample_nearest2d.vec)
@register_decomposition(aten.upsample_nearest3d.vec)
@aten.upsample_nearest1d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_nearest1d.vec.py_impl(DispatchKey.Autograd)
@aten.upsample_nearest2d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_nearest2d.vec.py_impl(DispatchKey.Autograd)
@aten.upsample_nearest3d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_nearest3d.vec.py_impl(Dis

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 1 class(es): Reduction

### Functions
This file defines 251 function(s): type_casts, inner, increase_prec, decrease_prec, _unsqueeze_to_dim, tanh_backward, sigmoid_backward, softplus_backward, elu_backward, fill_scalar, fill_tensor, hardsigmoid, hardsigmoid_backward, hardtanh_backward, hardswish, hardswish_backward, threshold_backward, leaky_relu_backward, gelu_backward, mish_backward, silu, silu_backward, _prelu_kernel, _prelu_kernel_backward, rrelu_with_noise_backward, log_sigmoid_backward, apply_loss_reduction, to_real_dtype, mse_loss, mse_loss_backward


## Key Components

The file contains 17964 words across 5376 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 181990 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
