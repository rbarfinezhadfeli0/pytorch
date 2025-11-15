# Documentation: `docs/torch/overrides.py_docs.md`

## File Metadata

- **Path**: `docs/torch/overrides.py_docs.md`
- **Size**: 54,310 bytes (53.04 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/overrides.py`

## File Metadata

- **Path**: `torch/overrides.py`
- **Size**: 105,497 bytes (103.02 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""
Python implementation of ``__torch_function__``

While most of the torch API and handling for ``__torch_function__`` happens
at the C++ level, some of the torch API is written in Python so we need
python-level handling for ``__torch_function__`` overrides as well. The main
developer-facing functionality in this file are handle_torch_function and
has_torch_function. See torch/functional.py and test/test_overrides.py
for usage examples.

Note
----
heavily inspired by NumPy's ``__array_function__`` (see:
https://github.com/pytorch/pytorch/issues/24015 and
https://www.numpy.org/neps/nep-0018-array-function-protocol.html
)

If changing this file in a way that can affect ``__torch_function__`` overhead,
please report the benchmarks in ``benchmarks/overrides_benchmark``. See the
instructions in the ``README.md`` in that directory.
"""

import __future__  # noqa: F404

import collections
import contextlib
import functools
import types
import warnings
from collections.abc import Callable, Iterable
from functools import wraps
from typing import Any, Optional, TypeVar
from typing_extensions import ParamSpec

import torch
from torch._C import (
    _add_docstr,
    _get_function_stack_at,
    _has_torch_function,
    _has_torch_function_unary,
    _has_torch_function_variadic,
    _is_torch_function_mode_enabled,
    _len_torch_function_stack,
    _pop_torch_function_stack,
    _push_on_torch_function_stack,
)


__all__ = [
    "get_ignored_functions",
    "get_overridable_functions",
    "get_testing_overrides",
    "handle_torch_function",
    "has_torch_function",
    "resolve_name",
    "is_tensor_like",
    "is_tensor_method_or_property",
    "wrap_torch_function",
    "enable_reentrant_dispatch",
]

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _disable_user_warnings(
    func: Callable[_P, _R],
    regex: str = ".*is deprecated, please use.*",
    module: str = "torch",
) -> Callable[_P, _R]:
    """
    Decorator that temporarily disables ``UserWarning``s for the given ``module`` if the warning message matches the
    given ``regex`` pattern.

    Arguments
    ---------
    func : function
        Function to disable the warnings for.
    regex : str
        A regex pattern compilable by ``re.compile``. This is used to match the ``UserWarning`` message.
    module : str
        The python module to which the filtering should be restricted.

    Returns
    -------
    function
        The wrapped function.
    """

    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, message=regex, module=module
            )
            return func(*args, **kwargs)

    return wrapper


@functools.cache
@_disable_user_warnings
def get_ignored_functions() -> set[Callable]:
    """
    Return public functions that cannot be overridden by ``__torch_function__``.

    Returns
    -------
    set[Callable]
        A tuple of functions that are publicly available in the torch API but cannot
        be overridden with ``__torch_function__``. Mostly this is because none of the
        arguments of these functions are tensors or tensor-likes.

    Examples
    --------
    >>> torch.Tensor.as_subclass in torch.overrides.get_ignored_functions()
    True
    >>> torch.add in torch.overrides.get_ignored_functions()
    False
    """
    Tensor = torch.Tensor
    return {
        torch.typename,
        torch.is_tensor,
        torch.is_storage,
        torch.set_default_tensor_type,
        torch.set_default_device,
        torch.get_default_device,
        torch.set_rng_state,
        torch.get_rng_state,
        torch.manual_seed,
        torch.initial_seed,
        torch.seed,
        torch.save,
        torch.load,
        torch.set_printoptions,
        torch.fork,
        torch.get_default_dtype,
        torch.get_num_interop_threads,
        torch.get_num_threads,
        torch.init_num_threads,
        torch.import_ir_module,
        torch.import_ir_module_from_buffer,
        torch.is_anomaly_enabled,
        torch.is_anomaly_check_nan_enabled,
        torch.is_grad_enabled,
        torch.merge_type_from_type_comment,
        torch.parse_ir,
        torch.parse_schema,
        torch.parse_type_comment,
        torch.set_anomaly_enabled,
        torch.set_flush_denormal,
        torch.set_num_interop_threads,
        torch.set_num_threads,
        torch.wait,
        torch.as_tensor,
        torch.from_numpy,
        torch.tensor,
        torch.default_generator,
        torch.has_cuda,
        torch.has_cudnn,
        torch.has_lapack,
        torch.device,
        torch.dtype,
        torch.finfo,
        torch.has_mkl,
        torch.has_mps,
        torch.has_mkldnn,
        torch.has_openmp,
        torch.iinfo,
        torch.memory_format,
        torch.qscheme,
        torch.set_grad_enabled,
        torch.no_grad,
        torch.enable_grad,
        torch.inference_mode,
        torch.is_inference_mode_enabled,
        torch.layout,
        torch.align_tensors,
        torch.arange,
        torch.as_strided,
        torch.bartlett_window,
        torch.blackman_window,
        torch.broadcast_shapes,
        torch.can_cast,
        torch.compile,
        torch.cudnn_affine_grid_generator,
        torch.cudnn_batch_norm,
        torch.cudnn_convolution,
        torch.cudnn_convolution_transpose,
        torch.cudnn_convolution_relu,
        torch.cudnn_convolution_add_relu,
        torch.cudnn_grid_sampler,
        torch.cudnn_is_acceptable,
        torch.empty,
        torch.empty_permuted,
        torch.empty_strided,
        torch.empty_quantized,
        torch.export.export,
        torch.export.load,
        torch.export.register_dataclass,
        torch.export.save,
        torch.eye,
        torch.fft.fftfreq,
        torch.fft.rfftfreq,
        torch.from_file,
        torch.full,
        torch.fill,
        torch.hamming_window,
        torch.hann_window,
        torch.kaiser_window,
        torch.linspace,
        torch.logspace,
        torch.mkldnn_adaptive_avg_pool2d,
        torch.mkldnn_convolution,
        torch.mkldnn_max_pool2d,
        torch.mkldnn_max_pool3d,
        torch.mkldnn_linear_backward_weights,
        torch.mkldnn_rnn_layer,
        torch.normal,
        torch.ones,
        torch.promote_types,
        torch.rand,
        torch.rand_like,
        torch.randn,
        torch.randn_like,
        torch.randint,
        torch.randint_like,
        torch.randperm,
        torch.range,
        torch.result_type,
        torch.scalar_tensor,
        torch.sparse_coo_tensor,
        torch.sparse_compressed_tensor,
        torch.sparse_csr_tensor,
        torch.sparse_csc_tensor,
        torch.sparse_bsr_tensor,
        torch.sparse_bsc_tensor,
        torch.sym_constrain_range,
        torch.sym_constrain_range_for_size,
        torch.sym_fresh_size,
        torch.tril_indices,
        torch.triu_indices,
        torch.vander,
        torch.zeros,
        torch._jit_internal.boolean_dispatch,
        torch.nn.functional.assert_int_or_pair,
        torch.nn.functional.upsample,
        torch.nn.functional.upsample_bilinear,
        torch.nn.functional.upsample_nearest,
        torch.nn.functional.has_torch_function,
        torch.nn.functional.has_torch_function_unary,
        torch.nn.functional.has_torch_function_variadic,
        torch.nn.functional.handle_torch_function,
        torch.nn.functional.scaled_grouped_mm,
        torch.nn.functional.scaled_mm,
        torch.nn.functional.sigmoid,
        torch.nn.functional.hardsigmoid,
        torch.nn.functional.tanh,
        torch.nn.functional._canonical_mask,
        torch.nn.functional._none_or_dtype,
        # Doesn't actually take or return tensor arguments
        torch.nn.init.calculate_gain,
        # These are deprecated; don't test them
        torch.nn.init.uniform,
        torch.nn.init.normal,
        torch.nn.init.constant,
        torch.nn.init.eye,
        torch.nn.init.dirac,
        torch.nn.init.xavier_uniform,
        torch.nn.init.xavier_normal,
        torch.nn.init.kaiming_uniform,
        torch.nn.init.kaiming_normal,
        torch.nn.init.orthogonal,
        torch.nn.init.sparse,
        torch.nested.to_padded_tensor,
        has_torch_function,
        handle_torch_function,
        torch.set_autocast_enabled,
        torch.is_autocast_enabled,
        torch.set_autocast_dtype,
        torch.get_autocast_dtype,
        torch.clear_autocast_cache,
        torch.set_autocast_cpu_enabled,
        torch.is_autocast_cpu_enabled,
        torch.set_autocast_xla_enabled,
        torch.is_autocast_xla_enabled,
        torch.set_autocast_ipu_enabled,
        torch.is_autocast_ipu_enabled,
        torch.set_autocast_cpu_dtype,
        torch.get_autocast_cpu_dtype,
        torch.set_autocast_ipu_dtype,
        torch.get_autocast_ipu_dtype,
        torch.get_autocast_gpu_dtype,
        torch.set_autocast_gpu_dtype,
        torch.get_autocast_xla_dtype,
        torch.set_autocast_xla_dtype,
        torch.autocast_increment_nesting,
        torch.autocast_decrement_nesting,
        torch.is_autocast_cache_enabled,
        torch.set_autocast_cache_enabled,
        torch.nn.functional.hardswish,
        torch.is_vulkan_available,
        torch.are_deterministic_algorithms_enabled,
        torch.use_deterministic_algorithms,
        torch.is_deterministic_algorithms_warn_only_enabled,
        torch.set_deterministic_debug_mode,
        torch.get_device_module,
        torch.get_deterministic_debug_mode,
        torch.set_float32_matmul_precision,
        torch.get_float32_matmul_precision,
        torch.unify_type_list,
        torch.is_warn_always_enabled,
        torch.set_warn_always,
        torch.vitals_enabled,
        torch.set_vital,
        torch.read_vitals,
        torch.vmap,
        torch.cond,
        torch.frombuffer,
        torch.asarray,
        torch._functional_sym_constrain_range,
        torch._make_dep_token,
        Tensor.__delitem__,
        Tensor.__dir__,
        Tensor.__getattribute__,
        Tensor.__init__,
        Tensor.__iter__,
        Tensor.__init_subclass__,
        Tensor.__delattr__,
        Tensor.__setattr__,
        Tensor.__torch_function__,
        Tensor.__torch_dispatch__,
        Tensor.__new__,
        Tensor.__class__,
        Tensor.__subclasshook__,
        Tensor.__hash__,
        Tensor.as_subclass,
        Tensor.eig,
        Tensor.lstsq,
        Tensor.reinforce,
        Tensor.new,
        Tensor.new_tensor,
        Tensor.new_empty,
        Tensor.new_empty_strided,
        Tensor.new_zeros,
        Tensor.new_ones,
        Tensor.new_full,
        Tensor._make_subclass,
        Tensor.solve,
        Tensor.symeig,
        Tensor.stride,
        Tensor.unflatten,
        Tensor.to_sparse_coo,
        Tensor.to_sparse_csr,
        Tensor.to_sparse_csc,
        Tensor.to_sparse_bsr,
        Tensor.to_sparse_bsc,
        Tensor._to_sparse,
        Tensor._to_sparse_csr,
        Tensor._to_sparse_csc,
        Tensor._to_sparse_bsr,
        Tensor._to_sparse_bsc,
        Tensor._typed_storage,
        Tensor._reduce_ex_internal,
        Tensor._fix_weakref,
        Tensor._view_func,
        Tensor._view_func_unsafe,
        Tensor._rev_view_func_unsafe,
        Tensor._dtensor__new__,
        Tensor._make_wrapper_subclass,
        Tensor._python_dispatch.__get__,
        Tensor._has_symbolic_sizes_strides.__get__,
        Tensor._conj,
        Tensor._conj_physical,
        Tensor._lazy_clone,
        Tensor._neg_view,
        Tensor._is_zerotensor,
        Tensor._is_all_true,
        Tensor._is_any_true,
        Tensor._addmm_activation,
        Tensor.to_padded_tensor,
        Tensor._use_count,
    }


@functools.cache
def get_default_nowrap_functions() -> set[Callable]:
    """
    Return public functions that do not wrap in a subclass when invoked by
    the default ``Tensor.__torch_function__`` that preserves subclasses.  Typically,
    these functions represent field accesses (i.e., retrieving a Tensor that
    is stored somewhere on the Tensor) as opposed to computation.  Users of
    these functions expect object identity to be preserved over multiple accesses
    (e.g., ``a.grad is a.grad``) which cannot be upheld if we're wrapping on
    the fly every time (furthermore, the tensor stored here might already be
    the subclass, in which case wrapping really ought not to happen).

    Not ALL property accessors have this property; for example ``Tensor.T`` actually
    just creates a new transposed tensor on the fly, and so we SHOULD interpose on
    these calls (you need to check the implementation of the function to see if
    this is the case or not).  Additionally, if a property accessor doesn't return a Tensor,
    it doesn't have to be on this list (though it is harmless if it is).
    """
    Tensor = torch.Tensor
    return {
        Tensor._base.__get__,
        Tensor.grad.__get__,
        Tensor._grad.__get__,
    }


@functools.cache
@_disable_user_warnings
def get_testing_overrides() -> dict[Callable, Callable]:
    """Return a dict containing dummy overrides for all overridable functions

    Returns
    -------
    Dict[Callable, Callable]
        A dictionary that maps overridable functions in the PyTorch API to
        lambda functions that have the same signature as the real function
        and unconditionally return -1. These lambda functions are useful
        for testing API coverage for a type that defines ``__torch_function__``.

    Examples
    --------
    >>> import inspect
    >>> my_add = torch.overrides.get_testing_overrides()[torch.add]
    >>> inspect.signature(my_add)
    <Signature (input, other, out=None)>
    """
    # Every function in the PyTorchAPI that can be overridden needs an entry
    # in this dict.
    #
    # Optimally we would use inspect to get the function signature and define
    # the lambda function procedurally but that is blocked by generating
    # function signatures for native kernels that can be consumed by inspect.
    # See Issue #28233.
    Tensor = torch.Tensor
    ret: dict[Callable, Callable] = {
        torch.abs: lambda input, out=None: -1,
        torch.absolute: lambda input, out=None: -1,
        torch.adaptive_avg_pool1d: lambda input, output_size: -1,
        torch.adaptive_max_pool1d: lambda inputs, output_size: -1,
        torch.acos: lambda input, out=None: -1,
        torch.adjoint: lambda input: -1,
        torch.arccos: lambda input, out=None: -1,
        torch.acosh: lambda input, out=None: -1,
        torch.arccosh: lambda input, out=None: -1,
        torch.add: lambda input, other, out=None: -1,
        torch.addbmm: lambda input, batch1, batch2, alpha=1, beta=1, out=None: -1,
        torch.addcdiv: lambda input, tensor1, tensor2, value=1, out=None: -1,
        torch.addcmul: lambda input, tensor1, tensor2, value=1, out=None: -1,
        torch.addmm: lambda input, mat1, mat2, beta=1, alpha=1, out=None: -1,
        torch.addmv: lambda input, mat, vec, beta=1, alpha=1, out=None: -1,
        torch.addr: lambda input, vec1, vec2, beta=1, alpha=1, out=None: -1,
        torch.affine_grid_generator: lambda theta, size, align_corners: -1,
        torch.all: lambda input, dim=None: -1,
        torch.allclose: lambda input, other, trol=1e-05, atol=1e-08, equal_nan=False: -1,
        torch.alpha_dropout: lambda input, p, train, inplace=False: -1,
        torch.amax: lambda input, dim=None: -1,
        torch.amin: lambda input, dim=None: -1,
        torch.aminmax: lambda input, dim=None, keepdim=False, out=None: -1,
        torch.angle: lambda input, out=None: -1,
        torch.any: lambda input, dim=None, keepdim=False, out=None: -1,
        torch.argmax: lambda input: -1,
        torch.argmin: lambda input: -1,
        torch.argsort: lambda input, dim=None: -1,
        torch.asin: lambda input, out=None: -1,
        torch._assert_async: lambda input, msg: -1,
        torch.arcsin: lambda input, out=None: -1,
        torch.asinh: lambda input, out=None: -1,
        torch.arcsinh: lambda input, out=None: -1,
        torch.atan: lambda input, out=None: -1,
        torch.arctan: lambda input, out=None: -1,
        torch.atan2: lambda input, other, out=None: -1,
        torch.arctan2: lambda input, other, out=None: -1,
        torch.atanh: lambda input, out=None: -1,
        torch.arctanh: lambda input, out=None: -1,
        torch.atleast_1d: lambda *tensors: -1,
        torch.atleast_2d: lambda *tensors: -1,
        torch.atleast_3d: lambda *tensors: -1,
        torch.avg_pool1d: lambda input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True: -1,
        torch.baddbmm: lambda input, batch1, batch2, alpha=1, beta=1, out=None: -1,
        torch.batch_norm: lambda input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled: -1,
        torch.batch_norm_backward_elemt: lambda grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count_tensor: -1,
        torch.batch_norm_backward_reduce: lambda grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g: -1,
        torch.batch_norm_elemt: lambda input, weight, bias, mean, invstd, eps: -1,
        torch.batch_norm_gather_stats: lambda input, mean, invstd, running_mean, running_var, momentum, eps, count: -1,
        torch.batch_norm_gather_stats_with_counts: lambda input, mean, invstd, running_mean, running_var, momentum, eps, count: -1,
        torch.batch_norm_stats: lambda input, eps: -1,
        torch.batch_norm_update_stats: lambda input, running_mean, running_var, momentum: -1,
        torch.bernoulli: lambda input, generator=None, out=None: -1,
        torch.bilinear: lambda input1, input2, weight, bias: -1,
        torch.binary_cross_entropy_with_logits: (
            lambda input, target, weight=None, size_average=None, reduce=None, reduction="mean", pos_weight=None: -1
        ),
        torch.bincount: lambda input, weights=None, minlength=0: -1,
        torch.binomial: lambda count, prob, generator=None: -1,
        torch.bitwise_and: lambda input, other, out=None: -1,
        torch.bitwise_not: lambda input, out=None: -1,
        torch.bitwise_or: lambda input, other, out=None: -1,
        torch.bitwise_xor: lambda input, other, out=None: -1,
        torch.bitwise_left_shift: lambda input, other, out=None: -1,
        torch.bitwise_right_shift: lambda input, other, out=None: -1,
        torch.block_diag: lambda *tensors: -1,
        torch.bmm: lambda input, mat2, out_dtype=None, out=None: -1,
        torch.broadcast_tensors: lambda *tensors: -1,
        torch.broadcast_to: lambda self, size: -1,
        torch.bucketize: lambda input, boundaries, out_int32=False, right=False, out=None: -1,
        torch.cartesian_prod: lambda *tensors: -1,
        torch.cat: lambda tensors, dim=0, out=None: -1,
        torch.concat: lambda tensors, dim=0, out=None: -1,  # alias for torch.cat
        torch.concatenate: lambda tensors, dim=0, out=None: -1,  # alias for torch.concatenate
        torch.cdist: lambda x1, x2, p=2.0, compute_mode="use_mm_for_euclid_dist_if_necessary": -1,
        torch.ceil: lambda input, out=None: -1,
        torch.celu: lambda input, alpha=1.0, inplace=False: -1,
        torch.chain_matmul: lambda *matrices, out=None: -1,
        torch.channel_shuffle: lambda input, groups: -1,
        torch.cholesky: lambda input, upper=False, out=None: -1,
        torch.linalg.cholesky: lambda input, out=None: -1,
        torch.linalg.cholesky_ex: lambda input, check_errors=False, out=None: -1,
        torch.cholesky_inverse: lambda input, upper=False, out=None: -1,
        torch.cholesky_solve: lambda input1, input2, upper=False, out=None: -1,
        torch.choose_qparams_optimized: lambda input, numel, n_bins, ratio, bit_width: -1,
        torch.chunk: lambda input, chunks, dim=0: -1,
        torch.clamp: lambda input, min=None, max=None, out=None: -1,
        torch.clip: lambda input, min=None, max=None, out=None: -1,
        torch.clamp_min: lambda input, min, out=None: -1,
        torch.clamp_max: lambda input, max, out=None: -1,
        torch.column_stack: lambda tensors, out=None: -1,
        torch.cov: lambda input, correction=1, fweights=None, aweights=None: -1,
        torch.clone: lambda input: -1,
        torch.combinations: lambda input, r=2, with_replacement=False: -1,
        torch.complex: lambda real, imag: -1,
        torch.copysign: lambda input, other, out=None: -1,
        torch.polar: lambda abs, ang: -1,
        torch.linalg.cond: lambda input, ord=None: -1,
        torch.conj: lambda input, out=None: -1,
        torch.conj_physical: lambda input, out=None: -1,
        torch.resolve_conj: lambda input, out=None: -1,
        torch.resolve_neg: lambda input, out=None: -1,
        torch.constant_pad_nd: lambda input, pad, value=0: -1,
        torch.conv1d: lambda input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1: -1,
        torch.conv2d: lambda input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1: -1,
        torch.conv3d: lambda input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1: -1,
        torch.convolution: lambda input, weight, bias, stride, padding, dilation, transposed, output_adding, groups: -1,
        torch.conv_tbc: lambda input, weight, bias, pad=0: -1,
        torch.conv_transpose1d: lambda input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1: -1,
        torch.conv_transpose2d: lambda input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1: -1,
        torch.conv_transpose3d: lambda input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1: -1,
        torch.corrcoef: lambda input: -1,
        torch.cos: lambda input, out=None: -1,
        torch.cosine_embedding_loss: lambda input1, input2, target, margin=0, size_average=None, reduce=None, reduction="mean": -1,
        torch.cosh: lambda input, out=None: -1,
        torch.cosine_similarity: lambda x1, x2, dim=1, eps=1e-8: -1,
        torch.count_nonzero: lambda input: -1,
        torch.cross: lambda input, other, dim=None, out=None: -1,
        torch.linalg.cross: lambda input, other, dim=-1, out=None: -1,
        torch.ctc_loss: (
            lambda log_probs, targets, input_lengths, target_lengths, blank=0, reduction="mean", zero_infinity=False: -1
        ),
        torch.cummax: lambda input, dim, out=None: -1,
        torch.cummin: lambda input, dim, out=None: -1,
        torch.cumprod: lambda input, dim, out=None, dtype=None: -1,
        torch.cumsum: lambda input, dim, out=None, dtype=None: -1,
        torch.cumulative_trapezoid: lambda y, x=None, dim=-1: -1,
        torch.logcumsumexp: lambda input, dim, out=None: -1,
        torch.deg2rad: lambda input, out=None: -1,
        torch.dequantize: lambda input: -1,
        torch.det: lambda input: -1,
        torch.linalg.det: lambda input: -1,  # alias for torch.det  # type: ignore[attr-defined]
        torch.detach: lambda input: -1,
        torch.diag: lambda input, diagonal=0, out=None: -1,
        torch.diag_embed: lambda input, diagonal=0, out=None: -1,
        torch.diagflat: lambda input, offset=0: -1,
        torch.diff: lambda input, n=1, dim=-1, prepend=None, append=None, out=None: -1,
        torch.diagonal: lambda input, offset=0, dim1=0, dim2=1: -1,
        torch.linalg.diagonal: lambda input, offset=0, dim1=-2, dim2=-1: -1,
        torch.diagonal_scatter: lambda input, src, offset=0, dim1=0, dim2=1: -1,
        torch.as_strided_scatter: lambda self, src, size, stride, storage_offset=None: -1,
        torch.digamma: lambda input, out=None: -1,
        torch.dist: lambda input, other, p=2: -1,
        torch.div: lambda input, other, rounding_mode=None, out=None: -1,
        torch.divide: lambda input, other, rounding_mode=None, out=None: -1,
        torch.dot: lambda input, other, out=None: -1,
        torch.dropout: lambda input, p, train, inplace=False: -1,
        torch.dsmm: lambda input, mat2, out_dtype=None: -1,
        torch.hsmm: lambda mat1, mat2: -1,
        torch.dsplit: lambda input, indices_or_sections: -1,
        torch.dstack: lambda tensors, out=None: -1,
        torch.linalg.eig: lambda input, out=None: -1,
        torch.linalg.eigvals: lambda input, out=None: -1,
        torch.linalg.eigh: lambda input, UPLO="L", out=None: -1,
        torch.linalg.eigvalsh: lambda input, UPLO="L", out=None: -1,
        torch.einsum: lambda equation, *operands: -1,
        torch.embedding: (
            lambda input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False: -1  # noqa: B950
        ),
        torch.embedding_bag: (
            lambda input, weight, offsets, max_norm=None, norm_type=2, scale_grad_by_freq=False, mode="mean", sparse=False, per_sample_weights=None, padding_idx=None: -1  # noqa: B950
        ),
        torch.empty_like: lambda input, dtype=None, layout=None, device=None, requires_grad=False: -1,
        torch.eq: lambda input, other, out=None: -1,
        torch.equal: lambda input, other: -1,
        torch.erf: lambda input, out=None: -1,
        torch.erfc: lambda input, out=None: -1,
        torch.erfinv: lambda input, out=None: -1,
        torch.exp: lambda input, out=None: -1,
        torch.exp2: lambda input, out=None: -1,
        torch.expm1: lambda input, out=None: -1,
        torch.fake_quantize_per_channel_affine: lambda input, scale, zero_point, axis, quant_min, quant_max: -1,
        torch.fake_quantize_per_tensor_affine: lambda input, scale, zero_point, quant_min, quant_max: -1,
        torch.fused_moving_avg_obs_fake_quant: (
            lambda x, observer_on, fake_quant_on, averaging_const, running_min, running_max, scale, zero_point, quant_min, quant_max, ch_axis, per_row_fake_quant=False, symmetric_quant=False: -1  # noqa: B950
        ),
        torch.fbgemm_linear_fp16_weight: lambda input, packed_weight, bias, output: -1,
        torch.fbgemm_linear_fp16_weight_fp32_activation: lambda input, packed_weight, bias, output: -1,
        torch.fbgemm_linear_int8_weight: lambda input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias: -1,  # noqa: B950
        torch.fbgemm_linear_int8_weight_fp32_activation: (
            lambda input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias: -1
        ),
        torch.fbgemm_linear_quantize_weight: lambda input: -1,
        torch.fbgemm_pack_gemm_matrix_fp16: lambda input: -1,
        torch.fbgemm_pack_quantized_matrix: lambda input, a, b: -1,
        torch.feature_alpha_dropout: lambda input, p, train: -1,
        torch.feature_dropout: lambda input, p, train: -1,
        torch.fft.ifft: lambda input, n=None, dim=-1, norm=None: -1,
        torch.fft.rfft: lambda input, n=None, dim=-1, norm=None: -1,
        torch.fft.irfft: lambda input, n=None, dim=-1, norm=None: -1,
        torch.fft.hfft: lambda input, n=None, dim=-1, norm=None: -1,
        torch.fft.ihfft: lambda input, n=None, dim=-1, norm=None: -1,
        torch.fft.hfft2: lambda input, s=None, dim=(-2, -1), norm=None: -1,
        torch.fft.ihfft2: lambda input, s=None, dim=(-2, -1), norm=None: -1,
        torch.fft.hfftn: lambda input, s=None, dim=-1, norm=None: -1,
        torch.fft.ihfftn: lambda input, s=None, dim=-1, norm=None: -1,
        torch.fft.fftn: lambda input, s=None, dim=None, norm=None: -1,
        torch.fft.ifftn: lambda input, s=None, dim=None, norm=None: -1,
        torch.fft.rfftn: lambda input, s=None, dim=None, norm=None: -1,
        torch.fft.irfftn: lambda input, s=None, dim=None, norm=None: -1,
        torch.fft.fft2: lambda input, s=None, dim=(-2, -1), norm=None: -1,
        torch.fft.ifft2: lambda input, s=None, dim=(-2, -1), norm=None: -1,
        torch.fft.rfft2: lambda input, s=None, dim=(-2, -1), norm=None: -1,
        torch.fft.irfft2: lambda input, s=None, dim=(-2, -1), norm=None: -1,
        torch.fft.fftshift: lambda input, dim=None: -1,
        torch.fft.ifftshift: lambda input, dim=None: -1,
        torch.fft.fft: lambda input, n=None, dim=-1, norm=None: -1,
        torch.fix: lambda input, out=None: -1,
        torch.flatten: lambda input, start_dim=0, end_dim=-1: -1,
        torch.flip: lambda input, dims: -1,
        torch.fliplr: lambda input: -1,
        torch.flipud: lambda input: -1,
        torch.frobenius_norm: lambda input, dim=None, keepdim=False, out=None: -1,
        torch.floor: lambda input, out=None: -1,
        torch.floor_divide: lambda input, other: -1,
        torch.float_power: lambda input, exponent, out=None: -1,
        torch.fmod: lambda input, other, out=None: -1,
        torch.frac: lambda input, out=None: -1,
        torch.frexp: lambda input, out=None: -1,
        torch.full_like: lambda input, fill_value, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False: -1,  # noqa: B950
        torch._functional_assert_async: lambda input, msg, dep_token: -1,
        torch.lu_unpack: lambda LU_data, LU_pivots, unpack_data=True, unpack_pivots=True: -1,
        torch.gather: lambda input, dim, index, out=None, sparse_grad=False: -1,
        torch.gcd: lambda input, other, out=None: -1,
        torch.ge: lambda input, other, out=None: -1,
        torch.get_device: lambda input: -1,
        torch.greater_equal: lambda input, other, out=None: -1,
        torch.geqrf: lambda input, out=None: -1,
        torch.i0: lambda input, out=None: -1,
        torch.inner: lambda input, other, out=None: -1,
        torch.outer: lambda input, vec2, out=None: -1,
        torch.ger: lambda input, vec2, out=None: -1,  # alias for torch.outer
        torch.gradient: lambda input, spacing=None, dim=None, edge_order=1: -1,
        torch.grid_sampler: lambda input, grid, interpolation_mode, padding_mode, align_corners: -1,
        torch.grid_sampler_2d: lambda input, grid, interpolation_mode, padding_mode, align_corners: -1,
        torch.grid_sampler_3d: lambda input, grid, interpolation_mode, padding_mode, align_corners: -1,
        torch.group_norm: lambda input, num_groups, weight=None, bias=None, eps=1e-05, cudnn_enabled=True: -1,
        torch.gru: lambda input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first: -1,
        torch.gru_cell: lambda input, hx, w_ih, w_hh, b_ih=None, b_hh=None: -1,
        torch.gt: lambda input, other, out=None: -1,
        torch.greater: lambda input, other, out=None: -1,
        torch.hardshrink: lambda input, lambd=0.5: -1,
        torch.hash_tensor: lambda input, dim=None, keepdim=False, mode=0, out=None: -1,
        torch.heaviside: lambda input, values, out=None: -1,
        torch.hinge_embedding_loss: lambda input, target, margin=1.0, size_average=None, reduce=None, reduction="mean": -1,  # noqa: B950
        torch.histc: lambda input, bins=100, min=0, max=0, out=None: -1,
        torch.histogram: lambda input, bins=100, min=None, max=None, weight=None, density=False, out=None: -1,
        torch.histogramdd: lambda input, bins, range=None, weight=None, density=False: -1,
        torch.linalg.householder_product: lambda input, tau: -1,
        torch.hspmm: lambda mat1, mat2, out=None: -1,
        torch.hsplit: lambda input, indices_or_sections: -1,
        torch.hstack: lambda tensors, out=None: -1,
        torch.hypot: lambda input, other, out=None: -1,
        torch.igamma: lambda input, other, out=None: -1,
        torch.igammac: lambda input, other, out=None: -1,
        torch.imag: lambda input, out=None: -1,
        torch.index_add: lambda input, dim, index, source: -1,
        torch.index_copy: lambda input, dim, index, source: -1,
        torch.index_put: lambda input, indices, values, accumulate=False: -1,
        torch.index_select: lambda input, dim, index, out=None: -1,
        torch.index_fill: lambda input, dim, index, value: -1,
        torch.index_reduce: lambda input, dim, index, source, reduce, include_input=True: -1,
        torch.isfinite: lambda tensor: -1,
        torch.isin: lambda e, te, assume_unique=False, invert=False: -1,
        torch.isinf: lambda tensor: -1,
        torch.isreal: lambda tensor: -1,
        torch.isposinf: lambda input, out=None: -1,
        torch.isneginf: lambda input, out=None: -1,
        torch.instance_norm: (
            lambda input, running_mean, running_var, weight, bias, use_input_stats, momentum, eps, cudnn_enabled: -1
        ),
        torch.int_repr: lambda input: -1,
        torch.inverse: lambda input, out=None: -1,
        torch.linalg.inv: lambda input, out=None: -1,
        torch.linalg.inv_ex: lambda input, check_errors=False, out=None: -1,
        torch.is_complex: lambda input: -1,
        torch.is_conj: lambda input: -1,
        torch.is_neg: lambda input: -1,
        torch.is_distributed: lambda input: -1,
        torch.is_inference: lambda input: -1,
        torch.is_floating_point: lambda input: -1,
        torch.is_nonzero: lambda input: -1,
        torch.is_same_size: lambda input, other: -1,
        torch.is_signed: lambda input: -1,
        torch.isclose: lambda input, other, rtol=1e-05, atol=1e-08, equal_nan=False: -1,
        torch.isnan: lambda input: -1,
        torch.istft: (
            lambda input, n_fft, hop_length=None, win_length=None, window=None, center=True, normalized=False, onesided=None, length=None, return_complex=False: -1  # noqa: B950
        ),
        torch.kl_div: lambda input, target, size_average=None, reduce=None, reduction="mean", log_target=False: -1,
        torch.kron: lambda input, other: -1,
        torch.kthvalue: lambda input, k, dim=None, keepdim=False, out=None: -1,
        torch.linalg.ldl_factor_ex: lambda input, hermitian=False, check_errors=False, out=None: -1,
        torch.linalg.ldl_factor: lambda input, hermitian=False, out=None: -1,
        torch.linalg.ldl_solve: lambda LD, pivots, B, hermitian=False, out=None: -1,
        torch.layer_norm: lambda input, normalized_shape, weight=None, bias=None, esp=1e-05, cudnn_enabled=True: -1,
        torch.lcm: lambda input, other, out=None: -1,
        torch.ldexp: lambda input, other, out=None: -1,
        torch.le: lambda input, other, out=None: -1,
        torch.less_equal: lambda input, other, out=None: -1,
        torch.lerp: lambda input, end, weight, out=None: -1,
        torch.lgamma: lambda input, out=None: -1,
        torch.lobpcg: lambda input, k=None, B=None, X=None, n=None, iK=None, niter=None, tol=None, largest=None, method=None, tracker=None, ortho_iparams=None, ortho_fparams=None, ortho_bparams=None: -1,  # noqa: B950
        torch.log: lambda input, out=None: -1,
        torch.log_softmax: lambda input, dim, dtype=None: -1,
        torch.log10: lambda input, out=None: -1,
        torch.log1p: lambda input, out=None: -1,
        torch.log2: lambda input, out=None: -1,
        torch.logaddexp: lambda input, other, out=None: -1,
        torch.logaddexp2: lambda input, other, out=None: -1,
        torch.logdet: lambda input: -1,
        torch.xlogy: lambda x, y, out=None: -1,
        torch.logical_and: lambda input, other, out=None: -1,
        torch.logical_not: lambda input, out=None: -1,
        torch.logical_or: lambda input, other, out=None: -1,
        torch.logical_xor: lambda input, other, out=None: -1,
        torch.logit: lambda input, eps=None: -1,
        torch.logsumexp: lambda input, names, keepdim=False, out=None: -1,
        torch.lstm: lambda data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional: -1,
        torch.lstm_cell: lambda input, hx, w_ih, w_hh, b_ih=None, b_hh=None: -1,
        torch.lt: lambda input, other, out=None: -1,
        torch.less: lambda input, other, out=None: -1,
        torch.lu: lambda A, pivot=True, get_infos=False, out=None: -1,
        torch.lu_solve: lambda b, LU_data, LU_pivots, out=None: -1,
        torch.margin_ranking_loss: lambda input1, input2, target, margin=0, size_average=None, reduce=None, reduction="mean": -1,  # type: ignore[attr-defined]  # noqa: B950
        torch.masked_fill: lambda input, mask, value: -1,
        torch.masked_scatter: lambda input, mask, source: -1,
        torch.masked_select: lambda input, mask, out=None: -1,
        torch.matmul: lambda input, other, out=None: -1,
        torch.linalg.lu: lambda input, pivot=True, out=None: -1,
        torch.linalg.lu_factor: lambda input, pivot=True, out=None: -1,
        torch.linalg.lu_factor_ex: lambda input, pivot=True, check_errors=False, out=None: -1,
        torch.linalg.lu_solve: lambda LU, pivots, B, left=True, adjoint=False, out=None: -1,
        torch.linalg.matmul: lambda input, other, out=None: -1,  # alias for torch.matmul
        torch.matrix_power: lambda input, n: -1,
        torch.linalg.matrix_power: lambda input, n, out=None: -1,
        torch.linalg.matrix_rank: lambda input, tol=None, hermitian=False: -1,
        torch.linalg.multi_dot: lambda tensors, out=None: -1,
        torch.matrix_exp: lambda input: -1,
        torch.linalg.matrix_exp: lambda input: -1,
        torch.max: lambda input, out=None: -1,
        torch.maximum: lambda input, other, out=None: -1,
        torch.fmax: lambda input, other, out=None: -1,
        torch.max_pool1d: lambda input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False: -1,
        torch.max_pool2d: lambda input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False: -1,
        torch.max_pool3d: lambda input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False: -1,
        torch.max_pool1d_with_indices: (
            lambda input, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False: -1
        ),
        torch.mean: lambda input, dim=None: -1,
        torch.nanmean: lambda input, dim=None, keepdim=False, dtype=None, out=None: -1,
        torch.median: lambda input, dim=None: -1,
        torch.nanmedian: lambda input, dim=None: -1,
        torch.meshgrid: lambda *tensors, **kwargs: -1,
        torch.min: lambda input, out=None: -1,
        torch.minimum: lambda input, other, out=None: -1,
        torch.fmin: lambda input, other, out=None: -1,
        torch.miopen_batch_norm: (
            lambda input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon: -1
        ),
        torch.miopen_convolution: lambda input, weight, bias, padding, stride, dilation, groups, benchmark, deterministic: -1,  # noqa: B950
        torch.miopen_convolution_add_relu: lambda input, weight, z, alpha, bias, stride, padding, dilation, groups: -1,
        torch.miopen_convolution_relu: lambda input, weight, bias, stride, padding, dilation, groups: -1,
        torch.miopen_convolution_transpose: (
            lambda input, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic: -1
        ),
        torch.miopen_depthwise_convolution: (
            lambda input, weight, bias, padding, stride, dilation, groups, benchmark, deterministic: -1
        ),
        torch.miopen_rnn: (
            lambda input, weight, weight_stride0, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state: -1  # noqa: B950
        ),
        torch.mm: lambda input, mat2, out_dtype=None, out=None: -1,
        torch.mode: lambda input, dim=-1, keepdim=False, out=None: -1,
        torch.movedim: lambda input, source, destination: -1,
        torch.moveaxis: lambda input, source, destination: -1,
        torch.msort: lambda input, descending=False, out=None: -1,
        torch.mul: lambda input, other, out=None: -1,
        torch.multiply: lambda input, other, out=None: -1,
        torch.multinomial: lambda input, num_samples, replacement=False, out=None: -1,
        torch.mv: lambda input, vec, out=None: -1,
        torch.mvlgamma: lambda input, p: -1,
        torch.narrow: lambda input, dim, start, length: -1,
        torch.nan_to_num: lambda input, nan=0.0, posinf=None, neginf=None, out=None: -1,
        torch.native_batch_norm: lambda input, weight, bias, running_mean, running_var, training, momentum, eps: -1,
        torch._native_batch_norm_legit: lambda input, weight, bias, training, momentum, eps: -1,
        torch.native_dropout: lambda input, p, train: -1,
        torch.native_layer_norm: lambda input, normalized_shape, weight=None, bias=None, eps=1e-05: -1,
        torch._fused_rms_norm: lambda input, normalized_shape, weight=None, eps=1e-05: -1,
        torch.native_group_norm: lambda input, weight, bias, N, C, HxW, group, eps: -1,
        torch.native_norm: lambda input, p=2, dim=None, keepdim=False, dtype=None: -1,
        torch.native_channel_shuffle: lambda input, groups: -1,
        torch.ne: lambda input, other, out=None: -1,
        torch.not_equal: lambda input, other, out=None: -1,
        torch.neg: lambda input, out=None: -1,
        torch.negative: lambda input, out=None: -1,
        torch.nextafter: lambda input, other, out=None: -1,
        torch.nn.functional.adaptive_avg_pool2d: lambda input, output_size: -1,
        torch.nn.functional.adaptive_avg_pool3d: lambda input, output_size: -1,
        torch.nn.functional.adaptive_max_pool1d: lambda input, output_size, return_indices=False: -1,
        torch.nn.functional.adaptive_max_pool1d_with_indices: lambda input, output_size, return_indices=False: -1,
        torch.nn.functional.adaptive_max_pool2d: lambda input, output_size, return_indices=False: -1,
        torch.nn.functional.adaptive_max_pool2d_with_indices: lambda input, output_size, return_indices=False: -1,
        torch.nn.functional.adaptive_max_pool3d: lambda input, output_size, return_indices=False: -1,
        torch.nn.functional.adaptive_max_pool3d_with_indices: lambda input, output_size, return_indices=False: -1,
        torch.nn.functional.affine_grid: lambda theta, size, align_corners=None: -1,
        torch.nn.functional.alpha_dropout: lambda input, p=0.5, training=False, inplace=False: -1,
        torch.nn.functional.avg_pool2d: (
            lambda input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None: -1  # noqa: B950
        ),
        torch.nn.functional.avg_pool3d: (
            lambda input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None: -1  # noqa: B950
        ),
        torch.nn.functional.batch_norm: (
            lambda input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05: -1
        ),
        torch.nn.functional.bilinear: lambda input1, input2, weight, bias=None: -1,
        torch.nn.functional.binary_cross_entropy: (
            lambda input, target, weight=None, size_average=None, reduce=None, reduction="mean": -1
        ),
        torch.nn.functional.binary_cross_entropy_with_logits: (
            lambda input, target, weight=None, size_average=None, reduce=None, reduction="mean", pos_weight=None: -1
        ),
        torch.nn.functional.celu: lambda input, alpha=1.0, inplace=False: -1,
        torch.nn.functional.cosine_embedding_loss: (
            lambda input1, input2, target, margin=0, size_average=None, reduce=None, reduction="mean": -1
        ),
        torch.nn.functional.cross_entropy: (
            lambda input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean", label_smoothing=0.0: -1  # noqa: B950
        ),
        torch.nn.functional.ctc_loss: (
            lambda log_probs, targets, input_lengths, target_lengths, blank=0, reduction="mean", zero_infinity=False: -1
        ),
        torch.nn.functional.dropout: lambda input, p=0.5, training=True, inplace=False: -1,
        torch.nn.functional.dropout1d: lambda input, p=0.5, training=True, inplace=False: -1,
        torch.nn.functional.dropout2d: lambda input, p=0.5, training=True, inplace=False: -1,
        torch.nn.functional.dropout3d: lambda input, p=0.5, training=True, inplace=False: -1,
        torch.nn.functional.elu: lambda input, alpha=1.0, inplace=False: -1,
        torch.nn.functional.embedding: (
            lambda input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False: -1  # noqa: B950
        ),
        torch.nn.functional.embedding_bag: (
            lambda input, weight, offsets=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, mode="mean", sparse=False, per_sample_weights=None, include_last_offset=False, padding_idx=None: -1  # noqa: B950
        ),
        torch.nn.functional.feature_alpha_dropout: lambda input, p=0.5, training=False, inplace=False: -1,
        torch.nn.functional.fold: lambda input, output_size, kernel_size, dilation=1, padding=0, stride=1: -1,
        torch.nn.functional.fractional_max_pool2d: (
            lambda input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None: -1  # noqa: B950
        ),
        torch.nn.functional.fractional_max_pool2d_with_indices: (
            lambda input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None: -1  # noqa: B950
        ),
        torch.nn.functional.fractional_max_pool3d: (
            lambda input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None: -1  # noqa: B950
        ),
        torch.nn.functional.fractional_max_pool3d_with_indices: (
            lambda input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None: -1  # noqa: B950
        ),
        torch.nn.functional.gaussian_nll_loss: lambda input, target, var, full=False, eps=1e-06, reduction="mean": -1,
        torch.nn.functional.gelu: lambda input, approximate="none": -1,
        torch.nn.functional.glu: lambda input, dim=-1: -1,
        torch.nn.functional.grid_sample: lambda input, grid, mode="bilinear", padding_mode="zeros", align_corners=None: -1,  # noqa: B950
        torch.nn.functional.group_norm: lambda input, num_groups, weight=None, bias=None, eps=1e-05: -1,
        torch.nn.functional.gumbel_softmax: lambda logits, tau=1, hard=False, eps=1e-10, dim=-1: -1,
        torch.nn.functional.hardshrink: lambda input, lambd=0.5: -1,
        torch.nn.functional.hardtanh: lambda input, min_val=-1.0, max_val=1.0, inplace=False: -1,
        torch.nn.functional.hinge_embedding_loss: (
            lambda input, target, margin=1.0, size_average=None, reduce=None, reduction="mean": -1
        ),
        torch.nn.functional.instance_norm: (
            lambda input, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=True, momentum=0.1, eps=1e-05: -1  # noqa: B950
        ),
        torch.nn.functional.interpolate: (
            lambda input, size=None, scale_factor=None, mode="nearest", align_corners=None, recompute_scale_factor=None, antialias=False: -1  # noqa: B950
        ),
        torch.nn.functional.kl_div: lambda input, target, size_average=None, reduce=None, reduction="mean", log_target=False: -1,  # noqa: B950
        torch.nn.functional.l1_loss: lambda input, target, size_average=None, reduce=None, reduction="mean", weight=None: -1,
        torch.nn.functional.layer_norm: lambda input, normalized_shape, weight=None, bias=None, eps=1e-05: -1,
        torch.nn.functional.leaky_relu: lambda input, negative_slope=0.01, inplace=False: -1,
        torch.nn.functional.linear: lambda input, weight, bias=None: -1,
        torch.nn.functional.local_response_norm: lambda input, size, alpha=0.0001, beta=0.75, k=1.0: -1,
        torch.nn.functional.log_softmax: lambda input, dim=None, _stacklevel=3, dtype=None: -1,
        torch.nn.functional.logsigmoid: lambda input: -1,
        torch.nn.functional.lp_pool1d: lambda input, norm_type, kernel_size, stride=None, ceil_mode=False: -1,
        torch.nn.functional.lp_pool2d: lambda input, norm_type, kernel_size, stride=None, ceil_mode=False: -1,
        torch.nn.functional.lp_pool3d: lambda input, norm_type, kernel_size, stride=None, ceil_mode=False: -1,
        torch.nn.functional.margin_ranking_loss: (
            lambda input1, input2, target, margin=0, size_average=None, reduce=None, reduction="mean": -1
        ),
        torch.nn.functional.max_pool1d: (
            lambda input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False: -1
        ),
        torch.nn.functional.max_pool1d_with_indices: (
            lambda input, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False: -1
        ),
        torch.nn.functional.max_pool2d: (
            lambda input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False: -1
        ),
        torch.nn.functional.max_pool2d_with_indices: (
            lambda input, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False: -1
        ),
        torch.nn.functional.max_pool3d: (
            lambda input, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False: -1
        ),
        torch.nn.functional.max_pool3d_with_indices: (
            lambda input, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False: -1
        ),
        torch.nn.functional.max_unpool1d: lambda input, indices, kernel_size, stride=None, padding=0, output_size=None: -1,  # noqa: B950
        torch.nn.functional.max_unpool2d: lambda input, indices, kernel_size, stride=None, padding=0, output_size=None: -1,  # noqa: B950
        torch.nn.functional.max_unpool3d: lambda input, indices, k
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/torch`):

- [`types.py_kw.md_docs.md`](./types.py_kw.md_docs.md)
- [`storage.py_docs.md_docs.md`](./storage.py_docs.md_docs.md)
- [`serialization.py_kw.md_docs.md`](./serialization.py_kw.md_docs.md)
- [`serialization.py_docs.md_docs.md`](./serialization.py_docs.md_docs.md)
- [`library.py_kw.md_docs.md`](./library.py_kw.md_docs.md)
- [`script.h_kw.md_docs.md`](./script.h_kw.md_docs.md)
- [`_sources.py_kw.md_docs.md`](./_sources.py_kw.md_docs.md)
- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)
- [`_torch_docs.py_docs.md_docs.md`](./_torch_docs.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `overrides.py_docs.md_docs.md`
- **Keyword Index**: `overrides.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
