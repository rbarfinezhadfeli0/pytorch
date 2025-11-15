# Documentation: common_methods_invocations.py

## File Metadata
- **Path**: `torch/testing/_internal/common_methods_invocations.py`
- **Size**: 1217342 bytes
- **Lines**: 25187
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# mypy: ignore-errors

from functools import wraps, partial
from itertools import product, chain, islice
import itertools
import functools
import copy
import operator
import random
import unittest
import math
import enum

import torch
import numpy as np
import numpy.typing as npt
from torch import inf, nan

from typing import Any, Union
from collections.abc import Sequence
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import (
    _dispatch_dtypes, floating_types, floating_types_and, complex_types, floating_and_complex_types,
    floating_and_complex_types_and, all_types_and_complex_and, all_types_and, all_types_and_complex, integral_types_and,
    empty_types, complex_types_and, integral_types, custom_types, all_types_complex_float8_and, float8_types,
)
from torch.testing._internal.common_device_type import \
    (onlyCPU, onlyCUDA, onlyNativeDeviceTypes, disablecuDNN, skipCUDAIfNoMagma, skipCUDAIfNoMagmaAndNoCusolver,
     skipCUDAIfNoCusolver, skipCPUIfNoLapack, skipCPUIfNoFFT, skipCUDAIf, precisionOverride,
     skipCPUIfNoMklSparse,
     toleranceOverride, tol, skipXPU)
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FLASH_ATTENTION, PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
    SM53OrLater, SM80OrLater, SM89OrLater, with_tf32_off, TEST_CUDNN, _get_torch_cuda_version,
    _get_torch_rocm_version,
)
from torch.testing._internal.common_quantized import (
    _bfloat16_to_float4_e2m1fn_x2,
)
from torch.testing._internal.common_utils import (
    make_fullrank_matrices_with_distinct_singular_values,
    TEST_WITH_ROCM, IS_FBCODE, IS_WINDOWS, IS_MACOS, IS_S390X, TEST_SCIPY,
    torch_to_numpy_dtype_dict, numpy_to_torch_dtype, TEST_WITH_ASAN,
    GRADCHECK_NONDET_TOL, slowTest, TEST_WITH_SLOW,
    TEST_WITH_TORCHINDUCTOR, MACOS_VERSION,
)
from torch.testing._utils import wrapper_set_seed

import torch._refs as refs  # noqa: F401
import torch._refs.nn.functional
import torch._refs.special
import torch._refs.linalg
import torch._prims as prims  # noqa: F401
from torch.utils import _pytree as pytree


from torch._vendor.packaging import version

from torch.testing._internal.opinfo.core import (  # noqa: F401
    L,
    M,
    S,
    XS,
    _NOTHING,
    _getattr_qual,
    DecorateInfo,
    SampleInput,
    ErrorInput,
    AliasInfo,
    NumericsFilter,
    OpInfo,
    _generate_reduction_inputs,
    _generate_reduction_kwargs,
    sample_inputs_reduction,
    ReductionOpInfo,
    reference_inputs_elementwise_binary,
    make_error_inputs_elementwise_binary,
    generate_elementwise_binary_tensors,
    generate_elementwise_binary_arbitrarily_strided_tensors,
    generate_elementwise_binary_small_value_tensors,
    generate_elementwise_binary_large_value_tensors,
    generate_elementwise_binary_extremal_value_tensors,
    generate_elementwise_binary_broadcasting_tensors,
    generate_elementwise_binary_with_scalar_samples,
    generate_elementwise_binary_with_scalar_and_type_promotion_samples,
    generate_elementwise_binary_noncontiguous_tensors,
    sample_inputs_elementwise_binary,
    BinaryUfuncInfo,
    sample_inputs_elementwise_unary,
    generate_elementwise_unary_tensors,
    generate_elementwise_unary_small_value_tensors,
    generate_elementwise_unary_large_value_tensors,
    generate_elementwise_unary_extremal_value_tensors,
    reference_inputs_elementwise_unary,
    UnaryUfuncInfo,
    sample_inputs_spectral_ops,
    SpectralFuncType,
    SpectralFuncInfo,
    ShapeFuncInfo,
    sample_inputs_foreach,
    ForeachFuncInfo,
    gradcheck_wrapper_hermitian_input,
    gradcheck_wrapper_ctc_loss,
    gradcheck_wrapper_triangular_input,
    gradcheck_wrapper_triangular_input_real_positive_diagonal,
    gradcheck_wrapper_masked_operation,
    gradcheck_wrapper_masked_pointwise_operation,
    clone_sample,
)
from torch.testing._internal.opinfo.refs import (  # NOQA: F401
    _find_referenced_opinfo,
    _inherit_constructor_args,
    PythonRefInfo,
    ReductionPythonRefInfo,
    ElementwiseUnaryPythonRefInfo,
    ElementwiseBinaryPythonRefInfo,
)
from torch.testing._internal.opinfo.utils import (
    np_unary_ufunc_integer_promotion_wrapper,
    reference_reduction_numpy,
    prod_numpy
)
from torch.testing._internal import opinfo
from torch.testing._internal.opinfo.definitions.linalg import (
    sample_inputs_linalg_cholesky,
    sample_inputs_linalg_cholesky_inverse,
    sample_inputs_cross,
    sample_inputs_linalg_qr_geqrf,
    sample_inputs_linalg_invertible,
    sample_inputs_lu_solve,
    sample_inputs_legacy_solve,
    sample_inputs_svd,
    sample_inputs_linalg_det_logdet_slogdet,
    sample_inputs_linalg_lu,
    sample_inputs_diagonal_diag_embed,
    error_inputs_diagonal_diag_embed,
)
from torch.testing._internal.opinfo.definitions.special import (
    sample_inputs_i0_i1,
    sample_inputs_polygamma,
    reference_polygamma,
)
from torch.testing._internal.opinfo.definitions._masked import (
    sample_inputs_softmax_variant,
)
from torch.testing._internal.opinfo.definitions.sparse import (
    error_inputs_sparse_like_fns,
    sample_inputs_sparse_like_fns,
    error_inputs_sparse_mul,
    sample_inputs_sparse_mul,
    error_inputs_sparse_reduction_sum,
    sample_inputs_sparse_reduction_sum
)

if TEST_SCIPY:
    from scipy import stats
    import scipy.spatial
    import scipy.special


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


# test if a tensor is close to an integer
def close_to_int(x, eps=0.1):
    if x.is_complex():
        y = torch.abs(torch.view_as_complex(torch.frac(torch.view_as_real(x))))
    else:
        y = torch.abs(torch.frac(x))
    return (y < eps) | (y > (1 - eps))


def sample_inputs_slice(op_info, device, dtype, requires_grad, **kwargs):

    make_input = partial(make_tensor, device=device, dtype=dtype,
                         low=None, high=None, requires_grad=requires_grad)

    yield SampleInput(make_input(3), 0)

    yield SampleInput(make_input(20, 30, 40), dim=1, start=1, end=-2)

    yield SampleInput(make_input(20, 30, 40), dim=1, start=1, end=-2, step=3)

    yield SampleInput(make_input(20, 30, 40), dim=0, start=-10, end=-2, step=2)


def sample_inputs_tensor_split(op_info, device, dtype, requires_grad, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype,
                         low=None, high=None, requires_grad=requires_grad)

    args_cases = (
        # Cases with tensor indices.
        (torch.tensor([1, 2, 3]),),
        (torch.tensor(1),),
        (torch.tensor([1, 2, 3]), 1),
        (torch.tensor([1, 4, 2, 5, 3, 6])[::2], 1),
        # Cases with list of indices.
        ((2, 4),),
        ((2, 4), 1),
        ((2, 4), -1),
        # Cases with integer section.
        (3,),
        (3, 1),
        (3, -1),
    )

    for args in args_cases:
        yield SampleInput(make_input((S, S, S)), args=args)


def sample_inputs_hsplit(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device,
                       low=None, high=None, requires_grad=requires_grad)
    yield SampleInput(make_arg(6), 2)
    yield SampleInput(make_arg(S, S, S), [1, 2, 3])

def sample_inputs_vsplit(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device,
                       low=None, high=None, requires_grad=requires_grad)
    yield SampleInput(make_arg(6, S), 2)
    yield SampleInput(make_arg(S, S, S), [1, 2, 3])

def sample_inputs_dsplit(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device,
                       low=None, high=None, requires_grad=requires_grad)
    yield SampleInput(make_arg(S, S, S), [1, 2, 3])
    yield SampleInput(make_arg(S, S, 6), 2)

def error_inputs_hsplit(op_info, device, **kwargs):
    make_arg = partial(make_tensor, dtype=torch.float32, device=device)
    err_msg1 = ("torch.hsplit requires a tensor with at least 1 dimension, "
                "but got a tensor with 0 dimensions!")
    yield ErrorInput(SampleInput(make_arg(()), 0), error_regex=err_msg1)

    err_msg2 = (f"torch.hsplit attempted to split along dimension 1, "
                f"but the size of the dimension {S} "
                f"is not divisible by the split_size 0!")
    yield ErrorInput(SampleInput(make_arg((S, S, S)), 0), error_regex=err_msg2)

    # Incorrect type for indices_or_section argument
    err_msg3 = ("received an invalid combination of arguments.")
    yield ErrorInput(
        SampleInput(make_arg((S, S, S)), "abc"),
        error_type=TypeError, error_regex=err_msg3)

def error_inputs_vsplit(op_info, device, **kwargs):
    make_arg = partial(make_tensor, dtype=torch.float32, device=device)
    err_msg1 = ("torch.vsplit requires a tensor with at least 2 dimension, "
                "but got a tensor with 1 dimensions!")
    yield ErrorInput(SampleInput(make_arg(S), 0), error_regex=err_msg1)

    err_msg2 = (f"torch.vsplit attempted to split along dimension 0, "
                f"but the size of the dimension {S} "
                f"is not divisible by the split_size 0!")
    yield ErrorInput(SampleInput(make_arg(S, S, S), 0),
                     error_regex=err_msg2)

    # Incorrect type for indices_or_section argument
    err_msg3 = ("received an invalid combination of arguments.")
    yield ErrorInput(SampleInput(make_arg(S, S, S), "abc"),
                     error_type=TypeError, error_regex=err_msg3)

def error_inputs_dsplit(op_info, device, **kwargs):
    make_arg = partial(make_tensor, dtype=torch.float32, device=device)
    err_msg1 = ("torch.dsplit requires a tensor with at least 3 dimension, "
                "but got a tensor with 1 dimensions!")
    yield ErrorInput(SampleInput(make_arg(S), 0), error_regex=err_msg1)

    err_msg2 = (f"torch.dsplit attempted to split along dimension 2, "
                f"but the size of the dimension {S} "
                f"is not divisible by the split_size 0!")
    yield ErrorInput(SampleInput(make_arg(S, S, S), 0), error_regex=err_msg2)


def sample_inputs_as_strided(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # input shape, output shape, output stride, output storage offset
    test_cases = (
        ((1,), (1,), (1,), 0),
        ((3, 3), (2, 2), (1, 2), 0),
        ((3, 3), (2, 2), (1, 2), 1),
        ((16,), (2, 2, 2, 2), (1, 1, 1, 1), 0),
        ((16,), (2, 1, 1, 2), (1, 7, 7, 1), 0),
    )

    for input_shape, output_shape, stride, storage_offset in test_cases:
        input_t = make_arg(input_shape)
        kwargs = dict(storage_offset=storage_offset)
        yield SampleInput(input_t, args=(output_shape, stride), kwargs=kwargs)

def sample_inputs_as_strided_partial_views(op_info, device, dtype, requires_grad, **kwargs):
    def make_arg():
        base = make_tensor((20,), device=device, dtype=dtype)
        return base[5:15].requires_grad_(requires_grad)

    # as_strided on offset, partial views
    yield SampleInput(make_arg(), (2, 2), (1, 2))
    yield SampleInput(make_arg(), (2, 2), (1, 2), storage_offset=0)
    yield SampleInput(make_arg(), (2, 2), (1, 2), storage_offset=10)

def sample_inputs_as_strided_scatter(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # input shape, output shape, output stride, output storage offset
    test_cases = [
        ((1,), (), (), 0),
        ((1,), (1,), (1,), 0),
        ((3, 3), (2, 2), (1, 2), 0),
        ((3, 3), (2, 2), (1, 2), 1),
        ((3, 3), (2, 2), (2, 1), 0),
        # Scatter to larger dimensions
        ((16,), (2, 2, 2, 2), (8, 4, 2, 1), 0),
        # Scatter to larger dimensions with strides inverted
        ((16,), (2, 1, 1, 2), (1, 2, 4, 8), 0),
    ]

    for input_shape, output_shape, stride, storage_offset in test_cases:
        input_t = make_arg(input_shape)
        input_src = make_arg(output_shape)
        yield SampleInput(input_t, input_src, output_shape, stride, storage_offset=storage_offset)


def error_inputs_as_strided_scatter(op_info, device, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=torch.float32, requires_grad=False)

    # Create a small tensor and try to scatter it out of bounds
    input_t = make_arg([4, 4])
    input_src = make_arg([2, 2])
    yield ErrorInput(
        SampleInput(input_t, input_src, [2, 2], [200, 200], storage_offset=0),
        error_regex="itemsize 4 requiring a storage size of 1604 are out of bounds for storage of size 64"
    )


def sample_inputs_combinations(op_info, device, dtype, requires_grad, **kwargs):
    inputs = (
        (0,),
        (0, 1),
        (0, 1, 2, 3),
    )

    rvals = [1, 2, 4]

    products = product(inputs, rvals, [False, True])

    for input_data, r, with_replacement in products:
        input_t = torch.tensor(input_data, device=device, dtype=dtype, requires_grad=requires_grad)
        yield SampleInput(input_t, r=r, with_replacement=with_replacement)

def sample_inputs_cartesian_prod(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(torch.tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # constructs 1-D tensors with varying number of elements
    a = make_arg((0,))
    b = make_arg((0, 1))
    c = make_arg((0, 1, 2, 3))

    # sample with only 1 tensor
    yield SampleInput(a)

    # sample with 2 tensors
    yield SampleInput(a, b)

    # sample with 3 tensors
    yield SampleInput(a, b, c)

def sample_inputs_cosine_similarity(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Ordered as input_shape, dict of dim and eps
    cases: tuple[tuple, dict] = (  # type: ignore[assignment]
        ((S, S), {'dim': 1}),
        ((S, 2), {'dim': -1}),
        ((S,), {'dim': 0, 'eps': 0.5}),
        ((), {'dim': 0}),
        ((S, S, M), {'dim': 2}),
        ((S, S), {})
    )

    for input_shape, kwargs in cases:
        yield SampleInput(make_arg(input_shape), args=(make_arg(input_shape),), kwargs=kwargs)
    # Test for Broadcasting
    yield SampleInput(make_arg((1, 2, 3)), args=(make_arg((2, 1, 3)),), kwargs={'dim': -1})
    yield SampleInput(make_arg((1, 2, 3)), args=(make_arg((2, 1, 3)),), kwargs={'dim': -2})
    yield SampleInput(make_arg((2, 3)), args=(make_arg((2, 1, 3)),), kwargs={'dim': -1})


def sample_inputs_item(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=False)

    cases = (
        (),
        (()),
        (1),
        ((1,)),
    )

    for shape in cases:
        yield SampleInput(make_arg(shape))

def error_inputs_item(op, device, **kwargs):
    make_arg = partial(make_tensor, dtype=torch.float32, device=device, requires_grad=False)

    cases = (
        (M),
        ((S,)),
        (S, S),
        (S, M, L),
    )

    for shape in cases:
        yield ErrorInput(
            SampleInput(make_arg(shape)), error_type=RuntimeError,
            error_regex="elements cannot be converted to Scalar")


def sample_inputs_batch_norm(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_arg_without_requires_grad = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    # Ordered as: input shape, kwargs for training, momentum, eps
    cases: tuple[tuple[int], dict] = (  # type: ignore[assignment]
        ((S, S, S), {'training': True, 'momentum': 0.5, 'eps': 0.6}),
        ((3, 2, 4), {'training': False, 'momentum': -1.2}),
        ((3, 1), {'training': True, 'momentum': 0.0}),
        ((0,), {'training': True}),
        ((0,), {'training': False}),
        ((3, 2, 3, 4), {'training': True, 'momentum': -1.0, 'eps': 0.5}),
        ((3, 2, 3, 4), {'training': False, 'momentum': -1.0, 'eps': 0.5}),
        ((2, 1), {}),
    )

    for input_shape, kwargs in cases:
        # args: running mean, running var, weight and bias should necessarily be of shape: (channels,)
        channels = input_shape[1] if len(input_shape) > 1 else 0
        weight = make_arg(channels) if channels > 0 else None
        bias = make_arg(channels) if channels > 0 else None
        running_mean = make_arg_without_requires_grad(channels, low=0)
        running_var = make_arg_without_requires_grad(channels, low=0)

        yield SampleInput(
            make_arg(input_shape),
            args=(
                running_mean,
                running_var,
                weight,
                bias
            ),
            kwargs=kwargs
        )

    # Checking for permutations of weights and biases as `None`
    is_training = [True, False, False]

    for training in is_training:
        yield SampleInput(
            make_arg(input_shape),
            args=(
                running_mean,
                running_var,
                make_arg(channels),
                make_arg(channels)
            ),
            kwargs={'training': training}
        )

    # Test case for no optional kwargs
    # running_mean and running_var are required in evaluation mode (training: False) but not in training mode
    yield SampleInput(make_arg((1, 2, 3)), args=(None, None, None, None), kwargs={'training': True})

def sample_inputs_softmax_backward_data(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )
    cases = [
        ((S,), 0),
        ((S, S), 0),
        ((S, M, S), -1),
    ]
    input_dtypes = [dtype]
    if dtype == torch.float and device == 'cuda':
        input_dtypes += [torch.float16]

    for (shape, dim), input_dtype in product(cases, input_dtypes):
        input = make_arg(shape)
        output = torch.nn.functional.softmax(input, dim=dim, dtype=input_dtype)
        yield SampleInput(make_arg(shape), output, dim, input_dtype)

def sample_inputs_native_batch_norm(op_info, device, dtype, requires_grad, **kwargs):
    samples = sample_inputs_batch_norm(op_info, device, dtype, requires_grad, **kwargs)
    for sample in samples:
        # torch.native_batch_norm does not support 0 numel tensors
        # IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
        if sample.input.numel() == 0:
            continue
        args = sample.args
        training = sample.kwargs.get('training', True)
        momentum = sample.kwargs.get('momentum', 0.5)
        eps = sample.kwargs.get('eps', 1e-5)
        yield SampleInput(sample.input, args=(args[2], args[3], args[0], args[1], training, momentum, eps))


def sample_inputs__native_batch_norm_legit(op_info, device, dtype, requires_grad, **kwargs):
    samples = sample_inputs_batch_norm(op_info, device, dtype, requires_grad, **kwargs)
    for sample in samples:
        # torch.native_batch_norm does not support 0 numel tensors
        # IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
        if sample.input.numel() == 0:
            continue
        args = sample.args
        training = sample.kwargs.get('training', True)
        momentum = sample.kwargs.get('momentum', 0.5)
        eps = sample.kwargs.get('eps', 1e-5)
        if args[0] is not None and args[1] is not None:
            yield SampleInput(sample.input, args=(args[2], args[3], args[0], args[1], training, momentum, eps))
        else:
            yield SampleInput(sample.input, args=(args[2], args[3], training, momentum, eps))

def sample_inputs__batch_norm_with_update(op_info, device, dtype, requires_grad, **kwargs):
    samples = sample_inputs_batch_norm(op_info, device, dtype, requires_grad, **kwargs)
    for sample in samples:
        # torch.native_batch_norm does not support 0 numel tensors
        # IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
        if sample.input.numel() == 0:
            continue
        args = sample.args
        momentum = sample.kwargs.get('momentum', 0.5)
        eps = sample.kwargs.get('eps', 1e-5)
        if any(args[i] is None for i in range(4)):
            continue
        yield SampleInput(sample.input, args=(args[2], args[3], args[0], args[1], momentum, eps))

def sample_inputs_nn_activation_relu(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases = (
        (()),
        ((S, )),
        ((S, S)),
        ((S, M, S))
    )

    for shape in cases:
        yield SampleInput(make_arg(shape))

def sample_inputs_prelu(op_info, device, dtype, requires_grad, **kwargs):
    op_kwargs = op_info.sample_kwargs(device, dtype, None)[0]
    yield from sample_inputs_elementwise_unary(op_info, device, dtype, requires_grad,
                                               op_kwargs=op_kwargs)

    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases = (
        (()),
        ((S, )),
        ((S, S)),
        ((S, M, S))
    )

    for shape in cases:
        for weight in [-1., 0., 0.8, 1.]:
            weight_tensor = torch.tensor(weight, device=device, dtype=dtype, requires_grad=requires_grad)
            yield SampleInput(make_arg(shape), args=(weight_tensor,))

        channel_size = shape[1] if len(shape) >= 2 else 1
        yield SampleInput(make_arg(shape), args=(make_arg((channel_size,)),))

    weight_tensor = torch.tensor(1., device=device, dtype=dtype, requires_grad=requires_grad)

    yield SampleInput(make_arg((S, S)), kwargs=dict(weight=weight_tensor,))
    yield SampleInput(make_arg((S, S)), kwargs=dict(weight=make_arg((S,)),))

def reference_inputs_prelu(op, device, dtype, requires_grad, **kwargs):
    yield from sample_inputs_prelu(op, device, dtype, requires_grad, **kwargs)
    yield from reference_inputs_elementwise_unary(op, device, dtype, requires_grad, **kwargs)

def sample_kwargs_prelu_scalar_weight(device, dtype, input):
    weight = torch.rand((), device=device, dtype=dtype)
    # NumPy does not support bfloat16, so we default to float32 (only for NumPy) in that case
    if dtype == torch.bfloat16:
        weight_cpu = weight.to(dtype=torch.float32, device="cpu")
    else:
        weight_cpu = weight.cpu()
    np_weight = weight_cpu.numpy()
    return ({'weight': weight}, {'weight': np_weight})

def error_inputs_prelu(op, device):
    # Weight has numel != 1, but self.ndim is zero-dim tensor
    inp = make_tensor((), device=device, dtype=torch.float32)
    weight = make_tensor((2,), device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(inp, kwargs={'weight': weight}),
                     error_regex="Not allow zero-dim input tensor.")

    # Weight has numel != 1, but numel does not match channel size
    inp = make_tensor((2, 8, 3), device=device, dtype=torch.float32)
    weight = make_tensor((9,), device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(inp, kwargs={'weight': weight}),
                     error_regex="Mismatch of parameter numbers and input channel size.")

    # Weight is neither a scalar nor 1-D tensor
    inp = make_tensor((2, 8, 3), device=device, dtype=torch.float32)
    weight = make_tensor((2, 4), device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(inp, kwargs={'weight': weight}),
                     error_regex="prelu: Expected `weight` to be a scalar or 1D tensor, but got: ndim = 2")

    # src and index tensors must have the same # of dimensions
def sample_inputs_norm(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # ord = inf is tested in inputs_norm_inf as it fails on some tests
    cases = [
        ((S, S), (2,), '2'),
        ((S, S), (0,), '0'),
        ((S, S), (0.5,), '0_5'),
        ((S, S), (1,), '1'),
        ((S, S), (3,), '3'),
        ((S, S), (-1,), 'neg_1'),
        ((S, S), (-2,), 'neg_2'),
        ((S, S), (-0.5,), 'neg_0_5'),
        ((S, S), (-1.5,), 'neg_1_5'),
    ]

    cases_nonzero_input = (
        ((S, S, S), (1.5,), '1_5_default'),
        ((S, S, S), (1.5, 1), '1_5_dim'),
        ((S, S, S), (1.5, -1), '1_5_neg_dim'),
        ((S, S, S), (1.5, 1, True), 'keepdim_1_5_dim'),
        ((S, S, S), (1.5, -1, True), 'keepdim_1_5_neg_dim'),
    )

    cases_posdim = (
        ((S, S), (-2, 1,), 'neg_2_dim'),
        ((S, S), (-1, 1,), 'neg_1_dim'),
        ((S, S), (0, 1,), '0_dim'),
        ((S, S), (1, 1,), '1_dim'),
        ((S, S), (2, 1,), '2_dim'),
        ((S, S), (3, 1,), '3_dim'),
        ((S, S, S), (2, 1), '2_dim'),
        ((S, S, S), (3, 1), '3_dim'),
        ((S, S, S), (2, 1, True), 'keepdim_2_dim'),
        ((S, S, S), (3, 1, True), 'keepdim_3_dim'),
        ((), (2, 0), '2_dim_scalar'),
        ((), (3, 0), '3_dim_scalar'),
        ((), (2, 0, True), 'keepdim_2_dim_scalar'),
        ((), (3, 0, True), 'keepdim_3_dim_scalar'),
    )

    cases_negdim = ((shape, args[:1] + (-args[1],) + args[2:], name.replace("_dim", "_neg_dim"))
                    for shape, args, name in cases_posdim)

    for shape, args, name in itertools.chain(cases, cases_posdim, cases_negdim):
        yield SampleInput(make_arg(shape), args=args, name=name)

    for shape, args, name in cases_nonzero_input:
        yield SampleInput(make_arg(shape, exclude_zero=True), args=args, name=name)


def sample_inputs_norm_fro(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases = (
        ((S, S), (), 'default'),
        ((S, S), ('fro',), 'fro_default'),
        ((S, S), ('fro', [0, 1],), 'fro'),
    )

    for shape, args, name in cases:
        yield SampleInput(make_arg(shape), args=args, name=name)


def sample_inputs_norm_nuc(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases = (
        ((S, S), ('nuc',), 'nuc'),
        ((S, S, S), ('nuc', [1, 2]), 'nuc_batched'),
    )

    for shape, args, name in cases:
        yield SampleInput(make_arg(shape), args=args, name=name)


def sample_inputs_norm_inf(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases = (
        ((S, S), (-inf,), '-inf'),
        ((S, S), (inf,), 'inf'),
        ((S, S), (inf, 1,), 'inf_2_dim'),
        ((S, S), (inf, -1,), 'inf_2_neg_dim'),
    )

    for shape, args, name in cases:
        yield SampleInput(make_arg(shape), args=args, name=name)


def sample_inputs_equal(op, device, dtype, requires_grad, **kwargs):
    make_arg = partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    shapes = (
        ((), ()),
        ((S,), ()),
        ((), (S,)),
        ((S, 1), (S,)),
        ((M, S), ()),
        ((S, S), (S, S))
    )

    for shape_lhs, shape_rhs in shapes:
        lhs = make_arg(shape_lhs)
        rhs = make_arg(shape_rhs)
        broadcasts_input = shape_lhs != torch.broadcast_shapes(shape_lhs, shape_rhs)

        yield SampleInput(lhs, args=(rhs,), broadcasts_input=broadcasts_input)
        if shape_lhs == shape_rhs:
            yield SampleInput(lhs, args=(lhs.clone().detach_(),))


def sample_inputs_jiterator(op, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    shapes = (
        ((), ()),
        ((S,), ()),
        ((S, 1), (S,)),
        ((M, S), ()),
        ((S, M, S), (M, S)),
        ((S, M, S), (S, M, S)),
        ((M, 1, S), (M, S)),
        ((M, 1, S), (1, M, S)),
        ((0, 1, 3), (0, 10, 3))
    )

    num_inputs = kwargs.get('num_inputs')
    sample_kwargs = kwargs.get('sample_kwargs', {})

    for shape_lhs, shape_rhs in shapes:
        lhs = make_arg(shape_lhs)
        args = [make_arg(shape_rhs) for _ in range(num_inputs - 1)]
        broadcasts_input = (shape_lhs != torch.broadcast_shapes(shape_lhs, shape_rhs))

        yield SampleInput(lhs, args=tuple(args), kwargs=sample_kwargs, broadcasts_input=broadcasts_input)

def sample_inputs_broadcast_shapes(op, device, dtype, requires_grad, **kwargs):
    shapes = (
        ((), ()),
        ((S,), ()),
        ((S, 1), (S,)),
        ((S, 1), S),
        ((M, S), ()),
        ((S, M, S), (M, S)),
        ((S, M, S), (S, M, S)),
        ((M, 1, S), (M, S)),
        ((M, 1, S), (1, M, S)),
        ((0, 1, 3), (0, 10, 3))
    )

    for shape in shapes:
        inp, *arg0 = shape
        yield SampleInput(inp, args=tuple(arg0))

def sample_inputs_add_sub(op, device, dtype, requires_grad, **kwargs):
    yield from sample_inputs_elementwise_binary(op, device, dtype, requires_grad, **kwargs)

    # Adds alpha kwarg cases
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    lhs = make_arg((S, S), **op.lhs_make_tensor_kwargs)
    rhs = make_arg((S, S), **op.rhs_make_tensor_kwargs)
    if dtype is not torch.bool:
        yield SampleInput(lhs, args=(rhs,), kwargs={'alpha': 2})
    else:
        yield SampleInput(lhs, args=(rhs,), kwargs={'alpha': True})
    neg_alpha = -3.125 if (dtype.is_floating_point or dtype.is_complex) else -3
    lhs = make_arg((S, S), **op.lhs_make_tensor_kwargs)
    rhs = make_arg((S, S), **op.rhs_make_tensor_kwargs)
    if dtype is not torch.bool:
        yield SampleInput(lhs, args=(rhs,), kwargs={'alpha': neg_alpha})
    else:
        yield SampleInput(lhs, args=(rhs,), kwargs={'alpha': False})

def error_inputs_arange(op, device, **kwargs):
    yield ErrorInput(SampleInput(0, args=(3, 0)), error_type=RuntimeError, error_regex='step must be nonzero')
    yield ErrorInput(SampleInput(0, args=(-3, 2)), error_type=RuntimeError,
                     error_regex='upper bound and lower bound inconsistent with step sign')
    yield ErrorInput(SampleInput(0, args=(3, -2)), error_type=RuntimeError,
                     error_regex='upper bound and lower bound inconsistent with step sign')
    yield ErrorInput(SampleInput(1549556900, args=(1549556828, 1989724)), error_type=RuntimeError,
                     error_regex='upper bound and lower bound inconsistent with step sign')
    yield ErrorInput(SampleInput(0, args=(float('inf'), 2)), error_type=RuntimeError, error_regex='unsupported range')
    yield ErrorInput(SampleInput(float('-inf'), args=(1, 2)), error_type=RuntimeError, error_regex='unsupported range')

def sample_inputs_arange(op, device, dtype, requires_grad, **kwargs):
    int_samples = (
        # positive direction
        (-1, 2, 2),
        # negative direction
        (2, -3, -1),
        # start == end
        (1, 1, 1),
        (1, 1, -1),
        # divides evenly
        (0, -8, -4),
        (1, 5, 2),
        # bool
        (False, True, True),
        # default step
        (0, 1, None),
        # default start
        (None, 3, None),
    )

    def to_float(start, end, step):
        start = start + 0.1 if start is not None else None
        end = end + 0.1
        step = float(step) if step is not None else None
        return start, end, step

    float_samples = (
        # includes endpoint
        (0., -8. - 1e-6, -4.),
        (1., 5. + 1e-6, 2.),
        (0., -8., -4.),
        (1., 5., 2.),
        *(to_float(start, end, step) for (start, end, step) in int_samples),
    )

    large_samples = (
        (0, 10000, None),
    )

    samples = int_samples + float_samples
    if dtype not in (torch.int8, torch.uint8):
        samples += large_samples

    for start, end, step in samples:
        if start is None:
            assert step is None
            # Pass end as positional arg
            yield SampleInput(end, kwargs={"dtype": dtype, "device": device})
            # (Similar to) calling torch.arange(end=3)
            yield SampleInput(0, kwargs={"end": end, "dtype": dtype, "device": device})
        elif step is None:
            yield SampleInput(start, args=(end,), kwargs={"dtype": dtype, "device": device})
        else:
            yield SampleInput(start, args=(end, step), kwargs={"dtype": dtype, "device": device})

    yield SampleInput(2)
    yield SampleInput(1, args=(3, 1))

def sample_inputs_randn(op, device, dtype, requires_grad, **kwargs):
    shapes = (
        (M,),
        (S, S)
    )

    for shape in shapes:
        yield SampleInput(input=shape, kwargs=dict(dtype=dtype, device=device, requires_grad=requires_grad))

def sample_inputs_normal(op, device, dtype, requires_grad, **kwargs):

    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=False)
    samples = (
        ((S, S), 0, 5),
        ((S, S, S), -2, 0.5),
    )
    for shape, mean, std in samples:
        yield SampleInput(make_arg(shape), args=(mean, std))

def error_inputs_normal(op, device, **kwargs):
    t = torch.zeros([10], device=device)
    invalid_std = -1
    yield ErrorInput(
        SampleInput(t, args=(0, invalid_std)),
        error_type=RuntimeError,
        error_regex=fr"normal expects std >= 0.0, but found std {invalid_std}",
    )

def sample_inputs_cauchy(op, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=False)
    samples = (
        ((M,), 0, 0.5),
        ((S, S), 0, 1),
        ((S, S, S), -2, 1),
    )
    for shape, median, gamma in samples:
        yield SampleInput(make_arg(shape), args=(median, gamma))


def error_inputs_cauchy(op, device, **kwargs):
    t = torch.zeros([10], device=device)
    invalid_scale = 0
    yield ErrorInput(
        SampleInput(t, args=(0, invalid_scale,)),
        error_type=RuntimeError,
        error_regex=fr"cauchy_ expects sigma > 0.0, but found sigma={invalid_scale}",
    )


def sample_inputs_exponential(op, device, dtype, requires_grad, **kwargs):

    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=False)
    samples = (
        ((M,), 0.5),
        ((S, S), 1),
        ((S, S, S), 1.5),
    )
    for shape, rate in samples:
        yield SampleInput(make_arg(shape), args=(rate,))


def error_inputs_exponential(op, device, **kwargs):
    t = torch.zeros([10], device=device)
    invalid_rate = 0
    yield ErrorInput(
        SampleInput(t, args=(invalid_rate,)),
        error_type=RuntimeError,
        error_regex=fr"exponential_ expects lambda > 0.0, but found lambda={invalid_rate}",
    )


def sample_inputs_geometric(op, device, dtype, requires_grad, **kwargs):

    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=False)
    samples = (
        ((M,), 0.2),
        ((S, S), 0.5),
        ((S, S, S), 0.8),
    )
    for shape, rate in samples:
        yield SampleInput(make_arg(shape), args=(rate,))


def error_inputs_geometric(op, device, **kwargs):
    t = torch.zeros([10], device=device)
    neg_prob = -1
    yield ErrorInput(
        SampleInput(t, args=(neg_prob,)),
        error_type=RuntimeError,
        error_regex=fr"geometric_ expects p to be in \(0, 1\), but got p={neg_prob}",
    )


def sample_inputs_log_normal(op, device, dtype, requires_grad, **kwargs):

    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=False)
    samples = (
        ((M,), 0, 0.25),
        ((S, S), 0.5, 1),
        ((S, S, S), 0, 0.5),
    )
    for shape, mean, std in samples:
        yield SampleInput(make_arg(shape), args=(mean, std))


def error_inputs_log_normal(op, device, **kwargs):
    t = torch.zeros([10], device=device)
    invalid_std = 0
    yield ErrorInput(
        SampleInput(t, args=(0, invalid_std)),
        error_type=RuntimeError,
        error_regex=fr"log_normal_ expects std > 0.0, but found std={invalid_std}",
    )


def sample_inputs_uniform(op, device, dtype, requires_grad, **kwargs):

    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=False)
    samples = (
        ((M,), -100, 100),
        ((S, S), 0, 1),
        ((S, S, S), 1, 2),
    )
    for shape, hi, lo in samples:
        yield SampleInput(make_arg(shape), args=(hi, lo))

def sample_inputs_ones_zeros(op, device, dtype, requires_grad, **kwargs):
    # this is a bit messy, as we want the args to be tuples
    # so if we pass size as a tuple, we have a tuple containing a tuple
    sizes = (
        (M,),
        (S, S),
    )
    for size in sizes:
        yield SampleInput(size, kwargs={'dtype': dtype, 'device': device})

def sample_inputs_full(op, device, dtype, requires_grad, **kwargs):
    def get_val(dtype):
        return make_tensor([], dtype=dtype, device="cpu").item()

    sizes = (
        (M,),
        (S, S),
    )
    fill_values = [get_val(dtype), get_val(torch.int)]

    for size, fill_value in product(sizes, fill_values):
        yield SampleInput(size, fill_value, dtype=dtype, device=device)


def error_inputs_uniform(op, device, **kwargs):
    t = torch.zeros([10], device=device)
    yield ErrorInput(
        SampleInput(t, args=(3, -1)),
        error_type=RuntimeError,
        error_regex=r"uniform_ expects to return a \[from, to\) range, but found from=3 > to=-1",
    )


def error_inputs_linspace(op, device, **kwargs):
    yield ErrorInput(SampleInput(0, args=(3, -1)), error_type=RuntimeError, error_regex='number of steps must be non-negative')
    yield ErrorInput(
        SampleInput(0, args=(3, 1.)),
        error_type=TypeError,
        error_regex="received an invalid combination of arguments - got \\(int, int, float",
    )
    yield ErrorInput(
        SampleInput(torch.tensor([1, 1], device=device), args=(torch.tensor([3, 3], device=device), 1)),
        error_type=RuntimeError,
        error_regex="only supports 0-dimensional start and end tensors"
    )


def sample_inputs_linspace(op, device, dtype, requires_grad, **kwargs):
    ends = (-3, 0, 1, 4, 50)
    starts = (-2., 0, 4.3, 50)
    nsteps = (0, 1, 50)
    # Extra case to replicate off-by-one issue on CUDA
    cases = list(product(starts, ends, nsteps)) + [(0, 7, 50)]
    for start, end, nstep in cases:
        if dtype == torch.uint8 and (end < 0 or start < 0):
            continue
        yield SampleInput(start, args=(end, nstep), kwargs={"dtype": dtype, "device": device})

    yield SampleInput(1, args=(3, 1))


def sample_inputs_linspace_tensor_overload(op, device, dtype, requires_grad, **kwargs):
    ends = (-3, 0, 1, 4, 50)
    starts = (-2., 0, 4.3, 50)
    nsteps = (0, 1, 50)
    is_start_end_tensors = ((True, True), (True, False), (False, True))
    make_arg = partial(torch.tensor, device=device, requires_grad=False)

    # Extra case to replicate off-by-one issue on CUDA
    cases = list(product(starts, ends, nsteps, is_start_end_tensors)) + [(0, 7, 50, (True, True))]
    for start, end, nstep, (is_start_tensor, is_end_tensor) in cases:
        if dtype == torch.uint8 and (end < 0 or start < 0):
            continue

        tensor_options = {"dtype": dtype, "device": device}
        if is_start_tensor:
            start = make_arg(start, dtype=torch.float32 if isinstance(start, float) else torch.int64)
        if is_end_tensor:
            end = make_arg(end, dtype=torch.float32 if isinstance(end, float) else torch.int64)

        yield SampleInput(start, args=(end, nstep), kwargs=tensor_options)

    yield SampleInput(1, args=(3, 1))


def sample_inputs_logspace(op, device, dtype, requires_grad, **kwargs):
    ends = (-3, 0, 1.2, 2, 4)
    starts = (-2., 0, 1, 2, 4.3)
    nsteps = (0, 1, 2, 4)
    bases = (2., 1.1) if dtype in (torch.int8, torch.uint8) else (None, 2., 3., 1.1, 5.)
    for start, end, nstep, base in product(starts, ends, nsteps, bases):
        if dtype == torch.uint8 and end < 0 or start < 0:
            continue
        if nstep == 1 and isinstance(start, float) and not (dtype.is_complex or dtype.is_floating_point):
            # https://github.com/pytorch/pytorch/issues/82242
            continue
        if base is None:
            yield SampleInput(start, args=(end, nstep), kwargs={"dtype": dtype, "device": device})
        else:
            yield SampleInput(start, args=(end, nstep, base), kwargs={"dtype": dtype, "device": device})

    yield SampleInput(1, args=(3, 1, 2.))


def sample_inputs_logspace_tensor_overload(op, device, dtype, requires_grad, **kwargs):
    ends = (-3, 0, 1.2, 2, 4)
    starts = (-2., 0, 1, 2, 4.3)
    nsteps = (0, 1, 2, 4)
    bases = (2., 1.1) if dtype in (torch.int8, torch.uint8) else (None, 2., 3., 1.1, 5.)
    is_start_end_tensors = ((True, True), (True, False), (False, True))
    make_arg = partial(torch.tensor, device=device)
    for start, end, nstep, base, (is_start_tensor, is_end_tensor) in product(starts, ends, nsteps, bases, is_start_end_tensors):
        if dtype == torch.uint8 and end < 0 or start < 0:
            continue
        if nstep == 1 and isinstance(start, float) and not (dtype.is_complex or dtype.is_floating_point):
            # https://github.com/pytorch/pytorch/issues/82242
            continue

        tensor_options = {"dtype": dtype, "device": device}

        if (is_start_tensor):
            start = make_arg(start, dtype=torch.float32 if isinstance(start, float) else torch.int64)
        if (is_end_tensor):
            end = make_arg(end, dtype=torch.float32 if isinstance(end, float) else torch.int64)

        if base is None:
            yield SampleInput(start, args=(end, nstep), kwargs=tensor_options)
        else:
            yield SampleInput(start, args=(end, nstep, base), kwargs=tensor_options)

    yield SampleInput(1, args=(3, 1, 2.))


def sample_inputs_isclose(op, device, dtype, requires_grad, **kwargs):
    yield from sample_inputs_elementwise_binary(op, device, dtype, requires_grad, **kwargs)

    # Creates additional inputs to test the rtol, atol, and equal_nan params
    rtols = [0., 1e-7]
    atols = [0., 1e-7]
    equal_nans = [False, True]

    products = product(rtols, atols, equal_nans)

    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    for rtol, atol, equal_nan in products:
        lhs = make_arg((S, S), **op.lhs_make_tensor_kwargs)
        rhs = make_arg((S, S), **op.rhs_make_tensor_kwargs)

        yield SampleInput(lhs, args=(rhs,),
                          kwargs=dict(rtol=rtol, atol=atol, equal_nan=equal_nan))


def error_inputs_isclose(op, device, **kwargs):
    make_float_arg = partial(make_tensor, device=device, dtype=torch.float, requires_grad=False)

    yield ErrorInput(SampleInput(make_float_arg(()), args=(make_float_arg(()),), kwargs={'rtol': -0.4}),
                     error_type=RuntimeError,
                     error_regex='rtol must be greater than or equal to zero')

    yield ErrorInput(SampleInput(make_float_arg(()), args=(make_float_arg(()),), kwargs={'atol': -0.4}),
                     error_type=RuntimeError,
                     error_regex='atol must be greater than or equal to zero')


def sample_inputs_t(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(make_arg((1, 2)))
    yield SampleInput(make_arg((2,)))
    yield SampleInput(make_arg(()))


def sample_inputs_mm(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def make_arg_conj(size):
        return make_arg(size).conj().requires_grad_(requires_grad)

    first_shape, second_shape = (S, M), (M, S)

    yield SampleInput(make_arg(first_shape), args=(make_arg(second_shape),))

    if dtype.is_complex:
        yield SampleInput(make_arg(first_shape), args=(make_arg_conj(second_shape),))

    # Matmul of empty matrices
    yield SampleInput(make_arg((0, S)), args=(make_arg(S, M),))
    yield SampleInput(make_arg((S, 0)), args=(make_arg(0, M),))


def sample_inputs_addmm(op_info, device, dtype, requires_grad, **kwargs):
    alpha_val = kwargs.get('alpha', 2 + 3j if dtype.is_complex else 0.6 if dtype.is_floating_point else 2)
    beta_val = kwargs.get('beta', 1 + 2j if dtype.is_complex else 0.2 if dtype.is_floating_point else 3)
    tests_list = [
        ((2, 3), (2, 2), (2, 3), False),
        ((3, 3), (3, 3), (3, 3), False),
    ]
    tests_with_lhs_broadcasting = [
        ((1,), (2, 2), (2, 3), True),
        ((), (2, 2), (2, 3), True),
    ]
    test_cases = tests_list + tests_with_lhs_broadcasting  # type: ignore[operator]

    kwargs = dict(alpha=alpha_val, beta=beta_val)
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    for shape_a, shape_b, shape_c, broadcasts_input in test_cases:
        yield SampleInput(
            make_arg(shape_a),
            make_arg(shape_b),
            make_arg(shape_c),
            **kwargs,
        ).with_metadata(broadcasts_input=broadcasts_input)

    if dtype.is_complex:
        shape = (3, 3)
        yield SampleInput(
            make_arg(shape),
            make_arg(shape, requires_grad=False).mH.requires_grad_(requires_grad),
            make_arg(shape),
            **kwargs,
        )
        yield SampleInput(
            make_arg(shape),
            make_arg(shape),
            make_arg(shape, requires_grad=False).mH.requires_grad_(requires_grad),
            **kwargs,
        )
    # addmm of empty matrices
    if dtype.is_floating_point:
        yield SampleInput(make_arg(S, M), make_arg(S, 0), make_arg(0, M), **kwargs)
        # empty matmul with broadcastable input
        yield SampleInput(make_arg(M), make_arg(S, 0), make_arg(0, M), **kwargs).with_metadata(broadcasts_input=True)

def sample_inputs_sparse_sampled_addmm(op_info, device, dtype, requires_grad, **kwargs):
    alpha = 2 + 3j if dtype.is_complex else 0.6
    beta = 1 + 2j if dtype.is_complex else 0.2
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # sparse.sampled_addmm performs: alpha * (A @ B) * sparse_ones_like(C) + beta * C
    for m, n, k in itertools.product([0, 5], repeat=3):
        yield SampleInput(
            torch.eye(m, n, device=device, dtype=dtype)
            .to_sparse_csr()
            .requires_grad_(requires_grad),
            make_arg((m, k)),
            make_arg((k, n)),
            alpha=alpha,
            beta=beta,
        )

def sample_inputs_sparse_mm_reduce(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    reductions = ["sum", "mean", "amax", "amin"]
    for m, k, reduce in product([5, 7], [3, 11], reductions):
        yield SampleInput(
            torch.eye(m, m)
            .to(device=device, dtype=dtype)
            .to_sparse_csr()
            .requires_grad_(requires_grad),
            make_arg((m, k)),
            reduce,
        )


def sample_inputs_mv(self, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad)
    yield SampleInput(make_arg(S, M), make_arg(M))

def sample_inputs_bmm(self, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad)
    yield SampleInput(make_arg(M, S, M), make_arg(M, M, S))

def sample_inputs_dot_vdot(self, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def make_arg_conj(size):
        return make_arg(size).conj().requires_grad_(requires_grad)

    yield SampleInput(make_arg((S, )), make_arg((S, )))
    if dtype.is_complex:
        # dot/vdot for (conj(input), conj(arg_tensor)) and (conj(input), arg_tensor)
        # is tested in test_conj_view (which tests operations with only conjugated input tensor
        # -- not conjugated arg tensors)
        yield SampleInput(make_arg((S, )), make_arg_conj((S, )))


def error_inputs_dot_vdot(op_info, device, is_ref=False, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=torch.float32)

    yield ErrorInput(SampleInput(make_input(1), args=(make_input(3, dtype=torch.float16),)),
                     error_regex='dot : expected both vectors to have same dtype')
    yield ErrorInput(SampleInput(make_input(1, 1), args=(make_input(3),)),
                     error_regex='1D tensors expected')
    yield ErrorInput(SampleInput(make_input(9), args=(make_input(3),)),
                     error_regex='inconsistent tensor size')
    if device != "cpu" and not is_ref:
        yield ErrorInput(SampleInput(make_input(3), args=(make_input(3, device="cpu"),)),
                         error_regex='Expected all tensors to be on the same device')


def sample_inputs_addmv(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    test_cases = (((S,), (S, M), (M,), 1, 1, False),
                  ((S,), (S, M), (M,), 0.2, 0.6, False),
                  )

    test_cases_with_broadcast = (((1,), (S, M), (M,), 1, 1, True),
                                 ((1,), (S, M), (M,), 0.2, 0.6, True),
                                 ((), (S, M), (M,), 1, 1, True),
                                 ((), (S, M), (M,), 0.2, 0.6, True),
                                 )

    cases = test_cases + test_cases_with_broadcast

    # addmv performs: beta * M + alpha * (mat @ vec)
    for size, mat, vec, beta, alpha, broadcasts_input in cases:
        yield SampleInput(make_arg(size), args=(make_arg(mat), make_arg(vec)),
                          kwargs=dict(beta=beta, alpha=alpha), broadcasts_input=broadcasts_input)

def sample_inputs_addbmm(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # input_shape, batch1_shape, batch2_shape, beta_val, alpha_val, is_broadcasting
    test_cases = [((S, M), (S, S, S), (S, S, M), 1, 1, False),
                  ((1,), (S, S, S), (S, S, M), 1, 1, True),
                  ((S, M), (S, S, S), (S, S, M), 0.6, 0.2, False),
                  ((1,), (S, S, S), (S, S, M), 0.6, 0.2, True),
                  ((), (S, S, S), (S, S, M), 1, 1, True),
                  ((), (S, S, S), (S, S, M), 0.6, 0.2, True),
                  ]

    for input_shape, batch1_shape, batch2_shape, beta, alpha, is_broadcasting in test_cases:
        if dtype.is_complex:
            beta_complex, alpha_complex = beta * (1 + 2j), alpha * (2 + 3j)
            yield SampleInput(make_arg(input_shape), args=(make_arg(batch1_shape), make_arg(batch2_shape)),
                              kwargs=dict(beta=beta_complex, alpha=alpha_complex), broadcasts_input=is_broadcasting)
        yield SampleInput(make_arg(input_shape), args=(make_arg(batch1_shape), make_arg(batch2_shape)),
                          kwargs=dict(beta=beta, alpha=alpha), broadcasts_input=is_broadcasting)

def sample_inputs_addcmul_addcdiv(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    test_cases = [(((S, S), (S, S), (S, S)), False),
                  (((S, S), (S, 1), (1, S)), False),
                  (((1,), (S, S, 1), (1, S)), True),
                  (((), (), ()), False),
                  (((S, S), (), ()), True),
                  (((), (S, S, 1), (1, S)), True)
                  ]

    for input_args, broadcasts_input in test_cases:
        # addcdiv should accept inputs with zero value
        # Currently, it throws ZeroDivisionError when the denominator is zero
        # TODO: exclude_zeros can be removed after https://github.com/pytorch/pytorch/issues/73638 is fixed
        args = tuple(make_arg(arg, exclude_zero=True) if isinstance(arg, tuple) else arg
                     for arg in input_args)
        yield SampleInput(*args).with_metadata(broadcasts_input=broadcasts_input)

        # addcdiv should accept inputs with zero value
        # Currently, it throws ZeroDivisionError when the denominator is zero
        # TODO: exclude_zeros can be removed after https://github.com/pytorch/pytorch/issues/73638 is fixed
        args = tuple(make_arg(arg, exclude_zero=True) if isinstance(arg, tuple) else arg
                     for arg in input_args)
        yield SampleInput(
            *args, value=3.14 if dtype.is_floating_point or dtype.is_complex else 3
        ).with_metadata(broadcasts_input=broadcasts_input)

def reference_inputs_addcmul_addcdiv(op_info, device, dtype, requires_grad, **kwargs):
    yield from sample_inputs_addcmul_addcdiv(
        op_info, device, dtype, requires_grad, **kwargs)

    # type promotion cases
    supported_dtypes = op_info.supported_dtypes(device)
    make_arg = partial(make_tensor, device=device, requires_grad=requires_grad)

    types = (
        (torch.float64, torch.complex128),
        (torch.bfloat16, torch.float32),
    )

    values = (
        None,
        True, False,
        3.14, 3,
        1.0, 1,
        0.0, 0,
        -3.14, -3,
        3.14 + 2.71j,
    )

    for (type2, type3), value in product(types, values):
        if (type2 not in supported_dtypes or
                type3 not in supported_dtypes):
            continue

        # RuntimeError: value cannot be converted without overflow
        if (type(value) is complex and
                type2 is not torch.complex128):
            continue

        arg1 = make_arg([5, 5], dtype=dtype)
        arg2 = make_arg([5, 5], dtype=type2)
        arg3 = make_arg([1, 5], dtype=type3)

        # TypeError: addcdiv(): argument 'value' must be Number, not NoneType
        if value is not None:
            yield SampleInput(arg1, args=(arg2, arg3), kwargs=dict(value=value))
        else:
            yield SampleInput(arg1, args=(arg2, arg3))

def sample_inputs_baddbmm(op_info, device, dtype, requires_grad, **kwargs):
    test_cases = [((S, S, M), (S, S, S), (S, S, M), 1, 1, False),
                  ((1,), (S, S, S), (S, S, M), 1, 1, True),
                  ((S, S, M), (S, S, S), (S, S, M), 0.6, 0.2, False),
                  ((1,), (S, S, S), (S, S, M), 0.6, 0.2, True),
                  ((), (S, S, S), (S, S, M), 1, 1, True),
                  ((), (S, S, S), (S, S, M), 0.6, 0.2, True),
                  ]
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, low=None, high=None)
    for (input_shape, batch1_shape, batch2_shape, alpha, beta, broadcasts_input) in test_cases:
        yield SampleInput(
            make_arg(input_shape),
            make_arg(batch1_shape),
            make_arg(batch2_shape),
            beta=beta,
            alpha=alpha
        ).with_metadata(broadcasts_input=broadcasts_input)

        if dtype.is_complex:
            yield SampleInput(
                make_arg(input_shape),
                make_arg(batch1_shape),
                make_arg(batch2_shape),
                beta=beta * (1 + 2j),
                alpha=alpha * (2 + 3j),
            ).with_metadata(broadcasts_input=broadcasts_input)

    if dtype.is_complex:
        shapes = [(S, S, S), (S, M, S), (S, S, M)]
        args = tuple(make_arg(s) for s in shapes)
        yield SampleInput(
            args[0].transpose_(-1, 1),
            args[1].transpose(-1, 1).conj().requires_grad_(requires_grad),
            args[2].transpose(-1, 1).conj().requires_grad_(requires_grad),
            beta=beta * (1 + 2j),
            alpha=alpha * (2 + 3j),
        )

# TODO: add reduction kwargs
def sample_inputs_multilabel_soft_margin_loss(op_info, device, dtype, requires_grad, **kwargs):
    _make_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    shapes = (
        (S,),
        (S, S),
    )

    for shape in shapes:
        # Produce one with weight and one without.
        yield SampleInput(_make_tensor(shape), args=(_make_tensor(shape, requires_grad=False),), kwargs={})
        yield SampleInput(_make_tensor(shape), args=(_make_tensor(shape, requires_grad=False),),
                          kwargs={'weight': _make_tensor(shape, requires_grad=False)})

def sample_inputs_addr(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, low=None, high=None
    )
    yield SampleInput(make_arg(S, M), make_arg(S), make_arg(M))

    yield SampleInput(make_arg(), make_arg(S), make_arg(M)).with_metadata(broadcasts_input=True)

    if dtype.is_complex:
        alpha, beta = 0.1 + 0.3j, 0.4 + 0.6j
    elif dtype.is_floating_point:
        alpha, beta = 0.2, 0.6
    else:
        alpha, beta = 2, 3

    yield SampleInput(make_arg(S, M), make_arg(S), make_arg(M), beta=beta, alpha=alpha)

    yield SampleInput(
        make_arg(),
        make_arg(S),
        make_arg(M),
        beta=beta,
        alpha=alpha,
    ).with_metadata(broadcasts_input=True)

    # These samples fail gradcheck
    if dtype.is_floating_point and not requires_grad:
        tensor_options = dict(device=device, dtype=dtype, requires_grad=requires_grad)
        yield SampleInput(
            torch.tensor([[math.nan]], **tensor_options),
            torch.tensor([0.0], **tensor_options),
            torch.tensor([0.0], **tensor_options),
            beta=0.0,
            alpha=0.0,
        ).with_metadata(broadcasts_input=True)

        yield SampleInput(
            torch.tensor([[0.0]], **tensor_options),
            torch.tensor([math.nan], **tensor_options),
            torch.tensor([math.nan], **tensor_options),
            beta=0.0,
            alpha=0.0,
        ).with_metadata(broadcasts_input=True)

def sample_inputs_zero_(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases = ((), (S, S, S), (S,))

    for shape in cases:
        yield SampleInput(make_arg(shape))

def sample_inputs_multi_margin_loss(op_info, device, dtype, requires_grad, **kwargs):
    _make_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(_make_tensor, dtype=torch.long, requires_grad=False)
    make_weight = partial(_make_tensor, requires_grad=False)

    inputs = (
        ((), make_target([], low=0, high=1), {}),
        ((S,), make_target([], low=0, high=S), {"p": 1}),
        ((S,), make_target([1], low=0, high=S), {"p": 2}),
        ((S, M), make_target([S], low=0, high=M), {"margin": 1.0}),
        ((S, M), make_target([S], low=0, high=M), {"margin": -3.14}),
        ((M, S), make_target([M], low=0, high=S), {"weight": None}),
        ((M, S), make_target([M], low=0, high=S), {"weight": make_weight([S], low=-10., high=10.)}),
        ((M, S), make_target([M], low=0, high=S), {"reduction": "none"}),
        ((M, S), make_target([M], low=0, high=S), {"reduction": "mean"}),
        ((M, S), make_target([M], low=0, high=S), {"reduction": "sum"}),
    )

    for input_shape, target, kwargs in inputs:
        yield SampleInput(_make_tensor(input_shape), args=(target,), kwargs=kwargs)


def reference_inputs_multi_margin_loss(op_info, device, dtype, requires_grad, **kwargs):
    yield from sample_inputs_multi_margin_loss(op_info, device, dtype, requires_grad, **kwargs)
    _make_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(_make_tensor, dtype=torch.long, requires_grad=False)
    make_weight = partial(_make_tensor, requires_grad=False)

    inputs = (
        ((), make_target([], low=0, high=1)),
        ((S,), make_target([], low=0, high=S)),
        ((S,), make_target([1], low=0, high=S)),
        ((M, S), make_target([M], low=0, high=S)),
    )
    ps = (1, 2)
    margins = (0, 7, -3.14)
    weights = (False, True)
    reductions = (None, "none", "mean", "sum")

    for (input_shape, target), p, margin, weight, reduction in product(inputs, ps, margins, weights, reductions):
        input = _make_tensor(input_shape)
        weight_shape = [input.size(-1)] if input.ndim > 0 else [1]
        weight = make_weight(weight_shape, low=-10., high=10.) if weight else None
        kwargs = {"p": p, "margin": margin, "weight": weight}
        if reduction is not None:
            kwargs["reduction"] = reduction
        yield SampleInput(input, args=(target,), kwargs=kwargs)


def error_inputs_multi_margin_loss(op, device, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=torch.float32)
    # invalid reduction
    yield ErrorInput(SampleInput(make_input(5, 4), args=(make_input(5,),), kwargs={'reduction': 'abc'}),
                     error_type=ValueError, error_regex='abc is not a valid value for reduction')
    # invalid input
    yield ErrorInput(SampleInput(make_input(5, 0), args=(make_input(5,),), kwargs={}),
                     error_type=RuntimeError,
                     error_regex=r'Expected non-empty vector or matrix with optional 0-dim batch size, but got: \[5, 0\]')
    yield ErrorInput(SampleInput(make_input(0,), args=(make_input(5,),), kwargs={}),
                     error_type=RuntimeError,
                     error_regex=r'Expected non-empty vector or matrix with optional 0-dim batch size, but got: \[0\]')
    # invalid target
    yield ErrorInput(SampleInput(make_input(5, 4), args=(make_input(5, 4),), kwargs={}),
                     error_type=RuntimeError, error_regex=r'inconsistent target size, expected 5 but got \[5, 4\]')
    # invalid target dtype
    yield ErrorInput(SampleInput(make_input(5, 4), args=(make_input(5,),), kwargs={}),
                     error_type=RuntimeError, error_regex='expected scalar type Long but found Float')
    # invalid weight
    yield ErrorInput(SampleInput(make_input(5, 4), args=(make_input(5,),), kwargs={'weight': make_input(())}),
                     error_type=ValueError, error_regex='weight must be one-dimensional')
    yield ErrorInput(SampleInput(make_input(5, 4), args=(make_input(5,),), kwargs={'weight': make_input(5, 4)}),
                     error_type=ValueError, error_regex='weight must be one-dimensional')
    yield ErrorInput(SampleInput(make_input(5, 4), args=(make_input(5,),), kwargs={'weight': make_input(5,)}),
                     error_type=RuntimeError, error_regex=r'inconsistent weight size, expected 4 but got \[5\]')
    # invalid p
    yield ErrorInput(SampleInput(make_input(5, 4), args=(make_input(5,),), kwargs={'p': 3}),
                     error_type=ValueError, error_regex='only p == 1 and p == 2 supported')


def sample_inputs_logsumexp(self, device, dtype, requires_grad, **kwargs):
    inputs = (
        ((), (0,), True),
        ((S, S), (1,), True),
        ((S, S), (1,), False),
        ((S, S), (-2,), False),
        ((S, S), (0, 1), False),
    )
    # Test large inputs to check numerical stability
    lows = (None, 1e3, 1e6) if dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128) else (None,)
    for low in lows:
        high = low * 2 if low is not None else None
        for shape, dim, keepdim in inputs:
            t = make_tensor(shape, dtype=dtype, device=device,
                            low=low, high=high,
                            requires_grad=requires_grad)
            yield SampleInput(t, dim, keepdim)

def reference_inputs_logsumexp(op, device, dtype, requires_grad, **kwargs):
    yield from sample_inputs_logsumexp(op, device, dtype, requires_grad, **kwargs)

    # https://github.com/pytorch/pytorch/issues/91843
    t = torch.tensor([20, 30, 100], dtype=dtype, device=device, requires_grad=requires_grad)
    yield SampleInput(t, 0, False)

    t = torch.tensor((), dtype=dtype, device=device, requires_grad=requires_grad)
    yield SampleInput(t, 0, False)

    # tests masking
    # https://github.com/pytorch/pytorch/pull/91860#pullrequestreview-1241344073
    t = torch.tensor(float("inf"))
    yield SampleInput(t, 0, True)

def sample_inputs_like_fns(self, device, dtype, requires_grad, **kwargs):
    inputs = [
        ((), {}),
        ((S, S), {}),
        ((0, S, 0), {}),
        ((S,), {'dtype': dtype, 'device': device}),
        # Hard-code some dtypes/devices. We want to test cases where the
        # (dtype, device) is different from the input's (dtype, device)
        ((S,), {'dtype': torch.double if device != 'mps:0' else torch.float}),
        ((S,), {'device': 'cpu'}),
        ((S,), {'dtype': torch.double, 'device': 'cpu'}),
    ]
    if torch.cuda.is_available():
        inputs.append(((S,), {'device': 'cuda'}))

    for shape, kwargs in inputs:
        t = make_tensor(shape, dtype=dtype, device=device,
                        low=None, high=None,
                        requires_grad=requires_grad)
        yield SampleInput(t, **kwargs)

def reference_inputs_like_fns(op, device, dtype, requires_grad, **kwargs):
    yield from sample_inputs_like_fns(op, device, dtype, requires_grad, **kwargs)

    # shape
    cases = (
        (), (0,), (1, 0), (1, 1, 4, 5), (5, 3, 0, 1), (1, 4, 3, 1, 1)
    )

    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    for shape in cases:
        yield SampleInput(make_arg(shape))
        yield SampleInput(make_arg(shape).transpose(0, -1))
        yield SampleInput(make_arg(shape, noncontiguous=True))
        yield SampleInput(make_arg(shape, noncontiguous=True).transpose(0, -1))

def sample_inputs_multilabel_margin_loss(op_info, device, dtype, requires_grad, **kwargs):
    _make_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(_make_tensor, dtype=torch.long, requires_grad=False)

    inputs = (
        ([], make_target([], low=0, high=1), {}),
        ([S], make_target([S], low=0, high=S), {}),
        ([M, S], make_target([M, S], low=0, high=S), {}),
        ([M, S], make_target([M, S], low=0, high=S), {"reduction": "none"}),
        ([M, S], make_target([M, S], low=0, high=S), {"reduction": "mean"}),
        ([M, S], make_target([M, S], low=0, high=S), {"reduction": "sum"}),
    )

    for shape, target, kwargs in inputs:
        yield SampleInput(_make_tensor(shape), args=(target,), kwargs=kwargs)


def reference_inputs_multilabel_margin_loss(op_info, device, dtype, requires_grad, **kwargs):
    yield from sample_inputs_multilabel_margin_loss(op_info, device, dtype, requires_grad, **kwargs)
    _make_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(_make_tensor, dtype=torch.long, requires_grad=False)
    make_target_tensor = partial(torch.tensor, device=device, dtype=torch.long, requires_grad=False)

    inputs = (
        # random tests including -1 target labels
        ([], make_target([], low=-1, high=1)),
        ([S], make_target([S], low=-1, high=S)),
        ([M, S], make_target([M, S], low=-1, high=S)),
        # repeated target labels and -1 (labels after the first -1 are ignored)
        ([], make_target_tensor(-1)),
        ([7], make_target_tensor([2, 0, 6, -1, 4, -1, 6])),
        ([4, 5], make_target_tensor([[4, -1, 0, -1, 2], [0, 0, 4, 1, 4], [-1, 3, -1, 1, 0], [4, 3, 2, 1, 0]])),
    )
    reductions = (None, "none", "mean", "sum")

    for (shape, target), reduction in product(inputs, reductions):
        kwargs = {}
        if reduction is not None:
            kwargs["reduction"] = reduction
        yield SampleInput(_make_tensor(shape), args=(target,), kwargs=kwargs)


def error_inputs_multilabel_margin_loss(op, device, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=torch.float32)
    # invalid reduction
    yield ErrorInput(SampleInput(make_input(5, 4), args=(make_input(5, 4),), kwargs={'reduction': 'abc'}),
                     error_type=ValueError, error_regex='abc is not a valid value for reduction')
    # invalid input
    yield ErrorInput(SampleInput(make_input(5, 0), args=(make_input(5, 4),), kwargs={}),
                     error_type=RuntimeError,
                     error_regex=r'Expected non-empty vector or matrix with optional 0-dim batch size, but got: \[5, 0\]')
    yield ErrorInput(SampleInput(make_input(0,), args=(make_input(0,),), kwargs={}),
                     error_type=RuntimeError,
                     error_regex=r'Expected non-empty vector or matrix with optional 0-dim batch size, but got: \[0\]')
    # invalid target
    yield ErrorInput(SampleInput(make_input(5, 4), args=(make_input(4,),), kwargs={}),
                     error_type=RuntimeError,
                     error_regex=r'inconsistent target size: \[4\] for input of size: \[5, 4\]')
    yield ErrorInput(SampleInput(make_input(5, 4), args=(make_input((),),), kwargs={}),
                     error_type=RuntimeError,
                     error_regex=r'inconsistent target size: \[\] for input of size: \[5, 4\]')


def get_independent_tensor(tensor):
    return tensor.clone().requires_grad_(tensor.requires_grad)

def sample_inputs_randint(self, device, dtype, requires_grad, **kwargs):
    low = 2
    high = 10

    for sample in sample_inputs_like_fns(self, device, dtype, requires_grad, **kwargs):
        sample.kwargs.setdefault('device', device)
        # With high
        yield SampleInput(high, sample.input.shape, *sample.args, **sample.kwargs)
        # With low and high
        yield SampleInput(low, high, sample.input.shape, *sample.args, **sample.kwargs)

def sample_inputs_randint_like(self, device, dtype, requires_grad, **kwargs):
    low = 2
    high = 10

    for sample in sample_inputs_like_fns(self, device, dtype, requires_grad, **kwargs):
        # With high
        yield SampleInput(
            sample.input,
            high,
            *sample.args,
            **sample.kwargs)
        # With low and high
        yield SampleInput(
            get_independent_tensor(sample.input),
            low,
            high,
            *sample.args,
            **sample.kwargs)

def sample_inputs_margin_ranking_loss(op_info, device, dtype, requires_grad, **kwargs):
    _make_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    shapes = (
        (),
        (S,),
        (S, S),
        (S, S, S),
    )

    margins = (0., 1.)
    reductions = ('sum', 'mean', 'none')

    for shape in shapes:
        for margin, reduction in product(margins, reductions):
            kwargs = {'margin': margin, 'reduction': reduction}
            yield SampleInput(_make_tensor(shape),
                              args=(_make_tensor(shape, requires_grad=False),
                                    _make_tensor(shape, requires_grad=False)),
                              kwargs=kwargs)

def reference_inputs_margin_ranking_loss(op, device, dtype, requires_grad, **kwargs):
    yield from sample_inputs_margin_ranking_loss(op, device, dtype, requires_grad, **kwargs)
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    for reduction in ('sum', 'mean', 'none'):
        if dtype.is_floating_point:  # only supports ints and floats
            # NaN propagation
            inp1 = make_input((10, ))
            inp1[2] = float('nan')
            inp2 = make_input((10, ))
            inp2[4] = float('nan')
            target = make_input((10, ))
            inp2[9] = float('nan')
            yield SampleInput(inp1, args=(inp2, target), kwargs={'reduction': reduction})

            # Inf handling
            inp1 = make_input((10, ))
            inp2[1] = float('inf')
            inp2 = make_input((10, ))
            inp2[4] = float('inf')
            target = make_input((10, ))
            inp2[7] = float('inf')
            yield SampleInput(inp1, args=(inp2, target), kwargs={'reduction': reduction})

        # Broadcasting
        inp1 = make_input((5, 2))
        inp2 = make_input((5, 1))
        target = make_input((1, 2))
        yield SampleInput(inp1, args=(inp2, target), kwargs={'reduction': reduction})

def error_inputs_margin_ranking_loss(op, device, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=torch.float32)
    # invalid reduction value.
    yield ErrorInput(SampleInput(make_input(5, 4), args=(make_input(5, 4), make_input(5, 4),), kwargs={'reduction': 'abc'}),
                     error_type=ValueError, error_regex='is not a valid value')
    # invalid input shapes
    yield ErrorInput(SampleInput(make_input(5, 4), args=(make_input(5, 4), make_input(5,),)),
                     error_regex='margin_ranking_loss : All input tensors should')

def sample_inputs_new_fns(self, device, dtype, requires_grad, *, is_strided=False, **kwargs):
    # input_shape, output_shape, strides, kwargs
    # lengths of output_shape and strides must be equal
    inputs = [
        ((), (), (), {}),
        ((S, S), (2, 0), (3, 4), {}),
        ((0, S, 0), (3, 2, 2), (1, 2, 3), {}),
        ((S,), (2, 3), (7, 8), {'dtype': dtype, 'device': device}),
        # Hard-code some dtypes/devices. We want to test cases where the
        # (dtype, device) is different from the input's (dtype, device)
        ((S,), (10,), (S,), {'dtype': torch.double if device != 'mps:0' else torch.float}),
        ((S,), (1, 1, 12), (S, L, M), {'device': 'cpu'}),
        ((S,), (2, 2, 2), (L, M, S), {'dtype': torch.double, 'device': 'cpu'}),
    ]
    if torch.cuda.is_available():
        inputs.append(((S,), (7, 2), (3, 4), {'device': 'cuda'}))

    for input_shape, output_shape, strides, kwargs in inputs:
        t = make_tensor(input_shape, dtype=dtype, device=device,
                        low=None, high=None,
                        requires_grad=requires_grad)
        if is_strided:
            yield SampleInput(t, output_shape, strides, **kwargs)
        else:
            yield SampleInput(t, output_shape, **kwargs)

def sample_inputs_empty_strided(op, device, dtype, requires_grad=False, **kwargs):

    inputs = [
        ((), (), {'dtype': dtype, 'device': device}),
        ((S,), (4,), {'dtype': dtype, 'device': device}),
        ((S, S), (2, 1), {'dtype': dtype, 'device': device}),
        ((S, S, S), (2, 0, 1), {'dtype': dtype, 'device': device}),
    ]

    for shape, strides, kwargs in inputs:
        yield SampleInput(shape, strides, requires_grad=requires_grad, **kwargs)

def sample_inputs_empty(op, device, dtype, requires_grad, **kwargs):
    # shape
    cases = (
        (), (0,), (1,), (1, 3, 5), (5, 3, 1), (1, 0, 5, 1),
    )

    for case in cases:
        yield SampleInput(case, device=device, dtype=dtype, requires_grad=requires_grad)

def sample_inputs_empty_permuted(op, device, dtype, requires_grad, **kwargs):
    # shape
    cases = (
        (), (0,), (1,), (1, 3, 5), (5, 3, 1), (1, 0, 5, 1),
    )

    for case in cases:
        for layout in itertools.permutations(range(len(case))):
            yield SampleInput(case, layout, device=device, dtype=dtype, requires_grad=requires_grad)

def error_inputs_empty_permuted(op_info, device, **kwargs):
    yield ErrorInput(
        SampleInput((2,), args=((0, 1),)),
        error_type=RuntimeError,
        error_regex="Number of dimensions in size does not match the length of the physical_layout"
    )
    yield ErrorInput(
        SampleInput((2,), args=((3,),)),
        error_type=RuntimeError,
        error_regex="Dimension out of range"
    )
    yield ErrorInput(
        SampleInput((2, 3), args=((0, 0),)),
        error_type=RuntimeError,
        error_regex="Duplicate dim not allowed"
    )

def sample_inputs_scalar_tensor(op, device, dtype, requires_grad, **kwargs):
    # Not including a scalar tensor in vals because meta tests start failing due to
    # lack of meta support for _local_scalar_dense
    # torch.tensor(2, device=device)
    vals = (-5, 0, 1)

    for item in vals:
        yield SampleInput(item, device=device, dtype=dtype, requires_grad=requires_grad)

def sample_inputs_eye(op, device, dtype, requires_grad, **kwargs):
    # only ints >= 0 are allowed for both arguments, unless m is omitted
    sizes = (None, 0, 1, 2, 3, 4, 7, L, M, S)

    for n, m in product(sizes, sizes):
        if n is None:
            continue

        # TODO: no layout
        _kwargs = {'device': device, 'dtype': dtype, 'requires_grad': requires_grad}
        if m is None:
            yield SampleInput(n, args=(), kwargs=_kwargs)
        else:
            yield SampleInput(n, args=(m,), kwargs=_kwargs)

def error_inputs_eye(op_info, device, **kwargs):
    # TODO: no layout
    _kwargs = {'device': device, 'dtype': torch.float32}

    yield ErrorInput(
        SampleInput(-1, args=(), kwargs=_kwargs),
        error_regex="n must be greater or equal to 0, got -1"
    )

    yield ErrorInput(
        SampleInput(-7, args=(42,), kwargs=_kwargs),
        error_regex="n must be greater or equal to 0, got -7"
    )

    yield ErrorInput(
        SampleInput(0, args=(-3,), kwargs=_kwargs),
        error_regex="m must be greater or equal to 0, got -3"
    )


def sample_inputs_new_full(self, device, dtype, requires_grad, **kwargs):
    def get_val(dtype):
        return make_tensor([], dtype=dtype, device="cpu").item()

    for sample in sample_inputs_new_fns(self, device, dtype, requires_grad, **kwargs):
        # The scalar we are passing to new_full must be the same dtype
        # as the one of the resulting tensor
        use_dtype = sample.kwargs.get('dtype', dtype)
        yield SampleInput(
            sample.input, *sample.args, get_val(use_dtype), **sample.kwargs)

def sample_inputs_full_like(self, device, dtype, requires_grad, **kwargs):
    def get_val(dtype):
        return make_tensor([], dtype=dtype, device="cpu").item()

    double_dtype = torch.double if device != "mps:0" else torch.float
    inputs = [
        ((), get_val(dtype), {}),
        ((S, S), get_val(dtype), {}),
        ((0, S, 0), get_val(dtype), {}),
        ((S,), get_val(dtype), {'dtype': dtype, 'device': device}),
        # Hard-code some dtypes/devices. We want to test cases where the
        # (dtype, device) is different from the input's (dtype, device)
        ((S,), get_val(double_dtype), {'dtype': double_dtype}),
        ((S,), get_val(dtype), {'device': 'cpu'}),
        ((S,), get_val(double_dtype), {'dtype': double_dtype, 'device': 'cpu'}),
    ]
    if torch.cuda.is_available():
        inputs.append(((S,), get_val(dtype), {'device': 'cuda'}))

    if torch.mps.is_available() and dtype not in [torch.float64, torch.complex128, torch.uint32, torch.uint16]:
        inputs.append(((S,), get_val(dtype), {'device': 'mps'}))

    if not dtype.is_signed:
        # For unsigned dtypes, negative values are converted.
        inputs.append(((S,), -get_val(dtype), {}))

    for shape, fill_value, kwargs in inputs:
        t = make_tensor(shape, dtype=dtype, device=device,
                        low=None, high=None,
                        requires_grad=requires_grad)
        yield SampleInput(t, fill_value, **kwargs)

def sample_inputs_multinomial(self, device, dtype, requires_grad, **kwargs):
    cases = [
        ([3], 3, {}),
        ([10], 3, {}),
        ([3, 10], 3, {}),
        ([3], 3, dict(replacement=False)),
        ([3], 3, dict(replacement=True)),
        ([3, 4], 4, dict(replacement=True)),
        ([3, 4], 4, dict(replacement=False)),
    ]

    for shape, num_samples, kwargs in cases:
        t = make_tensor(shape, dtype=dtype, device=device,
                        low=0, high=None,
                        requires_grad=requires_grad)
        yield SampleInput(t, num_samples, **kwargs)

def sample_inputs_normal_common(self, device, dtype, requires_grad, cases, **kwargs):
    def get_value_or_make_tensor(value_or_shape):
        if isinstance(value_or_shape, list):
            return make_tensor(value_or_shape, dtype=dtype, device=device,
                               low=0, high=None,
                               requires_grad=requires_grad)
        return value_or_shape

    for value_or_mean_shape, value_or_std_shape, kwargs in cases:
        mean = get_value_or_make_tensor(value_or_mean_shape)
        std = get_value_or_make_tensor(value_or_std_shape)
        yield SampleInput(mean, std, **kwargs)

def sample_inputs_normal_tensor_first(self, device, dtype, requires_grad, **kwargs):
    # value_or_size, value_or_size, kwargs
    cases = [
        ([], [], {}),
        ([3], [3], {}),
        ([3, 4, 2], [3, 4, 2], {}),
        ([2, 3], 1.1, {}),
        ([1, 2, 3], [5, 2, 3], {}),  # broadcasting
    ]

    return sample_inputs_normal_common(self, device, dtype, requires_grad, cases, **kwargs)

def sample_inputs_normal_tensor_second(self, device, dtype, requires_grad, **kwargs):
    yield SampleInput(1.6, 0.3, [2, 3], dtype=dtype, device=device)
    yield SampleInput(1.6, 0.3, [2, 2, 2], dtype=dtype, layout=torch.strided, device=device)
    yield SampleInput(2.7, make_tensor([4, 3], dtype=dtype, device=device, low=0, high=None, requires_grad=requires_grad))

def sample_inputs_bernoulli(self, device, dtype, requires_grad, **kwargs):
    shapes = [
        [3],
        [],
        [0, 3],
        [2, 3, 4],
    ]

    for shape in shapes:
        t = make_tensor(shape, dtype=dtype, device=device,
                        low=0, high=1,
                        requires_grad=requires_grad)
        yield SampleInput(t)

def error_inputs_bernoulli(op_info, device, **kwargs):
    # more than one element of the written-to tensor refers to a single memory location
    x = torch.rand((1,), device=device).expand((6,))
    err_msg = 'unsupported operation'
    yield ErrorInput(SampleInput(torch.rand_like(x), kwargs={'out': x}),
                     error_regex=err_msg)

def sample_inputs_logcumsumexp(self, device, dtype, requires_grad, **kwargs):
    inputs = (
        ((S, S, S), 0),
        ((S, S, S), 1),
        ((), 0),
    )

    for large_number in (True, False):
        for shape, dim in inputs:
            t = make_tensor(shape, dtype=dtype, device=device,
                            low=None, high=None,
                            requires_grad=requires_grad)

            if large_number and t.dim() > 0:
                t[0] = 10000
            yield SampleInput(t, dim)

def sample_inputs_trace(self, device, dtype, requires_grad, **kwargs):
    yield SampleInput(
        make_tensor((S, S), dtype=dtype, device=device,
                    low=None, high=None,
                    requires_grad=requires_grad))


def error_inputs_trace(op, device):
    yield ErrorInput(SampleInput(make_tensor((3, 4, 5), dtype=torch.float32, device=device)), error_regex="expected a matrix")


def sample_inputs_renorm(self, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    cases = (((S, S, S), (2, 1, 0.5)),
             ((S, S, S), (2, -1, 0.5)),
             ((S, S, S), (1, 2, 3)),
             ((S, S, S), (float('inf'), 2, 0.5)),
             )

    for shape, args in cases:
        yield SampleInput(make_arg(shape), args=args)


def sample_inputs_transpose_swapdims(self, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    cases = (((1, 2, 3), (-1, -2)),
             ((1, 2, 3), (-1, 2)),
             ((1, 2, 3), (1, -2)),
             ((1, 2, 3), (1, 2)),
             ((), (0, 0)),
             ((1, ), (0, 0)),
             ((M, M), (0, 1)),
             ((S, S, S), (2, 0)), )

    for shape, args in cases:
        yield SampleInput(make_arg(shape), args=args)

def _numpy_ref_transpose(a, dim0, dim1):
    if a.ndim <= 1:
        return a

    return np.swapaxes(a, dim0, dim1)

def sample_inputs_adjoint(self, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    shapes = ((1, 2, 3), (M, M), (S, S, S), (S, M, S), (M, S, M, S))
    return (SampleInput(make_arg(shape)) for shape in shapes)

def sample_inputs_T(self, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    shapes = ((M, M), (M, L))
    return (SampleInput(make_arg(shape)) for shape in shapes)

def error_inputs_T(self, device, has_ndims_error=False):
    make_arg = partial(make_tensor, device=device, dtype=torch.float32)

    # Deprecated behavior in regular PyTorch, but throws an error in primTorch:
    # https://github.com/pytorch/pytorch/issues/86968
    if has_ndims_error:
        # ndims == 1
        yield ErrorInput(SampleInput(make_arg(M)),
                         error_regex=(r'The use of `x\.T` on tensors of dimension other than 0 or 2 '
                                      r'to reverse their shape is not supported\.'))

        # ndims > 2
        yield ErrorInput(SampleInput(make_arg(M, S, L)),
                         error_regex=(r'The use of `x\.T` on tensors of dimension other than 0 or 2 '
                                      r'to reverse their shape is not supported\.'))


def sample_inputs_singular_matrix_factors(op_info, device, dtype, requires_grad=False):
    """
    This function produces two tensors of shape (*, m, k) and (*, n, k) with k <= min(m, n).
    Their matrix product could be used to generate tensor of shape (*, m, n) of rank k.
    """

    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    batches = [(), (2,)]
    size = [3, 4]
    for batch, m, n in product(batches, size, size):
        k = 2
        a = make_arg((*batch, m, k))
        b = make_arg((*batch, n, k))
        yield a, b


def sample_inputs_svd_lowrank(op_info, device, dtype, requires_grad=False, **kwargs):
    # Function that's well defined on the outputs for complex inputs
    def fn(usv):
        U, S, V = usv
        return U @ V.mH, S

    for (a, b) in sample_inputs_singular_matrix_factors(op_info, device, dtype, requires_grad):
        *batch, m, k = a.shape
        n = b.shape[-2]

        # NOTE: since svd_lowrank relies on non rank-revealing SVD,
        # it inherits the problem of unstable behavior with repeated
        # singular values including zeros.
        # Since we want to avoid (repeated) zeros as singular values,
        # we can only use k for q.
        # This issues could be resolved with using a rank-revealing SVD
        # which does not include "zero" singular values.
        yield SampleInput(a, b, q=k, M=None).with_metadata(output_process_fn_grad=fn)

    for (a, b) in sample_inputs_singular_matrix_factors(op_info, device, dtype, requires_grad):
        *batch, m, k = a.shape
        n = b.shape[-2]
        M = make_tensor((*batch, m, n), dtype=dtype, device=device, requires_grad=requires_grad)
        yield SampleInput(a, b, q=k, M=M).with_metadata(output_process_fn_grad=fn)

def chunk_iter(iterable, size):
    it = iter(iterable)
    while True:
        chunk = tuple(islice(it, size))
        if not chunk:
            break
        yield chunk

def sample_inputs_pca_lowrank(op_info, device, dtype, requires_grad=False, **kwargs):
    # we reuse samples from svd_lowrank which come in group of two with
    # kwarg['M'] = None and with kwarg['M'] = <some tensor>
    samples = sample_inputs_svd_lowrank(op_info, device, dtype, requires_grad, **kwargs)
    for s1, s2 in chunk_iter(samples, 2):
        del s1.kwargs['M']
        del s2.kwargs['M']
        s1.kwargs['center'] = False
        s2.kwargs['center'] = True
        yield s1
        yield s2

def np_sinc_with_fp16_as_fp32(x):
    # Wraps numpy's sinc function so that fp16 values are promoted to fp32
    # before sinc is invoked. Context: numpy's sinc returns NaN when evaluated
    # at 0 for fp16.
    if x.dtype == np.float16:
        return np.sinc(x.astype(np.float32))
    else:
        return np.sinc(x)

def sample_inputs_broadcast_to(op_info, device, dtype, requires_grad, **kwargs):
    test_cases = (
        ((S, 1, 1), (S, S, S)),
        ((S, 1, S), (S, S, S)),
        ((S, 1), (S, S, S)),
        ((1,), (S, S, S)),
        ((1, S), (1, 1, S)),
        ((), ()),
        ((), (1, 3, 2)),
    )

    return (
        SampleInput(
            make_tensor(size, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad),
            shape,
        ) for size, shape in test_cases)

def sample_inputs_broadcast_tensors(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    test_cases: tuple[tuple] = (((3,), (1, 2, 1), (1, 1), (5, 1, 1),),)

    for shape, *other_shapes in test_cases:
        yield SampleInput(make_arg(shape), args=tuple(make_arg(s) for s in other_shapes))

def reference_inputs_broadcast_tensors(op, device, dtype, requires_grad, **kwargs):
    yield from sample_inputs_broadcast_tensors(op, device, dtype, requires_grad, **kwargs)

    m = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    n = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad, noncontiguous=True)

    cases = (
        ((), (1, 1), (1, 1, 7, 1), (3, 1, 1)),
        ((3, 5, 6), (1, 3, 5, 6), (1, 1, 1, 1, 6), (8, 3, 5, 6))
    )

    for a, b, c, d in cases:
        yield SampleInput(m(a), args=(m(b), m(c), m(d)))
        yield SampleInput(n(a), args=(n(b), n(c), n(d)))

def sample_inputs_block_diag(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    test_cases: tuple[tuple] = (
        ((1, S), (2, S), (3, S),),
        ((S, 1), (S, 2), (S, 3),),
        ((1,), (2,), (3,),),
        ((2, S), (S,))
    )

    for shape, *other_shapes in test_cases:
        yield SampleInput(make_arg(shape), args=tuple(make_arg(s) for s in other_shapes))
        # We also want to test mixed complex-non-complex inputs to block_diag
        if dtype == torch.complex32 or dtype == torch.complex64:
            non_complex_dtype = torch.float32 if dtype == torch.complex32 else torch.float64
            make_arg_non_complex = partial(make_tensor, dtype=non_complex_dtype, device=device, requires_grad=requires_grad)
            yield SampleInput(make_arg_non_complex(shape), args=tuple(make_arg(s) for s in other_shapes))

def sample_inputs_cdist(op_info, device, dtype, requires_grad, **kwargs):
    small_S = 2
    test_cases = (
        ((S, S, 2), (S, S + 1, 2)),
        ((S, S), (S, S)),
        ((S, S, S), (S, S, S)),
        ((3, 5), (3, 5)),
        ((2, 3, 5), (2, 3, 5)),
        ((1, 2, 3), (1, 2, 3)),
        ((1, 1), (S, 1)),
        ((0, 5), (4, 5)),
        ((4, 5), (0, 5)),
        ((0, 4, 5), (3, 5)),
        ((4, 5), (0, 3, 5)),
        ((0, 4, 5), (1, 3, 5)),
        ((1, 4, 5), (0, 3, 5)),
        # Using S here would make this one test take 9s
        ((small_S, small_S, small_S + 1, 2), (small_S, small_S, small_S + 2, 2)),
        ((small_S, 1, 1, small_S), (1, small_S, small_S)),
        ((1, 1, small_S), (small_S, 1, small_S, small_S)),
    )

    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
        # FIXME add an override for JIT and revert 0. back to 0
        # since it's accepted by eager
        for p in [0., 1., 2., 3., 0.5, 1.5, 2.5, float("inf")]:
            for t1_size, t2_size in test_cases:
                # The args should never be non-contiguous as this is not supported in the backward
                yield SampleInput(make_arg(t1_size), make_arg(t2_size), p, cm)

def _fill_np(a, value):
    a = a.copy()
    a.fill(value)
    return a

def _fill_sample_kwargs(device, dtype, input):
    if dtype is torch.bool:
        value = True
    else:
        value = 3

    return ({'value': value}, {'value': value})

def sample_inputs_comparison_ops(op, device, dtype, requires_grad, **kwargs):
    yield from sample_inputs_elementwise_binary(op, device, dtype, requires_grad, **kwargs)

    # Adds a sample input where both tensors have the same values
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    lhs = make_arg((S, S))
    yield SampleInput(lhs, args=(lhs.clone(),))

def sample_inputs_stack(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # shape x number of tensors
    cases = (
        ((3, 4), 1),
        ((1, 2, 1, 4), 3),
        ((0, 1, 0), 2),)

    for shape, num_tensors in cases:
        tensors = [make_arg(shape) for _ in range(num_tensors)]
        for dim in range(-1, len(shape) - 1):
            yield SampleInput(tensors, args=(dim,))


def sample_inputs_chunk_cat(op_info, device, dtype, requires_grad, **kwargs):
    # 1. If input tensors have different ndims, dim should be non-negative and be less than the ndims of every input tensors.
    #    If all input tensors have the same ndims, we support both negative and non-negative dim.
    # 2. For wrapped_dim, all tensors should have the same size for 0,...,wrapped_dim-1 dimensions.
    #        No requirements for (wrapped_dim, ...)-th dimension.
    # 3. Expect positive num_chunks
    # 4. Expect non-empty input tensor list and each input tensor should have at least 1 element
    # 5. Non-contiguous input tensors are allowed.
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    same_ndim_cases = (
        (
            [
                torch.Size([1, 2, 3]),
                torch.Size([1, 2, 3]),
            ], -1, 5
        ),
        (
            [
                torch.Size([1, 2, 129]),
                torch.Size([1, 2, 297]),
            ], -1, 5
        ),
        (
            [
                torch.Size([1, 2, 3]),
                torch.Size([1, 2, 3]),
            ], 1, 5
        ),
        (
            [
                torch.Size([3, 3, 2, 1]),
                torch.Size([1, 4, 2, 2]),
                torch.Size([2, 1, 3, 3]),
            ], 0, 2
        ),
    )
    for sizes, dim, num_chunks in same_ndim_cases:
        tensors = [make_arg(size) for size in sizes]
        yield SampleInput(tensors, args=(dim, num_chunks))

    different_ndim_case = [
        torch.Size([2, 3, 3]),
        torch.Size([2, 3, 1, 2]),
        torch.Size([2, 3]),
        torch.Size([2, 3, 2]),
        torch.Size([2, 3, 271]),
    ]
    max_dim, num_chunks = 2, 3
    for dim in range(max_dim):
        tensors = []
        for size in different_ndim_case:
            tensors.append(make_arg(size))
        yield SampleInput(tensors, args=(dim, num_chunks))

    # non-contiguous
    for dim in range(max_dim):
        tensors = []
        for size in different_ndim_case:
            # make the last 2 dims column-major (i.e. non-contiguous)
            t = make_arg(size).transpose(-2, -1).contiguous().transpose(-2, -1)
            tensors.append(t)
        yield SampleInput(tensors, args=(dim, num_chunks))

def error_inputs_chunk_cat(op_info, device, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=torch.float32)

    # input tensors have different ndims but dim is negative
    sizes, dim, num_chunks = [torch.Size([2, 3]), torch.Size([4,])], -1, 3
    tensors = [make_arg(size) for size in sizes]
    yield ErrorInput(
        SampleInput(tensors, args=(dim, num_chunks)),
        error_regex='_chunk_cat expects non-negative dim when input tensors have different ndims',
    )

    # input tensors have different ndims but dim >= ndim of some input tensors
    sizes, dim, num_chunks = [torch.Size([2, 3]), torch.Size([4,])], 1, 3
    tensors = [make_arg(size) for size in sizes]
    yield ErrorInput(
        SampleInput(tensors, args=(dim, num_chunks)),
        error_regex='_chunk_cat expects dim < ndim for all input tensors',
    )

    # some tensors have different sizes for 0, ..., dim-1 dimensions.
    sizes, dim, num_chunks = [torch.Size([2, 3, 4]), torch.Size([4, 3])], 1, 3
    tensors = [make_arg(size) for size in sizes]
    yield ErrorInput(
        SampleInput(tensors, args=(dim, num_chunks)),
        error_regex='_chunk_cat expects same sizes of 0,...,dim-1 dimensions for all tensors',
    )

    # negative num_chunks
    sizes, dim, num_chunks = [torch.Size([2,]), torch.Size([3,])], 0, -1
    tensors = [make_arg(size) for size in sizes]
    yield ErrorInput(
        SampleInput(tensors, args=(dim, num_chunks)),
        error_regex='_chunk_cat expects positive num_chunks',
    )

    # zero as num_chunks
    sizes, dim, num_chunks = [torch.Size([2,]), torch.Size([3,])], 0, 0
    tensors = [make_arg(size) for size in sizes]
    yield ErrorInput(
        SampleInput(tensors, args=(dim, num_chunks)),
        error_regex='_chunk_cat expects positive num_chunks',
    )

    # empty input tensor list
    dim, num_chunks = 0, 1
    yield ErrorInput(
        SampleInput([], args=(dim, num_chunks)),
        error_regex='_chunk_cat expects a non-empty input tensor list',
    )

    # empty input tensor with 0 elements
    sizes, dim, num_chunks = [torch.Size([0,]), torch.Size([3,])], 0, 1
    tensors = [make_arg(size) for size in sizes]
    yield ErrorInput(
        SampleInput(tensors, args=(dim, num_chunks)),
        error_regex='_chunk_cat expects non-empty tensor',
    )


def sample_inputs_cat_concat(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases: tuple[tuple, tuple, dict] = (  # type: ignore[assignment]
        ((S, S), (S, S), {'dim': -1}),
        ((S, S), (S, S), {'dim': 1}),
        ((M, S), (S, S), {'dim': 0}),  # different shapes
        ((1, 2, 3), (1, 2, 3), {'dim': -2}),
        ((0,), (0,), {'dim': 0}),  # empty tensor
        ((0,), (S, S), {'dim': 1}),  # empty tensor with unempty and dim=1 (special case for legacy_cat_wrap_dim)
        ((0, S), (S, S), {'dim': 0}),
        ((1,), (1,), {})  # dim not passed, fallback to default
    )

    for input_shape1, input_shape2, kwargs in cases:
        yield SampleInput([make_arg(input_shape1), make_arg(input_shape2)], kwargs=kwargs)

    # from coat_lite_mini
    yield SampleInput([make_arg((2, 2, 2, 2), memory_format=torch.channels_last)], args=(1,),)

def error_inputs_cat(op_info, device, **kwargs):

    make_arg = partial(make_tensor, device=device, dtype=torch.float32)

    # error inputs for more than one element of the written-to tensor refer to a single memory location
    yield ErrorInput(SampleInput([make_arg((S, S)), make_arg((S, S))],
                                 kwargs={'out': make_arg((1, S)).expand((2 * S, S))}),
                     error_regex='unsupported operation')

    # error inputs for empty tensors
    yield ErrorInput(SampleInput([], kwargs={'dim': 1}),
                     error_regex='non-empty list of Tensors', error_type=ValueError)

    # error inputs for different sizes
    yield ErrorInput(SampleInput([make_arg((S, S, L, L)), make_arg((S, 0, L - 1, L))], kwargs={'dim': 1}),
                     error_regex='Sizes of tensors must match except in dimension')
    yield ErrorInput(SampleInput([make_arg((S, 0, L - 1, L)), make_arg((S, S, L, L))], kwargs={'dim': 1}),
                     error_regex='Sizes of tensors must match except in dimension')

    # error inputs for different dimensions
    yield ErrorInput(SampleInput([make_arg((S - 1, 0)), make_arg((S, 0, L - 1, L))], kwargs={'dim': 1}),
                     error_regex='Tensors must have same number of dimensions')
    yield ErrorInpu

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 10 class(es): _TestParamsMaxPoolBase, _TestParamsMaxPool1d, _TestParamsMaxPool2d, _TestParamsMaxPool3d, ForeachRightmostArgType, ForeachSampleInput, foreach_inputs_sample_func, foreach_max_sample_func, foreach_norm_sample_func, foreach_pointwise_sample_func

### Functions
This file defines 570 function(s): round_up, close_to_int, sample_inputs_slice, sample_inputs_tensor_split, sample_inputs_hsplit, sample_inputs_vsplit, sample_inputs_dsplit, error_inputs_hsplit, error_inputs_vsplit, error_inputs_dsplit, sample_inputs_as_strided, sample_inputs_as_strided_partial_views, make_arg, sample_inputs_as_strided_scatter, error_inputs_as_strided_scatter, sample_inputs_combinations, sample_inputs_cartesian_prod, sample_inputs_cosine_similarity, sample_inputs_item, error_inputs_item, sample_inputs_batch_norm, sample_inputs_softmax_backward_data, sample_inputs_native_batch_norm, sample_inputs__native_batch_norm_legit, sample_inputs__batch_norm_with_update, sample_inputs_nn_activation_relu, sample_inputs_prelu, reference_inputs_prelu, sample_kwargs_prelu_scalar_weight, error_inputs_prelu


## Key Components

The file contains 78075 words across 25187 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 1217342 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
