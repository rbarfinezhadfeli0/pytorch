# Documentation: `test/test_reductions.py`

## File Metadata

- **Path**: `test/test_reductions.py`
- **Size**: 179,273 bytes (175.07 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```python
# Owner(s): ["module: tests"]

import contextlib
import torch
import numpy as np

import math
from collections.abc import Sequence
import random
from functools import partial
from itertools import product, combinations, permutations
import warnings

from torch import inf, nan
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import (
    all_types_and_complex_and, get_all_math_dtypes, integral_types, complex_types, floating_types_and,
    integral_types_and, floating_and_complex_types_and, all_types_and, all_types,
)
from torch.testing._internal.common_utils import (
    TestCase, run_tests, skipIfNoSciPy, slowTest, torch_to_numpy_dtype_dict,
    parametrize,
    skipIfTorchDynamo,
    IS_WINDOWS)
from torch.testing._internal.common_device_type import (
    OpDTypes, expectedFailureMeta, instantiate_device_type_tests, onlyCPU, dtypes, dtypesIfCUDA, dtypesIfCPU,
    onlyNativeDeviceTypes, onlyCUDA, largeTensorTest, ops, precisionOverride)
from torch.testing._internal.common_methods_invocations import (
    ReductionOpInfo, ReductionPythonRefInfo, reduction_ops, reference_masked_ops)

# TODO: replace with make_tensor
def _generate_input(shape, dtype, device, with_extremal):
    if shape == ():
        x = torch.tensor((), dtype=dtype, device=device)
    else:
        if dtype.is_floating_point or dtype.is_complex:
            # work around torch.randn not being implemented for bfloat16
            if dtype == torch.bfloat16:
                x = torch.randn(*shape, device=device) * random.randint(30, 100)
                x = x.to(torch.bfloat16)
            else:
                x = torch.randn(*shape, dtype=dtype, device=device) * random.randint(30, 100)
            x[torch.randn(*shape) > 0.5] = 0
            if with_extremal and dtype.is_floating_point:
                # Use extremal values
                x[torch.randn(*shape) > 0.5] = float('nan')
                x[torch.randn(*shape) > 0.5] = float('inf')
                x[torch.randn(*shape) > 0.5] = float('-inf')
            elif with_extremal and dtype.is_complex:
                x[torch.randn(*shape) > 0.5] = complex('nan')
                x[torch.randn(*shape) > 0.5] = complex('inf')
                x[torch.randn(*shape) > 0.5] = complex('-inf')
        elif dtype == torch.bool:
            x = torch.zeros(shape, dtype=dtype, device=device)
            x[torch.randn(*shape) > 0.5] = True
        else:
            x = torch.randint(15, 100, shape, dtype=dtype, device=device)

    return x

# TODO: replace with make_tensor
def _rand_shape(dim, min_size, max_size):
    shape = []
    for _ in range(dim):
        shape.append(random.randint(min_size, max_size))
    return tuple(shape)

def _reduced_shape(shape, empty_dim_as_none=False, dim=None, keepdim=False):
    """Computes the expected reduced shape given dim and keepdim

    Args:
        shape: The shape to reduce
        dim : The dimensions to reduce
        keepdim: If true, reduced dimensions have size 1 in the reduced shape,
            otherwise they are removed from the reduced shape.

    Returns:
        The reduced shape
    """
    if dim is None or (empty_dim_as_none and dim == []):
        return [1] * len(shape) if keepdim else []

    # Wrap negative dims
    dim = dim if isinstance(dim, Sequence) else [dim]
    dim = {i if i >= 0 else len(shape) + i for i in dim}

    result = []
    for i, size in enumerate(shape):
        if i not in dim:
            result.append(size)
        elif keepdim:
            result.append(1)

    return result

class TestReductions(TestCase):

    ###########################################################################
    # ReductionOpInfo unit tests
    ###########################################################################

    def _test_dim_keepdim(self, op: ReductionOpInfo, device, *, ndim, **dim_keepdim):
        """Tests output shape for input with ndim and dim and keepdim kwargs"""
        shape = torch.randint(2, 5, (ndim,)).tolist()
        t = make_tensor(shape, dtype=torch.float, device=device)
        args, kwargs = next(op.generate_args_kwargs(t, **dim_keepdim))
        result = op(t, *args, **dim_keepdim, **kwargs)
        empty_dim_as_none = (op.name == "linalg.vector_norm" or op.name == "_refs.linalg.vector_norm")
        expected_shape = _reduced_shape(shape, empty_dim_as_none, **dim_keepdim)
        self.assertEqual(result.shape, expected_shape, f"""
        expected output shape to be {expected_shape} but got {list(result.shape)}
        for input shape {shape} and {dim_keepdim}
        """)

    # TODO(@heitorschueroff) combine cases with and without keepdim once
    # there's support for a @parametrize decorator.

    @ops(reduction_ops, dtypes=OpDTypes.none)
    def test_dim_default(self, device, op: ReductionOpInfo):
        """Tests that the default dim reduces all dimensions."""
        for ndim in range(3):
            self._test_dim_keepdim(op, device, ndim=ndim)

    @ops(reduction_ops, dtypes=OpDTypes.none)
    def test_dim_default_keepdim(self, device, op: ReductionOpInfo):
        """Tests that the default dim, when keepdim=True, reduces all dimensions to size 1."""
        for ndim in range(3):
            self._test_dim_keepdim(op, device, ndim=ndim, keepdim=True)

    @ops(reduction_ops, dtypes=OpDTypes.none)
    def test_dim_none(self, device, op: ReductionOpInfo):
        """Tests that dim=None reduces all dimensions."""
        for ndim in range(3):
            self._test_dim_keepdim(op, device, ndim=ndim, dim=None)

    @ops(reduction_ops, dtypes=OpDTypes.none)
    def test_dim_none_keepdim(self, device, op: ReductionOpInfo):
        """Tests that dim=None, when keepdim=True, reduces all dimensions to size 1."""
        for ndim in range(3):
            self._test_dim_keepdim(op, device, ndim=ndim, dim=None, keepdim=True)

    @ops(reduction_ops, dtypes=OpDTypes.none)
    def test_dim_single(self, device, op: ReductionOpInfo):
        """Tests that dim=i reduces dimension i."""
        self._test_dim_keepdim(op, device, ndim=0, dim=0)
        self._test_dim_keepdim(op, device, ndim=1, dim=0)
        self._test_dim_keepdim(op, device, ndim=2, dim=-1)
        self._test_dim_keepdim(op, device, ndim=3, dim=1)

    @ops(reduction_ops, dtypes=OpDTypes.none)
    def test_dim_single_keepdim(self, device, op: ReductionOpInfo):
        """Tests that dim=i, when keepdim=True, reduces dimension i to size 1."""
        self._test_dim_keepdim(op, device, ndim=0, dim=0, keepdim=True)
        self._test_dim_keepdim(op, device, ndim=1, dim=0, keepdim=True)
        self._test_dim_keepdim(op, device, ndim=2, dim=-1, keepdim=True)
        self._test_dim_keepdim(op, device, ndim=3, dim=1, keepdim=True)

    @ops(filter(lambda op: op.supports_multiple_dims, reduction_ops), dtypes=OpDTypes.none)
    def test_dim_empty(self, device, op: ReductionOpInfo):
        """Tests that dim=[] is a no-op"""
        self._test_dim_keepdim(op, device, ndim=0, dim=[])
        self._test_dim_keepdim(op, device, ndim=2, dim=[])

    @ops(filter(lambda op: op.supports_multiple_dims, reduction_ops), dtypes=OpDTypes.none)
    def test_dim_empty_keepdim(self, device, op: ReductionOpInfo):
        """Tests that dim=[], when keepdim=True, is a no-op"""
        self._test_dim_keepdim(op, device, ndim=0, dim=[], keepdim=True)
        self._test_dim_keepdim(op, device, ndim=2, dim=[], keepdim=True)

    @ops(filter(lambda op: op.supports_multiple_dims, reduction_ops), dtypes=OpDTypes.none)
    def test_dim_multi(self, device, op: ReductionOpInfo):
        """Tests that dim=[i, j, ...] reduces dimensions i, j, ...."""
        self._test_dim_keepdim(op, device, ndim=1, dim=[0])
        self._test_dim_keepdim(op, device, ndim=3, dim=[0, 2])

    @ops(filter(lambda op: op.supports_multiple_dims, reduction_ops), dtypes=OpDTypes.none)
    def test_dim_multi_keepdim(self, device, op: ReductionOpInfo):
        """Tests that dim=[i, j, ...], when keepdim=True, reduces dimensions i, j, .... to size 1."""
        self._test_dim_keepdim(op, device, ndim=1, dim=[0], keepdim=True)
        self._test_dim_keepdim(op, device, ndim=3, dim=[0, 2], keepdim=True)

    @ops(filter(lambda op: op.supports_multiple_dims, reduction_ops), dtypes=OpDTypes.none)
    def test_dim_multi_unsorted(self, device, op: ReductionOpInfo):
        """Tests that operator correctly handles unsorted dim list."""
        self._test_dim_keepdim(op, device, ndim=4, dim=[3, 0, 2])

    @ops(filter(lambda op: op.supports_multiple_dims, reduction_ops), dtypes=OpDTypes.none)
    def test_dim_multi_unsorted_keepdim(self, device, op: ReductionOpInfo):
        """Tests that operator correctly handles unsorted dim list when keepdim=True."""
        self._test_dim_keepdim(op, device, ndim=4, dim=[3, 0, 2], keepdim=True)

    @ops(filter(lambda op: op.supports_multiple_dims, reduction_ops), dtypes=OpDTypes.none)
    def test_dim_multi_duplicate(self, device, op: ReductionOpInfo):
        """Tests that an error is raised if dim has duplicate entries."""
        with self.assertRaises(RuntimeError):
            self._test_dim_keepdim(op, device, ndim=3, dim=[0, 1, 1, 2])

    @ops(filter(lambda op: not op.supports_multiple_dims, reduction_ops), dtypes=OpDTypes.none)
    def test_dim_multi_unsupported(self, device, op: ReductionOpInfo):
        """Tests that ops claiming to not support multi dim actually don't."""
        with self.assertRaises(TypeError):
            self._test_dim_keepdim(op, device, ndim=3, dim=[0, 2])

    @ops(reduction_ops, dtypes=OpDTypes.none)
    def test_dim_offbounds(self, device, op: ReductionOpInfo):
        """Tests that passing an off-bounds dim throws"""
        with self.assertRaises(IndexError):
            self._test_dim_keepdim(op, device, ndim=2, dim=2)

    @ops(reduction_ops, dtypes=OpDTypes.none)
    def test_dim_ndim_limit(self, device, op: ReductionOpInfo):
        """Tests that an exception is raised when reducing a tensor with more
        than 64 dims along some specific dimensions. dim=None is ok"""
        t = make_tensor([1] * 65, dtype=torch.float, device=device)
        with self.assertRaisesRegex(RuntimeError, "only tensors with up to 64 dims are supported"):
            op(t, dim=0)

    @ops(filter(lambda op: op.identity is not None, reduction_ops), dtypes=OpDTypes.supported)
    def test_identity(self, device, dtype, op: ReductionOpInfo):
        """Tests that the identity value is an identity for the operator"""
        t = make_tensor((10,), dtype=dtype, device=device)
        t[1::2] = op.identity
        args, kwargs = next(op.generate_args_kwargs(t))
        result = op(t[::2], *args, **kwargs)
        result_with_identity = op(t, *args, **kwargs)
        self.assertEqual(result, result_with_identity, """
        Adding identity value to the input tensor should not change the result.
        """)

    # TODO(@heitorschueroff) Update these to use the nan_policy kwarg once
    # it is added to reduction operators.

    @ops(filter(lambda op: op.nan_policy == 'propagate', reduction_ops), dtypes=OpDTypes.supported,
         allowed_dtypes=floating_and_complex_types_and(torch.bfloat16, torch.float16))
    def test_nan_policy_propagate(self, device, dtype, op: ReductionOpInfo):
        """Tests that nan is propagated to the output by default"""
        t = make_tensor((5,), dtype=dtype, device=device)
        t[2] = torch.nan
        args, kwargs = next(op.generate_args_kwargs(t))
        result = op(t, *args, **kwargs)
        self.assertTrue(result.isnan())

    @ops(filter(lambda op: op.nan_policy == 'omit', reduction_ops), dtypes=OpDTypes.supported,
         allowed_dtypes=floating_and_complex_types_and(torch.bfloat16, torch.float16))
    def test_nan_policy_omit(self, device, dtype, op: ReductionOpInfo):
        """Tests that NaN values do not affect the result."""
        t = make_tensor((10,), dtype=dtype, device=device)
        t[1::2] = torch.nan
        args, kwargs = next(op.generate_args_kwargs(t))
        result = op(t[::2], *args, **kwargs)
        result_with_nan = op(t, *args, **kwargs)
        self.assertEqual(result, result_with_nan)

    @ops(reduction_ops, dtypes=OpDTypes.supported)
    def test_result_dtype(self, device, dtype, op: ReductionOpInfo):
        """Tests that the result has the correct dtype"""
        t = make_tensor((5,), dtype=dtype, device=device)
        args, kwargs = next(op.generate_args_kwargs(t))
        result: torch.Tensor = op(t, *args, **kwargs)
        is_integral = dtype in integral_types_and(torch.bool)
        if op.promotes_int_to_float and is_integral:
            self.assertTrue(torch.is_floating_point(result))
        elif op.promotes_int_to_int64 and is_integral:
            self.assertEqual(result.dtype, torch.int64)
        elif op.result_dtype is not None:
            self.assertEqual(result.dtype, op.result_dtype)
        elif op.complex_to_real:
            _complex_to_real_dtype_map = {
                torch.complex128: torch.float64,
                torch.complex64: torch.float32,
                torch.complex32: torch.float16,
            }
            self.assertEqual(result.dtype, _complex_to_real_dtype_map.get(dtype, dtype))
        else:
            self.assertEqual(result.dtype, dtype)

    @ops(reduction_ops, dtypes=OpDTypes.none)
    def test_empty_tensor_empty_slice(self, device, op: ReductionOpInfo):
        """Tests for consistent behavior when reducing over an empty slice.

        The rules for reducing over an empty slice are as follows:
            - Return the identity value if the operator has one
            - Otherwise, return NaN if the operator promotes integral dtype to
              floating point dtypes.
            - Otherwise, raise an error

        See discussion here https://github.com/pytorch/pytorch/issues/61901
        """
        t = make_tensor((0, 2, 3), dtype=torch.float, device=device)
        for dim in [0] + [[0, 2]] if op.supports_multiple_dims else []:
            args, kwargs = next(op.generate_args_kwargs(t, dim=dim))
            if op.identity is not None:
                # Reducing along empty slice should return identity
                result = op(t, *args, dim=dim, **kwargs)
                self.assertEqual(result, torch.full_like(result, op.identity))
            elif op.promotes_int_to_float:
                # Reducing along empty slice should return NaN
                result = op(t, *args, dim=dim, **kwargs)
                self.assertEqual(result, torch.full_like(result, torch.nan))
            else:
                # Reducing along empty slice should raise an error
                if isinstance(op, ReductionPythonRefInfo):
                    # ref reductions throw RuntimeError for this
                    with self.assertRaises(RuntimeError):
                        op(t, *args, dim=dim, **kwargs)
                else:
                    with self.assertRaises(IndexError):
                        op(t, *args, dim=dim, **kwargs)

    @ops(reduction_ops, dtypes=OpDTypes.none)
    def test_empty_tensor_nonempty_slice(self, device, op: ReductionOpInfo):
        """Tests that reducing a nonempty slice of an empty tensor returns an
        empty tensor with the dimensions reduced."""
        t = make_tensor((0, 2, 3), dtype=torch.float, device=device)
        for dim in [1] + [[1, 2]] if op.supports_multiple_dims else []:
            args, kwargs = next(op.generate_args_kwargs(t, dim=dim))
            result = op(t, *args, dim=dim, **kwargs)
            self.assertEqual(result.shape, _reduced_shape(t.shape, dim=dim))

    def _test_noncontiguous(self, op: ReductionOpInfo, t: torch.Tensor, **reduction_kwargs):
        """Helper method to test noncontiguous input tensors."""
        assert not t.is_contiguous()

        t_contig = t.contiguous()
        for args, kwargs in op.generate_args_kwargs(t_contig, **reduction_kwargs):
            kwargs.update(reduction_kwargs)
            result = op(t, *args, **kwargs)
            expected = op(t_contig, *args, **kwargs)
            self.assertEqual(result, expected)

    @ops(reduction_ops)
    def test_noncontiguous_innermost(self, device, dtype, op: ReductionOpInfo):
        """Tests reducing along noncontiguous innermost dimension."""
        t = make_tensor((10, 10), dtype=dtype, device=device, low=-1, high=1)
        self._test_noncontiguous(op, t[:, ::2], dim=1)

    @ops(reduction_ops)
    def test_noncontiguous_outermost(self, device, dtype, op: ReductionOpInfo):
        """Tests reducing along noncontiguous outermost dimension."""
        t = make_tensor((10, 10), dtype=dtype, device=device, low=-1, high=1)
        self._test_noncontiguous(op, t[::2, :], dim=0)

    @ops(reduction_ops)
    def test_noncontiguous_all(self, device, dtype, op: ReductionOpInfo):
        """Tests reducing all dimensions of a noncontiguous tensor."""
        t = make_tensor((5, 5, 5), dtype=dtype, device=device, low=-1, high=1)
        self._test_noncontiguous(op, t[::2, ::3, 1:-1:2])

    @ops(reduction_ops)
    def test_noncontiguous_transposed(self, device, dtype, op: ReductionOpInfo):
        """Tests reducing a transposed tensor."""
        t = make_tensor((5, 5), dtype=dtype, device=device, low=-1, high=1)
        self._test_noncontiguous(op, t.T)

    @ops(reduction_ops)
    def test_noncontiguous_expanded(self, device, dtype, op: ReductionOpInfo):
        """Tests reducing a tensor with expanded singleton dimensions."""
        t = make_tensor((2, 3), dtype=dtype, device=device, low=-1, high=1)
        self._test_noncontiguous(op, t.unsqueeze(1).expand(-1, 5, -1))

    # NumPy does not support BFloat16 so we don't test that against reference
    # implementations. We also don't compare dtypes or test for different
    # keepdim because we already have other tests covering those.
    # The test_reference_testing in test_ops.py only uses the samples from
    # sample_inputs_func which do not test as exhaustively as these tests.

    def _test_ref(self, op: ReductionOpInfo, t: torch.Tensor, **reduction_kwargs):
        """Compares op against op.ref for the given input and reduction kwargs"""
        for args, kwargs in op.generate_args_kwargs(t, **reduction_kwargs):
            kwargs.update(reduction_kwargs)
            result = op(t, *args, **kwargs)
            expected = op.ref(t.detach().cpu().numpy(), *args, **kwargs)
            self.assertEqual(result, expected, exact_dtype=False)

    @ops(filter(lambda op: op.ref is not None, reduction_ops),
         allowed_dtypes=all_types_and_complex_and(torch.half, torch.bool))
    def test_ref_scalar_input(self, device, dtype, op: ReductionOpInfo):
        """Compares op against reference for scalar input tensors"""
        self._test_ref(op, make_tensor([], dtype=dtype, device=device))

    @ops(filter(lambda op: op.ref is not None, reduction_ops),
         allowed_dtypes=all_types_and_complex_and(torch.half, torch.bool))
    def test_ref_small_input(self, device, dtype, op: ReductionOpInfo):
        """Compares op against reference for small input tensors"""
        t = make_tensor((5, 3, 4, 2), dtype=dtype, device=device, low=-2, high=2, exclude_zero=True)
        self._test_ref(op, t)
        for dim in [0, 1, 3] + ([[0, 2], [1, 3]] if op.supports_multiple_dims else []):
            self._test_ref(op, t, dim=dim)

    @ops(filter(lambda op: op.ref is not None, reduction_ops),
         allowed_dtypes=[torch.float64])
    def test_ref_large_input_1D(self, device, dtype, op: ReductionOpInfo):
        """Compares op against reference for a large 1D input tensor to check stability"""
        self._test_ref(op, make_tensor((2 ** 20,), dtype=dtype, device=device, low=-1, high=1, exclude_zero=True))

    @ops(filter(lambda op: op.ref is not None, reduction_ops),
         allowed_dtypes=[torch.float64])
    def test_ref_large_input_2D(self, device, dtype, op: ReductionOpInfo):
        """Compares op against reference for a large 2D input tensor to test parallelism"""
        t = make_tensor((32, 2 ** 16), dtype=dtype, device=device, low=-1, high=1, exclude_zero=True)
        self._test_ref(op, t, dim=1)

    @largeTensorTest("8gb")
    @ops(filter(lambda op: op.ref is not None, reduction_ops),
         allowed_dtypes=[torch.float64])
    def test_ref_large_input_64bit_indexing(self, device, dtype, op: ReductionOpInfo):
        """Compares op against reference for a very large input tensor that requires 64 bit indexing"""
        self._test_ref(op, make_tensor((275000000,), dtype=dtype, device=device, low=-1, high=1, exclude_zero=True))

    @ops(filter(lambda op: op.ref is not None, reduction_ops),
         allowed_dtypes=all_types_and_complex_and(torch.half, torch.bool))
    def test_ref_duplicate_values(self, device, dtype, op: ReductionOpInfo):
        """Compares op against reference for input tensors with duplicate values"""
        t = make_tensor((4, 4), dtype=dtype, device=device, low=-2, high=2, exclude_zero=True)
        t[::2, ::2] = t[1::2, 1::2]
        self._test_ref(op, t)
        self._test_ref(op, t, dim=0)
        self._test_ref(op, t, dim=1)

    @ops(filter(lambda op: op.ref is not None, reduction_ops),
         allowed_dtypes=[torch.float32, torch.complex64])
    def test_ref_extremal_values(self, device, dtype, op: ReductionOpInfo):
        """Compares op against reference for input tensors with extremal values"""
        t = make_tensor((5,), dtype=dtype, device=device, exclude_zero=True)
        extremals = [0, 1, nan, inf, -inf]
        for extremal in extremals:
            t[2] = extremal
            self._test_ref(op, t)

    ###########################################################################
    # TODO: Legacy tests - port to ReductionOpInfo
    ###########################################################################

    def test_var_unbiased(self, device):
        tensor = torch.randn(100, device=device)
        self.assertEqual(tensor.var(0), tensor.var(0, unbiased=True))
        self.assertEqual(tensor.var(), tensor.var(unbiased=True))
        self.assertEqual(tensor.var(unbiased=False), tensor.var(0, unbiased=False))

        tensor = torch.tensor([1.0, 2.0], device=device)
        self.assertEqual(tensor.var(unbiased=True), 0.5)
        self.assertEqual(tensor.var(unbiased=False), 0.25)

        tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
        self.assertEqual(tensor.var(unbiased=True), 1.0)
        self.assertEqual(tensor.var(unbiased=False), 2.0 / 3.0)

        tensor = torch.randn(100, device=device)
        self.assertEqual(tensor.std(0), tensor.std(0, unbiased=True))
        self.assertEqual(tensor.std(), tensor.std(unbiased=True))
        self.assertEqual(tensor.std(unbiased=False), tensor.std(0, unbiased=False))

    def test_var_stability(self, device):
        tensor = torch.tensor([2281.5, 2281.25], device=device)
        self.assertEqual(tensor.var(dim=0), 0.03125)
        self.assertEqual(tensor.var(), 0.03125)

    def test_sum_dim_reduction_uint8_overflow(self, device):
        example = [[-1, 2, 1], [5, 3, 6]]
        x = torch.tensor(example, dtype=torch.uint8, device=device)
        self.assertEqual(x.sum(dtype=torch.uint8).item(), 16)
        self.assertEqual(x.sum(0, dtype=torch.uint8), torch.tensor([4, 5, 7], dtype=torch.uint8, device=device))
        self.assertEqual(x.sum(1, dtype=torch.uint8), torch.tensor([2, 14], dtype=torch.uint8, device=device))
        y = torch.tensor(example, dtype=torch.uint8, device=device)
        torch.sum(x, 0, out=y)
        self.assertEqual(x.sum(0, dtype=torch.uint8), y)

    def test_dim_reduction_less_than_64(self, device):
        sizes = [1] * 65
        x = torch.randn(sizes, device=device)
        ops = [torch.mean, torch.sum, torch.nansum, torch.std, torch.logsumexp, torch.std, torch.var,
               torch.norm]
        for op in ops:
            with self.assertRaisesRegex(RuntimeError, "only tensors with up to 64 dims are supported"):
                op(x, dim=64)
            with self.assertRaisesRegex(RuntimeError, "only tensors with up to 64 dims are supported"):
                op(x, dim=-1)

    @onlyCPU
    @dtypes(torch.float, torch.bfloat16)
    def test_dim_reduction_lastdim(self, device, dtype):
        x = torch.randn(3, 5, 40, device=device, dtype=dtype)
        x = x[:, :, 0:40:2]
        x2 = x.contiguous()
        ops = [torch.norm, torch.argmax, torch.argmin]
        for op in ops:
            y = op(x, dim=-1)
            y2 = op(x2, dim=-1)
            self.assertEqual(y, y2)

    @skipIfNoSciPy
    @dtypes(torch.float32, torch.double, torch.complex64, torch.complex128)
    def test_logsumexp(self, device, dtype):
        from scipy.special import logsumexp
        a = torch.randn(5, 4, device=device, dtype=dtype)
        # torch.exp(complex(inf, 0)) yields inf+nan*j instead of inf+0*j on CPU which disagrees with CUDA, C++ std::exp,
        # numpy and scipy. Skip inf testing on CPU. Related to https://github.com/pytorch/pytorch/issues/95740
        if torch.device(device) != torch.device('cpu'):
            a[0, 0] = inf
        a[1, :] = -inf
        actual = a.logsumexp(1)
        expected = logsumexp(a.cpu().numpy(), 1)
        self.assertEqual(expected.shape, actual.shape)
        self.assertEqual(expected, actual)

        # check that out is actually inplace
        b = torch.zeros(5, 2, device=device, dtype=dtype)
        c = b[:, 0]
        torch.logsumexp(a, 1, out=c)
        self.assertEqual(expected, b[:, 0])

    @skipIfNoSciPy
    def test_logsumexp_integral_promotion(self, device):
        from scipy.special import logsumexp
        # check integral inputs is promoted to floating point
        e = torch.randint(-100, 100, [5, 4], device=device)
        actual = e.logsumexp(1).to(torch.float64)
        expected = logsumexp(e.cpu().numpy(), 1)
        self.assertEqual(expected.shape, actual.shape)
        self.assertEqual(expected, actual)

    @skipIfNoSciPy
    @dtypes(torch.complex64, torch.complex128)
    def test_logcumsumexp_complex(self, device, dtype):
        # logcumsumexp is a more precise way to compute than ``log(cumsum(exp(a)))``
        # and faster than ``[log(sum(exp(a[:i]))) for i in range(a.shape[0])]``
        # the for-loop above should produce similar precision as logcumsumexp (it's just slower),
        # so it can be used as the expected values to check our computation

        # using logsumexp from scipy because by the time of writing this test code,
        # torch.logsumexp has not been implemented for complex numbers
        from scipy.special import logsumexp

        def zero_out_neg_inf(t):
            t = t.clone()
            idx = torch.logical_and(~(torch.isfinite(t)), torch.real(t) < 0)
            t[idx] = torch.real(t[idx]).to(t.dtype)
            return t

        def standardize_phase(t):
            t = torch.real(t) + 1j * (torch.imag(t) % (2 * np.pi))
            return t

        def logcumsumexp_slow(a, dim):
            res_lst = []
            for i in range(a.size(dim)):
                index = [slice(None, None, None) for _ in range(a.ndim)]
                index[dim] = slice(None, i + 1, None)
                a_inp = a[tuple(index)]
                res_lst.append(logsumexp(a_inp.cpu().numpy(), axis=dim, keepdims=True))
            res = np.concatenate(res_lst, axis=dim)
            return torch.as_tensor(res)

        def compare_logcumsumexp(a, expected=None):
            for i in range(a.ndim):
                actual = torch.logcumsumexp(a, dim=i)
                # if the expected is not given, then revert to scipy's logsumexp
                if expected is None:
                    expected2 = logcumsumexp_slow(a, dim=i)
                else:
                    expected2 = expected

                # move the imaginary values to (0, 2 * pi)
                actual = standardize_phase(actual)
                expected2 = standardize_phase(expected2)

                # zeroing the imaginary part of the element if the real part is -inf
                # as the imaginary part cannot be determined exactly and it does not
                # really matter if we take the exp of the output
                actual = zero_out_neg_inf(actual)
                expected2 = zero_out_neg_inf(expected2)
                self.assertEqual(expected2.shape, actual.shape)
                self.assertEqual(expected2, actual)

        # randomly specified values
        # in this case, scipy.logsumexp should be enough
        a1 = torch.randn((5, 10), dtype=dtype, device=device)
        compare_logcumsumexp(a1)

        # test with some non-normal values
        a2 = torch.tensor([1e3 + 0j, 1e-18 + 1e4j, 1e2 + 1e-8j], dtype=dtype, device=device)
        compare_logcumsumexp(a2)

        # handle special case involving infinites and nans
        # here we don't use scipy.logsumexp as it gives confusing answer on
        # some inf cases
        # see here:
        inf = float('inf')
        nan = float('nan')
        a3_input = torch.tensor([
            -inf + 4j,
            -inf + 1j,
            1.2 + 2.1j,
            1e10 + 1e20j,
            inf + 0j,
            inf + 1j,
            inf + 3j,
            nan + 2j,
        ])
        a3_expected = torch.tensor([
            -inf + 0j,
            -inf + 0j,
            1.2 + 2.1j,
            1e10 + 1e20j,
            inf + 0j,  # scipy's logsumexp gives (inf + 0.7853982j) here, unclear why
            inf + (np.pi / 4) * 1j,  # the imaginary part thanks to some weird behaviour of log(inf + infj)
            complex(inf, nan),
            complex(nan, nan),
        ])
        # windows give strange results on the second-to-last results where it gives inf + pi/4 j
        # instead of inf + nan j
        if not IS_WINDOWS:
            compare_logcumsumexp(a3_input, a3_expected)

        a4_input = torch.tensor([
            complex(-inf, inf),
            complex(-inf, inf),
            -inf + 1j,
            1.2 + 2.1j,
            complex(2.4, inf),
        ])
        a4_expected = torch.tensor([
            -inf + 0j,
            -inf + 0j,
            -inf + 0j,
            1.2 + 2.1j,
            complex(nan, nan),
        ])
        if not IS_WINDOWS:
            compare_logcumsumexp(a4_input, a4_expected)

    @onlyCPU
    def test_sum_parallel(self, device):
        # To use parallel branches we'll need to compare on tensors
        # that are relatively large. Even if this is run on a single
        # core machine these tests will still give you signal on
        # the correctness

        def _run_test(size):
            for dim in range(len(size) + 1):
                nv = np.round(np.random.rand(*size))  # 0s and 1s
                tv = torch.from_numpy(nv)
                # Parallelisim is only used if numel is
                # larger than grainsize defined in Parallel.h
                self.assertTrue(tv.numel() > 32768)
                if dim == len(size):
                    nvs = nv.sum()
                    tvs = tv.sum()
                else:
                    nvs = nv.sum(dim)
                    tvs = tv.sum(dim)
                diff = np.abs(nvs - tvs.numpy()).sum()
                self.assertEqual(diff, 0)

        _run_test([2, 3, 3, 3, 3, 2, 2, 3, 2, 3, 2, 3, 3])
        _run_test([4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        _run_test([1, 32 * 8 * 32 * 8])
        _run_test([1, 32770])

    # TODO: kill map2_ (and similar) uses and update to compare with NumPy
    # only works on CPU since this uses map2_, which is only supported on CPU
    def _testCSelection(self, torchfn, mathfn):
        # Two tensors
        size = (100, 100)
        a = torch.rand(*size)
        b = torch.rand(*size)
        c = torchfn(a, b)
        expected_c = torch.zeros(*size)
        expected_c.map2_(a, b, lambda _, a, b: mathfn(a, b))
        self.assertEqual(expected_c, c, atol=0, rtol=0)

    @onlyCPU
    def test_max_elementwise(self, device):
        self._testCSelection(torch.max, max)

    @onlyCPU
    def test_min_elementwise(self, device):
        self._testCSelection(torch.min, min)

    def test_all_any(self, device):
        def test(size):
            x = torch.ones(*size, device=device).byte()
            self.assertTrue(x.all())
            self.assertTrue(x.any())

            x[3] = 0
            self.assertFalse(x.all())
            self.assertTrue(x.any())

            x.zero_()
            self.assertFalse(x.all())
            self.assertFalse(x.any())

            x.fill_(2)
            self.assertTrue(x.all())
            self.assertTrue(x.any())

            x = torch.ones(*size, device=device).bool()
            self.assertTrue(x.all())
            self.assertTrue(x.any())

            x[3] = False
            self.assertFalse(x.all())
            self.assertTrue(x.any())

        test((10,))
        test((5, 5))

    def test_all_any_with_dim(self, device):
        def test(x):
            r1 = x.prod(dim=0, keepdim=False).byte()
            r2 = x.all(dim=0, keepdim=False)
            self.assertEqual(r1.shape, r2.shape)
            self.assertTrue((r1 == r2).all())

            r3 = x.sum(dim=1, keepdim=True).clamp(0, 1).byte()
            r4 = x.any(dim=1, keepdim=True)
            self.assertEqual(r3.shape, r4.shape)
            self.assertTrue((r3 == r4).all())

        test(torch.tensor([[0, 0, 0],
                           [0, 0, 1],
                           [0, 1, 1],
                           [1, 1, 1]], device=device, dtype=torch.uint8))

    def test_numpy_named_args(self, device):
        x1 = torch.randn(10, device=device)
        x2 = torch.randn(10, device=device)
        res1 = torch.add(input=x1, other=x2)
        res2 = torch.add(x1=x1, x2=x2)
        self.assertEqual(res1, res2)

        x1 = torch.randn(10, 10, 10, device=device)
        res1 = x1.sum(dim=(0, 2), keepdim=True)
        res2 = x1.sum(axis=(0, 2), keepdims=True)
        self.assertEqual(res1, res2)

    # TODO: kill this and replace with common creation ops
    def _make_tensors(self, shape, val_range=(-100, 100), use_floating=True, use_integral=True,
                      use_complex=False) -> dict[str, list[torch.Tensor]]:
        float_types = [torch.double,
                       torch.float]
        int_types = [torch.int64,
                     torch.int32,
                     torch.int16]

        complex_types = [torch.complex64,
                         torch.complex128]

        def make_contiguous(shape, dtype) -> torch.Tensor:
            if dtype in float_types:
                val = torch.randn(shape, dtype=dtype)
                val = val * ((val_range[1] - val_range[0]) / (math.pi * 2.0))
                val = val + ((val_range[1] - val_range[0]) / 2.0)
                val = torch.clamp(val, min=val_range[0], max=val_range[1])
                return val
            result = torch.zeros(shape, dtype=dtype)
            result.apply_(lambda x: random.randint(val_range[0], val_range[1]))
            return result

        def make_non_contiguous(shape, dtype) -> torch.Tensor:
            contig = make_contiguous(shape, dtype)
            non_contig = torch.empty(shape + (2, 2), dtype=dtype)[..., 0]
            non_contig = non_contig.select(-1, -1)
            non_contig.copy_(contig)
            self.assertFalse(non_contig.is_contiguous())
            return non_contig

        def make_contiguous_slice(size, dtype) -> torch.Tensor:
            contig = make_contiguous((1, size), dtype)
            non_contig = contig[:1, 1:size - 1]
            self.assertTrue(non_contig.is_contiguous())
            return contig

        types = []
        if use_floating:
            types += float_types
        if use_integral:
            types += int_types
        if use_complex:
            types += complex_types
        tensors: dict[str, list[torch.Tensor]] = {"cont": [], "noncont": [], "slice": []}
        for dtype in types:
            tensors["cont"].append(make_contiguous(shape, dtype))
            tensors["noncont"].append(make_non_contiguous(shape, dtype))
            tensors["slice"].append(make_contiguous_slice(sum(list(shape)), dtype))

        return tensors

    # TODO: refactor this to use comparators from common_utils
    def _assert_matches_numpy(self, t, n):
        self.assertEqual(n.shape, t.shape)
        if t.dtype == torch.float:
            self.assertEqual(n, t, rtol=1e-03, atol=1e-05, equal_nan=True)
        else:
            self.assertEqual(n, t, equal_nan=True)

    # TODO: update this and tests that use it to use the device argument properly
    def _test_dim_ops(self, pytorch_op, numpy_op,
                      use_floating=True, use_integral=True, use_complex=False):
        def do_one(tensors_dict, dim):
            for category, tensors in tensors_dict.items():
                if category == "slice":
                    dim = 0
                for tensor in tensors:
                    # we have no control over NumPy warnings...
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        expected = numpy_op(tensor.cpu().numpy(), dim)
                    actual = pytorch_op(tensor, dim)
                    self._assert_matches_numpy(actual, expected)
                    if torch.cuda.is_available():
                        self._assert_matches_numpy(pytorch_op(tensor.cuda(), dim).cpu(), expected)
        do_one(self._make_tensors((5, 400000), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 1)
        do_one(self._make_tensors((3, 5, 7), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 0)
        do_one(self._make_tensors((3, 5, 7), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 1)
        do_one(self._make_tensors((3, 5, 7), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 2)
        do_one(self._make_tensors((100000, ), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), -1)
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 0)
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 1)
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 2)
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), (1, 2))
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), (1, -1))
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), (0, 2))
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), (0, 2, 1))

    @slowTest
    @onlyCPU
    def test_sum_dim(self, device):
        self._test_dim_ops(
            lambda t, d: t.sum(d),
            lambda n, d: n.sum(d),
            use_floating=True, use_integral=True, use_complex=True)

    @onlyCPU
    def test_mean_dim(self, device):
        self._test_dim_ops(
            lambda t, d: t.mean(d),
            lambda n, d: n.mean(d),
            use_integral=False,
            use_complex=True)

    @onlyCPU
    def test_std_dim(self, device):
        for unbiased in [False, True]:
            self._test_dim_ops(
                lambda t, d: t.std(d, unbiased=unbiased),
                lambda n, d: n.std(d, ddof=1 if unbiased else 0),
                use_integral=False)

    @onlyCPU
    def test_var_dim(self, device):
        for unbiased in [False, True]:
            self._test_dim_ops(
                lambda t, d: t.var(d, unbiased=unbiased),
                lambda n, d: n.var(d, ddof=1 if unbiased else 0),
                use_integral=False)

    @onlyCPU
    @skipIfNoSciPy
    def test_logsumexp_dim(self, device):
        from scipy.special import logsumexp
        self._test_dim_ops(
            lambda t, d: t.logsumexp(d),
            lambda n, d: logsumexp(n, d),
            use_integral=False)

    @onlyCPU
    def test_mean_int_with_optdtype(self, device):
        a = make_tensor((3, 4, 5), dtype=torch.int64, device=device)

        # If the optional desired output type is given, the input
        # is internally cast.
        a_float = a.to(torch.float32)
        self.assertEqual(a_float.mean(), a.mean(dtype=torch.float32))

    @onlyCPU
    @dtypes(torch.half, torch.bfloat16, torch.float, torch.double)
    def test_mean_out_is_alias_of_return(self, dtype, device):
        a = torch.tensor([[[1.0, 1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0, 2.0]], [[3.0, 3.0, 3.0, 3.0]]],
                         dtype=dtype, device=device)
        out = torch.empty((1, 1, 4), dtype=dtype, device=device)

        return_out = torch.mean(a, dim=0, keepdim=True, out=out)
        target = torch.tensor([[[2.0, 2.0, 2.0, 2.0]]], dtype=dtype, device=device)
        self.assertTrue(torch._C._is_alias_of(out, return_out))
        self.assertTrue(torch.allclose(out, target))

    # TODO: update this and tests that use it to handle device properly
    def _test_reduce_integer_upcast(self, fn, has_out=True, test_complex=True):
        shape = (3, 4, 5)
        reduced_shape = fn(torch.ones(shape)).shape

        def _test_out(dtype, other_dtype):
            out = torch.ones(reduced_shape, dtype=dtype)
            result = fn(x, out=out)
            self.assertIs(out.dtype, result.dtype)
            self.assertEqual(fn(x.to(dtype)), result, exact_dtype=False)
            result = fn(x, out=out, dtype=dtype)
            self.assertIs(out.dtype, result.dtype)
            self.assertEqual(fn(x.to(dtype)), result, exact_dtype=False)
            # 'out' is favored over dtype, check error
            self.assertRaises(RuntimeError, lambda: fn(x, out=out, dtype=other_dtype))

        for dtype in [dtype for dtype in get_all_math_dtypes('cpu') if dtype != torch.float16]:
            x = torch.ones(shape, dtype=dtype)
            expected_dtype = dtype if dtype.is_floating_point or dtype.is_complex else torch.int64
            self.assertIs(expected_dtype, fn(x).dtype)
            self.assertEqual(fn(x.to(expected_dtype)), fn(x))

            if dtype.is_floating_point:
                other_dtype = torch.float32 if dtype == torch.float64 else torch.float64
            elif dtype.is_complex:
                other_dtype = torch.complex64 if dtype == torch.complex128 else torch.complex128
            else:
                other_dtype = torch.int32 if dtype != torch.int32 else torch.int16
            self.assertIs(other_dtype, fn(x, dtype=other_dtype).dtype)
            self.assertEqual(fn(x.to(other_dtype)), fn(x, dtype=other_dtype), exact_dtype=False)

            # test mixed int/float/complex
            if dtype.is_floating_point:
                mixed_dtypes = [torch.int32, torch.complex64]
            elif dtype.is_complex:
                mixed_dtypes = [torch.int32, torch.float32]
            else:
                mixed_dtypes = [torch.float32, torch.complex64]

            for mixed_dtype in mixed_dtypes:
                self.assertIs(mixed_dtype, fn(x, dtype=mixed_dtype).dtype)
                self.assertEqual(fn(x.to(mixed_dtype)), fn(x, dtype=mixed_dtype), exact_dtype=False)

                if has_out:
                    _test_out(dtype, other_dtype)
                    _test_out(dtype, mixed_dtype)

    @onlyCPU
    def test_sum_integer_upcast(self, device):
        self._test_reduce_integer_upcast(lambda x, **kwargs: torch.sum(x, **kwargs), False)
        self._test_reduce_integer_upcast(lambda x, **kwargs: torch.sum(x, 0, **kwargs))

    @onlyCPU
    def test_prod_integer_upcast(self, device):
        self._test_reduce_integer_upcast(lambda x, **kwargs: torch.prod(x, **kwargs), False)
        self._test_reduce_integer_upcast(lambda x, **kwargs: torch.prod(x, 0, **kwargs))

    @onlyCPU
    def test_cumsum_integer_upcast(self, device):
        self._test_reduce_integer_upcast(lambda x, **kwargs: torch.cumsum(x, 0, **kwargs))

    @onlyCPU
    def test_cumprod_integer_upcast(self, device):
        self._test_reduce_integer_upcast(lambda x, **kwargs: torch.cumprod(x, 0, **kwargs))

    @dtypes(*all_types())
    def test_mode(self, device, dtype):
        SIZE = 10
        x = torch.arange(1., SIZE * SIZE + 1, device=device, dtype=dtype).clone().resize_(SIZE, SIZE)
        x[:2] = 1
        x[:, :2] = 1
        x0 = x.clone()

        # Pre-calculated results.
        res1val = torch.ones(SIZE, device=device, dtype=dtype)
        # The indices are the position of the last appearance of the mode element.
        res1ind = torch.ones(SIZE, device=device, dtype=torch.long)
        res1ind[0] = SIZE - 1
        res1ind[1] = SIZE - 1

        res2val, res2ind = torch.mode(x, keepdim=False)
        self.assertEqual(res1val, res2val, atol=0, rtol=0)
        self.assertEqual(res1ind, res2ind, atol=0, rtol=0)

        # Test use of result tensor
        res2val = torch.tensor((), device=device, dtype=dtype)
        res2ind = torch.tensor((), device=device, dtype=torch.long)
        torch.mode(x, keepdim=False, out=(res2val, res2ind))
        self.assertEqual(res1val, res2val, atol=0, rtol=0)
        self.assertEqual(res1ind, res2ind, atol=0, rtol=0)

        # Test non-default dim
        res2val, res2ind = torch.mode(x, 0, False)
        self.assertEqual(res1val, res2val, atol=0, rtol=0)
        self.assertEqual(res1ind, res2ind, atol=0, rtol=0)

        # input unchanged
        self.assertEqual(x, x0, atol=0, rtol=0)

    def _test_mode_intervals(self, shape, intervals, device, dtype, v=1):
        x = torch.arange(0, shape[1], device=device, dtype=dtype).expand(shape)
        x = x.contiguous()
        x[:, v] = intervals[0][0]

        # Set the value of each interval to the mode "v"
        for (beg, end) in intervals:
            x[:, beg:end] = v

        values, indices = torch.mode(x, -1, False)

        # Check whether the returned indices correspond to the returned values
        self.assertTrue((x.gather(1, indices.unsqueeze(1)).t() == values).all())
        # Check whether the returned values are the mode
        self.assertTrue((values == v).all().item())

    @onlyCUDA
    @dtypes(*all_types_and(torch.half, torch.bfloat16))
    def test_mode_large(self, device, dtype):
        # i should be less than (d - 2) / 2
        def testset_for_shape(shape, i):
            d = shape[-1]
            # Mode only in the middle.
            self._test_mode_intervals(shape, [(i, d - i)], device, dtype)
            # Mode in discontiguous parts of the input.
            self._test_mode_intervals(shape, [(0, i), (i + 1, d - i - 1), (d - i, d)], device, dtype)

        # More than one line of (65535) thread blocks
        testset_for_shape((65536, 10), 3)

        # Max slice size (2048)
        testset_for_shape((10, 2048), 10)

        # Naive kernel for big slice sizes (> 2048)
        testset_for_shape((10, 4096), 10)

    def test_mode_boolean(self, device):
        shapes = [
            (10, 10),
            (4, 2048),
            (1, 4096),
        ]

        for shape in shapes:
            a = torch.zeros(shape, device=device, dtype=torch.bool)

            a[:, (shape[1] - 1) // 2:] = True
            values, indices = a.mode(-1)
            self.assertEqual(values, torch.ones(shape[0], dtype=torch.bool))
            indexed = a.gather(1, indices.unsqueeze(1)).squeeze(1)
            self.assertEqual(values, indexed)

            a.fill_(False)
            a[:, shape[1] // 2 + 1:] = True
            values, indices = a.mode(-1)
            print(indices)
            self.assertEqual(values, torch.zeros(shape[0], dtype=torch.bool))
            indexed = a.gather(1, indices.unsqueeze(1)).squeeze(1)
            self.assertEqual(values, indexed)


    @expectedFailureMeta  # mode only supports CPU and CUDA device type
    @onlyNativeDeviceTypes
    def test_mode_wrong_dtype(self, device):
        def test_for_dtypes(x_ty, v_ty, i_ty, message):
            x = torch.ones(10, device=device, dtype=x_ty)
            v = torch.ones(10, device=device, dtype=v_ty)
            i = torch.ones(10, device=device, dtype=i_ty)

            with self.assertRaisesRegex(RuntimeError, message):
                torch.mode(x, -1, True, out=(v, i))

        err_msg = "expected scalar type .* but got .* for "
        values_err = err_msg + "values"
        indices_err = err_msg + "indices"

        test_for_dtypes(torch.uint8, torch.int8, torch.long, values_err)
        test_for_dtypes(torch.int8, torch.int16, torch.long, values_err)
        test_for_dtypes(torch.int32, torch.float32, torch.long, values_err)
        test_for_dtypes(torch.float32, torch.float64, torch.long, values_err)

        test_for_dtypes(torch.uint8, torch.uint8, torch.int8, indices_err)
        test_for_dtypes(torch.int8, torch.int8, torch.int16, indices_err)
        test_for_dtypes(torch.int32, torch.int32, torch.float32, indices_err)
        test_for_dtypes(torch.float32, torch.float32, torch.float64, indices_err)

    @onlyCUDA
    def test_mode_wrong_device(self, device):
        # CPU Input Tensor
        x = torch.ones(2)

        with self.assertRaisesRegex(RuntimeError,
                                    "expected device .* but got .* for values"):
            values = torch.tensor(
```



## High-Level Overview


This Python file contains 1 class(es) and 217 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestReductions`

**Functions defined**: `_generate_input`, `_rand_shape`, `_reduced_shape`, `_test_dim_keepdim`, `test_dim_default`, `test_dim_default_keepdim`, `test_dim_none`, `test_dim_none_keepdim`, `test_dim_single`, `test_dim_single_keepdim`, `test_dim_empty`, `test_dim_empty_keepdim`, `test_dim_multi`, `test_dim_multi_keepdim`, `test_dim_multi_unsorted`, `test_dim_multi_unsorted_keepdim`, `test_dim_multi_duplicate`, `test_dim_multi_unsupported`, `test_dim_offbounds`, `test_dim_ndim_limit`

**Key imports**: contextlib, torch, numpy as np, math, Sequence, random, partial, product, combinations, permutations, warnings, inf, nan


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `torch`
- `numpy as np`
- `math`
- `collections.abc`: Sequence
- `random`
- `functools`: partial
- `itertools`: product, combinations, permutations
- `warnings`
- `torch.testing`: make_tensor
- `scipy.special`: logsumexp


## Code Patterns & Idioms

### Common Patterns

- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/test_reductions.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test`):

- [`test_file_check.py_docs.md`](./test_file_check.py_docs.md)
- [`test_jit_simple.py_docs.md`](./test_jit_simple.py_docs.md)
- [`test_mkldnn.py_docs.md`](./test_mkldnn.py_docs.md)
- [`test_expanded_weights.py_docs.md`](./test_expanded_weights.py_docs.md)
- [`test_overrides.py_docs.md`](./test_overrides.py_docs.md)
- [`test_decomp.py_docs.md`](./test_decomp.py_docs.md)
- [`test_show_pickle.py_docs.md`](./test_show_pickle.py_docs.md)
- [`test_utils_config_module.py_docs.md`](./test_utils_config_module.py_docs.md)
- [`test_mobile_optimizer.py_docs.md`](./test_mobile_optimizer.py_docs.md)
- [`test_type_info.py_docs.md`](./test_type_info.py_docs.md)


## Cross-References

- **File Documentation**: `test_reductions.py_docs.md`
- **Keyword Index**: `test_reductions.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
