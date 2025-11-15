# Documentation: test_binary_ufuncs.py

## File Metadata
- **Path**: `test/test_binary_ufuncs.py`
- **Size**: 183538 bytes
- **Lines**: 4580
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# Owner(s): ["module: tests"]
# ruff: noqa: F841

import itertools
import math
import operator
import random
import sys
import warnings
from functools import partial
from itertools import chain, product
from numbers import Number

import numpy as np

import torch
import torch.autograd.forward_ad as fwAD
from torch import inf, nan
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    deviceCountAtLeast,
    dtypes,
    dtypesIfCPU,
    dtypesIfCUDA,
    expectedFailureMeta,
    instantiate_device_type_tests,
    onlyCPU,
    onlyCUDA,
    onlyNativeDeviceTypes,
    OpDTypes,
    ops,
    precisionOverride,
    skipIf,
    skipMeta,
)
from torch.testing._internal.common_dtype import (
    all_types_and,
    all_types_and_complex_and,
    complex_types,
    floating_and_complex_types,
    floating_types_and,
    get_all_int_dtypes,
    get_all_math_dtypes,
    integral_types,
    integral_types_and,
)
from torch.testing._internal.common_methods_invocations import (
    binary_ufuncs,
    binary_ufuncs_and_refs,
    generate_elementwise_binary_broadcasting_tensors,
    generate_elementwise_binary_extremal_value_tensors,
    generate_elementwise_binary_large_value_tensors,
    generate_elementwise_binary_small_value_tensors,
    generate_elementwise_binary_tensors,
    generate_elementwise_binary_with_scalar_and_type_promotion_samples,
    generate_elementwise_binary_with_scalar_samples,
)
from torch.testing._internal.common_utils import (
    gradcheck,
    iter_indices,
    numpy_to_torch_dtype_dict,
    run_tests,
    set_default_dtype,
    skipIfTorchDynamo,
    slowTest,
    TEST_SCIPY,
    TestCase,
    torch_to_numpy_dtype_dict,
    xfailIfTorchDynamo,
)


if TEST_SCIPY:
    import scipy.integrate
    import scipy.special


# TODO: update to use opinfos consistently
class TestBinaryUfuncs(TestCase):
    # Generic tests for elementwise binary (AKA binary universal (u) functions (funcs))
    # TODO: below contiguous tensor results are compared with a variety of noncontiguous results.
    #   It would be interesting to have the lhs and rhs have different discontinuities.

    # Helper for comparing torch tensors and NumPy arrays
    # TODO: should this or assertEqual also validate that strides are equal?
    def assertEqualHelper(
        self, actual, expected, msg, *, dtype, exact_dtype=True, **kwargs
    ):
        assert isinstance(actual, torch.Tensor)

        # Some NumPy functions return scalars, not arrays
        if isinstance(expected, Number):
            self.assertEqual(actual.item(), expected, msg=msg, **kwargs)
        elif isinstance(expected, np.ndarray):
            # Handles exact dtype comparisons between arrays and tensors
            if exact_dtype:
                # Allows array dtype to be float32 when comparing with bfloat16 tensors
                #   since NumPy doesn't support the bfloat16 dtype
                # Also ops like scipy.special.erf, scipy.special.erfc, etc, promote float16
                # to float32
                if expected.dtype == np.float32:
                    assert actual.dtype in (
                        torch.float16,
                        torch.bfloat16,
                        torch.float32,
                    )
                else:
                    assert expected.dtype == torch_to_numpy_dtype_dict[actual.dtype]

            self.assertEqual(
                actual,
                torch.from_numpy(expected).to(actual.dtype),
                msg,
                exact_device=False,
                **kwargs,
            )
        else:
            self.assertEqual(actual, expected, msg, exact_device=False, **kwargs)

    # Tests that the function and its (array-accepting) reference produce the same
    #   values on given tensors
    def _test_reference_numerics(self, dtype, op, gen, equal_nan=True):
        def _helper_reference_numerics(
            expected, actual, msg, exact_dtype, equal_nan=True
        ):
            if not torch.can_cast(
                numpy_to_torch_dtype_dict[expected.dtype.type], dtype
            ):
                exact_dtype = False

            if dtype is torch.bfloat16 and expected.dtype == np.float32:
                # Ref: https://github.com/pytorch/pytorch/blob/master/torch/testing/_internal/common_utils.py#L1149
                self.assertEqualHelper(
                    actual,
                    expected,
                    msg,
                    dtype=dtype,
                    exact_dtype=exact_dtype,
                    rtol=16e-3,
                    atol=1e-5,
                )
            else:
                self.assertEqualHelper(
                    actual,
                    expected,
                    msg,
                    dtype=dtype,
                    equal_nan=equal_nan,
                    exact_dtype=exact_dtype,
                )

        for sample in gen:
            # Each sample input acquired from the generator is just one lhs tensor
            #   and one rhs tensor
            l = sample.input
            r = sample.args[0]

            numpy_sample = sample.numpy()
            l_numpy = numpy_sample.input
            r_numpy = numpy_sample.args[0]
            actual = op(l, r)
            expected = op.ref(l_numpy, r_numpy)

            # Dtype promo rules have changed since NumPy 2.
            # Specialize the backward-incompatible cases.
            if (
                np.__version__ > "2"
                and op.name in ("sub", "_refs.sub")
                and isinstance(l_numpy, np.ndarray)
            ):
                expected = expected.astype(l_numpy.dtype)

            # Crafts a custom error message for smaller, printable tensors
            def _numel(x):
                if isinstance(x, torch.Tensor):
                    return x.numel()
                # Assumes x is a scalar
                return 1

            if _numel(l) <= 100 and _numel(r) <= 100:
                msg = (
                    "Failed to produce expected results! Input lhs tensor was"
                    f" {l}, rhs tensor was {r}, torch result is {actual}, and reference result is"
                    f" {expected}."
                )
            else:
                msg = None

            exact_dtype = True
            if isinstance(actual, torch.Tensor):
                _helper_reference_numerics(
                    expected, actual, msg, exact_dtype, equal_nan
                )
            else:
                for x, y in zip(expected, actual):
                    # testing multi-outputs results
                    _helper_reference_numerics(x, y, msg, exact_dtype, equal_nan)

    # The following tests only apply to elementwise binary operators with references
    binary_ufuncs_with_references = list(
        filter(lambda op: op.ref is not None and op.ref is not None, binary_ufuncs)
    )

    @ops(binary_ufuncs_with_references)
    def test_reference_numerics(self, device, dtype, op):
        gen = generate_elementwise_binary_tensors(op, device=device, dtype=dtype)
        self._test_reference_numerics(dtype, op, gen, equal_nan=True)

    @ops(binary_ufuncs_with_references)
    def test_reference_numerics_small_values(self, device, dtype, op):
        if dtype is torch.bool:
            self.skipTest("Doesn't support bool!")

        gen = generate_elementwise_binary_small_value_tensors(
            op, device=device, dtype=dtype
        )
        self._test_reference_numerics(dtype, op, gen, equal_nan=True)

    @ops(
        binary_ufuncs_with_references,
        allowed_dtypes=(
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        ),
    )
    def test_reference_numerics_large_values(self, device, dtype, op):
        gen = generate_elementwise_binary_large_value_tensors(
            op, device=device, dtype=dtype
        )
        self._test_reference_numerics(dtype, op, gen, equal_nan=True)

    @ops(
        binary_ufuncs_with_references,
        allowed_dtypes=(
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        ),
    )
    def test_reference_numerics_extremal_values(self, device, dtype, op):
        gen = generate_elementwise_binary_extremal_value_tensors(
            op, device=device, dtype=dtype
        )
        self._test_reference_numerics(dtype, op, gen, equal_nan=True)

    # tests broadcasting and noncontiguous broadcasting behavior
    @ops(
        binary_ufuncs_with_references,
        allowed_dtypes=(
            torch.long,
            torch.float32,
        ),
    )
    def test_broadcasting(self, device, dtype, op):
        gen = generate_elementwise_binary_broadcasting_tensors(
            op, device=device, dtype=dtype
        )
        self._test_reference_numerics(dtype, op, gen, equal_nan=True)

    @ops(
        binary_ufuncs_with_references,
        allowed_dtypes=(torch.long, torch.float32, torch.complex64),
    )
    def test_scalar_support(self, device, dtype, op):
        gen = generate_elementwise_binary_with_scalar_samples(
            op, device=device, dtype=dtype
        )
        self._test_reference_numerics(dtype, op, gen, equal_nan=True)
        gen = generate_elementwise_binary_with_scalar_and_type_promotion_samples(
            op, device=device, dtype=dtype
        )
        self._test_reference_numerics(dtype, op, gen, equal_nan=True)

    @ops(binary_ufuncs)
    def test_contig_vs_every_other(self, device, dtype, op):
        lhs = make_tensor(
            (1026,), device=device, dtype=dtype, **op.lhs_make_tensor_kwargs
        )
        rhs = make_tensor(
            (1026,), device=device, dtype=dtype, **op.rhs_make_tensor_kwargs
        )

        lhs_non_contig = lhs[::2]
        rhs_non_contig = rhs[::2]

        self.assertTrue(lhs.is_contiguous())
        self.assertTrue(rhs.is_contiguous())

        self.assertFalse(lhs_non_contig.is_contiguous())
        self.assertFalse(rhs_non_contig.is_contiguous())

        expected = op(lhs, rhs)[::2]
        actual = op(lhs_non_contig, rhs_non_contig)
        self.assertEqual(expected, actual)

    @ops(binary_ufuncs)
    def test_contig_vs_transposed(self, device, dtype, op):
        lhs = make_tensor(
            (789, 357), device=device, dtype=dtype, **op.lhs_make_tensor_kwargs
        )
        rhs = make_tensor(
            (789, 357), device=device, dtype=dtype, **op.rhs_make_tensor_kwargs
        )

        lhs_non_contig = lhs.T
        rhs_non_contig = rhs.T

        self.assertTrue(lhs.is_contiguous())
        self.assertTrue(rhs.is_contiguous())

        self.assertFalse(lhs_non_contig.is_contiguous())
        self.assertFalse(rhs_non_contig.is_contiguous())

        expected = op(lhs, rhs).T
        actual = op(lhs_non_contig, rhs_non_contig)
        self.assertEqual(expected, actual)

    @ops(binary_ufuncs)
    def test_non_contig(self, device, dtype, op):
        shapes = ((5, 7), (1024,))
        for shape in shapes:
            lhs = make_tensor(
                shape, dtype=dtype, device=device, **op.lhs_make_tensor_kwargs
            )
            rhs = make_tensor(
                shape, dtype=dtype, device=device, **op.rhs_make_tensor_kwargs
            )

            lhs_non_contig = torch.empty(shape + (2,), device=device, dtype=dtype)[
                ..., 0
            ]
            lhs_non_contig.copy_(lhs)

            rhs_non_contig = torch.empty(shape + (2,), device=device, dtype=dtype)[
                ..., 0
            ]
            rhs_non_contig.copy_(rhs)

            self.assertTrue(lhs.is_contiguous())
            self.assertTrue(rhs.is_contiguous())

            self.assertFalse(lhs_non_contig.is_contiguous())
            self.assertFalse(rhs_non_contig.is_contiguous())

            expected = op(lhs, rhs)
            actual = op(lhs_non_contig, rhs_non_contig)
            self.assertEqual(expected, actual)

    @ops(binary_ufuncs)
    def test_non_contig_index(self, device, dtype, op):
        shape = (2, 2, 1, 2)
        lhs = make_tensor(
            shape, dtype=dtype, device=device, **op.lhs_make_tensor_kwargs
        )
        rhs = make_tensor(
            shape, dtype=dtype, device=device, **op.rhs_make_tensor_kwargs
        )

        lhs_non_contig = lhs[:, 1, ...]
        lhs = lhs_non_contig.contiguous()

        rhs_non_contig = rhs[:, 1, ...]
        rhs = rhs_non_contig.contiguous()

        self.assertTrue(lhs.is_contiguous())
        self.assertTrue(rhs.is_contiguous())

        self.assertFalse(lhs_non_contig.is_contiguous())
        self.assertFalse(rhs_non_contig.is_contiguous())

        expected = op(lhs, rhs)
        actual = op(lhs_non_contig, rhs_non_contig)
        self.assertEqual(expected, actual)

    @ops(binary_ufuncs)
    def test_non_contig_expand(self, device, dtype, op):
        shapes = [(1, 3), (1, 7), (5, 7)]
        for shape in shapes:
            lhs = make_tensor(
                shape, dtype=dtype, device=device, **op.lhs_make_tensor_kwargs
            )
            rhs = make_tensor(
                shape, dtype=dtype, device=device, **op.rhs_make_tensor_kwargs
            )

            lhs_non_contig = lhs.clone().expand(3, -1, -1)
            rhs_non_contig = rhs.clone().expand(3, -1, -1)

            self.assertTrue(lhs.is_contiguous())
            self.assertTrue(rhs.is_contiguous())

            self.assertFalse(lhs_non_contig.is_contiguous())
            self.assertFalse(rhs_non_contig.is_contiguous())

            expected = op(lhs, rhs)
            actual = op(lhs_non_contig, rhs_non_contig)
            for i in range(3):
                self.assertEqual(expected, actual[i])

    @ops(binary_ufuncs)
    def test_contig_size1(self, device, dtype, op):
        shape = (5, 100)
        lhs = make_tensor(
            shape, dtype=dtype, device=device, **op.lhs_make_tensor_kwargs
        )
        rhs = make_tensor(
            shape, dtype=dtype, device=device, **op.rhs_make_tensor_kwargs
        )

        lhs = lhs[:1, :50]
        lhs_alt = torch.empty(lhs.size(), device=device, dtype=dtype)
        lhs_alt.copy_(lhs)

        rhs = rhs[:1, :50]
        rhs_alt = torch.empty(rhs.size(), device=device, dtype=dtype)
        rhs_alt.copy_(rhs)

        self.assertTrue(lhs.is_contiguous())
        self.assertTrue(rhs.is_contiguous())

        self.assertTrue(lhs_alt.is_contiguous())
        self.assertTrue(rhs_alt.is_contiguous())

        expected = op(lhs, rhs)
        actual = op(lhs_alt, rhs_alt)
        self.assertEqual(expected, actual)

    @ops(binary_ufuncs)
    def test_contig_size1_large_dim(self, device, dtype, op):
        shape = (5, 2, 3, 1, 4, 5, 3, 2, 1, 2, 3, 4)
        lhs = make_tensor(
            shape, dtype=dtype, device=device, **op.lhs_make_tensor_kwargs
        )
        rhs = make_tensor(
            shape, dtype=dtype, device=device, **op.rhs_make_tensor_kwargs
        )

        lhs = lhs[:1, :, :, :, :, :, :, :, :, :, :, :]
        lhs_alt = torch.empty(lhs.size(), device=device, dtype=dtype)
        lhs_alt.copy_(lhs)

        rhs = rhs[:1, :, :, :, :, :, :, :, :, :, :, :]
        rhs_alt = torch.empty(rhs.size(), device=device, dtype=dtype)
        rhs_alt.copy_(rhs)

        self.assertTrue(lhs.is_contiguous())
        self.assertTrue(rhs.is_contiguous())

        self.assertTrue(lhs_alt.is_contiguous())
        self.assertTrue(rhs_alt.is_contiguous())

        expected = op(lhs, rhs)
        actual = op(lhs_alt, rhs_alt)
        self.assertEqual(expected, actual)

    @ops(binary_ufuncs)
    def test_batch_vs_slicing(self, device, dtype, op):
        shape = (32, 512)
        lhs = make_tensor(
            shape, dtype=dtype, device=device, **op.lhs_make_tensor_kwargs
        )
        rhs = make_tensor(
            shape, dtype=dtype, device=device, **op.rhs_make_tensor_kwargs
        )

        expected = op(lhs, rhs)

        actual = []
        for idx in range(32):
            actual.append(op(lhs[idx], rhs[idx]))
        actual = torch.stack(actual)

        self.assertEqual(expected, actual)

    # Tests that elementwise binary operators participate in type promotion properly
    # NOTE: because the cross-product of all possible type promotion tests is huge, this
    #   just spot checks some handwritten cases.
    # NOTE: It may be possible to refactor this test into something simpler
    @ops(binary_ufuncs_and_refs, dtypes=OpDTypes.none)
    def test_type_promotion(self, device, op):
        supported_dtypes = op.supported_dtypes(torch.device(device).type)

        make_lhs = partial(
            make_tensor, (5,), device=device, **op.lhs_make_tensor_kwargs
        )
        make_rhs = partial(
            make_tensor, (5,), device=device, **op.rhs_make_tensor_kwargs
        )

        make_rhs_scalar_tensor = partial(
            make_tensor, (), device="cpu", **op.rhs_make_tensor_kwargs
        )

        def _supported(dtypes):
            return all(x in supported_dtypes for x in dtypes)

        # int x int type promotion
        if _supported((torch.int16, torch.int32, torch.int64)):
            lhs_i16 = make_lhs(dtype=torch.int16)
            lhs_i32 = make_lhs(dtype=torch.int32)
            lhs_i64 = make_lhs(dtype=torch.int64)

            rhs_i16 = make_rhs(dtype=torch.int16)
            rhs_i32 = make_rhs(dtype=torch.int32)
            rhs_i64 = make_rhs(dtype=torch.int64)

            if op.promotes_int_to_float:
                default_dtype = torch.get_default_dtype()
                self.assertEqual(op(lhs_i16, rhs_i32).dtype, default_dtype)
                self.assertEqual(
                    op(lhs_i16, rhs_i32),
                    op(lhs_i16.to(default_dtype), rhs_i32.to(default_dtype)),
                )

                self.assertEqual(op(lhs_i32, rhs_i64).dtype, default_dtype)
                self.assertEqual(
                    op(lhs_i32, rhs_i64),
                    op(lhs_i32.to(default_dtype), rhs_i64.to(default_dtype)),
                )
            elif op.always_returns_bool:
                self.assertEqual(op(lhs_i16, rhs_i32).dtype, torch.bool)
                self.assertEqual(op(lhs_i32, rhs_i64).dtype, torch.bool)
            else:  # standard type promotion
                self.assertEqual(op(lhs_i16, rhs_i32).dtype, torch.int32)
                self.assertEqual(
                    op(lhs_i16, rhs_i32), op(lhs_i16.to(torch.int32), rhs_i32)
                )

                self.assertEqual(op(lhs_i32, rhs_i64).dtype, torch.int64)
                self.assertEqual(
                    op(lhs_i32, rhs_i64), op(lhs_i32.to(torch.int64), rhs_i64)
                )

            if op.supports_out:
                if not op.promotes_int_to_float:
                    # Integers can be safely cast to other integer types
                    out = torch.empty_like(lhs_i64)
                    self.assertEqual(op(lhs_i16, rhs_i32, out=out).dtype, torch.int64)
                    self.assertEqual(op(lhs_i16, rhs_i32), out, exact_dtype=False)

                    out = torch.empty_like(lhs_i16)
                    self.assertEqual(op(lhs_i32, rhs_i64, out=out).dtype, torch.int16)
                else:
                    # Float outs cannot be safely cast to integer types
                    with self.assertRaisesRegex(RuntimeError, "can't be cast"):
                        op(lhs_i16, rhs_i32, out=torch.empty_like(lhs_i64))

                if not op.always_returns_bool:
                    # Neither integer nor float outs can be cast to bool
                    with self.assertRaisesRegex(RuntimeError, "can't be cast"):
                        op(
                            lhs_i16,
                            rhs_i32,
                            out=torch.empty_like(lhs_i64, dtype=torch.bool),
                        )

                # All these output types can be cast to any float or complex type
                out = torch.empty_like(lhs_i64, dtype=torch.float16)
                self.assertEqual(op(lhs_i16, rhs_i32, out=out).dtype, torch.float16)

                out = torch.empty_like(lhs_i64, dtype=torch.bfloat16)
                self.assertEqual(op(lhs_i16, rhs_i32, out=out).dtype, torch.bfloat16)

                out = torch.empty_like(lhs_i64, dtype=torch.float32)
                self.assertEqual(op(lhs_i16, rhs_i32, out=out).dtype, torch.float32)
                self.assertEqual(op(lhs_i16, rhs_i32), out, exact_dtype=False)

                out = torch.empty_like(lhs_i64, dtype=torch.complex64)
                self.assertEqual(op(lhs_i16, rhs_i32, out=out).dtype, torch.complex64)
                self.assertEqual(op(lhs_i16, rhs_i32), out, exact_dtype=False)

        # float x float type promotion
        if _supported((torch.float32, torch.float64)):
            lhs_f32 = make_lhs(dtype=torch.float32)
            lhs_f64 = make_lhs(dtype=torch.float64)

            rhs_f32 = make_rhs(dtype=torch.float32)
            rhs_f64 = make_rhs(dtype=torch.float64)

            if op.always_returns_bool:
                self.assertEqual(op(lhs_f32, rhs_f64).dtype, torch.bool)
            else:  # normal float type promotion
                self.assertEqual(op(lhs_f32, rhs_f64).dtype, torch.float64)
                self.assertEqual(
                    op(lhs_f32, rhs_f64), op(lhs_f32.to(torch.float64), rhs_f64)
                )

            if op.supports_out:
                # All these output types can be cast to any float or complex type
                out = torch.empty_like(lhs_f64, dtype=torch.float16)
                self.assertEqual(op(lhs_f32, rhs_f64, out=out).dtype, torch.float16)

                out = torch.empty_like(lhs_f64, dtype=torch.bfloat16)
                self.assertEqual(op(lhs_f32, rhs_f64, out=out).dtype, torch.bfloat16)
                self.assertEqual(op(lhs_f32, rhs_f64), out, exact_dtype=False)

                out = torch.empty_like(lhs_f64, dtype=torch.float32)
                self.assertEqual(op(lhs_f32, rhs_f64, out=out).dtype, torch.float32)
                self.assertEqual(op(lhs_f32, rhs_f64), out, exact_dtype=False)

                out = torch.empty_like(lhs_f64, dtype=torch.complex64)
                self.assertEqual(op(lhs_f32, rhs_f64, out=out).dtype, torch.complex64)
                self.assertEqual(op(lhs_f32, rhs_f64), out, exact_dtype=False)

                if not op.always_returns_bool:
                    # float outs can't be cast to an integer dtype
                    with self.assertRaisesRegex(RuntimeError, "can't be cast"):
                        op(
                            lhs_f32,
                            rhs_f64,
                            out=torch.empty_like(lhs_f64, dtype=torch.int64),
                        )
                else:
                    # bool outs can be cast to an integer dtype
                    out = torch.empty_like(lhs_f64, dtype=torch.int64)
                    self.assertEqual(op(lhs_f32, rhs_f64, out=out).dtype, torch.int64)
                    self.assertEqual(op(lhs_f32, rhs_f64), out, exact_dtype=False)

        # complex x complex type promotion
        if _supported((torch.complex64, torch.complex128)):
            lhs_c64 = make_lhs(dtype=torch.complex64)
            lhs_c128 = make_lhs(dtype=torch.complex128)

            rhs_c64 = make_rhs(dtype=torch.complex64)
            rhs_c128 = make_rhs(dtype=torch.complex128)

            if op.always_returns_bool:
                self.assertEqual(op(lhs_c64, lhs_c128).dtype, torch.bool)
            else:  # normal complex type promotion
                self.assertEqual(op(lhs_c64, rhs_c128).dtype, torch.complex128)
                self.assertEqual(
                    op(lhs_c64, rhs_c128), op(lhs_c64.to(torch.complex128), rhs_c128)
                )

            if op.supports_out:
                # All these output types can be cast to any or complex type
                out = torch.empty_like(lhs_c64, dtype=torch.complex64)

                self.assertEqual(op(lhs_c64, rhs_c128, out=out).dtype, torch.complex64)
                result = op(lhs_c64, rhs_c128)
                self.assertEqual(result, out.to(result.dtype))

                if not op.always_returns_bool:
                    # complex outs can't be cast to float types
                    with self.assertRaisesRegex(RuntimeError, "can't be cast"):
                        op(
                            lhs_c64,
                            rhs_c128,
                            out=torch.empty_like(lhs_c64, dtype=torch.float64),
                        )
                    # complex outs can't be cast to an integer dtype
                    with self.assertRaisesRegex(RuntimeError, "can't be cast"):
                        op(
                            lhs_c64,
                            rhs_c128,
                            out=torch.empty_like(lhs_c64, dtype=torch.int64),
                        )
                else:
                    # bool outs can be cast to a float type
                    out = torch.empty_like(lhs_c64, dtype=torch.float64)
                    self.assertEqual(
                        op(lhs_c64, rhs_c128, out=out).dtype, torch.float64
                    )
                    self.assertEqual(op(lhs_c64, rhs_c128), out, exact_dtype=False)

                    # bool outs can be cast to an integer dtype
                    out = torch.empty_like(lhs_f64, dtype=torch.int64)
                    self.assertEqual(op(lhs_f32, rhs_f64, out=out).dtype, torch.int64)
                    self.assertEqual(op(lhs_f32, rhs_f64), out, exact_dtype=False)

        # int x float type promotion
        # Note: float type is the result dtype
        if _supported((torch.long, torch.float32)):
            lhs_i64 = make_lhs(dtype=torch.int64)
            rhs_f32 = make_rhs(dtype=torch.float32)

            result = op(lhs_i64, rhs_f32)
            expected_dtype = torch.float32 if not op.always_returns_bool else torch.bool
            self.assertEqual(result.dtype, expected_dtype)

        # float x complex type promotion
        # Note: complex type with highest "value type" is the result dtype
        if _supported((torch.float64, torch.complex64)):
            lhs_f64 = make_lhs(dtype=torch.float64)
            rhs_c64 = make_rhs(dtype=torch.complex64)

            result = op(lhs_f64, rhs_c64)
            expected_dtype = (
                torch.complex128 if not op.always_returns_bool else torch.bool
            )
            self.assertEqual(result.dtype, expected_dtype)

        # int x float scalar type promotion
        # Note: default float dtype is the result dtype
        if _supported((torch.int64, torch.float32)) and op.supports_rhs_python_scalar:
            lhs_i64 = make_lhs(dtype=torch.int64)
            rhs_f_scalar = 1.0

            result = op(lhs_i64, rhs_f_scalar)
            expected_dtype = (
                torch.get_default_dtype() if not op.always_returns_bool else torch.bool
            )
            self.assertEqual(result.dtype, expected_dtype)

            # repeats with a scalar float tensor, which should set the dtype
            rhs_f32_scalar_tensor = make_rhs_scalar_tensor(dtype=torch.float32)
            result = op(lhs_i64, rhs_f32_scalar_tensor)
            expected_dtype = torch.float32 if not op.always_returns_bool else torch.bool
            self.assertEqual(result.dtype, expected_dtype)

            # Additional test with double
            if _supported((torch.float64,)):
                rhs_f64_scalar_tensor = make_rhs_scalar_tensor(dtype=torch.float64)
                result = op(lhs_i64, rhs_f64_scalar_tensor)
                expected_dtype = (
                    torch.float64 if not op.always_returns_bool else torch.bool
                )
                self.assertEqual(result.dtype, expected_dtype)

        # float x complex scalar type promotion
        # Note: result dtype is complex with highest "value type" among all tensors
        if (
            _supported((torch.float32, torch.complex64))
            and op.supports_rhs_python_scalar
        ):
            lhs_f32 = make_lhs(dtype=torch.float32)
            rhs_c_scalar = complex(1, 1)

            result = op(lhs_f32, rhs_c_scalar)
            expected_dtype = (
                torch.complex64 if not op.always_returns_bool else torch.bool
            )
            self.assertEqual(result.dtype, expected_dtype)

            # repeats with a scalar complex tensor
            rhs_c64_scalar_tensor = make_rhs_scalar_tensor(dtype=torch.complex64)
            result = op(lhs_f32, rhs_c64_scalar_tensor)
            expected_dtype = (
                torch.complex64 if not op.always_returns_bool else torch.bool
            )
            self.assertEqual(result.dtype, expected_dtype)

            # Additional test with complexdouble
            if _supported((torch.complex128,)):
                rhs_c128_scalar_tensor = make_rhs_scalar_tensor(dtype=torch.complex128)
                result = op(lhs_f32, rhs_c128_scalar_tensor)
                # Value type of 1D+ Tensor (lhs_f32) takes priority over scalar tensor (rhs_c128).
                expected_dtype = (
                    torch.complex64 if not op.always_returns_bool else torch.bool
                )
                self.assertEqual(result.dtype, expected_dtype)

        # float x float scalar tensor
        # Note: result dtype is the type of the float tensor
        if _supported((torch.float32, torch.float64)) and op.supports_rhs_python_scalar:
            lhs_f32 = make_lhs(dtype=torch.float32)
            rhs_f64_scalar_tensor = make_rhs_scalar_tensor(dtype=torch.float64)

            result = op(lhs_f32, rhs_f64_scalar_tensor)
            expected_dtype = torch.float32 if not op.always_returns_bool else torch.bool
            self.assertEqual(result.dtype, expected_dtype)

        # complex x complex scalar tensor
        # Note: result dtype is the type of the complex tensor
        if (
            _supported((torch.complex64, torch.complex128))
            and op.supports_rhs_python_scalar
        ):
            lhs_c64 = make_lhs(dtype=torch.complex64)
            rhs_c128_scalar_tensor = make_rhs_scalar_tensor(dtype=torch.complex128)

            result = op(lhs_c64, rhs_c128_scalar_tensor)
            expected_dtype = (
                torch.complex64 if not op.always_returns_bool else torch.bool
            )
            self.assertEqual(result.dtype, expected_dtype)

        # scalar  x scalar
        # Note: result dtype is default float type
        if op.supports_two_python_scalars and _supported((torch.long, torch.float32)):
            rhs_f_scalar = 2.0
            for lhs in (1, 1.0):
                result = op(lhs, rhs_f_scalar)
                expected_dtype = (
                    torch.get_default_dtype()
                    if not op.always_returns_bool
                    else torch.bool
                )
                self.assertEqual(result.dtype, expected_dtype)

    # TODO: move to error input test
    @ops(binary_ufuncs, allowed_dtypes=(torch.float32,))
    def test_not_broadcastable(self, device, dtype, op):
        for shape_lhs, shape_rhs in (
            ((2,), (3,)),
            ((3, 1), (2, 1)),
            ((1, 3, 2), (3,)),
            ((3, 1, 2), (2, 1, 2)),
        ):
            lhs = make_tensor(
                shape_lhs, device=device, dtype=dtype, **op.lhs_make_tensor_kwargs
            )
            rhs = make_tensor(
                shape_rhs, device=device, dtype=dtype, **op.rhs_make_tensor_kwargs
            )

            try:
                broadcasted_shape = op(lhs, rhs).shape
            except RuntimeError:
                continue

            msg = (
                f"On {device}, torch.{op.name} broadcasts inputs shapes {shape_lhs} and {shape_rhs} into "
                f"{broadcasted_shape}, although they are not broadcastable."
            )
            raise AssertionError(msg)

    def test_add_broadcast_empty(self, device):
        # empty + empty
        self.assertRaises(
            RuntimeError,
            lambda: torch.randn(5, 0, device=device) + torch.randn(0, 5, device=device),
        )
        self.assertEqual(
            torch.randn(5, 0, device=device),
            torch.randn(0, device=device) + torch.randn(5, 0, device=device),
        )
        self.assertEqual(
            torch.randn(5, 0, 0, device=device),
            torch.randn(0, device=device) + torch.randn(5, 0, 1, device=device),
        )

        # scalar + empty
        self.assertEqual(
            torch.randn(5, 0, 6, device=device),
            torch.randn((), device=device) + torch.randn(5, 0, 6, device=device),
        )

        # non-empty, empty
        self.assertEqual(
            torch.randn(0, device=device),
            torch.randn(0, device=device) + torch.randn(1, device=device),
        )
        self.assertEqual(
            torch.randn(0, 7, 0, 6, 5, 0, 7, device=device),
            torch.randn(0, 7, 0, 6, 5, 0, 1, device=device)
            + torch.randn(1, 1, 5, 1, 7, device=device),
        )
        self.assertRaises(
            RuntimeError,
            lambda: torch.randn(7, 0, device=device) + torch.randn(2, 1, device=device),
        )

    def test_addcmul_scalars_as_floats(self, device):
        # zero-dim variables that don't require grad should bind to scalar arguments
        x = torch.tensor(2.0)
        y = torch.tensor(3.0, device=device)
        # 3 + (3 * 3) * 2
        self.assertEqual(y.addcmul(y, y, value=x), 21)

        x = torch.tensor(2.0, requires_grad=True)
        self.assertRaises(Exception, lambda: y.addcmul(y, y, value=x))

    # Tests that the binary operators and, or, and xor (as well as their reflected and inplace versions)
    # work properly (AKA &, ||, ^ and &=, |=, ^=)
    @dtypes(*integral_types_and(torch.bool))
    def test_bitwise_ops(self, device, dtype):
        # Tensor x Tensor and Tensor x Scalar ops
        ops = (
            operator.and_,
            operator.iand,
            operator.or_,
            operator.ior,
            operator.xor,
            operator.ixor,
        )
        inplace_ops = (operator.iand, operator.ior, operator.ixor)
        shapes = ((5,), (15, 15), (500, 500))

        for op, shape in itertools.product(ops, shapes):
            # Tests tensor x tensor case
            a = make_tensor(shape, device=device, dtype=dtype)
            b = make_tensor(shape, device=device, dtype=dtype)
            a_np = a.cpu().clone().numpy()
            b_np = b.cpu().clone().numpy()
            self.assertEqual(op(a, b), op(a_np, b_np))

            # Tests tensor x scalar case
            a = make_tensor(shape, device=device, dtype=dtype)
            b_scalar = make_tensor((), device="cpu", dtype=dtype).item()
            a_np = a.cpu().clone().numpy()
            self.assertEqual(op(a, b_scalar), op(a_np, b_scalar))

            # Tests scalar x tensor case
            a_scalar = make_tensor((), device="cpu", dtype=dtype).item()
            b = make_tensor(shape, device=device, dtype=dtype)
            b_np = b.cpu().clone().numpy()
            self.assertEqual(op(a_scalar, b), op(a_scalar, b_np))

            # Tests scalar x tensor case (for ops which aren't inplace)
            if op in inplace_ops:
                # Tests tensor x tensor case
                a = make_tensor(shape, device=device, dtype=dtype)
                b = make_tensor(shape, device=device, dtype=dtype)
                a_np = a.cpu().clone().numpy()
                b_np = b.cpu().clone().numpy()
                op(a, b)
                op(a_np, b_np)
                self.assertEqual(a, a_np)

                # Tests tensor x scalar case
                a = make_tensor(shape, device=device, dtype=dtype)
                b_scalar = make_tensor((), device="cpu", dtype=dtype).item()
                a_np = a.cpu().clone().numpy()
                op(a, b_scalar)
                op(a_np, b_scalar)
                self.assertEqual(a, a_np)

    def test_inplace_division(self, device):
        t = torch.rand(5, 5, device=device)
        id_before = id(t)
        t /= 2
        id_after = id(t)
        self.assertEqual(id_before, id_after)

    @dtypes(*all_types_and(torch.half, torch.bfloat16))
    def test_div_rounding_modes(self, device, dtype):
        if dtype.is_floating_point:
            low, high = -10.0, 10.0
        else:
            info = torch.iinfo(dtype)
            low, high = info.min, info.max

        a = make_tensor((100,), dtype=dtype, device=device, low=low, high=high)
        b = make_tensor((100,), dtype=dtype, device=device, low=low, high=high)

        # Avoid division by zero so we can test (a / b) * b == a
        if dtype.is_floating_point:
            eps = 0.1
            b[(-eps < b) & (b < eps)] = eps
        else:
            b[b == 0] = 1

        if not dtype.is_floating_point:
            # floor(a / b) * b can be < a, so fixup slightly to avoid underflow
            a = torch.where(a < 0, a + b, a)

        d_true = torch.divide(a, b, rounding_mode=None)
        self.assertTrue(d_true.is_floating_point())
        self.assertEqual(d_true * b, a.to(d_true.dtype))

        d_floor = torch.divide(a, b, rounding_mode="floor")
        if dtype not in (torch.bfloat16, torch.half):
            self.assertEqual(d_floor * b + torch.remainder(a, b), a)
        else:
            self.assertEqual(
                d_floor * b + torch.remainder(a.float(), b.float()),
                a,
                exact_dtype=False,
            )

        d_trunc = torch.divide(a, b, rounding_mode="trunc")
        rounding_unsupported = (
            dtype == torch.half
            and device != "cuda"
            or dtype == torch.bfloat16
            and device != "cpu"
        )
        d_ref = d_true.float() if rounding_unsupported else d_true
        self.assertEqual(d_trunc, d_ref.trunc().to(dtype))

    @dtypes(*floating_types_and(torch.bfloat16, torch.float16))
    def test_floor_div_extremal(self, device, dtype):
        for num, denom, shape in itertools.product(
            [torch.finfo(dtype).max * 0.7],
            [0.5, -0.5, 0.0],
            [(), (32,)],  # Scalar and vectorized
        ):
            a = torch.full(shape, num, dtype=dtype, device=device)
            b = torch.full(shape, denom, dtype=dtype, device=device)

            ref = np.floor_divide(num, denom).item()
            if ref > torch.finfo(dtype).max:
                ref = np.inf
            elif ref < torch.finfo(dtype).min:
                ref = -np.inf
            expect = torch.full(shape, ref, dtype=dtype, device=device)
            actual = torch.div(a, b, rounding_mode="floor")
            self.assertEqual(expect, actual)

    @dtypes(torch.bfloat16, torch.half, torch.float32, torch.float64)
    def test_div_rounding_nonfinite(self, device, dtype):
        # Compare division of special floating point values against NumPy
        num = torch.tensor(
            [1.0, -1.0, 0, 0.1, -0.1, np.pi, -np.pi, np.inf, -np.inf, np.nan],
            dtype=dtype,
            device=device,
        )
        # Divide by zero is tested separately
        denom = num[num != 0]

        a, b = num[None, :].clone(), denom[:, None].clone()

        # Compare bfloat16 against NumPy float
        exact_dtype = dtype != torch.bfloat16
        if exact_dtype:
            an, bn = a.cpu().numpy(), b.cpu().numpy()
        else:
            an, bn = a.float().cpu().numpy(), b.float().cpu().numpy()

        for mode, np_ref in ((None, np.true_divide), ("floor", np.floor_divide)):
            expect = np_ref(an, bn)
            kwargs = dict(rounding_mode=mode) if mode is not None else {}
            with set_default_dtype(torch.double):
                actual = torch.divide(a, b, **kwargs)
            self.assertEqual(
                actual,
                torch.from_numpy(expect),
                exact_device=False,
                exact_dtype=exact_dtype,
            )

        # Compare contiguous (likely vectorized) against non-contiguous (not vectorized)
        a_noncontig = torch.empty([2 * i for i in a.shape], dtype=dtype, device=device)[
            ::2, ::2
        ]
        a_noncontig[:] = a
        b_noncontig = torch.empty([2 * i for i in b.shape], dtype=dtype, device=device)[
            ::2, ::2
        ]
        b_noncontig[:] = b

        for rounding_mode in (None, "trunc", "floor"):
            expect = torch.divide(a_noncontig, b_noncontig, rounding_mode=rounding_mode)
            actual = torch.divide(a, b, rounding_mode=rounding_mode)
            self.assertEqual(actual, expect)

    @dtypes(torch.bfloat16, torch.half, torch.float32, torch.float64)
    def test_divide_by_zero_rounding(self, device, dtype):
        a = torch.tensor(
            [1.0, -1.0, 0, 0.1, -0.1, np.pi, -np.pi, np.inf, -np.inf, np.nan],
            dtype=dtype,
        )
        exact_dtype = dtype != torch.bfloat16
        if exact_dtype:
            an = a.cpu().numpy()
        else:
            an = a.float().cpu().numpy()

        zero = torch.zeros_like(a)

        # NOTE: NumPy's floor_divide rounding changed in 1.20.0 to be consistent with divide
        expect = np.divide(an, 0)
        for rounding_mode in (None, "floor"):
            # CPU scalar
            actual = torch.divide(a, 0, rounding_mode=rounding_mode)
            self.assertEqual(actual, expect, exact_dtype=exact_dtype)
            # Device tensor
            actual = torch.divide(a, zero, rounding_mode=rounding_mode)
            self.assertEqual(actual, expect, exact_dtype=exact_dtype)

    @dtypes(*all_types_and(torch.half))
    def test_div_rounding_numpy(self, device, dtype):
        info = torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)
        low, high = info.min, info.max

        # Compare division of random values against NumPy
        a = make_tensor((4096,), dtype=dtype, device=device, low=low, high=high)
        b = make_tensor((4096,), dtype=dtype, device=device, low=low, high=high)

        # Avoid division by zero which raises for integers and, for floats,
        # NumPy 1.20 changed floor_divide to follow IEEE rules for inf/nan
        # after dividing by zero.
        b[b == 0] = 1

        # Compare bfloat16 against NumPy float
        exact_dtype = dtype != torch.bfloat16

        if exact_dtype:
            an, bn = a.cpu().numpy(), b.cpu().numpy()
        else:
            an, bn = a.float().cpu().numpy(), b.float().cpu().numpy()

        for mode, np_ref in (
            (None, np.true_divide),
            ("floor", np.floor_divide),
            ("trunc", lambda a, b: np.trunc(np.true_divide(a, b)).astype(a.dtype)),
        ):
            expect = torch.from_numpy(np_ref(an, bn))

            kwargs = dict(rounding_mode=mode) if mode is not None else {}
            # Contiguous (likely vectorized)
            with set_default_dtype(torch.double):
                actual = torch.divide(a, b, **kwargs)
            self.assertEqual(
                actual, expect, exact_device=False, exact_dtype=exact_dtype
            )

            # Non-contiguous (not vectorized)
            expect = expect[::2]
            with set_default_dtype(torch.double):
                actual = torch.divide(a[::2], b[::2], **kwargs)

            self.assertEqual(
                actual, expect, exact_device=False, exact_dtype=exact_dtype
            )

    @dtypes(*complex_types())
    def test_complex_div_underflow_overflow(self, device, dtype):
        # test to make sure the complex division does not produce underflow or overflow
        # in the intermediate of its calculations
        # NOTE: the calculation still produces an error if the number is greater than
        # finfo.max / 2, but hopefully people realized that it's a dangerous region to work with
        finfo = torch.finfo(dtype)
        nom_lst = [
            complex(finfo.min / 2, finfo.min / 2),
            complex(finfo.max / 2, finfo.max / 2),
            complex(finfo.tiny, finfo.tiny),
            complex(finfo.tiny, 0.0),
            complex(0.0, 0.0),
        ]
        denom_lst = [
            complex(finfo.min / 2, finfo.min / 2),
            complex(finfo.max / 2, finfo.max / 2),
            complex(finfo.tiny, finfo.tiny),
            complex(0.0, finfo.tiny),
            complex(finfo.tiny, finfo.tiny),
        ]
        expected_lst = [
            complex(1.0, 0.0),
            complex(1.0, 0.0),
            complex(1.0, 0.0),
            complex(0.0, -1.0),
            complex(0.0, 0.0),
        ]
        nom = torch.tensor(nom_lst, dtype=dtype, device=device)
        denom = torch.tensor(denom_lst, dtype=dtype, device=device)
        expected = torch.tensor(expected_lst, dtype=dtype, device=device)
        res = nom / denom
        self.assertEqual(res, expected)

    # Tests that trying to add, inplace, a CUDA tensor to a CPU tensor
    #   throws the correct error message
    @onlyCUDA
    def test_cross_device_inplace_error_msg(self, device):
        a = torch.tensor(2.0)
        b = torch.tensor(2.0, device=device)
        with self.assertRaisesRegex(
            RuntimeError, "Expected all tensors to be on the same device"
        ):
            a += b

    # TODO: refactor this test into a more generic one, it's parked here currently
    @onlyNativeDeviceTypes
    def test_out_resize_warning(self, device):
        a = torch.tensor((1, 2, 3), device=device, dtype=torch.float32)
        b = torch.tensor((4, 5, 6), device=device, dtype=torch.float32)

        unary_inputs = (a,)
        binary_inputs = (a, b)
        unary_ops = (torch.ceil, torch.exp)
        binary_ops = (torch.add, torch.sub)
        for op in unary_ops + binary_ops:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                inputs = unary_inputs if op in unary_ops else binary_inputs

                # No warnings
                op(*inputs, out=torch.empty(3, device=device))
                op(*inputs, out=torch.empty(0, device=device))
                self.assertEqual(len(w), 0)

                # Cases that throw warnings
                op(*inputs, out=torch.empty(2, device=device))
                self.assertEqual(len(w), 1)
        # test that multi-d out doesn't trigger segfault
        arg1 = (torch.ones(2, 1, device=device), torch.ones(1, device=device))
        arg2 = (torch.ones(2, device=device), torch.ones(1, 1, device=device))
        outs = (
            torch.ones(2, 1, 1, 1, device=device),
            torch.ones(2, 2, 2, 2, device=device),
        )

        for a1, a2, o in zip(arg1, arg2, outs):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                torch.mul(a1, a2, out=o)
                self.assertEqual(len(w), 1)

    # Verifies that the inplace dunders (like idiv) actually are in place
    @expectedFailureMeta  # UserWarning not triggered
    @onlyNativeDeviceTypes
    def test_inplace_dunders(self, device):
        t = torch.randn((1,), device=device)
        expected = t.data_ptr()
        t += 1
        t -= 1
        t *= 1
        t /= 1
        t **= 1
        t //= 1
        t %= 1
        self.assertEqual(expected, t.data_ptr())

    def check_internal_mem_overlap(
        self, inplace_op, num_inputs, dtype, device, expected_failure=False
    ):
        if isinstance(inplace_op, str):
            inplace_op = getattr(torch.Tensor, inplace_op)
        input = torch.randn(1, dtype=dtype, device=device).expand(3, 3)
        inputs = [input] + [torch.randn_like(input) for i in range(num_inputs - 1)]
        if not expected_failure:
            with self.assertRaisesRegex(RuntimeError, "single memory location"):
                inplace_op(*inputs)
        else:
            with self.assertRaises(AssertionError):
                with self.assertRaisesRegex(RuntimeError, "single memory location"):
                    inplace_op(*inputs)

    def unary_check_input_output_mem_overlap(
        self, data, sz, op, expected_failure=False
    ):
        def _test(op, output, input):
            output_exp = torch.empty_like(output)
            op(input, out=output_exp)
            self.assertEqual(op(input, out=output), output_exp, msg=op.__name__)

        # output is identical to input:
        _test(op, output=data[0:sz], input=data[0:sz])
        # output and input are independent:
        _test(op, output=data[0:sz], input=data[sz : 2 * sz])
        # output partially overlaps with input:
        if not expected_failure:
            with self.assertRaisesRegex(RuntimeError, "unsupported operation"):
                _test(op, data[0:sz], data[1 : sz + 1])
        else:
            with self.assertRaises(AssertionError):
                with self.assertRaisesRegex(RuntimeError, "unsupported operation"):
                    _test(op, data[0:sz], data[1 : sz + 1])

    def binary_check_input_output_mem_overlap(self, op, device, expected_failure=False):
        sz = 3
        data = torch.randn(2 * sz, device=device)
        other = torch.randn(sz, device=device)

        self.unary_check_input_output_mem_overlap(
            data,
            sz,
            lambda input, out: op(other, input, out=out),
            expected_failure=expected_failure,
        )

        self.unary_check_input_output_mem_overlap(
            data,
            sz,
            lambda input, out: op(input, other, out=out),
            expected_failure=expected_failure,
        )

    # https://github.com/pytorch/pytorch/issues/126474
    @xfailIfTorchDynamo
    @dtypes(torch.double)
    def test_binary_op_mem_overlap(self, device, dtype):
        ops = [
            ("add", True, True, "cpu"),
            ("add", True, True, "cuda"),
            ("mul", True, True, "cpu"),
            ("mul", True, True, "cuda"),
            ("sub", True, True, "cpu"),
            ("sub", True, True, "cuda"),
            ("div", True, True, "cpu"),
            ("div", True, True, "cuda"),
            ("pow", True, True, "cpu"),
            ("pow", True, True, "cuda"),
            ("fmod", True, True, "cpu"),
            ("fmod", True, True, "cuda"),
            ("atan2", True, True, "cpu"),
            ("atan2", True, True, "cuda"),
            ("hypot", True, True, "cpu"),
            ("hypot", True, True, "cuda"),
            ("igamma", True, True, "cpu"),
            ("igamma", True, True, "cuda"),
            ("igammac", True, True, "cpu"),
            ("igammac", True, True, "cuda"),
            ("nextafter", True, True, "cpu"),
            ("nextafter", True, True, "cuda"),
            ("le", True, True, "cpu"),
            ("le", True, True, "cuda"),
            ("lt", True, True, "cpu"),
            ("lt", True, True, "cuda"),
            ("ge", True, True, "cpu"),
            ("ge", True, True, "cuda"),
            ("gt", True, True, "cpu"),
            ("gt", True, True, "cuda"),
            ("eq", True, True, "cpu"),
            ("eq", True, True, "cuda"),
            ("ne", True, True, "cpu"),
            ("ne", True, True, "cuda"),
            ("logical_and", True, True, "cpu"),
            ("logical_and", True, True, "cuda"),
            ("logical_or", True, True, "cpu"),
            ("logical_or", True, True, "cuda"),
            ("logical_xor", True, True, "cpu"),
            ("logical_xor", True, True, "cuda"),
        ]

        for (
            fn,
            has_input_output_mem_overlap_check,
            has_internal_mem_overlap_check,
            dev,
        ) in ops:
            if dev != device:
                continue
            out_op = getattr(torch, fn)
            inplace_op = getattr(torch.Tensor, fn + "_")
            self.check_internal_mem_overlap(
                inplace_op,
                2,
                dtype,
                device,
                expected_failure=not has_internal_mem_overlap_check,
            )

            self.binary_check_input_output_mem_overlap(
                out_op, device, expected_failure=not has_input_output_mem_overlap_check
            )

    def _do_pow_for_exponents(self, m1, exponents, pow_fn, atol):
        for num in exponents:
            if (
                isinstance(num, int)
                and num < 0
                and not m1.is_floating_point()
                and not m1.is_complex()
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"Integers to negative integer powers are not allowed\.",
                ):
                    torch.pow(m1[4], num)
            else:
                # base - tensor, exponent - number
                # contiguous
                res1 = torch.pow(m1[4], num)
                res2 = res1.clone().zero_()
                # `math.pow` has issues with complex exponentiation so we need to resort to normal `pow`.
                for i in range(res2.size(0)):
                    res2[i] = pow_fn(m1[4][i], num)
                rtol = 0 if atol is not None else None
                self.assertEqual(res1, res2, atol=atol, rtol=rtol)

                # non-contiguous
                res1 = torch.pow(m1[:, 4], num)
                res2 = res1.clone().zero_()
                for i in range(res2.size(0)):
                    res2[i] = pow_fn(m1[i, 4], num)
                self.assertEqual(res1, res2, atol=atol, rtol=rtol)

                # scalar ** tensor to enforce correct handling of dtypes for __rpow__().
                expected_dtype = torch.result_type(num, m1)
                res1 = num ** m1[4]
                res2 = (
                    torch.tensor(num, dtype=expected_dtype, device=m1.device) ** m1[4]
                )
                self.assertEqual(res1, res2)
                self.assertEqual(res1.dtype, expected_dtype)

    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_pow(self, device, dtype):
        m1 = torch.empty(0, dtype=dtype, device=device)
        if m1.is_floating_point() or m1.is_complex():
            m1 = (
                make_tensor((100, 100), low=0, high=1, dtype=dtype, device=device) + 0.5
            )
        else:
            # math.pow will overflow and throw exceptions for large integers
            range_high = 4 if dtype in (torch.int8, torch.uint8) else 10
            m1 = make_tensor(
                (100, 100), low=1, high=range_high, dtype=dtype, device=device
            )

        exponents = [-2.8, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 3.3, True, False]
        complex_exponents = [
            -2.5j,
            -1.0j,
            0j,
            1.0j,
            2.5j,
            1.0 + 1.0j,
            -1.0 - 1.5j,
            3.3j,
        ]
        if m1.is_complex():
            self._do_pow_for_exponents(m1, exponents + complex_exponents, pow, 10e-4)
        else:
            self._do_pow_for_exponents(m1, exponents, math.pow, None)
            will_raise_error = (
                dtype is torch.half and torch.device(device).type == "cpu"
            )
            if will_raise_error:
                # On CPU,
                # Half Tensor with complex exponents leads to computation dtype
                # of ComplexHalf for which this ops is not supported yet
                with self.assertRaisesRegex(
                    RuntimeError, "not implemented for 'ComplexHalf'"
                ):
                    self._do_pow_for_exponents(m1, complex_exponents, pow, 10e-4)
            else:
                self._do_pow_for_exponents(m1, complex_exponents, pow, 10e-4)

        # base - number, exponent - tensor
        # contiguous
        res1 = torch.pow(3, m1[4])
        res2 = res1.clone().zero_()
        for i in range(res2.size(0)):
            res2[i] = pow(3, m1[4, i])
        self.assertEqual(res1, res2)

        # non-contiguous
        res1 = torch.pow(3, m1[:, 4])
        res2 = res1.clone().zero_()
        for i in range(res2.size(0)):
            res2[i] = pow(3, m1[i][4])
        self.assertEqual(res1, res2)

    # TODO: refactor all these tests using opinfos properly
    def _test_pow(self, base, exponent, np_exponent=None):
        if np_exponent is None:
            np_exponent = exponent

        def to_np(value):
            if isinstance(value, torch.Tensor):
                return value.cpu().numpy()
            return value

        try:
            np_res = np.power(to_np(base), to_np(np_exponent))
            expected = (
                torch.from_numpy(np_res)
                if isinstance(np_res, np.ndarray)
                else torch.tensor(np_res, dtype=base.dtype)
            )
        except ValueError as e:
            err_msg = "Integers to negative integer powers are not allowed."
            self.assertEqual(str(e), err_msg)
            out = torch.empty_like(base)
            test_cases = [
                lambda: base.pow(exponent),
                lambda: base.pow_(exponent),
                lambda: torch.pow(base, exponent),
                lambda: torch.pow(base, exponent, out=out),
            ]
            for test_case in test_cases:
                self.assertRaisesRegex(RuntimeError, err_msg, test_case)
        else:
            if isinstance(base, torch.Tensor):
                actual = base.pow(exponent)
                self.assertEqual(actual, expected.to(actual))
                actual = base.clone()
                # When base is a 0-dim cpu tensor and exp is a cuda tensor, we exp `pow` to work but `pow_` to fail, since
                # `pow` will try to create the output tensor on a cuda device, but `pow_` needs to use the cpu tensor as the output
                if (
                    isinstance(exponent, torch.Tensor)
                    and base.dim() == 0
                    and base.device.type == "cpu"
                    and exponent.device.type == "cuda"
                ):
                    regex = "Expected all tensors to be on the same device, but found at least two devices, cuda.* and cpu!"
                    self.assertRaisesRegex(RuntimeError, regex, base.pow_, exponent)
                elif torch.can_cast(torch.result_type(base, exponent), base.dtype):
                    actual2 = actual.pow_(exponent)
                    self.assertEqual(actual, expected.to(actual))
                    self.assertEqual(actual2, expected.to(actual2))
                else:
                    self.assertRaisesRegex(
                        RuntimeError,
                        r"result type \w+ can't be cast to the desired output type \w+",
                        lambda: actual.pow_(exponent),
                    )

            actual = torch.pow(base, exponent)
            self.assertEqual(actual, expected.to(actual))

            actual2 = torch.pow(base, exponent, out=actual)
            self.assertEqual(actual, expected.to(actual))
            self.assertEqual(actual2, expected.to(actual))

    # We can potentially merge this into OpInfo, but one blocker is that the
    # first input must be a scalar. It is not as simple as just wrapping this in
    # a lambada that switches the inputs, because we also want to test samples inputs
    # where the second input is a scalar. The wrapper would need some more logic.
    def test_pow_scalar_base(self, device):
        a = (
            torch.arange(1, 13, dtype=torch.double, device=device)
            .view(3, 4)
            .requires_grad_()
        )
        gradcheck(lambda a: torch.pow(2, a), (a,))

    # Tests pow() for integral, floating-type tensors, with integral, floating-type
    # exponents (tensor or scalar), respectively. noncontiguous tensors are also tested.
    def test_int_and_float_pow(self, device):
        def _test_int_and_float_pow(dt, low, high, dev):
            test_cases = (
                ((4, 4), 0, (4, 1)),
                ((3, 1), 4, (3, 1)),
                ((2,), 4, (1,)),
                ((1,), 2, ()),
                ((513, 513), 4, (513,)),
                ((5, 5, 5), 5, (5,)),
                ((), 2, ()),
            )
            for base_shape, exp_scalar, exp_shape in test_cases:
                base_tensor = make_tensor(
                    base_shape, dtype=dt, device=dev, low=low, high=high
                )
                # int tensors don't take negative exponents
                if dt in [
                    torch.uint8,
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                ]:
                    exp_tensor = make_tensor(
                        exp_shape, dtype=dt, device=dev, low=0, high=high
                    )
                else:
                    exp_tensor = make_tensor(
                        exp_shape, dtype=dt, device=dev, low=low, high=high
                    )
                self._test_pow(base_tensor, exp_scalar)
                self._test_pow(base_tensor, exp_tensor)
                # test non-contiguous tensors as well
                base_tensor = make_tensor(
                    base_shape,
                    dtype=dt,
                    device=dev,
                    low=low,
                    high=high,
                    noncontiguous=True,
                )
                if dt in [
                    torch.uint8,
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                ]:
                    exp_tensor = make_tensor(
                        exp_shape,
                        dtype=dt,
                        device=dev,
                        low=0,
                        high=high,
                        noncontiguous=True,
                    )
                else:
                    exp_tensor = make_tensor(
                        exp_shape,
                        dtype=dt,
                        device=dev,
                        low=low,
                        high=high,
                        noncontiguous=True,
                    )
                self._test_pow(base_tensor, exp_scalar)
                self._test_pow(base_tensor, exp_tensor)

        _test_int_and_float_pow(torch.int8, -2, 2, device)
        _test_int_and_float_pow(torch.uint8, 0, 3, device)
        _test_int_and_float_pow(torch.int16, -5, 5, device)
        _test_int_and_float_pow(torch.int64, -10, 10, device)
        _test_int_and_float_pow(torch.int32, -10, 10, device)
        _test_int_and_float_pow(torch.float16, 0.0, 5.0, device)
        _test_int_and_float_pow(torch.float32, 0.0, 10.0, device)
        _test_int_and_float_pow(torch.float64, 0.0, 10.0, device)
        # pow's output would have some NaNs as well
        _test_int_and_float_pow(torch.float32, -10.0, 10.0, device)
        _test_int_and_float_pow(torch.float64, -10.0, 10.0, device)

    # Tests that a Runtime error occurs when a base tensor cannot be resized
    # by pow's inplace variant due to PyTorch's broadcasting semantics.
    def test_pow_inplace_resizing_exception(self, device):
        test_cases = (
            ((), (3,)),
            ((2,), (2, 1)),
            ((2, 1), (2, 2)),
            ((2, 2), (2, 1, 1)),
        )
        test_inputs = [
            (
                make_tensor(
                    base_size, dtype=torch.float64, device=device, high=10.0, low=0.0
                ),
                make_tensor(
                    exp_size, dtype=torch.float64, device=device, high=10.0, low=0.0
                ),
            )
            for base_size, exp_size in test_cases
        ]
        for base, exponent in test_inputs:
            regex = "doesn't match the broadcast shape"
            self.assertRaisesRegex(RuntimeError, regex, base.pow_, exponent)

    def test_int_tensor_pow_neg_ints(self, device):
        ints = [
            torch.iinfo(torch.int32).min,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            torch.iinfo(torch.int32).max,
        ]
        neg_ints = [torch.iinfo(torch.int32).min, -3, -2, -1]
        tensor = torch.tensor(ints, dtype=torch.int32, device=device)
        for pow in neg_ints:
            self._test_pow(tensor, pow)

    def test_long_tensor_pow_floats(self, device):
        ints = [0, 1, 23, 4567]
        floats = [0.0, 1 / 3, 1 / 2, 1.0, 3 / 2, 2.0]
        tensor = torch.tensor(ints, dtype=torch.int64, device=device)
        for pow in floats:
            self._test_pow(tensor, pow)

    @dtypes(*[torch.float32, torch.float64])
    def test_float_scalar_pow_float_tensor(self, device, dtype):
        floats = [2.0, -3 / 2, -1.0, -1 / 2, -1 / 3, 0.0, 1 / 3, 1 / 2, 1.0, 3 / 2, 2.0]
        exponent_shapes = (
            (1,),
            (2, 2),
            (2, 1),
            (2, 2, 2),
        )
        tensors = [
            make_tensor(shape, dtype=dtype, device=device, low=0)
            for shape in exponent_shapes
        ]
        floats_tensor = torch.tensor(floats, dtype=dtype, device=device)
        for base in floats:
            self._test_pow(base, floats_tensor)
            for tensor in tensors:
                self._test_pow(base, tensor)

    @onlyCUDA
    def test_cuda_tensor_pow_scalar_tensor(self, device):
        cuda_tensors = [
            torch.randn((3, 3), device=device),
            torch.tensor(3.0, device=device),
        ]
        scalar_tensors = [
            torch.tensor(5.0, device="cpu"),
            torch.tensor(-3),
            torch.tensor(1),
        ]
        for base, exp in product(cuda_tensors, scalar_tensors):
            self._test_pow(base, exp)

    @onlyCUDA
    def test_cpu_tensor_pow_cuda_scalar_tensor(self, device):
        cuda_tensors = [
            torch.tensor(5.0, device="cuda"),
            torch.tensor(-3, device="cuda"),
        ]
        for exp in cuda_tensors:
            base = torch.randn((3, 3), device="cpu")
            regex = "Expected all tensors to be on the same device, but found at least two devices, cuda.* and cpu!"
            self.assertRaisesRegex(RuntimeError, regex, torch.pow, base, exp)
        for exp in cuda_tensors:
            # Binary ops with a cpu + cuda tensor are allowed if the cpu tensor has 0 dimension
            base = torch.tensor(3.0, device="cpu")
            self._test_pow(base, exp)

    @onlyCUDA
    @dtypes(torch.complex64, torch.complex128)
    def test_pow_cuda_complex_extremal_passing(self, device, dtype):
        t = torch.tensor(complex(-1.0, float("inf")), dtype=dtype, device=device)
        cuda_out = t.pow(2)
        cpu_out = t.cpu().pow(2)
        self.assertEqual(cpu_out, cuda_out)

    @skipIfTorchDynamo()
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half))
    def test_complex_scalar_pow_tensor(self, device, dtype):
        complexes = [0.5j, 1.0 + 1.0j, -1.5j, 2.2 - 1.6j, 1 + 0j]
        first_exp = make_tensor((100,), dtype=dtype, device=device, low=-2, high=2)
        second_exp = make_tensor(
            (100,), dtype=dtype, device=device, low=-2, high=2, noncontiguous=True
        )
        first_exp[0] = first_exp[10] = first_exp[20] = 0
        second_exp[0] = second_exp[10] = second_exp[20] = 0
        for base in complexes:
            # On CPU,
            # Half Tensor with complex base leads to computation dtype
            # of ComplexHalf for which this ops is not supported yet
            # NOTE: pow has fast-path when base is 1 which supports
            # ComplexHalf
            will_raise_error = (
                torch.device(device).type == "cpu"
                and dtype is torch.half
                and base != (1 + 0j)
            )
            if will_raise_error:
                with self.assertRaisesRegex(
                    RuntimeError, "not implemented for 'ComplexHalf'"
                ):
                    self._test_pow(base, first_exp)
                    self._test_pow(base, second_exp)
            else:
                self._test_pow(base, first_exp)
                self._test_pow(base, second_exp)

    @onlyNativeDeviceTypes
    @skipMeta
    def test_pow_scalar_type_promotion(self, device):
        # Test against a scalar and non-scalar input
        inputs = [17, [17]]
        for input in inputs:
            # We expect the computation to be performed in uint8 (overflowing to 0), and then cast to int64
            input_tensor_uint8 = torch.tensor(input, dtype=torch.uint8, device=device)
            out_uint8_computation = torch.pow(
                2,
                input_tensor_uint8,
                out=torch.tensor(0, dtype=torch.int64, device=device),
            )

            # Computation should run in int64, and not overflow
            input_tensor_int64 = torch.tensor(input, dtype=torch.int64, device=device)
            out_int64_computation = torch.pow(
                2,
                input_tensor_int64,
                out=torch.tensor(0, dtype=torch.int64, device=device),
            )

            self.assertNotEqual(out_uint8_computation, out_int64_computation)
            self.assertEqual(
                out_uint8_computation.to(dtype=torch.uint8),
                out_int64_computation.to(dtype=torch.uint8),
            )

    def test_tensor_pow_tensor(self, device):
        def rotate(l, n):
            return l[-n:] + l[:-n]

        def test_tensor_pow_tensor(values, torch_type, numpy_type):
            vals_tensor = torch.tensor(values, dtype=torch_type, device=device)
            for i in range(len(values)):
                pows = rotate(values, i)
                pows_tensor = torch.tensor(pows, dtype=torch_type, device=device)
                self._test_pow(vals_tensor, pows_tensor)

        ints = [0, 1, 2, 3]
        test_tensor_pow_tensor(ints, torch.uint8, np.uint8)
        test_tensor_pow_tensor(ints, torch.int8, np.int8)
        test_tensor_pow_tensor(ints, torch.int16, np.int16)
        test_tensor_pow_tensor(ints, torch.int32, np.int32)
        test_tensor_pow_tensor(ints, torch.int64, np.int64)

        floats = [-3.0, -2.0, -1.0, -1 / 2, -1 / 3, 0.0, 1 / 3, 1 / 2, 1.0, 2.0, 3.0]
        test_tensor_pow_tensor(floats, torch.float16, np.float16)
        test_tensor_pow_tensor(floats, torch.float32, np.float32)
        test_tensor_pow_tensor(floats, torch.float64, np.float64)

    def test_logical_xor_with_nontrivial_alignment(self, device):
        # test tensor that is not aligned to multiple of 16 bytes
        size = 128
        a = torch.randn(size, device=device) > 0
        b = torch.randn(size, device=device) > 0
        c = torch.randn(size, device=device) > 0
        non_trivial_alignment = [1, 2, 4, 8, 15]
        for i in non_trivial_alignment:
            for j in non_trivial_alignment:
                for k in non_trivial_alignment:
                    a_ = a[i : 100 + i]
                    b_ = b[j : 100 + j]
                    c_ = c[k : 100 + k]
                    torch.logical_xor(a_, b_, out=c_)
                    for x, y, z in zip(a_.tolist(), b_.tolist(), c_.tolist()):
                        self.assertEqual(x ^ y, z)

    @dtypes(torch.float)
    def test_add_with_tail(self, device, dtype):
        # test tensor where there is a tail which is not a multiple
        # of GPU warp size
        for tail_size in [1, 63, 67, 130]:
            size = 4096 + tail_size
            a = torch.randn(size, device=device, dtype=dtype)
            b = torch.randn(size, device=device, dtype=dtype)
            c = a + b
            for x, y, z in zip(a.tolist(), b.tolist(), c.tolist()):
                self.assertEqual(x + y, z)

    # Tests that CUDA tensors on different devices cannot be used in the same
    # binary operation, and that CUDA "scalars" cannot be used in the same
    # binary operation as non-scalar CPU tensors.
    @deviceCountAtLeast(2)
    @onlyCUDA
    def test_cross_device_binary_ops(self, devices):
        vals = (1.0, (2.0,))
        cpu_tensor = torch.randn(2, 2)

        def do_test(op, a, b):
            with self.assertRaisesRegex(RuntimeError, "Expected all tensors.+"):
                op(a, b)
            with self.assertRaisesRegex(RuntimeError, "Expected all tensors.+"):
                op(b, a)
            with self.assertRaisesRegex(RuntimeError, "Expected all tensors.+"):
                op(a, cpu_tensor)
            with self.assertRaisesRegex(RuntimeError, "Expected all tensors.+"):
                op(cpu_tensor, a)

        for op in (
            operator.add,
            torch.add,
            operator.sub,
            torch.sub,
            operator.mul,
            torch.mul,
            operator.truediv,
            torch.true_divide,
            operator.floordiv,
            torch.floor_divide,
        ):
            for a, b in product(vals, vals):
                a = torch.tensor(a, device=devices[0])
                b = torch.tensor(b, device=devices[1])

            do_test(op, a, b)

    # This test ensures that a scalar Tensor can be safely used
    # in a binary operation in conjunction with a Tensor on all
    # available CUDA devices
    @deviceCountAtLeast(2)
    @onlyCUDA
    def test_binary_op_scalar_device_unspecified(self, devices):
        scalar_val = torch.tensor(1.0)
        for default_device in devices:
            with torch.cuda.device(default_device):
                for device in devices:
                    device_obj = torch.device(device)
                    x = torch.rand(3, device=device)
                    y0 = x * scalar_val
                    self.assertEqual(y0.device, device_obj)
                    y1 = scalar_val * x
                    self.assertEqual(y1.device, device_obj)
                    self.assertEqual(y0, y1)

    def test_div_and_floordiv_vs_python(self, device):
        # Tests torch division ops which can handle both arguments being
        #   scalars.
        def _scalar_helper(python_op, torch_op):
            for a, b in product(range(-10, 10), range(-10, 10)):
                for op in (lambda x: x * 0.5, lambda x: math.floor(x)):
                    a = op(a)
                    b = op(b)

                    # Skips zero divisors
                    if b == 0:
                        continue

                    expected = python_op(a, b)

                    actual_scalar = torch_op(a, b)

                    a_t = torch.tensor(a, device=device)
                    b_t = torch.tensor(b, device=device)

                    actual_tensor = torch_op(a_t, b_t)
                    actual_first_tensor = torch_op(a_t, b)
                    actual_second_tensor = torch_op(a, b_t)

                    self.assertEqual(actual_scalar, expected)
                    self.assertEqual(actual_tensor.item(), expected)
                    self.assertEqual(actual_first_tensor, actual_tensor)
                    self.assertEqual(actual_second_tensor, actual_tensor)

        _scalar_helper(operator.truediv, operator.truediv)
        _scalar_helper(operator.truediv, torch.true_divide)
        _scalar_helper(lambda a, b: math.floor(a / b), operator.floordiv)
        _scalar_helper(lambda a, b: math.floor(a / b), torch.floor_divide)

    @onlyNativeDeviceTypes
    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_div_and_floordiv_script_vs_python(self, device):
        # Creates jitted functions of two tensors
        def _wrapped_div(a, b):
            return a / b

        def _wrapped_floordiv(a, b):
            return a // b

        scripted_div = torch.jit.script(_wrapped_div)
        scripted_floordiv = torch.jit.script(_wrapped_floordiv)
        for a, b in product(range(-10, 10), range(-10, 10)):
            for op in (lambda x: x * 0.5, lambda x: math.floor(x)):
                a = op(a)
                b = op(b)

                # Skips zero divisors
                if b == 0:
                    continue

                expected_div = a / b
                expected_floordiv = math.floor(a / b)
                a_t = torch.tensor(a, device=device)
                b_t = torch.tensor(b, device=device)

                self.assertEqual(scripted_div(a_t, b_t), expected_div)
                self.assertEqual(scripted_floordiv(a_t, b_t), expected_floordiv)

        # Creates jitted functions of one tensor
        def _wrapped_div_scalar(a):
            return a / 5

        # NOTE: the JIT implements division as torch.reciprocal(a) * 5
        def _wrapped_rdiv_scalar(a):
            return 5 / a

        def _wrapped_floordiv_scalar(a):
            return a // 5

        # NOTE: this fails if the input is not an integer tensor
        # See https://github.com/pytorch/pytorch/issues/45199
        def _wrapped_rfloordiv_scalar(a):
            return 5 // a

        scripted_div_scalar = torch.jit.script(_wrapped_div_scalar)
        scripted_rdiv_scalar = torch.jit.script(_wrapped_rdiv_scalar)
        scripted_floordiv_scalar = torch.jit.script(_wrapped_floordiv_scalar)
        scripted_rfloordiv_scalar = torch.jit.script(_wrapped_rfloordiv_scalar)

        for a in range(-10, 10):
            for op in (lambda x: x * 0.5, lambda x: math.floor(x)):
                a = op(a)

                a_t = torch.tensor(a, device=device)

                self.assertEqual(a / 5, scripted_div_scalar(a_t))

                # Skips zero divisors
                if a == 0:
                    continue

                self.assertEqual(5 / a, scripted_rdiv_scalar(a_t))

                # Handles Issue 45199 (see comment above)
                if a_t.is_floating_point():
                    with self.assertRaises(RuntimeError):
                        scripted_rfloordiv_scalar(a_t)
                else:
                    # This should emit a UserWarning, why doesn't it?
                    # See issue gh-52387
                    self.assertEqual(5 // a, scripted_rfloordiv_scalar(a_t))

    @onlyNativeDeviceTypes
    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_idiv_and_ifloordiv_vs_python(self, device):
        def _wrapped_idiv_tensor(a, b):
            a /= b
            return a

        def _wrapped_idiv_scalar(a):
            a /= 5
            return a

        def _wrapped_true_divide__tensor(a, b):
            a.true_divide_(b)
            return a

        def _wrapped_true_divide__scalar(a):
            a.true_divide_(5)
            return a

        def _wrapped_floor_divide__tensor(a, b):
            a.floor_divide_(b)
            return a

        def _wrapped_floor_divide__scalar(a):
            a.floor_divide_(5)
            return a

        # The following functions are unsupported by the JIT
        def _wrapped_ifloordiv_tensor(a, b):
            a //= b
            return a

        def _wrapped_ifloordiv_scalar(a):
            a //= 5
            return a

        with self.assertRaises(torch.jit.frontend.NotSupportedError):
            scripted_ifloordiv_tensor = torch.jit.script(_wrapped_ifloordiv_tensor)

        with self.assertRaises(torch.jit.frontend.NotSupportedError):
            scripted_ifloordiv_scalar = torch.jit.script(_wrapped_ifloordiv_scalar)

        scripted_idiv_tensor = torch.jit.script(_wrapped_idiv_tensor)
        scripted_idiv_scalar = torch.jit.script(_wrapped_idiv_scalar)
        scripted_true_divide__tensor = torch.jit.script(_wrapped_true_divide__tensor)
        scripted_true_divide__scalar = torch.jit.script(_wrapped_true_divide__scalar)
        scripted_floor_divide__tensor = torch.jit.script(_wrapped_floor_divide__tensor)
        scripted_floor_divide__scalar = torch.jit.script(_wrapped_floor_divide__scalar)

        for a, b in product(range(-10, 10), range(-10, 10)):
            for op in (lambda x: x * 0.5, lambda x: math.floor(x)):
                a = op(a)
                b = op(b)

                # Skips zero divisors
                if b == 0:
                    continue

                expected_idiv = a / b
                expected_ifloordiv = a // b

                a_t = torch.tensor(a, device=device)
                b_t = torch.tensor(b, device=device)

                if a_t.is_floating_point():
                    tmp0 = a_t.clone()
                    tmp0 /= b

                    tmp1 = a_t.clone()
                    tmp1 /= b_t

                    self.assertEqual(tmp0.item(), expected_idiv)
                    self.assertEqual(tmp1.item(), expected_idiv)
                    self.assertEqual(
                        scripted_true_divide__tensor(a_t.clone(), b_t).item(),
                        expected_idiv,
                    )
                    self.assertEqual(
                        scripted_true_divide__scalar(a_t.clone()).item(), a / 5
                    )
                else:
                    tmp = a_t.clone()
                    with self.assertRaises(RuntimeError):
                        tmp /= b
                    with self.assertRaises(RuntimeError):
                        tmp /= b_t
                    with self.assertRaises(RuntimeError):
                        scripted_true_divide__tensor(tmp, b_t)
                    with self.assertRaises(RuntimeError):
                        scripted_true_divide__scalar(tmp)

                if not a_t.is_floating_point() and b_t.is_floating_point():
                    # Inplace modification fails because a float tensor is required
                    #   if the divisor is a float tensor
                    a_t.clone().floor_divide_(b_t)
                    scripted_floor_divide__tensor(a_t.clone(), b_t)
                    tmp = a_t.clone()
                    tmp //= b_t
                else:
                    # Inplace modification is OK when both or neither tensor is
                    #   a float tensor
                    self.assertEqual(
                        a_t.clone().floor_divide_(b_t).item(), expected_ifloordiv
                    )
                    self.assertEqual(
                        scripted_floor_divide__tensor(a_t.clone(), b_t).item(),
                        expected_ifloordiv,
                    )
                    tmp = a_t.clone()
                    tmp //= b_t
                    self.assertEqual(tmp.item(), expected_ifloordiv)

                self.assertEqual(scripted_floor_divide__scalar(a_t), math.floor(a / 5))

    # Tests binary op equivalence with Python builtin ops
    # Also tests that reverse operations are equivalent to forward ops
    # NOTE: division ops are tested separately above
    def test_binary_ops_with_scalars(self, device):
        for python_op, torch_op in (
            (operator.add, torch.add),
            (operator.sub, torch.sub),
            (operator.mul, torch.mul),
            (operator.truediv, torch.div),
        ):
            for a, b in product(range(-10, 10), range(-10, 10)):
                for op in (lambda x: x * 0.5, lambda x: math.floor(x)):
                    a = op(a)
                    b = op(b)

                    # Skips zero divisors
                    if b == 0 or a == 0:
                        continue

                    a_tensor = torch.tensor(a, device=device)
                    b_tensor = torch.tensor(b, device=device)
                    a_tensor_cpu = a_tensor.cpu()
                    b_tensor_cpu = b_tensor.cpu()
                    vals = (a, b, a_tensor, b_tensor, a_tensor_cpu, b_tensor_cpu)

                    for args in product(vals, vals):
                        first, second = args

                        first_scalar = (
                            first
                            if not isinstance(first, torch.Tensor)
                            else first.item()
                        )
                        second_scalar = (
                            second
                            if not isinstance(second, torch.Tensor)
                            else second.item()
                        )
                        expected = python_op(first_scalar, second_scalar)

                        self.assertEqual(expected, python_op(first, second))
                        self.assertEqual(expected, torch_op(first, second))

    @dtypes(
        *product(
            all_types_and(torch.half, torch.bfloat16, torch.bool),
            all_types_and(torch.half, torch.bfloat16, torch.bool),
        )
    )
    def test_maximum_minimum_type_promotion(self, device, dtypes):
        a = torch.tensor((0, 1), device=device, dtype=dtypes[0])
        b = torch.tensor((1, 0), device=device, dtype=dtypes[1])
        for op in (
            torch.maximum,
            torch.max,
            torch.fmax,
            torch.minimum,
            torch.min,
            torch.fmin,
        ):
            result = op(a, b)
            self.assertEqual(result.dtype, torch.result_type(a, b))

    @dtypes(*integral_types_and(torch.bool))
    def test_maximum_minimum_int_and_bool(self, device, dtype):
        ops = (
            (torch.maximum, torch.max, np.maximum),
            (torch.minimum, torch.min, np.minimum),
            (torch.fmax, None, np.fmax),
            (torch.fmin, None, np.fmin),
        )
        rng = np.random.default_rng()
        a_np = np.array(
            rng.integers(-100, 100, size=10), dtype=torch_to_numpy_dtype_dict[dtype]
        )
        b_np = np.array(
            rng.integers(-100, 100, size=10), dtype=torch_to_numpy_dtype_dict[dtype]
        )

        for torch_op, alias, numpy_op in ops:
            a_tensor = torch.from_numpy(a_np).to(device=device, dtype=dtype)
            b_tensor = torch.from_numpy(b_np).to(device=device, dtype=dtype)
            tensor_result = torch_op(a_tensor, b_tensor)

            out = torch.empty_like(a_tensor)
            torch_op(a_tensor, b_tensor, out=out)

            numpy_result = numpy_op(a_np, b_np)

            if alias is not None:
                alias_result = alias(a_tensor, b_tensor)
                self.assertEqual(alias_result, tensor_result)

            self.assertEqual(tensor_result, numpy_result)
            self.assertEqual(out, numpy_result)

    @precisionOverride({torch.bfloat16: 1e-2})
    @dtypes(*(floating_types_and(torch.half, torch.bfloat16)))
    def test_maximum_minimum_float(self, device, dtype):
        ops = (
            (torch.maximum, torch.max, np.maximum),
            (torch.minimum, torch.min, np.minimum),
            (torch.fmax, None, np.fmax),
            (torch.fmin, None, np.fmin),
        )

        if dtype == torch.bfloat16:
            a_np = np.random.randn(10).astype(np.float64)
            b_np = np.random.randn(10).astype(np.float64)
        else:
            a_np = np.random.randn(10).astype(torch_to_numpy_dtype_dict[dtype])
            b_np = np.random.randn(10).astype(torch_to_numpy_dtype_dict[dtype])

        for torch_op, alias, numpy_op in ops:
            numpy_result = numpy_op(a_np, b_np)

            a_tensor = torch.from_numpy(a_np).to(device=device, dtype=dtype)
            b_tensor = torch.from_numpy(b_np).to(device=device, dtype=dtype)
            tensor_result = torch_op(a_tensor, b_tensor)
            out = torch.empty_like(a_tensor)
            torch_op(a_tensor, b_tensor, out=out)

            if alias is not None:
                alias_result = alias(a_tensor, b_tensor)
                self.assertEqual(alias_result, tensor_result, exact_dtype=False)

            self.assertEqual(tensor_result, numpy_result, exact_dtype=False)
            self.assertEqual(out, numpy_result, exact_dtype=False)

    @dtypes(*(floating_types_and(torch.half, torch.bfloat16)))
    def test_maximum_minimum_float_nan_and_inf(self, device, dtype):
        # np.maximum and np.minimum functions compare input arrays element-wisely.
        # if one of the elements being compared is a NaN, then that element is returned.
        ops = (
            (torch.maximum, torch.max, np.maximum),
            (torch.minimum, torch.min, np.minimum),
            (torch.fmax, None, np.fmax),
            (torch.fmin, None, np.fmin),
        )
        a_vals = (
            float("inf"),
            -float("inf"),
            float("nan"),
            float("inf"),
            float("nan"),
            float("nan"),
            1,
            float("nan"),
        )
        b_vals = (
            -float("inf"),
            float("inf"),
            float("inf"),
            float("nan"),
            float("nan"),
            0,
            float("nan"),
            -5,
        )
        if dtype == torch.bfloat16:
            a_np = np.array(a_vals, dtype=np.float64)
            b_np = np.array(b_vals, dtype=np.float64)
        else:
            a_np = np.array(a_vals, dtype=torch_to_numpy_dtype_dict[dtype])
            b_np = np.array(b_vals, dtype=torch_to_numpy_dtype_dict[dtype])

        for torch_op, alias, numpy_op in ops:
            numpy_result = numpy_op(a_np, b_np)

            a_tensor = torch.from_numpy(a_np).to(device=device, dtype=dtype)
            b_tensor = torch.from_numpy(b_np).to(device=device, dtype=dtype)
            tensor_result = torch_op(a_tensor, b_tensor)

            out = torch.empty_like(a_tensor)
            torch_op(a_tensor, b_tensor, out=out)

            if alias is not None:
                alias_result = alias(a_tensor, b_tensor)
                self.assertEqual(alias_result, tensor_result)

            if dtype == torch.bfloat16:
                self.assertEqual(tensor_result, numpy_result, exact_dtype=False)
                self.assertEqual(out, numpy_result, exact_dtype=False)
            else:
                self.assertEqual(tensor_result, numpy_result)
                self.assertEqual(out, numpy_result)

    @dtypes(
        *product(
            complex_types(),
            all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool),
        )
    )
    def test_maximum_minimum_complex(self, device, dtypes):
        for torch_op in (
            torch.maximum,
            torch.minimum,
            torch.max,
            torch.min,
            torch.fmax,
            torch.fmin,
        ):
            with self.assertRaisesRegex(RuntimeError, ".+not implemented for.+"):
                torch_op(
                    torch.ones(1, device=device, dtype=dtypes[0]),
                    torch.ones(1, device=device, dtype=dtypes[1]),
                )

            with self.assertRaisesRegex(RuntimeError, ".+not implemented for.+"):
                torch_op(
                    torch.ones(1, device=device, dtype=dtypes[1]),
                    torch.ones(1, device=device, dtype=dtypes[0]),
                )

    @onlyCUDA
    def test_maximum_minimum_cross_device(self, device):
        a = torch.tensor((1, 2, -1))
        b = torch.tensor((3, 0, 4), device=device)
        ops = (torch.maximum, torch.minimum)

        for torch_op in ops:
            with self.assertRaisesRegex(
                RuntimeError, "Expected all tensors to be on the same device"
            ):
                torch_op(a, b)

            with self.assertRaisesRegex(
                RuntimeError, "Expected all tensors to be on the same device"
            ):
                torch_op(b, a)

        # test cuda tensor and cpu scalar
        ops = ((torch.maximum, np.maximum), (torch.minimum, np.minimum))
        a_np = np.array(1)
        b_np = np.array([3, 0, 4])

        for torch_op, numpy_op in ops:
            a_tensor = torch.from_numpy(a_np)
            b_tensor = torch.from_numpy(b_np).to(device=device)
            tensor_result_1 = torch_op(a_tensor, b_tensor)
            numpy_result_1 = numpy_op(a_np, b_np)
            tensor_result_2 = torch_op(b_tensor, a_tensor)
            numpy_result_2 = numpy_op(b_np, a_np)

            self.assertEqual(tensor_result_1, numpy_result_1)
            self.assertEqual(tensor_result_2, numpy_result_2)

    @dtypes(
        *product(
            floating_types_and(torch.half, torch.bfloat16),
            floating_types_and(torch.half, torch.bfloat16),
        )
    )
    def test_maximum_and_minimum_subgradient(self, device, dtypes):
        def run_test(f, a, b, expected_a_grad, expected_b_grad):
            a = torch.tensor(a, requires_grad=True, device=device, dtype=dtypes[0])
            b = torch.tensor(b, requires_grad=True, device=device, dtype=dtypes[1])
            z = f(a, b)
            z.sum().backward()
            self.assertEqual(a.grad, expected_a_grad)
            self.assertEqual(b.grad, expected_b_grad)

        run_test(
            torch.maximum,
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.5, 1.0],
            [1.0, 0.5, 0.0],
        )
        run_test(
            torch.minimum,
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.5, 0.0],
            [0.0, 0.5, 1.0],
        )

    def test_maximum_minimum_forward_ad_float32(self, device):
        # TODO: This should really be covered by OpInfo but it isn't. The problem
        # is that our gradient tests test using float64 but it should also test
        # float32
        x = torch.randn(3, device=device, dtype=torch.float32)
        y = torch.randn(3, device=device, dtype=torch.float32)
        tx = torch.randn(3, device=device, dtype=torch.float32)
        ty = torch.randn(3, device=device, dtype=torch.float32)

        with fwAD.dual_level():
            x_dual = fwAD.make_dual(x, tx)
            y_dual = fwAD.make_dual(y, ty)
            result = torch.maximum(x_dual, y_dual)
            _, result_tangent = fwAD.unpack_dual(result)

        expected = torch.where(x > y, tx, ty)
        self.assertEqual(result_tangent, expected)

        with fwAD.dual_level():
            x_dual = fwAD.make_dual(x, tx)
            y_dual = fwAD.make_dual(y, ty)
            result = torch.minimum(x_dual, y_dual)
            _, result_tangent = fwAD.unpack_dual(result)

        expected = torch.where(x < y, tx, ty)
        self.assertEqual(result_tangent, expected)

    # TODO: tests like this should be generic
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_mul_intertype_scalar(self, device, dtype):
        x = torch.tensor(1.5, dtype=dtype, device=device)
        y = torch.tensor(3, dtype=torch.int32, device=device)

        self.assertEqual(x * y, 4.5)
        self.assertEqual(y * x, 4.5)

        with self.assertRaisesRegex(
            RuntimeError, "can't be cast to the desired output type"
        ):
            y *= x
        x *= y
        self.assertEqual(x, 4.5)

    @onlyCPU
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_sub(self, device, dtype):
        if dtype in integral_types():
            # Before Python 3.10, floats were implicitly converted to ints, but with
            #   DeprecationWarning: an integer is required (got type float).
            #   Implicit conversion to integers using __int__ is deprecated,
            #   and may be removed in a future version of Python.
            # Since Python 3.10, that attempt gives an error.
            m1 = torch.tensor([2, 4], dtype=dtype, device=device)
            m2 = torch.tensor([1, 2], dtype=dtype, device=device)
            diff = torch.tensor([1, 2], dtype=dtype)
        else:
            m1 = torch.tensor([2.34, 4.44], dtype=dtype, device=device)
            m2 = torch.tensor([1.23, 2.33], dtype=dtype, device=device)
            diff = torch.tensor([1.11, 2.11], dtype=dtype)

        if dtype == torch.bool:
            self.assertRaises(RuntimeError, lambda: m1 - m2)
        elif dtype == torch.bfloat16 or dtype == torch.half:
            # bfloat16 has a lower precision so we have to have a separate check for it
            self.assertEqual(m1 - m2, diff, atol=0.01, rtol=0)
        else:
            self.assertEqual(m1 - m2, diff)

    # TODO: what is this test testing?
    @onlyCPU
    @dtypes(torch.float)
    def test_csub(self, device, dtype):
        # with a tensor
        a = torch.randn(100, 90, dtype=dtype, device=device)
        b = a.clone().normal_()

        res_add = torch.add(a, b, alpha=-1)
        res_csub = a.clone()
        res_csub.sub_(b)
        self.assertEqual(res_add, res_csub)

        # with a scalar
        a = torch.randn(100, 100, dtype=dtype, device=device)

        scalar = 123.5
        res_add = torch.add(a, -scalar)
        res_csub = a.clone()
        res_csub.sub_(scalar)
        self.assertEqual(res_add, res_csub)

    # TODO: reconcile with minimum/maximum tests
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_min_max_binary_op_nan(self, device, dtype):
        a = torch.rand(1000, dtype=dtype, device=device)
        b = torch.rand(1000, dtype=dtype, device=device)

        # 0:250: a -- nan, b -- not nan
        a[:250] = float("nan")
        # 250:500: a -- not nan, b -- nan
        b[250:500] = float("nan")
        # 500:750: a and b both nan
        a[500:750] = float("nan")
        b[500:750] = float("nan")
        # 750:1000: neither nan

        ma = torch.max(a, b)
        mi = torch.min(a, b)

        for i in range(750):
            self.assertTrue(
                torch.isnan(ma[i]),
                f"max(a, b): {ma[i]}, a: {a[i]}, b: {b[i]}",
            )
            self.assertTrue(
                torch.isnan(mi[i]),
                f"min(a, b): {mi[i]}, a: {a[i]}, b: {b[i]}",
            )

        for i in range(750, 1000):
            self.assertFalse(
                torch.isnan(ma[i]),
                f"max(a, b): {ma[i]}, a: {a[i]}, b: {b[i]}",
            )
            self.assertFalse(
                torch.isnan(mi[i]),
                f"min(a, b): {mi[i]}, a: {a[i]}, b: {b[i]}",
            )

    @dtypes(
        *product(
            all_types_and(torch.half, torch.bfloat16, torch.bool),
            all_types_and(torch.half, torch.bfloat16, torch.bool),
        )
    )
    def test_copysign(self, device, dtypes):
        def _test_copysign_numpy(a, b):
            torch_result = torch.copysign(a, b)

            if a.dtype == torch.bfloat16:
                np_a = a.to(torch.float).cpu().numpy()
            else:
                np_a = a.cpu().numpy()

            if b.dtype == torch.bfloat16:
                np_b = b.to(torch.float).cpu().numpy()
            else:
                np_b = b.cpu().numpy()
            expected = torch.from_numpy(np.copysign(np_a, np_b))
            # To handle inconsistencies of type promotion between PyTorch and Numpy
            # Applied for both arguments having integral precision and bfloat16
            types = integral_types_and(torch.bool, torch.bfloat16)
            if a.dtype in types or b.dtype in types:
                promoted_type = torch.promote_types(torch_result.dtype, expected.dtype)
                torch_result = torch_result.to(promoted_type)
                expected = expected.to(promoted_type)

            # Verify Value
            self.assertEqual(torch_result, expected)
            # Verify Sign
            # Use double copysign to verify the correctness of 0.0 and -0.0, since
            # it always True for self.assertEqual(0.0 == -0.0). So, we use 1 as the
            # magnitude to verify the sign between torch and numpy results, elementwise.
            # Special case: NaN conversions between FP32 and FP16 is not bitwise
            # equivalent to pass this assertion.
            if a.dtype != torch.float16 and b.dtype != torch.float16:
                self.assertEqual(
                    torch.copysign(torch.tensor(1.0), torch_result),
                    torch.copysign(torch.tensor(1.0), expected),
                )

        # Compare Result with NumPy
        # Type promotion
  

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 2 class(es): TestBinaryUfuncs, UnknownType

### Functions
This file defines 185 function(s): assertEqualHelper, _test_reference_numerics, _helper_reference_numerics, _numel, test_reference_numerics, test_reference_numerics_small_values, test_reference_numerics_large_values, test_reference_numerics_extremal_values, test_broadcasting, test_scalar_support, test_contig_vs_every_other, test_contig_vs_transposed, test_non_contig, test_non_contig_index, test_non_contig_expand, test_contig_size1, test_contig_size1_large_dim, test_batch_vs_slicing, test_type_promotion, _supported, test_not_broadcastable, test_add_broadcast_empty, test_addcmul_scalars_as_floats, test_bitwise_ops, test_inplace_division, test_div_rounding_modes, test_floor_div_extremal, test_div_rounding_nonfinite, test_divide_by_zero_rounding, test_div_rounding_numpy


## Key Components

The file contains 15024 words across 4580 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 183538 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
