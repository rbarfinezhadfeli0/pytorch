# Documentation: `test/test_binary_ufuncs.py`

## File Metadata

- **Path**: `test/test_binary_ufuncs.py`
- **Size**: 183,538 bytes (179.24 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
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
        ops =
```



## High-Level Overview


This Python file contains 2 class(es) and 185 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestBinaryUfuncs`, `UnknownType`

**Functions defined**: `assertEqualHelper`, `_test_reference_numerics`, `_helper_reference_numerics`, `_numel`, `test_reference_numerics`, `test_reference_numerics_small_values`, `test_reference_numerics_large_values`, `test_reference_numerics_extremal_values`, `test_broadcasting`, `test_scalar_support`, `test_contig_vs_every_other`, `test_contig_vs_transposed`, `test_non_contig`, `test_non_contig_index`, `test_non_contig_expand`, `test_contig_size1`, `test_contig_size1_large_dim`, `test_batch_vs_slicing`, `test_type_promotion`, `_supported`

**Key imports**: itertools, math, operator, random, sys, warnings, partial, chain, product, Number, numpy as np


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `itertools`
- `math`
- `operator`
- `random`
- `sys`
- `warnings`
- `functools`: partial
- `numbers`: Number
- `numpy as np`
- `torch`
- `torch.autograd.forward_ad as fwAD`
- `torch.testing`: make_tensor
- `scipy.integrate`
- `scipy.special`


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling
- **Automatic Differentiation**: Uses autograd for gradient computation


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

This is a test file. Run it with:

```bash
python test/test_binary_ufuncs.py
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

- **File Documentation**: `test_binary_ufuncs.py_docs.md`
- **Keyword Index**: `test_binary_ufuncs.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
