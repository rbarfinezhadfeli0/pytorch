# Documentation: `test/torch_np/numpy_tests/core/test_numeric.py`

## File Metadata

- **Path**: `test/torch_np/numpy_tests/core/test_numeric.py`
- **Size**: 109,619 bytes (107.05 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]

import functools
import itertools
import math
import platform
import sys
import warnings

import numpy
import pytest


IS_WASM = False
HAS_REFCOUNT = True

import operator
from unittest import expectedFailure as xfail, skipIf as skipif, SkipTest

from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
from pytest import raises as assert_raises

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xfailIfTorchDynamo,
    xpassIfTorchDynamo_np,
)


# If we are going to trace through these, we should use NumPy
# If testing on eager mode, we use torch._numpy
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.random import rand, randint, randn
    from numpy.testing import (
        assert_,
        assert_allclose,
        assert_almost_equal,
        assert_array_almost_equal,
        assert_array_equal,
        assert_equal,
        assert_warns,  # assert_array_max_ulp, HAS_REFCOUNT, IS_WASM
    )
else:
    import torch._numpy as np
    from torch._numpy.random import rand, randint, randn
    from torch._numpy.testing import (
        assert_,
        assert_allclose,
        assert_almost_equal,
        assert_array_almost_equal,
        assert_array_equal,
        assert_equal,
        assert_warns,  # assert_array_max_ulp, HAS_REFCOUNT, IS_WASM
    )


skip = functools.partial(skipif, True)


@instantiate_parametrized_tests
class TestResize(TestCase):
    def test_copies(self):
        A = np.array([[1, 2], [3, 4]])
        Ar1 = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        assert_equal(np.resize(A, (2, 4)), Ar1)

        Ar2 = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
        assert_equal(np.resize(A, (4, 2)), Ar2)

        Ar3 = np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4]])
        assert_equal(np.resize(A, (4, 3)), Ar3)

    def test_repeats(self):
        A = np.array([1, 2, 3])
        Ar1 = np.array([[1, 2, 3, 1], [2, 3, 1, 2]])
        assert_equal(np.resize(A, (2, 4)), Ar1)

        Ar2 = np.array([[1, 2], [3, 1], [2, 3], [1, 2]])
        assert_equal(np.resize(A, (4, 2)), Ar2)

        Ar3 = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
        assert_equal(np.resize(A, (4, 3)), Ar3)

    def test_zeroresize(self):
        A = np.array([[1, 2], [3, 4]])
        Ar = np.resize(A, (0,))
        assert_array_equal(Ar, np.array([]))
        assert_equal(A.dtype, Ar.dtype)

        Ar = np.resize(A, (0, 2))
        assert_equal(Ar.shape, (0, 2))

        Ar = np.resize(A, (2, 0))
        assert_equal(Ar.shape, (2, 0))

    def test_reshape_from_zero(self):
        # See also gh-6740
        A = np.zeros(0, dtype=np.float32)
        Ar = np.resize(A, (2, 1))
        assert_array_equal(Ar, np.zeros((2, 1), Ar.dtype))
        assert_equal(A.dtype, Ar.dtype)

    def test_negative_resize(self):
        A = np.arange(0, 10, dtype=np.float32)
        new_shape = (-10, -1)
        with pytest.raises((RuntimeError, ValueError)):
            np.resize(A, new_shape=new_shape)


@instantiate_parametrized_tests
class TestNonarrayArgs(TestCase):
    # check that non-array arguments to functions wrap them in arrays
    def test_choose(self):
        choices = [[0, 1, 2], [3, 4, 5], [5, 6, 7]]
        tgt = [5, 1, 5]
        a = [2, 0, 1]

        out = np.choose(a, choices)
        assert_equal(out, tgt)

    def test_clip(self):
        arr = [-1, 5, 2, 3, 10, -4, -9]
        out = np.clip(arr, 2, 7)
        tgt = [2, 5, 2, 3, 7, 2, 2]
        assert_equal(out, tgt)

    @xpassIfTorchDynamo_np  # (reason="TODO implement compress(...)")
    def test_compress(self):
        arr = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        tgt = [[5, 6, 7, 8, 9]]
        out = np.compress([0, 1], arr, axis=0)
        assert_equal(out, tgt)

    def test_count_nonzero(self):
        arr = [[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]]
        tgt = np.array([2, 3])
        out = np.count_nonzero(arr, axis=1)
        assert_equal(out, tgt)

    def test_cumproduct(self):
        A = [[1, 2, 3], [4, 5, 6]]
        assert_(np.all(np.cumprod(A) == np.array([1, 2, 6, 24, 120, 720])))

    def test_diagonal(self):
        a = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
        out = np.diagonal(a)
        tgt = [0, 5, 10]

        assert_equal(out, tgt)

    def test_mean(self):
        A = [[1, 2, 3], [4, 5, 6]]
        assert_(np.mean(A) == 3.5)
        assert_(np.all(np.mean(A, 0) == np.array([2.5, 3.5, 4.5])))
        assert_(np.all(np.mean(A, 1) == np.array([2.0, 5.0])))

        #    with warnings.catch_warnings(record=True) as w:
        #        warnings.filterwarnings('always', '', RuntimeWarning)
        assert_(np.isnan(np.mean([])))

    #        assert_(w[0].category is RuntimeWarning)

    def test_ptp(self):
        a = [3, 4, 5, 10, -3, -5, 6.0]
        assert_equal(np.ptp(a, axis=0), 15.0)

    def test_prod(self):
        arr = [[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]]
        tgt = [24, 1890, 600]

        assert_equal(np.prod(arr, axis=-1), tgt)

    def test_ravel(self):
        a = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        tgt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert_equal(np.ravel(a), tgt)

    def test_repeat(self):
        a = [1, 2, 3]
        tgt = [1, 1, 2, 2, 3, 3]

        out = np.repeat(a, 2)
        assert_equal(out, tgt)

    def test_reshape(self):
        arr = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        tgt = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        assert_equal(np.reshape(arr, (2, 6)), tgt)

    def test_round(self):
        arr = [1.56, 72.54, 6.35, 3.25]
        tgt = [1.6, 72.5, 6.4, 3.2]
        assert_equal(np.around(arr, decimals=1), tgt)
        s = np.float64(1.0)
        assert_equal(s.round(), 1.0)

    def test_round_2(self):
        s = np.float64(1.0)
        assert_(isinstance(s.round(), (np.float64, np.ndarray)))

    @xpassIfTorchDynamo_np  # (reason="scalar instances")
    @parametrize(
        "dtype",
        [
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.float16,
            np.float32,
            np.float64,
        ],
    )
    def test_dunder_round(self, dtype):
        s = dtype(1)
        assert_(isinstance(round(s), int))
        assert_(isinstance(round(s, None), int))
        assert_(isinstance(round(s, ndigits=None), int))
        assert_equal(round(s), 1)
        assert_equal(round(s, None), 1)
        assert_equal(round(s, ndigits=None), 1)

    @parametrize(
        "val, ndigits",
        [
            # pytest.param(
            #    2**31 - 1, -1, marks=pytest.mark.xfail(reason="Out of range of int32")
            # ),
            subtest((2**31 - 1, -1), decorators=[xpassIfTorchDynamo_np]),
            subtest(
                (2**31 - 1, 1 - math.ceil(math.log10(2**31 - 1))),
                decorators=[xpassIfTorchDynamo_np],
            ),
            subtest(
                (2**31 - 1, -math.ceil(math.log10(2**31 - 1))),
                decorators=[xpassIfTorchDynamo_np],
            ),
        ],
    )
    def test_dunder_round_edgecases(self, val, ndigits):
        assert_equal(round(val, ndigits), round(np.int32(val), ndigits))

    @xfail  # (reason="scalar instances")
    def test_dunder_round_accuracy(self):
        f = np.float64(5.1 * 10**73)
        assert_(isinstance(round(f, -73), np.float64))
        assert_array_max_ulp(round(f, -73), 5.0 * 10**73)
        assert_(isinstance(round(f, ndigits=-73), np.float64))
        assert_array_max_ulp(round(f, ndigits=-73), 5.0 * 10**73)

        i = np.int64(501)
        assert_(isinstance(round(i, -2), np.int64))
        assert_array_max_ulp(round(i, -2), 500)
        assert_(isinstance(round(i, ndigits=-2), np.int64))
        assert_array_max_ulp(round(i, ndigits=-2), 500)

    @xfail  # (raises=AssertionError, reason="gh-15896")
    def test_round_py_consistency(self):
        f = 5.1 * 10**73
        assert_equal(round(np.float64(f), -73), round(f, -73))

    def test_searchsorted(self):
        arr = [-8, -5, -1, 3, 6, 10]
        out = np.searchsorted(arr, 0)
        assert_equal(out, 3)

    def test_size(self):
        A = [[1, 2, 3], [4, 5, 6]]
        assert_(np.size(A) == 6)
        assert_(np.size(A, 0) == 2)
        assert_(np.size(A, 1) == 3)

    def test_squeeze(self):
        A = [[[1, 1, 1], [2, 2, 2], [3, 3, 3]]]
        assert_equal(np.squeeze(A).shape, (3, 3))
        assert_equal(np.squeeze(np.zeros((1, 3, 1))).shape, (3,))
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=0).shape, (3, 1))
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=-1).shape, (1, 3))
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=2).shape, (1, 3))
        assert_equal(np.squeeze([np.zeros((3, 1))]).shape, (3,))
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=0).shape, (3, 1))
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=2).shape, (1, 3))
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=-1).shape, (1, 3))

    def test_std(self):
        A = [[1, 2, 3], [4, 5, 6]]
        assert_almost_equal(np.std(A), 1.707825127659933)
        assert_almost_equal(np.std(A, 0), np.array([1.5, 1.5, 1.5]))
        assert_almost_equal(np.std(A, 1), np.array([0.81649658, 0.81649658]))

        #  with warnings.catch_warnings(record=True) as w:
        #      warnings.filterwarnings('always', '', RuntimeWarning)
        assert_(np.isnan(np.std([])))

    #      assert_(w[0].category is RuntimeWarning)

    def test_swapaxes(self):
        tgt = [[[0, 4], [2, 6]], [[1, 5], [3, 7]]]
        a = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
        out = np.swapaxes(a, 0, 2)
        assert_equal(out, tgt)

    def test_sum(self):
        m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        tgt = [[6], [15], [24]]
        out = np.sum(m, axis=1, keepdims=True)

        assert_equal(tgt, out)

    def test_take(self):
        tgt = [2, 3, 5]
        indices = [1, 2, 4]
        a = [1, 2, 3, 4, 5]

        out = np.take(a, indices)
        assert_equal(out, tgt)

    def test_trace(self):
        c = [[1, 2], [3, 4], [5, 6]]
        assert_equal(np.trace(c), 5)

    def test_transpose(self):
        arr = [[1, 2], [3, 4], [5, 6]]
        tgt = [[1, 3, 5], [2, 4, 6]]
        assert_equal(np.transpose(arr, (1, 0)), tgt)

    def test_var(self):
        A = [[1, 2, 3], [4, 5, 6]]
        assert_almost_equal(np.var(A), 2.9166666666666665)
        assert_almost_equal(np.var(A, 0), np.array([2.25, 2.25, 2.25]))
        assert_almost_equal(np.var(A, 1), np.array([0.66666667, 0.66666667]))

        #  with warnings.catch_warnings(record=True) as w:
        #      warnings.filterwarnings('always', '', RuntimeWarning)
        assert_(np.isnan(np.var([])))

    #      assert_(w[0].category is RuntimeWarning)


@xfail  # (reason="TODO")
class TestIsscalar(TestCase):
    def test_isscalar(self):
        assert_(np.isscalar(3.1))
        assert_(np.isscalar(np.int16(12345)))
        assert_(np.isscalar(False))
        assert_(np.isscalar("numpy"))
        assert_(not np.isscalar([3.1]))
        assert_(not np.isscalar(None))

        # PEP 3141
        from fractions import Fraction

        assert_(np.isscalar(Fraction(5, 17)))
        from numbers import Number

        assert_(np.isscalar(Number()))


class TestBoolScalar(TestCase):
    def test_logical(self):
        f = np.False_
        t = np.True_
        s = "xyz"
        assert_((t and s) is s)
        assert_((f and s) is f)

    def test_bitwise_or_eq(self):
        f = np.False_
        t = np.True_
        assert_((t | t) == t)
        assert_((f | t) == t)
        assert_((t | f) == t)
        assert_((f | f) == f)

    def test_bitwise_or_is(self):
        f = np.False_
        t = np.True_
        assert_(bool(t | t) is bool(t))
        assert_(bool(f | t) is bool(t))
        assert_(bool(t | f) is bool(t))
        assert_(bool(f | f) is bool(f))

    def test_bitwise_and_eq(self):
        f = np.False_
        t = np.True_
        assert_((t & t) == t)
        assert_((f & t) == f)
        assert_((t & f) == f)
        assert_((f & f) == f)

    def test_bitwise_and_is(self):
        f = np.False_
        t = np.True_
        assert_(bool(t & t) is bool(t))
        assert_(bool(f & t) is bool(f))
        assert_(bool(t & f) is bool(f))
        assert_(bool(f & f) is bool(f))

    def test_bitwise_xor_eq(self):
        f = np.False_
        t = np.True_
        assert_((t ^ t) == f)
        assert_((f ^ t) == t)
        assert_((t ^ f) == t)
        assert_((f ^ f) == f)

    def test_bitwise_xor_is(self):
        f = np.False_
        t = np.True_
        assert_(bool(t ^ t) is bool(f))
        assert_(bool(f ^ t) is bool(t))
        assert_(bool(t ^ f) is bool(t))
        assert_(bool(f ^ f) is bool(f))


class TestBoolArray(TestCase):
    def setUp(self):
        super().setUp()
        # offset for simd tests
        self.t = np.array([True] * 41, dtype=bool)[1::]
        self.f = np.array([False] * 41, dtype=bool)[1::]
        self.o = np.array([False] * 42, dtype=bool)[2::]
        self.nm = self.f.copy()
        self.im = self.t.copy()
        self.nm[3] = True
        self.nm[-2] = True
        self.im[3] = False
        self.im[-2] = False

    def test_all_any(self):
        assert_(self.t.all())
        assert_(self.t.any())
        assert_(not self.f.all())
        assert_(not self.f.any())
        assert_(self.nm.any())
        assert_(self.im.any())
        assert_(not self.nm.all())
        assert_(not self.im.all())
        # check bad element in all positions
        for i in range(256 - 7):
            d = np.array([False] * 256, dtype=bool)[7::]
            d[i] = True
            assert_(np.any(d))
            e = np.array([True] * 256, dtype=bool)[7::]
            e[i] = False
            assert_(not np.all(e))
            assert_array_equal(e, ~d)
        # big array test for blocked libc loops
        for i in list(range(9, 6000, 507)) + [7764, 90021, -10]:
            d = np.array([False] * 100043, dtype=bool)
            d[i] = True
            assert_(np.any(d), msg=f"{i!r}")
            e = np.array([True] * 100043, dtype=bool)
            e[i] = False
            assert_(not np.all(e), msg=f"{i!r}")

    def test_logical_not_abs(self):
        assert_array_equal(~self.t, self.f)
        assert_array_equal(np.abs(~self.t), self.f)
        assert_array_equal(np.abs(~self.f), self.t)
        assert_array_equal(np.abs(self.f), self.f)
        assert_array_equal(~np.abs(self.f), self.t)
        assert_array_equal(~np.abs(self.t), self.f)
        assert_array_equal(np.abs(~self.nm), self.im)
        np.logical_not(self.t, out=self.o)
        assert_array_equal(self.o, self.f)
        np.abs(self.t, out=self.o)
        assert_array_equal(self.o, self.t)

    def test_logical_and_or_xor(self):
        assert_array_equal(self.t | self.t, self.t)
        assert_array_equal(self.f | self.f, self.f)
        assert_array_equal(self.t | self.f, self.t)
        assert_array_equal(self.f | self.t, self.t)
        np.logical_or(self.t, self.t, out=self.o)
        assert_array_equal(self.o, self.t)
        assert_array_equal(self.t & self.t, self.t)
        assert_array_equal(self.f & self.f, self.f)
        assert_array_equal(self.t & self.f, self.f)
        assert_array_equal(self.f & self.t, self.f)
        np.logical_and(self.t, self.t, out=self.o)
        assert_array_equal(self.o, self.t)
        assert_array_equal(self.t ^ self.t, self.f)
        assert_array_equal(self.f ^ self.f, self.f)
        assert_array_equal(self.t ^ self.f, self.t)
        assert_array_equal(self.f ^ self.t, self.t)
        np.logical_xor(self.t, self.t, out=self.o)
        assert_array_equal(self.o, self.f)

        assert_array_equal(self.nm & self.t, self.nm)
        assert_array_equal(self.im & self.f, False)
        assert_array_equal(self.nm & True, self.nm)
        assert_array_equal(self.im & False, self.f)
        assert_array_equal(self.nm | self.t, self.t)
        assert_array_equal(self.im | self.f, self.im)
        assert_array_equal(self.nm | True, self.t)
        assert_array_equal(self.im | False, self.im)
        assert_array_equal(self.nm ^ self.t, self.im)
        assert_array_equal(self.im ^ self.f, self.im)
        assert_array_equal(self.nm ^ True, self.im)
        assert_array_equal(self.im ^ False, self.im)


@xfailIfTorchDynamo
class TestBoolCmp(TestCase):
    def setUp(self):
        super().setUp()
        self.f = np.ones(256, dtype=np.float32)
        self.ef = np.ones(self.f.size, dtype=bool)
        self.d = np.ones(128, dtype=np.float64)
        self.ed = np.ones(self.d.size, dtype=bool)
        # generate values for all permutation of 256bit simd vectors
        s = 0
        for i in range(32):
            self.f[s : s + 8] = [i & 2**x for x in range(8)]
            self.ef[s : s + 8] = [(i & 2**x) != 0 for x in range(8)]
            s += 8
        s = 0
        for i in range(16):
            self.d[s : s + 4] = [i & 2**x for x in range(4)]
            self.ed[s : s + 4] = [(i & 2**x) != 0 for x in range(4)]
            s += 4

        self.nf = self.f.copy()
        self.nd = self.d.copy()
        self.nf[self.ef] = np.nan
        self.nd[self.ed] = np.nan

        self.inff = self.f.copy()
        self.infd = self.d.copy()
        self.inff[::3][self.ef[::3]] = np.inf
        self.infd[::3][self.ed[::3]] = np.inf
        self.inff[1::3][self.ef[1::3]] = -np.inf
        self.infd[1::3][self.ed[1::3]] = -np.inf
        self.inff[2::3][self.ef[2::3]] = np.nan
        self.infd[2::3][self.ed[2::3]] = np.nan
        self.efnonan = self.ef.copy()
        self.efnonan[2::3] = False
        self.ednonan = self.ed.copy()
        self.ednonan[2::3] = False

        self.signf = self.f.copy()
        self.signd = self.d.copy()
        self.signf[self.ef] *= -1.0
        self.signd[self.ed] *= -1.0
        self.signf[1::6][self.ef[1::6]] = -np.inf
        self.signd[1::6][self.ed[1::6]] = -np.inf
        self.signf[3::6][self.ef[3::6]] = -np.nan
        self.signd[3::6][self.ed[3::6]] = -np.nan
        self.signf[4::6][self.ef[4::6]] = -0.0
        self.signd[4::6][self.ed[4::6]] = -0.0

    def test_float(self):
        # offset for alignment test
        for i in range(4):
            assert_array_equal(self.f[i:] > 0, self.ef[i:])
            assert_array_equal(self.f[i:] - 1 >= 0, self.ef[i:])
            assert_array_equal(self.f[i:] == 0, ~self.ef[i:])
            assert_array_equal(-self.f[i:] < 0, self.ef[i:])
            assert_array_equal(-self.f[i:] + 1 <= 0, self.ef[i:])
            r = self.f[i:] != 0
            assert_array_equal(r, self.ef[i:])
            r2 = self.f[i:] != np.zeros_like(self.f[i:])
            r3 = 0 != self.f[i:]
            assert_array_equal(r, r2)
            assert_array_equal(r, r3)
            # check bool == 0x1
            assert_array_equal(r.view(np.int8), r.astype(np.int8))
            assert_array_equal(r2.view(np.int8), r2.astype(np.int8))
            assert_array_equal(r3.view(np.int8), r3.astype(np.int8))

            # isnan on amd64 takes the same code path
            assert_array_equal(np.isnan(self.nf[i:]), self.ef[i:])
            assert_array_equal(np.isfinite(self.nf[i:]), ~self.ef[i:])
            assert_array_equal(np.isfinite(self.inff[i:]), ~self.ef[i:])
            assert_array_equal(np.isinf(self.inff[i:]), self.efnonan[i:])
            assert_array_equal(np.signbit(self.signf[i:]), self.ef[i:])

    def test_double(self):
        # offset for alignment test
        for i in range(2):
            assert_array_equal(self.d[i:] > 0, self.ed[i:])
            assert_array_equal(self.d[i:] - 1 >= 0, self.ed[i:])
            assert_array_equal(self.d[i:] == 0, ~self.ed[i:])
            assert_array_equal(-self.d[i:] < 0, self.ed[i:])
            assert_array_equal(-self.d[i:] + 1 <= 0, self.ed[i:])
            r = self.d[i:] != 0
            assert_array_equal(r, self.ed[i:])
            r2 = self.d[i:] != np.zeros_like(self.d[i:])
            r3 = 0 != self.d[i:]
            assert_array_equal(r, r2)
            assert_array_equal(r, r3)
            # check bool == 0x1
            assert_array_equal(r.view(np.int8), r.astype(np.int8))
            assert_array_equal(r2.view(np.int8), r2.astype(np.int8))
            assert_array_equal(r3.view(np.int8), r3.astype(np.int8))

            # isnan on amd64 takes the same code path
            assert_array_equal(np.isnan(self.nd[i:]), self.ed[i:])
            assert_array_equal(np.isfinite(self.nd[i:]), ~self.ed[i:])
            assert_array_equal(np.isfinite(self.infd[i:]), ~self.ed[i:])
            assert_array_equal(np.isinf(self.infd[i:]), self.ednonan[i:])
            assert_array_equal(np.signbit(self.signd[i:]), self.ed[i:])


@xpassIfTorchDynamo_np  # (reason="TODO")
class TestSeterr(TestCase):
    def test_default(self):
        err = np.geterr()
        assert_equal(
            err, dict(divide="warn", invalid="warn", over="warn", under="ignore")
        )

    def test_set(self):
        err = np.seterr()
        old = np.seterr(divide="print")
        assert_(err == old)
        new = np.seterr()
        assert_(new["divide"] == "print")
        np.seterr(over="raise")
        assert_(np.geterr()["over"] == "raise")
        assert_(new["divide"] == "print")
        np.seterr(**old)
        assert_(np.geterr() == old)

    @xfail
    @skipif(IS_WASM, reason="no wasm fp exception support")
    @skipif(platform.machine() == "armv5tel", reason="See gh-413.")
    def test_divide_err(self):
        with assert_raises(FloatingPointError):
            np.array([1.0]) / np.array([0.0])

        np.seterr(divide="ignore")
        np.array([1.0]) / np.array([0.0])

    @skipif(IS_WASM, reason="no wasm fp exception support")
    def test_errobj(self):
        olderrobj = np.geterrobj()
        self.called = 0
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                np.seterrobj([20000, 1, None])
                np.array([1.0]) / np.array([0.0])
                assert_equal(len(w), 1)

            def log_err(*args):
                self.called += 1
                extobj_err = args
                assert_(len(extobj_err) == 2)
                assert_("divide" in extobj_err[0])

            np.seterrobj([20000, 3, log_err])
            np.array([1.0]) / np.array([0.0])
            assert_equal(self.called, 1)

            np.seterrobj(olderrobj)
            np.divide(1.0, 0.0, extobj=[20000, 3, log_err])
            assert_equal(self.called, 2)
        finally:
            np.seterrobj(olderrobj)
            del self.called


@xfail  # (reason="TODO")
@instantiate_parametrized_tests
class TestFloatExceptions(TestCase):
    def assert_raises_fpe(self, fpeerr, flop, x, y):
        ftype = type(x)
        try:
            flop(x, y)
            assert_(False, f"Type {ftype} did not raise fpe error '{fpeerr}'.")
        except FloatingPointError as exc:
            assert_(
                str(exc).find(fpeerr) >= 0,
                f"Type {ftype} raised wrong fpe error '{exc}'.",
            )

    def assert_op_raises_fpe(self, fpeerr, flop, sc1, sc2):
        # Check that fpe exception is raised.
        #
        # Given a floating operation `flop` and two scalar values, check that
        # the operation raises the floating point exception specified by
        # `fpeerr`. Tests all variants with 0-d array scalars as well.

        self.assert_raises_fpe(fpeerr, flop, sc1, sc2)
        self.assert_raises_fpe(fpeerr, flop, sc1[()], sc2)
        self.assert_raises_fpe(fpeerr, flop, sc1, sc2[()])
        self.assert_raises_fpe(fpeerr, flop, sc1[()], sc2[()])

    # Test for all real and complex float types
    @skipif(IS_WASM, reason="no wasm fp exception support")
    @parametrize("typecode", np.typecodes["AllFloat"])
    def test_floating_exceptions(self, typecode):
        # Test basic arithmetic function errors
        ftype = np.dtype(typecode).type
        if np.dtype(ftype).kind == "f":
            # Get some extreme values for the type
            fi = np.finfo(ftype)
            ft_tiny = fi._machar.tiny
            ft_max = fi.max
            ft_eps = fi.eps
            underflow = "underflow"
            divbyzero = "divide by zero"
        else:
            # 'c', complex, corresponding real dtype
            rtype = type(ftype(0).real)
            fi = np.finfo(rtype)
            ft_tiny = ftype(fi._machar.tiny)
            ft_max = ftype(fi.max)
            ft_eps = ftype(fi.eps)
            # The complex types raise different exceptions
            underflow = ""
            divbyzero = ""
        overflow = "overflow"
        invalid = "invalid"

        # The value of tiny for double double is NaN, so we need to
        # pass the assert
        if not np.isnan(ft_tiny):
            self.assert_raises_fpe(underflow, operator.truediv, ft_tiny, ft_max)
            self.assert_raises_fpe(underflow, operator.mul, ft_tiny, ft_tiny)
        self.assert_raises_fpe(overflow, operator.mul, ft_max, ftype(2))
        self.assert_raises_fpe(overflow, operator.truediv, ft_max, ftype(0.5))
        self.assert_raises_fpe(overflow, operator.add, ft_max, ft_max * ft_eps)
        self.assert_raises_fpe(overflow, operator.sub, -ft_max, ft_max * ft_eps)
        self.assert_raises_fpe(overflow, np.power, ftype(2), ftype(2**fi.nexp))
        self.assert_raises_fpe(divbyzero, operator.truediv, ftype(1), ftype(0))
        self.assert_raises_fpe(invalid, operator.truediv, ftype(np.inf), ftype(np.inf))
        self.assert_raises_fpe(invalid, operator.truediv, ftype(0), ftype(0))
        self.assert_raises_fpe(invalid, operator.sub, ftype(np.inf), ftype(np.inf))
        self.assert_raises_fpe(invalid, operator.add, ftype(np.inf), ftype(-np.inf))
        self.assert_raises_fpe(invalid, operator.mul, ftype(0), ftype(np.inf))

    @skipif(IS_WASM, reason="no wasm fp exception support")
    def test_warnings(self):
        # test warning code path
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            np.divide(1, 0.0)
            assert_equal(len(w), 1)
            assert_("divide by zero" in str(w[0].message))
            np.array(1e300) * np.array(1e300)
            assert_equal(len(w), 2)
            assert_("overflow" in str(w[-1].message))
            np.array(np.inf) - np.array(np.inf)
            assert_equal(len(w), 3)
            assert_("invalid value" in str(w[-1].message))
            np.array(1e-300) * np.array(1e-300)
            assert_equal(len(w), 4)
            assert_("underflow" in str(w[-1].message))


class TestTypes(TestCase):
    def check_promotion_cases(self, promote_func):
        # tests that the scalars get coerced correctly.
        b = np.bool_(0)
        i8, i16, i32, i64 = np.int8(0), np.int16(0), np.int32(0), np.int64(0)
        u8 = np.uint8(0)
        f32, f64 = np.float32(0), np.float64(0)
        c64, c128 = np.complex64(0), np.complex128(0)

        # coercion within the same kind
        assert_equal(promote_func(i8, i16), np.dtype(np.int16))
        assert_equal(promote_func(i32, i8), np.dtype(np.int32))
        assert_equal(promote_func(i16, i64), np.dtype(np.int64))
        assert_equal(promote_func(f32, f64), np.dtype(np.float64))
        assert_equal(promote_func(c128, c64), np.dtype(np.complex128))

        # coercion between kinds
        assert_equal(promote_func(b, i32), np.dtype(np.int32))
        assert_equal(promote_func(b, u8), np.dtype(np.uint8))
        assert_equal(promote_func(i8, u8), np.dtype(np.int16))
        assert_equal(promote_func(u8, i32), np.dtype(np.int32))

        assert_equal(promote_func(f32, i16), np.dtype(np.float32))
        assert_equal(promote_func(f32, c64), np.dtype(np.complex64))
        assert_equal(promote_func(c128, f32), np.dtype(np.complex128))

        # coercion between scalars and 1-D arrays
        assert_equal(promote_func(np.array([b]), i8), np.dtype(np.int8))
        assert_equal(promote_func(np.array([b]), u8), np.dtype(np.uint8))
        assert_equal(promote_func(np.array([b]), i32), np.dtype(np.int32))
        assert_equal(promote_func(c64, np.array([f64])), np.dtype(np.complex128))
        assert_equal(
            promote_func(np.complex64(3j), np.array([f64])), np.dtype(np.complex128)
        )

        # coercion between scalars and 1-D arrays, where
        # the scalar has greater kind than the array
        assert_equal(promote_func(np.array([b]), f64), np.dtype(np.float64))
        assert_equal(promote_func(np.array([b]), i64), np.dtype(np.int64))
        assert_equal(promote_func(np.array([i8]), f64), np.dtype(np.float64))

    def check_promotion_cases_2(self, promote_func):
        # these are failing because of the "scalars do not upcast arrays" rule
        # Two first tests (i32 + f32 -> f64, and i64+f32 -> f64) xfail
        # until ufuncs implement the proper type promotion (ufunc loops?)
        i8, i32, i64 = np.int8(0), np.int32(0), np.int64(0)
        f32, f64 = np.float32(0), np.float64(0)
        c128 = np.complex128(0)

        assert_equal(promote_func(i32, f32), np.dtype(np.float64))
        assert_equal(promote_func(i64, f32), np.dtype(np.float64))

        assert_equal(promote_func(np.array([i8]), i64), np.dtype(np.int8))
        assert_equal(promote_func(f64, np.array([f32])), np.dtype(np.float32))

        # float and complex are treated as the same "kind" for
        # the purposes of array-scalar promotion, so that you can do
        # (0j + float32array) to get a complex64 array instead of
        # a complex128 array.
        assert_equal(promote_func(np.array([f32]), c128), np.dtype(np.complex64))

    def test_coercion(self):
        def res_type(a, b):
            return np.add(a, b).dtype

        self.check_promotion_cases(res_type)

        # Use-case: float/complex scalar * bool/int8 array
        #           shouldn't narrow the float/complex type
        for a in [np.array([True, False]), np.array([-3, 12], dtype=np.int8)]:
            b = 1.234 * a
            assert_equal(b.dtype, np.dtype("f8"), f"array type {a.dtype}")
            b = np.float64(1.234) * a
            assert_equal(b.dtype, np.dtype("f8"), f"array type {a.dtype}")
            b = np.float32(1.234) * a
            assert_equal(b.dtype, np.dtype("f4"), f"array type {a.dtype}")
            b = np.float16(1.234) * a
            assert_equal(b.dtype, np.dtype("f2"), f"array type {a.dtype}")

            b = 1.234j * a
            assert_equal(b.dtype, np.dtype("c16"), f"array type {a.dtype}")
            b = np.complex128(1.234j) * a
            assert_equal(b.dtype, np.dtype("c16"), f"array type {a.dtype}")
            b = np.complex64(1.234j) * a
            assert_equal(b.dtype, np.dtype("c8"), f"array type {a.dtype}")

        # The following use-case is problematic, and to resolve its
        # tricky side-effects requires more changes.
        #
        # Use-case: (1-t)*a, where 't' is a boolean array and 'a' is
        #            a float32, shouldn't promote to float64
        #
        # a = np.array([1.0, 1.5], dtype=np.float32)
        # t = np.array([True, False])
        # b = t*a
        # assert_equal(b, [1.0, 0.0])
        # assert_equal(b.dtype, np.dtype('f4'))
        # b = (1-t)*a
        # assert_equal(b, [0.0, 1.5])
        # assert_equal(b.dtype, np.dtype('f4'))
        #
        # Probably ~t (bitwise negation) is more proper to use here,
        # but this is arguably less intuitive to understand at a glance, and
        # would fail if 't' is actually an integer array instead of boolean:
        #
        # b = (~t)*a
        # assert_equal(b, [0.0, 1.5])
        # assert_equal(b.dtype, np.dtype('f4'))

    @xpassIfTorchDynamo_np  # (reason="'Scalars do not upcast arrays' rule")
    def test_coercion_2(self):
        def res_type(a, b):
            return np.add(a, b).dtype

        self.check_promotion_cases_2(res_type)

    def test_result_type(self):
        self.check_promotion_cases(np.result_type)

    @skip(reason="array(None) not supported")
    def test_tesult_type_2(self):
        assert_(np.result_type(None) == np.dtype(None))

    @skip(reason="no endianness in dtypes")
    def test_promote_types_endian(self):
        # promote_types should always return native-endian types
        assert_equal(np.promote_types("<i8", "<i8"), np.dtype("i8"))
        assert_equal(np.promote_types(">i8", ">i8"), np.dtype("i8"))

    def test_can_cast(self):
        assert_(np.can_cast(np.int32, np.int64))
        assert_(np.can_cast(np.float64, complex))
        assert_(not np.can_cast(complex, float))

        assert_(np.can_cast("i8", "f8"))
        assert_(not np.can_cast("i8", "f4"))

        assert_(np.can_cast("i8", "i8", "no"))

    @skip(reason="no endianness in dtypes")
    def test_can_cast_2(self):
        assert_(not np.can_cast("<i8", ">i8", "no"))

        assert_(np.can_cast("<i8", ">i8", "equiv"))
        assert_(not np.can_cast("<i4", ">i8", "equiv"))

        assert_(np.can_cast("<i4", ">i8", "safe"))
        assert_(not np.can_cast("<i8", ">i4", "safe"))

        assert_(np.can_cast("<i8", ">i4", "same_kind"))
        assert_(not np.can_cast("<i8", ">u4", "same_kind"))

        assert_(np.can_cast("<i8", ">u4", "unsafe"))

        assert_raises(TypeError, np.can_cast, "i4", None)
        assert_raises(TypeError, np.can_cast, None, "i4")

        # Also test keyword arguments
        assert_(np.can_cast(from_=np.int32, to=np.int64))

    @xpassIfTorchDynamo_np  # (reason="value-based casting?")
    def test_can_cast_values(self):
        # gh-5917
        for dt in [np.int8, np.int16, np.int32, np.int64] + [
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ]:
            ii = np.iinfo(dt)
            assert_(np.can_cast(ii.min, dt))
            assert_(np.can_cast(ii.max, dt))
            assert_(not np.can_cast(ii.min - 1, dt))
            assert_(not np.can_cast(ii.max + 1, dt))

        for dt in [np.float16, np.float32, np.float64, np.longdouble]:
            fi = np.finfo(dt)
            assert_(np.can_cast(fi.min, dt))
            assert_(np.can_cast(fi.max, dt))


# Custom exception class to test exception propagation in fromiter
class NIterError(Exception):
    pass


@skip(reason="NP_VER: fails on CI")
@xpassIfTorchDynamo_np  # (reason="TODO")
@instantiate_parametrized_tests
class TestFromiter(TestCase):
    def makegen(self):
        return (x**2 for x in range(24))

    def test_types(self):
        ai32 = np.fromiter(self.makegen(), np.int32)
        ai64 = np.fromiter(self.makegen(), np.int64)
        af = np.fromiter(self.makegen(), float)
        assert_(ai32.dtype == np.dtype(np.int32))
        assert_(ai64.dtype == np.dtype(np.int64))
        assert_(af.dtype == np.dtype(float))

    def test_lengths(self):
        expected = np.array(list(self.makegen()))
        a = np.fromiter(self.makegen(), int)
        a20 = np.fromiter(self.makegen(), int, 20)
        assert_(len(a) == len(expected))
        assert_(len(a20) == 20)
        assert_raises(ValueError, np.fromiter, self.makegen(), int, len(expected) + 10)

    def test_values(self):
        expected = np.array(list(self.makegen()))
        a = np.fromiter(self.makegen(), int)
        a20 = np.fromiter(self.makegen(), int, 20)
        assert_(np.all(a == expected, axis=0))
        assert_(np.all(a20 == expected[:20], axis=0))

    def load_data(self, n, eindex):
        # Utility method for the issue 2592 tests.
        # Raise an exception at the desired index in the iterator.
        for e in range(n):
            if e == eindex:
                raise NIterError(f"error at index {eindex}")
            yield e

    @parametrize("dtype", [int])
    @parametrize("count, error_index", [(10, 5), (10, 9)])
    def test_2592(self, count, error_index, dtype):
        # Test iteration exceptions are correctly raised. The data/generator
        # has `count` elements but errors at `error_index`
        iterable = self.load_data(count, error_index)
        with pytest.raises(NIterError):
            np.fromiter(iterable, dtype=dtype, count=count)

    @skip(reason="NP_VER: fails on CI")
    def test_empty_result(self):
        class MyIter:
            def __length_hint__(self):
                return 10

            def __iter__(self):
                return iter([])  # actual iterator is empty.

        res = np.fromiter(MyIter(), dtype="d")
        assert res.shape == (0,)
        assert res.dtype == "d"

    def test_too_few_items(self):
        msg = "iterator too short: Expected 10 but iterator had only 3 items."
        with pytest.raises(ValueError, match=msg):
            np.fromiter([1, 2, 3], count=10, dtype=int)

    def test_failed_itemsetting(self):
        with pytest.raises(TypeError):
            np.fromiter([1, None, 3], dtype=int)

        # The following manages to hit somewhat trickier code paths:
        iterable = ((2, 3, 4) for i in range(5))
        with pytest.raises(ValueError):
            np.fromiter(iterable, dtype=np.dtype((int, 2)))


@instantiate_parametrized_tests
class TestNonzeroAndCountNonzero(TestCase):
    def test_count_nonzero_list(self):
        lst = [[0, 1, 2, 3], [1, 0, 0, 6]]
        assert np.count_nonzero(lst) == 5
        assert_array_equal(np.count_nonzero(lst, axis=0), np.array([1, 1, 1, 2]))
        assert_array_equal(np.count_nonzero(lst, axis=1), np.array([3, 2]))

    def test_nonzero_trivial(self):
        assert_equal(np.count_nonzero(np.array([])), 0)
        assert_equal(np.count_nonzero(np.array([], dtype="?")), 0)
        assert_equal(np.nonzero(np.array([])), ([],))

        assert_equal(np.count_nonzero(np.array([0])), 0)
        assert_equal(np.count_nonzero(np.array([0], dtype="?")), 0)
        assert_equal(np.nonzero(np.array([0])), ([],))

        assert_equal(np.count_nonzero(np.array([1])), 1)
        assert_equal(np.count_nonzero(np.array([1], dtype="?")), 1)
        assert_equal(np.nonzero(np.array([1])), ([0],))

    def test_nonzero_trivial_differs(self):
        # numpy returns a python int, we return a 0D array
        assert isinstance(np.count_nonzero([]), np.ndarray)

    def test_nonzero_zerod(self):
        assert_equal(np.count_nonzero(np.array(0)), 0)
        assert_equal(np.count_nonzero(np.array(0, dtype="?")), 0)

        assert_equal(np.count_nonzero(np.array(1)), 1)
        assert_equal(np.count_nonzero(np.array(1, dtype="?")), 1)

    def test_nonzero_zerod_differs(self):
        # numpy returns a python int, we return a 0D array
        assert isinstance(np.count_nonzero(np.array(1)), np.ndarray)

    def test_nonzero_onedim(self):
        x = np.array([1, 0, 2, -1, 0, 0, 8])
        assert_equal(np.count_nonzero(x), 4)
        assert_equal(np.count_nonzero(x), 4)
        assert_equal(np.nonzero(x), ([0, 2, 3, 6],))

    def test_nonzero_onedim_differs(self):
        # numpy returns a python int, we return a 0D array
        x = np.array([1, 0, 2, -1, 0, 0, 8])
        assert isinstance(np.count_nonzero(x), np.ndarray)

    def test_nonzero_twodim(self):
        x = np.array([[0, 1, 0], [2, 0, 3]])
        assert_equal(np.count_nonzero(x.astype("i1")), 3)
        assert_equal(np.count_nonzero(x.astype("i2")), 3)
        assert_equal(np.count_nonzero(x.astype("i4")), 3)
        assert_equal(np.count_nonzero(x.astype("i8")), 3)
        assert_equal(np.nonzero(x), ([0, 1, 1], [1, 0, 2]))

        x = np.eye(3)
        assert_equal(np.count_nonzero(x.astype("i1")), 3)
        assert_equal(np.count_nonzero(x.astype("i2")), 3)
        assert_equal(np.count_nonzero(x.astype("i4")), 3)
        assert_equal(np.count_nonzero(x.astype("i8")), 3)
        assert_equal(np.nonzero(x), ([0, 1, 2], [0, 1, 2]))

    def test_sparse(self):
        # test special sparse condition boolean code path
        for i in range(20):
            c = np.zeros(200, dtype=bool)
            c[i::20] = True
            assert_equal(np.nonzero(c)[0], np.arange(i, 200 + i, 20))

            c = np.zeros(400, dtype=bool)
            c[10 + i : 20 + i] = True
            c[20 + i * 2] = True
            assert_equal(
                np.nonzero(c)[0],
                np.concatenate((np.arange(10 + i, 20 + i), [20 + i * 2])),
            )

    def test_count_nonzero_axis(self):
        # Basic check of functionality
        m = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])

        expected = np.array([1, 1, 1, 1, 1])
        assert_array_equal(np.count_nonzero(m, axis=0), expected)

        expected = np.array([2, 3])
        assert_array_equal(np.count_nonzero(m, axis=1), expected)

        assert isinstance(np.count_nonzero(m, axis=1), np.ndarray)

        assert_raises(ValueError, np.count_nonzero, m, axis=(1, 1))
        assert_raises(TypeError, np.count_nonzero, m, axis="foo")
        assert_raises(np.AxisError, np.count_nonzero, m, axis=3)
        assert_raises(TypeError, np.count_nonzero, m, axis=np.array([[1], [2]]))

    @parametrize("typecode", "efdFDBbhil?")
    def test_count_nonzero_axis_all_dtypes(self, typecode):
        # More thorough test that the axis argument is respected
        # for all dtypes and responds correctly when presented with
        # either integer or tuple arguments for axis

        m = np.zeros((3, 3), dtype=typecode)
        n = np.ones(1, dtype=typecode)

        m[0, 0] = n[0]
        m[1, 0] = n[0]

        expected = np.array([2, 0, 0], dtype=np.intp)
        result = np.count_nonzero(m, axis=0)
        assert_array_equal(result, expected)
        assert expected.dtype == result.dtype

        expected = np.array([1, 1, 0], dtype=np.intp)
        result = np.count_nonzero(m, axis=1)
        assert_array_equal(result, expected)
        assert expected.dtype == result.dtype

        expected = np.array(2)
        assert_array_equal(np.count_nonzero(m, axis=(0, 1)), expected)
        assert_array_equal(np.count_nonzero(m, axis=None), expected)
        assert_array_equal(np.count_nonzero(m), expected)

    def test_countnonzero_axis_empty(self):
        a = np.array([[0, 0, 1], [1, 0, 1]])
        assert_equal(np.count_nonzero(a, axis=()), a.astype(bool))

    def test_countnonzero_keepdims(self):
        a = np.array([[0, 0, 1, 0], [0, 3, 5, 0], [7, 9, 2, 0]])
        assert_array_equal(np.count_nonzero(a, axis=0, keepdims=True), [[1, 2, 3, 0]])
        assert_array_equal(np.count_nonzero(a, axis=1, keepdims=True), [[1], [2], [3]])
        assert_array_equal(np.count_nonzero(a, keepdims=True), [[6]])
        assert isinstance(np.count_nonzero(a, axis=1, keepdims=True), np.ndarray)


class TestIndex(TestCase):
    def test_boolean(self):
        a = rand(3, 5, 8)
        V = rand(5, 8)
        g1 = randint(0, 5, size=15)
        g2 = randint(0, 8, size=15)
        V[g1, g2] = -V[g1, g2]
        assert_(
            (np.array([a[0][V > 0], a[1][V > 0], a[2][V > 0]]) == a[:, V > 0]).all()
        )

    def test_boolean_edgecase(self):
        a = np.array([], dtype="int32")
        b = np.array([], dtype="bool")
        c = a[b]
        assert_equal(c, [])
        assert_equal(c.dtype, np.dtype("int32"))


@xpassIfTorchDynamo_np  # (reason="TODO")
class TestBinaryRepr(TestCase):
    def test_zero(self):
        assert_equal(np.binary_repr(0), "0")

    def test_positive(self):
        assert_equal(np.binary_repr(10), "1010")
        assert_equal(np.binary_repr(12522), "11000011101010")
        assert_equal(np.binary_repr(10736848), "101000111101010011010000")

    def test_negative(self):
        assert_equal(np.binary_repr(-1), "-1")
        assert_equal(np.binary_repr(-10), "-1010")
        assert_equal(np.binary_repr(-12522), "-11000011101010")
        assert_equal(np.binary_repr(-10736848), "-101000111101010011010000")

    def test_sufficient_width(self):
        assert_equal(np.binary_repr(0, width=5), "00000")
        assert_equal(np.binary_repr(10, width=7), "0001010")
        assert_equal(np.binary_repr(-5, width=7), "1111011")

    def test_neg_width_boundaries(self):
        # see gh-8670

        # Ensure that the example in the issue does not
        # break before proceeding to a more thorough test.
        assert_equal(np.binary_repr(-128, width=8), "10000000")

        for width in range(1, 11):
            num = -(2 ** (width - 1))
            exp = "1" + (width - 1) * "0"
            assert_equal(np.binary_repr(num, width=width), exp)

    def test_large_neg_int64(self):
        # See gh-14289.
        assert_equal(np.binary_repr(np.int64(-(2**62)), width=64), "11" + "0" * 62)


@xpassIfTorchDynamo_np  # (reason="TODO")
class TestBaseRepr(TestCase):
    def test_base3(self):
        assert_equal(np.base_repr(3**5, 3), "100000")

    def test_positive(self):
        assert_equal(np.base_repr(12, 10), "12")
        assert_equal(np.base_repr(12, 10, 4), "000012")
        assert_equal(np.base_repr(12, 4), "30")
        assert_equal(np.base_repr(3731624803700888, 36), "10QR0ROFCEW")

    def test_negative(self):
        assert_equal(np.base_repr(-12, 10), "-12")
        assert_equal(np.base_repr(-12, 10, 4), "-000012")
        assert_equal(np.base_repr(-12, 4), "-30")

    def test_base_range(self):
        with assert_raises(ValueError):
            np.base_repr(1, 1)
        with assert_raises(ValueError):
            np.base_repr(1, 37)


class TestArrayComparisons(TestCase):
    def test_array_equal(self):
        res = np.array_equal(np.array([1, 2]), np.array([1, 2]))
        assert_(res)
        assert_(type(res) is bool)
        res = np.array_equal(np.array([1, 2]), np.array([1, 2, 3]))
        assert_(not res)
        assert_(type(res) is bool)
        res = np.array_equal(np.array([1, 2]), np.array([3, 4]))
        assert_(not res)
        assert_(type(res) is bool)
        res = np.array_equal(np.array([1, 2]), np.array([1, 3]))
        assert_(not res)
        assert_(type(res) is bool)

    def test_array_equal_equal_nan(self):
        # Test array_equal with equal_nan kwarg
        a1 = np.array([1, 2, np.nan])
        a2 = np.array([1, np.nan, 2])
        a3 = np.array([1, 2, np.inf])

        # equal_nan=False by default
        assert_(not np.array_equal(a1, a1))
        assert_(np.array_equal(a1, a1, equal_nan=True))
        assert_(not np.array_equal(a1, a2, equal_nan=True))
        # nan's not conflated with inf's
        assert_(not np.array_equal(a1, a3, equal_nan=True))
        # 0-D arrays
        a = np.array(np.nan)
        assert_(not np.array_equal(a, a))
        assert_(np.array_equal(a, a, equal_nan=True))
        # Non-float dtype - equal_nan should have no effect
        a = np.array([1, 2, 3], dtype=int)
        assert_(np.array_equal(a, a))
        assert_(np.array_equal(a, a, equal_nan=True))
        # Multi-dimensional array
        a = np.array([[0, 1], [np.nan, 1]])
        assert_(not np.array_equal(a, a))
        assert_(np.array_equal(a, a, equal_nan=True))
        # Complex values
        a, b = [np.array([1 + 1j])] * 2
        a.real, b.imag = np.nan, np.nan
        assert_(not np.array_equal(a, b, equal_nan=False))
        assert_(np.array_equal(a, b, equal_nan=True))

    def test_none_compares_elementwise(self):
        a = np.ones(3)
        assert_equal(a.__eq__(None), [False, False, False])
        assert_equal(a.__ne__(None), [True, True, True])

    def test_array_equiv(self):
        res = np.array_equiv(np.array([1, 2]), np.array([1, 2]))
        assert_(res)
        assert_(type(res) is bool)
        res = np.array_equiv(np.array([1, 2]), np.array([1, 2, 3]))
        assert_(not res)
        assert_(type(res) is bool)
        res = np.array_equiv(np.array([1, 2]), np.array([3, 4]))
        assert_(not res)
        assert_(type(res) is bool)
        res = np.array_equiv(np.array([1, 2]), np.array([1, 3]))
        assert_(not res)
        assert_(type(res) is bool)

        res = np.array_equiv(np.array([1, 1]), np.array([1]))
        assert_(res)
        assert_(type(res) is bool)
        res = np.array_equiv(np.array([1, 1]), np.array([[1], [1]]))
        assert_(res)
        assert_(type(res) is bool)
        res = np.array_equiv(np.array([1, 2]), np.array([2]))
        assert_(not res)
        assert_(type(res) is bool)
        res = np.array_equiv(np.array([1, 2]), np.array([[1], [2]]))
        assert_(not res)
        assert_(type(res) is bool)
        res = np.array_equiv(
            np.array([1, 2]), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        )
        assert_(not res)
        assert_(type(res) is bool)


@instantiate_parametrized_tests
class TestClip(TestCase):
    def setUp(self):
        super().setUp()
        self.nr = 5
        self.nc = 3

    def fastclip(self, a, m, M, out=None, casting=None):
        if out is None:
            if casting is None:
                return a.clip(m, M)
            else:
                return a.clip(m, M, casting=casting)
        else:
            if casting is None:
                return a.clip(m, M, out)
            else:
                return a.clip(m, M, out, casting=casting)

    def clip(self, a, m, M, out=None):
        # use slow-clip
        selector = np.less(a, m) + 2 * np.greater(a, M)
        return selector.choose((a, m, M), out=out)

    # Handy functions
    def _generate_data(self, n, m):
        return randn(n, m)

    def _generate_data_complex(self, n, m):
        return randn(n, m) + 1.0j * rand(n, m)

    def _generate_flt_data(self, n, m):
        return (randn(n, m)).astype(np.float32)

    def _neg_byteorder(self, a):
        a = np.asarray(a)
        if sys.byteorder == "little":
            a = a.astype(a.dtype.newbyteorder(">"))
        else:
            a = a.astype(a.dtype.newbyteorder("<"))
        retu
```



## High-Level Overview


This Python file contains 39 class(es) and 255 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestResize`, `TestNonarrayArgs`, `TestIsscalar`, `TestBoolScalar`, `TestBoolArray`, `TestBoolCmp`, `TestSeterr`, `TestFloatExceptions`, `TestTypes`, `NIterError`, `TestFromiter`, `MyIter`, `TestNonzeroAndCountNonzero`, `TestIndex`, `TestBinaryRepr`, `TestBaseRepr`, `TestArrayComparisons`, `TestClip`, `TestAllclose`, `TestIsclose`

**Functions defined**: `test_copies`, `test_repeats`, `test_zeroresize`, `test_reshape_from_zero`, `test_negative_resize`, `test_choose`, `test_clip`, `test_compress`, `test_count_nonzero`, `test_cumproduct`, `test_diagonal`, `test_mean`, `test_ptp`, `test_prod`, `test_ravel`, `test_repeat`, `test_reshape`, `test_round`, `test_round_2`, `test_dunder_round`

**Key imports**: functools, itertools, math, platform, sys, warnings, numpy, pytest, operator, expectedFailure as xfail, skipIf as skipif, SkipTest


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/torch_np/numpy_tests/core`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `itertools`
- `math`
- `platform`
- `sys`
- `warnings`
- `numpy`
- `pytest`
- `operator`
- `unittest`: expectedFailure as xfail, skipIf as skipif, SkipTest
- `hypothesis`: given, strategies as st
- `hypothesis.extra`: numpy as hynp
- `numpy as np`
- `numpy.random`: rand, randint, randn
- `torch._numpy as np`
- `torch._numpy.random`: rand, randint, randn
- `fractions`: Fraction
- `numbers`: Number


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/torch_np/numpy_tests/core/test_numeric.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/torch_np/numpy_tests/core`):

- [`test_shape_base.py_docs.md`](./test_shape_base.py_docs.md)
- [`test_scalarinherit.py_docs.md`](./test_scalarinherit.py_docs.md)
- [`test_scalar_methods.py_docs.md`](./test_scalar_methods.py_docs.md)
- [`test_einsum.py_docs.md`](./test_einsum.py_docs.md)
- [`test_indexing.py_docs.md`](./test_indexing.py_docs.md)
- [`test_dlpack.py_docs.md`](./test_dlpack.py_docs.md)
- [`test_getlimits.py_docs.md`](./test_getlimits.py_docs.md)
- [`test_multiarray.py_docs.md`](./test_multiarray.py_docs.md)
- [`test_numerictypes.py_docs.md`](./test_numerictypes.py_docs.md)
- [`test_dtype.py_docs.md`](./test_dtype.py_docs.md)


## Cross-References

- **File Documentation**: `test_numeric.py_docs.md`
- **Keyword Index**: `test_numeric.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
