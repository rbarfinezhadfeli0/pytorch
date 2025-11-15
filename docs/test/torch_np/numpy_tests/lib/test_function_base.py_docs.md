# Documentation: test_function_base.py

## File Metadata
- **Path**: `test/torch_np/numpy_tests/lib/test_function_base.py`
- **Size**: 138973 bytes
- **Lines**: 3911
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# Owner(s): ["module: dynamo"]

import functools
import math
import operator
import sys
import warnings
from fractions import Fraction
from unittest import expectedFailure as xfail, skipIf as skipif

import hypothesis
import hypothesis.strategies as st
import numpy
import pytest
from hypothesis.extra.numpy import arrays
from pytest import raises as assert_raises

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    subtest,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xpassIfTorchDynamo_np,
)


skip = functools.partial(skipif, True)

HAS_REFCOUNT = True
IS_WASM = False
IS_PYPY = False

import string


# FIXME: make from torch._numpy
# These are commented, as if they are imported, some of the tests pass for the wrong reasons
# from numpy lib import digitize, piecewise, trapz, select, trim_zeros, interp
# FIXME: broken on numpy 2.0+
if int(numpy.__version__[0]) < 2:
    from numpy.lib import (
        delete,
        extract,
        insert,
        msort,
        place,
        setxor1d,
        unwrap,
        vectorize,
    )


# If we are going to trace through these, we should use NumPy
# If testing on eager mode, we use torch._numpy
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy import (
        angle,
        bartlett,
        blackman,
        corrcoef,
        cov,
        diff,
        digitize,
        flipud,
        gradient,
        hamming,
        hanning,
        i0,
        interp,
        kaiser,
        meshgrid,
        sinc,
        trapz,
        trim_zeros,
        unique,
    )
    from numpy.core.numeric import normalize_axis_tuple
    from numpy.random import rand
    from numpy.testing import (
        assert_,
        assert_allclose,
        assert_almost_equal,
        assert_array_almost_equal,
        assert_array_equal,
        assert_equal,
        assert_raises_regex,
        assert_warns,
        suppress_warnings,
    )
else:
    import torch._numpy as np
    from torch._numpy import (
        angle,
        bartlett,
        blackman,
        corrcoef,
        cov,
        diff,
        flipud,
        gradient,
        hamming,
        hanning,
        i0,
        kaiser,
        meshgrid,
        sinc,
        unique,
    )
    from torch._numpy._util import normalize_axis_tuple
    from torch._numpy.random import rand
    from torch._numpy.testing import (
        assert_,
        assert_allclose,
        assert_almost_equal,
        assert_array_almost_equal,
        assert_array_equal,
        assert_equal,
        assert_raises_regex,
        assert_warns,
        suppress_warnings,
    )


def get_mat(n):
    data = np.arange(n)
    #    data = np.add.outer(data, data)
    data = data[:, None] + data[None, :]
    return data


def _make_complex(real, imag):
    """
    Like real + 1j * imag, but behaves as expected when imag contains non-finite
    values
    """
    ret = np.zeros(np.broadcast(real, imag).shape, np.complex128)
    ret.real = real
    ret.imag = imag
    return ret


class TestRot90(TestCase):
    def test_basic(self):
        assert_raises(ValueError, np.rot90, np.ones(4))
        assert_raises(
            (ValueError, RuntimeError), np.rot90, np.ones((2, 2, 2)), axes=(0, 1, 2)
        )
        assert_raises(ValueError, np.rot90, np.ones((2, 2)), axes=(0, 2))
        assert_raises(ValueError, np.rot90, np.ones((2, 2)), axes=(1, 1))
        assert_raises(ValueError, np.rot90, np.ones((2, 2, 2)), axes=(-2, 1))

        a = [[0, 1, 2], [3, 4, 5]]
        b1 = [[2, 5], [1, 4], [0, 3]]
        b2 = [[5, 4, 3], [2, 1, 0]]
        b3 = [[3, 0], [4, 1], [5, 2]]
        b4 = [[0, 1, 2], [3, 4, 5]]

        for k in range(-3, 13, 4):
            assert_equal(np.rot90(a, k=k), b1)
        for k in range(-2, 13, 4):
            assert_equal(np.rot90(a, k=k), b2)
        for k in range(-1, 13, 4):
            assert_equal(np.rot90(a, k=k), b3)
        for k in range(0, 13, 4):
            assert_equal(np.rot90(a, k=k), b4)

        assert_equal(np.rot90(np.rot90(a, axes=(0, 1)), axes=(1, 0)), a)
        assert_equal(np.rot90(a, k=1, axes=(1, 0)), np.rot90(a, k=-1, axes=(0, 1)))

    def test_axes(self):
        a = np.ones((50, 40, 3))
        assert_equal(np.rot90(a).shape, (40, 50, 3))
        assert_equal(np.rot90(a, axes=(0, 2)), np.rot90(a, axes=(0, -1)))
        assert_equal(np.rot90(a, axes=(1, 2)), np.rot90(a, axes=(-2, -1)))

    def test_rotation_axes(self):
        a = np.arange(8).reshape((2, 2, 2))

        a_rot90_01 = [[[2, 3], [6, 7]], [[0, 1], [4, 5]]]
        a_rot90_12 = [[[1, 3], [0, 2]], [[5, 7], [4, 6]]]
        a_rot90_20 = [[[4, 0], [6, 2]], [[5, 1], [7, 3]]]
        a_rot90_10 = [[[4, 5], [0, 1]], [[6, 7], [2, 3]]]

        assert_equal(np.rot90(a, axes=(0, 1)), a_rot90_01)
        assert_equal(np.rot90(a, axes=(1, 0)), a_rot90_10)
        assert_equal(np.rot90(a, axes=(1, 2)), a_rot90_12)

        for k in range(1, 5):
            assert_equal(
                np.rot90(a, k=k, axes=(2, 0)),
                np.rot90(a_rot90_20, k=k - 1, axes=(2, 0)),
            )


class TestFlip(TestCase):
    def test_axes(self):
        assert_raises(np.AxisError, np.flip, np.ones(4), axis=1)
        assert_raises(np.AxisError, np.flip, np.ones((4, 4)), axis=2)
        assert_raises(np.AxisError, np.flip, np.ones((4, 4)), axis=-3)
        assert_raises(np.AxisError, np.flip, np.ones((4, 4)), axis=(0, 3))

    @skip(reason="no [::-1] indexing")
    def test_basic_lr(self):
        a = get_mat(4)
        b = a[:, ::-1]
        assert_equal(np.flip(a, 1), b)
        a = [[0, 1, 2], [3, 4, 5]]
        b = [[2, 1, 0], [5, 4, 3]]
        assert_equal(np.flip(a, 1), b)

    @skip(reason="no [::-1] indexing")
    def test_basic_ud(self):
        a = get_mat(4)
        b = a[::-1, :]
        assert_equal(np.flip(a, 0), b)
        a = [[0, 1, 2], [3, 4, 5]]
        b = [[3, 4, 5], [0, 1, 2]]
        assert_equal(np.flip(a, 0), b)

    def test_3d_swap_axis0(self):
        a = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])

        b = np.array([[[4, 5], [6, 7]], [[0, 1], [2, 3]]])

        assert_equal(np.flip(a, 0), b)

    def test_3d_swap_axis1(self):
        a = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])

        b = np.array([[[2, 3], [0, 1]], [[6, 7], [4, 5]]])

        assert_equal(np.flip(a, 1), b)

    def test_3d_swap_axis2(self):
        a = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])

        b = np.array([[[1, 0], [3, 2]], [[5, 4], [7, 6]]])

        assert_equal(np.flip(a, 2), b)

    def test_4d(self):
        a = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
        for i in range(a.ndim):
            assert_equal(np.flip(a, i), np.flipud(a.swapaxes(0, i)).swapaxes(i, 0))

    def test_default_axis(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([[6, 5, 4], [3, 2, 1]])
        assert_equal(np.flip(a), b)

    def test_multiple_axes(self):
        a = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])

        assert_equal(np.flip(a, axis=()), a)

        b = np.array([[[5, 4], [7, 6]], [[1, 0], [3, 2]]])

        assert_equal(np.flip(a, axis=(0, 2)), b)

        c = np.array([[[3, 2], [1, 0]], [[7, 6], [5, 4]]])

        assert_equal(np.flip(a, axis=(1, 2)), c)


class TestAny(TestCase):
    def test_basic(self):
        y1 = [0, 0, 1, 0]
        y2 = [0, 0, 0, 0]
        y3 = [1, 0, 1, 0]
        assert_(np.any(y1))
        assert_(np.any(y3))
        assert_(not np.any(y2))

    def test_nd(self):
        y1 = [[0, 0, 0], [0, 1, 0], [1, 1, 0]]
        assert_(np.any(y1))
        assert_array_equal(np.any(y1, axis=0), [1, 1, 0])
        assert_array_equal(np.any(y1, axis=1), [0, 1, 1])


class TestAll(TestCase):
    def test_basic(self):
        y1 = [0, 1, 1, 0]
        y2 = [0, 0, 0, 0]
        y3 = [1, 1, 1, 1]
        assert_(not np.all(y1))
        assert_(np.all(y3))
        assert_(not np.all(y2))
        assert_(np.all(~np.array(y2)))

    def test_nd(self):
        y1 = [[0, 0, 1], [0, 1, 1], [1, 1, 1]]
        assert_(not np.all(y1))
        assert_array_equal(np.all(y1, axis=0), [0, 0, 1])
        assert_array_equal(np.all(y1, axis=1), [0, 0, 1])


class TestCopy(TestCase):
    def test_basic(self):
        a = np.array([[1, 2], [3, 4]])
        a_copy = np.copy(a)
        assert_array_equal(a, a_copy)
        a_copy[0, 0] = 10
        assert_equal(a[0, 0], 1)
        assert_equal(a_copy[0, 0], 10)

    @xpassIfTorchDynamo_np  # (reason="order='F' not implemented")
    def test_order(self):
        # It turns out that people rely on np.copy() preserving order by
        # default; changing this broke scikit-learn:
        # github.com/scikit-learn/scikit-learn/commit/7842748cf777412c506
        a = np.array([[1, 2], [3, 4]])
        assert_(a.flags.c_contiguous)
        assert_(not a.flags.f_contiguous)
        a_fort = np.array([[1, 2], [3, 4]], order="F")
        assert_(not a_fort.flags.c_contiguous)
        assert_(a_fort.flags.f_contiguous)
        a_copy = np.copy(a)
        assert_(a_copy.flags.c_contiguous)
        assert_(not a_copy.flags.f_contiguous)
        a_fort_copy = np.copy(a_fort)
        assert_(not a_fort_copy.flags.c_contiguous)
        assert_(a_fort_copy.flags.f_contiguous)


@instantiate_parametrized_tests
class TestAverage(TestCase):
    def test_basic(self):
        y1 = np.array([1, 2, 3])
        assert_(np.average(y1, axis=0) == 2.0)
        y2 = np.array([1.0, 2.0, 3.0])
        assert_(np.average(y2, axis=0) == 2.0)
        y3 = [0.0, 0.0, 0.0]
        assert_(np.average(y3, axis=0) == 0.0)

        y4 = np.ones((4, 4))
        y4[0, 1] = 0
        y4[1, 0] = 2
        assert_almost_equal(y4.mean(0), np.average(y4, 0))
        assert_almost_equal(y4.mean(1), np.average(y4, 1))

        y5 = rand(5, 5)
        assert_almost_equal(y5.mean(0), np.average(y5, 0))
        assert_almost_equal(y5.mean(1), np.average(y5, 1))

    @skip(reason="NP_VER: fails on CI")
    @parametrize(
        "x, axis, expected_avg, weights, expected_wavg, expected_wsum",
        [
            ([1, 2, 3], None, [2.0], [3, 4, 1], [1.75], [8.0]),
            (
                [[1, 2, 5], [1, 6, 11]],
                0,
                [[1.0, 4.0, 8.0]],
                [1, 3],
                [[1.0, 5.0, 9.5]],
                [[4, 4, 4]],
            ),
        ],
    )
    def test_basic_keepdims(
        self, x, axis, expected_avg, weights, expected_wavg, expected_wsum
    ):
        avg = np.average(x, axis=axis, keepdims=True)
        assert avg.shape == np.shape(expected_avg)
        assert_array_equal(avg, expected_avg)

        wavg = np.average(x, axis=axis, weights=weights, keepdims=True)
        assert wavg.shape == np.shape(expected_wavg)
        assert_array_equal(wavg, expected_wavg)

        wavg, wsum = np.average(
            x, axis=axis, weights=weights, returned=True, keepdims=True
        )
        assert wavg.shape == np.shape(expected_wavg)
        assert_array_equal(wavg, expected_wavg)
        assert wsum.shape == np.shape(expected_wsum)
        assert_array_equal(wsum, expected_wsum)

    @skip(reason="NP_VER: fails on CI")
    def test_weights(self):
        y = np.arange(10)
        w = np.arange(10)
        actual = np.average(y, weights=w)
        desired = (np.arange(10) ** 2).sum() * 1.0 / np.arange(10).sum()
        assert_almost_equal(actual, desired)

        y1 = np.array([[1, 2, 3], [4, 5, 6]])
        w0 = [1, 2]
        actual = np.average(y1, weights=w0, axis=0)
        desired = np.array([3.0, 4.0, 5.0])
        assert_almost_equal(actual, desired)

        w1 = [0, 0, 1]
        actual = np.average(y1, weights=w1, axis=1)
        desired = np.array([3.0, 6.0])
        assert_almost_equal(actual, desired)

        # This should raise an error. Can we test for that ?
        # assert_equal(average(y1, weights=w1), 9./2.)

        # 2D Case
        w2 = [[0, 0, 1], [0, 0, 2]]
        desired = np.array([3.0, 6.0])
        assert_array_equal(np.average(y1, weights=w2, axis=1), desired)
        assert_equal(np.average(y1, weights=w2), 5.0)

        y3 = rand(5).astype(np.float32)
        w3 = rand(5).astype(np.float64)

        assert_(np.average(y3, weights=w3).dtype == np.result_type(y3, w3))

        # test weights with `keepdims=False` and `keepdims=True`
        x = np.array([2, 3, 4]).reshape(3, 1)
        w = np.array([4, 5, 6]).reshape(3, 1)

        actual = np.average(x, weights=w, axis=1, keepdims=False)
        desired = np.array([2.0, 3.0, 4.0])
        assert_array_equal(actual, desired)

        actual = np.average(x, weights=w, axis=1, keepdims=True)
        desired = np.array([[2.0], [3.0], [4.0]])
        assert_array_equal(actual, desired)

    def test_returned(self):
        y = np.array([[1, 2, 3], [4, 5, 6]])

        # No weights
        avg, scl = np.average(y, returned=True)
        assert_equal(scl, 6.0)

        avg, scl = np.average(y, 0, returned=True)
        assert_array_equal(scl, np.array([2.0, 2.0, 2.0]))

        avg, scl = np.average(y, 1, returned=True)
        assert_array_equal(scl, np.array([3.0, 3.0]))

        # With weights
        w0 = [1, 2]
        avg, scl = np.average(y, weights=w0, axis=0, returned=True)
        assert_array_equal(scl, np.array([3.0, 3.0, 3.0]))

        w1 = [1, 2, 3]
        avg, scl = np.average(y, weights=w1, axis=1, returned=True)
        assert_array_equal(scl, np.array([6.0, 6.0]))

        w2 = [[0, 0, 1], [1, 2, 3]]
        avg, scl = np.average(y, weights=w2, axis=1, returned=True)
        assert_array_equal(scl, np.array([1.0, 6.0]))

    def test_upcasting(self):
        typs = [
            ("i4", "i4", "f8"),
            ("i4", "f4", "f8"),
            ("f4", "i4", "f8"),
            ("f4", "f4", "f4"),
            ("f4", "f8", "f8"),
        ]
        for at, wt, rt in typs:
            a = np.array([[1, 2], [3, 4]], dtype=at)
            w = np.array([[1, 2], [3, 4]], dtype=wt)
            assert_equal(np.average(a, weights=w).dtype, np.dtype(rt))

    @skip(reason="support Fraction objects?")
    def test_average_class_without_dtype(self):
        # see gh-21988
        a = np.array([Fraction(1, 5), Fraction(3, 5)])
        assert_equal(np.average(a), Fraction(2, 5))


@xfail  # (reason="TODO: implement")
class TestSelect(TestCase):
    choices = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
    conditions = [
        np.array([False, False, False]),
        np.array([False, True, False]),
        np.array([False, False, True]),
    ]

    def _select(self, cond, values, default=0):
        output = []
        for m in range(len(cond)):
            output += [V[m] for V, C in zip(values, cond) if C[m]] or [default]
        return output

    def test_basic(self):
        choices = self.choices
        conditions = self.conditions
        assert_array_equal(
            select(conditions, choices, default=15),
            self._select(conditions, choices, default=15),
        )

        assert_equal(len(choices), 3)
        assert_equal(len(conditions), 3)

    def test_broadcasting(self):
        conditions = [np.array(True), np.array([False, True, False])]
        choices = [1, np.arange(12).reshape(4, 3)]
        assert_array_equal(select(conditions, choices), np.ones((4, 3)))
        # default can broadcast too:
        assert_equal(select([True], [0], default=[0]).shape, (1,))

    def test_return_dtype(self):
        assert_equal(select(self.conditions, self.choices, 1j).dtype, np.complex128)
        # But the conditions need to be stronger then the scalar default
        # if it is scalar.
        choices = [choice.astype(np.int8) for choice in self.choices]
        assert_equal(select(self.conditions, choices).dtype, np.int8)

        d = np.array([1, 2, 3, np.nan, 5, 7])
        m = np.isnan(d)
        assert_equal(select([m], [d]), [0, 0, 0, np.nan, 0, 0])

    def test_deprecated_empty(self):
        assert_raises(ValueError, select, [], [], 3j)
        assert_raises(ValueError, select, [], [])

    def test_non_bool_deprecation(self):
        choices = self.choices
        conditions = self.conditions[:]
        conditions[0] = conditions[0].astype(np.int_)
        assert_raises(TypeError, select, conditions, choices)
        conditions[0] = conditions[0].astype(np.uint8)
        assert_raises(TypeError, select, conditions, choices)
        assert_raises(TypeError, select, conditions, choices)

    def test_many_arguments(self):
        # This used to be limited by NPY_MAXARGS == 32
        conditions = [np.array([False])] * 100
        choices = [np.array([1])] * 100
        select(conditions, choices)


@xpassIfTorchDynamo_np  # (reason="TODO: implement")
@instantiate_parametrized_tests
class TestInsert(TestCase):
    def test_basic(self):
        a = [1, 2, 3]
        assert_equal(insert(a, 0, 1), [1, 1, 2, 3])
        assert_equal(insert(a, 3, 1), [1, 2, 3, 1])
        assert_equal(insert(a, [1, 1, 1], [1, 2, 3]), [1, 1, 2, 3, 2, 3])
        assert_equal(insert(a, 1, [1, 2, 3]), [1, 1, 2, 3, 2, 3])
        assert_equal(insert(a, [1, -1, 3], 9), [1, 9, 2, 9, 3, 9])
        assert_equal(insert(a, slice(-1, None, -1), 9), [9, 1, 9, 2, 9, 3])
        assert_equal(insert(a, [-1, 1, 3], [7, 8, 9]), [1, 8, 2, 7, 3, 9])
        b = np.array([0, 1], dtype=np.float64)
        assert_equal(insert(b, 0, b[0]), [0.0, 0.0, 1.0])
        assert_equal(insert(b, [], []), b)
        # Bools will be treated differently in the future:
        # assert_equal(insert(a, np.array([True]*4), 9), [9, 1, 9, 2, 9, 3, 9])
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always", "", FutureWarning)
            assert_equal(insert(a, np.array([True] * 4), 9), [1, 9, 9, 9, 9, 2, 3])
            assert_(w[0].category is FutureWarning)

    def test_multidim(self):
        a = [[1, 1, 1]]
        r = [[2, 2, 2], [1, 1, 1]]
        assert_equal(insert(a, 0, [1]), [1, 1, 1, 1])
        assert_equal(insert(a, 0, [2, 2, 2], axis=0), r)
        assert_equal(insert(a, 0, 2, axis=0), r)
        assert_equal(insert(a, 2, 2, axis=1), [[1, 1, 2, 1]])

        a = np.array([[1, 1], [2, 2], [3, 3]])
        b = np.arange(1, 4).repeat(3).reshape(3, 3)
        c = np.concatenate(
            (a[:, 0:1], np.arange(1, 4).repeat(3).reshape(3, 3).T, a[:, 1:2]), axis=1
        )
        assert_equal(insert(a, [1], [[1], [2], [3]], axis=1), b)
        assert_equal(insert(a, [1], [1, 2, 3], axis=1), c)
        # scalars behave differently, in this case exactly opposite:
        assert_equal(insert(a, 1, [1, 2, 3], axis=1), b)
        assert_equal(insert(a, 1, [[1], [2], [3]], axis=1), c)

        a = np.arange(4).reshape(2, 2)
        assert_equal(insert(a[:, :1], 1, a[:, 1], axis=1), a)
        assert_equal(insert(a[:1, :], 1, a[1, :], axis=0), a)

        # negative axis value
        a = np.arange(24).reshape((2, 3, 4))
        assert_equal(
            insert(a, 1, a[:, :, 3], axis=-1), insert(a, 1, a[:, :, 3], axis=2)
        )
        assert_equal(
            insert(a, 1, a[:, 2, :], axis=-2), insert(a, 1, a[:, 2, :], axis=1)
        )

        # invalid axis value
        assert_raises(np.AxisError, insert, a, 1, a[:, 2, :], axis=3)
        assert_raises(np.AxisError, insert, a, 1, a[:, 2, :], axis=-4)

        # negative axis value
        a = np.arange(24).reshape((2, 3, 4))
        assert_equal(
            insert(a, 1, a[:, :, 3], axis=-1), insert(a, 1, a[:, :, 3], axis=2)
        )
        assert_equal(
            insert(a, 1, a[:, 2, :], axis=-2), insert(a, 1, a[:, 2, :], axis=1)
        )

    def test_0d(self):
        a = np.array(1)
        with pytest.raises(np.AxisError):
            insert(a, [], 2, axis=0)
        with pytest.raises(TypeError):
            insert(a, [], 2, axis="nonsense")

    def test_index_array_copied(self):
        x = np.array([1, 1, 1])
        np.insert([0, 1, 2], x, [3, 4, 5])
        assert_equal(x, np.array([1, 1, 1]))

    def test_index_floats(self):
        with pytest.raises(IndexError):
            np.insert([0, 1, 2], np.array([1.0, 2.0]), [10, 20])
        with pytest.raises(IndexError):
            np.insert([0, 1, 2], np.array([], dtype=float), [])

    @skip(reason="NP_VER: fails on CI")
    @parametrize("idx", [4, -4])
    def test_index_out_of_bounds(self, idx):
        with pytest.raises(IndexError, match="out of bounds"):
            np.insert([0, 1, 2], [idx], [3, 4])


class TestAmax(TestCase):
    def test_basic(self):
        a = [3, 4, 5, 10, -3, -5, 6.0]
        assert_equal(np.amax(a), 10.0)
        b = [[3, 6.0, 9.0], [4, 10.0, 5.0], [8, 3.0, 2.0]]
        assert_equal(np.amax(b, axis=0), [8.0, 10.0, 9.0])
        assert_equal(np.amax(b, axis=1), [9.0, 10.0, 8.0])


class TestAmin(TestCase):
    def test_basic(self):
        a = [3, 4, 5, 10, -3, -5, 6.0]
        assert_equal(np.amin(a), -5.0)
        b = [[3, 6.0, 9.0], [4, 10.0, 5.0], [8, 3.0, 2.0]]
        assert_equal(np.amin(b, axis=0), [3.0, 3.0, 2.0])
        assert_equal(np.amin(b, axis=1), [3.0, 4.0, 2.0])


class TestPtp(TestCase):
    def test_basic(self):
        a = np.array([3, 4, 5, 10, -3, -5, 6.0])
        assert_equal(a.ptp(axis=0), 15.0)
        b = np.array([[3, 6.0, 9.0], [4, 10.0, 5.0], [8, 3.0, 2.0]])
        assert_equal(b.ptp(axis=0), [5.0, 7.0, 7.0])
        assert_equal(b.ptp(axis=-1), [6.0, 6.0, 6.0])

        assert_equal(b.ptp(axis=0, keepdims=True), [[5.0, 7.0, 7.0]])
        assert_equal(b.ptp(axis=(0, 1), keepdims=True), [[8.0]])


class TestCumsum(TestCase):
    def test_basic(self):
        ba = [1, 2, 10, 11, 6, 5, 4]
        ba2 = [[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]]
        for ctype in [
            np.int8,
            np.uint8,
            np.int16,
            np.int32,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        ]:
            a = np.array(ba, ctype)
            a2 = np.array(ba2, ctype)

            tgt = np.array([1, 3, 13, 24, 30, 35, 39], ctype)
            assert_array_equal(np.cumsum(a, axis=0), tgt)

            tgt = np.array([[1, 2, 3, 4], [6, 8, 10, 13], [16, 11, 14, 18]], ctype)
            assert_array_equal(np.cumsum(a2, axis=0), tgt)

            tgt = np.array([[1, 3, 6, 10], [5, 11, 18, 27], [10, 13, 17, 22]], ctype)
            assert_array_equal(np.cumsum(a2, axis=1), tgt)


class TestProd(TestCase):
    def test_basic(self):
        ba = [1, 2, 10, 11, 6, 5, 4]
        ba2 = [[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]]
        for ctype in [
            np.int16,
            np.int32,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        ]:
            a = np.array(ba, ctype)
            a2 = np.array(ba2, ctype)
            if ctype in ["1", "b"]:
                assert_raises(ArithmeticError, np.prod, a)
                assert_raises(ArithmeticError, np.prod, a2, 1)
            else:
                assert_equal(a.prod(axis=0), 26400)
                assert_array_equal(a2.prod(axis=0), np.array([50, 36, 84, 180], ctype))
                assert_array_equal(a2.prod(axis=-1), np.array([24, 1890, 600], ctype))


class TestCumprod(TestCase):
    def test_basic(self):
        ba = [1, 2, 10, 11, 6, 5, 4]
        ba2 = [[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]]
        for ctype in [
            np.int16,
            np.int32,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        ]:
            a = np.array(ba, ctype)
            a2 = np.array(ba2, ctype)
            if ctype in ["1", "b"]:
                assert_raises(ArithmeticError, np.cumprod, a)
                assert_raises(ArithmeticError, np.cumprod, a2, 1)
                assert_raises(ArithmeticError, np.cumprod, a)
            else:
                assert_array_equal(
                    np.cumprod(a, axis=-1),
                    np.array([1, 2, 20, 220, 1320, 6600, 26400], ctype),
                )
                assert_array_equal(
                    np.cumprod(a2, axis=0),
                    np.array([[1, 2, 3, 4], [5, 12, 21, 36], [50, 36, 84, 180]], ctype),
                )
                assert_array_equal(
                    np.cumprod(a2, axis=-1),
                    np.array(
                        [[1, 2, 6, 24], [5, 30, 210, 1890], [10, 30, 120, 600]], ctype
                    ),
                )


class TestDiff(TestCase):
    def test_basic(self):
        x = [1, 4, 6, 7, 12]
        out = np.array([3, 2, 1, 5])
        out2 = np.array([-1, -1, 4])
        out3 = np.array([0, 5])
        assert_array_equal(diff(x), out)
        assert_array_equal(diff(x, n=2), out2)
        assert_array_equal(diff(x, n=3), out3)

        x = [1.1, 2.2, 3.0, -0.2, -0.1]
        out = np.array([1.1, 0.8, -3.2, 0.1])
        assert_almost_equal(diff(x), out)

        x = [True, True, False, False]
        out = np.array([False, True, False])
        out2 = np.array([True, True])
        assert_array_equal(diff(x), out)
        assert_array_equal(diff(x, n=2), out2)

    def test_axis(self):
        x = np.zeros((10, 20, 30))
        x[:, 1::2, :] = 1
        exp = np.ones((10, 19, 30))
        exp[:, 1::2, :] = -1
        assert_array_equal(diff(x), np.zeros((10, 20, 29)))
        assert_array_equal(diff(x, axis=-1), np.zeros((10, 20, 29)))
        assert_array_equal(diff(x, axis=0), np.zeros((9, 20, 30)))
        assert_array_equal(diff(x, axis=1), exp)
        assert_array_equal(diff(x, axis=-2), exp)
        assert_raises(np.AxisError, diff, x, axis=3)
        assert_raises(np.AxisError, diff, x, axis=-4)

        x = np.array(1.11111111111, np.float64)
        assert_raises(ValueError, diff, x)

    def test_nd(self):
        x = 20 * rand(10, 20, 30)
        out1 = x[:, :, 1:] - x[:, :, :-1]
        out2 = out1[:, :, 1:] - out1[:, :, :-1]
        out3 = x[1:, :, :] - x[:-1, :, :]
        out4 = out3[1:, :, :] - out3[:-1, :, :]
        assert_array_equal(diff(x), out1)
        assert_array_equal(diff(x, n=2), out2)
        assert_array_equal(diff(x, axis=0), out3)
        assert_array_equal(diff(x, n=2, axis=0), out4)

    def test_n(self):
        x = list(range(3))
        assert_raises(ValueError, diff, x, n=-1)
        output = [diff(x, n=n) for n in range(1, 5)]
        expected_output = [[1, 1], [0], [], []]
        # assert_(diff(x, n=0) is x)
        for n, (expected, out) in enumerate(zip(expected_output, output), start=1):
            assert_(type(out) is np.ndarray)
            assert_array_equal(out, expected)
            assert_equal(out.dtype, np.int_)
            assert_equal(len(out), max(0, len(x) - n))

    def test_prepend(self):
        x = np.arange(5) + 1
        assert_array_equal(diff(x, prepend=0), np.ones(5))
        assert_array_equal(diff(x, prepend=[0]), np.ones(5))
        assert_array_equal(np.cumsum(np.diff(x, prepend=0)), x)
        assert_array_equal(diff(x, prepend=[-1, 0]), np.ones(6))

        x = np.arange(4).reshape(2, 2)
        result = np.diff(x, axis=1, prepend=0)
        expected = [[0, 1], [2, 1]]
        assert_array_equal(result, expected)
        result = np.diff(x, axis=1, prepend=[[0], [0]])
        assert_array_equal(result, expected)

        result = np.diff(x, axis=0, prepend=0)
        expected = [[0, 1], [2, 2]]
        assert_array_equal(result, expected)
        result = np.diff(x, axis=0, prepend=[[0, 0]])
        assert_array_equal(result, expected)

        assert_raises((ValueError, RuntimeError), np.diff, x, prepend=np.zeros((3, 3)))

        assert_raises(np.AxisError, diff, x, prepend=0, axis=3)

    def test_append(self):
        x = np.arange(5)
        result = diff(x, append=0)
        expected = [1, 1, 1, 1, -4]
        assert_array_equal(result, expected)
        result = diff(x, append=[0])
        assert_array_equal(result, expected)
        result = diff(x, append=[0, 2])
        expected = expected + [2]
        assert_array_equal(result, expected)

        x = np.arange(4).reshape(2, 2)
        result = np.diff(x, axis=1, append=0)
        expected = [[1, -1], [1, -3]]
        assert_array_equal(result, expected)
        result = np.diff(x, axis=1, append=[[0], [0]])
        assert_array_equal(result, expected)

        result = np.diff(x, axis=0, append=0)
        expected = [[2, 2], [-2, -3]]
        assert_array_equal(result, expected)
        result = np.diff(x, axis=0, append=[[0, 0]])
        assert_array_equal(result, expected)

        assert_raises((ValueError, RuntimeError), np.diff, x, append=np.zeros((3, 3)))

        assert_raises(np.AxisError, diff, x, append=0, axis=3)


@xpassIfTorchDynamo_np  # (reason="TODO: implement")
@instantiate_parametrized_tests
class TestDelete(TestCase):
    def setUp(self):
        self.a = np.arange(5)
        self.nd_a = np.arange(5).repeat(2).reshape(1, 5, 2)

    def _check_inverse_of_slicing(self, indices):
        a_del = delete(self.a, indices)
        nd_a_del = delete(self.nd_a, indices, axis=1)
        msg = f"Delete failed for obj: {indices!r}"
        assert_array_equal(setxor1d(a_del, self.a[indices,]), self.a, err_msg=msg)
        xor = setxor1d(nd_a_del[0, :, 0], self.nd_a[0, indices, 0])
        assert_array_equal(xor, self.nd_a[0, :, 0], err_msg=msg)

    def test_slices(self):
        lims = [-6, -2, 0, 1, 2, 4, 5]
        steps = [-3, -1, 1, 3]
        for start in lims:
            for stop in lims:
                for step in steps:
                    s = slice(start, stop, step)
                    self._check_inverse_of_slicing(s)

    def test_fancy(self):
        self._check_inverse_of_slicing(np.array([[0, 1], [2, 1]]))
        with pytest.raises(IndexError):
            delete(self.a, [100])
        with pytest.raises(IndexError):
            delete(self.a, [-100])

        self._check_inverse_of_slicing([0, -1, 2, 2])

        self._check_inverse_of_slicing([True, False, False, True, False])

        # not legal, indexing with these would change the dimension
        with pytest.raises(ValueError):
            delete(self.a, True)
        with pytest.raises(ValueError):
            delete(self.a, False)

        # not enough items
        with pytest.raises(ValueError):
            delete(self.a, [False] * 4)

    def test_single(self):
        self._check_inverse_of_slicing(0)
        self._check_inverse_of_slicing(-4)

    def test_0d(self):
        a = np.array(1)
        with pytest.raises(np.AxisError):
            delete(a, [], axis=0)
        with pytest.raises(TypeError):
            delete(a, [], axis="nonsense")

    def test_array_order_preserve(self):
        # See gh-7113
        k = np.arange(10).reshape(2, 5, order="F")
        m = delete(k, slice(60, None), axis=1)

        # 'k' is Fortran ordered, and 'm' should have the
        # same ordering as 'k' and NOT become C ordered
        assert_equal(m.flags.c_contiguous, k.flags.c_contiguous)
        assert_equal(m.flags.f_contiguous, k.flags.f_contiguous)

    def test_index_floats(self):
        with pytest.raises(IndexError):
            np.delete([0, 1, 2], np.array([1.0, 2.0]))
        with pytest.raises(IndexError):
            np.delete([0, 1, 2], np.array([], dtype=float))

    @parametrize(
        "indexer", [subtest(np.array([1]), name="array([1])"), subtest([1], name="[1]")]
    )
    def test_single_item_array(self, indexer):
        a_del_int = delete(self.a, 1)
        a_del = delete(self.a, indexer)
        assert_equal(a_del_int, a_del)

        nd_a_del_int = delete(self.nd_a, 1, axis=1)
        nd_a_del = delete(self.nd_a, np.array([1]), axis=1)
        assert_equal(nd_a_del_int, nd_a_del)

    def test_single_item_array_non_int(self):
        # Special handling for integer arrays must not affect non-integer ones.
        # If `False` was cast to `0` it would delete the element:
        res = delete(np.ones(1), np.array([False]))
        assert_array_equal(res, np.ones(1))

        # Test the more complicated (with axis) case from gh-21840
        x = np.ones((3, 1))
        false_mask = np.array([False], dtype=bool)
        true_mask = np.array([True], dtype=bool)

        res = delete(x, false_mask, axis=-1)
        assert_array_equal(res, x)
        res = delete(x, true_mask, axis=-1)
        assert_array_equal(res, x[:, :0])


@instantiate_parametrized_tests
class TestGradient(TestCase):
    def test_basic(self):
        v = [[1, 1], [3, 4]]
        x = np.array(v)
        dx = [np.array([[2.0, 3.0], [2.0, 3.0]]), np.array([[0.0, 0.0], [1.0, 1.0]])]
        assert_array_equal(gradient(x), dx)
        assert_array_equal(gradient(v), dx)

    def test_args(self):
        dx = np.cumsum(np.ones(5))
        dx_uneven = [1.0, 2.0, 5.0, 9.0, 11.0]
        f_2d = np.arange(25).reshape(5, 5)

        # distances must be scalars or have size equal to gradient[axis]
        gradient(np.arange(5), 3.0)
        gradient(np.arange(5), np.array(3.0))
        gradient(np.arange(5), dx)
        # dy is set equal to dx because scalar
        gradient(f_2d, 1.5)
        gradient(f_2d, np.array(1.5))

        gradient(f_2d, dx_uneven, dx_uneven)
        # mix between even and uneven spaces and
        # mix between scalar and vector
        gradient(f_2d, dx, 2)

        # 2D but axis specified
        gradient(f_2d, dx, axis=1)

        # 2d coordinate arguments are not yet allowed
        assert_raises_regex(
            ValueError,
            ".*scalars or 1d",
            gradient,
            f_2d,
            np.stack([dx] * 2, axis=-1),
            1,
        )

    def test_badargs(self):
        f_2d = np.arange(25).reshape(5, 5)
        x = np.cumsum(np.ones(5))

        # wrong sizes
        assert_raises(ValueError, gradient, f_2d, x, np.ones(2))
        assert_raises(ValueError, gradient, f_2d, 1, np.ones(2))
        assert_raises(ValueError, gradient, f_2d, np.ones(2), np.ones(2))
        # wrong number of arguments
        assert_raises(TypeError, gradient, f_2d, x)
        assert_raises(TypeError, gradient, f_2d, x, axis=(0, 1))
        assert_raises(TypeError, gradient, f_2d, x, x, x)
        assert_raises(TypeError, gradient, f_2d, 1, 1, 1)
        assert_raises(TypeError, gradient, f_2d, x, x, axis=1)
        assert_raises(TypeError, gradient, f_2d, 1, 1, axis=1)

    @torch._dynamo.config.patch(use_numpy_random_stream=True)
    def test_second_order_accurate(self):
        # Testing that the relative numerical error is less that 3% for
        # this example problem. This corresponds to second order
        # accurate finite differences for all interior and boundary
        # points.
        x = np.linspace(0, 1, 10)
        dx = x[1] - x[0]
        y = 2 * x**3 + 4 * x**2 + 2 * x
        analytical = 6 * x**2 + 8 * x + 2
        num_error = np.abs((np.gradient(y, dx, edge_order=2) / analytical) - 1)
        assert_(np.all(num_error < 0.03).item() is True)

        # test with unevenly spaced
        np.random.seed(0)
        x = np.sort(np.random.random(10))
        y = 2 * x**3 + 4 * x**2 + 2 * x
        analytical = 6 * x**2 + 8 * x + 2
        num_error = np.abs((np.gradient(y, x, edge_order=2) / analytical) - 1)
        assert_(np.all(num_error < 0.03).item() is True)

    def test_spacing(self):
        f = np.array([0, 2.0, 3.0, 4.0, 5.0, 5.0])
        f = np.tile(f, (6, 1)) + f.reshape(-1, 1)
        x_uneven = np.array([0.0, 0.5, 1.0, 3.0, 5.0, 7.0])
        x_even = np.arange(6.0)

        fdx_even_ord1 = np.tile([2.0, 1.5, 1.0, 1.0, 0.5, 0.0], (6, 1))
        fdx_even_ord2 = np.tile([2.5, 1.5, 1.0, 1.0, 0.5, -0.5], (6, 1))
        fdx_uneven_ord1 = np.tile([4.0, 3.0, 1.7, 0.5, 0.25, 0.0], (6, 1))
        fdx_uneven_ord2 = np.tile([5.0, 3.0, 1.7, 0.5, 0.25, -0.25], (6, 1))

        # evenly spaced
        for edge_order, exp_res in [(1, fdx_even_ord1), (2, fdx_even_ord2)]:
            res1 = gradient(f, 1.0, axis=(0, 1), edge_order=edge_order)
            res2 = gradient(f, x_even, x_even, axis=(0, 1), edge_order=edge_order)
            res3 = gradient(f, x_even, x_even, axis=None, edge_order=edge_order)
            assert_array_equal(res1, res2)
            assert_array_equal(res2, res3)
            assert_almost_equal(res1[0], exp_res.T)
            assert_almost_equal(res1[1], exp_res)

            res1 = gradient(f, 1.0, axis=0, edge_order=edge_order)
            res2 = gradient(f, x_even, axis=0, edge_order=edge_order)
            assert_(res1.shape == res2.shape)
            assert_almost_equal(res2, exp_res.T)

            res1 = gradient(f, 1.0, axis=1, edge_order=edge_order)
            res2 = gradient(f, x_even, axis=1, edge_order=edge_order)
            assert_(res1.shape == res2.shape)
            assert_array_equal(res2, exp_res)

        # unevenly spaced
        for edge_order, exp_res in [(1, fdx_uneven_ord1), (2, fdx_uneven_ord2)]:
            res1 = gradient(f, x_uneven, x_uneven, axis=(0, 1), edge_order=edge_order)
            res2 = gradient(f, x_uneven, x_uneven, axis=None, edge_order=edge_order)
            assert_array_equal(res1, res2)
            assert_almost_equal(res1[0], exp_res.T)
            assert_almost_equal(res1[1], exp_res)

            res1 = gradient(f, x_uneven, axis=0, edge_order=edge_order)
            assert_almost_equal(res1, exp_res.T)

            res1 = gradient(f, x_uneven, axis=1, edge_order=edge_order)
            assert_almost_equal(res1, exp_res)

        # mixed
        res1 = gradient(f, x_even, x_uneven, axis=(0, 1), edge_order=1)
        res2 = gradient(f, x_uneven, x_even, axis=(1, 0), edge_order=1)
        assert_array_equal(res1[0], res2[1])
        assert_array_equal(res1[1], res2[0])
        assert_almost_equal(res1[0], fdx_even_ord1.T)
        assert_almost_equal(res1[1], fdx_uneven_ord1)

        res1 = gradient(f, x_even, x_uneven, axis=(0, 1), edge_order=2)
        res2 = gradient(f, x_uneven, x_even, axis=(1, 0), edge_order=2)
        assert_array_equal(res1[0], res2[1])
        assert_array_equal(res1[1], res2[0])
        assert_almost_equal(res1[0], fdx_even_ord2.T)
        assert_almost_equal(res1[1], fdx_uneven_ord2)

    def test_specific_axes(self):
        # Testing that gradient can work on a given axis only
        v = [[1, 1], [3, 4]]
        x = np.array(v)
        dx = [np.array([[2.0, 3.0], [2.0, 3.0]]), np.array([[0.0, 0.0], [1.0, 1.0]])]
        assert_array_equal(gradient(x, axis=0), dx[0])
        assert_array_equal(gradient(x, axis=1), dx[1])
        assert_array_equal(gradient(x, axis=-1), dx[1])
        assert_array_equal(gradient(x, axis=(1, 0)), [dx[1], dx[0]])

        # test axis=None which means all axes
        assert_almost_equal(gradient(x, axis=None), [dx[0], dx[1]])
        # and is the same as no axis keyword given
        assert_almost_equal(gradient(x, axis=None), gradient(x))

        # test vararg order
        assert_array_equal(gradient(x, 2, 3, axis=(1, 0)), [dx[1] / 2.0, dx[0] / 3.0])
        # test maximal number of varargs
        assert_raises(TypeError, gradient, x, 1, 2, axis=1)

        assert_raises(np.AxisError, gradient, x, axis=3)
        assert_raises(np.AxisError, gradient, x, axis=-3)
        # assert_raises(TypeError, gradient, x, axis=[1,])

    def test_inexact_dtypes(self):
        for dt in [np.float16, np.float32, np.float64]:
            # dtypes should not be promoted in a different way to what diff does
            x = np.array([1, 2, 3], dtype=dt)
            assert_equal(gradient(x).dtype, np.diff(x).dtype)

    def test_values(self):
        # needs at least 2 points for edge_order ==1
        gradient(np.arange(2), edge_order=1)
        # needs at least 3 points for edge_order ==1
        gradient(np.arange(3), edge_order=2)

        assert_raises(ValueError, gradient, np.arange(0), edge_order=1)
        assert_raises(ValueError, gradient, np.arange(0), edge_order=2)
        assert_raises(ValueError, gradient, np.arange(1), edge_order=1)
        assert_raises(ValueError, gradient, np.arange(1), edge_order=2)
        assert_raises(ValueError, gradient, np.arange(2), edge_order=2)

    @parametrize(
        "f_dtype",
        [
            np.uint8,
        ],
    )
    def test_f_decreasing_unsigned_int(self, f_dtype):
        f = np.array([5, 4, 3, 2, 1], dtype=f_dtype)
        g = gradient(f)
        assert_array_equal(g, [-1] * len(f))

    @parametrize("f_dtype", [np.int8, np.int16, np.int32, np.int64])
    def test_f_signed_int_big_jump(self, f_dtype):
        maxint = np.iinfo(f_dtype).max
        x = np.array([1, 3])
        f = np.array([-1, maxint], dtype=f_dtype)
        dfdx = gradient(f, x)
        assert_array_equal(dfdx, [(maxint + 1) // 2] * 2)

    @parametrize(
        "x_dtype",
        [
            np.uint8,
        ],
    )
    def test_x_decreasing_unsigned(self, x_dtype):
        x = np.array([3, 2, 1], dtype=x_dtype)
        f = np.array([0, 2, 4])
        dfdx = gradient(f, x)
        assert_array_equal(dfdx, [-2] * len(x))

    @parametrize("x_dtype", [np.int8, np.int16, np.int32, np.int64])
    def test_x_signed_int_big_jump(self, x_dtype):
        minint = np.iinfo(x_dtype).min
        maxint = np.iinfo(x_dtype).max
        x = np.array([-1, maxint], dtype=x_dtype)
        f = np.array([minint // 2, 0])
        dfdx = gradient(f, x)
        assert_array_equal(dfdx, [0.5, 0.5])


class TestAngle(TestCase):
    def test_basic(self):
        x = [
            1 + 3j,
            np.sqrt(2) / 2.0 + 1j * np.sqrt(2) / 2,
            1,
            1j,
            -1,
            -1j,
            1 - 3j,
            -1 + 3j,
        ]
        y = angle(x)
        yo = [
            np.arctan(3.0 / 1.0),
            np.arctan(1.0),
            0,
            np.pi / 2,
            np.pi,
            -np.pi / 2.0,
            -np.arctan(3.0 / 1.0),
            np.pi - np.arctan(3.0 / 1.0),
        ]
        z = angle(x, deg=True)
        zo = np.array(yo) * 180 / np.pi
        assert_array_almost_equal(y, yo, 11)
        assert_array_almost_equal(z, zo, 11)


@xpassIfTorchDynamo_np
@instantiate_parametrized_tests
class TestTrimZeros(TestCase):
    a = np.array([0, 0, 1, 0, 2, 3, 4, 0])
    b = a.astype(float)
    c = a.astype(complex)
    #    d = a.astype(object)

    def values(self):
        attr_names = (
            "a",
            "b",
            "c",
        )  # "d")
        return (getattr(self, name) for name in attr_names)

    def test_basic(self):
        slc = np.s_[2:-1]
        for arr in self.values():
            res = trim_zeros(arr)
            assert_array_equal(res, arr[slc])

    def test_leading_skip(self):
        slc = np.s_[:-1]
        for arr in self.values():
            res = trim_zeros(arr, trim="b")
            assert_array_equal(res, arr[slc])

    def test_trailing_skip(self):
        slc = np.s_[2:]
        for arr in self.values():
            res = trim_zeros(arr, trim="F")
            assert_array_equal(res, arr[slc])

    def test_all_zero(self):
        for _arr in self.values():
            arr = np.zeros_like(_arr, dtype=_arr.dtype)

            res1 = trim_zeros(arr, trim="B")
            assert len(res1) == 0

            res2 = trim_zeros(arr, trim="f")
            assert len(res2) == 0

    def test_size_zero(self):
        arr = np.zeros(0)
        res = trim_zeros(arr)
        assert_array_equal(arr, res)

    @parametrize(
        "arr",
        [
            np.array([0, 2**62, 0]),
            #         np.array([0, 2**63, 0]),     # FIXME
            #         np.array([0, 2**64, 0])
        ],
    )
    def test_overflow(self, arr):
        slc = np.s_[1:2]
        res = trim_zeros(arr)
        assert_array_equal(res, arr[slc])

    def test_no_trim(self):
        arr = np.array([None, 1, None])
        res = trim_zeros(arr)
        assert_array_equal(arr, res)

    def test_list_to_list(self):
        res = trim_zeros(self.a.tolist())
        assert isinstance(res, list)


@xpassIfTorchDynamo_np  # (reason="TODO: implement")
class TestExtins(TestCase):
    def test_basic(self):
        a = np.array([1, 3, 2, 1, 2, 3, 3])
        b = extract(a > 1, a)
        assert_array_equal(b, [3, 2, 2, 3, 3])

    def test_place(self):
        # Make sure that non-np.ndarray objects
        # raise an error instead of doing nothing
        assert_raises(TypeError, place, [1, 2, 3], [True, False], [0, 1])

        a = np.array([1, 4, 3, 2, 5, 8, 7])
        place(a, [0, 1, 0, 1, 0, 1, 0], [2, 4, 6])
        assert_array_equal(a, [1, 2, 3, 4, 5, 6, 7])

        place(a, np.zeros(7), [])
        assert_array_equal(a, np.arange(1, 8))

        place(a, [1, 0, 1, 0, 1, 0, 1], [8, 9])
        assert_array_equal(a, [8, 2, 9, 4, 8, 6, 9])
        assert_raises_regex(
            ValueError,
            "Cannot insert from an empty array",
            lambda: place(a, [0, 0, 0, 0, 0, 1, 0], []),
        )

        # See Issue #6974
        a = np.array(["12", "34"])
        place(a, [0, 1], "9")
        assert_array_equal(a, ["12", "9"])

    def test_both(self):
        a = rand(10)
        mask = a > 0.5
        ac = a.copy()
        c = extract(mask, a)
        place(a, mask, 0)
        place(a, mask, c)
        assert_array_equal(a, ac)


# _foo1 and _foo2 are used in some tests in TestVectorize.


def _foo1(x, y=1.0):
    return y * math.floor(x)


def _foo2(x, y=1.0, z=0.0):
    return y * math.floor(x) + z


@skip  # (reason="vectorize not implemented")
class TestVectorize(TestCase):
    def test_simple(self):
        def addsubtract(a, b):
            if a > b:
                return a - b
            else:
                return a + b

        f = vectorize(addsubtract)
        r = f([0, 3, 6, 9], [1, 3, 5, 7])
        assert_array_equal(r, [1, 6, 1, 2])

    def test_scalar(self):
        def addsubtract(a, b):
            if a > b:
                return a - b
            else:
                return a + b

        f = vectorize(addsubtract)
        r = f([0, 3, 6, 9], 5)
        assert_array_equal(r, [5, 8, 1, 4])

    def test_large(self):
        x = np.linspace(-3, 2, 10000)
        f = vectorize(lambda x: x)
        y = f(x)
        assert_array_equal(y, x)

    def test_ufunc(self):
        f = vectorize(math.cos)
        args = np.array([0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi])
        r1 = f(args)
        r2 = np.cos(args)
        assert_array_almost_equal(r1, r2)

    def test_keywords(self):
        def foo(a, b=1):
            return a + b

        f = vectorize(foo)
        args = np.array([1, 2, 3])
        r1 = f(args)
        r2 = np.array([2, 3, 4])
        assert_array_equal(r1, r2)
        r1 = f(args, 2)
        r2 = np.array([3, 4, 5])
        assert_array_equal(r1, r2)

    def test_keywords_with_otypes_order1(self):
        # gh-1620: The second call of f would crash with
        # `ValueError: invalid number of arguments`.
        f = vectorize(_foo1, otypes=[float])
        # We're testing the caching of ufuncs by vectorize, so the order
        # of these function calls is an important part of the test.
        r1 = f(np.arange(3.0), 1.0)
        r2 = f(np.arange(3.0))
        assert_array_equal(r1, r2)

    def test_keywords_with_otypes_order2(self):
        # gh-1620: The second call of f would crash with
        # `ValueError: non-broadcastable output operand with shape ()
        # doesn't match the broadcast shape (3,)`.
        f = vectorize(_foo1, otypes=[float])
        # We're testing the caching of ufuncs by vectorize, so the order
        # of these function calls is an important part of the test.
        r1 = f(np.arange(3.0))
        r2 = f(np.arange(3.0), 1.0)
        assert_array_equal(r1, r2)

    def test_keywords_with_otypes_order3(self):
        # gh-1620: The third call of f would crash with
        # `ValueError: invalid number of arguments`.
        f = vectorize(_foo1, otypes=[float])
        # We're testing the caching of ufuncs by vectorize, so the order
        # of these function calls is an important part of the test.
        r1 = f(np.arange(3.0))
        r2 = f(np.arange(3.0), y=1.0)
        r3 = f(np.arange(3.0))
        assert_array_equal(r1, r2)
        assert_array_equal(r1, r3)

    def test_keywords_with_otypes_several_kwd_args1(self):
        # gh-1620 Make sure different uses of keyword arguments
        # don't break the vectorized function.
        f = vectorize(_foo2, otypes=[float])
        # We're testing the caching of ufuncs by vectorize, so the order
        # of these function calls is an important part of the test.
        r1 = f(10.4, z=100)
        r2 = f(10.4, y=-1)
        r3 = f(10.4)
        assert_equal(r1, _foo2(10.4, z=100))
        assert_equal(r2, _foo2(10.4, y=-1))
        assert_equal(r3, _foo2(10.4))

    def test_keywords_with_otypes_several_kwd_args2(self):
        # gh-1620 Make sure different uses of keyword arguments
        # don't break the vectorized function.
        f = vectorize(_foo2, otypes=[float])
        # We're testing the caching of ufuncs by vectorize, so the order
        # of these function calls is an important part of the test.
        r1 = f(z=100, x=10.4, y=-1)
        r2 = f(1, 2, 3)
        assert_equal(r1, _foo2(z=100, x=10.4, y=-1))
        assert_equal(r2, _foo2(1, 2, 3))

    def test_keywords_no_func_code(self):
        # This needs to test a function that has keywords but
        # no func_code attribute, since otherwise vectorize will
        # inspect the func_code.
        import random

        try:
            vectorize(random.randrange)  # Should succeed
        except Exception:
            raise AssertionError  # noqa: B904

    def test_keywords2_ticket_2100(self):
        # Test kwarg support: enhancement ticket 2100

        def foo(a, b=1):
            return a + b

        f = vectorize(foo)
        args = np.array([1, 2, 3])
        r1 = f(a=args)
        r2 = np.array([2, 3, 4])
        assert_array_equal(r1, r2)
        r1 = f(b=1, a=args)
        assert_array_equal(r1, r2)
        r1 = f(args, b=2)
        r2 = np.array([3, 4, 5])
        assert_array_equal(r1, r2)

    def test_keywords3_ticket_2100(self):
        # Test excluded with mixed positional and kwargs: ticket 2100
        def mypolyval(x, p):
            _p = list(p)
            res = _p.pop(0)
            while _p:
                res = res * x + _p.pop(0)
            return res

        vpolyval = np.vectorize(mypolyval, excluded=["p", 1])
        ans = [3, 6]
        assert_array_equal(ans, vpolyval(x=[0, 1], p=[1, 2, 3]))
        assert_array_equal(ans, vpolyval([0, 1], p=[1, 2, 3]))
        assert_array_equal(ans, vpolyval([0, 1], [1, 2, 3]))

    def test_keywords4_ticket_2100(self):
        # Test vectorizing function with no positional args.
        @vectorize
        def f(**kw):
            res = 1.0
            for _k in kw:
                res *= kw[_k]
            return res

        assert_array_equal(f(a=[1, 2], b=[3, 4]), [3, 8])

    def test_keywords5_ticket_2100(self):
        # Test vectorizing function with no kwargs args.
        @vectorize
        def f(*v):
            return np.prod(v)

        assert_array_equal(f([1, 2], [3, 4]), [3, 8])

    def test_coverage1_ticket_2100(self):
        def foo():
            return 1

        f = vectorize(foo)
        assert_array_equal(f(), 1)

    def test_assigning_docstring(self):
        def foo(x):
            """Original documentation"""
            return x

        f = vectorize(foo)
        assert_equal(f.__doc__, foo.__doc__)

        doc = "Provided documentation"
        f = vectorize(foo, doc=doc)
        assert_equal(f.__doc__, doc)

    def test_UnboundMethod_ticket_1156(self):
        # Regression test for issue 1156
        class Foo:
            b = 2

            def bar(self, a):
                return a**self.b

        assert_array_equal(vectorize(Foo().bar)(np.arange(9)), np.arange(9) ** 2)
        assert_array_equal(vectorize(Foo.bar)(Foo(), np.arange(9)), np.arange(9) ** 2)

    def test_execution_order_ticket_1487(self):
        # Regression test for dependence on execution order: issue 1487
        f1 = vectorize(lambda x: x)
        res1a = f1(np.arange(3))
        res1b = f1(np.arange(0.1, 3))
        f2 = vectorize(lambda x: x)
        res2b = f2(np.arange(0.1, 3))
        res2a = f2(np.arange(3))
        assert_equal(res1a, res2a)
        assert_equal(res1b, res2b)

    def test_string_ticket_1892(self):
        # Test vectorization over strings: issue 1892.
        f = np.vectorize(lambda x: x)
        s = string.digits * 10
        assert_equal(s, f(s))

    def test_cache(self):
        # Ensure that vectorized func called exactly once per argument.
        _calls = [0]

        @vectorize
        def f(x):
            _calls[0] += 1
            return x**2

        f.cache = True
        x = np.arange(5)
        assert_array_equal(f(x), x * x)
        assert_equal(_calls[0], len(x))

    def test_otypes(self):
        f = np.vectorize(lambda x: x)
        f.otypes = "i"
        x = np.arange(5)
        assert_array_equal(f(x), x)

    def test_signature_mean_last(self):
        def mean(a):
            return a.mean()

        f = vectorize(mean, signature="(n)->()")
        r = f([[1, 3], [2, 4]])
        assert_array_equal(r, [2, 3])

    def test_signature_center(self):
        def center(a):
            return a - a.mean()

        f = vectorize(center, signature="(n)->(n)")
        r = f([[1, 3], [2, 4]])
        assert_array_equal(r, [[-1, 1], [-1, 1]])

    def test_signature_two_outputs(self):
        f = vectorize(lambda x: (x, x), signature="()->(),()")
        r = f([1, 2, 3])
        assert_(isinstance(r, tuple) and len(r) == 2)
        assert_array_equal(r[0], [1, 2, 3])
        assert_array_equal(r[1], [1, 2, 3])

    def test_signature_outer(self):
        f = vectorize(np.outer, signature="(a),(b)->(a,b)")
        r = f([1, 2], [1, 2, 3])
        assert_array_equal(r, [[1, 2, 3], [2, 4, 6]])

        r = f([[[1, 2]]], [1, 2, 3])
        assert_array_equal(r, [[[[1, 2, 3], [2, 4, 6]]]])

        r = f([[1, 0], [2, 0]], [1, 2, 3])
        assert_array_equal(r, [[[1, 2, 3], [0, 0, 0]], [[2, 4, 6], [0, 0, 0]]])

        r = f([1, 2], [[1, 2, 3], [0, 0, 0]])
        assert_array_equal(r, [[[1, 2, 3], [2, 4, 6]], [[0, 0, 0], [0, 0, 0]]])

    def test_signature_computed_size(self):
        f = vectorize(lambda x: x[:-1], signature="(n)->(m)")
        r = f([1, 2, 3])
        assert_array_equal(r, [1, 2])

        r = f([[1, 2, 3], [2, 3, 4]])
        assert_array_equal(r, [[1, 2], [2, 3]])

    def test_signature_excluded(self):
        def foo(a, b=1):
            return a + b

        f = vectorize(foo, signature="()->()", excluded={"b"})
        assert_array_equal(f([1, 2, 3]), [2, 3, 4])
        assert_array_equal(f([1, 2, 3], b=0), [1, 2, 3])

    def test_signature_otypes(self):
        f = vectorize(lambda x: x, signature="(n)->(n)", otypes=["float64"])
        r = f([1, 2, 3])
        assert_equal(r.dtype, np.dtype("float64"))
        assert_array_equal(r, [1, 2, 3])

    def test_signature_invalid_inputs(self):
        f = vectorize(operator.add, signature="(n),(n)->(n)")
        with assert_raises_regex(TypeError, "wrong number of positional"):
            f([1, 2])
        with assert_raises_regex(ValueError, "does not have enough dimensions"):
            f(1, 2)
        with assert_raises_regex(ValueError, "inconsistent size for core dimension"):
            f([1, 2], [1, 2, 3])

        f = vectorize(operator.add, signature="()->()")
        with assert_raises_regex(TypeError, "wrong number of positional"):
            f(1, 2)

    def test_signature_invalid_outputs(self):
        f = vectorize(lambda x: x[:-1], signature="(n)->(n)")
        with assert_raises_regex(ValueError, "inconsistent size for core dimension"):
            f([1, 2, 3])

        f = vectorize(lambda x: x, signature="()->(),()")
        with assert_raises_regex(ValueError, "wrong number of outputs"):
            f(1)

        f = vectorize(lambda x: (x, x), signature="()->()")
        with assert_raises_regex(ValueError, "wrong number of outputs"):
            f([1, 2])

    def test_size_zero_output(self):
        # see issue 5868
        f = np.vectorize(lambda x: x)
        x = np.zeros([0, 5], dtype=int)
        with assert_raises_regex(ValueError, "otypes"):
            f(x)

        f.otypes = "i"
        assert_array_equal(f(x), x)

        f = np.vectorize(lambda x: x, signature="()->()")
        with assert_raises_regex(ValueError, "otypes"):
            f(x)

        f = np.vectorize(lambda x: x, signature="()->()", otypes="i")
        assert_array_equal(f(x), x)

        f = np.vectorize(lambda x: x, signature="(n)->(n)", otypes="i")
        assert_array_equal(f(x), x)

        f = np.vectorize(lambda x: x, signature="(n)->(n)")
        assert_array_equal(f(x.T), x.T)

        f = np.vectorize(lambda x: [x], signature="()->(n)", otypes="i")
        with assert_raises_regex(ValueError, "new output dimensions"):
            f(x)


@xpassIfTorchDynamo_np  # (reason="TODO: implement")
class TestDigitize(TestCase):
    def test_forward(self):
        x = np.arange(-6, 5)
        bins = np.arange(-5, 5)
        assert_array_equal(digitize(x, bins), np.arange(11))

    def test_reverse(self):
        x = np.arange(5, -6, -1)
        bins = np.arange(5, -5, -1)
        assert_array_equal(digitize(x, bins), np.arange(11))

    def test_random(self):
        x = rand(10)
        bin = np.linspace(x.min(), x.max(), 10)
        assert_(np.all(digitize(x, bin) != 0))

    def test_right_basic(self):
        x = [1, 5, 4, 10, 8, 11, 0]
        bins = [1, 5, 10]
        default_answer = [1, 2, 1, 3, 2, 3, 0]
        assert_array_equal(digitize(x, bins), default_answer)
        right_answer = [0, 1, 1, 2, 2, 3, 0]
        assert_array_equal(digitize(x, bins, True), right_answer)

    def test_right_open(self):
        x = np.arange(-6, 5)
        bins = np.arange(-6, 4)
        assert_array_equal(digitize(x, bins, True), np.arange(11))

    def test_right_open_reverse(self):
        x = np.arange(5, -6, -1)
        bins = np.arange(4, -6, -1)
        assert_array_equal(digitize(x, bins, True), np.arange(11))

    def test_right_open_random(self):
        x = rand(10)
        bins = np.linspace(x.min(), x.max(), 10)
        assert_(np.all(digitize(x, bins, True) != 10))

    def test_monotonic(self):
        x = [-1, 0, 1, 2]
        bins = [0, 0, 1]
        assert_array_equal(digitize(x, bins, False), [0, 2, 3, 3])
        assert_array_equal(digitize(x, bins, True), [0, 0, 2, 3])
        bins = [1, 1, 0]
        assert_array_equal(digitize(x, bins, False), [3, 2, 0, 0])
        assert_array_equal(digitize(x, bins, True), [3, 3, 2, 0])
        bins = [1, 1, 1, 1]
        assert_array_equal(digitize(x, bins, False), [0, 0, 4, 4])
        assert_array_equal(digitize(x, bins, True), [0, 0, 0, 4])
        bins = [0, 0, 1, 0]
        assert_raises(ValueError, digitize, x, bins)
        bins = [1, 1, 0, 1]
        assert_raises(ValueError, digitize, x, bins)

    def test_casting_error(self):
        x = [1, 2, 3 + 1.0j]
        bins = [1, 2, 3]
        assert_raises(TypeError, digitize, x, bins)
        x, bins = bins, x
        assert_raises(TypeError, digitize, x, bins)

    def test_large_integers_increasing(self):
        # gh-11022
        x = 2**54  # loses precision in a float
        assert_equal(np.digitize(x, [x - 1, x + 1]), 1)

    @xfail  # "gh-11022: np.core.multiarray._monoticity loses precision"
    def test_large_integers_decreasing(self):
        # gh-11022
        x = 2**54  # loses precision in a float
        assert_equal(np.digitize(x, [x + 1, x - 1]), 1)


@skip  # (reason="TODO: implement; here unwrap if from numpy")
class TestUnwrap(TestCase):
    def test_simple(self):
        # check that unwrap removes jumps greater that 2*pi
        assert_array_equal(unwrap([1, 1 + 2 * np.pi]), [1, 1])
        # check that unwrap maintains continuity
        assert_(np.all(diff(unwrap(rand(10) * 100)) < np.pi))

    def test_period(self):
        # check that unwrap removes jumps greater that 255
        assert_array_equal(unwrap([1, 1 + 256], period=255), [1, 2])
        # check that unwrap maintains continuity
        assert_(np.all(diff(unwrap(rand(10) * 1000, period=255)) < 255))
        # check simple case
        simple_seq = np.array([0, 75, 150, 225, 300])
        wrap_seq = np.mod(simple_seq, 255)
        assert_array_equal(unwrap(wrap_seq, period=255), simple_seq)
        # check custom discont value
        uneven_seq = np.array([0, 75, 150, 225, 300, 430])
        wrap_uneven = np.mod(uneven_seq, 250)
        no_discont = unwrap(wrap_uneven, period=250)
        assert_array_equal(no_discont, [0, 75, 150, 225, 300, 180])
        sm_discont = unwrap(wrap_uneven, period=250, discont=140)
        assert_array_equal(sm_discont, [0, 75, 150, 225, 300, 430])
        assert sm_discont.dtype == wrap_uneven.dtype


@instantiate_parametrized_tests
class TestFilterwindows(TestCase):
    @parametrize(
        "dtype", "Bbhil" + "efd"
    )  # np.typecodes["AllInteger"] + np.typecodes["Float"])
    @parametrize("M", [0, 1, 10])
    def test_hanning(self, dtype: str, M: int) -> None:
        scalar = M

        w = hanning(scalar)
        ref_dtype = np.result_type(dtype, np.float64)
        assert w.dtype == ref_dtype

        # check symmetry
        assert_allclose(w, flipud(w), atol=1e-15)

        # check known value
        if scalar < 1:
            assert_array_equal(w, np.array([]))
        elif scalar == 1:
            assert_array_equal(w, np.ones(1))
        else:
            assert_almost_equal(np.sum(w, axis=0), 4.500, 4)

    @parametrize(
        "dtype", "Bbhil" + "efd"
    )  # np.typecodes["AllInteger"] + np.typecodes["Float"])
    @parametrize("M", [0, 1, 10])
    def test_hamming(self, dtype: str, M: int) -> None:
        scalar = M

        w = hamming(scalar)
        ref_dtype = np.result_type(dtype, np.float64)
        assert w.dtype == ref_dtype

        # check symmetry
        assert_allclose(w, flipud(w), atol=1e-15)

        # check known value
        if scalar < 1:
            assert_array_equal(w, np.array([]))
        elif scalar == 1:
            assert_array_equal(w, np.ones(1))
        else:
            assert_almost_equal(np.sum(w, axis=0), 4.9400, 4)

    @parametrize(
        "dtype", "Bbhil" + "efd"
    )  # np.typecodes["AllInteger"] + np.typecodes["Float"])
    @parametrize("M", [0, 1, 10])
    def test_bartlett(self, dtype: str, M: int) -> None:
        scalar = M

        w = bartlett(scalar)
        ref_dtype = np.result_type(dtype, np.float64)
        assert w.dtype == ref_dtype

        # check symmetry
        assert_allclose(w, flipud(w), atol=1e-15)

        # check known value
        if scalar < 1:
            assert_array_equal(w, np.array([]))
        elif scalar == 1:
            assert_array_equal(w, np.ones(1))
        else:
            assert_almost_equal(np.sum(w, axis=0), 4.4444, 4)

    @parametrize(
        "dtype", "Bbhil" + "efd"
    )  # np.typecodes["AllInteger"] + np.typecodes["Float"])
    @parametrize("M", [0, 1, 10])
    def test_blackman(self, dtype: str, M: int) -> None:
        scalar = M

        w = blackman(scalar)
        ref_dtype = np.result_type(dtype, np.float64)
        assert w.dtype == ref_dtype

        # check symmetry
        assert_allclose(w, flipud(w), atol=1e-15)

        # check known value
        if scalar < 1:
            assert_array_equal(w, np.array([]))
        elif scalar == 1:
            assert_array_equal(w, np.ones(1))
        else:
            assert_almost_equal(np.sum(w, axis=0), 3.7800, 4)

    @parametrize(
        "dtype", "Bbhil" + "efd"
    )  # np.typecodes["AllInteger"] + np.typecodes["Float"])
    @parametrize("M", [0, 1, 10])
    def test_kaiser(self, dtype: str, M: int) -> None:
        scalar = M

        w = kaiser(scalar, 0)
        ref_dtype = np.result_type(dtype, np.float64)
        assert w.dtype == ref_dtype

        # check symmetry
        assert_equal(w, flipud(w))

        # check known value
        if scalar < 1:
            assert_array_equal(w, np.array([]))
        elif scalar == 1:
            assert_array_equal(w, np.ones(1))
        else:
            assert_almost_equal(np.sum(w, axis=0), 10, 15)


@xpassIfTorchDynamo_np  # (reason="TODO: implement")
class TestTrapz(TestCase):
    def test_simple(self):
        x = np.arange(-10, 10, 0.1)
        r = trapz(np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi), dx=0.1)
        # check integral of normal equals 1
        assert_almost_equal(r, 1, 7)

    def test_ndim(self):
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 2, 8)
        z = np.linspace(0, 3, 13)

        wx = np.ones_like(x) * (x[1] - x[0])
        wx[0] /= 2
        wx[-1] /= 2
        wy = np.ones_like(y) * (y[1] - y[0])
        wy[0] /= 2
        wy[-1] /= 2
        wz = np.ones_like(z) * (z[1] - z[0])
        wz[0] /= 2
        wz[-1] /= 2

        q = x[:, None, None] + y[None, :, None] + z[None, None, :]

        qx = (q * wx[:, None, None]).sum(axis=0)
        qy = (q * wy[None, :, None]).sum(axis=1)
        qz = (q * wz[None, None, :]).sum(axis=2)

        # n-d `x`
        r = trapz(q, x=x[:, None, None], axis=0)
        assert_almost_equal(r, qx)
        r = trapz(q, x=y[None, :, None], axis=1)
        assert_almost_equal(r, qy)
        r = trapz(q, x=z[None, None, :], axis=2)
        assert_almost_equal(r, qz)

        # 1-d `x`
        r = trapz(q, x=x, axis=0)
        assert_almost_equal(r, qx)
        r = trapz(q, x=y, axis=1)
        assert_almost_equal(r, qy)
        r = trapz(q, x=z, axis=2)
        assert_almost_equal(r, qz)


class TestSinc(TestCase):
    def test_simple(self):
        assert_(sinc(0) == 1)
        w = sinc(np.linspace(-1, 1, 100))
        # check symmetry
        assert_array_almost_equal(w, np.flipud(w), 7)

    def test_array_like(self):
        x = [0, 0.5]
        y1 = sinc(np.array(x))
        y2 = sinc(list(x))
        y3 = sinc(tuple(x))
        assert_array_equal(y1, y2)
        assert_array_equal(y1, y3)


class TestUnique(TestCase):
    def test_simple(self):
        x = np.array([4, 3, 2, 1, 1, 2, 3, 4, 0])
        assert_(np.all(unique(x) == [0, 1, 2, 3, 4]))

        assert_(unique(np.array([1, 1, 1, 1, 1])) == np.array([1]))

    @xpassIfTorchDynamo_np  # (reason="unique not implemented for 'ComplexDouble'")
    def test_simple_complex(self):
        x = np.array([5 + 6j, 1 + 1j, 1 + 10j, 10, 5 + 6j])
        assert_(np.all(unique(x) == [1 + 1j, 1 + 10j, 5 + 6j, 10]))


@xpassIfTorchDynamo_np  # (reason="TODO: implement")
class TestCheckFinite(TestCase):
    def test_simple(self):
        a = [1, 2, 3]
        b = [1, 2, np.inf]
        c = [1, 2, np.nan]
        np.lib.asarray_chkfinite(a)
        assert_raises(ValueError, np.lib.asarray_chkfinite, b)
        assert_raises(ValueError, np.lib.asarray_chkfinite, c)

    def test_dtype_order(self):
        # Regression test for missing dtype and order arguments
        a = [1, 2, 3]
        a = np.lib.asarray_chkfinite(a, order="F", dtype=np.float64)
        assert_(a.dtype == np.float64)


@instantiate_parametrized_tests
class TestCorrCoef(TestCase):
    A = np.array(
        [
            [0.15391142, 0.18045767, 0.14197213],
            [0.70461506, 0.96474128, 0.27906989],
            [0.9297531, 0.32296769, 0.19267156],
        ]
    )
    B = np.array(
        [
            [0.10377691, 0.5417086, 0.49807457],
            [0.82872117, 0.77801674, 0.39226705],
            [0.9314666, 0.66800209, 0.03538394],
        ]
    )
    res1 = np.array(
        [
            [1.0, 0.9379533, -0.04931983],
            [0.9379533, 1.0, 0.30007991],
            [-0.04931983, 0.30007991, 1.0],
        ]
    )
    res2 = np.array(
        [
            [1.0, 0.9379533, -0.04931983, 0.30151751, 0.66318558, 0.51532523],
            [0.9379533, 1.0, 0.30007991, -0.04781421, 0.88157256, 0.78052386],
            [-0.04931983, 0.30007991, 1.0, -0.96717111, 0.71483595, 0.83053601],
            [0.30151751, -0.04781421, -0.96717111, 1.0, -0.51366032, -0.66173113],
            [0.66318558, 0.88157256, 0.71483595, -0.51366032, 1.0, 0.98317823],
            [0.51532523, 0.78052386, 0.83053601, -0.66173113, 0.98317823, 1.0],
        ]
    )

    def test_non_array(self):
        assert_almost_equal(
            np.corrcoef([0, 1, 0], [1, 0, 1]), [[1.0, -1.0], [-1.0, 1.0]]
        )

    def test_simple(self):
        tgt1 = corrcoef(self.A)
        assert_almost_equal(tgt1, self.res1)
        assert_(np.all(np.abs(tgt1) <= 1.0))

        tgt2 = corrcoef(self.A, self.B)
        assert_almost_equal(tgt2, self.res2)
        assert_(np.all(np.abs(tgt2) <= 1.0))

    @skip(reason="deprecated in numpy, ignore")
    def test_ddof(self):
        # ddof raises DeprecationWarning
        with suppress_warnings() as sup:
            warnings.simplefilter("always")
            assert_warns(DeprecationWarning, corrcoef, self.A, ddof=-1)
            sup.filter(DeprecationWarning)
            # ddof has no or negligible effect on the function
            assert_almost_equal(corrcoef(self.A, ddof=-1), self.res1)
            assert_almost_equal(corrcoef(self.A, self.B, ddof=-1), self.res2)
            assert_almost_equal(corrcoef(self.A, ddof=3), self.res1)
            assert_almost_equal(corrcoef(self.A, self.B, ddof=3), self.res2)

    @skip(reason="deprecated in numpy, ignore")
    def test_bias(self):
        # bias raises DeprecationWarning
        with suppress_warnings() as sup:
            warnings.simplefilter("always")
            assert_warns(DeprecationWarning, corrcoef, self.A, self.B, 1, 0)
            assert_warns(DeprecationWarning, corrcoef, self.A, bias=0)
            sup.filter(DeprecationWarning)
            # bias has no or negligible effect on the function
            assert_almost_equal(corrcoef(self.A, bias=1), self.res1)

    def test_complex(self):
        x = np.array([[1, 2, 3], [1j, 2j, 3j]])
        res = corrcoef(x)
        tgt = np.array([[1.0, -1.0j], [1.0j, 1.0]])
        assert_allclose(res, tgt)
        assert_(np.all(np.abs(res) <= 1.0))

    def test_xy(self):
        x = np.array([[1, 2, 3]])
        y = np.array([[1j, 2j, 3j]])
        assert_allclose(np.corrcoef(x, y), np.array([[1.0, -1.0j], [1.0j, 1.0]]))

    def test_empty(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", RuntimeWarning)
            assert_array_equal(corrcoef(np.array([])), np.nan)
            assert_array_equal(
                corrcoef(np.array([]).reshape(0, 2)), np.array([]).reshape(0, 0)
            )
            assert_array_equal(
                corrcoef(np.array([]).reshape(2, 0)),
                np.array([[np.nan, np.nan], [np.nan, np.nan]]),
            )

    def test_extreme(self):
        x = [[1e-100, 1e100], [1e100, 1e-100]]
        c = corrcoef(x)
        assert_array_almost_equal(c, np.array([[1.0, -1.0], [-1.0, 1.0]]))
        assert_(np.all(np.abs(c) <= 1.0))

    @parametrize("test_type", [np.half, np.single, np.double])
    def test_corrcoef_dtype(self, test_type):
        cast_A = self.A.astype(test_type)
        res = corrcoef(cast_A, dtype=test_type)
        assert test_type == res.dtype


@instantiate_parametrized_tests
class TestCov(TestCase):
    x1 = np.array([[0, 2], [1, 1], [2, 0]]).T
    res1 = np.array([[1.0, -1.0], [-1.0, 1.0]])
    x2 = np.array([0.0, 1.0, 2.0], ndmin=2)
    frequencies = np.array([1, 4, 1])
    x2_repeats = np.array([[0.0], [1.0], [1.0], [1.0], [1.0], [2.0]]).T
    res2 = np.array([[0.4, -0.4], [-0.4, 0.4]])
    unit_frequencies = np.ones(3, dtype=np.int_)
    weights = np.array([1.0, 4.0, 1.0])
    res3 = np.array([[2.0 / 3.0, -2.0 / 3.0], [-2.0 / 3.0, 2.0 / 3.0]])
    unit_weights = np.ones(3)
    x3 = np.array([0.3942, 0.5969, 0.7730, 0.9918, 0.7964])

    def test_basic(self):
        assert_allclose(cov(self.x1), self.res1)

    def test_complex(self):
        x = np.array([[1, 2, 3], [1j, 2j, 3j]])
        res = np.array([[1.0, -1.0j], [1.0j, 1.0]])
        assert_allclose(cov(x), res)
        assert_allclose(cov(x, aweights=np.ones(3)), res)

    def test_xy(self):
        x = np.array([[1, 2, 3]])
        y = np.array([[1j, 2j, 3j]])
        assert_allclose(cov(x, y), np.array([[1.0, -1.0j], [1.0j, 1.0]]))

    def test_empty(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", RuntimeWarning)
            assert_array_equal(cov(np.array([])), np.nan)
            assert_array_equal(
                cov(np.array([]).reshape(0, 2)), np.array([]).reshape(0, 0)
            )
            assert_array_equal(
                cov(np.array([]).reshape(2, 0)),
                np.array([[np.nan, np.nan], [np.nan, np.nan]]),
            )

    def test_wrong_ddof(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", RuntimeWarning)
            assert_array_equal(
                cov(self.x1, ddof=5), np.array([[np.inf, -np.inf], [-np.inf, np.inf]])
            )

    def test_1D_rowvar(self):
        assert_allclose(cov(self.x3), cov(self.x3, rowvar=False))
        y = np.array([0.0780, 0.3107, 0.2111, 0.0334, 0.8501])
        assert_allclose(cov(self.x3, y), cov(self.x3, y, rowvar=False))

    def test_1D_variance(self):
        assert_allclose(cov(self.x3, ddof=1), np.var(self.x3, ddof=1))

    def test_fweights(self):
        assert_allclose(cov(self.x2, fweights=self.frequencies), cov(self.x2_repeats))
        assert_allclose(cov(self.x1, fweights=self.frequencies), self.res2)
        assert_allclose(cov(self.x1, fweights=self.unit_frequencies), self.res1)
        nonint = self.frequencies + 0.5
        assert_raises((TypeError, RuntimeError), cov, self.x1, fweights=nonint)
        f = np.ones((2, 3), dtype=np.int_)
        assert_raises(RuntimeError, cov, self.x1, fweights=f)
        f = np.ones(2, dtype=np.int_)
        assert_raises(RuntimeError, cov, self.x1, fweights=f)
        f = -1 * np.ones(3, dtype=np.int_)
        assert_raises((ValueError, RuntimeError), cov, self.x1, fweights=f)

    def test_aweights(self):
        assert_allclose(cov(self.x1, aweights=self.weights), self.res3)
        assert_allclose(
            cov(self.x1, aweights=3.0 * self.weights),
            cov(self.x1, aweights=self.weights),
        )
        assert_allclose(cov(self.x1, aweights=self.unit_weights), self.res1)
        w = np.ones((2, 3))
        assert_raises(RuntimeError, cov, self.x1, aweights=w)
        w = np.ones(2)
        assert_raises(RuntimeError, cov, self.x1, aweights=w)
        w = -1.0 * np.ones(3)
        assert_raises((ValueError, RuntimeError), cov, self.x1, aweights=w)

    def test_unit_fweights_and_aweights(self):
        assert_allclose(
            cov(self.x2, fweights=self.frequencies, aweights=self.unit_weights),
            cov(self.x2_repeats),
        )
        assert_allclose(
            cov(self.x1, fweights=self.frequencies, aweights=self.unit_weights),
            self.res2,
        )
        assert_allclose(
            cov(self.x1, fweights=self.unit_frequencies, aweights=self.unit_weights),
            self.res1,
        )
        assert_allclose(
            cov(self.x1, fweights=self.unit_frequencies, aweights=self.weights),
            self.res3,
        )
        assert_allclose(
            cov(self.x1, fweights=self.unit_frequencies, aweights=3.0 * self.weights),
            cov(self.x1, aweights=self.weights),
        )
        assert_allclose(
            cov(self.x1, fweights=self.unit_frequencies, aweights=self.unit_weights),
            self.res1,
        )

    @parametrize("test_type", [np.half, np.single, np.double])
    def test_cov_dtype(self, test_type):
        cast_x1 = self.x1.astype(test_type)
        res = cov(cast_x1, dtype=test_type)
        assert test_type == res.dtype


class Test_I0(TestCase):
    def test_simple(self):
        assert_almost_equal(i0(0.5), np.array(1.0634833707413234))

        # need at least one test above 8, as the implementation is piecewise
        A = np.array([0.49842636, 0.6969809, 0.22011976, 0.0155549, 10.0])
        expected = np.array(
            [1.06307822, 1.12518299, 1.01214991, 1.00006049, 2815.71662847]
        )
        assert_almost_equal(i0(A), expected)
        assert_almost_equal(i0(-A), expected)

        B = np.array(
            [
                [0.827002, 0.99959078],
                [0.89694769, 0.39298162],
                [0.37954418, 0.05206293],
                [0.36465447, 0.72446427],
                [0.48164949, 0.50324519],
            ]
        )
        assert_almost_equal(
            i0(B),
            np.array(
                [
                    [1.17843223, 1.26583466],
                    [1.21147086, 1.03898290],
                    [1.03633899, 1.00067775],
                    [1.03352052, 1.13557954],
                    [1.05884290, 1.06432317],
                ]
            ),
        )
        # Regression test for gh-11205
        i0_0 = np.i0([0.0])
        assert_equal(i0_0.shape, (1,))
        assert_array_equal(np.i0([0.0]), np.array([1.0]))

    def test_complex(self):
        a = np.array([0, 1 + 2j])
        with pytest.raises(
            (TypeError, RuntimeError),
            # match="i0 not supported for complex values"
        ):
            i0(a)


class TestKaiser(TestCase):
    def test_simple(self):
        assert_(np.isfinite(kaiser(1, 1.0)))
        assert_almost_equal(kaiser(0, 1.0), np.array([]))
        assert_almost_equal(kaiser(2, 1.0), np.array([0.78984831, 0.78984831]))
        assert_almost_equal(
            kaiser(5, 1.0),
            np.array([0.78984831, 0.94503323, 1.0, 0.94503323, 0.78984831]),
        )
        assert_almost_equal(
            kaiser(5, 1.56789),
            np.array([0.58285404, 0.88409679, 1.0, 0.88409679, 0.58285404]),
        )

    def test_int_beta(self):
        kaiser(3, 4)


@skip(reason="msort is deprecated, do not bother")
class TestMsort(TestCase):
    def test_simple(self):
        A = np.array(
            [
                [0.44567325, 0.79115165, 0.54900530],
                [0.36844147, 0.37325583, 0.96098397],
                [0.64864341, 0.52929049, 0.39172155],
            ]
        )
        with pytest.warns(DeprecationWarning, match="msort is deprecated"):
            assert_almost_equal(
                msort(A),
                np.array(
                    [
                        [0.36844147, 0.37325583, 0.39172155],
                        [0.44567325, 0.52929049, 0.54900530],
                        [0.64864341, 0.79115165, 0.96098397],
                    ]
                ),
            )


class TestMeshgrid(TestCase):
    def test_simple(self):
        [X, Y] = meshgrid([1, 2, 3], [4, 5, 6, 7])
        assert_array_equal(X, np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]))
        assert_array_equal(Y, np.array([[4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]]))

    def test_single_input(self):
        [X] = meshgrid([1, 2, 3, 4])
        assert_array_equal(X, np.array([1, 2, 3, 4]))

    def test_no_input(self):
        args = []
        assert_array_equal([], meshgrid(*args))
        assert_array_equal([], meshgrid(*args, copy=False))

    def test_indexing(self):
        x = [1, 2, 3]
        y = [4, 5, 6, 7]
        [X, Y] = meshgrid(x, y, indexing="ij")
        assert_array_equal(X, np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]))
        assert_array_equal(Y, np.array([[4, 5, 6, 7], [4, 5, 6, 7], [4, 5, 6, 7]]))

        # Test expected shapes:
        z = [8, 9]
        assert_(meshgrid(x, y)[0].shape == (4, 3))
        assert_(meshgrid(x, y, indexing="ij")[0].shape == (3, 4))
        assert_(meshgrid(x, y, z)[0].shape == (4, 3, 2))
        assert_(meshgrid(x, y, z, indexing="ij")[0].shape == (3, 4, 2))

        assert_raises(ValueError, meshgrid, x, y, indexing="notvalid")

    def test_sparse(self):
        [X, Y] = meshgrid([1, 2, 3], [4, 5, 6, 7], sparse=True)
        assert_array_equal(X, np.array([[1, 2, 3]]))
        assert_array_equal(Y, np.array([[4], [5], [6], [7]]))

    def test_invalid_arguments(self):
        # Test that meshgrid complains about invalid arguments
        # Regression test for issue #4755:
        # https://github.com/numpy/numpy/issues/4755
        assert_raises(TypeError, meshgrid, [1, 2, 3], [4, 5, 6, 7], indices="ij")

    def test_return_type(self):
        # Test for appropriate dtype in returned arrays.
        # Regression test for issue #5297
        # https://github.com/numpy/numpy/issues/5297
        x = np.arange(0, 10, dtype=np.float32)
        y = np.arange(10, 20, dtype=np.float64)

        X, Y = np.meshgrid(x, y)

        assert_(X.dtype == x.dtype)
        assert_(Y.dtype == y.dtype)

        # copy
        X, Y = np.meshgrid(x, y, copy=True)

        assert_(X.dtype == x.dtype)
        assert_(Y.dtype == y.dtype)

        # sparse
        X, Y = np.meshgrid(x, y, sparse=True)

        assert_(X.dtype == x.dtype)
        assert_(Y.dtype == y.dtype)

    def test_writeback(self):
        # Issue 8561
        X = np.array([1.1, 2.2])
        Y = np.array([3.3, 4.4])
        x, y = np.meshgrid(X, Y, sparse=False, copy=True)

        x[0, :] = 0
        assert_equal(x[0, :], 0)
        assert_equal(x[1, :], X)

    def test_nd_shape(self):
        a, b, c, d, e = np.meshgrid(*([0] * i for i in range(1, 6)))
        expected_shape = (2, 1, 3, 4, 5)
        assert_equal(a.shape, expected_shape)
        assert_equal(b.shape, expected_shape)
        assert_equal(c.shape, expected_shape)
        assert_equal(d.shape, expected_shape)
        assert_equal(e.shape, expected_shape)

    def test_nd_values(self):
        a, b, c = np.meshgrid([0], [1, 2], [3, 4, 5])
        assert_equal(a, [[[0, 0, 0]], [[0, 0, 0]]])
        assert_equal(b, [[[1, 1, 1]], [[2, 2, 2]]])
        assert_equal(c, [[[3, 4, 5]], [[3, 4, 5]]])

    def test_nd_indexing(self):
        a, b, c = np.meshgrid([0], [1, 2], [3, 4, 5], indexing="ij")
        assert_equal(a, [[[0, 0, 0], [0, 0, 0]]])
        assert_equal(b, [[[1, 1, 1], [2, 2, 2]]])
        assert_equal(c, [[[3, 4, 5], [3, 4, 5]]])


@xfail  # (reason="TODO: implement")
class TestPiecewise(TestCase):
    def test_simple(self):
        # Condition is single bool list
        x = piecewise([0, 0], [True, False], [1])
        assert_array_equal(x, [1, 0])

        # List of conditions: single bool list
        x = piecewise([0, 0], [[True, False]], [1])
        assert_array_equal(x, [1, 0])

        # Conditions is single bool array
        x = piecewise([0, 0], np.array([True, False]), [1])
        assert_array_equal(x, [1, 0])

        # Condition is single int array
        x = piecewise([0, 0], np.array([1, 0]), [1])
        assert_array_equal(x, [1, 0])

        # List of conditions: int array
        x = piecewise([0, 0], [np.array([1, 0])], [1])
        assert_array_equal(x, [1, 0])

        x = piecewise([0, 0], [[False, True]], [lambda x: -1])
        assert_array_equal(x, [0, -1])

        assert_raises_regex(
            ValueError,
            "1 or 2 functions are expected",
            piecewise,
            [0, 0],
            [[False, True]],
            [],
        )
        assert_raises_regex(
            ValueError,
            "1 or 2 functions are expected",
            piecewise,
            [0, 0],
            [[False, True]],
            [1, 2, 3],
        )

    def test_two_conditions(self):
        x = piecewise([1, 2], [[True, False], [False, True]], [3, 4])
        assert_array_equal(x, [3, 4])

    def test_scalar_domains_three_conditions(self):
        x = piecewise(3, [True, False, False], [4, 2, 0])
        assert_equal(x, 4)

    def test_default(self):
        # No value specified for x[1], should be 0
        x = piecewise([1, 2], [True, False], [2])
        assert_array_equal(x, [2, 0])

        # Should set x[1] to 3
        x = piecewise([1, 2], [True, False], [2, 3])
        assert_array_equal(x, [2, 3])

    def test_0d(self):
        x = np.array(3)
        y = piecewise(x, x > 3, [4, 0])
        assert_(y.ndim == 0)
        assert_(y == 0)

        x = 5
        y = piecewise(x, [True, False], [1, 0])
        assert_(y.ndim == 0)
        assert_(y == 1)

        # With 3 ranges (It was failing, before)
        y = piecewise(x, [False, False, True], [1, 2, 3])
        assert_array_equal(y, 3)

    def test_0d_comparison(self):
        x = 3
        y = piecewise(x, [x <= 3, x > 3], [4, 0])  # Should succeed.
        assert_equal(y, 4)

        # With 3 ranges (It was failing, before)
        x = 4
        y = piecewise(x, [x <= 3, (x > 3) * (x <= 5), x > 5], [1, 2, 3])
        assert_array_equal(y, 2)

        assert_raises_regex(
            ValueError,
            "2 or 3 functions are expected",
            piecewise,
            x,
            [x <= 3, x > 3],
            [1],
        )
        assert_raises_regex(
            ValueError,
            "2 or 3 functions are expected",
            piecewise,
            x,
            [x <= 3, x > 3],
            [1, 1, 1, 1],
        )

    def test_0d_0d_condition(self):
        x = np.array(3)
        c = np.array(x > 3)
        y = piecewise(x, [c], [1, 2])
        assert_equal(y, 2)

    def test_multidimensional_extrafunc(self):
        x = np.array([[-2.5, -1.5, -0.5], [0.5, 1.5, 2.5]])
        y = piecewise(x, [x < 0, x >= 2], [-1, 1, 3])
        assert_array_equal(y, np.array([[-1.0, -1.0, -1.0], [3.0, 3.0, 1.0]]))


@instantiate_parametrized_tests
class TestBincount(TestCase):
    def test_simple(self):
        y = np.bincount(np.arange(4))
        assert_array_equal(y, np.ones(4))

    def test_simple2(self):
        y = np.bincount(np.array([1, 5, 2, 4, 1]))
        assert_array_equal(y, np.array([0, 2, 1, 0, 1, 1]))

    def test_simple_weight(self):
        x = np.arange(4)
        w = np.array([0.2, 0.3, 0.5, 0.1])
        y = np.bincount(x, w)
        assert_array_equal(y, w)

    def test_simple_weight2(self):
        x = np.array([1, 2, 4, 5, 2])
        w = np.array([0.2, 0.3, 0.5, 0.1, 0.2])
        y = np.bincount(x, w)
        assert_array_equal(y, np.array([0, 0.2, 0.5, 0, 0.5, 0.1]))

    def test_with_minlength(self):
        x = np.array([0, 1, 0, 1, 1])
        y = np.bincount(x, minlength=3)
        assert_array_equal(y, np.array([2, 3, 0]))
        x = []
        y = np.bincount(x, minlength=0)
        assert_array_equal(y, np.array([]))

    def test_with_minlength_smaller_than_maxvalue(self):
        x = np.array([0, 1, 1, 2, 2, 3, 3])
        y = np.bincount(x, minlength=2)
        assert_array_equal(y, np.array([1, 2, 2, 2]))
        y = np.bincount(x, minlength=0)
        assert_array_equal(y, np.array([1, 2, 2, 2]))

    def test_with_minlength_and_weights(self):
        x = np.array([1, 2, 4, 5, 2])
        w = np.array([0.2, 0.3, 0.5, 0.1, 0.2])
        y = np.bincount(x, w, 8)
        assert_array_equal(y, np.array([0, 0.2, 0.5, 0, 0.5, 0.1, 0, 0]))

    def test_empty(self):
        x = np.array([], dtype=int)
        y = np.bincount(x)
        assert_array_equal(x, y)

    def test_empty_with_minlength(self):
        x = np.array([], dtype=int)
        y = np.bincount(x, minlength=5)
        assert_array_equal(y, np.zeros(5, dtype=int))

    def test_with_incorrect_minlength(self):
        x = np.array([], dtype=int)
        assert_raises(
            TypeError,
            #       "'str' object cannot be interpreted",
            lambda: np.bincount(x, minlength="foobar"),
        )
        assert_raises(
            (ValueError, RuntimeError),
            #       "must not be negative",
            lambda: np.bincount(x, minlength=-1),
        )

        x = np.arange(5)
        assert_raises(
            TypeError,
            #       "'str' object cannot be interpreted",
            lambda: np.bincount(x, minlength="foobar"),
        )
        assert_raises(
            (ValueError, RuntimeError),
            #      "must not be negative",
            lambda: np.bincount(x, minlength=-1),
        )

    @skipIfTorchDynamo()  # flaky test
    @skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
    def test_dtype_reference_leaks(self):
        # gh-6805
        intp_refcount = sys.getrefcount(np.dtype(np.intp))
        double_refcount = sys.getrefcount(np.dtype(np.double))

        for _ in range(10):
            np.bincount([1, 2, 3])
        assert_equal(sys.getrefcount(np.dtype(np.intp)), intp_refcount)
        assert_equal(sys.getrefcount(np.dtype(np.double)), double_refcount)

        for _ in range(10):
            np.bincount([1, 2, 3], [4, 5, 6])
        assert_equal(sys.getrefcount(np.dtype(np.intp)), intp_refcount)
        assert_equal(sys.getrefcount(np.dtype(np.double)), double_refcount)

    @parametrize("vals", [[[2, 2]], 2])
    def test_error_not_1d(self, vals):
        # Test that values has to be 1-D (both as array and nested list)
        vals_arr = np.asarray(vals)
        with assert_raises((ValueError, RuntimeError)):
            np.bincount(vals_arr)
        with assert_raises((ValueError, RuntimeError)):
            np.bincount(vals)


parametrize_interp_sc = parametrize(
    "sc",
    [
        subtest(lambda x: np.float64(x), name="real"),
        subtest(lambda x: _make_complex(x, 0), name="complex-real"),
        subtest(lambda x: _make_complex(0, x), name="complex-imag"),
        subtest(lambda x: _make_complex(x, np.multiply(x, -2)), name="complex-both"),
    ],
)


@xpassIfTorchDynamo_np  # (reason="TODO: implement")
@instantiate_parametrized_tests
class TestInterp(TestCase):
    def test_exceptions(self):
        assert_raises(ValueError, interp, 0, [], [])
        assert_raises(ValueError, interp, 0, [0], [1, 2])
        assert_raises(ValueError, interp, 0, [0, 1], [1, 2], period=0)
        assert_raises(ValueError, interp, 0, [], [], period=360)
        assert_raises(ValueError, interp, 0, [0], [1, 2], period=360)

    def test_basic(self):
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        x0 = np.linspace(0, 1, 50)
        assert_almost_equal(np.interp(x0, x, y), x0)

    def test_right_left_behavior(self):
        # Needs range of sizes to test different code paths.
        # size ==1 is special cased, 1 < size < 5 is linear search, and
        # size >= 5 goes through local search and possibly binary search.
        for size in range(1, 10):
            xp = np.arange(size, dtype=np.double)
            yp = np.ones(size, dtype=np.double)
            incpts = np.array([-1, 0, size - 1, size], dtype=np.double)
            decpts = incpts[::-1]

            incres = interp(incpts, xp, yp)
            decres = interp(decpts, xp, yp)
            inctgt = np.array([1, 1, 1, 1], dtype=float)
            dectgt = inctgt[::-1]
            assert_equal(incres, inctgt)
            assert_equal(decres, dectgt)

            incres = interp(incpts, xp, yp, left=0)
            decres = interp(decpts, xp, yp, left=0)
            inctgt = np.array([0, 1, 1, 1], dtype=float)
            dectgt = inctgt[::-1]
            assert_equal(incres, inctgt)
            assert_equal(decres, dectgt)

            incres = interp(incpts, xp, yp, right=2)
            decres = interp(decpts, xp, yp, right=2)
            inctgt = np.array([1, 1, 1, 2], dtype=float)
            dectgt = inctgt[::-1]
            assert_equal(incres, inctgt)
            assert_equal(decres, dectgt)

            incres = interp(incpts, xp, yp, left=0, right=2)
            decres = interp(decpts, xp, yp, left=0, right=2)
            inctgt = np.array([0, 1, 1, 2], dtype=float)
            dectgt = inctgt[::-1]
            assert_equal(incres, inctgt)
            assert_equal(decres, dectgt)

    def test_scalar_interpolation_point(self):
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        x0 = 0
        assert_almost_equal(np.interp(x0, x, y), x0)
        x0 = 0.3
        assert_almost_equal(np.interp(x0, x, y), x0)
        x0 = np.float32(0.3)
        assert_almost_equal(np.interp(x0, x, y), x0)
        x0 = np.float64(0.3)
        assert_almost_equal(np.interp(x0, x, y), x0)
        x0 = np.nan
        assert_almost_equal(np.interp(x0, x, y), x0)

    def test_non_finite_behavior_exact_x(self):
        x = [1, 2, 2.5, 3, 4]
        xp = [1, 2, 3, 4]
        fp = [1, 2, np.inf, 4]
        assert_almost_equal(np.interp(x, xp, fp), [1, 2, np.inf, np.inf, 4])
        fp = [1, 2, np.nan, 4]
        assert_almost_equal(np.interp(x, xp, fp), [1, 2, np.nan, np.nan, 4])

    @parametrize_interp_sc
    def test_non_finite_any_nan(self, sc):
        """test that nans are propagated"""
        assert_equal(np.interp(0.5, [np.nan, 1], sc([0, 10])), sc(np.nan))
        assert_equal(np.interp(0.5, [0, np.nan], sc([0, 10])), sc(np.nan))
        assert_equal(np.interp(0.5, [0, 1], sc([np.nan, 10])), sc(np.nan))
        assert_equal(np.interp(0.5, [0, 1], sc([0, np.nan])), sc(np.nan))

    @parametrize_interp_sc
    def test_non_finite_inf(self, sc):
        """Test that interp between opposite infs gives nan"""
        assert_equal(np.interp(0.5, [-np.inf, +np.inf], sc([0, 10])), sc(np.nan))
        assert_equal(np.interp(0.5, [0, 1], sc([-np.inf, +np.inf])), sc(np.nan))
        assert_equal(np.interp(0.5, [0, 1], sc([+np.inf, -np.inf])), sc(np.nan))

        # unless the y values are equal
        assert_equal(np.interp(0.5, [-np.inf, +np.inf], sc([10, 10])), sc(10))

    @parametrize_interp_sc
    def test_non_finite_half_inf_xf(self, sc):
        """Test that interp where both axes have a bound at inf gives nan"""
        assert_equal(np.interp(0.5, [-np.inf, 1], sc([-np.inf, 10])), sc(np.nan))
        assert_equal(np.interp(0.5, [-np.inf, 1], sc([+np.inf, 10])), sc(np.nan))
        assert_equal(np.interp(0.5, [-np.inf, 1], sc([0, -np.inf])), sc(np.nan))
        assert_equal(np.interp(0.5, [-np.inf, 1], sc([0, +np.inf])), sc(np.nan))
        assert_equal(np.interp(0.5, [0, +np.inf], sc([-np.inf, 10])), sc(np.nan))
        assert_equal(np.interp(0.5, [0, +np.inf], sc([+np.inf, 10])), sc(np.nan))
        assert_equal(np.interp(0.5, [0, +np.inf], sc([0, -np.inf])), sc(np.nan))
        assert_equal(np.interp(0.5, [0, +np.inf], sc([0, +np.inf])), sc(np.nan))

    @parametrize_interp_sc
    def test_non_finite_half_inf_x(self, sc):
        """Test interp where the x axis has a bound at inf"""
        assert_equal(np.interp(0.5, [-np.inf, -np.inf], sc([0, 10])), sc(10))
        assert_equal(np.interp(0.5, [-np.inf, 1], sc([0, 10])), sc(10))
        assert_equal(np.interp(0.5, [0, +np.inf], sc([0, 10])), sc(0))
        assert_equal(np.interp(0.5, [+np.inf, +np.inf], sc([0, 10])), sc(0))

    @parametrize_interp_sc
    def test_non_finite_half_inf_f(self, sc):
        """Test interp where the f axis has a bound at inf"""
        assert_equal(np.interp(0.5, [0, 1], sc([0, -np.inf])), sc(-np.inf))
        assert_equal(np.interp(0.5, [0, 1], sc([0, +np.inf])), sc(+np.inf))
        assert_equal(np.interp(0.5, [0, 1], sc([-np.inf, 10])), sc(-np.inf))
        assert_equal(np.interp(0.5, [0, 1], sc([+np.inf, 10])), sc(+np.inf))
        assert_equal(np.interp(0.5, [0, 1], sc([-np.inf, -np.inf])), sc(-np.inf))
        assert_equal(np.interp(0.5, [0, 1], sc([+np.inf, +np.inf])), sc(+np.inf))

    def test_complex_interp(self):
        # test complex interpolation
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5) + (1 + np.linspace(0, 1, 5)) * 1.0j
        x0 = 0.3
        y0 = x0 + (1 + x0) * 1.0j
        assert_almost_equal(np.interp(x0, x, y), y0)
        # test complex left and right
        x0 = -1
        left = 2 + 3.0j
        assert_almost_equal(np.interp(x0, x, y, left=left), left)
        x0 = 2.0
        right = 2 + 3.0j
        assert_almost_equal(np.interp(x0, x, y, right=right), right)
        # test complex non finite
        x = [1, 2, 2.5, 3, 4]
        xp = [1, 2, 3, 4]
        fp = [1, 2 + 1j, np.inf, 4]
        y = [1, 2 + 1j, np.inf + 0.5j, np.inf, 4]
        assert_almost_equal(np.interp(x, xp, fp), y)
        # test complex periodic
        x = [-180, -170, -185, 185, -10, -5, 0, 365]
        xp = [190, -190, 350, -350]
        fp = [5 + 1.0j, 10 + 2j, 3 + 3j, 4 + 4j]
        y = [
            7.5 + 1.5j,
            5.0 + 1.0j,
            8.75 + 1.75j,
            6.25 + 1.25j,
            3.0 + 3j,
            3.25 + 3.25j,
            3.5 + 3.5j,
            3.75 + 3.75j,
        ]
        assert_almost_equal(np.interp(x, xp, fp, period=360), y)

    def test_zero_dimensional_interpolation_point(self):
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        x0 = np.array(0.3)
        assert_almost_equal(np.interp(x0, x, y), x0.item())

        xp = np.array([0, 2, 4])
        fp = np.array([1, -1, 1])

        actual = np.interp(np.array(1), xp, fp)
        assert_equal(actual, 0)
        assert_(isinstance(actual, (np.float64, np.ndarray)))

        actual = np.interp(np.array(4.5), xp, fp, period=4)
        assert_equal(actual, 0.5)
        assert_(isinstance(actual, (np.float64, np.ndarray)))

    def test_if_len_x_is_small(self):
        xp = np.arange(0, 10, 0.0001)
        fp = np.sin(xp)
        assert_almost_equal(np.interp(np.pi, xp, fp), 0.0)

    def test_period(self):
        x = [-180, -170, -185, 185, -10, -5, 0, 365]
        xp = [190, -190, 350, -350]
        fp = [5, 10, 3, 4]
        y = [7.5, 5.0, 8.75, 6.25, 3.0, 3.25, 3.5, 3.75]
        assert_almost_equal(np.interp(x, xp, fp, period=360), y)
        x = np.array(x, order="F").reshape(2, -1)
        y = np.array(y, order="C").reshape(2, -1)
        assert_almost_equal(np.interp(x, xp, fp, period=360), y)


@instantiate_parametrized_tests
class TestPercentile(TestCase):
    @skip(reason="NP_VER: fails on CI; no method=")
    def test_basic(self):
        x = np.arange(8) * 0.5
        assert_equal(np.percentile(x, 0), 0.0)
        assert_equal(np.percentile(x, 100), 3.5)
        assert_equal(np.percentile(x, 50), 1.75)
        x[1] = np.nan
        assert_equal(np.percentile(x, 0), np.nan)
        assert_equal(np.percentile(x, 0, method="nearest"), np.nan)

    @skip(reason="support Fraction objects?")
    def test_fraction(self):
        x = [Fraction(i, 2) for i in range(8)]

        p = np.percentile(x, Fraction(0))
        assert_equal(p, Fraction(0))
        assert_equal(type(p), Fraction)

        p = np.percentile(x, Fraction(100))
        assert_equal(p, Fraction(7, 2))
        assert_equal(type(p), Fraction)

        p = np.percentile(x, Fraction(50))
        assert_equal(p, Fraction(7, 4))
        assert_equal(type(p), Fraction)

        p = np.percentile(x, [Fraction(50)])
        assert_equal(p, np.array([Fraction(7, 4)]))
        assert_equal(type(p), np.ndarray)

    def test_api(self):
        d = np.ones(5)
        np.percentile(d, 5, None, None, False)
        np.percentile(d, 5, None, None, False, "linear")
        o = np.ones((1,))
        np.percentile(d, 5, None, o, False, "linear")

    @xfail  # (reason="TODO: implement")
    def test_complex(self):
        arr_c = np.array([0.5 + 3.0j, 2.1 + 0.5j, 1.6 + 2.3j], dtype="D")
        assert_raises(TypeError, np.percentile, arr_c, 0.5)
        arr_c = np.array([0.5 + 3.0j, 2.1 + 0.5j, 1.6 + 2.3j], dtype="F")
        assert_raises(TypeError, np.percentile, arr_c, 0.5)

    def test_2D(self):
        x = np.array([[1, 1, 1], [1, 1, 1], [4, 4, 3], [1, 1, 1], [1, 1, 1]])
        assert_array_equal(np.percentile(x, 50, axis=0), [1, 1, 1])

    @skip(reason="NP_VER: fails on CI; no method=")
    @xpassIfTorchDynamo_np  # (reason="TODO: implement")
    @parametrize("dtype", np.typecodes["Float"])
    def test_linear_nan_1D(self, dtype):
        # METHOD 1 of H&F
        arr = np.asarray([

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 42 class(es): TestRot90, TestFlip, TestAny, TestAll, TestCopy, TestAverage, TestSelect, TestInsert, TestAmax, TestAmin, TestPtp, TestCumsum, TestProd, TestCumprod, TestDiff, TestDelete, TestGradient, TestAngle, TestTrimZeros, TestExtins

### Functions
This file defines 289 function(s): get_mat, _make_complex, test_basic, test_axes, test_rotation_axes, test_axes, test_basic_lr, test_basic_ud, test_3d_swap_axis0, test_3d_swap_axis1, test_3d_swap_axis2, test_4d, test_default_axis, test_multiple_axes, test_basic, test_nd, test_basic, test_nd, test_basic, test_order, test_basic, test_basic_keepdims, test_weights, test_returned, test_upcasting, test_average_class_without_dtype, _select, test_basic, test_broadcasting, test_return_dtype


## Key Components

The file contains 13232 words across 3911 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 138973 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
