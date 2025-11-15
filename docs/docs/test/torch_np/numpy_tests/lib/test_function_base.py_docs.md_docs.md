# Documentation: `docs/test/torch_np/numpy_tests/lib/test_function_base.py_docs.md`

## File Metadata

- **Path**: `docs/test/torch_np/numpy_tests/lib/test_function_base.py_docs.md`
- **Size**: 53,712 bytes (52.45 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/torch_np/numpy_tests/lib/test_function_base.py`

## File Metadata

- **Path**: `test/torch_np/numpy_tests/lib/test_function_base.py`
- **Size**: 138,973 bytes (135.72 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
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

 
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/torch_np/numpy_tests/lib`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/torch_np/numpy_tests/lib`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/torch_np/numpy_tests/lib/test_function_base.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/torch_np/numpy_tests/lib`):

- [`test_histograms.py_docs.md_docs.md`](./test_histograms.py_docs.md_docs.md)
- [`test_histograms.py_kw.md_docs.md`](./test_histograms.py_kw.md_docs.md)
- [`test_function_base.py_kw.md_docs.md`](./test_function_base.py_kw.md_docs.md)
- [`test_twodim_base.py_docs.md_docs.md`](./test_twodim_base.py_docs.md_docs.md)
- [`test_type_check.py_kw.md_docs.md`](./test_type_check.py_kw.md_docs.md)
- [`test_shape_base_.py_docs.md_docs.md`](./test_shape_base_.py_docs.md_docs.md)
- [`test_shape_base_.py_kw.md_docs.md`](./test_shape_base_.py_kw.md_docs.md)
- [`test_arraysetops.py_kw.md_docs.md`](./test_arraysetops.py_kw.md_docs.md)
- [`test_arraypad.py_kw.md_docs.md`](./test_arraypad.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_function_base.py_docs.md_docs.md`
- **Keyword Index**: `test_function_base.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
