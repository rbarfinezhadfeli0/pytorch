# Documentation: `docs/test/torch_np/numpy_tests/linalg/test_linalg.py_docs.md`

## File Metadata

- **Path**: `docs/test/torch_np/numpy_tests/linalg/test_linalg.py_docs.md`
- **Size**: 54,472 bytes (53.20 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/torch_np/numpy_tests/linalg/test_linalg.py`

## File Metadata

- **Path**: `test/torch_np/numpy_tests/linalg/test_linalg.py`
- **Size**: 80,006 bytes (78.13 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]
"""Test functions for linalg module"""

import functools
import itertools
import os
import subprocess
import sys
import textwrap
import traceback
from unittest import expectedFailure as xfail, skipIf as skipif, SkipTest

import numpy
import pytest
from numpy.linalg.linalg import _multi_dot_matrix_chain_order
from pytest import raises as assert_raises

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    slowTest as slow,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xpassIfTorchDynamo_np,
)


# If we are going to trace through these, we should use NumPy
# If testing on eager mode, we use torch._numpy
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy import (
        array,
        asarray,
        atleast_2d,
        cdouble,
        csingle,
        dot,
        double,
        identity,
        inf,
        linalg,
        matmul,
        single,
        swapaxes,
    )
    from numpy.linalg import LinAlgError, matrix_power, matrix_rank, multi_dot, norm
    from numpy.testing import (  # assert_raises_regex, HAS_LAPACK64, IS_WASM
        assert_,
        assert_allclose,
        assert_almost_equal,
        assert_array_equal,
        assert_equal,
        suppress_warnings,
    )

else:
    import torch._numpy as np
    from torch._numpy import (
        array,
        asarray,
        atleast_2d,
        cdouble,
        csingle,
        dot,
        double,
        identity,
        inf,
        linalg,
        matmul,
        single,
        swapaxes,
    )
    from torch._numpy.linalg import (
        LinAlgError,
        matrix_power,
        matrix_rank,
        multi_dot,
        norm,
    )
    from torch._numpy.testing import (
        assert_,
        assert_allclose,
        assert_almost_equal,
        assert_array_equal,
        assert_equal,
        suppress_warnings,
    )


skip = functools.partial(skipif, True)

IS_WASM = False
HAS_LAPACK64 = False


def consistent_subclass(out, in_):
    # For ndarray subclass input, our output should have the same subclass
    # (non-ndarray input gets converted to ndarray).
    return type(out) is (type(in_) if isinstance(in_, np.ndarray) else np.ndarray)


old_assert_almost_equal = assert_almost_equal


def assert_almost_equal(a, b, single_decimal=6, double_decimal=12, **kw):
    if asarray(a).dtype.type in (single, csingle):
        decimal = single_decimal
    else:
        decimal = double_decimal
    old_assert_almost_equal(a, b, decimal=decimal, **kw)


def get_real_dtype(dtype):
    return {single: single, double: double, csingle: single, cdouble: double}[dtype]


def get_complex_dtype(dtype):
    return {single: csingle, double: cdouble, csingle: csingle, cdouble: cdouble}[dtype]


def get_rtol(dtype):
    # Choose a safe rtol
    if dtype in (single, csingle):
        return 1e-5
    else:
        return 1e-11


# used to categorize tests
all_tags = {
    "square",
    "nonsquare",
    "hermitian",  # mutually exclusive
    "generalized",
    "size-0",
    "strided",  # optional additions
}


class LinalgCase:
    def __init__(self, name, a, b, tags=None):
        """
        A bundle of arguments to be passed to a test case, with an identifying
        name, the operands a and b, and a set of tags to filter the tests
        """
        if tags is None:
            tags = set()
        assert_(isinstance(name, str))
        self.name = name
        self.a = a
        self.b = b
        self.tags = frozenset(tags)  # prevent shared tags

    def check(self, do):
        """
        Run the function `do` on this test case, expanding arguments
        """
        do(self.a, self.b, tags=self.tags)

    def __repr__(self):
        return f"<LinalgCase: {self.name}>"


def apply_tag(tag, cases):
    """
    Add the given tag (a string) to each of the cases (a list of LinalgCase
    objects)
    """
    assert tag in all_tags, "Invalid tag"
    for case in cases:
        case.tags = case.tags | {tag}
    return cases


#
# Base test cases
#

np.random.seed(1234)

CASES = []

# square test cases
CASES += apply_tag(
    "square",
    [
        LinalgCase(
            "single",
            array([[1.0, 2.0], [3.0, 4.0]], dtype=single),
            array([2.0, 1.0], dtype=single),
        ),
        LinalgCase(
            "double",
            array([[1.0, 2.0], [3.0, 4.0]], dtype=double),
            array([2.0, 1.0], dtype=double),
        ),
        LinalgCase(
            "double_2",
            array([[1.0, 2.0], [3.0, 4.0]], dtype=double),
            array([[2.0, 1.0, 4.0], [3.0, 4.0, 6.0]], dtype=double),
        ),
        LinalgCase(
            "csingle",
            array([[1.0 + 2j, 2 + 3j], [3 + 4j, 4 + 5j]], dtype=csingle),
            array([2.0 + 1j, 1.0 + 2j], dtype=csingle),
        ),
        LinalgCase(
            "cdouble",
            array([[1.0 + 2j, 2 + 3j], [3 + 4j, 4 + 5j]], dtype=cdouble),
            array([2.0 + 1j, 1.0 + 2j], dtype=cdouble),
        ),
        LinalgCase(
            "cdouble_2",
            array([[1.0 + 2j, 2 + 3j], [3 + 4j, 4 + 5j]], dtype=cdouble),
            array(
                [[2.0 + 1j, 1.0 + 2j, 1 + 3j], [1 - 2j, 1 - 3j, 1 - 6j]], dtype=cdouble
            ),
        ),
        LinalgCase(
            "0x0",
            np.empty((0, 0), dtype=double),
            np.empty((0,), dtype=double),
            tags={"size-0"},
        ),
        LinalgCase("8x8", np.random.rand(8, 8), np.random.rand(8)),
        LinalgCase("1x1", np.random.rand(1, 1), np.random.rand(1)),
        LinalgCase("nonarray", [[1, 2], [3, 4]], [2, 1]),
    ],
)

# non-square test-cases
CASES += apply_tag(
    "nonsquare",
    [
        LinalgCase(
            "single_nsq_1",
            array([[1.0, 2.0, 3.0], [3.0, 4.0, 6.0]], dtype=single),
            array([2.0, 1.0], dtype=single),
        ),
        LinalgCase(
            "single_nsq_2",
            array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=single),
            array([2.0, 1.0, 3.0], dtype=single),
        ),
        LinalgCase(
            "double_nsq_1",
            array([[1.0, 2.0, 3.0], [3.0, 4.0, 6.0]], dtype=double),
            array([2.0, 1.0], dtype=double),
        ),
        LinalgCase(
            "double_nsq_2",
            array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=double),
            array([2.0, 1.0, 3.0], dtype=double),
        ),
        LinalgCase(
            "csingle_nsq_1",
            array(
                [[1.0 + 1j, 2.0 + 2j, 3.0 - 3j], [3.0 - 5j, 4.0 + 9j, 6.0 + 2j]],
                dtype=csingle,
            ),
            array([2.0 + 1j, 1.0 + 2j], dtype=csingle),
        ),
        LinalgCase(
            "csingle_nsq_2",
            array(
                [[1.0 + 1j, 2.0 + 2j], [3.0 - 3j, 4.0 - 9j], [5.0 - 4j, 6.0 + 8j]],
                dtype=csingle,
            ),
            array([2.0 + 1j, 1.0 + 2j, 3.0 - 3j], dtype=csingle),
        ),
        LinalgCase(
            "cdouble_nsq_1",
            array(
                [[1.0 + 1j, 2.0 + 2j, 3.0 - 3j], [3.0 - 5j, 4.0 + 9j, 6.0 + 2j]],
                dtype=cdouble,
            ),
            array([2.0 + 1j, 1.0 + 2j], dtype=cdouble),
        ),
        LinalgCase(
            "cdouble_nsq_2",
            array(
                [[1.0 + 1j, 2.0 + 2j], [3.0 - 3j, 4.0 - 9j], [5.0 - 4j, 6.0 + 8j]],
                dtype=cdouble,
            ),
            array([2.0 + 1j, 1.0 + 2j, 3.0 - 3j], dtype=cdouble),
        ),
        LinalgCase(
            "cdouble_nsq_1_2",
            array(
                [[1.0 + 1j, 2.0 + 2j, 3.0 - 3j], [3.0 - 5j, 4.0 + 9j, 6.0 + 2j]],
                dtype=cdouble,
            ),
            array([[2.0 + 1j, 1.0 + 2j], [1 - 1j, 2 - 2j]], dtype=cdouble),
        ),
        LinalgCase(
            "cdouble_nsq_2_2",
            array(
                [[1.0 + 1j, 2.0 + 2j], [3.0 - 3j, 4.0 - 9j], [5.0 - 4j, 6.0 + 8j]],
                dtype=cdouble,
            ),
            array(
                [[2.0 + 1j, 1.0 + 2j], [1 - 1j, 2 - 2j], [1 - 1j, 2 - 2j]],
                dtype=cdouble,
            ),
        ),
        LinalgCase("8x11", np.random.rand(8, 11), np.random.rand(8)),
        LinalgCase("1x5", np.random.rand(1, 5), np.random.rand(1)),
        LinalgCase("5x1", np.random.rand(5, 1), np.random.rand(5)),
        LinalgCase("0x4", np.random.rand(0, 4), np.random.rand(0), tags={"size-0"}),
        LinalgCase("4x0", np.random.rand(4, 0), np.random.rand(4), tags={"size-0"}),
    ],
)

# hermitian test-cases
CASES += apply_tag(
    "hermitian",
    [
        LinalgCase("hsingle", array([[1.0, 2.0], [2.0, 1.0]], dtype=single), None),
        LinalgCase("hdouble", array([[1.0, 2.0], [2.0, 1.0]], dtype=double), None),
        LinalgCase(
            "hcsingle", array([[1.0, 2 + 3j], [2 - 3j, 1]], dtype=csingle), None
        ),
        LinalgCase(
            "hcdouble", array([[1.0, 2 + 3j], [2 - 3j, 1]], dtype=cdouble), None
        ),
        LinalgCase("hempty", np.empty((0, 0), dtype=double), None, tags={"size-0"}),
        LinalgCase("hnonarray", [[1, 2], [2, 1]], None),
        LinalgCase("matrix_b_only", array([[1.0, 2.0], [2.0, 1.0]]), None),
        LinalgCase("hmatrix_1x1", np.random.rand(1, 1), None),
    ],
)


#
# Gufunc test cases
#
def _make_generalized_cases():
    new_cases = []

    for case in CASES:
        if not isinstance(case.a, np.ndarray):
            continue

        a = np.stack([case.a, 2 * case.a, 3 * case.a])
        if case.b is None:
            b = None
        else:
            b = np.stack([case.b, 7 * case.b, 6 * case.b])
        new_case = LinalgCase(
            case.name + "_tile3", a, b, tags=case.tags | {"generalized"}
        )
        new_cases.append(new_case)

        a = np.array([case.a] * 2 * 3).reshape((3, 2) + case.a.shape)
        if case.b is None:
            b = None
        else:
            b = np.array([case.b] * 2 * 3).reshape((3, 2) + case.b.shape)
        new_case = LinalgCase(
            case.name + "_tile213", a, b, tags=case.tags | {"generalized"}
        )
        new_cases.append(new_case)

    return new_cases


CASES += _make_generalized_cases()


#
# Test different routines against the above cases
#
class LinalgTestCase:
    TEST_CASES = CASES

    def check_cases(self, require=None, exclude=None):
        """
        Run func on each of the cases with all of the tags in require, and none
        of the tags in exclude
        """
        if require is None:
            require = set()
        if exclude is None:
            exclude = set()
        for case in self.TEST_CASES:
            # filter by require and exclude
            if case.tags & require != require:
                continue
            if case.tags & exclude:
                continue

            try:
                case.check(self.do)
            except Exception as e:
                msg = f"In test case: {case!r}\n\n"
                msg += traceback.format_exc()
                raise AssertionError(msg) from e


class LinalgSquareTestCase(LinalgTestCase):
    def test_sq_cases(self):
        self.check_cases(require={"square"}, exclude={"generalized", "size-0"})

    def test_empty_sq_cases(self):
        self.check_cases(require={"square", "size-0"}, exclude={"generalized"})


class LinalgNonsquareTestCase(LinalgTestCase):
    def test_nonsq_cases(self):
        self.check_cases(require={"nonsquare"}, exclude={"generalized", "size-0"})

    def test_empty_nonsq_cases(self):
        self.check_cases(require={"nonsquare", "size-0"}, exclude={"generalized"})


class HermitianTestCase(LinalgTestCase):
    def test_herm_cases(self):
        self.check_cases(require={"hermitian"}, exclude={"generalized", "size-0"})

    def test_empty_herm_cases(self):
        self.check_cases(require={"hermitian", "size-0"}, exclude={"generalized"})


class LinalgGeneralizedSquareTestCase(LinalgTestCase):
    @slow
    def test_generalized_sq_cases(self):
        self.check_cases(require={"generalized", "square"}, exclude={"size-0"})

    @slow
    def test_generalized_empty_sq_cases(self):
        self.check_cases(require={"generalized", "square", "size-0"})


class LinalgGeneralizedNonsquareTestCase(LinalgTestCase):
    @slow
    def test_generalized_nonsq_cases(self):
        self.check_cases(require={"generalized", "nonsquare"}, exclude={"size-0"})

    @slow
    def test_generalized_empty_nonsq_cases(self):
        self.check_cases(require={"generalized", "nonsquare", "size-0"})


class HermitianGeneralizedTestCase(LinalgTestCase):
    @slow
    def test_generalized_herm_cases(self):
        self.check_cases(require={"generalized", "hermitian"}, exclude={"size-0"})

    @slow
    def test_generalized_empty_herm_cases(self):
        self.check_cases(
            require={"generalized", "hermitian", "size-0"}, exclude={"none"}
        )


def dot_generalized(a, b):
    a = asarray(a)
    if a.ndim >= 3:
        if a.ndim == b.ndim:
            # matrix x matrix
            new_shape = a.shape[:-1] + b.shape[-1:]
        elif a.ndim == b.ndim + 1:
            # matrix x vector
            new_shape = a.shape[:-1]
        else:
            raise ValueError("Not implemented...")
        r = np.empty(new_shape, dtype=np.common_type(a, b))
        for c in itertools.product(*map(range, a.shape[:-2])):
            r[c] = dot(a[c], b[c])
        return r
    else:
        return dot(a, b)


def identity_like_generalized(a):
    a = asarray(a)
    if a.ndim >= 3:
        r = np.empty(a.shape, dtype=a.dtype)
        r[...] = identity(a.shape[-2])
        return r
    else:
        return identity(a.shape[0])


class SolveCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):
    # kept apart from TestSolve for use for testing with matrices.
    def do(self, a, b, tags):
        x = linalg.solve(a, b)
        assert_almost_equal(b, dot_generalized(a, x), single_decimal=5)
        assert_(consistent_subclass(x, b))


@instantiate_parametrized_tests
class TestSolve(SolveCases, TestCase):
    @parametrize("dtype", [single, double, csingle, cdouble])
    def test_types(self, dtype):
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        assert_equal(linalg.solve(x, x).dtype, dtype)

    @skip(reason="subclass")
    def test_0_size(self):
        class ArraySubclass(np.ndarray):
            pass

        # Test system of 0x0 matrices
        a = np.arange(8).reshape(2, 2, 2)
        b = np.arange(6).reshape(1, 2, 3).view(ArraySubclass)

        expected = linalg.solve(a, b)[:, 0:0, :]
        result = linalg.solve(a[:, 0:0, 0:0], b[:, 0:0, :])
        assert_array_equal(result, expected)
        assert_(isinstance(result, ArraySubclass))

        # Test errors for non-square and only b's dimension being 0
        assert_raises(linalg.LinAlgError, linalg.solve, a[:, 0:0, 0:1], b)
        assert_raises(ValueError, linalg.solve, a, b[:, 0:0, :])

        # Test broadcasting error
        b = np.arange(6).reshape(1, 3, 2)  # broadcasting error
        assert_raises(ValueError, linalg.solve, a, b)
        assert_raises(ValueError, linalg.solve, a[0:0], b[0:0])

        # Test zero "single equations" with 0x0 matrices.
        b = np.arange(2).reshape(1, 2).view(ArraySubclass)
        expected = linalg.solve(a, b)[:, 0:0]
        result = linalg.solve(a[:, 0:0, 0:0], b[:, 0:0])
        assert_array_equal(result, expected)
        assert_(isinstance(result, ArraySubclass))

        b = np.arange(3).reshape(1, 3)
        assert_raises(ValueError, linalg.solve, a, b)
        assert_raises(ValueError, linalg.solve, a[0:0], b[0:0])
        assert_raises(ValueError, linalg.solve, a[:, 0:0, 0:0], b)

    @skip(reason="subclass")
    def test_0_size_k(self):
        # test zero multiple equation (K=0) case.
        class ArraySubclass(np.ndarray):
            pass

        a = np.arange(4).reshape(1, 2, 2)
        b = np.arange(6).reshape(3, 2, 1).view(ArraySubclass)

        expected = linalg.solve(a, b)[:, :, 0:0]
        result = linalg.solve(a, b[:, :, 0:0])
        assert_array_equal(result, expected)
        assert_(isinstance(result, ArraySubclass))

        # test both zero.
        expected = linalg.solve(a, b)[:, 0:0, 0:0]
        result = linalg.solve(a[:, 0:0, 0:0], b[:, 0:0, 0:0])
        assert_array_equal(result, expected)
        assert_(isinstance(result, ArraySubclass))


class InvCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):
    def do(self, a, b, tags):
        a_inv = linalg.inv(a)
        assert_almost_equal(dot_generalized(a, a_inv), identity_like_generalized(a))
        assert_(consistent_subclass(a_inv, a))


@instantiate_parametrized_tests
class TestInv(InvCases, TestCase):
    @parametrize("dtype", [single, double, csingle, cdouble])
    def test_types(self, dtype):
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        assert_equal(linalg.inv(x).dtype, dtype)

    @skip(reason="subclass")
    def test_0_size(self):
        # Check that all kinds of 0-sized arrays work
        class ArraySubclass(np.ndarray):
            pass

        a = np.zeros((0, 1, 1), dtype=np.int_).view(ArraySubclass)
        res = linalg.inv(a)
        assert_(res.dtype.type is np.float64)
        assert_equal(a.shape, res.shape)
        assert_(isinstance(res, ArraySubclass))

        a = np.zeros((0, 0), dtype=np.complex64).view(ArraySubclass)
        res = linalg.inv(a)
        assert_(res.dtype.type is np.complex64)
        assert_equal(a.shape, res.shape)
        assert_(isinstance(res, ArraySubclass))


class EigvalsCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):
    def do(self, a, b, tags):
        ev = linalg.eigvals(a)
        evalues, evectors = linalg.eig(a)
        assert_almost_equal(ev, evalues)


@instantiate_parametrized_tests
class TestEigvals(EigvalsCases, TestCase):
    @parametrize("dtype", [single, double, csingle, cdouble])
    def test_types(self, dtype):
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        assert_equal(linalg.eigvals(x).dtype, dtype)
        x = np.array([[1, 0.5], [-1, 1]], dtype=dtype)
        assert_equal(linalg.eigvals(x).dtype, get_complex_dtype(dtype))

    @skip(reason="subclass")
    def test_0_size(self):
        # Check that all kinds of 0-sized arrays work
        class ArraySubclass(np.ndarray):
            pass

        a = np.zeros((0, 1, 1), dtype=np.int_).view(ArraySubclass)
        res = linalg.eigvals(a)
        assert_(res.dtype.type is np.float64)
        assert_equal((0, 1), res.shape)
        # This is just for documentation, it might make sense to change:
        assert_(isinstance(res, np.ndarray))

        a = np.zeros((0, 0), dtype=np.complex64).view(ArraySubclass)
        res = linalg.eigvals(a)
        assert_(res.dtype.type is np.complex64)
        assert_equal((0,), res.shape)
        # This is just for documentation, it might make sense to change:
        assert_(isinstance(res, np.ndarray))


class EigCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):
    def do(self, a, b, tags):
        evalues, evectors = linalg.eig(a)
        assert_allclose(
            dot_generalized(a, evectors),
            np.asarray(evectors) * np.asarray(evalues)[..., None, :],
            rtol=get_rtol(evalues.dtype),
        )
        assert_(consistent_subclass(evectors, a))


@instantiate_parametrized_tests
class TestEig(EigCases, TestCase):
    @parametrize("dtype", [single, double, csingle, cdouble])
    def test_types(self, dtype):
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        w, v = np.linalg.eig(x)
        assert_equal(w.dtype, dtype)
        assert_equal(v.dtype, dtype)

        x = np.array([[1, 0.5], [-1, 1]], dtype=dtype)
        w, v = np.linalg.eig(x)
        assert_equal(w.dtype, get_complex_dtype(dtype))
        assert_equal(v.dtype, get_complex_dtype(dtype))

    @skip(reason="subclass")
    def test_0_size(self):
        # Check that all kinds of 0-sized arrays work
        class ArraySubclass(np.ndarray):
            pass

        a = np.zeros((0, 1, 1), dtype=np.int_).view(ArraySubclass)
        res, res_v = linalg.eig(a)
        assert_(res_v.dtype.type is np.float64)
        assert_(res.dtype.type is np.float64)
        assert_equal(a.shape, res_v.shape)
        assert_equal((0, 1), res.shape)
        # This is just for documentation, it might make sense to change:
        assert_(isinstance(a, np.ndarray))

        a = np.zeros((0, 0), dtype=np.complex64).view(ArraySubclass)
        res, res_v = linalg.eig(a)
        assert_(res_v.dtype.type is np.complex64)
        assert_(res.dtype.type is np.complex64)
        assert_equal(a.shape, res_v.shape)
        assert_equal((0,), res.shape)
        # This is just for documentation, it might make sense to change:
        assert_(isinstance(a, np.ndarray))


@instantiate_parametrized_tests
class SVDBaseTests:
    hermitian = False

    @parametrize("dtype", [single, double, csingle, cdouble])
    def test_types(self, dtype):
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        u, s, vh = linalg.svd(x)
        assert_equal(u.dtype, dtype)
        assert_equal(s.dtype, get_real_dtype(dtype))
        assert_equal(vh.dtype, dtype)
        s = linalg.svd(x, compute_uv=False, hermitian=self.hermitian)
        assert_equal(s.dtype, get_real_dtype(dtype))


class SVDCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):
    def do(self, a, b, tags):
        u, s, vt = linalg.svd(a, False)
        assert_allclose(
            a,
            dot_generalized(
                np.asarray(u) * np.asarray(s)[..., None, :], np.asarray(vt)
            ),
            rtol=get_rtol(u.dtype),
        )
        assert_(consistent_subclass(u, a))
        assert_(consistent_subclass(vt, a))


class TestSVD(SVDCases, SVDBaseTests, TestCase):
    def test_empty_identity(self):
        """Empty input should put an identity matrix in u or vh"""
        x = np.empty((4, 0))
        u, s, vh = linalg.svd(x, compute_uv=True, hermitian=self.hermitian)
        assert_equal(u.shape, (4, 4))
        assert_equal(vh.shape, (0, 0))
        assert_equal(u, np.eye(4))

        x = np.empty((0, 4))
        u, s, vh = linalg.svd(x, compute_uv=True, hermitian=self.hermitian)
        assert_equal(u.shape, (0, 0))
        assert_equal(vh.shape, (4, 4))
        assert_equal(vh, np.eye(4))


class SVDHermitianCases(HermitianTestCase, HermitianGeneralizedTestCase):
    def do(self, a, b, tags):
        u, s, vt = linalg.svd(a, False, hermitian=True)
        assert_allclose(
            a,
            dot_generalized(
                np.asarray(u) * np.asarray(s)[..., None, :], np.asarray(vt)
            ),
            rtol=get_rtol(u.dtype),
        )

        def hermitian(mat):
            axes = list(range(mat.ndim))
            axes[-1], axes[-2] = axes[-2], axes[-1]
            return np.conj(np.transpose(mat, axes=axes))

        assert_almost_equal(
            np.matmul(u, hermitian(u)), np.broadcast_to(np.eye(u.shape[-1]), u.shape)
        )
        assert_almost_equal(
            np.matmul(vt, hermitian(vt)),
            np.broadcast_to(np.eye(vt.shape[-1]), vt.shape),
        )
        assert_equal(np.sort(s), np.flip(s, -1))
        assert_(consistent_subclass(u, a))
        assert_(consistent_subclass(vt, a))


class TestSVDHermitian(SVDHermitianCases, SVDBaseTests, TestCase):
    hermitian = True


class CondCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):
    # cond(x, p) for p in (None, 2, -2)

    def do(self, a, b, tags):
        c = asarray(a)  # a might be a matrix
        if "size-0" in tags:
            assert_raises(LinAlgError, linalg.cond, c)
            return

        # +-2 norms
        s = linalg.svd(c, compute_uv=False)
        assert_almost_equal(
            linalg.cond(a), s[..., 0] / s[..., -1], single_decimal=5, double_decimal=11
        )
        assert_almost_equal(
            linalg.cond(a, 2),
            s[..., 0] / s[..., -1],
            single_decimal=5,
            double_decimal=11,
        )
        assert_almost_equal(
            linalg.cond(a, -2),
            s[..., -1] / s[..., 0],
            single_decimal=5,
            double_decimal=11,
        )

        # Other norms
        cinv = np.linalg.inv(c)
        assert_almost_equal(
            linalg.cond(a, 1),
            abs(c).sum(-2).max(-1) * abs(cinv).sum(-2).max(-1),
            single_decimal=5,
            double_decimal=11,
        )
        assert_almost_equal(
            linalg.cond(a, -1),
            abs(c).sum(-2).min(-1) * abs(cinv).sum(-2).min(-1),
            single_decimal=5,
            double_decimal=11,
        )
        assert_almost_equal(
            linalg.cond(a, np.inf),
            abs(c).sum(-1).max(-1) * abs(cinv).sum(-1).max(-1),
            single_decimal=5,
            double_decimal=11,
        )
        assert_almost_equal(
            linalg.cond(a, -np.inf),
            abs(c).sum(-1).min(-1) * abs(cinv).sum(-1).min(-1),
            single_decimal=5,
            double_decimal=11,
        )
        assert_almost_equal(
            linalg.cond(a, "fro"),
            np.sqrt((abs(c) ** 2).sum(-1).sum(-1) * (abs(cinv) ** 2).sum(-1).sum(-1)),
            single_decimal=5,
            double_decimal=11,
        )


class TestCond(CondCases, TestCase):
    def test_basic_nonsvd(self):
        # Smoketest the non-svd norms
        A = array([[1.0, 0, 1], [0, -2.0, 0], [0, 0, 3.0]])
        assert_almost_equal(linalg.cond(A, inf), 4)
        assert_almost_equal(linalg.cond(A, -inf), 2 / 3)
        assert_almost_equal(linalg.cond(A, 1), 4)
        assert_almost_equal(linalg.cond(A, -1), 0.5)
        assert_almost_equal(linalg.cond(A, "fro"), np.sqrt(265 / 12))

    def test_singular(self):
        # Singular matrices have infinite condition number for
        # positive norms, and negative norms shouldn't raise
        # exceptions
        As = [np.zeros((2, 2)), np.ones((2, 2))]
        p_pos = [None, 1, 2, "fro"]
        p_neg = [-1, -2]
        for A, p in itertools.product(As, p_pos):
            # Inversion may not hit exact infinity, so just check the
            # number is large
            assert_(linalg.cond(A, p) > 1e15)
        for A, p in itertools.product(As, p_neg):
            linalg.cond(A, p)

    @skip(reason="NP_VER: fails on CI")  # (
    #    True, run=False, reason="Platform/LAPACK-dependent failure, see gh-18914"
    # )
    def test_nan(self):
        # nans should be passed through, not converted to infs
        ps = [None, 1, -1, 2, -2, "fro"]
        p_pos = [None, 1, 2, "fro"]

        A = np.ones((2, 2))
        A[0, 1] = np.nan
        for p in ps:
            c = linalg.cond(A, p)
            assert_(isinstance(c, np.float64))
            assert_(np.isnan(c))

        A = np.ones((3, 2, 2))
        A[1, 0, 1] = np.nan
        for p in ps:
            c = linalg.cond(A, p)
            assert_(np.isnan(c[1]))
            if p in p_pos:
                assert_(c[0] > 1e15)
                assert_(c[2] > 1e15)
            else:
                assert_(not np.isnan(c[0]))
                assert_(not np.isnan(c[2]))

    def test_stacked_singular(self):
        # Check behavior when only some of the stacked matrices are
        # singular
        np.random.seed(1234)
        A = np.random.rand(2, 2, 2, 2)
        A[0, 0] = 0
        A[1, 1] = 0

        for p in (None, 1, 2, "fro", -1, -2):
            c = linalg.cond(A, p)
            assert_equal(c[0, 0], np.inf)
            assert_equal(c[1, 1], np.inf)
            assert_(np.isfinite(c[0, 1]))
            assert_(np.isfinite(c[1, 0]))


class PinvCases(
    LinalgSquareTestCase,
    LinalgNonsquareTestCase,
    LinalgGeneralizedSquareTestCase,
    LinalgGeneralizedNonsquareTestCase,
):
    def do(self, a, b, tags):
        a_ginv = linalg.pinv(a)
        # `a @ a_ginv == I` does not hold if a is singular
        dot = dot_generalized
        assert_almost_equal(
            dot(dot(a, a_ginv), a), a, single_decimal=5, double_decimal=11
        )
        assert_(consistent_subclass(a_ginv, a))


class TestPinv(PinvCases, TestCase):
    pass


class PinvHermitianCases(HermitianTestCase, HermitianGeneralizedTestCase):
    def do(self, a, b, tags):
        a_ginv = linalg.pinv(a, hermitian=True)
        # `a @ a_ginv == I` does not hold if a is singular
        dot = dot_generalized
        assert_almost_equal(
            dot(dot(a, a_ginv), a), a, single_decimal=5, double_decimal=11
        )
        assert_(consistent_subclass(a_ginv, a))


class TestPinvHermitian(PinvHermitianCases, TestCase):
    pass


class DetCases(LinalgSquareTestCase, LinalgGeneralizedSquareTestCase):
    def do(self, a, b, tags):
        d = linalg.det(a)
        (s, ld) = linalg.slogdet(a)
        if asarray(a).dtype.type in (single, double):
            ad = asarray(a).astype(double)
        else:
            ad = asarray(a).astype(cdouble)
        ev = linalg.eigvals(ad)
        assert_almost_equal(d, np.prod(ev, axis=-1))
        assert_almost_equal(s * np.exp(ld), np.prod(ev, axis=-1), single_decimal=5)

        s = np.atleast_1d(s)
        ld = np.atleast_1d(ld)
        m = s != 0
        assert_almost_equal(np.abs(s[m]), 1)
        assert_equal(ld[~m], -inf)


@instantiate_parametrized_tests
class TestDet(DetCases, TestCase):
    def test_zero(self):
        # NB: comment out tests of type(det) is double : we return zero-dim arrays
        assert_equal(linalg.det([[0.0]]), 0.0)
        #    assert_equal(type(linalg.det([[0.0]])), double)
        assert_equal(linalg.det([[0.0j]]), 0.0)
        #    assert_equal(type(linalg.det([[0.0j]])), cdouble)

        assert_equal(linalg.slogdet([[0.0]]), (0.0, -inf))
        #    assert_equal(type(linalg.slogdet([[0.0]])[0]), double)
        #    assert_equal(type(linalg.slogdet([[0.0]])[1]), double)
        assert_equal(linalg.slogdet([[0.0j]]), (0.0j, -inf))

    #    assert_equal(type(linalg.slogdet([[0.0j]])[0]), cdouble)
    #    assert_equal(type(linalg.slogdet([[0.0j]])[1]), double)

    @parametrize("dtype", [single, double, csingle, cdouble])
    def test_types(self, dtype):
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        assert_equal(np.linalg.det(x).dtype, dtype)
        ph, s = np.linalg.slogdet(x)
        assert_equal(s.dtype, get_real_dtype(dtype))
        assert_equal(ph.dtype, dtype)

    def test_0_size(self):
        a = np.zeros((0, 0), dtype=np.complex64)
        res = linalg.det(a)
        assert_equal(res, 1.0)
        assert_(res.dtype.type is np.complex64)
        res = linalg.slogdet(a)
        assert_equal(res, (1, 0))
        assert_(res[0].dtype.type is np.complex64)
        assert_(res[1].dtype.type is np.float32)

        a = np.zeros((0, 0), dtype=np.float64)
        res = linalg.det(a)
        assert_equal(res, 1.0)
        assert_(res.dtype.type is np.float64)
        res = linalg.slogdet(a)
        assert_equal(res, (1, 0))
        assert_(res[0].dtype.type is np.float64)
        assert_(res[1].dtype.type is np.float64)


class LstsqCases(LinalgSquareTestCase, LinalgNonsquareTestCase):
    def do(self, a, b, tags):
        arr = np.asarray(a)
        m, n = arr.shape
        u, s, vt = linalg.svd(a, False)
        x, residuals, rank, sv = linalg.lstsq(a, b, rcond=-1)
        if m == 0:
            assert_((x == 0).all())
        if m <= n:
            assert_almost_equal(b, dot(a, x), single_decimal=5)
            assert_equal(rank, m)
        else:
            assert_equal(rank, n)
        #     assert_almost_equal(sv, sv.__array_wrap__(s))
        if rank == n and m > n:
            expect_resids = (np.asarray(abs(np.dot(a, x) - b)) ** 2).sum(axis=0)
            expect_resids = np.asarray(expect_resids)
            if np.asarray(b).ndim == 1:
                expect_resids = expect_resids.reshape(
                    1,
                )
                assert_equal(residuals.shape, expect_resids.shape)
        else:
            expect_resids = np.array([])  # .view(type(x))
        assert_almost_equal(residuals, expect_resids, single_decimal=5)
        assert_(np.issubdtype(residuals.dtype, np.floating))
        assert_(consistent_subclass(x, b))
        assert_(consistent_subclass(residuals, b))


@instantiate_parametrized_tests
class TestLstsq(LstsqCases, TestCase):
    @xpassIfTorchDynamo_np  # (reason="Lstsq: we use the future default =None")
    def test_future_rcond(self):
        a = np.array(
            [
                [0.0, 1.0, 0.0, 1.0, 2.0, 0.0],
                [0.0, 2.0, 0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0, 4.0],
                [0.0, 0.0, 0.0, 2.0, 3.0, 0.0],
            ]
        ).T

        b = np.array([1, 0, 0, 0, 0, 0])
        with suppress_warnings() as sup:
            w = sup.record(FutureWarning, "`rcond` parameter will change")
            x, residuals, rank, s = linalg.lstsq(a, b)
            assert_(rank == 4)
            x, residuals, rank, s = linalg.lstsq(a, b, rcond=-1)
            assert_(rank == 4)
            x, residuals, rank, s = linalg.lstsq(a, b, rcond=None)
            assert_(rank == 3)
            # Warning should be raised exactly once (first command)
            assert_(len(w) == 1)

    @parametrize(
        "m, n, n_rhs",
        [
            (4, 2, 2),
            (0, 4, 1),
            (0, 4, 2),
            (4, 0, 1),
            (4, 0, 2),
            #    (4, 2, 0),    # Intel MKL ERROR: Parameter 4 was incorrect on entry to DLALSD.
            (0, 0, 0),
        ],
    )
    def test_empty_a_b(self, m, n, n_rhs):
        a = np.arange(m * n).reshape(m, n)
        b = np.ones((m, n_rhs))
        x, residuals, rank, s = linalg.lstsq(a, b, rcond=None)
        if m == 0:
            assert_((x == 0).all())
        assert_equal(x.shape, (n, n_rhs))
        assert_equal(residuals.shape, ((n_rhs,) if m > n else (0,)))
        if m > n and n_rhs > 0:
            # residuals are exactly the squared norms of b's columns
            r = b - np.dot(a, x)
            assert_almost_equal(residuals, (r * r).sum(axis=-2))
        assert_equal(rank, min(m, n))
        assert_equal(s.shape, (min(m, n),))

    def test_incompatible_dims(self):
        # use modified version of docstring example
        x = np.array([0, 1, 2, 3])
        y = np.array([-1, 0.2, 0.9, 2.1, 3.3])
        A = np.vstack([x, np.ones(len(x))]).T
        #        with assert_raises_regex(LinAlgError, "Incompatible dimensions"):
        with assert_raises((RuntimeError, LinAlgError)):
            linalg.lstsq(A, y, rcond=None)


# @xfail  #(reason="no block()")
@skip  # FIXME: otherwise fails in setUp calling np.block
@instantiate_parametrized_tests
class TestMatrixPower(TestCase):
    def setUp(self):
        self.rshft_0 = np.eye(4)
        self.rshft_1 = self.rshft_0[[3, 0, 1, 2]]
        self.rshft_2 = self.rshft_0[[2, 3, 0, 1]]
        self.rshft_3 = self.rshft_0[[1, 2, 3, 0]]
        self.rshft_all = [self.rshft_0, self.rshft_1, self.rshft_2, self.rshft_3]
        self.noninv = array([[1, 0], [0, 0]])
        self.stacked = np.block([[[self.rshft_0]]] * 2)
        # FIXME the 'e' dtype might work in future
        self.dtnoinv = [object, np.dtype("e"), np.dtype("g"), np.dtype("G")]

    @parametrize("dt", [np.dtype(c) for c in "?bBhilefdFD"])
    def test_large_power(self, dt):
        rshft = self.rshft_1.astype(dt)
        assert_equal(matrix_power(rshft, 2**100 + 2**10 + 2**5 + 0), self.rshft_0)
        assert_equal(matrix_power(rshft, 2**100 + 2**10 + 2**5 + 1), self.rshft_1)
        assert_equal(matrix_power(rshft, 2**100 + 2**10 + 2**5 + 2), self.rshft_2)
        assert_equal(matrix_power(rshft, 2**100 + 2**10 + 2**5 + 3), self.rshft_3)

    @parametrize("dt", [np.dtype(c) for c in "?bBhilefdFD"])
    def test_power_is_zero(self, dt):
        def tz(M):
            mz = matrix_power(M, 0)
            assert_equal(mz, identity_like_generalized(M))
            assert_equal(mz.dtype, M.dtype)

        for mat in self.rshft_all:
            tz(mat.astype(dt))
            if dt is not object:
                tz(self.stacked.astype(dt))

    @parametrize("dt", [np.dtype(c) for c in "?bBhilefdFD"])
    def test_power_is_one(self, dt):
        def tz(mat):
            mz = matrix_power(mat, 1)
            assert_equal(mz, mat)
            assert_equal(mz.dtype, mat.dtype)

        for mat in self.rshft_all:
            tz(mat.astype(dt))
            if dt is not object:
                tz(self.stacked.astype(dt))

    @parametrize("dt", [np.dtype(c) for c in "?bBhilefdFD"])
    def test_power_is_two(self, dt):
        def tz(mat):
            mz = matrix_power(mat, 2)
            mmul = matmul if mat.dtype != object else dot
            assert_equal(mz, mmul(mat, mat))
            assert_equal(mz.dtype, mat.dtype)

        for mat in self.rshft_all:
            tz(mat.astype(dt))
            if dt is not object:
                tz(self.stacked.astype(dt))

    @parametrize("dt", [np.dtype(c) for c in "?bBhilefdFD"])
    def test_power_is_minus_one(self, dt):
        def tz(mat):
            invmat = matrix_power(mat, -1)
            mmul = matmul if mat.dtype != object else dot
            assert_almost_equal(mmul(invmat, mat), identity_like_generalized(mat))

        for mat in self.rshft_all:
            if dt not in self.dtnoinv:
                tz(mat.astype(dt))

    @parametrize("dt", [np.dtype(c) for c in "?bBhilefdFD"])
    def test_exceptions_bad_power(self, dt):
        mat = self.rshft_0.astype(dt)
        assert_raises(TypeError, matrix_power, mat, 1.5)
        assert_raises(TypeError, matrix_power, mat, [1])

    @parametrize("dt", [np.dtype(c) for c in "?bBhilefdFD"])
    def test_exceptions_non_square(self, dt):
        assert_raises(LinAlgError, matrix_power, np.array([1], dt), 1)
        assert_raises(LinAlgError, matrix_power, np.array([[1], [2]], dt), 1)
        assert_raises(LinAlgError, matrix_power, np.ones((4, 3, 2), dt), 1)

    @skipif(IS_WASM, reason="fp errors don't work in wasm")
    @parametrize("dt", [np.dtype(c) for c in "?bBhilefdFD"])
    def test_exceptions_not_invertible(self, dt):
        if dt in self.dtnoinv:
            return
        mat = self.noninv.astype(dt)
        assert_raises(LinAlgError, matrix_power, mat, -1)


class TestEigvalshCases(HermitianTestCase, HermitianGeneralizedTestCase):
    def do(self, a, b, tags):
        pytest.xfail(reason="sort complex")
        # note that eigenvalue arrays returned by eig must be sorted since
        # their order isn't guaranteed.
        ev = linalg.eigvalsh(a, "L")
        evalues, evectors = linalg.eig(a)
        evalues.sort(axis=-1)
        assert_allclose(ev, evalues, rtol=get_rtol(ev.dtype))

        ev2 = linalg.eigvalsh(a, "U")
        assert_allclose(ev2, evalues, rtol=get_rtol(ev.dtype))


@instantiate_parametrized_tests
class TestEigvalsh(TestCase):
    @parametrize("dtype", [single, double, csingle, cdouble])
    def test_types(self, dtype):
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        w = np.linalg.eigvalsh(x)
        assert_equal(w.dtype, get_real_dtype(dtype))

    def test_invalid(self):
        x = np.array([[1, 0.5], [0.5, 1]], dtype=np.float32)
        assert_raises((RuntimeError, ValueError), np.linalg.eigvalsh, x, UPLO="lrong")
        assert_raises((RuntimeError, ValueError), np.linalg.eigvalsh, x, "lower")
        assert_raises((RuntimeError, ValueError), np.linalg.eigvalsh, x, "upper")

    def test_UPLO(self):
        Klo = np.array([[0, 0], [1, 0]], dtype=np.double)
        Kup = np.array([[0, 1], [0, 0]], dtype=np.double)
        tgt = np.array([-1, 1], dtype=np.double)
        rtol = get_rtol(np.double)

        # Check default is 'L'
        w = np.linalg.eigvalsh(Klo)
        assert_allclose(w, tgt, rtol=rtol)
        # Check 'L'
        w = np.linalg.eigvalsh(Klo, UPLO="L")
        assert_allclose(w, tgt, rtol=rtol)
        # Check 'l'
        w = np.linalg.eigvalsh(Klo, UPLO="l")
        assert_allclose(w, tgt, rtol=rtol)
        # Check 'U'
        w = np.linalg.eigvalsh(Kup, UPLO="U")
        assert_allclose(w, tgt, rtol=rtol)
        # Check 'u'
        w = np.linalg.eigvalsh(Kup, UPLO="u")
        assert_allclose(w, tgt, rtol=rtol)

    def test_0_size(self):
        # Check that all kinds of 0-sized arrays work
        #     class ArraySubclass(np.ndarray):
        #         pass
        a = np.zeros((0, 1, 1), dtype=np.int_)  # .view(ArraySubclass)
        res = linalg.eigvalsh(a)
        assert_(res.dtype.type is np.float64)
        assert_equal((0, 1), res.shape)
        # This is just for documentation, it might make sense to change:
        assert_(isinstance(res, np.ndarray))

        a = np.zeros((0, 0), dtype=np.complex64)  # .view(ArraySubclass)
        res = linalg.eigvalsh(a)
        assert_(res.dtype.type is np.float32)
        assert_equal((0,), res.shape)
        # This is just for documentation, it might make sense to change:
        assert_(isinstance(res, np.ndarray))


class TestEighCases(HermitianTestCase, HermitianGeneralizedTestCase):
    def do(self, a, b, tags):
        pytest.xfail(reason="sort complex")
        # note that eigenvalue arrays returned by eig must be sorted since
        # their order isn't guaranteed.
        ev, evc = linalg.eigh(a)
        evalues, evectors = linalg.eig(a)
        evalues.sort(axis=-1)
        assert_almost_equal(ev, evalues)

        assert_allclose(
            dot_generalized(a, evc),
            np.asarray(ev)[..., None, :] * np.asarray(evc),
            rtol=get_rtol(ev.dtype),
        )

        ev2, evc2 = linalg.eigh(a, "U")
        assert_almost_equal(ev2, evalues)

        assert_allclose(
            dot_generalized(a, evc2),
            np.asarray(ev2)[..., None, :] * np.asarray(evc2),
            rtol=get_rtol(ev.dtype),
            err_msg=repr(a),
        )


@instantiate_parametrized_tests
class TestEigh(TestCase):
    @parametrize("dtype", [single, double, csingle, cdouble])
    def test_types(self, dtype):
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        w, v = np.linalg.eigh(x)
        assert_equal(w.dtype, get_real_dtype(dtype))
        assert_equal(v.dtype, dtype)

    def test_invalid(self):
        x = np.array([[1, 0.5], [0.5, 1]], dtype=np.float32)
        assert_raises((RuntimeError, ValueError), np.linalg.eigh, x, UPLO="lrong")
        assert_raises((RuntimeError, ValueError), np.linalg.eigh, x, "lower")
        assert_raises((RuntimeError, ValueError), np.linalg.eigh, x, "upper")

    def test_UPLO(self):
        Klo = np.array([[0, 0], [1, 0]], dtype=np.double)
        Kup = np.array([[0, 1], [0, 0]], dtype=np.double)
        tgt = np.array([-1, 1], dtype=np.double)
        rtol = get_rtol(np.double)

        # Check default is 'L'
        w, v = np.linalg.eigh(Klo)
        assert_allclose(w, tgt, rtol=rtol)
        # Check 'L'
        w, v = np.linalg.eigh(Klo, UPLO="L")
        assert_allclose(w, tgt, rtol=rtol)
        # Check 'l'
        w, v = np.linalg.eigh(Klo, UPLO="l")
        assert_allclose(w, tgt, rtol=rtol)
        # Check 'U'
        w, v = np.linalg.eigh(Kup, UPLO="U")
        assert_allclose(w, tgt, rtol=rtol)
        # Check 'u'
        w, v = np.linalg.eigh(Kup, UPLO="u")
        assert_allclose(w, tgt, rtol=rtol)

    def test_0_size(self):
        # Check that all kinds of 0-sized arrays work
        #        class ArraySubclass(np.ndarray):
        #            pass
        a = np.zeros((0, 1, 1), dtype=np.int_)  # .view(ArraySubclass)
        res, res_v = linalg.eigh(a)
        assert_(res_v.dtype.type is np.float64)
        assert_(res.dtype.type is np.float64)
        assert_equal(a.shape, res_v.shape)
        assert_equal((0, 1), res.shape)
        # This is just for documentation, it might make sense to change:
        assert_(isinstance(a, np.ndarray))

        a = np.zeros((0, 0), dtype=np.complex64)  # .view(ArraySubclass)
        res, res_v = linalg.eigh(a)
        assert_(res_v.dtype.type is np.complex64)
        assert_(res.dtype.type is np.float32)
        assert_equal(a.shape, res_v.shape)
        assert_equal((0,), res.shape)
        # This is just for documentation, it might make sense to change:
        assert_(isinstance(a, np.ndarray))


class _TestNormBase:
    dt = None
    dec = None

    @staticmethod
    def check_dtype(x, res):
        if issubclass(x.dtype.type, np.inexact):
            assert_equal(res.dtype, x.real.dtype)
        else:
            # For integer input, don't have to test float precision of output.
            assert_(issubclass(res.dtype.type, np.floating))


class _TestNormGeneral(_TestNormBase):
    def test_empty(self):
        assert_equal(norm([]), 0.0)
        assert_equal(norm(array([], dtype=self.dt)), 0.0)
        assert_equal(norm(atleast_2d(array([], dtype=self.dt))), 0.0)

    def test_vector_return_type(self):
        a = np.array([1, 0, 1])

        exact_types = "Bbhil"  # np.typecodes["AllInteger"]
        inexact_types = "efdFD"  # np.typecodes["AllFloat"]

        all_types = exact_types + inexact_types

        for each_type in all_types:
            at = a.astype(each_type)

            if each_type == np.dtype("float16"):
                # FIXME: move looping to parametrize, add decorators=[xfail]
                # pytest.xfail("float16**float64 => float64 (?)")
                raise SkipTest("float16**float64 => float64 (?)")

            an = norm(at, -np.inf)
            self.check_dtype(at, an)
            assert_almost_equal(an, 0.0)

            with suppress_warnings() as sup:
                sup.filter(RuntimeWarning, "divide by zero encountered")
                an = norm(at, -1)
                self.check_dtype(at, an)
                assert_almost_equal(an, 0.0)

            an = norm(at, 0)
            self.check_dtype(at, an)
            assert_almost_equal(an, 2)

            an = norm(at, 1)
            self.check_dtype(at, an)
            assert_almost_equal(an, 2.0)

            an = norm(at, 2)
            self.check_dtype(at, an)
            assert_almost_equal(an, an.dtype.type(2.0) ** an.dtype.type(1.0 / 2.0))

            an = norm(at, 4)
            self.check_dtype(at, an)
            assert_almost_equal(an, an.dtype.type(2.0) ** an.dtype.type(1.0 / 4.0))

            an = norm(at, np.inf)
            self.check_dtype(at, an)
            assert_almost_equal(an, 1.0)

    def test_vector(self):
        a = [1, 2, 3, 4]
        b = [-1, -2, -3, -4]
        c = [-1, 2, -3, 4]

        def _test(v):
            np.testing.assert_almost_equal(norm(v), 30**0.5, decimal=self.dec)
            np.testing.assert_almost_equal(norm(v, inf), 4.0, decimal=self.dec)
            np.testing.assert_almost_equal(norm(v, -inf), 1.0, decimal=self.dec)
            np.testing.assert_almost_equal(norm(v, 1), 10.0, decimal=self.dec)
            np.testing.assert_almost_equal(norm(v, -1), 12.0 / 25, decimal=self.dec)
            np.testing.assert_almost_equal(norm(v, 2), 30**0.5, decimal=self.dec)
            np.testing.assert_almost_equal(
                norm(v, -2), ((205.0 / 144) ** -0.5), decimal=self.dec
            )
            np.testing.assert_almost_equal(norm(v, 0), 4, decimal=self.dec)

        for v in (
            a,
            b,
            c,
        ):
            _test(v)

        for v in (
            array(a, dtype=self.dt),
            array(b, dtype=self.dt),
            array(c, dtype=self.dt),
        ):
            _test(v)

    def test_axis(self):
        # Vector norms.
        # Compare the use of `axis` with computing the norm of each row
        # or column separately.
        A = array([[1, 2, 3], [4, 5, 6]], dtype=self.dt)
        for order in [None, -1, 0, 1, 2, 3, np.inf, -np.inf]:
            expected0 = [norm(A[:, k], ord=order) for k in range(A.shape[1])]
            assert_almost_equal(norm(A, ord=order, axis=0), expected0)
            expected1 = [norm(A[k, :], ord=order) for k in range(A.shape[0])]
            assert_almost_equal(norm(A, ord=order, axis=1), expected1)

        # Matrix norms.
        B = np.arange(1, 25, dtype=self.dt).reshape(2, 3, 4)
        nd = B.ndim
        for order in [None, -2, 2, -1, 1, np.inf, -np.inf, "fro"]:
            for axis in itertools.combinations(range(-nd, nd), 2):
                row_axis, col_axis = axis
                if row_axis < 0:
                    row_axis += nd
                if col_axis < 0:
                    col_axis += nd
                if row_axis == col_axis:
                    assert_raises(
                        (RuntimeError, ValueError), norm, B, ord=order, axis=axis
                    )
                else:
                    n = norm(B, ord=order, axis=axis)

                    # The logic using k_index only works for nd = 3.
                    # This has to be changed if nd is increased.
                    k_index = nd - (row_axis + col_axis)
                    if row_axis < col_axis:
                        expected = [
                            norm(B[:].take(k, axis=k_index), ord=order)
                            for k in range(B.shape[k_index])
                        ]
                    else:
                        expected = [
                            norm(B[:].take(k, axis=k_index).T, ord=order)
                            for k in range(B.shape[k_index])
                        ]
                    assert_almost_equal(n, expected)

    def test_keepdims(self):
        A = np.arange(1, 25
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/torch_np/numpy_tests/linalg`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/torch_np/numpy_tests/linalg`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/torch_np/numpy_tests/linalg/test_linalg.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/torch_np/numpy_tests/linalg`):

- [`test_linalg.py_kw.md_docs.md`](./test_linalg.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_linalg.py_docs.md_docs.md`
- **Keyword Index**: `test_linalg.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
