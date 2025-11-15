# Documentation: `docs/test/torch_np/numpy_tests/core/test_multiarray.py_docs.md`

## File Metadata

- **Path**: `docs/test/torch_np/numpy_tests/core/test_multiarray.py_docs.md`
- **Size**: 54,098 bytes (52.83 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/torch_np/numpy_tests/core/test_multiarray.py`

## File Metadata

- **Path**: `test/torch_np/numpy_tests/core/test_multiarray.py`
- **Size**: 250,670 bytes (244.79 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]

import builtins
import collections.abc
import ctypes
import functools
import io
import itertools
import mmap
import operator
import os
import sys
import tempfile
import warnings
import weakref
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from tempfile import mkstemp
from unittest import expectedFailure as xfail, skipIf as skipif, SkipTest

import numpy
import pytest
from pytest import raises as assert_raises

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    slowTest as slow,
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
    from numpy.testing import (
        assert_,
        assert_allclose,
        assert_almost_equal,
        assert_array_almost_equal,
        assert_array_equal,
        assert_array_less,
        assert_equal,
        assert_raises_regex,
        assert_warns,
        suppress_warnings,
    )

else:
    import torch._numpy as np
    from torch._numpy.testing import (
        assert_,
        assert_allclose,
        assert_almost_equal,
        assert_array_almost_equal,
        assert_array_equal,
        assert_array_less,
        assert_equal,
        assert_raises_regex,
        assert_warns,
        suppress_warnings,
    )


skip = functools.partial(skipif, True)

IS_PYPY = False
IS_PYSTON = False
HAS_REFCOUNT = True

if numpy.__version__ > "2":
    # numpy 2.0 +, see https://numpy.org/doc/stable/release/2.0.0-notes.html#renamed-numpy-core-to-numpy-core
    from numpy._core.tests._locales import CommaDecimalPointLocale
else:
    from numpy.core.tests._locales import CommaDecimalPointLocale

from numpy.testing._private.utils import _no_tracing, requires_memory


# #### stubs to make pytest pass the collections stage ####


# defined in numpy/testing/_utils.py
def runstring(astr, dict):
    exec(astr, dict)


@contextmanager
def temppath(*args, **kwargs):
    """Context manager for temporary files.

    Context manager that returns the path to a closed temporary file. Its
    parameters are the same as for tempfile.mkstemp and are passed directly
    to that function. The underlying file is removed when the context is
    exited, so it should be closed at that time.

    Windows does not allow a temporary file to be opened if it is already
    open, so the underlying file must be closed after opening before it
    can be opened again.

    """
    fd, path = mkstemp(*args, **kwargs)
    os.close(fd)
    try:
        yield path
    finally:
        os.remove(path)


# FIXME
np.asanyarray = np.asarray
np.asfortranarray = np.asarray

# #### end stubs


def _aligned_zeros(shape, dtype=float, order="C", align=None):
    """
    Allocate a new ndarray with aligned memory.

    The ndarray is guaranteed *not* aligned to twice the requested alignment.
    Eg, if align=4, guarantees it is not aligned to 8. If align=None uses
    dtype.alignment."""
    dtype = np.dtype(dtype)
    if dtype == np.dtype(object):
        # Can't do this, fall back to standard allocation (which
        # should always be sufficiently aligned)
        if align is not None:
            raise ValueError("object array alignment not supported")
        return np.zeros(shape, dtype=dtype, order=order)
    if align is None:
        align = dtype.alignment
    if not hasattr(shape, "__len__"):
        shape = (shape,)
    size = functools.reduce(operator.mul, shape) * dtype.itemsize
    buf = np.empty(size + 2 * align + 1, np.uint8)

    ptr = buf.__array_interface__["data"][0]
    offset = ptr % align
    if offset != 0:
        offset = align - offset
    if (ptr % (2 * align)) == 0:
        offset += align

    # Note: slices producing 0-size arrays do not necessarily change
    # data pointer --- so we use and allocate size+1
    buf = buf[offset : offset + size + 1][:-1]
    buf.fill(0)
    data = np.ndarray(shape, dtype, buf, order=order)
    return data


@xpassIfTorchDynamo_np  # (reason="TODO: flags")
@instantiate_parametrized_tests
class TestFlag(TestCase):
    def setUp(self):
        self.a = np.arange(10)

    @xfail
    def test_writeable(self):
        mydict = locals()
        self.a.flags.writeable = False
        assert_raises(ValueError, runstring, "self.a[0] = 3", mydict)
        assert_raises(ValueError, runstring, "self.a[0:1].itemset(3)", mydict)
        self.a.flags.writeable = True
        self.a[0] = 5
        self.a[0] = 0

    def test_writeable_any_base(self):
        # Ensure that any base being writeable is sufficient to change flag;
        # this is especially interesting for arrays from an array interface.
        arr = np.arange(10)

        class subclass(np.ndarray):
            pass

        # Create subclass so base will not be collapsed, this is OK to change
        view1 = arr.view(subclass)
        view2 = view1[...]
        arr.flags.writeable = False
        view2.flags.writeable = False
        view2.flags.writeable = True  # Can be set to True again.

        arr = np.arange(10)

        class frominterface:
            def __init__(self, arr):
                self.arr = arr
                self.__array_interface__ = arr.__array_interface__

        view1 = np.asarray(frominterface)
        view2 = view1[...]
        view2.flags.writeable = False
        view2.flags.writeable = True

        view1.flags.writeable = False
        view2.flags.writeable = False
        with assert_raises(ValueError):
            # Must assume not writeable, since only base is not:
            view2.flags.writeable = True

    def test_writeable_from_readonly(self):
        # gh-9440 - make sure fromstring, from buffer on readonly buffers
        # set writeable False
        data = b"\x00" * 100
        vals = np.frombuffer(data, "B")
        assert_raises(ValueError, vals.setflags, write=True)
        types = np.dtype([("vals", "u1"), ("res3", "S4")])
        values = np.core.records.fromstring(data, types)
        vals = values["vals"]
        assert_raises(ValueError, vals.setflags, write=True)

    def test_writeable_from_buffer(self):
        data = bytearray(b"\x00" * 100)
        vals = np.frombuffer(data, "B")
        assert_(vals.flags.writeable)
        vals.setflags(write=False)
        assert_(vals.flags.writeable is False)
        vals.setflags(write=True)
        assert_(vals.flags.writeable)
        types = np.dtype([("vals", "u1"), ("res3", "S4")])
        values = np.core.records.fromstring(data, types)
        vals = values["vals"]
        assert_(vals.flags.writeable)
        vals.setflags(write=False)
        assert_(vals.flags.writeable is False)
        vals.setflags(write=True)
        assert_(vals.flags.writeable)

    @skipif(IS_PYPY, reason="PyPy always copies")
    def test_writeable_pickle(self):
        import pickle

        # Small arrays will be copied without setting base.
        # See condition for using PyArray_SetBaseObject in
        # array_setstate.
        a = np.arange(1000)
        for v in range(pickle.HIGHEST_PROTOCOL):
            vals = pickle.loads(pickle.dumps(a, v))
            assert_(vals.flags.writeable)
            assert_(isinstance(vals.base, bytes))

    def test_warnonwrite(self):
        a = np.arange(10)
        a.flags._warn_on_write = True
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            a[1] = 10
            a[2] = 10
            # only warn once
            assert_(len(w) == 1)

    @parametrize(
        "flag, flag_value, writeable",
        [
            ("writeable", True, True),
            # Delete _warn_on_write after deprecation and simplify
            # the parameterization:
            ("_warn_on_write", True, False),
            ("writeable", False, False),
        ],
    )
    def test_readonly_flag_protocols(self, flag, flag_value, writeable):
        a = np.arange(10)
        setattr(a.flags, flag, flag_value)

        class MyArr:
            __array_struct__ = a.__array_struct__

        assert memoryview(a).readonly is not writeable
        assert a.__array_interface__["data"][1] is not writeable
        assert np.asarray(MyArr()).flags.writeable is writeable

    @xfail
    def test_otherflags(self):
        assert_equal(self.a.flags.carray, True)
        assert_equal(self.a.flags["C"], True)
        assert_equal(self.a.flags.farray, False)
        assert_equal(self.a.flags.behaved, True)
        assert_equal(self.a.flags.fnc, False)
        assert_equal(self.a.flags.forc, True)
        assert_equal(self.a.flags.owndata, True)
        assert_equal(self.a.flags.writeable, True)
        assert_equal(self.a.flags.aligned, True)
        assert_equal(self.a.flags.writebackifcopy, False)
        assert_equal(self.a.flags["X"], False)
        assert_equal(self.a.flags["WRITEBACKIFCOPY"], False)

    @xfail  # invalid dtype
    def test_string_align(self):
        a = np.zeros(4, dtype=np.dtype("|S4"))
        assert_(a.flags.aligned)
        # not power of two are accessed byte-wise and thus considered aligned
        a = np.zeros(5, dtype=np.dtype("|S4"))
        assert_(a.flags.aligned)

    @xfail  # structured dtypes
    def test_void_align(self):
        a = np.zeros(4, dtype=np.dtype([("a", "i4"), ("b", "i4")]))
        assert_(a.flags.aligned)


@xpassIfTorchDynamo_np  # (reason="TODO: hash")
class TestHash(TestCase):
    # see #3793
    def test_int(self):
        for st, ut, s in [
            (np.int8, np.uint8, 8),
            (np.int16, np.uint16, 16),
            (np.int32, np.uint32, 32),
            (np.int64, np.uint64, 64),
        ]:
            for i in range(1, s):
                assert_equal(
                    hash(st(-(2**i))), hash(-(2**i)), err_msg=f"{st!r}: -2**{i:d}"
                )
                assert_equal(
                    hash(st(2 ** (i - 1))),
                    hash(2 ** (i - 1)),
                    err_msg=f"{st!r}: 2**{i - 1:d}",
                )
                assert_equal(
                    hash(st(2**i - 1)),
                    hash(2**i - 1),
                    err_msg=f"{st!r}: 2**{i:d} - 1",
                )

                i = max(i - 1, 1)
                assert_equal(
                    hash(ut(2 ** (i - 1))),
                    hash(2 ** (i - 1)),
                    err_msg=f"{ut!r}: 2**{i - 1:d}",
                )
                assert_equal(
                    hash(ut(2**i - 1)),
                    hash(2**i - 1),
                    err_msg=f"{ut!r}: 2**{i:d} - 1",
                )


@xpassIfTorchDynamo_np  # (reason="TODO: hash")
class TestAttributes(TestCase):
    def setUp(self):
        self.one = np.arange(10)
        self.two = np.arange(20).reshape(4, 5)
        self.three = np.arange(60, dtype=np.float64).reshape(2, 5, 6)

    def test_attributes(self):
        assert_equal(self.one.shape, (10,))
        assert_equal(self.two.shape, (4, 5))
        assert_equal(self.three.shape, (2, 5, 6))
        self.three.shape = (10, 3, 2)
        assert_equal(self.three.shape, (10, 3, 2))
        self.three.shape = (2, 5, 6)
        assert_equal(self.one.strides, (self.one.itemsize,))
        num = self.two.itemsize
        assert_equal(self.two.strides, (5 * num, num))
        num = self.three.itemsize
        assert_equal(self.three.strides, (30 * num, 6 * num, num))
        assert_equal(self.one.ndim, 1)
        assert_equal(self.two.ndim, 2)
        assert_equal(self.three.ndim, 3)
        num = self.two.itemsize
        assert_equal(self.two.size, 20)
        assert_equal(self.two.nbytes, 20 * num)
        assert_equal(self.two.itemsize, self.two.dtype.itemsize)

    @xfailIfTorchDynamo  # use ndarray.tensor._base to track the base tensor
    def test_attributes_2(self):
        assert_equal(self.two.base, np.arange(20))

    def test_dtypeattr(self):
        assert_equal(self.one.dtype, np.dtype(np.int_))
        assert_equal(self.three.dtype, np.dtype(np.float64))
        assert_equal(self.one.dtype.char, "l")
        assert_equal(self.three.dtype.char, "d")
        assert_(self.three.dtype.str[0] in "<>")
        assert_equal(self.one.dtype.str[1], "i")
        assert_equal(self.three.dtype.str[1], "f")

    def test_stridesattr(self):
        x = self.one

        def make_array(size, offset, strides):
            return np.ndarray(
                size,
                buffer=x,
                dtype=int,
                offset=offset * x.itemsize,
                strides=strides * x.itemsize,
            )

        assert_equal(make_array(4, 4, -1), np.array([4, 3, 2, 1]))
        assert_raises(ValueError, make_array, 4, 4, -2)
        assert_raises(ValueError, make_array, 4, 2, -1)
        assert_raises(ValueError, make_array, 8, 3, 1)
        assert_equal(make_array(8, 3, 0), np.array([3] * 8))
        # Check behavior reported in gh-2503:
        assert_raises(ValueError, make_array, (2, 3), 5, np.array([-2, -3]))
        make_array(0, 0, 10)

    def test_set_stridesattr(self):
        x = self.one

        def make_array(size, offset, strides):
            try:
                r = np.ndarray([size], dtype=int, buffer=x, offset=offset * x.itemsize)
            except Exception as e:
                raise RuntimeError(e)  # noqa: B904
            r.strides = strides = strides * x.itemsize
            return r

        assert_equal(make_array(4, 4, -1), np.array([4, 3, 2, 1]))
        assert_equal(make_array(7, 3, 1), np.array([3, 4, 5, 6, 7, 8, 9]))
        assert_raises(ValueError, make_array, 4, 4, -2)
        assert_raises(ValueError, make_array, 4, 2, -1)
        assert_raises(RuntimeError, make_array, 8, 3, 1)
        # Check that the true extent of the array is used.
        # Test relies on as_strided base not exposing a buffer.
        x = np.lib.stride_tricks.as_strided(np.arange(1), (10, 10), (0, 0))

        def set_strides(arr, strides):
            arr.strides = strides

        assert_raises(ValueError, set_strides, x, (10 * x.itemsize, x.itemsize))

        # Test for offset calculations:
        x = np.lib.stride_tricks.as_strided(
            np.arange(10, dtype=np.int8)[-1], shape=(10,), strides=(-1,)
        )
        assert_raises(ValueError, set_strides, x[::-1], -1)
        a = x[::-1]
        a.strides = 1
        a[::2].strides = 2

        # test 0d
        arr_0d = np.array(0)
        arr_0d.strides = ()
        assert_raises(TypeError, set_strides, arr_0d, None)

    def test_fill(self):
        for t in "?bhilqpBHILQPfdgFDGO":
            x = np.empty((3, 2, 1), t)
            y = np.empty((3, 2, 1), t)
            x.fill(1)
            y[...] = 1
            assert_equal(x, y)

    def test_fill_max_uint64(self):
        x = np.empty((3, 2, 1), dtype=np.uint64)
        y = np.empty((3, 2, 1), dtype=np.uint64)
        value = 2**64 - 1
        y[...] = value
        x.fill(value)
        assert_array_equal(x, y)

    def test_fill_struct_array(self):
        # Filling from a scalar
        x = np.array([(0, 0.0), (1, 1.0)], dtype="i4,f8")
        x.fill(x[0])
        assert_equal(x["f1"][1], x["f1"][0])
        # Filling from a tuple that can be converted
        # to a scalar
        x = np.zeros(2, dtype=[("a", "f8"), ("b", "i4")])
        x.fill((3.5, -2))
        assert_array_equal(x["a"], [3.5, 3.5])
        assert_array_equal(x["b"], [-2, -2])

    def test_fill_readonly(self):
        # gh-22922
        a = np.zeros(11)
        a.setflags(write=False)
        with pytest.raises(ValueError, match=".*read-only"):
            a.fill(0)


@instantiate_parametrized_tests
class TestArrayConstruction(TestCase):
    def test_array(self):
        d = np.ones(6)
        r = np.array([d, d])
        assert_equal(r, np.ones((2, 6)))

        d = np.ones(6)
        tgt = np.ones((2, 6))
        r = np.array([d, d])
        assert_equal(r, tgt)
        tgt[1] = 2
        r = np.array([d, d + 1])
        assert_equal(r, tgt)

        d = np.ones(6)
        r = np.array([[d, d]])
        assert_equal(r, np.ones((1, 2, 6)))

        d = np.ones(6)
        r = np.array([[d, d], [d, d]])
        assert_equal(r, np.ones((2, 2, 6)))

        d = np.ones((6, 6))
        r = np.array([d, d])
        assert_equal(r, np.ones((2, 6, 6)))

        tgt = np.ones((2, 3), dtype=bool)
        tgt[0, 2] = False
        tgt[1, 0:2] = False
        r = np.array([[True, True, False], [False, False, True]])
        assert_equal(r, tgt)
        r = np.array([[True, False], [True, False], [False, True]])
        assert_equal(r, tgt.T)

    @skip(reason="object arrays")
    def test_array_object(self):
        d = np.ones((6,))
        r = np.array([[d, d + 1], d + 2], dtype=object)
        assert_equal(len(r), 2)
        assert_equal(r[0], [d, d + 1])
        assert_equal(r[1], d + 2)

    def test_array_empty(self):
        assert_raises(TypeError, np.array)

    def test_0d_array_shape(self):
        assert np.ones(np.array(3)).shape == (3,)

    def test_array_copy_false(self):
        d = np.array([1, 2, 3])
        e = np.array(d, copy=False)
        d[1] = 3
        assert_array_equal(e, [1, 3, 3])

    @xpassIfTorchDynamo_np  # (reason="order='F'")
    def test_array_copy_false_2(self):
        d = np.array([1, 2, 3])
        e = np.array(d, copy=False, order="F")
        d[1] = 4
        assert_array_equal(e, [1, 4, 3])
        e[2] = 7
        assert_array_equal(d, [1, 4, 7])

    def test_array_copy_true(self):
        d = np.array([[1, 2, 3], [1, 2, 3]])
        e = np.array(d, copy=True)
        d[0, 1] = 3
        e[0, 2] = -7
        assert_array_equal(e, [[1, 2, -7], [1, 2, 3]])
        assert_array_equal(d, [[1, 3, 3], [1, 2, 3]])

    @xfail  # (reason="order='F'")
    def test_array_copy_true_2(self):
        d = np.array([[1, 2, 3], [1, 2, 3]])
        e = np.array(d, copy=True, order="F")
        d[0, 1] = 5
        e[0, 2] = 7
        assert_array_equal(e, [[1, 3, 7], [1, 2, 3]])
        assert_array_equal(d, [[1, 5, 3], [1, 2, 3]])

    @xfailIfTorchDynamo
    def test_array_cont(self):
        d = np.ones(10)[::2]
        assert_(np.ascontiguousarray(d).flags.c_contiguous)
        assert_(np.ascontiguousarray(d).flags.f_contiguous)
        assert_(np.asfortranarray(d).flags.c_contiguous)
        # assert_(np.asfortranarray(d).flags.f_contiguous)   # XXX: f ordering
        d = np.ones((10, 10))[::2, ::2]
        assert_(np.ascontiguousarray(d).flags.c_contiguous)
        # assert_(np.asfortranarray(d).flags.f_contiguous)

    @parametrize(
        "func",
        [
            subtest(np.array, name="array"),
            subtest(np.asarray, name="asarray"),
            subtest(np.asanyarray, name="asanyarray"),
            subtest(np.ascontiguousarray, name="ascontiguousarray"),
            subtest(np.asfortranarray, name="asfortranarray"),
        ],
    )
    def test_bad_arguments_error(self, func):
        with pytest.raises(TypeError):
            func(3, dtype="bad dtype")
        with pytest.raises(TypeError):
            func()  # missing arguments
        with pytest.raises(TypeError):
            func(1, 2, 3, 4, 5, 6, 7, 8)  # too many arguments

    @skip(reason="np.array w/keyword argument")
    @parametrize(
        "func",
        [
            subtest(np.array, name="array"),
            subtest(np.asarray, name="asarray"),
            subtest(np.asanyarray, name="asanyarray"),
            subtest(np.ascontiguousarray, name="ascontiguousarray"),
            subtest(np.asfortranarray, name="asfortranarray"),
        ],
    )
    def test_array_as_keyword(self, func):
        # This should likely be made positional only, but do not change
        # the name accidentally.
        if func is np.array:
            func(object=3)
        else:
            func(a=3)


class TestAssignment(TestCase):
    def test_assignment_broadcasting(self):
        a = np.arange(6).reshape(2, 3)

        # Broadcasting the input to the output
        a[...] = np.arange(3)
        assert_equal(a, [[0, 1, 2], [0, 1, 2]])
        a[...] = np.arange(2).reshape(2, 1)
        assert_equal(a, [[0, 0, 0], [1, 1, 1]])

        # For compatibility with <= 1.5, a limited version of broadcasting
        # the output to the input.
        #
        # This behavior is inconsistent with NumPy broadcasting
        # in general, because it only uses one of the two broadcasting
        # rules (adding a new "1" dimension to the left of the shape),
        # applied to the output instead of an input. In NumPy 2.0, this kind
        # of broadcasting assignment will likely be disallowed.
        a[...] = np.flip(np.arange(6)).reshape(1, 2, 3)
        assert_equal(a, [[5, 4, 3], [2, 1, 0]])
        # The other type of broadcasting would require a reduction operation.

        def assign(a, b):
            a[...] = b

        assert_raises(
            (RuntimeError, ValueError), assign, a, np.arange(12).reshape(2, 2, 3)
        )

    def test_assignment_errors(self):
        # Address issue #2276
        class C:
            pass

        a = np.zeros(1)

        def assign(v):
            a[0] = v

        assert_raises((RuntimeError, TypeError), assign, C())
        # assert_raises((TypeError, ValueError), assign, [1])  # numpy raises, we do not

    @skip(reason="object arrays")
    def test_unicode_assignment(self):
        # gh-5049
        from numpy.core.numeric import set_string_function

        @contextmanager
        def inject_str(s):
            """replace ndarray.__str__ temporarily"""
            set_string_function(lambda x: s, repr=False)
            try:
                yield
            finally:
                set_string_function(None, repr=False)

        a1d = np.array(["test"])
        a0d = np.array("done")
        with inject_str("bad"):
            a1d[0] = a0d  # previously this would invoke __str__
        assert_equal(a1d[0], "done")

        # this would crash for the same reason
        np.array([np.array("\xe5\xe4\xf6")])

    @skip(reason="object arrays")
    def test_stringlike_empty_list(self):
        # gh-8902
        u = np.array(["done"])
        b = np.array([b"done"])

        class bad_sequence:
            def __getitem__(self, value):
                pass

            def __len__(self):
                raise RuntimeError

        assert_raises(ValueError, operator.setitem, u, 0, [])
        assert_raises(ValueError, operator.setitem, b, 0, [])

        assert_raises(ValueError, operator.setitem, u, 0, bad_sequence())
        assert_raises(ValueError, operator.setitem, b, 0, bad_sequence())

    @skipif(
        "torch._numpy" == np.__name__,
        reason="torch._numpy does not support extended floats and complex dtypes",
    )
    def test_longdouble_assignment(self):
        # only relevant if longdouble is larger than float
        # we're looking for loss of precision

        for dtype in (np.longdouble, np.clongdouble):
            # gh-8902
            tinyb = np.nextafter(np.longdouble(0), 1).astype(dtype)
            tinya = np.nextafter(np.longdouble(0), -1).astype(dtype)

            # construction
            tiny1d = np.array([tinya])
            assert_equal(tiny1d[0], tinya)

            # scalar = scalar
            tiny1d[0] = tinyb
            assert_equal(tiny1d[0], tinyb)

            # 0d = scalar
            tiny1d[0, ...] = tinya
            assert_equal(tiny1d[0], tinya)

            # 0d = 0d
            tiny1d[0, ...] = tinyb[...]
            assert_equal(tiny1d[0], tinyb)

            # scalar = 0d
            tiny1d[0] = tinyb[...]
            assert_equal(tiny1d[0], tinyb)

            arr = np.array([np.array(tinya)])
            assert_equal(arr[0], tinya)

    @skip(reason="object arrays")
    def test_cast_to_string(self):
        # cast to str should do "str(scalar)", not "str(scalar.item())"
        # Example: In python2, str(float) is truncated, so we want to avoid
        # str(np.float64(...).item()) as this would incorrectly truncate.
        a = np.zeros(1, dtype="S20")
        a[:] = np.array(["1.12345678901234567890"], dtype="f8")
        assert_equal(a[0], b"1.1234567890123457")


class TestDtypedescr(TestCase):
    def test_construction(self):
        d1 = np.dtype("i4")
        assert_equal(d1, np.dtype(np.int32))
        d2 = np.dtype("f8")
        assert_equal(d2, np.dtype(np.float64))


@skip  # (reason="TODO: zero-rank?")   # FIXME: revert skip into xfail
class TestZeroRank(TestCase):
    def setUp(self):
        self.d = np.array(0), np.array("x", object)

    def test_ellipsis_subscript(self):
        a, b = self.d
        assert_equal(a[...], 0)
        assert_equal(b[...], "x")
        assert_(a[...].base is a)  # `a[...] is a` in numpy <1.9.
        assert_(b[...].base is b)  # `b[...] is b` in numpy <1.9.

    def test_empty_subscript(self):
        a, b = self.d
        assert_equal(a[()], 0)
        assert_equal(b[()], "x")
        assert_(type(a[()]) is a.dtype.type)
        assert_(type(b[()]) is str)

    def test_invalid_subscript(self):
        a, b = self.d
        assert_raises(IndexError, lambda x: x[0], a)
        assert_raises(IndexError, lambda x: x[0], b)
        assert_raises(IndexError, lambda x: x[np.array([], int)], a)
        assert_raises(IndexError, lambda x: x[np.array([], int)], b)

    def test_ellipsis_subscript_assignment(self):
        a, b = self.d
        a[...] = 42
        assert_equal(a, 42)
        b[...] = ""
        assert_equal(b.item(), "")

    def test_empty_subscript_assignment(self):
        a, b = self.d
        a[()] = 42
        assert_equal(a, 42)
        b[()] = ""
        assert_equal(b.item(), "")

    def test_invalid_subscript_assignment(self):
        a, b = self.d

        def assign(x, i, v):
            x[i] = v

        assert_raises(IndexError, assign, a, 0, 42)
        assert_raises(IndexError, assign, b, 0, "")
        assert_raises(ValueError, assign, a, (), "")

    def test_newaxis(self):
        a, b = self.d
        assert_equal(a[np.newaxis].shape, (1,))
        assert_equal(a[..., np.newaxis].shape, (1,))
        assert_equal(a[np.newaxis, ...].shape, (1,))
        assert_equal(a[..., np.newaxis].shape, (1,))
        assert_equal(a[np.newaxis, ..., np.newaxis].shape, (1, 1))
        assert_equal(a[..., np.newaxis, np.newaxis].shape, (1, 1))
        assert_equal(a[np.newaxis, np.newaxis, ...].shape, (1, 1))
        assert_equal(a[(np.newaxis,) * 10].shape, (1,) * 10)

    def test_invalid_newaxis(self):
        a, b = self.d

        def subscript(x, i):
            x[i]

        assert_raises(IndexError, subscript, a, (np.newaxis, 0))
        assert_raises(IndexError, subscript, a, (np.newaxis,) * 50)

    def test_constructor(self):
        x = np.ndarray(())
        x[()] = 5
        assert_equal(x[()], 5)
        y = np.ndarray((), buffer=x)
        y[()] = 6
        assert_equal(x[()], 6)

        # strides and shape must be the same length
        with pytest.raises(ValueError):
            np.ndarray((2,), strides=())
        with pytest.raises(ValueError):
            np.ndarray((), strides=(2,))

    def test_output(self):
        x = np.array(2)
        assert_raises(ValueError, np.add, x, [1], x)

    def test_real_imag(self):
        # contiguity checks are for gh-11245
        x = np.array(1j)
        xr = x.real
        xi = x.imag

        assert_equal(xr, np.array(0))
        assert_(type(xr) is np.ndarray)
        assert_equal(xr.flags.contiguous, True)
        assert_equal(xr.flags.f_contiguous, True)

        assert_equal(xi, np.array(1))
        assert_(type(xi) is np.ndarray)
        assert_equal(xi.flags.contiguous, True)
        assert_equal(xi.flags.f_contiguous, True)


class TestScalarIndexing(TestCase):
    def setUp(self):
        self.d = np.array([0, 1])[0]

    def test_ellipsis_subscript(self):
        a = self.d
        assert_equal(a[...], 0)
        assert_equal(a[...].shape, ())

    def test_empty_subscript(self):
        a = self.d
        assert_equal(a[()], 0)
        assert_equal(a[()].shape, ())

    def test_invalid_subscript(self):
        a = self.d
        assert_raises(IndexError, lambda x: x[0], a)
        assert_raises(IndexError, lambda x: x[np.array([], int)], a)

    def test_invalid_subscript_assignment(self):
        a = self.d

        def assign(x, i, v):
            x[i] = v

        assert_raises((IndexError, TypeError), assign, a, 0, 42)

    def test_newaxis(self):
        a = self.d
        assert_equal(a[np.newaxis].shape, (1,))
        assert_equal(a[..., np.newaxis].shape, (1,))
        assert_equal(a[np.newaxis, ...].shape, (1,))
        assert_equal(a[..., np.newaxis].shape, (1,))
        assert_equal(a[np.newaxis, ..., np.newaxis].shape, (1, 1))
        assert_equal(a[..., np.newaxis, np.newaxis].shape, (1, 1))
        assert_equal(a[np.newaxis, np.newaxis, ...].shape, (1, 1))
        assert_equal(a[(np.newaxis,) * 10].shape, (1,) * 10)

    def test_invalid_newaxis(self):
        a = self.d

        def subscript(x, i):
            x[i]

        assert_raises(IndexError, subscript, a, (np.newaxis, 0))

        # this assertion fails because 50 > NPY_MAXDIMS = 32
        # assert_raises(IndexError, subscript, a, (np.newaxis,)*50)

    @xfail  # (reason="pytorch disallows overlapping assignments")
    def test_overlapping_assignment(self):
        # With positive strides
        a = np.arange(4)
        a[:-1] = a[1:]
        assert_equal(a, [1, 2, 3, 3])

        a = np.arange(4)
        a[1:] = a[:-1]
        assert_equal(a, [0, 0, 1, 2])

        # With positive and negative strides
        a = np.arange(4)
        a[:] = a[::-1]
        assert_equal(a, [3, 2, 1, 0])

        a = np.arange(6).reshape(2, 3)
        a[::-1, :] = a[:, ::-1]
        assert_equal(a, [[5, 4, 3], [2, 1, 0]])

        a = np.arange(6).reshape(2, 3)
        a[::-1, ::-1] = a[:, ::-1]
        assert_equal(a, [[3, 4, 5], [0, 1, 2]])

        # With just one element overlapping
        a = np.arange(5)
        a[:3] = a[2:]
        assert_equal(a, [2, 3, 4, 3, 4])

        a = np.arange(5)
        a[2:] = a[:3]
        assert_equal(a, [0, 1, 0, 1, 2])

        a = np.arange(5)
        a[2::-1] = a[2:]
        assert_equal(a, [4, 3, 2, 3, 4])

        a = np.arange(5)
        a[2:] = a[2::-1]
        assert_equal(a, [0, 1, 2, 1, 0])

        a = np.arange(5)
        a[2::-1] = a[:1:-1]
        assert_equal(a, [2, 3, 4, 3, 4])

        a = np.arange(5)
        a[:1:-1] = a[2::-1]
        assert_equal(a, [0, 1, 0, 1, 2])


@skip(reason="object, void, structured dtypes")
@instantiate_parametrized_tests
class TestCreation(TestCase):
    """
    Test the np.array constructor
    """

    def test_from_attribute(self):
        class x:
            def __array__(self, dtype=None):
                pass

        assert_raises(ValueError, np.array, x())

    def test_from_string(self):
        types = np.typecodes["AllInteger"] + np.typecodes["Float"]
        nstr = ["123", "123"]
        result = np.array([123, 123], dtype=int)
        for type in types:
            msg = f"String conversion for {type}"
            assert_equal(np.array(nstr, dtype=type), result, err_msg=msg)

    def test_void(self):
        arr = np.array([], dtype="V")
        assert arr.dtype == "V8"  # current default
        # Same length scalars (those that go to the same void) work:
        arr = np.array([b"1234", b"1234"], dtype="V")
        assert arr.dtype == "V4"

        # Promoting different lengths will fail (pre 1.20 this worked)
        # by going via S5 and casting to V5.
        with pytest.raises(TypeError):
            np.array([b"1234", b"12345"], dtype="V")
        with pytest.raises(TypeError):
            np.array([b"12345", b"1234"], dtype="V")

        # Check the same for the casting path:
        arr = np.array([b"1234", b"1234"], dtype="O").astype("V")
        assert arr.dtype == "V4"
        with pytest.raises(TypeError):
            np.array([b"1234", b"12345"], dtype="O").astype("V")

    @parametrize(
        #  "idx", [pytest.param(Ellipsis, id="arr"), pytest.param((), id="scalar")]
        "idx",
        [subtest(Ellipsis, name="arr"), subtest((), name="scalar")],
    )
    def test_structured_void_promotion(self, idx):
        arr = np.array(
            [np.array(1, dtype="i,i")[idx], np.array(2, dtype="i,i")[idx]], dtype="V"
        )
        assert_array_equal(arr, np.array([(1, 1), (2, 2)], dtype="i,i"))
        # The following fails to promote the two dtypes, resulting in an error
        with pytest.raises(TypeError):
            np.array(
                [np.array(1, dtype="i,i")[idx], np.array(2, dtype="i,i,i")[idx]],
                dtype="V",
            )

    def test_too_big_error(self):
        # 45341 is the smallest integer greater than sqrt(2**31 - 1).
        # 3037000500 is the smallest integer greater than sqrt(2**63 - 1).
        # We want to make sure that the square byte array with those dimensions
        # is too big on 32 or 64 bit systems respectively.
        if np.iinfo("intp").max == 2**31 - 1:
            shape = (46341, 46341)
        elif np.iinfo("intp").max == 2**63 - 1:
            shape = (3037000500, 3037000500)
        else:
            return
        assert_raises(ValueError, np.empty, shape, dtype=np.int8)
        assert_raises(ValueError, np.zeros, shape, dtype=np.int8)
        assert_raises(ValueError, np.ones, shape, dtype=np.int8)

    @skipif(
        np.dtype(np.intp).itemsize != 8, reason="malloc may not fail on 32 bit systems"
    )
    def test_malloc_fails(self):
        # This test is guaranteed to fail due to a too large allocation
        with assert_raises(np.core._exceptions._ArrayMemoryError):
            np.empty(np.iinfo(np.intp).max, dtype=np.uint8)

    def test_zeros(self):
        types = np.typecodes["AllInteger"] + np.typecodes["AllFloat"]
        for dt in types:
            d = np.zeros((13,), dtype=dt)
            assert_equal(np.count_nonzero(d), 0)
            # true for ieee floats
            assert_equal(d.sum(), 0)
            assert_(not d.any())

            d = np.zeros(2, dtype="(2,4)i4")
            assert_equal(np.count_nonzero(d), 0)
            assert_equal(d.sum(), 0)
            assert_(not d.any())

            d = np.zeros(2, dtype="4i4")
            assert_equal(np.count_nonzero(d), 0)
            assert_equal(d.sum(), 0)
            assert_(not d.any())

            d = np.zeros(2, dtype="(2,4)i4, (2,4)i4")
            assert_equal(np.count_nonzero(d), 0)

    @slow
    def test_zeros_big(self):
        # test big array as they might be allocated different by the system
        types = np.typecodes["AllInteger"] + np.typecodes["AllFloat"]
        for dt in types:
            d = np.zeros((30 * 1024**2,), dtype=dt)
            assert_(not d.any())
            # This test can fail on 32-bit systems due to insufficient
            # contiguous memory. Deallocating the previous array increases the
            # chance of success.
            del d

    def test_zeros_obj(self):
        # test initialization from PyLong(0)
        d = np.zeros((13,), dtype=object)
        assert_array_equal(d, [0] * 13)
        assert_equal(np.count_nonzero(d), 0)

    def test_zeros_obj_obj(self):
        d = np.zeros(10, dtype=[("k", object, 2)])
        assert_array_equal(d["k"], 0)

    def test_zeros_like_like_zeros(self):
        # test zeros_like returns the same as zeros
        for c in np.typecodes["All"]:
            if c == "V":
                continue
            d = np.zeros((3, 3), dtype=c)
            assert_array_equal(np.zeros_like(d), d)
            assert_equal(np.zeros_like(d).dtype, d.dtype)
        # explicitly check some special cases
        d = np.zeros((3, 3), dtype="S5")
        assert_array_equal(np.zeros_like(d), d)
        assert_equal(np.zeros_like(d).dtype, d.dtype)
        d = np.zeros((3, 3), dtype="U5")
        assert_array_equal(np.zeros_like(d), d)
        assert_equal(np.zeros_like(d).dtype, d.dtype)

        d = np.zeros((3, 3), dtype="<i4")
        assert_array_equal(np.zeros_like(d), d)
        assert_equal(np.zeros_like(d).dtype, d.dtype)
        d = np.zeros((3, 3), dtype=">i4")
        assert_array_equal(np.zeros_like(d), d)
        assert_equal(np.zeros_like(d).dtype, d.dtype)

        d = np.zeros((3, 3), dtype="<M8[s]")
        assert_array_equal(np.zeros_like(d), d)
        assert_equal(np.zeros_like(d).dtype, d.dtype)
        d = np.zeros((3, 3), dtype=">M8[s]")
        assert_array_equal(np.zeros_like(d), d)
        assert_equal(np.zeros_like(d).dtype, d.dtype)

        d = np.zeros((3, 3), dtype="f4,f4")
        assert_array_equal(np.zeros_like(d), d)
        assert_equal(np.zeros_like(d).dtype, d.dtype)

    def test_empty_unicode(self):
        # don't throw decode errors on garbage memory
        for i in range(5, 100, 5):
            d = np.empty(i, dtype="U")
            str(d)

    def test_sequence_non_homogeneous(self):
        assert_equal(np.array([4, 2**80]).dtype, object)
        assert_equal(np.array([4, 2**80, 4]).dtype, object)
        assert_equal(np.array([2**80, 4]).dtype, object)
        assert_equal(np.array([2**80] * 3).dtype, object)
        assert_equal(np.array([[1, 1], [1j, 1j]]).dtype, complex)
        assert_equal(np.array([[1j, 1j], [1, 1]]).dtype, complex)
        assert_equal(np.array([[1, 1, 1], [1, 1j, 1.0], [1, 1, 1]]).dtype, complex)

    def test_non_sequence_sequence(self):
        """Should not segfault.

        Class Fail breaks the sequence protocol for new style classes, i.e.,
        those derived from object. Class Map is a mapping type indicated by
        raising a ValueError. At some point we may raise a warning instead
        of an error in the Fail case.

        """

        class Fail:
            def __len__(self):
                return 1

            def __getitem__(self, index):
                raise ValueError

        class Map:
            def __len__(self):
                return 1

            def __getitem__(self, index):
                raise KeyError

        a = np.array([Map()])
        assert_(a.shape == (1,))
        assert_(a.dtype == np.dtype(object))
        assert_raises(ValueError, np.array, [Fail()])

    def test_no_len_object_type(self):
        # gh-5100, want object array from iterable object without len()
        class Point2:
            def __init__(self) -> None:
                pass

            def __getitem__(self, ind):
                if ind in [0, 1]:
                    return ind
                else:
                    raise IndexError

        d = np.array([Point2(), Point2(), Point2()])
        assert_equal(d.dtype, np.dtype(object))

    def test_false_len_sequence(self):
        # gh-7264, segfault for this example
        class C:
            def __getitem__(self, i):
                raise IndexError

            def __len__(self):
                return 42

        a = np.array(C())  # segfault?
        assert_equal(len(a), 0)

    def test_false_len_iterable(self):
        # Special case where a bad __getitem__ makes us fall back on __iter__:
        class C:
            def __getitem__(self, x):
                raise Exception  # noqa: TRY002

            def __iter__(self):
                return iter(())

            def __len__(self):
                return 2

        a = np.empty(2)
        with assert_raises(ValueError):
            a[:] = C()  # Segfault!

        assert_equal(np.array(C()), list(C()))

    def test_failed_len_sequence(self):
        # gh-7393
        class A:
            def __init__(self, data):
                self._data = data

            def __getitem__(self, item):
                return type(self)(self._data[item])

            def __len__(self):
                return len(self._data)

        # len(d) should give 3, but len(d[0]) will fail
        d = A([1, 2, 3])
        assert_equal(len(np.array(d)), 3)

    def test_array_too_big(self):
        # Test that array creation succeeds for arrays addressable by intp
        # on the byte level and fails for too large arrays.
        buf = np.zeros(100)

        max_bytes = np.iinfo(np.intp).max
        for dtype in ["intp", "S20", "b"]:
            dtype = np.dtype(dtype)
            itemsize = dtype.itemsize

            np.ndarray(
                buffer=buf, strides=(0,), shape=(max_bytes // itemsize,), dtype=dtype
            )
            assert_raises(
                ValueError,
                np.ndarray,
                buffer=buf,
                strides=(0,),
                shape=(max_bytes // itemsize + 1,),
                dtype=dtype,
            )

    def _ragged_creation(self, seq):
        # without dtype=object, the ragged object raises
        with pytest.raises(ValueError, match=".*detected shape was"):
            np.array(seq)

        return np.array(seq, dtype=object)

    def test_ragged_ndim_object(self):
        # Lists of mismatching depths are treated as object arrays
        a = self._ragged_creation([[1], 2, 3])
        assert_equal(a.shape, (3,))
        assert_equal(a.dtype, object)

        a = self._ragged_creation([1, [2], 3])
        assert_equal(a.shape, (3,))
        assert_equal(a.dtype, object)

        a = self._ragged_creation([1, 2, [3]])
        assert_equal(a.shape, (3,))
        assert_equal(a.dtype, object)

    def test_ragged_shape_object(self):
        # The ragged dimension of a list is turned into an object array
        a = self._ragged_creation([[1, 1], [2], [3]])
        assert_equal(a.shape, (3,))
        assert_equal(a.dtype, object)

        a = self._ragged_creation([[1], [2, 2], [3]])
        assert_equal(a.shape, (3,))
        assert_equal(a.dtype, object)

        a = self._ragged_creation([[1], [2], [3, 3]])
        assert a.shape == (3,)
        assert a.dtype == object

    def test_array_of_ragged_array(self):
        outer = np.array([None, None])
        outer[0] = outer[1] = np.array([1, 2, 3])
        assert np.array(outer).shape == (2,)
        assert np.array([outer]).shape == (1, 2)

        outer_ragged = np.array([None, None])
        outer_ragged[0] = np.array([1, 2, 3])
        outer_ragged[1] = np.array([1, 2, 3, 4])
        # should both of these emit deprecation warnings?
        assert np.array(outer_ragged).shape == (2,)
        assert np.array([outer_ragged]).shape == (
            1,
            2,
        )

    def test_deep_nonragged_object(self):
        # None of these should raise, even though they are missing dtype=object
        np.array([[[Decimal(1)]]])
        np.array([1, Decimal(1)])
        np.array([[1], [Decimal(1)]])

    @parametrize("dtype", [object, "O,O", "O,(3)O", "(2,3)O"])
    @parametrize(
        "function",
        [
            np.ndarray,
            np.empty,
            lambda shape, dtype: np.empty_like(np.empty(shape, dtype=dtype)),
        ],
    )
    def test_object_initialized_to_None(self, function, dtype):
        # NumPy has support for object fields to be NULL (meaning None)
        # but generally, we should always fill with the proper None, and
        # downstream may rely on that.  (For fully initialized arrays!)
        arr = function(3, dtype=dtype)
        # We expect a fill value of None, which is not NULL:
        expected = np.array(None).tobytes()
        expected = expected * (arr.nbytes // len(expected))
        assert arr.tobytes() == expected


class TestBool(TestCase):
    @xfail  # (reason="bools not interned")
    def test_test_interning(self):
        a0 = np.bool_(0)
        b0 = np.bool_(False)
        assert_(a0 is b0)
        a1 = np.bool_(1)
        b1 = np.bool_(True)
        assert_(a1 is b1)
        assert_(np.array([True])[0] is a1)
        assert_(np.array(True)[()] is a1)

    def test_sum(self):
        d = np.ones(101, dtype=bool)
        assert_equal(d.sum(), d.size)
        assert_equal(d[::2].sum(), d[::2].size)
        # assert_equal(d[::-2].sum(), d[::-2].size)

    @xpassIfTorchDynamo_np  # (reason="frombuffer")
    def test_sum_2(self):
        d = np.frombuffer(b"\xff\xff" * 100, dtype=bool)
        assert_equal(d.sum(), d.size)
        assert_equal(d[::2].sum(), d[::2].size)
        assert_equal(d[::-2].sum(), d[::-2].size)

    def check_count_nonzero(self, power, length):
        powers = [2**i for i in range(length)]
        for i in range(2**power):
            l = [(i & x) != 0 for x in powers]
            a = np.array(l, dtype=bool)
            c = builtins.sum(l)
            assert_equal(np.count_nonzero(a), c)
            av = a.view(np.uint8)
            av *= 3
            assert_equal(np.count_nonzero(a), c)
            av *= 4
            assert_equal(np.count_nonzero(a), c)
            av[av != 0] = 0xFF
            assert_equal(np.count_nonzero(a), c)

    def test_count_nonzero(self):
        # check all 12 bit combinations in a length 17 array
        # covers most cases of the 16 byte unrolled code
        self.check_count_nonzero(12, 17)

    @slow
    def test_count_nonzero_all(self):
        # check all combinations in a length 17 array
        # covers all cases of the 16 byte unrolled code
        self.check_count_nonzero(17, 17)

    def test_count_nonzero_unaligned(self):
        # prevent mistakes as e.g. gh-4060
        for o in range(7):
            a = np.zeros((18,), dtype=bool)[o + 1 :]
            a[:o] = True
            assert_equal(np.count_nonzero(a), builtins.sum(a.tolist()))
            a = np.ones((18,), dtype=bool)[o + 1 :]
            a[:o] = False
            assert_equal(np.count_nonzero(a), builtins.sum(a.tolist()))

    def _test_cast_from_flexible(self, dtype):
        # empty string -> false
        for n in range(3):
            v = np.array(b"", (dtype, n))
            assert_equal(bool(v), False)
            assert_equal(bool(v[()]), False)
            assert_equal(v.astype(bool), False)
            assert_(isinstance(v.astype(bool), np.ndarray))
            assert_(v[()].astype(bool) is np.False_)

        # anything else -> true
        for n in range(1, 4):
            for val in [b"a", b"0", b" "]:
                v = np.array(val, (dtype, n))
                assert_equal(bool(v), True)
                assert_equal(bool(v[()]), True)
                assert_equal(v.astype(bool), True)
                assert_(isinstance(v.astype(bool), np.ndarray))
                assert_(v[()].astype(bool) is np.True_)

    @skip(reason="np.void")
    def test_cast_from_void(self):
        self._test_cast_from_flexible(np.void)

    @xfail  # (reason="See gh-9847")
    def test_cast_from_unicode(self):
        self._test_cast_from_flexible(np.str_)

    @xfail  # (reason="See gh-9847")
    def test_cast_from_bytes(self):
        self._test_cast_from_flexible(np.bytes_)


@instantiate_parametrized_tests
class TestMethods(TestCase):
    sort_kinds = ["quicksort", "heapsort", "stable"]

    @xpassIfTorchDynamo_np  # (reason="all(..., where=...)")
    def test_all_where(self):
        a = np.array([[True, False, True], [False, False, False], [True, True, True]])
        wh_full = np.array(
            [[True, False, True], [False, False, False], [True, False, True]]
        )
        wh_lower = np.array([[False], [False], [True]])
        for _ax in [0, None]:
            assert_equal(
                a.all(axis=_ax, where=wh_lower), np.all(a[wh_lower[:, 0], :], axis=_ax)
            )
            assert_equal(
                np.all(a, axis=_ax, where=wh_lower), a[wh_lower[:, 0], :].all(axis=_ax)
            )

        assert_equal(a.all(where=wh_full), True)
        assert_equal(np.all(a, where=wh_full), True)
        assert_equal(a.all(where=False), True)
        assert_equal(np.all(a, where=False), True)

    @xpassIfTorchDynamo_np  # (reason="any(..., where=...)")
    def test_any_where(self):
        a = np.array([[True, False, True], [False, False, False], [True, True, True]])
        wh_full = np.array(
            [[False, True, False], [True, True, True], [False, False, False]]
        )
        wh_middle = np.array([[False], [True], [False]])
        for _ax in [0, None]:
            assert_equal(
                a.any(axis=_ax, where=wh_middle),
                np.any(a[wh_middle[:, 0], :], axis=_ax),
            )
            assert_equal(
                np.any(a, axis=_ax, where=wh_middle),
                a[wh_middle[:, 0], :].any(axis=_ax),
            )
        assert_equal(a.any(where=wh_full), False)
        assert_equal(np.any(a, where=wh_full), False)
        assert_equal(a.any(where=False), False)
        assert_equal(np.any(a, where=False), False)

    @xpassIfTorchDynamo_np  # (reason="TODO: compress")
    def test_compress(self):
        tgt = [[5, 6, 7, 8, 9]]
        arr = np.arange(10).reshape(2, 5)
        out = arr.compress([0, 1], axis=0)
        assert_equal(out, tgt)

        tgt = [[1, 3], [6, 8]]
        out = arr.compress([0, 1, 0, 1, 0], axis=1)
        assert_equal(out, tgt)

        tgt = [[1], [6]]
        arr = np.arange(10).reshape(2, 5)
        out = arr.compress([0, 1], axis=1)
        assert_equal(out, tgt)

        arr = np.arange(10).reshape(2, 5)
        out = arr.compress([0, 1])
        assert_equal(out, 1)

    def test_choose(self):
        x = 2 * np.ones((3,), dtype=int)
        y = 3 * np.ones((3,), dtype=int)
        
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/torch_np/numpy_tests/core`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/torch_np/numpy_tests/core`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized
- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/torch_np/numpy_tests/core/test_multiarray.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/torch_np/numpy_tests/core`):

- [`test_scalar_methods.py_docs.md_docs.md`](./test_scalar_methods.py_docs.md_docs.md)
- [`test_einsum.py_docs.md_docs.md`](./test_einsum.py_docs.md_docs.md)
- [`test_scalarmath.py_kw.md_docs.md`](./test_scalarmath.py_kw.md_docs.md)
- [`test_scalarmath.py_docs.md_docs.md`](./test_scalarmath.py_docs.md_docs.md)
- [`test_shape_base.py_docs.md_docs.md`](./test_shape_base.py_docs.md_docs.md)
- [`test_numerictypes.py_docs.md_docs.md`](./test_numerictypes.py_docs.md_docs.md)
- [`test_scalar_ctors.py_docs.md_docs.md`](./test_scalar_ctors.py_docs.md_docs.md)
- [`test_scalar_methods.py_kw.md_docs.md`](./test_scalar_methods.py_kw.md_docs.md)
- [`test_indexing.py_docs.md_docs.md`](./test_indexing.py_docs.md_docs.md)
- [`test_scalar_ctors.py_kw.md_docs.md`](./test_scalar_ctors.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_multiarray.py_docs.md_docs.md`
- **Keyword Index**: `test_multiarray.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
