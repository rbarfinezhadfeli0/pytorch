# Documentation: test_multiarray.py

## File Metadata
- **Path**: `test/torch_np/numpy_tests/core/test_multiarray.py`
- **Size**: 250670 bytes
- **Lines**: 6989
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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
        x2 = 2 * np.ones((2, 3), dtype=int)
        y2 = 3 * np.ones((2, 3), dtype=int)
        ind = np.array([0, 0, 1])

        A = ind.choose((x, y))
        assert_equal(A, [2, 2, 3])

        A = ind.choose((x2, y2))
        assert_equal(A, [[2, 2, 3], [2, 2, 3]])

        A = ind.choose((x, y2))
        assert_equal(A, [[2, 2, 3], [2, 2, 3]])

        out = np.array(0)
        ret = np.choose(np.array(1), [10, 20, 30], out=out)
        assert out is ret
        assert_equal(out[()], 20)

    @xpassIfTorchDynamo_np  # (reason="choose(..., mode=...) not implemented")
    def test_choose_2(self):
        # gh-6272 check overlap on out
        x = np.arange(5)
        y = np.choose([0, 0, 0], [x[:3], x[:3], x[:3]], out=x[1:4], mode="wrap")
        assert_equal(y, np.array([0, 1, 2]))

    def test_prod(self):
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
                assert_raises(ArithmeticError, a.prod)
                assert_raises(ArithmeticError, a2.prod, axis=1)
            else:
                assert_equal(a.prod(axis=0), 26400)
                assert_array_equal(a2.prod(axis=0), np.array([50, 36, 84, 180], ctype))
                assert_array_equal(a2.prod(axis=-1), np.array([24, 1890, 600], ctype))

    def test_repeat(self):
        m = np.array([1, 2, 3, 4, 5, 6])
        m_rect = m.reshape((2, 3))

        A = m.repeat([1, 3, 2, 1, 1, 2])
        assert_equal(A, [1, 2, 2, 2, 3, 3, 4, 5, 6, 6])

        A = m.repeat(2)
        assert_equal(A, [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])

        A = m_rect.repeat([2, 1], axis=0)
        assert_equal(A, [[1, 2, 3], [1, 2, 3], [4, 5, 6]])

        A = m_rect.repeat([1, 3, 2], axis=1)
        assert_equal(A, [[1, 2, 2, 2, 3, 3], [4, 5, 5, 5, 6, 6]])

        A = m_rect.repeat(2, axis=0)
        assert_equal(A, [[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6]])

        A = m_rect.repeat(2, axis=1)
        assert_equal(A, [[1, 1, 2, 2, 3, 3], [4, 4, 5, 5, 6, 6]])

    @xpassIfTorchDynamo_np  # (reason="reshape(..., order='F')")
    def test_reshape(self):
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

        tgt = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        assert_equal(arr.reshape(2, 6), tgt)

        tgt = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        assert_equal(arr.reshape(3, 4), tgt)

        tgt = [[1, 10, 8, 6], [4, 2, 11, 9], [7, 5, 3, 12]]
        assert_equal(arr.reshape((3, 4), order="F"), tgt)

        tgt = [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]]
        assert_equal(arr.T.reshape((3, 4), order="C"), tgt)

    def test_round(self):
        def check_round(arr, expected, *round_args):
            assert_equal(arr.round(*round_args), expected)
            # With output array
            out = np.zeros_like(arr)
            res = arr.round(*round_args, out=out)
            assert_equal(out, expected)
            assert out is res

        check_round(np.array([1.2, 1.5]), [1, 2])
        check_round(np.array(1.5), 2)
        check_round(np.array([12.2, 15.5]), [10, 20], -1)
        check_round(np.array([12.15, 15.51]), [12.2, 15.5], 1)
        # Complex rounding
        check_round(np.array([4.5 + 1.5j]), [4 + 2j])
        check_round(np.array([12.5 + 15.5j]), [10 + 20j], -1)

    def test_squeeze(self):
        a = np.array([[[1], [2], [3]]])
        assert_equal(a.squeeze(), [1, 2, 3])
        assert_equal(a.squeeze(axis=(0,)), [[1], [2], [3]])
        #  assert_raises(ValueError, a.squeeze, axis=(1,))   # a noop in pytorch
        assert_equal(a.squeeze(axis=(2,)), [[1, 2, 3]])

    def test_transpose(self):
        a = np.array([[1, 2], [3, 4]])
        assert_equal(a.transpose(), [[1, 3], [2, 4]])
        assert_raises((RuntimeError, ValueError), lambda: a.transpose(0))
        assert_raises((RuntimeError, ValueError), lambda: a.transpose(0, 0))
        assert_raises((RuntimeError, ValueError), lambda: a.transpose(0, 1, 2))

    def test_sort(self):
        # test ordering for floats and complex containing nans. It is only
        # necessary to check the less-than comparison, so sorts that
        # only follow the insertion sort path are sufficient. We only
        # test doubles and complex doubles as the logic is the same.

        # check doubles
        msg = "Test real sort order with nans"
        a = np.array([np.nan, 1, 0])
        b = np.sort(a)
        assert_equal(b, np.flip(a), msg)

    @xpassIfTorchDynamo_np  # (reason="sort complex")
    def test_sort_complex_nans(self):
        # check complex
        msg = "Test complex sort order with nans"
        a = np.zeros(9, dtype=np.complex128)
        a.real += [np.nan, np.nan, np.nan, 1, 0, 1, 1, 0, 0]
        a.imag += [np.nan, 1, 0, np.nan, np.nan, 1, 0, 1, 0]
        b = np.sort(a)
        assert_equal(b, a[::-1], msg)

    # all c scalar sorts use the same code with different types
    # so it suffices to run a quick check with one type. The number
    # of sorted items must be greater than ~50 to check the actual
    # algorithm because quick and merge sort fall over to insertion
    # sort for small arrays.

    @parametrize("dtype", [np.uint8, np.float16, np.float32, np.float64])
    def test_sort_unsigned(self, dtype):
        a = np.arange(101, dtype=dtype)
        b = np.flip(a)
        for kind in self.sort_kinds:
            msg = f"scalar sort, kind={kind}"
            c = a.copy()
            c.sort(kind=kind)
            assert_equal(c, a, msg)
            c = b.copy()
            c.sort(kind=kind)
            assert_equal(c, a, msg)

    @parametrize(
        "dtype",
        [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64],
    )
    def test_sort_signed(self, dtype):
        a = np.arange(-50, 51, dtype=dtype)
        b = np.flip(a)
        for kind in self.sort_kinds:
            msg = f"scalar sort, kind={kind}"
            c = a.copy()
            c.sort(kind=kind)
            assert_equal(c, a, msg)
            c = b.copy()
            c.sort(kind=kind)
            assert_equal(c, a, msg)

    @xpassIfTorchDynamo_np  # (reason="sort complex")
    @parametrize("dtype", [np.float32, np.float64])
    @parametrize("part", ["real", "imag"])
    def test_sort_complex(self, part, dtype):
        # test complex sorts. These use the same code as the scalars
        # but the compare function differs.
        cdtype = {
            np.single: np.csingle,
            np.double: np.cdouble,
        }[dtype]
        a = np.arange(-50, 51, dtype=dtype)
        b = a[::-1].copy()
        ai = (a * (1 + 1j)).astype(cdtype)
        bi = (b * (1 + 1j)).astype(cdtype)
        setattr(ai, part, 1)
        setattr(bi, part, 1)
        for kind in self.sort_kinds:
            msg = f"complex sort, {part} part == 1, kind={kind}"
            c = ai.copy()
            c.sort(kind=kind)
            assert_equal(c, ai, msg)
            c = bi.copy()
            c.sort(kind=kind)
            assert_equal(c, ai, msg)

    def test_sort_axis(self):
        # check axis handling. This should be the same for all type
        # specific sorts, so we only check it for one type and one kind
        a = np.array([[3, 2], [1, 0]])
        b = np.array([[1, 0], [3, 2]])
        c = np.array([[2, 3], [0, 1]])
        d = a.copy()
        d.sort(axis=0)
        assert_equal(d, b, "test sort with axis=0")
        d = a.copy()
        d.sort(axis=1)
        assert_equal(d, c, "test sort with axis=1")
        d = a.copy()
        d.sort()
        assert_equal(d, c, "test sort with default axis")

    def test_sort_size_0(self):
        # check axis handling for multidimensional empty arrays
        a = np.array([])
        a = a.reshape(3, 2, 1, 0)
        for axis in range(-a.ndim, a.ndim):
            msg = f"test empty array sort with axis={axis}"
            assert_equal(np.sort(a, axis=axis), a, msg)
        msg = "test empty array sort with axis=None"
        assert_equal(np.sort(a, axis=None), a.ravel(), msg)

    @skip(reason="waaay tooo sloooow")
    def test_sort_degraded(self):
        # test degraded dataset would take minutes to run with normal qsort
        d = np.arange(1000000)
        do = d.copy()
        x = d
        # create a median of 3 killer where each median is the sorted second
        # last element of the quicksort partition
        while x.size > 3:
            mid = x.size // 2
            x[mid], x[-2] = x[-2], x[mid]
            x = x[:-2]

        assert_equal(np.sort(d), do)
        assert_equal(d[np.argsort(d)], do)

    @xfail  # (reason="order='F'")
    def test_copy(self):
        def assert_fortran(arr):
            assert_(arr.flags.fortran)
            assert_(arr.flags.f_contiguous)
            assert_(not arr.flags.c_contiguous)

        def assert_c(arr):
            assert_(not arr.flags.fortran)
            assert_(not arr.flags.f_contiguous)
            assert_(arr.flags.c_contiguous)

        a = np.empty((2, 2), order="F")
        # Test copying a Fortran array
        assert_c(a.copy())
        assert_c(a.copy("C"))
        assert_fortran(a.copy("F"))
        assert_fortran(a.copy("A"))

        # Now test starting with a C array.
        a = np.empty((2, 2), order="C")
        assert_c(a.copy())
        assert_c(a.copy("C"))
        assert_fortran(a.copy("F"))
        assert_c(a.copy("A"))

    @skip(reason="no .ctypes attribute")
    @parametrize("dtype", [np.int32])
    def test__deepcopy__(self, dtype):
        # Force the entry of NULLs into array
        a = np.empty(4, dtype=dtype)
        ctypes.memset(a.ctypes.data, 0, a.nbytes)

        # Ensure no error is raised, see gh-21833
        b = a.__deepcopy__({})

        a[0] = 42
        with pytest.raises(AssertionError):
            assert_array_equal(a, b)

    def test_argsort(self):
        # all c scalar argsorts use the same code with different types
        # so it suffices to run a quick check with one type. The number
        # of sorted items must be greater than ~50 to check the actual
        # algorithm because quick and merge sort fall over to insertion
        # sort for small arrays.

        for dtype in [np.int32, np.uint8, np.float32]:
            a = np.arange(101, dtype=dtype)
            b = np.flip(a)
            for kind in self.sort_kinds:
                msg = f"scalar argsort, kind={kind}, dtype={dtype}"
                assert_equal(a.copy().argsort(kind=kind), a, msg)
                assert_equal(b.copy().argsort(kind=kind), b, msg)

    @skip(reason="argsort complex")
    def test_argsort_complex(self):
        a = np.arange(101, dtype=np.float32)
        b = np.flip(a)

        # test complex argsorts. These use the same code as the scalars
        # but the compare function differs.
        ai = a * 1j + 1
        bi = b * 1j + 1
        for kind in self.sort_kinds:
            msg = f"complex argsort, kind={kind}"
            assert_equal(ai.copy().argsort(kind=kind), a, msg)
            assert_equal(bi.copy().argsort(kind=kind), b, msg)
        ai = a + 1j
        bi = b + 1j
        for kind in self.sort_kinds:
            msg = f"complex argsort, kind={kind}"
            assert_equal(ai.copy().argsort(kind=kind), a, msg)
            assert_equal(bi.copy().argsort(kind=kind), b, msg)

        # test argsort of complex arrays requiring byte-swapping, gh-5441
        for endianness in "<>":
            for dt in np.typecodes["Complex"]:
                arr = np.array([1 + 3.0j, 2 + 2.0j, 3 + 1.0j], dtype=endianness + dt)
                msg = f"byte-swapped complex argsort, dtype={dt}"
                assert_equal(arr.argsort(), np.arange(len(arr), dtype=np.intp), msg)

    @xpassIfTorchDynamo_np  # (reason="argsort axis TODO")
    def test_argsort_axis(self):
        # check axis handling. This should be the same for all type
        # specific argsorts, so we only check it for one type and one kind
        a = np.array([[3, 2], [1, 0]])
        b = np.array([[1, 1], [0, 0]])
        c = np.array([[1, 0], [1, 0]])
        assert_equal(a.copy().argsort(axis=0), b)
        assert_equal(a.copy().argsort(axis=1), c)
        assert_equal(a.copy().argsort(), c)

        # check axis handling for multidimensional empty arrays
        a = np.array([])
        a = a.reshape(3, 2, 1, 0)
        for axis in range(-a.ndim, a.ndim):
            msg = f"test empty array argsort with axis={axis}"
            assert_equal(np.argsort(a, axis=axis), np.zeros_like(a, dtype=np.intp), msg)
        msg = "test empty array argsort with axis=None"
        assert_equal(
            np.argsort(a, axis=None), np.zeros_like(a.ravel(), dtype=np.intp), msg
        )

        # check that stable argsorts are stable
        r = np.arange(100)
        # scalars
        a = np.zeros(100)
        assert_equal(a.argsort(kind="m"), r)
        # complex
        a = np.zeros(100, dtype=complex)
        assert_equal(a.argsort(kind="m"), r)
        # string
        a = np.array(["aaaaaaaaa" for i in range(100)])
        assert_equal(a.argsort(kind="m"), r)
        # unicode
        a = np.array(["aaaaaaaaa" for i in range(100)], dtype=np.str_)
        assert_equal(a.argsort(kind="m"), r)

    @xpassIfTorchDynamo_np  # (reason="TODO: searchsorted with nans differs in pytorch")
    @parametrize(
        "a",
        [
            subtest(np.array([0, 1, np.nan], dtype=np.float16), name="f16"),
            subtest(np.array([0, 1, np.nan], dtype=np.float32), name="f32"),
            subtest(np.array([0, 1, np.nan]), name="default_dtype"),
        ],
    )
    def test_searchsorted_floats(self, a):
        # test for floats arrays containing nans. Explicitly test
        # half, single, and double precision floats to verify that
        # the NaN-handling is correct.
        msg = f"Test real ({a.dtype}) searchsorted with nans, side='l'"
        b = a.searchsorted(a, side="left")
        assert_equal(b, np.arange(3), msg)
        msg = f"Test real ({a.dtype}) searchsorted with nans, side='r'"
        b = a.searchsorted(a, side="right")
        assert_equal(b, np.arange(1, 4), msg)
        # check keyword arguments
        a.searchsorted(v=1)
        x = np.array([0, 1, np.nan], dtype="float32")
        y = np.searchsorted(x, x[-1])
        assert_equal(y, 2)

    @xfail  # (
    #    reason="'searchsorted_out_cpu' not implemented for 'ComplexDouble'"
    # )
    def test_searchsorted_complex(self):
        # test for complex arrays containing nans.
        # The search sorted routines use the compare functions for the
        # array type, so this checks if that is consistent with the sort
        # order.
        # check double complex
        a = np.zeros(9, dtype=np.complex128)
        a.real += [0, 0, 1, 1, 0, 1, np.nan, np.nan, np.nan]
        a.imag += [0, 1, 0, 1, np.nan, np.nan, 0, 1, np.nan]
        msg = "Test complex searchsorted with nans, side='l'"
        b = a.searchsorted(a, side="left")
        assert_equal(b, np.arange(9), msg)
        msg = "Test complex searchsorted with nans, side='r'"
        b = a.searchsorted(a, side="right")
        assert_equal(b, np.arange(1, 10), msg)
        msg = "Test searchsorted with little endian, side='l'"
        a = np.array([0, 128], dtype="<i4")
        b = a.searchsorted(np.array(128, dtype="<i4"))
        assert_equal(b, 1, msg)
        msg = "Test searchsorted with big endian, side='l'"
        a = np.array([0, 128], dtype=">i4")
        b = a.searchsorted(np.array(128, dtype=">i4"))
        assert_equal(b, 1, msg)

    def test_searchsorted_n_elements(self):
        # Check 0 elements
        a = np.ones(0)
        b = a.searchsorted([0, 1, 2], "left")
        assert_equal(b, [0, 0, 0])
        b = a.searchsorted([0, 1, 2], "right")
        assert_equal(b, [0, 0, 0])
        a = np.ones(1)
        # Check 1 element
        b = a.searchsorted([0, 1, 2], "left")
        assert_equal(b, [0, 0, 1])
        b = a.searchsorted([0, 1, 2], "right")
        assert_equal(b, [0, 1, 1])
        # Check all elements equal
        a = np.ones(2)
        b = a.searchsorted([0, 1, 2], "left")
        assert_equal(b, [0, 0, 2])
        b = a.searchsorted([0, 1, 2], "right")
        assert_equal(b, [0, 2, 2])

    @xpassIfTorchDynamo_np  # (
    #    reason="RuntimeError: self.storage_offset() must be divisible by 8"
    # )
    def test_searchsorted_unaligned_array(self):
        # Test searching unaligned array
        a = np.arange(10)
        aligned = np.empty(a.itemsize * a.size + 1, dtype="uint8")
        unaligned = aligned[1:].view(a.dtype)
        unaligned[:] = a
        # Test searching unaligned array
        b = unaligned.searchsorted(a, "left")
        assert_equal(b, a)
        b = unaligned.searchsorted(a, "right")
        assert_equal(b, a + 1)
        # Test searching for unaligned keys
        b = a.searchsorted(unaligned, "left")
        assert_equal(b, a)
        b = a.searchsorted(unaligned, "right")
        assert_equal(b, a + 1)

    def test_searchsorted_resetting(self):
        # Test smart resetting of binsearch indices
        a = np.arange(5)
        b = a.searchsorted([6, 5, 4], "left")
        assert_equal(b, [5, 5, 4])
        b = a.searchsorted([6, 5, 4], "right")
        assert_equal(b, [5, 5, 5])

    def test_searchsorted_type_specific(self):
        # Test all type specific binary search functions
        types = "".join((np.typecodes["AllInteger"], np.typecodes["Float"]))
        for dt in types:
            if dt == "?":
                a = np.arange(2, dtype=dt)
                out = np.arange(2)
            else:
                a = np.arange(0, 5, dtype=dt)
                out = np.arange(5)
            b = a.searchsorted(a, "left")
            assert_equal(b, out)
            b = a.searchsorted(a, "right")
            assert_equal(b, out + 1)

    @xpassIfTorchDynamo_np  # (reason="ndarray ctor")
    def test_searchsorted_type_specific_2(self):
        # Test all type specific binary search functions
        types = "".join((np.typecodes["AllInteger"], np.typecodes["AllFloat"], "?"))
        for dt in types:
            if dt == "?":
                a = np.arange(2, dtype=dt)
            else:
                a = np.arange(0, 5, dtype=dt)

            # Test empty array, use a fresh array to get warnings in
            # valgrind if access happens.
            e = np.ndarray(shape=0, buffer=b"", dtype=dt)
            b = e.searchsorted(a, "left")
            assert_array_equal(b, np.zeros(len(a), dtype=np.intp))
            b = a.searchsorted(e, "left")
            assert_array_equal(b, np.zeros(0, dtype=np.intp))

    def test_searchsorted_with_invalid_sorter(self):
        a = np.array([5, 2, 1, 3, 4])
        assert_raises((TypeError, RuntimeError), np.searchsorted, a, 0, sorter=[1.1])
        assert_raises(
            (ValueError, RuntimeError), np.searchsorted, a, 0, sorter=[1, 2, 3, 4]
        )
        assert_raises(
            (ValueError, RuntimeError), np.searchsorted, a, 0, sorter=[1, 2, 3, 4, 5, 6]
        )

        # bounds check : XXX torch does not raise
        # assert_raises(ValueError, np.searchsorted, a, 4, sorter=[0, 1, 2, 3, 5])
        # assert_raises(ValueError, np.searchsorted, a, 0, sorter=[-1, 0, 1, 2, 3])
        # assert_raises(ValueError, np.searchsorted, a, 0, sorter=[4, 0, -1, 2, 3])

    @xpassIfTorchDynamo_np  # (reason="self.storage_offset() must be divisible by 8")
    def test_searchsorted_with_sorter(self):
        a = np.random.rand(300)
        s = a.argsort()
        b = np.sort(a)
        k = np.linspace(0, 1, 20)
        assert_equal(b.searchsorted(k), a.searchsorted(k, sorter=s))

        a = np.array([0, 1, 2, 3, 5] * 20)
        s = a.argsort()
        k = [0, 1, 2, 3, 5]
        expected = [0, 20, 40, 60, 80]
        assert_equal(a.searchsorted(k, side="left", sorter=s), expected)
        expected = [20, 40, 60, 80, 100]
        assert_equal(a.searchsorted(k, side="right", sorter=s), expected)

        # Test searching unaligned array
        keys = np.arange(10)
        a = keys.copy()
        np.random.shuffle(s)
        s = a.argsort()
        aligned = np.empty(a.itemsize * a.size + 1, dtype="uint8")
        unaligned = aligned[1:].view(a.dtype)
        # Test searching unaligned array
        unaligned[:] = a
        b = unaligned.searchsorted(keys, "left", s)
        assert_equal(b, keys)
        b = unaligned.searchsorted(keys, "right", s)
        assert_equal(b, keys + 1)
        # Test searching for unaligned keys
        unaligned[:] = keys
        b = a.searchsorted(unaligned, "left", s)
        assert_equal(b, keys)
        b = a.searchsorted(unaligned, "right", s)
        assert_equal(b, keys + 1)

        # Test all type specific indirect binary search functions
        types = "".join((np.typecodes["AllInteger"], np.typecodes["AllFloat"], "?"))
        for dt in types:
            if dt == "?":
                a = np.array([1, 0], dtype=dt)
                # We want the sorter array to be of a type that is different
                # from np.intp in all platforms, to check for #4698
                s = np.array([1, 0], dtype=np.int16)
                out = np.array([1, 0])
            else:
                a = np.array([3, 4, 1, 2, 0], dtype=dt)
                # We want the sorter array to be of a type that is different
                # from np.intp in all platforms, to check for #4698
                s = np.array([4, 2, 3, 0, 1], dtype=np.int16)
                out = np.array([3, 4, 1, 2, 0], dtype=np.intp)
            b = a.searchsorted(a, "left", s)
            assert_equal(b, out)
            b = a.searchsorted(a, "right", s)
            assert_equal(b, out + 1)
            # Test empty array, use a fresh array to get warnings in
            # valgrind if access happens.
            e = np.ndarray(shape=0, buffer=b"", dtype=dt)
            b = e.searchsorted(a, "left", s[:0])
            assert_array_equal(b, np.zeros(len(a), dtype=np.intp))
            b = a.searchsorted(e, "left", s)
            assert_array_equal(b, np.zeros(0, dtype=np.intp))

        # Test non-contiguous sorter array
        a = np.array([3, 4, 1, 2, 0])
        srt = np.empty((10,), dtype=np.intp)
        srt[1::2] = -1
        srt[::2] = [4, 2, 3, 0, 1]
        s = srt[::2]
        out = np.array([3, 4, 1, 2, 0], dtype=np.intp)
        b = a.searchsorted(a, "left", s)
        assert_equal(b, out)
        b = a.searchsorted(a, "right", s)
        assert_equal(b, out + 1)

    @xpassIfTorchDynamo_np  # (reason="TODO argpartition")
    @parametrize("dtype", "efdFDBbhil?")
    def test_argpartition_out_of_range(self, dtype):
        # Test out of range values in kth raise an error, gh-5469
        d = np.arange(10).astype(dtype=dtype)
        assert_raises(ValueError, d.argpartition, 10)
        assert_raises(ValueError, d.argpartition, -11)

    @xpassIfTorchDynamo_np  # (reason="TODO partition")
    @parametrize("dtype", "efdFDBbhil?")
    def test_partition_out_of_range(self, dtype):
        # Test out of range values in kth raise an error, gh-5469
        d = np.arange(10).astype(dtype=dtype)
        assert_raises(ValueError, d.partition, 10)
        assert_raises(ValueError, d.partition, -11)

    @xpassIfTorchDynamo_np  # (reason="TODO argpartition")
    def test_argpartition_integer(self):
        # Test non-integer values in kth raise an error/
        d = np.arange(10)
        assert_raises(TypeError, d.argpartition, 9.0)
        # Test also for generic type argpartition, which uses sorting
        # and used to not bound check kth
        d_obj = np.arange(10, dtype=object)
        assert_raises(TypeError, d_obj.argpartition, 9.0)

    @xpassIfTorchDynamo_np  # (reason="TODO partition")
    def test_partition_integer(self):
        # Test out of range values in kth raise an error, gh-5469
        d = np.arange(10)
        assert_raises(TypeError, d.partition, 9.0)
        # Test also for generic type partition, which uses sorting
        # and used to not bound check kth
        d_obj = np.arange(10, dtype=object)
        assert_raises(TypeError, d_obj.partition, 9.0)

    @xpassIfTorchDynamo_np  # (reason="TODO partition")
    @parametrize("kth_dtype", "Bbhil")
    def test_partition_empty_array(self, kth_dtype):
        # check axis handling for multidimensional empty arrays
        kth = np.array(0, dtype=kth_dtype)[()]
        a = np.array([])
        a.shape = (3, 2, 1, 0)
        for axis in range(-a.ndim, a.ndim):
            msg = f"test empty array partition with axis={axis}"
            assert_equal(np.partition(a, kth, axis=axis), a, msg)
        msg = "test empty array partition with axis=None"
        assert_equal(np.partition(a, kth, axis=None), a.ravel(), msg)

    @xpassIfTorchDynamo_np  # (reason="TODO argpartition")
    @parametrize("kth_dtype", "Bbhil")
    def test_argpartition_empty_array(self, kth_dtype):
        # check axis handling for multidimensional empty arrays
        kth = np.array(0, dtype=kth_dtype)[()]
        a = np.array([])
        a.shape = (3, 2, 1, 0)
        for axis in range(-a.ndim, a.ndim):
            msg = f"test empty array argpartition with axis={axis}"
            assert_equal(
                np.partition(a, kth, axis=axis), np.zeros_like(a, dtype=np.intp), msg
            )
        msg = "test empty array argpartition with axis=None"
        assert_equal(
            np.partition(a, kth, axis=None),
            np.zeros_like(a.ravel(), dtype=np.intp),
            msg,
        )

    @xpassIfTorchDynamo_np  # (reason="TODO partition")
    def test_partition(self):
        d = np.arange(10)
        assert_raises(TypeError, np.partition, d, 2, kind=1)
        assert_raises(ValueError, np.partition, d, 2, kind="nonsense")
        assert_raises(ValueError, np.argpartition, d, 2, kind="nonsense")
        assert_raises(ValueError, d.partition, 2, axis=0, kind="nonsense")
        assert_raises(ValueError, d.argpartition, 2, axis=0, kind="nonsense")
        for k in ("introselect",):
            d = np.array([])
            assert_array_equal(np.partition(d, 0, kind=k), d)
            assert_array_equal(np.argpartition(d, 0, kind=k), d)
            d = np.ones(1)
            assert_array_equal(np.partition(d, 0, kind=k)[0], d)
            assert_array_equal(
                d[np.argpartition(d, 0, kind=k)], np.partition(d, 0, kind=k)
            )

            # kth not modified
            kth = np.array([30, 15, 5])
            okth = kth.copy()
            np.partition(np.arange(40), kth)
            assert_array_equal(kth, okth)

            for r in ([2, 1], [1, 2], [1, 1]):
                d = np.array(r)
                tgt = np.sort(d)
                assert_array_equal(np.partition(d, 0, kind=k)[0], tgt[0])
                assert_array_equal(np.partition(d, 1, kind=k)[1], tgt[1])
                assert_array_equal(
                    d[np.argpartition(d, 0, kind=k)], np.partition(d, 0, kind=k)
                )
                assert_array_equal(
                    d[np.argpartition(d, 1, kind=k)], np.partition(d, 1, kind=k)
                )
                for i in range(d.size):
                    d[i:].partition(0, kind=k)
                assert_array_equal(d, tgt)

            for r in (
                [3, 2, 1],
                [1, 2, 3],
                [2, 1, 3],
                [2, 3, 1],
                [1, 1, 1],
                [1, 2, 2],
                [2, 2, 1],
                [1, 2, 1],
            ):
                d = np.array(r)
                tgt = np.sort(d)
                assert_array_equal(np.partition(d, 0, kind=k)[0], tgt[0])
                assert_array_equal(np.partition(d, 1, kind=k)[1], tgt[1])
                assert_array_equal(np.partition(d, 2, kind=k)[2], tgt[2])
                assert_array_equal(
                    d[np.argpartition(d, 0, kind=k)], np.partition(d, 0, kind=k)
                )
                assert_array_equal(
                    d[np.argpartition(d, 1, kind=k)], np.partition(d, 1, kind=k)
                )
                assert_array_equal(
                    d[np.argpartition(d, 2, kind=k)], np.partition(d, 2, kind=k)
                )
                for i in range(d.size):
                    d[i:].partition(0, kind=k)
                assert_array_equal(d, tgt)

            d = np.ones(50)
            assert_array_equal(np.partition(d, 0, kind=k), d)
            assert_array_equal(
                d[np.argpartition(d, 0, kind=k)], np.partition(d, 0, kind=k)
            )

            # sorted
            d = np.arange(49)
            assert_equal(np.partition(d, 5, kind=k)[5], 5)
            assert_equal(np.partition(d, 15, kind=k)[15], 15)
            assert_array_equal(
                d[np.argpartition(d, 5, kind=k)], np.partition(d, 5, kind=k)
            )
            assert_array_equal(
                d[np.argpartition(d, 15, kind=k)], np.partition(d, 15, kind=k)
            )

            # rsorted
            d = np.arange(47)[::-1]
            assert_equal(np.partition(d, 6, kind=k)[6], 6)
            assert_equal(np.partition(d, 16, kind=k)[16], 16)
            assert_array_equal(
                d[np.argpartition(d, 6, kind=k)], np.partition(d, 6, kind=k)
            )
            assert_array_equal(
                d[np.argpartition(d, 16, kind=k)], np.partition(d, 16, kind=k)
            )

            assert_array_equal(np.partition(d, -6, kind=k), np.partition(d, 41, kind=k))
            assert_array_equal(
                np.partition(d, -16, kind=k), np.partition(d, 31, kind=k)
            )
            assert_array_equal(
                d[np.argpartition(d, -6, kind=k)], np.partition(d, 41, kind=k)
            )

            # median of 3 killer, O(n^2) on pure median 3 pivot quickselect
            # exercises the median of median of 5 code used to keep O(n)
            d = np.arange(1000000)
            x = np.roll(d, d.size // 2)
            mid = x.size // 2 + 1
            assert_equal(np.partition(x, mid)[mid], mid)
            d = np.arange(1000001)
            x = np.roll(d, d.size // 2 + 1)
            mid = x.size // 2 + 1
            assert_equal(np.partition(x, mid)[mid], mid)

            # max
            d = np.ones(10)
            d[1] = 4
            assert_equal(np.partition(d, (2, -1))[-1], 4)
            assert_equal(np.partition(d, (2, -1))[2], 1)
            assert_equal(d[np.argpartition(d, (2, -1))][-1], 4)
            assert_equal(d[np.argpartition(d, (2, -1))][2], 1)
            d[1] = np.nan
            assert_(np.isnan(d[np.argpartition(d, (2, -1))][-1]))
            assert_(np.isnan(np.partition(d, (2, -1))[-1]))

            # equal elements
            d = np.arange(47) % 7
            tgt = np.sort(np.arange(47) % 7)
            np.random.shuffle(d)
            for i in range(d.size):
                assert_equal(np.partition(d, i, kind=k)[i], tgt[i])
            assert_array_equal(
                d[np.argpartition(d, 6, kind=k)], np.partition(d, 6, kind=k)
            )
            assert_array_equal(
                d[np.argpartition(d, 16, kind=k)], np.partition(d, 16, kind=k)
            )
            for i in range(d.size):
                d[i:].partition(0, kind=k)
            assert_array_equal(d, tgt)

            d = np.array(
                [0, 1, 2, 3, 4, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 9]
            )
            kth = [0, 3, 19, 20]
            assert_equal(np.partition(d, kth, kind=k)[kth], (0, 3, 7, 7))
            assert_equal(d[np.argpartition(d, kth, kind=k)][kth], (0, 3, 7, 7))

            d = np.array([2, 1])
            d.partition(0, kind=k)
            assert_raises(ValueError, d.partition, 2)
            assert_raises(np.AxisError, d.partition, 3, axis=1)
            assert_raises(ValueError, np.partition, d, 2)
            assert_raises(np.AxisError, np.partition, d, 2, axis=1)
            assert_raises(ValueError, d.argpartition, 2)
            assert_raises(np.AxisError, d.argpartition, 3, axis=1)
            assert_raises(ValueError, np.argpartition, d, 2)
            assert_raises(np.AxisError, np.argpartition, d, 2, axis=1)
            d = np.arange(10).reshape((2, 5))
            d.partition(1, axis=0, kind=k)
            d.partition(4, axis=1, kind=k)
            np.partition(d, 1, axis=0, kind=k)
            np.partition(d, 4, axis=1, kind=k)
            np.partition(d, 1, axis=None, kind=k)
            np.partition(d, 9, axis=None, kind=k)
            d.argpartition(1, axis=0, kind=k)
            d.argpartition(4, axis=1, kind=k)
            np.argpartition(d, 1, axis=0, kind=k)
            np.argpartition(d, 4, axis=1, kind=k)
            np.argpartition(d, 1, axis=None, kind=k)
            np.argpartition(d, 9, axis=None, kind=k)
            assert_raises(ValueError, d.partition, 2, axis=0)
            assert_raises(ValueError, d.partition, 11, axis=1)
            assert_raises(TypeError, d.partition, 2, axis=None)
            assert_raises(ValueError, np.partition, d, 9, axis=1)
            assert_raises(ValueError, np.partition, d, 11, axis=None)
            assert_raises(ValueError, d.argpartition, 2, axis=0)
            assert_raises(ValueError, d.argpartition, 11, axis=1)
            assert_raises(ValueError, np.argpartition, d, 9, axis=1)
            assert_raises(ValueError, np.argpartition, d, 11, axis=None)

            td = [
                (dt, s) for dt in [np.int32, np.float32, np.complex64] for s in (9, 16)
            ]
            for dt, s in td:
                aae = assert_array_equal
                at = assert_

                d = np.arange(s, dtype=dt)
                np.random.shuffle(d)
                d1 = np.tile(np.arange(s, dtype=dt), (4, 1))
                map(np.random.shuffle, d1)
                d0 = np.transpose(d1)
                for i in range(d.size):
                    p = np.partition(d, i, kind=k)
                    assert_equal(p[i], i)
                    # all before are smaller
                    assert_array_less(p[:i], p[i])
                    # all after are larger
                    assert_array_less(p[i], p[i + 1 :])
                    aae(p, d[np.argpartition(d, i, kind=k)])

                    p = np.partition(d1, i, axis=1, kind=k)
                    aae(p[:, i], np.array([i] * d1.shape[0], dtype=dt))
                    # array_less does not seem to work right
                    at(
                        (p[:, :i].T <= p[:, i]).all(),
                        msg=f"{i:d}: {p[:, i]!r} <= {p[:, :i].T!r}",
                    )
                    at(
                        (p[:, i + 1 :].T > p[:, i]).all(),
                        msg=f"{i:d}: {p[:, i]!r} < {p[:, i + 1 :].T!r}",
                    )
                    aae(
                        p,
                        d1[
                            np.arange(d1.shape[0])[:, None],
                            np.argpartition(d1, i, axis=1, kind=k),
                        ],
                    )

                    p = np.partition(d0, i, axis=0, kind=k)
                    aae(p[i, :], np.array([i] * d1.shape[0], dtype=dt))
                    # array_less does not seem to work right
                    at(
                        (p[:i, :] <= p[i, :]).all(),
                        msg=f"{i:d}: {p[i, :]!r} <= {p[:i, :]!r}",
                    )
                    at(
                        (p[i + 1 :, :] > p[i, :]).all(),
                        msg=f"{i:d}: {p[i, :]!r} < {p[:, i + 1 :]!r}",
                    )
                    aae(
                        p,
                        d0[
                            np.argpartition(d0, i, axis=0, kind=k),
                            np.arange(d0.shape[1])[None, :],
                        ],
                    )

                    # check inplace
                    dc = d.copy()
                    dc.partition(i, kind=k)
                    assert_equal(dc, np.partition(d, i, kind=k))
                    dc = d0.copy()
                    dc.partition(i, axis=0, kind=k)
                    assert_equal(dc, np.partition(d0, i, axis=0, kind=k))
                    dc = d1.copy()
                    dc.partition(i, axis=1, kind=k)
                    assert_equal(dc, np.partition(d1, i, axis=1, kind=k))

    def assert_partitioned(self, d, kth):
        prev = 0
        for k in np.sort(kth):
            assert_array_less(d[prev:k], d[k], err_msg=f"kth {k:d}")
            assert_(
                (d[k:] >= d[k]).all(),
                msg=f"kth {k:d}, {d[k:]!r} not greater equal {d[k]:d}",
            )
            prev = k + 1

    @xpassIfTorchDynamo_np  # (reason="TODO partition")
    def test_partition_iterative(self):
        d = np.arange(17)
        kth = (0, 1, 2, 429, 231)
        assert_raises(ValueError, d.partition, kth)
        assert_raises(ValueError, d.argpartition, kth)
        d = np.arange(10).reshape((2, 5))
        assert_raises(ValueError, d.partition, kth, axis=0)
        assert_raises(ValueError, d.partition, kth, axis=1)
        assert_raises(ValueError, np.partition, d, kth, axis=1)
        assert_raises(ValueError, np.partition, d, kth, axis=None)

        d = np.array([3, 4, 2, 1])
        p = np.partition(d, (0, 3))
        self.assert_partitioned(p, (0, 3))
        self.assert_partitioned(d[np.argpartition(d, (0, 3))], (0, 3))

        assert_array_equal(p, np.partition(d, (-3, -1)))
        assert_array_equal(p, d[np.argpartition(d, (-3, -1))])

        d = np.arange(17)
        np.random.shuffle(d)
        d.partition(range(d.size))
        assert_array_equal(np.arange(17), d)
        np.random.shuffle(d)
        assert_array_equal(np.arange(17), d[d.argpartition(range(d.size))])

        # test unsorted kth
        d = np.arange(17)
        np.random.shuffle(d)
        keys = np.array([1, 3, 8, -2])
        np.random.shuffle(d)
        p = np.partition(d, keys)
        self.assert_partitioned(p, keys)
        p = d[np.argpartition(d, keys)]
        self.assert_partitioned(p, keys)
        np.random.shuffle(keys)
        assert_array_equal(np.partition(d, keys), p)
        assert_array_equal(d[np.argpartition(d, keys)], p)

        # equal kth
        d = np.arange(20)[::-1]
        self.assert_partitioned(np.partition(d, [5] * 4), [5])
        self.assert_partitioned(np.partition(d, [5] * 4 + [6, 13]), [5] * 4 + [6, 13])
        self.assert_partitioned(d[np.argpartition(d, [5] * 4)], [5])
        self.assert_partitioned(
            d[np.argpartition(d, [5] * 4 + [6, 13])], [5] * 4 + [6, 13]
        )

        d = np.arange(12)
        np.random.shuffle(d)
        d1 = np.tile(np.arange(12), (4, 1))
        map(np.random.shuffle, d1)
        d0 = np.transpose(d1)

        kth = (1, 6, 7, -1)
        p = np.partition(d1, kth, axis=1)
        pa = d1[np.arange(d1.shape[0])[:, None], d1.argpartition(kth, axis=1)]
        assert_array_equal(p, pa)
        for i in range(d1.shape[0]):
            self.assert_partitioned(p[i, :], kth)
        p = np.partition(d0, kth, axis=0)
        pa = d0[np.argpartition(d0, kth, axis=0), np.arange(d0.shape[1])[None, :]]
        assert_array_equal(p, pa)
        for i in range(d0.shape[1]):
            self.assert_partitioned(p[:, i], kth)

    @xpassIfTorchDynamo_np  # (reason="TODO partition")
    def test_partition_fuzz(self):
        # a few rounds of random data testing
        for j in range(10, 30):
            for i in range(1, j - 2):
                d = np.arange(j)
                np.random.shuffle(d)
                d = d % np.random.randint(2, 30)
                idx = np.random.randint(d.size)
                kth = [0, idx, i, i + 1]
                tgt = np.sort(d)[kth]
                assert_array_equal(
                    np.partition(d, kth)[kth],
                    tgt,
                    err_msg=f"data: {d!r}\n kth: {kth!r}",
                )

    @xpassIfTorchDynamo_np  # (reason="TODO partition")
    @parametrize("kth_dtype", "Bbhil")
    def test_argpartition_gh5524(self, kth_dtype):
        #  A test for functionality of argpartition on lists.
        kth = np.array(1, dtype=kth_dtype)[()]
        d = [6, 7, 3, 2, 9, 0]
        p = np.argpartition(d, kth)
        self.assert_partitioned(np.array(d)[p], [1])

    @xpassIfTorchDynamo_np  # (reason="TODO order='F'")
    def test_flatten(self):
        x0 = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
        x1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], np.int32)
        y0 = np.array([1, 2, 3, 4, 5, 6], np.int32)
        y0f = np.array([1, 4, 2, 5, 3, 6], np.int32)
        y1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], np.int32)
        y1f = np.array([1, 5, 3, 7, 2, 6, 4, 8], np.int32)
        assert_equal(x0.flatten(), y0)
        assert_equal(x0.flatten("F"), y0f)
        assert_equal(x0.flatten("F"), x0.T.flatten())
        assert_equal(x1.flatten(), y1)
        assert_equal(x1.flatten("F"), y1f)
        assert_equal(x1.flatten("F"), x1.T.flatten())

    @parametrize("func", (np.dot, np.matmul))
    def test_arr_mult(self, func):
        a = np.array([[1, 0], [0, 1]])
        b = np.array([[0, 1], [1, 0]])
        d = np.arange(24).reshape(4, 6)
        ddt = np.array(
            [
                [55, 145, 235, 325],
                [145, 451, 757, 1063],
                [235, 757, 1279, 1801],
                [325, 1063, 1801, 2539],
            ]
        )
        dtd = np.array(
            [
                [504, 540, 576, 612, 648, 684],
                [540, 580, 620, 660, 700, 740],
                [576, 620, 664, 708, 752, 796],
                [612, 660, 708, 756, 804, 852],
                [648, 700, 752, 804, 856, 908],
                [684, 740, 796, 852, 908, 964],
            ]
        )

        # gemm vs syrk optimizations
        for et in [np.float32, np.float64, np.complex64, np.complex128]:
            eaf = a.astype(et)
            assert_equal(func(eaf, eaf), eaf)
            assert_equal(func(eaf.T, eaf), eaf)
            assert_equal(func(eaf, eaf.T), eaf)
            assert_equal(func(eaf.T, eaf.T), eaf)
            assert_equal(func(eaf.T.copy(), eaf), eaf)
            assert_equal(func(eaf, eaf.T.copy()), eaf)
            assert_equal(func(eaf.T.copy(), eaf.T.copy()), eaf)

        # syrk validations
        for et in [np.float32, np.float64, np.complex64, np.complex128]:
            eaf = a.astype(et)
            ebf = b.astype(et)
            assert_equal(func(ebf, ebf), eaf)
            assert_equal(func(ebf.T, ebf), eaf)
            assert_equal(func(ebf, ebf.T), eaf)
            assert_equal(func(ebf.T, ebf.T), eaf)
        # syrk - different shape
        for et in [np.float32, np.float64, np.complex64, np.complex128]:
            edf = d.astype(et)
            eddtf = ddt.astype(et)
            edtdf = dtd.astype(et)
            assert_equal(func(edf, edf.T), eddtf)
            assert_equal(func(edf.T, edf), edtdf)

            assert_equal(
                func(edf[: edf.shape[0] // 2, :], edf[::2, :].T),
                func(edf[: edf.shape[0] // 2, :].copy(), edf[::2, :].T.copy()),
            )
            assert_equal(
                func(edf[::2, :], edf[: edf.shape[0] // 2, :].T),
                func(edf[::2, :].copy(), edf[: edf.shape[0] // 2, :].T.copy()),
            )

    @skip(reason="dot/matmul with negative strides")
    @parametrize("func", (np.dot, np.matmul))
    def test_arr_mult_2(self, func):
        # syrk - different shape, stride, and view validations
        for et in [np.float32, np.float64, np.complex64, np.complex128]:
            edf = d.astype(et)
            assert_equal(
                func(edf[::-1, :], edf.T), func(edf[::-1, :].copy(), edf.T.copy())
            )
            assert_equal(
                func(edf[:, ::-1], edf.T), func(edf[:, ::-1].copy(), edf.T.copy())
            )
            assert_equal(func(edf, edf[::-1, :].T), func(edf, edf[::-1, :].T.copy()))
            assert_equal(func(edf, edf[:, ::-1].T), func(edf, edf[:, ::-1].T.copy()))

    @parametrize("func", (np.dot, np.matmul))
    @parametrize("dtype", "ifdFD")
    def test_no_dgemv(self, func, dtype):
        # check vector arg for contiguous before gemv
        # gh-12156
        a = np.arange(8.0, dtype=dtype).reshape(2, 4)
        b = np.broadcast_to(1.0, (4, 1))
        ret1 = func(a, b)
        ret2 = func(a, b.copy())
        assert_equal(ret1, ret2)

        ret1 = func(b.T, a.T)
        ret2 = func(b.T.copy(), a.T)
        assert_equal(ret1, ret2)

    @skip(reason="__array_interface__")
    @parametrize("func", (np.dot, np.matmul))
    @parametrize("dtype", "ifdFD")
    def test_no_dgemv_2(self, func, dtype):
        # check for unaligned data
        dt = np.dtype(dtype)
        a = np.zeros(8 * dt.itemsize // 2 + 1, dtype="int16")[1:].view(dtype)
        a = a.reshape(2, 4)
        b = a[0]
        # make sure it is not aligned
        assert_(a.__array_interface__["data"][0] % dt.itemsize != 0)
        ret1 = func(a, b)
        ret2 = func(a.copy(), b.copy())
        assert_equal(ret1, ret2)

        ret1 = func(b.T, a.T)
        ret2 = func(b.T.copy(), a.T.copy())
        assert_equal(ret1, ret2)

    def test_dot(self):
        a = np.array([[1, 0], [0, 1]])
        b = np.array([[0, 1], [1, 0]])
        c = np.array([[9, 1], [1, -9]])
        # function versus methods
        assert_equal(np.dot(a, b), a.dot(b))
        assert_equal(np.dot(np.dot(a, b), c), a.dot(b).dot(c))

        # test passing in an output array
        c = np.zeros_like(a)
        a.dot(b, c)
        assert_equal(c, np.dot(a, b))

        # test keyword args
        c = np.zeros_like(a)
        a.dot(b=b, out=c)
        assert_equal(c, np.dot(a, b))

    @xpassIfTorchDynamo_np  # (reason="_aligned_zeros")
    def test_dot_out_mem_overlap(self):
        np.random.seed(1)

        # Test BLAS and non-BLAS code paths, including all dtypes
        # that dot() supports
        dtypes = [np.dtype(code) for code in np.typecodes["All"] if code not in "USVM"]
        for dtype in dtypes:
            a = np.random.rand(3, 3).astype(dtype)

            # Valid dot() output arrays must be aligned
            b = _aligned_zeros((3, 3), dtype=dtype)
            b[...] = np.random.rand(3, 3)

            y = np.dot(a, b)
            x = np.dot(a, b, out=b)
            assert_equal(x, y, err_msg=repr(dtype))

            # Check invalid output array
            assert_raises(ValueError, np.dot, a, b, out=b[::2])
            assert_raises(ValueError, np.dot, a, b, out=b.T)

    @xpassIfTorchDynamo_np  # (reason="TODO: overlapping memor in matmul")
    def test_matmul_out(self):
        # overlapping memory
        a = np.arange(18).reshape(2, 3, 3)
        b = np.matmul(a, a)
        c = np.matmul(a, a, out=a)
        assert_(c is a)
        assert_equal(c, b)
        a = np.arange(18).reshape(2, 3, 3)
        c = np.matmul(a, a, out=a[::-1, ...])
        assert_(c.base is a.base)
        assert_equal(c, b)

    def test_diagonal(self):
        a = np.arange(12).reshape((3, 4))
        assert_equal(a.diagonal(), [0, 5, 10])
        assert_equal(a.diagonal(0), [0, 5, 10])
        assert_equal(a.diagonal(1), [1, 6, 11])
        assert_equal(a.diagonal(-1), [4, 9])
        assert_raises(np.AxisError, a.diagonal, axis1=0, axis2=5)
        assert_raises(np.AxisError, a.diagonal, axis1=5, axis2=0)
        assert_raises(np.AxisError, a.diagonal, axis1=5, axis2=5)
        assert_raises((ValueError, RuntimeError), a.diagonal, axis1=1, axis2=1)

        b = np.arange(8).reshape((2, 2, 2))
        assert_equal(b.diagonal(), [[0, 6], [1, 7]])
        assert_equal(b.diagonal(0), [[0, 6], [1, 7]])
        assert_equal(b.diagonal(1), [[2], [3]])
        assert_equal(b.diagonal(-1), [[4], [5]])
        assert_raises((ValueError, RuntimeError), b.diagonal, axis1=0, axis2=0)
        assert_equal(b.diagonal(0, 1, 2), [[0, 3], [4, 7]])
        assert_equal(b.diagonal(0, 0, 1), [[0, 6], [1, 7]])
        assert_equal(b.diagonal(offset=1, axis1=0, axis2=2), [[1], [3]])
        # Order of axis argument doesn't matter:
        assert_equal(b.diagonal(0, 2, 1), [[0, 3], [4, 7]])

    @xfail  # (reason="no readonly views")
    def test_diagonal_view_notwriteable(self):
        a = np.eye(3).diagonal()
        assert_(not a.flags.writeable)
        assert_(not a.flags.owndata)

        a = np.diagonal(np.eye(3))
        assert_(not a.flags.writeable)
        assert_(not a.flags.owndata)

        a = np.diag(np.eye(3))
        assert_(not a.flags.writeable)
        assert_(not a.flags.owndata)

    def test_diagonal_memleak(self):
        # Regression test for a bug that crept in at one point
        a = np.zeros((100, 100))
        if HAS_REFCOUNT:
            assert_(sys.getrefcount(a) < 50)
        for _ in range(100):
            a.diagonal()
        if HAS_REFCOUNT:
            assert_(sys.getrefcount(a) < 50)

    def test_size_zero_memleak(self):
        # Regression test for issue 9615
        # Exercises a special-case code path for dot products of length
        # zero in cblasfuncs (making it is specific to floating dtypes).
        a = np.array([], dtype=np.float64)
        x = np.array(2.0)
        for _ in range(100):
            np.dot(a, a, out=x)
        if HAS_REFCOUNT:
            assert_(sys.getrefcount(x) < 50)

    def test_trace(self):
        a = np.arange(12).reshape((3, 4))
        assert_equal(a.trace(), 15)
        assert_equal(a.trace(0), 15)
        assert_equal(a.trace(1), 18)
        assert_equal(a.trace(-1), 13)

        b = np.arange(8).reshape((2, 2, 2))
        assert_equal(b.trace(), [6, 8])
        assert_equal(b.trace(0), [6, 8])
        assert_equal(b.trace(1), [2, 3])
        assert_equal(b.trace(-1), [4, 5])
        assert_equal(b.trace(0, 0, 1), [6, 8])
        assert_equal(b.trace(0, 0, 2), [5, 9])
        assert_equal(b.trace(0, 1, 2), [3, 11])
        assert_equal(b.trace(offset=1, axis1=0, axis2=2), [1, 3])

        out = np.array(1)
        ret = a.trace(out=out)
        assert ret is out

    def test_put(self):
        icodes = np.typecodes["AllInteger"]
        fcodes = np.typecodes["AllFloat"]
        for dt in icodes + fcodes:
            tgt = np.array([0, 1, 0, 3, 0, 5], dt

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 76 class(es): TestFlag, subclass, frominterface, MyArr, TestHash, TestAttributes, TestArrayConstruction, TestAssignment, C, bad_sequence, TestDtypedescr, TestZeroRank, TestScalarIndexing, TestCreation, x, Fail, Map, Point2, C, C

### Functions
This file defines 497 function(s): runstring, temppath, _aligned_zeros, setUp, test_writeable, test_writeable_any_base, __init__, test_writeable_from_readonly, test_writeable_from_buffer, test_writeable_pickle, test_warnonwrite, test_readonly_flag_protocols, test_otherflags, test_string_align, test_void_align, test_int, setUp, test_attributes, test_attributes_2, test_dtypeattr, test_stridesattr, make_array, test_set_stridesattr, make_array, set_strides, test_fill, test_fill_max_uint64, test_fill_struct_array, test_fill_readonly, test_array


## Key Components

The file contains 23625 words across 6989 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 250670 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
