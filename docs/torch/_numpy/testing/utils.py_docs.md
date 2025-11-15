# Documentation: `torch/_numpy/testing/utils.py`

## File Metadata

- **Path**: `torch/_numpy/testing/utils.py`
- **Size**: 77,296 bytes (75.48 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# mypy: ignore-errors

"""
Utility function to facilitate testing.

"""

import contextlib
import gc
import operator
import os
import platform
import pprint
import re
import shutil
import sys
import warnings
from functools import wraps
from io import StringIO
from tempfile import mkdtemp, mkstemp
from warnings import WarningMessage

import torch._numpy as np
from torch._numpy import arange, asarray as asanyarray, empty, float32, intp, ndarray


__all__ = [
    "assert_equal",
    "assert_almost_equal",
    "assert_approx_equal",
    "assert_array_equal",
    "assert_array_less",
    "assert_string_equal",
    "assert_",
    "assert_array_almost_equal",
    "build_err_msg",
    "decorate_methods",
    "print_assert_equal",
    "verbose",
    "assert_",
    "assert_array_almost_equal_nulp",
    "assert_raises_regex",
    "assert_array_max_ulp",
    "assert_warns",
    "assert_no_warnings",
    "assert_allclose",
    "IgnoreException",
    "clear_and_catch_warnings",
    "temppath",
    "tempdir",
    "IS_PYPY",
    "HAS_REFCOUNT",
    "IS_WASM",
    "suppress_warnings",
    "assert_array_compare",
    "assert_no_gc_cycles",
    "break_cycles",
    "IS_PYSTON",
]


verbose = 0

IS_WASM = platform.machine() in ["wasm32", "wasm64"]
IS_PYPY = sys.implementation.name == "pypy"
IS_PYSTON = hasattr(sys, "pyston_version_info")
HAS_REFCOUNT = getattr(sys, "getrefcount", None) is not None and not IS_PYSTON


def assert_(val, msg=""):
    """
    Assert that works in release mode.
    Accepts callable msg to allow deferring evaluation until failure.

    The Python built-in ``assert`` does not work when executing code in
    optimized mode (the ``-O`` flag) - no byte-code is generated for it.

    For documentation on usage, refer to the Python documentation.

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    if not val:
        try:
            smsg = msg()
        except TypeError:
            smsg = msg
        raise AssertionError(smsg)


def gisnan(x):
    return np.isnan(x)


def gisfinite(x):
    return np.isfinite(x)


def gisinf(x):
    return np.isinf(x)


def build_err_msg(
    arrays,
    err_msg,
    header="Items are not equal:",
    verbose=True,
    names=("ACTUAL", "DESIRED"),
    precision=8,
):
    msg = ["\n" + header]
    if err_msg:
        if err_msg.find("\n") == -1 and len(err_msg) < 79 - len(header):
            msg = [msg[0] + " " + err_msg]
        else:
            msg.append(err_msg)
    if verbose:
        for i, a in enumerate(arrays):
            if isinstance(a, ndarray):
                # precision argument is only needed if the objects are ndarrays
                # r_func = partial(array_repr, precision=precision)
                r_func = ndarray.__repr__
            else:
                r_func = repr

            try:
                r = r_func(a)
            except Exception as exc:
                r = f"[repr failed for <{type(a).__name__}>: {exc}]"
            if r.count("\n") > 3:
                r = "\n".join(r.splitlines()[:3])
                r += "..."
            msg.append(f" {names[i]}: {r}")
    return "\n".join(msg)


def assert_equal(actual, desired, err_msg="", verbose=True):
    """
    Raises an AssertionError if two objects are not equal.

    Given two objects (scalars, lists, tuples, dictionaries or numpy arrays),
    check that all elements of these objects are equal. An exception is raised
    at the first conflicting values.

    When one of `actual` and `desired` is a scalar and the other is array_like,
    the function checks that each element of the array_like object is equal to
    the scalar.

    This function handles NaN comparisons as if NaN was a "normal" number.
    That is, AssertionError is not raised if both objects have NaNs in the same
    positions.  This is in contrast to the IEEE standard on NaNs, which says
    that NaN compared to anything must return False.

    Parameters
    ----------
    actual : array_like
        The object to check.
    desired : array_like
        The expected object.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
        If actual and desired are not equal.

    Examples
    --------
    >>> np.testing.assert_equal([4, 5], [4, 6])
    Traceback (most recent call last):
        ...
    AssertionError:
    Items are not equal:
    item=1
     ACTUAL: 5
     DESIRED: 6

    The following comparison does not raise an exception.  There are NaNs
    in the inputs, but they are in the same positions.

    >>> np.testing.assert_equal(np.array([1.0, 2.0, np.nan]), [1, 2, np.nan])

    """
    __tracebackhide__ = True  # Hide traceback for py.test

    num_nones = sum([actual is None, desired is None])
    if num_nones == 1:
        raise AssertionError(f"Not equal: {actual} != {desired}")
    elif num_nones == 2:
        return True
    # else, carry on

    if isinstance(actual, np.DType) or isinstance(desired, np.DType):
        result = actual == desired
        if not result:
            raise AssertionError(f"Not equal: {actual} != {desired}")
        else:
            return True

    if isinstance(desired, str) and isinstance(actual, str):
        assert actual == desired
        return

    if isinstance(desired, dict):
        if not isinstance(actual, dict):
            raise AssertionError(repr(type(actual)))
        assert_equal(len(actual), len(desired), err_msg, verbose)
        for k in desired:
            if k not in actual:
                raise AssertionError(repr(k))
            assert_equal(actual[k], desired[k], f"key={k!r}\n{err_msg}", verbose)
        return
    if isinstance(desired, (list, tuple)) and isinstance(actual, (list, tuple)):
        assert_equal(len(actual), len(desired), err_msg, verbose)
        for k in range(len(desired)):
            assert_equal(actual[k], desired[k], f"item={k!r}\n{err_msg}", verbose)
        return

    from torch._numpy import imag, iscomplexobj, isscalar, ndarray, real, signbit

    if isinstance(actual, ndarray) or isinstance(desired, ndarray):
        return assert_array_equal(actual, desired, err_msg, verbose)
    msg = build_err_msg([actual, desired], err_msg, verbose=verbose)

    # Handle complex numbers: separate into real/imag to handle
    # nan/inf/negative zero correctly
    # XXX: catch ValueError for subclasses of ndarray where iscomplex fail
    try:
        usecomplex = iscomplexobj(actual) or iscomplexobj(desired)
    except (ValueError, TypeError):
        usecomplex = False

    if usecomplex:
        if iscomplexobj(actual):
            actualr = real(actual)
            actuali = imag(actual)
        else:
            actualr = actual
            actuali = 0
        if iscomplexobj(desired):
            desiredr = real(desired)
            desiredi = imag(desired)
        else:
            desiredr = desired
            desiredi = 0
        try:
            assert_equal(actualr, desiredr)
            assert_equal(actuali, desiredi)
        except AssertionError:
            raise AssertionError(msg)  # noqa: B904

    # isscalar test to check cases such as [np.nan] != np.nan
    if isscalar(desired) != isscalar(actual):
        raise AssertionError(msg)

    # Inf/nan/negative zero handling
    try:
        isdesnan = gisnan(desired)
        isactnan = gisnan(actual)
        if isdesnan and isactnan:
            return  # both nan, so equal

        if desired == 0 and actual == 0:
            if not signbit(desired) == signbit(actual):
                raise AssertionError(msg)

    except (TypeError, ValueError, NotImplementedError):
        pass

    try:
        # Explicitly use __eq__ for comparison, gh-2552
        if not (desired == actual):
            raise AssertionError(msg)

    except (DeprecationWarning, FutureWarning) as e:
        # this handles the case when the two types are not even comparable
        if "elementwise == comparison" in e.args[0]:
            raise AssertionError(msg)  # noqa: B904
        else:
            raise


def print_assert_equal(test_string, actual, desired):
    """
    Test if two objects are equal, and print an error message if test fails.

    The test is performed with ``actual == desired``.

    Parameters
    ----------
    test_string : str
        The message supplied to AssertionError.
    actual : object
        The object to test for equality against `desired`.
    desired : object
        The expected result.

    Examples
    --------
    >>> np.testing.print_assert_equal(
    ...     "Test XYZ of func xyz", [0, 1], [0, 1]
    ... )  # doctest: +SKIP
    >>> np.testing.print_assert_equal(
    ...     "Test XYZ of func xyz", [0, 1], [0, 2]
    ... )  # doctest: +SKIP
    Traceback (most recent call last):
    ...
    AssertionError: Test XYZ of func xyz failed
    ACTUAL:
    [0, 1]
    DESIRED:
    [0, 2]

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    import pprint

    if actual != desired:
        msg = StringIO()
        msg.write(test_string)
        msg.write(" failed\nACTUAL: \n")
        pprint.pprint(actual, msg)
        msg.write("DESIRED: \n")
        pprint.pprint(desired, msg)
        raise AssertionError(msg.getvalue())


def assert_almost_equal(actual, desired, decimal=7, err_msg="", verbose=True):
    """
    Raises an AssertionError if two items are not equal up to desired
    precision.

    .. note:: It is recommended to use one of `assert_allclose`,
              `assert_array_almost_equal_nulp` or `assert_array_max_ulp`
              instead of this function for more consistent floating point
              comparisons.

    The test verifies that the elements of `actual` and `desired` satisfy.

        ``abs(desired-actual) < float64(1.5 * 10**(-decimal))``

    That is a looser test than originally documented, but agrees with what the
    actual implementation in `assert_array_almost_equal` did up to rounding
    vagaries. An exception is raised at conflicting values. For ndarrays this
    delegates to assert_array_almost_equal

    Parameters
    ----------
    actual : array_like
        The object to check.
    desired : array_like
        The expected object.
    decimal : int, optional
        Desired precision, default is 7.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
      If actual and desired are not equal up to specified precision.

    See Also
    --------
    assert_allclose: Compare two array_like objects for equality with desired
                     relative and/or absolute precision.
    assert_array_almost_equal_nulp, assert_array_max_ulp, assert_equal

    Examples
    --------
    >>> from torch._numpy.testing import assert_almost_equal
    >>> assert_almost_equal(2.3333333333333, 2.33333334)
    >>> assert_almost_equal(2.3333333333333, 2.33333334, decimal=10)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not almost equal to 10 decimals
     ACTUAL: 2.3333333333333
     DESIRED: 2.33333334

    >>> assert_almost_equal(
    ...     np.array([1.0, 2.3333333333333]), np.array([1.0, 2.33333334]), decimal=9
    ... )
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not almost equal to 9 decimals
    <BLANKLINE>
    Mismatched elements: 1 / 2 (50%)
    Max absolute difference: 6.666699636781459e-09
    Max relative difference: 2.8571569790287484e-09
     x: torch.ndarray([1.0000, 2.3333], dtype=float64)
     y: torch.ndarray([1.0000, 2.3333], dtype=float64)

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    from torch._numpy import imag, iscomplexobj, ndarray, real

    # Handle complex numbers: separate into real/imag to handle
    # nan/inf/negative zero correctly
    # XXX: catch ValueError for subclasses of ndarray where iscomplex fail
    try:
        usecomplex = iscomplexobj(actual) or iscomplexobj(desired)
    except ValueError:
        usecomplex = False

    def _build_err_msg():
        header = f"Arrays are not almost equal to {decimal:d} decimals"
        return build_err_msg([actual, desired], err_msg, verbose=verbose, header=header)

    if usecomplex:
        if iscomplexobj(actual):
            actualr = real(actual)
            actuali = imag(actual)
        else:
            actualr = actual
            actuali = 0
        if iscomplexobj(desired):
            desiredr = real(desired)
            desiredi = imag(desired)
        else:
            desiredr = desired
            desiredi = 0
        try:
            assert_almost_equal(actualr, desiredr, decimal=decimal)
            assert_almost_equal(actuali, desiredi, decimal=decimal)
        except AssertionError:
            raise AssertionError(_build_err_msg())  # noqa: B904

    if isinstance(actual, (ndarray, tuple, list)) or isinstance(
        desired, (ndarray, tuple, list)
    ):
        return assert_array_almost_equal(actual, desired, decimal, err_msg)
    try:
        # If one of desired/actual is not finite, handle it specially here:
        # check that both are nan if any is a nan, and test for equality
        # otherwise
        if not (gisfinite(desired) and gisfinite(actual)):
            if gisnan(desired) or gisnan(actual):
                if not (gisnan(desired) and gisnan(actual)):
                    raise AssertionError(_build_err_msg())
            else:
                if not desired == actual:
                    raise AssertionError(_build_err_msg())
            return
    except (NotImplementedError, TypeError):
        pass
    if abs(desired - actual) >= np.float64(1.5 * 10.0 ** (-decimal)):
        raise AssertionError(_build_err_msg())


def assert_approx_equal(actual, desired, significant=7, err_msg="", verbose=True):
    """
    Raises an AssertionError if two items are not equal up to significant
    digits.

    .. note:: It is recommended to use one of `assert_allclose`,
              `assert_array_almost_equal_nulp` or `assert_array_max_ulp`
              instead of this function for more consistent floating point
              comparisons.

    Given two numbers, check that they are approximately equal.
    Approximately equal is defined as the number of significant digits
    that agree.

    Parameters
    ----------
    actual : scalar
        The object to check.
    desired : scalar
        The expected object.
    significant : int, optional
        Desired precision, default is 7.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
      If actual and desired are not equal up to specified precision.

    See Also
    --------
    assert_allclose: Compare two array_like objects for equality with desired
                     relative and/or absolute precision.
    assert_array_almost_equal_nulp, assert_array_max_ulp, assert_equal

    Examples
    --------
    >>> np.testing.assert_approx_equal(
    ...     0.12345677777777e-20, 0.1234567e-20
    ... )  # doctest: +SKIP
    >>> np.testing.assert_approx_equal(
    ...     0.12345670e-20,
    ...     0.12345671e-20,  # doctest: +SKIP
    ...     significant=8,
    ... )
    >>> np.testing.assert_approx_equal(
    ...     0.12345670e-20,
    ...     0.12345672e-20,  # doctest: +SKIP
    ...     significant=8,
    ... )
    Traceback (most recent call last):
        ...
    AssertionError:
    Items are not equal to 8 significant digits:
     ACTUAL: 1.234567e-21
     DESIRED: 1.2345672e-21

    the evaluated condition that raises the exception is

    >>> abs(0.12345670e-20 / 1e-21 - 0.12345672e-20 / 1e-21) >= 10 ** -(8 - 1)
    True

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    import numpy as np

    (actual, desired) = map(float, (actual, desired))
    if desired == actual:
        return
    # Normalized the numbers to be in range (-10.0,10.0)
    # scale = float(pow(10,math.floor(math.log10(0.5*(abs(desired)+abs(actual))))))
    scale = 0.5 * (np.abs(desired) + np.abs(actual))
    scale = np.power(10, np.floor(np.log10(scale)))
    try:
        sc_desired = desired / scale
    except ZeroDivisionError:
        sc_desired = 0.0
    try:
        sc_actual = actual / scale
    except ZeroDivisionError:
        sc_actual = 0.0
    msg = build_err_msg(
        [actual, desired],
        err_msg,
        header=f"Items are not equal to {significant:d} significant digits:",
        verbose=verbose,
    )
    try:
        # If one of desired/actual is not finite, handle it specially here:
        # check that both are nan if any is a nan, and test for equality
        # otherwise
        if not (gisfinite(desired) and gisfinite(actual)):
            if gisnan(desired) or gisnan(actual):
                if not (gisnan(desired) and gisnan(actual)):
                    raise AssertionError(msg)
            else:
                if not desired == actual:
                    raise AssertionError(msg)
            return
    except (TypeError, NotImplementedError):
        pass
    if np.abs(sc_desired - sc_actual) >= np.power(10.0, -(significant - 1)):
        raise AssertionError(msg)


def assert_array_compare(
    comparison,
    x,
    y,
    err_msg="",
    verbose=True,
    header="",
    precision=6,
    equal_nan=True,
    equal_inf=True,
    *,
    strict=False,
):
    __tracebackhide__ = True  # Hide traceback for py.test
    from torch._numpy import all, array, asarray, bool_, inf, isnan, max

    x = asarray(x)
    y = asarray(y)

    def array2string(a):
        return str(a)

    # original array for output formatting
    ox, oy = x, y

    def func_assert_same_pos(x, y, func=isnan, hasval="nan"):
        """Handling nan/inf.

        Combine results of running func on x and y, checking that they are True
        at the same locations.

        """
        __tracebackhide__ = True  # Hide traceback for py.test
        x_id = func(x)
        y_id = func(y)
        # We include work-arounds here to handle three types of slightly
        # pathological ndarray subclasses:
        # (1) all() on `masked` array scalars can return masked arrays, so we
        #     use != True
        # (2) __eq__ on some ndarray subclasses returns Python booleans
        #     instead of element-wise comparisons, so we cast to bool_() and
        #     use isinstance(..., bool) checks
        # (3) subclasses with bare-bones __array_function__ implementations may
        #     not implement np.all(), so favor using the .all() method
        # We are not committed to supporting such subclasses, but it's nice to
        # support them if possible.
        if (x_id == y_id).all().item() is not True:
            msg = build_err_msg(
                [x, y],
                err_msg + f"\nx and y {hasval} location mismatch:",
                verbose=verbose,
                header=header,
                names=("x", "y"),
                precision=precision,
            )
            raise AssertionError(msg)
        # If there is a scalar, then here we know the array has the same
        # flag as it everywhere, so we should return the scalar flag.
        if isinstance(x_id, bool) or x_id.ndim == 0:
            return bool_(x_id)
        elif isinstance(y_id, bool) or y_id.ndim == 0:
            return bool_(y_id)
        else:
            return y_id

    try:
        if strict:
            cond = x.shape == y.shape and x.dtype == y.dtype
        else:
            cond = (x.shape == () or y.shape == ()) or x.shape == y.shape
        if not cond:
            if x.shape != y.shape:
                reason = f"\n(shapes {x.shape}, {y.shape} mismatch)"
            else:
                reason = f"\n(dtypes {x.dtype}, {y.dtype} mismatch)"
            msg = build_err_msg(
                [x, y],
                err_msg + reason,
                verbose=verbose,
                header=header,
                names=("x", "y"),
                precision=precision,
            )
            raise AssertionError(msg)

        flagged = bool_(False)

        if equal_nan:
            flagged = func_assert_same_pos(x, y, func=isnan, hasval="nan")

        if equal_inf:
            flagged |= func_assert_same_pos(
                x, y, func=lambda xy: xy == +inf, hasval="+inf"
            )
            flagged |= func_assert_same_pos(
                x, y, func=lambda xy: xy == -inf, hasval="-inf"
            )

        if flagged.ndim > 0:
            x, y = x[~flagged], y[~flagged]
            # Only do the comparison if actual values are left
            if x.size == 0:
                return
        elif flagged:
            # no sense doing comparison if everything is flagged.
            return

        val = comparison(x, y)

        if isinstance(val, bool):
            cond = val
            reduced = array([val])
        else:
            reduced = val.ravel()
            cond = reduced.all()

        # The below comparison is a hack to ensure that fully masked
        # results, for which val.ravel().all() returns np.ma.masked,
        # do not trigger a failure (np.ma.masked != True evaluates as
        # np.ma.masked, which is falsy).
        if not cond:
            n_mismatch = reduced.size - int(reduced.sum(dtype=intp))
            n_elements = flagged.size if flagged.ndim != 0 else reduced.size
            percent_mismatch = 100 * n_mismatch / n_elements
            remarks = [
                f"Mismatched elements: {n_mismatch} / {n_elements} ({percent_mismatch:.3g}%)"
            ]

            # with errstate(all='ignore'):
            # ignore errors for non-numeric types
            with contextlib.suppress(TypeError, RuntimeError):
                error = abs(x - y)
                if np.issubdtype(x.dtype, np.unsignedinteger):
                    error2 = abs(y - x)
                    np.minimum(error, error2, out=error)
                max_abs_error = max(error)
                remarks.append(
                    "Max absolute difference: " + array2string(max_abs_error.item())
                )

                # note: this definition of relative error matches that one
                # used by assert_allclose (found in np.isclose)
                # Filter values where the divisor would be zero
                nonzero = bool_(y != 0)
                if all(~nonzero):
                    max_rel_error = array(inf)
                else:
                    max_rel_error = max(error[nonzero] / abs(y[nonzero]))
                remarks.append(
                    "Max relative difference: " + array2string(max_rel_error.item())
                )

            err_msg += "\n" + "\n".join(remarks)
            msg = build_err_msg(
                [ox, oy],
                err_msg,
                verbose=verbose,
                header=header,
                names=("x", "y"),
                precision=precision,
            )
            raise AssertionError(msg)
    except ValueError:
        import traceback

        efmt = traceback.format_exc()
        header = f"error during assertion:\n\n{efmt}\n\n{header}"

        msg = build_err_msg(
            [x, y],
            err_msg,
            verbose=verbose,
            header=header,
            names=("x", "y"),
            precision=precision,
        )
        raise ValueError(msg)  # noqa: B904


def assert_array_equal(x, y, err_msg="", verbose=True, *, strict=False):
    """
    Raises an AssertionError if two array_like objects are not equal.

    Given two array_like objects, check that the shape is equal and all
    elements of these objects are equal (but see the Notes for the special
    handling of a scalar). An exception is raised at shape mismatch or
    conflicting values. In contrast to the standard usage in numpy, NaNs
    are compared like numbers, no assertion is raised if both objects have
    NaNs in the same positions.

    The usual caution for verifying equality with floating point numbers is
    advised.

    Parameters
    ----------
    x : array_like
        The actual object to check.
    y : array_like
        The desired, expected object.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.
    strict : bool, optional
        If True, raise an AssertionError when either the shape or the data
        type of the array_like objects does not match. The special
        handling for scalars mentioned in the Notes section is disabled.

    Raises
    ------
    AssertionError
        If actual and desired objects are not equal.

    See Also
    --------
    assert_allclose: Compare two array_like objects for equality with desired
                     relative and/or absolute precision.
    assert_array_almost_equal_nulp, assert_array_max_ulp, assert_equal

    Notes
    -----
    When one of `x` and `y` is a scalar and the other is array_like, the
    function checks that each element of the array_like object is equal to
    the scalar. This behaviour can be disabled with the `strict` parameter.

    Examples
    --------
    The first assert does not raise an exception:

    >>> np.testing.assert_array_equal(
    ...     [1.0, 2.33333, np.nan], [np.exp(0), 2.33333, np.nan]
    ... )

    Use `assert_allclose` or one of the nulp (number of floating point values)
    functions for these cases instead:

    >>> np.testing.assert_allclose(
    ...     [1.0, np.pi, np.nan], [1, np.sqrt(np.pi) ** 2, np.nan], rtol=1e-10, atol=0
    ... )

    As mentioned in the Notes section, `assert_array_equal` has special
    handling for scalars. Here the test checks that each value in `x` is 3:

    >>> x = np.full((2, 5), fill_value=3)
    >>> np.testing.assert_array_equal(x, 3)

    Use `strict` to raise an AssertionError when comparing a scalar with an
    array:

    >>> np.testing.assert_array_equal(x, 3, strict=True)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not equal
    <BLANKLINE>
    (shapes (2, 5), () mismatch)
     x: torch.ndarray([[3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3]])
     y: torch.ndarray(3)

    The `strict` parameter also ensures that the array data types match:

    >>> x = np.array([2, 2, 2])
    >>> y = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    >>> np.testing.assert_array_equal(x, y, strict=True)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not equal
    <BLANKLINE>
    (dtypes dtype("int64"), dtype("float32") mismatch)
     x: torch.ndarray([2, 2, 2])
     y: torch.ndarray([2., 2., 2.])
    """
    __tracebackhide__ = True  # Hide traceback for py.test
    assert_array_compare(
        operator.__eq__,
        x,
        y,
        err_msg=err_msg,
        verbose=verbose,
        header="Arrays are not equal",
        strict=strict,
    )


def assert_array_almost_equal(x, y, decimal=6, err_msg="", verbose=True):
    """
    Raises an AssertionError if two objects are not equal up to desired
    precision.

    .. note:: It is recommended to use one of `assert_allclose`,
              `assert_array_almost_equal_nulp` or `assert_array_max_ulp`
              instead of this function for more consistent floating point
              comparisons.

    The test verifies identical shapes and that the elements of ``actual`` and
    ``desired`` satisfy.

        ``abs(desired-actual) < 1.5 * 10**(-decimal)``

    That is a looser test than originally documented, but agrees with what the
    actual implementation did up to rounding vagaries. An exception is raised
    at shape mismatch or conflicting values. In contrast to the standard usage
    in numpy, NaNs are compared like numbers, no assertion is raised if both
    objects have NaNs in the same positions.

    Parameters
    ----------
    x : array_like
        The actual object to check.
    y : array_like
        The desired, expected object.
    decimal : int, optional
        Desired precision, default is 6.
    err_msg : str, optional
      The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
        If actual and desired are not equal up to specified precision.

    See Also
    --------
    assert_allclose: Compare two array_like objects for equality with desired
                     relative and/or absolute precision.
    assert_array_almost_equal_nulp, assert_array_max_ulp, assert_equal

    Examples
    --------
    the first assert does not raise an exception

    >>> np.testing.assert_array_almost_equal([1.0, 2.333, np.nan], [1.0, 2.333, np.nan])

    >>> np.testing.assert_array_almost_equal(
    ...     [1.0, 2.33333, np.nan], [1.0, 2.33339, np.nan], decimal=5
    ... )
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not almost equal to 5 decimals
    <BLANKLINE>
    Mismatched elements: 1 / 3 (33.3%)
    Max absolute difference: 5.999999999994898e-05
    Max relative difference: 2.5713661239633743e-05
     x: torch.ndarray([1.0000, 2.3333,    nan], dtype=float64)
     y: torch.ndarray([1.0000, 2.3334,    nan], dtype=float64)

    >>> np.testing.assert_array_almost_equal(
    ...     [1.0, 2.33333, np.nan], [1.0, 2.33333, 5], decimal=5
    ... )
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not almost equal to 5 decimals
    <BLANKLINE>
    x and y nan location mismatch:
     x: torch.ndarray([1.0000, 2.3333,    nan], dtype=float64)
     y: torch.ndarray([1.0000, 2.3333, 5.0000], dtype=float64)

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    from torch._numpy import any as npany, float_, issubdtype, number, result_type

    def compare(x, y):
        try:
            if npany(gisinf(x)) or npany(gisinf(y)):
                xinfid = gisinf(x)
                yinfid = gisinf(y)
                if not (xinfid == yinfid).all():
                    return False
                # if one item, x and y is +- inf
                if x.size == y.size == 1:
                    return x == y
                x = x[~xinfid]
                y = y[~yinfid]
        except (TypeError, NotImplementedError):
            pass

        # make sure y is an inexact type to avoid abs(MIN_INT); will cause
        # casting of x later.
        dtype = result_type(y, 1.0)
        y = asanyarray(y, dtype)
        z = abs(x - y)

        if not issubdtype(z.dtype, number):
            z = z.astype(float_)  # handle object arrays

        return z < 1.5 * 10.0 ** (-decimal)

    assert_array_compare(
        compare,
        x,
        y,
        err_msg=err_msg,
        verbose=verbose,
        header=f"Arrays are not almost equal to {decimal:d} decimals",
        precision=decimal,
    )


def assert_array_less(x, y, err_msg="", verbose=True):
    """
    Raises an AssertionError if two array_like objects are not ordered by less
    than.

    Given two array_like objects, check that the shape is equal and all
    elements of the first object are strictly smaller than those of the
    second object. An exception is raised at shape mismatch or incorrectly
    ordered values. Shape mismatch does not raise if an object has zero
    dimension. In contrast to the standard usage in numpy, NaNs are
    compared, no assertion is raised if both objects have NaNs in the same
    positions.



    Parameters
    ----------
    x : array_like
      The smaller object to check.
    y : array_like
      The larger object to compare.
    err_msg : string
      The error message to be printed in case of failure.
    verbose : bool
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
      If actual and desired objects are not equal.

    See Also
    --------
    assert_array_equal: tests objects for equality
    assert_array_almost_equal: test objects for equality up to precision



    Examples
    --------
    >>> np.testing.assert_array_less([1.0, 1.0, np.nan], [1.1, 2.0, np.nan])
    >>> np.testing.assert_array_less([1.0, 1.0, np.nan], [1, 2.0, np.nan])
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not less-ordered
    <BLANKLINE>
    Mismatched elements: 1 / 3 (33.3%)
    Max absolute difference: 1.0
    Max relative difference: 0.5
     x: torch.ndarray([1.,  1., nan], dtype=float64)
     y: torch.ndarray([1.,  2., nan], dtype=float64)

    >>> np.testing.assert_array_less([1.0, 4.0], 3)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not less-ordered
    <BLANKLINE>
    Mismatched elements: 1 / 2 (50%)
    Max absolute difference: 2.0
    Max relative difference: 0.6666666666666666
     x: torch.ndarray([1., 4.], dtype=float64)
     y: torch.ndarray(3)

    >>> np.testing.assert_array_less([1.0, 2.0, 3.0], [4])
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not less-ordered
    <BLANKLINE>
    (shapes (3,), (1,) mismatch)
     x: torch.ndarray([1., 2., 3.], dtype=float64)
     y: torch.ndarray([4])

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    assert_array_compare(
        operator.__lt__,
        x,
        y,
        err_msg=err_msg,
        verbose=verbose,
        header="Arrays are not less-ordered",
        equal_inf=False,
    )


def assert_string_equal(actual, desired):
    """
    Test if two strings are equal.

    If the given strings are equal, `assert_string_equal` does nothing.
    If they are not equal, an AssertionError is raised, and the diff
    between the strings is shown.

    Parameters
    ----------
    actual : str
        The string to test for equality against the expected string.
    desired : str
        The expected string.

    Examples
    --------
    >>> np.testing.assert_string_equal("abc", "abc")  # doctest: +SKIP
    >>> np.testing.assert_string_equal("abc", "abcd")  # doctest: +SKIP
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ...
    AssertionError: Differences in strings:
    - abc+ abcd?    +

    """
    # delay import of difflib to reduce startup time
    __tracebackhide__ = True  # Hide traceback for py.test
    import difflib

    if not isinstance(actual, str):
        raise AssertionError(repr(type(actual)))
    if not isinstance(desired, str):
        raise AssertionError(repr(type(desired)))
    if desired == actual:
        return

    diff = list(
        difflib.Differ().compare(actual.splitlines(True), desired.splitlines(True))
    )
    diff_list = []
    while diff:
        d1 = diff.pop(0)
        if d1.startswith("  "):
            continue
        if d1.startswith("- "):
            l = [d1]
            d2 = diff.pop(0)
            if d2.startswith("? "):
                l.append(d2)
                d2 = diff.pop(0)
            if not d2.startswith("+ "):
                raise AssertionError(repr(d2))
            l.append(d2)
            if diff:
                d3 = diff.pop(0)
                if d3.startswith("? "):
                    l.append(d3)
                else:
                    diff.insert(0, d3)
            if d2[2:] == d1[2:]:
                continue
            diff_list.extend(l)
            continue
        raise AssertionError(repr(d1))
    if not diff_list:
        return
    msg = f"Differences in strings:\n{''.join(diff_list).rstrip()}"
    if actual != desired:
        raise AssertionError(msg)


import unittest


class _Dummy(unittest.TestCase):
    def nop(self):
        pass


_d = _Dummy("nop")


def assert_raises_regex(exception_class, expected_regexp, *args, **kwargs):
    """
    assert_raises_regex(exception_class, expected_regexp, callable, *args,
                        **kwargs)
    assert_raises_regex(exception_class, expected_regexp)

    Fail unless an exception of class exception_class and with message that
    matches expected_regexp is thrown by callable when invoked with arguments
    args and keyword arguments kwargs.

    Alternatively, can be used as a context manager like `assert_raises`.

    Notes
    -----
    .. versionadded:: 1.9.0

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    return _d.assertRaisesRegex(exception_class, expected_regexp, *args, **kwargs)


def decorate_methods(cls, decorator, testmatch=None):
    """
    Apply a decorator to all methods in a class matching a regular expression.

    The given decorator is applied to all public methods of `cls` that are
    matched by the regular expression `testmatch`
    (``testmatch.search(methodname)``). Methods that are private, i.e. start
    with an underscore, are ignored.

    Parameters
    ----------
    cls : class
        Class whose methods to decorate.
    decorator : function
        Decorator to apply to methods
    testmatch : compiled regexp or str, optional
        The regular expression. Default value is None, in which case the
        nose default (``re.compile(r'(?:^|[\\b_\\.%s-])[Tt]est' % os.sep)``)
        is used.
        If `testmatch` is a string, it is compiled to a regular expression
        first.

    """
    if testmatch is None:
        testmatch = re.compile(rf"(?:^|[\\b_\\.{os.sep}-])[Tt]est")
    else:
        testmatch = re.compile(testmatch)
    cls_attr = cls.__dict__

    # delayed import to reduce startup time
    from inspect import isfunction

    methods = [_m for _m in cls_attr.values() if isfunction(_m)]
    for function in methods:
        try:
            if hasattr(function, "compat_func_name"):
                funcname = function.compat_func_name
            else:
                funcname = function.__name__
        except AttributeError:
            # not a function
            continue
        if testmatch.search(funcname) and not funcname.startswith("_"):
            setattr(cls, funcname, decorator(function))
    return


def _assert_valid_refcount(op):
    """
    Check that ufuncs don't mishandle refcount of object `1`.
    Used in a few regression tests.
    """
    if not HAS_REFCOUNT:
        return True

    import gc

    import numpy as np

    b = np.arange(100 * 100).reshape(100, 100)
    c = b
    i = 1

    gc.disable()
    try:
        rc = sys.getrefcount(i)
        for _ in range(15):
            d = op(b, c)
        assert_(sys.getrefcount(i) >= rc)
    finally:
        gc.enable()
    del d  # for pyflakes


def assert_allclose(
    actual,
    desired,
    rtol=1e-7,
    atol=0,
    equal_nan=True,
    err_msg="",
    verbose=True,
    check_dtype=False,
):
    """
    Raises an AssertionError if two objects are not equal up to desired
    tolerance.

    Given two array_like objects, check that their shapes and all elements
    are equal (but see the Notes for the special handling of a scalar). An
    exception is raised if the shapes mismatch or any values conflict. In
    contrast to the standard usage in numpy, NaNs are compared like numbers,
    no assertion is raised if both objects have NaNs in the same positions.

    The test is equivalent to ``allclose(actual, desired, rtol, atol)`` (note
    that ``allclose`` has different default values). It compares the difference
    between `actual` and `desired` to ``atol + rtol * abs(desired)``.

    .. versionadded:: 1.5.0

    Parameters
    ----------
    actual : array_like
        Array obtained.
    desired : array_like
        Array desired.
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.
    equal_nan : bool, optional.
        If True, NaNs will compare equal.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
        If actual and desired are not equal up to specified precision.

    See Also
    --------
    assert_array_almost_equal_nulp, assert_array_max_ulp

    Notes
    -----
    When one of `actual` and `desired` is a scalar and the other is
    array_like, the function checks that each element of the array_like
    object is equal to the scalar.

    Examples
    --------
    >>> x = [1e-5, 1e-3, 1e-1]
    >>> y = np.arccos(np.cos(x))
    >>> np.testing.assert_allclose(x, y, rtol=1e-5, atol=0)

    """
    __tracebackhide__ = True  # Hide traceback for py.test

    def compare(x, y):
        return np.isclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)

    actual, desired = asanyarray(actual), asanyarray(desired)
    header = f"Not equal to tolerance rtol={rtol:g}, atol={atol:g}"

    if check_dtype:
        assert actual.dtype == desired.dtype

    assert_array_compare(
        compare,
        actual,
        desired,
        err_msg=str(err_msg),
        verbose=verbose,
        header=header,
        equal_nan=equal_nan,
    )


def assert_array_almost_equal_nulp(x, y, nulp=1):
    """
    Compare two arrays relatively to their spacing.

    This is a relatively robust method to compare two arrays whose amplitude
    is variable.

    Parameters
    ----------
    x, y : array_like
        Input arrays.
    nulp : int, optional
        The maximum number of unit in the last place for tolerance (see Notes).
        Default is 1.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the spacing between `x` and `y` for one or more elements is larger
        than `nulp`.

    See Also
    --------
    assert_array_max_ulp : Check that all items of arrays differ in at most
        N Units in the Last Place.
    spacing : Return the distance between x and the nearest adjacent number.

    Notes
    -----
    An assertion is raised if the following condition is not met::

        abs(x - y) <= nulp * spacing(maximum(abs(x), abs(y)))

    Examples
    --------
    >>> x = np.array([1.0, 1e-10, 1e-20])
    >>> eps = np.finfo(x.dtype).eps
    >>> np.testing.assert_array_almost_equal_nulp(x, x * eps / 2 + x)  # doctest: +SKIP

    >>> np.testing.assert_array_almost_equal_nulp(x, x * eps + x)  # doctest: +SKIP
    Traceback (most recent call last):
      ...
    AssertionError: X and Y are not equal to 1 ULP (max is 2)

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    import numpy as np

    ax = np.abs(x)
    ay = np.abs(y)
    ref = nulp * np.spacing(np.where(ax > ay, ax, ay))
    if not np.all(np.abs(x - y) <= ref):
        if np.iscomplexobj(x) or np.iscomplexobj(y):
            msg = f"X and Y are not equal to {nulp:d} ULP"
        else:
            max_nulp = np.max(nulp_diff(x, y))
            msg = f"X and Y are not equal to {nulp:d} ULP (max is {max_nulp:g})"
        raise AssertionError(msg)


def assert_array_max_ulp(a, b, maxulp=1, dtype=None):
    """
    Check that all items of arrays differ in at most N Units in the Last Place.

    Parameters
    ----------
    a, b : array_like
        Input arrays to be compared.
    maxulp : int, optional
        The maximum number of units in the last place that elements of `a` and
        `b` can differ. Default is 1.
    dtype : dtype, optional
        Data-type to convert `a` and `b` to if given. Default is None.

    Returns
    -------
    ret : ndarray
        Array containing number of representable floating point numbers between
        items in `a` and `b`.

    Raises
    ------
    AssertionError
        If one or more elements differ by more than `maxulp`.

    Notes
    -----
    For computing the ULP difference, this API does not differentiate between
    various representations of NAN (ULP difference between 0x7fc00000 and 0xffc00000
    is zero).

    See Also
    --------
    assert_array_almost_equal_nulp : Compare two arrays relatively to their
        spacing.

    Examples
    --------
    >>> a = np.linspace(0.0, 1.0, 100)
    >>> res = np.testing.assert_array_max_ulp(a, np.arcsin(np.sin(a)))  # doctest: +SKIP

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    import numpy as np

    ret = nulp_diff(a, b, dtype)
    if not np.all(ret <= maxulp):
        raise AssertionError(
            f"Arrays are not almost equal up to {maxulp:g} "
            f"ULP (max difference is {np.max(ret):g} ULP)"
        )
    return ret


def nulp_diff(x, y, dtype=None):
    """For each item in x and y, return the number of representable floating
    points between them.

    Parameters
    ----------
    x : array_like
        first input array
    y : array_like
        second input array
    dtype : dtype, optional
        Data-type to convert `x` and `y` to if given. Default is None.

    Returns
    -------
    nulp : array_like
        number of representable floating point numbers between each item in x
        and y.

    Notes
    -----
    For computing the ULP difference, this API does not differentiate between
    various representations of NAN (ULP difference between 0x7fc00000 and 0xffc00000
    is zero).

    Examples
    --------
    # By definition, epsilon is the smallest number such as 1 + eps != 1, so
    # there should be exactly one ULP between 1 and 1 + eps
    >>> nulp_diff(1, 1 + np.finfo(x.dtype).eps)  # doctest: +SKIP
    1.0
    """
    import numpy as np

    if dtype:
        x = np.asarray(x, dtype=dtype)
        y = np.asarray(y, dtype=dtype)
    else:
        x = np.asarray(x)
        y = np.asarray(y)

    t = np.common_type(x, y)
    if np.iscomplexobj(x) or np.iscomplexobj(y):
        raise NotImplementedError("_nulp not implemented for complex array")

    x = np.array([x], dtype=t)
    y = np.array([y], dtype=t)

    x[np.isnan(x)] = np.nan
    y[np.isnan(y)] = np.nan

    if not x.shape == y.shape:
        raise ValueError(f"x and y do not have the same shape: {x.shape} - {y.shape}")

    def _diff(rx, ry, vdt):
        diff = np.asarray(rx - ry, dtype=vdt)
        return np.abs(diff)

    rx = integer_repr(x)
    ry = integer_repr(y)
    return _diff(rx, ry, t)


def _integer_repr(x, vdt, comp):
    # Reinterpret binary representation of the float as sign-magnitude:
    # take into account two-complement representation
    # See also
    # https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
    rx = x.view(vdt)
    if rx.size != 1:
        rx[rx < 0] = comp - rx[rx < 0]
    else:
        if rx < 0:
            rx = comp - rx

    return rx


def integer_repr(x):
    """Return the signed-magnitude interpretation of the binary representation
    of x."""
    import numpy as np

    if x.dtype == np.float16:
        return _integer_repr(x, np.int16, np.int16(-(2**15)))
    elif x.dtype == np.float32:
        return _integer_repr(x, np.int32, np.int32(-(2**31)))
    elif x.dtype == np.float64:
        return _integer_repr(x, np.int64, np.int64(-(2**63)))
    else:
        raise ValueError(f"Unsupported dtype {x.dtype}")


@contextlib.contextmanager
def _assert_warns_context(warning_class, name=None):
    __tracebackhide__ = True  # Hide traceback for py.test
    with suppress_warnings() as sup:
        l = sup.record(warning_class)
        yield
        if not len(l) > 0:
            name_str = f" when calling {name}" if name is not None else ""
            raise AssertionError("No warning raised" + name_str)


def assert_warns(warning_class, *args, **kwargs):
    """
    Fail unless the given callable throws the specified warning.

    A warning of class warning_class should be thrown by the callable when
    invoked with arguments args and keyword arguments kwargs.
    If a different type of warning is thrown, it will not be caught.

    If called with all arguments other than the warning class omitted, may be
    used as a context manager:

        with assert_warns(SomeWarning):
            do_something()

    The ability to be used as a context manager is new in NumPy v1.11.0.

    .. versionadded:: 1.4.0

    Parameters
    ----------
    warning_class : class
        The class defining the warning that `func` is expected to throw.
    func : callable, optional
        Callable to test
    *args : Arguments
        Arguments for `func`.
    **kwargs : Kwargs
        Keyword arguments for `func`.

    Returns
    -------
    The value returned by `func`.

    Examples
    --------
    >>> import warnings
    >>> def deprecated_func(num):
    ...     warnings.warn("Please upgrade", DeprecationWarning)
    ...     return num * num
    >>> with np.testing.assert_warns(DeprecationWarning):
    ...     assert deprecated_func(4) == 16
    >>> # or passing a func
    >>> ret = np.testing.assert_warns(DeprecationWarning, deprecated_func, 4)
    >>> assert ret == 16
    """
    if not args:
        return _assert_warns_context(warning_class)

    func = args[0]
    args = args[1:]
    with _assert_warns_context(warning_class, name=func.__name__):
        return func(*args, **kwargs)


@contextlib.contextmanager
def _assert_no_warnings_context(name=None):
    __tracebackhide__ = True  # Hide traceback for py.test
    with warnings.catch_warnings(record=True) as l:
        warnings.simplefilter("always")
        yield
        if len(l) > 0:
            name_str = f" when calling {name}" if name is not None else ""
            raise AssertionError(f"Got warnings{name_str}: {l}")


def assert_no_warnings(*args, **kwargs):
    """
    Fail if the given callable produces any warnings.

    If called with all arguments omitted, ma
```



## High-Level Overview

"""Utility function to facilitate testing.

This Python file contains 13 class(es) and 67 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_Dummy`, `IgnoreException`, `clear_and_catch_warnings`, `suppress_warnings`

**Functions defined**: `assert_`, `gisnan`, `gisfinite`, `gisinf`, `build_err_msg`, `assert_equal`, `print_assert_equal`, `assert_almost_equal`, `_build_err_msg`, `assert_approx_equal`, `assert_array_compare`, `array2string`, `func_assert_same_pos`, `assert_array_equal`, `assert_array_almost_equal`, `compare`, `assert_array_less`, `assert_string_equal`, `nop`, `assert_raises_regex`

**Key imports**: contextlib, gc, operator, os, platform, pprint, re, shutil, sys, warnings


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_numpy/testing`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `gc`
- `operator`
- `os`
- `platform`
- `pprint`
- `re`
- `shutil`
- `sys`
- `warnings`
- `functools`: wraps
- `io`: StringIO
- `tempfile`: mkdtemp, mkstemp
- `torch._numpy as np`
- `torch._numpy`: arange, asarray as asanyarray, empty, float32, intp, ndarray
- `torch._numpy.testing`: assert_almost_equal
- `numpy as np`
- `traceback`
- `of difflib to reduce startup time`
- `difflib`
- `unittest`
- `to reduce startup time`
- `inspect`: isfunction


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Context Manager**: Implements context manager protocol
- **Error Handling**: Includes exception handling


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
python torch/_numpy/testing/utils.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/_numpy/testing`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)


## Cross-References

- **File Documentation**: `utils.py_docs.md`
- **Keyword Index**: `utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
