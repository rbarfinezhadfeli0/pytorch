# Documentation: `docs/test/dynamo/cpython/3_13/test_math.py_docs.md`

## File Metadata

- **Path**: `docs/test/dynamo/cpython/3_13/test_math.py_docs.md`
- **Size**: 53,670 bytes (52.41 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/dynamo/cpython/3_13/test_math.py`

## File Metadata

- **Path**: `test/dynamo/cpython/3_13/test_math.py`
- **Size**: 125,375 bytes (122.44 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_math.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import (
    slowTest,
    run_tests,
    skipIfTorchDynamo,
)

__TestCase = CPythonTestCase


# redirect import statements
import sys
import importlib.abc

redirect_imports = (
    "test.mapping_tests",
    "test.typinganndata",
    "test.test_grammar",
    "test.test_math",
    "test.test_iter",
    "test.typinganndata.ann_module",
)

class RedirectImportFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        # Check if the import is the problematic one
        if fullname in redirect_imports:
            try:
                # Attempt to import the standalone module
                name = fullname.removeprefix("test.")
                r = importlib.import_module(name)
                # Redirect the module in sys.modules
                sys.modules[fullname] = r
                # Return a module spec from the found module
                return importlib.util.find_spec(name)
            except ImportError:
                return None
        return None

# Add the custom finder to sys.meta_path
sys.meta_path.insert(0, RedirectImportFinder())


# ======= END DYNAMO PATCH =======

# Python test set -- math module
# XXXX Should not do tests around zero only

from test.support import verbose, requires_IEEE_754
from test import support
import unittest
import fractions
import itertools
import decimal
import math
import os
import platform
import random
import struct
import sys


eps = 1E-05
NAN = float('nan')
INF = float('inf')
NINF = float('-inf')
FLOAT_MAX = sys.float_info.max
FLOAT_MIN = sys.float_info.min

# detect evidence of double-rounding: fsum is not always correctly
# rounded on machines that suffer from double rounding.
x, y = 1e16, 2.9999 # use temporary values to defeat peephole optimizer
HAVE_DOUBLE_ROUNDING = (x + y == 1e16 + 4)

# locate file with test values
if __name__ == '__main__':
    file = sys.argv[0]
else:
    file = __file__
test_dir = os.path.dirname(file) or os.curdir
math_testcases = os.path.join(test_dir, 'mathdata', 'math_testcases.txt')
test_file = os.path.join(test_dir, 'mathdata', 'cmath_testcases.txt')


def to_ulps(x):
    """Convert a non-NaN float x to an integer, in such a way that
    adjacent floats are converted to adjacent integers.  Then
    abs(ulps(x) - ulps(y)) gives the difference in ulps between two
    floats.

    The results from this function will only make sense on platforms
    where native doubles are represented in IEEE 754 binary64 format.

    Note: 0.0 and -0.0 are converted to 0 and -1, respectively.
    """
    n = struct.unpack('<q', struct.pack('<d', x))[0]
    if n < 0:
        n = ~(n+2**63)
    return n


# Here's a pure Python version of the math.factorial algorithm, for
# documentation and comparison purposes.
#
# Formula:
#
#   factorial(n) = factorial_odd_part(n) << (n - count_set_bits(n))
#
# where
#
#   factorial_odd_part(n) = product_{i >= 0} product_{0 < j <= n >> i; j odd} j
#
# The outer product above is an infinite product, but once i >= n.bit_length,
# (n >> i) < 1 and the corresponding term of the product is empty.  So only the
# finitely many terms for 0 <= i < n.bit_length() contribute anything.
#
# We iterate downwards from i == n.bit_length() - 1 to i == 0.  The inner
# product in the formula above starts at 1 for i == n.bit_length(); for each i
# < n.bit_length() we get the inner product for i from that for i + 1 by
# multiplying by all j in {n >> i+1 < j <= n >> i; j odd}.  In Python terms,
# this set is range((n >> i+1) + 1 | 1, (n >> i) + 1 | 1, 2).

def count_set_bits(n):
    """Number of '1' bits in binary expansion of a nonnnegative integer."""
    return 1 + count_set_bits(n & n - 1) if n else 0

def partial_product(start, stop):
    """Product of integers in range(start, stop, 2), computed recursively.
    start and stop should both be odd, with start <= stop.

    """
    numfactors = (stop - start) >> 1
    if not numfactors:
        return 1
    elif numfactors == 1:
        return start
    else:
        mid = (start + numfactors) | 1
        return partial_product(start, mid) * partial_product(mid, stop)

def py_factorial(n):
    """Factorial of nonnegative integer n, via "Binary Split Factorial Formula"
    described at http://www.luschny.de/math/factorial/binarysplitfact.html

    """
    inner = outer = 1
    for i in reversed(range(n.bit_length())):
        inner *= partial_product((n >> i + 1) + 1 | 1, (n >> i) + 1 | 1)
        outer *= inner
    return outer << (n - count_set_bits(n))

def ulp_abs_check(expected, got, ulp_tol, abs_tol):
    """Given finite floats `expected` and `got`, check that they're
    approximately equal to within the given number of ulps or the
    given absolute tolerance, whichever is bigger.

    Returns None on success and an error message on failure.
    """
    ulp_error = abs(to_ulps(expected) - to_ulps(got))
    abs_error = abs(expected - got)

    # Succeed if either abs_error <= abs_tol or ulp_error <= ulp_tol.
    if abs_error <= abs_tol or ulp_error <= ulp_tol:
        return None
    else:
        fmt = ("error = {:.3g} ({:d} ulps); "
               "permitted error = {:.3g} or {:d} ulps")
        return fmt.format(abs_error, ulp_error, abs_tol, ulp_tol)

def parse_mtestfile(fname):
    """Parse a file with test values

    -- starts a comment
    blank lines, or lines containing only a comment, are ignored
    other lines are expected to have the form
      id fn arg -> expected [flag]*

    """
    with open(fname, encoding="utf-8") as fp:
        for line in fp:
            # strip comments, and skip blank lines
            if '--' in line:
                line = line[:line.index('--')]
            if not line.strip():
                continue

            lhs, rhs = line.split('->')
            id, fn, arg = lhs.split()
            rhs_pieces = rhs.split()
            exp = rhs_pieces[0]
            flags = rhs_pieces[1:]

            yield (id, fn, float(arg), float(exp), flags)


def parse_testfile(fname):
    """Parse a file with test values

    Empty lines or lines starting with -- are ignored
    yields id, fn, arg_real, arg_imag, exp_real, exp_imag
    """
    with open(fname, encoding="utf-8") as fp:
        for line in fp:
            # skip comment lines and blank lines
            if line.startswith('--') or not line.strip():
                continue

            lhs, rhs = line.split('->')
            id, fn, arg_real, arg_imag = lhs.split()
            rhs_pieces = rhs.split()
            exp_real, exp_imag = rhs_pieces[0], rhs_pieces[1]
            flags = rhs_pieces[2:]

            yield (id, fn,
                   float(arg_real), float(arg_imag),
                   float(exp_real), float(exp_imag),
                   flags)


def result_check(expected, got, ulp_tol=5, abs_tol=0.0):
    # Common logic of MathTests.(ftest, test_testcases, test_mtestcases)
    """Compare arguments expected and got, as floats, if either
    is a float, using a tolerance expressed in multiples of
    ulp(expected) or absolutely (if given and greater).

    As a convenience, when neither argument is a float, and for
    non-finite floats, exact equality is demanded. Also, nan==nan
    as far as this function is concerned.

    Returns None on success and an error message on failure.
    """

    # Check exactly equal (applies also to strings representing exceptions)
    if got == expected:
        if not got and not expected:
            if math.copysign(1, got) != math.copysign(1, expected):
                return f"expected {expected}, got {got} (zero has wrong sign)"
        return None

    failure = "not equal"

    # Turn mixed float and int comparison (e.g. floor()) to all-float
    if isinstance(expected, float) and isinstance(got, int):
        got = float(got)
    elif isinstance(got, float) and isinstance(expected, int):
        expected = float(expected)

    if isinstance(expected, float) and isinstance(got, float):
        if math.isnan(expected) and math.isnan(got):
            # Pass, since both nan
            failure = None
        elif math.isinf(expected) or math.isinf(got):
            # We already know they're not equal, drop through to failure
            pass
        else:
            # Both are finite floats (now). Are they close enough?
            failure = ulp_abs_check(expected, got, ulp_tol, abs_tol)

    # arguments are not equal, and if numeric, are too far apart
    if failure is not None:
        fail_fmt = "expected {!r}, got {!r}"
        fail_msg = fail_fmt.format(expected, got)
        fail_msg += ' ({})'.format(failure)
        return fail_msg
    else:
        return None

class FloatLike:
    def __init__(self, value):
        self.value = value

    def __float__(self):
        return self.value

class IntSubclass(int):
    pass

# Class providing an __index__ method.
class MyIndexable(object):
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value

class BadDescr:
    def __get__(self, obj, objtype=None):
        raise ValueError

class MathTests(__TestCase):

    def ftest(self, name, got, expected, ulp_tol=5, abs_tol=0.0):
        """Compare arguments expected and got, as floats, if either
        is a float, using a tolerance expressed in multiples of
        ulp(expected) or absolutely, whichever is greater.

        As a convenience, when neither argument is a float, and for
        non-finite floats, exact equality is demanded. Also, nan==nan
        in this function.
        """
        failure = result_check(expected, got, ulp_tol, abs_tol)
        if failure is not None:
            self.fail("{}: {}".format(name, failure))

    def testConstants(self):
        # Ref: Abramowitz & Stegun (Dover, 1965)
        self.ftest('pi', math.pi, 3.141592653589793238462643)
        self.ftest('e', math.e, 2.718281828459045235360287)
        self.assertEqual(math.tau, 2*math.pi)

    def testAcos(self):
        self.assertRaises(TypeError, math.acos)
        self.ftest('acos(-1)', math.acos(-1), math.pi)
        self.ftest('acos(0)', math.acos(0), math.pi/2)
        self.ftest('acos(1)', math.acos(1), 0)
        self.assertRaises(ValueError, math.acos, INF)
        self.assertRaises(ValueError, math.acos, NINF)
        self.assertRaises(ValueError, math.acos, 1 + eps)
        self.assertRaises(ValueError, math.acos, -1 - eps)
        self.assertTrue(math.isnan(math.acos(NAN)))

    def testAcosh(self):
        self.assertRaises(TypeError, math.acosh)
        self.ftest('acosh(1)', math.acosh(1), 0)
        self.ftest('acosh(2)', math.acosh(2), 1.3169578969248168)
        self.assertRaises(ValueError, math.acosh, 0)
        self.assertRaises(ValueError, math.acosh, -1)
        self.assertEqual(math.acosh(INF), INF)
        self.assertRaises(ValueError, math.acosh, NINF)
        self.assertTrue(math.isnan(math.acosh(NAN)))

    def testAsin(self):
        self.assertRaises(TypeError, math.asin)
        self.ftest('asin(-1)', math.asin(-1), -math.pi/2)
        self.ftest('asin(0)', math.asin(0), 0)
        self.ftest('asin(1)', math.asin(1), math.pi/2)
        self.assertRaises(ValueError, math.asin, INF)
        self.assertRaises(ValueError, math.asin, NINF)
        self.assertRaises(ValueError, math.asin, 1 + eps)
        self.assertRaises(ValueError, math.asin, -1 - eps)
        self.assertTrue(math.isnan(math.asin(NAN)))

    def testAsinh(self):
        self.assertRaises(TypeError, math.asinh)
        self.ftest('asinh(0)', math.asinh(0), 0)
        self.ftest('asinh(1)', math.asinh(1), 0.88137358701954305)
        self.ftest('asinh(-1)', math.asinh(-1), -0.88137358701954305)
        self.assertEqual(math.asinh(INF), INF)
        self.assertEqual(math.asinh(NINF), NINF)
        self.assertTrue(math.isnan(math.asinh(NAN)))

    def testAtan(self):
        self.assertRaises(TypeError, math.atan)
        self.ftest('atan(-1)', math.atan(-1), -math.pi/4)
        self.ftest('atan(0)', math.atan(0), 0)
        self.ftest('atan(1)', math.atan(1), math.pi/4)
        self.ftest('atan(inf)', math.atan(INF), math.pi/2)
        self.ftest('atan(-inf)', math.atan(NINF), -math.pi/2)
        self.assertTrue(math.isnan(math.atan(NAN)))

    def testAtanh(self):
        self.assertRaises(TypeError, math.atan)
        self.ftest('atanh(0)', math.atanh(0), 0)
        self.ftest('atanh(0.5)', math.atanh(0.5), 0.54930614433405489)
        self.ftest('atanh(-0.5)', math.atanh(-0.5), -0.54930614433405489)
        self.assertRaises(ValueError, math.atanh, 1)
        self.assertRaises(ValueError, math.atanh, -1)
        self.assertRaises(ValueError, math.atanh, INF)
        self.assertRaises(ValueError, math.atanh, NINF)
        self.assertTrue(math.isnan(math.atanh(NAN)))

    def testAtan2(self):
        self.assertRaises(TypeError, math.atan2)
        self.ftest('atan2(-1, 0)', math.atan2(-1, 0), -math.pi/2)
        self.ftest('atan2(-1, 1)', math.atan2(-1, 1), -math.pi/4)
        self.ftest('atan2(0, 1)', math.atan2(0, 1), 0)
        self.ftest('atan2(1, 1)', math.atan2(1, 1), math.pi/4)
        self.ftest('atan2(1, 0)', math.atan2(1, 0), math.pi/2)
        self.ftest('atan2(1, -1)', math.atan2(1, -1), 3*math.pi/4)

        # math.atan2(0, x)
        self.ftest('atan2(0., -inf)', math.atan2(0., NINF), math.pi)
        self.ftest('atan2(0., -2.3)', math.atan2(0., -2.3), math.pi)
        self.ftest('atan2(0., -0.)', math.atan2(0., -0.), math.pi)
        self.assertEqual(math.atan2(0., 0.), 0.)
        self.assertEqual(math.atan2(0., 2.3), 0.)
        self.assertEqual(math.atan2(0., INF), 0.)
        self.assertTrue(math.isnan(math.atan2(0., NAN)))
        # math.atan2(-0, x)
        self.ftest('atan2(-0., -inf)', math.atan2(-0., NINF), -math.pi)
        self.ftest('atan2(-0., -2.3)', math.atan2(-0., -2.3), -math.pi)
        self.ftest('atan2(-0., -0.)', math.atan2(-0., -0.), -math.pi)
        self.assertEqual(math.atan2(-0., 0.), -0.)
        self.assertEqual(math.atan2(-0., 2.3), -0.)
        self.assertEqual(math.atan2(-0., INF), -0.)
        self.assertTrue(math.isnan(math.atan2(-0., NAN)))
        # math.atan2(INF, x)
        self.ftest('atan2(inf, -inf)', math.atan2(INF, NINF), math.pi*3/4)
        self.ftest('atan2(inf, -2.3)', math.atan2(INF, -2.3), math.pi/2)
        self.ftest('atan2(inf, -0.)', math.atan2(INF, -0.0), math.pi/2)
        self.ftest('atan2(inf, 0.)', math.atan2(INF, 0.0), math.pi/2)
        self.ftest('atan2(inf, 2.3)', math.atan2(INF, 2.3), math.pi/2)
        self.ftest('atan2(inf, inf)', math.atan2(INF, INF), math.pi/4)
        self.assertTrue(math.isnan(math.atan2(INF, NAN)))
        # math.atan2(NINF, x)
        self.ftest('atan2(-inf, -inf)', math.atan2(NINF, NINF), -math.pi*3/4)
        self.ftest('atan2(-inf, -2.3)', math.atan2(NINF, -2.3), -math.pi/2)
        self.ftest('atan2(-inf, -0.)', math.atan2(NINF, -0.0), -math.pi/2)
        self.ftest('atan2(-inf, 0.)', math.atan2(NINF, 0.0), -math.pi/2)
        self.ftest('atan2(-inf, 2.3)', math.atan2(NINF, 2.3), -math.pi/2)
        self.ftest('atan2(-inf, inf)', math.atan2(NINF, INF), -math.pi/4)
        self.assertTrue(math.isnan(math.atan2(NINF, NAN)))
        # math.atan2(+finite, x)
        self.ftest('atan2(2.3, -inf)', math.atan2(2.3, NINF), math.pi)
        self.ftest('atan2(2.3, -0.)', math.atan2(2.3, -0.), math.pi/2)
        self.ftest('atan2(2.3, 0.)', math.atan2(2.3, 0.), math.pi/2)
        self.assertEqual(math.atan2(2.3, INF), 0.)
        self.assertTrue(math.isnan(math.atan2(2.3, NAN)))
        # math.atan2(-finite, x)
        self.ftest('atan2(-2.3, -inf)', math.atan2(-2.3, NINF), -math.pi)
        self.ftest('atan2(-2.3, -0.)', math.atan2(-2.3, -0.), -math.pi/2)
        self.ftest('atan2(-2.3, 0.)', math.atan2(-2.3, 0.), -math.pi/2)
        self.assertEqual(math.atan2(-2.3, INF), -0.)
        self.assertTrue(math.isnan(math.atan2(-2.3, NAN)))
        # math.atan2(NAN, x)
        self.assertTrue(math.isnan(math.atan2(NAN, NINF)))
        self.assertTrue(math.isnan(math.atan2(NAN, -2.3)))
        self.assertTrue(math.isnan(math.atan2(NAN, -0.)))
        self.assertTrue(math.isnan(math.atan2(NAN, 0.)))
        self.assertTrue(math.isnan(math.atan2(NAN, 2.3)))
        self.assertTrue(math.isnan(math.atan2(NAN, INF)))
        self.assertTrue(math.isnan(math.atan2(NAN, NAN)))

    def testCbrt(self):
        self.assertRaises(TypeError, math.cbrt)
        self.ftest('cbrt(0)', math.cbrt(0), 0)
        self.ftest('cbrt(1)', math.cbrt(1), 1)
        self.ftest('cbrt(8)', math.cbrt(8), 2)
        self.ftest('cbrt(0.0)', math.cbrt(0.0), 0.0)
        self.ftest('cbrt(-0.0)', math.cbrt(-0.0), -0.0)
        self.ftest('cbrt(1.2)', math.cbrt(1.2), 1.062658569182611)
        self.ftest('cbrt(-2.6)', math.cbrt(-2.6), -1.375068867074141)
        self.ftest('cbrt(27)', math.cbrt(27), 3)
        self.ftest('cbrt(-1)', math.cbrt(-1), -1)
        self.ftest('cbrt(-27)', math.cbrt(-27), -3)
        self.assertEqual(math.cbrt(INF), INF)
        self.assertEqual(math.cbrt(NINF), NINF)
        self.assertTrue(math.isnan(math.cbrt(NAN)))

    def testCeil(self):
        self.assertRaises(TypeError, math.ceil)
        self.assertEqual(int, type(math.ceil(0.5)))
        self.assertEqual(math.ceil(0.5), 1)
        self.assertEqual(math.ceil(1.0), 1)
        self.assertEqual(math.ceil(1.5), 2)
        self.assertEqual(math.ceil(-0.5), 0)
        self.assertEqual(math.ceil(-1.0), -1)
        self.assertEqual(math.ceil(-1.5), -1)
        self.assertEqual(math.ceil(0.0), 0)
        self.assertEqual(math.ceil(-0.0), 0)
        #self.assertEqual(math.ceil(INF), INF)
        #self.assertEqual(math.ceil(NINF), NINF)
        #self.assertTrue(math.isnan(math.ceil(NAN)))

        with torch._dynamo.error_on_graph_break(False):
            class TestCeil:
                def __ceil__(self):
                    return 42
            class FloatCeil(float):
                def __ceil__(self):
                    return 42
            class TestNoCeil:
                pass
            class TestBadCeil:
                __ceil__ = BadDescr()
        self.assertEqual(math.ceil(TestCeil()), 42)
        self.assertEqual(math.ceil(FloatCeil()), 42)
        self.assertEqual(math.ceil(FloatLike(42.5)), 43)
        self.assertRaises(TypeError, math.ceil, TestNoCeil())
        self.assertRaises(ValueError, math.ceil, TestBadCeil())

        t = TestNoCeil()
        t.__ceil__ = lambda *args: args
        self.assertRaises(TypeError, math.ceil, t)
        self.assertRaises(TypeError, math.ceil, t, 0)

        self.assertEqual(math.ceil(FloatLike(+1.0)), +1.0)
        self.assertEqual(math.ceil(FloatLike(-1.0)), -1.0)

    @requires_IEEE_754
    def testCopysign(self):
        self.assertEqual(math.copysign(1, 42), 1.0)
        self.assertEqual(math.copysign(0., 42), 0.0)
        self.assertEqual(math.copysign(1., -42), -1.0)
        self.assertEqual(math.copysign(3, 0.), 3.0)
        self.assertEqual(math.copysign(4., -0.), -4.0)

        self.assertRaises(TypeError, math.copysign)
        # copysign should let us distinguish signs of zeros
        self.assertEqual(math.copysign(1., 0.), 1.)
        self.assertEqual(math.copysign(1., -0.), -1.)
        self.assertEqual(math.copysign(INF, 0.), INF)
        self.assertEqual(math.copysign(INF, -0.), NINF)
        self.assertEqual(math.copysign(NINF, 0.), INF)
        self.assertEqual(math.copysign(NINF, -0.), NINF)
        # and of infinities
        self.assertEqual(math.copysign(1., INF), 1.)
        self.assertEqual(math.copysign(1., NINF), -1.)
        self.assertEqual(math.copysign(INF, INF), INF)
        self.assertEqual(math.copysign(INF, NINF), NINF)
        self.assertEqual(math.copysign(NINF, INF), INF)
        self.assertEqual(math.copysign(NINF, NINF), NINF)
        self.assertTrue(math.isnan(math.copysign(NAN, 1.)))
        self.assertTrue(math.isnan(math.copysign(NAN, INF)))
        self.assertTrue(math.isnan(math.copysign(NAN, NINF)))
        self.assertTrue(math.isnan(math.copysign(NAN, NAN)))
        # copysign(INF, NAN) may be INF or it may be NINF, since
        # we don't know whether the sign bit of NAN is set on any
        # given platform.
        self.assertTrue(math.isinf(math.copysign(INF, NAN)))
        # similarly, copysign(2., NAN) could be 2. or -2.
        self.assertEqual(abs(math.copysign(2., NAN)), 2.)

    def testCos(self):
        self.assertRaises(TypeError, math.cos)
        self.ftest('cos(-pi/2)', math.cos(-math.pi/2), 0, abs_tol=math.ulp(1))
        self.ftest('cos(0)', math.cos(0), 1)
        self.ftest('cos(pi/2)', math.cos(math.pi/2), 0, abs_tol=math.ulp(1))
        self.ftest('cos(pi)', math.cos(math.pi), -1)
        try:
            self.assertTrue(math.isnan(math.cos(INF)))
            self.assertTrue(math.isnan(math.cos(NINF)))
        except ValueError:
            self.assertRaises(ValueError, math.cos, INF)
            self.assertRaises(ValueError, math.cos, NINF)
        self.assertTrue(math.isnan(math.cos(NAN)))

    @unittest.skipIf(sys.platform == 'win32' and platform.machine() in ('ARM', 'ARM64'),
                    "Windows UCRT is off by 2 ULP this test requires accuracy within 1 ULP")
    def testCosh(self):
        self.assertRaises(TypeError, math.cosh)
        self.ftest('cosh(0)', math.cosh(0), 1)
        self.ftest('cosh(2)-2*cosh(1)**2', math.cosh(2)-2*math.cosh(1)**2, -1) # Thanks to Lambert
        self.assertEqual(math.cosh(INF), INF)
        self.assertEqual(math.cosh(NINF), INF)
        self.assertTrue(math.isnan(math.cosh(NAN)))

    def testDegrees(self):
        self.assertRaises(TypeError, math.degrees)
        self.ftest('degrees(pi)', math.degrees(math.pi), 180.0)
        self.ftest('degrees(pi/2)', math.degrees(math.pi/2), 90.0)
        self.ftest('degrees(-pi/4)', math.degrees(-math.pi/4), -45.0)
        self.ftest('degrees(0)', math.degrees(0), 0)

    def testExp(self):
        self.assertRaises(TypeError, math.exp)
        self.ftest('exp(-1)', math.exp(-1), 1/math.e)
        self.ftest('exp(0)', math.exp(0), 1)
        self.ftest('exp(1)', math.exp(1), math.e)
        self.assertEqual(math.exp(INF), INF)
        self.assertEqual(math.exp(NINF), 0.)
        self.assertTrue(math.isnan(math.exp(NAN)))
        self.assertRaises(OverflowError, math.exp, 1000000)

    def testExp2(self):
        self.assertRaises(TypeError, math.exp2)
        self.ftest('exp2(-1)', math.exp2(-1), 0.5)
        self.ftest('exp2(0)', math.exp2(0), 1)
        self.ftest('exp2(1)', math.exp2(1), 2)
        self.ftest('exp2(2.3)', math.exp2(2.3), 4.924577653379665)
        self.assertEqual(math.exp2(INF), INF)
        self.assertEqual(math.exp2(NINF), 0.)
        self.assertTrue(math.isnan(math.exp2(NAN)))
        self.assertRaises(OverflowError, math.exp2, 1000000)

    def testFabs(self):
        self.assertRaises(TypeError, math.fabs)
        self.ftest('fabs(-1)', math.fabs(-1), 1)
        self.ftest('fabs(0)', math.fabs(0), 0)
        self.ftest('fabs(1)', math.fabs(1), 1)

    @skipIfTorchDynamo("infinite loop")
    def testFactorial(self):
        self.assertEqual(math.factorial(0), 1)
        total = 1
        for i in range(1, 1000):
            total *= i
            self.assertEqual(math.factorial(i), total)
            self.assertEqual(math.factorial(i), py_factorial(i))
        self.assertRaises(ValueError, math.factorial, -1)
        self.assertRaises(ValueError, math.factorial, -10**100)

    def testFactorialNonIntegers(self):
        self.assertRaises(TypeError, math.factorial, 5.0)
        self.assertRaises(TypeError, math.factorial, 5.2)
        self.assertRaises(TypeError, math.factorial, -1.0)
        self.assertRaises(TypeError, math.factorial, -1e100)
        self.assertRaises(TypeError, math.factorial, decimal.Decimal('5'))
        self.assertRaises(TypeError, math.factorial, decimal.Decimal('5.2'))
        self.assertRaises(TypeError, math.factorial, "5")

    # Other implementations may place different upper bounds.
    @support.cpython_only
    def testFactorialHugeInputs(self):
        # Currently raises OverflowError for inputs that are too large
        # to fit into a C long.
        self.assertRaises(OverflowError, math.factorial, 10**100)
        self.assertRaises(TypeError, math.factorial, 1e100)

    def testFloor(self):
        self.assertRaises(TypeError, math.floor)
        self.assertEqual(int, type(math.floor(0.5)))
        self.assertEqual(math.floor(0.5), 0)
        self.assertEqual(math.floor(1.0), 1)
        self.assertEqual(math.floor(1.5), 1)
        self.assertEqual(math.floor(-0.5), -1)
        self.assertEqual(math.floor(-1.0), -1)
        self.assertEqual(math.floor(-1.5), -2)
        #self.assertEqual(math.ceil(INF), INF)
        #self.assertEqual(math.ceil(NINF), NINF)
        #self.assertTrue(math.isnan(math.floor(NAN)))

        with torch._dynamo.error_on_graph_break(False):
            class TestFloor:
                def __floor__(self):
                    return 42
            class FloatFloor(float):
                def __floor__(self):
                    return 42
            class TestNoFloor:
                pass
            class TestBadFloor:
                __floor__ = BadDescr()
        self.assertEqual(math.floor(TestFloor()), 42)
        self.assertEqual(math.floor(FloatFloor()), 42)
        self.assertEqual(math.floor(FloatLike(41.9)), 41)
        self.assertRaises(TypeError, math.floor, TestNoFloor())
        self.assertRaises(ValueError, math.floor, TestBadFloor())

        t = TestNoFloor()
        t.__floor__ = lambda *args: args
        self.assertRaises(TypeError, math.floor, t)
        self.assertRaises(TypeError, math.floor, t, 0)

        self.assertEqual(math.floor(FloatLike(+1.0)), +1.0)
        self.assertEqual(math.floor(FloatLike(-1.0)), -1.0)

    def testFmod(self):
        self.assertRaises(TypeError, math.fmod)
        self.ftest('fmod(10, 1)', math.fmod(10, 1), 0.0)
        self.ftest('fmod(10, 0.5)', math.fmod(10, 0.5), 0.0)
        self.ftest('fmod(10, 1.5)', math.fmod(10, 1.5), 1.0)
        self.ftest('fmod(-10, 1)', math.fmod(-10, 1), -0.0)
        self.ftest('fmod(-10, 0.5)', math.fmod(-10, 0.5), -0.0)
        self.ftest('fmod(-10, 1.5)', math.fmod(-10, 1.5), -1.0)
        self.assertTrue(math.isnan(math.fmod(NAN, 1.)))
        self.assertTrue(math.isnan(math.fmod(1., NAN)))
        self.assertTrue(math.isnan(math.fmod(NAN, NAN)))
        self.assertRaises(ValueError, math.fmod, 1., 0.)
        self.assertRaises(ValueError, math.fmod, INF, 1.)
        self.assertRaises(ValueError, math.fmod, NINF, 1.)
        self.assertRaises(ValueError, math.fmod, INF, 0.)
        self.assertEqual(math.fmod(3.0, INF), 3.0)
        self.assertEqual(math.fmod(-3.0, INF), -3.0)
        self.assertEqual(math.fmod(3.0, NINF), 3.0)
        self.assertEqual(math.fmod(-3.0, NINF), -3.0)
        self.assertEqual(math.fmod(0.0, 3.0), 0.0)
        self.assertEqual(math.fmod(0.0, NINF), 0.0)
        self.assertRaises(ValueError, math.fmod, INF, INF)

    def testFrexp(self):
        self.assertRaises(TypeError, math.frexp)

        def testfrexp(name, result, expected):
            (mant, exp), (emant, eexp) = result, expected
            if abs(mant-emant) > eps or exp != eexp:
                self.fail('%s returned %r, expected %r'%\
                          (name, result, expected))

        testfrexp('frexp(-1)', math.frexp(-1), (-0.5, 1))
        testfrexp('frexp(0)', math.frexp(0), (0, 0))
        testfrexp('frexp(1)', math.frexp(1), (0.5, 1))
        testfrexp('frexp(2)', math.frexp(2), (0.5, 2))

        self.assertEqual(math.frexp(INF)[0], INF)
        self.assertEqual(math.frexp(NINF)[0], NINF)
        self.assertTrue(math.isnan(math.frexp(NAN)[0]))

    @requires_IEEE_754
    @unittest.skipIf(HAVE_DOUBLE_ROUNDING,
                         "fsum is not exact on machines with double rounding")
    def testFsum(self):
        # math.fsum relies on exact rounding for correct operation.
        # There's a known problem with IA32 floating-point that causes
        # inexact rounding in some situations, and will cause the
        # math.fsum tests below to fail; see issue #2937.  On non IEEE
        # 754 platforms, and on IEEE 754 platforms that exhibit the
        # problem described in issue #2937, we simply skip the whole
        # test.

        # Python version of math.fsum, for comparison.  Uses a
        # different algorithm based on frexp, ldexp and integer
        # arithmetic.
        from sys import float_info
        mant_dig = float_info.mant_dig
        etiny = float_info.min_exp - mant_dig

        def msum(iterable):
            """Full precision summation.  Compute sum(iterable) without any
            intermediate accumulation of error.  Based on the 'lsum' function
            at https://code.activestate.com/recipes/393090-binary-floating-point-summation-accurate-to-full-p/

            """
            tmant, texp = 0, 0
            for x in iterable:
                mant, exp = math.frexp(x)
                mant, exp = int(math.ldexp(mant, mant_dig)), exp - mant_dig
                if texp > exp:
                    tmant <<= texp-exp
                    texp = exp
                else:
                    mant <<= exp-texp
                tmant += mant
            # Round tmant * 2**texp to a float.  The original recipe
            # used float(str(tmant)) * 2.0**texp for this, but that's
            # a little unsafe because str -> float conversion can't be
            # relied upon to do correct rounding on all platforms.
            tail = max(len(bin(abs(tmant)))-2 - mant_dig, etiny - texp)
            if tail > 0:
                h = 1 << (tail-1)
                tmant = tmant // (2*h) + bool(tmant & h and tmant & 3*h-1)
                texp += tail
            return math.ldexp(tmant, texp)

        test_values = [
            ([], 0.0),
            ([0.0], 0.0),
            ([1e100, 1.0, -1e100, 1e-100, 1e50, -1.0, -1e50], 1e-100),
            ([1e100, 1.0, -1e100, 1e-100, 1e50, -1, -1e50], 1e-100),
            ([2.0**53, -0.5, -2.0**-54], 2.0**53-1.0),
            ([2.0**53, 1.0, 2.0**-100], 2.0**53+2.0),
            ([2.0**53+10.0, 1.0, 2.0**-100], 2.0**53+12.0),
            ([2.0**53-4.0, 0.5, 2.0**-54], 2.0**53-3.0),
            ([1./n for n in range(1, 1001)],
             float.fromhex('0x1.df11f45f4e61ap+2')),
            ([(-1.)**n/n for n in range(1, 1001)],
             float.fromhex('-0x1.62a2af1bd3624p-1')),
            ([1e16, 1., 1e-16], 10000000000000002.0),
            ([1e16-2., 1.-2.**-53, -(1e16-2.), -(1.-2.**-53)], 0.0),
            # exercise code for resizing partials array
            ([2.**n - 2.**(n+50) + 2.**(n+52) for n in range(-1074, 972, 2)] +
             [-2.**1022],
             float.fromhex('0x1.5555555555555p+970')),
            ]

        # Telescoping sum, with exact differences (due to Sterbenz)
        terms = [1.7**i for i in range(1001)]
        test_values.append((
            [terms[i+1] - terms[i] for i in range(1000)] + [-terms[1000]],
            -terms[0]
        ))

        for i, (vals, expected) in enumerate(test_values):
            try:
                actual = math.fsum(vals)
            except OverflowError:
                self.fail("test %d failed: got OverflowError, expected %r "
                          "for math.fsum(%.100r)" % (i, expected, vals))
            except ValueError:
                self.fail("test %d failed: got ValueError, expected %r "
                          "for math.fsum(%.100r)" % (i, expected, vals))
            self.assertEqual(actual, expected)

        from random import random, gauss, shuffle
        for j in range(1000):
            vals = [7, 1e100, -7, -1e100, -9e-20, 8e-20] * 10
            s = 0
            for i in range(200):
                v = gauss(0, random()) ** 7 - s
                s += v
                vals.append(v)
            shuffle(vals)

            s = msum(vals)
            self.assertEqual(msum(vals), math.fsum(vals))

        self.assertEqual(math.fsum([1.0, math.inf]), math.inf)
        self.assertTrue(math.isnan(math.fsum([math.nan, 1.0])))
        self.assertEqual(math.fsum([1e100, FloatLike(1.0), -1e100, 1e-100,
                                    1e50, FloatLike(-1.0), -1e50]), 1e-100)
        self.assertRaises(OverflowError, math.fsum, [1e+308, 1e+308])
        self.assertRaises(ValueError, math.fsum, [math.inf, -math.inf])
        self.assertRaises(TypeError, math.fsum, ['spam'])
        self.assertRaises(TypeError, math.fsum, 1)
        self.assertRaises(OverflowError, math.fsum, [10**1000])

        def bad_iter():
            yield 1.0
            raise ZeroDivisionError

        self.assertRaises(ZeroDivisionError, math.fsum, bad_iter())

    def testGcd(self):
        gcd = math.gcd
        self.assertEqual(gcd(0, 0), 0)
        self.assertEqual(gcd(1, 0), 1)
        self.assertEqual(gcd(-1, 0), 1)
        self.assertEqual(gcd(0, 1), 1)
        self.assertEqual(gcd(0, -1), 1)
        self.assertEqual(gcd(7, 1), 1)
        self.assertEqual(gcd(7, -1), 1)
        self.assertEqual(gcd(-23, 15), 1)
        self.assertEqual(gcd(120, 84), 12)
        self.assertEqual(gcd(84, -120), 12)
        self.assertEqual(gcd(1216342683557601535506311712,
                             436522681849110124616458784), 32)

        x = 434610456570399902378880679233098819019853229470286994367836600566
        y = 1064502245825115327754847244914921553977
        for c in (652560,
                  576559230871654959816130551884856912003141446781646602790216406874):
            a = x * c
            b = y * c
            self.assertEqual(gcd(a, b), c)
            self.assertEqual(gcd(b, a), c)
            self.assertEqual(gcd(-a, b), c)
            self.assertEqual(gcd(b, -a), c)
            self.assertEqual(gcd(a, -b), c)
            self.assertEqual(gcd(-b, a), c)
            self.assertEqual(gcd(-a, -b), c)
            self.assertEqual(gcd(-b, -a), c)

        self.assertEqual(gcd(), 0)
        self.assertEqual(gcd(120), 120)
        self.assertEqual(gcd(-120), 120)
        self.assertEqual(gcd(120, 84, 102), 6)
        self.assertEqual(gcd(120, 1, 84), 1)

        self.assertRaises(TypeError, gcd, 120.0)
        self.assertRaises(TypeError, gcd, 120.0, 84)
        self.assertRaises(TypeError, gcd, 120, 84.0)
        self.assertRaises(TypeError, gcd, 120, 1, 84.0)
        self.assertEqual(gcd(MyIndexable(120), MyIndexable(84)), 12)

    def testHypot(self):
        from decimal import Decimal
        from fractions import Fraction

        hypot = math.hypot

        # Test different numbers of arguments (from zero to five)
        # against a straightforward pure python implementation
        args = math.e, math.pi, math.sqrt(2.0), math.gamma(3.5), math.sin(2.1)
        for i in range(len(args)+1):
            self.assertAlmostEqual(
                hypot(*args[:i]),
                math.sqrt(sum(s**2 for s in args[:i]))
            )

        # Test allowable types (those with __float__)
        self.assertEqual(hypot(12.0, 5.0), 13.0)
        self.assertEqual(hypot(12, 5), 13)
        self.assertEqual(hypot(0.75, -1), 1.25)
        self.assertEqual(hypot(-1, 0.75), 1.25)
        self.assertEqual(hypot(0.75, FloatLike(-1.)), 1.25)
        self.assertEqual(hypot(FloatLike(-1.), 0.75), 1.25)
        self.assertEqual(hypot(Decimal(12), Decimal(5)), 13)
        self.assertEqual(hypot(Fraction(12, 32), Fraction(5, 32)), Fraction(13, 32))
        self.assertEqual(hypot(True, False, True, True, True), 2.0)

        # Test corner cases
        self.assertEqual(hypot(0.0, 0.0), 0.0)     # Max input is zero
        self.assertEqual(hypot(-10.5), 10.5)       # Negative input
        self.assertEqual(hypot(), 0.0)             # Negative input
        self.assertEqual(1.0,
            math.copysign(1.0, hypot(-0.0))        # Convert negative zero to positive zero
        )
        self.assertEqual(                          # Handling of moving max to the end
            hypot(1.5, 1.5, 0.5),
            hypot(1.5, 0.5, 1.5),
        )

        # Test handling of bad arguments
        with self.assertRaises(TypeError):         # Reject keyword args
            hypot(x=1)
        with self.assertRaises(TypeError):         # Reject values without __float__
            hypot(1.1, 'string', 2.2)
        int_too_big_for_float = 10 ** (sys.float_info.max_10_exp + 5)
        with self.assertRaises((ValueError, OverflowError)):
            hypot(1, int_too_big_for_float)

        # Any infinity gives positive infinity.
        self.assertEqual(hypot(INF), INF)
        self.assertEqual(hypot(0, INF), INF)
        self.assertEqual(hypot(10, INF), INF)
        self.assertEqual(hypot(-10, INF), INF)
        self.assertEqual(hypot(NAN, INF), INF)
        self.assertEqual(hypot(INF, NAN), INF)
        self.assertEqual(hypot(NINF, NAN), INF)
        self.assertEqual(hypot(NAN, NINF), INF)
        self.assertEqual(hypot(-INF, INF), INF)
        self.assertEqual(hypot(-INF, -INF), INF)
        self.assertEqual(hypot(10, -INF), INF)

        # If no infinity, any NaN gives a NaN.
        self.assertTrue(math.isnan(hypot(NAN)))
        self.assertTrue(math.isnan(hypot(0, NAN)))
        self.assertTrue(math.isnan(hypot(NAN, 10)))
        self.assertTrue(math.isnan(hypot(10, NAN)))
        self.assertTrue(math.isnan(hypot(NAN, NAN)))
        self.assertTrue(math.isnan(hypot(NAN)))

        # Verify scaling for extremely large values
        fourthmax = FLOAT_MAX / 4.0
        for n in range(32):
            self.assertTrue(math.isclose(hypot(*([fourthmax]*n)),
                                         fourthmax * math.sqrt(n)))

        # Verify scaling for extremely small values
        for exp in range(32):
            scale = FLOAT_MIN / 2.0 ** exp
            self.assertEqual(math.hypot(4*scale, 3*scale), 5*scale)

        self.assertRaises(TypeError, math.hypot, *([1.0]*18), 'spam')

    @requires_IEEE_754
    @unittest.skipIf(HAVE_DOUBLE_ROUNDING,
                     "hypot() loses accuracy on machines with double rounding")
    def testHypotAccuracy(self):
        # Verify improved accuracy in cases that were known to be inaccurate.
        #
        # The new algorithm's accuracy depends on IEEE 754 arithmetic
        # guarantees, on having the usual ROUND HALF EVEN rounding mode, on
        # the system not having double rounding due to extended precision,
        # and on the compiler maintaining the specified order of operations.
        #
        # This test is known to succeed on most of our builds.  If it fails
        # some build, we either need to add another skipIf if the cause is
        # identifiable; otherwise, we can remove this test entirely.

        hypot = math.hypot
        Decimal = decimal.Decimal
        high_precision = decimal.Context(prec=500)

        for hx, hy in [
            # Cases with a 1 ulp error in Python 3.7 compiled with Clang
            ('0x1.10e89518dca48p+29', '0x1.1970f7565b7efp+30'),
            ('0x1.10106eb4b44a2p+29', '0x1.ef0596cdc97f8p+29'),
            ('0x1.459c058e20bb7p+30', '0x1.993ca009b9178p+29'),
            ('0x1.378371ae67c0cp+30', '0x1.fbe6619854b4cp+29'),
            ('0x1.f4cd0574fb97ap+29', '0x1.50fe31669340ep+30'),
            ('0x1.494b2cdd3d446p+29', '0x1.212a5367b4c7cp+29'),
            ('0x1.f84e649f1e46dp+29', '0x1.1fa56bef8eec4p+30'),
            ('0x1.2e817edd3d6fap+30', '0x1.eb0814f1e9602p+29'),
            ('0x1.0d3a6e3d04245p+29', '0x1.32a62fea52352p+30'),
            ('0x1.888e19611bfc5p+29', '0x1.52b8e70b24353p+29'),

            # Cases with 2 ulp error in Python 3.8
            ('0x1.538816d48a13fp+29', '0x1.7967c5ca43e16p+29'),
            ('0x1.57b47b7234530p+29', '0x1.74e2c7040e772p+29'),
            ('0x1.821b685e9b168p+30', '0x1.677dc1c1e3dc6p+29'),
            ('0x1.9e8247f67097bp+29', '0x1.24bd2dc4f4baep+29'),
            ('0x1.b73b59e0cb5f9p+29', '0x1.da899ab784a97p+28'),
            ('0x1.94a8d2842a7cfp+30', '0x1.326a51d4d8d8ap+30'),
            ('0x1.e930b9cd99035p+29', '0x1.5a1030e18dff9p+30'),
            ('0x1.1592bbb0e4690p+29', '0x1.a9c337b33fb9ap+29'),
            ('0x1.1243a50751fd4p+29', '0x1.a5a10175622d9p+29'),
            ('0x1.57a8596e74722p+30', '0x1.42d1af9d04da9p+30'),

            # Cases with 1 ulp error in version fff3c28052e6b0
            ('0x1.ee7dbd9565899p+29', '0x1.7ab4d6fc6e4b4p+29'),
            ('0x1.5c6bfbec5c4dcp+30', '0x1.02511184b4970p+30'),
            ('0x1.59dcebba995cap+30', '0x1.50ca7e7c38854p+29'),
            ('0x1.768cdd94cf5aap+29', '0x1.9cfdc5571d38ep+29'),
            ('0x1.dcf137d60262ep+29', '0x1.1101621990b3ep+30'),
            ('0x1.3a2d006e288b0p+30', '0x1.e9a240914326cp+29'),
            ('0x1.62a32f7f53c61p+29', '0x1.47eb6cd72684fp+29'),
            ('0x1.d3bcb60748ef2p+29', '0x1.3f13c4056312cp+30'),
            ('0x1.282bdb82f17f3p+30', '0x1.640ba4c4eed3ap+30'),
            ('0x1.89d8c423ea0c6p+29', '0x1.d35dcfe902bc3p+29'),
        ]:
            x = float.fromhex(hx)
            y = float.fromhex(hy)
            with self.subTest(hx=hx, hy=hy, x=x, y=y):
                with decimal.localcontext(high_precision):
                    z = float((Decimal(x)**2 + Decimal(y)**2).sqrt())
                self.assertEqual(hypot(x, y), z)

    def testDist(self):
        from decimal import Decimal as D
        from fractions import Fraction as F

        dist = math.dist
        sqrt = math.sqrt

        # Simple exact cases
        self.assertEqual(dist((1.0, 2.0, 3.0), (4.0, 2.0, -1.0)), 5.0)
        self.assertEqual(dist((1, 2, 3), (4, 2, -1)), 5.0)

        # Test different numbers of arguments (from zero to nine)
        # against a straightforward pure python implementation
        for i in range(9):
            for j in range(5):
                p = tuple(random.uniform(-5, 5) for k in range(i))
                q = tuple(random.uniform(-5, 5) for k in range(i))
                self.assertAlmostEqual(
                    dist(p, q),
                    sqrt(sum((px - qx) ** 2.0 for px, qx in zip(p, q)))
                )

        # Test non-tuple inputs
        self.assertEqual(dist([1.0, 2.0, 3.0], [4.0, 2.0, -1.0]), 5.0)
        self.assertEqual(dist(iter([1.0, 2.0, 3.0]), iter([4.0, 2.0, -1.0])), 5.0)

        # Test allowable types (those with __float__)
        self.assertEqual(dist((14.0, 1.0), (2.0, -4.0)), 13.0)
        self.assertEqual(dist((14, 1), (2, -4)), 13)
        self.assertEqual(dist((FloatLike(14.), 1), (2, -4)), 13)
        self.assertEqual(dist((11, 1), (FloatLike(-1.), -4)), 13)
        self.assertEqual(dist((14, FloatLike(-1.)), (2, -6)), 13)
        self.assertEqual(dist((14, -1), (2, -6)), 13)
        self.assertEqual(dist((D(14), D(1)), (D(2), D(-4))), D(13))
        self.assertEqual(dist((F(14, 32), F(1, 32)), (F(2, 32), F(-4, 32))),
                         F(13, 32))
        self.assertEqual(dist((True, True, False, False, True, True),
                              (True, False, True, False, False, False)),
                         2.0)

        # Test corner cases
        self.assertEqual(dist((13.25, 12.5, -3.25),
                              (13.25, 12.5, -3.25)),
                         0.0)                      # Distance with self is zero
        self.assertEqual(dist((), ()), 0.0)        # Zero-dimensional case
        self.assertEqual(1.0,                      # Convert negative zero to positive zero
            math.copysign(1.0, dist((-0.0,), (0.0,)))
        )
        self.assertEqual(1.0,                      # Convert negative zero to positive zero
            math.copysign(1.0, dist((0.0,), (-0.0,)))
        )
        self.assertEqual(                          # Handling of moving max to the end
            dist((1.5, 1.5, 0.5), (0, 0, 0)),
            dist((1.5, 0.5, 1.5), (0, 0, 0))
        )

        # Verify tuple subclasses are allowed
        with torch._dynamo.error_on_graph_break(False):
            class T(tuple):
                pass
        self.assertEqual(dist(T((1, 2, 3)), ((4, 2, -1))), 5.0)

        # Test handling of bad arguments
        with self.assertRaises(TypeError):         # Reject keyword args
            dist(p=(1, 2, 3), q=(4, 5, 6))
        with self.assertRaises(TypeError):         # Too few args
            dist((1, 2, 3))
        with self.assertRaises(TypeError):         # Too many args
            dist((1, 2, 3), (4, 5, 6), (7, 8, 9))
        with self.assertRaises(TypeError):         # Scalars not allowed
            dist(1, 2)
        with self.assertRaises(TypeError):         # Reject values without __float__
            dist((1.1, 'string', 2.2), (1, 2, 3))
        with self.assertRaises(ValueError):        # Check dimension agree
            dist((1, 2, 3, 4), (5, 6, 7))
        with self.assertRaises(ValueError):        # Check dimension agree
            dist((1, 2, 3), (4, 5, 6, 7))
        with self.assertRaises(TypeError):
            dist((1,)*17 + ("spam",), (1,)*18)
        with self.assertRaises(TypeError):         # Rejects invalid types
            dist("abc", "xyz")
        int_too_big_for_float = 10 ** (sys.float_info.max_10_exp + 5)
        with self.assertRaises((ValueError, OverflowError)):
            dist((1, int_too_big_for_float), (2, 3))
        with self.assertRaises((ValueError, OverflowError)):
            dist((2, 3), (1, int_too_big_for_float))
        with self.assertRaises(TypeError):
            dist((1,), 2)
        with self.assertRaises(TypeError):
            dist([1], 2)

        with torch._dynamo.error_on_graph_break(False):
            class BadFloat:
                __float__ = BadDescr()

        with self.assertRaises(ValueError):
            dist([1], [BadFloat()])

        # Verify that the one dimensional case is equivalent to abs()
        for i in range(20):
            p, q = random.random(), random.random()
            self.assertEqual(dist((p,), (q,)), abs(p - q))

        # Test special values
        values = [NINF, -10.5, -0.0, 0.0, 10.5, INF, NAN]
        for p in itertools.product(values, repeat=3):
            for q in itertools.product(values, repeat=3):
                diffs = [px - qx for px, qx in zip(p, q)]
                if any(map(math.isinf, diffs)):
                    # Any infinite difference gives positive infinity.
                    self.assertEqual(dist(p, q), INF)
                elif any(map(math.isnan, diffs)):
                    # If no infinity, any NaN gives a NaN.
                    self.assertTrue(math.isnan(dist(p, q)))

        # Verify scaling for extremely large values
        fourthmax = FLOAT_MAX / 4.0
        for n in range(32):
            p = (fourthmax,) * n
            q = (0.0,) * n
            self.assertTrue(math.isclose(dist(p, q), fourthmax * math.sqrt(n)))
            self.assertTrue(math.isclose(dist(q, p), fourthmax * math.sqrt(n)))

        # Verify scaling for extremely small values
        for exp in range(32):
            scale = FLOAT_MIN / 2.0 ** exp
            p = (4*scale, 3*scale)
            q = (0.0, 0.0)
            self.assertEqual(math.dist(p, q), 5*scale)
            self.assertEqual(math.dist(q, p), 5*scale)

    def test_math_dist_leak(self):
        # gh-98897: Check for error handling does not leak memory
        with self.assertRaises(ValueError):
            math.dist([1, 2], [3, 4, 5])

    @slowTest
    def testIsqrt(self):
        # Test a variety of inputs, large and small.
        test_values = (
            list(range(1000))
            + list(range(10**6 - 1000, 10**6 + 1000))
            + [2**e + i for e in range(60, 200) for i in range(-40, 40)]
            + [3**9999, 10**5001]
        )

        for value in test_values:
            with self.subTest(value=value):
                s = math.isqrt(value)
                self.assertIs(type(s), int)
                self.assertLessEqual(s*s, value)
                self.assertLess(value, (s+1)*(s+1))

        # Negative values
        with self.assertRaises(ValueError):
            math.isqrt(-1)

        # Integer-like things
        s = math.isqrt(True)
        self.assertIs(type(s), int)
        self.assertEqual(s, 1)

        s = math.isqrt(False)
        self.assertIs(type(s), int)
        self.assertEqual(s, 0)

        with torch._dynamo.error_on_graph_break(False):
            class IntegerLike(object):
                def __init__(self, value):
                    self.value = value

                def __index__(self):
                    return self.value

        s = math.isqrt(IntegerLike(1729))
        self.assertIs(type(s), int)
        self.assertEqual(s, 41)

        with self.assertRaises(ValueError):
            math.isqrt(IntegerLike(-3))

        # Non-integer-like things
        bad_values = [
            3.5, "a string", decimal.Decimal("3.5"), 3.5j,
            100.0, -4.0,
        ]
        for value in bad_values:
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    math.isqrt(value)

    def test_lcm(self):
        lcm = math.lcm
        self.assertEqual(lcm(0, 0
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/dynamo/cpython/3_13`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/dynamo/cpython/3_13`, which is part of the **testing infrastructure**.



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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/dynamo/cpython/3_13/test_math.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/dynamo/cpython/3_13`):

- [`seq_tests.py_kw.md_docs.md`](./seq_tests.py_kw.md_docs.md)
- [`test_tuple.diff_kw.md_docs.md`](./test_tuple.diff_kw.md_docs.md)
- [`test_userdict.py_docs.md_docs.md`](./test_userdict.py_docs.md_docs.md)
- [`test_bool.diff_docs.md_docs.md`](./test_bool.diff_docs.md_docs.md)
- [`test_operator.py_docs.md_docs.md`](./test_operator.py_docs.md_docs.md)
- [`seq_tests.diff_docs.md_docs.md`](./seq_tests.diff_docs.md_docs.md)
- [`test_list.diff_kw.md_docs.md`](./test_list.diff_kw.md_docs.md)
- [`test_bool.py_docs.md_docs.md`](./test_bool.py_docs.md_docs.md)
- [`test_raise.py_docs.md_docs.md`](./test_raise.py_docs.md_docs.md)
- [`test_itertools.diff_kw.md_docs.md`](./test_itertools.diff_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_math.py_docs.md_docs.md`
- **Keyword Index**: `test_math.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
