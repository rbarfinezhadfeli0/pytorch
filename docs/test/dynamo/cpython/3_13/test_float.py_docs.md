# Documentation: `test/dynamo/cpython/3_13/test_float.py`

## File Metadata

- **Path**: `test/dynamo/cpython/3_13/test_float.py`
- **Size**: 71,283 bytes (69.61 KB)
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
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_float.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

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

import fractions
import operator
import os
import random
import sys
import struct
import time
import unittest

from test import support

VALID_UNDERSCORE_LITERALS = [
    '0_0_0',
    '4_2',
    '1_0000_0000',
    '0b1001_0100',
    '0xffff_ffff',
    '0o5_7_7',
    '1_00_00.5',
    '1_00_00.5e5',
    '1_00_00e5_1',
    '1e1_0',
    '.1_4',
    '.1_4e1',
    '0b_0',
    '0x_f',
    '0o_5',
    '1_00_00j',
    '1_00_00.5j',
    '1_00_00e5_1j',
    '.1_4j',
    '(1_2.5+3_3j)',
    '(.5_6j)',
]
INVALID_UNDERSCORE_LITERALS = [
    # Trailing underscores:
    '0_',
    '42_',
    '1.4j_',
    '0x_',
    '0b1_',
    '0xf_',
    '0o5_',
    '0 if 1_Else 1',
    # Underscores in the base selector:
    '0_b0',
    '0_xf',
    '0_o5',
    # Old-style octal, still disallowed:
    '0_7',
    '09_99',
    # Multiple consecutive underscores:
    '4_______2',
    '0.1__4',
    '0.1__4j',
    '0b1001__0100',
    '0xffff__ffff',
    '0x___',
    '0o5__77',
    '1e1__0',
    '1e1__0j',
    # Underscore right before a dot:
    '1_.4',
    '1_.4j',
    # Underscore right after a dot:
    '1._4',
    '1._4j',
    '._5',
    '._5j',
    # Underscore right after a sign:
    '1.0e+_1',
    '1.0e+_1j',
    # Underscore right before j:
    '1.4_j',
    '1.4e5_j',
    # Underscore right before e:
    '1_e1',
    '1.4_e1',
    '1.4_e1j',
    # Underscore right after e:
    '1e_1',
    '1.4e_1',
    '1.4e_1j',
    # Complex cases with parens:
    '(1+1.5_j_)',
    '(1+1.5_j)',
]

from math import isinf, isnan, copysign, ldexp
import math

try:
    import _testcapi
except ImportError:
    _testcapi = None

INF = float("inf")
NAN = float("nan")


#locate file with float format test values
test_dir = os.path.dirname(__file__) or os.curdir
format_testfile = os.path.join(test_dir, 'mathdata', 'formatfloat_testcases.txt')

class FloatSubclass(float):
    pass

class OtherFloatSubclass(float):
    pass

class GeneralFloatCases(__TestCase):

    def test_float(self):
        self.assertEqual(float(3.14), 3.14)
        self.assertEqual(float(314), 314.0)
        self.assertEqual(float("  3.14  "), 3.14)
        self.assertRaises(ValueError, float, "  0x3.1  ")
        self.assertRaises(ValueError, float, "  -0x3.p-1  ")
        self.assertRaises(ValueError, float, "  +0x3.p-1  ")
        self.assertRaises(ValueError, float, "++3.14")
        self.assertRaises(ValueError, float, "+-3.14")
        self.assertRaises(ValueError, float, "-+3.14")
        self.assertRaises(ValueError, float, "--3.14")
        self.assertRaises(ValueError, float, ".nan")
        self.assertRaises(ValueError, float, "+.inf")
        self.assertRaises(ValueError, float, ".")
        self.assertRaises(ValueError, float, "-.")
        self.assertRaises(TypeError, float, {})
        self.assertRaisesRegex(TypeError, "not 'dict'", float, {})
        # Lone surrogate
        self.assertRaises(ValueError, float, '\uD8F0')
        # check that we don't accept alternate exponent markers
        self.assertRaises(ValueError, float, "-1.7d29")
        self.assertRaises(ValueError, float, "3D-14")
        self.assertEqual(float("  \u0663.\u0661\u0664  "), 3.14)
        self.assertEqual(float("\N{EM SPACE}3.14\N{EN SPACE}"), 3.14)
        # extra long strings should not be a problem
        float(b'.' + b'1'*1000)
        float('.' + '1'*1000)
        # Invalid unicode string
        # See bpo-34087
        self.assertRaises(ValueError, float, '\u3053\u3093\u306b\u3061\u306f')

    def test_noargs(self):
        self.assertEqual(float(), 0.0)

    def test_underscores(self):
        for lit in VALID_UNDERSCORE_LITERALS:
            if not any(ch in lit for ch in 'jJxXoObB'):
                self.assertEqual(float(lit), eval(lit))
                self.assertEqual(float(lit), float(lit.replace('_', '')))
        for lit in INVALID_UNDERSCORE_LITERALS:
            if lit in ('0_7', '09_99'):  # octals are not recognized here
                continue
            if not any(ch in lit for ch in 'jJxXoObB'):
                self.assertRaises(ValueError, float, lit)
        # Additional test cases; nan and inf are never valid as literals,
        # only in the float() constructor, but we don't allow underscores
        # in or around them.
        self.assertRaises(ValueError, float, '_NaN')
        self.assertRaises(ValueError, float, 'Na_N')
        self.assertRaises(ValueError, float, 'IN_F')
        self.assertRaises(ValueError, float, '-_INF')
        self.assertRaises(ValueError, float, '-INF_')
        # Check that we handle bytes values correctly.
        self.assertRaises(ValueError, float, b'0_.\xff9')

    def test_non_numeric_input_types(self):
        # Test possible non-numeric types for the argument x, including
        # subclasses of the explicitly documented accepted types.
        with torch._dynamo.error_on_graph_break(False):
            class CustomStr(str): pass
            class CustomBytes(bytes): pass
            class CustomByteArray(bytearray): pass

        factories = [
            bytes,
            bytearray,
            lambda b: CustomStr(b.decode()),
            CustomBytes,
            CustomByteArray,
            memoryview,
        ]
        try:
            from array import array
        except ImportError:
            pass
        else:
            factories.append(lambda b: array('B', b))

        for f in factories:
            x = f(b" 3.14  ")
            with self.subTest(type(x)):
                self.assertEqual(float(x), 3.14)
                with self.assertRaisesRegex(ValueError, "could not convert"):
                    float(f(b'A' * 0x10))

    def test_float_memoryview(self):
        self.assertEqual(float(memoryview(b'12.3')[1:4]), 2.3)
        self.assertEqual(float(memoryview(b'12.3\x00')[1:4]), 2.3)
        self.assertEqual(float(memoryview(b'12.3 ')[1:4]), 2.3)
        self.assertEqual(float(memoryview(b'12.3A')[1:4]), 2.3)
        self.assertEqual(float(memoryview(b'12.34')[1:4]), 2.3)

    def test_error_message(self):
        def check(s):
            with self.assertRaises(ValueError, msg='float(%r)' % (s,)) as cm:
                float(s)
            self.assertEqual(str(cm.exception),
                'could not convert string to float: %r' % (s,))

        check('\xbd')
        check('123\xbd')
        check('  123 456  ')
        check(b'  123 456  ')
        # all whitespace (cf. https://github.com/python/cpython/issues/95605)
        check('')
        check(' ')
        check('\t \n')

        # non-ascii digits (error came from non-digit '!')
        check('\u0663\u0661\u0664!')
        # embedded NUL
        check('123\x00')
        check('123\x00 245')
        check('123\x00245')
        # byte string with embedded NUL
        check(b'123\x00')
        # non-UTF-8 byte string
        check(b'123\xa0')

    @support.run_with_locale('LC_NUMERIC', 'fr_FR', 'de_DE', '')
    def test_float_with_comma(self):
        # set locale to something that doesn't use '.' for the decimal point
        # float must not accept the locale specific decimal point but
        # it still has to accept the normal python syntax
        import locale
        if not locale.localeconv()['decimal_point'] == ',':
            self.skipTest('decimal_point is not ","')

        self.assertEqual(float("  3.14  "), 3.14)
        self.assertEqual(float("+3.14  "), 3.14)
        self.assertEqual(float("-3.14  "), -3.14)
        self.assertEqual(float(".14  "), .14)
        self.assertEqual(float("3.  "), 3.0)
        self.assertEqual(float("3.e3  "), 3000.0)
        self.assertEqual(float("3.2e3  "), 3200.0)
        self.assertEqual(float("2.5e-1  "), 0.25)
        self.assertEqual(float("5e-1"), 0.5)
        self.assertRaises(ValueError, float, "  3,14  ")
        self.assertRaises(ValueError, float, "  +3,14  ")
        self.assertRaises(ValueError, float, "  -3,14  ")
        self.assertRaises(ValueError, float, "  0x3.1  ")
        self.assertRaises(ValueError, float, "  -0x3.p-1  ")
        self.assertRaises(ValueError, float, "  +0x3.p-1  ")
        self.assertEqual(float("  25.e-1  "), 2.5)
        self.assertAlmostEqual(float("  .25e-1  "), .025)

    def test_floatconversion(self):
        # Make sure that calls to __float__() work properly
        with torch._dynamo.error_on_graph_break(False):
            class Foo1(object):
                def __float__(self):
                    return 42.

            class Foo2(float):
                def __float__(self):
                    return 42.

            class Foo3(float):
                def __new__(cls, value=0.):
                    return float.__new__(cls, 2*value)

                def __float__(self):
                    return self

            class Foo4(float):
                def __float__(self):
                    return 42

            # Issue 5759: __float__ not called on str subclasses (though it is on
            # unicode subclasses).
            class FooStr(str):
                def __float__(self):
                    return float(str(self)) + 1

        self.assertEqual(float(Foo1()), 42.)
        self.assertEqual(float(Foo2()), 42.)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(float(Foo3(21)), 42.)
        self.assertRaises(TypeError, float, Foo4(42))
        self.assertEqual(float(FooStr('8')), 9.)

        with torch._dynamo.error_on_graph_break(False):
            class Foo5:
                def __float__(self):
                    return ""
        self.assertRaises(TypeError, time.sleep, Foo5())

        with torch._dynamo.error_on_graph_break(False):
            # Issue #24731
            class F:
                def __float__(self):
                    return OtherFloatSubclass(42.)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(float(F()), 42.)
        with self.assertWarns(DeprecationWarning):
            self.assertIs(type(float(F())), float)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(FloatSubclass(F()), 42.)
        with self.assertWarns(DeprecationWarning):
            self.assertIs(type(FloatSubclass(F())), FloatSubclass)

        with torch._dynamo.error_on_graph_break(False):
            class MyIndex:
                def __init__(self, value):
                    self.value = value
                def __index__(self):
                    return self.value

        self.assertEqual(float(MyIndex(42)), 42.0)
        self.assertRaises(OverflowError, float, MyIndex(2**2000))

        with torch._dynamo.error_on_graph_break(False):
            class MyInt:
                def __int__(self):
                    return 42

        self.assertRaises(TypeError, float, MyInt())

    def test_keyword_args(self):
        with self.assertRaisesRegex(TypeError, 'keyword argument'):
            float(x='3.14')

    def test_keywords_in_subclass(self):
        with torch._dynamo.error_on_graph_break(False):
            class subclass(float):
                pass
        u = subclass(2.5)
        self.assertIs(type(u), subclass)
        self.assertEqual(float(u), 2.5)
        with self.assertRaises(TypeError):
            subclass(x=0)

        with torch._dynamo.error_on_graph_break(False):
            class subclass_with_init(float):
                def __init__(self, arg, newarg=None):
                    self.newarg = newarg
        u = subclass_with_init(2.5, newarg=3)
        self.assertIs(type(u), subclass_with_init)
        self.assertEqual(float(u), 2.5)
        self.assertEqual(u.newarg, 3)

        with torch._dynamo.error_on_graph_break(False):
            class subclass_with_new(float):
                def __new__(cls, arg, newarg=None):
                    self = super().__new__(cls, arg)
                    self.newarg = newarg
                    return self
        u = subclass_with_new(2.5, newarg=3)
        self.assertIs(type(u), subclass_with_new)
        self.assertEqual(float(u), 2.5)
        self.assertEqual(u.newarg, 3)

    def test_is_integer(self):
        self.assertFalse((1.1).is_integer())
        self.assertTrue((1.).is_integer())
        self.assertFalse(float("nan").is_integer())
        self.assertFalse(float("inf").is_integer())

    def test_floatasratio(self):
        for f, ratio in [
                (0.875, (7, 8)),
                (-0.875, (-7, 8)),
                (0.0, (0, 1)),
                (11.5, (23, 2)),
            ]:
            self.assertEqual(f.as_integer_ratio(), ratio)

        for i in range(10000):
            f = random.random()
            f *= 10 ** random.randint(-100, 100)
            n, d = f.as_integer_ratio()
            self.assertEqual(float(n).__truediv__(d), f)

        R = fractions.Fraction
        self.assertEqual(R(0, 1),
                         R(*float(0.0).as_integer_ratio()))
        self.assertEqual(R(5, 2),
                         R(*float(2.5).as_integer_ratio()))
        self.assertEqual(R(1, 2),
                         R(*float(0.5).as_integer_ratio()))
        self.assertEqual(R(4728779608739021, 2251799813685248),
                         R(*float(2.1).as_integer_ratio()))
        self.assertEqual(R(-4728779608739021, 2251799813685248),
                         R(*float(-2.1).as_integer_ratio()))
        self.assertEqual(R(-2100, 1),
                         R(*float(-2100.0).as_integer_ratio()))

        self.assertRaises(OverflowError, float('inf').as_integer_ratio)
        self.assertRaises(OverflowError, float('-inf').as_integer_ratio)
        self.assertRaises(ValueError, float('nan').as_integer_ratio)

    def test_float_containment(self):
        floats = (INF, -INF, 0.0, 1.0, NAN)
        for f in floats:
            self.assertIn(f, [f])
            self.assertIn(f, (f,))
            self.assertIn(f, {f})
            self.assertIn(f, {f: None})
            self.assertEqual([f].count(f), 1, "[].count('%r') != 1" % f)
            self.assertIn(f, floats)

        for f in floats:
            # nonidentical containers, same type, same contents
            self.assertTrue([f] == [f], "[%r] != [%r]" % (f, f))
            self.assertTrue((f,) == (f,), "(%r,) != (%r,)" % (f, f))
            self.assertTrue({f} == {f}, "{%r} != {%r}" % (f, f))
            self.assertTrue({f : None} == {f: None}, "{%r : None} != "
                                                   "{%r : None}" % (f, f))

            # identical containers
            l, t, s, d = [f], (f,), {f}, {f: None}
            self.assertTrue(l == l, "[%r] not equal to itself" % f)
            self.assertTrue(t == t, "(%r,) not equal to itself" % f)
            self.assertTrue(s == s, "{%r} not equal to itself" % f)
            self.assertTrue(d == d, "{%r : None} not equal to itself" % f)

    def assertEqualAndEqualSign(self, a, b):
        # fail unless a == b and a and b have the same sign bit;
        # the only difference from assertEqual is that this test
        # distinguishes -0.0 and 0.0.
        self.assertEqual((a, copysign(1.0, a)), (b, copysign(1.0, b)))

    def test_float_floor(self):
        self.assertIsInstance(float(0.5).__floor__(), int)
        self.assertEqual(float(0.5).__floor__(), 0)
        self.assertEqual(float(1.0).__floor__(), 1)
        self.assertEqual(float(1.5).__floor__(), 1)
        self.assertEqual(float(-0.5).__floor__(), -1)
        self.assertEqual(float(-1.0).__floor__(), -1)
        self.assertEqual(float(-1.5).__floor__(), -2)
        self.assertEqual(float(1.23e167).__floor__(), 1.23e167)
        self.assertEqual(float(-1.23e167).__floor__(), -1.23e167)
        self.assertRaises(ValueError, float("nan").__floor__)
        self.assertRaises(OverflowError, float("inf").__floor__)
        self.assertRaises(OverflowError, float("-inf").__floor__)

    def test_float_ceil(self):
        self.assertIsInstance(float(0.5).__ceil__(), int)
        self.assertEqual(float(0.5).__ceil__(), 1)
        self.assertEqual(float(1.0).__ceil__(), 1)
        self.assertEqual(float(1.5).__ceil__(), 2)
        self.assertEqual(float(-0.5).__ceil__(), 0)
        self.assertEqual(float(-1.0).__ceil__(), -1)
        self.assertEqual(float(-1.5).__ceil__(), -1)
        self.assertEqual(float(1.23e167).__ceil__(), 1.23e167)
        self.assertEqual(float(-1.23e167).__ceil__(), -1.23e167)
        self.assertRaises(ValueError, float("nan").__ceil__)
        self.assertRaises(OverflowError, float("inf").__ceil__)
        self.assertRaises(OverflowError, float("-inf").__ceil__)

    @support.requires_IEEE_754
    def test_float_mod(self):
        # Check behaviour of % operator for IEEE 754 special cases.
        # In particular, check signs of zeros.
        mod = operator.mod

        self.assertEqualAndEqualSign(mod(-1.0, 1.0), 0.0)
        self.assertEqualAndEqualSign(mod(-1e-100, 1.0), 1.0)
        self.assertEqualAndEqualSign(mod(-0.0, 1.0), 0.0)
        self.assertEqualAndEqualSign(mod(0.0, 1.0), 0.0)
        self.assertEqualAndEqualSign(mod(1e-100, 1.0), 1e-100)
        self.assertEqualAndEqualSign(mod(1.0, 1.0), 0.0)

        self.assertEqualAndEqualSign(mod(-1.0, -1.0), -0.0)
        self.assertEqualAndEqualSign(mod(-1e-100, -1.0), -1e-100)
        self.assertEqualAndEqualSign(mod(-0.0, -1.0), -0.0)
        self.assertEqualAndEqualSign(mod(0.0, -1.0), -0.0)
        self.assertEqualAndEqualSign(mod(1e-100, -1.0), -1.0)
        self.assertEqualAndEqualSign(mod(1.0, -1.0), -0.0)

    @support.requires_IEEE_754
    def test_float_pow(self):
        # test builtin pow and ** operator for IEEE 754 special cases.
        # Special cases taken from section F.9.4.4 of the C99 specification

        for pow_op in pow, operator.pow:
            # x**NAN is NAN for any x except 1
            self.assertTrue(isnan(pow_op(-INF, NAN)))
            self.assertTrue(isnan(pow_op(-2.0, NAN)))
            self.assertTrue(isnan(pow_op(-1.0, NAN)))
            self.assertTrue(isnan(pow_op(-0.5, NAN)))
            self.assertTrue(isnan(pow_op(-0.0, NAN)))
            self.assertTrue(isnan(pow_op(0.0, NAN)))
            self.assertTrue(isnan(pow_op(0.5, NAN)))
            self.assertTrue(isnan(pow_op(2.0, NAN)))
            self.assertTrue(isnan(pow_op(INF, NAN)))
            self.assertTrue(isnan(pow_op(NAN, NAN)))

            # NAN**y is NAN for any y except +-0
            self.assertTrue(isnan(pow_op(NAN, -INF)))
            self.assertTrue(isnan(pow_op(NAN, -2.0)))
            self.assertTrue(isnan(pow_op(NAN, -1.0)))
            self.assertTrue(isnan(pow_op(NAN, -0.5)))
            self.assertTrue(isnan(pow_op(NAN, 0.5)))
            self.assertTrue(isnan(pow_op(NAN, 1.0)))
            self.assertTrue(isnan(pow_op(NAN, 2.0)))
            self.assertTrue(isnan(pow_op(NAN, INF)))

            # (+-0)**y raises ZeroDivisionError for y a negative odd integer
            self.assertRaises(ZeroDivisionError, pow_op, -0.0, -1.0)
            self.assertRaises(ZeroDivisionError, pow_op, 0.0, -1.0)

            # (+-0)**y raises ZeroDivisionError for y finite and negative
            # but not an odd integer
            self.assertRaises(ZeroDivisionError, pow_op, -0.0, -2.0)
            self.assertRaises(ZeroDivisionError, pow_op, -0.0, -0.5)
            self.assertRaises(ZeroDivisionError, pow_op, 0.0, -2.0)
            self.assertRaises(ZeroDivisionError, pow_op, 0.0, -0.5)

            # (+-0)**y is +-0 for y a positive odd integer
            self.assertEqualAndEqualSign(pow_op(-0.0, 1.0), -0.0)
            self.assertEqualAndEqualSign(pow_op(0.0, 1.0), 0.0)

            # (+-0)**y is 0 for y finite and positive but not an odd integer
            self.assertEqualAndEqualSign(pow_op(-0.0, 0.5), 0.0)
            self.assertEqualAndEqualSign(pow_op(-0.0, 2.0), 0.0)
            self.assertEqualAndEqualSign(pow_op(0.0, 0.5), 0.0)
            self.assertEqualAndEqualSign(pow_op(0.0, 2.0), 0.0)

            # (-1)**+-inf is 1
            self.assertEqualAndEqualSign(pow_op(-1.0, -INF), 1.0)
            self.assertEqualAndEqualSign(pow_op(-1.0, INF), 1.0)

            # 1**y is 1 for any y, even if y is an infinity or nan
            self.assertEqualAndEqualSign(pow_op(1.0, -INF), 1.0)
            self.assertEqualAndEqualSign(pow_op(1.0, -2.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(1.0, -1.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(1.0, -0.5), 1.0)
            self.assertEqualAndEqualSign(pow_op(1.0, -0.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(1.0, 0.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(1.0, 0.5), 1.0)
            self.assertEqualAndEqualSign(pow_op(1.0, 1.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(1.0, 2.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(1.0, INF), 1.0)
            self.assertEqualAndEqualSign(pow_op(1.0, NAN), 1.0)

            # x**+-0 is 1 for any x, even if x is a zero, infinity, or nan
            self.assertEqualAndEqualSign(pow_op(-INF, 0.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(-2.0, 0.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(-1.0, 0.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(-0.5, 0.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(-0.0, 0.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(0.0, 0.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(0.5, 0.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(1.0, 0.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(2.0, 0.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(INF, 0.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(NAN, 0.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(-INF, -0.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(-2.0, -0.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(-1.0, -0.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(-0.5, -0.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(-0.0, -0.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(0.0, -0.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(0.5, -0.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(1.0, -0.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(2.0, -0.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(INF, -0.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(NAN, -0.0), 1.0)

            # x**y defers to complex pow for finite negative x and
            # non-integral y.
            self.assertEqual(type(pow_op(-2.0, -0.5)), complex)
            self.assertEqual(type(pow_op(-2.0, 0.5)), complex)
            self.assertEqual(type(pow_op(-1.0, -0.5)), complex)
            self.assertEqual(type(pow_op(-1.0, 0.5)), complex)
            self.assertEqual(type(pow_op(-0.5, -0.5)), complex)
            self.assertEqual(type(pow_op(-0.5, 0.5)), complex)

            # x**-INF is INF for abs(x) < 1
            self.assertEqualAndEqualSign(pow_op(-0.5, -INF), INF)
            self.assertEqualAndEqualSign(pow_op(-0.0, -INF), INF)
            self.assertEqualAndEqualSign(pow_op(0.0, -INF), INF)
            self.assertEqualAndEqualSign(pow_op(0.5, -INF), INF)

            # x**-INF is 0 for abs(x) > 1
            self.assertEqualAndEqualSign(pow_op(-INF, -INF), 0.0)
            self.assertEqualAndEqualSign(pow_op(-2.0, -INF), 0.0)
            self.assertEqualAndEqualSign(pow_op(2.0, -INF), 0.0)
            self.assertEqualAndEqualSign(pow_op(INF, -INF), 0.0)

            # x**INF is 0 for abs(x) < 1
            self.assertEqualAndEqualSign(pow_op(-0.5, INF), 0.0)
            self.assertEqualAndEqualSign(pow_op(-0.0, INF), 0.0)
            self.assertEqualAndEqualSign(pow_op(0.0, INF), 0.0)
            self.assertEqualAndEqualSign(pow_op(0.5, INF), 0.0)

            # x**INF is INF for abs(x) > 1
            self.assertEqualAndEqualSign(pow_op(-INF, INF), INF)
            self.assertEqualAndEqualSign(pow_op(-2.0, INF), INF)
            self.assertEqualAndEqualSign(pow_op(2.0, INF), INF)
            self.assertEqualAndEqualSign(pow_op(INF, INF), INF)

            # (-INF)**y is -0.0 for y a negative odd integer
            self.assertEqualAndEqualSign(pow_op(-INF, -1.0), -0.0)

            # (-INF)**y is 0.0 for y negative but not an odd integer
            self.assertEqualAndEqualSign(pow_op(-INF, -0.5), 0.0)
            self.assertEqualAndEqualSign(pow_op(-INF, -2.0), 0.0)

            # (-INF)**y is -INF for y a positive odd integer
            self.assertEqualAndEqualSign(pow_op(-INF, 1.0), -INF)

            # (-INF)**y is INF for y positive but not an odd integer
            self.assertEqualAndEqualSign(pow_op(-INF, 0.5), INF)
            self.assertEqualAndEqualSign(pow_op(-INF, 2.0), INF)

            # INF**y is INF for y positive
            self.assertEqualAndEqualSign(pow_op(INF, 0.5), INF)
            self.assertEqualAndEqualSign(pow_op(INF, 1.0), INF)
            self.assertEqualAndEqualSign(pow_op(INF, 2.0), INF)

            # INF**y is 0.0 for y negative
            self.assertEqualAndEqualSign(pow_op(INF, -2.0), 0.0)
            self.assertEqualAndEqualSign(pow_op(INF, -1.0), 0.0)
            self.assertEqualAndEqualSign(pow_op(INF, -0.5), 0.0)

            # basic checks not covered by the special cases above
            self.assertEqualAndEqualSign(pow_op(-2.0, -2.0), 0.25)
            self.assertEqualAndEqualSign(pow_op(-2.0, -1.0), -0.5)
            self.assertEqualAndEqualSign(pow_op(-2.0, -0.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(-2.0, 0.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(-2.0, 1.0), -2.0)
            self.assertEqualAndEqualSign(pow_op(-2.0, 2.0), 4.0)
            self.assertEqualAndEqualSign(pow_op(-1.0, -2.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(-1.0, -1.0), -1.0)
            self.assertEqualAndEqualSign(pow_op(-1.0, -0.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(-1.0, 0.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(-1.0, 1.0), -1.0)
            self.assertEqualAndEqualSign(pow_op(-1.0, 2.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(2.0, -2.0), 0.25)
            self.assertEqualAndEqualSign(pow_op(2.0, -1.0), 0.5)
            self.assertEqualAndEqualSign(pow_op(2.0, -0.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(2.0, 0.0), 1.0)
            self.assertEqualAndEqualSign(pow_op(2.0, 1.0), 2.0)
            self.assertEqualAndEqualSign(pow_op(2.0, 2.0), 4.0)

            # 1 ** large and -1 ** large; some libms apparently
            # have problems with these
            self.assertEqualAndEqualSign(pow_op(1.0, -1e100), 1.0)
            self.assertEqualAndEqualSign(pow_op(1.0, 1e100), 1.0)
            self.assertEqualAndEqualSign(pow_op(-1.0, -1e100), 1.0)
            self.assertEqualAndEqualSign(pow_op(-1.0, 1e100), 1.0)

            # check sign for results that underflow to 0
            self.assertEqualAndEqualSign(pow_op(-2.0, -2000.0), 0.0)
            self.assertEqual(type(pow_op(-2.0, -2000.5)), complex)
            self.assertEqualAndEqualSign(pow_op(-2.0, -2001.0), -0.0)
            self.assertEqualAndEqualSign(pow_op(2.0, -2000.0), 0.0)
            self.assertEqualAndEqualSign(pow_op(2.0, -2000.5), 0.0)
            self.assertEqualAndEqualSign(pow_op(2.0, -2001.0), 0.0)
            self.assertEqualAndEqualSign(pow_op(-0.5, 2000.0), 0.0)
            self.assertEqual(type(pow_op(-0.5, 2000.5)), complex)
            self.assertEqualAndEqualSign(pow_op(-0.5, 2001.0), -0.0)
            self.assertEqualAndEqualSign(pow_op(0.5, 2000.0), 0.0)
            self.assertEqualAndEqualSign(pow_op(0.5, 2000.5), 0.0)
            self.assertEqualAndEqualSign(pow_op(0.5, 2001.0), 0.0)

            # check we don't raise an exception for subnormal results,
            # and validate signs.  Tests currently disabled, since
            # they fail on systems where a subnormal result from pow
            # is flushed to zero (e.g. Debian/ia64.)
            #self.assertTrue(0.0 < pow_op(0.5, 1048) < 1e-315)
            #self.assertTrue(0.0 < pow_op(-0.5, 1048) < 1e-315)
            #self.assertTrue(0.0 < pow_op(0.5, 1047) < 1e-315)
            #self.assertTrue(0.0 > pow_op(-0.5, 1047) > -1e-315)
            #self.assertTrue(0.0 < pow_op(2.0, -1048) < 1e-315)
            #self.assertTrue(0.0 < pow_op(-2.0, -1048) < 1e-315)
            #self.assertTrue(0.0 < pow_op(2.0, -1047) < 1e-315)
            #self.assertTrue(0.0 > pow_op(-2.0, -1047) > -1e-315)

    def test_hash(self):
        for x in range(-30, 30):
            self.assertEqual(hash(float(x)), hash(x))
        self.assertEqual(hash(float(sys.float_info.max)),
                         hash(int(sys.float_info.max)))
        self.assertEqual(hash(float('inf')), sys.hash_info.inf)
        self.assertEqual(hash(float('-inf')), -sys.hash_info.inf)

    def test_hash_nan(self):
        value = float('nan')
        self.assertEqual(hash(value), object.__hash__(value))
        with torch._dynamo.error_on_graph_break(False):
            class H:
                def __hash__(self):
                    return 42
            class F(float, H):
                pass
        value = F('nan')
        self.assertEqual(hash(value), object.__hash__(value))


@unittest.skipUnless(hasattr(float, "__getformat__"), "requires __getformat__")
class FormatFunctionsTestCase(__TestCase):
    def test_getformat(self):
        self.assertIn(float.__getformat__('double'),
                      ['unknown', 'IEEE, big-endian', 'IEEE, little-endian'])
        self.assertIn(float.__getformat__('float'),
                      ['unknown', 'IEEE, big-endian', 'IEEE, little-endian'])
        self.assertRaises(ValueError, float.__getformat__, 'chicken')
        self.assertRaises(TypeError, float.__getformat__, 1)


BE_DOUBLE_INF = b'\x7f\xf0\x00\x00\x00\x00\x00\x00'
LE_DOUBLE_INF = bytes(reversed(BE_DOUBLE_INF))
BE_DOUBLE_NAN = b'\x7f\xf8\x00\x00\x00\x00\x00\x00'
LE_DOUBLE_NAN = bytes(reversed(BE_DOUBLE_NAN))

BE_FLOAT_INF = b'\x7f\x80\x00\x00'
LE_FLOAT_INF = bytes(reversed(BE_FLOAT_INF))
BE_FLOAT_NAN = b'\x7f\xc0\x00\x00'
LE_FLOAT_NAN = bytes(reversed(BE_FLOAT_NAN))

# on an IEEE platform, all we guarantee is that bit patterns
# representing infinities or NaNs do not raise an exception; all else
# is accident (today).
# let's also try to guarantee that -0.0 and 0.0 don't get confused.

class IEEEFormatTestCase(__TestCase):

    @support.requires_IEEE_754
    def test_double_specials_do_unpack(self):
        for fmt, data in [('>d', BE_DOUBLE_INF),
                          ('>d', BE_DOUBLE_NAN),
                          ('<d', LE_DOUBLE_INF),
                          ('<d', LE_DOUBLE_NAN)]:
            struct.unpack(fmt, data)

    @support.requires_IEEE_754
    def test_float_specials_do_unpack(self):
        for fmt, data in [('>f', BE_FLOAT_INF),
                          ('>f', BE_FLOAT_NAN),
                          ('<f', LE_FLOAT_INF),
                          ('<f', LE_FLOAT_NAN)]:
            struct.unpack(fmt, data)

    @support.requires_IEEE_754
    @unittest.skipIf(_testcapi is None, 'needs _testcapi')
    def test_serialized_float_rounding(self):
        FLT_MAX = _testcapi.FLT_MAX
        self.assertEqual(struct.pack("<f", 3.40282356e38), struct.pack("<f", FLT_MAX))
        self.assertEqual(struct.pack("<f", -3.40282356e38), struct.pack("<f", -FLT_MAX))

class FormatTestCase(__TestCase):

    def test_format(self):
        # these should be rewritten to use both format(x, spec) and
        # x.__format__(spec)

        self.assertEqual(format(0.0, 'f'), '0.000000')

        # the default is 'g', except for empty format spec
        self.assertEqual(format(0.0, ''), '0.0')
        self.assertEqual(format(0.01, ''), '0.01')
        self.assertEqual(format(0.01, 'g'), '0.01')

        # empty presentation type should format in the same way as str
        # (issue 5920)
        x = 100/7.
        self.assertEqual(format(x, ''), str(x))
        self.assertEqual(format(x, '-'), str(x))
        self.assertEqual(format(x, '>'), str(x))
        self.assertEqual(format(x, '2'), str(x))

        self.assertEqual(format(1.0, 'f'), '1.000000')

        self.assertEqual(format(-1.0, 'f'), '-1.000000')

        self.assertEqual(format( 1.0, ' f'), ' 1.000000')
        self.assertEqual(format(-1.0, ' f'), '-1.000000')
        self.assertEqual(format( 1.0, '+f'), '+1.000000')
        self.assertEqual(format(-1.0, '+f'), '-1.000000')

        # % formatting
        self.assertEqual(format(-1.0, '%'), '-100.000000%')

        # conversion to string should fail
        self.assertRaises(ValueError, format, 3.0, "s")

        # confirm format options expected to fail on floats, such as integer
        # presentation types
        for format_spec in 'sbcdoxX':
            self.assertRaises(ValueError, format, 0.0, format_spec)
            self.assertRaises(ValueError, format, 1.0, format_spec)
            self.assertRaises(ValueError, format, -1.0, format_spec)
            self.assertRaises(ValueError, format, 1e100, format_spec)
            self.assertRaises(ValueError, format, -1e100, format_spec)
            self.assertRaises(ValueError, format, 1e-100, format_spec)
            self.assertRaises(ValueError, format, -1e-100, format_spec)

        # issue 3382
        self.assertEqual(format(NAN, 'f'), 'nan')
        self.assertEqual(format(NAN, 'F'), 'NAN')
        self.assertEqual(format(INF, 'f'), 'inf')
        self.assertEqual(format(INF, 'F'), 'INF')

    @support.requires_IEEE_754
    def test_format_testfile(self):
        with open(format_testfile, encoding="utf-8") as testfile:
            for line in testfile:
                if line.startswith('--'):
                    continue
                line = line.strip()
                if not line:
                    continue

                lhs, rhs = map(str.strip, line.split('->'))
                fmt, arg = lhs.split()
                f = float(arg)
                self.assertEqual(fmt % f, rhs)
                self.assertEqual(fmt % -f, '-' + rhs)
                if fmt != '%r':
                    fmt2 = fmt[1:]
                    self.assertEqual(format(f, fmt2), rhs)
                    self.assertEqual(format(-f, fmt2), '-' + rhs)

    def test_issue5864(self):
        self.assertEqual(format(123.456, '.4'), '123.5')
        self.assertEqual(format(1234.56, '.4'), '1.235e+03')
        self.assertEqual(format(12345.6, '.4'), '1.235e+04')

    def test_issue35560(self):
        self.assertEqual(format(123.0, '00'), '123.0')
        self.assertEqual(format(123.34, '00f'), '123.340000')
        self.assertEqual(format(123.34, '00e'), '1.233400e+02')
        self.assertEqual(format(123.34, '00g'), '123.34')
        self.assertEqual(format(123.34, '00.10f'), '123.3400000000')
        self.assertEqual(format(123.34, '00.10e'), '1.2334000000e+02')
        self.assertEqual(format(123.34, '00.10g'), '123.34')
        self.assertEqual(format(123.34, '01f'), '123.340000')

        self.assertEqual(format(-123.0, '00'), '-123.0')
        self.assertEqual(format(-123.34, '00f'), '-123.340000')
        self.assertEqual(format(-123.34, '00e'), '-1.233400e+02')
        self.assertEqual(format(-123.34, '00g'), '-123.34')
        self.assertEqual(format(-123.34, '00.10f'), '-123.3400000000')
        self.assertEqual(format(-123.34, '00.10f'), '-123.3400000000')
        self.assertEqual(format(-123.34, '00.10e'), '-1.2334000000e+02')
        self.assertEqual(format(-123.34, '00.10g'), '-123.34')

class ReprTestCase(__TestCase):
    def test_repr(self):
        with open(os.path.join(os.path.split(__file__)[0],
                  'mathdata',
                  'floating_points.txt'), encoding="utf-8") as floats_file:
            for line in floats_file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                v = eval(line)
                self.assertEqual(v, eval(repr(v)))

    @unittest.skipUnless(getattr(sys, 'float_repr_style', '') == 'short',
                         "applies only when using short float repr style")
    def test_short_repr(self):
        # test short float repr introduced in Python 3.1.  One aspect
        # of this repr is that we get some degree of str -> float ->
        # str roundtripping.  In particular, for any numeric string
        # containing 15 or fewer significant digits, those exact same
        # digits (modulo trailing zeros) should appear in the output.
        # No more repr(0.03) -> "0.029999999999999999"!

        test_strings = [
            # output always includes *either* a decimal point and at
            # least one digit after that point, or an exponent.
            '0.0',
            '1.0',
            '0.01',
            '0.02',
            '0.03',
            '0.04',
            '0.05',
            '1.23456789',
            '10.0',
            '100.0',
            # values >= 1e16 get an exponent...
            '1000000000000000.0',
            '9999999999999990.0',
            '1e+16',
            '1e+17',
            # ... and so do values < 1e-4
            '0.001',
            '0.001001',
            '0.00010000000000001',
            '0.0001',
            '9.999999999999e-05',
            '1e-05',
            # values designed to provoke failure if the FPU rounding
            # precision isn't set correctly
            '8.72293771110361e+25',
            '7.47005307342313e+26',
            '2.86438000439698e+28',
            '8.89142905246179e+28',
            '3.08578087079232e+35',
            ]

        for s in test_strings:
            negs = '-'+s
            self.assertEqual(s, repr(float(s)))
            self.assertEqual(negs, repr(float(negs)))
            # Since Python 3.2, repr and str are identical
            self.assertEqual(repr(float(s)), str(float(s)))
            self.assertEqual(repr(float(negs)), str(float(negs)))

@support.requires_IEEE_754
class RoundTestCase(__TestCase):
    def assertFloatsAreIdentical(self, x, y):
        """assert that floats x and y are identical, in the sense that:
        (1) both x and y are nans, or
        (2) both x and y are infinities, with the same sign, or
        (3) both x and y are zeros, with the same sign, or
        (4) x and y are both finite and nonzero, and x == y

        """
        msg = 'floats {!r} and {!r} are not identical'

        if isnan(x) or isnan(y):
            if isnan(x) and isnan(y):
                return
        elif x == y:
            if x != 0.0:
                return
            # both zero; check that signs match
            elif copysign(1.0, x) == copysign(1.0, y):
                return
            else:
                msg += ': zeros have different signs'
        self.fail(msg.format(x, y))

    def test_inf_nan(self):
        self.assertRaises(OverflowError, round, INF)
        self.assertRaises(OverflowError, round, -INF)
        self.assertRaises(ValueError, round, NAN)
        self.assertRaises(TypeError, round, INF, 0.0)
        self.assertRaises(TypeError, round, -INF, 1.0)
        self.assertRaises(TypeError, round, NAN, "ceci n'est pas un integer")
        self.assertRaises(TypeError, round, -0.0, 1j)

    def test_inf_nan_ndigits(self):
        self.assertEqual(round(INF, 0), INF)
        self.assertEqual(round(-INF, 0), -INF)
        self.assertTrue(math.isnan(round(NAN, 0)))

    def test_large_n(self):
        for n in [324, 325, 400, 2**31-1, 2**31, 2**32, 2**100]:
            self.assertEqual(round(123.456, n), 123.456)
            self.assertEqual(round(-123.456, n), -123.456)
            self.assertEqual(round(1e300, n), 1e300)
            self.assertEqual(round(1e-320, n), 1e-320)
        self.assertEqual(round(1e150, 300), 1e150)
        self.assertEqual(round(1e300, 307), 1e300)
        self.assertEqual(round(-3.1415, 308), -3.1415)
        self.assertEqual(round(1e150, 309), 1e150)
        self.assertEqual(round(1.4e-315, 315), 1e-315)

    def test_small_n(self):
        for n in [-308, -309, -400, 1-2**31, -2**31, -2**31-1, -2**100]:
            self.assertFloatsAreIdentical(round(123.456, n), 0.0)
            self.assertFloatsAreIdentical(round(-123.456, n), -0.0)
            self.assertFloatsAreIdentical(round(1e300, n), 0.0)
            self.assertFloatsAreIdentical(round(1e-320, n), 0.0)

    def test_overflow(self):
        self.assertRaises(OverflowError, round, 1.6e308, -308)
        self.assertRaises(OverflowError, round, -1.7e308, -308)

    @unittest.skipUnless(getattr(sys, 'float_repr_style', '') == 'short',
                         "applies only when using short float repr style")
    def test_previous_round_bugs(self):
        # particular cases that have occurred in bug reports
        self.assertEqual(round(562949953421312.5, 1),
                          562949953421312.5)
        self.assertEqual(round(56294995342131.5, 3),
                         56294995342131.5)
        # round-half-even
        self.assertEqual(round(25.0, -1), 20.0)
        self.assertEqual(round(35.0, -1), 40.0)
        self.assertEqual(round(45.0, -1), 40.0)
        self.assertEqual(round(55.0, -1), 60.0)
        self.assertEqual(round(65.0, -1), 60.0)
        self.assertEqual(round(75.0, -1), 80.0)
        self.assertEqual(round(85.0, -1), 80.0)
        self.assertEqual(round(95.0, -1), 100.0)

    @unittest.skipUnless(getattr(sys, 'float_repr_style', '') == 'short',
                         "applies only when using short float repr style")
    def test_matches_float_format(self):
        # round should give the same results as float formatting
        for i in range(500):
            x = i/1000.
            self.assertEqual(float(format(x, '.0f')), round(x, 0))
            self.assertEqual(float(format(x, '.1f')), round(x, 1))
            self.assertEqual(float(format(x, '.2f')), round(x, 2))
            self.assertEqual(float(format(x, '.3f')), round(x, 3))

        for i in range(5, 5000, 10):
            x = i/1000.
            self.assertEqual(float(format(x, '.0f')), round(x, 0))
            self.assertEqual(float(format(x, '.1f')), round(x, 1))
            self.assertEqual(float(format(x, '.2f')), round(x, 2))
            self.assertEqual(float(format(x, '.3f')), round(x, 3))

        for i in range(500):
            x = random.random()
            self.assertEqual(float(format(x, '.0f')), round(x, 0))
            self.assertEqual(float(format(x, '.1f')), round(x, 1))
            self.assertEqual(float(format(x, '.2f')), round(x, 2))
            self.assertEqual(float(format(x, '.3f')), round(x, 3))

    def test_format_specials(self):
        # Test formatting of nans and infs.

        def test(fmt, value, expected):
            # Test with both % and format().
            self.assertEqual(fmt % value, expected, fmt)
            fmt = fmt[1:] # strip off the %
            self.assertEqual(format(value, fmt), expected, fmt)

        for fmt in ['%e', '%f', '%g', '%.0e', '%.6f', '%.20g',
                    '%#e', '%#f', '%#g', '%#.20e', '%#.15f', '%#.3g']:
            pfmt = '%+' + fmt[1:]
            sfmt = '% ' + fmt[1:]
            test(fmt, INF, 'inf')
            test(fmt, -INF, '-inf')
            test(fmt, NAN, 'nan')
            test(fmt, -NAN, 'nan')
            # When asking for a sign, it's always provided. nans are
            #  always positive.
            test(pfmt, INF, '+inf')
            test(pfmt, -INF, '-inf')
            test(pfmt, NAN, '+nan')
            test(pfmt, -NAN, '+nan')
            # When using ' ' for a sign code, only infs can be negative.
            #  Others have a space.
            test(sfmt, INF, ' inf')
            test(sfmt, -INF, '-inf')
            test(sfmt, NAN, ' nan')
            test(sfmt, -NAN, ' nan')

    def test_None_ndigits(self):
        for x in round(1.23), round(1.23, None), round(1.23, ndigits=None):
            self.assertEqual(x, 1)
            self.assertIsInstance(x, int)
        for x in round(1.78), round(1.78, None), round(1.78, ndigits=None):
            self.assertEqual(x, 2)
            self.assertIsInstance(x, int)


# Beginning with Python 2.6 float has cross platform compatible
# ways to create and represent inf and nan
class InfNanTest(__TestCase):
    def test_inf_from_str(self):
        self.assertTrue(isinf(float("inf")))
        self.assertTrue(isinf(float("+inf")))
        self.assertTrue(isinf(float("-inf")))
        self.assertTrue(isinf(float("infinity")))
        self.assertTrue(isinf(float("+infinity")))
        self.assertTrue(isinf(float("-infinity")))

        self.assertEqual(repr(float("inf")), "inf")
        self.assertEqual(repr(float("+inf")), "inf")
        self.assertEqual(repr(float("-inf")), "-inf")
        self.assertEqual(repr(float("infinity")), "inf")
        self.assertEqual(repr(float("+infinity")), "inf")
        self.assertEqual(repr(float("-infinity")), "-inf")

        self.assertEqual(repr(float("INF")), "inf")
        self.assertEqual(repr(float("+Inf")), "inf")
        self.assertEqual(repr(float("-iNF")), "-inf")
        self.assertEqual(repr(float("Infinity")), "inf")
        self.assertEqual(repr(float("+iNfInItY")), "inf")
        self.assertEqual(repr(float("-INFINITY")), "-inf")

        self.assertEqual(str(float("inf")), "inf")
        self.assertEqual(str(float("+inf")), "inf")
        self.assertEqual(str(float("-inf")), "-inf")
        self.assertEqual(str(float("infinity")), "inf")
        self.assertEqual(str(float("+infinity")), "inf")
        self.assertEqual(str(float("-infinity")), "-inf")

        self.assertRaises(ValueError, float, "info")
        self.assertRaises(ValueError, float, "+info")
        self.assertRaises(ValueError, float, "-info")
        self.assertRaises(ValueError, float, "in")
        self.assertRaises(ValueError, float, "+in")
        self.assertRaises(ValueError, float, "-in")
        self.assertRaises(ValueError, float, "infinit")
        self.assertRaises(ValueError, float, "+Infin")
        self.assertRaises(ValueError, float, "-INFI")
        self.assertRaises(ValueError, float, "infinitys")

        self.assertRaises(ValueError, float, "++Inf")
        self.assertRaises(ValueError, float, "-+inf")
        self.assertRaises(ValueError, float, "+-infinity")
        self.assertRaises(ValueError, float, "--Infinity")

    def test_inf_as_str(self):
        self.assertEqual(repr(1e300 * 1e300), "inf")
        self.assertEqual(repr(-1e300 * 1e300), "-inf")

        self.assertEqual(str(1e300 * 1e300), "inf")
        self.assertEqual(str(-1e300 * 1e300), "-inf")

    def test_nan_from_str(self):
        self.assertTrue(isnan(float("nan")))
        self.assertTrue(isnan(float("+nan")))
        self.assertTrue(isnan(float("-nan")))

        self.assertEqual(repr(float("nan")), "nan")
        self.assertEqual(repr(float("+nan")), "nan")
        self.assertEqual(repr(float("-nan")), "nan")

        self.assertEqual(repr(float("NAN")), "nan")
        self.assertEqual(repr(float("+NAn")), "nan")
        self.assertEqual(repr(float("-NaN")), "nan")

        self.assertEqual(str(float("nan")), "nan")
        self.assertEqual(str(float("+nan")), "nan")
        self.assertEqual(str(float("-nan")), "nan")

        self.assertRaises(ValueError, float, "nana")
        self.assertRaises(ValueError, float, "+nana")
        self.assertRaises(ValueError, float, "-nana")
        self.assertRaises(ValueError, float, "na")
        self.assertRaises(ValueError, float, "+na")
        self.assertRaises(ValueError, float, "-na")

        self.assertRaises(ValueError, float, "++nan")
        self.assertRaises(ValueError, float, "-+NAN")
        self.assertRaises(ValueError, float, "+-NaN")
        self.assertRaises(ValueError, float, "--nAn")

    def test_nan_as_str(self):
        self.assertEqual(repr(1e300 * 1e300 * 0), "nan")
        self.assertEqual(repr(-1e300 * 1e300 * 0), "nan")

        self.assertEqual(str(1e300 * 1e300 * 0), "nan")
        self.assertEqual(str(-1e300 * 1e300 * 0), "nan")

    def test_inf_signs(self):
        self.assertEqual(copysign(1.0, float('inf')), 1.0)
        self.assertEqual(copysign(1.0, float('-inf')), -1.0)

    def test_nan_signs(self):
        # The sign of float('nan') should be predictable.
        self.assertEqual(copysign(1.0, float('nan')), 1.0)
        self.assertEqual(copysign(1.0, float('-nan')), -1.0)


fromHex = float.fromhex
toHex = float.hex
class HexFloatTestCase(__TestCase):
    MAX = fromHex('0x.fffffffffffff8p+1024')  # max normal
    MIN = fromHex('0x1p-1022')                # min normal
    TINY = fromHex(
```



## High-Level Overview


This Python file contains 30 class(es) and 74 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `RedirectImportFinder`, `FloatSubclass`, `OtherFloatSubclass`, `GeneralFloatCases`, `CustomStr`, `CustomBytes`, `CustomByteArray`, `Foo1`, `Foo2`, `Foo3`, `Foo4`, `FooStr`, `Foo5`, `F`, `MyIndex`, `MyInt`, `subclass`, `subclass_with_init`, `subclass_with_new`, `H`

**Functions defined**: `find_spec`, `test_float`, `test_noargs`, `test_underscores`, `test_non_numeric_input_types`, `test_float_memoryview`, `test_error_message`, `check`, `test_float_with_comma`, `test_floatconversion`, `__float__`, `__float__`, `__new__`, `__float__`, `__float__`, `__float__`, `__float__`, `__float__`, `__init__`, `__index__`

**Key imports**: sys, torch, torch._dynamo.test_case, unittest, CPythonTestCase, run_tests, statements, sys, importlib.abc, is the problematic one


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo/cpython/3_13`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `torch`
- `torch._dynamo.test_case`
- `unittest`
- `torch.testing._internal.common_utils`: run_tests
- `statements`
- `importlib.abc`
- `is the problematic one`
- `the standalone module`
- `fractions`
- `operator`
- `os`
- `random`
- `struct`
- `time`
- `test`: support
- `math`: isinf, isnan, copysign, ldexp
- `_testcapi`
- `array`: array
- `locale`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/dynamo/cpython/3_13/test_float.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/dynamo/cpython/3_13`):

- [`mapping_tests.diff_docs.md`](./mapping_tests.diff_docs.md)
- [`test_generators.py_docs.md`](./test_generators.py_docs.md)
- [`test_dict.py_docs.md`](./test_dict.py_docs.md)
- [`test_generator_stop.diff_docs.md`](./test_generator_stop.diff_docs.md)
- [`test_sort.diff_docs.md`](./test_sort.diff_docs.md)
- [`test_list.diff_docs.md`](./test_list.diff_docs.md)
- [`test_userdict.diff_docs.md`](./test_userdict.diff_docs.md)
- [`test_generators.diff_docs.md`](./test_generators.diff_docs.md)
- [`test_userlist.py_docs.md`](./test_userlist.py_docs.md)


## Cross-References

- **File Documentation**: `test_float.py_docs.md`
- **Keyword Index**: `test_float.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
