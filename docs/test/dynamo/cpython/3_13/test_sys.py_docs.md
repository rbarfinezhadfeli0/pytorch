# Documentation: `test/dynamo/cpython/3_13/test_sys.py`

## File Metadata

- **Path**: `test/dynamo/cpython/3_13/test_sys.py`
- **Size**: 77,075 bytes (75.27 KB)
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
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_sys.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import (
    run_tests,
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

import builtins
import codecs
import _datetime
import gc
import io
import locale
import operator
import os
import random
import struct
import subprocess
import sys
import sysconfig
import test.support
from test import support
from test.support import os_helper
from test.support.script_helper import assert_python_ok, assert_python_failure
from test.support import threading_helper
from test.support import import_helper
from test.support import force_not_colorized
try:
    from test.support import interpreters
except ImportError:
    interpreters = None
import textwrap
import unittest
import warnings


def requires_subinterpreters(meth):
    """Decorator to skip a test if subinterpreters are not supported."""
    return unittest.skipIf(interpreters is None,
                           'subinterpreters required')(meth)


DICT_KEY_STRUCT_FORMAT = 'n2BI2n'

class DisplayHookTest(__TestCase):

    def test_original_displayhook(self):
        dh = sys.__displayhook__

        with support.captured_stdout() as out:
            dh(42)

        self.assertEqual(out.getvalue(), "42\n")
        self.assertEqual(builtins._, 42)

        del builtins._

        with support.captured_stdout() as out:
            dh(None)

        self.assertEqual(out.getvalue(), "")
        self.assertTrue(not hasattr(builtins, "_"))

        # sys.displayhook() requires arguments
        self.assertRaises(TypeError, dh)

        stdout = sys.stdout
        try:
            del sys.stdout
            self.assertRaises(RuntimeError, dh, 42)
        finally:
            sys.stdout = stdout

    def test_lost_displayhook(self):
        displayhook = sys.displayhook
        try:
            del sys.displayhook
            code = compile("42", "<string>", "single")
            self.assertRaises(RuntimeError, eval, code)
        finally:
            sys.displayhook = displayhook

    def test_custom_displayhook(self):
        def baddisplayhook(obj):
            raise ValueError

        with support.swap_attr(sys, 'displayhook', baddisplayhook):
            code = compile("42", "<string>", "single")
            self.assertRaises(ValueError, eval, code)


class ActiveExceptionTests(__TestCase):
    def test_exc_info_no_exception(self):
        self.assertEqual(sys.exc_info(), (None, None, None))

    def test_sys_exception_no_exception(self):
        self.assertEqual(sys.exception(), None)

    def test_exc_info_with_exception_instance(self):
        def f():
            raise ValueError(42)

        try:
            f()
        except Exception as e_:
            e = e_
            exc_info = sys.exc_info()

        self.assertIsInstance(e, ValueError)
        self.assertIs(exc_info[0], ValueError)
        self.assertIs(exc_info[1], e)
        self.assertIs(exc_info[2], e.__traceback__)

    def test_exc_info_with_exception_type(self):
        def f():
            raise ValueError

        try:
            f()
        except Exception as e_:
            e = e_
            exc_info = sys.exc_info()

        self.assertIsInstance(e, ValueError)
        self.assertIs(exc_info[0], ValueError)
        self.assertIs(exc_info[1], e)
        self.assertIs(exc_info[2], e.__traceback__)

    def test_sys_exception_with_exception_instance(self):
        def f():
            raise ValueError(42)

        try:
            f()
        except Exception as e_:
            e = e_
            exc = sys.exception()

        self.assertIsInstance(e, ValueError)
        self.assertIs(exc, e)

    def test_sys_exception_with_exception_type(self):
        def f():
            raise ValueError

        try:
            f()
        except Exception as e_:
            e = e_
            exc = sys.exception()

        self.assertIsInstance(e, ValueError)
        self.assertIs(exc, e)


class ExceptHookTest(__TestCase):

    @force_not_colorized
    def test_original_excepthook(self):
        try:
            raise ValueError(42)
        except ValueError as exc:
            with support.captured_stderr() as err:
                sys.__excepthook__(*sys.exc_info())

        self.assertTrue(err.getvalue().endswith("ValueError: 42\n"))

        self.assertRaises(TypeError, sys.__excepthook__)

    @force_not_colorized
    def test_excepthook_bytes_filename(self):
        # bpo-37467: sys.excepthook() must not crash if a filename
        # is a bytes string
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', BytesWarning)

            try:
                raise SyntaxError("msg", (b"bytes_filename", 123, 0, "text"))
            except SyntaxError as exc:
                with support.captured_stderr() as err:
                    sys.__excepthook__(*sys.exc_info())

        err = err.getvalue()
        self.assertIn("""  File "b'bytes_filename'", line 123\n""", err)
        self.assertIn("""    text\n""", err)
        self.assertTrue(err.endswith("SyntaxError: msg\n"))

    def test_excepthook(self):
        with test.support.captured_output("stderr") as stderr:
            with test.support.catch_unraisable_exception():
                sys.excepthook(1, '1', 1)
        self.assertTrue("TypeError: print_exception(): Exception expected for " \
                         "value, str found" in stderr.getvalue())

    # FIXME: testing the code for a lost or replaced excepthook in
    # Python/pythonrun.c::PyErr_PrintEx() is tricky.


class SysModuleTest(__TestCase):

    def tearDown(self):
        test.support.reap_children()

    def test_exit(self):
        # call with two arguments
        self.assertRaises(TypeError, sys.exit, 42, 42)

        # call without argument
        with self.assertRaises(SystemExit) as cm:
            sys.exit()
        self.assertIsNone(cm.exception.code)

        rc, out, err = assert_python_ok('-c', 'import sys; sys.exit()')
        self.assertEqual(rc, 0)
        self.assertEqual(out, b'')
        self.assertEqual(err, b'')

        # gh-125842: Windows uses 32-bit unsigned integers for exit codes
        # so a -1 exit code is sometimes interpreted as 0xffff_ffff.
        rc, out, err = assert_python_failure('-c', 'import sys; sys.exit(0xffff_ffff)')
        self.assertIn(rc, (-1, 0xff, 0xffff_ffff))
        self.assertEqual(out, b'')
        self.assertEqual(err, b'')

        # Overflow results in a -1 exit code, which may be converted to 0xff
        # or 0xffff_ffff.
        rc, out, err = assert_python_failure('-c', 'import sys; sys.exit(2**128)')
        self.assertIn(rc, (-1, 0xff, 0xffff_ffff))
        self.assertEqual(out, b'')
        self.assertEqual(err, b'')

        # call with integer argument
        with self.assertRaises(SystemExit) as cm:
            sys.exit(42)
        self.assertEqual(cm.exception.code, 42)

        # call with tuple argument with one entry
        # entry will be unpacked
        with self.assertRaises(SystemExit) as cm:
            sys.exit((42,))
        self.assertEqual(cm.exception.code, 42)

        # call with string argument
        with self.assertRaises(SystemExit) as cm:
            sys.exit("exit")
        self.assertEqual(cm.exception.code, "exit")

        # call with tuple argument with two entries
        with self.assertRaises(SystemExit) as cm:
            sys.exit((17, 23))
        self.assertEqual(cm.exception.code, (17, 23))

        # test that the exit machinery handles SystemExits properly
        rc, out, err = assert_python_failure('-c', 'raise SystemExit(47)')
        self.assertEqual(rc, 47)
        self.assertEqual(out, b'')
        self.assertEqual(err, b'')

        def check_exit_message(code, expected, **env_vars):
            rc, out, err = assert_python_failure('-c', code, **env_vars)
            self.assertEqual(rc, 1)
            self.assertEqual(out, b'')
            self.assertTrue(err.startswith(expected),
                "%s doesn't start with %s" % (ascii(err), ascii(expected)))

        # test that stderr buffer is flushed before the exit message is written
        # into stderr
        check_exit_message(
            r'import sys; sys.stderr.write("unflushed,"); sys.exit("message")',
            b"unflushed,message")

        # test that the exit message is written with backslashreplace error
        # handler to stderr
        check_exit_message(
            r'import sys; sys.exit("surrogates:\uDCFF")',
            b"surrogates:\\udcff")

        # test that the unicode message is encoded to the stderr encoding
        # instead of the default encoding (utf8)
        check_exit_message(
            r'import sys; sys.exit("h\xe9")',
            b"h\xe9", PYTHONIOENCODING='latin-1')

    @support.requires_subprocess()
    def test_exit_codes_under_repl(self):
        # GH-129900: SystemExit, or things that raised it, didn't
        # get their return code propagated by the REPL
        import tempfile

        exit_ways = [
            "exit",
            "__import__('sys').exit",
            "raise SystemExit"
        ]

        for exitfunc in exit_ways:
            for return_code in (0, 123):
                with self.subTest(exitfunc=exitfunc, return_code=return_code):
                    with tempfile.TemporaryFile("w+") as stdin:
                        stdin.write(f"{exitfunc}({return_code})\n")
                        stdin.seek(0)
                        proc = subprocess.run([sys.executable], stdin=stdin)
                        self.assertEqual(proc.returncode, return_code)

    def test_getdefaultencoding(self):
        self.assertRaises(TypeError, sys.getdefaultencoding, 42)
        # can't check more than the type, as the user might have changed it
        self.assertIsInstance(sys.getdefaultencoding(), str)

    # testing sys.settrace() is done in test_sys_settrace.py
    # testing sys.setprofile() is done in test_sys_setprofile.py

    def test_switchinterval(self):
        self.assertRaises(TypeError, sys.setswitchinterval)
        self.assertRaises(TypeError, sys.setswitchinterval, "a")
        self.assertRaises(ValueError, sys.setswitchinterval, -1.0)
        self.assertRaises(ValueError, sys.setswitchinterval, 0.0)
        orig = sys.getswitchinterval()
        # sanity check
        self.assertTrue(orig < 0.5, orig)
        try:
            for n in 0.00001, 0.05, 3.0, orig:
                sys.setswitchinterval(n)
                self.assertAlmostEqual(sys.getswitchinterval(), n)
        finally:
            sys.setswitchinterval(orig)

    def test_getrecursionlimit(self):
        limit = sys.getrecursionlimit()
        self.assertIsInstance(limit, int)
        self.assertGreater(limit, 1)

        self.assertRaises(TypeError, sys.getrecursionlimit, 42)

    def test_setrecursionlimit(self):
        old_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(10_005)
            self.assertEqual(sys.getrecursionlimit(), 10_005)

            self.assertRaises(TypeError, sys.setrecursionlimit)
            self.assertRaises(ValueError, sys.setrecursionlimit, -42)
        finally:
            sys.setrecursionlimit(old_limit)

    def test_recursionlimit_recovery(self):
        if hasattr(sys, 'gettrace') and sys.gettrace():
            self.skipTest('fatal error if run with a trace function')

        old_limit = sys.getrecursionlimit()
        def f():
            f()
        try:
            for depth in (50, 75, 100, 250, 1000):
                try:
                    sys.setrecursionlimit(depth)
                except RecursionError:
                    # Issue #25274: The recursion limit is too low at the
                    # current recursion depth
                    continue

                # Issue #5392: test stack overflow after hitting recursion
                # limit twice
                with self.assertRaises(RecursionError):
                    f()
                with self.assertRaises(RecursionError):
                    f()
        finally:
            sys.setrecursionlimit(old_limit)

    @test.support.cpython_only
    def test_setrecursionlimit_to_depth(self):
        # Issue #25274: Setting a low recursion limit must be blocked if the
        # current recursion depth is already higher than limit.

        old_limit = sys.getrecursionlimit()
        try:
            depth = support.get_recursion_depth()
            with self.subTest(limit=sys.getrecursionlimit(), depth=depth):
                # depth + 1 is OK
                sys.setrecursionlimit(depth + 1)

                # reset the limit to be able to call self.assertRaises()
                # context manager
                sys.setrecursionlimit(old_limit)
                with self.assertRaises(RecursionError) as cm:
                    sys.setrecursionlimit(depth)
            self.assertRegex(str(cm.exception),
                             "cannot set the recursion limit to [0-9]+ "
                             "at the recursion depth [0-9]+: "
                             "the limit is too low")
        finally:
            sys.setrecursionlimit(old_limit)

    @unittest.skipUnless(support.Py_GIL_DISABLED, "only meaningful if the GIL is disabled")
    @threading_helper.requires_working_threading()
    def test_racing_recursion_limit(self):
        from threading import Thread
        def something_recursive():
            def count(n):
                if n > 0:
                    return count(n - 1) + 1
                return 0

            count(50)

        def set_recursion_limit():
            for limit in range(100, 200):
                sys.setrecursionlimit(limit)

        threads = []
        for _ in range(5):
            threads.append(Thread(target=set_recursion_limit))

        for _ in range(5):
            threads.append(Thread(target=something_recursive))

        with threading_helper.catch_threading_exception() as cm:
            with threading_helper.start_threads(threads):
                pass

            if cm.exc_value:
                raise cm.exc_value

    def test_getwindowsversion(self):
        # Raise SkipTest if sys doesn't have getwindowsversion attribute
        test.support.get_attribute(sys, "getwindowsversion")
        v = sys.getwindowsversion()
        self.assertEqual(len(v), 5)
        self.assertIsInstance(v[0], int)
        self.assertIsInstance(v[1], int)
        self.assertIsInstance(v[2], int)
        self.assertIsInstance(v[3], int)
        self.assertIsInstance(v[4], str)
        self.assertRaises(IndexError, operator.getitem, v, 5)
        self.assertIsInstance(v.major, int)
        self.assertIsInstance(v.minor, int)
        self.assertIsInstance(v.build, int)
        self.assertIsInstance(v.platform, int)
        self.assertIsInstance(v.service_pack, str)
        self.assertIsInstance(v.service_pack_minor, int)
        self.assertIsInstance(v.service_pack_major, int)
        self.assertIsInstance(v.suite_mask, int)
        self.assertIsInstance(v.product_type, int)
        self.assertEqual(v[0], v.major)
        self.assertEqual(v[1], v.minor)
        self.assertEqual(v[2], v.build)
        self.assertEqual(v[3], v.platform)
        self.assertEqual(v[4], v.service_pack)

        # This is how platform.py calls it. Make sure tuple
        #  still has 5 elements
        maj, min, buildno, plat, csd = sys.getwindowsversion()

    def test_call_tracing(self):
        self.assertRaises(TypeError, sys.call_tracing, type, 2)

    @unittest.skipUnless(hasattr(sys, "setdlopenflags"),
                         'test needs sys.setdlopenflags()')
    def test_dlopenflags(self):
        self.assertTrue(hasattr(sys, "getdlopenflags"))
        self.assertRaises(TypeError, sys.getdlopenflags, 42)
        oldflags = sys.getdlopenflags()
        self.assertRaises(TypeError, sys.setdlopenflags)
        sys.setdlopenflags(oldflags+1)
        self.assertEqual(sys.getdlopenflags(), oldflags+1)
        sys.setdlopenflags(oldflags)

    @test.support.refcount_test
    def test_refcount(self):
        # n here originally had to be a global in order for this test to pass
        # while tracing with a python function. Tracing used to call
        # PyFrame_FastToLocals, which would add a copy of any locals to the
        # frame object, causing the ref count to increase by 2 instead of 1.
        # While that no longer happens (due to PEP 667), this test case retains
        # its original global-based implementation
        # PEP 683's immortal objects also made this point moot, since the
        # refcount for None doesn't change anyway. Maybe this test should be
        # using a different constant value? (e.g. an integer)
        global n
        self.assertRaises(TypeError, sys.getrefcount)
        c = sys.getrefcount(None)
        n = None
        # Singleton refcnts don't change
        self.assertEqual(sys.getrefcount(None), c)
        del n
        self.assertEqual(sys.getrefcount(None), c)
        if hasattr(sys, "gettotalrefcount"):
            self.assertIsInstance(sys.gettotalrefcount(), int)

    def test_getframe(self):
        self.assertRaises(TypeError, sys._getframe, 42, 42)
        self.assertRaises(ValueError, sys._getframe, 2000000000)
        self.assertTrue(
            SysModuleTest.test_getframe.__code__ \
            is sys._getframe().f_code
        )

    @unittest.expectedFailure
    def test_getframemodulename(self):
        # Default depth gets ourselves
        self.assertEqual(__name__, sys._getframemodulename())
        self.assertEqual("unittest.case", sys._getframemodulename(1))
        i = 0
        f = sys._getframe(i)
        while f:
            self.assertEqual(
                f.f_globals['__name__'],
                sys._getframemodulename(i) or '__main__'
            )
            i += 1
            f2 = f.f_back
            try:
                f = sys._getframe(i)
            except ValueError:
                break
            self.assertIs(f, f2)
        self.assertIsNone(sys._getframemodulename(i))

    # sys._current_frames() is a CPython-only gimmick.
    @threading_helper.reap_threads
    @threading_helper.requires_working_threading()
    def test_current_frames(self):
        import threading
        import traceback

        # Spawn a thread that blocks at a known place.  Then the main
        # thread does sys._current_frames(), and verifies that the frames
        # returned make sense.
        entered_g = threading.Event()
        leave_g = threading.Event()
        thread_info = []  # the thread's id

        def f123():
            g456()

        def g456():
            thread_info.append(threading.get_ident())
            entered_g.set()
            leave_g.wait()

        t = threading.Thread(target=f123)
        t.start()
        entered_g.wait()

        try:
            # At this point, t has finished its entered_g.set(), although it's
            # impossible to guess whether it's still on that line or has moved on
            # to its leave_g.wait().
            self.assertEqual(len(thread_info), 1)
            thread_id = thread_info[0]

            d = sys._current_frames()
            for tid in d:
                self.assertIsInstance(tid, int)
                self.assertGreater(tid, 0)

            main_id = threading.get_ident()
            self.assertIn(main_id, d)
            self.assertIn(thread_id, d)

            # Verify that the captured main-thread frame is _this_ frame.
            frame = d.pop(main_id)
            self.assertTrue(frame is sys._getframe())

            # Verify that the captured thread frame is blocked in g456, called
            # from f123.  This is a little tricky, since various bits of
            # threading.py are also in the thread's call stack.
            frame = d.pop(thread_id)
            stack = traceback.extract_stack(frame)
            for i, (filename, lineno, funcname, sourceline) in enumerate(stack):
                if funcname == "f123":
                    break
            else:
                self.fail("didn't find f123() on thread's call stack")

            self.assertEqual(sourceline, "g456()")

            # And the next record must be for g456().
            filename, lineno, funcname, sourceline = stack[i+1]
            self.assertEqual(funcname, "g456")
            self.assertIn(sourceline, ["leave_g.wait()", "entered_g.set()"])
        finally:
            # Reap the spawned thread.
            leave_g.set()
            t.join()

    @threading_helper.reap_threads
    @threading_helper.requires_working_threading()
    def test_current_exceptions(self):
        import threading
        import traceback

        # Spawn a thread that blocks at a known place.  Then the main
        # thread does sys._current_frames(), and verifies that the frames
        # returned make sense.
        g_raised = threading.Event()
        leave_g = threading.Event()
        thread_info = []  # the thread's id

        def f123():
            g456()

        def g456():
            thread_info.append(threading.get_ident())
            while True:
                try:
                    raise ValueError("oops")
                except ValueError:
                    g_raised.set()
                    if leave_g.wait(timeout=support.LONG_TIMEOUT):
                        break

        t = threading.Thread(target=f123)
        t.start()
        g_raised.wait(timeout=support.LONG_TIMEOUT)

        try:
            self.assertEqual(len(thread_info), 1)
            thread_id = thread_info[0]

            d = sys._current_exceptions()
            for tid in d:
                self.assertIsInstance(tid, int)
                self.assertGreater(tid, 0)

            main_id = threading.get_ident()
            self.assertIn(main_id, d)
            self.assertIn(thread_id, d)
            self.assertEqual(None, d.pop(main_id))

            # Verify that the captured thread frame is blocked in g456, called
            # from f123.  This is a little tricky, since various bits of
            # threading.py are also in the thread's call stack.
            exc_value = d.pop(thread_id)
            stack = traceback.extract_stack(exc_value.__traceback__.tb_frame)
            for i, (filename, lineno, funcname, sourceline) in enumerate(stack):
                if funcname == "f123":
                    break
            else:
                self.fail("didn't find f123() on thread's call stack")

            self.assertEqual(sourceline, "g456()")

            # And the next record must be for g456().
            filename, lineno, funcname, sourceline = stack[i+1]
            self.assertEqual(funcname, "g456")
            self.assertTrue((sourceline.startswith("if leave_g.wait(") or
                             sourceline.startswith("g_raised.set()")))
        finally:
            # Reap the spawned thread.
            leave_g.set()
            t.join()

    def test_attributes(self):
        self.assertIsInstance(sys.api_version, int)
        self.assertIsInstance(sys.argv, list)
        for arg in sys.argv:
            self.assertIsInstance(arg, str)
        self.assertIsInstance(sys.orig_argv, list)
        for arg in sys.orig_argv:
            self.assertIsInstance(arg, str)
        self.assertIn(sys.byteorder, ("little", "big"))
        self.assertIsInstance(sys.builtin_module_names, tuple)
        self.assertIsInstance(sys.copyright, str)
        self.assertIsInstance(sys.exec_prefix, str)
        self.assertIsInstance(sys.base_exec_prefix, str)
        self.assertIsInstance(sys.executable, str)
        self.assertEqual(len(sys.float_info), 11)
        self.assertEqual(sys.float_info.radix, 2)
        self.assertEqual(len(sys.int_info), 4)
        self.assertTrue(sys.int_info.bits_per_digit % 5 == 0)
        self.assertTrue(sys.int_info.sizeof_digit >= 1)
        self.assertGreaterEqual(sys.int_info.default_max_str_digits, 500)
        self.assertGreaterEqual(sys.int_info.str_digits_check_threshold, 100)
        self.assertGreater(sys.int_info.default_max_str_digits,
                           sys.int_info.str_digits_check_threshold)
        self.assertEqual(type(sys.int_info.bits_per_digit), int)
        self.assertEqual(type(sys.int_info.sizeof_digit), int)
        self.assertIsInstance(sys.int_info.default_max_str_digits, int)
        self.assertIsInstance(sys.int_info.str_digits_check_threshold, int)
        self.assertIsInstance(sys.hexversion, int)

        self.assertEqual(len(sys.hash_info), 9)
        self.assertLess(sys.hash_info.modulus, 2**sys.hash_info.width)
        # sys.hash_info.modulus should be a prime; we do a quick
        # probable primality test (doesn't exclude the possibility of
        # a Carmichael number)
        for x in range(1, 100):
            self.assertEqual(
                pow(x, sys.hash_info.modulus-1, sys.hash_info.modulus),
                1,
                "sys.hash_info.modulus {} is a non-prime".format(
                    sys.hash_info.modulus)
                )
        self.assertIsInstance(sys.hash_info.inf, int)
        self.assertIsInstance(sys.hash_info.nan, int)
        self.assertIsInstance(sys.hash_info.imag, int)
        algo = sysconfig.get_config_var("Py_HASH_ALGORITHM")
        if sys.hash_info.algorithm in {"fnv", "siphash13", "siphash24"}:
            self.assertIn(sys.hash_info.hash_bits, {32, 64})
            self.assertIn(sys.hash_info.seed_bits, {32, 64, 128})

            if algo == 1:
                self.assertEqual(sys.hash_info.algorithm, "siphash24")
            elif algo == 2:
                self.assertEqual(sys.hash_info.algorithm, "fnv")
            elif algo == 3:
                self.assertEqual(sys.hash_info.algorithm, "siphash13")
            else:
                self.assertIn(sys.hash_info.algorithm, {"fnv", "siphash13", "siphash24"})
        else:
            # PY_HASH_EXTERNAL
            self.assertEqual(algo, 0)
        self.assertGreaterEqual(sys.hash_info.cutoff, 0)
        self.assertLess(sys.hash_info.cutoff, 8)

        self.assertIsInstance(sys.maxsize, int)
        self.assertIsInstance(sys.maxunicode, int)
        self.assertEqual(sys.maxunicode, 0x10FFFF)
        self.assertIsInstance(sys.platform, str)
        self.assertIsInstance(sys.prefix, str)
        self.assertIsInstance(sys.base_prefix, str)
        self.assertIsInstance(sys.platlibdir, str)
        self.assertIsInstance(sys.version, str)
        vi = sys.version_info
        self.assertIsInstance(vi[:], tuple)
        self.assertEqual(len(vi), 5)
        self.assertIsInstance(vi[0], int)
        self.assertIsInstance(vi[1], int)
        self.assertIsInstance(vi[2], int)
        self.assertIn(vi[3], ("alpha", "beta", "candidate", "final"))
        self.assertIsInstance(vi[4], int)
        self.assertIsInstance(vi.major, int)
        self.assertIsInstance(vi.minor, int)
        self.assertIsInstance(vi.micro, int)
        self.assertIn(vi.releaselevel, ("alpha", "beta", "candidate", "final"))
        self.assertIsInstance(vi.serial, int)
        self.assertEqual(vi[0], vi.major)
        self.assertEqual(vi[1], vi.minor)
        self.assertEqual(vi[2], vi.micro)
        self.assertEqual(vi[3], vi.releaselevel)
        self.assertEqual(vi[4], vi.serial)
        self.assertTrue(vi > (1,0,0))
        self.assertIsInstance(sys.float_repr_style, str)
        self.assertIn(sys.float_repr_style, ('short', 'legacy'))
        if not sys.platform.startswith('win'):
            self.assertIsInstance(sys.abiflags, str)

    def test_thread_info(self):
        info = sys.thread_info
        self.assertEqual(len(info), 3)
        self.assertIn(info.name, ('nt', 'pthread', 'pthread-stubs', 'solaris', None))
        self.assertIn(info.lock, ('semaphore', 'mutex+cond', None))
        if sys.platform.startswith(("linux", "android", "freebsd")):
            self.assertEqual(info.name, "pthread")
        elif sys.platform == "win32":
            self.assertEqual(info.name, "nt")
        elif sys.platform == "emscripten":
            self.assertIn(info.name, {"pthread", "pthread-stubs"})
        elif sys.platform == "wasi":
            self.assertEqual(info.name, "pthread-stubs")

    @unittest.skipUnless(support.is_emscripten, "only available on Emscripten")
    def test_emscripten_info(self):
        self.assertEqual(len(sys._emscripten_info), 4)
        self.assertIsInstance(sys._emscripten_info.emscripten_version, tuple)
        self.assertIsInstance(sys._emscripten_info.runtime, (str, type(None)))
        self.assertIsInstance(sys._emscripten_info.pthreads, bool)
        self.assertIsInstance(sys._emscripten_info.shared_memory, bool)

    def test_43581(self):
        # Can't use sys.stdout, as this is a StringIO object when
        # the test runs under regrtest.
        self.assertEqual(sys.__stdout__.encoding, sys.__stderr__.encoding)

    def test_intern(self):
        has_is_interned = (test.support.check_impl_detail(cpython=True)
                           or hasattr(sys, '_is_interned'))
        self.assertRaises(TypeError, sys.intern)
        self.assertRaises(TypeError, sys.intern, b'abc')
        if has_is_interned:
            self.assertRaises(TypeError, sys._is_interned)
            self.assertRaises(TypeError, sys._is_interned, b'abc')
        s = "never interned before" + str(random.randrange(0, 10**9))
        self.assertTrue(sys.intern(s) is s)
        if has_is_interned:
            self.assertIs(sys._is_interned(s), True)
        s2 = s.swapcase().swapcase()
        if has_is_interned:
            self.assertIs(sys._is_interned(s2), False)
        self.assertTrue(sys.intern(s2) is s)
        if has_is_interned:
            self.assertIs(sys._is_interned(s2), False)

        # Subclasses of string can't be interned, because they
        # provide too much opportunity for insane things to happen.
        # We don't want them in the interned dict and if they aren't
        # actually interned, we don't want to create the appearance
        # that they are by allowing intern() to succeed.
        class S(str):
            def __hash__(self):
                return 123

        self.assertRaises(TypeError, sys.intern, S("abc"))
        if has_is_interned:
            self.assertIs(sys._is_interned(S("abc")), False)

    @support.cpython_only
    @requires_subinterpreters
    def test_subinterp_intern_dynamically_allocated(self):
        # Implementation detail: Dynamically allocated strings
        # are distinct between interpreters
        s = "never interned before" + str(random.randrange(0, 10**9))
        t = sys.intern(s)
        self.assertIs(t, s)

        interp = interpreters.create()
        interp.exec(textwrap.dedent(f'''
            import sys

            # set `s`, avoid parser interning & constant folding
            s = str({s.encode()!r}, 'utf-8')

            t = sys.intern(s)

            assert id(t) != {id(s)}, (id(t), {id(s)})
            assert id(t) != {id(t)}, (id(t), {id(t)})
            '''))

    @support.cpython_only
    @requires_subinterpreters
    def test_subinterp_intern_statically_allocated(self):
        # Implementation detail: Statically allocated strings are shared
        # between interpreters.
        # See Tools/build/generate_global_objects.py for the list
        # of strings that are always statically allocated.
        for s in ('__init__', 'CANCELLED', '<module>', 'utf-8',
                  '{{', '', '\n', '_', 'x', '\0', '\N{CEDILLA}', '\xff',
                  ):
            with self.subTest(s=s):
                t = sys.intern(s)

                interp = interpreters.create()
                interp.exec(textwrap.dedent(f'''
                    import sys

                    # set `s`, avoid parser interning & constant folding
                    s = str({s.encode()!r}, 'utf-8')

                    t = sys.intern(s)
                    assert id(t) == {id(t)}, (id(t), {id(t)})
                    '''))

    @support.cpython_only
    @requires_subinterpreters
    def test_subinterp_intern_singleton(self):
        # Implementation detail: singletons are used for 0- and 1-character
        # latin1 strings.
        for s in '', '\n', '_', 'x', '\0', '\N{CEDILLA}', '\xff':
            with self.subTest(s=s):
                interp = interpreters.create()
                interp.exec(textwrap.dedent(f'''
                    import sys

                    # set `s`, avoid parser interning & constant folding
                    s = str({s.encode()!r}, 'utf-8')

                    assert id(s) == {id(s)}
                    t = sys.intern(s)
                    '''))
                self.assertTrue(sys._is_interned(s))

    def test_sys_flags(self):
        self.assertTrue(sys.flags)
        attrs = ("debug",
                 "inspect", "interactive", "optimize",
                 "dont_write_bytecode", "no_user_site", "no_site",
                 "ignore_environment", "verbose", "bytes_warning", "quiet",
                 "hash_randomization", "isolated", "dev_mode", "utf8_mode",
                 "warn_default_encoding", "safe_path", "int_max_str_digits")
        for attr in attrs:
            self.assertTrue(hasattr(sys.flags, attr), attr)
            attr_type = bool if attr in ("dev_mode", "safe_path") else int
            self.assertEqual(type(getattr(sys.flags, attr)), attr_type, attr)
        self.assertTrue(repr(sys.flags))
        self.assertEqual(len(sys.flags), len(attrs))

        self.assertIn(sys.flags.utf8_mode, {0, 1, 2})

    def assert_raise_on_new_sys_type(self, sys_attr):
        # Users are intentionally prevented from creating new instances of
        # sys.flags, sys.version_info, and sys.getwindowsversion.
        arg = sys_attr
        attr_type = type(sys_attr)
        with self.assertRaises(TypeError):
            attr_type(arg)
        with self.assertRaises(TypeError):
            attr_type.__new__(attr_type, arg)

    def test_sys_flags_no_instantiation(self):
        self.assert_raise_on_new_sys_type(sys.flags)

    def test_sys_version_info_no_instantiation(self):
        self.assert_raise_on_new_sys_type(sys.version_info)

    def test_sys_getwindowsversion_no_instantiation(self):
        # Skip if not being run on Windows.
        test.support.get_attribute(sys, "getwindowsversion")
        self.assert_raise_on_new_sys_type(sys.getwindowsversion())

    @test.support.cpython_only
    def test_clear_type_cache(self):
        sys._clear_type_cache()

    @force_not_colorized
    @support.requires_subprocess()
    def test_ioencoding(self):
        env = dict(os.environ)

        # Test character: cent sign, encoded as 0x4A (ASCII J) in CP424,
        # not representable in ASCII.

        env["PYTHONIOENCODING"] = "cp424"
        p = subprocess.Popen([sys.executable, "-c", 'print(chr(0xa2))'],
                             stdout = subprocess.PIPE, env=env)
        out = p.communicate()[0].strip()
        expected = ("\xa2" + os.linesep).encode("cp424")
        self.assertEqual(out, expected)

        env["PYTHONIOENCODING"] = "ascii:replace"
        p = subprocess.Popen([sys.executable, "-c", 'print(chr(0xa2))'],
                             stdout = subprocess.PIPE, env=env)
        out = p.communicate()[0].strip()
        self.assertEqual(out, b'?')

        env["PYTHONIOENCODING"] = "ascii"
        p = subprocess.Popen([sys.executable, "-c", 'print(chr(0xa2))'],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             env=env)
        out, err = p.communicate()
        self.assertEqual(out, b'')
        self.assertIn(b'UnicodeEncodeError:', err)
        self.assertIn(rb"'\xa2'", err)

        env["PYTHONIOENCODING"] = "ascii:"
        p = subprocess.Popen([sys.executable, "-c", 'print(chr(0xa2))'],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             env=env)
        out, err = p.communicate()
        self.assertEqual(out, b'')
        self.assertIn(b'UnicodeEncodeError:', err)
        self.assertIn(rb"'\xa2'", err)

        env["PYTHONIOENCODING"] = ":surrogateescape"
        p = subprocess.Popen([sys.executable, "-c", 'print(chr(0xdcbd))'],
                             stdout=subprocess.PIPE, env=env)
        out = p.communicate()[0].strip()
        self.assertEqual(out, b'\xbd')

    @unittest.skipUnless(os_helper.FS_NONASCII,
                         'requires OS support of non-ASCII encodings')
    @unittest.skipUnless(sys.getfilesystemencoding() == locale.getpreferredencoding(False),
                         'requires FS encoding to match locale')
    @support.requires_subprocess()
    def test_ioencoding_nonascii(self):
        env = dict(os.environ)

        env["PYTHONIOENCODING"] = ""
        p = subprocess.Popen([sys.executable, "-c",
                                'print(%a)' % os_helper.FS_NONASCII],
                                stdout=subprocess.PIPE, env=env)
        out = p.communicate()[0].strip()
        self.assertEqual(out, os.fsencode(os_helper.FS_NONASCII))

    @unittest.skipIf(sys.base_prefix != sys.prefix,
                     'Test is not venv-compatible')
    @support.requires_subprocess()
    def test_executable(self):
        # sys.executable should be absolute
        self.assertEqual(os.path.abspath(sys.executable), sys.executable)

        # Issue #7774: Ensure that sys.executable is an empty string if argv[0]
        # has been set to a non existent program name and Python is unable to
        # retrieve the real program name

        # For a normal installation, it should work without 'cwd'
        # argument. For test runs in the build directory, see #7774.
        python_dir = os.path.dirname(os.path.realpath(sys.executable))
        p = subprocess.Popen(
            ["nonexistent", "-c",
             'import sys; print(sys.executable.encode("ascii", "backslashreplace"))'],
            executable=sys.executable, stdout=subprocess.PIPE, cwd=python_dir)
        stdout = p.communicate()[0]
        executable = stdout.strip().decode("ASCII")
        p.wait()
        self.assertIn(executable, ["b''", repr(sys.executable.encode("ascii", "backslashreplace"))])

    def check_fsencoding(self, fs_encoding, expected=None):
        self.assertIsNotNone(fs_encoding)
        codecs.lookup(fs_encoding)
        if expected:
            self.assertEqual(fs_encoding, expected)

    def test_getfilesystemencoding(self):
        fs_encoding = sys.getfilesystemencoding()
        if sys.platform == 'darwin':
            expected = 'utf-8'
        else:
            expected = None
        self.check_fsencoding(fs_encoding, expected)

    def c_locale_get_error_handler(self, locale, isolated=False, encoding=None):
        # Force the POSIX locale
        env = os.environ.copy()
        env["LC_ALL"] = locale
        env["PYTHONCOERCECLOCALE"] = "0"
        code = '\n'.join((
            'import sys',
            'def dump(name):',
            '    std = getattr(sys, name)',
            '    print("%s: %s" % (name, std.errors))',
            'dump("stdin")',
            'dump("stdout")',
            'dump("stderr")',
        ))
        args = [sys.executable, "-X", "utf8=0", "-c", code]
        if isolated:
            args.append("-I")
        if encoding is not None:
            env['PYTHONIOENCODING'] = encoding
        else:
            env.pop('PYTHONIOENCODING', None)
        p = subprocess.Popen(args,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,
                              env=env,
                              universal_newlines=True)
        stdout, stderr = p.communicate()
        return stdout

    def check_locale_surrogateescape(self, locale):
        out = self.c_locale_get_error_handler(locale, isolated=True)
        self.assertEqual(out,
                         'stdin: surrogateescape\n'
                         'stdout: surrogateescape\n'
                         'stderr: backslashreplace\n')

        # replace the default error handler
        out = self.c_locale_get_error_handler(locale, encoding=':ignore')
        self.assertEqual(out,
                         'stdin: ignore\n'
                         'stdout: ignore\n'
                         'stderr: backslashreplace\n')

        # force the encoding
        out = self.c_locale_get_error_handler(locale, encoding='iso8859-1')
        self.assertEqual(out,
                         'stdin: strict\n'
                         'stdout: strict\n'
                         'stderr: backslashreplace\n')
        out = self.c_locale_get_error_handler(locale, encoding='iso8859-1:')
        self.assertEqual(out,
                         'stdin: strict\n'
                         'stdout: strict\n'
                         'stderr: backslashreplace\n')

        # have no any effect
        out = self.c_locale_get_error_handler(locale, encoding=':')
        self.assertEqual(out,
                         'stdin: surrogateescape\n'
                         'stdout: surrogateescape\n'
                         'stderr: backslashreplace\n')
        out = self.c_locale_get_error_handler(locale, encoding='')
        self.assertEqual(out,
                         'stdin: surrogateescape\n'
                         'stdout: surrogateescape\n'
                         'stderr: backslashreplace\n')

    @support.requires_subprocess()
    def test_c_locale_surrogateescape(self):
        self.check_locale_surrogateescape('C')

    @support.requires_subprocess()
    def test_posix_locale_surrogateescape(self):
        self.check_locale_surrogateescape('POSIX')

    def test_implementation(self):
        # This test applies to all implementations equally.

        levels = {'alpha': 0xA, 'beta': 0xB, 'candidate': 0xC, 'final': 0xF}

        self.assertTrue(hasattr(sys.implementation, 'name'))
        self.assertTrue(hasattr(sys.implementation, 'version'))
        self.assertTrue(hasattr(sys.implementation, 'hexversion'))
        self.assertTrue(hasattr(sys.implementation, 'cache_tag'))

        version = sys.implementation.version
        self.assertEqual(version[:2], (version.major, version.minor))

        hexversion = (version.major << 24 | version.minor << 16 |
                      version.micro << 8 | levels[version.releaselevel] << 4 |
                      version.serial << 0)
        self.assertEqual(sys.implementation.hexversion, hexversion)

        # PEP 421 requires that .name be lower case.
        self.assertEqual(sys.implementation.name,
                         sys.implementation.name.lower())

    @test.support.cpython_only
    def test_debugmallocstats(self):
        # Test sys._debugmallocstats()
        from test.support.script_helper import assert_python_ok
        args = ['-c', 'import sys; sys._debugmallocstats()']
        ret, out, err = assert_python_ok(*args)

        # Output of sys._debugmallocstats() depends on configure flags.
        # The sysconfig vars are not available on Windows.
        if sys.platform != "win32":
            with_freelists = sysconfig.get_config_var("WITH_FREELISTS")
            with_pymalloc = sysconfig.get_config_var("WITH_PYMALLOC")
            if with_freelists:
                self.assertIn(b"free PyDictObjects", err)
            if with_pymalloc:
                self.assertIn(b'Small block threshold', err)
            if not with_freelists and not with_pymalloc:
                self.assertFalse(err)

        # The function has no parameter
        self.assertRaises(TypeError, sys._debugmallocstats, True)

    @unittest.skipUnless(hasattr(sys, "getallocatedblocks"),
                         "sys.getallocatedblocks unavailable on this build")
    def test_getallocatedblocks(self):
        try:
            import _testinternalcapi
        except ImportError:
            with_pymalloc = support.with_pymalloc()
        else:
            try:
                alloc_name = _testinternalcapi.pymem_getallocatorsname()
            except RuntimeError as exc:
                # "cannot get allocators name" (ex: tracemalloc is used)
                with_pymalloc = True
            else:
                with_pymalloc = (alloc_name in ('pymalloc', 'pymalloc_debug'))

        # Some sanity checks
        a = sys.getallocatedblocks()
        self.assertIs(type(a), int)
        if with_pymalloc:
            self.assertGreater(a, 0)
        else:
            # When WITH_PYMALLOC isn't available, we don't know anything
            # about the underlying implementation: the function might
            # return 0 or something greater.
            self.assertGreaterEqual(a, 0)
        try:
            # While we could imagine a Python session where the number of
            # multiple buffer objects would exceed the sharing of references,
            # it is unlikely to happen in a normal test run.
            self.assertLess(a, sys.gettotalrefcount())
        except AttributeError:
            # gettotalrefcount() not available
            pass
        gc.collect()
        b = sys.getallocatedblocks()
        self.assertLessEqual(b, a)
        gc.collect()
        c = sys.getallocatedblocks()
        self.assertIn(c, range(b - 50, b + 50))

    def test_is_gil_enabled(self):
        if support.Py_GIL_DISABLED:
            self.assertIs(type(sys._is_gil_enabled()), bool)
        else:
            self.assertTrue(sys._is_gil_enabled())

    def test_is_finalizing(self):
        self.assertIs(sys.is_finalizing(), False)
        # Don't use the atexit module because _Py_Finalizing is only set
        # after calling atexit callbacks
        code = """if 1:
            import sys

            class AtExit:
                is_finalizing = sys.is_finalizing
                print = print

                def __del__(self):
                    self.print(self.is_finalizing(), flush=True)

            # Keep a reference in the __main__ module namespace, so the
            # AtExit destructor will be called at Python exit
            ref = AtExit()
        """
        rc, stdout, stderr = assert_python_ok('-c', code)
        self.assertEqual(stdout.rstrip(), b'True')

    def test_issue20602(self):
        # sys.flags and sys.float_info were wiped during shutdown.
        code = """if 1:
            import sys
            class A:
                def __del__(self, sys=sys):
                    print(sys.flags)
                    print(sys.float_info)
            a = A()
            """
        rc, out, err = assert_python_ok('-c', code)
        out = out.splitlines()
        self.assertIn(b'sys.flags', out[0])
        self.assertIn(b'sys.float_info', out[1])

    def test_sys_ignores_cleaning_up_user_data(self):
        code = """if 1:
            import struct, sys

            class C:
                def __init__(self):
                    self.pack = struct.pack
                def __del__(self):
                    self.pack('I', -42)

            sys.x = C()
            """
        rc, stdout, stderr = assert_python_ok('-c', code)
        self.assertEqual(rc, 0)
        self.assertEqual(stdout.rstrip(), b"")
        self.assertEqual(stderr.rstrip(), b"")

    @unittest.skipUnless(sys.platform == "android", "Android only")
    def test_getandroidapilevel(self):
        level = sys.getandroidapilevel()
        self.assertIsInstance(level, int)
        self.assertGreater(level, 0)

    @force_not_colorized
    @support.requires_subprocess()
    def test_sys_tracebacklimit(self):
        code = """if 1:
            import sys
            def f1():
                1 / 0
            def f2():
                f1()
            sys.tracebacklimit = %r
            f2()
        """
        def check(tracebacklimit, expected):
            p = subprocess.Popen([sys.executable, '-c', code % tracebacklimit],
                                 stderr=subprocess.PIPE)
            out = p.communicate()[1]
            self.assertEqual(out.splitlines(), expected)

        traceback = [
            b'Traceback (most recent call last):',
            b'  File "<string>", line 8, in <module>',
            b'    f2()',
            b'    ~~^^',
            b'  File "<string>", line 6, in f2',
            b'    f1()',
            b'    ~~^^',
            b'  File "<string>", line 4, in f1',
            b'    1 / 0',
            b'    ~~^~~',
            b'ZeroDivisionError: division by zero'
        ]
        check(10, traceback)
        check(3, traceback)
        check(2, traceback[:1] + traceback[4:])
        check(1, traceback[:1] + traceback[7:])
        check(0, [traceback[-1]])
        check(-1, [traceb
```



## High-Level Overview


This Python file contains 36 class(es) and 134 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `RedirectImportFinder`, `DisplayHookTest`, `ActiveExceptionTests`, `ExceptHookTest`, `SysModuleTest`, `S`, `AtExit`, `A`, `C`, `MyType`, `UnraisableHookTest`, `BrokenDel`, `BrokenStrException`, `BrokenExceptionDel`, `A`, `B`, `X`, `SizeofTest`, `BadSizeof`, `InvalidSizeof`

**Functions defined**: `find_spec`, `requires_subinterpreters`, `test_original_displayhook`, `test_lost_displayhook`, `test_custom_displayhook`, `baddisplayhook`, `test_exc_info_no_exception`, `test_sys_exception_no_exception`, `test_exc_info_with_exception_instance`, `f`, `test_exc_info_with_exception_type`, `f`, `test_sys_exception_with_exception_instance`, `f`, `test_sys_exception_with_exception_type`, `f`, `test_original_excepthook`, `test_excepthook_bytes_filename`, `test_excepthook`, `tearDown`

**Key imports**: sys, torch, torch._dynamo.test_case, unittest, CPythonTestCase, statements, sys, importlib.abc, is the problematic one, the standalone module


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
- `statements`
- `importlib.abc`
- `is the problematic one`
- `the standalone module`
- `builtins`
- `codecs`
- `_datetime`
- `gc`
- `io`
- `locale`
- `operator`
- `os`
- `random`
- `struct`
- `subprocess`
- `sysconfig`
- `test.support`
- `test`: support
- `test.support.script_helper`: assert_python_ok, assert_python_failure


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized
- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/dynamo/cpython/3_13/test_sys.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/dynamo/cpython/3_13`):

- [`mapping_tests.diff_docs.md`](./mapping_tests.diff_docs.md)
- [`test_float.py_docs.md`](./test_float.py_docs.md)
- [`test_generators.py_docs.md`](./test_generators.py_docs.md)
- [`test_dict.py_docs.md`](./test_dict.py_docs.md)
- [`test_generator_stop.diff_docs.md`](./test_generator_stop.diff_docs.md)
- [`test_sort.diff_docs.md`](./test_sort.diff_docs.md)
- [`test_list.diff_docs.md`](./test_list.diff_docs.md)
- [`test_userdict.diff_docs.md`](./test_userdict.diff_docs.md)
- [`test_generators.diff_docs.md`](./test_generators.diff_docs.md)
- [`test_userlist.py_docs.md`](./test_userlist.py_docs.md)


## Cross-References

- **File Documentation**: `test_sys.py_docs.md`
- **Keyword Index**: `test_sys.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
