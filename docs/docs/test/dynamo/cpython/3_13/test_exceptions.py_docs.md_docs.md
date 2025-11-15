# Documentation: `docs/test/dynamo/cpython/3_13/test_exceptions.py_docs.md`

## File Metadata

- **Path**: `docs/test/dynamo/cpython/3_13/test_exceptions.py_docs.md`
- **Size**: 53,989 bytes (52.72 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/dynamo/cpython/3_13/test_exceptions.py`

## File Metadata

- **Path**: `test/dynamo/cpython/3_13/test_exceptions.py`
- **Size**: 91,122 bytes (88.99 KB)
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
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_exceptions.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import (
    run_tests,
    xfailIfTorchDynamo,
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

# Python test set -- part 5, built-in exceptions

import copy
import os
import sys
import unittest
import pickle
import weakref
import errno
from codecs import BOM_UTF8
from itertools import product
from textwrap import dedent

from test.support import (captured_stderr, check_impl_detail,
                          cpython_only, gc_collect,
                          no_tracing, script_helper,
                          SuppressCrashReport,
                          force_not_colorized)
from test.support.import_helper import import_module
from test.support.os_helper import TESTFN, unlink
from test.support.warnings_helper import check_warnings
from test import support

try:
    import _testcapi
    from _testcapi import INT_MAX
except ImportError:
    _testcapi = None
    INT_MAX = 2**31 - 1


class NaiveException(Exception):
    def __init__(self, x):
        self.x = x

class SlottedNaiveException(Exception):
    __slots__ = ('x',)
    def __init__(self, x):
        self.x = x

class BrokenStrException(Exception):
    def __str__(self):
        raise Exception("str() is broken")

# XXX This is not really enough, each *operation* should be tested!


class ExceptionTests(__TestCase):

    def raise_catch(self, exc, excname):
        with self.subTest(exc=exc, excname=excname):
            try:
                raise exc("spam")
            except exc as err:
                buf1 = str(err)
            try:
                raise exc("spam")
            except exc as err:
                buf2 = str(err)
            self.assertEqual(buf1, buf2)
            self.assertEqual(exc.__name__, excname)

    def testRaising(self):
        self.raise_catch(AttributeError, "AttributeError")
        self.assertRaises(AttributeError, getattr, sys, "undefined_attribute")

        self.raise_catch(EOFError, "EOFError")
        fp = open(TESTFN, 'w', encoding="utf-8")
        fp.close()
        fp = open(TESTFN, 'r', encoding="utf-8")
        savestdin = sys.stdin
        try:
            try:
                import marshal
                marshal.loads(b'')
            except EOFError:
                pass
        finally:
            sys.stdin = savestdin
            fp.close()
            unlink(TESTFN)

        self.raise_catch(OSError, "OSError")
        self.assertRaises(OSError, open, 'this file does not exist', 'r')

        self.raise_catch(ImportError, "ImportError")
        self.assertRaises(ImportError, __import__, "undefined_module")

        self.raise_catch(IndexError, "IndexError")
        x = []
        self.assertRaises(IndexError, x.__getitem__, 10)

        self.raise_catch(KeyError, "KeyError")
        x = {}
        self.assertRaises(KeyError, x.__getitem__, 'key')

        self.raise_catch(KeyboardInterrupt, "KeyboardInterrupt")

        self.raise_catch(MemoryError, "MemoryError")

        self.raise_catch(NameError, "NameError")
        try: x = undefined_variable
        except NameError: pass

        self.raise_catch(OverflowError, "OverflowError")
        x = 1
        for dummy in range(128):
            x += x  # this simply shouldn't blow up

        self.raise_catch(RuntimeError, "RuntimeError")
        self.raise_catch(RecursionError, "RecursionError")

        self.raise_catch(SyntaxError, "SyntaxError")
        try: exec('/\n')
        except SyntaxError: pass

        self.raise_catch(IndentationError, "IndentationError")

        self.raise_catch(TabError, "TabError")
        try: compile("try:\n\t1/0\n    \t1/0\nfinally:\n pass\n",
                     '<string>', 'exec')
        except TabError: pass
        else: self.fail("TabError not raised")

        self.raise_catch(SystemError, "SystemError")

        self.raise_catch(SystemExit, "SystemExit")
        self.assertRaises(SystemExit, sys.exit, 0)

        self.raise_catch(TypeError, "TypeError")
        try: [] + ()
        except TypeError: pass

        self.raise_catch(ValueError, "ValueError")
        self.assertRaises(ValueError, chr, 17<<16)

        self.raise_catch(ZeroDivisionError, "ZeroDivisionError")
        try: x = 1/0
        except ZeroDivisionError: pass

        self.raise_catch(Exception, "Exception")
        try: x = 1/0
        except Exception as e: pass

        self.raise_catch(StopAsyncIteration, "StopAsyncIteration")

    def testSyntaxErrorMessage(self):
        # make sure the right exception message is raised for each of
        # these code fragments

        def ckmsg(src, msg):
            with self.subTest(src=src, msg=msg):
                try:
                    compile(src, '<fragment>', 'exec')
                except SyntaxError as e:
                    if e.msg != msg:
                        self.fail("expected %s, got %s" % (msg, e.msg))
                else:
                    self.fail("failed to get expected SyntaxError")

        s = '''if 1:
        try:
            continue
        except:
            pass'''

        ckmsg(s, "'continue' not properly in loop")
        ckmsg("continue\n", "'continue' not properly in loop")
        ckmsg("f'{6 0}'", "invalid syntax. Perhaps you forgot a comma?")

    def testSyntaxErrorMissingParens(self):
        def ckmsg(src, msg, exception=SyntaxError):
            try:
                compile(src, '<fragment>', 'exec')
            except exception as e:
                if e.msg != msg:
                    self.fail("expected %s, got %s" % (msg, e.msg))
            else:
                self.fail("failed to get expected SyntaxError")

        s = '''print "old style"'''
        ckmsg(s, "Missing parentheses in call to 'print'. Did you mean print(...)?")

        s = '''print "old style",'''
        ckmsg(s, "Missing parentheses in call to 'print'. Did you mean print(...)?")

        s = 'print f(a+b,c)'
        ckmsg(s, "Missing parentheses in call to 'print'. Did you mean print(...)?")

        s = '''exec "old style"'''
        ckmsg(s, "Missing parentheses in call to 'exec'. Did you mean exec(...)?")

        s = 'exec f(a+b,c)'
        ckmsg(s, "Missing parentheses in call to 'exec'. Did you mean exec(...)?")

        # Check that we don't incorrectly identify '(...)' as an expression to the right
        # of 'print'

        s = 'print (a+b,c) $ 42'
        ckmsg(s, "invalid syntax")

        s = 'exec (a+b,c) $ 42'
        ckmsg(s, "invalid syntax")

        # should not apply to subclasses, see issue #31161
        s = '''if True:\nprint "No indent"'''
        ckmsg(s, "expected an indented block after 'if' statement on line 1", IndentationError)

        s = '''if True:\n        print()\n\texec "mixed tabs and spaces"'''
        ckmsg(s, "inconsistent use of tabs and spaces in indentation", TabError)

    def check(self, src, lineno, offset, end_lineno=None, end_offset=None, encoding='utf-8'):
        with self.subTest(source=src, lineno=lineno, offset=offset):
            with self.assertRaises(SyntaxError) as cm:
                compile(src, '<fragment>', 'exec')
            self.assertEqual(cm.exception.lineno, lineno)
            self.assertEqual(cm.exception.offset, offset)
            if end_lineno is not None:
                self.assertEqual(cm.exception.end_lineno, end_lineno)
            if end_offset is not None:
                self.assertEqual(cm.exception.end_offset, end_offset)

            if cm.exception.text is not None:
                if not isinstance(src, str):
                    src = src.decode(encoding, 'replace')
                line = src.split('\n')[lineno-1]
                self.assertIn(line, cm.exception.text)

    def test_error_offset_continuation_characters(self):
        check = self.check
        check('"\\\n"(1 for c in I,\\\n\\', 2, 2)

    def testSyntaxErrorOffset(self):
        check = self.check
        check('def fact(x):\n\treturn x!\n', 2, 10)
        check('1 +\n', 1, 4)
        check('def spam():\n  print(1)\n print(2)', 3, 10)
        check('Python = "Python" +', 1, 20)
        check('Python = "\u1e54\xfd\u0163\u0125\xf2\xf1" +', 1, 20)
        check(b'# -*- coding: cp1251 -*-\nPython = "\xcf\xb3\xf2\xee\xed" +',
              2, 19, encoding='cp1251')
        check(b'Python = "\xcf\xb3\xf2\xee\xed" +', 1, 10)
        check('x = "a', 1, 5)
        check('lambda x: x = 2', 1, 1)
        check('f{a + b + c}', 1, 2)
        check('[file for str(file) in []\n]', 1, 11)
        check('a = « hello » « world »', 1, 5)
        check('[\nfile\nfor str(file)\nin\n[]\n]', 3, 5)
        check('[file for\n str(file) in []]', 2, 2)
        check("ages = {'Alice'=22, 'Bob'=23}", 1, 9)
        check('match ...:\n    case {**rest, "key": value}:\n        ...', 2, 19)
        check("[a b c d e f]", 1, 2)
        check("for x yfff:", 1, 7)
        check("f(a for a in b, c)", 1, 3, 1, 15)
        check("f(a for a in b if a, c)", 1, 3, 1, 20)
        check("f(a, b for b in c)", 1, 6, 1, 18)
        check("f(a, b for b in c, d)", 1, 6, 1, 18)

        # Errors thrown by compile.c
        check('class foo:return 1', 1, 11)
        check('def f():\n  continue', 2, 3)
        check('def f():\n  break', 2, 3)
        check('try:\n  pass\nexcept:\n  pass\nexcept ValueError:\n  pass', 3, 1)
        check('try:\n  pass\nexcept*:\n  pass', 3, 8)
        check('try:\n  pass\nexcept*:\n  pass\nexcept* ValueError:\n  pass', 3, 8)

        # Errors thrown by the tokenizer
        check('(0x+1)', 1, 3)
        check('x = 0xI', 1, 6)
        check('0010 + 2', 1, 1)
        check('x = 32e-+4', 1, 8)
        check('x = 0o9', 1, 7)
        check('\u03b1 = 0xI', 1, 6)
        check(b'\xce\xb1 = 0xI', 1, 6)
        check(b'# -*- coding: iso8859-7 -*-\n\xe1 = 0xI', 2, 6,
              encoding='iso8859-7')
        check(b"""if 1:
            def foo():
                '''

            def bar():
                pass

            def baz():
                '''quux'''
            """, 9, 24)
        check("pass\npass\npass\n(1+)\npass\npass\npass", 4, 4)
        check("(1+)", 1, 4)
        check("[interesting\nfoo()\n", 1, 1)
        check(b"\xef\xbb\xbf#coding: utf8\nprint('\xe6\x88\x91')\n", 0, -1)
        check("""f'''
            {
            (123_a)
            }'''""", 3, 17)
        check("""f'''
            {
            f\"\"\"
            {
            (123_a)
            }
            \"\"\"
            }'''""", 5, 17)
        check('''f"""


            {
            6
            0="""''', 5, 13)
        check('b"fooжжж"'.encode(), 1, 1, 1, 10)

        # Errors thrown by symtable.c
        check('x = [(yield i) for i in range(3)]', 1, 7)
        check('def f():\n  from _ import *', 2, 17)
        check('def f(x, x):\n  pass', 1, 10)
        check('{i for i in range(5) if (j := 0) for j in range(5)}', 1, 38)
        check('def f(x):\n  nonlocal x', 2, 3)
        check('def f(x):\n  x = 1\n  global x', 3, 3)
        check('nonlocal x', 1, 1)
        check('def f():\n  global x\n  nonlocal x', 2, 3)

        # Errors thrown by future.c
        check('from __future__ import doesnt_exist', 1, 24)
        check('from __future__ import braces', 1, 24)
        check('x=1\nfrom __future__ import division', 2, 1)
        check('foo(1=2)', 1, 5)
        check('def f():\n  x, y: int', 2, 3)
        check('[*x for x in xs]', 1, 2)
        check('foo(x for x in range(10), 100)', 1, 5)
        check('for 1 in []: pass', 1, 5)
        check('(yield i) = 2', 1, 2)
        check('def f(*):\n  pass', 1, 7)

    @unittest.skipIf(INT_MAX >= sys.maxsize, "Downcasting to int is safe for col_offset")
    @support.requires_resource('cpu')
    @support.bigmemtest(INT_MAX, memuse=2, dry_run=False)
    def testMemoryErrorBigSource(self, size):
        src = b"if True:\n%*s" % (size, b"pass")
        with self.assertRaisesRegex(OverflowError, "Parser column offset overflow"):
            compile(src, '<fragment>', 'exec')

    @cpython_only
    def testSettingException(self):
        # test that setting an exception at the C level works even if the
        # exception object can't be constructed.

        with torch._dynamo.error_on_graph_break(False):
            class BadException(Exception):
                def __init__(self_):
                    raise RuntimeError("can't instantiate BadException")

            class InvalidException:
                pass

        @unittest.skipIf(_testcapi is None, "requires _testcapi")
        def test_capi1():
            try:
                _testcapi.raise_exception(BadException, 1)
            except TypeError as err:
                co = err.__traceback__.tb_frame.f_code
                self.assertEqual(co.co_name, "test_capi1")
                self.assertTrue(co.co_filename.endswith('test_exceptions.py'))
            else:
                self.fail("Expected exception")

        @unittest.skipIf(_testcapi is None, "requires _testcapi")
        def test_capi2():
            try:
                _testcapi.raise_exception(BadException, 0)
            except RuntimeError as err:
                tb = err.__traceback__.tb_next
                co = tb.tb_frame.f_code
                self.assertEqual(co.co_name, "__init__")
                self.assertTrue(co.co_filename.endswith('test_exceptions.py'))
                co2 = tb.tb_frame.f_back.f_code
                self.assertEqual(co2.co_name, "test_capi2")
            else:
                self.fail("Expected exception")

        @unittest.skipIf(_testcapi is None, "requires _testcapi")
        def test_capi3():
            self.assertRaises(SystemError, _testcapi.raise_exception,
                              InvalidException, 1)

        test_capi1()
        test_capi2()
        test_capi3()

    def test_WindowsError(self):
        try:
            WindowsError
        except NameError:
            pass
        else:
            self.assertIs(WindowsError, OSError)
            self.assertEqual(str(OSError(1001)), "1001")
            self.assertEqual(str(OSError(1001, "message")),
                             "[Errno 1001] message")
            # POSIX errno (9 aka EBADF) is untranslated
            w = OSError(9, 'foo', 'bar')
            self.assertEqual(w.errno, 9)
            self.assertEqual(w.winerror, None)
            self.assertEqual(str(w), "[Errno 9] foo: 'bar'")
            # ERROR_PATH_NOT_FOUND (win error 3) becomes ENOENT (2)
            w = OSError(0, 'foo', 'bar', 3)
            self.assertEqual(w.errno, 2)
            self.assertEqual(w.winerror, 3)
            self.assertEqual(w.strerror, 'foo')
            self.assertEqual(w.filename, 'bar')
            self.assertEqual(w.filename2, None)
            self.assertEqual(str(w), "[WinError 3] foo: 'bar'")
            # Unknown win error becomes EINVAL (22)
            w = OSError(0, 'foo', None, 1001)
            self.assertEqual(w.errno, 22)
            self.assertEqual(w.winerror, 1001)
            self.assertEqual(w.strerror, 'foo')
            self.assertEqual(w.filename, None)
            self.assertEqual(w.filename2, None)
            self.assertEqual(str(w), "[WinError 1001] foo")
            # Non-numeric "errno"
            w = OSError('bar', 'foo')
            self.assertEqual(w.errno, 'bar')
            self.assertEqual(w.winerror, None)
            self.assertEqual(w.strerror, 'foo')
            self.assertEqual(w.filename, None)
            self.assertEqual(w.filename2, None)

    @unittest.skipUnless(sys.platform == 'win32',
                         'test specific to Windows')
    def test_windows_message(self):
        """Should fill in unknown error code in Windows error message"""
        ctypes = import_module('ctypes')
        # this error code has no message, Python formats it as hexadecimal
        code = 3765269347
        with self.assertRaisesRegex(OSError, 'Windows Error 0x%x' % code):
            ctypes.pythonapi.PyErr_SetFromWindowsErr(code)

    def testAttributes(self):
        # test that exception attributes are happy

        exceptionList = [
            (BaseException, (), {}, {'args' : ()}),
            (BaseException, (1, ), {}, {'args' : (1,)}),
            (BaseException, ('foo',), {},
                {'args' : ('foo',)}),
            (BaseException, ('foo', 1), {},
                {'args' : ('foo', 1)}),
            (SystemExit, ('foo',), {},
                {'args' : ('foo',), 'code' : 'foo'}),
            (OSError, ('foo',), {},
                {'args' : ('foo',), 'filename' : None, 'filename2' : None,
                 'errno' : None, 'strerror' : None}),
            (OSError, ('foo', 'bar'), {},
                {'args' : ('foo', 'bar'),
                 'filename' : None, 'filename2' : None,
                 'errno' : 'foo', 'strerror' : 'bar'}),
            (OSError, ('foo', 'bar', 'baz'), {},
                {'args' : ('foo', 'bar'),
                 'filename' : 'baz', 'filename2' : None,
                 'errno' : 'foo', 'strerror' : 'bar'}),
            (OSError, ('foo', 'bar', 'baz', None, 'quux'), {},
                {'args' : ('foo', 'bar'), 'filename' : 'baz', 'filename2': 'quux'}),
            (OSError, ('errnoStr', 'strErrorStr', 'filenameStr'), {},
                {'args' : ('errnoStr', 'strErrorStr'),
                 'strerror' : 'strErrorStr', 'errno' : 'errnoStr',
                 'filename' : 'filenameStr'}),
            (OSError, (1, 'strErrorStr', 'filenameStr'), {},
                {'args' : (1, 'strErrorStr'), 'errno' : 1,
                 'strerror' : 'strErrorStr',
                 'filename' : 'filenameStr', 'filename2' : None}),
            (SyntaxError, (), {}, {'msg' : None, 'text' : None,
                'filename' : None, 'lineno' : None, 'offset' : None,
                'end_offset': None, 'print_file_and_line' : None}),
            (SyntaxError, ('msgStr',), {},
                {'args' : ('msgStr',), 'text' : None,
                 'print_file_and_line' : None, 'msg' : 'msgStr',
                 'filename' : None, 'lineno' : None, 'offset' : None,
                 'end_offset': None}),
            (SyntaxError, ('msgStr', ('filenameStr', 'linenoStr', 'offsetStr',
                           'textStr', 'endLinenoStr', 'endOffsetStr')), {},
                {'offset' : 'offsetStr', 'text' : 'textStr',
                 'args' : ('msgStr', ('filenameStr', 'linenoStr',
                                      'offsetStr', 'textStr',
                                      'endLinenoStr', 'endOffsetStr')),
                 'print_file_and_line' : None, 'msg' : 'msgStr',
                 'filename' : 'filenameStr', 'lineno' : 'linenoStr',
                 'end_lineno': 'endLinenoStr', 'end_offset': 'endOffsetStr'}),
            (SyntaxError, ('msgStr', 'filenameStr', 'linenoStr', 'offsetStr',
                           'textStr', 'endLinenoStr', 'endOffsetStr',
                           'print_file_and_lineStr'), {},
                {'text' : None,
                 'args' : ('msgStr', 'filenameStr', 'linenoStr', 'offsetStr',
                           'textStr', 'endLinenoStr', 'endOffsetStr',
                           'print_file_and_lineStr'),
                 'print_file_and_line' : None, 'msg' : 'msgStr',
                 'filename' : None, 'lineno' : None, 'offset' : None,
                 'end_lineno': None, 'end_offset': None}),
            (UnicodeError, (), {}, {'args' : (),}),
            (UnicodeEncodeError, ('ascii', 'a', 0, 1,
                                  'ordinal not in range'), {},
                {'args' : ('ascii', 'a', 0, 1,
                                           'ordinal not in range'),
                 'encoding' : 'ascii', 'object' : 'a',
                 'start' : 0, 'reason' : 'ordinal not in range'}),
            (UnicodeDecodeError, ('ascii', bytearray(b'\xff'), 0, 1,
                                  'ordinal not in range'), {},
                {'args' : ('ascii', bytearray(b'\xff'), 0, 1,
                                           'ordinal not in range'),
                 'encoding' : 'ascii', 'object' : b'\xff',
                 'start' : 0, 'reason' : 'ordinal not in range'}),
            (UnicodeDecodeError, ('ascii', b'\xff', 0, 1,
                                  'ordinal not in range'), {},
                {'args' : ('ascii', b'\xff', 0, 1,
                                           'ordinal not in range'),
                 'encoding' : 'ascii', 'object' : b'\xff',
                 'start' : 0, 'reason' : 'ordinal not in range'}),
            (UnicodeTranslateError, ("\u3042", 0, 1, "ouch"), {},
                {'args' : ('\u3042', 0, 1, 'ouch'),
                 'object' : '\u3042', 'reason' : 'ouch',
                 'start' : 0, 'end' : 1}),
            (NaiveException, ('foo',), {},
                {'args': ('foo',), 'x': 'foo'}),
            (SlottedNaiveException, ('foo',), {},
                {'args': ('foo',), 'x': 'foo'}),
            (AttributeError, ('foo',), dict(name='name', obj='obj'),
                dict(args=('foo',), name='name', obj='obj')),
        ]
        try:
            # More tests are in test_WindowsError
            exceptionList.append(
                (WindowsError, (1, 'strErrorStr', 'filenameStr'), {},
                    {'args' : (1, 'strErrorStr'),
                     'strerror' : 'strErrorStr', 'winerror' : None,
                     'errno' : 1,
                     'filename' : 'filenameStr', 'filename2' : None})
            )
        except NameError:
            pass

        for exc, args, kwargs, expected in exceptionList:
            try:
                e = exc(*args, **kwargs)
            except:
                print(f"\nexc={exc!r}, args={args!r}", file=sys.stderr)
                # raise
            else:
                # Verify module name
                if not type(e).__name__.endswith('NaiveException'):
                    self.assertEqual(type(e).__module__, 'builtins')
                # Verify no ref leaks in Exc_str()
                s = str(e)
                for checkArgName in expected:
                    value = getattr(e, checkArgName)
                    self.assertEqual(repr(value),
                                     repr(expected[checkArgName]),
                                     '%r.%s == %r, expected %r' % (
                                     e, checkArgName,
                                     value, expected[checkArgName]))

                # test for pickling support
                for p in [pickle]:
                    for protocol in range(p.HIGHEST_PROTOCOL + 1):
                        s = p.dumps(e, protocol)
                        new = p.loads(s)
                        for checkArgName in expected:
                            got = repr(getattr(new, checkArgName))
                            if exc == AttributeError and checkArgName == 'obj':
                                # See GH-103352, we're not pickling
                                # obj at this point. So verify it's None.
                                want = repr(None)
                            else:
                                want = repr(expected[checkArgName])
                            self.assertEqual(got, want,
                                             'pickled "%r", attribute "%s' %
                                             (e, checkArgName))

    def test_setstate(self):
        e = Exception(42)
        e.blah = 53
        self.assertEqual(e.args, (42,))
        self.assertEqual(e.blah, 53)
        self.assertRaises(AttributeError, getattr, e, 'a')
        self.assertRaises(AttributeError, getattr, e, 'b')
        e.__setstate__({'a': 1 , 'b': 2})
        self.assertEqual(e.args, (42,))
        self.assertEqual(e.blah, 53)
        self.assertEqual(e.a, 1)
        self.assertEqual(e.b, 2)
        e.__setstate__({'a': 11, 'args': (1,2,3), 'blah': 35})
        self.assertEqual(e.args, (1,2,3))
        self.assertEqual(e.blah, 35)
        self.assertEqual(e.a, 11)
        self.assertEqual(e.b, 2)

    def test_invalid_setstate(self):
        e = Exception(42)
        with self.assertRaisesRegex(TypeError, "state is not a dictionary"):
            e.__setstate__(42)

    def test_notes(self):
        for e in [BaseException(1), Exception(2), ValueError(3)]:
            with self.subTest(e=e):
                self.assertFalse(hasattr(e, '__notes__'))
                e.add_note("My Note")
                self.assertEqual(e.__notes__, ["My Note"])

                with self.assertRaises(TypeError):
                    e.add_note(42)
                self.assertEqual(e.__notes__, ["My Note"])

                e.add_note("Your Note")
                self.assertEqual(e.__notes__, ["My Note", "Your Note"])

                del e.__notes__
                self.assertFalse(hasattr(e, '__notes__'))

                e.add_note("Our Note")
                self.assertEqual(e.__notes__, ["Our Note"])

                e.__notes__ = 42
                self.assertEqual(e.__notes__, 42)

                with self.assertRaises(TypeError):
                    e.add_note("will not work")
                self.assertEqual(e.__notes__, 42)

    def testWithTraceback(self):
        try:
            raise IndexError(4)
        except Exception as e:
            tb = e.__traceback__

        e = BaseException().with_traceback(tb)
        self.assertIsInstance(e, BaseException)
        self.assertEqual(e.__traceback__, tb)

        e = IndexError(5).with_traceback(tb)
        self.assertIsInstance(e, IndexError)
        self.assertEqual(e.__traceback__, tb)

        with torch._dynamo.error_on_graph_break(False):
            class MyException(Exception):
                pass

        e = MyException().with_traceback(tb)
        self.assertIsInstance(e, MyException)
        self.assertEqual(e.__traceback__, tb)

    def testInvalidTraceback(self):
        try:
            Exception().__traceback__ = 5
        except TypeError as e:
            self.assertIn("__traceback__ must be a traceback", str(e))
        else:
            self.fail("No exception raised")

    def test_invalid_setattr(self):
        TE = TypeError
        exc = Exception()
        msg = "'int' object is not iterable"
        self.assertRaisesRegex(TE, msg, setattr, exc, 'args', 1)
        msg = "__traceback__ must be a traceback or None"
        self.assertRaisesRegex(TE, msg, setattr, exc, '__traceback__', 1)
        msg = "exception cause must be None or derive from BaseException"
        self.assertRaisesRegex(TE, msg, setattr, exc, '__cause__', 1)
        msg = "exception context must be None or derive from BaseException"
        self.assertRaisesRegex(TE, msg, setattr, exc, '__context__', 1)

    def test_invalid_delattr(self):
        TE = TypeError
        try:
            raise IndexError(4)
        except Exception as e:
            exc = e

        msg = "may not be deleted"
        self.assertRaisesRegex(TE, msg, delattr, exc, 'args')
        self.assertRaisesRegex(TE, msg, delattr, exc, '__traceback__')
        self.assertRaisesRegex(TE, msg, delattr, exc, '__cause__')
        self.assertRaisesRegex(TE, msg, delattr, exc, '__context__')

    def testNoneClearsTracebackAttr(self):
        try:
            raise IndexError(4)
        except Exception as e:
            tb = e.__traceback__

        e = Exception()
        e.__traceback__ = tb
        e.__traceback__ = None
        self.assertEqual(e.__traceback__, None)

    def testChainingAttrs(self):
        e = Exception()
        self.assertIsNone(e.__context__)
        self.assertIsNone(e.__cause__)

        e = TypeError()
        self.assertIsNone(e.__context__)
        self.assertIsNone(e.__cause__)

        with torch._dynamo.error_on_graph_break(False):
            class MyException(OSError):
                pass

        e = MyException()
        self.assertIsNone(e.__context__)
        self.assertIsNone(e.__cause__)

    def testChainingDescriptors(self):
        try:
            raise Exception()
        except Exception as exc:
            e = exc

        self.assertIsNone(e.__context__)
        self.assertIsNone(e.__cause__)
        self.assertFalse(e.__suppress_context__)

        e.__context__ = NameError()
        e.__cause__ = None
        self.assertIsInstance(e.__context__, NameError)
        self.assertIsNone(e.__cause__)
        self.assertTrue(e.__suppress_context__)
        e.__suppress_context__ = False
        self.assertFalse(e.__suppress_context__)

    def testKeywordArgs(self):
        # test that builtin exception don't take keyword args,
        # but user-defined subclasses can if they want
        self.assertRaises(TypeError, BaseException, a=1)

        with torch._dynamo.error_on_graph_break(False):
            class DerivedException(BaseException):
                def __init__(self, fancy_arg):
                    BaseException.__init__(self)
                    self.fancy_arg = fancy_arg

        x = DerivedException(fancy_arg=42)
        self.assertEqual(x.fancy_arg, 42)

    @no_tracing
    def testInfiniteRecursion(self):
        def f():
            return f()
        self.assertRaises(RecursionError, f)

        def g():
            try:
                return g()
            except ValueError:
                return -1
        self.assertRaises(RecursionError, g)

    def test_str(self):
        # Make sure both instances and classes have a str representation.
        self.assertTrue(str(Exception))
        self.assertTrue(str(Exception('a')))
        self.assertTrue(str(Exception('a', 'b')))

    def test_exception_cleanup_names(self):
        # Make sure the local variable bound to the exception instance by
        # an "except" statement is only visible inside the except block.
        try:
            raise Exception()
        except Exception as e:
            self.assertIsInstance(e, Exception)
        self.assertNotIn('e', locals())
        with self.assertRaises(UnboundLocalError):
            e

    def test_exception_cleanup_names2(self):
        # Make sure the cleanup doesn't break if the variable is explicitly deleted.
        try:
            raise Exception()
        except Exception as e:
            self.assertIsInstance(e, Exception)
            del e
        self.assertNotIn('e', locals())
        with self.assertRaises(UnboundLocalError):
            e

    def testExceptionCleanupState(self):
        # Make sure exception state is cleaned up as soon as the except
        # block is left. See #2507

        with torch._dynamo.error_on_graph_break(False):
            class MyException(Exception):
                def __init__(self, obj):
                    self.obj = obj
            class MyObj:
                pass

        def inner_raising_func():
            # Create some references in exception value and traceback
            local_ref = obj
            raise MyException(obj)

        # Qualified "except" with "as"
        obj = MyObj()
        wr = weakref.ref(obj)
        try:
            inner_raising_func()
        except MyException as e:
            pass
        obj = None
        gc_collect()  # For PyPy or other GCs.
        obj = wr()
        self.assertIsNone(obj)

        # Qualified "except" without "as"
        obj = MyObj()
        wr = weakref.ref(obj)
        try:
            inner_raising_func()
        except MyException:
            pass
        obj = None
        gc_collect()  # For PyPy or other GCs.
        obj = wr()
        self.assertIsNone(obj)

        # Bare "except"
        obj = MyObj()
        wr = weakref.ref(obj)
        try:
            inner_raising_func()
        except:
            pass
        obj = None
        gc_collect()  # For PyPy or other GCs.
        obj = wr()
        self.assertIsNone(obj)

        # "except" with premature block leave
        obj = MyObj()
        wr = weakref.ref(obj)
        for i in [0]:
            try:
                inner_raising_func()
            except:
                break
        obj = None
        gc_collect()  # For PyPy or other GCs.
        obj = wr()
        self.assertIsNone(obj)

        # "except" block raising another exception
        obj = MyObj()
        wr = weakref.ref(obj)
        try:
            try:
                inner_raising_func()
            except:
                raise KeyError
        except KeyError as e:
            # We want to test that the except block above got rid of
            # the exception raised in inner_raising_func(), but it
            # also ends up in the __context__ of the KeyError, so we
            # must clear the latter manually for our test to succeed.
            e.__context__ = None
            obj = None
            gc_collect()  # For PyPy or other GCs.
            obj = wr()
            # guarantee no ref cycles on CPython (don't gc_collect)
            if check_impl_detail(cpython=False):
                gc_collect()
            self.assertIsNone(obj)

        # Some complicated construct
        obj = MyObj()
        wr = weakref.ref(obj)
        try:
            inner_raising_func()
        except MyException:
            try:
                try:
                    raise
                finally:
                    raise
            except MyException:
                pass
        obj = None
        if check_impl_detail(cpython=False):
            gc_collect()
        obj = wr()
        self.assertIsNone(obj)

        # Inside an exception-silencing "with" block
        with torch._dynamo.error_on_graph_break(False):
            class Context:
                def __enter__(self):
                    return self
                def __exit__ (self, exc_type, exc_value, exc_tb):
                    return True
        obj = MyObj()
        wr = weakref.ref(obj)
        with Context():
            inner_raising_func()
        obj = None
        if check_impl_detail(cpython=False):
            gc_collect()
        obj = wr()
        self.assertIsNone(obj)

    def test_exception_target_in_nested_scope(self):
        # issue 4617: This used to raise a SyntaxError
        # "can not delete variable 'e' referenced in nested scope"
        def print_error():
            e
        try:
            something
        except Exception as e:
            print_error()
            # implicit "del e" here

    def test_generator_leaking(self):
        # Test that generator exception state doesn't leak into the calling
        # frame
        def yield_raise():
            try:
                raise KeyError("caught")
            except KeyError:
                yield sys.exception()
                yield sys.exception()
            yield sys.exception()
        g = yield_raise()
        self.assertIsInstance(next(g), KeyError)
        self.assertIsNone(sys.exception())
        self.assertIsInstance(next(g), KeyError)
        self.assertIsNone(sys.exception())
        self.assertIsNone(next(g))

        # Same test, but inside an exception handler
        try:
            raise TypeError("foo")
        except TypeError:
            g = yield_raise()
            self.assertIsInstance(next(g), KeyError)
            self.assertIsInstance(sys.exception(), TypeError)
            self.assertIsInstance(next(g), KeyError)
            self.assertIsInstance(sys.exception(), TypeError)
            self.assertIsInstance(next(g), TypeError)
            del g
            self.assertIsInstance(sys.exception(), TypeError)

    def test_generator_leaking2(self):
        # See issue 12475.
        def g():
            yield
        try:
            raise RuntimeError
        except RuntimeError:
            it = g()
            next(it)
        try:
            next(it)
        except StopIteration:
            pass
        self.assertIsNone(sys.exception())

    def test_generator_leaking3(self):
        # See issue #23353.  When gen.throw() is called, the caller's
        # exception state should be save and restored.
        def g():
            try:
                yield
            except ZeroDivisionError:
                yield sys.exception()
        it = g()
        next(it)
        try:
            1/0
        except ZeroDivisionError as e:
            self.assertIs(sys.exception(), e)
            gen_exc = it.throw(e)
            self.assertIs(sys.exception(), e)
            self.assertIs(gen_exc, e)
        self.assertIsNone(sys.exception())

    def test_generator_leaking4(self):
        # See issue #23353.  When an exception is raised by a generator,
        # the caller's exception state should still be restored.
        def g():
            try:
                1/0
            except ZeroDivisionError:
                yield sys.exception()
                raise
        it = g()
        try:
            raise TypeError
        except TypeError:
            # The caller's exception state (TypeError) is temporarily
            # saved in the generator.
            tp = type(next(it))
        self.assertIs(tp, ZeroDivisionError)
        try:
            next(it)
            # We can't check it immediately, but while next() returns
            # with an exception, it shouldn't have restored the old
            # exception state (TypeError).
        except ZeroDivisionError as e:
            self.assertIs(sys.exception(), e)
        # We used to find TypeError here.
        self.assertIsNone(sys.exception())

    def test_generator_doesnt_retain_old_exc(self):
        def g():
            self.assertIsInstance(sys.exception(), RuntimeError)
            yield
            self.assertIsNone(sys.exception())
        it = g()
        try:
            raise RuntimeError
        except RuntimeError:
            next(it)
        self.assertRaises(StopIteration, next, it)

    def test_generator_finalizing_and_sys_exception(self):
        # See #7173
        def simple_gen():
            yield 1
        def run_gen():
            gen = simple_gen()
            try:
                raise RuntimeError
            except RuntimeError:
                return next(gen)
        run_gen()
        gc_collect()
        self.assertIsNone(sys.exception())

    def _check_generator_cleanup_exc_state(self, testfunc):
        # Issue #12791: exception state is cleaned up as soon as a generator
        # is closed (reference cycles are broken).
        with torch._dynamo.error_on_graph_break(False):
            class MyException(Exception):
                def __init__(self, obj):
                    self.obj = obj
            class MyObj:
                pass

        def raising_gen():
            try:
                raise MyException(obj)
            except MyException:
                yield

        obj = MyObj()
        wr = weakref.ref(obj)
        g = raising_gen()
        next(g)
        testfunc(g)
        g = obj = None
        gc_collect()  # For PyPy or other GCs.
        obj = wr()
        self.assertIsNone(obj)

    def test_generator_throw_cleanup_exc_state(self):
        def do_throw(g):
            try:
                g.throw(RuntimeError())
            except RuntimeError:
                pass
        self._check_generator_cleanup_exc_state(do_throw)

    def test_generator_close_cleanup_exc_state(self):
        def do_close(g):
            g.close()
        self._check_generator_cleanup_exc_state(do_close)

    def test_generator_del_cleanup_exc_state(self):
        def do_del(g):
            g = None
        self._check_generator_cleanup_exc_state(do_del)

    def test_generator_next_cleanup_exc_state(self):
        def do_next(g):
            try:
                next(g)
            except StopIteration:
                pass
            else:
                self.fail("should have raised StopIteration")
        self._check_generator_cleanup_exc_state(do_next)

    def test_generator_send_cleanup_exc_state(self):
        def do_send(g):
            try:
                g.send(None)
            except StopIteration:
                pass
            else:
                self.fail("should have raised StopIteration")
        self._check_generator_cleanup_exc_state(do_send)

    def test_3114(self):
        # Bug #3114: in its destructor, MyObject retrieves a pointer to
        # obsolete and/or deallocated objects.
        with torch._dynamo.error_on_graph_break(False):
            class MyObject:
                def __del__(self):
                    nonlocal e
                    e = sys.exception()
        e = ()
        try:
            raise Exception(MyObject())
        except:
            pass
        gc_collect()  # For PyPy or other GCs.
        self.assertIsNone(e)

    def test_raise_does_not_create_context_chain_cycle(self):
        with torch._dynamo.error_on_graph_break(False):
            class A(Exception):
                pass
            class B(Exception):
                pass
            class C(Exception):
                pass

        # Create a context chain:
        # C -> B -> A
        # Then raise A in context of C.
        try:
            try:
                raise A
            except A as a_:
                a = a_
                try:
                    raise B
                except B as b_:
                    b = b_
                    try:
                        raise C
                    except C as c_:
                        c = c_
                        self.assertIsInstance(a, A)
                        self.assertIsInstance(b, B)
                        self.assertIsInstance(c, C)
                        self.assertIsNone(a.__context__)
                        self.assertIs(b.__context__, a)
                        self.assertIs(c.__context__, b)
                        raise a
        except A as e:
            exc = e

        # Expect A -> C -> B, without cycle
        self.assertIs(exc, a)
        self.assertIs(a.__context__, c)
        self.assertIs(c.__context__, b)
        self.assertIsNone(b.__context__)

    def test_no_hang_on_context_chain_cycle1(self):
        # See issue 25782. Cycle in context chain.

        def cycle():
            try:
                raise ValueError(1)
            except ValueError as ex:
                ex.__context__ = ex
                raise TypeError(2)

        try:
            cycle()
        except Exception as e:
            exc = e

        self.assertIsInstance(exc, TypeError)
        self.assertIsInstance(exc.__context__, ValueError)
        self.assertIs(exc.__context__.__context__, exc.__context__)

    def test_no_hang_on_context_chain_cycle2(self):
        # See issue 25782. Cycle at head of context chain.

        with torch._dynamo.error_on_graph_break(False):
            class A(Exception):
                pass
            class B(Exception):
                pass
            class C(Exception):
                pass

        # Context cycle:
        # +-----------+
        # V           |
        # C --> B --> A
        with self.assertRaises(C) as cm:
            try:
                raise A()
            except A as _a:
                a = _a
                try:
                    raise B()
                except B as _b:
                    b = _b
                    try:
                        raise C()
                    except C as _c:
                        c = _c
                        a.__context__ = c
                        raise c

        self.assertIs(cm.exception, c)
        # Verify the expected context chain cycle
        self.assertIs(c.__context__, b)
        self.assertIs(b.__context__, a)
        self.assertIs(a.__context__, c)

    def test_no_hang_on_context_chain_cycle3(self):
        # See issue 25782. Longer context chain with cycle.

        with torch._dynamo.error_on_graph_break(False):
            class A(Exception):
                pass
            class B(Exception):
                pass
            class C(Exception):
                pass
            class D(Exception):
                pass
            class E(Exception):
                pass

        # Context cycle:
        #             +-----------+
        #             V           |
        # E --> D --> C --> B --> A
        with self.assertRaises(E) as cm:
            try:
                raise A()
            except A as _a:
                a = _a
                try:
                    raise B()
                except B as _b:
                    b = _b
                    try:
                        raise C()
                    except C as _c:
                        c = _c
                        a.__context__ = c
                        try:
                            raise D()
                        except D as _d:
                            d = _d
                            e = E()
                            raise e

        self.assertIs(cm.exception, e)
        # Verify the expected context chain cycle
        self.assertIs(e.__context__, d)
        self.assertIs(d.__context__, c)
        self.assertIs(c.__context__, b)
        self.assertIs(b.__context__, a)
        self.assertIs(a.__context__, c)

    def test_context_of_exception_in_try_and_finally(self):
        try:
            try:
                te = TypeError(1)
                raise te
            finally:
                ve = ValueError(2)
                raise ve
        except Exception as e:
            exc = e

        self.assertIs(exc, ve)
        self.assertIs(exc.__context__, te)

    def test_context_of_exception_in_except_and_finally(self):
        try:
            try:
                te = TypeError(1)
                raise te
            except:
                ve = ValueError(2)
                raise ve
            finally:
                oe = OSError(3)
                raise oe
        except Exception as e:
            exc = e

        self.assertIs(exc, oe)
        self.assertIs(exc.__context__, ve)
        self.assertIs(exc.__context__.__context__, te)

    def test_context_of_exception_in_else_and_finally(self):
        try:
            try:
                pass
            except:
                pass
            else:
                ve = ValueError(1)
                raise ve
            finally:
                oe = OSError(2)
                raise oe
        except Exception as e:
            exc = e

        self.assertIs(exc, oe)
        self.assertIs(exc.__context__, ve)

    def test_unicode_change_attributes(self):
        # See issue 7309. This was a crasher.

        u = UnicodeEncodeError('baz', 'xxxxx', 1, 5, 'foo')
        self.assertEqual(str(u), "'baz' codec can't encode characters in position 1-4: foo")
        u.end = 2
        self.assertEqual(str(u), "'baz' codec can't encode character '\\x78' in position 1: foo")
        u.end = 5
        u.reason = 0x345345345345345345
        self.assertEqual(str(u), "'baz' codec can't encode characters in position 1-4: 965230951443685724997")
        u.encoding = 4000
        self.assertEqual(str(u), "'4000' codec can't encode characters in position 1-4: 965230951443685724997")
        u.start = 1000
        self.assertEqual(str(u), "'4000' codec can't encode characters in position 1000-4: 965230951443685724997")

        u = UnicodeDecodeError('baz', b'xxxxx', 1, 5, 'foo')
        self.assertEqual(str(u), "'baz' codec can't decode bytes in position 1-4: foo")
        u.end = 2
        self.assertEqual(str(u), "'baz' codec can't decode byte 0x78 in position 1: foo")
        u.end = 5
        u.reason = 0x345345345345345345
        self.assertEqual(str(u), "'baz' codec can't decode bytes in position 1-4: 965230951443685724997")
        u.encoding = 4000
        self.assertEqual(str(u), "'4000' codec can't decode bytes in position 1-4: 965230951443685724997")
        u.start = 1000
        self.assertEqual(str(u), "'4000' codec can't decode bytes in position 1000-4: 965230951443685724997")

        u = UnicodeTranslateError('xxxx', 1, 5, 'foo')
        self.assertEqual(str(u), "can't translate characters in position 1-4: foo")
        u.end = 2
        self.assertEqual(str(u), "can't translate character '\\x78' in position 1: foo")
        u.end = 5
        u.reason = 0x345345345345345345
        self.assertEqual(str(u), "can't translate characters in position 1-4: 965230951443685724997"
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
- **Context Manager**: Implements context manager protocol
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
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
python docs/test/dynamo/cpython/3_13/test_exceptions.py_docs.md
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

- **File Documentation**: `test_exceptions.py_docs.md_docs.md`
- **Keyword Index**: `test_exceptions.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
