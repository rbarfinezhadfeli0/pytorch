# Documentation: `docs/test/dynamo/cpython/3_13/test_list.py_docs.md`

## File Metadata

- **Path**: `docs/test/dynamo/cpython/3_13/test_list.py_docs.md`
- **Size**: 16,994 bytes (16.60 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/dynamo/cpython/3_13/test_list.py`

## File Metadata

- **Path**: `test/dynamo/cpython/3_13/test_list.py`
- **Size**: 13,426 bytes (13.11 KB)
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
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_list.py

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

import sys
import textwrap
import list_tests
from test.support import cpython_only
from test.support.script_helper import assert_python_ok
import pickle
import unittest

class ListTest(list_tests.CommonTest):
    type2test = list

    def test_basic(self):
        self.assertEqual(list([]), [])
        l0_3 = [0, 1, 2, 3]
        l0_3_bis = list(l0_3)
        self.assertEqual(l0_3, l0_3_bis)
        self.assertTrue(l0_3 is not l0_3_bis)
        self.assertEqual(list(()), [])
        self.assertEqual(list((0, 1, 2, 3)), [0, 1, 2, 3])
        self.assertEqual(list(''), [])
        self.assertEqual(list('spam'), ['s', 'p', 'a', 'm'])
        self.assertEqual(list(x for x in range(10) if x % 2),
                         [1, 3, 5, 7, 9])

        if sys.maxsize == 0x7fffffff:
            # This test can currently only work on 32-bit machines.
            # XXX If/when PySequence_Length() returns a ssize_t, it should be
            # XXX re-enabled.
            # Verify clearing of bug #556025.
            # This assumes that the max data size (sys.maxint) == max
            # address size this also assumes that the address size is at
            # least 4 bytes with 8 byte addresses, the bug is not well
            # tested
            #
            # Note: This test is expected to SEGV under Cygwin 1.3.12 or
            # earlier due to a newlib bug.  See the following mailing list
            # thread for the details:
            self.assertRaises(MemoryError, list, range(sys.maxsize // 2))

        # This code used to segfault in Py2.4a3
        x = []
        x.extend(-y for y in x)
        self.assertEqual(x, [])

    def test_keyword_args(self):
        with self.assertRaisesRegex(TypeError, 'keyword argument'):
            list(sequence=[])

    def test_keywords_in_subclass(self):
        with torch._dynamo.error_on_graph_break(False):
            class subclass(list):
                pass
        u = subclass([1, 2])
        self.assertIs(type(u), subclass)
        self.assertEqual(list(u), [1, 2])
        with self.assertRaises(TypeError):
            subclass(sequence=())

        with torch._dynamo.error_on_graph_break(False):
            class subclass_with_init(list):
                def __init__(self, seq, newarg=None):
                    super().__init__(seq)
                    self.newarg = newarg
        u = subclass_with_init([1, 2], newarg=3)
        self.assertIs(type(u), subclass_with_init)
        self.assertEqual(list(u), [1, 2])
        self.assertEqual(u.newarg, 3)

        with torch._dynamo.error_on_graph_break(False):
            class subclass_with_new(list):
                def __new__(cls, seq, newarg=None):
                    self = super().__new__(cls, seq)
                    self.newarg = newarg
                    return self
        u = subclass_with_new([1, 2], newarg=3)
        self.assertIs(type(u), subclass_with_new)
        self.assertEqual(list(u), [1, 2])
        self.assertEqual(u.newarg, 3)

    def test_truth(self):
        super().test_truth()
        self.assertTrue(not [])
        self.assertTrue([42])

    def test_identity(self):
        self.assertTrue([] is not [])

    def test_len(self):
        super().test_len()
        self.assertEqual(len([]), 0)
        self.assertEqual(len([0]), 1)
        self.assertEqual(len([0, 1, 2]), 3)

    def test_overflow(self):
        lst = [4, 5, 6, 7]
        n = int((sys.maxsize*2+2) // len(lst))
        def mul(a, b): return a * b
        def imul(a, b): a *= b
        self.assertRaises((MemoryError, OverflowError), mul, lst, n)
        self.assertRaises((MemoryError, OverflowError), imul, lst, n)

    def test_empty_slice(self):
        x = []
        x[:] = x
        self.assertEqual(x, [])

    def test_list_resize_overflow(self):
        # gh-97616: test new_allocated * sizeof(PyObject*) overflow
        # check in list_resize()
        lst = [0] * 65
        del lst[1:]
        self.assertEqual(len(lst), 1)

        size = sys.maxsize
        with self.assertRaises((MemoryError, OverflowError)):
            lst * size
        with self.assertRaises((MemoryError, OverflowError)):
            lst *= size

    def test_repr_mutate(self):
        with torch._dynamo.error_on_graph_break(False):
            class Obj:
                @staticmethod
                def __repr__():
                    try:
                        mylist.pop()
                    except IndexError:
                        pass
                    return 'obj'

        mylist = [Obj() for _ in range(5)]
        self.assertEqual(repr(mylist), '[obj, obj, obj]')

    def test_repr_large(self):
        # Check the repr of large list objects
        def check(n):
            l = [0] * n
            s = repr(l)
            self.assertEqual(s,
                '[' + ', '.join(['0'] * n) + ']')
        check(10)       # check our checking code
        check(1000000)

    def test_iterator_pickle(self):
        orig = self.type2test([4, 5, 6, 7])
        data = [10, 11, 12, 13, 14, 15]
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            # initial iterator
            itorig = iter(orig)
            d = pickle.dumps((itorig, orig), proto)
            it, a = pickle.loads(d)
            a[:] = data
            self.assertEqual(type(it), type(itorig))
            self.assertEqual(list(it), data)

            # running iterator
            next(itorig)
            d = pickle.dumps((itorig, orig), proto)
            it, a = pickle.loads(d)
            a[:] = data
            self.assertEqual(type(it), type(itorig))
            self.assertEqual(list(it), data[1:])

            # empty iterator
            for i in range(1, len(orig)):
                next(itorig)
            d = pickle.dumps((itorig, orig), proto)
            it, a = pickle.loads(d)
            a[:] = data
            self.assertEqual(type(it), type(itorig))
            self.assertEqual(list(it), data[len(orig):])

            # exhausted iterator
            self.assertRaises(StopIteration, next, itorig)
            d = pickle.dumps((itorig, orig), proto)
            it, a = pickle.loads(d)
            a[:] = data
            self.assertEqual(list(it), [])

    def test_reversed_pickle(self):
        orig = self.type2test([4, 5, 6, 7])
        data = [10, 11, 12, 13, 14, 15]
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            # initial iterator
            itorig = reversed(orig)
            d = pickle.dumps((itorig, orig), proto)
            it, a = pickle.loads(d)
            a[:] = data
            self.assertEqual(type(it), type(itorig))
            self.assertEqual(list(it), data[len(orig)-1::-1])

            # running iterator
            next(itorig)
            d = pickle.dumps((itorig, orig), proto)
            it, a = pickle.loads(d)
            a[:] = data
            self.assertEqual(type(it), type(itorig))
            self.assertEqual(list(it), data[len(orig)-2::-1])

            # empty iterator
            for i in range(1, len(orig)):
                next(itorig)
            d = pickle.dumps((itorig, orig), proto)
            it, a = pickle.loads(d)
            a[:] = data
            self.assertEqual(type(it), type(itorig))
            self.assertEqual(list(it), [])

            # exhausted iterator
            self.assertRaises(StopIteration, next, itorig)
            d = pickle.dumps((itorig, orig), proto)
            it, a = pickle.loads(d)
            a[:] = data
            self.assertEqual(list(it), [])

    def test_step_overflow(self):
        a = [0, 1, 2, 3, 4]
        a[1::sys.maxsize] = [0]
        self.assertEqual(a[3::sys.maxsize], [3])

    def test_no_comdat_folding(self):
        # Issue 8847: In the PGO build, the MSVC linker's COMDAT folding
        # optimization causes failures in code that relies on distinct
        # function addresses.
        with torch._dynamo.error_on_graph_break(False):
            class L(list): pass
        with self.assertRaises(TypeError):
            (3,) + L([1,2])

    def test_equal_operator_modifying_operand(self):
        # test fix for seg fault reported in bpo-38588 part 2.
        with torch._dynamo.error_on_graph_break(False):
            class X:
                def __eq__(self,other) :
                    list2.clear()
                    return NotImplemented

            class Y:
                def __eq__(self, other):
                    list1.clear()
                    return NotImplemented

            class Z:
                def __eq__(self, other):
                    list3.clear()
                    return NotImplemented

        list1 = [X()]
        list2 = [Y()]
        self.assertTrue(list1 == list2)

        list3 = [Z()]
        list4 = [1]
        self.assertFalse(list3 == list4)

    def test_lt_operator_modifying_operand(self):
        with torch._dynamo.error_on_graph_break(False):
            # See gh-120298
            class evil:
                def __lt__(self, other):
                    other.clear()
                    return NotImplemented

        a = [[evil()]]
        with self.assertRaises(TypeError):
            a[0] < a

    def test_list_index_modifing_operand(self):
        with torch._dynamo.error_on_graph_break(False):
            # See gh-120384
            class evil:
                def __init__(self, lst):
                    self.lst = lst
                def __iter__(self):
                    yield from self.lst
                    self.lst.clear()

        lst = list(range(5))
        operand = evil(lst)
        with self.assertRaises(ValueError):
            lst[::-1] = operand

    @cpython_only
    def test_preallocation(self):
        iterable = [0] * 10
        iter_size = sys.getsizeof(iterable)

        self.assertEqual(iter_size, sys.getsizeof(list([0] * 10)))
        self.assertEqual(iter_size, sys.getsizeof(list(range(10))))

    def test_count_index_remove_crashes(self):
        # bpo-38610: The count(), index(), and remove() methods were not
        # holding strong references to list elements while calling
        # PyObject_RichCompareBool().
        with torch._dynamo.error_on_graph_break(False):
            class X:
                def __eq__(self, other):
                    lst.clear()
                    return NotImplemented

        lst = [X()]
        with self.assertRaises(ValueError):
            lst.index(lst)

        with torch._dynamo.error_on_graph_break(False):
            class L(list):
                def __eq__(self, other):
                    str(other)
                    return NotImplemented

        lst = L([X()])
        lst.count(lst)

        lst = L([X()])
        with self.assertRaises(ValueError):
            lst.remove(lst)

        # bpo-39453: list.__contains__ was not holding strong references
        # to list elements while calling PyObject_RichCompareBool().
        lst = [X(), X()]
        3 in lst
        lst = [X(), X()]
        X() in lst

    def test_tier2_invalidates_iterator(self):
        # GH-121012
        for _ in range(100):
            a = [1, 2, 3]
            it = iter(a)
            for _ in it:
                pass
            a.append(4)
            self.assertEqual(list(it), [])

    @unittest.skip("Fails on python <=3.13.2 and passes on >=3.13.3")
    def test_deopt_from_append_list(self):
        # gh-132011: it used to crash, because
        # of `CALL_LIST_APPEND` specialization failure.
        code = textwrap.dedent("""
            l = []
            def lappend(l, x, y):
                l.append((x, y))
            for x in range(3):
                lappend(l, None, None)
            try:
                lappend(list, None, None)
            except TypeError:
                pass
            else:
                raise AssertionError
        """)

        rc, _, _ = assert_python_ok("-c", code)
        self.assertEqual(rc, 0)

if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 14 class(es) and 38 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `RedirectImportFinder`, `ListTest`, `subclass`, `subclass_with_init`, `subclass_with_new`, `Obj`, `L`, `X`, `Y`, `Z`, `evil`, `evil`, `X`, `L`

**Functions defined**: `find_spec`, `test_basic`, `test_keyword_args`, `test_keywords_in_subclass`, `__init__`, `__new__`, `test_truth`, `test_identity`, `test_len`, `test_overflow`, `mul`, `imul`, `test_empty_slice`, `test_list_resize_overflow`, `test_repr_mutate`, `__repr__`, `test_repr_large`, `check`, `test_iterator_pickle`, `test_reversed_pickle`

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
- `textwrap`
- `list_tests`
- `test.support`: cpython_only
- `test.support.script_helper`: assert_python_ok
- `pickle`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/dynamo/cpython/3_13/test_list.py
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

- **File Documentation**: `test_list.py_docs.md`
- **Keyword Index**: `test_list.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

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

- This file appears to involve **GPU/parallel computing** capabilities.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/dynamo/cpython/3_13/test_list.py_docs.md
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

- **File Documentation**: `test_list.py_docs.md_docs.md`
- **Keyword Index**: `test_list.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
