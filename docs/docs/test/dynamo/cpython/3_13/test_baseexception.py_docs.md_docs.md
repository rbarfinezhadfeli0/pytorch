# Documentation: `docs/test/dynamo/cpython/3_13/test_baseexception.py_docs.md`

## File Metadata

- **Path**: `docs/test/dynamo/cpython/3_13/test_baseexception.py_docs.md`
- **Size**: 13,136 bytes (12.83 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/dynamo/cpython/3_13/test_baseexception.py`

## File Metadata

- **Path**: `test/dynamo/cpython/3_13/test_baseexception.py`
- **Size**: 9,629 bytes (9.40 KB)
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
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_baseexception.py

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

import unittest
import builtins
import os
from platform import system as platform_system


class ExceptionClassTests(__TestCase):

    """Tests for anything relating to exception objects themselves (e.g.,
    inheritance hierarchy)"""

    def test_builtins_new_style(self):
        self.assertTrue(issubclass(Exception, object))

    def verify_instance_interface(self, ins):
        for attr in ("args", "__str__", "__repr__"):
            self.assertTrue(hasattr(ins, attr),
                    "%s missing %s attribute" %
                        (ins.__class__.__name__, attr))

    def test_inheritance(self):
        # Make sure the inheritance hierarchy matches the documentation
        exc_set = set()
        for object_ in builtins.__dict__.values():
            try:
                if issubclass(object_, BaseException):
                    exc_set.add(object_.__name__)
            except TypeError:
                pass

        inheritance_tree = open(
                os.path.join(os.path.split(__file__)[0], 'exception_hierarchy.txt'),
                encoding="utf-8")
        try:
            superclass_name = inheritance_tree.readline().rstrip()
            try:
                last_exc = getattr(builtins, superclass_name)
            except AttributeError:
                self.fail("base class %s not a built-in" % superclass_name)
            self.assertIn(superclass_name, exc_set,
                          '%s not found' % superclass_name)
            exc_set.discard(superclass_name)
            superclasses = []  # Loop will insert base exception
            last_depth = 0
            for exc_line in inheritance_tree:
                exc_line = exc_line.rstrip()
                depth = exc_line.rindex('─')
                exc_name = exc_line[depth+2:]  # Slice past space
                if '(' in exc_name:
                    paren_index = exc_name.index('(')
                    platform_name = exc_name[paren_index+1:-1]
                    exc_name = exc_name[:paren_index-1]  # Slice off space
                    if platform_system() != platform_name:
                        exc_set.discard(exc_name)
                        continue
                if '[' in exc_name:
                    left_bracket = exc_name.index('[')
                    exc_name = exc_name[:left_bracket-1]  # cover space
                try:
                    exc = getattr(builtins, exc_name)
                except AttributeError:
                    self.fail("%s not a built-in exception" % exc_name)
                if last_depth < depth:
                    superclasses.append((last_depth, last_exc))
                elif last_depth > depth:
                    while superclasses[-1][0] >= depth:
                        superclasses.pop()
                self.assertTrue(issubclass(exc, superclasses[-1][1]),
                "%s is not a subclass of %s" % (exc.__name__,
                    superclasses[-1][1].__name__))
                try:  # Some exceptions require arguments; just skip them
                    self.verify_instance_interface(exc())
                except TypeError:
                    pass
                self.assertIn(exc_name, exc_set)
                exc_set.discard(exc_name)
                last_exc = exc
                last_depth = depth
        finally:
            inheritance_tree.close()
        self.assertEqual(len(exc_set), 0, "%s not accounted for" % exc_set)

    interface_tests = ("length", "args", "str", "repr")

    def interface_test_driver(self, results):
        for test_name, (given, expected) in zip(self.interface_tests, results):
            self.assertEqual(given, expected, "%s: %s != %s" % (test_name,
                given, expected))

    def test_interface_single_arg(self):
        # Make sure interface works properly when given a single argument
        arg = "spam"
        exc = Exception(arg)
        results = ([len(exc.args), 1], [exc.args[0], arg],
                   [str(exc), str(arg)],
            [repr(exc), '%s(%r)' % (exc.__class__.__name__, arg)])
        self.interface_test_driver(results)

    def test_interface_multi_arg(self):
        # Make sure interface correct when multiple arguments given
        arg_count = 3
        args = tuple(range(arg_count))
        exc = Exception(*args)
        results = ([len(exc.args), arg_count], [exc.args, args],
                [str(exc), str(args)],
                [repr(exc), exc.__class__.__name__ + repr(exc.args)])
        self.interface_test_driver(results)

    def test_interface_no_arg(self):
        # Make sure that with no args that interface is correct
        exc = Exception()
        results = ([len(exc.args), 0], [exc.args, tuple()],
                [str(exc), ''],
                [repr(exc), exc.__class__.__name__ + '()'])
        self.interface_test_driver(results)

    def test_setstate_refcount_no_crash(self):
        # gh-97591: Acquire strong reference before calling tp_hash slot
        # in PyObject_SetAttr.
        import gc
        d = {}
        with torch._dynamo.error_on_graph_break(False):
            class HashThisKeyWillClearTheDict(str):
                def __hash__(self) -> int:
                    d.clear()
                    return super().__hash__()
            class Value(str):
                pass
        exc = Exception()

        d[HashThisKeyWillClearTheDict()] = Value()  # refcount of Value() is 1 now

        # Exception.__setstate__ should acquire a strong reference of key and
        # value in the dict. Otherwise, Value()'s refcount would go below
        # zero in the tp_hash call in PyObject_SetAttr(), and it would cause
        # crash in GC.
        exc.__setstate__(d)  # __hash__() is called again here, clearing the dict.

        # This GC would crash if the refcount of Value() goes below zero.
        gc.collect()


class UsageTests(__TestCase):

    """Test usage of exceptions"""

    def raise_fails(self, object_):
        """Make sure that raising 'object_' triggers a TypeError."""
        try:
            raise object_
        except TypeError:
            return  # What is expected.
        self.fail("TypeError expected for raising %s" % type(object_))

    def catch_fails(self, object_):
        """Catching 'object_' should raise a TypeError."""
        try:
            try:
                raise Exception
            except object_:
                pass
        except TypeError:
            pass
        except Exception:
            self.fail("TypeError expected when catching %s" % type(object_))

        try:
            try:
                raise Exception
            except (object_,):
                pass
        except TypeError:
            return
        except Exception:
            self.fail("TypeError expected when catching %s as specified in a "
                        "tuple" % type(object_))

    def test_raise_new_style_non_exception(self):
        # You cannot raise a new-style class that does not inherit from
        # BaseException; the ability was not possible until BaseException's
        # introduction so no need to support new-style objects that do not
        # inherit from it.
        with torch._dynamo.error_on_graph_break(False):
            class NewStyleClass(object):
                pass
        self.raise_fails(NewStyleClass)
        self.raise_fails(NewStyleClass())

    def test_raise_string(self):
        # Raising a string raises TypeError.
        self.raise_fails("spam")

    def test_catch_non_BaseException(self):
        # Trying to catch an object that does not inherit from BaseException
        # is not allowed.
        with torch._dynamo.error_on_graph_break(False):
            class NonBaseException(object):
                pass
        self.catch_fails(NonBaseException)
        self.catch_fails(NonBaseException())

    def test_catch_BaseException_instance(self):
        # Catching an instance of a BaseException subclass won't work.
        self.catch_fails(BaseException())

    def test_catch_string(self):
        # Catching a string is bad.
        self.catch_fails("spam")


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 10 class(es) and 17 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `RedirectImportFinder`, `ExceptionClassTests`, `HashThisKeyWillClearTheDict`, `Value`, `UsageTests`, `NewStyleClass`, `NonBaseException`

**Functions defined**: `find_spec`, `test_builtins_new_style`, `verify_instance_interface`, `test_inheritance`, `interface_test_driver`, `test_interface_single_arg`, `test_interface_multi_arg`, `test_interface_no_arg`, `test_setstate_refcount_no_crash`, `__hash__`, `raise_fails`, `catch_fails`, `test_raise_new_style_non_exception`, `test_raise_string`, `test_catch_non_BaseException`, `test_catch_BaseException_instance`, `test_catch_string`

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
- `builtins`
- `os`
- `platform`: system as platform_system
- `gc`


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/dynamo/cpython/3_13/test_baseexception.py
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

- **File Documentation**: `test_baseexception.py_docs.md`
- **Keyword Index**: `test_baseexception.py_kw.md`
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
python docs/test/dynamo/cpython/3_13/test_baseexception.py_docs.md
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

- **File Documentation**: `test_baseexception.py_docs.md_docs.md`
- **Keyword Index**: `test_baseexception.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
