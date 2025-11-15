# Documentation: `test/dynamo/cpython/3_13/test_generator_stop.py`

## File Metadata

- **Path**: `test/dynamo/cpython/3_13/test_generator_stop.py`
- **Size**: 2,412 bytes (2.36 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
from __future__ import generator_stop

# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_generator_stop.py

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


class TestPEP479(__TestCase):
    def test_stopiteration_wrapping(self):
        def f():
            raise StopIteration
        def g():
            yield f()
        with self.assertRaisesRegex(RuntimeError,
                                    "generator raised StopIteration"):
            next(g())

    def test_stopiteration_wrapping_context(self):
        def f():
            raise StopIteration
        def g():
            yield f()

        try:
            next(g())
        except RuntimeError as exc:
            self.assertIs(type(exc.__cause__), StopIteration)
            self.assertIs(type(exc.__context__), StopIteration)
            self.assertTrue(exc.__suppress_context__)
        else:
            self.fail('__cause__, __context__, or __suppress_context__ '
                      'were not properly set')


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `RedirectImportFinder`, `TestPEP479`

**Functions defined**: `find_spec`, `test_stopiteration_wrapping`, `f`, `g`, `test_stopiteration_wrapping_context`, `f`, `g`

**Key imports**: generator_stop, sys, torch, torch._dynamo.test_case, unittest, CPythonTestCase, run_tests, statements, sys, importlib.abc


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo/cpython/3_13`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: generator_stop
- `sys`
- `torch`
- `torch._dynamo.test_case`
- `unittest`
- `torch.testing._internal.common_utils`: run_tests
- `statements`
- `importlib.abc`
- `is the problematic one`
- `the standalone module`


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
python test/dynamo/cpython/3_13/test_generator_stop.py
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

- **File Documentation**: `test_generator_stop.py_docs.md`
- **Keyword Index**: `test_generator_stop.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
