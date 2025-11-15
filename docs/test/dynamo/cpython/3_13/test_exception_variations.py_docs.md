# Documentation: `test/dynamo/cpython/3_13/test_exception_variations.py`

## File Metadata

- **Path**: `test/dynamo/cpython/3_13/test_exception_variations.py`
- **Size**: 8,776 bytes (8.57 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_exception_variations.py

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

import unittest

class ExceptTestCases(__TestCase):
    def test_try_except_else_finally(self):
        hit_except = False
        hit_else = False
        hit_finally = False

        try:
            raise Exception('nyaa!')
        except:
            hit_except = True
        else:
            hit_else = True
        finally:
            hit_finally = True

        self.assertTrue(hit_except)
        self.assertTrue(hit_finally)
        self.assertFalse(hit_else)

    def test_try_except_else_finally_no_exception(self):
        hit_except = False
        hit_else = False
        hit_finally = False

        try:
            pass
        except:
            hit_except = True
        else:
            hit_else = True
        finally:
            hit_finally = True

        self.assertFalse(hit_except)
        self.assertTrue(hit_finally)
        self.assertTrue(hit_else)

    def test_try_except_finally(self):
        hit_except = False
        hit_finally = False

        try:
            raise Exception('yarr!')
        except:
            hit_except = True
        finally:
            hit_finally = True

        self.assertTrue(hit_except)
        self.assertTrue(hit_finally)

    def test_try_except_finally_no_exception(self):
        hit_except = False
        hit_finally = False

        try:
            pass
        except:
            hit_except = True
        finally:
            hit_finally = True

        self.assertFalse(hit_except)
        self.assertTrue(hit_finally)

    def test_try_except(self):
        hit_except = False

        try:
            raise Exception('ahoy!')
        except:
            hit_except = True

        self.assertTrue(hit_except)

    def test_try_except_no_exception(self):
        hit_except = False

        try:
            pass
        except:
            hit_except = True

        self.assertFalse(hit_except)

    def test_try_except_else(self):
        hit_except = False
        hit_else = False

        try:
            raise Exception('foo!')
        except:
            hit_except = True
        else:
            hit_else = True

        self.assertFalse(hit_else)
        self.assertTrue(hit_except)

    def test_try_except_else_no_exception(self):
        hit_except = False
        hit_else = False

        try:
            pass
        except:
            hit_except = True
        else:
            hit_else = True

        self.assertFalse(hit_except)
        self.assertTrue(hit_else)

    def test_try_finally_no_exception(self):
        hit_finally = False

        try:
            pass
        finally:
            hit_finally = True

        self.assertTrue(hit_finally)

    def test_nested(self):
        hit_finally = False
        hit_inner_except = False
        hit_inner_finally = False

        try:
            try:
                raise Exception('inner exception')
            except:
                hit_inner_except = True
            finally:
                hit_inner_finally = True
        finally:
            hit_finally = True

        self.assertTrue(hit_inner_except)
        self.assertTrue(hit_inner_finally)
        self.assertTrue(hit_finally)

    def test_nested_else(self):
        hit_else = False
        hit_finally = False
        hit_except = False
        hit_inner_except = False
        hit_inner_else = False

        try:
            try:
                pass
            except:
                hit_inner_except = True
            else:
                hit_inner_else = True

            raise Exception('outer exception')
        except:
            hit_except = True
        else:
            hit_else = True
        finally:
            hit_finally = True

        self.assertFalse(hit_inner_except)
        self.assertTrue(hit_inner_else)
        self.assertFalse(hit_else)
        self.assertTrue(hit_finally)
        self.assertTrue(hit_except)

    def test_nested_exception_in_except(self):
        hit_else = False
        hit_finally = False
        hit_except = False
        hit_inner_except = False
        hit_inner_else = False

        try:
            try:
                raise Exception('inner exception')
            except:
                hit_inner_except = True
                raise Exception('outer exception')
            else:
                hit_inner_else = True
        except:
            hit_except = True
        else:
            hit_else = True
        finally:
            hit_finally = True

        self.assertTrue(hit_inner_except)
        self.assertFalse(hit_inner_else)
        self.assertFalse(hit_else)
        self.assertTrue(hit_finally)
        self.assertTrue(hit_except)

    def test_nested_exception_in_else(self):
        hit_else = False
        hit_finally = False
        hit_except = False
        hit_inner_except = False
        hit_inner_else = False

        try:
            try:
                pass
            except:
                hit_inner_except = True
            else:
                hit_inner_else = True
                raise Exception('outer exception')
        except:
            hit_except = True
        else:
            hit_else = True
        finally:
            hit_finally = True

        self.assertFalse(hit_inner_except)
        self.assertTrue(hit_inner_else)
        self.assertFalse(hit_else)
        self.assertTrue(hit_finally)
        self.assertTrue(hit_except)

    def test_nested_exception_in_finally_no_exception(self):
        hit_else = False
        hit_finally = False
        hit_except = False
        hit_inner_except = False
        hit_inner_else = False
        hit_inner_finally = False

        try:
            try:
                pass
            except:
                hit_inner_except = True
            else:
                hit_inner_else = True
            finally:
                hit_inner_finally = True
                raise Exception('outer exception')
        except:
            hit_except = True
        else:
            hit_else = True
        finally:
            hit_finally = True

        self.assertFalse(hit_inner_except)
        self.assertTrue(hit_inner_else)
        self.assertTrue(hit_inner_finally)
        self.assertFalse(hit_else)
        self.assertTrue(hit_finally)
        self.assertTrue(hit_except)

    def test_nested_exception_in_finally_with_exception(self):
        hit_else = False
        hit_finally = False
        hit_except = False
        hit_inner_except = False
        hit_inner_else = False
        hit_inner_finally = False

        try:
            try:
                raise Exception('inner exception')
            except:
                hit_inner_except = True
            else:
                hit_inner_else = True
            finally:
                hit_inner_finally = True
                raise Exception('outer exception')
        except:
            hit_except = True
        else:
            hit_else = True
        finally:
            hit_finally = True


        self.assertTrue(hit_inner_except)
        self.assertFalse(hit_inner_else)
        self.assertTrue(hit_inner_finally)
        self.assertFalse(hit_else)
        self.assertTrue(hit_finally)
        self.assertTrue(hit_except)


if __name__ == '__main__':
    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 16 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `RedirectImportFinder`, `ExceptTestCases`

**Functions defined**: `find_spec`, `test_try_except_else_finally`, `test_try_except_else_finally_no_exception`, `test_try_except_finally`, `test_try_except_finally_no_exception`, `test_try_except`, `test_try_except_no_exception`, `test_try_except_else`, `test_try_except_else_no_exception`, `test_try_finally_no_exception`, `test_nested`, `test_nested_else`, `test_nested_exception_in_except`, `test_nested_exception_in_else`, `test_nested_exception_in_finally_no_exception`, `test_nested_exception_in_finally_with_exception`

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
python test/dynamo/cpython/3_13/test_exception_variations.py
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

- **File Documentation**: `test_exception_variations.py_docs.md`
- **Keyword Index**: `test_exception_variations.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
