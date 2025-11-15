# Documentation: `docs/test/torch_np/conftest.py_docs.md`

## File Metadata

- **Path**: `docs/test/torch_np/conftest.py_docs.md`
- **Size**: 5,419 bytes (5.29 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/torch_np/conftest.py`

## File Metadata

- **Path**: `test/torch_np/conftest.py`
- **Size**: 2,109 bytes (2.06 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# Owner(s): ["module: dynamo"]

import sys

import pytest

import torch._numpy as tnp


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: very slow tests")


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run slow tests")
    parser.addoption("--nonp", action="store_true", help="error when NumPy is accessed")


class Inaccessible:
    def __getattribute__(self, attr):
        raise RuntimeError(f"Using --nonp but accessed np.{attr}")


def pytest_sessionstart(session):
    if session.config.getoption("--nonp"):
        sys.modules["numpy"] = Inaccessible()


def pytest_generate_tests(metafunc):
    """
    Hook to parametrize test cases
    See https://docs.pytest.org/en/6.2.x/parametrize.html#pytest-generate-tests

    The logic here allows us to test with both NumPy-proper and torch._numpy.
    Normally we'd just test torch._numpy, e.g.

        import torch._numpy as np
        ...
        def test_foo():
            np.array([42])
            ...

    but this hook allows us to test NumPy-proper as well, e.g.

        def test_foo(np):
            np.array([42])
            ...

    np is a pytest parameter, which is either NumPy-proper or torch._numpy. This
    allows us to sanity check our own tests, so that tested behaviour is
    consistent with NumPy-proper.

    pytest will have test names respective to the library being tested, e.g.

        $ pytest --collect-only
        test_foo[torch._numpy]
        test_foo[numpy]

    """
    np_params = [tnp]

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        if not isinstance(np, Inaccessible):  # i.e. --nonp was used
            np_params.append(np)

    if "np" in metafunc.fixturenames:
        metafunc.parametrize("np", np_params)


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="slow test, use --runslow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

```



## High-Level Overview

"""    Hook to parametrize test cases    See https://docs.pytest.org/en/6.2.x/parametrize.html#pytest-generate-tests    The logic here allows us to test with both NumPy-proper and torch._numpy.    Normally we'd just test torch._numpy, e.g.        import torch._numpy as np        ...        def test_foo():            np.array([42])            ...    but this hook allows us to test NumPy-proper as well, e.g.        def test_foo(np):            np.array([42])            ...    np is a pytest parameter, which is either NumPy-proper or torch._numpy. This    allows us to sanity check our own tests, so that tested behaviour is

This Python file contains 1 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Inaccessible`

**Functions defined**: `pytest_configure`, `pytest_addoption`, `__getattribute__`, `pytest_sessionstart`, `pytest_generate_tests`, `test_foo`, `test_foo`, `pytest_collection_modifyitems`

**Key imports**: sys, pytest, torch._numpy as tnp, torch._numpy as np, numpy as np


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/torch_np`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `pytest`
- `torch._numpy as tnp`
- `torch._numpy as np`
- `numpy as np`


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
python test/torch_np/conftest.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/torch_np`):

- [`test_random.py_docs.md`](./test_random.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_ufuncs_basic.py_docs.md`](./test_ufuncs_basic.py_docs.md)
- [`test_function_base.py_docs.md`](./test_function_base.py_docs.md)
- [`test_basic.py_docs.md`](./test_basic.py_docs.md)
- [`test_binary_ufuncs.py_docs.md`](./test_binary_ufuncs.py_docs.md)
- [`test_indexing.py_docs.md`](./test_indexing.py_docs.md)
- [`test_ndarray_methods.py_docs.md`](./test_ndarray_methods.py_docs.md)
- [`test_reductions.py_docs.md`](./test_reductions.py_docs.md)


## Cross-References

- **File Documentation**: `conftest.py_docs.md`
- **Keyword Index**: `conftest.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/torch_np`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/torch_np`, which is part of the **core PyTorch library**.



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
python docs/test/torch_np/conftest.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/torch_np`):

- [`test_binary_ufuncs.py_docs.md_docs.md`](./test_binary_ufuncs.py_docs.md_docs.md)
- [`test_scalars_0D_arrays.py_docs.md_docs.md`](./test_scalars_0D_arrays.py_docs.md_docs.md)
- [`test_ndarray_methods.py_docs.md_docs.md`](./test_ndarray_methods.py_docs.md_docs.md)
- [`test_scalars_0D_arrays.py_kw.md_docs.md`](./test_scalars_0D_arrays.py_kw.md_docs.md)
- [`test_function_base.py_docs.md_docs.md`](./test_function_base.py_docs.md_docs.md)
- [`test_basic.py_docs.md_docs.md`](./test_basic.py_docs.md_docs.md)
- [`test_function_base.py_kw.md_docs.md`](./test_function_base.py_kw.md_docs.md)
- [`check_tests_conform.py_kw.md_docs.md`](./check_tests_conform.py_kw.md_docs.md)
- [`test_ufuncs_basic.py_kw.md_docs.md`](./test_ufuncs_basic.py_kw.md_docs.md)
- [`test_reductions.py_docs.md_docs.md`](./test_reductions.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `conftest.py_docs.md_docs.md`
- **Keyword Index**: `conftest.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
