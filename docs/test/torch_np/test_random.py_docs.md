# Documentation: `test/torch_np/test_random.py`

## File Metadata

- **Path**: `test/torch_np/test_random.py`
- **Size**: 4,267 bytes (4.17 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]

"""Light smoke test switching between numpy to pytorch random streams."""

from contextlib import contextmanager
from functools import partial

import numpy as _np
import pytest

import torch._dynamo.config as config
import torch._numpy as tnp
from torch._numpy.testing import assert_equal
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)


@contextmanager
def control_stream(use_numpy=False):
    with config.patch(use_numpy_random_stream=use_numpy):
        yield


@instantiate_parametrized_tests
class TestScalarReturn(TestCase):
    @parametrize("use_numpy", [True, False])
    @parametrize(
        "func",
        [
            tnp.random.normal,
            tnp.random.rand,
            partial(tnp.random.randint, 0, 5),
            tnp.random.randn,
            subtest(tnp.random.random, name="random_random"),
            subtest(tnp.random.random_sample, name="random_sample"),
            tnp.random.sample,
            tnp.random.uniform,
        ],
    )
    def test_rndm_scalar(self, func, use_numpy):
        # default `size` means a python scalar return
        with control_stream(use_numpy):
            r = func()
        assert isinstance(r, (int, float))

    @parametrize("use_numpy", [True, False])
    @parametrize(
        "func",
        [
            tnp.random.normal,
            tnp.random.rand,
            partial(tnp.random.randint, 0, 5),
            tnp.random.randn,
            subtest(tnp.random.random, name="random_random"),
            subtest(tnp.random.random_sample, name="random_sample"),
            tnp.random.sample,
            tnp.random.uniform,
        ],
    )
    def test_rndm_array(self, func, use_numpy):
        with control_stream(use_numpy):
            if func in (tnp.random.rand, tnp.random.randn):
                r = func(10)
            else:
                r = func(size=10)
        assert isinstance(r, tnp.ndarray)


@instantiate_parametrized_tests
class TestShuffle(TestCase):
    @parametrize("use_numpy", [True, False])
    def test_1d(self, use_numpy):
        ax = tnp.asarray([1, 2, 3, 4, 5, 6])
        ox = ax.copy()

        tnp.random.seed(1234)
        tnp.random.shuffle(ax)

        assert isinstance(ax, tnp.ndarray)
        assert not (ax == ox).all()

    @parametrize("use_numpy", [True, False])
    def test_2d(self, use_numpy):
        # np.shuffle only shuffles the first axis
        ax = tnp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        ox = ax.copy()

        tnp.random.seed(1234)
        tnp.random.shuffle(ax)

        assert isinstance(ax, tnp.ndarray)
        assert not (ax == ox).all()

    @parametrize("use_numpy", [True, False])
    def test_shuffle_list(self, use_numpy):
        # on eager, we refuse to shuffle lists
        # under dynamo, we always fall back to numpy
        # NB: this means that the random stream is different for
        # shuffling a list or an array when USE_NUMPY_STREAM == False
        x = [1, 2, 3]
        with pytest.raises(NotImplementedError):
            tnp.random.shuffle(x)


@instantiate_parametrized_tests
class TestChoice(TestCase):
    @parametrize("use_numpy", [True, False])
    def test_choice(self, use_numpy):
        kwds = dict(size=3, replace=False, p=[0.1, 0, 0.3, 0.6, 0])
        with control_stream(use_numpy):
            tnp.random.seed(12345)
            x = tnp.random.choice(5, **kwds)
            tnp.random.seed(12345)
            x_1 = tnp.random.choice(tnp.arange(5), **kwds)
            assert_equal(x, x_1)


class TestNumpyGlobal(TestCase):
    def test_numpy_global(self):
        with control_stream(use_numpy=True):
            tnp.random.seed(12345)
            x = tnp.random.uniform(0, 1, size=11)

        # check that the stream is identical to numpy's
        _np.random.seed(12345)
        x_np = _np.random.uniform(0, 1, size=11)
        assert_equal(x, tnp.asarray(x_np))

        # switch to the pytorch stream, variates differ
        with control_stream(use_numpy=False):
            tnp.random.seed(12345)
            x_1 = tnp.random.uniform(0, 1, size=11)

        assert not (x_1 == x).all()


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""Light smoke test switching between numpy to pytorch random streams."""from contextlib import contextmanagerfrom functools import partialimport numpy as _npimport pytestimport torch._dynamo.config as configimport torch._numpy as tnpfrom torch._numpy.testing import assert_equalfrom torch.testing._internal.common_utils import (    instantiate_parametrized_tests,    parametrize,    run_tests,    subtest,    TestCase,)@contextmanagerdef control_stream(use_numpy=False):    with config.patch(use_numpy_random_stream=use_numpy):        yield@instantiate_parametrized_testsclass TestScalarReturn(TestCase):    @parametrize("use_numpy", [True, False])    @parametrize(        "func",        [            tnp.random.normal,            tnp.random.rand,            partial(tnp.random.randint, 0, 5),            tnp.random.randn,            subtest(tnp.random.random, name="random_random"),            subtest(tnp.random.random_sample, name="random_sample"),            tnp.random.sample,            tnp.random.uniform,        ],    )    def test_rndm_scalar(self, func, use_numpy):        # default `size` means a python scalar return        with control_stream(use_numpy):            r = func()        assert isinstance(r, (int, float))

This Python file contains 4 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestScalarReturn`, `TestShuffle`, `TestChoice`, `TestNumpyGlobal`

**Functions defined**: `control_stream`, `test_rndm_scalar`, `test_rndm_array`, `test_1d`, `test_2d`, `test_shuffle_list`, `test_choice`, `test_numpy_global`

**Key imports**: contextmanager, partial, numpy as _np, pytest, torch._dynamo.config as config, torch._numpy as tnp, assert_equal


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/torch_np`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`: contextmanager
- `functools`: partial
- `numpy as _np`
- `pytest`
- `torch._dynamo.config as config`
- `torch._numpy as tnp`
- `torch._numpy.testing`: assert_equal


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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
python test/torch_np/test_random.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/torch_np`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_ufuncs_basic.py_docs.md`](./test_ufuncs_basic.py_docs.md)
- [`test_function_base.py_docs.md`](./test_function_base.py_docs.md)
- [`test_basic.py_docs.md`](./test_basic.py_docs.md)
- [`test_binary_ufuncs.py_docs.md`](./test_binary_ufuncs.py_docs.md)
- [`test_indexing.py_docs.md`](./test_indexing.py_docs.md)
- [`test_ndarray_methods.py_docs.md`](./test_ndarray_methods.py_docs.md)
- [`conftest.py_docs.md`](./conftest.py_docs.md)
- [`test_reductions.py_docs.md`](./test_reductions.py_docs.md)


## Cross-References

- **File Documentation**: `test_random.py_docs.md`
- **Keyword Index**: `test_random.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
