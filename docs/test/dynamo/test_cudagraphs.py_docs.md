# Documentation: `test/dynamo/test_cudagraphs.py`

## File Metadata

- **Path**: `test/dynamo/test_cudagraphs.py`
- **Size**: 5,736 bytes (5.60 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: cuda graphs"]

import functools
import unittest

import torch
import torch._dynamo
import torch._dynamo.config
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import same
from torch.testing._internal.common_utils import TEST_CUDA_GRAPH


def composed(*decs):
    def deco(f):
        for dec in reversed(decs):
            f = dec(f)
        return f

    return deco


def assert_aot_autograd_counter(ok=True):
    def deco(f):
        @functools.wraps(f)
        def wrap(self, *args, **kwargs):
            torch._dynamo.utils.counters.clear()
            r = f(self, *args, **kwargs)
            c_ok = torch._dynamo.utils.counters["aot_autograd"]["ok"]
            c_not_ok = torch._dynamo.utils.counters["aot_autograd"]["not_ok"]
            if ok:
                self.assertGreater(c_ok, 0)
                self.assertEqual(c_not_ok, 0)
            else:
                self.assertEqual(c_ok, 0)
                self.assertGreater(c_not_ok, 0)
            return r

        return wrap

    return deco


def patch_all(ok=True):
    return composed(
        torch._dynamo.config.patch(
            verify_correctness=True, automatic_dynamic_shapes=True
        ),
        assert_aot_autograd_counter(ok),
    )


N_ITERS = 5


@unittest.skipIf(not torch.cuda.is_available(), "these tests require cuda")
class TestAotCudagraphs(torch._dynamo.test_case.TestCase):
    @patch_all()
    def test_basic(self):
        def model(x, y):
            return (x + y) * y

        @torch.compile(backend="cudagraphs")
        def fn(x, y):
            for _ in range(N_ITERS):
                loss = model(x, y).sum()
                loss.backward()

        x = torch.randn(3, device="cuda", requires_grad=True)
        y = torch.randn(3, device="cuda")
        fn(x, y)

    @patch_all()
    def test_dtoh(self):
        def model(x, y):
            a = x + y
            b = a.cpu() * 3
            return b

        @torch.compile(backend="cudagraphs")
        def fn(x, y):
            for _ in range(N_ITERS):
                loss = model(x, y).sum()
                loss.backward()

        x = torch.randn(3, device="cuda", requires_grad=True)
        y = torch.randn(3, device="cuda")
        fn(x, y)

    @patch_all()
    def test_htod(self):
        def model(x, y):
            a = x + y
            return a * 3

        @torch.compile(backend="cudagraphs")
        def fn(x, y):
            for _ in range(N_ITERS):
                loss = model(x, y).sum()
                loss.backward()

        x = torch.randn(3, device="cuda", requires_grad=True)
        y = torch.randn((), device="cpu")
        fn(x, y)

    def test_mutate_input(self):
        def model(x, y):
            y.add_(3)
            return x * y

        @torch.compile(backend="cudagraphs")
        def fn(x, y):
            for i in range(N_ITERS):
                with self.subTest(i):
                    y_orig = y.clone()
                    loss = model(x, y).sum()
                    self.assertTrue(same(y, y_orig + 3))
                    loss.backward()

        x = torch.randn(3, device="cuda", requires_grad=True)
        y = torch.randn(3, device="cuda")
        fn(x, y)

    @patch_all()
    def test_mutate_constant(self):
        def model(x, y):
            c = torch.tensor(1)
            c.add_(2)
            return x * y * 0 + c

        @torch.compile(backend="cudagraphs")
        def fn(x, y):
            for i in range(N_ITERS):
                with self.subTest(i):
                    loss = model(x, y).sum()
                    self.assertTrue(same(loss, torch.tensor(3.0, device="cuda")))
                    loss.backward()

        x = torch.randn(1, device="cuda", requires_grad=True)
        y = torch.randn(1, device="cuda")
        fn(x, y)

    @patch_all()
    def test_factory(self):
        def model(y):
            x = torch.zeros(3, device="cuda:0")
            x.add_(3)
            return x * y

        @torch.compile(backend="cudagraphs")
        def fn(y):
            for i in range(N_ITERS):
                with self.subTest(i):
                    loss = model(y).sum()
                    loss.backward()

        y = torch.randn(3, device="cuda:0", requires_grad=True)
        fn(y)

    @patch_all()
    def test_mutated_metadata(self):
        # more tortured example at
        # https://github.com/pytorch/pytorch/issues/81385
        def model(x):
            x = x.clone()
            x.resize_(20)
            x.fill_(2)
            return x

        @torch.compile(backend="cudagraphs")
        def fn(x):
            for i in range(N_ITERS):
                with self.subTest(i):
                    rx = model(x)
                    self.assertTrue(same(rx, torch.full((20,), 2.0, device="cuda:0")))

        x = torch.empty(0, device="cuda:0")
        fn(x)

    @patch_all()
    def test_dead_fill(self):
        def model(x):
            x = x.clone()
            y = x[0:0]
            x.fill_(2)
            y.fill_(3)
            return x, y

        @torch.compile(backend="cudagraphs")
        def fn(x):
            for i in range(N_ITERS):
                with self.subTest(i):
                    rx, ry = model(x)
                    self.assertTrue(same(rx, torch.full((20,), 2.0, device="cuda:0")))
                    self.assertTrue(same(ry, torch.empty(0, device="cuda:0")))

        x = torch.empty(20, device="cuda:0")
        fn(x)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if not TEST_CUDA_GRAPH:
        if __name__ == "__main__":
            import sys

            sys.exit(0)
        raise unittest.SkipTest("cuda graph test is skipped")

    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 30 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestAotCudagraphs`

**Functions defined**: `composed`, `deco`, `assert_aot_autograd_counter`, `deco`, `wrap`, `patch_all`, `test_basic`, `model`, `fn`, `test_dtoh`, `model`, `fn`, `test_htod`, `model`, `fn`, `test_mutate_input`, `model`, `fn`, `test_mutate_constant`, `model`

**Key imports**: functools, unittest, torch, torch._dynamo, torch._dynamo.config, torch._dynamo.test_case, torch._dynamo.testing, same, TEST_CUDA_GRAPH, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `unittest`
- `torch`
- `torch._dynamo`
- `torch._dynamo.config`
- `torch._dynamo.test_case`
- `torch._dynamo.testing`
- `torch.testing._internal.common_utils`: TEST_CUDA_GRAPH
- `sys`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/dynamo/test_cudagraphs.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/dynamo`):

- [`test_guard_serialization.py_docs.md`](./test_guard_serialization.py_docs.md)
- [`test_subgraphs.py_docs.md`](./test_subgraphs.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_unspec.py_docs.md`](./test_unspec.py_docs.md)
- [`test_trace_rules.py_docs.md`](./test_trace_rules.py_docs.md)
- [`test_package.py_docs.md`](./test_package.py_docs.md)
- [`test_pre_dispatch.py_docs.md`](./test_pre_dispatch.py_docs.md)
- [`test_autograd_function.py_docs.md`](./test_autograd_function.py_docs.md)
- [`test_optimizers.py_docs.md`](./test_optimizers.py_docs.md)
- [`test_callback.py_docs.md`](./test_callback.py_docs.md)


## Cross-References

- **File Documentation**: `test_cudagraphs.py_docs.md`
- **Keyword Index**: `test_cudagraphs.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
