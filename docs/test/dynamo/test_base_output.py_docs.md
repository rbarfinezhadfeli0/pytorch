# Documentation: `test/dynamo/test_base_output.py`

## File Metadata

- **Path**: `test/dynamo/test_base_output.py`
- **Size**: 2,363 bytes (2.31 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]
import unittest.mock

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import same


try:
    from diffusers.models import unet_2d
except ImportError:
    unet_2d = None


def maybe_skip(fn):
    if unet_2d is None:
        return unittest.skip("requires diffusers")(fn)
    return fn


class TestBaseOutput(torch._dynamo.test_case.TestCase):
    @maybe_skip
    def test_create(self):
        def fn(a):
            tmp = unet_2d.UNet2DOutput(a + 1)
            return tmp

        torch._dynamo.testing.standard_test(self, fn=fn, nargs=1, expected_ops=1)

    @maybe_skip
    def test_assign(self):
        def fn(a):
            tmp = unet_2d.UNet2DOutput(a + 1)
            tmp.sample = a + 2
            return tmp

        args = [torch.randn(10)]
        obj1 = fn(*args)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        obj2 = opt_fn(*args)
        self.assertTrue(same(obj1.sample, obj2.sample))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def _common(self, fn, op_count):
        args = [
            unet_2d.UNet2DOutput(
                sample=torch.randn(10),
            )
        ]
        obj1 = fn(*args)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        obj2 = opt_fn(*args)
        self.assertTrue(same(obj1, obj2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, op_count)

    @maybe_skip
    def test_getattr(self):
        def fn(obj: unet_2d.UNet2DOutput):
            x = obj.sample * 10
            return x

        self._common(fn, 1)

    @maybe_skip
    def test_getitem(self):
        def fn(obj: unet_2d.UNet2DOutput):
            x = obj["sample"] * 10
            return x

        self._common(fn, 1)

    @maybe_skip
    def test_tuple(self):
        def fn(obj: unet_2d.UNet2DOutput):
            a = obj.to_tuple()
            return a[0] * 10

        self._common(fn, 1)

    @maybe_skip
    def test_index(self):
        def fn(obj: unet_2d.UNet2DOutput):
            return obj[0] * 10

        self._common(fn, 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 14 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestBaseOutput`

**Functions defined**: `maybe_skip`, `test_create`, `fn`, `test_assign`, `fn`, `_common`, `test_getattr`, `fn`, `test_getitem`, `fn`, `test_tuple`, `fn`, `test_index`, `fn`

**Key imports**: unittest.mock, torch, torch._dynamo.test_case, torch._dynamo.testing, same, unet_2d, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest.mock`
- `torch`
- `torch._dynamo.test_case`
- `torch._dynamo.testing`
- `diffusers.models`: unet_2d


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

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
python test/dynamo/test_base_output.py
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

- **File Documentation**: `test_base_output.py_docs.md`
- **Keyword Index**: `test_base_output.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
