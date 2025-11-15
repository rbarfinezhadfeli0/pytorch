# Documentation: `test/inductor/test_move_constructors_to_cuda.py`

## File Metadata

- **Path**: `test/inductor/test_move_constructors_to_cuda.py`
- **Size**: 3,306 bytes (3.23 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]

import functools
import unittest

import torch
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON


requires_multigpu = functools.partial(
    unittest.skipIf, not TEST_MULTIGPU, "requires multiple cuda devices"
)

aten = torch.ops.aten


class TestMoveConstructorsToCuda(TestCase):
    def _check_fn(self, func, expect_cpu, *args):
        out_eager = func(*args)

        out_compiled, code = run_and_get_code(torch.compile(func), *args)
        self.assertEqual(out_eager, out_compiled)

        assert len(code) == 1
        if expect_cpu:
            FileCheck().check("cpp_fused").run(code[0])
        else:
            FileCheck().check_not("cpp_fused").run(code[0])

    def test_simple(self):
        def foo(x):
            return x[torch.arange(x.shape[0])]

        inp = torch.rand(32, 77, 512, device="cuda")

        self._check_fn(foo, False, inp)

    def test_output_failure(self):
        def foo(x):
            tmp1 = torch.arange(x.shape[0])
            return tmp1, x[tmp1]

        inp = torch.rand(32, 77, 512, device="cuda")

        self._check_fn(foo, True, inp)

    def test_non_convertable_op_failure(self):
        def foo(x):
            y = torch.arange(x.shape[0])
            return x + y, torch.ones([4], device="cuda")

        inp = torch.rand([100])

        self._check_fn(foo, True, inp)

    def test_multiple_constructors(self):
        def foo(x):
            tmp1 = torch.arange(x.shape[0])
            o1 = x[tmp1]
            tmp2 = torch.arange(x.shape[1]).view([1, x.shape[1]])
            o2 = x[tmp2]
            return o1, o2, o1 + o2

        inp = torch.rand([200, 200])
        self._check_fn(foo, True, inp)

    def test_sets_equiv(self):
        @torch.compile()
        def foo(x):
            c1 = torch.ones([4], dtype=torch.long)
            c2 = torch.arange(-1, 3)
            return x[c1 + c2], c2 - 4 * 2

        inp = torch.rand([4]).cuda()
        _, code = run_and_get_code(foo, inp)
        FileCheck().check_not("triton.jit").run(code[0])

        @torch.compile()
        def foo(x):
            c2 = torch.arange(-1, 3)
            c1 = torch.ones([4], dtype=torch.long)
            return x[c1 + c2], c2 - 4 * 2

        _, code = run_and_get_code(foo, inp)
        FileCheck().check_not("triton.jit").run(code[0])

    @requires_multigpu()
    @unittest.skip("https://github.com/pytorch/pytorch/issues/139520")
    def test_multi_gpu(self):
        def foo(x):
            return (
                x[torch.arange(x.shape[0])],
                torch.ones([4], device="cuda:0"),
                torch.ones([4], device="cuda:1"),
            )

        # nyi, multi-gpu
        inp = torch.rand([100], device="cuda")
        self._check_fn(foo, True, inp)

    def test_no_gpu(self):
        def foo(x):
            return x[torch.arange(x.shape[0])]

        inp = torch.rand([100])
        self._check_fn(foo, True, inp)


if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA_AND_TRITON:
        run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 16 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestMoveConstructorsToCuda`

**Functions defined**: `_check_fn`, `test_simple`, `foo`, `test_output_failure`, `foo`, `test_non_convertable_op_failure`, `foo`, `test_multiple_constructors`, `foo`, `test_sets_equiv`, `foo`, `foo`, `test_multi_gpu`, `foo`, `test_no_gpu`, `foo`

**Key imports**: functools, unittest, torch, run_tests, TestCase, run_and_get_code, FileCheck, TEST_MULTIGPU, IS_LINUX, HAS_CUDA_AND_TRITON


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `unittest`
- `torch`
- `torch._inductor.test_case`: run_tests, TestCase
- `torch._inductor.utils`: run_and_get_code
- `torch.testing`: FileCheck
- `torch.testing._internal.common_cuda`: TEST_MULTIGPU
- `torch.testing._internal.common_utils`: IS_LINUX
- `torch.testing._internal.inductor_utils`: HAS_CUDA_AND_TRITON


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
python test/inductor/test_move_constructors_to_cuda.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/inductor`):

- [`test_benchmark_fusion.py_docs.md`](./test_benchmark_fusion.py_docs.md)
- [`test_op_dtype_prop.py_docs.md`](./test_op_dtype_prop.py_docs.md)
- [`test_custom_op_autotune.py_docs.md`](./test_custom_op_autotune.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_inductor_freezing.py_docs.md`](./test_inductor_freezing.py_docs.md)
- [`test_b2b_gemm.py_docs.md`](./test_b2b_gemm.py_docs.md)
- [`test_minifier_isolate.py_docs.md`](./test_minifier_isolate.py_docs.md)
- [`test_cutlass_backend.py_docs.md`](./test_cutlass_backend.py_docs.md)
- [`test_cache.py_docs.md`](./test_cache.py_docs.md)


## Cross-References

- **File Documentation**: `test_move_constructors_to_cuda.py_docs.md`
- **Keyword Index**: `test_move_constructors_to_cuda.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
