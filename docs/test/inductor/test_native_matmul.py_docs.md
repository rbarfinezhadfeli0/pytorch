# Documentation: `test/inductor/test_native_matmul.py`

## File Metadata

- **Path**: `test/inductor/test_native_matmul.py`
- **Size**: 4,796 bytes (4.68 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]


from collections.abc import Callable

import torch
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import same
from torch._inductor import config as inductor_config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_triton_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import skipIfXpu
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


aten = torch.ops.aten


@inductor_config.patch({"triton.native_matmul": True})
class TestTritonDotReduction(TestCase):
    def _check_equal(
        self,
        f: Callable,
        example_inputs: tuple[torch.Tensor],
    ):
        compiled = torch.compile(f)
        actual = compiled(*example_inputs)
        expect = f(*example_inputs)
        self.assertTrue(same(expect, actual))

    def _check_code(
        self,
        f: Callable,
        example_inputs: tuple[torch.Tensor],
        kernel_count: int,
        dot_count: int,
    ):
        f = torch.compile(f)
        code = run_and_get_triton_code(f, *example_inputs)
        FileCheck().check_regex(r"triton.*mm.*\.run\(").run(code)

        FileCheck().check_count(
            "@triton.jit",
            kernel_count,
        ).check_count(
            "tl.dot",
            dot_count,
        ).run(code)

    def test_matmul(self):
        def f(x, y):
            z = x @ y
            return z

        M, K, N = 128, 128, 128
        x = rand_strided((M, K), (K, 1), device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y))
        self._check_code(f, (x, y), 1, 1)

    def test_mm_1d_expand(self):
        def f(x, y, M, K):
            z = x[:, None].expand(M, K) @ y
            return z

        M, K, N = 128, 128, 128
        x = rand_strided((M,), (1,), device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y, M, K))
        self._check_code(f, (x, y, M, K), 1, 1)

    def test_mm_2_expand(self):
        def f(x, y, M, K):
            z = x[:, None].expand(M, K) @ y
            return z

        M, K, N = 128, 128, 128
        x = rand_strided((1,), (0,), device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y, M, K))
        self._check_code(f, (x, y, M, K), 1, 1)

    def test_matmul_fp16(self):
        def f(x, y):
            z = x @ y.to(x.dtype)
            return z

        M, K, N = 128, 128, 128
        x = rand_strided((M, K), (K, 1), dtype=torch.float16, device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), dtype=torch.float32, device=GPU_TYPE)

        self._check_equal(f, (x, y))
        self._check_code(f, (x, y), 1, 1)

    def test_reduction_mask_zeroout(self):
        def f(x, y):
            return (x + 1) @ (y - 2)

        M, K, N = 62, 62, 62
        x = rand_strided((M, K), (K, 1), device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y))
        self._check_code(f, (x, y), 1, 1)

    @skipIfXpu(
        msg="Intel triton issue: https://github.com/intel/intel-xpu-backend-for-triton/issues/5394"
    )
    def test_3mm_add(self):
        def f(x, y, z, w, r, t):
            return x @ y + z @ w + r @ t

        M, K, N = 128, 128, 128
        x = rand_strided((M, K), (K, 1), device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), device=GPU_TYPE)
        w = rand_strided((M, K), (K, 1), device=GPU_TYPE)
        z = rand_strided((K, N), (N, 1), device=GPU_TYPE)
        r = rand_strided((M, K), (K, 1), device=GPU_TYPE)
        t = rand_strided((K, N), (N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y, z, w, r, t))
        self._check_code(f, (x, y, z, w, r, t), 1, 3)

    def test_mm_complex(self):
        def f(x, y, z, w):
            return x[z] @ y + w + 3

        M, K, N = 128, 128, 128
        x = rand_strided((M, K), (K, 1), device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), device=GPU_TYPE)

        z = torch.randint(M, (M, K), dtype=torch.long, device=GPU_TYPE)
        w = rand_strided((M, N), (N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y, z, w))
        self._check_code(f, (x, y, z, w), 1, 1)

    def test_batchmatmul(self):
        def f(x, y):
            z = torch.bmm(x, y)
            return z

        B, M, K, N = 256, 128, 128, 128
        x = rand_strided((B, M, K), (M * K, K, 1), device=GPU_TYPE)
        y = rand_strided((B, K, N), (K * N, N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y))
        self._check_code(f, (x, y), 1, 1)


if HAS_GPU:
    torch.set_default_device(GPU_TYPE)

if __name__ == "__main__":
    if HAS_GPU:
        run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 18 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestTritonDotReduction`

**Functions defined**: `_check_equal`, `_check_code`, `test_matmul`, `f`, `test_mm_1d_expand`, `f`, `test_mm_2_expand`, `f`, `test_matmul_fp16`, `f`, `test_reduction_mask_zeroout`, `f`, `test_3mm_add`, `f`, `test_mm_complex`, `f`, `test_batchmatmul`, `f`

**Key imports**: Callable, torch, rand_strided, same, config as inductor_config, run_tests, TestCase, run_and_get_triton_code, FileCheck, skipIfXpu, GPU_TYPE, HAS_GPU


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Callable
- `torch`
- `torch._dynamo.testing`: rand_strided
- `torch._dynamo.utils`: same
- `torch._inductor`: config as inductor_config
- `torch._inductor.test_case`: run_tests, TestCase
- `torch._inductor.utils`: run_and_get_triton_code
- `torch.testing`: FileCheck
- `torch.testing._internal.common_utils`: skipIfXpu
- `torch.testing._internal.inductor_utils`: GPU_TYPE, HAS_GPU


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
python test/inductor/test_native_matmul.py
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
- [`test_move_constructors_to_cuda.py_docs.md`](./test_move_constructors_to_cuda.py_docs.md)
- [`test_cutlass_backend.py_docs.md`](./test_cutlass_backend.py_docs.md)
- [`test_cache.py_docs.md`](./test_cache.py_docs.md)


## Cross-References

- **File Documentation**: `test_native_matmul.py_docs.md`
- **Keyword Index**: `test_native_matmul.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
