# Documentation: `docs/test/inductor/test_combo_kernels.py_docs.md`

## File Metadata

- **Path**: `docs/test/inductor/test_combo_kernels.py_docs.md`
- **Size**: 22,752 bytes (22.22 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/inductor/test_combo_kernels.py`

## File Metadata

- **Path**: `test/inductor/test_combo_kernels.py`
- **Size**: 18,935 bytes (18.49 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]

import contextlib
import sys
import unittest

import torch
import torch._inductor
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA_AND_TRITON
from torch.testing._internal.triton_utils import requires_cuda_and_triton


aten = torch.ops.aten

try:
    try:
        from .test_torchinductor import check_model, check_model_cuda
    except ImportError:
        from test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
            check_model,
            check_model_cuda,
        )
except (unittest.SkipTest, ImportError) as e:
    sys.stderr.write(f"{type(e)}: {e}\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise


@instantiate_parametrized_tests
class ComboKernelTests(TestCase):
    check_model_cuda = check_model_cuda
    check_model_cpu = check_model
    check_kernel_count = True

    def setUp(self):
        super().setUp()
        torch._inductor.metrics.reset()
        self._test_stack = contextlib.ExitStack()
        self._test_stack.enter_context(
            torch._inductor.config.patch(
                {
                    "combo_kernels": True,
                    "benchmark_combo_kernel": False,
                }
            )
        )

    def tearDown(self):
        self._test_stack.close()
        torch._inductor.metrics.reset()
        super().tearDown()

    @requires_cuda_and_triton
    def test_activation_functions(self):
        def test_activations(a, b, c):
            a1 = torch.nn.functional.relu(a)
            b1 = torch.nn.functional.sigmoid(b)
            c1 = torch.nn.functional.tanh(c)
            return a1, b1, c1

        inps = [
            torch.rand(10, 10, device="cuda"),
            torch.rand(20, 20, device="cuda"),
            torch.rand(10, 10, device="cuda"),
        ]

        out_eager = test_activations(*inps)
        out_compiled = torch.compile(test_activations)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda_and_triton
    def test_reduce_functions(self):
        def test_reduce(a, b, c, d):
            a1 = torch.sum(a, dim=0)
            b1 = torch.max(b, dim=0)
            c1 = torch.min(c, dim=0)
            d1 = torch.nn.functional.tanh(d)

            return a1, b1, c1, d1

        inps = [
            torch.rand(10, 10, device="cuda"),
            torch.rand(20, 20, device="cuda"),
            torch.rand(10, 10, device="cuda"),
            torch.rand(30, 8, device="cuda"),
        ]

        out_eager = test_reduce(*inps)
        out_compiled = torch.compile(test_reduce)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(torch._inductor.metrics.generated_kernel_count <= 2)

    @requires_cuda_and_triton
    def test_mutated_args(self):
        def test_mutated(a, b, c, d):
            a.add_(1)
            b.sigmoid_()
            c = torch.add(c, 5)
            d.tanh_()

            return a, b, c, d

        inps = [
            torch.rand(10, 10, device="cuda"),
            torch.rand(20, 20, device="cuda"),
            torch.rand(10, 10, device="cuda"),
            torch.rand(30, 8, device="cuda"),
        ]

        out_eager = test_mutated(*inps)
        out_compiled = torch.compile(test_mutated)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @requires_cuda_and_triton
    def test_reduce_split(self):
        def fn(a, b):
            a1 = torch.linalg.vector_norm(a)
            b1 = torch.sum(b, dim=0)
            return a1, b1

        inps = [
            torch.rand(2048, 512, device="cuda"),
            torch.rand(20, 20, device="cuda"),
        ]
        out_eager = fn(*inps)
        out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)

    @requires_cuda_and_triton
    def test_2d_blocking_partitioning(self):
        def fn(a0, a1, a2, b0, b1, b2):
            c0 = torch.add(a0, b0)
            c1 = torch.add(a1, b1)
            c2 = torch.add(a2, b2)
            return c0, c1, c2

        self.check_model_cuda(
            fn,
            (
                torch.rand(30, 20, device="cuda"),
                torch.rand(40, 30, device="cuda"),
                torch.rand(36, 40, device="cuda"),
                torch.rand(30, 20, device="cuda"),
                torch.rand(30, 40, device="cuda").t(),
                torch.rand(40, 36, device="cuda").t(),
            ),
        )

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)


@instantiate_parametrized_tests
class ComboKernelBenchmarkTests(TestCase):
    check_model_cuda = check_model_cuda
    check_model_cpu = check_model
    check_kernel_count = True

    def setUp(self):
        super().setUp()
        torch._inductor.metrics.reset()
        self._test_stack = contextlib.ExitStack()
        self._test_stack.enter_context(
            torch._inductor.config.patch(
                {
                    "combo_kernels": True,
                    "benchmark_combo_kernel": True,
                }
            )
        )

    def tearDown(self):
        self._test_stack.close()
        torch._inductor.metrics.reset()
        super().tearDown()

    @requires_cuda_and_triton
    def test_activation_benchmark(self):
        def test_activations(a, b, c):
            a1 = torch.nn.functional.relu(a)
            b1 = torch.nn.functional.sigmoid(b)
            c1 = torch.nn.functional.tanh(c)
            return a1, b1, c1

        inps = [
            torch.rand(10, 10, device="cuda"),
            torch.rand(20, 20, device="cuda"),
            torch.rand(10, 10, device="cuda"),
        ]

        out_eager = test_activations(*inps)
        out_compiled = torch.compile(test_activations)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 5)

    @requires_cuda_and_triton
    def test_reduce_benchmark(self):
        def test_reduce(a, b, c, d):
            a1 = torch.sum(a, dim=0)
            b1 = torch.max(b, dim=0)
            c1 = torch.min(c, dim=0)
            d1 = torch.nn.functional.tanh(d)

            return a1, b1, c1, d1

        inps = [
            torch.rand(10, 10, device="cuda"),
            torch.rand(20, 20, device="cuda"),
            torch.rand(10, 10, device="cuda"),
            torch.rand(30, 8, device="cuda"),
        ]

        out_eager = test_reduce(*inps)
        out_compiled = torch.compile(test_reduce)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(4 < torch._inductor.metrics.generated_kernel_count <= 10)

    @requires_cuda_and_triton
    def test_mutated_benchmark(self):
        def test_mutated(a, b, c, d):
            a.add_(1)
            b.sigmoid_()
            c = torch.add(c, 5)
            d.tanh_()

            return a, b, c, d

        inps = [
            torch.rand(10, 10, device="cuda"),
            torch.rand(20, 20, device="cuda"),
            torch.rand(10, 10, device="cuda"),
            torch.rand(30, 8, device="cuda"),
        ]

        out_eager = test_mutated(*inps)
        out_compiled = torch.compile(test_mutated)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(torch._inductor.metrics.generated_kernel_count in [6, 9])

    @requires_cuda_and_triton
    def test_round_robin_dispatch(self):
        # combo kernel dispatch strategy: round robin
        def test_mutated(a, b, c, d):
            a.add_(1)
            b.sigmoid_()
            c = torch.add(c, 5)
            d.tanh_()

            return a, b, c, d

        inps = [
            torch.rand(10, 10, device="cuda"),
            torch.rand(20, 5, device="cuda"),
            torch.rand(10, 10, device="cuda"),
            torch.rand(5, 18, device="cuda"),
        ]

        out_eager = test_mutated(*inps)
        out_compiled = torch.compile(test_mutated)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 6)

    @requires_cuda_and_triton
    def test_2d_blocking_benchmark(self):
        def fn(a0, a1, a2, b0, b1, b2):
            c0 = torch.add(a0, b0)
            c1 = torch.add(a1, b1)
            c2 = torch.add(a2, b2)
            return c0, c1, c2

        self.check_model_cuda(
            fn,
            (
                torch.rand(30, 20, device="cuda"),
                torch.rand(40, 30, device="cuda"),
                torch.rand(36, 40, device="cuda"),
                torch.rand(30, 20, device="cuda"),
                torch.rand(30, 40, device="cuda").t(),
                torch.rand(40, 36, device="cuda").t(),
            ),
        )

        self.assertTrue(7 <= torch._inductor.metrics.generated_kernel_count <= 8)

    @requires_cuda_and_triton
    def test_persistent_reduction_no_x_dim(self):
        def fn(x, y):
            return x.sum(1), y.sum(1)

        inps = (
            torch.rand(16, 256, device="cuda"),
            torch.rand(32, 256, device="cuda"),
        )
        torch._dynamo.mark_dynamic(inps[0], 0, min=1, max=256)
        torch._dynamo.mark_dynamic(inps[1], 0, min=1, max=256)
        out_eager = fn(*inps)
        out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 4)


@instantiate_parametrized_tests
class ComboKernelDynamicShapesTests(TestCase):
    check_model_cuda = check_model_cuda
    check_model_cpu = check_model
    check_kernel_count = True

    def setUp(self):
        super().setUp()
        torch._inductor.metrics.reset()
        self._test_stack = contextlib.ExitStack()
        self._test_stack.enter_context(
            torch._inductor.config.patch(
                {
                    "combo_kernels": True,
                    "benchmark_combo_kernel": True,
                }
            )
        )
        self._test_stack.enter_context(
            torch._dynamo.config.patch(
                {
                    "automatic_dynamic_shapes": False,
                    "assume_static_by_default": False,
                }
            )
        )

    def tearDown(self):
        self._test_stack.close()
        torch._inductor.metrics.reset()
        super().tearDown()

    @requires_cuda_and_triton
    def test_dynamic_shapes_activations(self):
        def test_activations(a, b, c):
            a1 = torch.nn.functional.relu(a)
            b1 = torch.nn.functional.sigmoid(b)
            c1 = torch.nn.functional.tanh(c)
            return a1, b1, c1

        inps = [
            torch.rand(10, 10, device="cuda"),
            torch.rand(20, 20, device="cuda"),
            torch.rand(10, 10, device="cuda"),
        ]

        out_eager = test_activations(*inps)
        out_compiled = torch.compile(test_activations)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 5)

    @requires_cuda_and_triton
    def test_dynamic_shapes_2d_blocking(self):
        def fn(a0, a1, a2, b0, b1, b2):
            c0 = torch.add(a0, b0)
            c1 = torch.add(a1, b1)
            c2 = torch.add(a2, b2)
            return c0, c1, c2

        self.check_model_cuda(
            fn,
            (
                torch.rand(30, 20, device="cuda"),
                torch.rand(40, 30, device="cuda"),
                torch.rand(36, 40, device="cuda"),
                torch.rand(30, 20, device="cuda"),
                torch.rand(30, 40, device="cuda").t(),
                torch.rand(40, 36, device="cuda").t(),
            ),
        )

        self.assertTrue(7 <= torch._inductor.metrics.generated_kernel_count <= 8)

    @requires_cuda_and_triton
    def test_dynamic_shapes_reduce(self):
        def test_reduce(a, b, c, d):
            a1 = torch.sum(a, dim=0)
            b1 = torch.max(b, dim=0)
            c1 = torch.min(c, dim=0)
            d1 = torch.nn.functional.tanh(d)

            return a1, b1, c1, d1

        inps = [
            torch.rand(10, 10, device="cuda"),
            torch.rand(20, 20, device="cuda"),
            torch.rand(10, 10, device="cuda"),
            torch.rand(30, 8, device="cuda"),
        ]

        out_eager = test_reduce(*inps)
        out_compiled = torch.compile(test_reduce)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(4 < torch._inductor.metrics.generated_kernel_count <= 10)

    @requires_cuda_and_triton
    def test_dynamic_shapes_mutated(self):
        # combo kernel dispatch strategy: round robin
        def test_mutated(a, b, c, d):
            a.add_(1)
            b.sigmoid_()
            c = torch.add(c, 5)
            d.tanh_()

            return a, b, c, d

        inps = [
            torch.rand(10, 10, device="cuda"),
            torch.rand(20, 5, device="cuda"),
            torch.rand(10, 10, device="cuda"),
            torch.rand(5, 18, device="cuda"),
        ]

        out_eager = test_mutated(*inps)
        out_compiled = torch.compile(test_mutated)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 6)

    @requires_cuda_and_triton
    @torch._inductor.config.patch("combo_kernels_autotune", 0)
    def test_dynamic_shapes_activations_no_autotune(self):
        def test_activations(a, b, c):
            a1 = torch.nn.functional.relu(a)
            b1 = torch.nn.functional.sigmoid(b)
            c1 = torch.nn.functional.tanh(c)
            return a1, b1, c1

        inps = [
            torch.rand(10, 10, device="cuda"),
            torch.rand(20, 20, device="cuda"),
            torch.rand(10, 10, device="cuda"),
        ]

        out_eager = test_activations(*inps)
        out_compiled = torch.compile(test_activations)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 5)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch("automatic_dynamic_shapes", True)
    @torch._dynamo.config.patch("assume_static_by_default", True)
    def test_dynamic_shapes_persistent_reduction_no_x_dim(self):
        def fn(x, y):
            return x.sum(1), y.sum(1)

        inps = (
            torch.rand(16, 256, device="cuda"),
            torch.rand(32, 256, device="cuda"),
        )
        torch._dynamo.mark_dynamic(inps[0], 0, min=1, max=256)
        torch._dynamo.mark_dynamic(inps[1], 0, min=1, max=256)
        out_eager = fn(*inps)
        out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 4)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch("automatic_dynamic_shapes", True)
    @torch._dynamo.config.patch("assume_static_by_default", True)
    def test_dynamic_shapes_persistent_reduction_no_x_dim_2(self):
        def fn(x, y):
            return x.sum(2), y.sum(2)

        inps = (
            torch.rand(8, 16, 256, device="cuda"),
            torch.rand(8, 32, 256, device="cuda"),
        )
        torch._dynamo.mark_dynamic(inps[0], (0, 1), min=1, max=256)
        torch._dynamo.mark_dynamic(inps[1], (0, 1), min=1, max=256)
        out_eager = fn(*inps)
        out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 4)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch("automatic_dynamic_shapes", True)
    @torch._dynamo.config.patch("assume_static_by_default", True)
    def test_dynamic_shapes_2d_blocking_round_robin(self):
        def fn(a0, a1, a2, b0, b1, b2):
            c0 = torch.add(a0, b0)
            c1 = torch.add(a1, b1)
            c2 = torch.add(a2, b2)
            return c0, c1, c2

        inps = (
            torch.rand(20, 30, device="cuda"),
            torch.rand(30, 30, device="cuda"),
            torch.rand(40, 32, device="cuda"),
            torch.rand(30, 20, device="cuda").t(),
            torch.rand(30, 30, device="cuda").t(),
            torch.rand(32, 40, device="cuda").t(),
        )

        out_eager = fn(*inps)
        compiled = torch.compile(fn)
        out_compiled = compiled(*inps)
        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(5 <= torch._inductor.metrics.generated_kernel_count <= 6)
        torch._inductor.metrics.reset()

        inps = (
            torch.rand(24, 30, device="cuda"),
            torch.rand(32, 30, device="cuda"),
            torch.rand(48, 32, device="cuda"),
            torch.rand(30, 24, device="cuda").t(),
            torch.rand(30, 32, device="cuda").t(),
            torch.rand(32, 48, device="cuda").t(),
        )
        out_compiled = compiled(*inps)
        out_eager = fn(*inps)
        self.assertEqual(out_eager, out_compiled)
        self.assertTrue(5 <= torch._inductor.metrics.generated_kernel_count <= 6)

    @requires_cuda_and_triton
    @torch._dynamo.config.patch("automatic_dynamic_shapes", True)
    @torch._dynamo.config.patch("assume_static_by_default", True)
    @torch._inductor.config.patch("triton.autotune_at_compile_time", True)
    def test_dynamic_shapes_persistent_reduction_mixed_x_dim_cuda(self):
        def fn(x, y, z):
            return x.sum(1), y.mean(1), z.max(1)

        inps = (
            torch.rand(16, 128, device="cuda"),
            torch.rand(32, 128, device="cuda"),
            torch.rand(32, 256, device="cuda"),
        )
        torch._dynamo.mark_dynamic(inps[0], 0, min=1, max=256)
        torch._dynamo.mark_dynamic(inps[1], 0, min=1, max=256)
        torch._dynamo.mark_dynamic(inps[2], 0, min=1, max=256)
        out_eager = fn(*inps)
        out_compiled = torch.compile(fn)(*inps)

        self.assertEqual(out_eager, out_compiled)

    @requires_cuda_and_triton
    def test_helper_fn_defined(self):
        def fn(x, y, z):
            return x.sum(1), y.mean(1), z.cumsum(1)

        inps = (
            torch.rand(16, 128, device="cuda"),
            torch.rand(32, 128, device="cuda"),
            torch.rand(32, 256, device="cuda"),
        )

        out_eager = fn(*inps)
        fn_c = torch.compile(fn)
        out_compiled, code = run_and_get_code(fn_c, *inps)
        code = " ".join(code)
        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(code.count("def _triton_helper_fn_add0(arg0_0, arg1_0):"), 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if HAS_CPU or HAS_CUDA_AND_TRITON:
        run_tests(needs="filelock")

```



## High-Level Overview


This Python file contains 3 class(es) and 49 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ComboKernelTests`, `ComboKernelBenchmarkTests`, `ComboKernelDynamicShapesTests`

**Functions defined**: `setUp`, `tearDown`, `test_activation_functions`, `test_activations`, `test_reduce_functions`, `test_reduce`, `test_mutated_args`, `test_mutated`, `test_reduce_split`, `fn`, `test_2d_blocking_partitioning`, `fn`, `setUp`, `tearDown`, `test_activation_benchmark`, `test_activations`, `test_reduce_benchmark`, `test_reduce`, `test_mutated_benchmark`, `test_mutated`

**Key imports**: contextlib, sys, unittest, torch, torch._inductor, run_and_get_code, HAS_CPU, HAS_CUDA_AND_TRITON, requires_cuda_and_triton, check_model, check_model_cuda, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `sys`
- `unittest`
- `torch`
- `torch._inductor`
- `torch._inductor.utils`: run_and_get_code
- `torch.testing._internal.inductor_utils`: HAS_CPU, HAS_CUDA_AND_TRITON
- `torch.testing._internal.triton_utils`: requires_cuda_and_triton
- `.test_torchinductor`: check_model, check_model_cuda
- `torch._dynamo.test_case`: run_tests


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
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
python test/inductor/test_combo_kernels.py
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

- **File Documentation**: `test_combo_kernels.py_docs.md`
- **Keyword Index**: `test_combo_kernels.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/inductor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
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
python docs/test/inductor/test_combo_kernels.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/inductor`):

- [`test_snode_runtime.py_kw.md_docs.md`](./test_snode_runtime.py_kw.md_docs.md)
- [`test_metrics.py_docs.md_docs.md`](./test_metrics.py_docs.md_docs.md)
- [`test_flex_attention.py_kw.md_docs.md`](./test_flex_attention.py_kw.md_docs.md)
- [`test_cuda_repro.py_kw.md_docs.md`](./test_cuda_repro.py_kw.md_docs.md)
- [`test_fxir_backend.py_kw.md_docs.md`](./test_fxir_backend.py_kw.md_docs.md)
- [`test_split_cat_fx_passes.py_kw.md_docs.md`](./test_split_cat_fx_passes.py_kw.md_docs.md)
- [`test_mmdecomp.py_kw.md_docs.md`](./test_mmdecomp.py_kw.md_docs.md)
- [`test_torchinductor_codegen_config_overrides.py_kw.md_docs.md`](./test_torchinductor_codegen_config_overrides.py_kw.md_docs.md)
- [`test_aot_inductor_custom_ops.py_kw.md_docs.md`](./test_aot_inductor_custom_ops.py_kw.md_docs.md)
- [`test_minifier.py_kw.md_docs.md`](./test_minifier.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_combo_kernels.py_docs.md_docs.md`
- **Keyword Index**: `test_combo_kernels.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
