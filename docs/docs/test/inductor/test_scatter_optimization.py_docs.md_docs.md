# Documentation: `docs/test/inductor/test_scatter_optimization.py_docs.md`

## File Metadata

- **Path**: `docs/test/inductor/test_scatter_optimization.py_docs.md`
- **Size**: 10,591 bytes (10.34 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/inductor/test_scatter_optimization.py`

## File Metadata

- **Path**: `test/inductor/test_scatter_optimization.py`
- **Size**: 7,086 bytes (6.92 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]

import copy
import os
import unittest

import torch
from torch import nn
from torch._dynamo.utils import counters, same
from torch._inductor import metrics
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.test_case import TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


# set so that metrics appear
torch._logging.set_logs(inductor_metrics=True)

DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"


class TestScatterOpt(TestCase):
    def setUp(self):
        super().setUp()
        metrics.reset()
        counters.clear()

    def check_metric(self, val=1):
        self.assertEqual(val, metrics.num_matches_for_scatter_upon_const_tensor)

    def do_acc_test(self, f, *args):
        expect = f(*args)
        actual = torch.compile(f)(*args)
        self.assertTrue(same(expect, actual, tol=1e-3), f"{expect=}\n{actual=}\n")

    def test_3d_tensor(self):
        L, M, N = 2, 1024, 2048

        def f(x):
            y = torch.full([L, M, N], 3.14, dtype=torch.float)
            y.scatter_(2, x.unsqueeze(2), 2.718)
            return y

        x = torch.randint(0, N, (L, M), dtype=torch.int64)
        self.do_acc_test(f, x)
        expected_num_bytes = (
            L * M * N * torch.float.itemsize + L * M * torch.int64.itemsize
        )
        self.assertEqual(metrics.num_bytes_accessed, expected_num_bytes)

    def test_non_last_dim(self):
        """
        Test the case that the scatter dimension is not the last one.
        """
        M, N = 1024, 2048

        def f(x):
            y = torch.full([M, N], 3.14, dtype=torch.float)
            y.scatter_(0, x.unsqueeze(0), 2.718)
            return y

        x = torch.randint(0, M, (N,), dtype=torch.int64)
        self.do_acc_test(f, x)
        expected_num_bytes = M * N * torch.float.itemsize + N * torch.int64.itemsize
        self.assertEqual(metrics.num_bytes_accessed, expected_num_bytes)

    def test_neg_scatter_dim(self):
        M, N = 1024, 2048

        def f(x):
            y = torch.full([M, N], 3.14, dtype=torch.float)
            y.scatter_(-1, x.unsqueeze(1), 2.718)
            return y

        x = torch.randint(0, N, (M,), dtype=torch.int64)
        self.do_acc_test(f, x)
        expected_num_bytes = M * N * torch.float.itemsize + M * torch.int64.itemsize
        self.assertEqual(metrics.num_bytes_accessed, expected_num_bytes)

    def test_shorter_index_tensor(self):
        M, N = 1024, 2048

        def f(x):
            y = torch.full([M, N], 3.14, dtype=torch.float)
            y.scatter_(1, x.unsqueeze(1), 2.718)
            return y

        x = torch.randint(0, N, (M // 2,), dtype=torch.int64)
        self.do_acc_test(f, x)

        # no match since the index tensor is shorter. May support it in future.
        self.assertEqual(0, counters["inductor"]["pattern_matcher_count"])

    def test_nonzero_const_tensor(self):
        M, N = 1024, 2048

        def f(x):
            y = torch.full([M, N], 3.14, dtype=torch.float)
            y.scatter_(1, x.unsqueeze(1), 2.718)
            return y

        x = torch.randint(0, N, (M,), dtype=torch.int64)
        self.do_acc_test(f, x)
        expected_num_bytes = M * N * torch.float.itemsize + M * torch.int64.itemsize
        self.assertEqual(metrics.num_bytes_accessed, expected_num_bytes)

    def test_can_not_optimize_due_to_dense(self):
        M, N = 1024, 2048

        def f(x):
            y = torch.full([M, N], 0, dtype=torch.float)
            y.scatter_(1, x, 0.618)
            return y

        x = torch.randint(0, N, (M, N // 2), dtype=torch.int64)
        self.do_acc_test(f, x)
        expected_num_bytes = M * N * torch.float.itemsize + M * (N // 2) * (
            torch.int64.itemsize + torch.float.itemsize
        )
        # Use assertGreaterEqual rather than assertEqual due to the issue related
        # to StarDep mentioned here: https://github.com/pytorch/pytorch/pull/129043#discussion_r1651699706
        self.assertGreaterEqual(metrics.num_bytes_accessed, expected_num_bytes)

    def test_can_not_optimize_due_to_non_const(self):
        M, N = 1024, 2048

        def f(x, y):
            y.scatter_(1, x, 0.618)
            return y

        x = torch.randint(0, N, (M, 1), dtype=torch.int64)
        y = torch.randn([M, N])
        self.do_acc_test(f, x, y)

        # The generated code is quite in-efficient.
        # There are 3 kernels
        # 1. copy from arg to buf
        # 2. scatter upon buf
        # 3. copy buf back to arg
        # Link to the wrapper: https://gist.github.com/shunting314/d43b74e680b3e5b514f7c28160c39f40
        expected_num_bytes = 4 * M * N * torch.float.itemsize + M * (
            torch.int64.itemsize + torch.float.itemsize
        )
        self.assertGreaterEqual(metrics.num_bytes_accessed, expected_num_bytes)

        # the second kernel and third kernel are both mutation kernel. So we
        # overestimated the memory accessed
        # Update the test once the overestimiation is fixed.
        over_estimate = M * torch.float.itemsize + M * N * torch.float.itemsize
        self.assertEqual(metrics.num_bytes_accessed, expected_num_bytes + over_estimate)

    def test_cross_entropy_loss(self):
        """
        Match full+scatter in CEL and replaces it with a pointwise.

        Perf data on an A100 GPU:
        Without the scatter optimization:
          ms=47.340, peak_mem=10.524 GB
        With the scatter optimization:
          ms=42.768, peak_mem=7.227 GB
        """
        B, T, D, V = 32, 1024, 768, 50257
        if not DO_PERF_TEST:
            # use a smaller V if not doing perf test to avoid OOM
            # in CI
            V = V // 100
        ref_model = nn.Linear(D, V).to(torch.bfloat16)
        opt_model = copy.deepcopy(ref_model)
        ce = nn.CrossEntropyLoss()

        def f(m, x, label):
            ce(m(x).view(-1, V), label.view(-1)).backward()

        opt_f = torch.compile(f)

        x = torch.randn(B, T, D).to(torch.bfloat16)
        label = torch.randint(0, V, (B, T)).to(torch.int64)

        f(ref_model, x, label)
        ref_grad = ref_model.weight.grad
        opt_f(opt_model, x, label)
        act_grad = opt_model.weight.grad
        assert torch.allclose(ref_grad, act_grad, atol=1e-3, rtol=1e-3), (
            f"{ref_grad=}\n{act_grad=}"
        )

        self.check_metric()

        if DO_PERF_TEST:
            if GPU_TYPE == "xpu":
                raise unittest.SkipTest(
                    "torch.xpu.reset_peak_memory_stats not implemented."
                )
            torch.cuda.reset_peak_memory_stats()
            for _ in range(3):
                opt_f(opt_model, x, label)
            ms = benchmarker.benchmark_gpu(lambda: opt_f(opt_model, x, label))
            peak_mem = torch.cuda.max_memory_allocated() / 10**9
            print(f"{ms=:.3f}, {peak_mem=:.3f} GB")


if HAS_GPU:
    torch.set_default_device(GPU_TYPE)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 19 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestScatterOpt`

**Functions defined**: `setUp`, `check_metric`, `do_acc_test`, `test_3d_tensor`, `f`, `test_non_last_dim`, `f`, `test_neg_scatter_dim`, `f`, `test_shorter_index_tensor`, `f`, `test_nonzero_const_tensor`, `f`, `test_can_not_optimize_due_to_dense`, `f`, `test_can_not_optimize_due_to_non_const`, `f`, `test_cross_entropy_loss`, `f`

**Key imports**: copy, os, unittest, torch, nn, counters, same, metrics, benchmarker, TestCase, GPU_TYPE, HAS_GPU


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `os`
- `unittest`
- `torch`
- `torch._dynamo.utils`: counters, same
- `torch._inductor`: metrics
- `torch._inductor.runtime.benchmarking`: benchmarker
- `torch._inductor.test_case`: TestCase
- `torch.testing._internal.inductor_utils`: GPU_TYPE, HAS_GPU


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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
python test/inductor/test_scatter_optimization.py
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

- **File Documentation**: `test_scatter_optimization.py_docs.md`
- **Keyword Index**: `test_scatter_optimization.py_kw.md`
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

*No specific patterns automatically detected.*


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
python docs/test/inductor/test_scatter_optimization.py_docs.md
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

- **File Documentation**: `test_scatter_optimization.py_docs.md_docs.md`
- **Keyword Index**: `test_scatter_optimization.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
