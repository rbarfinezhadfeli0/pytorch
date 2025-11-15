# Documentation: `docs/test/inductor/test_compile_subprocess.py_docs.md`

## File Metadata

- **Path**: `docs/test/inductor/test_compile_subprocess.py_docs.md`
- **Size**: 12,810 bytes (12.51 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/inductor/test_compile_subprocess.py`

## File Metadata

- **Path**: `test/inductor/test_compile_subprocess.py`
- **Size**: 9,227 bytes (9.01 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: fx"]

#
# Tests compiling the inductor tests in a subprocess.
#

import contextlib
import importlib
import os
import sys
import time
import unittest
from unittest import mock
from unittest.mock import patch

import torch
import torch.library
from torch._inductor.compile_fx import _InProcessFxCompile, FxCompile, FxCompileMode
from torch._inductor.graph import GraphLowering
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import IS_CI, IS_WINDOWS, TEST_WITH_ASAN
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    IS_BIG_GPU,
    requires_gpu,
    requires_triton,
    RUN_CPU,
    RUN_GPU,
)


if IS_WINDOWS and IS_CI:
    # TODO(xuhancn) : Debug and confirm pass_fds status on Windows.
    sys.stderr.write(
        "Almost UTs failed: pass_fds not supported on Windows, skip them on Windows.\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("pass_fds not supported on Windows")


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
import inductor.test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
from inductor.test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    check_model,
    check_model_gpu,
    copy_tests,
    TestFailure,
)


importlib.import_module("filelock")

# xfail by default, set is_skip=True to skip
test_failures = {
    # TypeError: cannot pickle 'generator' object
    "test_layer_norm": TestFailure(("cpu", "cuda"), is_skip=True),
    "test_remove_noop_slice": TestFailure(("xpu"), is_skip=True),
    "test_remove_noop_slice1": TestFailure(("xpu"), is_skip=True),
    "test_remove_noop_slice_scatter": TestFailure(("xpu"), is_skip=True),
    "test_remove_noop_view_default": TestFailure(("xpu"), is_skip=True),
    "test_remove_noop_view_dtype": TestFailure(("xpu"), is_skip=True),
    # can not pickle ParametrizedConv2d
    "test_weight_norm_conv2d": TestFailure(("cpu", "cuda"), is_skip=True),
}


class TestSubprocess(TestCase):
    def setUp(self):
        torch._dynamo.reset()
        FxCompile._reset_stats()

        super().setUp()

        self._stack = contextlib.ExitStack()
        self._stack.enter_context(
            patch(
                "torch._inductor.compile_fx.fx_compile_mode",
                FxCompileMode.SUBPROCESS,
            )
        )

    def tearDown(self):
        # Check that the test didn't instigate an in-process compile - which
        # would mean that something about the fx graph failed to serialize. If
        # some tests are expected to fail then we should probably add a list of
        # expected failures here.
        self.assertEqual(
            FxCompile._compile_stats[type(_InProcessFxCompile)].codegen_and_compile, 0
        )
        self._stack.close()
        TestCase.tearDown(self)
        torch._dynamo.reset()

    @requires_gpu()
    @requires_triton()
    @unittest.skipIf(
        not IS_BIG_GPU, "Skipping triton backend only since not big GPU (not enough SM)"
    )
    def test_progressive(self):
        from triton.testing import do_bench

        from torch._inductor.compile_fx_async import _ProgressiveFxCompile

        torch._inductor.compile_fx.fx_compile_progressive = True

        x = torch.randn(1152, 1024, device=GPU_TYPE, dtype=torch.bfloat16)
        y = torch.randn(1024, 1024, device=GPU_TYPE, dtype=torch.bfloat16)

        @torch.compile(fullgraph=True, backend="inductor")
        def optimized(x, y):
            return (x @ y).relu()

        _ProgressiveFxCompile._reset_stats()

        source_codes: list[str] = []

        def save_output_code(code: str) -> None:
            source_codes.append(code)

        with contextlib.ExitStack() as stack:
            # When this bug is fixed, remove the cache disabling below
            assert torch._inductor.compile_fx_async.BUG_CACHES_DONT_WORK_WITH_ASYNC
            stack.enter_context(
                torch._inductor.config.patch(
                    autotune_local_cache=False, fx_graph_cache=False
                )
            )
            stack.enter_context(
                mock.patch.object(GraphLowering, "save_output_code", save_output_code)
            )
            stack.enter_context(
                torch._functorch.config.patch(enable_autograd_cache=False)
            )

            # How long to wait (in seconds) before giving up.
            TIMEOUT = 300
            # If non-None then how often (in seconds) to print a TICK message.
            TICK_REPORT = None

            start = time.time()
            last_report = start
            while _ProgressiveFxCompile._stat_optimized_runs < 4:
                time.sleep(0.25)

                optimized(x, y)

                now = time.time()
                if TICK_REPORT is not None and (now - last_report > TICK_REPORT):
                    print(f"*** TICK {int(now - start)}")
                    last_report = now

                if now - start > TIMEOUT:
                    raise RuntimeError(
                        "Test timed out before producing a progressively optimized compiled artifact."
                    )

            self.assertEqual(_ProgressiveFxCompile._stat_optimized_runs, 4)
            self.assertGreater(_ProgressiveFxCompile._stat_fast_runs, 0)
            self.assertGreaterEqual(_ProgressiveFxCompile._stat_bg_started, 1)
            self.assertGreaterEqual(_ProgressiveFxCompile._stat_bg_finished, 1)

        torch._inductor.compile_fx.fx_compile_progressive = False

        @torch.compile(fullgraph=True, backend="inductor")
        def baseline(x, y):
            return (x @ y).relu()

        # Warmup
        baseline(x, y)

        self.assertGreater(
            do_bench(lambda: baseline(x, y)), do_bench(lambda: optimized(x, y))
        )
        self.assertTrue("'max_autotune': True" in source_codes[-1])

    @patch("torch._inductor.compile_fx.fx_compile_async", True)
    def test_async(self):
        # Test that async+subprocess works.
        from torch._inductor.compile_fx_async import _AsyncFxCompile

        @torch.compile(fullgraph=True, backend="inductor")
        def model_add(x, y):
            out = x
            for _ in range(500):
                out = torch.add(out, y)
            return out

        _AsyncFxCompile._reset_stats()

        with contextlib.ExitStack() as stack:
            assert torch._inductor.compile_fx_async.BUG_CACHES_DONT_WORK_WITH_ASYNC
            stack.enter_context(
                torch._inductor.config.patch(
                    autotune_local_cache=False, fx_graph_cache=False
                )
            )
            stack.enter_context(
                torch._functorch.config.patch(enable_autograd_cache=False)
            )

            # How long to wait (in seconds) before giving up.
            TIMEOUT = 300
            # If non-None then how often (in seconds) to print a TICK message.
            TICK_REPORT = None

            start = time.time()
            last_report = start
            while True:
                start_stat_compiled_runs = _AsyncFxCompile._stat_compiled_runs
                # Sleep a bit so we don't drive the CPU unnecessarily.
                time.sleep(0.25)

                x = torch.randn(100, 100, requires_grad=True)
                y = torch.randn(100, 100, requires_grad=True)

                # Forward pass
                output = model_add(x, y)

                # Backward pass
                output.sum().backward()

                if _AsyncFxCompile._stat_compiled_runs - start_stat_compiled_runs == 2:
                    break

                # DEBUGGING: Print a periodic message so we know we're still
                # running...
                now = time.time()
                if TICK_REPORT is not None and (now - last_report > TICK_REPORT):
                    print(f"*** TICK {int(now - start)}")
                    last_report = now

                if now - start > TIMEOUT:
                    raise RuntimeError(
                        "Test timed out before producing a compiled artifact."
                    )

            self.assertGreater(_AsyncFxCompile._stat_compiled_runs, 1)
            # Make sure we ran eager at least once. Normally this will be
            # something like 80.
            self.assertGreater(_AsyncFxCompile._stat_eager_runs, 0)
            self.assertEqual(_AsyncFxCompile._stat_bg_started, 2)
            self.assertEqual(_AsyncFxCompile._stat_bg_finished, 2)


if RUN_CPU:

    class CpuTests(TestSubprocess):
        common = check_model
        device = "cpu"

    copy_tests(
        inductor.test_torchinductor.CommonTemplate, CpuTests, "cpu", test_failures
    )

if RUN_GPU and not TEST_WITH_ASAN:

    class GPUTests(TestSubprocess):
        common = check_model_gpu
        device = GPU_TYPE

    copy_tests(
        inductor.test_torchinductor.CommonTemplate, GPUTests, GPU_TYPE, test_failures
    )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if RUN_CPU or RUN_GPU:
        run_tests(needs="filelock")

```



## High-Level Overview


This Python file contains 3 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestSubprocess`, `CpuTests`, `GPUTests`

**Functions defined**: `setUp`, `tearDown`, `test_progressive`, `optimized`, `save_output_code`, `baseline`, `test_async`, `model_add`

**Key imports**: contextlib, importlib, os, sys, time, unittest, mock, patch, torch, torch.library


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `importlib`
- `os`
- `sys`
- `time`
- `unittest`
- `unittest.mock`: patch
- `torch`
- `torch.library`
- `torch._inductor.compile_fx`: _InProcessFxCompile, FxCompile, FxCompileMode
- `torch._inductor.graph`: GraphLowering
- `torch._inductor.test_case`: TestCase
- `torch.testing._internal.common_utils`: IS_CI, IS_WINDOWS, TEST_WITH_ASAN
- `inductor.test_torchinductor  `
- `triton.testing`: do_bench
- `torch._inductor.compile_fx_async`: _ProgressiveFxCompile


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data
- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/inductor/test_compile_subprocess.py
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

- **File Documentation**: `test_compile_subprocess.py_docs.md`
- **Keyword Index**: `test_compile_subprocess.py_kw.md`
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

- **Serialization**: Uses pickle - be cautious with untrusted data
- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/inductor/test_compile_subprocess.py_docs.md
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

- **File Documentation**: `test_compile_subprocess.py_docs.md_docs.md`
- **Keyword Index**: `test_compile_subprocess.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
