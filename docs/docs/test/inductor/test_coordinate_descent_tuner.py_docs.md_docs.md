# Documentation: `docs/test/inductor/test_coordinate_descent_tuner.py_docs.md`

## File Metadata

- **Path**: `docs/test/inductor/test_coordinate_descent_tuner.py_docs.md`
- **Size**: 7,697 bytes (7.52 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/inductor/test_coordinate_descent_tuner.py`

## File Metadata

- **Path**: `test/inductor/test_coordinate_descent_tuner.py`
- **Size**: 4,084 bytes (3.99 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]

import sys
import unittest
from unittest import mock

import torch
from torch._inductor.runtime.hints import TRITON_MAX_BLOCK
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


try:
    import triton  # @manual
except ImportError:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires triton")  # noqa: B904

from torch._inductor import config
from torch._inductor.runtime.coordinate_descent_tuner import CoordescTuner


config.benchmark_kernel = True
config.coordinate_descent_tuning = True

orig_compare_config = CoordescTuner.compare_config


def mock_compare_config_prefer_larger_XBLOCK(
    self, func, candidate_config, best_config, best_timing
):
    """
    self is the CoordescTuner object
    """
    if "XBLOCK" in candidate_config.kwargs:
        assert "XBLOCK" in best_config.kwargs
        if candidate_config.kwargs["XBLOCK"] < best_config.kwargs["XBLOCK"]:
            func(candidate_config)  # run func so the launcher will be created
            return False, best_timing * 1.1
        elif candidate_config.kwargs["XBLOCK"] > best_config.kwargs["XBLOCK"]:
            func(candidate_config)
            return True, best_timing * 0.9

    return orig_compare_config(self, func, candidate_config, best_config, best_timing)


class TestCoordinateDescentTuner(TestCase):
    def test_abs_function(self):
        """
        The benchmark result is simply abs(XBLOCK - 15)
        """
        tuner = CoordescTuner()
        baseline_config = triton.Config({"XBLOCK": 1}, num_warps=8, num_stages=1)

        def func(config):
            return abs(config.kwargs["XBLOCK"] - 15)

        best_config = tuner.autotune(func, baseline_config)
        self.assertTrue(best_config.kwargs.get("XBLOCK") == 16, str(best_config))

    def test_no_neighbors(self):
        """
        Test the case that there is no available neighbor values for a field.
        """

        # size hint for x being 1 limits the max XBLOCK we try to be 1
        tuner = CoordescTuner(size_hints={"x": 1})
        baseline_config = triton.Config({"XBLOCK": 1}, num_warps=8, num_stages=1)

        def func(config):
            return abs(config.kwargs["XBLOCK"] - 15)

        best_config = tuner.autotune(func, baseline_config)
        self.assertTrue(best_config.kwargs.get("XBLOCK") == 1, str(best_config))

    def test_get_neighbour_values(self):
        tuner = CoordescTuner()

        neighbours = tuner.get_neighbour_values("num_stages", 2, radius=2)
        self.assertEqual(set(neighbours), {1, 3, 4})
        neighbours = tuner.get_neighbour_values("num_warps", 2, radius=2)
        self.assertEqual(set(neighbours), {1, 4, 8})

    def test_persistent_reduction(self):
        def f(x):
            return x / x.sum(dim=-1, keepdim=True)

        with mock.patch.object(
            CoordescTuner, "compare_config", mock_compare_config_prefer_larger_XBLOCK
        ):
            x = torch.ones(2, 256).to(GPU_TYPE)
            expected = f(x)
            # the first call get correct result when cache miss. Don't know why yet
            _ = torch.compile(f)(x)
            actual = torch.compile(f)(x)
            self.assertTrue(
                torch.allclose(expected, actual, atol=1e-4, rtol=1e-4),
                f"Expected:\n{expected}\nActual:\n{actual}",
            )

    def test_value_too_large(self):
        # Simulate a reduction
        size_hints = {"x": 2**20, "y": 2**20}

        tuner = CoordescTuner(size_hints=size_hints)

        max_block = TRITON_MAX_BLOCK
        self.assertFalse(tuner.value_too_large("XBLOCK", max_block["X"]))
        self.assertTrue(tuner.value_too_large("XBLOCK", max_block["X"] * 2))
        self.assertFalse(tuner.value_too_large("R0_BLOCK", max_block["R0_"]))
        self.assertTrue(tuner.value_too_large("R0_BLOCK", max_block["R0_"] * 2))


if __name__ == "__main__":
    if IS_LINUX and HAS_GPU:
        run_tests()

```



## High-Level Overview

"""    self is the CoordescTuner object

This Python file contains 1 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestCoordinateDescentTuner`

**Functions defined**: `mock_compare_config_prefer_larger_XBLOCK`, `test_abs_function`, `func`, `test_no_neighbors`, `func`, `test_get_neighbour_values`, `test_persistent_reduction`, `f`, `test_value_too_large`

**Key imports**: sys, unittest, mock, torch, TRITON_MAX_BLOCK, run_tests, TestCase, IS_LINUX, GPU_TYPE, HAS_GPU, triton  , config


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `unittest`
- `torch`
- `torch._inductor.runtime.hints`: TRITON_MAX_BLOCK
- `torch._inductor.test_case`: run_tests, TestCase
- `torch.testing._internal.common_utils`: IS_LINUX
- `torch.testing._internal.inductor_utils`: GPU_TYPE, HAS_GPU
- `triton  `
- `torch._inductor`: config
- `torch._inductor.runtime.coordinate_descent_tuner`: CoordescTuner


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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
python test/inductor/test_coordinate_descent_tuner.py
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

- **File Documentation**: `test_coordinate_descent_tuner.py_docs.md`
- **Keyword Index**: `test_coordinate_descent_tuner.py_kw.md`
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
python docs/test/inductor/test_coordinate_descent_tuner.py_docs.md
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

- **File Documentation**: `test_coordinate_descent_tuner.py_docs.md_docs.md`
- **Keyword Index**: `test_coordinate_descent_tuner.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
