# Documentation: `docs/test/inductor/test_cuda_select_algorithm.py_docs.md`

## File Metadata

- **Path**: `docs/test/inductor/test_cuda_select_algorithm.py_docs.md`
- **Size**: 11,848 bytes (11.57 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/inductor/test_cuda_select_algorithm.py`

## File Metadata

- **Path**: `test/inductor/test_cuda_select_algorithm.py`
- **Size**: 8,062 bytes (7.87 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]
import functools
import sys
import unittest
from unittest.mock import patch

import torch
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config
import torch._inductor.select_algorithm as select_algorithm
from torch._dynamo.utils import counters
from torch._inductor.test_case import run_tests
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_quantized import (
    _calculate_dynamic_per_channel_qparams,
)
from torch.testing._internal.common_utils import (
    parametrize,
    TEST_CUDA,
    TEST_WITH_SLOW_GRADCHECK,
)


try:
    try:
        from . import test_cpu_select_algorithm, test_torchinductor
    except ImportError:
        import test_cpu_select_algorithm  # @manual=fbcode//caffe2/test/inductor:test_cpu_select_algorithm-library
        import test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise

check_model = test_torchinductor.check_model
BaseTestSelectAlgorithm = test_cpu_select_algorithm.BaseTestSelectAlgorithm


def patches(fn):
    def skip_cache(self, choices, name, key, benchmark, hint_override=None):
        if benchmark is None:
            return {}
        timings = benchmark(choices)
        for choice, timing in timings.items():
            if isinstance(choice, select_algorithm.ExternKernelCaller):
                timings[choice] = timing * 1000
        return timings

    for patcher in [
        dynamo_config.patch(verbose=True),
        dynamo_config.patch(inline_inbuilt_nn_modules=True),
        inductor_config.patch(
            debug=True,
            max_autotune=True,
            epilogue_fusion=True,
        ),
        patch.object(select_algorithm, "VERIFY", dict(atol=1e-4, rtol=1e-4)),
        patch.object(select_algorithm.AlgorithmSelectorCache, "lookup", skip_cache),
    ]:
        fn = patcher(fn)

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        counters.clear()
        torch.manual_seed(12345)
        return fn(*args, **kwargs)

    return wrapped


class TestSelectAlgorithmCuda(BaseTestSelectAlgorithm):
    common = check_model

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @dtypes(torch.bfloat16)
    @parametrize("batch_size", (1, 17, 32))
    @parametrize("mid_dim", (1, 8))
    @parametrize("in_features", (128, 144, 1024))
    @parametrize("out_features", (64, 65, 1024))
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @unittest.skipIf(TEST_WITH_SLOW_GRADCHECK, "Leaking memory")
    def test_int8_woq_mm_cuda(
        self, dtype, batch_size, mid_dim, in_features, out_features
    ):
        def _convert_weight_to_int8pack(w):
            # Move to CPU for quantization calculation, then back to original device
            device = w.device
            w_cpu = w.cpu()
            scale, zp = _calculate_dynamic_per_channel_qparams(
                w_cpu.to(torch.float), torch.int8
            )
            scale = torch.from_numpy(scale).to(device)
            zp = torch.from_numpy(zp).to(device)
            w_int8 = torch.ao.quantization.fx._decomposed.quantize_per_channel(
                input=w,
                scales=scale,
                zero_points=zp,
                axis=0,
                quant_min=-128,
                quant_max=127,
                dtype=torch.int8,
            )
            return w_int8, scale.to(torch.bfloat16)

        class M(torch.nn.Module):
            def __init__(self, w):
                super().__init__()
                self.linear_weight = torch.nn.Parameter(w, requires_grad=False)

            def forward(self, x, scale):
                return (
                    torch.nn.functional.linear(x, self.linear_weight.to(x.dtype))
                    * scale
                )

        counters.clear()
        # Currently, the corresponding torch.fx pattern only supports 3D x
        # Add 2D X case once the corresponding pattern-matcher pattern is added
        x = torch.rand((batch_size, mid_dim, in_features), dtype=dtype, device="cuda")
        w = torch.rand((out_features, in_features), dtype=dtype, device="cuda")
        w_int8pack, w_scales = _convert_weight_to_int8pack(w)
        w_scales = w_scales.to("cuda")
        mod = M(w_int8pack).eval()
        self.common(mod, (x, w_scales))
        self.assertEqual(counters["inductor"]["woq_matcher_count"], 1)

    @inductor_config.patch({"freezing": True, "cpp.enable_concat_linear": True})
    @patches
    @torch.no_grad
    @dtypes(torch.bfloat16)
    @parametrize("batch_size", (1, 32))
    @parametrize("mid_dim", (1, 8))
    @parametrize("in_features", (128,))
    @parametrize("out_features", (64,))
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @unittest.skipIf(TEST_WITH_SLOW_GRADCHECK, "Leaking memory")
    def test_int8_woq_mm_concat_cuda(
        self, dtype, batch_size, mid_dim, in_features, out_features
    ):
        def _convert_weight_to_int8pack(w):
            # Move to CPU for quantization calculation, then back to original device
            device = w.device
            w_cpu = w.cpu()
            scale, zp = _calculate_dynamic_per_channel_qparams(
                w_cpu.to(torch.float), torch.int8
            )
            scale = torch.from_numpy(scale).to(device)
            zp = torch.from_numpy(zp).to(device)
            w_int8 = torch.ao.quantization.fx._decomposed.quantize_per_channel(
                input=w,
                scales=scale,
                zero_points=zp,
                axis=0,
                quant_min=-128,
                quant_max=127,
                dtype=torch.int8,
            )
            return w_int8, scale.to(torch.bfloat16)

        class M(torch.nn.Module):
            def __init__(self, w1, w2, w3):
                super().__init__()
                self.w1 = torch.nn.Parameter(w1, requires_grad=False)
                self.w2 = torch.nn.Parameter(w2, requires_grad=False)
                self.w3 = torch.nn.Parameter(w3, requires_grad=False)

            def forward(self, x, scale1, scale2, scale3):
                # Ref: _linear_fp_act_int8_weight_impl in torchao/dtypes/uintx/plain_layout.py
                y1 = (
                    torch.mm(x.reshape(-1, x.shape[-1]), self.w1.t().to(x.dtype))
                    * scale1
                )
                y2 = (
                    torch.mm(x.reshape(-1, x.shape[-1]), self.w2.t().to(x.dtype))
                    * scale2
                )
                y3 = (
                    torch.mm(x.reshape(-1, x.shape[-1]), self.w3.t().to(x.dtype))
                    * scale3
                )
                return (
                    y1.reshape(*x.shape[:-1], y1.shape[-1]),
                    y2.reshape(*x.shape[:-1], y2.shape[-1]),
                    y3.reshape(*x.shape[:-1], y3.shape[-1]),
                )

        counters.clear()
        # Currently, the corresponding torch.fx pattern only supports 3D x
        # Add 2D X case once the corresponding pattern-matcher pattern is added
        x = torch.rand((batch_size, mid_dim, in_features), dtype=dtype, device="cuda")
        w1 = torch.rand((out_features, in_features), dtype=dtype, device="cuda")
        w2 = torch.rand((out_features, in_features), dtype=dtype, device="cuda")
        w3 = torch.rand((out_features, in_features), dtype=dtype, device="cuda")
        w1_int8pack, w1_scales = _convert_weight_to_int8pack(w1)
        w2_int8pack, w2_scales = _convert_weight_to_int8pack(w2)
        w3_int8pack, w3_scales = _convert_weight_to_int8pack(w3)
        mod = M(w1_int8pack, w2_int8pack, w3_int8pack).eval()
        self.common(mod, (x, w1_scales, w2_scales, w3_scales))
        self.assertEqual(counters["inductor"]["woq_matcher_count"], 3)


instantiate_device_type_tests(TestSelectAlgorithmCuda, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 3 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestSelectAlgorithmCuda`, `M`, `M`

**Functions defined**: `patches`, `skip_cache`, `wrapped`, `test_int8_woq_mm_cuda`, `_convert_weight_to_int8pack`, `__init__`, `forward`, `test_int8_woq_mm_concat_cuda`, `_convert_weight_to_int8pack`, `__init__`, `forward`

**Key imports**: functools, sys, unittest, patch, torch, torch._dynamo.config as dynamo_config, torch._inductor.config as inductor_config, torch._inductor.select_algorithm as select_algorithm, counters, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `sys`
- `unittest`
- `unittest.mock`: patch
- `torch`
- `torch._dynamo.config as dynamo_config`
- `torch._inductor.config as inductor_config`
- `torch._inductor.select_algorithm as select_algorithm`
- `torch._dynamo.utils`: counters
- `torch._inductor.test_case`: run_tests
- `.`: test_cpu_select_algorithm, test_torchinductor
- `test_cpu_select_algorithm  `
- `test_torchinductor  `


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/inductor/test_cuda_select_algorithm.py
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

- **File Documentation**: `test_cuda_select_algorithm.py_docs.md`
- **Keyword Index**: `test_cuda_select_algorithm.py_kw.md`
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

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/inductor/test_cuda_select_algorithm.py_docs.md
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

- **File Documentation**: `test_cuda_select_algorithm.py_docs.md_docs.md`
- **Keyword Index**: `test_cuda_select_algorithm.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
