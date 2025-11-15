# Documentation: `docs/test/inductor/test_efficient_conv_bn_eval.py_docs.md`

## File Metadata

- **Path**: `docs/test/inductor/test_efficient_conv_bn_eval.py_docs.md`
- **Size**: 10,651 bytes (10.40 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/inductor/test_efficient_conv_bn_eval.py`

## File Metadata

- **Path**: `test/inductor/test_efficient_conv_bn_eval.py`
- **Size**: 7,226 bytes (7.06 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]
import copy
import importlib
import itertools
import os
import sys

import torch
from torch import nn


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

from torch._dynamo.utils import counters
from torch._inductor import config as inductor_config
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_cuda import tf32_on_and_off
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU


importlib.import_module("functorch")
importlib.import_module("filelock")

from inductor.test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    copy_tests,
)


class ConvOp(nn.Module):
    expected_optimization_count = 1

    def __init__(
        self,
        conv_class,
        bn_class,
        use_bias,
        in_channels,
        out_channels,
        device,
        **kwargs,
    ):
        super().__init__()
        self.conv = conv_class(in_channels, out_channels, bias=use_bias, **kwargs).to(
            device
        )
        self.bn = bn_class(out_channels).to(device)

    def forward(self, x):
        x = self.conv(x)
        return self.bn(x)


class MultiUserConvOp(nn.Module):
    expected_optimization_count = 3

    def __init__(
        self,
        conv_class,
        bn_class,
        use_bias,
        in_channels,
        out_channels,
        device,
        **kwargs,
    ):
        super().__init__()
        self.conv1 = conv_class(in_channels, out_channels, bias=use_bias, **kwargs).to(
            device
        )
        self.bn1 = bn_class(out_channels).to(device)
        self.conv2 = conv_class(out_channels, out_channels, bias=use_bias, **kwargs).to(
            device
        )
        self.bn2 = bn_class(out_channels).to(device)
        self.conv3 = conv_class(out_channels, out_channels, bias=use_bias, **kwargs).to(
            device
        )
        self.bn3 = bn_class(out_channels).to(device)

    def forward(self, x):
        # this conv-bn pair can use efficient_conv_bn_eval
        x = self.bn1(self.conv1(input=x))
        # this conv-bn pair cannot use efficient_conv_bn_eval feature
        # just for the second forward of the `self.conv2`
        x = self.bn2(input=self.conv2(self.conv2(x)))
        # this conv-bn pair can use efficient_conv_bn_eval feature
        # just for the first forward of the `self.bn3`
        # test for multiple users of one computation node
        x = self.bn3(input=self.conv3(input=x))
        x = self.bn3(x) + x
        return x


class EfficientConvBNEvalTemplate(TestCase):
    @tf32_on_and_off(0.003)
    @inductor_config.patch({"efficient_conv_bn_eval_fx_passes": True})
    def test_basic(self):
        def test_conv_bn_eval(
            test_class, use_bias, module, sync_bn, decompose_nn_module
        ):
            from functorch import make_fx
            from torch._dispatch.python import enable_python_dispatcher

            kwargs = {"kernel_size": 3, "stride": 2} if module[0] != nn.Linear else {}
            mod_eager = test_class(
                module[0],
                module[1],
                use_bias,
                3,
                32,
                self.device,
                **kwargs,
            ).eval()
            # Copy module to test backward
            mod_optimized = copy.deepcopy(mod_eager)
            if sync_bn:
                mod_eager = nn.SyncBatchNorm.convert_sync_batchnorm(mod_eager).eval()
                mod_optimized = nn.SyncBatchNorm.convert_sync_batchnorm(
                    mod_optimized
                ).eval()
            torch._dynamo.reset()

            inps = [4, 3]
            # Conv shape goes from big to small, and ConvTranspose shape goes from small to big
            spatial_d = (
                4 if issubclass(module[0], nn.modules.conv._ConvTransposeNd) else 96
            )
            if module[0] is nn.Conv1d or module[0] is nn.ConvTranspose1d:
                inps += [spatial_d] * 1
            if module[0] is nn.Conv2d or module[0] is nn.ConvTranspose2d:
                inps += [spatial_d] * 2
            if module[0] is nn.Conv3d or module[0] is nn.ConvTranspose3d:
                inps += [spatial_d] * 3
            inp = torch.rand(inps).to(self.device)

            if decompose_nn_module:
                with enable_python_dispatcher():
                    mod_optimized = make_fx(mod_optimized, pre_dispatch=True)(inp)
            mod_optimized = torch.compile(mod_optimized)

            original_value = counters["inductor"]["efficient_conv_bn_eval"]

            optim_eager = torch.optim.SGD(mod_eager.parameters(), lr=1e-3)
            optim_optimized = torch.optim.SGD(mod_optimized.parameters(), lr=1e-3)

            optim_eager.zero_grad()
            optim_optimized.zero_grad()

            # test forward
            out_eager = mod_eager(inp)
            out_optimized = mod_optimized(inp)

            self.assertEqual(out_optimized, out_eager)

            out_eager.mean().backward()
            out_optimized.mean().backward()

            optim_eager.step()
            optim_optimized.step()
            # test forward (by testing forward again after one training iteration)
            inp_bw = torch.rand_like(inp)
            out_eager_bw = mod_eager(inp_bw)
            out_optimized_bw = mod_optimized(inp_bw)

            self.assertEqual(out_eager_bw, out_optimized_bw)
            current_value = counters["inductor"]["efficient_conv_bn_eval"]
            self.assertEqual(
                current_value - original_value, test_class.expected_optimization_count
            )

        conv_bias = [True, False]
        modules = [
            (nn.Linear, nn.BatchNorm1d),
            (nn.Conv1d, nn.BatchNorm1d),
            (nn.Conv2d, nn.BatchNorm2d),
            (nn.Conv3d, nn.BatchNorm3d),
            (nn.ConvTranspose1d, nn.BatchNorm1d),
            (nn.ConvTranspose2d, nn.BatchNorm2d),
            (nn.ConvTranspose3d, nn.BatchNorm3d),
        ]
        test_classes = [ConvOp, MultiUserConvOp]
        sync_bns = [False, True]
        decompose_nn_modules = [False, True]
        for (
            test_class,
            use_bias,
            module,
            sync_bn,
            decompose_nn_module,
        ) in itertools.product(
            test_classes,
            conv_bias,
            modules,
            sync_bns,
            decompose_nn_modules,
        ):
            test_conv_bn_eval(
                test_class, use_bias, module, sync_bn, decompose_nn_module
            )


if HAS_CPU and not torch.backends.mps.is_available():

    class EfficientConvBNEvalCpuTests(TestCase):
        device = "cpu"

    copy_tests(EfficientConvBNEvalTemplate, EfficientConvBNEvalCpuTests, "cpu")

if HAS_GPU:

    class EfficientConvBNEvalGpuTests(TestCase):
        device = GPU_TYPE

    copy_tests(EfficientConvBNEvalTemplate, EfficientConvBNEvalGpuTests, GPU_TYPE)

del EfficientConvBNEvalTemplate

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_GPU:
        run_tests(needs="filelock")

```



## High-Level Overview


This Python file contains 5 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ConvOp`, `MultiUserConvOp`, `EfficientConvBNEvalTemplate`, `EfficientConvBNEvalCpuTests`, `EfficientConvBNEvalGpuTests`

**Functions defined**: `__init__`, `forward`, `__init__`, `forward`, `test_basic`, `test_conv_bn_eval`

**Key imports**: copy, importlib, itertools, os, sys, torch, nn, counters, config as inductor_config, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `importlib`
- `itertools`
- `os`
- `sys`
- `torch`
- `torch._dynamo.utils`: counters
- `torch._inductor`: config as inductor_config
- `torch._inductor.test_case`: TestCase
- `torch.testing._internal.common_cuda`: tf32_on_and_off
- `torch.testing._internal.inductor_utils`: GPU_TYPE, HAS_CPU, HAS_GPU
- `functorch`: make_fx
- `torch._dispatch.python`: enable_python_dispatcher


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/inductor/test_efficient_conv_bn_eval.py
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

- **File Documentation**: `test_efficient_conv_bn_eval.py_docs.md`
- **Keyword Index**: `test_efficient_conv_bn_eval.py_kw.md`
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


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
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
python docs/test/inductor/test_efficient_conv_bn_eval.py_docs.md
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

- **File Documentation**: `test_efficient_conv_bn_eval.py_docs.md_docs.md`
- **Keyword Index**: `test_efficient_conv_bn_eval.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
