# Documentation: `docs/test/inductor/test_binary_folding.py_docs.md`

## File Metadata

- **Path**: `docs/test/inductor/test_binary_folding.py_docs.md`
- **Size**: 15,911 bytes (15.54 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/inductor/test_binary_folding.py`

## File Metadata

- **Path**: `test/inductor/test_binary_folding.py`
- **Size**: 12,347 bytes (12.06 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]
import functools
import importlib
import itertools
import os
import sys

import torch
from torch import nn
from torch._dynamo.utils import counters
from torch._inductor import config as inductor_config
from torch.testing._internal.common_cuda import TEST_CUDNN


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

from inductor.test_inductor_freezing import (  # @manual=fbcode//caffe2/test/inductor:inductor_freezing-library
    TestCase,
)
from inductor.test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    check_model,
    check_model_gpu,
    copy_tests,
)
from torch.testing._internal.inductor_utils import skipCUDAIf


importlib.import_module("functorch")
importlib.import_module("filelock")

from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU


aten = torch.ops.aten


class BinaryFoldingTemplate(TestCase):
    @skipCUDAIf(TEST_CUDNN, "CUDNN has accuracy issues for this test")
    def test_conv_binary_folding(self):
        @torch.no_grad()
        def test_conv_fusion(
            use_bias,
            module,
            op,
            scalar,
            add_tensor,
            expect_success,
            rtol=None,
            atol=None,
        ):
            class ConvOp(nn.Module):
                __constants__ = ["use_scalar"]

                def __init__(self, in_channels, out_channels, device, **kwargs):
                    super().__init__()
                    self.conv = module(
                        in_channels, out_channels, bias=use_bias, **kwargs
                    ).to(device)
                    self.use_scalar = scalar
                    tensor_size = [1 for _ in range(self.conv.weight.ndim)]
                    tensor_size[1] = self.conv.weight.size(0)
                    self.tensor = torch.nn.Parameter(
                        add_tensor
                        if add_tensor is not None
                        else torch.rand(tensor_size).to(device)
                    )
                    self.op = op

                def forward(self, x):
                    x = self.conv(x)
                    if self.use_scalar:
                        return self.op(x, 2.0)
                    else:
                        return self.op(x, self.tensor)

            torch._dynamo.reset()
            counters.clear()
            mod_eager = ConvOp(3, 32, self.device, kernel_size=3, stride=2).eval()
            out_optimized = torch.compile(mod_eager)

            inps = [4, 3, 4]
            if module is nn.Conv2d:
                inps.append(inps[-1])
            if module is nn.Conv3d:
                inps.append(inps[-1])
                inps.append(inps[-1])

            torch.manual_seed(1234)
            inp = torch.rand(inps).to(self.device)
            out_eager = mod_eager(inp)
            out_optimized = out_optimized(inp)
            self.assertEqual(out_optimized, out_eager, rtol=rtol, atol=atol)
            if expect_success:
                self.assertEqual(counters["inductor"]["binary_folding"], 1)
            else:
                self.assertEqual(counters["inductor"]["binary_folding"], 0)

        conv_bias = [True, False]
        modules = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
        use_scalar = [True, False]
        ops = [torch.add, torch.sub, torch.mul, torch.div]
        for use_bias, module, pytorch_op, scalar in itertools.product(
            conv_bias, modules, ops, use_scalar
        ):
            test_conv_fusion(
                use_bias,
                module,
                pytorch_op,
                scalar,
                add_tensor=None,
                expect_success=True,
            )

        for use_bias, pytorch_op in itertools.product(conv_bias, ops):
            # broadcasting add
            test_conv_fusion(
                use_bias,
                nn.Conv2d,
                pytorch_op,
                False,
                add_tensor=torch.rand(
                    32,
                    1,
                    32,
                ).to(self.device),
                expect_success=False,
            )

            # broadcasting add
            test_conv_fusion(
                use_bias,
                nn.Conv2d,
                pytorch_op,
                False,
                add_tensor=torch.rand(1, 1).to(self.device),
                expect_success=True,
            )

            # add with different dtype
            test_conv_fusion(
                use_bias,
                nn.Conv2d,
                pytorch_op,
                False,
                add_tensor=torch.tensor([2]).to(torch.float64).to(self.device),
                expect_success=False,
                # This test is for float32 conv fusion with different dtype, like float64,
                # which will not be fused. The tolerance of float64 is too tight
                # for float32 conv post fusion with float64 tensor. Will relax the tolerance
                # for this case.
                rtol=1.3e-6,
                atol=1e-5,
            )

    @inductor_config.patch({"freezing": True})
    def test_conv_bn_folding(self):
        @torch.no_grad()
        def test_conv_fusion(use_bias, module, expect_success):
            class ConvOp(nn.Module):
                def __init__(self, in_channels, out_channels, device, **kwargs):
                    super().__init__()
                    self.conv = module[0](
                        in_channels, out_channels, bias=use_bias, **kwargs
                    ).to(device)
                    self.bn = module[1](out_channels).to(device)

                def forward(self, x):
                    x = self.conv(x)
                    return self.bn(x)

            from torch._inductor.compile_fx import compile_fx, compile_fx_inner

            aten_binary = [
                aten.add.Tensor,
                aten.sub.Tensor,
                aten.mul.Tensor,
                aten.div.Tensor,
            ]
            n_binary_ops = 0

            def my_inner_compile(gm, example_inputs, *args, **kwargs):
                out = compile_fx_inner(gm, example_inputs, *args, **kwargs)
                nonlocal n_binary_ops
                binarry_ops = [n for n in gm.graph.nodes if n.target in aten_binary]
                n_binary_ops += len(binarry_ops)
                return out

            torch._dynamo.reset()
            mod_eager = ConvOp(3, 32, self.device, kernel_size=3, stride=2).eval()
            out_optimized = torch.compile(
                mod_eager,
                backend=functools.partial(compile_fx, inner_compile=my_inner_compile),
            )

            inps = [4, 3, 4]
            if module[0] is nn.Conv2d:
                inps.append(inps[-1])
            if module[0] is nn.Conv3d:
                inps.append(inps[-1])
                inps.append(inps[-1])

            inp = torch.rand(inps).to(self.device)
            out_eager = mod_eager(inp)
            out_optimized = out_optimized(inp)
            self.assertEqual(out_optimized, out_eager, atol=2e-04, rtol=1e-5)
            if expect_success:
                self.assertTrue(n_binary_ops == 0)
            else:
                self.assertTrue(n_binary_ops > 1)

        conv_bias = [True, False]
        modules = [
            (nn.Conv1d, nn.BatchNorm1d),
            (nn.Conv2d, nn.BatchNorm2d),
            (nn.Conv3d, nn.BatchNorm3d),
        ]
        for use_bias, module in itertools.product(conv_bias, modules):
            test_conv_fusion(
                use_bias,
                module,
                expect_success=True,
            )

    @inductor_config.patch({"enable_linear_binary_folding": True})
    def test_linear_binary_folding(self):
        @torch.no_grad()
        def test_linear_fusion(
            use_bias, op, scalar, add_tensor, expect_success, input_3d=False
        ):
            class LinearOp(nn.Module):
                __constants__ = ["use_scalar"]

                def __init__(self, in_channels, out_channels, device, **kwargs):
                    super().__init__()
                    self.linear = nn.Linear(
                        in_channels, out_channels, bias=use_bias, **kwargs
                    ).to(device)
                    self.use_scalar = scalar
                    tensor_size = [
                        self.linear.weight.size(0),
                    ]
                    self.tensor = torch.nn.Parameter(
                        add_tensor
                        if add_tensor is not None
                        else torch.rand(tensor_size).to(device)
                    )
                    self.op = op

                def forward(self, x):
                    x = self.linear(x)
                    if self.use_scalar:
                        return self.op(x, 2.0)
                    else:
                        return self.op(x, self.tensor)

            torch._dynamo.reset()
            counters.clear()
            mod_eager = LinearOp(3, 32, self.device).eval()
            out_optimized = torch.compile(mod_eager)

            torch.manual_seed(1234)
            if input_3d:
                inp = torch.rand([2, 4, 3]).to(self.device)
            else:
                inp = torch.rand([4, 3]).to(self.device)
            out_eager = mod_eager(inp)
            out_optimized = out_optimized(inp)
            self.assertEqual(out_optimized, out_eager, atol=5e-05, rtol=5e-06)
            if expect_success:
                self.assertEqual(counters["inductor"]["binary_folding"], 1)
            else:
                self.assertEqual(counters["inductor"]["binary_folding"], 0)

        linear_bias = [True, False]
        use_scalar = [True, False]
        ops = [torch.add, torch.sub, torch.mul, torch.div]
        add_tensor_size = [
            [
                32,
            ],
            [1, 32],
            [
                1,
            ],
            [1, 1],
        ]
        for use_bias, pytorch_op, scalar, tensor_size in itertools.product(
            linear_bias, ops, use_scalar, add_tensor_size
        ):
            test_linear_fusion(
                use_bias,
                pytorch_op,
                scalar,
                add_tensor=torch.rand(tensor_size).to(self.device),
                expect_success=True,
            )

        add_tensor_size.extend([[1, 1, 32], [1, 1, 1]])
        for use_bias, pytorch_op, scalar, tensor_size in itertools.product(
            linear_bias, ops, use_scalar, add_tensor_size
        ):
            test_linear_fusion(
                use_bias,
                pytorch_op,
                scalar,
                add_tensor=torch.rand(tensor_size).to(self.device),
                expect_success=True,
                input_3d=True,
            )

        # In the following test, the shape of 'add_tensor' does not satisfy
        # the requirements of binary folding, so it will not be folded.
        for use_bias, pytorch_op in itertools.product(linear_bias, ops):
            test_linear_fusion(
                use_bias,
                pytorch_op,
                False,
                add_tensor=torch.rand(
                    4,
                    32,
                ).to(self.device),
                expect_success=False,
            )

            test_linear_fusion(
                use_bias,
                pytorch_op,
                False,
                add_tensor=torch.rand(
                    4,
                    1,
                ).to(self.device),
                expect_success=False,
            )


if HAS_CPU and not torch.backends.mps.is_available():

    class FreezingCpuTests(TestCase):
        common = check_model
        device = "cpu"
        autocast = torch.cpu.amp.autocast

    copy_tests(BinaryFoldingTemplate, FreezingCpuTests, "cpu")

if HAS_GPU:

    class FreezingGpuTests(TestCase):
        common = check_model_gpu
        device = GPU_TYPE
        autocast = torch.amp.autocast(device_type=GPU_TYPE)

    copy_tests(BinaryFoldingTemplate, FreezingGpuTests, GPU_TYPE)


del BinaryFoldingTemplate

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_GPU:
        run_tests(needs="filelock")

```



## High-Level Overview


This Python file contains 6 class(es) and 13 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `BinaryFoldingTemplate`, `ConvOp`, `ConvOp`, `LinearOp`, `FreezingCpuTests`, `FreezingGpuTests`

**Functions defined**: `test_conv_binary_folding`, `test_conv_fusion`, `__init__`, `forward`, `test_conv_bn_folding`, `test_conv_fusion`, `__init__`, `forward`, `my_inner_compile`, `test_linear_binary_folding`, `test_linear_fusion`, `__init__`, `forward`

**Key imports**: functools, importlib, itertools, os, sys, torch, nn, counters, config as inductor_config, TEST_CUDNN


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `importlib`
- `itertools`
- `os`
- `sys`
- `torch`
- `torch._dynamo.utils`: counters
- `torch._inductor`: config as inductor_config
- `torch.testing._internal.common_cuda`: TEST_CUDNN
- `torch.testing._internal.inductor_utils`: skipCUDAIf
- `torch._inductor.compile_fx`: compile_fx, compile_fx_inner
- `torch._inductor.test_case`: run_tests


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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
python test/inductor/test_binary_folding.py
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

- **File Documentation**: `test_binary_folding.py_docs.md`
- **Keyword Index**: `test_binary_folding.py_kw.md`
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

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/inductor/test_binary_folding.py_docs.md
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

- **File Documentation**: `test_binary_folding.py_docs.md_docs.md`
- **Keyword Index**: `test_binary_folding.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
