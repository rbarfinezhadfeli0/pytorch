# Documentation: `docs/test/inductor/test_fx_fusion.py_docs.md`

## File Metadata

- **Path**: `docs/test/inductor/test_fx_fusion.py_docs.md`
- **Size**: 9,202 bytes (8.99 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/inductor/test_fx_fusion.py`

## File Metadata

- **Path**: `test/inductor/test_fx_fusion.py`
- **Size**: 5,982 bytes (5.84 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]
from collections.abc import Callable
from typing import Any

import torch
from torch._inductor.fx_passes.pre_grad import (
    linear_permute_fusion,
    linear_transpose,
    permute_linear_fusion,
    permute_matmul_fusion,
    sink_cat_after_pointwise,
    transpose_linear,
    transpose_matmul,
)
from torch._inductor.test_case import run_tests, TestCase
from torch.fx.passes.shape_prop import ShapeProp


PassFunc = Callable[[torch.fx.GraphModule, Any], torch.fx.GraphModule]


def chain_passes(*passes: PassFunc) -> PassFunc:
    def parent_pass(module: torch.fx.GraphModule, input: Any) -> torch.fx.GraphModule:
        for pass_ in passes:
            if isinstance(module, torch.fx.GraphModule):
                ShapeProp(module).propagate(*input)
            module = pass_(module)
        return module

    return parent_pass


def count_call(module: torch.fx.GraphModule, op: str, target_op: Any) -> int:
    return sum(
        1 if (n.op == op and n.target == target_op) else 0 for n in module.graph.nodes
    )


def count_call_function(module: torch.fx.GraphModule, target_op: Any) -> int:
    return count_call(module, "call_function", target_op)


def count_call_method(module: torch.fx.GraphModule, target_op: Any) -> int:
    return count_call(module, "call_method", target_op)


class TestFxFusion(TestCase):
    def test_sink_cat_after_pointwise(self):
        def test_kwarg(x, y):
            return torch.cat([x, y], dim=-1).view(-1).view(128).tanh()

        def test_arg(x, y):
            return torch.cat([x, y], -1).view(-1).view(128).tanh()

        def test_arg2(x, y):
            return torch.cat([x, y]).view(-1).view(128).tanh()

        def test_kwarg2(x, y):
            return torch.cat(tensors=[x, y], dim=0).tanh()

        def test_kwarg3(x, y):
            return torch.cat(tensors=[x, y], dim=0).view(128).tanh()

        trace_func = chain_passes(torch.fx.symbolic_trace, sink_cat_after_pointwise)
        inputs = [
            torch.randn(8, 8),
            torch.randn(8, 8),
        ]
        for f in [test_kwarg, test_arg, test_arg2, test_kwarg2, test_kwarg3]:
            traced = trace_func(f, inputs)
            torch.testing.assert_close(f(*inputs), traced(*inputs))
            self.assertEqual(count_call_method(traced, "tanh"), 2)

    def test_linear_permute_fusion(self):
        class TestModule(torch.nn.Module):
            def __init__(self, k: int, n: int, has_bias: bool):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(n, k))
                self.has_bias = has_bias
                if has_bias:
                    self.bias = torch.nn.Parameter(torch.randn(n))

            def forward(self, input: torch.Tensor):
                if self.has_bias:
                    a0 = torch.nn.functional.linear(input, self.weight, self.bias)
                else:
                    a0 = torch.nn.functional.linear(input, self.weight)
                b0 = a0.permute(0, 2, 1)
                return b0

        m, k, n = 16, 8, 4
        trace_func = chain_passes(torch.fx.symbolic_trace, linear_permute_fusion)
        for has_bias in [True, False]:
            module = TestModule(k, n, has_bias).eval()
            input = torch.randn(6, m, k)
            traced = trace_func(module, [input])
            num_linear = count_call_function(traced, torch.nn.functional.linear)
            num_linear_transpose = count_call_function(traced, linear_transpose)
            self.assertEqual(num_linear, 0)
            self.assertEqual(num_linear_transpose, 1)

            torch.testing.assert_close(module(input), traced(input))

    def test_permute_linear_fusion(self):
        class TestModule(torch.nn.Module):
            def __init__(self, k: int, n: int, has_bias: bool):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(n, k))
                self.has_bias = has_bias
                if has_bias:
                    self.bias = torch.nn.Parameter(torch.randn(n))

            def forward(self, input: torch.Tensor):
                input1 = input.permute(0, 2, 1)
                if self.has_bias:
                    return torch.nn.functional.linear(input1, self.weight, self.bias)
                return torch.nn.functional.linear(input1, self.weight)

        m, k, n = 16, 8, 4

        trace_func = chain_passes(torch.fx.symbolic_trace, permute_linear_fusion)
        for has_bias in [True, False]:
            module = TestModule(k, n, has_bias).eval()
            input = torch.randn(6, k, m)
            traced = trace_func(module, [input])
            num_linear = count_call_function(traced, torch.nn.functional.linear)
            num_transpose_linear = count_call_function(traced, transpose_linear)
            self.assertEqual(num_linear, 0)
            self.assertEqual(num_transpose_linear, 1)

            torch.testing.assert_close(module(input), traced(input))

    def test_permute_bmm_fusion(self):
        class TestModule(torch.nn.Module):
            def __init__(self, batch: int, k: int, n: int):
                super().__init__()
                self.other = torch.randn(batch, k, n)

            def forward(self, input: torch.Tensor):
                input1 = input.permute(0, 2, 1)
                output = torch.bmm(input1, self.other)
                return output

        batch, m, k, n = 6, 16, 8, 4

        trace_func = chain_passes(torch.fx.symbolic_trace, permute_matmul_fusion)
        module = TestModule(batch, k, n).eval()
        input = torch.randn(batch, k, m)
        traced = trace_func(module, [input])
        num_bmm = count_call_function(traced, torch.bmm)
        num_transpose_matmul = count_call_function(traced, transpose_matmul)
        self.assertEqual(num_bmm, 0)
        self.assertEqual(num_transpose_matmul, 1)

        torch.testing.assert_close(module(input), traced(input))


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 4 class(es) and 20 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestFxFusion`, `TestModule`, `TestModule`, `TestModule`

**Functions defined**: `chain_passes`, `parent_pass`, `count_call`, `count_call_function`, `count_call_method`, `test_sink_cat_after_pointwise`, `test_kwarg`, `test_arg`, `test_arg2`, `test_kwarg2`, `test_kwarg3`, `test_linear_permute_fusion`, `__init__`, `forward`, `test_permute_linear_fusion`, `__init__`, `forward`, `test_permute_bmm_fusion`, `__init__`, `forward`

**Key imports**: Callable, Any, torch, run_tests, TestCase, ShapeProp


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Callable
- `typing`: Any
- `torch`
- `torch._inductor.test_case`: run_tests, TestCase
- `torch.fx.passes.shape_prop`: ShapeProp


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/inductor/test_fx_fusion.py
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

- **File Documentation**: `test_fx_fusion.py_docs.md`
- **Keyword Index**: `test_fx_fusion.py_kw.md`
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
python docs/test/inductor/test_fx_fusion.py_docs.md
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

- **File Documentation**: `test_fx_fusion.py_docs.md_docs.md`
- **Keyword Index**: `test_fx_fusion.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
