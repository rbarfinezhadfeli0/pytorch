# Documentation: `docs/test/test_functionalization_of_rng_ops.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_functionalization_of_rng_ops.py_docs.md`
- **Size**: 15,178 bytes (14.82 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_functionalization_of_rng_ops.py`

## File Metadata

- **Path**: `test/test_functionalization_of_rng_ops.py`
- **Size**: 11,575 bytes (11.30 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: pt2"]
import functools
import sys
import unittest
from unittest.mock import patch

import torch
import torch.utils.checkpoint
from functorch.compile import aot_function, min_cut_rematerialization_partition, nop

from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)

from torch.testing._internal.common_utils import IS_CI, IS_WINDOWS, run_tests, TestCase

if IS_WINDOWS and IS_CI:
    sys.stderr.write("torch.compile not supported on windows")
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("torch.compile not supported on windows")


def count_philox_rand(gm, args, freq):
    assert [node.target for node in gm.graph.nodes].count(
        torch.ops.rngprims.philox_rand.default
    ) == freq
    return gm


class TestFunctionalizationRngOps(TestCase):
    @dtypes(torch.float32)
    @patch.object(torch._functorch.config, "functionalize_rng_ops", True)
    def test_rand_like(self, dtype, device):
        def fn(x):
            a = torch.rand_like(x) * x
            a = torch.rand_like(x) * a
            return a

        x = torch.rand(10, device=device, dtype=dtype)

        for seed in range(10):
            torch.cuda.manual_seed(seed)
            ref = fn(x)

            torch.cuda.manual_seed(seed)
            aot_fn = aot_function(fn, functools.partial(count_philox_rand, freq=2))
            res = aot_fn(x)

            self.assertEqual(ref, res)

    @dtypes(torch.float32)
    @patch.object(torch._functorch.config, "functionalize_rng_ops", True)
    def test_rand_like_dynamic(self, dtype, device):
        def fn(x):
            a = torch.rand_like(x) * x
            a = torch.rand_like(x) * a
            return a

        for seed in range(1, 10):
            shape = (seed, seed)
            x = torch.rand(shape, device=device, dtype=dtype)
            torch.cuda.manual_seed(seed)
            ref = fn(x)

            torch.cuda.manual_seed(seed)
            opt_fn = torch.compile(fn, backend="aot_eager", dynamic=True)
            res = opt_fn(x)

            self.assertEqual(ref, res)

    @dtypes(torch.float32)
    @patch.object(torch._functorch.config, "functionalize_rng_ops", True)
    def test_rand_like_dynamic_bwd(self, dtype, device):
        def fn(x):
            a = torch.rand_like(x) * x
            a = torch.rand_like(x) * a
            return a

        for seed in range(1, 10):
            shape = (seed, seed)
            x = torch.rand(shape, device=device, dtype=dtype, requires_grad=True)
            torch.cuda.manual_seed(seed)
            ref = fn(x)
            ref.sum().backward()

            torch.cuda.manual_seed(seed)
            opt_fn = torch.compile(fn, backend="aot_eager", dynamic=True)
            res = opt_fn(x)
            res.sum().backward()

            self.assertEqual(ref, res)

    @dtypes(torch.float32)
    @patch.object(torch._functorch.config, "functionalize_rng_ops", True)
    def test_rand(self, dtype, device):
        shape = (10,)

        def fn(x):
            a = torch.rand(*shape, device=device, dtype=dtype) * x
            a = torch.rand(*shape, device=device, dtype=dtype) * a
            return a

        x = torch.rand(*shape, device=device, dtype=dtype)

        for seed in range(10):
            torch.cuda.manual_seed(seed)
            ref = fn(x)

            torch.cuda.manual_seed(seed)
            aot_fn = aot_function(fn, functools.partial(count_philox_rand, freq=2))
            res = aot_fn(x)

            self.assertEqual(ref, res)

    @dtypes(torch.float32)
    @patch.object(torch._functorch.config, "functionalize_rng_ops", True)
    def test_autograd_function(self, dtype, device):
        shape = (16, 16)

        class Custom(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                a = torch.rand_like(x) * x
                a = torch.rand_like(x) * a
                return a

            @staticmethod
            def backward(ctx, grad_out):
                (x,) = ctx.saved_tensors
                return grad_out * torch.rand_like(grad_out) * torch.cos(x)

        custom = Custom.apply

        x = torch.rand(*shape, device=device, dtype=dtype, requires_grad=True)

        x_clone = x.detach().clone().requires_grad_(True)

        torch.cuda.manual_seed(123)
        ref = custom(x)
        ref.sum().backward()

        torch.cuda.manual_seed(123)
        fwd_compiler = functools.partial(count_philox_rand, freq=2)
        bwd_compiler = functools.partial(count_philox_rand, freq=1)
        aot_custom = aot_function(custom, fwd_compiler, bwd_compiler)
        res = aot_custom(x_clone)
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)

    @dtypes(torch.float32)
    @patch.object(torch._functorch.config, "functionalize_rng_ops", True)
    def test_multiple_subgraphs(self, dtype, device):
        # Checks that rng state is maintained when there are multiple aot traced
        # graphs.
        shape = (16, 16)

        class CustomOp1(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                a = torch.rand_like(x) * x
                a = torch.rand_like(x) * a
                return a

            @staticmethod
            def backward(ctx, grad_out):
                (x,) = ctx.saved_tensors
                return grad_out * torch.rand_like(grad_out) * torch.cos(x)

        class CustomOp2(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                a = torch.rand_like(x) * x
                return a

            @staticmethod
            def backward(ctx, grad_out):
                (x,) = ctx.saved_tensors
                return grad_out * torch.rand_like(grad_out) * torch.rand_like(x)

        custom_op1 = CustomOp1.apply
        custom_op2 = CustomOp2.apply

        def fn(x):
            a = custom_op1(x)
            b = a.sin()
            return custom_op2(b)

        fwd_compiler = functools.partial(count_philox_rand, freq=2)
        bwd_compiler = functools.partial(count_philox_rand, freq=1)
        aot_custom_op1 = aot_function(custom_op1, fwd_compiler, bwd_compiler)
        fwd_compiler = functools.partial(count_philox_rand, freq=1)
        bwd_compiler = functools.partial(count_philox_rand, freq=2)
        aot_custom_op2 = aot_function(custom_op2, fwd_compiler, bwd_compiler)

        def aot_fn(x):
            a = aot_custom_op1(x)
            b = a.sin()
            return aot_custom_op2(b)

        for seed in range(10):
            torch.cuda.manual_seed(seed)
            x = torch.rand(*shape, device=device, dtype=dtype, requires_grad=True)
            x_clone = x.detach().clone().requires_grad_(True)

            torch.cuda.manual_seed(seed)
            ref = fn(x)
            ref.sum().backward()

            torch.cuda.manual_seed(seed)
            res = aot_fn(x_clone)
            res.sum().backward()

            self.assertEqual(ref, res)
            self.assertEqual(x.grad, x_clone.grad)

    @dtypes(torch.float32)
    @patch.object(torch._functorch.config, "functionalize_rng_ops", True)
    def test_set_get_rng_state(self, dtype, device):
        def fn(x):
            a = torch.rand_like(x) * x
            state = torch.cuda.get_rng_state()
            a = torch.rand_like(x) * a
            torch.cuda.set_rng_state(state)
            a = torch.rand_like(x) * a
            return a

        x = torch.rand(10, device=device, dtype=dtype)

        for seed in range(10):
            torch.cuda.manual_seed(seed)
            ref = fn(x)

            torch.cuda.manual_seed(seed)
            fwd_compiler = functools.partial(count_philox_rand, freq=3)
            aot_fn = aot_function(fn, fwd_compiler)
            res = aot_fn(x)

            self.assertEqual(ref, res)

    @dtypes(torch.float32)
    @patch.object(torch._functorch.config, "functionalize_rng_ops", True)
    def test_min_cut_partitioner(self, dtype, device):
        # Checks that the calling convention is maintained
        shape = (16, 16)

        def fn(x):
            a = torch.rand_like(x) * x
            a = torch.rand_like(x) * a
            a = torch.sin(a)
            a = torch.sin(a)
            a = torch.sin(a)
            return a

        x = torch.rand(*shape, device=device, dtype=dtype, requires_grad=True)

        x_clone = x.detach().clone().requires_grad_(True)

        torch.cuda.manual_seed(123)
        ref = fn(x)
        ref.sum().backward()

        torch.cuda.manual_seed(123)
        fwd_compiler = functools.partial(count_philox_rand, freq=2)
        bwd_compiler = functools.partial(count_philox_rand, freq=0)
        aot_custom = aot_function(
            fn,
            fwd_compiler,
            bwd_compiler,
            partition_fn=min_cut_rematerialization_partition,
        )
        # aot_custom = aot_function(fn, fwd_compiler, bwd_compiler)
        res = aot_custom(x_clone)
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)

    # TODO - Dropout needs more work because of offset calculation
    @patch.object(torch._functorch.config, "functionalize_rng_ops", True)
    @dtypes(torch.float32)
    def test_checkpoint(self, dtype, device):
        def g(x, y):
            return torch.nn.functional.dropout(x, 0.6)

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(g, x, y, use_reentrant=False)

        # x = torch.rand(2, 2, device="cuda", requires_grad=True)
        x = torch.ones(2, 2, device="cuda", requires_grad=True)
        y = torch.rand(2, 2, device="cuda", requires_grad=True)
        torch.cuda.manual_seed(123)
        fn(x, y)

        # With checkpointing we should recompute dropout in bwd, and philox_rand is passed from fwd
        fwd_compiler = functools.partial(count_philox_rand, freq=1)
        bwd_compiler = functools.partial(count_philox_rand, freq=0)
        aot_fn = aot_function(fn, fwd_compiler, bwd_compiler)
        # We can't check accuracy here because rand_like generated different rand numbers than dropout
        res = aot_fn(x, y)
        res.sum().backward()

    @dtypes(torch.float32)
    @patch.object(torch._functorch.config, "functionalize_rng_ops", True)
    def test_dropout_decomp(self, dtype, device):
        def fn(x):
            return torch.nn.functional.dropout(x, 0.6) * x

        x = torch.rand(10, device=device, dtype=dtype)

        # Ensure the decomp is happening
        aot_fn = aot_function(fn, functools.partial(count_philox_rand, freq=1))
        # We can't check accuracy here because rand_like generated different rand numbers than dropout
        aot_fn(x)


only_for = ("cuda",)
instantiate_device_type_tests(TestFunctionalizationRngOps, globals(), only_for=only_for)


class NegativeTest(TestCase):
    @dtypes(torch.float32)
    @patch.object(torch._functorch.config, "functionalize_rng_ops", True)
    def test_on_cpu(self, dtype, device):
        def fn(x):
            a = torch.rand_like(x) * x
            a = torch.rand_like(x) * a
            return a

        x = torch.rand(10, device=device, dtype=dtype)

        aot_fn = aot_function(fn, nop)
        with self.assertRaises(RuntimeError):
            aot_fn(x)


only_for = ("cpu",)
instantiate_device_type_tests(NegativeTest, globals(), only_for=only_for)

if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 5 class(es) and 30 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestFunctionalizationRngOps`, `Custom`, `CustomOp1`, `CustomOp2`, `NegativeTest`

**Functions defined**: `count_philox_rand`, `test_rand_like`, `fn`, `test_rand_like_dynamic`, `fn`, `test_rand_like_dynamic_bwd`, `fn`, `test_rand`, `fn`, `test_autograd_function`, `forward`, `backward`, `test_multiple_subgraphs`, `forward`, `backward`, `forward`, `backward`, `fn`, `aot_fn`, `test_set_get_rng_state`

**Key imports**: functools, sys, unittest, patch, torch, torch.utils.checkpoint, aot_function, min_cut_rematerialization_partition, nop, IS_CI, IS_WINDOWS, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `sys`
- `unittest`
- `unittest.mock`: patch
- `torch`
- `torch.utils.checkpoint`
- `functorch.compile`: aot_function, min_cut_rematerialization_partition, nop
- `torch.testing._internal.common_utils`: IS_CI, IS_WINDOWS, run_tests, TestCase


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


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
python test/test_functionalization_of_rng_ops.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test`):

- [`test_file_check.py_docs.md`](./test_file_check.py_docs.md)
- [`test_jit_simple.py_docs.md`](./test_jit_simple.py_docs.md)
- [`test_mkldnn.py_docs.md`](./test_mkldnn.py_docs.md)
- [`test_expanded_weights.py_docs.md`](./test_expanded_weights.py_docs.md)
- [`test_overrides.py_docs.md`](./test_overrides.py_docs.md)
- [`test_decomp.py_docs.md`](./test_decomp.py_docs.md)
- [`test_show_pickle.py_docs.md`](./test_show_pickle.py_docs.md)
- [`test_utils_config_module.py_docs.md`](./test_utils_config_module.py_docs.md)
- [`test_mobile_optimizer.py_docs.md`](./test_mobile_optimizer.py_docs.md)
- [`test_type_info.py_docs.md`](./test_type_info.py_docs.md)


## Cross-References

- **File Documentation**: `test_functionalization_of_rng_ops.py_docs.md`
- **Keyword Index**: `test_functionalization_of_rng_ops.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/test_functionalization_of_rng_ops.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test`):

- [`test_ops.py_docs.md_docs.md`](./test_ops.py_docs.md_docs.md)
- [`test_tensorexpr.py_docs.md_docs.md`](./test_tensorexpr.py_docs.md_docs.md)
- [`pytest_shard_custom.py_docs.md_docs.md`](./pytest_shard_custom.py_docs.md_docs.md)
- [`test_weak.py_kw.md_docs.md`](./test_weak.py_kw.md_docs.md)
- [`test_view_ops.py_kw.md_docs.md`](./test_view_ops.py_kw.md_docs.md)
- [`test_varlen_attention.py_kw.md_docs.md`](./test_varlen_attention.py_kw.md_docs.md)
- [`test_namedtensor.py_docs.md_docs.md`](./test_namedtensor.py_docs.md_docs.md)
- [`test_binary_ufuncs.py_docs.md_docs.md`](./test_binary_ufuncs.py_docs.md_docs.md)
- [`test_ops_gradients.py_kw.md_docs.md`](./test_ops_gradients.py_kw.md_docs.md)
- [`test_torchfuzz_repros.py_docs.md_docs.md`](./test_torchfuzz_repros.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_functionalization_of_rng_ops.py_docs.md_docs.md`
- **Keyword Index**: `test_functionalization_of_rng_ops.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
