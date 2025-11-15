# Documentation: `docs/test/test_out_dtype_op.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_out_dtype_op.py_docs.md`
- **Size**: 13,029 bytes (12.72 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_out_dtype_op.py`

## File Metadata

- **Path**: `test/test_out_dtype_op.py`
- **Size**: 9,395 bytes (9.17 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# Owner(s): ["module: functorch"]
import unittest

import torch
import torch._dynamo
import torch._inductor
import torch._inductor.decomposition
from torch._higher_order_ops.out_dtype import out_dtype
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import (
    run_tests, TestCase, IS_WINDOWS, TEST_WITH_ROCM, IS_FBCODE, IS_REMOTE_GPU, TEST_CUDA
)
from torch.testing._internal.common_quantization import skipIfNoDynamoSupport
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import SM80OrLater, _get_torch_cuda_version


@unittest.skipIf(not torch._dynamo.is_dynamo_supported(), "dynamo isn't support")
class TestOutDtypeOp(TestCase):
    def test_out_dtype_make_fx(self):
        class M(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

            def forward(self, x):
                return out_dtype(
                    torch.ops.aten.mm.default, torch.int32, x, self.weight
                )

        weight = torch.randint(-128, 127, (5, 5), dtype=torch.int8)
        m = M(weight)
        x = torch.randint(-128, 127, (5, 5), dtype=torch.int8)

        gm = make_fx(m)(x)
        self.assertTrue(torch.allclose(m(x), gm(x)))

        gm = make_fx(torch.func.functionalize(M(weight)))(x)
        self.assertTrue(torch.allclose(m(x), gm(x)))

        FileCheck().check("torch.ops.higher_order.out_dtype").check("aten.mm.default").run(gm.code)
        self.assertTrue(torch.allclose(m(x), gm(x)))
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target is out_dtype:
                # Result of this node should be int32
                self.assertTrue(node.meta["val"].dtype, torch.int32)
                # Argument of this node should be int8
                self.assertTrue(node.args[2].meta["val"].dtype, torch.int8)

    def test_out_dtype_op_functional(self):
        class M(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

            def forward(self, x):
                return out_dtype(
                    torch.ops.aten.mm.default, torch.int32, x, self.weight
                )

        weight = torch.randint(-128, 127, (5, 5), dtype=torch.int8)
        m = M(weight)
        x = torch.randint(-128, 127, (5, 5), dtype=torch.int8)
        ep = torch.export.export(m, (x,), strict=True)
        FileCheck().check("torch.ops.higher_order.out_dtype").check(
            "aten.mm.default"
        ).run(ep.graph_module.code)
        self.assertTrue(torch.allclose(m(x), ep.module()(x)))
        for node in ep.graph.nodes:
            if node.op == "call_function" and node.target is out_dtype:
                # Result of this node should be int32
                self.assertTrue(node.meta["val"].dtype, torch.int32)
                # Argument of this node should be int8
                self.assertTrue(node.args[2].meta["val"].dtype, torch.int8)

    def test_out_dtype_mm_numerical(self):
        class M(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

            def forward(self, x):
                return out_dtype(
                    torch.ops.aten.mm.default, torch.int32, x, self.weight
                )

        weight = torch.randint(-128, 127, (5, 5), dtype=torch.int8)
        m = M(weight)
        x = torch.randint(-128, 127, (5, 5), dtype=torch.int8)

        gm = make_fx(m)(x)

        x_casted = x.to(torch.int32)
        weight_casted = weight.to(torch.int32)
        numerical_res = torch.ops.aten.mm.default(x_casted, weight_casted)
        self.assertTrue(torch.allclose(numerical_res, gm(x)))

    def test_out_dtype_dynamo(self):
        def f(x, y):
            return out_dtype(
                torch.ops.aten.mul.Scalar, torch.int32, x, y
            )

        inp = (torch.randint(-128, 127, (5, 5), dtype=torch.int8), 3.0)

        compiled = torch.compile(f, backend="eager", fullgraph=True)
        self.assertTrue(torch.allclose(f(*inp), compiled(*inp)))

    def test_out_dtype_mul_scalar_numerical(self):
        def f(x, y):
            return out_dtype(
                torch.ops.aten.mul.Scalar, torch.int32, x, y
            )

        inp = (torch.randint(-128, 127, (5, 5), dtype=torch.int8), 3.0)

        gm = make_fx(f)(*inp)
        numerical_res = torch.ops.aten.mul.Scalar(inp[0].to(dtype=torch.int32), 3)
        self.assertTrue(torch.allclose(numerical_res, gm(*inp)))

    def test_out_dtype_non_functional(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return out_dtype(
                    torch.ops.aten.add_.Tensor, torch.int32, x, y
                )

        with self.assertRaisesRegex(ValueError, "out_dtype's first argument needs to be a functional operator"):
            _ = torch.export.export(
                M(),
                (
                    torch.randint(-128, 127, (5, 5), dtype=torch.int8),
                    torch.randint(-128, 127, (5, 5), dtype=torch.int8),
                ),
                strict=True,
            )

    def test_out_dtype_non_op_overload(self):
        def f(x, y):
            return out_dtype(
                torch.add, torch.int32, x, y
            )

        with self.assertRaisesRegex(ValueError, "out_dtype's first argument must be an OpOverload"):
            f(torch.randint(-128, 127, (5, 5), dtype=torch.int8), torch.randint(-128, 127, (5, 5), dtype=torch.int8))

    def test_out_dtype_no_autograd(self):
        def f(x, y):
            return out_dtype(
                torch.ops.aten.mm.default, torch.int32, x, y
            )

        inp = (torch.randn(5, 5, requires_grad=True), torch.randn(5, 5, requires_grad=True))
        # error is delayed
        f(*inp)

        with torch.no_grad():
            f(*inp)

        with self.assertRaisesRegex(RuntimeError, "does not require grad and does not have a grad_fn"):
            out = f(*inp)
            loss = out - torch.ones(out.shape)
            loss.backward()

    @unittest.skipIf(IS_WINDOWS, "_int_mm unavailable")
    @unittest.skipIf(TEST_WITH_ROCM, "_int_mm unavailable")
    @unittest.skipIf(not SM80OrLater, "_int_mm unavailable")
    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "cublas runtime error")
    @unittest.skipIf(_get_torch_cuda_version() >= (11, 7), "_int_mm unavailable")
    @unittest.skipIf(not TEST_CUDA, "_int_mm unavailable")
    @skipIfNoDynamoSupport
    def test_out_dtype_inductor_decomp(self) -> None:
        def func(x, w):
            return out_dtype(torch.ops.aten.mm.default, torch.int32, x, w)

        w = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device="cuda")
        x = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device="cuda")

        ref = torch._int_mm(x, w)
        test_out = func(x, w)
        func_comp = torch.compile(func, fullgraph=True, mode="max-autotune")
        test_out_c = func_comp(x, w)
        self.assertTrue(torch.allclose(ref, test_out))
        self.assertTrue(torch.allclose(ref, test_out_c))

    @unittest.skipIf(not TEST_CUDA, "cuda only")
    def test_out_dtype_inductor_decomp_trace(self) -> None:
        def func(x, w):
            return out_dtype(torch.ops.aten.mm.default, torch.int32, x, w)

        w = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device="cuda")
        x = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device="cuda")

        # Check that make_fx with inductor decomps produces _int_mm
        decomp_table = torch._inductor.decomposition.select_decomp_table()
        gm = make_fx(func, decomp_table, tracing_mode="symbolic")(x, w)
        self.assertExpectedInline(gm.code.strip(), """\
def forward(self, x_1, w_1):
    _int_mm = torch.ops.aten._int_mm.default(x_1, w_1);  x_1 = w_1 = None
    return _int_mm""")

    @unittest.skipIf(not TEST_CUDA, "cuda only")
    def test_out_dtype_int_mm_default_trace(self) -> None:
        def func(x, w):
            return out_dtype(torch.ops.aten.mm.default, torch.int32, x, w)

        w = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device="cuda")
        x = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device="cuda")

        # By default, out_dtype is preserved in the trace
        gm = make_fx(func, tracing_mode="symbolic")(x, w)
        self.assertExpectedInline(gm.code.strip(), """\
def forward(self, x_1, w_1):
    out_dtype = torch.ops.higher_order.out_dtype(torch.ops.aten.mm.default, torch.int32, x_1, w_1);  x_1 = w_1 = None
    return out_dtype""")

    def test_out_dtype_wrong_output(self) -> None:
        def multiple_out(x):
            return out_dtype(
                torch.ops.aten.topk.default, torch.int32, x, 5
            )

        inp = (torch.randn(10),)

        with self.assertRaisesRegex(ValueError, "out_dtype's can only apply to ops that return a single tensor"):
            multiple_out(*inp)

        def singleton_list_out(x):
            return out_dtype(
                torch.ops.aten.split_copy.Tensor, torch.int32, x, 10
            )

        with self.assertRaisesRegex(ValueError, "out_dtype's can only apply to ops that return a single tensor"):
            singleton_list_out(*inp)

if __name__ == '__main__':
    run_tests()

```



## High-Level Overview


This Python file contains 5 class(es) and 30 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestOutDtypeOp`, `M`, `M`, `M`, `M`

**Functions defined**: `test_out_dtype_make_fx`, `__init__`, `forward`, `test_out_dtype_op_functional`, `__init__`, `forward`, `test_out_dtype_mm_numerical`, `__init__`, `forward`, `test_out_dtype_dynamo`, `f`, `test_out_dtype_mul_scalar_numerical`, `f`, `test_out_dtype_non_functional`, `forward`, `test_out_dtype_non_op_overload`, `f`, `test_out_dtype_no_autograd`, `f`, `test_out_dtype_inductor_decomp`

**Key imports**: unittest, torch, torch._dynamo, torch._inductor, torch._inductor.decomposition, out_dtype, make_fx, skipIfNoDynamoSupport, FileCheck, SM80OrLater, _get_torch_cuda_version


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`
- `torch`
- `torch._dynamo`
- `torch._inductor`
- `torch._inductor.decomposition`
- `torch._higher_order_ops.out_dtype`: out_dtype
- `torch.fx.experimental.proxy_tensor`: make_fx
- `torch.testing._internal.common_quantization`: skipIfNoDynamoSupport
- `torch.testing`: FileCheck
- `torch.testing._internal.common_cuda`: SM80OrLater, _get_torch_cuda_version


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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/test_out_dtype_op.py
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

- **File Documentation**: `test_out_dtype_op.py_docs.md`
- **Keyword Index**: `test_out_dtype_op.py_kw.md`
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

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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
python docs/test/test_out_dtype_op.py_docs.md
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

- **File Documentation**: `test_out_dtype_op.py_docs.md_docs.md`
- **Keyword Index**: `test_out_dtype_op.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
