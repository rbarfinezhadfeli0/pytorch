# Documentation: `docs/test/test_tensorexpr_pybind.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_tensorexpr_pybind.py_docs.md`
- **Size**: 19,766 bytes (19.30 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_tensorexpr_pybind.py`

## File Metadata

- **Path**: `test/test_tensorexpr_pybind.py`
- **Size**: 16,420 bytes (16.04 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["NNC"]

import torch
import numpy as np
import torch._C._te as te

from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.jit_utils import JitTestCase
import unittest

LLVM_ENABLED = torch._C._llvm_enabled()


def construct_adder(n: int, dtype=torch.float32):
    A = te.BufHandle("A", [n], dtype)
    B = te.BufHandle("B", [n], dtype)

    def compute(i):
        return A.load([i]) + B.load([i])

    C = te.Compute("C", [n], compute)

    loopnest = te.LoopNest([C])
    loopnest.prepare_for_codegen()
    stmt = te.simplify(loopnest.root_stmt())

    return te.construct_codegen("ir_eval", stmt, [A, B, C])


class TestTensorExprPyBind(JitTestCase):
    def test_simple_sum(self):
        n = 32
        cg = construct_adder(n)

        tA = torch.randn(n)
        tB = torch.randn(n)
        tC = torch.empty(n)
        cg.call([tA, tB, tC])
        torch.testing.assert_close(tA + tB, tC)

    def test_call_raw(self):
        n = 16
        cg = construct_adder(n, dtype=torch.float64)

        tA = torch.randn(n, dtype=torch.float64)
        tB = torch.randn(n, dtype=torch.float64)
        tC = torch.empty(n, dtype=torch.float64)
        cg.call_raw([tA.data_ptr(), tB.data_ptr(), tC.data_ptr()])
        torch.testing.assert_close(tA + tB, tC)

    def test_external_calls(self):
        dtype = torch.float32

        A = te.BufHandle("A", [1, 4], dtype)
        B = te.BufHandle("B", [4, 1], dtype)
        C = te.BufHandle("C", [1, 1], dtype)

        s = te.ExternalCall(C, "nnc_aten_matmul", [A, B], [])

        loopnest = te.LoopNest(s, [C])
        loopnest.prepare_for_codegen()
        codegen = te.construct_codegen("ir_eval", s, [A, B, C])

        tA = torch.ones(1, 4)
        tB = torch.ones(4, 1)
        tC = torch.empty(1, 1)
        codegen.call([tA, tB, tC])
        torch.testing.assert_close(torch.matmul(tA, tB), tC)

    def test_dynamic_shape(self):
        dN = te.VarHandle(torch.int32)
        A = te.BufHandle([dN], torch.float64)
        B = te.BufHandle([dN], torch.float64)

        def compute(i):
            return A.load(i) - B.load(i)

        C = te.Compute("C", [dN], compute)

        loopnest = te.LoopNest([C])
        loopnest.prepare_for_codegen()

        cg = te.construct_codegen("ir_eval", loopnest.simplify(), [A, B, C, dN])

        def test_with_shape(n):
            tA = torch.randn(n, dtype=torch.double)
            tB = torch.randn(n, dtype=torch.double)
            tC = torch.empty(n, dtype=torch.double)
            cg.call([tA, tB, tC, n])
            torch.testing.assert_close(tA - tB, tC)

        test_with_shape(8)
        test_with_shape(31)

    def test_dynamic_shape_2d(self):
        dN = te.VarHandle(torch.int32)
        dM = te.VarHandle(torch.int32)
        A = te.BufHandle([dN, dM], torch.float64)
        B = te.BufHandle([dN, dM], torch.float64)

        def compute(i, j):
            return A.load([i, j]) - B.load([i, j])

        C = te.Compute("C", [dN, dM], compute)

        loopnest = te.LoopNest([C])
        loopnest.prepare_for_codegen()

        cg = te.construct_codegen("ir_eval", loopnest.simplify(), [A, B, C, dN, dM])

        def test_with_shape(n, m):
            tA = torch.randn(n, m, dtype=torch.double)
            tB = torch.randn(n, m, dtype=torch.double)
            tC = torch.empty(n, m, dtype=torch.double)
            cg.call([tA, tB, tC, n, m])
            torch.testing.assert_close(tA - tB, tC)

        test_with_shape(2, 4)
        test_with_shape(5, 3)

    def test_dtype_error(self):
        te.BufHandle("a", [1], torch.float32)  # ok
        self.assertRaises(TypeError, lambda: te.BufHandle("a", [1], "float55"))

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_tensor_inputs(self):
        def f(a, b, c):
            return a + b + c

        device, size = "cpu", (4, 4)
        x = torch.rand(size, device=device)
        y = torch.rand(size, device=device)
        z = torch.rand(size, device=device)

        graph_str = """
graph(%a.1 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cpu),
      %b.1 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cpu),
      %c.1 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cpu)):
  %6 : int = prim::Constant[value=1]()
  %7 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cpu) = aten::add(%a.1, %b.1, %6)
  %3 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cpu) = aten::add(%7, %c.1, %6)
  return (%3)
        """
        graph = torch._C.parse_ir(graph_str)

        kernel = te.TensorExprKernel(graph)
        res1 = kernel.run((x, y, z))
        res2 = kernel.fallback((x, y, z))
        correct = f(x, y, z)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_scalar_inputs(self):
        def f(a, b, c):
            return a + b + c

        x = torch.tensor(0.1, dtype=torch.float, device="cpu")
        y = torch.tensor(0.6, dtype=torch.float, device="cpu")
        z = torch.tensor(0.7, dtype=torch.float, device="cpu")

        graph_str = """
graph(%a.1 : Float(requires_grad=0, device=cpu),
      %b.1 : Float(requires_grad=0, device=cpu),
      %c.1 : Float(requires_grad=0, device=cpu)):
  %3 : int = prim::Constant[value=1]()
  %6 : Float(requires_grad=0, device=cpu) = aten::add(%a.1, %b.1, %3)
  %9 : Float(requires_grad=0, device=cpu) = aten::add(%6, %c.1, %3)
  return (%9)
        """
        graph = torch._C.parse_ir(graph_str)

        kernel = te.TensorExprKernel(graph)
        res1 = kernel.run((x, y, z))
        res2 = kernel.fallback((x, y, z))
        correct = f(x, y, z)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_shape_prop(self):
        device, size = "cpu", (4, 4)
        x = torch.rand(size, device=device)
        y = torch.rand(size, device=device)

        graph_str = """
graph(%a : Tensor, %b : Tensor):
  %c : Tensor = aten::mul(%a, %b)
  return (%c)
        """
        graph = torch._C.parse_ir(graph_str)

        exception_thrown = False
        try:
            kernel = te.TensorExprKernel(graph)
        except RuntimeError:
            # Graph doesn't have shape info for inputs => compilation should
            # fail
            exception_thrown = True
        assert exception_thrown

        # Inject shape info and try compiling again
        example_inputs = [torch.rand(4, 4), torch.rand(4, 4)]
        torch._C._te.annotate_input_shapes(graph, example_inputs)
        torch._C._jit_pass_propagate_shapes_on_graph(graph)

        # Now compilation should pass
        kernel = te.TensorExprKernel(graph)

        res = kernel.run((x, y))
        correct = torch.mul(x, y)
        np.testing.assert_allclose(res.numpy(), correct.numpy(), atol=1e-5)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_shape_prop_module(self):
        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                return x * x + y

        graph = torch.jit.script(TestModule()).graph

        # Try compiling the graph as-is. It should fail because it doesn't have
        # shape info.
        exception_thrown = False
        try:
            kernel = te.TensorExprKernel(graph)
        except RuntimeError:
            exception_thrown = True
        assert exception_thrown

        # Try injecting shape info for graph inputs
        example_inputs = [torch.rand(4, 4), torch.rand(4, 4)]

        exception_thrown = False
        try:
            torch._C._te.annotate_input_shapes(graph, example_inputs)
        except RuntimeError:
            # Graph has a 'self' argument for which we can't set shapes
            exception_thrown = True
        assert exception_thrown

        # Remove 'self' argument and try annotating shapes one more time
        torch._C._te.remove_unused_self_argument(graph)

        # Inject shape info and try compiling again
        torch._C._te.annotate_input_shapes(graph, example_inputs)
        torch._C._jit_pass_propagate_shapes_on_graph(graph)

        # Now compilation should pass
        kernel = te.TensorExprKernel(graph)

        device, size = "cpu", (4, 4)
        x = torch.rand(size, device=device)
        y = torch.rand(size, device=device)

        res = kernel.run((x, y))
        correct = TestModule().forward(x, y)
        np.testing.assert_allclose(res.numpy(), correct.numpy(), atol=1e-5)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_t(self):
        def f(a):
            return a.t()

        device, size = "cpu", (3, 4)
        x = torch.rand(size, device=device)

        graph_str = """
graph(%a.1 : Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu)):
  %3 : Float(4, 3, strides=[4, 1], requires_grad=0, device=cpu) = aten::t(%a.1)
  return (%3)
        """
        graph = torch._C.parse_ir(graph_str)

        kernel = te.TensorExprKernel(graph)
        res1 = kernel.run((x,))
        res2 = kernel.fallback((x,))
        correct = f(x)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_transpose(self):
        def f(a):
            return a.transpose(-1, -2)

        device, size = "cpu", (3, 4)
        x = torch.rand(size, device=device)

        graph_str = """
graph(%a.1 : Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu)):
  %2 : int = prim::Constant[value=-1]()
  %3 : int = prim::Constant[value=-2]()
  %4 : Float(4, 3, strides=[4, 1], requires_grad=0, device=cpu) = aten::transpose(%a.1, %2, %3)
  return (%4)
        """
        graph = torch._C.parse_ir(graph_str)

        kernel = te.TensorExprKernel(graph)
        res1 = kernel.run((x,))
        res2 = kernel.fallback((x,))
        correct = f(x)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_permute(self):
        def f(a):
            return a.permute([2, 1, 0])

        device, size = "cpu", (3, 4, 5)
        x = torch.rand(size, device=device)

        graph_str = """
graph(%a.1 : Float(3, 4, 5, strides=[20, 5, 1], requires_grad=0, device=cpu)):
  %1 : int = prim::Constant[value=2]()
  %2 : int = prim::Constant[value=1]()
  %3 : int = prim::Constant[value=0]()
  %4 : int[] = prim::ListConstruct(%1, %2, %3)
  %5 : Float(5, 4, 3, strides=[12, 3, 1], requires_grad=0, device=cpu) = aten::permute(%a.1, %4)
  return (%5)
        """
        graph = torch._C.parse_ir(graph_str)

        kernel = te.TensorExprKernel(graph)
        res1 = kernel.run((x,))
        res2 = kernel.fallback((x,))
        correct = f(x)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_custom_lowering(self):
        def f(a):
            return a.nan_to_num()

        device = "cpu"
        x = torch.ones((2, 2), device=device)
        x[0, 0] = x[1, 1] = torch.nan
        graph_str = """
graph(%x : Float(2, 2, strides=[2, 1], requires_grad=0, device=cpu)):
    %none : NoneType = prim::Constant()
    %y : Float(2, 2, strides=[2, 1], requires_grad=0, device=cpu) = aten::nan_to_num(%x, %none, %none, %none)
    return (%y)
        """
        graph = torch._C.parse_ir(graph_str)

        def my_custom_lowering(inputs, out_shape, out_stride, out_type, device):
            def compute(idxs):
                load = inputs[0].as_buf().load(idxs)
                return te.ifThenElse(
                    te.ExprHandle.isnan(load), te.ExprHandle.float(0.0), load
                )

            return te.Compute2("custom_nan_to_num", out_shape, compute)

        kernel = te.TensorExprKernel(graph, {"aten::nan_to_num": my_custom_lowering})
        res1 = kernel.run((x,))
        res2 = kernel.fallback((x,))
        correct = f(x)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_expand(self):
        def f(a):
            return a.expand((2, 3, 4))

        device = "cpu"
        x = torch.rand((1, 3, 1), device=device)
        graph_str = """
graph(%a : Float(1, 3, 1, strides=[3, 1, 1], requires_grad=0, device=cpu)):
  %1 : int = prim::Constant[value=2]()
  %2 : int = prim::Constant[value=3]()
  %3 : int = prim::Constant[value=4]()
  %4 : int[] = prim::ListConstruct(%1, %2, %3)
  %5 : bool = prim::Constant[value=0]()
  %6 : Float(2, 3, 4, strides=[12, 4, 0], requires_grad=0, device=cpu) = aten::expand(%a, %4, %5)
  return (%6)
        """
        graph = torch._C.parse_ir(graph_str)

        kernel = te.TensorExprKernel(graph)
        res1 = kernel.run((x,))
        res2 = kernel.fallback((x,))
        correct = f(x)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_alloc_in_loop(self):
        a, tmp, b = (
            te.BufHandle(name, [1], torch.float32) for name in ["a", "tmp", "b"]
        )
        body = te.Block([tmp.store([0], a.load([0])), b.store([0], tmp.load([0]))])
        for _ in range(4):
            i = te.VarHandle("i", torch.int32)
            body = te.For.make(i, 0, 100, body)
        nest = te.LoopNest(body, [b])
        nest.prepare_for_codegen()
        f = te.construct_codegen("llvm", nest.simplify(), [a, b])
        ta, tb = (torch.ones(1) for _ in range(2))
        f.call([ta.data_ptr(), tb.data_ptr()])


class TestExprHandlePyBind(JitTestCase):
    def test_unary_ops(self):
        unary_operators = {
            torch.sin: torch._C._te.sin,
            torch.cos: torch._C._te.cos,
            torch.tan: torch._C._te.tan,
            torch.asin: torch._C._te.asin,
            torch.acos: torch._C._te.acos,
            torch.atan: torch._C._te.atan,
            torch.sinh: torch._C._te.sinh,
            torch.cosh: torch._C._te.cosh,
            torch.tanh: torch._C._te.tanh,
            torch.sigmoid: torch._C._te.sigmoid,
            torch.exp: torch._C._te.exp,
            torch.expm1: torch._C._te.expm1,
            torch.abs: torch._C._te.abs,
            torch.log: torch._C._te.log,
            torch.log2: torch._C._te.log2,
            torch.log10: torch._C._te.log10,
            torch.log1p: torch._C._te.log1p,
            torch.erf: torch._C._te.erf,
            torch.erfc: torch._C._te.erfc,
            torch.sqrt: torch._C._te.sqrt,
            torch.rsqrt: torch._C._te.rsqrt,
            torch.ceil: torch._C._te.ceil,
            torch.floor: torch._C._te.floor,
            torch.round: torch._C._te.round,
            torch.trunc: torch._C._te.trunc,
            torch.lgamma: torch._C._te.lgamma,
            torch.frac: torch._C._te.frac,
        }

        def construct_te_fn(op, n: int, dtype=torch.float32):
            A = torch._C._te.BufHandle("A", [n], dtype)

            def compute(i):
                return op(A.load([i]))

            C = te.Compute("C", [n], compute)

            loopnest = te.LoopNest([C])
            loopnest.prepare_for_codegen()
            stmt = te.simplify(loopnest.root_stmt())

            return te.construct_codegen("ir_eval", stmt, [A, C])

        n = 10
        a = torch.rand(n)
        for torch_op, te_op in unary_operators.items():
            ref = torch_op(a)

            te_fn = construct_te_fn(te_op, n, torch.float32)
            res = torch.empty(n)
            te_fn.call([a, res])
            assert torch.allclose(ref, res, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 3 class(es) and 35 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestTensorExprPyBind`, `TestModule`, `TestExprHandlePyBind`

**Functions defined**: `construct_adder`, `compute`, `test_simple_sum`, `test_call_raw`, `test_external_calls`, `test_dynamic_shape`, `compute`, `test_with_shape`, `test_dynamic_shape_2d`, `compute`, `test_with_shape`, `test_dtype_error`, `test_kernel_with_tensor_inputs`, `f`, `test_kernel_with_scalar_inputs`, `f`, `test_kernel_shape_prop`, `test_kernel_shape_prop_module`, `forward`, `test_kernel_with_t`

**Key imports**: torch, numpy as np, torch._C._te as te, run_tests, JitTestCase, unittest


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `numpy as np`
- `torch._C._te as te`
- `torch.testing._internal.common_utils`: run_tests
- `torch.testing._internal.jit_utils`: JitTestCase
- `unittest`


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

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
python test/test_tensorexpr_pybind.py
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

- **File Documentation**: `test_tensorexpr_pybind.py_docs.md`
- **Keyword Index**: `test_tensorexpr_pybind.py_kw.md`
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

- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

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
python docs/test/test_tensorexpr_pybind.py_docs.md
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

- **File Documentation**: `test_tensorexpr_pybind.py_docs.md_docs.md`
- **Keyword Index**: `test_tensorexpr_pybind.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
