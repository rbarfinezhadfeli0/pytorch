# Documentation: `docs/test/onnx/test_autograd_funs.py_docs.md`

## File Metadata

- **Path**: `docs/test/onnx/test_autograd_funs.py_docs.md`
- **Size**: 10,080 bytes (9.84 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/onnx/test_autograd_funs.py`

## File Metadata

- **Path**: `test/onnx/test_autograd_funs.py`
- **Size**: 6,642 bytes (6.49 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: onnx"]

import pytorch_test_common
from onnx_test_common import run_model_test

import torch
from torch.onnx import OperatorExportTypes
from torch.testing._internal import common_utils


class TestAutogradFuns(pytorch_test_common.ExportTestCase):
    opset_version = 20
    keep_initializers_as_inputs = False
    onnx_shape_inference = True

    def test_single_output(self):
        class SingleOut(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                result = i.exp()
                result = result.log()
                ctx.save_for_backward(result)
                return result

            @staticmethod
            def backward(ctx, grad_output):
                (result,) = ctx.saved_tensors
                return grad_output * result

        class Caller(torch.nn.Module):
            def forward(self, input):
                result = input + 5
                return SingleOut.apply(result) + 3

        model = Caller()
        input = torch.ones(1)
        run_model_test(self, model, input_args=(input,))

    def test_multi_output(self):
        class MultiOut(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                result_exp = i.exp()
                result_log = result_exp.log()
                ctx.save_for_backward(result_exp, result_log)
                return result_exp, result_log

            @staticmethod
            def backward(ctx, grad_output):
                (result,) = ctx.saved_tensors
                return grad_output * result

        class Caller(torch.nn.Module):
            def forward(self, input):
                return MultiOut.apply(input)

        model = Caller()
        input = torch.ones(1, 5)
        run_model_test(self, model, input_args=(input,))

    def test_partial_output(self):
        class PartialOut(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                values, _ = torch.topk(input, 3)
                return values

        class Caller(torch.nn.Module):
            def forward(self, input):
                return PartialOut.apply(input)

        model = Caller()
        input = torch.ones(1, 5)
        run_model_test(self, model, input_args=(input,))

    def test_nested_autograd(self):
        class Child(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                result = i.log()
                result_log = result.log()
                ctx.save_for_backward(result_log)
                return result_log

            @staticmethod
            def backward(ctx, grad_output):
                (result,) = ctx.saved_tensors
                return grad_output * result

        class Parent(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                result_exp = i.exp()
                result_log = Child.apply(result_exp)
                ctx.save_for_backward(result_exp, result_log)
                return result_exp, result_log

            @staticmethod
            def backward(ctx, grad_output):
                (result,) = ctx.saved_tensors
                return grad_output * result

        class Caller(torch.nn.Module):
            def forward(self, input):
                return Parent.apply(input)

        model = Caller()
        input = torch.ones(1, 5)
        run_model_test(self, model, input_args=(input,))

    # Run export in ONNX_FALLTHROUGH mode as torch.erf() is not supported
    def test_aten_unsupported(self):
        class Erf(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                erf_out = torch.special.erf(x)
                ctx.save_for_backward(erf_out)
                return erf_out

            @staticmethod
            def backward(ctx, grad_output):
                result = ctx.saved_tensors
                return torch.special.erfinv(result), None

        class Caller(torch.nn.Module):
            def forward(self, input):
                return Erf.apply(input)

        model = Caller()
        input = torch.ones(1, 5)

        # Test ONNX_FALLTHROUGH_MODE
        graph, _, _ = torch.onnx.utils._model_to_graph(
            model,
            (input,),
            operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
        )
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "prim::PythonOp")

        # Test ATEN_FALLBACK_MODE
        graph, _, _ = torch.onnx.utils._model_to_graph(
            model,
            (input,),
            operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
        )
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "aten::ATen")

    def test_inline_and_symbolic(self):
        class Exp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                ctx.save_for_backward(input)
                return i.exp()

            @staticmethod
            def symbolic(g, input):
                return g.op("Exp", input)

        class LogLog(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                ctx.save_for_backward(input)
                return i.log().log()

        class Caller(torch.nn.Module):
            def forward(self, input):
                exp_result = Exp.apply(input)
                return LogLog.apply(exp_result)

        model = Caller()
        input = torch.ones(1)
        run_model_test(self, model, input_args=(input,))

    def test_inline_with_scoped_tracing(self):
        class Exp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                ctx.save_for_backward(input)
                return i.exp()

            @staticmethod
            def symbolic(g, input):
                return g.op("Exp", input)

        class LogLog(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                ctx.save_for_backward(input)
                return i.log().log()

        class Caller(torch.nn.Module):
            def forward(self, input):
                exp_result = Exp.apply(input)
                return LogLog.apply(exp_result)

        model = Caller()
        input = torch.ones(1)

        torch.jit._trace._trace_module_map = {
            _m: torch.typename(type(_m)) for _m in model.modules()
        }
        run_model_test(self, model, input_args=(input,))
        torch.jit._trace._trace_module_map = None


if __name__ == "__main__":
    common_utils.run_tests()

```



## High-Level Overview


This Python file contains 18 class(es) and 31 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestAutogradFuns`, `SingleOut`, `Caller`, `MultiOut`, `Caller`, `PartialOut`, `Caller`, `Child`, `Parent`, `Caller`, `Erf`, `Caller`, `Exp`, `LogLog`, `Caller`, `Exp`, `LogLog`, `Caller`

**Functions defined**: `test_single_output`, `forward`, `backward`, `forward`, `test_multi_output`, `forward`, `backward`, `forward`, `test_partial_output`, `forward`, `forward`, `test_nested_autograd`, `forward`, `backward`, `forward`, `backward`, `forward`, `test_aten_unsupported`, `forward`, `backward`

**Key imports**: pytorch_test_common, run_model_test, torch, OperatorExportTypes, common_utils


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/onnx`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `pytorch_test_common`
- `onnx_test_common`: run_model_test
- `torch`
- `torch.onnx`: OperatorExportTypes
- `torch.testing._internal`: common_utils


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


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
python test/onnx/test_autograd_funs.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/onnx`):

- [`test_lazy_import.py_docs.md`](./test_lazy_import.py_docs.md)
- [`onnx_test_common.py_docs.md`](./onnx_test_common.py_docs.md)
- [`pytorch_test_common.py_docs.md`](./pytorch_test_common.py_docs.md)
- [`test_pytorch_onnx_shape_inference.py_docs.md`](./test_pytorch_onnx_shape_inference.py_docs.md)
- [`test_onnxscript_no_runtime.py_docs.md`](./test_onnxscript_no_runtime.py_docs.md)
- [`test_models_onnxruntime.py_docs.md`](./test_models_onnxruntime.py_docs.md)
- [`test_custom_ops.py_docs.md`](./test_custom_ops.py_docs.md)
- [`test_models.py_docs.md`](./test_models.py_docs.md)
- [`test_onnxscript_runtime.py_docs.md`](./test_onnxscript_runtime.py_docs.md)
- [`test_pytorch_onnx_onnxruntime_cuda.py_docs.md`](./test_pytorch_onnx_onnxruntime_cuda.py_docs.md)


## Cross-References

- **File Documentation**: `test_autograd_funs.py_docs.md`
- **Keyword Index**: `test_autograd_funs.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/onnx`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/onnx`, which is part of the **testing infrastructure**.



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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/onnx/test_autograd_funs.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/onnx`):

- [`test_pytorch_onnx_onnxruntime.py_docs.md_docs.md`](./test_pytorch_onnx_onnxruntime.py_docs.md_docs.md)
- [`test_models_onnxruntime.py_docs.md_docs.md`](./test_models_onnxruntime.py_docs.md_docs.md)
- [`test_utility_funs.py_kw.md_docs.md`](./test_utility_funs.py_kw.md_docs.md)
- [`test_autograd_funs.py_kw.md_docs.md`](./test_autograd_funs.py_kw.md_docs.md)
- [`test_fx_type_promotion.py_docs.md_docs.md`](./test_fx_type_promotion.py_docs.md_docs.md)
- [`test_onnx_opset.py_docs.md_docs.md`](./test_onnx_opset.py_docs.md_docs.md)
- [`verify.py_docs.md_docs.md`](./verify.py_docs.md_docs.md)
- [`pytorch_test_common.py_kw.md_docs.md`](./pytorch_test_common.py_kw.md_docs.md)
- [`test_models_quantized_onnxruntime.py_kw.md_docs.md`](./test_models_quantized_onnxruntime.py_kw.md_docs.md)
- [`test_models_onnxruntime.py_kw.md_docs.md`](./test_models_onnxruntime.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_autograd_funs.py_docs.md_docs.md`
- **Keyword Index**: `test_autograd_funs.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
