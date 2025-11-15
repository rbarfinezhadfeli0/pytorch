# Documentation: `test/onnx/test_onnxscript_no_runtime.py`

## File Metadata

- **Path**: `test/onnx/test_onnxscript_no_runtime.py`
- **Size**: 6,410 bytes (6.26 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: onnx"]

"""Test the support on onnxscript in PyTorch-ONNX converter."""

import io

import onnx

import onnxscript
from onnxscript.onnx_types import FLOAT

import torch
from torch.onnx._internal.torchscript_exporter import jit_utils
from torch.testing._internal import common_utils


class TestONNXScriptExport(common_utils.TestCase):
    # opset version is
    # 1. local function is supported after opset 15
    # 2. onnx-script requires users to determine opset in local function
    opset_version = 15

    def test_onnxscript_registration_with_multiple_models(self):
        from onnxscript.onnx_opset import opset15 as op

        # 1. Register Selu onnxscript function as custom Op
        custom_opset = onnxscript.values.Opset(domain="onnx-script", version=1)

        @onnxscript.script(custom_opset)
        def Selu(X):
            # default value is not supported by onnxscript
            alpha = 1.67326  # auto wrapped as Constants
            gamma = 1.0507
            alphaX = op.CastLike(alpha, X)
            gammaX = op.CastLike(gamma, X)
            neg = gammaX * (alphaX * op.Exp(X) - alphaX)
            pos = gammaX * X
            zero = op.CastLike(0, X)
            return op.Where(X <= zero, neg, pos)

        def custom_selu(g: jit_utils.GraphContext, X):
            return g.onnxscript_op(Selu, X).setType(X.type())

        torch.onnx.register_custom_op_symbolic(
            symbolic_name="aten::selu",
            symbolic_fn=custom_selu,
            opset_version=self.opset_version,
        )

        # 2. Register layer_norm onnxscript function as custom Op
        @onnxscript.script(custom_opset)
        def layer_norm(
            X, axes: list[int], weight: FLOAT[...], bias: FLOAT[...], eps: float
        ):
            mean = op.ReduceMean(X, axes=axes)
            D = X - mean  # op.Sub(X, mean)
            DD = D * D  # op.Mul(D, D)
            var = op.ReduceMean(DD, axes=axes)
            vareps = var + eps  # op.Add(var, eps)
            stddev = op.Sqrt(vareps)
            invstddev = op.Reciprocal(stddev)
            normalized = D * invstddev  # op.Mul(D, invstddev)
            normalizedw = op.CastLike(
                normalized, weight
            )  # Type issue if missing this Op
            normalizedscaled = normalizedw * weight  # op.Mul(normalized, weight)
            return normalizedscaled + bias

        @torch.onnx.symbolic_helper.parse_args("v", "is", "v", "v", "f", "none")
        def custom_layer_norm(
            g, input, normalized_shape, weight, bias, eps, cudnn_enable
        ):
            # comprehension is not supported by onnxscript
            axes = [-i for i in range(len(normalized_shape), 0, -1)]
            return g.onnxscript_op(
                layer_norm, input, weight, bias, axes_i=axes, eps_f=eps
            ).setType(input.type())

        torch.onnx.register_custom_op_symbolic(
            symbolic_name="aten::layer_norm",
            symbolic_fn=custom_layer_norm,
            opset_version=self.opset_version,
        )

        # 3. export two models
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        model_selu = torch.nn.SELU()
        selu_onnx = io.BytesIO()
        torch.onnx.export(
            model_selu, x, selu_onnx, opset_version=self.opset_version, dynamo=False
        )

        N, C = 3, 4
        y = torch.randn(N, C)
        model_layer_norm = torch.nn.LayerNorm(C)
        layer_norm_onnx = io.BytesIO()
        torch.onnx.export(
            model_layer_norm,
            y,
            layer_norm_onnx,
            opset_version=self.opset_version,
            dynamo=False,
        )

        # 4. test on models
        selu_proto = onnx.load(io.BytesIO(selu_onnx.getvalue()))
        layer_norm_proto = onnx.load(io.BytesIO(layer_norm_onnx.getvalue()))

        self.assertEqual(len(selu_proto.functions), 1)
        self.assertEqual(len(layer_norm_proto.functions), 1)
        self.assertEqual(selu_proto.functions[0].name, "Selu")
        self.assertEqual(layer_norm_proto.functions[0].name, "layer_norm")

    def test_loop_registration(self):
        # Control flow is tested for _find_onnxscript_op function in torch/onnx/utils.py,
        # which has recursive logic to go through every nodes with subgraph in model proto
        class NestedLoopsModel(torch.jit.ScriptModule):
            def __init__(self) -> None:
                super().__init__()
                self.selu = torch.nn.SELU()

            @torch.jit.script_method
            def forward(self, x):
                y = x
                for i in range(x.size(3)):
                    if i == 0:
                        y = self.selu(x)
                    else:
                        y += i
                return y

        model = NestedLoopsModel()
        inputs = torch.zeros(1, 2, 3, 4)

        from onnxscript.onnx_opset import opset15 as op

        custom_opset = onnxscript.values.Opset(domain="onnx-script", version=2)

        @onnxscript.script(custom_opset)
        def Selu(X):
            alpha = 1.6732632423543772848170429916717
            gamma = 1.0507009873554804934193349852946
            alphaX = op.CastLike(alpha, X)
            gammaX = op.CastLike(gamma, X)
            neg = gammaX * (alphaX * op.Exp(X) - alphaX)
            pos = gammaX * X
            zero = op.CastLike(0, X)
            return op.Where(X <= zero, neg, pos)

        def custom_selu(g, X):
            # domain of the Op should be aligned with onnx-script
            # setType API is required for custom Op to support
            # torchscript shape type inference
            print("custom_selu is used!")
            return g.onnxscript_op(Selu, X).setType(X.type())

        torch.onnx.register_custom_op_symbolic(
            symbolic_name="aten::selu",
            symbolic_fn=custom_selu,
            opset_version=15,
        )

        saved_model = io.BytesIO()
        torch.onnx.export(
            torch.jit.script(model),
            inputs,
            f=saved_model,
            opset_version=15,
            dynamo=False,
        )
        loop_selu_proto = onnx.load(io.BytesIO(saved_model.getvalue()))
        self.assertEqual(len(loop_selu_proto.functions), 1)


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )

```



## High-Level Overview

"""Test the support on onnxscript in PyTorch-ONNX converter."""import ioimport onnximport onnxscriptfrom onnxscript.onnx_types import FLOATimport torchfrom torch.onnx._internal.torchscript_exporter import jit_utilsfrom torch.testing._internal import common_utilsclass TestONNXScriptExport(common_utils.TestCase):    # opset version is    # 1. local function is supported after opset 15    # 2. onnx-script requires users to determine opset in local function    opset_version = 15    def test_onnxscript_registration_with_multiple_models(self):        from onnxscript.onnx_opset import opset15 as op        # 1. Register Selu onnxscript function as custom Op        custom_opset = onnxscript.values.Opset(domain="onnx-script", version=1)        @onnxscript.script(custom_opset)        def Selu(X):            # default value is not supported by onnxscript            alpha = 1.67326  # auto wrapped as Constants            gamma = 1.0507            alphaX = op.CastLike(alpha, X)            gammaX = op.CastLike(gamma, X)            neg = gammaX * (alphaX * op.Exp(X) - alphaX)            pos = gammaX * X            zero = op.CastLike(0, X)            return op.Where(X <= zero, neg, pos)        def custom_selu(g: jit_utils.GraphContext, X):            return g.onnxscript_op(Selu, X).setType(X.type())        torch.onnx.register_custom_op_symbolic(            symbolic_name="aten::selu",            symbolic_fn=custom_selu,            opset_version=self.opset_version,        )        # 2. Register layer_norm onnxscript function as custom Op

This Python file contains 2 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestONNXScriptExport`, `NestedLoopsModel`

**Functions defined**: `test_onnxscript_registration_with_multiple_models`, `Selu`, `custom_selu`, `layer_norm`, `custom_layer_norm`, `test_loop_registration`, `__init__`, `forward`, `Selu`, `custom_selu`

**Key imports**: io, onnx, onnxscript, FLOAT, torch, jit_utils, common_utils, opset15 as op, opset15 as op


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/onnx`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `io`
- `onnx`
- `onnxscript`
- `onnxscript.onnx_types`: FLOAT
- `torch`
- `torch.onnx._internal.torchscript_exporter`: jit_utils
- `torch.testing._internal`: common_utils
- `onnxscript.onnx_opset`: opset15 as op


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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
python test/onnx/test_onnxscript_no_runtime.py
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
- [`test_models_onnxruntime.py_docs.md`](./test_models_onnxruntime.py_docs.md)
- [`test_custom_ops.py_docs.md`](./test_custom_ops.py_docs.md)
- [`test_models.py_docs.md`](./test_models.py_docs.md)
- [`test_onnxscript_runtime.py_docs.md`](./test_onnxscript_runtime.py_docs.md)
- [`test_pytorch_onnx_onnxruntime_cuda.py_docs.md`](./test_pytorch_onnx_onnxruntime_cuda.py_docs.md)


## Cross-References

- **File Documentation**: `test_onnxscript_no_runtime.py_docs.md`
- **Keyword Index**: `test_onnxscript_no_runtime.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
