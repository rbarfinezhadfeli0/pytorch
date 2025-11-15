# Documentation: `test/onnx/test_pytorch_onnx_onnxruntime_cuda.py`

## File Metadata

- **Path**: `test/onnx/test_pytorch_onnx_onnxruntime_cuda.py`
- **Size**: 4,921 bytes (4.81 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: onnx"]

import unittest

import onnx_test_common
import onnxruntime  # noqa: F401
import parameterized
from onnx_test_common import MAX_ONNX_OPSET_VERSION, MIN_ONNX_OPSET_VERSION
from pytorch_test_common import (
    skipIfNoBFloat16Cuda,
    skipIfNoCuda,
    skipIfUnsupportedMinOpsetVersion,
    skipScriptTest,
)
from test_pytorch_onnx_onnxruntime import _parameterized_class_attrs_and_values

import torch
from torch.cuda.amp import autocast
from torch.testing._internal import common_utils


@parameterized.parameterized_class(
    **_parameterized_class_attrs_and_values(
        MIN_ONNX_OPSET_VERSION, MAX_ONNX_OPSET_VERSION
    ),
    class_name_func=onnx_test_common.parameterize_class_name,
)
class TestONNXRuntime_cuda(onnx_test_common._TestONNXRuntime):
    @skipIfUnsupportedMinOpsetVersion(9)
    @skipIfNoCuda
    def test_gelu_fp16(self):
        class GeluModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.gelu(x)

        x = torch.randn(
            2,
            4,
            5,
            6,
            requires_grad=True,
            dtype=torch.float16,
            device=torch.device("cuda"),
        )
        self.run_test(GeluModel(), x, rtol=1e-3, atol=1e-5)

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipIfNoCuda
    @skipScriptTest()
    def test_layer_norm_fp16(self):
        class LayerNormModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer_norm = torch.nn.LayerNorm([10, 10])

            @autocast()
            def forward(self, x):
                return self.layer_norm(x)

        x = torch.randn(
            20,
            5,
            10,
            10,
            requires_grad=True,
            dtype=torch.float16,
            device=torch.device("cuda"),
        )
        self.run_test(LayerNormModel().cuda(), x, rtol=1e-3, atol=1e-5)

    @skipIfUnsupportedMinOpsetVersion(12)
    @skipIfNoCuda
    @skipScriptTest()
    def test_softmaxCrossEntropy_fusion_fp16(self):
        class FusionModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.loss = torch.nn.NLLLoss(reduction="none")
                self.m = torch.nn.LogSoftmax(dim=1)

            @autocast()
            def forward(self, input, target):
                output = self.loss(self.m(2 * input), target)
                return output

        N, C = 5, 4
        input = torch.randn(N, 16, dtype=torch.float16, device=torch.device("cuda"))
        target = torch.empty(N, dtype=torch.long, device=torch.device("cuda")).random_(
            0, C
        )

        # using test data containing default ignore_index=-100
        target[target == 1] = -100
        self.run_test(FusionModel(), (input, target))

    @skipIfNoCuda
    @skipScriptTest()
    def test_apex_o2(self):
        class LinearModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(3, 5)

            def forward(self, x):
                return self.linear(x)

        try:
            from apex import amp
        except Exception as e:
            raise unittest.SkipTest("Apex is not available") from e
        input = torch.randn(3, 3, device=torch.device("cuda"))
        model = amp.initialize(LinearModel(), opt_level="O2")
        self.run_test(model, input)

    # ONNX supports bfloat16 for opsets >= 13
    # Add, Sub and Mul ops don't support bfloat16 cpu in onnxruntime.
    @skipIfUnsupportedMinOpsetVersion(13)
    @skipIfNoBFloat16Cuda
    def test_arithmetic_bfp16(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                y = torch.ones(3, 4, dtype=torch.bfloat16, device=torch.device("cuda"))
                x = x.type_as(y)
                return torch.mul(torch.add(x, y), torch.sub(x, y)).to(
                    dtype=torch.float16
                )

        x = torch.ones(
            3, 4, requires_grad=True, dtype=torch.float16, device=torch.device("cuda")
        )
        self.run_test(MyModule(), x, rtol=1e-3, atol=1e-5)

    @skipIfNoCuda
    def test_deduplicate_initializers_diff_devices(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = torch.nn.Parameter(
                    torch.ones(2, 3, device=torch.device("cpu"))
                )
                self.b = torch.nn.Parameter(torch.ones(3, device=torch.device("cuda")))

            def forward(self, x, y):
                return torch.matmul(self.w, x), y + self.b

        x = torch.randn(3, 3, device=torch.device("cpu"))
        y = torch.randn(3, 3, device=torch.device("cuda"))
        self.run_test(Model(), (x, y))


if __name__ == "__main__":
    common_utils.run_tests()

```



## High-Level Overview


This Python file contains 7 class(es) and 16 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestONNXRuntime_cuda`, `GeluModel`, `LayerNormModel`, `FusionModel`, `LinearModel`, `MyModule`, `Model`

**Functions defined**: `test_gelu_fp16`, `forward`, `test_layer_norm_fp16`, `__init__`, `forward`, `test_softmaxCrossEntropy_fusion_fp16`, `__init__`, `forward`, `test_apex_o2`, `__init__`, `forward`, `test_arithmetic_bfp16`, `forward`, `test_deduplicate_initializers_diff_devices`, `__init__`, `forward`

**Key imports**: unittest, onnx_test_common, onnxruntime  , parameterized, MAX_ONNX_OPSET_VERSION, MIN_ONNX_OPSET_VERSION, _parameterized_class_attrs_and_values, torch, autocast, common_utils, amp


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/onnx`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`
- `onnx_test_common`
- `onnxruntime  `
- `parameterized`
- `test_pytorch_onnx_onnxruntime`: _parameterized_class_attrs_and_values
- `torch`
- `torch.cuda.amp`: autocast
- `torch.testing._internal`: common_utils
- `apex`: amp


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/onnx/test_pytorch_onnx_onnxruntime_cuda.py
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


## Cross-References

- **File Documentation**: `test_pytorch_onnx_onnxruntime_cuda.py_docs.md`
- **Keyword Index**: `test_pytorch_onnx_onnxruntime_cuda.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
