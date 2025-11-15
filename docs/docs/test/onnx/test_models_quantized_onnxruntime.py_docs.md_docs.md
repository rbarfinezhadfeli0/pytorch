# Documentation: `docs/test/onnx/test_models_quantized_onnxruntime.py_docs.md`

## File Metadata

- **Path**: `docs/test/onnx/test_models_quantized_onnxruntime.py_docs.md`
- **Size**: 6,631 bytes (6.48 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/onnx/test_models_quantized_onnxruntime.py`

## File Metadata

- **Path**: `test/onnx/test_models_quantized_onnxruntime.py`
- **Size**: 3,429 bytes (3.35 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: onnx"]

import os
import unittest

import onnx_test_common
import parameterized
import PIL
import torchvision

import torch
from torch import nn
from torch.testing._internal import common_utils


def _get_test_image_tensor():
    data_dir = os.path.join(os.path.dirname(__file__), "assets")
    img_path = os.path.join(data_dir, "grace_hopper_517x606.jpg")
    input_image = PIL.Image.open(img_path)
    # Based on example from https://pytorch.org/hub/pytorch_vision_resnet/
    preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    return preprocess(input_image).unsqueeze(0)


# Due to precision error from quantization, check only that the top prediction matches.
class _TopPredictor(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        x = self.base_model(x)
        _, topk_id = torch.topk(x[0], 1)
        return topk_id


# TODO: All torchvision quantized model test can be written as single parameterized test case,
# after per-parameter test decoration is supported via #79979, or after they are all enabled,
# whichever is first.
@parameterized.parameterized_class(
    ("is_script",),
    [(True,), (False,)],
    class_name_func=onnx_test_common.parameterize_class_name,
)
class TestQuantizedModelsONNXRuntime(onnx_test_common._TestONNXRuntime):
    def run_test(self, model, inputs, *args, **kwargs):
        model = _TopPredictor(model)
        return super().run_test(model, inputs, *args, **kwargs)

    def test_mobilenet_v3(self):
        model = torchvision.models.quantization.mobilenet_v3_large(
            pretrained=True, quantize=True
        )
        self.run_test(model, _get_test_image_tensor())

    @unittest.skip("quantized::cat not supported")
    def test_inception_v3(self):
        model = torchvision.models.quantization.inception_v3(
            pretrained=True, quantize=True
        )
        self.run_test(model, _get_test_image_tensor())

    @unittest.skip("quantized::cat not supported")
    def test_googlenet(self):
        model = torchvision.models.quantization.googlenet(
            pretrained=True, quantize=True
        )
        self.run_test(model, _get_test_image_tensor())

    @unittest.skip("quantized::cat not supported")
    def test_shufflenet_v2_x0_5(self):
        model = torchvision.models.quantization.shufflenet_v2_x0_5(
            pretrained=True, quantize=True
        )
        self.run_test(model, _get_test_image_tensor())

    def test_resnet18(self):
        model = torchvision.models.quantization.resnet18(pretrained=True, quantize=True)
        self.run_test(model, _get_test_image_tensor())

    def test_resnet50(self):
        model = torchvision.models.quantization.resnet50(pretrained=True, quantize=True)
        self.run_test(model, _get_test_image_tensor())

    def test_resnext101_32x8d(self):
        model = torchvision.models.quantization.resnext101_32x8d(
            pretrained=True, quantize=True
        )
        self.run_test(model, _get_test_image_tensor())


if __name__ == "__main__":
    common_utils.run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_TopPredictor`, `TestQuantizedModelsONNXRuntime`

**Functions defined**: `_get_test_image_tensor`, `__init__`, `forward`, `run_test`, `test_mobilenet_v3`, `test_inception_v3`, `test_googlenet`, `test_shufflenet_v2_x0_5`, `test_resnet18`, `test_resnet50`, `test_resnext101_32x8d`

**Key imports**: os, unittest, onnx_test_common, parameterized, PIL, torchvision, torch, nn, common_utils


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/onnx`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `unittest`
- `onnx_test_common`
- `parameterized`
- `PIL`
- `torchvision`
- `torch`
- `torch.testing._internal`: common_utils


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/onnx/test_models_quantized_onnxruntime.py
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

- **File Documentation**: `test_models_quantized_onnxruntime.py_docs.md`
- **Keyword Index**: `test_models_quantized_onnxruntime.py_kw.md`
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

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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
python docs/test/onnx/test_models_quantized_onnxruntime.py_docs.md
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

- **File Documentation**: `test_models_quantized_onnxruntime.py_docs.md_docs.md`
- **Keyword Index**: `test_models_quantized_onnxruntime.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
