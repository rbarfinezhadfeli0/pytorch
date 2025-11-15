# Documentation: `test/quantization/core/experimental/apot_fx_graph_mode_qat.py`

## File Metadata

- **Path**: `test/quantization/core/experimental/apot_fx_graph_mode_qat.py`
- **Size**: 2,932 bytes (2.86 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```python
from torchvision.models.quantization.resnet import resnet18
from torch.ao.quantization.experimental.quantization_helper import (
    evaluate,
    prepare_data_loaders,
    training_loop
)

# training and validation dataset: full ImageNet dataset
data_path = '~/my_imagenet/'

train_batch_size = 30
eval_batch_size = 50

data_loader, data_loader_test = prepare_data_loaders(data_path)
criterion = nn.CrossEntropyLoss()  # noqa: F821
float_model = resnet18(pretrained=True)
float_model.eval()

# deepcopy the model since we need to keep the original model around
import copy
model_to_quantize = copy.deepcopy(float_model)

model_to_quantize.eval()

"""
Prepare model QAT for specified qconfig for torch.nn.Linear
"""
def prepare_qat_linear(qconfig):
    qconfig_dict = {"object_type": [(torch.nn.Linear, qconfig)]}  # noqa: F821
    prepared_model = prepare_fx(copy.deepcopy(float_model), qconfig_dict)  # fuse modules and insert observers  # noqa: F821
    training_loop(prepared_model, criterion, data_loader)
    prepared_model.eval()
    return prepared_model

"""
Prepare model with uniform activation, uniform weight
b=8, k=2
"""

prepared_model = prepare_qat_linear(uniform_qconfig_8bit)  # noqa: F821

top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print(f"Model #1 Evaluation accuracy on test dataset (b=8, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")

"""
Prepare model with uniform activation, uniform weight
b=4, k=2
"""

prepared_model = prepare_qat_linear(uniform_qconfig_4bit)  # noqa: F821

top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print(f"Model #1 Evaluation accuracy on test dataset (b=4, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")

"""
Prepare model with uniform activation, APoT weight
(b=8, k=2)
"""

prepared_model = prepare_qat_linear(apot_weights_qconfig_8bit)  # noqa: F821

top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print(f"Model #2 Evaluation accuracy on test dataset (b=8, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")

"""
Prepare model with uniform activation, APoT weight
(b=4, k=2)
"""

prepared_model = prepare_qat_linear(apot_weights_qconfig_4bit)  # noqa: F821

top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print(f"Model #2 Evaluation accuracy on test dataset (b=4, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")


"""
Prepare model with APoT activation and weight
(b=8, k=2)
"""

prepared_model = prepare_qat_linear(apot_qconfig_8bit)  # noqa: F821

top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print(f"Model #3 Evaluation accuracy on test dataset (b=8, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")

"""
Prepare model with APoT activation and weight
(b=4, k=2)
"""

prepared_model = prepare_qat_linear(apot_qconfig_4bit)  # noqa: F821

top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print(f"Model #3 Evaluation accuracy on test dataset (b=4, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")

```



## High-Level Overview

"""Prepare model QAT for specified qconfig for torch.nn.Linear

This Python file contains 0 class(es) and 1 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `prepare_qat_linear`

**Key imports**: resnet18, copy


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/quantization/core/experimental`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torchvision.models.quantization.resnet`: resnet18
- `copy`


## Code Patterns & Idioms

### Common Patterns

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
python test/quantization/core/experimental/apot_fx_graph_mode_qat.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/quantization/core/experimental`):

- [`test_adaround_eager.py_docs.md`](./test_adaround_eager.py_docs.md)
- [`test_fake_quantize.py_docs.md`](./test_fake_quantize.py_docs.md)
- [`test_floatx.py_docs.md`](./test_floatx.py_docs.md)
- [`test_quantizer.py_docs.md`](./test_quantizer.py_docs.md)
- [`test_bits.py_docs.md`](./test_bits.py_docs.md)
- [`apot_fx_graph_mode_ptq.py_docs.md`](./apot_fx_graph_mode_ptq.py_docs.md)
- [`test_quantized_tensor.py_docs.md`](./test_quantized_tensor.py_docs.md)
- [`test_nonuniform_observer.py_docs.md`](./test_nonuniform_observer.py_docs.md)
- [`test_linear.py_docs.md`](./test_linear.py_docs.md)


## Cross-References

- **File Documentation**: `apot_fx_graph_mode_qat.py_docs.md`
- **Keyword Index**: `apot_fx_graph_mode_qat.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
