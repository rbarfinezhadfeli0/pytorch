# Documentation: `test/onnx/model_defs/squeezenet.py`

## File Metadata

- **Path**: `test/onnx/model_defs/squeezenet.py`
- **Size**: 3,497 bytes (3.42 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```python
import torch
import torch.nn as nn
import torch.nn.init as init


class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(
            squeeze_planes, expand3x3_planes, kernel_size=3, padding=1
        )
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [
                self.expand1x1_activation(self.expand1x1(x)),
                self.expand3x3_activation(self.expand3x3(x)),
            ],
            1,
        )


class SqueezeNet(nn.Module):
    def __init__(self, version=1.0, num_classes=1000, ceil_mode=False):
        super().__init__()
        if version not in [1.0, 1.1]:
            raise ValueError(
                f"Unsupported SqueezeNet version {version}:1.0 or 1.1 expected"
            )
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=ceil_mode),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=ceil_mode),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=ceil_mode),
                Fire(512, 64, 256, 256),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=ceil_mode),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=ceil_mode),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=ceil_mode),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), final_conv, nn.ReLU(inplace=True), nn.AvgPool2d(13)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)

```



## High-Level Overview


This Python file contains 2 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Fire`, `SqueezeNet`

**Functions defined**: `__init__`, `forward`, `__init__`, `forward`

**Key imports**: torch, torch.nn as nn, torch.nn.init as init


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/onnx/model_defs`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.nn as nn`
- `torch.nn.init as init`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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
python test/onnx/model_defs/squeezenet.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/onnx/model_defs`):

- [`word_language_model.py_docs.md`](./word_language_model.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`rnn_model_with_packed_sequence.py_docs.md`](./rnn_model_with_packed_sequence.py_docs.md)
- [`mnist.py_docs.md`](./mnist.py_docs.md)
- [`lstm_flattening_result.py_docs.md`](./lstm_flattening_result.py_docs.md)
- [`srresnet.py_docs.md`](./srresnet.py_docs.md)
- [`op_test.py_docs.md`](./op_test.py_docs.md)
- [`dcgan.py_docs.md`](./dcgan.py_docs.md)
- [`super_resolution.py_docs.md`](./super_resolution.py_docs.md)


## Cross-References

- **File Documentation**: `squeezenet.py_docs.md`
- **Keyword Index**: `squeezenet.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
