# Documentation: `docs/test/onnx/model_defs/srresnet.py_docs.md`

## File Metadata

- **Path**: `docs/test/onnx/model_defs/srresnet.py_docs.md`
- **Size**: 5,805 bytes (5.67 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/onnx/model_defs/srresnet.py`

## File Metadata

- **Path**: `test/onnx/model_defs/srresnet.py`
- **Size**: 3,245 bytes (3.17 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```python
import math

from torch import nn
from torch.nn import init


def _initialize_orthogonal(conv):
    prelu_gain = math.sqrt(2)
    init.orthogonal(conv.weight, gain=prelu_gain)
    if conv.bias is not None:
        conv.bias.data.zero_()


class ResidualBlock(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(
            n_filters, n_filters, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.prelu = nn.PReLU(n_filters)
        self.conv2 = nn.Conv2d(
            n_filters, n_filters, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(n_filters)

        # Orthogonal initialisation
        _initialize_orthogonal(self.conv1)
        _initialize_orthogonal(self.conv2)

    def forward(self, x):
        residual = self.prelu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        return x + residual


class UpscaleBlock(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        self.upscaling_conv = nn.Conv2d(
            n_filters, 4 * n_filters, kernel_size=3, padding=1
        )
        self.upscaling_shuffler = nn.PixelShuffle(2)
        self.upscaling = nn.PReLU(n_filters)
        _initialize_orthogonal(self.upscaling_conv)

    def forward(self, x):
        return self.upscaling(self.upscaling_shuffler(self.upscaling_conv(x)))


class SRResNet(nn.Module):
    def __init__(self, rescale_factor, n_filters, n_blocks):
        super().__init__()
        self.rescale_levels = int(math.log(rescale_factor, 2))  # noqa: FURB163
        self.n_filters = n_filters
        self.n_blocks = n_blocks

        self.conv1 = nn.Conv2d(3, n_filters, kernel_size=9, padding=4)
        self.prelu1 = nn.PReLU(n_filters)

        for residual_block_num in range(1, n_blocks + 1):
            residual_block = ResidualBlock(self.n_filters)
            self.add_module(
                "residual_block" + str(residual_block_num),
                nn.Sequential(residual_block),
            )

        self.skip_conv = nn.Conv2d(
            n_filters, n_filters, kernel_size=3, padding=1, bias=False
        )
        self.skip_bn = nn.BatchNorm2d(n_filters)

        for upscale_block_num in range(1, self.rescale_levels + 1):
            upscale_block = UpscaleBlock(self.n_filters)
            self.add_module(
                "upscale_block" + str(upscale_block_num), nn.Sequential(upscale_block)
            )

        self.output_conv = nn.Conv2d(n_filters, 3, kernel_size=9, padding=4)

        # Orthogonal initialisation
        _initialize_orthogonal(self.conv1)
        _initialize_orthogonal(self.skip_conv)
        _initialize_orthogonal(self.output_conv)

    def forward(self, x):
        x_init = self.prelu1(self.conv1(x))
        x = self.residual_block1(x_init)
        for residual_block_num in range(2, self.n_blocks + 1):
            x = getattr(self, "residual_block" + str(residual_block_num))(x)
        x = self.skip_bn(self.skip_conv(x)) + x_init
        for upscale_block_num in range(1, self.rescale_levels + 1):
            x = getattr(self, "upscale_block" + str(upscale_block_num))(x)
        return self.output_conv(x)

```



## High-Level Overview


This Python file contains 3 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ResidualBlock`, `UpscaleBlock`, `SRResNet`

**Functions defined**: `_initialize_orthogonal`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`

**Key imports**: math, nn, init


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/onnx/model_defs`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `math`
- `torch`: nn
- `torch.nn`: init


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
python test/onnx/model_defs/srresnet.py
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
- [`squeezenet.py_docs.md`](./squeezenet.py_docs.md)
- [`op_test.py_docs.md`](./op_test.py_docs.md)
- [`dcgan.py_docs.md`](./dcgan.py_docs.md)
- [`super_resolution.py_docs.md`](./super_resolution.py_docs.md)


## Cross-References

- **File Documentation**: `srresnet.py_docs.md`
- **Keyword Index**: `srresnet.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/onnx/model_defs`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/onnx/model_defs`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

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
python docs/test/onnx/model_defs/srresnet.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/onnx/model_defs`):

- [`lstm_flattening_result.py_kw.md_docs.md`](./lstm_flattening_result.py_kw.md_docs.md)
- [`op_test.py_kw.md_docs.md`](./op_test.py_kw.md_docs.md)
- [`rnn_model_with_packed_sequence.py_docs.md_docs.md`](./rnn_model_with_packed_sequence.py_docs.md_docs.md)
- [`word_language_model.py_docs.md_docs.md`](./word_language_model.py_docs.md_docs.md)
- [`super_resolution.py_docs.md_docs.md`](./super_resolution.py_docs.md_docs.md)
- [`emb_seq.py_docs.md_docs.md`](./emb_seq.py_docs.md_docs.md)
- [`srresnet.py_kw.md_docs.md`](./srresnet.py_kw.md_docs.md)
- [`mnist.py_docs.md_docs.md`](./mnist.py_docs.md_docs.md)
- [`squeezenet.py_kw.md_docs.md`](./squeezenet.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `srresnet.py_docs.md_docs.md`
- **Keyword Index**: `srresnet.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
