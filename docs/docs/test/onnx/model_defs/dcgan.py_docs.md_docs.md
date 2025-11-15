# Documentation: `docs/test/onnx/model_defs/dcgan.py_docs.md`

## File Metadata

- **Path**: `docs/test/onnx/model_defs/dcgan.py_docs.md`
- **Size**: 5,510 bytes (5.38 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/onnx/model_defs/dcgan.py`

## File Metadata

- **Path**: `test/onnx/model_defs/dcgan.py`
- **Size**: 2,953 bytes (2.88 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```python
import torch
import torch.nn as nn


# configurable
bsz = 64
imgsz = 64
nz = 100
ngf = 64
ndf = 64
nc = 3


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _netG(nn.Module):
    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if self.ngpu > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class _netD(nn.Module):
    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        if self.ngpu > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1)

```



## High-Level Overview


This Python file contains 2 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_netG`, `_netD`

**Functions defined**: `weights_init`, `__init__`, `forward`, `__init__`, `forward`

**Key imports**: torch, torch.nn as nn


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/onnx/model_defs`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.nn as nn`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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
python test/onnx/model_defs/dcgan.py
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
- [`srresnet.py_docs.md`](./srresnet.py_docs.md)
- [`op_test.py_docs.md`](./op_test.py_docs.md)
- [`super_resolution.py_docs.md`](./super_resolution.py_docs.md)


## Cross-References

- **File Documentation**: `dcgan.py_docs.md`
- **Keyword Index**: `dcgan.py_kw.md`
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
python docs/test/onnx/model_defs/dcgan.py_docs.md
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
- [`srresnet.py_docs.md_docs.md`](./srresnet.py_docs.md_docs.md)
- [`mnist.py_docs.md_docs.md`](./mnist.py_docs.md_docs.md)
- [`squeezenet.py_kw.md_docs.md`](./squeezenet.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `dcgan.py_docs.md_docs.md`
- **Keyword Index**: `dcgan.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
