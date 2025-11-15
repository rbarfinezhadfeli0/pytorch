# Documentation: `docs/test/onnx/model_defs/word_language_model.py_docs.md`

## File Metadata

- **Path**: `docs/test/onnx/model_defs/word_language_model.py_docs.md`
- **Size**: 8,031 bytes (7.84 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/onnx/model_defs/word_language_model.py`

## File Metadata

- **Path**: `test/onnx/model_defs/word_language_model.py`
- **Size**: 4,635 bytes (4.53 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```python
# The model is from here:
#   https://github.com/pytorch/examples/blob/master/word_language_model/model.py

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(
        self,
        rnn_type,
        ntoken,
        ninp,
        nhid,
        nlayers,
        dropout=0.5,
        tie_weights=False,
        batchsize=2,
    ):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ["LSTM", "GRU"]:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {"RNN_TANH": "tanh", "RNN_RELU": "relu"}[rnn_type]
            except KeyError:
                raise ValueError(
                    """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']"""
                ) from None
            self.rnn = nn.RNN(
                ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout
            )
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError(
                    "When using the tied flag, nhid must be equal to emsize"
                )
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.hidden = self.init_hidden(batchsize)

    @staticmethod
    def repackage_hidden(h):
        """Detach hidden states from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple([RNNModel.repackage_hidden(v) for v in h])

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2))
        )
        self.hidden = RNNModel.repackage_hidden(hidden)
        return decoded.view(output.size(0), output.size(1), decoded.size(1))

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == "LSTM":
            return (
                weight.new(self.nlayers, bsz, self.nhid).zero_(),
                weight.new(self.nlayers, bsz, self.nhid).zero_(),
            )
        else:
            return weight.new(self.nlayers, bsz, self.nhid).zero_()


class RNNModelWithTensorHidden(RNNModel):
    """Supports GRU scripting."""

    @staticmethod
    def repackage_hidden(h):
        """Detach hidden states from their history."""
        return h.detach()

    def forward(self, input: Tensor, hidden: Tensor):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2))
        )
        self.hidden = RNNModelWithTensorHidden.repackage_hidden(hidden)
        return decoded.view(output.size(0), output.size(1), decoded.size(1))


class RNNModelWithTupleHidden(RNNModel):
    """Supports LSTM scripting."""

    @staticmethod
    def repackage_hidden(h: tuple[Tensor, Tensor]):
        """Detach hidden states from their history."""
        return (h[0].detach(), h[1].detach())

    def forward(self, input: Tensor, hidden: Optional[tuple[Tensor, Tensor]] = None):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2))
        )
        self.hidden = self.repackage_hidden(tuple(hidden))
        return decoded.view(output.size(0), output.size(1), decoded.size(1))

```



## High-Level Overview

"""Container module with an encoder, a recurrent module, and a decoder."""    def __init__(        self,        rnn_type,        ntoken,        ninp,        nhid,        nlayers,        dropout=0.5,        tie_weights=False,        batchsize=2,    ):        super().__init__()        self.drop = nn.Dropout(dropout)        self.encoder = nn.Embedding(ntoken, ninp)        if rnn_type in ["LSTM", "GRU"]:            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)        else:            try:                nonlinearity = {"RNN_TANH": "tanh", "RNN_RELU": "relu"}[rnn_type]            except KeyError:                raise ValueError(

This Python file contains 3 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `RNNModel`, `RNNModelWithTensorHidden`, `RNNModelWithTupleHidden`

**Functions defined**: `__init__`, `repackage_hidden`, `init_weights`, `forward`, `init_hidden`, `repackage_hidden`, `forward`, `repackage_hidden`, `forward`

**Key imports**: Optional, torch, torch.nn as nn, Tensor


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/onnx/model_defs`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Optional
- `torch`
- `torch.nn as nn`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
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
python test/onnx/model_defs/word_language_model.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/onnx/model_defs`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`rnn_model_with_packed_sequence.py_docs.md`](./rnn_model_with_packed_sequence.py_docs.md)
- [`mnist.py_docs.md`](./mnist.py_docs.md)
- [`lstm_flattening_result.py_docs.md`](./lstm_flattening_result.py_docs.md)
- [`squeezenet.py_docs.md`](./squeezenet.py_docs.md)
- [`srresnet.py_docs.md`](./srresnet.py_docs.md)
- [`op_test.py_docs.md`](./op_test.py_docs.md)
- [`dcgan.py_docs.md`](./dcgan.py_docs.md)
- [`super_resolution.py_docs.md`](./super_resolution.py_docs.md)


## Cross-References

- **File Documentation**: `word_language_model.py_docs.md`
- **Keyword Index**: `word_language_model.py_kw.md`
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
- **Error Handling**: Includes exception handling
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
python docs/test/onnx/model_defs/word_language_model.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/onnx/model_defs`):

- [`lstm_flattening_result.py_kw.md_docs.md`](./lstm_flattening_result.py_kw.md_docs.md)
- [`op_test.py_kw.md_docs.md`](./op_test.py_kw.md_docs.md)
- [`rnn_model_with_packed_sequence.py_docs.md_docs.md`](./rnn_model_with_packed_sequence.py_docs.md_docs.md)
- [`super_resolution.py_docs.md_docs.md`](./super_resolution.py_docs.md_docs.md)
- [`emb_seq.py_docs.md_docs.md`](./emb_seq.py_docs.md_docs.md)
- [`srresnet.py_kw.md_docs.md`](./srresnet.py_kw.md_docs.md)
- [`srresnet.py_docs.md_docs.md`](./srresnet.py_docs.md_docs.md)
- [`mnist.py_docs.md_docs.md`](./mnist.py_docs.md_docs.md)
- [`squeezenet.py_kw.md_docs.md`](./squeezenet.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `word_language_model.py_docs.md_docs.md`
- **Keyword Index**: `word_language_model.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
