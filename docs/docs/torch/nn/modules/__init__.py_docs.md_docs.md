# Documentation: `docs/torch/nn/modules/__init__.py_docs.md`

## File Metadata

- **Path**: `docs/torch/nn/modules/__init__.py_docs.md`
- **Size**: 9,350 bytes (9.13 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This is a **Python package initialization file**.

## Original Source

```markdown
# Documentation: `torch/nn/modules/__init__.py`

## File Metadata

- **Path**: `torch/nn/modules/__init__.py`
- **Size**: 6,494 bytes (6.34 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a **Python package initialization file**.

## Original Source

```python
from .module import Module  # usort: skip
from .linear import Bilinear, Identity, LazyLinear, Linear  # usort: skip
from .activation import (
    CELU,
    ELU,
    GELU,
    GLU,
    Hardshrink,
    Hardsigmoid,
    Hardswish,
    Hardtanh,
    LeakyReLU,
    LogSigmoid,
    LogSoftmax,
    Mish,
    MultiheadAttention,
    PReLU,
    ReLU,
    ReLU6,
    RReLU,
    SELU,
    Sigmoid,
    SiLU,
    Softmax,
    Softmax2d,
    Softmin,
    Softplus,
    Softshrink,
    Softsign,
    Tanh,
    Tanhshrink,
    Threshold,
)
from .adaptive import AdaptiveLogSoftmaxWithLoss
from .batchnorm import (
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    LazyBatchNorm1d,
    LazyBatchNorm2d,
    LazyBatchNorm3d,
    SyncBatchNorm,
)
from .channelshuffle import ChannelShuffle
from .container import (
    Container,
    ModuleDict,
    ModuleList,
    ParameterDict,
    ParameterList,
    Sequential,
)
from .conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    LazyConv1d,
    LazyConv2d,
    LazyConv3d,
    LazyConvTranspose1d,
    LazyConvTranspose2d,
    LazyConvTranspose3d,
)
from .distance import CosineSimilarity, PairwiseDistance
from .dropout import (
    AlphaDropout,
    Dropout,
    Dropout1d,
    Dropout2d,
    Dropout3d,
    FeatureAlphaDropout,
)
from .flatten import Flatten, Unflatten
from .fold import Fold, Unfold
from .instancenorm import (
    InstanceNorm1d,
    InstanceNorm2d,
    InstanceNorm3d,
    LazyInstanceNorm1d,
    LazyInstanceNorm2d,
    LazyInstanceNorm3d,
)
from .loss import (
    BCELoss,
    BCEWithLogitsLoss,
    CosineEmbeddingLoss,
    CrossEntropyLoss,
    CTCLoss,
    GaussianNLLLoss,
    HingeEmbeddingLoss,
    HuberLoss,
    KLDivLoss,
    L1Loss,
    MarginRankingLoss,
    MSELoss,
    MultiLabelMarginLoss,
    MultiLabelSoftMarginLoss,
    MultiMarginLoss,
    NLLLoss,
    NLLLoss2d,
    PoissonNLLLoss,
    SmoothL1Loss,
    SoftMarginLoss,
    TripletMarginLoss,
    TripletMarginWithDistanceLoss,
)
from .normalization import (
    CrossMapLRN2d,
    GroupNorm,
    LayerNorm,
    LocalResponseNorm,
    RMSNorm,
)
from .padding import (
    CircularPad1d,
    CircularPad2d,
    CircularPad3d,
    ConstantPad1d,
    ConstantPad2d,
    ConstantPad3d,
    ReflectionPad1d,
    ReflectionPad2d,
    ReflectionPad3d,
    ReplicationPad1d,
    ReplicationPad2d,
    ReplicationPad3d,
    ZeroPad1d,
    ZeroPad2d,
    ZeroPad3d,
)
from .pixelshuffle import PixelShuffle, PixelUnshuffle
from .pooling import (
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveAvgPool3d,
    AdaptiveMaxPool1d,
    AdaptiveMaxPool2d,
    AdaptiveMaxPool3d,
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    FractionalMaxPool2d,
    FractionalMaxPool3d,
    LPPool1d,
    LPPool2d,
    LPPool3d,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
    MaxUnpool1d,
    MaxUnpool2d,
    MaxUnpool3d,
)
from .rnn import GRU, GRUCell, LSTM, LSTMCell, RNN, RNNBase, RNNCell, RNNCellBase
from .sparse import Embedding, EmbeddingBag
from .transformer import (
    Transformer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from .upsampling import Upsample, UpsamplingBilinear2d, UpsamplingNearest2d


__all__ = [
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "AdaptiveLogSoftmaxWithLoss",
    "AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d",
    "AdaptiveMaxPool3d",
    "AlphaDropout",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "BCELoss",
    "BCEWithLogitsLoss",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "Bilinear",
    "CELU",
    "CTCLoss",
    "ChannelShuffle",
    "CircularPad1d",
    "CircularPad2d",
    "CircularPad3d",
    "ConstantPad1d",
    "ConstantPad2d",
    "ConstantPad3d",
    "Container",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "CosineEmbeddingLoss",
    "CosineSimilarity",
    "CrossEntropyLoss",
    "CrossMapLRN2d",
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "ELU",
    "Embedding",
    "EmbeddingBag",
    "FeatureAlphaDropout",
    "Flatten",
    "Fold",
    "FractionalMaxPool2d",
    "FractionalMaxPool3d",
    "GELU",
    "GLU",
    "GRU",
    "GRUCell",
    "GaussianNLLLoss",
    "GroupNorm",
    "Hardshrink",
    "Hardsigmoid",
    "Hardswish",
    "Hardtanh",
    "HingeEmbeddingLoss",
    "HuberLoss",
    "Identity",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "KLDivLoss",
    "L1Loss",
    "LPPool1d",
    "LPPool2d",
    "LPPool3d",
    "LSTM",
    "LSTMCell",
    "LayerNorm",
    "LazyBatchNorm1d",
    "LazyBatchNorm2d",
    "LazyBatchNorm3d",
    "LazyConv1d",
    "LazyConv2d",
    "LazyConv3d",
    "LazyConvTranspose1d",
    "LazyConvTranspose2d",
    "LazyConvTranspose3d",
    "LazyInstanceNorm1d",
    "LazyInstanceNorm2d",
    "LazyInstanceNorm3d",
    "LazyLinear",
    "LeakyReLU",
    "Linear",
    "LocalResponseNorm",
    "LogSigmoid",
    "LogSoftmax",
    "MSELoss",
    "MarginRankingLoss",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "MaxUnpool1d",
    "MaxUnpool2d",
    "MaxUnpool3d",
    "Mish",
    "Module",
    "ModuleDict",
    "ModuleList",
    "MultiLabelMarginLoss",
    "MultiLabelSoftMarginLoss",
    "MultiMarginLoss",
    "MultiheadAttention",
    "NLLLoss",
    "NLLLoss2d",
    "PReLU",
    "PairwiseDistance",
    "ParameterDict",
    "ParameterList",
    "PixelShuffle",
    "PixelUnshuffle",
    "PoissonNLLLoss",
    "RMSNorm",
    "RNN",
    "RNNBase",
    "RNNCell",
    "RNNCellBase",
    "RReLU",
    "ReLU",
    "ReLU6",
    "ReflectionPad1d",
    "ReflectionPad2d",
    "ReflectionPad3d",
    "ReplicationPad1d",
    "ReplicationPad2d",
    "ReplicationPad3d",
    "SELU",
    "Sequential",
    "SiLU",
    "Sigmoid",
    "SmoothL1Loss",
    "SoftMarginLoss",
    "Softmax",
    "Softmax2d",
    "Softmin",
    "Softplus",
    "Softshrink",
    "Softsign",
    "SyncBatchNorm",
    "Tanh",
    "Tanhshrink",
    "Threshold",
    "Transformer",
    "TransformerDecoder",
    "TransformerDecoderLayer",
    "TransformerEncoder",
    "TransformerEncoderLayer",
    "TripletMarginLoss",
    "TripletMarginWithDistanceLoss",
    "Unflatten",
    "Unfold",
    "Upsample",
    "UpsamplingBilinear2d",
    "UpsamplingNearest2d",
    "ZeroPad1d",
    "ZeroPad2d",
    "ZeroPad3d",
]

# Please keep this list sorted
assert __all__ == sorted(__all__)

```



## High-Level Overview


This Python file contains 0 class(es) and 0 function(s).

## Detailed Analysis

### Code Structure

**Key imports**: Module  , Bilinear, Identity, LazyLinear, Linear  , AdaptiveLogSoftmaxWithLoss, ChannelShuffle, CosineSimilarity, PairwiseDistance, Flatten, Unflatten, Fold, Unfold, PixelShuffle, PixelUnshuffle, GRU, GRUCell, LSTM, LSTMCell, RNN, RNNBase, RNNCell, RNNCellBase, Embedding, EmbeddingBag


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nn/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `.module`: Module  
- `.linear`: Bilinear, Identity, LazyLinear, Linear  
- `.adaptive`: AdaptiveLogSoftmaxWithLoss
- `.channelshuffle`: ChannelShuffle
- `.distance`: CosineSimilarity, PairwiseDistance
- `.flatten`: Flatten, Unflatten
- `.fold`: Fold, Unfold
- `.pixelshuffle`: PixelShuffle, PixelUnshuffle
- `.rnn`: GRU, GRUCell, LSTM, LSTMCell, RNN, RNNBase, RNNCell, RNNCellBase
- `.sparse`: Embedding, EmbeddingBag
- `.upsampling`: Upsample, UpsamplingBilinear2d, UpsamplingNearest2d


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/nn/modules`):

- [`fold.py_docs.md`](./fold.py_docs.md)
- [`rnn.py_docs.md`](./rnn.py_docs.md)
- [`channelshuffle.py_docs.md`](./channelshuffle.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`adaptive.py_docs.md`](./adaptive.py_docs.md)
- [`conv.py_docs.md`](./conv.py_docs.md)
- [`distance.py_docs.md`](./distance.py_docs.md)
- [`linear.py_docs.md`](./linear.py_docs.md)
- [`normalization.py_docs.md`](./normalization.py_docs.md)


## Cross-References

- **File Documentation**: `__init__.py_docs.md`
- **Keyword Index**: `__init__.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/nn/modules`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/nn/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/nn/modules`):

- [`sparse.py_docs.md_docs.md`](./sparse.py_docs.md_docs.md)
- [`instancenorm.py_kw.md_docs.md`](./instancenorm.py_kw.md_docs.md)
- [`activation.py_kw.md_docs.md`](./activation.py_kw.md_docs.md)
- [`container.py_docs.md_docs.md`](./container.py_docs.md_docs.md)
- [`distance.py_kw.md_docs.md`](./distance.py_kw.md_docs.md)
- [`pixelshuffle.py_kw.md_docs.md`](./pixelshuffle.py_kw.md_docs.md)
- [`module.py_docs.md_docs.md`](./module.py_docs.md_docs.md)
- [`batchnorm.py_docs.md_docs.md`](./batchnorm.py_docs.md_docs.md)
- [`transformer.py_kw.md_docs.md`](./transformer.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `__init__.py_docs.md_docs.md`
- **Keyword Index**: `__init__.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
