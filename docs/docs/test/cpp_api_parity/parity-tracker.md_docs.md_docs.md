# Documentation: `docs/test/cpp_api_parity/parity-tracker.md_docs.md`

## File Metadata

- **Path**: `docs/test/cpp_api_parity/parity-tracker.md_docs.md`
- **Size**: 9,171 bytes (8.96 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/cpp_api_parity/parity-tracker.md`

## File Metadata

- **Path**: `test/cpp_api_parity/parity-tracker.md`
- **Size**: 6,906 bytes (6.74 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```markdown
# C++ / Python API parity tracker

## torch::nn
API | Implementation Parity | Doc Parity
------------- | ------------- | -------------
torch::nn::Sequential|Yes|No
torch::nn::ModuleList|Yes|No
torch::nn::ModuleDict|No|No
torch::nn::ParameterList|No|No
torch::nn::ParameterDict|No|No
torch::nn::Conv1d|Yes|No
torch::nn::Conv2d|Yes|No
torch::nn::Conv3d|Yes|No
torch::nn::ConvTranspose1d|Yes|No
torch::nn::ConvTranspose2d|Yes|No
torch::nn::ConvTranspose3d|Yes|No
torch::nn::Unfold|Yes|No
torch::nn::Fold|Yes|No
torch::nn::MaxPool1d|Yes|No
torch::nn::MaxPool2d|Yes|No
torch::nn::MaxPool3d|Yes|No
torch::nn::MaxUnpool1d|Yes|No
torch::nn::MaxUnpool2d|Yes|No
torch::nn::MaxUnpool3d|Yes|No
torch::nn::AvgPool1d|Yes|No
torch::nn::AvgPool2d|Yes|No
torch::nn::AvgPool3d|Yes|No
torch::nn::FractionalMaxPool2d|Yes|No
torch::nn::FractionalMaxPool3d|Yes|No
torch::nn::LPPool1d|Yes|No
torch::nn::LPPool2d|Yes|No
torch::nn::LPPool3d|Yes|No
torch::nn::AdaptiveMaxPool1d|Yes|No
torch::nn::AdaptiveMaxPool2d|Yes|No
torch::nn::AdaptiveMaxPool3d|Yes|No
torch::nn::AdaptiveAvgPool1d|Yes|No
torch::nn::AdaptiveAvgPool2d|Yes|No
torch::nn::AdaptiveAvgPool3d|Yes|No
torch::nn::ReflectionPad1d|Yes|No
torch::nn::ReflectionPad2d|Yes|No
torch::nn::ReflectionPad3d|Yes|No
torch::nn::ReplicationPad1d|Yes|No
torch::nn::ReplicationPad2d|Yes|No
torch::nn::ReplicationPad3d|Yes|No
torch::nn::ZeroPad1d|Yes|No
torch::nn::ZeroPad2d|Yes|No
torch::nn::ZeroPad3d|Yes|No
torch::nn::ConstantPad1d|Yes|No
torch::nn::ConstantPad2d|Yes|No
torch::nn::ConstantPad3d|Yes|No
torch::nn::ELU|Yes|No
torch::nn::Hardshrink|Yes|No
torch::nn::Hardtanh|Yes|No
torch::nn::LeakyReLU|Yes|No
torch::nn::LogSigmoid|Yes|No
torch::nn::Mish|Yes|No
torch::nn::MultiheadAttention|No|No
torch::nn::PReLU|Yes|No
torch::nn::ReLU|Yes|No
torch::nn::ReLU6|Yes|No
torch::nn::RReLU|Yes|No
torch::nn::SELU|Yes|No
torch::nn::CELU|Yes|No
torch::nn::GELU|Yes|No
torch::nn::SiLU|Yes|No
torch::nn::Sigmoid|Yes|No
torch::nn::Softplus|Yes|No
torch::nn::Softshrink|Yes|No
torch::nn::Softsign|Yes|No
torch::nn::Tanh|Yes|No
torch::nn::Tanhshrink|Yes|No
torch::nn::Threshold|Yes|No
torch::nn::GLU|Yes|No
torch::nn::Softmin|Yes|No
torch::nn::Softmax|Yes|No
torch::nn::Softmax2d|Yes|No
torch::nn::LogSoftmax|Yes|No
torch::nn::AdaptiveLogSoftmaxWithLoss|Yes|No
torch::nn::BatchNorm1d|Yes|No
torch::nn::BatchNorm2d|Yes|No
torch::nn::BatchNorm3d|Yes|No
torch::nn::GroupNorm|Yes|No
torch::nn::SyncBatchNorm|No|No
torch::nn::InstanceNorm1d|Yes|No
torch::nn::InstanceNorm2d|Yes|No
torch::nn::InstanceNorm3d|Yes|No
torch::nn::LayerNorm|Yes|No
torch::nn::LocalResponseNorm|Yes|No
torch::nn::CrossMapLRN2d|Yes|No
torch::nn::RNN|Yes|No
torch::nn::LSTM|Yes|No
torch::nn::GRU|Yes|No
torch::nn::RNNCell|Yes|No
torch::nn::LSTMCell|Yes|No
torch::nn::GRUCell|Yes|No
torch::nn::Transformer|Yes|No
torch::nn::TransformerEncoder|No|No
torch::nn::TransformerDecoder|No|No
torch::nn::TransformerEncoderLayer|Yes|No
torch::nn::TransformerDecoderLayer|Yes|No
torch::nn::Identity|Yes|No
torch::nn::Linear|Yes|No
torch::nn::Bilinear|Yes|No
torch::nn::Flatten|Yes|No
torch::nn::Unflatten|Yes|No
torch::nn::Dropout|Yes|No
torch::nn::Dropout2d|Yes|No
torch::nn::Dropout3d|Yes|No
torch::nn::AlphaDropout|Yes|No
torch::nn::FeatureAlphaDropout|Yes|No
torch::nn::Embedding|Yes|No
torch::nn::EmbeddingBag|Yes|No
torch::nn::CosineSimilarity|Yes|No
torch::nn::PairwiseDistance|Yes|No
torch::nn::L1Loss|Yes|No
torch::nn::MSELoss|Yes|No
torch::nn::CrossEntropyLoss|Yes|No
torch::nn::CTCLoss|Yes|No
torch::nn::NLLLoss|Yes|No
torch::nn::PoissonNLLLoss|Yes|No
torch::nn::KLDivLoss|Yes|No
torch::nn::BCELoss|Yes|No
torch::nn::BCEWithLogitsLoss|Yes|No
torch::nn::MarginRankingLoss|Yes|No
torch::nn::HingeEmbeddingLoss|Yes|No
torch::nn::MultiLabelMarginLoss|Yes|No
torch::nn::SmoothL1Loss|Yes|No
torch::nn::HuberLoss|Yes|No
torch::nn::SoftMarginLoss|Yes|No
torch::nn::MultiLabelSoftMarginLoss|Yes|No
torch::nn::CosineEmbeddingLoss|Yes|No
torch::nn::MultiMarginLoss|Yes|No
torch::nn::TripletMarginLoss|Yes|No
torch::nn::PixelShuffle|Yes|No
torch::nn::PixelUnshuffle|Yes|No
torch::nn::Upsample|Yes|No
torch::nn::DataParallel|No|No
torch::nn::parallel::DistributedDataParallel|No|No
torch::nn::utils::clip_grad_norm_|Yes|No
torch::nn::utils::clip_grad_value_|Yes|No
torch::nn::utils::parameters_to_vector|Yes|No
torch::nn::utils::vector_to_parameters|Yes|No
torch::nn::utils::weight_norm|No|No
torch::nn::utils::remove_weight_norm|No|No
torch::nn::utils::spectral_norm|No|No
torch::nn::utils::remove_spectral_norm|No|No
torch::nn::utils::rnn::PackedSequence|Yes|No
torch::nn::utils::rnn::pack_padded_sequence|Yes|No
torch::nn::utils::rnn::pad_packed_sequence|Yes|No
torch::nn::utils::rnn::pad_sequence|Yes|No
torch::nn::utils::rnn::pack_sequence|Yes|No
torch::nn::SampleModule|Yes|Yes

## torch::nn::functional

API | Implementation Parity | Doc Parity
------------- | ------------- | -------------
F::conv1d|Yes|No
F::conv2d|Yes|No
F::conv3d|Yes|No
F::conv_transpose1d|Yes|No
F::conv_transpose2d|Yes|No
F::conv_transpose3d|Yes|No
F::unfold|Yes|No
F::fold|Yes|No
F::avg_pool1d|Yes|No
F::avg_pool2d|Yes|No
F::avg_pool3d|Yes|No
F::max_pool1d|Yes|No
F::max_pool2d|Yes|No
F::max_pool3d|Yes|No
F::max_unpool1d|Yes|No
F::max_unpool2d|Yes|No
F::max_unpool3d|Yes|No
F::lp_pool1d|Yes|No
F::lp_pool2d|Yes|No
F::lp_pool3d|Yes|No
F::adaptive_max_pool1d|Yes|No
F::adaptive_max_pool2d|Yes|No
F::adaptive_max_pool3d|Yes|No
F::adaptive_avg_pool1d|Yes|No
F::adaptive_avg_pool2d|Yes|No
F::adaptive_avg_pool3d|Yes|No
F::threshold|Yes|No
F::relu|Yes|No
F::hardtanh|Yes|No
F::relu6|Yes|No
F::elu|Yes|No
F::selu|Yes|No
F::celu|Yes|No
F::leaky_relu|Yes|No
F::prelu|Yes|No
F::rrelu|Yes|No
F::glu|Yes|No
F::gelu|Yes|No
F::silu|Yes|No
F::mish|Yes|No
F::logsigmoid|Yes|No
F::hardshrink|Yes|No
F::tanhshrink|Yes|No
F::softsign|Yes|No
F::softplus|Yes|No
F::softmin|Yes|No
F::softmax|Yes|No
F::softshrink|Yes|No
F::gumbel_softmax|Yes|No
F::log_softmax|Yes|No
F::batch_norm|Yes|No
F::instance_norm|Yes|No
F::layer_norm|Yes|No
F::local_response_norm|Yes|No
F::normalize|Yes|No
F::linear|Yes|No
F::bilinear|Yes|No
F::dropout|Yes|No
F::alpha_dropout|Yes|No
F::dropout2d|Yes|No
F::dropout3d|Yes|No
F::embedding|Yes|No
F::embedding_bag|Yes|No
F::one_hot|Yes|No
F::pairwise_distance|Yes|No
F::cosine_similarity|Yes|No
F::pdist|Yes|No
F::binary_cross_entropy|Yes|No
F::binary_cross_entropy_with_logits|Yes|No
F::poisson_nll_loss|Yes|No
F::cosine_embedding_loss|Yes|No
F::cross_entropy|Yes|No
F::ctc_loss|Yes|No
F::hinge_embedding_loss|Yes|No
F::kl_div|Yes|No
F::l1_loss|Yes|No
F::mse_loss|Yes|No
F::margin_ranking_loss|Yes|No
F::multilabel_margin_loss|Yes|No
F::multilabel_soft_margin_loss|Yes|No
F::multi_margin_loss|Yes|No
F::nll_loss|Yes|No
F::smooth_l1_loss|Yes|No
F::huber_loss|Yes|No
F::soft_margin_loss|Yes|No
F::triplet_margin_loss|Yes|No
F::pixel_shuffle|Yes|No
F::pad|Yes|No
F::interpolate|Yes|No
F::grid_sample|Yes|No
F::affine_grid|Yes|No
F::sample_functional|Yes|No
```



## High-Level Overview

This file is part of the PyTorch framework located at `test/cpp_api_parity`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp_api_parity`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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
python test/cpp_api_parity/parity-tracker.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp_api_parity`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`sample_functional.py_docs.md`](./sample_functional.py_docs.md)
- [`module_impl_check.py_docs.md`](./module_impl_check.py_docs.md)
- [`sample_module.py_docs.md`](./sample_module.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`parity_table_parser.py_docs.md`](./parity_table_parser.py_docs.md)
- [`functional_impl_check.py_docs.md`](./functional_impl_check.py_docs.md)


## Cross-References

- **File Documentation**: `parity-tracker.md_docs.md`
- **Keyword Index**: `parity-tracker.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp_api_parity`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp_api_parity`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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
python docs/test/cpp_api_parity/parity-tracker.md_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp_api_parity`):

- [`parity_table_parser.py_docs.md_docs.md`](./parity_table_parser.py_docs.md_docs.md)
- [`module_impl_check.py_docs.md_docs.md`](./module_impl_check.py_docs.md_docs.md)
- [`module_impl_check.py_kw.md_docs.md`](./module_impl_check.py_kw.md_docs.md)
- [`parity-tracker.md_kw.md_docs.md`](./parity-tracker.md_kw.md_docs.md)
- [`sample_module.py_kw.md_docs.md`](./sample_module.py_kw.md_docs.md)
- [`parity_table_parser.py_kw.md_docs.md`](./parity_table_parser.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`functional_impl_check.py_docs.md_docs.md`](./functional_impl_check.py_docs.md_docs.md)
- [`sample_module.py_docs.md_docs.md`](./sample_module.py_docs.md_docs.md)
- [`sample_functional.py_docs.md_docs.md`](./sample_functional.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `parity-tracker.md_docs.md_docs.md`
- **Keyword Index**: `parity-tracker.md_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
