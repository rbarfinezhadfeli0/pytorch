# Documentation: transformercoder.h

## File Metadata
- **Path**: `torch/csrc/api/include/torch/nn/modules/transformercoder.h`
- **Size**: 5203 bytes
- **Lines**: 152
- **Extension**: .h
- **Type**: Regular file

## Original Source

```h
#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>
#include <torch/nn/modules/container/modulelist.h>
#include <torch/nn/options/transformercoder.h>
#include <torch/nn/pimpl.h>

#include <torch/types.h>

#include <utility>

namespace torch::nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TransformerEncoder
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// TransformerEncoder module.
/// See
/// https://pytorch.org/docs/main/generated/torch.nn.TransformerEncoder.html
/// to learn abouut the exact behavior of this encoder layer module.
///
/// See the documentation for `torch::nn::TransformerEncoder` class to learn
/// what constructor arguments are supported for this encoder module.
///
/// Example:
/// ```
/// TransformerEncoderLayer encoderLayer(TransformerEncoderLayerOptions(512,
/// 8).dropout(0.1)); TransformerEncoder
/// encoder(TransformerEncoderOptions(encoderLayer,
/// 6).norm(LayerNorm(LayerNormOptions({2}))));
/// ```
class TORCH_API TransformerEncoderImpl
    : public Cloneable<TransformerEncoderImpl> {
 public:
  TransformerEncoderImpl(
      TransformerEncoderLayer encoder_layer,
      int64_t num_layers)
      : TransformerEncoderImpl(
            TransformerEncoderOptions(std::move(encoder_layer), num_layers)) {}
  explicit TransformerEncoderImpl(TransformerEncoderOptions options_);

  Tensor forward(
      const Tensor& src,
      const Tensor& src_mask = {},
      const Tensor& src_key_padding_mask = {});

  void reset() override;

  void reset_parameters();

 protected:
  FORWARD_HAS_DEFAULT_ARGS({1, AnyValue(Tensor())}, {2, AnyValue(Tensor())})

 public:
  /// options with which this `TransformerEncoder` was constructed
  TransformerEncoderOptions options;

  /// module list that contains all the encoder layers
  ModuleList layers = nullptr;

  /// optional normalization module
  AnyModule norm;
};

/// A `ModuleHolder` subclass for `TransformerEncoderImpl`.
/// See the documentation for `TransformerEncoderImpl` class to learn what
/// methods it provides, and examples of how to use `TransformerEncoder` with
/// `torch::nn::TransformerEncoderOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(TransformerEncoder);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TransformerDecoder
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// TransformerDecoder is a stack of N decoder layers.
/// See
/// https://pytorch.org/docs/main/generated/torch.nn.TransformerDecoder.html
/// to learn abouut the exact behavior of this decoder module
///
/// See the documentation for `torch::nn::TransformerDecoderOptions` class to
/// learn what constructor arguments are supported for this decoder module
///
/// Example:
/// ```
/// TransformerDecoderLayer decoder_layer(TransformerDecoderLayerOptions(512,
/// 8).dropout(0.1)); TransformerDecoder
/// transformer_decoder(TransformerDecoderOptions(decoder_layer,
/// 6).norm(LayerNorm(LayerNormOptions({2})))); const auto memory =
/// torch::rand({10, 32, 512}); const auto tgt = torch::rand({20, 32, 512});
/// auto out = transformer_decoder(tgt, memory);
/// ```
class TORCH_API TransformerDecoderImpl
    : public Cloneable<TransformerDecoderImpl> {
 public:
  TransformerDecoderImpl(
      TransformerDecoderLayer decoder_layer,
      int64_t num_layers)
      : TransformerDecoderImpl(
            TransformerDecoderOptions(std::move(decoder_layer), num_layers)) {}
  explicit TransformerDecoderImpl(TransformerDecoderOptions options_);

  void reset() override;

  void reset_parameters();

  /// Pass the inputs (and mask) through the decoder layer in turn.
  /// Args:
  ///       tgt: the sequence to the decoder layer (required).
  ///       memory: the sequence from the last layer of the encoder (required).
  ///       tgt_mask: the mask for the tgt sequence (optional).
  ///       memory_mask: the mask for the memory sequence (optional).
  ///       tgt_key_padding_mask: the mask for the tgt keys per batch
  ///       (optional). memory_key_padding_mask: the mask for the memory keys
  ///       per batch (optional).
  Tensor forward(
      const Tensor& tgt,
      const Tensor& memory,
      const Tensor& tgt_mask = {},
      const Tensor& memory_mask = {},
      const Tensor& tgt_key_padding_mask = {},
      const Tensor& memory_key_padding_mask = {});

  /// The options used to configure this module.
  TransformerDecoderOptions options;

  /// Cloned layers of decoder layers
  ModuleList layers{nullptr};

  /// optional layer normalization module
  AnyModule norm;

 protected:
  FORWARD_HAS_DEFAULT_ARGS(
      {2, AnyValue(Tensor())},
      {3, AnyValue(Tensor())},
      {4, AnyValue(Tensor())},
      {5, AnyValue(Tensor())})
};

/// A `ModuleHolder` subclass for `TransformerDecoderImpl`.
/// See the documentation for `TransformerDecoderImpl` class to learn what
/// methods it provides, and examples of how to use `TransformerDecoder` with
/// `torch::nn::TransformerDecoderOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(TransformerDecoder);

} // namespace torch::nn

```

## High-Level Overview

This file is part of the PyTorch repository. It is a C++/CUDA source/header file that may contain implementations, declarations, or kernel code.

## Detailed Walkthrough

### Classes
This file defines 8 class(es): to, TORCH_API, for, to, to, TORCH_API, for, to


## Key Components

The file contains 508 words across 152 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 5203 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
