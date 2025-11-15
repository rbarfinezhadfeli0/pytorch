# Documentation: `docs/torch/csrc/api/include/torch/nn/options/rnn.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/api/include/torch/nn/options/rnn.h_docs.md`
- **Size**: 10,644 bytes (10.39 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/api/include/torch/nn/options/rnn.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/nn/options/rnn.h`
- **Size**: 8,195 bytes (8.00 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/arg.h>
#include <torch/csrc/Export.h>
#include <torch/enum.h>
#include <torch/types.h>

namespace torch::nn {

namespace detail {

/// Common options for RNN, LSTM and GRU modules.
struct TORCH_API RNNOptionsBase {
  typedef std::variant<
      enumtype::kLSTM,
      enumtype::kGRU,
      enumtype::kRNN_TANH,
      enumtype::kRNN_RELU>
      rnn_options_base_mode_t;

  RNNOptionsBase(
      rnn_options_base_mode_t mode,
      int64_t input_size,
      int64_t hidden_size);

  TORCH_ARG(rnn_options_base_mode_t, mode);
  /// The number of features of a single sample in the input sequence `x`.
  TORCH_ARG(int64_t, input_size);
  /// The number of features in the hidden state `h`.
  TORCH_ARG(int64_t, hidden_size);
  /// The number of recurrent layers (cells) to use.
  TORCH_ARG(int64_t, num_layers) = 1;
  /// Whether a bias term should be added to all linear operations.
  TORCH_ARG(bool, bias) = true;
  /// If true, the input sequence should be provided as `(batch, sequence,
  /// features)`. If false (default), the expected layout is `(sequence, batch,
  /// features)`.
  TORCH_ARG(bool, batch_first) = false;
  /// If non-zero, adds dropout with the given probability to the output of each
  /// RNN layer, except the final layer.
  TORCH_ARG(double, dropout) = 0.0;
  /// Whether to make the RNN bidirectional.
  TORCH_ARG(bool, bidirectional) = false;
  /// Cell projection dimension. If 0, projections are not added. Can only be
  /// used for LSTMs.
  TORCH_ARG(int64_t, proj_size) = 0;
};

} // namespace detail

/// Options for the `RNN` module.
///
/// Example:
/// ```
/// RNN model(RNNOptions(128,
/// 64).num_layers(3).dropout(0.2).nonlinearity(torch::kTanh));
/// ```
struct TORCH_API RNNOptions {
  typedef std::variant<enumtype::kTanh, enumtype::kReLU> nonlinearity_t;

  RNNOptions(int64_t input_size, int64_t hidden_size);

  /// The number of expected features in the input `x`
  TORCH_ARG(int64_t, input_size);
  /// The number of features in the hidden state `h`
  TORCH_ARG(int64_t, hidden_size);
  /// Number of recurrent layers. E.g., setting ``num_layers=2``
  /// would mean stacking two RNNs together to form a `stacked RNN`,
  /// with the second RNN taking in outputs of the first RNN and
  /// computing the final results. Default: 1
  TORCH_ARG(int64_t, num_layers) = 1;
  /// The non-linearity to use. Can be either ``torch::kTanh`` or
  /// ``torch::kReLU``. Default: ``torch::kTanh``
  TORCH_ARG(nonlinearity_t, nonlinearity) = torch::kTanh;
  /// If ``false``, then the layer does not use bias weights `b_ih` and `b_hh`.
  /// Default: ``true``
  TORCH_ARG(bool, bias) = true;
  /// If ``true``, then the input and output tensors are provided
  /// as `(batch, seq, feature)`. Default: ``false``
  TORCH_ARG(bool, batch_first) = false;
  /// If non-zero, introduces a `Dropout` layer on the outputs of each
  /// RNN layer except the last layer, with dropout probability equal to
  /// `dropout`. Default: 0
  TORCH_ARG(double, dropout) = 0.0;
  /// If ``true``, becomes a bidirectional RNN. Default: ``false``
  TORCH_ARG(bool, bidirectional) = false;
};

/// Options for the `LSTM` module.
///
/// Example:
/// ```
/// LSTM model(LSTMOptions(2,
/// 4).num_layers(3).batch_first(false).bidirectional(true));
/// ```
struct TORCH_API LSTMOptions {
  LSTMOptions(int64_t input_size, int64_t hidden_size);

  /// The number of expected features in the input `x`
  TORCH_ARG(int64_t, input_size);
  /// The number of features in the hidden state `h`
  TORCH_ARG(int64_t, hidden_size);
  /// Number of recurrent layers. E.g., setting ``num_layers=2``
  /// would mean stacking two LSTMs together to form a `stacked LSTM`,
  /// with the second LSTM taking in outputs of the first LSTM and
  /// computing the final results. Default: 1
  TORCH_ARG(int64_t, num_layers) = 1;
  /// If ``false``, then the layer does not use bias weights `b_ih` and `b_hh`.
  /// Default: ``true``
  TORCH_ARG(bool, bias) = true;
  /// If ``true``, then the input and output tensors are provided
  /// as (batch, seq, feature). Default: ``false``
  TORCH_ARG(bool, batch_first) = false;
  /// If non-zero, introduces a `Dropout` layer on the outputs of each
  /// LSTM layer except the last layer, with dropout probability equal to
  /// `dropout`. Default: 0
  TORCH_ARG(double, dropout) = 0.0;
  /// If ``true``, becomes a bidirectional LSTM. Default: ``false``
  TORCH_ARG(bool, bidirectional) = false;
  /// Cell projection dimension. If 0, projections are not added
  TORCH_ARG(int64_t, proj_size) = 0;
};

/// Options for the `GRU` module.
///
/// Example:
/// ```
/// GRU model(GRUOptions(2,
/// 4).num_layers(3).batch_first(false).bidirectional(true));
/// ```
struct TORCH_API GRUOptions {
  GRUOptions(int64_t input_size, int64_t hidden_size);

  /// The number of expected features in the input `x`
  TORCH_ARG(int64_t, input_size);
  /// The number of features in the hidden state `h`
  TORCH_ARG(int64_t, hidden_size);
  /// Number of recurrent layers. E.g., setting ``num_layers=2``
  /// would mean stacking two GRUs together to form a `stacked GRU`,
  /// with the second GRU taking in outputs of the first GRU and
  /// computing the final results. Default: 1
  TORCH_ARG(int64_t, num_layers) = 1;
  /// If ``false``, then the layer does not use bias weights `b_ih` and `b_hh`.
  /// Default: ``true``
  TORCH_ARG(bool, bias) = true;
  /// If ``true``, then the input and output tensors are provided
  /// as (batch, seq, feature). Default: ``false``
  TORCH_ARG(bool, batch_first) = false;
  /// If non-zero, introduces a `Dropout` layer on the outputs of each
  /// GRU layer except the last layer, with dropout probability equal to
  /// `dropout`. Default: 0
  TORCH_ARG(double, dropout) = 0.0;
  /// If ``true``, becomes a bidirectional GRU. Default: ``false``
  TORCH_ARG(bool, bidirectional) = false;
};

namespace detail {

/// Common options for RNNCell, LSTMCell and GRUCell modules
struct TORCH_API RNNCellOptionsBase {
  RNNCellOptionsBase(
      int64_t input_size,
      int64_t hidden_size,
      bool bias,
      int64_t num_chunks);
  TORCH_ARG(int64_t, input_size);
  TORCH_ARG(int64_t, hidden_size);
  TORCH_ARG(bool, bias);
  TORCH_ARG(int64_t, num_chunks);
};

} // namespace detail

/// Options for the `RNNCell` module.
///
/// Example:
/// ```
/// RNNCell model(RNNCellOptions(20,
/// 10).bias(false).nonlinearity(torch::kReLU));
/// ```
struct TORCH_API RNNCellOptions {
  typedef std::variant<enumtype::kTanh, enumtype::kReLU> nonlinearity_t;

  RNNCellOptions(int64_t input_size, int64_t hidden_size);

  /// The number of expected features in the input `x`
  TORCH_ARG(int64_t, input_size);
  /// The number of features in the hidden state `h`
  TORCH_ARG(int64_t, hidden_size);
  /// If ``false``, then the layer does not use bias weights `b_ih` and `b_hh`.
  /// Default: ``true``
  TORCH_ARG(bool, bias) = true;
  /// The non-linearity to use. Can be either ``torch::kTanh`` or
  /// ``torch::kReLU``. Default: ``torch::kTanh``
  TORCH_ARG(nonlinearity_t, nonlinearity) = torch::kTanh;
};

/// Options for the `LSTMCell` module.
///
/// Example:
/// ```
/// LSTMCell model(LSTMCellOptions(20, 10).bias(false));
/// ```
struct TORCH_API LSTMCellOptions {
  LSTMCellOptions(int64_t input_size, int64_t hidden_size);

  /// The number of expected features in the input `x`
  TORCH_ARG(int64_t, input_size);
  /// The number of features in the hidden state `h`
  TORCH_ARG(int64_t, hidden_size);
  /// If ``false``, then the layer does not use bias weights `b_ih` and `b_hh`.
  /// Default: ``true``
  TORCH_ARG(bool, bias) = true;
};

/// Options for the `GRUCell` module.
///
/// Example:
/// ```
/// GRUCell model(GRUCellOptions(20, 10).bias(false));
/// ```
struct TORCH_API GRUCellOptions {
  GRUCellOptions(int64_t input_size, int64_t hidden_size);

  /// The number of expected features in the input `x`
  TORCH_ARG(int64_t, input_size);
  /// The number of features in the hidden state `h`
  TORCH_ARG(int64_t, hidden_size);
  /// If ``false``, then the layer does not use bias weights `b_ih` and `b_hh`.
  /// Default: ``true``
  TORCH_ARG(bool, bias) = true;
};

} // namespace torch::nn

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 15 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `detail`

**Classes/Structs**: `TORCH_API`, `TORCH_API`, `TORCH_API`, `TORCH_API`, `TORCH_API`, `TORCH_API`, `TORCH_API`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/include/torch/nn/options`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/arg.h`
- `torch/csrc/Export.h`
- `torch/enum.h`
- `torch/types.h`


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

Files in the same folder (`torch/csrc/api/include/torch/nn/options`):

- [`embedding.h_docs.md`](./embedding.h_docs.md)
- [`normalization.h_docs.md`](./normalization.h_docs.md)
- [`loss.h_docs.md`](./loss.h_docs.md)
- [`distance.h_docs.md`](./distance.h_docs.md)
- [`batchnorm.h_docs.md`](./batchnorm.h_docs.md)
- [`adaptive.h_docs.md`](./adaptive.h_docs.md)
- [`activation.h_docs.md`](./activation.h_docs.md)
- [`padding.h_docs.md`](./padding.h_docs.md)
- [`linear.h_docs.md`](./linear.h_docs.md)
- [`transformer.h_docs.md`](./transformer.h_docs.md)


## Cross-References

- **File Documentation**: `rnn.h_docs.md`
- **Keyword Index**: `rnn.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/api/include/torch/nn/options`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/api/include/torch/nn/options`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/api/include/torch/nn/options`):

- [`transformerlayer.h_docs.md_docs.md`](./transformerlayer.h_docs.md_docs.md)
- [`linear.h_kw.md_docs.md`](./linear.h_kw.md_docs.md)
- [`transformercoder.h_docs.md_docs.md`](./transformercoder.h_docs.md_docs.md)
- [`loss.h_kw.md_docs.md`](./loss.h_kw.md_docs.md)
- [`batchnorm.h_docs.md_docs.md`](./batchnorm.h_docs.md_docs.md)
- [`normalization.h_kw.md_docs.md`](./normalization.h_kw.md_docs.md)
- [`fold.h_docs.md_docs.md`](./fold.h_docs.md_docs.md)
- [`distance.h_kw.md_docs.md`](./distance.h_kw.md_docs.md)
- [`transformercoder.h_kw.md_docs.md`](./transformercoder.h_kw.md_docs.md)
- [`instancenorm.h_docs.md_docs.md`](./instancenorm.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `rnn.h_docs.md_docs.md`
- **Keyword Index**: `rnn.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
