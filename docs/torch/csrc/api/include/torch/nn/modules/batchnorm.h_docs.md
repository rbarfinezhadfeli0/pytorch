# Documentation: `torch/csrc/api/include/torch/nn/modules/batchnorm.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/nn/modules/batchnorm.h`
- **Size**: 8,225 bytes (8.03 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/functional/batchnorm.h>
#include <torch/nn/init.h>
#include <torch/nn/options/batchnorm.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

namespace torch::nn {

/// Base class for all (dimension-specialized) batchnorm and instancenorm
/// modules.
template <size_t D, typename Derived, typename DerivedOptions>
class NormImplBase : public torch::nn::Cloneable<Derived> {
 protected:
  virtual void _check_input_dim(const Tensor& input) = 0;

 public:
  NormImplBase(const DerivedOptions& options_) : options(options_) {
    NormImplBase::reset();
  }

  void reset() override {
    if (options.affine()) {
      weight = this->register_parameter(
          "weight", torch::empty({options.num_features()}));
      bias = this->register_parameter(
          "bias", torch::empty({options.num_features()}));
    } else {
      weight =
          this->register_parameter("weight", Tensor(), /*requires_grad=*/false);
      bias =
          this->register_parameter("bias", Tensor(), /*requires_grad=*/false);
    }
    if (options.track_running_stats()) {
      running_mean = this->register_buffer(
          "running_mean", torch::zeros({options.num_features()}));
      running_var = this->register_buffer(
          "running_var", torch::ones({options.num_features()}));
      num_batches_tracked = this->register_buffer(
          "num_batches_tracked", torch::tensor(0, torch::dtype(torch::kLong)));
    } else {
      running_mean = this->register_buffer("running_mean", Tensor());
      running_var = this->register_buffer("running_var", Tensor());
      num_batches_tracked =
          this->register_buffer("num_batches_tracked", Tensor());
    }
    reset_parameters();
  }

  void reset_running_stats() {
    if (options.track_running_stats()) {
      running_mean.zero_();
      running_var.fill_(1);
      num_batches_tracked.zero_();
    }
  }

  void reset_parameters() {
    reset_running_stats();
    if (options.affine()) {
      torch::nn::init::ones_(weight);
      torch::nn::init::zeros_(bias);
    }
  }

  /// The options with which this module was constructed.
  DerivedOptions options;

  /// The learned weight.
  /// Only defined if the `affine` option was `true` upon construction.
  Tensor weight;

  /// The learned bias.
  /// Only defined if the `affine` option was `true` upon construction.
  Tensor bias;

  /// The running mean.
  /// Only defined if the `track_running_stats` option was `true` upon
  /// construction.
  Tensor running_mean;

  /// The running variance.
  /// Only defined if the `track_running_stats` option was `true` upon
  /// construction.
  Tensor running_var;

  /// The number of the forward call.
  /// Only defined if the `track_running_stats` option was `true` upon
  /// construction.
  Tensor num_batches_tracked;
};

/// Base class for all (dimension-specialized) batchnorm modules.
template <size_t D, typename Derived>
class BatchNormImplBase : public NormImplBase<D, Derived, BatchNormOptions> {
 public:
  using NormImplBase<D, Derived, BatchNormOptions>::NormImplBase;

  Tensor forward(const Tensor& input) {
    this->_check_input_dim(input);
    double exponential_average_factor = 0.0;
    if (this->options.momentum().has_value()) {
      exponential_average_factor = this->options.momentum().value();
    }

    if (this->is_training() && this->options.track_running_stats()) {
      if (this->num_batches_tracked.defined()) {
        this->num_batches_tracked += 1;
        if (this->options.momentum() ==
            std::nullopt) { // use cumulative moving average
          exponential_average_factor =
              1.0 / this->num_batches_tracked.template item<double>();
        } else { // use exponential moving average
          exponential_average_factor = this->options.momentum().value();
        }
      }
    }

    return torch::nn::functional::detail::batch_norm(
        input,
        this->running_mean,
        this->running_var,
        this->weight,
        this->bias,
        this->is_training() || !this->options.track_running_stats(),
        /*momentum=*/exponential_average_factor,
        this->options.eps());
  }

  /// Pretty prints the `BatchNorm{1,2,3}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override {
    stream << std::boolalpha << "torch::nn::BatchNorm" << D << "d("
           << this->options.num_features() << ", "
           << "eps=" << this->options.eps() << ", "
           << "momentum=";

    if (this->options.momentum().has_value()) {
      stream << this->options.momentum().value();
    } else {
      stream << "None";
    }

    stream << ", "
           << "affine=" << this->options.affine() << ", "
           << "track_running_stats=" << this->options.track_running_stats()
           << ")";
  }
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BatchNorm1d
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the BatchNorm1d function.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.BatchNorm1d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::BatchNorm1dOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// BatchNorm1d
/// model(BatchNorm1dOptions(4).eps(0.5).momentum(0.1).affine(false).track_running_stats(true));
/// ```
class TORCH_API BatchNorm1dImpl : public BatchNormImplBase<1, BatchNorm1dImpl> {
 protected:
  void _check_input_dim(const Tensor& input) override;

 public:
  using BatchNormImplBase<1, BatchNorm1dImpl>::BatchNormImplBase;
};

/// A `ModuleHolder` subclass for `BatchNorm1dImpl`.
/// See the documentation for `BatchNorm1dImpl` class to learn what methods it
/// provides, and examples of how to use `BatchNorm1d` with
/// `torch::nn::BatchNorm1dOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(BatchNorm1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BatchNorm2d
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the BatchNorm2d function.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.BatchNorm2d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::BatchNorm2dOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// BatchNorm2d
/// model(BatchNorm2dOptions(4).eps(0.5).momentum(0.1).affine(false).track_running_stats(true));
/// ```
class TORCH_API BatchNorm2dImpl : public BatchNormImplBase<2, BatchNorm2dImpl> {
 protected:
  void _check_input_dim(const Tensor& input) override;

 public:
  using BatchNormImplBase<2, BatchNorm2dImpl>::BatchNormImplBase;
};

/// A `ModuleHolder` subclass for `BatchNorm2dImpl`.
/// See the documentation for `BatchNorm2dImpl` class to learn what methods it
/// provides, and examples of how to use `BatchNorm2d` with
/// `torch::nn::BatchNorm2dOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(BatchNorm2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BatchNorm3d
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the BatchNorm3d function.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.BatchNorm3d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::BatchNorm3dOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// BatchNorm3d
/// model(BatchNorm3dOptions(4).eps(0.5).momentum(0.1).affine(false).track_running_stats(true));
/// ```
class TORCH_API BatchNorm3dImpl : public BatchNormImplBase<3, BatchNorm3dImpl> {
 protected:
  void _check_input_dim(const Tensor& input) override;

 public:
  using BatchNormImplBase<3, BatchNorm3dImpl>::BatchNormImplBase;
};

/// A `ModuleHolder` subclass for `BatchNorm3dImpl`.
/// See the documentation for `BatchNorm3dImpl` class to learn what methods it
/// provides, and examples of how to use `BatchNorm3d` with
/// `torch::nn::BatchNorm3dOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(BatchNorm3d);

} // namespace torch::nn

```



## High-Level Overview


This C++ file contains approximately 16 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `for`, `NormImplBase`, `for`, `BatchNormImplBase`, `to`, `TORCH_API`, `for`, `to`, `to`, `TORCH_API`, `for`, `to`, `to`, `TORCH_API`, `for`, `to`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/include/torch/nn/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/nn/cloneable.h`
- `torch/nn/functional/batchnorm.h`
- `torch/nn/init.h`
- `torch/nn/options/batchnorm.h`
- `torch/nn/pimpl.h`
- `torch/types.h`


## Code Patterns & Idioms

### Common Patterns

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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/csrc/api/include/torch/nn/modules`):

- [`embedding.h_docs.md`](./embedding.h_docs.md)
- [`normalization.h_docs.md`](./normalization.h_docs.md)
- [`loss.h_docs.md`](./loss.h_docs.md)
- [`distance.h_docs.md`](./distance.h_docs.md)
- [`common.h_docs.md`](./common.h_docs.md)
- [`adaptive.h_docs.md`](./adaptive.h_docs.md)
- [`activation.h_docs.md`](./activation.h_docs.md)
- [`padding.h_docs.md`](./padding.h_docs.md)
- [`utils.h_docs.md`](./utils.h_docs.md)


## Cross-References

- **File Documentation**: `batchnorm.h_docs.md`
- **Keyword Index**: `batchnorm.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
