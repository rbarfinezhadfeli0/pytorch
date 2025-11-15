# Documentation: `docs/torch/csrc/api/include/torch/nn/modules/instancenorm.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/api/include/torch/nn/modules/instancenorm.h_docs.md`
- **Size**: 8,054 bytes (7.87 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/api/include/torch/nn/modules/instancenorm.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/nn/modules/instancenorm.h`
- **Size**: 5,514 bytes (5.38 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/nn/functional/instancenorm.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/options/instancenorm.h>

namespace torch::nn {

/// Base class for all (dimension-specialized) instance norm modules
template <size_t D, typename Derived>
// NOLINTNEXTLINE(bugprone-crtp-constructor-accessibility)
class InstanceNormImpl
    : public torch::nn::NormImplBase<D, Derived, InstanceNormOptions> {
 private:
  inline Tensor apply_instance_norm(const Tensor& input) {
    return torch::nn::functional::detail::instance_norm(
        input,
        this->running_mean,
        this->running_var,
        this->weight,
        this->bias,
        this->is_training() || !this->options.track_running_stats(),
        this->options.momentum(),
        this->options.eps());
  }

  inline Tensor handle_no_batch_input(const Tensor& input) {
    return this->apply_instance_norm(input.unsqueeze(0)).squeeze(0);
  }

 public:
  using torch::nn::NormImplBase<D, Derived, InstanceNormOptions>::NormImplBase;

  Tensor forward(const Tensor& input) {
    this->_check_input_dim(input);

    // For InstanceNorm1D, 2D is unbatched and 3D is batched
    // For InstanceNorm2D, 3D is unbatched and 4D is batched
    // For InstanceNorm3D, 4D is unbatched and 5D is batched
    // check if input does not have a batch-dim
    if (input.dim() == D + 1) {
      return this->handle_no_batch_input(input);
    }

    return this->apply_instance_norm(input);
  }

  /// Pretty prints the `InstanceNorm{1,2,3}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override {
    stream << std::boolalpha << "torch::nn::InstanceNorm" << D << "d("
           << this->options.num_features() << ", "
           << "eps=" << this->options.eps() << ", "
           << "momentum=" << this->options.momentum() << ", "
           << "affine=" << this->options.affine() << ", "
           << "track_running_stats=" << this->options.track_running_stats()
           << ")";
  }
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ InstanceNorm1d
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the InstanceNorm1d function.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.InstanceNorm1d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::InstanceNorm1dOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// InstanceNorm1d
/// model(InstanceNorm1dOptions(4).eps(0.5).momentum(0.1).affine(false).track_running_stats(true));
/// ```
class TORCH_API InstanceNorm1dImpl
    : public InstanceNormImpl<1, InstanceNorm1dImpl> {
 protected:
  void _check_input_dim(const Tensor& input) override;

 public:
  using InstanceNormImpl<1, InstanceNorm1dImpl>::InstanceNormImpl;
};

/// A `ModuleHolder` subclass for `InstanceNorm1dImpl`.
/// See the documentation for `InstanceNorm1dImpl` class to learn what methods
/// it provides, and examples of how to use `InstanceNorm1d` with
/// `torch::nn::InstanceNorm1dOptions`. See the documentation for `ModuleHolder`
/// to learn about PyTorch's module storage semantics.
TORCH_MODULE(InstanceNorm1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ InstanceNorm2d
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the InstanceNorm2d function.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.InstanceNorm2d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::InstanceNorm2dOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// InstanceNorm2d
/// model(InstanceNorm2dOptions(4).eps(0.5).momentum(0.1).affine(false).track_running_stats(true));
/// ```
class TORCH_API InstanceNorm2dImpl
    : public InstanceNormImpl<2, InstanceNorm2dImpl> {
 protected:
  void _check_input_dim(const Tensor& input) override;

 public:
  using InstanceNormImpl<2, InstanceNorm2dImpl>::InstanceNormImpl;
};

/// A `ModuleHolder` subclass for `InstanceNorm2dImpl`.
/// See the documentation for `InstanceNorm2dImpl` class to learn what methods
/// it provides, and examples of how to use `InstanceNorm2d` with
/// `torch::nn::InstanceNorm2dOptions`. See the documentation for `ModuleHolder`
/// to learn about PyTorch's module storage semantics.
TORCH_MODULE(InstanceNorm2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ InstanceNorm3d
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies the InstanceNorm3d function.
/// See https://pytorch.org/docs/main/nn.html#torch.nn.InstanceNorm3d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::InstanceNorm3dOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// InstanceNorm3d
/// model(InstanceNorm3dOptions(4).eps(0.5).momentum(0.1).affine(false).track_running_stats(true));
/// ```
class TORCH_API InstanceNorm3dImpl
    : public InstanceNormImpl<3, InstanceNorm3dImpl> {
 protected:
  void _check_input_dim(const Tensor& input) override;

 public:
  using InstanceNormImpl<3, InstanceNorm3dImpl>::InstanceNormImpl;
};

/// A `ModuleHolder` subclass for `InstanceNorm3dImpl`.
/// See the documentation for `InstanceNorm3dImpl` class to learn what methods
/// it provides, and examples of how to use `InstanceNorm3d` with
/// `torch::nn::InstanceNorm3dOptions`. See the documentation for `ModuleHolder`
/// to learn about PyTorch's module storage semantics.
TORCH_MODULE(InstanceNorm3d);

} // namespace torch::nn

```



## High-Level Overview


This C++ file contains approximately 14 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `for`, `InstanceNormImpl`, `to`, `TORCH_API`, `for`, `to`, `to`, `TORCH_API`, `for`, `to`, `to`, `TORCH_API`, `for`, `to`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/include/torch/nn/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/nn/functional/instancenorm.h`
- `torch/nn/modules/batchnorm.h`
- `torch/nn/options/instancenorm.h`


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
- [`batchnorm.h_docs.md`](./batchnorm.h_docs.md)
- [`adaptive.h_docs.md`](./adaptive.h_docs.md)
- [`activation.h_docs.md`](./activation.h_docs.md)
- [`padding.h_docs.md`](./padding.h_docs.md)
- [`utils.h_docs.md`](./utils.h_docs.md)


## Cross-References

- **File Documentation**: `instancenorm.h_docs.md`
- **Keyword Index**: `instancenorm.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/api/include/torch/nn/modules`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/api/include/torch/nn/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/csrc/api/include/torch/nn/modules`):

- [`transformerlayer.h_docs.md_docs.md`](./transformerlayer.h_docs.md_docs.md)
- [`linear.h_kw.md_docs.md`](./linear.h_kw.md_docs.md)
- [`transformercoder.h_docs.md_docs.md`](./transformercoder.h_docs.md_docs.md)
- [`loss.h_kw.md_docs.md`](./loss.h_kw.md_docs.md)
- [`batchnorm.h_docs.md_docs.md`](./batchnorm.h_docs.md_docs.md)
- [`normalization.h_kw.md_docs.md`](./normalization.h_kw.md_docs.md)
- [`fold.h_docs.md_docs.md`](./fold.h_docs.md_docs.md)
- [`distance.h_kw.md_docs.md`](./distance.h_kw.md_docs.md)
- [`utils.h_docs.md_docs.md`](./utils.h_docs.md_docs.md)
- [`transformercoder.h_kw.md_docs.md`](./transformercoder.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `instancenorm.h_docs.md_docs.md`
- **Keyword Index**: `instancenorm.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
