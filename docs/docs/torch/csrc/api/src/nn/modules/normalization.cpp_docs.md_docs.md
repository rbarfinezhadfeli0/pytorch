# Documentation: `docs/torch/csrc/api/src/nn/modules/normalization.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/api/src/nn/modules/normalization.cpp_docs.md`
- **Size**: 6,324 bytes (6.18 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/api/src/nn/modules/normalization.cpp`

## File Metadata

- **Path**: `torch/csrc/api/src/nn/modules/normalization.cpp`
- **Size**: 3,877 bytes (3.79 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/nn/modules/normalization.h>

#include <torch/cuda.h>
#include <torch/nn/init.h>
#include <torch/utils.h>

#include <ostream>
#include <utility>

namespace F = torch::nn::functional;

namespace torch::nn {

LayerNormImpl::LayerNormImpl(LayerNormOptions options_)
    : options(std::move(options_)) {
  LayerNormImpl::reset();
}

void LayerNormImpl::reset() {
  if (options.elementwise_affine()) {
    weight =
        register_parameter("weight", torch::empty(options.normalized_shape()));
    bias = register_parameter("bias", torch::empty(options.normalized_shape()));
  } else {
    weight =
        register_parameter("weight", torch::Tensor(), /*requires_grad=*/false);
    bias = register_parameter("bias", torch::Tensor(), /*requires_grad=*/false);
  }
  reset_parameters();
}

void LayerNormImpl::reset_parameters() {
  if (options.elementwise_affine()) {
    torch::nn::init::ones_(weight);
    torch::nn::init::zeros_(bias);
  }
}

void LayerNormImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha << "torch::nn::LayerNorm("
         << torch::IntArrayRef(options.normalized_shape())
         << ", eps=" << options.eps()
         << ", elementwise_affine=" << options.elementwise_affine() << ")";
}

torch::Tensor LayerNormImpl::forward(const Tensor& input) {
  return F::detail::layer_norm(
      input, options.normalized_shape(), weight, bias, options.eps());
}

// ============================================================================

LocalResponseNormImpl::LocalResponseNormImpl(
    const LocalResponseNormOptions& options_)
    : options(options_) {}

Tensor LocalResponseNormImpl::forward(const Tensor& input) {
  return F::detail::local_response_norm(
      input, options.size(), options.alpha(), options.beta(), options.k());
}

void LocalResponseNormImpl::reset() {}

void LocalResponseNormImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha << "torch::nn::LocalResponseNorm(" << options.size()
         << ", alpha=" << options.alpha() << ", beta=" << options.beta()
         << ", k=" << options.k() << ")";
}

// ============================================================================

void CrossMapLRN2dImpl::reset() {}

void CrossMapLRN2dImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha << "torch::nn::CrossMapLRN2d(" << options.size()
         << ", alpha=" << options.alpha() << ", beta=" << options.beta()
         << ", k=" << options.k() << ")";
}

torch::Tensor CrossMapLRN2dImpl::forward(const torch::Tensor& input) {
  return functions::CrossMapLRN2d::apply(input, options);
}

// ============================================================================

GroupNormImpl::GroupNormImpl(const GroupNormOptions& options_)
    : options(options_) { // NOLINT(modernize-pass-by-value)
  GroupNormImpl::reset();
}

void GroupNormImpl::reset() {
  if (options.affine()) {
    weight = register_parameter("weight", torch::empty(options.num_channels()));
    bias = register_parameter("bias", torch::empty(options.num_channels()));
  } else {
    weight =
        register_parameter("weight", torch::Tensor(), /*requires_grad=*/false);
    bias = register_parameter("bias", torch::Tensor(), /*requires_grad=*/false);
  }
  reset_parameters();
}

void GroupNormImpl::reset_parameters() {
  if (options.affine()) {
    torch::nn::init::ones_(weight);
    torch::nn::init::zeros_(bias);
  }
}

torch::Tensor GroupNormImpl::forward(const Tensor& input) {
  return F::detail::group_norm(
      input, options.num_groups(), weight, bias, options.eps());
}

void GroupNormImpl::pretty_print(std::ostream& stream) const {
  stream << std::boolalpha << "torch::nn::GroupNorm(" << options.num_groups()
         << ", " << options.num_channels() << ", eps=" << options.eps()
         << ", affine=" << options.affine() << ")";
}

} // namespace torch::nn

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 0 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `F`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/src/nn/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/nn/modules/normalization.h`
- `torch/cuda.h`
- `torch/nn/init.h`
- `torch/utils.h`
- `ostream`
- `utility`


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/csrc/api/src/nn/modules`):

- [`pooling.cpp_docs.md`](./pooling.cpp_docs.md)
- [`linear.cpp_docs.md`](./linear.cpp_docs.md)
- [`padding.cpp_docs.md`](./padding.cpp_docs.md)
- [`rnn.cpp_docs.md`](./rnn.cpp_docs.md)
- [`upsampling.cpp_docs.md`](./upsampling.cpp_docs.md)
- [`embedding.cpp_docs.md`](./embedding.cpp_docs.md)
- [`dropout.cpp_docs.md`](./dropout.cpp_docs.md)
- [`pixelshuffle.cpp_docs.md`](./pixelshuffle.cpp_docs.md)
- [`loss.cpp_docs.md`](./loss.cpp_docs.md)
- [`fold.cpp_docs.md`](./fold.cpp_docs.md)


## Cross-References

- **File Documentation**: `normalization.cpp_docs.md`
- **Keyword Index**: `normalization.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/api/src/nn/modules`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/api/src/nn/modules`, which is part of the **core PyTorch library**.



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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/csrc/api/src/nn/modules`):

- [`dropout.cpp_docs.md_docs.md`](./dropout.cpp_docs.md_docs.md)
- [`dropout.cpp_kw.md_docs.md`](./dropout.cpp_kw.md_docs.md)
- [`pooling.cpp_kw.md_docs.md`](./pooling.cpp_kw.md_docs.md)
- [`rnn.cpp_docs.md_docs.md`](./rnn.cpp_docs.md_docs.md)
- [`linear.cpp_docs.md_docs.md`](./linear.cpp_docs.md_docs.md)
- [`normalization.cpp_kw.md_docs.md`](./normalization.cpp_kw.md_docs.md)
- [`_functions.cpp_kw.md_docs.md`](./_functions.cpp_kw.md_docs.md)
- [`upsampling.cpp_docs.md_docs.md`](./upsampling.cpp_docs.md_docs.md)
- [`pixelshuffle.cpp_docs.md_docs.md`](./pixelshuffle.cpp_docs.md_docs.md)
- [`adaptive.cpp_kw.md_docs.md`](./adaptive.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `normalization.cpp_docs.md_docs.md`
- **Keyword Index**: `normalization.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
