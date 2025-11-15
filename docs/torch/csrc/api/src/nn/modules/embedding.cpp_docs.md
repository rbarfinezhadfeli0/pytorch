# Documentation: `torch/csrc/api/src/nn/modules/embedding.cpp`

## File Metadata

- **Path**: `torch/csrc/api/src/nn/modules/embedding.cpp`
- **Size**: 5,845 bytes (5.71 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/nn/modules/embedding.h>

#include <torch/nn/init.h>
#include <torch/types.h>
#include <torch/utils.h>

#include <ostream>
#include <utility>

namespace F = torch::nn::functional;

namespace torch::nn {
EmbeddingImpl::EmbeddingImpl(EmbeddingOptions options_)
    : options(std::move(options_)) {
  EmbeddingImpl::reset();
}

void EmbeddingImpl::reset() {
  if (options.padding_idx().has_value()) {
    if (options.padding_idx() > 0) {
      TORCH_CHECK(
          options.padding_idx() < options.num_embeddings(),
          "Padding_idx must be within num_embeddings");
    } else if (options.padding_idx() < 0) {
      TORCH_CHECK(
          options.padding_idx() >= -options.num_embeddings(),
          "Padding_idx must be within num_embedding");
      options.padding_idx(
          // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
          options.num_embeddings() + *options.padding_idx());
    }
  }

  if (!options._weight().defined()) {
    weight = register_parameter(
        "weight",
        torch::empty({options.num_embeddings(), options.embedding_dim()}));
    reset_parameters();
  } else {
    TORCH_CHECK(
        options._weight().sizes() ==
            torch::IntArrayRef(
                {options.num_embeddings(), options.embedding_dim()}),
        "Shape of _weight does not match num_embeddings and embedding_dim");
    weight = register_parameter("weight", options._weight());
  }
}

void EmbeddingImpl::reset_parameters() {
  torch::nn::init::normal_(weight);
  if (options.padding_idx().has_value()) {
    torch::NoGradGuard no_grad;
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    weight[*options.padding_idx()].fill_(0);
  }
}

void EmbeddingImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Embedding(num_embeddings=" << options.num_embeddings()
         << ", embedding_dim=" << options.embedding_dim();
  auto const& padding_idx_opt = options.padding_idx();
  if (padding_idx_opt.has_value()) {
    stream << ", padding_idx=" << padding_idx_opt.value();
  }
  auto const& max_norm_opt = options.max_norm();
  if (max_norm_opt.has_value()) {
    stream << ", max_norm=" << max_norm_opt.value();
  }
  if (options.norm_type() != 2) {
    stream << ", norm_type=" << options.norm_type();
  }
  if (options.scale_grad_by_freq()) {
    stream << ", scale_grad_by_freq=" << std::boolalpha
           << options.scale_grad_by_freq();
  }
  if (options.sparse()) {
    stream << ", sparse=" << std::boolalpha << options.sparse();
  }
  stream << ")";
}

torch::Tensor EmbeddingImpl::forward(const Tensor& input) {
  return F::detail::embedding(
      input,
      weight,
      options.padding_idx(),
      options.max_norm(),
      options.norm_type(),
      options.scale_grad_by_freq(),
      options.sparse());
}

EmbeddingBagImpl::EmbeddingBagImpl(EmbeddingBagOptions options_)
    : options(std::move(options_)) {
  EmbeddingBagImpl::reset();
}

void EmbeddingBagImpl::reset() {
  auto const& padding_idx_opt = options.padding_idx();
  if (padding_idx_opt.has_value()) {
    auto padding_idx = padding_idx_opt.value();
    if (padding_idx > 0) {
      TORCH_CHECK(
          padding_idx < options.num_embeddings(),
          "Padding_idx must be within num_embeddings");
    } else if (padding_idx < 0) {
      TORCH_CHECK(
          padding_idx >= -options.num_embeddings(),
          "Padding_idx must be within num_embedding");
      options.padding_idx(options.num_embeddings() + padding_idx);
    }
  }
  if (!options._weight().defined()) {
    weight = register_parameter(
        "weight",
        torch::empty({options.num_embeddings(), options.embedding_dim()}));
    reset_parameters();
  } else {
    TORCH_CHECK(
        options._weight().sizes() ==
            torch::IntArrayRef(
                {options.num_embeddings(), options.embedding_dim()}),
        "Shape of weight does not match num_embeddings and embedding_dim");
    weight = register_parameter("weight", options._weight());
  }
}

void EmbeddingBagImpl::reset_parameters() {
  auto const& padding_idx_opt = options.padding_idx();
  if (padding_idx_opt.has_value()) {
    torch::NoGradGuard no_grad;
    weight[*padding_idx_opt].fill_(0);
  }
  torch::nn::init::normal_(weight);
}

torch::Tensor EmbeddingBagImpl::forward(
    const Tensor& input,
    const Tensor& offsets,
    const Tensor& per_sample_weights) {
  return F::detail::embedding_bag(
      input,
      weight,
      offsets,
      options.max_norm(),
      options.norm_type(),
      options.scale_grad_by_freq(),
      options.mode(),
      options.sparse(),
      per_sample_weights,
      options.include_last_offset(),
      options.padding_idx());
}

void EmbeddingBagImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::EmbeddingBag(num_embeddings="
         << options.num_embeddings()
         << ", embedding_dim=" << options.embedding_dim();
  auto const& max_norm_opt = options.max_norm();
  if (max_norm_opt.has_value()) {
    stream << ", max_norm=" << *max_norm_opt;
  }
  if (options.norm_type() != 2) {
    stream << ", norm_type=" << options.norm_type();
  }
  if (options.scale_grad_by_freq()) {
    stream << ", scale_grad_by_freq=" << std::boolalpha
           << options.scale_grad_by_freq();
  }
  if (options.sparse()) {
    stream << ", sparse=" << std::boolalpha << options.sparse();
  }
  if (!std::get_if<enumtype::kMean>(&options.mode())) {
    stream << ", mode=" << torch::enumtype::get_enum_name(options.mode());
  }
  if (options.include_last_offset()) {
    stream << ", include_last_offset=" << std::boolalpha
           << options.include_last_offset();
  }
  auto const& padding_idx_opt = options.padding_idx();
  if (padding_idx_opt.has_value()) {
    stream << ", padding_idx=" << padding_idx_opt.value();
  }
  stream << ")";
}
} // namespace torch::nn

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

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

- `torch/nn/modules/embedding.h`
- `torch/nn/init.h`
- `torch/types.h`
- `torch/utils.h`
- `ostream`
- `utility`


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

Files in the same folder (`torch/csrc/api/src/nn/modules`):

- [`pooling.cpp_docs.md`](./pooling.cpp_docs.md)
- [`linear.cpp_docs.md`](./linear.cpp_docs.md)
- [`padding.cpp_docs.md`](./padding.cpp_docs.md)
- [`rnn.cpp_docs.md`](./rnn.cpp_docs.md)
- [`upsampling.cpp_docs.md`](./upsampling.cpp_docs.md)
- [`dropout.cpp_docs.md`](./dropout.cpp_docs.md)
- [`pixelshuffle.cpp_docs.md`](./pixelshuffle.cpp_docs.md)
- [`loss.cpp_docs.md`](./loss.cpp_docs.md)
- [`fold.cpp_docs.md`](./fold.cpp_docs.md)


## Cross-References

- **File Documentation**: `embedding.cpp_docs.md`
- **Keyword Index**: `embedding.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
