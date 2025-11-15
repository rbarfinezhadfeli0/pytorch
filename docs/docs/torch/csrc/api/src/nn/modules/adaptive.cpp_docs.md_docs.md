# Documentation: `docs/torch/csrc/api/src/nn/modules/adaptive.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/api/src/nn/modules/adaptive.cpp_docs.md`
- **Size**: 9,790 bytes (9.56 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/api/src/nn/modules/adaptive.cpp`

## File Metadata

- **Path**: `torch/csrc/api/src/nn/modules/adaptive.cpp`
- **Size**: 7,433 bytes (7.26 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/util/irange.h>
#include <torch/nn/modules/adaptive.h>
#include <torch/nn/options/activation.h>
#include <torch/nn/options/linear.h>

namespace F = torch::nn::functional;

using namespace torch::indexing;

namespace torch::nn {

ASMoutput::ASMoutput(Tensor output_, double loss_)
    : output(std::move(output_)), loss(loss_) {}

AdaptiveLogSoftmaxWithLossImpl::AdaptiveLogSoftmaxWithLossImpl(
    AdaptiveLogSoftmaxWithLossOptions options_)
    : options(std::move(options_)),
      shortlist_size(0),
      n_clusters(0),
      head_size(0) {
  AdaptiveLogSoftmaxWithLossImpl::reset();
}

void AdaptiveLogSoftmaxWithLossImpl::reset() {
  TORCH_CHECK(
      !options.cutoffs().empty(),
      "cutoffs should be a sequence of length larger than 0");
  TORCH_CHECK(
      std::is_sorted(options.cutoffs().begin(), options.cutoffs().end()) &&
          *std::min_element(
              options.cutoffs().begin(), options.cutoffs().end()) > 0 &&
          *std::max_element(
              options.cutoffs().begin(), options.cutoffs().end()) <=
              (options.n_classes() - 1) &&
          std::set<int64_t>(options.cutoffs().begin(), options.cutoffs().end())
                  .size() == options.cutoffs().size(),
      "cutoffs should be a sequence of unique, positive integers sorted in an increasing order, ",
      "where each value is between 1 and n_classes-1");
  TORCH_CHECK(options.div_value() != 0, "div_value should not be equal to 0");

  cutoffs = options.cutoffs();
  cutoffs.push_back(options.n_classes());

  shortlist_size = cutoffs[0];
  n_clusters = static_cast<int64_t>(cutoffs.size() - 1);
  head_size = shortlist_size + n_clusters;

  head = this->register_module(
      "head",
      Linear(LinearOptions(options.in_features(), head_size)
                 .bias(options.head_bias())));
  tail = this->register_module("tail", ModuleList());

  for (const auto i : c10::irange(n_clusters)) {
    int64_t hsz = static_cast<int64_t>(std::floor(
        static_cast<double>(options.in_features()) /
        std::pow(options.div_value(), (i + 1))));
    int64_t osz = cutoffs[i + 1] - cutoffs[i];

    Sequential projection(
        Linear(LinearOptions(options.in_features(), hsz).bias(false)),
        Linear(LinearOptions(hsz, osz).bias(false)));
    tail->push_back(projection);
  }
}

void AdaptiveLogSoftmaxWithLossImpl::reset_parameters() {
  head->reset_parameters();
  for (const auto i : c10::irange(tail->size())) {
    auto i2h = tail[i]->children()[0]->as<Linear>();
    auto h2o = tail[i]->children()[1]->as<Linear>();
    i2h->reset_parameters();
    h2o->reset_parameters();
  }
}

ASMoutput AdaptiveLogSoftmaxWithLossImpl::forward(
    const Tensor& input_,
    const Tensor& target_) {
  auto targ_dim = target_.dim();

  TORCH_CHECK(
      targ_dim == 1 || targ_dim == 0,
      "0D or 1D target tensor expected, multi-target not supported");

  if (targ_dim == 1) {
    TORCH_CHECK(
        input_.dim() == 2,
        "1D target tensor expects 2D input tensors, but found inputs with sizes ",
        input_.sizes(),
        ".");
  } else {
    TORCH_CHECK(
        input_.dim() == 1,
        "0D target tensor expects 1D input tensors, but found inputs with sizes ",
        input_.sizes(),
        ".");
  }

  bool is_batched = (targ_dim > 0);
  Tensor input = is_batched ? input_ : input_.unsqueeze(0);
  Tensor target = is_batched ? target_ : target_.unsqueeze(0);

  int64_t used_rows = 0;
  const int64_t batch_size = target.size(0);

  Tensor output = input.new_zeros(batch_size);
  Tensor gather_inds = target.new_empty(batch_size);

  auto cutoff_values = cutoffs;
  cutoff_values.insert(cutoff_values.begin(), 0);

  for (const auto i : c10::irange(cutoff_values.size() - 1)) {
    int64_t low_idx = cutoff_values[i];
    int64_t high_idx = cutoff_values[i + 1];

    const Tensor target_mask = (target >= low_idx) * (target < high_idx);
    const Tensor row_indices = target_mask.nonzero().squeeze();

    if (row_indices.numel() == 0) {
      continue;
    }

    if (i == 0) {
      gather_inds.index_copy_(0, row_indices, target.index({target_mask}));
    } else {
      Tensor relative_target = target.index({target_mask}) - low_idx;
      Tensor input_subset = input.index_select(0, row_indices);

      const Tensor cluster_output =
          tail[i - 1]->as<Sequential>()->forward(input_subset);
      int64_t cluster_index = shortlist_size + static_cast<int64_t>(i) - 1;

      gather_inds.index_fill_(0, row_indices, cluster_index);

      const Tensor cluster_logprob = F::log_softmax(cluster_output, 1);
      const Tensor local_logprob =
          cluster_logprob.gather(1, relative_target.unsqueeze(1));
      output.index_copy_(0, row_indices, local_logprob.squeeze(1));
    }

    used_rows += row_indices.numel();
  }

  TORCH_CHECK(
      used_rows == batch_size,
      "Target values should be in [0, ",
      options.n_classes() - 1,
      "], "
      "but values in range [",
      target.min().item().toDouble(),
      ", ",
      target.max().item().toDouble(),
      "] "
      "were found. ");

  const Tensor head_output = head(input);
  const Tensor head_logprob = F::log_softmax(head_output, 1);
  output += head_logprob.gather(1, gather_inds.unsqueeze(1)).squeeze();
  const double loss = (-output).mean().item().toDouble();

  if (!is_batched) {
    output = output.squeeze(0);
  }

  return ASMoutput(output, loss);
}

Tensor AdaptiveLogSoftmaxWithLossImpl::_get_full_log_prob(
    const Tensor& input,
    const Tensor& head_output) {
  Tensor out = input.new_empty({head_output.size(0), options.n_classes()});
  const Tensor head_logprob = F::log_softmax(head_output, 1);

  out.index_put_(
      {Slice(), Slice(None, shortlist_size)},
      head_logprob.index({Slice(), Slice(None, shortlist_size)}));

  for (const auto i : c10::irange(cutoffs.size() - 1)) {
    int64_t start_idx = cutoffs[i];
    int64_t stop_idx = cutoffs[i + 1];
    const Tensor cluster_output = tail[i]->as<Sequential>()->forward(input);
    const Tensor cluster_logprob = F::log_softmax(cluster_output, 1);
    auto output_logprob = cluster_logprob +
        head_logprob.index({Slice(), static_cast<int64_t>(shortlist_size + i)})
            .unsqueeze(1);

    out.index_put_({Slice(), Slice(start_idx, stop_idx)}, output_logprob);
  }
  return out;
}

Tensor AdaptiveLogSoftmaxWithLossImpl::AdaptiveLogSoftmaxWithLossImpl::log_prob(
    const Tensor& input) {
  const Tensor head_output = head(input);
  return _get_full_log_prob(input, head_output);
}

Tensor AdaptiveLogSoftmaxWithLossImpl::predict(const Tensor& input) {
  const Tensor head_output = head(input);
  Tensor output = torch::argmax(head_output, 1);
  auto not_in_shortlist = (output >= shortlist_size);
  auto all_in_shortlist = bitwise_not(not_in_shortlist.any());

  if (all_in_shortlist.item().toBool()) {
    return output;
  } else if (not_in_shortlist.all().item().toBool()) {
    const Tensor log_prob = _get_full_log_prob(input, head_output);
    return torch::argmax(log_prob, 1);
  } else {
    const Tensor log_prob = _get_full_log_prob(
        input.index({not_in_shortlist}), head_output.index({not_in_shortlist}));
    output.index_put_({not_in_shortlist}, torch::argmax(log_prob, 1));
    return output;
  }
}

void AdaptiveLogSoftmaxWithLossImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::AdaptiveLogSoftmaxWithLoss";
}

} // namespace torch::nn

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

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

- `c10/util/irange.h`
- `torch/nn/modules/adaptive.h`
- `torch/nn/options/activation.h`
- `torch/nn/options/linear.h`


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
- [`embedding.cpp_docs.md`](./embedding.cpp_docs.md)
- [`dropout.cpp_docs.md`](./dropout.cpp_docs.md)
- [`pixelshuffle.cpp_docs.md`](./pixelshuffle.cpp_docs.md)
- [`loss.cpp_docs.md`](./loss.cpp_docs.md)
- [`fold.cpp_docs.md`](./fold.cpp_docs.md)


## Cross-References

- **File Documentation**: `adaptive.cpp_docs.md`
- **Keyword Index**: `adaptive.cpp_kw.md`
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

- **File Documentation**: `adaptive.cpp_docs.md_docs.md`
- **Keyword Index**: `adaptive.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
