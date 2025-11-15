# Documentation: `docs/aten/src/ATen/functorch/PyTorchOperatorHacks.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/functorch/PyTorchOperatorHacks.cpp_docs.md`
- **Size**: 12,598 bytes (12.30 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/functorch/PyTorchOperatorHacks.cpp`

## File Metadata

- **Path**: `aten/src/ATen/functorch/PyTorchOperatorHacks.cpp`
- **Size**: 9,741 bytes (9.51 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/functorch/DynamicLayer.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/functorch/TensorWrapper.h>
#include <ATen/functorch/BatchedTensorImpl.h>
#include <ATen/Dispatch.h>
#include <c10/util/irange.h>
#include <c10/util/Exception.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/xnnpack/Engine.h>

namespace at::functorch {

// NOTE: [functorch's PyTorch Operator Hacks]
//
// This file contains hacks for composite PyTorch operators that are problematic.
// For example, the composite op might have in-place operations,
// or call data_ptr. We have some idea of how to fix these things in the long term
// e.g., upstream the changes to PyTorch.
//
// TODO: all of these should be fixed in a more blessed way. In particular,
// it is bad if any of these go out-of-sync with the implementations in
// pytorch/pytorch.

// TODO: upstream into core

namespace {
Tensor index_select_backward_hack(const Tensor& grad, IntArrayRef self_sizes, int64_t dim, const Tensor& index) {
  return at::zeros(self_sizes, grad.options()).index_add(dim, index, grad);
}

// TODO: linear is pretty important for performance, but I'm not sure how to work
// around the in-place.
Tensor linear_hack(const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt) {
  // See [Note: hacky wrapper removal for optional tensor]
  auto bias = bias_opt.has_value()
    ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
    : c10::MaybeOwned<Tensor>::owned(std::in_place);

  if (input.is_mkldnn()) {
    return at::mkldnn_linear(input, weight, *bias);
  }
#if defined(C10_MOBILE)
  if (at::native::xnnpack::use_linear(input, weight, *bias)) {
    return at::native::xnnpack::linear(input, weight, *bias);
  }
#endif
  if (input.dim() == 2 && bias->defined()) {
    // Fused op is marginally faster.
    return at::addmm(*bias, input, weight.t());
  }
  if (input.dim() == 3 && bias->defined() && input.is_contiguous()) {
    // Also hit the fused path for contiguous 3D input.
    const auto input_sizes = input.sizes();
    const auto result = at::addmm(*bias, input.view({input_sizes[0] * input_sizes[1], input_sizes[2]}), weight.t());
    return result.view({input_sizes[0], input_sizes[1], result.size(1)});
  }
  auto output = at::matmul(input, weight.t());
  if (bias->defined()) {
    const auto& stack = getDynamicLayerStack();
    bool any_vmap_layers = std::any_of(
        stack.begin(), stack.end(),
        [](const DynamicLayer& dl){ return dl.key() == TransformType::Vmap; });
    if (any_vmap_layers) {
      return output.add(*bias);
    }
    return output.add_(*bias);
  }
  return output;
}

inline at::Tensor apply_loss_reduction(const at::Tensor& unreduced, int64_t reduction) {
  if (reduction == at::Reduction::Mean) {
    return unreduced.mean();
  } else if (reduction == at::Reduction::Sum) {
    return unreduced.sum();
  }
  return unreduced;
}

Tensor binary_cross_entropy_with_logits_hack(
    const Tensor& input,
    const Tensor& target,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& pos_weight_opt,
    int64_t reduction) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  const Tensor& pos_weight = pos_weight_opt.value_or(Tensor());

  Tensor loss;
  auto max_val = (-input).clamp_min(0);
  if (pos_weight.defined()) {
    // pos_weight need to be broadcasted, thus mul(target) is not inplace.
    auto log_weight = (pos_weight - 1).mul(target).add_(1);
    loss = (1 - target).mul(input).add(log_weight.mul(((-max_val).exp_().add((-input - max_val).exp_())).log_().add_(max_val)));
  } else {
    loss = (1 - target).mul(input).add_(max_val).add_((-max_val).exp_().add((-input -max_val).exp_()).log_());
  }

  if (weight.defined()) {
    loss = loss * weight;
  }

  return apply_loss_reduction(loss, reduction);
}

Tensor trace_backward_decomp(const Tensor& grad, IntArrayRef sizes) {
  TORCH_CHECK(sizes.size() == 2, "expected matrix input");
  auto grad_input = at::zeros(sizes[0] * sizes[1], grad.options());
  auto indices = at::arange(0, grad_input.numel(), sizes[1] + 1, grad.options().dtype(at::kLong));
  // Workaround using index_put instead of yet unsupported index_fill_
  grad_input = grad_input.index_put({indices}, grad);
  return grad_input.view(sizes);
}
}

// dropout hack
// TODO: make the following changes in pytorch/pytorch
namespace dropout_hack {

namespace {

template<bool inplace>
using Ctype = std::conditional_t<inplace, Tensor&, Tensor>;

Tensor make_feature_noise(const Tensor& input) {
  auto input_sizes = input.sizes();
  TORCH_CHECK(input.dim() >= 2, "Feature dropout requires at least 2 dimensions in the input");
  std::vector<int64_t> sizes;
  sizes.reserve(input.dim());
  sizes.push_back(input_sizes[0]);
  sizes.push_back(input_sizes[1]);
  for ([[maybe_unused]] const auto i : c10::irange(2, input.dim())) {
    sizes.push_back(1);
  }
  // NB: THIS WAS CHANGED FROM THE ORIGINAL
  return at::empty(sizes, input.options());
}

bool is_fused_kernel_acceptable(const Tensor& input, double p) {
  return (input.is_cuda() || input.is_xpu() || input.is_lazy() || input.is_privateuseone()) && p > 0 && p < 1 && input.numel() > 0;
}

// NB: sure, we could have used different overloads here, but I would feel insecure
// knowing that this dispatch depends only on the constness of the references
template<bool inplace>
Tensor& multiply(Tensor& input, const Tensor& noise) {
  static_assert(inplace, "Wrong multiply overload triggered in Dropout.cpp");
  return input.mul_(noise);
}

template<bool inplace>
Tensor multiply(const Tensor& input, const Tensor& noise) {
  static_assert(!inplace, "Wrong multiply overload triggered in Dropout.cpp");
  return input.mul(noise);
}

template<bool feature_dropout, bool alpha_dropout, bool inplace, typename T>
Ctype<inplace> _dropout_impl(T& input, double p, bool train) {
  TORCH_CHECK(p >= 0 && p <= 1, "dropout probability has to be between 0 and 1, but got ", p);
  if (p == 0 || !train || input.numel() == 0) {
    return input;
  }

  if (p == 1) {
    return multiply<inplace>(input, at::zeros({}, input.options()));
  }

  at::Tensor b; // used for alpha_dropout only

  // NB: THIS WAS CHANGED FROM THE ORIGINAL
  Tensor noise;
  if (feature_dropout) {
    auto empty = make_feature_noise(input);
    noise = at::bernoulli(empty, 1 - p);
  } else {
    // NB: it is important that this is at::empty and not at::empty_like
    auto empty = at::empty({}, input.options()).expand(input.sizes());
    noise = at::bernoulli(empty, 1 - p);
  }

  if (alpha_dropout) {
    constexpr double alpha = 1.7580993408473766;
    double a = 1. / std::sqrt((alpha * alpha * p + 1) * (1 - p));
    b = noise.add(-1).mul_(alpha * a).add_(alpha * a * p);
    noise.mul_(a);
  } else {
    noise.div_(1 - p);
  }

  if (!alpha_dropout) {
    return multiply<inplace>(input, noise);
  } else {
    return multiply<inplace>(input, noise).add_(b);
  }
}

#define ALIAS_SPECIALIZATION(ALIAS_NAME, IS_FEATURE, IS_ALPHA)                      \
template <bool inplace, typename... Args>                                           \
Ctype<inplace> ALIAS_NAME(Args&&... args) {                                         \
  return _dropout_impl<IS_FEATURE, IS_ALPHA, inplace>(std::forward<Args>(args)...); \
}

ALIAS_SPECIALIZATION(_dropout,               false, false)
ALIAS_SPECIALIZATION(_feature_dropout,       true,  false)
ALIAS_SPECIALIZATION(_alpha_dropout,         false, true )
ALIAS_SPECIALIZATION(_feature_alpha_dropout, true,  true )

Tensor dropout(const Tensor& input, double p, bool train) {
  auto result = [&]() {
    NoNamesGuard guard;
    if (train && is_fused_kernel_acceptable(input, p)) {
      return std::get<0>(at::native_dropout(input, p, train));
    }
    return _dropout<false>(input, p, train);
  }();
  namedinference::propagate_names(result, input);
  return result;
}

Tensor& dropout_(Tensor& input, double p, bool train) {
  return _dropout<true>(input, p, train);
}

Tensor feature_dropout(const Tensor& input, double p, bool train) {
  return _feature_dropout<false>(input, p, train);
}

Tensor& feature_dropout_(Tensor& input, double p, bool train) {
  return _feature_dropout<true>(input, p, train);
}

Tensor alpha_dropout(const Tensor& input, double p, bool train) {
  return _alpha_dropout<false>(input, p, train);
}

Tensor& alpha_dropout_(Tensor& input, double p, bool train) {
  return _alpha_dropout<true>(input, p, train);
}

Tensor feature_alpha_dropout(const Tensor& input, double p, bool train) {
  return _feature_alpha_dropout<false>(input, p, train);
}

Tensor& feature_alpha_dropout_(Tensor& input, double p, bool train) {
  return _feature_alpha_dropout<true>(input, p, train);
}

}
} // dropout_hack

TORCH_LIBRARY_IMPL(aten, FuncTorchDynamicLayerFrontMode, m) {
  m.impl("index_select_backward", index_select_backward_hack);
  m.impl("linear", linear_hack);
  m.impl("binary_cross_entropy_with_logits", binary_cross_entropy_with_logits_hack);
  m.impl("trace_backward", trace_backward_decomp);

  m.impl("dropout", dropout_hack::dropout);
  m.impl("feature_dropout", dropout_hack::feature_dropout);
  m.impl("alpha_dropout", dropout_hack::alpha_dropout);
  m.impl("feature_alpha_dropout", dropout_hack::feature_alpha_dropout);

  m.impl("dropout_", dropout_hack::dropout_);
  m.impl("feature_dropout_", dropout_hack::feature_dropout_);
  m.impl("alpha_dropout_", dropout_hack::alpha_dropout_);
  m.impl("feature_alpha_dropout_", dropout_hack::feature_alpha_dropout_);
}

} // namespace at::functorch

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 19 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `dropout_hack`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/functorch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/functorch/DynamicLayer.h`
- `torch/library.h`
- `ATen/ATen.h`
- `ATen/WrapDimUtils.h`
- `ATen/functorch/TensorWrapper.h`
- `ATen/functorch/BatchedTensorImpl.h`
- `ATen/Dispatch.h`
- `c10/util/irange.h`
- `c10/util/Exception.h`
- `ATen/NamedTensorUtils.h`
- `ATen/native/LinearAlgebraUtils.h`
- `ATen/native/xnnpack/Engine.h`


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

Files in the same folder (`aten/src/ATen/functorch`):

- [`Interpreter.cpp_docs.md`](./Interpreter.cpp_docs.md)
- [`Interpreter.h_docs.md`](./Interpreter.h_docs.md)
- [`BatchRulesScatterOps.cpp_docs.md`](./BatchRulesScatterOps.cpp_docs.md)
- [`BatchRulesHelper.h_docs.md`](./BatchRulesHelper.h_docs.md)
- [`BatchedFallback.cpp_docs.md`](./BatchedFallback.cpp_docs.md)
- [`BatchRulesLinearAlgebra.cpp_docs.md`](./BatchRulesLinearAlgebra.cpp_docs.md)
- [`VmapModeRegistrations.cpp_docs.md`](./VmapModeRegistrations.cpp_docs.md)
- [`PlumbingHelper.h_docs.md`](./PlumbingHelper.h_docs.md)
- [`BatchRulesFactory.cpp_docs.md`](./BatchRulesFactory.cpp_docs.md)
- [`BatchedTensorImpl.cpp_docs.md`](./BatchedTensorImpl.cpp_docs.md)


## Cross-References

- **File Documentation**: `PyTorchOperatorHacks.cpp_docs.md`
- **Keyword Index**: `PyTorchOperatorHacks.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/functorch`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/functorch`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/aten/src/ATen/functorch`):

- [`BatchRulesNorm.cpp_docs.md_docs.md`](./BatchRulesNorm.cpp_docs.md_docs.md)
- [`FunctionalizeInterpreter.h_kw.md_docs.md`](./FunctionalizeInterpreter.h_kw.md_docs.md)
- [`TensorWrapper.cpp_kw.md_docs.md`](./TensorWrapper.cpp_kw.md_docs.md)
- [`PlumbingHelper.h_docs.md_docs.md`](./PlumbingHelper.h_docs.md_docs.md)
- [`BatchRulesNorm.cpp_kw.md_docs.md`](./BatchRulesNorm.cpp_kw.md_docs.md)
- [`LegacyBatchingRegistrations.cpp_kw.md_docs.md`](./LegacyBatchingRegistrations.cpp_kw.md_docs.md)
- [`BatchRulesHelper.h_docs.md_docs.md`](./BatchRulesHelper.h_docs.md_docs.md)
- [`Interpreter.h_docs.md_docs.md`](./Interpreter.h_docs.md_docs.md)
- [`BatchedTensorImpl.cpp_docs.md_docs.md`](./BatchedTensorImpl.cpp_docs.md_docs.md)
- [`BatchRulesDecompositions.cpp_kw.md_docs.md`](./BatchRulesDecompositions.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `PyTorchOperatorHacks.cpp_docs.md_docs.md`
- **Keyword Index**: `PyTorchOperatorHacks.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
