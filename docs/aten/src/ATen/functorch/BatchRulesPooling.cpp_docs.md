# Documentation: `aten/src/ATen/functorch/BatchRulesPooling.cpp`

## File Metadata

- **Path**: `aten/src/ATen/functorch/BatchRulesPooling.cpp`
- **Size**: 3,681 bytes (3.59 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/PlumbingHelper.h>
#include <ATen/functorch/BatchedFallback.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace at::functorch {

template <typename Func>
static std::tuple<Tensor, std::optional<int64_t>,Tensor, std::optional<int64_t>>
max_pool_with_indices_batch_rule_helper(
  const Tensor& self, std::optional<int64_t> self_bdim,
  IntArrayRef kernel_size, IntArrayRef stride,
  IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, int64_t n, Func pooling_fn) {

  auto logical_rank = rankWithoutBatchDim(self, self_bdim);
  TORCH_INTERNAL_ASSERT(logical_rank == n + 1 || logical_rank == n + 2);
  // Tensor[B, logical_rank...] -> just call max_poolnd
  if (logical_rank == n + 1) {
    auto self_ = moveBatchDimToFront(self, self_bdim);
    auto result = pooling_fn(
        self_, kernel_size, stride, padding, dilation, ceil_mode);
    return std::make_tuple(std::move(std::get<0>(result)), 0, std::move(std::get<1>(result)), 0);
  }
  // Tensor[B, N, logical_rank...] -> Tensor[B * N, logical_rank...]
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  auto bdim_size = self.size(self_bdim.value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  auto self_ = reshape_dim_into(self_bdim.value(), 0, self);
  auto result = pooling_fn(
      self_, kernel_size, stride, padding, dilation, ceil_mode);
  return std::make_tuple(
      reshape_dim_outof(0, bdim_size, std::get<0>(result)), 0,
      reshape_dim_outof(0, bdim_size, std::get<1>(result)), 0);
}

static std::tuple<Tensor, std::optional<int64_t>,Tensor, std::optional<int64_t>>
max_pool3d_with_indices_batch_rule(
    const Tensor& self, std::optional<int64_t> self_bdim,
    IntArrayRef kernel_size, IntArrayRef stride,
    IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
    return max_pool_with_indices_batch_rule_helper(self, self_bdim, kernel_size, stride, padding, dilation, ceil_mode, 3, at::max_pool3d_with_indices);
}

static std::tuple<Tensor, std::optional<int64_t>,Tensor, std::optional<int64_t>>
max_pool2d_with_indices_batch_rule(
    const Tensor& self, std::optional<int64_t> self_bdim,
    IntArrayRef kernel_size, IntArrayRef stride,
    IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
    return max_pool_with_indices_batch_rule_helper(self, self_bdim, kernel_size, stride, padding, dilation, ceil_mode, 2, at::max_pool2d_with_indices);
}

TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
  EXISTING_BDIM(_adaptive_avg_pool2d);
  EXISTING_BDIM_ALL_BOXED(_adaptive_avg_pool2d_backward);
  EXISTING_BDIM(_adaptive_avg_pool3d);
  EXISTING_BDIM_ALL_BOXED(_adaptive_avg_pool3d_backward);
  EXISTING_BDIM(avg_pool2d);
  EXISTING_BDIM(avg_pool3d);
  EXISTING_BDIM_ALL_BOXED(avg_pool2d_backward);
  EXISTING_BDIM_ALL_BOXED(avg_pool3d_backward);
  EXISTING_BDIM_ALL_BOXED(adaptive_max_pool2d);
  EXISTING_BDIM_ALL_BOXED(adaptive_max_pool3d);
  ALL_TENSORS_HAVE_OPTIONAL_BDIM_BOXED_CONTIG1(3, adaptive_max_pool2d_backward, 2);
  ALL_TENSORS_HAVE_OPTIONAL_BDIM_BOXED_CONTIG1(4, adaptive_max_pool3d_backward, 2);
  VMAP_SUPPORT(max_pool2d_with_indices, max_pool2d_with_indices_batch_rule);
  VMAP_SUPPORT(max_pool3d_with_indices, max_pool3d_with_indices_batch_rule);
  ALL_TENSORS_HAVE_OPTIONAL_BDIM_BOXED_CONTIG1(3, max_pool2d_with_indices_backward, 2);
  ALL_TENSORS_HAVE_OPTIONAL_BDIM_BOXED_CONTIG1(4, max_pool3d_with_indices_backward, 2);
}

} // namespace at::functorch

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/functorch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/functorch/BatchRulesHelper.h`
- `ATen/functorch/PlumbingHelper.h`
- `ATen/functorch/BatchedFallback.h`
- `ATen/core/dispatch/Dispatcher.h`


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

- **File Documentation**: `BatchRulesPooling.cpp_docs.md`
- **Keyword Index**: `BatchRulesPooling.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
