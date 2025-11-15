# Documentation: `aten/src/ATen/functorch/PlumbingHelper.cpp`

## File Metadata

- **Path**: `aten/src/ATen/functorch/PlumbingHelper.cpp`
- **Size**: 2,908 bytes (2.84 KB)
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

#include <ATen/functorch/TensorWrapper.h>
#include <ATen/functorch/DynamicLayer.h>
#include <ATen/functorch/BatchedTensorImpl.h>
#include <ATen/functorch/PlumbingHelper.h>

namespace at::functorch {

void vmap_check_escaped(const std::optional<DynamicLayer> &layer, const char* what) {
  TORCH_CHECK(
    layer.has_value(),
    "Either your tensor may have escaped from inside a function being vmapped and this is a user error ",
    "(see https://pytorch.org/functorch/stable/ux_limitations.html), "
    "or there is an internal functorch error in `",
    what,
    "` Please file an issue if it looks like the latter"
  )
}

Tensor makeBatched(Tensor tensor, std::optional<int64_t> bdim, int64_t level) {
  if (bdim.has_value()) {
    TORCH_INTERNAL_ASSERT(*bdim >= 0);
    TORCH_INTERNAL_ASSERT(*bdim < tensor.dim());
    return makeBatched(std::move(tensor), bdim.value(), level);
  }
  return tensor;
}

std::vector<Tensor> makeBatchedVector(std::vector<Tensor> tensors, std::optional<int64_t> bdim, int64_t level) {
  std::vector<Tensor> res;
  res.reserve(tensors.size());
  for (auto & tensor : tensors) {
    res.emplace_back(makeBatched(std::move(tensor), bdim, level));
  }
  return res;
}

std::tuple<Tensor, std::optional<int64_t>> unwrapTensorAtLevel(const Tensor& tensor, int64_t level) {
  auto* batched = maybeGetBatchedImpl(tensor);
  if (!batched) {
    return std::make_tuple(tensor, std::nullopt);
  }
  if (batched->level() == level) {
    return std::make_tuple(batched->value(), batched->bdim());
  }
  return std::make_tuple(tensor, std::nullopt);
}

bool isBatchedAtLevel(const Tensor& tensor, int64_t level) {
  auto result = unwrapTensorAtLevel(tensor, level);
  return std::get<1>(result).has_value();
}

bool isBatchedAtLevel(const std::optional<Tensor>& maybe_tensor, int64_t level) {
  if (!maybe_tensor.has_value()) {
    return false;
  }
  return isBatchedAtLevel(*maybe_tensor, level);
}

bool isBatchedAtLevel(ITensorListRef tensors, int64_t level) {
  for (const auto& tensor : tensors) {
    if (isBatchedAtLevel(tensor, level)) {
      return true;
    }
  }
  return false;
}

bool isBatchedAtLevel(const c10::List<std::optional<Tensor>>& maybe_tensors, int64_t level) {
  for (const auto idx : c10::irange(0, maybe_tensors.size())) {
    const auto& maybe_tensor = maybe_tensors.get(idx);
    if (isBatchedAtLevel(maybe_tensor, level)) {
      return true;
    }
  }
  return false;
}

bool areAnyBatchedAtLevel(ArrayRef<std::optional<Tensor>> maybe_tensors, int64_t level) {
  for (const auto& maybe_tensor : maybe_tensors) {
    if (isBatchedAtLevel(maybe_tensor, level)) {
      return true;
    }
  }
  return false;
}


} // namespace at::functorch

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 9 function(s).

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

- `ATen/functorch/TensorWrapper.h`
- `ATen/functorch/DynamicLayer.h`
- `ATen/functorch/BatchedTensorImpl.h`
- `ATen/functorch/PlumbingHelper.h`


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

- **File Documentation**: `PlumbingHelper.cpp_docs.md`
- **Keyword Index**: `PlumbingHelper.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
