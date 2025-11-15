# Documentation: `aten/src/ATen/functorch/PlumbingHelper.h`

## File Metadata

- **Path**: `aten/src/ATen/functorch/PlumbingHelper.h`
- **Size**: 2,854 bytes (2.79 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#pragma once
#include <ATen/Tensor.h>
#include <ATen/functorch/BatchedTensorImpl.h>
#include <ATen/functorch/DynamicLayer.h>

// NOTE: [vmap plumbing]
//
// Here's how "batching rules" work.
// - we register kernels to the Batched key
// - these kernels have the same signatures as the original operators.
//   For example, at::sin(Tensor self) accepts a Tensor, and the batched kernel
//   must also accept a Tensor
// - However, it is more natural for users to write a batching rule like the
//   following: sin_batch_rule(Tensor self, std::optional<int> self_bdim)
// - There is some codegenerated layer (the "plumbing") that wraps the user
//   defined batching rule (e.g. sin_batch_rule) in a kernel that can be
//   registered to the Batched key.
//
// The plumbing is responsible for wrapping a batching rule into a form that may
// be registered as the kernel for the batched key.

namespace at::functorch {

void vmap_check_escaped(const std::optional<DynamicLayer> &layer, const char* what);

// Create a BatchedTensor given a tensor, bdim, and level
TORCH_API Tensor makeBatched(Tensor tensor, std::optional<int64_t> bdim, int64_t level);

// Given a Tensor that may or may not be a BatchedTensor, unwrap it.
// If `tensor` is not a BatchedTensor, or is a BatchedTensor but the level
// doesn't match, then this returns (tensor, std::nullopt).
// Otherwise, it returns (unwrap(tensor), bdim).
TORCH_API std::tuple<Tensor, std::optional<int64_t>> unwrapTensorAtLevel(const Tensor& tensor, int64_t level);

// Creates a vector of BatchedTensor
TORCH_API std::vector<Tensor> makeBatchedVector(std::vector<Tensor> tensors, std::optional<int64_t> bdim, int64_t level);

// Returns True if ANY tensor in tensors is batched at level
TORCH_API bool isBatchedAtLevel(ITensorListRef tensors, int64_t level);
TORCH_API bool isBatchedAtLevel(const c10::List<std::optional<Tensor>>& maybe_tensors, int64_t level);
TORCH_API bool isBatchedAtLevel(const Tensor& tensor, int64_t level);
TORCH_API bool isBatchedAtLevel(const std::optional<Tensor>& maybe_tensor, int64_t level);

// Convenience helper. Returns true if any tensor is batched at level
TORCH_API bool areAnyBatchedAtLevel(ArrayRef<std::optional<Tensor>> maybe_tensors, int64_t level);

inline bool ivalueParticipatesInCurrentLevel(const IValue& ivalue) {
  if (ivalue.isTensor()) {
    auto maybe_level = maybeCurrentDynamicLayer();
    TORCH_INTERNAL_ASSERT(maybe_level.has_value());
    auto current_level = maybe_level->layerId();
    return isBatchedAtLevel(ivalue.toTensor(), current_level);
  }
  // TODO: should really check this
  return false;
}

} // namespace at::functorch

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 13 function(s).

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

- `ATen/Tensor.h`
- `ATen/functorch/BatchedTensorImpl.h`
- `ATen/functorch/DynamicLayer.h`


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
- [`BatchRulesFactory.cpp_docs.md`](./BatchRulesFactory.cpp_docs.md)
- [`BatchedTensorImpl.cpp_docs.md`](./BatchedTensorImpl.cpp_docs.md)


## Cross-References

- **File Documentation**: `PlumbingHelper.h_docs.md`
- **Keyword Index**: `PlumbingHelper.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
