# Documentation: `docs/aten/src/ATen/native/cpu/SerialStackImpl.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cpu/SerialStackImpl.h_docs.md`
- **Size**: 8,335 bytes (8.14 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cpu/SerialStackImpl.h`

## File Metadata

- **Path**: `aten/src/ATen/native/cpu/SerialStackImpl.h`
- **Size**: 5,456 bytes (5.33 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include <ATen/core/Tensor.h>

#include <ATen/MemoryOverlap.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/util/irange.h>

namespace at::native::detail {

struct InputMeta {
  void* data_ptr;
  int64_t inner_size;

  InputMeta(const Tensor& t, int64_t dim, int64_t inner)
      : data_ptr(t.data_ptr()), inner_size(t.sizes()[dim] * inner) {}
};

// This kernel is used by two TensorList types:
// 1. stack_serial_kernel uses at::ArrayRef<Tensor>
// 2. Static runtime calls this kernel directly (csrc/jit/runtime/static/ops.cpp) with
//    ProcessedNodeInputWrapper.
// When making changes, make sure that they are compatible with both types!
template <typename scalar_t, typename TensorListType>
void stack_serial_kernel_impl(Tensor& result, TensorListType tensors, int64_t dim) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      dim >= 0 && dim <= result.dim(),
      "dim out of range in stack_serial_kernel_impl");
  int64_t outer =
      result.numel() / (result.sizes()[dim] * result.strides()[dim]);
  scalar_t* result_data = result.data_ptr<scalar_t>();
  int64_t ninputs = tensors.size();
  std::vector<InputMeta> inputs;
  inputs.reserve(ninputs);
  for (const auto& tensor : tensors) {
    inputs.emplace_back(tensor, dim, tensor.strides()[dim]);
  }

  using Vec = vec::Vectorized<scalar_t>;
  scalar_t* result_ptr = result_data;
  for (const auto i : c10::irange(outer)) {
    for (const auto j : c10::irange(ninputs)) {
      int64_t local_inner = inputs[j].inner_size;
      scalar_t* input_ptr = (scalar_t*)(inputs[j].data_ptr) + i * local_inner;

      if (local_inner < Vec::size()) {
        for (const auto k : c10::irange(local_inner)) {
          result_ptr[k] = input_ptr[k];
        }
      } else {
        vec::map(
            [](Vec x) { return x; }, result_ptr, input_ptr, local_inner);
      }
      result_ptr += local_inner;
    }
  }
}

// Checks to see whether native stack can be invoked under these conditions:
// - result and input tensors are contiguous
// - only one thread is used
// - no type promotion has to occur
// - tensors dtype is Double or Float
template <typename TensorListType>
bool can_use_native_serial_stack_impl(Tensor& result, TensorListType tensors, int64_t dim) {
  TORCH_CHECK(!tensors.empty(), "expected a non-empty list of Tensors");
  const Tensor& first_tensor = tensors[0];
  // stack dimension should be in range [0,firstTensor.dim())
  // dim == firstTensor.dim() is a valid input, but it is handled by default code path
  // that uses unsqueeze
  if (dim >= first_tensor.dim()) return false;
  // Native stack doesn't apply any tensor is skipped.
  if (first_tensor.numel() == 0 && first_tensor.dim() == 1) return false;
  // there should be no type promotion
  if (result.dtype() != first_tensor.dtype()) return false;

  auto first_tensor_mem_format = first_tensor.suggest_memory_format();
  ScalarType dtype = first_tensor.scalar_type();

  if (!result.is_contiguous(first_tensor_mem_format)) {
    return false;
  }

  // fast path only works for Double and Float
  if (dtype != ScalarType::Double && dtype != ScalarType::Float) {
    return false;
  }

  // check remainder of inputs
#ifndef STRIP_ERROR_MESSAGES
  auto const &first_tensor_shape = first_tensor.sizes();
#endif
  for (const auto i : c10::irange(1, tensors.size())) {
    auto const &tensor = tensors[i];
    TORCH_CHECK(tensors[i].sizes() == first_tensor.sizes(),
      "stack expects each tensor to be equal size, but got ", first_tensor_shape,
      " at entry 0 and ", tensor.sizes(), " at entry ", i);

    // every tensor must be contiguous
    // tensor sizes and strides must be the same
    // there should be no type promotion
    if (!tensor.is_contiguous(first_tensor_mem_format) ||
      tensor.strides() != first_tensor.strides() ||
      tensor.dtype() != dtype) {
      return false;
    }
  }

  // fast native stack should only be used when it is not worth using multiple threads
  // or there is only one thread. Note that we aren't checking result.numel() here because
  // it may not have been resized and we want to defer that cost till later.
  int64_t numel_in_stack = first_tensor.numel() * tensors.size();
  return numel_in_stack < at::internal::GRAIN_SIZE || at::get_num_threads() == 1;
}

template <typename TensorListType, bool should_skip_overlap_check>
struct CanUseNativeSerialStack;

template <typename TensorListType>
struct CanUseNativeSerialStack<TensorListType, false> {
  static bool call(Tensor& result, TensorListType tensors, int64_t dim) {
    // Inputs cannot alias the output tensor
    for (const auto i : c10::irange(tensors.size())) {
      auto lap = at::get_overlap_status(result, tensors[i]);
      TORCH_CHECK(lap != at::MemOverlapStatus::Partial &&
          lap != at::MemOverlapStatus::Full, 0,
          "unsupported operation: the input tensors cannot refer to any of the "
          "output memory locations. Found overlap in input tensor ", i);
    }

    return can_use_native_serial_stack_impl(result, tensors, dim);
  }
};

template <typename TensorListType>
struct CanUseNativeSerialStack<TensorListType, true> {
  static bool call(Tensor& result, TensorListType tensors, int64_t dim) {
    return can_use_native_serial_stack_impl(result, tensors, dim);
  }
};

} // namespace at::native::detail

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 13 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `InputMeta`, `CanUseNativeSerialStack`, `CanUseNativeSerialStack`, `CanUseNativeSerialStack`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/MemoryOverlap.h`
- `ATen/Parallel.h`
- `ATen/TensorIterator.h`
- `ATen/cpu/vec/functional.h`
- `ATen/cpu/vec/vec.h`
- `c10/util/irange.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`aten/src/ATen/native/cpu`):

- [`UpSampleKernelAVXAntialias.h_docs.md`](./UpSampleKernelAVXAntialias.h_docs.md)
- [`SparseFactories.cpp_docs.md`](./SparseFactories.cpp_docs.md)
- [`UnfoldBackwardKernel.cpp_docs.md`](./UnfoldBackwardKernel.cpp_docs.md)
- [`int8mm_kernel.cpp_docs.md`](./int8mm_kernel.cpp_docs.md)
- [`LerpKernel.cpp_docs.md`](./LerpKernel.cpp_docs.md)
- [`UpSampleKernel.cpp_docs.md`](./UpSampleKernel.cpp_docs.md)
- [`scaled_modified_bessel_k0.cpp_docs.md`](./scaled_modified_bessel_k0.cpp_docs.md)
- [`DistributionKernels.cpp_docs.md`](./DistributionKernels.cpp_docs.md)
- [`CopyKernel.cpp_docs.md`](./CopyKernel.cpp_docs.md)
- [`SampledAddmmKernel.cpp_docs.md`](./SampledAddmmKernel.cpp_docs.md)


## Cross-References

- **File Documentation**: `SerialStackImpl.h_docs.md`
- **Keyword Index**: `SerialStackImpl.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/cpu`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/aten/src/ATen/native/cpu`):

- [`BinaryOpsKernel.cpp_docs.md_docs.md`](./BinaryOpsKernel.cpp_docs.md_docs.md)
- [`MultinomialKernel.cpp_kw.md_docs.md`](./MultinomialKernel.cpp_kw.md_docs.md)
- [`AmpGradScalerKernels.cpp_docs.md_docs.md`](./AmpGradScalerKernels.cpp_docs.md_docs.md)
- [`FusedSGDKernel.cpp_docs.md_docs.md`](./FusedSGDKernel.cpp_docs.md_docs.md)
- [`scaled_modified_bessel_k1.cpp_docs.md_docs.md`](./scaled_modified_bessel_k1.cpp_docs.md_docs.md)
- [`int_mm_kernel.h_docs.md_docs.md`](./int_mm_kernel.h_docs.md_docs.md)
- [`IsContiguous.h_docs.md_docs.md`](./IsContiguous.h_docs.md_docs.md)
- [`MaxPooling.cpp_docs.md_docs.md`](./MaxPooling.cpp_docs.md_docs.md)
- [`WeightNormKernel.cpp_kw.md_docs.md`](./WeightNormKernel.cpp_kw.md_docs.md)
- [`FusedAdamKernel.cpp_docs.md_docs.md`](./FusedAdamKernel.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `SerialStackImpl.h_docs.md_docs.md`
- **Keyword Index**: `SerialStackImpl.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
