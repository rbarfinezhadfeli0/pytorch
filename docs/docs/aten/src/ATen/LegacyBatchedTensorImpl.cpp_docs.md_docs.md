# Documentation: `docs/aten/src/ATen/LegacyBatchedTensorImpl.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/LegacyBatchedTensorImpl.cpp_docs.md`
- **Size**: 8,279 bytes (8.08 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/LegacyBatchedTensorImpl.cpp`

## File Metadata

- **Path**: `aten/src/ATen/LegacyBatchedTensorImpl.cpp`
- **Size**: 5,775 bytes (5.64 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/LegacyBatchedTensorImpl.h>

#include <ATen/WrapDimUtils.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

namespace at {

BatchedTensorImpl::BatchedTensorImpl(Tensor value, BatchDims bdims)
  : TensorImpl(
      c10::DispatchKeySet(DispatchKey::Batched),
      value.dtype(),
      value.device()
    )
  , value_(std::move(value))
  , bdims_(std::move(bdims))
{
  TORCH_INTERNAL_ASSERT(value_.defined());
  set_storage_access_should_throw();
  set_custom_sizes_strides(SizesStridesPolicy::CustomStrides);
  checkInvariants();

  const auto public_dims = value_.dim() - bdims_.size();
  const auto value_sizes = value_.sizes();
  const auto value_strides = value_.strides();
  sizes_and_strides_.resize(public_dims);
  for (const auto dim : c10::irange(public_dims)) {
    auto actual_dim = actualDim(static_cast<int64_t>(dim), /*wrap_dim=*/false);
    sizes_and_strides_.size_at_unchecked(dim) = value_sizes.at(actual_dim);
    sizes_and_strides_.stride_at_unchecked(dim) = value_strides.at(actual_dim);
  }
  storage_offset_ = value_.storage_offset();
  refresh_numel();
  refresh_contiguous();
}

int64_t BatchedTensorImpl::actualDim(int64_t dim, bool wrap_dim) const {
  if (wrap_dim) {
    const auto ndim = sizes_and_strides_.size();
    dim = maybe_wrap_dim(dim, static_cast<int64_t>(ndim));
  }
  auto is_bdim = createBatchDimBitset(bdims_);

  // Example: assume dim = 3, and is_bdim = 10010011000...
  // The 1's are batch dims and 0's are normal dims of the underlying value_ Tensor.
  // actualDim gives us the index of `dim` in the `value_` Tensor, which is equivalent
  // to asking "where does the 3rd (0-indexed) zero occur in the bitset?".
  // The answer to that is index 5.
  //
  // TODO(rzou): the PDEP instruction does exactly this
  // (https://stackoverflow.com/questions/7669057/find-nth-set-bit-in-an-int)
  // but it might require newer (>= ~2015) CPUs. We should clean this up
  // if/when we have dropped support for older CPUs.
  int64_t non_bdim_count = 0;
  for (const auto actual_dim : c10::irange(kVmapMaxTensorDims)) {
    if (is_bdim[actual_dim]) {
      continue;
    }
    if (non_bdim_count == dim) {
      return actual_dim;
    }
    non_bdim_count++;
  }
  // If we hit this assert, then that means
  // `non_bdim_count` + #num_bdims > kVmapMaxTensorDims. We restrict the number
  // of dims a BatchedTensorImpl can have to kVmapMaxTensorDims so this should
  // never be hit.
  TORCH_INTERNAL_ASSERT(false);
}

void BatchedTensorImpl::checkInvariants() const {
  int64_t prev_level = -1;
  for (const auto& bdim : bdims_) {
    TORCH_INTERNAL_ASSERT(bdim.level() > prev_level);
    prev_level = bdim.level();
  }
}

// The following are publicly exposed as methods of Tensor

IntArrayRef BatchedTensorImpl::strides_custom() const {
  return strides_default();
}

// TODO: implement proper contiguity on batched tensor, then put
// sizes_strides_policy back to Default
c10::SymBool BatchedTensorImpl::sym_is_contiguous_custom(at::MemoryFormat memory_format) const {
  TORCH_CHECK(memory_format == MemoryFormat::Contiguous,
      "NYI: querying is_contiguous inside of vmap for memory_format ",
      "other than torch.contiguous_format");
  return is_contiguous_;
}

// The following are some internal inherited methods that we do not support.
// They should never get called.
void BatchedTensorImpl::set_size(int64_t dim, int64_t new_size) {
  TORCH_INTERNAL_ASSERT(false, "Can't set_size for BatchedTensorImpl");
}
void BatchedTensorImpl::set_stride(int64_t dim, int64_t new_stride) {
  TORCH_INTERNAL_ASSERT(false, "Can't set_stride for BatchedTensorImpl");
}
void BatchedTensorImpl::set_storage_offset(int64_t storage_offset) {
  TORCH_INTERNAL_ASSERT(false, "Can't set_storage_offset for BatchedTensorImpl");
}
#ifdef DEBUG
bool BatchedTensorImpl::has_storage() const {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!storage_, "BatchedTensorImpl assumes that storage_ is never set");
  return false;
}
#endif

const char* BatchedTensorImpl::tensorimpl_type_name() const {
  return "BatchedTensorImpl";
}

Tensor makeBatched(Tensor tensor, BatchDims bdims) {
  TORCH_INTERNAL_ASSERT(!isBatchedTensor(tensor));
  auto tensor_dim = tensor.dim();
  TORCH_CHECK(
      tensor_dim <= kVmapMaxTensorDims,
      "vmap only supports tensors of dimensionality up to ", kVmapMaxTensorDims,
      "; got a tensor with dim ", tensor_dim);
  TORCH_INTERNAL_ASSERT(
      std::all_of(bdims.begin(), bdims.end(),
          [](const BatchDim& bdim) { return bdim.level() < kVmapNumLevels; }),
      "We only support up to ", kVmapNumLevels, " nested vmaps");
  return at::detail::make_tensor<BatchedTensorImpl>(std::move(tensor), std::move(bdims));
}

Tensor addBatchDim(Tensor tensor, int64_t level, int64_t dim) {
  const auto* batched = maybeGetBatchedImpl(tensor);
  if (!batched) {
    BatchDims bdims;
    bdims.emplace_back(level, dim);
    return at::detail::make_tensor<BatchedTensorImpl>(std::move(tensor), std::move(bdims));
  }
  BatchDims new_bdims(batched->bdims().begin(), batched->bdims().end());
  auto actual_bdim = batched->actualDim(dim, /*wrap_dim=*/true);
  new_bdims.emplace_back(level, actual_bdim);
  return makeBatched(batched->value(), std::move(new_bdims));
}

bool inplaceIsVmapCompatible(const Tensor& self, const Tensor& other) {
  const auto* other_batched = maybeGetBatchedImpl(other);
  if (!other_batched) {
    return true;
  }
  const auto* self_batched = maybeGetBatchedImpl(self);
  if (!self_batched) {
    // self is not batched but other is batched
    return false;
  }
  auto self_levels = createVmapLevelsBitset(self_batched->bdims());
  auto other_levels = createVmapLevelsBitset(other_batched->bdims());
  return self_levels == (self_levels | other_levels);
}

} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/LegacyBatchedTensorImpl.h`
- `ATen/WrapDimUtils.h`
- `c10/util/Exception.h`
- `c10/util/irange.h`


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

Files in the same folder (`aten/src/ATen`):

- [`TensorGeometry.cpp_docs.md`](./TensorGeometry.cpp_docs.md)
- [`ROCmFABackend.h_docs.md`](./ROCmFABackend.h_docs.md)
- [`Generator.h_docs.md`](./Generator.h_docs.md)
- [`ParallelCommon.cpp_docs.md`](./ParallelCommon.cpp_docs.md)
- [`ZeroTensorFallback.cpp_docs.md`](./ZeroTensorFallback.cpp_docs.md)
- [`CachedTensorUtils.h_docs.md`](./CachedTensorUtils.h_docs.md)
- [`LegacyBatchedFallback.cpp_docs.md`](./LegacyBatchedFallback.cpp_docs.md)
- [`TensorOptions.h_docs.md`](./TensorOptions.h_docs.md)
- [`ExpandUtils.h_docs.md`](./ExpandUtils.h_docs.md)
- [`TensorIteratorInternal.h_docs.md`](./TensorIteratorInternal.h_docs.md)


## Cross-References

- **File Documentation**: `LegacyBatchedTensorImpl.cpp_docs.md`
- **Keyword Index**: `LegacyBatchedTensorImpl.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/aten/src/ATen`):

- [`Dispatch.cpp_docs.md_docs.md`](./Dispatch.cpp_docs.md_docs.md)
- [`Context.cpp_docs.md_docs.md`](./Context.cpp_docs.md_docs.md)
- [`ThreadLocalState.cpp_docs.md_docs.md`](./ThreadLocalState.cpp_docs.md_docs.md)
- [`DeviceAccelerator.cpp_kw.md_docs.md`](./DeviceAccelerator.cpp_kw.md_docs.md)
- [`FunctionalInverses.cpp_kw.md_docs.md`](./FunctionalInverses.cpp_kw.md_docs.md)
- [`SequenceNumber.h_kw.md_docs.md`](./SequenceNumber.h_kw.md_docs.md)
- [`ThreadLocalPythonObjects.h_docs.md_docs.md`](./ThreadLocalPythonObjects.h_docs.md_docs.md)
- [`TensorNames.h_docs.md_docs.md`](./TensorNames.h_docs.md_docs.md)
- [`LegacyBatchedTensorImpl.h_docs.md_docs.md`](./LegacyBatchedTensorImpl.h_docs.md_docs.md)
- [`TensorOperators.h_docs.md_docs.md`](./TensorOperators.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `LegacyBatchedTensorImpl.cpp_docs.md_docs.md`
- **Keyword Index**: `LegacyBatchedTensorImpl.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
