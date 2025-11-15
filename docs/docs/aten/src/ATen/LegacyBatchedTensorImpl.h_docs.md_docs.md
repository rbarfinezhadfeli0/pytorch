# Documentation: `docs/aten/src/ATen/LegacyBatchedTensorImpl.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/LegacyBatchedTensorImpl.h_docs.md`
- **Size**: 8,077 bytes (7.89 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/LegacyBatchedTensorImpl.h`

## File Metadata

- **Path**: `aten/src/ATen/LegacyBatchedTensorImpl.h`
- **Size**: 5,563 bytes (5.43 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <bitset>

#include <ATen/ArrayRef.h>
#include <ATen/SmallVector.h>
#include <ATen/Tensor.h>

namespace at {

// We assume this in a few other places in the codebase,
// but there isn't a centralized definition.
constexpr int64_t kVmapMaxTensorDims = 64;

// The valid vmap levels range from [0, 64). This effectively means that we
// support a maximum of 64 nested vmaps.
constexpr int64_t kVmapNumLevels = 64;

// Store this number of elements of BatchDims on the stack. Most people will
// probably use <= 5 nested vmaps, but adjust this number as necessary.
constexpr int64_t kBatchDimsStackSize = 5;

// a BatchDim represents a "private" dimension on a Tensor created inside of
// vmap. It is a (level, dim) tuple, with the `dim` indicating which dimension
// is being vmap'ed over and the `level` being an identifier for which vmap
// said dimension was created inside. The `dim` corresponds to a "physical
// dim" - it is a dimension index on the underlying physical tensor that is
// being vmapped over.
struct BatchDim {
  BatchDim(int64_t level, int64_t dim) : dim_(dim), level_(level) {}
  int64_t dim() const {
    return dim_;
  }
  int64_t level() const {
    return level_;
  }

 private:
  int64_t dim_;
  int64_t level_;
};

using BatchDims = SmallVector<BatchDim, kBatchDimsStackSize>;
using BatchDimsRef = ArrayRef<BatchDim>;

// A BatchedTensorImpl holds an underlying Tensor and a list of BatchDim
// NB: We use the term "BatchedTensor" to mean a Tensor that is backed with a
// BatchedTensorImpl.
//
// The batch dimensions are treated as being "private"; they are not
// user-visible. For example, in the following Tensor,
//    bt = BatchedTensorImpl(ones(2, 3, 5, 7), [(lvl=1, dim=0), (lvl=2, dim=1)])
// dimensions 0 and 1 are batch dimensions.
//
// bt.sizes() returns (5, 7); bt.sum(0) performs a reduction over the (public)
// dim 0, which is equivalent to dim 3 in the underlying ones(2, 3, 5, 7)
// tensor.
struct TORCH_API BatchedTensorImpl : public c10::TensorImpl {
  explicit BatchedTensorImpl(Tensor value, BatchDims bdims);

  // Returns a reference to BatchDims that represent which dimensions of this
  // tensor are private.
  BatchDimsRef bdims() const {
    return bdims_;
  }

  // BatchedTensorImpl wraps a Tensor
  const Tensor& value() const {
    return value_;
  }

  // Given a public dimension index, return the dimension index in the
  // underlying value() tensor. For example, if we have
  //    bt = BatchedTensorImpl(ones(2, 3, 5, 7), [(lvl=1, dim=0), (lvl=2,
  //    dim=2)])
  // bt.actualDim(0) -> 1
  // bt.actualDim(1) -> 3
  // bt.actualDim(2) -> Error
  int64_t actualDim(int64_t dim, bool wrap_dim = true) const;

  // We have to override this because we opted into CustomStrides
  IntArrayRef strides_custom() const override;
  // Override a bunch of methods inherited from TensorImpl to return error
  // messages.
  c10::SymBool sym_is_contiguous_custom(
      at::MemoryFormat memory_format) const override;
  void set_size(int64_t dim, int64_t new_size) override;
  void set_stride(int64_t dim, int64_t new_stride) override;
  void set_storage_offset(int64_t storage_offset) override;
#ifdef DEBUG
  bool has_storage() const override;
#endif

 private:
  // see NOTE: [BatchedTensorImpl levels invariant]
  void checkInvariants() const;
  const char* tensorimpl_type_name() const override;

  Tensor value_;

  // Note: [BatchedTensorImpl levels invariant]
  // There is an invariant that the BatchDims must be stored in increasing
  // `level` order. That is, for i < j, bdims_[i].level must be less than
  // bdims_[j].level.
  BatchDims bdims_;
};

// NB: We use the term "BatchedTensor" to mean a Tensor that is backed with a
// BatchedTensorImpl.
inline bool isBatchedTensor(const Tensor& tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(DispatchKey::Batched);
}

// It is unsafe to call this on a Tensor that is not backed by a
// BatchedTensorImpl. Please use `maybeGetBatchedImpl` whenever possible.
inline BatchedTensorImpl* unsafeGetBatchedImpl(const Tensor& tensor) {
  return static_cast<BatchedTensorImpl*>(tensor.unsafeGetTensorImpl());
}

inline BatchedTensorImpl* maybeGetBatchedImpl(const Tensor& tensor) {
  if (!isBatchedTensor(tensor)) {
    return nullptr;
  }
  return unsafeGetBatchedImpl(tensor);
}

// Returns a bitset. If bit i is set, then that means dim i is a batchdim.
inline std::bitset<kVmapMaxTensorDims> createBatchDimBitset(
    BatchDimsRef bdims) {
  std::bitset<kVmapMaxTensorDims> is_bdim;
  for (const auto& bdim : bdims) {
    is_bdim.set(bdim.dim());
  }
  return is_bdim;
}

// Creates a bitset for all of the levels present in `bdims`
inline std::bitset<kVmapNumLevels> createVmapLevelsBitset(BatchDimsRef bdims) {
  std::bitset<kVmapNumLevels> result;
  for (const auto& bdim : bdims) {
    result.set(bdim.level());
  }
  return result;
}

inline std::ostream& operator<<(std::ostream& out, const BatchDim& bdim) {
  out << "(lvl=" << bdim.level() << ", dim=" << bdim.dim() << ")";
  return out;
}

// Use this to construct a BatchedTensor from a regular Tensor
TORCH_API Tensor makeBatched(Tensor tensor, BatchDims bdims);

// Adds a batch dim to `tensor`, returning a BatchedTensor
TORCH_API Tensor addBatchDim(Tensor tensor, int64_t level, int64_t dim);

// Checks if an inplace operation on self and other is "vmap compatible".
// See NOTE: [vmap-incompatible in-place operations] for the definition of this.
TORCH_API bool inplaceIsVmapCompatible(const Tensor& self, const Tensor& other);

} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 21 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `BatchDim`, `TORCH_API`, `a`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `bitset`
- `ATen/ArrayRef.h`
- `ATen/SmallVector.h`
- `ATen/Tensor.h`


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

- **File Documentation**: `LegacyBatchedTensorImpl.h_docs.md`
- **Keyword Index**: `LegacyBatchedTensorImpl.h_kw.md`
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
- [`TensorOperators.h_docs.md_docs.md`](./TensorOperators.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `LegacyBatchedTensorImpl.h_docs.md_docs.md`
- **Keyword Index**: `LegacyBatchedTensorImpl.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
