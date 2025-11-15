# Documentation: `c10/core/impl/SizesAndStrides.h`

## File Metadata

- **Path**: `c10/core/impl/SizesAndStrides.h`
- **Size**: 8,756 bytes (8.55 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <algorithm>
#include <cstdint>

#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/SmallVector.h>

#define C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE 5

namespace c10::impl {

// Packed container for TensorImpl sizes and strides.
// This design improves on the previous approach of using a pair of
// c10::SmallVector<int64_t, 5> by specializing for the operations we
// actually use and enforcing that the number of sizes is the same as
// the number of strides. The memory layout is as follows:
//
// 1 size_t for the size
// 5 eightbytes of inline sizes and 5 eightbytes of inline strides, OR pointer
// to out-of-line array
class C10_API SizesAndStrides {
 public:
  // TODO: different iterator types for sizes & strides to prevent
  // mixing the two accidentally.
  using sizes_iterator = int64_t*;
  using sizes_const_iterator = const int64_t*;
  using strides_iterator = int64_t*;
  using strides_const_iterator = const int64_t*;

  SizesAndStrides() {
    size_at_unchecked(0) = 0;
    stride_at_unchecked(0) = 1;
  }

  ~SizesAndStrides() {
    if (C10_UNLIKELY(!isInline())) {
      // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
      free(outOfLineStorage_);
    }
  }

  SizesAndStrides(const SizesAndStrides& rhs) : size_(rhs.size_) {
    if (C10_LIKELY(rhs.isInline())) {
      copyDataInline(rhs);
    } else {
      allocateOutOfLineStorage(size_);
      copyDataOutline(rhs);
    }
  }

  bool operator==(const SizesAndStrides& other) const {
    if (size_ != other.size_) {
      return false;
    }
    return !(
        isInline()
            ? std::memcmp(
                  inlineStorage_, other.inlineStorage_, sizeof(inlineStorage_))
            : std::memcmp(
                  outOfLineStorage_,
                  other.outOfLineStorage_,
                  storageBytes(size_)));
  }

  bool operator!=(const SizesAndStrides& other) const {
    return !(*this == other);
  }

  SizesAndStrides& operator=(const SizesAndStrides& rhs) {
    if (this == &rhs) {
      return *this;
    }
    if (C10_LIKELY(rhs.isInline())) {
      if (C10_UNLIKELY(!isInline())) {
        // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
        free(outOfLineStorage_);
      }
      copyDataInline(rhs);
    } else {
      if (isInline()) {
        allocateOutOfLineStorage(rhs.size_);
      } else {
        resizeOutOfLineStorage(rhs.size_);
      }
      copyDataOutline(rhs);
    }
    size_ = rhs.size_;
    return *this;
  }

  // Move from rhs. rhs.size() == 0 afterwards.
  SizesAndStrides(SizesAndStrides&& rhs) noexcept : size_(rhs.size_) {
    if (C10_LIKELY(isInline())) {
      memcpy(inlineStorage_, rhs.inlineStorage_, sizeof(inlineStorage_));
    } else {
      outOfLineStorage_ = rhs.outOfLineStorage_;
      rhs.outOfLineStorage_ = nullptr;
    }

    rhs.size_ = 0;
  }

  // Move from rhs. rhs.size() == 0 afterwards.
  SizesAndStrides& operator=(SizesAndStrides&& rhs) noexcept {
    if (this == &rhs) {
      return *this;
    }
    if (C10_LIKELY(rhs.isInline())) {
      if (C10_UNLIKELY(!isInline())) {
        // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
        free(outOfLineStorage_);
      }
      copyDataInline(rhs);
    } else {
      // They're outline. We're going to steal their vector.
      if (!isInline()) {
        // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
        free(outOfLineStorage_);
      }
      outOfLineStorage_ = rhs.outOfLineStorage_;
      rhs.outOfLineStorage_ = nullptr;
    }
    size_ = rhs.size_;
    rhs.size_ = 0;

    return *this;
  }

  size_t size() const noexcept {
    return size_;
  }

  const int64_t* sizes_data() const noexcept {
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[0];
    } else {
      return &outOfLineStorage_[0];
    }
  }

  int64_t* sizes_data() noexcept {
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[0];
    } else {
      return &outOfLineStorage_[0];
    }
  }

  sizes_const_iterator sizes_begin() const noexcept {
    return sizes_data();
  }

  sizes_iterator sizes_begin() noexcept {
    return sizes_data();
  }

  sizes_const_iterator sizes_end() const noexcept {
    return sizes_begin() + size();
  }

  sizes_iterator sizes_end() noexcept {
    return sizes_begin() + size();
  }

  IntArrayRef sizes_arrayref() const noexcept {
    return IntArrayRef{sizes_data(), size()};
  }

  void set_sizes(IntArrayRef newSizes) {
    resize(newSizes.size());
    std::copy(newSizes.begin(), newSizes.end(), sizes_begin());
  }

  void set_strides(IntArrayRef strides) {
    TORCH_INTERNAL_ASSERT(strides.size() == size());
    std::copy(strides.begin(), strides.end(), strides_begin());
  }

  const int64_t* strides_data() const noexcept {
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE];
    } else {
      return &outOfLineStorage_[size()];
    }
  }

  int64_t* strides_data() noexcept {
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE];
    } else {
      return &outOfLineStorage_[size()];
    }
  }

  strides_const_iterator strides_begin() const noexcept {
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE];
    } else {
      return &outOfLineStorage_[size()];
    }
  }

  strides_iterator strides_begin() noexcept {
    if (C10_LIKELY(isInline())) {
      return &inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE];
    } else {
      return &outOfLineStorage_[size()];
    }
  }

  strides_const_iterator strides_end() const noexcept {
    return strides_begin() + size();
  }

  strides_iterator strides_end() noexcept {
    return strides_begin() + size();
  }

  IntArrayRef strides_arrayref() const noexcept {
    return IntArrayRef{strides_data(), size()};
  }

  // Size accessors.
  int64_t size_at(size_t idx) const noexcept {
    assert(idx < size());
    return sizes_data()[idx];
  }

  int64_t& size_at(size_t idx) noexcept {
    assert(idx < size());
    return sizes_data()[idx];
  }

  int64_t size_at_unchecked(size_t idx) const noexcept {
    return sizes_data()[idx];
  }

  int64_t& size_at_unchecked(size_t idx) noexcept {
    return sizes_data()[idx];
  }

  // Size accessors.
  int64_t stride_at(size_t idx) const noexcept {
    assert(idx < size());
    return strides_data()[idx];
  }

  int64_t& stride_at(size_t idx) noexcept {
    assert(idx < size());
    return strides_data()[idx];
  }

  int64_t stride_at_unchecked(size_t idx) const noexcept {
    return strides_data()[idx];
  }

  int64_t& stride_at_unchecked(size_t idx) noexcept {
    return strides_data()[idx];
  }

  void resize(size_t newSize) {
    const auto oldSize = size();
    if (newSize == oldSize) {
      return;
    }
    if (C10_LIKELY(
            newSize <= C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE && isInline())) {
      if (oldSize < newSize) {
        const auto bytesToZero =
            (newSize - oldSize) * sizeof(inlineStorage_[0]);
        memset(&inlineStorage_[oldSize], 0, bytesToZero);
        memset(
            &inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE + oldSize],
            0,
            bytesToZero);
      }
      size_ = newSize;
    } else {
      resizeSlowPath(newSize, oldSize);
    }
  }

  void resizeSlowPath(size_t newSize, size_t oldSize);

 private:
  bool isInline() const noexcept {
    return size_ <= C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE;
  }

  void copyDataInline(const SizesAndStrides& rhs) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(rhs.isInline());
    memcpy(inlineStorage_, rhs.inlineStorage_, sizeof(inlineStorage_));
  }

  static size_t storageBytes(size_t size) noexcept {
    return size * 2 * sizeof(int64_t);
  }

  void allocateOutOfLineStorage(size_t size) {
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    outOfLineStorage_ = static_cast<int64_t*>(malloc(storageBytes(size)));
    TORCH_CHECK(
        outOfLineStorage_,
        "Could not allocate memory for Tensor SizesAndStrides!");
  }

  void resizeOutOfLineStorage(size_t newSize) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!isInline());
    outOfLineStorage_ = static_cast<int64_t*>(
        // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
        realloc(outOfLineStorage_, storageBytes(newSize)));
    TORCH_CHECK(
        outOfLineStorage_,
        "Could not allocate memory for Tensor SizesAndStrides!");
  }

  void copyDataOutline(const SizesAndStrides& rhs) noexcept {
    memcpy(outOfLineStorage_, rhs.outOfLineStorage_, storageBytes(rhs.size_));
  }

  size_t size_{1};
  union {
    int64_t* outOfLineStorage_;
    // NOLINTNEXTLINE(*c-array*)
    int64_t inlineStorage_[C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE * 2]{};
  };
};

} // namespace c10::impl

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 39 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`

**Classes/Structs**: `C10_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/core/impl`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `algorithm`
- `cstdint`
- `c10/macros/Macros.h`
- `c10/util/ArrayRef.h`
- `c10/util/SmallVector.h`


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

Files in the same folder (`c10/core/impl`):

- [`LocalDispatchKeySet.h_docs.md`](./LocalDispatchKeySet.h_docs.md)
- [`TorchDispatchModeTLS.cpp_docs.md`](./TorchDispatchModeTLS.cpp_docs.md)
- [`PythonDispatcherTLS.h_docs.md`](./PythonDispatcherTLS.h_docs.md)
- [`PyInterpreter.h_docs.md`](./PyInterpreter.h_docs.md)
- [`alloc_cpu.h_docs.md`](./alloc_cpu.h_docs.md)
- [`PythonDispatcherTLS.cpp_docs.md`](./PythonDispatcherTLS.cpp_docs.md)
- [`InlineEvent.h_docs.md`](./InlineEvent.h_docs.md)
- [`PyInterpreterHooks.h_docs.md`](./PyInterpreterHooks.h_docs.md)
- [`LocalDispatchKeySet.cpp_docs.md`](./LocalDispatchKeySet.cpp_docs.md)
- [`DeviceGuardImplInterface.cpp_docs.md`](./DeviceGuardImplInterface.cpp_docs.md)


## Cross-References

- **File Documentation**: `SizesAndStrides.h_docs.md`
- **Keyword Index**: `SizesAndStrides.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
