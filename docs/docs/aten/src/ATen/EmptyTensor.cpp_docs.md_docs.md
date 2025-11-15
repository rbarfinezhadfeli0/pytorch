# Documentation: `docs/aten/src/ATen/EmptyTensor.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/EmptyTensor.cpp_docs.md`
- **Size**: 19,010 bytes (18.56 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/EmptyTensor.cpp`

## File Metadata

- **Path**: `aten/src/ATen/EmptyTensor.cpp`
- **Size**: 16,320 bytes (15.94 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/EmptyTensor.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/detail/XPUHooksInterface.h>
#include <ATen/Context.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/CPUAllocator.h>
#include <c10/util/safe_numerics.h>

#include <limits>

namespace at::detail {
namespace {
c10::Allocator* GetCPUAllocatorMaybePinned(bool pin_memory) {
  if (pin_memory) {
    // NB: This is not quite right, if you somehow had both CUDA and PrivateUse1 initialized
    // in the same PyTorch build, you would ONLY ever get the CUDA pinned memory allocator.
    // To properly support this, see https://github.com/pytorch/pytorch/issues/14560

    std::optional<c10::DeviceType> opt_device_type = std::nullopt;
    // As mentioned in Note [Accelerator Context], the accelerators in PyTorch should be mutually exclusive,
    // and PrivateUse1 has the highest priority, followed by CUDA;
    // However, since exclusivity between accelerators cannot be guaranteed at present,
    // in order to ensure backward compatibility (previously the default was CUDA), CUDA are prioritized.
    if (at::globalContext().hasCUDA()) {
      opt_device_type = c10::DeviceType::CUDA;
    } else {
      opt_device_type = at::getAccelerator(false);
    }
    if (opt_device_type.has_value()) {
      return at::globalContext().getPinnedMemoryAllocator(opt_device_type);
    } else {
      TORCH_CHECK(
          false,
          "pin_memory=True requires a CUDA or other accelerator backend; "
          "no pinned memory allocator is available on this system.")
    }
  }

  return c10::GetCPUAllocator();
}

#ifndef C10_MOBILE
constexpr uint64_t storage_max() {
  // int64_t and size_t are used somewhat inconsistently throughout ATen.
  // To be safe, storage size calculations must fit in both types.
  constexpr auto int64_max = static_cast<uint64_t>(
      std::numeric_limits<int64_t>::max());
  constexpr auto size_max = static_cast<uint64_t>(
      std::numeric_limits<size_t>::max());
  return std::min(int64_max, size_max);
}
#endif

inline void raise_warning_for_complex_half(ScalarType dtype) {
  if (dtype == kComplexHalf) {
    TORCH_WARN_ONCE(
        "ComplexHalf support is experimental and many operators don't support it yet.");
  }
}

}  // namespace (anonymous)

size_t computeStorageNbytesContiguous(
    IntArrayRef sizes,
    size_t itemsize_bytes,
    size_t storage_offset
  ) {
  // Ignore overflow checks on mobile
#ifndef C10_MOBILE
  uint64_t size = 1;
  bool overflowed = c10::safe_multiplies_u64(sizes, &size);
  overflowed |= c10::add_overflows(size, storage_offset, &size);
  overflowed |= c10::mul_overflows(size, itemsize_bytes, &size);
  overflowed |= size > storage_max();
  TORCH_CHECK(!overflowed,
              "Storage size calculation overflowed with sizes=", sizes);
  return static_cast<size_t>(size);
#else
  const auto numel = c10::multiply_integers(sizes);
  return itemsize_bytes * (storage_offset + numel);
#endif
}

size_t computeStorageNbytes(
    IntArrayRef sizes,
    IntArrayRef strides,
    size_t itemsize_bytes,
    size_t storage_offset
  ) {
  TORCH_CHECK(
    sizes.size() == strides.size(),
    "dimensionality of sizes (",
    sizes.size(),
    ") must match dimensionality of strides (",
    strides.size(),
    ")");

  // Ignore overflow checks on mobile
#ifndef C10_MOBILE
  // size of the underlying storage is 1 bigger than the offset
  // of the last element according to stride
  uint64_t size = storage_offset + 1;
  bool overflowed = false;
  for (const auto i : c10::irange(sizes.size())) {
    if (sizes[i] == 0) {
      return 0;
    }

    uint64_t strided_size = 0;
    overflowed |= c10::mul_overflows(strides[i], sizes[i] - 1, &strided_size);
    overflowed |= c10::add_overflows(size, strided_size, &size);
  }
  overflowed |= c10::mul_overflows(size, itemsize_bytes, &size);
  overflowed |= size > storage_max();
  TORCH_CHECK(!overflowed,
              "Storage size calculation overflowed with sizes=",
              sizes, " and strides=", strides);
  return static_cast<size_t>(size);
#else
  // size of the underlying storage is 1 bigger than the offset
  // of the last element according to stride
  uint64_t size = 1;
  for (const auto i : c10::irange(sizes.size())) {
    if (sizes[i] == 0) {
      return 0;
    }

    size += strides[i] * (sizes[i] - 1);
  }
  return itemsize_bytes * (storage_offset + size);
#endif
}

SymInt computeStorageNbytesContiguous(
    SymIntArrayRef sizes,
    const SymInt& itemsize_bytes,
    const SymInt& storage_offset
  ) {
  const auto numel = c10::multiply_integers(sizes);
  return itemsize_bytes * (storage_offset + numel);
}

// not including mobile-only macros in this function,
// since mobile shouldn't be using symints.
SymInt computeStorageNbytes(
    SymIntArrayRef sizes,
    SymIntArrayRef strides,
    const SymInt& itemsize_bytes,
    const SymInt& storage_offset
  ) {
  TORCH_CHECK(
    sizes.size() == strides.size(),
    "dimensionality of sizes (",
    sizes.size(),
    ") must match dimensionality of strides (",
    strides.size(),
    ")");

  // size of the underlying storage is 1 bigger than the offset
  // of the last element according to stride
  SymInt size = 1;
  for (const auto i : c10::irange(sizes.size())) {
    if (TORCH_GUARD_OR_FALSE(sizes[i].sym_eq(0))) {
      return 0;
    }

    // NOTE: while this can technically return negative sizes for
    // 0-element tensors, there's a check in TensorShape:set_storage_meta__symint
    // that skips setting nbytes with unbacked expressions.
    // Would probably be safer to wrap this with a max(*, 0),
    // once our min/max symbolic reasoning improves.
    size += strides[i] * (sizes[i] - 1);
  }
  return itemsize_bytes * (storage_offset + size);
}

template <typename T>
static TensorBase _empty_generic(
    ArrayRef<T> size,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  at::detail::check_size_nonnegative(size);
  at::detail::raise_warning_for_complex_half(scalar_type);
  caffe2::TypeMeta dtype = scalarTypeToTypeMeta(scalar_type);
  auto size_bytes = computeStorageNbytesContiguous(size, dtype.itemsize());
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator,
      /*resizeable=*/true);

  auto tensor = detail::make_tensor_base<TensorImpl>(
      std::move(storage_impl), ks, dtype);
  // Default TensorImpl has size [0]
  // NB: test for meta dispatch key to avoid guarding on zero-ness
  if (ks.has(c10::DispatchKey::Meta) || size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->generic_set_sizes_contiguous(size);
  }

  if (memory_format_opt.has_value()) {
    // Restriding a just-created empty contiguous tensor does nothing.
    if (*memory_format_opt != MemoryFormat::Contiguous) {
      tensor.unsafeGetTensorImpl()->empty_tensor_restride(*memory_format_opt);
    }
  }

  return tensor;
}

TensorBase empty_generic(
    IntArrayRef size,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  return _empty_generic(size, allocator, ks, scalar_type, memory_format_opt);
}

TensorBase empty_generic_symint(
    SymIntArrayRef size,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  return _empty_generic(size, allocator, ks, scalar_type, memory_format_opt);
}

template <typename T>
static TensorBase _empty_strided_generic(
    T size,
    T stride,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type) {
  at::detail::check_size_nonnegative(size);
  at::detail::raise_warning_for_complex_half(scalar_type);
  caffe2::TypeMeta dtype = scalarTypeToTypeMeta(scalar_type);
  auto size_bytes = computeStorageNbytes(size, stride, dtype.itemsize());
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator,
      /*resizeable=*/true);

  auto tensor = detail::make_tensor_base<TensorImpl>(
      std::move(storage_impl), ks, dtype);
  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
  return tensor;
}

TensorBase empty_strided_generic(
    IntArrayRef size,
    IntArrayRef stride,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type) {
  return _empty_strided_generic<IntArrayRef>(size, stride, allocator, ks, scalar_type);
}

TensorBase empty_strided_symint_generic(
    SymIntArrayRef size,
    SymIntArrayRef stride,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type) {
  return _empty_strided_generic<SymIntArrayRef>(size, stride, allocator, ks, scalar_type);
}

TensorBase empty_cpu(IntArrayRef size, ScalarType dtype, bool pin_memory,
                     std::optional<c10::MemoryFormat> memory_format_opt) {
  auto allocator = GetCPUAllocatorMaybePinned(pin_memory);
  constexpr c10::DispatchKeySet cpu_ks(c10::DispatchKey::CPU);
  return empty_generic(size, allocator, cpu_ks, dtype, memory_format_opt);
}

TensorBase empty_cpu(
    IntArrayRef size,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device_or_default(device_opt).type() == DeviceType::CPU);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == Layout::Strided);

  auto pin_memory = pinned_memory_or_default(pin_memory_opt);
  auto dtype = dtype_or_default(dtype_opt);
  return empty_cpu(size, dtype, pin_memory, memory_format_opt);
}

TensorBase empty_cpu(
    IntArrayRef size, const TensorOptions &options) {
  return at::detail::empty_cpu(
      size,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt(),
      options.memory_format_opt());
}

TensorBase empty_strided_cpu(IntArrayRef size, IntArrayRef stride,
                             ScalarType dtype, bool pin_memory) {
  auto allocator = at::detail::GetCPUAllocatorMaybePinned(pin_memory);
  constexpr c10::DispatchKeySet cpu_ks(c10::DispatchKey::CPU);
  return at::detail::empty_strided_generic(
      size, stride, allocator, cpu_ks, dtype);
}

TensorBase empty_strided_cpu(
    IntArrayRef size,
    IntArrayRef stride,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device_or_default(device_opt).type() == DeviceType::CPU);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == Layout::Strided);

  auto pin_memory = pinned_memory_or_default(pin_memory_opt);
  auto dtype = dtype_or_default(dtype_opt);
  return at::detail::empty_strided_cpu(size, stride, dtype, pin_memory);
}

TensorBase empty_strided_cpu(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions &options) {
  return at::detail::empty_strided_cpu(
      size,
      stride,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt());
}

// The meta allocator ignores whatever allocation is requested and always
// gives you nullptr
struct MetaAllocator final : public at::Allocator {
  MetaAllocator() = default;
  ~MetaAllocator() override = default;
  static void deleter(void* const pointer) {
    TORCH_INTERNAL_ASSERT(!pointer);
  }
  DataPtr allocate(const size_t nbytes [[maybe_unused]]) override {
    return {nullptr, nullptr, &deleter, at::Device(DeviceType::Meta)};
  }
  DeleterFnPtr raw_deleter() const override {
    return deleter;
  }
  void copy_data(void* dest, const void* src, std::size_t count) const final {}
};

static MetaAllocator g_meta_alloc;

REGISTER_ALLOCATOR(kMeta, &g_meta_alloc)

TensorBase empty_meta(IntArrayRef size, ScalarType dtype,
                     std::optional<c10::MemoryFormat> memory_format_opt) {
  auto *allocator = GetAllocator(kMeta);
  constexpr c10::DispatchKeySet meta_dks(c10::DispatchKey::Meta);
  return at::detail::empty_generic(
      size, allocator, meta_dks, dtype, memory_format_opt);
}

TensorBase empty_meta(
  IntArrayRef size,
  std::optional<ScalarType> dtype_opt,
  std::optional<Layout> layout_opt,
  std::optional<Device> device_opt,
  std::optional<bool> pin_memory_opt,
  std::optional<c10::MemoryFormat> memory_format_opt
) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device_or_default(device_opt).type() == DeviceType::Meta);
  // NB: because there is no SparseMeta (yet), non-strided layout is
  // exerciseable
  TORCH_CHECK_NOT_IMPLEMENTED(
    layout_or_default(layout_opt) == Layout::Strided,
    "non-strided meta tensors not supported yet"
  );

  auto dtype = dtype_or_default(dtype_opt);
  return empty_meta(size, dtype, memory_format_opt);
}

TensorBase empty_symint_meta(
  SymIntArrayRef size,
  std::optional<ScalarType> dtype_opt,
  std::optional<Layout> layout_opt,
  std::optional<Device> device_opt,
  std::optional<bool> pin_memory_opt,
  std::optional<c10::MemoryFormat> memory_format_opt
) {
  auto *allocator = GetAllocator(kMeta);
  constexpr c10::DispatchKeySet ks(c10::DispatchKey::Meta);
  auto scalar_type = dtype_or_default(dtype_opt);
  return _empty_generic(size, allocator, ks, scalar_type, memory_format_opt);
}

TensorBase empty_meta(
    IntArrayRef size, const TensorOptions &options) {
  return at::detail::empty_meta(
      size,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt(),
      options.memory_format_opt());
}

TensorBase empty_strided_meta(IntArrayRef size, IntArrayRef stride,
                              ScalarType dtype) {
  auto *allocator = GetAllocator(kMeta);
  constexpr c10::DispatchKeySet meta_dks(c10::DispatchKey::Meta);
  return at::detail::empty_strided_generic(
      size, stride, allocator, meta_dks, dtype);
}

TensorBase empty_strided_meta(
    IntArrayRef size,
    IntArrayRef stride,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device_or_default(device_opt).type() == DeviceType::Meta);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == Layout::Strided);

  auto dtype = dtype_or_default(dtype_opt);
  return at::detail::empty_strided_meta(size, stride, dtype);
}

TensorBase empty_strided_meta(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions &options) {
  return at::detail::empty_strided_meta(
      size,
      stride,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt());
}

TensorBase empty_strided_symint_meta(SymIntArrayRef size, SymIntArrayRef stride,
                              ScalarType dtype) {
  auto *allocator = GetAllocator(kMeta);
  constexpr c10::DispatchKeySet meta_dks(c10::DispatchKey::Meta);
  return at::detail::empty_strided_symint_generic(
      size, stride, allocator, meta_dks, dtype);
}

TensorBase empty_strided_symint_meta(
    SymIntArrayRef size,
    SymIntArrayRef stride,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device_or_default(device_opt).type() == DeviceType::Meta);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == Layout::Strided);

  auto dtype = dtype_or_default(dtype_opt);
  return at::detail::empty_strided_symint_meta(size, stride, dtype);
}

TensorBase empty_strided_symint_meta(
    SymIntArrayRef size,
    SymIntArrayRef stride,
    const TensorOptions &options) {
  return at::detail::empty_strided_symint_meta(
      size,
      stride,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt());
}

} // namespace at::detail

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 53 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `MetaAllocator`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/EmptyTensor.h`
- `ATen/detail/CUDAHooksInterface.h`
- `ATen/detail/XPUHooksInterface.h`
- `ATen/Context.h`
- `ATen/detail/PrivateUse1HooksInterface.h`
- `c10/core/CPUAllocator.h`
- `c10/util/safe_numerics.h`
- `limits`


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

- **File Documentation**: `EmptyTensor.cpp_docs.md`
- **Keyword Index**: `EmptyTensor.cpp_kw.md`
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

- **File Documentation**: `EmptyTensor.cpp_docs.md_docs.md`
- **Keyword Index**: `EmptyTensor.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
