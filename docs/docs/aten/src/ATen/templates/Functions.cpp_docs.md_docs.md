# Documentation: `docs/aten/src/ATen/templates/Functions.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/templates/Functions.cpp_docs.md`
- **Size**: 5,533 bytes (5.40 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/templates/Functions.cpp`

## File Metadata

- **Path**: `aten/src/ATen/templates/Functions.cpp`
- **Size**: 3,107 bytes (3.03 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <array>

#include <ATen/Functions.h>
#include <ATen/Utils.h>
#include <c10/core/Allocator.h>

namespace at {

Tensor TensorMaker::make_tensor() {
   AutoDispatchBelowADInplaceOrView guard{}; // TODO: Remove.
   tracer::impl::NoTracerDispatchMode tracer_guard{};

   check_size_nonnegative(sizes_);

   TORCH_CHECK_VALUE(
       !deleter_ || !ctx_,
       "The deleter and context arguments are mutually exclusive.");

   if (device_ == std::nullopt) {
     device_ = globalContext().getDeviceFromPtr(data_, opts_.device().type());
   }

   if (opts_.device().has_index()) {
     // clang-format off
     TORCH_CHECK_VALUE(
         opts_.device() == *device_,
         "Specified device ", opts_.device(), " does not match device of data ", *device_);
     // clang-format on
   }

   std::size_t size_bytes = computeStorageSize();

   DataPtr data_ptr{};
   if (deleter_) {
     data_ptr = makeDataPtrFromDeleter();
   } else {
     data_ptr = makeDataPtrFromContext();
   }

   TORCH_CHECK(!resizeable_ || allocator_ != nullptr, "Must specify an allocator with allocator() if you want to use resizeable_storage()");
   Storage storage{Storage::use_byte_size_t{}, size_bytes, std::move(data_ptr), /*allocator=*/allocator_, /*resizable=*/resizeable_};

   Tensor tensor = detail::make_tensor<TensorImpl>(
       std::move(storage), opts_.computeDispatchKey(), opts_.dtype());

  TensorImpl* tensor_impl = tensor.unsafeGetTensorImpl();
  if (strides_) {
    tensor_impl->set_sizes_and_strides(sizes_, *strides_);
  } else {
    tensor_impl->set_sizes_contiguous(sizes_);
  }
  if (storage_offset_) {
    tensor_impl->set_storage_offset(*storage_offset_);
  }

  tensor_impl->set_requires_grad(opts_.requires_grad());

  return tensor;
 }

 std::size_t TensorMaker::computeStorageSize() const noexcept {
   std::size_t itemsize = opts_.dtype().itemsize();

   if (strides_) {
     auto storage_size = detail::computeStorageNbytes(sizes_, *strides_, itemsize);
     if (storage_offset_) {
       storage_size += storage_offset_.value() * itemsize;
     }
     return storage_size;
   }

   std::size_t size = 1;
   for (std::int64_t s : sizes_) {
     size *= static_cast<std::size_t>(s);
   }
   auto storage_size = size * itemsize;
   if (storage_offset_) {
     storage_size += storage_offset_.value() * itemsize;
   }
   return storage_size;
 }

 inline DataPtr TensorMaker::makeDataPtrFromDeleter() noexcept {
   return InefficientStdFunctionContext::makeDataPtr(data_, std::move(deleter_), *device_);
 }

 inline DataPtr TensorMaker::makeDataPtrFromContext() noexcept {
   return DataPtr{data_, ctx_.release(), ctx_.get_deleter(), *device_};
 }

 IntArrayRef TensorMaker::makeTempSizes() const noexcept {
   static std::int64_t zeros[5] = {0, 0, 0, 0, 0};
   if (opts_.has_memory_format()) {
     MemoryFormat format = *opts_.memory_format_opt();
     if (format == MemoryFormat::ChannelsLast) {
       return IntArrayRef(zeros, 4);
     }
     if (format == MemoryFormat::ChannelsLast3d) {
       return IntArrayRef(zeros, 5);
     }
   }
   return IntArrayRef(zeros, 1);
 }

} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/templates`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `array`
- `ATen/Functions.h`
- `ATen/Utils.h`
- `c10/core/Allocator.h`


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

Files in the same folder (`aten/src/ATen/templates`):

- [`NativeFunction.h_docs.md`](./NativeFunction.h_docs.md)
- [`DispatchKeyFunctions.h_docs.md`](./DispatchKeyFunctions.h_docs.md)
- [`aten_interned_strings.h_docs.md`](./aten_interned_strings.h_docs.md)
- [`UfuncCPUKernel.cpp_docs.md`](./UfuncCPUKernel.cpp_docs.md)
- [`DispatchKeyFunction.h_docs.md`](./DispatchKeyFunction.h_docs.md)
- [`LazyIr.h_docs.md`](./LazyIr.h_docs.md)
- [`RegisterDispatchDefinitions.ini_docs.md`](./RegisterDispatchDefinitions.ini_docs.md)
- [`RegisterDispatchKey.cpp_docs.md`](./RegisterDispatchKey.cpp_docs.md)
- [`MethodOperators.h_docs.md`](./MethodOperators.h_docs.md)


## Cross-References

- **File Documentation**: `Functions.cpp_docs.md`
- **Keyword Index**: `Functions.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/templates`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/templates`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/templates`):

- [`UfuncCPU.cpp_docs.md_docs.md`](./UfuncCPU.cpp_docs.md_docs.md)
- [`DispatchKeyNativeFunctions.h_kw.md_docs.md`](./DispatchKeyNativeFunctions.h_kw.md_docs.md)
- [`Function.h_docs.md_docs.md`](./Function.h_docs.md_docs.md)
- [`RedispatchFunctions.h_docs.md_docs.md`](./RedispatchFunctions.h_docs.md_docs.md)
- [`NativeFunction.h_kw.md_docs.md`](./NativeFunction.h_kw.md_docs.md)
- [`RegisterFunctionalization.cpp_kw.md_docs.md`](./RegisterFunctionalization.cpp_kw.md_docs.md)
- [`RegisterSchema.cpp_docs.md_docs.md`](./RegisterSchema.cpp_docs.md_docs.md)
- [`Operators.cpp_docs.md_docs.md`](./Operators.cpp_docs.md_docs.md)
- [`RegisterFunctionalization.cpp_docs.md_docs.md`](./RegisterFunctionalization.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `Functions.cpp_docs.md_docs.md`
- **Keyword Index**: `Functions.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
