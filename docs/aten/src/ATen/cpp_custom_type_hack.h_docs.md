# Documentation: `aten/src/ATen/cpp_custom_type_hack.h`

## File Metadata

- **Path**: `aten/src/ATen/cpp_custom_type_hack.h`
- **Size**: 5,430 bytes (5.30 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP

// YOU ARE IN THE WRONG PLACE! TURN BACK NOW!

// This code was a temporary hack to enable embedding arbitrary C++ structures
// into Tensors. THIS IS UNSAFE AND IS NOT SUPPORTED. IF YOU USE THIS CODE,
// IT __WILL__ BREAK.

// This code has been superseded by custom classes:
// https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html

// Please use custom classes and **DO NOT ADD MORE CALLSITES TO THINGS DEFINED
// IN THIS FILE**.

// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP
// STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP STOP

#include <ATen/TracerMode.h>
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace at::cpp_custom_type_hack {

template <typename T>
[[deprecated(
    "Use custom classes instead: "
    "https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html")]] bool
isa(const Tensor& packed) {
  return (packed.scalar_type() == kByte) &&
      (packed.storage().data_ptr().get_deleter() ==
       caffe2::TypeMeta::Make<T>().deleteFn());
}

template <typename T>
[[deprecated(
    "Use custom classes instead: "
    "https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html")]] T&
cast(const Tensor& packed) {
  TORCH_CHECK(
      packed.scalar_type() == kByte, "Expected temporary cpp type wrapper");
  TORCH_CHECK(
      packed.storage().data_ptr().get_deleter() ==
          caffe2::TypeMeta::Make<T>().deleteFn(),
      "Expected temporary cpp type wrapper of type ",
      caffe2::TypeMeta::TypeName<T>());
  return *reinterpret_cast<T*>(packed.storage().data_ptr().get());
}

template <typename T>
[[deprecated(
    "Use custom classes instead: "
    "https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html")]] Tensor
create(std::unique_ptr<T> ptr, TensorOptions options) {
  // None of this should trace, so turn off Tracer dispatching
  at::AutoDispatchBelowADInplaceOrView guard; // TODO: remove
  at::tracer::impl::NoTracerDispatchMode tracer_guard;

  // We store this instance away in a Tensor and register a deleter function
  // so that we do not leak memory. On the other side, we pull out the storage's
  // data_ptr and get the right typed pointer.
  void* raw_ptr = ptr.release();
  at::DataPtr at_ptr(
      raw_ptr, raw_ptr, caffe2::TypeMeta::Make<T>().deleteFn(), at::kCPU);

  // size doesn't really matter, but we can align it to the actual size
  // returning variables because one likely want to use this hack from python
  auto retval = at::empty({sizeof(T)}, options.device(kCPU).dtype(at::kByte));
  retval.storage().set_data_ptr_noswap(std::move(at_ptr));
  return retval;
}

} // namespace at::cpp_custom_type_hack

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

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

- `ATen/TracerMode.h`
- `ATen/core/Tensor.h`
- `ATen/Functions.h`
- `ATen/ops/empty.h`


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

- **File Documentation**: `cpp_custom_type_hack.h_docs.md`
- **Keyword Index**: `cpp_custom_type_hack.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
