# Documentation: `docs/aten/src/ATen/native/xnnpack/Shim.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/xnnpack/Shim.cpp_docs.md`
- **Size**: 4,801 bytes (4.69 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/xnnpack/Shim.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/xnnpack/Shim.cpp`
- **Size**: 2,401 bytes (2.34 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#ifndef USE_XNNPACK

#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/xnnpack/Engine.h>
#include <ATen/core/Tensor.h>

//
// This file is here so as to provide an implementation even in cases where
// PyTorch is compiled without XNNPACK support.  Under those scenarios, either
// all XNNPACK usage must be gated with #ifdefs at call-sites which would make
// for cluttered logic, or alternatively, all use can be routed to a central
// place, namely here, where available() calls return false preventing the
// XNNPACK related codepaths to be taken, and use of the actual operators
// trigger an error.
//

namespace at::native::xnnpack {
namespace internal {
namespace {

constexpr const char * const kError =
    "Not Implemented! Reason: PyTorch not built with XNNPACK support.";

} // namespace
} // namespace internal

bool available() {
    return false;
}

bool use_convolution2d(
    const Tensor& /*unused*/,
    const Tensor& /*unused*/,
    const at::OptionalIntArrayRef /*unused*/,
    const IntArrayRef /*unused*/,
    const IntArrayRef /*unused*/,
    const IntArrayRef /*unused*/,
    const int64_t /*unused*/,
    bool /*unused*/) {
  return false;
}

Tensor convolution2d(
    const Tensor& /*unused*/,
    const Tensor& /*unused*/,
    const Tensor& /*unused*/,
    const IntArrayRef /*unused*/,
    const IntArrayRef /*unused*/,
    const IntArrayRef /*unused*/,
    const int64_t /*unused*/) {
  TORCH_CHECK(false, internal::kError);
}

bool use_linear(
    const Tensor& /*unused*/,
    const Tensor& /*unused*/,
    const Tensor& /*unused*/) {
  return false;
}

Tensor linear(
    const Tensor& /*unused*/,
    const Tensor& /*unused*/,
    const Tensor& /*unused*/) {
  TORCH_CHECK(false, internal::kError);
}

bool use_max_pool2d(
    const Tensor& /*unused*/,
    const IntArrayRef /*unused*/,
    const IntArrayRef /*unused*/,
    IntArrayRef /*unused*/,
    const IntArrayRef /*unused*/,
    const bool /*unused*/,
    const float /*unused*/,
    const float /*unused*/) {
  return false;
}

Tensor max_pool2d(
    const Tensor& /*unused*/,
    const IntArrayRef /*unused*/,
    const IntArrayRef /*unused*/,
    IntArrayRef /*unused*/,
    const IntArrayRef /*unused*/,
    const bool /*unused*/,
    const float /*unused*/,
    const float /*unused*/) {
  TORCH_CHECK(false, internal::kError);
}

} // namespace at::native::xnnpack

#endif /* USE_XNNPACK */

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `internal`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/xnnpack`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/xnnpack/Common.h`
- `ATen/native/xnnpack/Engine.h`
- `ATen/core/Tensor.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`aten/src/ATen/native/xnnpack`):

- [`Engine.h_docs.md`](./Engine.h_docs.md)
- [`Linear.cpp_docs.md`](./Linear.cpp_docs.md)
- [`ChannelShuffle.cpp_docs.md`](./ChannelShuffle.cpp_docs.md)
- [`Convolution.h_docs.md`](./Convolution.h_docs.md)
- [`RegisterOpContextClass.cpp_docs.md`](./RegisterOpContextClass.cpp_docs.md)
- [`Common.h_docs.md`](./Common.h_docs.md)
- [`Convolution.cpp_docs.md`](./Convolution.cpp_docs.md)
- [`Activation.cpp_docs.md`](./Activation.cpp_docs.md)
- [`Linear.h_docs.md`](./Linear.h_docs.md)


## Cross-References

- **File Documentation**: `Shim.cpp_docs.md`
- **Keyword Index**: `Shim.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/xnnpack`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/xnnpack`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/aten/src/ATen/native/xnnpack`):

- [`MaxPooling.cpp_docs.md_docs.md`](./MaxPooling.cpp_docs.md_docs.md)
- [`Convolution.cpp_docs.md_docs.md`](./Convolution.cpp_docs.md_docs.md)
- [`Common.h_kw.md_docs.md`](./Common.h_kw.md_docs.md)
- [`Pooling.h_docs.md_docs.md`](./Pooling.h_docs.md_docs.md)
- [`RegisterOpContextClass.cpp_kw.md_docs.md`](./RegisterOpContextClass.cpp_kw.md_docs.md)
- [`AveragePooling.cpp_kw.md_docs.md`](./AveragePooling.cpp_kw.md_docs.md)
- [`OpContext.cpp_kw.md_docs.md`](./OpContext.cpp_kw.md_docs.md)
- [`ChannelShuffle.cpp_docs.md_docs.md`](./ChannelShuffle.cpp_docs.md_docs.md)
- [`MaxPooling.cpp_kw.md_docs.md`](./MaxPooling.cpp_kw.md_docs.md)
- [`Common.h_docs.md_docs.md`](./Common.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `Shim.cpp_docs.md_docs.md`
- **Keyword Index**: `Shim.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
