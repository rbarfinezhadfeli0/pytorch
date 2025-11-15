# Documentation: `aten/src/ATen/Utils.cpp`

## File Metadata

- **Path**: `aten/src/ATen/Utils.cpp`
- **Size**: 2,711 bytes (2.65 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/Utils.h>
#include <c10/util/accumulate.h>


#include <cstdlib>

namespace at {

int _crash_if_asan(int arg) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  volatile char x[3];
  x[arg] = 0;
  return x[0];
}

namespace detail {

template <typename T>
Tensor tensor_cpu(ArrayRef<T> values, const TensorOptions& options) {
  auto result = at::empty(values.size(), options);
  AT_ASSERT(result.is_contiguous());
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(result.scalar_type(), "tensor_cpu", [&] {
    std::copy(
        values.begin(), values.end(), result.template data_ptr<scalar_t>());
  });
  return result;
}

template <typename T>
Tensor tensor_backend(ArrayRef<T> values, const TensorOptions& options) {
  auto cpu_tensor = tensor_cpu(values, options.device(DeviceType::CPU));
  return cpu_tensor.to(options.device());
}

template <typename T>
Tensor tensor_complex_cpu(ArrayRef<T> values, const TensorOptions& options) {
  auto result = at::empty(values.size(), options);
  AT_ASSERT(result.is_contiguous());
  AT_DISPATCH_COMPLEX_TYPES(result.scalar_type(), "tensor_cpu", [&] {
    std::copy(
        values.begin(), values.end(), result.template data_ptr<scalar_t>());
  });
  return result;
}

template <typename T>
Tensor tensor_complex_backend(
    ArrayRef<T> values,
    const TensorOptions& options) {
  auto cpu_tensor = tensor_complex_cpu(values, options.device(DeviceType::CPU));
  return cpu_tensor.to(options.device());
}
} // namespace detail

#define TENSOR(T, _1)                                               \
  Tensor tensor(ArrayRef<T> values, const TensorOptions& options) { \
    if (options.device().type() != c10::DeviceType::CPU) {          \
      return at::detail::tensor_backend(values, options);           \
    } else {                                                        \
      return at::detail::tensor_cpu(values, options);               \
    }                                                               \
  }
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)
#undef TENSOR

#define TENSOR(T, _1)                                               \
  Tensor tensor(ArrayRef<T> values, const TensorOptions& options) { \
    if (options.device().type() != c10::DeviceType::CPU) {          \
      return at::detail::tensor_complex_backend(values, options);   \
    } else {                                                        \
      return at::detail::tensor_complex_cpu(values, options);       \
    }                                                               \
  }
AT_FORALL_COMPLEX_TYPES(TENSOR)
#undef TENSOR
} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `detail`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Context.h`
- `ATen/Dispatch.h`
- `ATen/Functions.h`
- `ATen/Utils.h`
- `c10/util/accumulate.h`
- `cstdlib`


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

- **File Documentation**: `Utils.cpp_docs.md`
- **Keyword Index**: `Utils.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
