# Documentation: `aten/src/ATen/templates/Functions.h`

## File Metadata

- **Path**: `aten/src/ATen/templates/Functions.h`
- **Size**: 4,637 bytes (4.53 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

// ${generated_comment}

#ifdef TORCH_ASSERT_NO_OPERATORS
#error This change adds a dependency on native_functions.yaml,            \
  meaning the file will need to be re-compiled every time an operator     \
  is changed or added. Consider if your change would be better placed in  \
  another file, or if a more specific header might achieve the same goal. \
  See NOTE: [Tensor vs. TensorBase]
#endif

#if defined(AT_PER_OPERATOR_HEADERS) && defined(TORCH_ASSERT_ONLY_METHOD_OPERATORS)
#error This change adds a dependency on all pytorch operators, meaning the     \
  file will need to be re-compiled every time an operator is changed or added. \
  Consider including a specific operator from <ATen/ops/{my_operator}.h> and   \
  see NOTE [TORCH_ASSERT_ONLY_METHOD_OPERATORS].
#endif

// NOTE: [TORCH_ASSERT_ONLY_METHOD_OPERATORS]
//
// In ATen, certain generated headers files include the definitions of
// every single operator in PyTorch. Unfortunately this means every
// time an operator signature is updated or changed in
// native_functions.yaml, you (and every other PyTorch developer) need
// to recompile every source file that includes any of these headers.
//
// To break up these header dependencies, and improve incremental
// build times for all PyTorch developers. These headers are split
// into per-operator headers in the `ATen/ops` folder. This limits
// incremental builds to only changes to methods of `Tensor`, or files
// that use the specific operator being changed. With `at::sum` as an
// example, you should include
//
//   <ATen/ops/sum.h>               // instead of ATen/Functions.h
//   <ATen/ops/sum_native.h>        // instead of ATen/NativeFunctions.h
//   <ATen/ops/sum_ops.h>           // instead of ATen/Operators.h
//   <ATen/ops/sum_cpu_dispatch.h>  // instead of ATen/CPUFunctions.h
//
// However, even if you're careful to use this in your own code.
// `Functions.h` might be included indirectly through another header
// without you realising. To avoid this, you can add
//
//   #define TORCH_ASSERT_ONLY_METHOD_OPERATORS
//
// to the top of your source file. This way any time the non-specific
// headers are included, the compiler will error out.
//
// Also, be aware that `ops` are not available in all build
// configurations (namely fb-internal) so you must guard these
// includes with `#ifdef AT_PER_OPERATOR_HEADERS`. e.g.
//
//   #ifndef AT_PER_OPERATOR_HEADERS
//   #include <ATen/Functions.h>
//   #else
//   #include <ATen/ops/sum.h>
//   #endif

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <c10/core/SymInt.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <optional>
#include <c10/util/OptionalArrayRef.h>

#include <ATen/ops/from_blob.h>
#include <ATen/ops/tensor.h>

${Functions_includes}

namespace at {

${Functions_declarations}

// Special C++ only overloads for std()-like functions (See gh-40287)
// These are needed because int -> bool conversion takes precedence over int -> IntArrayRef
// So, for example std(0) would select the std(unbiased=False) overload
inline Tensor var(const Tensor& self, int dim) {
  return at::var(self, IntArrayRef{dim});
}
inline std::tuple<Tensor, Tensor> var_mean(const Tensor& self, int dim) {
  return at::var_mean(self, IntArrayRef{dim});
}
inline Tensor std(const Tensor& self, int dim) {
  return at::std(self, IntArrayRef{dim});
}
inline std::tuple<Tensor, Tensor> std_mean(const Tensor& self, int dim) {
  return at::std_mean(self, IntArrayRef{dim});
}

inline int64_t numel(const Tensor& tensor) {
  return tensor.numel();
}

inline int64_t size(const Tensor& tensor, int64_t dim) {
  return tensor.size(dim);
}

inline int64_t stride(const Tensor& tensor, int64_t dim) {
  return tensor.stride(dim);
}

inline bool is_complex(const Tensor& tensor) {
  return tensor.is_complex();
}

inline bool is_floating_point(const Tensor& tensor) {
  return tensor.is_floating_point();
}

inline bool is_signed(const Tensor& tensor) {
  return tensor.is_signed();
}

inline bool is_inference(const Tensor& tensor) {
  return tensor.is_inference();
}

inline bool _is_zerotensor(const Tensor& tensor) {
  return tensor._is_zerotensor();
}

inline bool is_conj(const Tensor& tensor) {
  return tensor.is_conj();
}

inline Tensor conj(const Tensor& tensor) {
  return tensor.conj();
}

inline bool is_neg(const Tensor& tensor) {
  return tensor.is_neg();
}

}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 18 function(s).

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

- `ATen/Functions.h`
- `ATen/ops/sum.h`
- `ATen/Context.h`
- `ATen/DeviceGuard.h`
- `ATen/TensorUtils.h`
- `ATen/TracerMode.h`
- `ATen/core/Generator.h`
- `ATen/core/Reduction.h`
- `c10/core/SymInt.h`
- `ATen/core/Tensor.h`
- `c10/core/Scalar.h`
- `c10/core/Storage.h`
- `c10/core/TensorOptions.h`
- `c10/util/Deprecated.h`
- `optional`
- `c10/util/OptionalArrayRef.h`
- `ATen/ops/from_blob.h`
- `ATen/ops/tensor.h`


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

Files in the same folder (`aten/src/ATen/templates`):

- [`NativeFunction.h_docs.md`](./NativeFunction.h_docs.md)
- [`DispatchKeyFunctions.h_docs.md`](./DispatchKeyFunctions.h_docs.md)
- [`aten_interned_strings.h_docs.md`](./aten_interned_strings.h_docs.md)
- [`UfuncCPUKernel.cpp_docs.md`](./UfuncCPUKernel.cpp_docs.md)
- [`DispatchKeyFunction.h_docs.md`](./DispatchKeyFunction.h_docs.md)
- [`LazyIr.h_docs.md`](./LazyIr.h_docs.md)
- [`RegisterDispatchDefinitions.ini_docs.md`](./RegisterDispatchDefinitions.ini_docs.md)
- [`Functions.cpp_docs.md`](./Functions.cpp_docs.md)
- [`RegisterDispatchKey.cpp_docs.md`](./RegisterDispatchKey.cpp_docs.md)
- [`MethodOperators.h_docs.md`](./MethodOperators.h_docs.md)


## Cross-References

- **File Documentation**: `Functions.h_docs.md`
- **Keyword Index**: `Functions.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
