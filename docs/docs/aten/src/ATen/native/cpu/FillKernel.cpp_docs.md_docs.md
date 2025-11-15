# Documentation: `docs/aten/src/ATen/native/cpu/FillKernel.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cpu/FillKernel.cpp_docs.md`
- **Size**: 5,729 bytes (5.59 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cpu/FillKernel.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/cpu/FillKernel.cpp`
- **Size**: 2,987 bytes (2.92 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch_v2.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

#include <ATen/native/Fill.h>
#include <c10/core/Scalar.h>

namespace at::native {
namespace {


template <typename scalar_t>
void fill_non_native_type(TensorIterator& iter, const Scalar& value_scalar) {
  auto value = value_scalar.to<scalar_t>().x;
  using H = typename std::make_signed_t<decltype(value)>;  // Signed type has more acceleration
  // Reserve the representation of value. static_cast<H>(value) is implementation defined.
  H val = *reinterpret_cast<H*>(std::addressof(value));
  cpu_kernel_vec</*check_dynamic_cast=*/false>(
      iter,
      [val]() -> H { return val; },
      [val]() { return Vectorized<H>(val); });
}

template <>
void fill_non_native_type<c10::complex<at::Half>>(TensorIterator& iter, const Scalar& value_scalar) {
  static_assert(sizeof(c10::complex<at::Half>) == sizeof(int32_t), "Size of ComplexHalf should be 32-bits");
  auto value = c10::complex<at::Half>(value_scalar.to<c10::complex<float>>());
  auto val = *reinterpret_cast<int32_t*>(std::addressof(value));
  cpu_kernel_vec</*check_dynamic_cast=*/false>(
      iter,
      [val]() -> int32_t { return val; },
      [val]() { return Vectorized<int32_t>(val); });
}

void fill_kernel(TensorIterator& iter, const Scalar& value_scalar) {
  if (iter.dtype() == ScalarType::Half) {
    fill_non_native_type<at::Half>(iter, value_scalar);
  } else if (iter.dtype() == ScalarType::BFloat16) {
    fill_non_native_type<at::BFloat16>(iter, value_scalar);
  } else if (iter.dtype() == ScalarType::ComplexHalf) {
    fill_non_native_type<c10::complex<at::Half>>(iter, value_scalar);
  } else if (iter.dtype() == ScalarType::Float8_e4m3fn) {
    fill_non_native_type<at::Float8_e4m3fn>(iter, value_scalar);
  } else if (iter.dtype() == ScalarType::Float8_e5m2) {
    fill_non_native_type<at::Float8_e5m2>(iter, value_scalar);
  } else if (iter.dtype() == ScalarType::Float8_e4m3fnuz) {
    fill_non_native_type<at::Float8_e4m3fnuz>(iter, value_scalar);
  } else if (iter.dtype() == ScalarType::Float8_e5m2fnuz) {
    fill_non_native_type<at::Float8_e5m2fnuz>(iter, value_scalar);
  } else if (iter.dtype() == ScalarType::Float8_e8m0fnu) {
    // TODO(#146647): use macro here instead of spelling out each float8 dtype
    fill_non_native_type<at::Float8_e8m0fnu>(iter, value_scalar);
  } else {
    AT_DISPATCH_V2(
      iter.dtype(), "fill_cpu", AT_WRAP([&]() {
        scalar_t value = value_scalar.to<scalar_t>();
        cpu_kernel_vec(
            iter,
            [=]() -> scalar_t { return value; },
            [=]() { return Vectorized<scalar_t>(value); });
      }),
      AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), kBool, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES)
    );
  }
}

} // namespace

REGISTER_DISPATCH(fill_stub, &fill_kernel)

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `REGISTER_DISPATCH`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Dispatch_v2.h`
- `ATen/Parallel.h`
- `ATen/cpu/vec/vec.h`
- `ATen/cpu/vec/functional.h`
- `ATen/native/TensorIterator.h`
- `ATen/native/cpu/Loops.h`
- `ATen/native/Fill.h`
- `c10/core/Scalar.h`


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

- **File Documentation**: `FillKernel.cpp_docs.md`
- **Keyword Index**: `FillKernel.cpp_kw.md`
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

- **File Documentation**: `FillKernel.cpp_docs.md_docs.md`
- **Keyword Index**: `FillKernel.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
