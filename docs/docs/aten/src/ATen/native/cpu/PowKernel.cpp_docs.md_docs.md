# Documentation: `docs/aten/src/ATen/native/cpu/PowKernel.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cpu/PowKernel.cpp_docs.md`
- **Size**: 8,029 bytes (7.84 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cpu/PowKernel.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/cpu/PowKernel.cpp`
- **Size**: 5,258 bytes (5.13 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_NO_OPERATORS
#include <cmath>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/Pow.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cpu/Loops.h>

#include <c10/core/Scalar.h>

namespace at::native {

inline namespace CPU_CAPABILITY {

static void pow_tensor_tensor_kernel(TensorIteratorBase& iter) {
  const auto dtype = iter.common_dtype();
  if (isFloatingType(dtype) || isComplexType(dtype)) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, dtype, "pow", [&]() {

      using Vec = Vectorized<scalar_t>;
      cpu_kernel_vec(iter,
        [=](scalar_t base, scalar_t exp) -> scalar_t {
          return std::pow(base, exp);
        },
        [&](Vec base, Vec exp) -> Vec {
          return base.pow(exp);
        }
      );
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(dtype, "pow", [&]() {
      cpu_kernel(iter,
        [=](scalar_t base, scalar_t exp) -> scalar_t {
          return native::powi(base, exp);
        }
      );
    });
  }
}

// The source-code of kernels for float, double and complex types is similar,
// barring a small distinction - even if the output dtype is float, a double
// exponent can be used. But Complex types' computation doesn't allow standard
// & double-precision to be mixed, since std::pow takes either complex64 inputs,
// or complex128 inputs, but not both. So, in order to provide a common path for
// float, double & complex types, template parameter cast_scalar_t is being used
// to resolve the aforementioned distinction. This approach also allows BFloat16
// to use this common-path. Half cannot currently use it, as AVX2 support for
// sqrt & rsqrt doesn't currently exist for it.
template <typename scalar_t, typename cast_scalar_t, typename exp_scalar_t>
void pow_tensor_scalar_optimized_kernel(TensorIteratorBase& iter, const exp_scalar_t exp) {
  using Vec = Vectorized<scalar_t>;
  // .5 (sqrt), -.5 (rsqrt) and -1 (reciprocal) specializations are handled
  // in pow_tensor_scalar_kernel
  if (exp == 2.0) {
    cpu_kernel_vec(iter,
        [](scalar_t base) -> scalar_t {
          return base * base;
        },
        [](Vec base) -> Vec { return base * base; }
    );
  } else if (exp == 3.0) {
    cpu_kernel_vec(iter,
        [](scalar_t base) -> scalar_t {
          return base * base * base;
        },
        [](Vec base) -> Vec { return base * base * base; }
    );
  } else if (exp == -2.0) {
    cpu_kernel_vec(iter,
        [](scalar_t base) __ubsan_ignore_float_divide_by_zero__ -> scalar_t {
          return static_cast<cast_scalar_t>(1.0) / (base * base); },
        [](Vec base) -> Vec { return (base * base).reciprocal(); }
    );
  } else {
    cpu_kernel_vec(iter,
        [=](scalar_t base) -> scalar_t {
          return std::pow(base, static_cast<cast_scalar_t>(exp));
        },
        [=](Vec base) -> Vec {
          return base.pow(static_cast<cast_scalar_t>(exp));
        }
    );
  }
}

static void pow_tensor_scalar_kernel(
    TensorIteratorBase& iter,
    const Scalar& exp_scalar) {
  // prevent multiple calls to iter.common_dtype()
  const auto dtype = iter.common_dtype();

  if (dtype == ScalarType::Float || dtype == ScalarType::Double ||
      dtype == kBFloat16 || isComplexType(dtype)) {
    // Dispatch to fast specialization for sqrt, rsqrt and reciprocal
    if (exp_scalar.equal(.5)) {
      sqrt_kernel(iter);
      return;
    } else if (exp_scalar.equal(-0.5)) {
      rsqrt_kernel(iter);
      return;
    } else if (exp_scalar.equal(-1.0)) {
      reciprocal_kernel(iter);
      return;
    }
  }

  if (dtype == ScalarType::Float || dtype == ScalarType::Double) {
    AT_DISPATCH_FLOATING_TYPES(dtype, "pow", [&]() {
      pow_tensor_scalar_optimized_kernel<scalar_t, double>(
          iter, exp_scalar.to<double>());
    });
  } else if (isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES(dtype, "pow", [&]() {
      pow_tensor_scalar_optimized_kernel<scalar_t, scalar_t>(
          iter, exp_scalar.to<c10::complex<double>>());
    });
  } else if (dtype == ScalarType::Half) {
    [&]() {
      using scalar_t =
          c10::impl::ScalarTypeToCPPTypeT<ScalarType::Half>;
      const auto exp = exp_scalar.to<scalar_t>();
      using Vec = Vectorized<scalar_t>;
      cpu_kernel_vec(iter,
          [=](scalar_t base) -> scalar_t {
            return std::pow(base, exp);
          },
          [=](Vec base) -> Vec { return base.pow(exp); }
      );
    }();
  } else if (dtype == ScalarType::BFloat16) {
      AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, dtype, "pow", [&]() {
        pow_tensor_scalar_optimized_kernel<scalar_t, scalar_t>(
            iter, exp_scalar.to<scalar_t>());
      });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(dtype, "pow", [&]() {
      const scalar_t exp = exp_scalar.to<scalar_t>();
      cpu_kernel(iter, [=](scalar_t base) -> scalar_t {
        return native::powi(base, exp);
      });
    });
  }
}

} // anonymous namespace

ALSO_REGISTER_AVX512_DISPATCH(pow_tensor_tensor_stub, &CPU_CAPABILITY::pow_tensor_tensor_kernel)
ALSO_REGISTER_AVX512_DISPATCH(pow_tensor_scalar_stub, &CPU_CAPABILITY::pow_tensor_scalar_kernel)

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 13 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `ALSO_REGISTER_AVX512_DISPATCH`, `CPU_CAPABILITY`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `cmath`
- `ATen/Dispatch.h`
- `ATen/Parallel.h`
- `ATen/cpu/vec/vec.h`
- `ATen/native/TensorIterator.h`
- `ATen/native/Pow.h`
- `ATen/native/UnaryOps.h`
- `ATen/native/cpu/Loops.h`
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

- **File Documentation**: `PowKernel.cpp_docs.md`
- **Keyword Index**: `PowKernel.cpp_kw.md`
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

- **File Documentation**: `PowKernel.cpp_docs.md_docs.md`
- **Keyword Index**: `PowKernel.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
