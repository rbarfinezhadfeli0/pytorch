# Documentation: `aten/src/ATen/native/cpu/LinearAlgebraKernel.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/cpu/LinearAlgebraKernel.cpp`
- **Size**: 2,733 bytes (2.67 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/LinearAlgebra.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/cpu/Reduce.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/util/irange.h>

namespace at::native { namespace {

void addr_kernel(TensorIterator &iter,
                 const Scalar& beta, const Scalar& alpha) {
  if (iter.dtype() == ScalarType::Bool) {
    using scalar_t = bool;
    auto beta_val = beta.to<scalar_t>();
    auto alpha_val = alpha.to<scalar_t>();

    // when beta is false, values in self should be ignored,
    // nans and infs in self should not propagate.
    if (beta_val == false) {
      cpu_kernel(iter,
        [=](scalar_t /*self_val*/,
            scalar_t vec1_val,
            scalar_t vec2_val) -> scalar_t {
          return alpha_val && vec1_val && vec2_val;
        }
      );
    } else {
      cpu_kernel(iter,
        [=](scalar_t self_val,
            scalar_t vec1_val,
            scalar_t vec2_val) -> scalar_t {
          return (beta_val && self_val) || (alpha_val && vec1_val && vec2_val);
        }
      );
    }
    return;
  }

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf,
    iter.dtype(), "addr_cpu", [&]() {
      using Vec = Vectorized<scalar_t>;

      auto beta_val = beta.to<scalar_t>();
      auto alpha_val = alpha.to<scalar_t>();

      auto beta_vec = Vec(beta_val);
      auto alpha_vec = Vec(alpha_val);

      const scalar_t zero_val(0);
      // when beta == 0, values in self should be ignored,
      // nans and infs in self should not propagate.
      if (beta_val == zero_val) {
        cpu_kernel_vec(iter,
          [=](scalar_t /*self_val*/,
              scalar_t vec1_val,
              scalar_t vec2_val) __ubsan_ignore_undefined__ -> scalar_t {
            return alpha_val * vec1_val * vec2_val;
          },
          [=](Vec /*self_vec*/,
              Vec vec1_vec,
              Vec vec2_vec) __ubsan_ignore_undefined__ {
            return alpha_vec * vec1_vec * vec2_vec;
          }
        );
      } else {
        cpu_kernel_vec(iter,
          [=](scalar_t self_val,
              scalar_t vec1_val,
              scalar_t vec2_val) __ubsan_ignore_undefined__ -> scalar_t {
            return beta_val * self_val + alpha_val * vec1_val * vec2_val;
          },
          [=](Vec self_vec,
              Vec vec1_vec,
              Vec vec2_vec) __ubsan_ignore_undefined__ {
            return beta_vec * self_vec + alpha_vec * vec1_vec * vec2_vec;
          }
        );
      }
    }
  );
}

} // anonymous namespace

REGISTER_DISPATCH(addr_stub, &addr_kernel)
} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

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

- `ATen/native/LinearAlgebra.h`
- `ATen/core/Tensor.h`
- `ATen/Dispatch.h`
- `ATen/native/TensorIterator.h`
- `ATen/native/SharedReduceOps.h`
- `ATen/native/cpu/Reduce.h`
- `ATen/native/cpu/Loops.h`
- `c10/util/irange.h`


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

- **File Documentation**: `LinearAlgebraKernel.cpp_docs.md`
- **Keyword Index**: `LinearAlgebraKernel.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
