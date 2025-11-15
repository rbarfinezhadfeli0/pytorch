# Documentation: `docs/aten/src/ATen/native/cuda/PowKernel.cu_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/PowKernel.cu_docs.md`
- **Size**: 10,198 bytes (9.96 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/PowKernel.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/PowKernel.cu`
- **Size**: 7,350 bytes (7.18 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Pow.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/Pow.h>
#include <c10/core/Scalar.h>

namespace at::native {

// Forward declare some unary kernels
void rsqrt_kernel_cuda(TensorIteratorBase& iter);
void sqrt_kernel_cuda(TensorIteratorBase& iter);
void reciprocal_kernel_cuda(TensorIteratorBase& iter);

namespace {

void pow_tensor_scalar_kernel(TensorIteratorBase& iter, const Scalar& exp_scalar);

template <typename scalar_t>
void pow_scalar_tensor_impl(TensorIteratorBase& iter, scalar_t base) {
  gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t exp) -> scalar_t {
    return pow_(base, exp);
  });
}

template <typename value_t>
void pow_scalar_tensor_impl(TensorIteratorBase& iter, c10::complex<value_t> base) {
  // For complex, thrust::pow uses the identity
  // pow(a, b) = exp(log(a) * b)
  const auto fct = std::log(base);
  gpu_kernel(iter, [=]GPU_LAMBDA(c10::complex<value_t> exp) -> c10::complex<value_t> {
    return std::exp(fct * exp);
  });
}

/* complex<Half> support impl */
constexpr char pow_scalar_base_name[] = "pow_scalar_base_kernel";
template <>
void pow_scalar_tensor_impl(TensorIteratorBase& iter, c10::complex<at::Half> base) {
  using scalar_t = c10::complex<at::Half>;
  using opmath_t = at::opmath_type<scalar_t>;
  // For complex, thrust::pow uses the identity
  // pow(a, b) = exp(log(a) * b)
  const auto fct = std::log(opmath_t{base});
#if AT_USE_JITERATOR()
  static const auto pow_kernel_string =
      jiterator_stringify(template <typename T> T pow_scalar_base_kernel(T exp, T fct) {
        return std::exp(fct * exp);
      });
  jitted_gpu_kernel<pow_scalar_base_name, scalar_t, scalar_t, 1>(
      iter,
      pow_kernel_string,
      /*scalar_pos=*/at::cuda::jit::BinaryFuncVariant::NoScalar,
      /*scalar_val=*/0,
      /*extra_args=*/std::make_tuple(fct));
#else
  gpu_kernel(iter, [=] GPU_LAMBDA(scalar_t exp) -> scalar_t {
    return std::exp(fct * opmath_t{exp});
  });
#endif
}

namespace {

#if AT_USE_JITERATOR()
/* complex<Half> support impl */
constexpr char pow_name[] = "pow_kernel";
static const auto pow_kernel_string =
    jiterator_stringify(template <typename T> T pow_kernel(T base, T exp) {
      return std::pow(base, exp);
    });
#endif

/* complex<Half> support impl */
void pow_chalf_tensor_scalar_impl(TensorIteratorBase& iter, const Scalar& exp_scalar) {
  using scalar_t = c10::complex<at::Half>;
  using opmath_t = at::opmath_type<scalar_t>;
  auto exp = exp_scalar.to<opmath_t>();
#if AT_USE_JITERATOR()
  jitted_gpu_kernel<pow_name, scalar_t, scalar_t, 1>(
      iter,
      pow_kernel_string,
      /*scalar_pos=*/at::cuda::jit::BinaryFuncVariant::NoScalar,
      /*scalar_val=*/0,
      /*extra_args=*/std::make_tuple(exp));
#else
  gpu_kernel(iter, [=] GPU_LAMBDA(scalar_t base) -> scalar_t {
    return std::pow(opmath_t{base}, exp);
  });
#endif
}

}  // anonymous namespace

void pow_tensor_tensor_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (common_dtype == kComplexHalf) {
    using scalar_t = c10::complex<at::Half>;
    if (iter.is_cpu_scalar(1)) {
      const auto base = iter.scalar_value<scalar_t>(1);
      iter.remove_operand(1);
      pow_scalar_tensor_impl(iter, base);
    } else if (iter.is_cpu_scalar(2)) {
      const auto exp = iter.scalar_value<scalar_t>(2);
      iter.remove_operand(2);
      pow_chalf_tensor_scalar_impl(iter, exp);
    } else {
      using opmath_t = at::opmath_type<scalar_t>;
      TORCH_INTERNAL_ASSERT(!iter.is_cpu_scalar(1) && !iter.is_cpu_scalar(2));
#if AT_USE_JITERATOR()
      jitted_gpu_kernel<pow_name, scalar_t, scalar_t, 2>(
          iter, pow_kernel_string);
#else
      gpu_kernel(iter, [=] GPU_LAMBDA(scalar_t base, scalar_t exp) -> scalar_t {
            using opmath_t = at::opmath_type<scalar_t>;
            return pow_(opmath_t{base}, opmath_t{exp});
          });
#endif
    }
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
        kHalf, kBFloat16, iter.common_dtype(), "pow_cuda", [&] {
      if (iter.is_cpu_scalar(1)) {
        const auto base = iter.scalar_value<scalar_t>(1);
        iter.remove_operand(1);
        pow_scalar_tensor_impl(iter, base);
      } else if (iter.is_cpu_scalar(2)) {
        const auto exp = iter.scalar_value<scalar_t>(2);
        iter.remove_operand(2);
        pow_tensor_scalar_kernel(iter, exp);
      } else {
        gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t base, scalar_t exp) -> scalar_t {
          return pow_(base, exp);
        });
      }
    });
  }
}


template<typename Base_type, typename Exp_type>
void pow_tensor_scalar_kernel_impl(TensorIteratorBase& iter,
                                                 Exp_type exp) {
  const auto d_exp = static_cast<double>(exp);
  // .5 (sqrt), -.5 (rsqrt) and -1 (reciprocal) specializations are handled
  // in pow_tensor_scalar_kernel
  if (d_exp == 2) {
    gpu_kernel(iter, [=]GPU_LAMBDA(Base_type base) -> Base_type {
      return base * base;
    });
  } else if (d_exp == 3) {
    gpu_kernel(iter, [=]GPU_LAMBDA(Base_type base) -> Base_type {
      return base * base * base;
    });
  } else if (d_exp == -2) {
    gpu_kernel(iter, [=]GPU_LAMBDA(Base_type base) -> Base_type {
      return 1.0 / (base * base);
    });
  } else {
    gpu_kernel(iter, [=]GPU_LAMBDA(Base_type base) -> Base_type {
      return pow_(base, exp);
    });
  }
}

void pow_tensor_scalar_kernel(TensorIteratorBase& iter, const Scalar& exp_scalar) {
  // Dispatch to fast specialization for sqrt, rsqrt and reciprocal
  if (!exp_scalar.isComplex()) {
    if (exp_scalar.equal(.5)) {
      return sqrt_kernel_cuda(iter);
    } else if (exp_scalar.equal(-0.5)) {
      return rsqrt_kernel_cuda(iter);
    } else if (exp_scalar.equal(-1.0)) {
      return reciprocal_kernel_cuda(iter);
    }
  }
  if (isComplexType(iter.common_dtype()) || exp_scalar.isComplex()) {
    if (iter.common_dtype() == kComplexHalf) {
      using scalar_t = c10::complex<at::Half>;
      pow_chalf_tensor_scalar_impl(iter, exp_scalar);
      return;
    }
    AT_DISPATCH_COMPLEX_TYPES(iter.common_dtype(), "pow_cuda", [&]() {
      if (exp_scalar.equal(2.0)) {
        gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t base) -> scalar_t {
          return base * base;
        });
        return;
      }
      const auto exp = exp_scalar.to<scalar_t>();
      gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t base) -> scalar_t {
        return pow_(base, exp);
      });
    });
  } else if (isFloatingType(iter.common_dtype()) || exp_scalar.isIntegral(false)) {
    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(), "pow_cuda", [&]() {
      const auto exp = exp_scalar.to<scalar_t>();
      pow_tensor_scalar_kernel_impl<scalar_t>(iter, exp);
    });
  } else {
    TORCH_INTERNAL_ASSERT(false, "invalid combination of type in Pow function, common dtype:", iter.common_dtype(),
                                 "exp is integral?", exp_scalar.isIntegral(false));
  }
}

} // anonymous namespace

REGISTER_DISPATCH(pow_tensor_tensor_stub, &pow_tensor_tensor_kernel)
REGISTER_DISPATCH(pow_tensor_scalar_stub, &pow_tensor_scalar_kernel)

} // namespace at::native

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/cuda`.

## Detailed Analysis

### Code Structure

**Namespaces**: `REGISTER_DISPATCH`, `void`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Context.h`
- `ATen/Dispatch.h`
- `ATen/native/cuda/Loops.cuh`
- `ATen/native/cuda/JitLoops.cuh`
- `ATen/native/cuda/Pow.cuh`
- `ATen/native/DispatchStub.h`
- `ATen/native/TensorIterator.h`
- `ATen/native/Pow.h`
- `c10/core/Scalar.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`aten/src/ATen/native/cuda`):

- [`LogcumsumexpKernel.cu_docs.md`](./LogcumsumexpKernel.cu_docs.md)
- [`WeightNorm.cu_docs.md`](./WeightNorm.cu_docs.md)
- [`SparseBinaryOpIntersectionKernel.cu_docs.md`](./SparseBinaryOpIntersectionKernel.cu_docs.md)
- [`jit_utils.cpp_docs.md`](./jit_utils.cpp_docs.md)
- [`ReduceNormKernel.cu_docs.md`](./ReduceNormKernel.cu_docs.md)
- [`BinaryMiscOpsKernels.cu_docs.md`](./BinaryMiscOpsKernels.cu_docs.md)
- [`RowwiseScaledMM.h_docs.md`](./RowwiseScaledMM.h_docs.md)
- [`fused_adamw_amsgrad_impl.cuh_docs.md`](./fused_adamw_amsgrad_impl.cuh_docs.md)
- [`Col2Im.cu_docs.md`](./Col2Im.cu_docs.md)
- [`DistributionRandomKernel.cu_docs.md`](./DistributionRandomKernel.cu_docs.md)


## Cross-References

- **File Documentation**: `PowKernel.cu_docs.md`
- **Keyword Index**: `PowKernel.cu_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/cuda`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/aten/src/ATen/native/cuda`):

- [`DeviceSqrt.cuh_kw.md_docs.md`](./DeviceSqrt.cuh_kw.md_docs.md)
- [`UnaryGeometricAsinKernel.cu_kw.md_docs.md`](./UnaryGeometricAsinKernel.cu_kw.md_docs.md)
- [`Distributions.cpp_docs.md_docs.md`](./Distributions.cpp_docs.md_docs.md)
- [`fused_adamw_impl.cu_docs.md_docs.md`](./fused_adamw_impl.cu_docs.md_docs.md)
- [`TensorTopK.h_kw.md_docs.md`](./TensorTopK.h_kw.md_docs.md)
- [`ReduceOps.cpp_kw.md_docs.md`](./ReduceOps.cpp_kw.md_docs.md)
- [`FusedSgdKernel.cu_docs.md_docs.md`](./FusedSgdKernel.cu_docs.md_docs.md)
- [`Distributions.cu_kw.md_docs.md`](./Distributions.cu_kw.md_docs.md)
- [`block_reduce.cuh_docs.md_docs.md`](./block_reduce.cuh_docs.md_docs.md)
- [`fused_adagrad_impl.cuh_kw.md_docs.md`](./fused_adagrad_impl.cuh_kw.md_docs.md)


## Cross-References

- **File Documentation**: `PowKernel.cu_docs.md_docs.md`
- **Keyword Index**: `PowKernel.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
