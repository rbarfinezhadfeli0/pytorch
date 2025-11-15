# Documentation: `docs/aten/src/ATen/native/cuda/Lerp.cu_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/Lerp.cu_docs.md`
- **Size**: 7,630 bytes (7.45 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/Lerp.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/Lerp.cu`
- **Size**: 4,895 bytes (4.78 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/Lerp.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/OpMathType.h>

namespace at::native {
namespace {

void lerp_scalar_kernel(
    at::TensorIteratorBase& iter,
    const c10::Scalar& weight);

constexpr char lerp_tensor_name[] = "lerp_tensor";
void lerp_tensor_kernel(at::TensorIteratorBase& iter) {
  auto dtype = iter.common_dtype();
  if(at::isComplexType(dtype)) {
#if AT_USE_JITERATOR()
  static const auto lerp_tensor_string = jiterator_stringify(
      template <typename T>
      T lerp_tensor(T self_val, T end_val, T weight_val) {
        return (std::abs(weight_val) < 0.5)
            ? self_val + weight_val * (end_val - self_val)
            : end_val -
                (end_val - self_val) * (static_cast<T>(1) - weight_val);
      }
  ); // lerp_tensor_string
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "lerp_cuda", [&] {
      if (iter.is_cpu_scalar(3)) {
        auto weight_val = iter.scalar_value<scalar_t>(3);
        iter.remove_operand(3);
        return lerp_scalar_kernel(iter, weight_val);
      }

      jitted_gpu_kernel<
        /*name=*/ lerp_tensor_name,
        /*return_dtype=*/ scalar_t,
        /*common_dtype=*/ scalar_t,
        /*arity=*/ 3>(iter, lerp_tensor_string);
    });
#else
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "lerp_cuda", [&] {
      using opmath_t = at::opmath_type<scalar_t>;
      if (iter.is_cpu_scalar(3)) {
        auto weight_val = iter.scalar_value<scalar_t>(3);
        iter.remove_operand(3);
        return lerp_scalar_kernel(iter, weight_val);
      }

      at::native::gpu_kernel(
        iter,
        [] GPU_LAMBDA(
            scalar_t self_val,
            scalar_t end_val,
            scalar_t weight_val) -> scalar_t {
           opmath_t self_val_f = self_val;
           opmath_t end_val_f = end_val;
           opmath_t weight_val_f = weight_val;
          return lerp(self_val, end_val, weight_val);
        });
      });
#endif
  } else {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      dtype, "lerp_cuda",
      [&] {
        if (iter.is_cpu_scalar(3)) {
          auto weight_val = iter.scalar_value<scalar_t>(3);
          iter.remove_operand(3);
          return lerp_scalar_kernel(iter, weight_val);
        }

        at::native::gpu_kernel(
          iter,
          [] GPU_LAMBDA(
              scalar_t self_val,
              scalar_t end_val,
              scalar_t weight_val) -> scalar_t {
            return lerp(self_val, end_val, weight_val);
          });
      });
  }
}

constexpr char lerp_scalar_name[] = "lerp_scalar";
void lerp_scalar_kernel(at::TensorIteratorBase& iter, const c10::Scalar& weight) {
  auto dtype = iter.common_dtype();
  if (at::isComplexType(dtype)) {
#if AT_USE_JITERATOR()
  static const auto lerp_scalar_string = jiterator_stringify(
      template <typename T>
      T lerp_scalar(T self_val, T end_val, T weight_val) {
        return (std::abs(weight_val) < 0.5)
            ? self_val + weight_val * (end_val - self_val)
            : end_val -
                (end_val - self_val) * (static_cast<T>(1) - weight_val);
      }
  ); // lerp_scalar_string
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "lerp_cuda", [&] {
      using opmath_t = at::opmath_type<scalar_t>;
      auto weight_val = weight.to<opmath_t>();
      jitted_gpu_kernel<
        /*name=*/ lerp_scalar_name,
        /*return_dtype=*/ scalar_t,
        /*common_dtype=*/ scalar_t,
        /*arity=*/ 2>(
        iter,
        lerp_scalar_string,
        /*scalar_pos=*/ at::cuda::jit::BinaryFuncVariant::NoScalar,
        /*scalar_val=*/ 0,
        /*extra_args=*/ std::make_tuple(weight_val));
  });
#else
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "lerp_cuda", [&] {
    using opmath_t = at::opmath_type<scalar_t>;
    auto weight_val = weight.to<opmath_t>();
    at::native::gpu_kernel(
        iter,
        [=] GPU_LAMBDA(scalar_t self_val, scalar_t end_val) {
          opmath_t self_val_f = self_val;
          opmath_t end_val_f = end_val;
          return lerp(self_val, end_val, weight_val);
        });
  });
#endif
  } else {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      dtype, "lerp_cuda",
      [&]{
        using opmath_t = at::opmath_type<scalar_t>;
        auto weight_val = weight.to<opmath_t>();
        at::native::gpu_kernel(
            iter, [=] GPU_LAMBDA(scalar_t self_val, scalar_t end_val) {
              return lerp(self_val, end_val, weight_val);
            });
      });
    }
}

} // anonymous namespace

REGISTER_DISPATCH(lerp_kernel_tensor_weight, &lerp_tensor_kernel)
REGISTER_DISPATCH(lerp_kernel_scalar_weight, &lerp_scalar_kernel)

} // namespace at::native

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/cuda`.

## Detailed Analysis

### Code Structure

**Namespaces**: `REGISTER_DISPATCH`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/Lerp.h`
- `ATen/Dispatch.h`
- `ATen/TensorIterator.h`
- `ATen/native/cuda/Loops.cuh`
- `ATen/native/cuda/JitLoops.cuh`
- `ATen/OpMathType.h`


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

- **File Documentation**: `Lerp.cu_docs.md`
- **Keyword Index**: `Lerp.cu_kw.md`
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

- **File Documentation**: `Lerp.cu_docs.md_docs.md`
- **Keyword Index**: `Lerp.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
