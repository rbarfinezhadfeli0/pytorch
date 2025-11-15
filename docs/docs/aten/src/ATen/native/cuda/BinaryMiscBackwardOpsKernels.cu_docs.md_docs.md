# Documentation: `docs/aten/src/ATen/native/cuda/BinaryMiscBackwardOpsKernels.cu_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/BinaryMiscBackwardOpsKernels.cu_docs.md`
- **Size**: 7,751 bytes (7.57 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/BinaryMiscBackwardOpsKernels.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/BinaryMiscBackwardOpsKernels.cu`
- **Size**: 4,883 bytes (4.77 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/BinaryOps.h>

#include <limits>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/JitLoops.cuh>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native {

constexpr char sigmoid_backward_name[] = "sigmoid_backward";
void sigmoid_backward_kernel_cuda(TensorIteratorBase& iter) {
  auto dtype = iter.dtype();
  if(isComplexType(dtype)) {
#if AT_USE_JITERATOR()
    static const auto sigmoid_backward_string = jiterator_stringify(
        template <typename T>
        T sigmoid_backward(T a, T b) {
          return a * std::conj((T{1.} - b) * b);
        }
    ); // sigmoid_backward_string
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "sigmoid_backward_cuda", [&]() {
        jitted_gpu_kernel<
          /*name=*/ sigmoid_backward_name,
          /*return_dtype=*/ scalar_t,
          /*common_dtype=*/ scalar_t,
          /*arity=*/ 2>(iter, sigmoid_backward_string);
    });
#else
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "sigmoid_backward_cuda", [&]() {
      gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        using comp_t = at::opmath_type<scalar_t>;
        const auto one = comp_t{1.};
        const auto comp_b = static_cast<comp_t>(b);
        const auto comp_a = static_cast<comp_t>(a);
        return static_cast<scalar_t>(comp_a * std::conj((one - comp_b) * comp_b));
      });
    });
#endif
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, dtype, "sigmoid_backward_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return a * (scalar_t(1.) - b) * b;
      });
    });
  }
}

void logit_backward_kernel_cuda(TensorIteratorBase& iter, const Scalar& eps_scalar) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "logit_cuda",
      [&]() {
        using T_ACC = acc_type<scalar_t, true>;
        const T_ACC eps = eps_scalar.to<T_ACC>();
        if (eps < T_ACC(0)) {
          gpu_kernel(
              iter, [] GPU_LAMBDA(scalar_t dy, scalar_t x) -> scalar_t {
                const T_ACC dy_acc = static_cast<T_ACC>(dy);
                const T_ACC x_acc = static_cast<T_ACC>(x);
                return (x_acc < T_ACC(0) || x_acc > T_ACC(1))
                    ? std::numeric_limits<T_ACC>::quiet_NaN()
                    : dy_acc / (x_acc * (T_ACC(1) - x_acc));
              });
        } else {
          const T_ACC lo = eps;
          const T_ACC hi = T_ACC(1) - eps;
          gpu_kernel(
              iter, [lo, hi] GPU_LAMBDA(scalar_t dy, scalar_t x) -> scalar_t {
                const T_ACC dy_acc = static_cast<T_ACC>(dy);
                const T_ACC x_acc = static_cast<T_ACC>(x);
                return (x_acc < lo || x_acc > hi)
                    ? T_ACC(0)
                    : dy_acc / (x_acc * (T_ACC(1) - x_acc));
              });
        }
      });
}

constexpr char tanh_backward_name[] = "tanh_backward";
void tanh_backward_kernel_cuda(TensorIteratorBase& iter) {
  auto dtype = iter.dtype();
  if(isComplexType(dtype)) {
#if AT_USE_JITERATOR()
    static const auto tanh_backward_string = jiterator_stringify(
      template <typename T>
      T tanh_backward(T a, T b) {
        return a * std::conj(T{1.} - b * b);
      }
    ); // tanh_backward_string
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "tanh_backward_complex_cuda", [&]() {
      jitted_gpu_kernel<
          /*name=*/ tanh_backward_name,
          /*return_dtype=*/ scalar_t,
          /*common_dtype=*/ scalar_t,
          /*arity=*/ 2>(iter, tanh_backward_string);
    });
#else
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "tanh_backward_complex_cuda", [&]() {
      gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        using comp_t = at::opmath_type<scalar_t>;
        const auto one = comp_t{1.};
        const auto comp_b = static_cast<comp_t>(b);
        const auto comp_a = static_cast<comp_t>(a);
        return static_cast<scalar_t>(comp_a * std::conj(one - comp_b * comp_b));
      });
    });
#endif
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, dtype, "tanh_backward_cuda", [&]() {
      gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return a * (scalar_t{1.} - b * b);
      });
    });
  }
}

REGISTER_DISPATCH(sigmoid_backward_stub, &sigmoid_backward_kernel_cuda)
REGISTER_DISPATCH(logit_backward_stub, &logit_backward_kernel_cuda)
REGISTER_DISPATCH(tanh_backward_stub, &tanh_backward_kernel_cuda)

} // namespace at::native

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/cuda`.

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/BinaryOps.h`
- `limits`
- `ATen/AccumulateType.h`
- `ATen/Dispatch.h`
- `ATen/native/DispatchStub.h`
- `ATen/native/TensorIterator.h`
- `ATen/native/cuda/Loops.cuh`
- `ATen/native/cuda/JitLoops.cuh`


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

- **File Documentation**: `BinaryMiscBackwardOpsKernels.cu_docs.md`
- **Keyword Index**: `BinaryMiscBackwardOpsKernels.cu_kw.md`
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

- **File Documentation**: `BinaryMiscBackwardOpsKernels.cu_docs.md_docs.md`
- **Keyword Index**: `BinaryMiscBackwardOpsKernels.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
