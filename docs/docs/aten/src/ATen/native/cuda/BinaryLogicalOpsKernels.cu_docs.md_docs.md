# Documentation: `docs/aten/src/ATen/native/cuda/BinaryLogicalOpsKernels.cu_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/BinaryLogicalOpsKernels.cu_docs.md`
- **Size**: 7,017 bytes (6.85 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/BinaryLogicalOpsKernels.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/BinaryLogicalOpsKernels.cu`
- **Size**: 4,206 bytes (4.11 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native {

constexpr char logical_and_name[] = "logical_and_kernel";
void logical_and_kernel_cuda(TensorIterator& iter) {
  auto dtype = iter.common_dtype();
  if (at::isComplexType(dtype)) {
#if AT_USE_JITERATOR()
    static const auto logical_and_string = jiterator_stringify(
        template <typename T>
        bool logical_and_kernel(T a, T b) {
          return a && b;
        }
    ); // logical_and_string
    AT_DISPATCH_COMPLEX_TYPES(dtype, "logical_and_cuda", [&]() {
      jitted_gpu_kernel<
        /*name=*/ logical_and_name,
        /*return_dtype=*/ scalar_t,
        /*common_dtype=*/ scalar_t,
        /*arity=*/ 2>(iter, logical_and_string);
    }); // logical_and_string
#else
    AT_DISPATCH_COMPLEX_TYPES(dtype, "logical_and_cuda", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t, bool>(
          iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> bool {
        return a && b;
      });
    });
#endif
  } else {
    AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBool, ScalarType::BFloat16,
                               dtype, "logical_and_cuda", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t, bool>(
          iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> bool {
        return a && b;
      });
   });
  }
}

constexpr char logical_or_name[] = "logical_or_kernel";
void logical_or_kernel_cuda(TensorIterator& iter) {
  auto dtype = iter.common_dtype();
  if (at::isComplexType(dtype)) {
#if AT_USE_JITERATOR()
    static const auto logical_or_string = jiterator_stringify(
      template <typename T>
      bool logical_or_kernel(T a, T b) {
        return a || b;
      }
    ); // logical_or_string
    AT_DISPATCH_COMPLEX_TYPES(dtype, "logical_or_cuda", [&]() {
      jitted_gpu_kernel<
        /*name=*/ logical_or_name,
        /*return_dtype=*/ scalar_t,
        /*common_dtype=*/ scalar_t,
        /*arity=*/ 2>(iter, logical_or_string);
    });
#else
    AT_DISPATCH_COMPLEX_TYPES(dtype, "logical_or_cuda", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> bool {
        return a || b;
      });
    });
#endif
  } else {
  AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBool, ScalarType::BFloat16,
                             dtype, "logical_or_cuda", [&]() {
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t, bool>(
        iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> bool {
      return a || b;
    });
  });
  }
}

constexpr char logical_xor_name[] = "logical_xor_kernel";
void logical_xor_kernel_cuda(TensorIterator& iter) {
  auto dtype = iter.common_dtype();
  if (at::isComplexType(dtype)) {
#if AT_USE_JITERATOR()
    static const auto logical_xor_string = jiterator_stringify(
        template <typename T>
        bool logical_xor_kernel(T a, T b) {
          return bool(a) != bool(b);
        }
    );
    AT_DISPATCH_COMPLEX_TYPES(dtype, "logical_xor_cuda", [&]() {
      jitted_gpu_kernel<
        /*name=*/ logical_xor_name,
        /*return_dtype=*/ scalar_t,
        /*common_dtype=*/ scalar_t,
        /*arity=*/ 2>(iter, logical_xor_string);
    }); // logical_xor_string
#else
    AT_DISPATCH_COMPLEX_TYPES(dtype, "logical_xor_cuda", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> bool {
        return bool(a) != bool(b);
      });
    });
#endif
  } else {
  AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBool, ScalarType::BFloat16,
                             dtype, "logical_xor_cuda", [&]() {
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t, bool>(
        iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> bool {
      return bool(a) != bool(b);
    });
  });
  }
}

REGISTER_DISPATCH(logical_and_stub, &logical_and_kernel_cuda)
REGISTER_DISPATCH(logical_or_stub, &logical_or_kernel_cuda)
REGISTER_DISPATCH(logical_xor_stub, &logical_xor_kernel_cuda)


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

- `ATen/Dispatch.h`
- `ATen/native/DispatchStub.h`
- `ATen/native/cuda/Loops.cuh`
- `ATen/native/cuda/JitLoops.cuh`
- `ATen/native/TensorIterator.h`
- `ATen/native/BinaryOps.h`


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

- **File Documentation**: `BinaryLogicalOpsKernels.cu_docs.md`
- **Keyword Index**: `BinaryLogicalOpsKernels.cu_kw.md`
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

- **File Documentation**: `BinaryLogicalOpsKernels.cu_docs.md_docs.md`
- **Keyword Index**: `BinaryLogicalOpsKernels.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
