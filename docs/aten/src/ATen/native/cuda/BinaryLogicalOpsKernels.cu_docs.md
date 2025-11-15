# Documentation: BinaryLogicalOpsKernels.cu

## File Metadata
- **Path**: `aten/src/ATen/native/cuda/BinaryLogicalOpsKernels.cu`
- **Size**: 4206 bytes
- **Lines**: 128
- **Extension**: .cu
- **Type**: Regular file

## Original Source

```cu
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

This file is part of the PyTorch repository. It is a C++/CUDA source/header file that may contain implementations, declarations, or kernel code.

## Detailed Walkthrough


## Key Components

The file contains 362 words across 128 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 4206 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
