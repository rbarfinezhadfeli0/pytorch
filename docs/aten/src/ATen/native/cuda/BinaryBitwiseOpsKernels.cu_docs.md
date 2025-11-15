# Documentation: `aten/src/ATen/native/cuda/BinaryBitwiseOpsKernels.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/BinaryBitwiseOpsKernels.cu`
- **Size**: 2,210 bytes (2.16 KB)
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
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native {

template<typename scalar_t>
struct BitwiseAndFunctor {
  __device__ __forceinline__ scalar_t operator()(scalar_t a, scalar_t b) const {
    return a & b;
  }
};

template<>
struct BitwiseAndFunctor<bool> {
  __device__ __forceinline__ bool operator()(bool a, bool b) const {
    return a && b;
  }
};

void bitwise_and_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES_AND(kBool, iter.dtype(), "bitwise_and_cuda", [&]() {
    BitwiseAndFunctor<scalar_t> f;
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, f);
  });
}

template<typename scalar_t>
struct BitwiseOrFunctor {
  __device__ __forceinline__ scalar_t operator()(scalar_t a, scalar_t b) const {
    return a | b;
  }
};

template<>
struct BitwiseOrFunctor<bool> {
  __device__ __forceinline__ bool operator()(bool a, bool b) const {
    return a || b;
  }
};

void bitwise_or_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES_AND(kBool, iter.dtype(), "bitwise_or_cuda", [&]() {
    BitwiseOrFunctor<scalar_t> f;
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, f);
  });
}

template<typename scalar_t>
struct BitwiseXorFunctor {
  __device__ __forceinline__ scalar_t operator()(scalar_t a, scalar_t b) const {
    return a ^ b;
  }
};

template<>
struct BitwiseXorFunctor<bool> {
  __device__ __forceinline__ bool operator()(bool a, bool b) const {
    return a != b;
  }
};

void bitwise_xor_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES_AND(kBool, iter.dtype(), "bitwise_xor_cuda", [&]() {
    BitwiseXorFunctor<scalar_t> f;
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, f);
  });
}

REGISTER_DISPATCH(bitwise_and_stub, &bitwise_and_kernel_cuda)
REGISTER_DISPATCH(bitwise_or_stub, &bitwise_or_kernel_cuda)
REGISTER_DISPATCH(bitwise_xor_stub, &bitwise_xor_kernel_cuda)


} // namespace at::native

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/cuda`.

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `BitwiseAndFunctor`, `BitwiseAndFunctor`, `BitwiseOrFunctor`, `BitwiseOrFunctor`, `BitwiseXorFunctor`, `BitwiseXorFunctor`


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
- `ATen/native/TensorIterator.h`
- `ATen/native/BinaryOps.h`


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

- **File Documentation**: `BinaryBitwiseOpsKernels.cu_docs.md`
- **Keyword Index**: `BinaryBitwiseOpsKernels.cu_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
