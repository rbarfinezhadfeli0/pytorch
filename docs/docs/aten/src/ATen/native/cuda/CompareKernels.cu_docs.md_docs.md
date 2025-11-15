# Documentation: `docs/aten/src/ATen/native/cuda/CompareKernels.cu_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/CompareKernels.cu_docs.md`
- **Size**: 5,747 bytes (5.61 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/CompareKernels.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/CompareKernels.cu`
- **Size**: 3,021 bytes (2.95 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>


// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native { namespace {

enum class OpType {GE, GT, LE, LT};

template<typename scalar_t>
struct CompareFunctor{
  constexpr CompareFunctor(OpType op): op_(op) {};
  OpType op_;
  __device__ __forceinline__ bool operator() (scalar_t a, scalar_t b) const {
    if (op_ == OpType::GE) {
      return a >= b;
    } else if (op_ == OpType::GT) {
      return a > b;
    } else if (op_ == OpType::LE) {
      return a <= b;
    } else { //LT
      return a < b;
    }
  }
};

// Reflects the comparison operator, so reflect(op)(a, b) == op(b, a)
OpType reflect(OpType x) {
  switch (x) {
    case OpType::GE: return OpType::LE;
    case OpType::GT: return OpType::LT;
    case OpType::LE: return OpType::GE;
    case OpType::LT: return OpType::GT;
  }
  TORCH_INTERNAL_ASSERT(false, "Invalid OpType");
}

}  // namespace (anonymous)

template <typename scalar_t>
void compare_scalar_kernel(TensorIteratorBase &iter, OpType op, scalar_t rhs) {
  CompareFunctor<scalar_t> f(op);
  gpu_kernel(iter, [=] GPU_LAMBDA (scalar_t lhs) -> bool {
    return f(lhs, rhs);
  });
}

template <typename scalar_t>
void compare_kernel_impl(TensorIteratorBase &iter, OpType op) {
  // If either input is a cpu scalar, perform the equivalent comparison
  // where the scalar is on the right hand side. This saves us from
  // generating two otherwise identical kernels with mirrored
  // arguments.
  if (iter.is_cpu_scalar(1)) {
    const scalar_t lhs = iter.scalar_value<scalar_t>(1);
    iter.remove_operand(1);
    const DeviceGuard device_guard(iter.device(1));
    compare_scalar_kernel(iter, reflect(op), lhs);
  } else if (iter.is_cpu_scalar(2)) {
    const scalar_t rhs = iter.scalar_value<scalar_t>(2);
    iter.remove_operand(2);
    compare_scalar_kernel(iter, op, rhs);
  } else {
    CompareFunctor<scalar_t> f(op);
    gpu_kernel(iter, f);
  }
}

C10_NOINLINE void compare_kernel_with_scalars(TensorIteratorBase &iter, OpType op) {
  AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBFloat16, kBool, iter.common_dtype(), "compare_cuda", [&]() {
    compare_kernel_impl<scalar_t>(iter, op);
  });
}


void ge_kernel_cuda(TensorIteratorBase& iter) {
  compare_kernel_with_scalars(iter, OpType::GE);
}

void gt_kernel_cuda(TensorIteratorBase& iter) {
  compare_kernel_with_scalars(iter, OpType::GT);
}

void le_kernel_cuda(TensorIteratorBase& iter) {
  compare_kernel_with_scalars(iter, OpType::LE);
}

void lt_kernel_cuda(TensorIteratorBase& iter) {
  compare_kernel_with_scalars(iter, OpType::LT);
}

REGISTER_DISPATCH(ge_stub, &ge_kernel_cuda)
REGISTER_DISPATCH(gt_stub, &gt_kernel_cuda)
REGISTER_DISPATCH(le_stub, &le_kernel_cuda)
REGISTER_DISPATCH(lt_stub, &lt_kernel_cuda)

} // namespace at::native

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/cuda`.

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `OpType`, `CompareFunctor`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Dispatch.h`
- `ATen/native/BinaryOps.h`
- `ATen/native/DispatchStub.h`
- `ATen/native/TensorIterator.h`
- `ATen/native/cuda/Loops.cuh`


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

- **File Documentation**: `CompareKernels.cu_docs.md`
- **Keyword Index**: `CompareKernels.cu_kw.md`
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

- **File Documentation**: `CompareKernels.cu_docs.md_docs.md`
- **Keyword Index**: `CompareKernels.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
