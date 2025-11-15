# Documentation: `docs/aten/src/ATen/native/cuda/JitLoops.cuh_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/JitLoops.cuh_docs.md`
- **Size**: 9,725 bytes (9.50 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/JitLoops.cuh`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/JitLoops.cuh`
- **Size**: 6,907 bytes (6.75 KB)
- **Type**: CUDA Header File
- **Extension**: `.cuh`

## File Purpose

This is a cuda header file that is part of the PyTorch project.

## Original Source

```
#pragma once

#include <ATen/jit_macros.h>

#if AT_USE_JITERATOR()

#include <ATen/cuda/CUDAConfig.h>

#include <ATen/OpMathType.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/TensorIteratorDynamicCasting.h>

#include <ATen/native/cuda/MemoryAccess.cuh>

#include <ATen/native/cuda/CUDAJitLoops.cuh>

namespace at::native {

/* Note [Jiterator]
The "jiterator" simply just-in-time compiles the same kernels that
Loops.cuh (and CUDALoops.cuh) usually build. This reduces build time,
build size, and initial CUDA context size.

By default on non-Windows systems, it also caches compiled kernels in ~/.cache/torch/kernels.
This behavior is controlled with two environment variables:
  - USE_PYTORCH_KERNEL_CACHE, if set to zero then this will disable all cache use
  - PYTORCH_KERNEL_CACHE_PATH, if set specifies the folder to use for cached kernels

The jiterator currently has some limitations, however. It cannot:
  - handle math on complex datatypes
  - handle kernels with scalar parameters

These improvements will likely come soon.

For examples of how to use the jiterator see the i1 and gcd kernel
implementations, which pass jittable strings implementing their
operations instead of the typical CUDA functors.

To pass a runtime argument (similar to lambda captures in non-JIT kernels),
we need to pass to additional arguments to `jitted_gpu_kernel` by value.
Currently only primitive C++ types used for computation are valid.
The order of these extra arguments should be same as the order they appear
in kernel's function signature. (look at polygamma for example)

NOTE: One big restriction being that these arguments should be after the
arguments provided by TensorIterator. Eg. While capturing `n`, where
`scalar_t x` and `scalar_t y` are provided by TensorIterator,
* foo(scalar_t x, scalar_t y, int n) works!
* foo(int n, scalar_t x, scalar_y) doesn't work
* foo(scalar_t x, int n, scalar_y) doesn't work

*/

// Entrypoint for jitted GPU kernels.
// Only handles elementwise unary and binary kernels with a
//   common dtype and a single output.
// NOTE: this assumes the op's iterator has a common_dtype.
// NOTE: We use std::tuple instead of parameter pack
//  for `extra_args` due to following
// bug on older versions of clang
// https://bugs.llvm.org/show_bug.cgi?id=23029
template <
    char const* name,
    typename return_type,
    typename f_inputs_type,
    int arity,
    typename... Args>
void jitted_gpu_kernel(
    TensorIteratorBase& iter,
    const std::string& f,
    at::cuda::jit::BinaryFuncVariant scalar_pos =
        at::cuda::jit::BinaryFuncVariant::NoScalar,
    at::opmath_type<f_inputs_type> scalar_val = 0,
    std::tuple<Args...> extra_args = std::make_tuple()) {
  // TODO: much of preamble is common to both jitted_gpu_kernel and gpu_kernel
  //   Maybe it could be refactored?
  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(
      iter.device(arg).is_cuda(),
      "argument ", arg, ": expected a CUDA device but found ", iter.device(arg));
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      jitted_gpu_kernel<name, return_type, f_inputs_type, arity>(
          sub_iter, f, scalar_pos, scalar_val, extra_args);
    }

    return;
  }

  // Computes if dynamic casting is needed
  // Dynamic casting is needed if an input's dtype differs from the common dtype
  //   or if the result dtype differs from the output's dtype
  // Note: this is intentionally divergent from calling needs_dynamic_casting,
  //   which is more general and inspects a lambda to determine if dynamic
  //   casting is needed.
  bool needs_dynamic_casting = false;

  // Checks output
  const ScalarType return_scalar_type = c10::CppTypeToScalarType<return_type>::value;
  const auto dtype0 = iter.dtype(0);
  if (dtype0 != return_scalar_type) {
    needs_dynamic_casting = true;
  }

  // Checks input(s)
  const ScalarType inputs_scalar_type = c10::CppTypeToScalarType<f_inputs_type>::value;
  for (auto i = decltype(arity){1}; i < (arity + 1); ++i) {
    const auto dtypei = iter.dtype(i);
    if (dtypei != inputs_scalar_type) {
      needs_dynamic_casting = true;
      break;
    }
  }
  if (scalar_pos == at::cuda::jit::BinaryFuncVariant::NoScalar) {
    // NOTE: With `scalar_pos=NoScalar`,`scalar_val` is not used
    // for computation in the generated code and hence we pass a dummy
    // value of `0`.
    jitted_gpu_kernel_impl<
        /*name*/ name,
        /*return_type=*/return_type,
        /*f_inputs_type=*/f_inputs_type,
        arity,
        at::cuda::jit::BinaryFuncVariant::NoScalar>(
        iter, f, needs_dynamic_casting, /*scalar_val=*/scalar_val, extra_args);
  } else if (scalar_pos == at::cuda::jit::BinaryFuncVariant::RhsScalar) {
    jitted_gpu_kernel_impl<
        /*name*/ name,
        /*return_type=*/return_type,
        /*f_inputs_type=*/f_inputs_type,
        arity,
        at::cuda::jit::BinaryFuncVariant::RhsScalar>(
        iter,
        f,
        needs_dynamic_casting,
        scalar_val,
        extra_args);

  } else {
    jitted_gpu_kernel_impl<
        /*name*/ name,
        /*return_type=*/return_type,
        /*f_inputs_type=*/f_inputs_type,
        arity,
        at::cuda::jit::BinaryFuncVariant::LhsScalar>(
        iter,
        f,
        needs_dynamic_casting,
        scalar_val,
        extra_args);
  }
}

// TODO: support runtime state capture similar to `jitted_gpu_kernel`.
template <char const *name, typename return_type, typename f_inputs_type>
void opmath_jitted_gpu_kernel_with_scalars(TensorIteratorBase& iter, const std::string& f) {
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 3);
  //currently jiterator only handles binary functions where both inputs are of the same type (f_inputs_type)
  using opmath_t = at::opmath_type<f_inputs_type>;
  if (iter.is_cpu_scalar(1)) {
    auto scalar_val = iter.scalar_value<opmath_t>(1);
    iter.remove_operand(1);
    // TODO: When all kernels that use gpu_kernel_with_scalars are
    // ported to structured, this device guard can be deleted.  This
    // works around incorrect device guard generation for pre-structured
    // kernels device guards, but structured kernels do it right and
    // we can assume the device is already set correctly
    const OptionalDeviceGuard device_guard(iter.device(1));
    jitted_gpu_kernel<name, return_type, f_inputs_type, 1>(iter, f, at::cuda::jit::BinaryFuncVariant::LhsScalar, scalar_val);
  } else if (iter.is_cpu_scalar(2)) {
    auto scalar_val = iter.scalar_value<opmath_t>(2);
    iter.remove_operand(2);
    jitted_gpu_kernel<name, return_type, f_inputs_type, 1>(iter, f, at::cuda::jit::BinaryFuncVariant::RhsScalar, scalar_val);
  } else {
    jitted_gpu_kernel<name, return_type, f_inputs_type, 2>(iter, f);
  }
}

}  // namespace at::native

#endif // AT_USE_JITERATOR()

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/cuda`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/jit_macros.h`
- `ATen/cuda/CUDAConfig.h`
- `ATen/OpMathType.h`
- `ATen/TensorIterator.h`
- `ATen/native/TensorIteratorDynamicCasting.h`
- `ATen/native/cuda/MemoryAccess.cuh`
- `ATen/native/cuda/CUDAJitLoops.cuh`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

- **File Documentation**: `JitLoops.cuh_docs.md`
- **Keyword Index**: `JitLoops.cuh_kw.md`
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
- Implements or uses **caching** mechanisms.
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

- **File Documentation**: `JitLoops.cuh_docs.md_docs.md`
- **Keyword Index**: `JitLoops.cuh_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
