# Documentation: `docs/aten/src/ATen/native/cuda/ForeachBinaryOpScalarTensor.cu_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/ForeachBinaryOpScalarTensor.cu_docs.md`
- **Size**: 11,140 bytes (10.88 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/ForeachBinaryOpScalarTensor.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/ForeachBinaryOpScalarTensor.cu`
- **Size**: 8,177 bytes (7.99 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/cuda/ForeachMinMaxFunctors.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_foreach_add_native.h>
#include <ATen/ops/_foreach_div_native.h>
#include <ATen/ops/_foreach_mul_native.h>

#include <ATen/ops/empty_like_native.h>
#endif

namespace at::native {

template <typename T, template <class> class Op>
std::vector<Tensor> foreach_binary_op(
    TensorList tensors,
    const Tensor& scalar,
    const Scalar& alpha = 1) {
  TORCH_CHECK(
      scalar.dim() == 0 && scalar.numel() == 1,
      "scalar tensor expected to be 0 dim but it has ",
      scalar.dim(),
      " dimensions and ",
      scalar.numel(),
      " elements.");
  TORCH_CHECK(
      tensors[0].device() == scalar.device(),
      "scalar tensor expected to be on ",
      tensors[0].device(),
      " but is on ",
      scalar.device());
  std::vector<std::vector<at::Tensor>> tensor_lists;
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors.size());
  for (const auto& t : tensors) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  tensor_lists.emplace_back(tensors.vec());
  tensor_lists.emplace_back(std::move(vec_res));

  using opmath_t = at::opmath_type<T>;
  multi_tensor_apply<2>(
      tensor_lists,
      BinaryOpScalarTensorFunctor<
          T,
          /* depth */ 2,
          /* r_args_depth */ 1,
          /* res_arg_index */ 1>(),
      Op<opmath_t>(),
      scalar.data_ptr<T>(),
      alpha.to<opmath_t>());
  return std::move(tensor_lists[1]);
}

template <typename T, template <class> class Op>
void foreach_binary_op_(
    TensorList tensors,
    const Tensor& scalar,
    const Scalar& alpha = 1) {
  TORCH_CHECK(
      scalar.dim() == 0 && scalar.numel() == 1,
      "scalar tensor expected to be 0 dim but has ",
      scalar.dim(),
      " dimensions and ",
      scalar.numel(),
      " elements.");
  TORCH_CHECK(
      tensors[0].device() == scalar.device(),
      "scalar tensor is expected to be on ",
      tensors[0].device(),
      " but is on ",
      scalar.device());
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.emplace_back(tensors.vec());

  using opmath_t = at::opmath_type<T>;
  multi_tensor_apply<1>(
      tensor_lists,
      BinaryOpScalarTensorFunctor<
          T,
          /* depth */ 1,
          /* r_args_depth */ 1,
          /* res_arg_index */ 0>(),
      Op<opmath_t>(),
      scalar.data_ptr<T>(),
      alpha.to<opmath_t>());
  increment_version(tensors);
}

// TODO(crcrpar): Nest dispatch by looking up `scalar.scalar_type` for better
// coverage?
#define FOREACH_BINARY_OP_SCALAR_TENSOR(FUNCTION, NAME, OP, DIVISION_OP) \
  void foreach_tensor_##NAME##_tensor_kernel_cuda_(                      \
      TensorList tensors, const Tensor& scalar) {                        \
    if (scalar.device().type() == DeviceType::CPU) {                     \
      return at::native::foreach_tensor_##NAME##_scalar_kernel_cuda_(    \
          tensors, scalar.item());                                       \
    }                                                                    \
    check_foreach_api_restrictions(tensors);                             \
    if (!(can_use_fast_route(                                            \
              ArrayRef<TensorList>{tensors}, {}, DIVISION_OP) &&         \
          tensors[0].scalar_type() == scalar.scalar_type())) {           \
      return at::native::foreach_tensor_##NAME##_tensor_kernel_slow_(    \
          tensors, scalar);                                              \
    }                                                                    \
                                                                         \
    FUNCTION##_<OP>(tensors, scalar);                                    \
  }                                                                      \
                                                                         \
  std::vector<Tensor> foreach_tensor_##NAME##_tensor_kernel_cuda(        \
      TensorList tensors, const Tensor& scalar) {                        \
    if (scalar.device().type() == DeviceType::CPU) {                     \
      return at::native::foreach_tensor_##NAME##_scalar_kernel_cuda(     \
          tensors, scalar.item());                                       \
    }                                                                    \
    check_foreach_api_restrictions(tensors);                             \
    if (!(can_use_fast_route(                                            \
              ArrayRef<TensorList>{tensors}, {}, DIVISION_OP) &&         \
          tensors[0].scalar_type() == scalar.scalar_type())) {           \
      return at::native::foreach_tensor_##NAME##_tensor_kernel_slow(     \
          tensors, scalar);                                              \
    }                                                                    \
                                                                         \
    return FUNCTION<OP>(tensors, scalar);                                \
  }

#define FOREACH_BINARY_OP_SCALAR_TENSOR_ALPHA(FUNCTION, NAME, OP)      \
  void foreach_tensor_##NAME##_tensor_kernel_cuda_(                    \
      TensorList tensors, const Tensor& scalar, const Scalar& alpha) { \
    check_foreach_api_restrictions(tensors);                           \
    if (!(can_use_fast_route(ArrayRef<TensorList>{tensors}, alpha) &&  \
          tensors[0].scalar_type() == scalar.scalar_type())) {         \
      return at::native::foreach_tensor_##NAME##_tensor_kernel_slow_(  \
          tensors, scalar, alpha);                                     \
    }                                                                  \
                                                                       \
    FUNCTION##_<OP>(tensors, scalar, alpha);                           \
  }                                                                    \
                                                                       \
  std::vector<Tensor> foreach_tensor_##NAME##_tensor_kernel_cuda(      \
      TensorList tensors, const Tensor& scalar, const Scalar& alpha) { \
    check_foreach_api_restrictions(tensors);                           \
    if (!(can_use_fast_route(ArrayRef<TensorList>{tensors}, alpha) &&  \
          tensors[0].scalar_type() == scalar.scalar_type())) {         \
      return at::native::foreach_tensor_##NAME##_tensor_kernel_slow(   \
          tensors, scalar, alpha);                                     \
    }                                                                  \
                                                                       \
    return FUNCTION<OP>(tensors, scalar, alpha);                       \
  }

template <template <class> class Op>
std::vector<Tensor> all_types_complex_bool_half_bfloat16(
    TensorList tensors,
    const Tensor& scalar,
    const Scalar& alpha = 1) {
  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda",
      [&]() {
        return foreach_binary_op<scalar_t, Op>(tensors, scalar, alpha);
      });
}

template <template <class> class Op>
void all_types_complex_bool_half_bfloat16_(
    TensorList tensors,
    const Tensor& scalar,
    const Scalar& alpha = 1) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_cuda_",
      [&]() { foreach_binary_op_<scalar_t, Op>(tensors, scalar, alpha); });
}

FOREACH_BINARY_OP_SCALAR_TENSOR_ALPHA(
    all_types_complex_bool_half_bfloat16,
    add,
    std::plus);

FOREACH_BINARY_OP_SCALAR_TENSOR(
    all_types_complex_bool_half_bfloat16,
    mul,
    std::multiplies,
    /* div_op */ false);

FOREACH_BINARY_OP_SCALAR_TENSOR(
    all_types_complex_bool_half_bfloat16,
    div,
    std::divides,
    /* div_op */ true);

} // namespace at::native

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/cuda`.

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `Op`, `Op`, `Op`, `Op`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Dispatch.h`
- `ATen/native/BinaryOps.h`
- `ATen/native/ForeachUtils.h`
- `ATen/native/cuda/ForeachFunctors.cuh`
- `ATen/native/cuda/ForeachMinMaxFunctors.cuh`
- `ATen/NativeFunctions.h`
- `ATen/ops/_foreach_add_native.h`
- `ATen/ops/_foreach_div_native.h`
- `ATen/ops/_foreach_mul_native.h`
- `ATen/ops/empty_like_native.h`


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

- **File Documentation**: `ForeachBinaryOpScalarTensor.cu_docs.md`
- **Keyword Index**: `ForeachBinaryOpScalarTensor.cu_kw.md`
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

- **File Documentation**: `ForeachBinaryOpScalarTensor.cu_docs.md_docs.md`
- **Keyword Index**: `ForeachBinaryOpScalarTensor.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
