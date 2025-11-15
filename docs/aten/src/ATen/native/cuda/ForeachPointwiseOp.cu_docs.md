# Documentation: `aten/src/ATen/native/cuda/ForeachPointwiseOp.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/ForeachPointwiseOp.cu`
- **Size**: 12,129 bytes (11.84 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_foreach_add_native.h>
#include <ATen/ops/_foreach_addcdiv_native.h>
#include <ATen/ops/_foreach_addcmul_native.h>
#include <ATen/ops/_foreach_div_native.h>
#include <ATen/ops/_foreach_maximum_native.h>
#include <ATen/ops/_foreach_minimum_native.h>
#include <ATen/ops/_foreach_mul_native.h>
#include <ATen/ops/_foreach_sub_native.h>

#include <ATen/ops/empty_like_native.h>
#endif

namespace at::native {

template <template <class> class Op>
std::vector<Tensor> foreach_pointwise_op(
    TensorList input,
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& scalar) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(input.size());
  for (const auto& t : input) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  tensor_lists.emplace_back(input.vec());
  tensor_lists.emplace_back(tensors1.vec());
  tensor_lists.emplace_back(tensors2.vec());
  tensor_lists.emplace_back(std::move(vec_res));

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf,
      kBFloat16,
      input[0].scalar_type(),
      "foreach_pointwise_op_cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        multi_tensor_apply<4>(
            tensor_lists,
            PointwiseOpScalarFunctor<
                scalar_t,
                /* depth */ 4,
                /* r_args_depth */ 3,
                /* res_arg_index */ 3>(),
            Op<opmath_t>(),
            scalar.to<opmath_t>());
      });

  return std::move(tensor_lists[3]);
}

template <template <class> class Op>
void foreach_pointwise_op_(
    TensorList input,
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& scalar) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.emplace_back(input.vec());
  tensor_lists.emplace_back(tensors1.vec());
  tensor_lists.emplace_back(tensors2.vec());

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf,
      kBFloat16,
      input[0].scalar_type(),
      "foreach_pointwise_op__cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        multi_tensor_apply<3>(
            tensor_lists,
            PointwiseOpScalarFunctor<
                scalar_t,
                /* depth */ 3,
                /* r_args_depth */ 3,
                /* res_arg_index */ 0>(),
            Op<opmath_t>(),
            scalar.to<opmath_t>());
      });
  increment_version(input);
}

template <template <class> class Op>
void foreach_pointwise_op_(
    TensorList input,
    TensorList tensors1,
    TensorList tensors2,
    at::ArrayRef<Scalar> scalars) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.reserve(3);
  tensor_lists.emplace_back(input.vec());
  tensor_lists.emplace_back(tensors1.vec());
  tensor_lists.emplace_back(tensors2.vec());

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf,
      kBFloat16,
      input[0].scalar_type(),
      "foreach_pointwise_op__cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        multi_tensor_apply<3, opmath_t>(
            tensor_lists,
            scalars,
            PointwiseOpScalarListFunctor<
                scalar_t,
                /* depth */ 3,
                /* r_args_depth */ 3,
                /* res_arg_index */ 0>(),
            Op<opmath_t>());
      });
  increment_version(input);
}

template <template <class> class Op>
std::vector<Tensor> foreach_pointwise_op(
    TensorList input,
    TensorList tensors1,
    TensorList tensors2,
    at::ArrayRef<Scalar> scalars) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.reserve(4);
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(input.size());
  for (const auto& t : input) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  tensor_lists.emplace_back(input.vec());
  tensor_lists.emplace_back(tensors1.vec());
  tensor_lists.emplace_back(tensors2.vec());
  tensor_lists.emplace_back(std::move(vec_res));

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf,
      kBFloat16,
      input[0].scalar_type(),
      "foreach_pointwise_op_cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        multi_tensor_apply<4, opmath_t>(
            tensor_lists,
            scalars,
            PointwiseOpScalarListFunctor<
                scalar_t,
                /* depth */ 4,
                /* r_args_depth */ 3,
                /* res_arg_index */ 3>(),
            Op<opmath_t>());
      });

  return std::move(tensor_lists[3]);
}

#define FOREACH_POINTWISE_OP_SCALAR(NAME, OP)                           \
  std::vector<Tensor> foreach_tensor_##NAME##_scalar_cuda(              \
      TensorList input,                                                 \
      TensorList tensors1,                                              \
      TensorList tensors2,                                              \
      const Scalar& scalar) {                                           \
    check_foreach_api_restrictions(input, tensors1, tensors2);          \
                                                                        \
    if (!can_use_fast_route({input, tensors1, tensors2}, scalar) ||     \
        has_integral_tensor(input, /* includeBool */ true)) {           \
      return at::native::foreach_tensor_##NAME##_scalar_slow(           \
          input, tensors1, tensors2, scalar);                           \
    }                                                                   \
                                                                        \
    return foreach_pointwise_op<OP>(input, tensors1, tensors2, scalar); \
  }                                                                     \
                                                                        \
  void foreach_tensor_##NAME##_scalar_cuda_(                            \
      TensorList input,                                                 \
      TensorList tensors1,                                              \
      TensorList tensors2,                                              \
      const Scalar& scalar) {                                           \
    check_foreach_api_restrictions(input, tensors1, tensors2);          \
                                                                        \
    if (!can_use_fast_route({input, tensors1, tensors2}, scalar) ||     \
        has_integral_tensor(input, /* includeBool */ true)) {           \
      return at::native::foreach_tensor_##NAME##_scalar_slow_(          \
          input, tensors1, tensors2, scalar);                           \
    }                                                                   \
                                                                        \
    foreach_pointwise_op_<OP>(input, tensors1, tensors2, scalar);       \
  }

#define FOREACH_POINTWISE_OP_SCALARLIST(NAME, OP)                        \
  std::vector<Tensor> foreach_tensor_##NAME##_scalarlist_cuda(           \
      TensorList input,                                                  \
      TensorList tensors1,                                               \
      TensorList tensors2,                                               \
      at::ArrayRef<Scalar> scalars) {                                    \
    check_foreach_api_restrictions(input, tensors1, tensors2, scalars);  \
                                                                         \
    if (!can_use_fast_route({input, tensors1, tensors2}, scalars) ||     \
        has_integral_tensor(input, /* includeBool */ true)) {            \
      return at::native::foreach_tensor_##NAME##_scalarlist_slow(        \
          input, tensors1, tensors2, scalars);                           \
    }                                                                    \
                                                                         \
    return foreach_pointwise_op<OP>(input, tensors1, tensors2, scalars); \
  }                                                                      \
                                                                         \
  void foreach_tensor_##NAME##_scalarlist_cuda_(                         \
      TensorList input,                                                  \
      TensorList tensors1,                                               \
      TensorList tensors2,                                               \
      at::ArrayRef<Scalar> scalars) {                                    \
    check_foreach_api_restrictions(input, tensors1, tensors2, scalars);  \
                                                                         \
    if (!can_use_fast_route({input, tensors1, tensors2}, scalars) ||     \
        has_integral_tensor(input, /* includeBool */ true)) {            \
      return at::native::foreach_tensor_##NAME##_scalarlist_slow_(       \
          input, tensors1, tensors2, scalars);                           \
    }                                                                    \
                                                                         \
    foreach_pointwise_op_<OP>(input, tensors1, tensors2, scalars);       \
  }

#define FOREACH_POINTWISE_OP_TENSOR(NAME, OP)                             \
  std::vector<Tensor> foreach_tensor_##NAME##_tensor_cuda(                \
      TensorList input,                                                   \
      TensorList tensors1,                                                \
      TensorList tensors2,                                                \
      const Tensor& scalars_) {                                           \
    auto scalars = convert_tensor_to_scalar_list(scalars_, input.size()); \
    check_foreach_api_restrictions(input, tensors1, tensors2, scalars);   \
    if (!can_use_fast_route({input, tensors1, tensors2}) ||               \
        has_integral_tensor(input, /* includeBool */ true)) {             \
      return at::native::foreach_tensor_##NAME##_scalarlist_slow(         \
          input, tensors1, tensors2, scalars);                            \
    }                                                                     \
                                                                          \
    return foreach_pointwise_op<OP>(input, tensors1, tensors2, scalars);  \
  }                                                                       \
                                                                          \
  void foreach_tensor_##NAME##_tensor_cuda_(                              \
      TensorList input,                                                   \
      TensorList tensors1,                                                \
      TensorList tensors2,                                                \
      const Tensor& scalars_) {                                           \
    auto scalars = convert_tensor_to_scalar_list(scalars_, input.size()); \
    check_foreach_api_restrictions(input, tensors1, tensors2, scalars);   \
    if (!can_use_fast_route({input, tensors1, tensors2}, scalars) ||      \
        has_integral_tensor(input, /* includeBool */ true)) {             \
      return at::native::foreach_tensor_##NAME##_scalarlist_slow_(        \
          input, tensors1, tensors2, scalars);                            \
    }                                                                     \
                                                                          \
    foreach_pointwise_op_<OP>(input, tensors1, tensors2, scalars);        \
  }

FOREACH_POINTWISE_OP_SCALAR(addcmul, std::multiplies);
FOREACH_POINTWISE_OP_SCALAR(addcdiv, std::divides);
FOREACH_POINTWISE_OP_SCALARLIST(addcmul, std::multiplies);
FOREACH_POINTWISE_OP_SCALARLIST(addcdiv, std::divides);
FOREACH_POINTWISE_OP_TENSOR(addcdiv, std::divides);
FOREACH_POINTWISE_OP_TENSOR(addcmul, std::multiplies);

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
- `ATen/NumericUtils.h`
- `ATen/native/ForeachUtils.h`
- `ATen/native/cuda/ForeachFunctors.cuh`
- `ATen/NativeFunctions.h`
- `ATen/ops/_foreach_add_native.h`
- `ATen/ops/_foreach_addcdiv_native.h`
- `ATen/ops/_foreach_addcmul_native.h`
- `ATen/ops/_foreach_div_native.h`
- `ATen/ops/_foreach_maximum_native.h`
- `ATen/ops/_foreach_minimum_native.h`
- `ATen/ops/_foreach_mul_native.h`
- `ATen/ops/_foreach_sub_native.h`
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

- **File Documentation**: `ForeachPointwiseOp.cu_docs.md`
- **Keyword Index**: `ForeachPointwiseOp.cu_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
