# Documentation: `aten/src/ATen/native/cuda/ForeachBinaryOpScalarList.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/ForeachBinaryOpScalarList.cu`
- **Size**: 8,023 bytes (7.83 KB)
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
#include <ATen/ops/_foreach_clamp_max_native.h>
#include <ATen/ops/_foreach_clamp_min_native.h>
#include <ATen/ops/_foreach_div_native.h>
#include <ATen/ops/_foreach_mul_native.h>
#include <ATen/ops/_foreach_pow_native.h>
#include <ATen/ops/_foreach_sub_native.h>

#include <ATen/ops/empty_like_native.h>
#endif

namespace at::native {

template <typename T, template <class> class Op>
std::vector<Tensor> foreach_binary_op(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors.size());
  for (const auto& t : tensors) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  tensor_lists.emplace_back(tensors.vec());
  tensor_lists.emplace_back(std::move(vec_res));

  using opmath_t = at::opmath_type<T>;
  multi_tensor_apply<2, opmath_t>(
      tensor_lists,
      scalars,
      BinaryOpScalarListFunctor<
          T,
          /* depth */ 2,
          /* r_args_depth */ 1,
          /* res_arg_index */ 1>(),

      Op<opmath_t>());
  return std::move(tensor_lists[1]);
}

template <typename T, template <class> class Op>
void foreach_binary_op_(TensorList tensors, at::ArrayRef<Scalar> scalars) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.emplace_back(tensors.vec());

  using opmath_t = at::opmath_type<T>;
  multi_tensor_apply<1, opmath_t>(
      tensor_lists,
      scalars,
      BinaryOpScalarListFunctor<
          T,
          /* depth */ 1,
          /* r_args_depth */ 1,
          /* res_arg_index */ 0>(),
      Op<opmath_t>());
  increment_version(tensors);
}

template <template <class> class Op>
std::vector<Tensor> all_types_complex_bool_half_bfloat16(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalarlist_cuda",
      [&]() { return foreach_binary_op<scalar_t, Op>(tensors, scalars); });
}

template <template <class> class Op>
void all_types_complex_bool_half_bfloat16_(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalarlist_cuda_",
      [&]() { foreach_binary_op_<scalar_t, Op>(tensors, scalars); });
}

template <template <class> class Op>
std::vector<Tensor> all_types_half_bfloat16(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  return AT_DISPATCH_ALL_TYPES_AND2(
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalarlist_cuda",
      [&]() { return foreach_binary_op<scalar_t, Op>(tensors, scalars); });
}

template <template <class> class Op>
void all_types_half_bfloat16_(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  AT_DISPATCH_ALL_TYPES_AND2(
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalarlist_cuda_",
      [&]() { foreach_binary_op_<scalar_t, Op>(tensors, scalars); });
}

template <template <class> class Op>
std::vector<Tensor> all_types_complex_half_bfloat16(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalarlist_cuda",
      [&]() { return foreach_binary_op<scalar_t, Op>(tensors, scalars); });
}

template <template <class> class Op>
void all_types_complex_half_bfloat16_(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalarlist_cuda_",
      [&]() { foreach_binary_op_<scalar_t, Op>(tensors, scalars); });
}

#define FOREACH_BINARY_OP_SCALARLIST(FUNCTION, NAME, OP, DIV_OP)          \
  void foreach_tensor_##NAME##_scalarlist_kernel_cuda_(                   \
      TensorList tensors, at::ArrayRef<Scalar> scalars) {                 \
    check_foreach_api_restrictions(tensors, scalars);                     \
    if (!can_use_fast_route(tensors, scalars, DIV_OP)) {                  \
      return at::native::foreach_tensor_##NAME##_scalarlist_kernel_slow_( \
          tensors, scalars);                                              \
    }                                                                     \
                                                                          \
    FUNCTION##_<OP>(tensors, scalars);                                    \
  }                                                                       \
                                                                          \
  std::vector<Tensor> foreach_tensor_##NAME##_scalarlist_kernel_cuda(     \
      TensorList tensors, at::ArrayRef<Scalar> scalars) {                 \
    check_foreach_api_restrictions(tensors, scalars);                     \
    if (!can_use_fast_route(tensors, scalars, DIV_OP)) {                  \
      return at::native::foreach_tensor_##NAME##_scalarlist_kernel_slow(  \
          tensors, scalars);                                              \
    }                                                                     \
                                                                          \
    return FUNCTION<OP>(tensors, scalars);                                \
  }

FOREACH_BINARY_OP_SCALARLIST(
    all_types_complex_bool_half_bfloat16,
    add,
    std::plus,
    /*div_op*/ false);
FOREACH_BINARY_OP_SCALARLIST(
    all_types_complex_bool_half_bfloat16,
    mul,
    std::multiplies,
    /*div_op*/ false);
FOREACH_BINARY_OP_SCALARLIST(
    all_types_complex_bool_half_bfloat16,
    div,
    std::divides,
    /*div_op*/ true);
// See [Why is foreach_pow's division_op=true?]
FOREACH_BINARY_OP_SCALARLIST(
    all_types_complex_half_bfloat16,
    pow,
    power_functor,
    /*div_op*/ true);

// This does not use FOREACH_BINARY_OP_SCALARLIST because
// In the case of subtraction, we dont allow scalar to be boolean following the
// torch.sub logic
void foreach_tensor_sub_scalarlist_kernel_cuda_(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  check_foreach_api_restrictions(tensors, scalars);
  for (const auto i : c10::irange(tensors.size())) {
    sub_check(tensors[i], scalars[i]);
  }

  if (!can_use_fast_route({tensors}, scalars)) {
    return at::native::foreach_tensor_sub_scalarlist_kernel_slow_(
        tensors, scalars);
  }

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalarlist_cuda_",
      [&]() { foreach_binary_op_<scalar_t, std::minus>(tensors, scalars); });
}

std::vector<Tensor> foreach_tensor_sub_scalarlist_kernel_cuda(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  check_foreach_api_restrictions(tensors, scalars);
  for (const auto i : c10::irange(tensors.size())) {
    sub_check(tensors[i], scalars[i]);
  }

  if (!can_use_fast_route({tensors}, scalars)) {
    return at::native::foreach_tensor_sub_scalarlist_kernel_slow(
        tensors, scalars);
  }

  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalarlist_cuda_",
      [&]() {
        return foreach_binary_op<scalar_t, std::minus>(tensors, scalars);
      });
}

FOREACH_BINARY_OP_SCALARLIST(all_types_half_bfloat16, clamp_max, minimum, true);
FOREACH_BINARY_OP_SCALARLIST(all_types_half_bfloat16, clamp_min, maximum, true);

} // namespace at::native

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/cuda`.

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `Op`, `Op`, `Op`, `Op`, `Op`, `Op`, `Op`, `Op`


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
- `ATen/ops/_foreach_clamp_max_native.h`
- `ATen/ops/_foreach_clamp_min_native.h`
- `ATen/ops/_foreach_div_native.h`
- `ATen/ops/_foreach_mul_native.h`
- `ATen/ops/_foreach_pow_native.h`
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

- **File Documentation**: `ForeachBinaryOpScalarList.cu_docs.md`
- **Keyword Index**: `ForeachBinaryOpScalarList.cu_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
