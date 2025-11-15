# Keyword Index: `aten/src/ATen/native/cuda/ForeachUnaryOp.cu`

## File Information

- **Original File**: [aten/src/ATen/native/cuda/ForeachUnaryOp.cu](../../../../../../aten/src/ATen/native/cuda/ForeachUnaryOp.cu)
- **Documentation**: [`ForeachUnaryOp.cu_docs.md`](./ForeachUnaryOp.cu_docs.md)
- **Folder**: `aten/src/ATen/native/cuda`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`Abs`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`Op`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`Reciprocal`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`Round`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`Rsqrt`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`Sigmoid`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`Sign`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`Trunc`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`functor_name`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)

### Functions

- **`all_types_complex_bfloat16_half_bool_`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`all_types_half_complex_bfloat16_`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`floating_complex_half_`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`floating_complex_half_bfloat16_`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`floating_half_`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`floating_half_bfloat16_`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`foreach_tensor_abs_cuda_`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`foreach_tensor_neg_cuda_`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`foreach_tensor_zero_cuda_`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`foreach_unary_op_`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)

### Includes

- **`ATen/Dispatch.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/NativeFunctions.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/native/ForeachUtils.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/native/cuda/ForeachFunctors.cuh`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_abs_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_acos_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_asin_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_atan_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_ceil_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_cos_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_cosh_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_erf_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_erfc_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_exp_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_expm1_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_floor_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_frac_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_lgamma_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_log10_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_log1p_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_log2_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_log_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_neg_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_reciprocal_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_round_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_rsqrt_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_sigmoid_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_sign_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_sin_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_sinh_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_sqrt_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_tan_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_tanh_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_trunc_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/_foreach_zero_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`ATen/ops/empty_like_native.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`c10/cuda/CUDAMathCompat.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)
- **`c10/util/TypeSafeSignMath.h`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)

### Namespaces

- **`at`**: [ForeachUnaryOp.cu_docs.md](./ForeachUnaryOp.cu_docs.md)


## Keyword → Section Map

The following sections in the documentation cover these topics:

- **File Metadata**: Basic file information
- **Original Source**: Complete source code
- **High-Level Overview**: Purpose and role
- **Detailed Analysis**: In-depth code analysis
- **Architecture & Design**: Design patterns and structure
- **Dependencies**: Related modules and imports
- **Performance Considerations**: Efficiency and optimization
- **Security & Safety**: Security analysis
- **Testing & Usage**: How to use and test

---

*Generated by PyTorch Repository Documentation System*
