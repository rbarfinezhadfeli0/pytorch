# Documentation: `docs/aten/src/ATen/native/LinearAlgebra.cpp_kw.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/LinearAlgebra.cpp_kw.md`
- **Size**: 21,576 bytes (21.07 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `aten/src/ATen/native/LinearAlgebra.cpp`

## File Information

- **Original File**: [aten/src/ATen/native/LinearAlgebra.cpp](../../../../../aten/src/ATen/native/LinearAlgebra.cpp)
- **Documentation**: [`LinearAlgebra.cpp_docs.md`](./LinearAlgebra.cpp_docs.md)
- **Folder**: `aten/src/ATen/native`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`KronImpl`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)

### Functions

- **`KronImpl`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`_allocate_buffer`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`_blob_to_Tensor`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`_convert_weight_to_int4pack_cpu`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`_dyn_quant_matmul_4bit_cpu`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`_dyn_quant_pack_4bit_weight_cpu`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`_fill_matrix_powers`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`_int_mm_cpu`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`_linalg_cond_check_ord`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`_linalg_cond_empty_matrix`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`_linalg_cond_helper`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`_linalg_matrix_norm_checks`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`_linear_combination`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`_matmul_impl`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`_move_memory_if_cuda_input`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`_weight_int4pack_mm_cpu`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`_weight_int8pack_mm_cpu`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`addbmm`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`addbmm_impl_`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`addmm_impl_cpu_`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`addr`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`apply_mkldnn_matmul_heur`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`backward_analytic_function_of_a_matrix`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`baddbmm_cpu_kernel`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`baddbmm_with_gemm_`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`bmm_out_or_baddbmm_`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`build_addr_iter`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`chain_matmul`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`check_1d`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`check_addr_scalar`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`check_linalg_norm_dtype`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`common_checks_baddbmm_bmm`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`compute_T1`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`compute_T12`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`compute_T18`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`compute_T18_scale_square`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`compute_T2`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`compute_T4`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`compute_T8`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`conjugate_mutable_input_if_needed`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`det`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`frobenius_norm`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ger`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`get_matrix_rank_result_tensor`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`get_mkldnn_matmul_min_dim`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`get_mkldnn_matmul_min_size`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`if`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`inner`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`kron`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`linalg_cond`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`linalg_det`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`linalg_diagonal`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`linalg_matmul`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`linalg_matrix_exp`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`linalg_matrix_norm`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`linalg_matrix_power`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`linalg_matrix_power_impl`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`linalg_matrix_rank`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`linalg_multi_dot`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`linalg_norm`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`linalg_pinv`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`linalg_tensorinv`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`linalg_tensorsolve`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`logdet`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`lu_det_P`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`math_addr`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`matmul`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`matrix_chain_multiplication`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`matrix_exp`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`matrix_exp_backward`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`matrix_power`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`mexp`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`mexp_impl`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`multi_dot_impl`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`nuclear_norm`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`operator_1_norm`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`outer`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`pinverse`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`should_fold`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)

### Includes

- **`ATen/Context.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/Dispatch.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ExpandUtils.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/Functions.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/NamedTensorUtils.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/NativeFunctions.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/OpMathType.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/Parallel.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/TensorIndexing.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/TensorIterator.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/TensorOperators.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/TensorSubclassLikeUtils.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/TensorUtils.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/core/Tensor.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/cpu/Utils.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/native/CPUBlas.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/native/LinearAlgebra.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/native/LinearAlgebraUtils.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/native/ReduceOps.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/native/ReduceOpsUtils.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/native/Resize.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/native/cpu/int_mm_kernel.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/native/mkldnn/Matmul.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/native/mkldnn/Utils.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/_addmm_activation_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/_compute_linear_combination_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/_convert_weight_to_int4pack_for_cpu_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/_dyn_quant_matmul_4bit_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/_dyn_quant_pack_4bit_weight_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/_int_mm_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/_linalg_check_errors.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/_linalg_det.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/_linalg_det_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/_linalg_slogdet.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/_linalg_slogdet_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/_unsafe_view.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/_weight_int4pack_mm_for_cpu_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/_weight_int8pack_mm_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/abs.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/addbmm_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/addmm_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/addr.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/addr_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/arange.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/argsort.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/baddbmm_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/bmm.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/bmm_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/cat.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/ceil.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/chain_matmul_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/cumsum.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/det_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/diag_embed.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/diff.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/dot.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/dot_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/empty.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/empty_like.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/eye.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/floor.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/frobenius_norm_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/from_blob.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/full.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/full_like.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/gelu.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/ger_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/index_select.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/inner_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/is_complex_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/is_floating_point_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/kron_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_cond.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_cond_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_det.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_det_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_diagonal_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_eigh.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_eigvalsh.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_inv.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_inv_ex.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_lu_factor_ex.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_matmul_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_matrix_exp.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_matrix_exp_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_matrix_norm.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_matrix_norm_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_matrix_power_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_matrix_rank.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_matrix_rank_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_multi_dot_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_norm.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_norm_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_pinv.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_pinv_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_slogdet.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_slogdet_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_solve.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_svdvals.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_tensorinv.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_tensorinv_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_tensorsolve.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_tensorsolve_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_vector_norm.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/linalg_vector_norm_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/log2.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/logdet_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/matmul.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/matmul_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/matrix_exp_backward_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/matrix_exp_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/matrix_power_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/max.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/mm.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/mm_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/movedim.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/mul.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/mv.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/narrow.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/ne.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/norm.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/nuclear_norm_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/ones.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/outer.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/outer_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/pinverse_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/pow.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/prod.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/real.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/relu.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/slogdet_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/sort.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/sqrt.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/sum.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/tensordot.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/unique_consecutive.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/vdot_native.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/where.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/zeros.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`ATen/ops/zeros_like.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`c10/core/GradMode.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`c10/util/accumulate.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`c10/util/env.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`c10/util/irange.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`cpuinfo.h`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`limits`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`numeric`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`string`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`tuple`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`utility`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`variant`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)

### Namespaces

- **`Tensor`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`at`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`detail`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`meta`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)
- **`native`**: [LinearAlgebra.cpp_docs.md](./LinearAlgebra.cpp_docs.md)


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

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


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

Files in the same folder (`docs/aten/src/ATen/native`):

- [`AdaptiveMaxPooling2d.cpp_docs.md_docs.md`](./AdaptiveMaxPooling2d.cpp_docs.md_docs.md)
- [`Distributions.cpp_docs.md_docs.md`](./Distributions.cpp_docs.md_docs.md)
- [`im2col_shape_check.h_docs.md_docs.md`](./im2col_shape_check.h_docs.md_docs.md)
- [`ReduceOps.cpp_kw.md_docs.md`](./ReduceOps.cpp_kw.md_docs.md)
- [`Lerp.cpp_kw.md_docs.md`](./Lerp.cpp_kw.md_docs.md)
- [`CPUFallback.h_docs.md_docs.md`](./CPUFallback.h_docs.md_docs.md)
- [`MetaTensor.cpp_docs.md_docs.md`](./MetaTensor.cpp_docs.md_docs.md)
- [`Correlation.cpp_kw.md_docs.md`](./Correlation.cpp_kw.md_docs.md)
- [`im2col_shape_check.h_kw.md_docs.md`](./im2col_shape_check.h_kw.md_docs.md)
- [`UpSampleNearest2d.cpp_kw.md_docs.md`](./UpSampleNearest2d.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `LinearAlgebra.cpp_kw.md_docs.md`
- **Keyword Index**: `LinearAlgebra.cpp_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
