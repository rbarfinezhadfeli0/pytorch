# Keyword Index: `torch/testing/_internal/triton_utils.py`

## File Information

- **Original File**: [torch/testing/_internal/triton_utils.py](../../../../torch/testing/_internal/triton_utils.py)
- **Documentation**: [`triton_utils.py_docs.md`](./triton_utils.py_docs.md)
- **Folder**: `torch/testing/_internal`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Functions

- **`_dummy_early_config_prune`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`_get_strange_configs`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`add_4_times_kernel`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`add_kernel`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`add_kernel_2d_autotuned`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`add_kernel_autotuned`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`add_kernel_autotuned_weird_param_order`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`add_kernel_autotuned_with_unsupported_args`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`add_kernel_on_device_tma_new_api`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`add_kernel_on_device_tma_old_api`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`add_kernel_out_of_order_fn2`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`add_kernel_with_block_ptr`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`add_kernel_with_boolean_param`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`add_kernel_with_import`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`add_kernel_with_none_param_and_equal_to_1_arg`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`add_kernel_with_optional_param`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`add_kernel_with_scaling`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`add_kernel_with_tma_1d_new_api`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`add_kernel_with_tma_1d_old_api`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`add_kernel_with_tma_2d_new_api`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`add_kernel_with_tma_2d_old_api`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`atomic_add_kernel`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`cond_op_kernel`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`create_tensor_descriptor_shim`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`double_strided_kernel`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`indirection_kernel`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`inline_asm_kernel_is_pure_false`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`inline_asm_kernel_is_pure_true`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`kernel_inline_asm_double_quotes`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`kernel_inline_asm_single_quotes`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`kernel_with_block_ptr_2d`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`kernel_with_docstring_double_quotes`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`kernel_with_docstring_single_quotes`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`mul2_inplace_kernel`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`mul2_kernel`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`strange_config_matmul_kernel`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`sub_kernel`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`sub_kernel_autotuned`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`zero_negs`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)

### Imports

- **`has_triton`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`language`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`load`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`torch`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`torch.testing._internal.inductor_utils`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`torch.utils._triton`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`triton`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`triton.language`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)
- **`unittest`**: [triton_utils.py_docs.md](./triton_utils.py_docs.md)


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
