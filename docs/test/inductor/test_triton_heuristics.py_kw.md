# Keyword Index: `test/inductor/test_triton_heuristics.py`

## File Information

- **Original File**: [test/inductor/test_triton_heuristics.py](../../../test/inductor/test_triton_heuristics.py)
- **Documentation**: [`test_triton_heuristics.py_docs.md`](./test_triton_heuristics.py_docs.md)
- **Folder**: `test/inductor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`TestArgumentCloneAndRestore`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`TestTritonHeuristics`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)

### Functions

- **`_create_caching_autotuner`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`_create_tensor`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`_do_test`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`_get_cos_kernel_caching_autotuner_args`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`_test_artificial_zgrid`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`amd_sqr_kernel`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`fn`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`forward`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`get_autotuned_amd_sqr_kernel`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`grid`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`mock_triton_config`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`pre_hook`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`test_amd_special_config_args`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`test_artificial_grid_cpp_wrapper`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`test_artificial_zgrid`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`test_autotune_hints_to_configs`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`test_clone_args_with_non_zero_offset`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`test_clone_contiguous_args`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`test_clone_non_contiguous_args`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`test_pre_hook_assert`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`test_prune_configs_over_shared_memory_limit`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`test_template_function_ws`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`test_triton_config`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`triton_`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`triton_sqr`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)

### Imports

- **`HAS_WARP_SPEC`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`MagicMock`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`clone_preserve_strides`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`config`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`functools`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`math`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`rand_strided`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`run_tests`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`skipUnless`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`sys`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`torch`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`torch._dynamo.testing`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`torch._inductor`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`torch._inductor.runtime.hints`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`torch._inductor.runtime.triton_compat`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`torch._inductor.runtime.triton_helpers`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`torch._inductor.runtime.triton_heuristics`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`torch._inductor.template_heuristics.triton`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`torch._inductor.test_case`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`torch._inductor.utils`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`torch.testing._internal.inductor_utils`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`triton`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`triton.language`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`unittest`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)
- **`unittest.mock`**: [test_triton_heuristics.py_docs.md](./test_triton_heuristics.py_docs.md)


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
