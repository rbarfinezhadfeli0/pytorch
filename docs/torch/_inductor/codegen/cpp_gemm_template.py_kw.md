# Keyword Index: `torch/_inductor/codegen/cpp_gemm_template.py`

## File Information

- **Original File**: [torch/_inductor/codegen/cpp_gemm_template.py](../../../../torch/_inductor/codegen/cpp_gemm_template.py)
- **Documentation**: [`cpp_gemm_template.py_docs.md`](./cpp_gemm_template.py_docs.md)
- **Folder**: `torch/_inductor/codegen`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`CppGemmTemplate`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`CppWoqInt4GemmTemplate`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`CppWoqInt4GemmTemplateInstance`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`CppWoqInt4GemmTemplateMeta`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)

### Functions

- **`__getitem__`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`__init__`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`_bias_add_epilogue`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`_cache_blocking`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`_get_compensation_node`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`_is_int8_gemm`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`_maybe_remove_storage_offset`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`_thread_blocking`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`add_choices`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`block_weight`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`cache_blocking`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`check_if_block_weight`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`codegen_blocks`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`codegen_gemm_stub_def`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`codegen_m_loop_params`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`codegen_microkernel_def`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`codegen_multi_threads_params`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`codegen_n_loop_params`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`codegen_single_thread_params`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`copy_from_local_to_global_buffer_epilogue`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`copy_inner`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`expand_bias`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`gen_2d_view_of_epilogue_buf`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`get_better_blocking`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`get_blocking`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`get_cache_blocking`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`get_candidates`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`get_default_reindexers`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`get_factors`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`get_num_byte`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`get_occupancy`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`get_options`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`get_padded_n`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`get_padded_size`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`get_reindexer`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`is_int8_woq_gemm_small_m_dim`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`is_woq_int4`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`log_blockings`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`make_cache_blocking_cache`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`make_thread_blocking_cache`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`maybe_k_slicing`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`maybe_to_dense`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`need_copy_from_local_to_global_buffer_epilogue`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`normalize_shapes`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`pack_vnni_weight`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`postprocessor`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`prep_weight`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`preprocessor`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`prune_tensors`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`q_group_size`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`render`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`reorder_and_filter`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`share_storage`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`thread_blocking`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`transpose_w`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)

### Imports

- **`..`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`..._dynamo.utils`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`..kernel.mm_common`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`..select_algorithm`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`..utils`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`..virtualized`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`.cpp`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`.cpp_micro_gemm`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`.cpp_template`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`.cpp_template_kernel`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`.cpp_utils`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`Any`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`Callable`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`CppMicroGemmWoQInt4Amx`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`CppTemplate`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`CppTemplateKernel`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`DataProcessorTemplateWrapper`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`OrderedSet`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`collections.abc`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`config`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`contextlib`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`counters`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`functools`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`get_export_declaration`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`logging`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`lru_cache`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`math`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`mm_args`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`ops`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`patch`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`torch`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`torch.utils`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`torch.utils._ordered_set`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`typing`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)
- **`unittest.mock`**: [cpp_gemm_template.py_docs.md](./cpp_gemm_template.py_docs.md)


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
