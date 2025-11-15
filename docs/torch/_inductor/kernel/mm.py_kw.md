# Keyword Index: `torch/_inductor/kernel/mm.py`

## File Information

- **Original File**: [torch/_inductor/kernel/mm.py](../../../../torch/_inductor/kernel/mm.py)
- **Documentation**: [`mm.py_docs.md`](./mm.py_docs.md)
- **Folder**: `torch/_inductor/kernel`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`ContiguousTemplate`**: [mm.py_docs.md](./mm.py_docs.md)
- **`DecomposeKSugraphTemplate`**: [mm.py_docs.md](./mm.py_docs.md)

### Functions

- **`__init__`**: [mm.py_docs.md](./mm.py_docs.md)
- **`_compute_pid`**: [mm.py_docs.md](./mm.py_docs.md)
- **`_is_blockwise128x128_scaling`**: [mm.py_docs.md](./mm.py_docs.md)
- **`_is_blockwise1xTILESIZE_scaling`**: [mm.py_docs.md](./mm.py_docs.md)
- **`_is_int8_mat`**: [mm.py_docs.md](./mm.py_docs.md)
- **`_is_rowwise_scaling`**: [mm.py_docs.md](./mm.py_docs.md)
- **`_is_sm7x_or_older_gpu`**: [mm.py_docs.md](./mm.py_docs.md)
- **`_is_tensorwise_scaling`**: [mm.py_docs.md](./mm.py_docs.md)
- **`_to_dtype`**: [mm.py_docs.md](./mm.py_docs.md)
- **`apply_scaling`**: [mm.py_docs.md](./mm.py_docs.md)
- **`bias_addmm`**: [mm.py_docs.md](./mm.py_docs.md)
- **`blockwise128x128_scaling`**: [mm.py_docs.md](./mm.py_docs.md)
- **`blockwise1xTILESIZE_scaling`**: [mm.py_docs.md](./mm.py_docs.md)
- **`check_supported_striding`**: [mm.py_docs.md](./mm.py_docs.md)
- **`contiguous_addmm`**: [mm.py_docs.md](./mm.py_docs.md)
- **`contiguous_mm`**: [mm.py_docs.md](./mm.py_docs.md)
- **`decomposeK`**: [mm.py_docs.md](./mm.py_docs.md)
- **`dims_are_int`**: [mm.py_docs.md](./mm.py_docs.md)
- **`fallback`**: [mm.py_docs.md](./mm.py_docs.md)
- **`generate`**: [mm.py_docs.md](./mm.py_docs.md)
- **`get_context`**: [mm.py_docs.md](./mm.py_docs.md)
- **`get_scaling_options`**: [mm.py_docs.md](./mm.py_docs.md)
- **`get_size_hints`**: [mm.py_docs.md](./mm.py_docs.md)
- **`get_size_hints_strides`**: [mm.py_docs.md](./mm.py_docs.md)
- **`get_tile_size`**: [mm.py_docs.md](./mm.py_docs.md)
- **`has_zero_dim`**: [mm.py_docs.md](./mm.py_docs.md)
- **`is_col_major`**: [mm.py_docs.md](./mm.py_docs.md)
- **`is_desired_scaling`**: [mm.py_docs.md](./mm.py_docs.md)
- **`is_row_major`**: [mm.py_docs.md](./mm.py_docs.md)
- **`lazy_register_extern_choice`**: [mm.py_docs.md](./mm.py_docs.md)
- **`load_scales`**: [mm.py_docs.md](./mm.py_docs.md)
- **`mm_autoheuristic`**: [mm.py_docs.md](./mm.py_docs.md)
- **`tuned_addmm`**: [mm.py_docs.md](./mm.py_docs.md)
- **`tuned_int_mm`**: [mm.py_docs.md](./mm.py_docs.md)
- **`tuned_mm`**: [mm.py_docs.md](./mm.py_docs.md)
- **`tuned_scaled_mm`**: [mm.py_docs.md](./mm.py_docs.md)
- **`tuned_sparse_semi_structured_mm`**: [mm.py_docs.md](./mm.py_docs.md)

### Imports

- **`..`**: [mm.py_docs.md](./mm.py_docs.md)
- **`..codegen.cuda.gemm_template`**: [mm.py_docs.md](./mm.py_docs.md)
- **`..codegen.rocm.ck_tile_universal_gemm_template`**: [mm.py_docs.md](./mm.py_docs.md)
- **`..codegen.rocm.ck_universal_gemm_template`**: [mm.py_docs.md](./mm.py_docs.md)
- **`..codegen.subgraph`**: [mm.py_docs.md](./mm.py_docs.md)
- **`..decomposition`**: [mm.py_docs.md](./mm.py_docs.md)
- **`..ir`**: [mm.py_docs.md](./mm.py_docs.md)
- **`..kernel_inputs`**: [mm.py_docs.md](./mm.py_docs.md)
- **`..lowering`**: [mm.py_docs.md](./mm.py_docs.md)
- **`..select_algorithm`**: [mm.py_docs.md](./mm.py_docs.md)
- **`..utils`**: [mm.py_docs.md](./mm.py_docs.md)
- **`.mm_common`**: [mm.py_docs.md](./mm.py_docs.md)
- **`Any`**: [mm.py_docs.md](./mm.py_docs.md)
- **`AutoHeuristicSelectAlgorithm`**: [mm.py_docs.md](./mm.py_docs.md)
- **`Buffer`**: [mm.py_docs.md](./mm.py_docs.md)
- **`CKGemmTemplate`**: [mm.py_docs.md](./mm.py_docs.md)
- **`CKTileGemmTemplate`**: [mm.py_docs.md](./mm.py_docs.md)
- **`CUTLASS2xGemmTemplate`**: [mm.py_docs.md](./mm.py_docs.md)
- **`CppGemmTemplate`**: [mm.py_docs.md](./mm.py_docs.md)
- **`FixedLayout`**: [mm.py_docs.md](./mm.py_docs.md)
- **`MMKernelInputs`**: [mm.py_docs.md](./mm.py_docs.md)
- **`ScalingType`**: [mm.py_docs.md](./mm.py_docs.md)
- **`SubgraphChoiceCaller`**: [mm.py_docs.md](./mm.py_docs.md)
- **`TorchVersion`**: [mm.py_docs.md](./mm.py_docs.md)
- **`config`**: [mm.py_docs.md](./mm.py_docs.md)
- **`counters`**: [mm.py_docs.md](./mm.py_docs.md)
- **`enable_python_dispatcher`**: [mm.py_docs.md](./mm.py_docs.md)
- **`functools`**: [mm.py_docs.md](./mm.py_docs.md)
- **`gen_best_config`**: [mm.py_docs.md](./mm.py_docs.md)
- **`logging`**: [mm.py_docs.md](./mm.py_docs.md)
- **`make_fx`**: [mm.py_docs.md](./mm.py_docs.md)
- **`ops`**: [mm.py_docs.md](./mm.py_docs.md)
- **`realize_inputs`**: [mm.py_docs.md](./mm.py_docs.md)
- **`select_decomp_table`**: [mm.py_docs.md](./mm.py_docs.md)
- **`torch`**: [mm.py_docs.md](./mm.py_docs.md)
- **`torch._dispatch.python`**: [mm.py_docs.md](./mm.py_docs.md)
- **`torch._dynamo.utils`**: [mm.py_docs.md](./mm.py_docs.md)
- **`torch._inductor.autoheuristic.autoheuristic`**: [mm.py_docs.md](./mm.py_docs.md)
- **`torch._inductor.autoheuristic.autoheuristic_utils`**: [mm.py_docs.md](./mm.py_docs.md)
- **`torch._inductor.codegen.cpp_gemm_template`**: [mm.py_docs.md](./mm.py_docs.md)
- **`torch._inductor.ir`**: [mm.py_docs.md](./mm.py_docs.md)
- **`torch._inductor.remote_gemm_autotune_cache`**: [mm.py_docs.md](./mm.py_docs.md)
- **`torch._inductor.select_algorithm`**: [mm.py_docs.md](./mm.py_docs.md)
- **`torch._inductor.virtualized`**: [mm.py_docs.md](./mm.py_docs.md)
- **`torch.fx.experimental.proxy_tensor`**: [mm.py_docs.md](./mm.py_docs.md)
- **`torch.nn.functional`**: [mm.py_docs.md](./mm.py_docs.md)
- **`torch.torch_version`**: [mm.py_docs.md](./mm.py_docs.md)
- **`triton`**: [mm.py_docs.md](./mm.py_docs.md)
- **`typing`**: [mm.py_docs.md](./mm.py_docs.md)


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
