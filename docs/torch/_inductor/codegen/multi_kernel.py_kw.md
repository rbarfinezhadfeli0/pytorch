# Keyword Index: `torch/_inductor/codegen/multi_kernel.py`

## File Information

- **Original File**: [torch/_inductor/codegen/multi_kernel.py](../../../../torch/_inductor/codegen/multi_kernel.py)
- **Documentation**: [`multi_kernel.py_docs.md`](./multi_kernel.py_docs.md)
- **Folder**: `torch/_inductor/codegen`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`MultiKernel`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`MultiKernelCall`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`MultiKernelState`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`SizeHintMultiKernel`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`SizeHintMultiKernelCall`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`for`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`is`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`maintains`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)

### Functions

- **`__init__`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`_cache_shape_choice`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`_dist_heuristic`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`_get_cached_shape_choice`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`_get_filtered_args`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`_get_shape_cache_key`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`_merge_workspace_args`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`_metrics_table_row`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`_select_kernel_by_shape`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`benchmark_sub_kernels`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`cache_file_path`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`call_kernel`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`codegen_nan_check`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`define_kernel`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`dist`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`get_kernel_path`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`inner`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`inplace_update_buffers`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`inplaced_to_remove`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`kernels`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`load_cache`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`lookup_choice`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`merge_workspaces_inplace`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`record_choice`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`removed_buffers`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`run`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`store_cache`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`warn_mix_layout`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`wrap_fn`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)

### Imports

- **`..`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`..codecache`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`..runtime.benchmarking`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`..select_algorithm`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`..utils`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`..virtualized`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`.common`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`Any`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`GraphLowering`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`MultiTemplateBuffer`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`OrderedSet`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`TensorArg`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`TritonTemplateKernel`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`V`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`benchmarker`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`cache_on_self`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`code_hash`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`config`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`from`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`functools`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`get_metric_table`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`logging`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`math`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`os`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`pathlib`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`torch._inductor.graph`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`torch._inductor.ir`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`torch._inductor.metrics`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`torch.utils._ordered_set`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)
- **`typing`**: [multi_kernel.py_docs.md](./multi_kernel.py_docs.md)


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
