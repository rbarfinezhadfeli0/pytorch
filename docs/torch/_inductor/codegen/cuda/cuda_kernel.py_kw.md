# Keyword Index: `torch/_inductor/codegen/cuda/cuda_kernel.py`

## File Information

- **Original File**: [torch/_inductor/codegen/cuda/cuda_kernel.py](../../../../../torch/_inductor/codegen/cuda/cuda_kernel.py)
- **Documentation**: [`cuda_kernel.py_docs.md`](./cuda_kernel.py_docs.md)
- **Folder**: `torch/_inductor/codegen/cuda`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`CUDAKernel`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`CUDATemplateCaller`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`CUDATemplateKernel`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`LayoutArg`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`for`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`from`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`of`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`represents`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)

### Functions

- **`__init__`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`__str__`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`_normalize_idx`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`add_layout_arg`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`batch_stride`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`benchmark`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`call_kernel`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`call_name`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`check_not_null`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`cutlass_dtype`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`def_kernel`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`dtype`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`find_layout_arg`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`find_ld_idx`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`find_symbol`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`get_dynamic_shape_args`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`get_layout_args`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`get_ld`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`get_offset_args`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`get_signature`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`hash_key`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`info_dict`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`init_layout_args`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`kernel_hash_key`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`load`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`matches`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`max_valid_index`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`output_node`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`precompile`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`ptr`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`row_or_column_stride`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`size`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`store`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`stride`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)

### Imports

- **`...autotune_process`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`...ir`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`...utils`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`...virtualized`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`..common`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`..cpp_utils`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`.cuda_template`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`.cutlass_utils`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`Any`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`ArgInfo`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`BaseSchedulerNode`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`CUDABenchmarkRequest`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`CUDATemplate`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`CUTLASSTemplate`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`Callable`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`CppPrinter`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`CppWrapperCpu`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`DTYPE_TO_CUTLASS_TYPE`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`Expr`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`V`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`ValueRanges`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`collections`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`collections.abc`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`dataclass`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`dataclasses`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`defaultdict`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`do_bench_using_profiling`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`dtype`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`functools`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`itertools`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`logging`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`sympy`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`sympy_product`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`torch`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`torch._inductor.codegen.cpp_wrapper_cpu`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`torch._inductor.codegen.cuda.cuda_template`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`torch._inductor.config`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`torch._inductor.scheduler`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`torch._inductor.utils`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`torch.utils._sympy.value_ranges`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)
- **`typing`**: [cuda_kernel.py_docs.md](./cuda_kernel.py_docs.md)


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
