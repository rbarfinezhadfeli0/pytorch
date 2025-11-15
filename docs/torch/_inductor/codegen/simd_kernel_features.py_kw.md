# Keyword Index: `torch/_inductor/codegen/simd_kernel_features.py`

## File Information

- **Original File**: [torch/_inductor/codegen/simd_kernel_features.py](../../../../torch/_inductor/codegen/simd_kernel_features.py)
- **Documentation**: [`simd_kernel_features.py_docs.md`](./simd_kernel_features.py_docs.md)
- **Folder**: `torch/_inductor/codegen`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`DisableReduction`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`EnableReduction`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`MemoryEstimator`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`NodeScheduleMarker`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`SIMDKernelFeatures`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`class`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)

### Functions

- **`__add__`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`__bool__`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`__init__`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`__repr__`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`buf_accesses`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`buffer_read_counts`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`bytes`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`bytes_per_thread`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`compute`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`contains_op`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`contiguous_score`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`count_per_thread`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`filter`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`get`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`get_mutations`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`get_reduction_hint`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`has_non_contiguous_pw_in_reduction_kernel`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`has_reduction_var`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`is_reduction`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`make_flat_range`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`memory_stats`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`only_nodes`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`op_counts`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`reduction_hint`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`reduction_nodes`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`remove`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`remove_kernel_local`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`scheduler_nodes`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`scope`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`select_index_dtype`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`set_ranges`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`simulate_codegen`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)

### Imports

- **`...utils._ordered_set`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`...utils._sympy.functions`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`...utils._sympy.symbol`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`..dependencies`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`..runtime.hints`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`..scheduler`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`..utils`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`..virtualized`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`.simd`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`Any`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`CoalesceVarAnalysis`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`Dep`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`FloorDiv`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`Iterable`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`OrderedSet`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`ReductionHint`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`SIMDKernel`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`SIMDScheduling`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`SchedulerNode`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`V`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`__future__`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`annotations`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`cache_on_self`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`collections`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`collections.abc`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`dataclasses`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`functools`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`itertools`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`make_symbol`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`sympy`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`torch`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`torch._inductor.tiling_utils`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)
- **`typing`**: [simd_kernel_features.py_docs.md](./simd_kernel_features.py_docs.md)


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
