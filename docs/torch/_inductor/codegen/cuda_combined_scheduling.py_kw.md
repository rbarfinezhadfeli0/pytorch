# Keyword Index: `torch/_inductor/codegen/cuda_combined_scheduling.py`

## File Information

- **Original File**: [torch/_inductor/codegen/cuda_combined_scheduling.py](../../../../torch/_inductor/codegen/cuda_combined_scheduling.py)
- **Documentation**: [`cuda_combined_scheduling.py_docs.md`](./cuda_combined_scheduling.py_docs.md)
- **Folder**: `torch/_inductor/codegen`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`CUDACombinedScheduling`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)

### Functions

- **`__init__`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`benchmark_codegened_module`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`benchmark_combo_kernel`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`benchmark_fused_nodes`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`can_fuse_horizontal`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`can_fuse_vertical`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`choose_node_backend`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`codegen_combo_kernel`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`codegen_mix_order_reduction`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`codegen_node`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`codegen_sync`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`codegen_template`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`flush`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`generate_kernel_code_from_nodes`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`get_backend_features`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`group_fn`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)

### Imports

- **`..scheduler`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`.common`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`.cuda.cuda_cpp_scheduling`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`.cutedsl.cutedsl_scheduling`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`.rocm.rocm_cpp_scheduling`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`.triton`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`Any`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`BackendFeature`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`CUDACPPScheduling`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`CuteDSLScheduling`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`Expr`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`OrderedSet`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`ROCmCPPScheduling`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`Sequence`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`TritonScheduling`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`TypeAlias`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`__future__`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`annotations`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`collections.abc`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`sympy`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`torch`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`torch.utils._ordered_set`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)
- **`typing`**: [cuda_combined_scheduling.py_docs.md](./cuda_combined_scheduling.py_docs.md)


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
