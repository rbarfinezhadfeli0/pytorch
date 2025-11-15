# Keyword Index: `torch/_inductor/codegen/cpp_grouped_gemm_template.py`

## File Information

- **Original File**: [torch/_inductor/codegen/cpp_grouped_gemm_template.py](../../../../torch/_inductor/codegen/cpp_grouped_gemm_template.py)
- **Documentation**: [`cpp_grouped_gemm_template.py_docs.md`](./cpp_grouped_gemm_template.py_docs.md)
- **Folder**: `torch/_inductor/codegen`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`CppGroupedGemmTemplate`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)

### Functions

- **`__init__`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`_bias_add_epilogue`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`add_choices`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`get_deduplicated_act`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`maybe_to_dense`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`normalize_shapes`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`pack_weight`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`postprocessor`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`preprocessor`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`render`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`reorder_and_filter`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)

### Imports

- **`..`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`..._dynamo.utils`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`..kernel.mm_common`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`..select_algorithm`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`..utils`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`..virtualized`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`.cpp`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`.cpp_gemm_template`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`.cpp_micro_gemm`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`.cpp_template_kernel`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`.cpp_utils`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`Any`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`Callable`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`ChoiceCaller`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`CppMicroGemmAMX`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`CppTemplateKernel`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`OrderedSet`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`V`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`collections.abc`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`config`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`contextlib`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`counters`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`get_export_declaration`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`logging`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`mm_args`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`parallel_num_threads`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`patch`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`torch`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`torch.utils`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`torch.utils._ordered_set`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`typing`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)
- **`unittest.mock`**: [cpp_grouped_gemm_template.py_docs.md](./cpp_grouped_gemm_template.py_docs.md)


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
