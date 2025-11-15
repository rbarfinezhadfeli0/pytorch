# Documentation: `docs/torch/_inductor/codegen/cuda_combined_scheduling.py_kw.md`

## File Metadata

- **Path**: `docs/torch/_inductor/codegen/cuda_combined_scheduling.py_kw.md`
- **Size**: 5,183 bytes (5.06 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
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

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor/codegen`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor/codegen`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs/torch/_inductor/codegen`):

- [`wrapper_fxir.py_kw.md_docs.md`](./wrapper_fxir.py_kw.md_docs.md)
- [`simd.py_docs.md_docs.md`](./simd.py_docs.md_docs.md)
- [`mps_device_op_overrides.py_docs.md_docs.md`](./mps_device_op_overrides.py_docs.md_docs.md)
- [`simd_kernel_features.py_docs.md_docs.md`](./simd_kernel_features.py_docs.md_docs.md)
- [`segmented_tree.py_docs.md_docs.md`](./segmented_tree.py_docs.md_docs.md)
- [`triton_utils.py_docs.md_docs.md`](./triton_utils.py_docs.md_docs.md)
- [`wrapper.py_kw.md_docs.md`](./wrapper.py_kw.md_docs.md)
- [`mps.py_kw.md_docs.md`](./mps.py_kw.md_docs.md)
- [`cpu_device_op_overrides.py_kw.md_docs.md`](./cpu_device_op_overrides.py_kw.md_docs.md)
- [`cpp_gemm_template.py_kw.md_docs.md`](./cpp_gemm_template.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `cuda_combined_scheduling.py_kw.md_docs.md`
- **Keyword Index**: `cuda_combined_scheduling.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
