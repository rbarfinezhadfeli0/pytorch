# Documentation: `docs/torch/_inductor/codegen/wrapper_fxir.py_kw.md`

## File Metadata

- **Path**: `docs/torch/_inductor/codegen/wrapper_fxir.py_kw.md`
- **Size**: 11,495 bytes (11.23 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/_inductor/codegen/wrapper_fxir.py`

## File Information

- **Original File**: [torch/_inductor/codegen/wrapper_fxir.py](../../../../torch/_inductor/codegen/wrapper_fxir.py)
- **Documentation**: [`wrapper_fxir.py_docs.md`](./wrapper_fxir.py_docs.md)
- **Folder**: `torch/_inductor/codegen`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`SubgraphFxWrapperCodegen`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`UnbackedSymintsError`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`WrapperFxCodegen`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`class`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)

### Functions

- **`__init__`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`__post_init__`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_codegen_symbol`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_create_as_strided`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_free`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_allocate`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_buffer`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_comm_buffer_allocate`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_comm_buffer_free`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_comment`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_conditional`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_dynamic_scalar`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_enter_device_context_manager`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_enter_subgraph`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_exit_device_context_manager`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_exit_subgraph`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_extern_kernel_alloc`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_extern_kernel_common`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_extern_kernel_out`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_fallback_call`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_free`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_free_if_not_reused`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_graph_constants`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_graph_input_shapes`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_graph_inputs`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_index_put_fallback`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_kernel_call`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_kernel_definition`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_line_context`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_multi_output`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_null`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_outputs`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_reinterpret`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_reinterpret_helper`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_reuse`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_scatter_fallback`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_size_proxy`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_subgm_getattrs`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_sym_node`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_sym_nodes`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_symbolic_call_arg`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_triton_call`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_generate_unbacked_symbol_defs`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_get_buffer`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_get_subgm_attr`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_import_kernel`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_lookup_args`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_record_allocation`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_sympy_interp`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`add_constants_to_call_args`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`codegen_conditional`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`codegen_inputs`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`codegen_proxy`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`compile_graph`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`convert_key`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`crash_if_run`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`create`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`define_subgraph_launcher_fn`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`generate`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`generate_buffer`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`generate_buffer_or_none`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`generate_getattr`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`generate_item`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`generate_to_buffer`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`get_example`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`get_fx_graph_inputs`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`get_name`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`get_subgm_attr`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`is_subgraph`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`node_to_tuning_arg`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`replace`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`replace_floor_div`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`to_size_hint`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`tune_kernel`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`write_header`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)

### Imports

- **`..`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`..runtime.triton_compat`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`..utils`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`.common`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`.wrapper`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`Any`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`CachingAutotuner`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`Callable`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`Config`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`Counter`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`FloorDiv`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`GraphModule`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`LambdaFuture`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`OptimizedPythonReferenceAnalysis`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`V`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_pytree`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`_run_sympy_handler`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`cache_property_on_self`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`collections`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`collections.abc`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`config`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`convert_to_symint`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`dataclasses`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`driver`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`extern_kernels`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`functools`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`logging`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`operator`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`sympy`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`textwrap`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`torch`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`torch._export.passes._node_metadata_hook`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`torch._higher_order_ops.triton_kernel_wrap`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`torch._inductor.codecache`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`torch._inductor.runtime.triton_heuristics`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`torch._inductor.select_algorithm`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`torch._inductor.utils`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`torch._inductor.virtualized`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`torch._library.triton`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`torch.fx`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`torch.fx.experimental.symbolic_shapes`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`torch.utils`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`torch.utils._sympy.functions`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`torch.utils._sympy.interp`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`torch.utils._sympy.reference`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`torch.utils._sympy.solve`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`triton.runtime`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`try_solve`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`typing`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)
- **`wrap_triton`**: [wrapper_fxir.py_docs.md](./wrapper_fxir.py_docs.md)


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

- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

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

- **File Documentation**: `wrapper_fxir.py_kw.md_docs.md`
- **Keyword Index**: `wrapper_fxir.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
