# Documentation: `docs/torch/_inductor/graph.py_kw.md`

## File Metadata

- **Path**: `docs/torch/_inductor/graph.py_kw.md`
- **Size**: 12,803 bytes (12.50 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/_inductor/graph.py`

## File Information

- **Original File**: [torch/_inductor/graph.py](../../../torch/_inductor/graph.py)
- **Documentation**: [`graph.py_docs.md`](./graph.py_docs.md)
- **Folder**: `torch/_inductor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`GraphLowering`**: [graph.py_docs.md](./graph.py_docs.md)
- **`Model`**: [graph.py_docs.md](./graph.py_docs.md)
- **`SubgraphLowering`**: [graph.py_docs.md](./graph.py_docs.md)
- **`for`**: [graph.py_docs.md](./graph.py_docs.md)

### Functions

- **`__init__`**: [graph.py_docs.md](./graph.py_docs.md)
- **`_compile_to_module`**: [graph.py_docs.md](./graph.py_docs.md)
- **`_compile_to_module_lines`**: [graph.py_docs.md](./graph.py_docs.md)
- **`_get_output_names`**: [graph.py_docs.md](./graph.py_docs.md)
- **`_get_overload_packet`**: [graph.py_docs.md](./graph.py_docs.md)
- **`_update_scheduler`**: [graph.py_docs.md](./graph.py_docs.md)
- **`add_device_info`**: [graph.py_docs.md](./graph.py_docs.md)
- **`add_symbol_graph_input`**: [graph.py_docs.md](./graph.py_docs.md)
- **`add_tensor_constant`**: [graph.py_docs.md](./graph.py_docs.md)
- **`allocate_non_dup_const_name`**: [graph.py_docs.md](./graph.py_docs.md)
- **`call_function`**: [graph.py_docs.md](./graph.py_docs.md)
- **`call_method`**: [graph.py_docs.md](./graph.py_docs.md)
- **`call_module`**: [graph.py_docs.md](./graph.py_docs.md)
- **`can_inline_constant`**: [graph.py_docs.md](./graph.py_docs.md)
- **`codegen`**: [graph.py_docs.md](./graph.py_docs.md)
- **`codegen_subgraph`**: [graph.py_docs.md](./graph.py_docs.md)
- **`codegen_with_cpp_wrapper`**: [graph.py_docs.md](./graph.py_docs.md)
- **`compile_to_module`**: [graph.py_docs.md](./graph.py_docs.md)
- **`constant_name`**: [graph.py_docs.md](./graph.py_docs.md)
- **`count_bytes`**: [graph.py_docs.md](./graph.py_docs.md)
- **`create_deferred_runtime_asserts`**: [graph.py_docs.md](./graph.py_docs.md)
- **`debug`**: [graph.py_docs.md](./graph.py_docs.md)
- **`decide_layout_opt`**: [graph.py_docs.md](./graph.py_docs.md)
- **`extend_user_visible_output_strides`**: [graph.py_docs.md](./graph.py_docs.md)
- **`extract_autotune_inputs`**: [graph.py_docs.md](./graph.py_docs.md)
- **`extract_real_inputs`**: [graph.py_docs.md](./graph.py_docs.md)
- **`fake_mode`**: [graph.py_docs.md](./graph.py_docs.md)
- **`finalize`**: [graph.py_docs.md](./graph.py_docs.md)
- **`find_nodes_prefer_channels_last`**: [graph.py_docs.md](./graph.py_docs.md)
- **`format_new_defs`**: [graph.py_docs.md](./graph.py_docs.md)
- **`forward`**: [graph.py_docs.md](./graph.py_docs.md)
- **`freeze_runtime_asserts`**: [graph.py_docs.md](./graph.py_docs.md)
- **`get_allocation_size`**: [graph.py_docs.md](./graph.py_docs.md)
- **`get_allocation_storage_size`**: [graph.py_docs.md](./graph.py_docs.md)
- **`get_attr`**: [graph.py_docs.md](./graph.py_docs.md)
- **`get_buffer`**: [graph.py_docs.md](./graph.py_docs.md)
- **`get_current_device_or_throw`**: [graph.py_docs.md](./graph.py_docs.md)
- **`get_dep_size_hint`**: [graph.py_docs.md](./graph.py_docs.md)
- **`get_dtype`**: [graph.py_docs.md](./graph.py_docs.md)
- **`get_numel`**: [graph.py_docs.md](./graph.py_docs.md)
- **`get_original_value_of_constant`**: [graph.py_docs.md](./graph.py_docs.md)
- **`get_output_names`**: [graph.py_docs.md](./graph.py_docs.md)
- **`get_training_phase`**: [graph.py_docs.md](./graph.py_docs.md)
- **`get_user_visible_output_strides`**: [graph.py_docs.md](./graph.py_docs.md)
- **`getattr_recursive`**: [graph.py_docs.md](./graph.py_docs.md)
- **`has_feature`**: [graph.py_docs.md](./graph.py_docs.md)
- **`init_wrapper_code`**: [graph.py_docs.md](./graph.py_docs.md)
- **`is_convertible`**: [graph.py_docs.md](./graph.py_docs.md)
- **`is_grouped`**: [graph.py_docs.md](./graph.py_docs.md)
- **`is_in_out_channel`**: [graph.py_docs.md](./graph.py_docs.md)
- **`is_magic_method`**: [graph.py_docs.md](./graph.py_docs.md)
- **`is_small_channel`**: [graph.py_docs.md](./graph.py_docs.md)
- **`is_unspec_arg`**: [graph.py_docs.md](./graph.py_docs.md)
- **`log_module_code`**: [graph.py_docs.md](./graph.py_docs.md)
- **`make_assert`**: [graph.py_docs.md](./graph.py_docs.md)
- **`make_subgraph`**: [graph.py_docs.md](./graph.py_docs.md)
- **`mark_buffer_mutated`**: [graph.py_docs.md](./graph.py_docs.md)
- **`mark_nodes_dislike_padding`**: [graph.py_docs.md](./graph.py_docs.md)
- **`materialize`**: [graph.py_docs.md](./graph.py_docs.md)
- **`may_get_constant_buffer_dtype`**: [graph.py_docs.md](./graph.py_docs.md)
- **`maybe_propagate`**: [graph.py_docs.md](./graph.py_docs.md)
- **`normalize`**: [graph.py_docs.md](./graph.py_docs.md)
- **`output`**: [graph.py_docs.md](./graph.py_docs.md)
- **`placeholder`**: [graph.py_docs.md](./graph.py_docs.md)
- **`propagate_mutation`**: [graph.py_docs.md](./graph.py_docs.md)
- **`qualify_name`**: [graph.py_docs.md](./graph.py_docs.md)
- **`register`**: [graph.py_docs.md](./graph.py_docs.md)
- **`register_buffer`**: [graph.py_docs.md](./graph.py_docs.md)
- **`register_operation`**: [graph.py_docs.md](./graph.py_docs.md)
- **`register_operation_list`**: [graph.py_docs.md](./graph.py_docs.md)
- **`register_users_of`**: [graph.py_docs.md](./graph.py_docs.md)
- **`run`**: [graph.py_docs.md](./graph.py_docs.md)
- **`run_node`**: [graph.py_docs.md](./graph.py_docs.md)
- **`set_current_device`**: [graph.py_docs.md](./graph.py_docs.md)
- **`set_current_node`**: [graph.py_docs.md](./graph.py_docs.md)
- **`set_current_wrapper_code`**: [graph.py_docs.md](./graph.py_docs.md)
- **`static_sizes_strides`**: [graph.py_docs.md](./graph.py_docs.md)
- **`symbolic_sizes_strides`**: [graph.py_docs.md](./graph.py_docs.md)
- **`try_get_buffer`**: [graph.py_docs.md](./graph.py_docs.md)
- **`validate_can_generate_cpp_wrapper`**: [graph.py_docs.md](./graph.py_docs.md)
- **`warn_fallback`**: [graph.py_docs.md](./graph.py_docs.md)

### Imports

- **`.`**: [graph.py_docs.md](./graph.py_docs.md)
- **`.codecache`**: [graph.py_docs.md](./graph.py_docs.md)
- **`.codegen.common`**: [graph.py_docs.md](./graph.py_docs.md)
- **`.codegen.wrapper`**: [graph.py_docs.md](./graph.py_docs.md)
- **`.compile_fx`**: [graph.py_docs.md](./graph.py_docs.md)
- **`.dependencies`**: [graph.py_docs.md](./graph.py_docs.md)
- **`.exc`**: [graph.py_docs.md](./graph.py_docs.md)
- **`.fx_utils`**: [graph.py_docs.md](./graph.py_docs.md)
- **`.ir`**: [graph.py_docs.md](./graph.py_docs.md)
- **`.lowering`**: [graph.py_docs.md](./graph.py_docs.md)
- **`.runtime`**: [graph.py_docs.md](./graph.py_docs.md)
- **`.runtime.autotune_cache`**: [graph.py_docs.md](./graph.py_docs.md)
- **`.scheduler`**: [graph.py_docs.md](./graph.py_docs.md)
- **`.sizevars`**: [graph.py_docs.md](./graph.py_docs.md)
- **`.utils`**: [graph.py_docs.md](./graph.py_docs.md)
- **`.virtualized`**: [graph.py_docs.md](./graph.py_docs.md)
- **`Any`**: [graph.py_docs.md](./graph.py_docs.md)
- **`AutotuneCacheBundler`**: [graph.py_docs.md](./graph.py_docs.md)
- **`BackwardState`**: [graph.py_docs.md](./graph.py_docs.md)
- **`BaseSchedulerNode`**: [graph.py_docs.md](./graph.py_docs.md)
- **`Callable`**: [graph.py_docs.md](./graph.py_docs.md)
- **`CompilerBisector`**: [graph.py_docs.md](./graph.py_docs.md)
- **`ConstantSource`**: [graph.py_docs.md](./graph.py_docs.md)
- **`Dep`**: [graph.py_docs.md](./graph.py_docs.md)
- **`Expr`**: [graph.py_docs.md](./graph.py_docs.md)
- **`FakeScriptObject`**: [graph.py_docs.md](./graph.py_docs.md)
- **`FakeTensor`**: [graph.py_docs.md](./graph.py_docs.md)
- **`Graph`**: [graph.py_docs.md](./graph.py_docs.md)
- **`GraphModule`**: [graph.py_docs.md](./graph.py_docs.md)
- **`LazyString`**: [graph.py_docs.md](./graph.py_docs.md)
- **`ModuleType`**: [graph.py_docs.md](./graph.py_docs.md)
- **`Node`**: [graph.py_docs.md](./graph.py_docs.md)
- **`NullHandler`**: [graph.py_docs.md](./graph.py_docs.md)
- **`OrderedSet`**: [graph.py_docs.md](./graph.py_docs.md)
- **`PyCodeCache`**: [graph.py_docs.md](./graph.py_docs.md)
- **`PythonWrapperCodegen`**: [graph.py_docs.md](./graph.py_docs.md)
- **`Scheduler`**: [graph.py_docs.md](./graph.py_docs.md)
- **`SizeVarAllocator`**: [graph.py_docs.md](./graph.py_docs.md)
- **`_EffectType`**: [graph.py_docs.md](./graph.py_docs.md)
- **`__future__`**: [graph.py_docs.md](./graph.py_docs.md)
- **`_is_view_op`**: [graph.py_docs.md](./graph.py_docs.md)
- **`annotations`**: [graph.py_docs.md](./graph.py_docs.md)
- **`autotune_cache`**: [graph.py_docs.md](./graph.py_docs.md)
- **`clone_preserve_strides`**: [graph.py_docs.md](./graph.py_docs.md)
- **`collections`**: [graph.py_docs.md](./graph.py_docs.md)
- **`collections.abc`**: [graph.py_docs.md](./graph.py_docs.md)
- **`config`**: [graph.py_docs.md](./graph.py_docs.md)
- **`contextlib`**: [graph.py_docs.md](./graph.py_docs.md)
- **`contextmanager`**: [graph.py_docs.md](./graph.py_docs.md)
- **`copy`**: [graph.py_docs.md](./graph.py_docs.md)
- **`count_flops_fx`**: [graph.py_docs.md](./graph.py_docs.md)
- **`defake`**: [graph.py_docs.md](./graph.py_docs.md)
- **`defaultdict`**: [graph.py_docs.md](./graph.py_docs.md)
- **`device`**: [graph.py_docs.md](./graph.py_docs.md)
- **`extern_node_json_serializer`**: [graph.py_docs.md](./graph.py_docs.md)
- **`full_aoti_runtime_assert`**: [graph.py_docs.md](./graph.py_docs.md)
- **`functools`**: [graph.py_docs.md](./graph.py_docs.md)
- **`get_decompositions`**: [graph.py_docs.md](./graph.py_docs.md)
- **`get_layout_constraint_tag`**: [graph.py_docs.md](./graph.py_docs.md)
- **`int_oo`**: [graph.py_docs.md](./graph.py_docs.md)
- **`itertools`**: [graph.py_docs.md](./graph.py_docs.md)
- **`log_module_code`**: [graph.py_docs.md](./graph.py_docs.md)
- **`logging`**: [graph.py_docs.md](./graph.py_docs.md)
- **`magic_methods`**: [graph.py_docs.md](./graph.py_docs.md)
- **`no_dispatch`**: [graph.py_docs.md](./graph.py_docs.md)
- **`operator`**: [graph.py_docs.md](./graph.py_docs.md)
- **`os`**: [graph.py_docs.md](./graph.py_docs.md)
- **`output_code_log`**: [graph.py_docs.md](./graph.py_docs.md)
- **`re`**: [graph.py_docs.md](./graph.py_docs.md)
- **`sympy`**: [graph.py_docs.md](./graph.py_docs.md)
- **`sys`**: [graph.py_docs.md](./graph.py_docs.md)
- **`tensor`**: [graph.py_docs.md](./graph.py_docs.md)
- **`time`**: [graph.py_docs.md](./graph.py_docs.md)
- **`torch`**: [graph.py_docs.md](./graph.py_docs.md)
- **`torch._decomp`**: [graph.py_docs.md](./graph.py_docs.md)
- **`torch._dynamo.source`**: [graph.py_docs.md](./graph.py_docs.md)
- **`torch._dynamo.utils`**: [graph.py_docs.md](./graph.py_docs.md)
- **`torch._higher_order_ops.effects`**: [graph.py_docs.md](./graph.py_docs.md)
- **`torch._inductor.codecache`**: [graph.py_docs.md](./graph.py_docs.md)
- **`torch._inductor.compiler_bisector`**: [graph.py_docs.md](./graph.py_docs.md)
- **`torch._inductor.extern_node_serializer`**: [graph.py_docs.md](./graph.py_docs.md)
- **`torch._inductor.fb.utils`**: [graph.py_docs.md](./graph.py_docs.md)
- **`torch._library.fake_class_registry`**: [graph.py_docs.md](./graph.py_docs.md)
- **`torch._library.utils`**: [graph.py_docs.md](./graph.py_docs.md)
- **`torch._logging`**: [graph.py_docs.md](./graph.py_docs.md)
- **`torch._prims_common`**: [graph.py_docs.md](./graph.py_docs.md)
- **`torch._subclasses.fake_tensor`**: [graph.py_docs.md](./graph.py_docs.md)
- **`torch._utils_internal`**: [graph.py_docs.md](./graph.py_docs.md)
- **`torch.fx`**: [graph.py_docs.md](./graph.py_docs.md)
- **`torch.fx.experimental._backward_state`**: [graph.py_docs.md](./graph.py_docs.md)
- **`torch.fx.experimental.sym_node`**: [graph.py_docs.md](./graph.py_docs.md)
- **`torch.fx.experimental.symbolic_shapes`**: [graph.py_docs.md](./graph.py_docs.md)
- **`torch.fx.graph`**: [graph.py_docs.md](./graph.py_docs.md)
- **`torch.fx.node`**: [graph.py_docs.md](./graph.py_docs.md)
- **`torch.fx.passes.reinplace`**: [graph.py_docs.md](./graph.py_docs.md)
- **`torch.utils._mode_utils`**: [graph.py_docs.md](./graph.py_docs.md)
- **`torch.utils._ordered_set`**: [graph.py_docs.md](./graph.py_docs.md)
- **`torch.utils._sympy.numbers`**: [graph.py_docs.md](./graph.py_docs.md)
- **`types`**: [graph.py_docs.md](./graph.py_docs.md)
- **`typing`**: [graph.py_docs.md](./graph.py_docs.md)


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

This file is part of the PyTorch framework located at `docs/torch/_inductor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/_inductor`):

- [`freezing.py_docs.md_docs.md`](./freezing.py_docs.md_docs.md)
- [`lowering.py_kw.md_docs.md`](./lowering.py_kw.md_docs.md)
- [`quantized_lowerings.py_docs.md_docs.md`](./quantized_lowerings.py_docs.md_docs.md)
- [`select_algorithm.py_docs.md_docs.md`](./select_algorithm.py_docs.md_docs.md)
- [`kernel_inputs.py_kw.md_docs.md`](./kernel_inputs.py_kw.md_docs.md)
- [`compile_fx_ext.py_kw.md_docs.md`](./compile_fx_ext.py_kw.md_docs.md)
- [`extern_node_serializer.py_docs.md_docs.md`](./extern_node_serializer.py_docs.md_docs.md)
- [`mkldnn_lowerings.py_kw.md_docs.md`](./mkldnn_lowerings.py_kw.md_docs.md)
- [`ops_handler.py_docs.md_docs.md`](./ops_handler.py_docs.md_docs.md)
- [`test_operators.py_docs.md_docs.md`](./test_operators.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `graph.py_kw.md_docs.md`
- **Keyword Index**: `graph.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
