# Keyword Index: `torch/export/exported_program.py`

## File Information

- **Original File**: [torch/export/exported_program.py](../../../torch/export/exported_program.py)
- **Documentation**: [`exported_program.py_docs.md`](./exported_program.py_docs.md)
- **Folder**: `torch/export`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`CallSpec`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`ExportedProgram`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`as`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`class`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`desugaring`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`parameter`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`parameters`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`unwrapping`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`weights`**: [exported_program.py_docs.md](./exported_program.py_docs.md)

### Functions

- **`__call__`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`__init__`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`__str__`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_check_input_constraints`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_common_getitem_elimination_pass`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_convert_guards_to_code`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_create_graph_module_for_export`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_decompose_and_get_gm_with_new_signature_constants`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_decompose_exported_program`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_disable_prexisiting_fake_mode`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_eval`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_force_dispatch_to_orig_cia_callable`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_fx_collection_equivalence_fn`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_get_flat_args_with_check`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_get_shape_env`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_get_updated_graph_signature`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_get_updated_module_call_graph`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_get_updated_range_constraints`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_graph_module_flat_inputs`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_is_joint_ir_decomp`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_num_lifted_params_buffers`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_override_composite_implicit_decomp`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_remove_unnecessary_copy_op_pass`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_split_decomp_table_to_cia_and_python_decomp`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_train`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_transform_do_not_use`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_update`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_validate`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`buffers`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`call_spec`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`constants`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`default_decompositions`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`dialect`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`example_inputs`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`graph`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`graph_module`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`graph_signature`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`module`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`module_call_graph`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`named_buffers`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`named_parameters`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`parameters`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`range_constraints`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`replace_all_uses_with`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`run_decompositions`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`state_dict`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`tensor_constants`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`update_arg`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`validate`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`verifier`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`verifiers`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`wrapper`**: [exported_program.py_docs.md](./exported_program.py_docs.md)

### Imports

- **`._unlift`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`.graph_signature`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`Any`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`Callable`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`CustomDecompTable`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`FakeScriptObject`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`FakeTensorMode`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`PassManager`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`PassResult`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`ShapeEnv`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`ValueRanges`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`Verifier`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_ConstantAttributeType`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_PyTreeCodeGen`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_check_input_constraints_for_graph`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_check_inputs_match`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_graph_output_names`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_ignore_backend_decomps`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_materialize_and_lift_constants`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`_unlift_exported_program_lifted_states`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`aot_export_module`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`autograd_not_implemented`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`collections`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`collections.abc`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`compatibility`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`contextlib`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`contextmanager`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`copy`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`dataclasses`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`defaultdict`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`detect_fake_mode`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`first_call_function_nn_module_stack`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`functools`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`immutable_dict`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`insert_deferred_runtime_asserts`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`is_equivalent`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`operator`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`sympy`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`torch`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`torch._export.non_strict_utils`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`torch._export.passes._node_metadata_hook`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`torch._export.passes.lift_constants_pass`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`torch._export.utils`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`torch._export.verifier`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`torch._functorch._aot_autograd.input_output_analysis`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`torch._functorch._aot_autograd.subclass_parametrization`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`torch._functorch.aot_autograd`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`torch._guards`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`torch._higher_order_ops.utils`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`torch._library.fake_class_registry`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`torch._subclasses.fake_impls`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`torch._subclasses.fake_tensor`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`torch.export._trace`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`torch.export._tree_utils`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`torch.export.decomp_utils`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`torch.fx._compatibility`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`torch.fx._symbolic_trace`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`torch.fx._utils`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`torch.fx.experimental.symbolic_shapes`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`torch.fx.graph`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`torch.fx.immutable_collections`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`torch.fx.passes.infra.pass_base`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`torch.fx.passes.infra.pass_manager`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`torch.fx.passes.runtime_assert`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`torch.utils._pytree`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`torch.utils._sympy.value_ranges`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`tracing`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`types`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`typing`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`unset_fake_temporarily`**: [exported_program.py_docs.md](./exported_program.py_docs.md)
- **`warnings`**: [exported_program.py_docs.md](./exported_program.py_docs.md)


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
