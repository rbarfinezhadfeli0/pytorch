# Keyword Index: `torch/_dynamo/compiled_autograd.py`

## File Information

- **Original File**: [torch/_dynamo/compiled_autograd.py](../../../torch/_dynamo/compiled_autograd.py)
- **Documentation**: [`compiled_autograd.py_docs.md`](./compiled_autograd.py_docs.md)
- **Folder**: `torch/_dynamo`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`AutogradCompilerInstance`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`NaNChecker`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`Op`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`OpNamespace`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`to`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)

### Functions

- **`__call__`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`__init__`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`__repr__`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`_disable`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`_enable`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`accumulate`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`accumulate_grad`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`add`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`allocate_dummy`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`apply_functional`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`begin_capture`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`bind_backward_state`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`bind_function`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`bind_objects_to_proxies`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`call_aot_bwd_prologue`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`call_copy_slices_epilogue`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`call_copy_slices_prologue`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`check`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`copy_paste_aot_backward_graph`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`copy_slices_epilogue`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`copy_slices_prologue`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`cpp_tensor_pre_hook`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`create_graph_module`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`dce`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`delay_unpack_hook_nodes`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`dummy`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`end_capture`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`extract_bw_module`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`get`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`get_all_nodes`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`is_impure`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`is_placeholder`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`is_sym_node`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`log_compile_reasons`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`make_compile_context`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`make_subclass`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`make_unique`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`maybe_clone`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`move_graph_nodes_to_cuda`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`num_inputs`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`post_acc_grad_hook`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`post_hook`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`pre_hook`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`prep_with_graph`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`prep_with_inputs`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`proxy_call`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`proxy_call_aot_backward`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`proxy_call_backward`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`proxy_call_hook`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`proxy_subclass_constructor`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`remove_unused_sizes`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`reorder_accumulate_grad_nodes`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`reorder_post_acc_grad_hook_nodes`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`reorder_post_hook_nodes`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`reorder_pre_hook_nodes_to_mimic_eager`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`reorder_pre_hook_nodes_to_schedule_asap`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`reorder_tensor_pre_hook_nodes`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`reset`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`run_non_traceable_cpp_in_eager`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`runtime_wrapper`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`set_node_origin`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`snapshot_cudagraph_enabled`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`snapshot_verbose_logging_enabled`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`source`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`tensor_pre_hook`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`to_proxy`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`train`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`unpack_hook`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`validate_outputs`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`wrap_fake`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)

### Imports

- **`Any`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`BackwardState`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`Callable`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`CapturedTraceback`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`Counter`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`DimDynamic`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`FakeTensor`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`FakeTensorMode`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`FloatLikeType`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`GetItemSource`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`GraphModule`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`OrderedSet`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`Proxy`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`clone_preserve_strides`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`collections`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`collections.abc`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`compile_context`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`contextlib`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`cudagraph_trees`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`enable_python_dispatcher`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`eval_frame`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`functools`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`getArtifactLogger`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`it`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`itertools`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`operator`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`preserve_node_meta`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`this`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`time`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`torch`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`torch._dispatch.python`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`torch._dynamo`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`torch._dynamo.external_utils`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`torch._dynamo.source`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`torch._dynamo.utils`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`torch._functorch._aot_autograd.runtime_wrappers`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`torch._guards`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`torch._inductor`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`torch._logging`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`torch._prims_common`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`torch._subclasses`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`torch._subclasses.fake_tensor`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`torch.fx`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`torch.fx.experimental._backward_state`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`torch.fx.experimental.proxy_tensor`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`torch.fx.experimental.symbolic_shapes`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`torch.fx.proxy`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`torch.fx.traceback`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`torch.types`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`torch.utils._ordered_set`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`torch.utils._pytree`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`torch.utils._traceback`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)
- **`typing`**: [compiled_autograd.py_docs.md](./compiled_autograd.py_docs.md)


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
