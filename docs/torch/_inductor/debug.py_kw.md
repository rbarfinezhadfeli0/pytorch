# Keyword Index: `torch/_inductor/debug.py`

## File Information

- **Original File**: [torch/_inductor/debug.py](../../../torch/_inductor/debug.py)
- **Documentation**: [`debug.py_docs.md`](./debug.py_docs.md)
- **Folder**: `torch/_inductor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`DebugContext`**: [debug.py_docs.md](./debug.py_docs.md)
- **`DebugFormatter`**: [debug.py_docs.md](./debug.py_docs.md)
- **`class`**: [debug.py_docs.md](./debug.py_docs.md)

### Functions

- **`__enter__`**: [debug.py_docs.md](./debug.py_docs.md)
- **`__exit__`**: [debug.py_docs.md](./debug.py_docs.md)
- **`__getattr__`**: [debug.py_docs.md](./debug.py_docs.md)
- **`__init__`**: [debug.py_docs.md](./debug.py_docs.md)
- **`_dump_collective_schedule`**: [debug.py_docs.md](./debug.py_docs.md)
- **`_save_profile_data`**: [debug.py_docs.md](./debug.py_docs.md)
- **`_setup_log_capture`**: [debug.py_docs.md](./debug.py_docs.md)
- **`_write_ir`**: [debug.py_docs.md](./debug.py_docs.md)
- **`annotate_orig_fx_with_snodes`**: [debug.py_docs.md](./debug.py_docs.md)
- **`aot_inductor_minifier_wrapper`**: [debug.py_docs.md](./debug.py_docs.md)
- **`build_node_info`**: [debug.py_docs.md](./debug.py_docs.md)
- **`check_format`**: [debug.py_docs.md](./debug.py_docs.md)
- **`convert_sets_to_lists`**: [debug.py_docs.md](./debug.py_docs.md)
- **`copy`**: [debug.py_docs.md](./debug.py_docs.md)
- **`create_debug_dir`**: [debug.py_docs.md](./debug.py_docs.md)
- **`create_fx_from_snodes`**: [debug.py_docs.md](./debug.py_docs.md)
- **`create_kernel_information_json`**: [debug.py_docs.md](./debug.py_docs.md)
- **`create_mapping_pre_post_grad_nodes`**: [debug.py_docs.md](./debug.py_docs.md)
- **`create_node_mapping_kernel_to_post_grad`**: [debug.py_docs.md](./debug.py_docs.md)
- **`draw_buffers`**: [debug.py_docs.md](./debug.py_docs.md)
- **`draw_orig_fx_graph`**: [debug.py_docs.md](./debug.py_docs.md)
- **`dtype_to_str`**: [debug.py_docs.md](./debug.py_docs.md)
- **`dump_inductor_provenance_info`**: [debug.py_docs.md](./debug.py_docs.md)
- **`enable_aot_logging`**: [debug.py_docs.md](./debug.py_docs.md)
- **`filename`**: [debug.py_docs.md](./debug.py_docs.md)
- **`fopen`**: [debug.py_docs.md](./debug.py_docs.md)
- **`fopen_context`**: [debug.py_docs.md](./debug.py_docs.md)
- **`func1`**: [debug.py_docs.md](./debug.py_docs.md)
- **`fx_graph`**: [debug.py_docs.md](./debug.py_docs.md)
- **`fx_graph_transformed`**: [debug.py_docs.md](./debug.py_docs.md)
- **`get_fake_func`**: [debug.py_docs.md](./debug.py_docs.md)
- **`get_node_name_to_buf_meta`**: [debug.py_docs.md](./debug.py_docs.md)
- **`graph_diagram`**: [debug.py_docs.md](./debug.py_docs.md)
- **`handle_tensor`**: [debug.py_docs.md](./debug.py_docs.md)
- **`has_dot`**: [debug.py_docs.md](./debug.py_docs.md)
- **`ignored`**: [debug.py_docs.md](./debug.py_docs.md)
- **`in_output`**: [debug.py_docs.md](./debug.py_docs.md)
- **`ir_post_fusion`**: [debug.py_docs.md](./debug.py_docs.md)
- **`ir_pre_fusion`**: [debug.py_docs.md](./debug.py_docs.md)
- **`load_args_and_run_compile_fx_inner`**: [debug.py_docs.md](./debug.py_docs.md)
- **`log_autotuning_results`**: [debug.py_docs.md](./debug.py_docs.md)
- **`log_collective_schedule`**: [debug.py_docs.md](./debug.py_docs.md)
- **`log_graph_execution`**: [debug.py_docs.md](./debug.py_docs.md)
- **`log_ir_post_fusion`**: [debug.py_docs.md](./debug.py_docs.md)
- **`log_ir_pre_fusion`**: [debug.py_docs.md](./debug.py_docs.md)
- **`log_runtime_and_tensor_meta`**: [debug.py_docs.md](./debug.py_docs.md)
- **`output_code`**: [debug.py_docs.md](./debug.py_docs.md)
- **`record_and_log_graph_execution_order`**: [debug.py_docs.md](./debug.py_docs.md)
- **`reset_inductor_kernel_provenance_debug_handle`**: [debug.py_docs.md](./debug.py_docs.md)
- **`reset_log_level`**: [debug.py_docs.md](./debug.py_docs.md)
- **`reset_provenance_globals`**: [debug.py_docs.md](./debug.py_docs.md)
- **`save_args_for_compile_fx_inner`**: [debug.py_docs.md](./debug.py_docs.md)
- **`set_kernel_post_grad_provenance_tracing`**: [debug.py_docs.md](./debug.py_docs.md)
- **`to_list`**: [debug.py_docs.md](./debug.py_docs.md)
- **`update_orig_fx_node_name_to_buf_name`**: [debug.py_docs.md](./debug.py_docs.md)
- **`upload_tar`**: [debug.py_docs.md](./debug.py_docs.md)

### Imports

- **`.`**: [debug.py_docs.md](./debug.py_docs.md)
- **`.codegen.simd_kernel_features`**: [debug.py_docs.md](./debug.py_docs.md)
- **`.ir`**: [debug.py_docs.md](./debug.py_docs.md)
- **`.scheduler`**: [debug.py_docs.md](./debug.py_docs.md)
- **`.virtualized`**: [debug.py_docs.md](./debug.py_docs.md)
- **`AccuracyError`**: [debug.py_docs.md](./debug.py_docs.md)
- **`Any`**: [debug.py_docs.md](./debug.py_docs.md)
- **`Callable`**: [debug.py_docs.md](./debug.py_docs.md)
- **`DisableReduction`**: [debug.py_docs.md](./debug.py_docs.md)
- **`ExternKernel`**: [debug.py_docs.md](./debug.py_docs.md)
- **`FileLike`**: [debug.py_docs.md](./debug.py_docs.md)
- **`FileLock`**: [debug.py_docs.md](./debug.py_docs.md)
- **`FixedLayout`**: [debug.py_docs.md](./debug.py_docs.md)
- **`GraphModule`**: [debug.py_docs.md](./debug.py_docs.md)
- **`OrderedSet`**: [debug.py_docs.md](./debug.py_docs.md)
- **`V`**: [debug.py_docs.md](./debug.py_docs.md)
- **`_aoti_flatten_inputs`**: [debug.py_docs.md](./debug.py_docs.md)
- **`_extract_tensor_metadata`**: [debug.py_docs.md](./debug.py_docs.md)
- **`collections`**: [debug.py_docs.md](./debug.py_docs.md)
- **`collections.abc`**: [debug.py_docs.md](./debug.py_docs.md)
- **`compile_fx_inner`**: [debug.py_docs.md](./debug.py_docs.md)
- **`config`**: [debug.py_docs.md](./debug.py_docs.md)
- **`contextlib`**: [debug.py_docs.md](./debug.py_docs.md)
- **`copy`**: [debug.py_docs.md](./debug.py_docs.md)
- **`dataclasses`**: [debug.py_docs.md](./debug.py_docs.md)
- **`draw_graph`**: [debug.py_docs.md](./debug.py_docs.md)
- **`dump_to_minify`**: [debug.py_docs.md](./debug.py_docs.md)
- **`filelock`**: [debug.py_docs.md](./debug.py_docs.md)
- **`functools`**: [debug.py_docs.md](./debug.py_docs.md)
- **`functorch.compile`**: [debug.py_docs.md](./debug.py_docs.md)
- **`fx`**: [debug.py_docs.md](./debug.py_docs.md)
- **`getArtifactLogger`**: [debug.py_docs.md](./debug.py_docs.md)
- **`get_debug_dir`**: [debug.py_docs.md](./debug.py_docs.md)
- **`io`**: [debug.py_docs.md](./debug.py_docs.md)
- **`itertools`**: [debug.py_docs.md](./debug.py_docs.md)
- **`json`**: [debug.py_docs.md](./debug.py_docs.md)
- **`legalize_graph`**: [debug.py_docs.md](./debug.py_docs.md)
- **`load_args_and_run_compile_fx_inner`**: [debug.py_docs.md](./debug.py_docs.md)
- **`logging`**: [debug.py_docs.md](./debug.py_docs.md)
- **`os`**: [debug.py_docs.md](./debug.py_docs.md)
- **`os.path`**: [debug.py_docs.md](./debug.py_docs.md)
- **`patch`**: [debug.py_docs.md](./debug.py_docs.md)
- **`pickle`**: [debug.py_docs.md](./debug.py_docs.md)
- **`pstats`**: [debug.py_docs.md](./debug.py_docs.md)
- **`save_graph_repro`**: [debug.py_docs.md](./debug.py_docs.md)
- **`shutil`**: [debug.py_docs.md](./debug.py_docs.md)
- **`signpost_event`**: [debug.py_docs.md](./debug.py_docs.md)
- **`tarfile`**: [debug.py_docs.md](./debug.py_docs.md)
- **`torch`**: [debug.py_docs.md](./debug.py_docs.md)
- **`torch._dynamo.debug_utils`**: [debug.py_docs.md](./debug.py_docs.md)
- **`torch._dynamo.repro.after_aot`**: [debug.py_docs.md](./debug.py_docs.md)
- **`torch._dynamo.repro.aoti`**: [debug.py_docs.md](./debug.py_docs.md)
- **`torch._dynamo.utils`**: [debug.py_docs.md](./debug.py_docs.md)
- **`torch._functorch.aot_autograd`**: [debug.py_docs.md](./debug.py_docs.md)
- **`torch._inductor`**: [debug.py_docs.md](./debug.py_docs.md)
- **`torch._inductor.compile_fx`**: [debug.py_docs.md](./debug.py_docs.md)
- **`torch._inductor.debug`**: [debug.py_docs.md](./debug.py_docs.md)
- **`torch._logging`**: [debug.py_docs.md](./debug.py_docs.md)
- **`torch._logging._internal`**: [debug.py_docs.md](./debug.py_docs.md)
- **`torch._utils_internal`**: [debug.py_docs.md](./debug.py_docs.md)
- **`torch.fx.graph_module`**: [debug.py_docs.md](./debug.py_docs.md)
- **`torch.fx.passes.shape_prop`**: [debug.py_docs.md](./debug.py_docs.md)
- **`torch.fx.passes.tools_common`**: [debug.py_docs.md](./debug.py_docs.md)
- **`torch.types`**: [debug.py_docs.md](./debug.py_docs.md)
- **`torch.utils._ordered_set`**: [debug.py_docs.md](./debug.py_docs.md)
- **`torch.utils._pytree`**: [debug.py_docs.md](./debug.py_docs.md)
- **`trace_structured`**: [debug.py_docs.md](./debug.py_docs.md)
- **`traceback`**: [debug.py_docs.md](./debug.py_docs.md)
- **`tree_map`**: [debug.py_docs.md](./debug.py_docs.md)
- **`typing`**: [debug.py_docs.md](./debug.py_docs.md)
- **`unittest.mock`**: [debug.py_docs.md](./debug.py_docs.md)
- **`utils`**: [debug.py_docs.md](./debug.py_docs.md)


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
