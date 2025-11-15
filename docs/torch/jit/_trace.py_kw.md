# Keyword Index: `torch/jit/_trace.py`

## File Information

- **Original File**: [torch/jit/_trace.py](../../../torch/jit/_trace.py)
- **Documentation**: [`_trace.py_docs.md`](./_trace.py_docs.md)
- **Folder**: `torch/jit`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`Net`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`ONNXTracedModule`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`QualnameWrapper`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`TopLevelTracedModule`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`TracedModule`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`TracerWarning`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`TracingCheckError`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`_ExportOutcome`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`_ExportType`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`is`**: [_trace.py_docs.md](./_trace.py_docs.md)

### Functions

- **`__getattr__`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`__init__`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`__setattr__`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`__str__`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`_check_trace`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`_clone_inputs`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`_create_interpreter_name_lookup_fn`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`_get_interpreter_name_for_var`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`_get_name`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`_get_trace_graph`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`_reconstruct`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`_script_if_tracing`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`_time`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`_trace_impl`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`_unique_state_dict`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`_verify_equal`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`analyze_ts_result_with_export_result`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`check_unique`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`clone_input`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`compare_outputs`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`extra_repr`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`foo`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`forward`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`graph_diagnostic_info`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`ignore_lib_warnings`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`indent`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`is_tracing`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`make_module`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`make_tuple`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`maybe_warn_nondeterministic`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`register_submods`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`run_fwd_bwd`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`run_mod_and_filter_tensor_outputs`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`trace`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`trace_module`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`verify`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`weighted_kernel_sum`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`wrap_check_inputs`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`wrap_retval`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`wrapper`**: [_trace.py_docs.md](./_trace.py_docs.md)

### Imports

- **`Any`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`Callable`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`Enum`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`Module`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`ParamSpec`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`_CachedForward`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`_enabled`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`collections.abc`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`contextlib`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`copy`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`default_tolerances`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`difflib`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`enum`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`function`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`functools`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`inspect`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`log_torchscript_usage`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`os`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`re`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`sys`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`torch`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`torch._jit_internal`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`torch._utils_internal`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`torch.autograd`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`torch.jit._script`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`torch.jit._state`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`torch.nn`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`torch.testing._comparison`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`torch.utils._pytree`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`typing`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`typing_extensions`**: [_trace.py_docs.md](./_trace.py_docs.md)
- **`warnings`**: [_trace.py_docs.md](./_trace.py_docs.md)


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
