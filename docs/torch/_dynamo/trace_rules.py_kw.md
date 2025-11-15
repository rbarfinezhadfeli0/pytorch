# Keyword Index: `torch/_dynamo/trace_rules.py`

## File Information

- **Original File**: [torch/_dynamo/trace_rules.py](../../../torch/_dynamo/trace_rules.py)
- **Documentation**: [`trace_rules.py_docs.md`](./trace_rules.py_docs.md)
- **Folder**: `torch/_dynamo`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`FunctionIdSet`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`class`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)

### Functions

- **`__call__`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`__contains__`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`__init__`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`_allowed_callable_ids`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`_as_posix_path`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`_builtin_constant_ids`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`_builtin_function_ids`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`_disallowed_callable_ids`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`_load_obj_from_str`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`_lookup_inner`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`_maybe_init_lazy_module`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`_module_dir`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`_nonstrict_trace_callable_ids`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`_numpy_function_ids`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`_polyfilled_function_ids`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`_recompile_re`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`_strip_init_py`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`add`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`add_module_init_func`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`check`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`check_file`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`check_verbose`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`clear_lru_cache`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`dynamo_dir`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`f1`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`f2`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`f3`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`get_legacy_mod_inlinelist`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`get_mod_inlinelist`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`get_mod_skiplist`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`get_name`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`get_tensor_method`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`get_torch_obj_rule_map`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`is_aten_op_or_tensor_method`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`is_builtin_callable`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`is_builtin_constant`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`is_callable_allowed`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`is_callable_disallowed`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`is_forbidden`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`is_nonstrict_trace_callable`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`is_numpy`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`is_numpy_dtype`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`is_numpy_type_info`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`is_polyfilled_callable`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`is_supported`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`is_torch`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`is_torch_inline_allowed`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`load_object`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`lookup`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`lookup_callable`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`lookup_inner`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`remove`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)

### Imports

- **`.`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`.resume_execution`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`.utils`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`.variables`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`.variables.base`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`Any`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`Callable`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`Path`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`TORCH_DYNAMO_RESUME_IN_PREFIX`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`VariableTracker`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`_config_module`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`abc`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`builtins`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`collections`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`collections.abc`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`config`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`copy`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`dataclasses`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`defaultdict`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`find_spec`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`functools`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`importlib`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`importlib.util`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`inspect`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`is_fbcode`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`linecache`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`numpy`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`operator`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`os`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`pathlib`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`random`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`re`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`sys`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`torch`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`torch._dynamo`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`torch._environment`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`torch._inductor.test_operators`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`torch.distributed`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`torch.utils`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`torch.utils._content_store`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`traceback`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`types`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`typing`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)
- **`unittest`**: [trace_rules.py_docs.md](./trace_rules.py_docs.md)


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
