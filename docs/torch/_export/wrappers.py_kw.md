# Keyword Index: `torch/_export/wrappers.py`

## File Information

- **Original File**: [torch/_export/wrappers.py](../../../torch/_export/wrappers.py)
- **Documentation**: [`wrappers.py_docs.md`](./wrappers.py_docs.md)
- **Folder**: `torch/_export`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`ExportTracepoint`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`FooTensor`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`MyCoolCustomAutogradFunc`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`constructors`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`instance`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`spec`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`tensor`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`to`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`traceble`**: [wrappers.py_docs.md](./wrappers.py_docs.md)

### Functions

- **`__call__`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`__init__`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`__new__`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`_emit_flat_apply_call`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`_is_init`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`_mark_strict_experimental`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`_register_func_spec_proxy_in_tracer`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`_wrap_submodule`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`_wrap_submodules`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`allow_in_pre_dispatch_graph`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`apply`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`call`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`check_flattened`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`export_tracepoint_cpu`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`export_tracepoint_dispatch_mode`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`export_tracepoint_fake_tensor_mode`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`export_tracepoint_functional`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`mark_subclass_constructor_exportable_experimental`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`post_hook`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`pre_hook`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`update_module_call_signatures`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`wrapper`**: [wrappers.py_docs.md](./wrappers.py_docs.md)

### Imports

- **`DispatchKey`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`FakeTensorMode`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`HigherOrderOperator`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`_get_dispatch_mode_pre_dispatch`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`_maybe_find_pre_dispatch_tf_mode_for_export`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`_pytree`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`autograd_not_implemented`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`contextlib`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`contextmanager`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`functools`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`inspect`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`is_traceable_wrapper_subclass_type`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`strict_mode`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`torch`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`torch._C`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`torch._custom_ops`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`torch._export.utils`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`torch._higher_order_ops.flat_apply`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`torch._higher_order_ops.strict_mode`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`torch._higher_order_ops.utils`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`torch._ops`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`torch._subclasses.fake_tensor`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`torch.export.custom_ops`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`torch.fx.experimental.proxy_tensor`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`torch.utils`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`torch.utils._python_dispatch`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`wraps`**: [wrappers.py_docs.md](./wrappers.py_docs.md)


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
