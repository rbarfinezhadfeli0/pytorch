# Keyword Index: `torch/fx/proxy.py`

## File Information

- **Original File**: [torch/fx/proxy.py](../../../torch/fx/proxy.py)
- **Documentation**: [`proxy.py_docs.md`](./proxy.py_docs.md)
- **Folder**: `torch/fx`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`Attribute`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`GraphAppendingTracer`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`M`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`MetaProxy`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`ParameterProxy`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`Proxy`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`Scope`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`ScopeContextManager`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`Sub`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`TraceError`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`TracerBase`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`from`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`that`**: [proxy.py_docs.md](./proxy.py_docs.md)

### Functions

- **`__abs__`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`__bool__`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`__call__`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`__deepcopy__`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`__enter__`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`__exit__`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`__getattr__`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`__getstate__`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`__init__`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`__iter__`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`__len__`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`__repr__`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`__setstate__`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`__torch_function__`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`_create_arg_dict`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`_define_reflectable`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`_filter_traceback_frames`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`_find_user_frame`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`_no_nodes_error`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`_scope`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`assert_fn`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`create_arg`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`create_node`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`create_proxy`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`dim`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`find_tracer`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`forward`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`impl`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`iter`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`keys`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`ndim`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`nelement`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`node`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`numel`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`proxy`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`shape`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`size`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`to_bool`**: [proxy.py_docs.md](./proxy.py_docs.md)

### Imports

- **`._compatibility`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`.graph`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`.immutable_collections`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`.node`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`.operator_schemas`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`Any`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`Argument`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`Callable`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`CapturedTraceback`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`Graph`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`OrderedDict`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`_fx_map_aggregate`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`bisect`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`bisect_left`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`check_for_mutable_operation`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`collections`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`collections.abc`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`compatibility`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`copy`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`dataclasses`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`dis`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`enum`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`fields`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`getArtifactLogger`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`immutable_dict`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`inspect`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`logging`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`operator`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`sys`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`torch`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`torch._C`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`torch._logging`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`torch.fx.experimental.symbolic_shapes`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`torch.fx.traceback`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`torch.utils._traceback`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`traceback`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`typing`**: [proxy.py_docs.md](./proxy.py_docs.md)
- **`uninteresting_files`**: [proxy.py_docs.md](./proxy.py_docs.md)


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
