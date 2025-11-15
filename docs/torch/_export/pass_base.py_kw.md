# Keyword Index: `torch/_export/pass_base.py`

## File Information

- **Original File**: [torch/_export/pass_base.py](../../../torch/_export/pass_base.py)
- **Documentation**: [`pass_base.py_docs.md`](./pass_base.py_docs.md)
- **Folder**: `torch/_export`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`ExportInterpreter`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`ExportPassBaseError`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`ExportTracer`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`_ExportPassBaseDeprecatedDoNotUse`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`error`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`is`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`to`**: [pass_base.py_docs.md](./pass_base.py_docs.md)

### Functions

- **`__init__`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`_create_dummy_node_metadata`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`_fx`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`call`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`call_cond`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`call_function`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`call_getitem`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`call_map`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`call_method`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`call_module`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`call_operator`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`call_submodule`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`call_sym`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`create_arg`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`extract_input`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`get_attr`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`inputs`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`make_tensor_meta`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`make_val`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`on_attr`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`output`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`placeholder`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`run_node`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`set_metadata`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`trace`**: [pass_base.py_docs.md](./pass_base.py_docs.md)

### Imports

- **`Any`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`Callable`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`CodeGen`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`FakeTensor`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`FakeTensorMode`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`NodeMetadata`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`PassBase`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`ProxyValue`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`PythonKeyTracer`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`_extract_tensor_metadata`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`_pytree`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`_unstack_pytree`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`collections.abc`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`contextlib`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`enable_python_dispatcher`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`fx`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`nullcontext`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`operator`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`torch`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`torch._dispatch.python`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`torch._export.pass_infra.node_metadata`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`torch._export.pass_infra.proxy_value`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`torch._higher_order_ops.map`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`torch._subclasses`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`torch._subclasses.fake_tensor`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`torch.fx`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`torch.fx.experimental.proxy_tensor`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`torch.fx.experimental.symbolic_shapes`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`torch.fx.graph`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`torch.fx.passes.infra.pass_base`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`torch.fx.passes.shape_prop`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`torch.utils`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`traceback`**: [pass_base.py_docs.md](./pass_base.py_docs.md)
- **`typing`**: [pass_base.py_docs.md](./pass_base.py_docs.md)


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
