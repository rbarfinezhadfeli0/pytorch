# Keyword Index: `torch/_dynamo/functional_export.py`

## File Information

- **Original File**: [torch/_dynamo/functional_export.py](../../../torch/_dynamo/functional_export.py)
- **Documentation**: [`functional_export.py_docs.md`](./functional_export.py_docs.md)
- **Folder**: `torch/_dynamo`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`DynamoGraphTransformer`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`InShuffle`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`ModuleToTrace`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`OutShuffle`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`PyTreeifyOutput`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`Yield`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`from`**: [functional_export.py_docs.md](./functional_export.py_docs.md)

### Functions

- **`__init__`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`_create_flattened_inputs`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`_create_placeholder_mapping`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`_dynamo_graph_capture_for_export`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`_normalize_shuffle_graph`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`_process_nn_module_stack`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`_process_source_fn`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`_suggest_or_raise_constraint_violation`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`backend_dummy`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`clean_export_root`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`clean_export_root_string`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`clean_nn_module_stack_and_source_fn`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`dynamo_graph_capture_for_export`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`forward`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`inner`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`normalize_graph_module`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`output`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`placeholder`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`post_process_error_msg`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`pytreeify`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`run_node`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`transform`**: [functional_export.py_docs.md](./functional_export.py_docs.md)

### Imports

- **`.`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`Any`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`Callable`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`CaptureOutput`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`FakeTensorMode`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`Node`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`TracingContext`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`TreeSpec`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`UserErrorType`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`_ExportCodeGen`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`_RelaxedConstraint`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`_compiling_state_context`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`_get_input_paths`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`argument_names`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`collections`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`collections.abc`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`dataclass`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`dataclasses`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`dynamo_timed`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`enable_python_dispatcher`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`inspect`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`logging`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`make_fx`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`namedtuple`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`reset`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`sympy`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`torch`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`torch._dispatch.python`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`torch._dynamo.convert_frame`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`torch._dynamo.eval_frame`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`torch._dynamo.exc`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`torch._dynamo.utils`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`torch._export.utils`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`torch._guards`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`torch._subclasses.fake_tensor`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`torch.export._unlift`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`torch.export.dynamic_shapes`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`torch.fx`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`torch.fx.experimental.proxy_tensor`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`torch.fx.experimental.symbolic_shapes`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`torch.fx.graph`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`torch.utils._pytree`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`traceback`**: [functional_export.py_docs.md](./functional_export.py_docs.md)
- **`typing`**: [functional_export.py_docs.md](./functional_export.py_docs.md)


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
