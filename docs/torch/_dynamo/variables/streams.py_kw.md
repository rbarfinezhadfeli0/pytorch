# Keyword Index: `torch/_dynamo/variables/streams.py`

## File Information

- **Original File**: [torch/_dynamo/variables/streams.py](../../../../torch/_dynamo/variables/streams.py)
- **Documentation**: [`streams.py_docs.md`](./streams.py_docs.md)
- **Folder**: `torch/_dynamo/variables`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`EventVariable`**: [streams.py_docs.md](./streams.py_docs.md)
- **`StreamContextVariable`**: [streams.py_docs.md](./streams.py_docs.md)
- **`StreamVariable`**: [streams.py_docs.md](./streams.py_docs.md)
- **`SymbolicStreamState`**: [streams.py_docs.md](./streams.py_docs.md)

### Functions

- **`_`**: [streams.py_docs.md](./streams.py_docs.md)
- **`__init__`**: [streams.py_docs.md](./streams.py_docs.md)
- **`_get_event_by_index`**: [streams.py_docs.md](./streams.py_docs.md)
- **`_get_stream_arg`**: [streams.py_docs.md](./streams.py_docs.md)
- **`_get_stream_by_index`**: [streams.py_docs.md](./streams.py_docs.md)
- **`as_proxy`**: [streams.py_docs.md](./streams.py_docs.md)
- **`call_method`**: [streams.py_docs.md](./streams.py_docs.md)
- **`create`**: [streams.py_docs.md](./streams.py_docs.md)
- **`cur_stream`**: [streams.py_docs.md](./streams.py_docs.md)
- **`enter`**: [streams.py_docs.md](./streams.py_docs.md)
- **`enter_stream`**: [streams.py_docs.md](./streams.py_docs.md)
- **`exit`**: [streams.py_docs.md](./streams.py_docs.md)
- **`exit_stream`**: [streams.py_docs.md](./streams.py_docs.md)
- **`fn`**: [streams.py_docs.md](./streams.py_docs.md)
- **`fn_name`**: [streams.py_docs.md](./streams.py_docs.md)
- **`fork_stream`**: [streams.py_docs.md](./streams.py_docs.md)
- **`get_current_stream`**: [streams.py_docs.md](./streams.py_docs.md)
- **`get_stream`**: [streams.py_docs.md](./streams.py_docs.md)
- **`in_stream_context`**: [streams.py_docs.md](./streams.py_docs.md)
- **`join_stream`**: [streams.py_docs.md](./streams.py_docs.md)
- **`make_construct_in_graph_event_fn`**: [streams.py_docs.md](./streams.py_docs.md)
- **`make_construct_in_graph_stream_fn`**: [streams.py_docs.md](./streams.py_docs.md)
- **`module_name`**: [streams.py_docs.md](./streams.py_docs.md)
- **`new_event`**: [streams.py_docs.md](./streams.py_docs.md)
- **`new_stream`**: [streams.py_docs.md](./streams.py_docs.md)
- **`python_type`**: [streams.py_docs.md](./streams.py_docs.md)
- **`reconstruct`**: [streams.py_docs.md](./streams.py_docs.md)
- **`record_event`**: [streams.py_docs.md](./streams.py_docs.md)
- **`supports_graph_breaks`**: [streams.py_docs.md](./streams.py_docs.md)
- **`wait_event`**: [streams.py_docs.md](./streams.py_docs.md)
- **`wait_stream`**: [streams.py_docs.md](./streams.py_docs.md)

### Imports

- **`..`**: [streams.py_docs.md](./streams.py_docs.md)
- **`..bytecode_transformation`**: [streams.py_docs.md](./streams.py_docs.md)
- **`..codegen`**: [streams.py_docs.md](./streams.py_docs.md)
- **`..exc`**: [streams.py_docs.md](./streams.py_docs.md)
- **`..graph_bytecode_inputs`**: [streams.py_docs.md](./streams.py_docs.md)
- **`..guards`**: [streams.py_docs.md](./streams.py_docs.md)
- **`..source`**: [streams.py_docs.md](./streams.py_docs.md)
- **`..utils`**: [streams.py_docs.md](./streams.py_docs.md)
- **`.base`**: [streams.py_docs.md](./streams.py_docs.md)
- **`.builder`**: [streams.py_docs.md](./streams.py_docs.md)
- **`.constant`**: [streams.py_docs.md](./streams.py_docs.md)
- **`.ctx_manager`**: [streams.py_docs.md](./streams.py_docs.md)
- **`.lazy`**: [streams.py_docs.md](./streams.py_docs.md)
- **`Any`**: [streams.py_docs.md](./streams.py_docs.md)
- **`Callable`**: [streams.py_docs.md](./streams.py_docs.md)
- **`ConstDictVariable`**: [streams.py_docs.md](./streams.py_docs.md)
- **`ConstantVariable`**: [streams.py_docs.md](./streams.py_docs.md)
- **`CurrentStreamSource`**: [streams.py_docs.md](./streams.py_docs.md)
- **`FxTracebackAnnotateVariable`**: [streams.py_docs.md](./streams.py_docs.md)
- **`GuardBuilder`**: [streams.py_docs.md](./streams.py_docs.md)
- **`InstructionTranslator`**: [streams.py_docs.md](./streams.py_docs.md)
- **`LazyVariableTracker`**: [streams.py_docs.md](./streams.py_docs.md)
- **`PyCodegen`**: [streams.py_docs.md](./streams.py_docs.md)
- **`TYPE_CHECKING`**: [streams.py_docs.md](./streams.py_docs.md)
- **`TupleVariable`**: [streams.py_docs.md](./streams.py_docs.md)
- **`VariableTracker`**: [streams.py_docs.md](./streams.py_docs.md)
- **`cmp_name_to_op_mapping`**: [streams.py_docs.md](./streams.py_docs.md)
- **`collections`**: [streams.py_docs.md](./streams.py_docs.md)
- **`collections.abc`**: [streams.py_docs.md](./streams.py_docs.md)
- **`create_call_function`**: [streams.py_docs.md](./streams.py_docs.md)
- **`custom_op`**: [streams.py_docs.md](./streams.py_docs.md)
- **`graph_break_hints`**: [streams.py_docs.md](./streams.py_docs.md)
- **`has_side_effect`**: [streams.py_docs.md](./streams.py_docs.md)
- **`proxy_args_kwargs`**: [streams.py_docs.md](./streams.py_docs.md)
- **`torch`**: [streams.py_docs.md](./streams.py_docs.md)
- **`torch._dynamo.symbolic_convert`**: [streams.py_docs.md](./streams.py_docs.md)
- **`torch._dynamo.variables.dicts`**: [streams.py_docs.md](./streams.py_docs.md)
- **`torch._dynamo.variables.lists`**: [streams.py_docs.md](./streams.py_docs.md)
- **`torch._library.custom_ops`**: [streams.py_docs.md](./streams.py_docs.md)
- **`torch.fx`**: [streams.py_docs.md](./streams.py_docs.md)
- **`typing`**: [streams.py_docs.md](./streams.py_docs.md)
- **`wrap_fx_proxy_cls`**: [streams.py_docs.md](./streams.py_docs.md)


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
