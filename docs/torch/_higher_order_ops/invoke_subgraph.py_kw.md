# Keyword Index: `torch/_higher_order_ops/invoke_subgraph.py`

## File Information

- **Original File**: [torch/_higher_order_ops/invoke_subgraph.py](../../../torch/_higher_order_ops/invoke_subgraph.py)
- **Documentation**: [`invoke_subgraph.py_docs.md`](./invoke_subgraph.py_docs.md)
- **Folder**: `torch/_higher_order_ops`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`InvokeSubgraphAutogradOp`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`InvokeSubgraphHOP`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`class`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)

### Functions

- **`_`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`__call__`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`__init__`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`_invoke_subgraph_placeholder_wrapper`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`_unwrap_proxy`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`autograd_fn_callable`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`backward`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`create_fw_bw_graph`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`forward`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`gen_schema`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`get_invoke_subgraph_cache`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`get_output_metadata`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`graph_with_interpreter`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`inner`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`invoke_subgraph_placeholder`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`joint_fn`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`mark_compile_region`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`trace_joint_graph`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`trace_joint_graph_as_bwd`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`wrap`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)

### Imports

- **`Any`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`Callable`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`DispatchKey`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`GraphModule`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`HigherOrderOperator`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`HopSchemaGenerator`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`_CacheKeyState`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`_get_current_dispatch_mode`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`collections.abc`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`contextlib`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`create_joint`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`dataclass`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`dataclasses`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`detect_fake_mode`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`disable_functional_mode`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`dynamo_timed`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`extract_tensor_metadata`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`insert_deferred_runtime_asserts`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`nullcontext`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`suspend_functionalization`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`torch`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`torch._C`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`torch._dispatch.python`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`torch._dynamo.backends.debugging`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`torch._dynamo.utils`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`torch._functorch.aot_autograd`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`torch._guards`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`torch._higher_order_ops.auto_functionalize`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`torch._higher_order_ops.schema`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`torch._higher_order_ops.utils`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`torch._ops`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`torch._subclasses._fake_tensor_utils`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`torch._subclasses.fake_tensor`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`torch._subclasses.functional_tensor`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`torch.fx.experimental.proxy_tensor`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`torch.fx.graph_module`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`torch.fx.passes.runtime_assert`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`torch.utils._python_dispatch`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`torch.utils._pytree`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)
- **`typing`**: [invoke_subgraph.py_docs.md](./invoke_subgraph.py_docs.md)


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
