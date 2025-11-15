# Keyword Index: `torch/_dynamo/_trace_wrapped_higher_order_op.py`

## File Information

- **Original File**: [torch/_dynamo/_trace_wrapped_higher_order_op.py](../../../torch/_dynamo/_trace_wrapped_higher_order_op.py)
- **Documentation**: [`_trace_wrapped_higher_order_op.py_docs.md`](./_trace_wrapped_higher_order_op.py_docs.md)
- **Folder**: `torch/_dynamo`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`ModIndex`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`TraceWrapped`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`TransformGetItemToIndex`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)

### Functions

- **`_`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`__call__`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`__init__`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`__torch_function__`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`_assert_meta`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`_trace_wrapped_functionalized`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`_trace_wrapped_op_dense`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`apply`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`autograd_function_backward_rewritten`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`backward`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`forward`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`inner_fake`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`inner_trace`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`new_backward`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`self_invoke`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`setup_context`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`trace_wrapped`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`unwrap_proxies`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`zeros_and_scatter`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)

### Imports

- **`Any`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`BackwardState`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`DispatchKey`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`FakeTensorMode`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`HigherOrderOperator`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`ProxyTorchDispatchMode`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`TorchFunctionMode`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`_get_current_dispatch_mode`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`autograd_not_implemented`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`torch`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`torch._C`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`torch._higher_order_ops.utils`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`torch._ops`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`torch._subclasses`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`torch.fx.experimental._backward_state`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`torch.fx.experimental.proxy_tensor`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`torch.overrides`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`torch.utils._python_dispatch`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`torch.utils._pytree`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`tree_map_only`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)
- **`typing`**: [_trace_wrapped_higher_order_op.py_docs.md](./_trace_wrapped_higher_order_op.py_docs.md)


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
