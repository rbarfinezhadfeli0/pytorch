# Keyword Index: `test/dynamo/test_backward_higher_order_ops.py`

## File Information

- **Original File**: [test/dynamo/test_backward_higher_order_ops.py](../../../test/dynamo/test_backward_higher_order_ops.py)
- **Documentation**: [`test_backward_higher_order_ops.py_docs.md`](./test_backward_higher_order_ops.py_docs.md)
- **Folder**: `test/dynamo`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`BackwardHigherOrderOpTests`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`GraphModule`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`MyObj`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`_multiply_invoke`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)

### Functions

- **`__init__`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`_graph_break_invoke`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`_graph_breaking_fn`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`_multiply`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`_multiply_invoke`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`_side_effect_stateful_fn2`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`_side_effectful_invoke2`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`compiler_fn`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`fn`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`forward`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`fwd`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`inner_compiler`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`test_invoke_in_eager`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`test_invoke_in_pt2`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`test_invoke_in_pt2_compiled_autograd`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`test_invoke_in_pt2_compiled_autograd_graph_breaks`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`test_invoke_in_pt2_compiled_autograd_side_effect`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`test_invoke_make_bw`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`test_invoke_make_fx_forward_contrived`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)

### Imports

- **`_inductor`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`compiled_autograd`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`functools`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`itertools`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`make_fx`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`mock`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`normalize_gm`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`run_tests`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`torch`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`torch._dynamo`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`torch._dynamo._trace_wrapped_higher_order_op`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`torch._dynamo.test_case`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`torch._dynamo.testing`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`torch._dynamo.utils`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`torch.fx.experimental.proxy_tensor`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`trace_wrapped`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)
- **`unittest`**: [test_backward_higher_order_ops.py_docs.md](./test_backward_higher_order_ops.py_docs.md)


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
