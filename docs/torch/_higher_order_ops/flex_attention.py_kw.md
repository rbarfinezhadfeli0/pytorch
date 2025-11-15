# Keyword Index: `torch/_higher_order_ops/flex_attention.py`

## File Information

- **Original File**: [torch/_higher_order_ops/flex_attention.py](../../../torch/_higher_order_ops/flex_attention.py)
- **Documentation**: [`flex_attention.py_docs.md`](./flex_attention.py_docs.md)
- **Folder**: `torch/_higher_order_ops`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`FlexAttentionAutogradOp`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`FlexAttentionBackwardHOP`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`FlexAttentionHOP`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)

### Functions

- **`__call__`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`__init__`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_construct_strides`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_from_fun`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_math_attention_inner`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_maybe_new_buffer`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_permute_strides`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`backward`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`create_fw_bw_graph`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`flex_attention_autograd`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`flex_attention_backward_fake_tensor_mode`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`flex_attention_backward_functionalize`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`flex_attention_backward_proxy_torch_dispatch_mode`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`flex_attention_fake_impl`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`flex_attention_functionalize`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`flex_attention_proxy_torch_dispatch_mode`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`forward`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`fw_with_masks`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`joint_f`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`math_attention`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`sdpa_dense`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`sdpa_dense_backward`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`trace_flex_attention`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`trace_flex_attention_backward`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)

### Imports

- **`AOTConfig`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`Any`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`Callable`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`DispatchKey`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`FakeTensor`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`FunctionalTensor`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`GraphModule`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`HigherOrderOperator`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`Tensor`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`TransformGetItemToIndex`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_CachedTorchDispatchMode`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_vmap_for_bhqkv`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`collections.abc`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`def`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`detect_fake_mode`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`disable_functional_mode`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`disable_proxy_modes_tracing`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`get_fill_order`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`math`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`suspend_functionalization`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`torch`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`torch._C`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`torch._dispatch.python`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`torch._dynamo._trace_wrapped_higher_order_op`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`torch._functorch.aot_autograd`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`torch._guards`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`torch._higher_order_ops.utils`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`torch._inductor.ir`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`torch._ops`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`torch._subclasses`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`torch._subclasses.fake_tensor`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`torch._subclasses.functional_tensor`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`torch.fx.experimental.proxy_tensor`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`torch.fx.graph_module`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`torch.nn.attention.flex_attention`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`torch.utils._pytree`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`torch.utils.checkpoint`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`typing`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)


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
