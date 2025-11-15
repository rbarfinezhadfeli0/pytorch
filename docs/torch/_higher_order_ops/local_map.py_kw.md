# Keyword Index: `torch/_higher_order_ops/local_map.py`

## File Information

- **Original File**: [torch/_higher_order_ops/local_map.py](../../../torch/_higher_order_ops/local_map.py)
- **Documentation**: [`local_map.py_docs.md`](./local_map.py_docs.md)
- **Folder**: `torch/_higher_order_ops`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`LocalMapAutogradOp`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`LocalMapHOP`**: [local_map.py_docs.md](./local_map.py_docs.md)

### Functions

- **`__call__`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`__init__`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`_new_tensor`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`_redistribute`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`autograd_key`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`backward`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`call_local_map`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`create_hop_fw_bw`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`defer_inlining`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`fake_mode_key`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`forward`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`functional_mode_key`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`fw_with_masks`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`joint_f`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`prepare_fw_with_masks`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`proxy_mode_key`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`proxy_mode_key_common`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`real_impl`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`redistribute_bw_inputs`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`redistribute_bw_outputs`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`redistribute_fw_inputs`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`redistribute_fw_outputs`**: [local_map.py_docs.md](./local_map.py_docs.md)

### Imports

- **`AOTConfig`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`Any`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`Callable`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`DispatchKey`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`FakeTensor`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`FunctionalTensor`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`GraphModule`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`HigherOrderOperator`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`MemoryFormatMeta`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`ProxyTorchDispatchMode`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`_CachedTorchDispatchMode`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`collections.abc`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`contextlib`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`contextmanager`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`detect_fake_mode`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`disable_functional_mode`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`disable_proxy_modes_tracing`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`functools`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`has_free_unbacked_symbols`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`partition_fn`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`prepare_for_partitioner`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`suspend_functionalization`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`torch`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`torch._C`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`torch._dispatch.python`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`torch._functorch._aot_autograd.graph_capture`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`torch._functorch._aot_autograd.graph_compile`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`torch._functorch._aot_autograd.runtime_wrappers`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`torch._functorch._aot_autograd.schemas`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`torch._functorch.aot_autograd`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`torch._guards`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`torch._higher_order_ops.utils`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`torch._inductor.compile_fx`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`torch._ops`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`torch._subclasses.fake_tensor`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`torch._subclasses.functional_tensor`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`torch.fx`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`torch.fx.experimental.proxy_tensor`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`torch.fx.experimental.symbolic_shapes`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`torch.utils._pytree`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`torch.utils.checkpoint`**: [local_map.py_docs.md](./local_map.py_docs.md)
- **`typing`**: [local_map.py_docs.md](./local_map.py_docs.md)


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
