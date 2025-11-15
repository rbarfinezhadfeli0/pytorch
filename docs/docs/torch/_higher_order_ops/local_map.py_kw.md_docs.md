# Documentation: `docs/torch/_higher_order_ops/local_map.py_kw.md`

## File Metadata

- **Path**: `docs/torch/_higher_order_ops/local_map.py_kw.md`
- **Size**: 5,989 bytes (5.85 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
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

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_higher_order_ops`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_higher_order_ops`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/_higher_order_ops`):

- [`schema.py_docs.md_docs.md`](./schema.py_docs.md_docs.md)
- [`run_const_graph.py_docs.md_docs.md`](./run_const_graph.py_docs.md_docs.md)
- [`effects.py_kw.md_docs.md`](./effects.py_kw.md_docs.md)
- [`partitioner.py_docs.md_docs.md`](./partitioner.py_docs.md_docs.md)
- [`strict_mode.py_docs.md_docs.md`](./strict_mode.py_docs.md_docs.md)
- [`out_dtype.py_kw.md_docs.md`](./out_dtype.py_kw.md_docs.md)
- [`wrap.py_docs.md_docs.md`](./wrap.py_docs.md_docs.md)
- [`while_loop.py_kw.md_docs.md`](./while_loop.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`invoke_subgraph.py_docs.md_docs.md`](./invoke_subgraph.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `local_map.py_kw.md_docs.md`
- **Keyword Index**: `local_map.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
