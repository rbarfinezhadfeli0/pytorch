# Documentation: `docs/torch/distributed/_functional_collectives.py_kw.md`

## File Metadata

- **Path**: `docs/torch/distributed/_functional_collectives.py_kw.md`
- **Size**: 10,393 bytes (10.15 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/distributed/_functional_collectives.py`

## File Information

- **Original File**: [torch/distributed/_functional_collectives.py](../../../torch/distributed/_functional_collectives.py)
- **Documentation**: [`_functional_collectives.py_docs.md`](./_functional_collectives.py_docs.md)
- **Folder**: `torch/distributed`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`AsyncCollectiveTensor`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`_FromTorchTensor`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`needed`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`support`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`that`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)

### Functions

- **`__coerce_same_metadata_as_tangent__`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`__new__`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`__repr__`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`__tensor_flatten__`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`__tensor_unflatten__`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`__torch_dispatch__`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`_all_gather_into_tensor_coalesced_meta`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`_all_gather_into_tensor_coalesced_native_meta`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`_all_gather_into_tensor_meta`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`_all_gather_into_tensor_native_meta`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`_all_gather_into_tensor_out_native_meta`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`_all_reduce__meta`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`_all_reduce_coalesced__meta`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`_all_reduce_coalesced_meta`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`_all_reduce_meta`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`_all_to_all_single_meta`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`_are_we_tracing`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`_broadcast__meta`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`_broadcast_meta`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`_expand_group`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`_get_acs_underlying_tensor`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`_is_view_op`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`_make_all_gather_out_tensor`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`_maybe_wrap_tensor`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`_reduce_scatter_tensor_coalesced_meta`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`_reduce_scatter_tensor_coalesced_native_meta`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`_reduce_scatter_tensor_meta`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`_reduce_scatter_tensor_native_meta`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`_resolve_group_name`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`_wait_tensor_meta`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`all_gather_inplace`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`all_gather_into_tensor_coalesced`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`all_gather_tensor`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`all_gather_tensor_autograd`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`all_gather_tensor_inplace`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`all_reduce`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`all_reduce_coalesced`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`all_reduce_eager`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`all_reduce_inplace`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`all_reduce_wait_compiled`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`all_to_all_inplace`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`all_to_all_single`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`all_to_all_single_autograd`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`allow_inflight_collective_as_graph_input_ctx`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`backward`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`broadcast`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`cast_listint`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`cast_listlistint`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`forward`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`functional_collective`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`in`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`is_torchdynamo_compiling`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`mk_out_tensor`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`numpy`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`permute_tensor`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`reduce_scatter_tensor`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`reduce_scatter_tensor_autograd`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`reduce_scatter_tensor_coalesced`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`reduce_scatter_tensor_inplace`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`tolist`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`trigger_wait`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`unwrap`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`wait`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`wait_tensor`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`wrap`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)

### Imports

- **`.`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`Any`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`DeviceMesh`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`_functional_collectives_impl`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`contextlib`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`get_proxy_mode`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`is_dynamo_compiling`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`sys`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`torch`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`torch.compiler`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`torch.distributed`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`torch.distributed.device_mesh`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`torch.distributed.distributed_c10d`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`torch.fx.experimental.proxy_tensor`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`torch.utils._cxx_pytree`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`torch.utils._pytree`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`torchdynamo`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`tree_map_only`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`typing`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)
- **`warnings`**: [_functional_collectives.py_docs.md](./_functional_collectives.py_docs.md)


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

This file is part of the PyTorch framework located at `docs/torch/distributed`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/distributed`):

- [`_mesh_layout.py_docs.md_docs.md`](./_mesh_layout.py_docs.md_docs.md)
- [`run.py_docs.md_docs.md`](./run.py_docs.md_docs.md)
- [`device_mesh.py_docs.md_docs.md`](./device_mesh.py_docs.md_docs.md)
- [`_composable_state.py_docs.md_docs.md`](./_composable_state.py_docs.md_docs.md)
- [`run.py_kw.md_docs.md`](./run.py_kw.md_docs.md)
- [`_dist2.py_kw.md_docs.md`](./_dist2.py_kw.md_docs.md)
- [`_state_dict_utils.py_kw.md_docs.md`](./_state_dict_utils.py_kw.md_docs.md)
- [`rendezvous.py_kw.md_docs.md`](./rendezvous.py_kw.md_docs.md)
- [`rendezvous.py_docs.md_docs.md`](./rendezvous.py_docs.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `_functional_collectives.py_kw.md_docs.md`
- **Keyword Index**: `_functional_collectives.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
