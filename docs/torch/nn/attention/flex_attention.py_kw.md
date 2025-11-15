# Keyword Index: `torch/nn/attention/flex_attention.py`

## File Information

- **Original File**: [torch/nn/attention/flex_attention.py](../../../../torch/nn/attention/flex_attention.py)
- **Documentation**: [`flex_attention.py_docs.md`](./flex_attention.py_docs.md)
- **Folder**: `torch/nn/attention`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`AuxOutput`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`AuxRequest`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`BlockMask`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`FlexKernelOptions`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_ModificationType`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)

### Functions

- **`__getitem__`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`__init__`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`__repr__`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`__str__`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_adjust`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_adjust_num_blocks_and_indices`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_apply_kernel_options`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_broadcast_to_dim`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_convert_block_mask_to_mask`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_convert_mask_to_block_mask`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_create_empty_block_mask`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_create_sparse_block_from_block_mask`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_dense_to_ordered`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_enforce_mem_layouts`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_finalize_outputs`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_flatten`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_flatten_with_keys`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_flex_attention_hop_wrapper`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_get_mod_type`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_identity`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_ordered_to_dense`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_prod`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_round_up_to_multiple`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_sliced_mask_mod_error`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_transpose_ordered`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_unflatten`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_validate_device`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_validate_embed_dim`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_vmap_for_bhqkv`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_warn_once`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`and_mask`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`and_masks`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`as_tuple`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`causal_mask`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`cdiv`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`create_block_mask`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`create_block_vis`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`create_dense_one`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`create_mask`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`flex_attention`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`from_kv_blocks`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`is_col_major`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`is_row_major`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`noop_mask`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`numel`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`or_mask`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`or_masks`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`padding_needed_for_multiple`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`score_mod`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`shape`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`shape_or_none`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`sparsity`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`summarize_section`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`to`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`to_dense`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`to_string`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)

### Imports

- **`Any`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`Callable`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`DeviceLikeType`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`Enum`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`FlexKernelOptions`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`GetAttrKey`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`NotRequired`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`Tensor`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`TransformGetItemToIndex`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`TypedDict`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_set_compilation_env`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`_validate_sdpa_input`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`collections.abc`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`enum`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`flex_attention`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`functools`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`inspect`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`itertools`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`math`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`operator`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`torch`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`torch._dynamo._trace_wrapped_higher_order_op`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`torch._dynamo.backends.debugging`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`torch._higher_order_ops.flex_attention`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`torch._higher_order_ops.utils`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`torch._prims_common`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`torch.fx.experimental.proxy_tensor`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`torch.nn.attention._utils`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`torch.nn.attention.flex_attention`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`torch.utils._pytree`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`typing`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`typing_extensions`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)
- **`warnings`**: [flex_attention.py_docs.md](./flex_attention.py_docs.md)


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
