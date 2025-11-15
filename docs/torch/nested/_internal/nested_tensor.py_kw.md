# Keyword Index: `torch/nested/_internal/nested_tensor.py`

## File Information

- **Original File**: [torch/nested/_internal/nested_tensor.py](../../../../torch/nested/_internal/nested_tensor.py)
- **Documentation**: [`nested_tensor.py_docs.md`](./nested_tensor.py_docs.md)
- **Folder**: `torch/nested/_internal`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`NestedTensor`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`ViewBufferFromNested`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`ViewNestedFromBuffer`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`custom`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`logic`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`serialization`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)

### Functions

- **`__init__`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`__new__`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`__reduce_ex__`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`__repr__`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`__tensor_flatten__`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`__tensor_unflatten__`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`__torch_dispatch__`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`__torch_function__`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`_get_max_seqlen`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`_get_min_seqlen`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`_get_sdpa_extreme_seqlen`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`_is_contiguous_or_false`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`_load_val_from_tensor`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`_max_seqlen`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`_max_seqlen_tensor`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`_maybe_max_seqlen`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`_maybe_min_seqlen`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`_min_seqlen`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`_min_seqlen_tensor`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`_nt_view_dummy`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`_rebuild_njt`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`_store_val_in_tensor`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`backward`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`buffer_from_jagged`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`forward`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`get_tensor_symint`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`jagged_from_list`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`jagged_from_tensor_and_lengths`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`lengths`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`nested_from_padded`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`nested_view_from_values_offsets`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`nested_view_from_values_offsets_lengths`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`offsets`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`values`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)

### Imports

- **`.ops`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`DispatchKey`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`FakeTensor`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`NestedIntNode`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`WeakTensorKeyDictionary`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`is_contiguous_for_memory_format_or_false`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`is_expandable_to`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`jagged_torch_function`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`lookup_jagged`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`maybe_enable_thunkify`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`mb_unwrap_functional_tensor`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`to`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`torch`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`torch._C`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`torch._prims_common`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`torch._subclasses.fake_tensor`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`torch._subclasses.functional_tensor`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`torch.fx.experimental.proxy_tensor`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`torch.nested._internal.nested_int`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`torch.utils.weak`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)
- **`typing`**: [nested_tensor.py_docs.md](./nested_tensor.py_docs.md)


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
