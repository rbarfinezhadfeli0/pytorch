# Keyword Index: `aten/src/ATen/core/TensorBase.h`

## File Information

- **Original File**: [aten/src/ATen/core/TensorBase.h](../../../../../aten/src/ATen/core/TensorBase.h)
- **Documentation**: [`TensorBase.h_docs.md`](./TensorBase.h_docs.md)
- **Folder**: `aten/src/ATen/core`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`ExclusivelyOwnedTraits`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`MaybeOwnedTraits`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`Node`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`PtrTraits`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`Scalar`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`TORCH_API`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`Tensor`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`TensorBase`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`unsafe_borrow_t`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`which`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)

### Functions

- **`_is_zerotensor`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`_set_conj`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`_set_fw_grad`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`_set_neg`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`_set_zero`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`assignBorrow`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`constexpr`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`contiguous`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`createBorrow`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`debugBorrowIsValid`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`defined`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`destroyBorrow`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`device`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`dim`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`dtype`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`element_size`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`get_device`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`has_names`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`has_storage`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_alias_of`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_complex`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_conj`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_contiguous`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_contiguous_or_false`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_cpu`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_cuda`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_floating_point`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_hip`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_hpu`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_inference`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_ipu`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_lazy`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_maia`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_meta`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_metal`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_mkldnn`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_mps`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_mtia`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_neg`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_nested`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_non_overlapping_and_dense`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_privateuseone`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_quantized`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_signed`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_sparse`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_sparse_csr`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_ve`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_vulkan`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_xla`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`is_xpu`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`itemsize`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`key_set`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`layout`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`legacyExtractDispatchKey`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`make_tensor_base`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`names`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`nbytes`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`ndimension`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`numel`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`options`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`requires_grad`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`reset`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`scalar_type`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`share_memory_`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`size`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`sizes`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`storage_offset`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`stride`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`strides`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`suggest_memory_format`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`sym_is_contiguous`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`sym_nbytes`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`sym_numel`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`sym_size`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`sym_sizes`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`sym_storage_offset`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`sym_stride`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`sym_strides`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`variable_excluded_from_dispatch`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`wrap_tensor_impl`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)

### Includes

- **`ATen/StorageUtils.h`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`ATen/core/NamedTensor.h`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`ATen/core/QuantizerBase.h`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`ATen/core/TensorAccessor.h`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`c10/core/Device.h`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`c10/core/Layout.h`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`c10/core/MemoryFormat.h`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`c10/core/ScalarType.h`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`c10/core/ScalarTypeToTypeMeta.h`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`c10/core/Storage.h`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`c10/core/SymIntArrayRef.h`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`c10/core/TensorImpl.h`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`c10/core/TensorOptions.h`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`c10/core/UndefinedTensorImpl.h`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`c10/core/WrapDimMinimal.h`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`c10/util/C++17.h`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`c10/util/Exception.h`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`c10/util/ExclusivelyOwned.h`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`c10/util/ExclusivelyOwnedTensorTraits.h`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`c10/util/MaybeOwned.h`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`c10/util/intrusive_ptr.h`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`optional`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)

### Namespaces

- **`at`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`c10`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`detail`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`impl`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`symint`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)
- **`torch`**: [TensorBase.h_docs.md](./TensorBase.h_docs.md)


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
