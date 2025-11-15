# Keyword Index: `torch/csrc/utils/tensor_new.cpp`

## File Information

- **Original File**: [torch/csrc/utils/tensor_new.cpp](../../../../torch/csrc/utils/tensor_new.cpp)
- **Documentation**: [`tensor_new.cpp_docs.md`](./tensor_new.cpp_docs.md)
- **Folder**: `torch/csrc/utils`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`CheckSparseTensorInvariantsContext`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`CtorOrNew`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`T`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`a`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`from`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`indices`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`initialization`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)

### Functions

- **`_validate_sparse_bsc_tensor_args`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`_validate_sparse_bsr_tensor_args`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`_validate_sparse_compressed_tensor_args`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`_validate_sparse_compressed_tensor_args_template`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`_validate_sparse_coo_tensor_args`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`_validate_sparse_csc_tensor_args`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`_validate_sparse_csr_tensor_args`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`as_tensor`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`asarray`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`base_tensor_ctor`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`build_options`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`check_base_legacy_new`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`check_legacy_ctor_device`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`if`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`indexing_tensor_from_data`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`infer_scalar_type`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`internal_new_from_data`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`isValidDLPackCapsule`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`legacy_new_from_sequence`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`legacy_sparse_tensor_generic_ctor_new`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`legacy_tensor_ctor`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`legacy_tensor_generic_ctor_new`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`legacy_tensor_new`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`new_from_data_copy`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`new_tensor`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`new_with_sizes`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`new_with_storage`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`only_lift_cpu_tensors`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`recursive_store`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`set_only_lift_cpu_tensors`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`sparse_bsc_tensor_ctor`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`sparse_bsr_tensor_ctor`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`sparse_compressed_tensor_ctor`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`sparse_compressed_tensor_ctor_worker`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`sparse_coo_tensor_ctor`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`sparse_csc_tensor_ctor`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`sparse_csr_tensor_ctor`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`tensor_ctor`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`tensor_fromDLPack`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`tensor_fromDLPackImpl`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`tensor_frombuffer`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`typeIdWithDefault`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)

### Includes

- **`ATen/ATen.h`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`ATen/DLConvertor.h`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`ATen/InitialTensorOptions.h`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`ATen/NamedTensorUtils.h`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`ATen/NativeFunctions.h`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`ATen/SparseCsrTensorUtils.h`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`ATen/TracerMode.h`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`ATen/dlpack.h`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`c10/core/Backend.h`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`c10/core/DispatchKeySet.h`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`c10/core/Layout.h`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`c10/util/Exception.h`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`c10/util/irange.h`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`optional`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`pybind11/pybind11.h`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`stdexcept`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`torch/csrc/DynamicTypes.h`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`torch/csrc/Exceptions.h`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`torch/csrc/Size.h`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`torch/csrc/autograd/generated/variable_factories.h`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`torch/csrc/autograd/variable.h`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`torch/csrc/python_headers.h`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`torch/csrc/utils/device_lazy_init.h`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`torch/csrc/utils/numpy_stub.h`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`torch/csrc/utils/pybind.h`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`torch/csrc/utils/python_arg_parser.h`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`torch/csrc/utils/python_numbers.h`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`torch/csrc/utils/python_scalars.h`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`torch/csrc/utils/python_strings.h`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`torch/csrc/utils/tensor_new.h`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`torch/csrc/utils/tensor_numpy.h`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`vector`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)

### Namespaces

- **`Tensor`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`static`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)
- **`torch`**: [tensor_new.cpp_docs.md](./tensor_new.cpp_docs.md)


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
