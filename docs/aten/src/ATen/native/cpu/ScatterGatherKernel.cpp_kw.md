# Keyword Index: `aten/src/ATen/native/cpu/ScatterGatherKernel.cpp`

## File Information

- **Original File**: [aten/src/ATen/native/cpu/ScatterGatherKernel.cpp](../../../../../../aten/src/ATen/native/cpu/ScatterGatherKernel.cpp)
- **Documentation**: [`ScatterGatherKernel.cpp_docs.md`](./ScatterGatherKernel.cpp_docs.md)
- **Folder**: `aten/src/ATen/native/cpu`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`ReduceAdd`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`ReduceMaximum`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`ReduceMean`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`ReduceMinimum`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`ReduceMultiply`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`TensorAssign`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`_cpu_scatter_gather_dim_loop`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`cpu_scatter_gather_base_kernel`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)

### Functions

- **`constexpr`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`cpu_gather_expanded_index_kernel`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`cpu_scatter_reduce_expanded_index`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`create_acc_buffer`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`for`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`gather_cpu_kernel`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`gather_expanded_index_kernel`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`scatter_add_cpu_kernel`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`scatter_add_expanded_index_kernel`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`scatter_cpu_kernel`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`scatter_fill_cpu_kernel`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`scatter_reduce_cpu_kernel`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`scatter_reduce_expanded_index_kernel`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`scatter_reduce_two_cpu_kernel`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`scatter_scalar_reduce_cpu_kernel`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)

### Includes

- **`ATen/Config.h`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`ATen/Dispatch.h`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`ATen/Functions.h`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`ATen/NativeFunctions.h`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`ATen/NumericUtils.h`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`ATen/OpMathType.h`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`ATen/Parallel.h`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`ATen/core/Tensor.h`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`ATen/cpu/vec/functional.h`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`ATen/cpu/vec/vec.h`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`ATen/native/DispatchStub.h`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`ATen/native/NonEmptyUtils.h`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`ATen/native/TensorAdvancedIndexing.h`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`ATen/native/TensorIterator.h`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`ATen/native/cpu/ReduceUtils.h`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`ATen/ops/empty.h`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`ATen/ops/zeros.h`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`c10/util/irange.h`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`fbgemm/Utils.h`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)

### Namespaces

- **`REGISTER_DISPATCH`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`at`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)
- **`fbgemm`**: [ScatterGatherKernel.cpp_docs.md](./ScatterGatherKernel.cpp_docs.md)


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
