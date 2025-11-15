# Keyword Index: `test/inductor/test_snode_runtime.py`

## File Information

- **Original File**: [test/inductor/test_snode_runtime.py](../../../test/inductor/test_snode_runtime.py)
- **Documentation**: [`test_snode_runtime.py_docs.md`](./test_snode_runtime.py_docs.md)
- **Folder**: `test/inductor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`ComputeBoundedTests`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`MemoryBoundedTests`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`TestCase`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`TestCommAnalysis`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`UnsupportedTests`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)

### Functions

- **`T`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`_verify_runtime_estimation`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`assertNotZero`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`assertZero`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`calculate_runtime`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`compile_but_use_eager`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`f`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`fn`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`inner_compile`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`setUp`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`tearDown`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`test_addmm`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`test_all_gather_into_tensor`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`test_all_gather_into_tensor_coalesced`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`test_all_reduce`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`test_all_reduce_coalesced`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`test_bmm`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`test_conv1d`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`test_conv2d`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`test_conv2d_transpose`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`test_conv3d`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`test_dynamic`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`test_horizontal_reduction_pointwise`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`test_legacy_all_gather_into_tensor_coalesced`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`test_legacy_all_reduce`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`test_legacy_all_reduce_coalesced`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`test_mm`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`test_no_cuda`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`test_no_op`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`test_pointwise`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`test_reduce_scatter_tensor`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`test_reduce_scatter_tensor_coalesced`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`test_relu`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)

### Imports

- **`FakeStore`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`GPU_TYPE`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`TestCase`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`compile_fx`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`config`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`contextlib`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`estimate_nccl_collective_runtime`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`expectedFailureXPU`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`is_collective`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`run_tests`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`skipIf`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`torch`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`torch._inductor`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`torch._inductor.comm_analysis`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`torch._inductor.compile_fx`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`torch._inductor.test_case`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`torch._inductor.utils`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`torch.distributed`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`torch.testing._internal.common_device_type`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`torch.testing._internal.distributed.fake_pg`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`torch.testing._internal.inductor_utils`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)
- **`unittest`**: [test_snode_runtime.py_docs.md](./test_snode_runtime.py_docs.md)


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
