# Keyword Index: `test/inductor/test_kernel_benchmark.py`

## File Information

- **Original File**: [test/inductor/test_kernel_benchmark.py](../../../test/inductor/test_kernel_benchmark.py)
- **Documentation**: [`test_kernel_benchmark.py_docs.md`](./test_kernel_benchmark.py_docs.md)
- **Folder**: `test/inductor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`TestKernelBenchmark`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)

### Functions

- **`check_bandwidth`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`f`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`get_compiled_module`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`setUp`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`setUpClass`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`tearDownClass`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`test_fused_layernorm_bandwidth_computation`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`test_matmul_bandwidth_computation`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`test_matmul_triton_kernel_benchmark`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`test_mm_slice_add_bandwidth_computation`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`test_mm_slice_add_bandwidth_computation_2`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`test_mm_triton_kernel_benchmark`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`test_pw_kernel_benchmark`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`test_reduction_bandwidth_computation`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`test_remove_inductor_deps`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`test_remove_inductor_deps_multiple_kernels`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`test_remove_inductor_deps_scalar`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`test_remove_inductor_deps_templates`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`test_slice_add_bandwidth_computation`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`test_slice_add_cat_bandwidth_computation`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`test_slice_mm_bandwidth_computation`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`test_split_scan`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`test_star_dep`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`test_unused_input_bandwidth_computation`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`triton_`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`verify_compiled_kernels`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`verify_remove_inductor_deps`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)

### Imports

- **`FileCheck`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`GPU_TYPE`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`PyCodeCache`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`config`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`contextlib`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`fresh_cache`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`get_clean_triton`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`os`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`patch`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`rand_strided`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`run_tests`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`subprocess`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`sys`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`torch`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`torch._dynamo.testing`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`torch._inductor`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`torch._inductor.async_compile`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`torch._inductor.codecache`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`torch._inductor.test_case`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`torch._inductor.utils`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`torch.testing`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`torch.testing._internal.common_cuda`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`torch.testing._internal.inductor_utils`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`torch.utils._get_clean_triton`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`unittest`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`unittest.mock`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)
- **`xfailIfSM89`**: [test_kernel_benchmark.py_docs.md](./test_kernel_benchmark.py_docs.md)


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
