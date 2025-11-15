# Keyword Index: `test/inductor/test_cooperative_reductions.py`

## File Information

- **Original File**: [test/inductor/test_cooperative_reductions.py](../../../test/inductor/test_cooperative_reductions.py)
- **Documentation**: [`test_cooperative_reductions.py_docs.md`](./test_cooperative_reductions.py_docs.md)
- **Folder**: `test/inductor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`CooperativeReductionTests`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`MultiKernelCooperativeReductionTests`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`NoPersistCooperativeReductionTests`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`TestFixedConfigs`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`TestingHeuristics`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)

### Functions

- **`__init__`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`_check`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`fn`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`run_and_check`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`setUp`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`test_bool_reduction_fns`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`test_chained_reductions`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`test_fixed_config_with_larger_xblock_than_xnumel`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`test_fixed_configs`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`test_min_max_non_power_of_2_rsplit`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`test_non_power_of_2`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`test_reduce_split`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`test_reduction_fns`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`test_welford_non_power_of_2_rsplit`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`triton_kernel_kwargs`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)

### Imports

- **`Any`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`FixedTritonConfig`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`GPU_TYPE`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`IS_SM89`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`InductorChoices`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`SIMDKernelFeatures`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`TestCase`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`assert_close`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`config`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`run_and_get_code`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`run_tests`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`sympy`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`torch`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`torch._dynamo.test_case`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`torch._inductor`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`torch._inductor.choices`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`torch._inductor.codegen.simd_kernel_features`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`torch._inductor.codegen.triton`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`torch._inductor.test_case`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`torch._inductor.utils`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`torch.testing`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`torch.testing._internal.common_cuda`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`torch.testing._internal.inductor_utils`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`typing`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)
- **`unittest`**: [test_cooperative_reductions.py_docs.md](./test_cooperative_reductions.py_docs.md)


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
