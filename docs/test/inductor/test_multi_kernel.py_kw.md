# Keyword Index: `test/inductor/test_multi_kernel.py`

## File Information

- **Original File**: [test/inductor/test_multi_kernel.py](../../../test/inductor/test_multi_kernel.py)
- **Documentation**: [`test_multi_kernel.py_docs.md`](./test_multi_kernel.py_docs.md)
- **Folder**: `test/inductor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`MultiKernelTest`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`TransformerSnippet`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)

### Functions

- **`__init__`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`_contains_multi_kernel_code`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`_contains_size_hint_multi_kernel_code`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`example_inputs`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`f`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`fn`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`forward`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`make_cpp_wrapper_test`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`mock_run`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`test_batchnorm_training`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`test_inplace_update`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`test_layernorm`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`test_pass_same_arg_multi_times`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`test_reduction_scratch_buffer`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`test_softmax`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`test_softmax_force_non_persistent_reduction`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`test_softmax_warn_mixed_layout`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`test_sort_disables_multi_kernel`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`test_split_scan`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`test_transformer_snippet`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`test_transformer_snippet_with_fallback_random`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`test_triton_gemm`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`test_triton_relu_fused_gemm`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)

### Imports

- **`MultiKernelCall`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`TestCase`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`codecache`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`config`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`functional`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`make_tensor`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`nn`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`os`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`re`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`reset_rng_state`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`run_and_get_code`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`run_tests`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`torch`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`torch._dynamo.testing`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`torch._inductor`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`torch._inductor.codegen.multi_kernel`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`torch._inductor.test_case`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`torch._inductor.utils`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`torch.nn`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`torch.testing`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`torch.testing._internal.inductor_utils`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)
- **`unittest`**: [test_multi_kernel.py_docs.md](./test_multi_kernel.py_docs.md)


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
