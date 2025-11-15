# Keyword Index: `test/inductor/test_fp8.py`

## File Information

- **Original File**: [test/inductor/test_fp8.py](../../../test/inductor/test_fp8.py)
- **Documentation**: [`test_fp8.py_docs.md`](./test_fp8.py_docs.md)
- **Folder**: `test/inductor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`ScaledMMStridePass`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`TestFP8Lowering`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`TestFP8Types`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)

### Functions

- **`__call__`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`__init__`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`_fix_fp8_dtype_for_rocm`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`amax_fp8`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`f`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`fake_scaled_mm_impl`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`fake_scaled_mm_meta`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`forward`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`fp8_cast`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`fp8_matmul_unwrapped`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`fp8_saturated`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`linear`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`ln`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`ln_fp8`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`test_amax_along_with_fp8_quant`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`test_amax_fp8_quant`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`test_bad_cast`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`test_eager_fallback`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`test_layernorm_fp8_quant`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`test_layernorm_fp8_quant_benchmark`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`test_main_loop_scaling`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`test_mx_fp8_max_autotune`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`test_mx_fusion`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`test_rowwise_scaling`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`test_rowwise_scaling_acceptable_input_dims`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`test_rowwise_scaling_tma_template`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`test_scaled_mm_preserves_strides`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`test_tensorwise_scaling`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`test_tensorwise_scaling_acceptable_input_dims`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`test_tensorwise_scaling_tma_template`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`test_to_fp8_saturated`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`test_unacceptable_input_dims`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`test_unacceptable_scale_dims_rowwise_scaling`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`test_valid_cast`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`test_xblock_for_small_numel`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)

### Imports

- **`FileCheck`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`PatternMatcherPass`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`ScalingType`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`Tensor`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`Union`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`ceil_div`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`config`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`functools`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`has_triton_tma_device`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`run_and_get_code`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`run_tests`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`torch`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`torch._C`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`torch._inductor`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`torch._inductor.pattern_matcher`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`torch._inductor.test_case`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`torch._inductor.utils`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`torch.nn.functional`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`torch.testing._internal.common_cuda`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`torch.testing._internal.common_quantized`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`torch.testing._internal.inductor_utils`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`torch.utils._triton`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`typing`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)
- **`unittest`**: [test_fp8.py_docs.md](./test_fp8.py_docs.md)


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
