# Keyword Index: `test/test_matmul_cuda.py`

## File Information

- **Original File**: [test/test_matmul_cuda.py](../../test/test_matmul_cuda.py)
- **Documentation**: [`test_matmul_cuda.py_docs.md`](./test_matmul_cuda.py_docs.md)
- **Folder**: `test`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`TestMatmulCuda`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`TestMixedDtypesLinearCuda`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)

### Functions

- **`_convert_to_cpu`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`_expand_to_batch`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`blas_library_context`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`create_inputs`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`cublas_addmm`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`expand`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`grouped_mm_helper`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`is_addmm`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`is_batched`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`run_test`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`setUp`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`tearDown`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`test_addmm_baddmm_dtype_overload`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`test_cublas_addmm`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`test_cublas_addmm_alignment`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`test_cublas_addmm_bias_shapes`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`test_cublas_addmm_no_reduced_precision`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`test_cublas_addmm_reduced_precision`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`test_cublas_addmm_reduced_precision_fp16_accumulate`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`test_cublas_and_lt_reduced_precision_fp16_accumulate`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`test_cublas_baddbmm_large_input`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`test_cublas_batch_invariance_blackwell`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`test_cublas_deterministic`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`test_fp16_accum_and_fp32_out_failure`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`test_greencontext_carveout`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`test_grouped_gemm_2d_2d`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`test_grouped_gemm_2d_3d`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`test_grouped_gemm_3d_2d`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`test_grouped_gemm_3d_3d`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`test_grouped_gemm_compiled`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`test_input_dimension_checking_out_dtype`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`test_mixed_dtypes_linear`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`test_mm_bmm_dtype_overload`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`xfailIfSM100OrLaterNonRTXAndCondition`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)

### Imports

- **`Callable`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`IS_BIG_GPU`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`TestCase`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`collections.abc`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`contextlib`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`functools`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`itertools`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`make_tensor`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`partial`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`product`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`time`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`torch`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`torch._inductor.test_case`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`torch.quantization._quantized_conversions`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`torch.testing`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`torch.testing._internal.common_cuda`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`torch.testing._internal.common_device_type`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`torch.testing._internal.inductor_utils`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)
- **`unittest`**: [test_matmul_cuda.py_docs.md](./test_matmul_cuda.py_docs.md)


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
