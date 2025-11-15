# Keyword Index: `aten/src/ATen/native/quantized/cpu/qnnpack/src/operator-run.c`

## File Information

- **Original File**: [aten/src/ATen/native/quantized/cpu/qnnpack/src/operator-run.c](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/src/operator-run.c)
- **Documentation**: [`operator-run.c_docs.md`](./operator-run.c_docs.md)
- **Folder**: `aten/src/ATen/native/quantized/cpu/qnnpack/src`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`average_pooling_context`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`channel_shuffle_context`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`clamp_contiguous_context`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`clamp_strided_context`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`global_average_pooling_context`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`lut_contiguous_context`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`lut_strided_context`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`max_pooling_context`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`pytorch_q8gemm_sparse_parameters`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`pytorch_qnnp_conv_dynamic_quantization_params`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`q8add_contiguous_context`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`q8add_strided_context`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`q8conv_context`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`q8dwconv2d_context`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`q8dwconv3d_context`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`q8gemm_context`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`q8gemm_prepackA_sparse_dq_context`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`q8gemm_sparse_dq_context`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`q8gemm_xzp_context`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`q8sum_rows_context`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`u8softargmax_context`**: [operator-run.c_docs.md](./operator-run.c_docs.md)

### Functions

- **`compute_average_pooling_multipass`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`compute_average_pooling_unipass`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`compute_channel_shuffle_fixed`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`compute_channel_shuffle_variable`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`compute_clamp_contiguous`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`compute_clamp_strided`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`compute_dwconv2d_multiipass`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`compute_dwconv2d_unipass`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`compute_dwconv3d_multiipass`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`compute_global_average_pooling_multipass`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`compute_global_average_pooling_unipass`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`compute_lut_contiguous`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`compute_lut_strided`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`compute_max_pooling`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`compute_q8add_contiguous`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`compute_q8add_strided`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`compute_q8conv`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`compute_q8gemm`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`compute_q8gemm_prepack_a_sparse`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`compute_q8gemm_prepacked_sparse_dq`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`compute_q8gemm_sparse_dq`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`compute_q8gemm_xzp`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`compute_sum_rows`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`compute_u8softargmax`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`if`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`pytorch_qnnp_run_operator`**: [operator-run.c_docs.md](./operator-run.c_docs.md)

### Includes

- **`assert.h`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`malloc.h`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`pytorch_qnnpack.h`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`qnnpack/common.h`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`qnnpack/log.h`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`qnnpack/math.h`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`qnnpack/operator.h`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`qnnpack/params.h`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`stddef.h`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`stdint.h`**: [operator-run.c_docs.md](./operator-run.c_docs.md)
- **`string.h`**: [operator-run.c_docs.md](./operator-run.c_docs.md)


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
