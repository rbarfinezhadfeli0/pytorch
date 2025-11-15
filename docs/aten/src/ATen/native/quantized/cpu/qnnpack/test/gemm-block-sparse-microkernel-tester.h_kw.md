# Keyword Index: `aten/src/ATen/native/quantized/cpu/qnnpack/test/gemm-block-sparse-microkernel-tester.h`

## File Information

- **Original File**: [aten/src/ATen/native/quantized/cpu/qnnpack/test/gemm-block-sparse-microkernel-tester.h](../../../../../../../../../aten/src/ATen/native/quantized/cpu/qnnpack/test/gemm-block-sparse-microkernel-tester.h)
- **Documentation**: [`gemm-block-sparse-microkernel-tester.h_docs.md`](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **Folder**: `aten/src/ATen/native/quantized/cpu/qnnpack/test`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`GemmBlockSparseMicrokernelTester`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`pytorch_qnnp_conv_dynamic_quantization_params`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)

### Functions

- **`aStride`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`aZeroPoint`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`bZeroPoint`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`biasN`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`cStride`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`colBlockSize`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`fillBlockSparseWeights`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`iterations`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`k`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`ks`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`m`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`mr`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`multiplier`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`n`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`nr`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`printMatrix`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`qmax`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`qmin`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`rowBlockSize`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`sparsity`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`test`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`test_packed`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)

### Includes

- **`algorithm`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`cassert`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`cmath`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`cstddef`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`cstdlib`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`fp16.h`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`functional`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`pack_block_sparse.h`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`qnnpack/AlignedAllocator.h`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`qnnpack/params.h`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`qnnpack/requantization.h`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`random`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)
- **`vector`**: [gemm-block-sparse-microkernel-tester.h_docs.md](./gemm-block-sparse-microkernel-tester.h_docs.md)


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
