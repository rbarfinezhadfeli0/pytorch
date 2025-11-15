# Keyword Index: `docs/aten/src/ATen/native/hip/bgemm_kernels/bgemm_kernel_template.h_docs.md`

## File Information

- **Original File**: [docs/aten/src/ATen/native/hip/bgemm_kernels/bgemm_kernel_template.h_docs.md](../../../../../../../../docs/aten/src/ATen/native/hip/bgemm_kernels/bgemm_kernel_template.h_docs.md)
- **Documentation**: [`bgemm_kernel_template.h_docs.md_docs.md`](./bgemm_kernel_template.h_docs.md_docs.md)
- **Folder**: `docs/aten/src/ATen/native/hip/bgemm_kernels`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Identifiers

- **`A`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`ABLOCK_TRANSFER`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`ABlockLdsExtraM`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`ABlockTransferDstScalarPerVector_AK1`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`ABlockTransferSrcAccessOrder`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`ABlockTransferSrcVectorDim`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`ABlockTransferThreadClusterArrangeOrder`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`ABlockTransferThreadClusterLengths_AK0_M_AK1`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`AElementwiseOperation`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`AK1`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`ALayout`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`ATen`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`AccDataType`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`BBLOCK_TRANSFER`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`BBlockTransferDstScalarPerVector_BK1`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`BBlockTransferSrcAccessOrder`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`BBlockTransferSrcVectorDim`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`BBlockTransferThreadClusterArrangeOrder`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`BBlockTransferThreadClusterLengths_BK0_N_BK1`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`BElementOp`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`BElementwiseOperation`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`BK1`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`BLayout`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`B_DATA_TYPE`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`BlockSize`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`C`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`CDEElementOp`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`CDEShuffleBlockTransferScalarPerVectors`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`CDataType`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`CElementwiseOperation`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`CK`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`CSHUFFLEBLOCK_TRANSFER`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`CSHUFFLE_MXDL_PWPS`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`CSHUFFLE_NXDL_PWPS`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`CShuffleDataType`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`CShuffleMXdlPerWavePerShuffle`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`CShuffleNXdlPerWavePerShuffle`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`CUDABLAS_BGEMM_ARGTYPES`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Code`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`ColumnMajor`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Common`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Considerations`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`CshuffleType`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Define`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Dependencies`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Detailed`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`DeviceBatchedGemmMultiD_Xdl_CShuffle_V3`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Documentation`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`DsDataType`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`DsLayout`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Examples`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Extension`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`F32`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`For`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`GEMM`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`GEMM_SPEC`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`GPU`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`HIPBlas`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Header`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`High`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Index`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Is`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`KB`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`KBLOCK`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Keyword`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Level`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Library`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`MPerXDL`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`MakeArgument`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`MakeInvoker`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`NBLOCK`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`NPerBlock`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`NPerXDL`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Namespaces`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`No`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Notes`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Original`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Overview`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Patterns`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Repository`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Role`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Row`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`RowMajor`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Run`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`S`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Safety`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Security`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Sequence`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Source`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`StreamConfig`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Structure`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`TRANSA`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Tensor`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Test`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Testing`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`Tuple`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`WAVE_MAP_M`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`WAVE_TILE_M`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)
- **`WAVE_TILE_N`**: [bgemm_kernel_template.h_docs.md_docs.md](./bgemm_kernel_template.h_docs.md_docs.md)


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
