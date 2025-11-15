# Keyword Index: `aten/src/ATen/native/cpu/UpSampleKernel.cpp`

## File Information

- **Original File**: [aten/src/ATen/native/cpu/UpSampleKernel.cpp](../../../../../../aten/src/ATen/native/cpu/UpSampleKernel.cpp)
- **Documentation**: [`UpSampleKernel.cpp_docs.md`](./UpSampleKernel.cpp_docs.md)
- **Folder**: `aten/src/ATen/native/cpu`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`CheckAlmostAllZeroStrides`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`F`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`HelperInterpBase`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`HelperInterpCubic`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`HelperInterpLinear`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`HelperInterpNearest`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`HelperInterpNearestExact`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`Interpolate`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`being`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`to`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`using`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)

### Functions

- **`_compute_indices_min_size_weights`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`_compute_indices_min_size_weights_aa`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`_separable_upsample_generic_Nd_kernel_impl_single_dim`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`_upsample_nearest_exact1d_kernel_impl`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`_upsample_nearest_exact2d_kernel_impl`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`_upsample_nearest_exact3d_kernel_impl`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`_use_vectorized_kernel_cond_2d`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`aa_filter`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`basic_loop`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`basic_loop_aa_horizontal`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`basic_loop_aa_vertical`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`check_almost_all_zero_stride`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`constexpr`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`cpu_upsample_genNd_backward_aa`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`cpu_upsample_generic`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`cpu_upsample_generic_aa`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`cpu_upsample_linear_channels_last`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`cpu_upsample_nearest_channels_last`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`eval`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`if`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`init_indices_weights`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`interpolate`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`interpolate_aa_single_dim`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`interpolate_aa_single_dim_zero_strides`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`is_contiguous_stride`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`is_zero_stride`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`separable_upsample_generic_Nd_kernel_impl`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`upsample_bicubic2d_aa_backward_kernel_impl`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`upsample_bicubic2d_aa_kernel_impl`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`upsample_bicubic2d_kernel_impl`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`upsample_bilinear2d_aa_backward_kernel_impl`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`upsample_bilinear2d_aa_kernel_impl`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`upsample_bilinear2d_kernel_impl`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`upsample_bilinear2d_kernel_impl_float`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`upsample_generic_Nd_kernel_impl`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`upsample_linear1d_kernel_impl`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`upsample_nearest1d_kernel_impl`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`upsample_nearest2d_kernel_impl`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`upsample_nearest3d_kernel_impl`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`upsample_trilinear3d_kernel_impl`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)

### Includes

- **`ATen/Context.h`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`ATen/Dispatch.h`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`ATen/Functions.h`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`ATen/Parallel.h`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`ATen/TensorIterator.h`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`ATen/core/Tensor.h`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`ATen/cpu/vec/vec.h`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`ATen/native/UpSample.h`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`ATen/native/cpu/UpSampleKernelAVXAntialias.h`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`ATen/native/cpu/utils.h`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`ATen/ops/empty.h`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`ATen/ops/empty_native.h`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`ATen/ops/ones.h`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`c10/util/irange.h`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)

### Namespaces

- **`REGISTER_DISPATCH`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)
- **`at`**: [UpSampleKernel.cpp_docs.md](./UpSampleKernel.cpp_docs.md)


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
