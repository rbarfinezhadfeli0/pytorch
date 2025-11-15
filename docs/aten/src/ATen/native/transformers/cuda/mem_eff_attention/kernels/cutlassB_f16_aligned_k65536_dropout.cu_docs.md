# Documentation: `aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernels/cutlassB_f16_aligned_k65536_dropout.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernels/cutlassB_f16_aligned_k65536_dropout.cu`
- **Size**: 6,572 bytes (6.42 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// This file is auto-generated. See "generate_kernels.py"
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_backward.h>
using namespace PyTorchMemEffAttention;
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 128, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x64_k65536_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 128, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 128, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassB_f16_aligned_128x64_k65536_dropout_sm70` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 128, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x64_k65536_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 128, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 128, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassB_f16_aligned_128x64_k65536_dropout_sm75` is for sm75-sm80, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 128, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x64_k65536_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 128, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ <= 1200
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 128, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassB_f16_aligned_128x64_k65536_dropout_sm80` is for sm80-sm100, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k65536_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassB_f16_aligned_64x64_k65536_dropout_sm50` is for sm50-sm70, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k65536_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassB_f16_aligned_64x64_k65536_dropout_sm70` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k65536_dropout_sm75(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassB_f16_aligned_64x64_k65536_dropout_sm75` is for sm75-sm80, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k65536_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ <= 1200
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassB_f16_aligned_64x64_k65536_dropout_sm80` is for sm80-sm100, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernels`.

## Detailed Analysis

### Code Structure

**Namespaces**: `PyTorchMemEffAttention`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernels`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/transformers/cuda/mem_eff_attention/kernel_backward.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernels`):

- [`cutlassB_f32_notaligned_k65536.cu_docs.md`](./cutlassB_f32_notaligned_k65536.cu_docs.md)
- [`cutlassB_f32_notaligned_k65536_dropout.cu_docs.md`](./cutlassB_f32_notaligned_k65536_dropout.cu_docs.md)
- [`cutlassB_f16_notaligned_k64.cu_docs.md`](./cutlassB_f16_notaligned_k64.cu_docs.md)
- [`cutlassB.h_docs.md`](./cutlassB.h_docs.md)
- [`cutlassB_f16_aligned_k65536.cu_docs.md`](./cutlassB_f16_aligned_k65536.cu_docs.md)
- [`cutlassB_f16_aligned_k128.cu_docs.md`](./cutlassB_f16_aligned_k128.cu_docs.md)
- [`cutlassF_f16_aligned.cu_docs.md`](./cutlassF_f16_aligned.cu_docs.md)
- [`cutlassB_f32_aligned_k32.cu_docs.md`](./cutlassB_f32_aligned_k32.cu_docs.md)
- [`cutlassB_f32_aligned_k128.cu_docs.md`](./cutlassB_f32_aligned_k128.cu_docs.md)
- [`cutlassB_f16_aligned_k64_dropout.cu_docs.md`](./cutlassB_f16_aligned_k64_dropout.cu_docs.md)


## Cross-References

- **File Documentation**: `cutlassB_f16_aligned_k65536_dropout.cu_docs.md`
- **Keyword Index**: `cutlassB_f16_aligned_k65536_dropout.cu_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
