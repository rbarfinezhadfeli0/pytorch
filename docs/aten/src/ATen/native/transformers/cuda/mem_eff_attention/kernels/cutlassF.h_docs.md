# Documentation: `aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernels/cutlassF.h`

## File Metadata

- **Path**: `aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernels/cutlassF.h`
- **Size**: 25,950 bytes (25.34 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// This file is auto-generated. See "generate_kernels.py"
#pragma once
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>
using namespace PyTorchMemEffAttention;
// ======== bf16 / sm80 ========
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_64x64_rf_sm80(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_64x128_rf_sm80(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_32x128_gmem_sm80(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>::Params p);

template <typename T> void dispatch_cutlassF_bf16_sm80(T cb, int cc) {
    cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, true, true>(), fmha_cutlassF_bf16_aligned_64x64_rf_sm80);
    cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>(), fmha_cutlassF_bf16_aligned_64x128_rf_sm80);
    cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>(), fmha_cutlassF_bf16_aligned_32x128_gmem_sm80);
}

// ======== f16 / sm50 ========
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_64x64_rf_sm50(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_rf_sm50(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_gmem_sm50(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 65536, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_notaligned_64x64_rf_sm50(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 32, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_notaligned_32x128_rf_sm50(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 32, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_notaligned_32x128_gmem_sm50(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 32, 128, 65536, true, true>::Params p);

template <typename T> void dispatch_cutlassF_f16_sm50(T cb, int cc) {
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64, true, true>(), fmha_cutlassF_f16_aligned_64x64_rf_sm50);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 128, true, true>(), fmha_cutlassF_f16_aligned_32x128_rf_sm50);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 65536, true, true>(), fmha_cutlassF_f16_aligned_32x128_gmem_sm50);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 64, 64, 64, true, true>(), fmha_cutlassF_f16_notaligned_64x64_rf_sm50);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 32, 128, 128, true, true>(), fmha_cutlassF_f16_notaligned_32x128_rf_sm50);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 32, 128, 65536, true, true>(), fmha_cutlassF_f16_notaligned_32x128_gmem_sm50);
}

// ======== f16 / sm70 ========
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_64x64_rf_sm70(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_rf_sm70(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_gmem_sm70(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_notaligned_64x64_rf_sm70(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 32, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_notaligned_32x128_rf_sm70(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 32, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_notaligned_32x128_gmem_sm70(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 32, 128, 65536, true, true>::Params p);

template <typename T> void dispatch_cutlassF_f16_sm70(T cb, int cc) {
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, true, true>(), fmha_cutlassF_f16_aligned_64x64_rf_sm70);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, true, true>(), fmha_cutlassF_f16_aligned_32x128_rf_sm70);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, true, true>(), fmha_cutlassF_f16_aligned_32x128_gmem_sm70);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 64, 64, 64, true, true>(), fmha_cutlassF_f16_notaligned_64x64_rf_sm70);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 32, 128, 128, true, true>(), fmha_cutlassF_f16_notaligned_32x128_rf_sm70);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 32, 128, 65536, true, true>(), fmha_cutlassF_f16_notaligned_32x128_gmem_sm70);
}

// ======== f16 / sm75 ========
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_64x64_rf_sm75(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_rf_sm75(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_gmem_sm75(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 65536, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_notaligned_64x64_rf_sm75(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 32, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_notaligned_32x128_rf_sm75(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 32, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_notaligned_32x128_gmem_sm75(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 32, 128, 65536, true, true>::Params p);

template <typename T> void dispatch_cutlassF_f16_sm75(T cb, int cc) {
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64, true, true>(), fmha_cutlassF_f16_aligned_64x64_rf_sm75);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 128, true, true>(), fmha_cutlassF_f16_aligned_32x128_rf_sm75);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 65536, true, true>(), fmha_cutlassF_f16_aligned_32x128_gmem_sm75);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 64, 64, 64, true, true>(), fmha_cutlassF_f16_notaligned_64x64_rf_sm75);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 32, 128, 128, true, true>(), fmha_cutlassF_f16_notaligned_32x128_rf_sm75);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 32, 128, 65536, true, true>(), fmha_cutlassF_f16_notaligned_32x128_gmem_sm75);
}

// ======== f16 / sm80 ========
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_64x64_rf_sm80(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_64x128_rf_sm80(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_gmem_sm80(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>::Params p);

template <typename T> void dispatch_cutlassF_f16_sm80(T cb, int cc) {
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64, true, true>(), fmha_cutlassF_f16_aligned_64x64_rf_sm80);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>(), fmha_cutlassF_f16_aligned_64x128_rf_sm80);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>(), fmha_cutlassF_f16_aligned_32x128_gmem_sm80);
}

// ======== f32 / sm50 ========
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm50, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm50, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_64x64_rf_sm50(typename AttentionKernel<float, cutlass::arch::Sm50, true, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_32x128_rf_sm50(typename AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_32x128_gmem_sm50(typename AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 65536, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm50, false, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm50, false, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_notaligned_64x64_rf_sm50(typename AttentionKernel<float, cutlass::arch::Sm50, false, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm50, false, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm50, false, 32, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_notaligned_32x128_rf_sm50(typename AttentionKernel<float, cutlass::arch::Sm50, false, 32, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm50, false, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm50, false, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_notaligned_32x128_gmem_sm50(typename AttentionKernel<float, cutlass::arch::Sm50, false, 32, 128, 65536, true, true>::Params p);

template <typename T> void dispatch_cutlassF_f32_sm50(T cb, int cc) {
    cb(AttentionKernel<float, cutlass::arch::Sm50, true, 64, 64, 64, true, true>(), fmha_cutlassF_f32_aligned_64x64_rf_sm50);
    cb(AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 128, true, true>(), fmha_cutlassF_f32_aligned_32x128_rf_sm50);
    cb(AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 65536, true, true>(), fmha_cutlassF_f32_aligned_32x128_gmem_sm50);
    cb(AttentionKernel<float, cutlass::arch::Sm50, false, 64, 64, 64, true, true>(), fmha_cutlassF_f32_notaligned_64x64_rf_sm50);
    cb(AttentionKernel<float, cutlass::arch::Sm50, false, 32, 128, 128, true, true>(), fmha_cutlassF_f32_notaligned_32x128_rf_sm50);
    cb(AttentionKernel<float, cutlass::arch::Sm50, false, 32, 128, 65536, true, true>(), fmha_cutlassF_f32_notaligned_32x128_gmem_sm50);
}

// ======== f32 / sm70 ========
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm70, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm70, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_64x64_rf_sm70(typename AttentionKernel<float, cutlass::arch::Sm70, true, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_32x128_rf_sm70(typename AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_32x128_gmem_sm70(typename AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 65536, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm70, false, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm70, false, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_notaligned_64x64_rf_sm70(typename AttentionKernel<float, cutlass::arch::Sm70, false, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm70, false, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm70, false, 32, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_notaligned_32x128_rf_sm70(typename AttentionKernel<float, cutlass::arch::Sm70, false, 32, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm70, false, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm70, false, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_notaligned_32x128_gmem_sm70(typename AttentionKernel<float, cutlass::arch::Sm70, false, 32, 128, 65536, true, true>::Params p);

template <typename T> void dispatch_cutlassF_f32_sm70(T cb, int cc) {
    cb(AttentionKernel<float, cutlass::arch::Sm70, true, 64, 64, 64, true, true>(), fmha_cutlassF_f32_aligned_64x64_rf_sm70);
    cb(AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 128, true, true>(), fmha_cutlassF_f32_aligned_32x128_rf_sm70);
    cb(AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 65536, true, true>(), fmha_cutlassF_f32_aligned_32x128_gmem_sm70);
    cb(AttentionKernel<float, cutlass::arch::Sm70, false, 64, 64, 64, true, true>(), fmha_cutlassF_f32_notaligned_64x64_rf_sm70);
    cb(AttentionKernel<float, cutlass::arch::Sm70, false, 32, 128, 128, true, true>(), fmha_cutlassF_f32_notaligned_32x128_rf_sm70);
    cb(AttentionKernel<float, cutlass::arch::Sm70, false, 32, 128, 65536, true, true>(), fmha_cutlassF_f32_notaligned_32x128_gmem_sm70);
}

// ======== f32 / sm75 ========
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_64x64_rf_sm75(typename AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_32x128_rf_sm75(typename AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_32x128_gmem_sm75(typename AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 65536, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm75, false, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm75, false, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_notaligned_64x64_rf_sm75(typename AttentionKernel<float, cutlass::arch::Sm75, false, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm75, false, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm75, false, 32, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_notaligned_32x128_rf_sm75(typename AttentionKernel<float, cutlass::arch::Sm75, false, 32, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm75, false, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm75, false, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_notaligned_32x128_gmem_sm75(typename AttentionKernel<float, cutlass::arch::Sm75, false, 32, 128, 65536, true, true>::Params p);

template <typename T> void dispatch_cutlassF_f32_sm75(T cb, int cc) {
    cb(AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 64, true, true>(), fmha_cutlassF_f32_aligned_64x64_rf_sm75);
    cb(AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 128, true, true>(), fmha_cutlassF_f32_aligned_32x128_rf_sm75);
    cb(AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 65536, true, true>(), fmha_cutlassF_f32_aligned_32x128_gmem_sm75);
    cb(AttentionKernel<float, cutlass::arch::Sm75, false, 64, 64, 64, true, true>(), fmha_cutlassF_f32_notaligned_64x64_rf_sm75);
    cb(AttentionKernel<float, cutlass::arch::Sm75, false, 32, 128, 128, true, true>(), fmha_cutlassF_f32_notaligned_32x128_rf_sm75);
    cb(AttentionKernel<float, cutlass::arch::Sm75, false, 32, 128, 65536, true, true>(), fmha_cutlassF_f32_notaligned_32x128_gmem_sm75);
}

// ======== f32 / sm80 ========
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_64x64_rf_sm80(typename AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_64x128_rf_sm80(typename AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_32x128_gmem_sm80(typename AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>::Params p);

template <typename T> void dispatch_cutlassF_f32_sm80(T cb, int cc) {
    cb(AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>(), fmha_cutlassF_f32_aligned_64x64_rf_sm80);
    cb(AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 128, true, true>(), fmha_cutlassF_f32_aligned_64x128_rf_sm80);
    cb(AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>(), fmha_cutlassF_f32_aligned_32x128_gmem_sm80);
}


template <typename DT, typename T>
void dispatch_cutlassF(T cb, int cc = 0) {

    if (std::is_same_v<DT, cutlass::bfloat16_t> && 80 <= cc && cc <= 120) {
        dispatch_cutlassF_bf16_sm80(cb, cc);
    }
    if (std::is_same_v<DT, cutlass::half_t> && 50 <= cc && cc < 70) {
        dispatch_cutlassF_f16_sm50(cb, cc);
    }
    if (std::is_same_v<DT, cutlass::half_t> && 70 <= cc && cc < 75) {
        dispatch_cutlassF_f16_sm70(cb, cc);
    }
    if (std::is_same_v<DT, cutlass::half_t> && 75 <= cc && cc < 80) {
        dispatch_cutlassF_f16_sm75(cb, cc);
    }
    if (std::is_same_v<DT, cutlass::half_t> && 80 <= cc && cc <= 120) {
        dispatch_cutlassF_f16_sm80(cb, cc);
    }
    if (std::is_same_v<DT, float> && 50 <= cc && cc < 70) {
        dispatch_cutlassF_f32_sm50(cb, cc);
    }
    if (std::is_same_v<DT, float> && 70 <= cc && cc < 75) {
        dispatch_cutlassF_f32_sm70(cb, cc);
    }
    if (std::is_same_v<DT, float> && 75 <= cc && cc < 80) {
        dispatch_cutlassF_f32_sm75(cb, cc);
    }
    if (std::is_same_v<DT, float> && 80 <= cc && cc <= 120) {
        dispatch_cutlassF_f32_sm80(cb, cc);
    }
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 55 function(s).

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

- `ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h`


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

- **File Documentation**: `cutlassF.h_docs.md`
- **Keyword Index**: `cutlassF.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
