# Documentation: `aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernels/cutlassB.h`

## File Metadata

- **Path**: `aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernels/cutlassB.h`
- **Size**: 96,641 bytes (94.38 KB)
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
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_backward.h>
using namespace PyTorchMemEffAttention;
// ======== f16 / sm70 ========
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 32, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 32, true>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k32_seqaligned_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 32, true>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k32_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 64, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 64, true>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k64_seqaligned_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 64, true>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k64_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 128, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 128, true>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x64_k128_seqaligned_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 128, true>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x64_k128_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 128, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 128, true>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k128_seqaligned_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 128, true>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k128_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x64_k65536_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k65536_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k32_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k64_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 128, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 128, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x64_k128_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 128, 64, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k128_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 128, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x64_k65536_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 128, 64, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k65536_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_64x64_k32_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_64x64_k64_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 128, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 128, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_128x64_k128_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 128, 64, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_64x64_k128_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 128, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_128x64_k65536_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 128, 64, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_64x64_k65536_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_64x64_k32_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_64x64_k64_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 128, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 128, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_128x64_k128_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 128, 64, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_64x64_k128_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 128, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_128x64_k65536_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 128, 64, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_64x64_k65536_dropout_sm70(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 65536>::Params p);

template <typename T> void dispatch_cutlassB_f16_sm70(T cb, int cc) {
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 32, true>(), fmha_cutlassB_f16_aligned_64x64_k32_seqaligned_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 32>(), fmha_cutlassB_f16_aligned_64x64_k32_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 64, true>(), fmha_cutlassB_f16_aligned_64x64_k64_seqaligned_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 64>(), fmha_cutlassB_f16_aligned_64x64_k64_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 128, true>(), fmha_cutlassB_f16_aligned_128x64_k128_seqaligned_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 128>(), fmha_cutlassB_f16_aligned_128x64_k128_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 128, true>(), fmha_cutlassB_f16_aligned_64x64_k128_seqaligned_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 128>(), fmha_cutlassB_f16_aligned_64x64_k128_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 65536>(), fmha_cutlassB_f16_aligned_128x64_k65536_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 64, 64, 65536>(), fmha_cutlassB_f16_aligned_64x64_k65536_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 32>(), fmha_cutlassB_f16_aligned_64x64_k32_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 64>(), fmha_cutlassB_f16_aligned_64x64_k64_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 128, 64, 128>(), fmha_cutlassB_f16_aligned_128x64_k128_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 128>(), fmha_cutlassB_f16_aligned_64x64_k128_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 128, 64, 65536>(), fmha_cutlassB_f16_aligned_128x64_k65536_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 64, 64, 65536>(), fmha_cutlassB_f16_aligned_64x64_k65536_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 32>(), fmha_cutlassB_f16_notaligned_64x64_k32_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 64>(), fmha_cutlassB_f16_notaligned_64x64_k64_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 128, 64, 128>(), fmha_cutlassB_f16_notaligned_128x64_k128_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 128>(), fmha_cutlassB_f16_notaligned_64x64_k128_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 128, 64, 65536>(), fmha_cutlassB_f16_notaligned_128x64_k65536_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 64, 64, 65536>(), fmha_cutlassB_f16_notaligned_64x64_k65536_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 32>(), fmha_cutlassB_f16_notaligned_64x64_k32_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 64>(), fmha_cutlassB_f16_notaligned_64x64_k64_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 128, 64, 128>(), fmha_cutlassB_f16_notaligned_128x64_k128_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 128>(), fmha_cutlassB_f16_notaligned_64x64_k128_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 128, 64, 65536>(), fmha_cutlassB_f16_notaligned_128x64_k65536_dropout_sm70);
    cb(AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 65536>(), fmha_cutlassB_f16_notaligned_64x64_k65536_dropout_sm70);
}

// ======== bf16 / sm80 ========
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 32, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 32, true>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k32_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 32, true>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k32_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 64, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 64, true>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k64_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 64, true>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 64>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k64_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 64, 96>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 64, 96>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x64_k96_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 64, 96>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 128, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 128, true>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k128_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 128, true>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 128>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k128_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 64, 128, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 64, 128, true>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k128_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 64, 128, true>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k128_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 64, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 128, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x64_k65536_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 128, 64, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k65536_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 64, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k32_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 64>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k64_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 128>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k128_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k128_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 64, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 128, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x64_k65536_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 128, 64, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k65536_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 64, 65536>::Params p);

template <typename T> void dispatch_cutlassB_bf16_sm80(T cb, int cc) {
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 32, true>(), fmha_cutlassB_bf16_aligned_64x64_k32_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 32>(), fmha_cutlassB_bf16_aligned_64x64_k32_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 64, true>(), fmha_cutlassB_bf16_aligned_64x64_k64_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 64>(), fmha_cutlassB_bf16_aligned_64x64_k64_sm80);
    if (cc == 86 || cc == 89) cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 64, 96>(), fmha_cutlassB_bf16_aligned_128x64_k96_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 128, true>(), fmha_cutlassB_bf16_aligned_128x128_k128_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 128>(), fmha_cutlassB_bf16_aligned_128x128_k128_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 64, 128, true>(), fmha_cutlassB_bf16_aligned_64x64_k128_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 64, 128>(), fmha_cutlassB_bf16_aligned_64x64_k128_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 128, 64, 65536>(), fmha_cutlassB_bf16_aligned_128x64_k65536_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 64, 65536>(), fmha_cutlassB_bf16_aligned_64x64_k65536_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 32>(), fmha_cutlassB_bf16_aligned_64x64_k32_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 64>(), fmha_cutlassB_bf16_aligned_64x64_k64_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 128>(), fmha_cutlassB_bf16_aligned_128x128_k128_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 64, 128>(), fmha_cutlassB_bf16_aligned_64x64_k128_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 128, 64, 65536>(), fmha_cutlassB_bf16_aligned_128x64_k65536_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 64, 65536>(), fmha_cutlassB_bf16_aligned_64x64_k65536_dropout_sm80);
}

// ======== f16 / sm80 ========
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 32, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 32, true>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k32_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 32, true>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k32_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 64, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 64, true>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k64_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 64, true>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k64_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 64, 96>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 64, 96>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x64_k96_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 64, 96>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 128, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 128, true>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k128_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 128, true>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k128_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 64, 128, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 64, 128, true>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k128_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 64, 128, true>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k128_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 64, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 128, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x64_k65536_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 128, 64, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k65536_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 64, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k32_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k64_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k128_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k128_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 64, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 128, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 128, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x64_k65536_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 128, 64, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k65536_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 64, 65536>::Params p);

template <typename T> void dispatch_cutlassB_f16_sm80(T cb, int cc) {
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 32, true>(), fmha_cutlassB_f16_aligned_64x64_k32_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 32>(), fmha_cutlassB_f16_aligned_64x64_k32_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 64, true>(), fmha_cutlassB_f16_aligned_64x64_k64_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 64>(), fmha_cutlassB_f16_aligned_64x64_k64_sm80);
    if (cc == 86 || cc == 89) cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 64, 96>(), fmha_cutlassB_f16_aligned_128x64_k96_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 128, true>(), fmha_cutlassB_f16_aligned_128x128_k128_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 128>(), fmha_cutlassB_f16_aligned_128x128_k128_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 64, 128, true>(), fmha_cutlassB_f16_aligned_64x64_k128_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 64, 128>(), fmha_cutlassB_f16_aligned_64x64_k128_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 128, 64, 65536>(), fmha_cutlassB_f16_aligned_128x64_k65536_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 64, 65536>(), fmha_cutlassB_f16_aligned_64x64_k65536_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 32>(), fmha_cutlassB_f16_aligned_64x64_k32_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 64>(), fmha_cutlassB_f16_aligned_64x64_k64_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 128>(), fmha_cutlassB_f16_aligned_128x128_k128_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 64, 128>(), fmha_cutlassB_f16_aligned_64x64_k128_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 128, 64, 65536>(), fmha_cutlassB_f16_aligned_128x64_k65536_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 64, 65536>(), fmha_cutlassB_f16_aligned_64x64_k65536_dropout_sm80);
}

// ======== f16 / sm50 ========
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k32_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k64_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k128_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k65536_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k32_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k64_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k128_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k65536_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_64x64_k32_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_64x64_k64_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_64x64_k128_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_64x64_k65536_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_64x64_k32_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_64x64_k64_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_64x64_k128_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_notaligned_64x64_k65536_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 65536>::Params p);

template <typename T> void dispatch_cutlassB_f16_sm50(T cb, int cc) {
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 32>(), fmha_cutlassB_f16_aligned_64x64_k32_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 64>(), fmha_cutlassB_f16_aligned_64x64_k64_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 128>(), fmha_cutlassB_f16_aligned_64x64_k128_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 65536>(), fmha_cutlassB_f16_aligned_64x64_k65536_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 32>(), fmha_cutlassB_f16_aligned_64x64_k32_dropout_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 64>(), fmha_cutlassB_f16_aligned_64x64_k64_dropout_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 128>(), fmha_cutlassB_f16_aligned_64x64_k128_dropout_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, true, false, 64, 64, 65536>(), fmha_cutlassB_f16_aligned_64x64_k65536_dropout_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 32>(), fmha_cutlassB_f16_notaligned_64x64_k32_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 64>(), fmha_cutlassB_f16_notaligned_64x64_k64_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 128>(), fmha_cutlassB_f16_notaligned_64x64_k128_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, false, false, 64, 64, 65536>(), fmha_cutlassB_f16_notaligned_64x64_k65536_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 32>(), fmha_cutlassB_f16_notaligned_64x64_k32_dropout_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 64>(), fmha_cutlassB_f16_notaligned_64x64_k64_dropout_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 128>(), fmha_cutlassB_f16_notaligned_64x64_k128_dropout_sm50);
    cb(AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, false, true, false, 64, 64, 65536>(), fmha_cutlassB_f16_notaligned_64x64_k65536_dropout_sm50);
}

// ======== f32 / sm50 ========
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, false, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_64x64_k32_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, false, 64, 64, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, false, 64, 64, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, false, 64, 64, 64>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_64x64_k64_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, false, 64, 64, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, false, 64, 64, 128>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_64x64_k128_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, false, 64, 64, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, false, 64, 64, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, false, 64, 64, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_64x64_k65536_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, true, false, false, 64, 64, 65536>::Params p);
#if defined(CUDA_VERSION) && CUDA_VERSION == 12040 && !defined(USE_ROCM)
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 32, 32, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 32, 32, 32>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_32x32_k32_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 32, 32, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 32, 32, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 32, 32, 64>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_32x32_k64_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 32, 32, 64>::Params p);
#else
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 64, 64, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 64, 64, 32>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_64x64_k32_dropout_sm50(typename AttentionBackwardKernel<cutlass::arch::Sm50, float, true, true, false, 64, 64, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKerne
```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 178 function(s).

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
- [`cutlassB_f16_aligned_k65536.cu_docs.md`](./cutlassB_f16_aligned_k65536.cu_docs.md)
- [`cutlassB_f16_aligned_k128.cu_docs.md`](./cutlassB_f16_aligned_k128.cu_docs.md)
- [`cutlassF_f16_aligned.cu_docs.md`](./cutlassF_f16_aligned.cu_docs.md)
- [`cutlassB_f32_aligned_k32.cu_docs.md`](./cutlassB_f32_aligned_k32.cu_docs.md)
- [`cutlassB_f32_aligned_k128.cu_docs.md`](./cutlassB_f32_aligned_k128.cu_docs.md)
- [`cutlassB_f16_aligned_k64_dropout.cu_docs.md`](./cutlassB_f16_aligned_k64_dropout.cu_docs.md)


## Cross-References

- **File Documentation**: `cutlassB.h_docs.md`
- **Keyword Index**: `cutlassB.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
