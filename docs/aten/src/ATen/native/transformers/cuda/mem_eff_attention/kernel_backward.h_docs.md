# Documentation: `aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_backward.h`

## File Metadata

- **Path**: `aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_backward.h`
- **Size**: 100,035 bytes (97.69 KB)
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
#pragma once

#include <cmath>
#include <type_traits>
#include <vector>

#include <cuda_fp16.h>
#include <curand_kernel.h>

#include <ATen/cuda/PhiloxUtils.cuh>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/scale_type.h>
#include <cutlass/fast_math.h>
#include <cutlass/functional.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/vector.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/tensor_ref.h>

#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/epilogue/threadblock/epilogue_smem_accumulator.h>
#include <cutlass/epilogue/warp/fragment_iterator_tensor_op.h>
#include <cutlass/epilogue/warp/tile_iterator_tensor_op.h>
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/gemm/threadblock/default_mma.h>
#include <cutlass/gemm/threadblock/default_mma_core_simt.h>
#include <cutlass/gemm/threadblock/default_mma_core_sm70.h>
#include <cutlass/gemm/threadblock/default_mma_core_sm75.h>
#include <cutlass/gemm/threadblock/default_mma_core_sm80.h>
#include <cutlass/integer_subbyte.h>
#include <cutlass/matrix_shape.h>
#include <cutlass/platform/platform.h>
#include <cutlass/transform/threadblock/predicated_tile_iterator.h>
#include <cutlass/transform/threadblock/vector_iterator.h>

#include <ATen/native/transformers/cuda/mem_eff_attention/debug_utils.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/gemm_kernel_utils.h>

#include <ATen/native/transformers/cuda/mem_eff_attention/gemm/custom_mma.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/gemm/find_default_mma.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/gemm/mma_accum_lambda_iterator.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/gemm/mma_from_smem.h>

#include <ATen/native/transformers/cuda/mem_eff_attention/epilogue/epilogue_pipelined.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/iterators/epilogue_predicated_tile_iterator.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/transform/tile_smem_loader.h>

#include <cinttypes>
#include <c10/util/Exception.h>

using namespace gemm_kernel_utils;

namespace PyTorchMemEffAttention {
namespace {

template <typename FragmentType, int32_t kNumThreads>
struct GmemTile {
  /*
    Helper functions to efficient store/load RF to gmem

    GEMM accumulators have a particular format on A100, and
    it takes some compute/shared-memory to rearrange them to
    a RowMajor or ColumnMajor format in global memory through
    an Epilogue. The same complexity goes for loading into RF.

    This class loads/stores RF as they are, and can be used for
    efficient accumulation across gemms for instance:

    ```
    GmemTile tile;
    for (int i = 0; i < N; ++i) {
      // ...

      Fragment accum;
      if (i == 0) {
        accum.clear();
      } else {
        tile.load(accum);
      }
      mma(accum, ...);
      if (i < N-1) {
        // Store for next GEMM
        tile.store(accum);
      } else {
        // Store in tensor (eg RowMajor)
        epilogue(accum);
      }

      // ...
    }
    ```
  */

  // 128bits per thread
  using AccessType = cutlass::Array<float, 4>;
  static constexpr int32_t kBytes = sizeof(AccessType);
  static constexpr int32_t kStride = kNumThreads * AccessType::kElements;
  static constexpr int32_t kNumIters =
      FragmentType::kElements / AccessType::kElements;
  static constexpr int32_t kElementsStored =
      kNumThreads * FragmentType::kElements;
  static_assert(
      FragmentType::kElements % AccessType::kElements == 0,
      "fragment not aligned on 128 bits");

  float* ptr;

  CUTLASS_DEVICE void load(FragmentType& fragment, int thread_id) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kNumIters; ++i) {
      AccessType* __restrict__ gmem_ptr = reinterpret_cast<AccessType*>(
          ptr + thread_id * AccessType::kElements + i * kStride);
      AccessType sub_fragment;
      cutlass::arch::global_load<AccessType, kBytes>(
          sub_fragment, gmem_ptr, true);
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < AccessType::kElements; ++j) {
        fragment[i * AccessType::kElements + j] = sub_fragment[j];
      }
    }
  }

  CUTLASS_DEVICE void store(FragmentType const& fragment, int thread_id) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kNumIters; ++i) {
      AccessType* __restrict__ gmem_ptr = reinterpret_cast<AccessType*>(
          ptr + thread_id * AccessType::kElements + i * kStride);
      AccessType sub_fragment;
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < AccessType::kElements; ++j) {
        sub_fragment[j] = fragment[i * AccessType::kElements + j];
      }
      cutlass::arch::global_store<AccessType, kBytes>(
          sub_fragment, gmem_ptr, true);
    }
  }

  CUTLASS_DEVICE void storeAtomicAdd(
      FragmentType const& fragment,
      int thread_id) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kNumIters; ++i) {
      float* gmem_ptr = ptr + thread_id * AccessType::kElements + i * kStride;
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < AccessType::kElements; ++j) {
        float val = fragment[i * AccessType::kElements + j];
        float* ptr = gmem_ptr + j;
        atomicAdd(ptr, val);
      }
    }
  }
};

struct AtomicLock {
  CUTLASS_DEVICE static void acquire(
      int32_t* lock,
      int set_val,
      int thread_id) {
    if (thread_id == 0) {
      while (atomicCAS(lock, 0 /*cmp*/, set_val /*setval*/) != set_val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(40);
#endif
      }
    }
    __syncthreads();
  }
  CUTLASS_DEVICE static void release(int32_t* lock, int thread_id) {
    if (thread_id == 0) {
      int status = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
      asm volatile("st.global.release.gpu.b32 [%0], %1;\n"
                   :
                   : "l"(lock), "r"(status));
#else
      asm volatile("st.global.cg.b32 [%0], %1;\n" : : "l"(lock), "r"(status));
#endif
    }
  }
};

template <typename scalar_t, typename Arch>
constexpr int getWarpsPerSmBw() {
  bool is_half = !cutlass::platform::is_same<scalar_t, float>::value;
  if (Arch::kMinComputeCapability >= 80) {
    return is_half ? 12 : 8;
  }
  return 8;
}
} // namespace

template <
    // which arch we target (eg `cutlass::arch::Sm80`)
    typename ArchTag_,
    // input/output type
    typename scalar_t_,
    // run optimized kernel because memory accesses will be aligned
    bool kIsAligned_,
    // use dropout if enabled
    bool kApplyDropout_,
    // when doing a GEMM, preload the next one (uses more shmem)
    bool kPreload_,
    // block dimensions
    int kBlockSizeI_,
    int kBlockSizeJ_,
    // upperbound on `max(value.shape[-1], query.shape[-1])`
    int kMaxK_ = (int)cutlass::platform::numeric_limits<uint32_t>::max(),
    // assumes that `cu_seqlen` is None, and
    // (1) `num_queries % kBlockSizeI == 0`
    // (2) `num_keys % kBlockSizeJ == 0`
    bool kKeysQueriesAlignedToBlockSize_ = false>
struct AttentionBackwardKernel {
  enum CustomMaskType {
    NoCustomMask = 0,
    CausalFromTopLeft = 1,
    CausalFromBottomRight = 2,
    NumCustomMaskTypes,
  };
  using scalar_t = scalar_t_;
  using output_t = scalar_t;
  using output_accum_t = float;
  using lse_scalar_t = float;
  using accum_t = float;
  using ArchTag = ArchTag_;
  static constexpr bool kIsAligned = kIsAligned_;
  static constexpr bool kApplyDropout = kApplyDropout_;
  static constexpr bool kPreload = kPreload_;
  static constexpr int kBlockSizeI = kBlockSizeI_;
  static constexpr int kBlockSizeJ = kBlockSizeJ_;
  static constexpr int kMaxK = kMaxK_;
  static constexpr bool kKeysQueriesAlignedToBlockSize =
      kKeysQueriesAlignedToBlockSize_;

  static constexpr int64_t kWarpSize = 32;

  // If this is true, we store and accumulate dK/dV in RF
  // rather than going back to gmem every time
  static constexpr bool kIsHalf = cutlass::sizeof_bits<scalar_t>::value <= 16;
  static constexpr bool kOutputInRF = kIsHalf && kMaxK <= kBlockSizeI;
  static_assert(
      !kPreload ||
          (kIsHalf && ArchTag::kMinComputeCapability >= 80 && kOutputInRF),
      "preload MMA not supported");
  static constexpr bool kPrologueQK = kPreload;
  static constexpr bool kPrologueGV = kPreload;
  static constexpr bool kPrologueDOV = kPreload;
  static constexpr bool kPrologueGQ = kPreload;
  static constexpr bool kPrologueGK = kPreload;

  static constexpr int64_t kNumWarpsPerBlock =
      (kBlockSizeI * kBlockSizeJ) / (32 * 32);

  // Compute delta for the f16 kernels
  // TODO: Figure out why it's slower on the f32 kernels
  // (something due to RF pressure?)
  // TODO: Remove condition on `kOutputInRF` - this is needed to work
  // around a compiler bug on V100, not exactly sure why but I spent
  // too much time on this already. Reproducible with
  // (B, Mq, Mkv, K) = (1, 1, 1, 136) for instance
  static constexpr bool kKernelComputesDelta =
      kIsHalf && (kOutputInRF || ArchTag::kMinComputeCapability != 70);

  // Launch bounds
  static constexpr int64_t kNumThreads = kWarpSize * kNumWarpsPerBlock;
  static constexpr int64_t kMinBlocksPerSm =
      getWarpsPerSmBw<scalar_t, ArchTag>() / kNumWarpsPerBlock;

  using GemmType = DefaultGemmType<ArchTag, scalar_t>;
  using DefaultConfig =
      typename cutlass::gemm::device::DefaultGemmConfiguration<
          typename GemmType::OpClass,
          ArchTag,
          scalar_t,
          scalar_t,
          scalar_t, // ElementC
          accum_t // ElementAccumulator
          >;
  static constexpr auto kOptimalAlignement = cutlass::platform::max(
      DefaultConfig::kAlignmentA,
      DefaultConfig::kAlignmentB);
  static constexpr auto kMinimumAlignment = GemmType::kMinimumAlignment;

  struct MatmulQK {
    /*
    attn_T = k_j @ q_i.transpose(-2, -1) # matmul
    attn_T = (attn_T - logsumexp[i_start:i_end].unsqueeze(1).transpose(-2,
    -1)).exp() # epilogue

    with attn_T.shape = (kBlockSizeJ, kBlockSizeI)
    */
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeJ, kBlockSizeI, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using DefaultMma = typename cutlass::gemm::threadblock::DefaultMma<
        scalar_t, // ElementA
        cutlass::layout::RowMajor, // LayoutA
        kIsAligned ? DefaultConfig::kAlignmentA : GemmType::kMinimumAlignment,
        scalar_t, // ElementB
        cutlass::layout::ColumnMajor, // LayoutB
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment,
        accum_t, // ElementC
        cutlass::layout::RowMajor, // LayoutC
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        DefaultConfig::kStages,
        typename GemmType::Operator,
        false, // AccumulatorsInRowMajor = false,
        cutlass::gemm::SharedMemoryClearOption::kNone>;
    using MmaCore = typename DefaultMma::MmaCore;
    using Mma =
        typename MakeCustomMma<typename DefaultMma::ThreadblockMma, kMaxK>::Mma;

    // used for efficient load of bias tile (Bij) from global memory to shared
    // memory
    using BiasLoader = TileSmemLoader<
        scalar_t,
        // Bij is applied to transposed attn matrix tile (Pij.T). Bij is loaded
        // row-major but needs to have transposed shape so we get the same
        // elements.
        cutlass::MatrixShape<ThreadblockShape::kN, ThreadblockShape::kM>,
        MmaCore::kThreads,
        // input restriction: kv_len has to be a multiple of this value
        128 / cutlass::sizeof_bits<scalar_t>::value>;

    // Epilogue to store to shared-memory in a format that we can use later for
    // the second matmul
    using B2bGemm = typename cutlass::gemm::threadblock::B2bGemm<
        typename Mma::Operator::IteratorC,
        typename Mma::Operator,
        scalar_t,
        WarpShape,
        ThreadblockShape>;
    using AccumLambdaIterator = typename DefaultMmaAccumLambdaIterator<
        typename Mma::Operator::IteratorC,
        accum_t,
        kWarpSize>::Iterator;
    using AccumulatorSharedStorage = typename B2bGemm::AccumulatorSharedStorage;
  };

  struct MatmulGradV {
    /*
    grad_v[j_start:j_end] += attn_T @ do_i # matmul

    Dimensions: (kBlockSizeJ * kNumWarpsPerBlock, kBlockSizeI, K)
    (we might need to iterate multiple times on K)
    */
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeJ, kBlockSizeI, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using InstructionShape = typename GemmType::InstructionShape;

    using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        DefaultConfig::kAlignmentA,
        scalar_t, // ElementB,
        cutlass::layout::RowMajor, // LayoutB,
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment,
        output_t,
        cutlass::layout::RowMajor, // LayoutC,
        accum_t,
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        typename DefaultConfig::EpilogueOutputOp,
        void, // ThreadblockSwizzle - not used
        DefaultConfig::kStages,
        false, // SplitKSerial
        typename GemmType::Operator>;

    // if dropout:
    //   for computing dVj += (Pij.T * Zij) @ dOi
    //   Pij_dropped.T = Pij.T * Zij is computed on the fly as fragments of
    //   Pij.T are loaded in. The reason we do it this way is because Pij.T and
    //   Zij are reused in later steps, while Pij_dropped.T is only needed in
    //   this step. computing Pij_dropped.T on the fly allows us to avoid
    //   keeping all 3 of Pij_dropped.T, Pij.T, and Zij in shared memory at the
    //   same time.
    // if no dropout:
    //   for computing dVj += Pij.T @ dOi
    using WarpIteratorA = typename cutlass::gemm::threadblock::
        DefaultWarpIteratorAFromSharedMemory<
            typename DefaultGemm::Mma::Operator::Shape, // WarpShape
            typename DefaultGemm::Mma::Operator::
                InstructionShape, // InstructionShape
            typename DefaultGemm::Mma::Operator::
                IteratorA, // RegularWarpIterator
            typename DefaultGemm::Mma::Policy // Policy
            >::WarpIterator;
    using DefaultMmaFromSmem =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultGemm::Mma,
            MatmulQK::AccumulatorSharedStorage::Shape::kN,
            WarpIteratorA,
            kApplyDropout>; // kScaleOperandA

    using Mma = typename DefaultMmaFromSmem::Mma;
    using IteratorB = typename Mma::IteratorB;
    using WarpCount = typename Mma::WarpCount;

    // Epilogue
    using DefaultOutputOp = typename DefaultConfig::EpilogueOutputOp;
    using DefaultEpilogue = typename DefaultGemm::Epilogue;
    using OutputTileIterator =
        typename cutlass::epilogue::threadblock::MakePrefetchableIterator<
            typename DefaultEpilogue::OutputTileIterator>::Iterator;
    using AccumTileGmem = GmemTile<typename Mma::FragmentC, (int)kNumThreads>;
  };

  struct MatmulDOIVJ {
    /*
    doi_t_vj = do_i @ v_j.transpose(-2, -1) # matmul
    tmp = (doi_t_vj - Di.unsqueeze(1)) * attn # inplace / epilogue?
    */
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeI, kBlockSizeJ, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;

    using ElementC = output_t;
    using ElementAccum = accum_t;

    // no-op output op - epilogue just stores result to global memory
    using BiasGradEpilogueOutputOp =
        typename cutlass::epilogue::thread::LinearCombination<
            ElementC,
            DefaultConfig::EpilogueOutputOp::kCount,
            typename DefaultConfig::EpilogueOutputOp::ElementAccumulator,
            typename DefaultConfig::EpilogueOutputOp::ElementCompute,
            cutlass::epilogue::thread::ScaleType::Nothing>;

    using DefaultGemm = typename cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA
        cutlass::layout::RowMajor, // LayoutA
        kIsAligned ? DefaultConfig::kAlignmentA : GemmType::kMinimumAlignment,
        scalar_t, // ElementB
        cutlass::layout::ColumnMajor, // LayoutB
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment,
        ElementC, // ElementC
        cutlass::layout::RowMajor, // LayoutC
        ElementAccum, // ElementAccumulator
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        BiasGradEpilogueOutputOp, // EpilogueOutputOp
        void, // ThreadblockSwizzle (not used)
        // multiple preloads, dropout Zij tile, and 3 stages push us over shared
        // memory capacity on A100. set a ceiling on number of stages to save
        // shared memory if dropout is in use.
        kPreload && kApplyDropout && (kBlockSizeI * kBlockSizeJ > 64 * 64)
            ? cutlass::const_min(2, DefaultConfig::kStages)
            : DefaultConfig::kStages, // Stages
        false, // SplitKSerial
        typename GemmType::Operator,
        cutlass::gemm::SharedMemoryClearOption::kNone>;
    using Mma = typename MakeCustomMma<typename DefaultGemm::Mma, kMaxK>::Mma;
    using AccumLambdaIterator = typename DefaultMmaAccumLambdaIterator<
        typename Mma::Operator::IteratorC,
        ElementAccum,
        kWarpSize>::Iterator;

    // epilogue used to write bias gradient, which is just the output of this
    // matmul with some operations applied to the fragment
    using BiasGradEpilogue = typename DefaultGemm::Epilogue;

    // Epilogue to store to shared-memory in a format that we can use later for
    // the second matmul
    using B2bGemm = typename cutlass::gemm::threadblock::B2bGemm<
        typename DefaultGemm::Mma::Operator::IteratorC,
        typename DefaultGemm::Mma::Operator,
        scalar_t,
        WarpShape,
        ThreadblockShape>;
    using AccumulatorSharedStorage = typename B2bGemm::AccumulatorSharedStorage;
  };

  struct MatmulGradQ {
    // grad_q <- tmp @ k_j
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeI, kBlockSizeJ, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using InstructionShape = typename GemmType::InstructionShape;

    using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        DefaultConfig::kAlignmentA,
        scalar_t, // ElementB,
        cutlass::layout::RowMajor, // LayoutB,
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment,
        output_t,
        cutlass::layout::RowMajor, // LayoutC,
        accum_t,
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        typename DefaultConfig::EpilogueOutputOp,
        void, // ThreadblockSwizzle - not used
        DefaultConfig::kStages,
        false, // SplitKSerial
        typename GemmType::Operator>;

    using WarpIteratorA = typename cutlass::gemm::threadblock::
        DefaultWarpIteratorAFromSharedMemory<
            typename DefaultGemm::Mma::Operator::Shape,
            typename DefaultGemm::Mma::Operator::InstructionShape,
            typename DefaultGemm::Mma::Operator::IteratorA,
            typename DefaultGemm::Mma::Policy>::WarpIterator;
    using DefaultMmaFromSmem =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultGemm::Mma,
            MatmulDOIVJ::AccumulatorSharedStorage::Shape::kN,
            WarpIteratorA,
            false>; // kScaleOperandA
    using Mma = typename DefaultMmaFromSmem::Mma;
    using IteratorB = typename Mma::IteratorB;
    using WarpCount = typename Mma::WarpCount;

    // Epilogue
    using DefaultOutputOp = typename DefaultConfig::EpilogueOutputOp;
    using DefaultEpilogue = typename DefaultGemm::Epilogue;
    using OutputTileIterator =
        typename cutlass::epilogue::threadblock::MakePrefetchableIterator<
            typename DefaultEpilogue::OutputTileIterator>::Iterator;
    using AccumTileGmem = GmemTile<typename Mma::FragmentC, (int)kNumThreads>;
  };
  struct MatmulGradK {
    // grad_k <- tmp.transpose(-2, -1) @ q_i
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeJ, kBlockSizeI, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using InstructionShape = typename GemmType::InstructionShape;

    using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        DefaultConfig::kAlignmentA,
        scalar_t, // ElementB,
        cutlass::layout::RowMajor, // LayoutB,
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment,
        output_t,
        cutlass::layout::RowMajor, // LayoutC,
        accum_t,
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        typename DefaultConfig::EpilogueOutputOp,
        void, // ThreadblockSwizzle - not used
        DefaultConfig::kStages,
        false, // SplitKSerial
        typename GemmType::Operator>;

    using WarpIteratorA = typename cutlass::gemm::threadblock::
        DefaultWarpIteratorAFromSharedMemory<
            typename DefaultGemm::Mma::Operator::Shape,
            typename DefaultGemm::Mma::Operator::InstructionShape,
            typename DefaultGemm::Mma::Operator::IteratorA,
            typename DefaultGemm::Mma::Policy>::WarpIterator;
    using DefaultMmaFromSmemN =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultGemm::Mma,
            MatmulQK::AccumulatorSharedStorage::Shape::kN, // kMaxK
            WarpIteratorA,
            false>; // kScaleOperandA
    using DefaultMmaFromSmemT =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultGemm::Mma,
            MatmulDOIVJ::AccumulatorSharedStorage::Shape::kM, // kMaxK
            WarpIteratorA,
            false, // kScaleOperandA
            kPreload>; // kTransposeA
    using DefaultMmaFromSmem = typename cutlass::platform::conditional<
        DefaultMmaFromSmemT::kIsTransposedA,
        DefaultMmaFromSmemT,
        DefaultMmaFromSmemN>::type;
    using Mma = typename DefaultMmaFromSmem::Mma;
    using IteratorB = typename Mma::IteratorB;
    using WarpCount = typename Mma::WarpCount;

    // Epilogue
    using DefaultOutputOp = typename DefaultConfig::EpilogueOutputOp;
    using DefaultEpilogue = typename DefaultGemm::Epilogue;
    using OutputTileIterator =
        typename cutlass::epilogue::threadblock::MakePrefetchableIterator<
            typename DefaultEpilogue::OutputTileIterator>::Iterator;
    using AccumTileGmem = GmemTile<typename Mma::FragmentC, (int)kNumThreads>;
  };

  // NOTE: nvcc 12.4 has correctness errors with this on M60 (sm52)
  // when there is an attention bias. Let's just disable it for now.
  static constexpr auto kMinSm = ArchTag::kMinComputeCapability;
  static constexpr bool kEnableSplitKeys = kMinSm >= 70;

  static constexpr bool kNeedsAccumGradQ = kEnableSplitKeys ||
      !cutlass::platform::is_same<output_accum_t, output_t>::value;
  static constexpr bool kNeedsAccumGradK = !kOutputInRF &&
      !cutlass::platform::is_same<output_accum_t, output_t>::value;
  static constexpr bool kNeedsAccumGradV = !kOutputInRF &&
      !cutlass::platform::is_same<output_accum_t, output_t>::value;

  struct GradQTempStorage {
    int32_t lock;
    int32_t counter;
    int32_t pad[2]; // pad to 128bits
    output_accum_t buffer[MatmulGradQ::AccumTileGmem::kElementsStored];
  };

  struct Params {
    // Input tensors
    const scalar_t* query_ptr = nullptr; // [Mq, nH, K]
    const scalar_t* key_ptr = nullptr; // [Mk, nH, K]
    const scalar_t* value_ptr = nullptr; // [Mk, nH, Kv]
    const scalar_t* bias_ptr = nullptr;
    const lse_scalar_t* logsumexp_ptr = nullptr; // [nH, Mq]
    const scalar_t* output_ptr = nullptr; // [Mq, nH, Kv]
    const scalar_t* grad_output_ptr = nullptr; // [Mq, nH, Kv]
    accum_t* delta_ptr = nullptr; // [nH, Mq]
    const int32_t* cu_seqlens_q_ptr = nullptr;
    const int32_t* cu_seqlens_k_ptr = nullptr;

    // Output tensors
    output_t* grad_query_ptr = nullptr; //  [Mq, nH, K]
    output_t* grad_key_ptr = nullptr; //    [Mk, nH, K]
    output_t* grad_value_ptr = nullptr; //  [Mk, nH, Kv]
    output_t* grad_bias_ptr = nullptr;

    // Accumulators
    output_accum_t* workspace = nullptr; // [Mq, Kq] + [Mkv, Kq] + [Mkv, Kv]
    output_accum_t* workspace_gv =
        nullptr; // (will be calculated by the kernel)
    GradQTempStorage* workspace_gq =
        nullptr; // (will be calculated by the kernel)

    // Sliding window. ignored if == 0
    int32_t window_size = 0;

    // Scale
    accum_t scale = 1.0f;

    // Dimensions/strides
    int32_t head_dim = -1;
    int32_t head_dim_value = -1;
    int32_t num_queries = -1;
    int32_t num_keys = -1;
    int32_t num_heads = -1;
    uint8_t custom_mask_type = NoCustomMask;

    int64_t q_strideM = -1;
    int64_t k_strideM = -1;
    int64_t v_strideM = -1;
    int64_t bias_strideM = 0;
    int64_t gO_strideM = -1;
    int64_t gB_strideM = -1;
    int8_t gQKV_strideM_multiplier = 1; // 3 for packed, 1 otherwise

    at::PhiloxCudaState rng_engine_inputs = {0, 0};

    // RNG sequence offset based on batch_id and head_id
    unsigned long long dropout_batch_head_rng_offset = 0;
    float dropout_prob = 0.0f;

    CUTLASS_HOST_DEVICE int64_t o_strideM() const {
      return head_dim_value * num_heads;
    }
    CUTLASS_HOST_DEVICE int64_t gQ_strideM() const {
      return gQKV_strideM_multiplier * num_heads * head_dim;
    }
    CUTLASS_HOST_DEVICE int64_t gK_strideM() const {
      return gQKV_strideM_multiplier * num_heads * head_dim;
    }
    CUTLASS_HOST_DEVICE int64_t gV_strideM() const {
      return gQKV_strideM_multiplier * num_heads * head_dim_value;
    }

    // Everything below is only used in `advance_to_block`
    // and shouldn't use registers
    int64_t o_strideH = -1;
    int32_t q_strideH = -1;
    int32_t k_strideH = -1;
    int32_t v_strideH = -1;
    int64_t bias_strideH = 0;
    int64_t o_strideB = -1;
    int64_t q_strideB = -1;
    int64_t k_strideB = -1;
    int64_t v_strideB = -1;
    int64_t bias_strideB = 0;
    int64_t lse_strideB = -1;
    int64_t lse_strideH = -1;
    int64_t delta_strideB = -1;
    int64_t delta_strideH = -1;
    int32_t num_batches = -1;
    int16_t num_splits_key = 1; // We use `gridDim.x` inside kernel

    int64_t gO_strideB = 0;
    int64_t gQ_strideB = 0;
    int64_t gK_strideB = 0;
    int64_t gV_strideB = 0;
    int64_t gB_strideB = 0;
    int64_t gO_strideH = 0;
    int64_t gQ_strideH = 0;
    int64_t gK_strideH = 0;
    int64_t gV_strideH = 0;
    int64_t gB_strideH = 0;

    CUTLASS_HOST_DEVICE int16_t num_splits_key_device() const {
#ifdef __CUDA_ARCH__
      return kEnableSplitKeys ? gridDim.x : 1;
#else
      return num_splits_key; // for host-side tests
#endif
    }
    CUTLASS_HOST_DEVICE int16_t split_key_device() const {
#ifdef __CUDA_ARCH__
      return kEnableSplitKeys ? blockIdx.x : 0;
#else
      return 0; // for host-side tests
#endif
    }

    CUTLASS_DEVICE bool advance_to_block() {
      int64_t batch_id = blockIdx.z;
      int32_t head_id = blockIdx.y;

      if (kNeedsAccumGradQ || kNeedsAccumGradK || kNeedsAccumGradV) {
        assert(workspace_size() == 0 || workspace != nullptr);

        workspace += (batch_id * num_heads + head_id) * workspace_strideBH();
        workspace = warp_uniform(workspace);
        workspace_gv = workspace + workspace_elements_gk();
        workspace_gq =
            (GradQTempStorage*)(workspace_gv + workspace_elements_gv());
        if (kEnableSplitKeys) {
          workspace_gv += workspace_elements_gv() * split_key_device() /
              num_splits_key_device();
          workspace += workspace_elements_gk() * split_key_device() /
              num_splits_key_device();
        }
      } else {
        workspace = nullptr;
      }

      // Advance pointers that depend on the total concatenated
      // number of queries, as `num_queries` is modified in the block
      // below
      dropout_batch_head_rng_offset =
          batch_id * (num_heads * num_queries * num_keys) +
          head_id * (num_queries * num_keys);
      logsumexp_ptr += batch_id * lse_strideB + head_id * lse_strideH;

      if (cu_seqlens_q_ptr != nullptr) {
        assert(cu_seqlens_k_ptr != nullptr);
        cu_seqlens_q_ptr += batch_id;
        cu_seqlens_k_ptr += batch_id;
        int32_t q_start = cu_seqlens_q_ptr[0];
        int32_t k_start = cu_seqlens_k_ptr[0];
        int64_t q_next_start = cu_seqlens_q_ptr[1];
        int64_t k_next_start = cu_seqlens_k_ptr[1];
        assert(q_next_start - q_start <= num_queries);
        assert(k_next_start - k_start <= num_keys);
        num_queries = q_next_start - q_start;
        num_keys = k_next_start - k_start;

        // Jump manually
        batch_id = 0;

        query_ptr += q_start * q_strideM;
        key_ptr += k_start * k_strideM;
        value_ptr += k_start * v_strideM;
        assert(bias_ptr == nullptr);
        assert(grad_bias_ptr == nullptr);
        output_ptr += q_start * o_strideM();
        grad_output_ptr += q_start * gO_strideM;
        delta_ptr += q_start;

        grad_query_ptr += q_start * gQ_strideM();
        grad_key_ptr += k_start * gK_strideM();
        grad_value_ptr += k_start * gV_strideM();
      }

      query_ptr += batch_id * q_strideB + head_id * q_strideH;
      key_ptr += batch_id * k_strideB + head_id * k_strideH;
      value_ptr += batch_id * v_strideB + head_id * v_strideH;
      if (bias_ptr != nullptr) {
        bias_ptr += batch_id * bias_strideB + head_id * bias_strideH;
      }
      output_ptr += batch_id * o_strideB + head_id * o_strideH;
      grad_output_ptr += batch_id * gO_strideB + head_id * gO_strideH;
      delta_ptr += batch_id * delta_strideB + head_id * delta_strideH;

      grad_query_ptr += batch_id * gQ_strideB + head_id * gQ_strideH;
      grad_key_ptr += batch_id * gK_strideB + head_id * gK_strideH;
      grad_value_ptr += batch_id * gV_strideB + head_id * gV_strideH;
      if (grad_bias_ptr != nullptr) {
        grad_bias_ptr += batch_id * gB_strideB + head_id * gB_strideH;
      }

      // Some values are modified above
      // Signal to the compiler that they are the same in all threads
      // and can be stored in warp-uniform registers (Sm75+)
      num_queries = warp_uniform(num_queries);
      num_keys = warp_uniform(num_keys);
      custom_mask_type = warp_uniform(custom_mask_type);

      query_ptr = warp_uniform(query_ptr);
      key_ptr = warp_uniform(key_ptr);
      value_ptr = warp_uniform(value_ptr);
      bias_ptr = warp_uniform(bias_ptr);
      logsumexp_ptr = warp_uniform(logsumexp_ptr);
      output_ptr = warp_uniform(output_ptr);
      grad_output_ptr = warp_uniform(grad_output_ptr);
      delta_ptr = warp_uniform(delta_ptr);

      grad_query_ptr = warp_uniform(grad_query_ptr);
      grad_key_ptr = warp_uniform(grad_key_ptr);
      grad_value_ptr = warp_uniform(grad_value_ptr);
      grad_bias_ptr = warp_uniform(grad_bias_ptr);

#if 0
      PRINT_T0("[b:%d h:%d] dp[0]:%f Q:%f K:%f V:%f LSE:%f",
        int(blockIdx.z), int(blockIdx.y),
        float(delta_ptr[0]),
        float(query_ptr[0]), float(key_ptr[0]), float(value_ptr[0]),
        float(logsumexp_ptr[0])
      )
#endif
      return true;
    }

    __host__ dim3 getBlocksGrid() const {
      return dim3(num_splits_key, num_heads, num_batches);
    }
    __host__ dim3 getThreadsGrid() const {
      return dim3(kWarpSize * kNumWarpsPerBlock, 1, 1);
    }
    CUTLASS_HOST_DEVICE int64_t workspace_elements_gk() const {
      if (!kNeedsAccumGradK) {
        return 0;
      }
      return num_splits_key * kBlockSizeJ *
          align_up(head_dim, kBlockSizeI);
    }
    CUTLASS_HOST_DEVICE int64_t workspace_elements_gv() const {
      if (!kNeedsAccumGradV) {
        return 0;
      }
      return num_splits_key * kBlockSizeJ *
          align_up(head_dim_value, kBlockSizeI);
    }
    CUTLASS_HOST_DEVICE int64_t workspace_elements_gq() const {
      if (!kNeedsAccumGradQ) {
        return 0;
      }
      int num_blocks = ceil_div(num_queries, kBlockSizeI);
      int num_cols = ceil_div(head_dim, MatmulGradQ::ThreadblockShape::kN);
      return num_blocks * num_cols * sizeof(GradQTempStorage) /
          sizeof(output_accum_t);
    }
    CUTLASS_HOST_DEVICE int64_t workspace_strideBH() const {
      // Aligned on 128bits
      return align_up(
          workspace_elements_gk() + workspace_elements_gv() +
              workspace_elements_gq(),
          int64_t(4));
    }
    CUTLASS_HOST_DEVICE int64_t workspace_size() const {
      // Returns size of buffer we need to run this kernel
      return num_batches * num_heads * workspace_strideBH() * sizeof(float);
    }
    CUTLASS_HOST_DEVICE bool should_zero_workspace() const {
      return num_splits_key > 1 || window_size > 0;
    }
  };

  // shared storage for keeping Zij matrix. not needed if we aren't using
  // dropout, in which case we use an empty array to save shared memory
  using ZijSharedStorage = typename cutlass::platform::conditional<
      kApplyDropout,
      typename MatmulQK::AccumulatorSharedStorage,
      // dummy shared storage object that takes up no space.
      typename cutlass::gemm::threadblock::AccumulatorSharedStorage<
#ifdef _WIN32
          // windows builds throw the error:
          // "type containing an unknown-size array is not allowed"
          // if we try to make Zij shared storage zero-sized.
          // To get around this just make it sized 1 on windows.
          typename cutlass::gemm::GemmShape<1, 1, 0>,
#else
          typename cutlass::gemm::GemmShape<0, 0, 0>,
#endif
          typename MatmulQK::AccumulatorSharedStorage::Element,
          typename MatmulQK::AccumulatorSharedStorage::Layout,
          typename cutlass::MatrixShape<0, 0>>>::type;

  struct SharedStoragePrologue {
    struct {
      cutlass::Array<accum_t, kBlockSizeI> di; // (do_i * o_i).sum(-1)
      typename MatmulQK::Mma::SharedStorageA mm_qk_k;
    } persistent;
    union {
      struct {
        // part1 - after Q.K / dV / dO.V
        union {
          // 1. efficient load of bias tile Bij, which is then applied to Pij
          typename MatmulQK::BiasLoader::SmemTile bias;
          // 4. store Pij. it is needed:
          // - in dVj += (Pij.T * Zij) @ dOi
          // - in dSij = Pij * (dPij - Di)
          // 6. dVj += (Pij.T * Zij) @ dOi
          // 10. write to fragment
          typename MatmulQK::AccumulatorSharedStorage attn_shared_storage;
        };
        // 5. store Zij. it is needed in dVj += (Pij.T * Zij) @ dOi
        ZijSharedStorage zij;

        union {
          // 2. prologue for dVj
          // 6. workspace for dVj += (Pij.T * Zij) @ dOi
          typename MatmulGradV::Mma::SharedStorage mm_gradV;
          // 7. dVj epilogue
          typename MatmulGradV::DefaultEpilogue::SharedStorage gradV_epilogue;
        };

        // 3. prologue for dPij_dropped
        // 8. used in dPij_dropped = dOi @ Vj.T
        typename MatmulDOIVJ::Mma::SharedStorage mm_doivj;
      } part1;

      struct {
        // part2 - dQ
        union {
          typename MatmulQK::AccumulatorSharedStorage
              tmpT_shared_storage; // (from part1)
          typename MatmulDOIVJ::AccumulatorSharedStorage tmp_shared_storage;
        };
        typename MatmulGradK::Mma::SharedStorage mm_gradK; // (preload)
        typename MatmulGradQ::Mma::SharedStorage mm_gradQ; // (preload)
        union {
          // store dB = dSij to global memory
          typename MatmulDOIVJ::BiasGradEpilogue::SharedStorage gradB_epilogue;
          typename MatmulGradQ::DefaultEpilogue::SharedStorage gradQ_epilogue;
        };

      } part2;

      struct {
        // part3 - after last iteration on dQ's epilogue / dK
        union {
          typename MatmulQK::AccumulatorSharedStorage
              tmpT_shared_storage; // (from part1)
          typename MatmulDOIVJ::AccumulatorSharedStorage tmp_shared_storage;
        };
        typename MatmulGradK::Mma::SharedStorage mm_gradK; // (preload)
        typename MatmulGradQ::DefaultEpilogue::SharedStorage
            gradQ_epilogue_lastIter;

        typename MatmulGradK::DefaultEpilogue::SharedStorage gradK_epilogue;
      } part3;

      struct {
        // part4 - after last iteration on dK's epilogue / preload next K.Q_t
        typename MatmulQK::Mma::SharedStorageB mm_qk_q;

        // If we reach end of current key, dump RF->gmem with "final" epilogues
        typename MatmulGradK::DefaultEpilogue::SharedStorage
            gradK_epilogue_final;
        typename MatmulGradV::DefaultEpilogue::SharedStorage
            gradV_epilogue_final;
      } part4;
    };
    static void print_size() {
      // Field size
#define FSZ(f) int((sizeof(((SharedStoragePrologue*)0)->f)))

      printf("Total smem: %d bytes\n", int(sizeof(SharedStoragePrologue)));
      printf("  persistent: %db\n", FSZ(persistent));
      printf("    mm_qk_k: %db\n", FSZ(persistent.mm_qk_k));
      printf("  part1: %db\n", FSZ(part1));
      printf("    bias: %db\n", FSZ(part1.bias));
      printf("    attn_shared_storage: %db\n", FSZ(part1.attn_shared_storage));
      printf("    zij: %db\n", FSZ(part1.zij));
      printf("    mm_gradV: %db\n", FSZ(part1.mm_gradV));
      printf("    gradV_epilogue: %db\n", FSZ(part1.gradV_epilogue));
      printf("    mm_doivj: %db\n", FSZ(part1.mm_doivj));
      printf("  part2: %db\n", FSZ(part2));
      printf("    tmpT_shared_storage: %db\n", FSZ(part2.tmpT_shared_storage));
      printf("    tmp_shared_storage: %db\n", FSZ(part2.tmp_shared_storage));
      printf("    mm_gradK: %db\n", FSZ(part2.mm_gradK));
      printf("    mm_gradQ: %db\n", FSZ(part2.mm_gradQ));
      printf("    gradB_epilogue: %db\n", FSZ(part2.gradB_epilogue));
      printf("    gradQ_epilogue: %db\n", FSZ(part2.gradQ_epilogue));
      printf("  part3: %db\n", FSZ(part3));
      printf("    tmpT_shared_storage: %db\n", FSZ(part3.tmpT_shared_storage));
      printf("  part4: %db\n", FSZ(part4));
      printf("    mm_qk_q: %db\n", FSZ(part4.mm_qk_q));
      printf(
          "    gradK_epilogue_final: %db\n", FSZ(part4.gradK_epilogue_final));
      printf(
          "    gradV_epilogue_final: %db\n", FSZ(part4.gradV_epilogue_final));
    }
// ===========================================
#define FIELD(INSIDE_STRUCT, FIELDNAME) \
  CUTLASS_DEVICE auto& FIELDNAME() {    \
    return INSIDE_STRUCT.FIELDNAME;     \
  }

    FIELD(persistent, di)
    FIELD(persistent, mm_qk_k)
    FIELD(part1, bias)
    FIELD(part1, attn_shared_storage)
    FIELD(part1, zij)
    FIELD(part1, mm_gradV)
    FIELD(part1, gradV_epilogue)
    FIELD(part1, mm_doivj)
    FIELD(part2, mm_gradK)
    FIELD(part2, mm_gradQ)
    FIELD(part2, gradB_epilogue)
    FIELD(part2, gradQ_epilogue)
    FIELD(part2, tmp_shared_storage)
    FIELD(part3, tmpT_shared_storage)
    FIELD(part3, gradQ_epilogue_lastIter)
    FIELD(part3, gradK_epilogue)
    FIELD(part4, mm_qk_q)
    FIELD(part4, gradK_epilogue_final)
    FIELD(part4, gradV_epilogue_final)
  };

  struct SharedStorageNoPrologue {
    struct {
      cutlass::Array<accum_t, kBlockSizeI> di; // (do_i * o_i).sum(-1)
    } persistent;
    union {
      struct {
        // part1 - Q.K matmul
        typename MatmulQK::Mma::SharedStorageA mm_qk_k;
        typename MatmulQK::Mma::SharedStorageB mm_qk_q;
      } part1;

      struct {
        // part2 - compute gradV
        union {
          // 1. efficient load of bias tile Bij, which is then applied to Pij
          typename MatmulQK::BiasLoader::SmemTile bias;
          // 2. store Pij to shared memory. it is needed:
          // - in this step, where it is used in dVj += (Pij.T * Zij) @ dOi
          // - in next step where it is used in dSij = Pij * (dPij - Di)
          typename MatmulQK::AccumulatorSharedStorage attn_shared_storage;
        };
        // 3. store Zij. it is needed in this step, where it is used
        // to compute Pij_dropped = Pij * Zij on the fly as fragments of Pij are
        // loaded for the computation of dVj.
        ZijSharedStorage zij;

        union {
          typename MatmulGradV::Mma::SharedStorage mm_gradV;
          typename MatmulGradV::DefaultEpilogue::SharedStorage gradV_epilogue;
        };
      } part2;

      struct {
        // part3 - DO.V matmul
        union {
          // first compute dPij = (dOi @ Vj.T) * Zij
          // and dSij = Pij * (dPij - Di)
          struct {
            // (from part2) - Pij for computing dSij = Pij * (dPij - Di)
            typename MatmulQK::AccumulatorSharedStorage attn_shared_storage;
            // matmul to compute dOiVj
            typename MatmulDOIVJ::Mma::SharedStorage mm_doivj;
          };
          // then store dB = dSij to global memory
          typename MatmulDOIVJ::BiasGradEpilogue::SharedStorage gradB_epilogue;
        };
      } part3;

      struct {
        // part4 - compute gradQ
        typename MatmulQK::AccumulatorSharedStorage
            tmpT_shared_storage; // (from part2)
        typename MatmulDOIVJ::AccumulatorSharedStorage tmp_shared_storage;
        union {
          typename MatmulGradQ::Mma::SharedStorage mm_gradQ;
          typename MatmulGradQ::DefaultEpilogue::SharedStorage gradQ_epilogue;
          typename MatmulGradQ::DefaultEpilogue::SharedStorage
              gradQ_epilogue_lastIter;
        };
      } part4;

      struct {
        // part5 - compute gradK
        typename MatmulQK::AccumulatorSharedStorage
            tmpT_shared_storage; // (from part2)
        typename MatmulDOIVJ::AccumulatorSharedStorage tmp_shared_storage;
        union {
          typename MatmulGradK::Mma::SharedStorage mm_gradK;
          typename MatmulGradK::DefaultEpilogue::SharedStorage gradK_epilogue;
        };
      } part5;

      struct {
        // part6 - store RF accumulated into gmem
        typename MatmulGradK::DefaultEpilogue::SharedStorage
            gradK_epilogue_final;
        typename MatmulGradV::DefaultEpilogue::SharedStorage
            gradV_epilogue_final;
      } part6;
    };
    static void print_size() {
#define FIELD_SIZEOF(f) int((sizeof(((SharedStorageNoPrologue*)0)->f)))
      printf("Total smem: %d bytes\n", int(sizeof(SharedStorageNoPrologue)));
      printf("  persistent: %db\n", FIELD_SIZEOF(persistent));
      printf("  part1: %db\n", FIELD_SIZEOF(part1));
      printf("  part2: %db\n", FIELD_SIZEOF(part2));
      printf("  part3: %db\n", FIELD_SIZEOF(part3));
      printf("  part4: %db\n", FIELD_SIZEOF(part4));
      printf("  part5: %db\n", FIELD_SIZEOF(part5));
      printf("  part6: %db\n", FIELD_SIZEOF(part6));
    }
// ===========================================
#define FIELD(INSIDE_STRUCT, FIELDNAME) \
  CUTLASS_DEVICE auto& FIELDNAME() {    \
    return INSIDE_STRUCT.FIELDNAME;     \
  }

    FIELD(persistent, di)
    FIELD(part1, mm_qk_k)
    FIELD(part1, mm_qk_q)
    FIELD(part2, bias)
    FIELD(part2, attn_shared_storage)
    FIELD(part2, zij)
    FIELD(part2, mm_gradV)
    FIELD(part2, gradV_epilogue)
    FIELD(part3, mm_doivj)
    FIELD(part3, gradB_epilogue)
    FIELD(part4, tmpT_shared_storage)
    FIELD(part4, tmp_shared_storage)
    FIELD(part4, mm_gradQ)
    FIELD(part4, gradQ_epilogue)
    FIELD(part4, gradQ_epilogue_lastIter)
    FIELD(part5, mm_gradK)
    FIELD(part5, gradK_epilogue)
    FIELD(part6, gradK_epilogue_final)
    FIELD(part6, gradV_epilogue_final)
  };

  using SharedStorage = typename cutlass::platform::conditional<
      kPreload,
      SharedStoragePrologue,
      SharedStorageNoPrologue>::type;

  struct OutputFragments {
    typename MatmulGradV::Mma::FragmentC gradV;
    typename MatmulGradK::Mma::FragmentC gradK;

    CUTLASS_DEVICE void clear() {
      gradV.clear();
      gradK.clear();
    }
  };

  static bool __host__ check_supported(Params const& p) {
    CHECK_ALIGNED_PTR(p.query_ptr, kMinimumAlignment);
    CHECK_ALIGNED_PTR(p.key_ptr, kMinimumAlignment);
    CHECK_ALIGNED_PTR(p.value_ptr, kMinimumAlignment);
    CHECK_ALIGNED_PTR(p.output_ptr, kMinimumAlignment);
    CHECK_ALIGNED_PTR(p.grad_output_ptr, kMinimumAlignment);
    CHECK_ALIGNED_PTR(p.bias_ptr, kMinimumAlignment);
    TORCH_CHECK(
        p.num_heads <= 1 || p.lse_strideH % 8 == 0,
        "LSE is not correctly aligned (strideH)");
    TORCH_CHECK(
        p.num_batches <= 1 || p.lse_strideB % 8 == 0,
        "LSE is not correctly aligned (strideB)");
    TORCH_CHECK(
        p.num_heads <= 1 || p.q_strideH % kMinimumAlignment == 0,
        "query is not correctly aligned (strideH)");
    TORCH_CHECK(
        p.num_heads <= 1 || p.k_strideH % kMinimumAlignment == 0,
        "key is not correctly aligned (strideH)");
    TORCH_CHECK(
        p.num_heads <= 1 || p.v_strideH % kMinimumAlignment == 0,
        "value is not correctly aligned (strideH)");
    TORCH_CHECK(
        p.num_batches <= 1 || p.q_strideB % kMinimumAlignment == 0,
        "query is not correctly aligned (strideB)");
    TORCH_CHECK(
        p.num_batches <= 1 || p.k_strideB % kMinimumAlignment == 0,
        "key is not correctly aligned (strideB)");
    TORCH_CHECK(
        p.num_batches <= 1 || p.v_strideB % kMinimumAlignment == 0,
        "value is not correctly aligned (strideB)");
    TORCH_CHECK(
        p.q_strideM % kMinimumAlignment == 0,
        "query is not correctly aligned (strideM)");
    TORCH_CHECK(
        p.k_strideM % kMinimumAlignment == 0,
        "key is not correctly aligned (strideM)");
    TORCH_CHECK(
        p.v_strideM % kMinimumAlignment == 0,
        "value is not correctly aligned (strideM)");
    if (p.bias_ptr) {
      TORCH_CHECK(
          p.num_batches <= 1 || p.bias_strideB % kMinimumAlignment == 0,
          "attn_bias is not correctly aligned (strideB). ",
          "attn_bias.stride(0) = ", p.bias_strideB, ", and should be a "
          "multiple of ", kMinimumAlignment, ".");
      TORCH_CHECK(
          p.num_heads <= 1 || p.bias_strideH % kMinimumAlignment == 0,
          "attn_bias is not correctly aligned (strideH) ."
          "attn_bias.stride(1) = ", p.bias_strideH, ", and should be a "
          "multiple of ", kMinimumAlignment, ".");
      TORCH_CHECK(
          p.num_queries <= 1 || p.bias_strideM % kMinimumAlignment == 0,
          "attn_bias is not correctly aligned (strideM). "
          "attn_bias.stride(2) = ", p.bias_strideM, ", and should be a ",
          "multiple of ", kMinimumAlignment, ".");
    }
    if (p.grad_bias_ptr) {
      TORCH_CHECK(
          p.num_batches <= 1 || p.gB_strideB % kMinimumAlignment == 0,
          "attn_bias.grad is not correctly aligned (strideB)");
      TORCH_CHECK(
          p.num_heads <= 1 || p.gB_strideH % kMinimumAlignment == 0,
          "attn_bias.grad is not correctly aligned (strideH)");
      TORCH_CHECK(
          p.gB_strideM % kMinimumAlignment == 0,
          "attn_bias.grad is not correctly aligned (strideM)");
    }
    TORCH_CHECK(
        !(p.cu_seqlens_q_ptr && p.bias_ptr),
        "CuSeqlen + bias not implemented yet");
    TORCH_CHECK(
        p.custom_mask_type < NumCustomMaskTypes,
        "Invalid value for `custom_mask_type`");
    TORCH_CHECK(
        p.dropout_prob <= 1.0f && p.dropout_prob >= 0.0f,
        "Invalid value for `dropout_prob`");
    TORCH_CHECK(
        kApplyDropout || p.dropout_prob == 0.0f,
        "Set `kApplyDropout`=True to support `dropout_prob > 0`");
    TORCH_CHECK(p.head_dim > 0, "Invalid value for `head_dim`");
    TORCH_CHECK(p.head_dim_value > 0, "Invalid value for `head_dim_value`");
    TORCH_CHECK(p.num_queries > 0, "Invalid value for `num_queries`");
    TORCH_CHECK(p.num_keys > 0, "Invalid value for `num_keys`");
    TORCH_CHECK(p.num_heads > 0, "Invalid value for `num_heads`");
    TORCH_CHECK(p.num_batches > 0, "Invalid value for `num_batches`");
    TORCH_CHECK(p.head_dim <= kMaxK, "kMaxK: Expected `head_dim < kMaxK`");
    TORCH_CHECK(
        p.head_dim_value <= kMaxK, "kMaxK: Expected `head_dim_value < kMaxK`");
    if (kKeysQueriesAlignedToBlockSize) {
      TORCH_CHECK(
          p.cu_seqlens_k_ptr == nullptr,
          "This kernel does not support cu_seqlen");
      TORCH_CHECK(
          p.cu_seqlens_q_ptr == nullptr,
          "This kernel does not support cu_seqlen");
      TORCH_CHECK(
          p.num_queries % kBlockSizeI == 0,
          "kKeysQueriesAlignedToBlockSize condition not respected");
      TORCH_CHECK(
          p.num_keys % kBlockSizeJ == 0,
          "kKeysQueriesAlignedToBlockSize condition not respected");
    }
    TORCH_CHECK(
        kEnableSplitKeys || p.num_splits_key == 1, "SplitKeys is disabled");
    TORCH_CHECK(
        p.num_splits_key > 0, "Invalid `num_splits_key` (expected >0)");
    TORCH_CHECK(
        p.num_splits_key <= cutlass::ceil_div(p.num_keys, kBlockSizeJ),
        "Invalid `num_splits_key` (",
        p.num_splits_key,
        ") - too large for `num_keys` = ",
        p.num_keys);
    if (p.window_size != 0) {
      T
```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 160 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `PyTorchMemEffAttention`, `template`, `gemm_kernel_utils`

**Classes/Structs**: `GmemTile`, `loads`, `AtomicLock`, `AttentionBackwardKernel`, `MatmulQK`, `MatmulGradV`, `MatmulDOIVJ`, `MatmulGradQ`, `MatmulGradK`, `GradQTempStorage`, `Params`, `SharedStoragePrologue`, `SharedStorageNoPrologue`, `OutputFragments`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/transformers/cuda/mem_eff_attention`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `cmath`
- `type_traits`
- `vector`
- `cuda_fp16.h`
- `curand_kernel.h`
- `ATen/cuda/PhiloxUtils.cuh`
- `cutlass/cutlass.h`
- `cutlass/epilogue/thread/linear_combination.h`
- `cutlass/epilogue/thread/scale_type.h`
- `cutlass/fast_math.h`
- `cutlass/functional.h`
- `cutlass/gemm/gemm.h`
- `cutlass/layout/matrix.h`
- `cutlass/layout/vector.h`
- `cutlass/numeric_conversion.h`
- `cutlass/numeric_types.h`
- `cutlass/tensor_ref.h`
- `cutlass/epilogue/thread/linear_combination_relu.h`
- `cutlass/epilogue/threadblock/epilogue_smem_accumulator.h`
- `cutlass/epilogue/warp/fragment_iterator_tensor_op.h`
- `cutlass/epilogue/warp/tile_iterator_tensor_op.h`
- `cutlass/gemm/device/default_gemm_configuration.h`
- `cutlass/gemm/kernel/default_gemm.h`
- `cutlass/gemm/threadblock/default_mma.h`
- `cutlass/gemm/threadblock/default_mma_core_simt.h`
- `cutlass/gemm/threadblock/default_mma_core_sm70.h`
- `cutlass/gemm/threadblock/default_mma_core_sm75.h`
- `cutlass/gemm/threadblock/default_mma_core_sm80.h`
- `cutlass/integer_subbyte.h`
- `cutlass/matrix_shape.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`aten/src/ATen/native/transformers/cuda/mem_eff_attention`):

- [`kernel_forward.h_docs.md`](./kernel_forward.h_docs.md)
- [`gemm_kernel_utils.h_docs.md`](./gemm_kernel_utils.h_docs.md)
- [`debug_utils.h_docs.md`](./debug_utils.h_docs.md)
- [`pytorch_utils.h_docs.md`](./pytorch_utils.h_docs.md)


## Cross-References

- **File Documentation**: `kernel_backward.h_docs.md`
- **Keyword Index**: `kernel_backward.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
