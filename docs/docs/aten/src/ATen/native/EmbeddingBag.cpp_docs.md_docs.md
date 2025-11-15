# Documentation: `docs/aten/src/ATen/native/EmbeddingBag.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/EmbeddingBag.cpp_docs.md`
- **Size**: 53,424 bytes (52.17 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/EmbeddingBag.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/EmbeddingBag.cpp`
- **Size**: 71,034 bytes (69.37 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/EmbeddingBag.h>

#include <ATen/native/CPUBlas.h>
#include <ATen/native/NonSymbolicBC.h>

#include <c10/util/irange.h>
#include <c10/util/Half.h>

#ifdef USE_FBGEMM
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wextra-semi")
#include <fbgemm/Fbgemm.h>
#include <fbgemm/FbgemmConvert.h>
C10_DIAGNOSTIC_POP()
#else
#include <caffe2/perfkernels/embedding_lookup_idx.h>
#endif

#include <cstring>
#include <tuple>
#include <utility>
#include <vector>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_embedding_bag.h>
#include <ATen/ops/_embedding_bag_backward_native.h>
#include <ATen/ops/_embedding_bag_dense_backward.h>
#include <ATen/ops/_embedding_bag_dense_backward_native.h>
#include <ATen/ops/_embedding_bag_forward_only.h>
#include <ATen/ops/_embedding_bag_forward_only_native.h>
#include <ATen/ops/_embedding_bag_native.h>
#include <ATen/ops/_embedding_bag_per_sample_weights_backward_native.h>
#include <ATen/ops/_embedding_bag_sparse_backward.h>
#include <ATen/ops/_embedding_bag_sparse_backward_native.h>
#include <ATen/ops/embedding_backward_native.h>
#include <ATen/ops/embedding_bag_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/max.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/zero_native.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

template<typename scalar_t>
scalar_t dot_impl(int64_t n, const scalar_t *x, int64_t incx, const scalar_t *y, int64_t incy);

static void make_offset2bag(const Tensor &offsets, Tensor& offset2bag) {
  offset2bag.index_add_(
      0, offsets, at::ones_like(offsets, LEGACY_CONTIGUOUS_MEMORY_FORMAT)); // offset2bag = [1 0 1 0 1]
  offset2bag[0] -= 1;                     // offset2bag = [0 0 1 0 1]
  offset2bag = offset2bag.cumsum(0, offset2bag.scalar_type());     // offset2bag = [0 0 1 1 2]
}

namespace {

std::pair<c10::MaybeOwned<Tensor>, c10::MaybeOwned<Tensor>> promoteIndicesAndOffsets(
    const Tensor& indices,
    const Tensor& offsets) {
  const auto commonType =
      promoteTypes(offsets.scalar_type(), indices.scalar_type());
  return {
      indices.scalar_type() == commonType ? c10::MaybeOwned<Tensor>::borrowed(indices)
                                          : c10::MaybeOwned<Tensor>::owned(indices.toType(commonType)),
      offsets.scalar_type() == commonType ? c10::MaybeOwned<Tensor>::borrowed(offsets)
                                          : c10::MaybeOwned<Tensor>::owned(offsets.toType(commonType))};
}

// Determines if we can use a fast implementation for index_select_add, which
// is only applicable if special conditions are met
template<typename index_t>
bool is_fast_path_index_select(const Tensor& src, Tensor& output, index_t padding_idx) {
  return (src.scalar_type() == kFloat || src.scalar_type() == kHalf ||
          src.scalar_type() == kBFloat16) &&
      src.strides()[1] == 1 && output.strides()[1] == 1 &&
      padding_idx < static_cast<index_t>(0);
}

// Determines if we can use a fast implementation for index_select_scale_add,
// which is only applicable if special conditions are met
template<typename index_t>
bool is_fast_path_index_select_scale(const Tensor& src, const Tensor& scale, Tensor& output, index_t padding_idx) {
  return (src.scalar_type() == kFloat || src.scalar_type() == kHalf ||
          src.scalar_type() == kBFloat16) &&
      src.strides()[1] == 1 && output.strides()[1] == 1 &&
      scale.strides()[0] == 1 && padding_idx < static_cast<index_t>(0);
}

template<typename index_t>
bool is_fast_path(const Tensor& src, const std::optional<Tensor>& scale, Tensor& output, index_t padding_idx) {
  return (scale.has_value() && scale.value().defined()) ?
         is_fast_path_index_select_scale(src, scale.value(), output, padding_idx) :
         is_fast_path_index_select(src, output, padding_idx);
}

// This function combines index_select (using select_indices as the index) and
// index_add (using add_indices as the index), without creating an intermediary
// tensor to hold the selected embeddings
template <typename data_t, typename index_t>
std::enable_if_t<std::is_same_v<data_t, double>, void>
index_select_add(
    const Tensor& select_indices,
    const Tensor& add_indices,
    const Tensor& src,
    Tensor& output,
    [[maybe_unused]] const Tensor& offsets,
    [[maybe_unused]] bool include_last_offset,
    Tensor& bag_size,
    index_t padding_idx,
    [[maybe_unused]] _EmbeddingBagKernelCache* fbgemm_kernel_cache) {
  TORCH_CHECK(select_indices.numel() == add_indices.numel());
  auto* add_indices_data = add_indices.const_data_ptr<index_t>();
  auto* select_indices_data = select_indices.const_data_ptr<index_t>();
  auto* src_data = src.const_data_ptr<data_t>();
  auto* output_data = output.data_ptr<data_t>();
  index_t* bag_size_data = nullptr;
  if (bag_size.defined()) {
    bag_size_data = bag_size.data_ptr<index_t>();
  }
  auto numel = add_indices.numel();
  int64_t ddim = src.size(1);
  auto vocab_size = src.size(0);
  auto src_stride0 = src.strides()[0];
  auto src_stride1 = src.strides()[1];
  auto output_stride0 = output.strides()[0];
  auto output_stride1 = output.strides()[1];

  for (const auto i : c10::irange(numel)) {
    // We can skip indices equal to padding_idx so they are not included in
    // the reduction
    auto idx = select_indices_data[i];
    TORCH_CHECK(
        idx >= 0 && idx < vocab_size,
        "embedding_bag: Expected idx >= 0 && idx < num_embeddings but found idx to be ",
        idx);
    if (idx != padding_idx) {
      at::native::cpublas::axpy<data_t>(ddim, 1,
              src_data + src_stride0 * idx, src_stride1,
              output_data + output_stride0 * add_indices_data[i], output_stride1);
    } else if (bag_size_data) {
      // Decrement bag_size to reflect that the index is padded
      bag_size_data[add_indices_data[i]]--;
    }
  }
}

namespace {
template <typename index_t>
void fbgemm_spmdm_report_error_(
    int64_t output_size,
    int index_size,
    int64_t N,
    const index_t* offsets,
    const index_t* indices) {
  for (const auto m : c10::irange(output_size)) {
    for (index_t i = offsets[m]; i < offsets[m + 1]; ++i) {
      TORCH_CHECK(i < index_size);
      index_t idx = indices[i];
      TORCH_CHECK(
          0 <= idx && idx < N,
          "Index ",
          i,
          " of input takes value ",
          idx,
          " which is not in the valid range [0, ",
          N,
          ")");
    }
  }
  TORCH_CHECK(
      offsets[output_size] == index_size,
      "Your input appears to be incorrect: the last offset value should be "
       "the size of the indices tensor, but it seems not to be the case.");
}
} // namespace

template <typename data_t, typename index_t>
std::enable_if_t<
    std::is_same_v<data_t, at::Half> || std::is_same_v<data_t, at::BFloat16>,
    void>
index_select_add(
    const Tensor& select_indices,
    const Tensor& add_indices,
    const Tensor& src,
    Tensor& output,
    const Tensor& offsets,
    bool include_last_offset,
    Tensor& bag_size,
    index_t padding_idx,
    _EmbeddingBagKernelCache* fbgemm_kernel_cache) {
  int64_t ddim = src.size(1);
  auto* select_indices_data = select_indices.const_data_ptr<index_t>();
  auto* output_data = output.data_ptr<data_t>();

  if (is_fast_path_index_select(src, output, padding_idx)) {
    auto src_contig = src.contiguous();
    auto* src_data = src_contig.const_data_ptr<data_t>();
    int64_t output_size = offsets.numel() - 1;
    auto* offsets_data = offsets.const_data_ptr<index_t>();
    std::vector<index_t> offsets_include_last;

    if (include_last_offset) {
      output_size = offsets.numel() - 1;
    } else {
      output_size = offsets.numel();
      offsets_include_last.resize(offsets.numel() + 1);
      if (offsets.numel() > 0) {
        std::memcpy(
            offsets_include_last.data(),
            offsets.const_data_ptr<index_t>(),
            sizeof(index_t) * offsets.numel());
      }
      offsets_include_last[offsets.numel()] = select_indices.numel();
      offsets_data = offsets_include_last.data();
    }
#if defined(USE_FBGEMM)
    constexpr bool isbf16 = std::is_same_v<data_t, at::Half> ? false : true;
    auto kernel_16bit_index_t = fbgemm_kernel_cache
        ? fbgemm_kernel_cache
              ->getCallback</* has_weight */ false, index_t, uint16_t>(ddim)
        : fbgemm::GenerateEmbeddingSpMDM<uint16_t, index_t, index_t, uint16_t>(
              /* block_size */ ddim,
              /* has_weight */ false,
              /* normalize_by_lengths */ false,
              /* prefetch */ 16,
              /* is_weight_positional */ false,
              /* use_offsets */ true,
              /* is_bf16_out */ isbf16,
              /* is_bf16_in */ isbf16);
    at::parallel_for(
        0, output_size, 1, [&](index_t start_idx, index_t end_idx) {
          bool success = kernel_16bit_index_t(
              /* output_size */ end_idx - start_idx,
              /* index_size */ offsets_data[end_idx] - offsets_data[start_idx],
              /* data_size */ src.size(0),
              /* input */ reinterpret_cast<const uint16_t*>(src_data),
              /* indices */ select_indices_data + offsets_data[start_idx],
              /* offsets_or_lengths */ offsets_data + start_idx,
              /* weights */ nullptr,
              /* output */
              reinterpret_cast<uint16_t*>(output_data + start_idx * ddim));
          if (!success) {
            fbgemm_spmdm_report_error_(
                end_idx - start_idx,
                offsets_data[end_idx] - offsets_data[start_idx],
                src.size(0),
                offsets_data + start_idx,
                select_indices_data + offsets_data[start_idx]);
          }
        });
#else
    // Initialize the intermediate output buffer to be 0.
    Tensor output_fp32 = at::zeros({output_size, ddim}, output.options().dtype(at::kFloat));
    auto* output_data_fp32 = output_fp32.data_ptr<float>();
    using bVec = vec::Vectorized<BFloat16>;
    using fVec = vec::Vectorized<float>;
    at::parallel_for(
        0, output_size, 1, [&](index_t start_idx, index_t end_idx) {
          caffe2::EmbeddingLookupIdx(
              /*block_size=*/ddim,
              /*output_size=*/end_idx - start_idx,
              /*index_size=*/offsets_data[end_idx] - offsets_data[start_idx],
              /*data_size=*/src.size(0),
              /*input=*/src_data,
              /*indices=*/select_indices_data + offsets_data[start_idx],
              /*offsets=*/offsets_data + start_idx,
              /*weights=*/nullptr,
              /*scale_bias=*/nullptr,
              /*normalize_by_lengths=*/false,
              /*out=*/output_data_fp32 + start_idx * ddim);
          for (int64_t i = start_idx; i < end_idx; i++) {
            // Convert FP32 intermediate buffer result back to 16 bit for
            // output dtype
            if constexpr (std::is_same_v<data_t, at::Half>) {
              // FP16
              for (const auto d : c10::irange(ddim)) {
                (output_data + i * ddim)[d] =
                    static_cast<data_t>((output_data_fp32 + ddim * i)[d]);
              }
            } else {
              // BF16
              int64_t d = 0;
              for (; d < ddim - (ddim % bVec::size()); d += bVec::size()) {
                fVec temp_fp32_0 = fVec::loadu(output_data_fp32 + ddim * i + d);
                fVec temp_fp32_1 =
                    fVec::loadu(output_data_fp32 + ddim * i + d + fVec::size());
                convert_float_bfloat16(temp_fp32_0, temp_fp32_1)
                    .store(output_data + i * ddim + d);
              }
              for (; d < ddim; d++) {
                (output_data + i * ddim)[d] =
                    static_cast<data_t>((output_data_fp32 + ddim * i)[d]);
              }
            }
          }
        });
#endif
  } else {
    TORCH_CHECK(select_indices.numel() == add_indices.numel());
    auto* src_data = src.const_data_ptr<data_t>();
    auto* add_indices_data = add_indices.const_data_ptr<index_t>();
    index_t* bag_size_data = nullptr;
    if (bag_size.defined()) {
      bag_size_data = bag_size.data_ptr<index_t>();
    }
    auto vocab_size = src.size(0);
    auto src_stride0 = src.strides()[0];
    auto src_stride1 = src.strides()[1];
    auto output_stride0 = output.strides()[0];
    auto output_stride1 = output.strides()[1];
    auto numel = add_indices.numel();

    Tensor src_fp32 = at::empty({ddim}, src.options().dtype(at::kFloat));
    auto* src_data_fp32 = src_fp32.mutable_data_ptr<float>();

    // Initialize the intermediate output buffer to be 0.
    Tensor output_fp32 =
        at::zeros({output.size(0), ddim}, output.options().dtype(at::kFloat));
    auto* output_data_fp32 = output_fp32.data_ptr<float>();

    for (const auto i : c10::irange(numel)) {
      // We can skip indices equal to padding_idx so they are not included in
      // the reduction
      auto idx = select_indices_data[i];
      TORCH_CHECK(
          idx >= 0 && idx < vocab_size,
          "embedding_bag: Expected idx >= 0 && idx < num_embeddings but found idx to be ",
          idx);
      if (idx != padding_idx) {
        // Copy src_data + src_stride0 * idx to src_data_fp32
        for (const auto d : c10::irange(ddim)) {
          src_data_fp32[d] = static_cast<float>(
              (src_data + src_stride0 * idx)[d * src_stride1]);
        }
        at::native::cpublas::axpy<float>(
            ddim,
            1,
            src_data_fp32,
            1,
            output_data_fp32 + ddim * add_indices_data[i],
            1);

      } else if (bag_size_data) {
        // Decrement bag_size to reflect that the index is padded
        bag_size_data[add_indices_data[i]]--;
      }
    }
    for (const auto i : c10::irange(output.size(0))) {
      // Convert FP32 intermediate buffer result back to 16 bit for output
      // dtype
      for (const auto d : c10::irange(ddim)) {
        (output_data + output_stride0 * i)[d * output_stride1] =
            static_cast<data_t>((output_data_fp32 + ddim * i)[d]);
      }
    }
  }
}
template<typename data_t, typename index_t>
std::enable_if_t<std::is_same_v<data_t, float>, void>
index_select_add(const Tensor &select_indices,
                             const Tensor &add_indices,
                             const Tensor &src,
                             Tensor &output,
                             const Tensor& offsets,
                             bool include_last_offset,
                             Tensor &bag_size,
                             index_t padding_idx,
                             _EmbeddingBagKernelCache* fbgemm_kernel_cache) {
  int64_t ddim = src.size(1);
  auto* select_indices_data = select_indices.const_data_ptr<index_t>();
  auto* output_data = output.data_ptr<float>();

  if (is_fast_path_index_select(src, output, padding_idx)) {
    auto src_contig = src.contiguous();
    auto* src_data = src_contig.const_data_ptr<float>();
    int64_t output_size = offsets.numel() - 1;
    auto* offsets_data = offsets.const_data_ptr<index_t>();
    std::vector<index_t> offsets_include_last;

    if (include_last_offset) {
      output_size = offsets.numel() - 1;
    } else {
      output_size = offsets.numel();
      offsets_include_last.resize(offsets.numel() + 1);
      if (offsets.numel() > 0) {
        std::memcpy(
            offsets_include_last.data(),
            offsets.const_data_ptr<index_t>(),
            sizeof(index_t) * offsets.numel());
      }
      offsets_include_last[offsets.numel()] = select_indices.numel();
      offsets_data = offsets_include_last.data();
    }

#ifdef USE_FBGEMM
    auto kernel_fp32_index_t =
      fbgemm_kernel_cache ?
      fbgemm_kernel_cache->getCallback</* has_weight */ false, index_t, float>(ddim) :
      fbgemm::GenerateEmbeddingSpMDM<float, index_t, index_t>(
        /* block_size */ddim,
        /* has_weight */false,
        /* normalize_by_lengths */false,
        /* prefetch */16,
        /* is_weight_positional */false,
        /* use_offsets */true
      );
#endif
    at::parallel_for(
        0, output_size, 1, [&](index_t start_idx, index_t end_idx) {
#ifdef USE_FBGEMM
          bool success = kernel_fp32_index_t(
            /* output_size */end_idx - start_idx,
            /* index_size */offsets_data[end_idx] - offsets_data[start_idx],
            /* data_size */src.size(0),
            /* input */src_data,
            /* indices */select_indices_data + offsets_data[start_idx],
            /* offsets_or_lengths */offsets_data + start_idx,
            /* weights */nullptr,
            /* output */output_data + start_idx * ddim);
          if (!success) {
            fbgemm_spmdm_report_error_(
                end_idx - start_idx,
                offsets_data[end_idx] - offsets_data[start_idx],
                src.size(0),
                offsets_data + start_idx,
                select_indices_data + offsets_data[start_idx]);
          }
#else
          caffe2::EmbeddingLookupIdx(
              /*block_size=*/ddim,
              /*output_size=*/end_idx - start_idx,
              /*index_size=*/offsets_data[end_idx] - offsets_data[start_idx],
              /*data_size=*/src.size(0),
              /*input=*/src_data,
              /*indices=*/select_indices_data + offsets_data[start_idx],
              /*offsets=*/offsets_data + start_idx,
              /*weights=*/nullptr,
              /*scale_bias=*/nullptr,
              /*normalize_by_lengths=*/false,
              /*out=*/output_data + start_idx * ddim);
#endif
        });
  } else {
    AT_ASSERT(select_indices.numel() == add_indices.numel());
    auto* src_data = src.const_data_ptr<float>();
    auto* add_indices_data = add_indices.const_data_ptr<index_t>();
    index_t* bag_size_data = nullptr;
    if (bag_size.defined()) {
      bag_size_data = bag_size.data_ptr<index_t>();
    }
    auto vocab_size = src.size(0);
    auto src_stride0 = src.strides()[0];
    auto src_stride1 = src.strides()[1];
    auto output_stride0 = output.strides()[0];
    auto output_stride1 = output.strides()[1];
    auto numel = add_indices.numel();
    for (const auto i : c10::irange(numel)) {
      // We can skip indices equal to padding_idx so they are not included in
      // the reduction
      auto idx = select_indices_data[i];
      TORCH_CHECK(
          idx >= 0 && idx < vocab_size,
          "embedding_bag: Expected idx >= 0 && idx < num_embeddings but found idx to be ",
          idx);
      if (idx != padding_idx) {
        at::native::cpublas::axpy<float>(
            ddim,
            1,
            src_data + src_stride0 * idx,
            src_stride1,
            output_data + output_stride0 * add_indices_data[i],
            output_stride1);
      } else if (bag_size_data) {
        // Decrement bag_size to reflect that the index is padded
        bag_size_data[add_indices_data[i]]--;
      }
    }
  }
}

// This function fuses the following three fns:
// index_select (using select_indices as the index)
// mul (scaling by per_sample_weights)
// index_add (using add_indices as the index)
template <typename data_t, typename index_t>
std::enable_if_t<std::is_same_v<data_t, double>, void>
index_select_scale_add(
    const Tensor& select_indices,
    const Tensor& add_indices,
    const Tensor& scale,
    const Tensor& src,
    Tensor& output,
    [[maybe_unused]] const Tensor& offsets,
    [[maybe_unused]] bool include_last_offset,
    Tensor& bag_size,
    index_t padding_idx,
    [[maybe_unused]] _EmbeddingBagKernelCache* fbgemm_kernel_cache) {
  AT_ASSERT(select_indices.numel() == add_indices.numel());
  auto* add_indices_data = add_indices.const_data_ptr<index_t>();
  auto* select_indices_data = select_indices.const_data_ptr<index_t>();
  auto* src_data = src.const_data_ptr<data_t>();
  auto* output_data = output.data_ptr<data_t>();
  index_t* bag_size_data = nullptr;
  if (bag_size.defined()) {
    bag_size_data = bag_size.data_ptr<index_t>();
  }
  auto numel = add_indices.numel();
  int64_t ddim = src.size(1);
  auto vocab_size = src.size(0);
  auto src_stride0 = src.strides()[0];
  auto src_stride1 = src.strides()[1];
  auto output_stride0 = output.strides()[0];
  auto output_stride1 = output.strides()[1];

  auto* scale_data = scale.const_data_ptr<data_t>();
  auto scale_stride = scale.strides()[0];

  for (const auto i : c10::irange(numel)) {
    // We can skip indices equal to padding_idx so they are not included in
    // the reduction
    auto idx = select_indices_data[i];
    TORCH_CHECK(
        idx >= 0 && idx < vocab_size,
        "embedding_bag: Expected idx >= 0 && idx < num_embeddings but found idx to be ",
        idx);
    if (idx != padding_idx) {
      auto* src_base = src_data + src_stride0 * idx;
      auto* output_base = output_data + output_stride0 * add_indices_data[i];
      auto element_scale = scale_data[i * scale_stride];
      for (const auto j : c10::irange(ddim)) {
        output_base[j * output_stride1] += src_base[j * src_stride1] * element_scale;
      }
    } else if (bag_size_data) {
      // Decrement bag_size to reflect that the index is padded
      bag_size_data[add_indices_data[i]]--;
    }
  }
}

template <typename data_t, typename index_t>
std::enable_if_t<
    std::is_same_v<data_t, at::Half> || std::is_same_v<data_t, at::BFloat16>,
    void>
index_select_scale_add(
    const Tensor& select_indices,
    const Tensor& add_indices,
    const Tensor& scale,
    const Tensor& src,
    Tensor& output,
    const Tensor& offsets,
    bool include_last_offset,
    Tensor& bag_size,
    index_t padding_idx,
    _EmbeddingBagKernelCache* fbgemm_kernel_cache) {
  int64_t ddim = src.size(1);
  auto* scale_data = scale.const_data_ptr<data_t>();
  auto* select_indices_data = select_indices.const_data_ptr<index_t>();
  auto* output_data = output.data_ptr<data_t>();

  if (is_fast_path_index_select_scale(src, scale, output, padding_idx)) {
    auto src_contig = src.contiguous();
    auto* src_data = src_contig.const_data_ptr<data_t>();
    int64_t output_size = offsets.numel() - 1;
    auto* offsets_data = offsets.const_data_ptr<index_t>();
    std::vector<index_t> offsets_include_last;

    if (include_last_offset) {
      output_size = offsets.numel() - 1;
    } else {
      output_size = offsets.numel();
      offsets_include_last.resize(offsets.numel() + 1);
      std::memcpy(
          offsets_include_last.data(),
          offsets.const_data_ptr<index_t>(),
          sizeof(index_t) * offsets.numel());
      offsets_include_last[offsets.numel()] = select_indices.numel();
      offsets_data = offsets_include_last.data();
    }

    Tensor scale_fp32 = at::empty(scale.sizes(), scale.options().dtype(at::kFloat));
    auto* scale_data_fp32 = scale_fp32.mutable_data_ptr<float>();

#if defined(USE_FBGEMM)
    constexpr bool isbf16 = std::is_same_v<data_t, at::Half> ? false : true;
    if constexpr (isbf16) {
      fbgemm::Bfloat16ToFloat_simd(
          reinterpret_cast<const fbgemm::bfloat16*>(scale_data),
          scale_data_fp32,
          scale_fp32.numel());
    } else {
      fbgemm::Float16ToFloat_simd(
          reinterpret_cast<const fbgemm::float16*>(scale_data),
          scale_data_fp32,
          scale_fp32.numel());
    }
    auto kernel_16bit_index_t = fbgemm_kernel_cache
        ? fbgemm_kernel_cache
              ->getCallback</* has_weight */ true, index_t, uint16_t>(ddim)
        : fbgemm::GenerateEmbeddingSpMDM<uint16_t, index_t, index_t, uint16_t>(
              /* block_size */ ddim,
              /* has_weight */ true,
              /* normalize_by_lengths */ false,
              /* prefetch */ 16,
              /* is_weight_positional */ false,
              /* use_offsets */ true,
              /* is_bf16_out */ isbf16,
              /* is_bf16_in */ isbf16);
    at::parallel_for(
        0, output_size, 1, [&](index_t start_idx, index_t end_idx) {
          bool success = kernel_16bit_index_t(
              /* output_size */ end_idx - start_idx,
              /* index_size */ offsets_data[end_idx] - offsets_data[start_idx],
              /* data_size */ src.size(0),
              /* input */ reinterpret_cast<const uint16_t*>(src_data),
              /* indices */ select_indices_data + offsets_data[start_idx],
              /* offsets_or_lengths */ offsets_data + start_idx,
              /* weights */ scale_data_fp32 + offsets_data[start_idx],
              /* output */
              reinterpret_cast<uint16_t*>(output_data + start_idx * ddim));
          if (!success) {
            fbgemm_spmdm_report_error_(
                end_idx - start_idx,
                offsets_data[end_idx] - offsets_data[start_idx],
                src.size(0),
                offsets_data + start_idx,
                select_indices_data + offsets_data[start_idx]);
          }
        });
#else
    // Initialize the intermediate output buffer to be 0.
    Tensor output_fp32 =
        at::zeros({output_size, ddim}, output.options().dtype(at::kFloat));
    auto* output_data_fp32 = output_fp32.data_ptr<float>();
    for (const auto i : c10::irange(scale.numel())) {
      scale_data_fp32[i] = static_cast<float>(scale_data[i]);
    }
    using bVec = vec::Vectorized<BFloat16>;
    using fVec = vec::Vectorized<float>;
    at::parallel_for(
        0, output_size, 1, [&](index_t start_idx, index_t end_idx) {
          caffe2::EmbeddingLookupIdx(
              /*block_size=*/ddim,
              /*output_size=*/end_idx - start_idx,
              /*index_size=*/offsets_data[end_idx] - offsets_data[start_idx],
              /*data_size=*/src.size(0),
              /*input=*/src_data,
              /*indices=*/select_indices_data + offsets_data[start_idx],
              /*offsets=*/offsets_data + start_idx,
              /*weights=*/scale_data_fp32 + offsets_data[start_idx],
              /*scale_bias=*/nullptr,
              /*normalize_by_lengths=*/false,
              /*out=*/output_data_fp32 + start_idx * ddim);
          for (int64_t i = start_idx; i < end_idx; i++) {
            // Convert FP32 intermediate buffer result back to 16 bit for
            // output dtype
            if constexpr (std::is_same_v<data_t, at::Half>) {
              // FP16
              for (const auto d : c10::irange(ddim)) {
                (output_data + i * ddim)[d] =
                    static_cast<data_t>((output_data_fp32 + ddim * i)[d]);
              }
            } else {
              // BF16
              int64_t d = 0;
              for (; d < ddim - (ddim % bVec::size()); d += bVec::size()) {
                fVec temp_fp32_0 = fVec::loadu(output_data_fp32 + ddim * i + d);
                fVec temp_fp32_1 =
                    fVec::loadu(output_data_fp32 + ddim * i + d + fVec::size());
                convert_float_bfloat16(temp_fp32_0, temp_fp32_1)
                    .store(output_data + i * ddim + d);
              }
              for (; d < ddim; d++) {
                (output_data + i * ddim)[d] =
                    static_cast<data_t>((output_data_fp32 + ddim * i)[d]);
              }
            }
          }
        });
#endif
  } else {
    AT_ASSERT(select_indices.numel() == add_indices.numel());
    auto* src_data = src.const_data_ptr<data_t>();
    auto* add_indices_data = add_indices.const_data_ptr<index_t>();
    index_t* bag_size_data = nullptr;
    if (bag_size.defined()) {
      bag_size_data = bag_size.data_ptr<index_t>();
    }
    auto vocab_size = src.size(0);
    auto src_stride0 = src.strides()[0];
    auto src_stride1 = src.strides()[1];
    auto output_stride0 = output.strides()[0];
    auto output_stride1 = output.strides()[1];
    auto scale_stride = scale.strides()[0];
    auto numel = add_indices.numel();

    // Initialize the intermediate output buffer to be 0.
    Tensor output_fp32 =
        at::zeros({output.size(0), ddim}, output.options().dtype(at::kFloat));
    auto* output_data_fp32 = output_fp32.data_ptr<float>();

    for (const auto i : c10::irange(numel)) {
      // We can skip indices equal to padding_idx so they are not included in
      // the reduction
      auto idx = select_indices_data[i];
      TORCH_CHECK(
          idx >= 0 && idx < vocab_size,
          "embedding_bag: Expected idx >= 0 && idx < num_embeddings but found idx to be ",
          idx);
      if (idx != padding_idx) {
        auto* src_base = src_data + src_stride0 * idx;
        auto* output_base_fp32 = output_data_fp32 + ddim * add_indices_data[i];
        auto element_scale = scale_data[i * scale_stride];
        for (const auto j : c10::irange(ddim)) {
          output_base_fp32[j] += static_cast<float>(src_base[j * src_stride1]) *
              static_cast<float>(element_scale);
        }
      } else if (bag_size_data) {
        // Decrement bag_size to reflect that the index is padded
        bag_size_data[add_indices_data[i]]--;
      }
    }
    for (const auto i : c10::irange(output.size(0))) {
      // Convert FP32 intermediate buffer result back to 16 bit for output
      // dtype
      for (const auto d : c10::irange(ddim)) {
        (output_data + output_stride0 * i)[d * output_stride1] =
            static_cast<data_t>((output_data_fp32 + ddim * i)[d]);
      }
    }
  }
}
template<typename data_t, typename index_t>
std::enable_if_t<std::is_same_v<data_t, float>, void>
index_select_scale_add(const Tensor &select_indices,
                                          const Tensor &add_indices,
                                          const Tensor &scale,
                                          const Tensor &src,
                                          Tensor &output,
                                          const Tensor& offsets,
                                          bool include_last_offset,
                                          Tensor &bag_size,
                                          index_t padding_idx,
                                          _EmbeddingBagKernelCache* fbgemm_kernel_cache) {
  int64_t ddim = src.size(1);
  auto* scale_data = scale.const_data_ptr<float>();
  auto* select_indices_data = select_indices.const_data_ptr<index_t>();
  auto* output_data = output.data_ptr<float>();

  if (is_fast_path_index_select_scale(src, scale, output, padding_idx)) {
    auto src_contig = src.contiguous();
    auto* src_data = src_contig.const_data_ptr<float>();
    int64_t output_size = offsets.numel() - 1;
    auto* offsets_data = offsets.const_data_ptr<index_t>();
    std::vector<index_t> offsets_include_last;

    if (include_last_offset) {
      output_size = offsets.numel() - 1;
    } else {
      output_size = offsets.numel();
      offsets_include_last.resize(offsets.numel() + 1);
      std::memcpy(
          offsets_include_last.data(),
          offsets.const_data_ptr<index_t>(),
          sizeof(index_t) * offsets.numel());
      offsets_include_last[offsets.numel()] = select_indices.numel();
      offsets_data = offsets_include_last.data();
    }

#ifdef USE_FBGEMM
    auto kernel_fp32_index_t =
      fbgemm_kernel_cache ?
      fbgemm_kernel_cache->getCallback</* has_weight */ true, index_t, float>(ddim) :
      fbgemm::GenerateEmbeddingSpMDM<float, index_t, index_t>(
        /* block_size */ddim,
        /* has_weight */true,
        /* normalize_by_lengths */false,
        /* prefetch */16,
        /* is_weight_positional */false,
        /* use_offsets */true
      );
#endif
    at::parallel_for(
        0, output_size, 1, [&](index_t start_idx, index_t end_idx) {
#ifdef USE_FBGEMM
          bool success = kernel_fp32_index_t(
            /* output_size */end_idx - start_idx,
            /* index_size */offsets_data[end_idx] - offsets_data[start_idx],
            /* data_size */src.size(0),
            /* input */src_data,
            /* indices */select_indices_data + offsets_data[start_idx],
            /* offsets_or_lengths */offsets_data + start_idx,
            /* weights */scale_data + offsets_data[start_idx],
            /* output */output_data + start_idx * ddim);
          if (!success) {
            fbgemm_spmdm_report_error_(
                end_idx - start_idx,
                offsets_data[end_idx] - offsets_data[start_idx],
                src.size(0),
                offsets_data + start_idx,
                select_indices_data + offsets_data[start_idx]);
          }
#else
          caffe2::EmbeddingLookupIdx(
              /*block_size=*/ddim,
              /*output_size=*/end_idx - start_idx,
              /*index_size=*/offsets_data[end_idx] - offsets_data[start_idx],
              /*data_size=*/src.size(0),
              /*input=*/src_data,
              /*indices=*/select_indices_data + offsets_data[start_idx],
              /*offsets=*/offsets_data + start_idx,
              /*weights=*/scale_data + offsets_data[start_idx],
              /*scale_bias=*/nullptr,
              /*normalize_by_lengths=*/false,
              /*out=*/output_data + start_idx * ddim);
#endif
        });
  } else {
    AT_ASSERT(select_indices.numel() == add_indices.numel());
    auto* src_data = src.const_data_ptr<float>();
    auto* add_indices_data = add_indices.const_data_ptr<index_t>();
    index_t* bag_size_data = nullptr;
    if (bag_size.defined()) {
      bag_size_data = bag_size.data_ptr<index_t>();
    }
    auto vocab_size = src.size(0);
    auto src_stride0 = src.strides()[0];
    auto src_stride1 = src.strides()[1];
    auto output_stride0 = output.strides()[0];
    auto output_stride1 = output.strides()[1];
    auto scale_stride = scale.strides()[0];
    auto numel = add_indices.numel();


    for (const auto i : c10::irange(numel)) {
      // We can skip indices equal to padding_idx so they are not included in
      // the reduction
      auto idx = select_indices_data[i];
      TORCH_CHECK(
          idx >= 0 && idx < vocab_size,
          "embedding_bag: Expected idx >= 0 && idx < num_embeddings but found idx to be ",
          idx);
      if (idx != padding_idx) {
        auto* src_base = src_data + src_stride0 * idx;
        auto* output_base = output_data + output_stride0 * add_indices_data[i];
        auto element_scale = scale_data[i * scale_stride];
        for (const auto j : c10::irange(ddim)) {
          output_base[j * output_stride1] += src_base[j * src_stride1] * element_scale;
        }
      } else if (bag_size_data) {
        // Decrement bag_size to reflect that the index is padded
        bag_size_data[add_indices_data[i]]--;
      }
    }
  }
}

}  // namespace

void check_arguments(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const int64_t mode,
    const std::optional<Tensor>& per_sample_weights,
    bool include_last_offset) {
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarTypes("embedding_bag", indices_arg, {kLong, kInt});
  auto offsets_arg = TensorArg(offsets, "offsets", 1);
  checkScalarTypes("embedding_bag", offsets_arg, {kLong, kInt});
  checkSameType("embedding_bag", indices_arg, offsets_arg);
  auto weight_arg = TensorArg(weight, "weight", 1);
  checkScalarTypes(
      "embedding_bag", weight_arg, {kHalf, kBFloat16, kFloat, kDouble});

  AT_DISPATCH_INDEX_TYPES(offsets.scalar_type(), "_embedding_bag_cpu_impl", [&]() {
    if (offsets.size(0) > 0) {
      index_t offset_0 = offsets.const_data_ptr<index_t>()[0];
      index_t offset_n = offsets.const_data_ptr<index_t>()[offsets.size(0)-1];
      TORCH_CHECK(offset_0 == 0, "offsets[0] has to be 0, i.e., the first sequence "
                                "in the mini-batch has to start from position 0. "
                                "However, got ", offsets[0]);
      TORCH_CHECK(offset_n <= indices.size(0), "offsets[-1] can not "
                  "be greater than input's length ", indices.size(0), " but got offsets[-1] of ",
                  offset_n);
    }
  });

  if (per_sample_weights.has_value() && per_sample_weights.value().defined()) {
    TORCH_CHECK(
        mode == EmbeddingBagMode::SUM,
        "embedding_bag: per_sample_weights only supported with mode='sum'");
    auto per_input_weights_arg = TensorArg(
        per_sample_weights.value(),"per_sample_weights", 1);
    checkSameType("embedding_bag", weight_arg, per_input_weights_arg);
    TORCH_CHECK(per_sample_weights.value().dim() == 1);
    TORCH_CHECK(per_sample_weights.value().numel() == indices.numel());
  }

  if (include_last_offset) {
    TORCH_CHECK(
        offsets.size(0) >= 1,
        "include_last_offset: number of offset should be at least 1");
  }
}

void make_bag_size_out(
    Tensor& bag_size_out,
    const Tensor& offsets,
    const Tensor& indices,
    const int64_t mode,
    const bool include_last_offset,
    const bool requires_grad) {
  if (requires_grad || mode == EmbeddingBagMode::MEAN ||
      mode == EmbeddingBagMode::MAX) {
    auto num_bags = offsets.size(0) - (include_last_offset ? 1 : 0);
    at::native::resize_(bag_size_out, {num_bags}, std::nullopt);
    // Compute this for EmbeddingBagMode::MEAN and EmbeddingBagMode::MAX (latter
    // needed for backwards)
    if (num_bags != 1) {
      bag_size_out.slice(0, 0, bag_size_out.size(0) - 1, 1) =
          offsets.slice(0, 1, num_bags, 1) -
          offsets.slice(0, 0, num_bags - 1, 1);
    }
    if (num_bags > 0) {
      bag_size_out[-1] = indices.size(0) - offsets[num_bags - 1];
    }
  } else {
    at::native::resize_(bag_size_out, offsets.sizes(), std::nullopt);
  }
}

void make_max_indices_out(
    Tensor& max_indices_out,
    const Tensor& weight,
    [[maybe_unused]] const Tensor& indices,
    const Tensor& offsets,
    const Tensor& bag_size,
    const int64_t mode,
    bool include_last_offset) {
  int64_t numBags = offsets.size(0);
  if (mode == EmbeddingBagMode::MAX) {
    if (include_last_offset) {
      TORCH_CHECK(
        numBags >= 1, "include_last_offset: numBags should be at least 1");
      numBags -= 1;
    }
    at::native::resize_(max_indices_out, {numBags, weight.sizes()[1]}, std::nullopt);
    at::native::zero_(max_indices_out);
  } else {
    at::native::resize_(max_indices_out, bag_size.sizes(), std::nullopt);
  }
}

void make_offset2bag_out(
    Tensor& offset2bag,
    Tensor& output,
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const int64_t mode,
    const std::optional<Tensor>& per_sample_weights,
    const int64_t padding_idx) {
  // To save compute, if we are going to go down the fast path case for the 'sum'
  // mode, we skip calculating offset2bag, since it is not going to be used.
  bool fast_path_sum = is_fast_path(weight, per_sample_weights, output, padding_idx);

  if (mode == EmbeddingBagMode::MEAN || mode == EmbeddingBagMode::MAX ||
      !fast_path_sum) {
    at::native::resize_(offset2bag, {indices.size(0) + 1}, std::nullopt);
    at::native::zero_(offset2bag);

    int64_t offsets_size = offsets.size(0);
    bool include_last_offset = (output.size(0) == offsets_size - 1);
    // when include_last_offset is true, ignore the last index in offset.
    // fix segfault when include_last_offset is true and offsets[-1] != indices.size(0)
    // see https://github.com/pytorch/pytorch/issues/89677 for more details.
    Tensor _offsets = offsets;
    if (include_last_offset) {
      _offsets = offsets.narrow(0, 0, offsets_size - 1);
    }
    make_offset2bag(_offsets, offset2bag);
    at::native::resize_(offset2bag, {indices.size(0)}, std::nullopt);
    // only initialize output in slow path
    at::native::zero_(output);
  }
}

static Tensor make_bag_size(
    const Tensor& offsets,
    const Tensor& indices,
    const int64_t mode,
    const bool include_last_offset,
    const bool requires_grad) {
  Tensor bag_size = at::empty(offsets.sizes(), offsets.options());
  make_bag_size_out(bag_size, offsets, indices, mode, include_last_offset, requires_grad);
  return bag_size;
}

static Tensor make_max_indices(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& bag_size,
    const int64_t mode,
    bool include_last_offset) {
  Tensor max_indices = at::empty(bag_size.sizes(), offsets.options());
  make_max_indices_out(max_indices, weight, indices, offsets, bag_size, mode, include_last_offset);
  return max_indices;
}

static Tensor make_offset2bag(
    Tensor& output,
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const int64_t mode,
    const std::optional<Tensor>& per_sample_weights,
    const int64_t padding_idx) {
  Tensor offset2bag = at::empty({0}, offsets.options());
  make_offset2bag_out(offset2bag, output, weight, indices, offsets, mode, per_sample_weights, padding_idx);
  return offset2bag;
}

static Tensor apply_bag_size(
    const int64_t mode,
    Tensor &output,
    const Tensor &bag_size) {
  if (mode == EmbeddingBagMode::MEAN) {
    auto bag_size_ = at::max(bag_size, at::ones_like(bag_size, LEGACY_CONTIGUOUS_MEMORY_FORMAT))
                         .to(output.options())
                         .unsqueeze(1)
                         .expand_as(output);
    output /= bag_size_;
  }
  return output;
}

static Tensor apply_bag_size_backward(
    const int64_t mode,
    Tensor &output,
    const Tensor &offset2bag,
    const Tensor &bag_size) {
  if (mode == EmbeddingBagMode::MEAN) {
    auto inv_bag_size_ = (1 / bag_size.to(output.options()))
                           .unsqueeze(1)
                           .index_select(0, offset2bag);
    output *= inv_bag_size_;
  }
  return output;
}

template <typename scalar_t>
static void embedding_bag_cpu_max_out(
    Tensor* max_indices,
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offset2bag,
    const Tensor& output,
    [[maybe_unused]] bool include_last_offset,
    Tensor& bag_size,
    int64_t padding_idx) {
  int64_t numIndices = indices.numel();
  int64_t featureSize = weight.size(1);
  int64_t vocab_size = weight.size(0);
  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_bag_cpu_max_out", [&] {
    auto* indices_data = indices.const_data_ptr<index_t>();
    auto* offset2bag_data = offset2bag.data_ptr<index_t>();

    index_t* max_indices_data = nullptr;
    int64_t max_indices_stride = 0;
    if (max_indices) {
      max_indices_data = max_indices->data_ptr<index_t>();
      max_indices_stride = max_indices->strides()[0];
    }

    auto* weight_data = weight.const_data_ptr<scalar_t>();
    auto* output_data = output.data_ptr<scalar_t>();
    auto* bag_size_data = bag_size.data_ptr<index_t>();
    auto weight_stride0 = weight.strides()[0];
    auto weight_stride1 = weight.strides()[1];
    auto output_stride = output.strides()[0];
    int64_t numBags = bag_size.size(0);
    std::vector<bool> bag_empty(numBags, true);

    for (const auto i : c10::irange(numIndices)) {
      auto bag = offset2bag_data[i];
      auto word_idx = indices_data[i];
      TORCH_CHECK(
          word_idx >= 0 && word_idx < vocab_size,
          "embedding_bag: Expected idx >= 0 && idx < num_embeddings but found idx to be ",
          word_idx);
      if (word_idx != static_cast<index_t>(padding_idx)) {
        bool is_first_for_bag = bag_empty[bag];
        for (const auto dim : c10::irange(featureSize)) {
          auto& current_item = output_data[output_stride * bag + dim];
          auto weight_item =
              weight_data[weight_stride0 * word_idx + dim * weight_stride1];

          if (is_first_for_bag || (weight_item > current_item)) {
            current_item = weight_item;
            if (max_indices_data) {
              max_indices_data[max_indices_stride * bag + dim] = word_idx;
            }
          }
        }
        if (is_first_for_bag) {
          bag_empty[bag] = false;
        }
      } else {
        // Decrement bag_size to reflect that the index is padded
        bag_size_data[bag]--;
      }
    }
  });
}

void _embedding_bag_cpu_impl_out(Tensor& output, Tensor& offset2bag,
                            Tensor& bag_size, Tensor* max_indices,
                            const Tensor &weight, const Tensor &indices,
                            const Tensor &offsets, const int64_t mode,
                            const std::optional<Tensor>& per_sample_weights,
                            bool include_last_offset, int64_t padding_idx, _EmbeddingBagKernelCache* fbgemm_kernel_cache) {
  if (mode == EmbeddingBagMode::MEAN || mode == EmbeddingBagMode::SUM) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, weight.scalar_type(), "embedding_bag_no_grad_cpu_out",
      [&indices, &offset2bag, &per_sample_weights, &weight, &output, &offsets, &include_last_offset, &mode, &bag_size, &padding_idx, &fbgemm_kernel_cache]() {
      AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_bag_no_grad_cpu_out",
        [&indices, &offset2bag, &per_sample_weights, &weight, &output, &offsets, &include_last_offset, &mode, &bag_size, &padding_idx, &fbgemm_kernel_cache]() {
        if (per_sample_weights.has_value() && per_sample_weights.value().defined()) {
          TORCH_INTERNAL_ASSERT(mode == EmbeddingBagMode::SUM);
          index_select_scale_add<scalar_t, index_t>(
            indices, offset2bag, per_sample_weights.value(), weight, output, offsets, include_last_offset, bag_size, padding_idx, fbgemm_kernel_cache);
        } else {
          index_select_add<scalar_t, index_t>(indices, offset2bag, weight, output, offsets, include_last_offset, bag_size, padding_idx, fbgemm_kernel_cache);
        }
      });
    });
    apply_bag_size(mode, output, bag_size);
    if (mode == EmbeddingBagMode::SUM) {
      // make bag_size output deterministic
      at::native::zero_(bag_size);
    }
    if (max_indices) {
      max_indices->copy_(bag_size);
    }
  } else { // EmbeddingBagMode::MAX
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        weight.scalar_type(),
        "embedding_bag_cpu_max_out",
        [&]() {
          embedding_bag_cpu_max_out<scalar_t>(
              max_indices,
              weight,
              indices,
              offset2bag,
              output,
              include_last_offset,
              bag_size,
              padding_idx);
        });
  }
}

// Assumes all input tensors except for `weight` are contiguous.
// See NOTE [ embedding_bag Native Functions ] in native_functions.yaml for details
static std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_cpu_impl(
    const Tensor& weight,
    const Tensor& indices_,
    const Tensor& offsets_,
    const int64_t mode,
    const Tensor& per_sample_weights,
    bool include_last_offset,
    int64_t padding_idx,
    bool requires_grad) {
  TORCH_CHECK(indices_.dim() == 1 || indices_.dim() == 2,
      "input has to be a 1D or 2D Tensor, but got Tensor of dimension ",
      indices_.dim());
  if (indices_.dim() == 1) {
    TORCH_CHECK(offsets_.dim() == 1,
        "offsets has to be a 1D Tensor, but got Tensor of dimension ",
        offsets_.dim());
  }
  TORCH_CHECK(weight.dim() == 2,
      "weight has to be a 2D Tensor, but got Tensor of dimension ",
      weight.dim());
  auto [indicesMaybeOwned, offsetsMaybeOwned] = promoteIndicesAndOffsets(indices_, offsets_);
  const auto& indices = *indicesMaybeOwned;
  const auto& offsets = *offsetsMaybeOwned;
  check_arguments(weight, indices, offsets, mode, per_sample_weights, include_last_offset);

  Tensor output = at::empty(
      {include_last_offset ? offsets.size(0) - 1 : offsets.size(0),
       weight.sizes()[1]},
      weight.options());

  Tensor offset2bag = make_offset2bag(output, weight, indices, offsets, mode, per_sample_weights, padding_idx);

  Tensor bag_size = make_bag_size(offsets, indices, mode, include_last_offset, requires_grad);

  Tensor max_indices = make_max_indices(weight, indices, offsets, bag_size, mode, include_last_offset);

  _embedding_bag_cpu_impl_out(output, offset2bag,
                          bag_size, &max_indices,
                          weight, indices, offsets,
                          mode, per_sample_weights,
                          include_last_offset, padding_idx);

  return std::make_tuple(std::move(output), std::move(offset2bag), std::move(bag_size), std::move(max_indices));
}

// embedding_bag wrapper to enforce contiguity in tensors other than `weight`.
// This is created to save extra `.contiguous()` call in backward.
// See NOTE [ embedding_bag Native Functions ] in native_functions.yaml for details
std::tuple<Tensor, Tensor, Tensor, Tensor>
embedding_bag(const Tensor &weight, const Tensor &indices,
              const Tensor &offsets, const bool scale_grad_by_freq,
              const int64_t mode, bool sparse, const std::optional<Tensor>& per_sample_weights_opt,
              bool include_last_offset, std::optional<int64_t> padding_idx_opt) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned = at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;
  int64_t padding_idx = -1;

  if (padding_idx_opt.has_value()) {
    auto num_embeddings = weight.size(0);
    padding_idx = padding_idx_opt.value();
    TORCH_CHECK(
      (padding_idx >= -num_embeddings) && (padding_idx < num_embeddings),
      "padding_idx must be within the number of embeddings, -", num_e
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs/aten/src/ATen/native`):

- [`AdaptiveMaxPooling2d.cpp_docs.md_docs.md`](./AdaptiveMaxPooling2d.cpp_docs.md_docs.md)
- [`Distributions.cpp_docs.md_docs.md`](./Distributions.cpp_docs.md_docs.md)
- [`im2col_shape_check.h_docs.md_docs.md`](./im2col_shape_check.h_docs.md_docs.md)
- [`ReduceOps.cpp_kw.md_docs.md`](./ReduceOps.cpp_kw.md_docs.md)
- [`Lerp.cpp_kw.md_docs.md`](./Lerp.cpp_kw.md_docs.md)
- [`CPUFallback.h_docs.md_docs.md`](./CPUFallback.h_docs.md_docs.md)
- [`MetaTensor.cpp_docs.md_docs.md`](./MetaTensor.cpp_docs.md_docs.md)
- [`Correlation.cpp_kw.md_docs.md`](./Correlation.cpp_kw.md_docs.md)
- [`im2col_shape_check.h_kw.md_docs.md`](./im2col_shape_check.h_kw.md_docs.md)
- [`UpSampleNearest2d.cpp_kw.md_docs.md`](./UpSampleNearest2d.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `EmbeddingBag.cpp_docs.md_docs.md`
- **Keyword Index**: `EmbeddingBag.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
