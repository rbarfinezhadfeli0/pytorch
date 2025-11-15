# Documentation: `docs/aten/src/ATen/native/Convolution.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/Convolution.cpp_docs.md`
- **Size**: 53,440 bytes (52.19 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/Convolution.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/Convolution.cpp`
- **Size**: 96,547 bytes (94.28 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/Parallel.h>
#include <ATen/TensorOperators.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/ConvolutionMM3d.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/Pool.h>
#include <ATen/native/cpu/DepthwiseConvKernel.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/native/xnnpack/Engine.h>
#include <c10/core/GradMode.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>
#include <c10/macros/Macros.h>
#include <algorithm>
#include <limits>
#include <utility>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/permute.h>
#endif

#if AT_NNPACK_ENABLED()
#include <nnpack.h>
#endif

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/Utils.h>
#endif

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_conv_depthwise2d.h>
#include <ATen/ops/_convolution.h>
#include <ATen/ops/_convolution_double_backward_native.h>
#include <ATen/ops/_convolution_mode.h>
#include <ATen/ops/_convolution_mode_native.h>
#include <ATen/ops/_convolution_native.h>
#include <ATen/ops/_mps_convolution.h>
#include <ATen/ops/_mps_convolution_transpose.h>
#include <ATen/ops/_nnpack_available.h>
#include <ATen/ops/_nnpack_spatial_convolution.h>
#include <ATen/ops/_slow_conv2d_backward.h>
#include <ATen/ops/_unsafe_view.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/constant_pad_nd.h>
#include <ATen/ops/conv1d_native.h>
#include <ATen/ops/conv2d_native.h>
#include <ATen/ops/conv3d_native.h>
#include <ATen/ops/conv_depthwise3d.h>
#include <ATen/ops/conv_transpose1d_native.h>
#include <ATen/ops/conv_transpose2d_native.h>
#include <ATen/ops/conv_transpose3d_native.h>
#include <ATen/ops/convolution.h>
#include <ATen/ops/convolution_backward_native.h>
#include <ATen/ops/convolution_backward_overrideable.h>
#include <ATen/ops/convolution_backward_overrideable_native.h>
#include <ATen/ops/convolution_native.h>
#include <ATen/ops/convolution_overrideable.h>
#include <ATen/ops/convolution_overrideable_native.h>
#include <ATen/ops/cudnn_convolution.h>
#include <ATen/ops/cudnn_convolution_transpose.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/miopen_convolution.h>
#include <ATen/ops/miopen_convolution_transpose.h>
#include <ATen/ops/miopen_depthwise_convolution.h>
#include <ATen/ops/mkldnn_convolution.h>
#include <ATen/ops/mps_convolution_backward.h>
#include <ATen/ops/mps_convolution_transpose_backward.h>
#include <ATen/ops/slow_conv3d.h>
#include <ATen/ops/slow_conv_dilated2d.h>
#include <ATen/ops/slow_conv_dilated3d.h>
#include <ATen/ops/slow_conv_transpose2d.h>
#include <ATen/ops/slow_conv_transpose3d.h>
#include <ATen/ops/thnn_conv2d.h>
#include <ATen/ops/view_as_real.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#endif

constexpr int MIOPEN_DIM_MAX = 5;

namespace at::native {


static bool conv_benchmark_empty_cache = true;

// Check workload to activate fast depthwise FP16 cudnn conv kernels
template <typename T>
static bool check_cudnn_depthwise_workload(const at::Tensor& input, T stride) {
  auto w = at::symint::size<T>(input, 3);  // same as h
  auto ch = at::symint::size<T>(input, 1);
  auto bs = at::symint::size<T>(input, 0);
  if (stride==1) {
    if (w >= 7) {
      // All batch sizes and nb_channels
      if (w >= 112) {
        return true;
      }

      // large nb_channels
      if (ch >= 1024) {
        // NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if (w >= 56) {
          return true;
        } else if (bs >= 32) {
          return true;
        }
      }

      // batch_size specific
      if (bs >= 128) {
        // NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if (ch >= 512) {
          return true;
        } else if (ch >= 64) {
          if (w >= 14) {
            return true;
          }
        } else if ((ch >= 32) && (w >=28)) {
          return true;
        }
      } else if (bs >= 64) {
        // NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if ((ch >= 256) && (w >= 14)) {
          return true;
        } else if ((ch >= 32) && (w >= 28)) {
          return true;
        }
      } else if (bs >= 32) {
        // NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if ((ch >= 256) && (w >= 14)) {
          return true;
        } else if ((ch >= 128) && (w >= 28)) {
          return true;
        } else if ((ch >= 32) && (w >= 56)) {
          return true;
        }
      } else if (bs >= 16) {
        if ((ch >= 1024) && (w >= 14)) {
          return true;
        }
        // NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if ((ch >= 256) && (w >= 28)) {
          return true;
        } else if ((ch >= 32) && (w >= 56)) {
          return true;
        }
      } else if (bs >= 8) {
        // NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if ((ch >= 512) && (w >= 28)) {
          return true;
        } else if ((ch >= 64) && (w >= 56)) {
          return true;
        }
      }
    }
  } else if (stride==2) {
    if (ch < 256) {
      return false;
    }

    if (w >= 7) {
      if (bs >= 128) {
        // NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if (ch >= 1024) {
          return true;
        } else if ((ch >= 512) && (w >= 14)) {
          return true;
        } else if (w >= 28) {
          return true;
        }
      } else if (bs >= 64) {
        // NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if ((ch >= 512) && (w >= 14)) {
          return true;
        } else if (w >= 28) {
          return true;
        }
      } else if (bs >= 32) {
        // NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if ((ch >= 1024) && (w >= 14)) {
          return true;
        } else if (w >= 28) {
          return true;
        }
      } else if (bs >= 16) {
        // NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if ((ch >= 512) && (w >= 28)) {
          return true;
        } else if (w >= 56) {
          return true;
        }
      } else if (bs >= 8) {
        // NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if ((ch >= 1024) && (w >= 28)) {
          return true;
        } else if (w >= 56) {
          return true;
        }
      } else if (bs >= 1) {
        if ((ch >= 512) && (w >=112)) {
          return true;
        }
      }
    }
  }
  return false;
}

// simplified version for cudnn 8.2 and above
template <typename T>
static bool check_cudnn_depthwise_workload_with_filter(const at::Tensor& input, T stride, const at::Tensor& weight) {
  // 1D conv
  if(at::symint::size<T>(input, 2) == 1 && stride == 1){
    return true;
  }

  // 2d conv
  // only square filters
  if (at::symint::size<T>(weight, 2) != at::symint::size<T>(weight, 3)) return false;
  auto filter = at::symint::size<T>(weight, 3);
  // only 1/3/5 filter
  if (filter != 1 && filter != 3 && filter != 5) return false;
  // we don't enforce square input but only check width to reduce heuristic space
  if (at::symint::size<T>(input, 3) < 7) return false; // min width 7
  auto w = at::symint::size<T>(input, 3);
  // only 1/2 stride, use cudnn for all stride 1
  if (stride == 1) return true;
  if (stride != 2) return false;

  auto ch = at::symint::size<T>(input, 1);
  auto bs = at::symint::size<T>(input, 0);
  // special case since bs1 show good perf in lots of cases
  if (bs == 1) {
    if (filter == 1 && w <= 28) return true;
    if (filter == 3 || filter == 5) return true;
  } else {
    if (filter == 1 && bs <= 16 && ch >= 128 && w <= 7) return true;
    if (filter == 3 || filter == 5) {
      if ((ch >= 512) || (ch >= 256 && w >= 28)) return true;
    }
  }
  return false;
}


#if defined(C10_MOBILE)
static bool xnnpack_use_convolution2d(
    const Tensor& input,
    const Tensor& weight,
    const at::OptionalIntArrayRef bias_sizes_opt,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const int64_t groups,
    const bool transposed) {
  return xnnpack::use_convolution2d(input, weight, bias_sizes_opt, padding, stride, dilation, groups, transposed);
}

static bool xnnpack_use_convolution2d(
    const Tensor& input,
    const Tensor& weight,
    const at::OptionalSymIntArrayRef bias_sizes_opt,
    const SymIntArrayRef padding,
    const SymIntArrayRef stride,
    const SymIntArrayRef dilation,
    const c10::SymInt groups,
    const bool transposed) {
  // Never use xnnpack for symbolic tracing
  return false;
}
#endif

// This struct is templated so that we can run backend selection in a dynamic
// shapes context; all of the real kernel selection in eager mode runs with
// int64_t
template <typename T>
struct ConvParams {
  std::vector<T> stride;
  std::vector<T> padding;
  std::vector<T> dilation;
  bool transposed{};
  std::vector<T> output_padding;
  T groups{};
  bool benchmark{};
  bool deterministic{};
  bool cudnn_enabled{};
  bool allow_tf32{};

  bool is_strided() const {
    return std::any_of(
      stride.cbegin(), stride.cend(), [](const T& s) { return s != 1; });
  }

  bool is_dilated() const {
    return std::any_of(
      dilation.cbegin(), dilation.cend(), [](const T& d) { return d != 1; });
  }

  bool is_padded() const {
    return std::any_of(
      padding.cbegin(), padding.cend(), [](const T& p) { return p != 0; });
  }

  bool is_output_padding_neg() const {
    return std::any_of(
      output_padding.cbegin(),
      output_padding.cend(),
      [](const T& p) { return p < 0; });
  }

  bool is_output_padding_big() const {
    // Revisit this with std::views::zip at C++20.
    for (auto i: c10::irange(output_padding.size())) {
      if (output_padding[i] >= stride[i]) {
        return true;
      }
    }
    return false;
  }

  bool is_padding_neg() const {
    return std::any_of(
      padding.cbegin(), padding.cend(), [](const T& p) { return p < 0; });
  }

  bool is_dilation_neg() const {
    return std::any_of(
      dilation.cbegin(), dilation.cend(), [](const T& d) { return d < 0; });
  }

  bool is_stride_nonpos() const {
    return std::any_of(
      stride.cbegin(), stride.cend(), [](const T& s) { return s <= 0; });
  }

  void view1d_as_2d() {
    if (stride.size() == 1) {
      stride.insert(stride.begin(), 1);
      padding.insert(padding.begin(), 0);
      dilation.insert(dilation.begin(), 1);
      output_padding.insert(output_padding.begin(), 0);
    }
  }

  bool use_cpu_depthwise3x3_winograd(const at::Tensor& input, const at::Tensor& weight, const std::optional<at::Tensor>& bias) const {
#if defined(__ARM_NEON__) || (defined(__riscv_v_intrinsic) && __riscv_v_intrinsic>=12000)
    // Currently only 3x3 depthwise convolutions on tensors of float are supported.
    return (input.ndimension() == 4) &&
           (at::symint::size<T>(input, 1) == groups) &&
           (weight.ndimension() == 4 ) &&
           (at::symint::size<T>(weight, 0) % at::symint::size<T>(input, 1) == 0) &&
           (at::symint::size<T>(weight, 1) == 1) &&
           (at::symint::size<T>(weight, 2) == 3) &&
           (at::symint::size<T>(weight, 3) == 3) &&
           (input.device().is_cpu()) &&
           (input.scalar_type() == at::kFloat) &&
           input.is_contiguous() &&
           (weight.device().is_cpu()) &&
           (weight.scalar_type() == at::kFloat) &&
           weight.is_contiguous() &&
           (!bias.has_value() || bias->is_contiguous()) &&
           !is_strided() &&
           !is_dilated() &&
           !transposed;
#else
    return false;
#endif
  }

  bool needs_64bit_indexing_no_split(const at::Tensor& input, const at::Tensor& weight) const {
    constexpr int64_t int_max = std::numeric_limits<int>::max();
    auto numel_input = at::symint::numel<T>(input);
    // empty input
    if (numel_input == 0) {
      return false;
    }
    // input size can not be reduced to the range of int by splitting the batch dim
    auto n = at::symint::size<T>(input, 0);
    if (numel_input / n > int_max) {
      return true;
    }
    // output size can not be reduced to the range of int by splitting the batch dim
    T outsize = 1;
    if (transposed) {
      auto o = conv_input_size(at::symint::sizes<T>(input), at::symint::sizes<T>(weight), padding, output_padding, stride, dilation, groups);
      outsize = c10::multiply_integers(o.begin() + 1, o.end());
    } else {
      auto o = conv_output_size(at::symint::sizes<T>(input), at::symint::sizes<T>(weight), padding, stride, dilation);
      outsize = c10::multiply_integers(o.begin() + 1, o.end());
    }
    return outsize > int_max;
  }

  bool use_cudnn(const at::Tensor& input, const at::Tensor& weight) const {
  // Note [Mobile check segfaults]
  // cudnn and miopen are guaranteed not to be on mobile, and T102591915 / T110194934 suggest
  // that maybe the compiledWithCuDNN() check sometimes segfaults (though I can't imagine how)
#if !defined(C10_MOBILE)
    if (!detail::getCUDAHooks().compiledWithCuDNN() || !input.is_cuda() || !cudnn_enabled) {
      return false;
    }
    static long cudnn_version = detail::getCUDAHooks().versionRuntimeCuDNN();
    // broken on cuDNN 9.8 - 9.14
    if (cudnn_version >= 90800 && cudnn_version < 91500) {
      if (cudnn_conv_suggest_memory_format(input, weight) == at::MemoryFormat::Contiguous &&
          (input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf) &&
          weight.dim() == 5) {
        for (int i = 2; i < weight.dim(); i++) {
          if (weight.size(i) != 1) {
            return false;
          }
        }
      }
    }
    if (needs_64bit_indexing_no_split(input, weight)) {
      if (!(cudnn_version >= 90300 && at::native::cudnnv8_enabled_check_debug())) {
        TORCH_WARN_ONCE("cuDNN cannot be used for large non-batch-splittable convolutions"
                        " if the V8 API is not enabled or before cuDNN version 9.3+."
                        " Consider upgrading cuDNN and/or enabling the V8 API for better efficiency.");
        return false;
      }
    }
    if (input.scalar_type() == at::kBFloat16 || weight.scalar_type() == at::kBFloat16) {
      if (!(detail::getCUDAHooks().supportsBFloat16ConvolutionWithCuDNNv8() && at::native::cudnnv8_enabled_check_debug())) {
        return false;
      }
    }
    if (cudnn_conv_suggest_memory_format(input, weight) == at::MemoryFormat::Contiguous) {
      if (is_dilated()) {
        return detail::getCUDAHooks().supportsDilatedConvolutionWithCuDNN() && !is_output_padding_big();
      }
    }
    return !is_output_padding_big();
#else
    return false;
#endif
  }

  // Use cudnn for FP16 depthwise convolutions
  bool use_cudnn_depthwise(const at::Tensor& input, const at::Tensor& weight) const  {
    if (!cudnn_enabled || !detail::getCUDAHooks().compiledWithCuDNN() || !input.is_cuda()) {
      return false;
    }
    // native kernel doesn't support 64-bit non-splittable case
    if (!(canUse32BitIndexMath(input) && canUse32BitIndexMath(weight))) {
      static long cudnn_version = detail::getCUDAHooks().compiledWithCuDNN() ? detail::getCUDAHooks().versionRuntimeCuDNN() : -1;
      // TODO(eqy): remove this once cuDNN fixes 64-bit depthwise support, first broken in 9.11x
      if (cudnn_conv_suggest_memory_format(input, weight) != at::MemoryFormat::Contiguous) {
        if (cudnn_version < 0 || cudnn_version > 91000) {
          return false;
        }
      }

      if (!(cudnn_version >= 90300 && at::native::cudnnv8_enabled_check_debug())) {
        TORCH_WARN_ONCE("cuDNN cannot be used for large non-batch-splittable convolutions"
                        " if the V8 API is not enabled or before cuDNN version 9.3+."
                        " Upgrade cuDNN or enable the V8 API to use cuDNN for 64-bit depthwise convolutions.");
        return false;
      } else {
        return true;
      }
    }
    if (cudnn_conv_suggest_memory_format(input, weight) != at::MemoryFormat::Contiguous) {
      // always use cudnn_depthwise for channels_last format
      return true;
    }
    if (detail::getCUDAHooks().supportsDepthwiseConvolutionWithCuDNN()) {
      bool kernel_cond =  (use_cudnn(input, weight) &&
                           input.scalar_type() == kHalf && // only for FP16
                           weight.scalar_type() == kHalf &&
                           is_depthwise(input, weight) &&
                           input.ndimension() == 4 &&   // TODO: 5-D contiguous depthwise is not supported yet, need benchmarks
                           !is_dilated() && // no dilation supported
                           (stride[0] == stride[1] || at::symint::size<T>(input, 2) == 1) && // square or 1d
                           at::symint::size<T>(input, 1) >= 32); // min 32 channels supported)
      if (kernel_cond) {
        return check_cudnn_depthwise_workload_with_filter<T>(input, stride[1], weight);
      }
      return false;
    } else {
      return false;
    }
  }

  bool use_miopen(const at::Tensor& input, const at::Tensor& weight, bool bias_defined) const  {
    if (needs_64bit_indexing_no_split(input, weight)) {
      return false;
    }
    return ((input.scalar_type() == at::kFloat) || (input.scalar_type() == at::kHalf) || (input.scalar_type() == at::kBFloat16))
           && cudnn_enabled
           && input.is_cuda()
           && detail::getCUDAHooks().compiledWithMIOpen()
           && input.dim() <= MIOPEN_DIM_MAX
           && !(groups > 1 && is_dilated()) // MIOpen currently does not support dilation with groups of size > 1
           ;
  }
  bool use_mkldnn(const at::Tensor& input, const at::Tensor& weight) const  {
#if AT_MKLDNN_ENABLED()
    if (!at::globalContext().userEnabledMkldnn()) {
      return false;
    }
    if (transposed && is_output_padding_big()) {
      return false;
    }
    if (input.device().is_cpu() &&
        ((input.scalar_type() == at::kBFloat16 && mkldnn_bf16_device_check()) ||
         (input.scalar_type() == at::kHalf && mkldnn_fp16_device_check()))) {
      return true;
    }
    return (input.is_mkldnn()) || // input is mkldnn Tensor
      (input.device().is_cpu() &&
       input.scalar_type() == kFloat && // only on CPU Float Tensors
       // For 1x1 filters, MKLDNN is faster than THNN when multi-threaded,
       // but THNN is faster when single-threaded.
       (is_strided() || is_dilated() || at::symint::size<T>(input, 0) >= 16 ||
        at::symint::size<T>(weight, -1) != 1 || at::symint::size<T>(weight, -2) != 1 || at::get_num_threads() > 1) &&
       (groups > 1
        || (at::symint::size<T>(weight, -1) > 3 && at::symint::size<T>(weight, -2) > 3)
        || at::symint::size<T>(input, 0) > 1
        || at::symint::size<T>(input, 0)*at::symint::size<T>(input, 1)*at::symint::size<T>(input, 2)*at::symint::size<T>(input, 3) > 20480) // for some case, native is faster
        );

#endif
    return false;
  }
  bool use_nnpack(const at::Tensor& input, const at::Tensor& weight) const  {
#if AT_NNPACK_ENABLED()
    return at::globalContext().userEnabledNNPACK() &&
           at::_nnpack_available() &&
           input.device().is_cpu() &&
           input.scalar_type() == kFloat && // only on CPU Float Tensors
           !is_dilated() && // or dilation
           !transposed &&   // or transposed tensors
           input.ndimension() == 4 && // must be in NCHW format
           weight.ndimension() == 4 &&
           (at::symint::size<T>(weight, 2) < 17) && (at::symint::size<T>(weight, 3) < 17) && // NNPACK only supports kernels up to 16x16
           (padding[0] < at::symint::size<T>(weight, 2)) && (padding[1] < at::symint::size<T>(weight, 3)) // NNPACK only supports padding < kernel_size. See https://github.com/pytorch/pytorch/issues/90142.
#if !defined(C10_MOBILE)
           && at::symint::size<T>(input, 0) >= 16 // ensure large enough batch size to ensure perf, tuneable
#endif
       ;
#endif
    return false;
  }
  bool use_xnnpack(const at::Tensor& input, const at::Tensor& weight,
                   const at::OptionalArrayRef<T> bias_sizes_opt) const {
#if defined(C10_MOBILE)
    if (!transposed) {
      // NB: for the call here, it MATTERS that we are templated. If you
      // untemplate this to always use SymInt, the function
      // xnnpack_use_convolution2d will always return false
      return (at::symint::size<T>(input, 1) == groups) &&
              xnnpack_use_convolution2d(
                  input,
                  weight,
                  bias_sizes_opt,
                  padding,
                  stride,
                  dilation,
                  groups,
                  transposed);
    }
#endif
    return false;
  }

  bool use_mps(const at::Tensor& input, const at::Tensor& weight) const {
    // These checks need to be expanded. Currently we have very limited set of
    // checks for MPS.
#ifdef USE_MPS
    if (needs_64bit_indexing_no_split(input, weight)) {
      return false;
    }
    if (!input.is_mps()) {
      return false;
    }
    return true;
#else
    return false;
#endif
  }

  // We currently only have depthwise support for the case where groups ==
  // nInputPlane and nInputPlane == nOutputPlane (the latter due to the lack of
  // a depthwise multiplier)
  bool is_depthwise(const at::Tensor& input, const at::Tensor& weight) const  {
    return input.is_cuda() &&
           !transposed &&
           (input.ndimension() == 4 || input.ndimension() == 5) &&
           at::symint::size<T>(input, 1) == groups &&
           groups > 1 && // no point if there is only a single group
           at::symint::size<T>(weight, 0) % at::symint::size<T>(input, 1) == 0; // output channels must be a multiple of input channels
  }
};

DEFINE_DISPATCH(conv_depthwise2d_backward_stub);
DEFINE_DISPATCH(conv_depthwise3d_backward_stub);
DEFINE_DISPATCH(cudnn_convolution_backward_stub);
DEFINE_DISPATCH(cudnn_convolution_transpose_backward_stub);
DEFINE_DISPATCH(slow_conv_transpose3d_backward_stub);
DEFINE_DISPATCH(convolution_depthwise3x3_winograd_stub);
DEFINE_DISPATCH(miopen_convolution_backward_stub);
DEFINE_DISPATCH(miopen_convolution_transpose_backward_stub);
DEFINE_DISPATCH(miopen_depthwise_convolution_backward_stub);
DEFINE_DISPATCH(mkldnn_convolution_backward_stub);
DEFINE_DISPATCH(mkldnn_convolution_transpose_stub);
DEFINE_DISPATCH(mkldnn_convolution_transpose_backward_stub);
DEFINE_DISPATCH(slow_conv_dilated2d_backward_stub);
DEFINE_DISPATCH(slow_conv_dilated3d_backward_stub);
DEFINE_DISPATCH(slow_conv_transpose2d_backward_stub);
REGISTER_NO_CPU_DISPATCH(conv_depthwise2d_backward_stub)
REGISTER_NO_CPU_DISPATCH(conv_depthwise3d_backward_stub)
REGISTER_NO_CPU_DISPATCH(cudnn_convolution_backward_stub)
REGISTER_NO_CPU_DISPATCH(cudnn_convolution_transpose_backward_stub)
REGISTER_NO_CPU_DISPATCH(miopen_convolution_backward_stub)
REGISTER_NO_CPU_DISPATCH(miopen_convolution_transpose_backward_stub)
REGISTER_NO_CPU_DISPATCH(miopen_depthwise_convolution_backward_stub)

template <typename T>
static std::ostream& operator<<(std::ostream & out, const ConvParams<T>& params) {
  out << "ConvParams {"
      << "  stride = " << IntArrayRef{params.stride}
      << "  padding = " << ArrayRef<T>{params.padding}
      << "  dilation = " << IntArrayRef{params.dilation}
      << "  transposed = " << params.transposed
      << "  output_padding = " << ArrayRef<T>{params.output_padding}
      << "  groups = " << params.groups
      << "  benchmark = " << params.benchmark
      << "  deterministic = " << params.deterministic
      << "  cudnn_enabled = " << params.cudnn_enabled
      << "  allow_tf32 = " << params.allow_tf32
      << "}";
  return out;
}

template <typename T>
static void check_shape_forward(const at::Tensor& input,
                                const c10::ArrayRef<T>& weight_sizes, const at::Tensor& bias,
                                const ConvParams<T>& params) {
  int64_t k = input.ndimension();
  int64_t weight_dim = weight_sizes.size();
  auto groups = params.groups;
  const auto& padding = params.padding;
  const auto& dilation = params.dilation;
  bool transposed = params.transposed;

  TORCH_CHECK(!params.is_padding_neg(), "negative padding is not supported");
  TORCH_CHECK(!params.is_output_padding_neg(), "negative output_padding is not supported");
  TORCH_CHECK(!params.is_stride_nonpos(), "non-positive stride is not supported");
  TORCH_CHECK(!params.is_dilation_neg(), "dilation should be greater than zero");
  TORCH_CHECK(groups > 0, "expected groups to be greater than 0, but got groups=", groups);

  TORCH_CHECK(weight_dim == k,
           "Expected ", weight_dim, "-dimensional input for ", weight_dim,
           "-dimensional weight ", weight_sizes, ", but got ", k, "-dimensional input of size ",
           at::symint::sizes<T>(input), " instead");
  TORCH_CHECK(weight_sizes[0] >= groups,
           "Given groups=", groups, ", expected weight to be at least ", groups,
           " at dimension 0, but got weight of size ", weight_sizes, " instead");
  TORCH_CHECK(weight_sizes[0] % groups == 0,
           "Given groups=", groups, ", expected weight to be divisible by ",
           groups, " at dimension 0, but got weight of size [", weight_sizes,
           "] instead");

  if (!transposed) {
    std::vector<T> input_shape;
    std::vector<T> kernel_shape;
    bool kernel_size_correct = true;

    TORCH_CHECK(at::symint::size<T>(input, 1) == (weight_sizes[1] * groups),
                "Given groups=", groups, ", weight of size ", weight_sizes,
                ", expected input", at::symint::sizes<T>(input), " to have ",
                (weight_sizes[1] * groups), " channels, but got ", at::symint::size<T>(input, 1),
                " channels instead");

    TORCH_CHECK(!bias.defined() || (bias.ndimension() == 1 && at::symint::size<T>(bias, 0) == weight_sizes[0]),
             "Given weight of size ", weight_sizes,
             ", expected bias to be 1-dimensional with ", weight_sizes[0], " elements",
             ", but got bias of size ", at::symint::sizes<T>(bias), " instead");

    for (const auto i : c10::irange(2, k)) {
      // T could be int64_t or SymInt, Specialized numeric_limts<SymInt> in c10/core/SymInt.h
      TORCH_CHECK(padding[i-2] <= (std::numeric_limits<T>::max() - padding[i-2]),
                  "Given padding=", padding[i-2], " at dimension ", i-2, " , expected padding to be at most ",
                  (std::numeric_limits<T>::max() / 2));
      input_shape.push_back(at::symint::size<T>(input, i) + 2 * padding[i-2]);
      // log new kernel size considering dilation
      kernel_shape.push_back(dilation[i-2] * (weight_sizes[i]-1) + 1);
      if (input_shape.back() < kernel_shape.back()) {
        kernel_size_correct = false;
      }
    }

    TORCH_CHECK(input_shape.size() == kernel_shape.size(), "Inconsistent shape between Input and Kernel");

    if (!kernel_size_correct) {
      // If kernel size is incorrect
      std::ostringstream input_ss;
      std::ostringstream kernel_ss;
      std::string separator;

      for (int i = 0, len = input_shape.size(); i < len; ++i) {
        input_ss << separator << input_shape[i];
        kernel_ss << separator << kernel_shape[i];
        separator = " x ";
      }

      TORCH_CHECK(false, "Calculated padded input size per channel: (", input_ss.str(), "). "
               "Kernel size: (", kernel_ss.str(), "). Kernel size can't be greater than actual input size");
    }
  } else { // transposed
    for (const auto i : c10::irange(2, k)) {
      TORCH_CHECK(padding[i-2] <= (std::numeric_limits<T>::max() - padding[i-2]),
                  "Given padding=", padding[i-2], " at dimension ", i-2, " , expected padding to be at most ",
                  (std::numeric_limits<T>::max() / 2));
    }
    TORCH_CHECK(at::symint::size<T>(input, 1) == weight_sizes[0],
             "Given transposed=", transposed, ", weight of size ", weight_sizes,
             ", expected input", at::symint::sizes<T>(input), " to have ", weight_sizes[0],
             " channels, but got ", at::symint::size<T>(input, 1), " channels instead");
    TORCH_CHECK(!bias.defined() || (bias.ndimension() == 1 && at::symint::size<T>(bias, 0) == weight_sizes[1] * groups),
             "Given transposed=", transposed, ", weight of size ", weight_sizes,
             ", expected bias to be 1-dimensional with ", weight_sizes[1] * groups, " elements",
             ", but got bias of size ", at::symint::sizes<T>(bias), " instead");
  }
}

template <typename T>
static void check_shape_backward(
    const at::Tensor& input,
    const c10::ArrayRef<T>& weight_sizes,
    const ConvParams<T>& params) {
  check_shape_forward<T>(input, weight_sizes, /*bias=*/ Tensor(), params);
}

// Given an input tensor and an expected number of spatial dimensions, checks that the
// input is a valid shape and returns the batched form of the input.
//
// Args:
//     input (Tensor): Input tensor
//     num_spatial_dims (int): Number of spatial dimensions expected for the input
//     func_name (string): Function name to produce a nice error message for invalid input
//
// Returns a std::tuple containing:
//     batched_input (Tensor): Input with a batch dimension
//     is_batched (bool): Indicates whether the original input was already batched
static std::tuple<Tensor, bool> batchify(
    const Tensor& input,
    const int64_t num_spatial_dims,
    const std::string& func_name) {
  // assume NTs are always batched
  if (input.is_nested()) {
    return std::make_tuple(input, true);
  }
  const auto dim_count_no_batch = num_spatial_dims + 1;
  const auto dim_count_batch = dim_count_no_batch + 1;
  const auto is_batched = (input.dim() == dim_count_batch);
  TORCH_CHECK(input.dim() == dim_count_no_batch || is_batched,
      "Expected ", dim_count_no_batch, "D (unbatched) or ", dim_count_batch,
      "D (batched) input to ", func_name, ", but got input of size: ", input.sizes());
  return std::make_tuple(is_batched ? input : input.unsqueeze(0), is_batched);
}

static void check_input_same_type_as_parameters(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias) {
  TORCH_CHECK(input.options().type_equal(weight.options()),
      "Input type (", input.toString(), ") and weight type (", weight.toString(),
      ") should be the same");
  TORCH_CHECK(!bias.defined() || (input.options().type_equal(bias.options())),
      "Input type (", input.toString(), ") and bias type (", bias.toString(),
      ") should be the same");
}

static void check_input_same_type_as_parameters(
    const Tensor& input,
    const Tensor& weight) {
  check_input_same_type_as_parameters(input, weight, /*bias=*/ Tensor());
}

#if AT_MKLDNN_ENABLED()
static void check_input_same_type_as_parameters(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const ConvBackend backend) {
  if (backend == ConvBackend::Mkldnn || backend == ConvBackend::MkldnnTranspose) {
    TORCH_CHECK(input.options().type_equal(weight.options())
        || (input.is_mkldnn() && weight.device().is_cpu() && weight.scalar_type() == kFloat),
        "Input type (", input.toString(), ") and weight type (", weight.toString(),
        ") should be the same or input should be a MKLDNN tensor and weight is a dense tensor");
    TORCH_CHECK(!bias.defined() || (input.options().type_equal(bias.options()))
        || (input.is_mkldnn() && bias.device().is_cpu() && bias.scalar_type() == kFloat),
        "Input type (", input.toString(), ") and bias type (", bias.toString(),
        ") should be the same or input should be a MKLDNN tensor and bias is a dense tensor");
  } else {
    check_input_same_type_as_parameters(input, weight, bias);
  }
}
#endif

static auto view4d(const at::Tensor& tensor) -> at::Tensor {
  TORCH_CHECK(tensor.ndimension() == 3,
           "expected 3D tensor, got tensor with ", tensor.ndimension(),
           " dimensions instead");
  return tensor.unsqueeze(2);
}

static auto view3d(const at::Tensor& tensor) -> at::Tensor {
  TORCH_CHECK(tensor.ndimension() == 4,
           "expected 4D tensor, got tensor with ", tensor.ndimension(),
           " dimensions instead");
  return tensor.squeeze(2);
}

static at::Tensor subtensor(at::Tensor& tensor, int64_t dim, int64_t groups, int64_t g) {
  if (!tensor.defined()) {
    return at::Tensor();
  }
  const auto memory_format = tensor.suggest_memory_format();
  int64_t n = tensor.sizes()[dim] / groups;
  return tensor.narrow(dim, n * g, n).contiguous(memory_format);
}

namespace {

std::pair<Tensor, Tensor> complex_to_real(const Tensor& inp) {
  auto inp_view_as_complex = at::view_as_real(inp);
  auto dim_i = inp_view_as_complex.dim() - 1;
  auto i_r = inp_view_as_complex.select(dim_i, 0);
  auto i_i = inp_view_as_complex.select(dim_i, 1);
  return std::make_pair(i_r, i_i);
}

at::Tensor complex_convolution(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    SymIntArrayRef stride,
    SymIntArrayRef padding,
    SymIntArrayRef dilation,
    bool transposed,
    SymIntArrayRef output_padding,
    const c10::SymInt& groups) {
  check_input_same_type_as_parameters(input, weight, bias);
  auto [i_r, i_i] = complex_to_real(input.resolve_conj());
  auto [w_r, w_i] = complex_to_real(weight.resolve_conj());

  // [NOTE] Complex Convolution
  // conv(W, x, b) = conv(Wr, xr, br) - conv(Wi, xi, 0) + i(conv(Wi, xr, bi) + conv(Wr, xi, 0))
  // where W, x and b are all complex inputs.
  // With Gauss Trick:
  // a = conv(Wr, xr, br),
  // b = conv(Wi, xi, 0),
  // c = conv(Wr + Wi, xr + xi, bi + br)
  // conv(W, x, b) = a - b + i(c - a - b)
  Tensor a, b, c;
  if (!bias.defined()) {
    a = at::convolution_symint(i_r, w_r, bias, stride, padding, dilation, transposed, output_padding, groups);
    b = at::convolution_symint(i_i, w_i, bias, stride, padding, dilation, transposed, output_padding, groups);
    c = at::convolution_symint(i_r + i_i, w_r + w_i, bias, stride, padding, dilation, transposed, output_padding, groups);
  } else {
    auto [b_r, b_i] = complex_to_real(bias.resolve_conj());
    a = at::convolution_symint(i_r, w_r, b_r, stride, padding, dilation, transposed, output_padding, groups);
    b = at::convolution_symint(i_i, w_i, Tensor(), stride, padding, dilation, transposed, output_padding, groups);
    c = at::convolution_symint(i_r + i_i, w_r + w_i, b_r + b_i, stride, padding, dilation, transposed, output_padding, groups);
  }

  auto i = c10::Scalar(c10::complex<double>(0, 1));
  return a - b + i * (c - a - b);
}

at::Tensor complex_convolution_mode(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_opt,
    c10::SymIntArrayRef stride,
    std::string_view padding,
    c10::SymIntArrayRef dilation,
    const c10::SymInt& groups) {
  auto bias = bias_opt.value_or(Tensor());
  check_input_same_type_as_parameters(input, weight, bias);
  auto [i_r, i_i] = complex_to_real(input.resolve_conj());
  auto [w_r, w_i] = complex_to_real(weight.resolve_conj());

  // See [NOTE] Complex Convolution
  Tensor a, b, c;
  if (!bias.defined()) {
    a = at::_convolution_mode_symint(i_r, w_r, bias, stride, padding, dilation, groups);
    b = at::_convolution_mode_symint(i_i, w_i, bias, stride, padding, dilation, groups);
    c = at::_convolution_mode_symint(i_r + i_i, w_r + w_i, bias, stride, padding, dilation, groups);
  } else {
    auto [b_r, b_i] = complex_to_real(bias.resolve_conj());
    a = at::_convolution_mode_symint(i_r, w_r, b_r, stride, padding, dilation, groups);
    b = at::_convolution_mode_symint(i_i, w_i, Tensor(), stride, padding, dilation, groups);
    c = at::_convolution_mode_symint(i_r + i_i, w_r + w_i, b_r + b_i, stride, padding, dilation, groups);
  }

  auto i = c10::Scalar(c10::complex<double>(0, 1));
  return a - b + i * (c - a - b);
}

} // namespace

at::Tensor conv1d_symint(
    const Tensor& input_, const Tensor& weight, const std::optional<Tensor>& bias_opt,
    SymIntArrayRef stride, SymIntArrayRef padding, SymIntArrayRef dilation, c10::SymInt groups) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  TORCH_CHECK(
    !bias.defined() || bias.dtype() == input_.dtype(),
    "Input type (",
    input_.dtype().name(),
    ") and bias type (",
    bias.dtype().name(),
    ") should be the same");

  auto [input, is_batched] = batchify(input_, /*num_spatial_dims=*/ 1, "conv1d");
  Tensor output;
  if (at::isComplexType(input_.scalar_type())) {
    output = complex_convolution(input, weight, bias, stride, padding, dilation, false, {0}, groups);
  } else {
    output = at::convolution_symint(input, weight, bias, stride, padding, dilation, false, {0}, groups);
  }
  return is_batched ? std::move(output) : output.squeeze(0);
}

at::Tensor conv2d_symint(
    const Tensor& input_, const Tensor& weight, const std::optional<Tensor>& bias_opt,
    SymIntArrayRef stride, SymIntArrayRef padding, SymIntArrayRef dilation, c10::SymInt groups) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  TORCH_CHECK(
    !bias.defined() || bias.dtype() == input_.dtype(),
    "Input type (",
    input_.dtype().name(),
    ") and bias type (",
    bias.dtype().name(),
    ") should be the same");

  auto [input, is_batched] = batchify(input_, /*num_spatial_dims=*/ 2, "conv2d");
  Tensor output;
  if (at::isComplexType(input_.scalar_type())) {
    output = complex_convolution(input, weight, bias, stride, padding, dilation, false, {{0, 0}}, groups);
  } else {
    output = at::convolution_symint(input, weight, bias, stride, padding, dilation, false, {{0, 0}}, groups);
  }
  return is_batched ? std::move(output) : output.squeeze(0);
}

at::Tensor conv3d_symint(
    const Tensor& input_, const Tensor& weight, const std::optional<Tensor>& bias_opt,
    SymIntArrayRef stride, SymIntArrayRef padding, SymIntArrayRef dilation, c10::SymInt groups) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  TORCH_CHECK(
    !bias.defined() || bias.dtype() == input_.dtype(),
    "Input type (",
    input_.dtype().name(),
    ") and bias type (",
    bias.dtype().name(),
    ") should be the same");

  auto [input, is_batched] = batchify(input_, /*num_spatial_dims=*/ 3, "conv3d");
  Tensor output;
  if (at::isComplexType(input_.scalar_type())) {
    output = complex_convolution(input, weight, bias, stride, padding, dilation, false, {{0, 0, 0}}, groups);
  } else {
    output = at::convolution_symint(input, weight, bias, stride, padding, dilation, false, {{0, 0, 0}}, groups);
  }
  return is_batched ? std::move(output) : output.squeeze(0);
}


static Tensor convolution_same(
    const Tensor &input, const Tensor &weight, const Tensor &bias,
    SymIntArrayRef stride, SymIntArrayRef dilation, const c10::SymInt& groups) {

  auto k = weight.dim();
  TORCH_CHECK(k > 2, "weight should have at least three dimensions");
  TORCH_CHECK(groups > 0, "non-positive groups is not supported");
  auto dim = static_cast<size_t>(k - 2);
  auto weight_sizes = weight.sym_sizes();
  auto input_sizes = input.sym_sizes();
  TORCH_CHECK(k == input.dim(),
              "Expected ", k, "-dimensional input for ",
              k, "-dimensional weight", weight_sizes, ", but got ",
              input.dim(), "-dimensional input of size ",
              input.sizes(), " instead");
  TORCH_CHECK(stride.size() == dim || stride.size() == 1U,
              "stride cannot broadcast to ", dim, " dimensions");
  TORCH_CHECK(dilation.size() == dim || dilation.size() == 1U,
              "dilation cannot broadcast to ", dim, " dimensions");
  for (auto i: c10::irange(stride.size())) {
    TORCH_CHECK(stride[i] == 1, "padding='same' is not supported for strided convolutions");
  }

  // Calculate the correct padding
  SymDimVector padding_l, padding_r;
  bool symmetric_padding = true;
  for (auto i: c10::irange(dim)) {
    auto s = stride.size() == 1 ? stride[0] : stride[i];
    auto d = dilation.size() == 1 ? dilation[0] : dilation[i];
    auto pad = pooling_same_mode_padding_lr(
        input_sizes[i + 2], weight_sizes[i + 2], s, d);
    padding_l.push_back(pad.first);
    padding_r.push_back(pad.second);
    if (pad.first != pad.second) {
      symmetric_padding = false;
    }
  }

  if (symmetric_padding) {
    // All backends handle symmetric padding natively
    SymDimVector output_padding(dim);
    return at::convolution_symint(input, weight, bias, stride, padding_l, dilation,
                               false, output_padding, groups);
  }

  TORCH_WARN_ONCE("Using padding='same' with even kernel lengths and odd dilation may"
                  " require a zero-padded copy of the input be created");
  SmallVector<c10::SymInt, kDimVectorStaticSize * 2> pad_nd(static_cast<size_t>(2 * dim));
  for (auto i: c10::irange(dim)) {
    // Apply padding by the difference, leaving only a symmetric padding
    auto delta_pad = padding_r[i] - padding_l[i];
    auto pad_idx = 2 * (dim - 1 - i);  // F.pad goes from last dim to first
    if (delta_pad > 0) {
      pad_nd[pad_idx + 1] = delta_pad;
    } else {
      pad_nd[pad_idx] = delta_pad;
      padding_l[i] = padding_r[i];
    }
  }
  auto padded_input = at::constant_pad_nd_symint(input, pad_nd, 0);
  SymDimVector output_padding(dim);
  return at::convolution_symint(padded_input, weight, bias, stride, padding_l,
                                dilation, false, output_padding, groups);
}

Tensor _convolution_mode_symint(
    const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt,
    SymIntArrayRef stride, std::string_view padding, SymIntArrayRef dilation,
    c10::SymInt groups) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  if (padding == "same") {
    return at::native::convolution_same(
        input, weight, bias, stride, dilation, groups);
  } else if (padding == "valid") {
    return at::convolution_symint(
        input, weight, bias, stride, {{0}}, dilation, false, {{0}}, groups);
  }
  TORCH_CHECK(false, "Invalid padding string: '", padding, "'");
}

at::Tensor conv1d_padding_symint(
    const Tensor& input_, const Tensor& weight, const std::optional<Tensor>& bias,
    c10::SymIntArrayRef stride, std::string_view padding, c10::SymIntArrayRef dilation,
    c10::SymInt groups) {
  auto [input, is_batched] = batchify(input_, /*num_spatial_dims=*/ 1, "conv1d");
  Tensor output;
  if (at::isComplexType(input_.scalar_type())) {
    output = complex_convolution_mode(input, weight, bias, stride, padding, dilation, groups);
  } else {
    output = at::_convolution_mode_symint(input, weight, bias, stride, padding, dilation, groups);
  }
  return is_batched ? std::move(output) : output.squeeze(0);
}

at::Tensor conv2d_padding_symint(
    const Tensor& input_, const Tensor& weight, const std::optional<Tensor>& bias,
    c10::SymIntArrayRef stride, std::string_view padding, c10::SymIntArrayRef dilation,
    c10::SymInt groups) {
  auto [input, is_batched] = batchify(input_, /*num_spatial_dims=*/ 2, "conv2d");
  Tensor output;
  if (at::isComplexType(input_.scalar_type())) {
    output = complex_convolution_mode(input, weight, bias, stride, padding, dilation, groups);
  } else {
    output = at::_convolution_mode_symint(input, weight, bias, stride, padding, dilation, groups);
  }
  return is_batched ? std::move(output) : output.squeeze(0);
}

at::Tensor conv3d_padding_symint(
    const Tensor& input_, const Tensor& weight, const std::optional<Tensor>& bias,
    c10::SymIntArrayRef stride, std::string_view padding, c10::SymIntArrayRef dilation,
    c10::SymInt groups) {
  auto [input, is_batched] = batchify(input_, /*num_spatial_dims=*/ 3, "conv3d");
  Tensor output;
  if (at::isComplexType(input_.scalar_type())) {
    output = complex_convolution_mode(input, weight, bias, stride, padding, dilation, groups);
  } else {
    output = at::_convolution_mode_symint(input, weight, bias, stride, padding, dilation, groups);
  }
  return is_batched ? std::move(output) : output.squeeze(0);
}

at::Tensor conv_transpose1d_symint(
    const Tensor& input_, const Tensor& weight, const std::optional<Tensor>& bias_opt,
    SymIntArrayRef stride, SymIntArrayRef padding, SymIntArrayRef output_padding, c10::SymInt groups, SymIntArrayRef dilation) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto [input, is_batched] = batchify(input_, /*num_spatial_dims=*/ 1, "conv_transpose1d");
  Tensor output;
  if (at::isComplexType(input_.scalar_type())) {
    output = complex_convolution(
      input, weight, bias, stride, padding, dilation, true, output_padding, groups);
  } else {
    output = at::convolution_symint(
      input, weight, bias, stride, padding, dilation, true, output_padding, groups);
  }
  return is_batched ? std::move(output) : output.squeeze(0);
}

at::Tensor conv_transpose2d_symint(
    const Tensor& input_, const Tensor& weight, const std::optional<Tensor>& bias_opt,
    SymIntArrayRef stride, SymIntArrayRef padding, SymIntArrayRef output_padding, c10::SymInt groups, SymIntArrayRef dilation) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto [input, is_batched] = batchify(input_, /*num_spatial_dims=*/ 2, "conv_transpose2d");
  Tensor output;
  if (at::isComplexType(input_.scalar_type())) {
    output = complex_convolution(
      input, weight, bias, stride, padding, dilation, true, output_padding, groups);
  } else {
    output = at::convolution_symint(
      input, weight, bias, stride, padding, dilation, true, output_padding, groups);
  }
  return is_batched ? std::move(output) : output.squeeze(0);
}

at::Tensor conv_transpose3d_symint(
    const Tensor& input_, const Tensor& weight, const std::optional<Tensor>& bias_opt,
    SymIntArrayRef stride, SymIntArrayRef padding, SymIntArrayRef output_padding, c10::SymInt groups, SymIntArrayRef dilation) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto [input, is_batched] = batchify(input_, /*num_spatial_dims=*/ 3, "conv_transpose3d");
  Tensor output;
  if (at::isComplexType(input_.scalar_type())) {
    output = complex_convolution(
      input, weight, bias, stride, padding, dilation, true, output_padding, groups);
  } else {
    output = at::convolution_symint(
      input, weight, bias, stride, padding, dilation, true, output_padding, groups);
  }
  return is_batched ? std::move(output) : output.squeeze(0);
}

at::Tensor convolution(
    const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
    bool transposed, IntArrayRef output_padding, int64_t groups) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto& ctx = at::globalContext();
  // See Note [Enabling Deterministic Operations]
  bool deterministic = ctx.deterministicCuDNN() || ctx.deterministicAlgorithms();
  return at::_convolution(input, weight, bias, stride, padding, dilation,
                          transposed, output_padding, groups,
                          ctx.benchmarkCuDNN(), deterministic, ctx.userEnabledCuDNN(), ctx.allowTF32CuDNN(at::Float32Op::CONV));
}

at::Tensor convolution_overrideable(
    const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
    bool transposed, IntArrayRef output_padding, int64_t groups) {
  TORCH_CHECK_NOT_IMPLEMENTED(false, "convolution_overrideable not implemented. You are likely triggering this with tensor backend other than CPU/CUDA/MKLDNN, if this is intended, please use TORCH_LIBRARY_IMPL to override this function ");
}

// Function to select the convolution backend based on the inputs and params.
// This overload is used within the convolution internals but not exposed to python.
// NB: The forward pass provides a bias tensor while the backward pass provides
// a bool indicating whether the bias is defined. This is done to save memory by
// avoiding saving the full bias tensor for backward.
template <typename T>
static ConvBackend _select_conv_backend(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const at::OptionalArrayRef<T> bias_sizes_opt,
    const bool need_backward,
    const ConvParams<T>& params) {

  // don't send empty inputs through backends
  if 
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
- May involve **JIT compilation** or compilation optimizations.
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

- **File Documentation**: `Convolution.cpp_docs.md_docs.md`
- **Keyword Index**: `Convolution.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
