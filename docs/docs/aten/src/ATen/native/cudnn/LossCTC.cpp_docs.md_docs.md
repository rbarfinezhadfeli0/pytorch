# Documentation: `docs/aten/src/ATen/native/cudnn/LossCTC.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cudnn/LossCTC.cpp_docs.md`
- **Size**: 14,199 bytes (13.87 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cudnn/LossCTC.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/cudnn/LossCTC.cpp`
- **Size**: 11,291 bytes (11.03 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#if AT_CUDNN_ENABLED()
#include <ATen/cudnn/Descriptors.h>
#endif

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_assert_async.h>
#include <ATen/ops/_cudnn_ctc_loss.h>
#include <ATen/ops/_cudnn_ctc_loss_native.h>
#include <ATen/ops/_use_cudnn_ctc_loss.h>
#include <ATen/ops/_use_cudnn_ctc_loss_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/le.h>
#include <ATen/ops/lt.h>
#endif

#if (!AT_CUDNN_ENABLED())

namespace at {
namespace native {

// See Note [ATen preprocessor philosophy]

bool _use_cudnn_ctc_loss(
    const Tensor& log_probs,
    const Tensor& targets,
    IntArrayRef input_lengths,
    IntArrayRef target_lengths,
    int64_t BLANK) {
  return false;
}

bool _use_cudnn_ctc_loss_tensor(
    const Tensor& log_probs,
    const Tensor& targets,
    const Tensor& input_lengths,
    const Tensor& target_lengths,
    int64_t BLANK) {
  return false;
}

std::tuple<Tensor, Tensor> _cudnn_ctc_loss(
    const Tensor& log_probs,
    const Tensor& targets,
    IntArrayRef input_lengths,
    IntArrayRef target_lengths,
    int64_t BLANK,
    bool deterministic,
    bool zero_infinity) {
  TORCH_CHECK(
      false, "cudnn_ctc_loss: ATen not compiled with cuDNN >= 7 support");
}

std::tuple<Tensor, Tensor> _cudnn_ctc_loss_tensor(
    const Tensor& log_probs,
    const Tensor& targets,
    const Tensor& input_lengths,
    const Tensor& target_lengths,
    int64_t BLANK,
    bool deterministic,
    bool zero_infinity) {
  TORCH_CHECK(
      false, "cudnn_ctc_loss: ATen not compiled with cuDNN >= 8 support");
}

} // namespace native
} // namespace at

#else // AT_CUDNN_ENABLED

#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>

#include <ATen/TensorUtils.h>
#include <c10/util/irange.h>

namespace at {
namespace native {

bool _use_cudnn_ctc_loss(
    const Tensor& log_probs,
    const Tensor& targets,
    IntArrayRef input_lengths,
    IntArrayRef target_lengths,
    int64_t BLANK) {
  auto& ctx = at::globalContext();

  bool use_cudnn = ctx.userEnabledCuDNN() && (BLANK == 0) &&
      (targets.dim() == 1) && (log_probs.scalar_type() == at::kFloat) &&
      (targets.scalar_type() == at::kInt) &&
      (targets.device().type() == at::kCPU) && (targets.is_contiguous()) &&
      (log_probs.device().type() == at::kCUDA) && (log_probs.dim() == 3);

  if (use_cudnn) {
    // we don't know that input_lengths and target_lengths have the same size
    // (they should, but we didn't check yet)
    int64_t max_input_length = log_probs.size(0);
    for (const auto input_length : input_lengths) {
      use_cudnn = use_cudnn && ((input_length == max_input_length) ? 1 : 0);
    }
    for (const auto b : c10::irange(target_lengths.size())) {
      // target length < 256 is documented, but we see illegal memory accesses
      // when target lengths > input lengths for CuDNN
      use_cudnn = use_cudnn && (target_lengths[b] < 256) &&
          (target_lengths[b] <= input_lengths[b]);
    }
  }
  return use_cudnn;
}

bool _use_cudnn_ctc_loss_tensor(
    const Tensor& log_probs,
    const Tensor& targets,
    const Tensor& input_lengths,
    const Tensor& target_lengths,
    int64_t BLANK) {
  auto& ctx = at::globalContext();

  bool use_cudnn = ctx.userEnabledCuDNN() && (BLANK == 0) &&
      (targets.dim() == 1) && (log_probs.scalar_type() == at::kFloat) &&
      (targets.scalar_type() == at::kInt) &&
      (log_probs.device().type() == at::kCUDA) && (targets.is_contiguous()) &&
      (log_probs.dim() == 3) && (input_lengths.scalar_type() == at::kInt) &&
      (target_lengths.scalar_type() == at::kInt);

  if (use_cudnn) {
    if (at::cuda::currentStreamCaptureStatus() ==
        at::cuda::CaptureStatus::None) {
      Tensor tlc = target_lengths.to(Device(at::kCPU), at::kLong).contiguous();
      IntArrayRef tl(tlc.data_ptr<int64_t>(), tlc.numel());
      for (const auto b : c10::irange(tl.size())) {
        // target length < 256 is documented, but we see illegal memory accesses
        // when target lengths > input lengths for CuDNN
        Tensor ilc = input_lengths.to(Device(at::kCPU), at::kLong).contiguous();
        Tensor tlc =
            target_lengths.to(Device(at::kCPU), at::kLong).contiguous();
        IntArrayRef il(ilc.const_data_ptr<int64_t>(), ilc.numel());
        IntArrayRef tl(tlc.data_ptr<int64_t>(), tlc.numel());
        use_cudnn = use_cudnn && (tl[b] < 256) && (tl[b] <= il[b]);
        if (!use_cudnn) {
          break;
        }
      }
    } else {
      if (target_lengths.device().type() != at::kCUDA ||
          input_lengths.device().type() != at::kCUDA) {
        TORCH_CHECK(
            false,
            "CTCLoss cannot be graph captured with CPU length tensors. "
            "Move CPU length tensors to GPU memory to enable graph capture.")
      }
      at::_assert_async(at::lt(input_lengths.max(), 256));
      at::_assert_async(at::le(target_lengths, input_lengths).all());
    }
  }

  return use_cudnn;
}

std::tuple<Tensor, Tensor> _cudnn_ctc_loss(
    const Tensor& log_probs_t,
    const Tensor& targets_t,
    IntArrayRef input_lengths_,
    IntArrayRef target_lengths_,
    int64_t BLANK,
    bool deterministic,
    bool zero_infinity) {
  (void)zero_infinity; // only used for backward
  const CheckedFrom c = "cudnn_ctc_loss";
  const TensorArg log_probs{log_probs_t, "log_probs", 1};
  const TensorArg targets{targets_t, "targets", 2};
  checkDim(c, log_probs, 3);
  checkScalarType(c, log_probs, kFloat);
  checkDim(c, targets, 1);
  checkScalarType(c, targets, kInt);
  checkContiguous(c, targets); // ?
  checkBackend(c, {*log_probs}, Backend::CUDA);
  checkBackend(c, {*targets}, Backend::CPU);
  const auto batch_size = log_probs->size(1);
  TORCH_CHECK(
      static_cast<int64_t>(input_lengths_.size()) == batch_size,
      "input_lengths needs to have size to match batch_size");
  TORCH_CHECK(
      static_cast<int64_t>(target_lengths_.size()) == batch_size,
      "target_lengths needs to have size to match batch_size");

  std::vector<int> input_lengths(input_lengths_.begin(), input_lengths_.end());
  std::vector<int> target_lengths(
      target_lengths_.begin(), target_lengths_.end());

  TORCH_CHECK(BLANK == 0, "blank must be label 0 for cudnn_ctc_loss");
  // checked in dispatch:
  // assert other conditions for cudnnCTCLoss: all label lengths <= 256
  // all input lengths = logprob.size(0)

  const auto handle = getCudnnHandle();

  const cudnnCTCLossAlgo_t algo =
      (deterministic ? CUDNN_CTC_LOSS_ALGO_DETERMINISTIC
                     : CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC);

  CTCLossDescriptor ctc_loss_desc;

  // so the CuDNN gradient semantics have changed between 7.1 and 7.6,
  // this is CuDNN 7.6 only, see PyTorch 1.2 for older CuDNN.
  ctc_loss_desc.setEx(
      CUDNN_DATA_FLOAT, CUDNN_LOSS_NORMALIZATION_SOFTMAX, CUDNN_PROPAGATE_NAN);
  TensorDescriptor log_probs_desc{log_probs_t};
  Tensor grad = at::empty_like(log_probs_t, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  TensorDescriptor grad_desc{grad};

  size_t workspace_size;
  AT_CUDNN_CHECK(cudnnGetCTCLossWorkspaceSize(
      handle,
      log_probs_desc.desc(),
      grad_desc.desc(),
      targets->data_ptr<int>(),
      target_lengths.data(),
      input_lengths.data(),
      algo,
      ctc_loss_desc.desc(),
      &workspace_size));

  Tensor workspace =
      at::empty(workspace_size, log_probs->options().dtype(kByte));
  Tensor costs = at::empty({log_probs->size(1)}, log_probs->options());

  AT_CUDNN_CHECK(cudnnCTCLoss(
      handle,
      log_probs_desc.desc(),
      log_probs_t.data_ptr(),
      targets->data_ptr<int>(),
      target_lengths.data(),
      input_lengths.data(),
      costs.data_ptr(),
      grad_desc.desc(),
      grad.data_ptr(),
      algo,
      ctc_loss_desc.desc(),
      workspace.data_ptr(),
      workspace_size));
  return std::make_tuple(costs, grad);
}

std::tuple<Tensor, Tensor> _cudnn_ctc_loss_tensor(
    const Tensor& log_probs_t,
    const Tensor& targets_t,
    const Tensor& input_lengths,
    const Tensor& target_lengths,
    int64_t BLANK,
    bool deterministic,
    bool zero_infinity) {
  Tensor targets_t_ = targets_t;
  Tensor input_lengths_ = input_lengths;
  Tensor target_lengths_ = target_lengths;
  if (targets_t.device().type() == at::kCPU) {
    targets_t_ = targets_t.to(Device(at::kCUDA));
  }
  if (input_lengths.device().type() == at::kCPU) {
    input_lengths_ = input_lengths.to(Device(at::kCUDA));
  }
  if (input_lengths.device().type() == at::kCPU) {
    target_lengths_ = target_lengths.to(Device(at::kCUDA));
  }

  const CheckedFrom c = "cudnn_ctc_loss";
  const TensorArg log_probs{log_probs_t, "log_probs", 1};
  const TensorArg targets{targets_t_, "targets", 2};
  checkDim(c, log_probs, 3);
  checkScalarType(c, log_probs, kFloat);
  checkDim(c, targets, 1);
  checkScalarType(c, targets, kInt);
  checkContiguous(c, targets); // ?
  checkBackend(c, {*log_probs}, Backend::CUDA);
  checkBackend(c, {*targets}, Backend::CUDA);
  const auto batch_size = log_probs->size(1);
  int64_t input_lengths_size =
      !input_lengths_.sizes().empty() ? input_lengths_.size(0) : 1;
  int64_t target_lengths_size =
      !target_lengths_.sizes().empty() ? target_lengths_.size(0) : 1;
  TORCH_CHECK(
      input_lengths_size == batch_size,
      "input_lengths needs to have size to match batch_size");
  TORCH_CHECK(
      target_lengths_size == batch_size,
      "target_lengths needs to have size to match batch_size");

  TORCH_CHECK(BLANK == 0, "blank must be label 0 for cudnn_ctc_loss");
  // checked in dispatch:
  // assert other conditions for cudnnCTCLoss: all label lengths <= 256
  // all input lengths = logprob.size(0)

  const auto handle = getCudnnHandle();

  const cudnnCTCLossAlgo_t algo =
      (deterministic ? CUDNN_CTC_LOSS_ALGO_DETERMINISTIC
                     : CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC);

  CTCLossDescriptor ctc_loss_desc;

  ctc_loss_desc.set_v8_v9(
      CUDNN_DATA_FLOAT,
      CUDNN_LOSS_NORMALIZATION_SOFTMAX,
      CUDNN_PROPAGATE_NAN,
      255);
  TensorDescriptor log_probs_desc{log_probs_t};
  Tensor grad = at::empty_like(log_probs_t, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  TensorDescriptor grad_desc{grad};

  size_t workspace_size;
  AT_CUDNN_CHECK(cudnnGetCTCLossWorkspaceSize_v8(
      handle,
      algo,
      ctc_loss_desc.desc(),
      log_probs_desc.desc(),
      grad_desc.desc(),
      &workspace_size));
  Tensor workspace =
      at::empty(workspace_size, log_probs->options().dtype(kByte));
  Tensor costs = at::empty({log_probs->size(1)}, log_probs->options());

  AT_CUDNN_CHECK(cudnnCTCLoss_v8(
      handle,
      algo,
      ctc_loss_desc.desc(),
      log_probs_desc.desc(),
      log_probs_t.data_ptr(),
      targets_t_.data_ptr<int>(),
      target_lengths_.data_ptr<int>(),
      input_lengths_.data_ptr<int>(),
      costs.data_ptr(),
      grad_desc.desc(),
      grad.data_ptr(),
      workspace_size,
      workspace.data_ptr()

          ));
  return std::make_tuple(costs, grad);
}

} // namespace native
} // namespace at

#endif

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `native`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cudnn`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Config.h`
- `ATen/core/Tensor.h`
- `ATen/cuda/CUDAConfig.h`
- `ATen/cuda/CUDAGraphsUtils.cuh`
- `ATen/cudnn/Descriptors.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/_assert_async.h`
- `ATen/ops/_cudnn_ctc_loss.h`
- `ATen/ops/_cudnn_ctc_loss_native.h`
- `ATen/ops/_use_cudnn_ctc_loss.h`
- `ATen/ops/_use_cudnn_ctc_loss_native.h`
- `ATen/ops/empty.h`
- `ATen/ops/empty_like.h`
- `ATen/ops/le.h`
- `ATen/ops/lt.h`
- `ATen/cudnn/Types.h`
- `ATen/cudnn/Utils.h`
- `ATen/TensorUtils.h`
- `c10/util/irange.h`


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

Files in the same folder (`aten/src/ATen/native/cudnn`):

- [`MHA.cpp_docs.md`](./MHA.cpp_docs.md)
- [`Conv_v7.cpp_docs.md`](./Conv_v7.cpp_docs.md)
- [`RNN.cpp_docs.md`](./RNN.cpp_docs.md)
- [`ConvShared.cpp_docs.md`](./ConvShared.cpp_docs.md)
- [`MHA.h_docs.md`](./MHA.h_docs.md)
- [`ConvShared.h_docs.md`](./ConvShared.h_docs.md)
- [`RNNUtils.h_docs.md`](./RNNUtils.h_docs.md)
- [`BatchNorm.h_docs.md`](./BatchNorm.h_docs.md)
- [`Conv_v8.cpp_docs.md`](./Conv_v8.cpp_docs.md)
- [`GridSampler.cpp_docs.md`](./GridSampler.cpp_docs.md)


## Cross-References

- **File Documentation**: `LossCTC.cpp_docs.md`
- **Keyword Index**: `LossCTC.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/cudnn`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/cudnn`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/aten/src/ATen/native/cudnn`):

- [`BatchNorm.cpp_kw.md_docs.md`](./BatchNorm.cpp_kw.md_docs.md)
- [`GridSampler.cpp_kw.md_docs.md`](./GridSampler.cpp_kw.md_docs.md)
- [`ConvShared.cpp_docs.md_docs.md`](./ConvShared.cpp_docs.md_docs.md)
- [`RNN.cpp_docs.md_docs.md`](./RNN.cpp_docs.md_docs.md)
- [`MHA.cpp_kw.md_docs.md`](./MHA.cpp_kw.md_docs.md)
- [`AffineGridGenerator.cpp_docs.md_docs.md`](./AffineGridGenerator.cpp_docs.md_docs.md)
- [`Conv_v8.cpp_kw.md_docs.md`](./Conv_v8.cpp_kw.md_docs.md)
- [`Conv_v7.cpp_kw.md_docs.md`](./Conv_v7.cpp_kw.md_docs.md)
- [`AffineGridGenerator.cpp_kw.md_docs.md`](./AffineGridGenerator.cpp_kw.md_docs.md)
- [`BatchNorm.h_kw.md_docs.md`](./BatchNorm.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `LossCTC.cpp_docs.md_docs.md`
- **Keyword Index**: `LossCTC.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
