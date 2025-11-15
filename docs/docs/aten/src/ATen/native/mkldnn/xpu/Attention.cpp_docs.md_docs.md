# Documentation: `docs/aten/src/ATen/native/mkldnn/xpu/Attention.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/mkldnn/xpu/Attention.cpp_docs.md`
- **Size**: 12,982 bytes (12.68 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/mkldnn/xpu/Attention.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/mkldnn/xpu/Attention.cpp`
- **Size**: 10,527 bytes (10.28 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/Context.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <c10/util/Array.h>
#include <torch/library.h>

namespace {
bool check_head_dim_size_xpu(sdp::sdp_params const& params, bool debug) {
  const auto query_size_last = params.query.sym_size(-1);
  const auto key_size_last = params.key.sym_size(-1);
  const auto value_size_last = params.value.sym_size(-1);
  if (query_size_last != key_size_last) {
    if (debug) {
      TORCH_WARN(
          "OneDNN attention requires q,k to have the same last dimension.",
          " Got Query.size(-1): ",
          query_size_last,
          ", Key.size(-1): ",
          key_size_last,
          " instead.");
    }
    return false;
  }

  constexpr int MAX_HEAD_DIM = 576;
  const auto max_size_last = query_size_last.max(value_size_last);
  if (max_size_last > MAX_HEAD_DIM) {
    if (debug) {
      TORCH_WARN(
          "OneDNN attention requires q,k,v to have head dimension less than ",
          MAX_HEAD_DIM,
          ". Got ",
          max_size_last,
          " instead.");
    }
    return false;
  }
  return true;
}

bool check_no_grad(sdp::sdp_params const& params, bool debug) {
  const bool any_inputs_require_grad = params.query.requires_grad() ||
      params.key.requires_grad() || params.value.requires_grad();
  const bool gradmode_enabled = at::GradMode::is_enabled();
  if (debug && any_inputs_require_grad && gradmode_enabled) {
    TORCH_WARN("Backward or grad to be supported.");
  }
  return !any_inputs_require_grad || !gradmode_enabled;
}

bool can_use_overrideable_attention(sdp::sdp_params const& params, bool debug) {
  constexpr auto supported_dtypes = c10::array_of<at::ScalarType>(
      at::kFloat, at::kBFloat16, at::kHalf); // double is not supported

  // Define gate functions that determine if a flash kernel can be run
  constexpr auto constraints = c10::array_of<bool (*)(
      sdp::sdp_params const&, bool)>(
      sdp::check_nested_tensor,
      sdp::check_for_dropout,
      sdp::check_tensor_shapes,
      sdp::check_batch_size_and_num_heads_dense<true /*supports GQA*/>,
      sdp::check_attn_mask_shape,
      sdp::check_nonzero_sequence_lengths_dense,
      sdp::check_last_dim_stride_equals_1_dense<false /*ignore_singleton_dim*/>,
      check_head_dim_size_xpu,
      check_no_grad);
  for (auto& constraint : constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }
  return sdp::check_tensor_dtype(params, supported_dtypes, debug);
}

bool can_use_flash_attention(sdp::sdp_params const& params, bool debug) {
  // Currently, XPU fallbacks flash attention to overridable
  return can_use_overrideable_attention(params, debug);
}

bool can_use_cudnn_attention(sdp::sdp_params const& params, bool debug) {
  if (debug) {
    TORCH_WARN("XPU don't support SDPA cudnn attention backend.");
  }
  return false;
}

bool can_use_mem_efficien_attention(sdp::sdp_params const& params, bool debug) {
  if (debug) {
    TORCH_WARN("XPU don't support SDPA mem efficient attention backend.");
  }
  return false;
}

bool priority_order_init = false;

std::array<sdp::SDPBackend, sdp::num_backends> priority_order(
    sdp::sdp_params const& params) {
  if (!priority_order_init) {
    priority_order_init = true;
    const std::vector<int64_t> priority_order = {
        static_cast<int64_t>(at::SDPBackend::overrideable),
        static_cast<int64_t>(at::SDPBackend::math),
        static_cast<int64_t>(at::SDPBackend::flash_attention),
        static_cast<int64_t>(at::SDPBackend::efficient_attention),
        static_cast<int64_t>(at::SDPBackend::cudnn_attention)};
    at::globalContext().setSDPPriorityOrder(priority_order);
  }
  return at::globalContext().sDPPriorityOrder();
}

sdp::SDPBackend select_sdp_backend_xpu(sdp::sdp_params const& kernel_params) {
  // This function defines the priority order of the different sdp backends
  // 1. Flash Attention
  // 2. Math fallback
  auto& ctx = at::globalContext();
  // use overridable linked to onednn as overridable implementation
  if (!ctx.userEnabledMathSDP() && !ctx.userEnabledOverrideableSDP() &&
      !ctx.userEnabledFlashSDP()) {
    return sdp::SDPBackend::error;
  }

  // Get ideal kernel ordering
  const auto ordering = priority_order(kernel_params);

  // Because TORCHCHECK checks if condition is true we negate debug so that
  // The statements will be printed when debug is true
  bool print_debug = false;
  for (auto& backend : ordering) {
    switch (backend) {
      case sdp::SDPBackend::overrideable:
        if (ctx.userEnabledOverrideableSDP() &&
            can_use_overrideable_attention(kernel_params, print_debug)) {
          return sdp::SDPBackend::overrideable;
        }
        break;
      case sdp::SDPBackend::math:
        if (ctx.userEnabledMathSDP()) {
          return sdp::SDPBackend::math;
        }
        break;
      case sdp::SDPBackend::flash_attention:
        if (ctx.userEnabledFlashSDP() &&
            can_use_flash_attention(kernel_params, print_debug)) {
          TORCH_WARN_ONCE(
              "SDPA Flash Attention backend is not supported on XPU, falling back to OVERRIDEABLE backend.");
          return sdp::SDPBackend::overrideable;
        }
        break;
      case sdp::SDPBackend::cudnn_attention:
        if (ctx.userEnabledCuDNNSDP() &&
            can_use_cudnn_attention(kernel_params, print_debug)) {
          TORCH_CHECK(false, "Invalid backend");
        }
        break;
      case sdp::SDPBackend::efficient_attention:
        if (ctx.userEnabledMemEfficientSDP() &&
            can_use_mem_efficien_attention(kernel_params, print_debug)) {
          TORCH_CHECK(false, "Invalid backend");
        }
        break;
      default:
        TORCH_CHECK(false, "Invalid backend");
    }
  }
  // If we have gotten to this point then two things have happened:
  // 1. can_use_overridable_attention did not satisfy the constraints to be ran
  // 2. The user has explicitly disabled the math kernel
  // We then re-run the kernel checks with debug enabled to print out the
  // reason why the kernel was not selected

  print_debug = true;
  TORCH_WARN("Flash attention kernel not used because:");
  can_use_flash_attention(kernel_params, print_debug);
  TORCH_WARN("Overrideable attention kernel not used because:");
  can_use_overrideable_attention(kernel_params, print_debug);
  TORCH_WARN("CuDNN attention kernel not used because:");
  can_use_cudnn_attention(kernel_params, print_debug);
  TORCH_WARN("Memory Efficient attention kernel not used because:");
  can_use_mem_efficien_attention(kernel_params, print_debug);
  TORCH_CHECK(!print_debug, "No available kernel. Aborting execution.")
  return sdp::SDPBackend::error;
}
} // namespace

namespace at::native {
int64_t _fused_sdp_choice_xpu(
    const at::Tensor& query_,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  sdp::sdp_params kernel_params{
      query_, key, value, attn_mask_, dropout_p, is_causal, enable_gqa};
  auto backend = select_sdp_backend_xpu(kernel_params);

  if (backend == sdp::SDPBackend::error) {
    TORCH_CHECK(
        false,
        "No viable backend for scaled_dot_product_attention was found. ",
        "This is likely due to turning off both the math kernel and the overrideable kernels.");
  }
  return static_cast<int64_t>(backend);
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    c10::SymInt,
    c10::SymInt,
    at::Tensor,
    at::Tensor,
    at::Tensor>
_scaled_dot_product_fused_attention_overrideable_xpu(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale) {
  TORCH_INTERNAL_ASSERT(
      query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
      "scaled_dot_product_fused_attention_overrideable_xpu: Accept only 4 dims inputs shape of {(B), H, T, K}");
  TORCH_INTERNAL_ASSERT(
      (key.size(0) == value.size(0)) && (key.size(1) == value.size(1)) &&
          (key.size(2) == value.size(2)),
      "scaled_dot_product_fused_attention_overrideable_xpu: K/V should have the same batch / seq / num_head");
  TORCH_INTERNAL_ASSERT(
      query.size(3) == key.size(3),
      "scaled_dot_product_fused_attention_overrideable_xpu: Q/K should have the same head_dim");
  TORCH_INTERNAL_ASSERT(
      query.size(1) % key.size(1) == 0,
      "scaled_dot_product_fused_attention_overrideable_xpu: number of heads in K/V must divide number of heads in Q");
  TORCH_INTERNAL_ASSERT(
      dropout_p == 0.0,
      "scaled_dot_product_fused_attention_overrideable_xpu: Currently do not support dropout > 0");
  TORCH_INTERNAL_ASSERT(
      !(attn_bias.has_value() && is_causal),
      "scaled_dot_product_fused_attention_overrideable_xpu: attn_bias cannot present with is_causal");

  const int64_t batch_size = query.size(0);
  const int64_t num_head_q = query.size(1);
  const int64_t num_head_kv = key.size(1);
  const int64_t head_dim_qk = query.size(3);
  const int64_t head_dim_v = value.size(3);
  const int64_t seq_len_q = query.size(2);
  const int64_t seq_len_kv = key.size(2);

  at::Tensor output;
  std::vector<int64_t> output_shape = {
      batch_size, num_head_q, seq_len_q, head_dim_v};
  alloc_with_matching_layout(query, output, output_shape);
  at::Tensor logsumexp, debug_attn_mask; // not supported

  at::native::onednn::sdpa(
      batch_size,
      seq_len_q,
      seq_len_kv,
      num_head_q,
      num_head_kv,
      head_dim_qk,
      head_dim_v,
      query,
      key,
      value,
      attn_bias,
      is_causal,
      scale.has_value() ? scale.value() : (1.0 / std::sqrt(head_dim_qk)),
      output,
      false,
      logsumexp);

  // rng not used
  auto philox_seed = at::empty({}, at::dtype(at::kLong));
  auto philox_offset = at::empty({}, at::dtype(at::kLong));
  return std::make_tuple(
      output,
      logsumexp,
      /* cum_seq_q */ at::Tensor(),
      /* cum_seq_k */ at::Tensor(),
      seq_len_q,
      seq_len_kv,
      philox_seed,
      philox_offset,
      debug_attn_mask);
}

REGISTER_XPU_DISPATCH(_fused_sdp_choice_stub, &_fused_sdp_choice_xpu);
} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `namespace`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/mkldnn/xpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Context.h`
- `ATen/native/mkldnn/xpu/detail/oneDNN.h`
- `ATen/native/transformers/attention.h`
- `ATen/native/transformers/sdp_utils.h`
- `ATen/native/transformers/sdp_utils_cpp.h`
- `c10/util/Array.h`
- `torch/library.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


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

Files in the same folder (`aten/src/ATen/native/mkldnn/xpu`):

- [`Conv.h_docs.md`](./Conv.h_docs.md)
- [`Linear.cpp_docs.md`](./Linear.cpp_docs.md)
- [`qlinear.h_docs.md`](./qlinear.h_docs.md)
- [`qconv.cpp_docs.md`](./qconv.cpp_docs.md)
- [`ScaledBlas.cpp_docs.md`](./ScaledBlas.cpp_docs.md)
- [`qconv.h_docs.md`](./qconv.h_docs.md)
- [`FusionUtils.cpp_docs.md`](./FusionUtils.cpp_docs.md)
- [`FusionUtils.h_docs.md`](./FusionUtils.h_docs.md)
- [`Conv.cpp_docs.md`](./Conv.cpp_docs.md)


## Cross-References

- **File Documentation**: `Attention.cpp_docs.md`
- **Keyword Index**: `Attention.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/mkldnn/xpu`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/mkldnn/xpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/aten/src/ATen/native/mkldnn/xpu`):

- [`FusionUtils.cpp_kw.md_docs.md`](./FusionUtils.cpp_kw.md_docs.md)
- [`qconv.cpp_docs.md_docs.md`](./qconv.cpp_docs.md_docs.md)
- [`Conv.h_kw.md_docs.md`](./Conv.h_kw.md_docs.md)
- [`qconv.h_docs.md_docs.md`](./qconv.h_docs.md_docs.md)
- [`ScaledBlas.cpp_docs.md_docs.md`](./ScaledBlas.cpp_docs.md_docs.md)
- [`Attention.cpp_kw.md_docs.md`](./Attention.cpp_kw.md_docs.md)
- [`FusionUtils.h_docs.md_docs.md`](./FusionUtils.h_docs.md_docs.md)
- [`Blas.cpp_docs.md_docs.md`](./Blas.cpp_docs.md_docs.md)
- [`Conv.cpp_docs.md_docs.md`](./Conv.cpp_docs.md_docs.md)
- [`qconv.cpp_kw.md_docs.md`](./qconv.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `Attention.cpp_docs.md_docs.md`
- **Keyword Index**: `Attention.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
