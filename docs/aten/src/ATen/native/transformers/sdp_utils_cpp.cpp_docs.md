# Documentation: `aten/src/ATen/native/transformers/sdp_utils_cpp.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/transformers/sdp_utils_cpp.cpp`
- **Size**: 3,580 bytes (3.50 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <c10/util/Array.h>
namespace sdp {
namespace {

std::array<SDPBackend, num_backends> priority_order_cpp(sdp_params const& params) {
  constexpr std::array<SDPBackend, num_backends> default_order{
      SDPBackend::flash_attention,
      SDPBackend::math};

  return default_order;
}

bool check_head_dim_size_cpp(sdp_params const& params, bool debug) {
  const auto query_size_last = params.query.sym_size(-1);
  const auto key_size_last = params.key.sym_size(-1);
  const auto value_size_last = params.value.sym_size(-1);
  if (!(query_size_last == key_size_last &&
        query_size_last == value_size_last)) {
    if (debug) {
      TORCH_WARN(
          "Flash attention requires q,k,v to have the same last dimension.",
          " Got Query.size(-1): ",
          query_size_last,
          ", Key.size(-1): ",
          params.key.sym_size(-1),
          ", Value.size(-1): ",
          params.value.sym_size(-1),
          " instead.");
    }
    return false;
  }
  return true;
}

bool use_flash_attention_cpp(sdp_params const& params, bool debug) {
  constexpr auto cpp_supported_flash_dtypes =
      c10::array_of<at::ScalarType>(at::kFloat, at::kDouble, at::kBFloat16, at::kHalf);

  // Define gate functions that determine if a flash kernel can be run
  constexpr auto constraints = c10::array_of<bool (*)(sdp_params const&, bool)>(
      check_runtime_disabled_flash,
      check_nested_tensor,
      check_for_dropout,
      check_tensor_shapes,
      check_batch_size_and_num_heads_dense<true /*supports_grouped_query_attention*/>,
      check_attn_mask_shape,
      check_head_dim_size_cpp,
      check_nonzero_sequence_lengths_dense,
      check_last_dim_stride_equals_1_dense<false /*ignore_singleton_dim*/>);
  for (auto& constraint : constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }

  return check_tensor_dtype(params, cpp_supported_flash_dtypes, debug);
}
} // namespace

SDPBackend select_sdp_backend_cpp(sdp_params const& kernel_params) {
  // This function defines the priority order of the different sdp backends
  // 1. Flash Attention
  // 2. Math fallback
  auto& ctx = at::globalContext();
  if (!ctx.userEnabledMathSDP() && !ctx.userEnabledFlashSDP()) {
    return SDPBackend::error;
  }
  // Get ideal kernel ordering
  const auto ordering = priority_order_cpp(kernel_params);

  // Because TORCHCHECK checks if condition is true we negate debug so that
  // The statements will be printed when debug is true
  bool print_debug = false;
  for (auto& backend : ordering) {
    switch (backend) {
      case SDPBackend::flash_attention:
        if (use_flash_attention_cpp(kernel_params, print_debug)) {
          return SDPBackend::flash_attention;
        }
        break;
      case SDPBackend::math:
        if (ctx.userEnabledMathSDP()) {
          return SDPBackend::math;
        }
        break;
      default:
        TORCH_CHECK(false, "Invalid backend");
    }
  }
  // If we have gotten to this point then two things have happened:
  // 1. use_flash_attention did not satisfy the
  // constraints to be ran
  // 2. The user has explicitly disabled the math kernel
  // We then re-run the kernel checks with debug enabled to print out the
  // reason why the kernel was not selected

  print_debug = true;
  TORCH_WARN("Flash attention kernel not used because:");
  use_flash_attention_cpp(kernel_params, print_debug);
  TORCH_CHECK(!print_debug, "No available kernel.  Aborting execution.")
  return SDPBackend::error;
}
} // namespace sdp

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `SDPBackend`, `sdp`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/transformers`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/transformers/sdp_utils_cpp.h`
- `c10/util/Array.h`


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

Files in the same folder (`aten/src/ATen/native/transformers`):

- [`sdp_utils.h_docs.md`](./sdp_utils.h_docs.md)
- [`sdp_utils_cpp.h_docs.md`](./sdp_utils_cpp.h_docs.md)
- [`transformer.cpp_docs.md`](./transformer.cpp_docs.md)
- [`attention.cpp_docs.md`](./attention.cpp_docs.md)
- [`attention.h_docs.md`](./attention.h_docs.md)


## Cross-References

- **File Documentation**: `sdp_utils_cpp.cpp_docs.md`
- **Keyword Index**: `sdp_utils_cpp.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
