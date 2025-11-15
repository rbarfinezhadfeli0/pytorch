# Documentation: `docs/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/native/Extra.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/native/Extra.cpp_docs.md`
- **Size**: 8,009 bytes (7.82 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/native/Extra.cpp`

## File Metadata

- **Path**: `test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/native/Extra.cpp`
- **Size**: 5,746 bytes (5.61 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```cpp
#include "Extra.h"

namespace at::native::openreg {

at::Tensor quantize_per_tensor(
    const at::Tensor& self,
    double scale,
    int64_t zero_point,
    at::ScalarType dtype) {
  return at::native::quantize_per_tensor(self, scale, zero_point, dtype);
}

int64_t _fused_sdp_choice(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  auto backend = sdp::SDPBackend::overrideable;
  return static_cast<int64_t>(backend);
}

void quantize_tensor_per_tensor_affine_stub(
    const at::Tensor& rtensor,
    at::Tensor& qtensor,
    double scale,
    int64_t zero_point) {}

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
_scaled_dot_product_fused_attention_overrideable(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale) {
  const int64_t batch_size = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t head_dim_v = value.size(3);
  const int64_t max_seqlen_q = query.size(2);
  const int64_t max_seqlen_kv = key.size(2);

  auto opts = query.options();
  auto output =
      at::empty({batch_size, num_heads, max_seqlen_q, head_dim_v}, opts);
  auto logsumexp =
      at::empty({batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
  auto debug_attn_mask = at::empty(
      {batch_size, num_heads, max_seqlen_q, max_seqlen_kv},
      opts.dtype(at::kFloat));
  auto philox_seed = at::empty({}, at::dtype(at::kLong));
  auto philox_offset = at::empty({}, at::dtype(at::kLong));

  return std::make_tuple(
      output,
      logsumexp,
      at::Tensor(),
      at::Tensor(),
      max_seqlen_q,
      max_seqlen_kv,
      philox_seed,
      philox_offset,
      debug_attn_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_fused_attention_overrideable_backward(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& attn_bias,
    std::array<bool, 4> grad_input_mask,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const at::Tensor& cum_seq_q,
    const at::Tensor& cum_seq_k,
    int64_t max_q,
    int64_t max_k,
    double dropout_p,
    bool is_causal,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    std::optional<double> scale) {
  return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
      at::empty_like(query),
      at::empty_like(key),
      at::empty_like(value),
      at::empty_like(attn_bias));
}

namespace {
struct CustomAutogradFnReturnsSelf
    : public torch::autograd::Function<CustomAutogradFnReturnsSelf> {
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor self) {
    return self;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    return {grad_output[0] * 0.5};
  }
};

struct CustomAutogradFnAliasing
    : public torch::autograd::Function<CustomAutogradFnAliasing> {
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor self) {
    return self.view_symint(self.sym_sizes());
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    return {grad_output[0] * 0.5};
  }
};
} // namespace

at::Tensor custom_autograd_fn_returns_self(at::Tensor x) {
  return CustomAutogradFnReturnsSelf::apply(x);
}

at::Tensor custom_autograd_fn_aliasing(at::Tensor x) {
  return CustomAutogradFnAliasing::apply(x);
}

/*
 This implementation is only used to test stub registration, so not all
 capabilities are fully supported.

 Current Limitations:
 - dtype: Float only
 - input tensor: must be contiguous layout
*/
// LITERALINCLUDE START: STUB ABS
void abs_kernel(at::TensorIteratorBase& iter) {
  TORCH_CHECK(iter.ntensors() == 2, "Abs kernel expects 2 tensors");
  TORCH_CHECK(
      iter.common_dtype() == at::ScalarType::Float,
      "Abs kernel only supports float type");

  auto& output_tensor = iter.tensor(0);
  auto& input_tensor = iter.tensor(1);

  TORCH_CHECK(
      input_tensor.sizes() == output_tensor.sizes(),
      "Input and output tensor sizes must match.");

  auto abs_loop = [](float* out_ptr, const float* in_ptr, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
      out_ptr[i] = std::abs(in_ptr[i]);
    }
  };

  MemoryGuard guard(input_tensor, output_tensor);

  if (iter.is_contiguous()) {
    abs_loop(
        static_cast<float*>(iter.data_ptr(0)),
        static_cast<float*>(iter.data_ptr(1)),
        iter.numel());
  } else {
    TORCH_CHECK(
        input_tensor.is_contiguous(), "Input tensor must be contiguous.")

    auto output = at::empty(
        input_tensor.sizes(),
        input_tensor.options().memory_format(
            input_tensor.suggest_memory_format()));

    MemoryGuard guard(output);

    abs_loop(
        static_cast<float*>(output.data_ptr()),
        static_cast<float*>(iter.data_ptr(1)),
        iter.numel());

    output_tensor.copy_(output);
  }
}
// LITERALINCLUDE END: STUB ABS

at::Tensor& abs_out(const at::Tensor& self, at::Tensor& out) {
  return at::native::abs_out(self, out);
}

at::Tensor custom_abs(at::Tensor x) {
  return at::abs(x);
}

} // namespace at::native::openreg

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 13 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `CustomAutogradFnReturnsSelf`, `CustomAutogradFnAliasing`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/native`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `Extra.h`


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

This is a test file. Run it with:

```bash
python test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/native/Extra.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/native`):

- [`Minimal.cpp_docs.md`](./Minimal.cpp_docs.md)
- [`Common.h_docs.md`](./Common.h_docs.md)
- [`Minimal.h_docs.md`](./Minimal.h_docs.md)
- [`Extra.h_docs.md`](./Extra.h_docs.md)


## Cross-References

- **File Documentation**: `Extra.cpp_docs.md`
- **Keyword Index**: `Extra.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/native`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/native`, which is part of the **core PyTorch library**.



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

This is a test file. Run it with:

```bash
python docs/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/native/Extra.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/native`):

- [`Minimal.h_kw.md_docs.md`](./Minimal.h_kw.md_docs.md)
- [`Extra.h_kw.md_docs.md`](./Extra.h_kw.md_docs.md)
- [`Common.h_kw.md_docs.md`](./Common.h_kw.md_docs.md)
- [`Minimal.cpp_kw.md_docs.md`](./Minimal.cpp_kw.md_docs.md)
- [`Minimal.h_docs.md_docs.md`](./Minimal.h_docs.md_docs.md)
- [`Common.h_docs.md_docs.md`](./Common.h_docs.md_docs.md)
- [`Extra.cpp_kw.md_docs.md`](./Extra.cpp_kw.md_docs.md)
- [`Extra.h_docs.md_docs.md`](./Extra.h_docs.md_docs.md)
- [`Minimal.cpp_docs.md_docs.md`](./Minimal.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `Extra.cpp_docs.md_docs.md`
- **Keyword Index**: `Extra.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
