# Documentation: `docs/aten/src/ATen/cudnn/AutocastRNN.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/cudnn/AutocastRNN.cpp_docs.md`
- **Size**: 8,015 bytes (7.83 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/cudnn/AutocastRNN.cpp`

## File Metadata

- **Path**: `aten/src/ATen/cudnn/AutocastRNN.cpp`
- **Size**: 5,551 bytes (5.42 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <torch/library.h>

// pulls in AT_CUDNN_ENABLED() as defined by cmake
#include <ATen/cuda/CUDAConfig.h>

#if AT_CUDNN_ENABLED()
#include <ATen/native/cudnn/RNNUtils.h>
#endif


namespace at::autocast {

/********************************************************************************
Autocast wrapper for CuDNN RNNs (the weight reflattening needs special attention)
********************************************************************************/

// To be registered for the "_cudnn_rnn(...)" schema.
// _cudnn_rnn is autograd-exposed (test_autocast_cudnn_rnn in test_cuda.py includes a test to confirm)
static std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>
_cudnn_rnn_cast_reflatten(const Tensor & input,
                          TensorList weight,
                          int64_t weight_stride0,
                          const std::optional<Tensor>& weight_buf_opt,
                          const Tensor& hx,
                          const std::optional<Tensor>& cx,
                          int64_t mode,
                          int64_t hidden_size,
                          int64_t proj_size,
                          int64_t num_layers,
                          bool batch_first,
                          double dropout,
                          bool train,
                          bool bidirectional,
                          IntArrayRef batch_sizes,
                          const std::optional<Tensor>& dropout_state) {
#if AT_CUDNN_ENABLED()
  c10::impl::ExcludeDispatchKeyGuard no_autocast(DispatchKey::Autocast);

  for (const auto& t : weight) {
    TORCH_CHECK(weight[0].scalar_type() == t.scalar_type(), "Weight scalar types do not match.");
  }
  // weight_stride0 is the number of weight tensors per layer and direction, as seen by model.parameters().
  // If bias is enabled, there are 4 such tensors (ih and hh weights, ih and hh biases).
  // If bias is not enabled, there are 2 (ih and hh weights).
  // This organization holds for all rnn types (RNN, GRU, and LSTM). If LSTM with projections is
  // used, additional hr weight is added.
  if (proj_size > 0) {
    TORCH_INTERNAL_ASSERT((weight_stride0 == 3) || (weight_stride0 == 5),
                          "weight_stride0 must be 3 (if no bias) or 5 (if bias) for LSTM with projections.  Received ",
                          weight_stride0);
  } else {
    TORCH_INTERNAL_ASSERT((weight_stride0 == 2) || (weight_stride0 == 4),
                          "weight_stride0 must be 2 (if no bias) or 4 (if bias).  Received ",
                          weight_stride0);
  }


  Tensor weight_buf, redispatch_weight_buf;
  std::vector<Tensor> redispatch_weight;
  // There's an implicit contract here with native/cudnn/RNN.cpp:_cudnn_impl, which calls at:_cudnn_rnn.
  // Code here assumes if _cudnn_impl passes weight_buf_opt containing a defined tensor, that tensor
  // is valid flat storage of the weights in their incoming dtype.
  if (weight_buf_opt.has_value()) {
    weight_buf = *weight_buf_opt;
  }
  bool needs_cast_and_flatten = (weight_buf.defined() ?
                                 // weight_buf is valid.  Only change it if it's eligible and not already FP16.
                                 is_eligible(weight_buf) && (weight_buf.scalar_type() != at::kHalf) :
                                 // weight_buf is not valid.  Only create it if other weights are eligible and not already FP16.
                                 is_eligible(weight[0]) && (weight[0].scalar_type() != at::kHalf));
  if (needs_cast_and_flatten) {
    // Casts weight tensors to FP16 and ensures all weights for all layers are views into a large flat buffer,
    // with the right locations and layouts expected by cudnn.
    // This is (and should be) autograd-exposed.
    bool include_bias = true;
    if (weight_stride0 == 2 || (weight_stride0 == 3 && proj_size > 0)) {
      include_bias = false;
    }
    std::tie(redispatch_weight_buf, redispatch_weight) =
        at::native::cudnn_rnn::copy_weights_to_flat_buf_views(
            weight,
            weight_stride0,
            input.size(-1),
            mode,
            hidden_size,
            proj_size,
            num_layers,
            batch_first,
            bidirectional,
            /*flat_buf_datatype=*/at::native::getCudnnDataTypeFromScalarType(at::kHalf), // could just hardcode CUDNN_DATA_HALF
            /*flat_buf_options=*/weight[0].options().dtype(at::kHalf),
            /*set_orig_weights_to_flat_buf=*/false,
            /*allow_type_change=*/true,
            /*include_bias=*/include_bias);
  }
  return at::_cudnn_rnn(
      cached_cast(at::kHalf, input),
      needs_cast_and_flatten ? TensorList(redispatch_weight) : weight,
      weight_stride0,
      needs_cast_and_flatten ? redispatch_weight_buf : weight_buf,
      cached_cast(at::kHalf, hx),
      cached_cast(at::kHalf, cx),
      mode,
      hidden_size,
      proj_size,
      num_layers,
      batch_first,
      dropout,
      train,
      bidirectional,
      batch_sizes,
      dropout_state);
#else // AT_CUDNN_ENABLED()
  TORCH_CHECK(false, "autocast::_cudnn_rnn_cast_reflatten: ATen not compiled with cuDNN support");
  return {Tensor{}, Tensor{}, Tensor{}, Tensor{}, Tensor{}}; // never reached, placates the compiler
#endif // AT_CUDNN_ENABLED()
}

namespace {
TORCH_LIBRARY_IMPL(aten, Autocast, m) {
  m.impl("_cudnn_rnn",
         TORCH_FN((&at::autocast::_cudnn_rnn_cast_reflatten)));
}
} // anonymous namespace

} // namespace at::autocast

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 13 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/cudnn`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ATen.h`
- `ATen/autocast_mode.h`
- `torch/library.h`
- `ATen/cuda/CUDAConfig.h`
- `ATen/native/cudnn/RNNUtils.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`aten/src/ATen/cudnn`):

- [`Descriptors.cpp_docs.md`](./Descriptors.cpp_docs.md)
- [`Handle.h_docs.md`](./Handle.h_docs.md)
- [`Handles.h_docs.md`](./Handles.h_docs.md)
- [`Types.h_docs.md`](./Types.h_docs.md)
- [`Descriptors.h_docs.md`](./Descriptors.h_docs.md)
- [`Utils.h_docs.md`](./Utils.h_docs.md)
- [`Handle.cpp_docs.md`](./Handle.cpp_docs.md)
- [`Types.cpp_docs.md`](./Types.cpp_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)


## Cross-References

- **File Documentation**: `AutocastRNN.cpp_docs.md`
- **Keyword Index**: `AutocastRNN.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/cudnn`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/cudnn`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/cudnn`):

- [`Handle.cpp_docs.md_docs.md`](./Handle.cpp_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`Descriptors.h_kw.md_docs.md`](./Descriptors.h_kw.md_docs.md)
- [`Types.cpp_kw.md_docs.md`](./Types.cpp_kw.md_docs.md)
- [`Handles.h_docs.md_docs.md`](./Handles.h_docs.md_docs.md)
- [`Handle.h_kw.md_docs.md`](./Handle.h_kw.md_docs.md)
- [`Descriptors.cpp_docs.md_docs.md`](./Descriptors.cpp_docs.md_docs.md)
- [`cudnn-wrapper.h_kw.md_docs.md`](./cudnn-wrapper.h_kw.md_docs.md)
- [`Handles.h_kw.md_docs.md`](./Handles.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `AutocastRNN.cpp_docs.md_docs.md`
- **Keyword Index**: `AutocastRNN.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
