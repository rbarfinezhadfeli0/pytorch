# Documentation: `torch/csrc/api/src/nn/modules/_functions.cpp`

## File Metadata

- **Path**: `torch/csrc/api/src/nn/modules/_functions.cpp`
- **Size**: 4,534 bytes (4.43 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/util/irange.h>
#include <torch/nn/modules/_functions.h>

using namespace torch::autograd;

namespace torch::nn::functions {

Variable CrossMapLRN2d::forward(
    AutogradContext* ctx,
    const Variable& input,
    const CrossMapLRN2dOptions& options) {
  ctx->saved_data["size"] = options.size();
  ctx->saved_data["alpha"] = options.alpha();
  ctx->saved_data["beta"] = options.beta();
  ctx->saved_data["k"] = options.k();
  ctx->saved_data["scale"] = torch::Tensor();

  TORCH_CHECK(input.dim() == 4);

  ctx->saved_data["scale"] = ctx->saved_data["scale"].toTensor().defined()
      ? ctx->saved_data["scale"]
      : torch::empty({0}, input.options());

  torch::Tensor output = torch::empty({0}, input.options());

  int64_t channels = input.size(1);

  output.resize_as_(input);
  ctx->saved_data["scale"].toTensor().resize_as_(input);

  /// use output storage as temporary buffer
  auto input_square = output;
  torch::pow_out(input_square, input, 2);

  int64_t pre_pad =
      static_cast<int64_t>((ctx->saved_data["size"].toInt() - 1) / 2 + 1);
  int64_t pre_pad_crop = pre_pad > channels ? channels : pre_pad;

  auto scale_first = ctx->saved_data["scale"].toTensor().select(1, 0);
  scale_first.zero_();

  /// compute first feature map normalization
  for (const auto c : c10::irange(pre_pad_crop)) {
    scale_first.add_(input_square.select(1, c));
  }

  /// reuse computations for next feature maps normalization
  /// by adding the next feature map and removing the previous
  torch::Tensor scale_previous, scale_current, square_next, square_previous;

  for (const auto c : c10::irange(1, channels)) {
    scale_previous = ctx->saved_data["scale"].toTensor().select(1, c - 1);
    scale_current = ctx->saved_data["scale"].toTensor().select(1, c);
    scale_current.copy_(scale_previous);

    if (c < channels - pre_pad + 1) {
      square_next = input_square.select(1, c + pre_pad - 1);
      scale_current.add_(square_next, 1);
    }

    if (c > pre_pad) {
      square_previous = input_square.select(1, c - pre_pad);
      scale_current.add_(square_previous, -1);
    }
  }

  ctx->saved_data["scale"]
      .toTensor()
      .mul_(
          ctx->saved_data["alpha"].toDouble() /
          static_cast<double>(ctx->saved_data["size"].toInt()))
      .add_(ctx->saved_data["k"].toInt());

  torch::pow_out(
      output,
      ctx->saved_data["scale"].toTensor(),
      -ctx->saved_data["beta"].toDouble());
  output.mul_(input);

  ctx->save_for_backward({input, output});
  return output;
}

variable_list CrossMapLRN2d::backward(
    AutogradContext* ctx,
    variable_list grad_outputs) {
  auto const& grad_output = grad_outputs[0];
  auto input = ctx->get_saved_variables()[0];
  auto output = ctx->get_saved_variables()[1];
  auto grad_input = torch::empty({0}, grad_output.options());

  int64_t batch_size = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);

  auto padded_ratio = torch::empty(
      {channels + ctx->saved_data["size"].toInt() - 1,
       input_height,
       input_width},
      input.options());
  auto accum_ratio = torch::empty({input_height, input_width}, input.options());
  double cache_ratio_value = 2 * ctx->saved_data["alpha"].toDouble() *
      ctx->saved_data["beta"].toDouble() /
      static_cast<double>(ctx->saved_data["size"].toInt());
  int64_t inversePrePad =
      (ctx->saved_data["size"].toInt() -
       (ctx->saved_data["size"].toInt() - 1) / 2);

  grad_input.resize_as_(input);
  torch::pow_out(
      grad_input,
      ctx->saved_data["scale"].toTensor(),
      -ctx->saved_data["beta"].toDouble())
      .mul_(grad_output);

  padded_ratio.zero_();
  auto padded_ratio_center = padded_ratio.narrow(0, inversePrePad, channels);

  for (const auto n : c10::irange(batch_size)) {
    torch::mul_out(padded_ratio_center, grad_output[n], output[n]);
    padded_ratio_center.div_(ctx->saved_data["scale"].toTensor()[n]);
    torch::sum_out(
        accum_ratio,
        padded_ratio.narrow(0, 0, ctx->saved_data["size"].toInt() - 1),
        0,
        /*keepdim=*/false);
    for (const auto c : c10::irange(channels)) {
      accum_ratio.add_(padded_ratio[c + ctx->saved_data["size"].toInt() - 1]);
      grad_input[n][c].addcmul_(input[n][c], accum_ratio, -cache_ratio_value);
      accum_ratio.add_(padded_ratio[c], -1);
    }
  }

  return variable_list{
      grad_input, Variable(), Variable(), Variable(), Variable()};
}

} // namespace torch::nn::functions

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 1 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/src/nn/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/irange.h`
- `torch/nn/modules/_functions.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

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

Files in the same folder (`torch/csrc/api/src/nn/modules`):

- [`pooling.cpp_docs.md`](./pooling.cpp_docs.md)
- [`linear.cpp_docs.md`](./linear.cpp_docs.md)
- [`padding.cpp_docs.md`](./padding.cpp_docs.md)
- [`rnn.cpp_docs.md`](./rnn.cpp_docs.md)
- [`upsampling.cpp_docs.md`](./upsampling.cpp_docs.md)
- [`embedding.cpp_docs.md`](./embedding.cpp_docs.md)
- [`dropout.cpp_docs.md`](./dropout.cpp_docs.md)
- [`pixelshuffle.cpp_docs.md`](./pixelshuffle.cpp_docs.md)
- [`loss.cpp_docs.md`](./loss.cpp_docs.md)
- [`fold.cpp_docs.md`](./fold.cpp_docs.md)


## Cross-References

- **File Documentation**: `_functions.cpp_docs.md`
- **Keyword Index**: `_functions.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
