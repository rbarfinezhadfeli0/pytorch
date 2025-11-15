# Documentation: `torch/csrc/jit/passes/utils/op_registry.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/utils/op_registry.cpp`
- **Size**: 5,750 bytes (5.62 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/passes/utils/op_registry.h>

// Location for Commonly Used Shape registries

namespace torch::jit {

// Requirements:
//   dims           : preserved from the first argument
//   scalar type    : preserved from the first argument (doesn't have to
//                    match other arguments)
//   device         : always matching and preserved
//   tensor inputs  : *
//   tensor outputs : 1
// NB: those ops (with slight adjustments) are good candidates for restarts.
//     Knowing the type and device of weights or biases is usually enough to
//     infer the output type.
std::shared_ptr<OperatorSet> nn_ops_first_input_preserving() {
  std::shared_ptr<OperatorSet> ops = std::make_shared<OperatorSet>(OperatorSet{
      "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor",
      "aten::conv1d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor",
      "aten::conv2d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor",
      "aten::conv3d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor",
      "aten::conv_tbc(Tensor self, Tensor weight, Tensor bias, int pad) -> Tensor",
      "aten::conv_transpose1d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding, int groups, int[] dilation) -> Tensor",
      "aten::conv_transpose2d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding, int groups, int[] dilation) -> Tensor",
      "aten::conv_transpose3d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding, int groups, int[] dilation) -> Tensor",
      "aten::convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor",
      "aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled) -> Tensor", // deprecated _convolution
      "aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) -> Tensor",
      "aten::adaptive_avg_pool1d(Tensor self, int[] output_size) -> Tensor",
      "aten::adaptive_avg_pool2d(Tensor self, int[] output_size) -> Tensor",
      "aten::adaptive_avg_pool3d(Tensor self, int[] output_size) -> Tensor",
      "aten::avg_pool1d(Tensor self, int[] kernel_size, int[] stride, int[] padding, bool ceil_mode, bool count_include_pad) -> Tensor",
      "aten::avg_pool2d(Tensor self, int[] kernel_size, int[] stride, int[] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor",
      "aten::avg_pool3d(Tensor self, int[] kernel_size, int[] stride, int[] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor",
      "aten::max_pool1d(Tensor self, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode) -> Tensor",
      "aten::max_pool2d(Tensor self, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode) -> Tensor",
      "aten::max_pool3d(Tensor self, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode) -> Tensor",
      "aten::max_unpool2d(Tensor self, Tensor indices, int[] output_size) -> Tensor",
      "aten::max_unpool3d(Tensor self, Tensor indices, int[] output_size, int[] stride, int[] padding) -> Tensor",
      "aten::reflection_pad1d(Tensor self, int[] padding) -> Tensor",
      "aten::reflection_pad2d(Tensor self, int[] padding) -> Tensor",
      "aten::reflection_pad3d(Tensor self, int[] padding) -> Tensor",
      "aten::replication_pad1d(Tensor self, int[] padding) -> Tensor",
      "aten::replication_pad2d(Tensor self, int[] padding) -> Tensor",
      "aten::replication_pad3d(Tensor self, int[] padding) -> Tensor",
      "aten::upsample_bilinear2d(Tensor self, int[] output_size, bool align_corners, float? scales_h, float? scales_w) -> Tensor",
      "aten::upsample_linear1d(Tensor self, int[] output_size, bool align_corners, float? scales) -> Tensor",
      "aten::upsample_nearest1d(Tensor self, int[] output_size, float? scales) -> Tensor",
      "aten::upsample_nearest2d(Tensor self, int[] output_size, float? scales_h, float? scales_w) -> Tensor",
      "aten::upsample_nearest3d(Tensor self, int[] output_size, float? scales_d, float? scales_h, float? scales_w) -> Tensor",
      "aten::upsample_trilinear3d(Tensor self, int[] output_size, bool align_corners, float? scales_d, float? scales_h, float? scales_w) -> Tensor",
      "aten::prelu(Tensor self, Tensor weight) -> Tensor",

      // Added because Hardswish is really hard to convert to metatensors
      "aten::hardswish(Tensor self) -> Tensor",
      "aten::hardswish_(Tensor self) -> Tensor",
  });
  return ops;
}

// Requirements:
//   dims           : Changed from first argument
//   scalar type    : preserved from the first argument
//   device         : always matching and preserved
//   tensor inputs  : 1
//   tensor outputs : 1
std::shared_ptr<OperatorSet> ops_one_tensor_in_shape_transform() {
  std::shared_ptr<OperatorSet> ops = std::make_shared<OperatorSet>(OperatorSet{
      "aten::flatten(Tensor self, int start_dim, int end_dim) -> Tensor",
  });
  return ops;
}
} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/passes/utils/op_registry.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`torch/csrc/jit/passes/utils`):

- [`op_registry.h_docs.md`](./op_registry.h_docs.md)
- [`optimization_utils.h_docs.md`](./optimization_utils.h_docs.md)
- [`optimization_utils.cpp_docs.md`](./optimization_utils.cpp_docs.md)
- [`subgraph_utils.cpp_docs.md`](./subgraph_utils.cpp_docs.md)
- [`check_alias_annotation.h_docs.md`](./check_alias_annotation.h_docs.md)
- [`memory_dag.h_docs.md`](./memory_dag.h_docs.md)
- [`memory_dag.cpp_docs.md`](./memory_dag.cpp_docs.md)
- [`subgraph_utils.h_docs.md`](./subgraph_utils.h_docs.md)
- [`check_alias_annotation.cpp_docs.md`](./check_alias_annotation.cpp_docs.md)


## Cross-References

- **File Documentation**: `op_registry.cpp_docs.md`
- **Keyword Index**: `op_registry.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
