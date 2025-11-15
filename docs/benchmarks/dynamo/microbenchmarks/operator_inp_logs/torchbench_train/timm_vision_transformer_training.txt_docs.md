# Documentation: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/timm_vision_transformer_training.txt`

## File Metadata

- **Path**: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/timm_vision_transformer_training.txt`
- **Size**: 4,679 bytes (4.57 KB)
- **Type**: Source File (.txt)
- **Extension**: `.txt`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```
Operator: aten._softmax.default
cnt: 12, ((T([8, 6, 197, 197], f16), -1, False), {})
Operator: aten._softmax_backward_data.default
cnt: 12, ((T([8, 6, 197, 197], f16), T([8, 6, 197, 197], f16), -1, f16), {})
Operator: aten._unsafe_view.default
cnt: 36, ((T([8, 6, 197, 64], f16), [48, 197, 64]), {})
cnt: 12, ((T([8, 6, 64, 197], f16), [48, 64, 197]), {})
cnt: 12, ((T([48, 197, 197], f16), [8, 6, 197, 197]), {})
cnt: 12, ((T([48, 197, 64], f16), [8, 6, 197, 64]), {})
cnt: 12, ((T([8, 197, 6, 64], f16), [8, 197, 384]), {})
cnt: 12, ((T([8, 197, 3, 6, 64], f16), [8, 197, 1152]), {})
Operator: aten.add.Tensor
cnt: 1, ((T([8, 197, 384], f16), T([1, 197, 384], f16)), {})
cnt: 48, ((T([8, 197, 384], f16), T([8, 197, 384], f16)), {})
Operator: aten.addmm.default
cnt: 12, ((T([1152], f16), T([1576, 384], f16), T([384, 1152], f16, stride=(1, 384))), {})
cnt: 12, ((T([384], f16), T([1576, 384], f16), T([384, 384], f16, stride=(1, 384))), {})
cnt: 12, ((T([1536], f16), T([1576, 384], f16), T([384, 1536], f16, stride=(1, 384))), {})
cnt: 12, ((T([384], f16), T([1576, 1536], f16), T([1536, 384], f16, stride=(1, 1536))), {})
cnt: 1, ((T([1000], f16), T([8, 384], f16, stride=(75648, 1)), T([384, 1000], f16, stride=(1, 384))), {})
Operator: aten.bmm.default
cnt: 12, ((T([48, 197, 64], f16), T([48, 64, 197], f16)), {})
cnt: 12, ((T([48, 197, 197], f16), T([48, 197, 64], f16)), {})
cnt: 12, ((T([48, 197, 197], f16, stride=(38809, 1, 197)), T([48, 197, 64], f16)), {})
cnt: 12, ((T([48, 197, 64], f16), T([48, 64, 197], f16, stride=(12608, 1, 64))), {})
cnt: 12, ((T([48, 64, 197], f16, stride=(12608, 1, 64)), T([48, 197, 197], f16)), {})
cnt: 12, ((T([48, 197, 197], f16), T([48, 197, 64], f16, stride=(12608, 1, 197))), {})
Operator: aten.cat.default
cnt: 1, (([T([8, 1, 384], f16, stride=(0, 384, 1)), T([8, 196, 384], f16, stride=(75264, 1, 196))], 1), {})
Operator: aten.clone.default
cnt: 1, ((T([8, 3, 224, 224], f16),), {})
Operator: aten.convolution.default
cnt: 1, ((T([8, 3, 224, 224], f16), T([384, 3, 16, 16], f16), T([384], f16), [16, 16], [0, 0], [1, 1], False, [0, 0], 1), {})
Operator: aten.convolution_backward.default
cnt: 1, ((T([8, 384, 14, 14], f16, stride=(75648, 1, 5376, 384)), T([8, 3, 224, 224], f16), T([384, 3, 16, 16], f16), [384], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]), {})
Operator: aten.copy_.default
cnt: 1, ((T([8, 3, 224, 224], f16), T([8, 3, 224, 224], f16)), {})
Operator: aten.div.Tensor
cnt: 2, ((T([], f16), 8000), {})
Operator: aten.gelu.default
cnt: 12, ((T([8, 197, 1536], f16),), {})
Operator: aten.gelu_backward.default
cnt: 12, ((T([8, 197, 1536], f16), T([8, 197, 1536], f16)), {})
Operator: aten.mm.default
cnt: 1, ((T([8, 1000], f16, stride=(0, 0)), T([1000, 384], f16)), {})
cnt: 1, ((T([1000, 8], f16, stride=(0, 0)), T([8, 384], f16, stride=(75648, 1))), {})
cnt: 12, ((T([1576, 384], f16), T([384, 1536], f16)), {})
cnt: 12, ((T([384, 1576], f16, stride=(1, 384)), T([1576, 1536], f16)), {})
cnt: 12, ((T([1576, 1536], f16), T([1536, 384], f16)), {})
cnt: 12, ((T([1536, 1576], f16, stride=(1, 1536)), T([1576, 384], f16)), {})
cnt: 12, ((T([1576, 384], f16), T([384, 384], f16)), {})
cnt: 12, ((T([384, 1576], f16, stride=(1, 384)), T([1576, 384], f16)), {})
cnt: 12, ((T([1576, 1152], f16), T([1152, 384], f16)), {})
cnt: 12, ((T([1152, 1576], f16, stride=(1, 1152)), T([1576, 384], f16)), {})
Operator: aten.mul.Tensor
cnt: 24, ((T([8, 6, 197, 197], f16), 0.125), {})
Operator: aten.native_layer_norm.default
cnt: 25, ((T([8, 197, 384], f16), [384], T([384], f16), T([384], f16), 1e-06), {})
Operator: aten.native_layer_norm_backward.default
cnt: 25, ((T([8, 197, 384], f16), T([8, 197, 384], f16), [384], T([8, 197, 1], f32), T([8, 197, 1], f32), T([384], f16), T([384], f16), [True, True, True]), {})
Operator: aten.select_backward.default
cnt: 1, ((T([8, 384], f16), [8, 197, 384], 1, 0), {})
Operator: aten.slice_backward.default
cnt: 1, ((T([8, 197, 384], f16), [8, 197, 384], 0, 0, 9223372036854775807, 1), {})
Operator: aten.stack.default
cnt: 12, (([T([8, 6, 197, 64], f16), T([8, 6, 197, 64], f16, stride=(75648, 12608, 1, 197)), T([8, 6, 197, 64], f16)],), {})
Operator: aten.sum.SymInt
cnt: 1, ((T([8, 1000], f16, stride=(0, 0)), [0], True), {})
cnt: 24, ((T([1576, 384], f16), [0], True), {})
cnt: 12, ((T([1576, 1536], f16), [0], True), {})
cnt: 12, ((T([1576, 1152], f16), [0], True), {})
cnt: 1, ((T([8, 197, 384], f16), [0], True), {})
cnt: 1, ((T([8, 1, 384], f16, stride=(75648, 384, 1)), [0], True), {})
Operator: aten.sum.default
cnt: 1, ((T([8, 1000], f16),), {})
Operator: aten.unbind.int
cnt: 12, ((T([3, 8, 6, 197, 64], f16, stride=(384, 226944, 64, 1152, 1)),), {})

```



## High-Level Overview

This file is part of the PyTorch framework located at `benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


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

Files in the same folder (`benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train`):

- [`yolov3_training.txt_docs.md`](./yolov3_training.txt_docs.md)
- [`pytorch_stargan_training.txt_docs.md`](./pytorch_stargan_training.txt_docs.md)
- [`tts_angular_training.txt_docs.md`](./tts_angular_training.txt_docs.md)
- [`squeezenet1_1_training.txt_docs.md`](./squeezenet1_1_training.txt_docs.md)
- [`attention_is_all_you_need_pytorch_training.txt_docs.md`](./attention_is_all_you_need_pytorch_training.txt_docs.md)
- [`timm_regnet_training.txt_docs.md`](./timm_regnet_training.txt_docs.md)
- [`dcgan_training.txt_docs.md`](./dcgan_training.txt_docs.md)
- [`pytorch_struct_training.txt_docs.md`](./pytorch_struct_training.txt_docs.md)
- [`Background_Matting_training.txt_docs.md`](./Background_Matting_training.txt_docs.md)
- [`fambench_dlrm_training.txt_docs.md`](./fambench_dlrm_training.txt_docs.md)


## Cross-References

- **File Documentation**: `timm_vision_transformer_training.txt_docs.md`
- **Keyword Index**: `timm_vision_transformer_training.txt_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
