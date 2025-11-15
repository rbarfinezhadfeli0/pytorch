# Documentation: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/resnet18_training.txt`

## File Metadata

- **Path**: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/resnet18_training.txt`
- **Size**: 7,054 bytes (6.89 KB)
- **Type**: Source File (.txt)
- **Extension**: `.txt`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```
Operator: aten.add.Tensor
cnt: 1, ((T([16, 512, 7, 7], f16), T([16, 512, 7, 7], f16)), {})
cnt: 2, ((T([16, 256, 14, 14], f16), T([16, 256, 14, 14], f16)), {})
cnt: 2, ((T([16, 128, 28, 28], f16), T([16, 128, 28, 28], f16)), {})
cnt: 3, ((T([16, 64, 56, 56], f16), T([16, 64, 56, 56], f16)), {})
Operator: aten.add_.Tensor
cnt: 2, ((T([16, 64, 56, 56], f16), T([16, 64, 56, 56], f16)), {})
cnt: 2, ((T([16, 128, 28, 28], f16), T([16, 128, 28, 28], f16)), {})
cnt: 2, ((T([16, 256, 14, 14], f16), T([16, 256, 14, 14], f16)), {})
cnt: 2, ((T([16, 512, 7, 7], f16), T([16, 512, 7, 7], f16)), {})
Operator: aten.addmm.default
cnt: 1, ((T([1000], f16), T([16, 512], f16), T([512, 1000], f16, stride=(1, 512))), {})
Operator: aten.clone.default
cnt: 1, ((T([16, 3, 224, 224], f16),), {})
Operator: aten.convolution.default
cnt: 1, ((T([16, 3, 224, 224], f16), T([64, 3, 7, 7], f16), None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), {})
cnt: 4, ((T([16, 64, 56, 56], f16), T([64, 64, 3, 3], f16), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([16, 64, 56, 56], f16), T([128, 64, 3, 3], f16), None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 3, ((T([16, 128, 28, 28], f16), T([128, 128, 3, 3], f16), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([16, 64, 56, 56], f16), T([128, 64, 1, 1], f16), None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([16, 128, 28, 28], f16), T([256, 128, 3, 3], f16), None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 3, ((T([16, 256, 14, 14], f16), T([256, 256, 3, 3], f16), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([16, 128, 28, 28], f16), T([256, 128, 1, 1], f16), None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([16, 256, 14, 14], f16), T([512, 256, 3, 3], f16), None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 3, ((T([16, 512, 7, 7], f16), T([512, 512, 3, 3], f16), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([16, 256, 14, 14], f16), T([512, 256, 1, 1], f16), None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), {})
Operator: aten.convolution_backward.default
cnt: 3, ((T([16, 512, 7, 7], f16), T([16, 512, 7, 7], f16), T([512, 512, 3, 3], f16), [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([16, 512, 7, 7], f16), T([16, 256, 14, 14], f16), T([512, 256, 1, 1], f16), [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([16, 512, 7, 7], f16), T([16, 256, 14, 14], f16), T([512, 256, 3, 3], f16), [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 3, ((T([16, 256, 14, 14], f16), T([16, 256, 14, 14], f16), T([256, 256, 3, 3], f16), [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([16, 256, 14, 14], f16), T([16, 128, 28, 28], f16), T([256, 128, 1, 1], f16), [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([16, 256, 14, 14], f16), T([16, 128, 28, 28], f16), T([256, 128, 3, 3], f16), [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 3, ((T([16, 128, 28, 28], f16), T([16, 128, 28, 28], f16), T([128, 128, 3, 3], f16), [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([16, 128, 28, 28], f16), T([16, 64, 56, 56], f16), T([128, 64, 1, 1], f16), [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([16, 128, 28, 28], f16), T([16, 64, 56, 56], f16), T([128, 64, 3, 3], f16), [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 4, ((T([16, 64, 56, 56], f16), T([16, 64, 56, 56], f16), T([64, 64, 3, 3], f16), [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([16, 64, 112, 112], f16), T([16, 3, 224, 224], f16), T([64, 3, 7, 7], f16), [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]), {})
Operator: aten.copy_.default
cnt: 1, ((T([16, 3, 224, 224], f16), T([16, 3, 224, 224], f16)), {})
Operator: aten.div.Scalar
cnt: 1, ((T([16, 512, 7, 7], f16, stride=(512, 1, 0, 0)), 49), {})
Operator: aten.div.Tensor
cnt: 2, ((T([], f16), 16000), {})
Operator: aten.max_pool2d_with_indices.default
cnt: 1, ((T([16, 64, 112, 112], f16), [3, 3], [2, 2], [1, 1]), {})
Operator: aten.max_pool2d_with_indices_backward.default
cnt: 1, ((T([16, 64, 56, 56], f16), T([16, 64, 112, 112], f16), [3, 3], [2, 2], [1, 1], [1, 1], False, T([16, 64, 56, 56], i64)), {})
Operator: aten.mean.dim
cnt: 1, ((T([16, 512, 7, 7], f16), [-1, -2], True), {})
Operator: aten.mm.default
cnt: 1, ((T([16, 1000], f16, stride=(0, 0)), T([1000, 512], f16)), {})
cnt: 1, ((T([1000, 16], f16, stride=(0, 0)), T([16, 512], f16)), {})
Operator: aten.native_batch_norm.default
cnt: 1, ((T([16, 64, 112, 112], f16), T([64], f16), T([64], f16), T([64], f16), T([64], f16), False, 0.1, 1e-05), {})
cnt: 4, ((T([16, 64, 56, 56], f16), T([64], f16), T([64], f16), T([64], f16), T([64], f16), False, 0.1, 1e-05), {})
cnt: 5, ((T([16, 128, 28, 28], f16), T([128], f16), T([128], f16), T([128], f16), T([128], f16), False, 0.1, 1e-05), {})
cnt: 5, ((T([16, 256, 14, 14], f16), T([256], f16), T([256], f16), T([256], f16), T([256], f16), False, 0.1, 1e-05), {})
cnt: 5, ((T([16, 512, 7, 7], f16), T([512], f16), T([512], f16), T([512], f16), T([512], f16), False, 0.1, 1e-05), {})
Operator: aten.native_batch_norm_backward.default
cnt: 5, ((T([16, 512, 7, 7], f16), T([16, 512, 7, 7], f16), T([512], f16), T([512], f16), T([512], f16), T([512], f32), T([512], f32), False, 1e-05, [True, True, True]), {})
cnt: 5, ((T([16, 256, 14, 14], f16), T([16, 256, 14, 14], f16), T([256], f16), T([256], f16), T([256], f16), T([256], f32), T([256], f32), False, 1e-05, [True, True, True]), {})
cnt: 5, ((T([16, 128, 28, 28], f16), T([16, 128, 28, 28], f16), T([128], f16), T([128], f16), T([128], f16), T([128], f32), T([128], f32), False, 1e-05, [True, True, True]), {})
cnt: 4, ((T([16, 64, 56, 56], f16), T([16, 64, 56, 56], f16), T([64], f16), T([64], f16), T([64], f16), T([64], f32), T([64], f32), False, 1e-05, [True, True, True]), {})
cnt: 1, ((T([16, 64, 112, 112], f16), T([16, 64, 112, 112], f16), T([64], f16), T([64], f16), T([64], f16), T([64], f32), T([64], f32), False, 1e-05, [True, True, True]), {})
Operator: aten.relu_.default
cnt: 1, ((T([16, 64, 112, 112], f16),), {})
cnt: 4, ((T([16, 64, 56, 56], f16),), {})
cnt: 4, ((T([16, 128, 28, 28], f16),), {})
cnt: 4, ((T([16, 256, 14, 14], f16),), {})
cnt: 4, ((T([16, 512, 7, 7], f16),), {})
Operator: aten.sum.SymInt
cnt: 1, ((T([16, 1000], f16, stride=(0, 0)), [0], True), {})
Operator: aten.sum.default
cnt: 1, ((T([16, 1000], f16),), {})
Operator: aten.threshold_backward.default
cnt: 4, ((T([16, 512, 7, 7], f16), T([16, 512, 7, 7], f16), 0), {})
cnt: 4, ((T([16, 256, 14, 14], f16), T([16, 256, 14, 14], f16), 0), {})
cnt: 4, ((T([16, 128, 28, 28], f16), T([16, 128, 28, 28], f16), 0), {})
cnt: 4, ((T([16, 64, 56, 56], f16), T([16, 64, 56, 56], f16), 0), {})
cnt: 1, ((T([16, 64, 112, 112], f16), T([16, 64, 112, 112], f16), 0), {})

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

- **File Documentation**: `resnet18_training.txt_docs.md`
- **Keyword Index**: `resnet18_training.txt_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
