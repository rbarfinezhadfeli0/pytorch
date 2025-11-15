# Documentation: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/vgg16_training.txt`

## File Metadata

- **Path**: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/vgg16_training.txt`
- **Size**: 5,871 bytes (5.73 KB)
- **Type**: Source File (.txt)
- **Extension**: `.txt`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```
Operator: aten._adaptive_avg_pool2d.default
cnt: 1, ((T([64, 512, 7, 7], f16), [7, 7]), {})
Operator: aten._adaptive_avg_pool2d_backward.default
cnt: 1, ((T([64, 512, 7, 7], f16), T([64, 512, 7, 7], f16)), {})
Operator: aten.addmm.default
cnt: 1, ((T([4096], f16), T([64, 25088], f16), T([25088, 4096], f16, stride=(1, 25088))), {})
cnt: 1, ((T([4096], f16), T([64, 4096], f16), T([4096, 4096], f16, stride=(1, 4096))), {})
cnt: 1, ((T([1000], f16), T([64, 4096], f16), T([4096, 1000], f16, stride=(1, 4096))), {})
Operator: aten.clone.default
cnt: 1, ((T([64, 3, 224, 224], f16),), {})
Operator: aten.convolution.default
cnt: 1, ((T([64, 3, 224, 224], f16), T([64, 3, 3, 3], f16), T([64], f16), [1, 1], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([64, 64, 224, 224], f16), T([64, 64, 3, 3], f16), T([64], f16), [1, 1], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([64, 64, 112, 112], f16), T([128, 64, 3, 3], f16), T([128], f16), [1, 1], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([64, 128, 112, 112], f16), T([128, 128, 3, 3], f16), T([128], f16), [1, 1], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([64, 128, 56, 56], f16), T([256, 128, 3, 3], f16), T([256], f16), [1, 1], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 2, ((T([64, 256, 56, 56], f16), T([256, 256, 3, 3], f16), T([256], f16), [1, 1], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([64, 256, 28, 28], f16), T([512, 256, 3, 3], f16), T([512], f16), [1, 1], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 2, ((T([64, 512, 28, 28], f16), T([512, 512, 3, 3], f16), T([512], f16), [1, 1], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 3, ((T([64, 512, 14, 14], f16), T([512, 512, 3, 3], f16), T([512], f16), [1, 1], [1, 1], [1, 1], False, [0, 0], 1), {})
Operator: aten.convolution_backward.default
cnt: 3, ((T([64, 512, 14, 14], f16), T([64, 512, 14, 14], f16), T([512, 512, 3, 3], f16), [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 2, ((T([64, 512, 28, 28], f16), T([64, 512, 28, 28], f16), T([512, 512, 3, 3], f16), [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 1, ((T([64, 512, 28, 28], f16), T([64, 256, 28, 28], f16), T([512, 256, 3, 3], f16), [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 2, ((T([64, 256, 56, 56], f16), T([64, 256, 56, 56], f16), T([256, 256, 3, 3], f16), [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 1, ((T([64, 256, 56, 56], f16), T([64, 128, 56, 56], f16), T([256, 128, 3, 3], f16), [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 1, ((T([64, 128, 112, 112], f16), T([64, 128, 112, 112], f16), T([128, 128, 3, 3], f16), [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 1, ((T([64, 128, 112, 112], f16), T([64, 64, 112, 112], f16), T([128, 64, 3, 3], f16), [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 1, ((T([64, 64, 224, 224], f16), T([64, 64, 224, 224], f16), T([64, 64, 3, 3], f16), [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 1, ((T([64, 64, 224, 224], f16), T([64, 3, 224, 224], f16), T([64, 3, 3, 3], f16), [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [False, True, True]), {})
Operator: aten.copy_.default
cnt: 1, ((T([64, 3, 224, 224], f16), T([64, 3, 224, 224], f16)), {})
Operator: aten.div.Tensor
cnt: 2, ((T([], f16), 64000), {})
Operator: aten.max_pool2d_with_indices.default
cnt: 1, ((T([64, 64, 224, 224], f16), [2, 2], [2, 2]), {})
cnt: 1, ((T([64, 128, 112, 112], f16), [2, 2], [2, 2]), {})
cnt: 1, ((T([64, 256, 56, 56], f16), [2, 2], [2, 2]), {})
cnt: 1, ((T([64, 512, 28, 28], f16), [2, 2], [2, 2]), {})
cnt: 1, ((T([64, 512, 14, 14], f16), [2, 2], [2, 2]), {})
Operator: aten.max_pool2d_with_indices_backward.default
cnt: 1, ((T([64, 512, 7, 7], f16), T([64, 512, 14, 14], f16), [2, 2], [2, 2], [0, 0], [1, 1], False, T([64, 512, 7, 7], i64)), {})
cnt: 1, ((T([64, 512, 14, 14], f16), T([64, 512, 28, 28], f16), [2, 2], [2, 2], [0, 0], [1, 1], False, T([64, 512, 14, 14], i64)), {})
cnt: 1, ((T([64, 256, 28, 28], f16), T([64, 256, 56, 56], f16), [2, 2], [2, 2], [0, 0], [1, 1], False, T([64, 256, 28, 28], i64)), {})
cnt: 1, ((T([64, 128, 56, 56], f16), T([64, 128, 112, 112], f16), [2, 2], [2, 2], [0, 0], [1, 1], False, T([64, 128, 56, 56], i64)), {})
cnt: 1, ((T([64, 64, 112, 112], f16), T([64, 64, 224, 224], f16), [2, 2], [2, 2], [0, 0], [1, 1], False, T([64, 64, 112, 112], i64)), {})
Operator: aten.mm.default
cnt: 1, ((T([64, 1000], f16, stride=(0, 0)), T([1000, 4096], f16)), {})
cnt: 1, ((T([1000, 64], f16, stride=(0, 0)), T([64, 4096], f16)), {})
cnt: 1, ((T([64, 4096], f16), T([4096, 4096], f16)), {})
cnt: 1, ((T([4096, 64], f16, stride=(1, 4096)), T([64, 4096], f16)), {})
cnt: 1, ((T([64, 4096], f16), T([4096, 25088], f16)), {})
cnt: 1, ((T([4096, 64], f16, stride=(1, 4096)), T([64, 25088], f16)), {})
Operator: aten.relu_.default
cnt: 2, ((T([64, 64, 224, 224], f16),), {})
cnt: 2, ((T([64, 128, 112, 112], f16),), {})
cnt: 3, ((T([64, 256, 56, 56], f16),), {})
cnt: 3, ((T([64, 512, 28, 28], f16),), {})
cnt: 3, ((T([64, 512, 14, 14], f16),), {})
cnt: 2, ((T([64, 4096], f16),), {})
Operator: aten.sum.SymInt
cnt: 1, ((T([64, 1000], f16, stride=(0, 0)), [0], True), {})
cnt: 2, ((T([64, 4096], f16), [0], True), {})
Operator: aten.sum.default
cnt: 1, ((T([64, 1000], f16),), {})
Operator: aten.threshold_backward.default
cnt: 2, ((T([64, 4096], f16), T([64, 4096], f16), 0), {})
cnt: 3, ((T([64, 512, 14, 14], f16), T([64, 512, 14, 14], f16), 0), {})
cnt: 3, ((T([64, 512, 28, 28], f16), T([64, 512, 28, 28], f16), 0), {})
cnt: 3, ((T([64, 256, 56, 56], f16), T([64, 256, 56, 56], f16), 0), {})
cnt: 2, ((T([64, 128, 112, 112], f16), T([64, 128, 112, 112], f16), 0), {})
cnt: 2, ((T([64, 64, 224, 224], f16), T([64, 64, 224, 224], f16), 0), {})

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

- **File Documentation**: `vgg16_training.txt_docs.md`
- **Keyword Index**: `vgg16_training.txt_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
