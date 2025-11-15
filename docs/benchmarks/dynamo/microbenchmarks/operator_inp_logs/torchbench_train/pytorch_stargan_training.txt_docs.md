# Documentation: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/pytorch_stargan_training.txt`

## File Metadata

- **Path**: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/pytorch_stargan_training.txt`
- **Size**: 5,776 bytes (5.64 KB)
- **Type**: Source File (.txt)
- **Extension**: `.txt`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```
Operator: aten.add.Tensor
cnt: 12, ((T([16, 256, 32, 32], f16), T([16, 256, 32, 32], f16)), {})
Operator: aten.cat.default
cnt: 1, (([T([16, 3, 128, 128], f16), T([16, 5, 128, 128], f16)], 1), {})
Operator: aten.clone.default
cnt: 1, ((T([16, 3, 128, 128], f16),), {})
cnt: 1, ((T([16, 5], f16),), {})
Operator: aten.convolution.default
cnt: 1, ((T([16, 8, 128, 128], f16), T([64, 8, 7, 7], f16), None, [1, 1], [3, 3], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([16, 64, 128, 128], f16), T([128, 64, 4, 4], f16), None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([16, 128, 64, 64], f16), T([256, 128, 4, 4], f16), None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 12, ((T([16, 256, 32, 32], f16), T([256, 256, 3, 3], f16), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([16, 256, 32, 32], f16), T([256, 128, 4, 4], f16), None, [2, 2], [1, 1], [1, 1], True, [0, 0], 1), {})
cnt: 1, ((T([16, 128, 64, 64], f16), T([128, 64, 4, 4], f16), None, [2, 2], [1, 1], [1, 1], True, [0, 0], 1), {})
cnt: 1, ((T([16, 64, 128, 128], f16), T([3, 64, 7, 7], f16), None, [1, 1], [3, 3], [1, 1], False, [0, 0], 1), {})
Operator: aten.convolution_backward.default
cnt: 1, ((T([16, 3, 128, 128], f16), T([16, 64, 128, 128], f16), T([3, 64, 7, 7], f16), [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([16, 64, 128, 128], f16), T([16, 128, 64, 64], f16), T([128, 64, 4, 4], f16), [0], [2, 2], [1, 1], [1, 1], True, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([16, 128, 64, 64], f16), T([16, 256, 32, 32], f16), T([256, 128, 4, 4], f16), [0], [2, 2], [1, 1], [1, 1], True, [0, 0], 1, [True, True, False]), {})
cnt: 12, ((T([16, 256, 32, 32], f16), T([16, 256, 32, 32], f16), T([256, 256, 3, 3], f16), [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([16, 256, 32, 32], f16), T([16, 128, 64, 64], f16), T([256, 128, 4, 4], f16), [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([16, 128, 64, 64], f16), T([16, 64, 128, 128], f16), T([128, 64, 4, 4], f16), [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([16, 64, 128, 128], f16), T([16, 8, 128, 128], f16), T([64, 8, 7, 7], f16), [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]), {})
Operator: aten.copy_.default
cnt: 1, ((T([16, 3, 128, 128], f16), T([16, 3, 128, 128], f16)), {})
cnt: 1, ((T([16, 5], f16), T([16, 5], f16)), {})
cnt: 4, ((T([64], f16), T([64], f16)), {})
cnt: 4, ((T([128], f16), T([128], f16)), {})
cnt: 26, ((T([256], f16), T([256], f16)), {})
cnt: 4, ((T([16, 64, 128, 128], f16), T([16, 64, 128, 128], f16)), {})
cnt: 2, ((T([1, 1024, 128, 128], f16), T([1, 1024, 128, 128], f16)), {})
cnt: 4, ((T([16, 128, 64, 64], f16), T([16, 128, 64, 64], f16)), {})
cnt: 2, ((T([1, 2048, 64, 64], f16), T([1, 2048, 64, 64], f16)), {})
cnt: 14, ((T([16, 256, 32, 32], f16), T([16, 256, 32, 32], f16)), {})
cnt: 7, ((T([1, 4096, 32, 32], f16), T([1, 4096, 32, 32], f16)), {})
Operator: aten.div.Tensor
cnt: 2, ((T([], f16), 786432), {})
Operator: aten.mean.dim
cnt: 4, ((T([16, 64], f16), [0]), {})
cnt: 4, ((T([16, 128], f16), [0]), {})
cnt: 26, ((T([16, 256], f16), [0]), {})
Operator: aten.native_batch_norm.default
cnt: 2, ((T([1, 1024, 128, 128], f16), T([1024], f16), T([1024], f16), T([1024], f16), T([1024], f16), False, 0.1, 1e-05), {})
cnt: 2, ((T([1, 2048, 64, 64], f16), T([2048], f16), T([2048], f16), T([2048], f16), T([2048], f16), False, 0.1, 1e-05), {})
cnt: 13, ((T([1, 4096, 32, 32], f16), T([4096], f16), T([4096], f16), T([4096], f16), T([4096], f16), False, 0.1, 1e-05), {})
Operator: aten.native_batch_norm_backward.default
cnt: 2, ((T([1, 1024, 128, 128], f16), T([1, 1024, 128, 128], f16), T([1024], f16), T([1024], f16), T([1024], f16), T([1024], f32), T([1024], f32), False, 1e-05, [True, True, True]), {})
cnt: 2, ((T([1, 2048, 64, 64], f16), T([1, 2048, 64, 64], f16), T([2048], f16), T([2048], f16), T([2048], f16), T([2048], f32), T([2048], f32), False, 1e-05, [True, True, True]), {})
cnt: 13, ((T([1, 4096, 32, 32], f16), T([1, 4096, 32, 32], f16), T([4096], f16), T([4096], f16), T([4096], f16), T([4096], f32), T([4096], f32), False, 1e-05, [True, True, True]), {})
Operator: aten.new_empty_strided.default
cnt: 2, ((T([1, 1024, 128, 128], f16), [1, 1024, 128, 128], [16777216, 16384, 128, 1]), {})
cnt: 2, ((T([1, 2048, 64, 64], f16), [1, 2048, 64, 64], [8388608, 4096, 64, 1]), {})
cnt: 7, ((T([1, 4096, 32, 32], f16), [1, 4096, 32, 32], [4194304, 1024, 32, 1]), {})
Operator: aten.new_zeros.default
cnt: 2, ((T([16, 64, 128, 128], f16), [16777216]), {})
cnt: 2, ((T([16, 128, 64, 64], f16), [8388608]), {})
cnt: 7, ((T([16, 256, 32, 32], f16), [4194304]), {})
Operator: aten.relu_.default
cnt: 2, ((T([16, 64, 128, 128], f16),), {})
cnt: 2, ((T([16, 128, 64, 64], f16),), {})
cnt: 7, ((T([16, 256, 32, 32], f16),), {})
Operator: aten.repeat.default
cnt: 1, ((T([16, 5, 1, 1], f16), [1, 1, 128, 128]), {})
cnt: 8, ((T([64], f16), [16]), {})
cnt: 8, ((T([128], f16), [16]), {})
cnt: 52, ((T([256], f16), [16]), {})
Operator: aten.sum.default
cnt: 1, ((T([16, 3, 128, 128], f16),), {})
Operator: aten.sum.dim_IntList
cnt: 4, ((T([16, 64], f16), [0]), {})
cnt: 4, ((T([16, 128], f16), [0]), {})
cnt: 26, ((T([16, 256], f16), [0]), {})
Operator: aten.tanh.default
cnt: 1, ((T([16, 3, 128, 128], f16),), {})
Operator: aten.tanh_backward.default
cnt: 1, ((T([16, 3, 128, 128], f16, stride=(0, 0, 0, 0)), T([16, 3, 128, 128], f16)), {})
Operator: aten.threshold_backward.default
cnt: 2, ((T([16, 64, 128, 128], f16), T([16, 64, 128, 128], f16), 0), {})
cnt: 2, ((T([16, 128, 64, 64], f16), T([16, 128, 64, 64], f16), 0), {})
cnt: 7, ((T([16, 256, 32, 32], f16), T([16, 256, 32, 32], f16), 0), {})

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
- [`tts_angular_training.txt_docs.md`](./tts_angular_training.txt_docs.md)
- [`squeezenet1_1_training.txt_docs.md`](./squeezenet1_1_training.txt_docs.md)
- [`attention_is_all_you_need_pytorch_training.txt_docs.md`](./attention_is_all_you_need_pytorch_training.txt_docs.md)
- [`timm_regnet_training.txt_docs.md`](./timm_regnet_training.txt_docs.md)
- [`dcgan_training.txt_docs.md`](./dcgan_training.txt_docs.md)
- [`pytorch_struct_training.txt_docs.md`](./pytorch_struct_training.txt_docs.md)
- [`Background_Matting_training.txt_docs.md`](./Background_Matting_training.txt_docs.md)
- [`fambench_dlrm_training.txt_docs.md`](./fambench_dlrm_training.txt_docs.md)


## Cross-References

- **File Documentation**: `pytorch_stargan_training.txt_docs.md`
- **Keyword Index**: `pytorch_stargan_training.txt_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
