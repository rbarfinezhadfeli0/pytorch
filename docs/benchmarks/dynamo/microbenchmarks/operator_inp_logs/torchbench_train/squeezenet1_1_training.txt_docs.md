# Documentation: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/squeezenet1_1_training.txt`

## File Metadata

- **Path**: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/squeezenet1_1_training.txt`
- **Size**: 8,352 bytes (8.16 KB)
- **Type**: Source File (.txt)
- **Extension**: `.txt`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```
Operator: aten.add.Tensor
cnt: 2, ((T([32, 64, 13, 13], f16), T([32, 64, 13, 13], f16)), {})
cnt: 2, ((T([32, 48, 13, 13], f16), T([32, 48, 13, 13], f16)), {})
cnt: 2, ((T([32, 32, 27, 27], f16), T([32, 32, 27, 27], f16)), {})
cnt: 2, ((T([32, 16, 55, 55], f16), T([32, 16, 55, 55], f16)), {})
Operator: aten.cat.default
cnt: 2, (([T([32, 64, 55, 55], f16), T([32, 64, 55, 55], f16)], 1), {})
cnt: 2, (([T([32, 128, 27, 27], f16), T([32, 128, 27, 27], f16)], 1), {})
cnt: 2, (([T([32, 192, 13, 13], f16), T([32, 192, 13, 13], f16)], 1), {})
cnt: 2, (([T([32, 256, 13, 13], f16), T([32, 256, 13, 13], f16)], 1), {})
Operator: aten.clone.default
cnt: 1, ((T([32, 3, 224, 224], f16),), {})
Operator: aten.convolution.default
cnt: 1, ((T([32, 3, 224, 224], f16), T([64, 3, 3, 3], f16), T([64], f16), [2, 2], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([32, 64, 55, 55], f16), T([16, 64, 1, 1], f16), T([16], f16), [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 2, ((T([32, 16, 55, 55], f16), T([64, 16, 1, 1], f16), T([64], f16), [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 2, ((T([32, 16, 55, 55], f16), T([64, 16, 3, 3], f16), T([64], f16), [1, 1], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([32, 128, 55, 55], f16), T([16, 128, 1, 1], f16), T([16], f16), [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([32, 128, 27, 27], f16), T([32, 128, 1, 1], f16), T([32], f16), [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 2, ((T([32, 32, 27, 27], f16), T([128, 32, 1, 1], f16), T([128], f16), [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 2, ((T([32, 32, 27, 27], f16), T([128, 32, 3, 3], f16), T([128], f16), [1, 1], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([32, 256, 27, 27], f16), T([32, 256, 1, 1], f16), T([32], f16), [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([32, 256, 13, 13], f16), T([48, 256, 1, 1], f16), T([48], f16), [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 2, ((T([32, 48, 13, 13], f16), T([192, 48, 1, 1], f16), T([192], f16), [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 2, ((T([32, 48, 13, 13], f16), T([192, 48, 3, 3], f16), T([192], f16), [1, 1], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([32, 384, 13, 13], f16), T([48, 384, 1, 1], f16), T([48], f16), [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([32, 384, 13, 13], f16), T([64, 384, 1, 1], f16), T([64], f16), [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 2, ((T([32, 64, 13, 13], f16), T([256, 64, 1, 1], f16), T([256], f16), [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 2, ((T([32, 64, 13, 13], f16), T([256, 64, 3, 3], f16), T([256], f16), [1, 1], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([32, 512, 13, 13], f16), T([64, 512, 1, 1], f16), T([64], f16), [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([32, 512, 13, 13], f16), T([1000, 512, 1, 1], f16), T([1000], f16), [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
Operator: aten.convolution_backward.default
cnt: 1, ((T([32, 1000, 13, 13], f16), T([32, 512, 13, 13], f16), T([1000, 512, 1, 1], f16), [1000], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 2, ((T([32, 256, 13, 13], f16), T([32, 64, 13, 13], f16), T([256, 64, 3, 3], f16), [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 2, ((T([32, 256, 13, 13], f16), T([32, 64, 13, 13], f16), T([256, 64, 1, 1], f16), [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 1, ((T([32, 64, 13, 13], f16), T([32, 512, 13, 13], f16), T([64, 512, 1, 1], f16), [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 1, ((T([32, 64, 13, 13], f16), T([32, 384, 13, 13], f16), T([64, 384, 1, 1], f16), [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 2, ((T([32, 192, 13, 13], f16), T([32, 48, 13, 13], f16), T([192, 48, 3, 3], f16), [192], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 2, ((T([32, 192, 13, 13], f16), T([32, 48, 13, 13], f16), T([192, 48, 1, 1], f16), [192], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 1, ((T([32, 48, 13, 13], f16), T([32, 384, 13, 13], f16), T([48, 384, 1, 1], f16), [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 1, ((T([32, 48, 13, 13], f16), T([32, 256, 13, 13], f16), T([48, 256, 1, 1], f16), [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 2, ((T([32, 128, 27, 27], f16), T([32, 32, 27, 27], f16), T([128, 32, 3, 3], f16), [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 2, ((T([32, 128, 27, 27], f16), T([32, 32, 27, 27], f16), T([128, 32, 1, 1], f16), [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 1, ((T([32, 32, 27, 27], f16), T([32, 256, 27, 27], f16), T([32, 256, 1, 1], f16), [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 1, ((T([32, 32, 27, 27], f16), T([32, 128, 27, 27], f16), T([32, 128, 1, 1], f16), [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 2, ((T([32, 64, 55, 55], f16), T([32, 16, 55, 55], f16), T([64, 16, 3, 3], f16), [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 2, ((T([32, 64, 55, 55], f16), T([32, 16, 55, 55], f16), T([64, 16, 1, 1], f16), [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 1, ((T([32, 16, 55, 55], f16), T([32, 128, 55, 55], f16), T([16, 128, 1, 1], f16), [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 1, ((T([32, 16, 55, 55], f16), T([32, 64, 55, 55], f16), T([16, 64, 1, 1], f16), [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 1, ((T([32, 64, 111, 111], f16), T([32, 3, 224, 224], f16), T([64, 3, 3, 3], f16), [64], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]), {})
Operator: aten.copy_.default
cnt: 1, ((T([32, 3, 224, 224], f16), T([32, 3, 224, 224], f16)), {})
Operator: aten.div.Scalar
cnt: 1, ((T([32, 1000, 13, 13], f16, stride=(0, 0, 0, 0)), 169), {})
Operator: aten.div.Tensor
cnt: 2, ((T([], f16), 32000), {})
Operator: aten.max_pool2d_with_indices.default
cnt: 1, ((T([32, 64, 111, 111], f16), [3, 3], [2, 2], [0, 0], [1, 1], True), {})
cnt: 1, ((T([32, 128, 55, 55], f16), [3, 3], [2, 2], [0, 0], [1, 1], True), {})
cnt: 1, ((T([32, 256, 27, 27], f16), [3, 3], [2, 2], [0, 0], [1, 1], True), {})
Operator: aten.max_pool2d_with_indices_backward.default
cnt: 1, ((T([32, 256, 13, 13], f16), T([32, 256, 27, 27], f16), [3, 3], [2, 2], [0, 0], [1, 1], True, T([32, 256, 13, 13], i64)), {})
cnt: 1, ((T([32, 128, 27, 27], f16), T([32, 128, 55, 55], f16), [3, 3], [2, 2], [0, 0], [1, 1], True, T([32, 128, 27, 27], i64)), {})
cnt: 1, ((T([32, 64, 55, 55], f16), T([32, 64, 111, 111], f16), [3, 3], [2, 2], [0, 0], [1, 1], True, T([32, 64, 55, 55], i64)), {})
Operator: aten.mean.dim
cnt: 1, ((T([32, 1000, 13, 13], f16), [-1, -2], True), {})
Operator: aten.relu_.default
cnt: 1, ((T([32, 64, 111, 111], f16),), {})
cnt: 2, ((T([32, 16, 55, 55], f16),), {})
cnt: 4, ((T([32, 64, 55, 55], f16),), {})
cnt: 2, ((T([32, 32, 27, 27], f16),), {})
cnt: 4, ((T([32, 128, 27, 27], f16),), {})
cnt: 2, ((T([32, 48, 13, 13], f16),), {})
cnt: 4, ((T([32, 192, 13, 13], f16),), {})
cnt: 2, ((T([32, 64, 13, 13], f16),), {})
cnt: 4, ((T([32, 256, 13, 13], f16),), {})
cnt: 1, ((T([32, 1000, 13, 13], f16),), {})
Operator: aten.sum.default
cnt: 1, ((T([32, 1000], f16),), {})
Operator: aten.threshold_backward.default
cnt: 1, ((T([32, 1000, 13, 13], f16), T([32, 1000, 13, 13], f16), 0), {})
cnt: 4, ((T([32, 256, 13, 13], f16, stride=(86528, 169, 13, 1)), T([32, 256, 13, 13], f16), 0), {})
cnt: 2, ((T([32, 64, 13, 13], f16), T([32, 64, 13, 13], f16), 0), {})
cnt: 4, ((T([32, 192, 13, 13], f16, stride=(64896, 169, 13, 1)), T([32, 192, 13, 13], f16), 0), {})
cnt: 2, ((T([32, 48, 13, 13], f16), T([32, 48, 13, 13], f16), 0), {})
cnt: 4, ((T([32, 128, 27, 27], f16, stride=(186624, 729, 27, 1)), T([32, 128, 27, 27], f16), 0), {})
cnt: 2, ((T([32, 32, 27, 27], f16), T([32, 32, 27, 27], f16), 0), {})
cnt: 4, ((T([32, 64, 55, 55], f16, stride=(387200, 3025, 55, 1)), T([32, 64, 55, 55], f16), 0), {})
cnt: 2, ((T([32, 16, 55, 55], f16), T([32, 16, 55, 55], f16), 0), {})
cnt: 1, ((T([32, 64, 111, 111], f16), T([32, 64, 111, 111], f16), 0), {})

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
- [`attention_is_all_you_need_pytorch_training.txt_docs.md`](./attention_is_all_you_need_pytorch_training.txt_docs.md)
- [`timm_regnet_training.txt_docs.md`](./timm_regnet_training.txt_docs.md)
- [`dcgan_training.txt_docs.md`](./dcgan_training.txt_docs.md)
- [`pytorch_struct_training.txt_docs.md`](./pytorch_struct_training.txt_docs.md)
- [`Background_Matting_training.txt_docs.md`](./Background_Matting_training.txt_docs.md)
- [`fambench_dlrm_training.txt_docs.md`](./fambench_dlrm_training.txt_docs.md)


## Cross-References

- **File Documentation**: `squeezenet1_1_training.txt_docs.md`
- **Keyword Index**: `squeezenet1_1_training.txt_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
