# Documentation: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/maml_omniglot_training.txt`

## File Metadata

- **Path**: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/maml_omniglot_training.txt`
- **Size**: 4,672 bytes (4.56 KB)
- **Type**: Source File (.txt)
- **Extension**: `.txt`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```
Operator: aten.addmm.default
cnt: 1, ((T([5], f16), T([5, 64], f16), T([64, 5], f16, stride=(1, 64))), {})
Operator: aten.clone.default
cnt: 1, ((T([5, 1, 28, 28], f16),), {})
Operator: aten.convolution.default
cnt: 1, ((T([5, 1, 28, 28], f16), T([64, 1, 3, 3], f16), T([64], f16), [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([5, 64, 13, 13], f16, stride=(10816, 1, 832, 64)), T([64, 64, 3, 3], f16), T([64], f16), [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([5, 64, 5, 5], f16, stride=(1600, 1, 320, 64)), T([64, 64, 3, 3], f16), T([64], f16), [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
Operator: aten.convolution_backward.default
cnt: 1, ((T([5, 64, 3, 3], f16, stride=(576, 1, 192, 64)), T([5, 64, 5, 5], f16, stride=(1600, 1, 320, 64)), T([64, 64, 3, 3], f16), [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 1, ((T([5, 64, 11, 11], f16, stride=(7744, 1, 704, 64)), T([5, 64, 13, 13], f16, stride=(10816, 1, 832, 64)), T([64, 64, 3, 3], f16), [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 1, ((T([5, 64, 26, 26], f16, stride=(43264, 1, 1664, 64)), T([5, 1, 28, 28], f16), T([64, 1, 3, 3], f16), [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]), {})
Operator: aten.copy_.default
cnt: 1, ((T([5, 1, 28, 28], f16), T([5, 1, 28, 28], f16)), {})
cnt: 2, ((T([64, 64, 3, 3], f16), T([64, 64, 3, 3], f16, stride=(576, 1, 192, 64))), {})
Operator: aten.div.Tensor
cnt: 2, ((T([], f16), 25), {})
Operator: aten.max_pool2d_with_indices.default
cnt: 1, ((T([5, 64, 26, 26], f16, stride=(43264, 1, 1664, 64)), [2, 2], [2, 2]), {})
cnt: 1, ((T([5, 64, 11, 11], f16, stride=(7744, 1, 704, 64)), [2, 2], [2, 2]), {})
cnt: 1, ((T([5, 64, 3, 3], f16, stride=(576, 1, 192, 64)), [2, 2], [2, 2]), {})
Operator: aten.max_pool2d_with_indices_backward.default
cnt: 1, ((T([5, 64, 1, 1], f16), T([5, 64, 3, 3], f16, stride=(576, 1, 192, 64)), [2, 2], [2, 2], [0, 0], [1, 1], False, T([5, 64, 1, 1], i64)), {})
cnt: 1, ((T([5, 64, 5, 5], f16, stride=(1600, 1, 320, 64)), T([5, 64, 11, 11], f16, stride=(7744, 1, 704, 64)), [2, 2], [2, 2], [0, 0], [1, 1], False, T([5, 64, 5, 5], i64, stride=(1600, 1, 320, 64))), {})
cnt: 1, ((T([5, 64, 13, 13], f16, stride=(10816, 1, 832, 64)), T([5, 64, 26, 26], f16, stride=(43264, 1, 1664, 64)), [2, 2], [2, 2], [0, 0], [1, 1], False, T([5, 64, 13, 13], i64, stride=(10816, 1, 832, 64))), {})
Operator: aten.mm.default
cnt: 2, ((T([5, 5], f16, stride=(0, 0)), T([5, 64], f16)), {})
Operator: aten.native_batch_norm.default
cnt: 1, ((T([5, 64, 26, 26], f16, stride=(43264, 1, 1664, 64)), T([64], f16), T([64], f16), T([64], f16), T([64], f16), False, 1.0, 1e-05), {})
cnt: 1, ((T([5, 64, 11, 11], f16, stride=(7744, 1, 704, 64)), T([64], f16), T([64], f16), T([64], f16), T([64], f16), False, 1.0, 1e-05), {})
cnt: 1, ((T([5, 64, 3, 3], f16, stride=(576, 1, 192, 64)), T([64], f16), T([64], f16), T([64], f16), T([64], f16), False, 1.0, 1e-05), {})
Operator: aten.native_batch_norm_backward.default
cnt: 1, ((T([5, 64, 3, 3], f16, stride=(576, 1, 192, 64)), T([5, 64, 3, 3], f16, stride=(576, 1, 192, 64)), T([64], f16), T([64], f16), T([64], f16), T([64], f32), T([64], f32), False, 1e-05, [True, True, True]), {})
cnt: 1, ((T([5, 64, 11, 11], f16, stride=(7744, 1, 704, 64)), T([5, 64, 11, 11], f16, stride=(7744, 1, 704, 64)), T([64], f16), T([64], f16), T([64], f16), T([64], f32), T([64], f32), False, 1e-05, [True, True, True]), {})
cnt: 1, ((T([5, 64, 26, 26], f16, stride=(43264, 1, 1664, 64)), T([5, 64, 26, 26], f16, stride=(43264, 1, 1664, 64)), T([64], f16), T([64], f16), T([64], f16), T([64], f32), T([64], f32), False, 1e-05, [True, True, True]), {})
Operator: aten.new_empty_strided.default
cnt: 2, ((T([64, 64, 3, 3], f16, stride=(576, 1, 192, 64)), [64, 64, 3, 3], [576, 9, 3, 1]), {'dtype': f16, 'layout': torch.strided, 'device': 'cuda'})
Operator: aten.relu_.default
cnt: 1, ((T([5, 64, 26, 26], f16, stride=(43264, 1, 1664, 64)),), {})
cnt: 1, ((T([5, 64, 11, 11], f16, stride=(7744, 1, 704, 64)),), {})
cnt: 1, ((T([5, 64, 3, 3], f16, stride=(576, 1, 192, 64)),), {})
Operator: aten.sum.SymInt
cnt: 1, ((T([5, 5], f16, stride=(0, 0)), [0], True), {})
Operator: aten.sum.default
cnt: 1, ((T([5, 5], f16),), {})
Operator: aten.threshold_backward.default
cnt: 1, ((T([5, 64, 3, 3], f16, stride=(576, 1, 192, 64)), T([5, 64, 3, 3], f16, stride=(576, 1, 192, 64)), 0), {})
cnt: 1, ((T([5, 64, 11, 11], f16, stride=(7744, 1, 704, 64)), T([5, 64, 11, 11], f16, stride=(7744, 1, 704, 64)), 0), {})
cnt: 1, ((T([5, 64, 26, 26], f16, stride=(43264, 1, 1664, 64)), T([5, 64, 26, 26], f16, stride=(43264, 1, 1664, 64)), 0), {})

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

- This file appears to involve **GPU/parallel computing** capabilities.

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

- **File Documentation**: `maml_omniglot_training.txt_docs.md`
- **Keyword Index**: `maml_omniglot_training.txt_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
