# Documentation: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/tts_angular_training.txt`

## File Metadata

- **Path**: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/tts_angular_training.txt`
- **Size**: 3,265 bytes (3.19 KB)
- **Type**: Source File (.txt)
- **Extension**: `.txt`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```
Operator: aten._cudnn_rnn.default
cnt: 1, ((T([64, 50, 40], f16), [T([3072, 40], f16), T([3072, 768], f16), T([3072], f16), T([3072], f16)], 4, None, T([1, 64, 768], f16), T([1, 64, 768], f16), 2, 768, 0, 1, True, 0.0, True, False, [], None), {})
cnt: 2, ((T([64, 50, 256], f16), [T([3072, 256], f16), T([3072, 768], f16), T([3072], f16), T([3072], f16)], 4, None, T([1, 64, 768], f16), T([1, 64, 768], f16), 2, 768, 0, 1, True, 0.0, True, False, [], None), {})
Operator: aten._cudnn_rnn_backward.default
cnt: 2, ((T([64, 50, 256], f16), [T([3072, 256], f16), T([3072, 768], f16), T([3072], f16), T([3072], f16)], 4, T([3151872], f16), T([1, 64, 768], f16), T([1, 64, 768], f16), T([64, 50, 768], f16, stride=(768, 49152, 1)), T([64, 50, 768], f16), None, None, 2, 768, 0, 1, True, 0.0, True, False, [], None, T([24576016], u8), [True, False, False, True]), {})
cnt: 1, ((T([64, 50, 40], f16), [T([3072, 40], f16), T([3072, 768], f16), T([3072], f16), T([3072], f16)], 4, T([2488320], f16), T([1, 64, 768], f16), T([1, 64, 768], f16), T([64, 50, 768], f16, stride=(768, 49152, 1)), T([64, 50, 768], f16), None, None, 2, 768, 0, 1, True, 0.0, True, False, [], None, T([24576016], u8), [False, False, False, True]), {})
Operator: aten._unsafe_view.default
cnt: 3, ((T([64, 50, 768], f16), [3200, 768]), {})
cnt: 3, ((T([3200, 256], f16), [64, 50, 256]), {})
cnt: 2, ((T([64, 50, 256], f16), [3200, 256]), {})
Operator: aten.add.Tensor
cnt: 1, ((T([64, 256], f16), T([64, 256], f16)), {})
Operator: aten.clamp_min.default
cnt: 1, ((T([64, 1], f16), 1e-12), {})
Operator: aten.clone.default
cnt: 1, ((T([64, 50, 40], f16),), {})
Operator: aten.copy_.default
cnt: 1, ((T([64, 50, 40], f16), T([64, 50, 40], f16)), {})
Operator: aten.div.Tensor
cnt: 2, ((T([64, 256], f16, stride=(12800, 1)), T([64, 256], f16, stride=(1, 0))), {})
cnt: 2, ((T([], f16), 16384), {})
cnt: 1, ((T([64, 256], f16), T([64, 256], f16, stride=(1, 0))), {})
cnt: 1, ((T([64, 256], f16, stride=(0, 0)), T([64, 256], f16, stride=(1, 0))), {})
cnt: 1, ((T([64, 256], f16, stride=(12800, 1)), T([64, 1], f16)), {})
Operator: aten.eq.Scalar
cnt: 1, ((T([64, 1], f16), 0), {})
Operator: aten.ge.Scalar
cnt: 1, ((T([64, 1], f16), 1e-12), {})
Operator: aten.masked_fill_.Scalar
cnt: 1, ((T([64, 256], f16), T([64, 1], b8), 0), {})
Operator: aten.mm.default
cnt: 3, ((T([3200, 768], f16), T([768, 256], f16, stride=(1, 768))), {})
cnt: 3, ((T([256, 3200], f16, stride=(1, 256)), T([3200, 768], f16)), {})
cnt: 3, ((T([3200, 256], f16), T([256, 768], f16)), {})
Operator: aten.mul.Tensor
cnt: 1, ((T([64, 256], f16), T([64, 256], f16)), {})
cnt: 1, ((T([64, 1], f16), T([64, 256], f16)), {})
Operator: aten.neg.default
cnt: 1, ((T([64, 256], f16, stride=(0, 0)),), {})
Operator: aten.norm.ScalarOpt_dim
cnt: 1, ((T([64, 256], f16, stride=(12800, 1)), 2, [1], True), {})
Operator: aten.select_backward.default
cnt: 1, ((T([64, 256], f16), [64, 50, 256], 1, -1), {})
Operator: aten.slice_backward.default
cnt: 1, ((T([64, 50, 256], f16), [64, 50, 256], 0, 0, 9223372036854775807, 1), {})
Operator: aten.sum.SymInt
cnt: 1, ((T([64, 256], f16), [1], True), {})
Operator: aten.sum.default
cnt: 1, ((T([64, 256], f16),), {})
Operator: aten.where.self
cnt: 1, ((T([64, 1], b8), T([64, 1], f16), T([], f16)), {})

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
- [`squeezenet1_1_training.txt_docs.md`](./squeezenet1_1_training.txt_docs.md)
- [`attention_is_all_you_need_pytorch_training.txt_docs.md`](./attention_is_all_you_need_pytorch_training.txt_docs.md)
- [`timm_regnet_training.txt_docs.md`](./timm_regnet_training.txt_docs.md)
- [`dcgan_training.txt_docs.md`](./dcgan_training.txt_docs.md)
- [`pytorch_struct_training.txt_docs.md`](./pytorch_struct_training.txt_docs.md)
- [`Background_Matting_training.txt_docs.md`](./Background_Matting_training.txt_docs.md)
- [`fambench_dlrm_training.txt_docs.md`](./fambench_dlrm_training.txt_docs.md)


## Cross-References

- **File Documentation**: `tts_angular_training.txt_docs.md`
- **Keyword Index**: `tts_angular_training.txt_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
