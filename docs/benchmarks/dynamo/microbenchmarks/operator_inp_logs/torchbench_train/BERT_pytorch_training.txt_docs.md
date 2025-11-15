# Documentation: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/BERT_pytorch_training.txt`

## File Metadata

- **Path**: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/BERT_pytorch_training.txt`
- **Size**: 4,920 bytes (4.80 KB)
- **Type**: Source File (.txt)
- **Extension**: `.txt`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```
Operator: aten._softmax.default
cnt: 12, ((T([16, 12, 128, 128], f16), -1, False), {})
Operator: aten._softmax_backward_data.default
cnt: 12, ((T([16, 12, 128, 128], f16), T([16, 12, 128, 128], f16), -1, f16), {})
Operator: aten._unsafe_view.default
cnt: 36, ((T([16, 12, 128, 64], f16), [192, 128, 64]), {})
cnt: 12, ((T([16, 12, 64, 128], f16), [192, 64, 128]), {})
cnt: 12, ((T([192, 128, 128], f16), [16, 12, 128, 128]), {})
cnt: 12, ((T([192, 128, 64], f16), [16, 12, 128, 64]), {})
cnt: 24, ((T([16, 128, 12, 64], f16), [16, 128, 768]), {})
cnt: 12, ((T([16, 128, 768], f16), [2048, 768]), {})
Operator: aten.add.Tensor
cnt: 1, ((T([16, 128, 768], f16), T([1, 128, 768], f16)), {})
cnt: 120, ((T([16, 128, 768], f16), T([16, 128, 768], f16)), {})
cnt: 24, ((T([16, 128, 1], f16), 1e-06), {})
cnt: 24, ((T([16, 128, 768], f16), T([768], f16)), {})
cnt: 1, ((T([16, 128, 768], f16, stride=(0, 0, 0)), T([16, 128, 768], f16)), {})
Operator: aten.addmm.default
cnt: 48, ((T([768], f16), T([2048, 768], f16), T([768, 768], f16, stride=(1, 768))), {})
cnt: 12, ((T([3072], f16), T([2048, 768], f16), T([768, 3072], f16, stride=(1, 768))), {})
cnt: 12, ((T([768], f16), T([2048, 3072], f16), T([3072, 768], f16, stride=(1, 3072))), {})
Operator: aten.bmm.default
cnt: 12, ((T([192, 128, 64], f16), T([192, 64, 128], f16)), {})
cnt: 12, ((T([192, 128, 128], f16), T([192, 128, 64], f16)), {})
cnt: 12, ((T([192, 128, 128], f16, stride=(16384, 1, 128)), T([192, 128, 64], f16)), {})
cnt: 12, ((T([192, 128, 64], f16), T([192, 64, 128], f16, stride=(8192, 1, 64))), {})
cnt: 12, ((T([192, 64, 128], f16, stride=(8192, 1, 64)), T([192, 128, 128], f16)), {})
cnt: 12, ((T([192, 128, 128], f16), T([192, 128, 64], f16, stride=(8192, 1, 128))), {})
Operator: aten.clone.default
cnt: 2, ((T([16, 128], i64),), {})
Operator: aten.copy_.default
cnt: 2, ((T([16, 128], i64), T([16, 128], i64)), {})
Operator: aten.div.Scalar
cnt: 24, ((T([16, 128, 768], f16, stride=(128, 1, 0)), 768), {})
Operator: aten.div.Tensor
cnt: 96, ((T([16, 128, 768], f16), T([16, 128, 1], f16)), {})
cnt: 24, ((T([16, 12, 128, 128], f16), 8.0), {})
cnt: 2, ((T([], f16), 1572864), {})
cnt: 24, ((T([16, 128, 1], f16), T([16, 128, 1], f16)), {})
Operator: aten.embedding.default
cnt: 1, ((T([20005, 768], f16), T([16, 128], i64), 0), {})
cnt: 1, ((T([3, 768], f16), T([16, 128], i64), 0), {})
Operator: aten.embedding_dense_backward.default
cnt: 1, ((T([16, 128, 768], f16), T([16, 128], i64), 3, 0, False), {})
cnt: 1, ((T([16, 128, 768], f16), T([16, 128], i64), 20005, 0, False), {})
Operator: aten.eq.Scalar
cnt: 12, ((T([16, 1, 128, 128], b8), 0), {})
cnt: 24, ((T([16, 128, 1], f16), 0), {})
Operator: aten.gelu.default
cnt: 12, ((T([16, 128, 3072], f16),), {})
Operator: aten.gelu_backward.default
cnt: 12, ((T([16, 128, 3072], f16), T([16, 128, 3072], f16)), {})
Operator: aten.gt.Scalar
cnt: 1, ((T([16, 128], i64), 0), {})
Operator: aten.masked_fill.Scalar
cnt: 12, ((T([16, 12, 128, 128], f16), T([16, 1, 128, 128], b8), -65504.0), {})
cnt: 12, ((T([16, 12, 128, 128], f16), T([16, 1, 128, 128], b8), 0), {})
Operator: aten.masked_fill_.Scalar
cnt: 24, ((T([16, 128, 1], f16), T([16, 128, 1], b8), 0), {})
Operator: aten.mean.dim
cnt: 48, ((T([16, 128, 768], f16), [-1], True), {})
Operator: aten.mm.default
cnt: 1, ((T([2048, 768], f16, stride=(0, 0)), T([768, 3072], f16)), {})
cnt: 1, ((T([768, 2048], f16, stride=(0, 0)), T([2048, 3072], f16)), {})
cnt: 12, ((T([2048, 3072], f16), T([3072, 768], f16)), {})
cnt: 12, ((T([3072, 2048], f16, stride=(1, 3072)), T([2048, 768], f16)), {})
cnt: 48, ((T([2048, 768], f16), T([768, 768], f16)), {})
cnt: 48, ((T([768, 2048], f16, stride=(1, 768)), T([2048, 768], f16)), {})
cnt: 11, ((T([2048, 768], f16), T([768, 3072], f16)), {})
cnt: 11, ((T([768, 2048], f16, stride=(1, 768)), T([2048, 3072], f16)), {})
Operator: aten.mul.Scalar
cnt: 24, ((T([16, 128, 1], f16), 2), {})
cnt: 24, ((T([16, 128, 1], f16), 0.002607561929595828), {})
Operator: aten.mul.Tensor
cnt: 24, ((T([768], f16), T([16, 128, 768], f16)), {})
cnt: 48, ((T([16, 128, 768], f16), T([16, 128, 768], f16)), {})
cnt: 24, ((T([16, 128, 768], f16), T([768], f16)), {})
cnt: 24, ((T([16, 128, 1], f16), T([16, 128, 768], f16)), {})
Operator: aten.neg.default
cnt: 48, ((T([16, 128, 768], f16),), {})
Operator: aten.repeat.default
cnt: 1, ((T([16, 1, 128], b8), [1, 128, 1]), {})
Operator: aten.std.correction
cnt: 24, ((T([16, 128, 768], f16), [-1]), {'correction': 1, 'keepdim': True})
Operator: aten.sub.Tensor
cnt: 48, ((T([16, 128, 768], f16), T([16, 128, 1], f16)), {})
Operator: aten.sum.SymInt
cnt: 1, ((T([2048, 768], f16, stride=(0, 0)), [0], True), {})
cnt: 12, ((T([2048, 3072], f16), [0], True), {})
cnt: 48, ((T([16, 128, 768], f16), [0, 1], True), {})
cnt: 48, ((T([16, 128, 768], f16), [2], True), {})
cnt: 59, ((T([2048, 768], f16), [0], True), {})
Operator: aten.sum.default
cnt: 1, ((T([16, 128, 768], f16),), {})

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

- **File Documentation**: `BERT_pytorch_training.txt_docs.md`
- **Keyword Index**: `BERT_pytorch_training.txt_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
