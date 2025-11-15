# Documentation: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/hf_DistilBert_training.txt`

## File Metadata

- **Path**: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/hf_DistilBert_training.txt`
- **Size**: 4,141 bytes (4.04 KB)
- **Type**: Source File (.txt)
- **Extension**: `.txt`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```
Operator: aten._softmax.default
cnt: 6, ((T([8, 12, 512, 512], f16), -1, False), {})
Operator: aten._softmax_backward_data.default
cnt: 6, ((T([8, 12, 512, 512], f16), T([8, 12, 512, 512], f16), -1, f16), {})
Operator: aten._unsafe_view.default
cnt: 18, ((T([8, 12, 512, 64], f16), [96, 512, 64]), {})
cnt: 6, ((T([8, 12, 64, 512], f16), [96, 64, 512]), {})
cnt: 6, ((T([96, 512, 512], f16), [8, 12, 512, 512]), {})
cnt: 6, ((T([96, 512, 64], f16), [8, 12, 512, 64]), {})
cnt: 12, ((T([8, 512, 12, 64], f16), [8, 512, 768]), {})
cnt: 6, ((T([8, 512, 768], f16), [4096, 768]), {})
Operator: aten.add.Tensor
cnt: 1, ((T([8, 512, 768], f16), T([1, 512, 768], f16)), {})
cnt: 36, ((T([8, 512, 768], f16), T([8, 512, 768], f16)), {})
cnt: 1, ((T([30522, 768], f16), T([30522, 768], f16)), {})
Operator: aten.addmm.default
cnt: 25, ((T([768], f16), T([4096, 768], f16), T([768, 768], f16, stride=(1, 768))), {})
cnt: 6, ((T([3072], f16), T([4096, 768], f16), T([768, 3072], f16, stride=(1, 768))), {})
cnt: 6, ((T([768], f16), T([4096, 3072], f16), T([3072, 768], f16, stride=(1, 3072))), {})
cnt: 1, ((T([30522], f16), T([4096, 768], f16), T([768, 30522], f16, stride=(1, 768))), {})
Operator: aten.bmm.default
cnt: 6, ((T([96, 512, 64], f16), T([96, 64, 512], f16)), {})
cnt: 6, ((T([96, 512, 512], f16), T([96, 512, 64], f16)), {})
cnt: 6, ((T([96, 512, 512], f16, stride=(262144, 1, 512)), T([96, 512, 64], f16)), {})
cnt: 6, ((T([96, 512, 64], f16), T([96, 64, 512], f16, stride=(32768, 1, 64))), {})
cnt: 6, ((T([96, 64, 512], f16, stride=(32768, 1, 64)), T([96, 512, 512], f16)), {})
cnt: 6, ((T([96, 512, 512], f16), T([96, 512, 64], f16, stride=(32768, 1, 512))), {})
Operator: aten.clone.default
cnt: 1, ((T([8, 512], i64),), {})
Operator: aten.copy_.default
cnt: 1, ((T([8, 512], i64), T([8, 512], i64)), {})
Operator: aten.div.Tensor
cnt: 6, ((T([8, 12, 512, 64], f16, stride=(393216, 64, 768, 1)), 8.0), {})
cnt: 2, ((T([], f16), 125018112), {})
cnt: 6, ((T([8, 12, 512, 64], f16), 8.0), {})
Operator: aten.embedding.default
cnt: 1, ((T([30522, 768], f16), T([8, 512], i64), 0), {})
cnt: 1, ((T([512, 768], f16), T([1, 512], i64)), {})
Operator: aten.embedding_dense_backward.default
cnt: 1, ((T([1, 512, 768], f16), T([1, 512], i64), 512, -1, False), {})
cnt: 1, ((T([8, 512, 768], f16), T([8, 512], i64), 30522, 0, False), {})
Operator: aten.eq.Scalar
cnt: 6, ((T([8, 512], f32), 0), {})
Operator: aten.gelu.default
cnt: 6, ((T([8, 512, 3072], f16),), {})
cnt: 1, ((T([8, 512, 768], f16),), {})
Operator: aten.gelu_backward.default
cnt: 1, ((T([8, 512, 768], f16), T([8, 512, 768], f16)), {})
cnt: 6, ((T([8, 512, 3072], f16), T([8, 512, 3072], f16)), {})
Operator: aten.masked_fill.Scalar
cnt: 6, ((T([8, 12, 512, 512], f16), T([8, 12, 512, 512], b8, stride=(512, 0, 0, 1)), 0), {})
Operator: aten.masked_fill.Tensor
cnt: 6, ((T([8, 12, 512, 512], f16), T([8, 12, 512, 512], b8, stride=(512, 0, 0, 1)), T([], f32)), {})
Operator: aten.mm.default
cnt: 1, ((T([4096, 30522], f16, stride=(0, 0)), T([30522, 768], f16)), {})
cnt: 1, ((T([30522, 4096], f16, stride=(0, 0)), T([4096, 768], f16)), {})
cnt: 25, ((T([4096, 768], f16), T([768, 768], f16)), {})
cnt: 25, ((T([768, 4096], f16, stride=(1, 768)), T([4096, 768], f16)), {})
cnt: 6, ((T([4096, 768], f16), T([768, 3072], f16)), {})
cnt: 6, ((T([768, 4096], f16, stride=(1, 768)), T([4096, 3072], f16)), {})
cnt: 6, ((T([4096, 3072], f16), T([3072, 768], f16)), {})
cnt: 6, ((T([3072, 4096], f16, stride=(1, 3072)), T([4096, 768], f16)), {})
Operator: aten.native_layer_norm.default
cnt: 14, ((T([8, 512, 768], f16), [768], T([768], f16), T([768], f16), 1e-12), {})
Operator: aten.native_layer_norm_backward.default
cnt: 14, ((T([8, 512, 768], f16), T([8, 512, 768], f16), [768], T([8, 512, 1], f32), T([8, 512, 1], f32), T([768], f16), T([768], f16), [True, True, True]), {})
Operator: aten.sum.SymInt
cnt: 1, ((T([4096, 30522], f16, stride=(0, 0)), [0], True), {})
cnt: 31, ((T([4096, 768], f16), [0], True), {})
cnt: 6, ((T([4096, 3072], f16), [0], True), {})
cnt: 1, ((T([8, 512, 768], f16), [0], True), {})
Operator: aten.sum.default
cnt: 1, ((T([8, 512, 30522], f16),), {})

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

- **File Documentation**: `hf_DistilBert_training.txt_docs.md`
- **Keyword Index**: `hf_DistilBert_training.txt_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
