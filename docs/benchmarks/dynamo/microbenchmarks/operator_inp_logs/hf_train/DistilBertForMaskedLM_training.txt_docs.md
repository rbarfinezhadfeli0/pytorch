# Documentation: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/hf_train/DistilBertForMaskedLM_training.txt`

## File Metadata

- **Path**: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/hf_train/DistilBertForMaskedLM_training.txt`
- **Size**: 4,504 bytes (4.40 KB)
- **Type**: Source File (.txt)
- **Extension**: `.txt`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```
Operator: aten._log_softmax.default
cnt: 1, ((T([2048, 30522], f16), 1, False), {})
Operator: aten._log_softmax_backward_data.default
cnt: 1, ((T([2048, 30522], f16), T([2048, 30522], f16), 1, f16), {})
Operator: aten._softmax.default
cnt: 6, ((T([16, 12, 128, 128], f16), -1, False), {})
Operator: aten._softmax_backward_data.default
cnt: 6, ((T([16, 12, 128, 128], f16), T([16, 12, 128, 128], f16), -1, f16), {})
Operator: aten._unsafe_view.default
cnt: 18, ((T([16, 12, 128, 64], f16), [192, 128, 64]), {})
cnt: 6, ((T([16, 12, 64, 128], f16), [192, 64, 128]), {})
cnt: 6, ((T([192, 128, 128], f16), [16, 12, 128, 128]), {})
cnt: 6, ((T([192, 128, 64], f16), [16, 12, 128, 64]), {})
cnt: 12, ((T([16, 128, 12, 64], f16), [16, 128, 768]), {})
cnt: 6, ((T([16, 128, 768], f16), [2048, 768]), {})
Operator: aten.add.Tensor
cnt: 1, ((T([16, 128, 768], f16), T([1, 128, 768], f16)), {})
cnt: 36, ((T([16, 128, 768], f16), T([16, 128, 768], f16)), {})
cnt: 1, ((T([30522, 768], f16), T([30522, 768], f16)), {})
Operator: aten.addmm.default
cnt: 25, ((T([768], f16), T([2048, 768], f16), T([768, 768], f16, stride=(1, 768))), {})
cnt: 6, ((T([3072], f16), T([2048, 768], f16), T([768, 3072], f16, stride=(1, 768))), {})
cnt: 6, ((T([768], f16), T([2048, 3072], f16), T([3072, 768], f16, stride=(1, 3072))), {})
cnt: 1, ((T([30522], f16), T([2048, 768], f16), T([768, 30522], f16, stride=(1, 768))), {})
Operator: aten.bmm.default
cnt: 6, ((T([192, 128, 64], f16), T([192, 64, 128], f16)), {})
cnt: 6, ((T([192, 128, 128], f16), T([192, 128, 64], f16)), {})
cnt: 6, ((T([192, 128, 128], f16, stride=(16384, 1, 128)), T([192, 128, 64], f16)), {})
cnt: 6, ((T([192, 128, 64], f16), T([192, 64, 128], f16, stride=(8192, 1, 64))), {})
cnt: 6, ((T([192, 64, 128], f16, stride=(8192, 1, 64)), T([192, 128, 128], f16)), {})
cnt: 6, ((T([192, 128, 128], f16), T([192, 128, 64], f16, stride=(8192, 1, 128))), {})
Operator: aten.clone.default
cnt: 2, ((T([16, 128], i64),), {})
Operator: aten.copy_.default
cnt: 2, ((T([16, 128], i64), T([16, 128], i64)), {})
Operator: aten.div.Tensor
cnt: 6, ((T([16, 12, 128, 64], f16, stride=(98304, 64, 768, 1)), 8.0), {})
cnt: 6, ((T([16, 12, 128, 64], f16), 8.0), {})
Operator: aten.embedding.default
cnt: 1, ((T([30522, 768], f16), T([16, 128], i64), 0), {})
cnt: 1, ((T([512, 768], f16), T([1, 128], i64)), {})
Operator: aten.embedding_dense_backward.default
cnt: 1, ((T([1, 128, 768], f16), T([1, 128], i64), 512, -1, False), {})
cnt: 1, ((T([16, 128, 768], f16), T([16, 128], i64), 30522, 0, False), {})
Operator: aten.eq.Scalar
cnt: 6, ((T([16, 128], f32), 0), {})
Operator: aten.gelu.default
cnt: 6, ((T([16, 128, 3072], f16),), {})
cnt: 1, ((T([16, 128, 768], f16),), {})
Operator: aten.gelu_backward.default
cnt: 1, ((T([16, 128, 768], f16), T([16, 128, 768], f16)), {})
cnt: 6, ((T([16, 128, 3072], f16), T([16, 128, 3072], f16)), {})
Operator: aten.masked_fill.Scalar
cnt: 6, ((T([16, 12, 128, 128], f16), T([16, 12, 128, 128], b8, stride=(128, 0, 0, 1)), 0), {})
Operator: aten.masked_fill.Tensor
cnt: 6, ((T([16, 12, 128, 128], f16), T([16, 12, 128, 128], b8, stride=(128, 0, 0, 1)), T([], f32)), {})
Operator: aten.mm.default
cnt: 1, ((T([2048, 30522], f16), T([30522, 768], f16)), {})
cnt: 1, ((T([30522, 2048], f16, stride=(1, 30522)), T([2048, 768], f16)), {})
cnt: 25, ((T([2048, 768], f16), T([768, 768], f16)), {})
cnt: 25, ((T([768, 2048], f16, stride=(1, 768)), T([2048, 768], f16)), {})
cnt: 6, ((T([2048, 768], f16), T([768, 3072], f16)), {})
cnt: 6, ((T([768, 2048], f16, stride=(1, 768)), T([2048, 3072], f16)), {})
cnt: 6, ((T([2048, 3072], f16), T([3072, 768], f16)), {})
cnt: 6, ((T([3072, 2048], f16, stride=(1, 3072)), T([2048, 768], f16)), {})
Operator: aten.native_layer_norm.default
cnt: 14, ((T([16, 128, 768], f16), [768], T([768], f16), T([768], f16), 1e-12), {})
Operator: aten.native_layer_norm_backward.default
cnt: 14, ((T([16, 128, 768], f16), T([16, 128, 768], f16), [768], T([16, 128, 1], f32), T([16, 128, 1], f32), T([768], f16), T([768], f16), [True, True, True]), {})
Operator: aten.nll_loss_backward.default
cnt: 1, ((T([], f16), T([2048, 30522], f16), T([2048], i64), None, 1, -100, T([], f16)), {})
Operator: aten.nll_loss_forward.default
cnt: 1, ((T([2048, 30522], f16), T([2048], i64), None, 1, -100), {})
Operator: aten.sum.SymInt
cnt: 1, ((T([2048, 30522], f16), [0], True), {})
cnt: 31, ((T([2048, 768], f16), [0], True), {})
cnt: 6, ((T([2048, 3072], f16), [0], True), {})
cnt: 1, ((T([16, 128, 768], f16), [0], True), {})

```



## High-Level Overview

This file is part of the PyTorch framework located at `benchmarks/dynamo/microbenchmarks/operator_inp_logs/hf_train`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/dynamo/microbenchmarks/operator_inp_logs/hf_train`, which is part of the PyTorch project infrastructure.



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

Files in the same folder (`benchmarks/dynamo/microbenchmarks/operator_inp_logs/hf_train`):

- [`DistillGPT2_training.txt_docs.md`](./DistillGPT2_training.txt_docs.md)
- [`CamemBert_training.txt_docs.md`](./CamemBert_training.txt_docs.md)
- [`PegasusForConditionalGeneration_training.txt_docs.md`](./PegasusForConditionalGeneration_training.txt_docs.md)
- [`PLBartForConditionalGeneration_training.txt_docs.md`](./PLBartForConditionalGeneration_training.txt_docs.md)
- [`MegatronBertForCausalLM_training.txt_docs.md`](./MegatronBertForCausalLM_training.txt_docs.md)
- [`PLBartForCausalLM_training.txt_docs.md`](./PLBartForCausalLM_training.txt_docs.md)
- [`OPTForCausalLM_training.txt_docs.md`](./OPTForCausalLM_training.txt_docs.md)
- [`Speech2Text2ForCausalLM_training.txt_docs.md`](./Speech2Text2ForCausalLM_training.txt_docs.md)
- [`GoogleFnet_training.txt_docs.md`](./GoogleFnet_training.txt_docs.md)
- [`DebertaForMaskedLM_training.txt_docs.md`](./DebertaForMaskedLM_training.txt_docs.md)


## Cross-References

- **File Documentation**: `DistilBertForMaskedLM_training.txt_docs.md`
- **Keyword Index**: `DistilBertForMaskedLM_training.txt_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
