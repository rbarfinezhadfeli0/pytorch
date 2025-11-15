# Documentation: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/hf_train/GoogleFnet_training.txt`

## File Metadata

- **Path**: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/hf_train/GoogleFnet_training.txt`
- **Size**: 4,457 bytes (4.35 KB)
- **Type**: Source File (.txt)
- **Extension**: `.txt`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```
Operator: aten._fft_c2c.default
cnt: 12, ((T([1, 512, 768], c32), [1, 2], 0, True), {})
cnt: 12, ((T([1, 512, 768], c32), [1, 2], 0, False), {})
Operator: aten._log_softmax.default
cnt: 1, ((T([512, 32000], f16), 1, False), {})
Operator: aten._log_softmax_backward_data.default
cnt: 1, ((T([512, 32000], f16), T([512, 32000], f16), 1, f16), {})
Operator: aten._to_copy.default
cnt: 12, ((T([1, 512, 768], f16),), {'dtype': c32})
Operator: aten.add.Tensor
cnt: 28, ((T([1, 512, 768], f16), T([1, 512, 768], f16)), {})
cnt: 24, ((T([1, 512, 768], f16), T([1, 512, 768], f16, stride=(786432, 1536, 2))), {})
cnt: 36, ((T([1, 512, 3072], f16), T([1, 512, 3072], f16)), {})
cnt: 12, ((T([1, 512, 3072], f16), 1.0), {})
cnt: 1, ((T([1, 512, 768], f16), 1.0), {})
cnt: 1, ((T([32000, 768], f16), T([32000, 768], f16)), {})
Operator: aten.add_.Tensor
cnt: 1, ((T([1, 512, 768], f16), T([1, 512, 768], f16)), {})
Operator: aten.addmm.default
cnt: 2, ((T([768], f16), T([512, 768], f16), T([768, 768], f16, stride=(1, 768))), {})
cnt: 12, ((T([3072], f16), T([512, 768], f16), T([768, 3072], f16, stride=(1, 768))), {})
cnt: 12, ((T([768], f16), T([512, 3072], f16), T([3072, 768], f16, stride=(1, 3072))), {})
cnt: 1, ((T([768], f16), T([1, 768], f16), T([768, 768], f16, stride=(1, 768))), {})
cnt: 1, ((T([32000], f16), T([512, 768], f16), T([768, 32000], f16, stride=(1, 768))), {})
Operator: aten.clone.default
cnt: 2, ((T([1, 512], i64),), {})
Operator: aten.copy_.default
cnt: 2, ((T([1, 512], i64), T([1, 512], i64)), {})
Operator: aten.embedding.default
cnt: 1, ((T([32000, 768], f16), T([1, 512], i64), 3), {})
cnt: 1, ((T([4, 768], f16), T([1, 512], i64)), {})
cnt: 1, ((T([512, 768], f16), T([1, 512], i64)), {})
Operator: aten.embedding_dense_backward.default
cnt: 1, ((T([1, 512, 768], f16), T([1, 512], i64), 512, -1, False), {})
cnt: 1, ((T([1, 512, 768], f16), T([1, 512], i64), 4, -1, False), {})
cnt: 1, ((T([1, 512, 768], f16), T([1, 512], i64), 32000, 3, False), {})
Operator: aten.mm.default
cnt: 1, ((T([512, 32000], f16), T([32000, 768], f16)), {})
cnt: 1, ((T([32000, 512], f16, stride=(1, 32000)), T([512, 768], f16)), {})
cnt: 2, ((T([512, 768], f16), T([768, 768], f16)), {})
cnt: 2, ((T([768, 512], f16, stride=(1, 768)), T([512, 768], f16)), {})
cnt: 12, ((T([512, 768], f16), T([768, 3072], f16)), {})
cnt: 12, ((T([768, 512], f16, stride=(1, 768)), T([512, 3072], f16)), {})
cnt: 12, ((T([512, 3072], f16), T([3072, 768], f16)), {})
cnt: 12, ((T([3072, 512], f16, stride=(1, 3072)), T([512, 768], f16)), {})
Operator: aten.mul.Scalar
cnt: 1, ((T([1, 512, 768], f16), 3.0), {})
cnt: 12, ((T([1, 512, 3072], f16), 3.0), {})
Operator: aten.mul.Tensor
cnt: 24, ((T([1, 512, 3072], f16), 0.5), {})
cnt: 24, ((T([1, 512, 3072], f16), 0.044715), {})
cnt: 24, ((T([1, 512, 3072], f16), 0.7978845608028654), {})
cnt: 48, ((T([1, 512, 3072], f16), T([1, 512, 3072], f16)), {})
cnt: 2, ((T([1, 512, 768], f16), 0.5), {})
cnt: 2, ((T([1, 512, 768], f16), 0.044715), {})
cnt: 2, ((T([1, 512, 768], f16), 0.7978845608028654), {})
cnt: 4, ((T([1, 512, 768], f16), T([1, 512, 768], f16)), {})
Operator: aten.native_layer_norm.default
cnt: 26, ((T([1, 512, 768], f16), [768], T([768], f16), T([768], f16), 1e-12), {})
Operator: aten.native_layer_norm_backward.default
cnt: 26, ((T([1, 512, 768], f16), T([1, 512, 768], f16), [768], T([1, 512, 1], f32), T([1, 512, 1], f32), T([768], f16), T([768], f16), [True, True, True]), {})
Operator: aten.nll_loss_backward.default
cnt: 1, ((T([], f16), T([512, 32000], f16), T([512], i64), None, 1, -100, T([], f16)), {})
Operator: aten.nll_loss_forward.default
cnt: 1, ((T([512, 32000], f16), T([512], i64), None, 1, -100), {})
Operator: aten.pow.Tensor_Scalar
cnt: 12, ((T([1, 512, 3072], f16), 3.0), {})
cnt: 1, ((T([1, 512, 768], f16), 3.0), {})
cnt: 1, ((T([1, 512, 768], f16), 2.0), {})
cnt: 12, ((T([1, 512, 3072], f16), 2.0), {})
Operator: aten.select_backward.default
cnt: 12, ((T([1, 512, 768], f16), [1, 512, 768, 2], 3, 0), {})
Operator: aten.sum.SymInt
cnt: 1, ((T([512, 32000], f16), [0], True), {})
cnt: 14, ((T([512, 768], f16), [0], True), {})
cnt: 12, ((T([512, 3072], f16), [0], True), {})
Operator: aten.tanh.default
cnt: 12, ((T([1, 512, 3072], f16),), {})
cnt: 1, ((T([1, 768], f16),), {})
cnt: 1, ((T([1, 512, 768], f16),), {})
Operator: aten.tanh_backward.default
cnt: 1, ((T([1, 512, 768], f16), T([1, 512, 768], f16)), {})
cnt: 12, ((T([1, 512, 3072], f16), T([1, 512, 3072], f16)), {})

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
- [`DebertaForMaskedLM_training.txt_docs.md`](./DebertaForMaskedLM_training.txt_docs.md)


## Cross-References

- **File Documentation**: `GoogleFnet_training.txt_docs.md`
- **Keyword Index**: `GoogleFnet_training.txt_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
