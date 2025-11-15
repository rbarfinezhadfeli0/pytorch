# Documentation: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/hf_train/YituTechConvBert_training.txt`

## File Metadata

- **Path**: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/hf_train/YituTechConvBert_training.txt`
- **Size**: 7,324 bytes (7.15 KB)
- **Type**: Source File (.txt)
- **Extension**: `.txt`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```
Operator: aten._log_softmax.default
cnt: 1, ((T([512, 30522], f16), 1, False), {})
Operator: aten._log_softmax_backward_data.default
cnt: 1, ((T([512, 30522], f16), T([512, 30522], f16), 1, f16), {})
Operator: aten._softmax.default
cnt: 12, ((T([3072, 9, 1], f16), 1, False), {})
cnt: 12, ((T([1, 6, 512, 512], f16), -1, False), {})
Operator: aten._softmax_backward_data.default
cnt: 12, ((T([1, 6, 512, 512], f16), T([1, 6, 512, 512], f16), -1, f16), {})
cnt: 12, ((T([3072, 9, 1], f16), T([3072, 9, 1], f16), 1, f16), {})
Operator: aten._to_copy.default
cnt: 1, ((T([1, 1, 1, 512], f32),), {'dtype': f16})
Operator: aten._unsafe_view.default
cnt: 12, ((T([1, 512, 54], f16), [1, 512, 54]), {})
cnt: 12, ((T([1, 512, 384, 9], f16), [3072, 64, 9]), {})
cnt: 12, ((T([3072, 64, 1], f16), [3072, 64, 1]), {})
cnt: 12, ((T([6, 512, 512], f16), [1, 6, 512, 512]), {})
cnt: 12, ((T([6, 512, 64], f16), [1, 6, 512, 64]), {})
cnt: 12, ((T([512, 384], f16), [3072, 64, 1]), {})
cnt: 24, ((T([1, 512, 6, 64], f16), [1, 512, 384]), {})
Operator: aten.add.Tensor
cnt: 86, ((T([1, 512, 768], f16), T([1, 512, 768], f16)), {})
cnt: 12, ((T([1, 512, 54], f16), T([54], f16)), {})
cnt: 12, ((T([1, 6, 512, 512], f16), T([1, 1, 1, 512], f16)), {})
cnt: 12, ((T([1, 512, 384], f16), T([1, 512, 384], f16)), {})
cnt: 12, ((T([1, 512, 768], f16), T([1, 512, 768], f16, stride=(393216, 1, 512))), {})
cnt: 1, ((T([30522, 768], f16), T([30522, 768], f16)), {})
Operator: aten.add_.Tensor
cnt: 12, ((T([1, 384, 512], f16), T([384, 1], f16)), {})
Operator: aten.addmm.default
cnt: 48, ((T([384], f16), T([512, 768], f16), T([768, 384], f16, stride=(1, 768))), {})
cnt: 13, ((T([768], f16), T([512, 768], f16), T([768, 768], f16, stride=(1, 768))), {})
cnt: 12, ((T([3072], f16), T([512, 768], f16), T([768, 3072], f16, stride=(1, 768))), {})
cnt: 12, ((T([768], f16), T([512, 3072], f16), T([3072, 768], f16, stride=(1, 3072))), {})
cnt: 1, ((T([30522], f16), T([512, 768], f16), T([768, 30522], f16, stride=(1, 768))), {})
Operator: aten.bmm.default
cnt: 12, ((T([1, 512, 384], f16, stride=(512, 1, 512)), T([1, 384, 54], f16, stride=(384, 1, 384))), {})
cnt: 12, ((T([3072, 64, 9], f16), T([3072, 9, 1], f16)), {})
cnt: 12, ((T([6, 512, 64], f16, stride=(64, 384, 1)), T([6, 64, 512], f16, stride=(64, 1, 384))), {})
cnt: 24, ((T([6, 512, 512], f16), T([6, 512, 64], f16, stride=(64, 384, 1))), {})
cnt: 12, ((T([6, 512, 512], f16, stride=(262144, 1, 512)), T([6, 512, 64], f16, stride=(64, 768, 1))), {})
cnt: 12, ((T([6, 512, 64], f16, stride=(64, 768, 1)), T([6, 64, 512], f16, stride=(64, 1, 384))), {})
cnt: 12, ((T([6, 64, 512], f16, stride=(64, 1, 384)), T([6, 512, 512], f16)), {})
cnt: 12, ((T([3072, 9, 64], f16, stride=(576, 1, 9)), T([3072, 64, 1], f16)), {})
cnt: 12, ((T([3072, 64, 1], f16), T([3072, 1, 9], f16)), {})
cnt: 12, ((T([1, 384, 512], f16), T([1, 512, 54], f16)), {})
cnt: 12, ((T([1, 512, 54], f16), T([1, 54, 384], f16)), {})
Operator: aten.cat.default
cnt: 12, (([T([1, 512, 6, 64], f16), T([1, 512, 6, 64], f16)], 2), {})
Operator: aten.clone.default
cnt: 2, ((T([1, 512], i64),), {})
Operator: aten.convolution.default
cnt: 12, ((T([1, 768, 512], f16, stride=(393216, 1, 768)), T([768, 1, 9], f16), None, [1], [4], [1], False, [0], 768), {})
cnt: 12, ((T([1, 768, 512], f16), T([384, 768, 1], f16), None, [1], [0], [1], False, [0], 1), {})
Operator: aten.convolution_backward.default
cnt: 12, ((T([1, 384, 512], f16, stride=(196608, 1, 384)), T([1, 768, 512], f16), T([384, 768, 1], f16), [0], [1], [0], [1], False, [0], 1, [True, True, False]), {})
cnt: 12, ((T([1, 768, 512], f16), T([1, 768, 512], f16, stride=(393216, 1, 768)), T([768, 1, 9], f16), [0], [1], [4], [1], False, [0], 768, [True, True, False]), {})
Operator: aten.copy_.default
cnt: 2, ((T([1, 512], i64), T([1, 512], i64)), {})
cnt: 12, ((T([54, 384], f16), T([54, 384], f16, stride=(1, 54))), {})
Operator: aten.div.Tensor
cnt: 24, ((T([1, 6, 512, 512], f16), 8.0), {})
Operator: aten.embedding.default
cnt: 1, ((T([30522, 768], f16), T([1, 512], i64), 0), {})
cnt: 1, ((T([512, 768], f16), T([1, 512], i64)), {})
cnt: 1, ((T([2, 768], f16), T([1, 512], i64)), {})
Operator: aten.embedding_dense_backward.default
cnt: 1, ((T([1, 512, 768], f16), T([1, 512], i64), 2, -1, False), {})
cnt: 1, ((T([1, 512, 768], f16), T([1, 512], i64), 512, -1, False), {})
cnt: 1, ((T([1, 512, 768], f16), T([1, 512], i64), 30522, 0, False), {})
Operator: aten.gelu.default
cnt: 12, ((T([1, 512, 3072], f16),), {})
cnt: 1, ((T([1, 512, 768], f16),), {})
Operator: aten.gelu_backward.default
cnt: 1, ((T([1, 512, 768], f16), T([1, 512, 768], f16)), {})
cnt: 12, ((T([1, 512, 3072], f16), T([1, 512, 3072], f16)), {})
Operator: aten.im2col.default
cnt: 12, ((T([1, 384, 512, 1], f16), [9, 1], [1, 1], [4, 0], [1, 1]), {})
Operator: aten.im2col_backward.default
cnt: 12, ((T([1, 3456, 512], f16, stride=(1769472, 1, 3456)), [512, 1], [9, 1], [1, 1], [4, 0], [1, 1]), {})
Operator: aten.mm.default
cnt: 1, ((T([512, 30522], f16), T([30522, 768], f16)), {})
cnt: 1, ((T([30522, 512], f16, stride=(1, 30522)), T([512, 768], f16)), {})
cnt: 13, ((T([512, 768], f16), T([768, 768], f16)), {})
cnt: 13, ((T([768, 512], f16, stride=(1, 768)), T([512, 768], f16)), {})
cnt: 12, ((T([512, 768], f16), T([768, 3072], f16)), {})
cnt: 12, ((T([768, 512], f16, stride=(1, 768)), T([512, 3072], f16)), {})
cnt: 12, ((T([512, 3072], f16), T([3072, 768], f16)), {})
cnt: 12, ((T([3072, 512], f16, stride=(1, 3072)), T([512, 768], f16)), {})
cnt: 24, ((T([512, 384], f16, stride=(1, 512)), T([384, 768], f16)), {})
cnt: 24, ((T([384, 512], f16), T([512, 768], f16)), {})
cnt: 24, ((T([512, 384], f16), T([384, 768], f16)), {})
cnt: 24, ((T([384, 512], f16, stride=(1, 384)), T([512, 768], f16)), {})
Operator: aten.mul.Tensor
cnt: 1, ((T([1, 1, 1, 512], f16), -65504.0), {})
cnt: 12, ((T([1, 512, 384], f16, stride=(196608, 1, 512)), T([1, 512, 384], f16)), {})
cnt: 12, ((T([1, 512, 384], f16), T([1, 512, 384], f16, stride=(196608, 1, 512))), {})
cnt: 12, ((T([1, 512, 384], f16), T([1, 512, 384], f16)), {})
Operator: aten.native_layer_norm.default
cnt: 26, ((T([1, 512, 768], f16), [768], T([768], f16), T([768], f16), 1e-12), {})
Operator: aten.native_layer_norm_backward.default
cnt: 26, ((T([1, 512, 768], f16), T([1, 512, 768], f16), [768], T([1, 512, 1], f32), T([1, 512, 1], f32), T([768], f16), T([768], f16), [True, True, True]), {})
Operator: aten.new_empty_strided.default
cnt: 12, ((T([54, 384], f16, stride=(1, 54)), [54, 384], [384, 1]), {'dtype': f16, 'layout': torch.strided, 'device': 'cuda'})
Operator: aten.nll_loss_backward.default
cnt: 1, ((T([], f16), T([512, 30522], f16), T([512], i64), None, 1, -100, T([], f16)), {})
Operator: aten.nll_loss_forward.default
cnt: 1, ((T([512, 30522], f16), T([512], i64), None, 1, -100), {})
Operator: aten.rsub.Scalar
cnt: 1, ((T([1, 1, 1, 512], f16), 1.0), {})
Operator: aten.sum.SymInt
cnt: 1, ((T([512, 30522], f16), [0], True), {})
cnt: 25, ((T([512, 768], f16), [0], True), {})
cnt: 12, ((T([512, 3072], f16), [0], True), {})
cnt: 24, ((T([512, 384], f16, stride=(1, 512)), [0], True), {})
cnt: 12, ((T([1, 512, 54], f16), [0, 1], True), {})
cnt: 12, ((T([1, 384, 54], f16), [0], True), {})
cnt: 12, ((T([1, 384, 512], f16, stride=(196608, 1, 384)), [0, 2], True), {})
cnt: 24, ((T([512, 384], f16), [0], True), {})

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

- **File Documentation**: `YituTechConvBert_training.txt_docs.md`
- **Keyword Index**: `YituTechConvBert_training.txt_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
