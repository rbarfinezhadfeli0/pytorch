# Documentation: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/hf_train/Speech2Text2ForCausalLM_training.txt`

## File Metadata

- **Path**: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/hf_train/Speech2Text2ForCausalLM_training.txt`
- **Size**: 4,467 bytes (4.36 KB)
- **Type**: Source File (.txt)
- **Extension**: `.txt`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```
Operator: aten._log_softmax.default
cnt: 1, ((T([8192, 10000], f16), 1, False), {})
Operator: aten._log_softmax_backward_data.default
cnt: 1, ((T([8192, 10000], f16), T([8192, 10000], f16), 1, f16), {})
Operator: aten._softmax.default
cnt: 6, ((T([256, 128, 128], f16), -1, False), {})
Operator: aten._softmax_backward_data.default
cnt: 6, ((T([256, 128, 128], f16), T([256, 128, 128], f16), -1, f16), {})
Operator: aten._to_copy.default
cnt: 1, ((T([128, 128], f32),), {'dtype': f16})
cnt: 1, ((T([64, 1, 128, 128], f16, stride=(0, 16384, 128, 1)),), {'dtype': f16, 'layout': torch.strided, 'device': 'cuda'})
cnt: 1, ((T([64, 128], b8),), {'dtype': i32})
cnt: 1, ((T([64, 128], i64),), {'dtype': i32, 'layout': torch.strided, 'device': 'cuda'})
cnt: 1, ((T([64, 128], i32),), {'dtype': i64})
Operator: aten._unsafe_view.default
cnt: 18, ((T([64, 128, 4, 64], f16), [64, 128, 256]), {})
cnt: 1, ((T([8192, 10000], f16), [64, 128, 10000]), {})
cnt: 6, ((T([64, 4, 128, 64], f16), [256, 128, 64]), {})
cnt: 6, ((T([64, 128, 256], f16), [8192, 256]), {})
Operator: aten.add.Tensor
cnt: 1, ((T([128], i64), 1), {})
cnt: 1, ((T([64, 128], i32), 0), {})
cnt: 1, ((T([64, 128], i64), 1), {})
cnt: 37, ((T([64, 128, 256], f16), T([64, 128, 256], f16)), {})
cnt: 6, ((T([64, 4, 128, 128], f16), T([64, 1, 128, 128], f16)), {})
cnt: 1, ((T([10000, 256], f16), T([10000, 256], f16)), {})
Operator: aten.addmm.default
cnt: 24, ((T([256], f16), T([8192, 256], f16), T([256, 256], f16, stride=(1, 256))), {})
cnt: 6, ((T([2048], f16), T([8192, 256], f16), T([256, 2048], f16, stride=(1, 256))), {})
cnt: 6, ((T([256], f16), T([8192, 2048], f16), T([2048, 256], f16, stride=(1, 2048))), {})
Operator: aten.bmm.default
cnt: 12, ((T([256, 128, 64], f16), T([256, 64, 128], f16, stride=(8192, 1, 64))), {})
cnt: 12, ((T([256, 128, 128], f16), T([256, 128, 64], f16)), {})
cnt: 6, ((T([256, 128, 128], f16, stride=(16384, 1, 128)), T([256, 128, 64], f16)), {})
cnt: 6, ((T([256, 64, 128], f16, stride=(8192, 1, 64)), T([256, 128, 128], f16)), {})
Operator: aten.clone.default
cnt: 2, ((T([64, 128], i64),), {})
Operator: aten.copy_.default
cnt: 2, ((T([64, 128], i64), T([64, 128], i64)), {})
Operator: aten.cumsum.default
cnt: 1, ((T([64, 128], i32), 1), {})
Operator: aten.embedding.default
cnt: 1, ((T([10000, 256], f16), T([64, 128], i64), 1), {})
Operator: aten.embedding_dense_backward.default
cnt: 1, ((T([64, 128, 256], f16), T([64, 128], i64), 10000, 1, False), {})
Operator: aten.index_select.default
cnt: 1, ((T([1026, 256], f16), 0, T([8192], i64)), {})
Operator: aten.lt.Tensor
cnt: 1, ((T([128], i64), T([128, 1], i64)), {})
Operator: aten.masked_fill_.Scalar
cnt: 1, ((T([128, 128], f32), T([128, 128], b8), 0), {})
Operator: aten.mm.default
cnt: 1, ((T([8192, 256], f16), T([256, 10000], f16, stride=(1, 256))), {})
cnt: 1, ((T([10000, 8192], f16, stride=(1, 10000)), T([8192, 256], f16)), {})
cnt: 1, ((T([8192, 10000], f16), T([10000, 256], f16)), {})
cnt: 6, ((T([8192, 256], f16), T([256, 2048], f16)), {})
cnt: 6, ((T([256, 8192], f16, stride=(1, 256)), T([8192, 2048], f16)), {})
cnt: 6, ((T([8192, 2048], f16), T([2048, 256], f16)), {})
cnt: 6, ((T([2048, 8192], f16, stride=(1, 2048)), T([8192, 256], f16)), {})
cnt: 24, ((T([8192, 256], f16), T([256, 256], f16)), {})
cnt: 24, ((T([256, 8192], f16, stride=(1, 256)), T([8192, 256], f16)), {})
Operator: aten.mul.Tensor
cnt: 2, ((T([64, 128, 256], f16), 16.0), {})
cnt: 1, ((T([64, 128], i32), T([64, 128], i32)), {})
cnt: 12, ((T([64, 128, 256], f16), 0.125), {})
Operator: aten.native_layer_norm.default
cnt: 12, ((T([64, 128, 256], f16), [256], T([256], f16), T([256], f16), 1e-05), {})
Operator: aten.native_layer_norm_backward.default
cnt: 12, ((T([64, 128, 256], f16), T([64, 128, 256], f16), [256], T([64, 128, 1], f32), T([64, 128, 1], f32), T([256], f16), T([256], f16), [True, True, True]), {})
Operator: aten.ne.Scalar
cnt: 1, ((T([64, 128], i64), 1), {})
Operator: aten.nll_loss_backward.default
cnt: 1, ((T([], f16), T([8192, 10000], f16), T([8192], i64), None, 1, -100, T([], f16)), {})
Operator: aten.nll_loss_forward.default
cnt: 1, ((T([8192, 10000], f16), T([8192], i64), None, 1, -100), {})
Operator: aten.relu.default
cnt: 6, ((T([64, 128, 2048], f16),), {})
Operator: aten.sum.SymInt
cnt: 30, ((T([8192, 256], f16), [0], True), {})
cnt: 6, ((T([8192, 2048], f16), [0], True), {})
Operator: aten.threshold_backward.default
cnt: 6, ((T([64, 128, 2048], f16), T([64, 128, 2048], f16), 0), {})

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
- [`GoogleFnet_training.txt_docs.md`](./GoogleFnet_training.txt_docs.md)
- [`DebertaForMaskedLM_training.txt_docs.md`](./DebertaForMaskedLM_training.txt_docs.md)


## Cross-References

- **File Documentation**: `Speech2Text2ForCausalLM_training.txt_docs.md`
- **Keyword Index**: `Speech2Text2ForCausalLM_training.txt_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
