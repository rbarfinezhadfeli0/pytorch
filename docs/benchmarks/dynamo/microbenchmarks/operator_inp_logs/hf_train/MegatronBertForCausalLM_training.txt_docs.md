# Documentation: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/hf_train/MegatronBertForCausalLM_training.txt`

## File Metadata

- **Path**: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/hf_train/MegatronBertForCausalLM_training.txt`
- **Size**: 4,831 bytes (4.72 KB)
- **Type**: Source File (.txt)
- **Extension**: `.txt`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```
Operator: aten._log_softmax.default
cnt: 1, ((T([254, 29056], f16), 1, False), {})
Operator: aten._log_softmax_backward_data.default
cnt: 1, ((T([254, 29056], f16), T([254, 29056], f16), 1, f16), {})
Operator: aten._softmax.default
cnt: 24, ((T([2, 16, 128, 128], f16), -1, False), {})
Operator: aten._softmax_backward_data.default
cnt: 24, ((T([2, 16, 128, 128], f16), T([2, 16, 128, 128], f16), -1, f16), {})
Operator: aten._to_copy.default
cnt: 1, ((T([2, 1, 1, 128], f32),), {'dtype': f16})
Operator: aten._unsafe_view.default
cnt: 72, ((T([2, 16, 128, 64], f16), [32, 128, 64]), {})
cnt: 24, ((T([2, 16, 64, 128], f16), [32, 64, 128]), {})
cnt: 24, ((T([32, 128, 128], f16), [2, 16, 128, 128]), {})
cnt: 24, ((T([32, 128, 64], f16), [2, 16, 128, 64]), {})
cnt: 48, ((T([2, 128, 16, 64], f16), [2, 128, 1024]), {})
cnt: 24, ((T([2, 128, 1024], f16), [256, 1024]), {})
Operator: aten.add.Tensor
cnt: 145, ((T([2, 128, 1024], f16), T([2, 128, 1024], f16)), {})
cnt: 24, ((T([2, 16, 128, 128], f16), T([2, 1, 1, 128], f16)), {})
cnt: 1, ((T([29056, 1024], f16), T([29056, 1024], f16)), {})
Operator: aten.add_.Tensor
cnt: 1, ((T([2, 128, 1024], f16), T([1, 128, 1024], f16)), {})
Operator: aten.addmm.default
cnt: 97, ((T([1024], f16), T([256, 1024], f16), T([1024, 1024], f16, stride=(1, 1024))), {})
cnt: 24, ((T([4096], f16), T([256, 1024], f16), T([1024, 4096], f16, stride=(1, 1024))), {})
cnt: 24, ((T([1024], f16), T([256, 4096], f16), T([4096, 1024], f16, stride=(1, 4096))), {})
cnt: 1, ((T([29056], f16), T([256, 1024], f16), T([1024, 29056], f16, stride=(1, 1024))), {})
Operator: aten.bmm.default
cnt: 24, ((T([32, 128, 64], f16), T([32, 64, 128], f16)), {})
cnt: 24, ((T([32, 128, 128], f16), T([32, 128, 64], f16)), {})
cnt: 24, ((T([32, 128, 128], f16, stride=(16384, 1, 128)), T([32, 128, 64], f16)), {})
cnt: 24, ((T([32, 128, 64], f16), T([32, 64, 128], f16, stride=(8192, 1, 64))), {})
cnt: 24, ((T([32, 64, 128], f16, stride=(8192, 1, 64)), T([32, 128, 128], f16)), {})
cnt: 24, ((T([32, 128, 128], f16), T([32, 128, 64], f16, stride=(8192, 1, 128))), {})
Operator: aten.clone.default
cnt: 2, ((T([2, 128], i64),), {})
Operator: aten.copy_.default
cnt: 2, ((T([2, 128], i64), T([2, 128], i64)), {})
Operator: aten.div.Tensor
cnt: 48, ((T([2, 16, 128, 128], f16), 8.0), {})
Operator: aten.embedding.default
cnt: 1, ((T([29056, 1024], f16), T([2, 128], i64), 0), {})
cnt: 1, ((T([2, 1024], f16), T([2, 128], i64)), {})
cnt: 1, ((T([512, 1024], f16), T([1, 128], i64)), {})
Operator: aten.embedding_dense_backward.default
cnt: 1, ((T([1, 128, 1024], f16), T([1, 128], i64), 512, -1, False), {})
cnt: 1, ((T([2, 128, 1024], f16), T([2, 128], i64), 2, -1, False), {})
cnt: 1, ((T([2, 128, 1024], f16), T([2, 128], i64), 29056, 0, False), {})
Operator: aten.gelu.default
cnt: 24, ((T([2, 128, 4096], f16),), {})
cnt: 1, ((T([2, 128, 1024], f16),), {})
Operator: aten.gelu_backward.default
cnt: 1, ((T([2, 128, 1024], f16), T([2, 128, 1024], f16)), {})
cnt: 24, ((T([2, 128, 4096], f16), T([2, 128, 4096], f16)), {})
Operator: aten.mm.default
cnt: 1, ((T([256, 29056], f16), T([29056, 1024], f16)), {})
cnt: 1, ((T([29056, 256], f16, stride=(1, 29056)), T([256, 1024], f16)), {})
cnt: 97, ((T([256, 1024], f16), T([1024, 1024], f16)), {})
cnt: 97, ((T([1024, 256], f16, stride=(1, 1024)), T([256, 1024], f16)), {})
cnt: 24, ((T([256, 1024], f16), T([1024, 4096], f16)), {})
cnt: 24, ((T([1024, 256], f16, stride=(1, 1024)), T([256, 4096], f16)), {})
cnt: 24, ((T([256, 4096], f16), T([4096, 1024], f16)), {})
cnt: 24, ((T([4096, 256], f16, stride=(1, 4096)), T([256, 1024], f16)), {})
Operator: aten.mul.Tensor
cnt: 1, ((T([2, 1, 1, 128], f16), -65504.0), {})
Operator: aten.native_layer_norm.default
cnt: 50, ((T([2, 128, 1024], f16), [1024], T([1024], f16), T([1024], f16), 1e-12), {})
Operator: aten.native_layer_norm_backward.default
cnt: 50, ((T([2, 128, 1024], f16), T([2, 128, 1024], f16), [1024], T([2, 128, 1], f32), T([2, 128, 1], f32), T([1024], f16), T([1024], f16), [True, True, True]), {})
Operator: aten.nll_loss_backward.default
cnt: 1, ((T([], f16), T([254, 29056], f16), T([254], i64), None, 1, -100, T([], f16)), {})
Operator: aten.nll_loss_forward.default
cnt: 1, ((T([254, 29056], f16), T([254], i64), None, 1, -100), {})
Operator: aten.rsub.Scalar
cnt: 1, ((T([2, 1, 1, 128], f16), 1.0), {})
Operator: aten.slice_backward.default
cnt: 1, ((T([2, 127, 29056], f16), [2, 127, 29056], 2, 0, 9223372036854775807, 1), {})
cnt: 1, ((T([2, 127, 29056], f16), [2, 128, 29056], 1, 0, -1, 1), {})
cnt: 1, ((T([2, 128, 29056], f16), [2, 128, 29056], 0, 0, 9223372036854775807, 1), {})
Operator: aten.sum.SymInt
cnt: 1, ((T([256, 29056], f16), [0], True), {})
cnt: 121, ((T([256, 1024], f16), [0], True), {})
cnt: 24, ((T([256, 4096], f16), [0], True), {})
cnt: 1, ((T([2, 128, 1024], f16), [0], True), {})

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
- [`PLBartForCausalLM_training.txt_docs.md`](./PLBartForCausalLM_training.txt_docs.md)
- [`OPTForCausalLM_training.txt_docs.md`](./OPTForCausalLM_training.txt_docs.md)
- [`Speech2Text2ForCausalLM_training.txt_docs.md`](./Speech2Text2ForCausalLM_training.txt_docs.md)
- [`GoogleFnet_training.txt_docs.md`](./GoogleFnet_training.txt_docs.md)
- [`DebertaForMaskedLM_training.txt_docs.md`](./DebertaForMaskedLM_training.txt_docs.md)


## Cross-References

- **File Documentation**: `MegatronBertForCausalLM_training.txt_docs.md`
- **Keyword Index**: `MegatronBertForCausalLM_training.txt_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
