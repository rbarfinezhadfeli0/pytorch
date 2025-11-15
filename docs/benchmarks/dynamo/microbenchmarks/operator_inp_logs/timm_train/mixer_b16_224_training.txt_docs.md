# Documentation: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/timm_train/mixer_b16_224_training.txt`

## File Metadata

- **Path**: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/timm_train/mixer_b16_224_training.txt`
- **Size**: 4,505 bytes (4.40 KB)
- **Type**: Source File (.txt)
- **Extension**: `.txt`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```
Operator: aten._log_softmax.default
cnt: 1, ((T([64, 1000], f16), 1, False), {})
Operator: aten._log_softmax_backward_data.default
cnt: 1, ((T([64, 1000], f16), T([64, 1000], f16), 1, f16), {})
Operator: aten._unsafe_view.default
cnt: 12, ((T([64, 768, 384], f16), [64, 768, 384]), {})
cnt: 12, ((T([64, 768, 196], f16), [49152, 196]), {})
Operator: aten.add.Tensor
cnt: 12, ((T([64, 768, 384], f16), T([384], f16)), {})
cnt: 12, ((T([64, 196, 768], f16, stride=(150528, 1, 196)), T([64, 196, 768], f16, stride=(150528, 1, 196))), {})
cnt: 12, ((T([64, 196, 768], f16, stride=(150528, 1, 196)), T([64, 196, 768], f16)), {})
cnt: 12, ((T([64, 196, 768], f16), T([64, 196, 768], f16)), {})
cnt: 12, ((T([64, 196, 768], f16), T([64, 196, 768], f16, stride=(150528, 1, 196))), {})
Operator: aten.addmm.default
cnt: 12, ((T([196], f16), T([49152, 384], f16), T([384, 196], f16, stride=(1, 384))), {})
cnt: 12, ((T([3072], f16), T([12544, 768], f16), T([768, 3072], f16, stride=(1, 768))), {})
cnt: 12, ((T([768], f16), T([12544, 3072], f16), T([3072, 768], f16, stride=(1, 3072))), {})
cnt: 1, ((T([1000], f16), T([64, 768], f16), T([768, 1000], f16, stride=(1, 768))), {})
Operator: aten.bmm.default
cnt: 12, ((T([64, 768, 196], f16, stride=(150528, 1, 768)), T([64, 196, 384], f16, stride=(0, 1, 196))), {})
cnt: 12, ((T([64, 196, 768], f16), T([64, 768, 384], f16)), {})
cnt: 12, ((T([64, 768, 384], f16), T([64, 384, 196], f16, stride=(0, 196, 1))), {})
Operator: aten.clone.default
cnt: 1, ((T([64, 3, 224, 224], f16),), {})
Operator: aten.convolution.default
cnt: 1, ((T([64, 3, 224, 224], f16), T([768, 3, 16, 16], f16), T([768], f16), [16, 16], [0, 0], [1, 1], False, [0, 0], 1), {})
Operator: aten.convolution_backward.default
cnt: 1, ((T([64, 768, 14, 14], f16, stride=(150528, 1, 10752, 768)), T([64, 3, 224, 224], f16), T([768, 3, 16, 16], f16), [768], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]), {})
Operator: aten.copy_.default
cnt: 1, ((T([64, 3, 224, 224], f16), T([64, 3, 224, 224], f16)), {})
cnt: 12, ((T([384, 196], f16), T([384, 196], f16, stride=(1, 384))), {})
Operator: aten.div.Scalar
cnt: 1, ((T([64, 196, 768], f16, stride=(768, 0, 1)), 196), {})
Operator: aten.gelu.default
cnt: 12, ((T([64, 768, 384], f16),), {})
cnt: 12, ((T([64, 196, 3072], f16),), {})
Operator: aten.gelu_backward.default
cnt: 12, ((T([64, 196, 3072], f16), T([64, 196, 3072], f16)), {})
cnt: 12, ((T([64, 768, 384], f16), T([64, 768, 384], f16)), {})
Operator: aten.lift_fresh_copy.default
cnt: 1, ((T([64], i64),), {})
Operator: aten.mean.dim
cnt: 1, ((T([64, 196, 768], f16), [1]), {})
Operator: aten.mm.default
cnt: 1, ((T([64, 1000], f16), T([1000, 768], f16)), {})
cnt: 1, ((T([1000, 64], f16, stride=(1, 1000)), T([64, 768], f16)), {})
cnt: 12, ((T([12544, 768], f16), T([768, 3072], f16)), {})
cnt: 12, ((T([768, 12544], f16, stride=(1, 768)), T([12544, 3072], f16)), {})
cnt: 12, ((T([12544, 3072], f16), T([3072, 768], f16)), {})
cnt: 12, ((T([3072, 12544], f16, stride=(1, 3072)), T([12544, 768], f16)), {})
cnt: 12, ((T([49152, 196], f16), T([196, 384], f16)), {})
cnt: 12, ((T([196, 49152], f16, stride=(1, 196)), T([49152, 384], f16)), {})
Operator: aten.native_layer_norm.default
cnt: 25, ((T([64, 196, 768], f16, stride=(150528, 1, 196)), [768], T([768], f16), T([768], f16), 1e-06), {})
Operator: aten.native_layer_norm_backward.default
cnt: 13, ((T([64, 196, 768], f16), T([64, 196, 768], f16, stride=(150528, 1, 196)), [768], T([64, 196, 1], f32), T([64, 196, 1], f32), T([768], f16), T([768], f16), [True, True, True]), {})
cnt: 12, ((T([64, 196, 768], f16, stride=(150528, 1, 196)), T([64, 196, 768], f16, stride=(150528, 1, 196)), [768], T([64, 196, 1], f32), T([64, 196, 1], f32), T([768], f16), T([768], f16), [True, True, True]), {})
Operator: aten.new_empty_strided.default
cnt: 12, ((T([384, 196], f16, stride=(1, 384)), [384, 196], [196, 1]), {'dtype': f16, 'layout': torch.strided, 'device': 'cuda'})
Operator: aten.nll_loss_backward.default
cnt: 1, ((T([], f16), T([64, 1000], f16), T([64], i64), None, 1, -100, T([], f16)), {})
Operator: aten.nll_loss_forward.default
cnt: 1, ((T([64, 1000], f16), T([64], i64), None, 1, -100), {})
Operator: aten.sum.SymInt
cnt: 1, ((T([64, 1000], f16), [0], True), {})
cnt: 12, ((T([12544, 768], f16), [0], True), {})
cnt: 12, ((T([12544, 3072], f16), [0], True), {})
cnt: 12, ((T([49152, 196], f16), [0], True), {})
cnt: 12, ((T([64, 768, 384], f16), [0, 1], True), {})
cnt: 12, ((T([64, 196, 384], f16), [0], True), {})

```



## High-Level Overview

This file is part of the PyTorch framework located at `benchmarks/dynamo/microbenchmarks/operator_inp_logs/timm_train`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/dynamo/microbenchmarks/operator_inp_logs/timm_train`, which is part of the PyTorch project infrastructure.



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

Files in the same folder (`benchmarks/dynamo/microbenchmarks/operator_inp_logs/timm_train`):

- [`jx_nest_base_training.txt_docs.md`](./jx_nest_base_training.txt_docs.md)
- [`convnext_base_training.txt_docs.md`](./convnext_base_training.txt_docs.md)
- [`gluon_xception65_training.txt_docs.md`](./gluon_xception65_training.txt_docs.md)
- [`swin_base_patch4_window7_224_training.txt_docs.md`](./swin_base_patch4_window7_224_training.txt_docs.md)
- [`pit_b_224_training.txt_docs.md`](./pit_b_224_training.txt_docs.md)
- [`pnasnet5large_training.txt_docs.md`](./pnasnet5large_training.txt_docs.md)
- [`gmixer_24_224_training.txt_docs.md`](./gmixer_24_224_training.txt_docs.md)
- [`botnet26t_256_training.txt_docs.md`](./botnet26t_256_training.txt_docs.md)
- [`nfnet_l0_training.txt_docs.md`](./nfnet_l0_training.txt_docs.md)
- [`crossvit_9_240_training.txt_docs.md`](./crossvit_9_240_training.txt_docs.md)


## Cross-References

- **File Documentation**: `mixer_b16_224_training.txt_docs.md`
- **Keyword Index**: `mixer_b16_224_training.txt_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
