# Documentation: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/timm_train/vit_base_patch16_224_training.txt`

## File Metadata

- **Path**: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/timm_train/vit_base_patch16_224_training.txt`
- **Size**: 5,129 bytes (5.01 KB)
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
Operator: aten._softmax.default
cnt: 12, ((T([64, 12, 197, 197], f16), -1, False), {})
Operator: aten._softmax_backward_data.default
cnt: 12, ((T([64, 12, 197, 197], f16), T([64, 12, 197, 197], f16), -1, f16), {})
Operator: aten._unsafe_view.default
cnt: 36, ((T([64, 12, 197, 64], f16), [768, 197, 64]), {})
cnt: 12, ((T([64, 12, 64, 197], f16), [768, 64, 197]), {})
cnt: 12, ((T([768, 197, 197], f16), [64, 12, 197, 197]), {})
cnt: 12, ((T([768, 197, 64], f16), [64, 12, 197, 64]), {})
cnt: 12, ((T([64, 197, 12, 64], f16), [64, 197, 768]), {})
cnt: 12, ((T([64, 197, 3, 12, 64], f16), [64, 197, 2304]), {})
Operator: aten.add.Tensor
cnt: 1, ((T([64, 197, 768], f16), T([1, 197, 768], f16)), {})
cnt: 48, ((T([64, 197, 768], f16), T([64, 197, 768], f16)), {})
Operator: aten.addmm.default
cnt: 12, ((T([2304], f16), T([12608, 768], f16), T([768, 2304], f16, stride=(1, 768))), {})
cnt: 12, ((T([768], f16), T([12608, 768], f16), T([768, 768], f16, stride=(1, 768))), {})
cnt: 12, ((T([3072], f16), T([12608, 768], f16), T([768, 3072], f16, stride=(1, 768))), {})
cnt: 12, ((T([768], f16), T([12608, 3072], f16), T([3072, 768], f16, stride=(1, 3072))), {})
cnt: 1, ((T([1000], f16), T([64, 768], f16, stride=(151296, 1)), T([768, 1000], f16, stride=(1, 768))), {})
Operator: aten.bmm.default
cnt: 12, ((T([768, 197, 64], f16), T([768, 64, 197], f16)), {})
cnt: 12, ((T([768, 197, 197], f16), T([768, 197, 64], f16)), {})
cnt: 12, ((T([768, 197, 197], f16, stride=(38809, 1, 197)), T([768, 197, 64], f16)), {})
cnt: 12, ((T([768, 197, 64], f16), T([768, 64, 197], f16, stride=(12608, 1, 64))), {})
cnt: 12, ((T([768, 64, 197], f16, stride=(12608, 1, 64)), T([768, 197, 197], f16)), {})
cnt: 12, ((T([768, 197, 197], f16), T([768, 197, 64], f16, stride=(12608, 1, 197))), {})
Operator: aten.cat.default
cnt: 1, (([T([64, 1, 768], f16, stride=(0, 768, 1)), T([64, 196, 768], f16, stride=(150528, 1, 196))], 1), {})
Operator: aten.clone.default
cnt: 1, ((T([64, 3, 224, 224], f16),), {})
Operator: aten.convolution.default
cnt: 1, ((T([64, 3, 224, 224], f16), T([768, 3, 16, 16], f16), T([768], f16), [16, 16], [0, 0], [1, 1], False, [0, 0], 1), {})
Operator: aten.convolution_backward.default
cnt: 1, ((T([64, 768, 14, 14], f16, stride=(151296, 1, 10752, 768)), T([64, 3, 224, 224], f16), T([768, 3, 16, 16], f16), [768], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]), {})
Operator: aten.copy_.default
cnt: 1, ((T([64, 3, 224, 224], f16), T([64, 3, 224, 224], f16)), {})
Operator: aten.gelu.default
cnt: 12, ((T([64, 197, 3072], f16),), {})
Operator: aten.gelu_backward.default
cnt: 12, ((T([64, 197, 3072], f16), T([64, 197, 3072], f16)), {})
Operator: aten.lift_fresh_copy.default
cnt: 1, ((T([64], i64),), {})
Operator: aten.mm.default
cnt: 1, ((T([64, 1000], f16), T([1000, 768], f16)), {})
cnt: 1, ((T([1000, 64], f16, stride=(1, 1000)), T([64, 768], f16, stride=(151296, 1))), {})
cnt: 12, ((T([12608, 768], f16), T([768, 3072], f16)), {})
cnt: 12, ((T([768, 12608], f16, stride=(1, 768)), T([12608, 3072], f16)), {})
cnt: 12, ((T([12608, 3072], f16), T([3072, 768], f16)), {})
cnt: 12, ((T([3072, 12608], f16, stride=(1, 3072)), T([12608, 768], f16)), {})
cnt: 12, ((T([12608, 768], f16), T([768, 768], f16)), {})
cnt: 12, ((T([768, 12608], f16, stride=(1, 768)), T([12608, 768], f16)), {})
cnt: 12, ((T([12608, 2304], f16), T([2304, 768], f16)), {})
cnt: 12, ((T([2304, 12608], f16, stride=(1, 2304)), T([12608, 768], f16)), {})
Operator: aten.mul.Tensor
cnt: 24, ((T([64, 12, 197, 197], f16), 0.125), {})
Operator: aten.native_layer_norm.default
cnt: 25, ((T([64, 197, 768], f16), [768], T([768], f16), T([768], f16), 1e-06), {})
Operator: aten.native_layer_norm_backward.default
cnt: 25, ((T([64, 197, 768], f16), T([64, 197, 768], f16), [768], T([64, 197, 1], f32), T([64, 197, 1], f32), T([768], f16), T([768], f16), [True, True, True]), {})
Operator: aten.nll_loss_backward.default
cnt: 1, ((T([], f16), T([64, 1000], f16), T([64], i64), None, 1, -100, T([], f16)), {})
Operator: aten.nll_loss_forward.default
cnt: 1, ((T([64, 1000], f16), T([64], i64), None, 1, -100), {})
Operator: aten.select_backward.default
cnt: 1, ((T([64, 768], f16), [64, 197, 768], 1, 0), {})
Operator: aten.slice_backward.default
cnt: 1, ((T([64, 197, 768], f16), [64, 197, 768], 0, 0, 9223372036854775807, 1), {})
Operator: aten.stack.default
cnt: 12, (([T([64, 12, 197, 64], f16), T([64, 12, 197, 64], f16, stride=(151296, 12608, 1, 197)), T([64, 12, 197, 64], f16)],), {})
Operator: aten.sum.SymInt
cnt: 1, ((T([64, 1000], f16), [0], True), {})
cnt: 24, ((T([12608, 768], f16), [0], True), {})
cnt: 12, ((T([12608, 3072], f16), [0], True), {})
cnt: 12, ((T([12608, 2304], f16), [0], True), {})
cnt: 1, ((T([64, 197, 768], f16), [0], True), {})
cnt: 1, ((T([64, 1, 768], f16, stride=(151296, 768, 1)), [0], True), {})
Operator: aten.unbind.int
cnt: 12, ((T([3, 64, 12, 197, 64], f16, stride=(768, 453888, 64, 2304, 1)),), {})

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

- **File Documentation**: `vit_base_patch16_224_training.txt_docs.md`
- **Keyword Index**: `vit_base_patch16_224_training.txt_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
