# Documentation: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/timm_train/convmixer_768_32_training.txt`

## File Metadata

- **Path**: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/timm_train/convmixer_768_32_training.txt`
- **Size**: 2,866 bytes (2.80 KB)
- **Type**: Source File (.txt)
- **Extension**: `.txt`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```
Operator: aten._log_softmax.default
cnt: 1, ((T([32, 1000], f16), 1, False), {})
Operator: aten._log_softmax_backward_data.default
cnt: 1, ((T([32, 1000], f16), T([32, 1000], f16), 1, f16), {})
Operator: aten.add.Tensor
cnt: 64, ((T([32, 768, 32, 32], f16), T([32, 768, 32, 32], f16)), {})
Operator: aten.add_.Tensor
cnt: 65, ((T([], i64), 1), {})
Operator: aten.addmm.default
cnt: 1, ((T([1000], f16), T([32, 768], f16), T([768, 1000], f16, stride=(1, 768))), {})
Operator: aten.clone.default
cnt: 1, ((T([32, 3, 224, 224], f16),), {})
Operator: aten.convolution.default
cnt: 1, ((T([32, 3, 224, 224], f16), T([768, 3, 7, 7], f16), T([768], f16), [7, 7], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 32, ((T([32, 768, 32, 32], f16), T([768, 1, 7, 7], f16), T([768], f16), [1, 1], [3, 3], [1, 1], False, [0, 0], 768), {})
cnt: 32, ((T([32, 768, 32, 32], f16), T([768, 768, 1, 1], f16), T([768], f16), [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
Operator: aten.convolution_backward.default
cnt: 32, ((T([32, 768, 32, 32], f16), T([32, 768, 32, 32], f16), T([768, 768, 1, 1], f16), [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 32, ((T([32, 768, 32, 32], f16), T([32, 768, 32, 32], f16), T([768, 1, 7, 7], f16), [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True]), {})
cnt: 1, ((T([32, 768, 32, 32], f16), T([32, 3, 224, 224], f16), T([768, 3, 7, 7], f16), [768], [7, 7], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]), {})
Operator: aten.copy_.default
cnt: 1, ((T([32, 3, 224, 224], f16), T([32, 3, 224, 224], f16)), {})
Operator: aten.div.Scalar
cnt: 1, ((T([32, 768, 32, 32], f16, stride=(768, 1, 0, 0)), 1024), {})
Operator: aten.lift_fresh_copy.default
cnt: 1, ((T([32], i64),), {})
Operator: aten.mean.dim
cnt: 1, ((T([32, 768, 32, 32], f16), [-1, -2], True), {})
Operator: aten.mm.default
cnt: 1, ((T([32, 1000], f16), T([1000, 768], f16)), {})
cnt: 1, ((T([1000, 32], f16, stride=(1, 1000)), T([32, 768], f16)), {})
Operator: aten.native_batch_norm.default
cnt: 65, ((T([32, 768, 32, 32], f16), T([768], f16), T([768], f16), T([768], f16), T([768], f16), True, 0.1, 1e-05), {})
Operator: aten.native_batch_norm_backward.default
cnt: 65, ((T([32, 768, 32, 32], f16), T([32, 768, 32, 32], f16), T([768], f16), T([768], f16), T([768], f16), T([768], f32), T([768], f32), True, 1e-05, [True, True, True]), {})
Operator: aten.nll_loss_backward.default
cnt: 1, ((T([], f16), T([32, 1000], f16), T([32], i64), None, 1, -100, T([], f16)), {})
Operator: aten.nll_loss_forward.default
cnt: 1, ((T([32, 1000], f16), T([32], i64), None, 1, -100), {})
Operator: aten.relu.default
cnt: 65, ((T([32, 768, 32, 32], f16),), {})
Operator: aten.sum.SymInt
cnt: 1, ((T([32, 1000], f16), [0], True), {})
Operator: aten.threshold_backward.default
cnt: 65, ((T([32, 768, 32, 32], f16), T([32, 768, 32, 32], f16), 0), {})

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

- **File Documentation**: `convmixer_768_32_training.txt_docs.md`
- **Keyword Index**: `convmixer_768_32_training.txt_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
