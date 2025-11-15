# Documentation: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/fastNLP_Bert_training.txt`

## File Metadata

- **Path**: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/fastNLP_Bert_training.txt`
- **Size**: 8,281 bytes (8.09 KB)
- **Type**: Source File (.txt)
- **Extension**: `.txt`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```
Operator: aten._index_put_impl_.default
cnt: 1, ((T([6, 474, 768], f16), [T([6, 474], i64, stride=(1, 0)), T([6, 474], i64, stride=(475, 1))], T([6, 474, 768], f16), True, True), {})
Operator: aten._softmax.default
cnt: 12, ((T([6, 12, 476, 476], f16), -1, False), {})
Operator: aten._softmax_backward_data.default
cnt: 12, ((T([6, 12, 476, 476], f16), T([6, 12, 476, 476], f16), -1, f16), {})
Operator: aten._to_copy.default
cnt: 1, ((T([6, 474], i64),), {'dtype': i64, 'layout': torch.strided, 'device': "torch.device('cpu')"})
cnt: 1, ((T([6], i64),), {'dtype': i64, 'device': 'cuda'})
cnt: 1, ((T([6, 476], b8),), {'dtype': i64})
cnt: 1, ((T([6, 1, 1, 476], i64),), {'dtype': f16})
Operator: aten._unsafe_view.default
cnt: 36, ((T([6, 12, 476, 64], f16), [72, 476, 64]), {})
cnt: 12, ((T([6, 12, 64, 476], f16), [72, 64, 476]), {})
cnt: 12, ((T([72, 476, 476], f16), [6, 12, 476, 476]), {})
cnt: 12, ((T([72, 476, 64], f16), [6, 12, 476, 64]), {})
cnt: 24, ((T([6, 476, 12, 64], f16), [6, 476, 768]), {})
cnt: 12, ((T([6, 476, 768], f16), [2856, 768]), {})
Operator: aten.add.Tensor
cnt: 6, ((T([], i64), 1), {})
cnt: 6, ((T([], i64), 2), {})
cnt: 1, ((T([6], i64), 1), {})
cnt: 74, ((T([6, 476, 768], f16), T([6, 476, 768], f16)), {})
cnt: 12, ((T([6, 12, 476, 476], f16), T([6, 1, 1, 476], f16)), {})
cnt: 12, ((T([6, 476, 3072], f16), 1.0), {})
cnt: 1, ((T([], f16), 0), {})
cnt: 1, ((T([], f16), T([], f16)), {})
cnt: 1, ((T([6, 474, 2], f16), T([6, 474, 2], f16)), {})
cnt: 12, ((T([6, 476, 3072], f16), T([6, 476, 3072], f16)), {})
Operator: aten.addmm.default
cnt: 48, ((T([768], f16), T([2856, 768], f16), T([768, 768], f16, stride=(1, 768))), {})
cnt: 12, ((T([3072], f16), T([2856, 768], f16), T([768, 3072], f16, stride=(1, 768))), {})
cnt: 12, ((T([768], f16), T([2856, 3072], f16), T([3072, 768], f16, stride=(1, 3072))), {})
cnt: 1, ((T([768], f16), T([6, 768], f16, stride=(365568, 1)), T([768, 768], f16, stride=(1, 768))), {})
cnt: 1, ((T([2], f16), T([2844, 768], f16), T([768, 2], f16, stride=(1, 768))), {})
Operator: aten.bitwise_xor.Tensor
cnt: 1, ((T([6, 1], i64, stride=(476, 1)), T([6, 476], i64)), {})
Operator: aten.bmm.default
cnt: 12, ((T([72, 476, 64], f16), T([72, 64, 476], f16)), {})
cnt: 12, ((T([72, 476, 476], f16), T([72, 476, 64], f16)), {})
cnt: 12, ((T([72, 476, 476], f16, stride=(226576, 1, 476)), T([72, 476, 64], f16)), {})
cnt: 12, ((T([72, 476, 64], f16), T([72, 64, 476], f16, stride=(30464, 1, 64))), {})
cnt: 12, ((T([72, 64, 476], f16, stride=(30464, 1, 64)), T([72, 476, 476], f16)), {})
cnt: 12, ((T([72, 476, 476], f16), T([72, 476, 64], f16, stride=(30464, 1, 476))), {})
Operator: aten.cat.default
cnt: 1, (([T([6, 474, 768], f16)], -1), {})
Operator: aten.clone.default
cnt: 1, ((T([6, 474], i64),), {})
Operator: aten.copy_.default
cnt: 1, ((T([6, 474], i64), T([6, 474], i64)), {})
cnt: 6, ((T([474], i64), T([474], i64)), {})
cnt: 1, ((T([6, 474], i64, stride=(475, 1)), T([6, 474], i64)), {})
cnt: 1, ((T([6, 474, 768], f16), T([6, 474, 768], f16)), {})
cnt: 1, ((T([1, 6, 474, 768], f16), T([1, 6, 474, 768], f16)), {})
Operator: aten.cumsum.default
cnt: 1, ((T([6, 476], i64), -1), {})
cnt: 1, ((T([6, 474], i64), -1), {})
Operator: aten.div.Tensor
cnt: 24, ((T([6, 12, 476, 476], f16), 8.0), {})
cnt: 24, ((T([6, 476, 3072], f16), 1.4142135623730951), {})
cnt: 4, ((T([], f16), 2844), {})
cnt: 2, ((T([], f16), 2), {})
Operator: aten.embedding.default
cnt: 1, ((T([21128, 768], f16), T([6, 476], i64), 0), {})
cnt: 1, ((T([512, 768], f16), T([6, 476], i64, stride=(0, 1))), {})
cnt: 1, ((T([2, 768], f16), T([6, 476], i64)), {})
Operator: aten.embedding_dense_backward.default
cnt: 1, ((T([6, 476, 768], f16), T([6, 476], i64), 2, -1, False), {})
cnt: 1, ((T([6, 476, 768], f16), T([6, 476], i64, stride=(0, 1)), 512, -1, False), {})
cnt: 1, ((T([6, 476, 768], f16), T([6, 476], i64), 21128, 0, False), {})
Operator: aten.eq.Scalar
cnt: 1, ((T([6, 474], b8), False), {})
cnt: 1, ((T([6, 476], i64), 511), {})
cnt: 1, ((T([6, 474, 1], b8), False), {})
Operator: aten.erf.default
cnt: 12, ((T([6, 476, 3072], f16),), {})
Operator: aten.exp.default
cnt: 12, ((T([6, 476, 3072], f16),), {})
Operator: aten.fill_.Scalar
cnt: 6, ((T([476], i64), 1), {})
cnt: 1, ((T([6], i64, stride=(476,)), 2057), {})
Operator: aten.flip.default
cnt: 2, ((T([6, 476], i64), [-1]), {})
Operator: aten.fmod.Scalar
cnt: 1, ((T([6, 476], i64), 2), {})
Operator: aten.ge.Scalar
cnt: 1, ((T([6, 474], i64, stride=(475, 1)), 474), {})
Operator: aten.index.Tensor
cnt: 1, ((T([2869], i64), [T([6, 474], i64)]), {})
cnt: 1, ((T([6, 474, 768], f16, stride=(365568, 768, 1)), [T([6, 474], i64, stride=(1, 0)), T([6, 474], i64, stride=(475, 1))]), {})
Operator: aten.index_put_.default
cnt: 1, ((T([6, 476], i64), [T([6], i64), T([6], i64)], T([], i64)), {})
Operator: aten.masked_fill.Scalar
cnt: 1, ((T([6, 474], i64), T([6, 474], b8), 0), {})
cnt: 2, ((T([6, 474, 768], f16), T([6, 474, 1], b8), 0), {})
Operator: aten.masked_fill_.Scalar
cnt: 1, ((T([6, 474], i64, stride=(475, 1)), T([6, 474], b8), 0), {})
Operator: aten.max.default
cnt: 2, ((T([6], i64),), {})
Operator: aten.mm.default
cnt: 1, ((T([2844, 2], f16), T([2, 768], f16)), {})
cnt: 1, ((T([2, 2844], f16, stride=(1, 2)), T([2844, 768], f16)), {})
cnt: 12, ((T([2856, 768], f16), T([768, 3072], f16)), {})
cnt: 12, ((T([768, 2856], f16, stride=(1, 768)), T([2856, 3072], f16)), {})
cnt: 12, ((T([2856, 3072], f16), T([3072, 768], f16)), {})
cnt: 12, ((T([3072, 2856], f16, stride=(1, 3072)), T([2856, 768], f16)), {})
cnt: 48, ((T([2856, 768], f16), T([768, 768], f16)), {})
cnt: 48, ((T([768, 2856], f16, stride=(1, 768)), T([2856, 768], f16)), {})
Operator: aten.mul.Scalar
cnt: 12, ((T([6, 476, 3072], f16), 1.1283791670955126), {})
Operator: aten.mul.Tensor
cnt: 1, ((T([6, 1, 1, 476], f16), -10000.0), {})
cnt: 24, ((T([6, 476, 3072], f16), 0.5), {})
cnt: 48, ((T([6, 476, 3072], f16), T([6, 476, 3072], f16)), {})
Operator: aten.native_layer_norm.default
cnt: 25, ((T([6, 476, 768], f16), [768], T([768], f16), T([768], f16), 1e-12), {})
Operator: aten.native_layer_norm_backward.default
cnt: 25, ((T([6, 476, 768], f16), T([6, 476, 768], f16), [768], T([6, 476, 1], f32), T([6, 476, 1], f32), T([768], f16), T([768], f16), [True, True, True]), {})
Operator: aten.ne.Scalar
cnt: 1, ((T([6, 474], i64), 0), {})
Operator: aten.neg.default
cnt: 12, ((T([6, 476, 3072], f16),), {})
Operator: aten.new_empty_strided.default
cnt: 1, ((T([1, 6, 474, 768], f16), [1, 6, 474, 768], [2184192, 364032, 768, 1]), {})
Operator: aten.new_full.default
cnt: 1, ((T([6, 474], i64), [6, 476], 2457), {'dtype': i64, 'layout': torch.strided, 'device': 'cuda', 'pin_memory': False})
Operator: aten.new_zeros.default
cnt: 1, ((T([6, 476, 768], f16), [1, 6, 474, 768]), {'dtype': f16, 'layout': torch.strided, 'device': 'cuda', 'pin_memory': False})
cnt: 1, ((T([6, 474], i64), [6, 475]), {'dtype': i64, 'layout': torch.strided, 'device': 'cuda', 'pin_memory': False})
cnt: 1, ((T([6, 474, 768], f16), [6, 474, 768]), {'dtype': f16, 'layout': torch.strided, 'device': 'cuda'})
Operator: aten.pow.Tensor_Scalar
cnt: 12, ((T([6, 476, 3072], f16), 2), {})
Operator: aten.rsub.Scalar
cnt: 1, ((T([6, 1, 1, 476], f16), 1.0), {})
Operator: aten.select_backward.default
cnt: 1, ((T([6, 474], f16, stride=(0, 0)), [6, 474, 2], 2, 1), {})
cnt: 1, ((T([6, 474], f16, stride=(0, 0)), [6, 474, 2], 2, 0), {})
Operator: aten.slice_backward.default
cnt: 2, ((T([6, 474, 2], f16), [6, 474, 2], 1, 0, 9223372036854775807, 1), {})
cnt: 2, ((T([6, 474, 2], f16), [6, 474, 2], 0, 0, 9223372036854775807, 1), {})
cnt: 1, ((T([6, 474, 768], f16), [6, 476, 768], 1, 1, -1, 1), {})
cnt: 1, ((T([6, 476, 768], f16), [6, 476, 768], 0, 0, 9223372036854775807, 1), {})
Operator: aten.stack.default
cnt: 1, (([T([6, 474, 768], f16)],), {})
Operator: aten.sum.SymInt
cnt: 1, ((T([2844, 2], f16), [0], True), {})
cnt: 60, ((T([2856, 768], f16), [0], True), {})
cnt: 12, ((T([2856, 3072], f16), [0], True), {})
Operator: aten.sum.default
cnt: 2, ((T([6, 474], f16, stride=(948, 2)),), {})
Operator: aten.sum.dim_IntList
cnt: 1, ((T([6, 474], b8), [-1]), {})
cnt: 2, ((T([6, 474], i64), [-1]), {})
Operator: aten.tanh.default
cnt: 1, ((T([6, 768], f16),), {})
Operator: aten.unbind.int
cnt: 1, ((T([1, 6, 474, 768], f16),), {})

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

- **File Documentation**: `fastNLP_Bert_training.txt_docs.md`
- **Keyword Index**: `fastNLP_Bert_training.txt_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
