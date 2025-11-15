# Documentation: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/pytorch_struct_training.txt`

## File Metadata

- **Path**: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/pytorch_struct_training.txt`
- **Size**: 3,538 bytes (3.46 KB)
- **Type**: Source File (.txt)
- **Extension**: `.txt`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```
Operator: aten._log_softmax.default
cnt: 1, ((T([30, 4771], f16, stride=(1, 30)), -1, False), {})
cnt: 1, ((T([30, 3600], f16), -1, False), {})
cnt: 1, ((T([30], f16), -1, False), {})
Operator: aten._log_softmax_backward_data.default
cnt: 1, ((T([30], f16), T([30], f16), -1, f16), {})
cnt: 1, ((T([30, 3600], f16), T([30, 3600], f16), -1, f16), {})
cnt: 1, ((T([30, 4771], f16), T([30, 4771], f16), -1, f16), {})
Operator: aten.add.Tensor
cnt: 4, ((T([30, 256], f16), T([30, 256], f16)), {})
cnt: 1, ((T([], f16), 0), {})
cnt: 2, ((T([], f16), T([], f16)), {})
cnt: 4, ((T([30, 256], f16, stride=(1, 30)), T([30, 256], f16)), {})
Operator: aten.addmm.default
cnt: 10, ((T([256], f16), T([30, 256], f16), T([256, 256], f16, stride=(1, 256))), {})
Operator: aten.bmm.default
cnt: 1, ((T([1, 4771, 256], f16), T([1, 256, 30], f16, stride=(256, 1, 256))), {})
cnt: 1, ((T([1, 30, 256], f16), T([1, 256, 3600], f16, stride=(256, 1, 256))), {})
cnt: 1, ((T([1, 1, 256], f16), T([1, 256, 30], f16, stride=(256, 1, 256))), {})
cnt: 1, ((T([1, 256, 1], f16), T([1, 1, 30], f16)), {})
cnt: 1, ((T([1, 1, 30], f16), T([1, 30, 256], f16)), {})
cnt: 1, ((T([1, 256, 30], f16, stride=(7680, 1, 256)), T([1, 30, 3600], f16)), {})
cnt: 1, ((T([1, 30, 3600], f16), T([1, 3600, 256], f16)), {})
cnt: 1, ((T([1, 256, 4771], f16, stride=(1221376, 1, 256)), T([1, 4771, 30], f16, stride=(4771, 1, 4771))), {})
cnt: 1, ((T([1, 4771, 30], f16, stride=(4771, 1, 4771)), T([1, 30, 256], f16)), {})
Operator: aten.clone.default
cnt: 1, ((T([40, 29], i64, stride=(1, 40)),), {})
Operator: aten.copy_.default
cnt: 1, ((T([40, 29], i64, stride=(1, 40)), T([40, 29], i64, stride=(1, 40))), {})
cnt: 1, ((T([60, 60, 256], f16), T([60, 60, 256], f16, stride=(60, 1, 3600))), {})
Operator: aten.div.Tensor
cnt: 2, ((T([], f16), 34800), {})
cnt: 2, ((T([], f16), 4320000), {})
cnt: 2, ((T([], f16), 1200), {})
cnt: 2, ((T([], f16), 3), {})
Operator: aten.gather.default
cnt: 1, ((T([40, 29, 30, 4771], f16, stride=(0, 0, 4771, 1)), 3, T([40, 29, 30, 1], i64, stride=(1, 40, 0, 1))), {})
Operator: aten.mm.default
cnt: 8, ((T([30, 256], f16), T([256, 256], f16)), {})
cnt: 8, ((T([256, 30], f16, stride=(1, 256)), T([30, 256], f16)), {})
cnt: 2, ((T([30, 256], f16, stride=(1, 30)), T([256, 256], f16)), {})
cnt: 2, ((T([256, 30], f16), T([30, 256], f16)), {})
Operator: aten.new_empty_strided.default
cnt: 1, ((T([60, 60, 256], f16, stride=(60, 1, 3600)), [60, 60, 256], [15360, 256, 1]), {'dtype': f16, 'layout': torch.strided, 'device': 'cuda'})
Operator: aten.new_zeros.default
cnt: 1, ((T([40, 29, 30, 1], f16, stride=(0, 0, 0, 1)), [40, 29, 30, 4771]), {})
Operator: aten.relu.default
cnt: 8, ((T([30, 256], f16),), {})
Operator: aten.scatter_add.default
cnt: 1, ((T([40, 29, 30, 4771], f16), 3, T([40, 29, 30, 1], i64, stride=(1, 40, 0, 1)), T([40, 29, 30, 1], f16, stride=(0, 0, 0, 1))), {})
Operator: aten.sum.SymInt
cnt: 1, ((T([40, 30], f16, stride=(0, 0)), [0], True), {})
cnt: 8, ((T([30, 256], f16), [0], True), {})
cnt: 2, ((T([30, 256], f16, stride=(1, 30)), [0], True), {})
cnt: 1, ((T([40, 30, 60, 60], f16, stride=(0, 0, 0, 0)), [0], True), {})
cnt: 1, ((T([40, 29, 30, 4771], f16), [0, 1], True), {})
Operator: aten.sum.default
cnt: 1, ((T([40, 29, 30], f16),), {})
cnt: 1, ((T([40, 30, 60, 60], f16, stride=(0, 3600, 60, 1)),), {})
cnt: 1, ((T([40, 30], f16, stride=(0, 1)),), {})
Operator: aten.threshold_backward.default
cnt: 4, ((T([30, 256], f16, stride=(1, 30)), T([30, 256], f16), 0), {})
cnt: 4, ((T([30, 256], f16), T([30, 256], f16), 0), {})

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
- [`Background_Matting_training.txt_docs.md`](./Background_Matting_training.txt_docs.md)
- [`fambench_dlrm_training.txt_docs.md`](./fambench_dlrm_training.txt_docs.md)


## Cross-References

- **File Documentation**: `pytorch_struct_training.txt_docs.md`
- **Keyword Index**: `pytorch_struct_training.txt_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
