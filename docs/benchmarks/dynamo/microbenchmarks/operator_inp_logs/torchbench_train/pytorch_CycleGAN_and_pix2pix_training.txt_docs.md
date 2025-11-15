# Documentation: pytorch_CycleGAN_and_pix2pix_training.txt

## File Metadata
- **Path**: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/pytorch_CycleGAN_and_pix2pix_training.txt`
- **Size**: 5181 bytes
- **Lines**: 67
- **Extension**: .txt
- **Type**: Regular file

## Original Source

```txt
Operator: aten.add.Tensor
cnt: 18, ((T([1, 256, 64, 64], f16), T([1, 256, 64, 64], f16)), {})
Operator: aten.clone.default
cnt: 1, ((T([1, 3, 256, 256], f16),), {})
Operator: aten.convolution.default
cnt: 1, ((T([1, 3, 262, 262], f16), T([64, 3, 7, 7], f16), T([64], f16), [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([1, 64, 256, 256], f16), T([128, 64, 3, 3], f16), T([128], f16), [2, 2], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([1, 128, 128, 128], f16), T([256, 128, 3, 3], f16), T([256], f16), [2, 2], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 18, ((T([1, 256, 66, 66], f16), T([256, 256, 3, 3], f16), T([256], f16), [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([1, 256, 64, 64], f16), T([256, 128, 3, 3], f16), T([128], f16), [2, 2], [1, 1], [1, 1], True, [1, 1], 1), {})
cnt: 1, ((T([1, 128, 128, 128], f16), T([128, 64, 3, 3], f16), T([64], f16), [2, 2], [1, 1], [1, 1], True, [1, 1], 1), {})
cnt: 1, ((T([1, 64, 262, 262], f16), T([3, 64, 7, 7], f16), T([3], f16), [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
Operator: aten.convolution_backward.default
cnt: 1, ((T([1, 3, 256, 256], f16), T([1, 64, 262, 262], f16), T([3, 64, 7, 7], f16), [3], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 1, ((T([1, 64, 256, 256], f16), T([1, 128, 128, 128], f16), T([128, 64, 3, 3], f16), [64], [2, 2], [1, 1], [1, 1], True, [1, 1], 1, [True, True, True]), {})
cnt: 1, ((T([1, 128, 128, 128], f16), T([1, 256, 64, 64], f16), T([256, 128, 3, 3], f16), [128], [2, 2], [1, 1], [1, 1], True, [1, 1], 1, [True, True, True]), {})
cnt: 18, ((T([1, 256, 64, 64], f16), T([1, 256, 66, 66], f16), T([256, 256, 3, 3], f16), [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 1, ((T([1, 256, 64, 64], f16), T([1, 128, 128, 128], f16), T([256, 128, 3, 3], f16), [256], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 1, ((T([1, 128, 128, 128], f16), T([1, 64, 256, 256], f16), T([128, 64, 3, 3], f16), [128], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 1, ((T([1, 64, 256, 256], f16), T([1, 3, 262, 262], f16), T([64, 3, 7, 7], f16), [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]), {})
Operator: aten.copy_.default
cnt: 1, ((T([1, 3, 256, 256], f16), T([1, 3, 256, 256], f16)), {})
cnt: 2, ((T([64, 256, 256], f16), T([64, 256, 256], f16)), {})
cnt: 4, ((T([1, 64, 256, 256], f16), T([1, 64, 256, 256], f16)), {})
cnt: 2, ((T([128, 128, 128], f16), T([128, 128, 128], f16)), {})
cnt: 4, ((T([1, 128, 128, 128], f16), T([1, 128, 128, 128], f16)), {})
cnt: 10, ((T([256, 64, 64], f16), T([256, 64, 64], f16)), {})
cnt: 20, ((T([1, 256, 64, 64], f16), T([1, 256, 64, 64], f16)), {})
Operator: aten.div.Tensor
cnt: 2, ((T([], f16), 196608), {})
Operator: aten.native_batch_norm.default
cnt: 2, ((T([1, 64, 256, 256], f16), None, None, None, None, True, 0.1, 1e-05), {})
cnt: 2, ((T([1, 128, 128, 128], f16), None, None, None, None, True, 0.1, 1e-05), {})
cnt: 19, ((T([1, 256, 64, 64], f16), None, None, None, None, True, 0.1, 1e-05), {})
Operator: aten.native_batch_norm_backward.default
cnt: 2, ((T([1, 64, 256, 256], f16), T([1, 64, 256, 256], f16), None, None, None, T([64], f32), T([64], f32), True, 1e-05, [True, False, False]), {})
cnt: 2, ((T([1, 128, 128, 128], f16), T([1, 128, 128, 128], f16), None, None, None, T([128], f32), T([128], f32), True, 1e-05, [True, False, False]), {})
cnt: 19, ((T([1, 256, 64, 64], f16), T([1, 256, 64, 64], f16), None, None, None, T([256], f32), T([256], f32), True, 1e-05, [True, False, False]), {})
Operator: aten.new_empty_strided.default
cnt: 2, ((T([1, 64, 256, 256], f16), [1, 64, 256, 256], [4194304, 65536, 256, 1]), {})
cnt: 2, ((T([1, 128, 128, 128], f16), [1, 128, 128, 128], [2097152, 16384, 128, 1]), {})
cnt: 10, ((T([1, 256, 64, 64], f16), [1, 256, 64, 64], [1048576, 4096, 64, 1]), {})
Operator: aten.new_zeros.default
cnt: 2, ((T([64, 256, 256], f16), [4194304]), {})
cnt: 2, ((T([128, 128, 128], f16), [2097152]), {})
cnt: 10, ((T([256, 64, 64], f16), [1048576]), {})
Operator: aten.reflection_pad2d.default
cnt: 1, ((T([1, 3, 256, 256], f16), [3, 3, 3, 3]), {})
cnt: 18, ((T([1, 256, 64, 64], f16), [1, 1, 1, 1]), {})
cnt: 1, ((T([1, 64, 256, 256], f16), [3, 3, 3, 3]), {})
Operator: aten.reflection_pad2d_backward.default
cnt: 1, ((T([1, 64, 262, 262], f16), T([1, 64, 256, 256], f16), [3, 3, 3, 3]), {})
cnt: 18, ((T([1, 256, 66, 66], f16), T([1, 256, 64, 64], f16), [1, 1, 1, 1]), {})
Operator: aten.relu_.default
cnt: 2, ((T([1, 64, 256, 256], f16),), {})
cnt: 2, ((T([1, 128, 128, 128], f16),), {})
cnt: 10, ((T([1, 256, 64, 64], f16),), {})
Operator: aten.sum.default
cnt: 1, ((T([1, 3, 256, 256], f16),), {})
Operator: aten.tanh.default
cnt: 1, ((T([1, 3, 256, 256], f16),), {})
Operator: aten.tanh_backward.default
cnt: 1, ((T([1, 3, 256, 256], f16, stride=(0, 0, 0, 0)), T([1, 3, 256, 256], f16)), {})
Operator: aten.threshold_backward.default
cnt: 2, ((T([1, 64, 256, 256], f16), T([1, 64, 256, 256], f16), 0), {})
cnt: 2, ((T([1, 128, 128, 128], f16), T([1, 128, 128, 128], f16), 0), {})
cnt: 10, ((T([1, 256, 64, 64], f16), T([1, 256, 64, 64], f16), 0), {})

```

## High-Level Overview

This file is part of the PyTorch repository. It is a source or configuration file.

## Detailed Walkthrough


## Key Components

The file contains 906 words across 67 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 5181 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
