# Documentation: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/timm_train/lcnet_050_training.txt`

## File Metadata

- **Path**: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/timm_train/lcnet_050_training.txt`
- **Size**: 14,104 bytes (13.77 KB)
- **Type**: Source File (.txt)
- **Extension**: `.txt`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```
Operator: aten._log_softmax.default
cnt: 1, ((T([128, 1000], f16), 1, False), {})
Operator: aten._log_softmax_backward_data.default
cnt: 1, ((T([128, 1000], f16), T([128, 1000], f16), 1, f16), {})
Operator: aten.add.Tensor
cnt: 27, ((T([], i64), 1), {})
cnt: 1, ((T([128, 256, 7, 7], f16), T([128, 256, 7, 7], f16)), {})
cnt: 1, ((T([128, 128, 7, 7], f16), T([128, 128, 7, 7], f16)), {})
Operator: aten.addmm.default
cnt: 1, ((T([1000], f16), T([128, 1280], f16), T([1280, 1000], f16, stride=(1, 1280))), {})
Operator: aten.clone.default
cnt: 1, ((T([128, 3, 224, 224], f16),), {})
cnt: 2, ((T([128, 8, 112, 112], f16),), {})
cnt: 1, ((T([128, 16, 112, 112], f16),), {})
cnt: 1, ((T([128, 16, 56, 56], f16),), {})
cnt: 3, ((T([128, 32, 56, 56], f16),), {})
cnt: 1, ((T([128, 32, 28, 28], f16),), {})
cnt: 3, ((T([128, 64, 28, 28], f16),), {})
cnt: 1, ((T([128, 64, 14, 14], f16),), {})
cnt: 11, ((T([128, 128, 14, 14], f16),), {})
cnt: 1, ((T([128, 128, 7, 7], f16),), {})
cnt: 3, ((T([128, 256, 7, 7], f16),), {})
cnt: 1, ((T([128, 1280, 1, 1], f16),), {})
Operator: aten.convolution.default
cnt: 1, ((T([128, 3, 224, 224], f16), T([8, 3, 3, 3], f16), None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([128, 8, 112, 112], f16), T([8, 1, 3, 3], f16), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8), {})
cnt: 1, ((T([128, 8, 112, 112], f16), T([16, 8, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([128, 16, 112, 112], f16), T([16, 1, 3, 3], f16), None, [2, 2], [1, 1], [1, 1], False, [0, 0], 16), {})
cnt: 1, ((T([128, 16, 56, 56], f16), T([32, 16, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([128, 32, 56, 56], f16), T([32, 1, 3, 3], f16), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32), {})
cnt: 1, ((T([128, 32, 56, 56], f16), T([32, 32, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([128, 32, 56, 56], f16), T([32, 1, 3, 3], f16), None, [2, 2], [1, 1], [1, 1], False, [0, 0], 32), {})
cnt: 1, ((T([128, 32, 28, 28], f16), T([64, 32, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([128, 64, 28, 28], f16), T([64, 1, 3, 3], f16), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 64), {})
cnt: 1, ((T([128, 64, 28, 28], f16), T([64, 64, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([128, 64, 28, 28], f16), T([64, 1, 3, 3], f16), None, [2, 2], [1, 1], [1, 1], False, [0, 0], 64), {})
cnt: 1, ((T([128, 64, 14, 14], f16), T([128, 64, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 5, ((T([128, 128, 14, 14], f16), T([128, 1, 5, 5], f16), None, [1, 1], [2, 2], [1, 1], False, [0, 0], 128), {})
cnt: 5, ((T([128, 128, 14, 14], f16), T([128, 128, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([128, 128, 14, 14], f16), T([128, 1, 5, 5], f16), None, [2, 2], [2, 2], [1, 1], False, [0, 0], 128), {})
cnt: 1, ((T([128, 128, 1, 1], f16), T([32, 128, 1, 1], f16), T([32], f16), [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([128, 32, 1, 1], f16), T([128, 32, 1, 1], f16), T([128], f16), [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([128, 128, 7, 7], f16), T([256, 128, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([128, 256, 7, 7], f16), T([256, 1, 5, 5], f16), None, [1, 1], [2, 2], [1, 1], False, [0, 0], 256), {})
cnt: 1, ((T([128, 256, 1, 1], f16), T([64, 256, 1, 1], f16), T([64], f16), [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([128, 64, 1, 1], f16), T([256, 64, 1, 1], f16), T([256], f16), [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([128, 256, 7, 7], f16), T([256, 256, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([128, 256, 1, 1], f16), T([1280, 256, 1, 1], f16), T([1280], f16), [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
Operator: aten.convolution_backward.default
cnt: 1, ((T([128, 1280, 1, 1], f16), T([128, 256, 1, 1], f16), T([1280, 256, 1, 1], f16), [1280], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 1, ((T([128, 256, 7, 7], f16), T([128, 256, 7, 7], f16), T([256, 256, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([128, 256, 1, 1], f16), T([128, 64, 1, 1], f16), T([256, 64, 1, 1], f16), [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 1, ((T([128, 64, 1, 1], f16), T([128, 256, 1, 1], f16), T([64, 256, 1, 1], f16), [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 1, ((T([128, 256, 7, 7], f16), T([128, 256, 7, 7], f16), T([256, 1, 5, 5], f16), [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 256, [True, True, False]), {})
cnt: 1, ((T([128, 256, 7, 7], f16), T([128, 128, 7, 7], f16), T([256, 128, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([128, 128, 1, 1], f16), T([128, 32, 1, 1], f16), T([128, 32, 1, 1], f16), [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 1, ((T([128, 32, 1, 1], f16), T([128, 128, 1, 1], f16), T([32, 128, 1, 1], f16), [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]), {})
cnt: 1, ((T([128, 128, 7, 7], f16), T([128, 128, 14, 14], f16), T([128, 1, 5, 5], f16), [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 128, [True, True, False]), {})
cnt: 5, ((T([128, 128, 14, 14], f16), T([128, 128, 14, 14], f16), T([128, 128, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 5, ((T([128, 128, 14, 14], f16), T([128, 128, 14, 14], f16), T([128, 1, 5, 5], f16), [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 128, [True, True, False]), {})
cnt: 1, ((T([128, 128, 14, 14], f16), T([128, 64, 14, 14], f16), T([128, 64, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([128, 64, 14, 14], f16), T([128, 64, 28, 28], f16), T([64, 1, 3, 3], f16), [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False]), {})
cnt: 1, ((T([128, 64, 28, 28], f16), T([128, 64, 28, 28], f16), T([64, 64, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([128, 64, 28, 28], f16), T([128, 64, 28, 28], f16), T([64, 1, 3, 3], f16), [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False]), {})
cnt: 1, ((T([128, 64, 28, 28], f16), T([128, 32, 28, 28], f16), T([64, 32, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([128, 32, 28, 28], f16), T([128, 32, 56, 56], f16), T([32, 1, 3, 3], f16), [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]), {})
cnt: 1, ((T([128, 32, 56, 56], f16), T([128, 32, 56, 56], f16), T([32, 32, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([128, 32, 56, 56], f16), T([128, 32, 56, 56], f16), T([32, 1, 3, 3], f16), [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]), {})
cnt: 1, ((T([128, 32, 56, 56], f16), T([128, 16, 56, 56], f16), T([32, 16, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([128, 16, 56, 56], f16), T([128, 16, 112, 112], f16), T([16, 1, 3, 3], f16), [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False]), {})
cnt: 1, ((T([128, 16, 112, 112], f16), T([128, 8, 112, 112], f16), T([16, 8, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([128, 8, 112, 112], f16), T([128, 8, 112, 112], f16), T([8, 1, 3, 3], f16), [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]), {})
cnt: 1, ((T([128, 8, 112, 112], f16), T([128, 3, 224, 224], f16), T([8, 3, 3, 3], f16), [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]), {})
Operator: aten.copy_.default
cnt: 1, ((T([128, 3, 224, 224], f16), T([128, 3, 224, 224], f16)), {})
Operator: aten.div.Scalar
cnt: 2, ((T([128, 256, 7, 7], f16, stride=(256, 1, 0, 0)), 49), {})
cnt: 1, ((T([128, 128, 7, 7], f16, stride=(128, 1, 0, 0)), 49), {})
Operator: aten.hardsigmoid.default
cnt: 1, ((T([128, 128, 1, 1], f16),), {})
cnt: 1, ((T([128, 256, 1, 1], f16),), {})
Operator: aten.hardsigmoid_backward.default
cnt: 1, ((T([128, 256, 1, 1], f16), T([128, 256, 1, 1], f16)), {})
cnt: 1, ((T([128, 128, 1, 1], f16), T([128, 128, 1, 1], f16)), {})
Operator: aten.hardswish_.default
cnt: 2, ((T([128, 8, 112, 112], f16),), {})
cnt: 1, ((T([128, 16, 112, 112], f16),), {})
cnt: 1, ((T([128, 16, 56, 56], f16),), {})
cnt: 3, ((T([128, 32, 56, 56], f16),), {})
cnt: 1, ((T([128, 32, 28, 28], f16),), {})
cnt: 3, ((T([128, 64, 28, 28], f16),), {})
cnt: 1, ((T([128, 64, 14, 14], f16),), {})
cnt: 11, ((T([128, 128, 14, 14], f16),), {})
cnt: 1, ((T([128, 128, 7, 7], f16),), {})
cnt: 3, ((T([128, 256, 7, 7], f16),), {})
cnt: 1, ((T([128, 1280, 1, 1], f16),), {})
Operator: aten.hardswish_backward.default
cnt: 1, ((T([128, 1280, 1, 1], f16), T([128, 1280, 1, 1], f16)), {})
cnt: 3, ((T([128, 256, 7, 7], f16), T([128, 256, 7, 7], f16)), {})
cnt: 1, ((T([128, 128, 7, 7], f16), T([128, 128, 7, 7], f16)), {})
cnt: 11, ((T([128, 128, 14, 14], f16), T([128, 128, 14, 14], f16)), {})
cnt: 1, ((T([128, 64, 14, 14], f16), T([128, 64, 14, 14], f16)), {})
cnt: 3, ((T([128, 64, 28, 28], f16), T([128, 64, 28, 28], f16)), {})
cnt: 1, ((T([128, 32, 28, 28], f16), T([128, 32, 28, 28], f16)), {})
cnt: 3, ((T([128, 32, 56, 56], f16), T([128, 32, 56, 56], f16)), {})
cnt: 1, ((T([128, 16, 56, 56], f16), T([128, 16, 56, 56], f16)), {})
cnt: 1, ((T([128, 16, 112, 112], f16), T([128, 16, 112, 112], f16)), {})
cnt: 2, ((T([128, 8, 112, 112], f16), T([128, 8, 112, 112], f16)), {})
Operator: aten.lift_fresh_copy.default
cnt: 1, ((T([128], i64),), {})
Operator: aten.mean.dim
cnt: 1, ((T([128, 128, 7, 7], f16), [2, 3], True), {})
cnt: 1, ((T([128, 256, 7, 7], f16), [2, 3], True), {})
cnt: 1, ((T([128, 256, 7, 7], f16), [-1, -2], True), {})
Operator: aten.mm.default
cnt: 1, ((T([128, 1000], f16), T([1000, 1280], f16)), {})
cnt: 1, ((T([1000, 128], f16, stride=(1, 1000)), T([128, 1280], f16)), {})
Operator: aten.mul.Tensor
cnt: 2, ((T([128, 128, 7, 7], f16), T([128, 128, 1, 1], f16)), {})
cnt: 2, ((T([128, 256, 7, 7], f16), T([128, 256, 1, 1], f16)), {})
cnt: 1, ((T([128, 256, 7, 7], f16), T([128, 256, 7, 7], f16)), {})
cnt: 1, ((T([128, 128, 7, 7], f16), T([128, 128, 7, 7], f16)), {})
Operator: aten.native_batch_norm.default
cnt: 2, ((T([128, 8, 112, 112], f16), T([8], f16), T([8], f16), T([8], f16), T([8], f16), True, 0.1, 1e-05), {})
cnt: 1, ((T([128, 16, 112, 112], f16), T([16], f16), T([16], f16), T([16], f16), T([16], f16), True, 0.1, 1e-05), {})
cnt: 1, ((T([128, 16, 56, 56], f16), T([16], f16), T([16], f16), T([16], f16), T([16], f16), True, 0.1, 1e-05), {})
cnt: 3, ((T([128, 32, 56, 56], f16), T([32], f16), T([32], f16), T([32], f16), T([32], f16), True, 0.1, 1e-05), {})
cnt: 1, ((T([128, 32, 28, 28], f16), T([32], f16), T([32], f16), T([32], f16), T([32], f16), True, 0.1, 1e-05), {})
cnt: 3, ((T([128, 64, 28, 28], f16), T([64], f16), T([64], f16), T([64], f16), T([64], f16), True, 0.1, 1e-05), {})
cnt: 1, ((T([128, 64, 14, 14], f16), T([64], f16), T([64], f16), T([64], f16), T([64], f16), True, 0.1, 1e-05), {})
cnt: 11, ((T([128, 128, 14, 14], f16), T([128], f16), T([128], f16), T([128], f16), T([128], f16), True, 0.1, 1e-05), {})
cnt: 1, ((T([128, 128, 7, 7], f16), T([128], f16), T([128], f16), T([128], f16), T([128], f16), True, 0.1, 1e-05), {})
cnt: 3, ((T([128, 256, 7, 7], f16), T([256], f16), T([256], f16), T([256], f16), T([256], f16), True, 0.1, 1e-05), {})
Operator: aten.native_batch_norm_backward.default
cnt: 3, ((T([128, 256, 7, 7], f16), T([128, 256, 7, 7], f16), T([256], f16), T([256], f16), T([256], f16), T([256], f32), T([256], f32), True, 1e-05, [True, True, True]), {})
cnt: 1, ((T([128, 128, 7, 7], f16), T([128, 128, 7, 7], f16), T([128], f16), T([128], f16), T([128], f16), T([128], f32), T([128], f32), True, 1e-05, [True, True, True]), {})
cnt: 11, ((T([128, 128, 14, 14], f16), T([128, 128, 14, 14], f16), T([128], f16), T([128], f16), T([128], f16), T([128], f32), T([128], f32), True, 1e-05, [True, True, True]), {})
cnt: 1, ((T([128, 64, 14, 14], f16), T([128, 64, 14, 14], f16), T([64], f16), T([64], f16), T([64], f16), T([64], f32), T([64], f32), True, 1e-05, [True, True, True]), {})
cnt: 3, ((T([128, 64, 28, 28], f16), T([128, 64, 28, 28], f16), T([64], f16), T([64], f16), T([64], f16), T([64], f32), T([64], f32), True, 1e-05, [True, True, True]), {})
cnt: 1, ((T([128, 32, 28, 28], f16), T([128, 32, 28, 28], f16), T([32], f16), T([32], f16), T([32], f16), T([32], f32), T([32], f32), True, 1e-05, [True, True, True]), {})
cnt: 3, ((T([128, 32, 56, 56], f16), T([128, 32, 56, 56], f16), T([32], f16), T([32], f16), T([32], f16), T([32], f32), T([32], f32), True, 1e-05, [True, True, True]), {})
cnt: 1, ((T([128, 16, 56, 56], f16), T([128, 16, 56, 56], f16), T([16], f16), T([16], f16), T([16], f16), T([16], f32), T([16], f32), True, 1e-05, [True, True, True]), {})
cnt: 1, ((T([128, 16, 112, 112], f16), T([128, 16, 112, 112], f16), T([16], f16), T([16], f16), T([16], f16), T([16], f32), T([16], f32), True, 1e-05, [True, True, True]), {})
cnt: 2, ((T([128, 8, 112, 112], f16), T([128, 8, 112, 112], f16), T([8], f16), T([8], f16), T([8], f16), T([8], f32), T([8], f32), True, 1e-05, [True, True, True]), {})
Operator: aten.nll_loss_backward.default
cnt: 1, ((T([], f16), T([128, 1000], f16), T([128], i64), None, 1, -100, T([], f16)), {})
Operator: aten.nll_loss_forward.default
cnt: 1, ((T([128, 1000], f16), T([128], i64), None, 1, -100), {})
Operator: aten.relu_.default
cnt: 1, ((T([128, 32, 1, 1], f16),), {})
cnt: 1, ((T([128, 64, 1, 1], f16),), {})
Operator: aten.sum.SymInt
cnt: 1, ((T([128, 1000], f16), [0], True), {})
cnt: 1, ((T([128, 256, 7, 7], f16), [2, 3], True), {})
cnt: 1, ((T([128, 128, 7, 7], f16), [2, 3], True), {})
Operator: aten.threshold_backward.default
cnt: 1, ((T([128, 64, 1, 1], f16), T([128, 64, 1, 1], f16), 0), {})
cnt: 1, ((T([128, 32, 1, 1], f16), T([128, 32, 1, 1], f16), 0), {})

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

- **File Documentation**: `lcnet_050_training.txt_docs.md`
- **Keyword Index**: `lcnet_050_training.txt_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
