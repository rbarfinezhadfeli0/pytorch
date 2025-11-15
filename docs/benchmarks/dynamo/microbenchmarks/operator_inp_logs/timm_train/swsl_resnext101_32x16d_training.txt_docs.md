# Documentation: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/timm_train/swsl_resnext101_32x16d_training.txt`

## File Metadata

- **Path**: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/timm_train/swsl_resnext101_32x16d_training.txt`
- **Size**: 14,246 bytes (13.91 KB)
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
cnt: 2, ((T([32, 2048, 7, 7], f16), T([32, 2048, 7, 7], f16)), {})
cnt: 23, ((T([32, 1024, 14, 14], f16), T([32, 1024, 14, 14], f16)), {})
cnt: 4, ((T([32, 512, 28, 28], f16), T([32, 512, 28, 28], f16)), {})
cnt: 3, ((T([32, 256, 56, 56], f16), T([32, 256, 56, 56], f16)), {})
cnt: 1, ((T([32, 64, 56, 56], f16), T([32, 64, 56, 56], f16)), {})
Operator: aten.add_.Tensor
cnt: 104, ((T([], i64), 1), {})
cnt: 3, ((T([32, 256, 56, 56], f16), T([32, 256, 56, 56], f16)), {})
cnt: 4, ((T([32, 512, 28, 28], f16), T([32, 512, 28, 28], f16)), {})
cnt: 23, ((T([32, 1024, 14, 14], f16), T([32, 1024, 14, 14], f16)), {})
cnt: 3, ((T([32, 2048, 7, 7], f16), T([32, 2048, 7, 7], f16)), {})
Operator: aten.addmm.default
cnt: 1, ((T([1000], f16), T([32, 2048], f16), T([2048, 1000], f16, stride=(1, 2048))), {})
Operator: aten.clone.default
cnt: 1, ((T([32, 3, 224, 224], f16),), {})
Operator: aten.convolution.default
cnt: 1, ((T([32, 3, 224, 224], f16), T([64, 3, 7, 7], f16), None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([32, 64, 56, 56], f16), T([512, 64, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 3, ((T([32, 512, 56, 56], f16), T([512, 16, 3, 3], f16), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32), {})
cnt: 3, ((T([32, 512, 56, 56], f16), T([256, 512, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([32, 64, 56, 56], f16), T([256, 64, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 2, ((T([32, 256, 56, 56], f16), T([512, 256, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([32, 256, 56, 56], f16), T([1024, 256, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([32, 1024, 56, 56], f16), T([1024, 32, 3, 3], f16), None, [2, 2], [1, 1], [1, 1], False, [0, 0], 32), {})
cnt: 4, ((T([32, 1024, 28, 28], f16), T([512, 1024, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([32, 256, 56, 56], f16), T([512, 256, 1, 1], f16), None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 3, ((T([32, 512, 28, 28], f16), T([1024, 512, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 3, ((T([32, 1024, 28, 28], f16), T([1024, 32, 3, 3], f16), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32), {})
cnt: 1, ((T([32, 512, 28, 28], f16), T([2048, 512, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([32, 2048, 28, 28], f16), T([2048, 64, 3, 3], f16), None, [2, 2], [1, 1], [1, 1], False, [0, 0], 32), {})
cnt: 23, ((T([32, 2048, 14, 14], f16), T([1024, 2048, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([32, 512, 28, 28], f16), T([1024, 512, 1, 1], f16), None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 22, ((T([32, 1024, 14, 14], f16), T([2048, 1024, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 22, ((T([32, 2048, 14, 14], f16), T([2048, 64, 3, 3], f16), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32), {})
cnt: 1, ((T([32, 1024, 14, 14], f16), T([4096, 1024, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([32, 4096, 14, 14], f16), T([4096, 128, 3, 3], f16), None, [2, 2], [1, 1], [1, 1], False, [0, 0], 32), {})
cnt: 3, ((T([32, 4096, 7, 7], f16), T([2048, 4096, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([32, 1024, 14, 14], f16), T([2048, 1024, 1, 1], f16), None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 2, ((T([32, 2048, 7, 7], f16), T([4096, 2048, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 2, ((T([32, 4096, 7, 7], f16), T([4096, 128, 3, 3], f16), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32), {})
Operator: aten.convolution_backward.default
cnt: 3, ((T([32, 2048, 7, 7], f16), T([32, 4096, 7, 7], f16), T([2048, 4096, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 2, ((T([32, 4096, 7, 7], f16), T([32, 4096, 7, 7], f16), T([4096, 128, 3, 3], f16), [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]), {})
cnt: 2, ((T([32, 4096, 7, 7], f16), T([32, 2048, 7, 7], f16), T([4096, 2048, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([32, 2048, 7, 7], f16), T([32, 1024, 14, 14], f16), T([2048, 1024, 1, 1], f16), [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([32, 4096, 7, 7], f16), T([32, 4096, 14, 14], f16), T([4096, 128, 3, 3], f16), [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]), {})
cnt: 1, ((T([32, 4096, 14, 14], f16), T([32, 1024, 14, 14], f16), T([4096, 1024, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 23, ((T([32, 1024, 14, 14], f16), T([32, 2048, 14, 14], f16), T([1024, 2048, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 22, ((T([32, 2048, 14, 14], f16), T([32, 2048, 14, 14], f16), T([2048, 64, 3, 3], f16), [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]), {})
cnt: 22, ((T([32, 2048, 14, 14], f16), T([32, 1024, 14, 14], f16), T([2048, 1024, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([32, 1024, 14, 14], f16), T([32, 512, 28, 28], f16), T([1024, 512, 1, 1], f16), [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([32, 2048, 14, 14], f16), T([32, 2048, 28, 28], f16), T([2048, 64, 3, 3], f16), [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]), {})
cnt: 1, ((T([32, 2048, 28, 28], f16), T([32, 512, 28, 28], f16), T([2048, 512, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 4, ((T([32, 512, 28, 28], f16), T([32, 1024, 28, 28], f16), T([512, 1024, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 3, ((T([32, 1024, 28, 28], f16), T([32, 1024, 28, 28], f16), T([1024, 32, 3, 3], f16), [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]), {})
cnt: 3, ((T([32, 1024, 28, 28], f16), T([32, 512, 28, 28], f16), T([1024, 512, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([32, 512, 28, 28], f16), T([32, 256, 56, 56], f16), T([512, 256, 1, 1], f16), [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([32, 1024, 28, 28], f16), T([32, 1024, 56, 56], f16), T([1024, 32, 3, 3], f16), [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]), {})
cnt: 1, ((T([32, 1024, 56, 56], f16), T([32, 256, 56, 56], f16), T([1024, 256, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 3, ((T([32, 256, 56, 56], f16), T([32, 512, 56, 56], f16), T([256, 512, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 3, ((T([32, 512, 56, 56], f16), T([32, 512, 56, 56], f16), T([512, 16, 3, 3], f16), [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]), {})
cnt: 2, ((T([32, 512, 56, 56], f16), T([32, 256, 56, 56], f16), T([512, 256, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([32, 256, 56, 56], f16), T([32, 64, 56, 56], f16), T([256, 64, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([32, 512, 56, 56], f16), T([32, 64, 56, 56], f16), T([512, 64, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([32, 64, 112, 112], f16), T([32, 3, 224, 224], f16), T([64, 3, 7, 7], f16), [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]), {})
Operator: aten.copy_.default
cnt: 1, ((T([32, 3, 224, 224], f16), T([32, 3, 224, 224], f16)), {})
Operator: aten.div.Scalar
cnt: 1, ((T([32, 2048, 7, 7], f16, stride=(2048, 1, 0, 0)), 49), {})
Operator: aten.lift_fresh_copy.default
cnt: 1, ((T([32], i64),), {})
Operator: aten.max_pool2d_with_indices.default
cnt: 1, ((T([32, 64, 112, 112], f16), [3, 3], [2, 2], [1, 1]), {})
Operator: aten.max_pool2d_with_indices_backward.default
cnt: 1, ((T([32, 64, 56, 56], f16), T([32, 64, 112, 112], f16), [3, 3], [2, 2], [1, 1], [1, 1], False, T([32, 64, 56, 56], i64)), {})
Operator: aten.mean.dim
cnt: 1, ((T([32, 2048, 7, 7], f16), [-1, -2], True), {})
Operator: aten.mm.default
cnt: 1, ((T([32, 1000], f16), T([1000, 2048], f16)), {})
cnt: 1, ((T([1000, 32], f16, stride=(1, 1000)), T([32, 2048], f16)), {})
Operator: aten.native_batch_norm.default
cnt: 1, ((T([32, 64, 112, 112], f16), T([64], f16), T([64], f16), T([64], f16), T([64], f16), True, 0.1, 1e-05), {})
cnt: 6, ((T([32, 512, 56, 56], f16), T([512], f16), T([512], f16), T([512], f16), T([512], f16), True, 0.1, 1e-05), {})
cnt: 4, ((T([32, 256, 56, 56], f16), T([256], f16), T([256], f16), T([256], f16), T([256], f16), True, 0.1, 1e-05), {})
cnt: 1, ((T([32, 1024, 56, 56], f16), T([1024], f16), T([1024], f16), T([1024], f16), T([1024], f16), True, 0.1, 1e-05), {})
cnt: 7, ((T([32, 1024, 28, 28], f16), T([1024], f16), T([1024], f16), T([1024], f16), T([1024], f16), True, 0.1, 1e-05), {})
cnt: 5, ((T([32, 512, 28, 28], f16), T([512], f16), T([512], f16), T([512], f16), T([512], f16), True, 0.1, 1e-05), {})
cnt: 1, ((T([32, 2048, 28, 28], f16), T([2048], f16), T([2048], f16), T([2048], f16), T([2048], f16), True, 0.1, 1e-05), {})
cnt: 45, ((T([32, 2048, 14, 14], f16), T([2048], f16), T([2048], f16), T([2048], f16), T([2048], f16), True, 0.1, 1e-05), {})
cnt: 24, ((T([32, 1024, 14, 14], f16), T([1024], f16), T([1024], f16), T([1024], f16), T([1024], f16), True, 0.1, 1e-05), {})
cnt: 1, ((T([32, 4096, 14, 14], f16), T([4096], f16), T([4096], f16), T([4096], f16), T([4096], f16), True, 0.1, 1e-05), {})
cnt: 5, ((T([32, 4096, 7, 7], f16), T([4096], f16), T([4096], f16), T([4096], f16), T([4096], f16), True, 0.1, 1e-05), {})
cnt: 4, ((T([32, 2048, 7, 7], f16), T([2048], f16), T([2048], f16), T([2048], f16), T([2048], f16), True, 0.1, 1e-05), {})
Operator: aten.native_batch_norm_backward.default
cnt: 4, ((T([32, 2048, 7, 7], f16), T([32, 2048, 7, 7], f16), T([2048], f16), T([2048], f16), T([2048], f16), T([2048], f32), T([2048], f32), True, 1e-05, [True, True, True]), {})
cnt: 5, ((T([32, 4096, 7, 7], f16), T([32, 4096, 7, 7], f16), T([4096], f16), T([4096], f16), T([4096], f16), T([4096], f32), T([4096], f32), True, 1e-05, [True, True, True]), {})
cnt: 1, ((T([32, 4096, 14, 14], f16), T([32, 4096, 14, 14], f16), T([4096], f16), T([4096], f16), T([4096], f16), T([4096], f32), T([4096], f32), True, 1e-05, [True, True, True]), {})
cnt: 24, ((T([32, 1024, 14, 14], f16), T([32, 1024, 14, 14], f16), T([1024], f16), T([1024], f16), T([1024], f16), T([1024], f32), T([1024], f32), True, 1e-05, [True, True, True]), {})
cnt: 45, ((T([32, 2048, 14, 14], f16), T([32, 2048, 14, 14], f16), T([2048], f16), T([2048], f16), T([2048], f16), T([2048], f32), T([2048], f32), True, 1e-05, [True, True, True]), {})
cnt: 1, ((T([32, 2048, 28, 28], f16), T([32, 2048, 28, 28], f16), T([2048], f16), T([2048], f16), T([2048], f16), T([2048], f32), T([2048], f32), True, 1e-05, [True, True, True]), {})
cnt: 5, ((T([32, 512, 28, 28], f16), T([32, 512, 28, 28], f16), T([512], f16), T([512], f16), T([512], f16), T([512], f32), T([512], f32), True, 1e-05, [True, True, True]), {})
cnt: 7, ((T([32, 1024, 28, 28], f16), T([32, 1024, 28, 28], f16), T([1024], f16), T([1024], f16), T([1024], f16), T([1024], f32), T([1024], f32), True, 1e-05, [True, True, True]), {})
cnt: 1, ((T([32, 1024, 56, 56], f16), T([32, 1024, 56, 56], f16), T([1024], f16), T([1024], f16), T([1024], f16), T([1024], f32), T([1024], f32), True, 1e-05, [True, True, True]), {})
cnt: 4, ((T([32, 256, 56, 56], f16), T([32, 256, 56, 56], f16), T([256], f16), T([256], f16), T([256], f16), T([256], f32), T([256], f32), True, 1e-05, [True, True, True]), {})
cnt: 6, ((T([32, 512, 56, 56], f16), T([32, 512, 56, 56], f16), T([512], f16), T([512], f16), T([512], f16), T([512], f32), T([512], f32), True, 1e-05, [True, True, True]), {})
cnt: 1, ((T([32, 64, 112, 112], f16), T([32, 64, 112, 112], f16), T([64], f16), T([64], f16), T([64], f16), T([64], f32), T([64], f32), True, 1e-05, [True, True, True]), {})
Operator: aten.nll_loss_backward.default
cnt: 1, ((T([], f16), T([32, 1000], f16), T([32], i64), None, 1, -100, T([], f16)), {})
Operator: aten.nll_loss_forward.default
cnt: 1, ((T([32, 1000], f16), T([32], i64), None, 1, -100), {})
Operator: aten.relu_.default
cnt: 1, ((T([32, 64, 112, 112], f16),), {})
cnt: 6, ((T([32, 512, 56, 56], f16),), {})
cnt: 3, ((T([32, 256, 56, 56], f16),), {})
cnt: 1, ((T([32, 1024, 56, 56], f16),), {})
cnt: 7, ((T([32, 1024, 28, 28], f16),), {})
cnt: 4, ((T([32, 512, 28, 28], f16),), {})
cnt: 1, ((T([32, 2048, 28, 28], f16),), {})
cnt: 45, ((T([32, 2048, 14, 14], f16),), {})
cnt: 23, ((T([32, 1024, 14, 14], f16),), {})
cnt: 1, ((T([32, 4096, 14, 14], f16),), {})
cnt: 5, ((T([32, 4096, 7, 7], f16),), {})
cnt: 3, ((T([32, 2048, 7, 7], f16),), {})
Operator: aten.sum.SymInt
cnt: 1, ((T([32, 1000], f16), [0], True), {})
Operator: aten.threshold_backward.default
cnt: 3, ((T([32, 2048, 7, 7], f16), T([32, 2048, 7, 7], f16), 0), {})
cnt: 5, ((T([32, 4096, 7, 7], f16), T([32, 4096, 7, 7], f16), 0), {})
cnt: 1, ((T([32, 4096, 14, 14], f16), T([32, 4096, 14, 14], f16), 0), {})
cnt: 23, ((T([32, 1024, 14, 14], f16), T([32, 1024, 14, 14], f16), 0), {})
cnt: 45, ((T([32, 2048, 14, 14], f16), T([32, 2048, 14, 14], f16), 0), {})
cnt: 1, ((T([32, 2048, 28, 28], f16), T([32, 2048, 28, 28], f16), 0), {})
cnt: 4, ((T([32, 512, 28, 28], f16), T([32, 512, 28, 28], f16), 0), {})
cnt: 7, ((T([32, 1024, 28, 28], f16), T([32, 1024, 28, 28], f16), 0), {})
cnt: 1, ((T([32, 1024, 56, 56], f16), T([32, 1024, 56, 56], f16), 0), {})
cnt: 3, ((T([32, 256, 56, 56], f16), T([32, 256, 56, 56], f16), 0), {})
cnt: 6, ((T([32, 512, 56, 56], f16), T([32, 512, 56, 56], f16), 0), {})
cnt: 1, ((T([32, 64, 112, 112], f16), T([32, 64, 112, 112], f16), 0), {})

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

- **File Documentation**: `swsl_resnext101_32x16d_training.txt_docs.md`
- **Keyword Index**: `swsl_resnext101_32x16d_training.txt_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
