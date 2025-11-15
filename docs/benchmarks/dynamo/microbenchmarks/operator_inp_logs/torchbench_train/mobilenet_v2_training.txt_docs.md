# Documentation: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/mobilenet_v2_training.txt`

## File Metadata

- **Path**: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/mobilenet_v2_training.txt`
- **Size**: 17,385 bytes (16.98 KB)
- **Type**: Source File (.txt)
- **Extension**: `.txt`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```
Operator: aten.add.Tensor
cnt: 2, ((T([96, 24, 56, 56], f16), T([96, 24, 56, 56], f16)), {})
cnt: 4, ((T([96, 32, 28, 28], f16), T([96, 32, 28, 28], f16)), {})
cnt: 6, ((T([96, 64, 14, 14], f16), T([96, 64, 14, 14], f16)), {})
cnt: 4, ((T([96, 96, 14, 14], f16), T([96, 96, 14, 14], f16)), {})
cnt: 4, ((T([96, 160, 7, 7], f16), T([96, 160, 7, 7], f16)), {})
Operator: aten.addmm.default
cnt: 1, ((T([1000], f16), T([96, 1280], f16), T([1280, 1000], f16, stride=(1, 1280))), {})
Operator: aten.clone.default
cnt: 1, ((T([96, 3, 224, 224], f16),), {})
cnt: 2, ((T([96, 32, 112, 112], f16),), {})
cnt: 1, ((T([96, 96, 112, 112], f16),), {})
cnt: 1, ((T([96, 96, 56, 56], f16),), {})
cnt: 3, ((T([96, 144, 56, 56], f16),), {})
cnt: 1, ((T([96, 144, 28, 28], f16),), {})
cnt: 5, ((T([96, 192, 28, 28], f16),), {})
cnt: 1, ((T([96, 192, 14, 14], f16),), {})
cnt: 8, ((T([96, 384, 14, 14], f16),), {})
cnt: 5, ((T([96, 576, 14, 14], f16),), {})
cnt: 1, ((T([96, 576, 7, 7], f16),), {})
cnt: 6, ((T([96, 960, 7, 7], f16),), {})
cnt: 1, ((T([96, 1280, 7, 7], f16),), {})
Operator: aten.convolution.default
cnt: 1, ((T([96, 3, 224, 224], f16), T([32, 3, 3, 3], f16), None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([96, 32, 112, 112], f16), T([32, 1, 3, 3], f16), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32), {})
cnt: 1, ((T([96, 32, 112, 112], f16), T([16, 32, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([96, 16, 112, 112], f16), T([96, 16, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([96, 96, 112, 112], f16), T([96, 1, 3, 3], f16), None, [2, 2], [1, 1], [1, 1], False, [0, 0], 96), {})
cnt: 1, ((T([96, 96, 56, 56], f16), T([24, 96, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 2, ((T([96, 24, 56, 56], f16), T([144, 24, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([96, 144, 56, 56], f16), T([144, 1, 3, 3], f16), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 144), {})
cnt: 1, ((T([96, 144, 56, 56], f16), T([24, 144, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([96, 144, 56, 56], f16), T([144, 1, 3, 3], f16), None, [2, 2], [1, 1], [1, 1], False, [0, 0], 144), {})
cnt: 1, ((T([96, 144, 28, 28], f16), T([32, 144, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 3, ((T([96, 32, 28, 28], f16), T([192, 32, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 2, ((T([96, 192, 28, 28], f16), T([192, 1, 3, 3], f16), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 192), {})
cnt: 2, ((T([96, 192, 28, 28], f16), T([32, 192, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([96, 192, 28, 28], f16), T([192, 1, 3, 3], f16), None, [2, 2], [1, 1], [1, 1], False, [0, 0], 192), {})
cnt: 1, ((T([96, 192, 14, 14], f16), T([64, 192, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 4, ((T([96, 64, 14, 14], f16), T([384, 64, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 4, ((T([96, 384, 14, 14], f16), T([384, 1, 3, 3], f16), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 384), {})
cnt: 3, ((T([96, 384, 14, 14], f16), T([64, 384, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([96, 384, 14, 14], f16), T([96, 384, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 3, ((T([96, 96, 14, 14], f16), T([576, 96, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 2, ((T([96, 576, 14, 14], f16), T([576, 1, 3, 3], f16), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 576), {})
cnt: 2, ((T([96, 576, 14, 14], f16), T([96, 576, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([96, 576, 14, 14], f16), T([576, 1, 3, 3], f16), None, [2, 2], [1, 1], [1, 1], False, [0, 0], 576), {})
cnt: 1, ((T([96, 576, 7, 7], f16), T([160, 576, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 3, ((T([96, 160, 7, 7], f16), T([960, 160, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 3, ((T([96, 960, 7, 7], f16), T([960, 1, 3, 3], f16), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 960), {})
cnt: 2, ((T([96, 960, 7, 7], f16), T([160, 960, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([96, 960, 7, 7], f16), T([320, 960, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([96, 320, 7, 7], f16), T([1280, 320, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
Operator: aten.convolution_backward.default
cnt: 1, ((T([96, 1280, 7, 7], f16), T([96, 320, 7, 7], f16), T([1280, 320, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([96, 320, 7, 7], f16), T([96, 960, 7, 7], f16), T([320, 960, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 3, ((T([96, 960, 7, 7], f16), T([96, 960, 7, 7], f16), T([960, 1, 3, 3], f16), [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 960, [True, True, False]), {})
cnt: 3, ((T([96, 960, 7, 7], f16), T([96, 160, 7, 7], f16), T([960, 160, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 2, ((T([96, 160, 7, 7], f16), T([96, 960, 7, 7], f16), T([160, 960, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([96, 160, 7, 7], f16), T([96, 576, 7, 7], f16), T([160, 576, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([96, 576, 7, 7], f16), T([96, 576, 14, 14], f16), T([576, 1, 3, 3], f16), [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 576, [True, True, False]), {})
cnt: 3, ((T([96, 576, 14, 14], f16), T([96, 96, 14, 14], f16), T([576, 96, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 2, ((T([96, 96, 14, 14], f16), T([96, 576, 14, 14], f16), T([96, 576, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 2, ((T([96, 576, 14, 14], f16), T([96, 576, 14, 14], f16), T([576, 1, 3, 3], f16), [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 576, [True, True, False]), {})
cnt: 1, ((T([96, 96, 14, 14], f16), T([96, 384, 14, 14], f16), T([96, 384, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 4, ((T([96, 384, 14, 14], f16), T([96, 384, 14, 14], f16), T([384, 1, 3, 3], f16), [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 384, [True, True, False]), {})
cnt: 4, ((T([96, 384, 14, 14], f16), T([96, 64, 14, 14], f16), T([384, 64, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 3, ((T([96, 64, 14, 14], f16), T([96, 384, 14, 14], f16), T([64, 384, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([96, 64, 14, 14], f16), T([96, 192, 14, 14], f16), T([64, 192, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([96, 192, 14, 14], f16), T([96, 192, 28, 28], f16), T([192, 1, 3, 3], f16), [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False]), {})
cnt: 3, ((T([96, 192, 28, 28], f16), T([96, 32, 28, 28], f16), T([192, 32, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 2, ((T([96, 32, 28, 28], f16), T([96, 192, 28, 28], f16), T([32, 192, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 2, ((T([96, 192, 28, 28], f16), T([96, 192, 28, 28], f16), T([192, 1, 3, 3], f16), [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False]), {})
cnt: 1, ((T([96, 32, 28, 28], f16), T([96, 144, 28, 28], f16), T([32, 144, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([96, 144, 28, 28], f16), T([96, 144, 56, 56], f16), T([144, 1, 3, 3], f16), [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 144, [True, True, False]), {})
cnt: 2, ((T([96, 144, 56, 56], f16), T([96, 24, 56, 56], f16), T([144, 24, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([96, 24, 56, 56], f16), T([96, 144, 56, 56], f16), T([24, 144, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([96, 144, 56, 56], f16), T([96, 144, 56, 56], f16), T([144, 1, 3, 3], f16), [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 144, [True, True, False]), {})
cnt: 1, ((T([96, 24, 56, 56], f16), T([96, 96, 56, 56], f16), T([24, 96, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([96, 96, 56, 56], f16), T([96, 96, 112, 112], f16), T([96, 1, 3, 3], f16), [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 96, [True, True, False]), {})
cnt: 1, ((T([96, 96, 112, 112], f16), T([96, 16, 112, 112], f16), T([96, 16, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([96, 16, 112, 112], f16), T([96, 32, 112, 112], f16), T([16, 32, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([96, 32, 112, 112], f16), T([96, 32, 112, 112], f16), T([32, 1, 3, 3], f16), [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]), {})
cnt: 1, ((T([96, 32, 112, 112], f16), T([96, 3, 224, 224], f16), T([32, 3, 3, 3], f16), [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]), {})
Operator: aten.copy_.default
cnt: 1, ((T([96, 3, 224, 224], f16), T([96, 3, 224, 224], f16)), {})
Operator: aten.div.Scalar
cnt: 1, ((T([96, 1280, 7, 7], f16, stride=(1280, 1, 0, 0)), 49), {})
Operator: aten.div.Tensor
cnt: 2, ((T([], f16), 96000), {})
Operator: aten.hardtanh_.default
cnt: 2, ((T([96, 32, 112, 112], f16), 0.0, 6.0), {})
cnt: 1, ((T([96, 96, 112, 112], f16), 0.0, 6.0), {})
cnt: 1, ((T([96, 96, 56, 56], f16), 0.0, 6.0), {})
cnt: 3, ((T([96, 144, 56, 56], f16), 0.0, 6.0), {})
cnt: 1, ((T([96, 144, 28, 28], f16), 0.0, 6.0), {})
cnt: 5, ((T([96, 192, 28, 28], f16), 0.0, 6.0), {})
cnt: 1, ((T([96, 192, 14, 14], f16), 0.0, 6.0), {})
cnt: 8, ((T([96, 384, 14, 14], f16), 0.0, 6.0), {})
cnt: 5, ((T([96, 576, 14, 14], f16), 0.0, 6.0), {})
cnt: 1, ((T([96, 576, 7, 7], f16), 0.0, 6.0), {})
cnt: 6, ((T([96, 960, 7, 7], f16), 0.0, 6.0), {})
cnt: 1, ((T([96, 1280, 7, 7], f16), 0.0, 6.0), {})
Operator: aten.hardtanh_backward.default
cnt: 1, ((T([96, 1280, 7, 7], f16), T([96, 1280, 7, 7], f16), 0.0, 6.0), {})
cnt: 6, ((T([96, 960, 7, 7], f16), T([96, 960, 7, 7], f16), 0.0, 6.0), {})
cnt: 1, ((T([96, 576, 7, 7], f16), T([96, 576, 7, 7], f16), 0.0, 6.0), {})
cnt: 5, ((T([96, 576, 14, 14], f16), T([96, 576, 14, 14], f16), 0.0, 6.0), {})
cnt: 8, ((T([96, 384, 14, 14], f16), T([96, 384, 14, 14], f16), 0.0, 6.0), {})
cnt: 1, ((T([96, 192, 14, 14], f16), T([96, 192, 14, 14], f16), 0.0, 6.0), {})
cnt: 5, ((T([96, 192, 28, 28], f16), T([96, 192, 28, 28], f16), 0.0, 6.0), {})
cnt: 1, ((T([96, 144, 28, 28], f16), T([96, 144, 28, 28], f16), 0.0, 6.0), {})
cnt: 3, ((T([96, 144, 56, 56], f16), T([96, 144, 56, 56], f16), 0.0, 6.0), {})
cnt: 1, ((T([96, 96, 56, 56], f16), T([96, 96, 56, 56], f16), 0.0, 6.0), {})
cnt: 1, ((T([96, 96, 112, 112], f16), T([96, 96, 112, 112], f16), 0.0, 6.0), {})
cnt: 2, ((T([96, 32, 112, 112], f16), T([96, 32, 112, 112], f16), 0.0, 6.0), {})
Operator: aten.mean.dim
cnt: 1, ((T([96, 1280, 7, 7], f16), [-1, -2], True), {})
Operator: aten.mm.default
cnt: 1, ((T([96, 1000], f16, stride=(0, 0)), T([1000, 1280], f16)), {})
cnt: 1, ((T([1000, 96], f16, stride=(0, 0)), T([96, 1280], f16)), {})
Operator: aten.native_batch_norm.default
cnt: 2, ((T([96, 32, 112, 112], f16), T([32], f16), T([32], f16), T([32], f16), T([32], f16), False, 0.1, 1e-05), {})
cnt: 1, ((T([96, 16, 112, 112], f16), T([16], f16), T([16], f16), T([16], f16), T([16], f16), False, 0.1, 1e-05), {})
cnt: 1, ((T([96, 96, 112, 112], f16), T([96], f16), T([96], f16), T([96], f16), T([96], f16), False, 0.1, 1e-05), {})
cnt: 1, ((T([96, 96, 56, 56], f16), T([96], f16), T([96], f16), T([96], f16), T([96], f16), False, 0.1, 1e-05), {})
cnt: 2, ((T([96, 24, 56, 56], f16), T([24], f16), T([24], f16), T([24], f16), T([24], f16), False, 0.1, 1e-05), {})
cnt: 3, ((T([96, 144, 56, 56], f16), T([144], f16), T([144], f16), T([144], f16), T([144], f16), False, 0.1, 1e-05), {})
cnt: 1, ((T([96, 144, 28, 28], f16), T([144], f16), T([144], f16), T([144], f16), T([144], f16), False, 0.1, 1e-05), {})
cnt: 3, ((T([96, 32, 28, 28], f16), T([32], f16), T([32], f16), T([32], f16), T([32], f16), False, 0.1, 1e-05), {})
cnt: 5, ((T([96, 192, 28, 28], f16), T([192], f16), T([192], f16), T([192], f16), T([192], f16), False, 0.1, 1e-05), {})
cnt: 1, ((T([96, 192, 14, 14], f16), T([192], f16), T([192], f16), T([192], f16), T([192], f16), False, 0.1, 1e-05), {})
cnt: 4, ((T([96, 64, 14, 14], f16), T([64], f16), T([64], f16), T([64], f16), T([64], f16), False, 0.1, 1e-05), {})
cnt: 8, ((T([96, 384, 14, 14], f16), T([384], f16), T([384], f16), T([384], f16), T([384], f16), False, 0.1, 1e-05), {})
cnt: 3, ((T([96, 96, 14, 14], f16), T([96], f16), T([96], f16), T([96], f16), T([96], f16), False, 0.1, 1e-05), {})
cnt: 5, ((T([96, 576, 14, 14], f16), T([576], f16), T([576], f16), T([576], f16), T([576], f16), False, 0.1, 1e-05), {})
cnt: 1, ((T([96, 576, 7, 7], f16), T([576], f16), T([576], f16), T([576], f16), T([576], f16), False, 0.1, 1e-05), {})
cnt: 3, ((T([96, 160, 7, 7], f16), T([160], f16), T([160], f16), T([160], f16), T([160], f16), False, 0.1, 1e-05), {})
cnt: 6, ((T([96, 960, 7, 7], f16), T([960], f16), T([960], f16), T([960], f16), T([960], f16), False, 0.1, 1e-05), {})
cnt: 1, ((T([96, 320, 7, 7], f16), T([320], f16), T([320], f16), T([320], f16), T([320], f16), False, 0.1, 1e-05), {})
cnt: 1, ((T([96, 1280, 7, 7], f16), T([1280], f16), T([1280], f16), T([1280], f16), T([1280], f16), False, 0.1, 1e-05), {})
Operator: aten.native_batch_norm_backward.default
cnt: 1, ((T([96, 1280, 7, 7], f16), T([96, 1280, 7, 7], f16), T([1280], f16), T([1280], f16), T([1280], f16), T([1280], f32), T([1280], f32), False, 1e-05, [True, True, True]), {})
cnt: 1, ((T([96, 320, 7, 7], f16), T([96, 320, 7, 7], f16), T([320], f16), T([320], f16), T([320], f16), T([320], f32), T([320], f32), False, 1e-05, [True, True, True]), {})
cnt: 6, ((T([96, 960, 7, 7], f16), T([96, 960, 7, 7], f16), T([960], f16), T([960], f16), T([960], f16), T([960], f32), T([960], f32), False, 1e-05, [True, True, True]), {})
cnt: 3, ((T([96, 160, 7, 7], f16), T([96, 160, 7, 7], f16), T([160], f16), T([160], f16), T([160], f16), T([160], f32), T([160], f32), False, 1e-05, [True, True, True]), {})
cnt: 1, ((T([96, 576, 7, 7], f16), T([96, 576, 7, 7], f16), T([576], f16), T([576], f16), T([576], f16), T([576], f32), T([576], f32), False, 1e-05, [True, True, True]), {})
cnt: 5, ((T([96, 576, 14, 14], f16), T([96, 576, 14, 14], f16), T([576], f16), T([576], f16), T([576], f16), T([576], f32), T([576], f32), False, 1e-05, [True, True, True]), {})
cnt: 3, ((T([96, 96, 14, 14], f16), T([96, 96, 14, 14], f16), T([96], f16), T([96], f16), T([96], f16), T([96], f32), T([96], f32), False, 1e-05, [True, True, True]), {})
cnt: 8, ((T([96, 384, 14, 14], f16), T([96, 384, 14, 14], f16), T([384], f16), T([384], f16), T([384], f16), T([384], f32), T([384], f32), False, 1e-05, [True, True, True]), {})
cnt: 4, ((T([96, 64, 14, 14], f16), T([96, 64, 14, 14], f16), T([64], f16), T([64], f16), T([64], f16), T([64], f32), T([64], f32), False, 1e-05, [True, True, True]), {})
cnt: 1, ((T([96, 192, 14, 14], f16), T([96, 192, 14, 14], f16), T([192], f16), T([192], f16), T([192], f16), T([192], f32), T([192], f32), False, 1e-05, [True, True, True]), {})
cnt: 5, ((T([96, 192, 28, 28], f16), T([96, 192, 28, 28], f16), T([192], f16), T([192], f16), T([192], f16), T([192], f32), T([192], f32), False, 1e-05, [True, True, True]), {})
cnt: 3, ((T([96, 32, 28, 28], f16), T([96, 32, 28, 28], f16), T([32], f16), T([32], f16), T([32], f16), T([32], f32), T([32], f32), False, 1e-05, [True, True, True]), {})
cnt: 1, ((T([96, 144, 28, 28], f16), T([96, 144, 28, 28], f16), T([144], f16), T([144], f16), T([144], f16), T([144], f32), T([144], f32), False, 1e-05, [True, True, True]), {})
cnt: 3, ((T([96, 144, 56, 56], f16), T([96, 144, 56, 56], f16), T([144], f16), T([144], f16), T([144], f16), T([144], f32), T([144], f32), False, 1e-05, [True, True, True]), {})
cnt: 2, ((T([96, 24, 56, 56], f16), T([96, 24, 56, 56], f16), T([24], f16), T([24], f16), T([24], f16), T([24], f32), T([24], f32), False, 1e-05, [True, True, True]), {})
cnt: 1, ((T([96, 96, 56, 56], f16), T([96, 96, 56, 56], f16), T([96], f16), T([96], f16), T([96], f16), T([96], f32), T([96], f32), False, 1e-05, [True, True, True]), {})
cnt: 1, ((T([96, 96, 112, 112], f16), T([96, 96, 112, 112], f16), T([96], f16), T([96], f16), T([96], f16), T([96], f32), T([96], f32), False, 1e-05, [True, True, True]), {})
cnt: 1, ((T([96, 16, 112, 112], f16), T([96, 16, 112, 112], f16), T([16], f16), T([16], f16), T([16], f16), T([16], f32), T([16], f32), False, 1e-05, [True, True, True]), {})
cnt: 2, ((T([96, 32, 112, 112], f16), T([96, 32, 112, 112], f16), T([32], f16), T([32], f16), T([32], f16), T([32], f32), T([32], f32), False, 1e-05, [True, True, True]), {})
Operator: aten.sum.SymInt
cnt: 1, ((T([96, 1000], f16, stride=(0, 0)), [0], True), {})
Operator: aten.sum.default
cnt: 1, ((T([96, 1000], f16),), {})

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

- **File Documentation**: `mobilenet_v2_training.txt_docs.md`
- **Keyword Index**: `mobilenet_v2_training.txt_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
