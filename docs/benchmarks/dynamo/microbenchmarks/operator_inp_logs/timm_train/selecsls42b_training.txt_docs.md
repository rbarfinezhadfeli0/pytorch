# Documentation: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/timm_train/selecsls42b_training.txt`

## File Metadata

- **Path**: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/timm_train/selecsls42b_training.txt`
- **Size**: 17,146 bytes (16.74 KB)
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
cnt: 1, ((T([128, 152, 14, 14], f16, stride=(178752, 196, 14, 1)), T([128, 152, 14, 14], f16)), {})
cnt: 2, ((T([128, 304, 14, 14], f16, stride=(178752, 196, 14, 1)), T([128, 304, 14, 14], f16)), {})
cnt: 1, ((T([128, 152, 14, 14], f16, stride=(119168, 196, 14, 1)), T([128, 152, 14, 14], f16)), {})
cnt: 1, ((T([128, 304, 14, 14], f16, stride=(119168, 196, 14, 1)), T([128, 304, 14, 14], f16)), {})
cnt: 1, ((T([128, 72, 28, 28], f16, stride=(338688, 784, 28, 1)), T([128, 72, 28, 28], f16)), {})
cnt: 2, ((T([128, 144, 28, 28], f16, stride=(338688, 784, 28, 1)), T([128, 144, 28, 28], f16)), {})
cnt: 1, ((T([128, 72, 28, 28], f16, stride=(225792, 784, 28, 1)), T([128, 72, 28, 28], f16)), {})
cnt: 1, ((T([128, 144, 28, 28], f16, stride=(225792, 784, 28, 1)), T([128, 144, 28, 28], f16)), {})
cnt: 1, ((T([128, 32, 56, 56], f16, stride=(602112, 3136, 56, 1)), T([128, 32, 56, 56], f16)), {})
cnt: 2, ((T([128, 64, 56, 56], f16, stride=(602112, 3136, 56, 1)), T([128, 64, 56, 56], f16)), {})
cnt: 1, ((T([128, 32, 56, 56], f16, stride=(401408, 3136, 56, 1)), T([128, 32, 56, 56], f16)), {})
cnt: 1, ((T([128, 64, 56, 56], f16, stride=(401408, 3136, 56, 1)), T([128, 64, 56, 56], f16)), {})
Operator: aten.add_.Tensor
cnt: 41, ((T([], i64), 1), {})
Operator: aten.addmm.default
cnt: 1, ((T([1000], f16), T([128, 1024], f16), T([1024, 1000], f16, stride=(1, 1024))), {})
Operator: aten.cat.default
cnt: 1, (([T([128, 64, 56, 56], f16), T([128, 32, 56, 56], f16), T([128, 32, 56, 56], f16)], 1), {})
cnt: 1, (([T([128, 64, 56, 56], f16), T([128, 32, 56, 56], f16), T([128, 32, 56, 56], f16), T([128, 64, 56, 56], f16)], 1), {})
cnt: 1, (([T([128, 144, 28, 28], f16), T([128, 72, 28, 28], f16), T([128, 72, 28, 28], f16)], 1), {})
cnt: 1, (([T([128, 144, 28, 28], f16), T([128, 72, 28, 28], f16), T([128, 72, 28, 28], f16), T([128, 144, 28, 28], f16)], 1), {})
cnt: 1, (([T([128, 304, 14, 14], f16), T([128, 152, 14, 14], f16), T([128, 152, 14, 14], f16)], 1), {})
cnt: 1, (([T([128, 304, 14, 14], f16), T([128, 152, 14, 14], f16), T([128, 152, 14, 14], f16), T([128, 304, 14, 14], f16)], 1), {})
Operator: aten.clone.default
cnt: 1, ((T([128, 3, 224, 224], f16),), {})
Operator: aten.convolution.default
cnt: 1, ((T([128, 3, 224, 224], f16), T([32, 3, 3, 3], f16), None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([128, 32, 112, 112], f16), T([64, 32, 3, 3], f16), None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 2, ((T([128, 64, 56, 56], f16), T([64, 64, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 4, ((T([128, 64, 56, 56], f16), T([32, 64, 3, 3], f16), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 2, ((T([128, 32, 56, 56], f16), T([64, 32, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([128, 128, 56, 56], f16), T([64, 128, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([128, 64, 56, 56], f16), T([64, 64, 3, 3], f16), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([128, 192, 56, 56], f16), T([128, 192, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([128, 128, 56, 56], f16), T([144, 128, 3, 3], f16), None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 2, ((T([128, 144, 28, 28], f16), T([144, 144, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 4, ((T([128, 144, 28, 28], f16), T([72, 144, 3, 3], f16), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 2, ((T([128, 72, 28, 28], f16), T([144, 72, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([128, 288, 28, 28], f16), T([144, 288, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([128, 144, 28, 28], f16), T([144, 144, 3, 3], f16), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([128, 432, 28, 28], f16), T([288, 432, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([128, 288, 28, 28], f16), T([304, 288, 3, 3], f16), None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 2, ((T([128, 304, 14, 14], f16), T([304, 304, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 4, ((T([128, 304, 14, 14], f16), T([152, 304, 3, 3], f16), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 2, ((T([128, 152, 14, 14], f16), T([304, 152, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([128, 608, 14, 14], f16), T([304, 608, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([128, 304, 14, 14], f16), T([304, 304, 3, 3], f16), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([128, 912, 14, 14], f16), T([480, 912, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([128, 480, 14, 14], f16), T([960, 480, 3, 3], f16), None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([128, 960, 7, 7], f16), T([1024, 960, 3, 3], f16), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([128, 1024, 7, 7], f16), T([1280, 1024, 3, 3], f16), None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), {})
cnt: 1, ((T([128, 1280, 4, 4], f16), T([1024, 1280, 1, 1], f16), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {})
Operator: aten.convolution_backward.default
cnt: 1, ((T([128, 1024, 4, 4], f16), T([128, 1280, 4, 4], f16), T([1024, 1280, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([128, 1280, 4, 4], f16), T([128, 1024, 7, 7], f16), T([1280, 1024, 3, 3], f16), [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([128, 1024, 7, 7], f16), T([128, 960, 7, 7], f16), T([1024, 960, 3, 3], f16), [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([128, 960, 7, 7], f16), T([128, 480, 14, 14], f16), T([960, 480, 3, 3], f16), [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([128, 480, 14, 14], f16), T([128, 912, 14, 14], f16), T([480, 912, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 4, ((T([128, 152, 14, 14], f16), T([128, 304, 14, 14], f16), T([152, 304, 3, 3], f16), [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 2, ((T([128, 304, 14, 14], f16), T([128, 152, 14, 14], f16), T([304, 152, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 2, ((T([128, 304, 14, 14], f16), T([128, 304, 14, 14], f16), T([304, 304, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([128, 304, 14, 14], f16), T([128, 304, 14, 14], f16), T([304, 304, 3, 3], f16), [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([128, 304, 14, 14], f16), T([128, 608, 14, 14], f16), T([304, 608, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([128, 304, 14, 14], f16), T([128, 288, 28, 28], f16), T([304, 288, 3, 3], f16), [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([128, 288, 28, 28], f16), T([128, 432, 28, 28], f16), T([288, 432, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 4, ((T([128, 72, 28, 28], f16), T([128, 144, 28, 28], f16), T([72, 144, 3, 3], f16), [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 2, ((T([128, 144, 28, 28], f16), T([128, 72, 28, 28], f16), T([144, 72, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 2, ((T([128, 144, 28, 28], f16), T([128, 144, 28, 28], f16), T([144, 144, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([128, 144, 28, 28], f16), T([128, 144, 28, 28], f16), T([144, 144, 3, 3], f16), [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([128, 144, 28, 28], f16), T([128, 288, 28, 28], f16), T([144, 288, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([128, 144, 28, 28], f16), T([128, 128, 56, 56], f16), T([144, 128, 3, 3], f16), [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([128, 128, 56, 56], f16), T([128, 192, 56, 56], f16), T([128, 192, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 4, ((T([128, 32, 56, 56], f16), T([128, 64, 56, 56], f16), T([32, 64, 3, 3], f16), [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 2, ((T([128, 64, 56, 56], f16), T([128, 32, 56, 56], f16), T([64, 32, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 2, ((T([128, 64, 56, 56], f16), T([128, 64, 56, 56], f16), T([64, 64, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([128, 64, 56, 56], f16), T([128, 64, 56, 56], f16), T([64, 64, 3, 3], f16), [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([128, 64, 56, 56], f16), T([128, 128, 56, 56], f16), T([64, 128, 1, 1], f16), [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([128, 64, 56, 56], f16), T([128, 32, 112, 112], f16), T([64, 32, 3, 3], f16), [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), {})
cnt: 1, ((T([128, 32, 112, 112], f16), T([128, 3, 224, 224], f16), T([32, 3, 3, 3], f16), [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]), {})
Operator: aten.copy_.default
cnt: 1, ((T([128, 3, 224, 224], f16), T([128, 3, 224, 224], f16)), {})
Operator: aten.div.Scalar
cnt: 1, ((T([128, 1024, 4, 4], f16, stride=(1024, 1, 0, 0)), 16), {})
Operator: aten.lift_fresh_copy.default
cnt: 1, ((T([128], i64),), {})
Operator: aten.mean.dim
cnt: 1, ((T([128, 1024, 4, 4], f16), [-1, -2], True), {})
Operator: aten.mm.default
cnt: 1, ((T([128, 1000], f16), T([1000, 1024], f16)), {})
cnt: 1, ((T([1000, 128], f16, stride=(1, 1000)), T([128, 1024], f16)), {})
Operator: aten.native_batch_norm.default
cnt: 1, ((T([128, 32, 112, 112], f16), T([32], f16), T([32], f16), T([32], f16), T([32], f16), True, 0.1, 1e-05), {})
cnt: 7, ((T([128, 64, 56, 56], f16), T([64], f16), T([64], f16), T([64], f16), T([64], f16), True, 0.1, 1e-05), {})
cnt: 4, ((T([128, 32, 56, 56], f16), T([32], f16), T([32], f16), T([32], f16), T([32], f16), True, 0.1, 1e-05), {})
cnt: 1, ((T([128, 128, 56, 56], f16), T([128], f16), T([128], f16), T([128], f16), T([128], f16), True, 0.1, 1e-05), {})
cnt: 7, ((T([128, 144, 28, 28], f16), T([144], f16), T([144], f16), T([144], f16), T([144], f16), True, 0.1, 1e-05), {})
cnt: 4, ((T([128, 72, 28, 28], f16), T([72], f16), T([72], f16), T([72], f16), T([72], f16), True, 0.1, 1e-05), {})
cnt: 1, ((T([128, 288, 28, 28], f16), T([288], f16), T([288], f16), T([288], f16), T([288], f16), True, 0.1, 1e-05), {})
cnt: 7, ((T([128, 304, 14, 14], f16), T([304], f16), T([304], f16), T([304], f16), T([304], f16), True, 0.1, 1e-05), {})
cnt: 4, ((T([128, 152, 14, 14], f16), T([152], f16), T([152], f16), T([152], f16), T([152], f16), True, 0.1, 1e-05), {})
cnt: 1, ((T([128, 480, 14, 14], f16), T([480], f16), T([480], f16), T([480], f16), T([480], f16), True, 0.1, 1e-05), {})
cnt: 1, ((T([128, 960, 7, 7], f16), T([960], f16), T([960], f16), T([960], f16), T([960], f16), True, 0.1, 1e-05), {})
cnt: 1, ((T([128, 1024, 7, 7], f16), T([1024], f16), T([1024], f16), T([1024], f16), T([1024], f16), True, 0.1, 1e-05), {})
cnt: 1, ((T([128, 1280, 4, 4], f16), T([1280], f16), T([1280], f16), T([1280], f16), T([1280], f16), True, 0.1, 1e-05), {})
cnt: 1, ((T([128, 1024, 4, 4], f16), T([1024], f16), T([1024], f16), T([1024], f16), T([1024], f16), True, 0.1, 1e-05), {})
Operator: aten.native_batch_norm_backward.default
cnt: 1, ((T([128, 1024, 4, 4], f16), T([128, 1024, 4, 4], f16), T([1024], f16), T([1024], f16), T([1024], f16), T([1024], f32), T([1024], f32), True, 1e-05, [True, True, True]), {})
cnt: 1, ((T([128, 1280, 4, 4], f16), T([128, 1280, 4, 4], f16), T([1280], f16), T([1280], f16), T([1280], f16), T([1280], f32), T([1280], f32), True, 1e-05, [True, True, True]), {})
cnt: 1, ((T([128, 1024, 7, 7], f16), T([128, 1024, 7, 7], f16), T([1024], f16), T([1024], f16), T([1024], f16), T([1024], f32), T([1024], f32), True, 1e-05, [True, True, True]), {})
cnt: 1, ((T([128, 960, 7, 7], f16), T([128, 960, 7, 7], f16), T([960], f16), T([960], f16), T([960], f16), T([960], f32), T([960], f32), True, 1e-05, [True, True, True]), {})
cnt: 1, ((T([128, 480, 14, 14], f16), T([128, 480, 14, 14], f16), T([480], f16), T([480], f16), T([480], f16), T([480], f32), T([480], f32), True, 1e-05, [True, True, True]), {})
cnt: 4, ((T([128, 152, 14, 14], f16), T([128, 152, 14, 14], f16), T([152], f16), T([152], f16), T([152], f16), T([152], f32), T([152], f32), True, 1e-05, [True, True, True]), {})
cnt: 7, ((T([128, 304, 14, 14], f16), T([128, 304, 14, 14], f16), T([304], f16), T([304], f16), T([304], f16), T([304], f32), T([304], f32), True, 1e-05, [True, True, True]), {})
cnt: 1, ((T([128, 288, 28, 28], f16), T([128, 288, 28, 28], f16), T([288], f16), T([288], f16), T([288], f16), T([288], f32), T([288], f32), True, 1e-05, [True, True, True]), {})
cnt: 4, ((T([128, 72, 28, 28], f16), T([128, 72, 28, 28], f16), T([72], f16), T([72], f16), T([72], f16), T([72], f32), T([72], f32), True, 1e-05, [True, True, True]), {})
cnt: 7, ((T([128, 144, 28, 28], f16), T([128, 144, 28, 28], f16), T([144], f16), T([144], f16), T([144], f16), T([144], f32), T([144], f32), True, 1e-05, [True, True, True]), {})
cnt: 1, ((T([128, 128, 56, 56], f16), T([128, 128, 56, 56], f16), T([128], f16), T([128], f16), T([128], f16), T([128], f32), T([128], f32), True, 1e-05, [True, True, True]), {})
cnt: 4, ((T([128, 32, 56, 56], f16), T([128, 32, 56, 56], f16), T([32], f16), T([32], f16), T([32], f16), T([32], f32), T([32], f32), True, 1e-05, [True, True, True]), {})
cnt: 7, ((T([128, 64, 56, 56], f16), T([128, 64, 56, 56], f16), T([64], f16), T([64], f16), T([64], f16), T([64], f32), T([64], f32), True, 1e-05, [True, True, True]), {})
cnt: 1, ((T([128, 32, 112, 112], f16), T([128, 32, 112, 112], f16), T([32], f16), T([32], f16), T([32], f16), T([32], f32), T([32], f32), True, 1e-05, [True, True, True]), {})
Operator: aten.nll_loss_backward.default
cnt: 1, ((T([], f16), T([128, 1000], f16), T([128], i64), None, 1, -100, T([], f16)), {})
Operator: aten.nll_loss_forward.default
cnt: 1, ((T([128, 1000], f16), T([128], i64), None, 1, -100), {})
Operator: aten.relu_.default
cnt: 1, ((T([128, 32, 112, 112], f16),), {})
cnt: 7, ((T([128, 64, 56, 56], f16),), {})
cnt: 4, ((T([128, 32, 56, 56], f16),), {})
cnt: 1, ((T([128, 128, 56, 56], f16),), {})
cnt: 7, ((T([128, 144, 28, 28], f16),), {})
cnt: 4, ((T([128, 72, 28, 28], f16),), {})
cnt: 1, ((T([128, 288, 28, 28], f16),), {})
cnt: 7, ((T([128, 304, 14, 14], f16),), {})
cnt: 4, ((T([128, 152, 14, 14], f16),), {})
cnt: 1, ((T([128, 480, 14, 14], f16),), {})
cnt: 1, ((T([128, 960, 7, 7], f16),), {})
cnt: 1, ((T([128, 1024, 7, 7], f16),), {})
cnt: 1, ((T([128, 1280, 4, 4], f16),), {})
cnt: 1, ((T([128, 1024, 4, 4], f16),), {})
Operator: aten.sum.SymInt
cnt: 1, ((T([128, 1000], f16), [0], True), {})
Operator: aten.threshold_backward.default
cnt: 1, ((T([128, 1024, 4, 4], f16), T([128, 1024, 4, 4], f16), 0), {})
cnt: 1, ((T([128, 1280, 4, 4], f16), T([128, 1280, 4, 4], f16), 0), {})
cnt: 1, ((T([128, 1024, 7, 7], f16), T([128, 1024, 7, 7], f16), 0), {})
cnt: 1, ((T([128, 960, 7, 7], f16), T([128, 960, 7, 7], f16), 0), {})
cnt: 1, ((T([128, 480, 14, 14], f16), T([128, 480, 14, 14], f16), 0), {})
cnt: 1, ((T([128, 152, 14, 14], f16, stride=(178752, 196, 14, 1)), T([128, 152, 14, 14], f16), 0), {})
cnt: 7, ((T([128, 304, 14, 14], f16), T([128, 304, 14, 14], f16), 0), {})
cnt: 2, ((T([128, 152, 14, 14], f16), T([128, 152, 14, 14], f16), 0), {})
cnt: 1, ((T([128, 152, 14, 14], f16, stride=(119168, 196, 14, 1)), T([128, 152, 14, 14], f16), 0), {})
cnt: 1, ((T([128, 288, 28, 28], f16), T([128, 288, 28, 28], f16), 0), {})
cnt: 1, ((T([128, 72, 28, 28], f16, stride=(338688, 784, 28, 1)), T([128, 72, 28, 28], f16), 0), {})
cnt: 7, ((T([128, 144, 28, 28], f16), T([128, 144, 28, 28], f16), 0), {})
cnt: 2, ((T([128, 72, 28, 28], f16), T([128, 72, 28, 28], f16), 0), {})
cnt: 1, ((T([128, 72, 28, 28], f16, stride=(225792, 784, 28, 1)), T([128, 72, 28, 28], f16), 0), {})
cnt: 1, ((T([128, 128, 56, 56], f16), T([128, 128, 56, 56], f16), 0), {})
cnt: 1, ((T([128, 32, 56, 56], f16, stride=(602112, 3136, 56, 1)), T([128, 32, 56, 56], f16), 0), {})
cnt: 7, ((T([128, 64, 56, 56], f16), T([128, 64, 56, 56], f16), 0), {})
cnt: 2, ((T([128, 32, 56, 56], f16), T([128, 32, 56, 56], f16), 0), {})
cnt: 1, ((T([128, 32, 56, 56], f16, stride=(401408, 3136, 56, 1)), T([128, 32, 56, 56], f16), 0), {})
cnt: 1, ((T([128, 32, 112, 112], f16), T([128, 32, 112, 112], f16), 0), {})

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

- **File Documentation**: `selecsls42b_training.txt_docs.md`
- **Keyword Index**: `selecsls42b_training.txt_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
