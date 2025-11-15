# Documentation: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/hf_BigBird_training.txt`

## File Metadata

- **Path**: `benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/hf_BigBird_training.txt`
- **Size**: 16,654 bytes (16.26 KB)
- **Type**: Source File (.txt)
- **Extension**: `.txt`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```
Operator: aten._softmax.default
cnt: 24, ((T([2, 12, 64, 1024], f16), -1, False), {})
cnt: 24, ((T([2, 12, 64, 448], f16), -1, False), {})
cnt: 12, ((T([2, 12, 12, 64, 512], f16), -1, False), {})
Operator: aten._softmax_backward_data.default
cnt: 24, ((T([2, 12, 64, 1024], f16), T([2, 12, 64, 1024], f16), -1, f16), {})
cnt: 24, ((T([2, 12, 64, 448], f16), T([2, 12, 64, 448], f16), -1, f16), {})
cnt: 12, ((T([2, 12, 12, 64, 512], f16), T([2, 12, 12, 64, 512], f16), -1, f16), {})
Operator: aten._to_copy.default
cnt: 12, ((T([2, 1, 12, 64, 192], f32),), {'dtype': f16})
cnt: 12, ((T([2, 1, 1024, 1], f32),), {'dtype': f16})
cnt: 12, ((T([2, 1, 1, 1024], f32),), {'dtype': f16})
cnt: 12, ((T([12, 14, 3], i32),), {'dtype': i64, 'device': 'cuda'})
Operator: aten._unsafe_view.default
cnt: 24, ((T([2, 12, 16, 64, 64], f16), [384, 64, 64]), {})
cnt: 96, ((T([2, 12, 64, 64], f16), [24, 64, 64]), {})
cnt: 48, ((T([2, 12, 1024, 64], f16), [24, 1024, 64]), {})
cnt: 24, ((T([2, 12, 12, 64, 64], f16), [288, 64, 64]), {})
cnt: 24, ((T([2, 12, 12, 192, 64], f16), [288, 192, 64]), {})
cnt: 24, ((T([2, 12, 12, 64, 64, 1], f16), [24, 768, 64]), {})
cnt: 48, ((T([2, 12, 64, 64, 1, 1], f16), [24, 64, 64]), {})
cnt: 24, ((T([2, 1024, 12, 64], f16), [2, 1024, 768]), {})
cnt: 12, ((T([2, 1024, 768], f16), [2048, 768]), {})
Operator: aten.add.Tensor
cnt: 76, ((T([2, 1024, 768], f16), T([2, 1024, 768], f16)), {})
cnt: 24, ((T([1008], i64), T([1008], i64)), {})
cnt: 36, ((T([2, 1024, 3072], f16), T([2, 1024, 3072], f16)), {})
cnt: 12, ((T([2, 1024, 3072], f16), 1.0), {})
cnt: 1, ((T([2, 1024, 768], f16), 1.0), {})
cnt: 360, ((T([2, 12, 16, 64, 64], f16), T([2, 12, 16, 64, 64], f16)), {})
cnt: 36, ((T([2, 12, 12, 64, 512], f16), T([2, 12, 12, 64, 512], f16)), {})
cnt: 48, ((T([2, 12, 14, 192, 64], f16), T([2, 12, 14, 192, 64], f16)), {})
cnt: 36, ((T([2, 12, 12, 64, 64], f16), T([2, 12, 12, 64, 64], f16)), {})
cnt: 24, ((T([2, 12, 1024, 64], f16), T([2, 12, 1024, 64], f16)), {})
cnt: 12, ((T([2, 12, 1024, 64], f16, stride=(786432, 65536, 1, 1024)), T([2, 12, 1024, 64], f16, stride=(786432, 65536, 1, 1024))), {})
cnt: 12, ((T([2, 12, 1024, 64], f16, stride=(786432, 65536, 1, 1024)), T([2, 12, 1024, 64], f16)), {})
cnt: 1, ((T([50358, 768], f16), T([50358, 768], f16)), {})
Operator: aten.add_.Tensor
cnt: 1, ((T([2, 1024, 768], f16), T([1, 1024, 768], f16)), {})
cnt: 24, ((T([2, 12, 64, 1024], f16), T([2, 1, 1, 1024], f16)), {})
cnt: 24, ((T([2, 12, 64, 448], f16), T([2, 12, 64, 448], f32)), {})
cnt: 12, ((T([2, 12, 12, 64, 192], f16), T([2, 1, 12, 64, 192], f16)), {})
cnt: 24, ((T([2, 12, 12, 64, 64], f16), T([2, 1, 1, 1, 64], f16)), {})
cnt: 12, ((T([2, 12, 12, 64, 192], f16), T([2, 12, 12, 64, 192], f32)), {})
cnt: 36, ((T([2, 12, 12, 64, 64], f16), T([2, 12, 12, 64, 64], f16)), {})
Operator: aten.addmm.default
cnt: 49, ((T([768], f16), T([2048, 768], f16), T([768, 768], f16, stride=(1, 768))), {})
cnt: 12, ((T([3072], f16), T([2048, 768], f16), T([768, 3072], f16, stride=(1, 768))), {})
cnt: 12, ((T([768], f16), T([2048, 3072], f16), T([3072, 768], f16, stride=(1, 3072))), {})
cnt: 1, ((T([768], f16), T([2, 768], f16, stride=(786432, 1)), T([768, 768], f16, stride=(1, 768))), {})
cnt: 1, ((T([50358], f16), T([2048, 768], f16), T([768, 50358], f16, stride=(1, 768))), {})
Operator: aten.bmm.default
cnt: 48, ((T([24, 64, 64], f16), T([24, 64, 1024], f16, stride=(65536, 1, 64))), {})
cnt: 48, ((T([24, 64, 1024], f16), T([24, 1024, 64], f16)), {})
cnt: 48, ((T([24, 64, 64], f16), T([24, 64, 448], f16, stride=(28672, 1, 64))), {})
cnt: 48, ((T([24, 64, 448], f16), T([24, 448, 64], f16)), {})
cnt: 48, ((T([288, 64, 64], f16), T([288, 64, 192], f16, stride=(12288, 1, 64))), {})
cnt: 24, ((T([24, 768, 64], f16), T([24, 64, 64], f16)), {})
cnt: 24, ((T([288, 64, 192], f16, stride=(32768, 512, 1)), T([288, 192, 64], f16)), {})
cnt: 24, ((T([24, 768, 64], f16, stride=(393216, 512, 1)), T([24, 64, 64], f16)), {})
cnt: 24, ((T([24, 1024, 64], f16, stride=(65536, 1, 1024)), T([24, 64, 64], f16)), {})
cnt: 24, ((T([24, 64, 64], f16, stride=(4096, 1, 64)), T([24, 64, 1024], f16)), {})
cnt: 24, ((T([24, 448, 64], f16, stride=(28672, 1, 448)), T([24, 64, 64], f16)), {})
cnt: 24, ((T([24, 64, 64], f16, stride=(4096, 1, 64)), T([24, 64, 448], f16)), {})
cnt: 24, ((T([24, 64, 768], f16, stride=(393216, 1, 512)), T([24, 768, 64], f16)), {})
cnt: 48, ((T([24, 768, 64], f16), T([24, 64, 64], f16, stride=(4096, 1, 64))), {})
cnt: 24, ((T([288, 192, 64], f16, stride=(32768, 1, 512)), T([288, 64, 64], f16)), {})
cnt: 24, ((T([24, 64, 768], f16, stride=(49152, 1, 64)), T([24, 768, 64], f16)), {})
cnt: 24, ((T([288, 64, 64], f16, stride=(4096, 1, 64)), T([288, 64, 192], f16)), {})
cnt: 24, ((T([288, 64, 192], f16), T([288, 192, 64], f16)), {})
Operator: aten.cat.default
cnt: 1, (([T([2, 12, 64], f32, stride=(1024, 64, 1)), T([2, 12, 64], f32, stride=(1024, 64, 1)), T([2, 12, 64], f32, stride=(1024, 64, 1))], 2), {})
cnt: 12, (([T([1, 12, 14, 3], i64), T([1, 12, 14, 3], i64)],), {})
cnt: 48, (([T([2, 12, 64, 64], f16, stride=(786432, 64, 768, 1)), T([2, 12, 64, 64], f16, stride=(786432, 64, 768, 1)), T([2, 12, 64, 64], f16, stride=(786432, 64, 768, 1)), T([2, 12, 64, 64], f16, stride=(786432, 64, 768, 1)), T([2, 12, 192, 64], f16, stride=(2064384, 172032, 64, 1))], 2), {})
cnt: 12, (([T([2, 1, 1, 192], f16, stride=(1024, 1024, 1024, 1)), T([2, 1, 1, 64], f16, stride=(1024, 1024, 1024, 1)), T([2, 1, 1, 192], f16)], 3), {})
cnt: 24, (([T([2, 12, 64, 256], f32), T([2, 12, 64, 192], f32, stride=(2064384, 172032, 192, 1))], 3), {})
cnt: 24, (([T([2, 12, 12, 64, 64], f16, stride=(786432, 64, 49152, 768, 1)), T([2, 12, 12, 64, 64], f16, stride=(786432, 64, 49152, 768, 1)), T([2, 12, 12, 64, 64], f16, stride=(786432, 64, 49152, 768, 1))], 3), {})
cnt: 12, (([T([2, 12, 12, 64, 64], f16), T([2, 12, 12, 64, 192], f16), T([2, 12, 12, 64, 192], f16), T([2, 12, 12, 64, 64], f16)], -1), {})
cnt: 12, (([T([2, 1, 1, 64], f16, stride=(1024, 1024, 1024, 1)), T([2, 1, 1, 192], f16, stride=(1024, 1024, 1024, 1)), T([2, 1, 1, 192], f16)], 3), {})
cnt: 12, (([T([2, 12, 1, 64, 64], f16), T([2, 12, 1, 64, 64], f16), T([2, 12, 12, 64, 64], f16), T([2, 12, 1, 64, 64], f16), T([2, 12, 1, 64, 64], f16)], 2), {})
Operator: aten.clone.default
cnt: 1, ((T([2, 1024], i64),), {})
Operator: aten.copy_.default
cnt: 1, ((T([2, 1024], i64), T([2, 1024], i64)), {})
cnt: 12, ((T([2, 12, 12, 64, 64], f16), T([2, 12, 12, 64, 64], f16, stride=(786432, 64, 49152, 768, 1))), {})
cnt: 36, ((T([288, 64, 64], f16), T([288, 64, 64], f16)), {})
cnt: 36, ((T([2, 12, 12, 64, 64], f16), T([2, 12, 12, 64, 64], f16)), {})
Operator: aten.div.Tensor
cnt: 2, ((T([], f16), 103133184), {})
Operator: aten.embedding.default
cnt: 1, ((T([50358, 768], f16), T([2, 1024], i64), 0), {})
cnt: 1, ((T([2, 768], f16), T([2, 1024], i64, stride=(0, 1))), {})
cnt: 1, ((T([4096, 768], f16), T([1, 1024], i64)), {})
Operator: aten.embedding_dense_backward.default
cnt: 1, ((T([1, 1024, 768], f16), T([1, 1024], i64), 4096, -1, False), {})
cnt: 1, ((T([2, 1024, 768], f16), T([2, 1024], i64, stride=(0, 1)), 2, -1, False), {})
cnt: 1, ((T([2, 1024, 768], f16), T([2, 1024], i64), 50358, 0, False), {})
Operator: aten.floor_divide.default
cnt: 24, ((T([1008], i64), 42), {})
Operator: aten.index.Tensor
cnt: 24, ((T([16, 64], f32), [T([504], i64)]), {})
Operator: aten.index_add.default
cnt: 24, ((T([384, 64, 64], f16), 0, T([1008], i64), T([1008, 64, 64], f16)), {})
Operator: aten.index_select.default
cnt: 24, ((T([384, 64, 64], f16), 0, T([1008], i64)), {})
Operator: aten.minimum.default
cnt: 24, ((T([2, 1, 1, 448], f16), T([2, 12, 64, 448], f32)), {})
Operator: aten.mm.default
cnt: 1, ((T([2048, 50358], f16, stride=(0, 0)), T([50358, 768], f16)), {})
cnt: 1, ((T([50358, 2048], f16, stride=(0, 0)), T([2048, 768], f16)), {})
cnt: 49, ((T([2048, 768], f16), T([768, 768], f16)), {})
cnt: 49, ((T([768, 2048], f16, stride=(1, 768)), T([2048, 768], f16)), {})
cnt: 12, ((T([2048, 768], f16), T([768, 3072], f16)), {})
cnt: 12, ((T([768, 2048], f16, stride=(1, 768)), T([2048, 3072], f16)), {})
cnt: 12, ((T([2048, 3072], f16), T([3072, 768], f16)), {})
cnt: 12, ((T([3072, 2048], f16, stride=(1, 3072)), T([2048, 768], f16)), {})
Operator: aten.mul.Scalar
cnt: 1, ((T([2, 1024, 768], f16), 3.0), {})
cnt: 12, ((T([2, 1024, 3072], f16), 3.0), {})
Operator: aten.mul.Tensor
cnt: 1, ((T([2, 12, 64, 1], f32, stride=(1024, 64, 1, 1)), T([2, 12, 1, 192], f32)), {})
cnt: 12, ((T([2, 1, 14, 64, 1], f32, stride=(1024, 1, 64, 1, 1)), T([2, 12, 14, 1, 192], f32)), {})
cnt: 24, ((T([1008], i64), 16), {})
cnt: 48, ((T([2, 12, 64, 1024], f16), 0.125), {})
cnt: 24, ((T([2, 1, 1, 1024], f16), -10000.0), {})
cnt: 48, ((T([2, 12, 64, 448], f16), 0.125), {})
cnt: 24, ((T([2, 12, 64, 448], f32), -10000.0), {})
cnt: 24, ((T([2, 12, 12, 64, 192], f16), 0.125), {})
cnt: 24, ((T([2, 12, 12, 64, 64], f16), 0.125), {})
cnt: 12, ((T([2, 1, 12, 64, 192], f16), -10000.0), {})
cnt: 24, ((T([2, 1, 1, 1, 64], f16), -10000.0), {})
cnt: 12, ((T([2, 12, 12, 64, 192], f32), -10000.0), {})
cnt: 12, ((T([2, 12, 1024, 64], f16), T([2, 1, 1024, 1], f16)), {})
cnt: 24, ((T([2, 1024, 3072], f16), 0.5), {})
cnt: 24, ((T([2, 1024, 3072], f16), 0.044715), {})
cnt: 24, ((T([2, 1024, 3072], f16), 0.7978845608028654), {})
cnt: 48, ((T([2, 1024, 3072], f16), T([2, 1024, 3072], f16)), {})
cnt: 2, ((T([2, 1024, 768], f16), 0.5), {})
cnt: 2, ((T([2, 1024, 768], f16), 0.044715), {})
cnt: 2, ((T([2, 1024, 768], f16), 0.7978845608028654), {})
cnt: 4, ((T([2, 1024, 768], f16), T([2, 1024, 768], f16)), {})
cnt: 12, ((T([2, 12, 1024, 64], f16, stride=(786432, 64, 768, 1)), T([2, 1, 1024, 1], f16)), {})
cnt: 24, ((T([2, 12, 12, 64, 64], f16, stride=(4718592, 393216, 32768, 512, 1)), 0.125), {})
cnt: 24, ((T([2, 12, 12, 64, 192], f16, stride=(4718592, 393216, 32768, 512, 1)), 0.125), {})
Operator: aten.native_layer_norm.default
cnt: 26, ((T([2, 1024, 768], f16), [768], T([768], f16), T([768], f16), 1e-12), {})
Operator: aten.native_layer_norm_backward.default
cnt: 26, ((T([2, 1024, 768], f16), T([2, 1024, 768], f16), [768], T([2, 1024, 1], f32), T([2, 1024, 1], f32), T([768], f16), T([768], f16), [True, True, True]), {})
Operator: aten.new_empty_strided.default
cnt: 36, ((T([288, 64, 64], f16), [288, 64, 64], [4096, 64, 1]), {})
Operator: aten.new_ones.default
cnt: 24, ((T([2, 1, 1, 1024], f16), [2, 1, 1, 192]), {'dtype': f16, 'layout': torch.strided, 'device': 'cuda', 'pin_memory': False})
cnt: 24, ((T([2, 12, 14, 64, 192], f32), [2, 12, 64, 256]), {'dtype': f32, 'layout': torch.strided, 'device': 'cuda', 'pin_memory': False})
Operator: aten.new_zeros.default
cnt: 12, ((T([2, 12, 12, 64, 64], f16, stride=(786432, 64, 49152, 768, 1)), [1179648]), {})
cnt: 24, ((T([1008, 64, 64], f16), [384, 64, 64]), {'dtype': f16, 'layout': torch.strided, 'device': 'cuda'})
Operator: aten.pow.Tensor_Scalar
cnt: 12, ((T([2, 1024, 3072], f16), 3.0), {})
cnt: 1, ((T([2, 1024, 768], f16), 3.0), {})
cnt: 1, ((T([2, 1024, 768], f16), 2.0), {})
cnt: 12, ((T([2, 1024, 3072], f16), 2.0), {})
Operator: aten.rsub.Scalar
cnt: 24, ((T([2, 1, 1, 1024], f16), 1.0), {})
cnt: 24, ((T([2, 12, 64, 448], f32), 1.0), {})
cnt: 12, ((T([2, 1, 12, 64, 192], f16), 1.0), {})
cnt: 24, ((T([2, 1, 1, 1, 64], f16, stride=(1024, 1024, 1024, 64, 1)), 1.0), {})
cnt: 12, ((T([2, 12, 12, 64, 192], f32, stride=(2064384, 172032, 12288, 192, 1)), 1.0), {})
Operator: aten.select_backward.default
cnt: 24, ((T([2, 12, 64, 64], f16), [2, 12, 16, 64, 64], 2, -1), {})
cnt: 12, ((T([2, 12, 64, 64], f16), [2, 12, 16, 64, 64], 2, -2), {})
cnt: 12, ((T([2, 12, 192, 64], f16, stride=(344064, 28672, 64, 1)), [2, 12, 14, 192, 64], 2, -1), {})
cnt: 24, ((T([2, 12, 64, 64], f16, stride=(344064, 28672, 64, 1)), [2, 12, 16, 64, 64], 2, -1), {})
cnt: 12, ((T([2, 12, 64, 64], f16, stride=(344064, 28672, 64, 1)), [2, 12, 16, 64, 64], 2, -2), {})
cnt: 12, ((T([2, 12, 64, 64], f16, stride=(344064, 28672, 64, 1)), [2, 12, 16, 64, 64], 2, -3), {})
cnt: 24, ((T([2, 12, 64, 64], f16, stride=(344064, 28672, 64, 1)), [2, 12, 16, 64, 64], 2, 0), {})
cnt: 12, ((T([2, 12, 192, 64], f16, stride=(344064, 28672, 1, 448)), [2, 12, 14, 192, 64], 2, -1), {})
cnt: 24, ((T([2, 12, 64, 64], f16, stride=(344064, 28672, 1, 448)), [2, 12, 16, 64, 64], 2, -1), {})
cnt: 12, ((T([2, 12, 64, 64], f16, stride=(344064, 28672, 1, 448)), [2, 12, 16, 64, 64], 2, -2), {})
cnt: 12, ((T([2, 12, 64, 64], f16, stride=(344064, 28672, 1, 448)), [2, 12, 16, 64, 64], 2, -3), {})
cnt: 24, ((T([2, 12, 64, 64], f16, stride=(344064, 28672, 1, 448)), [2, 12, 16, 64, 64], 2, 0), {})
cnt: 24, ((T([2, 12, 64, 64], f16), [2, 12, 16, 64, 64], 2, 0), {})
cnt: 12, ((T([2, 12, 64, 64], f16, stride=(49152, 4096, 1, 64)), [2, 12, 16, 64, 64], 2, -1), {})
cnt: 12, ((T([2, 12, 64, 64], f16, stride=(49152, 4096, 1, 64)), [2, 12, 16, 64, 64], 2, 0), {})
cnt: 12, ((T([2, 12, 64, 64], f16), [2, 12, 16, 64, 64], 2, 1), {})
cnt: 12, ((T([2, 12, 192, 64], f16, stride=(344064, 28672, 64, 1)), [2, 12, 14, 192, 64], 2, 0), {})
cnt: 12, ((T([2, 12, 64, 64], f16, stride=(344064, 28672, 64, 1)), [2, 12, 16, 64, 64], 2, 2), {})
cnt: 12, ((T([2, 12, 64, 64], f16, stride=(344064, 28672, 64, 1)), [2, 12, 16, 64, 64], 2, 1), {})
cnt: 12, ((T([2, 12, 192, 64], f16, stride=(344064, 28672, 1, 448)), [2, 12, 14, 192, 64], 2, 0), {})
cnt: 12, ((T([2, 12, 64, 64], f16, stride=(344064, 28672, 1, 448)), [2, 12, 16, 64, 64], 2, 2), {})
cnt: 12, ((T([2, 12, 64, 64], f16, stride=(344064, 28672, 1, 448)), [2, 12, 16, 64, 64], 2, 1), {})
Operator: aten.slice_backward.default
cnt: 372, ((T([2, 12, 16, 64, 64], f16), [2, 12, 16, 64, 64], 1, 0, 9223372036854775807, 1), {})
cnt: 372, ((T([2, 12, 16, 64, 64], f16), [2, 12, 16, 64, 64], 0, 0, 9223372036854775807, 1), {})
cnt: 72, ((T([2, 12, 14, 192, 64], f16), [2, 12, 14, 192, 64], 1, 0, 9223372036854775807, 1), {})
cnt: 72, ((T([2, 12, 14, 192, 64], f16), [2, 12, 14, 192, 64], 0, 0, 9223372036854775807, 1), {})
cnt: 12, ((T([2, 12, 12, 64, 64], f16), [2, 12, 12, 64, 512], 4, -64, 9223372036854775807, 1), {})
cnt: 48, ((T([2, 12, 12, 64, 512], f16), [2, 12, 12, 64, 512], 3, 0, 9223372036854775807, 1), {})
cnt: 48, ((T([2, 12, 12, 64, 512], f16), [2, 12, 12, 64, 512], 2, 0, 9223372036854775807, 1), {})
cnt: 48, ((T([2, 12, 12, 64, 512], f16), [2, 12, 12, 64, 512], 1, 0, 9223372036854775807, 1), {})
cnt: 48, ((T([2, 12, 12, 64, 512], f16), [2, 12, 12, 64, 512], 0, 0, 9223372036854775807, 1), {})
cnt: 12, ((T([2, 12, 12, 64, 64], f16), [2, 12, 12, 64, 512], 4, 0, 64, 1), {})
cnt: 12, ((T([2, 12, 12, 192, 64], f16), [2, 12, 14, 192, 64], 2, 1, -1, 1), {})
cnt: 12, ((T([2, 12, 12, 64, 192], f16), [2, 12, 12, 64, 512], 4, 256, -64, 1), {})
cnt: 12, ((T([2, 12, 12, 64, 192], f16), [2, 12, 12, 64, 512], 4, 64, 256, 1), {})
cnt: 12, ((T([2, 12, 12, 192, 64], f16, stride=(1769472, 147456, 12288, 1, 192)), [2, 12, 14, 192, 64], 2, 1, -1, 1), {})
cnt: 12, ((T([2, 12, 12, 64, 64], f16), [2, 12, 16, 64, 64], 2, 2, -2, 1), {})
cnt: 12, ((T([2, 12, 12, 64, 64], f16, stride=(1769472, 147456, 12288, 64, 1)), [2, 12, 16, 64, 64], 2, 3, -1, 1), {})
cnt: 12, ((T([2, 12, 12, 64, 64], f16, stride=(1769472, 147456, 12288, 64, 1)), [2, 12, 16, 64, 64], 2, 2, -2, 1), {})
cnt: 12, ((T([2, 12, 12, 64, 64], f16, stride=(1769472, 147456, 12288, 64, 1)), [2, 12, 16, 64, 64], 2, 1, -3, 1), {})
cnt: 12, ((T([2, 12, 12, 64, 64], f16, stride=(1769472, 147456, 12288, 1, 192)), [2, 12, 16, 64, 64], 2, 3, -1, 1), {})
cnt: 12, ((T([2, 12, 12, 64, 64], f16, stride=(1769472, 147456, 12288, 1, 192)), [2, 12, 16, 64, 64], 2, 2, -2, 1), {})
cnt: 12, ((T([2, 12, 12, 64, 64], f16, stride=(1769472, 147456, 12288, 1, 192)), [2, 12, 16, 64, 64], 2, 1, -3, 1), {})
Operator: aten.stack.default
cnt: 12, (([T([504, 64], f32), T([504, 64], f32)],), {})
Operator: aten.sum.SymInt
cnt: 1, ((T([2048, 50358], f16, stride=(0, 0)), [0], True), {})
cnt: 61, ((T([2048, 768], f16), [0], True), {})
cnt: 12, ((T([2048, 3072], f16), [0], True), {})
cnt: 1, ((T([2, 1024, 768], f16), [0], True), {})
Operator: aten.sum.default
cnt: 1, ((T([2, 1024, 50358], f16),), {})
Operator: aten.tanh.default
cnt: 12, ((T([2, 1024, 3072], f16),), {})
cnt: 1, ((T([2, 768], f16),), {})
cnt: 1, ((T([2, 1024, 768], f16),), {})
Operator: aten.tanh_backward.default
cnt: 1, ((T([2, 1024, 768], f16), T([2, 1024, 768], f16)), {})
cnt: 12, ((T([2, 1024, 3072], f16), T([2, 1024, 3072], f16)), {})
Operator: aten.unbind.int
cnt: 12, ((T([2, 16, 64], f32),), {})
cnt: 12, ((T([2, 12, 14, 3], i64),), {})
Operator: aten.unsqueeze_.default
cnt: 1, ((T([2, 12, 64, 192], f32), 1), {})
cnt: 12, ((T([12, 14, 3], i64), 0), {})
cnt: 48, ((T([2, 12, 64, 64], f16), 2), {})

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

- **File Documentation**: `hf_BigBird_training.txt_docs.md`
- **Keyword Index**: `hf_BigBird_training.txt_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
