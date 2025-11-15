# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/bench/softargmax.cc`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/bench/softargmax.cc`
- **Size**: 3,020 bytes (2.95 KB)
- **Type**: C++ Source Code
- **Extension**: `.cc`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <pytorch_qnnpack.h>

#include <benchmark/benchmark.h>

static void softargmax_q8(benchmark::State& state) {
  const size_t batchSize = static_cast<size_t>(state.range(0));
  const size_t channels = static_cast<size_t>(state.range(1));

  std::random_device randomDevice;
  auto rng = std::mt19937(randomDevice());
  auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

  std::vector<uint8_t> input(batchSize * channels);
  std::vector<uint8_t> output(batchSize * channels);
  std::generate(input.begin(), input.end(), std::ref(u8rng));
  std::fill(output.begin(), output.end(), 0xA5);

  pytorch_qnnp_status status = pytorch_qnnp_initialize();
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to initialize QNNPACK");
  }

  pytorch_qnnp_operator_t softArgMaxOperator = nullptr;
  status = pytorch_qnnp_create_softargmax_nc_q8(
      channels,
      1.0f /* input scale */,
      0 /* output zero point */,
      1.0f / 256.0f /* output scale */,
      0 /* flags */,
      &softArgMaxOperator);
  if (status != pytorch_qnnp_status_success || softArgMaxOperator == nullptr) {
    state.SkipWithError("failed to create SoftArgMax operator");
  }

  status = pytorch_qnnp_setup_softargmax_nc_q8(
      softArgMaxOperator,
      batchSize,
      input.data(),
      channels /* input:stride */,
      output.data(),
      channels /* output:stride */);
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to setup SoftArgMax operator");
  }

  for (auto _ : state) {
    status = pytorch_qnnp_run_operator(
        softArgMaxOperator, nullptr /* thread pool */);
    if (status != pytorch_qnnp_status_success) {
      state.SkipWithError("failed to run SoftArgMax operator");
    }
  }

  const size_t itemsPerIteration = batchSize * channels;
  state.SetItemsProcessed(
      int64_t(state.iterations()) * int64_t(itemsPerIteration));

  const size_t bytesPerIteration = 2 * itemsPerIteration * sizeof(uint8_t);
  state.SetBytesProcessed(
      int64_t(state.iterations()) * int64_t(bytesPerIteration));

  status = pytorch_qnnp_delete_operator(softArgMaxOperator);
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to delete SoftArgMax operator");
  }
}

static void CharacteristicArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "C"});

  /* CIFAR-10 */
  b->Args({1, 10});
  /* CIFAR-100 */
  b->Args({1, 100});
  /* ImageNet-1K */
  b->Args({1, 1000});
  /* ImageNet-1K+1 */
  b->Args({1, 1001});
  /* ImageNet-22K */
  b->Args({1, 21841});
}

BENCHMARK(softargmax_q8)->Apply(CharacteristicArguments);

#ifndef PYTORCH_QNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu/qnnpack/bench`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `algorithm`
- `cmath`
- `functional`
- `random`
- `vector`
- `pytorch_qnnpack.h`
- `benchmark/benchmark.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`aten/src/ATen/native/quantized/cpu/qnnpack/bench`):

- [`tanh.cc_docs.md`](./tanh.cc_docs.md)
- [`global-average-pooling.cc_docs.md`](./global-average-pooling.cc_docs.md)
- [`q8gemm_sparse.cc_docs.md`](./q8gemm_sparse.cc_docs.md)
- [`requantization.cc_docs.md`](./requantization.cc_docs.md)
- [`max-pooling.cc_docs.md`](./max-pooling.cc_docs.md)
- [`hgemm.cc_docs.md`](./hgemm.cc_docs.md)
- [`add.cc_docs.md`](./add.cc_docs.md)
- [`q8gemm.cc_docs.md`](./q8gemm.cc_docs.md)
- [`convolution.cc_docs.md`](./convolution.cc_docs.md)


## Cross-References

- **File Documentation**: `softargmax.cc_docs.md`
- **Keyword Index**: `softargmax.cc_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
