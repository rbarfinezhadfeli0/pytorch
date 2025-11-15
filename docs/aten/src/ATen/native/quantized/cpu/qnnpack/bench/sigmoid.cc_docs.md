# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/bench/sigmoid.cc`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/bench/sigmoid.cc`
- **Size**: 2,960 bytes (2.89 KB)
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

static void sigmoid_q8(benchmark::State& state) {
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

  pytorch_qnnp_operator_t sigmoidOperator = nullptr;
  status = pytorch_qnnp_create_sigmoid_nc_q8(
      channels,
      127 /* input zero point */,
      1.0f /* input scale */,
      0 /* output zero point */,
      1.0f / 256.0f /* output scale */,
      0 /* output min */,
      255 /* output max */,
      0 /* flags */,
      &sigmoidOperator);
  if (status != pytorch_qnnp_status_success || sigmoidOperator == nullptr) {
    state.SkipWithError("failed to create Sigmoid operator");
  }

  status = pytorch_qnnp_setup_sigmoid_nc_q8(
      sigmoidOperator,
      batchSize,
      input.data(),
      channels /* input:stride */,
      output.data(),
      channels /* output:stride */);
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to setup Sigmoid operator");
  }

  for (auto _ : state) {
    status =
        pytorch_qnnp_run_operator(sigmoidOperator, nullptr /* thread pool */);
    if (status != pytorch_qnnp_status_success) {
      state.SkipWithError("failed to run Sigmoid operator");
    }
  }

  const size_t itemsPerIteration = batchSize * channels;
  state.SetItemsProcessed(
      int64_t(state.iterations()) * int64_t(itemsPerIteration));

  const size_t bytesPerIteration = 2 * itemsPerIteration * sizeof(uint8_t);
  state.SetBytesProcessed(
      int64_t(state.iterations()) * int64_t(bytesPerIteration));

  status = pytorch_qnnp_delete_operator(sigmoidOperator);
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to delete Sigmoid operator");
  }
}

static void CharacteristicArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "C"});

  int32_t c = 16;
  for (int32_t n = 224; n >= 7; n /= 2) {
    b->Args({n * n, c});
    c *= 2;
  }
}

BENCHMARK(sigmoid_q8)->Apply(CharacteristicArguments);

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
- [`softargmax.cc_docs.md`](./softargmax.cc_docs.md)
- [`global-average-pooling.cc_docs.md`](./global-average-pooling.cc_docs.md)
- [`q8gemm_sparse.cc_docs.md`](./q8gemm_sparse.cc_docs.md)
- [`requantization.cc_docs.md`](./requantization.cc_docs.md)
- [`max-pooling.cc_docs.md`](./max-pooling.cc_docs.md)
- [`hgemm.cc_docs.md`](./hgemm.cc_docs.md)
- [`add.cc_docs.md`](./add.cc_docs.md)
- [`q8gemm.cc_docs.md`](./q8gemm.cc_docs.md)
- [`convolution.cc_docs.md`](./convolution.cc_docs.md)


## Cross-References

- **File Documentation**: `sigmoid.cc_docs.md`
- **Keyword Index**: `sigmoid.cc_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
