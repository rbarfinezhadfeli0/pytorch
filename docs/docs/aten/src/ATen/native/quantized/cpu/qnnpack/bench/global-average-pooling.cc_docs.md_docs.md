# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/bench/global-average-pooling.cc_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/bench/global-average-pooling.cc_docs.md`
- **Size**: 5,544 bytes (5.41 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/bench/global-average-pooling.cc`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/bench/global-average-pooling.cc`
- **Size**: 2,945 bytes (2.88 KB)
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
#include <cfloat>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include <pytorch_qnnpack.h>

#include <benchmark/benchmark.h>

static void global_average_pooling_q8(benchmark::State& state) {
  const size_t batchSize = state.range(0);
  const size_t inputHeight = state.range(1);
  const size_t inputWidth = state.range(2);
  const size_t channels = state.range(3);

  std::random_device randomDevice;
  auto rng = std::mt19937(randomDevice());
  auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

  const size_t inputPixelStride = channels;
  const size_t outputPixelStride = channels;

  std::vector<uint8_t> input(
      batchSize * inputHeight * inputWidth * inputPixelStride);
  std::generate(input.begin(), input.end(), std::ref(u8rng));
  std::vector<uint8_t> output(batchSize * outputPixelStride);

  pytorch_qnnp_status status = pytorch_qnnp_initialize();
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to initialize QNNPACK");
  }

  pytorch_qnnp_operator_t globalPoolingOperator = nullptr;
  status = pytorch_qnnp_create_global_average_pooling_nwc_q8(
      channels,
      127 /* input zero point */,
      0.75f /* input scale */,
      127 /* output zero point */,
      1.25f /* output scale */,
      0,
      255,
      0 /* flags */,
      &globalPoolingOperator);
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to create Global Average Pooling operator");
  }

  status = pytorch_qnnp_setup_global_average_pooling_nwc_q8(
      globalPoolingOperator,
      batchSize,
      inputHeight * inputWidth,
      input.data(),
      inputPixelStride,
      output.data(),
      outputPixelStride);
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to setup Global Average Pooling operator");
  }

  for (auto _ : state) {
    pytorch_qnnp_run_operator(globalPoolingOperator, nullptr /* thread pool */);
  }

  status = pytorch_qnnp_delete_operator(globalPoolingOperator);
  if (status != pytorch_qnnp_status_success) {
    state.SkipWithError("failed to delete Global Average Pooling operator");
  }
  globalPoolingOperator = nullptr;

  state.SetBytesProcessed(
      uint64_t(state.iterations()) * batchSize *
      (inputHeight * inputWidth + 1) * channels * sizeof(uint8_t));
}

static void ImageNetArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "C"});

  /*       N  IH  IW    C */
  b->Args({1, 7, 7, 1000});
  b->Args({1, 13, 13, 1000});
}

BENCHMARK(global_average_pooling_q8)->Apply(ImageNetArguments);

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
- `cfloat`
- `chrono`
- `cmath`
- `functional`
- `iostream`
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
- [`q8gemm_sparse.cc_docs.md`](./q8gemm_sparse.cc_docs.md)
- [`requantization.cc_docs.md`](./requantization.cc_docs.md)
- [`max-pooling.cc_docs.md`](./max-pooling.cc_docs.md)
- [`hgemm.cc_docs.md`](./hgemm.cc_docs.md)
- [`add.cc_docs.md`](./add.cc_docs.md)
- [`q8gemm.cc_docs.md`](./q8gemm.cc_docs.md)
- [`convolution.cc_docs.md`](./convolution.cc_docs.md)


## Cross-References

- **File Documentation**: `global-average-pooling.cc_docs.md`
- **Keyword Index**: `global-average-pooling.cc_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/quantized/cpu/qnnpack/bench`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/quantized/cpu/qnnpack/bench`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


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

Files in the same folder (`docs/aten/src/ATen/native/quantized/cpu/qnnpack/bench`):

- [`sgemm.cc_kw.md_docs.md`](./sgemm.cc_kw.md_docs.md)
- [`channel-shuffle.cc_docs.md_docs.md`](./channel-shuffle.cc_docs.md_docs.md)
- [`q8gemm.cc_docs.md_docs.md`](./q8gemm.cc_docs.md_docs.md)
- [`requantization.cc_docs.md_docs.md`](./requantization.cc_docs.md_docs.md)
- [`tanh.cc_docs.md_docs.md`](./tanh.cc_docs.md_docs.md)
- [`hardsigmoid.cc_docs.md_docs.md`](./hardsigmoid.cc_docs.md_docs.md)
- [`softargmax.cc_docs.md_docs.md`](./softargmax.cc_docs.md_docs.md)
- [`q8gemm.cc_kw.md_docs.md`](./q8gemm.cc_kw.md_docs.md)
- [`sigmoid.cc_kw.md_docs.md`](./sigmoid.cc_kw.md_docs.md)


## Cross-References

- **File Documentation**: `global-average-pooling.cc_docs.md_docs.md`
- **Keyword Index**: `global-average-pooling.cc_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
