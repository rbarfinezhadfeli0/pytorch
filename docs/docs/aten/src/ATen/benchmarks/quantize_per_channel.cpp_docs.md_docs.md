# Documentation: `docs/aten/src/ATen/benchmarks/quantize_per_channel.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/benchmarks/quantize_per_channel.cpp_docs.md`
- **Size**: 4,808 bytes (4.70 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file contains **examples or benchmarks**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/benchmarks/quantize_per_channel.cpp`

## File Metadata

- **Path**: `aten/src/ATen/benchmarks/quantize_per_channel.cpp`
- **Size**: 2,816 bytes (2.75 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```cpp
#include <ATen/ATen.h>
#include <iostream>

#include <benchmark/benchmark.h>

static void quantize_per_channel_4d_contiguous(benchmark::State& state) {
  const size_t batches = static_cast<size_t>(state.range(0));
  const size_t channels = static_cast<size_t>(state.range(1));
  const size_t height = static_cast<size_t>(state.range(2));
  const size_t width = static_cast<size_t>(state.range(3));

  at::Tensor a = at::rand({batches, channels, height, width});
  at::Tensor scales = at::rand({channels});
  at::Tensor zero_points = at::randint(
      0, 10, {channels}, at::TensorOptions().dtype(at::ScalarType::Int));

  at::Tensor qa;
  for (auto _ : state) {
    qa = at::native::quantize_per_channel(
        a, scales, zero_points, 1, at::ScalarType::QUInt8);
  }
}

static void quantize_per_channel_4d_channels_last(benchmark::State& state) {
  const size_t batches = static_cast<size_t>(state.range(0));
  const size_t channels = static_cast<size_t>(state.range(1));
  const size_t height = static_cast<size_t>(state.range(2));
  const size_t width = static_cast<size_t>(state.range(3));

  at::Tensor a = at::rand(
      {batches, channels, height, width},
      at::TensorOptions().memory_format(at::MemoryFormat::ChannelsLast));
  at::Tensor scales = at::rand({channels});
  at::Tensor zero_points = at::randint(
      0, 10, {channels}, at::TensorOptions().dtype(at::ScalarType::Int));

  at::Tensor qa;
  for (auto _ : state) {
    qa = at::native::quantize_per_channel(
        a, scales, zero_points, 1, at::ScalarType::QUInt8);
  }
}

static void quantize_per_channel_2d(benchmark::State& state) {
  const size_t channels = static_cast<size_t>(state.range(0));
  const size_t nelem = static_cast<size_t>(state.range(1));

  at::Tensor a = at::rand({channels, nelem});
  at::Tensor scales = at::rand({channels});
  at::Tensor zero_points = at::randint(
      0, 10, {channels}, at::TensorOptions().dtype(at::ScalarType::Int));

  at::Tensor qa;
  for (auto _ : state) {
    qa = at::native::quantize_per_channel(
        a, scales, zero_points, 0, at::ScalarType::QUInt8);
  }
}

static void GenerateSizes4d(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "C", "H", "W"});

  for (size_t n = 16; n < 256; n *= 2) {
    for (size_t c = 4; c < 256; c *= 2) {
      for (size_t hw = 4; hw < 256; hw *= 2) {
        b->Args({n, c, hw, hw});
      }
    }
  }
}

static void GenerateSizes2d(benchmark::internal::Benchmark* b) {
  b->ArgNames({"C", "N"});

  for (size_t c = 4; c < 512; c *= 2) {
    for (size_t n = 4; n < 512; n *= 2) {
      b->Args({c, n});
    }
  }
}

BENCHMARK(quantize_per_channel_2d)->Apply(GenerateSizes2d);
BENCHMARK(quantize_per_channel_4d_contiguous)->Apply(GenerateSizes4d);
BENCHMARK(quantize_per_channel_4d_channels_last)->Apply(GenerateSizes4d);
BENCHMARK_MAIN();

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/benchmarks`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ATen.h`
- `iostream`
- `benchmark/benchmark.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`aten/src/ATen/benchmarks`):

- [`tensor_add.cpp_docs.md`](./tensor_add.cpp_docs.md)
- [`stateful_conv1d.cpp_docs.md`](./stateful_conv1d.cpp_docs.md)


## Cross-References

- **File Documentation**: `quantize_per_channel.cpp_docs.md`
- **Keyword Index**: `quantize_per_channel.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/benchmarks`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/benchmarks`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/aten/src/ATen/benchmarks`):

- [`stateful_conv1d.cpp_docs.md_docs.md`](./stateful_conv1d.cpp_docs.md_docs.md)
- [`tensor_add.cpp_kw.md_docs.md`](./tensor_add.cpp_kw.md_docs.md)
- [`quantize_per_channel.cpp_kw.md_docs.md`](./quantize_per_channel.cpp_kw.md_docs.md)
- [`stateful_conv1d.cpp_kw.md_docs.md`](./stateful_conv1d.cpp_kw.md_docs.md)
- [`tensor_add.cpp_docs.md_docs.md`](./tensor_add.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `quantize_per_channel.cpp_docs.md_docs.md`
- **Keyword Index**: `quantize_per_channel.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
