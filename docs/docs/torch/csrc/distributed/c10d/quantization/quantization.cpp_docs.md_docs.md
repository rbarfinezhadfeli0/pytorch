# Documentation: `docs/torch/csrc/distributed/c10d/quantization/quantization.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/c10d/quantization/quantization.cpp_docs.md`
- **Size**: 5,093 bytes (4.97 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/c10d/quantization/quantization.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/quantization/quantization.cpp`
- **Size**: 2,819 bytes (2.75 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/distributed/c10d/quantization/quantization.h>
#include <torch/csrc/distributed/c10d/quantization/quantization_utils.h>
#include <torch/library.h>

namespace torch::distributed::c10d::quantization {

// TODO: The kernels are copied from fbgemm_gpu, we should dedup them later

static void FloatToBFloat16Quantized_ref(
    const float* const input,
    const size_t nrows,
    const size_t ncols,
    uint16_t* const output) {
  for (const auto row : c10::irange(nrows)) {
    const float* input_row = input + row * ncols;
    uint16_t* output_row = output + row * ncols;

    for (const auto col : c10::irange(ncols)) {
      output_row[col] =
          (*reinterpret_cast<const uint32_t*>(input_row + col) + (1 << 15)) >>
          16;
    }
  }
}

static void BFloat16QuantizedToFloat_ref(
    const at::BFloat16* const input,
    const size_t nrows,
    const size_t ncols,
    float* const output) {
  for (const auto row : c10::irange(nrows)) {
    const at::BFloat16* input_row = input + row * ncols;
    float* output_row = output + row * ncols;

    for (const auto col : c10::irange(ncols)) {
      uint32_t val_fp32 = static_cast<uint32_t>(
                              reinterpret_cast<const uint16_t*>(input_row)[col])
          << 16;
      reinterpret_cast<uint32_t*>(output_row)[col] = val_fp32;
    }
  }
}

at::Tensor _float_to_bfloat16_cpu(const at::Tensor& input) {
  TENSOR_ON_CPU(input);
  // Currently it supports 2D inputs
  TENSOR_NDIM_EQUALS(input, 2);

  const auto input_sizes = input.sizes();
  const auto nrows = input_sizes[0];
  const auto ncols = input_sizes[1];
  auto output = at::empty({nrows, ncols}, input.options().dtype(at::kHalf));

  FloatToBFloat16Quantized_ref(
      input.const_data_ptr<float>(),
      nrows,
      ncols,
      reinterpret_cast<uint16_t*>(output.mutable_data_ptr<at::Half>()));

  return output;
}

at::Tensor _bfloat16_to_float_cpu(const at::Tensor& input) {
  TENSOR_ON_CPU(input);
  // Currently it supports 2D inputs
  TENSOR_NDIM_EQUALS(input, 2);

  const auto input_sizes = input.sizes();
  const auto nrows = input_sizes[0];
  const auto ncols = input_sizes[1];

  auto output = at::empty({nrows, ncols}, input.options().dtype(at::kFloat));
  BFloat16QuantizedToFloat_ref(
      reinterpret_cast<const at::BFloat16*>(input.const_data_ptr<at::Half>()),
      nrows,
      ncols,
      output.mutable_data_ptr<float>());

  return output;
}

TORCH_LIBRARY(quantization, m) {
  m.def("_Bfloat16QuantizedToFloat(Tensor input) -> Tensor");
  m.def("_FloatToBfloat16Quantized(Tensor input) -> Tensor");
}

TORCH_LIBRARY_IMPL(quantization, CPU, m) {
  m.impl("_Bfloat16QuantizedToFloat", _bfloat16_to_float_cpu);
  m.impl("_FloatToBfloat16Quantized", _float_to_bfloat16_cpu);
}

} // namespace torch::distributed::c10d::quantization

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/c10d/quantization`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/distributed/c10d/quantization/quantization.h`
- `torch/csrc/distributed/c10d/quantization/quantization_utils.h`
- `torch/library.h`


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

Files in the same folder (`torch/csrc/distributed/c10d/quantization`):

- [`quantization_gpu.cu_docs.md`](./quantization_gpu.cu_docs.md)
- [`quantization_gpu.h_docs.md`](./quantization_gpu.h_docs.md)
- [`quantization_utils.h_docs.md`](./quantization_utils.h_docs.md)
- [`quantization.h_docs.md`](./quantization.h_docs.md)


## Cross-References

- **File Documentation**: `quantization.cpp_docs.md`
- **Keyword Index**: `quantization.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/distributed/c10d/quantization`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/distributed/c10d/quantization`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/distributed/c10d/quantization`):

- [`quantization.h_docs.md_docs.md`](./quantization.h_docs.md_docs.md)
- [`quantization_utils.h_kw.md_docs.md`](./quantization_utils.h_kw.md_docs.md)
- [`quantization.cpp_kw.md_docs.md`](./quantization.cpp_kw.md_docs.md)
- [`quantization_gpu.cu_kw.md_docs.md`](./quantization_gpu.cu_kw.md_docs.md)
- [`quantization_gpu.h_kw.md_docs.md`](./quantization_gpu.h_kw.md_docs.md)
- [`quantization_utils.h_docs.md_docs.md`](./quantization_utils.h_docs.md_docs.md)
- [`quantization.h_kw.md_docs.md`](./quantization.h_kw.md_docs.md)
- [`quantization_gpu.cu_docs.md_docs.md`](./quantization_gpu.cu_docs.md_docs.md)
- [`quantization_gpu.h_docs.md_docs.md`](./quantization_gpu.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `quantization.cpp_docs.md_docs.md`
- **Keyword Index**: `quantization.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
