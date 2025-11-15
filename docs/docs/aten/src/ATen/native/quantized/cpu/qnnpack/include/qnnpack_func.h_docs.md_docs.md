# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/include/qnnpack_func.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/include/qnnpack_func.h_docs.md`
- **Size**: 6,306 bytes (6.16 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/include/qnnpack_func.h`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/include/qnnpack_func.h`
- **Size**: 4,146 bytes (4.05 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <cstdlib>
#include <qnnpack/operator.h>

namespace qnnpack {
class PrePackConvWeights final {
 public:
  PrePackConvWeights(
      const pytorch_qnnp_operator_t convolution,
      const uint8_t* kernel_zero_points,
      const uint8_t* kernel,
      const int32_t* bias);

  void* getPackedWeights() const
  {
    return packed_weights_;
  }

  int64_t getOutputChannels() const
  {
    return output_channels_;
  }

  ~PrePackConvWeights()
  {
    if (packed_weights_ != nullptr) {
      free(packed_weights_);
    }
  }

  PrePackConvWeights() = delete;
  PrePackConvWeights(const PrePackConvWeights&) = delete;
  PrePackConvWeights& operator=(const PrePackConvWeights&) = delete;

 private:
  void* packed_weights_ = nullptr;
  int64_t output_channels_;
};

class PackBMatrix final {
 public:
  PackBMatrix(
      size_t input_channels,
      size_t output_channels,
      const uint8_t* kernel_zero_points,
      const float* requantization_scale,
      const uint8_t* kernel,
      const int32_t* bias);

  // This constructor is to be used for dynamic mode
  // quantization. In dynamic mode, we dont yet support
  // per channel quantization, and paying the cost of
  // memory allocation for per channel zero point and
  // requant scale will hurt performance.
  PackBMatrix(
      size_t input_channels,
      size_t output_channels,
      const uint8_t kernel_zero_point,
      const float requantization_scale,
      const uint8_t* kernel,
      const int32_t* bias);

  void* getPackedWeights() const
  {
    return packed_weights_;
  }

  void unpackWeights(
      const uint8_t* kernel_zero_points,
      int8_t* kernel
    ) const;

  size_t getInputChannels() const
  {
    return input_channels_;
  }

  size_t getOutputChannels() const
  {
    return output_channels_;
  }

  ~PackBMatrix()
  {
    if (packed_weights_ != nullptr) {
      free(packed_weights_);
    }
  }

  PackBMatrix() = delete;
  PackBMatrix(const PackBMatrix&) = delete;
  PackBMatrix& operator=(const PackBMatrix&) = delete;

 private:
  void* packed_weights_ = nullptr;
  size_t input_channels_;
  size_t output_channels_;
};

enum pytorch_qnnp_status qnnpackLinear(
    const size_t batch_size,
    const size_t input_channels,
    const size_t output_channels,
    const uint8_t input_zero_point,
    const uint8_t* kernel_zero_points,
    const float* requantization_scales,
    const uint8_t output_zero_point,
    const uint8_t output_min,
    const uint8_t output_max,
    const uint8_t* input,
    const size_t input_stride,
    void* packed_weights,
    uint8_t* output,
    const size_t output_stride,
    pthreadpool_t threadpool);

enum pytorch_qnnp_status qnnpackConv(
    const pytorch_qnnp_operator_t convolution,
    void* packed_weights,
    const size_t batch_size,
    const size_t input_depth,
    const size_t input_height,
    const size_t input_width,
    const uint8_t input_zero_point,
    const uint8_t* input,
    const uint8_t* kernel_zero_points,
    const float* requantization_scales,
    const uint8_t output_zero_point,
    const uint8_t output_min,
    const uint8_t output_max,
    uint8_t* output,
    pthreadpool_t threadpool);

enum pytorch_qnnp_status qnnpackDeConv(
    const pytorch_qnnp_operator_t deconvolution,
    void* packed_weights,
    const size_t batch_size,
    const size_t input_height,
    const size_t input_width,
    const uint8_t input_zero_point,
    const uint8_t* input,
    const uint8_t* kernel_zero_points,
    const float* requantization_scales,
    const uint8_t output_zero_point,
    const uint8_t output_min,
    const uint8_t output_max,
    uint8_t* output,
    pthreadpool_t threadpool);

enum pytorch_qnnp_status qnnpackLinearDynamic(
    const size_t batch_size,
    const size_t input_channels,
    const size_t output_channels,
    const uint8_t input_zero_point,
    const uint8_t* kernel_zero_points,
    const float* dequantization_scales,
    const uint8_t* input,
    const size_t input_stride,
    void* packed_weights,
    const float* bias,
    float* output,
    const size_t output_stride,
    pthreadpool_t threadpool);

} // namespace qnnpack

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `qnnpack`

**Classes/Structs**: `PrePackConvWeights`, `PackBMatrix`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu/qnnpack/include`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `cstdlib`
- `qnnpack/operator.h`


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

Files in the same folder (`aten/src/ATen/native/quantized/cpu/qnnpack/include`):

- [`pack_block_sparse.h_docs.md`](./pack_block_sparse.h_docs.md)
- [`pytorch_qnnpack.h_docs.md`](./pytorch_qnnpack.h_docs.md)


## Cross-References

- **File Documentation**: `qnnpack_func.h_docs.md`
- **Keyword Index**: `qnnpack_func.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/quantized/cpu/qnnpack/include`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/quantized/cpu/qnnpack/include`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/quantized/cpu/qnnpack/include`):

- [`pack_block_sparse.h_docs.md_docs.md`](./pack_block_sparse.h_docs.md_docs.md)
- [`qnnpack_func.h_kw.md_docs.md`](./qnnpack_func.h_kw.md_docs.md)
- [`pytorch_qnnpack.h_docs.md_docs.md`](./pytorch_qnnpack.h_docs.md_docs.md)
- [`pytorch_qnnpack.h_kw.md_docs.md`](./pytorch_qnnpack.h_kw.md_docs.md)
- [`pack_block_sparse.h_kw.md_docs.md`](./pack_block_sparse.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `qnnpack_func.h_docs.md_docs.md`
- **Keyword Index**: `qnnpack_func.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
