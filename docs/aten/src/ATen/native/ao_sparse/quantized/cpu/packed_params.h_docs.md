# Documentation: `aten/src/ATen/native/ao_sparse/quantized/cpu/packed_params.h`

## File Metadata

- **Path**: `aten/src/ATen/native/ao_sparse/quantized/cpu/packed_params.h`
- **Size**: 2,735 bytes (2.67 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <cstdint>

#include <ATen/core/ivalue.h>

namespace ao::sparse {

// <Weight, bias, out_features_block_size, in_features_block_size>
using LinearPackedSerializationType =
    std::tuple<at::Tensor, std::optional<at::Tensor>, std::vector<int64_t>>;

#define SPARSE_LINEAR_PACKED_PARAM_SERIALIZATION_VERSION 2

using BCSRSerializationType =
    std::tuple<
        int64_t,                    // Serialization Version
        std::optional<at::Tensor>,  // Bias
        int64_t,                    // Out Features (Row) Block Size
        int64_t,                    // In Features (Column) Block Size
        at::Tensor,                 // Weight Scales (single element vector if per-tensor) (float)
        at::Tensor,                 // Wrapper for Weight Zero Points (single element vector if per-tensor) (int8_t)
        bool,                       // Quantization Scheme (true: per tensor, false: per channel)
        at::Tensor,                 // Wrapper for Row Block Indices (int8_t, int16_t, or int32_t)
        at::Tensor,                 // Wrapper for Column Block Indices (int8_t, int16_t, or int32_t)
        at::Tensor,                 // Wrapper for Non-Zero Weight Values, each +128 (uint8_t)
        int64_t,                    // Number of Output Channels
        int64_t                     // Number of Input Channels
    >;

using BCSR =
    std::tuple<
        std::vector<int8_t>,    // Non-Zero Weight Values
        std::vector<int32_t>,   // Compressed Row Block Indices
        std::vector<int32_t>    // Column Block Indices
    >;

struct LinearPackedParamsBase : public torch::jit::CustomClassHolder {
 public:
  LinearPackedParamsBase(
      const int64_t out_features_block_size,
      const int64_t in_features_block_size)
      : out_features_block_size_(out_features_block_size),
        in_features_block_size_(in_features_block_size) {}

  virtual at::Tensor apply(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) = 0;
  virtual at::Tensor apply_relu(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) = 0;

  virtual at::Tensor apply_dynamic(const at::Tensor& input) = 0;
  virtual at::Tensor apply_dynamic_relu(const at::Tensor& input) = 0;

  virtual LinearPackedSerializationType unpack() = 0;

  virtual BCSRSerializationType serialize() = 0;

  virtual std::optional<at::Tensor> bias() = 0;

  virtual void set_bias(const std::optional<at::Tensor>& bias) {
    throw std::runtime_error(
        "set_bias is not implemented for this packed "
        "parameter type");
  }

 protected:
  const int64_t out_features_block_size_, in_features_block_size_;
};

}  // namespace ao::sparse

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 14 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `ao`

**Classes/Structs**: `LinearPackedParamsBase`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/ao_sparse/quantized/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `cstdint`
- `ATen/core/ivalue.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`aten/src/ATen/native/ao_sparse/quantized/cpu`):

- [`fbgemm_utils.h_docs.md`](./fbgemm_utils.h_docs.md)
- [`qlinear_serialize.cpp_docs.md`](./qlinear_serialize.cpp_docs.md)
- [`qlinear_dynamic.cpp_docs.md`](./qlinear_dynamic.cpp_docs.md)
- [`qlinear_unpack.cpp_docs.md`](./qlinear_unpack.cpp_docs.md)
- [`fbgemm_utils.cpp_docs.md`](./fbgemm_utils.cpp_docs.md)
- [`qlinear_deserialize.cpp_docs.md`](./qlinear_deserialize.cpp_docs.md)
- [`qlinear_prepack.cpp_docs.md`](./qlinear_prepack.cpp_docs.md)
- [`qnnpack_utils.h_docs.md`](./qnnpack_utils.h_docs.md)
- [`qlinear.cpp_docs.md`](./qlinear.cpp_docs.md)


## Cross-References

- **File Documentation**: `packed_params.h_docs.md`
- **Keyword Index**: `packed_params.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
