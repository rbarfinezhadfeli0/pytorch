# Documentation: `docs/aten/src/ATen/native/kleidiai/kai_pack.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/kleidiai/kai_pack.h_docs.md`
- **Size**: 4,815 bytes (4.70 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/kleidiai/kai_pack.h`

## File Metadata

- **Path**: `aten/src/ATen/native/kleidiai/kai_pack.h`
- **Size**: 2,709 bytes (2.65 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <ATen/Config.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <torch/library.h>
#if AT_KLEIDIAI_ENABLED()

namespace at::native::kleidiai {

template <typename T>
void kai_pack_rhs_groupwise_int4(
    T& kernel,
    const Tensor& weight_packed,
    const Tensor& weight,
    const Tensor& scales,
    const std::optional<Tensor>& bias,
    const int64_t n,
    const int64_t k,
    const int64_t bl,
    const int64_t rhs_stride,
    const int64_t scale_stride) {
  const auto& ukernel = kernel.ukernel;
  const size_t nr = ukernel.get_nr();
  const size_t kr = ukernel.get_kr();
  const size_t sr = ukernel.get_sr();
  auto weight_packed_data =
      reinterpret_cast<uint8_t*>(weight_packed.data_ptr());
  const auto weight_data = weight.data_ptr<uint8_t>();
  auto scales_data = scales.const_data_ptr();

  if (weight_data == nullptr) {
    AT_ERROR("kai_pack_rhs_channelwise_int4: Weight data pointer is null");
  }

  if (scales_data == nullptr) {
    AT_ERROR("kai_pack_rhs_channelwise_int4: Scales data pointer is null");
  }

  float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : NULL;
  auto& params = kernel.rhs_pack_params;

  kernel.kai_run_rhs_pack(
      /*num_groups=*/1,
      n,
      k,
      nr,
      kr,
      sr,
      bl,
      (const uint8_t*)(weight_data),
      rhs_stride,
      bias_ptr,
      scales_data,
      scale_stride,
      weight_packed_data,
      0,
      &params);
}

template <typename T>
void kai_pack_rhs_channelwise_int4(
    T& kernel,
    const Tensor& weight_packed,
    const Tensor& weight,
    const Tensor& scales,
    const std::optional<Tensor>& bias,
    const int64_t n,
    const int64_t k) {
  const auto& ukernel = kernel.ukernel;
  const size_t nr = ukernel.get_nr();
  const size_t kr = ukernel.get_kr();
  const size_t sr = ukernel.get_sr();
  auto weight_packed_data =
      reinterpret_cast<uint8_t*>(weight_packed.data_ptr());
  const auto weight_data = weight.data_ptr<uint8_t>();
  const auto scales_data = scales.data_ptr<float>();

  if (weight_data == nullptr) {
    AT_ERROR("kai_pack_rhs_channelwise_int4: Weight data pointer is null");
  }

  if (scales_data == nullptr) {
    AT_ERROR("kai_pack_rhs_channelwise_int4: Scales data pointer is null");
  }

  float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : NULL;
  auto& params = kernel.rhs_pack_params;

  kernel.kai_run_rhs_pack(
      /*num_groups=*/1,
      n,
      k,
      nr,
      kr,
      sr,
      (const uint8_t*)(weight_data),
      (const float*)(bias_ptr),
      (const float*)(scales_data),
      weight_packed_data,
      0,
      &params);
}

} // namespace at::native::kleidiai

#endif

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/kleidiai`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Config.h`
- `ATen/core/Tensor.h`
- `ATen/ops/empty.h`
- `torch/library.h`


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

Files in the same folder (`aten/src/ATen/native/kleidiai`):

- [`kai_kernels.h_docs.md`](./kai_kernels.h_docs.md)
- [`kai_ukernel_interface.cpp_docs.md`](./kai_ukernel_interface.cpp_docs.md)
- [`kai_kernels.cpp_docs.md`](./kai_kernels.cpp_docs.md)
- [`kai_ukernel_interface.h_docs.md`](./kai_ukernel_interface.h_docs.md)


## Cross-References

- **File Documentation**: `kai_pack.h_docs.md`
- **Keyword Index**: `kai_pack.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/kleidiai`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/kleidiai`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/kleidiai`):

- [`kai_ukernel_interface.h_docs.md_docs.md`](./kai_ukernel_interface.h_docs.md_docs.md)
- [`kai_kernels.h_kw.md_docs.md`](./kai_kernels.h_kw.md_docs.md)
- [`kai_kernels.cpp_kw.md_docs.md`](./kai_kernels.cpp_kw.md_docs.md)
- [`kai_ukernel_interface.cpp_kw.md_docs.md`](./kai_ukernel_interface.cpp_kw.md_docs.md)
- [`kai_ukernel_interface.cpp_docs.md_docs.md`](./kai_ukernel_interface.cpp_docs.md_docs.md)
- [`kai_pack.h_kw.md_docs.md`](./kai_pack.h_kw.md_docs.md)
- [`kai_ukernel_interface.h_kw.md_docs.md`](./kai_ukernel_interface.h_kw.md_docs.md)
- [`kai_kernels.h_docs.md_docs.md`](./kai_kernels.h_docs.md_docs.md)
- [`kai_kernels.cpp_docs.md_docs.md`](./kai_kernels.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `kai_pack.h_docs.md_docs.md`
- **Keyword Index**: `kai_pack.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
