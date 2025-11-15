# Documentation: `docs/aten/src/ATen/native/kleidiai/kai_ukernel_interface.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/kleidiai/kai_ukernel_interface.cpp_docs.md`
- **Size**: 6,810 bytes (6.65 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/kleidiai/kai_ukernel_interface.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/kleidiai/kai_ukernel_interface.cpp`
- **Size**: 4,707 bytes (4.60 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/native/kleidiai/kai_ukernel_interface.h>

#if AT_KLEIDIAI_ENABLED()

namespace at::native::kleidiai {

// Kernel Mapping - Groupwise
std::unordered_map<kai_kernel_id, kai_matmul_ukernel_f32_qa8dxp_qs4c32p> groupwise_8bit_4bit_kernels =
    {{kai_kernel_id::
          matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
      {{kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
        kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
        kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
        kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
        kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
        kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
        kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
        kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
        kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
        kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
        kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod}}},
     {kai_kernel_id::matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_4x8x32_neon_i8mm,
      {{kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
        kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
        kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
        kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
        kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
        kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
        kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
        kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
        kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
        kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
        kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm}}}};

kai_matmul_ukernel_f32_qa8dxp_qs4c32p kai_select_groupwise_matmul_ukernel(
    kai_kernel_id id) {
  return groupwise_8bit_4bit_kernels.at(id);
}

// Kernel Mapping - Channelwise
std::unordered_map<kai_kernel_id, kai_matmul_ukernel_f32_qa8dxp_qs4cxp> channelwise_8bit_4bit_kernels =
    {{kai_kernel_id::matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
      {{kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
        kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
        kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
        kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
        kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
        kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
        kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
        kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
        kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
        kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
        kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod}}},
     {kai_kernel_id::matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
      {{kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
        kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
        kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
        kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
        kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
        kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
        kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
        kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
        kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
        kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
        kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm}}}};

kai_matmul_ukernel_f32_qa8dxp_qs4cxp kai_select_channelwise_matmul_ukernel(
    const kai_kernel_id id) {
  return channelwise_8bit_4bit_kernels.at(id);
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

- `ATen/native/kleidiai/kai_ukernel_interface.h`


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

- [`kai_pack.h_docs.md`](./kai_pack.h_docs.md)
- [`kai_kernels.h_docs.md`](./kai_kernels.h_docs.md)
- [`kai_kernels.cpp_docs.md`](./kai_kernels.cpp_docs.md)
- [`kai_ukernel_interface.h_docs.md`](./kai_ukernel_interface.h_docs.md)


## Cross-References

- **File Documentation**: `kai_ukernel_interface.cpp_docs.md`
- **Keyword Index**: `kai_ukernel_interface.cpp_kw.md`
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
- [`kai_pack.h_docs.md_docs.md`](./kai_pack.h_docs.md_docs.md)
- [`kai_pack.h_kw.md_docs.md`](./kai_pack.h_kw.md_docs.md)
- [`kai_ukernel_interface.h_kw.md_docs.md`](./kai_ukernel_interface.h_kw.md_docs.md)
- [`kai_kernels.h_docs.md_docs.md`](./kai_kernels.h_docs.md_docs.md)
- [`kai_kernels.cpp_docs.md_docs.md`](./kai_kernels.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `kai_ukernel_interface.cpp_docs.md_docs.md`
- **Keyword Index**: `kai_ukernel_interface.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
