# Documentation: `docs/aten/src/ATen/native/cuda/cutlass_extensions/gemm/kernel/mixed_gemm_B_layout.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/cutlass_extensions/gemm/kernel/mixed_gemm_B_layout.h_docs.md`
- **Size**: 6,373 bytes (6.22 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/cutlass_extensions/gemm/kernel/mixed_gemm_B_layout.h`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/cutlass_extensions/gemm/kernel/mixed_gemm_B_layout.h`
- **Size**: 3,802 bytes (3.71 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
/*
  This file exists so that we use the same weight layout for MoE grouped gemm and regular gemm when the weight is
  quantized. The preprocessing code reads this template to know how to organize the quantized weight matrices
  to be consumed by CUTLASS.

  Note that for int4, ThreadBlockK MUST be 64.

 */

#pragma once

#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/platform/platform.h>

#include <ATen/native/cuda/cutlass_extensions/arch/mma.h>
#include <ATen/native/cuda/cutlass_extensions/tile_interleaved_layout.h>

namespace cutlass {
namespace gemm {
namespace kernel {

template<typename TypeB, typename Arch, typename Enable = void>
struct LayoutDetailsB {
};

// Volta specialiations. Volta will dequantize before STS, so we need a different operator
template<typename TypeB>
struct LayoutDetailsB<TypeB, arch::Sm70> {
    static constexpr int ThreadblockK      = 64;
    using Layout                           = layout::RowMajor;
    static constexpr int ElementsPerAccess = 8;
    using Operator                         = cutlass::arch::OpMultiplyAdd;
};

// Specializations for Turing+ when B is FP16. These are currently only used for MoE networks.
// TODO - Switch this to column major for weights since gemms should be more performant.
template<typename Arch>
struct LayoutDetailsB<half_t, Arch, typename platform::enable_if<Arch::kMinComputeCapability >= 75>::type> {
    static constexpr int ThreadblockK      = 64;
    using Layout                           = layout::RowMajor;
    static constexpr int ElementsPerAccess = 128 / cutlass::sizeof_bits<half_t>::value;
    using Operator                         = cutlass::arch::OpMultiplyAdd;
};

template<typename Arch>
struct LayoutDetailsB<bfloat16_t, Arch, typename platform::enable_if<Arch::kMinComputeCapability >= 75>::type> {
    static constexpr int ThreadblockK      = 64;
    using Layout                           = layout::RowMajor;
    static constexpr int ElementsPerAccess = 128 / cutlass::sizeof_bits<bfloat16_t>::value;
    using Operator                         = cutlass::arch::OpMultiplyAdd;
};

// Specializations for Turing+ when B is quantized. These can use the operator OpMultiplyAddDequantizeInterleavedBToA,
// which signals that we want to dequantize after loading from smem.
template<typename Arch>
struct LayoutDetailsB<uint8_t, Arch, typename platform::enable_if<Arch::kMinComputeCapability >= 75>::type> {
    static constexpr int ThreadblockK = 64;

private:
    static constexpr int ElementsPerCacheLine = 128 * 8 / sizeof_bits<uint8_t>::value;
    static constexpr int ColumnsInterleaved   = ElementsPerCacheLine / ThreadblockK;

public:
    using Layout                           = layout::ColumnMajorTileInterleave<ThreadblockK, ColumnsInterleaved>;
    static constexpr int ElementsPerAccess = 128 / cutlass::sizeof_bits<uint8_t>::value;
    using Operator                         = cutlass::arch::OpMultiplyAddDequantizeInterleavedBToA;
};

template<typename Arch>
struct LayoutDetailsB<uint4b_t, Arch, typename platform::enable_if<Arch::kMinComputeCapability >= 75>::type> {
    static constexpr int ThreadblockK = 64;

private:
    static constexpr int ElementsPerCacheLine = 128 * 8 / sizeof_bits<uint4b_t>::value;
    static constexpr int ColumnsInterleaved   = ElementsPerCacheLine / ThreadblockK;

public:
    using Layout                           = layout::ColumnMajorTileInterleave<ThreadblockK, ColumnsInterleaved>;
    static constexpr int ElementsPerAccess = 128 / cutlass::sizeof_bits<uint4b_t>::value;
    using Operator                         = cutlass::arch::OpMultiplyAddDequantizeInterleavedBToA;
};

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 0 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `cutlass`, `gemm`, `kernel`

**Classes/Structs**: `LayoutDetailsB`, `LayoutDetailsB`, `LayoutDetailsB`, `LayoutDetailsB`, `LayoutDetailsB`, `LayoutDetailsB`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cuda/cutlass_extensions/gemm/kernel`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `cutlass/layout/matrix.h`
- `cutlass/numeric_types.h`
- `cutlass/arch/arch.h`
- `cutlass/arch/mma.h`
- `cutlass/platform/platform.h`
- `ATen/native/cuda/cutlass_extensions/arch/mma.h`
- `ATen/native/cuda/cutlass_extensions/tile_interleaved_layout.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.

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

Files in the same folder (`aten/src/ATen/native/cuda/cutlass_extensions/gemm/kernel`):

- [`default_fpA_intB_traits.h_docs.md`](./default_fpA_intB_traits.h_docs.md)
- [`fpA_intB_gemm.h_docs.md`](./fpA_intB_gemm.h_docs.md)


## Cross-References

- **File Documentation**: `mixed_gemm_B_layout.h_docs.md`
- **Keyword Index**: `mixed_gemm_B_layout.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/cuda/cutlass_extensions/gemm/kernel`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/cuda/cutlass_extensions/gemm/kernel`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/aten/src/ATen/native/cuda/cutlass_extensions/gemm/kernel`):

- [`default_fpA_intB_traits.h_kw.md_docs.md`](./default_fpA_intB_traits.h_kw.md_docs.md)
- [`mixed_gemm_B_layout.h_kw.md_docs.md`](./mixed_gemm_B_layout.h_kw.md_docs.md)
- [`default_fpA_intB_traits.h_docs.md_docs.md`](./default_fpA_intB_traits.h_docs.md_docs.md)
- [`fpA_intB_gemm.h_docs.md_docs.md`](./fpA_intB_gemm.h_docs.md_docs.md)
- [`fpA_intB_gemm.h_kw.md_docs.md`](./fpA_intB_gemm.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `mixed_gemm_B_layout.h_docs.md_docs.md`
- **Keyword Index**: `mixed_gemm_B_layout.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
