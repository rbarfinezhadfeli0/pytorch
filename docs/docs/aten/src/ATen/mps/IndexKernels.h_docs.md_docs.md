# Documentation: `docs/aten/src/ATen/mps/IndexKernels.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/mps/IndexKernels.h_docs.md`
- **Size**: 11,779 bytes (11.50 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/mps/IndexKernels.h`

## File Metadata

- **Path**: `aten/src/ATen/mps/IndexKernels.h`
- **Size**: 9,476 bytes (9.25 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

namespace at::mps {

static const char* SCATTER_OPS_TEMPLATE = R"METAL_SCATTER(
template<typename Y, typename X>
Y cast(const X x);

template<>
{1} cast<{1}, {0}>(const {0} x) {{
 return {2};
}}

kernel void scatter_kernel_n(uint linear_index          [[thread_position_in_grid]],
                             constant void * src_       [[buffer(0)]],
                             device void * dst_         [[buffer(1)]],
                             constant uint32_t * size   [[buffer(2)]],
                             constant uint32_t * stride [[buffer(3)]],
                            constant uint32_t & numel   [[buffer(4)]],
                            constant int32_t & ndim     [[buffer(5)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    uint64_t dst_offs = 0;
    auto dst_idx = linear_index;
    for(int dim = ndim - 1; dim >= 0; --dim) {{
      dst_offs += stride[dim] * (dst_idx % size[dim]);
      dst_idx /= size[dim];
    }}

    dst[dst_offs] = cast<{1}>(src[linear_index]);
}}

kernel void scatter_kernel_4(uint linear_index              [[thread_position_in_grid]],
                             constant void * src_           [[buffer(0)]],
                             device void * dst_             [[buffer(1)]],
                             constant packed_uint4 & size   [[buffer(2)]],
                             constant packed_uint4 & stride [[buffer(3)]],
                             constant uint32_t & numel      [[buffer(4)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    packed_uint4 local_index;
    local_index.x = linear_index / (size[3] * size[2] * size[1]) % size[0];
    local_index.y = linear_index / (size[3] * size[2]) % size[1];
    local_index.z = linear_index / size[3] % size[2];
    local_index.w = linear_index % size[3];

    const packed_uint4 strided_index = local_index * stride;
    dst[strided_index.x + strided_index.y + strided_index.z + strided_index.w] = cast<{1}>(src[linear_index]);
}}

kernel void scatter_kernel_3(uint linear_index              [[thread_position_in_grid]],
                             constant void * src_           [[buffer(0)]],
                             device void * dst_             [[buffer(1)]],
                             constant packed_uint3 & size   [[buffer(2)]],
                             constant packed_uint3 & stride [[buffer(3)]],
                             constant uint32_t & numel      [[buffer(4)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    packed_uint3 local_index;
    local_index.x = linear_index / (size[2] * size[1]) % size[0];
    local_index.y = linear_index / size[2] % size[1];
    local_index.z = linear_index % size[2];

    const packed_uint3 strided_index = local_index * stride;
    dst[strided_index.x + strided_index.y + strided_index.z] = cast<{1}>(src[linear_index]);
}}

kernel void scatter_kernel_2(uint linear_index              [[thread_position_in_grid]],
                             constant void * src_           [[buffer(0)]],
                             device void * dst_             [[buffer(1)]],
                             constant packed_uint2 & size   [[buffer(2)]],
                             constant packed_uint2 & stride [[buffer(3)]],
                             constant uint32_t & numel      [[buffer(4)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    packed_uint2 local_index;
    local_index.x = linear_index / size[1] % size[0];
    local_index.y = linear_index % size[1];

    const packed_uint2 strided_index = local_index * stride;
    dst[strided_index.x + strided_index.y] = cast<{1}>(src[linear_index]);
}}

kernel void scatter_kernel_1(uint linear_index              [[thread_position_in_grid]],
                             constant void * src_           [[buffer(0)]],
                             device void * dst_             [[buffer(1)]],
                             constant int & size            [[buffer(2)]],
                             constant int & stride          [[buffer(3)]],
                             constant uint32_t & numel      [[buffer(4)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    const int local_index = linear_index % size;
    const int strided_index = local_index * stride;
    dst[strided_index] = cast<{1}>(src[linear_index]);
}}
)METAL_SCATTER";

static const char* GATHER_OPS_TEMPLATE = R"METAL_GATHER(
template<typename Y, typename X>
Y cast(const X x);

template<>
{1} cast<{1}, {0}>(const {0} x) {{
 return {2};
}}

kernel void gather_kernel_n(uint linear_index           [[thread_position_in_grid]],
                            constant void * src_        [[buffer(0)]],
                            device void * dst_          [[buffer(1)]],
                            constant uint32_t * size    [[buffer(2)]],
                            constant uint32_t * stride  [[buffer(3)]],
                            constant uint32_t & numel   [[buffer(4)]],
                            constant int32_t & ndim     [[buffer(5)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    uint64_t src_offs = 0;
    auto src_idx = linear_index;
    for(int dim = ndim - 1; dim >= 0; --dim) {{
      src_offs += stride[dim] * (src_idx % size[dim]);
      src_idx /= size[dim];
    }}

    dst[linear_index] = cast<{1}>(src[src_offs]);
}}

kernel void gather_kernel_4(uint linear_index               [[thread_position_in_grid]],
                            constant void * src_            [[buffer(0)]],
                            device void * dst_              [[buffer(1)]],
                            constant packed_uint4 & size    [[buffer(2)]],
                            constant packed_uint4 & stride  [[buffer(3)]],
                            constant uint32_t & numel       [[buffer(4)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    packed_uint4 local_index;
    local_index.x = linear_index / (size[3] * size[2] * size[1]) % size[0];
    local_index.y = linear_index / (size[3] * size[2]) % size[1];
    local_index.z = linear_index / size[3] % size[2];
    local_index.w = linear_index % size[3];

    const packed_uint4 strided_index = local_index * stride;
    dst[linear_index] = cast<{1}>(src[strided_index.x + strided_index.y + strided_index.z + strided_index.w]);
}}

kernel void gather_kernel_3(uint linear_index               [[thread_position_in_grid]],
                            constant void * src_            [[buffer(0)]],
                            device void * dst_              [[buffer(1)]],
                            constant packed_uint3 & size    [[buffer(2)]],
                            constant packed_uint3 & stride  [[buffer(3)]],
                            constant uint32_t & numel       [[buffer(4)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    packed_uint3 local_index;
    local_index.x = linear_index / (size[2] * size[1]) % size[0];
    local_index.y = linear_index / size[2] % size[1];
    local_index.z = linear_index % size[2];

    const packed_uint3 strided_index = local_index * stride;
    dst[linear_index] = cast<{1}>(src[strided_index.x + strided_index.y + strided_index.z]);
}}

kernel void gather_kernel_2(uint linear_index               [[thread_position_in_grid]],
                            constant void * src_            [[buffer(0)]],
                            device void * dst_              [[buffer(1)]],
                            constant packed_uint2 & size    [[buffer(2)]],
                            constant packed_uint2 & stride  [[buffer(3)]],
                            constant uint32_t & numel       [[buffer(4)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    packed_uint2 local_index;
    local_index.x = linear_index / size[1] % size[0];
    local_index.y = linear_index % size[1];

    const packed_uint2 strided_index = local_index * stride;
    dst[linear_index] = cast<{1}>(src[strided_index.x + strided_index.y]);
}}

kernel void gather_kernel_1(uint linear_index               [[thread_position_in_grid]],
                            constant void * src_            [[buffer(0)]],
                            device void * dst_              [[buffer(1)]],
                            constant int & size             [[buffer(2)]],
                            constant int & stride           [[buffer(3)]],
                            constant uint32_t & numel       [[buffer(4)]]) {{
    if (linear_index >= numel) return;

    constant {0} * src = (constant {0} *)src_;
    device {1} * dst = (device {1} *)dst_;

    const int local_index = linear_index % size;
    const int strided_index = local_index * stride;
    dst[linear_index] = cast<{1}>(src[strided_index]);
}}
)METAL_GATHER";
} // namespace at::mps

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/mps`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*No includes detected.*


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

Files in the same folder (`aten/src/ATen/mps`):

- [`MPSProfiler.h_docs.md`](./MPSProfiler.h_docs.md)
- [`MPSAllocator.h_docs.md`](./MPSAllocator.h_docs.md)
- [`MPSDevice.h_docs.md`](./MPSDevice.h_docs.md)
- [`MPSAllocatorInterface.h_docs.md`](./MPSAllocatorInterface.h_docs.md)
- [`MPSEvent.h_docs.md`](./MPSEvent.h_docs.md)
- [`MPSGuardImpl.h_docs.md`](./MPSGuardImpl.h_docs.md)
- [`MPSHooks.h_docs.md`](./MPSHooks.h_docs.md)
- [`EmptyTensor.h_docs.md`](./EmptyTensor.h_docs.md)
- [`MPSGeneratorImpl.h_docs.md`](./MPSGeneratorImpl.h_docs.md)


## Cross-References

- **File Documentation**: `IndexKernels.h_docs.md`
- **Keyword Index**: `IndexKernels.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/mps`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/mps`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/mps`):

- [`MPSAllocatorInterface.h_kw.md_docs.md`](./MPSAllocatorInterface.h_kw.md_docs.md)
- [`MPSAllocatorInterface.h_docs.md_docs.md`](./MPSAllocatorInterface.h_docs.md_docs.md)
- [`MPSHooks.h_kw.md_docs.md`](./MPSHooks.h_kw.md_docs.md)
- [`MPSDevice.h_kw.md_docs.md`](./MPSDevice.h_kw.md_docs.md)
- [`IndexKernels.h_kw.md_docs.md`](./IndexKernels.h_kw.md_docs.md)
- [`MPSProfiler.h_docs.md_docs.md`](./MPSProfiler.h_docs.md_docs.md)
- [`MPSGuardImpl.h_kw.md_docs.md`](./MPSGuardImpl.h_kw.md_docs.md)
- [`EmptyTensor.cpp_docs.md_docs.md`](./EmptyTensor.cpp_docs.md_docs.md)
- [`MPSEvent.h_docs.md_docs.md`](./MPSEvent.h_docs.md_docs.md)
- [`MPSAllocator.h_kw.md_docs.md`](./MPSAllocator.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `IndexKernels.h_docs.md_docs.md`
- **Keyword Index**: `IndexKernels.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
