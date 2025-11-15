# Documentation: `docs/aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.h_docs.md`
- **Size**: 7,455 bytes (7.28 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.h`

## File Metadata

- **Path**: `aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.h`
- **Size**: 5,354 bytes (5.23 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <cstddef>

#include <namespace_config.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

namespace FLASH_NAMESPACE {

TORCH_API
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_fwd(const at::Tensor &q,         // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
        std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        std::optional<at::Generator> gen_);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_varlen_fwd(const at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               std::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               std::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
               std::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
               std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               bool is_causal,
               int window_size_left,
               int window_size_right,
               const float softcap,
               const bool return_softmax,
               std::optional<at::Generator> gen_);


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_bwd(const at::Tensor &dout,  // batch_size x seqlen_q x num_heads, x head_size_og
        const at::Tensor &q,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &softmax_lse,     // b x h x seqlen_q
        std::optional<at::Tensor> &dq_,   // batch_size x seqlen_q x num_heads x head_size
        std::optional<at::Tensor> &dk_,   // batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &dv_,   // batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,         // probability to drop
        const float softmax_scale,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool deterministic,
        const at::Tensor philox_seed,
        const at::Tensor philox_offset);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_varlen_bwd(const at::Tensor &dout,  // total_q x num_heads, x head_size
               const at::Tensor &q,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &v,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &out,   // total_q x num_heads x head_size
               const at::Tensor &softmax_lse,     // b x h x s   softmax logsumexp
               std::optional<at::Tensor> &dq_,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               std::optional<at::Tensor> &dk_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               std::optional<at::Tensor> &dv_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               const int max_seqlen_q,
               const int max_seqlen_k,          // max sequence length to choose the kernel
               const float p_dropout,         // probability to drop
               const float softmax_scale,
               const bool zero_tensors,
               const bool is_causal,
               int window_size_left,
               int window_size_right,
               const float softcap,
               const bool deterministic,
               const at::Tensor philox_seed,
               const at::Tensor philox_offset);

} // namespace FLASH_NAMESPACE

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 0 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `FLASH_NAMESPACE`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/transformers/cuda/flash_attn`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `cstddef`
- `namespace_config.h`
- `ATen/core/Tensor.h`
- `c10/util/Exception.h`


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

Files in the same folder (`aten/src/ATen/native/transformers/cuda/flash_attn`):

- [`philox.cuh_docs.md`](./philox.cuh_docs.md)
- [`flash_api.cpp_docs.md`](./flash_api.cpp_docs.md)
- [`static_switch.h_docs.md`](./static_switch.h_docs.md)


## Cross-References

- **File Documentation**: `flash_api.h_docs.md`
- **Keyword Index**: `flash_api.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/transformers/cuda/flash_attn`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/transformers/cuda/flash_attn`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/transformers/cuda/flash_attn`):

- [`flash_api.cpp_kw.md_docs.md`](./flash_api.cpp_kw.md_docs.md)
- [`philox.cuh_docs.md_docs.md`](./philox.cuh_docs.md_docs.md)
- [`flash_api.cpp_docs.md_docs.md`](./flash_api.cpp_docs.md_docs.md)
- [`static_switch.h_docs.md_docs.md`](./static_switch.h_docs.md_docs.md)
- [`static_switch.h_kw.md_docs.md`](./static_switch.h_kw.md_docs.md)
- [`flash_api.h_kw.md_docs.md`](./flash_api.h_kw.md_docs.md)
- [`philox.cuh_kw.md_docs.md`](./philox.cuh_kw.md_docs.md)


## Cross-References

- **File Documentation**: `flash_api.h_docs.md_docs.md`
- **Keyword Index**: `flash_api.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
