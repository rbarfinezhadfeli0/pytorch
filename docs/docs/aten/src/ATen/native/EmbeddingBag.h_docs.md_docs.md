# Documentation: `docs/aten/src/ATen/native/EmbeddingBag.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/EmbeddingBag.h_docs.md`
- **Size**: 7,898 bytes (7.71 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/EmbeddingBag.h`

## File Metadata

- **Path**: `aten/src/ATen/native/EmbeddingBag.h`
- **Size**: 5,228 bytes (5.11 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <cstdint>

#ifdef USE_FBGEMM
#include <fbgemm/FbgemmEmbedding.h>
#endif

namespace at::native {

enum class EmbeddingBagMode {
  SUM = 0,
  MEAN = 1,
  MAX = 2,
};

[[maybe_unused]] static bool operator==(int64_t op1, EmbeddingBagMode op2) {
  return op1 == static_cast<int64_t>(op2);
}

[[maybe_unused]] static bool operator!=(int64_t op1, EmbeddingBagMode op2) {
  return !(op1 == op2);
}

void check_arguments(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const int64_t mode,
    const std::optional<Tensor>& per_sample_weights,
    bool include_last_offset);

void make_bag_size_out(
    Tensor& bag_size_out,
    const Tensor& offsets,
    const Tensor& indices,
    const int64_t mode,
    const bool include_last_offset,
    const bool requires_grad);

void make_max_indices_out(
    Tensor& max_indices_out,
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& bag_size,
    const int64_t mode,
    bool include_last_offset);

void make_offset2bag_out(
    Tensor& offset2bag,
    Tensor& output,
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const int64_t mode,
    const std::optional<Tensor>& per_sample_weights,
    const int64_t padding_idx = -1);

#ifdef USE_FBGEMM

template<bool has_weight, typename TIndex, typename TData>
struct _CallbackAndBlockSize {
    using TCallback = typename fbgemm::EmbeddingSpMDMKernelSignature<TData, TIndex, TIndex, TData>::Type;

    int64_t blockSize = -1;
    TCallback callback = nullptr;

    static TCallback generateCallback(int64_t block_size) {
        return fbgemm::GenerateEmbeddingSpMDM<TData, TIndex, TIndex, TData>(
                block_size,
                has_weight,
                /* normalize_by_lengths */false,
                /* prefetch */16,
                /* is_weight_positional */false,
                /* use_offsets */true);
    }

    _CallbackAndBlockSize() = default;

    explicit _CallbackAndBlockSize(std::optional<int64_t> maybe_block_size)
      : blockSize(maybe_block_size.value_or(-1))
      , callback(maybe_block_size.has_value() ? generateCallback(maybe_block_size.value()) : nullptr)
    {}
};

template<typename... StorageMixins>
struct _EmbeddingBagKernelCacheImpl : private StorageMixins... {

    _EmbeddingBagKernelCacheImpl() = default;
    // use each of the mixins to store corresponding kernel and block size
    explicit _EmbeddingBagKernelCacheImpl(std::optional<int64_t> maybe_block_size)
      : StorageMixins(maybe_block_size)...
    {}

    // this method is thread safe (call sites may call from different threads)
    template<bool has_weight, typename TIndex, typename TData>
    typename _CallbackAndBlockSize<has_weight, TIndex, TData>::TCallback
    getCallback(int64_t block_size) const {
        // if the cache doesn't store the kernel for the incoming block size
        // (so it is different from the one stored in corresponding mixin)
        // regenerate the kernel (not writing it into the cache so we avoid locks)
        if (block_size != _CallbackAndBlockSize<has_weight, TIndex, TData>::blockSize) {
            return _CallbackAndBlockSize<has_weight, TIndex, TData>::generateCallback(block_size);
        }
        // else retrieve the cached kernel from the corresponding mixin
        return _CallbackAndBlockSize<has_weight, TIndex, TData>::callback;
    }
};

// instantiate the cache with the list of storage mixins
// for each of the 8 _EmbeddingBagKernelCache* usages in the EmbeddingBag.cpp impl file
using _EmbeddingBagKernelCache = _EmbeddingBagKernelCacheImpl<
    _CallbackAndBlockSize<true, int32_t, float>,
    _CallbackAndBlockSize<false, int32_t, float>,
    _CallbackAndBlockSize<true, int64_t, float>,
    _CallbackAndBlockSize<false, int64_t, float>,
    _CallbackAndBlockSize<true, int32_t, unsigned short>,
    _CallbackAndBlockSize<false, int32_t, unsigned short>,
    _CallbackAndBlockSize<true, int64_t, unsigned short>,
    _CallbackAndBlockSize<false, int64_t, unsigned short>>;
#else
struct _EmbeddingBagKernelCache {
    explicit _EmbeddingBagKernelCache(std::optional<int64_t> /* maybe_block_size */) {}
};
#endif

void _embedding_bag_cpu_impl_out(Tensor& output, Tensor& offset2bag,
    Tensor& bag_size, Tensor* max_indices,
    const Tensor &weight, const Tensor &indices,
    const Tensor &offsets, const int64_t mode = 0,
    const std::optional<Tensor>& per_sample_weights = std::nullopt,
    bool include_last_offset = false,
    int64_t padding_idx = -1,
    _EmbeddingBagKernelCache* fbgemm_kernel_cache = nullptr);

void _embedding_bag_cpu_out(
    at::Tensor& output,
    at::Tensor& offset2bag,
    at::Tensor& bag_size,
    at::Tensor* p_max_indices,
    const at::Tensor& weight,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    const bool scale_grad_by_freq,
    const int64_t mode,
    const bool sparse,
    const std::optional<at::Tensor>& per_sample_weights,
    const bool include_last_offset,
    const std::optional<int64_t>& padding_idx,
    _EmbeddingBagKernelCache* fbgemm_kernel_cache = nullptr);

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 13 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `EmbeddingBagMode`, `_CallbackAndBlockSize`, `_EmbeddingBagKernelCacheImpl`, `_EmbeddingBagKernelCache`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/Config.h`
- `cstdint`
- `fbgemm/FbgemmEmbedding.h`


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

Files in the same folder (`aten/src/ATen/native`):

- [`LossMulti.h_docs.md`](./LossMulti.h_docs.md)
- [`NaiveConvolutionTranspose3d.cpp_docs.md`](./NaiveConvolutionTranspose3d.cpp_docs.md)
- [`UnaryOps.cpp_docs.md`](./UnaryOps.cpp_docs.md)
- [`ResizeCommon.h_docs.md`](./ResizeCommon.h_docs.md)
- [`FusedAdagrad.cpp_docs.md`](./FusedAdagrad.cpp_docs.md)
- [`SharedReduceOps.h_docs.md`](./SharedReduceOps.h_docs.md)
- [`SpectralOpsUtils.h_docs.md`](./SpectralOpsUtils.h_docs.md)
- [`FractionalMaxPooling.h_docs.md`](./FractionalMaxPooling.h_docs.md)
- [`TensorDimApply.h_docs.md`](./TensorDimApply.h_docs.md)
- [`Lerp.cpp_docs.md`](./Lerp.cpp_docs.md)


## Cross-References

- **File Documentation**: `EmbeddingBag.h_docs.md`
- **Keyword Index**: `EmbeddingBag.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native`):

- [`AdaptiveMaxPooling2d.cpp_docs.md_docs.md`](./AdaptiveMaxPooling2d.cpp_docs.md_docs.md)
- [`Distributions.cpp_docs.md_docs.md`](./Distributions.cpp_docs.md_docs.md)
- [`im2col_shape_check.h_docs.md_docs.md`](./im2col_shape_check.h_docs.md_docs.md)
- [`ReduceOps.cpp_kw.md_docs.md`](./ReduceOps.cpp_kw.md_docs.md)
- [`Lerp.cpp_kw.md_docs.md`](./Lerp.cpp_kw.md_docs.md)
- [`CPUFallback.h_docs.md_docs.md`](./CPUFallback.h_docs.md_docs.md)
- [`MetaTensor.cpp_docs.md_docs.md`](./MetaTensor.cpp_docs.md_docs.md)
- [`Correlation.cpp_kw.md_docs.md`](./Correlation.cpp_kw.md_docs.md)
- [`im2col_shape_check.h_kw.md_docs.md`](./im2col_shape_check.h_kw.md_docs.md)
- [`UpSampleNearest2d.cpp_kw.md_docs.md`](./UpSampleNearest2d.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `EmbeddingBag.h_docs.md_docs.md`
- **Keyword Index**: `EmbeddingBag.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
