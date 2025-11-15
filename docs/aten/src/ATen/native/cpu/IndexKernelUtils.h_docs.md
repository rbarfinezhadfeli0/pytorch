# Documentation: `aten/src/ATen/native/cpu/IndexKernelUtils.h`

## File Metadata

- **Path**: `aten/src/ATen/native/cpu/IndexKernelUtils.h`
- **Size**: 2,927 bytes (2.86 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <ATen/native/TensorIterator.h>
#include <c10/util/irange.h>

namespace at::native {

inline bool is_constant_index(int ntensor, const int64_t* strides) {
  AT_ASSERT(ntensor >= 3);
  for (const auto arg : c10::irange(2, ntensor)) {
    if (strides[arg] != 0) {
      return false;
    }
  }
  return true;
}


struct Indexer {
  Indexer(int64_t num_indexers, char** indexers, const int64_t* indexer_strides,
          IntArrayRef original_sizes, IntArrayRef original_strides)
    : num_indexers(num_indexers)
    , indexers(indexers)
    , indexer_strides(indexer_strides)
    , original_strides(original_strides.data())
    , original_sizes(original_sizes.data()) {
    AT_ASSERT(static_cast<int64_t>(original_strides.size()) == num_indexers);
    AT_ASSERT(static_cast<int64_t>(original_sizes.size()) == num_indexers);
  }

  int64_t num_indexers;
  char** indexers;
  const int64_t* indexer_strides;
  const int64_t* original_strides;
  const int64_t* original_sizes;

  int64_t get(int64_t idx) {
    int64_t offset = 0;
    for (const auto j : c10::irange(num_indexers)) {
      int64_t value = *(int64_t*)&indexers[j][idx * indexer_strides[j]];
      int64_t size = original_sizes[j];
      TORCH_CHECK_INDEX(value >= -size && value < size,
                        "index ", value, " is out of bounds for dimension ", j, " with size ", size);
      if (value < 0) {
        value += size;
      }
      offset += value * original_strides[j];
    }
    return offset;
  }
};

template <typename scalar_t, typename func_t>
void cpu_index_kernel(TensorIteratorBase& iter, IntArrayRef index_size, IntArrayRef index_stride,
                      const func_t& f, bool serial_execution=false)
{
  int ntensor = iter.ntensors();
  // When launch the index parallel version, set a relative small grain size less than the INTERNAL::GRAIN_SIZE
  // to make the whole available thread numbers get more balanced work load and a better cache location.
  // The grain size here is chosen by the op benchmark to overcome the thread launch overhead
  const int index_parallel_grain_size = 3000;
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    auto indexer = Indexer(ntensor - 2, &data[2], &strides[2], index_size, index_stride);
    char* dst = data[0];
    char* src = data[1];
    if (is_constant_index(ntensor, strides)) {
      // specialization for when every element uses the same index
      int64_t offset = indexer.get(0);
      for (const auto i : c10::irange(n)) {
        f(dst + strides[0] * i, src + strides[1] * i, offset);
      }
    } else {
      for (const auto i : c10::irange(n)) {
        int64_t offset = indexer.get(i);
        f(dst + strides[0] * i, src + strides[1] * i, offset);
      }
    }
  };
  if (serial_execution) {
    iter.serial_for_each(loop, {0, iter.numel()});
  } else {
    iter.for_each(loop, index_parallel_grain_size);
  }
}
} // at
// native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `Indexer`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/TensorIterator.h`
- `c10/util/irange.h`


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

Files in the same folder (`aten/src/ATen/native/cpu`):

- [`UpSampleKernelAVXAntialias.h_docs.md`](./UpSampleKernelAVXAntialias.h_docs.md)
- [`SparseFactories.cpp_docs.md`](./SparseFactories.cpp_docs.md)
- [`UnfoldBackwardKernel.cpp_docs.md`](./UnfoldBackwardKernel.cpp_docs.md)
- [`int8mm_kernel.cpp_docs.md`](./int8mm_kernel.cpp_docs.md)
- [`LerpKernel.cpp_docs.md`](./LerpKernel.cpp_docs.md)
- [`UpSampleKernel.cpp_docs.md`](./UpSampleKernel.cpp_docs.md)
- [`scaled_modified_bessel_k0.cpp_docs.md`](./scaled_modified_bessel_k0.cpp_docs.md)
- [`DistributionKernels.cpp_docs.md`](./DistributionKernels.cpp_docs.md)
- [`CopyKernel.cpp_docs.md`](./CopyKernel.cpp_docs.md)
- [`SampledAddmmKernel.cpp_docs.md`](./SampledAddmmKernel.cpp_docs.md)


## Cross-References

- **File Documentation**: `IndexKernelUtils.h_docs.md`
- **Keyword Index**: `IndexKernelUtils.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
