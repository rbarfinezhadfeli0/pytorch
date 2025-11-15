# Documentation: `aten/src/ATen/cuda/detail/TensorInfo.cuh`

## File Metadata

- **Path**: `aten/src/ATen/cuda/detail/TensorInfo.cuh`
- **Size**: 3,239 bytes (3.16 KB)
- **Type**: CUDA Header File
- **Extension**: `.cuh`

## File Purpose

This is a cuda header file that is part of the PyTorch project.

## Original Source

```
#pragma once

#include <ATen/CollapseDims.h>

namespace at::cuda::detail {

#define MAX_TENSORINFO_DIMS 25

// CUDA kernel argument that defines tensor layout
template <typename T, typename IndexType>
struct TensorInfo {
  TensorInfo();
  TensorInfo(T* p,
             int dim,
             IndexType sz[MAX_TENSORINFO_DIMS],
             IndexType st[MAX_TENSORINFO_DIMS]);

  // Set the size of the given dimension to 1, as if it were a
  // reduction dim (allows you to calculate offsets of the reduction
  // slice)
  void reduceDim(int dim);

  // See note on [collapse dims].
  int collapseDims(const int excludeDim = -1);

  // Contiguous tensors of more than one dimension are collapsed down
  // to one tensor
  __host__ __device__ inline bool isContiguous() const {
    return (dims == 1 && strides[0] == 1);
  }

  T* data;
  IndexType sizes[MAX_TENSORINFO_DIMS];
  IndexType strides[MAX_TENSORINFO_DIMS];
  int dims;
};

template <typename T, typename IndexType>
TensorInfo<T, IndexType>::TensorInfo() {
  data = nullptr;
  dims = 0;
}

template <typename T, typename IndexType>
TensorInfo<T, IndexType>::TensorInfo(T* p,
                                     int dim,
                                     IndexType sz[MAX_TENSORINFO_DIMS],
                                     IndexType st[MAX_TENSORINFO_DIMS]) {
  data = p;
  dims = dim;
  TORCH_CHECK(dims < MAX_TENSORINFO_DIMS, "CUDA Tensors cannot have more than 25 dimensions");

  for (int i = 0; i < dim; ++i) {
    sizes[i] = sz[i];
    strides[i] = st[i];
  }
}

template <typename T, typename IndexType>
void
TensorInfo<T, IndexType>::reduceDim(int dim) {
  TORCH_CHECK(dim < dims && dim >= 0, "expected dim between 0 and dims - 1");
  sizes[dim] = 1;
}

template <typename T, typename IndexType>
int
TensorInfo<T, IndexType>::collapseDims(const int excludeDim) {
  auto result = at::collapse_dims(sizes, strides, dims, excludeDim);
  dims = std::get<1>(result);
  return std::get<0>(result);
}

// Translate a linear index for the apply to a T* offset;
// specialized on `Dims` to reduce nvcc compilation time
template <typename T, typename IndexType, int Dims>
struct IndexToOffset {
  static __host__ __device__ IndexType get(
    IndexType linearId,
    const TensorInfo<T, IndexType>& info) {

    IndexType offset = 0;

    // Uses static dims
    for (int i = Dims - 1; i > 0; --i) {
      IndexType curDimIndex = linearId % info.sizes[i];
      IndexType curDimOffset = curDimIndex * info.strides[i];
      offset += curDimOffset;
      linearId /= info.sizes[i];
    }

    return offset + linearId * info.strides[0];
  }
};

// Uses dynamic (runtime) instead of static (compiletime) dims
template <typename T, typename IndexType>
struct IndexToOffset<T, IndexType, -1> {
  static inline __host__ __device__ IndexType get(
    IndexType linearId,
    const TensorInfo<T, IndexType>& info) {

      IndexType offset = 0;

      for (int i = info.dims - 1; i > 0; --i) {
        IndexType curDimIndex = linearId % info.sizes[i];
        IndexType curDimOffset = curDimIndex * info.strides[i];
        offset += curDimOffset;
        linearId /= info.sizes[i];
      }

      return offset + linearId * info.strides[0];
  }
};

} // namespace at::cuda::detail

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/cuda/detail`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/cuda/detail`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/CollapseDims.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`aten/src/ATen/cuda/detail`):

- [`BLASConstants.cu_docs.md`](./BLASConstants.cu_docs.md)
- [`BLASConstants.h_docs.md`](./BLASConstants.h_docs.md)
- [`KernelUtils.h_docs.md`](./KernelUtils.h_docs.md)
- [`OffsetCalculator.cuh_docs.md`](./OffsetCalculator.cuh_docs.md)
- [`CUDAHooks.h_docs.md`](./CUDAHooks.h_docs.md)
- [`IndexUtils.cu_docs.md`](./IndexUtils.cu_docs.md)
- [`LazyNVRTC.cpp_docs.md`](./LazyNVRTC.cpp_docs.md)
- [`LazyNVRTC.h_docs.md`](./LazyNVRTC.h_docs.md)
- [`UnpackRaw.cuh_docs.md`](./UnpackRaw.cuh_docs.md)
- [`DeviceThreadHandles.h_docs.md`](./DeviceThreadHandles.h_docs.md)


## Cross-References

- **File Documentation**: `TensorInfo.cuh_docs.md`
- **Keyword Index**: `TensorInfo.cuh_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
