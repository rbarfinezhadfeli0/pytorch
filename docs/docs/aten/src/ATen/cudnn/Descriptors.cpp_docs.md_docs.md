# Documentation: `docs/aten/src/ATen/cudnn/Descriptors.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/cudnn/Descriptors.cpp_docs.md`
- **Size**: 9,026 bytes (8.81 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/cudnn/Descriptors.cpp`

## File Metadata

- **Path**: `aten/src/ATen/cudnn/Descriptors.cpp`
- **Size**: 6,755 bytes (6.60 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/cudnn/Descriptors.h>

#include <ATen/ATen.h>
#include <c10/util/irange.h>

#include <array>
#include <iostream>
#include <sstream>

// NOLINTBEGIN(*c-arrays*)
namespace at::native {

namespace {

inline cudnnDataType_t getDataType(const at::Tensor& t) {
  auto scalar_type = t.scalar_type();
  if (scalar_type == at::kFloat) {
    return CUDNN_DATA_FLOAT;
  } else if (scalar_type == at::kHalf) {
    return CUDNN_DATA_HALF;
  } else if (scalar_type == at::kDouble) {
    return CUDNN_DATA_DOUBLE;
  }
    else if (scalar_type == at::kBFloat16) {
    return CUDNN_DATA_BFLOAT16;
  } else if (scalar_type == at::kQInt8) {
    return CUDNN_DATA_INT8;
  }
  TORCH_CHECK(false, "TensorDescriptor does not support ", scalar_type);
}

} // anonymous namespace

void RNNDataDescriptor::set(const at::Tensor &t, const cudnnRNNDataLayout_t layout, const int maxSeqLength, const int batchSize, const int vectorSize, const int* seqLengthArray) {
  set(getDataType(t), layout, maxSeqLength, batchSize, vectorSize, seqLengthArray);
}

void TensorDescriptor::set(const at::Tensor &t, at::MemoryFormat memory_format, size_t pad) {
  set(getDataType(t), t.sizes(), t.strides(), pad,
    memory_format == at::MemoryFormat::ChannelsLast ||
    memory_format == at::MemoryFormat::ChannelsLast3d);
}

void TensorDescriptor::set(const at::Tensor &t, size_t pad) {
  auto memory_format = t.suggest_memory_format();
  set(getDataType(t), t.sizes(), t.strides(), pad,
    memory_format == at::MemoryFormat::ChannelsLast ||
    memory_format == at::MemoryFormat::ChannelsLast3d);
}

void TensorDescriptor::set(cudnnDataType_t datatype, IntArrayRef t_sizes, IntArrayRef t_strides, size_t pad) {
  set(datatype, t_sizes, t_strides, pad,
    is_channels_last_strides_2d(t_sizes, t_strides) ||
    is_channels_last_strides_3d(t_sizes, t_strides));
}

void TensorDescriptor::set(cudnnDataType_t datatype, IntArrayRef t_sizes, IntArrayRef t_strides, size_t pad, bool nhwc) {
  size_t dim = t_sizes.size();
  if (dim > CUDNN_DIM_MAX || pad > CUDNN_DIM_MAX)
    TORCH_CHECK(false, "cuDNN supports only up to ", CUDNN_DIM_MAX, " dimensions");
  int size[CUDNN_DIM_MAX];
  int stride[CUDNN_DIM_MAX];
  for (const auto i : c10::irange(dim)) {
    size[i] = static_cast<int>(t_sizes[i]);
    stride[i] = static_cast<int>(t_strides[i]);
  }
  for (const auto i : c10::irange(dim, pad)) {
    size[i] = 1;
    stride[i] = 1;
  }
  set(datatype, static_cast<int>(std::max(dim, pad)), size, stride, nhwc);
}

std::string cudnnTypeToString(cudnnDataType_t dtype) {
  switch (dtype) {
    case CUDNN_DATA_FLOAT:
      return "CUDNN_DATA_FLOAT";
    case CUDNN_DATA_DOUBLE:
      return "CUDNN_DATA_DOUBLE";
    case CUDNN_DATA_HALF:
      return "CUDNN_DATA_HALF";
    case CUDNN_DATA_BFLOAT16:
      return "CUDNN_DATA_BFLOAT16";
    case CUDNN_DATA_INT8:
      return "CUDNN_DATA_INT8";
    case CUDNN_DATA_INT32:
      return "CUDNN_DATA_INT32";
    case CUDNN_DATA_INT8x4:
      return "CUDNN_DATA_INT8x4";
    case CUDNN_DATA_UINT8:
      return "CUDNN_DATA_UINT8";
    case CUDNN_DATA_UINT8x4:
      return "CUDNN_DATA_UINT8x4";
    default:
      std::ostringstream oss;
      oss << "(unknown data-type " << static_cast<int>(dtype) << ")";
      return oss.str();
  }
}

std::ostream& operator<<(std::ostream & out, const TensorDescriptor& d) {
  out << "TensorDescriptor " << static_cast<void*>(d.desc()) << "\n";
  int nbDims = 0;
  int dimA[CUDNN_DIM_MAX];
  int strideA[CUDNN_DIM_MAX];
  cudnnDataType_t dtype{};
  cudnnGetTensorNdDescriptor(d.desc(), CUDNN_DIM_MAX, &dtype, &nbDims, dimA, strideA);
  out << "    type = " << cudnnTypeToString(dtype) << "\n";
  out << "    nbDims = " << nbDims << "\n";
  // Read out only nbDims of the arrays!
  out << "    dimA = ";
  for (auto i : ArrayRef<int>{dimA, static_cast<size_t>(nbDims)}) {
    out << i << ", ";
  }
  out << "\n";
  out << "    strideA = ";
  for (auto i : ArrayRef<int>{strideA, static_cast<size_t>(nbDims)}) {
    out << i << ", ";
  }
  out << "\n";
  return out;
}

void TensorDescriptor::print() { std::cout << *this; }

void FilterDescriptor::set(const at::Tensor &t, const at::MemoryFormat memory_format, int64_t pad) {
  auto dim = t.ndimension();
  if (dim > CUDNN_DIM_MAX || pad > CUDNN_DIM_MAX)
  TORCH_CHECK(false, "cuDNN supports only up to ", CUDNN_DIM_MAX, " dimensions");
  // NB: It is possible for this test to be insufficient, because the
  // Tensor passed in to set the filter descriptor may not be the actual
  // Tensor whose data pointer is passed to cuDNN.  Nevertheless,
  // that is the common case, so we can catch most client errors with this test.
  TORCH_CHECK(t.is_contiguous(memory_format),
    "cuDNN filters (a.k.a. weights) must be contiguous in desired memory_format\n",
    "Weight sizes: ", t.sizes(), "\n",
    "Weight strides: ", t.strides(), "\n",
    "cuDNN suggested memory_format: ", memory_format);

  std::array<int, CUDNN_DIM_MAX> size;
  for (const auto i : c10::irange(dim)) {
    size[i] = static_cast<int>(t.size(i));
  }
  for (const auto i : c10::irange(dim, pad)) {
    size[i] = 1;
  }
  dim = std::max(dim, pad);
  cudnnTensorFormat_t filter_format{};
  switch(memory_format) {
    case at::MemoryFormat::Contiguous:
      filter_format = CUDNN_TENSOR_NCHW;
      break;
    case at::MemoryFormat::ChannelsLast:
    case at::MemoryFormat::ChannelsLast3d:
      filter_format = CUDNN_TENSOR_NHWC;
      break;
    default:
      TORCH_INTERNAL_ASSERT(false, "unsupported memory_format for cuDNN filters");
  }
  set(getDataType(t), static_cast<int>(dim), size.data(), filter_format);
}

std::string cudnnMemoryFormatToString(cudnnTensorFormat_t tformat) {
  switch (tformat) {
    case CUDNN_TENSOR_NCHW:
      return "CUDNN_TENSOR_NCHW";
    case CUDNN_TENSOR_NHWC:
      return "CUDNN_TENSOR_NHWC";
    default:
      std::ostringstream oss;
      oss << "(unknown cudnn tensor format " << static_cast<int>(tformat) << ")";
      return oss.str();
  }
}

std::ostream& operator<<(std::ostream & out, const FilterDescriptor& d) {
  out << "FilterDescriptor " << static_cast<void*>(d.desc()) << "\n";
  int nbDims = 0;
  int dimA[CUDNN_DIM_MAX];
  cudnnDataType_t dtype{};
  cudnnTensorFormat_t tformat{};
  cudnnGetFilterNdDescriptor(d.desc(), CUDNN_DIM_MAX, &dtype, &tformat, &nbDims, dimA);
  out << "    type = " << cudnnTypeToString(dtype) << "\n";
  out << "    tensor_format = " << cudnnMemoryFormatToString(tformat) << "\n";
  out << "    nbDims = " << nbDims << "\n";
  // Read out only nbDims of the arrays!
  out << "    dimA = ";
  for (auto i : ArrayRef<int>{dimA, static_cast<size_t>(nbDims)}) {
    out << i << ", ";
  }
  out << "\n";
  return out;
}

void FilterDescriptor::print() { std::cout << *this; }

}
// NOLINTEND(*c-arrays*)

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `void`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/cudnn`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/cudnn/Descriptors.h`
- `ATen/ATen.h`
- `c10/util/irange.h`
- `array`
- `iostream`
- `sstream`


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

Files in the same folder (`aten/src/ATen/cudnn`):

- [`Handle.h_docs.md`](./Handle.h_docs.md)
- [`Handles.h_docs.md`](./Handles.h_docs.md)
- [`Types.h_docs.md`](./Types.h_docs.md)
- [`Descriptors.h_docs.md`](./Descriptors.h_docs.md)
- [`Utils.h_docs.md`](./Utils.h_docs.md)
- [`Handle.cpp_docs.md`](./Handle.cpp_docs.md)
- [`AutocastRNN.cpp_docs.md`](./AutocastRNN.cpp_docs.md)
- [`Types.cpp_docs.md`](./Types.cpp_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)


## Cross-References

- **File Documentation**: `Descriptors.cpp_docs.md`
- **Keyword Index**: `Descriptors.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/cudnn`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/cudnn`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/cudnn`):

- [`Handle.cpp_docs.md_docs.md`](./Handle.cpp_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`Descriptors.h_kw.md_docs.md`](./Descriptors.h_kw.md_docs.md)
- [`Types.cpp_kw.md_docs.md`](./Types.cpp_kw.md_docs.md)
- [`Handles.h_docs.md_docs.md`](./Handles.h_docs.md_docs.md)
- [`Handle.h_kw.md_docs.md`](./Handle.h_kw.md_docs.md)
- [`cudnn-wrapper.h_kw.md_docs.md`](./cudnn-wrapper.h_kw.md_docs.md)
- [`AutocastRNN.cpp_docs.md_docs.md`](./AutocastRNN.cpp_docs.md_docs.md)
- [`Handles.h_kw.md_docs.md`](./Handles.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `Descriptors.cpp_docs.md_docs.md`
- **Keyword Index**: `Descriptors.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
