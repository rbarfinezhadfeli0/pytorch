# Documentation: `docs/aten/src/ATen/native/IndexingUtils.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/IndexingUtils.h_docs.md`
- **Size**: 8,957 bytes (8.75 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/IndexingUtils.h`

## File Metadata

- **Path**: `aten/src/ATen/native/IndexingUtils.h`
- **Size**: 6,292 bytes (6.14 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <ATen/ExpandUtils.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/core/IListRef.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/nonzero.h>
#endif

namespace at::native {

[[noreturn]]
static void invalid_mask(const Tensor & self, int64_t idx, const Tensor & mask, int64_t maskIdx) {
  TORCH_CHECK_INDEX(false, "The shape of the mask ", mask.sizes(), " at index ", maskIdx,
  " does not match the shape of the indexed tensor ", self.sizes(), " at index ", idx);
}

[[maybe_unused]] static std::vector<Tensor> expandTensors(
    const Tensor& self,
    IOptTensorListRef indices,
    bool ensure_same_device = false) {
  // If indices come in as ByteTensor or BoolTensor (masks), expand them into
  // the equivalent indexing by LongTensors
  std::vector<Tensor> result;
  for (const auto& index_opt : indices) {
    if (!index_opt.has_value()) {
      result.emplace_back();
    } else {
      const auto& index = *index_opt;
      if (index.scalar_type() == kByte || index.scalar_type() == kBool) {
        if (index.scalar_type() == kByte) {
          TORCH_WARN("indexing with dtype torch.uint8 is now deprecated," \
          " please use a dtype torch.bool instead.");
        }
        // The sizes of the ByteTensor mask or bool tensor must match the sizes of the
        // corresponding dimensions in self
        for (const auto j : c10::irange(index.dim())) {
          int64_t srcIdx = static_cast<int64_t>(result.size() + j);
          if (index.size(j) != self.size(srcIdx)) {
            invalid_mask(self, srcIdx, index, j);
          }
        }
        // Replace with nonzeros
        at::Tensor nonzero;
        if (ensure_same_device && index.device() != self.device()) {
          bool non_blocking = index.is_cpu() && self.device().is_cuda();
          auto out = at::empty({0}, index.options().dtype(kLong).pinned_memory(non_blocking));
          nonzero = at::nonzero_out(out, index).to(self.device(), non_blocking);
        } else {
          nonzero = index.nonzero();
        }
        for (const auto j : c10::irange(index.dim())) {
          result.emplace_back(nonzero.select(1, j));
        }
      } else if (ensure_same_device && index.device() != self.device()) {
        result.emplace_back(index.to(self.device()));
      } else {
        result.emplace_back(index);
      }
    }
  }
  return result;
}

[[maybe_unused]] static void checkIndexTensorTypes(
    IOptTensorListRef indices,
    bool allow_int = false) {
  for (const auto& tensor : indices) {
    if (tensor.has_value() && tensor->defined()) {
      auto scalarType = tensor->scalar_type();
      if (allow_int) {
        if (scalarType != kLong && scalarType != kByte && scalarType != kBool && scalarType != kInt) {
            TORCH_CHECK_INDEX(false, "tensors used as indices must be long, int, byte or bool tensors");
        }
      } else {
        if (scalarType != kLong && scalarType != kByte && scalarType != kBool) {
            TORCH_CHECK_INDEX(false, "tensors used as indices must be long, byte or bool tensors");
        }
      }
    }
  }
}

inline torch::List<std::optional<Tensor>> toListOfOptionalTensors(ArrayRef<Tensor> list) {
  torch::List<std::optional<Tensor>> result;
  result.reserve(list.size());
  for (const Tensor& a : list) {
    result.push_back(a);
  }
  return result;
}

inline torch::List<std::optional<Tensor>> toListOfOptionalTensors(ArrayRef<IValue> list) {
  torch::List<std::optional<Tensor>> result;
  result.reserve(list.size());
  for (const IValue& a : list) {
    result.push_back(a.isTensor() ? std::optional<Tensor>(a.toTensor()) : std::optional<Tensor>());
  }
  return result;
}

[[maybe_unused]] static bool hasContiguousSubspace(TensorList tl) {
  // true if all the non-null tensors are adjacent
  auto isDefined = [](const Tensor & tensor){ return tensor.defined(); };
  auto isNull = [](const Tensor & tensor){ return !tensor.defined(); };
  auto start = std::find_if(tl.begin(), tl.end(), isDefined);
  auto stop = std::find_if(tl.rbegin(), tl.rend(), isDefined);
  auto it = std::find_if(start, stop.base(), isNull);
  return it == stop.base();
}

// Transposes the tensor and indices together so that all the non-null indices
// index the first k dimensions of the tensor. Returns the transposed tensor
// and the reordered indices. For example:
// transposeToFront(tensor, {nullptr, a, nullptr, b})
// returns
// tensor.permute([1, 3, 0, 2]), {a, b, nullptr, nullptr}
[[maybe_unused]] static std::tuple<Tensor, std::vector<Tensor>> transposeToFront(
    const Tensor& self,
    TensorList indices) {
  std::vector<int64_t> dims;
  std::vector<Tensor> transposedIndices;
  dims.reserve(self.dim());
  for (const auto i : c10::irange(self.dim())) {
    if (indices[i].defined()) {
      dims.push_back(i);
      transposedIndices.emplace_back(indices[i]);
    }
  }
  for (const auto i : c10::irange(self.dim())) {
    if (!indices[i].defined()) {
      dims.push_back(i);
      transposedIndices.emplace_back();
    }
  }
  return std::make_tuple(self.permute(dims), std::move(transposedIndices));
}

inline std::tuple<Tensor, std::vector<Tensor>, std::vector<int64_t>>
transposeToFrontAndInvPerm(const Tensor& self, TensorList indices) {
  std::vector<int64_t> dims;
  std::vector<int64_t> invPerm;
  std::vector<Tensor> transposedIndices;
  dims.reserve(self.dim());
  invPerm.resize(self.dim());
  for (const auto i : c10::irange(self.dim())) {
    if (indices[i].defined()) {
      dims.push_back(i);
      transposedIndices.emplace_back(indices[i]);
    }
  }
  for (const auto i : c10::irange(self.dim())) {
    if (!indices[i].defined()) {
      dims.push_back(i);
      transposedIndices.emplace_back();
    }
  }
  for (const auto i : c10::irange(self.dim())) {
    invPerm[dims[i]] = i;
  }
  return std::make_tuple(self.permute(dims), std::move(transposedIndices), std::move(invPerm));
}

struct AdvancedIndex {
  AdvancedIndex(const Tensor& src, TensorList indices);

  Tensor src;
  std::vector<Tensor> indices;
  DimVector indexed_sizes;
  DimVector indexed_strides;
  int64_t dims_before;
  int64_t dims_after;
};


} //namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `AdvancedIndex`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ExpandUtils.h`
- `ATen/native/CanUse32BitIndexMath.h`
- `ATen/native/TensorIterator.h`
- `ATen/core/IListRef.h`
- `c10/util/irange.h`
- `ATen/Functions.h`
- `ATen/ops/empty.h`
- `ATen/ops/nonzero.h`


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

- **File Documentation**: `IndexingUtils.h_docs.md`
- **Keyword Index**: `IndexingUtils.h_kw.md`
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

- **File Documentation**: `IndexingUtils.h_docs.md_docs.md`
- **Keyword Index**: `IndexingUtils.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
