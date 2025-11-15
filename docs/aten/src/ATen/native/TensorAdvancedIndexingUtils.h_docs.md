# Documentation: `aten/src/ATen/native/TensorAdvancedIndexingUtils.h`

## File Metadata

- **Path**: `aten/src/ATen/native/TensorAdvancedIndexingUtils.h`
- **Size**: 3,113 bytes (3.04 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/TensorIterator.h>

namespace at::native {
namespace {
#ifndef STRIP_ERROR_MESSAGES
inline std::string shapes_as_str(TensorList tensors) {
  std::ostringstream os;
  bool first = true;
  for (auto& tensor : tensors) {
    if (tensor.defined()) {
      if (!first) {
        os << ", ";
      }
      os << tensor.sizes();
      first = false;
    }
  }
  return os.str();
}
#endif
} // anonymous namespace

inline std::tuple<bool, Tensor> canDispatchToMaskedFill(
    const Tensor& self,
    const torch::List<std::optional<at::Tensor>>& indices,
    const Tensor& value) {
  if (!(value.numel() == 1 && value.device().is_cpu())) {
    return std::make_tuple(false, Tensor());
  }
  int64_t num_ind = 0;
  Tensor mask;
  auto self_device = self.device();
  for (const std::optional<Tensor>& i : indices) {
    if (!i.has_value() || !(*i).defined()) {
      if (!mask.defined()) {
        num_ind++;
      }
    } else {
      const Tensor& index = *i;
      if ((index.scalar_type() != kByte && index.scalar_type() != kBool) ||
          index.device() != self_device || mask.defined()) {
        return std::make_tuple(false, Tensor());
      } else {
        mask = index;
        for (const auto j : c10::irange(index.dim())) {
          int64_t srcIdx = num_ind + j;
          TORCH_CHECK_INDEX(
              index.size(j) == self.size(srcIdx),
              "The shape of the mask ",
              index.sizes(),
              " at index ",
              j,
              " does not match the shape of the indexed tensor ",
              self.sizes(),
              " at index ",
              srcIdx);
        }
        num_ind += mask.ndimension();
      }
    }
  }
  for ([[maybe_unused]] const auto i :
       c10::irange(num_ind, self.ndimension())) {
    mask = mask.unsqueeze(-1);
  }
  return std::make_tuple(true, mask);
}

inline AdvancedIndex make_info(Tensor self, IOptTensorListRef orig) {
  checkIndexTensorTypes(orig, /*allow_int*/ true);
  // first expand BoolTensor (masks) or ByteTensor (masks) into 1 or more
  // LongTensors
  auto indices = expandTensors(self, orig, /*ensure_same_device=*/true);
  // next broadcast all index tensors together
  try {
    indices = expand_outplace(indices);
  } catch (std::exception&) {
    TORCH_CHECK_INDEX(
        false,
        "shape mismatch: indexing tensors could not be broadcast together"
        " with shapes ",
        shapes_as_str(indices));
  }
  // add missing null Tensors so that it matches self.dim()
  while (indices.size() < (size_t)self.dim()) {
    indices.emplace_back();
  }
  // if the non-null indices are not all adjacent, transpose self and indices
  // together so that they're adjacent at the front
  if (!hasContiguousSubspace(indices)) {
    std::tie(self, indices) = transposeToFront(self, indices);
  }
  for (auto& indice : indices) {
    if (indice.defined() && indice.dtype() == at::kInt) {
      indice = indice.to(at::kLong);
    }
  }

  return AdvancedIndex(self, indices);
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `inline`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/native/IndexingUtils.h`
- `ATen/native/TensorIterator.h`


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

- **File Documentation**: `TensorAdvancedIndexingUtils.h_docs.md`
- **Keyword Index**: `TensorAdvancedIndexingUtils.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
