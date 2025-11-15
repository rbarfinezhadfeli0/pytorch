# Documentation: `docs/aten/src/ATen/SparseCsrTensorUtils.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/SparseCsrTensorUtils.h_docs.md`
- **Size**: 20,445 bytes (19.97 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/SparseCsrTensorUtils.h`

## File Metadata

- **Path**: `aten/src/ATen/SparseCsrTensorUtils.h`
- **Size**: 17,774 bytes (17.36 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Operators.h>
#else
#include <ATen/ops/_sparse_compressed_tensor_unsafe.h>
#include <ATen/ops/resize_as_sparse_native.h>
#endif

#define AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(LAYOUT, NAME, ...) \
  [&] {                                                              \
    const auto& the_layout = LAYOUT;                                 \
    switch (the_layout) {                                            \
      case kSparseCsr:                                               \
      case kSparseCsc:                                               \
      case kSparseBsr:                                               \
      case kSparseBsc:                                               \
        return __VA_ARGS__();                                        \
      default:                                                       \
        TORCH_CHECK(                                                 \
            false,                                                   \
            NAME,                                                    \
            " expected sparse compressed tensor layout but got ",    \
            the_layout);                                             \
    }                                                                \
  }()

#define AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(                \
    LAYOUT, NAME, ROW_DIM_ACTION, COLUMN_DIM_ACTION)              \
  [&]() {                                                         \
    const auto& the_layout = LAYOUT;                              \
    switch (the_layout) {                                         \
      case kSparseCsr:                                            \
      case kSparseBsr:                                            \
        return (ROW_DIM_ACTION)();                                \
      case kSparseCsc:                                            \
      case kSparseBsc:                                            \
        return (COLUMN_DIM_ACTION)();                             \
      default:                                                    \
        TORCH_CHECK(                                              \
            false,                                                \
            NAME,                                                 \
            " expected sparse compressed tensor layout but got ", \
            the_layout);                                          \
    }                                                             \
  }()

#define AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(              \
    LAYOUT, NAME, NO_BLOCK_ACTION, BLOCK_ACTION)                  \
  [&]() {                                                         \
    const auto& the_layout = LAYOUT;                              \
    switch (the_layout) {                                         \
      case kSparseCsr:                                            \
      case kSparseCsc:                                            \
        return (NO_BLOCK_ACTION)();                               \
      case kSparseBsr:                                            \
      case kSparseBsc:                                            \
        return (BLOCK_ACTION)();                                  \
      default:                                                    \
        TORCH_CHECK(                                              \
            false,                                                \
            NAME,                                                 \
            " expected sparse compressed tensor layout but got ", \
            the_layout);                                          \
    }                                                             \
  }()

#define AT_DISPATCH_SPARSE_ROW_COMPRESSED_LAYOUTS(                    \
    LAYOUT, NAME, ROW_DIM_ACTION)                                     \
  [&]() {                                                             \
    const auto& the_layout = LAYOUT;                                  \
    switch (the_layout) {                                             \
      case kSparseCsr:                                                \
      case kSparseBsr:                                                \
        return (ROW_DIM_ACTION)();                                    \
      default:                                                        \
        TORCH_CHECK(                                                  \
            false,                                                    \
            NAME,                                                     \
            " expected sparse row compressed tensor layout but got ", \
            the_layout);                                              \
    }                                                                 \
  }()

#define AT_DISPATCH_SPARSE_COL_COMPRESSED_LAYOUTS(                       \
    LAYOUT, NAME, COL_DIM_ACTION)                                        \
  [&]() {                                                                \
    const auto& the_layout = LAYOUT;                                     \
    switch (the_layout) {                                                \
      case kSparseCsc:                                                   \
      case kSparseBsc:                                                   \
        return (COL_DIM_ACTION)();                                       \
      default:                                                           \
        TORCH_CHECK(                                                     \
            false,                                                       \
            NAME,                                                        \
            " expected sparse column compressed tensor layout but got ", \
            the_layout);                                                 \
    }                                                                    \
  }()

#define AT_DISPATCH_SPARSE_COMPRESSED_NONBLOCK_LAYOUTS(LAYOUT, NAME, ACTION)  \
  [&]() {                                                                     \
    const auto& the_layout = LAYOUT;                                          \
    switch (the_layout) {                                                     \
      case kSparseCsr:                                                        \
      case kSparseCsc:                                                        \
        return (ACTION)();                                                    \
      default:                                                                \
        TORCH_CHECK(                                                          \
            false,                                                            \
            NAME,                                                             \
            " expected sparse compressed (non-block) tensor layout but got ", \
            the_layout);                                                      \
    }                                                                         \
  }()

#define AT_DISPATCH_SPARSE_COMPRESSED_BLOCK_LAYOUTS(LAYOUT, NAME, ACTION) \
  [&]() {                                                                 \
    const auto& the_layout = LAYOUT;                                      \
    switch (the_layout) {                                                 \
      case kSparseBsr:                                                    \
      case kSparseBsc:                                                    \
        return (ACTION)();                                                \
      default:                                                            \
        TORCH_CHECK(                                                      \
            false,                                                        \
            NAME,                                                         \
            " expected sparse compressed block tensor layout but got ",   \
            the_layout);                                                  \
    }                                                                     \
  }()

#define AT_DISPATCH_SPARSE_VALUE_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                   \
      TYPE,                                             \
      NAME,                                             \
      AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND4(      \
          kComplexHalf, kHalf, kBool, kBFloat16, __VA_ARGS__))

namespace at::sparse_csr {

// Implements RAII object to manage checking sparse tensor invariants:
class CheckSparseTensorInvariants {
  bool old_state;

 public:
  CheckSparseTensorInvariants(bool state)
      : old_state(at::globalContext().checkSparseTensorInvariants()) {
    at::globalContext().setCheckSparseTensorInvariants(state);
  }
  CheckSparseTensorInvariants(CheckSparseTensorInvariants&& other) = delete;
  CheckSparseTensorInvariants(const CheckSparseTensorInvariants&) = delete;
  CheckSparseTensorInvariants& operator=(const CheckSparseTensorInvariants&) =
      delete;
  CheckSparseTensorInvariants& operator=(CheckSparseTensorInvariants&&) =
      delete;

  ~CheckSparseTensorInvariants() {
    at::globalContext().setCheckSparseTensorInvariants(old_state);
  }
};

using SparseCsrTensor = Tensor;

inline bool is_sparse_compressed(const Layout& layout) {
  switch (layout) {
    case kSparseCsr:
    case kSparseCsc:
    case kSparseBsr:
    case kSparseBsc:
      return true;
    default:;
  }
  return false;
}

inline bool is_sparse_compressed(const Tensor& self) {
  return is_sparse_compressed(self.layout());
}

inline SparseCsrTensorImpl* get_sparse_csr_impl(const SparseCsrTensor& self) {
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(
      self.layout(), "get_sparse_csr_impl", [&] {});
  return static_cast<SparseCsrTensorImpl*>(self.unsafeGetTensorImpl());
}

inline std::string layoutToString(
    Layout layout,
    bool upper = false,
    bool lower = false) {
  switch (layout) {
    case kSparseCsr:
      return (upper ? "CSR" : (lower ? "csr" : "Csr"));
    case kSparseCsc:
      return (upper ? "CSC" : (lower ? "csc" : "Csc"));
    case kSparseBsr:
      return (upper ? "BSR" : (lower ? "bsr" : "Bsr"));
    case kSparseBsc:
      return (upper ? "BSC" : (lower ? "bsc" : "Bsc"));
    default:
      TORCH_CHECK(false, "Not a sparse compressed layout:", layout);
      return "";
  }
}

inline bool isCompressedRow(Layout layout) {
  return AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
      layout, "isCompressedRow", [&] { return true; }, [&] { return false; });
}

inline bool isCompressedColumn(Layout layout) {
  return AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
      layout,
      "isCompressedColumn",
      [&] { return false; },
      [&] { return true; });
}

inline std::string compressedIndicesName(Layout layout) {
  return AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
      layout,
      "compressedIndicesName",
      [&] { return "crow_indices"; },
      [&] { return "ccol_indices"; });
}

inline std::string plainIndicesName(Layout layout) {
  return AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
      layout,
      "plainIndicesName",
      [&] { return "col_indices"; },
      [&] { return "row_indices"; });
}

inline std::string compressedDimName(Layout layout) {
  switch (layout) {
    case kSparseCsr:
      return "row";
    case kSparseCsc:
      return "column";
    case kSparseBsr:
      return "row block";
    case kSparseBsc:
      return "column block";
    default:
      TORCH_CHECK(false, "Not a sparse compressed layout:", layout);
      return "";
  }
}

inline std::string plainDimName(Layout layout) {
  switch (layout) {
    case kSparseCsr:
      return "column";
    case kSparseCsc:
      return "row";
    case kSparseBsr:
      return "column block";
    case kSparseBsc:
      return "row block";
    default:
      TORCH_CHECK(false, "Not a sparse compressed layout:", layout);
      return "";
  }
}

inline size_t rowDimension(Layout layout, IntArrayRef size) {
  return size.size() - (isCompressedRow(layout) ? 2 : 1);
}

inline size_t columnDimension(Layout layout, IntArrayRef size) {
  return size.size() - (isCompressedColumn(layout) ? 2 : 1);
}

inline size_t compressedDimension(
    Layout layout,
    IntArrayRef size,
    size_t dense_ndim = 0) {
  return size.size() - dense_ndim - (isCompressedRow(layout) ? 2 : 1);
}

inline size_t plainDimension(
    Layout layout,
    IntArrayRef size,
    size_t dense_ndim = 0) {
  return size.size() - dense_ndim - (isCompressedRow(layout) ? 1 : 2);
}

inline int64_t numBatchDimensions(Tensor const& self) {
  return AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
      self.layout(),
      "numBatchDimensions",
      [&self] { return self.crow_indices().dim() - 1; },
      [&self] { return self.ccol_indices().dim() - 1; });
}

inline std::pair<Tensor, Tensor> getCompressedPlainIndices(Tensor const& self) {
  return AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
      self.layout(),
      "getCompressedPlainIndices",
      [&self] {
        return std::make_pair(self.crow_indices(), self.col_indices());
      },
      [&self] {
        return std::make_pair(self.ccol_indices(), self.row_indices());
      });
}

inline ScalarType getIndexDtype(Tensor const& self) {
  switch (self.layout()) {
    case kSparseCsr:
    case kSparseBsr:
      return self.crow_indices().scalar_type();
    case kSparseCsc:
    case kSparseBsc:
      return self.ccol_indices().scalar_type();
    case kSparse:
      return self._indices().scalar_type();
    default:
      return ScalarType::Long;
  }
}

inline Layout flip_compressed_layout(Layout layout) {
  switch (layout) {
    case kSparseCsr:
      return kSparseCsc;
    case kSparseCsc:
      return kSparseCsr;
    case kSparseBsr:
      return kSparseBsc;
    case kSparseBsc:
      return kSparseBsr;
    default:
      TORCH_CHECK(false, "Not a sparse compressed layout:", layout);
      return kSparseCsr;
  }
}

inline DimVector getBlockSize(Tensor const& self) {
  int64_t n_batch = numBatchDimensions(self);
  return at::DimVector(self.values().sizes().slice(n_batch + 1, 2));
}

inline at::OptionalArray<at::SymInt> getSymIntBlockSize(Tensor const& self) {
  if (self.layout() == at::kSparseBsr || self.layout() == at::kSparseBsc) {
    int64_t n_batch = numBatchDimensions(self);
    return self.values().sym_sizes().slice(n_batch + 1, 2).vec();
  } else {
    return {};
  }
}

template <typename binary_op_t, typename binary_op_out_t>
inline bool only_sparse_compressed_binary_op_trivial_cases(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha,
    Tensor& out,
    const binary_op_t& binary_op,
    const binary_op_out_t& binary_op_out) {
  // Only sparse compressed! Just like the name says :)
  TORCH_INTERNAL_ASSERT(at::sparse_csr::is_sparse_compressed(self));
  TORCH_INTERNAL_ASSERT(at::sparse_csr::is_sparse_compressed(other));
  TORCH_INTERNAL_ASSERT(at::sparse_csr::is_sparse_compressed(out));

  // Bypass BLAS if there are matches in (self, other, out)
  if (self.is_same(out) && self.is_same(other)) {
    binary_op_out(self.values(), other.values(), alpha);
    return true;
  }
  if (self.is_same(other)) {
    auto [compressed_indices, plain_indices] =
        at::sparse_csr::getCompressedPlainIndices(self);
    static_cast<SparseCsrTensorImpl*>(out.unsafeGetTensorImpl())
        ->set_member_tensors(
            compressed_indices,
            plain_indices,
            binary_op(self.values(), other.values(), alpha),
            self.sizes());
    return true;
  }
  return false;
}

inline bool only_sparse_compressed_add_trivial_cases(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha,
    Tensor& out) {
  return only_sparse_compressed_binary_op_trivial_cases(
      self,
      other,
      alpha,
      out,
      [](const Tensor& v1, const Tensor& v2, const Scalar& alpha) {
        return v1.add(v2, alpha);
      },
      [](const Tensor& v1, const Tensor& v2, const Scalar& alpha) {
        return v1.add_(v2, alpha);
      });
}

inline Tensor to_type(const Tensor& input, ScalarType dtype) {
  auto [compressed_indices, plain_indices] =
      at::sparse_csr::getCompressedPlainIndices(input);
  return at::_sparse_compressed_tensor_unsafe(
      compressed_indices,
      plain_indices,
      std::move(input.values()).to(dtype),
      input.sizes(),
      dtype,
      input.layout(),
      input.device(),
      input.options().pinned_memory_opt());
}

template <typename acc_t, typename scalar_t>
inline std::tuple<Tensor, Tensor> create_acc_buffer(
    TensorOptions option,
    ScalarType type,
    int64_t nnz = -1) {
  Tensor new_values, new_values_acc;
  constexpr bool need_acc = !std::is_same_v<scalar_t, acc_t>;
  bool is_integral = at::isIntegralType(type, /*includeBool=*/true);
  if constexpr (need_acc) {
    auto acc_dtype = CppTypeToScalarType<acc_t>::value;
    new_values_acc = at::empty({}, option.dtype(acc_dtype));
    new_values = is_integral ? new_values_acc : at::empty({}, option);
  } else {
    new_values = new_values_acc = at::empty({}, option);
  }
  if (nnz != -1) {
    return std::make_tuple(
        new_values.resize_(nnz), new_values_acc.resize_(nnz));
  } else {
    return std::make_tuple(new_values, new_values_acc);
  }
}

inline void copy_from_acc_buffer(Tensor& new_values, Tensor& new_values_acc) {
  if (!new_values_acc.is_same(new_values)) {
    new_values.copy_(new_values_acc);
  }
}

} // namespace at::sparse_csr

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 41 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `CheckSparseTensorInvariants`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/SparseCsrTensorImpl.h`
- `ATen/SparseTensorImpl.h`
- `ATen/core/Tensor.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/Operators.h`
- `ATen/ops/_sparse_compressed_tensor_unsafe.h`
- `ATen/ops/resize_as_sparse_native.h`


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

Files in the same folder (`aten/src/ATen`):

- [`TensorGeometry.cpp_docs.md`](./TensorGeometry.cpp_docs.md)
- [`ROCmFABackend.h_docs.md`](./ROCmFABackend.h_docs.md)
- [`Generator.h_docs.md`](./Generator.h_docs.md)
- [`ParallelCommon.cpp_docs.md`](./ParallelCommon.cpp_docs.md)
- [`ZeroTensorFallback.cpp_docs.md`](./ZeroTensorFallback.cpp_docs.md)
- [`CachedTensorUtils.h_docs.md`](./CachedTensorUtils.h_docs.md)
- [`LegacyBatchedFallback.cpp_docs.md`](./LegacyBatchedFallback.cpp_docs.md)
- [`TensorOptions.h_docs.md`](./TensorOptions.h_docs.md)
- [`ExpandUtils.h_docs.md`](./ExpandUtils.h_docs.md)
- [`TensorIteratorInternal.h_docs.md`](./TensorIteratorInternal.h_docs.md)


## Cross-References

- **File Documentation**: `SparseCsrTensorUtils.h_docs.md`
- **Keyword Index**: `SparseCsrTensorUtils.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen`):

- [`Dispatch.cpp_docs.md_docs.md`](./Dispatch.cpp_docs.md_docs.md)
- [`Context.cpp_docs.md_docs.md`](./Context.cpp_docs.md_docs.md)
- [`ThreadLocalState.cpp_docs.md_docs.md`](./ThreadLocalState.cpp_docs.md_docs.md)
- [`DeviceAccelerator.cpp_kw.md_docs.md`](./DeviceAccelerator.cpp_kw.md_docs.md)
- [`FunctionalInverses.cpp_kw.md_docs.md`](./FunctionalInverses.cpp_kw.md_docs.md)
- [`SequenceNumber.h_kw.md_docs.md`](./SequenceNumber.h_kw.md_docs.md)
- [`ThreadLocalPythonObjects.h_docs.md_docs.md`](./ThreadLocalPythonObjects.h_docs.md_docs.md)
- [`TensorNames.h_docs.md_docs.md`](./TensorNames.h_docs.md_docs.md)
- [`LegacyBatchedTensorImpl.h_docs.md_docs.md`](./LegacyBatchedTensorImpl.h_docs.md_docs.md)
- [`TensorOperators.h_docs.md_docs.md`](./TensorOperators.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `SparseCsrTensorUtils.h_docs.md_docs.md`
- **Keyword Index**: `SparseCsrTensorUtils.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
