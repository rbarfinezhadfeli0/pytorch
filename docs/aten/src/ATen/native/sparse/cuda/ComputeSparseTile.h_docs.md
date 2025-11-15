# Documentation: `aten/src/ATen/native/sparse/cuda/ComputeSparseTile.h`

## File Metadata

- **Path**: `aten/src/ATen/native/sparse/cuda/ComputeSparseTile.h`
- **Size**: 6,129 bytes (5.99 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/native/sparse/cuda/SparseSemiStructuredPack.h>
#include <ATen/native/sparse/cuda/StaticSort.h>
#include <cutlass/bfloat16.h>
#include <cutlass/half.h>
#include <cutlass/platform/platform.h>
#include <cutlass/version.h>

// Given 4x4 values, computes the selected indices that will remain after 2:4
// sparsification, as a bitmask.
// NOTE: Algorithms might select LESS than 8 values in total in some cases.

namespace at::native {

template <typename Element, typename Pointwise>
struct TileValueOrderedT {
  union {
    struct {
      Element value;
      uint2b_t col;
      uint2b_t row;
    } parts;
    uint32_t raw;
  };
  CUTLASS_DEVICE bool operator<(
      TileValueOrderedT<Element, Pointwise> const& other) const {
    return Pointwise::apply(parts.value) < Pointwise::apply(other.parts.value);
  }
  CUTLASS_DEVICE TileValueOrderedT() {}
};

// Operations that we can apply to rank the values
struct IdentityOp {
  template <typename T>
  static T CUTLASS_HOST_DEVICE apply(T const& x) {
    return x;
  }
};
// Can be applied to rank based on absolute value
struct AbsOp {
  template <typename T>
  static T CUTLASS_HOST_DEVICE apply(T const& x) {
    return cutlass::abs(x);
  }
};

// Given 4x4 values, computes the selected indices that will remain after 2:4
// sparsification, as a bitmask. We have 2 constraints:
// (1) At most 2 values per line
// (2) At most 2 values per column
// This means we can select at most 8 values in total.
// ALGO: We use a greedy algorithm, where we take values in the 4x4
// tile in descending order. If a value fits (because the line/col is not
// already full), we select it. Then we move on to the next one.
// NOTE: This algorithm might select LESS than 8 values in total in some cases.
// NOTE (2): RF are not indexable, so we shouldn't rely on indexing
//   values at any point, otherwise they will be stored in local memory.
template <typename Op = IdentityOp>
struct LargestValuesGreedy {
  template <typename T>
  static CUTLASS_DEVICE T outOfBoundsFillValue() {
    return -cutlass::platform::numeric_limits<T>::infinity();
  }

  template <typename Tile4x4Accessor>
  CUTLASS_DEVICE Indices4x4 operator()(Tile4x4Accessor values) {
    using TileValueOrdered =
        TileValueOrderedT<typename Tile4x4Accessor::Element, Op>;
    using TileValuesFragment = cutlass::Array<TileValueOrdered, 4 * 4>;
    Indices4x4 indices;
    TileValuesFragment values_ordered;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < 4; ++j) {
        TileValueOrdered& v = values_ordered[i * 4 + j];
        v.parts.value = values.at(i, j).get();
        v.parts.col = uint2b_t(j);
        v.parts.row = uint2b_t(i);
      }
    }
    // Use a sorting network (aka without branches) to avoid
    // warp divergence
    StaticSort<TileValuesFragment::kElements> sorter;
    sorter(values_ordered);

    // bitmask to store how many we have selected on a given row/col
    // 0 selected: (numPerRow >> 2*row) = 00 (0)
    // 1 selected: (numPerRow >> 2*row) = 01 (1)
    // 2 selected: (numPerRow >> 2*row) = 11 (3)
    uint32_t numPerRow = 0;
    uint32_t numPerCol = 0;
    indices = 0;

    // Take as many as we can, starting with the largest values
    CUTLASS_PRAGMA_UNROLL
    for (int i = values_ordered.size() - 1; i >= 0; i--) {
      auto& e = values_ordered[i];

      uint32_t rcount = uint2b_t(numPerRow >> 2 * e.parts.row);
      uint32_t ccount = uint2b_t(numPerCol >> 2 * e.parts.col);
      // NOTE: This is more efficient (yet equivalent) to:
      // `rcount != 3 && ccount != 3`
      bool selected = (rcount + ccount) <= 2;
      indices |= selected << (e.parts.col + 4 * e.parts.row);

      numPerRow |= (rcount + selected) << 2 * e.parts.row;
      numPerCol |= (ccount + selected) << 2 * e.parts.col;
    }
    return indices;
  }
};

// We consider each rows independently in order
// This is to ensure that a row's sparsity pattern is only determined
// by its values and the rows before (but never the rows after)
// This enforces causality strictly
template <typename Op = IdentityOp>
struct Causal1122 {
  template <typename T>
  static CUTLASS_DEVICE T outOfBoundsFillValue() {
    return -cutlass::platform::numeric_limits<T>::infinity();
  }

  template <typename Tile4x4Accessor>
  CUTLASS_DEVICE Indices4x4 operator()(Tile4x4Accessor values) {
    static constexpr int kMaxValuesPerRow[] = {1, 1, 2, 2};
    using TileValueOrdered =
        TileValueOrderedT<typename Tile4x4Accessor::Element, Op>;
    using TileValuesFragment = cutlass::Array<TileValueOrdered, 4>;
    Indices4x4 indices = 0;

    uint32_t numPerCol = 0; // <- see doc in `LargestValuesGreedy`

    CUTLASS_PRAGMA_UNROLL
    for (int row = 0; row < 4; ++row) {
      int row_count = 0;
      TileValuesFragment values_ordered;
      CUTLASS_PRAGMA_UNROLL
      for (int col = 0; col < 4; ++col) {
        TileValueOrdered& v = values_ordered[col];
        v.parts.value = values.at(row, col).get();
        v.parts.col = uint2b_t(col);
      }
      // Use a sorting network (aka without branches) to avoid
      // warp divergence
      StaticSort<TileValuesFragment::kElements> sorter;
      sorter(values_ordered);

      // Take as many as we can, starting with the largest values
      CUTLASS_PRAGMA_UNROLL
      for (int i = values_ordered.size() - 1; i >= 0; i--) {
        auto& e = values_ordered[i];

        uint32_t ccount = uint2b_t(numPerCol >> 2 * e.parts.col);
        bool selected = ccount != 3 && (row_count < kMaxValuesPerRow[row]);
        indices |= selected << (e.parts.col + 4 * row);
        numPerCol |= (ccount + selected) << 2 * e.parts.col;
        row_count += selected;
      }
    }
    return indices;
  }
};

template <typename T>
void named_algorithms(T callback) {
  callback(LargestValuesGreedy<IdentityOp>(), "largest_values_greedy");
  callback(Causal1122<IdentityOp>(), "causal1122");
  callback(LargestValuesGreedy<AbsOp>(), "largest_abs_values_greedy");
  // default one
  callback(LargestValuesGreedy<IdentityOp>(), "");
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 20 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `TileValueOrderedT`, `IdentityOp`, `AbsOp`, `LargestValuesGreedy`, `Causal1122`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/sparse/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/sparse/cuda/SparseSemiStructuredPack.h`
- `ATen/native/sparse/cuda/StaticSort.h`
- `cutlass/bfloat16.h`
- `cutlass/half.h`
- `cutlass/platform/platform.h`
- `cutlass/version.h`


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

Files in the same folder (`aten/src/ATen/native/sparse/cuda`):

- [`cuSPARSELtOps.cpp_docs.md`](./cuSPARSELtOps.cpp_docs.md)
- [`SparseCsrTensorMath.cu_docs.md`](./SparseCsrTensorMath.cu_docs.md)
- [`SparseSemiStructuredOps.cu_docs.md`](./SparseSemiStructuredOps.cu_docs.md)
- [`SparseCUDABlas.h_docs.md`](./SparseCUDABlas.h_docs.md)
- [`SparseMatMul.cu_docs.md`](./SparseMatMul.cu_docs.md)
- [`SparseCUDATensorMath.cuh_docs.md`](./SparseCUDATensorMath.cuh_docs.md)
- [`cuSPARSELtOps.h_docs.md`](./cuSPARSELtOps.h_docs.md)
- [`SparseBlas.cpp_docs.md`](./SparseBlas.cpp_docs.md)
- [`StaticSort.h_docs.md`](./StaticSort.h_docs.md)
- [`SparseSemiStructuredApplyDense.cu_docs.md`](./SparseSemiStructuredApplyDense.cu_docs.md)


## Cross-References

- **File Documentation**: `ComputeSparseTile.h_docs.md`
- **Keyword Index**: `ComputeSparseTile.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
