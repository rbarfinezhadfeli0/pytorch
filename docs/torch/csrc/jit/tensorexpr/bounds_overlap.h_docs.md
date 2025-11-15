# Documentation: bounds_overlap.h

## File Metadata
- **Path**: `torch/csrc/jit/tensorexpr/bounds_overlap.h`
- **Size**: 4471 bytes
- **Lines**: 121
- **Extension**: .h
- **Type**: Regular file

## Original Source

```h
#pragma once

#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/ir.h>

#include <utility>
#include <vector>

namespace torch::jit::tensorexpr::analysis {

// A simple class containing the start and end of a range in a single dimension.
struct TORCH_API Bound {
  ExprPtr start{nullptr};
  ExprPtr end{nullptr};

  // This stores whether or not the start and end of this Bound have previously
  // been swapped. This occurs when the bound is in a loop with a negative
  // stride.
  bool swapped{false};

  Bound() = default;
  Bound(ExprPtr s, ExprPtr e) : start(std::move(s)), end(std::move(e)) {}

  void print() const;
  bool equals(const Bound& other) const;

  // The comparison operators are conservative. If the compare operator returns
  // true, it means that all the elements satisfy the logical expression. But
  // the false does not mean the opposite comparison is satisfied. It could be
  // but not always.
  bool operator==(const Bound& other) const;
  bool operator!=(const Bound& other) const;
  bool operator<(const Bound& other) const;
  bool operator<=(const Bound& other) const;
  bool operator>(const Bound& other) const;
  bool operator>=(const Bound& other) const;

  void swap() noexcept {
    std::swap(start, end);
    swapped = !swapped;
  }
};

struct BoundHash {
  size_t operator()(const Bound& b) const {
    return std::hash<ExprPtr>()(b.start) ^ std::hash<ExprPtr>()(b.end);
  }
};

// The type of overlap found. Each condition is true only if none of the
// previous conditions hold.
//     ContainedOrEqual: All elements in the Bound A are in the Bound B (this
//                       includes the case where the bounds are equal).
//     Contains: All elements in the Bound B are in the Bound B.
//     PartialOverlap: Any elements in the Bound B are in the Bound A.
//     NoOverlap: No elements in the Bound A are in the bound B.
enum class OverlapKind {
  ContainedOrEqual,
  Contains,
  PartialOverlap,
  NoOverlap
};

// The Bound comparison result.
//     True: Every Bound element always satisfies the given comparison operator
//     False: Every Bound element always does NOT satisfy the given comparison
//     operator
//     NotDetermined: Some elements satisfy the given comparison operator and
//     some elements not
enum class CmpEvalResult { True, False, NotDetermined };

// Returns the kind of overlap between Bound A and Bound A in a single
// dimension.
OverlapKind TORCH_API boundOverlap(const Bound& A, const Bound& B);

// The comparison is conservative and the compare result is deterministic.
// It means that every element of the Bound to be compared needs to satisfy
// the given comparison operator.
CmpEvalResult TORCH_API compareBound(
    const Bound& a,
    const Bound& b,
    const CompareSelectOperation& cmp_op);

// A multi dimensional bound representing the bound of a set of indices.
using IndexBounds = std::vector<Bound>;

// Returns true if two IndexBounds are equivalent.
bool TORCH_API indexBoundsEquals(const IndexBounds& A, const IndexBounds& B);

// Flattens a multi dimensional bound to a single dimension. The IndexBounds "a"
// *must* encapsulate the entire range of the buffer.
Bound TORCH_API flattenBounds(const IndexBounds& a);

// Determines the kind of overlap in X dimensions.
OverlapKind TORCH_API overlaps(const IndexBounds& a, const IndexBounds& b);

// Returns the Bound slices created by subtracing bound B from bound A.
// Multiple Bounds can be returned in the case where B slices A into two
// distinct regions with no overlap.
//
// For example:
//    subtractBound((0, 10), (2, 4)) => [(0, 1), (5, 10)]
//       bound A: (0, 10)
//       bound B: (2, 4)
//       If we remove slice (2, 4) from the slice (0, 10), we will be left
//       with 2 slices, one at the start (0, 1), and one at the end (5, 10).
//       So, the result of this subtraction is [(0, 1), (5, 10)].
//
// Note: this doesn't use IndexBounds because the Bounds returned do not
// represent multiple different dimensions.
std::vector<Bound> TORCH_API subtractBound(const Bound& a, const Bound& b);

// Returns the bound slices created by subtracting the IndexBounds B from A.
std::vector<IndexBounds> TORCH_API subtractIndicesBounds(
    const IndexBounds& A,
    const IndexBounds& B,
    OverlapKind overlap);
std::vector<IndexBounds> TORCH_API
subtractIndicesBounds(const IndexBounds& A, const IndexBounds& B);

} // namespace torch::jit::tensorexpr::analysis

```

## High-Level Overview

This file is part of the PyTorch repository. It is a C++/CUDA source/header file that may contain implementations, declarations, or kernel code.

## Detailed Walkthrough

### Classes
This file defines 3 class(es): containing, OverlapKind, CmpEvalResult

### Structures
This file defines 2 struct(s): TORCH_API, BoundHash


## Key Components

The file contains 640 words across 121 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 4471 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
