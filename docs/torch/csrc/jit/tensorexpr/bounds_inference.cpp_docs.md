# Documentation: `torch/csrc/jit/tensorexpr/bounds_inference.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/tensorexpr/bounds_inference.cpp`
- **Size**: 10,286 bytes (10.04 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/tensorexpr/bounds_inference.h>

#include <torch/csrc/jit/tensorexpr/bounds_overlap.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>

#include <c10/util/irange.h>

#include <iostream>
#include <utility>

namespace torch::jit::tensorexpr {

using namespace analysis;

template <typename Container>
static BoundsInfo mergeTensorAccesses(
    const Container& accesses,
    const std::unordered_map<VarPtr, BufPtr>& varToBuf,
    bool distinctAccessKinds) {
  BoundsInfo ret;
  for (auto& access : accesses) {
    if (access->type() == AccessType::Input ||
        access->type() == AccessType::Output) {
      continue;
    }

    auto vtbIt = varToBuf.find(access->var());
    TORCH_INTERNAL_ASSERT(vtbIt != varToBuf.end(), buildErrorMessage());
    BufPtr buf = vtbIt->second;
    std::vector<TensorAccessBoundsInfo>& infos = ret[buf];

    bool added = false;
    // This loop should be small, max of 2 (kLoad, kStore).
    for (auto& TABI : infos) {
      TensorAccessKind kind = access->isWrite() ? kStore : kLoad;
      if (!distinctAccessKinds || kind == TABI.kind) {
        TORCH_INTERNAL_ASSERT(
            TABI.start.size() == access->bounds().size(), buildErrorMessage());
        TORCH_INTERNAL_ASSERT(
            TABI.stop.size() == access->bounds().size(), buildErrorMessage());
        for (size_t i = 0; i < TABI.start.size(); ++i) {
          TABI.start[i] = IRSimplifier::simplify(
              alloc<Min>(TABI.start[i], access->bounds()[i].start, true));
          TABI.stop[i] = IRSimplifier::simplify(
              alloc<Max>(TABI.stop[i], access->bounds()[i].end, true));
          added = true;

          if (kind != TABI.kind) {
            TABI.kind = kMutate;
          }
        }
      }
    }

    if (!added) {
      TensorAccessBoundsInfo info;
      info.kind = access->isWrite() ? kStore : kLoad;

      for (auto& b : access->bounds()) {
        info.start.push_back(b.start);
        info.stop.push_back(b.end);
      }

      infos.push_back(info);
    }
  }

  return ret;
}

static std::unordered_map<VarPtr, BufPtr> getAllBufs(const StmtPtr& s) {
  std::unordered_map<VarPtr, BufPtr> varToBuf;

  auto bufs = NodeFinder<Buf>::find(s);
  for (const auto& b : bufs) {
    varToBuf[b->base_handle()] = b;
  }
  return varToBuf;
}

static std::unordered_map<VarPtr, BufPtr> getAllBufs(const ExprPtr& e) {
  std::unordered_map<VarPtr, BufPtr> varToBuf;

  auto bufs = NodeFinder<Buf>::find(e);
  for (const auto& b : bufs) {
    varToBuf[b->base_handle()] = b;
  }
  return varToBuf;
}

BoundsInfo inferBounds(const StmtPtr& s, bool distinctAccessKinds) {
  auto varToBuf = getAllBufs(s);

  MemDependencyChecker checker;
  s->accept(&checker);

  return mergeTensorAccesses(
      checker.getHistory(), varToBuf, distinctAccessKinds);
}

BoundsInfo getInferredBounds(
    MemDependencyChecker& analyzer,
    const StmtPtr& s,
    bool distinctAccessKinds) {
  return mergeTensorAccesses(
      analyzer.accessesWithin(s), getAllBufs(s), distinctAccessKinds);
}

BoundsInfo getInferredBounds(
    MemDependencyChecker& analyzer,
    const ExprPtr& e,
    bool distinctAccessKinds) {
  return mergeTensorAccesses(
      analyzer.accessesWithin(e), getAllBufs(e), distinctAccessKinds);
}

void printBoundsInfo(const BoundsInfo& v) {
  std::cerr << "Access vector {\n";
  for (auto& pair : v) {
    std::cerr << *pair.first << " in [";
    bool first = true;
    for (auto& b : pair.second) {
      if (!first) {
        std::cerr << ", ";
      }
      std::cerr << ((b.kind == kLoad) ? "LOAD" : "STORE") << "(";
      int i = 0;
      if (b.start.empty()) {
        std::cerr << "0";
      }
      for (auto& s : b.start) {
        if (i != 0) {
          std::cerr << ", ";
        }
        std::cerr << *s;
        i++;
      }
      std::cerr << "; ";
      i = 0;
      if (b.stop.empty()) {
        std::cerr << "0";
      }
      for (auto& s : b.stop) {
        if (i != 0) {
          std::cerr << ", ";
        }
        std::cerr << *s;
        i++;
      }
      std::cerr << ")";
      first = false;
    }
    std::cerr << "]\n";
  }
  std::cerr << "}\n";
}

std::vector<ExprPtr> getBoundExtents(
    const std::vector<TensorAccessBoundsInfo>& infos) {
  std::vector<ExprPtr> starts;
  std::vector<ExprPtr> stops;

  // Find the safe size of the temporary buffer by determining the outer
  // extents of a union of all bounds.
  for (const TensorAccessBoundsInfo& p : infos) {
    for (const auto i : c10::irange(p.start.size())) {
      if (starts.size() <= i) {
        starts.push_back(p.start[i]);
      } else {
        starts[i] =
            IRSimplifier::simplify(alloc<Min>(starts[i], p.start[i], true));
      }

      if (stops.size() <= i) {
        stops.push_back(p.stop[i]);
      } else {
        stops[i] =
            IRSimplifier::simplify(alloc<Max>(stops[i], p.stop[i], true));
      }
    }
  }

  std::vector<ExprPtr> extents;
  for (size_t i = 0; i < starts.size(); ++i) {
    ExprPtr dim = IRSimplifier::simplify(
        alloc<Add>(alloc<Sub>(stops[i], starts[i]), immLike(stops[i], 1)));

    extents.push_back(dim);
  }

  return extents;
}

using BoundSet = std::unordered_set<Bound, BoundHash>;

static BoundSet convertBounds(
    const std::vector<TensorAccessBoundsInfo>& bounds,
    TensorAccessKind filter = kMutate) {
  BoundSet ret;
  for (auto& TABI : bounds) {
    if (filter == kMutate || TABI.kind == filter) {
      for (size_t i = 0; i < TABI.start.size(); ++i) {
        ret.insert(Bound(TABI.start[i], TABI.stop[i]));
      }
    }
  }
  return ret;
}

static BoundSet convertBounds(
    BoundsInfo& bounds,
    const BufPtr& buf,
    TensorAccessKind filter = kMutate) {
  auto it = bounds.find(buf);
  if (it == bounds.end()) {
    return BoundSet();
  }

  return convertBounds(it->second, filter);
}

HazardKind getPotentialHazards(
    MemDependencyChecker& analyzer,
    const StmtPtr& A,
    const StmtPtr& B) {
  BoundsInfo aBounds = getInferredBounds(analyzer, A, true);
  BoundsInfo bBounds = getInferredBounds(analyzer, B, true);

  for (auto& pair : bBounds) {
    BufPtr buf = pair.first;
    if (aBounds.find(buf) == aBounds.end()) {
      continue;
    }

    auto aWrites = convertBounds(aBounds, buf, kStore);
    auto aReads = convertBounds(aBounds, buf, kLoad);

    auto bWrites = convertBounds(pair.second, kStore);
    auto bReads = convertBounds(pair.second, kLoad);

    // First, RAW.
    for (auto& bR : bReads) {
      for (auto& aW : aWrites) {
        if (boundOverlap(bR, aW) != OverlapKind::NoOverlap) {
          return HazardKind::ReadAfterWrite;
        }
      }
    }

    // Then WAR.
    for (auto& bW : bWrites) {
      for (auto& aR : aReads) {
        if (boundOverlap(bW, aR) != OverlapKind::NoOverlap) {
          return HazardKind::WriteAfterRead;
        }
      }
    }

    // Then WAW.
    for (auto& bW : bWrites) {
      for (auto& aW : aWrites) {
        if (boundOverlap(bW, aW) != OverlapKind::NoOverlap) {
          return HazardKind::WriteAfterWrite;
        }
      }
    }
  }

  return HazardKind::NoDependency;
}

static IndexBounds getIndexBounds(const TensorAccessBoundsInfo& tabi) {
  TORCH_INTERNAL_ASSERT(
      tabi.start.size() == tabi.stop.size(), buildErrorMessage());
  IndexBounds ret(tabi.start.size());
  if (tabi.start.empty()) {
    return ret;
  }
  for (size_t i = 0; i < tabi.start.size(); ++i) {
    ret[i] = Bound(tabi.start[i], tabi.stop[i]);
  }
  return ret;
}

static std::vector<IndexBounds> getIndexBounds(
    const std::vector<TensorAccessBoundsInfo>& vTABI,
    TensorAccessKind filter = kMutate) {
  std::vector<IndexBounds> bounds;
  for (auto& TABI : vTABI) {
    if (filter == kMutate || TABI.kind == filter) {
      bounds.push_back(getIndexBounds(TABI));
    }
  }
  return bounds;
}

static bool hasConflictingOverlap(
    const BoundsInfo& aBounds,
    const BoundsInfo& bBounds,
    TensorAccessKind aFilter = kMutate,
    TensorAccessKind bFilter = kMutate) {
  using IndexBoundsInfo = std::unordered_map<BufPtr, std::vector<IndexBounds>>;
  IndexBoundsInfo aIndexBoundsInfo;
  for (auto& aBound : aBounds) {
    aIndexBoundsInfo[aBound.first] = getIndexBounds(aBound.second, aFilter);
  }
  IndexBoundsInfo bIndexBoundsInfo;
  for (auto& bBound : bBounds) {
    bIndexBoundsInfo[bBound.first] = getIndexBounds(bBound.second, bFilter);
  }

  for (auto& aBound : aBounds) {
    auto bIt = bBounds.find(aBound.first);
    if (bIt == bBounds.end()) {
      continue;
    }
    auto aIndexBounds = aIndexBoundsInfo[aBound.first];
    auto bIndexBounds = bIndexBoundsInfo[bIt->first];
    auto aTABIs = aBound.second;
    auto bTABIs = bIt->second;
    for (size_t i = 0; i < aTABIs.size(); ++i) {
      for (size_t j = 0; j < bTABIs.size(); ++j) {
        auto aTABI = aTABIs[i];
        auto bTABI = bTABIs[j];
        if (aTABI.kind == kLoad && bTABI.kind == kLoad) {
          continue;
        }
        auto overlap = overlaps(aIndexBounds[i], bIndexBounds[j]);
        if (overlap != OverlapKind::NoOverlap) {
          return true;
        }
      }
    }
  }
  return false;
}

bool hasConflictingOverlap(
    analysis::MemDependencyChecker& analyzer,
    const StmtPtr& A,
    const StmtPtr& B) {
  BoundsInfo aBounds = getInferredBounds(analyzer, A, true);
  BoundsInfo bBounds = getInferredBounds(analyzer, B, true);
  return hasConflictingOverlap(aBounds, bBounds);
}

bool isOverlapping(
    analysis::MemDependencyChecker& analyzer,
    const StorePtr& S1,
    const StorePtr& S2) {
  BoundsInfo s1Bounds = getInferredBounds(analyzer, S1, true);
  BoundsInfo s2Bounds = getInferredBounds(analyzer, S2, true);
  return hasConflictingOverlap(s1Bounds, s2Bounds, kStore, kStore);
}

bool isOverlapping(
    analysis::MemDependencyChecker& analyzer,
    const StorePtr& S,
    const LoadPtr& L) {
  BoundsInfo sBounds = getInferredBounds(analyzer, S, true);
  BoundsInfo lBounds = getInferredBounds(analyzer, L, true);
  return hasConflictingOverlap(sBounds, lBounds, kStore, kLoad);
}

} // namespace torch::jit::tensorexpr

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 23 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `analysis`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/tensorexpr`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/tensorexpr/bounds_inference.h`
- `torch/csrc/jit/tensorexpr/bounds_overlap.h`
- `torch/csrc/jit/tensorexpr/expr.h`
- `torch/csrc/jit/tensorexpr/ir.h`
- `torch/csrc/jit/tensorexpr/ir_printer.h`
- `torch/csrc/jit/tensorexpr/ir_simplifier.h`
- `torch/csrc/jit/tensorexpr/ir_visitor.h`
- `torch/csrc/jit/tensorexpr/stmt.h`
- `c10/util/irange.h`
- `iostream`
- `utility`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`torch/csrc/jit/tensorexpr`):

- [`codegen.cpp_docs.md`](./codegen.cpp_docs.md)
- [`bounds_overlap.h_docs.md`](./bounds_overlap.h_docs.md)
- [`eval.h_docs.md`](./eval.h_docs.md)
- [`cuda_codegen.cpp_docs.md`](./cuda_codegen.cpp_docs.md)
- [`ir_simplifier.h_docs.md`](./ir_simplifier.h_docs.md)
- [`reduction.h_docs.md`](./reduction.h_docs.md)
- [`kernel.cpp_docs.md`](./kernel.cpp_docs.md)
- [`cuda_codegen.h_docs.md`](./cuda_codegen.h_docs.md)
- [`external_functions_core.cpp_docs.md`](./external_functions_core.cpp_docs.md)
- [`graph_opt.h_docs.md`](./graph_opt.h_docs.md)


## Cross-References

- **File Documentation**: `bounds_inference.cpp_docs.md`
- **Keyword Index**: `bounds_inference.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
