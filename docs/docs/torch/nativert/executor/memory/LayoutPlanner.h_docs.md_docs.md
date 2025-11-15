# Documentation: `docs/torch/nativert/executor/memory/LayoutPlanner.h_docs.md`

## File Metadata

- **Path**: `docs/torch/nativert/executor/memory/LayoutPlanner.h_docs.md`
- **Size**: 7,208 bytes (7.04 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nativert/executor/memory/LayoutPlanner.h`

## File Metadata

- **Path**: `torch/nativert/executor/memory/LayoutPlanner.h`
- **Size**: 4,374 bytes (4.27 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <condition_variable>
#include <functional>
#include <thread>

#include <c10/macros/Macros.h>
#include <c10/util/CallOnce.h>
#include <c10/util/FbcodeMaps.h>
#include <c10/util/LeftRight.h>

#include <torch/nativert/executor/memory/AliasAnalyzer.h>
#include <torch/nativert/executor/memory/FunctionSchema.h>
#include <torch/nativert/executor/memory/LayoutPlannerAlgorithm.h>
#include <torch/nativert/executor/memory/LayoutPlannerSettings.h>
#include <torch/nativert/graph/Graph.h>

namespace {
constexpr inline std::memory_order drop_release(std::memory_order m) noexcept {
  return (
      m == std::memory_order_release
          ? std::memory_order_relaxed
          : ((m == std::memory_order_acq_rel || m == std::memory_order_seq_cst)
                 ? std::memory_order_acquire
                 : m));
}
// derivation of
// https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p0493r5.pdf
template <typename T>
void atomic_set_max(
    std::atomic<T>* pv,
    typename std::atomic<T>::value_type v,
    std::memory_order m = std::memory_order_seq_cst) noexcept {
  auto const mr = drop_release(m);
  auto t = (mr != m) ? pv->fetch_add(0, m) : pv->load(mr);
  while (std::max(v, t) != t) {
    if (pv->compare_exchange_weak(t, v, m, mr)) {
      return;
    }
  }
}
} // namespace

namespace torch::nativert {

class LayoutPlanner {
 public:
  explicit LayoutPlanner(
      const Graph& graph,
      const c10::FastMap<std::string /* target */, FunctionSchema>&
          kernelSchemas,
      const std::vector<bool>& persistentValues,
      const torch::nativert::LayoutPlannerSettings& settings);
#if !defined(_MSC_VER)
  TORCH_API // TODO Doesn't work on msvc.
#endif
      ~LayoutPlanner();

  LayoutPlanner(LayoutPlanner&& other) = delete;
  LayoutPlanner(const LayoutPlanner& other) = delete;
  LayoutPlanner operator=(LayoutPlanner&& other) = delete;
  LayoutPlanner& operator=(const LayoutPlanner& other) = delete;

  void start_worker_if_not_started();

  const std::vector<ValueId>& get_planned_values() const;
  const std::vector<ValueId>& get_unplanned_values() const;

#ifndef NDEBUG
  const AliasAnalyzer& get_alias_analyzer() const {
    return alias_analyzer_;
  }
#endif

  size_t num_values() const {
    return managed_values_.size();
  }

  bool is_managed(ValueId id) {
    TORCH_CHECK(static_cast<size_t>(id) < managed_values_.size());
    return managed_values_[id];
  }

  C10_ALWAYS_INLINE void try_update_max_size_at_index(size_t idx, size_t size) {
    atomic_set_max<size_t>(&planned_values_historical_max_nbytes_[idx], size);
  }

  C10_ALWAYS_INLINE
  void with_plan(std::function<void(const LayoutPlan&)>&& cb) {
    plan_.read(
        std::forward<std::function<void(const LayoutPlan&)>>(std::move(cb)));
  }

 private:
#ifdef LayoutPlannerTests_TEST_FRIENDS
  LayoutPlannerTests_TEST_FRIENDS;
#endif

  // we need some way of mapping graph values to other information
  // (e.g.,  allocation spec, max historical size)
  //
  // since there is a 1:1 mapping to/from each of these
  // we can create+initialize them here
  //
  // note: planning algorithms are allowed to change the ordering
  // of allocation specs -- so we pass the index of the spec during
  // it's insertion s.t., each execution frame can use it to
  // reference the correct associated max historical size / underlying
  // tensor value
  void initialize_vectors(
      c10::FastMap<const Value*, AllocationSpec> value_to_allocation_spec);

  void run_periodic(const std::function<void()>& f);
  void create_plan();

  // variables for managing the state of the
  // interval worker thread that refreshes
  // the plan
  std::condition_variable cv_;
  std::mutex mutex_;
  bool stopped_{false};
  std::thread worker_;

  std::vector<ValueId> unplanned_values_;

  std::vector<ValueId> planned_values_;
  std::vector<AllocationSpec> planned_allocation_specs_;
  std::vector<std::atomic_size_t> planned_values_historical_max_nbytes_;

  // managed_values_[value_id] == true
  // if graph.values()[value_id] has
  // an associated allocation spec
  std::vector<bool> managed_values_;

  LayoutPlannerAlgorithm* algorithm_;
  c10::LeftRight<LayoutPlan> plan_;

  c10::once_flag worker_once_flag_;

#ifndef NDEBUG
  AliasAnalyzer alias_analyzer_;
#endif
  torch::nativert::LayoutPlannerSettings settings_;
};

} // namespace torch::nativert

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `namespace`, `torch`

**Classes/Structs**: `LayoutPlanner`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nativert/executor/memory`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `condition_variable`
- `functional`
- `thread`
- `c10/macros/Macros.h`
- `c10/util/CallOnce.h`
- `c10/util/FbcodeMaps.h`
- `c10/util/LeftRight.h`
- `torch/nativert/executor/memory/AliasAnalyzer.h`
- `torch/nativert/executor/memory/FunctionSchema.h`
- `torch/nativert/executor/memory/LayoutPlannerAlgorithm.h`
- `torch/nativert/executor/memory/LayoutPlannerSettings.h`
- `torch/nativert/graph/Graph.h`


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

Files in the same folder (`torch/nativert/executor/memory`):

- [`Bump.h_docs.md`](./Bump.h_docs.md)
- [`AliasAnalyzer.cpp_docs.md`](./AliasAnalyzer.cpp_docs.md)
- [`LayoutManager.cpp_docs.md`](./LayoutManager.cpp_docs.md)
- [`DisjointStorageGroups.h_docs.md`](./DisjointStorageGroups.h_docs.md)
- [`LayoutPlanner.cpp_docs.md`](./LayoutPlanner.cpp_docs.md)
- [`GreedyBySize.h_docs.md`](./GreedyBySize.h_docs.md)
- [`Bump.cpp_docs.md`](./Bump.cpp_docs.md)
- [`DisjointStorageGroups.cpp_docs.md`](./DisjointStorageGroups.cpp_docs.md)
- [`LayoutPlannerAlgorithm.h_docs.md`](./LayoutPlannerAlgorithm.h_docs.md)


## Cross-References

- **File Documentation**: `LayoutPlanner.h_docs.md`
- **Keyword Index**: `LayoutPlanner.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/nativert/executor/memory`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/nativert/executor/memory`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/nativert/executor/memory`):

- [`LayoutPlannerSettings.h_kw.md_docs.md`](./LayoutPlannerSettings.h_kw.md_docs.md)
- [`LayoutManager.h_kw.md_docs.md`](./LayoutManager.h_kw.md_docs.md)
- [`AliasAnalyzer.h_kw.md_docs.md`](./AliasAnalyzer.h_kw.md_docs.md)
- [`FunctionSchema.cpp_docs.md_docs.md`](./FunctionSchema.cpp_docs.md_docs.md)
- [`LayoutPlanner.cpp_kw.md_docs.md`](./LayoutPlanner.cpp_kw.md_docs.md)
- [`DisjointStorageGroups.cpp_docs.md_docs.md`](./DisjointStorageGroups.cpp_docs.md_docs.md)
- [`DisjointStorageGroups.h_kw.md_docs.md`](./DisjointStorageGroups.h_kw.md_docs.md)
- [`GreedyBySize.cpp_docs.md_docs.md`](./GreedyBySize.cpp_docs.md_docs.md)
- [`FunctionSchema.cpp_kw.md_docs.md`](./FunctionSchema.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `LayoutPlanner.h_docs.md_docs.md`
- **Keyword Index**: `LayoutPlanner.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
