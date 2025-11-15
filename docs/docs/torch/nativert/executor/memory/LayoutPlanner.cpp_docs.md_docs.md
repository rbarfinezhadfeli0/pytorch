# Documentation: `docs/torch/nativert/executor/memory/LayoutPlanner.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/nativert/executor/memory/LayoutPlanner.cpp_docs.md`
- **Size**: 10,410 bytes (10.17 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nativert/executor/memory/LayoutPlanner.cpp`

## File Metadata

- **Path**: `torch/nativert/executor/memory/LayoutPlanner.cpp`
- **Size**: 7,701 bytes (7.52 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/nativert/executor/memory/LayoutPlanner.h>

#include <c10/util/Enumerate.h>

#include <torch/nativert/executor/ExecutionPlanner.h>
#include <torch/nativert/executor/memory/AliasAnalyzer.h>
#include <torch/nativert/executor/memory/Bump.h>
#include <torch/nativert/executor/memory/DisjointStorageGroups.h>
#include <torch/nativert/executor/memory/GreedyBySize.h>

namespace torch::nativert {

LayoutPlanner::LayoutPlanner(
    const Graph& graph,
    const c10::FastMap<std::string /* target */, FunctionSchema>& kernelSchemas,
    const std::vector<bool>& persistentValues,
    const torch::nativert::LayoutPlannerSettings& settings)
    : managed_values_(graph.values().size()),
#ifndef NDEBUG
      alias_analyzer_(graph, kernelSchemas),
#endif
      settings_(settings) {
#ifndef NDEBUG
  auto& alias_analyzer = alias_analyzer_;
#else
  auto alias_analyzer = AliasAnalyzer(graph, kernelSchemas);
#endif

  auto value_to_allocation_spec = c10::FastMap<const Value*, AllocationSpec>{};

  std::set<const Value*> input_values_set_;
  for (const auto* nv : graph.userInputs()) {
    if (nv->type() == Type::Kind::Tensor) {
      input_values_set_.insert(nv);
    }
  }

  const auto& tensor_meta = graph.tensorValuesMeta();

  for (auto&& [i, node] : at::enumerate(graph.nodes())) {
    // only manage out variant values
    if (const auto schemaIt = kernelSchemas.find(std::string(node.target()));
        schemaIt == kernelSchemas.end() ||
        schemaIt->second.kernel_kind() != OpKernelKind::kStaticDispatchKernel) {
      VLOG(1) << "not able to plan outputs for node " << node.target()
              << " as it is derived from an unsupported kernel kind.";
      continue;
    }

    for (const auto& output : node.outputs()) {
      // don't manage persistent values
      if (bool is_persistent = persistentValues[output->id()]; is_persistent) {
        VLOG(1)
            << "not planning " << output->name()
            << " as it is a persistent value (likely a weight or const-folded)";
        continue;
      }

      // only manage tensors
      if (bool is_tensor = output->type().kind() == Type::Kind::Tensor;
          !is_tensor) {
        VLOG(1) << "not planning " << output->name()
                << " as it is not a raw tensor. type: " << output->type();
        continue;
      }

      // output storage ownership must be given to the caller.
      if (const auto& values_associated_with_output =
              alias_analyzer.values_associated_with_output_storage();
          values_associated_with_output.find(output) !=
          values_associated_with_output.end()) {
        VLOG(1)
            << "not planning " << output->name()
            << " as its underlying storage may be associated with a graph output";
        continue;
      }

      // inputs are borrowed -- this is merely a sanity check
      if (input_values_set_.find(output) != input_values_set_.end()) {
        VLOG(1) << "not planning " << output->name()
                << " as it is a graph input that is borrowed from the user";
        continue;
      }

      // don't plan aliases -- they don't own the associated dataptr
      if (bool is_alias = alias_analyzer.is_alias(output); is_alias) {
        VLOG(1) << "not planning " << output->name() << " as it is an alias";
        continue;
      }

      if (bool is_not_consumed = output->users().empty(); is_not_consumed) {
        VLOG(1) << "not planning " << output->name() << " as it has no users";
        continue;
      }

      if (auto meta_it = tensor_meta.find(std::string(output->name()));
          meta_it != tensor_meta.end()) {
        if (const auto& meta = meta_it->second; meta.device() == c10::kCPU) {
          auto& spec = value_to_allocation_spec[output];
          spec.lifetime = alias_analyzer.lifetime(output);
          managed_values_[output->id()] = true;
          continue;
        } else {
          VLOG(1) << "tensor " << output->name()
                  << " not placed on cpu so we cannot plan it";
        }
      } else /* possible if runtime pass didn't populate meta info */ {
        VLOG(1) << "tensor " << output->name() << " has no meta information";
      }

      managed_values_[output->id()] = true;
      value_to_allocation_spec[output].lifetime =
          alias_analyzer.lifetime(output);
    }
  }

  LOG(INFO) << "layout planner created with " << value_to_allocation_spec.size()
            << " values";

  switch (settings_.algorithmType()) {
    case torch::nativert::LayoutPlannerAlgorithmType::Bump: {
      algorithm_ = &BumpAllocationPlanner;
      break;
    }
    case torch::nativert::LayoutPlannerAlgorithmType::GreedyBySize: {
      algorithm_ = &GreedyBySizeAllocationPlanner;
      break;
    }
    case LayoutPlannerAlgorithmType::DisjointStorageGroups: {
      algorithm_ = &DisjointStorageGroupsPlanner;
      break;
    }
  }

  TORCH_CHECK(algorithm_ != nullptr, "algorithm can't be null");

  initialize_vectors(value_to_allocation_spec);

  auto exec_planner = ExecutionPlanner{graph};
  auto p = exec_planner.createPlan();
  for (const auto& freeable : p->valuesToFree) {
    for (const auto v : freeable) {
      if (!is_managed(v)) {
        unplanned_values_.push_back(v);
      }
    }
  }
}

void LayoutPlanner::initialize_vectors(
    c10::FastMap<const Value*, AllocationSpec> value_to_allocation_spec) {
  size_t num_managed = value_to_allocation_spec.size();

  planned_values_.resize(num_managed);
  planned_allocation_specs_.resize(num_managed);
  planned_values_historical_max_nbytes_ =
      std::vector<std::atomic_size_t>(num_managed);

  size_t i = 0;
  for (auto& [v, spec] : value_to_allocation_spec) {
    TORCH_CHECK(
        spec.lifetime.start <= spec.lifetime.end,
        "lifetime start must be before lifetime end");

    planned_values_[i] = v->id();
    planned_values_historical_max_nbytes_[i] = spec.size;
    planned_allocation_specs_[i] = spec;

    i++;
  }

  // for sanity in case anyone tries to use this after this method
  // is called with a bunch of junk (i.e., moved specs) in it
  value_to_allocation_spec.clear();
}

const std::vector<ValueId>& LayoutPlanner::get_planned_values() const {
  return planned_values_;
}

const std::vector<ValueId>& LayoutPlanner::get_unplanned_values() const {
  return unplanned_values_;
}

void LayoutPlanner::start_worker_if_not_started() {
  c10::call_once(worker_once_flag_, [&]() {
    // make sure plan is populated by the time this
    // returns for the first time :P
    create_plan();
    worker_ =
        std::thread([this]() { run_periodic([this] { create_plan(); }); });
  });
}

LayoutPlanner::~LayoutPlanner() {
  {
    std::unique_lock<std::mutex> l(mutex_);
    stopped_ = true;
  }
  cv_.notify_one();
  if (worker_.joinable()) {
    worker_.join();
  }
}

void LayoutPlanner::run_periodic(const std::function<void()>& f) {
  std::unique_lock<std::mutex> l(mutex_);
  while (!cv_.wait_for(
      l, settings_.planningInterval(), [&]() { return stopped_; })) {
    f();
  }
}

void LayoutPlanner::create_plan() {
  // update spec sizes to use historical maximums set
  // by execution frames before creating the new plan
  bool updated = false;
  for (const auto i : c10::irange(planned_allocation_specs_.size())) {
    auto& spec = planned_allocation_specs_[i];
    if (const auto new_size = planned_values_historical_max_nbytes_[i].load(
            std::memory_order_relaxed);
        new_size > spec.size) {
      spec.size = new_size;
      updated = true;
    }
  }

  if (updated) {
    plan_.write([p_new = (*algorithm_)(planned_allocation_specs_)](
                    LayoutPlan& plan) { plan = p_new; });
  }
}

} // namespace torch::nativert

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nativert/executor/memory`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/nativert/executor/memory/LayoutPlanner.h`
- `c10/util/Enumerate.h`
- `torch/nativert/executor/ExecutionPlanner.h`
- `torch/nativert/executor/memory/AliasAnalyzer.h`
- `torch/nativert/executor/memory/Bump.h`
- `torch/nativert/executor/memory/DisjointStorageGroups.h`
- `torch/nativert/executor/memory/GreedyBySize.h`


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
- [`LayoutPlanner.h_docs.md`](./LayoutPlanner.h_docs.md)
- [`LayoutManager.cpp_docs.md`](./LayoutManager.cpp_docs.md)
- [`DisjointStorageGroups.h_docs.md`](./DisjointStorageGroups.h_docs.md)
- [`GreedyBySize.h_docs.md`](./GreedyBySize.h_docs.md)
- [`Bump.cpp_docs.md`](./Bump.cpp_docs.md)
- [`DisjointStorageGroups.cpp_docs.md`](./DisjointStorageGroups.cpp_docs.md)
- [`LayoutPlannerAlgorithm.h_docs.md`](./LayoutPlannerAlgorithm.h_docs.md)


## Cross-References

- **File Documentation**: `LayoutPlanner.cpp_docs.md`
- **Keyword Index**: `LayoutPlanner.cpp_kw.md`
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
- [`LayoutPlanner.h_docs.md_docs.md`](./LayoutPlanner.h_docs.md_docs.md)
- [`LayoutPlanner.cpp_kw.md_docs.md`](./LayoutPlanner.cpp_kw.md_docs.md)
- [`DisjointStorageGroups.cpp_docs.md_docs.md`](./DisjointStorageGroups.cpp_docs.md_docs.md)
- [`DisjointStorageGroups.h_kw.md_docs.md`](./DisjointStorageGroups.h_kw.md_docs.md)
- [`GreedyBySize.cpp_docs.md_docs.md`](./GreedyBySize.cpp_docs.md_docs.md)
- [`FunctionSchema.cpp_kw.md_docs.md`](./FunctionSchema.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `LayoutPlanner.cpp_docs.md_docs.md`
- **Keyword Index**: `LayoutPlanner.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
