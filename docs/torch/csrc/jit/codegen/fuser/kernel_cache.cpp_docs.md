# Documentation: `torch/csrc/jit/codegen/fuser/kernel_cache.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/codegen/fuser/kernel_cache.cpp`
- **Size**: 2,702 bytes (2.64 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/codegen/fuser/kernel_cache.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/shape_analysis.h>

#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace torch::jit::fuser {

struct KernelCacheImpl {
  // Note: std::unordered_map does not invalidate references even if rehashing
  // occurs. This is a critical property for thread-safety.
  std::mutex mutex_;
  int64_t kernel_counter{0};

  // Map of fusion key to KernelSpec
  std::unordered_map<int64_t, KernelSpec> specMap_;

  // Map of pretty-printed graph string to fusion key
  // Used to check if a graph has already been cached in specMap_
  std::unordered_map<std::string, int64_t> graphToKey_;
};

static KernelCacheImpl& getKernelCache() {
  static KernelCacheImpl cache;
  return cache;
}

int64_t debugNumCachedKernelSpecs() {
  auto& cache = getKernelCache();
  std::lock_guard<std::mutex> guard{cache.mutex_};
  return cache.specMap_.size();
}

std::shared_ptr<Graph> normalizeGraphForCache(
    const std::shared_ptr<Graph>& graph) {
  auto result = Canonicalize(graph, /*keep_unique_names=*/false);
  EraseShapeInformation(result);
  return result;
}

// TODO: lookup by historic string key to start, then issue key
// as appropriate for faster lookup in the future
// precondition: graph has been normalized via normalizeGraphForCache
int64_t store(std::shared_ptr<Graph> graph) {
  auto& cache = getKernelCache();
  std::string repr = graph->toString(false);

  std::lock_guard<std::mutex> guard{cache.mutex_};
  const auto key = cache.kernel_counter++;
  cache.specMap_.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(key),
      std::forward_as_tuple(key, graph));
  cache.graphToKey_.emplace(std::move(repr), key);
  return key;
}

// XXX: Does not grab mutex
static std::optional<KernelSpec*> nolock_retrieve(
    KernelCacheImpl& cache,
    const int64_t key) {
  auto it = cache.specMap_.find(key);
  if (it == cache.specMap_.end())
    return std::nullopt;
  return &(it->second);
}

std::optional<KernelSpec*> retrieve(const int64_t key) {
  auto& cache = getKernelCache();
  std::lock_guard<std::mutex> guard{cache.mutex_};
  return nolock_retrieve(cache, key);
}

// precondition: graph has been normalized via normalizeGraphForCache
std::optional<KernelSpec*> lookupGraph(const std::shared_ptr<Graph>& graph) {
  auto& cache = getKernelCache();
  std::string repr = graph->toString(false);

  std::lock_guard<std::mutex> guard{cache.mutex_};
  auto it = cache.graphToKey_.find(repr);
  if (it == cache.graphToKey_.end())
    return std::nullopt;
  return nolock_retrieve(cache, it->second);
}

} // namespace torch::jit::fuser

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `KernelCacheImpl`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/codegen/fuser`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/codegen/fuser/kernel_cache.h`
- `torch/csrc/jit/passes/canonicalize.h`
- `torch/csrc/jit/passes/shape_analysis.h`
- `cstdint`
- `mutex`
- `unordered_map`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`torch/csrc/jit/codegen/fuser`):

- [`compiler.h_docs.md`](./compiler.h_docs.md)
- [`codegen.cpp_docs.md`](./codegen.cpp_docs.md)
- [`kernel_spec.h_docs.md`](./kernel_spec.h_docs.md)
- [`executor.h_docs.md`](./executor.h_docs.md)
- [`fallback.h_docs.md`](./fallback.h_docs.md)
- [`arg_spec.h_docs.md`](./arg_spec.h_docs.md)
- [`fused_kernel.h_docs.md`](./fused_kernel.h_docs.md)
- [`tensor_info.h_docs.md`](./tensor_info.h_docs.md)
- [`executor.cpp_docs.md`](./executor.cpp_docs.md)
- [`tensor_desc.h_docs.md`](./tensor_desc.h_docs.md)


## Cross-References

- **File Documentation**: `kernel_cache.cpp_docs.md`
- **Keyword Index**: `kernel_cache.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
