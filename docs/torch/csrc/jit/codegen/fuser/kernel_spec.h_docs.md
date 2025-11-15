# Documentation: `torch/csrc/jit/codegen/fuser/kernel_spec.h`

## File Metadata

- **Path**: `torch/csrc/jit/codegen/fuser/kernel_spec.h`
- **Size**: 4,401 bytes (4.30 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/codegen/fuser/arg_spec.h>
#include <torch/csrc/jit/codegen/fuser/fused_kernel.h>
#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <optional>

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace torch::jit::fuser {

// Helper struct containing partition information: the number of tensors
// created and the dimension the partitioning is performed on.
// Note: created during upfront compilation, once the tensors are known
// at runtime the partition info is logically combined with the tensor
// descriptions to create PartitionDesc objects.
struct TORCH_API PartitionInfo {
  PartitionInfo(const int64_t _nSubTensors, const int64_t _dim)
      : nSubTensors_{_nSubTensors}, dim_{_dim} {}

  int64_t nSubTensors() const {
    return nSubTensors_;
  }
  int64_t dim() const {
    return dim_;
  }

 private:
  int64_t nSubTensors_;
  int64_t dim_;
};

// "Kernel Specification." - Contains device-independent fusion information.
// Each kernel specification contains a map of instantiated generated functions
// that implement some or most of its functionality. Multiple generated
// functions are needed by each abstract specification because of different
// devices (cpu vs gpu, different gpus) and different inputs (int vs float,
// contiguous vs discontiguous).
// Note: uses a mutex to control access to its kernel store
// Note: unordered containers do not invalidate references/pointers on
//   rehashing, which is critical for thread-safety.
// TODO: allow abstract kernels to use multiple generated kernels
// TODO: allow abstract kernels to reuse generated kernels from common pool
struct TORCH_API KernelSpec {
  // Note: assumes the spec is a single block
  // Note: This is the appropriate place to generalize if you want to add other
  //  passes to upfront compilation that walk the graph.
  KernelSpec(const int64_t _key, const std::shared_ptr<Graph>& _graph)
      : key_{_key},
        graph_{_graph},
        code_{_graph, "<fused code>"},
        nInputs_{_graph->inputs().size()}

  {
    // No need to iterate over reference since n is pointer
    for (const auto n : graph_->nodes()) {
      static_assert(std::is_pointer_v<decltype(n)>, "n must be a pointer");
      if (n->kind() == aten::rand_like) {
        has_random_ = true;
        break;
      }
    }
    nTensorInputs_ = std::count_if(
        graph_->inputs().begin(), graph_->inputs().end(), [](const Value* v) {
          return v->type()->isSubtypeOf(*TensorType::get());
        });
  }

  // Getters
  int64_t key() const {
    return key_;
  }
  std::shared_ptr<Graph> graph() const {
    return graph_;
  }
  const Code& code() const {
    return code_;
  }
  int64_t nInputs() const {
    return nInputs_;
  }
  int64_t nTensorInputs() const {
    return nTensorInputs_;
  }

  std::vector<std::vector<int64_t>>& inputBroadcastGroups() {
    return inputBroadcastGroups_;
  }
  const std::vector<std::vector<int64_t>>& inputBroadcastGroups() const {
    return inputBroadcastGroups_;
  }

  std::vector<PartitionInfo>& inputChunks() {
    return inputChunks_;
  }
  const std::vector<PartitionInfo>& inputChunks() const {
    return inputChunks_;
  }

  bool hasRandom() const {
    return has_random_;
  }

  // Cache functions
  std::optional<std::shared_ptr<FusedKernel>> findKernel(
      const ArgSpec& arg_spec) const {
    std::lock_guard<std::mutex> guard{mutex_};
    const auto it = kernels_.find(arg_spec);
    if (it == kernels_.end())
      return std::nullopt;
    return it->second;
  }
  void cacheKernel(
      const ArgSpec& arg_spec,
      const std::shared_ptr<FusedKernel>& kernel) const {
    std::lock_guard<std::mutex> guard{mutex_};
    kernels_.emplace(arg_spec, kernel);
  }

 private:
  int64_t key_;
  std::shared_ptr<Graph> graph_;
  Code code_;
  uint64_t nInputs_;
  uint64_t nTensorInputs_{};
  std::vector<std::vector<int64_t>> inputBroadcastGroups_;
  std::vector<PartitionInfo> inputChunks_;
  bool has_random_{false};
  mutable std::mutex mutex_;
  mutable std::
      unordered_map<ArgSpec, std::shared_ptr<FusedKernel>, c10::hash<ArgSpec>>
          kernels_;
};

} // namespace torch::jit::fuser

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `containing`, `TORCH_API`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/codegen/fuser`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ATen.h`
- `ATen/core/stack.h`
- `torch/csrc/Export.h`
- `torch/csrc/jit/codegen/fuser/arg_spec.h`
- `torch/csrc/jit/codegen/fuser/fused_kernel.h`
- `torch/csrc/jit/codegen/fuser/interface.h`
- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/jit/runtime/interpreter.h`
- `optional`
- `cstdint`
- `memory`
- `mutex`
- `unordered_map`
- `vector`


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
- [`executor.h_docs.md`](./executor.h_docs.md)
- [`fallback.h_docs.md`](./fallback.h_docs.md)
- [`arg_spec.h_docs.md`](./arg_spec.h_docs.md)
- [`fused_kernel.h_docs.md`](./fused_kernel.h_docs.md)
- [`tensor_info.h_docs.md`](./tensor_info.h_docs.md)
- [`executor.cpp_docs.md`](./executor.cpp_docs.md)
- [`tensor_desc.h_docs.md`](./tensor_desc.h_docs.md)


## Cross-References

- **File Documentation**: `kernel_spec.h_docs.md`
- **Keyword Index**: `kernel_spec.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
