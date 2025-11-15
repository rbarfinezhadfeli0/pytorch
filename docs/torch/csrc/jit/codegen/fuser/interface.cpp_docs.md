# Documentation: `torch/csrc/jit/codegen/fuser/interface.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/codegen/fuser/interface.cpp`
- **Size**: 3,153 bytes (3.08 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/codegen/fuser/interface.h>

#include <torch/csrc/jit/codegen/fuser/compiler.h>
#include <torch/csrc/jit/codegen/fuser/executor.h>
#include <torch/csrc/jit/codegen/fuser/fallback.h>
#include <torch/csrc/jit/codegen/fuser/kernel_cache.h>

#include <c10/util/Exception.h>
#include <c10/util/Flags.h>
#include <stdexcept>

namespace torch::jit {

namespace detail {

#ifdef TORCH_ENABLE_LLVM
bool cpu_fuser_enabled = true;
#else
static bool cpu_fuser_enabled = false;
#endif

// note: this doesn't necessarily enable NNC because NVFuser might override it
static bool gpu_fuser_enabled = true;

} // namespace detail

int64_t registerFusion(const Node* fusion_group) {
  return fuser::registerFusion(fusion_group);
}

void runFusion(const int64_t key, Stack& stack) {
  const auto result = fuser::runFusion(key, stack);
  if (!result)
    fuser::runFallback(key, stack);
}

bool canFuseOnCPU() {
  return fuser::hasFusionBackend(DeviceType::CPU) && detail::cpu_fuser_enabled;
}

bool canFuseOnGPU() {
  return fuser::hasFusionBackend(DeviceType::CUDA) && detail::gpu_fuser_enabled;
}

void overrideCanFuseOnCPU(bool value) {
  detail::cpu_fuser_enabled = value;
}

void overrideCanFuseOnGPU(bool value) {
  detail::gpu_fuser_enabled = value;
}

// Uses the above interface by stuffing the graph into a node and treating that
// node as a fusion group.
std::vector<at::Tensor> debugLaunchGraph(
    Graph& graph,
    at::ArrayRef<at::Tensor> inputs) {
  // Creates a fusion group node
  auto wrapper_graph = std::make_shared<Graph>();
  Node* fusion_group = wrapper_graph->insertNode(
      wrapper_graph->createWithSubgraph(prim::FusionGroup));
  fusion_group->g_(attr::Subgraph, graph.copy());
  for (size_t i = 0; i < graph.inputs().size(); ++i) {
    fusion_group->addInput(wrapper_graph->addInput());
  }
  for (size_t i = 0; i < graph.outputs().size(); ++i) {
    wrapper_graph->registerOutput(fusion_group->addOutput());
  }

  // Creates the stack, registers and runs the fusion
  Stack stack = fmap<IValue>(inputs);
  const auto key = fuser::registerFusion(fusion_group);
  fuser::runFusion(key, stack);
  return fmap(stack, [](const IValue& iv) { return iv.toTensor(); });
}

std::string debugGetFusedKernelCode(
    Graph& graph,
    at::ArrayRef<at::Tensor> inputs) {
  // Creates a fusion group node
  auto wrapper_graph = std::make_shared<Graph>();
  Node* fusion_group = wrapper_graph->insertNode(
      wrapper_graph->createWithSubgraph(prim::FusionGroup));
  fusion_group->g_(attr::Subgraph, graph.copy());
  for (size_t i = 0; i < graph.inputs().size(); ++i) {
    fusion_group->addInput(wrapper_graph->addInput());
  }
  for (size_t i = 0; i < graph.outputs().size(); ++i) {
    wrapper_graph->registerOutput(fusion_group->addOutput());
  }

  // Creates the stack, registers and runs the fusion
  Stack stack = fmap<IValue>(inputs);
  const auto key = fuser::registerFusion(fusion_group);

  std::string code;
  TORCH_CHECK(
      fuser::runFusion(key, stack, &code), "Could not run fusion for graph")

  return code;
}

size_t nCompiledKernels() {
  return fuser::nCompiledKernels();
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `detail`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/codegen/fuser`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/codegen/fuser/interface.h`
- `torch/csrc/jit/codegen/fuser/compiler.h`
- `torch/csrc/jit/codegen/fuser/executor.h`
- `torch/csrc/jit/codegen/fuser/fallback.h`
- `torch/csrc/jit/codegen/fuser/kernel_cache.h`
- `c10/util/Exception.h`
- `c10/util/Flags.h`
- `stdexcept`


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

- **File Documentation**: `interface.cpp_docs.md`
- **Keyword Index**: `interface.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
