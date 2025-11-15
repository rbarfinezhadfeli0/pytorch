# Documentation: `docs/torch/csrc/jit/codegen/cuda/interface.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/codegen/cuda/interface.cpp_docs.md`
- **Size**: 6,272 bytes (6.12 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/codegen/cuda/interface.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/codegen/cuda/interface.cpp`
- **Size**: 4,011 bytes (3.92 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/codegen/cuda/interface.h>

#include <ATen/DynamicLibrary.h>
#include <ATen/core/dispatch/OperatorOptions.h>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/TensorShape.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/register_ops_utils.h>

namespace torch::jit::fuser::cuda {

static std::atomic<bool> cuda_fusion_guard_mode{true};

bool isEnabled() {
  TORCH_WARN_ONCE("torch::jit::fuser::cuda::isEnabled() is deprecated");
  return false;
}

bool setEnabled(bool is_enabled) {
  TORCH_WARN_ONCE("torch::jit::fuser::cuda::setEnabled() is deprecated");
  TORCH_INTERNAL_ASSERT(
      !is_enabled,
      "nvfuser support in torchscript is removed and cannot be enabled!");
  return false;
}

bool canBeEnabled() {
  TORCH_WARN_ONCE(
      "torch::jit::fuser::cuda::nvfuserCanBeEnabled() is deprecated");
  return false;
}

bool getSingletonFusion() {
  TORCH_WARN_ONCE(
      "torch::jit::fuser::cuda::getSingletonFusion() is deprecated");
  return false;
}

bool setSingletonFusion(bool value) {
  TORCH_WARN_ONCE(
      "torch::jit::fuser::cuda::setSingletonFusion() is deprecated");
  TORCH_INTERNAL_ASSERT(
      !value,
      "nvfuser support in torchscript is removed and singleton fusion cannot be enabled!");
  return false;
}

bool getHorizontalFusion() {
  TORCH_WARN_ONCE(
      "torch::jit::fuser::cuda::getHorizontalFusion() is deprecated");
  return false;
}

bool setHorizontalFusion(bool value) {
  TORCH_WARN_ONCE(
      "torch::jit::fuser::cuda::setHorizontalFusion() is deprecated");
  TORCH_INTERNAL_ASSERT(
      !value,
      "nvfuser support in torchscript is removed and horizontal fusion cannot be enabled!");
  return false;
}

std::atomic<bool>& getCudaFusionGuardMode() {
  TORCH_WARN_ONCE(
      "torch::jit::fuser::cuda::getCudaFusionGuardMode() is deprecated");
  return cuda_fusion_guard_mode;
}

CudaFuserInterface* getFuserInterface() {
  static CudaFuserInterface fuser_interface_;
  return &fuser_interface_;
}

void compileFusionGroup(Node* fusion_node) {
  TORCH_WARN_ONCE(
      "torch::jit::fuser::cuda::compileFusionGroup() is deprecated");
  TORCH_CHECK(
      getFuserInterface()->fn_compile_n != nullptr,
      "Running the CUDA fuser requires a CUDA build.");
  getFuserInterface()->fn_compile_n(fusion_node);
}

void runFusionGroup(const Node* fusion_node, Stack& stack) {
  TORCH_WARN_ONCE("torch::jit::fuser::cuda::runFusionGroup() is deprecated");
  TORCH_CHECK(
      getFuserInterface()->fn_run_n_s != nullptr,
      "Running the CUDA fuser requires a CUDA build.");
  getFuserInterface()->fn_run_n_s(fusion_node, stack);
}

void fuseGraph(std::shared_ptr<Graph>& graph) {
  if (!isEnabled()) {
    return;
  }

  TORCH_WARN_ONCE("nvfuser integration in TorchScript is deprecated.");
  TORCH_CHECK(
      getFuserInterface()->fn_fuse_graph != nullptr,
      "Running the CUDA fuser requires a CUDA build.");
  getFuserInterface()->fn_fuse_graph(graph);
}

bool canFuseNode(const Node* node) {
  TORCH_WARN_ONCE("torch::jit::fuser::cuda::canFuseNode() is deprecated");
  return getFuserInterface()->fn_can_fuse_n != nullptr &&
      getFuserInterface()->fn_can_fuse_n(node);
}

void InsertProfileNodesForCUDAFuser(ProfilingRecord* pr) {
  TORCH_WARN_ONCE(
      "torch::jit::fuser::cuda::InsertProfileNodesForCUDAFuser() is deprecated");
  if (getFuserInterface()->fn_insert_profile_inodes) {
    getFuserInterface()->fn_insert_profile_inodes(pr);
  }
}

bool profileNode(const Node* node) {
  TORCH_WARN_ONCE("torch::jit::fuser::cuda::profileNode() is deprecated");
  return getFuserInterface()->fn_profile_n != nullptr &&
      getFuserInterface()->fn_profile_n(node);
}

bool skipNode(const std::string& symbol_str, bool flip) {
  TORCH_WARN_ONCE("torch::jit::fuser::cuda::skipNode() is deprecated");
  return getFuserInterface()->fn_skip_n != nullptr &&
      getFuserInterface()->fn_skip_n(symbol_str, flip);
}

} // namespace torch::jit::fuser::cuda

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 17 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/codegen/cuda`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/codegen/cuda/interface.h`
- `ATen/DynamicLibrary.h`
- `ATen/core/dispatch/OperatorOptions.h`
- `ATen/native/NonSymbolicBC.h`
- `ATen/native/TensorShape.h`
- `c10/util/irange.h`
- `torch/csrc/jit/runtime/custom_operator.h`
- `torch/csrc/jit/runtime/register_ops_utils.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`torch/csrc/jit/codegen/cuda`):

- [`README.md_docs.md`](./README.md_docs.md)
- [`interface.h_docs.md`](./interface.h_docs.md)


## Cross-References

- **File Documentation**: `interface.cpp_docs.md`
- **Keyword Index**: `interface.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/codegen/cuda`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/codegen/cuda`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/torch/csrc/jit/codegen/cuda`):

- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`interface.h_docs.md_docs.md`](./interface.h_docs.md_docs.md)
- [`interface.cpp_kw.md_docs.md`](./interface.cpp_kw.md_docs.md)
- [`README.md_kw.md_docs.md`](./README.md_kw.md_docs.md)
- [`interface.h_kw.md_docs.md`](./interface.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `interface.cpp_docs.md_docs.md`
- **Keyword Index**: `interface.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
