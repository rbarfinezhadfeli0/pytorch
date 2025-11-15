# Documentation: `docs/torch/csrc/jit/codegen/onednn/kernel.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/codegen/onednn/kernel.h_docs.md`
- **Size**: 5,300 bytes (5.18 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/codegen/onednn/kernel.h`

## File Metadata

- **Path**: `torch/csrc/jit/codegen/onednn/kernel.h`
- **Size**: 2,694 bytes (2.63 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <unordered_map>

#include <oneapi/dnnl/dnnl_graph.hpp>
#include <torch/csrc/jit/codegen/onednn/LlgaTensorImpl.h>
#include <torch/csrc/jit/codegen/onednn/graph_helper.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/interpreter.h>

#include <c10/util/CallOnce.h>

namespace torch::jit::fuser::onednn {

using ArgSpec = LlgaTensorDesc;
using ArgSpecs = std::vector<ArgSpec>;
using RunArg = dnnl::graph::tensor;
using RunArgs = std::vector<RunArg>;
using TensorArgs = std::vector<at::Tensor>;

class LlgaKernel {
 public:
  explicit LlgaKernel(const Node* fusionNode);

  void run(Stack& stack);

  void initialize(const TensorArgs& inputs);

  const std::string& debugName() const {
    return debugName_;
  }

 private:
  bool useOpaqueLayout(size_t offset) const;

  // PyTorch copy constants inside the subgraph instead of referencing them.
  // Constants inputs to the partition are no longer in the graph->inputs().
  // Need use the tid retrieved from the partition to find the missing
  // constant inputs.
  void initializeConstantInputs();

  ArgSpecs initializeInputSpecs(const TensorArgs& inputs);

  ArgSpecs initializeOutputSpecs() const;

  dnnl::graph::compiled_partition compile(
      const dnnl::graph::partition& partition);

  std::map<size_t, int64_t> initializeTensorIdToOccurence() const;

  std::tuple<RunArgs, RunArgs> prepareRunArgs(
      const TensorArgs& inputs,
      TensorArgs& outputs) const;

  static std::string genDebugName() {
    static size_t debugId = 0;
    return "LlgaPartition_" + std::to_string(debugId++);
  }

  static dnnl::graph::logical_tensor toLogicalTensor(const ArgSpec& s) {
    return s.logical_tensor();
  }

  at::Device device_ = at::kCPU;
  const Node* fusionNode_;
  std::shared_ptr<Graph> graph_;
  int64_t nGraphInputs_ = 0; // number of inputs to graph_ on the IR
  int64_t nOutputs_ = 0;
  std::map<size_t, Value*> tensorIdToValue_;
  std::vector<int64_t> runArgsIdx_;
  dnnl::graph::partition partition_;
  // nPartitionInputs_ is the actual number of inputs to partition_ of graph_
  // needed by the backend.
  // nPartitionInputs_ = nGraphInputs_ + constantInputs_.size() since Constant
  // inputs are copied to the inside of the subgraph
  int64_t nPartitionInputs_;
  dnnl::graph::compiled_partition compilation_;
  std::set<size_t> initializedInputIds_;
  std::vector<Value*> constantValues_;
  TensorArgs constantInputs_;
  ArgSpecs inputSpecs_;
  ArgSpecs outputSpecs_;
  std::vector<dnnl::graph::logical_tensor> constantLogicalTensors_;
  std::string debugName_;
  c10::once_flag initialized_flag;
  bool is_initialized_ = false;
};

} // namespace torch::jit::fuser::onednn

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `LlgaKernel`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/codegen/onednn`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `unordered_map`
- `oneapi/dnnl/dnnl_graph.hpp`
- `torch/csrc/jit/codegen/onednn/LlgaTensorImpl.h`
- `torch/csrc/jit/codegen/onednn/graph_helper.h`
- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/jit/runtime/interpreter.h`
- `c10/util/CallOnce.h`


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

Files in the same folder (`torch/csrc/jit/codegen/onednn`):

- [`graph_rewriter.cpp_docs.md`](./graph_rewriter.cpp_docs.md)
- [`guard_shape.cpp_docs.md`](./guard_shape.cpp_docs.md)
- [`prepare_binary.h_docs.md`](./prepare_binary.h_docs.md)
- [`kernel.cpp_docs.md`](./kernel.cpp_docs.md)
- [`graph_fuser.h_docs.md`](./graph_fuser.h_docs.md)
- [`decompose_silu.cpp_docs.md`](./decompose_silu.cpp_docs.md)
- [`prepare_binary.cpp_docs.md`](./prepare_binary.cpp_docs.md)
- [`graph_helper.cpp_docs.md`](./graph_helper.cpp_docs.md)
- [`register_interface.cpp_docs.md`](./register_interface.cpp_docs.md)


## Cross-References

- **File Documentation**: `kernel.h_docs.md`
- **Keyword Index**: `kernel.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/codegen/onednn`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/codegen/onednn`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/csrc/jit/codegen/onednn`):

- [`graph_rewriter.cpp_docs.md_docs.md`](./graph_rewriter.cpp_docs.md_docs.md)
- [`kernel.cpp_docs.md_docs.md`](./kernel.cpp_docs.md_docs.md)
- [`decompose_silu.cpp_kw.md_docs.md`](./decompose_silu.cpp_kw.md_docs.md)
- [`defer_size_check.h_kw.md_docs.md`](./defer_size_check.h_kw.md_docs.md)
- [`graph_fuser.h_kw.md_docs.md`](./graph_fuser.h_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`graph_fuser.h_docs.md_docs.md`](./graph_fuser.h_docs.md_docs.md)
- [`interface.h_docs.md_docs.md`](./interface.h_docs.md_docs.md)
- [`layout_propagation.h_kw.md_docs.md`](./layout_propagation.h_kw.md_docs.md)
- [`graph_helper.cpp_kw.md_docs.md`](./graph_helper.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `kernel.h_docs.md_docs.md`
- **Keyword Index**: `kernel.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
