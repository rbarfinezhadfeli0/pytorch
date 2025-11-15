# Documentation: `torch/csrc/jit/backends/xnnpack/xnnpack_graph_builder.h`

## File Metadata

- **Path**: `torch/csrc/jit/backends/xnnpack/xnnpack_graph_builder.h`
- **Size**: 3,235 bytes (3.16 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/Functions.h>
#include <ATen/Utils.h>
#include <torch/torch.h>
#include <xnnpack.h>
#include <unordered_set>
#include <vector>

#include <torch/csrc/jit/backends/xnnpack/serialization/serializer.h>

namespace torch {
namespace jit {
namespace xnnpack {
namespace delegate {

class XNNGraph {
 private:
  const float output_min = -std::numeric_limits<float>::infinity();
  const float output_max = std::numeric_limits<float>::infinity();

  // serializer class
  XNNSerializer _serializer;
  // xnn subgraph
  xnn_subgraph_t _subgraph_ptr;
  // Set of all the tensor values throughout the jit graph
  std::unordered_set<torch::jit::Value*> _intermediate_tensors;
  // Set of all the tensor values mapped to the xnnpack ids
  std::unordered_map<torch::jit::Value*, uint32_t> _val_to_ids;
  // Vector containing the torch valued inputs/outputs,
  // must be ordered to preserve the order of input/outputs
  std::vector<torch::jit::Value*> _inputs;
  std::vector<torch::jit::Value*> _outputs;

  // Graph passes for optimizing and tracing torchscript graph
  // Essentially massaging the graph into a digestiable format for
  // xnnpack graph lowering.
  std::shared_ptr<torch::jit::Graph> optimizeAndTraceGraph(
      std::shared_ptr<torch::jit::Graph> graph,
      std::vector<c10::IValue>& example_inputs);

  // Gather all the intermediate tensor values within a graph. This
  // skips through all prim constants. The purpose of this is for defining
  // the tensor values beforehand for the xnnpack subgraph.
  void gatherTensorValues(std::shared_ptr<torch::jit::Graph>& graph);

  // Gathers the tensor values in a give node
  void gatherNodeInputs(torch::jit::Node& node);

  // Helper function to determine if a jit value is a graph input
  bool isGraphInput(torch::jit::Value* val);

  // Helper function to determine if a jit value is a graph output
  bool isGraphOutput(torch::jit::Value* val);

  // Defines all xnnpack nodes for the nodes in the graph
  void defineAllNodes(std::shared_ptr<torch::jit::Graph>& graph);

  // Defines all xnn tensor values used throughout the graph
  void defineAllTensorValues();

  // Makes a pass through the graph and throws if any ops are unsupported
  void checkOpsToDelegate(std::shared_ptr<torch::jit::Graph>& graph);

 public:
  XNNGraph() : _serializer(), _subgraph_ptr(nullptr) {
    xnn_status status = xnn_initialize(/*allocator =*/nullptr);
    TORCH_CHECK(xnn_status_success == status, "Failed to initialize xnnpack");
  }

  ~XNNGraph() {
    xnn_deinitialize();
    if (_subgraph_ptr != nullptr) {
      xnn_delete_subgraph(_subgraph_ptr);
    }
  }

  void buildXNNGraph(
      std::shared_ptr<torch::jit::Graph>& graph,
      std::vector<c10::IValue> example_inputs);

  void runGraphOnInputs(
      std::vector<at::Tensor> tensor_inputs,
      std::vector<at::Tensor> tensor_outputs);

  std::string serializedXNNGraph();

  std::vector<std::vector<long>> getGraphOutputShapes();
};

} // namespace delegate
} // namespace xnnpack
} // namespace jit
} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `jit`, `delegate`, `torch`, `xnnpack`

**Classes/Structs**: `XNNGraph`, `XNNSerializer`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/backends/xnnpack`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Functions.h`
- `ATen/Utils.h`
- `torch/torch.h`
- `xnnpack.h`
- `unordered_set`
- `vector`
- `torch/csrc/jit/backends/xnnpack/serialization/serializer.h`


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

Files in the same folder (`torch/csrc/jit/backends/xnnpack`):

- [`xnnpack_graph_builder.cpp_docs.md`](./xnnpack_graph_builder.cpp_docs.md)
- [`xnnpack_backend_lib.cpp_docs.md`](./xnnpack_backend_lib.cpp_docs.md)
- [`xnnpack_backend_preprocess.cpp_docs.md`](./xnnpack_backend_preprocess.cpp_docs.md)


## Cross-References

- **File Documentation**: `xnnpack_graph_builder.h_docs.md`
- **Keyword Index**: `xnnpack_graph_builder.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
