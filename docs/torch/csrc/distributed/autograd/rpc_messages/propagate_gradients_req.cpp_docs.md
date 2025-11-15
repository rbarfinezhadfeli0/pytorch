# Documentation: `torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_req.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_req.cpp`
- **Size**: 2,911 bytes (2.84 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_req.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/jit/serialization/pickle.h>

#include <c10/util/irange.h>

namespace torch::distributed::autograd {

using rpc::Message;
using rpc::MessageType;
using torch::autograd::Variable;

PropagateGradientsReq::PropagateGradientsReq(
    const AutogradMetadata& autogradMetadata,
    std::vector<Variable> grads,
    bool retainGraph)
    : autogradMetadata_(autogradMetadata),
      grads_(std::move(grads)),
      retainGraph_(retainGraph) {}

c10::intrusive_ptr<Message> PropagateGradientsReq::toMessageImpl() && {
  std::vector<at::IValue> ivalues;
  // Add all the grad tensors.
  ivalues.reserve(grads_.size() + 3);
  for (const auto& grad : grads_) {
    ivalues.emplace_back(grad);
  }

  // Now add autograd metadata.
  ivalues.emplace_back(autogradMetadata_.autogradContextId);
  ivalues.emplace_back(autogradMetadata_.autogradMessageId);

  // Add retain graph.
  ivalues.emplace_back(retainGraph_);

  // Now pickle using JIT pickler.
  std::vector<torch::Tensor> tensorTable;
  std::vector<char> payload =
      jit::pickle(c10::ivalue::Tuple::create(std::move(ivalues)), &tensorTable);

  return c10::make_intrusive<Message>(
      std::move(payload),
      std::move(tensorTable),
      MessageType::BACKWARD_AUTOGRAD_REQ);
}

std::unique_ptr<PropagateGradientsReq> PropagateGradientsReq::fromMessage(
    const Message& message) {
  // Unpickle the message and retrieve tupleElements.
  auto payload = message.payload().data();
  auto payload_size = message.payload().size();
  IValue tuple = jit::unpickle(
      payload,
      payload_size,
      *rpc::RpcAgent::getCurrentRpcAgent()->getTypeResolver(),
      message.tensors());
  const auto& tupleElements = tuple.toTupleRef().elements();

  // Build PropagateGradientsReq.
  TORCH_INTERNAL_ASSERT(tupleElements.size() >= 3);

  // Retrieve retainGraph.
  bool retainGraph = tupleElements.back().toBool();

  // Build AutogradMetadata.
  int64_t autogradMessageId = tupleElements[tupleElements.size() - 2].toInt();
  int64_t autogradContextId = tupleElements[tupleElements.size() - 3].toInt();

  AutogradMetadata autogradMetadata(autogradContextId, autogradMessageId);

  // Retrieve the gradient tensors.
  std::vector<Variable> grads(tupleElements.size() - 3);
  for (const auto i : c10::irange(tupleElements.size() - 3)) {
    grads[i] = tupleElements[i].toTensor();
  }

  return std::make_unique<PropagateGradientsReq>(
      autogradMetadata, grads, retainGraph);
}

const AutogradMetadata& PropagateGradientsReq::getAutogradMetadata() {
  return autogradMetadata_;
}

const std::vector<torch::autograd::Variable>& PropagateGradientsReq::
    getGrads() {
  return grads_;
}

bool PropagateGradientsReq::retainGraph() {
  return retainGraph_;
}

} // namespace torch::distributed::autograd

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 1 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/autograd/rpc_messages`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_req.h`
- `torch/csrc/distributed/rpc/rpc_agent.h`
- `torch/csrc/jit/serialization/pickle.h`
- `c10/util/irange.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/csrc/distributed/autograd/rpc_messages`):

- [`rpc_with_profiling_req.h_docs.md`](./rpc_with_profiling_req.h_docs.md)
- [`rpc_with_profiling_resp.cpp_docs.md`](./rpc_with_profiling_resp.cpp_docs.md)
- [`rref_backward_resp.h_docs.md`](./rref_backward_resp.h_docs.md)
- [`rref_backward_req.h_docs.md`](./rref_backward_req.h_docs.md)
- [`cleanup_autograd_context_resp.h_docs.md`](./cleanup_autograd_context_resp.h_docs.md)
- [`rpc_with_profiling_resp.h_docs.md`](./rpc_with_profiling_resp.h_docs.md)
- [`autograd_metadata.h_docs.md`](./autograd_metadata.h_docs.md)
- [`rpc_with_profiling_req.cpp_docs.md`](./rpc_with_profiling_req.cpp_docs.md)
- [`propagate_gradients_resp.cpp_docs.md`](./propagate_gradients_resp.cpp_docs.md)
- [`rpc_with_autograd.cpp_docs.md`](./rpc_with_autograd.cpp_docs.md)


## Cross-References

- **File Documentation**: `propagate_gradients_req.cpp_docs.md`
- **Keyword Index**: `propagate_gradients_req.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
