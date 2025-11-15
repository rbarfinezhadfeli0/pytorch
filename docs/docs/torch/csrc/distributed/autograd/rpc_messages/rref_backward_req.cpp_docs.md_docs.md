# Documentation: `docs/torch/csrc/distributed/autograd/rpc_messages/rref_backward_req.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/autograd/rpc_messages/rref_backward_req.cpp_docs.md`
- **Size**: 5,006 bytes (4.89 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/autograd/rpc_messages/rref_backward_req.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/autograd/rpc_messages/rref_backward_req.cpp`
- **Size**: 2,192 bytes (2.14 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/distributed/autograd/rpc_messages/rref_backward_req.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/jit/serialization/pickle.h>

namespace torch::distributed::autograd {

using rpc::Message;
using rpc::MessageType;

RRefBackwardReq::RRefBackwardReq(
    const rpc::RRefId& rrefId,
    int64_t autogradContextId,
    bool retainGraph)
    : rrefId_(rrefId),
      autogradContextId_(autogradContextId),
      retainGraph_(retainGraph) {}

c10::intrusive_ptr<Message> RRefBackwardReq::toMessageImpl() && {
  std::vector<at::IValue> ivalues;

  // Add all the fields.
  ivalues.emplace_back(rrefId_.toIValue());
  ivalues.emplace_back(autogradContextId_);
  ivalues.emplace_back(retainGraph_);

  // Now pickle using JIT pickler.
  std::vector<torch::Tensor> tensorTable;
  std::vector<char> payload =
      jit::pickle(c10::ivalue::Tuple::create(std::move(ivalues)), &tensorTable);

  return c10::make_intrusive<Message>(
      std::move(payload),
      std::move(tensorTable),
      MessageType::RREF_BACKWARD_REQ);
}

std::unique_ptr<RRefBackwardReq> RRefBackwardReq::fromMessage(
    const Message& message) {
  // Unpickle the message and retrieve tupleElements.
  auto payload = message.payload().data();
  auto payload_size = message.payload().size();
  IValue tuple = jit::unpickle(
      payload,
      payload_size,
      *rpc::RpcAgent::getCurrentRpcAgent()->getTypeResolver(),
      message.tensors());
  const auto& tupleElements = std::move(*std::move(tuple).toTuple()).elements();

  // Build RRefBackwardReq.
  TORCH_INTERNAL_ASSERT(tupleElements.size() == 3);

  // Retrieve all fields.
  bool retainGraph = tupleElements[2].toBool();
  int64_t autogradContextId = tupleElements[1].toInt();
  rpc::RRefId rrefId = rpc::RRefId::fromIValue(tupleElements[0]);

  return std::make_unique<RRefBackwardReq>(
      rrefId, autogradContextId, retainGraph);
}

const rpc::RRefId& RRefBackwardReq::getRRefId() const {
  return rrefId_;
}

int64_t RRefBackwardReq::getAutogradContextId() const {
  return autogradContextId_;
}

bool RRefBackwardReq::retainGraph() const {
  return retainGraph_;
}

} // namespace torch::distributed::autograd

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 0 function(s).

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

- `torch/csrc/distributed/autograd/rpc_messages/rref_backward_req.h`
- `torch/csrc/distributed/rpc/rpc_agent.h`
- `torch/csrc/jit/serialization/pickle.h`


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

- **File Documentation**: `rref_backward_req.cpp_docs.md`
- **Keyword Index**: `rref_backward_req.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/distributed/autograd/rpc_messages`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/distributed/autograd/rpc_messages`, which is part of the **core PyTorch library**.



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

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/csrc/distributed/autograd/rpc_messages`):

- [`propagate_gradients_req.h_kw.md_docs.md`](./propagate_gradients_req.h_kw.md_docs.md)
- [`rref_backward_req.cpp_kw.md_docs.md`](./rref_backward_req.cpp_kw.md_docs.md)
- [`propagate_gradients_req.cpp_kw.md_docs.md`](./propagate_gradients_req.cpp_kw.md_docs.md)
- [`cleanup_autograd_context_resp.h_kw.md_docs.md`](./cleanup_autograd_context_resp.h_kw.md_docs.md)
- [`autograd_metadata.h_docs.md_docs.md`](./autograd_metadata.h_docs.md_docs.md)
- [`rpc_with_profiling_resp.cpp_docs.md_docs.md`](./rpc_with_profiling_resp.cpp_docs.md_docs.md)
- [`rpc_with_profiling_req.cpp_docs.md_docs.md`](./rpc_with_profiling_req.cpp_docs.md_docs.md)
- [`autograd_metadata.cpp_kw.md_docs.md`](./autograd_metadata.cpp_kw.md_docs.md)
- [`rref_backward_req.h_kw.md_docs.md`](./rref_backward_req.h_kw.md_docs.md)
- [`propagate_gradients_req.h_docs.md_docs.md`](./propagate_gradients_req.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `rref_backward_req.cpp_docs.md_docs.md`
- **Keyword Index**: `rref_backward_req.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
