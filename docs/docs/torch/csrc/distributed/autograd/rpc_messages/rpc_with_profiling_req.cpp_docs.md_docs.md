# Documentation: `docs/torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_req.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_req.cpp_docs.md`
- **Size**: 8,435 bytes (8.24 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_req.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_req.cpp`
- **Size**: 5,668 bytes (5.54 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_req.h>
#include <torch/csrc/distributed/rpc/utils.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <vector>

namespace torch::distributed::autograd {

constexpr auto kProfilingResponseElementExpectedSize = 3;

using rpc::RpcCommandBase;

// This constructor is called when creating the RpcWithProfilingReq on the
// client.
RpcWithProfilingReq::RpcWithProfilingReq(
    rpc::MessageType messageType,
    c10::intrusive_ptr<rpc::Message> wrappedMessage,
    torch::autograd::profiler::ProfilerConfig&& profilerConfig,
    rpc::ProfilingId profilingKeyId)
    : messageType_(messageType),
      wrappedMessage_(std::move(wrappedMessage)),
      tensors_(wrappedMessage_->tensors()),
      profilerConfig_(std::move(profilerConfig)),
      profilingKeyId_(profilingKeyId) {
  TORCH_INTERNAL_ASSERT(
      messageType_ == rpc::MessageType::RUN_WITH_PROFILING_REQ,
      c10::str(
          "Incorrect message type, expected message type ",
          rpc::MessageType::RUN_WITH_PROFILING_REQ));
  wrappedMessageType_ = wrappedMessage_->type();
}

// this constructor is only called in fromMessage() which is called in
// deserializeRequest(). It is called when reconstructing the
// RpcWithProfilingReq on the remote end.
RpcWithProfilingReq::RpcWithProfilingReq(
    rpc::MessageType messageType,
    std::unique_ptr<rpc::RpcCommandBase> wrappedRpc,
    rpc::MessageType wrappedMessageType,
    std::vector<torch::Tensor> tensors,
    torch::autograd::profiler::ProfilerConfig&& profilerConfig,
    rpc::ProfilingId profilingKeyId)
    : messageType_(messageType),
      wrappedRpc_(std::move(wrappedRpc)),
      wrappedMessageType_(wrappedMessageType),
      tensors_(std::move(tensors)),
      profilerConfig_(std::move(profilerConfig)),
      profilingKeyId_(profilingKeyId) {
  TORCH_INTERNAL_ASSERT(wrappedRpc_ != nullptr, "wrappedRpc can't be null");
}

rpc::MessageType RpcWithProfilingReq::wrappedMessageType() const {
  return wrappedMessageType_;
}

void RpcWithProfilingReq::setWrappedRpc(
    std::unique_ptr<RpcCommandBase> wrappedRpc) {
  wrappedRpc_ = std::move(wrappedRpc);
}

c10::intrusive_ptr<rpc::Message> RpcWithProfilingReq::toMessageImpl() && {
  // save the original message ID and type before moving it.
  auto wrappedMsgId = wrappedMessage_->id();
  auto wrappedMsgType = wrappedMessage_->type();
  // destructively move the wrappedMessage and get the payload. Now the payload
  // of wrappedMessage won't be in a valid state.
  auto wrappedPayload = std::move(*wrappedMessage_).movePayload();
  // The wrapped payload should not be empty
  TORCH_INTERNAL_ASSERT(
      !wrappedPayload.empty(), "Wrapped payload should not be empty.");
  // Create the ivalues to send over. We need to send the original message type
  // and id, as well as some profiling metadata.
  std::vector<at::IValue> ivalues{
      wrappedMsgType, profilerConfig_.toIValue(), profilingKeyId_.toIValue()};
  // Pickle it into a char payload to be sent over the wire.
  std::vector<torch::Tensor> tensorTable;
  std::vector<char> profilingPayload =
      jit::pickle(c10::ivalue::Tuple::create(std::move(ivalues)), &tensorTable);
  // add the profiling payload to the wrapped payload
  rpc::writeWrappedPayload(wrappedPayload, profilingPayload);
  // Put the wrapped payload into a message to return.
  auto returnMsg = c10::make_intrusive<rpc::Message>(
      std::move(wrappedPayload),
      std::move(tensors_),
      messageType_,
      wrappedMsgId);

  return returnMsg;
}

RpcCommandBase& RpcWithProfilingReq::wrappedRpc() {
  TORCH_INTERNAL_ASSERT(wrappedRpc_ != nullptr, "wrappedRpc cannot be null!");
  return *wrappedRpc_;
}

torch::autograd::profiler::ProfilerConfig RpcWithProfilingReq::
    getProfilingConfig() const {
  return profilerConfig_;
}

const rpc::ProfilingId& RpcWithProfilingReq::getProfilingId() const {
  return profilingKeyId_;
}

std::unique_ptr<RpcWithProfilingReq> RpcWithProfilingReq::fromMessage(
    const rpc::Message& message) {
  rpc::MessageType origMsgType = message.type();
  std::vector<torch::Tensor> tensors = message.tensors();
  int64_t msgId = message.id();
  auto payload = message.payload();
  auto tupleElements = rpc::readWrappedPayload(payload, message);
  // Ensure that we have the expected number of elements
  TORCH_INTERNAL_ASSERT(
      tupleElements.size() == kProfilingResponseElementExpectedSize,
      c10::str(
          "Expected payload of size ",
          kProfilingResponseElementExpectedSize,
          " but got ",
          tupleElements.size()));
  rpc::MessageType wrappedMsgType =
      static_cast<rpc::MessageType>(tupleElements[0].toInt());
  // Create a config to be enabled on this node that is a replica of the
  // state on the requesting node.
  torch::autograd::profiler::ProfilerConfig cfg =
      torch::autograd::profiler::ProfilerConfig::fromIValue(tupleElements[1]);

  rpc::ProfilingId profilerId = rpc::ProfilingId::fromIValue(tupleElements[2]);

  // Create new message type and build wrapped RPC
  auto wrappedMessage = c10::make_intrusive<rpc::Message>(
      std::move(payload), std::move(tensors), wrappedMsgType, msgId);
  TORCH_INTERNAL_ASSERT(
      wrappedMessage->isRequest(),
      "Messages wrapped with profiling requests must be requests.");
  std::unique_ptr<RpcCommandBase> wrappedRpc =
      deserializeRequest(*wrappedMessage);

  return std::make_unique<RpcWithProfilingReq>(
      origMsgType,
      std::move(wrappedRpc),
      wrappedMsgType,
      std::move(wrappedMessage->tensors()),
      std::move(cfg),
      profilerId);
}
} // namespace torch::distributed::autograd

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

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

- `torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_req.h`
- `torch/csrc/distributed/rpc/utils.h`
- `torch/csrc/jit/serialization/pickle.h`
- `vector`


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
- [`propagate_gradients_resp.cpp_docs.md`](./propagate_gradients_resp.cpp_docs.md)
- [`rpc_with_autograd.cpp_docs.md`](./rpc_with_autograd.cpp_docs.md)


## Cross-References

- **File Documentation**: `rpc_with_profiling_req.cpp_docs.md`
- **Keyword Index**: `rpc_with_profiling_req.cpp_kw.md`
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
- [`autograd_metadata.cpp_kw.md_docs.md`](./autograd_metadata.cpp_kw.md_docs.md)
- [`rref_backward_req.h_kw.md_docs.md`](./rref_backward_req.h_kw.md_docs.md)
- [`propagate_gradients_req.h_docs.md_docs.md`](./propagate_gradients_req.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `rpc_with_profiling_req.cpp_docs.md_docs.md`
- **Keyword Index**: `rpc_with_profiling_req.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
