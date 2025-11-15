# Documentation: `docs/torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_resp.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_resp.cpp_docs.md`
- **Size**: 8,653 bytes (8.45 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_resp.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_resp.cpp`
- **Size**: 5,841 bytes (5.70 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/util/irange.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_resp.h>
#include <torch/csrc/distributed/rpc/utils.h>
#include <torch/csrc/jit/serialization/pickle.h>

namespace torch::distributed::autograd {
using rpc::RpcCommandBase;

constexpr auto kProfileEventsStartIdx = 3;
// This constructor is called when creating the RpcProfilingResp before sending
// it as a message over the wire.
RpcWithProfilingResp::RpcWithProfilingResp(
    rpc::MessageType messageType,
    c10::intrusive_ptr<rpc::Message> wrappedMessage,
    std::vector<torch::autograd::profiler::LegacyEvent> profiledEvents,
    rpc::ProfilingId profilingId)
    : messageType_(messageType),
      wrappedMessage_(std::move(wrappedMessage)),
      tensors_(wrappedMessage_->tensors()),
      profiledEvents_(std::move(profiledEvents)),
      profilingId_(profilingId) {
  TORCH_INTERNAL_ASSERT(
      messageType_ == rpc::MessageType::RUN_WITH_PROFILING_RESP,
      "Incorrect Message type");
  wrappedMessageType_ = wrappedMessage_->type();
}
// this constructor is called in fromMessage() which is called when
// reconstructing this RPC command when processing a message of this type
RpcWithProfilingResp::RpcWithProfilingResp(
    rpc::MessageType messageType,
    std::unique_ptr<rpc::RpcCommandBase> wrappedRpc,
    rpc::MessageType wrappedMessageType,
    std::vector<torch::Tensor> tensors,
    std::vector<torch::autograd::profiler::LegacyEvent> profiledEvents,
    rpc::ProfilingId profilingId)
    : messageType_(messageType),
      wrappedRpc_(std::move(wrappedRpc)),
      wrappedMessageType_(wrappedMessageType),
      tensors_(std::move(tensors)),
      profiledEvents_(std::move(profiledEvents)),
      profilingId_(profilingId) {
  TORCH_INTERNAL_ASSERT(wrappedRpc_ != nullptr, "wrapped RPC cannot be null");
}

std::unique_ptr<RpcCommandBase> RpcWithProfilingResp::moveWrappedRpc() && {
  TORCH_INTERNAL_ASSERT(wrappedRpc_ != nullptr, "wrappedRpc cannot be null!");
  return std::move(wrappedRpc_);
}

rpc::MessageType RpcWithProfilingResp::wrappedMessageType() const {
  return wrappedMessageType_;
}

std::vector<torch::autograd::profiler::LegacyEvent> RpcWithProfilingResp::
    getProfiledEvents() const {
  return profiledEvents_;
}

const rpc::ProfilingId& RpcWithProfilingResp::getProfilingId() const {
  return profilingId_;
}

void RpcWithProfilingResp::setWrappedRpc(
    std::unique_ptr<RpcCommandBase> wrappedRpc) {
  wrappedRpc_ = std::move(wrappedRpc);
}

c10::intrusive_ptr<rpc::Message> RpcWithProfilingResp::toMessageImpl() && {
  auto wrappedMsgId = wrappedMessage_->id();
  auto wrappedMsgType = wrappedMessage_->type();
  auto wrappedPayload = std::move(*wrappedMessage_).movePayload();
  // Wrapped payload should not be empty
  TORCH_INTERNAL_ASSERT(
      !wrappedPayload.empty(), "Wrapped payload cannot be empty");
  // Create ivalues to send over
  std::vector<at::IValue> ivalues{wrappedMsgType, profilingId_.toIValue()};
  // Attach the serialized events.
  ivalues.emplace_back(static_cast<int32_t>(profiledEvents_.size()));
  for (const auto& e : profiledEvents_) {
    ivalues.emplace_back(e.toIValue());
  }
  std::vector<torch::Tensor> tensorTable;
  std::vector<char> profilingPayload =
      jit::pickle(c10::ivalue::Tuple::create(std::move(ivalues)), &tensorTable);
  rpc::writeWrappedPayload(wrappedPayload, profilingPayload);

  auto returnMsg = c10::make_intrusive<rpc::Message>(
      std::move(wrappedPayload),
      std::move(tensors_),
      messageType_,
      wrappedMsgId);
  return returnMsg;
}

RpcCommandBase& RpcWithProfilingResp::wrappedRpc() {
  TORCH_INTERNAL_ASSERT(wrappedRpc_ != nullptr, "wrappedRpc cannot be null!");
  return *wrappedRpc_;
}

// Runs on client when deserializing this message.
std::unique_ptr<RpcWithProfilingResp> RpcWithProfilingResp::fromMessage(
    const rpc::Message& message) {
  rpc::MessageType origMsgType = message.type();
  std::vector<torch::Tensor> tensors = message.tensors();
  int64_t msgId = message.id();
  auto payload = message.payload();
  auto tupleElements = rpc::readWrappedPayload(payload, message);
  // Ensure that we have the expected number of elements
  TORCH_INTERNAL_ASSERT(
      tupleElements.size() >= kProfileEventsStartIdx,
      c10::str(
          "Expected payload size of at least ",
          kProfileEventsStartIdx,
          " but got size ",
          tupleElements.size()));
  rpc::MessageType wrappedMsgType =
      static_cast<rpc::MessageType>(tupleElements[0].toInt());
  rpc::ProfilingId profilingId = rpc::ProfilingId::fromIValue(tupleElements[1]);
  auto profiledEventsSize = tupleElements[2].toInt();
  std::vector<torch::autograd::profiler::LegacyEvent> remoteEvents;
  remoteEvents.reserve(profiledEventsSize);
  for (const auto i : c10::irange(
           kProfileEventsStartIdx,
           kProfileEventsStartIdx + profiledEventsSize)) {
    TORCH_CHECK(static_cast<size_t>(i) < tupleElements.size());
    // Reconstruct remote event from the ivalues.
    torch::autograd::profiler::LegacyEvent fromIvalueEvent =
        torch::autograd::profiler::LegacyEvent::fromIValue(tupleElements[i]);
    remoteEvents.push_back(std::move(fromIvalueEvent));
  }

  auto wrappedMessage = c10::make_intrusive<rpc::Message>(
      std::move(payload), std::move(tensors), wrappedMsgType, msgId);
  TORCH_INTERNAL_ASSERT(
      wrappedMessage->isResponse(),
      "Messages wrapped with profiling response must be responses.");
  std::unique_ptr<RpcCommandBase> wrappedRpc =
      deserializeResponse(*wrappedMessage, wrappedMsgType);
  return std::make_unique<RpcWithProfilingResp>(
      origMsgType,
      std::move(wrappedRpc),
      wrappedMsgType,
      std::move(wrappedMessage->tensors()),
      std::move(remoteEvents),
      profilingId);
}
} // namespace torch::distributed::autograd

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `remote`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/autograd/rpc_messages`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/irange.h`
- `torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_resp.h`
- `torch/csrc/distributed/rpc/utils.h`
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
- [`rref_backward_resp.h_docs.md`](./rref_backward_resp.h_docs.md)
- [`rref_backward_req.h_docs.md`](./rref_backward_req.h_docs.md)
- [`cleanup_autograd_context_resp.h_docs.md`](./cleanup_autograd_context_resp.h_docs.md)
- [`rpc_with_profiling_resp.h_docs.md`](./rpc_with_profiling_resp.h_docs.md)
- [`autograd_metadata.h_docs.md`](./autograd_metadata.h_docs.md)
- [`rpc_with_profiling_req.cpp_docs.md`](./rpc_with_profiling_req.cpp_docs.md)
- [`propagate_gradients_resp.cpp_docs.md`](./propagate_gradients_resp.cpp_docs.md)
- [`rpc_with_autograd.cpp_docs.md`](./rpc_with_autograd.cpp_docs.md)


## Cross-References

- **File Documentation**: `rpc_with_profiling_resp.cpp_docs.md`
- **Keyword Index**: `rpc_with_profiling_resp.cpp_kw.md`
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
- [`rpc_with_profiling_req.cpp_docs.md_docs.md`](./rpc_with_profiling_req.cpp_docs.md_docs.md)
- [`autograd_metadata.cpp_kw.md_docs.md`](./autograd_metadata.cpp_kw.md_docs.md)
- [`rref_backward_req.h_kw.md_docs.md`](./rref_backward_req.h_kw.md_docs.md)
- [`propagate_gradients_req.h_docs.md_docs.md`](./propagate_gradients_req.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `rpc_with_profiling_resp.cpp_docs.md_docs.md`
- **Keyword Index**: `rpc_with_profiling_resp.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
