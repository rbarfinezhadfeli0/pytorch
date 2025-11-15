# Documentation: `torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_resp.h`

## File Metadata

- **Path**: `torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_resp.h`
- **Size**: 2,470 bytes (2.41 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/distributed/rpc/types.h>

namespace torch::distributed::autograd {
class TORCH_API RpcWithProfilingResp : public rpc::RpcCommandBase {
 public:
  // For sending RPCs over the wire
  RpcWithProfilingResp(
      rpc::MessageType messageType,
      c10::intrusive_ptr<rpc::Message> wrappedMessage,
      std::vector<torch::autograd::profiler::LegacyEvent> profiledEvents,
      rpc::ProfilingId profilingId);

  // For receiving RPCs. Used in from message when converting a message received
  // over the wire.
  RpcWithProfilingResp(
      rpc::MessageType messageType,
      std::unique_ptr<rpc::RpcCommandBase> wrappedRpc,
      rpc::MessageType wrappedMessageType,
      std::vector<torch::Tensor> tensors,
      std::vector<torch::autograd::profiler::LegacyEvent> profiledEvents,
      rpc::ProfilingId profilingId);
  c10::intrusive_ptr<rpc::Message> toMessageImpl() && override;
  static std::unique_ptr<RpcWithProfilingResp> fromMessage(
      const rpc::Message& message);
  // Retrieve remote Events
  std::vector<torch::autograd::profiler::LegacyEvent> getProfiledEvents() const;
  // Retrieve the globally unique profiling ID corresponding to this command.
  const rpc::ProfilingId& getProfilingId() const;
  // Retrieve the original RPC which this ProfilingRPC wraps.
  RpcCommandBase& wrappedRpc();
  // Destructively move the wrapped RPC.
  std::unique_ptr<RpcCommandBase> moveWrappedRpc() &&;
  // Message type of the wrapped RPC
  rpc::MessageType wrappedMessageType() const;
  // Set the wrapped RPC for this RPC.
  void setWrappedRpc(std::unique_ptr<RpcCommandBase> wrappedRpc);

 private:
  // message type
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const rpc::MessageType messageType_;
  // wrapped message
  c10::intrusive_ptr<rpc::Message> wrappedMessage_;
  std::unique_ptr<RpcCommandBase> wrappedRpc_;
  rpc::MessageType wrappedMessageType_;
  std::vector<torch::Tensor> tensors_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const std::vector<torch::autograd::profiler::LegacyEvent> profiledEvents_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const rpc::ProfilingId profilingId_;
};
} // namespace torch::distributed::autograd

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/autograd/rpc_messages`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/autograd/profiler.h`
- `torch/csrc/distributed/rpc/message.h`
- `torch/csrc/distributed/rpc/rpc_agent.h`
- `torch/csrc/distributed/rpc/rpc_command_base.h`
- `torch/csrc/distributed/rpc/types.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


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

Files in the same folder (`torch/csrc/distributed/autograd/rpc_messages`):

- [`rpc_with_profiling_req.h_docs.md`](./rpc_with_profiling_req.h_docs.md)
- [`rpc_with_profiling_resp.cpp_docs.md`](./rpc_with_profiling_resp.cpp_docs.md)
- [`rref_backward_resp.h_docs.md`](./rref_backward_resp.h_docs.md)
- [`rref_backward_req.h_docs.md`](./rref_backward_req.h_docs.md)
- [`cleanup_autograd_context_resp.h_docs.md`](./cleanup_autograd_context_resp.h_docs.md)
- [`autograd_metadata.h_docs.md`](./autograd_metadata.h_docs.md)
- [`rpc_with_profiling_req.cpp_docs.md`](./rpc_with_profiling_req.cpp_docs.md)
- [`propagate_gradients_resp.cpp_docs.md`](./propagate_gradients_resp.cpp_docs.md)
- [`rpc_with_autograd.cpp_docs.md`](./rpc_with_autograd.cpp_docs.md)


## Cross-References

- **File Documentation**: `rpc_with_profiling_resp.h_docs.md`
- **Keyword Index**: `rpc_with_profiling_resp.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
