# Documentation: `docs/torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h_docs.md`
- **Size**: 6,344 bytes (6.20 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h`

## File Metadata

- **Path**: `torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h`
- **Size**: 3,506 bytes (3.42 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/distributed/autograd/rpc_messages/autograd_metadata.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>

namespace torch::distributed::autograd {

// Represents an RPC that includes autograd information. This class basically
// wraps another `RpcCommandBase` object which represents the actual RPC and has
// additional autograd information associated with that RPC.
class TORCH_API RpcWithAutograd final : public rpc::RpcCommandBase {
 public:
  // Used when we are sending an RPC over the wire.
  RpcWithAutograd(
      rpc::worker_id_t fromWorkerId,
      rpc::MessageType messageType,
      const AutogradMetadata& autogradMetadata,
      c10::intrusive_ptr<rpc::Message> wrappedMessage,
      rpc::DeviceMap deviceMap = {});

  // Used when receiving an RPC over the wire.
  RpcWithAutograd(
      rpc::worker_id_t fromWorkerId,
      rpc::MessageType messageType,
      const AutogradMetadata& autogradMetadata,
      std::unique_ptr<rpc::RpcCommandBase> wrappedRpc,
      rpc::MessageType wrappedMessageType,
      std::vector<torch::Tensor> tensors,
      rpc::DeviceMap deviceMap = {});

  c10::intrusive_ptr<rpc::Message> toMessageImpl() && override;

  static std::unique_ptr<RpcWithAutograd> fromMessage(
      const rpc::Message& message);

  // Retrieves tensors as part of this RPC, which need to be considered for
  // autograd computations.
  std::vector<torch::Tensor>& tensors();

  const AutogradMetadata& autogradMetadata() const;

  RpcCommandBase& wrappedRpc();

  void setWrappedRpc(std::unique_ptr<RpcCommandBase> wrappedRpc);

  std::unique_ptr<RpcCommandBase> moveWrappedRpc() &&;

  // Message type of the wrapped RPC.
  rpc::MessageType wrappedMessageType() const;

  // Retrieve the worker id from which the RPC originated.
  rpc::worker_id_t fromWorkerId() const;

  // Retrieve the device map.
  const rpc::DeviceMap& deviceMap();

 private:
  // WorkerId from which this RPC originated. This is necessary for knowing
  // which worker we need to contact during the backward pass.
  rpc::worker_id_t fromWorkerId_;

  // Message type for this call.
  rpc::MessageType messageType_;

  AutogradMetadata autogradMetadata_;

  // Since wrappedMessage_ is destructively constructed from wrappedRpc_,
  // they are valid exclusively. They are used for different purpose.
  // wrappedRpc_ is used while constructing receive rpcWithAutograd;
  // wrappedMessage_ is used while constructing send rpcWithAutograd;

  // When receive rpcWithAutograd is constructed fromMessage, it is valid;
  // When send rpcWithAutograd is constructed before toMessage, it is nullptr;
  std::unique_ptr<RpcCommandBase> wrappedRpc_;

  // Serialized message representing wrappedRpc_. Used mostly as a cache to
  // avoid serializing the request twice.
  // When receive rpcWithAutograd is constructed fromMessage, it is nullptr;
  // When send rpcWithAutograd is constructed before toMessage, it is valid;
  c10::intrusive_ptr<rpc::Message> wrappedMessage_;

  // message type of the wrappedMessage, this is stored separately since
  // wrappedMessage_ is not always guaranteed to be populated.
  rpc::MessageType wrappedMessageType_;

  // Tensors part of the wrappedRpc that need to be considered for autograd.
  std::vector<torch::Tensor> tensors_;

  // Device mapping for tensors that are sent across an RPC to another node.
  rpc::DeviceMap deviceMap_;
};

} // namespace torch::distributed::autograd

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `basically`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/autograd/rpc_messages`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/distributed/autograd/rpc_messages/autograd_metadata.h`
- `torch/csrc/distributed/rpc/rpc_agent.h`
- `torch/csrc/distributed/rpc/rpc_command_base.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

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
- [`rpc_with_profiling_resp.h_docs.md`](./rpc_with_profiling_resp.h_docs.md)
- [`autograd_metadata.h_docs.md`](./autograd_metadata.h_docs.md)
- [`rpc_with_profiling_req.cpp_docs.md`](./rpc_with_profiling_req.cpp_docs.md)
- [`propagate_gradients_resp.cpp_docs.md`](./propagate_gradients_resp.cpp_docs.md)
- [`rpc_with_autograd.cpp_docs.md`](./rpc_with_autograd.cpp_docs.md)


## Cross-References

- **File Documentation**: `rpc_with_autograd.h_docs.md`
- **Keyword Index**: `rpc_with_autograd.h_kw.md`
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

- Implements or uses **caching** mechanisms.
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

- **File Documentation**: `rpc_with_autograd.h_docs.md_docs.md`
- **Keyword Index**: `rpc_with_autograd.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
