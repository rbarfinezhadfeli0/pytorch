# Documentation: `docs/torch/csrc/distributed/autograd/utils.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/autograd/utils.h_docs.md`
- **Size**: 4,900 bytes (4.79 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/autograd/utils.h`

## File Metadata

- **Path**: `torch/csrc/distributed/autograd/utils.h`
- **Size**: 2,647 bytes (2.58 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/distributed/autograd/context/context.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_resp.h>

namespace torch::distributed::autograd {

// This method is used to attach the 'send' autograd function to the autograd
// graph when we use RPC. This method creates a new 'send' autograd function
// and attaches the provided tensors as next_edges to the 'send' function. In
// addition to this, it also registers the send function in the provided
// autograd context. Finally, the RPC message is updated with appropriate
// autograd information for the recipient.
TORCH_API void addSendRpcBackward(
    const ContextPtr& autogradContext,
    const AutogradMetadata& autogradMetadata,
    std::vector<torch::Tensor>& tensors);

// This method is used to attach the 'recv' autograd function to the autograd
// graph when we use RPC. This method creates a new 'recv' autograd function
// and attaches the provided tensors as inputs to the 'recv' function. It
// creates a new autograd context if needed and registers the 'recv' function
// with this context.
//
// Returns a pointer to the autograd context created.
TORCH_API ContextPtr addRecvRpcBackward(
    const AutogradMetadata& autogradMetadata,
    std::vector<torch::Tensor>& tensors,
    rpc::worker_id_t fromWorkerId,
    const rpc::DeviceMap& deviceMap);

// This method is a wrapper utility used internally to wrap autograd info
// and attach autograd function for each type of rpc call if it has valid
// context and tensors require grads or forceGradRecording is true, in this
// case, return RpcWithAutograd message; otherwise return original rpc message.
// NB: forceGradRecording is useful when the request does not contain any tensor
// but the corresponding response does.
TORCH_API c10::intrusive_ptr<rpc::Message> getMessageWithAutograd(
    const rpc::worker_id_t dstId,
    c10::intrusive_ptr<rpc::Message> wrappedRpcMsg,
    rpc::MessageType msgType,
    bool forceGradRecording = false,
    const rpc::DeviceMap& deviceMap = {});

// Send message after autograd checking
TORCH_API c10::intrusive_ptr<c10::ivalue::Future> sendMessageWithAutograd(
    rpc::RpcAgent& agent,
    const rpc::WorkerInfo& dst,
    c10::intrusive_ptr<rpc::Message> wrappedRpcMsg,
    bool forceGradRecording = false,
    const float rpcTimeoutSeconds = torch::distributed::rpc::kUnsetRpcTimeout,
    bool forceDisableProfiling = false);

} // namespace torch::distributed::autograd

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/autograd`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/distributed/autograd/context/context.h`
- `torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h`
- `torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_req.h`
- `torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_resp.h`


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

Files in the same folder (`torch/csrc/distributed/autograd`):

- [`python_autograd.h_docs.md`](./python_autograd.h_docs.md)
- [`utils.cpp_docs.md`](./utils.cpp_docs.md)
- [`autograd.cpp_docs.md`](./autograd.cpp_docs.md)
- [`init.cpp_docs.md`](./init.cpp_docs.md)
- [`autograd.h_docs.md`](./autograd.h_docs.md)


## Cross-References

- **File Documentation**: `utils.h_docs.md`
- **Keyword Index**: `utils.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/distributed/autograd`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/distributed/autograd`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/csrc/distributed/autograd`):

- [`utils.cpp_docs.md_docs.md`](./utils.cpp_docs.md_docs.md)
- [`python_autograd.h_kw.md_docs.md`](./python_autograd.h_kw.md_docs.md)
- [`utils.h_kw.md_docs.md`](./utils.h_kw.md_docs.md)
- [`autograd.h_docs.md_docs.md`](./autograd.h_docs.md_docs.md)
- [`init.cpp_kw.md_docs.md`](./init.cpp_kw.md_docs.md)
- [`init.cpp_docs.md_docs.md`](./init.cpp_docs.md_docs.md)
- [`autograd.h_kw.md_docs.md`](./autograd.h_kw.md_docs.md)
- [`python_autograd.h_docs.md_docs.md`](./python_autograd.h_docs.md_docs.md)
- [`utils.cpp_kw.md_docs.md`](./utils.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `utils.h_docs.md_docs.md`
- **Keyword Index**: `utils.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
