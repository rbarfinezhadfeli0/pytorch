# Documentation: `docs/torch/csrc/distributed/rpc/utils.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/rpc/utils.h_docs.md`
- **Size**: 6,438 bytes (6.29 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/rpc/utils.h`

## File Metadata

- **Path**: `torch/csrc/distributed/rpc/utils.h`
- **Size**: 3,790 bytes (3.70 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/core/Device.h>
#include <c10/core/Event.h>
#include <c10/core/Stream.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/utils/byte_order.h>

namespace torch::distributed::rpc {

// Parse error message and return RPCErrorType based on the message.
TORCH_API RPCErrorType getRPCErrorType(const JitFuture& jitFuture);
// Create an error string given the error description and error type
TORCH_API std::string makeRPCError(
    const std::string& rpcErrorStr,
    RPCErrorType errorType);

// Given an RPC message received as a request over the wire, deserialize it into
// the appropriate 'RpcCommandBase' type.
TORCH_API std::unique_ptr<RpcCommandBase> deserializeRequest(
    const Message& request);

// Given an RPC message received as a response over the wire, deserialize it
// into the appropriate 'RpcCommandBase' type, if the response is
// FORWARD_AUTOGRAD_RESP type, unwrap it, attach recvBackward() functions
// to received tensors and set the wrappedMsgType to its wrapped message type.
TORCH_API std::unique_ptr<RpcCommandBase> deserializeResponse(
    const Message& response,
    MessageType& wrappedMsgType);

// Given an RPC message received as a response over the wire, deserialize it
// into the valid IValue if the message is for a script rpc result,
// otherwise deserialize it into dummy none ivalue that will never be used.
// In this deserialization, we also attach recv rpc backward functions if
// needed.
IValue deserializeResptoIValueInternal(
    RpcCommandBase& rpc,
    MessageType messageType);
TORCH_API IValue deserializeRespToIValue(const Message& message);

// Note: format is subject to change and intended for RPCs.
// For saving persistently to disk, use torch::save().
TORCH_API std::string wireSerialize(
    const std::vector<char>& payload,
    const std::vector<at::Tensor>& tensors);

TORCH_API std::pair<std::vector<char>, std::vector<at::Tensor>> wireDeserialize(
    const void* data,
    size_t data_size);

// We use vector<char> as the type of blobs because it's what rpc::Message uses
// for its payload, even though it has the disadvantage that it cannot be
// allocated with uninitialized memory: it is always zeroed out.

// Some Tensors are effectively views of larger Tensors, where only a small
// subset of the Storage data is referenced. This normally is good and avoids
// copies when kept locally, but if we naively push the whole Storage over the
// wire, we'll end up with excess network traffic. This change clones tensors if
// we'd save at least half the data, and over a minimum hurdle.
TORCH_API c10::List<at::Tensor> cloneSparseTensors(
    const std::vector<at::Tensor>& tensors);

// Combines an original payload and wrapped payload into the original payload.
// Used to generate the overall payload for the wrapped RPC.
TORCH_API void writeWrappedPayload(
    std::vector<char>& originalPayload,
    std::vector<char>& additionalPayload);

// Reads the additional, wrapped payload from a wrapped RPC off of the input
// payload. After this, payload will contain the payload of the original,
// un-wrapped RPC.
TORCH_API std::vector<at::IValue> readWrappedPayload(
    std::vector<char>& payload,
    const rpc::Message& message);

// Takes a list of events from autograd profiler and populates them into
// profiledEvents to be carried over RPC.
TORCH_API void populateRemoteProfiledEvents(
    std::vector<torch::autograd::profiler::LegacyEvent>& profiledEvents,
    const torch::autograd::profiler::ProfilerConfig& profilerConfig,
    const std::vector<std::vector<torch::autograd::profiler::LegacyEvent>>&
        eventLists);

} // namespace torch::distributed::rpc

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/rpc`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/Device.h`
- `c10/core/Event.h`
- `c10/core/Stream.h`
- `torch/csrc/autograd/profiler.h`
- `torch/csrc/distributed/rpc/rpc_command_base.h`
- `torch/csrc/jit/serialization/pickle.h`
- `torch/csrc/utils/byte_order.h`


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

Files in the same folder (`torch/csrc/distributed/rpc`):

- [`request_callback.cpp_docs.md`](./request_callback.cpp_docs.md)
- [`python_rpc_handler.cpp_docs.md`](./python_rpc_handler.cpp_docs.md)
- [`tensorpipe_agent.h_docs.md`](./tensorpipe_agent.h_docs.md)
- [`torchscript_functions.cpp_docs.md`](./torchscript_functions.cpp_docs.md)
- [`utils.cpp_docs.md`](./utils.cpp_docs.md)
- [`unpickled_python_call.cpp_docs.md`](./unpickled_python_call.cpp_docs.md)
- [`request_callback.h_docs.md`](./request_callback.h_docs.md)
- [`rref_context.cpp_docs.md`](./rref_context.cpp_docs.md)
- [`request_callback_impl.h_docs.md`](./request_callback_impl.h_docs.md)
- [`py_rref.h_docs.md`](./py_rref.h_docs.md)


## Cross-References

- **File Documentation**: `utils.h_docs.md`
- **Keyword Index**: `utils.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/distributed/rpc`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/distributed/rpc`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/distributed/rpc`):

- [`script_resp.cpp_docs.md_docs.md`](./script_resp.cpp_docs.md_docs.md)
- [`python_rpc_handler.cpp_docs.md_docs.md`](./python_rpc_handler.cpp_docs.md_docs.md)
- [`tensorpipe_utils.h_kw.md_docs.md`](./tensorpipe_utils.h_kw.md_docs.md)
- [`request_callback_impl.h_docs.md_docs.md`](./request_callback_impl.h_docs.md_docs.md)
- [`types.cpp_docs.md_docs.md`](./types.cpp_docs.md_docs.md)
- [`utils.cpp_docs.md_docs.md`](./utils.cpp_docs.md_docs.md)
- [`rref_impl.h_kw.md_docs.md`](./rref_impl.h_kw.md_docs.md)
- [`rpc_agent.cpp_kw.md_docs.md`](./rpc_agent.cpp_kw.md_docs.md)
- [`request_callback_impl.cpp_kw.md_docs.md`](./request_callback_impl.cpp_kw.md_docs.md)
- [`script_call.cpp_docs.md_docs.md`](./script_call.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `utils.h_docs.md_docs.md`
- **Keyword Index**: `utils.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
