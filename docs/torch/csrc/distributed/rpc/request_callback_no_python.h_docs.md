# Documentation: `torch/csrc/distributed/rpc/request_callback_no_python.h`

## File Metadata

- **Path**: `torch/csrc/distributed/rpc/request_callback_no_python.h`
- **Size**: 3,904 bytes (3.81 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/request_callback.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/distributed/rpc/rref_impl.h>
#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/distributed/rpc/script_remote_call.h>

namespace torch::distributed::rpc {

// RequestCallback implementation with no Python dependencies.
class TORCH_API RequestCallbackNoPython : public RequestCallback {
 public:
  c10::intrusive_ptr<JitFuture> processMessage(
      Message& request,
      std::vector<c10::Stream> streams) const override;

 protected:
  virtual std::unique_ptr<RpcCommandBase> deserializePythonRpcCommand(
      std::unique_ptr<RpcCommandBase> rpc,
      const MessageType& messageType) const;

  virtual c10::intrusive_ptr<JitFuture> processScriptCall(
      RpcCommandBase& rpc,
      const std::vector<c10::Stream>& streams) const;

  virtual c10::intrusive_ptr<JitFuture> processPythonCall(
      RpcCommandBase& rpc,
      const std::vector<c10::Stream>& streams) const;

  c10::intrusive_ptr<JitFuture> assignOwnerRRef(
      const RRefId& rrefId,
      const RRefId& forkId,
      const c10::intrusive_ptr<JitFuture>& valueFuture) const;

  virtual c10::intrusive_ptr<JitFuture> processScriptRemoteCall(
      RpcCommandBase& rpc,
      const std::vector<c10::Stream>& streams) const;

  virtual c10::intrusive_ptr<JitFuture> processPythonRemoteCall(
      RpcCommandBase& rpc,
      const std::vector<c10::Stream>& streams) const;

  c10::intrusive_ptr<JitFuture> retrieveOwnerRRef(const RRefId& rrefId) const;

  c10::intrusive_ptr<JitFuture> processScriptRRefFetchCall(
      RpcCommandBase& rpc) const;

  virtual c10::intrusive_ptr<JitFuture> processPythonRRefFetchCall(
      RpcCommandBase& rpc) const;

  c10::intrusive_ptr<JitFuture> processRRefUserDelete(
      RpcCommandBase& rpc) const;

  c10::intrusive_ptr<JitFuture> processRRefChildAccept(
      RpcCommandBase& rpc) const;

  c10::intrusive_ptr<JitFuture> processRRefForkRequest(
      RpcCommandBase& rpc) const;

  c10::intrusive_ptr<JitFuture> processForwardAutogradReq(
      RpcCommandBase& rpc,
      const std::vector<c10::Stream>& streams) const;

  c10::intrusive_ptr<JitFuture> processBackwardAutogradReq(
      RpcCommandBase& rpc,
      const std::vector<c10::Stream>& streams) const;

  c10::intrusive_ptr<JitFuture> processCleanupAutogradContextReq(
      RpcCommandBase& rpc) const;

  c10::intrusive_ptr<JitFuture> processRunWithProfilingReq(
      RpcCommandBase& rpc) const;

  virtual void handleRRefDelete(c10::intrusive_ptr<RRef>& rref) const;

  c10::intrusive_ptr<JitFuture> processRpc(
      RpcCommandBase& rpc,
      const MessageType& messageType,
      const std::vector<c10::Stream>& streams) const;

  virtual c10::intrusive_ptr<JitFuture> processRpcWithErrors(
      RpcCommandBase& rpc,
      const MessageType& messageType,
      const std::vector<c10::Stream>& streams) const;

  c10::intrusive_ptr<Message> handleError(
      const std::exception& e,
      const MessageType messageType,
      int64_t messageId) const;

  virtual bool cudaAvailable() const;

  virtual c10::intrusive_ptr<JitFuture> processRRefBackward(
      RpcCommandBase& rpc) const;

  // Helpers to run user-defined functions, operators and other computations.

  c10::intrusive_ptr<JitFuture> runJitOperator(
      const jit::Operator& op,
      std::vector<at::IValue>& stack,
      const std::vector<c10::Stream>& streams) const;

  // Helpers to convert various kinds of objects into already-completed futures.

  c10::intrusive_ptr<JitFuture> asFuture(IValue value, TypePtr type) const;

  c10::intrusive_ptr<JitFuture> asFuture(
      c10::intrusive_ptr<Message> message) const;

  c10::intrusive_ptr<JitFuture> asFuture(std::exception_ptr err) const;
};

} // namespace torch::distributed::rpc

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/rpc`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/distributed/rpc/message.h`
- `torch/csrc/distributed/rpc/request_callback.h`
- `torch/csrc/distributed/rpc/rpc_command_base.h`
- `torch/csrc/distributed/rpc/rref_impl.h`
- `torch/csrc/distributed/rpc/script_call.h`
- `torch/csrc/distributed/rpc/script_remote_call.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

- **File Documentation**: `request_callback_no_python.h_docs.md`
- **Keyword Index**: `request_callback_no_python.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
