# Documentation: `docs/torch/csrc/distributed/rpc/torchscript_functions.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/rpc/torchscript_functions.cpp_docs.md`
- **Size**: 8,778 bytes (8.57 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/rpc/torchscript_functions.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/rpc/torchscript_functions.cpp`
- **Size**: 5,779 bytes (5.64 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/ThreadLocalState.h>
#include <fmt/format.h>
#include <torch/csrc/autograd/record_function_ops.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/profiler/remote_profiler_manager.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/rref_proto.h>
#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/distributed/rpc/torchscript_functions.h>
#include <torch/csrc/distributed/rpc/utils.h>

namespace torch::distributed::rpc {

c10::intrusive_ptr<JitFuture> rpcTorchscript(
    const std::string& dstWorkerName,
    const c10::QualifiedName& qualifiedName,
    const c10::FunctionSchema& functionSchema,
    std::vector<c10::IValue> stack,
    const float rpcTimeoutSeconds,
    const bool isAsyncExecution) {
  c10::intrusive_ptr<torch::autograd::profiler::PythonRecordFunction> record;
  auto shouldProfile = torch::autograd::profiler::profilerEnabled() &&
      !torch::distributed::rpc::RemoteProfilerManager::getInstance()
           .isCurrentKeySet();
  if (shouldProfile) {
    auto rpcAsyncJitKey = fmt::format(
        "rpc_async_jit#{}({} -> {})",
        qualifiedName
            .qualifiedName(), /* name of torchscript function being run */
        RpcAgent::getCurrentRpcAgent()->getWorkerInfo().name_,
        dstWorkerName);
    record =
        torch::autograd::profiler::record_function_enter_new(rpcAsyncJitKey);
    auto& remoteProfilerManager =
        torch::distributed::rpc::RemoteProfilerManager::getInstance();
    remoteProfilerManager.setCurrentKey(rpcAsyncJitKey);
  }
  auto scriptCall = std::make_unique<ScriptCall>(
      qualifiedName, std::move(stack), isAsyncExecution);
  auto rpcAgentPtr = RpcAgent::getCurrentRpcAgent();
  auto jitFuture = autograd::sendMessageWithAutograd(
      *rpcAgentPtr,
      rpcAgentPtr->getWorkerInfo(dstWorkerName),
      std::move(*scriptCall).toMessage(),
      true /*forceGradRecording*/,
      rpcTimeoutSeconds);

  // Get function return type to construct JitFuture.
  auto const& returns = functionSchema.returns();
  // Script call only allows single IValue returned.
  TORCH_INTERNAL_ASSERT(
      returns.size() == 1,
      "Return value of an annotated torchScript function should be a single "
      "IValue.",
      returns.size());
  auto returnType = returns.at(0).type();

  // Create a JIT future and pass it to futMessage's callback to set state
  // of the JIT future.
  auto futPtr = jitFuture->createInstance(returnType);
  jitFuture->addCallback(at::wrapPropagateTLSState([futPtr](JitFuture& future) {
    if (future.hasError()) {
      futPtr->setError(future.exception_ptr());
    } else {
      futPtr->markCompleted(
          deserializeRespToIValue(
              *future.constValue().toCustomClass<Message>()),
          future.storages());
    }
  }));
  if (shouldProfile) {
    auto profiledFutPtr =
        torch::autograd::profiler::_call_end_callbacks_on_fut_new(
            record, futPtr);
    return profiledFutPtr;
  }
  return futPtr;
}

c10::intrusive_ptr<RRef> remoteTorchscript(
    const std::string& dstWorkerName,
    const c10::QualifiedName& qualifiedName,
    const c10::FunctionSchema& functionSchema,
    std::vector<c10::IValue>& stack,
    const float rpcTimeoutSeconds,
    const bool isAsyncExecution) {
  auto rpcAgentPtr = RpcAgent::getCurrentRpcAgent();
  auto dstWorkerInfo = rpcAgentPtr->getWorkerInfo(dstWorkerName);
  auto& ctx = RRefContext::getInstance();

  // Get function return type to construct UserRRef.
  auto const& returns = functionSchema.returns();
  // Script call only allows single IValue returned.
  TORCH_INTERNAL_ASSERT(
      returns.size() == 1,
      "Return value of an annotated torchScript function should be a single "
      "IValue.",
      returns.size());
  auto returnType = returns.at(0).type();

  if (ctx.getWorkerId() != dstWorkerInfo.id_) {
    auto userRRefPtr = ctx.createUserRRef(dstWorkerInfo.id_, returnType);

    auto scriptRemoteCall = std::make_unique<ScriptRemoteCall>(
        qualifiedName,
        std::move(stack),
        userRRefPtr->rrefId(),
        userRRefPtr->forkId(),
        isAsyncExecution);

    auto jitFuture = torch::distributed::autograd::sendMessageWithAutograd(
        *rpcAgentPtr,
        dstWorkerInfo,
        std::move(*scriptRemoteCall).toMessage(),
        true /*forceGradRecording*/,
        rpcTimeoutSeconds /* timeout */);

    userRRefPtr->registerOwnerCreationFuture(jitFuture);
    ctx.addPendingUser(userRRefPtr->forkId(), userRRefPtr);
    jitFuture->addCallback(at::wrapPropagateTLSState(
        [forkId{userRRefPtr->forkId()}](JitFuture& future) {
          callback::confirmPendingUser(future, forkId);
        }));

    return userRRefPtr;
  } else {
    auto ownerRRefPtr = ctx.createOwnerRRef(returnType);
    // prevent this owner RRef from being deleted due to other forks
    ctx.addSelfAsFork(ownerRRefPtr);

    auto scriptRemoteCall = std::make_unique<ScriptRemoteCall>(
        qualifiedName,
        std::move(stack),
        ownerRRefPtr->rrefId(),
        ownerRRefPtr->rrefId(),
        isAsyncExecution);

    auto jitFuture = torch::distributed::autograd::sendMessageWithAutograd(
        *rpcAgentPtr,
        dstWorkerInfo,
        std::move(*scriptRemoteCall).toMessage(),
        true /*forceGradRecording*/,
        rpcTimeoutSeconds /* timeout */);

    ownerRRefPtr->registerOwnerCreationFuture(jitFuture);
    jitFuture->addCallback(at::wrapPropagateTLSState(
        [ownerRRefId = ownerRRefPtr->rrefId()](JitFuture& future) {
          callback::finishCreatingOwnerRRef(future, ownerRRefId);
        }));
    return ownerRRefPtr;
  }
}

} // namespace torch::distributed::rpc

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 0 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `JitFuture`, `UserRRef`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/rpc`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ThreadLocalState.h`
- `fmt/format.h`
- `torch/csrc/autograd/record_function_ops.h`
- `torch/csrc/distributed/autograd/utils.h`
- `torch/csrc/distributed/rpc/message.h`
- `torch/csrc/distributed/rpc/profiler/remote_profiler_manager.h`
- `torch/csrc/distributed/rpc/rpc_agent.h`
- `torch/csrc/distributed/rpc/rref_proto.h`
- `torch/csrc/distributed/rpc/script_call.h`
- `torch/csrc/distributed/rpc/torchscript_functions.h`
- `torch/csrc/distributed/rpc/utils.h`


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
- [`utils.cpp_docs.md`](./utils.cpp_docs.md)
- [`unpickled_python_call.cpp_docs.md`](./unpickled_python_call.cpp_docs.md)
- [`request_callback.h_docs.md`](./request_callback.h_docs.md)
- [`rref_context.cpp_docs.md`](./rref_context.cpp_docs.md)
- [`request_callback_impl.h_docs.md`](./request_callback_impl.h_docs.md)
- [`py_rref.h_docs.md`](./py_rref.h_docs.md)


## Cross-References

- **File Documentation**: `torchscript_functions.cpp_docs.md`
- **Keyword Index**: `torchscript_functions.cpp_kw.md`
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

- This file appears to involve **GPU/parallel computing** capabilities.
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

- **File Documentation**: `torchscript_functions.cpp_docs.md_docs.md`
- **Keyword Index**: `torchscript_functions.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
