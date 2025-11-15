# Documentation: `docs/torch/csrc/distributed/autograd/utils.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/autograd/utils.cpp_docs.md`
- **Size**: 9,463 bytes (9.24 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/autograd/utils.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/autograd/utils.cpp`
- **Size**: 6,823 bytes (6.66 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/ThreadLocalState.h>
#include <c10/util/ThreadLocalDebugInfo.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/functions/recvrpc_backward.h>
#include <torch/csrc/distributed/autograd/functions/sendrpc_backward.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/rpc/profiler/remote_profiler_manager.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/types.h>

namespace torch::distributed::autograd {

using torch::distributed::autograd::AutogradMetadata;
using torch::distributed::autograd::RpcWithAutograd;
using torch::distributed::rpc::JitFuture;
using torch::distributed::rpc::Message;
using torch::distributed::rpc::MessageType;
using torch::distributed::rpc::RpcAgent;
using torch::distributed::rpc::WorkerInfo;

void addSendRpcBackward(
    const ContextPtr& autogradContext,
    const AutogradMetadata& autogradMetadata,
    std::vector<torch::Tensor>& tensors) {
  // Attach autograd information only for tensors requiring grad.
  std::vector<torch::Tensor> tensors_with_grad;
  std::copy_if(
      tensors.begin(),
      tensors.end(),
      std::back_inserter(tensors_with_grad),
      [](const torch::Tensor& t) { return t.requires_grad(); });

  // Attach the appropriate autograd edges.
  auto grad_fn = std::make_shared<SendRpcBackward>();
  grad_fn->set_next_edges(
      torch::autograd::collect_next_edges(tensors_with_grad));

  // Add the appropriate input metadata for the grad_fn.
  for (const auto& tensor : tensors_with_grad) {
    grad_fn->add_input_metadata(tensor);
  }

  // Record the send autograd function in our current context.
  autogradContext->addSendFunction(grad_fn, autogradMetadata.autogradMessageId);
}

ContextPtr addRecvRpcBackward(
    const AutogradMetadata& autogradMetadata,
    std::vector<torch::Tensor>& tensors,
    rpc::worker_id_t fromWorkerId,
    const rpc::DeviceMap& deviceMap) {
  // Initialize autograd context if necessary.
  auto& autogradContainer = DistAutogradContainer::getInstance();
  auto autogradContext =
      autogradContainer.getOrCreateContext(autogradMetadata.autogradContextId);

  if (!tensors.empty() && torch::autograd::compute_requires_grad(tensors)) {
    // Attach the tensors as inputs to the autograd function.
    auto grad_fn = std::make_shared<RecvRpcBackward>(
        autogradMetadata, autogradContext, fromWorkerId, deviceMap);
    for (auto& tensor : tensors) {
      if (tensor.requires_grad()) {
        torch::autograd::set_history(tensor, grad_fn);
      }
    }

    // Now update the autograd context with the necessary information.
    autogradContext->addRecvFunction(
        grad_fn, autogradMetadata.autogradMessageId);
  }

  return autogradContext;
}

static c10::intrusive_ptr<Message> getMessageWithProfiling(
    c10::intrusive_ptr<torch::distributed::rpc::Message> wrappedRpcMessage,
    MessageType msgType,
    torch::autograd::profiler::ProfilerConfig&& profilerConfig) {
  auto& remoteProfilerManager =
      torch::distributed::rpc::RemoteProfilerManager::getInstance();

  auto key = remoteProfilerManager.getCurrentProfilingKey();
  // generate a globally unique Id
  auto globallyUniqueProfilingId = remoteProfilerManager.getNextProfilerId();
  // Save a mapping of ID -> RPC profiling key and unset the current TLS key.
  remoteProfilerManager.saveRPCKey(globallyUniqueProfilingId, key);
  remoteProfilerManager.unsetCurrentKey();
  auto wrappedProfilingMsg = RpcWithProfilingReq(
      msgType,
      std::move(wrappedRpcMessage),
      std::move(profilerConfig),
      globallyUniqueProfilingId);

  return std::move(wrappedProfilingMsg).toMessage();
}

c10::intrusive_ptr<Message> getMessageWithAutograd(
    const rpc::worker_id_t dstId,
    c10::intrusive_ptr<torch::distributed::rpc::Message> wrappedRpcMsg,
    MessageType msgType,
    bool forceGradRecording,
    const rpc::DeviceMap& deviceMap) {
  auto& autogradContainer = DistAutogradContainer::getInstance();

  // If there is no valid context and no tensor requires grads, send original
  // rpc message. otherwise, attach grad info and grad functions and send
  // rpcWithAutograd message.
  auto tensorsRequireGrad =
      torch::autograd::compute_requires_grad(wrappedRpcMsg->tensors());
  if (!autogradContainer.hasValidContext() ||
      (!forceGradRecording && !tensorsRequireGrad)) {
    return wrappedRpcMsg;
  }

  // Retrieve the appropriate context to modify.
  auto autogradContext = autogradContainer.currentContext();

  // Wrap the original rpc with autograd information.
  AutogradMetadata autogradMetadata(
      autogradContext->contextId(), autogradContainer.newAutogradMessageId());
  auto rpcWithAutograd = std::make_unique<RpcWithAutograd>(
      RpcAgent::getCurrentRpcAgent()->getWorkerInfo().id_,
      msgType,
      autogradMetadata,
      std::move(wrappedRpcMsg),
      deviceMap);

  if (tensorsRequireGrad) {
    // Record autograd information for 'send'.
    addSendRpcBackward(
        autogradContext, autogradMetadata, rpcWithAutograd->tensors());
  }
  // Record the workerID
  autogradContext->addKnownWorkerId(dstId);

  return std::move(*rpcWithAutograd).toMessage();
}

c10::intrusive_ptr<JitFuture> sendMessageWithAutograd(
    RpcAgent& agent,
    const WorkerInfo& dst,
    c10::intrusive_ptr<torch::distributed::rpc::Message> wrappedRpcMsg,
    bool forceGradRecording,
    const float rpcTimeoutSeconds,
    bool forceDisableProfiling) {
  auto msg = getMessageWithAutograd(
      dst.id_,
      std::move(wrappedRpcMsg),
      MessageType::FORWARD_AUTOGRAD_REQ,
      forceGradRecording,
      agent.getDeviceMap(dst));

  // If profiler is enabled, wrap this message with profiling metadata that will
  // tell the remote end to process this request with the profiler enabled.
  if (!forceDisableProfiling) {
    switch (torch::profiler::impl::profilerType()) {
      case torch::profiler::impl::ActiveProfilerType::LEGACY: {
        auto profilerConfig = torch::autograd::profiler::getProfilerConfig();
        auto msgWithProfiling = getMessageWithProfiling(
            std::move(msg),
            rpc::MessageType::RUN_WITH_PROFILING_REQ,
            std::move(profilerConfig));
        return agent.send(dst, std::move(msgWithProfiling), rpcTimeoutSeconds);
      }
      case torch::profiler::impl::ActiveProfilerType::KINETO:
        TORCH_WARN_ONCE(
            "Profiling a distributed call with the Kineto profiler will profile "
            "the caller, but not the worker.");
        break;
      default:
        break;
    }
  }

  return agent.send(dst, std::move(msg), rpcTimeoutSeconds);
  ;
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

This file is located in `torch/csrc/distributed/autograd`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ThreadLocalState.h`
- `c10/util/ThreadLocalDebugInfo.h`
- `torch/csrc/autograd/functions/utils.h`
- `torch/csrc/autograd/profiler.h`
- `torch/csrc/distributed/autograd/context/container.h`
- `torch/csrc/distributed/autograd/functions/recvrpc_backward.h`
- `torch/csrc/distributed/autograd/functions/sendrpc_backward.h`
- `torch/csrc/distributed/autograd/utils.h`
- `torch/csrc/distributed/rpc/profiler/remote_profiler_manager.h`
- `torch/csrc/distributed/rpc/rpc_agent.h`
- `torch/csrc/distributed/rpc/types.h`


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

Files in the same folder (`torch/csrc/distributed/autograd`):

- [`python_autograd.h_docs.md`](./python_autograd.h_docs.md)
- [`utils.h_docs.md`](./utils.h_docs.md)
- [`autograd.cpp_docs.md`](./autograd.cpp_docs.md)
- [`init.cpp_docs.md`](./init.cpp_docs.md)
- [`autograd.h_docs.md`](./autograd.h_docs.md)


## Cross-References

- **File Documentation**: `utils.cpp_docs.md`
- **Keyword Index**: `utils.cpp_kw.md`
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

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
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

- [`python_autograd.h_kw.md_docs.md`](./python_autograd.h_kw.md_docs.md)
- [`utils.h_docs.md_docs.md`](./utils.h_docs.md_docs.md)
- [`utils.h_kw.md_docs.md`](./utils.h_kw.md_docs.md)
- [`autograd.h_docs.md_docs.md`](./autograd.h_docs.md_docs.md)
- [`init.cpp_kw.md_docs.md`](./init.cpp_kw.md_docs.md)
- [`init.cpp_docs.md_docs.md`](./init.cpp_docs.md_docs.md)
- [`autograd.h_kw.md_docs.md`](./autograd.h_kw.md_docs.md)
- [`python_autograd.h_docs.md_docs.md`](./python_autograd.h_docs.md_docs.md)
- [`utils.cpp_kw.md_docs.md`](./utils.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `utils.cpp_docs.md_docs.md`
- **Keyword Index**: `utils.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
