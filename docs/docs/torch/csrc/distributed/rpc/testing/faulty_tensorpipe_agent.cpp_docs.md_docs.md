# Documentation: `docs/torch/csrc/distributed/rpc/testing/faulty_tensorpipe_agent.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/rpc/testing/faulty_tensorpipe_agent.cpp_docs.md`
- **Size**: 8,471 bytes (8.27 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/rpc/testing/faulty_tensorpipe_agent.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/rpc/testing/faulty_tensorpipe_agent.cpp`
- **Size**: 6,201 bytes (6.06 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```cpp
#ifdef USE_TENSORPIPE

#include <torch/csrc/distributed/rpc/testing/faulty_tensorpipe_agent.h>
#include <torch/csrc/distributed/rpc/utils.h>

namespace torch::distributed::rpc {

static std::string fromVecToString(const std::vector<char>& vec) {
  return std::string(vec.begin(), vec.end());
}

FaultyTensorPipeAgent::FaultyTensorPipeAgent(
    const c10::intrusive_ptr<::c10d::Store>& store,
    std::string selfName,
    worker_id_t selfId,
    int worldSize,
    FaultyTensorPipeRpcBackendOptions opts,
    std::unordered_map<std::string, DeviceMap> reverseDeviceMaps,
    std::vector<c10::Device> devices,
    std::unique_ptr<RequestCallback> callback)
    : TensorPipeAgent(
          store,
          std::move(selfName),
          selfId,
          worldSize,
          std::move(opts),
          std::move(reverseDeviceMaps),
          std::move(devices),
          std::move(callback)),
      // NOLINTNEXTLINE(bugprone-use-after-move)
      numFailSends_(opts.numFailSends),
      // NOLINTNEXTLINE(bugprone-use-after-move)
      messageTypesToFail_(parseMessagesToFailInput(opts.messagesToFail)),
      // NOLINTNEXTLINE(bugprone-use-after-move)
      messageTypesToDelay_(parseMessagesToDelay(opts.messagesToDelay)) {}

std::vector<MessageType> FaultyTensorPipeAgent::parseMessagesToFailInput(
    const std::vector<std::string>& messagesToFail) const {
  // Since we can only pass strings corresponding to the Message Types from the
  // python tests, we must parse the list of strings and resolve the actual
  // types. We will then check this list of types in the send function to
  // determine whether we should fail or not.
  std::vector<MessageType> messageTypesToFail;
  messageTypesToFail.reserve(messagesToFail.size());
  for (const auto& msgString : messagesToFail) {
    messageTypesToFail.push_back(messageStringToType(msgString));
  }
  return messageTypesToFail;
}

std::unordered_map<MessageType, float, std::hash<int>> FaultyTensorPipeAgent::
    parseMessagesToDelay(const std::unordered_map<std::string, float>&
                             messageTypesToDelay) const {
  std::unordered_map<MessageType, float, std::hash<int>> delayMessages;
  for (const auto& messagePair : messageTypesToDelay) {
    float delay = messagePair.second;
    TORCH_CHECK(
        delay >= 0,
        "Delays passed to FaultyTensorPipeAgent must be non-negative.")
    delayMessages.insert({messageStringToType(messagePair.first), delay});
  }
  return delayMessages;
}

c10::intrusive_ptr<JitFuture> FaultyTensorPipeAgent::send(
    const WorkerInfo& to,
    c10::intrusive_ptr<Message> message,
    const float rpcTimeoutSeconds,
    const DeviceMap& /* unused */) {
  // We only fail control messages that have been specified by the test case.
  // For all other messages, we just send them without any failures.
  if (!shouldFailMessage(message->type())) {
    return TensorPipeAgent::send(to, std::move(message), rpcTimeoutSeconds);
  }

  // This send function checks the failMessageCountMap_ to check whether
  // we must fail the next send. If the send must be failed, we set an error
  // on the returned future immediately and increment the counter in the map,
  // otherwise we just call the TensorPipeAgent send.
  const auto key = fromVecToString(message->payload());
  std::unique_lock<std::mutex> lock(failMapMutex_);
  auto it = failMessageCountMap_.find(key);
  if (it == failMessageCountMap_.end()) {
    failMessageCountMap_[key] = 0;
  }
  if (failMessageCountMap_[key] < numFailSends_) {
    failMessageCountMap_[key]++;
    lock.unlock();
    auto jitFuture = c10::make_intrusive<JitFuture>(at::AnyClassType::get());
    jitFuture->setError(std::make_exception_ptr(std::runtime_error(makeRPCError(
        c10::str("Send attempt failed intentionally for ", key),
        RPCErrorType::INTENTIONAL_FAILURE))));
    return jitFuture;
  } else {
    lock.unlock();
    return TensorPipeAgent::send(to, std::move(message), rpcTimeoutSeconds);
  }
}

void FaultyTensorPipeAgent::pipeWrite(
    const std::shared_ptr<tensorpipe::Pipe>& pipe,
    const c10::intrusive_ptr<Message>& rpcMessage,
    std::vector<c10::Device>&& devices,
    std::vector<c10::Stream> streams,
    std::function<void(const tensorpipe::Error&)> fn) noexcept {
  float msgDelay = getDelayForMessage(rpcMessage->type());
  if (msgDelay != 0) {
    // Sleep for the specified delay for the message.
    std::this_thread::sleep_for(std::chrono::milliseconds(
        static_cast<int>(msgDelay * kSecToMsConversion)));
  }
  TensorPipeAgent::pipeWrite(pipe, rpcMessage, std::move(devices), streams, fn);
}

bool FaultyTensorPipeAgent::shouldFailMessage(MessageType type) const {
  // Return true if the input message type is in the messageTypesToFail_ list
  return (
      std::find(messageTypesToFail_.begin(), messageTypesToFail_.end(), type) !=
      messageTypesToFail_.end());
}

float FaultyTensorPipeAgent::getDelayForMessage(MessageType type) const {
  const auto& it = messageTypesToDelay_.find(type);
  return it == messageTypesToDelay_.end() ? 0 : it->second;
}

MessageType FaultyTensorPipeAgent::messageStringToType(
    const std::string& messageString) const {
  // Lazily constructed map that returns string to message type mapping
  static std::unordered_map<std::string, MessageType> msgMap = {
      {"RREF_FORK_REQUEST", MessageType::RREF_FORK_REQUEST},
      {"RREF_CHILD_ACCEPT", MessageType::RREF_CHILD_ACCEPT},
      {"RREF_USER_DELETE", MessageType::RREF_USER_DELETE},
      {"CLEANUP_AUTOGRAD_CONTEXT_REQ",
       MessageType::CLEANUP_AUTOGRAD_CONTEXT_REQ},
      {"PYTHON_REMOTE_CALL", MessageType::PYTHON_REMOTE_CALL},
      {"SCRIPT_REMOTE_CALL", MessageType::SCRIPT_REMOTE_CALL},
      {"PYTHON_CALL", MessageType::PYTHON_CALL},
      {"SCRIPT_CALL", MessageType::SCRIPT_CALL},
      {"PYTHON_RREF_FETCH_CALL", MessageType::PYTHON_RREF_FETCH_CALL},
      {"SCRIPT_RREF_FETCH_CALL", MessageType::SCRIPT_RREF_FETCH_CALL}};
  const auto& it = msgMap.find(messageString);
  TORCH_CHECK(
      it != msgMap.end(),
      "No mapping to rpc::MessageType exists for ",
      messageString);
  return it->second;
}

} // namespace torch::distributed::rpc

#endif // USE_TENSORPIPE

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/rpc/testing`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/distributed/rpc/testing/faulty_tensorpipe_agent.h`
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

This is a test file. Run it with:

```bash
python torch/csrc/distributed/rpc/testing/faulty_tensorpipe_agent.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/csrc/distributed/rpc/testing`):

- [`testing.h_docs.md`](./testing.h_docs.md)
- [`faulty_tensorpipe_agent.h_docs.md`](./faulty_tensorpipe_agent.h_docs.md)
- [`init.cpp_docs.md`](./init.cpp_docs.md)


## Cross-References

- **File Documentation**: `faulty_tensorpipe_agent.cpp_docs.md`
- **Keyword Index**: `faulty_tensorpipe_agent.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/distributed/rpc/testing`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/distributed/rpc/testing`, which is part of the **core PyTorch library**.



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

This is a test file. Run it with:

```bash
python docs/torch/csrc/distributed/rpc/testing/faulty_tensorpipe_agent.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/csrc/distributed/rpc/testing`):

- [`faulty_tensorpipe_agent.h_kw.md_docs.md`](./faulty_tensorpipe_agent.h_kw.md_docs.md)
- [`init.cpp_kw.md_docs.md`](./init.cpp_kw.md_docs.md)
- [`faulty_tensorpipe_agent.cpp_kw.md_docs.md`](./faulty_tensorpipe_agent.cpp_kw.md_docs.md)
- [`init.cpp_docs.md_docs.md`](./init.cpp_docs.md_docs.md)
- [`testing.h_kw.md_docs.md`](./testing.h_kw.md_docs.md)
- [`testing.h_docs.md_docs.md`](./testing.h_docs.md_docs.md)
- [`faulty_tensorpipe_agent.h_docs.md_docs.md`](./faulty_tensorpipe_agent.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `faulty_tensorpipe_agent.cpp_docs.md_docs.md`
- **Keyword Index**: `faulty_tensorpipe_agent.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
