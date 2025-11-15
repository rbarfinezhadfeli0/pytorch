# Documentation: `torch/csrc/distributed/rpc/testing/faulty_tensorpipe_agent.h`

## File Metadata

- **Path**: `torch/csrc/distributed/rpc/testing/faulty_tensorpipe_agent.h`
- **Size**: 3,786 bytes (3.70 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```c
#pragma once

#ifdef USE_TENSORPIPE

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/tensorpipe_agent.h>

namespace torch::distributed::rpc {

struct TORCH_API FaultyTensorPipeRpcBackendOptions
    : public TensorPipeRpcBackendOptions {
  FaultyTensorPipeRpcBackendOptions(
      int num_worker_threads,
      float rpc_timeout,
      std::string init_method,
      std::vector<std::string> messages_to_fail,
      std::unordered_map<std::string, float> messages_to_delay,
      int num_fail_sends = 0)
      : TensorPipeRpcBackendOptions(
            num_worker_threads,
            std::optional<std::vector<std::string>>(),
            std::optional<std::vector<std::string>>(),
            rpc_timeout,
            std::move(init_method)),
        messagesToFail(std::move(messages_to_fail)),
        messagesToDelay(std::move(messages_to_delay)),
        numFailSends(num_fail_sends) {
    TORCH_CHECK(numFailSends >= 0, "numFailSends should be non-negative");
  }

  std::vector<std::string> messagesToFail;
  std::unordered_map<std::string, float> messagesToDelay;
  int numFailSends;
};

class TORCH_API FaultyTensorPipeAgent : public TensorPipeAgent {
 public:
  FaultyTensorPipeAgent(
      const c10::intrusive_ptr<::c10d::Store>& store,
      std::string selfName,
      worker_id_t selfId,
      int worldSize,
      FaultyTensorPipeRpcBackendOptions opts,
      std::unordered_map<std::string, DeviceMap> reverseDeviceMaps,
      std::vector<c10::Device> devices,
      std::unique_ptr<RequestCallback> callback);

  // Faulty send function for this class.
  c10::intrusive_ptr<JitFuture> send(
      const WorkerInfo& to,
      c10::intrusive_ptr<Message> message,
      const float rpcTimeoutSeconds = torch::distributed::rpc::kUnsetRpcTimeout,
      const DeviceMap& deviceMap = {}) override;

  // Add delay to writes
  void pipeWrite(
      const std::shared_ptr<tensorpipe::Pipe>& pipe,
      const c10::intrusive_ptr<Message>& rpcMessage,
      std::vector<c10::Device>&& devices,
      std::vector<c10::Stream> streams,
      std::function<void(const tensorpipe::Error&)> fn) noexcept override;

 protected:
  // This function checks the messageTypesToFail_ to determine whether to use
  // the faulty send or not.
  bool shouldFailMessage(MessageType type) const;

 private:
  // This function parses the list of strings passed in by the python tests and
  // resolves the Message Types that must use the faulty send.
  std::vector<MessageType> parseMessagesToFailInput(
      const std::vector<std::string>& messagesToFail) const;

  // Returns amount of time in seconds to delay sending of the given message
  // type.
  float getDelayForMessage(MessageType type) const;

  // Parse message types that we should inject arbitrary delays for.
  std::unordered_map<MessageType, float, std::hash<int>> parseMessagesToDelay(
      const std::unordered_map<std::string, float>& messageTypesToDelay) const;

  // Number of sends to intentionally fail before allowing one to succeed.
  const int numFailSends_;

  // Vector of the MessageTypes that we must use the faulty send for. This is
  // parsed based on a list of strings passed in by the python tests.
  const std::vector<MessageType> messageTypesToFail_;

  // Mapping of message types to amount we should delay send for in the ::send()
  // function.
  std::unordered_map<MessageType, float, std::hash<int>> messageTypesToDelay_;

  // Map to track the number of sends we've failed for each RPC.
  std::unordered_map<std::string, int> failMessageCountMap_;

  // Mutex to guard failMessageCountMap_
  std::mutex failMapMutex_;

  MessageType messageStringToType(const std::string& messageString) const;
};

} // namespace torch::distributed::rpc

#endif // USE_TENSORPIPE

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `TORCH_API`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/rpc/testing`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/distributed/rpc/message.h`
- `torch/csrc/distributed/rpc/tensorpipe_agent.h`


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
python torch/csrc/distributed/rpc/testing/faulty_tensorpipe_agent.h
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/csrc/distributed/rpc/testing`):

- [`testing.h_docs.md`](./testing.h_docs.md)
- [`faulty_tensorpipe_agent.cpp_docs.md`](./faulty_tensorpipe_agent.cpp_docs.md)
- [`init.cpp_docs.md`](./init.cpp_docs.md)


## Cross-References

- **File Documentation**: `faulty_tensorpipe_agent.h_docs.md`
- **Keyword Index**: `faulty_tensorpipe_agent.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
