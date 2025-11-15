# Documentation: `docs/torch/csrc/distributed/rpc/rref_proto.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/rpc/rref_proto.h_docs.md`
- **Size**: 8,152 bytes (7.96 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/rpc/rref_proto.h`

## File Metadata

- **Path**: `torch/csrc/distributed/rpc/rref_proto.h`
- **Size**: 5,357 bytes (5.23 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/distributed/rpc/types.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <vector>

namespace torch::distributed::rpc {

// Temporary solution of RRef operations.
// TODO: Remove all these messages and use rpc + registered functions instead.
class TORCH_API RRefMessageBase : public RpcCommandBase {
 public:
  RRefMessageBase(const RRefId& rrefId, MessageType type)
      : rrefId_(rrefId), type_(type) {}

  const RRefId& rrefId();

 protected:
  // NOLINTNEXTLINE(cppcoreguidelines*)
  const RRefId rrefId_;
  // NOLINTNEXTLINE(cppcoreguidelines*)
  const MessageType type_;
};

class TORCH_API ForkMessageBase : public RRefMessageBase {
 public:
  ForkMessageBase(const RRefId& rrefId, const ForkId& forkId, MessageType type)
      : RRefMessageBase(rrefId, type), forkId_(forkId) {}

  const ForkId& forkId();

  c10::intrusive_ptr<Message> toMessageImpl() && override;
  static std::pair<RRefId, ForkId> fromMessage(
      const Message& message,
      MessageType type);

 protected:
  // NOLINTNEXTLINE(cppcoreguidelines*)
  const ForkId forkId_;
};

// UserRRef uses this message to fetch the remote RRef value from the owner.
class TORCH_API ScriptRRefFetchCall final : public RRefMessageBase {
 public:
  ScriptRRefFetchCall(worker_id_t fromWorkerId, const RRefId& rrefId)
      : RRefMessageBase(rrefId, MessageType::SCRIPT_RREF_FETCH_CALL),
        fromWorkerId_(fromWorkerId) {}

  inline worker_id_t fromWorkerId() const {
    return fromWorkerId_;
  }

  c10::intrusive_ptr<Message> toMessageImpl() && override;
  static std::unique_ptr<ScriptRRefFetchCall> fromMessage(
      const Message& message);

 private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const worker_id_t fromWorkerId_;
};

class TORCH_API PythonRRefFetchCall final : public RRefMessageBase {
 public:
  PythonRRefFetchCall(worker_id_t fromWorkerId, const RRefId& rrefId)
      : RRefMessageBase(rrefId, MessageType::PYTHON_RREF_FETCH_CALL),
        fromWorkerId_(fromWorkerId) {}

  c10::intrusive_ptr<Message> toMessageImpl() && override;
  static std::unique_ptr<PythonRRefFetchCall> fromMessage(
      const Message& message);

 private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const worker_id_t fromWorkerId_;
};

// OwnerRRef uses this message to send the RRef value to a remote UserRRef
class TORCH_API RRefFetchRet : public RpcCommandBase {
 public:
  RRefFetchRet(std::vector<at::IValue> values, MessageType type)
      : values_(std::move(values)), type_(type) {}

  const std::vector<at::IValue>& values();
  c10::intrusive_ptr<Message> toMessageImpl() && override;

 private:
  std::vector<at::IValue> values_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const MessageType type_;
};

class TORCH_API ScriptRRefFetchRet final : public RRefFetchRet {
 public:
  explicit ScriptRRefFetchRet(std::vector<at::IValue> values)
      : RRefFetchRet(std::move(values), MessageType::SCRIPT_RREF_FETCH_RET) {}

  static std::unique_ptr<ScriptRRefFetchRet> fromMessage(
      const Message& message);
};

class TORCH_API PythonRRefFetchRet final : public RRefFetchRet {
 public:
  explicit PythonRRefFetchRet(std::vector<at::IValue> values)
      : RRefFetchRet(std::move(values), MessageType::PYTHON_RREF_FETCH_RET) {}

  static std::unique_ptr<PythonRRefFetchRet> fromMessage(
      const Message& message);
};

// UserRRef (regardless it's the creator or not) uses this message to notify
// OwnerRRef on delete.
class TORCH_API RRefUserDelete final : public ForkMessageBase {
 public:
  RRefUserDelete(const RRefId& rrefId, const ForkId& forkId)
      : ForkMessageBase(rrefId, forkId, MessageType::RREF_USER_DELETE) {}

  static std::unique_ptr<RRefUserDelete> fromMessage(const Message& message);
};

class TORCH_API RemoteRet final : public ForkMessageBase {
 public:
  RemoteRet(const RRefId& rrefId, const ForkId& forkId)
      : ForkMessageBase(rrefId, forkId, MessageType::REMOTE_RET) {}

  static std::unique_ptr<RemoteRet> fromMessage(const Message& message);
};

// A child RRef uses this message to notify its parent that the child has been
// confirmed by the owner.
class TORCH_API RRefChildAccept final : public RpcCommandBase {
 public:
  explicit RRefChildAccept(const ForkId& forkId) : forkId_(forkId) {}

  const ForkId& forkId() const;

  c10::intrusive_ptr<Message> toMessageImpl() && override;
  static std::unique_ptr<RRefChildAccept> fromMessage(const Message& message);

 private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const ForkId forkId_;
};

// A child RRef uses this message to send a fork request to the owner.
class TORCH_API RRefForkRequest final : public ForkMessageBase {
 public:
  RRefForkRequest(const RRefId& rrefId, const ForkId& forkId)
      : ForkMessageBase(rrefId, forkId, MessageType::RREF_FORK_REQUEST) {}

  static std::unique_ptr<RRefForkRequest> fromMessage(const Message& message);
};

class TORCH_API RRefAck final : public RpcCommandBase {
 public:
  RRefAck() = default;

  c10::intrusive_ptr<Message> toMessageImpl() && override;
  static std::unique_ptr<RRefAck> fromMessage(const Message& message);
};

} // namespace torch::distributed::rpc

```



## High-Level Overview


This C++ file contains approximately 12 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `TORCH_API`, `TORCH_API`, `TORCH_API`, `TORCH_API`, `TORCH_API`, `TORCH_API`, `TORCH_API`, `TORCH_API`, `TORCH_API`, `TORCH_API`, `TORCH_API`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/rpc`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/distributed/rpc/message.h`
- `torch/csrc/distributed/rpc/rpc_command_base.h`
- `torch/csrc/distributed/rpc/types.h`
- `torch/csrc/jit/runtime/operator.h`
- `vector`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

- **File Documentation**: `rref_proto.h_docs.md`
- **Keyword Index**: `rref_proto.h_kw.md`
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

- **File Documentation**: `rref_proto.h_docs.md_docs.md`
- **Keyword Index**: `rref_proto.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
