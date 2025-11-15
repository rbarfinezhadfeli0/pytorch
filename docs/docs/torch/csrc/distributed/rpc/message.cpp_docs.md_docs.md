# Documentation: `docs/torch/csrc/distributed/rpc/message.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/rpc/message.cpp_docs.md`
- **Size**: 5,524 bytes (5.39 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/rpc/message.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/rpc/message.cpp`
- **Size**: 3,088 bytes (3.02 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/custom_class.h>

namespace torch::distributed::rpc {

Message::Message() = default;

Message::Message(
    std::vector<char>&& payload,
    std::vector<torch::Tensor>&& tensors,
    MessageType type)
    : payload_(std::move(payload)), tensors_(std::move(tensors)), type_(type) {}

Message::Message(
    std::vector<char>&& payload,
    std::vector<torch::Tensor>&& tensors,
    MessageType type,
    int64_t id)
    : payload_(std::move(payload)),
      tensors_(std::move(tensors)),
      type_(type),
      id_(id) {}

std::vector<char>&& Message::movePayload() && {
  return std::move(payload_);
}

std::vector<char>& Message::payload() {
  return payload_;
}

const std::vector<char>& Message::payload() const {
  return payload_;
}

std::vector<torch::Tensor>&& Message::moveTensors() && {
  return std::move(tensors_);
}

std::vector<torch::Tensor>& Message::tensors() {
  return tensors_;
}

const std::vector<torch::Tensor>& Message::tensors() const {
  return tensors_;
}

MessageType Message::type() const {
  return type_;
}

bool Message::isRequest() const {
  return static_cast<int>(MessageTypeFlags::REQUEST_TYPE) &
      static_cast<int>(type_);
}

bool Message::isResponse() const {
  return static_cast<int>(MessageTypeFlags::RESPONSE_TYPE) &
      static_cast<int>(type_);
}

int64_t Message::id() const {
  return id_;
}

void Message::setId(int64_t id) {
  id_ = id;
}

std::vector<c10::weak_intrusive_ptr<c10::StorageImpl>> Message::getStorages()
    const {
  // Sparse tensors do not have storage. Instead, a sparse tensor
  // contains two tensors indices and values, and both contain storage.
  std::vector<c10::weak_intrusive_ptr<c10::StorageImpl>> storages;
  storages.reserve(2 * tensors_.size());
  for (const auto& tensor : tensors_) {
    if (tensor.is_sparse()) {
      storages.emplace_back(tensor._indices().storage().getWeakStorageImpl());
      storages.emplace_back(tensor._values().storage().getWeakStorageImpl());
    } else {
      storages.emplace_back(tensor.storage().getWeakStorageImpl());
    }
  }
  return storages;
}

c10::intrusive_ptr<Message> createExceptionResponse(
    const std::exception& e,
    int64_t id) {
  return createExceptionResponse(e.what(), id);
}

c10::intrusive_ptr<Message> createExceptionResponse(
    const std::string& exceptionStr,
    int64_t id) {
  std::vector<char> payload(exceptionStr.begin(), exceptionStr.end());
  return c10::make_intrusive<Message>(
      std::move(payload),
      std::vector<torch::Tensor>(),
      MessageType::EXCEPTION,
      id);
}

namespace {

// NB: need to call torch::class_ to register Message in the map returned by
// c10::getCustomClassTypeMap(). Otherwise, Message cannot be wrapped within
// an IValue.
// NB: add this line here instead of in rpc/init.cpp because 1) we have C++
// only tests that won't run rpc/init.cpp; 2) Message is not meant to be
// visible from Python.
static const auto message = torch::class_<Message>("rpc", "_Message");

} // namespace

} // namespace torch::distributed::rpc

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 1 function(s).

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

- `torch/csrc/distributed/rpc/message.h`
- `torch/custom_class.h`


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

- **File Documentation**: `message.cpp_docs.md`
- **Keyword Index**: `message.cpp_kw.md`
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

- **File Documentation**: `message.cpp_docs.md_docs.md`
- **Keyword Index**: `message.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
