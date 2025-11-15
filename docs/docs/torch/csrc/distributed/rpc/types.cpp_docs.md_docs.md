# Documentation: `docs/torch/csrc/distributed/rpc/types.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/rpc/types.cpp_docs.md`
- **Size**: 6,287 bytes (6.14 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/rpc/types.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/rpc/types.cpp`
- **Size**: 3,746 bytes (3.66 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/distributed/rpc/types.h>

namespace torch::distributed::rpc {

// Thread local flag to enforce rref JIT pickling to be allowed only
// in the scope of an rpc call. For other scopes like when model is
// saved by calling torch.save(), rref is not allowed to be pickled directly.
static thread_local bool allowJitRRefPickle = false;

bool getAllowJitRRefPickle() {
  return allowJitRRefPickle;
}

void enableJitRRefPickle() {
  allowJitRRefPickle = true;
}

void disableJitRRefPickle() {
  allowJitRRefPickle = false;
}

static_assert(
    // NOLINTNEXTLINE(misc-redundant-expression)
    std::numeric_limits<local_id_t>::max() <=
        std::numeric_limits<int64_t>::max(),
    "The max value of local_id_t must be within the range of int64_t");
static_assert(
    std::numeric_limits<worker_id_t>::max() <=
        std::numeric_limits<int64_t>::max(),
    "The max value of worker_id_t must be within the range of int64_t");

///////////////////////////  JitRRefPickleGuard   ///////////////////////////
JitRRefPickleGuard::JitRRefPickleGuard() {
  allowJitRRefPickle = true;
}
JitRRefPickleGuard::~JitRRefPickleGuard() {
  allowJitRRefPickle = false;
}

///////////////////////////  GloballyUniqueId   ///////////////////////////

GloballyUniqueId::GloballyUniqueId(worker_id_t createdOn, local_id_t localId)
    : createdOn_(createdOn), localId_(localId) {}

bool GloballyUniqueId::operator==(const GloballyUniqueId& other) const {
  return createdOn_ == other.createdOn_ && localId_ == other.localId_;
}

bool GloballyUniqueId::operator!=(const GloballyUniqueId& other) const {
  return createdOn_ != other.createdOn_ || localId_ != other.localId_;
}

at::IValue GloballyUniqueId::toIValue() const {
  return c10::ivalue::Tuple::create(
      {static_cast<int64_t>(createdOn_), static_cast<int64_t>(localId_)});
}

GloballyUniqueId GloballyUniqueId::fromIValue(const at::IValue& ivalue) {
  TORCH_INTERNAL_ASSERT(
      ivalue.isTuple(),
      "GloballyUniqueId::fromIValue expected ivalue to be a tuple.");
  const auto& ivalues = ivalue.toTupleRef().elements();
  TORCH_CHECK(
      ivalues.size() == 2,
      "Constructing GloballyUniqueId from ivalue "
      "expects a GenericList of two elements, but got ",
      ivalues.size());

  TORCH_CHECK(
      ivalues[0].toInt() <= std::numeric_limits<worker_id_t>::max(),
      "GloballyUniqueId createdOn out of range, got ",
      ivalues[0].toInt());
  worker_id_t createdOn = static_cast<worker_id_t>(ivalues[0].toInt());

  TORCH_CHECK(
      ivalues[1].toInt() <= std::numeric_limits<local_id_t>::max(),
      "GloballyUniqueId localId out of range, got ",
      ivalues[1].toInt());
  local_id_t localId = ivalues[1].toInt();

  return GloballyUniqueId(createdOn, localId);
}

std::ostream& operator<<(std::ostream& os, GloballyUniqueId const& globalId) {
  return os << "GloballyUniqueId(created_on=" << globalId.createdOn_
            << ", local_id=" << globalId.localId_ << ")";
}

///////////////////////////  SerializedPyObj   ///////////////////////////

std::vector<at::IValue> SerializedPyObj::toIValues() && {
  std::vector<at::IValue> ivalues;
  ivalues.reserve(tensors_.size() + 1);
  for (auto& tensor : tensors_) {
    ivalues.emplace_back(std::move(tensor));
  }
  ivalues.emplace_back(std::move(payload_));
  return ivalues;
}

SerializedPyObj SerializedPyObj::fromIValues(std::vector<at::IValue> values) {
  std::string payload = values.back().toStringRef();
  values.pop_back();
  std::vector<at::Tensor> tensors;
  tensors.reserve(values.size());
  for (auto& value : values) {
    tensors.emplace_back(std::move(value).toTensor());
  }
  return SerializedPyObj(std::move(payload), std::move(tensors));
}

} // namespace torch::distributed::rpc

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

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

- **File Documentation**: `types.cpp_docs.md`
- **Keyword Index**: `types.cpp_kw.md`
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
- [`utils.cpp_docs.md_docs.md`](./utils.cpp_docs.md_docs.md)
- [`rref_impl.h_kw.md_docs.md`](./rref_impl.h_kw.md_docs.md)
- [`rpc_agent.cpp_kw.md_docs.md`](./rpc_agent.cpp_kw.md_docs.md)
- [`request_callback_impl.cpp_kw.md_docs.md`](./request_callback_impl.cpp_kw.md_docs.md)
- [`script_call.cpp_docs.md_docs.md`](./script_call.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `types.cpp_docs.md_docs.md`
- **Keyword Index**: `types.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
