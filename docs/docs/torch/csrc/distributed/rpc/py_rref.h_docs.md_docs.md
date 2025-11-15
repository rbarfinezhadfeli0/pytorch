# Documentation: `docs/torch/csrc/distributed/rpc/py_rref.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/rpc/py_rref.h_docs.md`
- **Size**: 5,496 bytes (5.37 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/rpc/py_rref.h`

## File Metadata

- **Path**: `torch/csrc/distributed/rpc/py_rref.h`
- **Size**: 2,965 bytes (2.90 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/distributed/rpc/rref_impl.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::distributed::rpc {

// NOLINTNEXTLINE(performance-enum-size)
enum RRefProxyType { RPC_SYNC, RPC_ASYNC, REMOTE };

// Python wrapper of an RRef shared_ptr that supports Python
// pickle and unpickle.
class PYBIND11_EXPORT PyRRef {
 public:
  // The first ctor can only be called while holding GIL. See its implementation
  // for more explanations.
  explicit PyRRef(const py::object& value, const py::object& type_hint);
  explicit PyRRef(c10::intrusive_ptr<RRef> rref);
  PyRRef(const PyRRef&) = default;
  ~PyRRef();

  bool isOwner() const;
  bool confirmedByOwner() const;
  WorkerInfo owner() const;
  std::string ownerName() const;
  py::object toHere(
      const float timeoutSeconds =
          torch::distributed::rpc::kUnsetRpcTimeout) const;
  py::object localValue() const;
  std::string str() const;
  py::tuple pickle() const;
  static PyRRef unpickle(const py::tuple& t);
  c10::IValue toIValue() const;
  // Future that is associated with the creation of this RRef on the remote end.
  // This is only used to get the future corresponding to the rref for profiling
  // use cases.
  c10::intrusive_ptr<JitFuture> getFuture() const;
  // Keeps track of the future responsible for profiling owner creation
  // acknowledgement
  c10::intrusive_ptr<JitFuture> getProfilingFuture() const;
  // Sets the future responsible for profiling owner creation acknowledgement.
  // This future is set from python to be a future that returns when profiling
  // callbacks have been run.
  void setProfilingFuture(c10::intrusive_ptr<JitFuture> profilingFuture);

  // create a proxy on this RRef, which can be used to launch RPC on the owner
  // of this RRef to run functions on the object referenced by this RRef.
  py::object createRRefProxy(
      const RRefProxyType& mode,
      float timeoutSeconds = rpc::kUnsetRpcTimeout) const;

  // get the type of the data object referenced by this RRef. Timeout argument
  // is only used in the first invocation of this function as an argument to the
  // RPC to the owner node of the RRef.
  py::object getRRefType(
      float timeout = rpc::kUnsetRpcTimeout,
      bool blocking = true);

  // Run the backward pass with the RRef as the root.
  void backward(int64_t autogradContextId, bool retainGraph);

  // Helper static function to run backward on a given rref.
  static void backward(
      int64_t autogradContextId,
      bool retainGraph,
      const c10::intrusive_ptr<RRef>& rref);

  // Specialization of backward if the rref is an OwnerRRef.
  static void backwardOwnerRRef(
      int64_t autogradContextId,
      bool retainGraph,
      IValue value);

 private:
  c10::intrusive_ptr<RRef> rref_;
  std::optional<c10::intrusive_ptr<JitFuture>> profilingFuture_;
  std::optional<py::object> type_;
};

} // namespace torch::distributed::rpc

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 18 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `PYBIND11_EXPORT`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/rpc`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/distributed/rpc/rref_impl.h`
- `torch/csrc/python_headers.h`
- `torch/csrc/utils/pybind.h`


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


## Cross-References

- **File Documentation**: `py_rref.h_docs.md`
- **Keyword Index**: `py_rref.h_kw.md`
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

- **File Documentation**: `py_rref.h_docs.md_docs.md`
- **Keyword Index**: `py_rref.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
