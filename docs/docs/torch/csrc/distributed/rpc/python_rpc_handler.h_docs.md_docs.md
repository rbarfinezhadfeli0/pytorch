# Documentation: `docs/torch/csrc/distributed/rpc/python_rpc_handler.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/rpc/python_rpc_handler.h_docs.md`
- **Size**: 7,763 bytes (7.58 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/rpc/python_rpc_handler.h`

## File Metadata

- **Path**: `torch/csrc/distributed/rpc/python_rpc_handler.h`
- **Size**: 4,954 bytes (4.84 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/types.h>
#include <torch/csrc/jit/frontend/script_type_parser.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::distributed::rpc {

// Singleton class provides interface to execute python UDF remote call
// and deserialize the returned results by running python function
// in internal_rpc_utilities.
// The singleton object is constructed at first when RPC agent is
// constructed, where the python function in
// torch/distributed/internal_rpc_utils.py are imported only once.
class PYBIND11_EXPORT PythonRpcHandler {
 public:
  struct RRefProxyFunctions {
    py::object rrefProxyCtor_;
    py::object rpcSync_;
    py::object rpcAsync_;
    py::object remote_;
  };

  struct RRefTypeFunctions {
    py::object onOwner_;
    py::object onUser_;
  };

  static PythonRpcHandler& getInstance();

  // Run a pickled Python UDF and return the result py::object
  py::object runPythonUdf(const py::object& pythonUdf);

  // Serialized a py::object into a string
  SerializedPyObj serialize(const py::object& obj);

  // Deserialize a string into a py::object
  py::object deserialize(const SerializedPyObj& serializedObj);

  // Check if obj is RemoteException, then throw it
  void handleException(const py::object& obj);
  // Alternative if the caller is already holding the GIL.
  void handleExceptionGILHeld(const py::object& obj);
  // Check if obj is an RemoteException instance.
  bool isRemoteException(const py::object& obj);

  // Explicitly clean up py::objects to avoid segment faults when
  // py::objects with CPython are cleaned up later at program exit
  // See similar issues reported https://github.com/pybind/pybind11/issues/1598
  // and https://github.com/pybind/pybind11/issues/1493
  // Our local tests also caught this segment faults if py::objects are cleaned
  // up at program exit. The explanation is: CPython cleans up most critical
  // utilities before cleaning up PythonRpcHandler singleton, so when
  // PythonRpcHandler singleton cleans up py::objects and call dec_ref(), it
  // will crash.
  // The solution is to clean up py::objects earlier when Rpc agent join().
  // Be note that py::objects can not be cleaned up when Rpc agent is destroyed
  // as well, as Rpc agent is global variable and it will have same issue as
  // PythonRpcHandler.
  void cleanup();

  std::shared_ptr<torch::jit::CompilationUnit> jitCompilationUnit();

  // Parse the string to recover the jit_type, this is used for RRef python
  // pickling/unpickling type recovery. The type string inference rule is as
  // follows:
  // 1. first try to parse if this is primitive types.
  //    i.e. TensorType, IntType, PyObjectType, etc.
  // 2. if not primitive type, we query the python_cu to see if it is a
  //    class type or interface type registered in python
  // We use a ScriptTypeParser instance with custom PythonTypeResolver
  // to resolve types according to the above rules.
  TypePtr parseTypeFromStr(const std::string& typeStr);

  // Return a set of Python functions for RRef helpers.
  const RRefProxyFunctions& getRRefProxyFunctions() const;

  // Return a set of Python functions to retrieve the type of the object
  // referenced by a given RRef.
  const RRefTypeFunctions& getRRefTypeFunctions() const;

  PythonRpcHandler(const PythonRpcHandler&) = delete;
  PythonRpcHandler& operator=(const PythonRpcHandler&) = delete;
  PythonRpcHandler(PythonRpcHandler&&) = delete;
  PythonRpcHandler& operator=(PythonRpcHandler&&) = delete;

 private:
  void init();
  PythonRpcHandler();
  ~PythonRpcHandler() = default;

  // Ref to `torch.distributed.rpc.internal._run_function`.
  py::object pyRunFunction_;

  // Ref to `torch.distributed.rpc.internal.serialize`.
  py::object pySerialize_;

  // Ref to `torch.distributed.rpc.internal.deserialize`.
  py::object pyDeserialize_;

  // Ref to 'torch.distributed.rpc.internal._handle_exception'
  py::object pyHandleException_;

  // Python functions for RRef proxy
  RRefProxyFunctions rrefProxyFunctions_;

  // Ref to 'torch.distributed.rpc.api._rref_typeof_on_'
  RRefTypeFunctions rrefTypeFunctions_;

  // Shared ptr to python compilation unit in jit, it is constructed in python
  // side (see _python_cu = torch._C.CompilationUnit() in jit/__init__.py)
  // and imported in C++ (see get_python_cu() in
  // csrc/jit/python/pybind_utils.h). We import the compilation unit here only
  // once for less cost and thread safety.
  std::shared_ptr<torch::jit::CompilationUnit> jitCompilationUnit_;

  // jit type parser to parse type_str back to TypePtr for RRef type
  // recovery when pickling and unpickling RRef
  std::shared_ptr<jit::ScriptTypeParser> typeParser_;

  // Indicates whether or not we have properly initialized the handler.
  bool initialized_;

  // Lock to protect initialization.
  std::mutex init_lock_;
};

} // namespace torch::distributed::rpc

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `provides`, `PYBIND11_EXPORT`, `RRefProxyFunctions`, `RRefTypeFunctions`, `type`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/rpc`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/distributed/rpc/message.h`
- `torch/csrc/distributed/rpc/types.h`
- `torch/csrc/jit/frontend/script_type_parser.h`
- `torch/csrc/utils/pybind.h`


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

- **File Documentation**: `python_rpc_handler.h_docs.md`
- **Keyword Index**: `python_rpc_handler.h_kw.md`
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

- **File Documentation**: `python_rpc_handler.h_docs.md_docs.md`
- **Keyword Index**: `python_rpc_handler.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
