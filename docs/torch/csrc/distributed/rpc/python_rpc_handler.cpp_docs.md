# Documentation: `torch/csrc/distributed/rpc/python_rpc_handler.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/rpc/python_rpc_handler.cpp`
- **Size**: 7,449 bytes (7.27 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/python_compat.h>

namespace torch::distributed::rpc {

namespace {

constexpr auto kInternalModule = "torch.distributed.rpc.internal";

// A macro that grabs the GIL, profiling the acquisition time. The average GIL
// acquisition time will be recorded in RpcAgent's getMetrics().
#define PROFILE_GIL_SCOPED_ACQUIRE                                       \
  std::chrono::time_point<std::chrono::high_resolution_clock> startTime; \
  auto shouldProfileGIL =                                                \
      RpcAgent::getCurrentRpcAgent()->isGILProfilingEnabled();           \
  if (shouldProfileGIL) {                                                \
    startTime = std::chrono::high_resolution_clock::now();               \
  }                                                                      \
  pybind11::gil_scoped_acquire ag;                                       \
  if (shouldProfileGIL) {                                                \
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(    \
        std::chrono::high_resolution_clock::now() - startTime);          \
    RpcAgent::getCurrentRpcAgent()->addGilWaitTime(dur);                 \
  }

// PythonTypeResolver that inherits from Script::Resolver to
// support resolving types together with ScriptTypeParser.
struct PythonTypeResolver : public jit::Resolver {
  std::shared_ptr<jit::SugaredValue> resolveValue(
      const std::string& /* unused */,
      torch::jit::GraphFunction& /* unused */,
      const jit::SourceRange& /* unused */) override {
    TORCH_INTERNAL_ASSERT(
        false, "RPC Type resolver does not need to resolve value");
  }

  TypePtr resolveType(
      const std::string& name,
      const jit::SourceRange& /* unused */) override {
    if (name == "PyObject") {
      return PyObjectType::get();
    }
    return PythonRpcHandler::getInstance().jitCompilationUnit()->get_type(name);
  }
};

py::object getFunction(const py::object& module, const char* name) {
  py::object fn = module.attr(name);
  TORCH_CHECK(
      py::isinstance<py::function>(fn),
      "attribute ",
      name,
      " is not a function");
  return fn;
}

void cleanupPyObj(py::object& obj) {
  obj.dec_ref();
  // explicitly setting PyObject* to nullptr to prevent py::object's dtor to
  // decref on the PyObject again.
  // See Note [Destructing py::object] in python_ivalue.h
  obj.ptr() = nullptr;
}

} // namespace

void PythonRpcHandler::init() {
  std::lock_guard<std::mutex> guard(init_lock_);
  if (!initialized_) {
    PROFILE_GIL_SCOPED_ACQUIRE;
    py::object rpcInternal = py::module::import(kInternalModule);
    py::object rpcApi = py::module::import("torch.distributed.rpc.api");
    py::object rrefProxy =
        py::module::import("torch.distributed.rpc.rref_proxy");

    pyRunFunction_ = getFunction(rpcInternal, "_run_function");
    pySerialize_ = getFunction(rpcInternal, "serialize");
    pyDeserialize_ = getFunction(rpcInternal, "deserialize");
    pyHandleException_ = getFunction(rpcInternal, "_handle_exception");

    rrefTypeFunctions_.onOwner_ = getFunction(rpcApi, "_rref_typeof_on_owner");
    rrefTypeFunctions_.onUser_ = getFunction(rpcApi, "_rref_typeof_on_user");

    rrefProxyFunctions_.rpcSync_ = getFunction(rpcApi, "rpc_sync");
    rrefProxyFunctions_.rpcAsync_ = getFunction(rpcApi, "rpc_async");
    rrefProxyFunctions_.remote_ = getFunction(rpcApi, "remote");
    rrefProxyFunctions_.rrefProxyCtor_ = getFunction(rrefProxy, "RRefProxy");

    jitCompilationUnit_ = torch::jit::get_python_cu();
    typeParser_ = std::make_shared<jit::ScriptTypeParser>(
        std::make_shared<PythonTypeResolver>());
    initialized_ = true;
  }
}

PythonRpcHandler::PythonRpcHandler() : initialized_(false) {}

void PythonRpcHandler::cleanup() {
  std::lock_guard<std::mutex> guard(init_lock_);
  PROFILE_GIL_SCOPED_ACQUIRE;
  cleanupPyObj(pyRunFunction_);
  cleanupPyObj(pySerialize_);
  cleanupPyObj(pyDeserialize_);
  cleanupPyObj(pyHandleException_);

  cleanupPyObj(rrefProxyFunctions_.rpcSync_);
  cleanupPyObj(rrefProxyFunctions_.rpcAsync_);
  cleanupPyObj(rrefProxyFunctions_.remote_);
  cleanupPyObj(rrefProxyFunctions_.rrefProxyCtor_);

  jitCompilationUnit_ = nullptr;
  typeParser_ = nullptr;
  initialized_ = false;
}

PythonRpcHandler& PythonRpcHandler::getInstance() {
  // A thread could hold GIL when calling PythonRpcHandler::getInstance(),
  // meantime another thread could have been doing static data
  // initialization by calling `new PythonRpcHandler()`, inside of which GIL is
  // also required. Static data initialization is thread-safe, so the thread
  // holding the GIL will wait for the other thread to finish static data
  // initializing before going forward. Because the initialization can't
  // proceed without GIL, there is a deadlock. We ask the calling thread to
  // release GIL to avoid this situation.
  TORCH_INTERNAL_ASSERT(!PyGILState_Check());
  // Leaky singleton to avoid module destructor race.
  static PythonRpcHandler* handler = new PythonRpcHandler();
  handler->init();
  return *handler;
}

std::shared_ptr<torch::jit::CompilationUnit> PythonRpcHandler::
    jitCompilationUnit() {
  return jitCompilationUnit_;
}

py::object PythonRpcHandler::runPythonUdf(const py::object& pythonUdf) {
  PROFILE_GIL_SCOPED_ACQUIRE;
  // Throw a descriptive error message if pyRunFunction_ is already cleaned up.
  TORCH_INTERNAL_ASSERT(
      !pyRunFunction_.is_none(),
      "Cannot run python UDF since pyRunFunction_ is None. Check if python RPC "
      "handler is already cleaned up.");
  return pyRunFunction_(pythonUdf);
}

SerializedPyObj PythonRpcHandler::serialize(const py::object& obj) {
  PROFILE_GIL_SCOPED_ACQUIRE;
  py::tuple t = pySerialize_(obj);
  return SerializedPyObj(
      t[0].cast<std::string>(), t[1].cast<std::vector<torch::Tensor>>());
}

py::object PythonRpcHandler::deserialize(const SerializedPyObj& serializedObj) {
  PROFILE_GIL_SCOPED_ACQUIRE;
  // NB: pyDeserialize_ can return an AttributeError if the deserialize() Python
  // function fails. Functions consuming the result needs to handle such error
  // properly.
  return pyDeserialize_(
      py::bytes(serializedObj.payload_), serializedObj.tensors_);
}

void PythonRpcHandler::handleException(const py::object& obj) {
  PROFILE_GIL_SCOPED_ACQUIRE;
  pyHandleException_(obj);
}

void PythonRpcHandler::handleExceptionGILHeld(const py::object& obj) {
  TORCH_CHECK(PyGILState_Check(), "GIL should be held");
  pyHandleException_(obj);
}

bool PythonRpcHandler::isRemoteException(const py::object& obj) {
  PROFILE_GIL_SCOPED_ACQUIRE;
  auto type = py::type::handle_of(obj);
  auto moduleName = type.attr("__module__").cast<std::string>();
  auto qualName = type.attr("__qualname__").cast<std::string>();
  return moduleName == kInternalModule && qualName == "RemoteException";
}

TypePtr PythonRpcHandler::parseTypeFromStr(const std::string& type_str) {
  return typeParser_->parseType(type_str);
}

const PythonRpcHandler::RRefProxyFunctions& PythonRpcHandler::
    getRRefProxyFunctions() const {
  return rrefProxyFunctions_;
}

const PythonRpcHandler::RRefTypeFunctions& PythonRpcHandler::
    getRRefTypeFunctions() const {
  return rrefTypeFunctions_;
}

} // namespace torch::distributed::rpc

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `void`

**Classes/Structs**: `PythonTypeResolver`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/rpc`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/distributed/rpc/python_rpc_handler.h`
- `torch/csrc/distributed/rpc/rpc_agent.h`
- `torch/csrc/jit/python/pybind_utils.h`
- `torch/csrc/utils/python_compat.h`


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
- [`tensorpipe_agent.h_docs.md`](./tensorpipe_agent.h_docs.md)
- [`torchscript_functions.cpp_docs.md`](./torchscript_functions.cpp_docs.md)
- [`utils.cpp_docs.md`](./utils.cpp_docs.md)
- [`unpickled_python_call.cpp_docs.md`](./unpickled_python_call.cpp_docs.md)
- [`request_callback.h_docs.md`](./request_callback.h_docs.md)
- [`rref_context.cpp_docs.md`](./rref_context.cpp_docs.md)
- [`request_callback_impl.h_docs.md`](./request_callback_impl.h_docs.md)
- [`py_rref.h_docs.md`](./py_rref.h_docs.md)


## Cross-References

- **File Documentation**: `python_rpc_handler.cpp_docs.md`
- **Keyword Index**: `python_rpc_handler.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
