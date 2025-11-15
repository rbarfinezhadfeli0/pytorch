# Documentation: `torch/csrc/distributed/rpc/script_remote_call.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/rpc/script_remote_call.cpp`
- **Size**: 2,617 bytes (2.56 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/script_remote_call.h>

#include <torch/csrc/jit/serialization/pickle.h>

namespace torch::distributed::rpc {

ScriptRemoteCall::ScriptRemoteCall(
    std::shared_ptr<Operator> op,
    std::vector<at::IValue>&& stack,
    const RRefId& retRRefId,
    const ForkId& retForkId)
    : ScriptCall(std::move(op), std::move(stack)),
      retRRefId_(retRRefId),
      retForkId_(retForkId) {}

ScriptRemoteCall::ScriptRemoteCall(
    const c10::QualifiedName& qualifiedName,
    std::vector<at::IValue>&& stack,
    const RRefId& retRRefId,
    const ForkId& retForkId,
    const bool isAsyncExecution)
    : ScriptCall(qualifiedName, std::move(stack), isAsyncExecution),
      retRRefId_(retRRefId),
      retForkId_(retForkId) {}

std::unique_ptr<ScriptRemoteCall> ScriptRemoteCall::fromIValues(
    std::vector<at::IValue>& ivalues) {
  // remove the last element from values and convert it back to an RRef
  auto retForkId = RRefId::fromIValue(ivalues.back());
  ivalues.pop_back();
  auto retRRefId = ForkId::fromIValue(ivalues.back());
  ivalues.pop_back();

  auto scriptCallPtr = ScriptCall::fromIValues(ivalues);

  if (scriptCallPtr->hasOp()) {
    return std::make_unique<ScriptRemoteCall>(
        scriptCallPtr->op(), std::move(ivalues), retRRefId, retForkId);
  } else {
    return std::make_unique<ScriptRemoteCall>(
        scriptCallPtr->qualifiedName(),
        std::move(ivalues),
        retRRefId,
        retForkId,
        scriptCallPtr->isAsyncExecution());
  }
}

c10::intrusive_ptr<Message> ScriptRemoteCall::toMessageImpl() && {
  std::vector<IValue> ivalues;
  ScriptCall::toIValues(ivalues);
  ivalues.emplace_back(retRRefId_.toIValue());
  ivalues.emplace_back(retForkId_.toIValue());

  std::vector<torch::Tensor> tensor_table;
  auto payload = jit::pickle(
      c10::ivalue::Tuple::create(std::move(ivalues)), &tensor_table);

  return c10::make_intrusive<Message>(
      std::move(payload),
      std::move(tensor_table),
      MessageType::SCRIPT_REMOTE_CALL);
}

std::unique_ptr<ScriptRemoteCall> ScriptRemoteCall::fromMessage(
    const Message& message) {
  auto payload = message.payload().data();
  auto payload_size = message.payload().size();

  auto value = jit::unpickle(
      payload,
      payload_size,
      *RpcAgent::getCurrentRpcAgent()->getTypeResolver(),
      message.tensors());
  auto values = value.toTupleRef().elements().vec();
  TORCH_CHECK(!values.empty(), "Malformed message: empty values unpickled");
  return fromIValues(values);
}

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

- `torch/csrc/distributed/rpc/rpc_agent.h`
- `torch/csrc/distributed/rpc/script_remote_call.h`
- `torch/csrc/jit/serialization/pickle.h`


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
- [`py_rref.h_docs.md`](./py_rref.h_docs.md)


## Cross-References

- **File Documentation**: `script_remote_call.cpp_docs.md`
- **Keyword Index**: `script_remote_call.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
