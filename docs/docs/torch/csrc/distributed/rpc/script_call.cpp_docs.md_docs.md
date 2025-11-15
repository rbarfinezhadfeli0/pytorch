# Documentation: `docs/torch/csrc/distributed/rpc/script_call.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/rpc/script_call.cpp_docs.md`
- **Size**: 7,721 bytes (7.54 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/rpc/script_call.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/rpc/script_call.cpp`
- **Size**: 5,137 bytes (5.02 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/jit/serialization/pickle.h>

namespace torch::distributed::rpc {

const std::string ScriptCall::BUILTIN_OP_NAMESPACE_("torch.ops.aten.");
const std::string ScriptCall::ATEN_PREFIX_("aten::");

ScriptCall::ScriptCall(
    std::shared_ptr<Operator> op,
    // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
    std::vector<at::IValue>&& stack)
    : op_(std::move(op)), stack_(stack), isAsyncExecution_(false) {}

ScriptCall::ScriptCall(
    const c10::QualifiedName& qualifiedName,
    // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
    std::vector<at::IValue>&& stack,
    const bool isAsyncExecution)
    : qualifiedName_(qualifiedName),
      stack_(stack),
      isAsyncExecution_(isAsyncExecution) {}

bool ScriptCall::hasOp() const {
  return op_.has_value();
}

std::shared_ptr<Operator> ScriptCall::op() const {
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  return op_.value();
}

bool ScriptCall::hasQualifiedName() const {
  return qualifiedName_.has_value();
}

const c10::QualifiedName& ScriptCall::qualifiedName() const {
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  return qualifiedName_.value();
}

const std::vector<at::IValue>& ScriptCall::stack() const {
  return stack_;
}

std::vector<at::IValue>& ScriptCall::stackRef() {
  return stack_;
}

void ScriptCall::toIValues(std::vector<at::IValue>& ivalues) const {
  for (auto& value : stack_) {
    ivalues.push_back(value);
  }

  if (op_.has_value()) {
    TORCH_CHECK(
        !hasQualifiedName(),
        "It is builtin operator call, qualifiedName_ should not be set.");
    // TODO: replace this with a real overload_name when FunctionSchema supports
    // that.
    ivalues.emplace_back(toString((*op_)->schema()));
    // insert qualified name
    auto opName = (*op_)->schema().name();
    TORCH_CHECK(
        opName.find("::") == opName.rfind("::") &&
            opName.rfind(ATEN_PREFIX_) == 0,
        "Unexpected operator name ",
        opName);
    // aten::add -> torch.ops.aten.add
    opName.replace(0, ATEN_PREFIX_.length(), BUILTIN_OP_NAMESPACE_);
    ivalues.emplace_back(std::move(opName));
  } else if (hasQualifiedName()) {
    ivalues.emplace_back(isAsyncExecution());
    TORCH_CHECK(
        !hasOp(),
        "It is TorchScript function call, operator should not be set.");
    ivalues.emplace_back(qualifiedName().qualifiedName());
  } else {
    TORCH_INTERNAL_ASSERT(
        false,
        "Either builtin operator or TorchScript function name should be set.");
  }
}

std::unique_ptr<ScriptCall> ScriptCall::fromIValues(
    std::vector<at::IValue>& ivalues) {
  TORCH_INTERNAL_ASSERT(
      ivalues.size() > 1,
      "At least 2 IValues are required to build a ScriptCall.");

  // Last element in the vector is always qualifiedName for both
  // builtin operator and TorchScript function
  // If the qualifiedName is not a builtin operator name, then treat it
  // as TorchScript function name
  std::string qualifiedName = ivalues.back().toStringRef();

  if (qualifiedName.rfind(BUILTIN_OP_NAMESPACE_) == 0) {
    ivalues.pop_back();
    const std::string& str_schema = ivalues.back().toStringRef();
    auto op = matchOperator(str_schema);

    ivalues.pop_back();
    // remove str_schema from ivalues
    return std::make_unique<ScriptCall>(op, std::move(ivalues));
  } else {
    ivalues.pop_back();
    bool isAsyncExecution = ivalues.back().toBool();
    ivalues.pop_back();
    return std::make_unique<ScriptCall>(
        c10::QualifiedName(qualifiedName),
        std::move(ivalues),
        isAsyncExecution);
  }
}

c10::intrusive_ptr<Message> ScriptCall::toMessageImpl() && {
  std::vector<IValue> ivalues;
  toIValues(ivalues);

  std::vector<torch::Tensor> tensor_table;
  auto payload = jit::pickle(
      c10::ivalue::Tuple::create(std::move(ivalues)), &tensor_table);

  return c10::make_intrusive<Message>(
      std::move(payload), std::move(tensor_table), MessageType::SCRIPT_CALL);
}

std::unique_ptr<ScriptCall> ScriptCall::fromMessage(const Message& message) {
  auto payload = message.payload().data();
  auto payload_size = message.payload().size();
  auto value = jit::unpickle(
      payload,
      payload_size,
      *RpcAgent::getCurrentRpcAgent()->getTypeResolver(),
      message.tensors());

  auto values = value.toTupleRef().elements().vec();
  return fromIValues(values);
}

std::shared_ptr<Operator> ScriptCall::matchOperator(
    const std::string& str_schema) {
  // TODO: This is a temporary solution. We should pass enough information to
  // allow deterministically matched to one operator.

  // extract symbol from the schema
  auto schema = torch::jit::parseSchema(str_schema);
  auto symbol = at::Symbol::fromQualString(schema.name());

  for (auto op : torch::jit::getAllOperatorsFor(symbol)) {
    if (toString(op->schema()) == str_schema) {
      return op;
    }
  }

  TORCH_CHECK(false, "Cannot find matching operator for schema ", str_schema);
}

} // namespace torch::distributed::rpc

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

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
- `torch/csrc/distributed/rpc/script_call.h`
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

- **File Documentation**: `script_call.cpp_docs.md`
- **Keyword Index**: `script_call.cpp_kw.md`
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


## Cross-References

- **File Documentation**: `script_call.cpp_docs.md_docs.md`
- **Keyword Index**: `script_call.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
