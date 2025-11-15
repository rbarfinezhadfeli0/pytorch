# Documentation: `torch/csrc/distributed/rpc/script_call.h`

## File Metadata

- **Path**: `torch/csrc/distributed/rpc/script_call.h`
- **Size**: 2,486 bytes (2.43 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <optional>
#include <vector>

namespace torch::distributed::rpc {

using torch::jit::Operator;

// A ScriptCall instance represents an invocation of a builtin operator for a
// TorchScript function. If it is a builtin operator, it
// contains a shared ptr to the `Operator` and a list of arguments.
// If it is a TorchScript function, it contains a non empty qualifiedName string
// to the TorchScript function schema name and a list of arguments.
class TORCH_API ScriptCall : public RpcCommandBase {
 public:
  // Constructor for builtin operator call.
  ScriptCall(std::shared_ptr<Operator> op, std::vector<at::IValue>&& stack);
  // Constructor for TorchScript function call.
  ScriptCall(
      const c10::QualifiedName& qualifiedName,
      std::vector<at::IValue>&& stack,
      const bool isAsyncExecution = false);

  bool hasOp() const;
  std::shared_ptr<Operator> op() const;
  bool hasQualifiedName() const;
  const c10::QualifiedName& qualifiedName() const;
  // return the argument stack of this builtin operator
  const std::vector<at::IValue>& stack() const;
  std::vector<at::IValue>& stackRef();
  inline bool isAsyncExecution() const {
    return isAsyncExecution_;
  }

  c10::intrusive_ptr<Message> toMessageImpl() && override;
  static std::unique_ptr<ScriptCall> fromMessage(const Message& message);

  ~ScriptCall() override = default;

 protected:
  virtual void toIValues(std::vector<at::IValue>& ivalues) const;
  static std::unique_ptr<ScriptCall> fromIValues(
      std::vector<at::IValue>& ivalues);

 private:
  // Given an operator symbol and a string schema, return the matched operator.
  static std::shared_ptr<Operator> matchOperator(const std::string& str_schema);

  static const std::string BUILTIN_OP_NAMESPACE_;
  static const std::string ATEN_PREFIX_;

  // This field has value if this ScriptCall represents invocation of a builtin
  // operator.
  std::optional<std::shared_ptr<Operator>> op_;
  // This field has non empty string if this ScriptCall represents invocation of
  // an annotated torchscript function defined by users.
  std::optional<const c10::QualifiedName> qualifiedName_;
  std::vector<at::IValue> stack_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const bool isAsyncExecution_;
};

} // namespace torch::distributed::rpc

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/rpc`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/distributed/rpc/message.h`
- `torch/csrc/distributed/rpc/rpc_command_base.h`
- `torch/csrc/jit/runtime/operator.h`
- `optional`
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

- **File Documentation**: `script_call.h_docs.md`
- **Keyword Index**: `script_call.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
