# Documentation: `torch/csrc/jit/runtime/interpreter.h`

## File Metadata

- **Path**: `torch/csrc/jit/runtime/interpreter.h`
- **Size**: 4,991 bytes (4.87 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <memory>
#include <optional>
#include <vector>

#include <ATen/ThreadLocalState.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/frontend/source_range.h>

TORCH_DECLARE_bool(torch_jit_disable_warning_prints);
TORCH_DECLARE_bool(torch_jit_enable_rethrow_caught_exception);

namespace at {
class Tensor;
TORCH_API void launch(std::function<void()> func);
} // namespace at
namespace c10 {
struct IValue;
struct OperatorName;
} // namespace c10

namespace torch::jit {

// The interpreter run Graphs with Tensor inputs and Tensor outputs
// a separate component in the autograd handles unwrapping and wrapping
// variable objects for use in the interpreter.
namespace interpreter {
struct CodeImpl;
}

struct Node;
struct GraphExecutor;
struct InterpreterStateImpl;
struct Graph;
struct Node;
struct Instruction;
using Stack = std::vector<c10::IValue>;
using c10::ivalue::Future;
using TaskLauncher = std::function<void(std::function<void()>)>;

bool TORCH_API in_torchscript_runtime();

struct TORCH_API Code {
  Code() = default;
  explicit Code(interpreter::CodeImpl* pImpl);
  // remaining_bailout_depth is irrelevant in a `Code` object unless the `Code`
  // is directly created by `GraphExecutor` in which case it's likely to contain
  // `prim::BailOut`s to control the maximum depth of bailout chains
  explicit Code(
      const std::shared_ptr<Graph>& graph,
      std::string function_name,
      size_t remaining_bailout_depth = 0);

  const std::vector<GraphExecutor*>& grad_executors();
  const std::vector<GraphExecutor*>& diff_graph_op_executors();

  explicit operator bool() const {
    return pImpl != nullptr;
  }
  size_t num_inputs() const;
  size_t num_outputs() const;
  size_t num_bailouts() const;
  const std::vector<c10::IValue>& constant_table() const;
  const std::vector<c10::TypePtr>& type_table() const;
  const std::vector<Instruction>& instructions() const;
  const std::unordered_map<std::string, size_t>& op_to_num_specified_args()
      const;
  const std::vector<Node*>& instructions_source() const;
  void request_bailout(size_t index);
  size_t register_size() const;
  std::shared_ptr<Graph> graph() const;

 private:
  std::shared_ptr<interpreter::CodeImpl> pImpl;
  friend struct InterpreterStateImpl;
  friend std::ostream& operator<<(std::ostream& out, const Code& code);
};

struct TORCH_API MobileCode : Code {
  explicit MobileCode(
      const std::shared_ptr<Graph>& graph,
      std::string function_name,
      bool emit_default_input_instructions = true,
      bool support_default_args_before_out = true,
      bool emit_promoted_ops = true,
      size_t remaining_bailout_depth = 0);
};

struct InterpreterState {
  TORCH_API InterpreterState(
      const Code& code,
      TaskLauncher taskLauncher = at::launch);
  TORCH_API void run(Stack& stack);
  TORCH_API c10::intrusive_ptr<Future> runAsync(Stack& stack);
  c10::intrusive_ptr<Future> getFuture();

 private:
  InterpreterState(c10::intrusive_ptr<c10::intrusive_ptr_target> pImpl);
  // Ideally we should use c10::intrusive_ptr<InterpreterStateImpl> for pImpl;
  // but intrusive_ptr requires full definition of InterpreterStateImpl,
  // which we need to hide in the header.
  c10::intrusive_ptr<c10::intrusive_ptr_target> pImpl;
  friend struct InterpreterStateImpl;
};

// Created by wait()
struct Suspend : public std::exception {
  const char* what() const noexcept override {
    return "Suspend";
  }

  explicit Suspend(c10::intrusive_ptr<Future> future_)
      : future(std::move(future_)) {}

  c10::intrusive_ptr<Future> future;
};

// InterpreterContinuation propagates dist_autograd_context_id
// through (and only through) the forward pass manually, other
// thread local settings are propagated with ThreadLocalState
struct InterpreterContinuation {
  InterpreterContinuation(
      InterpreterState state_,
      Stack stack_,
      int64_t dist_autograd_context_id = 0,
      std::optional<at::ThreadLocalState> tls_state = std::nullopt)
      : state(std::move(state_)),
        stack(std::move(stack_)),
        tls_state_(std::move(tls_state))
#ifdef USE_DISTRIBUTED
        ,
        dist_autograd_context_id_(dist_autograd_context_id)
#endif
  {
  }

  void operator()();

 private:
  InterpreterState state;
  Stack stack;
  std::optional<at::ThreadLocalState> tls_state_ = std::nullopt;
#ifdef USE_DISTRIBUTED
  int64_t dist_autograd_context_id_;
#endif
};

// what is the tensors type, including state from the current execution context
// that modifies how the tensor behaves. For instance if no_grad is enabled
// this will cause the TensorType to have requires_grad=False.
TORCH_API at::TensorTypePtr tensorTypeInCurrentExecutionContext(
    const at::Tensor& t);

// current (TLS) TorchScript interpreter callstack
TORCH_API std::vector<StackEntry> currentCallstack();
TORCH_API std::vector<std::string> currentModuleHierarchy();

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 17 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`, `torch`, `interpreter`, `at`

**Classes/Structs**: `Tensor`, `IValue`, `OperatorName`, `CodeImpl`, `Node`, `GraphExecutor`, `InterpreterStateImpl`, `Graph`, `Node`, `Instruction`, `TORCH_API`, `InterpreterStateImpl`, `TORCH_API`, `InterpreterState`, `InterpreterStateImpl`, `Suspend`, `InterpreterContinuation`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `memory`
- `optional`
- `vector`
- `ATen/ThreadLocalState.h`
- `ATen/core/ivalue.h`
- `ATen/core/jit_type.h`
- `torch/csrc/Export.h`
- `torch/csrc/jit/frontend/source_range.h`


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

Files in the same folder (`torch/csrc/jit/runtime`):

- [`decomposition_registry.h_docs.md`](./decomposition_registry.h_docs.md)
- [`register_distributed_ops.cpp_docs.md`](./register_distributed_ops.cpp_docs.md)
- [`instruction.h_docs.md`](./instruction.h_docs.md)
- [`argument_spec.cpp_docs.md`](./argument_spec.cpp_docs.md)
- [`instruction.cpp_docs.md`](./instruction.cpp_docs.md)
- [`symbolic_script.h_docs.md`](./symbolic_script.h_docs.md)
- [`register_prim_ops_fulljit.cpp_docs.md`](./register_prim_ops_fulljit.cpp_docs.md)
- [`symbolic_shape_registry_util.cpp_docs.md`](./symbolic_shape_registry_util.cpp_docs.md)
- [`logging.h_docs.md`](./logging.h_docs.md)


## Cross-References

- **File Documentation**: `interpreter.h_docs.md`
- **Keyword Index**: `interpreter.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
