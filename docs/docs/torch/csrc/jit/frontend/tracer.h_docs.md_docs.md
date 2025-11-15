# Documentation: `docs/torch/csrc/jit/frontend/tracer.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/frontend/tracer.h_docs.md`
- **Size**: 15,795 bytes (15.42 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/frontend/tracer.h`

## File Metadata

- **Path**: `torch/csrc/jit/frontend/tracer.h`
- **Size**: 12,801 bytes (12.50 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/Dimname.h>
#include <ATen/core/class_type.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/stack.h>
#include <ATen/core/symbol.h>
#include <c10/util/Exception.h>
#include <torch/csrc/Export.h>

#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/utils/variadic.h>

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace torch::jit {
struct Node;
struct Value;
struct Graph;
struct Module;

namespace tracer {

using ::c10::ivalue::Shared;

using ::c10::IValue;
using ::c10::ivalue::Future;

using ::c10::ArrayRef;
using ::c10::TupleType;
using ::c10::TupleTypePtr;
using ::c10::ivalue::ConstantString;

using torch::autograd::Variable;
using variable_list = std::vector<Variable>;

TORCH_API std::atomic<bool>& getTracerStateWarnMode();

struct TORCH_API TracingState
    : public std::enable_shared_from_this<TracingState> {
  TracingState();
  ~TracingState();

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::shared_ptr<Graph> graph;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  bool warn = getTracerStateWarnMode();
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  bool strict = true;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  bool force_outplace = false;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::function<std::string(const Variable& var)> lookup_var_name_fn =
      [](const Variable& var) { return ""; };

  void enterFrame() {
    env_stack.emplace_back();
  }

  void leaveFrame() {
    env_stack.pop_back();
  }

  void setValue(const IValue& v, Value* value);
  void delValue(const IValue& var);
  Value* getValue(const IValue& var);
  Value* getOutput(const IValue& var, size_t i);
  bool hasValue(const IValue& var) const;

  Node* createNode(c10::Symbol op_name, size_t num_outputs);
  void insertNode(Node* node);

 private:
  using WeakIValue = at::WeakIValue;

  struct WeakIValueHasher {
    size_t operator()(const WeakIValue& t) const {
      return t.hash();
    }
  };

  struct WeakIValueEq {
    bool operator()(const WeakIValue& t1, const WeakIValue& t2) const {
      return t1.isSameIdentity(t2);
    }
  };

  using Frame =
      std::unordered_map<WeakIValue, Value*, WeakIValueHasher, WeakIValueEq>;
  std::vector<Frame> env_stack;
};

// This is meant to be used as a thread local place, where we can store extra
// info that gets lost when we call into ATen from Python bindings. One example
// for when this happens is when we get an IntArrayRef argument with e.g. sizes
// for view. When tracing, those might be tensors, which let us encode extra
// data dependencies, but once they get to the ATen call where we actually have
// the tracing logic, they get converted into a raw IntArrayRef, and we loose
// all information. To prevent this, we temporarily stash it in here.
struct ArgumentStash {
  struct IntArrayRefTrace : std::vector<Value*> {
    IntArrayRefTrace(size_t size) : std::vector<Value*>(size, nullptr) {}
  };

  static bool empty() {
    return stash.intlists.empty();
  }

  TORCH_API static void stashIntArrayRefElem(
      const std::string& arg_name,
      size_t size,
      size_t idx,
      const Variable& var);

  static bool hasIntArrayRef(const std::string& arg_name) {
    return stash.intlists.count(arg_name) > 0;
  }

  static IntArrayRefTrace popIntArrayRef(const std::string& arg_name) {
    auto info = std::move(stash.intlists.at(arg_name));
    stash.intlists.erase(arg_name);
    return info;
  }

  // Value stashing: Use these methods to stash arguments which correspond
  // to regular Value*'s in the graph. i.e. they don't require special
  // handling like in the case of IntArrayRefs
  TORCH_API static void stashValue(
      const std::string& arg_name,
      size_t idx,
      const Variable& var,
      const c10::TypePtr& type = nullptr);

  static bool hasValue(const std::string& arg_name) {
    return stash.values.count(arg_name) > 0;
  }

  static Value* popValue(const std::string& arg_name) {
    auto info = stash.values.at(arg_name);
    stash.values.erase(arg_name);
    return info;
  }

 private:
  static thread_local ArgumentStash stash;
  std::unordered_map<std::string, IntArrayRefTrace> intlists;
  std::unordered_map<std::string, Value*> values;
};

// Retrieve or set the current tracing state. Returns a nullptr if tracing is
// disabled.
TORCH_API const std::shared_ptr<TracingState>& getTracingState();
TORCH_API void setTracingState(std::shared_ptr<TracingState> state);

inline bool isTracing() {
  return static_cast<bool>(getTracingState());
}

using warn_fn_type = void (*)(const std::string& msg);
TORCH_API extern const char* WARN_PYTHON_DATAFLOW;
TORCH_API extern const char* WARN_CONSTRUCTOR;
TORCH_API extern const char* WARN_RESIZE;
TORCH_API extern const char* STRICT_TRACER_MSG;
TORCH_API void _do_warn(const char* _reason, const char* _kind);
inline void warn(const char* _reason, const char* _kind = nullptr) {
  if (const auto& state = getTracingState()) {
    if (!state->warn)
      return;
    _do_warn(_reason, _kind);
  }
}
TORCH_API void setWarn(warn_fn_type fn);

struct TORCH_API NoWarn {
  NoWarn() : state(getTracingState()) {
    if (state) {
      prev = state->warn;
      state->warn = false;
    }
  }
  ~NoWarn() {
    if (state) {
      state->warn = prev;
    }
  }
  std::shared_ptr<TracingState> state;
  bool prev{false};
};

struct WithNestedTracingFrame {
  WithNestedTracingFrame() {
    getTracingState()->enterFrame();
  }

  ~WithNestedTracingFrame() {
    getTracingState()->leaveFrame();
  }
};
TORCH_API void recordSourceLocation(Node* n);
TORCH_API void setRecordSourceLocation(void (*v)(Node*));

TORCH_API std::vector<StackEntry> pythonCallstack();
TORCH_API void setPythonCallstack(std::vector<StackEntry> (*v)());

// Having finished adding a new 'node' to the graph IR 'setValueTrace'
// associates this node with an output variable, so that further operations
// involving this variable know which node in the IR to reference.
TORCH_API void setValueTrace(const IValue& v, Value* value);

TORCH_API void delValueTrace(const IValue& var);

TORCH_API std::function<void()> pauseTracing();

TORCH_API Value* getValueTrace(const IValue& var);

TORCH_API std::pair<std::shared_ptr<TracingState>, Stack> trace(
    Stack inputs,
    const std::function<Stack(Stack)>& traced_fn,
    std::function<std::string(const Variable&)> var_name_lookup_fn,
    bool strict = true,
    bool force_outplace = false,
    Module* self = nullptr,
    const std::vector<std::string>& argument_names = {});

TORCH_API void abandon();

// NB: those serve both as an intermediate steps in addInputs below,
// as well as the overloads that terminate template recursion
TORCH_API void addInputs(Node* n, const char* name, int64_t value);
TORCH_API void addInputs(Node* n, const char* name, const c10::SymInt& value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    std::optional<int64_t> value);
TORCH_API void addInputs(Node* n, const char* name, bool value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const std::optional<bool>& value);
TORCH_API void addInputs(Node* n, const char* name, double value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const std::optional<double>& value);
TORCH_API void addInputs(Node* n, const char* name, const at::Scalar& value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const std::optional<at::Scalar>& value);
TORCH_API void addInputs(Node* n, const char* name, const at::Tensor& value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const std::optional<at::Tensor>& value);
TORCH_API void addInputs(Node* n, const char* name, ArrayRef<int64_t> value);
TORCH_API void addInputs(Node* n, const char* name, c10::SymIntArrayRef value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    std::optional<c10::SymInt> value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const std::optional<ArrayRef<int64_t>>& value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const at::OptionalIntArrayRef& opt_value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const at::OptionalSymIntArrayRef& opt_value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    ArrayRef<at::Tensor> value,
    bool allow_undefined = false);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const std::vector<at::Tensor>& value,
    bool allow_undefined = false);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    at::ITensorListRef value,
    bool allow_undefined = false);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const List<std::optional<at::Tensor>>& value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    ArrayRef<c10::intrusive_ptr<c10::ivalue::Object>> value,
    const c10::ClassTypePtr& class_type);
TORCH_API void addInputs(Node* n, const char* name, ArrayRef<double> value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const std::optional<ArrayRef<double>>& value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const std::string_view value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const std::optional<std::string_view>& value);
TORCH_API void addInputs(Node* n, const char* name, at::Device value);
TORCH_API void addInputs(Node* n, const char* name, c10::Stream stream);
TORCH_API void addInputs(Node* n, const char* name, at::Layout value);
TORCH_API void addInputs(Node* n, const char* name, at::ScalarType value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const std::optional<at::ScalarType>& value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const std::optional<at::Device>& value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const std::optional<at::Layout>& value);
TORCH_API void addInputs(Node* n, const char* name, at::MemoryFormat value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    std::optional<at::DimnameList> value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const std::optional<at::MemoryFormat>& value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const std::optional<at::Generator>& value);

inline void addInputs(
    Node* n,
    const char* name,
    const std::vector<bool>& value) {
  TORCH_CHECK(false, "Tracing a list of bool type is currently not supported!");
}

template <typename T>
void addInputs(Node* n, const char* name, ArrayRef<T> value) {
  TORCH_CHECK(
      false, "Tracing a list of arbitrary type is currently not supported!");
}
template <typename K, typename V>
void addInputs(
    Node* n,
    const char* name,
    const std::unordered_map<K, V>& value) {
  TORCH_CHECK(
      false, "Tracing a dict of arbitrary types is currently not supported!");
}

template <size_t N>
void addInputs(Node* n, const char* name, std::array<bool, N> value) {
  throw std::runtime_error(
      "Found an unsupported argument type in the JIT tracer. File a bug report.");
}

TORCH_API void addInputs(
    Node* n,
    const char* name,
    const c10::intrusive_ptr<c10::ivalue::Object>& obj);

TORCH_API void ensureUniqueIfOutOfPlaced(
    const char* name,
    const at::Tensor& tensor);
TORCH_API void ensureUniqueIfOutOfPlaced(
    const char* name,
    const std::optional<at::Tensor>& tensor);

template <
    typename T,
    typename = std::enable_if_t<
        (!std::is_convertible_v<std::decay_t<T>, at::TensorList> &&
         !std::is_convertible_v<std::decay_t<T>, c10::List<at::Tensor>> &&
         !std::is_convertible_v<std::decay_t<T>, at::Tensor> &&
         !std::is_convertible_v<
             std::decay_t<T>,
             c10::intrusive_ptr<c10::ivalue::Object>>)>>
void addOutput(Node* node, T&& /*unused*/) {
  TORCH_CHECK(
      false,
      "Found an unsupported argument type ",
      c10::demangle_type<T>(),
      " in the JIT tracer. File a bug report.");
}
TORCH_API void addOutput(Node* node, const at::Tensor& tensor);
TORCH_API void setOutput(Value* value, const at::Tensor& output);
TORCH_API void addOutput(Node* node, const std::vector<at::Tensor>& list);
TORCH_API void addOutput(Node* node, const c10::List<at::Tensor>& list);
TORCH_API void addOutput(
    Node* node,
    const c10::intrusive_ptr<c10::ivalue::Object>& output);

TORCH_API autograd::Variable getSizeOf(
    const autograd::Variable& var,
    int64_t dim);

TORCH_API autograd::Variable getNumelOf(const autograd::Variable& var);

} // namespace tracer
} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 77 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `tracer`, `torch`

**Classes/Structs**: `Node`, `Value`, `Graph`, `Module`, `TORCH_API`, `WeakIValueHasher`, `WeakIValueEq`, `ArgumentStash`, `IntArrayRefTrace`, `TORCH_API`, `WithNestedTracingFrame`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/frontend`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Dimname.h`
- `ATen/core/class_type.h`
- `ATen/core/jit_type.h`
- `ATen/core/stack.h`
- `ATen/core/symbol.h`
- `c10/util/Exception.h`
- `torch/csrc/Export.h`
- `torch/csrc/jit/frontend/source_range.h`
- `torch/csrc/utils/variadic.h`
- `cstdint`
- `memory`
- `unordered_map`
- `vector`


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

Files in the same folder (`torch/csrc/jit/frontend`):

- [`canonicalize_modified_loop.cpp_docs.md`](./canonicalize_modified_loop.cpp_docs.md)
- [`schema_matching.cpp_docs.md`](./schema_matching.cpp_docs.md)
- [`source_range.h_docs.md`](./source_range.h_docs.md)
- [`exit_transforms.h_docs.md`](./exit_transforms.h_docs.md)
- [`function_schema_parser.h_docs.md`](./function_schema_parser.h_docs.md)
- [`inline_loop_condition.h_docs.md`](./inline_loop_condition.h_docs.md)
- [`mini_environment.h_docs.md`](./mini_environment.h_docs.md)
- [`tree_views.cpp_docs.md`](./tree_views.cpp_docs.md)
- [`function_schema_parser.cpp_docs.md`](./function_schema_parser.cpp_docs.md)
- [`tracer.cpp_docs.md`](./tracer.cpp_docs.md)


## Cross-References

- **File Documentation**: `tracer.h_docs.md`
- **Keyword Index**: `tracer.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/frontend`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/frontend`, which is part of the **core PyTorch library**.



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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/csrc/jit/frontend`):

- [`strtod.h_kw.md_docs.md`](./strtod.h_kw.md_docs.md)
- [`tree_views.cpp_docs.md_docs.md`](./tree_views.cpp_docs.md_docs.md)
- [`function_schema_parser.cpp_docs.md_docs.md`](./function_schema_parser.cpp_docs.md_docs.md)
- [`tree.h_kw.md_docs.md`](./tree.h_kw.md_docs.md)
- [`versioned_symbols.cpp_kw.md_docs.md`](./versioned_symbols.cpp_kw.md_docs.md)
- [`parser.cpp_kw.md_docs.md`](./parser.cpp_kw.md_docs.md)
- [`lexer.h_kw.md_docs.md`](./lexer.h_kw.md_docs.md)
- [`parser.cpp_docs.md_docs.md`](./parser.cpp_docs.md_docs.md)
- [`convert_to_ssa.h_docs.md_docs.md`](./convert_to_ssa.h_docs.md_docs.md)
- [`error_report.cpp_kw.md_docs.md`](./error_report.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `tracer.h_docs.md_docs.md`
- **Keyword Index**: `tracer.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
