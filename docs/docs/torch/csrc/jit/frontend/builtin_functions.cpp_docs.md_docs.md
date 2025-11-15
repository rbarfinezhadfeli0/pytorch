# Documentation: `docs/torch/csrc/jit/frontend/builtin_functions.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/frontend/builtin_functions.cpp_docs.md`
- **Size**: 9,085 bytes (8.87 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/frontend/builtin_functions.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/frontend/builtin_functions.cpp`
- **Size**: 6,274 bytes (6.13 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/frontend/builtin_functions.h>

#include <ATen/code_template.h>
#include <caffe2/serialize/versions.h>
#include <torch/csrc/api/include/torch/jit.h>
#include <torch/csrc/jit/frontend/resolver.h>

namespace torch::jit {

static auto scalar_operators_source = at::jit::CodeTemplate(
    R"SCRIPT(
def mul(a : ${Scalar}, b : Tensor) -> Tensor:
  return b * a
def add(a : ${Scalar}, b : Tensor) -> Tensor:
  return b + a
def ne(a : ${Scalar}, b : Tensor) -> Tensor:
  return b != a
def eq(a : ${Scalar}, b : Tensor) -> Tensor:
  return b == a
def sub(a : ${Scalar}, b : Tensor) -> Tensor:
  return torch.neg(b) + a
def div(a : ${Scalar}, b : Tensor) -> Tensor:
  return torch.reciprocal(b) * a
)SCRIPT");

static auto scalar_operators_no_complex_source = at::jit::CodeTemplate(
    R"SCRIPT(
def lt(a : ${Scalar}, b : Tensor) -> Tensor:
  return b > a
def le(a : ${Scalar}, b : Tensor) -> Tensor:
  return b >= a
def gt(a : ${Scalar}, b : Tensor) -> Tensor:
  return b < a
def ge(a : ${Scalar}, b : Tensor) -> Tensor:
  return b <= a
)SCRIPT");

static auto _ntuple_ops = at::jit::CodeTemplate(
    R"SCRIPT(
def _${name}(x: BroadcastingList${Length}[${Scalar}]) -> List[${Scalar}]:
  return x
)SCRIPT");

static auto floordiv = at::jit::CodeTemplate(
    R"SCRIPT(
def floordiv(self : Tensor, other : ${Rhs_Type}) -> Tensor:
  return torch.floor_divide(self, other)
)SCRIPT");

static auto tensor_properties =
    R"SCRIPT(
def ndim(a : Tensor) -> int:
  return a.dim()
def T(a : Tensor) -> Tensor:
  return a.numpy_T()
def H(a : Tensor) -> Tensor:
  return a.matrix_H()
def mT(a : Tensor) -> Tensor:
  return a.mT
def mH(a : Tensor) -> Tensor:
  return a.mH
def shape(a : Tensor) -> List[int]:
  return a.size()
)SCRIPT";

// _assert_int_or_pair is only here for backwards-compatibility with the
// aten::_assert_int_or_pair op which was removed once we were able to compile
// torch.nn.functional.assert_int_or_pair
// list_with_default also needs to be here for BC
static auto aten_ops =
    R"SCRIPT(
def _assert_int_or_pair(vals: List[int], name: str, message: str):
  pass
def list_with_default(out_size: List[int], defaults: List[int]):
  assert len(defaults) > len(out_size)
  return out_size
def _assert(condition : bool, message : str):
  assert condition, message
# existing device operator is registered with input name `a`, which prevents
# torch.device(type="cuda") from working. add shim-layer here
def device(type: str):
  return torch.device(type)
def type(self: Tensor, dtype: int, non_blocking: bool=False, copy: bool=False) -> Tensor:
  return self.to(dtype, non_blocking, copy)
)SCRIPT";

// an additional overload for Tensor variant of _assert
const auto aten_ops_additional =
    R"SCRIPT(
def _assert(condition : Tensor, message : str):
  assert bool(condition), message
def __contains__(self: str, key: str):
    return self.find(key, 0, len(self)) != -1
)SCRIPT";

struct BuiltinFunctionRegistry {
  const std::vector<Function*>& getAllBuiltinFunctionsFor(Symbol name) {
    const static std::vector<Function*> empty;
    // when initializing the builtin function library, we will re-enter
    // getAllBuiltinFunctionsFor since it is called in the compiler to
    // lookup builtins and initializing the builtin functions calls the
    // compiler. To avoid deadlocking, we use a recursive mutex (same thread can
    // re-lock, the mutex without waiting), and report no loaded builtins during
    // init.
    std::lock_guard<std::recursive_mutex> guard(mutex);
    if (state == INITIALIZING) {
      return empty;
    } else if (state == UNINITIALIZED) {
      state = INITIALIZING;
      loadBuiltinFunctions();
      state = INITIALIZED;
    }
    AT_ASSERT(state == INITIALIZED);
    auto it = builtins_by_name_.find(name);
    if (it == builtins_by_name_.end())
      return empty;
    return it->second;
  }

 private:
  void loadSource(const std::string& source, const std::string& the_namespace) {
    std::shared_ptr<CompilationUnit> cu = std::make_shared<CompilationUnit>();
    modules.emplace_back(cu);
    cu->define(std::nullopt, source, nativeResolver(), /*self=*/nullptr);
    for (auto& method : cu->get_functions()) {
      builtins_by_name_[Symbol::fromQualString(
                            the_namespace + "::" + method->name())]
          .push_back(method);
    }
  }

  void loadBuiltinFunctions() {
    for (auto scalar : {"float", "int", "complex"}) {
      at::jit::TemplateEnv env;
      env.s("Scalar", scalar);
      loadSource(scalar_operators_source.format(env), "aten");
    }

    for (auto scalar : {"float", "int"}) {
      at::jit::TemplateEnv env;
      env.s("Scalar", scalar);
      loadSource(scalar_operators_no_complex_source.format(env), "aten");
    }

    using str_pair = std::pair<std::string, std::string>;
    const std::vector<str_pair> name_len = {
        str_pair("single", "1"),
        str_pair("pair", "2"),
        str_pair("triple", "3"),
        str_pair("quadruple", "4"),
    };
    for (const auto scalar : {"float", "int"}) {
      for (const auto& pair : name_len) {
        at::jit::TemplateEnv env;
        env.s("Scalar", scalar);
        env.s("name", pair.first);
        env.s("Length", pair.second);
        loadSource(_ntuple_ops.format(env), "aten");
      }
    }
    for (auto rhs : {"number", "Tensor"}) {
      at::jit::TemplateEnv env;
      env.s("Rhs_Type", rhs);
      loadSource(floordiv.format(env), "aten");
    }

    loadSource(aten_ops, "aten");
    loadSource(aten_ops_additional, "aten");

    // These are under `prim` instead of `aten` since they exist to bind certain
    // tensor property getters to corresponding methods
    loadSource(tensor_properties, "prim");
  }
  enum {
    UNINITIALIZED = 0,
    INITIALIZING = 1,
    // typo in the original code, keeping for compatibility
    INTIIALIZING = 1, // codespell:ignore
    INITIALIZED = 2
  } state = UNINITIALIZED;
  std::recursive_mutex mutex;
  std::vector<std::shared_ptr<CompilationUnit>> modules;
  std::unordered_map<Symbol, std::vector<Function*>> builtins_by_name_;
};

const std::vector<Function*>& getAllBuiltinFunctionsFor(Symbol name) {
  static BuiltinFunctionRegistry registry;
  return registry.getAllBuiltinFunctionsFor(name);
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 31 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `BuiltinFunctionRegistry`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/frontend`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/frontend/builtin_functions.h`
- `ATen/code_template.h`
- `caffe2/serialize/versions.h`
- `torch/csrc/api/include/torch/jit.h`
- `torch/csrc/jit/frontend/resolver.h`


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


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

- **File Documentation**: `builtin_functions.cpp_docs.md`
- **Keyword Index**: `builtin_functions.cpp_kw.md`
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

- **Neural Network**: Defines or uses PyTorch neural network components


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

- **File Documentation**: `builtin_functions.cpp_docs.md_docs.md`
- **Keyword Index**: `builtin_functions.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
