# Documentation: `docs/torch/csrc/utils/schema_info.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/utils/schema_info.h_docs.md`
- **Size**: 6,219 bytes (6.07 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/utils/schema_info.h`

## File Metadata

- **Path**: `torch/csrc/utils/schema_info.h`
- **Size**: 3,789 bytes (3.70 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/jit/frontend/function_schema_parser.h>
#include <unordered_set>

namespace torch::utils {

using SchemaSpecialCasePair =
    std::pair<c10::FunctionSchema, std::unordered_set<std::string>>;
/**
 * class SchemaInfo
 *
 * FunctionSchema wrapper that publicizes argument value specific operator
 * behavior (mutation, aliasing, special cases, etc...)
 */

struct TORCH_API SchemaInfo {
 public:
  explicit SchemaInfo(c10::FunctionSchema schema)
      : schema_(std::move(schema)),
        alias_maps_current_(false),
        has_init_(false) {}
  explicit SchemaInfo(const char* signature)
      : schema_(torch::jit::parseSchema(signature)),
        alias_maps_current_(false),
        has_init_(false) {}

  bool is_mutable();

  bool is_mutable(const c10::SchemaArgument& argument);

  bool is_mutable(std::string_view name);

  bool has_argument(std::string_view name);

  bool is_nondeterministic() const;

  // Returns whether lhs and rhs may alias directly.
  // This does not account for cases where lhs or rhs are a container that
  // may contain elements that alias the other argument.
  // Besides the checks already included in FunctionSchema::may_alias, this
  // method also accounts special aliasing cases causes by aliasing argument
  // values supplied from addArgumentValue.
  bool may_alias(
      const c10::SchemaArgument& lhs,
      const c10::SchemaArgument& rhs);

  // Returns whether lhs and rhs may alias directly or whether lhs/rhs are a
  // container that may contain elements that alias the other argument. Besides
  // the checks already included in FunctionSchema::may_contain_alias, this
  // method also accounts for special aliasing cases causes by aliasing argument
  // values supplied from addArgumentValue. bidirectional = false only returns
  // whether lhs may contain an alias of rhs while bidirectional = true returns
  // both directions.
  bool may_contain_alias(
      const c10::SchemaArgument& lhs,
      const c10::SchemaArgument& rhs,
      bool bidirectional = true);

  void addArgumentValue(const std::string& name, const at::IValue& value);

  void addArgumentValues(
      const std::vector<std::optional<at::IValue>>& value_list);

  void addArgumentValues(
      const std::unordered_map<std::string, at::IValue>& values);

  bool hasInputArgumentNamed(const std::string& name) const;

 private:
  // This function enforces more conservative results when the TORCH_WARN is
  // triggered from above due to duplicates in an argument list
  void ensureConservativity(
      const std::unordered_set<at::Symbol>& duplicates,
      const std::vector<c10::Argument>& arguments_list,
      c10::SchemaArgType type);

  void initSchemaInfo();

  void generateAliasMaps();

  bool mayContainAliasImpl(
      const c10::SchemaArgument& lhs,
      const c10::SchemaArgument& rhs);

  static std::vector<c10::FunctionSchema> getNonDeterministicOps();

  static std::vector<SchemaSpecialCasePair> getTrainingOps();

  const std::unordered_set<c10::SchemaArgument>& wildcardSet();

  const std::unordered_set<c10::SchemaArgument>& containerSet();

  // Set of all wildcard arguments
  std::unordered_set<c10::SchemaArgument> wildcard_set_;

  // Set of all container arguments
  std::unordered_set<c10::SchemaArgument> container_set_;

  // Map of argument IValues
  std::unordered_map<std::string, at::IValue> value_map_;

  // Alias map of inputs with each other
  std::vector<std::unordered_set<size_t>> input_alias_map_;

  // Alias map of outputs to inputs
  std::vector<std::unordered_set<size_t>> output_alias_map_;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const c10::FunctionSchema schema_;

  bool alias_maps_current_;

  bool has_init_;
};
} // namespace torch::utils

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 17 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `SchemaInfo`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/frontend/function_schema_parser.h`
- `unordered_set`


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

Files in the same folder (`torch/csrc/utils`):

- [`tensor_list.h_docs.md`](./tensor_list.h_docs.md)
- [`disable_torch_function.cpp_docs.md`](./disable_torch_function.cpp_docs.md)
- [`tensor_new.cpp_docs.md`](./tensor_new.cpp_docs.md)
- [`tensor_apply.cpp_docs.md`](./tensor_apply.cpp_docs.md)
- [`cpp_stacktraces.cpp_docs.md`](./cpp_stacktraces.cpp_docs.md)
- [`numpy_stub.h_docs.md`](./numpy_stub.h_docs.md)
- [`nested.h_docs.md`](./nested.h_docs.md)
- [`nested.cpp_docs.md`](./nested.cpp_docs.md)
- [`six.h_docs.md`](./six.h_docs.md)
- [`python_scalars.h_docs.md`](./python_scalars.h_docs.md)


## Cross-References

- **File Documentation**: `schema_info.h_docs.md`
- **Keyword Index**: `schema_info.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/utils`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/utils`, which is part of the **core PyTorch library**.



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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/csrc/utils`):

- [`python_tuples.h_kw.md_docs.md`](./python_tuples.h_kw.md_docs.md)
- [`six.h_kw.md_docs.md`](./six.h_kw.md_docs.md)
- [`tensor_types.cpp_docs.md_docs.md`](./tensor_types.cpp_docs.md_docs.md)
- [`tensor_list.h_kw.md_docs.md`](./tensor_list.h_kw.md_docs.md)
- [`verbose.h_kw.md_docs.md`](./verbose.h_kw.md_docs.md)
- [`invalid_arguments.cpp_kw.md_docs.md`](./invalid_arguments.cpp_kw.md_docs.md)
- [`tensor_apply.h_kw.md_docs.md`](./tensor_apply.h_kw.md_docs.md)
- [`cuda_enabled.h_docs.md_docs.md`](./cuda_enabled.h_docs.md_docs.md)
- [`tensor_layouts.h_docs.md_docs.md`](./tensor_layouts.h_docs.md_docs.md)
- [`variadic.h_kw.md_docs.md`](./variadic.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `schema_info.h_docs.md_docs.md`
- **Keyword Index**: `schema_info.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
