# Documentation: `docs/torch/csrc/jit/ir/graph_utils.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/ir/graph_utils.cpp_docs.md`
- **Size**: 5,246 bytes (5.12 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/ir/graph_utils.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/ir/graph_utils.cpp`
- **Size**: 2,916 bytes (2.85 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/ir/graph_utils.h>

namespace torch::jit {

TypePtr getTensorType(const at::Tensor& t, bool complete) {
  auto r = TensorType::create(t);
  if (!complete) {
    r = r->dimensionedOnly();
  }
  return r;
}

TypePtr inferShapeAndTypeForInput(
    TypePtr input_type,
    Stack::const_iterator& s_iter,
    const Stack::const_iterator& s_iter_end,
    bool complete) {
  if (auto tuple_type = input_type->cast<TupleType>()) {
    std::vector<TypePtr> types;
    for (const auto& sub_type : tuple_type->containedTypes()) {
      TORCH_INTERNAL_ASSERT(s_iter != s_iter_end);
      types.emplace_back(
          inferShapeAndTypeForInput(sub_type, s_iter, s_iter_end, complete));
    }
    return TupleType::create(types);
  } else if (auto list_type = input_type->cast<ListType>()) {
    const TypePtr& sub_type = list_type->getElementType();
    auto elem_type =
        inferShapeAndTypeForInput(sub_type, s_iter, s_iter_end, complete);
    return ListType::create(elem_type);
  } else if (auto tensor_type = input_type->cast<TensorType>()) {
    auto type = getTensorType(s_iter->toTensor(), complete);
    s_iter++;
    return type;
  } else if (auto optional_type = input_type->cast<OptionalType>()) {
    const TypePtr& sub_type = optional_type->getElementType();
    auto elem_type =
        inferShapeAndTypeForInput(sub_type, s_iter, s_iter_end, complete);
    return OptionalType::create(elem_type);
  } else {
    // Primitive type, keep as is.
    s_iter++;
    return input_type;
  }
}

void setInputTensorTypes(
    Graph& g,
    const Stack& stack,
    bool complete,
    const std::vector<int>& param_count_list) {
  at::ArrayRef<Value*> input_values = g.inputs();
  auto s_iter = stack.begin();
  size_t list_idx = 0;
  if (!param_count_list.empty()) {
    TORCH_INTERNAL_ASSERT(
        input_values.size() == param_count_list.size(),
        " input_values:",
        input_values.size(),
        " vs param_count_list:",
        param_count_list.size());
  }
  for (auto v : input_values) {
    // Leave packed param types alone. This is needed for downstream passes
    // (like alias analysis) to work properly. This will be unpacked later
    // in unpackQuantizedWeights.
    if (auto named_type = v->type()->cast<c10::NamedType>()) {
      if (auto qualname = named_type->name()) {
        if (getCustomClass(qualname->qualifiedName())) {
          if (param_count_list.empty()) {
            AT_ASSERT(s_iter != stack.end());
            s_iter++;
          } else {
            if (param_count_list[list_idx] > 0) {
              AT_ASSERT(s_iter != stack.end());
            }
            s_iter += param_count_list[list_idx];
          }
          list_idx++;
          continue;
        }
      }
    }
    auto type =
        inferShapeAndTypeForInput(v->type(), s_iter, stack.end(), complete);
    v->setType(type);
    list_idx++;
  }
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/ir`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/ir/graph_utils.h`


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

Files in the same folder (`torch/csrc/jit/ir`):

- [`node_hashing.h_docs.md`](./node_hashing.h_docs.md)
- [`constants.cpp_docs.md`](./constants.cpp_docs.md)
- [`subgraph_matcher.h_docs.md`](./subgraph_matcher.h_docs.md)
- [`scope.cpp_docs.md`](./scope.cpp_docs.md)
- [`graph_node_list.h_docs.md`](./graph_node_list.h_docs.md)
- [`type_hashing.cpp_docs.md`](./type_hashing.cpp_docs.md)
- [`ir.h_docs.md`](./ir.h_docs.md)
- [`ir.cpp_docs.md`](./ir.cpp_docs.md)
- [`irparser.cpp_docs.md`](./irparser.cpp_docs.md)
- [`node_hashing.cpp_docs.md`](./node_hashing.cpp_docs.md)


## Cross-References

- **File Documentation**: `graph_utils.cpp_docs.md`
- **Keyword Index**: `graph_utils.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/ir`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/ir`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/jit/ir`):

- [`subgraph_matcher.h_docs.md_docs.md`](./subgraph_matcher.h_docs.md_docs.md)
- [`node_hashing.h_kw.md_docs.md`](./node_hashing.h_kw.md_docs.md)
- [`subgraph_matcher.h_kw.md_docs.md`](./subgraph_matcher.h_kw.md_docs.md)
- [`graph_utils.h_kw.md_docs.md`](./graph_utils.h_kw.md_docs.md)
- [`irparser.cpp_docs.md_docs.md`](./irparser.cpp_docs.md_docs.md)
- [`constants.h_docs.md_docs.md`](./constants.h_docs.md_docs.md)
- [`scope.h_kw.md_docs.md`](./scope.h_kw.md_docs.md)
- [`scope.h_docs.md_docs.md`](./scope.h_docs.md_docs.md)
- [`irparser.cpp_kw.md_docs.md`](./irparser.cpp_kw.md_docs.md)
- [`scope.cpp_docs.md_docs.md`](./scope.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `graph_utils.cpp_docs.md_docs.md`
- **Keyword Index**: `graph_utils.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
