# Documentation: `torch/csrc/jit/frontend/lexer.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/frontend/lexer.cpp`
- **Size**: 2,449 bytes (2.39 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/frontend/lexer.h>

#include <c10/util/Exception.h>

#include <cstring>
#include <string>
#include <unordered_map>

namespace torch::jit {

static const std::unordered_map<int, int> binary_prec = {
    {TK_IF, 1},
    {TK_FOR, 1},
    {TK_AND, 2},
    {TK_OR, 2},
    // reserve a level for unary not
    {TK_IN, 4},
    {TK_NOTIN, 4},
    {'<', 4},
    {'>', 4},
    {TK_IS, 4},
    {TK_ISNOT, 4},
    {TK_EQ, 4},
    {TK_LE, 4},
    {TK_GE, 4},
    {TK_NE, 4},
    {'|', 5},
    {'^', 6},
    {'&', 7},
    {TK_LSHIFT, 8},
    {TK_RSHIFT, 8},
    {'+', 9},
    {'-', 9},
    {'*', 10},
    {'/', 10},
    {TK_FLOOR_DIV, 10},
    {'%', 10},
    {'@', 10},
    {TK_POW, 11},
};

static const std::unordered_map<int, int> unary_prec = {
    {TK_NOT, 3},
    {'~', 3},
    {'-', 10},
    {'*', 10},
};

bool SharedParserData::isUnary(int kind, int* prec) {
  auto it = unary_prec.find(kind);
  if (it != unary_prec.end()) {
    *prec = it->second;
    return true;
  }
  return false;
}
bool SharedParserData::isBinary(int kind, int* prec) {
  auto it = binary_prec.find(kind);
  if (it != binary_prec.end()) {
    *prec = it->second;
    return true;
  }
  return false;
}

C10_EXPORT int stringToKind(const std::string& str) {
  static std::unordered_map<std::string, int> str_to_kind = []() {
    std::unordered_map<std::string, int> ret_str_to_kind;
    ret_str_to_kind.reserve(std::strlen(valid_single_char_tokens));
    for (const char* tok = valid_single_char_tokens; *tok; tok++) {
      ret_str_to_kind[std::string(1, *tok)] = static_cast<unsigned char>(*tok);
    }
#define DEFINE_CASE(tok, _, str) \
  if (std::string(str) != "")    \
    ret_str_to_kind[str] = tok;
    TC_FORALL_TOKEN_KINDS(DEFINE_CASE)
#undef DEFINE_CASE
    return ret_str_to_kind;
  }();
  try {
    return str_to_kind.at(str);
  } catch (std::out_of_range&) {
    throw std::out_of_range("unknown token in stringToKind");
  }
}

C10_EXPORT std::string kindToString(int kind) {
  if (kind < 256)
    return std::string(1, static_cast<char>(kind));
  switch (kind) {
#define DEFINE_CASE(tok, str, _) \
  case tok:                      \
    return str;
    TC_FORALL_TOKEN_KINDS(DEFINE_CASE)
#undef DEFINE_CASE
    default:
      TORCH_CHECK(false, "Unknown kind: ", kind);
  }
}

C10_EXPORT SharedParserData& sharedParserData() {
  static SharedParserData data; // safely handles multi-threaded init
  return data;
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/frontend`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/frontend/lexer.h`
- `c10/util/Exception.h`
- `cstring`
- `string`
- `unordered_map`


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

- **File Documentation**: `lexer.cpp_docs.md`
- **Keyword Index**: `lexer.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
