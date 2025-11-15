# Documentation: `torch/csrc/jit/frontend/versioned_symbols.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/frontend/versioned_symbols.cpp`
- **Size**: 4,019 bytes (3.92 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/frontend/versioned_symbols.h>

#include <caffe2/serialize/versions.h>
#include <torch/csrc/api/include/torch/jit.h>

#include <unordered_map>

namespace torch::jit {
// Note [Versioned Symbols]
// When the schema or behavior of a symbol changes, serialized Torchscript
// programs using that symbol are likely to break. To prevent those breaks,
// the symbol's historic behavior can be implemented as a Torchscript builtin
// and when an older Torchscript program is loaded the program's uses of the
// symbol can be replaced with the builtin.
//
// For example, a function _test_serialization_subcmul(a, b, alpha) might have
// been improperly implemented as (b - alpha * a).
// Some users may have written and serialized programs using that function,
// however, and fixing it to perform (a - alpha * b) would break their programs.
// Using the "Versioned Symbol" pattern lets you replace
// _test_serialization_subcmul in older programs with a builtin
// _test_serialization_subcmul<version_range> that implements the historic
// behavior. That way old programs preserve their semantics while new programs
// can take advantage of the fix.
//
// To do this:
//
// 1) Identify the file version range where the symbol should be replaced,
//    e.g. versions 0 to 2, inclusive.
// 2) Create one or more builtins implementing the symbol's historic behavior.
//    These should be named <function>_<start_version>_<end_version> and
//    go into the "upgraders" namespace.
//    For example, the test-only aten::_test_serialization_subcmul has a builtin
//    for its "historic" behavior called
//    upgraders::_test_serialization_subcmul_0_2.
// 3) Add a mapping from the symbol to the corresponding SymbolRange
//    in the symbol_range_map (below).
//
// To test your versioning:
//
// 1) Serialize a module demonstrating the historic behavior.
// 2) Save it to test/jit/fixtures.
// 3) Implement your new behavior and bump the version counter.
// 4) Write the builtins and extend the symbol_range_map per the above
//    instructions.
// 5) Create a test in jit/test_save_load.py that loads the old module
//    and verifies it exhibits the historic behavior, then saves and
//    loads the same module and verifies it exhibits the current behavior.
//    See test_versioned_symbols for an example.

// Helper to hold the version range (inclusive on both ends) and the symbol
// to map to for that range.
struct SymbolRange {
  SymbolRange(
      const uint64_t _start_version,
      const uint64_t _end_version,
      const Symbol _sym)
      : start_version_{_start_version},
        end_version_{_end_version},
        sym_{_sym} {}
  const uint64_t start_version_;
  const uint64_t end_version_;
  const Symbol sym_;
};

static std::unordered_map<Symbol, SymbolRange> symbol_range_map({
    {Symbol::fromQualString("aten::_test_serialization_subcmul"),
     {0,
      2,
      Symbol::fromQualString("upgraders::_test_serialization_subcmul_0_2")}},
    {Symbol::fromQualString("aten::div"),
     {0, 3, Symbol::fromQualString("upgraders::div_0_3")}},
    {Symbol::fromQualString("aten::div_"),
     {0, 3, Symbol::fromQualString("upgraders::div__0_3")}},
    {Symbol::fromQualString("aten::full"),
     {0, 4, Symbol::fromQualString("upgraders::full_0_4")}},
});

static std::unordered_map<NodeKind, uint64_t> kind_min_version_map({
    {aten::div, 4},
    {aten::div_, 4},
    {aten::full, 5}, // NOLINT(cppcoreguidelines-avoid-magic-numbers)
});

Symbol get_symbol_for_version(const Symbol name, const uint64_t version) {
  auto it = symbol_range_map.find(name);
  if (it == symbol_range_map.end()) {
    return name;
  }

  auto& entry = it->second;
  if (entry.start_version_ <= version && entry.end_version_ >= version) {
    return entry.sym_;
  }

  return name;
}

uint64_t get_min_version_for_kind(const NodeKind& kind) {
  auto it = kind_min_version_map.find(kind);
  if (it == kind_min_version_map.end()) {
    return 0;
  }

  return it->second;
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `SymbolRange`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/frontend`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/frontend/versioned_symbols.h`
- `caffe2/serialize/versions.h`
- `torch/csrc/api/include/torch/jit.h`
- `unordered_map`


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

- **File Documentation**: `versioned_symbols.cpp_docs.md`
- **Keyword Index**: `versioned_symbols.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
