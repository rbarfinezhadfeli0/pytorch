# Documentation: `docs/torch/csrc/jit/operator_upgraders/version_map.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/operator_upgraders/version_map.cpp_docs.md`
- **Size**: 7,293 bytes (7.12 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/operator_upgraders/version_map.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/operator_upgraders/version_map.cpp`
- **Size**: 4,937 bytes (4.82 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/operator_upgraders/version_map.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch::jit {

// this flag is used to make sure the elements in the version map
// are sorted according to when the upgraders are introduced.
static bool isVersionMapSorted = false;

// Main entry point for all operators that have valid upgraders.
// Note for developers: The list of upgraders should be SORTED
// by the version number where the upgrader is registered.
static std::unordered_map<std::string, std::vector<UpgraderEntry>> operatorVersionMap(
    {{"aten::logspace",
      {{9,
        "logspace_0_8",
        "aten::logspace(Scalar start, Scalar end, int? steps=None, float base=10.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"}}},
     {"aten::logspace.out",
      {{9,
        "logspace_out_0_8",
        "aten::logspace.out(Scalar start, Scalar end, int? steps=None, float base=10.0, *, Tensor(a!) out) -> Tensor(a!)"}}},
     {"aten::linspace",
      {{8,
        "linspace_0_7",
        "aten::linspace(Scalar start, Scalar end, int? steps=None, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"}}},
     {"aten::linspace.out",
      {{8,
        "linspace_out_0_7",
        "aten::linspace.out(Scalar start, Scalar end, int? steps=None, *, Tensor(a!) out) -> Tensor(a!)"}}},
     {"aten::div.Tensor",
      {{4,
        "div_Tensor_0_3",
        "aten::div.Tensor(Tensor self, Tensor other) -> Tensor"}}},
     {"aten::div.Tensor_mode",
      {{4,
        "div_Tensor_mode_0_3",
        "aten::div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor"}}},
     {"aten::div.Scalar",
      {{4,
        "div_Scalar_0_3",
        "aten::div.Scalar(Tensor self, Scalar other) -> Tensor"}}},
     {"aten::div.Scalar_mode",
      {{4,
        "div_Scalar_mode_0_3",
        "aten::div.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor"}}},
     {"aten::div.out",
      {{4,
        "div_out_0_3",
        "aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"}}},
     {"aten::div.out_mode",
      {{4,
        "div_out_mode_0_3",
        "aten::div.out_mode(Tensor self, Tensor other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)"}}},
     {"aten::div_.Tensor",
      {{4,
        "div__Tensor_0_3",
        "aten::div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)"}}},
     {"aten::div_.Tensor_mode",
      {{4,
        "div__Tensor_mode_0_3",
        "aten::div_.Tensor_mode(Tensor(a!) self, Tensor other, *, str? rounding_mode) -> Tensor(a!)"}}},
     {"aten::div_.Scalar",
      {{4,
        "div__Scalar_0_3",
        "aten::div_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)"}}},
     {"aten::div_.Scalar_mode",
      {{4,
        "div__Scalar_mode_0_3",
        "aten::div_.Scalar_mode(Tensor(a!) self, Scalar other, *, str? rounding_mode) -> Tensor(a!)"}}},
     {"aten::full",
      {{5,
        "full_0_4",
        "aten::full(int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"}}},
     {"aten::full.names",
      {{5,
        "full_names_0_4",
        "aten::full.names(int[] size, Scalar fill_value, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"}}},
     {"aten::full.out",
      {{5,
        "full_out_0_4",
        "aten::full.out(int[] size, Scalar fill_value, *, Tensor(a!) out) -> Tensor(a!)"}}},
     {"aten::gelu", {{10, "gelu_0_9", "aten::gelu(Tensor self) -> Tensor"}}},
     {"aten::gelu.out",
      {{10,
        "gelu_out_0_9",
        "aten::gelu.out(Tensor self, *, Tensor(a!) out) -> Tensor"}}}});

const std::unordered_map<std::string, std::vector<UpgraderEntry>>&
get_operator_version_map() {
  if (!isVersionMapSorted) {
    for (auto entry : operatorVersionMap) {
      std::sort(
          entry.second.begin(),
          entry.second.end(),
          [](const auto& a, const auto& b) {
            return a.bumped_at_version > b.bumped_at_version;
          });
    }
    isVersionMapSorted = true;
  }
  return operatorVersionMap;
}

void test_only_add_entry(const std::string& op_name, UpgraderEntry entry) {
  test_only_reset_flag();
  operatorVersionMap[op_name].emplace_back(std::move(entry));
}

void test_only_remove_entry(const std::string& op_name) {
  test_only_reset_flag();
  operatorVersionMap.erase(op_name);
}

void test_only_reset_flag() {
  isVersionMapSorted = false;
}

static bool calculatePackageVersionBasedOnUpgraders = false;

void calculate_package_version_based_on_upgraders(bool val) {
  calculatePackageVersionBasedOnUpgraders = val;
}

bool get_version_calculator_flag() {
  return calculatePackageVersionBasedOnUpgraders;
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/operator_upgraders`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/operator_upgraders/version_map.h`
- `algorithm`
- `string`
- `unordered_map`
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

Files in the same folder (`torch/csrc/jit/operator_upgraders`):

- [`utils.cpp_docs.md`](./utils.cpp_docs.md)
- [`upgraders_entry.cpp_docs.md`](./upgraders_entry.cpp_docs.md)
- [`utils.h_docs.md`](./utils.h_docs.md)
- [`version_map.h_docs.md`](./version_map.h_docs.md)
- [`upgraders.cpp_docs.md`](./upgraders.cpp_docs.md)
- [`upgraders.h_docs.md`](./upgraders.h_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`upgraders_entry.h_docs.md`](./upgraders_entry.h_docs.md)


## Cross-References

- **File Documentation**: `version_map.cpp_docs.md`
- **Keyword Index**: `version_map.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/operator_upgraders`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/operator_upgraders`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/jit/operator_upgraders`):

- [`utils.cpp_docs.md_docs.md`](./utils.cpp_docs.md_docs.md)
- [`version_map.h_kw.md_docs.md`](./version_map.h_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`upgraders_entry.cpp_docs.md_docs.md`](./upgraders_entry.cpp_docs.md_docs.md)
- [`utils.h_docs.md_docs.md`](./utils.h_docs.md_docs.md)
- [`upgraders.h_kw.md_docs.md`](./upgraders.h_kw.md_docs.md)
- [`upgraders_entry.cpp_kw.md_docs.md`](./upgraders_entry.cpp_kw.md_docs.md)
- [`version_map.cpp_kw.md_docs.md`](./version_map.cpp_kw.md_docs.md)
- [`utils.h_kw.md_docs.md`](./utils.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `version_map.cpp_docs.md_docs.md`
- **Keyword Index**: `version_map.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
