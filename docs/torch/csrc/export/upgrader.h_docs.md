# Documentation: `torch/csrc/export/upgrader.h`

## File Metadata

- **Path**: `torch/csrc/export/upgrader.h`
- **Size**: 5,349 bytes (5.22 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <nlohmann/json.hpp>
#include <functional>
#include <string>
#include <vector>

namespace torch::_export {

/// Function type for upgrading JSON fields during schema version migration.
/// Takes a JSON field and returns the upgraded version of that field.
using UpgraderFunction = std::function<nlohmann::json(const nlohmann::json&)>;

/// Structure containing upgrader information for a specific keypath.
/// The version is stored as the map key in the registry, so it's not
/// duplicated here.
struct Upgrader {
  /// Path to the field that should be upgraded (e.g., {"graph_module", "graph",
  /// "nodes"}) Assuming top-level is a JSON object that represents
  /// ExportedProgram
  std::vector<std::string> keypath;

  /// Function that performs the actual upgrade transformation
  UpgraderFunction upgrade_func;

  /// Constructor for creating an upgrader with keypath and function
  Upgrader(std::vector<std::string> kp, UpgraderFunction func);

  /// Comparator for maintaining bottom-up ordering in the registry.
  /// Deeper keypaths are processed first to ensure safe upgrade application
  /// without conflicts between parent and child field modifications.
  bool operator<(const Upgrader& other) const;
};

/// Register an upgrader function for a specific schema version and keypath.
///
/// This function allows registration of custom upgrade logic that will be
/// applied when upgrading artifacts from the specified version. Upgraders
/// are applied in bottom-up order (deeper keypaths first) to prevent
/// conflicts between parent and child field modifications.
///
/// @param version The schema version this upgrader applies to
/// @param keypath The key path to the field that should be upgraded
/// @param upgrade_func Function that performs the upgrade transformation
void registerUpgrader(
    int version,
    const std::vector<std::string>& keypath,
    const UpgraderFunction& upgrade_func);

/// Register an upgrader function using dot-separated keypath notation.
///
/// Convenience overload that accepts dot-separated keypath strings for
/// simpler syntax. For example: "graph_module.graph.nodes" instead of
/// {"graph_module", "graph", "nodes"}.
///
/// @param version The schema version this upgrader applies to
/// @param dot_keypath Dot-separated keypath string (e.g., "graph.nodes")
/// @param upgrade_func Function that performs the upgrade transformation
void registerUpgrader(
    int version,
    const std::string& dot_keypath,
    const UpgraderFunction& upgrade_func);

/// Deregister an upgrader function for a specific schema version and keypath.
///
/// This function allows removal of previously registered upgrade logic for
/// the specified version and keypath. This is useful for testing scenarios
/// where you need to clean up registered upgraders or modify upgrader
/// behavior dynamically.
///
/// @param version The schema version to deregister the upgrader from
/// @param keypath The key path to the field that should be deregistered
/// @return true if an upgrader was found and removed, false otherwise
bool deregisterUpgrader(int version, const std::vector<std::string>& keypath);

/// Deregister an upgrader function using dot-separated keypath notation.
///
/// Convenience overload that accepts dot-separated keypath strings for
/// simpler syntax. For example: "graph_module.graph.nodes" instead of
/// {"graph_module", "graph", "nodes"}.
///
/// @param version The schema version to deregister the upgrader from
/// @param dot_keypath Dot-separated keypath string (e.g., "graph.nodes")
/// @return true if an upgrader was found and removed, false otherwise
bool deregisterUpgrader(int version, const std::string& dot_keypath);

/// Utility function for throwing consistent upgrader errors.
///
/// This function formats error messages in a standardized way for upgrader
/// failures, including version information and optional problematic object
/// details for debugging.
///
/// @param upgrader_name Name of the upgrader that failed
/// @param from_version Source schema version being upgraded from
/// @param error_message Descriptive error message
/// @param problematic_object Optional JSON object that caused the error
/// @throws std::runtime_error Always throws with formatted error message
void throwUpgraderError(
    const std::string& upgrader_name,
    int from_version,
    const std::string& error_message,
    const nlohmann::json& problematic_object = nlohmann::json::object());

/// Upgrade a JSON artifact to a specific target version with available
/// upgraders until a target version is reached.
///
/// This handles major version upgrade only. For minor version upgrade,
/// e.g. adding a new field with default value, it's automatically handled by
/// the default constructor in generated_serialization_types.h.
///
/// @param artifact The JSON artifact to upgrade(passed by value: function
/// operates on a local copy, original remains unmodified)
/// @param target_version The target schema version to upgrade to
/// @return The upgraded JSON artifact with updated schema version
/// @throws std::runtime_error if artifact is missing schema_version field
/// @throws std::runtime_error if final version doesn't match target version
nlohmann::json upgrade(nlohmann::json artifact, int target_version);

} // namespace torch::_export

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `Upgrader`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/export`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `nlohmann/json.hpp`
- `functional`
- `string`
- `vector`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


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

Files in the same folder (`torch/csrc/export`):

- [`pt2_archive_constants.h_docs.md`](./pt2_archive_constants.h_docs.md)
- [`pybind.cpp_docs.md`](./pybind.cpp_docs.md)
- [`example_upgraders.h_docs.md`](./example_upgraders.h_docs.md)
- [`example_upgraders.cpp_docs.md`](./example_upgraders.cpp_docs.md)
- [`pybind.h_docs.md`](./pybind.h_docs.md)
- [`upgrader.cpp_docs.md`](./upgrader.cpp_docs.md)


## Cross-References

- **File Documentation**: `upgrader.h_docs.md`
- **Keyword Index**: `upgrader.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
