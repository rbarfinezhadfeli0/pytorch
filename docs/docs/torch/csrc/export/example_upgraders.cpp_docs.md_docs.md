# Documentation: `docs/torch/csrc/export/example_upgraders.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/export/example_upgraders.cpp_docs.md`
- **Size**: 5,055 bytes (4.94 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/export/example_upgraders.cpp`

## File Metadata

- **Path**: `torch/csrc/export/example_upgraders.cpp`
- **Size**: 2,920 bytes (2.85 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/export/example_upgraders.h>
#include <torch/csrc/export/upgrader.h>

namespace torch::_export {

/// Register test upgraders for the upgrader system.
/// and shows some common upgrade patterns.
static bool test_upgraders_registered = false;

void registerExampleUpgraders() {
  if (test_upgraders_registered) {
    return;
  }

  registerUpgrader(
      0,
      "graph_module.graph.nodes",
      [](const nlohmann::json& nodes_array) -> nlohmann::json {
        nlohmann::json upgraded_nodes = nodes_array;

        // Process each node in the nodes array
        for (auto& node : upgraded_nodes) {
          if (node.contains("metadata") && node["metadata"].is_object()) {
            // Process each metadata key-value pair
            for (auto& [key, value] : node["metadata"].items()) {
              if (key == "nn_module_stack") {
                // Transform nn_module_stack values by prepending prefix
                if (value.is_string()) {
                  std::string stack_str = value.get<std::string>();
                  value = "test_upgrader_" + stack_str;
                } else {
                  throwUpgraderError(
                      "version_0_upgrader_registered",
                      0,
                      "nn_module_stack metadata value must be a string, got: " +
                          std::string(value.type_name()),
                      node);
                }
              }
              // Other metadata keys remain unchanged
            }
          }
        }

        return upgraded_nodes;
      });

  registerUpgrader(
      0,
      "graph_module.graph",
      [](const nlohmann::json& graph_obj) -> nlohmann::json {
        nlohmann::json upgraded_graph = graph_obj;

        // Rename field if it exists in the graph object
        if (upgraded_graph.contains("old_test_field")) {
          upgraded_graph["new_test_field"] = upgraded_graph["old_test_field"];
          upgraded_graph.erase("old_test_field");
        }

        return upgraded_graph;
      });

  registerUpgrader(
      1,
      std::vector<std::string>{"graph_module", "graph"},
      [](const nlohmann::json& graph_obj) -> nlohmann::json {
        nlohmann::json upgraded_graph = graph_obj;

        // Continue the field renaming chain from version 0
        if (upgraded_graph.contains("new_test_field")) {
          upgraded_graph["new_test_field2"] = upgraded_graph["new_test_field"];
          upgraded_graph.erase("new_test_field");
        }

        return upgraded_graph;
      });

  test_upgraders_registered = true;
}

/// Deregister test upgraders for the upgrader system.
void deregisterExampleUpgraders() {
  deregisterUpgrader(0, "graph_module.graph.nodes");
  deregisterUpgrader(0, "graph_module.graph");
  deregisterUpgrader(1, std::vector<std::string>{"graph_module", "graph"});
  test_upgraders_registered = false;
}

} // namespace torch::_export

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/export`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/export/example_upgraders.h`
- `torch/csrc/export/upgrader.h`


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
- [`upgrader.h_docs.md`](./upgrader.h_docs.md)
- [`pybind.h_docs.md`](./pybind.h_docs.md)
- [`upgrader.cpp_docs.md`](./upgrader.cpp_docs.md)


## Cross-References

- **File Documentation**: `example_upgraders.cpp_docs.md`
- **Keyword Index**: `example_upgraders.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/export`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/export`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/csrc/export`):

- [`upgrader.cpp_kw.md_docs.md`](./upgrader.cpp_kw.md_docs.md)
- [`pt2_archive_constants.h_docs.md_docs.md`](./pt2_archive_constants.h_docs.md_docs.md)
- [`example_upgraders.h_docs.md_docs.md`](./example_upgraders.h_docs.md_docs.md)
- [`example_upgraders.h_kw.md_docs.md`](./example_upgraders.h_kw.md_docs.md)
- [`example_upgraders.cpp_kw.md_docs.md`](./example_upgraders.cpp_kw.md_docs.md)
- [`pybind.h_docs.md_docs.md`](./pybind.h_docs.md_docs.md)
- [`upgrader.cpp_docs.md_docs.md`](./upgrader.cpp_docs.md_docs.md)
- [`pt2_archive_constants.h_kw.md_docs.md`](./pt2_archive_constants.h_kw.md_docs.md)
- [`upgrader.h_docs.md_docs.md`](./upgrader.h_docs.md_docs.md)
- [`pybind.h_kw.md_docs.md`](./pybind.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `example_upgraders.cpp_docs.md_docs.md`
- **Keyword Index**: `example_upgraders.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
