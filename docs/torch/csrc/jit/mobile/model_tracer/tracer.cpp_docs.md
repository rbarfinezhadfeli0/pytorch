# Documentation: `torch/csrc/jit/mobile/model_tracer/tracer.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/mobile/model_tracer/tracer.cpp`
- **Size**: 6,811 bytes (6.65 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <iostream>
#include <sstream>
#include <string>

/**
 * The tracer.cpp generates a binary that accepts multiple Torch Mobile Model(s)
 * (with bytecode.pkl), each of which has at least 1 bundled
 * input. This binary then feeds the bundled input(s) into each corresponding
 * model and executes it using the lite interpreter.
 *
 * Both root operators as well as called operators are recorded and saved
 * into a YAML file (whose path is provided on the command line).
 *
 * Note: Root operators may include primary and other operators that
 * are not invoked using the dispatcher, and hence they may not show
 * up in the Traced Operator list.
 *
 */

#include <ATen/core/dispatch/ObservedOperators.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/model_tracer/KernelDTypeTracer.h>
#include <torch/csrc/jit/mobile/model_tracer/MobileModelRunner.h>
#include <torch/csrc/jit/mobile/model_tracer/OperatorCallTracer.h>
#include <torch/csrc/jit/mobile/model_tracer/TensorUtils.h>
#include <torch/csrc/jit/mobile/model_tracer/TracerRunner.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/parse_operators.h>
#include <torch/script.h>

typedef std::map<std::string, std::set<std::string>> kt_type;

C10_DEFINE_string(
    model_input_path,
    "",
    "A comma separated list of path(s) to the input model file(s) (.ptl).");

C10_DEFINE_string(
    build_yaml_path,
    "",
    "The path of the output YAML file containing traced operator information.");

#define REQUIRE_STRING_ARG(name)                            \
  if (FLAGS_##name.empty()) {                               \
    std::cerr << "You must specify the flag --" #name "\n"; \
    return 1;                                               \
  }

#define REQUIRE_INT_ARG(name)                               \
  if (FLAGS_##name == -1) {                                 \
    std::cerr << "You must specify the flag --" #name "\n"; \
    return 1;                                               \
  }

static void printOpYAML(
    std::ostream& out,
    int indent,
    const std::string& op_name,
    bool is_used_for_training,
    bool is_root_operator,
    bool include_all_overloads) {
  out << std::string(indent, ' ') << op_name << ":" << '\n';
  out << std::string(indent + 2, ' ')
      << "is_used_for_training: " << (is_used_for_training ? "true" : "false")
      << '\n';
  out << std::string(indent + 2, ' ')
      << "is_root_operator: " << (is_root_operator ? "true" : "false") << '\n';
  out << std::string(indent + 2, ' ')
      << "include_all_overloads: " << (include_all_overloads ? "true" : "false")
      << '\n';
}

static void printOpsYAML(
    std::ostream& out,
    const std::set<std::string>& operator_list,
    bool is_used_for_training,
    bool is_root_operator,
    bool include_all_overloads) {
  for (auto& it : operator_list) {
    printOpYAML(out, 2, it, false, is_root_operator, false);
  }
}

static void printDTypeYAML(
    std::ostream& out,
    int indent,
    const std::string& kernel_tag_name,
    const std::set<std::string>& dtypes) {
  std::string indent_str = std::string(indent, ' ');
  out << indent_str << kernel_tag_name << ":" << '\n';
  for (auto& dtype : dtypes) {
    out << indent_str << "- " << dtype << '\n';
  }
}

static void printDTypesYAML(
    std::ostream& out,
    const torch::jit::mobile::KernelDTypeTracer::kernel_tags_type&
        kernel_tags) {
  for (auto& it : kernel_tags) {
    printDTypeYAML(out, 2, it.first, it.second);
  }
}

static void printCustomClassesYAML(
    std::ostream& out,
    const torch::jit::mobile::CustomClassTracer::custom_classes_type&
        loaded_classes) {
  for (auto& class_name : loaded_classes) {
    out << "- " << class_name << '\n';
  }
}

/**
 * Runs multiple PyTorch lite interpreter models, and additionally writes
 * out a list of root and called operators, kernel dtypes, and loaded/used
 * TorchBind custom classes.
 */
int main(int argc, char* argv[]) {
  if (!c10::ParseCommandLineFlags(&argc, &argv)) {
    std::cerr << "Failed to parse command line flags!" << '\n';
    return 1;
  }

  REQUIRE_STRING_ARG(model_input_path);
  REQUIRE_STRING_ARG(build_yaml_path);

  std::istringstream sin(FLAGS_model_input_path);
  std::ofstream yaml_out(FLAGS_build_yaml_path);

  std::cout << "Output: " << FLAGS_build_yaml_path << '\n';
  torch::jit::mobile::TracerResult tracer_result;
  std::vector<std::string> model_input_paths;

  for (std::string model_input_path;
       std::getline(sin, model_input_path, ',');) {
    std::cout << "Processing: " << model_input_path << '\n';
    model_input_paths.push_back(model_input_path);
  }

  try {
    tracer_result = torch::jit::mobile::trace_run(model_input_paths);
  } catch (std::exception& ex) {
    std::cerr
        << "ModelTracer has not been able to load the module for the following reasons:\n"
        << ex.what()
        << "\nPlease consider opening an issue at https://github.com/pytorch/pytorch/issues "
        << "with the detailed error message." << '\n';

    throw ex;
  }

  if (tracer_result.traced_operators.size() <=
      torch::jit::mobile::always_included_traced_ops.size()) {
    std::cerr
        << c10::str(
               "Error traced_operators size: ",
               tracer_result.traced_operators.size(),
               ". Expected the traced operator list to be bigger then the default size ",
               torch::jit::mobile::always_included_traced_ops.size(),
               ". Please report a bug in PyTorch.")
        << '\n';
  }

  // If the op exist in both traced_ops and root_ops, leave it in root_ops only
  for (const auto& root_op : tracer_result.root_ops) {
    if (tracer_result.traced_operators.find(root_op) !=
        tracer_result.traced_operators.end()) {
      tracer_result.traced_operators.erase(root_op);
    }
  }

  yaml_out << "include_all_non_op_selectives: false" << '\n';
  yaml_out << "build_features: []" << '\n';
  yaml_out << "operators:" << '\n';
  printOpsYAML(
      yaml_out,
      tracer_result.root_ops,
      false /* is_used_for_training */,
      true /* is_root_operator */,
      false /* include_all_overloads */);
  printOpsYAML(
      yaml_out,
      tracer_result.traced_operators,
      false /* is_used_for_training */,
      false /* is_root_operator */,
      false /* include_all_overloads */);

  yaml_out << "kernel_metadata:";
  if (tracer_result.called_kernel_tags.empty()) {
    yaml_out << " []";
  }
  yaml_out << '\n';
  printDTypesYAML(yaml_out, tracer_result.called_kernel_tags);

  yaml_out << "custom_classes:";
  if (tracer_result.loaded_classes.empty()) {
    yaml_out << " []";
  }
  yaml_out << '\n';
  printCustomClassesYAML(yaml_out, tracer_result.loaded_classes);

  return 0;
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 16 function(s).

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/mobile/model_tracer`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `iostream`
- `sstream`
- `string`
- `ATen/core/dispatch/ObservedOperators.h`
- `torch/csrc/autograd/grad_mode.h`
- `torch/csrc/jit/mobile/import.h`
- `torch/csrc/jit/mobile/model_tracer/KernelDTypeTracer.h`
- `torch/csrc/jit/mobile/model_tracer/MobileModelRunner.h`
- `torch/csrc/jit/mobile/model_tracer/OperatorCallTracer.h`
- `torch/csrc/jit/mobile/model_tracer/TensorUtils.h`
- `torch/csrc/jit/mobile/model_tracer/TracerRunner.h`
- `torch/csrc/jit/mobile/module.h`
- `torch/csrc/jit/mobile/parse_operators.h`
- `torch/script.h`


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

Files in the same folder (`torch/csrc/jit/mobile/model_tracer`):

- [`CustomClassTracer.cpp_docs.md`](./CustomClassTracer.cpp_docs.md)
- [`BuildFeatureTracer.cpp_docs.md`](./BuildFeatureTracer.cpp_docs.md)
- [`KernelDTypeTracer.cpp_docs.md`](./KernelDTypeTracer.cpp_docs.md)
- [`OperatorCallTracer.h_docs.md`](./OperatorCallTracer.h_docs.md)
- [`TracerRunner.h_docs.md`](./TracerRunner.h_docs.md)
- [`TensorUtils.h_docs.md`](./TensorUtils.h_docs.md)
- [`BuildFeatureTracer.h_docs.md`](./BuildFeatureTracer.h_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`KernelDTypeTracer.h_docs.md`](./KernelDTypeTracer.h_docs.md)


## Cross-References

- **File Documentation**: `tracer.cpp_docs.md`
- **Keyword Index**: `tracer.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
