# Documentation: `docs/torch/csrc/jit/mobile/model_tracer/MobileModelRunner.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/mobile/model_tracer/MobileModelRunner.h_docs.md`
- **Size**: 7,865 bytes (7.68 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/mobile/model_tracer/MobileModelRunner.h`

## File Metadata

- **Path**: `torch/csrc/jit/mobile/model_tracer/MobileModelRunner.h`
- **Size**: 5,086 bytes (4.97 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <mutex>
#include <sstream>

#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/script.h>

namespace torch::jit::mobile {

class MobileModelRunner {
  std::shared_ptr<torch::jit::mobile::Module> module_;

 public:
  explicit MobileModelRunner(std::string const& file_path) {
    module_ = std::make_shared<torch::jit::mobile::Module>(
        torch::jit::_load_for_mobile(file_path));
  }

  MobileModelRunner(
      std::string const& file_path,
      uint64_t module_load_options) {
    std::unordered_map<std::string, std::string> extra_files;
    module_ = std::make_shared<torch::jit::mobile::Module>(
        torch::jit::_load_for_mobile(
            file_path,
            at::Device(at::DeviceType::CPU, 0),
            extra_files,
            module_load_options));
  }

  MobileModelRunner(std::stringstream oss) {
    module_ = std::make_shared<torch::jit::mobile::Module>(
        torch::jit::_load_for_mobile(oss, at::Device(at::DeviceType::CPU, 0)));
  }

  /**
   * Returns true if the list of operators passed in has a Metal GPU operator,
   * and false otherwise.
   *
   */
  static bool set_has_metal_gpu_operators(std::set<std::string> const& op_list);

  /**
   * Fetches the set of root operators in the file "extra/mobile_info.json"
   * within the .ptl archive at location file_path.
   *
   * An exception is thrown if:
   *
   * 1. The file at file_path does not exist, or
   * 2. The contents of extra/mobile_info.json is not a JSON, or
   * 3. The file extra/mobile_info.json does not exist, or
   * 4. The JSON is malformed in some way and the operator list can not be
   * extracted correctly.
   *
   */
  static std::set<std::string> get_operators_from_mobile_info_json(
      std::string const& file_path);

  static std::vector<std::vector<at::IValue>> ivalue_to_bundled_inputs(
      const c10::IValue& bundled_inputs);

  static std::unordered_map<std::string, std::string>
  ivalue_to_bundled_inputs_map(const c10::IValue& bundled_inputs);

  /**
   * Fetches all the bundled inputs of the loaded mobile model.
   *
   * A bundled input itself is of type std::vector<at::IValue> and the
   * elements of this vector<> are the arguments that the "forward"
   * method of the model accepts. i.e. each of the at::IValue is a
   * single argument to the model's "forward" method.
   *
   * The outer vector holds a bundled input. For models with bundled
   * inputs, the outer most vector will have size > 0.
   */
  std::vector<std::vector<at::IValue>> get_all_bundled_inputs();

  /**
   * Fetches all the bundled inputs for all functions of the loaded mobile
   * model.
   *
   * The mapping is from 'function_names' eg 'forward' to bundled inputs for
   * that function
   *
   * A bundled input itself is of type std::vector<at::IValue> and the
   * elements of this vector<> are the arguments that the corresponding
   * method of the model accepts. i.e. each of the at::IValue in the entry
   * for forward is a single argument to the model's "forward" method.
   *
   * The outer vector of each value holds a bundled input. For models with
   * bundled inputs, the outer most vector will have size > 0.
   */
  std::unordered_map<std::string, std::vector<std::vector<at::IValue>>>
  get_many_functions_bundled_inputs();

  /**
   * Returns true if a model possesses get_bundled_inputs_functions_and_info()
   */
  bool has_new_style_bundled_inputs() const {
    return module_->find_method("get_bundled_inputs_functions_and_info") !=
        std::nullopt;
  }

  /**
   * For each tensor in bundled inputs, call the user-provided function 'func'.
   */
  void for_each_tensor_in_bundled_inputs(
      std::function<void(const ::at::Tensor&)> const& func);

  /**
   * Get the root operators directly called by this model's Bytecode.
   */
  std::set<std::string> get_root_operators() {
    return torch::jit::mobile::_export_operator_list(*module_);
  }

  /**
   * Runs the model against all of the provided inputs using the model's
   * "forward" method. Returns an std::vector<at::IValue>, where each element
   * of the returned vector is one of the return values from calling forward().
   */
  std::vector<at::IValue> run_with_inputs(
      std::vector<std::vector<at::IValue>> const& bundled_inputs);

  /**
   * Runs the model against all of the provided inputs for all the specified
   * function. Returns an std::vector<at::IValue>, where each element
   * of the returned vector is one of the return values from calling the
   * method named "function_name" on this model.
   */
  std::vector<at::IValue> run_with_inputs(
      const std::string& function_name,
      std::vector<std::vector<at::IValue>> const& bundled_inputs) const;

  /**
   * Attempts to run all functions in the passed in list if they exist. All
   * funcs should require no args
   */
  void run_argless_functions(const std::vector<std::string>& functions);
};

} // namespace torch::jit::mobile

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `MobileModelRunner`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/mobile/model_tracer`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `mutex`
- `sstream`
- `torch/csrc/autograd/grad_mode.h`
- `torch/csrc/jit/mobile/import.h`
- `torch/csrc/jit/mobile/module.h`
- `torch/csrc/jit/serialization/export.h`
- `torch/script.h`


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

Files in the same folder (`torch/csrc/jit/mobile/model_tracer`):

- [`CustomClassTracer.cpp_docs.md`](./CustomClassTracer.cpp_docs.md)
- [`BuildFeatureTracer.cpp_docs.md`](./BuildFeatureTracer.cpp_docs.md)
- [`KernelDTypeTracer.cpp_docs.md`](./KernelDTypeTracer.cpp_docs.md)
- [`OperatorCallTracer.h_docs.md`](./OperatorCallTracer.h_docs.md)
- [`TracerRunner.h_docs.md`](./TracerRunner.h_docs.md)
- [`tracer.cpp_docs.md`](./tracer.cpp_docs.md)
- [`TensorUtils.h_docs.md`](./TensorUtils.h_docs.md)
- [`BuildFeatureTracer.h_docs.md`](./BuildFeatureTracer.h_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`KernelDTypeTracer.h_docs.md`](./KernelDTypeTracer.h_docs.md)


## Cross-References

- **File Documentation**: `MobileModelRunner.h_docs.md`
- **Keyword Index**: `MobileModelRunner.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/mobile/model_tracer`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/mobile/model_tracer`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/jit/mobile/model_tracer`):

- [`OperatorCallTracer.cpp_kw.md_docs.md`](./OperatorCallTracer.cpp_kw.md_docs.md)
- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)
- [`TensorUtils.cpp_docs.md_docs.md`](./TensorUtils.cpp_docs.md_docs.md)
- [`OperatorCallTracer.cpp_docs.md_docs.md`](./OperatorCallTracer.cpp_docs.md_docs.md)
- [`CustomClassTracer.h_docs.md_docs.md`](./CustomClassTracer.h_docs.md_docs.md)
- [`BuildFeatureTracer.h_docs.md_docs.md`](./BuildFeatureTracer.h_docs.md_docs.md)
- [`CustomClassTracer.cpp_docs.md_docs.md`](./CustomClassTracer.cpp_docs.md_docs.md)
- [`TracerRunner.h_docs.md_docs.md`](./TracerRunner.h_docs.md_docs.md)
- [`TensorUtils.cpp_kw.md_docs.md`](./TensorUtils.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `MobileModelRunner.h_docs.md_docs.md`
- **Keyword Index**: `MobileModelRunner.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
