# Documentation: `torch/csrc/jit/mobile/model_tracer/MobileModelRunner.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/mobile/model_tracer/MobileModelRunner.cpp`
- **Size**: 8,021 bytes (7.83 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/mobile/model_tracer/MobileModelRunner.h>
#include <torch/csrc/jit/mobile/model_tracer/TensorUtils.h>

namespace torch::jit::mobile {

std::vector<std::vector<at::IValue>> MobileModelRunner::
    ivalue_to_bundled_inputs(const c10::IValue& bundled_inputs) {
  CAFFE_ENFORCE(
      bundled_inputs.isList(),
      "Expected get_all_bundled_inputs to ",
      "return a list but got a ",
      bundled_inputs.tagKind(),
      " instead");

  c10::List<at::IValue> all_inputs = bundled_inputs.toList();
  CAFFE_ENFORCE(
      !all_inputs.empty(),
      "Expected at least 1 bundled input, ",
      "but found none. Please use ",
      "torch.utils.bundled_inputs.augment_model_with_bundled_inputs to add.");

  std::vector<std::vector<at::IValue>> ret;
  for (at::IValue input : all_inputs) {
    CAFFE_ENFORCE(
        input.isTuple(),
        "Expected list element to be a tuple ",
        "but got a ",
        input.tagKind(),
        " instead");
    ret.push_back(input.toTupleRef().elements());
  }

  return ret;
}

std::unordered_map<std::string, std::string> MobileModelRunner::
    ivalue_to_bundled_inputs_map(const c10::IValue& bundled_inputs) {
  CAFFE_ENFORCE(
      bundled_inputs.isGenericDict(),
      "Expected get_bundled_inputs_functions_and_info to ",
      "return a dict but got a ",
      bundled_inputs.tagKind(),
      " instead");

  c10::Dict<at::IValue, at::IValue> all_inputs = bundled_inputs.toGenericDict();
  CAFFE_ENFORCE(
      !all_inputs.empty(),
      "Expected at least 1 function with bundled inputs, ",
      "but found none. Please use ",
      "torch.utils.bundled_inputs.augment_model_with_bundled_inputs to add.");

  std::unordered_map<std::string, std::string> ret;
  for (auto& input : all_inputs) {
    const at::IValue& function_name = input.key();
    const at::IValue& nested_dict = input.value();
    CAFFE_ENFORCE(
        function_name.isString(),
        "Expected function with inputs to be a string ",
        "but got a ",
        function_name.tagKind(),
        " instead");
    CAFFE_ENFORCE(
        nested_dict.isGenericDict(),
        "Expected function name to map to dictionary ",
        "but got a ",
        nested_dict.tagKind(),
        " instead");

    // Got the nested dict now need to convert that into std types
    c10::Dict<at::IValue, at::IValue> function_and_info_ival_dict =
        nested_dict.toGenericDict();
    std::unordered_map<std::string, std::vector<std::string>>
        function_and_info_dict;
    for (auto& entry : function_and_info_ival_dict) {
      const at::IValue& key = entry.key();
      const at::IValue& value = entry.value();
      CAFFE_ENFORCE(
          key.isString(),
          "Expected extra information key to be a string ",
          "but got a ",
          value.tagKind(),
          " instead");
      CAFFE_ENFORCE(
          value.isList(),
          "Expected extra information values to be a list ",
          "but got a ",
          value.tagKind(),
          " instead");

      // Got the value of the nested dict entry now need to convert it to std
      // types
      std::vector<std::string> data_list;
      c10::List<at::IValue> ival_data = value.toList();
      for (at::IValue data : ival_data) {
        CAFFE_ENFORCE(
            data.isString(),
            "Expected list element of nested dict entries to be a string ",
            "but got a ",
            data.tagKind(),
            " instead");
        data_list.push_back(data.toStringRef());
      }

      // Add entry into std type mapping
      function_and_info_dict[key.toStringRef()] = data_list;
    }

    // Could store the full mapping of std types, but the 'info' section isn't
    // needed here
    std::string input_function =
        function_and_info_dict["get_inputs_function_name"][0];
    ret[function_name.toStringRef()] = input_function;
  }

  return ret;
}

std::vector<std::vector<at::IValue>> MobileModelRunner::
    get_all_bundled_inputs() {
  auto has_bundled_input = module_->find_method("get_all_bundled_inputs");
  CAFFE_ENFORCE(
      has_bundled_input,
      "Model does not have bundled inputs. ",
      "Use torch.utils.bundled_inputs.augment_model_with_bundled_inputs to add.");

  c10::IValue bundled_inputs = module_->run_method("get_all_bundled_inputs");
  return ivalue_to_bundled_inputs(bundled_inputs);
}

std::unordered_map<std::string, std::vector<std::vector<at::IValue>>>
MobileModelRunner::get_many_functions_bundled_inputs() {
  auto has_bundled_input =
      module_->find_method("get_bundled_inputs_functions_and_info");
  CAFFE_ENFORCE(
      has_bundled_input,
      "Model does not have bundled inputs. ",
      "Use torch.utils.bundled_inputs.augment_many_model_functions_with_bundled_inputs to add.");

  auto ival_bundled_inputs_mapping =
      module_->run_method("get_bundled_inputs_functions_and_info");
  auto bundled_inputs_mapping =
      ivalue_to_bundled_inputs_map(ival_bundled_inputs_mapping);

  std::unordered_map<std::string, std::vector<std::vector<at::IValue>>> ret;

  for (auto& entry : bundled_inputs_mapping) {
    std::string function_name = entry.first;
    std::string function_to_call = entry.second;

    auto has_func_to_call = module_->find_method(function_to_call);
    CAFFE_ENFORCE(
        has_func_to_call,
        "Model does not have ",
        function_to_call,
        "Use torch.utils.bundled_inputs.augment_many_model_functions_with_bundled_inputs to add.");

    c10::IValue bundled_inputs = module_->run_method(function_to_call);
    ret[function_name] = ivalue_to_bundled_inputs(bundled_inputs);
  }
  return ret;
}

std::vector<at::IValue> MobileModelRunner::run_with_inputs(
    std::vector<std::vector<at::IValue>> const& bundled_inputs) {
  std::vector<at::IValue> ret;
  ret.reserve(bundled_inputs.size());
  for (std::vector<at::IValue> const& input : bundled_inputs) {
    ret.emplace_back(module_->forward(input));
  }
  return ret;
}

std::vector<at::IValue> MobileModelRunner::run_with_inputs(
    const std::string& function_name,
    std::vector<std::vector<at::IValue>> const& bundled_inputs) const {
  std::vector<at::IValue> ret;
  ret.reserve(bundled_inputs.size());
  auto has_bundled_input = module_->find_method(function_name);
  CAFFE_ENFORCE(
      has_bundled_input,
      "Model does not have the method named ",
      function_name,
      "Please ensure that it was exported correctly");
  for (std::vector<at::IValue> const& input : bundled_inputs) {
    auto func = module_->get_method(function_name);
    ret.emplace_back(func(input));
  }
  return ret;
}

void MobileModelRunner::run_argless_functions(
    const std::vector<std::string>& functions) {
  for (auto& function_name : functions) {
    if (module_->find_method(function_name)) {
      module_->run_method(function_name);
    }
  }
}

bool MobileModelRunner::set_has_metal_gpu_operators(
    std::set<std::string> const& op_list) {
  for (std::string const& op : op_list) {
    if (op.find("metal::") == 0 || op.find("metal_prepack::") == 0 ||
        op.find("metal_prepack_unet::") == 0) {
      return true;
    }
  }
  return false;
}

void MobileModelRunner::for_each_tensor_in_bundled_inputs(
    std::function<void(const ::at::Tensor&)> const& func) {
  if (has_new_style_bundled_inputs()) {
    // Get the bundled inputs and access the arg level ivalues stored within
    auto bundled_inputs_mapping = this->get_many_functions_bundled_inputs();

    // Loop over functions
    for (auto& entry : bundled_inputs_mapping) {
      std::vector<std::vector<at::IValue>> bundled_inputs = entry.second;
      // Loop through inputs
      for (const std::vector<at::IValue>& input : bundled_inputs) {
        // Loop through values in an input
        for (const at::IValue& iv : input) {
          for_each_tensor_in_ivalue(iv, func);
        }
      }
    }
  } else {
    c10::IValue iv = module_->run_method("get_all_bundled_inputs");
    for_each_tensor_in_ivalue(iv, func);
  }
}
} // namespace torch::jit::mobile

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/mobile/model_tracer`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/mobile/model_tracer/MobileModelRunner.h`
- `torch/csrc/jit/mobile/model_tracer/TensorUtils.h`


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

- **File Documentation**: `MobileModelRunner.cpp_docs.md`
- **Keyword Index**: `MobileModelRunner.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
