# Documentation: `binaries/aot_model_compiler.cc`

## File Metadata

- **Path**: `binaries/aot_model_compiler.cc`
- **Size**: 5,304 bytes (5.18 KB)
- **Type**: C++ Source Code
- **Extension**: `.cc`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <sstream>
#include <string>

#include <ATen/core/jit_type.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/backends/backend_detail.h>
#include <torch/csrc/jit/backends/backend_preprocess.h>
#include <torch/csrc/jit/mobile/nnc/aot_compiler.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/tensorexpr/graph_opt.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/script.h>

C10_DEFINE_string(model, "", "The torch script model to optimize.");
C10_DEFINE_string(model_name, "", "The name of the model.");
C10_DEFINE_string(model_version, "", "The version of the model.");
C10_DEFINE_string(
    input_dims,
    "",
    "The dimensions of input TensorCPUs using comma separated numbers."
    "If multiple inputs needed, use semicolon to separate "
    "the dimension of different tensors.");
C10_DEFINE_string(
    input_types,
    "float",
    "The dtype of input TensorCPUs."
    "If multiple inputs needed, use semicolon to separate "
    "the dtype of different tensors."
    "Supported dtypes: float, int64, uint8");
C10_DEFINE_string(
    input_memory_formats,
    "",
    "Input memory format."
    "If multiple inputs needed, use semicolon to separate."
    "Supported values: contiguous, channels_last");
C10_DEFINE_string(
    dynamic_dims,
    "",
    "Comma separated dimensions of input tensors that can be dynamic");
C10_DEFINE_string(method_name, "forward", "The name of the method.");
C10_DEFINE_string(
    output_llvm,
    "",
    "Name of the output llvm assembly to be saved.");
C10_DEFINE_string(output_model, "", "Name of the output model to be saved.");

namespace {

std::vector<std::string> split(
    char separator,
    const std::string& string,
    bool ignore_empty = true) {
  std::vector<std::string> pieces;
  std::stringstream ss(string);
  std::string item;
  while (getline(ss, item, separator)) {
    if (!ignore_empty || !item.empty()) {
      pieces.push_back(std::move(item));
    }
  }
  return pieces;
}

c10::Dict<c10::IValue, c10::IValue> createCompileSpec() {
  c10::Dict<c10::IValue, c10::IValue> compile_spec(
      c10::StringType::get(), c10::AnyType::get());
  c10::Dict<c10::IValue, c10::IValue> method_spec(
      c10::StringType::get(), c10::AnyType::get());
  method_spec.insert("sizes", FLAGS_input_dims);
  method_spec.insert("types", FLAGS_input_types);
  method_spec.insert("memory_formats", FLAGS_input_memory_formats);
  method_spec.insert("dynamic_sizes", FLAGS_dynamic_dims);
  method_spec.insert("asmfile", FLAGS_output_llvm);
  method_spec.insert("model_name", FLAGS_model_name);
  method_spec.insert("model_version", FLAGS_model_version);
  compile_spec.insert(FLAGS_method_name, method_spec);
  return compile_spec;
}

} // namespace

int main(int argc, char** argv) {
  c10::SetUsageMessage(
      "Run NNC AOT compiler for pytorch model. Example usage:\n"
      "build/bin/aot_model_compiler"
      " --model=<model file>"
      " --model_name=<model name>"
      " --model_version=<model version>"
      " --input_dims=<input dimensions like '1,3,224,224;2,2'>"
      " --input_types=<input dtypes like 'float;float'>"
      " --input_memory_formats=<input memory formats like 'channels_last;contiguous'>"
      " [--method_name=<method name>]"
      " [--output_llvm=<llvm assembly output file path>]"
      " [--output_model=<output model file path>]");

  if (!c10::ParseCommandLineFlags(&argc, &argv)) {
    std::cerr << "Failed to parse command line flags!" << std::endl;
    std::cout << c10::UsageMessage() << std::endl;
    return 1;
  }

  CAFFE_ENFORCE(!FLAGS_model.empty(), c10::UsageMessage());
  CAFFE_ENFORCE(!FLAGS_model_name.empty(), c10::UsageMessage());
  CAFFE_ENFORCE(!FLAGS_model_version.empty(), c10::UsageMessage());
  CAFFE_ENFORCE(!FLAGS_input_dims.empty(), c10::UsageMessage());
  const auto dims_size = split(';', FLAGS_input_dims).size();
  CAFFE_ENFORCE(
      dims_size == split(';', FLAGS_input_types).size(),
      "Number of input_dims and input_types should be the same");
  const auto mem_formats_size = split(';', FLAGS_input_memory_formats).size();
  CAFFE_ENFORCE(
      mem_formats_size == 0 || mem_formats_size == dims_size,
      "Number of input_memory_formats should be 0 (default contiguous) or the same as number of input_dims");
  if (FLAGS_output_llvm.empty()) {
    FLAGS_output_llvm =
        FLAGS_model.substr(0, FLAGS_model.find('.')) + ".compiled.ll";
  }

  std::string output_model_name = FLAGS_output_model;
  if (output_model_name.empty()) {
    output_model_name =
        FLAGS_model.substr(0, FLAGS_model.find('.')) + ".compiled.pt";
  }

  auto m = torch::jit::load(FLAGS_model);
  m.eval();
  auto frozen_m = torch::jit::freeze_module(m.clone());

  auto compile_spec = createCompileSpec();
  auto any_dict_ty =
      c10::DictType::create(c10::StringType::get(), c10::AnyType::get());
  auto compiled_module = torch::jit::detail::codegen_backend_module(
      "nnc", frozen_m, compile_spec, any_dict_ty);
  compiled_module._save_for_mobile(output_model_name);
  std::cout << "The compiled model was saved to " << output_model_name
            << std::endl;
  return 0;
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `int`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `binaries`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file includes:

- `sstream`
- `string`
- `ATen/core/jit_type.h`
- `c10/core/ScalarType.h`
- `torch/csrc/jit/backends/backend.h`
- `torch/csrc/jit/backends/backend_detail.h`
- `torch/csrc/jit/backends/backend_preprocess.h`
- `torch/csrc/jit/mobile/nnc/aot_compiler.h`
- `torch/csrc/jit/passes/freeze_module.h`
- `torch/csrc/jit/serialization/export.h`
- `torch/csrc/jit/serialization/import.h`
- `torch/csrc/jit/tensorexpr/graph_opt.h`
- `torch/csrc/jit/tensorexpr/kernel.h`
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

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`binaries`):

- [`parallel_info.cc_docs.md`](./parallel_info.cc_docs.md)
- [`at_launch_benchmark.cc_docs.md`](./at_launch_benchmark.cc_docs.md)
- [`speed_benchmark_torch.cc_docs.md`](./speed_benchmark_torch.cc_docs.md)
- [`load_benchmark_torch.cc_docs.md`](./load_benchmark_torch.cc_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`optimize_for_mobile.cc_docs.md`](./optimize_for_mobile.cc_docs.md)
- [`dump_operator_names.cc_docs.md`](./dump_operator_names.cc_docs.md)
- [`compare_models_torch.cc_docs.md`](./compare_models_torch.cc_docs.md)
- [`record_function_benchmark.cc_docs.md`](./record_function_benchmark.cc_docs.md)


## Cross-References

- **File Documentation**: `aot_model_compiler.cc_docs.md`
- **Keyword Index**: `aot_model_compiler.cc_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
