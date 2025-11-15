# Documentation: `binaries/optimize_for_mobile.cc`

## File Metadata

- **Path**: `binaries/optimize_for_mobile.cc`
- **Size**: 3,849 bytes (3.76 KB)
- **Type**: C++ Source Code
- **Extension**: `.cc`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string>
#include <sstream>
#include <torch/script.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/passes/metal_rewrite.h>
#include <torch/csrc/jit/passes/vulkan_rewrite.h>
#include <torch/csrc/jit/passes/xnnpack_rewrite.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/serialization/export.h>

C10_DEFINE_string(model, "", "The torch script model to optimize.");
C10_DEFINE_string(
    output,
    "",
    "Name of the output model to be saved.");
C10_DEFINE_string(backend, "", "The backend to be optimized");
C10_DEFINE_string(preserved_methods, "", "Methods to be preserved")

int main(int argc, char** argv) {
  c10::SetUsageMessage(
    "\nRun optimization pass for pytorch model. Example usage:\n"
    "./optimize_for_mobile"
    " --model=<model_file>"
    " [--output=<output_file_name>]"
    " [--backend=<cpu|vulkan|metal>]"
    " [--preserved_methods=<method_names>]"
  );

  if (!c10::ParseCommandLineFlags(&argc, &argv)) {
    std::cerr << "Failed to parse command line flags!" << std::endl;
    std::cout << c10::UsageMessage() << std::endl;
    return 1;
  }

  CAFFE_ENFORCE(FLAGS_model != "", c10::UsageMessage());

  std::string output_model_name =
    FLAGS_model.substr(0, FLAGS_model.find(".")) + "_optimized.ptl";

  if (FLAGS_output != "") {
    output_model_name = FLAGS_output;
  }

  std::vector<std::string> preserved_methods;
  if(FLAGS_preserved_methods != ""){
    std::stringstream ss(FLAGS_preserved_methods);
    std::string m;
    while(std::getline(ss, m, ';')){
      if(m != ""){
        preserved_methods.emplace_back(std::move(m));
      }
    }
    std::cout<<"The following methods will be preserved:"<<std::endl;
    for(auto& str : preserved_methods){
      std::cout<<str<<std::endl;
    }
  }

  auto module = torch::jit::load(FLAGS_model);
  auto ops = torch::jit::export_opnames(module);
  std::cout << "\npt_operator_library(" << std::endl;
  std::cout << "\tname = \"old_op_library\"," << std::endl;
  std::cout << "\tops = [" << std::endl;
  for (auto const& op: ops) {
    std::cout << "\t\t\"" << op << "\"," << std::endl;
  }
  std::cout << "\t],\n)\n" << std::endl;

  torch::jit::Module optimized_module;
  if (FLAGS_backend == "" || FLAGS_backend == "cpu") {
    optimized_module = torch::jit::optimizeForMobile(module);
  } else if (FLAGS_backend == "vulkan") {
    optimized_module = torch::jit::vulkanOptimizeForMobile(
        module, std::set<MobileOptimizerType>(), preserved_methods);
  } else if (FLAGS_backend == "metal"){
    optimized_module = torch::jit::metalOptimizeForMobile(module, preserved_methods);
  }else{
    CAFFE_ENFORCE(false, "Unknown backend: " + FLAGS_backend);
  }
  auto new_ops = torch::jit::export_opnames(optimized_module);
  std::cout << "\npt_operator_library(" << std::endl;
  std::cout << "\tname = \"new_op_library\"," << std::endl;
  std::cout << "\tops = [" << std::endl;
  for (auto const& op: new_ops) {
    std::cout << "\t\t\"" << op << "\"," << std::endl;
  }
  std::cout << "\t],\n)\n" << std::endl;
  optimized_module._save_for_mobile(output_model_name);
  std::cout << "The optimized model for lite interpreter was saved to " << output_model_name << std::endl;
  return 0;
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `binaries`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file includes:

- `string`
- `sstream`
- `torch/script.h`
- `torch/csrc/jit/api/module.h`
- `torch/csrc/jit/passes/metal_rewrite.h`
- `torch/csrc/jit/passes/vulkan_rewrite.h`
- `torch/csrc/jit/passes/xnnpack_rewrite.h`
- `torch/csrc/jit/serialization/import.h`
- `torch/csrc/jit/serialization/export.h`


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

Files in the same folder (`binaries`):

- [`parallel_info.cc_docs.md`](./parallel_info.cc_docs.md)
- [`at_launch_benchmark.cc_docs.md`](./at_launch_benchmark.cc_docs.md)
- [`speed_benchmark_torch.cc_docs.md`](./speed_benchmark_torch.cc_docs.md)
- [`aot_model_compiler.cc_docs.md`](./aot_model_compiler.cc_docs.md)
- [`load_benchmark_torch.cc_docs.md`](./load_benchmark_torch.cc_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`dump_operator_names.cc_docs.md`](./dump_operator_names.cc_docs.md)
- [`compare_models_torch.cc_docs.md`](./compare_models_torch.cc_docs.md)
- [`record_function_benchmark.cc_docs.md`](./record_function_benchmark.cc_docs.md)


## Cross-References

- **File Documentation**: `optimize_for_mobile.cc_docs.md`
- **Keyword Index**: `optimize_for_mobile.cc_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
