# Documentation: `binaries/dump_operator_names.cc`

## File Metadata

- **Path**: `binaries/dump_operator_names.cc`
- **Size**: 2,901 bytes (2.83 KB)
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

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <c10/util/Flags.h>

#include <fstream>

namespace torch {
namespace jit {
void dump_opnames(const Module& m, std::unordered_set<std::string>& opnames) {
  auto methods = m.get_methods();
  for (const auto& method : methods) {
    const auto& func = method.function();
    std::cout << "function name: " << func.name() << std::endl;
    auto graph = toGraphFunction(func).graph()->copy();
    torch::jit::Code code(graph, "");
    for (size_t i = 0; i < code.instructions().size(); ++i) {
      auto ins = code.instructions()[i];
      auto node = code.instructions_source()[i];
      if (ins.op == OpCode::OP) {
        auto opname = node->schema().operator_name();
        std::string namestr = opname.name;
        if (!opname.overload_name.empty())
          namestr += "." + opname.overload_name;
        std::cout << "    " << namestr << std::endl;
        opnames.emplace(namestr);
      }
    }
  }
  for (const auto& sub_m : m.children()) {
    std::cout << "sub module name: " << sub_m.type()->name()->qualifiedName() << std::endl;
    dump_opnames(sub_m, opnames);
  }
}
}
}

C10_DEFINE_string(model, "", "The given torch script model.");
C10_DEFINE_string(output, "", "The output yaml file of operator list.");

int main(int argc, char** argv) {
  c10::SetUsageMessage(
    "Dump operators in a script module and its sub modules.\n"
    "Example usage:\n"
    "./dump_operator_names"
    " --model=<model_file>"
    " --output=<output.yaml>");

  if (!c10::ParseCommandLineFlags(&argc, &argv)) {
    std::cerr << "Failed to parse command line flags!" << std::endl;
    return 1;
  }

  CAFFE_ENFORCE_GE(FLAGS_model.size(), 0, "Model file must be specified.");
  CAFFE_ENFORCE_GE(FLAGS_output.size(), 0, "Output yaml file must be specified.");

  auto m = torch::jit::load(FLAGS_model);
  std::unordered_set<std::string> opnames;
  torch::jit::dump_opnames(m, opnames);
  std::ofstream ofile(FLAGS_output);
  std::cout << "-- Final List --" << std::endl;
  for (const auto& name : opnames) {
    std::cout << name << std::endl;
    ofile << "- " << name << std::endl;
  }
  ofile.close();
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `jit`, `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `binaries`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/api/module.h`
- `torch/csrc/jit/mobile/module.h`
- `torch/csrc/jit/serialization/import.h`
- `torch/csrc/jit/runtime/instruction.h`
- `c10/util/Flags.h`
- `fstream`


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
- [`optimize_for_mobile.cc_docs.md`](./optimize_for_mobile.cc_docs.md)
- [`compare_models_torch.cc_docs.md`](./compare_models_torch.cc_docs.md)
- [`record_function_benchmark.cc_docs.md`](./record_function_benchmark.cc_docs.md)


## Cross-References

- **File Documentation**: `dump_operator_names.cc_docs.md`
- **Keyword Index**: `dump_operator_names.cc_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
