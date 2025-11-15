# Documentation: `binaries/load_benchmark_torch.cc`

## File Metadata

- **Path**: `binaries/load_benchmark_torch.cc`
- **Size**: 2,815 bytes (2.75 KB)
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
#include <vector>

#include <ATen/ATen.h>
#include "caffe2/core/timer.h"
#include "caffe2/utils/string_utils.h"
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/script.h>

#include <c10/mobile/CPUCachingAllocator.h>

#include <chrono>
using namespace std::chrono;

C10_DEFINE_string(model, "", "The given torch script model to benchmark.");
C10_DEFINE_int(iter, 10, "The number of iterations to run.");
C10_DEFINE_bool(
  report_pep,
  true,
  "Whether to print performance stats for AI-PEP.");

int main(int argc, char** argv) {
  c10::SetUsageMessage(
    "Run model load time benchmark for pytorch model.\n"
    "Example usage:\n"
    "./load_benchmark_torch"
    " --model=<model_file>"
    " --iter=20");
  if (!c10::ParseCommandLineFlags(&argc, &argv)) {
    std::cerr << "Failed to parse command line flags!" << std::endl;
    return 1;
  }

  std::cout << "Starting benchmark." << std::endl;
  CAFFE_ENFORCE(
      FLAGS_iter >= 0,
      "Number of main runs should be non negative, provided ",
      FLAGS_iter,
      ".");

  caffe2::Timer timer;
  std::vector<long> times;

  for (int i = 0; i < FLAGS_iter; ++i) {
    auto start = high_resolution_clock::now();

#if BUILD_LITE_INTERPRETER
    auto module = torch::jit::_load_for_mobile(FLAGS_model);
#else
    auto module = torch::jit::load(FLAGS_model);
#endif

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    times.push_back(duration.count());
  }

  const double micros = static_cast<double>(timer.MicroSeconds());
  if (FLAGS_report_pep) {
    for (auto t : times) {
      std::cout << R"(PyTorchObserver {"type": "NET", "unit": "us", )"
                << R"("metric": "latency", "value": ")"
                << t << R"("})" << std::endl;
    }
  }

  const double iters = static_cast<double>(FLAGS_iter);
  std::cout << "Main run finished. Microseconds per iter: "
            << micros / iters
            << ". Iters per second: " << 1000.0 * 1000 * iters / micros
            << std::endl;

  return 0;
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 1 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `std`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `binaries`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file includes:

- `string`
- `vector`
- `ATen/ATen.h`
- `caffe2/core/timer.h`
- `caffe2/utils/string_utils.h`
- `torch/csrc/autograd/grad_mode.h`
- `torch/csrc/jit/mobile/module.h`
- `torch/csrc/jit/mobile/import.h`
- `torch/csrc/jit/serialization/import.h`
- `torch/script.h`
- `c10/mobile/CPUCachingAllocator.h`
- `chrono`


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

Files in the same folder (`binaries`):

- [`parallel_info.cc_docs.md`](./parallel_info.cc_docs.md)
- [`at_launch_benchmark.cc_docs.md`](./at_launch_benchmark.cc_docs.md)
- [`speed_benchmark_torch.cc_docs.md`](./speed_benchmark_torch.cc_docs.md)
- [`aot_model_compiler.cc_docs.md`](./aot_model_compiler.cc_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`optimize_for_mobile.cc_docs.md`](./optimize_for_mobile.cc_docs.md)
- [`dump_operator_names.cc_docs.md`](./dump_operator_names.cc_docs.md)
- [`compare_models_torch.cc_docs.md`](./compare_models_torch.cc_docs.md)
- [`record_function_benchmark.cc_docs.md`](./record_function_benchmark.cc_docs.md)


## Cross-References

- **File Documentation**: `load_benchmark_torch.cc_docs.md`
- **Keyword Index**: `load_benchmark_torch.cc_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
