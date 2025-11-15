# Documentation: `binaries/record_function_benchmark.cc`

## File Metadata

- **Path**: `binaries/record_function_benchmark.cc`
- **Size**: 3,871 bytes (3.78 KB)
- **Type**: C++ Source Code
- **Extension**: `.cc`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp

#include <torch/torch.h>
#include <ATen/record_function.h>

#include "c10/util/Flags.h"

#include <chrono>
#include <iostream>
#include <ctime>

C10_DEFINE_int(iter, 10000, "Number of iterations");
C10_DEFINE_int(sampled_iter, 10e6,
    "Number of iterations for the sampled observer benchmark");

namespace {
const int kTensorSize = 16;
const int kSmallTensorSize = 1;
const float kLowSamplingProb = 0.0001;
}

void addTestCallback(
    double sampling_prob = 1.0,
    at::RecordFunctionCallback::StartCallback fn =
        [](const at::RecordFunction&) -> std::unique_ptr<at::ObserverContext> { return nullptr; }) {
  auto cb = at::RecordFunctionCallback(
      fn,
      [](const at::RecordFunction&, at::ObserverContext*) {})
    .needsInputs(false);
  if (sampling_prob < 1.0) {
    cb.samplingProb(sampling_prob);
  }
  at::addGlobalCallback(cb);
}

float runTensorGEMMBench(int tensor_size, int iter) {
  typedef std::chrono::high_resolution_clock clock;
  typedef std::chrono::microseconds us;
  std::chrono::time_point<clock> start_time = clock::now();
  auto inp = torch::randn({tensor_size, tensor_size});
  for (auto idx = 0; idx < iter; ++idx) {
    torch::mm(inp, inp);
  }
  auto duration = static_cast<float>(
      std::chrono::duration_cast<us>(clock::now() - start_time).count());
  return duration;
}

float runPureRecordFunctionBench(int iter) {
  typedef std::chrono::high_resolution_clock clock;
  typedef std::chrono::microseconds us;
  std::chrono::time_point<clock> start_time = clock::now();
  for (auto idx = 0; idx < iter; ++idx) {
    auto step_callbacks = at::getStepCallbacksUnlessEmpty(at::RecordScope::USER_SCOPE);
    if (step_callbacks.has_value()) {
      at::RecordFunction guard(std::move(*step_callbacks));
      guard.before("Test", -1);
    }
  }
  auto duration = static_cast<float>(
      std::chrono::duration_cast<us>(clock::now() - start_time).count());
  return duration;
}

void runBenchmark() {
  float duration = 0;
  for (auto tensor_size : std::set<int>({kSmallTensorSize, kTensorSize})) {
    duration = runTensorGEMMBench(tensor_size, FLAGS_iter);
    std::cout << "Tensor GEMM benchmark ("
              << tensor_size
              << "x"
              << tensor_size
              << ", " << FLAGS_iter << "): " << duration
              << " us." << std::endl;
  }
  duration = runPureRecordFunctionBench(FLAGS_iter);
  std::cout << "Pure RecordFunction benchmark ("
            << FLAGS_iter << "): "
            << duration
            << " us." << std::endl;
}

int main(int argc, char** argv) {
  if (!c10::ParseCommandLineFlags(&argc, &argv)) {
    std::cout << "Failed to parse command line flags" << std::endl;
    return -1;
  }

  at::enableRecordFunction();
  at::clearCallbacks();

  std::cout << "Warm up" << std::endl;
  runBenchmark();

  std::cout << "Running without observers" << std::endl;
  runBenchmark();

  addTestCallback();
  std::cout << "Running with empty non-sampled observer" << std::endl;
  runBenchmark();
  at::clearCallbacks();

  addTestCallback(kLowSamplingProb);
  std::cout << "Running with empty sampled observer" << std::endl;
  runBenchmark();
  at::clearCallbacks();

  std::cout << "Checking number of sampled observer invocations" << std::endl;
  static int cb_count = 0;
  addTestCallback(
      kLowSamplingProb,
      [](const at::RecordFunction&) -> std::unique_ptr<at::ObserverContext> {
        ++cb_count;
        return nullptr;
      }
  );

  auto duration = runPureRecordFunctionBench(FLAGS_sampled_iter);

  std::cout << "Pure RecordFunction runtime of " << FLAGS_sampled_iter
            << " iterations: " << duration
            << " us, number of callback invocations: " << cb_count
            << ", expected number: ~" << (int)(FLAGS_sampled_iter * kLowSamplingProb)
            << " invocations" << std::endl;

  at::clearCallbacks();
  return 0;
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `binaries`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file includes:

- `torch/torch.h`
- `ATen/record_function.h`
- `c10/util/Flags.h`
- `chrono`
- `iostream`
- `ctime`


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

Files in the same folder (`binaries`):

- [`parallel_info.cc_docs.md`](./parallel_info.cc_docs.md)
- [`at_launch_benchmark.cc_docs.md`](./at_launch_benchmark.cc_docs.md)
- [`speed_benchmark_torch.cc_docs.md`](./speed_benchmark_torch.cc_docs.md)
- [`aot_model_compiler.cc_docs.md`](./aot_model_compiler.cc_docs.md)
- [`load_benchmark_torch.cc_docs.md`](./load_benchmark_torch.cc_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`optimize_for_mobile.cc_docs.md`](./optimize_for_mobile.cc_docs.md)
- [`dump_operator_names.cc_docs.md`](./dump_operator_names.cc_docs.md)
- [`compare_models_torch.cc_docs.md`](./compare_models_torch.cc_docs.md)


## Cross-References

- **File Documentation**: `record_function_benchmark.cc_docs.md`
- **Keyword Index**: `record_function_benchmark.cc_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
