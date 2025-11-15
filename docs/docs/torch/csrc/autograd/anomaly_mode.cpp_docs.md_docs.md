# Documentation: `docs/torch/csrc/autograd/anomaly_mode.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/autograd/anomaly_mode.cpp_docs.md`
- **Size**: 4,769 bytes (4.66 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/autograd/anomaly_mode.cpp`

## File Metadata

- **Path**: `torch/csrc/autograd/anomaly_mode.cpp`
- **Size**: 2,281 bytes (2.23 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/util/Backtrace.h>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/anomaly_mode.h>
#include <torch/csrc/autograd/function.h>
#include <mutex>

namespace torch::autograd {

bool AnomalyMode::_enabled = false;
bool AnomalyMode::_check_nan = true;

namespace {
std::mutex& get_anomaly_guard_lock() {
  static std::mutex anomaly_guard_lock{};
  return anomaly_guard_lock;
}

uint32_t& get_anomaly_counter() {
  static uint32_t counter = 0;
  return counter;
}
} // namespace

DetectAnomalyGuard::DetectAnomalyGuard(bool check_nan) {
  TORCH_WARN_ONCE(
      "This mode should be enabled only for debugging as the different tests will slow down your program execution.");
  std::lock_guard<std::mutex> lock(get_anomaly_guard_lock());
  uint32_t& counter = get_anomaly_counter();
  counter++;
  this->prev_check_nan_ = AnomalyMode::should_check_nan();
  AnomalyMode::set_enabled(true, check_nan);
}

DetectAnomalyGuard::~DetectAnomalyGuard() {
  std::lock_guard<std::mutex> lock(get_anomaly_guard_lock());
  uint32_t& counter = get_anomaly_counter();
  counter--;
  AnomalyMode::set_enabled(counter > 0, this->prev_check_nan_);
}

AnomalyMetadata::~AnomalyMetadata() = default;

void AnomalyMetadata::store_stack() {
  traceback_ = c10::get_backtrace(/* frames_to_skip */ 1);
}

void AnomalyMetadata::print_stack(const std::string& current_node_name) {
  TORCH_WARN(
      "Error detected in ",
      current_node_name,
      ". ",
      "Traceback of forward call that caused the error:\n",
      traceback_);

  auto& cur_parent = parent_;
  // if there is no "parent_" in metadata, then it means this metadata's node
  // is the root and stop printing the traceback
  while (cur_parent) {
    auto parent_metadata = cur_parent->metadata();
    TORCH_WARN(
        "\n\n",
        "Previous calculation was induced by ",
        cur_parent->name(),
        ". "
        "Traceback of forward call that induced the previous calculation:\n",
        parent_metadata->traceback_);
    // get the parent of this node, if this node is a root, pyparent is simply
    // null
    cur_parent = parent_metadata->parent_;
  }
}

void AnomalyMetadata::assign_parent(const std::shared_ptr<Node>& parent_node) {
  parent_ = parent_node;
}

} // namespace torch::autograd

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 1 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `DetectAnomalyGuard`, `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/autograd`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/Backtrace.h`
- `c10/util/Exception.h`
- `torch/csrc/autograd/anomaly_mode.h`
- `torch/csrc/autograd/function.h`
- `mutex`


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

Files in the same folder (`torch/csrc/autograd`):

- [`graph_task.h_docs.md`](./graph_task.h_docs.md)
- [`python_function.cpp_docs.md`](./python_function.cpp_docs.md)
- [`profiler.h_docs.md`](./profiler.h_docs.md)
- [`TraceTypeManual.cpp_docs.md`](./TraceTypeManual.cpp_docs.md)
- [`python_autograd.h_docs.md`](./python_autograd.h_docs.md)
- [`variable_info.cpp_docs.md`](./variable_info.cpp_docs.md)
- [`jit_decomp_interface.h_docs.md`](./jit_decomp_interface.h_docs.md)
- [`input_buffer.cpp_docs.md`](./input_buffer.cpp_docs.md)
- [`python_variable.h_docs.md`](./python_variable.h_docs.md)
- [`python_nn_functions.h_docs.md`](./python_nn_functions.h_docs.md)


## Cross-References

- **File Documentation**: `anomaly_mode.cpp_docs.md`
- **Keyword Index**: `anomaly_mode.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/autograd`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/autograd`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


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

Files in the same folder (`docs/torch/csrc/autograd`):

- [`python_cpp_function.h_kw.md_docs.md`](./python_cpp_function.h_kw.md_docs.md)
- [`anomaly_mode.cpp_kw.md_docs.md`](./anomaly_mode.cpp_kw.md_docs.md)
- [`python_nested_functions_manual.cpp_kw.md_docs.md`](./python_nested_functions_manual.cpp_kw.md_docs.md)
- [`variable_info.h_docs.md_docs.md`](./variable_info.h_docs.md_docs.md)
- [`python_nn_functions.h_docs.md_docs.md`](./python_nn_functions.h_docs.md_docs.md)
- [`python_cpp_function.h_docs.md_docs.md`](./python_cpp_function.h_docs.md_docs.md)
- [`profiler_legacy.cpp_kw.md_docs.md`](./profiler_legacy.cpp_kw.md_docs.md)
- [`saved_variable.cpp_docs.md_docs.md`](./saved_variable.cpp_docs.md_docs.md)
- [`python_fft_functions.h_docs.md_docs.md`](./python_fft_functions.h_docs.md_docs.md)
- [`python_autograd.h_kw.md_docs.md`](./python_autograd.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `anomaly_mode.cpp_docs.md_docs.md`
- **Keyword Index**: `anomaly_mode.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
