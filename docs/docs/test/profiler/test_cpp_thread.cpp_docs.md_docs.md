# Documentation: `docs/test/profiler/test_cpp_thread.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/profiler/test_cpp_thread.cpp_docs.md`
- **Size**: 6,378 bytes (6.23 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/profiler/test_cpp_thread.cpp`

## File Metadata

- **Path**: `test/profiler/test_cpp_thread.cpp`
- **Size**: 3,753 bytes (3.67 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp

#include <torch/csrc/autograd/profiler_kineto.h>  // @manual
#include <torch/torch.h>
#include <string>

using namespace torch::autograd::profiler;

void blueprint(const std::string& text) {
  printf("\33[94m%s\33[0m\n", text.c_str());
}

/**
 * We're emulating a C++ training engine calling into Python to allow Python
 * code controlling how profiling should be done.
 */
class ProfilerEventHandler
    : public std::enable_shared_from_this<ProfilerEventHandler> {
 public:
  static std::shared_ptr<ProfilerEventHandler> Handler;
  static void Register(const std::shared_ptr<ProfilerEventHandler>& handler) {
    Handler = handler;
  }

 public:
  virtual ~ProfilerEventHandler() {}
  virtual void onIterationStart(int) {}
  virtual void emulateTraining(int, int) {}
};
std::shared_ptr<ProfilerEventHandler> ProfilerEventHandler::Handler;

class ProfilerEventHandlerTrampoline : public ProfilerEventHandler {
 public:
  virtual void onIterationStart(int iteration) override {
    PYBIND11_OVERRIDE(void, ProfilerEventHandler, onIterationStart, iteration);
  }
  virtual void emulateTraining(int iteration, int thread_id) override {
    PYBIND11_OVERRIDE(
        void, ProfilerEventHandler, emulateTraining, iteration, thread_id);
  }
};

/**
 * This is the entry point for the C++ training engine.
 */
void start_threads(int thread_count, int iteration_count, bool attach) {
  blueprint("start_cpp_threads called");

  static std::atomic<int> barrier = 0;
  barrier = 0;
  static std::atomic<int> another_barrier = 0;
  another_barrier = 0;
  thread_local bool enabled_in_main_thread = false;

  std::vector<std::thread> threads;
  for (int id = 0; id < thread_count; id++) {
    blueprint("starting thread " + std::to_string(id));
    threads.emplace_back([thread_count, iteration_count, id, attach]() {
      for (int iteration = 0; iteration < iteration_count; iteration++) {
        if (id == 0) {
          ProfilerEventHandler::Handler->onIterationStart(iteration);
        }

        // this barrier makes sure all child threads will be turned on
        // with profiling when main thread is enabled
        ++barrier;
        while (barrier % thread_count) {
          std::this_thread::yield();
        }

        if (id > 0 && attach) {
          bool enabled = isProfilerEnabledInMainThread();
          if (enabled != enabled_in_main_thread) {
            if (enabled) {
              enableProfilerInChildThread();
            } else {
              disableProfilerInChildThread();
            }
            enabled_in_main_thread = enabled;
          }
        }

        ProfilerEventHandler::Handler->emulateTraining(iteration, id);

        // We need another barrier here to ensure that the main thread doesn't
        // stop the profiler while other threads are still using it. This fixes
        // https://github.com/pytorch/pytorch/issues/132331
        ++another_barrier;
        while (another_barrier % thread_count) {
          std::this_thread::yield();
        }
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }
}

PYBIND11_MODULE(profiler_test_cpp_thread_lib, m) {
  py::class_<
      ProfilerEventHandler,
      ProfilerEventHandlerTrampoline,
      std::shared_ptr<ProfilerEventHandler>>(m, "ProfilerEventHandler")
      .def(py::init<>())
      .def_static("Register", &ProfilerEventHandler::Register)
      .def(
          "onIterationStart",
          &ProfilerEventHandler::onIterationStart,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "emulateTraining",
          &ProfilerEventHandler::emulateTraining,
          py::call_guard<py::gil_scoped_release>());

  m.def(
      "start_threads",
      &start_threads,
      py::call_guard<py::gil_scoped_release>());
};

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `ProfilerEventHandler`, `ProfilerEventHandlerTrampoline`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/profiler`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/autograd/profiler_kineto.h`
- `torch/torch.h`
- `string`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/profiler/test_cpp_thread.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/profiler`):

- [`profiler_utils_mock_events.json_docs.md`](./profiler_utils_mock_events.json_docs.md)
- [`test_memory_profiler.py_docs.md`](./test_memory_profiler.py_docs.md)
- [`test_execution_trace.py_docs.md`](./test_execution_trace.py_docs.md)
- [`test_python_tracer.py_docs.md`](./test_python_tracer.py_docs.md)
- [`test_record_function.py_docs.md`](./test_record_function.py_docs.md)
- [`test_torch_tidy.py_docs.md`](./test_torch_tidy.py_docs.md)
- [`test_cpp_thread_lib.pyi_docs.md`](./test_cpp_thread_lib.pyi_docs.md)
- [`test_profiler_tree.py_docs.md`](./test_profiler_tree.py_docs.md)
- [`test_cpp_thread.py_docs.md`](./test_cpp_thread.py_docs.md)


## Cross-References

- **File Documentation**: `test_cpp_thread.cpp_docs.md`
- **Keyword Index**: `test_cpp_thread.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/profiler`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/profiler`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/profiler/test_cpp_thread.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/profiler`):

- [`test_record_function.py_kw.md_docs.md`](./test_record_function.py_kw.md_docs.md)
- [`profiler_utils_mock_events.json_docs.md_docs.md`](./profiler_utils_mock_events.json_docs.md_docs.md)
- [`test_profiler.py_kw.md_docs.md`](./test_profiler.py_kw.md_docs.md)
- [`test_torch_tidy.py_kw.md_docs.md`](./test_torch_tidy.py_kw.md_docs.md)
- [`test_memory_profiler.py_kw.md_docs.md`](./test_memory_profiler.py_kw.md_docs.md)
- [`test_profiler_tree.py_docs.md_docs.md`](./test_profiler_tree.py_docs.md_docs.md)
- [`test_kineto.py_docs.md_docs.md`](./test_kineto.py_docs.md_docs.md)
- [`test_execution_trace.py_kw.md_docs.md`](./test_execution_trace.py_kw.md_docs.md)
- [`test_cpp_thread.py_kw.md_docs.md`](./test_cpp_thread.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_cpp_thread.cpp_docs.md_docs.md`
- **Keyword Index**: `test_cpp_thread.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
