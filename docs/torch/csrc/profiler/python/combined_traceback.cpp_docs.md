# Documentation: `torch/csrc/profiler/python/combined_traceback.cpp`

## File Metadata

- **Path**: `torch/csrc/profiler/python/combined_traceback.cpp`
- **Size**: 7,116 bytes (6.95 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/profiler/python/combined_traceback.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pythoncapi_compat.h>
namespace py = pybind11;

namespace torch {
// Locking:
// We need to free PyCodeObjects when ~StackContext runs, but
// CUDACachingAllocator may hold its device lock when ~StackContext runs.

// Because the thread calling the allocator _may_ hold the GIL,
// attempting to lock the GIL in ~StackContext can deadlock:
// T0: GIL Lock -> Call Allocator    ->| Waiting Device Lock
// T1: Call Allocator -> Device Lock ->| Waiting GIL Lock
// Instead the destructor defers freeing stack frames by putting them in
// to_free_frames. We still need a lock to manage this vector, but
// we can ensure an overall lock ordering of GIL -> device_lock ->
// to_free_frames_mutex because ::gather is called outside of the device lock.

namespace {
static std::mutex to_free_frames_mutex;
static std::vector<CapturedTraceback::PyFrame> to_free_frames;
struct PythonTraceback : public CapturedTraceback::Python {
  std::vector<CapturedTraceback::PyFrame> gather() override {
    if (!Py_IsInitialized()) {
      return {};
    }
    std::vector<CapturedTraceback::PyFrame> frames;
    py::gil_scoped_acquire acquire;
    {
      std::lock_guard<std::mutex> lock(to_free_frames_mutex);
      for (CapturedTraceback::PyFrame f : to_free_frames) {
        Py_XDECREF(f.code);
      }
      to_free_frames.clear();
    }
    PyFrameObject* f = PyEval_GetFrame();
    Py_XINCREF(f);
    while (f) {
      frames.emplace_back(
          CapturedTraceback::PyFrame{PyFrame_GetCode(f), PyFrame_GetLasti(f)});
      auto f_back = PyFrame_GetBack(f);
      Py_XDECREF(f);
      f = f_back;
    }
    return frames;
  }
  void release(std::vector<CapturedTraceback::PyFrame>& frames) override {
    std::lock_guard<std::mutex> lock(to_free_frames_mutex);
    to_free_frames.insert(to_free_frames.end(), frames.begin(), frames.end());
  }
  using void_visitproc = int (*)(void* self, void* arg);
  int traverse(
      std::vector<CapturedTraceback::PyFrame>& frames,
      void_visitproc visit,
      void* arg) override {
    for (auto& f : frames) {
      Py_VISIT(f.code);
    }
    return 0;
  }
  int clear(std::vector<CapturedTraceback::PyFrame>& frames) override {
    for (auto& f : frames) {
      Py_CLEAR(f.code);
    }
    return 0;
  }
  void appendSymbolized(
      const std::vector<CapturedTraceback::PyFrame>& to_symbolize,
      SymbolizedTracebacks& result) override {
    py::gil_scoped_acquire acquire;
    py::str line_s = "line";
    py::str name_s = "name";
    py::str filename_s = "filename";

    auto torch = py::module::import("torch");
    py::object stack_frames_for_code;
    if (py::hasattr(torch, "_inductor")) {
      py::object inductor = torch.attr("_inductor");
      if (py::hasattr(inductor, "codecache")) {
        stack_frames_for_code = inductor.attr("codecache")
                                    .attr("PyCodeCache")
                                    .attr("stack_frames_for_code");
      }
    }
    for (const auto& f : to_symbolize) {
      auto f_code = (PyCodeObject*)f.code;
      py::handle filename = f_code->co_filename;
      py::handle funcname = f_code->co_name;
      auto lineno = PyCode_Addr2Line(f_code, f.lasti);
      result.tracebacks.emplace_back();
      result.tracebacks.back().push_back(result.all_frames.size());
      result.all_frames.emplace_back(unwind::Frame{
          py::cast<std::string>(filename),
          py::cast<std::string>(funcname),
          (uint64_t)lineno});
      // find all the additional frames associated with inductor generated
      // code
      if (stack_frames_for_code.ptr()) {
        py::object extra = stack_frames_for_code(filename, lineno);
        if (!extra.is_none()) {
          for (py::handle h : extra) {
            result.tracebacks.back().push_back(result.all_frames.size());
            result.all_frames.emplace_back(unwind::Frame{
                py::cast<std::string>(h[filename_s]),
                py::cast<std::string>(h[name_s]),
                py::cast<uint64_t>(h[line_s])});
          }
        }
      }
    }
  }
};

} // namespace

std::vector<nlohmann::json> json_symbolize(
    std::vector<CapturedTraceback*>& to_symbolize) {
  std::unordered_map<CapturedTraceback*, uint64_t> cached_frames;
  std::vector<CapturedTraceback*> unique_frames;
  for (const auto& sc : to_symbolize) {
    auto it = cached_frames.find(sc);
    if (it == cached_frames.end()) {
      cached_frames.try_emplace(sc, unique_frames.size());
      unique_frames.push_back(sc);
    }
  }
  auto s = symbolize(unique_frames);

  std::string line_s = "line";
  std::string name_s = "name";
  std::string filename_s = "filename";
  std::vector<nlohmann::json> all_frames;

  for (const auto& f : s.all_frames) {
    nlohmann::json d;
    d[name_s] = f.funcname;
    d[filename_s] = f.filename;
    d[line_s] = f.lineno;
    all_frames.emplace_back(std::move(d));
  }

  std::vector<nlohmann::json> py_unique_frames;
  for (const auto& t : s.tracebacks) {
    nlohmann::json l;
    for (const auto& e : t) {
      l.emplace_back(all_frames.at(e));
    }
    py_unique_frames.push_back(std::move(l));
  }

  std::vector<nlohmann::json> result;
  result.reserve(to_symbolize.size());
  for (const auto& sc : to_symbolize) {
    result.push_back(py_unique_frames.at(cached_frames.at(sc)));
  }
  return result;
}

std::vector<py::object> py_symbolize(
    std::vector<CapturedTraceback*>& to_symbolize) {
  // we dedup repeated to_symbolize objects to prevent
  // creating a bunch of duplicated frame objects
  std::unordered_map<CapturedTraceback*, uint64_t> cached_frames;
  std::vector<CapturedTraceback*> unique_frames;
  for (const auto& sc : to_symbolize) {
    auto it = cached_frames.find(sc);
    if (it == cached_frames.end()) {
      cached_frames.insert({sc, unique_frames.size()});
      unique_frames.push_back(sc);
    }
  }
  auto s = symbolize(unique_frames);

  py::str line_s = "line";
  py::str name_s = "name";
  py::str filename_s = "filename";
  std::vector<py::dict> all_frames;
  for (const auto& f : s.all_frames) {
    py::dict d;
    d[name_s] = f.funcname;
    d[filename_s] = f.filename;
    d[line_s] = f.lineno;
    all_frames.emplace_back(std::move(d));
  }

  std::vector<py::object> py_unique_frames;
  for (const auto& t : s.tracebacks) {
    py::list l;
    for (const auto& e : t) {
      l.append(all_frames.at(e));
    }
    py_unique_frames.push_back(std::move(l));
  }

  std::vector<py::object> result;
  result.reserve(to_symbolize.size());
  for (const auto& sc : to_symbolize) {
    result.push_back(py_unique_frames.at(cached_frames.at(sc)));
  }
  return result;
}

void freeDeadCapturedTracebackFrames() {
  std::lock_guard<std::mutex> lock(to_free_frames_mutex);
  for (CapturedTraceback::PyFrame f : to_free_frames) {
    Py_XDECREF(f.code);
  }
  to_free_frames.clear();
}

void installCapturedTracebackPython() {
  CapturedTraceback::addPythonUnwinder(new PythonTraceback());
}

} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `std`, `torch`, `py`

**Classes/Structs**: `PythonTraceback`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/profiler/python`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/profiler/python/combined_traceback.h`
- `torch/csrc/python_headers.h`
- `torch/csrc/utils/pybind.h`
- `torch/csrc/utils/pythoncapi_compat.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.

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

Files in the same folder (`torch/csrc/profiler/python`):

- [`init.h_docs.md`](./init.h_docs.md)
- [`combined_traceback.h_docs.md`](./combined_traceback.h_docs.md)
- [`init.cpp_docs.md`](./init.cpp_docs.md)
- [`pybind.h_docs.md`](./pybind.h_docs.md)


## Cross-References

- **File Documentation**: `combined_traceback.cpp_docs.md`
- **Keyword Index**: `combined_traceback.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
