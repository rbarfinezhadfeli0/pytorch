# Documentation: `docs/torch/csrc/DataLoader.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/DataLoader.cpp_docs.md`
- **Size**: 11,679 bytes (11.41 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/DataLoader.cpp`

## File Metadata

- **Path**: `torch/csrc/DataLoader.cpp`
- **Size**: 9,263 bytes (9.05 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/DataLoader.h>

// Together with `torch/utils/data/_utils/signal_handling.py`, the following
// is an effort to do our best to provide some error message to users when a
// worker dies due to error / critical signals.
//
// See NOTE [ Signal handling in multiprocessing data loading ] for more
// details.

// TODO: The following don't work on Windows. Specifically, sigaction, waitid
// calls, and SIGCHLD handler. Currently, dummy implementations are provided
// for Windows.

#ifndef _WIN32

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/python_numbers.h>

#include <c10/util/irange.h>
#include <fmt/format.h>

#include <sys/wait.h>
#include <csignal>
#include <map>
#include <set>
#include <sstream>

using namespace torch;

// Critical signal handlers should be registered on worker processes before
// doing work.
// The handler will raise default handler so that the kill information will be
// retrieved from main process.
// Python handle is _set_worker_signal_handlers().
#define SIGNAL_HANDLER(SIGNAL, HANDLER_NAME, ERROR_MSG)                    \
  static void HANDLER_NAME(int sig, siginfo_t* info, void* ctx) {          \
    auto _w =                                                              \
        write(STDERR_FILENO, ERROR_MSG, sizeof(ERROR_MSG) / sizeof(char)); \
    (void)_w;                                                              \
    struct sigaction sa{};                                                 \
    sa.sa_handler = SIG_DFL;                                               \
    sa.sa_flags = 0;                                                       \
    if (sigemptyset(&sa.sa_mask) != 0 ||                                   \
        sigaction(SIGNAL, &sa, nullptr) != 0) {                            \
      _exit(EXIT_FAILURE);                                                 \
    } else {                                                               \
      raise(SIGNAL);                                                       \
    }                                                                      \
  }

// signal(2) is really not portable. So use sigaction.
// http://man7.org/linux/man-pages/man2/signal.2.html
static void setSignalHandler(
    int signal,
    void (*handler)(int, siginfo_t*, void*),
    struct sigaction* old_sa_ptr) {
  struct sigaction sa{};
  sa.sa_sigaction = handler;
  sa.sa_flags = SA_RESTART | SA_SIGINFO | SA_NOCLDSTOP | SA_NODEFER;
  if (sigemptyset(&sa.sa_mask) != 0 ||
      sigaction(signal, &sa, old_sa_ptr) != 0) {
    std::ostringstream oss;
    oss << "An error occurred while setting handler for " << strsignal(signal)
        << ".";
    TORCH_CHECK(false, oss.str());
  }
}

SIGNAL_HANDLER(
    SIGBUS,
    handler_SIGBUS,
    "ERROR: Unexpected bus error encountered in worker. "
    "This might be caused by insufficient shared memory (shm).\n")
SIGNAL_HANDLER(
    SIGSEGV,
    handler_SIGSEGV,
    "ERROR: Unexpected segmentation fault encountered in worker.\n")
SIGNAL_HANDLER(
    SIGFPE,
    handler_SIGFPE,
    "ERROR: Unexpected floating-point exception encountered in worker.\n")

// When an error happened in DataLoader methods and Python starts to exit, the
// error trace will keep the loader alive, and Python may kill the children
// processes first before deleting the loader object. Then the cleaning up
// methods in DataLoader.__del__ are not yet called, and SIGCHILD will print an
// error saying a worker is killed by SIGTERM. So we suppress SIGTERM from main
// loader process here to avoid this by _exit(EXIT_SUCCESS). Note that if we
// exit with nonzero code, the loader SIGCHLD handler may report RuntimeError
// again, and then it defeats the whole purpose.
static void handler_SIGTERM(int sig, siginfo_t* info, void* ctx) {
  if (info->si_pid == getppid()) {
    _exit(EXIT_SUCCESS);
  }
  struct sigaction sa{};
  sa.sa_handler = SIG_DFL;
  sa.sa_flags = 0;
  if (sigemptyset(&sa.sa_mask) != 0 || sigaction(SIGTERM, &sa, nullptr) != 0) {
    _exit(EXIT_FAILURE);
  } else {
    raise(SIGTERM);
  }
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
__attribute__((weak)) void setDataLoaderSignalHandlers() {}

static PyObject* THPModule_setWorkerSignalHandlers(
    PyObject* module,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  setSignalHandler(SIGBUS, &handler_SIGBUS, nullptr);
  setSignalHandler(SIGSEGV, &handler_SIGSEGV, nullptr);
  setSignalHandler(SIGTERM, &handler_SIGTERM, nullptr);
  setSignalHandler(SIGFPE, &handler_SIGFPE, nullptr);
  setDataLoaderSignalHandlers();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static std::map<int64_t, std::set<pid_t>> worker_pids = {};

static PyObject* THPModule_errorIfAnyWorkerFails(
    PyObject* module,
    PyObject* noargs) {
  HANDLE_TH_ERRORS

  // Only check the pids we care about
  for (auto& w : worker_pids) {
    auto& pid_set = w.second;
    for (auto worker_pid : pid_set) {
      // Use waitid rather than waitpid so that we can set NOWAIT, and that
      // Python and other handlers can get whatever info they want about the
      // child.
      siginfo_t infop{};
      infop.si_pid = 0;
      auto error =
          waitid(P_PID, worker_pid, &infop, WEXITED | WNOHANG | WNOWAIT);
      // ignore errors and case with no waitable child
      if (error < 0 || infop.si_pid == 0)
        continue;
      if (infop.si_code == CLD_EXITED &&
          infop.si_status != EXIT_SUCCESS) { // exit with error
        auto error_msg = fmt::format(
            "DataLoader worker (pid {}) exited unexpectedly with exit code {}. "
            "Details are lost due to multiprocessing. Rerunning with "
            "num_workers=0 may give better error trace.",
            worker_pid,
            infop.si_status);
        // This is necessary. Otherwise, the runtime error will kill the other
        // workers, and trigger this again.
        pid_set.clear();
        TORCH_CHECK(false, error_msg);
      } else if (
          infop.si_code == CLD_KILLED ||
          infop.si_code == CLD_DUMPED) { // killed by signal
        auto error_msg = fmt::format(
            "DataLoader worker (pid {}) is killed by signal: {}. ",
            worker_pid,
            strsignal(infop.si_status));
        if (infop.si_status == SIGBUS) {
          error_msg +=
              "It is possible that dataloader's workers are out of shared memory. "
              "Please try to raise your shared memory limit.";
        }
        // This is necessary. Otherwise, the runtime error will kill the other
        // workers, and trigger this again.
        pid_set.clear();
        TORCH_CHECK(false, error_msg);
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// We don't want to exit on any SIGCHLD from any child. child_pids is a tuple
// of pids we are interested in.
static PyObject* THPModule_setWorkerPIDs(PyObject* module, PyObject* args) {
  HANDLE_TH_ERRORS
  TORCH_CHECK_TYPE(
      PyTuple_GET_SIZE(args) == 2,
      "_set_worker_pids expects exactly 2 arguments.");
  int64_t key = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0));
  TORCH_CHECK_VALUE(
      worker_pids.find(key) == worker_pids.end(),
      "_set_worker_pids should be called only once for each _BaseDataLoaderIter.");
  PyObject* child_pids = PyTuple_GET_ITEM(args, 1);
  TORCH_CHECK_TYPE(
      PyTuple_Check(child_pids),
      "_set_worker_pids expects a tuple for child_pids, but got ",
      Py_TYPE(child_pids)->tp_name,
      ".");
  std::set<pid_t> pids_set = {};
  auto size = PyTuple_GET_SIZE(child_pids);
  for (const auto idx : c10::irange(size)) {
    PyObject* obj = PyTuple_GET_ITEM(child_pids, idx);
    pids_set.insert(static_cast<pid_t>(THPUtils_unpackLong(obj)));
  }

  worker_pids[key] = pids_set;

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_removeWorkerPIDs(
    PyObject* module,
    PyObject* loader_id) {
  HANDLE_TH_ERRORS

  int64_t key = THPUtils_unpackLong(loader_id);
  auto it = worker_pids.find(key);
  TORCH_CHECK_VALUE(
      it != worker_pids.end(),
      "Cannot find worker information for _BaseDataLoaderIter with id ",
      key);
  worker_pids.erase(it);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

#undef SIGNAL_HANDLER

#else
// dummy implementations for windows

static PyObject* THPModule_setWorkerSignalHandlers(
    PyObject* module,
    PyObject* _ignored) {
  Py_RETURN_NONE;
}

static PyObject* THPModule_setWorkerPIDs(PyObject* module, PyObject* _ignored) {
  Py_RETURN_NONE;
}

static PyObject* THPModule_removeWorkerPIDs(
    PyObject* module,
    PyObject* _ignored) {
  Py_RETURN_NONE;
}

static PyObject* THPModule_errorIfAnyWorkerFails(
    PyObject* module,
    PyObject* _ignored) {
  Py_RETURN_NONE;
}

#endif

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays)
PyMethodDef DataLoaderMethods[] = {
    {"_set_worker_signal_handlers",
     THPModule_setWorkerSignalHandlers,
     METH_NOARGS,
     nullptr},
    {"_set_worker_pids", THPModule_setWorkerPIDs, METH_VARARGS, nullptr},
    {"_remove_worker_pids", THPModule_removeWorkerPIDs, METH_O, nullptr},
    {"_error_if_any_worker_fails",
     THPModule_errorIfAnyWorkerFails,
     METH_NOARGS,
     nullptr},
    {nullptr, nullptr, 0, nullptr}};

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 15 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `sigaction`, `sigaction`, `sigaction`, `sigaction`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/DataLoader.h`
- `torch/csrc/Exceptions.h`
- `torch/csrc/utils/python_numbers.h`
- `c10/util/irange.h`
- `fmt/format.h`
- `sys/wait.h`
- `csignal`
- `map`
- `set`
- `sstream`


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

Files in the same folder (`torch/csrc`):

- [`itt_wrapper.cpp_docs.md`](./itt_wrapper.cpp_docs.md)
- [`Generator.h_docs.md`](./Generator.h_docs.md)
- [`Export.h_docs.md`](./Export.h_docs.md)
- [`MemoryFormat.h_docs.md`](./MemoryFormat.h_docs.md)
- [`Size.h_docs.md`](./Size.h_docs.md)
- [`stub.c_docs.md`](./stub.c_docs.md)
- [`Device.h_docs.md`](./Device.h_docs.md)
- [`Layout.h_docs.md`](./Layout.h_docs.md)
- [`Exceptions.h_docs.md`](./Exceptions.h_docs.md)
- [`PyInterpreter.h_docs.md`](./PyInterpreter.h_docs.md)


## Cross-References

- **File Documentation**: `DataLoader.cpp_docs.md`
- **Keyword Index**: `DataLoader.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


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

Files in the same folder (`docs/torch/csrc`):

- [`DeviceAccelerator.cpp_kw.md_docs.md`](./DeviceAccelerator.cpp_kw.md_docs.md)
- [`Exceptions.cpp_docs.md_docs.md`](./Exceptions.cpp_docs.md_docs.md)
- [`utils.cpp_docs.md_docs.md`](./utils.cpp_docs.md_docs.md)
- [`Exceptions.h_docs.md_docs.md`](./Exceptions.h_docs.md_docs.md)
- [`serialization.cpp_kw.md_docs.md`](./serialization.cpp_kw.md_docs.md)
- [`QScheme.cpp_kw.md_docs.md`](./QScheme.cpp_kw.md_docs.md)
- [`DataLoader.cpp_kw.md_docs.md`](./DataLoader.cpp_kw.md_docs.md)
- [`Size.h_docs.md_docs.md`](./Size.h_docs.md_docs.md)
- [`DeviceAccelerator.h_kw.md_docs.md`](./DeviceAccelerator.h_kw.md_docs.md)
- [`Device.cpp_kw.md_docs.md`](./Device.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `DataLoader.cpp_docs.md_docs.md`
- **Keyword Index**: `DataLoader.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
