# Documentation: `docs/torch/nativert/python/Bindings.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/nativert/python/Bindings.cpp_docs.md`
- **Size**: 4,848 bytes (4.73 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nativert/python/Bindings.cpp`

## File Metadata

- **Path**: `torch/nativert/python/Bindings.cpp`
- **Size**: 2,883 bytes (2.82 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <unordered_map>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#include <torch/nativert/ModelRunner.h>

namespace py = pybind11;

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

namespace torch {
namespace nativert {

using torch::nativert::detail::argsToIValue;

void initModelRunnerPybind(py::module& m) {
#if !defined(OVRSOURCE)
  shared_ptr_class_<ModelRunner>(m, "PyModelRunner")
      .def(
          py::init<const std::string&, const std::string&>(),
          py::arg("packagePath"),
          py::arg("modelName"))
      .def(
          "run",
          [](torch::nativert::ModelRunner& self,
             py::args pyargs,
             const py::kwargs& pykwargs) {
            std::vector<c10::IValue> args;
            for (const auto i : c10::irange(pyargs.size())) {
              auto ivalue =
                  torch::jit::toIValue(pyargs[i], c10::AnyType::get());
              args.push_back(std::move(ivalue));
            }
            std::unordered_map<std::string, c10::IValue> kwargs;
            for (const auto& [key, pyarg] : pykwargs) {
              auto ivalue = torch::jit::toIValue(pyarg, c10::AnyType::get());
              kwargs[py::str(key)] = std::move(ivalue);
            }
            c10::IValue ret = self.run(args, kwargs);
            return torch::jit::createPyObjectForStack({ret});
          })
      .def(
          "__call__",
          [](torch::nativert::ModelRunner& self,
             py::args pyargs,
             const py::kwargs& pykwargs) {
            std::vector<c10::IValue> args;
            for (const auto i : c10::irange(pyargs.size())) {
              auto ivalue =
                  torch::jit::toIValue(pyargs[i], c10::AnyType::get());
              args.push_back(std::move(ivalue));
            }
            std::unordered_map<std::string, c10::IValue> kwargs;
            for (const auto& [key, pyarg] : pykwargs) {
              auto ivalue = torch::jit::toIValue(pyarg, c10::AnyType::get());
              kwargs[py::str(key)] = std::move(ivalue);
            }
            c10::IValue ret = self.run(args, kwargs);
            return torch::jit::createPyObjectForStack({ret});
          })
      .def(
          "run_with_flat_inputs_and_outputs",
          [](torch::nativert::ModelRunner& self, py::args pyargs) {
            std::vector<c10::IValue> args;
            for (const auto i : c10::irange(pyargs.size())) {
              auto ivalue =
                  torch::jit::toIValue(pyargs[i], c10::AnyType::get());
              args.push_back(std::move(ivalue));
            }

            auto rets = self.runWithFlatInputsAndOutputs(std::move(args));
            return torch::jit::createPyObjectForStack(std::move(rets));
          });
#endif // !defined(OVRSOURCE)
}

} // namespace nativert
} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 1 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `nativert`, `torch`, `py`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nativert/python`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `unordered_map`
- `torch/csrc/jit/python/pybind_utils.h`
- `torch/csrc/utils/pybind.h`
- `torch/nativert/ModelRunner.h`


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

Files in the same folder (`torch/nativert/python`):

- [`Bindings.h_docs.md`](./Bindings.h_docs.md)


## Cross-References

- **File Documentation**: `Bindings.cpp_docs.md`
- **Keyword Index**: `Bindings.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/nativert/python`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/nativert/python`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/nativert/python`):

- [`Bindings.h_docs.md_docs.md`](./Bindings.h_docs.md_docs.md)
- [`Bindings.h_kw.md_docs.md`](./Bindings.h_kw.md_docs.md)
- [`Bindings.cpp_kw.md_docs.md`](./Bindings.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `Bindings.cpp_docs.md_docs.md`
- **Keyword Index**: `Bindings.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
