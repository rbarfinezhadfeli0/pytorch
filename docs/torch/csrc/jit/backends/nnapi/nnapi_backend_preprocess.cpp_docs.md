# Documentation: `torch/csrc/jit/backends/nnapi/nnapi_backend_preprocess.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/backends/nnapi/nnapi_backend_preprocess.cpp`
- **Size**: 4,525 bytes (4.42 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/backends/backend_preprocess.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

// Converts model to Android NNAPI backend and serializes it for mobile
// Returns a dictionary with preprocessed items:
//    "shape_compute_module": torch::jit::Module,
//    "ser_model": at::Tensor,
//    "weights": List[torch.Tensor],
//    "inp_mem_fmts": List[int],
//    "out_mem_fmts": List[int]
//
// method_compile_spec should contain a Tensor or
// Tensor List which bundles several input parameters:
// shape, dtype, quantization, and dimorder (NHWC/NCHW)
// For input shapes, use 0 for run/load time flexible input
//
// The compile_spec should include the format:
// {"forward": {"inputs": at::Tensor}}
// OR {"forward": {"inputs": c10::List<at::Tensor>}}
// Example input Tensor:
// torch.tensor([[1.0, -1.0, 2.0, -2.0]]).unsqueeze(-1).unsqueeze(-1)
//
// In the future, preprocess will accept a dedicated object
static c10::IValue preprocess(
    const torch::jit::Module& mod,
    const c10::Dict<c10::IValue, c10::IValue>& method_compile_spec,
    const torch::jit::BackendDebugHandleGenerator& generate_debug_handles) {
  // Import the python function for processing modules to Android NNAPI backend
  py::gil_scoped_acquire gil;
  py::object pyModule = py::module_::import("torch.backends._nnapi.prepare");
  py::object pyMethod = pyModule.attr("process_for_nnapi");

  // Wrap the c module in a RecursiveScriptModule
  auto wrapped_mod =
      py::module::import("torch.jit._recursive").attr("wrap_cpp_module")(mod);
  wrapped_mod.attr("eval")();

  // Test that method_compile_spec contains the necessary keys and
  // Tensor/TensorList input
  c10::IValue inp;
  std::string error;
  if (!method_compile_spec.contains("forward")) {
    error = R"(method_compile_spec does not contain the "forward" key.)";
  } else {
    auto innerDict = method_compile_spec.at("forward");
    if (!innerDict.isGenericDict() ||
        !innerDict.toGenericDict().contains("inputs")) {
      error =
          R"(method_compile_spec does not contain a dictionary with an "inputs" key, under it's "forward" key.)";
    } else {
      inp = innerDict.toGenericDict().at("inputs");
      if (!inp.isTensor() && !inp.isTensorList()) {
        error =
            R"(method_compile_spec does not contain either a Tensor or TensorList, under it's "inputs" key.)";
      }
    }
  }
  if (!error.empty()) {
    throw std::runtime_error(
        error +
        "\nmethod_compile_spec should contain a Tensor or Tensor List which bundles input parameters:"
        " shape, dtype, quantization, and dimorder."
        "\nFor input shapes, use 0 for run/load time flexible input."
        "\nmethod_compile_spec must use the following format:"
        "\n{\"forward\": {\"inputs\": at::Tensor}} OR {\"forward\": {\"inputs\": c10::List<at::Tensor>}}");
  }

  // Convert input to a Tensor or a python list of Tensors
  py::list nnapi_processed;
  if (inp.isTensor()) {
    nnapi_processed = pyMethod(wrapped_mod, inp.toTensor());
  } else {
    py::list pyInp;
    for (at::Tensor inpElem : inp.toTensorList()) {
      pyInp.append(inpElem);
    }
    nnapi_processed = pyMethod(wrapped_mod, pyInp);
  }

  // Cast and insert processed items into dict
  c10::Dict<c10::IValue, c10::IValue> dict(
      c10::StringType::get(), c10::AnyType::get());
  dict.insert("ser_model", py::cast<at::Tensor>(nnapi_processed[1]));

  // Serialize shape_compute_module for mobile
  auto shape_compute_module =
      py::cast<torch::jit::Module>(nnapi_processed[0].attr("_c"));
  std::stringstream ss;
  shape_compute_module._save_for_mobile(ss);
  dict.insert("shape_compute_module", ss.str());

  // transform Python lists to C++ c10::List
  c10::List<at::Tensor> weights(
      py::cast<std::vector<at::Tensor>>(nnapi_processed[2]));
  for (auto i = 0U; i < weights.size(); i++) {
    weights.set(i, weights.get(i).contiguous());
  }
  c10::List<int64_t> inp_mem_fmts(
      py::cast<std::vector<int64_t>>(nnapi_processed[3]));
  c10::List<int64_t> out_mem_fmts(
      py::cast<std::vector<int64_t>>(nnapi_processed[4]));
  dict.insert("weights", weights);
  dict.insert("inp_mem_fmts", inp_mem_fmts);
  dict.insert("out_mem_fmts", out_mem_fmts);

  return dict;
}

constexpr auto backend_name = "nnapi";
static auto pre_reg =
    torch::jit::backend_preprocess_register(backend_name, preprocess);

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `py`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/backends/nnapi`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `pybind11/pybind11.h`
- `torch/csrc/jit/backends/backend.h`
- `torch/csrc/jit/backends/backend_preprocess.h`
- `torch/csrc/jit/python/pybind_utils.h`
- `torch/csrc/utils/pybind.h`


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

Files in the same folder (`torch/csrc/jit/backends/nnapi`):

- [`nnapi_backend_lib.cpp_docs.md`](./nnapi_backend_lib.cpp_docs.md)


## Cross-References

- **File Documentation**: `nnapi_backend_preprocess.cpp_docs.md`
- **Keyword Index**: `nnapi_backend_preprocess.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
