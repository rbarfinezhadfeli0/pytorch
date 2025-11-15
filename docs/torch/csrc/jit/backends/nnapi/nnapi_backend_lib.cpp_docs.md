# Documentation: `torch/csrc/jit/backends/nnapi/nnapi_backend_lib.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/backends/nnapi/nnapi_backend_lib.cpp`
- **Size**: 4,543 bytes (4.44 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <memory>

#include <ATen/nnapi/nnapi_bind.h>
#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/backends/backend_exception.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>

namespace torch::jit {

// Implementation of Android NNAPI Backend delegate

// The Android Neural Networks API (NNAPI) is an Android C API designed
// for running computationally intensive operations for machine learning on
// Android devices. The API is available on all Android devices running
// Android 8.1 (API level 27) or higher.

// Implementation is reflective of caffe2/torch/backends/_nnapi/prepare.py
// NnapiModule.forward()
class NnapiBackend : public PyTorchBackendInterface {
 public:
  // Constructor.
  explicit NnapiBackend() = default;
  ~NnapiBackend() override = default;

  bool is_available() override {
    return true;
  }

  c10::impl::GenericDict compile(
      c10::IValue processed,
      c10::impl::GenericDict method_compile_spec) override {
    // Wrap processed in dictionary: {"forward": processed}
    auto dict = processed.toGenericDict();
    c10::Dict<c10::IValue, c10::IValue> handles(
        c10::StringType::get(), c10::AnyType::get());
    handles.insert("forward", dict);
    return c10::impl::toGenericDict(handles);
  }

  c10::impl::GenericList execute(
      c10::IValue handle,
      c10::impl::GenericList inputs) override {
    // Convert inputs to Tensors
    c10::List<at::Tensor> tensorInp;
    for (c10::IValue element : inputs) {
      tensorInp.push_back(element.toTensor());
    }

    // Lazily call init()
    if (comp_ == nullptr) {
      init(handle, tensorInp);
    }
    TORCH_CHECK(comp_ != nullptr)

    c10::List<at::Tensor> outputs;
    for (at::Tensor out : out_templates_) {
      outputs.push_back(at::empty_like(out));
    }

    // Adjust input memory formats
    auto dict = handle.toGenericDict();
    auto inp_mem_fmts = dict.at("inp_mem_fmts").toIntList();
    TORCH_CHECK(tensorInp.size() == inp_mem_fmts.size());
    std::vector<at::Tensor> fixed_inputs;
    for (auto i = 0U; i < tensorInp.size(); i++) {
      int fmt = inp_mem_fmts[i];
      // These constants match the values in DimOrder in serializer.py
      // 0: NCHW, 1: NHWC
      // TODO: See if it's possible to use those directly.
      if (fmt == 0) {
        fixed_inputs.push_back(tensorInp.get(i).contiguous());
      } else if (fmt == 1) {
        fixed_inputs.push_back(
            tensorInp.get(i).permute({0, 2, 3, 1}).contiguous());
      } else {
        TORCH_CHECK(false, "Invalid mem_fmt");
      }
    }

    comp_->run(fixed_inputs, outputs.vec());

    // Adjust output memory formats
    auto out_mem_fmts = dict.at("out_mem_fmts").toIntList();
    TORCH_CHECK(outputs.size() == out_mem_fmts.size());
    for (auto i = 0U; i < outputs.size(); i++) {
      int fmt = out_mem_fmts[i];
      // These constants match the values in DimOrder in serializer.py
      // 0: NCHW, 1: NHWC
      // TODO: See if it's possible to use those directly.
      if (fmt == 1) {
        outputs.set(i, outputs.get(i).permute({0, 3, 1, 2}));
      } else {
        TORCH_CHECK(fmt == 0, "Invalid mem_fmt");
      }
    }

    return c10::impl::toList(outputs);
  }

 private:
  // The following variables are modified by init() during execution,
  // and cannot be passed through the handles dictionary
  std::unique_ptr<torch::nnapi::bind::NnapiCompilation> comp_;
  c10::List<at::Tensor> out_templates_;

  // Runs once per model initialization
  // Cannot be moved to compile(), because init() requires actual inputs
  void init(const c10::IValue& handle, const c10::List<at::Tensor>& inputs) {
    TORCH_CHECK(comp_ == nullptr);
    auto dict = handle.toGenericDict();

    // Get ser_model
    auto ser_model = dict.at("ser_model").toTensor();
    // Load shape computation module
    std::stringstream ss;
    auto shape_ptr = dict.at("shape_compute_module").toString();
    ss.str(*shape_ptr);
    auto shape_compute_module = _load_for_mobile(ss);
    out_templates_ =
        shape_compute_module.run_method("prepare", ser_model, inputs)
            .toTensorList();

    // Create and initialize NnapiCompilation object
    comp_ = std::make_unique<torch::nnapi::bind::NnapiCompilation>();
    auto weights = dict.at("weights").toTensorVector();
    comp_->init(ser_model, weights);
  }
};

namespace {
constexpr auto backend_name = "nnapi";
static auto cls = torch::jit::backend<NnapiBackend>(backend_name);
} // namespace

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `NnapiBackend`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/backends/nnapi`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `memory`
- `ATen/nnapi/nnapi_bind.h`
- `torch/csrc/jit/backends/backend.h`
- `torch/csrc/jit/backends/backend_exception.h`
- `torch/csrc/jit/mobile/import.h`
- `torch/csrc/jit/mobile/module.h`


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

- [`nnapi_backend_preprocess.cpp_docs.md`](./nnapi_backend_preprocess.cpp_docs.md)


## Cross-References

- **File Documentation**: `nnapi_backend_lib.cpp_docs.md`
- **Keyword Index**: `nnapi_backend_lib.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
