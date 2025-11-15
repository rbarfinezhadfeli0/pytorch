# Documentation: `docs/test/inductor/custom_ops.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/inductor/custom_ops.cpp_docs.md`
- **Size**: 19,331 bytes (18.88 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/inductor/custom_ops.cpp`

## File Metadata

- **Path**: `test/inductor/custom_ops.cpp`
- **Size**: 16,696 bytes (16.30 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```cpp
#include <torch/csrc/api/include/torch/types.h>  // @manual=fbcode//caffe2:libtorch

#include <torch/csrc/inductor/aoti_torch/c/shim.h> // @manual
#include <torch/csrc/inductor/aoti_torch/utils.h> // @manual

#include <cstdint>
#include <iostream>
#include <string>

namespace at {

Tensor custom_add_impl(Tensor t1, Tensor t2) {
  return t1 + t2;
}

std::tuple<Tensor, std::optional<Tensor>, std::optional<Tensor>> fn_with_optional_tensor_output_impl(Tensor t1, Tensor t2) {
  Tensor t3 = t1 + t2;
  Tensor t4 = t1 - t2;
  Tensor t5;
  return {t3, t4, t5};
}

std::tuple<Tensor, std::optional<Tensor>, std::optional<Tensor>> fn_with_optional_tensor_output_meta(Tensor t1, Tensor t2) {
  Tensor t3 = t1.clone();
  Tensor t4 = t1.clone();
  Tensor t5;
  return {t3, t4, t5};
}

std::tuple<Tensor, std::optional<Tensor>, std::optional<Tensor>> fn_with_optional_tensor_output_2_impl(Tensor t1, Tensor t2) {
  Tensor t3 = t1 + t2;
  Tensor t4;
  Tensor t5 = t1 - t2;
  return {t3, t4, t5};
}

std::tuple<Tensor, std::optional<Tensor>, std::optional<Tensor>> fn_with_optional_tensor_output_2_meta(Tensor t1, Tensor t2) {
  Tensor t3 = t1.clone();
  Tensor t4;
  Tensor t5 = t1.clone();
  return {t3, t4, t5};
}

std::tuple<Tensor, std::optional<Tensor>, std::optional<Tensor>, std::optional<Tensor>> fn_with_optional_tensor_nullopt_output_impl(Tensor t1, Tensor t2) {
  Tensor t3 = t1 + t2;
  Tensor t4;
  Tensor t5 = t1 - t2;
  return {t3, t4, t5, std::nullopt};
}


std::tuple<Tensor, std::optional<Tensor>, std::optional<Tensor>, std::optional<Tensor>> fn_with_optional_tensor_nullopt_output_meta(Tensor t1, Tensor t2) {
  Tensor t3 = t1.clone();
  Tensor t4;
  Tensor t5 = t1.clone();
  return {t3, t4, t5, std::nullopt};
}

std::tuple<Tensor, std::optional<Tensor>, std::optional<Tensor>, int64_t, int64_t> fn_with_int_output_impl(Tensor t1, Tensor t2, int64_t i1) {
  Tensor t3 = t1 + t2;
  Tensor t4 = t1 - t2;
  Tensor t5;
  int64_t i2 = 0;
  int64_t i3 = 0;
  return {t3, t4, t5, i2, i3};
}

std::tuple<Tensor, std::optional<Tensor>, std::optional<Tensor>, int64_t, int64_t> fn_with_int_output_meta(Tensor t1, Tensor t2, int64_t i1) {
  Tensor t3 = t1.clone();
  Tensor t4 = t1.clone();
  Tensor t5;
  int64_t i2 = 0;
  int64_t i3 = 0;
  return {t3, t4, t5, i2, i3};
}

Tensor fn_with_all_inputs_impl(
    const Tensor& tensor,
    const c10::List<Tensor>& tensors,
    const c10::List<std::optional<Tensor>>& optional_tensors,
    const bool b8,
    const c10::List<bool>& b8s,
    const int64_t i64,
    const c10::List<int64_t>& i64s,
    const int64_t& symint,
    const IntArrayRef symints,
    const double f64,
    const c10::List<double>& f64s,
    const at::Scalar& scalar,
    at::ArrayRef<at::Scalar> scalars,
    const std::string& string,
    const std::vector<std::string>& strings,
    // const c10::ScalarType& dtype,
    // const MemoryFormat& memory_format,
    // const Layout& layout,
    const Device& device,
    // optional
    const std::optional<Tensor>& o_tensor,
    const std::optional<c10::List<Tensor>>& o_tensors,
    const std::optional<bool>& o_b8,
    const std::optional<c10::List<bool>>& o_b8s,
    const std::optional<int64_t>& o_i64,
    const std::optional<c10::List<int64_t>>& o_i64s,
    const std::optional<int64_t>& o_symint,
    const std::optional<IntArrayRef>& o_symints,
    const std::optional<double>& o_f64,
    const std::optional<c10::List<double>>& o_f64s,
    const std::optional<at::Scalar>& o_scalar,
    const std::optional<at::ArrayRef<at::Scalar>>& o_scalars,
    const std::optional<std::string>& o_string,
    const std::optional<std::vector<std::string>>& o_strings,
    // const std::optional<c10::ScalarType>& o_dtype,
    // const std::optional<MemoryFormat>& o_memory_format,
    // const std::optional<Layout>& o_layout,
    const std::optional<Device>& o_device) {
  std::cout << "tensor shape: " << tensor.sizes() << std::endl;

  std::cout << "tensors shape: ";
  for (auto t : tensors) {
    std::cout << t.get().toTensor().sizes() << ", ";
  }
  std::cout << std::endl;

  std::cout << "optional tensors shape: ";
  for (auto t : optional_tensors) {
    if (t.get().toOptional<Tensor>().has_value()) {
      std::cout << t.get().toTensor().sizes() << ", ";
    } else {
      std::cout << "None, ";
    }
  }
  std::cout << std::endl;

  std::cout << "b8 " << c10::IValue(b8) << std::endl;
  std::cout << "b8s " << c10::IValue(b8s) << std::endl;
  std::cout << "i64 " << c10::IValue(i64) << std::endl;
  std::cout << "i64s " << c10::IValue(i64s) << std::endl;
  std::cout << "symint " << c10::IValue(symint) << std::endl;
  std::cout << "symints " << c10::IValue(symints) << std::endl;
  std::cout << "f64 " << c10::IValue(f64) << std::endl;
  std::cout << "f64s " << c10::IValue(f64s) << std::endl;
  std::cout << "scalar " << c10::IValue(scalar) << std::endl;
  std::cout << "scalars " << c10::IValue(scalars) << std::endl;
  std::cout << "string " << c10::IValue(string) << std::endl;
  std::cout << "strings " << c10::IValue(strings) << std::endl;
  // std::cout << "dtype " << c10::IValue(dtype) << std::endl;
  // std::cout << "memory_format " << c10::IValue(memory_format) << std::endl;
  // std::cout << "layout " << c10::IValue(layout) << std::endl;
  std::cout << "device " << c10::IValue(device) << std::endl;

  std::cout << "o_tensor "
            << (o_tensor.has_value() ? c10::IValue(o_tensor.value().sizes())
                                     : "None")
            << std::endl;

  std::cout << "o_tensors shape: ";
  if (o_tensors.has_value()) {
    for (auto t : o_tensors.value()) {
      std::cout << t.get().toTensor().sizes() << ", ";
    }
  } else {
    std::cout << "None";
  }
  std::cout << std::endl;

  std::cout << "o_b8 "
            << (o_b8.has_value() ? c10::IValue(o_b8.value()) : "None")
            << std::endl;
  std::cout << "o_b8s "
            << (o_b8s.has_value() ? c10::IValue(o_b8s.value()) : "None")
            << std::endl;
  std::cout << "o_i64 "
            << (o_i64.has_value() ? c10::IValue(o_i64.value()) : "None")
            << std::endl;
  std::cout << "o_i64s "
            << (o_i64s.has_value() ? c10::IValue(o_i64s.value()) : "None")
            << std::endl;
  std::cout << "o_symint "
            << (o_symint.has_value() ? c10::IValue(o_symint.value()) : "None")
            << std::endl;
  std::cout << "o_symints "
            << (o_symints.has_value() ? c10::IValue(o_symints.value()) : "None")
            << std::endl;
  std::cout << "o_f64 "
            << (o_f64.has_value() ? c10::IValue(o_f64.value()) : "None")
            << std::endl;
  std::cout << "o_f64s "
            << (o_f64s.has_value() ? c10::IValue(o_f64s.value()) : "None")
            << std::endl;
  std::cout << "o_scalar "
            << (o_scalar.has_value() ? c10::IValue(o_scalar.value()) : "None")
            << std::endl;
  std::cout << "o_scalars "
            << (o_scalars.has_value() ? c10::IValue(o_scalars.value()) : "None")
            << std::endl;
  std::cout << "o_string "
            << (o_string.has_value() ? c10::IValue(o_string.value()) : "None")
            << std::endl;
  std::cout << "o_strings "
            << (o_strings.has_value() ? c10::IValue(o_strings.value()) : "None")
            << std::endl;
  // std::cout << "o_dtype "
  //           << (o_dtype.has_value() ? c10::IValue(o_dtype.value()) : "None")
  //           << std::endl;
  // std::cout << "o_memory_format "
  //           << (o_memory_format.has_value()
  //                   ? c10::IValue(o_memory_format.value())
  //                   : "None")
  //           << std::endl;
  // std::cout << "o_layout "
  //           << (o_layout.has_value() ? c10::IValue(o_layout.value()) : "None")
  //           << std::endl;
  std::cout << "o_device "
            << (o_device.has_value() ? c10::IValue(o_device.value()) : "None")
            << std::endl;

  int64_t int_hash = 0;
  int_hash ^= i64;
  for (auto i : i64s) {
    int_hash ^= i;
  }
  if (o_i64.has_value()) {
    int_hash ^= o_i64.value();
  }
  if (o_i64s.has_value()) {
    for (auto i : o_i64s.value()) {
      int_hash ^= i;
    }
  }

  int_hash ^= symint;
  for (auto i : symints) {
    int_hash ^= i;
  }
  if (o_symint.has_value()) {
    int_hash ^= o_symint.value();
  }
  if (o_symints.has_value()) {
    for (auto i : o_symints.value()) {
      int_hash ^= i;
    }
  }

  return tensor + int_hash;
}

Tensor fn_with_default_input_impl(const Tensor& tensor, const int64_t i64) {
  return tensor + i64;
}

std::tuple<Tensor, Tensor> fn_with_tuple_output_impl(
    const Tensor& tensor,
    const int64_t i64) {
  return {tensor + i64, tensor - i64};
}

std::vector<Tensor> fn_with_list_output_impl(
    TensorList tensors,
    const int64_t i64) {
  std::vector<Tensor> outputs;
  for (auto& t : tensors) {
    outputs.emplace_back(t + i64);
  }
  return outputs;
}

std::tuple<Tensor, std::vector<Tensor>> fn_with_mix_outputs_impl(
    const Tensor& tensor,
    TensorList tensors) {
  std::vector<Tensor> outputs;
  for (auto& t : tensors) {
    outputs.emplace_back(t + 2);
  }
  return {tensor + 1, outputs};
}

std::tuple<Tensor, Tensor> fn_with_input_mutation_impl(
    Tensor& t0,
    const Tensor& t1,
    Tensor& t2) {
  t0.add_(1);
  t2.sub_(1);
  return {t1 + 1, t1 + 2};
}

void fn_out_variant_without_return_impl(
    const Tensor& x,
    Tensor& out) {
  out.add_(x);
}

// NOLINTBEGIN(clang-diagnostic-unused-parameter)
Tensor fn_with_all_inputs_meta(
    const Tensor& tensor,
    const c10::List<Tensor>& tensors,
    const c10::List<std::optional<Tensor>>& optional_tensors,
    const bool b8,
    const c10::List<bool>& b8s,
    const int64_t i64,
    const c10::List<int64_t>& i64s,
    const c10::SymInt& symint,
    c10::SymIntArrayRef symints,
    const double f64,
    const c10::List<double>& f64s,
    const at::Scalar& scalar,
    at::ArrayRef<at::Scalar> scalars,
    const std::string& string,
    const std::vector<std::string>& strings,
    // const c10::ScalarType& dtype,
    // const MemoryFormat& memory_format,
    // const Layout& layout,
    const Device& device,
    // optional
    const std::optional<Tensor>& o_tensor,
    const std::optional<c10::List<Tensor>>& o_tensors,
    const std::optional<bool>& o_b8,
    const std::optional<c10::List<bool>>& o_b8s,
    const std::optional<int64_t>& o_i64,
    const std::optional<c10::List<int64_t>>& o_i64s,
    const std::optional<c10::SymInt>& o_symint,
    at::OptionalSymIntArrayRef o_symints,
    const std::optional<double>& o_f64,
    const std::optional<c10::List<double>>& o_f64s,
    const std::optional<at::Scalar>& o_scalar,
    const std::optional<at::ArrayRef<at::Scalar>>& o_scalars,
    const std::optional<std::string>& o_string,
    const std::optional<std::vector<std::string>>& o_strings,
    // const std::optional<c10::ScalarType>& o_dtype,
    // const std::optional<MemoryFormat>& o_memory_format,
    // const std::optional<Layout>& o_layout,
    const std::optional<Device>& o_device) {
  return tensor;
}

Tensor fn_with_default_input_meta(const Tensor& tensor, const int64_t i64) {
  return tensor.clone();
}

std::tuple<Tensor, Tensor> fn_with_tuple_output_meta(
    const Tensor& tensor,
    const int64_t i64) {
  return {tensor.clone(), tensor.clone()};
}

std::vector<Tensor> fn_with_list_output_meta(
    TensorList tensors,
    const int64_t i64) {
  std::vector<Tensor> outputs;
  for (auto& t : tensors) {
    outputs.push_back(t.clone());
  }
  return outputs;
}

std::tuple<Tensor, std::vector<Tensor>> fn_with_mix_outputs_meta(
    const Tensor& tensor,
    TensorList tensors) {
  std::vector<Tensor> outputs;
  for (auto& t : tensors) {
    outputs.push_back(t.clone());
  }
  return {tensor.clone(), outputs};
}

std::tuple<Tensor, Tensor> fn_with_input_mutation_meta(
    Tensor& t0,
    const Tensor& t1,
    Tensor& t2) {
  return {t1.clone(), t1.clone()};
}

void fn_out_variant_without_return_meta(
    const Tensor& x,
    Tensor& out) {
}

Tensor fn_square_impl(const Tensor& tensor) {
  return tensor * tensor;
}

Tensor fn_square_meta(const Tensor& tensor) {
  return at::empty_like(tensor);
}
} // namespace at


extern "C" {
  AOTI_TORCH_EXPORT AOTITorchError
  aoti_torch_cpu_fn_square(
      AtenTensorHandle input,
      AtenTensorHandle* ret) {
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto tmp_result = at::fn_square_impl(
          torch::aot_inductor::resolve_tensor_dispatch_flags(input));
      *ret = torch::aot_inductor::new_tensor_handle(std::move(tmp_result));
    });
  }

  AOTI_TORCH_EXPORT AOTITorchError
  aoti_torch_cuda_fn_square(
      AtenTensorHandle input,
      AtenTensorHandle* ret) {
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto tmp_result = at::fn_square_impl(
          torch::aot_inductor::resolve_tensor_dispatch_flags(input));
      *ret = torch::aot_inductor::new_tensor_handle(std::move(tmp_result));
    });
  }
}

TORCH_LIBRARY(aoti_custom_ops, m) {
  m.def("custom_add(Tensor t1, Tensor t2) -> Tensor");
  m.def("fn_with_optional_tensor_output(Tensor t1, Tensor t2) -> (Tensor, Tensor?, Tensor?)");
  m.def("fn_with_optional_tensor_output_2(Tensor t1, Tensor t2) -> (Tensor, Tensor?, Tensor?)");
  m.def("fn_with_optional_tensor_nullopt_output(Tensor t1, Tensor t2) -> (Tensor, Tensor?, Tensor?, Tensor?)");
  m.def("fn_with_int_output(Tensor t1, Tensor t2, int i) -> (Tensor, Tensor?, Tensor?, int, int)");
  m.def(
      "fn_with_all_inputs(Tensor tensor, "
      "Tensor[] tensors, "
      "Tensor?[] optional_tensors, "
      "bool b8, bool[] b8s, "
      "int i64, int[] i64s, "
      "SymInt symint, SymInt[] symints, "
      "float f64, float[] f64s, "
      "Scalar scalar, Scalar[] scalars, "
      "str string, str[] strings, "
      // "ScalarType dtype, "
      // "MemoryFormat memory_format, "
      // "Layout layout, "
      "Device device, "
      "*, "
      "Tensor? o_tensor, Tensor[]? o_tensors, "
      "bool? o_b8, bool[]? o_b8s, "
      "int? o_i64, int[]? o_i64s, "
      "SymInt? o_symint, SymInt[]? o_symints, "
      "float? o_f64, float[]? o_f64s, "
      "Scalar? o_scalar, Scalar[]? o_scalars, "
      "str? o_string, str[]? o_strings, "
      // "ScalarType? o_dtype, "
      // "MemoryFormat? o_memory_format, "
      // "Layout? o_layout, "
      "Device? o_device) -> Tensor");

  m.def("fn_with_default_input(Tensor t, int i=3) -> Tensor");

  m.def("fn_with_tuple_output(Tensor t, int i) -> (Tensor, Tensor)");

  m.def("fn_with_list_output(Tensor[] tensors, int i) -> Tensor[]");

  m.def(
      "fn_with_mix_outputs(Tensor t, Tensor[] tensors) -> (Tensor, Tensor[])");

  m.def(
      "fn_with_input_mutation(Tensor(a!) t0, Tensor t1, Tensor(b!) t2) -> (Tensor, Tensor)");

  m.def("fn_out_variant_without_return(Tensor x, Tensor(a!) out) -> ()");
  m.def("fn_square(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(aoti_custom_ops, CompositeExplicitAutograd, m) {
  m.impl("custom_add", at::custom_add_impl);
  m.impl("fn_with_optional_tensor_output", at::fn_with_optional_tensor_output_impl);
  m.impl("fn_with_optional_tensor_output_2", at::fn_with_optional_tensor_output_2_impl);
  m.impl("fn_with_optional_tensor_nullopt_output", at::fn_with_optional_tensor_nullopt_output_impl);
  m.impl("fn_with_int_output", at::fn_with_int_output_impl);
  m.impl("fn_with_all_inputs", at::fn_with_all_inputs_impl);
  m.impl("fn_with_default_input", at::fn_with_default_input_impl);
  m.impl("fn_with_tuple_output", at::fn_with_tuple_output_impl);
  m.impl("fn_with_list_output", at::fn_with_list_output_impl);
  m.impl("fn_with_mix_outputs", at::fn_with_mix_outputs_impl);
  m.impl("fn_with_input_mutation", at::fn_with_input_mutation_impl);
  m.impl("fn_out_variant_without_return", at::fn_out_variant_without_return_impl);
  m.impl("fn_square", at::fn_square_impl);
}

TORCH_LIBRARY_IMPL(aoti_custom_ops, Meta, m) {
  m.impl("fn_with_optional_tensor_output", at::fn_with_optional_tensor_output_meta);
  m.impl("fn_with_optional_tensor_output_2", at::fn_with_optional_tensor_output_2_meta);
  m.impl("fn_with_optional_tensor_nullopt_output", at::fn_with_optional_tensor_nullopt_output_meta);
  m.impl("fn_with_int_output", at::fn_with_int_output_meta);
  m.impl("fn_with_all_inputs", at::fn_with_all_inputs_meta);
  m.impl("fn_with_default_input", at::fn_with_default_input_meta);
  m.impl("fn_with_tuple_output", at::fn_with_tuple_output_meta);
  m.impl("fn_with_list_output", at::fn_with_list_output_meta);
  m.impl("fn_with_mix_outputs", at::fn_with_mix_outputs_meta);
  m.impl("fn_with_input_mutation", at::fn_with_input_mutation_meta);
  m.impl("fn_out_variant_without_return", at::fn_out_variant_without_return_meta);
  m.impl("fn_square", at::fn_square_meta);
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/api/include/torch/types.h`
- `torch/csrc/inductor/aoti_torch/c/shim.h`
- `torch/csrc/inductor/aoti_torch/utils.h`
- `cstdint`
- `iostream`
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
python test/inductor/custom_ops.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/inductor`):

- [`test_benchmark_fusion.py_docs.md`](./test_benchmark_fusion.py_docs.md)
- [`test_op_dtype_prop.py_docs.md`](./test_op_dtype_prop.py_docs.md)
- [`test_custom_op_autotune.py_docs.md`](./test_custom_op_autotune.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_inductor_freezing.py_docs.md`](./test_inductor_freezing.py_docs.md)
- [`test_b2b_gemm.py_docs.md`](./test_b2b_gemm.py_docs.md)
- [`test_minifier_isolate.py_docs.md`](./test_minifier_isolate.py_docs.md)
- [`test_move_constructors_to_cuda.py_docs.md`](./test_move_constructors_to_cuda.py_docs.md)
- [`test_cutlass_backend.py_docs.md`](./test_cutlass_backend.py_docs.md)
- [`test_cache.py_docs.md`](./test_cache.py_docs.md)


## Cross-References

- **File Documentation**: `custom_ops.cpp_docs.md`
- **Keyword Index**: `custom_ops.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/inductor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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
python docs/test/inductor/custom_ops.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/inductor`):

- [`test_snode_runtime.py_kw.md_docs.md`](./test_snode_runtime.py_kw.md_docs.md)
- [`test_metrics.py_docs.md_docs.md`](./test_metrics.py_docs.md_docs.md)
- [`test_flex_attention.py_kw.md_docs.md`](./test_flex_attention.py_kw.md_docs.md)
- [`test_cuda_repro.py_kw.md_docs.md`](./test_cuda_repro.py_kw.md_docs.md)
- [`test_fxir_backend.py_kw.md_docs.md`](./test_fxir_backend.py_kw.md_docs.md)
- [`test_split_cat_fx_passes.py_kw.md_docs.md`](./test_split_cat_fx_passes.py_kw.md_docs.md)
- [`test_mmdecomp.py_kw.md_docs.md`](./test_mmdecomp.py_kw.md_docs.md)
- [`test_torchinductor_codegen_config_overrides.py_kw.md_docs.md`](./test_torchinductor_codegen_config_overrides.py_kw.md_docs.md)
- [`test_aot_inductor_custom_ops.py_kw.md_docs.md`](./test_aot_inductor_custom_ops.py_kw.md_docs.md)
- [`test_minifier.py_kw.md_docs.md`](./test_minifier.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `custom_ops.cpp_docs.md_docs.md`
- **Keyword Index**: `custom_ops.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
